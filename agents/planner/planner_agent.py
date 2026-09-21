"""Planner agent - reasons over structured state, emits a SkillIntent.

Uses LangChain structured output when an LLM key is available; otherwise falls
back to a deterministic rule-based planner so the framework runs offline.
"""

from __future__ import annotations

from typing import Any

from core.config import AppConfig
from core.constants import LLMProvider, ScreenState
from core.logger import get_logger
from core.models import SkillDefinition, SkillIntent
from core.state import StructuredGameState

logger = get_logger("planner")

_SYSTEM_PROMPT = """You are the planner for a closed-loop PC game automation agent.
You decide the SINGLE next skill to execute from a fixed list of available skills.

You are given STRUCTURED VISUAL EVIDENCE extracted from the real game screen:
- `visual_features` are exact OpenCV pixel measurements (edges, motion, brightness,
  UI-rectangle count, colours). Treat them as GROUND TRUTH - the screen is the
  absolute source of truth.
- `ocr_texts` are the words literally visible on screen right now.
- `scene` (when present) is a vision-model reading of the frame.

Rules:
- You MUST choose a skill name that appears EXACTLY in the provided available skills.
- You never produce keyboard/mouse commands directly; only a skill intent.
- Reason ONLY from the provided evidence. NEVER invent UI, text, or objects that
  are not in the evidence. If evidence is insufficient, choose an observation skill.
- High `motion_score` means the game is animating (likely active gameplay);
  near-zero motion with several `ui_element_count` means a static menu.
- Prefer deterministic progress toward the user goal.
- Provide a short reason grounded in the evidence and a confidence in [0,1].
"""


class PlannerAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm: Any | None = None
        self._structured_llm: Any | None = None
        self._init_llm()

    @property
    def llm_available(self) -> bool:
        return self._structured_llm is not None

    def _init_llm(self) -> None:
        llm_cfg = self.config.llm
        try:
            if llm_cfg.provider is LLMProvider.OPENAI:
                if not self.config.openai_api_key:
                    logger.info("planner_llm_disabled", reason="no OpenAI API key")
                    return
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(model=llm_cfg.planner_model,
                                       temperature=llm_cfg.temperature,
                                       api_key=self.config.openai_api_key,
                                       timeout=llm_cfg.request_timeout_s)
            else:
                if not self.config.anthropic_api_key:
                    logger.info("planner_llm_disabled", reason="no Anthropic API key")
                    return
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(model=llm_cfg.planner_model,
                                          temperature=llm_cfg.temperature,
                                          api_key=self.config.anthropic_api_key,
                                          timeout=llm_cfg.request_timeout_s)
            self._structured_llm = self._llm.with_structured_output(SkillIntent)
            logger.info("planner_llm_ready", provider=str(llm_cfg.provider))
        except Exception as exc:  # noqa: BLE001
            logger.warning("planner_llm_init_failed", error=str(exc))
            self._llm = None
            self._structured_llm = None

    def plan(self, *, state: StructuredGameState, user_goal: str,
             available_skills: list[SkillDefinition]) -> SkillIntent:
        if not available_skills:
            return SkillIntent(skill="observe", reason="No skills available; observing.",
                               confidence=0.3, goal=user_goal)
        if self.llm_available:
            try:
                intent = self._plan_with_llm(state, user_goal, available_skills)
                if intent is not None:
                    return intent
            except Exception as exc:  # noqa: BLE001
                logger.warning("planner_llm_failed", error=str(exc))
        return self._plan_deterministic(state, user_goal, available_skills)

    def _plan_with_llm(self, state: StructuredGameState, user_goal: str,
                       available_skills: list[SkillDefinition]) -> SkillIntent | None:
        names = [s.name for s in available_skills]
        skill_lines = "\n".join(f"- {s.name}: {s.description}" for s in available_skills)
        gs = state.game_state or {}
        # Minecraft pixel-HUD facts (crosshair/health/block) are ground truth.
        mc_hud = {k: gs.get(k) for k in (
            "crosshair_visible", "health", "hunger", "selected_hotbar_slot",
            "block_under_crosshair", "outdoors", "mob_count", "day_time",
            "critical_health", "under_threat", "phase", "is_paused",
            "inventory_open", "gameplay_objective",
        ) if k in gs}
        evidence = {
            "game": state.game,
            "screen_state": str(state.screen),
            "screen_confidence": round(state.overall_confidence, 3),
            "objective": state.objective.model_dump(),
            "minecraft_hud": mc_hud,                       # pixel-measured truth
            "visual_features": gs.get("_visual", {}),      # OpenCV ground truth
            "ocr_texts": gs.get("_ocr_texts", [])[:30],    # literal on-screen text
            "signal_agreement": gs.get("_signal_agreement", False),
            "scene": gs.get("_scene"),                     # VLM reading (if any)
            "recent_history": [h.model_dump(mode="json") for h in state.history[-3:]],
        }
        user_prompt = (
            f"User goal: {user_goal}\n\n"
            f"STRUCTURED VISUAL EVIDENCE (from the real screen):\n{evidence}\n\n"
            f"Available skills (choose exactly one name):\n{skill_lines}\n\n"
            f"Return a SkillIntent selecting one of: {names}"
        )
        intent: SkillIntent = self._structured_llm.invoke(
            [("system", _SYSTEM_PROMPT), ("human", user_prompt)]
        )
        if intent.skill not in names:
            logger.warning("planner_invalid_skill", chosen=intent.skill, allowed=names)
            return None
        if not intent.goal:
            intent.goal = user_goal
        return intent

    def _plan_deterministic(self, state: StructuredGameState, user_goal: str,
                            available_skills: list[SkillDefinition]) -> SkillIntent:
        """Game-agnostic fallback driven by fused visual evidence.

        No game-specific hard-coding: decisions come from the OpenCV visual
        features + screen state, so this works for any game whose skills expose
        the relevant preconditions.
        """
        names = {s.name for s in available_skills}
        vf = (state.game_state or {}).get("_visual", {})
        has_visual = bool(vf)  # real pixel measurements are present
        motion = float(vf.get("motion_score", 0.0))
        ui_count = int(vf.get("ui_element_count", 0))
        edge_density = float(vf.get("edge_density", 0.0))
        brightness = float(vf.get("brightness_mean", 0.0))
        screen = state.screen
        gs = state.game_state or {}
        # The crosshair is deterministic pixel-truth: when it is visible we are
        # unambiguously in-world, so we must NOT fall back to `observe` on a low
        # fused confidence. In-world frames have no OCR text and OpenCV can misread
        # the HUD as a menu, dragging fused confidence below the 0.45 threshold -
        # which previously trapped the agent in an endless observe loop.
        mc_crosshair = state.game == "minecraft" and bool(gs.get("crosshair_visible"))

        # 1) Unknown / low confidence -> observe (gather more evidence).
        if not mc_crosshair and (
                screen == ScreenState.UNKNOWN or state.overall_confidence < 0.45):
            if "observe" in names:
                return SkillIntent(skill="observe",
                                   reason="Screen unknown/low confidence; gathering evidence.",
                                   confidence=0.6, goal=user_goal)

        # 2) Loading screen -> wait. Only infer loading from raw pixels when we
        #    actually HAVE pixel measurements (avoid treating empty state as dark).
        if screen == ScreenState.GAME_LOADING or (
                has_visual and brightness < 0.06 and edge_density < 0.03):
            if "wait" in names:
                return SkillIntent(skill="wait", parameters={"duration_ms": 2000},
                                   reason="Loading (dark, no structure); waiting.",
                                   confidence=0.7, goal=user_goal)

        # 3) Xbox app pre-game screen -> launch the game if such a skill exists.
        if screen == ScreenState.XBOX_APP:
            launch = next((n for n in names if n.startswith("xbox_launch")), None)
            if launch:
                return SkillIntent(skill=launch, reason="On Xbox app; launching game.",
                                   confidence=0.8, goal=user_goal)

        # 3b) Minecraft gameplay: decide from the pixel HUD (crosshair truth).
        #     `mc_crosshair`/`gs` were resolved above so this branch is reached
        #     even when fused confidence is low (the crosshair is ground truth).
        if mc_crosshair:
            mc = self._plan_minecraft_gameplay(gs, names, user_goal)
            if mc is not None:
                return mc
        # If paused/inventory in Minecraft, resume first.
        if state.game == "minecraft" and gs.get("is_paused") and "mc_unpause" in names:
            return SkillIntent(skill="mc_unpause", reason="Paused; resuming gameplay.",
                               confidence=0.8, goal=user_goal)

        # 4) Active gameplay -> objective (nav/task) or a movement skill.
        #    Checked before the menu branch so a navigation objective in gameplay
        #    is honoured even when visual features are absent (e.g. tests).
        if screen == ScreenState.GAMEPLAY or motion >= 0.05:
            obj = state.objective
            if obj.type == "complete_task" and "complete_task" in names:
                return SkillIntent(skill="complete_task", target=obj.target,
                                   reason="In gameplay; completing task.",
                                   confidence=0.7, goal=user_goal)
            if obj.type == "navigate" and obj.target and "navigate_to_room" in names:
                return SkillIntent(skill="navigate_to_room", target=obj.target,
                                   parameters={"target": obj.target},
                                   reason=f"In gameplay; navigating to {obj.target}.",
                                   confidence=0.7, goal=user_goal)
            move = next((n for n in names if n in ("mc_move", "move_forward")), None)
            if move:
                return SkillIntent(skill=move, parameters={"direction": "forward"},
                                   reason="In gameplay; exploring forward.",
                                   confidence=0.55, goal=user_goal)

        # 5) Static menu (low motion, several UI rectangles) -> menu skill.
        if screen in (ScreenState.MENU, ScreenState.MAIN_MENU) or (
                has_visual and motion < 0.03 and ui_count >= 3):
            start = next((n for n in names
                          if any(k in n for k in ("start", "singleplayer", "continue",
                                                  "click_button"))), None)
            if start:
                return SkillIntent(skill=start,
                                   reason=f"Static menu ({ui_count} UI elements); advancing.",
                                   confidence=0.65, goal=user_goal)

        # 6) Default: observe, else first available skill.
        if "observe" in names:
            return SkillIntent(skill="observe", reason="Default observation.",
                               confidence=0.5, goal=user_goal)
        return SkillIntent(skill=sorted(names)[0],
                           reason="Fallback to first available skill.",
                           confidence=0.4, goal=user_goal)

    @staticmethod
    def _plan_minecraft_gameplay(gs: dict, names: set[str],
                                 user_goal: str) -> SkillIntent | None:
        """Screen-truth Minecraft gameplay decision, in strict priority order.

        Everything here keys off pixel-measured HUD facts (crosshair confirmed
        in-world). Priorities: survive -> fight/flee -> pursue wood -> mine ->
        explore. Returns None to let the generic branches handle edge cases.
        """
        # Use `or default` so None HUD values (fresh state / failed read) never
        # crash int()/str(); a loading frame can leave these keys set to None.
        health = int(gs.get("health") or 20)
        under_threat = bool(gs.get("under_threat"))
        block = str(gs.get("block_under_crosshair") or "unknown")
        day_time = str(gs.get("day_time") or "day")

        # P1: survival - critical health while threatened => flee.
        if under_threat and (health <= 8) and "mc_flee_threat" in names:
            return SkillIntent(skill="mc_flee_threat",
                               reason=f"Low health ({health}) under threat; fleeing.",
                               confidence=0.9, goal=user_goal)
        # P2: threat present => fight what we're aimed at.
        if under_threat and "mc_fight" in names:
            return SkillIntent(skill="mc_fight",
                               reason="Hostile mob detected; attacking.",
                               confidence=0.8, goal=user_goal)
        # P3: wood is directly under the crosshair => chop it.
        if block == "wood" and "mc_chop_wood" in names:
            return SkillIntent(skill="mc_chop_wood",
                               reason="Wood under crosshair; chopping.",
                               confidence=0.85, goal=user_goal)
        # P4: looking at stone => mine it.
        if block == "stone" and "mc_mine_ground" in names:
            return SkillIntent(skill="mc_mine_ground",
                               reason="Stone under crosshair; mining.",
                               confidence=0.75, goal=user_goal)
        # P5: night and outdoors => mine down for a quick shelter.
        if day_time == "night" and "mc_mine_ground" in names:
            return SkillIntent(skill="mc_mine_ground",
                               reason="Night; digging down for cover.",
                               confidence=0.6, goal=user_goal)
        # P6: nothing useful ahead => look around to find a target.
        if block in ("unknown", "grass", "dirt", "leaves") and "mc_look_around" in names:
            return SkillIntent(skill="mc_look_around", parameters={"dx": 250, "dy": 0},
                               reason=f"Block ahead is '{block}'; scanning for wood/stone.",
                               confidence=0.55, goal=user_goal)
        # P7: approach whatever is ahead.
        if "mc_approach" in names:
            return SkillIntent(skill="mc_approach", parameters={"duration_ms": 500},
                               reason="Moving forward to find resources.",
                               confidence=0.5, goal=user_goal)
        return None
