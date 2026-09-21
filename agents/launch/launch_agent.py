"""Skill-driven, screen-truth launch agent.

This replaces the old hard-coded, fixed-timing launch script. The launch flow
is described DECLARATIVELY in ``skills/games/minecraft/xbox_launch.yaml`` under
``launch_states`` / ``launch_config``. This agent reads that skill file and, on
every cycle:

    1. Captures the LIVE screen (the single source of truth).
    2. Perceives it with OpenCV (crosshair pixel-truth via MinecraftVision) and
       OCR (the literal on-screen text), scoped to whichever window (Xbox app or
       Minecraft) is currently in front.
    3. Matches the observation against the declarative state ``detect`` blocks
       and performs the FIRST matching state's ``action`` - locating click
       targets by OCR at run time (never hard-coded pixel positions).
    4. Re-observes. Slow app/menu rendering just means the same state keeps
       matching until the screen actually changes - there are NO blind sleeps
       between steps.

When no declarative state matches for ``confuse_limit`` consecutive cycles, the
agent escalates to the VLM/LLM to classify the frame and choose the closest
action. The screen is authoritative; the LLM only arbitrates genuine ambiguity.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.config import AppConfig, get_config
from core.constants import LLMProvider
from core.exceptions import WindowNotFoundError
from core.logger import get_logger
from tools.desktop.screen_finder import ScreenFinder
from tools.desktop.winapp import XboxDesktopAutomation
from tools.vision.minecraft_vision import MinecraftVision

logger = get_logger("launch_agent")


class Observation:
    """One cycle's screen truth: window titles + OCR text + crosshair pixel."""

    def __init__(self, *, window_titles: list[str], ocr_text: str,
                 crosshair: bool, frame: np.ndarray | None) -> None:
        self.window_titles = window_titles
        self.ocr_text = ocr_text
        self.crosshair = crosshair
        self.frame = frame

    def has_window(self, substring: str) -> bool:
        s = substring.lower()
        return any(s in (t or "").lower() for t in self.window_titles)


class LaunchAgent:
    """Reads the launch skill file and drives Minecraft to the in-world state."""

    def __init__(self, *, config: AppConfig | None = None,
                 xbox: XboxDesktopAutomation | None = None,
                 finder: ScreenFinder | None = None,
                 vision: MinecraftVision | None = None) -> None:
        self.config = config or get_config()
        self.xbox = xbox or XboxDesktopAutomation()
        self.finder = finder or ScreenFinder()
        self.vision = vision or MinecraftVision()
        self._vlm: Any | None = None
        self._states, self._cfg = self._load_skill()

    # ------------------------------------------------------------------ #
    # Skill loading
    # ------------------------------------------------------------------ #
    def _load_skill(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = (Path(self.config.skills_dir) / "games" / "minecraft"
                / "xbox_launch.yaml")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            logger.error("launch_skill_load_failed", error=str(exc))
            return [], {}
        states = data.get("launch_states", []) or []
        cfg = data.get("launch_config", {}) or {}
        logger.info("launch_skill_loaded", states=len(states),
                    names=[s.get("name") for s in states])
        return states, cfg


    # ------------------------------------------------------------------ #
    # Perception - the live screen is the source of truth
    # ------------------------------------------------------------------ #
    @staticmethod
    def _list_window_titles() -> list[str]:
        titles: list[str] = []
        try:
            import win32gui

            def _cb(hwnd, _acc):  # noqa: ANN001 - win32 callback signature
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if text:
                        titles.append(text)

            win32gui.EnumWindows(_cb, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("enum_windows_failed", error=str(exc))
        return titles

    def _point_finder_at(self, obs_titles: list[str]) -> None:
        """Scope OCR capture to the frontmost relevant window (game > xbox)."""
        game = str(self._cfg.get("window_game", "Minecraft"))
        xbox = str(self._cfg.get("window_xbox", "XBOX"))
        target = None
        if any(game.lower() in (t or "").lower() for t in obs_titles):
            target = game
        elif any(xbox.lower() in (t or "").lower() for t in obs_titles):
            target = xbox
        if target is None:
            self.finder.region = None
            return
        try:
            from tools.capture.screen_capture import ScreenCapture
            rect = ScreenCapture.find_window_rect(f".*{target}.*")
            self.finder.region = rect  # None -> full desktop fallback
        except Exception as exc:  # noqa: BLE001
            logger.debug("region_scope_failed", error=str(exc))
            self.finder.region = None

    def observe(self) -> Observation:
        titles = self._list_window_titles()
        self._point_finder_at(titles)
        frame: np.ndarray | None = None
        ocr_text = ""
        crosshair = False
        try:
            frame, _, _ = self.finder.screenshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("observe_screenshot_failed", error=str(exc))
        if frame is not None:
            try:
                results = self.finder._ocr.read_text(frame, use_cache=False)  # noqa: SLF001
                ocr_text = " ".join(r.text.lower() for r in results)
            except Exception as exc:  # noqa: BLE001
                logger.debug("observe_ocr_failed", error=str(exc))
            try:
                crosshair = bool(self.vision.analyse(frame).crosshair_visible)
            except Exception as exc:  # noqa: BLE001
                logger.debug("observe_vision_failed", error=str(exc))
        return Observation(window_titles=titles, ocr_text=ocr_text,
                           crosshair=crosshair, frame=frame)

    # ------------------------------------------------------------------ #
    # Declarative condition matching
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        return [str(v) for v in (value if isinstance(value, list) else [value])]

    def _conditions_hold(self, cond: dict[str, Any], obs: Observation) -> bool:
        if not cond:
            return True
        # window_present / window_absent each accept a single value OR a list.
        if "window_present" in cond and not all(
                obs.has_window(w) for w in self._as_list(cond["window_present"])):
            return False
        if "window_absent" in cond and any(
                obs.has_window(w) for w in self._as_list(cond["window_absent"])):
            return False
        if "crosshair" in cond and bool(cond["crosshair"]) != obs.crosshair:
            return False
        any_text = cond.get("any_text")
        if any_text and not any(str(t).lower() in obs.ocr_text for t in any_text):
            return False
        all_text = cond.get("all_text")
        if all_text and not all(str(t).lower() in obs.ocr_text for t in all_text):
            return False
        not_text = cond.get("not_text")
        if not_text and any(str(t).lower() in obs.ocr_text for t in not_text):
            return False
        return True

    def _match_state(self, obs: Observation) -> dict[str, Any] | None:
        for state in self._states:
            if self._conditions_hold(state.get("detect", {}), obs):
                return state
        return None

    # ------------------------------------------------------------------ #
    # Actions - all click targets located by OCR at run time
    # ------------------------------------------------------------------ #
    def _do_action(self, state: dict[str, Any], obs: Observation) -> None:
        action = state.get("action", {}) or {}
        op = action.get("op", "wait")
        name = state.get("name", "?")
        logger.info("launch_action", state=name, op=op)

        if op == "launch_xbox":
            self.xbox.launch_xbox_app()
            try:
                self.xbox.wait_for_xbox_ready(timeout_s=40.0)
                self.xbox.maximize()
            except WindowNotFoundError as exc:
                logger.warning("xbox_not_ready_yet", error=str(exc))
            return

        if op == "search":
            query = str(action.get("query")
                        or self._cfg.get("search_query", "minecraft for windows"))
            self._focus_search()
            self.finder.type_text(query, clear_first=True)
            self.finder.press_enter()
            logger.info("launch_search_submitted", query=query)
            return

        if op == "click_text":
            targets = action.get("targets", []) or []
            exclude = tuple(action.get("exclude", []) or ())
            for t in targets:
                hit = self.finder.find_text(str(t), exclude=exclude)
                if hit is not None:
                    self.finder.click_at(*hit)
                    logger.info("launch_clicked_text", target=t, at=list(hit))
                    return
            logger.info("launch_click_text_missing", targets=targets)
            return

        if op == "click_world":
            self._click_world()
            return

        if op == "wait":
            ms = int(action.get("ms", 2000))
            time.sleep(min(ms / 1000.0, 10.0))
            return

        # op == "done" (or unknown): no physical action.
        return

    def _focus_search(self) -> None:
        """Focus the Xbox search box: UIA Edit -> OCR 'search' -> hotkey."""
        try:
            window = self.xbox._window()  # noqa: SLF001
            edits = window.descendants(control_type="Edit")
            if edits:
                edits[0].click_input()
                time.sleep(0.3)
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("focus_search_uia_failed", error=str(exc))
        if self.finder.find_text("search") is not None:
            if self.finder.click_text("search", timeout_s=2.0):
                return
        # Last resort: many Xbox builds focus search on typing after Ctrl+F.
        self.finder.press_key("f")

    def _click_world(self) -> None:
        """Prefer an existing world; else create one - all located by OCR."""
        for t in ("play world", "play selected world"):
            hit = self.finder.find_text(t)
            if hit is not None:
                self.finder.click_at(*hit)
                logger.info("launch_world_play", target=t)
                return
        # Click just below the "Worlds" header to hit the first world row.
        header = self.finder.find_text("worlds")
        if header is not None:
            self.finder.click_at(header[0], header[1] + 90)
            time.sleep(1.0)
            if self.finder.click_text("play", timeout_s=3.0):
                return
        # Otherwise create a fresh world.
        for t in ("create new world", "create new"):
            hit = self.finder.find_text(t)
            if hit is not None:
                self.finder.click_at(*hit)
                time.sleep(2.0)
                self.finder.click_text("create", timeout_s=8.0)
                logger.info("launch_world_created")
                return
        # Nothing matched: confirm the highlighted default.
        self.finder.press_enter()

    # ------------------------------------------------------------------ #
    # VLM/LLM arbitration when the screen doesn't match any known state
    # ------------------------------------------------------------------ #
    def _ensure_vlm(self) -> Any | None:
        if self._vlm is not None:
            return self._vlm
        try:
            api_key = (self.config.openai_api_key
                       if self.config.llm.provider is LLMProvider.OPENAI
                       else self.config.anthropic_api_key)
            if not api_key:
                return None
            from tools.vision.vlm_engine import VLMEngine
            self._vlm = VLMEngine(provider=self.config.llm.provider,
                                  model=self.config.llm.vlm_model, api_key=api_key,
                                  max_tokens=self.config.llm.max_tokens,
                                  timeout_s=self.config.llm.request_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("launch_vlm_unavailable", error=str(exc))
            self._vlm = None
        return self._vlm

    def _arbitrate_with_vlm(self, obs: Observation) -> dict[str, Any] | None:
        """Ask the VLM which known state we are on when detection is ambiguous."""
        vlm = self._ensure_vlm()
        if vlm is None or obs.frame is None:
            return None
        try:
            scene = vlm.analyze_scene(
                obs.frame, game="minecraft",
                goal="Launch Minecraft for Windows and reach the in-game world.",
                ocr_texts=obs.ocr_text.split(),
                visual_summary=f"crosshair={obs.crosshair} windows={obs.window_titles[:5]}",
            )
            s = str(scene.screen_state)
            logger.info("launch_vlm_arbitration", screen=s, sees=scene.what_i_see[:80])
            # Map the VLM's generic screen verdict onto a declarative state.
            mapping = {
                "xbox_app": "xbox_home", "game_pass": "xbox_home",
                "main_menu": "mc_title", "menu": "mc_title",
                "game_loading": "loading", "gameplay": "in_world",
            }
            target = mapping.get(s)
            if target:
                return next((st for st in self._states
                             if st.get("name") == target), None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("launch_vlm_failed", error=str(exc))
        return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> bool:
        if not self._states:
            logger.error("launch_no_states")
            return False
        poll_s = float(self._cfg.get("poll_interval_s", 1.5))
        confuse_limit = int(self._cfg.get("confuse_limit", 3))
        # Timing is PROGRESS-based, not a fixed schedule. We only abort if the
        # flow STALLS (no state change) for `stall_timeout_s`; a slow-but-
        # advancing launch is never killed. `hard_cap_s` is a large safety net.
        stall_timeout_s = float(self._cfg.get("stall_timeout_s", 180))
        hard_cap_s = float(self._cfg.get("hard_cap_s", 900))

        # Warm up OCR BEFORE any timer starts - the first PaddleOCR call has a
        # heavy cold-start (~40-50s) that must not eat the stall budget.
        self._warmup_ocr()

        start = time.time()
        last_progress = start
        last_state_name: str | None = None
        confused = 0

        while (time.time() - start) < hard_cap_s:
            obs = self.observe()
            if obs.crosshair:
                logger.info("launch_in_world_confirmed")
                return True

            state = self._match_state(obs)
            if state is None:
                confused += 1
                logger.info("launch_unmatched", cycle=confused,
                            windows=obs.window_titles[:4],
                            ocr_sample=obs.ocr_text[:120])
                if confused >= confuse_limit:
                    state = self._arbitrate_with_vlm(obs)
                    confused = 0
                if state is None:
                    if (time.time() - last_progress) >= stall_timeout_s:
                        logger.warning("launch_stalled", stalled_s=round(
                            time.time() - last_progress, 1), last_state=last_state_name)
                        return False
                    time.sleep(poll_s)
                    continue
            else:
                confused = 0

            # Progress = the matched state changed since the last cycle. As long
            # as the launch keeps advancing, the stall timer keeps resetting.
            name = state.get("name")
            if name != last_state_name:
                logger.info("launch_state", state=name,
                            desc=state.get("description", "")[:60])
                last_state_name = name
                last_progress = time.time()

            if state.get("action", {}).get("op") == "done":
                logger.info("launch_done_state", state=name)
                return True

            self._do_action(state, obs)

            if (time.time() - last_progress) >= stall_timeout_s:
                logger.warning("launch_stalled", stalled_s=round(
                    time.time() - last_progress, 1), last_state=last_state_name)
                return False
            time.sleep(poll_s)

        logger.warning("launch_hard_cap_reached", elapsed_s=round(time.time() - start, 1))
        return False

    def _warmup_ocr(self) -> None:
        """Trigger PaddleOCR's one-time model load before the stall timer runs."""
        try:
            warmup = getattr(self.finder._ocr, "warmup", None)  # noqa: SLF001
            if callable(warmup):
                warmup()
                logger.info("launch_ocr_warmed")
        except Exception as exc:  # noqa: BLE001
            logger.debug("launch_ocr_warmup_failed", error=str(exc))


