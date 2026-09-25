"""LangGraph orchestration for the closed-loop automation engine."""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.execution.action_agent import ActionAgent
from agents.perception.perception_agent import PerceptionAgent
from agents.perception.state_agent import StateBuilder
from agents.planner.planner_agent import PlannerAgent
from agents.recovery.recovery_agent import RecoveryAgent
from agents.skills.skill_agent import SkillRegistry
from agents.verification.verifier import VerificationAgent
from core.config import AppConfig
from core.constants import NodeName, VerificationResult
from core.exceptions import EmergencyStopError, ForegroundError
from core.logger import RunLogger, get_logger
from core.models import (
    ActionPlan,
    HistoryEntry,
    PerceptionResult,
    RecoveryContext,
    SkillIntent,
    VerificationOutcome,
)
from core.state import GameAgentState, dump_structured, load_structured
from games.common.game_interface import GameAdapter
from tools.input.keyboard import KeyboardExecutor
from tools.input.mouse import MouseExecutor

logger = get_logger("orchestrator")


class GameAutomationEngine:
    """Builds and runs the LangGraph closed-loop automation graph."""

    def __init__(self, config: AppConfig, adapter: GameAdapter, *,
                 skill_registry: SkillRegistry, perception: PerceptionAgent,
                 planner: PlannerAgent, action_agent: ActionAgent,
                 verifier: VerificationAgent, recovery: RecoveryAgent,
                 keyboard: KeyboardExecutor, mouse: MouseExecutor,
                 checkpointer: Any | None = None) -> None:
        self.config = config
        self.adapter = adapter
        self.skills = skill_registry
        self.perception = perception
        self.planner = planner
        self.action_agent = action_agent
        self.verifier = verifier
        self.recovery = recovery
        self.keyboard = keyboard
        self.mouse = mouse
        self.checkpointer = checkpointer
        self.state_builder = StateBuilder(self.skills.states)
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        g: StateGraph = StateGraph(GameAgentState)
        g.add_node(NodeName.INITIALIZE, self._node_initialize)
        g.add_node(NodeName.LOAD_SKILLS, self._node_load_skills)
        g.add_node(NodeName.CAPTURE_SCREEN, self._node_capture_screen)
        g.add_node(NodeName.PERCEPTION, self._node_perception)
        g.add_node(NodeName.UPDATE_STATE, self._node_update_state)
        g.add_node(NodeName.PLANNER, self._node_planner)
        g.add_node(NodeName.SKILL_SELECTOR, self._node_skill_selector)
        g.add_node(NodeName.ACTION_GENERATOR, self._node_action_generator)
        g.add_node(NodeName.ACTION_VALIDATOR, self._node_action_validator)
        g.add_node(NodeName.EXECUTOR, self._node_executor)
        g.add_node(NodeName.VERIFICATION, self._node_verification)
        g.add_node(NodeName.RECOVERY, self._node_recovery)
        g.add_node(NodeName.FINISH, self._node_finish)

        g.add_edge(START, NodeName.INITIALIZE)
        g.add_edge(NodeName.INITIALIZE, NodeName.LOAD_SKILLS)
        g.add_edge(NodeName.LOAD_SKILLS, NodeName.CAPTURE_SCREEN)
        g.add_edge(NodeName.CAPTURE_SCREEN, NodeName.PERCEPTION)
        g.add_edge(NodeName.PERCEPTION, NodeName.UPDATE_STATE)
        g.add_conditional_edges(NodeName.UPDATE_STATE, self._route_after_update,
                                {"plan": NodeName.PLANNER, "finish": NodeName.FINISH})
        g.add_edge(NodeName.PLANNER, NodeName.SKILL_SELECTOR)
        g.add_edge(NodeName.SKILL_SELECTOR, NodeName.ACTION_GENERATOR)
        g.add_edge(NodeName.ACTION_GENERATOR, NodeName.ACTION_VALIDATOR)
        g.add_edge(NodeName.ACTION_VALIDATOR, NodeName.EXECUTOR)
        g.add_edge(NodeName.EXECUTOR, NodeName.VERIFICATION)
        g.add_conditional_edges(NodeName.VERIFICATION, self._route_after_verification,
                                {"success": NodeName.CAPTURE_SCREEN,
                                 "failure": NodeName.RECOVERY,
                                 "uncertain": NodeName.CAPTURE_SCREEN,
                                 "finish": NodeName.FINISH})
        g.add_conditional_edges(NodeName.RECOVERY, self._route_after_recovery,
                                {"perception": NodeName.CAPTURE_SCREEN,
                                 "finish": NodeName.FINISH})
        g.add_edge(NodeName.FINISH, END)

        if self.checkpointer is not None:
            return g.compile(checkpointer=self.checkpointer)
        return g.compile()

    def _run_logger(self, state: GameAgentState) -> RunLogger:
        return RunLogger(state.get("run_id", "unknown"), state.get("game", "unknown"))

    def _log_entry(self, node: str, **fields: Any) -> list[dict[str, Any]]:
        return [{"node": node, "ts": time.time(), **fields}]

    # -- setup nodes ---------------------------------------------------- #
    def _node_initialize(self, state: GameAgentState) -> dict[str, Any]:
        run_id = state.get("run_id") or uuid.uuid4().hex
        structured = self.adapter.get_initial_state()
        if state.get("user_goal"):
            structured.objective.description = state["user_goal"]
        RunLogger(run_id, self.adapter.get_game_name()).event(
            "initialize", goal=state.get("user_goal", ""))
        return {"run_id": run_id, "game": self.adapter.get_game_name(),
                "structured": dump_structured(structured), "iteration": 0,
                "finished": False, "log": self._log_entry("initialize", run_id=run_id)}

    def _node_load_skills(self, state: GameAgentState) -> dict[str, Any]:
        structured = load_structured(state)
        available = [s.name for s in self.skills.list_available(structured)]
        structured.available_actions = available
        return {"structured": dump_structured(structured),
                "log": self._log_entry("load_skills", count=len(available))}

    # -- perception nodes ----------------------------------------------- #
    def _node_capture_screen(self, state: GameAgentState) -> dict[str, Any]:
        return {"log": self._log_entry("capture_screen")}

    def _node_perception(self, state: GameAgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        force_vlm = bool(state.get("_force_vlm"))
        try:
            perception = self.perception.perceive(run_id=run_id, force_vlm=force_vlm)
        except Exception as exc:  # noqa: BLE001
            logger.error("perception_node_failed", error=str(exc))
            perception = PerceptionResult(changed=True)
        perception = self.state_builder.build_perception(perception, prefer_vlm=force_vlm)
        return {"perception": perception.model_dump(mode="json"),
                "log": self._log_entry("perception", screen=str(perception.screen_state),
                                       confidence=perception.overall_confidence)}

    def _node_update_state(self, state: GameAgentState) -> dict[str, Any]:
        structured = load_structured(state)
        perception_raw = state.get("perception")
        if perception_raw:
            perception = PerceptionResult.model_validate(perception_raw)
            game_updates = self.adapter.interpret_perception(perception, structured)
            self.state_builder.merge_into_state(structured, perception,
                                                game_state_updates=game_updates)
            # Game-specific authoritative screen override (e.g. Minecraft
            # crosshair => gameplay) wins over the generic fused classification.
            override = getattr(self.adapter, "authoritative_screen", None)
            if callable(override):
                forced = override(structured.game_state)
                if forced is not None:
                    structured.screen = forced
                    # A pixel-truth override (e.g. Minecraft crosshair => GAMEPLAY)
                    # is deterministic, so promote the confidence too. Otherwise the
                    # low FUSED confidence (in-world frames have no OCR text and can
                    # be misread by OpenCV) keeps the planner stuck on `observe`.
                    if str(forced) == "gameplay":
                        structured.overall_confidence = max(
                            structured.overall_confidence, 0.92)
        iteration = int(state.get("iteration", 0)) + 1
        return {"structured": dump_structured(structured), "iteration": iteration,
                "log": self._log_entry("update_state", iteration=iteration,
                                       screen=str(structured.screen))}

    # -- planning / skill selection ------------------------------------- #
    def _node_planner(self, state: GameAgentState) -> dict[str, Any]:
        structured = load_structured(state)
        available = self.skills.list_available(structured)
        intent = self.planner.plan(state=structured,
                                   user_goal=state.get("user_goal",
                                                       self.adapter.get_default_goal()),
                                   available_skills=available)
        structured.last_intent = intent
        structured.current_skill = intent.skill
        return {"structured": dump_structured(structured),
                "intent": intent.model_dump(mode="json"),
                "log": self._log_entry("planner", skill=intent.skill,
                                       confidence=intent.confidence, reason=intent.reason)}

    def _node_skill_selector(self, state: GameAgentState) -> dict[str, Any]:
        structured = load_structured(state)
        intent = SkillIntent.model_validate(state["intent"])
        try:
            skill = self.skills.resolve(intent.skill, structured)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skill_resolution_failed", skill=intent.skill, error=str(exc))
            return {"verification_result": VerificationResult.FAILURE.value,
                    "log": self._log_entry("skill_selector", error=str(exc))}
        params = self.skills.apply_defaults(skill, {**intent.parameters})
        if intent.target and "target" not in params:
            params["target"] = intent.target
        return {"intent": {**intent.model_dump(mode="json"), "parameters": params},
                "log": self._log_entry("skill_selector", skill=skill.name)}

    # -- action generation / validation / execution --------------------- #
    def _node_action_generator(self, state: GameAgentState) -> dict[str, Any]:
        intent = SkillIntent.model_validate(state["intent"])
        skill = self.skills.get(intent.skill)
        try:
            plans = self.action_agent.build(skill, intent, intent.parameters)
        except Exception as exc:  # noqa: BLE001
            logger.error("action_generation_failed", error=str(exc))
            return {"action_plan": None,
                    "verification_result": VerificationResult.FAILURE.value,
                    "log": self._log_entry("action_generator", error=str(exc))}
        plan_dumps = [p.model_dump(mode="json") for p in plans]
        return {"action_plan": {"plans": plan_dumps, "skill": skill.name},
                "log": self._log_entry("action_generator", skill=skill.name,
                                       plans=len(plan_dumps))}

    def _node_action_validator(self, state: GameAgentState) -> dict[str, Any]:
        raw = state.get("action_plan") or {}
        plans = raw.get("plans", [])
        try:
            for p in plans:
                self.action_agent.validate(ActionPlan.model_validate(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning("action_validation_failed", error=str(exc))
            return {"verification_result": VerificationResult.FAILURE.value,
                    "log": self._log_entry("action_validator", error=str(exc))}
        return {"log": self._log_entry("action_validator", plans=len(plans))}

    def _node_executor(self, state: GameAgentState) -> dict[str, Any]:
        if state.get("emergency_stop"):
            return {"finished": True, "finish_reason": "emergency_stop",
                    "log": self._log_entry("executor", aborted=True)}
        raw = state.get("action_plan") or {}
        plans = raw.get("plans", [])
        structured = load_structured(state)
        results = []
        try:
            for p in plans:
                plan = ActionPlan.model_validate(p)
                if str(plan.action_type).startswith("mouse"):
                    result = self.mouse.execute(plan)
                elif str(plan.action_type) == "wait":
                    time.sleep(min(plan.duration_ms / 1000.0, 10.0))
                    result = None
                else:
                    result = self.keyboard.execute(plan)
                if result is not None:
                    results.append(result.model_dump(mode="json"))
                    structured.last_action = plan
                    structured.last_action_result = result
        except EmergencyStopError:
            # Abort-level: the emergency stop must halt the whole engine.
            logger.warning("executor_emergency_stop")
            return {"finished": True, "finish_reason": "emergency_stop",
                    "log": self._log_entry("executor", aborted=True)}
        except ForegroundError as exc:
            # The target game window lost focus (e.g. the user alt-tabbed, or the
            # console/IDE is foreground). This is a TRANSIENT, recoverable
            # condition - NOT a reason to crash the run. Surface it as a soft
            # verification failure so the recovery agent can re-focus and retry,
            # instead of the KBM executor's re-raise propagating out of the graph.
            logger.warning("executor_foreground_lost", error=str(exc))
            return {"structured": dump_structured(structured),
                    "verification_result": VerificationResult.FAILURE.value,
                    "verification": {"result": VerificationResult.FAILURE.value,
                                     "reason": "target window not in foreground"},
                    "log": self._log_entry("executor", foreground_lost=True)}
        return {"structured": dump_structured(structured),
                "action_result": {"results": results},
                "log": self._log_entry("executor", executed=len(results))}

    # -- verification --------------------------------------------------- #
    def _node_verification(self, state: GameAgentState) -> dict[str, Any]:
        before = load_structured(state)
        intent = SkillIntent.model_validate(state["intent"])
        skill = self.skills.get(intent.skill)
        run_id = state.get("run_id", "unknown")

        # Fast path: skills that declare `verification_strategy: none` (all the
        # per-frame gameplay skills - mc_move/mc_look_around/mc_approach/etc.) do
        # NOT need a second full perception pass (~40-80s). The next loop iteration
        # re-perceives anyway, so we skip straight to the no-op verdict here.
        if str(skill.verification_strategy) == "none":
            outcome = self.verifier.verify(
                skill, before=before, after_perception=None, after_state=before)
            after = before.model_copy(deep=True)
            after.last_verification = outcome
            after.push_history(HistoryEntry(
                iteration=int(state.get("iteration", 0)), skill=skill.name,
                action_type=str(after.last_action.action_type) if after.last_action else "",
                verification=VerificationResult(outcome.result),
                screen_state=after.screen, note=outcome.reason))
            if VerificationResult(outcome.result) is VerificationResult.SUCCESS:
                after.recovery = RecoveryContext(max_attempts=self.config.max_recovery_attempts)
            return {"structured": dump_structured(after),
                    "verification": outcome.model_dump(mode="json"),
                    "verification_result": str(outcome.result),
                    "log": self._log_entry("verification", result=str(outcome.result),
                                           reason=outcome.reason, fast_path=True)}

        try:
            post = self.perception.perceive(run_id=run_id)
            post = self.state_builder.build_perception(post)
        except Exception as exc:  # noqa: BLE001
            logger.error("verification_perception_failed", error=str(exc))
            post = PerceptionResult(changed=True)
        after = before.model_copy(deep=True)
        game_updates = self.adapter.interpret_perception(post, after)
        self.state_builder.merge_into_state(after, post, game_state_updates=game_updates)
        outcome: VerificationOutcome = self.verifier.verify(
            skill, before=before, after_perception=post, after_state=after)
        after.last_verification = outcome
        after.push_history(HistoryEntry(
            iteration=int(state.get("iteration", 0)), skill=skill.name,
            action_type=str(after.last_action.action_type) if after.last_action else "",
            verification=VerificationResult(outcome.result),
            screen_state=after.screen, note=outcome.reason))
        if VerificationResult(outcome.result) is VerificationResult.SUCCESS:
            after.recovery = RecoveryContext(max_attempts=self.config.max_recovery_attempts)
        return {"structured": dump_structured(after),
                "perception": post.model_dump(mode="json"),
                "verification": outcome.model_dump(mode="json"),
                "verification_result": str(outcome.result),
                "log": self._log_entry("verification", result=str(outcome.result),
                                       reason=outcome.reason)}

    # -- recovery ------------------------------------------------------- #
    def _node_recovery(self, state: GameAgentState) -> dict[str, Any]:
        structured = load_structured(state)
        intent_raw = state.get("intent")
        skill = None
        if intent_raw:
            try:
                skill = self.skills.get(SkillIntent.model_validate(intent_raw).skill)
            except Exception:  # noqa: BLE001
                skill = None
        reason = (state.get("verification") or {}).get("reason", "unknown failure")
        context = structured.recovery
        context.max_attempts = self.config.max_recovery_attempts
        decision, context = self.recovery.decide(skill=skill, context=context,
                                                 state=structured, failure_reason=reason)
        structured.recovery = context
        if decision.reset_perception:
            self.perception.reset()
        return {"structured": dump_structured(structured),
                "_force_vlm": decision.force_vlm,
                "finished": not decision.should_continue,
                "finish_reason": None if decision.should_continue else "recovery_exhausted",
                "log": self._log_entry("recovery",
                                       strategy=decision.strategy.value if decision.strategy else None,
                                       should_continue=decision.should_continue,
                                       attempt=context.attempt)}

    def _node_finish(self, state: GameAgentState) -> dict[str, Any]:
        reason = state.get("finish_reason") or "completed"
        self._run_logger(state).final_outcome(reason, iterations=state.get("iteration", 0))
        return {"finished": True, "finish_reason": reason,
                "log": self._log_entry("finish", reason=reason)}

    # -- routing -------------------------------------------------------- #
    def _route_after_update(self, state: GameAgentState) -> str:
        if state.get("finished") or state.get("emergency_stop"):
            return "finish"
        if int(state.get("iteration", 0)) >= int(
                state.get("max_iterations", self.config.max_iterations)):
            return "finish"
        if load_structured(state).objective.achieved:
            return "finish"
        return "plan"

    def _route_after_verification(self, state: GameAgentState) -> str:
        if state.get("finished") or state.get("emergency_stop"):
            return "finish"
        if int(state.get("iteration", 0)) >= int(
                state.get("max_iterations", self.config.max_iterations)):
            return "finish"
        result = state.get("verification_result")
        if result == VerificationResult.SUCCESS.value:
            return "success"
        if result == VerificationResult.UNCERTAIN.value:
            return "uncertain"
        return "failure"

    def _route_after_recovery(self, state: GameAgentState) -> str:
        return "finish" if state.get("finished") else "perception"

    # -- public API ----------------------------------------------------- #
    def run(self, *, user_goal: str, run_id: str | None = None,
            max_iterations: int | None = None) -> dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex
        initial: GameAgentState = {
            "run_id": run_id, "game": self.adapter.get_game_name(),
            "user_goal": user_goal or self.adapter.get_default_goal(),
            "max_iterations": max_iterations or self.config.max_iterations,
            "iteration": 0, "finished": False,
        }
        cfg = {"configurable": {"thread_id": run_id},
               "recursion_limit": self.config.max_iterations * 12 + 50}
        return self._graph.invoke(initial, config=cfg)

    @property
    def graph(self) -> Any:
        return self._graph
