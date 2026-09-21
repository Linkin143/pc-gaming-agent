"""Recovery agent - bounded retries with prioritised strategies (no infinite loops)."""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import RecoveryStrategy
from core.logger import get_logger
from core.models import RecoveryContext, SkillDefinition
from core.state import StructuredGameState

logger = get_logger("recovery")


@dataclass
class RecoveryDecision:
    should_continue: bool
    strategy: RecoveryStrategy | None
    force_vlm: bool
    reset_perception: bool
    replan: bool
    reason: str


_DEFAULT_ORDER: list[RecoveryStrategy] = [
    RecoveryStrategy.REOBSERVE,
    RecoveryStrategy.RERUN_OCR,
    RecoveryStrategy.ESCALATE_VLM,
    RecoveryStrategy.RETRY,
    RecoveryStrategy.RETURN_TO_KNOWN_STATE,
]


class RecoveryAgent:
    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts

    def decide(self, *, skill: SkillDefinition | None, context: RecoveryContext,
               state: StructuredGameState,
               failure_reason: str) -> tuple[RecoveryDecision, RecoveryContext]:
        context.max_attempts = self.max_attempts
        context.failed_skill = skill.name if skill else context.failed_skill
        context.last_error = failure_reason

        if context.exhausted:
            logger.warning("recovery_exhausted", attempts=context.attempt,
                           max_attempts=context.max_attempts, skill=context.failed_skill)
            return (RecoveryDecision(False, RecoveryStrategy.ABORT, False, False, False,
                                     "Recovery attempts exhausted."), context)

        order = self._strategy_order(skill)
        strategy = order[min(context.attempt, len(order) - 1)]
        context.attempt += 1
        context.last_strategy = strategy
        context.history.append(strategy.value)
        decision = self._decision_for(strategy)
        logger.info("recovery_attempt", attempt=context.attempt, strategy=strategy.value,
                    skill=context.failed_skill, reason=failure_reason)
        return decision, context

    @staticmethod
    def _strategy_order(skill: SkillDefinition | None) -> list[RecoveryStrategy]:
        if skill and skill.recovery_strategy:
            order: list[RecoveryStrategy] = []
            for s in skill.recovery_strategy:
                try:
                    order.append(RecoveryStrategy(s))
                except ValueError:
                    continue
            if order:
                return order
        return list(_DEFAULT_ORDER)

    @staticmethod
    def _decision_for(strategy: RecoveryStrategy) -> RecoveryDecision:
        mapping = {
            RecoveryStrategy.REOBSERVE: RecoveryDecision(True, strategy, False, True, True, "Re-observe."),
            RecoveryStrategy.RERUN_OCR: RecoveryDecision(True, strategy, False, True, True, "Re-run OCR."),
            RecoveryStrategy.RERUN_VISION: RecoveryDecision(True, strategy, False, True, True, "Re-run vision."),
            RecoveryStrategy.ESCALATE_VLM: RecoveryDecision(True, strategy, True, True, True, "Escalate to VLM."),
            RecoveryStrategy.REBUILD_STATE: RecoveryDecision(True, strategy, True, True, True, "Rebuild state."),
            RecoveryStrategy.RECALCULATE_TARGET: RecoveryDecision(True, strategy, False, True, True, "Recalculate target."),
            RecoveryStrategy.ALTERNATIVE_SKILL: RecoveryDecision(True, strategy, False, False, True, "Alternative skill."),
            RecoveryStrategy.RETRY: RecoveryDecision(True, strategy, False, True, True, "Retry."),
            RecoveryStrategy.ADJUST_PARAMETERS: RecoveryDecision(True, strategy, False, True, True, "Adjust params."),
            RecoveryStrategy.RETURN_TO_KNOWN_STATE: RecoveryDecision(True, strategy, True, True, True, "Return to known state."),
        }
        return mapping.get(strategy,
                           RecoveryDecision(True, strategy, False, True, True, "Generic recovery."))
