"""Integration tests for the bounded recovery loop."""

from __future__ import annotations

from agents.recovery.recovery_agent import RecoveryAgent
from core.models import RecoveryContext
from core.state import new_structured_state


def test_recovery_is_bounded(registry):
    agent = RecoveryAgent(max_attempts=3)
    state = new_structured_state("among_us")
    skill = registry.get("move_right")
    ctx = RecoveryContext(max_attempts=3)

    decisions = []
    for _ in range(5):
        decision, ctx = agent.decide(
            skill=skill, context=ctx, state=state, failure_reason="no movement"
        )
        decisions.append(decision.should_continue)

    # It must stop continuing once attempts are exhausted (no infinite loop).
    assert decisions[0] is True
    assert decisions[-1] is False


def test_recovery_uses_skill_strategy_order(registry):
    agent = RecoveryAgent(max_attempts=3)
    state = new_structured_state("among_us")
    skill = registry.get("complete_task")  # recovery: reobserve, recalculate_target, retry
    ctx = RecoveryContext(max_attempts=3)
    decision, ctx = agent.decide(
        skill=skill, context=ctx, state=state, failure_reason="task failed"
    )
    assert decision.strategy.value == "reobserve"
    assert ctx.attempt == 1


def test_recovery_escalates_to_vlm(registry):
    agent = RecoveryAgent(max_attempts=5)
    state = new_structured_state("among_us")
    skill = registry.get("detect_room")  # recovery: reobserve, escalate_vlm
    ctx = RecoveryContext(max_attempts=5)
    agent.decide(skill=skill, context=ctx, state=state, failure_reason="x")
    decision, ctx = agent.decide(
        skill=skill, context=ctx, state=state, failure_reason="x"
    )
    assert decision.strategy.value == "escalate_vlm"
    assert decision.force_vlm is True
