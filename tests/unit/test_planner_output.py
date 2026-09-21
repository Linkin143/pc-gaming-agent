"""Unit tests for the planner's deterministic fallback (LLM forced off).

These verify the game-agnostic, evidence-driven fallback logic. We force the
LLM off so the tests are deterministic and do not depend on ambient API keys.
"""

from __future__ import annotations

from agents.planner.planner_agent import PlannerAgent
from core.constants import ScreenState
from core.state import Objective, new_structured_state


def _deterministic_planner(config) -> PlannerAgent:
    planner = PlannerAgent(config)
    # Force the deterministic path regardless of whether an API key is present.
    planner._llm = None  # noqa: SLF001
    planner._structured_llm = None  # noqa: SLF001
    return planner


def test_planner_selects_from_available_only(config, registry):
    planner = _deterministic_planner(config)
    assert planner.llm_available is False

    state = new_structured_state("among_us")
    state.screen = ScreenState.GAMEPLAY
    state.overall_confidence = 0.9
    available = registry.list_available(state)
    intent = planner.plan(state=state, user_goal="Complete tasks", available_skills=available)
    names = {s.name for s in available}
    assert intent.skill in names


def test_planner_observes_when_unknown(config, registry):
    planner = _deterministic_planner(config)
    state = new_structured_state("among_us")
    state.screen = ScreenState.UNKNOWN
    state.overall_confidence = 0.1
    available = registry.list_available(state)
    intent = planner.plan(state=state, user_goal="x", available_skills=available)
    assert intent.skill == "observe"


def test_planner_navigates_for_navigation_objective(config, registry):
    planner = _deterministic_planner(config)
    state = new_structured_state("among_us")
    state.screen = ScreenState.GAMEPLAY
    state.overall_confidence = 0.9
    state.objective = Objective(type="navigate", target="Electrical")
    available = registry.list_available(state)
    intent = planner.plan(state=state, user_goal="Go to Electrical", available_skills=available)
    assert intent.skill == "navigate_to_room"
    assert intent.target == "Electrical"


def test_planner_output_confidence_in_range(config, registry):
    planner = _deterministic_planner(config)
    state = new_structured_state("among_us")
    state.screen = ScreenState.GAMEPLAY
    available = registry.list_available(state)
    intent = planner.plan(state=state, user_goal="x", available_skills=available)
    assert 0.0 <= intent.confidence <= 1.0
