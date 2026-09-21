"""Unit tests for skill loading, validation, precondition eval."""

from __future__ import annotations

import pytest

from core.constants import ScreenState
from core.exceptions import PreconditionError, SkillNotFoundError
from core.state import new_structured_state


def test_loads_common_and_game_skills(mc_registry):
    names = {s.name for s in mc_registry.all_skills()}
    assert "move_right" in names       # common
    assert "mc_move" in names          # game-specific


def test_precondition_true_when_screen_matches(mc_registry):
    state = new_structured_state("minecraft")
    state.screen = ScreenState.GAMEPLAY
    assert mc_registry.evaluate_precondition("state.screen == gameplay", state) is True


def test_precondition_false_when_screen_differs(mc_registry):
    state = new_structured_state("minecraft")
    state.screen = ScreenState.MAIN_MENU
    assert mc_registry.evaluate_precondition("state.screen == gameplay", state) is False


def test_resolve_raises_on_unmet_precondition(mc_registry):
    state = new_structured_state("minecraft")
    state.screen = ScreenState.MAIN_MENU
    with pytest.raises(PreconditionError):
        mc_registry.resolve("mc_move", state)


def test_get_unknown_skill_raises(mc_registry):
    with pytest.raises(SkillNotFoundError):
        mc_registry.get("does_not_exist")


def test_apply_defaults_clamps_range(mc_registry):
    skill = mc_registry.get("mc_move")
    params = mc_registry.apply_defaults(skill, {"duration_ms": 99999})
    assert params["duration_ms"] <= 1500


def test_list_available_filters_by_precondition(mc_registry):
    state = new_structured_state("minecraft")
    state.screen = ScreenState.GAMEPLAY
    available = {s.name for s in mc_registry.list_available(state)}
    assert "mc_move" in available
    # xbox_maximize requires the xbox_app screen, so it should not be available.
    assert "xbox_maximize" not in available
