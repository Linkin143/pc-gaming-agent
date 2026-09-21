"""Unit tests for action generation and validation.

Design note: a ``mouse_click`` MAY omit a position (it then clicks at the current
cursor location - used by Minecraft mine/attack). Only ``mouse_move`` and
``mouse_drag`` strictly require a position.
"""

from __future__ import annotations

import pytest

from agents.execution.action_agent import ActionAgent
from core.constants import ActionType
from core.exceptions import ActionValidationError
from core.models import ActionPlan, Point


def test_mouse_move_requires_position():
    agent = ActionAgent()
    plan = ActionPlan(action_type=ActionType.MOUSE_MOVE)
    with pytest.raises(ActionValidationError):
        agent.validate(plan)


def test_relative_mouse_move_passes_with_delta():
    """A relative look move carries a delta (not an absolute position) and is valid."""
    agent = ActionAgent()
    agent.validate(ActionPlan(action_type=ActionType.MOUSE_MOVE,
                              relative=True, delta=(250, 0)))


def test_relative_mouse_move_requires_delta():
    agent = ActionAgent()
    with pytest.raises(ActionValidationError):
        agent.validate(ActionPlan(action_type=ActionType.MOUSE_MOVE, relative=True))


def test_mouse_drag_requires_end_position():
    agent = ActionAgent()
    plan = ActionPlan(action_type=ActionType.MOUSE_DRAG, position=Point(x=5, y=5))
    with pytest.raises(ActionValidationError):
        agent.validate(plan)


def test_mouse_click_without_position_is_allowed():
    agent = ActionAgent()
    # Clicking at the current cursor position is valid (mine/attack use this).
    agent.validate(ActionPlan(action_type=ActionType.MOUSE_CLICK))


def test_valid_mouse_click_with_position_passes():
    agent = ActionAgent()
    agent.validate(ActionPlan(action_type=ActionType.MOUSE_CLICK,
                              position=Point(x=10, y=20)))


def test_keyboard_press_requires_key():
    agent = ActionAgent()
    with pytest.raises(ActionValidationError):
        agent.validate(ActionPlan(action_type=ActionType.KEYBOARD_PRESS))


def test_keyboard_hold_duration_clamped():
    agent = ActionAgent()
    plan = ActionPlan(action_type=ActionType.KEYBOARD_HOLD, key="w", duration_ms=9000)
    agent.validate(plan)
    assert plan.duration_ms <= 2000


def test_key_combo_requires_keys():
    agent = ActionAgent()
    with pytest.raises(ActionValidationError):
        agent.validate(ActionPlan(action_type=ActionType.KEY_COMBO))
