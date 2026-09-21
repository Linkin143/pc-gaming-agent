"""Action agent - converts a validated SkillIntent into a structured ActionPlan.

This layer never touches the OS; it only produces ActionPlan objects and
validates them against skill constraints and global safety limits.
"""

from __future__ import annotations

from typing import Any

from core.constants import (
    DEFAULT_MAX_HOLD_MS,
    DEFAULT_MIN_HOLD_MS,
    MOVEMENT_KEYS,
    ActionStrategy,
    ActionType,
    KeyMode,
    MouseButton,
)
from core.exceptions import ActionValidationError
from core.logger import get_logger
from core.models import ActionPlan, Point, SkillDefinition, SkillIntent

logger = get_logger("action")

_STRATEGY_TO_ACTION: dict[str, tuple[ActionType, KeyMode | None]] = {
    ActionStrategy.KEYBOARD_TAP.value: (ActionType.KEYBOARD_PRESS, KeyMode.TAP),
    ActionStrategy.KEYBOARD_HOLD.value: (ActionType.KEYBOARD_HOLD, KeyMode.HOLD),
    ActionStrategy.KEYBOARD_RELEASE.value: (ActionType.KEYBOARD_RELEASE, KeyMode.RELEASE),
    ActionStrategy.KEY_COMBO.value: (ActionType.KEY_COMBO, None),
    ActionStrategy.MOUSE_MOVE.value: (ActionType.MOUSE_MOVE, None),
    ActionStrategy.MOUSE_CLICK.value: (ActionType.MOUSE_CLICK, None),
    ActionStrategy.MOUSE_DOUBLE_CLICK.value: (ActionType.MOUSE_DOUBLE_CLICK, None),
    ActionStrategy.MOUSE_DRAG.value: (ActionType.MOUSE_DRAG, None),
    ActionStrategy.WAIT.value: (ActionType.WAIT, None),
}


class ActionAgent:
    def __init__(self, game_action_builder: Any | None = None) -> None:
        self.game_action_builder = game_action_builder

    def build(self, skill: SkillDefinition, intent: SkillIntent,
              params: dict[str, Any]) -> list[ActionPlan]:
        strategy = str(skill.action_strategy)
        if strategy == ActionStrategy.OBSERVE.value:
            return []
        if strategy == ActionStrategy.GAME_SPECIFIC.value:
            if self.game_action_builder is None:
                raise ActionValidationError(
                    "Skill requires a game-specific action builder but none is registered.",
                    context={"skill": skill.name},
                )
            plans = self.game_action_builder.build(skill, intent, params)
            for p in plans:
                self.validate(p)
            return plans
        plan = self._build_simple(skill, params)
        self.validate(plan)
        return [plan]

    def _build_simple(self, skill: SkillDefinition, params: dict[str, Any]) -> ActionPlan:
        strategy = str(skill.action_strategy)
        if strategy not in _STRATEGY_TO_ACTION:
            raise ActionValidationError("Unknown action strategy.",
                                        context={"strategy": strategy, "skill": skill.name})
        action_type, mode = _STRATEGY_TO_ACTION[strategy]
        plan = ActionPlan(action_type=action_type, mode=mode, skill_name=skill.name,
                          description=skill.description)
        key = params.get("key") or (skill.input.key if skill.input else None)
        if key is None and skill.name in MOVEMENT_KEYS:
            key = MOVEMENT_KEYS[skill.name]

        if action_type in (ActionType.KEYBOARD_PRESS, ActionType.KEYBOARD_HOLD,
                           ActionType.KEYBOARD_RELEASE):
            plan.key = key
            plan.duration_ms = int(params.get("duration_ms", 0) or 0)
        elif action_type is ActionType.KEY_COMBO:
            keys = params.get("keys")
            if isinstance(keys, str):
                keys = [k.strip() for k in keys.replace("+", ",").split(",") if k.strip()]
            plan.keys = keys or (skill.input.keys if skill.input else None)
        elif action_type in (ActionType.MOUSE_MOVE, ActionType.MOUSE_CLICK,
                             ActionType.MOUSE_DOUBLE_CLICK, ActionType.MOUSE_DRAG):
            plan.position = Point(x=int(params["x"]), y=int(params["y"]))
            if action_type is ActionType.MOUSE_DRAG:
                plan.end_position = Point(x=int(params["end_x"]), y=int(params["end_y"]))
            button = params.get("button") or (
                str(skill.input.button) if skill.input and skill.input.button else "left"
            )
            plan.button = MouseButton(button)
            plan.duration_ms = int(params.get("duration_ms", 0) or 0)
        elif action_type is ActionType.WAIT:
            plan.duration_ms = int(params.get("duration_ms", 0) or 0)
        return plan

    def validate(self, plan: ActionPlan) -> None:
        action = str(plan.action_type)
        if action in ("keyboard_press", "keyboard_hold", "keyboard_release"):
            if not plan.key:
                raise ActionValidationError("Keyboard action missing key.",
                                            context={"plan": action})
        if action == "keyboard_hold":
            if not (DEFAULT_MIN_HOLD_MS <= plan.duration_ms <= DEFAULT_MAX_HOLD_MS):
                plan.duration_ms = max(DEFAULT_MIN_HOLD_MS,
                                       min(plan.duration_ms, DEFAULT_MAX_HOLD_MS))
        if action == "key_combo" and not plan.keys:
            raise ActionValidationError("Key combo missing keys.")
        # Mouse clicks may omit position (act at current cursor); moves/drags need it.
        # A RELATIVE mouse_move carries a delta instead of an absolute position.
        if action == "mouse_move" and plan.relative:
            if plan.delta is None:
                raise ActionValidationError("Relative mouse move missing delta.",
                                            context={"plan": action})
        elif action in ("mouse_move", "mouse_drag") and plan.position is None:
            raise ActionValidationError("Mouse action missing position.",
                                        context={"plan": action})
        if action == "mouse_drag" and plan.end_position is None:
            raise ActionValidationError("Drag missing end position.")
        logger.debug("action_validated", action=action, skill=plan.skill_name)
