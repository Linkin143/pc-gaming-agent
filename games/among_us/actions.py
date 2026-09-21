"""Among Us game-specific action builder (keyboard/mouse only, bounded bursts)."""

from __future__ import annotations

from typing import Any

from core.constants import ActionType, KeyMode
from core.logger import get_logger
from core.models import ActionPlan, SkillDefinition, SkillIntent
from games.common.game_interface import GameActionBuilder

logger = get_logger("among_us.actions")

_DIRECTION_KEYS = {"up": "w", "down": "s", "left": "a", "right": "d"}


class AmongUsActionBuilder(GameActionBuilder):
    def build(self, skill: SkillDefinition, intent: SkillIntent,
              params: dict[str, Any]) -> list[ActionPlan]:
        name = skill.name
        if name in ("navigate_to_room", "navigate_to_task", "navigate_to"):
            return self._navigate(skill, params)
        if name == "complete_task":
            return [self._tap(skill, "e")]
        if name == "vote_player":
            return self._vote(skill, params)
        logger.warning("unmapped_game_skill", skill=name)
        return [self._tap(skill, "e")]

    def _tap(self, skill: SkillDefinition, key: str) -> ActionPlan:
        return ActionPlan(action_type=ActionType.KEYBOARD_PRESS, mode=KeyMode.TAP,
                          key=key, skill_name=skill.name, description=f"Tap {key}.")

    def _navigate(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        direction = params.get("direction", "right")
        key = _DIRECTION_KEYS.get(direction, "d")
        duration = int(params.get("step_duration_ms", 350))
        return [ActionPlan(action_type=ActionType.KEYBOARD_HOLD, mode=KeyMode.HOLD,
                           key=key, duration_ms=duration, skill_name=skill.name,
                           description=f"Move {direction}.")]

    def _vote(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        return [self._tap(skill, "enter")]
