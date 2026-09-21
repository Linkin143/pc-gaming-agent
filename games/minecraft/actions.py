"""Minecraft game-specific action builder.

Translates ``game_specific`` Minecraft skills into concrete, bounded ActionPlan
sequences using keyboard/mouse only. Movement uses short WASD bursts; looking
uses relative mouse moves; mine/place use mouse buttons. The closed loop
re-observes between actions rather than holding keys indefinitely.
"""

from __future__ import annotations

from typing import Any

from core.constants import ActionType, KeyMode, MouseButton
from core.logger import get_logger
from core.models import ActionPlan, Point, SkillDefinition, SkillIntent
from games.common.game_interface import GameActionBuilder

logger = get_logger("minecraft.actions")

# Minecraft default key bindings.
_MOVE_KEYS = {"forward": "w", "backward": "s", "left": "a", "right": "d"}


class MinecraftActionBuilder(GameActionBuilder):
    """Builds ActionPlans for Minecraft game-specific skills."""

    def __init__(self) -> None:
        # Lazily created desktop automation for xbox_* launch skills.
        self._xbox: Any | None = None

    def _get_xbox(self) -> Any:
        if self._xbox is None:
            from tools.desktop.winapp import XboxDesktopAutomation
            self._xbox = XboxDesktopAutomation()
        return self._xbox

    # -- xbox launch skills (desktop UIA, not keyboard/mouse ActionPlans) --- #
    def _handle_xbox(self, name: str, params: dict[str, Any]) -> bool:
        xbox = self._get_xbox()
        search = params.get("query") or params.get("name") or "Minecraft for Windows"
        try:
            if name == "xbox_open_app":
                xbox.launch_xbox_app()
                xbox.wait_for_xbox_ready()
            elif name == "xbox_maximize":
                xbox.maximize()
            elif name == "xbox_search_game":
                xbox.search_game(search)
            elif name == "xbox_select_installed_game":
                xbox.select_game_from_results(params.get("name") or "Minecraft")
            elif name == "xbox_click_play":
                xbox.click_play()
            elif name == "xbox_launch_minecraft":
                from agents.launch.launch_agent import LaunchAgent

                # Skill-driven, screen-truth launch: reads the declarative
                # launch state-machine and decides each action from the live
                # screen (OpenCV crosshair + OCR), escalating to VLM on doubt.
                LaunchAgent(xbox=xbox).run()
            else:
                return False
            logger.info("xbox_skill_done", skill=name)
        except Exception as exc:  # noqa: BLE001 - desktop flow is best-effort
            logger.warning("xbox_skill_failed", skill=name, error=str(exc))
        return True

    def build(
        self, skill: SkillDefinition, intent: SkillIntent, params: dict[str, Any]
    ) -> list[ActionPlan]:
        name = skill.name
        if name.startswith("xbox_"):
            self._handle_xbox(name, params)
            return []  # No keyboard/mouse ActionPlan; desktop layer handled it.
        if name in ("mc_move", "mc_approach"):
            return self._move(skill, params)
        if name == "mc_jump":
            return [self._tap(skill, "space")]
        if name in ("mc_sprint_forward", "mc_flee_threat"):
            return self._sprint(skill, params)
        if name in ("mc_look", "mc_look_around"):
            return self._look(skill, params)
        if name == "mc_unpause":
            return [self._tap(skill, "esc")]
        if name in ("mc_mine_block", "mc_chop_wood"):
            return [self._hold_mouse(skill, MouseButton.LEFT, params.get("duration_ms", 1500))]
        if name == "mc_mine_ground":
            # Look down first, then hold-mine the block below.
            return [
                self._look_delta(skill, 0, 220),
                self._hold_mouse(skill, MouseButton.LEFT, params.get("duration_ms", 1500)),
            ]
        if name in ("mc_fight",):
            return [self._hold_mouse(skill, MouseButton.LEFT, 400)]
        if name == "mc_place_block":
            return [self._click_mouse(skill, MouseButton.RIGHT)]
        if name == "mc_attack":
            return [self._click_mouse(skill, MouseButton.LEFT)]
        if name == "mc_open_inventory":
            return [self._tap(skill, "e")]
        if name == "mc_select_hotbar":
            slot = int(params.get("slot", 1))
            slot = max(1, min(slot, 9))
            return [self._tap(skill, str(slot))]
        if name == "mc_pause":
            return [self._tap(skill, "esc")]
        if name in ("mc_start_singleplayer", "mc_click_button"):
            return self._click_at(skill, params)
        logger.warning("unmapped_mc_skill", skill=name)
        return [self._tap(skill, "esc")]

    # -- primitives ------------------------------------------------------- #
    def _tap(self, skill: SkillDefinition, key: str) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.KEYBOARD_PRESS, mode=KeyMode.TAP,
            key=key, skill_name=skill.name, description=f"Tap {key}.",
        )

    def _move(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        direction = params.get("direction", "forward")
        key = _MOVE_KEYS.get(direction, "w")
        duration = int(params.get("duration_ms", 400))
        return [
            ActionPlan(
                action_type=ActionType.KEYBOARD_HOLD, mode=KeyMode.HOLD,
                key=key, duration_ms=duration, skill_name=skill.name,
                description=f"Move {direction}.",
            )
        ]

    def _sprint(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        # Sprint in Minecraft for Windows (Bedrock) is triggered by HOLDING the
        # forward key (auto-sprint) - NOT Shift+W. Shift is the sneak/crouch key,
        # so the old combo made the player creep forward instead of sprinting.
        duration = int(params.get("duration_ms", 600))
        return [
            ActionPlan(
                action_type=ActionType.KEYBOARD_HOLD, mode=KeyMode.HOLD,
                key="w", duration_ms=duration, skill_name=skill.name,
                description="Sprint forward (hold W).",
            )
        ]

    def _look(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        return [self._look_delta(skill, int(params.get("dx", 0)),
                                 int(params.get("dy", 0)))]

    def _look_delta(self, skill: SkillDefinition, dx: int, dy: int) -> ActionPlan:
        # Look via a RELATIVE mouse delta. Minecraft captures the mouse with raw
        # input and ignores absolute cursor positioning, so the previous approach
        # (moving the OS cursor to an absolute screen offset) turned the camera by
        # nothing. `relative`/`delta` route through MouseExecutor._move_relative,
        # which emits a true relative movement event the game reads.
        return ActionPlan(
            action_type=ActionType.MOUSE_MOVE,
            relative=True, delta=(int(dx), int(dy)),
            duration_ms=80, skill_name=skill.name, description="Look (relative delta).",
        )

    def _hold_mouse(self, skill: SkillDefinition, button: MouseButton, ms: int) -> ActionPlan:
        # Real press-and-hold for mining / continuous attack.
        return ActionPlan(
            action_type=ActionType.MOUSE_HOLD, button=button,
            position=None, duration_ms=int(ms), skill_name=skill.name,
            description=f"Hold {button} for {ms}ms.",
        )

    def _click_mouse(self, skill: SkillDefinition, button: MouseButton) -> ActionPlan:
        return ActionPlan(
            action_type=ActionType.MOUSE_CLICK, button=button, position=None,
            skill_name=skill.name, description=f"Click {button}.",
        )

    def _click_at(self, skill: SkillDefinition, params: dict[str, Any]) -> list[ActionPlan]:
        if "x" in params and "y" in params:
            return [
                ActionPlan(
                    action_type=ActionType.MOUSE_CLICK, button=MouseButton.LEFT,
                    position=Point(x=int(params["x"]), y=int(params["y"])),
                    skill_name=skill.name, description="Click UI button.",
                )
            ]
        # No coordinates: press Enter to activate the focused button.
        return [self._tap(skill, "enter")]
