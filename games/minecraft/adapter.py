"""Minecraft adapter - implements the generic GameAdapter interface."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.constants import ScreenState
from core.models import PerceptionResult
from core.state import Objective, StructuredGameState
from games.common.game_interface import GameActionBuilder, GameAdapter
from games.minecraft import perception as mc_perception
from games.minecraft.actions import MinecraftActionBuilder
from games.minecraft.state import MinecraftState


class MinecraftAdapter(GameAdapter):
    """Adapter mapping generic perception to Minecraft structured state."""

    GAME_NAME = "minecraft"
    # Minecraft for Windows title bar reads "Minecraft".
    WINDOW_TITLE_RE = ".*Minecraft.*"
    # Display name used when searching the Xbox app.
    SEARCH_NAME = "Minecraft for Windows"

    def __init__(self) -> None:
        self._action_builder = MinecraftActionBuilder()

    def get_game_name(self) -> str:
        return self.GAME_NAME

    def get_window_title_regex(self) -> str:
        return self.WINDOW_TITLE_RE

    def get_search_name(self) -> str:
        return self.SEARCH_NAME

    def get_initial_state(self) -> StructuredGameState:
        state = StructuredGameState(game=self.GAME_NAME, screen=ScreenState.UNKNOWN)
        state.game_state = MinecraftState().to_game_state()
        state.objective = Objective(type="reach_title", description="Reach the title screen.")
        return state

    def interpret_perception(
        self, perception: PerceptionResult, state: StructuredGameState
    ) -> dict[str, Any]:
        previous = None
        if state.game_state:
            try:
                previous = MinecraftState.model_validate(
                    {k: v for k, v in state.game_state.items() if not k.startswith("_")}
                )
            except Exception:  # noqa: BLE001 - start fresh on malformed state
                previous = None
        frame = self._load_frame(perception)
        mc_state = mc_perception.build_state(perception, previous, frame=frame)
        return mc_state.to_game_state()

    @staticmethod
    def _load_frame(perception: PerceptionResult) -> np.ndarray | None:
        """Reload the captured frame (BGR) from its saved screenshot reference.

        The real pixels are the source of truth for the HUD analyser. We read the
        PNG the perception layer already saved so we don't re-capture.
        """
        ref = perception.screenshot_ref
        if ref is None or not ref.path:
            return None
        try:
            img = cv2.imread(ref.path, cv2.IMREAD_COLOR)  # BGR
            return img if img is not None and img.size else None
        except Exception:  # noqa: BLE001
            return None

    def authoritative_screen(self, game_state: dict[str, Any]) -> ScreenState | None:
        """Pixel-truth screen override: the crosshair is the in-world ground truth.

        Returns a ScreenState that must win over the generic fused classification,
        or None to leave the fused verdict untouched.
        """
        if not game_state:
            return None
        phase = game_state.get("phase")
        if game_state.get("crosshair_visible") and not (
                game_state.get("is_paused") or game_state.get("inventory_open")):
            return ScreenState.GAMEPLAY
        if game_state.get("is_paused") or game_state.get("inventory_open"):
            return ScreenState.MENU
        if phase == "loading":
            return ScreenState.GAME_LOADING
        if phase == "world_select":
            return ScreenState.MENU
        if phase == "menu":
            return ScreenState.MAIN_MENU
        return None

    def get_action_builder(self) -> GameActionBuilder | None:
        return self._action_builder

    def get_default_goal(self) -> str:
        return "Launch Minecraft for Windows and reach the in-game world."


def get_adapter() -> MinecraftAdapter:
    """Factory used by the adapter registry."""
    return MinecraftAdapter()
