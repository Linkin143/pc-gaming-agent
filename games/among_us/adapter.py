"""Among Us adapter - implements the generic GameAdapter interface."""

from __future__ import annotations

from typing import Any

from core.constants import ScreenState
from core.models import PerceptionResult
from core.state import Objective, StructuredGameState
from games.among_us import perception as au_perception
from games.among_us.actions import AmongUsActionBuilder
from games.among_us.state import AmongUsState
from games.common.game_interface import GameActionBuilder, GameAdapter


class AmongUsAdapter(GameAdapter):
    GAME_NAME = "among_us"
    WINDOW_TITLE_RE = ".*Among Us.*"
    SEARCH_NAME = "Among Us"

    def __init__(self) -> None:
        self._action_builder = AmongUsActionBuilder()

    def get_game_name(self) -> str:
        return self.GAME_NAME

    def get_window_title_regex(self) -> str:
        return self.WINDOW_TITLE_RE

    def get_search_name(self) -> str:
        return self.SEARCH_NAME

    def get_initial_state(self) -> StructuredGameState:
        state = StructuredGameState(game=self.GAME_NAME, screen=ScreenState.UNKNOWN)
        state.game_state = AmongUsState().to_game_state()
        state.objective = Objective(type="idle", description="Awaiting goal.")
        return state

    def interpret_perception(self, perception: PerceptionResult,
                             state: StructuredGameState) -> dict[str, Any]:
        previous = None
        if state.game_state:
            try:
                previous = AmongUsState.model_validate(state.game_state)
            except Exception:  # noqa: BLE001
                previous = None
        return au_perception.build_state(perception, previous).to_game_state()

    def get_action_builder(self) -> GameActionBuilder | None:
        return self._action_builder

    def get_default_goal(self) -> str:
        return "Complete all crewmate tasks in Among Us."
