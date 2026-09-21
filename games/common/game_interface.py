"""Abstract game adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import ActionPlan, PerceptionResult, SkillDefinition, SkillIntent
from core.state import StructuredGameState


class GameActionBuilder(ABC):
    """Builds ActionPlans for game-specific skill strategies."""

    @abstractmethod
    def build(self, skill: SkillDefinition, intent: SkillIntent,
              params: dict[str, Any]) -> list[ActionPlan]:
        raise NotImplementedError


class GameAdapter(ABC):
    """Base class every game integration must implement."""

    @abstractmethod
    def get_game_name(self) -> str:
        ...

    @abstractmethod
    def get_window_title_regex(self) -> str:
        ...

    @abstractmethod
    def get_initial_state(self) -> StructuredGameState:
        ...

    @abstractmethod
    def interpret_perception(self, perception: PerceptionResult,
                             state: StructuredGameState) -> dict[str, Any]:
        ...

    def get_action_builder(self) -> GameActionBuilder | None:
        return None

    def get_search_name(self) -> str:
        """Display name used when searching the Xbox app (defaults to title)."""
        return self.get_game_name().replace("_", " ").title()

    def get_default_goal(self) -> str:
        return f"Play {self.get_game_name()}"
