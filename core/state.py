"""State model for PC-GAF (rich Pydantic state + LangGraph TypedDict wrapper)."""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from core.constants import DEFAULT_HISTORY_SIZE, ScreenState
from core.models import (
    ActionPlan,
    ActionResult,
    HistoryEntry,
    PerceptionResult,
    RecoveryContext,
    ScreenshotRef,
    SkillIntent,
    VerificationOutcome,
)


class PlayerState(BaseModel):
    """Generic player state; games may extend via the ``extra`` dict."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    visible: bool = False
    position: dict[str, int] | None = None
    alive: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class Objective(BaseModel):
    """The current objective the planner is pursuing."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: str = "idle"
    target: str | None = None
    description: str = ""
    achieved: bool = False


class StructuredGameState(BaseModel):
    """The structured representation of the current environment (generic)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    game: str = "unknown"
    application_state: str = "unknown"
    screen: ScreenState = ScreenState.UNKNOWN

    player: PlayerState = Field(default_factory=PlayerState)
    detected_objects: list[dict[str, Any]] = Field(default_factory=list)
    ocr_results: list[dict[str, Any]] = Field(default_factory=list)
    ui_elements: list[dict[str, Any]] = Field(default_factory=list)
    screenshot_ref: ScreenshotRef | None = None

    available_actions: list[str] = Field(default_factory=list)
    objective: Objective = Field(default_factory=Objective)
    current_skill: str | None = None
    last_intent: SkillIntent | None = None
    last_action: ActionPlan | None = None
    last_action_result: ActionResult | None = None
    last_verification: VerificationOutcome | None = None

    ocr_confidence: float = 0.0
    vision_confidence: float = 0.0
    overall_confidence: float = 0.0
    recovery: RecoveryContext = Field(default_factory=RecoveryContext)
    history: list[HistoryEntry] = Field(default_factory=list)

    game_state: dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)

    def push_history(self, entry: HistoryEntry, limit: int = DEFAULT_HISTORY_SIZE) -> None:
        self.history.append(entry)
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def apply_perception(self, perception: PerceptionResult) -> None:
        self.screen = ScreenState(perception.screen_state)
        self.detected_objects = [b.model_dump() for b in perception.detected_objects]
        self.ocr_results = [o.model_dump() for o in perception.ocr_results]
        self.ocr_confidence = perception.ocr_confidence
        self.vision_confidence = perception.vision_confidence
        self.overall_confidence = perception.overall_confidence
        if perception.screenshot_ref is not None:
            self.screenshot_ref = perception.screenshot_ref
        self.updated_at = time.time()


def _last_write(existing: Any, new: Any) -> Any:
    return new if new is not None else existing


def _accumulate(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    return (existing or []) + (new or [])


class GameAgentState(TypedDict, total=False):
    """The LangGraph graph state."""

    run_id: str
    game: str
    user_goal: str
    structured: Annotated[dict[str, Any], _last_write]
    intent: Annotated[dict[str, Any] | None, _last_write]
    action_plan: Annotated[dict[str, Any] | None, _last_write]
    action_result: Annotated[dict[str, Any] | None, _last_write]
    perception: Annotated[dict[str, Any] | None, _last_write]
    verification: Annotated[dict[str, Any] | None, _last_write]
    verification_result: Annotated[str | None, _last_write]
    iteration: Annotated[int, _last_write]
    max_iterations: int
    finished: Annotated[bool, _last_write]
    finish_reason: Annotated[str | None, _last_write]
    emergency_stop: Annotated[bool, _last_write]
    _force_vlm: Annotated[bool, _last_write]
    log: Annotated[list[dict[str, Any]], _accumulate]


def new_structured_state(game: str = "unknown") -> StructuredGameState:
    return StructuredGameState(game=game)


def load_structured(state: GameAgentState) -> StructuredGameState:
    raw = state.get("structured")
    if not raw:
        return new_structured_state(state.get("game", "unknown"))
    return StructuredGameState.model_validate(raw)


def dump_structured(structured: StructuredGameState) -> dict[str, Any]:
    return structured.model_dump(mode="json")
