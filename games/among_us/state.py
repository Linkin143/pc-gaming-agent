"""Among Us game-specific state models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    room: str | None = None
    completed: bool = False


class AmongUsState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str | None = None
    tasks: list[Task] = Field(default_factory=list)
    task_progress: int = 0
    alive: bool = True
    is_impostor: bool = False
    meeting_active: bool = False
    task_active: bool = False
    has_pending_tasks: bool = False

    def to_game_state(self) -> dict:
        return self.model_dump(mode="json")
