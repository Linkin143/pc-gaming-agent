"""Among Us-specific perception helpers."""

from __future__ import annotations

from core.models import PerceptionResult
from games.among_us.state import AmongUsState

ROOMS = ["Cafeteria", "Weapons", "Navigation", "O2", "Shields", "Communications",
         "Storage", "Admin", "Electrical", "Lower Engine", "Upper Engine",
         "Security", "Reactor", "MedBay"]
_TASK = ["use", "download", "upload", "swipe", "align", "calibrate", "fix"]
_MEETING = ["emergency meeting", "dead body reported", "discuss", "who is"]


def detect_room(perception: PerceptionResult) -> str | None:
    text = perception.all_text().lower()
    for room in ROOMS:
        if room.lower() in text:
            return room
    return None


def build_state(perception: PerceptionResult,
                previous: AmongUsState | None = None) -> AmongUsState:
    state = previous.model_copy(deep=True) if previous else AmongUsState()
    room = detect_room(perception)
    if room:
        state.room = room
    text = perception.all_text().lower()
    state.task_active = any(k in text for k in _TASK)
    state.meeting_active = any(k in text for k in _MEETING)
    state.has_pending_tasks = any(not t.completed for t in state.tasks)
    return state
