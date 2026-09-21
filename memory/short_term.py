"""Short-term memory - a bounded ring buffer of recent history entries."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from core.constants import DEFAULT_HISTORY_SIZE
from core.models import HistoryEntry


class ShortTermMemory:
    def __init__(self, capacity: int = DEFAULT_HISTORY_SIZE) -> None:
        self.capacity = capacity
        self._buffer: deque[HistoryEntry] = deque(maxlen=capacity)

    def add(self, entry: HistoryEntry) -> None:
        self._buffer.append(entry)

    def recent(self, n: int | None = None) -> list[HistoryEntry]:
        items = list(self._buffer)
        return items if n is None else items[-n:]

    def extend(self, entries: Iterable[HistoryEntry]) -> None:
        for e in entries:
            self.add(e)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
