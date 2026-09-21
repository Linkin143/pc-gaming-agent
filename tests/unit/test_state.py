"""Unit tests for the state models."""

from __future__ import annotations

from core.constants import ScreenState
from core.models import HistoryEntry, OCRResult, PerceptionResult, ScreenshotRef
from core.state import StructuredGameState, dump_structured, load_structured, new_structured_state


def test_new_state_defaults():
    state = new_structured_state("minecraft")
    assert state.game == "minecraft"
    assert state.screen == ScreenState.UNKNOWN


def test_state_roundtrip_serialisation():
    state = new_structured_state("minecraft")
    state.screen = ScreenState.GAMEPLAY
    restored = load_structured({"structured": dump_structured(state), "game": "minecraft"})
    assert restored.screen == ScreenState.GAMEPLAY


def test_push_history_bounded():
    state = new_structured_state()
    for i in range(50):
        state.push_history(HistoryEntry(iteration=i), limit=10)
    assert len(state.history) == 10


def test_apply_perception_updates_confidence():
    state = new_structured_state("minecraft")
    perception = PerceptionResult(screen_state=ScreenState.GAMEPLAY,
                                  ocr_results=[OCRResult(text="Tasks", confidence=0.9)],
                                  ocr_confidence=0.9, overall_confidence=0.9,
                                  screenshot_ref=ScreenshotRef(path="x.png", width=10, height=10))
    state.apply_perception(perception)
    assert state.screen == ScreenState.GAMEPLAY
    assert state.ocr_confidence == 0.9


def test_state_has_no_raw_pixels():
    assert "image" not in StructuredGameState.model_fields
