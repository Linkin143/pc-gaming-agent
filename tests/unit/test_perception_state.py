"""Unit tests for perception->state conversion and OCR parsing."""

from __future__ import annotations

import numpy as np

from agents.perception.state_agent import StateBuilder
from core.constants import ScreenState
from core.models import OCRResult, PerceptionResult
from games.among_us import perception as au
from tools.ocr.paddle_engine import PaddleOCREngine


def _states_yaml():
    return {
        "states": {
            "gameplay": {"screen": "gameplay", "keywords": ["use", "report", "tasks"]},
            "main_menu": {"screen": "main_menu", "keywords": ["online", "freeplay"]},
        }
    }


def test_state_builder_classifies_gameplay():
    builder = StateBuilder(_states_yaml())
    ocr = [OCRResult(text="Use Report Tasks", confidence=0.9)]
    screen, conf = builder.classify_screen(ocr)
    assert screen == ScreenState.GAMEPLAY
    assert conf > 0


def test_state_builder_unknown_when_no_match():
    builder = StateBuilder(_states_yaml())
    ocr = [OCRResult(text="zzz", confidence=0.9)]
    screen, conf = builder.classify_screen(ocr)
    assert screen == ScreenState.UNKNOWN
    assert conf == 0.0


def test_build_perception_sets_confidence():
    builder = StateBuilder(_states_yaml())
    perception = PerceptionResult(
        ocr_results=[OCRResult(text="Use Tasks", confidence=0.8)],
        ocr_confidence=0.8,
    )
    out = builder.build_perception(perception)
    assert out.screen_state == ScreenState.GAMEPLAY
    assert out.overall_confidence >= 0.0


def test_among_us_detect_room():
    perception = PerceptionResult(ocr_results=[OCRResult(text="Electrical", confidence=0.9)])
    assert au.detect_room(perception) == "Electrical"


def test_among_us_build_state_flags_task():
    perception = PerceptionResult(
        ocr_results=[OCRResult(text="Download Data  Electrical", confidence=0.9)]
    )
    st = au.build_state(perception)
    assert st.room == "Electrical"
    assert st.task_active is True


def test_ocr_legacy_parser():
    engine = PaddleOCREngine()
    raw = [[[[[0, 0], [10, 0], [10, 10], [0, 10]], ("Hello", 0.99)]]]
    results = engine._parse_legacy(raw)
    assert results[0].text == "Hello"
    assert results[0].confidence == 0.99
    assert results[0].rect is not None


def test_ocr_predict_parser():
    engine = PaddleOCREngine()
    raw = [{"rec_texts": ["A", "B"], "rec_scores": [0.9, 0.8],
            "rec_polys": [[[0, 0], [5, 0], [5, 5], [0, 5]], [[0, 0], [5, 0], [5, 5], [0, 5]]]}]
    results = engine._parse_predict(raw)
    assert [r.text for r in results] == ["A", "B"]


def test_frame_difference_zero_for_identical():
    from tools.capture.screen_capture import ScreenCapture

    a = np.zeros((20, 20, 3), dtype=np.uint8)
    assert ScreenCapture.frame_difference(a, a.copy()) == 0.0
