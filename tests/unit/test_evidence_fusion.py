"""Unit tests for OpenCV visual features and multi-signal evidence fusion."""

from __future__ import annotations

import numpy as np

from agents.perception.evidence_fuser import EvidenceFuser
from core.constants import ScreenState
from core.models import OCRResult, StructuredSceneAnalysis, VisualFeatures
from tools.vision.opencv_engine import OpenCVEngine


def _states():
    return {"states": {
        "main_menu": {"screen": "main_menu", "keywords": ["play", "settings"]},
        "gameplay": {"screen": "gameplay", "keywords": ["health", "hotbar"]},
    }}


def test_extract_features_returns_bounded_values():
    eng = OpenCVEngine()
    frame = np.random.randint(0, 255, (200, 320, 3), dtype=np.uint8)
    vf = eng.extract_features(frame)
    assert 0.0 <= vf.edge_density <= 1.0
    assert 0.0 <= vf.brightness_mean <= 1.0
    assert vf.ui_element_count >= 0


def test_motion_score_zero_for_identical_frames():
    eng = OpenCVEngine()
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    vf = eng.extract_features(frame, prev_frame=frame.copy())
    assert vf.motion_score == 0.0


def test_vision_classifies_loading_when_dark():
    fuser = EvidenceFuser(_states())
    dark = VisualFeatures(brightness_mean=0.02, edge_density=0.01)
    state, conf = fuser.classify_from_vision(dark)
    assert state == ScreenState.GAME_LOADING
    assert conf > 0.5


def test_vision_classifies_gameplay_when_motion():
    fuser = EvidenceFuser(_states())
    active = VisualFeatures(motion_score=0.2, center_complexity=0.1)
    state, _ = fuser.classify_from_vision(active)
    assert state == ScreenState.GAMEPLAY


def test_ocr_keyword_classification():
    fuser = EvidenceFuser(_states())
    ocr = [OCRResult(text="Play", confidence=0.9),
           OCRResult(text="Settings", confidence=0.9)]
    state, conf = fuser.classify_from_ocr(ocr)
    assert state == ScreenState.MAIN_MENU
    assert conf > 0


def test_agreement_boosts_confidence():
    fuser = EvidenceFuser(_states())
    # Vision says gameplay; OCR says gameplay -> they agree.
    vf = VisualFeatures(motion_score=0.2, center_complexity=0.1)
    ocr = [OCRResult(text="health hotbar", confidence=0.9)]
    ev = fuser.fuse(visual_features=vf, ocr_results=ocr, scene=None,
                    methods_used=["opencv", "ocr"])
    assert ev.screen_state == ScreenState.GAMEPLAY
    assert ev.signal_agreement is True


def test_should_escalate_when_signals_conflict():
    fuser = EvidenceFuser(_states())
    assert fuser.should_escalate((ScreenState.GAMEPLAY, 0.5),
                                 (ScreenState.MENU, 0.5)) is True
    assert fuser.should_escalate((ScreenState.UNKNOWN, 0.2),
                                 (ScreenState.UNKNOWN, 0.0)) is True


def test_vlm_vote_included_when_present():
    fuser = EvidenceFuser(_states())
    vf = VisualFeatures()  # unknown from vision
    scene = StructuredSceneAnalysis(screen_state="gameplay", confidence=0.9)
    ev = fuser.fuse(visual_features=vf, ocr_results=[], scene=scene,
                    methods_used=["opencv", "vlm"])
    assert ev.scene is not None
    assert "gameplay" in ev.votes
