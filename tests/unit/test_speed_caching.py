"""Tests for the execution-speed / caching optimisations.

Cover: OCR result caching (cache hit avoids re-inference), the fast 16x9 image
key, in-world OCR skip in the perception agent, OpenCV feature-extraction
downscaling, and the verification fast-path for `none`-strategy skills.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import ScreenState
from core.models import OCRResult
from tools.ocr.paddle_engine import PaddleOCREngine
from tools.vision.opencv_engine import OpenCVEngine


# ----------------------------- OCR caching ------------------------------ #

class _CountingOCR(PaddleOCREngine):
    """PaddleOCREngine whose actual inference is replaced by a call counter."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.infer_calls = 0

    def _run(self, engine, image):  # noqa: ANN001 - override the heavy inference
        self.infer_calls += 1
        return [OCRResult(text="play", confidence=0.9)]

    def _ensure_engine(self):  # avoid loading the real PaddleOCR model
        return object()


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)


def test_ocr_cache_hit_skips_second_inference():
    ocr = _CountingOCR(cache_ttl_seconds=8.0)
    frame = _frame(1)
    ocr.read_text(frame, use_cache=True)
    ocr.read_text(frame.copy(), use_cache=True)   # identical content -> cache hit
    assert ocr.infer_calls == 1


def test_ocr_cache_miss_on_changed_frame():
    ocr = _CountingOCR(cache_ttl_seconds=8.0)
    ocr.read_text(_frame(1), use_cache=True)
    ocr.read_text(_frame(2), use_cache=True)       # different content -> re-infer
    assert ocr.infer_calls == 2


def test_ocr_use_cache_false_always_infers():
    ocr = _CountingOCR(cache_ttl_seconds=8.0)
    frame = _frame(1)
    ocr.read_text(frame, use_cache=False)
    ocr.read_text(frame, use_cache=False)
    assert ocr.infer_calls == 2


def test_image_key_stable_and_distinct():
    a = _frame(1)
    assert PaddleOCREngine._image_key(a) == PaddleOCREngine._image_key(a.copy())
    assert PaddleOCREngine._image_key(a) != PaddleOCREngine._image_key(_frame(2))


# ------------------- OpenCV feature-extraction downscale ----------------- #

def test_extract_features_downscales_large_frame():
    eng = OpenCVEngine()
    big = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    vf = eng.extract_features(big)          # must not raise; runs on the small copy
    assert 0.0 <= vf.edge_density <= 1.0
    assert 0.0 <= vf.brightness_mean <= 1.0


def test_downscale_pair_scales_prev_to_match():
    big = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    prev = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    small, small_prev = OpenCVEngine._downscale_pair(big, prev)
    assert small.shape[1] == OpenCVEngine._FEATURE_TARGET_W
    assert small_prev is not None and small_prev.shape == small.shape


def test_downscale_pair_noop_for_small_frame():
    small_in = np.zeros((100, 200, 3), dtype=np.uint8)
    out, prev = OpenCVEngine._downscale_pair(small_in, None)
    assert out.shape == small_in.shape        # already small: unchanged
    assert prev is None


def test_motion_score_survives_downscale():
    eng = OpenCVEngine()
    a = np.zeros((1080, 1920, 3), dtype=np.uint8)
    b = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    vf = eng.extract_features(b, prev_frame=a)   # big diff -> high motion
    assert vf.motion_score > 0.5


# --------------- perception agent: in-world OCR skip --------------------- #

def _perception_agent(game, ocr, config):
    from agents.perception.perception_agent import PerceptionAgent

    class _Cap:
        def capture_full(self):
            class _R:
                image = _frame(3)
                region = None
                timestamp = 0.0
            return _R()

        def save(self, *_a, **_k):
            return None

    return PerceptionAgent(config, capture=_Cap(), ocr=ocr, vlm=None, game=game)


def test_perception_skips_ocr_when_in_world(config):
    ocr = _CountingOCR()
    agent = _perception_agent("minecraft", ocr, config)
    # Force the crosshair check to report in-world.
    agent._is_in_world = lambda _frame: True
    agent.perceive(run_id="t", save_screenshot=False)
    assert ocr.infer_calls == 0            # OCR skipped entirely in-world


def test_perception_runs_ocr_when_not_in_world(config):
    ocr = _CountingOCR()
    agent = _perception_agent("minecraft", ocr, config)
    agent._is_in_world = lambda _frame: False
    agent.perceive(run_id="t", save_screenshot=False)
    assert ocr.infer_calls == 1            # OCR runs when not in-world


def test_is_in_world_false_for_non_minecraft(config):
    agent = _perception_agent("among_us", _CountingOCR(), config)
    assert agent._is_in_world(_frame(4)) is False


# ---------------- ROI crop + precomputed reuse (no 2nd OCR) --------------- #

def test_read_text_roi_runs_once_and_translates_coords():
    ocr = _CountingOCR(cache_ttl_seconds=8.0)
    frame = _frame(5)
    results = ocr.read_text_roi(frame, use_cache=True)
    assert ocr.infer_calls == 1
    # coords are translated back to the full-frame origin (>= ROI offset).
    for r in results:
        if r.rect is not None:
            assert r.rect.x >= int(480 * ocr._ROI_LEFT) - 1


def test_read_text_roi_crops_smaller_than_full():
    ocr = _CountingOCR()
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi, ox, oy = ocr._crop_roi(big)
    assert roi.shape[1] < big.shape[1] and roi.shape[0] < big.shape[0]
    assert ox > 0 and oy > 0


def test_match_in_frame_whitespace_insensitive_target():
    """Regression: OCR renders 'Minecraft for Windows' as one token
    'MinecraftforWindows'. The spaced target must still match it (and must NOT
    fall through to a different 'Minecraft Launcher' tile)."""
    from core.models import OCRResult, Rect
    from tools.desktop.screen_finder import ScreenFinder

    ocr = _CountingOCR()
    sf = ScreenFinder(ocr=ocr)
    precomputed = [
        OCRResult(text="MinecraftforWindows", confidence=0.9,
                  rect=Rect(x=380, y=750, width=120, height=20)),
        OCRResult(text="Minecraft Launcher", confidence=0.95,
                  rect=Rect(x=960, y=750, width=120, height=20)),
    ]
    # Spaced target should hit the FOR WINDOWS tile (~centre 440,760), not launcher.
    hit = sf._match_in_frame(_frame(9), "minecraft for windows", 0.35, (),
                             precomputed=precomputed)
    assert hit == (440, 760)
    # And the launcher target still resolves to the launcher tile.
    hit2 = sf._match_in_frame(_frame(9), "minecraft launcher", 0.35, (),
                              precomputed=precomputed)
    assert hit2 == (1020, 760)


def test_match_in_frame_prefers_button_over_paragraph():
    """Regression: the game card has a real 'Play' button (lower OCR confidence)
    AND a legal sentence '...while you play.' (higher confidence). The exact
    button label must win over the substring-in-paragraph, despite lower conf."""
    from core.models import OCRResult, Rect
    from tools.desktop.screen_finder import ScreenFinder

    ocr = _CountingOCR()
    sf = ScreenFinder(ocr=ocr)
    precomputed = [
        OCRResult(text="Play", confidence=0.75,
                  rect=Rect(x=387, y=619, width=40, height=20)),   # centre (407,629)
        OCRResult(text="and associated data while you play.", confidence=0.90,
                  rect=Rect(x=1135, y=829, width=200, height=20)),  # centre (1235,839)
    ]
    hit = sf._match_in_frame(_frame(11), "play", 0.35,
                             ("replay", "gameplay"), precomputed=precomputed)
    assert hit == (407, 629)   # the button, not the paragraph


def test_read_text_roi_cache_hit_does_not_double_offset():
    """Regression: read_text_roi() must NOT mutate the cached result objects.

    On a cache HIT read_text() returns the SAME objects (rects in ROI-local
    coords). The old code did `r.rect = Rect(x + ox, ...)` in place, so every
    reuse added the ROI offset again - a button read at (968,578) drifted to
    (1123,701), then (1278,825)... and the agent clicked empty space next to the
    real button (the mc_title 'clicks Marketplace' stall). Coordinates must be
    identical no matter how many times the cached frame is re-read."""
    from core.models import OCRResult, Rect

    class _RectOCR(PaddleOCREngine):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.infer_calls = 0

        def _run(self, engine, image):  # noqa: ANN001
            self.infer_calls += 1
            # ROI-local coordinate (as OCR returns before offset translation).
            return [OCRResult(text="Play", confidence=0.8,
                              rect=Rect(x=10, y=20, width=40, height=16))]

        def _ensure_engine(self):
            return object()

    ocr = _RectOCR(cache_ttl_seconds=8.0)
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    r1 = ocr.read_text_roi(big, use_cache=True)          # cache miss -> infer
    r2 = ocr.read_text_roi(big.copy(), use_cache=True)   # cache hit  -> reuse
    r3 = ocr.read_text_roi(big.copy(), use_cache=True)   # cache hit  -> reuse
    assert ocr.infer_calls == 1                          # only one real inference
    # Every call must yield IDENTICAL, correctly-offset coordinates.
    assert r1[0].rect is not None
    c1 = (r1[0].rect.x, r1[0].rect.y)
    assert (r2[0].rect.x, r2[0].rect.y) == c1
    assert (r3[0].rect.x, r3[0].rect.y) == c1
    # And the offset was applied exactly once (ROI-local 10,20 + ROI origin).
    ox = int(1920 * PaddleOCREngine._ROI_LEFT)
    oy = int(1080 * PaddleOCREngine._ROI_TOP)
    assert c1 == (10 + ox, 20 + oy)


def test_match_in_frame_fuzzy_button_label_ocr_misread():
    """Regression: Minecraft's title-screen 'Play' button is OCR'd as 'Flay'
    (P->F). A short standalone label within 1 edit of the target must still
    match, so the launch flow doesn't stall on mc_title. But the fuzzy fallback
    must NOT trigger inside a longer paragraph that happens to be 1 edit off."""
    from core.models import OCRResult, Rect
    from tools.desktop.screen_finder import ScreenFinder

    ocr = _CountingOCR()
    sf = ScreenFinder(ocr=ocr)
    precomputed = [
        OCRResult(text="Flay", confidence=0.52,
                  rect=Rect(x=380, y=619, width=40, height=20)),   # centre (400,629)
        OCRResult(text="Marketplace", confidence=0.89,
                  rect=Rect(x=900, y=700, width=200, height=20)),
    ]
    hit = sf._match_in_frame(_frame(12), "play", 0.35, (), precomputed=precomputed)
    assert hit == (400, 629)   # fuzzy-matched the misread button

    # An EXACT/substring match must always beat a fuzzy one, even at lower conf.
    precomputed2 = [
        OCRResult(text="Flay", confidence=0.99,
                  rect=Rect(x=380, y=619, width=40, height=20)),
        OCRResult(text="Play", confidence=0.40,
                  rect=Rect(x=380, y=800, width=40, height=20)),   # centre (400,810)
    ]
    hit2 = sf._match_in_frame(_frame(12), "play", 0.35, (), precomputed=precomputed2)
    assert hit2 == (400, 810)   # the real 'Play' (tier 0) wins over 'Flay' (tier 3)


def test_match_in_frame_precomputed_does_no_ocr():
    """The critical fix: passing precomputed results must NOT trigger any OCR."""
    from core.models import OCRResult, Rect
    from tools.desktop.screen_finder import ScreenFinder

    ocr = _CountingOCR()
    sf = ScreenFinder(ocr=ocr)
    precomputed = [OCRResult(text="play", confidence=0.9,
                             rect=Rect(x=10, y=20, width=40, height=15))]
    hit = sf._match_in_frame(_frame(6), "play", 0.3, (), precomputed=precomputed)
    assert hit == (30, 27)                 # centre of the rect
    assert ocr.infer_calls == 0            # zero OCR inference


def test_match_in_frame_without_precomputed_uses_cache():
    """Without precomputed results, _match_in_frame uses the cache-first ROI path."""
    ocr = _CountingOCR(cache_ttl_seconds=8.0)
    from tools.desktop.screen_finder import ScreenFinder
    sf = ScreenFinder(ocr=ocr)
    frame = _frame(7)
    sf._match_in_frame(frame, "play", 0.3, ())
    sf._match_in_frame(frame.copy(), "play", 0.3, ())  # same frame -> cache hit
    assert ocr.infer_calls == 1


# ------------------- OCR backend selection (RapidOCR/ONNX) --------------- #

def test_resolve_backend_explicit_and_env(monkeypatch):
    from tools.ocr.paddle_engine import _resolve_backend
    assert _resolve_backend("paddle") == "paddle"
    assert _resolve_backend("rapid") == "rapid"
    monkeypatch.setenv("PCGAF_OCR_BACKEND", "paddle")
    assert _resolve_backend(None) == "paddle"


def test_resolve_backend_auto_prefers_rapid_when_present(monkeypatch):
    monkeypatch.delenv("PCGAF_OCR_BACKEND", raising=False)
    import importlib.util
    import tools.ocr.paddle_engine as pe
    # auto -> 'rapid' iff rapidocr_onnxruntime importable, else 'paddle'.
    expected = "rapid" if importlib.util.find_spec("rapidocr_onnxruntime") else "paddle"
    assert pe._resolve_backend(None) == expected


def test_get_shared_ocr_paddle_backend_is_paddle_type():
    from tools.ocr.paddle_engine import PaddleOCREngine, get_shared_ocr
    eng = get_shared_ocr(language="en", backend="paddle")
    assert isinstance(eng, PaddleOCREngine)


# ----------------------- motion-gated OCR bypass ------------------------- #

def test_launch_perceive_skips_ocr_on_low_motion(config, monkeypatch):
    """When motion is below the change threshold and we already have OCR text,
    perceive() must reuse the previous text and NOT call OCR again."""
    from agents.launch.launch_agent import LaunchAgent
    from core.models import VisualFeatures

    class _Stub:
        def __getattr__(self, _n):
            return lambda *a, **k: None

    agent = LaunchAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    # Seed a previous OCR text and a previous frame so the gate can engage.
    agent._prev_ocr_text = "minecraft for windows play"
    agent._prev_frame = _frame(1)
    agent._cfg["ocr_change_threshold"] = 0.006

    calls = {"ocr": 0}

    class _Finder:
        region = None

        class _OCR:
            def read_text_roi(self, *_a, **_k):
                calls["ocr"] += 1
                return []
        _ocr = _OCR()

        def screenshot(self):
            return (_frame(1), 0, 0)

    class _Vision:
        def analyse(self, _frame):
            class _H:
                crosshair_visible = False
            return _H()

    class _OpenCV:
        def extract_features(self, _f, _p):
            return VisualFeatures(motion_score=0.0)   # zero motion -> gate engages

    agent.finder = _Finder()
    agent.vision = _Vision()
    agent.opencv = _OpenCV()
    monkeypatch.setattr("agents.launch.launch_agent.classify_screen_state",
                        lambda _vf: ("unknown", 0.2))
    monkeypatch.setattr(agent, "_list_window_titles", lambda: ["XBOX"])
    monkeypatch.setattr(agent, "_point_finder_at", lambda _t: None)
    monkeypatch.setattr(agent, "_fuse_l1_l2", lambda _b: None)
    monkeypatch.setattr(agent, "_match_state", lambda _b: None)
    monkeypatch.setattr(agent, "_save_frame", lambda *a, **k: None)

    bundle = agent.perceive()
    assert calls["ocr"] == 0                       # OCR skipped
    assert bundle.ocr_text == "minecraft for windows play"


# ------------- launch-config timing knobs (perf regression) -------------- #

def test_game_card_has_post_click_settle(config):
    """game_card must declare a settle wait so the Xbox Play button isn't
    re-clicked before it goes live (the 8-cycle stall fix)."""
    import yaml
    from pathlib import Path
    d = yaml.safe_load(Path("skills/games/minecraft/xbox_launch.yaml")
                       .read_text(encoding="utf-8"))
    sts = {s["name"]: s for s in d["launch_states"]}
    assert int(sts["game_card"]["action"].get("post_click_ms", 0)) >= 2000


def test_poll_interval_reduced(config):
    """poll_interval_s should be small now that OCR is ~1-2s, not ~40s."""
    import yaml
    from pathlib import Path
    d = yaml.safe_load(Path("skills/games/minecraft/xbox_launch.yaml")
                       .read_text(encoding="utf-8"))
    assert float(d["launch_config"]["poll_interval_s"]) <= 0.75
