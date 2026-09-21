"""Live gaming tests.

These require a real Windows desktop, the Xbox app, and (for gameplay) a running
game. They are marked ``live`` and skipped by default. Run explicitly with::

    pytest -m live tests/gaming
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


@pytest.mark.live
def test_screen_capture_produces_frame(config):
    from tools.capture.screen_capture import ScreenCapture

    cap = ScreenCapture(config.screenshots_dir)
    try:
        result = cap.capture_full()
        assert result.width > 0 and result.height > 0
        assert result.image.ndim == 3
    finally:
        cap.close()


@pytest.mark.live
def test_xbox_detection():
    from tools.desktop.winapp import XboxDesktopAutomation

    xbox = XboxDesktopAutomation()
    # Only asserts the API is callable; running state depends on the machine.
    assert isinstance(xbox.is_xbox_running(), bool)


@pytest.mark.live
def test_foreground_title_readable():
    from tools.input.safety import get_foreground_title

    assert isinstance(get_foreground_title(), str)


@pytest.mark.live
def test_ocr_reads_synthetic_text(config):
    import cv2
    import numpy as np

    from tools.ocr.paddle_engine import PaddleOCREngine

    img = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "ELECTRICAL", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    engine = PaddleOCREngine()
    results = engine.read_text(img)
    joined = " ".join(r.text.upper() for r in results)
    assert "ELECTRICAL" in joined
