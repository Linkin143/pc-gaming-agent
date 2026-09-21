"""Vision agent - a thin agent-level wrapper around the OpenCV engine."""

from __future__ import annotations

import numpy as np

from core.models import Rect, VisionResult
from tools.vision.opencv_engine import OpenCVEngine


class VisionAgent:
    def __init__(self, engine: OpenCVEngine | None = None) -> None:
        self.engine = engine or OpenCVEngine()

    def change(self, prev: np.ndarray, curr: np.ndarray, threshold: float) -> VisionResult:
        return self.engine.detect_change(prev, curr, threshold)

    def contours(self, image: np.ndarray, min_area: int = 100) -> VisionResult:
        return self.engine.detect_contours(image, min_area=min_area)

    def template(self, image: np.ndarray, template: np.ndarray, threshold: float = 0.8) -> VisionResult:
        return self.engine.template_match(image, template, threshold=threshold)

    def roi(self, image: np.ndarray, region: Rect) -> np.ndarray:
        return self.engine.extract_roi(image, region)
