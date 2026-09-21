"""OCR agent - a thin agent-level wrapper around the PaddleOCR engine."""

from __future__ import annotations

import numpy as np

from core.models import OCRResult, Rect
from tools.ocr.paddle_engine import PaddleOCREngine


class OCRAgent:
    def __init__(self, engine: PaddleOCREngine | None = None, language: str = "en") -> None:
        self.engine = engine or PaddleOCREngine(language=language)

    def read(self, image: np.ndarray) -> list[OCRResult]:
        return self.engine.read_text(image)

    def read_region(self, image: np.ndarray, region: Rect) -> list[OCRResult]:
        return self.engine.read_region(image, region)

    def find(self, image: np.ndarray, query: str, threshold: float = 0.0) -> OCRResult | None:
        return self.engine.find_text(image, query, threshold=threshold)

    def mean_confidence(self, results: list[OCRResult]) -> float:
        if not results:
            return 0.0
        return float(sum(r.confidence for r in results) / len(results))
