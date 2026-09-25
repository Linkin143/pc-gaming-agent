"""Lightweight ONNX-Runtime OCR engine (RapidOCR / PP-OCRv4-mobile ONNX).

This is the fast UI-text OCR the architecture calls for: no PaddlePaddle, no
document-orientation / dewarping / text-line-orientation models, no PP-OCRv6
full-screen pipeline. On CPU it reads clean game-UI text in ~1-2s versus the
~30-45s of the Paddle "medium" pipeline, at equal or better accuracy on the
upright, high-contrast text of the Xbox app and Minecraft menus.

It reuses PaddleOCREngine's ROI cropping, thumbnail-hash cache, and OCRResult
contract by subclassing it and overriding only the model init + inference, so
every existing caller (ScreenFinder, LaunchAgent, PerceptionAgent) works
unchanged - read_text / read_text_roi / read_region / find_text / warmup are all
inherited and behave identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.exceptions import OCRError
from core.logger import get_logger
from core.models import OCRResult, Rect
from tools.ocr.paddle_engine import PaddleOCREngine

logger = get_logger("ocr.rapid")


class RapidOCREngine(PaddleOCREngine):
    """RapidOCR (ONNX Runtime) drop-in with the PaddleOCREngine interface."""

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from rapidocr_onnxruntime import RapidOCR
            except Exception as exc:  # noqa: BLE001
                raise OCRError("rapidocr-onnxruntime is not importable.",
                               context={"error": str(exc)}) from exc
            # Defaults are the PP-OCRv4-mobile ONNX det+rec models - small, CPU
            # friendly, English-capable. No angle/doc/unwarp stages are loaded.
            self._engine = RapidOCR()
            logger.info("rapidocr_initialised", language=self.language)
            return self._engine

    def _run(self, engine: Any, image: np.ndarray) -> list[OCRResult]:
        # RapidOCR returns (result, elapse) where result is a list of
        # [box(4x2), text, score] or None when nothing is detected.
        out: list[OCRResult] = []
        result, _ = engine(image)
        if not result:
            return out
        for item in result:
            try:
                box, text, score = item[0], item[1], item[2]
            except (ValueError, TypeError, IndexError):
                continue
            out.append(OCRResult(text=str(text), confidence=float(score),
                                 rect=self._poly_to_rect(box)))
        return out
