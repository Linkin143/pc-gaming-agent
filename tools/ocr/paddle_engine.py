"""PaddleOCR text-extraction engine (supports v3.x predict + legacy ocr)."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from core.exceptions import OCRError
from core.logger import get_logger
from core.models import OCRResult, Rect

logger = get_logger("ocr")

# Process-wide shared engines keyed by language, so the heavy PaddleOCR model
# pipeline is initialised only once even when several components (launch flow +
# closed-loop engine) each ask for an OCR engine.
_SHARED: dict[str, "PaddleOCREngine"] = {}


def get_shared_ocr(language: str = "en", cache_ttl_seconds: float = 2.0) -> "PaddleOCREngine":
    """Return a shared PaddleOCREngine for ``language`` (created once)."""
    eng = _SHARED.get(language)
    if eng is None:
        eng = PaddleOCREngine(language=language, cache_ttl_seconds=cache_ttl_seconds)
        _SHARED[language] = eng
    return eng


class PaddleOCREngine:
    """Lazy, thread-safe wrapper around PaddleOCR supporting v3.x and legacy."""

    def __init__(self, language: str = "en", cache_ttl_seconds: float = 2.0) -> None:
        self.language = language
        self.cache_ttl = cache_ttl_seconds
        self._engine: Any | None = None
        self._lock = threading.Lock()
        self._cache: tuple[int, float, list[OCRResult]] | None = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from paddleocr import PaddleOCR
            except Exception as exc:  # noqa: BLE001
                raise OCRError("PaddleOCR is not importable.", context={"error": str(exc)}) from exc
            try:
                self._engine = PaddleOCR(use_textline_orientation=True, lang=self.language)
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.language)
            logger.info("paddleocr_initialised", language=self.language)
            return self._engine

    def warmup(self) -> None:
        self._ensure_engine()

    def read_text(self, image: np.ndarray, *, use_cache: bool = True) -> list[OCRResult]:
        if use_cache and self._cache is not None:
            key, ts, cached = self._cache
            if key == self._image_key(image) and (time.time() - ts) < self.cache_ttl:
                return cached
        engine = self._ensure_engine()
        try:
            results = self._run(engine, image)
        except OCRError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRError("OCR inference failed.", context={"error": str(exc)}) from exc
        if use_cache:
            self._cache = (self._image_key(image), time.time(), results)
        return results

    def read_region(self, image: np.ndarray, region: Rect) -> list[OCRResult]:
        h, w = image.shape[:2]
        x1 = max(0, min(region.x, w - 1))
        y1 = max(0, min(region.y, h - 1))
        x2 = max(0, min(region.x + region.width, w))
        y2 = max(0, min(region.y + region.height, h))
        results = self.read_text(image[y1:y2, x1:x2], use_cache=False)
        for r in results:
            if r.rect is not None:
                r.rect = Rect(x=r.rect.x + x1, y=r.rect.y + y1,
                              width=r.rect.width, height=r.rect.height)
        return results

    def find_text(self, image: np.ndarray, query: str, *, threshold: float = 0.0) -> OCRResult | None:
        q = query.lower()
        for r in self.read_text(image):
            if r.confidence >= threshold and q in r.text.lower():
                return r
        return None

    def _run(self, engine: Any, image: np.ndarray) -> list[OCRResult]:
        if hasattr(engine, "predict"):
            return self._parse_predict(engine.predict(image))
        return self._parse_legacy(engine.ocr(image))

    def _parse_predict(self, raw: Any) -> list[OCRResult]:
        out: list[OCRResult] = []
        if not raw:
            return out
        for page in raw:
            data = page.json if hasattr(page, "json") else page
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            if not isinstance(data, dict):
                continue
            texts = data.get("rec_texts") or data.get("rec_text") or []
            scores = data.get("rec_scores") or data.get("rec_score") or []
            polys = data.get("rec_polys") or data.get("dt_polys") or []
            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 0.0
                rect = self._poly_to_rect(polys[i]) if i < len(polys) else None
                out.append(OCRResult(text=str(text), confidence=conf, rect=rect))
        return out

    def _parse_legacy(self, raw: Any) -> list[OCRResult]:
        out: list[OCRResult] = []
        if not raw:
            return out
        pages = raw if isinstance(raw, list) else [raw]
        for page in pages:
            if not page:
                continue
            for line in page:
                try:
                    box, (text, score) = line[0], line[1]
                except (ValueError, TypeError, IndexError):
                    continue
                out.append(OCRResult(text=str(text), confidence=float(score),
                                     rect=self._poly_to_rect(box)))
        return out

    @staticmethod
    def _poly_to_rect(poly: Any) -> Rect | None:
        try:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            return Rect(x=int(max(0, x_min)), y=int(max(0, y_min)),
                        width=max(1, int(round(x_max - x_min))),
                        height=max(1, int(round(y_max - y_min))))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _image_key(image: np.ndarray) -> int:
        return hash(image[::16, ::16].tobytes())
