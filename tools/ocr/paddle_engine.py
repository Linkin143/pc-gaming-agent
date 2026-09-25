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

# Process-wide shared engines keyed by (backend, language), so the model pipeline
# is initialised only once even when several components (launch flow + closed-loop
# engine) each ask for an OCR engine.
_SHARED: dict[str, "PaddleOCREngine"] = {}


def _resolve_backend(backend: str | None) -> str:
    """Pick the OCR backend: explicit arg > PCGAF_OCR_BACKEND env > auto.

    'auto' prefers the fast RapidOCR (ONNX) engine when installed and falls back
    to PaddleOCR otherwise, so the fast path is used by default without breaking
    environments that only have PaddlePaddle.
    """
    import importlib.util
    import os

    choice = (backend or os.environ.get("PCGAF_OCR_BACKEND") or "auto").lower()
    if choice == "auto":
        if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
            return "rapid"
        return "paddle"
    return choice


def get_shared_ocr(language: str = "en", cache_ttl_seconds: float = 2.0,
                   backend: str | None = None) -> "PaddleOCREngine":
    """Return a shared OCR engine for ``language`` (created once per backend).

    ``backend``: 'rapid' (RapidOCR ONNX, fast), 'paddle' (PaddleOCR), or None/'auto'
    to auto-select the fastest available. Both engines expose an identical API and
    OCRResult contract, so callers are unaffected by which one is chosen.
    """
    resolved = _resolve_backend(backend)
    key = f"{resolved}:{language}"
    eng = _SHARED.get(key)
    if eng is None:
        if resolved == "rapid":
            try:
                from tools.ocr.rapid_engine import RapidOCREngine
                eng = RapidOCREngine(language=language,
                                     cache_ttl_seconds=cache_ttl_seconds)
            except Exception as exc:  # noqa: BLE001 - fall back to paddle
                logger.warning("rapid_ocr_unavailable_fallback_paddle", error=str(exc))
                eng = PaddleOCREngine(language=language,
                                      cache_ttl_seconds=cache_ttl_seconds)
        else:
            eng = PaddleOCREngine(language=language, cache_ttl_seconds=cache_ttl_seconds)
        _SHARED[key] = eng
        logger.info("ocr_backend_selected", backend=resolved, language=language)
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
                self._engine = PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False, use_doc_unwarping=False, lang=self.language)
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

    # Interactive-UI ROI fractions. Xbox/Minecraft buttons, tiles and labels all
    # live inside the central band; skipping the outer margins cuts the pixel
    # count to ~the centre 70% of the frame with no accuracy loss on UI text.
    # (Search bar, result tiles, game-card Play, and the Minecraft menu buttons
    # are all comfortably inside these bounds at 1080p.)
    _ROI_TOP = 0.12
    _ROI_BOTTOM = 0.90
    _ROI_LEFT = 0.08
    _ROI_RIGHT = 0.92

    def _crop_roi(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Crop to the interactive UI region; returns (roi, x_offset, y_offset)."""
        h, w = image.shape[:2]
        x0, x1 = int(w * self._ROI_LEFT), int(w * self._ROI_RIGHT)
        y0, y1 = int(h * self._ROI_TOP), int(h * self._ROI_BOTTOM)
        if x1 <= x0 or y1 <= y0:
            return image, 0, 0
        return image[y0:y1, x0:x1], x0, y0

    def read_text_roi(self, image: np.ndarray, *, use_cache: bool = True,
                      adjust_coords: bool = True) -> list[OCRResult]:
        """OCR only the interactive UI ROI, translating bbox coords back to the
        full-frame origin. Processes far fewer pixels than the full frame while
        preserving accuracy on buttons/labels. API-compatible with read_text
        (returns the same list[OCRResult] with full-frame coordinates)."""
        roi, ox, oy = self._crop_roi(image)
        results = self.read_text(roi, use_cache=use_cache)
        if adjust_coords and (ox or oy):
            # IMPORTANT: do NOT mutate the result objects in place. On a cache
            # HIT, read_text() returns the SAME cached list/objects, whose rects
            # are stored in ROI-local coordinates. Mutating them (r.rect = ...)
            # would add the ROI offset again on every reuse, so a button read at
            # (968,578) drifts to (1123,701) on the 2nd call, (1278,825) on the
            # 3rd, ... - the agent then clicks empty space next to the real
            # button (this caused the mc_title "clicks Marketplace" stall).
            # Build NEW objects instead, leaving the cached ROI-local ones intact.
            translated: list[OCRResult] = []
            for r in results:
                if r.rect is not None:
                    new_rect = Rect(x=r.rect.x + ox, y=r.rect.y + oy,
                                    width=r.rect.width, height=r.rect.height)
                    translated.append(OCRResult(text=r.text, confidence=r.confidence,
                                                rect=new_rect))
                else:
                    translated.append(r)
            return translated
        return results

    def read_region(self, image: np.ndarray, region: Rect) -> list[OCRResult]:
        h, w = image.shape[:2]
        x1 = max(0, min(region.x, w - 1))
        y1 = max(0, min(region.y, h - 1))
        x2 = max(0, min(region.x + region.width, w))
        y2 = max(0, min(region.y + region.height, h))
        results = self.read_text(image[y1:y2, x1:x2], use_cache=False)
        # Build new objects rather than mutating in place (see read_text_roi):
        # read_text may return cached objects, and mutating their rects would
        # corrupt the cache with repeated offset additions.
        translated: list[OCRResult] = []
        for r in results:
            if r.rect is not None:
                new_rect = Rect(x=r.rect.x + x1, y=r.rect.y + y1,
                                width=r.rect.width, height=r.rect.height)
                translated.append(OCRResult(text=r.text, confidence=r.confidence,
                                            rect=new_rect))
            else:
                translated.append(r)
        return translated

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
        # A 16x9 nearest-neighbour thumbnail (432 bytes) uniquely fingerprints a
        # frame far faster than hashing a strided view of a 1920x1080 image. Two
        # visually-identical frames hash equal, so the cache hits across a stable
        # screen; any real change flips enough thumbnail pixels to miss.
        try:
            import cv2

            thumb = cv2.resize(image, (16, 9), interpolation=cv2.INTER_NEAREST)
            return hash(thumb.tobytes())
        except Exception:  # noqa: BLE001 - fall back to the strided hash
            return hash(image[::32, ::32].tobytes())
