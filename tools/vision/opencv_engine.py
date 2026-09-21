"""Deterministic OpenCV vision engine (change detection, ROI, template, contours)."""

from __future__ import annotations

import cv2
import numpy as np

from core.exceptions import VisionError
from core.logger import get_logger
from core.models import BoundingBox, Point, Rect, VisionResult, VisualFeatures

logger = get_logger("opencv")

# Coarse HSV colour bins for naming dominant colours (hue in OpenCV is 0-179).
_COLOR_BINS = [
    ("red", 0, 10), ("orange", 11, 24), ("yellow", 25, 34),
    ("green", 35, 85), ("cyan", 86, 100), ("blue", 101, 130),
    ("purple", 131, 155), ("pink", 156, 179),
]


class OpenCVEngine:
    """Wraps common OpenCV operations behind a structured interface."""

    def change_score(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        if frame_a is None or frame_b is None or frame_a.shape != frame_b.shape:
            return 1.0
        diff = cv2.absdiff(frame_a, frame_b)
        return float(diff.mean() / 255.0)

    def detect_change(self, frame_a: np.ndarray, frame_b: np.ndarray, threshold: float) -> VisionResult:
        score = self.change_score(frame_a, frame_b)
        return VisionResult(
            method="change_detection", change_score=min(score, 1.0), confidence=1.0,
            metadata={"changed": score >= threshold, "threshold": threshold},
        )

    def extract_roi(self, image: np.ndarray, region: Rect) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, min(region.x, w - 1))
        y1 = max(0, min(region.y, h - 1))
        x2 = max(0, min(region.x + region.width, w))
        y2 = max(0, min(region.y + region.height, h))
        if x2 <= x1 or y2 <= y1:
            raise VisionError("ROI is empty after clamping.", context={"region": region.as_tuple()})
        return image[y1:y2, x1:x2].copy()

    def preprocess(self, image: np.ndarray, *, grayscale: bool = True, denoise: bool = True) -> np.ndarray:
        out = image
        if grayscale and len(out.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if denoise:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        return out

    def template_match(self, image: np.ndarray, template: np.ndarray, *,
                       threshold: float = 0.8, max_matches: int = 10) -> VisionResult:
        try:
            img_gray = self.preprocess(image, denoise=False)
            tpl_gray = self.preprocess(template, denoise=False)
            result = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error as exc:  # pragma: no cover
            raise VisionError("Template match failed.", context={"error": str(exc)}) from exc
        th, tw = tpl_gray.shape[:2]
        ys, xs = np.where(result >= threshold)
        boxes: list[BoundingBox] = []
        for x, y in sorted(zip(xs, ys), key=lambda p: -result[p[1], p[0]]):
            rect = Rect(x=int(x), y=int(y), width=int(tw), height=int(th))
            if self._overlaps_existing(rect, boxes):
                continue
            boxes.append(BoundingBox(rect=rect, label="template", confidence=float(result[y, x])))
            if len(boxes) >= max_matches:
                break
        return VisionResult(method="template_match", matches=boxes,
                            confidence=boxes[0].confidence if boxes else 0.0,
                            metadata={"threshold": threshold, "count": len(boxes)})

    @staticmethod
    def _overlaps_existing(rect: Rect, boxes: list[BoundingBox], iou: float = 0.3) -> bool:
        for b in boxes:
            bx = b.rect
            ix1, iy1 = max(rect.x, bx.x), max(rect.y, bx.y)
            ix2 = min(rect.x + rect.width, bx.x + bx.width)
            iy2 = min(rect.y + rect.height, bx.y + bx.height)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            union = rect.width * rect.height + bx.width * bx.height - inter
            if union > 0 and inter / union >= iou:
                return True
        return False

    def detect_contours(self, image: np.ndarray, *, min_area: int = 100,
                        max_results: int = 25) -> VisionResult:
        gray = self.preprocess(image)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[BoundingBox] = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append(BoundingBox(rect=Rect(x=int(x), y=int(y), width=int(w), height=int(h)),
                                     label="contour", confidence=1.0))
            if len(boxes) >= max_results:
                break
        return VisionResult(method="contours", matches=boxes,
                            confidence=1.0 if boxes else 0.0, metadata={"count": len(boxes)})

    @staticmethod
    def estimate_position(box: BoundingBox) -> Point:
        return box.rect.center

    # ------------------------------------------------------------------ #
    # Game-agnostic visual feature extraction (the pixel-level truth)
    # ------------------------------------------------------------------ #
    def extract_features(self, frame: np.ndarray,
                         prev_frame: np.ndarray | None = None) -> VisualFeatures:
        """Measure game-agnostic visual features from a real game frame.

        Every value is a direct pixel measurement - never a guess - so the
        planner can treat them as ground truth.
        """
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            # Edge density: fraction of pixels that are edges (structure/UI).
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / float(edges.size)

            # Brightness: mean luminance normalised to 0-1.
            brightness_mean = float(gray.mean()) / 255.0

            # Motion: normalised frame difference vs the previous frame.
            motion_score = 0.0
            if prev_frame is not None and prev_frame.shape == frame.shape:
                motion_score = float(cv2.absdiff(frame, prev_frame).mean()) / 255.0

            # Text-region density: small high-contrast horizontal blobs approximate
            # text. Use morphological gradient + threshold.
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            _, text_bin = cv2.threshold(grad, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_region_density = float(np.count_nonzero(text_bin)) / float(text_bin.size)

            # Center complexity: edge density in the central 50% (gameplay tends
            # to be busy in the centre; menus tend to be sparser/centered text).
            cy0, cy1 = int(h * 0.25), int(h * 0.75)
            cx0, cx1 = int(w * 0.25), int(w * 0.75)
            center_edges = edges[cy0:cy1, cx0:cx1]
            center_complexity = (float(np.count_nonzero(center_edges))
                                 / float(center_edges.size)) if center_edges.size else 0.0

            # UI element count: rectangular contours of a plausible button size.
            ui_count = self._count_ui_rects(edges, w, h)

            # Dominant colours: top HSV hue bins by pixel mass.
            colors = self._dominant_colors(frame)

            return VisualFeatures(
                edge_density=min(edge_density, 1.0),
                brightness_mean=min(brightness_mean, 1.0),
                motion_score=min(motion_score, 1.0),
                text_region_density=min(text_region_density, 1.0),
                center_complexity=min(center_complexity, 1.0),
                ui_element_count=ui_count,
                dominant_colors=colors,
            )
        except Exception as exc:  # noqa: BLE001 - features must never crash the loop
            logger.warning("feature_extraction_failed", error=str(exc))
            return VisualFeatures()

    @staticmethod
    def _count_ui_rects(edges: np.ndarray, w: int, h: int) -> int:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        min_area = (w * h) * 0.002
        max_area = (w * h) * 0.5
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(ch) if ch else 0.0
            # Button/panel-like: wider than tall, not a thin line.
            if 1.2 <= aspect <= 12.0 and ch > 8:
                count += 1
        return count

    @staticmethod
    def _dominant_colors(frame: np.ndarray, top_k: int = 3) -> list[str]:
        if frame.ndim != 3:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]
        total = h_chan.size
        # Achromatic pixels (low saturation) are dark/white/grey by brightness.
        achroma = s_chan < 40
        counts: dict[str, int] = {}
        dark = int(np.count_nonzero(achroma & (v_chan < 60)))
        light = int(np.count_nonzero(achroma & (v_chan >= 180)))
        grey = int(np.count_nonzero(achroma)) - dark - light
        if dark:
            counts["dark"] = dark
        if light:
            counts["white"] = light
        if grey > 0:
            counts["grey"] = grey
        chroma = ~achroma
        for name, lo, hi in _COLOR_BINS:
            mask = chroma & (h_chan >= lo) & (h_chan <= hi)
            c = int(np.count_nonzero(mask))
            if c > total * 0.02:
                counts[name] = counts.get(name, 0) + c
        return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]]
