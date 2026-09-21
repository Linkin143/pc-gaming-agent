"""Fast screen capture using ``mss`` (references saved to disk, not into state)."""

from __future__ import annotations

import time
from pathlib import Path

import mss
import numpy as np
from PIL import Image

from core.exceptions import CaptureError
from core.logger import get_logger
from core.models import Rect, ScreenshotRef

logger = get_logger("capture")


class CaptureResult:
    """Holds a captured frame plus metadata."""

    def __init__(self, image: np.ndarray, region: Rect, timestamp: float) -> None:
        self.image = image
        self.region = region
        self.timestamp = timestamp
        self.ref: ScreenshotRef | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


class ScreenCapture:
    """Screen capture wrapper around ``mss``."""

    def __init__(self, screenshots_dir: Path) -> None:
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._sct = getattr(mss, "MSS", mss.mss)()

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # noqa: BLE001
            pass

    def _monitor(self) -> dict[str, int]:
        return self._sct.monitors[0]

    def capture_full(self) -> CaptureResult:
        mon = self._monitor()
        return self._grab(mon["left"], mon["top"], mon["width"], mon["height"])

    def capture_region(self, region: Rect) -> CaptureResult:
        return self._grab(region.x, region.y, region.width, region.height)

    @staticmethod
    def find_window_rect(title_re: str) -> Rect | None:
        """Return the absolute-pixel rect of the first window matching title_re.

        Uses win32 so capture can be scoped to a single game window instead of
        the whole (multi-monitor) desktop, which keeps OCR/OpenCV focused on the
        real game and avoids reading other windows.
        """
        try:
            import re

            import win32gui

            pattern = re.compile(title_re, re.IGNORECASE)
            found: list[Rect] = []

            def _cb(hwnd, _acc):  # noqa: ANN001 - win32 callback signature
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd) or ""
                if not pattern.search(title):
                    return
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                if right - left > 0 and bottom - top > 0:
                    found.append(Rect(x=int(left), y=int(top),
                                      width=int(right - left), height=int(bottom - top)))

            win32gui.EnumWindows(_cb, None)
            return found[0] if found else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("find_window_rect_failed", title_re=title_re, error=str(exc))
            return None

    def capture_window(self, title_re: str) -> CaptureResult:
        """Capture only the region of the window matching ``title_re``.

        Falls back to a full-desktop capture if the window cannot be located.
        """
        rect = self.find_window_rect(title_re)
        if rect is None:
            logger.debug("capture_window_fallback_full", title_re=title_re)
            return self.capture_full()
        return self._grab(rect.x, rect.y, rect.width, rect.height)

    def _grab(self, left: int, top: int, width: int, height: int) -> CaptureResult:
        bbox = {"left": left, "top": top, "width": width, "height": height}
        try:
            raw = self._sct.grab(bbox)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError("Screen capture failed.",
                               context={"bbox": bbox, "error": str(exc)}) from exc
        frame = np.ascontiguousarray(np.asarray(raw)[:, :, :3])
        region = Rect(x=left, y=top, width=width, height=height)
        return CaptureResult(image=frame, region=region, timestamp=time.time())

    def save(self, result: CaptureResult, run_id: str, tag: str = "frame") -> ScreenshotRef:
        run_dir = self.screenshots_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{tag}_{int(result.timestamp * 1000)}.png"
        try:
            Image.fromarray(result.image[:, :, ::-1]).save(path)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError("Failed to save screenshot.",
                               context={"path": str(path), "error": str(exc)}) from exc
        ref = ScreenshotRef(path=str(path), region=result.region,
                            width=result.width, height=result.height,
                            timestamp=result.timestamp)
        result.ref = ref
        logger.debug("screenshot_saved", path=str(path), tag=tag)
        return ref

    @staticmethod
    def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 1.0
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return float(diff.mean() / 255.0)
