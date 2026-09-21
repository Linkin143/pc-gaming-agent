"""OCR + OpenCV + keyboard/mouse (KBM) screen interaction utility.

This is the accuracy layer for targeted actions: it captures a specific window
region, locates on-screen text via PaddleOCR (converting bbox pixel offsets to
absolute screen coordinates), and drives clicks/typing through ``pynput`` so
input works in games and hardware-accelerated apps that ignore UIA invokes.

Coordinate model
----------------
We capture only the target window's rectangle (``win32gui.GetWindowRect``) so
every OCR bounding box is an offset from that rectangle's origin. Adding the
window's ``left``/``top`` yields absolute screen coordinates, correct even on
multi-monitor setups whose virtual-desktop origin is negative.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.logger import get_logger
from core.models import OCRResult, Rect
from tools.ocr.paddle_engine import PaddleOCREngine, get_shared_ocr

logger = get_logger("screen_finder")


class ScreenFinder:
    """Locate text on screen and act on it with real keyboard/mouse events."""

    def __init__(self, ocr: PaddleOCREngine | None = None) -> None:
        # Reuse the process-wide shared OCR so PaddleOCR loads only once.
        self._ocr = ocr or get_shared_ocr()
        self._kb: Any | None = None
        self._mouse: Any | None = None
        self.region: Rect | None = None

    # -- lazy input controllers ----------------------------------------- #
    def _keyboard(self) -> Any:
        if self._kb is None:
            from pynput.keyboard import Controller
            self._kb = Controller()
        return self._kb

    def _mouse_ctl(self) -> Any:
        if self._mouse is None:
            from pynput.mouse import Controller
            self._mouse = Controller()
        return self._mouse

    # -- window region resolution --------------------------------------- #
    def set_region_from_window(self, window: Any) -> Rect | None:
        rect = self._window_rect(window)
        if rect is not None:
            self.region = rect
            logger.debug("region_set", rect=rect.as_tuple())
        return rect

    @staticmethod
    def _window_rect(window: Any) -> Rect | None:
        try:
            r = window.rectangle()
            return Rect(x=int(r.left), y=int(r.top),
                        width=max(1, int(r.right - r.left)),
                        height=max(1, int(r.bottom - r.top)))
        except Exception:  # noqa: BLE001
            pass
        try:
            import win32gui

            hwnd = window if isinstance(window, int) else window.handle
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return Rect(x=int(left), y=int(top),
                        width=max(1, int(right - left)),
                        height=max(1, int(bottom - top)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("window_rect_failed", error=str(exc))
            return None

    # -- capture --------------------------------------------------------- #
    def _grab(self, region: Rect | None) -> tuple[np.ndarray, int, int]:
        import mss

        with (getattr(mss, "MSS", mss.mss)()) as sct:
            if region is not None:
                bbox = {"left": region.x, "top": region.y,
                        "width": region.width, "height": region.height}
                origin_x, origin_y = region.x, region.y
            else:
                mon = sct.monitors[0]
                bbox = {"left": mon["left"], "top": mon["top"],
                        "width": mon["width"], "height": mon["height"]}
                origin_x, origin_y = mon["left"], mon["top"]
            raw = sct.grab(bbox)
        frame = np.ascontiguousarray(np.asarray(raw)[:, :, :3])
        return frame, origin_x, origin_y

    def screenshot(self) -> tuple[np.ndarray, int, int]:
        return self._grab(self.region)

    # -- OCR search ------------------------------------------------------ #
    def find_text(self, query: str, *, min_confidence: float = 0.35,
                  exclude: tuple[str, ...] = ()) -> tuple[int, int] | None:
        """Return absolute (x, y) centre of the best OCR line matching query."""
        frame, ox, oy = self.screenshot()
        hit = self._match_in_frame(frame, query, min_confidence, exclude)
        if hit is None:
            return None
        return ox + hit[0], oy + hit[1]

    def _match_in_frame(self, frame: np.ndarray, query: str, min_confidence: float,
                        exclude: tuple[str, ...]) -> tuple[int, int] | None:
        q = query.lower()
        best: OCRResult | None = None
        try:
            results = self._ocr.read_text(frame, use_cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ocr_failed_in_find", error=str(exc))
            return None
        for r in results:
            text = r.text.lower()
            if r.confidence < min_confidence or r.rect is None:
                continue
            if any(x in text for x in exclude):
                continue
            if q in text and (best is None or r.confidence > best.confidence):
                best = r
        if best is None or best.rect is None:
            return None
        return best.rect.center.x, best.rect.center.y

    def wait_for_text(self, query: str, *, timeout_s: float = 15.0,
                      poll_s: float = 1.0, min_confidence: float = 0.35,
                      exclude: tuple[str, ...] = ()) -> tuple[int, int] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            hit = self.find_text(query, min_confidence=min_confidence, exclude=exclude)
            if hit is not None:
                logger.info("text_found", query=query, at=hit)
                return hit
            time.sleep(poll_s)
        logger.warning("text_not_found", query=query, timeout_s=timeout_s)
        return None

    # -- actions (real KBM events via pynput) --------------------------- #
    def click_at(self, x: int, y: int, *, settle_s: float = 0.4) -> None:
        from pynput.mouse import Button

        mouse = self._mouse_ctl()
        mouse.position = (int(x), int(y))
        time.sleep(0.15)
        mouse.click(Button.left, 1)
        logger.info("clicked", x=int(x), y=int(y))
        time.sleep(settle_s)

    def click_text(self, query: str, *, timeout_s: float = 15.0,
                   min_confidence: float = 0.35,
                   exclude: tuple[str, ...] = ()) -> bool:
        hit = self.wait_for_text(query, timeout_s=timeout_s,
                                 min_confidence=min_confidence, exclude=exclude)
        if hit is None:
            return False
        self.click_at(*hit)
        return True

    def type_text(self, text: str, *, clear_first: bool = True) -> None:
        from pynput.keyboard import Key

        kb = self._keyboard()
        if clear_first:
            kb.press(Key.ctrl); kb.press("a")
            kb.release("a"); kb.release(Key.ctrl)
            time.sleep(0.1)
            kb.press(Key.backspace); kb.release(Key.backspace)
            time.sleep(0.1)
        kb.type(text)
        logger.info("typed_text", text=text)

    def press_enter(self) -> None:
        self.press_key("enter")

    def press_key(self, key_name: str) -> None:
        from pynput.keyboard import Key, KeyCode

        specials = {
            "enter": Key.enter, "esc": Key.esc, "escape": Key.esc,
            "tab": Key.tab, "space": Key.space, "backspace": Key.backspace,
            "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
        }
        kb = self._keyboard()
        key = specials.get(key_name.lower(),
                           KeyCode.from_char(key_name) if len(key_name) == 1 else None)
        if key is None:
            logger.warning("unknown_key", key=key_name)
            return
        kb.press(key)
        time.sleep(0.05)
        kb.release(key)
        logger.info("pressed_key", key=key_name)
