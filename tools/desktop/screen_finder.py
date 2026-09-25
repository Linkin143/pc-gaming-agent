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
                        exclude: tuple[str, ...],
                        precomputed: list[OCRResult] | None = None,
                        ) -> tuple[int, int] | None:
        q = query.lower()
        best: OCRResult | None = None
        try:
            # Reuse OCR results already computed this cycle when provided - this is
            # the key fix that eliminates the second ~40s OCR scan that used to run
            # inside every click_text/find_text. Otherwise fall back to a
            # cache-first ROI scan (use_cache=True) so an unchanged frame is free.
            if precomputed is not None:
                results = precomputed
            else:
                results = self._ocr.read_text_roi(frame, use_cache=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ocr_failed_in_find", error=str(exc))
            return None
        # Compare with spaces removed on BOTH sides. RapidOCR frequently renders
        # tile labels without spaces (e.g. "Minecraft for Windows" -> the single
        # token "MinecraftforWindows"), so a spaced query like "minecraft for
        # windows" would never substring-match and the search would wrongly fall
        # through to the next target ("minecraft launcher"). Normalising removes
        # that whitespace ambiguity without loosening which tile is selected
        # ("minecraftforwindows" still never matches "minecraftlauncher").
        def _norm(s: str) -> str:
            return "".join(s.lower().split())

        def _edit_distance(a: str, b: str, cap: int = 2) -> int:
            """Levenshtein distance, short-circuited once it exceeds ``cap``.
            Used only for the fuzzy button-label fallback below, so a cheap
            O(len(a)*len(b)) DP is plenty (labels are a handful of chars)."""
            if abs(len(a) - len(b)) > cap:
                return cap + 1
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                row_min = i
                for j, cb in enumerate(b, 1):
                    cost = 0 if ca == cb else 1
                    val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                    cur.append(val)
                    row_min = min(row_min, val)
                if row_min > cap:      # whole row already past the cap
                    return cap + 1
                prev = cur
            return prev[-1]

        q_norm = _norm(query)
        q_toklen = len(query.split())

        # A clickable button label is a SHORT, near-standalone text element. A
        # paragraph that merely *contains* the query word (e.g. the legal line
        # "...while you play.") must never win over the real "Play" button, even
        # though OCR gives the paragraph higher confidence. So we rank candidates:
        #   tier 0 (best): exact normalised match  (text == query)
        #   tier 1       : query is a whole-word run in a SHORT label
        #                  (label has <= query_words + 2 tokens)
        #   tier 2       : substring inside a longer block of text
        #   tier 3 (worst): FUZZY match on a short label - tolerates a 1-char OCR
        #                   misread on a standalone button (e.g. Minecraft's title
        #                   "Play" is read as "Flay", "Elay"...). Only applies when
        #                   NO exact/substring match exists, and only to short
        #                   labels (<= query chars + 2), so it never loosens which
        #                   tile is picked on the busy search-results page.
        # Within a tier, higher OCR confidence wins.
        best_key: tuple[int, float] | None = None
        # A conservative edit-distance budget: single-word queries allow 1 typo,
        # longer targets scale up slightly. Never fuzzy-match very short queries
        # (<=2 chars) where one edit could match unrelated words.
        fuzz_cap = 0 if len(q_norm) <= 2 else (1 if len(q_norm) <= 5 else 2)
        for r in results:
            text = r.text.lower()
            text_norm = _norm(r.text)
            if r.confidence < min_confidence or r.rect is None:
                continue
            if any(_norm(x) in text_norm or x in text for x in exclude):
                continue
            toklen = len(r.text.split())
            if q_norm not in text_norm:
                # No exact/substring hit. Try a fuzzy match, but ONLY on short,
                # near-standalone labels so we don't fuzz-match inside paragraphs.
                if (fuzz_cap > 0 and toklen <= q_toklen + 1
                        and abs(len(text_norm) - len(q_norm)) <= fuzz_cap
                        and _edit_distance(text_norm, q_norm, fuzz_cap) <= fuzz_cap):
                    tier = 3
                else:
                    continue
            elif text_norm == q_norm:
                tier = 0
            elif toklen <= q_toklen + 2:
                tier = 1
            else:
                tier = 2
            # Lower tier is better; higher confidence breaks ties. Encode as a
            # sort key where bigger = better: (-tier, confidence).
            key = (-tier, r.confidence)
            if best_key is None or key > best_key:
                best_key = key
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
