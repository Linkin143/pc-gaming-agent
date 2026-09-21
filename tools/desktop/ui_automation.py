"""Windows UI Automation (UIA3) inspection via ``pywinauto``."""

from __future__ import annotations

from typing import Any

from core.exceptions import UIElementNotFoundError, WindowNotFoundError
from core.logger import get_logger
from core.models import Rect, UIElement

logger = get_logger("uia")


class UIInspector:
    """Thin wrapper over pywinauto's UIA3 backend."""

    def __init__(self) -> None:
        self._desktop: Any | None = None

    def _get_desktop(self) -> Any:
        if self._desktop is None:
            from pywinauto import Desktop
            self._desktop = Desktop(backend="uia")
        return self._desktop

    def get_foreground_window(self) -> UIElement:
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            return UIElement(name=title, control_type="Window",
                             rect=Rect(x=rect[0], y=rect[1],
                                       width=max(1, rect[2] - rect[0]),
                                       height=max(1, rect[3] - rect[1])))
        except Exception as exc:  # noqa: BLE001
            raise WindowNotFoundError("Failed to read foreground window.",
                                      context={"error": str(exc)}) from exc

    def list_windows(self, title_re: str) -> list[Any]:
        """Return ALL top-level windows whose title matches ``title_re``.

        Uses ``windows()`` (plural) so it never raises on multiple matches -
        unlike ``window()`` which is ambiguous when >1 window matches.
        """
        import re

        pattern = re.compile(title_re, re.IGNORECASE)
        matches: list[Any] = []
        try:
            for w in self._get_desktop().windows():
                try:
                    if pattern.search(w.window_text() or ""):
                        matches.append(w)
                except Exception:  # noqa: BLE001 - some handles are transient
                    continue
        except Exception as exc:  # noqa: BLE001
            raise WindowNotFoundError("Failed to enumerate windows.",
                                      context={"title_re": title_re, "error": str(exc)}) from exc
        return matches

    def find_window(self, title_re: str) -> Any:
        """Return the BEST top-level window matching ``title_re``.

        Resilient to multiple matches: prefers the first visible + enabled
        window, falling back to the first match. Raises only when none match.
        """
        candidates = self.list_windows(title_re)
        if not candidates:
            raise WindowNotFoundError("Window not found.", context={"title_re": title_re})
        for w in candidates:
            try:
                if w.is_visible() and w.is_enabled():
                    return w
            except Exception:  # noqa: BLE001
                continue
        return candidates[0]

    def get_ui_tree(self, title_re: str, *, max_depth: int = 4) -> list[UIElement]:
        window = self.find_window(title_re)
        elements: list[UIElement] = []
        self._walk(window, elements, depth=0, max_depth=max_depth)
        logger.debug("ui_tree_read", title_re=title_re, count=len(elements))
        return elements

    def _walk(self, node: Any, out: list[UIElement], *, depth: int, max_depth: int) -> None:
        if depth > max_depth:
            return
        try:
            out.append(self._to_element(node))
            for child in node.children():
                self._walk(child, out, depth=depth + 1, max_depth=max_depth)
        except Exception:  # noqa: BLE001
            return

    def _to_element(self, node: Any) -> UIElement:
        info = node.element_info
        rect_obj = getattr(info, "rectangle", None)
        rect = None
        if rect_obj is not None:
            try:
                rect = Rect(x=int(rect_obj.left), y=int(rect_obj.top),
                            width=max(1, int(rect_obj.right - rect_obj.left)),
                            height=max(1, int(rect_obj.bottom - rect_obj.top)))
            except Exception:  # noqa: BLE001
                rect = None
        return UIElement(name=info.name or "",
                         control_type=getattr(info, "control_type", "") or "",
                         automation_id=getattr(info, "automation_id", "") or "",
                         class_name=getattr(info, "class_name", "") or "", rect=rect)

    def find_element(self, title_re: str, *, name: str,
                     control_type: str | None = None) -> UIElement:
        for el in self.get_ui_tree(title_re):
            if el.name == name and (control_type is None or el.control_type == control_type):
                return el
        raise UIElementNotFoundError("UI element not found.",
                                     context={"name": name, "control_type": control_type})
