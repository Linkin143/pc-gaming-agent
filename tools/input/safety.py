"""Shared safety controls for the deterministic input executors."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

from core.exceptions import EmergencyStopError, ForegroundError
from core.logger import get_logger

logger = get_logger("input.safety")

EMERGENCY_STOP = threading.Event()


def trigger_emergency_stop() -> None:
    EMERGENCY_STOP.set()
    logger.warning("emergency_stop_triggered")


def clear_emergency_stop() -> None:
    EMERGENCY_STOP.clear()


def check_emergency_stop() -> None:
    if EMERGENCY_STOP.is_set():
        raise EmergencyStopError("Emergency stop is active; input aborted.")


def get_foreground_title() -> str:
    try:
        import win32gui

        hwnd = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(hwnd) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("foreground_title_unavailable", error=str(exc))
        return ""


def focus_window(title_substring: str) -> bool:
    """Best-effort: bring the window whose title contains ``title_substring`` to
    the foreground. Returns True if a matching window was found and focused.

    Used both before a run starts and when ``validate_foreground`` detects the
    game lost focus, so a transient focus-steal (terminal/IDE grabbing focus
    right after launch) self-heals instead of aborting the whole run.
    """
    if not title_substring:
        return False
    try:
        import win32con
        import win32gui

        target = title_substring.lower()
        found: list[int] = []

        def _cb(hwnd, _acc):  # noqa: ANN001 - win32 callback signature
            if not win32gui.IsWindowVisible(hwnd):
                return
            if target in (win32gui.GetWindowText(hwnd) or "").lower():
                found.append(hwnd)

        win32gui.EnumWindows(_cb, None)
        if not found:
            return False
        hwnd = found[0]
        # Un-minimise if needed, then raise + focus.
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except Exception:  # noqa: BLE001
            pass
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("focus_window_failed", title=title_substring, error=str(exc))
        return False


def validate_foreground(expected_substring: str | None, *, require: bool,
                        retry_s: float = 1.5) -> None:
    """Ensure the target window is focused before sending input.

    Windows briefly drops a window from the foreground during its own
    click-to-focus handling (and the game can take a moment to come forward right
    after launch), which would otherwise raise a spurious ForegroundError on a
    perfectly healthy action. We first ACTIVELY try to re-focus the target
    window, then (when ``retry_s`` > 0) wait and re-check ONCE before failing,
    absorbing that transient. Pass ``retry_s=0`` for strict, immediate behaviour.
    """
    if not require or not expected_substring:
        return
    expected = expected_substring.lower()
    title = get_foreground_title()
    if expected in title.lower():
        return
    # Proactively pull the game window back to the foreground before failing.
    focus_window(expected_substring)
    if retry_s > 0:
        time.sleep(retry_s)
        title = get_foreground_title()
        if expected in title.lower():
            return
    raise ForegroundError("Target window is not in the foreground.",
                          context={"expected": expected_substring, "actual": title})


@contextmanager
def guaranteed_release(release_callback) -> Iterator[None]:
    try:
        yield
    finally:
        try:
            release_callback()
        except Exception as exc:  # noqa: BLE001
            logger.error("release_failed", error=str(exc))
