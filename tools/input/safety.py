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


def validate_foreground(expected_substring: str | None, *, require: bool,
                        retry_s: float = 0.5) -> None:
    """Ensure the target window is focused before sending input.

    Windows briefly drops a window from the foreground during its own
    click-to-focus handling, which would otherwise raise a spurious
    ForegroundError on a perfectly healthy action. When ``retry_s`` > 0 we wait
    that long and re-check ONCE before failing, absorbing that transient. Pass
    ``retry_s=0`` for the strict, immediate behaviour.
    """
    if not require or not expected_substring:
        return
    expected = expected_substring.lower()
    title = get_foreground_title()
    if expected in title.lower():
        return
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
