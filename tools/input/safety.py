"""Shared safety controls for the deterministic input executors."""

from __future__ import annotations

import threading
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


def validate_foreground(expected_substring: str | None, *, require: bool) -> None:
    if not require or not expected_substring:
        return
    title = get_foreground_title()
    if expected_substring.lower() not in title.lower():
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
