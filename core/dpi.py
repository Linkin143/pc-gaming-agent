"""Per-monitor DPI awareness for correct screen-coordinate math.

The whole framework captures the screen with ``mss`` (physical pixels) and clicks
with ``pynput`` (logical pixels). If the process is DPI-*unaware*, Windows returns
full-resolution captures but virtualises cursor coordinates, so on a scaled
display (125%/150%) every click lands off-target by the scale factor - which made
the launcher click ~198px away from the Minecraft tile and never open the game.

Declaring the process per-monitor DPI-aware makes capture and clicks share ONE
physical-pixel coordinate space, so OCR bounding boxes map 1:1 to click points on
any monitor/scale. This must be called ONCE, as early as possible, before any
capture or input. It is a safe no-op on non-Windows platforms.
"""

from __future__ import annotations

from core.logger import get_logger

logger = get_logger("dpi")

_APPLIED = False


def set_dpi_awareness() -> bool:
    """Make the process per-monitor DPI-aware (idempotent). Returns True on success.

    Tries the modern PER_MONITOR_AWARE_V2 context first, then falls back to older
    APIs for compatibility with earlier Windows builds. Never raises.
    """
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import ctypes
    except Exception:  # noqa: BLE001 - non-CPython / restricted env
        return False

    # 1) Windows 10 1703+: SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2 = -4)
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
            if user32.SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2):
                _APPLIED = True
                logger.info("dpi_awareness_set", api="PerMonitorV2")
                return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("dpi_permonitorv2_failed", error=str(exc))

    # 2) Windows 8.1+: shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE=2)
    try:
        shcore = ctypes.windll.shcore
        PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        _APPLIED = True
        logger.info("dpi_awareness_set", api="shcore.PerMonitor")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("dpi_shcore_failed", error=str(exc))

    # 3) Vista+: user32.SetProcessDPIAware() (system-DPI aware; better than nothing)
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        _APPLIED = True
        logger.info("dpi_awareness_set", api="SetProcessDPIAware")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("dpi_setprocessdpiaware_failed", error=str(exc))

    return False
