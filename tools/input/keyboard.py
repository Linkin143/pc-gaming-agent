"""Deterministic keyboard executor built on ``pynput`` (the only key presser)."""

from __future__ import annotations

import time
from typing import Any

from pynput.keyboard import Controller, Key, KeyCode

from core.constants import DEFAULT_MAX_HOLD_MS, DEFAULT_MIN_HOLD_MS
from core.exceptions import (
    EmergencyStopError,
    ExecutionError,
    ForegroundError,
    StuckKeyError,
)
from core.logger import get_logger
from core.models import ActionPlan, ActionResult
from tools.input.safety import check_emergency_stop, guaranteed_release, validate_foreground

logger = get_logger("input.keyboard")

def _opt(name: str) -> Key | None:
    """Return a pynput Key by attribute name, or None if unsupported on this OS.

    Keys such as ``print_screen`` / ``num_lock`` exist on some platforms only;
    resolving them defensively keeps the mapping cross-platform and crash-free.
    """
    return getattr(Key, name, None)


_SPECIAL_KEYS: dict[str, Key] = {
    "esc": Key.esc, "escape": Key.esc, "enter": Key.enter, "return": Key.enter,
    "space": Key.space, "tab": Key.tab, "shift": Key.shift, "ctrl": Key.ctrl,
    "control": Key.ctrl, "alt": Key.alt, "up": Key.up, "down": Key.down,
    "left": Key.left, "right": Key.right, "backspace": Key.backspace,
    "delete": Key.delete, "home": Key.home, "end": Key.end,
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
    "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
    "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
    "shift_r": Key.shift_r, "ctrl_r": Key.ctrl_r, "alt_r": Key.alt_r,
    "alt_gr": Key.alt_gr,
}

# Optional keys that may not exist on every platform/pynput build - added only
# when the attribute resolves, so an absent key never crashes import.
for _alias, _attr in (
    ("page_up", "page_up"), ("pageup", "page_up"),
    ("page_down", "page_down"), ("pagedown", "page_down"),
    ("insert", "insert"), ("caps_lock", "caps_lock"), ("capslock", "caps_lock"),
    ("num_lock", "num_lock"), ("numlock", "num_lock"),
    ("scroll_lock", "scroll_lock"), ("print_screen", "print_screen"),
    ("printscreen", "print_screen"), ("menu", "menu"), ("pause", "pause"),
    ("win", "cmd"), ("super", "cmd"), ("cmd", "cmd"),
    ("f13", "f13"), ("f14", "f14"), ("f15", "f15"), ("f16", "f16"),
):
    _resolved = _opt(_attr)
    if _resolved is not None:
        _SPECIAL_KEYS[_alias] = _resolved


class KeyboardExecutor:
    """Executes validated keyboard ActionPlan objects."""

    def __init__(self, *, target_window: str | None = None,
                 require_foreground: bool = True, dry_run: bool = False,
                 tap_ms: int = 50, combo_ms: int = 60) -> None:
        self.target_window = target_window
        self.require_foreground = require_foreground
        self.dry_run = dry_run
        # A tap must stay down long enough for a 60Hz game to sample it. 30ms was
        # borderline on a loaded machine; 50ms guarantees at least ~3 frames of
        # key-down while keeping latency negligible.
        self.tap_ms = max(1, int(tap_ms))
        self.combo_ms = max(1, int(combo_ms))
        self._controller = Controller()
        self._held: set[Any] = set()

    def _resolve_key(self, key: str) -> Any:
        if not key:
            raise ExecutionError("Empty key.", context={"key": key})
        low = key.strip().lower()
        if low in _SPECIAL_KEYS:
            return _SPECIAL_KEYS[low]
        if len(key) == 1:
            return KeyCode.from_char(key)
        raise ExecutionError("Unknown key name.", context={"key": key})

    def _press(self, key_obj: Any) -> None:
        # Register in `_held` BEFORE the physical press so that, if press() raises,
        # release_all() still knows to release it - preventing a stuck key.
        self._held.add(key_obj)
        if not self.dry_run:
            self._controller.press(key_obj)

    def _release(self, key_obj: Any) -> None:
        # Idempotent: skip keys we no longer hold so the explicit release plus the
        # guaranteed_release finally don't emit a redundant OS release event.
        if key_obj not in self._held:
            return
        # Release physically FIRST, then deregister, so a failed release keeps the
        # key in `_held` for a later release_all() retry rather than leaking it.
        if not self.dry_run:
            self._controller.release(key_obj)
        self._held.discard(key_obj)

    def release_all(self) -> None:
        for key_obj in list(self._held):
            try:
                self._release(key_obj)
            except Exception as exc:  # noqa: BLE001
                raise StuckKeyError("Failed to release a held key.",
                                    context={"error": str(exc)}) from exc

    def execute(self, plan: ActionPlan) -> ActionResult:
        start = time.perf_counter()
        try:
            check_emergency_stop()
            validate_foreground(self.target_window, require=self.require_foreground)
            self._dispatch(plan)
            latency = (time.perf_counter() - start) * 1000.0
            logger.info("physical_input", action=str(plan.action_type), key=plan.key,
                        keys=plan.keys, duration_ms=plan.duration_ms,
                        latency_ms=round(latency, 2), dry_run=self.dry_run)
            return ActionResult(success=True, action_type=str(plan.action_type),
                                skill_name=plan.skill_name, latency_ms=round(latency, 2))
        except (EmergencyStopError, ForegroundError, StuckKeyError):
            # Abort-level conditions must PROPAGATE, not be swallowed into a
            # retryable ActionResult: emergency-stop has to halt the whole engine,
            # and a foreground/stuck-key failure means input is unsafe to continue.
            self.release_all()
            raise
        except Exception as exc:  # noqa: BLE001
            self.release_all()
            latency = (time.perf_counter() - start) * 1000.0
            logger.error("keyboard_action_failed", error=str(exc), action=str(plan.action_type))
            return ActionResult(success=False, action_type=str(plan.action_type),
                                skill_name=plan.skill_name, latency_ms=round(latency, 2),
                                error=str(exc))

    def _dispatch(self, plan: ActionPlan) -> None:
        action = str(plan.action_type)
        if action == "keyboard_press":
            self._tap(plan.key or "")
        elif action == "keyboard_hold":
            self._hold(plan.key or "", plan.duration_ms)
        elif action == "keyboard_release":
            self._release(self._resolve_key(plan.key or ""))
        elif action == "key_combo":
            self._combo(plan.keys or [])
        else:
            raise ExecutionError("Unsupported keyboard action.", context={"action": action})

    def _tap(self, key: str) -> None:
        key_obj = self._resolve_key(key)
        with guaranteed_release(lambda: self._release(key_obj)):
            self._press(key_obj)
            time.sleep(self.tap_ms / 1000.0)
            self._release(key_obj)

    def _hold(self, key: str, duration_ms: int) -> None:
        duration_ms = max(DEFAULT_MIN_HOLD_MS, min(duration_ms or 0, DEFAULT_MAX_HOLD_MS))
        key_obj = self._resolve_key(key)
        with guaranteed_release(lambda: self._release(key_obj)):
            self._press(key_obj)
            time.sleep(duration_ms / 1000.0)
            self._release(key_obj)

    def _combo(self, keys: list[str]) -> None:
        key_objs = [self._resolve_key(k) for k in keys]

        def _release_combo() -> None:
            for k in reversed(key_objs):
                self._release(k)

        with guaranteed_release(_release_combo):
            for k in key_objs:
                self._press(k)
                # One hardware event per key: a game needs a frame between the
                # modifier(s) and the action key to register a chord (Ctrl+Shift+F3).
                time.sleep(0.005)
            time.sleep(self.combo_ms / 1000.0)
            _release_combo()
