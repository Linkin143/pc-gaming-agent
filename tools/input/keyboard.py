"""Deterministic keyboard executor built on ``pynput`` (the only key presser)."""

from __future__ import annotations

import time
from typing import Any

from pynput.keyboard import Controller, Key, KeyCode

from core.constants import DEFAULT_MAX_HOLD_MS, DEFAULT_MIN_HOLD_MS
from core.exceptions import ExecutionError, StuckKeyError
from core.logger import get_logger
from core.models import ActionPlan, ActionResult
from tools.input.safety import check_emergency_stop, guaranteed_release, validate_foreground

logger = get_logger("input.keyboard")

_SPECIAL_KEYS: dict[str, Key] = {
    "esc": Key.esc, "escape": Key.esc, "enter": Key.enter, "return": Key.enter,
    "space": Key.space, "tab": Key.tab, "shift": Key.shift, "ctrl": Key.ctrl,
    "control": Key.ctrl, "alt": Key.alt, "up": Key.up, "down": Key.down,
    "left": Key.left, "right": Key.right, "backspace": Key.backspace,
    "delete": Key.delete, "home": Key.home, "end": Key.end,
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
    "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
}


class KeyboardExecutor:
    """Executes validated keyboard ActionPlan objects."""

    def __init__(self, *, target_window: str | None = None,
                 require_foreground: bool = True, dry_run: bool = False) -> None:
        self.target_window = target_window
        self.require_foreground = require_foreground
        self.dry_run = dry_run
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
        if not self.dry_run:
            self._controller.press(key_obj)
        self._held.add(key_obj)

    def _release(self, key_obj: Any) -> None:
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
            time.sleep(0.03)
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
            time.sleep(0.05)
            _release_combo()
