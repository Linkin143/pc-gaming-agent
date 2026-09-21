"""Deterministic mouse executor built on ``pynput`` (the only mouse actuator)."""

from __future__ import annotations

import time

from pynput.mouse import Button, Controller

from core.exceptions import ExecutionError
from core.logger import get_logger
from core.models import ActionPlan, ActionResult
from tools.input.safety import check_emergency_stop, validate_foreground

logger = get_logger("input.mouse")

_BUTTONS: dict[str, Button] = {
    "left": Button.left, "right": Button.right, "middle": Button.middle,
}


class MouseExecutor:
    """Executes validated mouse ActionPlan objects."""

    def __init__(self, *, target_window: str | None = None,
                 require_foreground: bool = True, dry_run: bool = False,
                 screen_size: tuple[int, int] | None = None) -> None:
        self.target_window = target_window
        self.require_foreground = require_foreground
        self.dry_run = dry_run
        self._controller = Controller()
        self._screen_size = screen_size or self._detect_screen_size()

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        try:
            import win32api
            return (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))
        except Exception:  # noqa: BLE001
            return (1920, 1080)

    def _resolve_button(self, name: str | None) -> Button:
        return _BUTTONS.get((name or "left").lower(), Button.left)

    def _bounds_check(self, x: int, y: int) -> None:
        w, h = self._screen_size
        if x < -w or x > 2 * w or y < -h or y > 2 * h:
            raise ExecutionError("Mouse coordinate out of bounds.",
                                 context={"x": x, "y": y, "screen": self._screen_size})

    def _move(self, x: int, y: int, duration_ms: int) -> None:
        self._bounds_check(x, y)
        if self.dry_run:
            return
        steps = max(1, duration_ms // 15) if duration_ms else 1
        start = self._controller.position
        for i in range(1, steps + 1):
            nx = int(start[0] + (x - start[0]) * i / steps)
            ny = int(start[1] + (y - start[1]) * i / steps)
            self._controller.position = (nx, ny)
            if duration_ms:
                time.sleep((duration_ms / 1000.0) / steps)
        self._controller.position = (x, y)

    def execute(self, plan: ActionPlan) -> ActionResult:
        start = time.perf_counter()
        try:
            check_emergency_stop()
            validate_foreground(self.target_window, require=self.require_foreground)
            self._dispatch(plan)
            latency = (time.perf_counter() - start) * 1000.0
            logger.info("physical_input", action=str(plan.action_type),
                        position=plan.position.model_dump() if plan.position else None,
                        button=str(plan.button) if plan.button else None,
                        latency_ms=round(latency, 2), dry_run=self.dry_run)
            return ActionResult(success=True, action_type=str(plan.action_type),
                                skill_name=plan.skill_name, latency_ms=round(latency, 2))
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000.0
            logger.error("mouse_action_failed", error=str(exc), action=str(plan.action_type))
            return ActionResult(success=False, action_type=str(plan.action_type),
                                skill_name=plan.skill_name, latency_ms=round(latency, 2),
                                error=str(exc))

    def _dispatch(self, plan: ActionPlan) -> None:
        action = str(plan.action_type)
        button = self._resolve_button(str(plan.button) if plan.button else None)

        if action == "mouse_move":
            self._require_pos(plan)
            self._move(plan.position.x, plan.position.y, plan.duration_ms)
        elif action == "mouse_click":
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, plan.duration_ms)
            if not self.dry_run:
                self._controller.click(button, 1)
        elif action == "mouse_double_click":
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, plan.duration_ms)
            if not self.dry_run:
                self._controller.click(button, 2)
        elif action == "mouse_hold":
            # Press and hold the button for duration_ms (mining/continuous attack).
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, 0)
            hold_ms = max(0, min(plan.duration_ms, 5000))
            if not self.dry_run:
                self._controller.press(button)
                try:
                    time.sleep(hold_ms / 1000.0)
                finally:
                    self._controller.release(button)
        elif action == "mouse_drag":
            self._require_pos(plan)
            if plan.end_position is None:
                raise ExecutionError("Drag requires end_position.")
            self._move(plan.position.x, plan.position.y, 0)
            if not self.dry_run:
                self._controller.press(button)
            self._move(plan.end_position.x, plan.end_position.y, plan.duration_ms)
            if not self.dry_run:
                self._controller.release(button)
        else:
            raise ExecutionError("Unsupported mouse action.", context={"action": action})

    @staticmethod
    def _require_pos(plan: ActionPlan) -> None:
        if plan.position is None:
            raise ExecutionError("Mouse action requires a position.",
                                 context={"action": str(plan.action_type)})
