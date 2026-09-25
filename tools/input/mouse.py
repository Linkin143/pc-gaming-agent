"""Deterministic mouse executor built on ``pynput`` (the only mouse actuator)."""

from __future__ import annotations

import time

from pynput.mouse import Button, Controller

from core.constants import DEFAULT_MAX_HOLD_MS
from core.exceptions import EmergencyStopError, ExecutionError, ForegroundError
from core.logger import get_logger
from core.models import ActionPlan, ActionResult
from tools.input.safety import check_emergency_stop, guaranteed_release, validate_foreground

logger = get_logger("input.mouse")

_BUTTONS: dict[str, Button] = {
    "left": Button.left, "right": Button.right, "middle": Button.middle,
}

# A single relative look event should never sweep more than this many pixels;
# anything larger is a bug (bad delta) that would spin a raw-input camera wildly.
_MAX_REL_DELTA = 1000


class MouseExecutor:
    """Executes validated mouse ActionPlan objects."""

    def __init__(self, *, target_window: str | None = None,
                 require_foreground: bool = True, dry_run: bool = False,
                 screen_size: tuple[int, int] | None = None,
                 settle_ms: int = 30) -> None:
        self.target_window = target_window
        self.require_foreground = require_foreground
        self.dry_run = dry_run
        # Settle time after a move, before the click fires: gives the game ~2
        # frames (at 60fps) to register the cursor at its new position so the
        # click lands on the intended target rather than a stale one.
        self.settle_ms = max(0, int(settle_ms))
        self._controller = Controller()
        self._screen_size = screen_size or self._detect_screen_size()
        # Buttons physically pressed but not yet released (mouse_hold / mouse_drag).
        # Tracked so release_all() can recover from a mid-action crash.
        self._held_buttons: set[Button] = set()

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        try:
            import win32api
            return (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))
        except Exception:  # noqa: BLE001
            return (1920, 1080)

    def _resolve_button(self, name: str | None) -> Button:
        return _BUTTONS.get((name or "left").lower(), Button.left)

    # Multi-monitor slack: secondary displays can sit slightly off the primary
    # rectangle, but a valid coordinate never lands further than this beyond it.
    _EDGE_SLACK = 200

    def _bounds_check(self, x: int, y: int) -> None:
        w, h = self._screen_size
        if not (-self._EDGE_SLACK <= x <= w + self._EDGE_SLACK) or \
                not (-self._EDGE_SLACK <= y <= h + self._EDGE_SLACK):
            raise ExecutionError("Mouse coordinate out of bounds.",
                                 context={"x": x, "y": y, "screen": self._screen_size})

    # -- button state tracking (stuck-button recovery) ------------------ #
    def _press_button(self, button: Button) -> None:
        self._held_buttons.add(button)
        if not self.dry_run:
            self._controller.press(button)

    def _release_button(self, button: Button) -> None:
        # Idempotent: only release a button we actually hold, so the explicit
        # release inside a with-block plus the guaranteed_release finally don't
        # emit a redundant (and potentially confusing) second OS release event.
        if button not in self._held_buttons:
            return
        if not self.dry_run:
            self._controller.release(button)
        self._held_buttons.discard(button)

    def release_all(self) -> None:
        """Release every button we still hold - called on error/abort so a crash
        mid mouse_hold / mouse_drag never leaves a button physically stuck down."""
        for button in list(self._held_buttons):
            try:
                if not self.dry_run:
                    self._controller.release(button)
            except Exception as exc:  # noqa: BLE001
                logger.error("mouse_release_failed", button=str(button), error=str(exc))
            self._held_buttons.discard(button)

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

    def _move_relative(self, dx: int, dy: int) -> None:
        """Send a RELATIVE mouse delta (for raw-input games like Minecraft).

        ``pynput``'s ``Controller.move`` emits a relative movement event, which
        mouse-captured games read directly - unlike setting an absolute position,
        which such games ignore. Bounds are not applicable to relative deltas.
        """
        # Clamp each axis so a bad delta can't spin a raw-input camera out of
        # control (e.g. a mis-scaled look of tens of thousands of pixels).
        dx = max(-_MAX_REL_DELTA, min(int(dx), _MAX_REL_DELTA))
        dy = max(-_MAX_REL_DELTA, min(int(dy), _MAX_REL_DELTA))
        if self.dry_run:
            return
        self._controller.move(dx, dy)

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
        except (EmergencyStopError, ForegroundError):
            # Abort-level conditions must PROPAGATE so the engine can halt; still
            # release any held button first so nothing is left stuck down.
            self.release_all()
            raise
        except Exception as exc:  # noqa: BLE001
            self.release_all()
            latency = (time.perf_counter() - start) * 1000.0
            logger.error("mouse_action_failed", error=str(exc), action=str(plan.action_type))
            return ActionResult(success=False, action_type=str(plan.action_type),
                                skill_name=plan.skill_name, latency_ms=round(latency, 2),
                                error=str(exc))

    def _dispatch(self, plan: ActionPlan) -> None:
        action = str(plan.action_type)
        button = self._resolve_button(str(plan.button) if plan.button else None)

        if action == "mouse_move":
            if plan.relative:
                dx, dy = plan.delta or (0, 0)
                self._move_relative(int(dx), int(dy))
            else:
                self._require_pos(plan)
                self._move(plan.position.x, plan.position.y, plan.duration_ms)
        elif action == "mouse_click":
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, plan.duration_ms)
                self._settle()
            if not self.dry_run:
                self._controller.click(button, 1)
        elif action == "mouse_double_click":
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, plan.duration_ms)
                self._settle()
            if not self.dry_run:
                self._controller.click(button, 2)
        elif action == "mouse_hold":
            # Press and hold the button for duration_ms (mining/continuous attack).
            if plan.position is not None:
                self._move(plan.position.x, plan.position.y, 0)
            hold_ms = max(0, min(plan.duration_ms, DEFAULT_MAX_HOLD_MS))
            # Tracked press/release so a crash mid-hold is recovered by release_all.
            with guaranteed_release(lambda: self._release_button(button)):
                self._press_button(button)
                time.sleep(hold_ms / 1000.0)
                self._release_button(button)
        elif action == "mouse_drag":
            self._require_pos(plan)
            if plan.end_position is None:
                raise ExecutionError("Drag requires end_position.")
            self._move(plan.position.x, plan.position.y, 0)
            self._settle()
            # Press-drag-release with guaranteed release: if the move between
            # endpoints throws, the button is still lifted (no stuck drag).
            with guaranteed_release(lambda: self._release_button(button)):
                self._press_button(button)
                self._move(plan.end_position.x, plan.end_position.y, plan.duration_ms)
                self._release_button(button)
        else:
            raise ExecutionError("Unsupported mouse action.", context={"action": action})

    def _settle(self) -> None:
        """Pause briefly after a move so the game registers the new cursor spot
        before the click/press fires (avoids clicking a stale target)."""
        if self.settle_ms and not self.dry_run:
            time.sleep(self.settle_ms / 1000.0)

    @staticmethod
    def _require_pos(plan: ActionPlan) -> None:
        if plan.position is None:
            raise ExecutionError("Mouse action requires a position.",
                                 context={"action": str(plan.action_type)})
