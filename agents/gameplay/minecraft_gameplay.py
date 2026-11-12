"""Explicit, deterministic Minecraft gameplay controller.

Unlike the LLM/rule planner loop, this controller fires the gameplay mechanics
DIRECTLY and in a fixed, auditable order, once the game is confirmed stable
(loading finished + in-world). It is the "just do these exact inputs" path the
user asked for:

    - forward  -> hold W
    - backward -> hold S
    - right    -> hold D
    - left     -> hold A
    - look/camera -> relative mouse move (raw-input friendly)
    - break/attack -> left mouse (hold/click)
    - build/place  -> right mouse click
    - jump     -> tap Space

Every input still goes through the SAME safe executors (KeyboardExecutor /
MouseExecutor) the rest of the framework uses, so emergency-stop, foreground
validation, and guaranteed key/button release all still apply.
"""

from __future__ import annotations

import time

from core.constants import ActionType, KeyMode, MouseButton
from core.logger import get_logger
from core.models import ActionPlan
from tools.capture.screen_capture import ScreenCapture
from tools.input.keyboard import KeyboardExecutor
from tools.input.mouse import MouseExecutor
from tools.vision.minecraft_vision import MinecraftVision

logger = get_logger("mc_gameplay")


class MinecraftGameplayController:
    """Fires explicit Minecraft gameplay commands once the game is stable."""

    def __init__(self, *, keyboard: KeyboardExecutor, mouse: MouseExecutor,
                 capture: ScreenCapture, window_title_re: str = ".*Minecraft.*",
                 dry_run: bool = False) -> None:
        self.keyboard = keyboard
        self.mouse = mouse
        self.capture = capture
        self.window_title_re = window_title_re
        self.dry_run = dry_run
        self.vision = MinecraftVision()

    # ------------------------------------------------------------------ #
    # Stability gate: wait until loading is done and we are in-world
    # ------------------------------------------------------------------ #
    def wait_until_stable(self, *, timeout_s: float = 180.0,
                          required_streak: int = 3, poll_s: float = 1.0) -> bool:
        """Block until the in-world crosshair is seen for ``required_streak``
        consecutive frames (loading screen gone, gameplay stable).

        Returns True once stable, False if ``timeout_s`` elapses first.
        """
        logger.info("gameplay_wait_stable", timeout_s=timeout_s,
                    required_streak=required_streak)
        deadline = time.time() + timeout_s
        streak = 0
        while time.time() < deadline:
            frame = self._grab()
            in_world = frame is not None and self.vision.analyse(frame).crosshair_visible
            if in_world:
                streak += 1
                logger.debug("gameplay_stable_frame", streak=streak)
                if streak >= required_streak:
                    logger.info("gameplay_stable_confirmed")
                    return True
            else:
                streak = 0
            time.sleep(poll_s)
        logger.warning("gameplay_stable_timeout")
        return False

    def _grab(self):
        try:
            result = self.capture.capture_window(self.window_title_re)
            return result.image
        except Exception as exc:  # noqa: BLE001 - capture must not crash the loop
            logger.warning("gameplay_capture_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------ #
    # Explicit mechanic primitives (each one fires a real input)
    # ------------------------------------------------------------------ #
    def move(self, direction: str, duration_ms: int = 600) -> None:
        """Hold the WASD key for ``direction`` for ``duration_ms``."""
        key = {"forward": "w", "backward": "s", "left": "a", "right": "d"}.get(
            direction.lower(), "w")
        logger.info("gameplay_move", direction=direction, key=key, duration_ms=duration_ms)
        self.keyboard.execute(ActionPlan(
            action_type=ActionType.KEYBOARD_HOLD, mode=KeyMode.HOLD,
            key=key, duration_ms=duration_ms, skill_name="explicit_move",
            description=f"Move {direction} (hold {key}).",
        ))

    def move_forward(self, duration_ms: int = 600) -> None:
        self.move("forward", duration_ms)

    def move_backward(self, duration_ms: int = 600) -> None:
        self.move("backward", duration_ms)

    def move_left(self, duration_ms: int = 600) -> None:
        self.move("left", duration_ms)

    def move_right(self, duration_ms: int = 600) -> None:
        self.move("right", duration_ms)

    def look(self, dx: int, dy: int) -> None:
        """Move the camera/view via a relative mouse delta."""
        logger.info("gameplay_look", dx=dx, dy=dy)
        self.mouse.execute(ActionPlan(
            action_type=ActionType.MOUSE_MOVE, relative=True, delta=(int(dx), int(dy)),
            duration_ms=80, skill_name="explicit_look",
            description=f"Look ({dx},{dy}).",
        ))

    def jump(self) -> None:
        """Tap Space to jump."""
        logger.info("gameplay_jump")
        self.keyboard.execute(ActionPlan(
            action_type=ActionType.KEYBOARD_PRESS, mode=KeyMode.TAP,
            key="space", skill_name="explicit_jump", description="Jump (Space).",
        ))

    def break_block(self, duration_ms: int = 1200) -> None:
        """Hold left mouse to break/mine the block under the crosshair."""
        logger.info("gameplay_break_block", duration_ms=duration_ms)
        self.mouse.execute(ActionPlan(
            action_type=ActionType.MOUSE_HOLD, button=MouseButton.LEFT,
            duration_ms=duration_ms, skill_name="explicit_break",
            description="Break block (hold left click).",
        ))

    def attack(self) -> None:
        """Single left click (attack / quick hit)."""
        logger.info("gameplay_attack")
        self.mouse.execute(ActionPlan(
            action_type=ActionType.MOUSE_CLICK, button=MouseButton.LEFT,
            skill_name="explicit_attack", description="Attack (left click).",
        ))

    def place_block(self) -> None:
        """Right click to place/build the held block."""
        logger.info("gameplay_place_block")
        self.mouse.execute(ActionPlan(
            action_type=ActionType.MOUSE_CLICK, button=MouseButton.RIGHT,
            skill_name="explicit_place", description="Place block (right click).",
        ))

    # ------------------------------------------------------------------ #
    # A concrete, explicit demonstration sequence
    # ------------------------------------------------------------------ #
    def run_demo_sequence(self, *, wait_stable: bool = True) -> bool:
        """Fire the exact mechanics the user asked for, in order.

        Returns True if the sequence ran, False if the game never stabilised.
        """
        if wait_stable and not self.wait_until_stable():
            logger.warning("gameplay_demo_skipped_not_stable")
            return False

        logger.info("gameplay_demo_start", dry_run=self.dry_run)

        # 1) Movement: forward, backward, left, right (W / S / A / D).
        self.move_forward(800)
        time.sleep(0.2)
        self.move_backward(600)
        time.sleep(0.2)
        self.move_left(600)
        time.sleep(0.2)
        self.move_right(600)
        time.sleep(0.2)

        # 2) Camera / view movement (relative mouse): right, left, up, down.
        self.look(250, 0)
        time.sleep(0.2)
        self.look(-250, 0)
        time.sleep(0.2)
        self.look(0, 150)
        time.sleep(0.2)
        self.look(0, -150)
        time.sleep(0.2)

        # 3) Jump (Space).
        self.jump()
        time.sleep(0.3)

        # 4) Break a block (hold left click).
        self.break_block(1200)
        time.sleep(0.3)

        # 5) Place a block (right click).
        self.place_block()
        time.sleep(0.3)

        logger.info("gameplay_demo_done")
        return True

    def run_loop(self, *, iterations: int = 20, wait_stable: bool = True) -> bool:
        """Repeat an explore-and-mine routine for ``iterations`` cycles.

        Each cycle: look around, walk forward, then mine ahead - a simple,
        fully deterministic gameplay loop with no LLM involvement.
        """
        if wait_stable and not self.wait_until_stable():
            logger.warning("gameplay_loop_skipped_not_stable")
            return False

        logger.info("gameplay_loop_start", iterations=iterations)
        for i in range(iterations):
            logger.info("gameplay_loop_cycle", cycle=i + 1, total=iterations)
            self.look(180, 0)         # scan the view
            time.sleep(0.15)
            self.move_forward(700)    # walk forward
            time.sleep(0.15)
            self.break_block(900)     # mine whatever is ahead
            time.sleep(0.15)
            if i % 4 == 3:
                self.jump()           # periodic hop to clear obstacles
                time.sleep(0.15)
        logger.info("gameplay_loop_done")
        return True

