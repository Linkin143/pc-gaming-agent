"""Verification agent - compares expected vs observed state."""

from __future__ import annotations

from core.constants import (
    DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    ScreenState,
    VerificationResult,
    VerificationStrategy,
)
from core.logger import get_logger
from core.models import PerceptionResult, SkillDefinition, VerificationOutcome
from core.state import StructuredGameState

logger = get_logger("verification")

_MIN_POSITION_DELTA = 4.0


class VerificationAgent:
    def __init__(self, ocr_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD) -> None:
        self.ocr_threshold = ocr_threshold

    def verify(self, skill: SkillDefinition, *, before: StructuredGameState,
               after_perception: PerceptionResult,
               after_state: StructuredGameState) -> VerificationOutcome:
        strategy = VerificationStrategy(skill.verification_strategy)
        handler = {
            VerificationStrategy.NONE: self._verify_none,
            VerificationStrategy.PLAYER_POSITION_CHANGE: self._verify_position,
            VerificationStrategy.TEXT_PRESENT: self._verify_text_present,
            VerificationStrategy.TEXT_ABSENT: self._verify_text_absent,
            VerificationStrategy.SCREEN_TRANSITION: self._verify_screen_transition,
            VerificationStrategy.TASK_PROGRESS: self._verify_task_progress,
            VerificationStrategy.MENU_OPEN: self._verify_menu_open,
            VerificationStrategy.MENU_CLOSE: self._verify_menu_close,
            VerificationStrategy.OBJECT_PRESENT: self._verify_object_present,
            VerificationStrategy.STATE_EQUALS: self._verify_state_equals,
        }.get(strategy, self._verify_none)
        outcome = handler(skill, before, after_perception, after_state)
        outcome.strategy = strategy
        logger.info("verification_result", result=str(outcome.result), strategy=str(strategy),
                    skill=skill.name, confidence=outcome.confidence, reason=outcome.reason)
        return outcome

    def _verify_none(self, skill, before, perception, after) -> VerificationOutcome:
        return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.5,
                                   reason="No verification strategy defined; assuming success.")

    def _verify_position(self, skill, before, perception, after) -> VerificationOutcome:
        prev = before.player.position or {}
        curr = after.player.position or {}
        if not prev or not curr:
            return VerificationOutcome(result=VerificationResult.UNCERTAIN, confidence=0.3,
                                       reason="Player position unavailable before/after.")
        dx = abs(curr.get("x", 0) - prev.get("x", 0))
        dy = abs(curr.get("y", 0) - prev.get("y", 0))
        if (dx + dy) >= _MIN_POSITION_DELTA:
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.8,
                                       reason=f"Player moved by ({dx},{dy}).")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.6,
                                   reason="Player did not move appreciably.")

    def _expected_text(self, skill) -> str:
        return str(skill.verification.get("text") or skill.verification.get("expected_text")
                   or skill.expected_result or "")

    def _verify_text_present(self, skill, before, perception, after) -> VerificationOutcome:
        target = self._expected_text(skill).lower()
        text = perception.all_text().lower()
        if not target:
            return self._verify_none(skill, before, perception, after)
        if target in text:
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.85,
                                       reason=f"Expected text '{target}' present.")
        if perception.ocr_confidence < self.ocr_threshold:
            return VerificationOutcome(result=VerificationResult.UNCERTAIN, confidence=0.4,
                                       reason="Expected text not found; OCR confidence low.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.7,
                                   reason=f"Expected text '{target}' not present.")

    def _verify_text_absent(self, skill, before, perception, after) -> VerificationOutcome:
        target = self._expected_text(skill).lower()
        text = perception.all_text().lower()
        if target and target not in text:
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.8,
                                       reason=f"Text '{target}' correctly absent.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.6,
                                   reason=f"Text '{target}' still present.")

    def _expected_state(self, skill) -> str | None:
        exp = skill.verification.get("expected_state")
        return str(exp) if exp else None

    def _verify_screen_transition(self, skill, before, perception, after) -> VerificationOutcome:
        expected = self._expected_state(skill)
        if expected:
            if str(after.screen) == expected:
                return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.85,
                                           reason=f"Reached expected screen '{expected}'.")
            return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.65,
                                       reason=f"Screen is '{after.screen}', expected '{expected}'.")
        if str(before.screen) != str(after.screen):
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.7,
                                       reason=f"Screen changed {before.screen} -> {after.screen}.")
        return VerificationOutcome(result=VerificationResult.UNCERTAIN, confidence=0.4,
                                   reason="No screen transition detected.")

    def _verify_task_progress(self, skill, before, perception, after) -> VerificationOutcome:
        b = before.game_state.get("task_progress")
        a = after.game_state.get("task_progress")
        if b is None or a is None:
            return VerificationOutcome(result=VerificationResult.UNCERTAIN, confidence=0.35,
                                       reason="Task progress unavailable.")
        if a > b:
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.85,
                                       reason=f"Task progress increased {b} -> {a}.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.6,
                                   reason="Task progress did not increase.")

    def _verify_menu_open(self, skill, before, perception, after) -> VerificationOutcome:
        if after.screen in (ScreenState.MENU, ScreenState.DIALOG):
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.75,
                                       reason="Menu/dialog is open.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.55,
                                   reason="Menu did not open.")

    def _verify_menu_close(self, skill, before, perception, after) -> VerificationOutcome:
        if after.screen not in (ScreenState.MENU, ScreenState.DIALOG):
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.75,
                                       reason="Menu/dialog is closed.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.55,
                                   reason="Menu did not close.")

    def _verify_object_present(self, skill, before, perception, after) -> VerificationOutcome:
        if perception.detected_objects:
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.7,
                                       reason=f"Detected {len(perception.detected_objects)} object(s).")
        return VerificationOutcome(result=VerificationResult.UNCERTAIN, confidence=0.4,
                                   reason="No objects detected.")

    def _verify_state_equals(self, skill, before, perception, after) -> VerificationOutcome:
        expected = self._expected_state(skill) or skill.verification.get("target_state")
        if not expected:
            return self._verify_none(skill, before, perception, after)
        if str(after.screen) == str(expected):
            return VerificationOutcome(result=VerificationResult.SUCCESS, confidence=0.85,
                                       reason=f"State equals '{expected}'.")
        return VerificationOutcome(result=VerificationResult.FAILURE, confidence=0.6,
                                   reason=f"State '{after.screen}' != expected '{expected}'.")
