"""State builder - converts perception results into structured game state."""

from __future__ import annotations

from typing import Any

from core.constants import ScreenState
from core.logger import get_logger
from core.models import OCRResult, PerceptionResult
from core.state import StructuredGameState

logger = get_logger("state_builder")


class StateBuilder:
    def __init__(self, state_definitions: dict[str, Any] | None = None) -> None:
        self._states: dict[str, Any] = (state_definitions or {}).get("states", {})

    def classify_screen(self, ocr_results: list[OCRResult]) -> tuple[ScreenState, float]:
        text = " ".join(r.text.lower() for r in ocr_results)
        best_state = ScreenState.UNKNOWN
        best_hits = 0
        best_total = 1
        for _name, spec in self._states.items():
            keywords = [k.lower() for k in spec.get("keywords", [])]
            if not keywords:
                continue
            hits = sum(1 for k in keywords if k in text)
            if hits > best_hits:
                best_hits = hits
                best_total = len(keywords)
                best_state = self._to_screen_state(spec.get("screen", "unknown"))
        confidence = (best_hits / best_total) if best_hits else 0.0
        return best_state, round(min(confidence, 1.0), 3)

    @staticmethod
    def _to_screen_state(value: str) -> ScreenState:
        try:
            return ScreenState(value)
        except ValueError:
            return ScreenState.UNKNOWN

    def build_perception(self, perception: PerceptionResult, *,
                         prefer_vlm: bool = False) -> PerceptionResult:
        """Finalise the screen_state/confidence.

        If the perception already carries fused evidence (produced by the new
        PerceptionAgent), trust it - it already combined OpenCV + OCR + VLM.
        Otherwise fall back to the legacy OCR keyword classification so older
        code paths and tests keep working.
        """
        if perception.evidence is not None:
            perception.screen_state = perception.evidence.screen_state
            perception.overall_confidence = perception.evidence.confidence
            return perception

        screen, ocr_score = self.classify_screen(perception.ocr_results)
        if prefer_vlm and perception.vlm_result and perception.vlm_result.classification:
            vlm_screen = self._to_screen_state(perception.vlm_result.classification)
            if vlm_screen is not ScreenState.UNKNOWN:
                screen = vlm_screen
        perception.screen_state = screen
        perception.ocr_confidence = max(perception.ocr_confidence, ocr_score)
        perception.overall_confidence = max(
            perception.ocr_confidence, perception.vision_confidence,
            perception.vlm_result.confidence if perception.vlm_result else 0.0,
        )
        return perception

    def merge_into_state(self, state: StructuredGameState, perception: PerceptionResult, *,
                         game_state_updates: dict[str, Any] | None = None) -> StructuredGameState:
        previous_screen = state.screen
        state.apply_perception(perception)
        # Surface the visual evidence into game_state so the planner (and the
        # LLM prompt) can reason over concrete pixel measurements.
        if perception.evidence is not None:
            ev = perception.evidence
            state.game_state["_visual"] = ev.visual_features.model_dump()
            state.game_state["_ocr_texts"] = ev.ocr_texts[:30]
            state.game_state["_signal_agreement"] = ev.signal_agreement
            if ev.scene is not None:
                state.game_state["_scene"] = {
                    "what_i_see": ev.scene.what_i_see,
                    "hint": ev.scene.recommended_action_hint,
                    "ui": ev.scene.visible_ui_elements,
                }
        if game_state_updates:
            state.game_state.update(game_state_updates)
        if previous_screen != state.screen:
            logger.info("state_transition", from_state=str(previous_screen),
                        to_state=str(state.screen), confidence=state.overall_confidence)
        return state
