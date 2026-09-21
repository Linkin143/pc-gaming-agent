"""Evidence fuser - combines OpenCV (L1) + OCR (L2) + VLM (L3) into one verdict.

The real screen is the absolute truth. OpenCV measures it deterministically,
OCR reads its text, and the VLM interprets it only when the cheaper signals
disagree. This module fuses the three into a single :class:`PerceptionEvidence`
with a voted screen state and a fused confidence - no single signal can
hallucinate the state on its own.
"""

from __future__ import annotations

from typing import Any

from core.constants import ScreenState
from core.logger import get_logger
from core.models import (
    OCRResult,
    PerceptionEvidence,
    StructuredSceneAnalysis,
    VisualFeatures,
)

logger = get_logger("fuser")

# Fusion weights (only signals that actually ran contribute).
_W_VISION = 0.40
_W_OCR = 0.35
_W_VLM = 0.25


class EvidenceFuser:
    """Fuses multi-layer perception signals into grounded evidence."""

    def __init__(self, state_definitions: dict[str, Any] | None = None) -> None:
        self._states: dict[str, Any] = (state_definitions or {}).get("states", {})

    # -- Layer 1: OpenCV visual classification (deterministic) ---------- #
    def classify_from_vision(self, vf: VisualFeatures) -> tuple[ScreenState, float]:
        # Active animation is the strongest signal - check it first. A frame with
        # motion is never a loading screen.
        if vf.motion_score >= 0.06 and vf.center_complexity >= 0.05:
            return ScreenState.GAMEPLAY, min(0.5 + vf.motion_score, 0.9)
        # Dark, structureless AND static -> loading.
        if (vf.brightness_mean < 0.06 and vf.edge_density < 0.03
                and vf.motion_score < 0.03):
            return ScreenState.GAME_LOADING, 0.7
        if vf.motion_score < 0.03 and (vf.ui_element_count >= 3
                                       or vf.text_region_density >= 0.08):
            return ScreenState.MENU, 0.55
        if vf.motion_score < 0.03 and vf.center_complexity >= 0.08:
            return ScreenState.GAMEPLAY, 0.5
        return ScreenState.UNKNOWN, 0.2

    # -- Layer 2: OCR keyword classification ---------------------------- #
    def classify_from_ocr(self, ocr_results: list[OCRResult]) -> tuple[ScreenState, float]:
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

    def should_escalate(self, vision: tuple[ScreenState, float],
                        ocr: tuple[ScreenState, float]) -> bool:
        """Escalate to the VLM only when cheap signals are weak or disagree."""
        v_state, v_conf = vision
        o_state, o_conf = ocr
        if v_state == ScreenState.UNKNOWN and o_state == ScreenState.UNKNOWN:
            return True
        if v_state != o_state and (v_conf < 0.7 and o_conf < 0.7):
            return True
        return max(v_conf, o_conf) < 0.5

    # -- Fusion --------------------------------------------------------- #
    def fuse(self, *, visual_features: VisualFeatures, ocr_results: list[OCRResult],
             scene: StructuredSceneAnalysis | None,
             methods_used: list[str]) -> PerceptionEvidence:
        v_state, v_conf = self.classify_from_vision(visual_features)
        o_state, o_conf = self.classify_from_ocr(ocr_results)

        votes: dict[str, float] = {}
        votes[str(v_state)] = votes.get(str(v_state), 0.0) + _W_VISION * v_conf
        votes[str(o_state)] = votes.get(str(o_state), 0.0) + _W_OCR * o_conf
        if scene is not None and scene.screen_state:
            s_state = self._to_screen_state(scene.screen_state)
            votes[str(s_state)] = votes.get(str(s_state), 0.0) + _W_VLM * scene.confidence

        non_unknown = {k: v for k, v in votes.items() if k != str(ScreenState.UNKNOWN)}
        tally = non_unknown or votes
        winner = max(tally, key=tally.get) if tally else str(ScreenState.UNKNOWN)
        winner_state = self._to_screen_state(winner)

        total_weight = _W_VISION + _W_OCR + (_W_VLM if scene is not None else 0.0)
        confidence = round(min(tally.get(winner, 0.0) / max(total_weight, 1e-6), 1.0), 3)

        agreement = (v_state == o_state and v_state != ScreenState.UNKNOWN)
        if agreement:
            confidence = min(1.0, confidence + 0.15)

        evidence = PerceptionEvidence(
            screen_state=winner_state,
            confidence=confidence,
            visual_features=visual_features,
            ocr_texts=[r.text for r in ocr_results if r.text.strip()],
            scene=scene,
            signal_agreement=agreement,
            votes={k: round(v, 3) for k, v in votes.items()},
            methods_used=methods_used,
        )
        logger.info("evidence_fused", screen=str(winner_state), confidence=confidence,
                    vision=str(v_state), ocr=str(o_state),
                    vlm=(scene.screen_state if scene else None), agreement=agreement)
        return evidence
