"""Hierarchical, cost-aware, evidence-fusing perception coordinator.

The real screen is the absolute truth. Every cycle we:
  L1  OpenCV  -> deterministic VisualFeatures (edges, motion, colours, UI count)
  L2  OCR     -> the literal text on screen
  L3  VLM     -> schema-constrained scene analysis, ONLY when L1+L2 disagree
Then fuse all three into a single PerceptionEvidence packet the planner reasons
over. No single signal can hallucinate the state on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.perception.evidence_fuser import EvidenceFuser
from core.config import AppConfig
from core.constants import PerceptionMethod, ScreenState
from core.logger import get_logger
from core.models import PerceptionResult, Rect, ScreenshotRef, VisualFeatures
from tools.capture.screen_capture import ScreenCapture
from tools.ocr.paddle_engine import PaddleOCREngine
from tools.vision.opencv_engine import OpenCVEngine
from tools.vision.vlm_engine import VLMEngine

logger = get_logger("perception")


class PerceptionAgent:
    def __init__(self, config: AppConfig, *, capture: ScreenCapture,
                 opencv: OpenCVEngine | None = None, ocr: PaddleOCREngine | None = None,
                 vlm: VLMEngine | None = None, screen_candidates: list[str] | None = None,
                 fuser: EvidenceFuser | None = None,
                 game: str = "unknown", goal: str = "",
                 window_title_re: str | None = None) -> None:
        self.config = config
        self.capture = capture
        self.opencv = opencv or OpenCVEngine()
        self.ocr = ocr
        self.vlm = vlm
        self.fuser = fuser or EvidenceFuser()
        self.game = game
        self.goal = goal
        # When set, perception captures ONLY this window's region (the real game
        # window) instead of the whole desktop - keeps OCR/OpenCV on the game.
        self.window_title_re = window_title_re
        self.screen_candidates = screen_candidates or [s.value for s in ScreenState]
        self._prev_frame: np.ndarray | None = None
        self._last_result: PerceptionResult | None = None
        self._prev_screen: str = "unknown"
        self._mc_vision: Any | None = None   # lazy MinecraftVision for OCR-skip check

    def set_context(self, *, game: str | None = None, goal: str | None = None,
                    window_title_re: str | None = None) -> None:
        if game is not None:
            self.game = game
        if goal is not None:
            self.goal = goal
        if window_title_re is not None:
            self.window_title_re = window_title_re
            # New scene: drop the previous frame so motion isn't spuriously huge.
            self._prev_frame = None

    def reset(self) -> None:
        self._prev_frame = None
        self._last_result = None

    def _is_in_world(self, frame: np.ndarray) -> bool:
        """Fast crosshair pixel check - True only for in-world Minecraft frames.

        Used to skip the expensive OCR pass during gameplay (the HUD is icons, not
        text). Non-Minecraft games always return False so their OCR is unaffected.
        """
        if self.game != "minecraft" or frame is None:
            return False
        try:
            if self._mc_vision is None:
                from tools.vision.minecraft_vision import MinecraftVision
                self._mc_vision = MinecraftVision()
            return bool(self._mc_vision.analyse(frame).crosshair_visible)
        except Exception as exc:  # noqa: BLE001 - never let this block perception
            logger.debug("in_world_check_failed", error=str(exc))
            return False

    @staticmethod
    def _visual_summary(vf: VisualFeatures) -> str:
        return (f"edges={vf.edge_density:.2f} motion={vf.motion_score:.2f} "
                f"brightness={vf.brightness_mean:.2f} ui_rects={vf.ui_element_count} "
                f"text_density={vf.text_region_density:.2f} "
                f"center={vf.center_complexity:.2f} colors={vf.dominant_colors}")

    def perceive(self, *, run_id: str, region: Rect | None = None,
                 force_vlm: bool = False, save_screenshot: bool = True) -> PerceptionResult:
        pc = self.config.perception
        methods: list[str] = []

        if region is not None:
            capture_result = self.capture.capture_region(region)
        elif self.window_title_re:
            # Scope capture to the real game window (not the whole desktop).
            capture_result = self.capture.capture_window(self.window_title_re)
        else:
            capture_result = self.capture.capture_full()
        frame = capture_result.image

        # -- change detection (cheap short-circuit) ------------------------ #
        change_score = 1.0
        if self._prev_frame is not None:
            change_score = self.opencv.change_score(self._prev_frame, frame)
            if (change_score < pc.change_detection_threshold
                    and self._last_result is not None and not force_vlm):
                logger.debug("perception_cached", change_score=round(change_score, 4))
                self._last_result.changed = False
                self._last_result.methods_used = [PerceptionMethod.CACHE]
                return self._last_result

        result = PerceptionResult(changed=True)

        # Screenshot save is DEFERRED until after fusion: writing a 1080p PNG every
        # cycle is wasteful, so we only persist frames that are actually worth the
        # audit trail (the VLM ran, or the screen state changed).
        ref: ScreenshotRef | None = None

        # -- L1: OpenCV visual features (deterministic ground truth) ------- #
        vf = self.opencv.extract_features(frame, self._prev_frame)
        result.visual_features = vf
        result.vision_confidence = max(vf.edge_density, vf.motion_score)
        methods.append(str(PerceptionMethod.OPENCV))
        try:
            vision = self.opencv.detect_contours(frame, min_area=250)
            result.vision_results.append(vision)
            result.detected_objects = vision.matches
        except Exception as exc:  # noqa: BLE001
            logger.warning("opencv_contours_failed", error=str(exc))

        # -- L2: OCR ------------------------------------------------------- #
        # Skip OCR entirely for in-world Minecraft frames: the HUD is icons (read
        # by MinecraftVision, not OCR), so a ~40s PaddleOCR pass returns nothing
        # useful. Detecting the crosshair is a few-ms pixel check.
        skip_ocr = self._is_in_world(frame)
        if self.ocr is not None and not skip_ocr:
            try:
                # ROI crop: OCR only the interactive UI band, not the full frame.
                ocr_results = self.ocr.read_text_roi(frame)
                result.ocr_results = ocr_results
                result.ocr_confidence = (
                    float(sum(r.confidence for r in ocr_results) / len(ocr_results))
                    if ocr_results else 0.0
                )
                methods.append(str(PerceptionMethod.OCR))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ocr_failed", error=str(exc))
        elif skip_ocr:
            logger.debug("ocr_skipped_in_world")

        # -- Decide on L3 (VLM) via the fuser's escalation policy ---------- #
        v_vote = self.fuser.classify_from_vision(vf)
        o_vote = self.fuser.classify_from_ocr(result.ocr_results)
        scene = None
        want_vlm = force_vlm or (pc.enable_vlm and self.vlm is not None
                                 and self.fuser.should_escalate(v_vote, o_vote))
        if want_vlm and self.vlm is not None:
            try:
                scene = self.vlm.analyze_scene(
                    frame, game=self.game, goal=self.goal,
                    ocr_texts=[r.text for r in result.ocr_results],
                    visual_summary=self._visual_summary(vf),
                    prev_screen=self._prev_screen,
                )
                methods.append(str(PerceptionMethod.VLM))
                logger.info("vlm_scene", screen=scene.screen_state,
                            confidence=scene.confidence, sees=scene.what_i_see[:80])
            except Exception as exc:  # noqa: BLE001 - VLM failure is non-fatal
                logger.warning("vlm_failed", error=str(exc))

        # -- Fuse the three layers into grounded evidence ------------------ #
        evidence = self.fuser.fuse(visual_features=vf, ocr_results=result.ocr_results,
                                   scene=scene, methods_used=methods)

        # Persist the frame ONLY when it is worth the audit trail: the VLM ran, the
        # screen state changed, or the caller explicitly asked. Skipping the PNG
        # write on stable frames removes ~100ms per cycle.
        if save_screenshot and (
                want_vlm or force_vlm
                or str(evidence.screen_state) != self._prev_screen):
            try:
                ref = self.capture.save(capture_result, run_id=run_id, tag="perception")
            except Exception as exc:  # noqa: BLE001 - saving is non-critical
                logger.debug("screenshot_save_skipped", error=str(exc))
        result.screenshot_ref = ref
        evidence.screenshot_ref = ref
        result.evidence = evidence
        result.screen_state = evidence.screen_state
        result.overall_confidence = evidence.confidence
        result.methods_used = [PerceptionMethod(m) for m in methods
                               if m in PerceptionMethod._value2member_map_]
        result.extra["change_score"] = round(change_score, 4)
        result.extra["votes"] = evidence.votes

        self._prev_frame = frame
        self._prev_screen = str(evidence.screen_state)
        self._last_result = result
        return result
