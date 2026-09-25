"""Skill-driven, screen-truth launch agent.

This replaces the old hard-coded, fixed-timing launch script. The launch flow
is described DECLARATIVELY in ``skills/games/minecraft/xbox_launch.yaml`` under
``launch_states`` / ``launch_config``. This agent reads that skill file and, on
every cycle:

    1. Captures the LIVE screen (the single source of truth).
    2. Perceives it with OpenCV (crosshair pixel-truth via MinecraftVision) and
       OCR (the literal on-screen text), scoped to whichever window (Xbox app or
       Minecraft) is currently in front.
    3. Matches the observation against the declarative state ``detect`` blocks
       and performs the FIRST matching state's ``action`` - locating click
       targets by OCR at run time (never hard-coded pixel positions).
    4. Re-observes. Slow app/menu rendering just means the same state keeps
       matching until the screen actually changes - there are NO blind sleeps
       between steps.

When no declarative state matches for ``confuse_limit`` consecutive cycles, the
agent escalates to the VLM/LLM to classify the frame and choose the closest
action. The screen is authoritative; the LLM only arbitrates genuine ambiguity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.config import AppConfig, get_config
from core.constants import LLMProvider
from core.exceptions import WindowNotFoundError
from core.logger import get_logger
from core.models import StructuredSceneAnalysis, VisualFeatures
from tools.desktop.screen_finder import ScreenFinder
from tools.desktop.winapp import XboxDesktopAutomation
from tools.vision.minecraft_vision import MinecraftVision
from tools.vision.opencv_engine import OpenCVEngine, classify_screen_state

logger = get_logger("launch_agent")

# VLM-vote weight when it participates (OpenCV/OCR carry the deterministic base).
_W_OPENCV = 0.35
_W_OCR = 0.40
_W_VLM = 0.25


@dataclass
class PerceptionBundle:
    """One cycle's three-layer screen truth (OpenCV + OCR + optional VLM).

    Every action decision is made from THIS object, and the frame it was built
    from is saved to ``screenshots/`` so each decision has an audit trail.
    """

    # -- Raw capture ---------------------------------------------------- #
    frame: np.ndarray | None = None
    screenshot_path: str | None = None
    window_titles: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    # -- L1: OpenCV (deterministic pixel measurements) ------------------ #
    visual_features: VisualFeatures | None = None
    opencv_state: str = "unknown"
    opencv_confidence: float = 0.0

    # -- L2: OCR + crosshair pixel truth -------------------------------- #
    ocr_text: str = ""
    crosshair: bool = False

    # -- L3: VLM (populated only when needed) --------------------------- #
    vlm_scene: StructuredSceneAnalysis | None = None
    vlm_action_hint: str = ""

    # -- Fused verdict -------------------------------------------------- #
    fused_state: str = "unknown"
    fused_confidence: float = 0.0
    signals_agree: bool = False

    def has_window(self, substring: str) -> bool:
        s = substring.lower()
        return any(s in (t or "").lower() for t in self.window_titles)


# Backwards-compatible alias (older tests/imports referenced ``Observation``).
Observation = PerceptionBundle


class LaunchAgent:
    """Reads the launch skill file and drives Minecraft to the in-world state."""

    def __init__(self, *, config: AppConfig | None = None,
                 xbox: XboxDesktopAutomation | None = None,
                 finder: ScreenFinder | None = None,
                 vision: MinecraftVision | None = None,
                 opencv: OpenCVEngine | None = None) -> None:
        self.config = config or get_config()
        # Ensure the process is DPI-aware before any capture/click, in case the
        # LaunchAgent is constructed directly (not via main()). Idempotent.
        try:
            from core.dpi import set_dpi_awareness
            set_dpi_awareness()
        except Exception:  # noqa: BLE001 - never block launch on this
            pass
        self.xbox = xbox or XboxDesktopAutomation()
        self.finder = finder or ScreenFinder()
        self.vision = vision or MinecraftVision()
        self.opencv = opencv or OpenCVEngine()
        self._vlm: Any | None = None
        self._vlm_failures: int = 0         # consecutive VLM call failures
        self._vlm_disabled: bool = False    # set True after too many failures
        self._prev_frame: np.ndarray | None = None
        self._prev_ocr_text: str = ""       # last cycle's OCR text (motion-gate reuse)
        self._shot_dir: Path | None = None  # per-run screenshot folder
        self._states, self._cfg = self._load_skill()

    # ------------------------------------------------------------------ #
    # Screenshot audit trail - every perceived frame is saved to disk
    # ------------------------------------------------------------------ #
    def _ensure_shot_dir(self) -> Path:
        if self._shot_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._shot_dir = Path(self.config.screenshots_dir) / f"launch_{stamp}"
            try:
                self._shot_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("shot_dir_create_failed", error=str(exc))
        return self._shot_dir

    def _save_frame(self, frame: np.ndarray | None, tag: str,
                    confidence: float) -> str | None:
        if frame is None:
            return None
        try:
            import cv2
            folder = self._ensure_shot_dir()
            ts = datetime.now().strftime("%H-%M-%S")
            safe_tag = "".join(c if c.isalnum() else "_" for c in tag)[:24]
            path = folder / f"{ts}_{safe_tag}_{confidence:.2f}.png"
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception as exc:  # noqa: BLE001 - saving must never break the loop
            logger.debug("save_frame_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------ #
    # Skill loading
    # ------------------------------------------------------------------ #
    def _load_skill(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = (Path(self.config.skills_dir) / "games" / "minecraft"
                / "xbox_launch.yaml")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            logger.error("launch_skill_load_failed", error=str(exc))
            return [], {}
        states = data.get("launch_states", []) or []
        cfg = data.get("launch_config", {}) or {}
        logger.info("launch_skill_loaded", states=len(states),
                    names=[s.get("name") for s in states])
        return states, cfg


    # ------------------------------------------------------------------ #
    # Perception - the live screen is the source of truth
    # ------------------------------------------------------------------ #
    @staticmethod
    def _list_window_titles() -> list[str]:
        titles: list[str] = []
        try:
            import win32gui

            def _cb(hwnd, _acc):  # noqa: ANN001 - win32 callback signature
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if text:
                        titles.append(text)

            win32gui.EnumWindows(_cb, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("enum_windows_failed", error=str(exc))
        return titles

    def _point_finder_at(self, obs_titles: list[str]) -> None:
        """Scope OCR capture to the frontmost relevant window (game > xbox)."""
        game = str(self._cfg.get("window_game", "Minecraft"))
        xbox = str(self._cfg.get("window_xbox", "XBOX"))
        target = None
        if any(game.lower() in (t or "").lower() for t in obs_titles):
            target = game
        elif any(xbox.lower() in (t or "").lower() for t in obs_titles):
            target = xbox
        if target is None:
            self.finder.region = None
            return
        try:
            from tools.capture.screen_capture import ScreenCapture
            rect = ScreenCapture.find_window_rect(f".*{target}.*")
            self.finder.region = rect  # None -> full desktop fallback
        except Exception as exc:  # noqa: BLE001
            logger.debug("region_scope_failed", error=str(exc))
            self.finder.region = None

    def perceive(self) -> PerceptionBundle:
        """Three-layer perception: OpenCV (L1) + OCR/crosshair (L2) + VLM (L3).

        Produces a fused verdict every cycle and saves the frame to
        ``screenshots/`` for an audit trail. The VLM (L3) is invoked only when
        the cheap L1+L2 signals are weak or disagree - the screen is always the
        source of truth; the VLM merely arbitrates ambiguity.
        """
        titles = self._list_window_titles()
        self._point_finder_at(titles)

        bundle = PerceptionBundle(window_titles=titles, timestamp=time.time())
        try:
            bundle.frame, _, _ = self.finder.screenshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("perceive_screenshot_failed", error=str(exc))

        if bundle.frame is not None:
            # -- L1: OpenCV deterministic pixel measurements --------------- #
            try:
                vf = self.opencv.extract_features(bundle.frame, self._prev_frame)
                bundle.visual_features = vf
                bundle.opencv_state, bundle.opencv_confidence = classify_screen_state(vf)
            except Exception as exc:  # noqa: BLE001
                logger.debug("perceive_opencv_failed", error=str(exc))
            # -- Crosshair pixel truth FIRST (a few ms) -------------------- #
            # If we are already in-world the crosshair is visible; the loop returns
            # success immediately and OCR (the ~40s cost) is pointless because the
            # in-world HUD is icons, not text. So read the crosshair before OCR and
            # skip OCR entirely when it is present.
            try:
                bundle.crosshair = bool(self.vision.analyse(bundle.frame).crosshair_visible)
            except Exception as exc:  # noqa: BLE001
                logger.debug("perceive_vision_failed", error=str(exc))

            # -- L2: OCR text (skipped in-world; motion-gated + cached) --------- #
            if not bundle.crosshair:
                # Change-detection gate: if the frame barely changed since last
                # cycle AND we already have OCR text, the on-screen text is the same
                # - reuse it and skip OCR entirely (the architecture's "NO
                # SIGNIFICANT CHANGE -> cache result" branch). motion_score is the
                # normalised mean pixel delta already computed by L1 this cycle.
                motion = float(getattr(bundle.visual_features, "motion_score", 1.0)
                               if bundle.visual_features else 1.0)
                change_thresh = float(self._cfg.get("ocr_change_threshold", 0.006))
                if (self._prev_frame is not None and self._prev_ocr_text
                        and motion < change_thresh):
                    bundle.ocr_text = self._prev_ocr_text
                    logger.debug("ocr_skipped_no_change", motion=round(motion, 4))
                else:
                    try:
                        # use_cache=True + ROI crop: an unchanged screen within the
                        # cache TTL reuses the last OCR result; read_text_roi
                        # processes only the interactive UI band so a genuine miss
                        # is markedly faster than a full-frame scan.
                        results = self.finder._ocr.read_text_roi(  # noqa: SLF001
                            bundle.frame, use_cache=True)
                        bundle.ocr_text = " ".join(r.text.lower() for r in results)
                        self._prev_ocr_text = bundle.ocr_text
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("perceive_ocr_failed", error=str(exc))
            self._prev_frame = bundle.frame

        # -- Fuse L1 + L2 into a state verdict ----------------------------- #
        self._fuse_l1_l2(bundle)

        # -- L3: VLM only when GENUINELY needed --------------------------- #
        # The VLM costs ~40s per call, so we must not fire it when the cheap
        # signals already resolved the screen. It runs only when:
        #   * OCR did NOT confidently match any declarative state, OR
        #   * the fused confidence is weak, OR
        #   * the matched state's action is `click_world` (world-select uses the
        #     VLM hint to pick a world by name).
        # Crucially we DROP the old `not signals_agree` trigger: OpenCV's coarse
        # vote (menu/gameplay/unknown) rarely equals OCR's precise state, so that
        # clause fired the VLM almost every cycle even on a confident OCR match -
        # which is what made each launch cycle take ~80s.
        matched = self._match_state(bundle)
        matched_op = (matched.get("action", {}) or {}).get("op") if matched else None
        # A state may declare `skip_vlm: true` when its OCR match is unambiguous
        # (e.g. game_card/mc_title/search_results) - then the VLM adds latency with
        # no value and is suppressed even below the confidence threshold.
        state_skips_vlm = bool((matched or {}).get("skip_vlm", False))
        vlm_threshold = float(self._cfg.get("vlm_threshold", 0.75))
        if bundle.frame is not None and not bundle.crosshair and (
                matched is None
                or (bundle.fused_confidence < vlm_threshold and not state_skips_vlm)
                or matched_op == "click_world"):
            self._augment_with_vlm(bundle)

        # -- Persist the frame with the final verdict in its name ---------- #
        bundle.screenshot_path = self._save_frame(
            bundle.frame, tag=bundle.fused_state, confidence=bundle.fused_confidence)

        logger.info(
            "perception_cycle",
            opencv={"state": bundle.opencv_state,
                    "conf": round(bundle.opencv_confidence, 2)},
            ocr={"state": self._matched_state_name(bundle),
                 "text_len": len(bundle.ocr_text)},
            vlm=({"state": str(bundle.vlm_scene.screen_state),
                  "hint": bundle.vlm_action_hint[:48]} if bundle.vlm_scene else None),
            fused={"state": bundle.fused_state,
                   "conf": round(bundle.fused_confidence, 2),
                   "agree": bundle.signals_agree},
            crosshair=bundle.crosshair,
            screenshot=bundle.screenshot_path,
        )
        return bundle

    # Backwards-compatible name (older callers/tests used ``observe``).
    def observe(self) -> PerceptionBundle:
        return self.perceive()

    # ------------------------------------------------------------------ #
    # Evidence fusion
    # ------------------------------------------------------------------ #
    # Map OpenCV's coarse categories onto declarative launch-state names so the
    # OpenCV vote can agree/disagree with the OCR (declarative) match.
    _OPENCV_TO_STATE = {
        "gameplay": "in_world",
        "loading": "loading",
        "menu": "mc_title",  # a Minecraft menu-like screen (best coarse guess)
        "unknown": None,
    }

    def _matched_state_name(self, bundle: PerceptionBundle) -> str | None:
        state = self._match_state(bundle)
        return state.get("name") if state else None

    def _fuse_l1_l2(self, bundle: PerceptionBundle) -> None:
        """Combine the OpenCV vote and the OCR declarative match into a verdict."""
        ocr_state = self._matched_state_name(bundle)
        # OCR/declarative match confidence: strong because it is keyword-precise.
        ocr_conf = 0.8 if ocr_state else 0.0
        opencv_state = self._OPENCV_TO_STATE.get(bundle.opencv_state)

        votes: dict[str, float] = {}
        if ocr_state:
            votes[ocr_state] = votes.get(ocr_state, 0.0) + _W_OCR * ocr_conf
        if opencv_state:
            votes[opencv_state] = (votes.get(opencv_state, 0.0)
                                   + _W_OPENCV * bundle.opencv_confidence)

        if votes:
            winner = max(votes, key=votes.get)
            total = _W_OCR * (1.0 if ocr_state else 0.0) + _W_OPENCV * (
                1.0 if opencv_state else 0.0)
            bundle.fused_state = winner
            bundle.fused_confidence = round(min(votes[winner] / max(total, 1e-6), 1.0), 3)
        else:
            bundle.fused_state = ocr_state or "unknown"
            bundle.fused_confidence = ocr_conf

        # Agreement means the DETERMINISTIC signals (OpenCV L1 + OCR L2) concur.
        # The VLM (L3) is folded in later by _augment_with_vlm, which recomputes
        # agreement to include its vote; here we only claim agreement when both
        # cheap signals produced the SAME non-empty state. This is used purely as a
        # confidence bonus, so it must not over-claim when they actually differ.
        bundle.signals_agree = bool(
            ocr_state and opencv_state and ocr_state == opencv_state)
        if bundle.signals_agree:
            bundle.fused_confidence = min(1.0, bundle.fused_confidence + 0.15)

    # ------------------------------------------------------------------ #
    # Declarative condition matching
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        return [str(v) for v in (value if isinstance(value, list) else [value])]

    def _conditions_hold(self, cond: dict[str, Any], obs: Observation) -> bool:
        if not cond:
            return True
        # window_present / window_absent each accept a single value OR a list.
        if "window_present" in cond and not all(
                obs.has_window(w) for w in self._as_list(cond["window_present"])):
            return False
        if "window_absent" in cond and any(
                obs.has_window(w) for w in self._as_list(cond["window_absent"])):
            return False
        if "crosshair" in cond and bool(cond["crosshair"]) != obs.crosshair:
            return False
        any_text = cond.get("any_text")
        if any_text and not any(str(t).lower() in obs.ocr_text for t in any_text):
            return False
        all_text = cond.get("all_text")
        if all_text and not all(str(t).lower() in obs.ocr_text for t in all_text):
            return False
        not_text = cond.get("not_text")
        if not_text and any(str(t).lower() in obs.ocr_text for t in not_text):
            return False
        return True

    def _match_state(self, obs: Observation) -> dict[str, Any] | None:
        for state in self._states:
            if self._conditions_hold(state.get("detect", {}), obs):
                return state
        return None

    # ------------------------------------------------------------------ #
    # Actions - all click targets located by OCR at run time
    # ------------------------------------------------------------------ #
    def _do_action(self, state: dict[str, Any], obs: PerceptionBundle) -> None:
        action = state.get("action", {}) or {}
        op = action.get("op", "wait")
        name = state.get("name", "?")
        logger.info("launch_action", state=name, op=op)

        if op == "launch_xbox":
            self.xbox.launch_xbox_app()
            try:
                self.xbox.wait_for_xbox_ready(timeout_s=40.0)
                self.xbox.maximize()
            except WindowNotFoundError as exc:
                logger.warning("xbox_not_ready_yet", error=str(exc))
            return

        if op == "search":
            query = str(action.get("query")
                        or self._cfg.get("search_query", "minecraft for windows"))
            self._focus_search()
            self.finder.type_text(query, clear_first=True)
            self.finder.press_enter()
            logger.info("launch_search_submitted", query=query)
            return

        if op == "click_text":
            targets = action.get("targets", []) or []
            exclude = tuple(action.get("exclude", []) or ())
            # Reuse the SAME frame + OCR results already computed by perceive() this
            # cycle - no second screenshot, no second OCR scan. This removes the
            # ~40s duplicate OCR that previously ran inside every click_text action.
            # We OCR the bundle frame once (cache hit, since perceive() just ran it)
            # and hand the results to every target match as `precomputed`.
            if obs.frame is not None:
                frame, ox, oy = obs.frame, 0, 0
                try:
                    precomputed = self.finder._ocr.read_text_roi(  # noqa: SLF001
                        frame, use_cache=True)
                except Exception:  # noqa: BLE001
                    precomputed = None
            else:
                frame, ox, oy = self.finder.screenshot()
                precomputed = None

            # CRITICAL FIX: Ensure the CORRECT window is focused before clicking
            # Minecraft states -> focus Minecraft window
            # Xbox states -> focus Xbox window
            minecraft_states = ("mc_title", "world_select", "loading")
            xbox_states = ("game_card", "search_results", "xbox_home", "game_launching")
            if name in minecraft_states:
                self._focus_minecraft_window()
            elif name in xbox_states:
                self._focus_xbox_window()

            for t in targets:
                local = self.finder._match_in_frame(  # noqa: SLF001
                    frame, str(t), 0.35, exclude, precomputed=precomputed)
                if local is not None:
                    hit = (ox + local[0], oy + local[1])

                    # COORDINATE VALIDATION: Ensure click is within expected window bounds
                    if not self._validate_click_coordinates(hit, name):
                        logger.warning("click_coord_validation_failed", target=t, hit=hit, state=name)
                        continue

                    # ROBUST CLICK: For mc_title, use a more deliberate click
                    if name == "mc_title":
                        self._robust_click_minecraft_play(hit)
                    elif name in ("game_card", "search_results"):
                        # Xbox app buttons sometimes need slightly longer settle
                        self.finder.click_at(*hit, settle_s=1.0)
                    else:
                        self.finder.click_at(*hit)
                    logger.info("launch_clicked_text", target=t, at=list(hit))

                    # VERIFY CLICK REGISTRATION: Wait briefly then check if screen changed
                    post_ms = int(action.get("post_click_ms", 0) or 0)
                    if post_ms > 0:
                        logger.debug("post_click_settle", ms=post_ms)
                        time.sleep(min(post_ms / 1000.0, 10.0))

                    # CROSSHAIR POLLING: For title screen, poll for crosshair to confirm transition
                    if name == "mc_title":
                        if self._await_crosshair_after_click():
                            logger.info("launch_in_world_confirmed", via="post_click_crosshair_poll")
                            return

                    # VERIFY SCREEN CHANGED: Re-perceive to confirm transition
                    if self._verify_screen_transition(name):
                        logger.info("launch_click_verified_transition", from_state=name)
                        return
                    else:
                        logger.warning("launch_click_no_transition", target=t, state=name)
                        # Don't return - let the loop re-perceive and potentially retry
                        return

            logger.info("launch_click_text_missing", targets=targets)
            return

        if op == "click_world":
            self._click_world(vlm_hint=obs.vlm_action_hint)
            # click_world already has crosshair polling via _await_crosshair_after_world_enter
            return

        if op == "wait":
            ms = int(action.get("ms", 2000))
            time.sleep(min(ms / 1000.0, 10.0))
            return

        # op == "done" (or unknown): no physical action.
        return

    def _focus_search(self) -> None:
        """Focus the Xbox search box: UIA Edit -> OCR 'search' -> hotkey."""
        try:
            window = self.xbox._window()  # noqa: SLF001
            edits = window.descendants(control_type="Edit")
            if edits:
                edits[0].click_input()
                time.sleep(0.3)
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("focus_search_uia_failed", error=str(exc))
        if self.finder.find_text("search") is not None:
            if self.finder.click_text("search", timeout_s=2.0):
                return
        # Last resort: many Xbox builds focus search on typing after Ctrl+F.
        self.finder.press_key("f")

    def _click_world(self, vlm_hint: str = "") -> None:
        """Select an existing world (e.g. "My World") and enter it.

        The Bedrock world screen has a TOP nav bar ("Play | Friends |
        Marketplace") whose bare "Play" tab must NEVER be treated as the enter
        button - clicking it just re-navigates the same screen (the old bug that
        stalled here). The real confirm control is the "Play World" / "Play
        Selected World" button at the BOTTOM, and pressing Enter on a selected
        world row also enters it. So we:
          0. If the VLM named a specific world (e.g. "My World"), click that row.
          1. Click a WORLD ROW (below the "Worlds" header) to select the world.
          2. Confirm with the specific "play world"/"play selected world" button,
             else press Enter (Bedrock enters the highlighted world).
          3. Only as a last resort click a bare "play" in the LOWER portion of
             the screen (never the top nav tab).
        """
        # 0) VLM-guided: if the VLM's hint names a world, target it by name.
        for wname in self._world_names_from_hint(vlm_hint):
            hit = self.finder.find_text(wname)
            if hit is not None:
                self.finder.click_at(*hit)
                time.sleep(0.8)
                self.finder.press_enter()
                logger.info("launch_world_vlm_named", world=wname)
                return

        # 1) EXTENDED SEARCH: Try to find "my world" or other world names
        # by doing a full-frame OCR scan (not just ROI) since thumbnails
        # might be outside the default ROI.
        for wname in ("my world", "myworld", "world", "survival", "creative"):
            hit = self._find_world_name_full_frame(wname)
            if hit is not None:
                self.finder.click_at(*hit)
                time.sleep(0.8)
                self.finder.press_enter()
                logger.info("launch_world_found_by_name", world=wname, at=list(hit))
                return

        # 2) Select the first world row (just under the "Worlds" list header).
        selected_row = False
        header = self.finder.find_text("worlds")
        if header is not None:
            self.finder.click_at(header[0], header[1] + 90)
            selected_row = True
            logger.info("launch_world_row_selected", at=[header[0], header[1] + 90])
            time.sleep(1.0)

        # 3) Confirm via the explicit play-world button (unambiguous text).
        for t in ("play world", "play selected world"):
            hit = self.finder.find_text(t)
            if hit is not None:
                self.finder.click_at(*hit)
                logger.info("launch_world_play", target=t)
                return

        # 3b) If a row is selected, Enter reliably enters it in Bedrock.
        if selected_row:
            self.finder.press_enter()
            logger.info("launch_world_enter")
            return

        # 4) Last resort: a bare "play" ONLY in the lower part of the frame
        #    (avoids the top nav "Play" tab at the very top of the screen).
        lower = self._find_text_below(("play",), min_y_fraction=0.5)
        if lower is not None:
            self.finder.click_at(*lower)
            logger.info("launch_world_play_lower", at=list(lower))
            return

        # 5) No world present: create a fresh one.
        for t in ("create new world", "create new"):
            hit = self.finder.find_text(t)
            if hit is not None:
                self.finder.click_at(*hit)
                time.sleep(2.0)
                self.finder.click_text("create", timeout_s=8.0)
                logger.info("launch_world_created")
                return

        # 6) Give up gracefully: confirm the highlighted default.
        self.finder.press_enter()

    def _find_world_name_full_frame(self, query: str) -> tuple[int, int] | None:
        """Search for world name across full frame (not just ROI).
        
        World thumbnails might be outside the default interactive UI ROI.
        Does a one-time full-frame OCR scan.
        """
        try:
            frame, ox, oy = self.finder.screenshot()
            # Force full-frame OCR without ROI crop
            results = self.finder._ocr.read_text(frame, use_cache=False)
            q = query.lower()
            for r in results:
                if r.confidence < 0.35 or r.rect is None:
                    continue
                if q in r.text.lower():
                    cx, cy = r.rect.center.x, r.rect.center.y
                    return ox + cx, oy + cy
        except Exception as exc:  # noqa: BLE001
            logger.debug("full_frame_ocr_failed", error=str(exc))
        return None

    def _find_text_below(self, targets: tuple[str, ...],
                         min_y_fraction: float) -> tuple[int, int] | None:
        """Find any target text whose click point is below min_y_fraction of the
        current capture region - used to skip top-nav tabs and hit lower buttons.
        """
        region = getattr(self.finder, "region", None)
        if region is None:
            return None
        threshold_y = region.y + int(region.height * min_y_fraction)
        for t in targets:
            hit = self.finder.find_text(t)
            if hit is not None and hit[1] >= threshold_y:
                return hit
        return None

    @staticmethod
    def _world_names_from_hint(vlm_hint: str) -> list[str]:
        """Extract candidate world-name click targets from a VLM action hint.

        The VLM may say e.g. "click the 'My World' row" - we pull quoted names
        and always include the common default "my world" so we can target the
        actual world row by its label rather than a generic Play button.
        """
        names: list[str] = []
        if vlm_hint:
            import re
            names.extend(m.strip().lower()
                         for m in re.findall(r"['\"]([^'\"]{2,40})['\"]", vlm_hint))
            low = vlm_hint.lower()
            if "my world" in low and "my world" not in names:
                names.append("my world")
        if "my world" not in names:
            names.append("my world")
        return names

    # ------------------------------------------------------------------ #
    # VLM/LLM arbitration when the screen doesn't match any known state
    # ------------------------------------------------------------------ #
    def _ensure_vlm(self) -> Any | None:
        if self._vlm is not None:
            return self._vlm
        try:
            api_key = (self.config.openai_api_key
                       if self.config.llm.provider is LLMProvider.OPENAI
                       else self.config.anthropic_api_key)
            if not api_key:
                return None
            from tools.vision.vlm_engine import VLMEngine
            self._vlm = VLMEngine(provider=self.config.llm.provider,
                                  model=self.config.llm.vlm_model, api_key=api_key,
                                  max_tokens=self.config.llm.max_tokens,
                                  timeout_s=self.config.llm.request_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("launch_vlm_unavailable", error=str(exc))
            self._vlm = None
        return self._vlm

    # Map the VLM's generic screen verdict onto a declarative launch-state name.
    _VLM_TO_STATE = {
        "xbox_app": "xbox_home", "game_pass": "xbox_home",
        "main_menu": "mc_title", "menu": "mc_title",
        "game_loading": "loading", "gameplay": "in_world",
    }

    def _augment_with_vlm(self, bundle: PerceptionBundle) -> None:
        """L3: ask the VLM to read the frame, then fold its verdict into the bundle.

        The VLM sees the actual screenshot plus the L1 (OpenCV) + L2 (OCR)
        evidence as grounding context. Its screen_state re-votes the fused
        verdict (weight ``_W_VLM``) and its recommended_action_hint is stored so
        the action layer (e.g. world selection) can act on it.
        """
        # Circuit breaker: once the VLM has failed too many times in a row (e.g. a
        # bad model name returning HTTP 404/400), stop calling it for the rest of
        # the run and degrade gracefully to OCR-only. A broken VLM must never turn
        # into a runaway retry loop (the 156-call failure seen in the live run).
        if self._vlm_disabled:
            return
        vlm = self._ensure_vlm()
        if vlm is None or bundle.frame is None:
            return
        try:
            scene = vlm.analyze_scene(
                bundle.frame, game="minecraft",
                goal="Launch Minecraft for Windows and reach the in-game world.",
                ocr_texts=bundle.ocr_text.split(),
                visual_summary=(f"opencv_state={bundle.opencv_state} "
                                f"crosshair={bundle.crosshair} "
                                f"windows={bundle.window_titles[:5]}"),
            )
            self._vlm_failures = 0          # a good call resets the breaker
            bundle.vlm_scene = scene
            bundle.vlm_action_hint = scene.recommended_action_hint or ""
            logger.info("launch_vlm_scene", screen=str(scene.screen_state),
                        conf=round(scene.confidence, 2),
                        hint=bundle.vlm_action_hint[:60])
            target = self._VLM_TO_STATE.get(str(scene.screen_state))
            if target:
                # Re-fuse: add the VLM vote and, if it now wins, adopt its state.
                base = bundle.fused_confidence * (_W_OPENCV + _W_OCR)
                vlm_weight = _W_VLM * float(scene.confidence)
                if target == bundle.fused_state:
                    bundle.fused_confidence = round(
                        min((base + vlm_weight) / (_W_OPENCV + _W_OCR + _W_VLM), 1.0), 3)
                    bundle.signals_agree = True
                elif vlm_weight > base:
                    bundle.fused_state = target
                    bundle.fused_confidence = round(
                        min(vlm_weight / (_W_OPENCV + _W_OCR + _W_VLM), 1.0), 3)
        except Exception as exc:  # noqa: BLE001
            self._vlm_failures += 1
            max_fail = int(self._cfg.get("vlm_max_failures", 3))
            logger.warning("launch_vlm_failed", error=str(exc),
                           failures=self._vlm_failures)
            if self._vlm_failures >= max_fail:
                self._vlm_disabled = True
                logger.warning("launch_vlm_disabled",
                               reason=f"{self._vlm_failures} consecutive failures; "
                                      "continuing OCR-only for the rest of this run")

    def _arbitrate_with_vlm(self, obs: PerceptionBundle) -> dict[str, Any] | None:
        """Compat shim: return the declarative state the VLM points to (if any)."""
        self._augment_with_vlm(obs)
        if obs.vlm_scene is None:
            return None
        target = self._VLM_TO_STATE.get(str(obs.vlm_scene.screen_state))
        if target:
            return next((st for st in self._states
                         if st.get("name") == target), None)
        return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> bool:
        if not self._states:
            logger.error("launch_no_states")
            return False
        poll_s = float(self._cfg.get("poll_interval_s", 1.5))
        confuse_limit = int(self._cfg.get("confuse_limit", 3))
        # Timing is PROGRESS-based, not a fixed schedule. We only abort if the
        # flow STALLS (no state change) for `stall_timeout_s`; a slow-but-
        # advancing launch is never killed. `hard_cap_s` is a large safety net.
        stall_timeout_s = float(self._cfg.get("stall_timeout_s", 180))
        hard_cap_s = float(self._cfg.get("hard_cap_s", 900))
        # A state that keeps matching but never transitions (e.g. the title "Play"
        # that needs several clicks) should NOT die on the wall-clock stall timer
        # while it is still acting. Instead we bound the NUMBER of actions a single
        # state may fire before we treat it as a genuine loop.
        max_same_state = int(self._cfg.get("max_same_state_actions", 8))

        # Warm up OCR BEFORE any timer starts - the first PaddleOCR call has a
        # heavy cold-start (~40-50s) that must not eat the stall budget.
        self._warmup_ocr()

        start = time.time()
        last_progress = start
        last_state_name: str | None = None
        same_state_actions = 0
        confused = 0

        while (time.time() - start) < hard_cap_s:
            # perceive() runs L1 (OpenCV) + L2 (OCR/crosshair) + L3 (VLM when the
            # cheap signals are weak) and saves the frame to screenshots/.
            bundle = self.perceive()
            if bundle.crosshair:
                logger.info("launch_in_world_confirmed")
                return True

            # Deterministic declarative match first (OCR/window/crosshair rules).
            state = self._match_state(bundle)
            if state is None:
                # No declarative match: fall back to the VLM's fused verdict, which
                # perceive() may already have produced this cycle.
                confused += 1
                logger.info("launch_unmatched", cycle=confused,
                            fused_state=bundle.fused_state,
                            fused_conf=round(bundle.fused_confidence, 2),
                            windows=bundle.window_titles[:4],
                            ocr_sample=bundle.ocr_text[:120])
                target = self._VLM_TO_STATE.get(
                    str(bundle.vlm_scene.screen_state)) if bundle.vlm_scene else None
                target = target or bundle.fused_state
                state = next((st for st in self._states
                              if st.get("name") == target), None)
                if state is None and confused >= confuse_limit:
                    state = self._arbitrate_with_vlm(bundle)
                    confused = 0
                if state is None:
                    if (time.time() - last_progress) >= stall_timeout_s:
                        logger.warning("launch_stalled", stalled_s=round(
                            time.time() - last_progress, 1), last_state=last_state_name)
                        return False
                    time.sleep(poll_s)
                    continue
            else:
                confused = 0

            # Progress = the matched state changed since the last cycle. As long
            # as the launch keeps advancing, the stall timer keeps resetting.
            name = state.get("name")
            if name != last_state_name:
                logger.info("launch_state", state=name,
                            desc=state.get("description", "")[:60])
                last_state_name = name
                same_state_actions = 0        # new state => reset the action budget
                last_progress = time.time()
                # Drop the cached OCR text so the motion-gate on the NEXT cycle
                # cannot reuse the previous screen's text across a real
                # transition (e.g. game_card -> blank game_launching). Stale text
                # would otherwise cause a wrong declarative match / VLM decision.
                self._prev_ocr_text = ""

            op = state.get("action", {}).get("op")
            if op == "done":
                logger.info("launch_done_state", state=name)
                return True

            self._do_action(state, bundle)

            # After entering a world, the crosshair appears within seconds. Poll
            # for it CHEAPLY (crosshair pixel check only - no OCR, no VLM) so we
            # confirm success promptly instead of waiting for a full ~40-80s
            # perception cycle (which previously let the hard cap fire first).
            if op == "click_world":
                if self._await_crosshair_after_world_enter():
                    logger.info("launch_in_world_confirmed", via="post_world_poll")
                    return True

            # A pure WAIT is NOT a loop iteration - the state is intentionally
            # idling for an EXTERNAL transition (e.g. the Minecraft process
            # starting, which for Bedrock/trial routinely takes 30-90s - far
            # longer than max_same_state * poll). Charging waits against the
            # per-state action budget killed game_launching after only ~17s
            # (8 waits * ~2s) before the game window ever appeared. So waits are
            # bounded by the WALL-CLOCK stall_timeout_s instead: we do NOT
            # increment the action budget and do NOT reset last_progress, so a
            # genuinely hung transitional screen still times out after
            # stall_timeout_s (default 180s) while a slow-but-normal launch is
            # allowed to complete.
            if op == "wait":
                waited = time.time() - last_progress
                if waited >= stall_timeout_s:
                    logger.warning("launch_stalled", stalled_s=round(waited, 1),
                                   last_state=last_state_name,
                                   reason=f"waited {round(waited, 1)}s in state "
                                          f"'{name}' without a transition")
                    return False
                time.sleep(poll_s)
                continue

            # A successful non-wait action (e.g. a click) IS progress: reset the
            # wall-clock stall timer so a multi-click state (e.g. mc_title "Play")
            # is not killed mid-sequence. Genuine click loops are caught by the
            # per-state action budget below.
            same_state_actions += 1
            last_progress = time.time()
            if same_state_actions >= max_same_state:
                logger.warning("launch_stalled", stalled_s=round(
                    time.time() - start, 1), last_state=last_state_name,
                    reason=f"{same_state_actions} actions in state '{name}' "
                           "without a transition")
                return False
            time.sleep(poll_s)

        logger.warning("launch_hard_cap_reached", elapsed_s=round(time.time() - start, 1))
        return False

    def _await_crosshair_after_world_enter(self) -> bool:
        """Cheaply poll for the in-world crosshair right after entering a world.

        Uses ONLY the fast MinecraftVision crosshair pixel check (a few ms) - no
        OCR, no VLM - so it confirms the in-world transition within seconds of it
        happening instead of waiting for the next full perception cycle.
        """
        # Poll BRIEFLY for the crosshair after a world-enter click. A real enter
        # shows the crosshair within a few seconds; if it has not appeared in this
        # short window the click did NOT enter the world (e.g. it hit the wrong
        # tile), so we must RE-OBSERVE and try again rather than block. The old
        # 20*3s = 60s wait meant every FAILED world click cost a full minute -
        # 6 failed attempts = ~6 wasted minutes (the "waited too much" stall).
        # Genuinely slow world LOADING is handled separately by the main loop's
        # `loading` state (bounded by the 180s wall-clock stall timer), so a short
        # poll here does not risk missing a slow-but-successful load: the next
        # perception cycle will see the `loading` screen and keep waiting.
        polls = int(self._cfg.get("world_enter_polls", 8))
        poll_s = float(self._cfg.get("world_enter_poll_s", 1.5))
        for _ in range(max(1, polls)):
            time.sleep(poll_s)
            try:
                frame, _, _ = self.finder.screenshot()
                if frame is not None and self.vision.analyse(frame).crosshair_visible:
                    return True
            except Exception as exc:  # noqa: BLE001 - a failed poll is non-fatal
                logger.debug("world_enter_poll_failed", error=str(exc))
        return False

    def _warmup_ocr(self) -> None:
        """Trigger PaddleOCR's one-time model load before the stall timer runs."""
        try:
            warmup = getattr(self.finder._ocr, "warmup", None)  # noqa: SLF001
            if callable(warmup):
                warmup()
                logger.info("launch_ocr_warmed")
        except Exception as exc:  # noqa: BLE001
            logger.debug("launch_ocr_warmup_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Accuracy improvements: window focus, coordinate validation, click verification
    # ------------------------------------------------------------------ #

    def _focus_minecraft_window(self) -> bool:
        """Bring the Minecraft window to foreground before clicking.
        
        Returns True if successfully focused, False otherwise.
        """
        try:
            win = self.xbox.inspector.find_window(r".*Minecraft.*")
            win.set_focus()
            # Verify window is actually foreground
            import win32gui
            for _ in range(5):
                fg = win32gui.GetForegroundWindow()
                if fg == win.handle:
                    break
                time.sleep(0.1)
            logger.info("minecraft_window_focused", handle=win.handle)
            time.sleep(0.5)  # Allow focus to fully settle
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("minecraft_focus_failed", error=str(exc))
            return False

    def _focus_xbox_window(self) -> bool:
        """Bring the Xbox app window to foreground before clicking."""
        try:
            win = self.xbox._best_xbox_window()  # noqa: SLF001
            win.set_focus()
            # Verify window is actually foreground
            import win32gui
            for _ in range(5):
                fg = win32gui.GetForegroundWindow()
                if fg == win.handle:
                    break
                time.sleep(0.1)
            logger.info("xbox_window_focused", handle=win.handle)
            time.sleep(0.3)  # Allow focus to settle
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("xbox_focus_failed", error=str(exc))
            return False

    def _robust_click_minecraft_play(self, hit: tuple[int, int]) -> None:
        """More deliberate click for Minecraft title screen Play button.
        
        Minecraft's Play button sometimes needs a slightly longer press or
        the window needs extra time to process the click.
        """
        x, y = hit
        logger.debug("robust_click_play_attempt", x=x, y=y)
        # Move to position and hold slightly longer
        self.finder.click_at(x, y, settle_s=0.8)
        # Double-tap for reliability (some UWP buttons need this)
        time.sleep(0.2)
        self.finder.click_at(x, y, settle_s=0.5)
        # Also try Enter key as fallback (Play button is usually focused)
        time.sleep(0.3)
        self.finder.press_enter()
        logger.debug("robust_click_play_complete")

    def _validate_click_coordinates(self, hit: tuple[int, int], state_name: str) -> bool:
        """Validate click coordinates fall within the expected window bounds.
        
        For Minecraft states, ensures click is within the Minecraft window rect.
        For Xbox states, ensures click is within the Xbox window rect.
        """
        try:
            if state_name in ("mc_title", "world_select", "loading"):
                win = self.xbox.inspector.find_window(r".*Minecraft.*")
            else:
                win = self.xbox._best_xbox_window()  # noqa: SLF001

            rect = win.rectangle()
            x, y = hit
            # Add small margin for safety
            margin = 50
            if (rect.left - margin <= x <= rect.right + margin and
                    rect.top - margin <= y <= rect.bottom + margin):
                return True
            logger.warning("click_outside_window", hit=hit, state=state_name,
                           window_rect={"left": rect.left, "top": rect.top,
                                        "right": rect.right, "bottom": rect.bottom})
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("coord_validation_failed", error=str(exc))
            return True  # Allow click if validation fails (fail-open)

    def _await_crosshair_after_click(self) -> bool:
        """Poll for crosshair after clicking Play on title screen.
        
        Similar to _await_crosshair_after_world_enter but shorter timeout
        since title screen -> world select should be fast.
        """
        polls = int(self._cfg.get("title_click_polls", 5))
        poll_s = float(self._cfg.get("title_click_poll_s", 1.0))
        for _ in range(max(1, polls)):
            time.sleep(poll_s)
            try:
                frame, _, _ = self.finder.screenshot()
                if frame is not None and self.vision.analyse(frame).crosshair_visible:
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("title_click_poll_failed", error=str(exc))
        return False

    def _verify_screen_transition(self, from_state: str) -> bool:
        """Verify the screen actually changed after a click action.
        
        Takes a quick perception cycle and checks if the state is different
        from the expected 'from_state'.
        """
        try:
            # Quick perception without VLM
            bundle = self.perceive()
            if bundle.crosshair:
                return True  # In world = success
            
            matched = self._match_state(bundle)
            if matched is None:
                return False  # No match = uncertain
            
            new_state = matched.get("name")
            # Consider it a transition if we moved to a different state
            # or if we're in a known "next" state
            expected_next = {
                "mc_title": ("world_select", "loading", "in_world"),
                "search_results": ("game_card", "game_launching"),
                "game_card": ("game_launching", "mc_title"),
                "xbox_home": ("search_results",),
                "desktop": ("xbox_app", "xbox_home"),
            }
            
            if new_state != from_state:
                logger.info("screen_transition_detected", from_state=from_state, to_state=new_state)
                return True
            
            # Allow staying in same state if it's a valid intermediate (e.g., waiting)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("verify_transition_failed", error=str(exc))
            return False


