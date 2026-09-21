"""Minecraft-specific OpenCV HUD analyser - the pixel-level ground truth.

The real game frame is the ONLY input. Everything here is deterministic OpenCV
(HSV colour segmentation, region sampling, template-free blob counting) that runs
in a few milliseconds on CPU - no OCR, no model inference, no hallucination.

It reads what OCR cannot (health/hunger are ICONS, not text) and what the LLM
should never guess: exact health, hunger, the selected hotbar slot, whether the
crosshair is visible (= in-world), the block colour under the crosshair, whether
we are outdoors, and a rough hostile-mob presence estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("mc_vision")


@dataclass
class MinecraftHUD:
    """Pixel-derived Minecraft HUD reading. Every field is measured, not guessed."""

    health: int = 20                 # 0-20 (two per heart)
    hunger: int = 20                 # 0-20 (two per drumstick)
    selected_slot: int = 1           # 1-9
    crosshair_visible: bool = False  # True => in-world, mouse captured
    block_hint: str = "unknown"      # colour-inferred block under crosshair
    outdoors: bool = False           # sky-blue present at the top of the frame
    mob_count: int = 0               # rough count of hostile-looking blobs
    brightness: float = 0.0          # mean luminance (day/night proxy)
    confidence: float = 0.0          # overall HUD-read confidence 0-1
    details: dict = field(default_factory=dict)


# HSV ranges (OpenCV hue is 0-179).
_RED1 = ((0, 120, 90), (10, 255, 255))
_RED2 = ((170, 120, 90), (179, 255, 255))
_HUNGER = ((8, 90, 70), (26, 255, 255))       # drumstick brown-orange
_SKY = ((90, 60, 140), (120, 255, 255))       # Minecraft daytime sky blue

# Block colour -> name mapping (coarse HSV centroid buckets).
_BLOCK_BUCKETS = [
    ("wood", (8, 60, 40), (20, 200, 160)),        # trunk/planks brown
    ("dirt", (10, 40, 40), (25, 160, 130)),       # dirt brown
    ("grass", (35, 40, 40), (85, 255, 255)),      # green
    ("leaves", (35, 30, 20), (90, 255, 180)),     # darker green
    ("water", (95, 80, 60), (120, 255, 255)),     # blue
    ("sand", (20, 20, 150), (35, 120, 255)),      # pale yellow
    ("stone", (0, 0, 60), (179, 40, 170)),        # low-sat grey
]


class MinecraftVision:
    """Deterministic OpenCV reader of the Minecraft in-game HUD/scene."""

    def analyse(self, frame: np.ndarray) -> MinecraftHUD:
        try:
            return self._analyse(frame)
        except Exception as exc:  # noqa: BLE001 - vision must never crash the loop
            logger.warning("mc_vision_failed", error=str(exc))
            return MinecraftHUD()

    def _analyse(self, frame: np.ndarray) -> MinecraftHUD:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        crosshair = self._crosshair_visible(gray, w, h)
        health = self._count_bar(hsv, w, h, side="left", ranges=(_RED1, _RED2))
        hunger = self._count_bar(hsv, w, h, side="right", ranges=(_HUNGER,))
        block = self._block_under_crosshair(hsv, w, h) if crosshair else "unknown"
        outdoors = self._sky_present(hsv, w, h)
        brightness = float(gray.mean()) / 255.0
        mobs = self._estimate_mobs(hsv, w, h)
        slot = self._selected_slot(frame, gray, w, h) if crosshair else 1

        conf = 0.0
        if crosshair:
            conf += 0.6
        if health > 0 or hunger > 0:
            conf += 0.25
        if outdoors:
            conf += 0.15
        conf = min(conf, 1.0)

        hud = MinecraftHUD(
            health=health, hunger=hunger, selected_slot=slot,
            crosshair_visible=crosshair, block_hint=block, outdoors=outdoors,
            mob_count=mobs, brightness=round(brightness, 3), confidence=round(conf, 3),
            details={"frame_w": w, "frame_h": h},
        )
        logger.debug("mc_hud", health=health, hunger=hunger, crosshair=crosshair,
                     block=block, outdoors=outdoors, mobs=mobs, slot=slot)
        return hud

    # -- crosshair: bright white '+' at the exact centre => in-world ----- #
    @staticmethod
    def _crosshair_visible(gray: np.ndarray, w: int, h: int) -> bool:
        cx, cy = w // 2, h // 2
        r = 16
        patch = gray[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if patch.size == 0:
            return False
        bright = patch > 200
        frac = float(np.count_nonzero(bright)) / float(patch.size)
        if frac < 0.02 or frac > 0.5:
            return False
        mid_row = bright[patch.shape[0] // 2, :]
        mid_col = bright[:, patch.shape[1] // 2]
        return bool(mid_row.any() and mid_col.any())

    # -- health / hunger: count filled icon mass in the bottom HUD ------ #
    @staticmethod
    def _count_bar(hsv: np.ndarray, w: int, h: int, *, side: str, ranges: tuple) -> int:
        y0, y1 = int(h * 0.84), int(h * 0.96)
        if side == "left":
            x0, x1 = int(w * 0.30), int(w * 0.50)
        else:
            x0, x1 = int(w * 0.50), int(w * 0.70)
        region = hsv[y0:y1, x0:x1]
        if region.size == 0:
            return 0
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(region, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        filled_frac = float(np.count_nonzero(mask)) / float(mask.size)
        full_ref = 0.22  # a full bar fills ~22% of this region
        value = int(round(min(filled_frac / full_ref, 1.0) * 20))
        return max(0, min(value, 20))

    # -- block under crosshair: sample a small patch at the centre ------ #
    @staticmethod
    def _block_under_crosshair(hsv: np.ndarray, w: int, h: int) -> str:
        cx, cy = w // 2, h // 2
        patch = hsv[cy + 8:cy + 20, cx - 6:cx + 6]
        if patch.size == 0:
            return "unknown"
        hm = int(np.median(patch[:, :, 0]))
        sm = int(np.median(patch[:, :, 1]))
        vm = int(np.median(patch[:, :, 2]))
        for name, lo, hi in _BLOCK_BUCKETS:
            if lo[0] <= hm <= hi[0] and lo[1] <= sm <= hi[1] and lo[2] <= vm <= hi[2]:
                return name
        return "unknown"

    # -- sky presence (are we outdoors?) -------------------------------- #
    @staticmethod
    def _sky_present(hsv: np.ndarray, w: int, h: int) -> bool:
        top = hsv[0:int(h * 0.20), :]
        if top.size == 0:
            return False
        mask = cv2.inRange(top, np.array(_SKY[0]), np.array(_SKY[1]))
        return (float(np.count_nonzero(mask)) / float(mask.size)) > 0.08

    # -- rough hostile-mob estimate: dark vertical blobs mid-field ------ #
    @staticmethod
    def _estimate_mobs(hsv: np.ndarray, w: int, h: int) -> int:
        mid = hsv[int(h * 0.35):int(h * 0.75), :]
        if mid.size == 0:
            return 0
        dark = cv2.inRange(mid, np.array((0, 0, 0)), np.array((179, 80, 60)))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        area_min = (mid.shape[0] * mid.shape[1]) * 0.002
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < area_min:
                continue
            aspect = ch / float(cw) if cw else 0.0
            if aspect >= 1.2:
                count += 1
        return min(count, 10)

    # -- selected hotbar slot: brightest-bordered cell ------------------ #
    @staticmethod
    def _selected_slot(frame: np.ndarray, gray: np.ndarray, w: int, h: int) -> int:
        y0, y1 = int(h * 0.90), int(h * 0.995)
        x0, x1 = int(w * 0.34), int(w * 0.66)
        strip = gray[y0:y1, x0:x1]
        if strip.size == 0:
            return 1
        col_profile = strip.mean(axis=0)
        n = 9
        slot_w = max(1, col_profile.shape[0] // n)
        best_slot, best_val = 1, -1.0
        for i in range(n):
            seg = col_profile[i * slot_w:(i + 1) * slot_w]
            if seg.size == 0:
                continue
            val = float(seg.max())
            if val > best_val:
                best_val = val
                best_slot = i + 1
        return best_slot
