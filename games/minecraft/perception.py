"""Minecraft-specific perception: fuse pixel HUD (OpenCV) + OCR into state.

The crosshair (a white '+' at frame centre) is the single hard binary signal:
crosshair visible => in-world gameplay; absent => a menu. Health/hunger/slot/
block are pixel-measured (icons OCR cannot read). Menus are read from OCR text.
"""

from __future__ import annotations

import numpy as np

from core.models import PerceptionResult
from games.minecraft.state import MinecraftState
from tools.vision.minecraft_vision import MinecraftHUD, MinecraftVision

_TITLE_KEYWORDS = ["play", "singleplayer", "multiplayer", "settings", "marketplace"]
_LOADING_KEYWORDS = ["loading", "building terrain", "generating world", "mojang"]
_PAUSE_KEYWORDS = ["game menu", "resume game", "back to game", "quit to title"]
_INVENTORY_KEYWORDS = ["crafting", "inventory", "survival inventory", "recipe book"]
_WORLDSEL_KEYWORDS = ["create new world", "worlds", "play selected world"]

_vision = MinecraftVision()


def _has(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def detect_title_screen(perception: PerceptionResult) -> bool:
    return _has(perception.all_text().lower(), _TITLE_KEYWORDS)


def detect_loading(perception: PerceptionResult) -> bool:
    return _has(perception.all_text().lower(), _LOADING_KEYWORDS)


def detect_paused(perception: PerceptionResult) -> bool:
    return _has(perception.all_text().lower(), _PAUSE_KEYWORDS)


def detect_inventory(perception: PerceptionResult) -> bool:
    return _has(perception.all_text().lower(), _INVENTORY_KEYWORDS)


def _day_time(brightness: float) -> str:
    if brightness >= 0.45:
        return "day"
    if brightness >= 0.28:
        return "dusk"
    return "night"


def build_state(perception: PerceptionResult, previous: MinecraftState | None = None,
                frame: np.ndarray | None = None) -> MinecraftState:
    """Build/refresh Minecraft state from pixel HUD (if frame given) + OCR."""
    state = previous.model_copy(deep=True) if previous else MinecraftState()
    prev_health = previous.health if previous else 20
    text = perception.all_text().lower()

    # -- Pixel HUD (ground truth) -------------------------------------- #
    hud: MinecraftHUD | None = None
    if frame is not None:
        hud = _vision.analyse(frame)
        state.crosshair_visible = hud.crosshair_visible
        state.health = hud.health
        state.hunger = hud.hunger
        state.selected_hotbar_slot = hud.selected_slot
        state.block_under_crosshair = hud.block_hint
        state.outdoors = hud.outdoors
        state.mob_count = hud.mob_count
        state.brightness = hud.brightness
        state.day_time = _day_time(hud.brightness)
        state.hud_confidence = hud.confidence

    # -- Menu / phase detection ---------------------------------------- #
    on_title = detect_title_screen(perception)
    loading = detect_loading(perception)
    paused = detect_paused(perception)
    inv = detect_inventory(perception)
    world_sel = _has(text, _WORLDSEL_KEYWORDS)

    state.on_title_screen = on_title
    state.is_paused = paused
    state.inventory_open = inv

    # Crosshair is the authoritative in-world signal; text refines the phase.
    in_world = bool(hud and hud.crosshair_visible) and not (paused or inv or on_title)
    state.in_world = in_world

    if in_world:
        state.phase = "gameplay"
        state.world_loaded = True
    elif inv:
        state.phase = "gameplay"      # inventory is a gameplay sub-screen
    elif paused:
        state.phase = "gameplay"      # paused over a world
    elif loading:
        state.phase = "loading"
        state.world_loaded = False
    elif world_sel:
        state.phase = "world_select"
    elif on_title:
        state.phase = "menu"
    # else: keep previous phase (avoids flapping to launch mid-session)

    # -- Danger flags -------------------------------------------------- #
    state.critical_health = in_world and state.health <= 6
    state.under_threat = in_world and state.mob_count > 0
    state.took_damage = in_world and state.health < prev_health

    state.detected_buttons = [w for w in _TITLE_KEYWORDS if w in text]
    return state
