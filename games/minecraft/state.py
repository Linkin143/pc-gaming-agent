"""Minecraft-specific structured state stored under ``game_state``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MinecraftState(BaseModel):
    """Serialisable Minecraft state: launch flow + full in-world gameplay.

    HUD fields (health/hunger/slot/crosshair/block) are pixel-measured by
    :class:`~tools.vision.minecraft_vision.MinecraftVision` - deterministic
    ground truth. Objective/inventory fields track gameplay progress.
    """

    model_config = ConfigDict(extra="forbid")

    # -- Phase / launch-menu flow -------------------------------------- #
    phase: str = "launch"  # launch|menu|world_select|loading|gameplay
    on_title_screen: bool = False
    in_world: bool = False
    world_loaded: bool = False
    is_paused: bool = False
    inventory_open: bool = False

    # -- Live HUD (pixel-measured every frame) ------------------------- #
    health: int = 20
    hunger: int = 20
    selected_hotbar_slot: int = 1
    crosshair_visible: bool = False
    block_under_crosshair: str = "unknown"
    outdoors: bool = False
    mob_count: int = 0
    brightness: float = 0.0
    day_time: str = "unknown"  # day|night|dusk|dawn (from brightness)

    # -- Danger / status flags ----------------------------------------- #
    critical_health: bool = False
    under_threat: bool = False
    took_damage: bool = False   # health dropped vs previous frame

    # -- Gameplay objective tracking ----------------------------------- #
    gameplay_objective: str = "explore"  # explore|chop_wood|craft|build|mine|survive
    objective_progress: dict[str, int] = Field(default_factory=dict)

    # -- Inventory knowledge (from OCR on inventory screen) ------------ #
    inventory: dict[str, int] = Field(default_factory=dict)
    has_crafting_table: bool = False
    has_wooden_pickaxe: bool = False
    has_stone_pickaxe: bool = False

    # -- Telemetry ----------------------------------------------------- #
    detected_buttons: list[str] = Field(default_factory=list)
    hud_confidence: float = 0.0

    def to_game_state(self) -> dict:
        return self.model_dump(mode="json")
