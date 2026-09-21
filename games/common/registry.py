"""Game adapter registry.

Maps a game id to its GameAdapter implementation. Adding a new game means
registering it here (and shipping its YAML skill package) - no core changes.
"""

from __future__ import annotations

from core.exceptions import ConfigError
from games.common.game_interface import GameAdapter


def get_game_adapter(game: str) -> GameAdapter:
    """Return the adapter for ``game`` (lazy import to avoid heavy deps)."""
    key = game.strip().lower()
    if key == "among_us":
        from games.among_us.adapter import AmongUsAdapter

        return AmongUsAdapter()
    if key == "minecraft":
        from games.minecraft.adapter import MinecraftAdapter

        return MinecraftAdapter()
    raise ConfigError(
        f"No game adapter registered for '{game}'.",
        context={"available": available_games()},
    )


def available_games() -> list[str]:
    return ["among_us", "minecraft"]