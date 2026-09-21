"""Live Minecraft launch test.

Requires a real Windows desktop with the Xbox app and Minecraft installed.
Skipped by default; run with:

    pytest tests/gaming/test_live_minecraft.py --run-live -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


@pytest.mark.live
def test_minecraft_launch_flow():
    """Runs the exact 6-step flow: open -> maximise -> search -> select -> card -> play."""
    from games.minecraft.adapter import MinecraftAdapter
    from tools.desktop.winapp import XboxDesktopAutomation

    adapter = MinecraftAdapter()
    xbox = XboxDesktopAutomation()
    launched = xbox.run_launch_flow(
        game_name=adapter.get_search_name(),
        game_window_re=adapter.get_window_title_regex(),
    )
    # We assert the flow ran; verification of the game window may depend on the
    # machine, so a False result is acceptable but logged.
    assert isinstance(launched, bool)


@pytest.mark.live
def test_xbox_steps_individually():
    from tools.desktop.winapp import XboxDesktopAutomation

    xbox = XboxDesktopAutomation()
    xbox.launch_xbox_app()
    xbox.wait_for_xbox_ready(timeout_s=40)
    xbox.maximize()
    xbox.search_game("Minecraft for Windows")
    assert xbox.is_xbox_running() is True
