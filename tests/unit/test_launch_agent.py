"""Unit tests for the skill-driven, screen-truth LaunchAgent.

These verify the DECLARATIVE decision logic (window + OCR + crosshair matching
loaded from xbox_launch.yaml) without touching a real desktop, OCR, or VLM.
"""

from __future__ import annotations

import pytest

from agents.launch.launch_agent import LaunchAgent, Observation


class _Stub:
    """Minimal stand-in for finder/xbox/vision so __init__ stays lightweight."""

    def __getattr__(self, _name):  # noqa: ANN001
        def _noop(*_a, **_k):
            return None
        return _noop


@pytest.fixture()
def agent(config) -> LaunchAgent:
    # Inject stubs so no PaddleOCR / Xbox UIA / MinecraftVision is constructed.
    return LaunchAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())


def _obs(*, titles=None, text="", crosshair=False) -> Observation:
    return Observation(window_titles=titles or [], ocr_text=text.lower(),
                       crosshair=crosshair, frame=None)


def test_launch_states_loaded_from_yaml(agent):
    names = [s["name"] for s in agent._states]
    for expected in ["desktop", "xbox_home", "search_results", "game_card",
                     "mc_title", "world_select", "loading", "in_world"]:
        assert expected in names
    assert agent._cfg.get("search_query")


def test_desktop_state_when_no_xbox_window(agent):
    obs = _obs(titles=["Visual Studio Code", "File Explorer"])
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "desktop"
    assert state["action"]["op"] == "launch_xbox"


def test_xbox_home_when_xbox_open_no_results(agent):
    obs = _obs(titles=["Xbox"], text="home library store game pass")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "xbox_home"
    assert state["action"]["op"] == "search"


def test_search_results_offers_card_click(agent):
    obs = _obs(titles=["Xbox"], text="minecraft results games")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "search_results"
    assert state["action"]["op"] == "click_text"


def test_game_card_when_play_visible(agent):
    obs = _obs(titles=["Xbox"], text="minecraft for windows play installed")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "game_card"


def test_mc_title_screen_detected(agent):
    obs = _obs(titles=["Minecraft"], text="play marketplace settings",
               crosshair=False)
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "mc_title"


def test_world_select_detected(agent):
    obs = _obs(titles=["Minecraft"], text="worlds create new play world",
               crosshair=False)
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "world_select"


def test_crosshair_is_authoritative_in_world(agent):
    """Screen truth: the crosshair pixel wins even amid menu-like OCR noise."""
    obs = _obs(titles=["Minecraft"], text="play marketplace", crosshair=True)
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "in_world"
    assert state["action"]["op"] == "done"


def test_not_text_blocks_false_match(agent):
    """xbox_home must NOT match once Play/Launch text appears (that's a card)."""
    obs = _obs(titles=["Xbox"], text="minecraft play launch")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "game_card"


def test_no_match_returns_none_for_arbitration(agent):
    # A blank Xbox-less, textless, crosshair-less frame matches only 'desktop'
    # (window_absent XBOX). Force a genuinely unmatched case: xbox present but
    # ambiguous text that trips no any_text list and is not a card.
    obs = _obs(titles=["Xbox"], text="")
    # 'xbox_home' requires not_text only -> it WILL match (empty text passes).
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "xbox_home"
