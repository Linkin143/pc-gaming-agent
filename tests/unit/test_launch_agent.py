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


def test_game_card_requires_minecraft_and_play(agent):
    """A real card shows BOTH 'minecraft' AND a Play/Launch button."""
    obs = _obs(titles=["Xbox"], text="minecraft for windows play installed")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "game_card"


def test_xbox_home_play_tiles_do_not_false_match_game_card(agent):
    """Xbox home page has 'Play' tiles but no Minecraft card -> NOT game_card."""
    obs = _obs(titles=["Xbox"], text="home store library play halo forza")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "xbox_home"


def test_blank_transition_screen_waits_not_searches(agent):
    """After clicking Play the Xbox screen briefly blanks -> wait, don't search."""
    obs = _obs(titles=["Xbox"], text="")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "game_launching"
    assert state["action"]["op"] == "wait"


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
    """A frame with minecraft + play is a card, not search_results/home."""
    obs = _obs(titles=["Xbox"], text="minecraft play launch")
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "game_card"


def test_search_results_page_not_matched_as_game_card(agent):
    """Regression (live-run stall): the Xbox search-results page shows
    'minecraft' AND a 'Play with Game Pass' heading, which used to false-match
    game_card and loop forever clicking a non-navigating heading. The
    'search results for' text must route it to search_results instead."""
    obs = _obs(
        titles=["Xbox"],
        text=("search results for minecraft  minecraft for windows  "
              "minecraft launcher  play with game pass"),
    )
    state = agent._match_state(obs)
    assert state is not None and state["name"] == "search_results"
    # And it should try to open the real tile, never click the heading.
    assert state["action"]["op"] == "click_text"
    assert "minecraft for windows" in state["action"]["targets"]


def test_game_pass_heading_excluded_from_play_click(agent):
    """The game_card click must not target the 'Play with Game Pass' heading."""
    game_card = next(s for s in agent._states if s["name"] == "game_card")
    assert "with game pass" in game_card["action"]["exclude"]


class _RecordingFinder:
    """Fake ScreenFinder recording clicks/keys, with scriptable OCR hits."""

    def __init__(self, hits: dict, region=None) -> None:
        self._hits = hits          # text -> (x, y) or None
        self.region = region
        self.clicks: list[tuple[int, int]] = []
        self.keys: list[str] = []

    def find_text(self, query, **_kw):  # noqa: ANN001
        return self._hits.get(query.lower())

    def click_at(self, x, y, **_kw):  # noqa: ANN001
        self.clicks.append((int(x), int(y)))

    def press_enter(self):
        self.keys.append("enter")

    def click_text(self, query, **_kw):  # noqa: ANN001
        hit = self._hits.get(query.lower())
        if hit is not None:
            self.clicks.append(tuple(hit))
            return True
        return False


def _make_agent(config, finder):
    return LaunchAgent(config=config, xbox=_Stub(), finder=finder, vision=_Stub())


def test_click_world_selects_row_then_enters_not_top_nav(config):
    """The bug fix: select the 'My World' row then Enter - never the top 'Play' tab."""
    # OCR sees the "Worlds" header at y=150 and a top-nav "Play" tab at y=20.
    finder = _RecordingFinder(hits={"worlds": (300, 150), "play": (915, 20)})
    agent = _make_agent(config, finder)
    agent._click_world()
    # Clicked the world row (header_y + 90 = 240), NOT the top-nav play at y=20.
    assert (300, 240) in finder.clicks
    assert (915, 20) not in finder.clicks
    # Confirmed entry with Enter (row was selected).
    assert "enter" in finder.keys


def test_click_world_prefers_play_world_button(config):
    """When an explicit 'Play World' button exists, click it directly."""
    finder = _RecordingFinder(hits={"worlds": (300, 150),
                                    "play world": (500, 800),
                                    "play": (915, 20)})
    agent = _make_agent(config, finder)
    agent._click_world()
    assert (500, 800) in finder.clicks
    assert (915, 20) not in finder.clicks


def test_click_world_uses_vlm_named_world(config):
    """When the VLM names the world, click that row by its label directly."""
    finder = _RecordingFinder(hits={"my world": (400, 300), "worlds": (300, 150)})
    agent = _make_agent(config, finder)
    agent._click_world(vlm_hint="click the 'My World' row to enter it")
    assert (400, 300) in finder.clicks
    assert "enter" in finder.keys


# ----------------------------- L1 OpenCV vote ----------------------------- #

def test_opencv_classifies_menu_and_gameplay():
    from core.models import VisualFeatures
    from tools.vision.opencv_engine import classify_screen_state

    menu = VisualFeatures(motion_score=0.005, ui_element_count=5,
                          edge_density=0.2, brightness_mean=0.4)
    state, conf = classify_screen_state(menu)
    assert state == "menu" and conf > 0.0

    gameplay = VisualFeatures(motion_score=0.15, center_complexity=0.2)
    state, conf = classify_screen_state(gameplay)
    assert state == "gameplay" and conf > 0.5

    loading = VisualFeatures(motion_score=0.0, brightness_mean=0.02,
                             edge_density=0.01)
    state, _ = classify_screen_state(loading)
    assert state == "loading"


# ------------------------- L1 + L2 fusion verdict ------------------------- #

def test_fusion_agreement_boosts_confidence(config):
    """When OpenCV (menu->mc_title) and OCR (mc_title) agree, confidence rises."""
    from core.models import VisualFeatures

    agent = LaunchAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    bundle = _obs(titles=["Minecraft"], text="play marketplace settings")
    # Force an OpenCV 'menu' vote (maps to mc_title, same as the OCR match).
    bundle.visual_features = VisualFeatures(motion_score=0.0, ui_element_count=6)
    bundle.opencv_state, bundle.opencv_confidence = "menu", 0.55
    agent._fuse_l1_l2(bundle)
    assert bundle.fused_state == "mc_title"
    assert bundle.signals_agree is True
    assert bundle.fused_confidence > 0.8  # agreement bonus applied


def test_fusion_ocr_only_when_opencv_unknown(config):
    agent = LaunchAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    bundle = _obs(titles=["Minecraft"], text="worlds create new my world")
    bundle.opencv_state, bundle.opencv_confidence = "unknown", 0.2
    agent._fuse_l1_l2(bundle)
    assert bundle.fused_state == "world_select"
    assert bundle.signals_agree is False
