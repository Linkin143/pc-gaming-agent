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


# --------------------- mc_title multi-click stall fix --------------------- #

def test_mc_title_declares_post_click_settle(agent):
    """Regression: the Minecraft title 'Play' needs a settle delay so the next
    OCR cycle sees world-select instead of re-clicking Play."""
    mc_title = next(s for s in agent._states if s["name"] == "mc_title")
    assert int(mc_title["action"].get("post_click_ms", 0)) >= 3000


def test_same_state_action_budget_configured(agent):
    """Regression: a state that keeps matching must be bounded by an action
    budget (not only the wall-clock stall timer, which resets per action)."""
    assert int(agent._cfg.get("max_same_state_actions", 0)) >= 1


def test_wait_state_not_killed_by_action_budget(agent, monkeypatch):
    """Regression (live-run stall launch_20260926_005809): a transitional 'wait'
    state (game_launching) must NOT be killed after max_same_state_actions waits.

    Minecraft (Bedrock/trial) can take 30-90s from the Xbox Play click to the
    title screen. The old loop charged every wait against the 8-action budget, so
    game_launching died after ~17s (8 waits) before the game window appeared. A
    pure wait must instead be bounded by the WALL-CLOCK stall_timeout_s: it should
    survive far more than max_same_state waits, and time out only after the
    wall-clock budget elapses."""
    import agents.launch.launch_agent as la

    # Tight budgets so the test is fast: 3-action budget, 20s wall-clock stall.
    agent._cfg["max_same_state_actions"] = 3
    agent._cfg["stall_timeout_s"] = 20.0
    agent._cfg["hard_cap_s"] = 10_000.0
    agent._cfg["poll_interval_s"] = 0.0

    # Virtual clock: each time.time() advances 5s; time.sleep is a no-op. This
    # lets many "waits" pass quickly while the wall-clock budget still elapses.
    clock = {"t": 0.0}
    def _now():
        return clock["t"]
    def _sleep(_s):
        clock["t"] += 5.0
    monkeypatch.setattr(la.time, "time", _now)
    monkeypatch.setattr(la.time, "sleep", _sleep)
    agent._warmup_ocr = lambda: None

    # Always perceive the blank game_launching screen (Xbox present, no text, no
    # crosshair, never any Minecraft window -> never transitions).
    from agents.launch.launch_agent import PerceptionBundle
    def _perceive():
        b = PerceptionBundle(window_titles=["Xbox"], timestamp=0.0)
        b.ocr_text = ""
        b.crosshair = False
        b.frame = object()
        return b
    agent.perceive = _perceive

    waits = {"n": 0}
    def _count_do(state, obs):
        if (state.get("action", {}) or {}).get("op") == "wait":
            waits["n"] += 1
        # don't actually sleep inside the action for this test
    agent._do_action = _count_do

    result = agent.run()

    assert result is False                     # eventually stalls (game never opens)
    # It must have waited MANY more times than the 3-action budget before the
    # wall-clock stall fired - proving waits aren't charged to the action budget.
    assert waits["n"] > 3
    # And it stopped because the wall-clock budget elapsed (~20s / 5s step ≈ 4+).
    assert clock["t"] >= 20.0


def test_do_action_honours_post_click_ms(agent, monkeypatch):
    """_do_action must sleep for post_click_ms after a successful click_text so a
    slow-navigating button (mc_title Play) is given time to render the next screen."""
    import agents.launch.launch_agent as la

    # Fake finder: screenshot returns a dummy frame+origin, _match_in_frame always
    # 'finds' the target, click_at records the click.
    clicks: list[tuple[int, int]] = []

    class _F:
        region = None

        def screenshot(self):
            return (object(), 0, 0)

        def _match_in_frame(self, _frame, _t, _conf, _exclude, precomputed=None):
            return (10, 20)

        def click_at(self, x, y, **_k):
            clicks.append((x, y))

    slept: list[float] = []
    monkeypatch.setattr(la.time, "sleep", lambda s: slept.append(s))
    agent.finder = _F()

    state = {"name": "mc_title",
             "action": {"op": "click_text", "targets": ["play"],
                        "post_click_ms": 4000}}
    agent._do_action(state, _obs(titles=["Minecraft"], text="play"))

    assert clicks == [(10, 20)]
    # The 4000 ms settle (=4.0 s) must have been requested.
    assert any(abs(s - 4.0) < 1e-6 for s in slept)


# --------------------- VLM cost-control (cycle speed) --------------------- #

class _CountingVLMAgent(LaunchAgent):
    """LaunchAgent that records whether the VLM (L3) was invoked this cycle."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.vlm_calls = 0

    def _augment_with_vlm(self, bundle):  # noqa: ANN001
        self.vlm_calls += 1


def _bundle_for(agent, *, titles, text, crosshair=False):
    """Build a PerceptionBundle-like object via the real dataclass path."""
    from agents.launch.launch_agent import PerceptionBundle
    b = PerceptionBundle(window_titles=titles, timestamp=0.0)
    b.ocr_text = text.lower()
    b.crosshair = crosshair
    return b


def _run_vlm_decision(agent, bundle):
    """Replicate perceive()'s L3-trigger block against a prebuilt bundle."""
    agent._fuse_l1_l2(bundle)
    matched = agent._match_state(bundle)
    matched_op = (matched.get("action", {}) or {}).get("op") if matched else None
    vlm_threshold = float(agent._cfg.get("vlm_threshold", 0.55))
    # Simulate a present frame so the frame-is-not-None guard passes.
    bundle.frame = object()
    if bundle.frame is not None and not bundle.crosshair and (
            matched is None
            or bundle.fused_confidence < vlm_threshold
            or matched_op == "click_world"):
        agent._augment_with_vlm(bundle)


def test_vlm_skipped_on_confident_declarative_match(config):
    """A confident OCR match (e.g. game_card) must NOT fire the ~40s VLM."""
    agent = _CountingVLMAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    bundle = _bundle_for(agent, titles=["Xbox"],
                         text="minecraft for windows play installed")
    _run_vlm_decision(agent, bundle)
    assert bundle.fused_state == "game_card"
    assert agent.vlm_calls == 0


def test_vlm_fires_for_world_select(config):
    """world_select uses the VLM hint to pick a world by name, so it must fire."""
    agent = _CountingVLMAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    bundle = _bundle_for(agent, titles=["Minecraft"],
                         text="worlds create new play world")
    _run_vlm_decision(agent, bundle)
    assert bundle.fused_state == "world_select"
    assert agent.vlm_calls == 1


def test_vlm_fires_when_unmatched(config):
    """A frame matching NO declarative state must still escalate to the VLM.

    A Minecraft window whose OCR text matches none of the Minecraft states (no
    crosshair, no menu/loading/world keywords) is genuinely unmatched - the
    desktop/xbox_home fallbacks only catch non-Minecraft or XBOX windows.
    """
    agent = _CountingVLMAgent(config=config, xbox=_Stub(), finder=_Stub(), vision=_Stub())
    bundle = _bundle_for(agent, titles=["Minecraft"], text="qwerty zxcvb")
    assert agent._match_state(bundle) is None      # sanity: truly unmatched
    _run_vlm_decision(agent, bundle)
    assert agent.vlm_calls == 1


# ------------------- post-world-enter crosshair poll --------------------- #

def test_await_crosshair_after_world_enter_returns_true(agent, monkeypatch):
    """The fast post-click_world poll returns True as soon as the crosshair shows,
    without running OCR or the VLM."""
    import agents.launch.launch_agent as la

    class _HUD:
        crosshair_visible = True

    class _Vision:
        def analyse(self, _frame):
            return _HUD()

    class _F:
        region = None

        def screenshot(self):
            return (object(), 0, 0)

    agent.vision = _Vision()
    agent.finder = _F()
    agent._cfg["world_enter_polls"] = 5
    agent._cfg["world_enter_poll_s"] = 0.0
    monkeypatch.setattr(la.time, "sleep", lambda _s: None)
    assert agent._await_crosshair_after_world_enter() is True


def test_await_crosshair_times_out_without_crosshair(agent, monkeypatch):
    import agents.launch.launch_agent as la

    class _HUD:
        crosshair_visible = False

    class _Vision:
        def analyse(self, _frame):
            return _HUD()

    class _F:
        region = None

        def screenshot(self):
            return (object(), 0, 0)

    agent.vision = _Vision()
    agent.finder = _F()
    agent._cfg["world_enter_polls"] = 3
    agent._cfg["world_enter_poll_s"] = 0.0
    monkeypatch.setattr(la.time, "sleep", lambda _s: None)
    assert agent._await_crosshair_after_world_enter() is False


def test_hard_cap_raised_for_slow_launch(agent):
    """The hard cap must be generous enough that a ~16-min launch isn't cut off
    right as it enters the world."""
    assert float(agent._cfg.get("hard_cap_s", 0)) >= 1200
