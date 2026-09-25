"""Robustness tests for the deterministic keyboard/mouse (KBM) executors.

These verify the hardening added to tools/input/{keyboard,mouse,safety}.py:
extended key map, configurable/atomic press timing, stuck-key & stuck-button
recovery, emergency-stop / foreground propagation, relative-delta clamping,
tighter bounds, post-move settle, and guaranteed button release on drag/hold.

All tests run in ``dry_run=True`` so NO real OS input is emitted; where physical
side-effects matter we inject a fake controller that records calls.
"""

from __future__ import annotations

import pytest

from core.constants import DEFAULT_MAX_HOLD_MS
from core.exceptions import EmergencyStopError, ExecutionError, ForegroundError
from core.models import ActionPlan, Point
from tools.input import safety
from tools.input.keyboard import _SPECIAL_KEYS, KeyboardExecutor
from tools.input.mouse import _MAX_REL_DELTA, MouseExecutor


class _RecordingController:
    """Records press/release/click/move calls without touching the OS."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.position = (0, 0)

    def press(self, obj):
        self.events.append(("press", obj))

    def release(self, obj):
        self.events.append(("release", obj))

    def click(self, button, count):
        self.events.append(("click", button, count))

    def move(self, dx, dy):
        self.events.append(("move", dx, dy))


@pytest.fixture(autouse=True)
def _clear_emergency_stop():
    safety.clear_emergency_stop()
    yield
    safety.clear_emergency_stop()


def _kb(**kw) -> KeyboardExecutor:
    return KeyboardExecutor(require_foreground=False, dry_run=True, **kw)


def _mouse(**kw) -> MouseExecutor:
    return MouseExecutor(require_foreground=False, dry_run=True,
                         screen_size=(1920, 1080), **kw)


# --------------------------- K1: key map -------------------------------- #

def test_extended_special_keys_present():
    for k in ("f9", "f12", "page_up", "page_down", "insert", "caps_lock",
              "num_lock", "shift_r", "ctrl_r"):
        assert k in _SPECIAL_KEYS, f"missing key alias: {k}"


def test_resolve_known_and_char_keys():
    kb = _kb()
    assert kb._resolve_key("f11") is _SPECIAL_KEYS["f11"]
    assert kb._resolve_key("w") is not None


def test_resolve_unknown_key_raises():
    with pytest.raises(ExecutionError):
        _kb()._resolve_key("totally_not_a_key")


# --------------------------- K2: tap timing ----------------------------- #

def test_tap_ms_is_configurable_and_floored():
    assert _kb(tap_ms=80).tap_ms == 80
    assert _kb(tap_ms=0).tap_ms >= 1
    assert _kb().tap_ms == 50


# ------------------- K5: atomic held-key tracking ----------------------- #

def test_press_registers_before_physical_and_release_clears():
    kb = _kb()
    key = kb._resolve_key("w")
    kb._press(key)
    assert key in kb._held
    kb._release(key)
    assert key not in kb._held


def test_release_all_clears_every_held_key():
    kb = _kb()
    for ch in ("w", "a", "s", "d"):
        kb._press(kb._resolve_key(ch))
    assert len(kb._held) == 4
    kb.release_all()
    assert kb._held == set()


# ------------- K6: abort-level exceptions propagate --------------------- #

def test_emergency_stop_propagates_from_keyboard():
    safety.trigger_emergency_stop()
    with pytest.raises(EmergencyStopError):
        _kb().execute(ActionPlan(action_type="keyboard_press", key="w"))


def test_foreground_error_propagates_from_keyboard(monkeypatch):
    monkeypatch.setattr(safety, "get_foreground_title", lambda: "Some Other App")
    monkeypatch.setattr(safety.time, "sleep", lambda _s: None)
    kb = KeyboardExecutor(target_window="Minecraft",
                          require_foreground=True, dry_run=True)
    with pytest.raises(ForegroundError):
        kb.execute(ActionPlan(action_type="keyboard_press", key="w"))


def test_ordinary_failure_returns_result_not_raises():
    res = _kb().execute(ActionPlan(action_type="keyboard_press", key=""))
    assert res.success is False
    assert res.error


# ---------------------- S2: foreground grace retry ---------------------- #

def test_validate_foreground_retry_absorbs_transient(monkeypatch):
    titles = iter(["Wrong Window", "My Minecraft Window"])
    monkeypatch.setattr(safety, "get_foreground_title", lambda: next(titles))
    monkeypatch.setattr(safety.time, "sleep", lambda _s: None)
    safety.validate_foreground("minecraft", require=True, retry_s=0.5)


def test_validate_foreground_no_retry_is_strict(monkeypatch):
    monkeypatch.setattr(safety, "get_foreground_title", lambda: "Wrong Window")
    with pytest.raises(ForegroundError):
        safety.validate_foreground("minecraft", require=True, retry_s=0.0)


# ----------------- M3: relative-delta magnitude clamp ------------------- #

def test_relative_move_clamps_extreme_delta():
    m = _mouse()
    rec = _RecordingController()
    m._controller = rec
    m.dry_run = False
    m._move_relative(999999, -999999)
    assert rec.events == [("move", _MAX_REL_DELTA, -_MAX_REL_DELTA)]


# --------------------------- M4: bounds check --------------------------- #

def test_bounds_check_rejects_far_out_of_range():
    with pytest.raises(ExecutionError):
        _mouse()._bounds_check(5000, 5000)


def test_bounds_check_allows_small_multimonitor_slack():
    m = _mouse()
    m._bounds_check(-50, -50)
    m._bounds_check(1920 + 100, 1080 + 100)


# ------------- M5: stuck-button recovery + propagation ------------------ #

def test_mouse_hold_tracks_and_releases_button():
    m = _mouse()
    rec = _RecordingController()
    m._controller = rec
    m.dry_run = False
    m.execute(ActionPlan(action_type="mouse_hold", duration_ms=10))
    presses = [e for e in rec.events if e[0] == "press"]
    releases = [e for e in rec.events if e[0] == "release"]
    assert len(presses) == len(releases) == 1
    assert m._held_buttons == set()


def test_release_all_lifts_stuck_button():
    m = _mouse()
    rec = _RecordingController()
    m._controller = rec
    m.dry_run = False
    m._press_button(m._resolve_button("left"))
    assert m._held_buttons
    m.release_all()
    assert m._held_buttons == set()
    assert any(e[0] == "release" for e in rec.events)


def test_emergency_stop_propagates_from_mouse():
    safety.trigger_emergency_stop()
    with pytest.raises(EmergencyStopError):
        _mouse().execute(ActionPlan(action_type="mouse_click",
                                    position=Point(x=10, y=10)))


# --------------------------- M6: hold clamp ----------------------------- #

def test_mouse_hold_clamped_to_max(monkeypatch):
    m = _mouse()
    m._controller = _RecordingController()
    m.dry_run = False
    slept: list[float] = []
    import tools.input.mouse as mm
    monkeypatch.setattr(mm.time, "sleep", lambda s: slept.append(s))
    # 9000ms is a valid ActionPlan duration (<=10000) but exceeds the 2000ms
    # max-hold, so it must be clamped down to DEFAULT_MAX_HOLD_MS.
    m.execute(ActionPlan(action_type="mouse_hold", duration_ms=9000))
    assert max(slept) <= DEFAULT_MAX_HOLD_MS / 1000.0 + 1e-9


# --------------------------- M2: settle delay --------------------------- #

def test_click_settles_after_move(monkeypatch):
    m = _mouse(settle_ms=30)
    m._controller = _RecordingController()
    m.dry_run = False
    slept: list[float] = []
    import tools.input.mouse as mm
    monkeypatch.setattr(mm.time, "sleep", lambda s: slept.append(s))
    m.execute(ActionPlan(action_type="mouse_click", position=Point(x=100, y=100)))
    assert any(abs(s - 0.03) < 1e-6 for s in slept)


# --------------------------- M1: drag release --------------------------- #

def test_drag_releases_button_even_if_move_fails():
    m = _mouse()
    rec = _RecordingController()
    m._controller = rec
    m.dry_run = False

    calls = {"n": 0}
    real_move = m._move

    def _flaky_move(x, y, dur):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated move failure")
        return real_move(x, y, dur)

    m._move = _flaky_move
    res = m.execute(ActionPlan(action_type="mouse_drag",
                               position=Point(x=1, y=1), end_position=Point(x=2, y=2)))
    assert res.success is False
    assert m._held_buttons == set()
    assert any(e[0] == "release" for e in rec.events)
