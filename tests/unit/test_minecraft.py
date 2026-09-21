"""Unit tests for the Minecraft skill package and adapter."""

from __future__ import annotations

import numpy as np

from core.constants import ScreenState
from core.models import OCRResult, PerceptionResult, SkillIntent
from games.minecraft import perception as mc
from games.minecraft.actions import MinecraftActionBuilder
from games.minecraft.adapter import MinecraftAdapter


def _crosshair_frame(w: int = 960, h: int = 540) -> np.ndarray:
    """A synthetic in-world frame: a white '+' crosshair at the exact centre."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    frame[cy - 1:cy + 2, cx - 8:cx + 8] = 255
    frame[cy - 8:cy + 8, cx - 1:cx + 2] = 255
    return frame


def test_minecraft_skills_loaded(mc_registry):
    names = {s.name for s in mc_registry.all_skills()}
    # Launch flow skills
    for s in ["xbox_open_app", "xbox_maximize", "xbox_search_game",
              "xbox_select_installed_game", "xbox_click_play", "xbox_launch_minecraft"]:
        assert s in names
    # Gameplay skills
    for s in ["mc_move", "mc_jump", "mc_mine_block", "mc_start_singleplayer"]:
        assert s in names


def test_minecraft_states_parsed(mc_registry):
    states = mc_registry.states.get("states", {})
    assert "main_menu" in states
    assert "gameplay" in states


def test_adapter_window_and_search():
    adapter = MinecraftAdapter()
    assert adapter.get_game_name() == "minecraft"
    assert "Minecraft" in adapter.get_window_title_regex()
    assert adapter.get_search_name() == "Minecraft for Windows"


def test_perception_detects_title_screen():
    p = PerceptionResult(ocr_results=[OCRResult(text="Play Singleplayer Settings", confidence=0.9)])
    assert mc.detect_title_screen(p) is True


def test_in_world_requires_crosshair_not_just_text():
    """Screen truth: OCR keywords alone do NOT imply in-world; the crosshair does."""
    p = PerceptionResult(ocr_results=[OCRResult(text="Health Hunger Hotbar", confidence=0.9)])
    st_text_only = mc.build_state(p)
    assert st_text_only.in_world is False
    st_pixel = mc.build_state(p, frame=_crosshair_frame())
    assert st_pixel.crosshair_visible is True
    assert st_pixel.in_world is True
    assert st_pixel.phase == "gameplay"


def test_adapter_authoritative_screen_from_crosshair():
    adapter = MinecraftAdapter()
    assert adapter.authoritative_screen({"crosshair_visible": True}) == ScreenState.GAMEPLAY
    assert adapter.authoritative_screen({"is_paused": True}) == ScreenState.MENU
    assert adapter.authoritative_screen({}) is None


def test_action_builder_chop_wood_is_mouse_hold(mc_registry):
    builder = MinecraftActionBuilder()
    plans = builder.build(mc_registry.get("mc_chop_wood"),
                          SkillIntent(skill="mc_chop_wood"), {"duration_ms": 1500})
    assert plans
    assert str(plans[0].action_type) == "mouse_hold"
    assert plans[0].duration_ms == 1500


def test_action_builder_move(mc_registry):
    builder = MinecraftActionBuilder()
    plans = builder.build(mc_registry.get("mc_move"),
                          SkillIntent(skill="mc_move"),
                          {"direction": "forward", "duration_ms": 400})
    assert plans
    assert plans[0].key == "w"


def test_sprint_holds_w_not_shift(mc_registry):
    """Sprint in Bedrock is HOLD W (auto-sprint), never Shift+W (that is sneak)."""
    builder = MinecraftActionBuilder()
    plans = builder.build(mc_registry.get("mc_sprint_forward"),
                          SkillIntent(skill="mc_sprint_forward"),
                          {"duration_ms": 600})
    assert plans
    assert str(plans[0].action_type) == "keyboard_hold"
    assert plans[0].key == "w"
    assert plans[0].keys is None  # not a Shift+W combo


def test_look_around_uses_relative_delta(mc_registry):
    """Look must send a RELATIVE mouse delta - Minecraft ignores absolute cursor
    positioning, so the old absolute-offset move turned the camera by nothing."""
    builder = MinecraftActionBuilder()
    plans = builder.build(mc_registry.get("mc_look_around"),
                          SkillIntent(skill="mc_look_around"),
                          {"dx": 250, "dy": 0})
    assert plans
    plan = plans[0]
    assert str(plan.action_type) == "mouse_move"
    assert plan.relative is True
    assert plan.delta == (250, 0)
    assert plan.position is None  # no absolute coordinate


def test_state_builder_classifies_minecraft(mc_registry):
    from agents.perception.state_agent import StateBuilder
    builder = StateBuilder(mc_registry.states)
    ocr = [OCRResult(text="Play Singleplayer Marketplace", confidence=0.9)]
    screen, conf = builder.classify_screen(ocr)
    assert screen == ScreenState.MAIN_MENU
    assert conf > 0
