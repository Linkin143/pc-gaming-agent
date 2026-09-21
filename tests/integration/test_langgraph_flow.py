"""Integration test for the full LangGraph closed loop (Minecraft, dry-run)."""

from __future__ import annotations

from core.constants import ScreenState
from core.models import OCRResult, PerceptionResult


class FakePerception:
    """Deterministic perception reporting the Minecraft title screen."""

    def __init__(self) -> None:
        self.calls = 0

    def perceive(self, *, run_id, region=None, force_vlm=False, save_screenshot=True):
        self.calls += 1
        return PerceptionResult(
            screen_state=ScreenState.MAIN_MENU,
            ocr_results=[OCRResult(text="Play Singleplayer Settings", confidence=0.95)],
            ocr_confidence=0.95, overall_confidence=0.95, changed=True,
        )

    def reset(self):
        pass


def _build_engine(config):
    from agents.orchestrator.builder import build_engine
    config.execution.dry_run = True
    config.perception.enable_vlm = False
    engine = build_engine(game="minecraft", config=config, enable_ocr=False,
                          enable_vlm=False, dry_run=True)
    engine.perception = FakePerception()
    # Force the deterministic planner so the test never makes live LLM calls,
    # regardless of whether an API key is present in the environment.
    engine.planner._llm = None            # noqa: SLF001
    engine.planner._structured_llm = None  # noqa: SLF001
    return engine


def test_graph_runs_and_finishes(config):
    engine = _build_engine(config)
    final = engine.run(user_goal="Reach the world", max_iterations=3)
    assert final["finished"] is True
    assert final["iteration"] >= 1


def test_graph_emits_structured_logs(config):
    engine = _build_engine(config)
    final = engine.run(user_goal="Reach the world", max_iterations=2)
    nodes = {entry["node"] for entry in final.get("log", [])}
    for expected in ("initialize", "perception", "planner", "verification", "finish"):
        assert expected in nodes


def test_graph_respects_max_iterations(config):
    engine = _build_engine(config)
    final = engine.run(user_goal="Reach the world", max_iterations=1)
    assert final["iteration"] <= 2
