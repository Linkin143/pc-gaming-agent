"""Shared pytest fixtures and path setup for PC-GAF tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.skills.skill_agent import SkillRegistry  # noqa: E402
from core.config import reload_config  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", default=False,
                     help="Run tests marked @pytest.mark.live (need a real desktop/game).")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: requires a real desktop/game.")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    run_live = (config.getoption("--run-live", default=False)
                or os.environ.get("PCGAF_RUN_LIVE", "").strip() == "1")
    if run_live:
        return
    skip = pytest.mark.skip(reason="live test; pass --run-live or PCGAF_RUN_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def config():
    cfg = reload_config()
    cfg.execution.dry_run = True
    cfg.perception.enable_vlm = False
    return cfg


@pytest.fixture()
def registry(config) -> SkillRegistry:
    return SkillRegistry(config.skills_dir).load("among_us")


@pytest.fixture()
def mc_registry(config) -> SkillRegistry:
    return SkillRegistry(config.skills_dir).load("minecraft")
