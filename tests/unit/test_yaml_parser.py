"""Unit tests for YAML skill parsing and schema validation."""

from __future__ import annotations

import pytest

from core.exceptions import SkillValidationError
from core.models import SkillDefinition


def test_every_skill_defines_contract_fields(registry):
    for skill in registry.all_skills():
        assert skill.name
        assert isinstance(skill.preconditions, list)
        assert isinstance(skill.perception_requirements, list)
        assert skill.timeout_ms >= 50
        assert skill.retry_policy.max_attempts >= 0
        assert isinstance(skill.recovery_strategy, list)


def test_skill_alias_fields_parse():
    raw = {
        "skill": "demo",
        "perception": ["screenshot"],
        "retry": {"max_attempts": 3},
        "recovery": ["reobserve"],
    }
    skill = SkillDefinition.model_validate(raw)
    assert skill.name == "demo"
    assert skill.perception_requirements == ["screenshot"]
    assert skill.retry_policy.max_attempts == 3
    # use_enum_values=True stores plain string values.
    assert str(skill.recovery_strategy[0]) == "reobserve"


def test_invalid_skill_missing_name_raises():
    with pytest.raises(Exception):
        SkillDefinition.model_validate({"description": "no name"})


def test_registry_rejects_bad_timeout(tmp_path):
    from agents.skills.skill_agent import SkillRegistry

    bad = tmp_path / "common"
    bad.mkdir()
    (bad / "bad.yaml").write_text(
        "skills:\n  - skill: broken\n    timeout_ms: 10\n", encoding="utf-8"
    )
    reg = SkillRegistry(tmp_path)
    with pytest.raises(SkillValidationError):
        reg.load(None)
