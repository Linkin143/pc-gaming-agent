"""Skill registry and YAML loader.

Loads common skills first, then the selected game's skills (game overrides
common). Every skill is validated against SkillDefinition. Only valid skills
whose preconditions pass the current state are exposed to the planner.
"""

from __future__ import annotations

import operator
import re
from pathlib import Path
from typing import Any

import yaml

from core.exceptions import PreconditionError, SkillNotFoundError, SkillValidationError
from core.logger import get_logger
from core.models import SkillDefinition, SkillParameterSpec
from core.state import StructuredGameState

logger = get_logger("skills")

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}
_CONDITION_RE = re.compile(r"^\s*(?P<lhs>[\w\.]+)\s*(?P<op>==|!=|>=|<=|>|<)\s*(?P<rhs>.+?)\s*$")


def _coerce_literal(value: str) -> Any:
    v = value.strip().strip('"').strip("'")
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _resolve_path(root: Any, dotted: str) -> Any:
    parts = dotted.split(".")
    if parts and parts[0] == "state":
        parts = parts[1:]
    current: Any = root
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            break
    return current


class SkillRegistry:
    """Loads, validates, and resolves skills for the planner and action agent."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillDefinition] = {}
        self._states: dict[str, Any] = {}
        self._loaded_game: str | None = None

    def load(self, game: str | None = None) -> "SkillRegistry":
        self._skills.clear()
        self._load_dir(self.skills_dir / "common", category="common")
        if game:
            game_dir = self.skills_dir / "games" / game
            self._load_dir(game_dir, category="game_specific", game=game)
            self._load_states(game_dir / "states.yaml")
            self._loaded_game = game
        logger.info("skills_loaded", game=game, count=len(self._skills),
                    skills=sorted(self._skills.keys()))
        return self

    def _load_dir(self, directory: Path, *, category: str, game: str | None = None) -> None:
        if not directory.exists():
            logger.warning("skills_dir_missing", path=str(directory))
            return
        for path in sorted(directory.glob("*.yaml")):
            if path.name == "states.yaml":
                continue
            self._load_file(path, category=category, game=game)

    def _load_file(self, path: Path, *, category: str, game: str | None) -> None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise SkillValidationError(
                f"Failed to parse YAML skill file: {path.name}",
                context={"path": str(path), "error": str(exc)},
            ) from exc
        for raw in data.get("skills", []):
            raw.setdefault("category", category)
            if game and not raw.get("game"):
                raw["game"] = game
            try:
                skill = SkillDefinition.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                raise SkillValidationError(
                    f"Invalid skill definition in {path.name}",
                    context={"skill": raw.get("skill"), "error": str(exc)},
                ) from exc
            self._skills[skill.name] = skill

    def _load_states(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            self._states = yaml.safe_load(fh) or {}

    @property
    def states(self) -> dict[str, Any]:
        return self._states

    def all_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def has(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> SkillDefinition:
        if name not in self._skills:
            raise SkillNotFoundError(
                f"Skill '{name}' is not registered.",
                context={"available": sorted(self._skills.keys())},
            )
        return self._skills[name]

    def evaluate_precondition(self, condition: str, state: StructuredGameState) -> bool:
        if "||" in condition:
            return any(self.evaluate_precondition(c, state) for c in condition.split("||"))
        if "&&" in condition:
            return all(self.evaluate_precondition(c, state) for c in condition.split("&&"))
        match = _CONDITION_RE.match(condition)
        if not match:
            logger.warning("precondition_unparsed", condition=condition)
            return True
        lhs = _resolve_path(state, match.group("lhs"))
        op = _OPERATORS[match.group("op")]
        rhs = _coerce_literal(match.group("rhs"))
        try:
            lhs_value = getattr(lhs, "value", lhs)
            return bool(op(lhs_value, rhs))
        except TypeError:
            return False

    def preconditions_met(self, skill: SkillDefinition, state: StructuredGameState) -> bool:
        return all(self.evaluate_precondition(c, state) for c in skill.preconditions)

    def list_available(self, state: StructuredGameState) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if self.preconditions_met(s, state)]

    def resolve(self, name: str, state: StructuredGameState) -> SkillDefinition:
        skill = self.get(name)
        if not self.preconditions_met(skill, state):
            unmet = [c for c in skill.preconditions if not self.evaluate_precondition(c, state)]
            raise PreconditionError(
                f"Preconditions not met for skill '{name}'.", context={"unmet": unmet},
            )
        return skill

    def apply_defaults(self, skill: SkillDefinition, params: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(params)
        for name, spec in skill.parameters.items():
            if name not in resolved or resolved[name] is None:
                if spec.default is not None:
                    resolved[name] = spec.default
            if name in resolved:
                resolved[name] = self._clamp(spec, resolved[name])
        return resolved

    @staticmethod
    def _clamp(spec: SkillParameterSpec, value: Any) -> Any:
        if spec.type in {"integer", "number"} and isinstance(value, (int, float)):
            if spec.minimum is not None and value < spec.minimum:
                value = spec.minimum
            if spec.maximum is not None and value > spec.maximum:
                value = spec.maximum
            if spec.type == "integer":
                value = int(value)
        return value
