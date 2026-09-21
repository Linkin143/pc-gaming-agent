"""Structured JSON logging for PC-GAF (structlog-based)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    log_file: Path | None = None,
) -> None:
    """Configure structlog + stdlib logging once at startup."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(format="%(message)s", level=log_level, handlers=handlers)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "pcgaf") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class RunLogger:
    """Binds run-level context (run_id/game) and offers typed event helpers."""

    def __init__(self, run_id: str, game: str, name: str = "pcgaf") -> None:
        self._log = get_logger(name).bind(run_id=run_id, game=game)
        self.run_id = run_id
        self.game = game

    def bind(self, **kwargs: Any) -> "RunLogger":
        clone = RunLogger.__new__(RunLogger)
        clone._log = self._log.bind(**kwargs)
        clone.run_id = self.run_id
        clone.game = self.game
        return clone

    def event(self, event: str, **fields: Any) -> None:
        self._log.info(event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log.warning(event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log.error(event, **fields)

    def debug(self, event: str, **fields: Any) -> None:
        self._log.debug(event, **fields)

    def agent_decision(self, agent: str, node: str, **fields: Any) -> None:
        self.event("agent_decision", agent=agent, node=node, **fields)

    def state_transition(self, from_state: str, to_state: str, **fields: Any) -> None:
        self.event("state_transition", from_state=from_state, to_state=to_state, **fields)

    def skill_selected(self, skill: str, **fields: Any) -> None:
        self.event("skill_selected", skill=skill, **fields)

    def action_plan(self, action: str, **fields: Any) -> None:
        self.event("action_plan", action=action, **fields)

    def physical_input(self, action: str, latency_ms: float, **fields: Any) -> None:
        self.event("physical_input", action=action, latency_ms=latency_ms, **fields)

    def screenshot(self, ref: str, **fields: Any) -> None:
        self.event("screenshot", screenshot_ref=ref, **fields)

    def ocr_result(self, confidence: float, **fields: Any) -> None:
        self.event("ocr_result", confidence=confidence, **fields)

    def vision_result(self, confidence: float, **fields: Any) -> None:
        self.event("vision_result", confidence=confidence, **fields)

    def verification_result(self, result: str, **fields: Any) -> None:
        self.event("verification_result", result=result, **fields)

    def recovery_attempt(self, attempt: int, strategy: str, **fields: Any) -> None:
        self.event("recovery_attempt", attempt=attempt, strategy=strategy, **fields)

    def final_outcome(self, result: str, **fields: Any) -> None:
        self.event("final_outcome", result=result, **fields)
