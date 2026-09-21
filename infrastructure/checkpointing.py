"""LangGraph checkpointer factory (InMemory by default; optional SQLite)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger("checkpointing")


def create_checkpointer(*, persistent: bool = False, db_path: Path | None = None) -> Any:
    if persistent:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

            path = str(db_path) if db_path else ":memory:"
            logger.info("checkpointer_sqlite", path=path)
            return SqliteSaver.from_conn_string(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite_saver_unavailable", error=str(exc))
    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("checkpointer_memory")
    return InMemorySaver()
