"""Run persistence - report writing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger("persistence")


class RunReporter:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, run_id: str, report: dict[str, Any]) -> Path:
        report = {**report, "run_id": run_id, "generated_at": time.time()}
        path = self.reports_dir / f"{run_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("report_written", path=str(path))
        return path

    def read_report(self, run_id: str) -> dict[str, Any] | None:
        path = self.reports_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
