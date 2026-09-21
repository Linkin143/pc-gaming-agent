"""Configuration for PC-GAF (defaults -> YAML -> environment)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    DEFAULT_CHANGE_DETECTION_THRESHOLD,
    DEFAULT_MAX_LOOP_ITERATIONS,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    DEFAULT_VISION_CONFIDENCE_THRESHOLD,
    LLMProvider,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class PerceptionConfig(BaseModel):
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD
    vision_confidence_threshold: float = DEFAULT_VISION_CONFIDENCE_THRESHOLD
    change_detection_threshold: float = DEFAULT_CHANGE_DETECTION_THRESHOLD
    enable_vlm: bool = True
    vlm_escalation_confidence: float = 0.6
    ocr_language: str = "en"
    cache_ttl_seconds: float = 2.0


class ExecutionConfig(BaseModel):
    require_foreground: bool = True
    default_tap_ms: int = 60
    default_hold_ms: int = 300
    min_hold_ms: int = 50
    max_hold_ms: int = 2000
    mouse_move_duration_ms: int = 120
    emergency_stop_hotkey: str = "esc+esc+esc"
    dry_run: bool = False


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    planner_model: str = "gpt-4o-mini"
    vlm_model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout_s: float = 60.0


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PCGAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, v: Any) -> Any:
        # Treat empty / whitespace-only keys in .env as "not set".
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip() if isinstance(v, str) else v

    game: str = "among_us"
    max_iterations: int = DEFAULT_MAX_LOOP_ITERATIONS
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS
    log_level: str = "INFO"
    log_json: bool = True

    screenshots_dir: Path = PROJECT_ROOT / "screenshots"
    logs_dir: Path = PROJECT_ROOT / "logs"
    reports_dir: Path = PROJECT_ROOT / "reports"
    skills_dir: Path = PROJECT_ROOT / "skills"
    checkpoint_db: Path = PROJECT_ROOT / "reports" / "checkpoints.sqlite"

    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def ensure_dirs(self) -> None:
        for path in (self.screenshots_dir, self.logs_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_yaml_overrides(base: AppConfig) -> AppConfig:
    app_yaml = _load_yaml(CONFIG_DIR / "app.yaml")
    perception_yaml = _load_yaml(CONFIG_DIR / "perception.yaml")
    execution_yaml = _load_yaml(CONFIG_DIR / "execution.yaml")

    data = base.model_dump()
    data.update({k: v for k, v in app_yaml.items() if v is not None})
    if perception_yaml:
        data["perception"] = {**data.get("perception", {}), **perception_yaml}
    if execution_yaml:
        data["execution"] = {**data.get("execution", {}), **execution_yaml}
    return AppConfig.model_validate(data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    base = AppConfig()  # type: ignore[call-arg]
    config = _merge_yaml_overrides(base)
    config.ensure_dirs()
    return config


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()
