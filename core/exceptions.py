"""Domain exception hierarchy for PC-GAF."""

from __future__ import annotations

from typing import Any


class PCGAFError(Exception):
    """Base class for all framework errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


class SkillError(PCGAFError):
    """Base class for skill-related failures."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill is not present in the registry."""


class PreconditionError(SkillError):
    """Raised when a skill's preconditions are not satisfied by current state."""


class SkillValidationError(SkillError):
    """Raised when a skill's YAML definition fails schema validation."""


class ActionError(PCGAFError):
    """Base class for action generation/execution failures."""


class ActionValidationError(ActionError):
    """Raised when an ActionPlan violates safety or skill constraints."""


class ExecutionError(ActionError):
    """Raised when the deterministic executor fails to run an action."""


class StuckKeyError(ExecutionError):
    """Raised when a held key cannot be safely released."""


class EmergencyStopError(ActionError):
    """Raised when execution is aborted by the emergency-stop signal."""


class PerceptionError(PCGAFError):
    """Base class for perception failures."""


class CaptureError(PerceptionError):
    """Raised when screen capture fails."""


class OCRError(PerceptionError):
    """Raised when the OCR engine fails."""


class VisionError(PerceptionError):
    """Raised when the OpenCV engine fails."""


class VLMError(PerceptionError):
    """Raised when the VLM engine fails or returns an unusable response."""


class VerificationError(PCGAFError):
    """Raised when verification cannot be performed."""


class RecoveryError(PCGAFError):
    """Base class for recovery failures."""


class MaxRetriesExceeded(RecoveryError):
    """Raised when bounded retries are exhausted."""


class RecoveryFailed(RecoveryError):
    """Raised when no recovery strategy succeeds."""


class WindowError(PCGAFError):
    """Base class for Windows/desktop automation failures."""


class WindowNotFoundError(WindowError):
    """Raised when a target application window cannot be located."""


class ForegroundError(WindowError):
    """Raised when the target window is not in the foreground for input."""


class UIElementNotFoundError(WindowError):
    """Raised when a UI Automation element cannot be found."""


class StateError(PCGAFError):
    """Raised when the structured state is invalid or inconsistent."""


class ConfigError(PCGAFError):
    """Raised when configuration is missing or invalid."""
