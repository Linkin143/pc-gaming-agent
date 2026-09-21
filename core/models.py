"""Shared Pydantic models used as structured contracts throughout PC-GAF."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.constants import (
    ActionStrategy,
    ActionType,
    KeyMode,
    MouseButton,
    PerceptionMethod,
    RecoveryStrategy,
    ScreenState,
    VerificationResult,
    VerificationStrategy,
)


class FrameworkModel(BaseModel):
    """Base model with shared configuration for every framework contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=False,
    )


class Point(FrameworkModel):
    """A 2D screen coordinate in absolute pixels."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class Rect(FrameworkModel):
    """A rectangular region of interest in absolute pixels.

    ``x``/``y`` may be negative on multi-monitor setups where secondary displays
    sit to the left/above the primary (the Windows virtual desktop origin).
    """

    x: int = Field(..., description="Left edge (may be negative on multi-monitor).")
    y: int = Field(..., description="Top edge (may be negative on multi-monitor).")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width // 2, y=self.y + self.height // 2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class BoundingBox(FrameworkModel):
    """A detected object's bounding box plus optional label/confidence."""

    rect: Rect
    label: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OCRResult(FrameworkModel):
    """A single OCR detection."""

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rect: Rect | None = None


class VisionResult(FrameworkModel):
    """Output of an OpenCV operation (matches, contours, change score)."""

    method: str
    matches: list[BoundingBox] = Field(default_factory=list)
    change_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VLMResult(FrameworkModel):
    """Structured output from a Vision-Language Model call."""

    description: str = ""
    answer: str | None = None
    classification: str | None = None
    candidates: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_provider: str | None = None


class ScreenshotRef(FrameworkModel):
    """A reference to a screenshot stored on disk (never the raw bytes)."""

    path: str
    region: Rect | None = None
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    timestamp: float = Field(default_factory=time.time)


class PerceptionResult(FrameworkModel):
    """Aggregated structured observation produced by the perception pipeline."""

    screen_state: ScreenState = ScreenState.UNKNOWN
    methods_used: list[PerceptionMethod] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    vision_results: list[VisionResult] = Field(default_factory=list)
    vlm_result: VLMResult | None = None
    detected_objects: list[BoundingBox] = Field(default_factory=list)
    screenshot_ref: ScreenshotRef | None = None
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    vision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    changed: bool = True
    timestamp: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)
    # Fused, evidence-grounded perception (populated by the perception pipeline).
    visual_features: "VisualFeatures | None" = None
    evidence: "PerceptionEvidence | None" = None

    def all_text(self) -> str:
        """Concatenate every OCR line, useful for keyword matching."""
        return "\n".join(r.text for r in self.ocr_results)


class VisualFeatures(FrameworkModel):
    """Game-agnostic visual measurements from OpenCV (deterministic ground truth).

    These are exact pixel measurements from the real game frame - the absolute
    source of truth. The planner treats them as facts, not guesses.
    """

    edge_density: float = Field(default=0.0, ge=0.0, le=1.0)
    brightness_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    text_region_density: float = Field(default=0.0, ge=0.0, le=1.0)
    center_complexity: float = Field(default=0.0, ge=0.0, le=1.0)
    ui_element_count: int = Field(default=0, ge=0)
    dominant_colors: list[str] = Field(default_factory=list)


class StructuredSceneAnalysis(FrameworkModel):
    """Schema-constrained VLM output (prevents free-text hallucination)."""

    screen_state: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    what_i_see: str = ""
    recommended_action_hint: str = ""
    visible_ui_elements: list[str] = Field(default_factory=list)
    player_visible: bool = False
    anomaly: str | None = None


class PerceptionEvidence(FrameworkModel):
    """Fused evidence from OpenCV (L1) + OCR (L2) + VLM (L3).

    This is what the planner reasons over - never the raw screenshot. Every
    field is grounded in a concrete signal so the LLM cannot invent state.
    """

    screen_state: ScreenState = ScreenState.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_features: VisualFeatures = Field(default_factory=VisualFeatures)
    ocr_texts: list[str] = Field(default_factory=list)
    scene: StructuredSceneAnalysis | None = None
    signal_agreement: bool = False
    votes: dict[str, float] = Field(default_factory=dict)
    methods_used: list[str] = Field(default_factory=list)
    screenshot_ref: ScreenshotRef | None = None


class SkillIntent(FrameworkModel):
    """The planner's structured output. Never raw keyboard/mouse calls."""

    skill: str = Field(..., description="Name of a registered skill.")
    parameters: dict[str, Any] = Field(default_factory=dict)
    target: str | None = Field(default=None, description="Optional target, e.g. a room name.")
    goal: str = Field(default="", description="High-level objective being pursued.")
    expected_result: str = Field(default="")
    reason: str = Field(default="", description="Why the planner chose this skill.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SkillParameterSpec(FrameworkModel):
    """Schema describing a single skill parameter (parsed from YAML)."""

    type: Literal["integer", "number", "string", "boolean"] = "string"
    minimum: float | None = Field(default=None, alias="min")
    maximum: float | None = Field(default=None, alias="max")
    default: Any | None = None
    choices: list[Any] | None = None
    required: bool = False

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class RetryPolicy(FrameworkModel):
    """Bounded retry configuration for a skill."""

    max_attempts: int = Field(default=2, ge=0, le=10)
    backoff_ms: int = Field(default=100, ge=0, le=5000)


class SkillInputSpec(FrameworkModel):
    """The default physical input associated with a simple skill."""

    type: Literal["keyboard", "mouse", "none"] = "none"
    key: str | None = None
    keys: list[str] | None = None
    button: MouseButton | None = None

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class SkillDefinition(FrameworkModel):
    """A fully validated skill loaded from YAML (implements the skill contract)."""

    name: str = Field(..., alias="skill")
    game: str | None = None
    category: Literal["common", "game_specific"] = "common"
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    perception_requirements: list[str] = Field(default_factory=list, alias="perception")
    parameters: dict[str, SkillParameterSpec] = Field(default_factory=dict)
    action_strategy: ActionStrategy = ActionStrategy.OBSERVE
    input: SkillInputSpec = Field(default_factory=SkillInputSpec)
    expected_result: str = ""
    verification_strategy: VerificationStrategy = VerificationStrategy.NONE
    verification: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(default=5000, ge=50, le=120000)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, alias="retry")
    recovery_strategy: list[RecoveryStrategy] = Field(default_factory=list, alias="recovery")

    model_config = ConfigDict(populate_by_name=True, extra="ignore", use_enum_values=True)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("skill name must be a non-empty string")
        return v.strip()


class ActionPlan(FrameworkModel):
    """A structured, executable command (built by the action agent, not the LLM)."""

    action_type: ActionType
    key: str | None = None
    keys: list[str] | None = None
    mode: KeyMode | None = None
    button: MouseButton | None = None
    position: Point | None = None
    end_position: Point | None = None
    duration_ms: int = Field(default=0, ge=0, le=10000)
    skill_name: str = ""
    description: str = ""


class ActionResult(FrameworkModel):
    """Outcome of executing an :class:`ActionPlan`."""

    success: bool
    action_type: str
    skill_name: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationOutcome(FrameworkModel):
    """The verifier's structured judgement."""

    result: VerificationResult
    strategy: VerificationStrategy = VerificationStrategy.NONE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    timestamp: float = Field(default_factory=time.time)


class RecoveryContext(FrameworkModel):
    """State carried between recovery attempts to enforce bounded retries."""

    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=0)
    last_error: str | None = None
    last_strategy: RecoveryStrategy | None = None
    failed_skill: str | None = None
    history: list[str] = Field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.attempt >= self.max_attempts


class UIElement(FrameworkModel):
    """A structured representation of a Windows UI Automation element."""

    name: str = ""
    control_type: str = ""
    automation_id: str = ""
    class_name: str = ""
    rect: Rect | None = None
    is_enabled: bool = True
    is_visible: bool = True
    value: str | None = None
    children_count: int = 0


class HistoryEntry(FrameworkModel):
    """A compact record of one loop iteration for short-term memory."""

    iteration: int
    skill: str = ""
    action_type: str = ""
    verification: VerificationResult | None = None
    screen_state: ScreenState | None = None
    note: str = ""
    timestamp: float = Field(default_factory=time.time)


# Resolve forward references used by PerceptionResult (VisualFeatures /
# PerceptionEvidence are declared after it).
PerceptionResult.model_rebuild()
