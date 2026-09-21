"""Enumerations and constant values shared across the PC-GAF framework."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """A string enum whose members serialise to their plain string value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ScreenState(StrEnum):
    """Generic screen/scene categories usable across games."""

    UNKNOWN = "unknown"
    DESKTOP = "desktop"
    XBOX_APP = "xbox_app"
    GAME_PASS = "game_pass"
    GAME_LOADING = "game_loading"
    MAIN_MENU = "main_menu"
    LOBBY = "lobby"
    GAMEPLAY = "gameplay"
    MENU = "menu"
    DIALOG = "dialog"
    RESULTS = "results"
    ERROR = "error"


class VerificationResult(StrEnum):
    """Outcome of comparing expected state with observed state."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


class ActionType(StrEnum):
    """The physical action categories the executor understands."""

    KEYBOARD_PRESS = "keyboard_press"
    KEYBOARD_HOLD = "keyboard_hold"
    KEYBOARD_RELEASE = "keyboard_release"
    KEY_COMBO = "key_combo"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_HOLD = "mouse_hold"   # press+hold a button for duration_ms (e.g. mining)
    MOUSE_DRAG = "mouse_drag"
    WAIT = "wait"


class KeyMode(StrEnum):
    """How a keyboard key should be actuated."""

    TAP = "tap"
    HOLD = "hold"
    RELEASE = "release"


class MouseButton(StrEnum):
    """Mouse button identifiers."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PerceptionMethod(StrEnum):
    """Ordered from cheapest to most expensive perception strategy."""

    CACHE = "cache"
    CHANGE_DETECTION = "change_detection"
    OPENCV = "opencv"
    OCR = "ocr"
    VLM = "vlm"


class RecoveryStrategy(StrEnum):
    """Recovery tactics, roughly ordered from cheap to expensive."""

    REOBSERVE = "reobserve"
    RERUN_OCR = "rerun_ocr"
    RERUN_VISION = "rerun_vision"
    ESCALATE_VLM = "escalate_vlm"
    REBUILD_STATE = "rebuild_state"
    RECALCULATE_TARGET = "recalculate_target"
    ALTERNATIVE_SKILL = "alternative_skill"
    RETRY = "retry"
    ADJUST_PARAMETERS = "adjust_parameters"
    RETURN_TO_KNOWN_STATE = "return_to_known_state"
    ABORT = "abort"


class SkillCategory(StrEnum):
    """Whether a skill is shared or game-specific."""

    COMMON = "common"
    GAME_SPECIFIC = "game_specific"


class ActionStrategy(StrEnum):
    """How the action agent should convert a skill into an ActionPlan."""

    KEYBOARD_TAP = "keyboard_tap"
    KEYBOARD_HOLD = "keyboard_hold"
    KEYBOARD_RELEASE = "keyboard_release"
    KEY_COMBO = "key_combo"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_DRAG = "mouse_drag"
    WAIT = "wait"
    OBSERVE = "observe"
    COMPOSITE = "composite"
    GAME_SPECIFIC = "game_specific"


class VerificationStrategy(StrEnum):
    """Named verification strategies referenced from YAML skills."""

    NONE = "none"
    PLAYER_POSITION_CHANGE = "player_position_change"
    TEXT_PRESENT = "text_present"
    TEXT_ABSENT = "text_absent"
    SCREEN_TRANSITION = "screen_transition"
    TASK_PROGRESS = "task_progress"
    MENU_OPEN = "menu_open"
    MENU_CLOSE = "menu_close"
    OBJECT_PRESENT = "object_present"
    STATE_EQUALS = "state_equals"


class NodeName(StrEnum):
    """LangGraph node identifiers (kept as constants to avoid typos)."""

    INITIALIZE = "initialize"
    LOAD_SKILLS = "load_skills"
    CAPTURE_SCREEN = "capture_screen"
    PERCEPTION = "perception"
    UPDATE_STATE = "update_state"
    PLANNER = "planner"
    SKILL_SELECTOR = "skill_selector"
    ACTION_GENERATOR = "action_generator"
    ACTION_VALIDATOR = "action_validator"
    EXECUTOR = "executor"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    CHECKPOINT = "checkpoint"
    FINISH = "finish"


class LLMProvider(StrEnum):
    """Supported LLM/VLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default timing/limits (overridable through configuration).
DEFAULT_ACTION_TIMEOUT_MS: int = 5000
DEFAULT_SKILL_TIMEOUT_MS: int = 8000
DEFAULT_MIN_HOLD_MS: int = 50
DEFAULT_MAX_HOLD_MS: int = 2000
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_MAX_RECOVERY_ATTEMPTS: int = 3
DEFAULT_MAX_LOOP_ITERATIONS: int = 200
DEFAULT_HISTORY_SIZE: int = 25

# Perception confidence thresholds (0.0 - 1.0).
DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = 0.75
DEFAULT_VISION_CONFIDENCE_THRESHOLD: float = 0.70
DEFAULT_CHANGE_DETECTION_THRESHOLD: float = 0.02

# WASD movement key mapping used by common navigation skills.
MOVEMENT_KEYS: dict[str, str] = {
    "move_forward": "w",
    "move_backward": "s",
    "move_left": "a",
    "move_right": "d",
}
