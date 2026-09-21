"""LangChain structured tools exposing deterministic capabilities.

These wrap the framework's deterministic engines as LangChain ``StructuredTool``
objects with strict Pydantic input schemas and JSON-compatible outputs. Physical
input tools are isolated from planner reasoning: they are provided here for
completeness/agentic experimentation but the orchestrator drives execution
through the deterministic executors, not through free-form tool calls.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.skills.skill_agent import SkillRegistry
from core.models import ActionResult, Rect
from core.state import StructuredGameState
from tools.capture.screen_capture import ScreenCapture
from tools.input.keyboard import KeyboardExecutor
from tools.input.mouse import MouseExecutor
from tools.ocr.paddle_engine import PaddleOCREngine


# --------------------------------------------------------------------------- #
# Input schemas
# --------------------------------------------------------------------------- #
class CaptureInput(BaseModel):
    region: Rect | None = Field(default=None, description="Optional ROI to capture.")


class OCRInput(BaseModel):
    query: str | None = Field(default=None, description="Optional text to search for.")


class ListSkillsInput(BaseModel):
    only_available: bool = Field(default=True)


class KeyboardInput(BaseModel):
    key: str
    hold_ms: int = Field(default=0, ge=0, le=2000)


class MouseInput(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    button: str = "left"
    double: bool = False


class LangChainToolset:
    """Factory that builds the framework's LangChain tools bound to live engines."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        capture: ScreenCapture,
        ocr: PaddleOCREngine | None,
        keyboard: KeyboardExecutor,
        mouse: MouseExecutor,
        state_provider: Any,
        run_id: str = "toolset",
    ) -> None:
        self.registry = registry
        self.capture = capture
        self.ocr = ocr
        self.keyboard = keyboard
        self.mouse = mouse
        self.state_provider = state_provider
        self.run_id = run_id

    # -- tool implementations ------------------------------------------- #
    def _capture_screen(self, region: Rect | None = None) -> dict[str, Any]:
        result = (
            self.capture.capture_region(region) if region else self.capture.capture_full()
        )
        ref = self.capture.save(result, run_id=self.run_id, tag="tool")
        return ref.model_dump(mode="json")

    def _read_screen_text(self, query: str | None = None) -> dict[str, Any]:
        if self.ocr is None:
            return {"error": "OCR disabled"}
        frame = self.capture.capture_full().image
        results = self.ocr.read_text(frame)
        payload = [r.model_dump(mode="json") for r in results]
        if query:
            payload = [r for r in payload if query.lower() in r["text"].lower()]
        return {"results": payload, "count": len(payload)}

    def _list_available_skills(self, only_available: bool = True) -> dict[str, Any]:
        state: StructuredGameState = self.state_provider()
        skills = (
            self.registry.list_available(state)
            if only_available
            else self.registry.all_skills()
        )
        return {"skills": [s.name for s in skills]}

    def _get_game_state(self) -> dict[str, Any]:
        state: StructuredGameState = self.state_provider()
        return state.model_dump(mode="json")

    def _execute_keyboard_action(self, key: str, hold_ms: int = 0) -> dict[str, Any]:
        from core.constants import ActionType, KeyMode
        from core.models import ActionPlan

        plan = ActionPlan(
            action_type=ActionType.KEYBOARD_HOLD if hold_ms else ActionType.KEYBOARD_PRESS,
            key=key,
            mode=KeyMode.HOLD if hold_ms else KeyMode.TAP,
            duration_ms=hold_ms,
        )
        result: ActionResult = self.keyboard.execute(plan)
        return result.model_dump(mode="json")

    def _execute_mouse_action(
        self, x: int, y: int, button: str = "left", double: bool = False
    ) -> dict[str, Any]:
        from core.constants import ActionType, MouseButton
        from core.models import ActionPlan, Point

        plan = ActionPlan(
            action_type=ActionType.MOUSE_DOUBLE_CLICK if double else ActionType.MOUSE_CLICK,
            position=Point(x=x, y=y),
            button=MouseButton(button),
        )
        result = self.mouse.execute(plan)
        return result.model_dump(mode="json")

    # -- assembly -------------------------------------------------------- #
    def build_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._capture_screen, name="capture_screen",
                description="Capture the screen (or a region); returns a screenshot reference.",
                args_schema=CaptureInput,
            ),
            StructuredTool.from_function(
                func=self._read_screen_text, name="read_screen_text",
                description="Run OCR on the current screen; optionally filter by query text.",
                args_schema=OCRInput,
            ),
            StructuredTool.from_function(
                func=self._list_available_skills, name="list_available_skills",
                description="List skills whose preconditions currently pass.",
                args_schema=ListSkillsInput,
            ),
            StructuredTool.from_function(
                func=self._get_game_state, name="get_game_state",
                description="Return the current structured game state as JSON.",
            ),
            StructuredTool.from_function(
                func=self._execute_keyboard_action, name="execute_keyboard_action",
                description="Execute a validated keyboard action (isolated executor).",
                args_schema=KeyboardInput,
            ),
            StructuredTool.from_function(
                func=self._execute_mouse_action, name="execute_mouse_action",
                description="Execute a validated mouse action (isolated executor).",
                args_schema=MouseInput,
            ),
        ]
