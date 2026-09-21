"""Vision-Language Model abstraction (OpenAI / Anthropic), used only when needed."""

from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np

from core.constants import LLMProvider
from core.exceptions import VLMError
from core.logger import get_logger
from core.models import StructuredSceneAnalysis, VLMResult

logger = get_logger("vlm")

_SCENE_SCHEMA_HINT = (
    "Respond with ONLY a JSON object (no prose, no code fences) with EXACTLY "
    "these keys: "
    '{"screen_state": one of [unknown, desktop, xbox_app, game_pass, '
    "game_loading, main_menu, lobby, gameplay, menu, dialog, results, error], "
    '"confidence": number 0..1, "what_i_see": short factual description, '
    '"recommended_action_hint": short phrase, "visible_ui_elements": list of '
    'strings, "player_visible": boolean, "anomaly": string or null}. '
    "Report ONLY what is actually visible in the image. Do not invent elements."
)


class VLMEngine:
    """Provider-agnostic vision-language model client."""

    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, model: str = "gpt-4o",
                 *, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, timeout_s: float = 60.0) -> None:
        self.provider = LLMProvider(provider)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            if self.provider is LLMProvider.OPENAI:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_s)
            else:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key, timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise VLMError("Failed to initialise VLM client.",
                           context={"provider": self.provider, "error": str(exc)}) from exc
        return self._client

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise VLMError("Failed to encode image for VLM.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def describe_scene(self, image: np.ndarray, prompt: str | None = None) -> VLMResult:
        prompt = prompt or "Describe what is happening on this game screen concisely."
        return VLMResult(description=self._call(image, prompt), confidence=0.6,
                         raw_provider=str(self.provider))

    def answer_question(self, image: np.ndarray, question: str) -> VLMResult:
        text = self._call(image, question)
        return VLMResult(description=text, answer=text, confidence=0.6,
                         raw_provider=str(self.provider))

    def classify_screen(self, image: np.ndarray, candidates: list[str]) -> VLMResult:
        options = ", ".join(candidates)
        prompt = (
            "You are classifying a video-game screenshot. "
            f"Choose exactly ONE label from this list that best matches: [{options}]. "
            "Reply with only the label."
        )
        text = self._call(image, prompt).strip().lower()
        chosen = next((c for c in candidates if c.lower() in text), None)
        scores = {c: (1.0 if c == chosen else 0.0) for c in candidates}
        return VLMResult(description=text, classification=chosen, candidates=scores,
                         confidence=0.7 if chosen else 0.0, raw_provider=str(self.provider))

    def analyze_scene(self, image: np.ndarray, *, game: str, goal: str,
                      ocr_texts: list[str], visual_summary: str,
                      prev_screen: str = "unknown") -> StructuredSceneAnalysis:
        """Interpret the real frame into a schema-constrained scene analysis.

        The VLM is given the actual screenshot PLUS grounding context (OpenCV
        summary + OCR text + prior state). Output is parsed into a strict schema
        so the model cannot emit free-text hallucinations.
        """
        game_hint = ""
        if game and game.lower() == "minecraft":
            game_hint = (
                "\nThis is Minecraft. Key cues: a white '+' crosshair at the exact "
                "centre means we are IN-WORLD (screen_state=gameplay). Hearts "
                "(bottom-left) = health, drumsticks (bottom-right) = hunger, a "
                "row of 9 slots at the bottom centre = the hotbar. Buttons like "
                "'Play', 'Singleplayer', 'Marketplace' mean a menu; 'Create New "
                "World' means the world-select screen; 'Resume Game'/'Quit to "
                "Title' means the pause menu. In recommended_action name the "
                "nearest useful thing (e.g. 'chop the tree ahead', 'mine stone').\n"
            )
        context = (
            f"Game: {game}\n"
            f"Goal: {goal}\n"
            f"Previous screen state: {prev_screen}\n"
            f"OpenCV visual summary: {visual_summary}\n"
            f"OCR text detected on screen: {ocr_texts[:40]}\n"
            f"{game_hint}\n"
            "Using the image as the source of truth (the OpenCV/OCR values are "
            "corroborating evidence), analyse the current screen.\n"
            + _SCENE_SCHEMA_HINT
        )
        raw = self._call(image, context)
        return self._parse_scene(raw)

    @staticmethod
    def _parse_scene(raw: str) -> StructuredSceneAnalysis:
        text = raw.strip()
        # Strip accidental code fences.
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        # Extract the first {...} block.
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            return StructuredSceneAnalysis.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - never trust unparsed model output
            logger.warning("scene_parse_failed", error=str(exc), raw=raw[:200])
            return StructuredSceneAnalysis(screen_state="unknown", confidence=0.0,
                                           what_i_see=raw[:160])

    def _call(self, image: np.ndarray, prompt: str) -> str:
        client = self._ensure_client()
        b64 = self.encode_image(image)
        try:
            if self.provider is LLMProvider.OPENAI:
                return self._call_openai(client, b64, prompt)
            return self._call_anthropic(client, b64, prompt)
        except VLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VLMError("VLM request failed.",
                           context={"provider": self.provider, "error": str(exc)}) from exc

    def _call_openai(self, client: Any, b64: str, prompt: str) -> str:
        resp = client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_anthropic(self, client: Any, b64: str, prompt: str) -> str:
        # `temperature` is deprecated on newer Claude models (claude-sonnet-5,
        # haiku-4-5+) and returns HTTP 400 if sent; omit it for Anthropic.
        resp = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": b64}},
            ]}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
