"""Dependency-injection builder for the automation engine."""

from __future__ import annotations

from agents.execution.action_agent import ActionAgent
from agents.orchestrator.graph import GameAutomationEngine
from agents.perception.evidence_fuser import EvidenceFuser
from agents.perception.perception_agent import PerceptionAgent
from agents.planner.planner_agent import PlannerAgent
from agents.recovery.recovery_agent import RecoveryAgent
from agents.skills.skill_agent import SkillRegistry
from agents.verification.verifier import VerificationAgent
from core.config import AppConfig, get_config
from core.constants import ScreenState
from core.logger import configure_logging, get_logger
from games.common.registry import get_game_adapter
from infrastructure.checkpointing import create_checkpointer
from tools.capture.screen_capture import ScreenCapture
from tools.input.keyboard import KeyboardExecutor
from tools.input.mouse import MouseExecutor
from tools.ocr.paddle_engine import get_shared_ocr
from tools.vision.opencv_engine import OpenCVEngine
from tools.vision.vlm_engine import VLMEngine

logger = get_logger("builder")


def build_engine(*, game: str | None = None, config: AppConfig | None = None,
                 enable_ocr: bool = True, enable_vlm: bool = True,
                 dry_run: bool | None = None,
                 persistent_checkpoints: bool = False) -> GameAutomationEngine:
    config = config or get_config()
    game = game or config.game
    configure_logging(level=config.log_level, json_output=config.log_json,
                      log_file=config.logs_dir / "pcgaf.log")

    adapter = get_game_adapter(game)
    registry = SkillRegistry(config.skills_dir).load(game)

    capture = ScreenCapture(config.screenshots_dir)
    opencv = OpenCVEngine()
    ocr = (get_shared_ocr(language=config.perception.ocr_language,
                          cache_ttl_seconds=config.perception.cache_ttl_seconds)
           if enable_ocr else None)
    vlm: VLMEngine | None = None
    if enable_vlm and config.perception.enable_vlm:
        api_key = (config.openai_api_key if config.llm.provider.value == "openai"
                   else config.anthropic_api_key)
        if api_key:
            vlm = VLMEngine(provider=config.llm.provider, model=config.llm.vlm_model,
                            api_key=api_key, temperature=config.llm.temperature,
                            max_tokens=config.llm.max_tokens,
                            timeout_s=config.llm.request_timeout_s)
        else:
            logger.info("vlm_disabled_no_key")

    fuser = EvidenceFuser(state_definitions=registry.states)
    perception = PerceptionAgent(config, capture=capture, opencv=opencv, ocr=ocr, vlm=vlm,
                                 screen_candidates=[s.value for s in ScreenState],
                                 fuser=fuser, game=game,
                                 goal=adapter.get_default_goal(),
                                 window_title_re=adapter.get_window_title_regex())
    planner = PlannerAgent(config)
    action_agent = ActionAgent(game_action_builder=adapter.get_action_builder())
    verifier = VerificationAgent(ocr_threshold=config.perception.ocr_confidence_threshold)
    recovery = RecoveryAgent(max_attempts=config.max_recovery_attempts)

    dry = config.execution.dry_run if dry_run is None else dry_run
    target_window = adapter.get_window_title_regex().strip(".*")
    keyboard = KeyboardExecutor(target_window=target_window,
                                require_foreground=config.execution.require_foreground and not dry,
                                dry_run=dry)
    mouse = MouseExecutor(target_window=target_window,
                          require_foreground=config.execution.require_foreground and not dry,
                          dry_run=dry)
    checkpointer = create_checkpointer(persistent=persistent_checkpoints,
                                       db_path=config.checkpoint_db)

    engine = GameAutomationEngine(config, adapter, skill_registry=registry,
                                  perception=perception, planner=planner,
                                  action_agent=action_agent, verifier=verifier,
                                  recovery=recovery, keyboard=keyboard, mouse=mouse,
                                  checkpointer=checkpointer)
    logger.info("engine_built", game=game, dry_run=dry, ocr=enable_ocr, vlm=vlm is not None)
    return engine
