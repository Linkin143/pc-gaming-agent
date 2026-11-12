"""PC-GAF command-line entry point.

Examples:
    python main.py skills --game minecraft
    python main.py launch --game minecraft
    python main.py run --game minecraft --goal "Reach the world" --dry-run

Tip: the simplest way is `python run.py minecraft` (see run.py).
"""

from __future__ import annotations

import argparse
import sys

from core.config import get_config
from core.logger import configure_logging, get_logger

logger = get_logger("main")


def _cmd_skills(args: argparse.Namespace) -> int:
    from agents.skills.skill_agent import SkillRegistry

    config = get_config()
    registry = SkillRegistry(config.skills_dir).load(args.game)
    print(f"Loaded {len(registry.all_skills())} skills for '{args.game}':")
    for skill in sorted(registry.all_skills(), key=lambda s: s.name):
        print(f"  - {skill.name:26s} [{skill.category}] {skill.description[:56]}")
    return 0


def _cmd_launch(args: argparse.Namespace) -> int:
    from games.common.registry import get_game_adapter

    adapter = get_game_adapter(args.game)
    window_re = adapter.get_window_title_regex()

    # Minecraft uses the skill-driven, screen-truth LaunchAgent: it reads the
    # declarative launch state-machine from xbox_launch.yaml and decides every
    # action from the LIVE screen (OpenCV crosshair + OCR), escalating to the
    # VLM/LLM only when the frame matches no known state. No hard-coded timing.
    if args.game == "minecraft":
        from agents.launch.launch_agent import LaunchAgent

        ok = LaunchAgent().run()
        print(f"Minecraft launch {'succeeded' if ok else 'incomplete'} "
              f"(screen-truth skill-driven).")
        return 0 if ok else 2

    from tools.desktop.winapp import XboxDesktopAutomation

    search_name = args.name or adapter.get_search_name()
    xbox = XboxDesktopAutomation()
    ok = xbox.run_launch_flow(game_name=search_name, game_window_re=window_re)
    print(f"Launch {'succeeded' if ok else 'not verified'} for {args.game} "
          f"(searched '{search_name}').")
    return 0 if ok else 2


def _cmd_run(args: argparse.Namespace) -> int:
    from agents.orchestrator.builder import build_engine

    config = get_config()
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    engine = build_engine(game=args.game, config=config, enable_ocr=not args.no_ocr,
                          enable_vlm=not args.no_vlm, dry_run=args.dry_run,
                          persistent_checkpoints=args.persist)
    goal = args.goal or engine.adapter.get_default_goal()
    # Pull the live game window to the foreground before the loop starts. After a
    # `--launch`, the terminal/IDE often still owns focus; without this the very
    # first action raises ForegroundError and the run stalls before any gameplay.
    if not args.dry_run:
        from tools.input.safety import focus_window

        window_sub = engine.adapter.get_window_title_regex().strip(".*")
        if focus_window(window_sub):
            import time as _t
            _t.sleep(1.0)  # let Windows settle the foreground before input
            logger.info("game_window_focused", window=window_sub)
    logger.info("run_start", game=args.game, goal=goal, dry_run=args.dry_run)
    final = engine.run(user_goal=goal, max_iterations=config.max_iterations)
    print(f"Finished: {final.get('finished')} | reason: {final.get('finish_reason')} "
          f"| iterations: {final.get('iteration')}")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    """Explicit, deterministic Minecraft gameplay (no LLM planner loop).

    Waits until the game is stable (loading done + in-world crosshair), then
    fires the exact mechanics directly: W/S/A/D movement, relative-mouse camera,
    left-click break, right-click place, Space jump.
    """
    if args.game != "minecraft":
        print("The 'play' command currently supports only Minecraft.")
        return 2

    from agents.orchestrator.builder import build_minecraft_gameplay_controller
    from tools.input.safety import focus_window

    config = get_config()
    controller = build_minecraft_gameplay_controller(config=config, dry_run=args.dry_run)

    # Make sure the game window owns focus before we send any input.
    if not args.dry_run:
        import time as _t
        if focus_window("Minecraft"):
            _t.sleep(1.0)
            logger.info("game_window_focused", window="Minecraft")

    if args.loop:
        ok = controller.run_loop(iterations=args.iterations, wait_stable=not args.no_wait)
    else:
        ok = controller.run_demo_sequence(wait_stable=not args.no_wait)
    print(f"Gameplay {'completed' if ok else 'did not run (game not stable)'}.")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pc-gaming-agent", description=__doc__)
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_skills = sub.add_parser("skills", help="List loaded skills for a game.")
    p_skills.add_argument("--game", default="minecraft")
    p_skills.set_defaults(func=_cmd_skills)

    p_launch = sub.add_parser("launch", help="Launch a game via the Xbox app.")
    p_launch.add_argument("--game", default="minecraft")
    p_launch.add_argument("--name", default=None, help="Display name to search for.")
    p_launch.add_argument("--no-gameplay", action="store_true",
                          help="Stop after launch; do not navigate into gameplay.")
    p_launch.set_defaults(func=_cmd_launch)

    p_run = sub.add_parser("run", help="Run the closed-loop automation engine.")
    p_run.add_argument("--game", default="minecraft")
    p_run.add_argument("--goal", default=None)
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--no-ocr", action="store_true")
    p_run.add_argument("--no-vlm", action="store_true")
    p_run.add_argument("--persist", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_play = sub.add_parser(
        "play", help="Explicit deterministic Minecraft gameplay (no LLM loop).")
    p_play.add_argument("--game", default="minecraft")
    p_play.add_argument("--dry-run", action="store_true",
                        help="Run the sequence but send no real input.")
    p_play.add_argument("--loop", action="store_true",
                        help="Repeat an explore-and-mine loop instead of the one-shot demo.")
    p_play.add_argument("--iterations", type=int, default=20,
                        help="Loop cycles when --loop is set (default 20).")
    p_play.add_argument("--no-wait", action="store_true",
                        help="Skip the stability gate and act immediately.")
    p_play.set_defaults(func=_cmd_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Declare DPI awareness FIRST - before any screen capture or input - so mss
    # captures and pynput clicks share one physical-pixel coordinate space. On a
    # scaled display this is what makes OCR bounding boxes map 1:1 to click points
    # (previously the mismatch clicked ~198px off the Minecraft tile).
    from core.dpi import set_dpi_awareness

    set_dpi_awareness()

    parser = build_parser()
    args = parser.parse_args(argv)
    config = get_config()
    configure_logging(level=args.log_level or config.log_level,
                      json_output=config.log_json,
                      log_file=config.logs_dir / "pcgaf.log")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
