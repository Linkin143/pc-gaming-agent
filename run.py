#!/usr/bin/env python
"""PC-GAF simple runner - the one command to remember.

USAGE (from the pc-gaming-agent/ folder):

    python run.py minecraft            # run Minecraft automation
    python run.py mc                   # 'mc' is an alias for minecraft
    python run.py mc --launch          # launch, then EXPLICIT gameplay (W/A/S/D...)
    python run.py mc --launch --loop   # launch, then repeat explore-and-mine loop
    python run.py mc --launch --ai     # launch, then the LLM planner loop instead
    python run.py mc --dry-run         # no real keyboard/mouse input is sent
    python run.py mc --skills          # just list the game's skills
    python run.py mc --launch --dry-run
    python run.py among_us             # run Among Us ('au' alias also works)

Flags:
    --launch    Launch the Xbox app + game (Search -> game card -> Play) first.
                For Minecraft this is followed by the EXPLICIT deterministic
                gameplay controller (press W/S/A/D, move the mouse for the
                camera, left-click to break, right-click to place, Space to
                jump) once the game is stable - no LLM in the loop.
    --loop      Run the repeating explore-and-mine gameplay loop (with --launch).
    --ai        Use the LLM/rule planner loop after launch instead of the
                explicit controller.
    --dry-run   Run the sequence but never send real input (safe testing).
    --skills    Print the loaded skills for the game and exit.
    --goal "…"  Override the default goal for the AI run (with --ai).
    --max N     Number of gameplay loop cycles (or AI iterations with --ai).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main  # noqa: E402

ALIASES = {
    "mc": "minecraft",
    "au": "among_us",
    "minecraft": "minecraft",
    "among_us": "among_us",
}


def _print_usage() -> None:
    print("Usage: python run.py <game> [--launch] [--dry-run] [--skills]")
    print("  Games : minecraft (mc)  |  among_us (au)")
    print("  --launch    launch the Xbox app + game first")
    print("  --dry-run   run without sending real keyboard/mouse input")
    print("  --skills    list the game's skills and exit")
    print('  --goal "…"  override the run goal')


def run(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_usage()
        return 0

    game = ALIASES.get(args[0].lower(), args[0].lower())
    flags = args[1:]

    # --skills: list and exit.
    if "--skills" in flags:
        return main(["skills", "--game", game])

    # --launch: run the Xbox launch flow first; stop if it fails.
    launched = "--launch" in flags
    if launched:
        rc = main(["launch", "--game", game])
        if rc != 0:
            return rc
        flags = [f for f in flags if f != "--launch"]

    # After a Minecraft launch we hand off to the EXPLICIT, deterministic
    # gameplay controller (W/S/A/D, mouse camera, left/right click, Space) rather
    # than the LLM planner loop - it waits for the game to stabilise then fires
    # the exact inputs. Use `--ai` to force the old planner-driven `run` loop.
    use_ai = "--ai" in flags
    if game == "minecraft" and launched and not use_ai:
        play_argv = ["play", "--game", game]
        if "--dry-run" in flags:
            play_argv.append("--dry-run")
        # --max N implies the repeating loop with N cycles.
        max_cycles: str | None = None
        if "--max" in flags:
            idx = flags.index("--max")
            if idx + 1 < len(flags):
                max_cycles = flags[idx + 1]
        if "--loop" in flags or max_cycles is not None:
            play_argv.append("--loop")
            if max_cycles is not None:
                play_argv += ["--iterations", max_cycles]
        return main(play_argv)

    # Build the 'run' command (AI planner loop).
    run_argv = ["run", "--game", game]
    if "--dry-run" in flags:
        # Dry-run does real screen capture but sends no input; cap iterations so
        # it finishes quickly for smoke testing.
        run_argv += ["--dry-run", "--no-ocr", "--no-vlm", "--max-iterations", "5"]
    if "--goal" in flags:
        idx = flags.index("--goal")
        if idx + 1 < len(flags):
            run_argv += ["--goal", flags[idx + 1]]
    if "--max" in flags:
        idx = flags.index("--max")
        if idx + 1 < len(flags):
            run_argv += ["--max-iterations", flags[idx + 1]]
    return main(run_argv)


if __name__ == "__main__":
    raise SystemExit(run())
