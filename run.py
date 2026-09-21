#!/usr/bin/env python
"""PC-GAF simple runner - the one command to remember.

USAGE (from the pc-gaming-agent/ folder):

    python run.py minecraft            # run Minecraft automation
    python run.py mc                   # 'mc' is an alias for minecraft
    python run.py mc --launch          # launch via Xbox app first, then run
    python run.py mc --dry-run         # no real keyboard/mouse input is sent
    python run.py mc --skills          # just list the game's skills
    python run.py mc --launch --dry-run
    python run.py among_us             # run Among Us ('au' alias also works)

Flags:
    --launch    Launch the Xbox app + game (Search -> game card -> Play) first.
    --dry-run   Run the closed loop but never send real input (safe testing).
    --skills    Print the loaded skills for the game and exit.
    --goal "…"  Override the default goal for the run.
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
    if "--launch" in flags:
        rc = main(["launch", "--game", game])
        if rc != 0:
            return rc
        flags = [f for f in flags if f != "--launch"]

    # Build the 'run' command.
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
