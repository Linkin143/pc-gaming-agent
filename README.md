# PC Gaming Agentic Automation Framework (PC-GAF)

A production-grade, multi-agentic AI framework for automating Windows PC games
launched through the **Xbox PC application / Game Pass**, built with **Python +
LangGraph + LangChain**, Windows UI Automation (UIA3), OpenCV, PaddleOCR, an
optional VLM tier, and a strictly deterministic keyboard/mouse execution layer.

Supported games: **Minecraft for Windows** (default) and **Among Us**.

---

## The one command to remember

From the `pc-gaming-agent/` folder:

```bash
python run.py minecraft            # run Minecraft automation (real input)
python run.py mc                   # 'mc' is an alias for minecraft
python run.py mc --launch          # launch via the Xbox app first, then run
python run.py mc --dry-run         # safe test: no real keyboard/mouse is sent
python run.py mc --skills          # list the game's skills and exit
python run.py mc --launch --dry-run
python run.py among_us             # Among Us ('au' alias also works)
```

Flags: `--launch`, `--dry-run`, `--skills`, `--max N`, `--goal "…"`.

---

## Minecraft launch flow (the 6 steps)

`python run.py mc --launch` performs exactly:

1. **Launch** the Xbox app (`xbox:` protocol)
2. **Maximise** the Xbox window
3. **Search** "Minecraft for Windows"
4. **Click the installed** Minecraft result
5. **Open** its game card
6. **Tap Play/Launch**, then wait for the Minecraft window

These are implemented in `tools/desktop/winapp.py` (pywinauto UIA3) and exposed as
skills in `skills/games/minecraft/xbox_launch.yaml`. Once the game window is in
the foreground, control transitions to the perception + keyboard/mouse gameplay
layer.

---

## The golden loop

```
STATE -> PLAN -> SELECT SKILL -> ACTION -> VALIDATE -> EXECUTE ->
OBSERVE -> VERIFY -> UPDATE STATE -> PLAN
```

with a bounded failure/recovery loop. Implemented as a typed LangGraph
`StateGraph` with checkpointing, conditional edges, and hard iteration/recovery
bounds (no infinite loops).

Guarantees: the LLM never touches the OS (planner emits a `SkillIntent`; only the
deterministic executors press keys); every action is verified before continuing;
perception is cost-aware (change-detection -> OpenCV -> OCR -> VLM, escalating
only when needed).

---

## Layout

```
pc-gaming-agent/
├── run.py             # simple runner: python run.py minecraft
├── main.py            # full CLI (skills / launch / run)
├── agents/            # planner, perception, skills, execution, verification, recovery, orchestrator
├── skills/            # YAML skills: common/ + games/minecraft/ + games/among_us/
├── games/             # adapters: common interface + minecraft + among_us
├── tools/             # desktop (UIA3), capture (mss), vision (OpenCV+VLM), ocr (PaddleOCR), input
├── core/              # state, models, config, constants, exceptions, logger
├── memory/  infrastructure/  configs/  tests/
```

---

## Testing

```bash
python -m pytest tests                       # unit + integration (live auto-skip)
python -m pytest tests/gaming --run-live -v  # live desktop/game tests
```

> Note: in Git Bash use `PCGAF_RUN_LIVE=1 python -m pytest ...` or the
> `--run-live` flag; `set VAR=...` is CMD-only syntax and won't export in bash.

---

## Adding a new game

1. Add `skills/games/<game>/*.yaml` (states + navigation + gameplay + perception).
2. Implement a `GameAdapter` in `games/<game>/adapter.py` (+ optional action builder).
3. Register it in `games/common/registry.py`.

The core orchestration engine is untouched.
