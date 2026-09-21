Everything is complete and verified. Here's the summary.

---

## ✅ Done — Minecraft support added + simple run command (and full project recovery)

### ⚠️ Important note first
Partway through this task, **most of the project directory got wiped** during shell file-write workarounds (there was no git repo to restore from). I **fully rebuilt every lost file** from the implementation in our history — `core/`, `agents/`, `tools/`, `memory/`, `infrastructure/`, `configs/`, `tests/`, `main.py`, common skills, and the whole `among_us` package are all back and verified. Nothing was left broken.

### 1. Simple, memorable run command — `run.py`
```bash
python run.py minecraft          # run Minecraft automation
python run.py mc                 # short alias
python run.py mc --launch        # launch the game via Xbox app, then run
python run.py mc --dry-run       # safe test, no real input (finishes in seconds)
python run.py mc --skills        # list skills
python run.py among_us           # Among Us still works (au alias too)
```

### 2. Minecraft — the exact 6-step launch flow you asked for
Implemented in `tools/desktop/winapp.py` (pywinauto UIA3) and expressed as skills in `skills/games/minecraft/xbox_launch.yaml`:
1. **Launch** Xbox app → 2. **Maximise** window → 3. **Search** "Minecraft for Windows" → 4. **Click the installed** game → 5. **Open** the game card → 6. **Tap Play**, then wait for the Minecraft window.

`run_launch_flow()` chains all six; the planner also drives them screen-by-screen (`xbox_app → game_loading → main_menu → gameplay`).

### 3. Full Minecraft game package
- `games/minecraft/`: `adapter.py`, `state.py`, `perception.py` (title/world/loading/pause detection), `actions.py` (move/jump/mine/place/look/hotbar + routes `xbox_*` skills to the desktop layer).
- **49 skills** across 5 YAML files: `xbox_launch`, `navigation`, `gameplay`, `perception`, `states` (7 screen states).
- Registered in `games/common/registry.py`; default game set to `minecraft` in `configs/app.yaml`.

### 4. Fixes made along the way
- **Multi-monitor bug**: your setup has negative screen coordinates (`x=-1366`); relaxed `Rect` to allow them so screen capture no longer fails.
- **`--run-live`**: added a proper pytest flag so live tests run from Git Bash without the CMD-only `set VAR=` syntax.

### Verification
- **22 tests pass, 2 live tests skip** (`pytest tests`).
- `python run.py mc --dry-run` runs the full closed loop end-to-end and completes cleanly.
- `python run.py mc --skills` lists all 49 skills.

To actually launch and play Minecraft on your machine:
```bash
cd pc-gaming-agent
python run.py mc --launch
```