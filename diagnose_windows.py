"""Diagnostic: list top-level windows and which one we pick as the Xbox app.

Run this if `--launch` cannot find the Xbox window:

    python diagnose_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.desktop.ui_automation import UIInspector  # noqa: E402
from tools.desktop.winapp import XboxDesktopAutomation  # noqa: E402


def main() -> int:
    inspector = UIInspector()
    print("=== Top-level windows containing 'xbox' (case-insensitive) ===")
    for w in inspector.list_windows(".*[Xx]box.*"):
        try:
            title = w.window_text()
            cls = w.element_info.class_name
            visible = w.is_visible()
            print(f"  title={title!r:40s} class={cls!r:28s} visible={visible}")
        except Exception as exc:  # noqa: BLE001
            print(f"  <unreadable window: {exc}>")

    xbox = XboxDesktopAutomation()

    print("\n=== is_xbox_running() (strict UWP check) ===")
    print(f"  running={xbox.is_xbox_running()}")

    print("\n=== Selected Xbox app window (full search) ===")
    try:
        w = xbox._best_xbox_window()
        print(f"  PICKED: title={w.window_text()!r} class={w.element_info.class_name!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  none found: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
