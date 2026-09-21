"""Xbox PC application / Game Pass desktop automation.

Drives the Windows Xbox app through pywinauto's UIA3 backend to:

  1. launch the Xbox app
  2. maximise its window
  3. search for a game (e.g. "Minecraft for Windows")
  4. click the installed game in the results
  5. open its game card
  6. tap the Play/Launch button

Once the game window comes to the foreground, control transitions to the
perception + keyboard/mouse gameplay layer (this class intentionally stops
there). FlaUI, WinAppDriver, and Xbox-controller automation are NOT used.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from core.exceptions import WindowNotFoundError
from core.logger import get_logger
from tools.desktop.ui_automation import UIInspector

logger = get_logger("xbox")

# Protocol URI that launches the Xbox app on Windows.
_XBOX_URI = "xbox:"
_XBOX_TITLE_RE = ".*Xbox.*"
# Title patterns ordered most-to-least specific. The main Xbox PC app window is
# titled exactly "Xbox"; the broad fallback is safe because window selection
# now scores/filters candidates (Game Bar etc. are excluded).
_XBOX_TITLE_CANDIDATES = (r"^Xbox$", r"^Xbox Game Pass$", r"^Xbox App$", r".*Xbox.*")
# UWP apps host their window inside an ApplicationFrameWindow; the Game Bar uses
# a different class, which lets us disambiguate.
_XBOX_FRAME_CLASS = "ApplicationFrameWindow"
# Only these window classes host a real Xbox / Store app frame. Anything else
# (VS Code = Chrome_WidgetWin_1, File Explorer = CabinetWClass, browsers, etc.)
# must never be mistaken for the Xbox app just because its title says "Xbox".
_XBOX_ALLOWED_CLASSES = (
    "ApplicationFrameWindow",       # standard UWP app frame
    "Windows.UI.Core.CoreWindow",   # older UWP host
    "WinUIDesktopWin32Window",      # WinUI 3 host
)
_XBOX_EXCLUDE_TITLES = (
    "game bar", "console companion", "settings",
    "visual studio", "file explorer", " - code", "- explorer",
)
_PLAY_LABELS = ("play", "launch", "resume", "start game")
_INSTALL_LABELS = ("install", "get", "download")


class XboxDesktopAutomation:
    """Automates the pre-game Xbox / Game Pass desktop flow."""

    def __init__(self, inspector: UIInspector | None = None) -> None:
        self.inspector = inspector or UIInspector()
        self._app: Any | None = None

    # ------------------------------------------------------------------ #
    # Window discovery / selection (ambiguity-safe)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_uwp_class(cls: str) -> bool:
        return any(c in (cls or "") for c in _XBOX_ALLOWED_CLASSES)

    def _best_xbox_window_strict(self) -> Any:
        """Match ONLY a real Xbox UWP window (allowed class); no loose fallback.

        This is the authoritative "is the Xbox app present?" check. It rejects
        VS Code, File Explorer, browsers and anything else that merely has the
        word 'Xbox' in its title (e.g. because the project folder is named
        Xbox_PC_Automation). Raises WindowNotFoundError if none match.
        """
        for pattern in _XBOX_TITLE_CANDIDATES:
            for w in self.inspector.list_windows(pattern):
                try:
                    title = (w.window_text() or "").lower()
                    cls = w.element_info.class_name or ""
                    if any(x in title for x in _XBOX_EXCLUDE_TITLES):
                        continue
                    if self._is_uwp_class(cls):
                        return w
                except Exception:  # noqa: BLE001
                    continue
        raise WindowNotFoundError("Xbox UWP window not found.",
                                  context={"tried": list(_XBOX_TITLE_CANDIDATES)})

    def _best_xbox_window(self) -> Any:
        """Return the real Xbox app window, disambiguating from other windows.

        Strict class-filtered match first; then a guarded fallback that STILL
        requires an allowed UWP class (so VS Code / Explorer are never matched).
        """
        try:
            return self._best_xbox_window_strict()
        except WindowNotFoundError:
            pass
        # Guarded fallback: broad title regex but the class must still be UWP.
        for w in self.inspector.list_windows(_XBOX_TITLE_RE):
            try:
                title = (w.window_text() or "").lower()
                cls = w.element_info.class_name or ""
                if any(x in title for x in _XBOX_EXCLUDE_TITLES):
                    continue
                if self._is_uwp_class(cls) and w.is_visible():
                    return w
            except Exception:  # noqa: BLE001
                continue
        raise WindowNotFoundError("Xbox app window not found.",
                                  context={"tried": list(_XBOX_TITLE_CANDIDATES)})

    # ------------------------------------------------------------------ #
    # App lifecycle
    # ------------------------------------------------------------------ #
    def is_xbox_running(self) -> bool:
        """True only if a real Xbox UWP window is present (never VS Code, etc.)."""
        try:
            self._best_xbox_window_strict()
            return True
        except WindowNotFoundError:
            return False

    def launch_xbox_app(self) -> None:
        """Start the Xbox app via its protocol handler if not already running."""
        if self.is_xbox_running():
            logger.info("xbox_already_running")
            return
        logger.info("launching_xbox_app")
        subprocess.Popen(["cmd", "/c", "start", "", _XBOX_URI], shell=False)

    def wait_for_xbox_ready(self, timeout_s: float = 40.0) -> Any:
        """Poll until the Xbox window exists, returning its window spec."""
        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                window = self._best_xbox_window()
                self._app = window
                logger.info("xbox_ready", title=window.window_text())
                return window
            except WindowNotFoundError as exc:
                last_error = exc
                time.sleep(1.0)
        raise WindowNotFoundError(
            "Xbox app did not become ready in time.",
            context={"timeout_s": timeout_s, "error": str(last_error)},
        )

    def _window(self) -> Any:
        if self._app is None:
            self._app = self._best_xbox_window()
        return self._app

    # ------------------------------------------------------------------ #
    # Step 2: maximise
    # ------------------------------------------------------------------ #
    def maximize(self) -> None:
        """Maximise the Xbox window so all controls are reliably visible."""
        window = self._window()
        try:
            window.set_focus()
            if hasattr(window, "maximize"):
                window.maximize()
                logger.info("xbox_maximized")
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("maximize_api_failed", error=str(exc))
        try:
            from pywinauto.keyboard import send_keys

            window.set_focus()
            send_keys("{VK_LWIN down}{UP}{VK_LWIN up}")
            logger.info("xbox_maximized_hotkey")
        except Exception as exc:  # noqa: BLE001
            logger.warning("maximize_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Step 3: search
    # ------------------------------------------------------------------ #
    def search_game(self, name: str) -> None:
        """Type a game name into the Xbox search box (with keyboard fallback)."""
        window = self._window()
        window.set_focus()
        try:
            edits = window.descendants(control_type="Edit")
            if edits:
                box = edits[0]
                box.set_focus()
                try:
                    box.set_edit_text("")
                except Exception:  # noqa: BLE001
                    pass
                box.type_keys(name, with_spaces=True, set_foreground=True)
                logger.info("game_search_typed", game=name)
                time.sleep(1.5)
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_box_not_found", error=str(exc))
        try:
            from pywinauto.keyboard import send_keys

            window.set_focus()
            send_keys("^e")
            time.sleep(0.5)
            send_keys(name.replace(" ", "{SPACE}"))
            logger.info("game_search_typed_fallback", game=name)
            time.sleep(1.5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_fallback_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Steps 4-5: select the installed game and open its card
    # ------------------------------------------------------------------ #
    def select_game_from_results(self, name: str, *, prefer_installed: bool = True) -> bool:
        """Click a search result matching ``name`` and open its game card."""
        window = self._window()
        needle = name.lower()
        installed_hit = None
        first_hit = None
        try:
            for ctrl in window.descendants():
                info = ctrl.element_info
                low = (info.name or "").lower()
                if needle in low:
                    if first_hit is None:
                        first_hit = ctrl
                    if prefer_installed and ("install" not in low):
                        installed_hit = ctrl
                        break
            target = installed_hit or first_hit
            if target is not None:
                target.click_input()
                logger.info("game_card_opened", game=name,
                            matched=target.element_info.name)
                time.sleep(2.0)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("select_game_failed", error=str(exc))
        logger.warning("game_not_found_in_results", game=name)
        return False

    # ------------------------------------------------------------------ #
    # Step 6: press Play / Launch
    # ------------------------------------------------------------------ #
    def click_play(self) -> bool:
        """Click a Play/Launch button on the game card (Install as fallback)."""
        window = self._window()
        try:
            buttons = list(window.descendants(control_type="Button"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("buttons_enumeration_failed", error=str(exc))
            buttons = []
        for btn in buttons:
            label = (btn.element_info.name or "").lower()
            if any(t in label for t in _PLAY_LABELS):
                try:
                    btn.click_input()
                    logger.info("play_button_clicked", label=label)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("play_click_failed", error=str(exc))
        for btn in buttons:
            label = (btn.element_info.name or "").lower()
            if any(t in label for t in _INSTALL_LABELS):
                try:
                    btn.click_input()
                    logger.info("install_button_clicked", label=label)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("install_click_failed", error=str(exc))
        logger.warning("play_button_not_found")
        return False

    def click_play_or_install(self) -> bool:
        """Backwards-compatible alias."""
        return self.click_play()

    # ------------------------------------------------------------------ #
    # Launch verification (boundary to gameplay layer)
    # ------------------------------------------------------------------ #
    def verify_game_launched(self, window_title_re: str, timeout_s: float = 90.0) -> bool:
        """Wait until the game's own window appears (control then transitions)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                self.inspector.find_window(window_title_re)
                logger.info("game_launch_verified", title_re=window_title_re)
                return True
            except WindowNotFoundError:
                time.sleep(2.0)
        logger.warning("game_launch_unverified", title_re=window_title_re)
        return False

    # ------------------------------------------------------------------ #
    # Full flow (the exact 6 steps requested)
    # ------------------------------------------------------------------ #
    def run_launch_flow(
        self,
        game_name: str,
        game_window_re: str,
        *,
        ready_timeout_s: float = 40.0,
        verify: bool = True,
    ) -> bool:
        """launch -> maximise -> search -> select installed -> open card -> play."""
        self.launch_xbox_app()                       # 1
        self.wait_for_xbox_ready(timeout_s=ready_timeout_s)
        self.maximize()                              # 2
        time.sleep(1.0)
        self.search_game(game_name)                  # 3
        time.sleep(1.5)
        self.select_game_from_results(game_name)     # 4 + 5
        time.sleep(1.5)
        played = self.click_play()                   # 6
        if not verify:
            return played
        return self.verify_game_launched(game_window_re)
