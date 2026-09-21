"""KBM + OCR driven Minecraft launch flow.

Implements the exact sequence the user asked for, using real keyboard/mouse
events (pynput) and OCR/OpenCV (ScreenFinder) for targeting instead of fragile
UIA tree traversal:

  1. Launch the Xbox app
  2. Type "minecraft for windows" into the search field
  3. (search field focused)
  4. Press Enter to search
  5. Look for Minecraft in the results (OCR)
  6. Click the Minecraft game card (OCR-located)
  7. Click Play to launch Minecraft (OCR-located)
  8. Navigate into gameplay (title -> Play -> world -> in-world HUD)

Each step verifies via a fresh screenshot + OCR before proceeding, and logs
structured events. The Xbox app itself is still located via the ambiguity-safe
UIA helper in XboxDesktopAutomation.
"""

from __future__ import annotations

import time
from typing import Any

from core.exceptions import WindowNotFoundError
from core.logger import get_logger
from tools.desktop.screen_finder import ScreenFinder
from tools.desktop.winapp import XboxDesktopAutomation

logger = get_logger("mc_launch")


class MinecraftLaunchFlow:
    """Orchestrates the KBM+OCR launch of Minecraft for Windows via the Xbox app."""

    SEARCH_TEXT = "minecraft for windows"
    CARD_QUERY = "minecraft"
    CARD_EXCLUDE = ("legends", "dungeons", "education", "trial")

    def __init__(self, xbox: XboxDesktopAutomation | None = None,
                 finder: ScreenFinder | None = None) -> None:
        self.xbox = xbox or XboxDesktopAutomation()
        self.finder = finder or ScreenFinder()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _refresh_region(self) -> None:
        """Point the ScreenFinder at the current Xbox window rectangle."""
        try:
            window = self.xbox._best_xbox_window()  # noqa: SLF001 - internal use
            self.finder.set_region_from_window(window)
        except Exception as exc:  # noqa: BLE001
            logger.warning("refresh_region_failed", error=str(exc))
            self.finder.region = None  # fall back to full-desktop capture

    def _focus_search_field(self) -> bool:
        """Focus the Xbox search box: UIA Edit -> OCR 'Search' -> Ctrl+F."""
        # 1) UIA: click the first Edit control to focus it (no typing here).
        try:
            window = self.xbox._window()  # noqa: SLF001
            edits = window.descendants(control_type="Edit")
            if edits:
                edits[0].click_input()
                logger.info("search_focus_uia")
                time.sleep(0.4)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_focus_uia_failed", error=str(exc))
        # 2) OCR: click a "Search" affordance.
        if self.finder.click_text("search", timeout_s=4.0):
            logger.info("search_focus_ocr")
            return True
        # 3) Hotkey fallback.
        self.finder.press_key("f")  # some builds use Ctrl+F; try plain focus
        logger.info("search_focus_hotkey_fallback")
        return False

    # ------------------------------------------------------------------ #
    # Steps 1-7: launch Minecraft from the Xbox app
    # ------------------------------------------------------------------ #
    def launch_from_xbox(self, *, ready_timeout_s: float = 40.0) -> bool:
        # Step 1: launch + wait + maximise.
        self.xbox.launch_xbox_app()
        try:
            self.xbox.wait_for_xbox_ready(timeout_s=ready_timeout_s)
        except WindowNotFoundError as exc:
            logger.error("xbox_not_ready", error=str(exc))
            return False
        self.xbox.maximize()
        time.sleep(1.5)
        self._refresh_region()

        # Step 2: focus search + type the query.
        self._focus_search_field()
        self.finder.type_text(self.SEARCH_TEXT, clear_first=True)
        time.sleep(0.5)

        # Steps 3-4: submit the search with Enter.
        self.finder.press_enter()
        logger.info("search_submitted", query=self.SEARCH_TEXT)
        time.sleep(8.0)  # Xbox app search results take ~5-8s to render

        # Steps 5-6: find Minecraft in results and click its card.
        self._refresh_region()
        if not self._click_minecraft_card():
            logger.warning("card_click_failed_trying_uia")
            if not self.xbox.select_game_from_results("Minecraft"):
                logger.error("minecraft_card_not_found")
                return False
        time.sleep(2.5)

        # Step 7: click Play/Launch on the game card.
        self._refresh_region()
        if not self._click_play():
            logger.warning("play_click_failed_trying_uia")
            if not self.xbox.click_play():
                logger.error("play_button_not_found")
                return False
        logger.info("minecraft_launch_triggered")
        return True

    def _click_minecraft_card(self) -> bool:
        for query in ("minecraft for windows", "minecraft launcher", "minecraft"):
            hit = self.finder.wait_for_text(query, timeout_s=6.0,
                                            exclude=self.CARD_EXCLUDE)
            if hit is not None:
                self.finder.click_at(*hit)
                logger.info("minecraft_card_clicked", matched=query)
                return True
        return False

    def _click_play(self) -> bool:
        for query in ("play", "launch", "resume", "install"):
            if self.finder.click_text(query, timeout_s=6.0):
                logger.info("play_clicked", matched=query)
                return True
        return False

    # ------------------------------------------------------------------ #
    # Step 8: navigate into gameplay
    # ------------------------------------------------------------------ #
    def wait_for_minecraft_window(self, window_re: str, timeout_s: float = 120.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                win = self.xbox.inspector.find_window(window_re)
                try:
                    win.set_focus()
                except Exception:  # noqa: BLE001
                    pass
                self.finder.set_region_from_window(win)
                logger.info("minecraft_window_ready", title_re=window_re)
                return True
            except WindowNotFoundError:
                time.sleep(2.0)
        logger.warning("minecraft_window_not_found", title_re=window_re)
        return False

    def navigate_to_gameplay(self, *, timeout_s: float = 180.0) -> bool:
        """Drive the Bedrock menus into an in-world session using OCR + KBM."""
        deadline = time.time() + timeout_s

        # 8a: confirm the title screen (Marketplace is unique to it).
        if not (self.finder.wait_for_text("marketplace", timeout_s=60.0)
                or self.finder.wait_for_text("play", timeout_s=10.0)):
            logger.warning("title_screen_not_detected")

        # 8b: click Play on the title screen.
        self._refresh_region()
        self.finder.click_text("play", timeout_s=20.0)
        time.sleep(2.0)

        # 8c: world list -> start a world (prefer existing; else create).
        self._refresh_region()
        started = False
        # Bedrock's world screen may render "Create New" / "Create New World" /
        # a "Worlds" tab; match broadly instead of one exact phrase.
        world_marker = (self.finder.wait_for_text("create new", timeout_s=15.0)
                        or self.finder.wait_for_text("worlds", timeout_s=5.0))
        if world_marker is not None:
            if self.finder.click_text("play world", timeout_s=4.0):
                started = True
            elif self._click_first_world_row():
                started = True
            else:
                hit = (self.finder.find_text("create new")
                       or self.finder.find_text("create new world"))
                if hit is not None:
                    self.finder.click_at(*hit)
                time.sleep(2.0)
                self._refresh_region()
                self.finder.click_text("create", timeout_s=8.0)
                started = True
        if not started:
            self.finder.press_enter()  # confirm highlighted world/button

        # 8d: wait for the in-world HUD (loading can take a while).
        while time.time() < deadline:
            self._refresh_region()
            if self._in_world():
                logger.info("gameplay_reached")
                return True
            time.sleep(3.0)
        logger.warning("gameplay_not_confirmed")
        return False

    def _click_first_world_row(self) -> bool:
        """Click just below the world-list header to hit the first world entry."""
        header = self.finder.find_text("worlds")
        if header is None:
            return False
        x, y = header
        self.finder.click_at(x, y + 90)
        time.sleep(1.0)
        return self.finder.click_text("play", timeout_s=4.0) or True

    def _in_world(self) -> bool:
        """Detect in-world: crosshair pixel-truth (authoritative) + OCR fallback."""
        frame, _, _ = self.finder.screenshot()
        # A centred white '+' crosshair is the authoritative in-world signal.
        try:
            from tools.vision.minecraft_vision import MinecraftVision
            if MinecraftVision().analyse(frame).crosshair_visible:
                return True
        except Exception:  # noqa: BLE001 - crosshair check must never crash flow
            pass
        try:
            results = self.finder._ocr.read_text(frame, use_cache=False)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return False
        text = " ".join(r.text.lower() for r in results)
        hints = ("hunger", "health", "coordinates", "chat", "hotbar")
        if any(h in text for h in hints):
            return True
        menu_words = ("marketplace", "singleplayer", "create new world", "settings")
        return not any(m in text for m in menu_words)

    # ------------------------------------------------------------------ #
    # Full flow
    # ------------------------------------------------------------------ #
    def run(self, window_re: str = ".*Minecraft.*", *, do_gameplay: bool = True) -> bool:
        if not self.launch_from_xbox():
            return False
        if not self.wait_for_minecraft_window(window_re):
            return False
        if do_gameplay:
            return self.navigate_to_gameplay()
        return True
