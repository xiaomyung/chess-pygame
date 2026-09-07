import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import pygame as pg

from chessshootout import paths
from chessshootout.infra import env
from chessshootout.online.client import probe_server_health
from chessshootout.server.protocol import HealthStatus, PROTOCOL_VERSION
from chessshootout.frontend.game.variant import Variant
from chessshootout.frontend.menu.options_rows import (
    ActionRow, PathRow, RevealRow, TextRow, ToggleRow, NotchRow, SegmentedRow,
    TONE_IDLE, TONE_OK, TONE_WARN,
)


log = logging.getLogger("chess.frontend")

SETTINGS_WRITE_DELAY_MS = 400
SERVER_PROBE_BUTTON_IDLE = "Test"
SERVER_PROBE_BUTTON_PENDING = "Testing…"
SERVER_SWITCH_BLOCKED_MESSAGE = "Leave the online session to switch servers"
INVALID_SERVER_ADDR_MESSAGE = "Invalid server address"


class SettingsController:
    """
    The Options screen's brain. It builds every row the player sees, turns each
    press into a stored setting, and makes that setting bite straight away --
    volumes, launch mode, game defaults, where saved games live, and which
    server the client talks to. The Options view only draws and clicks what
    this hands it
    """

    def __init__(self, frontend: Any) -> None:
        """
        Build the controller once at startup, alongside the rest of the shell.
        Nothing is read or written here; the rows themselves are built on the
        first visit to Options

        :param frontend: the Frontend shell, this controller's route to the
            window, sound, toasts, the menu and the online coordinator
        """
        self.frontend = frontend
        self._deferred_env_writes: dict[str, tuple[Callable[[], None], int]] = {}
        self._data_folder_row: PathRow | None = None
        self._custom_server_row: TextRow | None = None
        self._server_probe_lock = threading.Lock()
        self._server_probe_pending = False
        self._server_probe_result: tuple[str, tuple[str, str]] | None = None
        self._server_probe_gen = 0

    def commit_options_exit(self) -> None:
        """
        Settle everything Options left half-finished, run when the player
        leaves the screen and again when the game quits. A folder or address
        still being typed is committed here, the connection test is cleared,
        the menu picks up the new default time control, and every delayed
        write goes to disk at once
        """
        self._validate_data_folder_on_exit()
        if self._custom_server_row is not None:
            self._commit_custom_server_addr(self._custom_server_row.current_text())
        self._reset_server_probe()
        self.frontend.menu.apply_default_time_settings()
        self._flush_deferred_env_writes(force=True)

    def _online_session_active(self) -> bool:
        """
        Whether an online session is live right now, which is what makes the
        server setting read-only -- repointing the client mid-game would
        strand the player away from the match they are in

        :returns: True while the client is connected to a server
        """
        return cast(bool, self.frontend.coordinator.is_connected())

    def _apply_server_mode(self, mode: str) -> None:
        """
        Switch the client between the official servers and the player's own
        address, refusing with a toast while an online session is live.
        Choosing the mode already stored does nothing at all

        :param mode: server mode to store, official or custom
        """
        if mode == env.get_server_mode():
            return
        if self._online_session_active():
            self.frontend.toast.show(SERVER_SWITCH_BLOCKED_MESSAGE)
            return
        env.set_server_mode(mode)
        self._announce_server_target_change()

    def _commit_custom_server_addr(self, typed: str) -> None:
        """
        Store the address of the server the player runs themselves. A
        malformed address is refused with a toast and the old one is kept;
        blank text and an unchanged address are both no-ops, so merely
        stepping past the row never rewrites the setting

        :param typed: address exactly as typed, host or host:port
        """
        if not typed or typed == env.get_custom_server_addr():
            return
        if self._online_session_active():
            self.frontend.toast.show(SERVER_SWITCH_BLOCKED_MESSAGE)
            return
        if not env.set_custom_server_addr(typed):
            self.frontend.toast.show(INVALID_SERVER_ADDR_MESSAGE)
            return
        self._announce_server_target_change()
        if env.get_server_mode() == env.SERVER_MODE_CUSTOM:
            self.frontend.toast.show(f"Server set to {env.get_server_addr()}")

    def _announce_server_target_change(self) -> None:
        """
        Tell the rest of the app that the client now points somewhere else:
        the old connection-test verdict is dropped and the coordinator lets go
        of anything it was holding about the previous server
        """
        self._reset_server_probe()
        self.frontend.coordinator.on_server_target_changed()
        log.info("server target set mode=%s addr=%s", env.get_server_mode(),
                 env.get_server_addr())

    def _on_test_server(self) -> None:
        """
        Run the Connection test from the Options row, so the player can see a
        server answers before they queue for a match. The request goes out on
        a background thread and a second press while one is in flight is
        ignored rather than queued
        """
        with self._server_probe_lock:
            if self._server_probe_pending:
                return
            self._server_probe_gen += 1
            gen = self._server_probe_gen
            self._server_probe_pending = True
            self._server_probe_result = None
        addr = self._probe_target_addr()
        threading.Thread(target=self._server_probe_worker, args=(addr, gen),
                         daemon=True).start()

    def _probe_target_addr(self) -> str:
        """
        Decide which address the connection test should hit. On a custom
        server that is whatever the address row holds right now, committed or
        not, so the test always matches the address the player is looking at

        :returns: server address to test, empty when none is configured
        """
        if env.get_server_mode() != env.SERVER_MODE_CUSTOM:
            return env.get_server_addr()
        if self._custom_server_row is None:
            return env.get_custom_server_addr()
        return self._custom_server_row.current_text() or env.get_custom_server_addr()

    def _server_probe_worker(self, addr: str, gen: int) -> None:
        """
        Run one connection test off the main thread and leave the verdict
        where the Options row can draw it. Each probe carries a generation
        stamp, so a slow answer to an abandoned test discards itself instead
        of overwriting a newer one

        :param addr: server address being tested
        :param gen: probe generation this worker was started for
        """
        try:
            started = time.monotonic()
            health = probe_server_health(addr)
            latency_ms = int((time.monotonic() - started) * 1000)
            outcome = self._describe_health(health, latency_ms)
            with self._server_probe_lock:
                if gen != self._server_probe_gen:
                    return
                self._server_probe_result = (addr, outcome)
            log.debug("server probe addr=%s outcome=%s", addr, outcome[1])
        finally:
            with self._server_probe_lock:
                if gen == self._server_probe_gen:
                    self._server_probe_pending = False

    def _describe_health(self, health: dict[str, Any] | None,
                         latency_ms: int) -> tuple[str, str]:
        """
        Turn a server's health answer into the one-line verdict the Options
        row shows: unreachable, a protocol mismatch that would make playing
        there impossible, a server that is full or behind on its own upkeep,
        or an OK line with the round trip and the version the server is running

        :param health: health summary the server sent back, or None when it
            never answered
        :param latency_ms: how long the health request took, in milliseconds
        :returns: tone constant and the message to draw beside the row
        """
        if health is None:
            return (TONE_WARN, "Unreachable")
        version = health.get("version")
        if version != PROTOCOL_VERSION:
            seen = "?" if version is None else version
            return (TONE_WARN,
                    f"Protocol mismatch: server v{seen} ≠ client v{PROTOCOL_VERSION}")
        status = health.get("status", HealthStatus.OK)
        if status == HealthStatus.FULL:
            return (TONE_WARN, f"Server full · {latency_ms} ms")
        if status == HealthStatus.DEGRADED:
            return (TONE_WARN, f"Degraded · {latency_ms} ms")
        app_version = health.get("app_version") or ""
        suffix = f" · v{app_version}" if app_version else ""
        return (TONE_OK, f"OK · {latency_ms} ms{suffix}")

    def _probe_button_label(self) -> str:
        """
        Label for the Connection row's button, which reads as testing while a
        probe is out so the player can see their press registered

        :returns: the button label for the current test state
        """
        with self._server_probe_lock:
            pending = self._server_probe_pending
        return SERVER_PROBE_BUTTON_PENDING if pending else SERVER_PROBE_BUTTON_IDLE

    def _probe_status(self) -> tuple[str, str]:
        """
        The verdict line drawn beside the Connection row. A result stops
        showing the moment the player edits the address, so an old verdict can
        never be read as describing the server now typed

        :returns: tone constant and message, both idle and blank when there is
            nothing to report
        """
        with self._server_probe_lock:
            pending = self._server_probe_pending
            probed = self._server_probe_result
        if pending or probed is None:
            return (TONE_IDLE, "")
        addr, outcome = probed
        if addr != self._probe_target_addr():
            return (TONE_IDLE, "")
        return outcome

    def _reset_server_probe(self) -> None:
        """
        Forget the connection test, whether it is finished or still out, so
        the row goes back to blank. Bumping the generation is what makes a
        worker still running throw its own answer away
        """
        with self._server_probe_lock:
            self._server_probe_gen += 1
            self._server_probe_pending = False
            self._server_probe_result = None

    def _validate_data_folder_on_exit(self) -> None:
        """
        Apply a games folder the player typed but never confirmed, as they
        leave Options. A path that cannot be written to is refused with a
        toast, so saved games are never pointed somewhere they would be lost
        """
        if self._data_folder_row is None:
            return
        typed = self._data_folder_row.current_text()
        if not typed:
            return
        typed = os.path.abspath(os.path.expanduser(typed))
        if os.path.normpath(typed) == os.path.normpath(str(paths.get_data_dir())):
            return
        if not paths.is_writable_dir(typed):
            self.frontend.toast.show("That folder isn't writable")
            return
        self._apply_data_folder_change(typed)

    def apply_hide_opp_marks(self, value: bool) -> None:
        """
        Store whether the opponent's shared arrows and highlights are hidden,
        and make it bite at once rather than from the next game: an online
        opponent is told to stop sending marks, and anything already drawn on
        the current game is wiped

        :param value: True to hide the marks the opponent shares
        """
        env.set_hide_opp_marks(value)
        game = self.frontend.game
        coordinator = self.frontend.coordinator
        if coordinator.is_connected() and game.variant == Variant.ONLINE:
            coordinator.set_marks_visibility(value)
        if value:
            game.apply_opp_marks_shield()

    def _set_master_volume(self, value: float) -> None:
        """
        Move the overall volume live while the player drags the slider,
        without touching the stored setting -- that write waits until they let
        go, so a drag does not hammer the settings file

        :param value: new master volume, 0.0 silent to 1.0 full
        """
        self.frontend.sound_manager.set_master_volume(value)

    def _set_menu_volume(self, value: float) -> None:
        """
        Move the menu volume live while the player drags the slider. Menu
        sounds are scaled by this and by the master volume together, so this
        only ever quietens the menus relative to the game

        :param value: new menu volume, 0.0 silent to 1.0 full
        """
        self.frontend.sound_manager.set_menu_volume(value)

    def _commit_master_volume(self) -> None:
        """
        Save the master volume the mixer is actually playing at, so the level
        the player settled on comes back at the next launch
        """
        env.set_master_volume(self.frontend.sound_manager.master_volume)

    def defer_master_volume_write(self) -> None:
        """
        Ask for the master volume to be saved shortly. The volume control on
        the game rail calls this after each nudge and the Options slider on
        every release, so holding either one down writes once at the end
        instead of on every step
        """
        self._defer_env_write("master_volume", self._commit_master_volume)

    def _commit_menu_volume(self) -> None:
        """
        Save the menu volume the mixer is actually playing at, so the level
        the player settled on comes back at the next launch
        """
        env.set_menu_volume(self.frontend.sound_manager.menu_volume)

    def _defer_env_write(self, key: str, commit: Callable[[], None]) -> None:
        """
        Hold a settings write back for a moment, which is what keeps a slider
        drag from rewriting the settings file on every pixel. A second request
        under the same name replaces the first, so only the value the player
        ends on is written

        :param key: setting name, at most one pending write per name
        :param commit: the write to run once the delay is up
        """
        self._deferred_env_writes[key] = (commit, pg.time.get_ticks() + SETTINGS_WRITE_DELAY_MS)

    def _flush_deferred_env_writes(self, force: bool = False) -> None:
        """
        Run the held-back settings writes whose delay has elapsed, called once
        a frame by the shell. Forcing runs every pending write straight away,
        which is what leaving Options and quitting the game both do

        :param force: True to write everything now and ignore the delay
        """
        if not self._deferred_env_writes:
            return
        now = pg.time.get_ticks()
        for key in list(self._deferred_env_writes):
            commit, due = self._deferred_env_writes[key]
            if force or now >= due:
                del self._deferred_env_writes[key]
                commit()

    def build_settings_sections(self) -> list[tuple[str, list[Any]]]:
        """
        Build the whole Options screen: the section headings, and under each
        of them the rows the player sees, already wired to the getters and
        setters that read and write the stored settings. Two rows are kept a
        reference to as well, because they hold text that has to be committed
        when the player leaves

        :returns: sections in display order, each a heading and its rows
        """
        window = self.frontend.window
        sound_manager = self.frontend.sound_manager
        self._data_folder_row = PathRow(
            "Games folder", "Where PGNs are saved", window,
            lambda: str(paths.get_data_dir()),
            self._on_change_data_folder, self._on_reset_data_folder,
            suffix="/" + paths.GAMES_SUBDIR)
        self._custom_server_row = TextRow(
            "Custom address", "Host or host:port of the server you run",
            window, env.get_custom_server_addr, placeholder="host or host:port",
            on_commit=self._commit_custom_server_addr)
        time_options = [(label, label) for label in env.TIME_CONTROL_VALUES]
        incr_options = [(label, label) for label in env.INCREMENT_VALUES]
        return [
            ("Audio", [
                NotchRow("Master volume", "", lambda: sound_manager.master_volume,
                         self._set_master_volume, on_tick=sound_manager.play_ui_tick,
                         on_release=self.defer_master_volume_write),
                NotchRow("Menu volume", "", lambda: sound_manager.menu_volume,
                         self._set_menu_volume, on_tick=sound_manager.play_ui_tick,
                         on_release=lambda: self._defer_env_write(
                             "menu_volume", self._commit_menu_volume)),
                ToggleRow("Mute all sound", "Silence every shot and callout",
                          lambda: not sound_manager.enabled,
                          lambda muted: sound_manager.set_enabled(not muted)),
            ]),
            ("Display", [
                SegmentedRow("Launch mode",
                             "How the window opens — applies on next launch",
                             [("Windowed", "windowed"), ("Maximized", "maximized"),
                              ("Fullscreen", "fullscreen")],
                             env.get_launch_mode, env.set_launch_mode),
            ]),
            ("Focus mode", [
                SegmentedRow("Show in focus",
                             "What stays on screen when the panel is hidden",
                             [("Nothing", "nothing"), ("Time Line", "line"),
                              ("Full strips", "strips")],
                             env.get_focus_show, env.set_focus_show),
            ]),
            ("Game", [
                SegmentedRow("Default time", "Minutes on the clock, or untimed",
                             time_options, env.get_default_time_control,
                             env.set_default_time_control, mono=True, variant="cells"),
                SegmentedRow("Default increment", "Seconds added each move",
                             incr_options, env.get_default_increment,
                             env.set_default_increment, mono=True, variant="cells"),
                ToggleRow("Auto-queen",
                          "Promotions auto-pick the queen — the skill check still fires",
                          env.get_auto_queen, env.set_auto_queen),
                self._data_folder_row,
            ]),
            ("Online", [
                SegmentedRow("Server", "Official servers, or your own address",
                             [("Official", env.SERVER_MODE_OFFICIAL),
                              ("Custom", env.SERVER_MODE_CUSTOM)],
                             env.get_server_mode, self._apply_server_mode),
                RevealRow(self._custom_server_row,
                          lambda: env.get_server_mode() == env.SERVER_MODE_CUSTOM),
                ActionRow("Connection", "Check the server answers before you queue",
                          self._probe_button_label, self._on_test_server,
                          self._probe_status),
                ToggleRow("Hide opponent's marks",
                          "Never show the arrows and highlights they share",
                          env.get_hide_opp_marks, self.apply_hide_opp_marks),
            ]),
            ("Performance", [
                ToggleRow("Show FPS", "Frame rate in the title bar",
                          env.get_show_fps, env.set_show_fps),
                ToggleRow("Show avg / min FPS", "Rolling render-rate over recent frames",
                          env.get_show_frame_stats, env.set_show_frame_stats),
                ToggleRow("Show 1% low FPS", "Worst-1% frames — the stutter metric",
                          env.get_show_1pct_low, env.set_show_1pct_low),
                ToggleRow("Show frame time", "Milliseconds of render work per frame",
                          env.get_show_frametime, env.set_show_frametime),
                ToggleRow("Show ping", "Network latency in the title bar",
                          env.get_show_ping, env.set_show_ping),
            ]),
        ]

    def _on_change_data_folder(self) -> None:
        """
        Open the in-app folder browser so the player can pick where saved
        games live, wired to the Change button on the games-folder row
        """
        self.frontend.directory_browser.show(
            str(paths.get_data_dir()),
            on_select=self._apply_data_folder_change,
            on_error=self.frontend.toast.show,
        )

    def _on_reset_data_folder(self) -> None:
        """
        Put saved games back in the platform's default place. The stored
        override is cleared rather than set to today's default path, so the
        folder keeps following the platform if that ever moves
        """
        default = paths.get_default_data_dir()
        self._apply_data_folder_change(str(default), to_default=True)

    def _apply_data_folder_change(self, new_dir: str, to_default: bool = False) -> None:
        """
        Point saved games at a new folder, asking first whether the PGNs
        already on disk should come along. Nothing is asked when the folder
        works out the same or there is nothing there to move

        :param new_dir: folder to keep game data in, an absolute path
        :param to_default: True when this is a reset, which clears the stored
            override instead of saving the path
        """
        old_games = str(paths.get_games_dir())
        new_games = os.path.join(new_dir, paths.GAMES_SUBDIR)
        if os.path.normpath(old_games) == os.path.normpath(new_games):
            self._commit_data_dir(new_dir, to_default, None)
            return
        if os.path.isdir(old_games):
            pgns = [f for f in os.listdir(old_games) if f.endswith(".pgn")]
        else:
            pgns = []
        if pgns:
            count = len(pgns)
            suffix = "s" if count != 1 else ""
            self.frontend.confirm_modal.show(
                f"Move {count} saved game{suffix} to the new folder?",
                on_yes=lambda: self._commit_data_dir(new_dir, to_default, old_games),
                on_no=lambda: self._commit_data_dir(new_dir, to_default, None),
                yes_label="Move", no_label="Don't move",
                on_extra=lambda: None, extra_label="Cancel", emoji="📁",
            )
        else:
            self._commit_data_dir(new_dir, to_default, None)

    def _commit_data_dir(self, new_dir: str, to_default: bool,
                         move_from: str | None) -> None:
        """
        Store the new game-data folder, first carrying the old saved games
        across when asked to. A move that fails abandons the whole change, so
        the setting can never end up pointing where the games did not arrive

        :param new_dir: folder to store, an absolute path
        :param to_default: True to clear the override rather than store a path
        :param move_from: folder whose PGNs should be moved across, or None to
            leave them where they are
        """
        new_games = os.path.join(new_dir, paths.GAMES_SUBDIR)
        if move_from is not None and not self._move_pgns(move_from, new_games):
            return
        env.set_data_dir(None if to_default else new_dir)
        self.frontend.toast.show("Data folder updated")

    def _move_pgns(self, src: str, dst: str) -> bool:
        """
        Carry the player's saved games over to the new folder. A file whose
        name is already taken there is given a numbered name instead of
        overwriting what is there, and the old folder is removed only if
        moving left it empty

        :param src: folder the PGNs are in now
        :param dst: folder they should end up in, created when missing
        :returns: True when every game arrived
        """
        try:
            os.makedirs(dst, exist_ok=True)
            for name in os.listdir(src):
                if not name.endswith(".pgn"):
                    continue
                target = os.path.join(dst, name)
                if os.path.exists(target):
                    target = self._unique_pgn_name(dst, name)
                shutil.move(os.path.join(src, name), target)
        except OSError:
            self.frontend.toast.show("Could not move games")
            return False
        try:
            os.rmdir(src)
        except OSError:
            pass
        return True

    def _unique_pgn_name(self, dst: str, name: str) -> str:
        """
        Find a free filename for a saved game whose name is already taken in
        the destination, counting a numeric suffix up until nothing is there

        :param dst: destination folder being written into
        :param name: PGN filename that clashed
        :returns: full path of a name that does not exist yet
        """
        base, ext = os.path.splitext(name)
        i = 1
        while True:
            candidate = os.path.join(dst, f"{base}-{i}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1
