"""Logging hygiene guards for the v2.7.0 logging pass.

Two runtime checks (a live idle game must not emit any log record per frame,
and a finished game's result must log exactly once no matter how many more
frames render while the result screen is up) plus two static checks (no log
call literally sits inside a known per-frame draw/update function, and no log
call anywhere references a skill-check secret or geometry seed).
"""

import ast
import logging
import os
import re

import pytest

import chessshootout

from tests.conftest import pygame_display
from tests.frontend.focus_helpers import FakeTicks, install_clock, make_app, start_game
from tests.helpers import read_source_without_docstrings
from chessshootout.backend.pieces import PieceColor
from chessshootout.frontend.screens.base import Nav

PACKAGE_ROOT = os.path.dirname(os.path.abspath(chessshootout.__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)

LOG_CALL_RE = re.compile(r"\blog\.(debug|info|warning|error|exception|critical)\(")

STEADY_STATE_LOG_CALL_RE = re.compile(r"\blog\.(debug|info|warning|error|critical)\(")
"""Excludes log.exception: it can only run inside an active `except` clause,
so unlike the other levels it structurally cannot fire every steady-state
frame — a per-frame function is allowed a guarded error-path log.exception."""

FRAME_FUNCTIONS = {
    "chessshootout/frontend/frontend.py": ["draw_frame"],
    "chessshootout/frontend/board/board.py": ["draw_board", "update_drag_physics"],
    "chessshootout/frontend/game/result_flow.py": ["update_result_pending"],
    "chessshootout/frontend/game/give_time.py": ["update_give_time_hold"],
    "chessshootout/server/sweep.py": ["step_all"],
    "chessshootout/server/app.py": ["_sweep_loop"],
}


_pg = pygame_display(1000, 800)


class _CaptureHandler(logging.Handler):

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def capture_chess_logs():
    handler = _CaptureHandler()
    root = logging.getLogger()
    saved_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)
    root.setLevel(saved_level)


def _chess_records(handler):
    return [r for r in handler.records if r.name.startswith("chess.")]


def test_idle_draw_frame_emits_no_log_records(monkeypatch, capture_chess_logs):
    """A live game with no player input must not log per frame — game start
    already logged before the capture window opens, so anything captured here
    would be a genuine per-frame leak."""
    clock = FakeTicks()
    install_clock(monkeypatch, clock)
    app = start_game(make_app())
    capture_chess_logs.records.clear()
    for _ in range(120):
        clock.advance(16)
        app.draw_frame()
    offenders = _chess_records(capture_chess_logs)
    assert offenders == [], "draw_frame must not log per frame: " + "; ".join(
        f"{r.name}:{r.getMessage()}" for r in offenders)


def test_finished_game_result_final_logs_exactly_once_across_many_frames(
        monkeypatch, capture_chess_logs):
    """The result screen stays on screen for many frames after a resign;
    _on_result_final is re-entered every one of those frames via
    _update_result_pending, so its own log line must be guarded to fire once."""
    clock = FakeTicks()
    install_clock(monkeypatch, clock)
    app = start_game(make_app())
    capture_chess_logs.records.clear()
    app.game._perform_resign()
    for _ in range(120):
        clock.advance(16)
        app.draw_frame()
    game_end_lines = [r for r in _chess_records(capture_chess_logs)
                      if r.getMessage().startswith("game end result=")]
    assert len(game_end_lines) == 1, (
        f"expected exactly one game-end breadcrumb, got {len(game_end_lines)}")


def _write_review_pgn(tmp_path):
    path = tmp_path / "hygiene-review.pgn"
    path.write_text(
        '[White "alice"]\n[Black "bob"]\n[Result "1-0"]\n'
        '[TimeControl "300+0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n',
        encoding="utf-8",
    )
    return path


def _enter_history(app):
    app.menu.goto_history()
    app.draw_frame()
    return app


def _enter_options(app):
    app.menu.goto_view("options")
    app.draw_frame()
    return app


def _enter_review(app, path):
    app.request_nav(Nav("review", {"pgn_path": str(path), "return_to": "menu"}))
    app._execute_pending_nav()
    app.draw_frame()
    return app


def _online_ish(app):
    """Flips a live local game into the online-shaped code paths (variant +
    local_color) without a real client/socket — cheap, but exercises the
    per-frame branches that check `variant == "online"` in draw_frame's call
    chain (auto-end countdown, connection-state strip lookups, ...)."""
    app.game.variant = "online"
    app.game.match.local_color = PieceColor.WHITE
    app.draw_frame()
    return app


def _idle_frame_offenders(app, clock, capture, n=100, step=16):
    app.coordinator._last_reconnect_probe_ms = clock()
    capture.records.clear()
    for _ in range(n):
        clock.advance(step)
        app.draw_frame()
    return _chess_records(capture)


def test_idle_draw_frame_emits_no_log_records_on_every_screen(
        monkeypatch, tmp_path, capture_chess_logs):
    """The single-screen regression above (test_idle_draw_frame_emits_no_log_records)
    only covers a live game. Every screen's pure draw_frame() work — menu,
    history, review, and a local game reshaped into the online code paths —
    must be equally silent."""
    clock = FakeTicks()
    install_clock(monkeypatch, clock)

    scenarios = [
        ("menu", make_app()),
        ("game", start_game(make_app())),
        ("online-ish game", _online_ish(start_game(make_app()))),
        ("history", _enter_history(make_app())),
        ("options", _enter_options(make_app())),
        ("review", _enter_review(make_app(), _write_review_pgn(tmp_path))),
    ]

    for label, scenario_app in scenarios:
        offenders = _idle_frame_offenders(scenario_app, clock, capture_chess_logs)
        assert offenders == [], f"{label} draw_frame must not log per frame: " + "; ".join(
            f"{r.name}:{r.getMessage()}" for r in offenders)


INFO_ALLOWLIST_PREFIXES = (
    "client starting",
    "pygame init ok",
    "frontend ready",
    "setting persisted",
    "server target set ",
    "screen switch",
    "menu view",
    "game start",
    "game enter",
    "game end",
    "review enter",
    "undo ply=",
    "takeback applied ply=",
    "takeback requested",
    "draw offer sent",
    "focus mode toggled",
    "give time granted",
    "give time received",
    "share marks toggled",
    "quick chat sent",
    "skillcheck fired",
    "skillcheck resolved",
    "skillcheck move locked",
    "promoted unconfirmed online result locally",
    "match found",
    "online flow begin",
    "online flow cancel",
    "connect to ",
    "online session teardown",
    "reclaim available",
    "reclaim dropped",
    "rematch offer received",
    "rematch response sent",
    "rematch requested",
    "rematch update event=",
    "offer received type=",
    "offer responded type=",
    "reconnect: resume begin",
    "reconnect: resume ok",
    "connect addr=",
    "matchmake ok",
    "reconnect-resume addr=",
    "reconnect succeeded; resuming game",
    "ws dropped mid-game; attempting reconnect",
    "session ended state=disconnected",
    "state-sync ok room=",
    "ws result reason=",
    "ws connecting ",
)
"""Every INFO-level prefix that should ever appear under a chess.* logger,
across the whole client (menu/game/history/review + the online coordinator +
the background OnlineClient/transport threads) — not just the ones the
scripted session below actually exercises. One entry per event kind; a
message is allowed if it starts with one of these, everything else is a
stray internal leaking at INFO."""


def _run_scripted_local_session(tmp_path):
    """menu -> start local game -> play a move each side -> resign -> back to
    menu -> history -> open review -> back -> options -> back to play -> quit
    flush, entirely offline."""
    app = make_app()
    app._on_start_game({
        "mode": "single_screen", "nickname": "alice", "side": "white",
        "time_minutes": 5, "increment_seconds": 0,
    })
    app.draw_frame()
    assert app.game.match.apply_san("e4").legal
    assert app.game.match.apply_san("e5").legal
    app.game._perform_resign()
    app.draw_frame()
    saved_path = app.game.result_flow._last_saved_pgn_path
    assert saved_path is not None

    app._on_back_to_menu()
    app.menu.goto_history()
    app._execute_pending_nav()
    app._open_pgn_review(saved_path)
    app._execute_pending_nav()
    app.review._on_menu()
    app._execute_pending_nav()

    app.menu.goto_view("options")
    app.menu.goto_view("play")

    for screen in app.screens.values():
        screen.on_app_exit()
    app.coordinator.on_app_exit()
    app.settings._flush_deferred_env_writes(force=True)
    return app


def test_scripted_local_session_info_lines_match_the_allowlist(
        monkeypatch, tmp_path, caplog):
    """Level-correctness regression: every chess.* INFO record from a full
    menu -> game -> resign -> history -> review -> quit pass must be a
    documented story beat, never a stray internal."""
    monkeypatch.setenv("CHESS_DATA_DIR", str(tmp_path))
    with caplog.at_level(logging.INFO):
        _run_scripted_local_session(tmp_path)

    info_records = [r for r in caplog.records
                    if r.name.startswith("chess.") and r.levelno == logging.INFO]
    unexpected = [f"{r.name}:{r.getMessage()}" for r in info_records
                  if not r.getMessage().startswith(INFO_ALLOWLIST_PREFIXES)]
    assert unexpected == [], f"unrecognized INFO lines leaking: {unexpected}"

    messages = [r.getMessage() for r in info_records]
    for expected_prefix in (
        "screen switch menu -> game", "game start mode=single_screen",
        "game enter variant=local", "game end result=",
        "screen switch game -> menu", "menu view play -> history",
        "screen switch menu -> review", "review enter path=",
        "screen switch review -> menu", "menu view history -> options",
        "menu view options -> play",
    ):
        assert any(m.startswith(expected_prefix) for m in messages), (
            f"expected story beat missing: {expected_prefix!r} not in {messages}")


def test_an_options_pass_over_the_server_rows_emits_no_stray_info(monkeypatch, caplog):
    """A full Options pass over the #89 server rows — enter, flip the mode
    (the ONE new INFO line, a user action), let the reveal animate closed for
    many frames, leave — must emit only allowlisted INFO. The reveal tick, the
    probe status getters, and the exit-time probe reset all run in that window
    and must stay silent."""
    import pygame as pg

    try:
        app = _enter_options(make_app())
        with caplog.at_level(logging.INFO):
            app.settings._apply_server_mode("official")
            app.coordinator._last_reconnect_probe_ms = pg.time.get_ticks()
            for _ in range(40):
                app.draw_frame()
            app.menu.goto_view("play")

        info_records = [r for r in caplog.records
                        if r.name.startswith("chess.") and r.levelno == logging.INFO]
        messages = [r.getMessage() for r in info_records]
        unexpected = [f"{r.name}:{r.getMessage()}" for r in info_records
                      if not r.getMessage().startswith(INFO_ALLOWLIST_PREFIXES)]
        assert unexpected == [], f"unrecognized INFO lines leaking: {unexpected}"
        assert any(m.startswith("server target set mode=official") for m in messages), \
            "the mode switch is a documented story beat"
    finally:
        for var in ("CHESS_SERVER_MODE", "CHESS_CUSTOM_SERVER_ADDR", "CHESS_SERVER_ADDR"):
            os.environ.pop(var, None)


def _slice_function(path, name):
    """The function's own lines, docstring-blind. The span comes from the AST of
    the real file (a text scan for `def <name>(` could match inside a string and
    slice the wrong region), and the text comes from the blanked source, whose
    line numbers are preserved -- so a docstring inside a per-frame function is
    invisible to the log-call regex below, while its code is not."""
    with open(path, encoding="utf-8") as f:
        original = f.read()
    tree = ast.parse(original, filename=path)
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(nodes) == 1, f"expected exactly one def {name} in {path}, found {len(nodes)}"
    node = nodes[0]
    lines = read_source_without_docstrings(path).splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno])


def test_no_log_calls_in_known_per_frame_functions():
    """Static tripwire: a future PR pasting a debug log directly into
    draw_frame/draw_board/etc. fails here even before it ships a runtime leak."""
    offenders = []
    for rel_path, names in FRAME_FUNCTIONS.items():
        full = os.path.join(REPO_ROOT, rel_path)
        for name in names:
            body = _slice_function(full, name)
            if STEADY_STATE_LOG_CALL_RE.search(body):
                offenders.append(f"{rel_path}:{name}")
    assert offenders == [], f"per-frame functions must not log directly: {offenders}"


def _log_call_spans(text):
    spans = []
    for m in LOG_CALL_RE.finditer(text):
        i = m.end() - 1
        depth = 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append(text[m.start():i + 1])
    return spans


def test_no_secret_or_seed_in_any_log_call():
    """The skill-check `room.skillcheck_secret` (which-fires selector) and the
    per-check geometry `seed` must never be logged, on either side."""
    assert os.path.isdir(PACKAGE_ROOT), f"expected a walkable package dir at {PACKAGE_ROOT}"
    offenders = []
    scanned = 0
    for dirpath, _, filenames in os.walk(PACKAGE_ROOT):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            scanned += 1
            text = read_source_without_docstrings(path)
            for span in _log_call_spans(text):
                if re.search(r"secret|seed", span, re.IGNORECASE):
                    offenders.append(f"{os.path.relpath(path, REPO_ROOT)}: {span.splitlines()[0]}")
    assert scanned > 50, f"only scanned {scanned} files, guard root is likely wrong"
    assert offenders == [], f"log calls must never reference secret/seed: {offenders}"
