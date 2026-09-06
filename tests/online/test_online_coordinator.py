"""OnlineCoordinator: the client-side service owning the online connection
lifecycle end-to-end (step 5 of the screen-architecture refactor).

Covers the subscriber protocol (subscribe/unsubscribe, the double-subscribe
guard, the assert on a board-level event arriving with no subscriber), the
no-subscriber RESULT fallback straight to app.game (the menu rematch-window
save pin), the match-found -> Nav("game", plain payload) -> subscribed
GameScreen flow, the full back-to-menu rematch window, the menu Reconnect
button glue, the coordinator.update-before-screen.update frame ordering, and
the on_app_exit teardown fan-out."""

import logging
from unittest.mock import MagicMock

import pygame as pg
import pytest

from tests.conftest import pygame_display
from chessshootout.backend.pieces import PieceColor
from chessshootout.backend.utils import Square
from chessshootout.frontend import online_coordinator as coordinator_module
from chessshootout.frontend.online_coordinator import (
    APPLY_FAILED_LABEL, ONLINE_HARD_FAILURE_REASONS, ONLINE_TRANSIENT_REASON_LABELS,
    RECONNECT_PROBE_MAX_ATTEMPTS, TOAST_REASON_MAX_CHARS,
)
from chessshootout.frontend.screens.game import IDLE_RESIGN_NOTICE_SECONDS, IdleWindow
from chessshootout.online.client import Event, OnlineClient
from chessshootout.server.protocol import (
    MIN_CLIENT_VERSION, MatchmakeRequest, Reason, client_version_outdated,
)
from tests.helpers import (
    fake_uuid4, make_app, online_start_payload as _online_start_payload,
    start_single_screen,
)


_pygame_init = pygame_display(1000, 800)


def _mock_client(room_id="room-1"):
    client = MagicMock()
    client.room_id = room_id
    return client


def _wired_app(**overrides):
    app = make_app(1000, 800)
    app.coordinator.client = _mock_client()
    app.coordinator._start_online_game(_online_start_payload(**overrides))
    return app


def test_game_screen_subscribes_on_online_entry_and_unsubscribes_on_exit():
    app = make_app(1000, 800)
    assert app.coordinator._subscriber is None

    app.coordinator.client = _mock_client()
    app.coordinator._start_online_game(_online_start_payload())
    assert app.coordinator._subscriber is app.game

    app.switch_to("menu")
    assert app.coordinator._subscriber is None


def test_local_game_entry_never_subscribes():
    app = start_single_screen(make_app(1000, 800))
    assert app.screen.name == "game"
    assert app.coordinator._subscriber is None


def test_double_subscribe_asserts():
    app = make_app(1000, 800)
    app.coordinator.subscribe(app.game)
    with pytest.raises(AssertionError):
        app.coordinator.subscribe(app.game)


def test_unsubscribe_is_always_safe():
    app = make_app(1000, 800)
    app.coordinator.unsubscribe(app.game)
    assert app.coordinator._subscriber is None
    app.coordinator.subscribe(app.game)
    app.coordinator.unsubscribe(app.game)
    app.coordinator.unsubscribe(app.game)
    assert app.coordinator._subscriber is None


def test_board_level_event_with_no_subscriber_is_dropped_and_logged(caplog):
    """A board-level event with nobody listening is impossible by design, so it is
    an ERROR — but it must not take the frame down with it. _drain_online_inbound
    already swallows every handler exception, so raising here only ever degraded
    into "log it and carry on" anyway; doing that explicitly is the same behavior
    with one strategy instead of two."""
    app = make_app(1000, 800)
    assert app.coordinator._subscriber is None
    with caplog.at_level(logging.ERROR, logger="chess.frontend"):
        app.coordinator._forward_board_event("on_remote_move", {"from": "e2", "to": "e4"})
    assert any("no subscriber" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("method_name", [
    "on_remote_move", "on_skillcheck_required", "on_takeback",
    "on_give_time", "on_spectate",
])
def test_every_board_level_method_is_gated_by_the_subscriber_check(method_name, caplog):
    app = make_app(1000, 800)
    with caplog.at_level(logging.ERROR, logger="chess.frontend"):
        app.coordinator._forward_board_event(method_name, {})
    assert any("no subscriber" in r.getMessage() for r in caplog.records)


def test_result_with_no_subscriber_still_saves_and_scores(tmp_path, monkeypatch):
    """The no-subscriber RESULT pin: forwarding must land on the persistent
    GameScreen object directly, so the real save choke point (_on_result_final,
    which also awards the series score) still fires even though nobody
    subscribed to receive it."""
    monkeypatch.setenv("CHESS_DATA_DIR", str(tmp_path))
    app = _wired_app()
    app.game.match.try_move(Square(6, 4), Square(4, 4))
    app.coordinator.unsubscribe(app.game)
    assert app.coordinator._subscriber is None

    app.coordinator._handle_online_result({"reason": "resignation", "winner_color": "white"})

    assert app.game.manual_result == "white_wins_by_resignation"
    assert app.game.result_flow._last_saved_pgn_path is not None
    assert app.game.result_flow.series_scores["alice"] == 1.0


def test_offer_and_connection_status_never_require_a_subscriber():
    app = _wired_app()
    app.switch_to("menu")
    assert app.coordinator._subscriber is None
    app.coordinator._push_offer_banner("draw_offered")
    app.coordinator._handle_connection_status({"opp_state": "reconnecting"})
    assert app.game._opp_disconnected_at_ms is not None


def test_match_found_transition_ends_with_a_subscribed_online_game_screen():
    app = make_app(1000, 800)
    app.coordinator.client = _mock_client()
    payload = _online_start_payload(
        your_color="black", white_name="alice", black_name="bob",
        started_seconds_ago=0.0,
    )

    app.coordinator._begin_match_found_transition(payload)
    assert app.coordinator.match_found_modal.is_visible()
    assert app.coordinator._pending_game_start_payload == payload

    app.coordinator._finish_match_found()

    assert app.screen.name == "game"
    assert app.game.variant == "online"
    assert app.game._chosen_side == "black"
    assert app.game.white_name == "alice"
    assert app.game.black_name == "bob"
    assert app.coordinator._subscriber is app.game


def test_rematch_from_menu_window_end_to_end():
    app = _wired_app(white_name="alice", black_name="bob", your_color="white")
    client = app.coordinator.client
    app.game.manual_result = "white_wins_by_resignation"

    app._on_back_to_menu()
    assert app.screen.name == "menu"
    assert app.coordinator.client is client, "session retained for the rematch window"
    client.send_left_result.assert_called_once()
    assert app.coordinator._subscriber is None

    app.coordinator._handle_rematch_request()
    assert not app.coordinator.offer_banners.is_empty(), "rematch banner shows on the menu"
    assert app.game.result_menu.rematch_offered is True

    app.coordinator._accept_rematch()
    client.send_rematch_response.assert_called_once_with(True)

    fresh_payload = _online_start_payload(
        your_color="black", white_name="bob", black_name="alice",
        started_seconds_ago=0.0, rematch=True,
    )
    app.coordinator._begin_match_found_transition(fresh_payload)
    app.coordinator._finish_match_found()

    assert app.screen.name == "game"
    assert app.game.variant == "online"
    assert app.game._chosen_side == "black"
    assert app.coordinator._subscriber is app.game


def _spectate_check(kind, **overrides):
    payload = {"kind": kind, "seed": "seed-1", "value_diff": 3, "deadline_ms": 5000.0,
               "from": "e4", "to": "d5", "promotion": None, "captured_value": 3}
    payload.update(overrides)
    return payload


def _spectate_shot(**overrides):
    payload = {"elapsed_ms": 100.0, "miss_count": 0, "won": False, "progress": 1}
    payload.update(overrides)
    return payload


def test_spectate_shot_clamps_hostile_coords_and_progress(monkeypatch):
    """The mover's own shots are clamped by the whack view before they go on the
    wire, but the relayed spectate copy is unbounded on the server model — so the
    client clamps what it consumes. Coordinates land in a pg.draw.circle centre,
    which raises on a value pygame can't fit in a C int."""
    from chessshootout.frontend.screens.game import SPECTATE_PROGRESS_MAX

    app = _wired_app()
    seen = []
    monkeypatch.setattr(app.game.skillcheck_overlay, "spectate_shot",
                        lambda *a, **kw: seen.append((a, kw)))

    app.game.on_spectate(_spectate_shot(
        target_row=1e308 * 10, target_col=-4096.0, progress=10 ** 9,
        elapsed_ms=float("nan")))

    (elapsed, miss_count, won), kwargs = seen[0]
    assert kwargs["target"] is None, "a non-finite coordinate is dropped, not drawn"
    assert kwargs["progress"] == SPECTATE_PROGRESS_MAX
    assert elapsed == 0.0
    assert (miss_count, won) == (0, False)


def test_spectate_shot_keeps_honest_values_byte_identical(monkeypatch):
    app = _wired_app()
    seen = []
    monkeypatch.setattr(app.game.skillcheck_overlay, "spectate_shot",
                        lambda *a, **kw: seen.append((a, kw)))

    app.game.on_spectate(_spectate_shot(
        elapsed_ms=1234.5, miss_count=2, won=True, progress=3,
        target_row=3.25, target_col=7.75, direction="up"))

    (elapsed, miss_count, won), kwargs = seen[0]
    assert (elapsed, miss_count, won) == (1234.5, 2, True)
    assert kwargs == {"progress": 3, "direction": "up", "target": (3.25, 7.75)}


@pytest.mark.parametrize("row, col, expected", [
    pytest.param(9.5, 4.0, (8.0, 4.0), id="row_above_the_board"),
    pytest.param(-3.0, 4.0, (0.0, 4.0), id="row_below_zero"),
    pytest.param(2.0, 4096.0, (2.0, 8.0), id="col_above_the_board"),
])
def test_spectate_shot_clamps_finite_coords_into_the_board_domain(
        row, col, expected, monkeypatch):
    app = _wired_app()
    seen = []
    monkeypatch.setattr(app.game.skillcheck_overlay, "spectate_shot",
                        lambda *a, **kw: seen.append(kw))

    app.game.on_spectate(_spectate_shot(target_row=row, target_col=col))

    assert seen[0]["target"] == expected


def test_hostile_spectate_shot_survives_a_real_drawn_frame():
    """End to end with the real spectate overlay: the coordinates only reach a
    draw call on the next frame, so the frame is what has to survive."""
    app = _wired_app()
    app.coordinator.client.opp_state = "connected"
    app.game.on_spectate(_spectate_check("whack"))
    assert app.game.skillcheck_overlay.is_active()

    app.game.on_spectate(_spectate_shot(target_row=float("inf"),
                                        target_col=1e308 * 10, progress=10 ** 6))
    app.draw_frame()
    app.draw_frame()


@pytest.mark.parametrize("direction", [
    pytest.param("sideways", id="not_a_compass_point"),
    pytest.param("UP", id="right_word_wrong_case"),
    pytest.param("", id="empty_string"),
    pytest.param(7, id="not_even_a_string"),
    pytest.param(["up"], id="unhashable"),
])
def test_spectated_direction_outside_the_engine_set_is_dropped(direction, monkeypatch):
    """combo_view keys _RECEPTOR_DIRS/_DIR_ANGLE by the direction verbatim, and
    a spectate frame is applied outside _guarded_apply -- an unknown value would
    sit in _receptor_flash/_flying until the next draw() raised KeyError."""
    app = _wired_app()
    seen = []
    monkeypatch.setattr(app.game.skillcheck_overlay, "spectate_shot",
                        lambda *a, **kw: seen.append(kw))

    app.game.on_spectate(_spectate_shot(direction=direction))

    assert seen[0]["direction"] is None


def test_hostile_spectated_direction_survives_a_real_combo_draw():
    """End to end with the real combo controller: the bad direction only reaches
    the lookup tables on the next painted frame, so the frame is the assertion."""
    app = _wired_app()
    app.coordinator.client.opp_state = "connected"
    app.game.on_spectate(_spectate_check("combo"))
    controller = app.game.skillcheck_overlay._controller

    app.game.on_spectate(_spectate_shot(won=True, progress=1, direction="sideways"))

    assert "sideways" not in controller._receptor_flash
    assert all(entry[0] != "sideways" for entry in controller._flying)
    app.draw_frame()
    app.draw_frame()


def test_every_engine_direction_is_relayed_untouched(monkeypatch):
    from chessshootout.skillcheck.combo import COMBO_DIRECTIONS

    app = _wired_app()
    seen = []
    monkeypatch.setattr(app.game.skillcheck_overlay, "spectate_shot",
                        lambda *a, **kw: seen.append(kw))

    for direction in COMBO_DIRECTIONS:
        app.game.on_spectate(_spectate_shot(direction=direction))

    assert [kw["direction"] for kw in seen] == list(COMBO_DIRECTIONS)


@pytest.mark.parametrize("raw", [
    pytest.param("not-a-number", id="non_numeric_string"),
    pytest.param(None, id="missing"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative_inf"),
    pytest.param({"row": 1}, id="wrong_type_entirely"),
])
def test_finite_float_falls_back_on_anything_it_cannot_use(raw):
    """Every ms field on the wire goes through this: a float() that raises and a
    float() that succeeds into NaN/inf are both the caller's default, so no
    non-finite value ever reaches a timer or a draw call."""
    from chessshootout.frontend.screens.game import _finite_float

    assert _finite_float(raw) == 0.0
    assert _finite_float(raw, 42.5) == 42.5
    assert _finite_float(raw, None) is None


@pytest.mark.parametrize("raw", [
    pytest.param("not-a-number", id="non_numeric_string"),
    pytest.param(None, id="missing"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
])
def test_board_coord_drops_what_it_cannot_clamp(raw):
    from chessshootout.frontend.screens.game import _board_coord

    assert _board_coord(raw) is None


def test_board_coord_clamps_finite_values_to_the_board():
    from chessshootout.frontend.screens.game import _board_coord
    from chessshootout.server.protocol import SKILLCHECK_TARGET_MAX

    assert _board_coord(-1.0) == 0.0
    assert _board_coord(1e12) == SKILLCHECK_TARGET_MAX
    assert _board_coord("3.5") == 3.5, "a numeric string is still a coordinate"


@pytest.mark.parametrize("payload", [
    pytest.param({}, id="no_coords_at_all"),
    pytest.param({"target_row": 1.0}, id="row_only"),
    pytest.param({"target_col": 1.0}, id="col_only"),
    pytest.param({"target_row": 1.0, "target_col": "x"}, id="one_unusable"),
])
def test_spectate_target_needs_both_coordinates(payload):
    from chessshootout.frontend.screens.game import _spectate_target

    assert _spectate_target(payload) is None


def test_hostile_combo_spectate_progress_cannot_run_away():
    """combo_view replays the missing prompts one by one (`while self._progress <
    progress`), so an unbounded server int is an unbounded loop plus the juice
    allocations each step spawns."""
    from chessshootout.frontend.screens.game import SPECTATE_PROGRESS_MAX

    app = _wired_app()
    app.coordinator.client.opp_state = "connected"
    app.game.on_spectate(_spectate_check("combo"))
    controller = app.game.skillcheck_overlay._controller

    app.game.on_spectate(_spectate_shot(won=True, progress=10 ** 9))

    assert controller._progress <= SPECTATE_PROGRESS_MAX
    app.draw_frame()


def test_unknown_error_reason_is_truncated_before_it_is_toasted():
    """An unmapped reason is shown verbatim, and the toast sizes itself from the
    rendered text — an unbounded reason is an unbounded surface."""
    app = _wired_app()

    app.coordinator._handle_online_error({"reason": "boom" * 10_000})

    assert app.toast.is_visible()
    assert len(app.toast._bubbles[-1]["message"]) == TOAST_REASON_MAX_CHARS


def test_a_non_string_reason_falls_back_to_the_generic_label():
    app = _wired_app()
    app.coordinator._handle_online_error({"reason": {"nested": "payload"}})
    assert app.toast._bubbles[-1]["message"] == "Server error"


def test_mapped_reasons_still_toast_their_full_label():
    app = _wired_app()
    app.coordinator._handle_online_error({"reason": Reason.RATE_LIMITED})
    assert app.toast._bubbles[-1]["message"] == "Slow down a bit"


QUEUE_TIMEOUT_LABEL = ONLINE_TRANSIENT_REASON_LABELS[Reason.QUEUE_TIMEOUT]


def _search_config(**overrides):
    config = {"nickname": "alice", "time_minutes": 5, "increment_seconds": 0,
              "side": "random"}
    config.update(overrides)
    return config


def _searching_app(monkeypatch, client=None):
    """Drive the real matchmaking entry point up to the "Searching…" wait modal.

    The coordinator builds its own client, so the factory is swapped for a
    stand-in; OnlineClient.connect is neutered on top of that so a test passing
    a real (hand-wired) client still never opens a socket or spawns a thread."""
    monkeypatch.setenv("CHESS_SERVER_ADDR", "127.0.0.1:9")
    monkeypatch.setattr(OnlineClient, "connect", lambda self, addr, request: None)
    app = make_app(1000, 800)
    monkeypatch.setattr(coordinator_module, "OnlineClient",
                        lambda: client if client is not None else _mock_client())
    app.coordinator._begin_online_flow(_search_config())
    assert app.coordinator.wait_modal.is_visible()
    assert app.menu.play_view_visible() is False
    return app


class _ClosingLoop:
    """An event loop that reports itself running and then raises the instant work
    is scheduled on it — the window the server's close(4004) opens right behind
    the queue_timeout frame: the session coroutine finishes and the loop closes
    between the is_running() check and the schedule."""

    def is_running(self):
        return True

    def is_closed(self):
        return False

    def call_soon_threadsafe(self, *args, **kwargs):
        raise RuntimeError("Event loop is closed")


def _queued_client_with_dead_loop(reason):
    client = OnlineClient()
    client._loop = _ClosingLoop()
    client._in_queue = True
    client._room_id = "room-1"
    client._session_token = "tok"
    client._inbound.put(Event("error", {"reason": reason}))
    return client


def test_queue_timeout_ends_the_search_like_the_cancel_button(monkeypatch):
    """The server drops a room that has sat in the queue for QUEUE_MAX_WAIT_SECONDS
    and says so before closing the socket. The player is still queued — no game to
    fall back to — so the wait modal has to come down and the menu has to come
    back, exactly as if Cancel had been clicked, plus one explanatory toast."""
    app = _searching_app(monkeypatch)
    client = app.coordinator.client

    app.coordinator._handle_online_error({"reason": Reason.QUEUE_TIMEOUT})

    assert not app.coordinator.wait_modal.is_visible()
    assert app.toast.message == QUEUE_TIMEOUT_LABEL
    assert Reason.QUEUE_TIMEOUT not in app.toast.message, "no raw wire code in the toast"
    assert app.coordinator.client is None
    assert app.coordinator._wait_started_at_ms is None
    client.cancel_queue.assert_called_once()
    assert app.screen is app.menu
    assert app.menu.play_view_visible() is True


def test_queue_timeout_never_reaches_the_hard_failure_modal(monkeypatch):
    """Timing out of the queue is an ordinary outcome of a long search, not a
    broken connection — it must not raise the confirm+Retry modal the
    unreachable/room-full family gets, and it must toast exactly once."""
    app = _searching_app(monkeypatch)

    app.coordinator._handle_online_error({"reason": Reason.QUEUE_TIMEOUT})

    assert not app.confirm_modal.is_visible()
    assert Reason.QUEUE_TIMEOUT not in ONLINE_HARD_FAILURE_REASONS
    assert len(app.toast._bubbles) == 1


def test_search_restarts_cleanly_after_a_queue_timeout(monkeypatch):
    """Nothing about the timed-out search may linger: the next Play click has to
    build a fresh client and a fresh wait modal."""
    app = _searching_app(monkeypatch)
    app.coordinator._handle_online_error({"reason": Reason.QUEUE_TIMEOUT})

    second = _mock_client("room-2")
    monkeypatch.setattr(coordinator_module, "OnlineClient", lambda: second)
    app.coordinator._begin_online_flow(
        _search_config(time_minutes=10, increment_seconds=5))

    assert app.coordinator.client is second
    second.connect.assert_called_once()
    assert app.coordinator.wait_modal.is_visible()
    assert app.coordinator._wait_started_at_ms is not None
    assert app.coordinator._pending_game_start_payload is None
    assert app.menu.play_view_visible() is False


def test_queue_timeout_clears_the_search_even_when_the_close_lands_first(monkeypatch):
    """Ordering pin: the error frame and the close(4004) arrive back to back, so
    the session thread can already be gone by the time the frame drains the
    event. Tearing the queue down then schedules onto a loop that is closing
    underneath the caller — that used to escape as a RuntimeError, get swallowed
    by _guarded_apply, and leave "Searching…" on screen forever."""
    app = _searching_app(
        monkeypatch, client=_queued_client_with_dead_loop(Reason.QUEUE_TIMEOUT))
    client = app.coordinator.client

    app.coordinator._drain_online_inbound()

    assert app.toast.message == QUEUE_TIMEOUT_LABEL
    assert app.toast.message != APPLY_FAILED_LABEL
    assert not app.coordinator.wait_modal.is_visible()
    assert app.coordinator.client is None
    assert client.state == "disconnected"
    assert app.menu.play_view_visible() is True


def test_match_found_and_game_start_clip_oversize_names():
    """A name is rendered in full into a surface before it is clipped to the
    widget, so the length bound has to land before the modal and the strips.

    The bound under test is env.clip_nickname's, not the server validator's --
    they happen to be equal, so pinning the protocol constant here would stay
    green even if the client stopped clipping."""
    from chessshootout.infra.env import _NICKNAME_MAX_LEN

    app = make_app(1000, 800)
    app.coordinator.client = _mock_client()
    app.coordinator.client.opp_state = "connected"
    payload = _online_start_payload(white_name="a" * 400_000, black_name="bob")

    app.coordinator._begin_match_found_transition(payload)
    assert app.coordinator.match_found_modal.me_name == "a" * _NICKNAME_MAX_LEN

    app.coordinator._finish_match_found()
    assert app.game.white_name == "a" * _NICKNAME_MAX_LEN
    assert app.game.black_name == "bob"
    assert set(app.game.result_flow.series_scores) == {"a" * _NICKNAME_MAX_LEN, "bob"}
    app.draw_frame()


def test_reconnect_available_reflects_the_pending_probe_result():
    app = make_app(1000, 800)
    assert app.coordinator.reconnect_available() is False
    app.coordinator._pending_reconnect = {
        "addr": "localhost:8000", "room_id": "room-1", "session_token": "tok",
    }
    assert app.coordinator.reconnect_available() is True


def test_reconnect_delegates_to_the_probe_glue(monkeypatch):
    app = make_app(1000, 800)
    called = []
    monkeypatch.setattr(app.coordinator, "_on_reconnect_active_game",
                        lambda: called.append(True))
    app.coordinator.reconnect()
    assert called == [True]


def _pending_entry(addr="old.host:8000", room_id="room-1"):
    return {"addr": addr, "room_id": room_id, "session_token": "tok"}


def test_on_server_target_changed_drops_the_pending_reclaim_and_hides_the_button():
    """A reclaim is only valid against the host it was probed on. Switching the
    server target while one is pending left the menu offering Reconnect for the
    OLD host -- clicking it resumed a game on a server the player just left."""
    app = make_app(1000, 800)
    app.coordinator._pending_reconnect = _pending_entry()
    app.menu.set_reconnect_available(True)
    assert app.menu.play_view.reconnect_available is True

    app.coordinator.on_server_target_changed()

    assert app.coordinator._pending_reconnect is None
    assert app.coordinator.reconnect_available() is False
    assert app.menu.play_view.reconnect_available is False


def test_on_server_target_changed_discards_an_in_flight_probe_result(monkeypatch):
    """The probe runs on a worker thread against the address captured at spawn
    time, so clearing the pending entry is not enough -- a probe already talking
    to the old host would land its reclaim a moment later. The generation bump
    is what makes that late result unadoptable."""
    monkeypatch.setattr(coordinator_module, "probe_active_game",
                        lambda addr, uuid: {"room_id": "stale-room", "session_token": "tok"})
    monkeypatch.setattr(coordinator_module, "fetch_resume",
                        lambda addr, room, token: {"fen": ""})
    app = make_app(1000, 800)
    with app.coordinator._pending_reconnect_lock:
        app.coordinator._reconnect_probe_gen += 1
        spawn_gen = app.coordinator._reconnect_probe_gen

    app.coordinator.on_server_target_changed()
    app.coordinator._reconnect_probe_worker("old.host:8000", "uuid", spawn_gen)

    assert app.coordinator._pending_reconnect is None
    assert app.coordinator.reconnect_available() is False

    app.coordinator._reconnect_probe_worker(
        "new.host:8000", "uuid", app.coordinator._reconnect_probe_gen)
    assert app.coordinator._pending_reconnect is not None


def test_on_server_target_changed_reopens_probing_for_the_new_host():
    """The attempt budget belongs to the host that burned it. An unreachable old
    address that exhausted RECONNECT_PROBE_MAX_ATTEMPTS must not leave the new
    target unprobed forever, so the counter and the interval gate both reset."""
    app = make_app(1000, 800)
    app.coordinator._reconnect_probe_attempts = coordinator_module.RECONNECT_PROBE_MAX_ATTEMPTS
    app.coordinator._last_reconnect_probe_ms = pg.time.get_ticks()

    app.coordinator.on_server_target_changed()

    assert app.coordinator._reconnect_probe_attempts == 0
    assert app.coordinator._last_reconnect_probe_ms == 0


def test_on_server_target_changed_with_nothing_pending_is_a_safe_no_op(caplog):
    """Every server-row edit calls this, and most of them happen with no reclaim
    in sight -- it must stay silent and leave the button off."""
    app = make_app(1000, 800)
    assert app.coordinator._pending_reconnect is None

    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        app.coordinator.on_server_target_changed()
        app.coordinator.on_server_target_changed()

    assert app.coordinator._pending_reconnect is None
    assert app.coordinator.reconnect_available() is False
    assert app.menu.play_view.reconnect_available is False
    assert [r for r in caplog.records if r.name == "chess.frontend"] == []


def test_coordinator_update_runs_before_screen_update(monkeypatch):
    app = start_single_screen(make_app(1000, 800))
    order = []

    real_coordinator_update = app.coordinator.update

    def coordinator_spy(now):
        order.append("coordinator")
        real_coordinator_update(now)

    real_screen_update = app.screen.update

    def screen_spy(now):
        order.append("screen")
        return real_screen_update(now)

    monkeypatch.setattr(app.coordinator, "update", coordinator_spy)
    monkeypatch.setattr(app.screen, "update", screen_spy)

    app.draw_frame()

    assert order == ["coordinator", "screen"]


def test_game_screen_on_app_exit_flushes_a_pending_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_DATA_DIR", str(tmp_path))
    app = start_single_screen(make_app(1000, 800))
    app.game.match.try_move(Square(6, 4), Square(4, 4))
    app.game.manual_result = "white_wins_by_resignation"
    assert app.game.result_flow._last_saved_pgn_path is None

    app.game.on_app_exit()

    assert app.game.result_flow._last_saved_pgn_path is not None


def test_game_screen_on_app_exit_is_a_no_op_without_a_pending_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_DATA_DIR", str(tmp_path))
    app = start_single_screen(make_app(1000, 800))
    app.game.on_app_exit()
    assert app.game.result_flow._last_saved_pgn_path is None


def test_coordinator_on_app_exit_disconnects_the_client():
    app = make_app(1000, 800)
    client = MagicMock()
    app.coordinator.client = client
    app.coordinator.on_app_exit()
    client.disconnect.assert_called_once()


def test_coordinator_on_app_exit_is_a_no_op_with_no_client():
    app = make_app(1000, 800)
    app.coordinator.client = None
    app.coordinator.on_app_exit()
    assert app.coordinator.client is None


def test_run_teardown_fans_out_on_app_exit_to_every_screen_and_the_coordinator(monkeypatch):
    app = make_app(1000, 800)
    app.running = False
    calls = []
    for name, screen in app.screens.items():
        monkeypatch.setattr(screen, "on_app_exit", lambda name=name: calls.append(name))
    monkeypatch.setattr(app.coordinator, "on_app_exit", lambda: calls.append("coordinator"))
    monkeypatch.setattr(app.chrome, "shutdown", lambda: None)
    monkeypatch.setattr(pg, "quit", lambda: None)

    app.run()

    assert set(calls) == {"menu", "game", "review", "coordinator"}


def test_search_cancel_on_menu_does_not_self_switch(caplog):
    """Cancelling matchmaking happens ON the menu screen (the wait modal covers
    it) — _return_to_menu_card must re-show the start card without a pointless
    menu -> menu exit/enter cycle polluting the lifecycle log."""
    app = make_app(1000, 800)
    app.menu.hide_play_view()
    menu = app.screen
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        app.coordinator._return_to_menu_card()
    assert app.screen is menu
    assert app.menu.play_view_visible() is True
    assert not any("screen switch" in r.getMessage() for r in caplog.records)


def test_return_to_menu_from_game_still_switches():
    app = start_single_screen(make_app(1000, 800))
    app.coordinator._return_to_menu_card()
    assert app.screen is app.menu
    assert app.menu.play_view_visible() is True


def test_every_send_goes_through_the_coordinator_not_around_it():
    """The client's lifecycle (connect/disconnect/reconnect/retain-for-rematch) is the
    coordinator's job, so nothing else may hold a reference to it. Six sends used to
    reach past the coordinator into .client and re-implement the None check in three
    modules; a guard here is cheaper than finding the seventh by hand."""
    import ast
    import os
    import chessshootout

    root = os.path.dirname(os.path.abspath(chessshootout.__file__))
    coordinator_path = os.path.join(root, "frontend", "online_coordinator.py")
    scanned = 0
    offenders = []
    for dirpath, _, filenames in os.walk(os.path.join(root, "frontend")):
        if "__pycache__" in dirpath:
            continue
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            if path == coordinator_path:
                continue
            scanned += 1
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "coordinator"
                        and node.attr == "client"):
                    offenders.append(f"{os.path.relpath(path, root)}:{node.lineno}")
    assert scanned >= 25, f"only scanned {scanned} frontend files, guard root is likely wrong"
    assert offenders == [], (
        f"reach past the coordinator to its client: {offenders} — add a send_* facade "
        f"on OnlineCoordinator instead")


def test_send_facades_are_inert_with_no_client():
    app = make_app(1000, 800)
    assert app.coordinator.client is None

    assert app.coordinator.send_resign() is False
    assert app.coordinator.send_draw_offer() is False
    assert app.coordinator.send_takeback_request() is False
    app.coordinator.send_give_time(1500)
    app.coordinator.send_skill_check_shot(120.0)
    app.coordinator.send_move("e2", "e4", None)
    assert app.coordinator.is_connected() is False
    assert app.coordinator.opponent_state() is None
    assert app.coordinator.ping_ms() is None


def test_remote_move_dismisses_the_draw_offer_banner():
    """The server silently invalidates a pending offer the instant a move lands
    (handlers.py's _apply_move). The client mirrors that off the move_applied
    event itself — no decline is ever sent, since the server already cleared it."""
    app = _wired_app()
    client = app.coordinator.client
    app.game.on_remote_move = lambda payload: None
    app.coordinator._push_offer_banner("draw_offered")
    assert not app.coordinator.offer_banners.is_empty()

    app.coordinator._handle_remote_move_applied({"from": "e2", "to": "e4", "san": "e4"})

    assert app.coordinator.offer_banners.is_empty()
    client.send_draw_response.assert_not_called()


def test_remote_move_dismisses_the_takeback_offer_banner():
    app = _wired_app()
    client = app.coordinator.client
    app.game.on_remote_move = lambda payload: None
    app.coordinator._push_offer_banner("takeback_offered")
    assert not app.coordinator.offer_banners.is_empty()

    app.coordinator._handle_remote_move_applied({"from": "e2", "to": "e4", "san": "e4"})

    assert app.coordinator.offer_banners.is_empty()
    client.send_takeback_response.assert_not_called()


def test_resyncing_remote_move_applied_is_dropped_before_dismissing_offers():
    """_handle_remote_move_applied bails out on the pre-existing _resyncing guard
    before it ever reaches the dismiss helper — a stray move_applied mid-resync
    must not touch banner state at all (in practice _begin_resync already cleared
    it, and resume carries no offer field to rebuild it from)."""
    app = _wired_app()
    app.coordinator._push_offer_banner("draw_offered")
    app.coordinator._resyncing = True

    app.coordinator._handle_remote_move_applied({"from": "e2", "to": "e4", "san": "e4"})

    assert app.coordinator.offer_banners.count() == 1


def test_local_move_send_dismisses_an_incoming_takeback_offer_banner():
    """The recipient scenario from the bug report: opponent offers a takeback,
    I ignore it and move instead — that is the implicit decline. send_local_move
    fires only after the move already landed on my own board (Match._fire_local_move),
    so dismissing here is never premature."""
    app = _wired_app()
    client = app.coordinator.client
    app.coordinator._push_offer_banner("takeback_offered")
    assert not app.coordinator.offer_banners.is_empty()

    app.coordinator.send_local_move(Square(6, 4), Square(4, 4), None)

    assert app.coordinator.offer_banners.is_empty()
    client.send_takeback_response.assert_not_called()


def test_local_move_send_dismisses_an_incoming_draw_offer_banner():
    app = _wired_app()
    client = app.coordinator.client
    app.coordinator._push_offer_banner("draw_offered")

    app.coordinator.send_local_move(Square(6, 4), Square(4, 4), None)

    assert app.coordinator.offer_banners.is_empty()
    client.send_draw_response.assert_not_called()


def test_move_landing_never_dismisses_the_rematch_banner():
    """Rematch offers live post-game, when no move can land — but the dismiss
    helper must stay scoped to draw/takeback regardless, so a future rematch-window
    edge case can't silently eat the rematch banner too."""
    app = _wired_app()
    app.coordinator._handle_rematch_request()
    assert app.coordinator.offer_banners.count() == 1

    app.coordinator.send_local_move(Square(6, 4), Square(4, 4), None)

    assert app.coordinator.offer_banners.count() == 1
    assert app.coordinator._rematch_offered is True


def test_unbind_clears_every_field_that_gates_online_behaviour():
    """Four call sites cleared four different subsets of the same nine fields, and they
    had drifted — a teardown left the idle window and the disconnect stamps set.
    They are inert once variant is back to local, but only because of that; one unbind
    now clears the whole set."""
    app = make_app(1000, 800)
    game = app.game
    game.variant = "online"
    game.match.local_color = PieceColor.WHITE
    game.match.on_local_move_applied = lambda *a: None
    game._idle_window = IdleWindow(Reason.ABORTED, PieceColor.WHITE, 12345, 60.0)
    game._opp_disconnected_at_ms = 6789
    game._local_disconnected_at_ms = 4321
    app.coordinator._prev_online_state = "reconnecting"

    app.coordinator.unbind_game_from_online()

    assert game.variant == "local"
    assert game.match.local_color is None
    assert game.match.on_local_move_applied is None
    assert game._idle_window is None
    assert game._opp_disconnected_at_ms is None
    assert game._local_disconnected_at_ms is None
    assert app.coordinator._prev_online_state is None


class _IdleWindowSubscriber:

    def __init__(self):
        self.seen = []

    def on_idle_window(self, payload):
        self.seen.append(payload)


def test_idle_window_event_routes_to_the_subscriber():
    """idle_window uses the subscriber-else-app.game fallback form (like result):
    with a subscriber bound, the payload lands there and app.game's own window
    stays untouched — the event went through the subscriber, not around it."""
    app = make_app(1000, 800)
    sub = _IdleWindowSubscriber()
    app.coordinator.subscribe(sub)
    payload = {"outcome": "aborted", "color": "white", "seconds_remaining": 42.0}

    app.coordinator._handle_online_event(Event("idle_window", payload))

    assert sub.seen == [payload]
    assert app.game._idle_window is None


def test_idle_window_event_without_a_subscriber_falls_back_to_app_game():
    """Strip-level state lives on the persistent GameScreen object, which exists
    even when it is not the active screen — an idle window pushed while nobody
    is subscribed must still arm the badge there (same fallback as result), for
    as long as the screen is still bound to the online session."""
    app = make_app(1000, 800)
    app.game.variant = "online"
    app.game.match.local_color = PieceColor.BLACK
    assert app.coordinator._subscriber is None

    app.coordinator._handle_online_event(Event("idle_window", {
        "outcome": "aborted", "color": "white", "seconds_remaining": 30.0,
    }))

    window = app.game._idle_window
    assert window is not None
    assert window.outcome == Reason.ABORTED
    assert window.color == PieceColor.WHITE


def test_a_late_push_after_back_to_menu_never_arms_or_toasts():
    """on_idle_window is gated on the online context — the same
    variant+local_color pair _compute_auto_end reads. After back-to-menu
    unbinds the screen, a straggler push (or one against a plain local game)
    must not arm a badge or toast an idle notice over it."""
    app = _wired_app()
    app._on_back_to_menu()
    assert app.game.variant == "local"

    app.coordinator._handle_online_event(Event("idle_window", {
        "outcome": "resignation", "color": "black", "seconds_remaining": 5.0,
    }))

    assert app.game._idle_window is None
    assert not any(b["key"] == "idle_resign" for b in app.toast._bubbles)


def test_the_waiting_side_gets_one_idle_notice_toast():
    """The notice is time-gated on IDLE_RESIGN_NOTICE_SECONDS: the forced
    ply-2 push carries the full window, and refresh pushes keep carrying nearly
    all of it while the opponent is actively present, so none of those may
    toast. At or under the threshold — a genuine idle spell — the toast shows
    once, keyed 'idle_resign' so repeat pushes re-stamp the same bubble instead
    of stacking new ones. Abort-window pushes never toast, however low they
    burn — nobody loses on those."""
    app = _wired_app()
    app.coordinator._handle_idle_window({
        "outcome": "resignation", "color": "black", "seconds_remaining": 60.0,
    })
    assert not any(b["key"] == "idle_resign" for b in app.toast._bubbles), \
        "a fresh full window means the opponent just moved — no idle chatter"

    app.coordinator._handle_idle_window({
        "outcome": "aborted", "color": "black",
        "seconds_remaining": IDLE_RESIGN_NOTICE_SECONDS - 10.0,
    })
    assert not any(b["key"] == "idle_resign" for b in app.toast._bubbles)

    payload = {"outcome": "resignation", "color": "black",
               "seconds_remaining": IDLE_RESIGN_NOTICE_SECONDS}
    app.coordinator._handle_idle_window(payload)
    app.coordinator._handle_idle_window(payload)

    bubbles = [b for b in app.toast._bubbles if b["key"] == "idle_resign"]
    assert len(bubbles) == 1
    assert "bob" in bubbles[0]["message"]


def test_the_idler_gets_no_notice_toast():
    """The idler already sees the Resign-in badge on their own strip; toasting
    them too would double-announce their own countdown. Remaining time is under
    the notice threshold, so the guard under test is the color one."""
    app = _wired_app()
    app.coordinator._handle_idle_window({
        "outcome": "resignation", "color": "white",
        "seconds_remaining": IDLE_RESIGN_NOTICE_SECONDS - 10.0,
    })
    assert app.game._idle_window is not None
    assert not any(b["key"] == "idle_resign" for b in app.toast._bubbles)


def test_resume_goes_through_the_subscriber_protocol():
    """game_resumed is the biggest board-level event there is — it rebuilds the whole
    match. It used to be the ONE event the coordinator applied itself, reaching into
    GameScreen's match/board/skillcheck_session directly while the other eight went
    through the typed subscriber. It now goes through on_resume like the rest."""
    app = make_app(1000, 800)
    app.game.variant = "online"
    app.game._time_control = (300, 0)
    app.coordinator.subscribe(app.game)
    seen = []
    app.game.on_resume = lambda payload: seen.append(payload)

    payload = {"fen": "", "move_history": [{"san": "e4"}], "clock": {}}
    app.coordinator._handle_game_resumed(payload)

    assert seen == [payload]
    assert app.coordinator._resyncing is False


def test_resume_with_an_inactive_game_screen_still_reaches_the_game():
    """Same fallback the result path has: app.game exists even when it is not the
    active screen, so a resume for a live online game is never dropped just because
    nobody is subscribed."""
    app = make_app(1000, 800)
    app.game.variant = "online"
    app.game._time_control = (300, 0)
    assert app.coordinator._subscriber is None

    app.coordinator._handle_game_resumed({
        "fen": "",
        "move_history": [{"san": "e4"}, {"san": "e5"}],
        "clock": {"white_remaining": 111.0, "black_remaining": 99.0, "running_for": None},
    })

    assert [e.san for e in app.game.match.move_history] == ["e4", "e5"]
    assert app.game.match.clock.white_remaining == 111.0


def test_teardown_on_menu_does_not_reenter_the_menu():
    """REGRESSION: the rematch window dying (opponent left, room expired) while the
    player already sits on the menu used to switch_to("menu") unconditionally — a
    self-switch runs the full exit->enter cycle, replaying the menu entry
    transitions as a visible interface reload. On the menu, teardown now only
    surfaces the Play view; the full switch is reserved for arriving from the
    game screen."""
    app = _wired_app()
    app.switch_to("menu")
    enter_spy = MagicMock(wraps=app.menu.enter)
    app.menu.enter = enter_spy
    app.coordinator._tear_down_online_session("test")
    assert app.screen is app.menu
    enter_spy.assert_not_called()
    assert app.menu.play_view_visible()


def test_teardown_from_game_screen_still_returns_to_menu():
    app = _wired_app()
    assert app.screen is app.game
    app.coordinator._tear_down_online_session("test")
    assert app.screen is app.menu


def test_teardown_returns_through_the_shared_menu_card_route():
    """Teardown carried its own copy of _return_to_menu_card's body. Going through
    the real thing means a torn-down session also lands on the menu with a fresh
    set of Reconnect probe attempts, like every other route back -- otherwise a
    session that died after three fruitless probes never probes again."""
    app = _wired_app()
    app.coordinator._reconnect_probe_attempts = RECONNECT_PROBE_MAX_ATTEMPTS

    app.coordinator._tear_down_online_session("test")

    assert app.screen is app.menu
    assert app.menu.play_view_visible()
    assert app.coordinator._reconnect_probe_attempts == 0


def test_opponent_left_while_viewing_the_result_does_not_yank_to_menu():
    """REGRESSION (v2.10.0 live smoke): the leaver's grace had already elapsed
    pre-result, so result and rematch_update(opponent_left) arrived in the same
    drain and the teardown navigation stole the screen 2ms after the VICTORY
    modal appeared. When the game screen is showing a final result, ending the
    rematch window drops the session but stays put -- the player leaves via the
    result modal's own buttons."""
    app = _wired_app()
    assert app.screen is app.game
    app.coordinator._handle_online_result(
        {"reason": "abandonment", "winner_color": "black"})
    assert app.game.current_result() is not None
    app.coordinator._handle_rematch_update({"event": "opponent_left"})
    assert app.screen is app.game, "stay on the result screen"
    assert app.coordinator.client is None, "session is still torn down"
    assert app.game.current_result() is not None, "result stays adopted"
    assert app.toast.is_visible()


def test_the_matchmake_request_states_which_build_this_client_is(monkeypatch):
    """The request dict is the SINGLE place `client_version` is stamped -- the
    transport only forwards the finished model. Read straight off
    paths.get_app_version(), which is empty in a source checkout and the release
    version in a packaged build, so what the gate judges is what the footer
    shows."""
    app = make_app(1000, 800)
    monkeypatch.setattr(coordinator_module.paths, "get_app_version", lambda: "2.13.0")
    monkeypatch.setattr(coordinator_module.env, "get_or_create_client_uuid",
                        lambda: fake_uuid4(7))
    monkeypatch.setattr(coordinator_module.env, "get_country", lambda: None)
    monkeypatch.setattr(coordinator_module.env, "get_hide_opp_marks", lambda: False)
    sent = []

    class _RecordingClient:
        """Stands in for the real OnlineClient so the request is captured
        without a socket; only the two calls this path makes are needed."""

        def __init__(self):
            self.room_id = None
            self.state = "disconnected"

        def connect(self, addr, request):
            sent.append((addr, request))

        def disconnect(self):
            pass

    monkeypatch.setattr(coordinator_module, "OnlineClient", _RecordingClient)
    app.coordinator._online_config = {"nickname": "Alice", "time_minutes": 5,
                                      "increment_seconds": 0, "side": "random"}

    app.coordinator._on_server_addr_connect("localhost:8000")

    assert len(sent) == 1
    addr, request = sent[0]
    assert addr == "localhost:8000"
    assert request["client_version"] == "2.13.0"
    assert MatchmakeRequest(**request).client_version == "2.13.0", \
        "the key has to be the model's field name, or it is silently dropped"


def test_a_source_run_states_no_version_and_is_admitted_by_the_gate(monkeypatch):
    """A checkout ships no assets/version.txt, so get_app_version() answers
    empty -- and the server's empty-version exemption is what keeps `make run`
    playable against a deployed server."""
    app = make_app(1000, 800)
    monkeypatch.setattr(coordinator_module.paths, "get_app_version", lambda: "")
    monkeypatch.setattr(coordinator_module.env, "get_or_create_client_uuid",
                        lambda: fake_uuid4(7))
    monkeypatch.setattr(coordinator_module.env, "get_country", lambda: None)
    monkeypatch.setattr(coordinator_module.env, "get_hide_opp_marks", lambda: False)
    sent = []

    class _RecordingClient:
        """Captures the request the same way, for the versionless case."""

        def __init__(self):
            self.room_id = None
            self.state = "disconnected"

        def connect(self, addr, request):
            sent.append(request)

        def disconnect(self):
            pass

    monkeypatch.setattr(coordinator_module, "OnlineClient", _RecordingClient)
    app.coordinator._online_config = {"nickname": "Alice", "time_minutes": 5,
                                      "increment_seconds": 0, "side": "random"}

    app.coordinator._on_server_addr_connect("localhost:8000")

    assert sent[0]["client_version"] == ""
    assert client_version_outdated(sent[0]["client_version"], MIN_CLIENT_VERSION) is False
