"""Give 15 sec: local clock cap, debounce, toast wording, online round-trip.

The cap (Clock.add_time) and the giver/receiver toast formatters are pure
logic; the two server tests drive a real WebSocket round-trip end-to-end.
"""
import json
import random

from unittest import mock
from unittest.mock import MagicMock

import pygame as pg
import pytest

from tests.conftest import pygame_display
from chessshootout.domain.match import ONLINE
from chessshootout.backend.pieces import PieceColor
from chessshootout.server.protocol import PROTOCOL_VERSION
from tests.server.conftest import ALICE, BOB, auth_msg


_pygame_init = pygame_display(1000, 800)


def _make_app():
    from chessshootout.frontend.frontend import Frontend
    app = Frontend(1000, 800)
    app.sound_manager = MagicMock()
    return app


def _start_local(app, time_minutes=3, incr=0):
    app._on_start_game({
        "mode": "single_screen", "nickname": "Alice",
        "time_minutes": time_minutes, "increment_seconds": incr,
        "side": "white",
    })


def _hold_give_time(app, hold_ms=0, *, over=True):
    """Simulate a press-and-hold of the give button: start on click, run the
    frame-driven tick loop as if held for `hold_ms`, then release. Mirrors the
    real draw_frame poll (mouse pressed + pointer over the button rect). A
    hold_ms of 0 is a tap (the tap-floor gives one 15s tick)."""
    app.game.give_time.on_give_time()
    if not app.game.give_time.holding:
        return
    start = app.game.give_time._give_time_hold_start_ms
    with mock.patch("pygame.mouse.get_pressed", return_value=(True, False, False)), \
         mock.patch.object(app.game.give_time, "_pointer_over_give_button", return_value=over), \
         mock.patch("pygame.time.get_ticks", return_value=start + hold_ms):
        app.game.give_time.update_give_time_hold()
    with mock.patch("pygame.mouse.get_pressed", return_value=(False, False, False)), \
         mock.patch("pygame.time.get_ticks", return_value=start + hold_ms):
        app.game.give_time.update_give_time_hold()


def test_local_tap_gives_15_to_active_clock_side():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    _hold_give_time(app, hold_ms=0)
    assert app.game.match.clock.white_remaining == 115.0
    assert not app.game.give_time.holding


def test_local_hold_ramps_15_per_100ms():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    _hold_give_time(app, hold_ms=500)
    assert app.game.match.clock.white_remaining == 175.0


def test_local_hold_caps_at_initial_seconds():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 150.0
    _hold_give_time(app, hold_ms=5000)
    assert app.game.match.clock.white_remaining == 180.0
    assert not app.game.give_time.holding


def test_local_tap_after_first_move_gives_time_to_black():
    from chessshootout.backend.utils import Square
    app = _make_app()
    _start_local(app)
    app.game.match.try_move(Square(6, 4), Square(4, 4))
    app.game.match.clock.black_remaining = 100.0
    before_white = app.game.match.clock.white_remaining
    _hold_give_time(app, hold_ms=0)
    assert app.game.match.clock.white_remaining == before_white
    assert app.game.match.clock.black_remaining == 115.0


def test_local_cooldown_blocks_immediate_second_hold():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    _hold_give_time(app, hold_ms=0)
    assert app.game.match.clock.white_remaining == 115.0
    app.game.give_time.on_give_time()
    assert not app.game.give_time.holding
    assert app.game.match.clock.white_remaining == 115.0


@pytest.mark.parametrize(
    "start_remaining, expected_remaining, toast_fmt",
    [
        pytest.param(
            180.0, 180.0, "{name} already at maximum time",
            id="full_cap_no_add_announces_maximum",
        ),
        pytest.param(
            173.0, 180.0, "Gave 7 sec to {name}",
            id="partial_cap_announces_actual_amount",
        ),
        pytest.param(
            100.0, 115.0, "Gave 15 sec to {name}",
            id="full_amount_uses_recipient_nickname",
        ),
    ],
)
def test_local_give_time_caps_and_toasts(
    monkeypatch, start_remaining, expected_remaining, toast_fmt
):
    app = _make_app()
    _start_local(app, time_minutes=3, incr=0)
    app.game.match.clock.white_remaining = start_remaining
    toast_calls = []
    monkeypatch.setattr(app.toast, "show", lambda msg, **kw: toast_calls.append(msg))
    _hold_give_time(app, hold_ms=0)
    assert app.game.match.clock.white_remaining == expected_remaining
    assert toast_calls == [toast_fmt.format(name=app.game.white_name)]


def test_local_at_max_starts_no_hold():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 180.0
    app.game.give_time.on_give_time()
    assert not app.game.give_time.holding


def test_local_noop_when_no_clock():
    app = _make_app()
    app._on_start_game({
        "mode": "single_screen", "nickname": "Alice",
        "time_minutes": None, "increment_seconds": 0,
        "side": "white",
    })
    app.game.give_time.on_give_time()
    assert app.game.match.clock is None
    assert not app.game.give_time.holding


def test_local_noop_when_game_over():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    app.game.manual_result = "white_wins_by_resignation"
    before = app.game.match.clock.white_remaining
    _hold_give_time(app, hold_ms=0)
    assert app.game.match.clock.white_remaining == before
    assert not app.game.give_time.holding


def test_hold_cancelled_on_new_game():
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    app.game.give_time.on_give_time()
    assert app.game.give_time.holding
    app.game._reset_to_new_game()
    assert not app.game.give_time.holding


@pytest.mark.parametrize("abort", ["resync", "skillcheck", "result"])
def test_hold_cancelled_on_abort_state(monkeypatch, abort):
    app = _make_app()
    _start_local(app)
    app.game.match.clock.white_remaining = 100.0
    app.game.give_time.on_give_time()
    assert app.game.give_time.holding
    if abort == "resync":
        app.coordinator._resyncing = True
    elif abort == "skillcheck":
        monkeypatch.setattr(app.game.skillcheck_overlay, "is_active", lambda: True)
    elif abort == "result":
        app.game.manual_result = "white_wins_by_resignation"
    with mock.patch("pygame.mouse.get_pressed", return_value=(True, False, False)), \
         mock.patch.object(app.game.give_time, "_pointer_over_give_button", return_value=True):
        app.game.give_time.update_give_time_hold()
    assert not app.game.give_time.holding
    assert app.game.match.clock.white_remaining == 100.0


def test_disabled_keys_excludes_give_time_when_clock_present_and_idle():
    app = _make_app()
    _start_local(app)
    app.game.give_time._last_give_time_at_ms = -10_000
    assert "give_time" not in app.game._right_menu_disabled_keys()


def test_disabled_keys_includes_give_time_during_debounce():
    app = _make_app()
    _start_local(app)
    app.game.give_time._last_give_time_at_ms = pg.time.get_ticks()
    assert "give_time" in app.game._right_menu_disabled_keys()


def test_disabled_keys_includes_give_time_when_no_clock():
    app = _make_app()
    app._on_start_game({
        "mode": "single_screen", "nickname": "Alice",
        "time_minutes": None, "increment_seconds": 0,
        "side": "white",
    })
    assert "give_time" in app.game._right_menu_disabled_keys()


def test_disabled_keys_includes_give_time_after_result():
    app = _make_app()
    _start_local(app)
    app.game.manual_result = "white_wins_by_resignation"
    assert "give_time" in app.game._right_menu_disabled_keys()


def test_online_client_send_give_time_enqueues():
    from chessshootout.online.client import OnlineClient
    client = OnlineClient()
    client._loop = MagicMock()
    client._loop.is_closed.return_value = False
    client._outbound = MagicMock()
    client.send_give_time(300)
    client._loop.call_soon_threadsafe.assert_called_once()
    args = client._loop.call_soon_threadsafe.call_args.args
    method, method_args = args[1]
    assert method == "send_give_time"
    assert method_args == (300,)


def test_online_hold_sends_one_message_on_release_with_hold_ms():
    app = _online_app()
    app.game.match.setup_clock(300, 0)
    app.game.match.clock.start()
    app.game.match.clock.black_remaining = 100.0
    _hold_give_time(app, hold_ms=500)
    app.coordinator.client.send_give_time.assert_called_once()
    (hold_ms_arg,) = app.coordinator.client.send_give_time.call_args.args
    assert hold_ms_arg == 500
    assert not app.game.give_time.holding


def test_online_hold_does_not_send_until_release():
    app = _online_app()
    app.game.match.setup_clock(300, 0)
    app.game.match.clock.start()
    app.game.match.clock.black_remaining = 100.0
    app.game.give_time.on_give_time()
    assert app.game.give_time.holding
    start = app.game.give_time._give_time_hold_start_ms
    with mock.patch("pygame.mouse.get_pressed", return_value=(True, False, False)), \
         mock.patch.object(app.game.give_time, "_pointer_over_give_button", return_value=True), \
         mock.patch("pygame.time.get_ticks", return_value=start + 300):
        app.game.give_time.update_give_time_hold()
    app.coordinator.client.send_give_time.assert_not_called()
    with mock.patch("pygame.mouse.get_pressed", return_value=(False, False, False)), \
         mock.patch("pygame.time.get_ticks", return_value=start + 300):
        app.game.give_time.update_give_time_hold()
    app.coordinator.client.send_give_time.assert_called_once()


def _online_app():
    app = _make_app()
    app.coordinator._start_online_game({
        "white_name": "Alice", "black_name": "Bob", "your_color": "white",
        "time_minutes": 5, "increment_seconds": 0,
    })
    app.coordinator.client = MagicMock()
    app.game.match.mode = ONLINE
    app.game.match.local_color = PieceColor.WHITE
    return app


@pytest.mark.parametrize(
    "granted_by, seconds_added, expected_toasts",
    [
        pytest.param("white", 15.0, ["Gave 15 sec to Bob"], id="giver_full_recipient_toast"),
        pytest.param("white", 0.0, ["Bob already at maximum time"], id="giver_cap_toast"),
        pytest.param("black", 15.0, ["Bob gave you 15 seconds"], id="receiver_giver_toast"),
        pytest.param("black", 0.0, [], id="receiver_silent_on_zero_add"),
    ],
)
def test_time_granted_routes_toast(monkeypatch, granted_by, seconds_added, expected_toasts):
    app = _online_app()
    toast_calls = []
    monkeypatch.setattr(app.toast, "show", lambda msg, **kw: toast_calls.append(msg))
    app.coordinator._handle_time_granted({
        "granted_by": granted_by, "seconds_added": seconds_added, "clock": {},
    })
    assert toast_calls == expected_toasts


def _matchmake(client, *, uuid, nickname, side, time=5, inc=0):
    return client.post("/matchmake", json={
        "version": PROTOCOL_VERSION,
        "client_uuid": uuid, "nickname": nickname,
        "time_minutes": time, "increment_seconds": inc,
        "side_preference": side,
    }).json()


def test_give_time_message_hold_ms_defaults_and_bounds():
    import pydantic
    from chessshootout.server.protocol import GiveTimeMessage, GIVE_TIME_MAX_HOLD_MS
    assert GiveTimeMessage().hold_ms == 0
    parsed = GiveTimeMessage.model_validate_json('{"type":"give_time","hold_ms":300}')
    assert parsed.hold_ms == 300
    with pytest.raises(pydantic.ValidationError):
        GiveTimeMessage(hold_ms=-1)
    with pytest.raises(pydantic.ValidationError):
        GiveTimeMessage(hold_ms=GIVE_TIME_MAX_HOLD_MS + 1)


def test_server_broadcasts_time_granted_to_both_peers(client):
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white",
                   time=5, inc=0)
    b = _matchmake(client, uuid=BOB, nickname="Bob", side="black",
                   time=5, inc=0)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            room = client.app.state.rooms.get(a["room_id"])
            room.backend.clock.white_remaining = 200.0
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "give_time"}))
            granted_w = json.loads(ws_w.receive_text())
            granted_b = json.loads(ws_b.receive_text())
            assert granted_w["type"] == "time_granted"
            assert granted_w["granted_by"] == "black"
            assert granted_w["seconds_added"] == pytest.approx(15.0)
            assert granted_w["clock"]["white_remaining"] == pytest.approx(215.0, abs=1.0)
            assert granted_b == granted_w


def test_server_hold_ms_grants_proportional_ticks(client):
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white",
                   time=5, inc=0)
    b = _matchmake(client, uuid=BOB, nickname="Bob", side="black",
                   time=5, inc=0)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            room = client.app.state.rooms.get(a["room_id"])
            room.backend.clock.white_remaining = 100.0
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "give_time", "hold_ms": 300}))
            granted = json.loads(ws_w.receive_text())
            ws_b.receive_text()
            assert granted["seconds_added"] == pytest.approx(45.0)
            assert granted["clock"]["white_remaining"] == pytest.approx(145.0, abs=1.0)


def test_server_caps_at_initial_seconds_and_still_broadcasts(client):
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white",
                   time=5, inc=0)
    b = _matchmake(client, uuid=BOB, nickname="Bob", side="black",
                   time=5, inc=0)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "give_time"}))
            granted = json.loads(ws_w.receive_text())
            ws_b.receive_text()
            assert granted["type"] == "time_granted"
            assert granted["seconds_added"] == pytest.approx(0.0)
            assert granted["granted_by"] == "black"
