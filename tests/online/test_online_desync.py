"""Desync detection + /resume recovery.

Invariant under test: a client detects a dropped/out-of-order broadcast by
comparing the server's `ply` field against `len(move_history)`, fires
request_state_sync, and gates every further move_applied/takeback behind the
`_resyncing` flag until _handle_game_resumed clears it.
"""
import ast
import json
import logging
import os
import random

from unittest.mock import MagicMock

import pygame as pg
import pytest
from fastapi.testclient import TestClient

import chessshootout
from tests.conftest import pygame_display
from chessshootout.domain.match import ONLINE
from chessshootout.frontend.online_coordinator import ResyncCause
from chessshootout.backend.pieces import PieceColor
from chessshootout.backend.utils import square_from_coord
from chessshootout.server import connections as connections_module
from chessshootout.server.app import PROTOCOL_VERSION, create_app
from chessshootout.server.handlers import (
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS,
)
from chessshootout.server.protocol import MoveAppliedMessage
from tests.helpers import FakeClock, fake_uuid4, read_source_without_docstrings


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)


_pygame_init = pygame_display(1000, 800)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def app(clock):
    return create_app(now_provider=clock, max_rooms=8)


@pytest.fixture
def client(app):
    return TestClient(app)


def _matchmake(client, *, uuid, nickname, side):
    return client.post("/matchmake", json={
        "version": PROTOCOL_VERSION,
        "client_uuid": uuid, "nickname": nickname,
        "time_minutes": 5, "increment_seconds": 0,
        "side_preference": side,
    }).json()


def _auth(token):
    return {"version": PROTOCOL_VERSION, "type": "auth", "session_token": token}


def _move(from_sq, to_sq):
    return {"version": PROTOCOL_VERSION, "type": "move",
            "from": from_sq, "to": to_sq}


def _paired_ws(client):
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white")
    b = _matchmake(client, uuid=BOB, nickname="Bob", side="black")
    return a, b


def _recv_type(ws, msg_type):
    """Next frame of the given type. Protocol v5 force-pushes idle_window after
    plies 1 and 2, so positional reads are no longer stable."""
    while True:
        msg = json.loads(ws.receive_text())
        if msg["type"] == msg_type:
            return msg


def test_move_applied_includes_ply(client):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps(_move("e2", "e4")))
            applied_w = _recv_type(ws_w, "move_applied")
            applied_b = _recv_type(ws_b, "move_applied")
            assert applied_w["ply"] == 1
            assert applied_b["ply"] == 1
            ws_b.send_text(json.dumps(_move("e7", "e5")))
            applied_w2 = _recv_type(ws_w, "move_applied")
            applied_b2 = _recv_type(ws_b, "move_applied")
            assert applied_w2["ply"] == 2
            assert applied_b2["ply"] == 2


def test_takeback_applied_includes_ply(client):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps(_move("e2", "e4")))
            _recv_type(ws_w, "move_applied")
            _recv_type(ws_b, "move_applied")
            ws_b.send_text(json.dumps(_move("e7", "e5")))
            _recv_type(ws_w, "move_applied")
            _recv_type(ws_b, "move_applied")
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "takeback_request"}))
            _recv_type(ws_w, "takeback_offered")
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "takeback_response",
                                       "accept": True}))
            tb_w = _recv_type(ws_w, "takeback_applied")
            tb_b = _recv_type(ws_b, "takeback_applied")
            assert tb_w["ply"] == 1
            assert tb_b["ply"] == 1


def test_dropped_broadcast_pushes_reconnecting_to_surviving_peer(client, monkeypatch):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()

            real_send = connections_module.send
            dropped = {"once": False}

            async def flaky_send(ws, message):
                if (isinstance(message, MoveAppliedMessage)
                        and not dropped["once"]):
                    dropped["once"] = True
                    return False
                return await real_send(ws, message)

            monkeypatch.setattr(connections_module, "send", flaky_send)

            ws_w.send_text(json.dumps(_move("e2", "e4")))
            seen_types = set()
            for _ in range(2):
                msg = json.loads(ws_b.receive_text())
                seen_types.add(msg["type"])
            assert "connection_status" in seen_types


def _ping(ply):
    return {"version": PROTOCOL_VERSION, "type": "ping", "ply": ply}


def _past_the_transit_grace(clock):
    """Pairing runs the real broadcast_game_start, which stamps a history change
    at the fake clock's frozen zero -- and a stamp with no previous length
    excuses EVERY reported ply while it is fresh. Every test that wants a
    heartbeat judged has to step past that window first."""
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)


def test_ping_with_matching_ply_pongs_without_directive(client, clock):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)
            ws_w.send_text(json.dumps(_ping(0)))
            msg = json.loads(ws_w.receive_text())
            assert msg["type"] == "pong"


def test_ping_with_wrong_ply_directs_resync_and_flags_opponent(client, clock):
    """A directive costs the client a full /resume, so it takes a mismatch that
    survives RESYNC_STABLE_MISMATCH_HEARTBEATS judged heartbeats -- the first
    one only earns a pong."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)
            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS - 1):
                ws_w.send_text(json.dumps(_ping(7)))
                assert json.loads(ws_w.receive_text())["type"] == "pong"
            ws_w.send_text(json.dumps(_ping(7)))
            got = {json.loads(ws_w.receive_text())["type"] for _ in range(2)}
            assert got == {"pong", "resync_directive"}
            status = json.loads(ws_b.receive_text())
            assert status["type"] == "connection_status"
            assert status["opp_state"] == "resyncing"


def test_a_heartbeat_racing_the_opponents_move_is_not_a_desync(client, clock):
    """The bug this whole commit is about, end to end over the wire: white's
    move lands on the server and black's heartbeat -- already in the air --
    still reports the position before it. Black's board is fine; the broadcast
    simply has not arrived yet."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)
            ws_w.send_text(json.dumps(_move("e2", "e4")))
            _recv_type(ws_w, "move_applied")
            _recv_type(ws_b, "move_applied")

            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS + 2):
                ws_b.send_text(json.dumps(_ping(0)))
            ws_w.send_text(json.dumps(_quick_chat()))

            seen = [m["type"] for m in _drain_until_chat(ws_b)]
            assert seen.count("pong") == RESYNC_STABLE_MISMATCH_HEARTBEATS + 2
            assert "resync_directive" not in seen


def test_a_heartbeat_racing_a_takeback_is_not_a_desync(client, clock):
    """The mirror case: after an accepted takeback both clients are briefly one
    AHEAD of the server, which is the opposite sign to a move and would fail a
    one-directional tolerance."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)
            ws_w.send_text(json.dumps(_move("e2", "e4")))
            _recv_type(ws_w, "move_applied")
            _recv_type(ws_b, "move_applied")
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "takeback_request"}))
            _recv_type(ws_b, "takeback_offered")
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "takeback_response",
                                       "accept": True}))
            _recv_type(ws_w, "takeback_applied")
            _recv_type(ws_b, "takeback_applied")

            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS + 2):
                ws_w.send_text(json.dumps(_ping(1)))
            ws_b.send_text(json.dumps(_quick_chat()))

            seen = [m["type"] for m in _drain_until_chat(ws_w)]
            assert seen.count("pong") == RESYNC_STABLE_MISMATCH_HEARTBEATS + 2
            assert "resync_directive" not in seen


def _quick_chat():
    return {"version": PROTOCOL_VERSION, "type": "quick_chat", "preset": 0}


def _drain_until_chat(ws):
    """Sentinel drain: quick_chat is relayed straight through with no state of
    its own, so the frame after it is a hard end-of-stream marker -- everything
    the flapping pings produced has to arrive ahead of it."""
    seen = []
    while True:
        msg = json.loads(ws.receive_text())
        if msg["type"] == "quick_chat_received":
            return seen
        seen.append(msg)


def test_flapping_ping_notifies_the_opponent_once_per_window(client, clock):
    """Griefing pin: `ply` is client-supplied, so a hostile client can alternate
    correct/incorrect plies at the websocket rate limit (30/s) and toggle the
    opponent's resync state 15 times a second. Every `resyncing` transition
    toasts the opponent, and toasts dedupe by key, so the opponent ends up with
    a permanently stuck banner plus UI state flapping at 15 Hz.

    The server debounces the NOTIFY direction only: the opponent is told
    `resyncing` at most once per RESYNC_NOTIFY_MIN_INTERVAL_SECONDS per
    connection. Clears stay ungated on purpose -- `connected` raises no toast,
    and delaying it would leave a stale "opponent is resyncing" banner up after
    a genuine recovery, which is the honest-path cost this fix must not pay."""
    from chessshootout.server.handlers import RESYNC_NOTIFY_MIN_INTERVAL_SECONDS

    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)

            for _ in range(4):
                for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
                    ws_w.send_text(json.dumps(_ping(7)))
                ws_w.send_text(json.dumps(_ping(0)))
            ws_w.send_text(json.dumps(_quick_chat()))

            states = [m["opp_state"] for m in _drain_until_chat(ws_b)
                      if m["type"] == "connection_status"]
            assert states == ["resyncing", "connected"]

            clock.advance(RESYNC_NOTIFY_MIN_INTERVAL_SECONDS + 0.1)
            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
                ws_w.send_text(json.dumps(_ping(7)))
            status = json.loads(ws_b.receive_text())
            assert status["type"] == "connection_status"
            assert status["opp_state"] == "resyncing", (
                "a fresh window must notify again -- this is a debounce, not a latch")


def test_a_pure_ply_alternation_costs_the_opponent_nothing(client, clock):
    """The cheapest form of the griefing pattern is now free to ignore: the
    strike counter sits in FRONT of the debounce, so a wrong ply followed by a
    right one is a first strike immediately wiped. Nothing reaches the notify
    gate, and nothing reaches the client either."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)

            for i in range(8):
                ws_w.send_text(json.dumps(_ping(7 if i % 2 == 0 else 0)))
            ws_w.send_text(json.dumps(_quick_chat()))

            assert _drain_until_chat(ws_b) == []
            ws_b.send_text(json.dumps(_quick_chat()))
            assert [m["type"] for m in _drain_until_chat(ws_w)] == ["pong"] * 8


def test_flapping_ping_cannot_amplify_resync_directives(client, clock):
    """The directive rides back to the sender, and each one drives a /resume, so
    an unbounded 30/s stream is self-inflicted amplification against the server's
    own state-rebuild path. One per RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS is
    enough: honest clients ping on the 2 s heartbeat, so their recovery cadence
    is untouched."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)

            for _ in range(8):
                ws_w.send_text(json.dumps(_ping(7)))

            # Read the exact frame count the debounce must produce -- eight
            # pongs and one directive -- straight off the pinging socket, where
            # per-connection ordering makes it deterministic. The opponent's
            # chat cannot serve as the end marker here: it is sent on the OTHER
            # socket, so on a loaded runner its relay can overtake the tail of
            # this socket's pongs. It still proves nothing FURTHER is queued.
            kinds = [json.loads(ws_w.receive_text())["type"] for _ in range(9)]
            assert kinds.count("pong") == 8
            assert kinds.count("resync_directive") == 1

            ws_b.send_text(json.dumps(_quick_chat()))
            assert _drain_until_chat(ws_w) == [], "the window produced nothing else"


def test_sustained_desync_keeps_directing_resync_promptly(client, clock):
    """The debounce must not slow a real recovery: a client that stays behind
    earns its next directive as soon as it has mismatched afresh, and the
    directive interval is shorter than the heartbeat, so what holds the second
    order back is only the strikes -- never the gate.

    Sending an order spends the strikes that bought it, so the heartbeat right
    after one gets a pong and nothing else. That is the point: a client that
    never answers must not be handed a fresh /resume order every couple of
    seconds for the rest of the game."""
    from chessshootout.server.handlers import RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS
    from chessshootout.server.protocol import HEARTBEAT_INTERVAL_SECONDS

    assert RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS < HEARTBEAT_INTERVAL_SECONDS

    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            _past_the_transit_grace(clock)

            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS - 1):
                ws_w.send_text(json.dumps(_ping(7)))
                assert json.loads(ws_w.receive_text())["type"] == "pong"
            ws_w.send_text(json.dumps(_ping(7)))
            first = {json.loads(ws_w.receive_text())["type"] for _ in range(2)}
            assert first == {"pong", "resync_directive"}

            for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS - 1):
                clock.advance(HEARTBEAT_INTERVAL_SECONDS)
                ws_w.send_text(json.dumps(_ping(7)))
                assert json.loads(ws_w.receive_text())["type"] == "pong", \
                    "the ignored order is not simply repeated on the next beat"
            clock.advance(HEARTBEAT_INTERVAL_SECONDS)
            ws_w.send_text(json.dumps(_ping(7)))
            second = {json.loads(ws_w.receive_text())["type"] for _ in range(2)}
            assert second == {"pong", "resync_directive"}, \
                "but a client still behind after a fresh pair is directed again"


def test_reconnecting_client_gets_opponent_present_snapshot(client):
    """On reconnect the server tells the returning client its opponent's real
    presence, so a client that dropped can't be stuck showing the opponent red."""
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
        ws_b.send_text(json.dumps(_auth(b["session_token"])))
        with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
            ws_w.send_text(json.dumps(_auth(a["session_token"])))
            ws_w.receive_text()                       # game_start
            ws_b.receive_text()                       # game_start
        # white dropped -> black is told "reconnecting"; drain it
        assert json.loads(ws_b.receive_text())["type"] == "connection_status"
        with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w2:
            ws_w2.send_text(json.dumps(_auth(a["session_token"])))
            snap = json.loads(ws_w2.receive_text())
            assert snap["type"] == "connection_status"
            assert snap["opp_state"] == "connected", "opponent b is still here"


def test_reconnecting_client_told_opponent_still_gone(client):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()                       # game started, both present
        # black dropped inside; now white drops too
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w2:
        ws_w2.send_text(json.dumps(_auth(a["session_token"])))
        snap = json.loads(ws_w2.receive_text())
        assert snap["type"] == "connection_status"
        assert snap["opp_state"] == "reconnecting", "opponent is still gone"


def test_new_matchmake_abandons_in_progress_game(client):
    a, b = _paired_ws(client)
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(_auth(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(_auth(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps(_move("e2", "e4")))
            ws_w.receive_text()
            ws_b.receive_text()
            resp = _matchmake(client, uuid=ALICE, nickname="Alice", side="white")
            assert "room_id" in resp
            results = []
            for _ in range(3):
                msg = json.loads(ws_b.receive_text())
                if msg["type"] == "result":
                    results.append(msg)
                    break
            assert results and results[0]["reason"] == "abandonment"
            assert results[0]["winner_color"] == "black"


def _online_app():
    from chessshootout.frontend.frontend import Frontend
    app = Frontend(1000, 800)
    app.sound_manager = MagicMock()
    app.coordinator.client = MagicMock()
    app.coordinator.client.get_ping_ms.return_value = None
    app.coordinator.client.is_server_silent.return_value = False
    app.coordinator.client.heartbeat_interval.return_value = 2.0
    app.coordinator.subscribe(app.game)
    app.screen = app.game
    app.game.variant = "online"
    app.game.white_name = "Alice"
    app.game.black_name = "Bob"
    app.game.match.mode = ONLINE
    app.game.match.local_color = PieceColor.WHITE
    return app


def test_remote_move_with_correct_ply_applies():
    app = _online_app()
    payload = {"from": "e2", "to": "e4", "san": "e4", "ply": 1,
               "clock": {}}
    app.coordinator._handle_remote_move_applied(payload)
    assert app.coordinator._resyncing is False
    app.coordinator.client.request_state_sync.assert_not_called()
    assert len(app.game.match.move_history) == 1


def test_remote_move_with_skipped_ply_triggers_resync():
    """Empty local history (ply 0) vs server ply 3 means plies 1-2 were missed."""
    app = _online_app()
    payload = {"from": "e7", "to": "e5", "san": "e5", "ply": 3,
               "clock": {}}
    app.coordinator._handle_remote_move_applied(payload)
    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()
    assert len(app.game.match.move_history) == 0


def test_remote_move_with_illegal_payload_triggers_resync():
    """Ply matches but from/to is illegal (no piece on e3); apply returns legal=False."""
    app = _online_app()
    payload = {"from": "e3", "to": "e4", "san": "e4", "ply": 1,
               "clock": {}}
    app.coordinator._handle_remote_move_applied(payload)
    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()


def test_resync_gate_drops_subsequent_move_applied():
    """A held gate drops the move without applying it or firing a new request."""
    app = _online_app()
    app.coordinator._resyncing = True
    payload = {"from": "e2", "to": "e4", "san": "e4", "ply": 1,
               "clock": {}}
    app.coordinator._handle_remote_move_applied(payload)
    assert len(app.game.match.move_history) == 0
    app.coordinator.client.request_state_sync.assert_not_called()


def test_takeback_applied_with_skipped_ply_triggers_resync():
    """Local has 1 ply but server's post-takeback ply 5 is impossible without misses."""
    from chessshootout.backend.utils import Square
    app = _online_app()
    app.game.match.try_move(Square(6, 4), Square(4, 4))
    payload = {"clock": {}, "fen": "", "ply": 5}
    app.coordinator._handle_takeback_applied(payload)
    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()


def test_game_resumed_clears_resync_gate():
    app = _online_app()
    app.game._time_control = (300, 0)
    app.game.match.setup_clock(300, 0)
    app.coordinator._resyncing = True
    payload = {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "move_history": [{"san": "e4"}, {"san": "e5"}],
        "clock": {"white_remaining": 300.0, "black_remaining": 300.0,
                  "running_for": None},
    }
    app.coordinator._handle_game_resumed(payload)
    assert app.coordinator._resyncing is False
    assert len(app.game.match.move_history) == 2


def test_begin_resync_is_idempotent_during_inflight_request():
    """The in-flight flag suppresses duplicate requests across repeated calls."""
    app = _online_app()
    app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)
    app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)
    app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)
    assert app.coordinator._resyncing is True
    assert app.coordinator.client.request_state_sync.call_count == 1


def test_resync_directive_triggers_resync():
    """A server resync directive (sent when a heartbeat shows the client is behind)
    drives the standard /resume recovery."""
    from chessshootout.online.client import Event
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._handle_online_event(Event("resync_directive", {}))
    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()


def test_resyncing_shows_toast_each_frame():
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._resyncing = True
    app.coordinator._resync_started_at_ms = pg.time.get_ticks()
    app.coordinator._update_online_phase()
    assert app.toast.message == "Resyncing…"


def test_resyncing_self_heals_after_timeout():
    from chessshootout.frontend.online_coordinator import RESYNC_TIMEOUT_MS
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._resyncing = True
    app.coordinator._resync_started_at_ms = pg.time.get_ticks() - (RESYNC_TIMEOUT_MS + 1000)
    app.coordinator._update_online_phase()
    assert app.coordinator._resyncing is False


def test_resync_timeout_escalates_to_reconnect():
    """A resync that never lands escalates to a full reconnect, so the opponent's abandon
    countdown starts and recovery runs through the standard reconnect path."""
    from chessshootout.frontend.online_coordinator import RESYNC_TIMEOUT_MS
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._resyncing = True
    app.coordinator._resync_started_at_ms = pg.time.get_ticks() - (RESYNC_TIMEOUT_MS + 1000)
    app.coordinator._update_online_phase()
    assert app.coordinator._resyncing is False
    app.coordinator.client.force_reconnect.assert_called_once()


def test_resync_timeout_escalation_logs_a_warning_not_an_info(caplog):
    """Escalating a stuck resync to a full reconnect is a degraded path, not a
    routine state transition — it must not read as just another INFO line."""
    from chessshootout.frontend.online_coordinator import RESYNC_TIMEOUT_MS
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._resyncing = True
    app.coordinator._resync_started_at_ms = pg.time.get_ticks() - (RESYNC_TIMEOUT_MS + 1000)
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        app.coordinator._update_online_phase()
    lines = [r for r in caplog.records if "resync timed out" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING


def test_resync_timeout_no_escalation_when_already_reconnecting():
    from chessshootout.frontend.online_coordinator import RESYNC_TIMEOUT_MS
    app = _online_app()
    app.coordinator.client.state = "reconnecting"
    app.coordinator._resyncing = True
    app.coordinator._resync_started_at_ms = pg.time.get_ticks() - (RESYNC_TIMEOUT_MS + 1000)
    app.coordinator._update_online_phase()
    assert app.coordinator._resyncing is False
    app.coordinator.client.force_reconnect.assert_not_called()


def test_online_error_room_lost_clears_resyncing():
    app = _online_app()
    app.coordinator._resyncing = True
    app.coordinator._handle_online_error({"reason": "room_lost"})
    assert app.coordinator._resyncing is False


def test_opponent_resyncing_status_shows_toast():
    """The server drives the opponent-resyncing indication via connection_status."""
    app = _online_app()
    app.coordinator._handle_connection_status({"opp_state": "resyncing"})
    assert app.toast.message == "Opponent is resyncing…"
    assert app.game._opp_disconnected_at_ms is None


def test_heartbeat_sent_when_interval_elapsed():
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._last_heartbeat_sent_ms = pg.time.get_ticks() - 5000
    app.coordinator._send_heartbeat_if_due()
    app.coordinator.client.send_ping.assert_called_once_with(len(app.game.match.move_history))


def test_heartbeat_not_sent_before_interval():
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._last_heartbeat_sent_ms = pg.time.get_ticks()
    app.coordinator._send_heartbeat_if_due()
    app.coordinator.client.send_ping.assert_not_called()


def test_server_silence_escalates_to_reconnect():
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator.client.is_server_silent.return_value = True
    app.coordinator._send_heartbeat_if_due()
    app.coordinator.client.force_reconnect.assert_called_once()
    app.coordinator.client.send_ping.assert_not_called()


def test_server_silence_logs_a_warning_not_an_info(caplog):
    """A missed heartbeat is a degraded connection, not routine chatter."""
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator.client.is_server_silent.return_value = True
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        app.coordinator._send_heartbeat_if_due()
    lines = [r for r in caplog.records if "heartbeat silent" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING


def test_resync_gate_does_not_outlive_the_session_it_belongs_to():
    """_resyncing gates give-time holds and result promotion and drives a
    per-frame "Resyncing…" toast. Dropping the client without clearing it left a
    phantom toast on the start menu and, if the next match paired inside the 8 s
    window, silently gated the new game."""
    for drop in ("_tear_down_online_session", "_on_online_cancel"):
        app = _online_app()
        app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)
        assert app.coordinator._resyncing is True

        getattr(app.coordinator, drop)()

        assert app.coordinator._resyncing is False, drop


def test_resync_gate_is_cleared_when_the_socket_is_kept_for_a_rematch():
    app = _online_app()
    app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)

    app.coordinator.retain_for_rematch(True)

    assert app.coordinator._resyncing is False


def test_resume_with_no_active_online_game_is_dropped_and_clears_the_gate():
    """The post-game rematch window keeps the socket alive after the user is back
    on the menu (variant flipped to "local"). A late /resume reply there has no
    live game to rebuild — it must not replay moves into the inactive screen."""
    app = _online_app()
    app.game.variant = "local"
    app.coordinator._begin_resync(ResyncCause.SERVER_DIRECTIVE)

    app.coordinator._handle_game_resumed({
        "fen": "",
        "move_history": [{"san": "e4"}, {"san": "e5"}],
        "clock": {},
    })

    assert app.game.match.move_history == []
    assert app.coordinator._resyncing is False


BLOCKED_ARROW = {"from": "e2", "to": "e4"}
BLOCKED_MARKS = {(square_from_coord("e2"), square_from_coord("e4")),
                 square_from_coord("c3")}


def _blocked_payload():
    return {"action": "blocked", "arrows": [BLOCKED_ARROW],
            "highlights": ["c3"], "share_muted": False}


def _resumable_app():
    app = _online_app()
    app.game._time_control = (300, 0)
    app.game.match.setup_clock(300, 0)
    return app


def _resumed_payload(**extra):
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_history": [],
        "clock": {"white_remaining": 300.0, "black_remaining": 300.0,
                  "running_for": None},
    }
    payload.update(extra)
    return payload


def test_marks_state_is_dropped_not_buffered_while_resyncing():
    """Replaying annotation state/deltas on top of the /resume snapshot is the
    unsafe option: a mark the opponent deleted before the snapshot comes back
    from a buffered `add` and sticks until they toggle it again. Every resync
    exit is followed by an authoritative snapshot (the timeout escalates to a
    reconnect, which re-emits game_resumed), so marks state is dropped outright
    rather than queued."""
    app = _online_app()
    app.coordinator._resyncing = True

    app.coordinator._handle_annotation_delta(
        {"action": "add", "kind": "arrow", "from": "e2", "to": "e4"})
    app.coordinator._handle_annotations_state(
        {"sharing": True, "highlights": ["c3"], "arrows": [BLOCKED_ARROW]})

    assert app.coordinator._resync_buffer == []
    assert app.game.board.annotations.opp_arrows == []
    assert app.game.board.annotations.opp_highlighted_squares == set()


def test_a_blocked_notification_survives_a_cancelled_resync():
    """`annotations_blocked` is the one payload a /resume cannot rebuild: it is a
    one-shot notification (toast + flag_own), so it keeps buffering while the
    gate is up and replays on every exit -- including the timeout cancel. The
    toast assertion is the end-to-end claim: in the real timeout-cancel path a
    force_reconnect /resume follows and _restore_resumed_annotations hard-resets
    annotations.flagged, so only the toast truly survives. The flagged assertion
    is unit-local to this test, which has no follow-up resume."""
    app = _online_app()
    app.coordinator._resyncing = True
    app.coordinator._handle_annotations_blocked(_blocked_payload())
    assert app.game.board.annotations.flagged == set()

    app.coordinator._end_resync()

    assert app.toast.message == "Some marks can't be shared"
    assert app.game.board.annotations.flagged == BLOCKED_MARKS


def test_a_blocked_notification_replays_after_a_resume():
    """The replay has to run AFTER on_resume, not before: on_resume wipes
    annotations.flagged, so a blocked notification replayed ahead of it would be
    erased by the very snapshot that cannot carry it."""
    app = _resumable_app()
    app.coordinator._resyncing = True
    app.coordinator._handle_annotations_blocked(_blocked_payload())

    app.coordinator._handle_game_resumed(_resumed_payload())

    assert app.coordinator._resyncing is False
    assert app.coordinator._resync_buffer == []
    assert app.game.board.annotations.flagged == BLOCKED_MARKS


def test_a_cancelled_resync_with_no_subscriber_replays_nothing(caplog):
    """Teardown paths (_drop_client, retain_for_rematch, room_lost) cancel the
    resync once the GameScreen has already unsubscribed. Replaying there would
    toast onto the start menu and trip _forward_board_event's "no subscriber"
    log.error, so the buffer is dropped instead."""
    app = _online_app()
    app.coordinator._resyncing = True
    app.coordinator._handle_annotations_blocked(_blocked_payload())
    app.coordinator.unsubscribe(app.game)

    with caplog.at_level(logging.ERROR, logger="chess.frontend"):
        app.coordinator._end_resync()

    assert app.coordinator._resync_buffer == []
    assert app.toast.message is None
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_a_teardown_exit_drops_the_buffer_even_with_a_live_subscriber():
    """room_lost and the hard-failure confirms fire while the GameScreen is
    still subscribed, and the abandon path drops the client before the menu
    switch -- the no-subscriber guard misses all three orderings. Those exits
    pass replay=False: the game is over, so a late blocked notification must
    not toast over the end-of-game modal."""
    app = _online_app()
    app.coordinator._resyncing = True
    app.coordinator._handle_annotations_blocked(_blocked_payload())

    app.coordinator._end_resync(replay=False)

    assert app.coordinator._resync_buffer == []
    assert app.toast.message is None
    assert app.game.board.annotations.flagged == set()


def test_a_resume_snapshot_is_never_overwritten_by_stale_deltas():
    """The resurrection regression: the opponent draws an arrow and deletes it,
    the server's snapshot omits it, and the mid-resync `add` used to be replayed
    on top of the fresh board -- putting a mark back that the server had already
    settled as gone."""
    app = _resumable_app()
    arrow = (square_from_coord("e2"), square_from_coord("e4"))
    app.game.board.annotations.set_opp(set(), [arrow])
    app.coordinator._resyncing = True

    app.coordinator._handle_annotation_delta(
        {"action": "add", "kind": "arrow", "from": "e2", "to": "e4"})
    app.coordinator._handle_game_resumed(_resumed_payload(
        black_annotations={"sharing": True, "highlights": [], "arrows": []}))

    assert app.game.board.annotations.opp_arrows == []


# ---------------------------------------------------------------------------
# The directive the client is allowed to ignore, and the causes it must name.
# ---------------------------------------------------------------------------


def _directive(**payload):
    from chessshootout.online.client import Event
    return Event("resync_directive", payload)


def test_a_directive_the_client_has_already_outrun_is_dropped():
    """The directive rides back on the same socket a move broadcast does, so it
    can arrive AFTER the update it was written about. Rebuilding the whole game
    at that point is a pointless "Resyncing..." toast in a healthy game: the
    client is already on the ply the server was ordering it to reach."""
    app = _online_app()
    app.coordinator.client.state = "connected"

    app.coordinator._handle_online_event(_directive(server_ply=0))

    assert app.coordinator._resyncing is False
    app.coordinator.client.request_state_sync.assert_not_called()


def test_a_directive_about_a_different_ply_is_obeyed():
    app = _online_app()
    app.coordinator.client.state = "connected"

    app.coordinator._handle_online_event(_directive(server_ply=4))

    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing_server_ply"),
        pytest.param({"server_ply": None}, id="null_server_ply"),
        pytest.param({"server_ply": "0"}, id="string_server_ply"),
        pytest.param({"server_ply": 0.0}, id="float_server_ply"),
    ],
)
def test_an_unreadable_directive_is_still_obeyed(payload):
    """The drop is an optimisation, never a way to ignore the server. Anything
    the client cannot read as a ply must fall through to the resync rather than
    quietly comparing None against None and doing nothing."""
    app = _online_app()
    app.coordinator.client.state = "connected"

    app.coordinator._handle_online_event(_directive(**payload))

    assert app.coordinator._resyncing is True


def test_an_offboard_client_obeys_a_directive_about_ply_zero():
    """The nastiest false equality the drop could produce: a client that is off
    the board reports no ply, and a fresh game sits at ply 0. Comparing those as
    equal would leave a genuinely stranded client refusing every order it got."""
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.game.variant = "local"

    app.coordinator._handle_online_event(_directive(server_ply=0))

    assert app.coordinator._resyncing is True


def test_a_repeated_directive_while_resyncing_changes_nothing(caplog):
    """The server directive debounce is one per second, so a client whose
    /resume is slow will be ordered again mid-repair. That must not restart the
    self-heal timer -- doing so would let a wedged resync spin forever -- and it
    must go by in total silence: a repair already running is not news, and a
    stuck client would otherwise write a WARNING every second it stayed
    stuck."""
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._handle_online_event(_directive(server_ply=4))
    started_at = app.coordinator._resync_started_at_ms

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="chess.frontend"):
        app.coordinator._handle_online_event(_directive(server_ply=4))

    assert app.coordinator._resync_started_at_ms == started_at
    assert app.coordinator.client.request_state_sync.call_count == 1
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def _cause_server_directive():
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._handle_online_event(_directive(server_ply=4))
    return app


def _cause_move_rejected():
    from chessshootout.server.protocol import Reason
    app = _online_app()
    app.coordinator._handle_online_error(
        {"reason": Reason.INVALID_MOVE_FORMAT, "msg_type": "move"})
    return app


def _cause_move_ply_gap():
    app = _online_app()
    app.coordinator._handle_remote_move_applied(
        {"from": "e7", "to": "e5", "san": "e5", "ply": 3, "clock": {}})
    return app


def _cause_move_illegal():
    app = _online_app()
    app.coordinator._handle_remote_move_applied(
        {"from": "e3", "to": "e4", "san": "e4", "ply": 1, "clock": {}})
    return app


def _cause_takeback_ply_gap():
    from chessshootout.backend.utils import Square
    app = _online_app()
    app.game.match.try_move(Square(6, 4), Square(4, 4))
    app.coordinator._handle_takeback_applied({"clock": {}, "fen": "", "ply": 5})
    return app


def _cause_verdict_lost():
    from chessshootout.frontend.online_coordinator import SKILLCHECK_WATCHDOG_SLACK_MS
    from chessshootout.skillcheck.wheel import SKILLCHECK_DEADLINE_MS
    from tests.online.test_online_skillcheck_client import (
        _capture_board, _online_app as _skillcheck_app, _required_payload,
    )
    app = _skillcheck_app()
    app.screen = app.game
    frm, to = _capture_board(app)
    app.game.skillcheck_session.skillcheck_gate(frm, to)
    app.coordinator._handle_skill_check_required(_required_payload(frm, to))
    app.game.skillcheck_session.online_skillcheck_opened_ms = (
        pg.time.get_ticks() - SKILLCHECK_DEADLINE_MS - SKILLCHECK_WATCHDOG_SLACK_MS - 100)
    app.coordinator._tick_skillcheck_watchdog()
    return app


def _cause_result_apply_failed():
    app = _online_app()

    def _explode():
        raise RuntimeError("the verdict could not be played out")

    app.game.skillcheck_session.online_verdict_action = _explode
    app.game.on_result({"reason": "resignation", "winner_color": "black"})
    return app


CAUSE_DRIVERS = {
    ResyncCause.SERVER_DIRECTIVE: _cause_server_directive,
    ResyncCause.MOVE_REJECTED: _cause_move_rejected,
    ResyncCause.MOVE_PLY_GAP: _cause_move_ply_gap,
    ResyncCause.MOVE_ILLEGAL: _cause_move_illegal,
    ResyncCause.TAKEBACK_PLY_GAP: _cause_takeback_ply_gap,
    ResyncCause.VERDICT_LOST: _cause_verdict_lost,
    ResyncCause.RESULT_APPLY_FAILED: _cause_result_apply_failed,
}


@pytest.mark.parametrize("cause", sorted(CAUSE_DRIVERS))
def test_every_resync_entry_point_names_itself_once(cause, caplog):
    """A resync is only visible to us through the crash log, and "it started
    resyncing" on its own is a support ticket with no next step. Every entry
    point writes exactly one line naming which one it was, after the
    idempotency guard so a repeat cannot double it."""
    with caplog.at_level(logging.DEBUG, logger="chess.frontend"):
        app = CAUSE_DRIVERS[cause]()

    assert app.coordinator._resyncing is True
    lines = [r for r in caplog.records if r.getMessage().startswith("resync begin")]
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING
    assert lines[0].getMessage().startswith(f"resync begin cause={cause} client_ply=")


def test_the_cause_vocabulary_is_closed_and_fully_exercised():
    assert set(CAUSE_DRIVERS) == _resync_cause_values()


def _resync_cause_values():
    return {v for k, v in vars(ResyncCause).items()
            if not k.startswith("_") and isinstance(v, str)}


def _begin_resync_call_causes(path):
    """Every argument passed to a _begin_resync call in one file, as the
    ResyncCause attribute name it names."""
    tree = ast.parse(read_source_without_docstrings(path), filename=path)
    causes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "_begin_resync":
            continue
        assert len(node.args) == 1, f"{path}:{node.lineno}: _begin_resync takes one cause"
        arg = node.args[0]
        assert (isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == "ResyncCause"), \
            f"{path}:{node.lineno}: the cause must be a ResyncCause member"
        causes.append(arg.attr)
    return causes


def test_every_resync_call_site_names_a_real_cause():
    """The enum is only worth having if nothing bypasses it: a bare
    _begin_resync() or a hand-typed string would put an unsearchable line in the
    crash log. The set has to be exhausted too -- a member nobody passes is a
    cause we thought about and never wired."""
    package_root = os.path.dirname(os.path.abspath(chessshootout.__file__))
    frontend_root = os.path.join(package_root, "frontend")
    assert os.path.isdir(frontend_root), f"guard root is wrong: {frontend_root}"
    used, scanned = [], 0
    for dirpath, _, filenames in os.walk(frontend_root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            scanned += 1
            used.extend(_begin_resync_call_causes(os.path.join(dirpath, name)))
    assert scanned >= 40, f"only scanned {scanned} files, guard root is likely wrong"
    assert len(used) == 8, f"expected the eight known call sites, found {len(used)}"
    assert {getattr(ResyncCause, name) for name in used} == _resync_cause_values()


# ---------------------------------------------------------------------------
# Which ply the heartbeat is allowed to claim.
# ---------------------------------------------------------------------------


def test_a_live_online_board_reports_its_own_ply():
    from chessshootout.backend.utils import Square
    app = _online_app()
    app.game.match.try_move(Square(6, 4), Square(4, 4))

    assert app.coordinator._heartbeat_ply() == 1


def test_a_client_on_another_screen_claims_no_ply():
    """The heartbeat keeps running from the menu, the history view and the
    review screen. Reporting the game screen's ply from there is a claim about
    a board nobody is looking at."""
    app = _online_app()
    app.screen = app.menu

    assert app.coordinator._heartbeat_ply() is None


def test_a_local_game_claims_no_ply():
    app = _online_app()
    app.game.variant = "local"

    assert app.coordinator._heartbeat_ply() is None


def test_a_client_still_behind_the_match_found_card_claims_no_ply():
    """The pairing that broke this worst: the socket is up and the heartbeat is
    already running while the match-found card counts down, and the board behind
    it is still the PREVIOUS game -- a mismatch of up to a whole game, every
    single pairing and every single rematch."""
    app = _online_app()
    app.coordinator._pending_game_start_payload = {"your_color": "white"}

    assert app.coordinator._heartbeat_ply() is None


def test_adopting_a_resumed_game_releases_the_pairing_it_was_holding(monkeypatch, tmp_path):
    """A pairing left pending would pin the reporter to None for the rest of the
    session, so the server could never judge this client again. Reconnect adopts
    a whole game without ever finishing the card, so it has to clear it."""
    monkeypatch.setenv("CHESS_DATA_DIR", str(tmp_path))
    app = _resumable_app()
    app.coordinator.client.room_id = "room-1"
    app.coordinator._pending_game_start_payload = {"your_color": "white"}

    app.coordinator._adopt_resumed_game(_resumed_payload(
        your_color="white", white_name="Alice", black_name="Bob",
        time_minutes=5, increment_seconds=0))

    assert app.coordinator._pending_game_start_payload is None
    assert app.coordinator._heartbeat_ply() == 0, \
        "and the heartbeat can report the board it just adopted"


def test_no_heartbeat_is_sent_while_a_skill_check_verdict_is_pending():
    """LOAD-BEARING: the verdict flourish plays out over several hundred ms and
    can outlast the server's transit grace, and during it the server has already
    applied the move while this client has not. Staying silent is what keeps
    that window out of the server's judgement entirely."""
    app = _online_app()
    app.coordinator.client.state = "connected"
    app.coordinator._last_heartbeat_sent_ms = pg.time.get_ticks() - 5000
    app.game.skillcheck_session.online_verdict_action = lambda: None

    app.coordinator._send_heartbeat_if_due()

    app.coordinator.client.send_ping.assert_not_called()

    app.game.skillcheck_session.online_verdict_action = None
    app.coordinator._send_heartbeat_if_due()
    app.coordinator.client.send_ping.assert_called_once_with(0)


def test_the_spectator_is_silent_through_the_verdict_too():
    """The opponent watches the check as a read-only mirror and applies the same
    verdict action, so their window is the same shape as the mover's. Driven
    through the real verdict the server broadcasts -- skill_check_result, which
    only ever carries a miss -- so the flag being set is the production path's
    doing rather than the test's."""
    from tests.online.test_online_skillcheck_client import (
        _capture_board, _online_app as _skillcheck_app, _result, _spectate_payload,
    )
    app = _skillcheck_app("black")
    app.screen = app.game
    frm, to = _capture_board(app)
    app.coordinator._handle_skill_check_spectate(_spectate_payload(frm, to))

    app.coordinator._handle_skill_check_result(_result(frm, to))

    assert app.game.skillcheck_session.online_verdict_action is not None, \
        "the real spectate verdict is what opens the window, not a hand-set flag"
    app.coordinator._last_heartbeat_sent_ms = pg.time.get_ticks() - 5000

    app.coordinator._send_heartbeat_if_due()

    assert app.coordinator.client.pings == 0


def test_a_rejected_quiet_move_asks_for_the_whole_state_back():
    """The one window where this client is AHEAD of the server: it applied its
    own quiet move locally and the server refused it. No heartbeat can catch
    that -- the client is on a ply the server will never reach -- so the
    rejection itself has to drive the repair."""
    from chessshootout.server.protocol import Reason
    app = _online_app()

    app.coordinator._handle_online_error(
        {"reason": Reason.NOT_YOUR_TURN, "msg_type": "move"})

    assert app.coordinator._resyncing is True
    app.coordinator.client.request_state_sync.assert_called_once()


def test_a_rejected_skill_check_shot_is_not_a_desync():
    """Scoped by msg_type on purpose: a refused skill-check input says nothing
    about the board, and resyncing there would tear down a live check."""
    from chessshootout.server.protocol import Reason
    app = _online_app()

    app.coordinator._handle_online_error(
        {"reason": Reason.INVALID_MOVE_FORMAT, "msg_type": "skill_check_shot"})

    assert app.coordinator._resyncing is False
    app.coordinator.client.request_state_sync.assert_not_called()


def test_a_takeback_refusal_still_only_toasts():
    """not_your_turn answering a takeback request is an ordinary game-state
    answer with its own toast, and must not be swept into the move-rejection
    branch."""
    from chessshootout.server.protocol import Reason
    app = _online_app()

    app.coordinator._handle_online_error(
        {"reason": Reason.NOT_YOUR_TURN, "msg_type": "takeback_request"})

    assert app.coordinator._resyncing is False
    assert app.toast.message == "Take back is only available right after your move"
