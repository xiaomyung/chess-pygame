"""WS dispatch DEBUG observability + per-WS rate limit.

The dispatch path emits a DEBUG line of the shape
``ws dispatch room=… uuid=… type=… latency_ms=… outcome=…``; we assert that
shape (and the field values) by capturing the records emitted during a real
in-process WS session, plus a unit test on ``dispatch`` so the outcome string
stays part of every handler's public contract.
"""
import json
import logging
import random

import pytest
from httpx import ASGITransport, AsyncClient

from chessshootout.server.broadcasts import broadcast_game_start
from chessshootout.server.handlers import HANDLERS, dispatch
from chessshootout.server.protocol import (
    GRACE_SECONDS, IDLE_RESIGN_SECONDS, PROTOCOL_VERSION,
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS, Reason,
)
from chessshootout.server.limits import WS_MESSAGES_PER_SECOND
from chessshootout.server.ws_session import _ws_session
from chessshootout.server.sweep import PREGAME_CONNECT_GRACE_SECONDS
from tests.server.conftest import ALICE, BOB, RecordingWS, auth_msg, pair_room


def _matchmake(client, *, uuid, side):
    return client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": uuid,
        "nickname": uuid[:8], "time_minutes": 5, "increment_seconds": 0,
        "side_preference": side,
    })


def _parse_dispatch_line(line):
    """``ws dispatch room=… uuid=… type=… latency_ms=… outcome=…`` -> field dict."""
    body = line.split("ws dispatch", 1)[1].strip()
    fields = {}
    for token in body.split():
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def test_handlers_dispatch_table_covers_all_known_message_types():
    expected = {
        "move", "resign", "draw_offer", "draw_response",
        "rematch_request", "rematch_response", "left_result",
        "takeback_request", "takeback_response",
        "give_time", "ping", "skill_check_shot",
        "annotations_state", "annotation_delta", "set_marks_visibility",
        "quick_chat",
    }
    assert set(HANDLERS.keys()) == expected


def test_dispatch_debug_log_reports_move_room_type_and_outcome(client, caplog):
    """The first move's dispatch line carries all five keys plus type=move,
    outcome=applied, and the room id it acted on — verified against
    handle_move (returns "applied") and the app.py log format."""
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    room_id = r1.json()["room_id"]
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        with client.websocket_connect(f"/ws/{room_id}") as ws_w:
            ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
            with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
                ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
                ws_w.receive_text()
                ws_b.receive_text()
                ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                            "type": "move",
                                            "from": "e2", "to": "e4"}))
                ws_w.receive_text()
                ws_b.receive_text()
    dispatch_lines = [r.getMessage() for r in caplog.records
                      if "ws dispatch" in r.getMessage()]
    assert dispatch_lines, "expected at least one ws dispatch DEBUG line"
    move_fields = next(
        (f for f in map(_parse_dispatch_line, dispatch_lines)
         if f.get("type") == "move"), None)
    assert move_fields is not None, "expected a ws dispatch line for the move"
    assert set(move_fields) == {"room", "uuid", "type", "latency_ms", "outcome"}
    assert move_fields["room"] == room_id
    assert move_fields["uuid"] == ALICE[:8]
    assert move_fields["outcome"] == "applied"
    assert float(move_fields["latency_ms"]) >= 0.0


def test_matchmake_logs_room_created_then_room_paired_with_both_short_uuids(client, caplog):
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        r1 = _matchmake(client, uuid=ALICE, side="white")
        room_id = r1.json()["room_id"]
        assert any(
            f"room created room={room_id} uuid={ALICE[:8]}" in r.getMessage()
            for r in caplog.records), "first player logs a room-created breadcrumb"
        r2 = _matchmake(client, uuid=BOB, side="black")
        assert r2.json()["room_id"] == room_id
    paired = [r.getMessage() for r in caplog.records if r.getMessage().startswith("room paired")]
    assert paired, "pairing must log a dedicated breadcrumb"
    assert f"room={room_id}" in paired[0]
    assert ALICE[:8] in paired[0]
    assert BOB[:8] in paired[0]


def test_resume_logs_served_with_ply_count(client, caplog):
    r1 = _matchmake(client, uuid=ALICE, side="white")
    _matchmake(client, uuid=BOB, side="black")
    room_id = r1.json()["room_id"]
    token = r1.json()["session_token"]
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        resp = client.post("/resume", json={
            "version": PROTOCOL_VERSION, "room_id": room_id, "session_token": token,
        })
    assert resp.status_code == 200
    served = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("resume served")]
    assert served == [f"resume served room={room_id} color=white ply=0"]


def test_per_ws_rate_limit_constant_is_documented():
    """Pin the documented 30/sec WS threshold so the limiter can't drift silently."""
    assert WS_MESSAGES_PER_SECOND == 30


def test_per_ws_rate_limit_emits_rate_limited_error(client):
    """A burst past the per-WS cap yields at least one rate_limited error.

    Bogus moves are used because they're cheap and never mutate game state, so
    the burst response is pure rate-limit feedback.
    """
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            for _ in range(WS_MESSAGES_PER_SECOND + 5):
                ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                            "type": "move",
                                            "from": "z9", "to": "a1"}))
            seen = []
            for _ in range(WS_MESSAGES_PER_SECOND + 5):
                msg = json.loads(ws_w.receive_text())
                seen.append(msg)
            reasons = [m.get("reason") for m in seen if m.get("type") == "error"]
            assert Reason.RATE_LIMITED in reasons, (
                "expected at least one rate_limited error in burst response")


@pytest.mark.asyncio
async def test_dispatch_returns_invalid_message_for_unknown_type(app):
    """dispatch returns (msg_type, "invalid_message") and sends an error before
    consulting any handler when the type is unknown, so room may be None."""
    class _FakeWS:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)
    ws = _FakeWS()

    msg_type, outcome = await dispatch(app, ws, room=None, color="white",
                                         raw='{"type":"made_up","version":1}')
    assert msg_type == "made_up"
    assert outcome == "invalid_message"
    assert any(p.get("reason") == Reason.INVALID_MESSAGE for p in ws.sent)


@pytest.mark.asyncio
async def test_the_dispatch_outcome_separates_a_tolerated_ply_from_a_directed_one(
    app, clock,
):
    """The outcome word is the only place a tolerated mismatch is visible in
    production: prod runs at INFO, and both cases used to return the same
    `ping`. The identical heartbeat now reads as `ping_inflight` while the move
    is still in flight and `ping_directed` once it plainly is not, and the
    dispatch line carries that word verbatim (pinned above for `move`)."""
    room, ws_w, ws_b = await _wired_room(app)
    await broadcast_game_start(app.state.connections, room, clock)
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    await _play(app, room, ws_w, ws_b, [("e2", "e4")])

    _, inflight = await dispatch(app, ws_b, room, "black", _msg(type="ping", ply=0))
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    outcomes = [await dispatch(app, ws_b, room, "black", _msg(type="ping", ply=0))
                for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS)]

    assert inflight == "ping_inflight"
    assert [o for _, o in outcomes] == (
        ["ping_strike"] * (RESYNC_STABLE_MISMATCH_HEARTBEATS - 1) + ["ping_directed"])


FINALIZE_PREFIX = "game finalized"

REMOVED_GAME_END_PREFIXES = (
    "game over ", "idle timeout ", "resign room=", "abandonment room=",
    "draw mutual",
)


def _msg(**fields):
    return json.dumps({"version": PROTOCOL_VERSION, **fields})


async def _wired_room(app):
    """A paired room with both sockets filed, the shape every ending starts from."""
    rooms = app.state.rooms
    room = await pair_room(rooms)
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    rooms.mark_connected(room.room_id, "white")
    rooms.mark_connected(room.room_id, "black")
    return room, ws_w, ws_b


async def _play(app, room, ws_w, ws_b, moves):
    for i, (frm, to) in enumerate(moves):
        color = "white" if i % 2 == 0 else "black"
        ws = ws_w if color == "white" else ws_b
        await dispatch(app, ws, room, color,
                       _msg(type="move", **{"from": frm, "to": to}))


async def _end_by_checkmate(app, clock):
    """handlers.py, the commonest ending of all: mate landing on a played move."""
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b,
                [("f2", "f3"), ("e7", "e5"), ("g2", "g4"), ("d8", "h4")])
    return room


async def _end_by_resignation(app, clock):
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    await dispatch(app, ws_w, room, "white", _msg(type="resign"))
    return room


async def _end_by_reciprocated_draw_offer(app, clock):
    """Offering into a standing offer is an acceptance -- handle_draw_offer's own
    finalize, distinct from the draw_response one below."""
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    await dispatch(app, ws_w, room, "white", _msg(type="draw_offer"))
    await dispatch(app, ws_b, room, "black", _msg(type="draw_offer"))
    return room


async def _end_by_accepted_draw_response(app, clock):
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    await dispatch(app, ws_w, room, "white", _msg(type="draw_offer"))
    await dispatch(app, ws_b, room, "black", _msg(type="draw_response", accept=True))
    return room


async def _end_by_flag_fall(app, clock):
    rooms = app.state.rooms
    room = await pair_room(rooms, time_minutes=1)
    app.state.connections.add(room.room_id, room.white.client_uuid, RecordingWS())
    app.state.connections.add(room.room_id, room.black.client_uuid, RecordingWS())
    room.plies_ever = 1
    room.first_move_at = clock()
    clock.advance(70)
    await app.state.sweep.step_clock_and_idle_windows()
    return room


async def _end_by_idle_window(app, clock):
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    clock.advance(IDLE_RESIGN_SECONDS + 1)
    await app.state.sweep.step_clock_and_idle_windows()
    return room


async def _end_by_grace_expiry(app, clock):
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    app.state.rooms.mark_disconnected(room.room_id, "black")
    clock.advance(GRACE_SECONDS + 1)
    await app.state.sweep.step_grace_expired()
    return room


async def _end_by_matchmake_abandon(app, clock):
    """app.py: searching again while a game is running gives that game away."""
    room, ws_w, ws_b = await _wired_room(app)
    await _play(app, room, ws_w, ws_b, [("e2", "e4"), ("e7", "e5")])
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://obs") as http:
        r = await http.post("/matchmake", json={
            "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "A",
            "time_minutes": 10, "increment_seconds": 0, "side_preference": "random",
        })
    assert r.status_code == 200
    return room


FINALIZE_PATHS = [
    pytest.param(_end_by_checkmate, Reason.CHECKMATE, "black", 4, 0.0,
                 id="handlers_checkmate_after_move"),
    pytest.param(_end_by_resignation, Reason.RESIGNATION, "black", 2, 0.0,
                 id="handlers_resign"),
    pytest.param(_end_by_reciprocated_draw_offer, Reason.DRAW_AGREEMENT, "none", 2, 0.0,
                 id="handlers_draw_offer_reciprocated"),
    pytest.param(_end_by_accepted_draw_response, Reason.DRAW_AGREEMENT, "none", 2, 0.0,
                 id="handlers_draw_response_accepted"),
    pytest.param(_end_by_flag_fall, Reason.TIMEOUT, "black", 1, 70.0,
                 id="sweep_flag_fall"),
    pytest.param(_end_by_idle_window, Reason.RESIGNATION, "black", 2,
                 IDLE_RESIGN_SECONDS + 1.0, id="sweep_idle_window"),
    pytest.param(_end_by_grace_expiry, Reason.ABANDONMENT, "white", 2,
                 GRACE_SECONDS + 1.0, id="sweep_grace_expired"),
    pytest.param(_end_by_matchmake_abandon, Reason.ABANDONMENT, "black", 2, 0.0,
                 id="app_matchmake_abandons"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("drive, reason, winner, plies, duration", FINALIZE_PATHS)
async def test_every_ending_logs_exactly_one_game_finalized_line(
    app, clock, caplog, drive, reason, winner, plies, duration,
):
    """Eight code paths end a game and they used to log eight different things --
    or, for the commonest one of all (mate after a move), nothing at all. They all
    run through finalize_and_broadcast, so the one line lives there: whatever ends
    a game, the journal gets exactly one `game finalized` with the same fields.

    The parametrisation is the point. A ninth ending added anywhere else would
    only be covered here if it, too, went through the funnel."""
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        room = await drive(app, clock)
    assert room.result is not None, "the driver has to actually end the game"
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith(FINALIZE_PREFIX)]
    assert lines == [
        f"game finalized room={room.room_id} reason={reason} winner={winner} "
        f"plies={plies} duration_s={duration:.1f}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("drive, reason, winner, plies, duration", FINALIZE_PATHS)
async def test_no_ending_still_logs_one_of_the_replaced_lines(
    app, clock, caplog, drive, reason, winner, plies, duration,
):
    """The cut-over half: `game over`, `idle timeout`, `resign room=`,
    `abandonment room=` and `draw mutual` are gone for good. Left in place beside
    the funnel line they would double-report every ending, which is exactly the
    noise the funnel exists to remove."""
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        room = await drive(app, clock)
    assert room.result is not None, \
        "the driver has to actually end the game, or an empty log proves nothing"
    stale = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith(REMOVED_GAME_END_PREFIXES)]
    assert stale == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(_end_by_reciprocated_draw_offer, id="offer_into_a_standing_offer"),
        pytest.param(_end_by_accepted_draw_response, id="explicit_accept"),
    ],
)
async def test_both_ways_of_accepting_a_draw_log_the_same_line(app, clock, caplog, drive):
    """A draw is agreed either by answering the offer or by offering back, and the
    two used to read differently (`draw mutual room=…` had no `by=` at all), so
    grepping `draw accepted` found only half of them."""
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        room = await drive(app, clock)
    accepted = [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("draw accepted")]
    assert accepted == [f"draw accepted room={room.room_id} by=black"]


class _ExplodingWS(RecordingWS):
    """Hands out the queued frames, then fails the way a broken transport does:
    with something that is neither WebSocketDisconnect nor RuntimeError."""

    def __init__(self, frames):
        super().__init__()
        self._frames = list(frames)

    async def receive_text(self):
        if self._frames:
            return self._frames.pop(0)
        raise ValueError("transport went sideways")


@pytest.mark.asyncio
async def test_unexpected_ws_recv_failure_logs_a_traceback(app, caplog):
    """The catch-all arm used to log `exc=%r` at WARNING: a repr, no stack, no
    level an operator alerts on. An exception nobody anticipated is exactly the
    case where the traceback IS the diagnosis."""
    rooms = app.state.rooms
    room = await pair_room(rooms)
    ws = _ExplodingWS([json.dumps(auth_msg("ta"))])

    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        await _ws_session(app, ws, room.room_id)

    failures = [r for r in caplog.records if r.getMessage().startswith("ws recv failed")]
    assert len(failures) == 1
    record = failures[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None, "no traceback attached"
    assert record.exc_info[0] is ValueError
    assert record.getMessage() == f"ws recv failed room={room.room_id} color=white"


@pytest.mark.asyncio
async def test_the_pre_game_orphan_drop_ends_a_room_without_finalizing_it(
    app, clock, caplog,
):
    """One of the two endings that deliberately never reach the funnel: a pairing
    both players walked away from before a move is dropped outright, with no
    result written -- so nothing was finalized and the journal must not claim it
    was. (The other is the shutdown frame, which tells live sockets the process is
    going down without touching any room's result; pinned in test_server_app.)"""
    rooms = app.state.rooms
    room = await pair_room(rooms)
    clock.advance(PREGAME_CONNECT_GRACE_SECONDS + 1)

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        app.state.sweep.step_drop_orphans_pre_game()

    assert rooms.get(room.room_id) is None
    assert room.result is None
    assert [r.getMessage() for r in caplog.records
            if r.getMessage().startswith(FINALIZE_PREFIX)] == []
