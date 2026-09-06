import asyncio
import json
import logging
import random
import sys

import pytest

from chessshootout.server import __main__ as server_main
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from slowapi import Limiter

from chessshootout.server import app as app_module
from chessshootout.server.app import _sweep_loop, create_app
from chessshootout.server.limits import MATCHMAKE_PER_IP_LIMIT, UuidRateLimiter
from chessshootout.server.ws_session import MAX_INBOUND_MESSAGE_BYTES
from chessshootout.server.broadcasts import broadcast_game_start, idle_window_wire
from chessshootout.server.connections import ConnectionRegistry
from chessshootout.server.handlers import (
    RESYNC_DIRECTIVE, RESYNC_GATE_PRUNE_THRESHOLD, RESYNC_NOTIFY,
    RESYNC_NOTIFY_FLAP_FLOOR_SECONDS, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS,
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS,
    _ResyncGate, handle_ping,
)
from chessshootout.server.protocol import (
    FIRST_MOVE_ABORT_SECONDS, GRACE_SECONDS, HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_MISS_LIMIT, HEARTBEAT_TIMEOUT_SECONDS, HealthStatus,
    CLIENT_VERSION_MAX_LEN, IDLE_RESIGN_SECONDS, MAX_INCREMENT_SECONDS,
    MAX_TIME_MINUTES, MIN_CLIENT_VERSION, MIN_INCREMENT_SECONDS, MIN_TIME_MINUTES,
    PROTOCOL_VERSION, Reason, WS_CLOSE_INVALID_TOKEN, WS_CLOSE_PAYLOAD_TOO_LARGE,
    WS_CLOSE_SERVER_SHUTDOWN, WS_CLOSE_SUPERSEDED,
)
from chessshootout.server.rooms import QUEUE_ABANDON_SECONDS, RoomManager
from chessshootout.server.sweep import SWEEP_STALE_SECONDS, Sweep
from tests.helpers import FakeClock, fake_uuid4
from tests.server.conftest import ALICE, BOB, auth_msg
from tests.server.test_server_broadcasts import RecordingWS
from tests.server.test_server_logging import KV_TOKEN_RE


async def _sweep(app):
    await app.state.sweep.step_all()


def test_registry_add_returns_displaced_socket():
    reg = ConnectionRegistry()
    ws_old, ws_new = object(), object()
    assert reg.add("r", "u", ws_old) is None
    assert reg.add("r", "u", ws_new) is ws_old
    assert reg.add("r", "u", ws_new) is None


def test_registry_remove_is_identity_guarded():
    reg = ConnectionRegistry()
    ws_old, ws_new = object(), object()
    reg.add("r", "u", ws_old)
    reg.add("r", "u", ws_new)
    assert reg.remove("r", "u", ws_old) is False
    assert reg._by_room["r"]["u"] is ws_new
    assert reg.remove("r", "u", ws_new) is True
    assert "r" not in reg._by_room


def test_registry_remove_unknown_room_returns_false():
    reg = ConnectionRegistry()
    assert reg.remove("nope", "u", object()) is False


APP_STATE_TYPES = {
    "rooms": RoomManager,
    "connections": ConnectionRegistry,
    "limiter": Limiter,
    "started_at": float,
    "reclaim_limiter": UuidRateLimiter,
    "annotation_limiter": UuidRateLimiter,
    "chat_limiter": UuidRateLimiter,
    "moderation_enabled": bool,
    "sweep": Sweep,
}
PER_APP_OBJECTS = ("rooms", "connections", "limiter", "reclaim_limiter",
                   "annotation_limiter", "chat_limiter")


def test_app_state_carries_every_shared_service(app):
    """create_app's contract with everything downstream of it: eleven names on
    app.state, of the right kinds. Every handler, the sweep loop and the ws
    session reach their collaborators through exactly these, so a rename or a
    dropped assignment is a runtime AttributeError deep inside a request rather
    than an import error at boot."""
    for name, expected in APP_STATE_TYPES.items():
        value = getattr(app.state, name)
        assert isinstance(value, expected), f"app.state.{name} is a {type(value).__name__}"
    assert callable(app.state.now), "the injected monotonic clock"
    assert callable(app.state.now_ms), "and the same clock in milliseconds"
    assert app.state.now_ms() == pytest.approx(app.state.now() * 1000.0)


def test_two_apps_share_no_mutable_state():
    """Nothing built in create_app may live at module scope. Two apps in one
    process -- which is exactly what the test suite is, a couple of hundred of
    them -- must not see each other's rooms, sockets or spent allowances."""
    a = create_app(now_provider=FakeClock(), max_rooms=8)
    b = create_app(now_provider=FakeClock(), max_rooms=8)
    for name in PER_APP_OBJECTS:
        assert getattr(a.state, name) is not getattr(b.state, name), \
            f"app.state.{name} is shared between two applications"


CARL = fake_uuid4(3)
ZED = fake_uuid4(99)


def _matchmake(client, *, uuid=ALICE, nickname="Alice", time=5, inc=0, side="random"):
    return client.post("/matchmake", json={
        "version": PROTOCOL_VERSION,
        "client_uuid": uuid, "nickname": nickname,
        "time_minutes": time, "increment_seconds": inc,
        "side_preference": side,
    })


def test_root_manifest_names_the_gameserver(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "gameserver"
    assert body["version"] == PROTOCOL_VERSION
    assert "/ws/{room_id}" in body["endpoints"]


def test_root_manifest_publishes_the_oldest_accepted_build(client):
    """MIN_CLIENT_VERSION is a source constant baked into the image, so the
    manifest is the only way an operator can read back which floor the running
    container is actually enforcing -- `curl -s <server>/` after a deploy."""
    body = client.get("/").json()
    assert body["min_client_version"] == MIN_CLIENT_VERSION


def test_health_returns_zero_rooms_initially(client):
    """/healthz exposes status, rooms_active, queue_depth, uptime_s, version, app_version."""
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["rooms_active"] == 0
    assert body["queue_depth"] == 0
    assert body["uptime_s"] >= 0.0
    assert body["version"] == PROTOCOL_VERSION


def test_health_reports_app_version_via_metadata(client):
    """app_version is additive (protocol `version` int is unchanged) and resolves
    from installed dist metadata. It is NOT asserted equal to the pyproject version:
    an editable dev install freezes dist metadata at install time, so it can lag the
    current pyproject value; only a fresh (Docker) wheel matches."""
    from importlib.metadata import version as pkg_version

    body = client.get("/healthz").json()
    assert isinstance(body["app_version"], str)
    assert body["app_version"] == pkg_version("chess-shootout")
    assert isinstance(body["version"], int)


def test_matchmake_returns_room_and_token(client):
    r = _matchmake(client)
    assert r.status_code == 200
    body = r.json()
    assert "room_id" in body and "session_token" in body


def test_matchmake_refuses_a_body_stamped_with_another_protocol(client):
    """Until now no HTTP route checked `version` at all: a build on the previous
    protocol was paired happily and then fell out at the websocket handshake,
    which is what left the player watching the search spinner for ever. The
    refusal is 426 so the client can decide on the status alone."""
    r = client.post("/matchmake", json={
        "version": PROTOCOL_VERSION - 1, "client_uuid": ALICE, "nickname": "Alice",
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
    })
    assert r.status_code == 426
    assert r.json() == {"detail": {"reason": Reason.VERSION_MISMATCH}}
    assert client.get("/healthz").json()["queue_depth"] == 0


def test_a_protocol_gap_refusal_says_so_in_the_journal(client, caplog):
    """The gate refuses before the endpoint's own matchmake line, so with no
    line of its own the request left no trace at all: an operator looking into
    a player who cannot get into a game would see nothing rather than the
    reason. The version is an int off a validated model, so it can carry no
    forged second line."""
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        r = client.post("/matchmake", json={
            "version": PROTOCOL_VERSION - 1, "client_uuid": ALICE, "nickname": "Alice",
            "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
        })

    assert r.status_code == 426
    records = [rec.getMessage() for rec in caplog.records
               if rec.name == "chess.server.app"]
    assert records == [f"matchmake rejected uuid={ALICE[:8]} "
                       f"reason={Reason.VERSION_MISMATCH} "
                       f"version={PROTOCOL_VERSION - 1}"]
    prefix = records[0][:KV_TOKEN_RE.search(records[0]).start()]
    assert prefix.strip() and "=" not in prefix and prefix[0].islower()
    assert dict(KV_TOKEN_RE.findall(records[0])).keys() == {"uuid", "reason", "version"}


@pytest.mark.parametrize(
    "client_version",
    [
        pytest.param("0.0.1", id="an_old_stamped_build"),
        pytest.param("garbage", id="a_version_nobody_can_read"),
        pytest.param("2.13.0-rc1", id="a_decorated_version_is_not_a_version"),
    ],
)
def test_matchmake_refuses_an_outdated_build_and_names_the_floor(client, client_version):
    """The refusal carries the minimum, because the update card names both
    numbers and the client only ever learns the server's floor from here."""
    r = client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
        "client_version": client_version,
    })
    assert r.status_code == 426
    assert r.json() == {"detail": {"reason": Reason.CLIENT_OUTDATED,
                                   "min_version": MIN_CLIENT_VERSION}}
    assert client.get("/healthz").json()["queue_depth"] == 0


@pytest.mark.parametrize(
    "client_version",
    [
        pytest.param("", id="a_source_run_states_no_version"),
        pytest.param(MIN_CLIENT_VERSION, id="exactly_the_minimum"),
        pytest.param("9.9.9", id="a_build_newer_than_this_server"),
    ],
)
def test_matchmake_admits_every_build_at_or_above_the_floor(client, client_version):
    """The gate turns away only what it must. The empty version is the
    source-run exemption -- a checkout ships no version.txt, and the protocol
    number already turns an incompatible build away -- and a newer build must
    never be locked out by a server that has not been redeployed yet."""
    r = client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
        "client_version": client_version,
    })
    assert r.status_code == 200


def test_an_outdated_request_leaves_the_senders_existing_room_alone(client, app, clock):
    """The gate runs before the endpoint's give-up-your-old-seat block, which
    would otherwise abandon (and finalize as a loss) the game this same uuid is
    already playing. A downgraded or spoofed build must not be a way to end
    somebody's live game from outside it."""
    assert _matchmake(client, uuid=ALICE).status_code == 200
    assert _matchmake(client, uuid=BOB).status_code == 200
    rooms = app.state.rooms
    live = list(rooms._active.values())
    assert len(live) == 1, "Alice and Bob are paired into one room"
    live[0].first_move_at = clock()
    before = rooms.in_progress_room_for(ALICE)
    assert before is not None, "the game has started, so it is abandonable"
    room, color = before

    refused = client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
        "client_version": "0.0.1",
    })

    assert refused.status_code == 426
    after = rooms.in_progress_room_for(ALICE)
    assert after is not None
    assert after[0] is room and after[1] == color
    assert room.result is None, "the refused request finalized nothing"
    assert app.state.rooms.rooms_active == 1


def test_a_refused_build_still_spends_its_per_ip_matchmake_allowance(client):
    """The limiter decorator sits outside the handler, so a 426 costs the same
    allowance a 200 does. That is the point: refusing outdated builds must not
    hand an attacker an unmetered endpoint to hammer."""
    budget = int(MATCHMAKE_PER_IP_LIMIT.split("/")[0])
    payload = {
        "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
        "client_version": "0.0.1",
    }
    for _ in range(budget):
        assert client.post("/matchmake", json=payload).status_code == 426
    limited = client.post("/matchmake", json=payload)
    assert limited.status_code == 429
    assert limited.json()["detail"]["reason"] == Reason.RATE_LIMITED


def test_the_outdated_refusal_logs_the_parsed_version_and_never_the_raw_text(
    client, caplog,
):
    """`client_version` is attacker-supplied text that reaches an operator's
    journal, so a newline in it would forge a whole extra log line. The line
    prints the value re-written from the numbers the server parsed, and prints
    `unparseable` when there were none -- one record either way."""
    forged = "2.13.0\nroom created room=x"
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        r = client.post("/matchmake", json={
            "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
            "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
            "client_version": forged,
        })
    assert r.status_code == 426
    records = [rec.getMessage() for rec in caplog.records
               if rec.name == "chess.server.app"]
    assert len(records) == 1, f"one record per refused build, got {records}"
    assert records[0] == (f"matchmake rejected uuid={ALICE[:8]} "
                          f"reason={Reason.CLIENT_OUTDATED} version=unparseable")
    assert not any("\n" in message for message in records)


def test_the_outdated_refusal_logs_a_readable_version_when_there_is_one(client, caplog):
    """The other half: a version that parses is named, so the operator can see
    which build is being turned away rather than only that one was."""
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        client.post("/matchmake", json={
            "version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
            "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
            "client_version": "2.12.2",
        })
    records = [rec.getMessage() for rec in caplog.records
               if rec.name == "chess.server.app"]
    assert records == [f"matchmake rejected uuid={ALICE[:8]} "
                       f"reason={Reason.CLIENT_OUTDATED} version=2.12.2"]


@pytest.mark.parametrize(
    "body, field",
    [
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ALICE, "nickname": "Alice",
             "time_minutes": 0, "increment_seconds": 0, "side_preference": "random"},
            "time_minutes", id="zero_time_minutes",
        ),
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ZED, "nickname": "Z",
             "time_minutes": 5, "increment_seconds": -1},
            "increment_seconds", id="negative_increment",
        ),
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ZED,
             "nickname": "", "time_minutes": 5, "increment_seconds": 0},
            "nickname", id="empty_nickname",
        ),
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ZED, "nickname": "Z",
             "time_minutes": MAX_TIME_MINUTES + 1, "increment_seconds": 0},
            "time_minutes", id="time_minutes_over_cap",
        ),
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ZED, "nickname": "Z",
             "time_minutes": 5, "increment_seconds": MAX_INCREMENT_SECONDS + 1},
            "increment_seconds", id="increment_over_cap",
        ),
        pytest.param(
            {"version": PROTOCOL_VERSION, "client_uuid": ZED, "nickname": "Z",
             "time_minutes": 5, "increment_seconds": 0,
             "client_version": "9" * (CLIENT_VERSION_MAX_LEN + 1)},
            "client_version", id="client_version_too_long",
        ),
    ],
)
def test_matchmake_rejects_invalid_field(client, caplog, body, field):
    """Each invalid matchmake field is rejected with 422 before any room is created.

    The two over-cap cases are the SECURITY half: an unbounded time control mints
    a fresh never-pairing queue bucket per distinct pair, so the rejection has to
    land before enqueue touches `_queue`.

    The 422 body is the closed reason envelope both sides share, so WHERE the
    rejection happened can no longer be read off the response -- the caplog
    assertion carries that instead: `field=body.<name>` proves MatchmakeRequest's
    own bounds refused the request at the model boundary and the endpoint body
    never ran. The endpoint used to re-check the time control by hand and answer
    `{"reason": "invalid_time_control"}` -- dead code, since the model's bounds are
    strictly tighter, and this is the assertion that fails if such a shadowing hand
    check ever comes back."""
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        r = client.post("/matchmake", json=body)
    assert r.status_code == 422
    assert r.json() == {"detail": {"reason": Reason.INVALID_FIELD}}
    rejected = [rec.getMessage() for rec in caplog.records
                if rec.getMessage().startswith("request rejected")]
    assert len(rejected) == 1, f"one WARNING per refused body, got {rejected}"
    assert "path=/matchmake" in rejected[0]
    assert f"field=body.{field}" in rejected[0]
    assert client.get("/healthz").json()["rooms_active"] == 0
    assert client.get("/healthz").json()["queue_depth"] == 0


def test_the_rejection_warning_never_carries_the_rejected_value(client, caplog):
    """The refused body is attacker-controlled, so the WARNING names the field and
    the error kind and stops there. Logging the value would put arbitrary text --
    newlines included -- into the operator's journal through a public endpoint."""
    leaky = "leak-me-9a7f3c"
    with caplog.at_level(logging.DEBUG, logger="chess.server.app"):
        r = client.post("/matchmake", json={
            "version": PROTOCOL_VERSION, "client_uuid": leaky, "nickname": "Z",
            "time_minutes": 5, "increment_seconds": 0,
        })
    assert r.status_code == 422
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("field=body.client_uuid" in m for m in messages)
    assert not any(leaky in m for m in messages), "the rejected value must not be logged"


def test_matchmake_model_bounds_subsume_the_removed_hand_checked_floor():
    """The endpoint's `time_minutes < 1 or increment_seconds < 0` guard could never
    fire: MatchmakeRequest already refuses both at the model boundary, so the body
    was rejected a layer earlier. Removing it stays safe only while the model floors
    remain at least as tight as the guard's were -- lowering either would reopen a
    hole the guard is no longer there to (never actually) cover."""
    assert MIN_TIME_MINUTES >= 1
    assert MIN_INCREMENT_SECONDS >= 0


def test_matchmake_rejects_already_in_game(client):
    r1 = _matchmake(client, uuid=ALICE)
    r2 = _matchmake(client, uuid=BOB)
    assert r1.status_code == 200 and r2.status_code == 200
    r3 = _matchmake(client, uuid=ALICE)
    assert r3.status_code == 409


def test_matchmake_replaces_the_callers_own_stale_queue_slot(app, client):
    """A player who walked away from a queue slot -- app killed, socket gone, no
    DELETE -- used to be answered 409 already_in_game on every retry until the
    120s abandoned-queue reaper ran, so hitting Play again simply did nothing for
    up to two minutes. The caller's OWN queued room is released and replaced
    instead; only rooms belonging to somebody else still refuse."""
    first = _matchmake(client, uuid=ALICE, time=5)

    second = _matchmake(client, uuid=ALICE, time=10)

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["room_id"] != first.json()["room_id"], "a fresh slot, not the old one"
    assert app.state.rooms.get(first.json()["room_id"]) is None, "the stale slot is gone"
    assert client.get("/healthz").json()["queue_depth"] == 1, "one waiter, never two"
    assert app.state.rooms._queue.keys() == {(10, 0)}, "the emptied 5+0 bucket goes too"


def test_matchmake_after_dropping_a_queued_socket_succeeds(client):
    """The same release, driven the way a real client hits it: matchmake, open the
    websocket to the queued room, drop it, then search again straight away."""
    first = _matchmake(client, uuid=ALICE)
    body = first.json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(json.dumps(auth_msg(body["session_token"])))

    assert _matchmake(client, uuid=ALICE).status_code == 200


def test_cancel_matchmake_removes_from_queue(client):
    """Cancelling frees the uuid: it can re-matchmake afterwards."""
    r = _matchmake(client, uuid=ALICE)
    body = r.json()
    cancel = client.request("DELETE", "/matchmake", json={
        "version": PROTOCOL_VERSION,
        "room_id": body["room_id"], "session_token": body["session_token"],
    })
    assert cancel.status_code == 200
    r2 = _matchmake(client, uuid=ALICE)
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_abandoned_queue_flood_stops_locking_the_server_out(app, client, clock):
    """SECURITY, end to end: matchmake with a distinct uuid and a distinct time
    control per request so nothing ever pairs, then never open a websocket and
    never cancel. Queue depth counts against max_rooms, and before the reap
    existed those slots were held until process restart -- max_rooms cheap HTTP
    calls answered every later player with 503 room_full forever. One sweep past
    the TTL has to give the capacity back."""
    max_rooms = app.state.rooms._max_rooms
    for i in range(max_rooms):
        assert _matchmake(client, uuid=fake_uuid4(200 + i), nickname=f"N{i}",
                          time=i + 1).status_code == 200
    assert client.get("/healthz").json()["queue_depth"] == max_rooms
    blocked = _matchmake(client, uuid=ZED, nickname="Z", time=100)
    assert blocked.status_code == 503
    assert blocked.json()["detail"]["reason"] == Reason.ROOM_FULL

    clock.advance(QUEUE_ABANDON_SECONDS + 1)
    await _sweep(app)

    assert client.get("/healthz").json()["queue_depth"] == 0
    assert app.state.rooms._queue == {}, "emptied time-control buckets go too"
    assert _matchmake(client, uuid=ZED, nickname="Z", time=100).status_code == 200


def test_matchmake_refused_at_capacity_logs_the_load_that_caused_it(app, client, caplog):
    """The refusal is an operator's first sign that a box is at its cap, and
    `matchmake rejected reason=server_full` said only that it happened -- not
    whether the rooms were games or waiters, which is the difference between
    "raise MAX_ROOMS" and "the queue is leaking". The `reason=` value is the code
    the caller is actually answered with (503 room_full); the old line named an
    internal string that appeared nowhere on the wire.

    The fill is the same distinct-uuid/distinct-time shape as the reap test above,
    so nothing pairs and the whole cap is queue depth."""
    max_rooms = app.state.rooms._max_rooms
    for i in range(max_rooms):
        assert _matchmake(client, uuid=fake_uuid4(200 + i), nickname=f"N{i}",
                          time=i + 1).status_code == 200

    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        blocked = _matchmake(client, uuid=ZED, nickname="Z", time=100)

    assert blocked.status_code == 503
    assert blocked.json()["detail"]["reason"] == Reason.ROOM_FULL
    refusals = [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("matchmake rejected")]
    assert refusals == [
        f"matchmake rejected reason={Reason.ROOM_FULL} rooms_active=0 "
        f"queue_depth={max_rooms} max_rooms={max_rooms}"
    ]


class _ClosingProbeWS(RecordingWS):
    """A filed socket that notes whether the shutdown line was already in the
    journal by the time the server got round to closing it."""

    def __init__(self, caplog):
        super().__init__()
        self._caplog = caplog
        self.line_logged_first = None

    async def close(self, code=1000):
        self.line_logged_first = any(
            rec.getMessage().startswith("gameserver shutting down")
            for rec in self._caplog.records)
        await super().close(code)


def test_shutdown_logs_the_load_it_was_carrying(app, caplog):
    """A stopped process used to leave no trace at all, so a clean stop and a
    crash read identically in the journal.

    The line is the FIRST statement of the lifespan's `finally`, so its numbers
    describe what was torn down rather than the empty server left behind -- the
    probe sockets assert that ordering directly, since they are told the server
    is going down only after the line is written. The fake clock is never
    advanced inside the block, so uptime_s pins the arithmetic exactly."""
    sockets = [_ClosingProbeWS(caplog), _ClosingProbeWS(caplog)]
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        with TestClient(app) as live:
            r1 = _matchmake(live, uuid=ALICE, nickname="A")
            r2 = _matchmake(live, uuid=BOB, nickname="B")
            assert r2.json()["room_id"] == r1.json()["room_id"]
            room = app.state.rooms.get(r1.json()["room_id"])
            for slot, ws in zip((room.white, room.black), sockets):
                app.state.connections.add(room.room_id, slot.client_uuid, ws)

    lines = [rec.getMessage() for rec in caplog.records
             if rec.getMessage().startswith("gameserver shutting down")]
    assert lines == ["gameserver shutting down uptime_s=0.0 rooms_active=1 "
                     "queue_depth=0 sockets=2"]
    assert not [rec for rec in caplog.records
                if rec.getMessage().startswith("game finalized")], \
        "a shutdown ends the process, not the game -- no room's result is written"
    for ws in sockets:
        assert ws.closed_with == WS_CLOSE_SERVER_SHUTDOWN
        assert ws.line_logged_first is True, "the line must precede the teardown"


def test_cancel_matchmake_is_rate_limited_per_ip(client):
    """SECURITY: DELETE /matchmake was the one public endpoint with no per-IP
    limiter, so it was a free unauthenticated way to hammer the room manager's
    lock. It now shares the matchmake budget and answers 429 past it."""
    budget = int(MATCHMAKE_PER_IP_LIMIT.split("/")[0])
    payload = {"version": PROTOCOL_VERSION, "room_id": fake_uuid4(77),
               "session_token": "t"}
    for _ in range(budget):
        assert client.request("DELETE", "/matchmake", json=payload).status_code == 404
    limited = client.request("DELETE", "/matchmake", json=payload)
    assert limited.status_code == 429
    assert limited.json().get("detail", {}).get("reason") == Reason.RATE_LIMITED


def test_search_and_cancel_share_one_matchmake_budget(client):
    """slowapi keys its buckets by request PATH, not by endpoint function, so
    POST and DELETE /matchmake draw on the same 60/minute allowance rather than
    one each. Pinned because it is the surprising half of the fix: the shared
    budget is still ~30x what a human clicking find/cancel can spend, but a
    future split of the two limits would silently double the real ceiling."""
    budget = int(MATCHMAKE_PER_IP_LIMIT.split("/")[0])
    payload = {"version": PROTOCOL_VERSION, "room_id": fake_uuid4(77),
               "session_token": "t"}
    for _ in range(budget - 1):
        assert client.request("DELETE", "/matchmake", json=payload).status_code == 404
    assert _matchmake(client, uuid=ALICE).status_code == 200
    assert _matchmake(client, uuid=BOB).status_code == 429


def test_two_apps_never_share_a_per_ip_matchmake_budget(client):
    """The per-IP limiter is a `Limiter` built inside create_app and bound by the
    @limiter.limit decorator, so its allowance belongs to ONE application. If the
    decorator ever closed over a module-level limiter instead, every test app in
    the process would draw on one wall-clock budget and the suite would start
    failing in whichever order it happened to run.

    The fill uses a single uuid on purpose: each POST releases the caller's own
    queue slot before enqueueing again, so 60 searches spend 60 allowance without
    ever holding more than one room."""
    budget = int(MATCHMAKE_PER_IP_LIMIT.split("/")[0])
    for _ in range(budget):
        assert _matchmake(client, uuid=ALICE).status_code == 200
    limited = _matchmake(client, uuid=ALICE)
    assert limited.status_code == 429
    assert limited.json()["detail"]["reason"] == Reason.RATE_LIMITED

    other = TestClient(create_app(now_provider=FakeClock(), max_rooms=8))
    assert _matchmake(other, uuid=ALICE).status_code == 200, \
        "a second application starts with its own untouched allowance"


def test_two_apps_each_authenticate_on_their_own_websocket_router():
    """The websocket route lives on a MODULE-LEVEL router shared by every
    application in the process -- unlike the HTTP router, which is rebuilt per app
    because its limiter decorators must not share a budget. That is only safe
    while the endpoint reads its application off the live connection instead of
    closing over one: two apps must each pair and authenticate their own players,
    and a token minted by one must mean nothing to the other."""
    apps = [create_app(now_provider=FakeClock(), max_rooms=8) for _ in range(2)]
    clients = [TestClient(a) for a in apps]
    sessions = []
    for c in clients:
        random.seed(0)
        white = _matchmake(c, uuid=ALICE, side="white").json()
        black = _matchmake(c, uuid=BOB, side="black").json()
        assert white["room_id"] == black["room_id"]
        sessions.append((white, black))
    assert sessions[0][0]["room_id"] != sessions[1][0]["room_id"]

    for c, (white, black) in zip(clients, sessions):
        with c.websocket_connect(f"/ws/{white['room_id']}") as ws_w:
            ws_w.send_text(json.dumps(auth_msg(white["session_token"])))
            with c.websocket_connect(f"/ws/{black['room_id']}") as ws_b:
                ws_b.send_text(json.dumps(auth_msg(black["session_token"])))
                assert json.loads(ws_w.receive_text())["type"] == "game_start"
                assert json.loads(ws_b.receive_text())["type"] == "game_start"

    foreign = sessions[0][0]
    with clients[1].websocket_connect(f"/ws/{foreign['room_id']}") as ws:
        ws.send_text(json.dumps(auth_msg(foreign["session_token"])))
        with pytest.raises(Exception):
            ws.receive_text()


def test_cancel_with_bogus_token_rejected(client):
    r = _matchmake(client, uuid=ALICE)
    body = r.json()
    cancel = client.request("DELETE", "/matchmake", json={
        "version": PROTOCOL_VERSION,
        "room_id": body["room_id"], "session_token": "bogus",
    })
    assert cancel.status_code == 401


def test_ws_rejects_bad_auth_token(client):
    _matchmake(client, uuid=ALICE)
    r2 = _matchmake(client, uuid=BOB)
    body = r2.json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "auth",
                                  "session_token": "bogus"}))
        with pytest.raises(Exception):
            ws.receive_text()


def test_ws_rejects_non_auth_first_message(client):
    _matchmake(client, uuid=ALICE)
    r2 = _matchmake(client, uuid=BOB)
    body = r2.json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                  "from": "e2", "to": "e4"}))
        with pytest.raises(Exception):
            ws.receive_text()


def _fat_multibyte_frame(**fields):
    """A frame that FITS the character count and BUSTS the byte count: every pad
    character is two bytes in UTF-8, so len() reads well under the cap while the
    wire payload is roughly double it."""
    payload = json.dumps(
        dict(version=PROTOCOL_VERSION, pad="é" * (MAX_INBOUND_MESSAGE_BYTES - 200),
             **fields),
        ensure_ascii=False)
    assert len(payload) < MAX_INBOUND_MESSAGE_BYTES < len(payload.encode("utf-8"))
    return payload


def test_ws_handshake_over_the_byte_cap_closes_as_too_large(client):
    """SECURITY: the cap is a BYTE budget — it is the very number handed to
    uvicorn's ws_max_size — but it used to be measured with len() on the decoded
    text, i.e. characters. A multi-byte frame was therefore up to 4x the intended
    ceiling, and create_app used without uvicorn's framing limit (tests, embedding)
    had no second line of defence at all. The close CODE is what proves which check
    fired: 1009 is the size guard, 4000 would mean the frame was accepted and only
    then failed auth."""
    _matchmake(client, uuid=ALICE)
    body = _matchmake(client, uuid=BOB).json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(_fat_multibyte_frame(type="auth", session_token="bogus"))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == WS_CLOSE_PAYLOAD_TOO_LARGE


def test_ws_handshake_under_the_byte_cap_reaches_the_token_check(client):
    """The control for the size guard: an ordinary bad-token handshake must still
    be refused as an invalid token, not as an oversize frame."""
    _matchmake(client, uuid=ALICE)
    body = _matchmake(client, uuid=BOB).json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "auth",
                                 "session_token": "bogus"}))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == WS_CLOSE_INVALID_TOKEN


def test_in_session_frame_over_the_byte_cap_closes_as_too_large(client):
    """The same byte budget guards every later frame, not just the handshake."""
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, side="white").json()
    b = _matchmake(client, uuid=BOB, side="black").json()
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(_fat_multibyte_frame(type="move", **{"from": "e2", "to": "e4"}))
            with pytest.raises(WebSocketDisconnect) as exc:
                ws_w.receive_text()
    assert exc.value.code == WS_CLOSE_PAYLOAD_TOO_LARGE


def test_ws_rejects_version_mismatch_auth(client):
    """An old client (one protocol version behind) is refused with a
    version_mismatch error before the socket closes."""
    _matchmake(client, uuid=ALICE)
    r2 = _matchmake(client, uuid=BOB)
    body = r2.json()
    with client.websocket_connect(f"/ws/{body['room_id']}") as ws:
        ws.send_text(json.dumps({"version": PROTOCOL_VERSION - 1, "type": "auth",
                                  "session_token": body["session_token"]}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["reason"] == Reason.VERSION_MISMATCH
        with pytest.raises(Exception):
            ws.receive_text()


def test_two_clients_pair_and_get_game_start(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    a = r1.json()
    b = r2.json()
    assert a["room_id"] == b["room_id"]
    with client.websocket_connect(f"/ws/{a['room_id']}") as ws_a:
        ws_a.send_text(json.dumps(auth_msg(a["session_token"])))
        with client.websocket_connect(f"/ws/{b['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
            msg_a = json.loads(ws_a.receive_text())
            msg_b = json.loads(ws_b.receive_text())
            assert msg_a["type"] == "game_start"
            assert msg_b["type"] == "game_start"
            assert msg_a["your_color"] == "white"
            assert msg_b["your_color"] == "black"
            assert msg_a["white_name"] == "Alice"
            assert msg_a["black_name"] == "Alice"
            assert "started_seconds_ago" in msg_a
            assert msg_a["started_seconds_ago"] == pytest.approx(0.0, abs=1.0)


def test_full_short_game_e4_e5_resign(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                        "from": "e2", "to": "e4"}))
            applied_w = json.loads(ws_w.receive_text())
            applied_b = json.loads(ws_b.receive_text())
            assert applied_w["type"] == "move_applied"
            assert applied_w["san"] == "e4"
            assert applied_b["from"] == "e2"
            assert json.loads(ws_w.receive_text())["type"] == "idle_window"
            assert json.loads(ws_b.receive_text())["type"] == "idle_window"
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                        "from": "e7", "to": "e5"}))
            ws_w.receive_text()
            ws_b.receive_text()
            assert json.loads(ws_w.receive_text())["type"] == "idle_window"
            assert json.loads(ws_b.receive_text())["type"] == "idle_window"
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "resign"}))
            res_w = json.loads(ws_w.receive_text())
            ws_b.receive_text()
            assert res_w["type"] == "result"
            assert res_w["reason"] == Reason.RESIGNATION
            assert res_w["winner_color"] == "black"


def _recv(ws):
    return json.loads(ws.receive_text())


def test_rematch_round_trip_swaps_colors_and_relays_move(client):
    """End-to-end over real websockets: pair, finish a game, offer + accept a
    rematch, and confirm colours swap and the first move of the new game flows."""
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            _recv(ws_w)
            _recv(ws_b)
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "resign"}))
            assert _recv(ws_w)["type"] == "result"
            assert _recv(ws_b)["type"] == "result"
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "rematch_request"}))
            assert _recv(ws_w)["type"] == "rematch_request"
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "rematch_response", "accept": True}))
            gs_w = _recv(ws_w)
            gs_b = _recv(ws_b)
            assert gs_w["type"] == "game_start" and gs_w["rematch"] is True
            assert gs_w["your_color"] == "black"
            assert gs_b["your_color"] == "white"
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                        "from": "e2", "to": "e4"}))
            ap_b = _recv(ws_b)
            ap_w = _recv(ws_w)
            assert ap_b["type"] == "move_applied" and ap_b["san"] == "e4"
            assert ap_w["from"] == "e2" and ap_w["to"] == "e4"


def test_rematch_decline_notifies_offerer(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            _recv(ws_w)
            _recv(ws_b)
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "resign"}))
            _recv(ws_w)
            _recv(ws_b)
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "rematch_request"}))
            assert _recv(ws_b)["type"] == "rematch_request"
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "rematch_response", "accept": False}))
            upd = _recv(ws_w)
            assert upd["type"] == "rematch_update" and upd["event"] == "declined"


def _auth(ws, body):
    ws.send_text(json.dumps(auth_msg(body["session_token"])))


def test_a_rematch_reroutes_heartbeats_and_the_teardown_to_the_new_color(
    app, client, clock,
):
    """A rematch swaps the colours underneath two sockets that never dropped, so
    the session loop re-reads `room.color_of(uuid)` on every frame instead of
    trusting the colour it authenticated with. Both consumers of that re-read are
    pinned here: the per-frame `touch_seen` (which is what keeps the heartbeat
    timeout off a live player) and the teardown's `mark_disconnected`.

    Alice authenticates as white and ends the rematch as black; if either site
    still used the auth-time colour, Bob's seat would be the one stamped -- and
    Alice would be swept as silent while typing.
    """
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    room = app.state.rooms.get(r1.json()["room_id"])
    with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
        _auth(ws_b, r2.json())
        with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
            _auth(ws_w, r1.json())
            assert _recv(ws_w)["type"] == "game_start"
            assert _recv(ws_b)["type"] == "game_start"
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "resign"}))
            _recv(ws_w)
            _recv(ws_b)
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "rematch_request"}))
            assert _recv(ws_w)["type"] == "rematch_request"
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "rematch_response", "accept": True}))
            assert _recv(ws_w)["your_color"] == "black"
            assert _recv(ws_b)["your_color"] == "white"
            assert room.black.client_uuid == ALICE, "alice now sits in the black seat"

            clock.advance(5)
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                       "type": "ping", "ply": 0}))
            assert _recv(ws_w)["type"] == "pong"

            assert room.black.last_seen == pytest.approx(5.0), \
                "alice's heartbeat stamps the seat she holds NOW"
            assert room.white.last_seen == pytest.approx(0.0), \
                "and never bob's, whose colour she merely authenticated as"

        notice = _recv(ws_b)
        assert notice["type"] == "connection_status"
        assert notice["opp_state"] == "reconnecting"
        assert room.black.connected is False, "the teardown marks the seat she holds NOW"
        assert room.white.connected is True, "bob's own socket is still up"


def test_a_socket_whose_seat_vanishes_falls_back_to_its_auth_color(app, client, caplog):
    """The `or auth_color` fallback in the session teardown. `color_of` answers
    None once the room no longer seats that player, and the teardown still has to
    mark, log and notify -- a KeyError or a `None` colour here would strand the
    opponent on a board that never says "reconnecting".

    Emptying the seat under a live socket is the shape that reaches it: the next
    frame breaks the loop with no current colour, and the auth-time colour is the
    only identity left to report."""
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    room = app.state.rooms.get(r1.json()["room_id"])
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            _auth(ws_b, r2.json())
            with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
                _auth(ws_w, r1.json())
                assert _recv(ws_w)["type"] == "game_start"
                assert _recv(ws_b)["type"] == "game_start"
                room.white = None
                assert room.color_of(ALICE) is None
                ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                           "type": "ping", "ply": 0}))
            notice = _recv(ws_b)

    assert notice["type"] == "connection_status"
    assert notice["opp_state"] == "reconnecting"
    assert f"ws disconnected room={room.room_id} color=white" in [
        r.getMessage() for r in caplog.records], \
        "the breadcrumb names the colour the socket authenticated as"


def test_out_of_turn_move_rejected(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                        "from": "e7", "to": "e5"}))
            err = json.loads(ws_b.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == Reason.NOT_YOUR_TURN
            assert err["msg_type"] == "move"


def test_draw_offer_allowed_off_turn(client):
    """Draws may be offered at any moment, regardless of whose turn it is: black
    (not on move at the start) offers and white receives the draw_offered relay."""
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "draw_offer"}))
            msg = json.loads(ws_w.receive_text())
            assert msg["type"] == "draw_offered"


def test_takeback_request_off_turn_tags_msg_type(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "takeback_request"}))
            err = json.loads(ws_w.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == Reason.NOT_YOUR_TURN
            assert err["msg_type"] == "takeback_request"


def test_takeback_request_with_no_moves_played_rejected_with_reason(client):
    """Black (not on move at the start) requests a takeback before any move has
    landed -- there is nothing to undo, and the client-mapped reason must ride
    along instead of a silent no-op."""
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_b.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                        "type": "takeback_request"}))
            err = json.loads(ws_b.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == Reason.NO_TAKEBACK_AVAILABLE
            assert err["msg_type"] == "takeback_request"


def test_invalid_move_format_rejected(client):
    random.seed(0)
    r1 = _matchmake(client, uuid=ALICE, side="white")
    r2 = _matchmake(client, uuid=BOB, side="black")
    with client.websocket_connect(f"/ws/{r1.json()['room_id']}") as ws_w:
        ws_w.send_text(json.dumps(auth_msg(r1.json()["session_token"])))
        with client.websocket_connect(f"/ws/{r2.json()['room_id']}") as ws_b:
            ws_b.send_text(json.dumps(auth_msg(r2.json()["session_token"])))
            ws_w.receive_text()
            ws_b.receive_text()
            ws_w.send_text(json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                                        "from": "z9", "to": "a1"}))
            err = json.loads(ws_w.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == Reason.INVALID_MOVE_FORMAT


@pytest.mark.asyncio
async def test_first_move_timeout_aborts_room(app, clock):
    random.seed(0)
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    clock.advance(FIRST_MOVE_ABORT_SECONDS + 1)
    await _sweep(app)
    assert room.result == ("aborted", None)


@pytest.mark.asyncio
async def test_clock_flag_during_play_broadcasts_timeout(app, clock):
    random.seed(0)
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=1, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=1, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 1
    clock.advance(70)
    await _sweep(app)
    assert room.result is not None
    assert room.result[0] == Reason.TIMEOUT


@pytest.mark.asyncio
async def _paired_in_progress_room(rooms, clock):
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    room.first_move_at = clock()
    room.white.connected = True
    room.black.connected = True
    return room


async def _resync_room(app, clock):
    """A room whose game has really been announced, with the clock parked past
    the heartbeat transit grace. Both matter: an unannounced room refuses to
    judge a heartbeat at all, and the game-start stamp excuses every ply until
    the grace runs out."""
    room = await _paired_in_progress_room(app.state.rooms, clock)
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    await broadcast_game_start(app.state.connections, room, clock)
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    return room, ws_w, ws_b


async def _lagging_spell(app, room, ws, color, ply=7):
    """One lagging spell as the server now defines it: a mismatch that survives
    RESYNC_STABLE_MISMATCH_HEARTBEATS judged heartbeats."""
    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
        await handle_ping(app, ws, room, color, _ping_raw(ply))


def _ping_raw(ply):
    return json.dumps({"version": PROTOCOL_VERSION, "type": "ping", "ply": ply})


def _opp_states(ws):
    return [m["opp_state"] for m in ws.of_type("connection_status")]


async def test_a_lagging_spell_after_a_recovery_notifies_the_opponent_again(app, clock):
    """REGRESSION: the notify gate was a flat 5s interval keyed on (room, color),
    and nothing reset it when the player caught up. A second genuine lagging spell
    starting inside that window was therefore silently unannounced — the opponent's
    strip read 'connected' while resync directives were still being pushed at the
    lagging client. Recovering now re-arms the gate down to the flap floor, so the
    next real spell is announced. Each spell is now a PAIR of mismatching
    heartbeats, because a single one is tolerated as a race."""
    room, ws_w, ws_b = await _resync_room(app, clock)
    opened_at = clock()
    await _lagging_spell(app, room, ws_w, "white")
    await handle_ping(app, ws_w, room, "white", _ping_raw(0))
    clock.advance(RESYNC_NOTIFY_FLAP_FLOOR_SECONDS + 0.1)

    await _lagging_spell(app, room, ws_w, "white")

    assert _opp_states(ws_b) == ["resyncing", "connected", "resyncing"]
    assert clock() - opened_at < RESYNC_NOTIFY_MIN_INTERVAL_SECONDS, \
        "and well inside the interval that used to swallow it"


async def test_a_ply_flap_inside_the_floor_still_notifies_only_once(app, clock):
    """The reason the gate exists at all: a client whose reported ply oscillates
    would otherwise toggle the opponent's connection strip at heartbeat rate. The
    re-arm above must not reopen that door — clearing only drops the wait to the
    flap floor, and a same-instant flap never gets past it. The flap is driven
    in pairs so it clears the strike counter; a pure alternation never even
    reaches the gate, which is asserted separately."""
    room, ws_w, ws_b = await _resync_room(app, clock)
    for _ in range(4):
        await _lagging_spell(app, room, ws_w, "white")
        await handle_ping(app, ws_w, room, "white", _ping_raw(0))
    assert _opp_states(ws_b) == ["resyncing", "connected"]


async def test_a_pure_ply_alternation_never_reaches_the_notify_gate(app, clock):
    """The strike counter sits IN FRONT of the debounce, so the cheapest form of
    the griefing pattern -- alternate a wrong ply with a right one at the socket
    rate -- now costs the opponent nothing at all: every wrong ply is a first
    strike and every right one wipes it."""
    room, ws_w, ws_b = await _resync_room(app, clock)
    for i in range(8):
        await handle_ping(app, ws_w, room, "white", _ping_raw(7 if i % 2 == 0 else 0))

    assert _opp_states(ws_b) == []
    assert ws_w.of_type("resync_directive") == []


def _fill_gate(gate, count, now, interval=RESYNC_NOTIFY_MIN_INTERVAL_SECONDS):
    for i in range(count):
        assert gate.allow((f"room-{i}", "white", RESYNC_NOTIFY), now, interval)


def test_resync_gate_prunes_expired_keys_and_keeps_live_ones():
    """The gate is keyed per (room, color, tag) and rooms churn, so without the
    prune it is an unbounded dict that only ever grows on a busy server. The prune
    fires on insertion past the threshold and must drop exactly the keys whose
    window has already passed — a key still inside its window survives, or the
    prune itself becomes a way to bypass the interval."""
    gate = _ResyncGate()
    live = ("live-room", "white", RESYNC_DIRECTIVE)
    assert gate.allow(live, 4.9, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS)
    _fill_gate(gate, RESYNC_GATE_PRUNE_THRESHOLD - 1, 0.0)
    assert len(gate._open_at) == RESYNC_GATE_PRUNE_THRESHOLD

    fresh = ("fresh-room", "white", RESYNC_NOTIFY)
    assert gate.allow(fresh, 6.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS)

    assert set(gate._open_at) == {live, fresh}, \
        "the 511 windows that closed at 5.0 went; the one open until 9.9 stayed"
    assert gate.allow(live, 6.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS) is False, \
        "the survivor still holds its remaining window"
    assert gate.allow(fresh, 6.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS) is False, \
        "and the key inserted through the prune gates like any other"
    assert gate.allow(fresh, 11.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS) is True


def test_resync_gate_reopen_never_delays_an_already_open_window():
    """reopen() lowers the wait, it never raises it: a player who recovers long
    after the last notification must not have a fresh cooldown imposed on them."""
    gate = _ResyncGate()
    key = ("room", "white", RESYNC_NOTIFY)
    assert gate.allow(key, 0.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS)
    gate.reopen(key, 100.0, RESYNC_NOTIFY_FLAP_FLOOR_SECONDS)
    assert gate.allow(key, 100.0, RESYNC_NOTIFY_MIN_INTERVAL_SECONDS) is True
    gate.reopen(("absent", "white", RESYNC_NOTIFY), 0.0, RESYNC_NOTIFY_FLAP_FLOOR_SECONDS)
    assert "absent" not in [k[0] for k in gate._open_at], "reopen never mints a key"


async def test_grace_expiry_without_desync_awards_opponent(app, clock):
    """A plain disconnect (no desync signalled) that never recovers is a deliberate
    leave: the waiting player wins by abandonment. Sits at ply 3 — the first ply
    with no idle window armed — because at plies 1-2 the idle window deliberately
    expires first in the same step_all pass (pinned in test_server_sweep)."""
    rooms = app.state.rooms
    room = await _paired_in_progress_room(rooms, clock)
    room.plies_ever = 3
    room.idle_since = None
    rooms.mark_disconnected(room.room_id, "white")
    clock.advance(GRACE_SECONDS + 1)
    await _sweep(app)
    assert room.result == (Reason.ABANDONMENT, "black")


async def test_grace_expiry_with_desync_still_awards_opponent(app, clock):
    """REGRESSION (v2.10.0 live smoke): desync_active is a sticky flag -- any
    /resume sets it and only the player's NEXT applied move clears it. The old
    desync branch aborted the game with no winner, so a deliberate leave right
    after a resync robbed the stayer of the abandonment win. Rule: with moves
    played, a grace expiry ALWAYS awards the opponent; only zero-ply games
    convert to aborted (finalize_result's central guard). Ply 3 for the same
    reason as the test above: below it the idle window wins the step_all pass."""
    rooms = app.state.rooms
    room = await _paired_in_progress_room(rooms, clock)
    room.plies_ever = 3
    room.idle_since = None
    room.white.desync_active = True
    rooms.mark_disconnected(room.room_id, "white")
    clock.advance(GRACE_SECONDS + 1)
    await _sweep(app)
    assert room.result == (Reason.ABANDONMENT, "black")


async def test_grace_expiry_with_desync_at_zero_plies_aborts(app, clock):
    rooms = app.state.rooms
    room = await _paired_in_progress_room(rooms, clock)
    room.white.desync_active = True
    rooms.mark_disconnected(room.room_id, "white")
    clock.advance(GRACE_SECONDS + 1)
    await _sweep(app)
    assert room.result == (Reason.ABORTED, None)


@pytest.mark.asyncio
async def test_resume_ticks_clock_before_snapshotting(app, client, clock):
    """/resume ticks the clock before snapshotting, so its reply reflects elapsed
    time as of the request rather than the last sweep tick (a stale snapshot would
    return the pre-advance white_remaining)."""
    random.seed(0)
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    room.first_move_at = clock()
    initial_white = room.backend.clock.white_remaining
    clock.advance(7)
    r = client.post("/resume", json={
        "version": PROTOCOL_VERSION,
        "room_id": room.room_id,
        "session_token": "ta",
    })
    assert r.status_code == 200
    snap = r.json()["clock"]
    assert snap["white_remaining"] == pytest.approx(initial_white - 7, abs=0.01)
    assert snap["running_for"] == "white"


async def _resumable_room(app, clock):
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    room.first_move_at = clock()
    return room


def _resume(client, room, token="ta"):
    return client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id, "session_token": token,
    })


@pytest.mark.asyncio
async def test_resume_reports_the_armed_idle_window(app, client, clock):
    room = await _resumable_room(app, clock)
    room.plies_ever = 2
    room.idle_since = clock()
    clock.advance(12)

    r = _resume(client, room)

    assert r.status_code == 200
    window = r.json()["idle_window"]
    assert window["outcome"] == "resignation"
    assert window["color"] == "white"
    assert window["seconds_remaining"] == pytest.approx(IDLE_RESIGN_SECONDS - 12)


@pytest.mark.asyncio
async def test_resume_reports_no_window_after_ply_three(app, client, clock):
    room = await _resumable_room(app, clock)
    room.plies_ever = 3
    room.idle_since = None

    r = _resume(client, room)

    assert r.status_code == 200
    assert r.json()["idle_window"] is None


@pytest.mark.asyncio
async def test_idle_window_wire_is_none_for_a_backend_less_room(app, clock):
    """color_to_move() is None on a queued/unpaired room (no backend), and
    IdleWindowWire(color=None) would raise — turning /resume into a 500 for
    any shape where idle_since is set without a paired backend. The wire
    builder bails to None instead."""
    room = await app.state.rooms.enqueue(
        client_uuid=ALICE, nickname="A", session_token="ta",
        time_minutes=5, increment_seconds=0, side_preference="white")
    assert room.backend is None
    room.idle_since = clock()

    assert idle_window_wire(room, clock()) is None


@pytest.mark.asyncio
async def test_resume_does_not_reset_the_idle_window(app, client, clock):
    """The anti-dodge posture shared with expired skill-check pendings: a resume
    is not deliberate presence, and restamping idle_since here would hand the
    idler a trivial disconnect/resume stall loop. /resume only REPORTS the
    remaining time so the reconnected client renders an honest badge."""
    room = await _resumable_room(app, clock)
    room.plies_ever = 2
    armed_at = clock()
    room.idle_since = armed_at
    clock.advance(30)

    r = _resume(client, room)

    assert r.status_code == 200
    assert room.idle_since == armed_at
    assert r.json()["idle_window"]["seconds_remaining"] == pytest.approx(
        IDLE_RESIGN_SECONDS - 30)


class _FakeOldSocket:
    def __init__(self):
        self.closed_with = None

    async def close(self, code=1000):
        self.closed_with = code


@pytest.mark.asyncio
async def test_reclaim_closes_a_still_registered_old_socket(app, client):
    """A reclaim while the previous socket is still registered must revoke it
    (mirrors the WS_CLOSE_SUPERSEDED pattern already used when a fresh
    connection displaces an old one) -- otherwise the stale socket keeps
    working on the rotated-out session token."""
    rooms = app.state.rooms
    connections = app.state.connections
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    room = await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                               time_minutes=5, increment_seconds=0, side_preference="black")
    old_ws = _FakeOldSocket()
    connections.add(room.room_id, ALICE, old_ws)

    r = client.post("/reclaim", json={"version": PROTOCOL_VERSION, "client_uuid": ALICE})

    assert r.status_code == 200
    assert r.json()["session_token"] != "ta"
    assert old_ws.closed_with == WS_CLOSE_SUPERSEDED


def _run_server_main(monkeypatch, argv):
    captured = {}

    def _fake_run(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs

    monkeypatch.setenv("CHESS_MAX_ROOMS", "8")
    monkeypatch.setattr(server_main.uvicorn, "run", _fake_run)
    monkeypatch.setattr(server_main.logging_setup, "configure", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", argv)
    server_main._main()
    return captured


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["chess-server", "--max-rooms", "8"], id="serve"),
        pytest.param(["chess-server", "--max-rooms", "8", "--reload"], id="reload"),
    ],
)
def test_runner_caps_inbound_websocket_frames_at_the_protocol_layer(monkeypatch, argv):
    """SECURITY: uvicorn's ws_max_size defaults to 16 MiB, so an unauthenticated
    socket could buffer a 16 MB frame in memory before _authenticate_ws ever got
    to measure it -- N of those OOM the container inside the auth window. The
    runner now hands the websockets impl the same ceiling the app enforces, so
    oversize frames die during framing instead of after buffering. Both entry
    paths (plain serve and --reload, which goes through the app factory) must
    carry it: a dev-only gap is still a shipped gap."""
    kwargs = _run_server_main(monkeypatch, argv)["kwargs"]
    assert kwargs["ws_max_size"] == MAX_INBOUND_MESSAGE_BYTES
    assert kwargs["ws_ping_interval"] is None


def test_runner_app_factory_builds_a_real_app(monkeypatch):
    """--reload hands uvicorn an import string plus factory=True; the factory has
    to stay importable and return a working app or the reload path dies at boot."""
    kwargs = _run_server_main(monkeypatch, ["chess-server", "--reload"])["kwargs"]
    assert kwargs["factory"] is True
    monkeypatch.setenv("CHESS_MAX_ROOMS", "8")
    built = server_main._app_factory()
    assert built.state.rooms is not None


@pytest.mark.asyncio
async def test_reclaim_with_no_live_socket_is_a_clean_noop(app, client):
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")

    r = client.post("/reclaim", json={"version": PROTOCOL_VERSION, "client_uuid": ALICE})

    assert r.status_code == 200


@pytest.mark.asyncio
async def test_the_housekeeping_loop_keeps_ticking_after_a_pass_raises(
        app, monkeypatch, allow_sweep_failures):
    """The loop used to die on the first exception that escaped a pass, and
    /healthz went on answering 200: clocks, grace and queue reaping stopped for
    the life of the process with nothing to show for it. Now the failure is
    recorded and the next tick still runs."""
    sweep = app.state.sweep
    passes = []

    async def _boom():
        passes.append("tick")
        raise RuntimeError("whole pass exploded")

    monkeypatch.setattr(sweep, "step_all", _boom)
    monkeypatch.setattr(app_module, "CLOCK_TICK_INTERVAL_SECONDS", 0)

    task = asyncio.create_task(_sweep_loop(app))
    for _ in range(500):
        if len(passes) >= 3:
            break
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait([task], timeout=1)

    assert len(passes) >= 3, "one failed pass must not end the loop"
    assert sweep.failure_count >= 3
    assert isinstance(sweep._last_failure, RuntimeError)


@pytest.mark.asyncio
async def test_the_housekeeping_loop_cancels_cleanly_at_shutdown(app, monkeypatch):
    """Shutdown cancels the task; the loop has to absorb that one cancellation
    and return, rather than let it surface as an error at the end of a normal
    stop."""
    monkeypatch.setattr(app_module, "CLOCK_TICK_INTERVAL_SECONDS", 0)
    task = asyncio.create_task(_sweep_loop(app))
    for _ in range(100):
        if app.state.sweep._running:
            break
        await asyncio.sleep(0)
    assert app.state.sweep._running, "the loop marks itself running as it starts"

    task.cancel()
    await asyncio.wait([task], timeout=1)

    assert task.done()
    assert not task.cancelled(), "the loop swallows its own cancellation"
    assert task.exception() is None


def test_healthz_reports_ok_with_a_housekeeping_age(client):
    """The two fields an operator and the in-game server picker read. An app
    whose loop never started reports age zero -- there is nothing to be late
    for before the first tick."""
    body = client.get("/healthz").json()
    assert body["status"] == HealthStatus.OK
    assert body["housekeeping_age_s"] == 0.0


def test_healthz_reports_full_once_the_room_cap_is_reached(app, client):
    """`full` is computed off exactly the predicate that refuses matchmaking, so
    a player told the server is full can confirm it, and a monitor sees the same
    thing. Filled with a distinct uuid and time control per request so nothing
    pairs and every slot counts."""
    max_rooms = app.state.rooms._max_rooms
    for i in range(max_rooms - 1):
        assert _matchmake(client, uuid=fake_uuid4(200 + i), nickname=f"N{i}",
                          time=i + 1).status_code == 200
    assert client.get("/healthz").json()["status"] == HealthStatus.OK

    assert _matchmake(client, uuid=ZED, nickname="Z", time=100).status_code == 200
    r = client.get("/healthz")
    assert r.status_code == 200, "a full server still answers 200"
    assert r.json()["status"] == HealthStatus.FULL
    assert _matchmake(client, uuid=CARL, nickname="C", time=120).status_code == 503


@pytest.mark.parametrize(
    "age, expected",
    [
        pytest.param(SWEEP_STALE_SECONDS, HealthStatus.OK, id="exactly_at_the_limit"),
        pytest.param(SWEEP_STALE_SECONDS + 0.001, HealthStatus.DEGRADED,
                     id="a_hair_past_the_limit"),
    ],
)
def test_healthz_turns_degraded_only_past_the_staleness_limit(
        app, client, clock, age, expected):
    """The boundary itself, both sides of it. The limit is loose so one slow
    socket send inside a walk cannot flip a server's health, which only works if
    `at the limit` still counts as healthy."""
    app.state.sweep.mark_running()
    clock.advance(age)
    r = client.get("/healthz")
    assert r.status_code == 200, "a degraded server still answers 200"
    body = r.json()
    assert body["status"] == expected
    assert body["housekeeping_age_s"] == pytest.approx(age)


def test_a_full_server_reads_full_even_while_it_is_also_behind(app, client, clock):
    """Only one status fits in the field, so the order is fixed: no capacity is
    the thing a waiting player is actually hitting, and it wins."""
    app.state.sweep.mark_running()
    clock.advance(SWEEP_STALE_SECONDS + 10)
    for i in range(app.state.rooms._max_rooms):
        assert _matchmake(client, uuid=fake_uuid4(200 + i), nickname=f"N{i}",
                          time=i + 1).status_code == 200
    assert app.state.sweep.is_stale is True
    assert client.get("/healthz").json()["status"] == HealthStatus.FULL


def test_startup_logs_the_tuning_values_once(app, caplog):
    """Every knob an operator can move, printed where they can read back what
    the process actually resolved -- an env typo or a clamp is otherwise
    invisible until a timeout misbehaves in production."""
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        with TestClient(app):
            pass
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("tuning ")]
    assert len(lines) == 1, f"one tuning line per startup, got {lines}"
    line = lines[0]
    assert f"grace={GRACE_SECONDS:.1f}" in line
    assert f"heartbeat={HEARTBEAT_INTERVAL_SECONDS:.1f}" in line
    assert f"miss_limit={HEARTBEAT_MISS_LIMIT:d}" in line
    assert f"heartbeat_timeout={HEARTBEAT_TIMEOUT_SECONDS:.1f}" in line
    assert f"tick={app_module.CLOCK_TICK_INTERVAL_SECONDS:.2f}" in line
    assert f"sweep_stale={SWEEP_STALE_SECONDS:.1f}" in line
    assert f"transit_grace={RESYNC_TRANSIT_GRACE_SECONDS:.2f}" in line
    assert f"stable_heartbeats={RESYNC_STABLE_MISMATCH_HEARTBEATS:d}" in line
