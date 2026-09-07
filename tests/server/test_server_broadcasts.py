"""broadcasts.py + connections.py: the finalize-race guard (a resign and a
mate landing at the same instant must not double-broadcast a contradictory
result) and the broadcast-send-failure -> mark_disconnected seam (a socket
that throws on send must not be left looking "connected" until the next
heartbeat sweep notices, ~HEARTBEAT_TIMEOUT_SECONDS later).

Plus the static half of the same invariant: finalize_and_broadcast is the ONE
place a result is written, and the AST guard at the bottom is what keeps it
that way.
"""
import ast
import logging
import os

import pytest

import chessshootout
from chessshootout.server.broadcasts import finalize_and_broadcast, push_idle_window
from chessshootout.server.connections import broadcast
from chessshootout.server.protocol import PROTOCOL_VERSION, Reason, ResultMessage
from chessshootout.server.rooms import RoomManager
from tests.helpers import read_source_without_docstrings
from tests.server.conftest import ALICE, RecordingWS, pair_room

PACKAGE_ROOT = os.path.dirname(os.path.abspath(chessshootout.__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
SERVER_ROOT = os.path.join(PACKAGE_ROOT, "server")
FINALIZE_CALLER = "chessshootout/server/broadcasts.py"


class FailingWS:
    async def send_json(self, payload):
        raise RuntimeError("connection reset")


class _RacingWS(RecordingWS):
    """A socket that, on receiving the winning caller's own result broadcast,
    simulates a second concurrent finalize (e.g. a mate landing) racing in
    with a contradictory reason -- exactly the scenario two truly concurrent
    handler tasks would produce."""

    def __init__(self, rooms, connections, room):
        super().__init__()
        self._rooms = rooms
        self._connections = connections
        self._room = room
        self._fired = False

    async def send_json(self, payload):
        await super().send_json(payload)
        if not self._fired and payload.get("type") == "result":
            self._fired = True
            await finalize_and_broadcast(self._rooms, self._connections, self._room,
                                         Reason.CHECKMATE, winner_color="white")


@pytest.mark.asyncio
async def test_losing_finalize_race_does_not_double_broadcast(app, clock):
    rooms = app.state.rooms
    connections = app.state.connections
    room = await pair_room(rooms)
    room.first_move_at = clock()
    racing_white = _RacingWS(rooms, connections, room)
    ws_black = RecordingWS()
    connections.add(room.room_id, room.white.client_uuid, racing_white)
    connections.add(room.room_id, room.black.client_uuid, ws_black)

    await finalize_and_broadcast(rooms, connections, room, Reason.RESIGNATION,
                                 winner_color="black")

    assert room.result == (Reason.RESIGNATION, "black"), "the first finalize wins the state"
    expected = [{
        "version": PROTOCOL_VERSION, "type": "result",
        "reason": Reason.RESIGNATION, "winner_color": "black",
    }]
    assert racing_white.of_type("result") == expected, \
        "the racing (losing) finalize must not re-broadcast its own contradictory reason"
    assert ws_black.of_type("result") == expected, "both sides see one consistent result"


@pytest.mark.asyncio
async def test_broadcast_marks_disconnected_on_send_failure(app, clock):
    rooms = app.state.rooms
    connections = app.state.connections
    room = await pair_room(rooms)
    room.first_move_at = clock()
    rooms.mark_connected(room.room_id, "white")
    rooms.mark_connected(room.room_id, "black")
    connections.add(room.room_id, room.white.client_uuid, FailingWS())
    ws_black = RecordingWS()
    connections.add(room.room_id, room.black.client_uuid, ws_black)

    await broadcast(rooms, connections, room,
                    ResultMessage(reason=Reason.RESIGNATION, winner_color="black"))

    assert room.white.connected is False, "a failed send must flip the slot to disconnected"
    assert room.white.disconnected_at is not None
    opp_notice = ws_black.of_type("connection_status")
    assert opp_notice and opp_notice[-1]["opp_state"] == "reconnecting"


@pytest.mark.asyncio
async def test_broadcast_send_failure_to_an_already_disconnected_slot_is_a_noop(app, clock):
    """mark_disconnected is idempotent, so a send failure against a slot that
    was never marked connected (or already disconnected) doesn't stamp a
    fresh disconnected_at over the real one."""
    rooms = app.state.rooms
    connections = app.state.connections
    room = await pair_room(rooms)
    room.first_move_at = clock()
    connections.add(room.room_id, room.white.client_uuid, FailingWS())

    await broadcast(rooms, connections, room,
                    ResultMessage(reason=Reason.RESIGNATION, winner_color="black"))

    assert room.white.connected is False
    assert room.white.disconnected_at is None, \
        "the guard leaves the never-connected slot's timer untouched (no fresh stamp)"


@pytest.mark.asyncio
async def test_push_idle_window_bails_on_a_backend_less_room(app, clock):
    """A queued room has no backend, so color_to_move() is None — a push
    against that shape (idle_since set but nobody paired) must bail up front
    instead of building IdleWindowMessage(color=None) and raising mid-
    broadcast. The untouched idle_pushed_at proves the early return."""
    rooms = app.state.rooms
    room = await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                               time_minutes=5, increment_seconds=0,
                               side_preference="white")
    assert room.backend is None
    room.idle_since = clock()

    await push_idle_window(rooms, app.state.connections, room, clock(), force=True)

    assert room.idle_pushed_at is None


def _finalize_result_call_lines(path):
    tree = ast.parse(read_source_without_docstrings(path), filename=path)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "finalize_result":
            lines.append(node.lineno)
    return sorted(lines)


def test_finalize_result_is_called_from_broadcasts_and_nowhere_else():
    """RoomManager.finalize_result writes room.result and awards the series
    point, and it is deliberately NOT the public way to end a game:
    finalize_and_broadcast wraps it, and only broadcasts when its own call was
    the one that applied (the losing side of the race above stays silent).

    A second caller anywhere under server/ would be a result that lands on the
    room without ever reaching the players — the exact shape of the race this
    file exists for. AST rather than a text scan, so `finalize_result` in prose
    or in the `def` at rooms.py cannot register as a call.
    """
    assert callable(RoomManager.finalize_result), \
        "guard is keyed on the name; a rename must be made loud, not silent"
    assert os.path.isdir(SERVER_ROOT), f"expected a walkable package dir at {SERVER_ROOT}"
    callers = {}
    scanned = 0
    for dirpath, _, filenames in os.walk(SERVER_ROOT):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            scanned += 1
            lines = _finalize_result_call_lines(path)
            if lines:
                callers[os.path.relpath(path, REPO_ROOT)] = lines
    assert scanned >= 8, f"only scanned {scanned} files, guard root is likely wrong"
    assert sorted(callers) == [FINALIZE_CALLER], \
        f"finalize_result must be called only from broadcasts.py, found {callers}"
    assert len(callers[FINALIZE_CALLER]) == 1, \
        f"and exactly once inside it, found {callers[FINALIZE_CALLER]}"


FINALIZE_PREFIX = "game finalized"


def _finalize_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith(FINALIZE_PREFIX)]


@pytest.mark.asyncio
async def test_one_game_finalized_line_per_game_however_often_finalize_is_called(
    app, clock, caplog,
):
    """The funnel line sits INSIDE the `applied` guard, so it inherits the
    once-per-game property from the result itself. Outside the guard, every
    late resign, sweep tick or reconnect racing an already-ended game would
    file a fresh ending in the journal."""
    rooms = app.state.rooms
    room = await pair_room(rooms)
    room.first_move_at = clock()
    room.plies_ever = 3
    connections = app.state.connections
    connections.add(room.room_id, room.white.client_uuid, RecordingWS())

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        for _ in range(3):
            await finalize_and_broadcast(rooms, connections, room,
                                         Reason.RESIGNATION, winner_color="black")

    assert _finalize_lines(caplog) == [
        f"game finalized room={room.room_id} reason={Reason.RESIGNATION} "
        f"winner=black plies=3 duration_s=0.0"
    ]


@pytest.mark.asyncio
async def test_the_finalize_that_loses_the_race_logs_nothing(app, clock, caplog):
    """Same race as the double-broadcast test above, read from the journal: the
    contradictory reason the losing caller carried must not reach the log any
    more than it reaches the players."""
    rooms = app.state.rooms
    connections = app.state.connections
    room = await pair_room(rooms)
    room.first_move_at = clock()
    room.plies_ever = 5
    connections.add(room.room_id, room.white.client_uuid,
                    _RacingWS(rooms, connections, room))

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        await finalize_and_broadcast(rooms, connections, room,
                                     Reason.RESIGNATION, winner_color="black")

    lines = _finalize_lines(caplog)
    assert len(lines) == 1
    assert Reason.CHECKMATE not in lines[0], "the losing reason must not be logged"


@pytest.mark.asyncio
async def test_finalizing_a_room_that_never_started_logs_nothing(app, caplog):
    """A room still waiting in the queue is not in `_active`, so finalize_result
    refuses it. Nothing happened to that game, so nothing is reported -- and the
    line's `duration_s` may never be computed off the None `started_at` such a
    room could carry."""
    rooms = app.state.rooms
    room = await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                               time_minutes=5, increment_seconds=0,
                               side_preference="white")

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        await finalize_and_broadcast(rooms, app.state.connections, room,
                                     Reason.ABANDONMENT, winner_color="black")

    assert room.result is None
    assert _finalize_lines(caplog) == []


@pytest.mark.asyncio
async def test_a_zero_ply_abort_logs_the_rewritten_reason_and_the_wait(
    app, clock, caplog,
):
    """`plies` and `reason` come from the STORED result, not from the arguments:
    an abandonment with no ply played is recorded as an abort, and the line has
    to say abort too -- that is the difference between "somebody walked out of a
    game" and "a pairing never got going", which is what an operator reads the
    field for. duration_s counts from pairing, so it is the whole wait."""
    rooms = app.state.rooms
    room = await pair_room(rooms)
    connections = app.state.connections
    connections.add(room.room_id, room.white.client_uuid, RecordingWS())
    clock.advance(45.25)

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        await finalize_and_broadcast(rooms, connections, room,
                                     Reason.ABANDONMENT, winner_color="black")

    assert room.result == (Reason.ABORTED, None)
    assert _finalize_lines(caplog) == [
        f"game finalized room={room.room_id} reason={Reason.ABORTED} winner=none "
        f"plies=0 duration_s=45.2"
    ]
