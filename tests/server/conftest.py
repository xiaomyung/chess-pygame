import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chessshootout.backend.backend import Backend
from chessshootout.backend.pieces import PieceColor, PieceType
from chessshootout.backend.utils import Square, coord_from_square
from chessshootout.server.app import create_app
from chessshootout.server.handlers import handle_skill_check_shot
from chessshootout.server.rooms import Room, RoomManager
from chessshootout.skillcheck import online
from chessshootout.skillcheck.types import SkillCheckKind
from tests.helpers import FakeClock, auth_msg, fake_uuid4, make_backend, piece, sq


__all__ = ["ALICE", "APP_KEY", "BOB", "auth_msg"]

KV_TOKEN_RE = re.compile(r"(?:^|[\s(])([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)
APP_KEY: pytest.StashKey[FastAPI] = pytest.StashKey()


@pytest.fixture
def clock() -> FakeClock:
    """
    Give each server test its own monotonic clock stand-in, so grace periods,
    skill-check deadlines and idle windows are stepped instantly instead of
    waited out

    :returns: a fresh clock sitting at zero seconds
    """
    return FakeClock()


@pytest.fixture
def app(clock: FakeClock, request: pytest.FixtureRequest) -> FastAPI:
    """
    Build a server application wired to the test clock, with a small room cap so
    the server-full path is reachable without creating a hundred rooms. The
    application is left on the test item as well, so the housekeeping check can
    still reach it after this fixture has been torn down

    :param clock: fake monotonic clock the app reads every timestamp from
    :param request: the running test, whose item carries the application on
    :returns: the application under test
    """
    application = create_app(now_provider=clock, max_rooms=8)
    request.node.stash[APP_KEY] = application
    return application


@pytest.fixture
def allow_sweep_failures() -> bool:
    """
    Opt one test out of the clean-housekeeping teardown check, for the handful
    that break a step on purpose to prove a failure stays contained. Requesting
    it is the whole opt-out; the flag itself is only read as present

    :returns: True, so a test can also assert it asked for the opt-out
    """
    return True


@pytest.fixture(autouse=True)
def clean_sweep(request: pytest.FixtureRequest) -> Iterator[None]:
    """
    Fail any server test that left a swallowed housekeeping failure behind. The
    sweep now contains a failing step instead of dying, which would otherwise
    turn a broken step into a silently passing test in the ~30 tests that drive
    those steps directly. Tests without an app never build one to be checked

    :param request: the running test, read for the fixtures it asked for
    :returns: a context that runs the check after the test body
    """
    yield
    assert_sweep_clean(request)


def assert_sweep_clean(request: pytest.FixtureRequest) -> None:
    """
    Run the housekeeping check the teardown fixture applies, as a plain call so
    the check itself can be driven by a test instead of only through pytest. A
    test that asked to be excused, and one that never built an app, both pass
    without a look

    :param request: the test that just ran, read for the fixtures it asked for
        and the application stashed on its item
    """
    if "allow_sweep_failures" in request.fixturenames:
        return
    app = request.node.stash.get(APP_KEY, None)
    if app is None:
        return
    sweep = app.state.sweep
    if sweep.failure_count:
        raise AssertionError(
            f"housekeeping swallowed {sweep.failure_count} failure(s)"
        ) from sweep._last_failure


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """
    Drive the app in-process, which exercises the real HTTP endpoints and the
    real websocket route without binding a port or starting a server

    :param app: the application under test
    :returns: a test client bound to that application
    """
    return TestClient(app)


class RecordingWS:
    """
    The stand-in socket every server test sends through: it keeps whatever the
    server wrote to it and the close code it was closed with, so a test reads
    the wire instead of reaching into server state
    """

    def __init__(self) -> None:
        """
        Start an open socket that has been sent nothing yet
        """
        self.sent: list[dict[str, Any]] = []
        self.closed_with: int | None = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        """
        Record one outbound frame in the order it was sent

        :param payload: the message the server wrote to this socket
        """
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        """
        Note the code the socket was closed with, which several tests assert on
        in place of a real close handshake

        :param code: websocket close code the server chose
        """
        self.closed_with = code

    def types(self) -> list[str]:
        """
        List the message types sent so far, for the tests that care about the
        order of a conversation rather than its contents

        :returns: every sent message's type, in order
        """
        return [m["type"] for m in self.sent]

    def of_type(self, t: str) -> list[dict[str, Any]]:
        """
        Pick out every message of one type, which is how a test asserts on the
        one frame it cares about without counting the rest

        :param t: message type to keep
        :returns: the matching messages, in the order they were sent
        """
        return [m for m in self.sent if m["type"] == t]


async def pair_room(rooms: RoomManager, *, time_minutes: int = 5,
                    tokens: tuple[str, str] = ("ta", "tb")) -> Room:
    """
    Seat both test players on the same time control so they pair into one live
    room, which is the starting position of nearly every server test. Callers
    that need sockets, a clock stamp or a particular board add those on top

    :param rooms: the room manager to enqueue into
    :param time_minutes: base minutes per side, also the queue key
    :param tokens: session tokens for the white and black seat, in that order
    :returns: the room both players now hold a seat in
    """
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token=tokens[0],
                        time_minutes=time_minutes, increment_seconds=0,
                        side_preference="white")
    return await rooms.enqueue(client_uuid=BOB, nickname="B", session_token=tokens[1],
                               time_minutes=time_minutes, increment_seconds=0,
                               side_preference="black")


def capture_backend() -> Backend:
    """
    Build the small board every capture scenario starts from: a white queen
    that can take a black pawn, with both kings well out of the way so the
    capture is the only thing the position is about

    :returns: an engine sitting on that position, white to move
    """
    return make_backend({
        sq(7, 4): piece(PieceType.KING, PieceColor.WHITE),
        sq(0, 4): piece(PieceType.KING, PieceColor.BLACK),
        sq(4, 3): piece(PieceType.QUEEN, PieceColor.WHITE),
        sq(3, 3): piece(PieceType.PAWN, PieceColor.BLACK),
    })


def seed_for(backend: Backend, frm: Square, to: Square, want_kind: SkillCheckKind,
             locks: set[tuple[Square, Square]] | None = None) -> str:
    """
    Find a room secret that makes one particular move fire the kind of check a
    test is about. The server draws the kind from the secret, so a test brute
    forces the secret rather than reaching past the selection it is exercising

    :param backend: engine holding the position the move is judged against
    :param frm: square the piece leaves
    :param to: square it lands on
    :param want_kind: the check kind that move should fire
    :param locks: moves already locked out this turn, or None for none
    :returns: a secret that selects that kind for that move
    """
    locks = locks or set()
    for i in range(4000):
        secret = f"secret-{i}"
        if online.select_kind(secret, 0, backend, frm, to, locks) == want_kind:
            return secret
    raise AssertionError(f"no secret for {want_kind}")


async def capture_room(app: FastAPI, clock: FakeClock, kind: SkillCheckKind, *,
                       backend: Backend | None = None, frm: Square = Square(4, 3),
                       to: Square = Square(3, 3),
                       ) -> tuple[Room, RecordingWS, RecordingWS, Square, Square]:
    """
    The one paired-room factory every capture scenario builds on: a board with
    a legal capture, a secret brute-forced to select the wanted kind for that
    move, and a recording socket per side. The board and squares are arguments
    so the promotion and black-mover boards reuse the same wiring

    :param app: the application under test, whose rooms and sockets are used
    :param clock: fake clock the room's timestamps are stamped from
    :param kind: the check kind the capture should fire
    :param backend: board to play on, or None for the default capture board
    :param frm: square the capturing piece leaves
    :param to: square it captures on
    :returns: the room, the white and black sockets, and the two squares
    """
    room = await pair_room(app.state.rooms)
    room.backend = capture_backend() if backend is None else backend
    room.first_move_at = clock()
    room.started_at = clock()
    room.skillcheck_secret = seed_for(room.backend, frm, to, kind)
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    return room, ws_w, ws_b, frm, to


def move_raw(frm: Square, to: Square, promotion: str | None = None) -> str:
    """
    Write the move frame a client sends, as the raw JSON text the handler
    parses, so a test drives the same bytes a real client would

    :param frm: square the piece leaves
    :param to: square it lands on
    :param promotion: piece letter to promote to, or None for an ordinary move
    :returns: the frame as JSON text
    """
    payload = {"type": "move", "from": coord_from_square(frm), "to": coord_from_square(to)}
    if promotion is not None:
        payload["promotion"] = promotion
    return json.dumps(payload)


def shot_raw(elapsed: int, direction: str | None = None,
             target: tuple[float, float] | None = None) -> str:
    """
    Write the skill-check shot frame a client sends, as raw JSON text. An
    omitted direction or target is left out of the payload entirely, so a wheel
    shot is byte-identical to what a wheel client really sends

    :param elapsed: milliseconds the client claims the shot took
    :param direction: combo direction pressed, or None
    :param target: board-space row and column shot at, or None
    :returns: the frame as JSON text
    """
    payload: dict[str, Any] = {"type": "skill_check_shot", "client_elapsed_ms": elapsed}
    if direction is not None:
        payload["direction"] = direction
    if target is not None:
        payload["target_row"], payload["target_col"] = target
    return json.dumps(payload)


async def fire(app: FastAPI, clock: FakeClock, room: Room, color: str, elapsed: int,
               direction: str | None = None,
               target: tuple[float, float] | None = None) -> str | None:
    """
    The single shot driver for all four kinds: parks the fake clock at the
    check-relative elapsed and posts the raw frame through the real handler

    :param app: the application under test
    :param clock: fake clock, moved to the moment of the shot
    :param room: the room holding the pending check
    :param color: which side is shooting
    :param elapsed: milliseconds the client claims the shot took
    :param direction: combo direction pressed, or None
    :param target: board-space row and column shot at, or None
    :returns: whatever the handler reports about the shot
    """
    clock.set((room.pending_skillcheck.start_ms + elapsed) / 1000.0)
    ws = app.state.connections.get_for_color(room, color)
    return await handle_skill_check_shot(app, ws, room, color,
                                         shot_raw(elapsed, direction, target))


def win_elapsed(pending: Any) -> int:
    """
    Solve the stored check for an elapsed time that wins it. Tests read the
    server's own seed and search for a winning shot rather than guessing the
    seed, which mirrors how the server adjudicates

    :param pending: the pending check to solve, read for its seed and kind
    :returns: milliseconds that win this check
    """
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    for e in range(int(online.SKILLCHECK_HUMAN_FLOOR_MS), int(online.SKILLCHECK_DEADLINE_MS)):
        if online.shot_wins(pending.kind, ch, e, pending.miss_count):
            return e
    raise AssertionError("no winning elapsed for the stored seed")
