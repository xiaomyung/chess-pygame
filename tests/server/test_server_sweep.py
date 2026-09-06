"""Sweep step methods exercised in isolation via a fake clock.

The Sweep class wraps the per-tick lifecycle: each step does one thing
(skill-check deadlines, clock + idle windows, heartbeat timeout, grace
expiry, pre-game orphan drop, abandoned/timed-out queue reaping,
post-game rematch window, finished-room GC) so we drive each
independently without the full asyncio loop.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from chessshootout.backend.utils import Square
from chessshootout.server.app import create_app
from chessshootout.server.protocol import (
    GRACE_SECONDS, HEARTBEAT_TIMEOUT_SECONDS, QUEUE_MAX_WAIT_SECONDS, Reason)
from chessshootout.server.rooms import POST_GAME_DISCONNECT_GRACE, QUEUE_ABANDON_SECONDS
from chessshootout.server.sweep import (
    PREGAME_CONNECT_GRACE_SECONDS, SWEEP_ERROR_LOG_INTERVAL_SECONDS,
    SWEEP_STALE_SECONDS, WS_CLOSE_QUEUE_TIMEOUT)
from tests.server.test_server_broadcasts import RecordingWS
from tests.helpers import FakeClock, fake_uuid4
from tests.server.conftest import ALICE, APP_KEY, BOB, clean_sweep


async def _pair(rooms, time_minutes=5):
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=time_minutes, increment_seconds=0,
                        side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=time_minutes, increment_seconds=0,
                        side_preference="black")
    return list(rooms._active.values())[0]


@pytest.fixture
def sweep(app):
    return app.state.sweep


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "time_minutes, plies_ever, set_first_move, arm_idle, advance, "
    "expected_result, expected_reason",
    [
        pytest.param(5, 0, False, True, 61, ("aborted", None), None,
                     id="no_first_move_aborts"),
        pytest.param(1, 1, True, True, 70, None, Reason.TIMEOUT,
                     id="flagged_clock_times_out"),
        pytest.param(5, 1, True, True, 61, ("aborted", None), None,
                     id="black_never_replies_aborts"),
        pytest.param(5, 2, True, True, 61, (Reason.RESIGNATION, "black"), None,
                     id="silence_after_both_first_moves_resigns"),
        pytest.param(180, 3, True, False, 600, None, None,
                     id="ply_three_never_arms"),
    ],
)
async def test_sweep_step_clock_and_idle_windows(sweep, app, clock, time_minutes,
                                                 plies_ever, set_first_move, arm_idle,
                                                 advance, expected_result,
                                                 expected_reason):
    """One step, one armed idle window per IDLE_WINDOW_BY_PLIES row, plus the
    clock branch it shares the walk with.

    Plies 0 and 1 expire as a fixed ("aborted", None) — nobody loses when the
    game never really started (ply 1 is issue #81: black never replies to
    white's first move). Ply 2 expires as a RESIGNATION awarding the opponent
    of the side to move (issue #82: both sides proved present, so silence is a
    forfeit). Ply 3+ never arms — a 10-minute stall on a 3-hour clock stays a
    running game. The flagged case pins the clock branch still firing when the
    idle window is also due-ish: expected differs in kind per case, so each
    carries its own expected — never flattened.
    """
    room = await _pair(app.state.rooms, time_minutes=time_minutes)
    room.started_at = clock()
    room.plies_ever = plies_ever
    if set_first_move:
        room.first_move_at = clock()
    room.idle_since = clock() if arm_idle else None
    clock.advance(advance)
    await sweep.step_clock_and_idle_windows()
    if expected_reason is not None:
        assert room.result is not None
        assert room.result[0] == expected_reason
    elif expected_result is not None:
        assert room.result == expected_result
    else:
        assert room.result is None


@pytest.mark.asyncio
async def test_a_flag_in_the_same_tick_beats_the_idle_resign(sweep, app, clock):
    """The clock branch wins the sweep pass unconditionally: when a flag fall and
    the idle-resign expiry are both due in the same sweep pass, the clock branch
    runs first and the idle branch's post-await result re-check stands down.
    finalize_result is idempotent anyway, but the re-check makes the tie-break
    intentional — a bullet game where white idles to death on the clock is a
    TIMEOUT, not an idle resignation."""
    room = await _pair(app.state.rooms, time_minutes=1)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 2
    room.idle_since = clock()
    clock.advance(70)
    await sweep.step_clock_and_idle_windows()
    assert room.result is not None
    assert room.result[0] == Reason.TIMEOUT


@pytest.mark.asyncio
async def test_a_bullet_flag_before_the_ply_one_abort_deadline_is_a_timeout(
    sweep, app, clock,
):
    """Locked product decision: first event wins — a flag that lands with or
    before the abort deadline stands as a normal timeout. At 1+0 the replier's
    whole clock fits inside the 60 s ply-1 abort window, so black never
    replying to white's first move flags on the clock branch before the idle
    branch gets a look: a real TIMEOUT with a series point for white, not a
    no-fault abort."""
    room = await _pair(app.state.rooms, time_minutes=1)
    room.started_at = clock()
    assert room.backend.try_move(Square(6, 4), Square(4, 4)).legal
    room.first_move_at = clock()
    room.plies_ever = 1
    room.idle_since = clock()
    clock.advance(61)
    await sweep.step_clock_and_idle_windows()
    assert room.result == (Reason.TIMEOUT, "white")
    assert room.series_scores == {"A": 1.0}


@pytest.mark.asyncio
async def test_no_idle_resignation_while_the_would_be_winner_is_disconnected(
    sweep, app, clock,
):
    """The inverse of the aligned cases below: black plays ply 2 and then
    drops, so white sits through an "Abandon in" countdown — awarding black an
    idle-resign win in that state would contradict the badge on white's
    screen. The idle branch skips the pass whenever the would-be winner's slot
    is disconnected and leaves the room to the grace sweep, where abandonment
    awards the CONNECTED player: white, the idler."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 2
    room.idle_since = clock()
    room.black.connected = True
    rooms.mark_disconnected(room.room_id, "black")
    clock.advance(61)
    await sweep.step_clock_and_idle_windows()
    assert room.result is None, "no idle forfeit may crown a disconnected winner"
    await sweep.step_grace_expired()
    assert room.result == (Reason.ABANDONMENT, "white")


@pytest.mark.asyncio
async def test_the_idle_resign_beats_a_same_pass_grace_abandonment(sweep, app, clock):
    """step_clock_and_idle_windows precedes step_grace_expired in step_all, so an
    idler who also disconnected at window start forfeits by RESIGNATION, not
    ABANDONMENT — deterministic, and both reasons would name the same winner."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 2
    room.idle_since = clock()
    room.white.connected = True
    rooms.mark_disconnected(room.room_id, "white")
    clock.advance(61)
    await sweep.step_all()
    assert room.result == (Reason.RESIGNATION, "black")


@pytest.mark.asyncio
async def test_a_disconnect_does_not_dodge_the_idle_window(sweep, app, clock):
    """Same posture as expired skill-check pendings: pulling the plug neither
    pauses nor resets the idle window. No /resume happened, nothing restamped
    idle_since, and the forfeit still lands on schedule."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 2
    armed_at = clock()
    room.idle_since = armed_at
    room.white.connected = True
    rooms.mark_disconnected(room.room_id, "white")
    clock.advance(61)
    assert room.idle_since == armed_at, "the disconnect never touched the window"
    await sweep.step_clock_and_idle_windows()
    assert room.result == (Reason.RESIGNATION, "black")


@pytest.mark.asyncio
async def test_the_black_never_moved_abort_beats_a_same_tick_abandonment(sweep, app, clock):
    """Issue #81's real-world shape: black connects, never replies, and often
    also drops the socket. The ply-1 abort window expires in the same pass the
    grace would — step order makes the ABORTED outcome win, so nobody gets a
    winner and nobody scores a series point off a game that never started."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 1
    room.idle_since = clock()
    room.black.connected = True
    rooms.mark_disconnected(room.room_id, "black")
    clock.advance(61)
    await sweep.step_all()
    assert room.result == ("aborted", None)
    assert room.series_scores == {}


@pytest.mark.asyncio
async def test_sweep_step_grace_expired_without_desync_awards_opponent(sweep, app, clock):
    room = await _pair(app.state.rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 1
    room.white.connected = True
    app.state.rooms.mark_disconnected(room.room_id, "white")
    clock.advance(61)
    await sweep.step_grace_expired()
    assert room.result == (Reason.ABANDONMENT, "black")


@pytest.mark.asyncio
async def test_sweep_step_grace_expired_with_desync_awards_opponent(sweep, app, clock):
    """See test_server_app: the desync flag never downgrades an abandonment win
    once moves were played; zero-ply games abort via finalize_result instead."""
    room = await _pair(app.state.rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 1
    room.white.connected = True
    room.white.desync_active = True
    app.state.rooms.mark_disconnected(room.room_id, "white")
    clock.advance(61)
    await sweep.step_grace_expired()
    assert room.result == (Reason.ABANDONMENT, "black")


async def test_post_game_leaver_grace_restarts_at_the_result(sweep, app, clock):
    """REGRESSION (v2.10.0 live smoke): the winner never saw the VICTORY screen.
    The leaver's pre-result disconnected_at also satisfied the post-game rematch
    grace, so opponent_left fired in the same sweep pass as the result and the
    client tore the session down instantly. finalize_result now restamps a
    disconnected slot's clock to ended_at: the post-game window gets its full
    grace measured from the result, not from the original disconnect."""
    room = await _pair(app.state.rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.plies_ever = 1
    room.white.connected = True
    app.state.rooms.mark_disconnected(room.room_id, "white")
    clock.advance(61)
    await sweep.step_grace_expired()
    assert room.result == (Reason.ABANDONMENT, "black")
    assert room.white.disconnected_at == room.ended_at
    ws_black = RecordingWS()
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_black)
    clock.advance(POST_GAME_DISCONNECT_GRACE - 1)
    await sweep.step_post_game()
    assert app.state.rooms.get(room.room_id) is room, "room survives inside the fresh grace"
    assert not ws_black.of_type("rematch_update")
    clock.advance(2)
    await sweep.step_post_game()
    updates = ws_black.of_type("rematch_update")
    assert updates and updates[-1]["event"] == "opponent_left"


CARL = fake_uuid4(3)
DAVE = fake_uuid4(4)


@pytest.mark.asyncio
async def test_sweep_step_grace_expired_revalidates_a_reconnect_that_lands_mid_pass(
    sweep, app, clock,
):
    """grace_expired_rooms() snapshots eagerly, then the loop awaits a broadcast
    per room -- if a reconnect for a LATER room in the batch lands during an
    earlier room's await, that later room must not be steamrolled by the stale
    snapshot read. A fake socket on room_a's present side reconnects room_b's
    gone player as a side effect of being sent room_a's own result broadcast,
    which is exactly the kind of state change a truly concurrent task could
    cause mid-await."""
    rooms = app.state.rooms
    connections = app.state.connections

    room_a = await _pair(rooms)
    room_a.started_at = clock()
    room_a.first_move_at = clock()
    room_a.plies_ever = 1
    room_a.white.connected = True
    room_a.black.connected = True

    await rooms.enqueue(client_uuid=CARL, nickname="C", session_token="tc",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    room_b = await rooms.enqueue(client_uuid=DAVE, nickname="D", session_token="td",
                                 time_minutes=5, increment_seconds=0, side_preference="black")
    room_b.started_at = clock()
    room_b.first_move_at = clock()
    room_b.plies_ever = 1
    room_b.white.connected = True
    room_b.black.connected = True

    rooms.mark_disconnected(room_a.room_id, "white")
    rooms.mark_disconnected(room_b.room_id, "white")
    clock.advance(GRACE_SECONDS + 1)

    class ReconnectingWS:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)
            rooms.mark_connected(room_b.room_id, "white")

    connections.add(room_a.room_id, room_a.black.client_uuid, ReconnectingWS())

    await sweep.step_grace_expired()

    assert room_a.result == (Reason.ABANDONMENT, "black")
    assert room_b.result is None, "the reconnect that landed mid-pass must be honored"
    assert room_b.white.connected is True


@pytest.mark.asyncio
async def test_sweep_step_drop_orphans_pre_game(sweep, app, clock):
    """A paired pre-game room with no live ws survives within the connect
    grace (so client ws handshakes can still land) and is dropped past it."""
    rooms = app.state.rooms
    await _pair(rooms)
    assert rooms.rooms_active == 1
    sweep.step_drop_orphans_pre_game()
    assert rooms.rooms_active == 1
    clock.advance(PREGAME_CONNECT_GRACE_SECONDS)
    sweep.step_drop_orphans_pre_game()
    assert rooms.rooms_active == 0


@pytest.mark.asyncio
async def test_sweep_step_drop_orphans_skips_after_first_move(sweep, app, clock):
    """After the first move a missing connection starts the grace timer
    instead of dropping the room immediately."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.first_move_at = clock()
    sweep.step_drop_orphans_pre_game()
    assert rooms.rooms_active == 1


@pytest.mark.asyncio
async def test_sweep_step_all_runs_in_documented_order(sweep, app, clock, monkeypatch):
    calls = []

    async def _trace_clock():
        calls.append("clock_and_idle")

    async def _trace_grace():
        calls.append("grace")

    async def _trace_heartbeat():
        calls.append("heartbeat_timeout")

    def _trace_drop():
        calls.append("drop_orphans")

    def _trace_reap_queue():
        calls.append("reap_queue")

    async def _trace_timeout_queue():
        calls.append("timeout_queue")

    async def _trace_post_game():
        calls.append("post_game")

    monkeypatch.setattr(sweep, "step_clock_and_idle_windows", _trace_clock)
    monkeypatch.setattr(sweep, "step_grace_expired", _trace_grace)
    monkeypatch.setattr(sweep, "step_heartbeat_timeout", _trace_heartbeat)
    monkeypatch.setattr(sweep, "step_drop_orphans_pre_game", _trace_drop)
    monkeypatch.setattr(sweep, "step_reap_abandoned_queue", _trace_reap_queue)
    monkeypatch.setattr(sweep, "step_reap_timed_out_queue", _trace_timeout_queue)
    monkeypatch.setattr(sweep, "step_post_game", _trace_post_game)
    monkeypatch.setattr(sweep.rooms, "gc_finished_rooms",
                        lambda: calls.append("gc"))
    await sweep.step_all()
    assert calls == ["clock_and_idle", "heartbeat_timeout", "grace",
                     "drop_orphans", "reap_queue", "timeout_queue", "post_game", "gc"]


async def _queue_alone(rooms, time_minutes=5):
    return await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                               time_minutes=time_minutes, increment_seconds=0,
                               side_preference="white")


@pytest.mark.asyncio
async def test_sweep_reaps_a_queued_room_nobody_ever_connected_to(sweep, app, clock):
    """SECURITY: POST /matchmake with no websocket and no DELETE used to pin a
    queue slot until process restart, and queue depth counts against max_rooms --
    ~max_rooms such requests locked every later player out with 503 room_full.
    The sweep now reaps the abandoned waiter once it is past the TTL."""
    rooms = app.state.rooms
    room = await _queue_alone(rooms)
    clock.advance(QUEUE_ABANDON_SECONDS - 1)
    sweep.step_reap_abandoned_queue()
    assert rooms.queue_depth == 1, "a waiter inside the TTL is never reaped"
    clock.advance(2)
    sweep.step_reap_abandoned_queue()
    assert rooms.queue_depth == 0
    assert rooms.get(room.room_id) is None


@pytest.mark.asyncio
async def test_the_abandoned_reap_never_touches_a_queued_player_holding_a_live_socket(
        sweep, app, clock):
    """A real player waiting for a match auths a websocket to the queued room and
    holds it open with no timeout on the client side, so age alone cannot mean
    abandoned. The live socket is the liveness proof: only a queued room with no
    connection behind it is reapable by THIS step, which is exactly the shape of
    the HTTP-only flood. A connected waiter is evicted by the hard TTL step
    instead, which tells it why before closing the socket."""
    rooms = app.state.rooms
    room = await _queue_alone(rooms)
    app.state.connections.add(room.room_id, ALICE, RecordingWS())
    clock.advance(QUEUE_MAX_WAIT_SECONDS * 10)
    sweep.step_reap_abandoned_queue()
    assert rooms.get(room.room_id) is room
    assert rooms.queue_depth == 1


@pytest.mark.asyncio
async def test_a_connected_waiter_survives_the_abandon_ttl_and_dies_at_the_hard_ttl(
        sweep, app, clock):
    """SECURITY: queue depth counts against max_rooms, and the abandoned-queue reap
    spares anything holding a socket — so ~max_rooms clients that connect and then
    sit there forever pin every slot and answer real matchmaking with 503 for the
    life of the process. The hard TTL is liveness-independent: past it the waiter is
    dequeued whatever its socket is doing, told why, and closed."""
    rooms = app.state.rooms
    room = await _queue_alone(rooms)
    ws = RecordingWS()
    app.state.connections.add(room.room_id, ALICE, ws)

    clock.advance(QUEUE_ABANDON_SECONDS + 1)
    await sweep.step_reap_timed_out_queue()
    assert rooms.queue_depth == 1, "the abandon TTL is not the hard TTL"

    clock.advance(QUEUE_MAX_WAIT_SECONDS)
    await sweep.step_reap_timed_out_queue()

    assert rooms.queue_depth == 0
    assert rooms.get(room.room_id) is None
    assert rooms._queue == {}, "the emptied time-control bucket goes with it"
    assert rooms._uuid_to_room == {}, "and the uuid is free to matchmake again"
    assert ws.of_type("error")[-1]["reason"] == Reason.QUEUE_TIMEOUT, \
        "the waiter is told its search is over instead of hanging on 'searching'"
    assert ws.closed_with == WS_CLOSE_QUEUE_TIMEOUT


@pytest.mark.asyncio
async def test_the_hard_ttl_reap_needs_no_socket_and_leaves_paired_rooms_alone(
        sweep, app, clock):
    """The socketless waiter (already the abandoned reap's job) must not crash the
    notify path, and a paired room inherits the queued room's created_at, so the
    TTL walk must stay inside the queue."""
    rooms = app.state.rooms
    orphan = await rooms.enqueue(client_uuid=fake_uuid4(91), nickname="Q",
                                 session_token="tq", time_minutes=3,
                                 increment_seconds=0, side_preference="white")
    paired = await _pair(rooms, time_minutes=10)
    paired.first_move_at = clock()
    clock.advance(QUEUE_MAX_WAIT_SECONDS + 1)

    await sweep.step_reap_timed_out_queue()

    assert rooms.get(orphan.room_id) is None
    assert rooms.get(paired.room_id) is paired
    assert rooms.rooms_active == 1


@pytest.mark.asyncio
async def test_sweep_reap_leaves_paired_rooms_alone(sweep, app, clock):
    """The reap walks the queue only: a paired room keeps the `created_at` of the
    queued room it grew from, so an active game that outlives the TTL must be
    untouched."""
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    clock.advance(QUEUE_ABANDON_SECONDS * 10)
    sweep.step_reap_abandoned_queue()
    assert rooms.get(room.room_id) is room
    assert rooms.rooms_active == 1


@pytest.mark.asyncio
async def test_heartbeat_timeout_marks_disconnected(sweep, app, clock):
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.white.connected = True
    room.white.last_seen = clock()
    clock.advance(HEARTBEAT_TIMEOUT_SECONDS + 1)
    await sweep.step_heartbeat_timeout()
    assert room.white.connected is False
    assert room.white.disconnected_at is not None


@pytest.mark.asyncio
async def test_heartbeat_timeout_ignores_fresh_pings(sweep, app, clock):
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.first_move_at = clock()
    room.white.connected = True
    room.white.last_seen = clock()
    clock.advance(1)
    await sweep.step_heartbeat_timeout()
    assert room.white.connected is True


@pytest.mark.asyncio
async def test_heartbeat_timeout_skips_pre_first_move_and_finished(sweep, app, clock):
    rooms = app.state.rooms
    room = await _pair(rooms)
    room.started_at = clock()
    room.white.connected = True
    room.white.last_seen = clock()
    clock.advance(HEARTBEAT_TIMEOUT_SECONDS + 1)
    await sweep.step_heartbeat_timeout()
    assert room.white.connected is True
    room.first_move_at = clock()
    rooms.finalize_result(room.room_id, Reason.RESIGNATION, winner_color="white")
    await sweep.step_heartbeat_timeout()
    assert room.white.connected is True


async def _pair_nth(rooms, n, time_minutes=5):
    """One more paired room, with uuids nobody else in the test is using, so a
    single test can hold several live games at once."""
    for offset, side in ((0, "white"), (1, "black")):
        await rooms.enqueue(client_uuid=fake_uuid4(300 + 2 * n + offset),
                            nickname=f"P{n}{side[0]}", session_token=f"t{n}{offset}",
                            time_minutes=time_minutes, increment_seconds=0,
                            side_preference=side)
    return list(rooms._active.values())[-1]


@pytest.mark.asyncio
async def test_a_failing_step_never_stops_the_steps_that_come_after_it(
        sweep, monkeypatch, allow_sweep_failures):
    """Per-step isolation. The old loop died on the first exception any step
    raised and /healthz stayed 200 -- clocks, grace and queue reaping silently
    stopped for the life of the process. The first step is broken here on
    purpose; every later step in the same pass still has to run."""
    calls = []

    async def _boom():
        raise RuntimeError("skillcheck step exploded")

    def _tracer(name):
        def _sync():
            calls.append(name)
        return _sync

    async def _atracer(name):
        calls.append(name)

    monkeypatch.setattr(sweep, "step_skillcheck_deadline", _boom)
    monkeypatch.setattr(sweep, "step_clock_and_idle_windows",
                        lambda: _atracer("clock_and_idle"))
    monkeypatch.setattr(sweep, "step_heartbeat_timeout", lambda: _atracer("heartbeat"))
    monkeypatch.setattr(sweep, "step_grace_expired", lambda: _atracer("grace"))
    monkeypatch.setattr(sweep, "step_drop_orphans_pre_game", _tracer("drop_orphans"))
    monkeypatch.setattr(sweep, "step_reap_abandoned_queue", _tracer("reap_queue"))
    monkeypatch.setattr(sweep, "step_reap_timed_out_queue",
                        lambda: _atracer("timeout_queue"))
    monkeypatch.setattr(sweep, "step_post_game", lambda: _atracer("post_game"))
    monkeypatch.setattr(sweep.rooms, "gc_finished_rooms", _tracer("gc"))

    await sweep.step_all()

    assert calls == ["clock_and_idle", "heartbeat", "grace", "drop_orphans",
                     "reap_queue", "timeout_queue", "post_game", "gc"]
    assert sweep.failure_count == 1
    assert isinstance(sweep._last_failure, RuntimeError)


@pytest.mark.asyncio
async def test_a_poisoned_room_leaves_its_siblings_ticking(
        sweep, app, clock, monkeypatch, allow_sweep_failures):
    """Per-ROOM isolation inside one step, the finer half of the ladder. One
    room whose backend raises mid-walk used to take every room behind it in the
    same iteration order down with it -- so a single corrupt game froze other
    people's clocks. The sibling still has to flag."""
    rooms = app.state.rooms
    poisoned = await _pair_nth(rooms, 0, time_minutes=1)
    healthy = await _pair_nth(rooms, 1, time_minutes=1)
    for room in (poisoned, healthy):
        room.started_at = clock()
        room.first_move_at = clock()
        room.plies_ever = 2

    def _boom():
        raise RuntimeError("poisoned backend")

    monkeypatch.setattr(poisoned.backend, "tick_clock", _boom)
    clock.advance(70)

    await sweep.step_clock_and_idle_windows()

    assert poisoned.result is None
    assert healthy.result is not None and healthy.result[0] == Reason.TIMEOUT
    assert sweep.failure_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_type", [asyncio.CancelledError, KeyboardInterrupt],
                         ids=["cancelled", "keyboard_interrupt"])
async def test_a_cancel_or_an_interrupt_is_never_swallowed_by_the_step_guard(
        sweep, exc_type):
    """The net catches Exception, never BaseException: a shutdown cancel has to
    reach the loop that owns it and Ctrl-C has to reach the process. Swallowing
    either would leave a server that cannot be stopped."""
    async def _raise():
        raise exc_type()

    with pytest.raises(exc_type):
        await sweep._guard("post_game", _raise)
    assert sweep.failure_count == 0


@pytest.mark.asyncio
async def test_a_clean_pass_restamps_the_freshness_and_a_dirty_one_does_not(
        sweep, clock, monkeypatch, allow_sweep_failures):
    """The stamp is the whole basis of the degraded verdict, so it may only move
    when EVERY step of a pass got through. A pass that half-worked reports as if
    it had not run at all."""
    sweep.mark_running()
    clock.advance(10)
    await sweep.step_all()
    assert sweep.age_s == 0.0

    async def _boom():
        raise RuntimeError("post-game step exploded")

    monkeypatch.setattr(sweep, "step_post_game", _boom)
    clock.advance(10)
    await sweep.step_all()
    assert sweep.age_s == pytest.approx(10.0), "a dirty pass leaves the stamp alone"
    clock.advance(10)
    assert sweep.age_s == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_the_freshness_age_reads_zero_until_the_timer_starts(sweep, clock):
    """Before the loop runs there is nothing to be late for. Without this a
    server would answer degraded for its first half minute of life, and every
    in-process test app -- which never starts the loop -- would too."""
    clock.advance(600)
    assert sweep.age_s == 0.0
    assert sweep.is_stale is False

    sweep.mark_running()
    clock.advance(SWEEP_STALE_SECONDS)
    assert sweep.age_s == pytest.approx(SWEEP_STALE_SECONDS)
    assert sweep.is_stale is False, "exactly at the limit is still healthy"
    clock.advance(0.1)
    assert sweep.is_stale is True


@pytest.mark.asyncio
async def test_repeated_failures_are_reported_once_per_interval_with_a_count(
        sweep, clock, caplog, allow_sweep_failures):
    """A fault that repeats every tick would write ten ERROR lines a second and
    bury everything else in the journal. The first one is immediate, the rest
    are counted into the next line the throttle lets through."""
    with caplog.at_level(logging.ERROR, logger="chess.server.app"):
        for _ in range(5):
            sweep._note_failure("post_game", RuntimeError("nope"))
        records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(records) == 1, "four repeats inside the interval stay quiet"
        assert "step=post_game" in records[0].getMessage()
        assert "suppressed=0" in records[0].getMessage()
        assert records[0].exc_info is not None, "the first report carries a traceback"

        clock.advance(SWEEP_ERROR_LOG_INTERVAL_SECONDS)
        sweep._note_failure("grace_expired", RuntimeError("still nope"))

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(records) == 2
    assert "step=grace_expired" in records[1].getMessage()
    assert "suppressed=4" in records[1].getMessage()
    assert sweep.failure_count == 6


@pytest.mark.asyncio
async def test_a_pass_that_fails_outright_stops_the_freshness_stamp(
        sweep, clock, allow_sweep_failures):
    """note_unhandled is the loop's own arm of the ladder: a failure that
    escaped every step guard still has to degrade the health report rather than
    kill the loop."""
    sweep.mark_running()
    clock.advance(SWEEP_STALE_SECONDS)
    await sweep.step_all()
    assert sweep.age_s == 0.0

    clock.advance(SWEEP_STALE_SECONDS + 0.1)
    sweep.note_unhandled(RuntimeError("the whole pass exploded"))

    assert sweep.failure_count == 1
    assert isinstance(sweep._last_failure, RuntimeError)
    assert sweep.is_stale is True


@pytest.mark.parametrize(
    "fixturenames, stashed, expect_raise",
    [
        pytest.param((), True, True, id="swallowed_failure_fails_the_test"),
        pytest.param(("allow_sweep_failures",), True, False, id="opt_out_is_honoured"),
        pytest.param((), False, False, id="a_test_without_an_app_is_skipped"),
    ],
)
def test_the_clean_sweep_teardown_check_catches_a_swallowed_failure(
        fixturenames, stashed, expect_raise):
    """The suite's own safety net, driven by hand. ~30 tests call sweep steps
    directly and assert on room state; now that a step failure is contained
    rather than raised, a broken step would leave those tests passing on a room
    nothing ever touched. This is the check that stops that -- built on its own
    app so the autouse copy watching this test has nothing to find."""
    app = create_app(now_provider=FakeClock(), max_rooms=8)
    app.state.sweep._note_failure("post_game", RuntimeError("swallowed"))
    node = SimpleNamespace(stash={APP_KEY: app} if stashed else {})
    request = SimpleNamespace(fixturenames=fixturenames, node=node)

    gen = clean_sweep._get_wrapped_function()(request)
    next(gen)
    if not expect_raise:
        with pytest.raises(StopIteration):
            gen.send(None)
        return
    with pytest.raises(AssertionError) as excinfo:
        gen.send(None)
    assert "swallowed 1 failure" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
