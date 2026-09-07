"""What the heartbeat is allowed to say before the server calls it a desync.

The old rule was one line long: `ply != len(move_history)` -> order a resync.
Every legitimately out-of-step moment therefore produced a "Resyncing..." toast
in a perfectly healthy game -- the instant after a move (the opponent is one
behind until the broadcast lands), the instant after a takeback (both are one
ahead), the skill-check window, and the whole match-found card, during which the
client's heartbeat was still reporting the PREVIOUS game's ply.

Five layers now stand between a heartbeat and a repair, and this file drives all
of them through the real handlers with a fake clock:

  1. the client reports no ply at all when it is not on a live online board,
  2. a game the server has not announced yet is never judged,
  3. a ply explained by a very recent history change is news in flight,
  4. a mismatch has to survive RESYNC_STABLE_MISMATCH_HEARTBEATS judged pings,
  5. the existing notify/directive debounces still sit on top.

The stamp that layer 3 reads is an invariant, not a convenience: every server
side mutation of move_history must stamp it. The AST guard at the bottom is what
keeps a fifth mutation site from being added without one.
"""
import ast
import json
import logging
import os

import pytest

import chessshootout
from chessshootout.backend.utils import square_from_coord
from chessshootout.server.broadcasts import broadcast_game_start
from chessshootout.server.handlers import (
    RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS, _newest_change_age, handle_move,
    handle_ping, handle_takeback_request, handle_takeback_response,
)
from chessshootout.server.protocol import (
    HEARTBEAT_INTERVAL_SECONDS, PROTOCOL_VERSION, RESYNC_STABLE_MISMATCH_HEARTBEATS,
    RESYNC_STRIKE_TTL_SECONDS, RESYNC_TRANSIT_GRACE_SECONDS, Reason, _transit_grace,
)
from chessshootout.server.rooms import HISTORY_CHANGE_WINDOW, PendingSkillCheck
from chessshootout.skillcheck.types import SkillCheckKind
from tests.helpers import read_source_without_docstrings
from tests.server.conftest import ALICE, BOB
from tests.server.test_server_broadcasts import RecordingWS
from tests.server.test_server_skillcheck import (
    _capture_room, _fire, _move_raw as _capture_move_raw, _win_elapsed,
)

PACKAGE_ROOT = os.path.dirname(os.path.abspath(chessshootout.__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
SERVER_ROOT = os.path.join(PACKAGE_ROOT, "server")

HISTORY_CALL_NAMES = {"try_move", "promote", "undo", "new_game", "Backend"}
HISTORY_MUTATORS = {"_apply_move", "handle_takeback_response", "enqueue",
                    "reset_for_rematch"}
SKILLCHECK_HOLD_MS = 5000.0

PING_OUTCOMES = (
    "ping", "ping_pending", "ping_pregame", "ping_offboard",
    "ping_inflight", "ping_strike", "ping_directed", "ping_gated",
)


async def _paired(app, clock):
    """A paired room with both seats wired to a recording socket, nothing
    announced yet: this is the pre-game window layer 2 protects."""
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    room.started_at = clock()
    room.first_move_at = clock()
    room.white.connected = True
    room.black.connected = True
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    return room, ws_w, ws_b


async def _live_room(app, clock):
    """A paired room whose game has really been announced through
    broadcast_game_start -- which stamps a history change of its own -- with the
    clock parked past the transit grace so that stamp no longer excuses anything.
    Every judged-heartbeat test must start from here, or the game-start stamp
    silently forgives the ply the test is trying to prove is wrong."""
    room, ws_w, ws_b = await _paired(app, clock)
    await broadcast_game_start(app.state.connections, room, clock)
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    return room, ws_w, ws_b


def _ping_raw(ply):
    return json.dumps({"version": PROTOCOL_VERSION, "type": "ping", "ply": ply})


def _move_raw(from_sq, to_sq):
    return json.dumps({"version": PROTOCOL_VERSION, "type": "move",
                       "from": from_sq, "to": to_sq})


def _directives(ws):
    return ws.of_type("resync_directive")


async def _strike_up_to_directive(app, clock, room, ws, color, ply):
    """Land exactly the strikes the constant allows before a directive, so a
    test that wants the directive itself only has to send one more ping. The
    count is against whatever this socket has already been sent, because a
    directive spends the streak and a client can earn a second one."""
    before = len(_directives(ws))
    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS - 1):
        assert await handle_ping(app, ws, room, color, _ping_raw(ply)) == "ping_strike"
    assert len(_directives(ws)) == before


async def test_a_heartbeat_one_ply_behind_a_fresh_move_is_news_in_flight(app, clock):
    """The commonest false positive there was: white's move lands on the server,
    and black's heartbeat -- already in the air -- still reports the position
    before it. Nothing is wrong with black's board; the broadcast simply has not
    arrived yet."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))

    out = await handle_ping(app, ws_b, room, "black", _ping_raw(0))

    assert out == "ping_inflight"
    assert _directives(ws_b) == []
    assert room.black.ply_mismatch_streak == 0, "an excused ping is not a strike"


async def test_the_same_heartbeat_after_the_grace_is_judged_for_real(app, clock):
    """The excuse is a window, not a blanket: a client still reporting the old
    ply a whole transit grace later has genuinely missed the broadcast."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)

    out = await handle_ping(app, ws_b, room, "black", _ping_raw(0))

    assert out == "ping_strike"
    assert room.black.ply_mismatch_streak == 1


async def test_a_heartbeat_one_ply_ahead_after_a_takeback_is_news_in_flight(app, clock):
    """The mirror case, and the reason the stamp records the PREVIOUS length
    rather than a direction flag: after an accepted takeback both clients are
    briefly one ahead of the server, which is the opposite sign to a move."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    await handle_move(app, ws_b, room, "black", _move_raw("e7", "e5"))
    await handle_takeback_request(app, ws_b, room, "black", "{}")
    await handle_takeback_response(app, ws_w, room, "white", json.dumps(
        {"type": "takeback_response", "accept": True}))
    assert len(room.backend.move_history) == 1

    out = await handle_ping(app, ws_b, room, "black", _ping_raw(2))

    assert out == "ping_inflight"
    assert _directives(ws_b) == []


async def test_a_heartbeat_one_ply_ahead_of_a_move_is_never_excused(app, clock):
    """Direction-awareness comes free from comparing against the previous
    length: after a move only a client BEHIND matches it. A client claiming a
    ply the server has never reached is wrong no matter how recent the move."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))

    out = await handle_ping(app, ws_b, room, "black", _ping_raw(2))

    assert out == "ping_strike"


async def test_two_plies_in_one_instant_both_stay_excused(app, clock):
    """Why the stamp is a deque and not a single slot: two plies can land
    between one client's heartbeats, and a single slot would leave the client
    that is two behind unexplained the moment the second ply overwrote the
    first."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    await handle_move(app, ws_b, room, "black", _move_raw("e7", "e5"))

    assert await handle_ping(app, ws_b, room, "black", _ping_raw(0)) == "ping_inflight"
    assert await handle_ping(app, ws_b, room, "black", _ping_raw(1)) == "ping_inflight"
    assert await handle_ping(app, ws_b, room, "black", _ping_raw(3)) == "ping_strike"


async def test_the_stamp_window_keeps_the_most_recent_changes(app, clock):
    """The deque is bounded, so a long game cannot grow it; the bound has to be
    generous enough that the oldest entry it drops is far outside the grace."""
    room, ws_w, ws_b = await _live_room(app, clock)
    for _ in range(HISTORY_CHANGE_WINDOW + 4):
        room.note_history_change(clock(), 0)

    assert len(room.history_changes) == HISTORY_CHANGE_WINDOW


async def test_a_rematch_starts_from_a_clean_stamp_and_streak(app, clock):
    """A rematch swaps the seats and builds a brand new board into the SAME
    room. Carrying the old game's stamps over would excuse plies from a game
    that no longer exists, and carrying the strikes over would let a client
    that lagged in game one be repaired in game two for nothing."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert room.white.ply_mismatch_streak > 0
    room.result = (Reason.RESIGNATION, "black")

    assert app.state.rooms.reset_for_rematch(room.room_id) is True

    assert list(room.history_changes) == []
    assert room.white.ply_mismatch_streak == 0
    assert room.black.ply_mismatch_streak == 0
    assert room.game_start_broadcast is False
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_pregame", \
        "and the fresh board is not judged until the players have been told about it"


async def test_a_won_skill_check_stamps_the_history_like_a_quiet_move(app, clock):
    """A skill-check win lands its held move through the same _apply_move, so it
    stamps too. Without that, the whole verdict flourish -- during which the
    client deliberately sends no heartbeat -- would be followed by a first
    heartbeat that looks a ply behind."""
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, SkillCheckKind.WHEEL)
    await broadcast_game_start(app.state.connections, room, clock)
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    await handle_move(app, ws_w, room, "white", _capture_move_raw(frm, to))
    pending = room.pending_skillcheck

    assert await _fire(app, clock, room, "white", _win_elapsed(pending)) == "applied"
    assert len(room.backend.move_history) == 1
    assert await handle_ping(app, ws_b, room, "black", _ping_raw(0)) == "ping_inflight"


async def test_a_heartbeat_is_not_judged_before_the_game_is_announced(app, clock):
    """The match-found card runs for three seconds while the socket is already
    up, and after a rematch reset the room briefly holds a fresh board nobody
    has been told about. Both used to report the previous game's ply -- a
    mismatch of up to a whole game."""
    room, ws_w, ws_b = await _paired(app, clock)

    out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_pregame"
    assert _directives(ws_w) == []
    assert room.white.desync_active is False
    assert room.white.ply_mismatch_streak == 0


async def test_a_client_off_the_board_reports_no_ply_and_earns_no_strike(app, clock):
    """A heartbeat still keeps the connection alive while the player sits on the
    menu, in the rematch window or behind a modal. It claims no ply there, and a
    claim of nothing can never be wrong."""
    room, ws_w, ws_b = await _live_room(app, clock)

    out = await handle_ping(app, ws_w, room, "white", _ping_raw(None))

    assert out == "ping_offboard"
    assert _directives(ws_w) == []
    assert room.white.ply_mismatch_streak == 0


async def test_an_omitted_ply_reads_the_same_as_an_explicit_null(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)

    out = await handle_ping(app, ws_w, room, "white",
                            json.dumps({"version": PROTOCOL_VERSION, "type": "ping"}))

    assert out == "ping_offboard"


async def test_an_offboard_heartbeat_does_not_clear_a_live_resync(app, clock):
    """Off-board is an abstention, not a recovery: the player has proved
    nothing about the board they will come back to. Clearing here would hide a
    real desync behind a trip to the menu."""
    room, ws_w, ws_b = await _live_room(app, clock)
    room.white.desync_active = True

    await handle_ping(app, ws_w, room, "white", _ping_raw(None))

    assert room.white.desync_active is True
    assert ws_b.of_type("connection_status") == []


async def test_a_mismatch_must_survive_the_stable_heartbeat_count(app, clock):
    """The count, not a timer: a single mismatching heartbeat is the shape of a
    race, a repeated one is the shape of a broken board."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)

    out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_directed"
    assert len(_directives(ws_w)) == 1
    assert _directives(ws_w)[0]["server_ply"] == 0


async def test_strikes_in_the_same_instant_still_earn_the_directive(app, clock):
    """"Two judged heartbeats, however close together" is deliberate: a client
    that has genuinely lost the plot must not have its repair delayed just
    because its pings arrived back to back."""
    room, ws_w, ws_b = await _live_room(app, clock)
    at = clock()
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)

    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"
    assert clock() == at, "no time passed at all between the strikes"


async def test_a_matching_heartbeat_resets_the_streak(app, clock):
    """Strikes have to be consecutive, or a client that hiccups once every few
    minutes would eventually accumulate its way into a spurious repair."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)

    assert await handle_ping(app, ws_w, room, "white", _ping_raw(0)) == "ping"
    assert room.white.ply_mismatch_streak == 0

    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert _directives(ws_w) == []


async def test_an_inflight_heartbeat_neither_sets_nor_clears_the_resync_flag(app, clock):
    """An excused ping is not evidence in either direction: it must not raise a
    resync, and it must not report a lagging client as recovered."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    room.black.desync_active = True

    assert await handle_ping(app, ws_b, room, "black", _ping_raw(0)) == "ping_inflight"

    assert room.black.desync_active is True
    assert ws_w.of_type("connection_status") == []


async def test_a_fresh_socket_clears_the_strike_streak(app, clock):
    """A reconnect rebuilds the client's whole state from /resume, so whatever
    it was mismatching about before is gone. Carrying the streak across would
    let a pre-drop strike combine with a post-reconnect race into a repair
    neither of them earned."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)

    app.state.rooms.mark_connected(room.room_id, "white")

    assert room.white.ply_mismatch_streak == 0
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert _directives(ws_w) == []


async def test_a_second_directive_is_held_inside_the_directive_interval(app, clock):
    """Each directive drives a /resume, so the debounce is what stops a client
    that keeps mismatching from amplifying itself against the state-rebuild
    path. The strike counter sits in front of it, never instead of it: a client
    that ignores an order pays for the next one in strikes first, and the
    interval still holds that one back when both land inside the same second."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"

    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_gated"
    assert len(_directives(ws_w)) == 1

    clock.advance(RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS + 0.1)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"
    assert len(_directives(ws_w)) == 2


async def test_an_ignored_directive_costs_a_fresh_pair_of_strikes(app, clock):
    """The directive is what the strikes buy, so sending one spends them. Left
    standing, the streak parked itself at the threshold for the rest of the
    spell and every later mismatching heartbeat became a directive candidate
    guarded by nothing but the one-second interval."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"
    assert room.white.ply_mismatch_streak == 0

    clock.advance(RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS + 0.1)
    out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_strike", "the interval has reopened; the streak has not"
    assert len(_directives(ws_w)) == 1


async def test_a_resume_forgets_the_strike_streak_it_is_answering(app, client, clock):
    """The real sequence: strikes earn a directive, the client obeys it with a
    /resume, and the first heartbeat after that still reports the old ply
    because the answer is only just being applied. That heartbeat is a first
    strike, not a second one -- /resume handed over the whole state, so
    whatever was counted against this player before it is spent."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"

    resp = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id,
        "session_token": room.white.session_token,
    })

    assert resp.status_code == 200
    assert room.white.ply_mismatch_streak == 0
    clock.advance(RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS + 0.1)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"
    assert len(_directives(ws_w)) == 1, "one order, one answer, no second order"


async def test_a_resume_tells_the_opponent_the_board_is_being_rebuilt(app, client, clock):
    """The other half of the same block, and the reason it is guarded at all:
    a /resume from a live socket is a repair in progress, so the opponent's
    strip says so instead of showing a player who has just gone quiet. Only the
    strike reset sits outside that guard -- being told the whole state is not
    conditional on anyone else hearing about it."""
    room, ws_w, ws_b = await _live_room(app, clock)

    resp = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id,
        "session_token": room.white.session_token,
    })

    assert resp.status_code == 200
    assert room.white.desync_active is True
    assert [m["opp_state"] for m in ws_b.of_type("connection_status")] == ["resyncing"]


async def test_strikes_further_apart_than_the_ttl_never_add_up(app, clock):
    """Strikes are consecutive in TIME as well as in count. Without the TTL a
    client that mismatches once an hour -- a genuine one-off race each time --
    eventually accumulated its way into a repair order it never earned, because
    nothing but a matching heartbeat ever reset the counter."""
    room, ws_w, ws_b = await _live_room(app, clock)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"
    clock.advance(RESYNC_STRIKE_TTL_SECONDS + 0.1)

    out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_strike", "the stale strike expired; this one starts afresh"
    assert room.white.ply_mismatch_streak == 1
    assert _directives(ws_w) == []


async def test_the_ttl_is_long_enough_for_the_strikes_it_counts(app, clock):
    """The TTL has to outlast the heartbeats the streak is measured in, or the
    rule it guards could never be met: two judged heartbeats one interval apart,
    plus the transit grace one of them may have spent being excused."""
    assert RESYNC_STRIKE_TTL_SECONDS > (
        HEARTBEAT_INTERVAL_SECONDS * RESYNC_STABLE_MISMATCH_HEARTBEATS)
    room, ws_w, ws_b = await _live_room(app, clock)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"
    clock.advance(HEARTBEAT_INTERVAL_SECONDS)

    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"


async def test_an_inflight_heartbeat_between_two_strikes_resets_the_streak(app, clock):
    """A client that reports exactly the length the history had a moment ago is
    demonstrably following the history, whatever it mismatched about before. It
    is proof of tracking, so it spends the strikes the same way a matching
    heartbeat does."""
    room, ws_w, ws_b = await _live_room(app, clock)
    assert await handle_ping(app, ws_b, room, "black", _ping_raw(7)) == "ping_strike"
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))

    assert await handle_ping(app, ws_b, room, "black", _ping_raw(0)) == "ping_inflight"
    assert room.black.ply_mismatch_streak == 0

    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    assert await handle_ping(app, ws_b, room, "black", _ping_raw(7)) == "ping_strike"
    assert _directives(ws_b) == []


async def test_a_caught_up_heartbeat_clears_the_resync_even_during_a_check(app, clock):
    """An exact match is never a resync, so it is settled before the skill-check
    gate rather than behind it. A check can run for five seconds; a player who
    was being repaired and has since caught up must not stay reported as broken
    for the whole of it."""
    room, ws_w, ws_b = await _live_room(app, clock)
    now_ms = app.state.now_ms()
    room.pending_skillcheck = PendingSkillCheck(
        color="white", from_sq=square_from_coord("e4"), to_sq=square_from_coord("d5"),
        promotion=None, kind=SkillCheckKind.WHEEL, seed="0" * 32, value_diff=0,
        start_ms=now_ms, expires_at_ms=now_ms + SKILLCHECK_HOLD_MS,
    )
    room.white.mark_desynced()

    assert await handle_ping(app, ws_w, room, "white", _ping_raw(0)) == "ping"

    assert room.white.desync_active is False
    assert [m["opp_state"] for m in ws_b.of_type("connection_status")] == ["connected"]


async def test_a_gated_heartbeat_writes_no_line_of_its_own(app, clock, caplog):
    """The line rides the order, not the mismatch: a heartbeat held back by the
    directive interval sent nothing, so there is nothing for an operator to
    read about it."""
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_directed"
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        caplog.clear()
        out = await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    assert out == "ping_gated"
    assert [r.getMessage() for r in caplog.records] == []


async def test_a_resume_does_not_restart_the_lagging_spells_line(
        app, client, clock, caplog):
    """/resume raises the desync flag itself, so the spell is already running by
    the time the first directive is written. It is still one spell and still one
    line, however long the client keeps lagging afterwards."""
    room, ws_w, ws_b = await _live_room(app, clock)
    resp = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id,
        "session_token": room.white.session_token,
    })
    assert resp.status_code == 200
    assert room.white.desync_active is True

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        for _ in range(10):
            await handle_ping(app, ws_w, room, "white", _ping_raw(7))
            clock.advance(HEARTBEAT_INTERVAL_SECONDS)

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("resync directive")]
    assert len(lines) == 1
    assert len(_directives(ws_w)) > 1


async def test_a_resume_on_a_finished_room_tells_the_opponent_nothing(
        app, client, clock):
    """The result screen is not a repair in progress. A /resume there -- which
    is exactly what a client does when it reconnects to a game that ended while
    it was away -- used to light the opponent's strip up with 'resyncing' for a
    game neither of them is playing any more."""
    room, ws_w, ws_b = await _live_room(app, clock)
    room.result = (Reason.RESIGNATION, "black")

    resp = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id,
        "session_token": room.white.session_token,
    })

    assert resp.status_code == 200
    assert room.white.desync_active is False
    assert ws_b.of_type("connection_status") == []


async def test_a_landed_move_clears_the_movers_strike(app, clock):
    """A player who makes a legal move has proved their board is the server's:
    the move was generated on it and accepted against it. That is the same
    proof a matching heartbeat gives, so it spends the strikes too -- otherwise
    a strike from before the move survives into the next mismatch and turns the
    first heartbeat that lags behind a broadcast into a repair order."""
    room, ws_w, ws_b = await _live_room(app, clock)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"
    assert room.white.ply_mismatch_streak == 1

    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))

    assert room.white.ply_mismatch_streak == 0
    assert room.white.desync_active is False, "nothing was ever repaired here"
    clock.advance(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    assert await handle_ping(app, ws_w, room, "white", _ping_raw(7)) == "ping_strike"
    assert _directives(ws_w) == []


async def test_a_lagging_spell_logs_one_line_however_long_it_lasts(app, clock, caplog):
    """Heartbeats are client-driven and arrive every couple of seconds per
    player. One line per directive would make a single stuck client the loudest
    thing in the journal, so the line rides the desync_active flip instead."""
    room, ws_w, ws_b = await _live_room(app, clock)
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        for _ in range(30):
            await handle_ping(app, ws_w, room, "white", _ping_raw(7))
            clock.advance(HEARTBEAT_INTERVAL_SECONDS)

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("resync directive")]
    assert len(lines) == 1
    assert f"room={room.room_id}" in lines[0]
    assert "color=white" in lines[0]
    assert "client_ply=7" in lines[0]
    assert "server_ply=0" in lines[0]
    assert f"streak={RESYNC_STABLE_MISMATCH_HEARTBEATS}" in lines[0]
    assert len(_directives(ws_w)) > 1, "the directives themselves keep coming"


async def test_recovery_logs_its_own_line_once(app, clock, caplog):
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    await handle_ping(app, ws_w, room, "white", _ping_raw(7))

    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        await handle_ping(app, ws_w, room, "white", _ping_raw(0))
        await handle_ping(app, ws_w, room, "white", _ping_raw(0))

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("resync cleared")]
    assert lines == [f"resync cleared room={room.room_id} color=white"]


async def test_the_logged_age_is_the_time_since_the_last_history_change(app, clock):
    """age_s is the operator's first question -- was the client behind a move
    that had only just landed, or behind nothing at all? Past the pre-game gate
    a room always carries at least the game-start stamp, so the field is always
    defined."""
    room, ws_w, ws_b = await _live_room(app, clock)
    assert room.history_changes, "broadcast_game_start always leaves a stamp"

    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    clock.advance(4.0)

    assert _newest_change_age(room, clock()) == pytest.approx(4.0)


def test_the_age_of_a_room_that_has_never_changed_reads_as_zero():
    """Defensive only: the pre-game gate means no judged heartbeat can reach the
    line with an empty deque, and a formatting crash inside a log call is a
    silly way to lose a game."""

    class _Empty:
        history_changes = ()

    assert _newest_change_age(_Empty(), 12.0) == 0.0


@pytest.mark.parametrize(
    "interval, expected",
    [
        pytest.param(0.5, 0.375, id="fast_heartbeat_scales_down"),
        pytest.param(2.0, 1.5, id="default_heartbeat_hits_the_cap"),
        pytest.param(10.0, 1.5, id="slow_heartbeat_is_capped"),
    ],
)
def test_the_transit_grace_stays_inside_one_heartbeat(interval, expected):
    """Tested on the pure function rather than the module constant, so the
    relation is provable without reloading the module under a patched env. Two
    properties matter: it covers a round trip plus a client frame, and it is
    strictly shorter than one heartbeat, so at most one heartbeat per history
    change is ever forgiven."""
    grace = _transit_grace(interval)
    assert grace == pytest.approx(expected)
    assert grace < interval


def test_the_shipped_transit_grace_comes_from_the_shipped_heartbeat():
    assert RESYNC_TRANSIT_GRACE_SECONDS == _transit_grace(HEARTBEAT_INTERVAL_SECONDS)
    assert RESYNC_TRANSIT_GRACE_SECONDS < HEARTBEAT_INTERVAL_SECONDS


async def _scenario_ping(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    return room, ws_w, "white", _ping_raw(0)


async def _scenario_ping_pending(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    now_ms = app.state.now_ms()
    room.pending_skillcheck = PendingSkillCheck(
        color="white", from_sq=square_from_coord("e4"), to_sq=square_from_coord("d5"),
        promotion=None, kind=SkillCheckKind.WHEEL, seed="0" * 32, value_diff=0,
        start_ms=now_ms, expires_at_ms=now_ms + SKILLCHECK_HOLD_MS,
    )
    return room, ws_w, "white", _ping_raw(7)


async def _scenario_ping_pregame(app, clock):
    room, ws_w, ws_b = await _paired(app, clock)
    return room, ws_w, "white", _ping_raw(7)


async def _scenario_ping_offboard(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    return room, ws_w, "white", _ping_raw(None)


async def _scenario_ping_inflight(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    await handle_move(app, ws_w, room, "white", _move_raw("e2", "e4"))
    return room, ws_b, "black", _ping_raw(0)


async def _scenario_ping_strike(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    return room, ws_w, "white", _ping_raw(7)


async def _scenario_ping_directed(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    return room, ws_w, "white", _ping_raw(7)


async def _scenario_ping_gated(app, clock):
    room, ws_w, ws_b = await _live_room(app, clock)
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    await handle_ping(app, ws_w, room, "white", _ping_raw(7))
    await _strike_up_to_directive(app, clock, room, ws_w, "white", 7)
    return room, ws_w, "white", _ping_raw(7)


PING_SCENARIOS = {
    "ping": _scenario_ping,
    "ping_pending": _scenario_ping_pending,
    "ping_pregame": _scenario_ping_pregame,
    "ping_offboard": _scenario_ping_offboard,
    "ping_inflight": _scenario_ping_inflight,
    "ping_strike": _scenario_ping_strike,
    "ping_directed": _scenario_ping_directed,
    "ping_gated": _scenario_ping_gated,
}


@pytest.mark.parametrize("outcome", PING_OUTCOMES)
async def test_every_judged_heartbeat_answers_with_a_pong(app, clock, outcome):
    """The pong is what the client's is_server_silent escalation counts. A
    branch that returns without one turns a tolerated mismatch into a full
    reconnect a few heartbeats later -- exactly the outcome this whole file
    exists to avoid -- so every branch after a valid parse pongs."""
    room, ws, color, raw = await PING_SCENARIOS[outcome](app, clock)
    before = len(ws.of_type("pong"))

    out = await handle_ping(app, ws, room, color, raw)

    assert out == outcome
    assert len(ws.of_type("pong")) == before + 1


async def test_an_unparseable_heartbeat_is_refused_before_the_pong(app, clock):
    """The one branch that does NOT pong, and deliberately: a frame the server
    cannot read is not evidence that a client is alive and well."""
    room, ws_w, ws_b = await _live_room(app, clock)

    out = await handle_ping(app, ws_w, room, "white", "not json at all")

    assert out == "invalid_ping"
    assert ws_w.of_type("pong") == []


def _ping_outcome_words():
    """Every string literal handle_ping can return, read off the AST so a new
    branch cannot be added without joining the vocabulary. The whole returned
    expression is searched, not just a bare constant, so picking between two
    words on the way out still declares both."""
    source = read_source_without_docstrings(
        os.path.join(SERVER_ROOT, "handlers.py"))
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_ping")
    words = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                words.add(child.value)
    return words


def test_the_ping_outcome_vocabulary_is_closed():
    """The outcome word is this handler's only appearance in the production
    journal (`ws dispatch ... outcome=`), so it is the only way an operator can
    tell a tolerated mismatch from a directed one. A new word nobody documented
    would read as a mystery in the logs."""
    assert RESYNC_STABLE_MISMATCH_HEARTBEATS >= 2, \
        "ping_strike is unreachable if one mismatch is enough"
    assert _ping_outcome_words() == set(PING_OUTCOMES) | {"invalid_ping"}


class _HistoryCallVisitor(ast.NodeVisitor):
    """Walks a module and files every history-changing call under the def it
    is written in. A nested def is a def of its own, so the visitor descends
    into it under its own name rather than crediting the outer one -- plain
    ast.walk cannot express that, since skipping a nested FunctionDef in the
    loop body does not stop walk() from yielding everything inside it."""

    def __init__(self):
        """Start with no enclosing def and nothing filed."""
        self.sites = {}
        self._enclosing = None

    def _visit_def(self, node):
        """Make this def the enclosing one for everything it contains, then
        hand the name back to whatever def contains it."""
        outer, self._enclosing = self._enclosing, node.name
        self.generic_visit(node)
        self._enclosing = outer

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def

    def visit_Call(self, node):
        """File a call under the def it sits in when it can change a move
        history, then keep walking so nested calls are seen too."""
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) \
            else getattr(func, "id", None)
        if name in HISTORY_CALL_NAMES and self._enclosing is not None:
            self.sites.setdefault(self._enclosing, []).append(node.lineno)
        self.generic_visit(node)


def _history_call_sites(path):
    """Enclosing def name for every call that can change a Backend's move
    history, keyed by name because that is what a new site would be spelled as."""
    visitor = _HistoryCallVisitor()
    visitor.visit(ast.parse(read_source_without_docstrings(path), filename=path))
    return visitor.sites


def test_only_the_four_stamping_functions_touch_the_move_history():
    """The tolerance in this file is only as honest as the stamps it reads. A
    fifth place that grows or shrinks move_history without calling
    note_history_change would make every client look desynced for one heartbeat
    -- the exact bug this commit removes, reintroduced somewhere new.

    enqueue is the deliberate carve-out: it builds a brand new Backend into a
    room the pre-game gate is still protecting, so there is no client state to
    describe. reset_for_rematch clears the deque instead of appending to it."""
    assert os.path.isdir(SERVER_ROOT), f"expected a walkable package dir at {SERVER_ROOT}"
    found = {}
    scanned = 0
    for dirpath, _, filenames in os.walk(SERVER_ROOT):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            scanned += 1
            for fn_name, lines in _history_call_sites(
                    os.path.join(dirpath, name)).items():
                found.setdefault(fn_name, []).extend(lines)
    assert scanned >= 8, f"only scanned {scanned} files, guard root is likely wrong"
    assert set(found) == HISTORY_MUTATORS, (
        f"move_history is mutated outside the stamping functions: {sorted(found)}")
