"""The authoritative server skill-check state machine, driven through the real
handlers with a fake clock and recording websockets. This is the security core:
the move gate (pending/locked guards before any mutation), the one-shot-latch shot
handler (server-owned miss_count, deadline re-check), WIN/FAIL resolution, terminal
hygiene (resign/flag clear pending, takeback blocked, ping no-resync), the
single-resolver sweep deadline, the server-secret selection, and the resume payload.

The geometry seed is server-generated per check, so winning/failing tests READ the
stored pending.seed and solve for an elapsed the pure engine scores — they never
guess the seed (mirrors how the server itself adjudicates).
"""

import json

import pytest

from chessshootout.backend.pieces import PieceColor, PieceType
from chessshootout.backend.utils import Square, coord_from_square
from chessshootout.server.broadcasts import finalize_and_broadcast, resolve_skillcheck_fail
from fastapi.testclient import TestClient

from chessshootout.server.handlers import (
    handle_move, handle_ping, handle_skill_check_shot, handle_takeback_request,
    handle_takeback_response,
)
from chessshootout.server.protocol import (
    PROTOCOL_VERSION, SKILLCHECK_TARGET_MAX, Reason)
from chessshootout.server.rooms import PendingSkillCheck
from chessshootout.skillcheck import mole, online
from chessshootout.skillcheck.combo import (
    COMBO_DIRECTIONS, COMBO_MAX_WRONGS, COMBO_MIN_INTER_PRESS_MS)
from chessshootout.skillcheck.triggers import compute_facts
from chessshootout.skillcheck.types import SkillCheckKind, SkillCheckOutcome
from tests.helpers import fake_uuid4, make_backend, piece, sq
from tests.server.conftest import ALICE, BOB

WHEEL = SkillCheckKind.WHEEL
AIM = SkillCheckKind.AIM
WHACK = SkillCheckKind.WHACK
COMBO = SkillCheckKind.COMBO
ALL_KINDS = (WHEEL, AIM, WHACK, COMBO)


class RecordingWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

    def types(self):
        return [m["type"] for m in self.sent]

    def of_type(self, t):
        return [m for m in self.sent if m["type"] == t]


async def _pair(app):
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="A", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="B", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    return list(rooms._active.values())[0]


def _qxp_backend():
    return make_backend({
        sq(7, 4): piece(PieceType.KING, PieceColor.WHITE),
        sq(0, 4): piece(PieceType.KING, PieceColor.BLACK),
        sq(4, 3): piece(PieceType.QUEEN, PieceColor.WHITE),
        sq(3, 3): piece(PieceType.PAWN, PieceColor.BLACK),
    })


def _seed_for(backend, frm, to, want_kind, locks=None):
    locks = locks or set()
    for i in range(4000):
        secret = "secret-{}".format(i)
        if online.select_kind(secret, 0, backend, frm, to, locks) == want_kind:
            return secret
    raise AssertionError("no secret for {}".format(want_kind))


async def _capture_room(app, clock, kind, *, backend=None,
                        frm=Square(4, 3), to=Square(3, 3)):
    """The one paired-room factory every capture scenario builds on: a board with
    a legal capture, a secret brute-forced to select `kind` for that move, and a
    recording socket per side. `backend`/`frm`/`to` swap in the promotion and
    black-mover boards without re-typing the wiring."""
    room = await _pair(app)
    room.backend = _qxp_backend() if backend is None else backend
    room.first_move_at = clock()
    room.started_at = clock()
    room.skillcheck_secret = _seed_for(room.backend, frm, to, kind)
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    return room, ws_w, ws_b, frm, to


def _promo_capture_backend():
    # white pawn a7 captures black rook b8 -> a capturing promotion
    return make_backend({
        sq(7, 4): piece(PieceType.KING, PieceColor.WHITE),
        sq(0, 4): piece(PieceType.KING, PieceColor.BLACK),
        sq(1, 0): piece(PieceType.PAWN, PieceColor.WHITE),
        sq(0, 1): piece(PieceType.ROOK, PieceColor.BLACK),
    })


async def _promo_capture_room(app, clock, kind):
    return await _capture_room(app, clock, kind, backend=_promo_capture_backend(),
                               frm=Square(1, 0), to=Square(0, 1))


def _move_raw(frm, to, promotion=None):
    payload = {"type": "move", "from": coord_from_square(frm), "to": coord_from_square(to)}
    if promotion is not None:
        payload["promotion"] = promotion
    return json.dumps(payload)


def _win_elapsed(pending):
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    for e in range(int(online.SKILLCHECK_HUMAN_FLOOR_MS), int(online.SKILLCHECK_DEADLINE_MS)):
        if online.shot_wins(pending.kind, ch, e, pending.miss_count):
            return e
    raise AssertionError("no winning elapsed for the stored seed")


def _aim_miss_elapsed(pending):
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    for e in range(int(online.SKILLCHECK_HUMAN_FLOOR_MS), 2500):
        if (not online.shot_wins(AIM, ch, e, pending.miss_count)
                and not online.aim_expired(ch, e, pending.miss_count)):
            return e
    raise AssertionError("no aim-miss elapsed")


def _wheel_loss_elapsed(pending):
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    for e in range(int(online.SKILLCHECK_HUMAN_FLOOR_MS), int(online.SKILLCHECK_DEADLINE_MS)):
        if not online.shot_wins(WHEEL, ch, e, pending.miss_count):
            return e
    raise AssertionError("no losing wheel elapsed for the stored seed")


def _shot_raw(elapsed, direction=None, target=None):
    payload = {"type": "skill_check_shot", "client_elapsed_ms": elapsed}
    if direction is not None:
        payload["direction"] = direction
    if target is not None:
        payload["target_row"], payload["target_col"] = target
    return json.dumps(payload)


async def _fire(app, clock, room, color, elapsed, direction=None, target=None):
    """The single shot driver for all four kinds: parks the fake clock at the
    check-relative elapsed and posts the raw frame through the real handler.
    Omitted direction/target are omitted from the JSON, so a wheel/aim shot is
    byte-identical to the payload a wheel client actually sends."""
    clock.set((room.pending_skillcheck.start_ms + elapsed) / 1000.0)
    ws = app.state.connections.get_for_color(room, color)
    return await handle_skill_check_shot(app, ws, room, color,
                                         _shot_raw(elapsed, direction, target))


@pytest.mark.asyncio
async def test_quiet_move_applies_immediately_no_check(app, clock):
    room = await _pair(app)
    room.first_move_at = clock()
    ws = RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws)
    out = await handle_move(app, ws, room, "white", _move_raw(Square(6, 4), Square(4, 4)))
    assert out == "applied"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1


@pytest.mark.asyncio
async def test_capture_fires_check_and_does_not_mutate_the_board(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    out = await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert out == "skillcheck:wheel"
    assert room.pending_skillcheck is not None
    assert len(room.backend.move_history) == 0, "selection only probes; the board never mutated"


@pytest.mark.asyncio
async def test_capturing_underpromotion_values_the_promoted_piece(app, clock):
    # a capturing knight-underpromotion: value_diff must score as the PROMOTED piece
    # (pawn - knight = -2), proving handle_move forwards msg.promotion into
    # value_diff_for rather than scoring the captured rook (pawn - rook = -4).
    room, ws_w, ws_b, frm, to = await _promo_capture_room(app, clock, WHEEL)
    facts = compute_facts(room.backend, frm, to, room.skillcheck_locks)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to, promotion="n"))
    expected = online.value_diff_for(facts, "n")
    assert expected == -2, "pawn(1) - knight(3); the promoted piece, not the captured rook"
    assert room.pending_skillcheck.value_diff == expected
    req = ws_w.of_type("skill_check_required")[-1]
    spec = ws_b.of_type("skill_check_spectate")[-1]
    assert req["value_diff"] == expected == spec["value_diff"], "both wires carry the promo value"
    assert req["promotion"] == "n", "the underpromotion choice rides along to the client"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [WHEEL, WHACK])
async def test_placement_and_square_keys_never_leak_into_skillcheck_payloads(app, clock, kind):
    # board placement is render-only and server-authoritative geometry; no
    # 'placement'/'square' key may appear in the required, spectate, or resume
    # payloads, so it can never feed adjudication or leak the relocated dial
    # position. For whack the hole layout is derived server-side and must never
    # ride any wire either, nor may last_input_ms — the anti-mash gate is paced
    # on the server wall clock and knowing it would let a client time its bursts.
    # last_hit_pop is the deliberate exception: it is mover knowledge and rides
    # ONLY the resume wire (asserted below), never the live required/spectate pair.
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    live = ws_w.of_type("skill_check_required") + ws_b.of_type("skill_check_spectate")
    assert live, "the check fired and emitted payloads"
    resume = _resume_payload(app, room, "white").model_dump(by_alias=True)
    for p in live + [resume]:
        for key in ("placement", "square", "hole", "holes", "hole_squares",
                    "last_input_ms", "plies_ever"):
            assert key not in p, f"'{key}' must never appear on a skillcheck wire payload"
    for p in live:
        assert "last_hit_pop" not in p, "the live challenge pair never carries mover progress"
    assert resume["last_hit_pop"] == -1, "resume echoes it; nothing has been hit yet"


@pytest.mark.asyncio
async def test_required_to_mover_enriched_spectate_to_opponent(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert ws_w.of_type("skill_check_required"), "mover gets the playable challenge"
    assert not ws_w.of_type("skill_check_spectate")
    spec = ws_b.of_type("skill_check_spectate")
    assert spec and spec[0]["kind"] == "aim"
    assert spec[0]["seed"] == room.pending_skillcheck.seed, "the opponent mirrors the real seed"
    assert (spec[0]["from"], spec[0]["to"]) == ("d4", "d5"), "anchored at the move squares"
    req = ws_w.of_type("skill_check_required")[0]
    assert (req["seed"], req["value_diff"]) == (spec[0]["seed"], spec[0]["value_diff"]), \
        "both sides reconstruct the identical challenge"


@pytest.mark.asyncio
async def test_shot_relays_the_winning_position_to_the_opponent(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    elapsed = _win_elapsed(room.pending_skillcheck)
    await _fire(app, clock, room, "white", elapsed)
    relay = ws_b.of_type("skill_check_spectate_shot")
    assert relay and relay[0]["won"] is True
    assert relay[0]["miss_count"] == 0, "the pre-shot count the mover fired at"
    assert relay[0]["elapsed_ms"] == elapsed, "the server-adjudicated position, not raw client time"


@pytest.mark.asyncio
async def test_shot_relays_an_aim_miss_to_the_opponent(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", _aim_miss_elapsed(room.pending_skillcheck))
    assert out == "skillcheck_miss"
    relay = ws_b.of_type("skill_check_spectate_shot")
    assert relay and relay[0]["won"] is False
    assert relay[0]["miss_count"] == 0, "relayed at the pre-increment count"
    assert room.pending_skillcheck.miss_count == 1, "the server escalates after the relay"


@pytest.mark.asyncio
async def test_shot_relays_a_wheel_fail_to_the_opponent(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", _wheel_loss_elapsed(room.pending_skillcheck))
    assert out == "skillcheck_fail"
    relay = ws_b.of_type("skill_check_spectate_shot")
    assert relay and relay[0]["won"] is False


@pytest.mark.asyncio
async def test_shot_relay_is_skipped_when_the_opponent_is_gone(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    app.state.connections.remove(room.room_id, room.black.client_uuid, ws_b)
    out = await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert out.startswith("applied"), "the win still resolves with no opponent to relay to"
    assert not ws_b.of_type("skill_check_spectate_shot")


@pytest.mark.asyncio
async def test_aim_miss_relay_is_skipped_when_the_opponent_is_gone(app, clock):
    # the opp_ws-is-None relay skip is covered for a WIN; cover it for a MISS too:
    # an aim miss with the opponent disconnected must still resolve as a miss,
    # escalate the server miss_count, and never attempt a spectate relay or crash.
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    app.state.connections.remove(room.room_id, room.black.client_uuid, ws_b)
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", _aim_miss_elapsed(room.pending_skillcheck))
    assert out == "skillcheck_miss", "the miss resolves with no opponent to relay to"
    assert room.pending_skillcheck is not None, "an aim miss keeps the check pending"
    assert room.pending_skillcheck.miss_count == 1, "the server still escalates after a lone miss"
    assert not ws_b.of_type("skill_check_spectate_shot"), "no relay was attempted"


@pytest.mark.asyncio
async def test_move_while_pending_is_rejected_without_mutation(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    before = room.pending_skillcheck
    out = await handle_move(app, ws_w, room, "white", _move_raw(Square(7, 4), Square(6, 4)))
    assert out == "pending"
    assert room.pending_skillcheck is before, "the held check is untouched"
    assert len(room.backend.move_history) == 0
    assert ws_w.of_type("error")[-1]["reason"] == Reason.SKILLCHECK_PENDING


@pytest.mark.asyncio
async def test_winning_shot_applies_the_move_and_clears_pending(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert out == "applied"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1
    applied = ws_b.of_type("move_applied")[-1]
    assert applied["skill_check_kind"] == "wheel" and applied["skill_check_won"] is True


@pytest.mark.asyncio
async def test_failing_shot_locks_the_move_clears_pending_broadcasts_result(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", 50)  # below the human floor -> fail
    assert out == "skillcheck_fail"
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert len(room.backend.move_history) == 0
    res = ws_w.of_type("skill_check_result")[-1]
    assert res["won"] is False


@pytest.mark.asyncio
async def test_wheel_is_one_shot_a_non_winning_shot_fails(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    ch = online.challenge_from(WHEEL, room.pending_skillcheck.seed,
                               room.pending_skillcheck.value_diff)
    miss = next(e for e in range(120, 800) if not online.shot_wins(WHEEL, ch, e))
    out = await _fire(app, clock, room, "white", miss)
    assert out == "skillcheck_fail", "a wheel never stays pending after its single shot"


@pytest.mark.asyncio
async def test_late_shot_past_deadline_fails_even_if_geometry_would_win(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", int(online.SKILLCHECK_DEADLINE_MS) + 200)
    assert out == "skillcheck_fail"
    assert (frm, to) in room.skillcheck_locks


@pytest.mark.asyncio
async def test_aim_miss_keeps_pending_and_increments_server_miss_count(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", _aim_miss_elapsed(room.pending_skillcheck))
    assert out == "skillcheck_miss"
    assert room.pending_skillcheck is not None
    assert room.pending_skillcheck.miss_count == 1
    assert not ws_b.of_type("skill_check_result"), "a miss broadcasts nothing"


@pytest.mark.asyncio
async def test_aim_hit_after_a_miss_applies_the_move(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _aim_miss_elapsed(room.pending_skillcheck))
    out = await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert out == "applied"
    assert len(room.backend.move_history) == 1


@pytest.mark.asyncio
async def test_shot_with_no_pending_is_a_noop(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    out = await handle_skill_check_shot(app, ws_w, room, "white", '{"type":"skill_check_shot"}')
    assert out == "noop"


@pytest.mark.asyncio
async def test_opponent_shot_is_ignored_not_a_fail(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await handle_skill_check_shot(app, ws_b, room, "black", '{"type":"skill_check_shot"}')
    assert out == "noop"
    assert room.pending_skillcheck is not None, "the opponent cannot grief the mover's check"
    assert not ws_b.of_type("skill_check_result")


@pytest.mark.asyncio
async def test_pending_is_a_one_shot_latch(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    out = await handle_skill_check_shot(app, ws_w, room, "white", '{"type":"skill_check_shot"}')
    assert out == "noop", "a second shot after resolution finds pending cleared"
    assert len(room.backend.move_history) == 1, "the move applied exactly once"


@pytest.mark.asyncio
async def test_locked_move_is_rejected_outright_not_rerolled(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", 50)  # fail -> lock
    out = await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert out == "locked"
    assert room.pending_skillcheck is None, "the locked move never re-fires a check"
    assert ws_w.of_type("error")[-1]["reason"] == Reason.MOVE_LOCKED


@pytest.mark.asyncio
async def test_locks_clear_on_the_next_applied_ply(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", 50)
    assert room.skillcheck_locks
    await handle_move(app, ws_w, room, "white", _move_raw(Square(7, 4), Square(6, 4)))
    assert room.skillcheck_locks == set(), "an applied move clears the per-ply locks"


@pytest.mark.asyncio
async def test_resign_clears_pending_and_a_later_shot_is_noop(app, clock):
    from chessshootout.server.handlers import handle_resign
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await handle_resign(app, ws_w, room, "white", "{}")
    assert room.pending_skillcheck is None
    out = await handle_skill_check_shot(app, ws_w, room, "white", '{"type":"skill_check_shot"}')
    assert out == "noop"


@pytest.mark.asyncio
async def test_takeback_is_blocked_while_a_check_is_pending(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    room.takeback_offered_by = None
    out = await handle_takeback_request(app, ws_b, room, "black", "{}")
    assert out == "pending", "undo while a move is held would corrupt history"


@pytest.mark.asyncio
async def test_ping_emits_no_resync_while_pending(app, clock):
    """A held move legitimately sits at the same ply, so the mover's heartbeat
    matches and is answered as a plain caught-up ping. A mismatching one is not
    judged at all while a check is running -- the position is mid-flight."""
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert await handle_ping(app, ws_w, room, "white", '{"type":"ping","ply":0}') == "ping"
    out = await handle_ping(app, ws_w, room, "white", '{"type":"ping","ply":7}')
    assert out == "ping_pending"
    assert not ws_w.of_type("resync_directive")
    assert room.white.desync_active is False


@pytest.mark.asyncio
async def test_sweep_auto_fails_a_pending_check_at_the_deadline(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 100) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert room.result is None, "auto-fail locks the move; it does not end the game"
    assert ws_b.of_type("skill_check_result"), "the spectator is told the check failed"


@pytest.mark.asyncio
async def test_sweep_auto_fails_even_while_the_mover_is_disconnected(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    app.state.rooms.mark_disconnected(room.room_id, "white")
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 100) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None, "the deadline runs regardless of presence"


@pytest.mark.asyncio
async def test_sweep_leaves_a_live_pending_check_alone(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance(1.0)  # still well within the 5s window
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is not None


@pytest.mark.asyncio
async def test_movers_clock_keeps_running_during_a_held_check(app, clock):
    room = await _pair(app)
    room.backend = _qxp_backend()
    room.backend.setup_clock(60, 0, now_provider=clock)
    room.first_move_at = clock()
    frm, to = Square(4, 3), Square(3, 3)
    room.skillcheck_secret = _seed_for(room.backend, frm, to, WHEEL)
    ws_w, ws_b = RecordingWS(), RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    before = room.backend.clock.white_remaining
    clock.advance(2.0)
    room.backend.tick_clock()
    assert room.backend.clock.white_remaining < before, "still white's turn; their clock ticks"


@pytest.mark.asyncio
async def test_selection_secret_never_appears_in_any_client_payload(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    secret = room.skillcheck_secret
    assert secret, "the room has a secret"
    blob = json.dumps(ws_w.sent + ws_b.sent)
    assert secret not in blob, "the selection secret is never sent to either client"


@pytest.mark.asyncio
async def test_secret_is_fresh_per_room_at_pairing(app, clock):
    await _pair(app)
    await app.state.rooms.enqueue(
        client_uuid=fake_uuid4(3), nickname="C", session_token="tc",
        time_minutes=5, increment_seconds=0, side_preference="white")
    await app.state.rooms.enqueue(
        client_uuid=fake_uuid4(4), nickname="D", session_token="td",
        time_minutes=5, increment_seconds=0, side_preference="black")
    rooms = [r for r in app.state.rooms._active.values()]
    secrets_seen = {r.skillcheck_secret for r in rooms}
    assert len(secrets_seen) == len(rooms) >= 2, "each room gets its own secret"
    assert all(s for s in secrets_seen)


@pytest.mark.asyncio
async def test_now_ms_is_seconds_times_1000(app, clock):
    clock.set(12.5)
    assert app.state.now_ms() == 12500.0


@pytest.mark.asyncio
async def test_pending_start_ms_is_stamped_in_milliseconds(app, clock):
    clock.set(7.0)
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert room.pending_skillcheck.start_ms == 7000.0
    assert room.pending_skillcheck.expires_at_ms == 7000.0 + online.SKILLCHECK_DEADLINE_MS


def _resume_payload(app, room, color):
    from chessshootout.server.routes_http import _pending_skillcheck_wire
    return _pending_skillcheck_wire(room, app.state.now_ms)


@pytest.mark.asyncio
async def test_resume_wire_carries_live_pending_with_elapsed(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance(1.8)
    wire = _resume_payload(app, room, "white")
    assert wire is not None
    assert wire.kind == "aim"
    assert wire.seed == room.pending_skillcheck.seed
    assert wire.elapsed_ms == pytest.approx(1800.0, abs=1.0), "the timer never restarts from zero"


@pytest.mark.asyncio
async def test_resume_wire_is_none_after_the_check_resolves(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", 50)  # fail
    assert _resume_payload(app, room, "white") is None, "a resolved check leaves a free turn"


@pytest.mark.asyncio
async def test_resume_seed_is_identical_across_repeated_resumes(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    a = _resume_payload(app, room, "white")
    b = _resume_payload(app, room, "white")
    assert a.seed == b.seed == room.pending_skillcheck.seed, "resume never re-rolls the seed"


@pytest.mark.asyncio
async def test_resolve_fail_is_idempotent(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    first = await resolve_skillcheck_fail(app.state.rooms, app.state.connections, room)
    second = await resolve_skillcheck_fail(app.state.rooms, app.state.connections, room)
    assert first is not None and second is None, "a second resolve finds pending already cleared"


async def _shoot_claiming(app, clock, room, color, *, arrives_at_ms, claims_ms,
                          direction=None, target=None):
    """A shot that physically lands at arrives_at_ms but reports claims_ms as the elapsed."""
    clock.set((room.pending_skillcheck.start_ms + arrives_at_ms) / 1000.0)
    ws = app.state.connections.get_for_color(room, color)
    return await handle_skill_check_shot(app, ws, room, color,
                                         _shot_raw(claims_ms, direction, target))


@pytest.mark.asyncio
async def test_honest_shot_is_judged_at_the_clients_rendered_elapsed(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    win = _win_elapsed(room.pending_skillcheck)
    out = await _fire(app, clock, room, "white", win)
    assert out.startswith("applied"), "the server judges the exact moment the player saw"


@pytest.mark.asyncio
async def test_network_lag_inside_the_bound_does_not_steal_a_clean_hit(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    win = _win_elapsed(room.pending_skillcheck)
    # claims the visual win moment, but the packet lands 80ms later — still inside [raw-bound, raw]
    out = await _shoot_claiming(app, clock, room, "white", arrives_at_ms=win + 80, claims_ms=win)
    assert out.startswith("applied"), "latency inside the bound never steals a clean hit"


def _clamp_discriminating_arrival(pending):
    """An arrival time where the clamp floor WINS and the raw arrival LOSES.

    Picking a fixed arrival and asserting whatever the engine happens to score
    made this test pass ~92% of runs on its own: for most random arcs both the
    claimed 0 ms (sub-floor) and the raw arrival lose, so a server that ignored
    the clamp entirely produced the same verdict. Only an arrival whose
    raw-minus-bound moment is a WIN while the raw moment is a LOSS separates
    'judged at the clamp floor' from 'judged at claimed' and from 'judged at raw'.
    """
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    bound = int(online.SKILLCHECK_LAG_BOUND_MS)
    floor = int(online.SKILLCHECK_HUMAN_FLOOR_MS)
    for arrival in range(floor + bound + 1, int(online.SKILLCHECK_DEADLINE_MS)):
        if (online.shot_wins(WHEEL, ch, arrival - bound, 0)
                and not online.shot_wins(WHEEL, ch, arrival, 0)):
            return arrival
    raise AssertionError("no arrival separates the clamp floor from the raw arrival")


@pytest.mark.asyncio
async def test_a_too_early_claim_is_clamped_to_the_lag_bound(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    arrival = _clamp_discriminating_arrival(room.pending_skillcheck)

    out = await _shoot_claiming(app, clock, room, "white",
                                arrives_at_ms=arrival, claims_ms=0.0)

    assert out.startswith("applied"), \
        "a forged-early claim is judged at raw-bound, never at 0 and never at raw"


@pytest.mark.asyncio
async def test_required_message_carries_the_move_squares(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    req = ws_w.of_type("skill_check_required")[-1]
    assert req["from"] == coord_from_square(frm)
    assert req["to"] == coord_from_square(to)
    assert req["promotion"] is None


@pytest.mark.asyncio
async def test_pending_wire_carries_squares_color_and_promotion(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    wire = _resume_payload(app, room, "white")
    assert wire.from_sq == coord_from_square(frm)
    assert wire.to_sq == coord_from_square(to)
    assert wire.color == "white"
    assert wire.promotion is None


@pytest.mark.asyncio
async def test_resume_wire_omits_a_pending_past_its_deadline(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 50) / 1000.0)
    assert room.pending_skillcheck is not None, "the sweep has not run yet"
    assert _resume_payload(app, room, "white") is None, \
        "a reconnect must not re-hand an already-dead check"


@pytest.mark.asyncio
async def test_handle_move_agrees_with_resume_at_the_expiry_boundary(app, clock):
    """/resume's view (no pending once now_ms > expires_at_ms) and handle_move's
    move gate must agree at the identical boundary instant, even inside the
    up-to-100ms window before the sweep next ticks. handle_move now resolves
    the expired check inline (the same resolve_skillcheck_fail the sweep would
    use) instead of rejecting a retry of the SAME move as still-pending -- the
    retry is judged fresh, against the now-current lock, and rejected as
    locked rather than pending."""
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 50) / 1000.0)
    assert room.pending_skillcheck is not None, "the sweep has not run yet"
    assert _resume_payload(app, room, "white") is None, "resume already treats it as gone"

    out = await handle_move(app, ws_w, room, "white", _move_raw(frm, to))

    assert out == "locked", "the expired check resolves as a proper fail before the retry is judged"
    assert room.pending_skillcheck is None
    assert room.skillcheck_log == [SkillCheckOutcome(1, "wheel", False, "Qxd5")], \
        "the miss consequence is still recorded, not silently skipped"
    assert ws_b.of_type("skill_check_result"), "the opponent still hears the check resolved"


@pytest.mark.asyncio
async def test_handle_move_applies_a_different_move_after_an_expired_unswept_pending(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 50) / 1000.0)

    out = await handle_move(app, ws_w, room, "white", _move_raw(Square(7, 4), Square(6, 4)))

    assert out == "applied", "handle_move now agrees with resume instead of rejecting as pending"
    assert len(room.backend.move_history) == 1


class _FinalizingWS(RecordingWS):
    """A socket that, on receiving the inline expired-check resolve broadcast,
    simulates a concurrent resign/flag finalize landing during
    resolve_skillcheck_fail's await -- exactly the interleave two real handler
    tasks on the shared loop would produce."""

    def __init__(self, rooms, connections, room):
        super().__init__()
        self._rooms = rooms
        self._connections = connections
        self._room = room
        self._fired = False

    async def send_json(self, payload):
        self.sent.append(payload)
        if not self._fired and payload.get("type") == "skill_check_result":
            self._fired = True
            await finalize_and_broadcast(self._rooms, self._connections, self._room,
                                         Reason.RESIGNATION, winner_color="black")


@pytest.mark.asyncio
async def test_move_after_expired_pending_bails_when_the_game_finalizes_mid_resolve(app, clock):
    # REGRESSION: handle_move resolves an expired-but-unswept check inline via
    # resolve_skillcheck_fail, which awaits a broadcast. If a concurrent finalize
    # (resign/flag) lands during that await, handle_move must re-read room.result
    # and bail -- never apply a fresh ply onto an already-finalized game.
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    finalizing = _FinalizingWS(app.state.rooms, app.state.connections, room)
    app.state.connections.add(room.room_id, room.black.client_uuid, finalizing)
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 50) / 1000.0)  # expired, unswept

    out = await handle_move(app, ws_w, room, "white", _move_raw(Square(7, 4), Square(6, 4)))

    assert out == "already_over", "a finalize during the inline resolve preempts the retry"
    assert room.result == (Reason.RESIGNATION, "black")
    assert len(room.backend.move_history) == 0, "no ply lands on an already-finalized game"
    assert not ws_w.of_type("move_applied"), "no move_applied is broadcast after the result"


@pytest.mark.asyncio
async def test_handle_move_still_blocks_a_genuinely_live_pending_check(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    before = room.pending_skillcheck
    out = await handle_move(app, ws_w, room, "white", _move_raw(Square(7, 4), Square(6, 4)))
    assert out == "pending"
    assert room.pending_skillcheck is before


@pytest.mark.asyncio
async def test_shot_and_sweep_in_the_same_tick_resolve_exactly_once(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", int(online.SKILLCHECK_DEADLINE_MS) + 10)
    await app.state.sweep.step_skillcheck_deadline()
    assert out == "skillcheck_fail"
    assert len(ws_b.of_type("skill_check_result")) == 1, "exactly one fail broadcast"


@pytest.mark.asyncio
async def test_aim_shot_and_sweep_at_the_shrink_boundary_resolve_exactly_once(app, clock):
    # the AIM analogue of the same-tick race: the piece shrinks to nothing (an aim
    # fail) at the SAME tick a shot arrives, well before the 5s deadline. The shot
    # handler resolves first; resolve_skillcheck_fail is a latch, so the trailing
    # sweep finds pending already cleared and adds NO second broadcast.
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    pending.miss_count = 8  # heavy misses -> the victim vanishes long before 5s
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    gone_at = next(e for e in range(200, int(online.SKILLCHECK_DEADLINE_MS))
                   if online.aim_expired(ch, e, pending.miss_count))
    assert gone_at < online.SKILLCHECK_DEADLINE_MS, "the shrink outruns the 5s deadline"
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", gone_at)  # clock now sits at gone_at
    await app.state.sweep.step_skillcheck_deadline()
    assert out == "skillcheck_fail", "a shot once the piece is gone is a fail, and it wins the race"
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert len(ws_b.of_type("skill_check_result")) == 1, "the latch yields exactly one fail"
    assert room.skillcheck_log == [SkillCheckOutcome(1, "aim", False, "Qxd5")], \
        "recorded once, by the shot resolver"


class _RacingSpectatorWS(RecordingWS):
    """A spectator socket that runs `racer` the instant the spectate-shot lands on
    it -- the one await inside the shot handler, and therefore the only window a
    concurrent resolver (deadline sweep, resign/flag finalize) can slip into. Fires
    once so the racer's own broadcast back to this socket cannot recurse."""

    def __init__(self, racer):
        super().__init__()
        self._racer = racer
        self._fired = False

    async def send_json(self, payload):
        self.sent.append(payload)
        if not self._fired and payload.get("type") == "skill_check_spectate_shot":
            self._fired = True
            await self._racer()


@pytest.mark.asyncio
async def test_a_won_shot_is_claimed_before_the_relay_so_a_sweep_cannot_steal_it(app, clock):
    """REGRESSION: the spectate-shot relay is the shot handler's only await, and
    the deadline sweep runs on the same loop. A whack forged one whiff past the cap
    is is_dead -- so the sweep WILL fail it -- while a shot on an up pop still wins:
    the exact overlap where the two resolvers disagree. The handler therefore claims
    the pending (nulls it) BEFORE relaying, so the sweep that runs during the relay
    finds nothing to resolve and the mirrored won=True is never contradicted. With
    the claim after the relay the opponent saw won=True, then a fail broadcast, and
    the capture never landed."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    pending.progress = ch.hits_required - 1
    pending.miss_count = mole.MOLE_MAX_WHIFFS
    racing = _RacingSpectatorWS(app.state.sweep.step_skillcheck_deadline)
    app.state.connections.add(room.room_id, room.black.client_uuid, racing)

    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                      target=_hole_center(holes, ch.pops[0].hole))

    assert racing.of_type("skill_check_spectate_shot")[-1]["won"] is True
    assert out == "applied", "the claimed win survives the sweep racing in mid-relay"
    assert len(room.backend.move_history) == 1
    assert room.skillcheck_log == [SkillCheckOutcome(1, "whack", True, "Qxd5")]
    assert not racing.of_type("skill_check_result"), \
        "no fail contradicts the won=True the opponent already mirrored"
    assert (frm, to) not in room.skillcheck_locks


@pytest.mark.asyncio
async def test_sweep_fails_an_aim_check_when_the_piece_has_shrunk(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    pending.miss_count = 8  # heavy misses -> the victim shrinks to 0 well before 5s
    ch = online.challenge_from(pending.kind, pending.seed, pending.value_diff)
    gone_at = next(e for e in range(200, int(online.SKILLCHECK_DEADLINE_MS))
                   if online.aim_expired(ch, e, pending.miss_count))
    assert gone_at < online.SKILLCHECK_DEADLINE_MS, "the shrink outruns the 5s deadline"
    clock.set((pending.start_ms + gone_at + 1) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None, "the fail lands when the piece is gone, not at 5s"
    assert (frm, to) in room.skillcheck_locks


@pytest.mark.asyncio
async def test_sweep_does_not_early_fail_an_aim_check_while_the_piece_remains(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, AIM)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance(0.5)  # piece still large, well inside the deadline
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is not None, "an on-screen aim check is not swept early"


@pytest.mark.asyncio
async def test_won_check_appends_an_outcome_to_the_server_log(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert room.skillcheck_log == [SkillCheckOutcome(1, "wheel", True, "Qxd5")]


@pytest.mark.asyncio
async def test_failed_check_appends_a_fail_outcome_with_the_whiffed_san(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _wheel_loss_elapsed(room.pending_skillcheck))
    assert room.skillcheck_log == [SkillCheckOutcome(1, "wheel", False, "Qxd5")]
    assert len(room.backend.move_history) == 0, "a failed check lands no ply"


@pytest.mark.asyncio
async def test_sweep_resolved_fail_also_records_the_outcome(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await resolve_skillcheck_fail(app.state.rooms, app.state.connections, room)
    assert room.skillcheck_log == [SkillCheckOutcome(1, "wheel", False, "Qxd5")]


@pytest.mark.asyncio
async def test_a_pending_check_is_absent_from_the_log_until_it_resolves(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert room.pending_skillcheck is not None
    assert room.skillcheck_log == [], "an unresolved check is never recorded"
    await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert len(room.skillcheck_log) == 1, "recorded exactly once, at resolution"


@pytest.mark.asyncio
async def test_resume_endpoint_carries_the_skillcheck_log(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    resp = TestClient(app).post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id, "session_token": "ta"})
    assert resp.status_code == 200
    assert resp.json()["skillcheck_log"] == [
        {"ply": 1, "kind": "wheel", "won": True, "san": "Qxd5"}]


@pytest.mark.asyncio
async def test_takeback_drops_the_outcomes_for_the_undone_ply(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    await _fire(app, clock, room, "white", _win_elapsed(room.pending_skillcheck))
    assert len(room.skillcheck_log) == 1 and len(room.backend.move_history) == 1
    room.takeback_offered_by = "white"
    raw = json.dumps({"type": "takeback_response", "accept": True})
    await handle_takeback_response(app, ws_b, room, "black", raw)
    assert room.skillcheck_log == [], "the undone ply's outcome is gone"
    assert len(room.backend.move_history) == 0


@pytest.mark.asyncio
async def test_reset_for_rematch_clears_the_skillcheck_log(app, clock):
    room = await _pair(app)
    room.result = ("resignation", "white")
    room.skillcheck_log = [SkillCheckOutcome(1, "wheel", True, "Qxd5")]
    room.skillcheck_locks = {(Square(4, 3), Square(3, 3))}
    app.state.rooms.reset_for_rematch(room.room_id)
    assert room.skillcheck_log == []
    assert room.skillcheck_locks == set()


@pytest.mark.asyncio
async def test_handle_move_stamps_the_tc_capped_deadline(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    assert pending.deadline_ms == online.SKILLCHECK_DEADLINE_MS, "5+0 -> the 5s base"
    assert pending.expires_at_ms == pending.start_ms + pending.deadline_ms
    req = ws_w.of_type("skill_check_required")[0]
    spec = ws_b.of_type("skill_check_spectate")[0]
    assert req["deadline_ms"] == pending.deadline_ms == spec["deadline_ms"]


@pytest.mark.asyncio
async def test_a_short_time_control_shrinks_the_check_deadline(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    room.time_minutes = 0.5  # 30s game -> 10% = 3s, below the 5s base
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    assert pending.deadline_ms == 3000.0
    assert pending.expires_at_ms == pending.start_ms + 3000.0
    assert ws_w.of_type("skill_check_required")[0]["deadline_ms"] == 3000.0


@pytest.mark.asyncio
async def test_a_shot_past_the_capped_deadline_fails_even_under_the_base(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHEEL)
    room.time_minutes = 0.5  # deadline 3000ms
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _fire(app, clock, room, "white", 3500)  # past 3000 cap, under 5000 base
    assert out == "skillcheck_fail"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 0, "the move never lands past the capped deadline"
    assert room.skillcheck_log == [SkillCheckOutcome(1, "wheel", False, "Qxd5")]


def _full_challenge(pending):
    """The exact challenge the server adjudicates against: full params, incl. the
    deadline and captured_value the two new kinds scale by."""
    return online.challenge_from(pending.kind, pending.seed, pending.value_diff,
                                 pending.deadline_ms, pending.captured_value)


def _holes_for(room):
    """Rebuild the whack hole layout independently of the tuple the server cached
    at mint: seeded off the stored pending, anchored at the capture square,
    avoiding every occupied square of the room's authoritative backend. The
    occupancy scan goes through the production derivation, so a change there is
    caught here instead of being mirrored by a private copy."""
    pending = room.pending_skillcheck
    backend = room.backend
    return mole.hole_squares(pending.seed, pending.captured_value,
                             (pending.to_sq.row, pending.to_sq.col),
                             mole.occupied_squares(backend.state, backend.SIZE),
                             backend.SIZE)


def _pop_mid(ch, index):
    pop = ch.pops[index]
    return int((pop.t_up_ms + pop.t_down_ms) / 2)


def _hole_center(holes, hole):
    row, col = holes[hole]
    return (row + 0.5, col + 0.5)


async def _whack_room(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, WHACK)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    return room, ws_w, ws_b, frm, to, pending, _full_challenge(pending), _holes_for(room)


def _qxp_black_backend():
    return make_backend({
        sq(7, 4): piece(PieceType.KING, PieceColor.WHITE),
        sq(0, 4): piece(PieceType.KING, PieceColor.BLACK),
        sq(3, 3): piece(PieceType.QUEEN, PieceColor.BLACK),
        sq(4, 3): piece(PieceType.PAWN, PieceColor.WHITE),
    }, turn=PieceColor.BLACK)


async def _black_whack_room(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(
        app, clock, WHACK, backend=_qxp_black_backend(),
        frm=Square(3, 3), to=Square(4, 3))
    await handle_move(app, ws_b, room, "black", _move_raw(frm, to))
    pending = room.pending_skillcheck
    return room, ws_w, ws_b, frm, to, pending, _full_challenge(pending), _holes_for(room)


async def _combo_room(app, clock):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, COMBO)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    pending = room.pending_skillcheck
    return room, ws_w, ws_b, frm, to, pending, _full_challenge(pending)


@pytest.mark.asyncio
async def test_gated_capture_mints_the_captured_value_into_pending_and_both_wires(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    assert pending.captured_value == 1, "queen takes pawn: the captured piece's value"
    req = ws_w.of_type("skill_check_required")[-1]
    spec = ws_b.of_type("skill_check_spectate")[-1]
    assert req["captured_value"] == 1 == spec["captured_value"]
    assert req["kind"] == "whack" == spec["kind"]


@pytest.mark.asyncio
async def test_whack_true_position_hits_up_to_quota_apply_the_move(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    required = ch.hits_required
    assert required >= 2, "a 5+0 whack demands multiple hits"
    outs = []
    for i in range(required):
        outs.append(await _fire(app, clock, room, "white", _pop_mid(ch, i),
                                target=_hole_center(holes, ch.pops[i].hole)))
    assert outs[:-1] == ["skillcheck_hit"] * (required - 1), \
        "pre-quota hits keep the check pending without a resolution"
    assert outs[-1] == "applied"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1
    applied = ws_b.of_type("move_applied")[-1]
    assert applied["skill_check_kind"] == "whack" and applied["skill_check_won"] is True
    assert room.skillcheck_log == [SkillCheckOutcome(1, "whack", True, "Qxd5")]


@pytest.mark.asyncio
async def test_black_mover_whack_pit_centres_hit_up_to_quota_and_apply(app, clock):
    """Pit-square centres sit 0.30 rows off the oval centre under EITHER
    orientation (0.30 < the 0.52 half-height), so a black mover shooting dead
    centre must win exactly like a white mover does."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _black_whack_room(app, clock)
    assert pending.color == "black"
    outs = []
    for i in range(ch.hits_required):
        outs.append(await _fire(app, clock, room, "black", _pop_mid(ch, i),
                                target=_hole_center(holes, ch.pops[i].hole)))
    assert outs[:-1] == ["skillcheck_hit"] * (ch.hits_required - 1)
    assert outs[-1] == "applied"
    assert len(room.backend.move_history) == 1
    applied = ws_w.of_type("move_applied")[-1]
    assert applied["skill_check_kind"] == "whack" and applied["skill_check_won"] is True


@pytest.mark.asyncio
async def test_server_mirrors_the_whack_oval_for_a_black_mover(app, clock):
    """The asymmetric pin: a black mover renders their own colour at the bottom,
    so the popped piece rises toward HIGHER board rows on their screen and the
    server must adjudicate the CY lift mirrored below the pit row. (row + 0.8)
    is the mirrored oval's exact centre -- a hit for black, while for a white
    mover the same offset is 0.6 rows past the unflipped centre: a miss."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _black_whack_room(app, clock)
    row, col = holes[ch.pops[0].hole]
    out = await _fire(app, clock, room, "black", _pop_mid(ch, 0),
                      target=(row + 0.8, col + 0.5))
    assert out == "skillcheck_hit" and pending.progress == 1
    below = await _fire(app, clock, room, "black", _pop_mid(ch, 1),
                        target=(holes[ch.pops[1].hole][0] + 0.2,
                                holes[ch.pops[1].hole][1] + 0.5))
    assert below == "skillcheck_miss", \
        "the unflipped centre offset is 0.6 rows off the mirrored oval for black"


@pytest.mark.asyncio
async def test_white_mover_keeps_the_unflipped_whack_oval(app, clock):
    """The white half of the asymmetric pin, in its own room: (row + 0.8) must
    stay a miss and (row + 0.2) the exact oval centre."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    row, col = holes[ch.pops[0].hole]
    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                      target=(row + 0.8, col + 0.5))
    assert out == "skillcheck_miss" and pending.progress == 0
    assert pending.miss_count == 1
    out2 = await _fire(app, clock, room, "white", _pop_mid(ch, 0) + 200,
                       target=(row + 0.2, col + 0.5))
    assert out2 == "skillcheck_hit", "the white mover's oval centre stays 0.30 rows ABOVE"


@pytest.mark.asyncio
async def test_the_shot_handler_reads_the_shared_orientation_helper(app, clock, monkeypatch):
    """The handler's flipped flag must be the RETURN VALUE of
    online.adjudicated_flipped(pending.color) -- the same importable rule the
    client normalises its wire target with -- not a re-inlined color comparison.
    handlers.py binds the online MODULE (from chessshootout.skillcheck import
    online) and resolves the attribute at call time, so patching the helper on
    that module and demanding shot_wins receives the stub's unique sentinel BY
    IDENTITY pins the routing: an inline `pending.color == "black"` would hand
    shot_wins a plain True instead. The sentinel is truthy, so the real
    adjudication still scores the black mover's mirrored-oval centre as the
    exact hit the real helper (pinned below to return True for black) yields."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _black_whack_room(app, clock)
    real_flipped = online.adjudicated_flipped
    sentinel = object()
    asked = []

    def fake_flipped(color):
        asked.append(color)
        return sentinel

    seen = []
    real_shot = online.shot_wins

    def spy(*args, **kwargs):
        seen.append(kwargs["flipped"])
        return real_shot(*args, **kwargs)

    monkeypatch.setattr(online, "adjudicated_flipped", fake_flipped)
    monkeypatch.setattr(online, "shot_wins", spy)
    row, col = holes[ch.pops[0].hole]
    out = await _fire(app, clock, room, "black", _pop_mid(ch, 0),
                      target=(row + 0.8, col + 0.5))
    assert out == "skillcheck_hit" and pending.progress == 1
    assert asked == [pending.color] == ["black"]
    assert len(seen) == 1 and seen[0] is sentinel, \
        "the handler forwards the helper's return itself, not a re-derived bool"
    assert real_flipped("black") is True, "the shared rule the sentinel stands in for"


@pytest.mark.asyncio
async def test_whack_hit_at_a_wrong_hole_increments_miss_only(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    wrong_hole = next(h for h in range(ch.hole_count) if h != ch.pops[0].hole)
    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                      target=_hole_center(holes, wrong_hole))
    assert out == "skillcheck_miss", "an empty hole while another pop is up is a plain miss"
    assert pending.miss_count == 1 and pending.progress == 0
    assert room.pending_skillcheck is pending
    assert not ws_b.of_type("skill_check_result")


@pytest.mark.asyncio
async def test_whack_second_shot_in_the_same_up_window_credits_once(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    center = _hole_center(holes, ch.pops[0].hole)
    first = await _fire(app, clock, room, "white", _pop_mid(ch, 0), target=center)
    second = await _fire(app, clock, room, "white", _pop_mid(ch, 0) + 200, target=center)
    assert first == "skillcheck_hit" and pending.progress == 1
    assert second == "skillcheck_miss", "the same pop can never be credited twice"
    assert pending.progress == 1 and pending.miss_count == 1


@pytest.mark.asyncio
async def test_whack_shots_inside_the_min_input_gap_are_silent_noops(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    ws_b.sent.clear()
    wrong_hole = next(h for h in range(ch.hole_count) if h != ch.pops[0].hole)
    mid = _pop_mid(ch, 0)
    center = _hole_center(holes, ch.pops[0].hole)
    first = await _fire(app, clock, room, "white", mid,
                        target=_hole_center(holes, wrong_hole))
    assert first == "skillcheck_miss" and pending.miss_count == 1
    throttled = await _fire(app, clock, room, "white", mid + 40, target=center)
    assert throttled == "noop", "a shot 40ms after the last input is autofire, dropped whole"
    assert pending.miss_count == 1 and pending.progress == 0, "no state change at all"
    assert len(ws_b.of_type("skill_check_spectate_shot")) == 1, "no relay for the dropped shot"
    third = await _fire(app, clock, room, "white", mid + 200, target=center)
    assert third == "skillcheck_hit", "a legitimately spaced follow-up still lands"
    assert pending.progress == 1


def _spaced_miss_target(ch, holes):
    """A guaranteed-whiff position for shots taken around the first pop's window:
    a hole neither of the first two pops uses, so no pop can be up there whatever
    the exact elapsed lands on."""
    wrong_hole = next(h for h in range(ch.hole_count)
                      if h not in (ch.pops[0].hole, ch.pops[1].hole))
    return _hole_center(holes, wrong_hole)


@pytest.mark.asyncio
async def test_whack_third_spaced_whiff_fails_and_locks_the_move(app, clock):
    """The whiff cap in the live shot path: three properly spaced on-board misses
    (valid coords, wrong hole) fail the check the instant the 3rd lands — the
    handler's terminal probe runs at miss_count + 1, so the whiff being processed
    counts. The schedule is untouched (quota still reachable), proving the cap is
    what killed it, and each whiff relayed to the spectator at the pre-increment
    count exactly like every other miss."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    target = _spaced_miss_target(ch, holes)
    mid = _pop_mid(ch, 0)
    outs = [await _fire(app, clock, room, "white", mid + 200 * i, target=target)
            for i in range(mole.MOLE_MAX_WHIFFS)]
    assert outs == ["skillcheck_miss"] * (mole.MOLE_MAX_WHIFFS - 1) + ["skillcheck_fail"]
    assert not ch.quota_unreachable(mid + 200 * (mole.MOLE_MAX_WHIFFS - 1) + 1, 0, -1), \
        "the pops alone were still winnable — only the whiff cap ended it"
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert len(room.backend.move_history) == 0
    assert room.skillcheck_log == [SkillCheckOutcome(1, "whack", False, "Qxd5")]
    relays = ws_b.of_type("skill_check_spectate_shot")
    assert [r["miss_count"] for r in relays] == list(range(mole.MOLE_MAX_WHIFFS)), \
        "every whiff relays once, at the pre-increment count the mover fired at"
    assert all(r["won"] is False for r in relays)
    assert ws_b.of_type("skill_check_result")[-1]["won"] is False, "the opponent hears the fail"


@pytest.mark.asyncio
async def test_whack_two_whiffs_then_true_hits_still_win(app, clock):
    """The cap is >= 3: two whiffs spend no part of the hit quota and leave the
    run fully winnable through the unchanged hit path."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    target = _spaced_miss_target(ch, holes)
    mid = _pop_mid(ch, 0)
    for i in range(mole.MOLE_MAX_WHIFFS - 1):
        assert await _fire(app, clock, room, "white", mid + 100 * i,
                           target=target) == "skillcheck_miss"
    assert pending.miss_count == mole.MOLE_MAX_WHIFFS - 1, "one whiff short of the cap"
    outs = [await _fire(app, clock, room, "white", mid + 300,
                        target=_hole_center(holes, ch.pops[0].hole))]
    for i in range(1, ch.hits_required):
        outs.append(await _fire(app, clock, room, "white", _pop_mid(ch, i),
                                target=_hole_center(holes, ch.pops[i].hole)))
    assert outs == ["skillcheck_hit"] * (ch.hits_required - 1) + ["applied"]
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1
    applied = ws_b.of_type("move_applied")[-1]
    assert applied["skill_check_kind"] == "whack" and applied["skill_check_won"] is True


@pytest.mark.asyncio
async def test_throttled_whack_shots_never_advance_the_whiff_cap(app, clock):
    """The anti-mash gate sits BEFORE the whiff bookkeeping: a burst of too-fast
    shots at two-whiffs-from-death is dropped whole — no third whiff, no fail, no
    relay, no gate movement — so autofire can neither cheat the cap nor trip it."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    target = _spaced_miss_target(ch, holes)
    mid = _pop_mid(ch, 0)
    assert await _fire(app, clock, room, "white", mid, target=target) == "skillcheck_miss"
    assert await _fire(app, clock, room, "white", mid + 100,
                       target=target) == "skillcheck_miss"
    assert pending.miss_count == mole.MOLE_MAX_WHIFFS - 1, "one whiff from death"
    ws_b.sent.clear()
    for burst in range(1, 6):
        assert await _fire(app, clock, room, "white", mid + 100 + burst * 10,
                           target=target) == "noop", "autofire inside the 80ms gate drops whole"
    assert room.pending_skillcheck is pending, "no throttled shot reached the terminal probe"
    assert pending.miss_count == mole.MOLE_MAX_WHIFFS - 1, "and none advanced the cap"
    assert not ws_b.of_type("skill_check_spectate_shot"), "fully silent: no relay either"
    assert await _fire(app, clock, room, "white", mid + 300,
                       target=_hole_center(holes, ch.pops[0].hole)) == "skillcheck_hit", \
        "an honestly spaced follow-up still lands after the burst"


@pytest.mark.asyncio
async def test_forged_client_elapsed_buys_no_spacing_past_the_anti_mash_gate(app, clock):
    """The gate is paced on the ADJUDICATED elapsed, and the adjudication is capped
    at the real arrival (min(max(claimed, raw - 200ms), raw)). That ceiling is what
    keeps a crafted client honest: two sends in the same instant adjudicate to the
    same elapsed however much separation the second one claims, so the sustained
    input rate can never exceed the real one. Same-instant sends must always
    throttle: the second shot is dropped whole, with no relay, no miss, and no gate
    movement."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    wrong_hole = next(h for h in range(ch.hole_count) if h != ch.pops[0].hole)
    mid = _pop_mid(ch, 0)
    first = await _fire(app, clock, room, "white", mid,
                        target=_hole_center(holes, wrong_hole))
    assert first == "skillcheck_miss" and pending.miss_count == 1
    ws_b.sent.clear()
    gate_before = pending.last_input_ms

    ws = app.state.connections.get_for_color(room, "white")
    forged = _shot_raw(mid + 200, target=_hole_center(holes, ch.pops[0].hole))
    second = await handle_skill_check_shot(app, ws, room, "white", forged)

    assert second == "noop", "zero wall-clock spacing throttles however the client claims"
    assert pending.miss_count == 1 and pending.progress == 0, "no state change at all"
    assert pending.last_input_ms == gate_before, "a dropped shot never moves the gate"
    assert not ws_b.of_type("skill_check_spectate_shot"), "no relay for the dropped shot"


@pytest.mark.asyncio
async def test_jittered_arrivals_keep_a_legitimately_spaced_combo_press(app, clock):
    """REGRESSION: the gate used to compare SERVER ARRIVAL deltas, so network
    jitter that compressed two honestly spaced presses silently ate one of them —
    the client completed its run locally while the server sat a press short and let
    the check die at the deadline. Here the presses leave 80ms apart (the client's
    own floor) but land only 40ms apart because the first one lagged 80ms and the
    second only 40ms. Adjudicated elapsed is what the player actually saw, so both
    count; the arrival gap never does."""
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    gate = online.min_inter_input_ms(COMBO)
    sent = (320, 320 + int(COMBO_MIN_INTER_PRESS_MS))
    arrivals = (sent[0] + 80, sent[1] + 40)
    assert arrivals[1] - arrivals[0] < gate < sent[1] - sent[0], \
        "the jitter really does compress an honest gap below the gate"

    outs = [await _shoot_claiming(app, clock, room, "white", arrives_at_ms=arrivals[i],
                                  claims_ms=sent[i], direction=ch.prompts[i])
            for i in range(2)]

    assert outs == ["skillcheck_hit", "skillcheck_hit"], "neither honest press is eaten"
    assert pending.progress == 2
    assert len(ws_b.of_type("skill_check_spectate_shot")) == 2


@pytest.mark.asyncio
async def test_a_mash_claiming_sub_gate_spacing_is_still_dropped(app, clock):
    """The other half of the same gate: adjudicating on the claimed elapsed must
    not become a way to pace a burst. A client that mashes and reports the truth —
    presses 20ms apart — is throttled on exactly that reported spacing, and the
    dropped press moves neither the gate nor any check state."""
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    first = await _shoot_claiming(app, clock, room, "white", arrives_at_ms=400,
                                  claims_ms=320, direction=ch.prompts[0])
    assert first == "skillcheck_hit" and pending.last_input_ms == 320
    ws_b.sent.clear()

    second = await _shoot_claiming(app, clock, room, "white", arrives_at_ms=440,
                                   claims_ms=340, direction=ch.prompts[1])

    assert second == "noop", "20ms of claimed spacing is under the 50ms combo gate"
    assert pending.progress == 1 and pending.miss_count == 0, "no state change at all"
    assert pending.last_input_ms == 320, "a dropped press never moves the gate"
    assert not ws_b.of_type("skill_check_spectate_shot"), "no relay for the dropped press"


@pytest.mark.asyncio
async def test_whack_shot_without_target_fields_is_a_noop(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0))
    assert out == "noop"
    assert pending.miss_count == 0 and pending.progress == 0
    assert not ws_b.of_type("skill_check_spectate_shot")


@pytest.mark.asyncio
async def test_whack_out_of_range_target_is_rejected_at_validation(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0), target=(8.5, 0.5))
    assert out == "noop", "an off-board target fails Pydantic validation before any logic"
    assert pending.miss_count == 0 and pending.progress == 0
    assert not ws_b.of_type("skill_check_spectate_shot")


def _whack_quota_dead_elapsed(pending, ch):
    dead_at = next(e for e in range(0, int(pending.deadline_ms))
                   if ch.quota_unreachable(e, 0, -1))
    assert dead_at < pending.deadline_ms, "quota death precedes the absolute deadline"
    return dead_at


@pytest.mark.asyncio
@pytest.mark.parametrize("death", ["quota", "whiff_cap"])
@pytest.mark.parametrize("surface", ["move_gate", "resume", "sweep"])
async def test_whack_dead_pending_is_dead_at_every_surface(app, clock, surface, death):
    """Once the check is unwinnable — too few pops remain for the quota, OR the
    whiff cap is spent — the pending is dead everywhere at the same instant: the
    move gate resolves it inline, /resume omits it, and the sweep fails it, all
    through the one is_dead predicate. The live shot handler resolves the 3rd
    whiff inline at miss_count + 1, so a stored pending never actually reaches
    the cap; the cap-dead state is forged here to prove that if any path ever
    let it persist, every surface would still agree it is dead."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    if death == "quota":
        dead_at = _whack_quota_dead_elapsed(pending, ch)
    else:
        pending.miss_count = mole.MOLE_MAX_WHIFFS
        dead_at = _pop_mid(ch, 0)
        assert not ch.quota_unreachable(dead_at + 1, 0, -1), \
            "mid-schedule the quota is reachable — only the whiff cap kills it"
    clock.set((pending.start_ms + dead_at + 1) / 1000.0)
    if surface == "resume":
        assert room.pending_skillcheck is pending, "nothing swept it yet"
        assert _resume_payload(app, room, "white") is None, \
            "a reconnect must not re-hand a dead check"
        return
    if surface == "move_gate":
        out = await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
        assert out == "locked", "the dead check resolves inline; the retry hits the fresh lock"
    else:
        await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert len(room.backend.move_history) == 0
    assert room.skillcheck_log == [SkillCheckOutcome(1, "whack", False, "Qxd5")]
    assert ws_b.of_type("skill_check_result"), "the opponent hears the fail"


def _post_resume(app, room, token):
    resp = TestClient(app).post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id, "session_token": token})
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [WHACK, COMBO])
async def test_resume_resolves_a_dead_pending_and_serves_the_resulting_lock(app, clock, kind):
    """/resume used to HIDE a dead pending without resolving it, so the payload
    carried neither the check NOR the (from,to) lock its failure produces: a
    reconnecting mover was handed a board where the doomed capture still looked
    playable, and only the next sweep tick (up to 100ms later, or never if the room
    was mid-drop) corrected it. The endpoint now fails the check first and answers
    with the post-fail state. Both terminal shapes are covered — whack dying on an
    unreachable quota and combo dying on the wrongs cap — because the two reach
    is_dead through different check_expired branches."""
    if kind == WHACK:
        room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
        dead_at = _whack_quota_dead_elapsed(pending, ch)
    else:
        room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
        pending.miss_count = COMBO_MAX_WRONGS
        dead_at = 400
    clock.set((pending.start_ms + dead_at + 1) / 1000.0)
    assert pending.is_dead(app.state.now_ms()), "the forged state really is terminal"

    payload = _post_resume(app, room, "ta")

    assert payload["pending_skillcheck"] is None, "a dead check is never re-handed"
    assert payload["skillcheck_locks"] == [
        {"from": coord_from_square(frm), "to": coord_from_square(to)}], \
        "and the lock its failure produced rides the same payload"
    assert room.pending_skillcheck is None
    assert room.skillcheck_log == [SkillCheckOutcome(1, kind.value, False, "Qxd5")]
    assert ws_b.of_type("skill_check_result"), "the opponent hears the fail too"


@pytest.mark.asyncio
async def test_resume_leaves_a_live_pending_and_its_move_unlocked(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    payload = _post_resume(app, room, "ta")
    assert payload["pending_skillcheck"]["kind"] == "whack"
    assert payload["skillcheck_locks"] == [], "a live check locks nothing yet"
    assert room.pending_skillcheck is pending


@pytest.mark.asyncio
async def test_sweep_leaves_a_quota_reachable_whack_pending_alone(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    clock.set((pending.start_ms + _pop_mid(ch, 0)) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is pending, "quota still reachable mid-schedule"


@pytest.mark.asyncio
async def test_resume_wire_echoes_progress_and_captured_value_mid_whack(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                target=_hole_center(holes, ch.pops[0].hole))
    wire = _resume_payload(app, room, "white")
    assert wire.kind == "whack"
    assert wire.progress == 1, "the earned hit survives a reconnect"
    assert wire.captured_value == 1, "the client rebuilds the identical hole layout"
    resp = TestClient(app).post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id, "session_token": "ta"})
    assert resp.status_code == 200
    served = resp.json()["pending_skillcheck"]
    assert (served["progress"], served["captured_value"]) == (1, 1)


@pytest.mark.asyncio
async def test_whack_spectate_shots_carry_progress_coords_and_won(app, clock):
    """The relay is built from the mover's already-bounded shot, so every mirrored
    coordinate is on the board -- which is also what lets SkillCheckSpectateShotMessage
    carry the same ge/lt bounds as the inbound twin without ever refusing a real
    relay: a whole winning whack run has to survive the tightened model."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    required = ch.hits_required
    for i in range(required):
        await _fire(app, clock, room, "white", _pop_mid(ch, i),
                    target=_hole_center(holes, ch.pops[i].hole))
    relays = ws_b.of_type("skill_check_spectate_shot")
    assert len(relays) == required
    assert [r["progress"] for r in relays] == list(range(1, required + 1))
    assert [r["won"] for r in relays] == [False] * (required - 1) + [True]
    row, col = _hole_center(holes, ch.pops[0].hole)
    assert (relays[0]["target_row"], relays[0]["target_col"]) == (row, col), \
        "the opponent mirrors the mover's actual shot position"
    assert relays[0]["direction"] is None
    for r in relays:
        assert 0.0 <= r["target_row"] < SKILLCHECK_TARGET_MAX
        assert 0.0 <= r["target_col"] < SKILLCHECK_TARGET_MAX


def _bare_pending(**kw):
    base = dict(color="white", from_sq=Square(4, 3), to_sq=Square(3, 3), promotion=None,
                kind=WHACK, seed="s", value_diff=0, start_ms=0.0, expires_at_ms=5000.0)
    base.update(kw)
    return PendingSkillCheck(**base)


def test_materialising_a_derived_cache_never_changes_pending_equality():
    """_challenge and holes are derived caches keyed entirely off the real fields.
    If they counted toward __eq__, two pendings built from identical inputs would
    stop comparing equal the moment one of them lazily built its challenge or was
    minted with a hole layout -- a trap for any identity check on the pending."""
    a, b = _bare_pending(), _bare_pending()
    assert a == b, "identical inputs, identical pending"
    assert a.challenge is not None, "materialise the lazy cache on ONE side only"
    assert b._challenge is None, "the other side is still unmaterialised"
    assert a == b, "the derived challenge stays out of the equality"
    a.holes = ((0, 0), (7, 7))
    assert a == b, "the cached hole layout stays out of it too"
    assert _bare_pending(seed="other") != b, "the real inputs still separate two pendings"
    assert a.challenge is a.challenge, "the challenge is built once and reused"


@pytest.mark.asyncio
async def test_whack_holes_are_derived_once_at_mint_not_per_shot(app, clock):
    """Rebuilding the pit layout inside the shot handler made the geometry walk
    attacker-paced: every extra shot bought another radius search. The layout is
    frozen for the check's lifetime (the board cannot move while one is pending),
    so it is derived once when the check is armed -- and must equal what an
    independent re-derivation from the same backend produces."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    assert pending.holes == holes, "the minted layout is exactly the layout walk's output"
    assert len(pending.holes) == ch.hole_count, "one pit per hole the schedule uses"
    minted = pending.holes
    out = await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                      target=_hole_center(holes, ch.pops[0].hole))
    assert out == "skillcheck_hit", "adjudication still runs against the minted layout"
    assert pending.holes is minted, "a shot never re-derives the geometry"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [WHEEL, AIM, COMBO])
async def test_non_positional_kinds_mint_no_hole_layout(app, clock, kind):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    assert room.pending_skillcheck.holes == (), \
        "only the positional kind pays for a pit layout"


@pytest.mark.asyncio
async def test_resume_wire_echoes_the_credited_pop_mid_whack(app, clock):
    """A mid-whack reconnect must know WHICH pop was already credited: without it
    the restored client either re-credits a pop that is still up (a free hit) or
    treats it as unspent and shoots a dead target."""
    room, ws_w, ws_b, frm, to, pending, ch, holes = await _whack_room(app, clock)
    assert _resume_payload(app, room, "white").last_hit_pop == -1, "nothing hit yet"
    await _fire(app, clock, room, "white", _pop_mid(ch, 0),
                target=_hole_center(holes, ch.pops[0].hole))
    assert pending.last_hit_pop == 0, "the server credited the first pop"
    wire = _resume_payload(app, room, "white")
    assert wire.last_hit_pop == 0, "and the resume wire hands that back verbatim"
    resp = TestClient(app).post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": room.room_id, "session_token": "ta"})
    assert resp.status_code == 200
    assert resp.json()["pending_skillcheck"]["last_hit_pop"] == 0


@pytest.mark.asyncio
async def test_combo_full_correct_run_applies_the_move(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    assert pending.captured_value == 1
    outs = []
    for i, direction in enumerate(ch.prompts):
        outs.append(await _fire(app, clock, room, "white", 320 + 200 * i,
                                direction=direction))
    assert outs[:-1] == ["skillcheck_hit"] * (ch.prompt_count - 1)
    assert outs[-1] == "applied"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1
    applied = ws_b.of_type("move_applied")[-1]
    assert applied["skill_check_kind"] == "combo" and applied["skill_check_won"] is True
    assert room.skillcheck_log == [SkillCheckOutcome(1, "combo", True, "Qxd5")]


@pytest.mark.asyncio
async def test_combo_wrong_press_increments_miss_and_keeps_pending(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    wrong = next(d for d in COMBO_DIRECTIONS if d != ch.prompts[0])
    out = await _fire(app, clock, room, "white", 320, direction=wrong)
    assert out == "skillcheck_miss"
    assert pending.miss_count == 1 and pending.progress == 0
    assert room.pending_skillcheck is pending
    assert not ws_b.of_type("skill_check_result")


@pytest.mark.asyncio
async def test_combo_third_wrong_terminates_immediately(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    wrong = next(d for d in COMBO_DIRECTIONS if d != ch.prompts[0])
    outs = [await _fire(app, clock, room, "white", 320 + 200 * i, direction=wrong)
            for i in range(COMBO_MAX_WRONGS)]
    assert outs == ["skillcheck_miss"] * (COMBO_MAX_WRONGS - 1) + ["skillcheck_fail"], \
        "the wrong being processed counts: the 3rd terminates NOW, not on a later probe"
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert len(room.backend.move_history) == 0
    assert room.skillcheck_log == [SkillCheckOutcome(1, "combo", False, "Qxd5")]


@pytest.mark.asyncio
async def test_combo_two_wrongs_then_a_correct_run_still_wins(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    wrong = next(d for d in COMBO_DIRECTIONS if d != ch.prompts[0])
    for i in range(COMBO_MAX_WRONGS - 1):
        assert await _fire(app, clock, room, "white", 320 + 200 * i,
                           direction=wrong) == "skillcheck_miss"
    assert pending.miss_count == COMBO_MAX_WRONGS - 1, "one wrong short of the cap"
    out = None
    for i, direction in enumerate(ch.prompts):
        out = await _fire(app, clock, room, "white", 800 + 200 * i, direction=direction)
    assert out == "applied", "two wrongs leave the run fully winnable"
    assert len(room.backend.move_history) == 1


@pytest.mark.asyncio
async def test_combo_press_without_direction_is_a_noop(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", 320)
    assert out == "noop"
    assert pending.miss_count == 0 and pending.progress == 0
    assert not ws_b.of_type("skill_check_spectate_shot")


@pytest.mark.asyncio
async def test_combo_mash_burst_cannot_clear_the_run(app, clock):
    """A scripted 10ms mash: combo's 50ms server gate drops four of every five
    presses and the wrongs cap kills the run long before the deadline. The mash
    always presses off-prompt — a direction cycle against the run's random seed
    occasionally brute-forced the sequence and flaked the suite."""
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    elapsed, out = 120, None
    while room.pending_skillcheck is not None and elapsed < int(pending.deadline_ms):
        expected = ch.prompts[min(room.pending_skillcheck.progress, len(ch.prompts) - 1)]
        wrong = next(d for d in COMBO_DIRECTIONS if d != expected)
        out = await _fire(app, clock, room, "white", elapsed, direction=wrong)
        elapsed += 10
    assert len(room.backend.move_history) == 0, "the mash never lands the capture"
    assert room.pending_skillcheck is None and out == "skillcheck_fail"
    assert (frm, to) in room.skillcheck_locks
    assert room.skillcheck_log == [SkillCheckOutcome(1, "combo", False, "Qxd5")]


@pytest.mark.asyncio
async def test_combo_resume_mid_run_restores_progress(app, clock):
    room, ws_w, ws_b, frm, to, pending, ch = await _combo_room(app, clock)
    for i in range(2):
        await _fire(app, clock, room, "white", 320 + 200 * i, direction=ch.prompts[i])
    wire = _resume_payload(app, room, "white")
    assert wire.kind == "combo" and wire.progress == 2
    assert wire.miss_count == 0 and wire.captured_value == 1


async def _terminal_win(app, clock, room):
    """A guaranteed-winning input for the stored pending of ANY kind, derived
    from the stored challenge; whack/combo are fast-forwarded to quota-1 first."""
    pending = room.pending_skillcheck
    ch = _full_challenge(pending)
    if pending.kind in (WHEEL, AIM):
        return await _fire(app, clock, room, pending.color, _win_elapsed(pending))
    if pending.kind == WHACK:
        pending.progress = ch.hits_required - 1
        return await _fire(app, clock, room, pending.color, _pop_mid(ch, 0),
                           target=_hole_center(_holes_for(room), ch.pops[0].hole))
    pending.progress = ch.prompt_count - 1
    return await _fire(app, clock, room, pending.color, 400,
                       direction=ch.prompts[ch.prompt_count - 1])


def _deadline_shot_kwargs(kind):
    if kind == WHACK:
        return {"target": (0.5, 0.5)}
    if kind == COMBO:
        return {"direction": "up"}
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ALL_KINDS)
async def test_terminal_win_applies_and_latches_for_every_kind(app, clock, kind):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _terminal_win(app, clock, room)
    assert out == "applied"
    assert room.pending_skillcheck is None
    assert len(room.backend.move_history) == 1
    again = await handle_skill_check_shot(app, ws_w, room, "white",
                                          '{"type":"skill_check_shot"}')
    assert again == "noop", "the one-shot latch holds across all four kinds"
    assert len(room.backend.move_history) == 1, "the move applied exactly once"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ALL_KINDS)
async def test_sweep_auto_fails_past_the_absolute_deadline_for_every_kind(app, clock, kind):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 100) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None
    assert (frm, to) in room.skillcheck_locks
    assert room.result is None, "auto-fail locks the move; it does not end the game"
    assert ws_b.of_type("skill_check_result")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ALL_KINDS)
async def test_sweep_auto_fails_while_the_mover_is_disconnected_for_every_kind(app, clock, kind):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    app.state.rooms.mark_disconnected(room.room_id, "white")
    clock.advance((online.SKILLCHECK_DEADLINE_MS + 100) / 1000.0)
    await app.state.sweep.step_skillcheck_deadline()
    assert room.pending_skillcheck is None, "a disconnect can never dodge a pending check"
    assert (frm, to) in room.skillcheck_locks


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ALL_KINDS)
async def test_shot_and_sweep_same_tick_resolve_once_for_every_kind(app, clock, kind):
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    ws_b.sent.clear()
    out = await _fire(app, clock, room, "white", int(online.SKILLCHECK_DEADLINE_MS) + 10,
                      **_deadline_shot_kwargs(kind))
    await app.state.sweep.step_skillcheck_deadline()
    assert out == "skillcheck_fail"
    assert len(ws_b.of_type("skill_check_result")) == 1, "exactly one fail broadcast"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ALL_KINDS)
async def test_a_finalize_during_the_spectate_relay_preempts_the_win_for_every_kind(
        app, clock, kind):
    """The claim-before-relay ordering must not let a won shot write a ply onto an
    already-finished game: a resign or flag landing during the relay await ends the
    room, and room.result is re-read after the await for exactly that reason. The
    spectate mirror still went out -- the opponent then gets the result frame, which
    supersedes it -- but no ply, no series-scoring capture, no log entry lands."""
    room, ws_w, ws_b, frm, to = await _capture_room(app, clock, kind)

    async def _resign():
        await finalize_and_broadcast(app.state.rooms, app.state.connections, room,
                                     Reason.RESIGNATION, winner_color="black")

    racing = _RacingSpectatorWS(_resign)
    app.state.connections.add(room.room_id, room.black.client_uuid, racing)
    await handle_move(app, ws_w, room, "white", _move_raw(frm, to))
    out = await _terminal_win(app, clock, room)
    assert racing.of_type("skill_check_spectate_shot"), "the spectate-shot was relayed first"
    assert out == "already_over", "the post-await result re-read bails for every kind"
    assert len(room.backend.move_history) == 0
    assert room.skillcheck_log == []
    assert not ws_w.of_type("move_applied"), "no ply is broadcast after the result"
