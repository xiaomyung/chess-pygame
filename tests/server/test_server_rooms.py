import random

import pytest

from chessshootout.server.protocol import (
    FIRST_MOVE_ABORT_SECONDS, GRACE_SECONDS, IDLE_RESIGN_SECONDS, Reason,
)
from chessshootout.server.rooms import (
    AlreadyInGameError, GameAlreadyStartedError, InvalidTokenError, NotInRoomError,
    QUEUE_ABANDON_SECONDS, REMATCH_ABSOLUTE_CAP_SECONDS, REMATCH_IDLE_SECONDS, Room,
    RoomManager, ServerFullError, SharedAnnotations,
)
from tests.helpers import FakeClock


def _enqueue_kwargs(client_uuid, **overrides):
    base = dict(
        client_uuid=client_uuid,
        nickname=client_uuid.title(),
        session_token=f"tok-{client_uuid}",
        time_minutes=5,
        increment_seconds=0,
        side_preference="random",
    )
    base.update(overrides)
    return base


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def manager(clock):
    return RoomManager(now_provider=clock)


@pytest.mark.asyncio
async def test_enqueue_first_player_creates_pending_room(manager):
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert room.color_of("alice") in ("white", "black")
    assert not room.is_paired()
    assert room.backend is None
    assert manager.rooms_active == 0
    assert manager.queue_depth == 1


@pytest.mark.asyncio
async def test_enqueue_second_matching_player_pairs_them(manager):
    room1 = await manager.enqueue(**_enqueue_kwargs("alice"))
    room2 = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room1.room_id == room2.room_id
    assert room2.color_of("alice") != room2.color_of("bob")
    assert room2.is_paired()
    assert room2.backend is not None
    assert len(room2.backend.move_history) == 0
    assert room2.started_at == 0.0
    assert manager.rooms_active == 1
    assert manager.queue_depth == 0


@pytest.mark.asyncio
async def test_enqueue_pairs_only_within_same_time_control(manager):
    r1 = await manager.enqueue(**_enqueue_kwargs("alice", time_minutes=5))
    r2 = await manager.enqueue(**_enqueue_kwargs("bob", time_minutes=10))
    assert r1.room_id != r2.room_id
    assert not r1.is_paired()
    assert not r2.is_paired()
    assert manager.rooms_active == 0
    assert manager.queue_depth == 2


@pytest.mark.asyncio
async def test_enqueue_resolves_color_conflict_randomly(manager):
    """Two white-preferring players: the clash is broken by coin flip, but the
    pair must always come out on opposite colors regardless of the flip."""
    random.seed(42)
    await manager.enqueue(**_enqueue_kwargs("alice", side_preference="white"))
    room = await manager.enqueue(**_enqueue_kwargs("bob", side_preference="white"))
    assert {room.color_of("alice"), room.color_of("bob")} == {"white", "black"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_pref, second_pref, alice_color, bob_color",
    [
        pytest.param("white", "black", "white", "black",
                     id="explicit_complementary_sides_honored"),
        pytest.param("black", "random", "black", "white",
                     id="random_takes_opposite_of_explicit"),
    ],
)
async def test_enqueue_deterministic_color_resolution(
    manager, first_pref, second_pref, alice_color, bob_color,
):
    """Non-clashing color prefs resolve deterministically (no coin flip)."""
    await manager.enqueue(**_enqueue_kwargs("alice", side_preference=first_pref))
    room = await manager.enqueue(**_enqueue_kwargs("bob", side_preference=second_pref))
    assert room.color_of("alice") == alice_color
    assert room.color_of("bob") == bob_color


@pytest.mark.asyncio
async def test_enqueue_rejects_already_in_game(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    await manager.enqueue(**_enqueue_kwargs("bob"))
    with pytest.raises(AlreadyInGameError):
        await manager.enqueue(**_enqueue_kwargs("alice"))


@pytest.mark.asyncio
async def test_cancel_wait_removes_from_queue(manager):
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    await manager.cancel_wait(room.room_id, session_token="tok-alice")
    assert manager.queue_depth == 0
    assert manager.get(room.room_id) is None
    room2 = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert room2.room_id != room.room_id


@pytest.mark.asyncio
async def test_cancel_wait_invalid_token_rejected(manager):
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    with pytest.raises(InvalidTokenError):
        await manager.cancel_wait(room.room_id, session_token="bogus")
    assert manager.get(room.room_id) is room


@pytest.mark.asyncio
async def test_cancel_wait_unknown_room_raises(manager):
    with pytest.raises(NotInRoomError):
        await manager.cancel_wait("no-such-room", session_token="anything")


@pytest.mark.asyncio
async def test_cancel_wait_on_a_started_game_raises_game_already_started(manager):
    """A paired room is out of the queue, so cancelling it is not "not in room"
    but "too late" -- its own exception, which the route answers with the
    already_started status instead of a 404."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.is_paired()
    with pytest.raises(GameAlreadyStartedError):
        await manager.cancel_wait(room.room_id, session_token="tok-alice")


@pytest.mark.parametrize("exc_type", [ServerFullError, GameAlreadyStartedError])
def test_refusal_errors_are_not_runtime_errors(exc_type):
    """Both used to be `RuntimeError("server_full")` / `("game_already_started")`
    string-matched at the route. A plain Exception subclass is what makes the
    `except` arm precise: were either one still a RuntimeError, an unrelated
    RuntimeError raised deeper in the room manager would be caught by the same
    arm and answered as a polite refusal."""
    assert issubclass(exc_type, Exception)
    assert not issubclass(exc_type, RuntimeError)


@pytest.mark.asyncio
async def test_drop_player_starts_grace_timer(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    color = room.color_of("alice")
    manager.mark_connected(room.room_id, "white")
    manager.mark_connected(room.room_id, "black")
    manager.mark_disconnected(room.room_id, color)
    clock.advance(GRACE_SECONDS - 1)
    assert list(manager.grace_expired_rooms()) == []
    clock.advance(2)
    expired = list(manager.grace_expired_rooms())
    assert len(expired) == 1
    expired_room, abandoned_color = expired[0]
    assert expired_room.room_id == room.room_id
    assert abandoned_color == color


@pytest.mark.asyncio
async def test_finalize_result_aborts_with_no_winner(manager, clock):
    """A disconnect abort records a no-winner result and is idempotent (first sticks)."""
    clock.advance(12)
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "aborted")
    assert room.result == ("aborted", None)
    assert room.ended_at == 12.0
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    assert room.result == ("aborted", None)


@pytest.mark.asyncio
async def test_finalize_result_reports_whether_it_applied(manager):
    """The return value tells a racing caller (e.g. resign vs. mate landing at the
    same instant) whether IT won the state, so it knows whether to broadcast."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert manager.finalize_result(room.room_id, "checkmate", winner_color="white") is True
    assert manager.finalize_result(room.room_id, "resignation", winner_color="black") is False
    assert manager.finalize_result("no-such-room", "checkmate", winner_color="white") is False


@pytest.mark.asyncio
async def test_active_rooms_returns_an_iteration_safe_snapshot(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    snapshot = manager.active_rooms()
    assert snapshot == [room]
    assert isinstance(snapshot, list), \
        "callers mutate rooms while iterating; a live view would break"


@pytest.mark.asyncio
async def test_in_progress_room_for_only_matches_started_unfinished_games(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    assert manager.in_progress_room_for("alice") is None
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert manager.in_progress_room_for("alice") is None
    room.first_move_at = 1.0
    in_prog = manager.in_progress_room_for("alice")
    assert in_prog is not None
    assert in_prog[0] is room
    assert in_prog[1] == room.color_of("alice")
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    assert manager.in_progress_room_for("alice") is None


@pytest.mark.asyncio
async def test_release_for_new_game_frees_active_and_queued_uuids(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.release_for_new_game("alice")
    assert manager.get(room.room_id) is None
    await manager.enqueue(**_enqueue_kwargs("alice"))
    manager.release_for_new_game("bob")
    await manager.enqueue(**_enqueue_kwargs("bob"))


@pytest.mark.asyncio
async def test_clock_keeps_ticking_during_grace(manager, clock):
    """Clock is server-authoritative and never pauses on disconnect."""
    await manager.enqueue(**_enqueue_kwargs("alice", time_minutes=1, increment_seconds=0))
    room = await manager.enqueue(**_enqueue_kwargs("bob", time_minutes=1, increment_seconds=0))
    manager.mark_disconnected(room.room_id, "white")
    clock.advance(70)
    room.backend.tick_clock()
    assert room.backend.clock.flagged is not None


@pytest.mark.asyncio
async def test_reset_for_rematch_swaps_colors_and_clears_state(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    pre_white_uuid = room.white.client_uuid
    pre_black_uuid = room.black.client_uuid
    room.draw_offered_by = "white"
    room.takeback_offered_by = "white"
    room.rematch_offered_by = {"white", "black"}
    room.game_start_broadcast = True
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    assert room.result is not None
    assert room.white.at_result is True
    assert room.black.at_result is True
    assert room.last_rematch_activity_at == room.ended_at

    assert manager.reset_for_rematch(room.room_id) is True
    assert room.result is None
    assert room.draw_offered_by is None
    assert room.takeback_offered_by is None
    assert room.rematch_offered_by == set()
    assert room.game_start_broadcast is False
    assert room.ended_at is None
    assert room.last_rematch_activity_at is None
    assert room.first_move_at is None
    assert room.white.at_result is False
    assert room.black.at_result is False
    assert room.white.client_uuid == pre_black_uuid
    assert room.black.client_uuid == pre_white_uuid
    assert len(room.backend.move_history) == 0
    assert room.backend.clock is not None


@pytest.mark.asyncio
async def test_reset_for_rematch_returns_false_when_room_gone(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    manager.drop_room_now(room.room_id)
    assert manager.reset_for_rematch(room.room_id) is False


@pytest.mark.asyncio
async def test_reset_for_rematch_returns_false_when_not_finished(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert manager.reset_for_rematch(room.room_id) is False


@pytest.mark.asyncio
async def test_enqueue_stores_country_per_slot(manager):
    await manager.enqueue(**_enqueue_kwargs("alice", country="US"))
    room = await manager.enqueue(**_enqueue_kwargs("bob", country="RO"))
    alice_color = room.color_of("alice")
    assert room.slot(alice_color).country == "US"
    assert room.slot(room.opp_color(alice_color)).country == "RO"


@pytest.mark.asyncio
async def test_enqueue_country_defaults_none(manager):
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert room.slot(room.color_of("alice")).country is None


@pytest.mark.asyncio
async def test_rematch_swaps_country_with_player(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice", country="US"))
    room = await manager.enqueue(**_enqueue_kwargs("bob", country="RO"))
    pre_white_country = room.white.country
    pre_black_country = room.black.country
    room.game_start_broadcast = True
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    manager.reset_for_rematch(room.room_id)
    assert room.white.country == pre_black_country
    assert room.black.country == pre_white_country


@pytest.mark.asyncio
async def test_gc_drops_finished_rooms_after_idle(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    clock.advance(REMATCH_IDLE_SECONDS - 1)
    manager.gc_finished_rooms()
    assert manager.rooms_active == 1
    clock.advance(2)
    manager.gc_finished_rooms()
    assert manager.rooms_active == 0
    new_room = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert new_room.room_id != room.room_id


@pytest.mark.asyncio
async def test_gc_rematch_activity_extends_idle_until_cap(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    for _ in range(2):
        clock.advance(REMATCH_IDLE_SECONDS - 1)
        manager.mark_rematch_activity(room)
        manager.gc_finished_rooms()
        assert manager.rooms_active == 1
    clock.advance(REMATCH_ABSOLUTE_CAP_SECONDS)
    manager.gc_finished_rooms()
    assert manager.rooms_active == 0


@pytest.mark.asyncio
async def test_get_finds_queued_room_before_pairing(manager):
    """First player's WS auth happens before the second matchmakes.
    `get(room_id)` must resolve queued rooms too, else auth fails."""
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    found = manager.get(room.room_id)
    assert found is room


@pytest.mark.asyncio
async def test_color_of_returns_correct_color(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.color_of("alice") in ("white", "black")
    assert room.color_of("bob") in ("white", "black")
    assert room.color_of("alice") != room.color_of("bob")
    assert room.color_of("unknown") is None
    assert room.slot(room.color_of("alice")).client_uuid == "alice"
    assert room.slot(room.color_of("bob")).client_uuid == "bob"


@pytest.mark.asyncio
async def test_slot_by_token_returns_correct_slot(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    color, slot = room.slot_by_token("tok-alice")
    assert slot.client_uuid == "alice"
    assert color == room.color_of("alice")
    color, slot = room.slot_by_token("bogus")
    assert slot is None and color is None


@pytest.mark.asyncio
async def test_series_scores_award_win_and_persist_across_rematch(manager):
    """A win gives the winner +1 keyed by nickname; a later draw gives both +0.5;
    the tally survives the color swap of a rematch."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    rid = room.room_id
    white_name = room.white.nickname
    black_name = room.black.nickname
    manager.finalize_result(rid, "checkmate", "white")
    assert room.series_scores[white_name] == 1.0
    assert room.score_for("white") == 1.0
    assert room.score_for("black") == 0.0
    manager.reset_for_rematch(rid)
    manager.finalize_result(rid, "draw_repetition", None)
    assert room.series_scores[white_name] == 1.5
    assert room.series_scores[black_name] == 0.5


@pytest.mark.asyncio
async def test_series_scores_not_awarded_on_abort(manager):
    """An abort is neutral: no series point to either player."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "aborted", None)
    assert room.series_scores == {}
    assert room.score_for("white") == 0.0
    assert room.score_for("black") == 0.0


def test_annotations_for_returns_the_matching_color_store():
    room = Room(room_id="r", time_minutes=5, increment_seconds=0, created_at=0.0)
    assert room.annotations_for("white") is room.annotations_white
    assert room.annotations_for("black") is room.annotations_black
    assert isinstance(room.annotations_for("white"), SharedAnnotations)
    assert room.annotations_white is not room.annotations_black


def test_shared_annotations_clear_marks_preserves_sharing():
    ann = SharedAnnotations(sharing=True)
    ann.highlights.add("e4")
    ann.arrows.append(("e2", "e4"))
    ann.clear_marks()
    assert ann.highlights == set()
    assert ann.arrows == []
    assert ann.sharing is True


def test_shared_annotations_reset_clears_marks_and_sharing():
    ann = SharedAnnotations(sharing=True)
    ann.highlights.add("e4")
    ann.arrows.append(("e2", "e4"))
    ann.reset()
    assert ann.highlights == set() and ann.arrows == []
    assert ann.sharing is False


@pytest.mark.asyncio
async def test_finalize_result_resets_both_shared_annotation_stores(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    for color in ("white", "black"):
        ann = room.annotations_for(color)
        ann.sharing = True
        ann.highlights.add("e4")
        ann.arrows.append(("e2", "e4"))
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    for color in ("white", "black"):
        ann = room.annotations_for(color)
        assert ann.sharing is False
        assert ann.highlights == set()
        assert ann.arrows == []


@pytest.mark.asyncio
async def test_reset_for_rematch_installs_fresh_annotation_stores(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    white_before = room.annotations_white
    black_before = room.annotations_black
    room.annotations_white.sharing = True
    room.annotations_white.highlights.add("d4")
    room.annotations_black.arrows.append(("g1", "f3"))
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    assert manager.reset_for_rematch(room.room_id) is True
    assert room.annotations_white is not white_before
    assert room.annotations_black is not black_before
    assert room.annotations_white.sharing is False
    assert room.annotations_white.highlights == set()
    assert room.annotations_black.arrows == []


async def test_timeout_with_no_plies_finalizes_as_aborted_draw(manager):
    """User rule: a flag fall before any move landed is nobody's win -- the game
    aborts with no winner and awards no series points. Resignation at zero plies
    deliberately KEEPS its winner (an explicit choice, unlike a dead clock)."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.plies_ever == 0
    manager.finalize_result(room.room_id, "timeout", "white")
    assert room.result == ("aborted", None)
    assert room.series_scores == {}


async def test_timeout_after_a_ply_keeps_its_winner(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    room.plies_ever = 1
    manager.finalize_result(room.room_id, "timeout", "white")
    assert room.result == ("timeout", "white")


async def test_zero_ply_resignation_keeps_its_winner(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "resignation", "black")
    assert room.result == ("resignation", "black")


async def test_zero_ply_draw_agreement_stays_a_draw(manager):
    """ZERO_PLY_ABORT_REASONS only covers dead-clock and walk-away (timeout,
    abandonment); a mutually agreed draw is a deliberate two-sided choice, so a
    draw_agreement at zero plies is NOT rewritten to aborted. The result stays
    draw_agreement and _award_series still splits the point +0.5/+0.5, exactly as
    it would after a full game -- agreeing to a draw before moving is a real draw,
    not a cancelled game."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.plies_ever == 0
    manager.finalize_result(room.room_id, "draw_agreement", None)
    assert room.result == ("draw_agreement", None)
    assert room.series_scores[room.white.nickname] == 0.5
    assert room.series_scores[room.black.nickname] == 0.5
    assert room.score_for("white") == 0.5
    assert room.score_for("black") == 0.5


async def test_zero_ply_abandonment_finalizes_as_aborted_draw(manager):
    """Leaving before any move landed cancels the game instead of awarding a win:
    nobody played, nobody scores (same rule as a zero-ply flag fall)."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    manager.finalize_result(room.room_id, "abandonment", "black")
    assert room.result == ("aborted", None)
    assert room.series_scores == {}


def _bare_room(**overrides):
    base = dict(room_id="r", time_minutes=5, increment_seconds=0, created_at=0.0)
    base.update(overrides)
    return Room(**base)


@pytest.mark.parametrize(
    "plies, expected",
    [
        pytest.param(0, (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
                     id="ply_zero_abort_window"),
        pytest.param(1, (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
                     id="ply_one_abort_window"),
        pytest.param(2, (Reason.RESIGNATION, IDLE_RESIGN_SECONDS),
                     id="ply_two_resign_window"),
        pytest.param(3, None, id="ply_three_disarmed"),
    ],
)
def test_idle_window_table_arms_by_plies_ever(plies, expected):
    """The whole lifecycle policy in one table: plies 0 and 1 get the no-winner
    abort window, ply 2 the somebody-loses resign window, and from ply 3 the
    game never arms an idle window again."""
    room = _bare_room()
    room.plies_ever = plies
    assert room.idle_window() == expected


def test_idle_remaining_floors_at_zero_and_is_none_when_disarmed():
    room = _bare_room()
    room.plies_ever = 2
    room.idle_since = 10.0
    assert room.idle_remaining(10.0) == IDLE_RESIGN_SECONDS
    assert room.idle_remaining(30.0) == IDLE_RESIGN_SECONDS - 20.0
    assert room.idle_remaining(10.0 + IDLE_RESIGN_SECONDS + 5.0) == 0.0, \
        "expired windows report 0.0, never a negative countdown"
    room.idle_since = None
    assert room.idle_remaining(30.0) is None
    room.idle_since = 10.0
    room.plies_ever = 3
    assert room.idle_remaining(30.0) is None, \
        "a stale idle_since past the table's last row means no window at all"


@pytest.mark.asyncio
async def test_pairing_arms_the_abort_window_from_started_at(manager, clock):
    """The queued waiter carries no window (there is nobody to abort against);
    the moment the second player pairs, the abort window arms anchored at
    started_at — the same instant game_start's started_seconds_ago measures
    from, so server and client count down the same 60 seconds."""
    queued = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert queued.idle_since is None
    clock.advance(5)
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.started_at == 5.0
    assert room.idle_since == room.started_at
    assert room.idle_window() == (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS)


@pytest.mark.asyncio
async def test_finalize_result_disarms_the_idle_window(manager):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    room.idle_pushed_at = 1.0
    assert room.idle_since is not None
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    assert room.idle_since is None
    assert room.idle_pushed_at is None


@pytest.mark.asyncio
async def test_reset_for_rematch_rearms_the_abort_window(manager, clock):
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    room.plies_ever = 6
    manager.finalize_result(room.room_id, "checkmate", winner_color="white")
    room.idle_pushed_at = 99.0
    clock.advance(30)
    assert manager.reset_for_rematch(room.room_id) is True
    assert room.plies_ever == 0
    assert room.idle_since == room.started_at == clock()
    assert room.idle_pushed_at is None
    assert room.idle_window() == (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS)


def test_resignation_is_not_a_zero_ply_abort_reason():
    """ZERO_PLY_ABORT_REASONS must never grow "resignation". The idle forfeit
    can only fire at plies_ever == 2, so adding it would change nothing for the
    idle feature — but it would silently flip a HAND resign at zero plies from a
    resignation with a winner (the deliberate choice pinned by
    test_zero_ply_resignation_keeps_its_winner) into a no-winner abort."""
    assert "resignation" not in RoomManager.ZERO_PLY_ABORT_REASONS
    assert RoomManager.ZERO_PLY_ABORT_REASONS == ("timeout", "abandonment")


@pytest.mark.asyncio
async def test_color_to_move_is_none_without_a_backend(manager):
    """The unified helper (it replaced handlers._color_to_move) guards the
    pre-pairing shape instead of raising on backend=None."""
    assert _bare_room().color_to_move() is None
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    assert room.color_to_move() == "white"


async def test_stale_queued_rooms_ignores_fresh_waiters(manager, clock):
    """SECURITY (queue GC): nothing but age decides staleness, and a waiter that
    has not yet reached the TTL is never listed. Without this floor the reap
    would evict players in the middle of a normal match search."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    clock.advance(QUEUE_ABANDON_SECONDS - 1)
    assert manager.stale_queued_rooms() == []
    clock.advance(2)
    assert len(manager.stale_queued_rooms()) == 1


async def test_stale_queued_rooms_never_lists_active_or_paired_rooms(manager, clock):
    """Only `_queue` entries are reapable. A paired room carries the same
    `created_at` as the queued room it grew out of, so an age-only filter that
    walked `_active` too would delete live games the moment they aged past the
    TTL."""
    await manager.enqueue(**_enqueue_kwargs("alice"))
    room = await manager.enqueue(**_enqueue_kwargs("bob"))
    clock.advance(QUEUE_ABANDON_SECONDS * 10)
    assert manager.stale_queued_rooms() == []
    assert manager.get(room.room_id) is room
    assert manager.rooms_active == 1


async def test_drop_queued_room_frees_the_slot_and_the_uuid(manager, clock):
    """SECURITY (finding: queued rooms were never garbage-collected). A client
    that matchmakes and then vanishes without DELETE /matchmake used to pin its
    queue slot forever; the reap must release the room AND its `_uuid_to_room`
    entry, otherwise the uuid stays wedged in AlreadyInGame."""
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    clock.advance(QUEUE_ABANDON_SECONDS + 1)
    stale = manager.stale_queued_rooms()
    assert stale == [room]
    assert manager.drop_queued_room(room) is True
    assert manager.queue_depth == 0
    assert manager.get(room.room_id) is None
    assert manager.drop_queued_room(room) is False


async def test_reaped_uuid_can_matchmake_again_cleanly(manager, clock):
    """After the reap the same client re-queues into a brand new room instead of
    hitting AlreadyInGameError."""
    room = await manager.enqueue(**_enqueue_kwargs("alice"))
    clock.advance(QUEUE_ABANDON_SECONDS + 1)
    manager.drop_queued_room(room)
    fresh = await manager.enqueue(**_enqueue_kwargs("alice"))
    assert fresh.room_id != room.room_id
    assert manager.queue_depth == 1


async def test_reaping_a_queued_room_frees_capacity_against_max_rooms(clock):
    """The bug that made the queue leak exploitable: `enqueue` counts queue depth
    against `_max_rooms`, so abandoned waiters permanently consumed the room
    budget and every later matchmake got refused as full."""
    manager = RoomManager(now_provider=clock, max_rooms=2)
    stale = [await manager.enqueue(**_enqueue_kwargs(name, time_minutes=minutes))
             for name, minutes in (("alice", 5), ("bob", 10))]
    with pytest.raises(ServerFullError):
        await manager.enqueue(**_enqueue_kwargs("carl", time_minutes=15))
    clock.advance(QUEUE_ABANDON_SECONDS + 1)
    for room in manager.stale_queued_rooms():
        manager.drop_queued_room(room)
    assert manager.queue_depth == 0
    assert all(manager.get(room.room_id) is None for room in stale)
    assert await manager.enqueue(**_enqueue_kwargs("carl", time_minutes=15)) is not None


@pytest.mark.parametrize(
    "drain",
    [
        pytest.param(
            lambda m, room: m.drop_queued_room(room), id="reap",
        ),
        pytest.param(
            lambda m, room: m.release_for_new_game("alice"), id="release_for_new_game",
        ),
    ],
)
async def test_emptying_a_time_control_bucket_removes_it(manager, drain):
    """`_queue` is a defaultdict keyed by the caller-supplied (minutes, increment)
    pair. Every drain path must delete the emptied bucket, else a flood of
    distinct time controls grows the dict without bound even after the rooms
    themselves are gone."""
    room = await manager.enqueue(**_enqueue_kwargs("alice", time_minutes=7,
                                                   increment_seconds=3))
    assert (7, 3) in manager._queue
    drain(manager, room)
    assert (7, 3) not in manager._queue
    assert manager.queue_depth == 0


async def test_pairing_removes_the_emptied_bucket(manager):
    """Pairing pops the waiting room out of its bucket; the leftover empty list
    must go too."""
    await manager.enqueue(**_enqueue_kwargs("alice", time_minutes=7))
    await manager.enqueue(**_enqueue_kwargs("bob", time_minutes=7))
    assert manager._queue == {}


async def test_cancel_wait_removes_the_emptied_bucket(manager):
    room = await manager.enqueue(**_enqueue_kwargs("alice", time_minutes=7))
    await manager.cancel_wait(room.room_id, session_token="tok-alice")
    assert manager._queue == {}
