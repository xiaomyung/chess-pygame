import asyncio
import json
import secrets
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from fastapi import FastAPI, WebSocket
from pydantic import ValidationError

from chessshootout.backend.backend import Backend
from chessshootout.backend.clock import Clock
from chessshootout.backend.fen import export_fen
from chessshootout.backend.pieces import PieceColor
from chessshootout.backend.utils import (
    PROMO_TYPE_BY_LETTER, Square, coord_from_square, square_from_coord,
)
from chessshootout.server import logging_setup
from chessshootout.server.connections import ConnectionRegistry, broadcast, send
from chessshootout.server.broadcasts import (
    broadcast_game_start, finalize_and_broadcast, push_idle_window,
    resolve_skillcheck_fail)
from chessshootout.server.protocol import (
    AnnotationDeltaMessage, AnnotationsBlockedMessage, AnnotationsStateMessage,
    ArrowWire, ClockSnapshot, ConnectionStatusMessage,
    DrawOfferedMessage, DrawResponseMessage,
    ErrorMessage, GIVE_TIME_SECONDS, GIVE_TIME_TICK_MS, GiveTimeMessage,
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_SHARED_ARROWS, MAX_SHARED_HIGHLIGHTS, MODERATION_TRIP_LIMIT,
    MoveAppliedMessage, MoveMessage,
    PingMessage, PongMessage, QuickChatMessage, QuickChatReceivedMessage, Reason,
    RematchRequestMessage, RematchResponseMessage, RematchUpdateMessage,
    ResyncDirectiveMessage, SetMarksVisibilityMessage,
    SkillCheckRequiredMessage, SkillCheckShotMessage, SkillCheckSpectateMessage,
    SkillCheckSpectateShotMessage,
    TakebackAppliedMessage, TakebackOfferedMessage,
    TakebackResponseMessage, TimeGrantedMessage,
)
from chessshootout.server.moderation import detector
from chessshootout.server.moderation.load import ModerationLoad
from chessshootout.server.rooms import (
    PendingSkillCheck, PlayerSlot, Room, SharedAnnotations)
from chessshootout.server.sweep import RESULT_REASON_BY_GAME_RESULT
from chessshootout.skillcheck import mole, online
from chessshootout.skillcheck.triggers import compute_facts
from chessshootout.skillcheck.types import SkillCheckKind, SkillCheckOutcome, TriggerFacts


log = logging_setup.get_logger("chess.server.app")

RESYNC_NOTIFY_MIN_INTERVAL_SECONDS = 5.0
RESYNC_NOTIFY_FLAP_FLOOR_SECONDS = 1.0
RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS = 1.0
RESYNC_GATE_PRUNE_THRESHOLD = 512

RESYNC_NOTIFY = "notify"
RESYNC_DIRECTIVE = "directive"


def _transit_grace(interval: float) -> float:
    """
    Work out how long a heartbeat may still describe the position as it was
    just before the latest change. It has to cover a round trip plus a client
    frame, and stay strictly under one heartbeat so at most one heartbeat per
    change is ever forgiven

    :param interval: seconds between heartbeats, as the server asks for them.
    :returns: seconds a history change keeps excusing a stale heartbeat.
    """
    return min(1.5, interval * 0.75)


RESYNC_TRANSIT_GRACE_SECONDS = _transit_grace(HEARTBEAT_INTERVAL_SECONDS)
RESYNC_STABLE_MISMATCH_HEARTBEATS = 2

IDLE_ACTIVITY_TYPES = frozenset({
    "move", "skill_check_shot", "draw_offer", "draw_response",
    "takeback_request", "takeback_response", "give_time", "quick_chat",
    "annotations_state", "annotation_delta", "set_marks_visibility",
})


class _ResyncGate:
    """
    Debounce for the desync repair traffic, keyed by room, colour and which
    kind of repair it is. A heartbeat arrives every couple of seconds, so
    without this a board that has fallen behind would earn its opponent a
    notification and itself an order to refetch on every single ping
    """

    def __init__(self) -> None:
        """
        Start with every gate open, so the first event of each kind fires
        immediately and the debounce only bites on repeats
        """
        self._open_at: dict[tuple[str, str, str], float] = {}

    def allow(self, key: tuple[str, str, str], now: float, interval: float) -> bool:
        """
        Ask whether one debounced event may fire now, and close its gate
        behind it when it may. This is what keeps the resync notification and
        the resync directive from following every heartbeat

        :param key: room id, colour and which kind of repair is being gated.
        :param now: monotonic seconds, the server's own clock.
        :param interval: seconds this gate stays shut once the event fires.
        :returns: True when the caller should go ahead and fire the event.
        """
        if now < self._open_at.get(key, now):
            return False
        if len(self._open_at) >= RESYNC_GATE_PRUNE_THRESHOLD:
            self._prune(now)
        self._open_at[key] = now + interval
        return True

    def reopen(self, key: tuple[str, str, str], now: float, delay: float) -> None:
        """
        Bring a closed gate forward so the next event may fire sooner, used
        when a player catches up and a fresh lagging spell should be reported
        promptly instead of waiting out the old interval. A gate that is not
        closed is left alone

        :param key: room id, colour and which kind of repair is being gated.
        :param now: monotonic seconds, the server's own clock.
        :param delay: seconds from now the gate may open at the latest.
        """
        if key in self._open_at:
            self._open_at[key] = min(self._open_at[key], now + delay)

    def _prune(self, now: float) -> None:
        """
        Forget the gates that have already opened again, so a long-lived
        server does not keep one entry per room and colour it has ever seen

        :param now: monotonic seconds; gates due at or before this are gone.
        """
        for key, at in list(self._open_at.items()):
            if at <= now:
                del self._open_at[key]


def _app_state_singleton(app: FastAPI, name: str, factory: Callable[[], Any]) -> Any:
    """
    Fetch a helper that lives on the app for the whole process, building it on
    first use. Keeps per-process machinery such as the resync gate and the
    moderation meter off module globals, so every test app gets its own

    :param app: the FastAPI application that owns the shared server state.
    :param name: attribute the helper is stored under on app.state.
    :param factory: builds the helper the first time it is asked for.
    :returns: the helper already stored, or the one just built.
    """
    value = getattr(app.state, name, None)
    if value is None:
        value = factory()
        setattr(app.state, name, value)
    return value


def _resync_gate(app: FastAPI) -> _ResyncGate:
    """
    Get this server's resync debounce gate. One gate serves every room, since
    it is keyed by room and colour internally

    :param app: the FastAPI application that owns the shared server state.
    :returns: the process-wide resync gate.
    """
    return cast(_ResyncGate, _app_state_singleton(app, "resync_gate", _ResyncGate))


def _moderation_load(app: FastAPI) -> ModerationLoad:
    """
    Get this server's moderation CPU meter, which decides whether a player may
    keep having their shared marks screened. One meter serves every room, so a
    single busy game cannot spend the whole server's screening time

    :param app: the FastAPI application that owns the shared server state.
    :returns: the process-wide moderation load meter.
    """
    return cast(ModerationLoad,
                _app_state_singleton(app, "moderation_load", ModerationLoad))


def clock_snapshot(clock: Clock | None) -> ClockSnapshot:
    """
    Package both players' remaining time for the wire, in the shape every
    frame that carries clocks expects. A game with no clock reports zeros
    rather than nothing, so the client always has numbers to draw

    :param clock: the room's engine clock, or None for an unclocked game.
    :returns: the clock reading to put on the wire.
    """
    if clock is None:
        return ClockSnapshot(white_remaining=0.0, black_remaining=0.0, running_for=None)
    running = None
    if clock.running_for is not None:
        running = "white" if clock.running_for == PieceColor.WHITE else "black"
    return ClockSnapshot(
        white_remaining=clock.white_remaining,
        black_remaining=clock.black_remaining,
        running_for=running,
    )


def arrow_wires(pairs: Iterable[tuple[str, str]]) -> list[ArrowWire]:
    """
    Dress stored arrows up for the wire. Arrows live server-side as plain
    square pairs, and this is the single place they become wire models --
    the live relays and the resume snapshot both come through here

    :param pairs: arrows as origin and destination squares in algebraic form.
    :returns: the same arrows as wire models, in the order given.
    """
    return [ArrowWire(from_sq=a[0], to_sq=a[1]) for a in pairs]


def peek_type(raw: str) -> Any:
    """
    Read just the type out of an inbound frame, so the dispatcher can pick a
    handler without fully parsing a message that may well be junk. Anything
    that is not JSON carrying a type is reported as unknown rather than raised

    :param raw: the websocket text frame exactly as it arrived.
    :returns: the type field exactly as the frame carried it, which is a message
        type for a well-formed frame and None when the frame is unusable.
    """
    try:
        return json.loads(raw).get("type")
    except (json.JSONDecodeError, ValueError):
        return None


async def dispatch(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                   raw: str) -> tuple[str | None, str]:
    """
    Route one inbound websocket frame to the handler that owns that message
    type, then refresh the idle countdown when the frame was a deliberate
    action. An unknown type is refused with an error frame instead of dropping
    the connection

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the frame arrived on.
    :param room: the room this connection is playing in.
    :param color: the side that sent the frame, white or black.
    :param raw: the websocket text frame exactly as it arrived.
    :returns: the message type and the handler's short outcome word, for logs.
    """
    msg_type = peek_type(raw)
    handler = HANDLERS.get(msg_type)
    if handler is None:
        await send(websocket, ErrorMessage(reason=Reason.INVALID_MESSAGE))
        return msg_type, "invalid_message"
    verdict = await handler(app, websocket, room, color, raw)
    if msg_type in IDLE_ACTIVITY_TYPES:
        await _touch_idle_window(app, room, color)
    return msg_type, verdict


async def _touch_idle_window(app: FastAPI, room: Room, color: str) -> None:
    """
    Restart the idle countdown after a deliberate action by the side to move,
    and push the fresh reading to both players. Only the auto-resign window
    ever resets -- the abort windows before the first moves never do, because
    that clock is not running yet and the abort deadline guards the room slot
    itself rather than a player's thinking time

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room whose countdown is being refreshed.
    :param color: the side whose message arrived, white or black.
    """
    window = room.idle_window()
    if (room.result is not None or room.idle_since is None or window is None
            or window.outcome != Reason.RESIGNATION or color != room.color_to_move()):
        return
    now = app.state.now()
    room.idle_since = now
    await push_idle_window(app.state.rooms, app.state.connections, room, now)


async def handle_move(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                      raw: str) -> str:
    """
    Play one move for the side that sent it: confirm it is their turn and the
    move is not locked out, then either land it or start the skill check it
    triggers. A check that was already running and has since died is resolved
    as a miss here first, by exactly the same expiry rule every other surface
    applies

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the move arrived on.
    :param room: the room the move is being played in.
    :param color: the side that sent the move, white or black.
    :param raw: the websocket text frame holding the move.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.backend is None:
        return "no_backend"
    if room.result is not None:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.GAME_ALREADY_OVER, msg_type="move"))
        return "already_over"
    pending = room.pending_skillcheck
    if pending is not None and pending.is_dead(app.state.now_ms()):
        await resolve_skillcheck_fail(rooms, connections, room)
        if room.result is not None:
            return "already_over"
        pending = None
    if pending is not None:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.SKILLCHECK_PENDING, msg_type="move"))
        return "pending"
    try:
        msg = MoveMessage.model_validate_json(raw)
        from_sq = square_from_coord(msg.from_sq)
        to_sq = square_from_coord(msg.to_sq)
    except (ValidationError, ValueError):
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.INVALID_MOVE_FORMAT, msg_type="move"))
        return "invalid_move_format"
    expected = room.color_to_move()
    if expected != color:
        log.info("move rejected room=%s mover=%s expected=%s reason=not_your_turn",
                 room.room_id, color, expected)
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.NOT_YOUR_TURN, msg_type="move"))
        return "not_your_turn"
    if (from_sq, to_sq) in room.skillcheck_locks:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.MOVE_LOCKED, msg_type="move"))
        return "locked"
    facts = compute_facts(room.backend, from_sq, to_sq, room.skillcheck_locks)
    if facts is None:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.INVALID_MOVE_FORMAT, msg_type="move"))
        return "illegal"
    kind = online.select_kind(room.skillcheck_secret, room.plies_ever, room.backend,
                              from_sq, to_sq, room.skillcheck_locks, facts)
    if kind == SkillCheckKind.NONE:
        return await _apply_move(app, room, color, from_sq, to_sq, msg.promotion,
                                 skill_kind=None, skill_won=None, answers="move")
    return await _arm_skillcheck(app, room, color, kind, from_sq, to_sq,
                                 msg.promotion, facts)


async def _arm_skillcheck(app: FastAPI, room: Room, color: str, kind: SkillCheckKind,
                          from_sq: Square, to_sq: Square, promotion: str | None,
                          facts: TriggerFacts) -> str:
    """
    Start the skill check a move triggered, holding the move back until the
    player beats it. The challenge is described to the mover and mirrored to
    the opponent as a read-only spectate view; its seed is drawn fresh for
    this check and carries nothing of the room secret, and for whack-a-mole
    the server works out the hole squares itself rather than trusting a client

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the move was played in.
    :param color: the side that must beat the check, white or black.
    :param kind: which of the four challenges this move drew.
    :param from_sq: origin square of the move being held back.
    :param to_sq: destination square of the move being held back.
    :param promotion: promotion piece letter, or None when not promoting.
    :param facts: what the move does -- capture, promotion, piece values.
    :returns: short outcome word naming the kind of check that fired.
    """
    connections = app.state.connections
    value_diff = online.value_diff_for(facts, promotion)
    seed = secrets.token_hex(16)
    start_ms = app.state.now_ms()
    deadline_ms = online.skillcheck_deadline_ms(room.time_minutes * 60)
    holes: tuple[tuple[int, int], ...] = ()
    if kind == SkillCheckKind.WHACK:
        holes = mole.holes_for(seed, facts.captured_value, (to_sq.row, to_sq.col),
                               cast(Backend, room.backend).state,
                               cast(Backend, room.backend).SIZE)
    room.pending_skillcheck = PendingSkillCheck(
        color=color, from_sq=from_sq, to_sq=to_sq, promotion=promotion,
        kind=kind, seed=seed, value_diff=value_diff,
        start_ms=start_ms, expires_at_ms=start_ms + deadline_ms, deadline_ms=deadline_ms,
        captured_value=facts.captured_value, holes=holes,
    )
    log.info("skillcheck fired room=%s mover=%s kind=%s", room.room_id, color, kind.value)
    geometry = dict(
        kind=kind.value, seed=seed, value_diff=value_diff, deadline_ms=deadline_ms,
        captured_value=facts.captured_value,
        from_sq=coord_from_square(from_sq), to_sq=coord_from_square(to_sq),
        promotion=promotion)
    await send(connections.get_for_color(room, color),
               SkillCheckRequiredMessage(**geometry))
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, SkillCheckSpectateMessage(**geometry))
    return f"skillcheck:{kind.value}"


async def _apply_move(app: FastAPI, room: Room, color: str, from_sq: Square,
                      to_sq: Square, promotion: str | None,
                      *, skill_kind: str | None, skill_won: bool | None,
                      answers: str) -> str:
    """
    Land a move on the server's own board and tell both players -- the single
    place a ply actually becomes real. It advances the server-only ply counter
    that seeds which move draws which check, stamps the moment the history
    grew so a heartbeat still in flight is not mistaken for a drifted board,
    clears the per-turn move locks and both players' shared marks, records the
    skill check the move came through, and ends the game right here when the
    position is final

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the move belongs to.
    :param color: the side playing the move, white or black.
    :param from_sq: origin square of the move.
    :param to_sq: destination square of the move.
    :param promotion: promotion piece letter; a queen when the mover said
        nothing and the pawn has reached the last rank.
    :param skill_kind: kind of check the move came through, None when none ran.
    :param skill_won: whether that check was won, None when none ran.
    :param answers: message type a refusal is quoting back, so the client can
        tell a rejected move from a rejected skill-check input.
    :returns: short outcome word, naming the result reason when the game ended.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    backend = cast(Backend, room.backend)
    prev_len = len(backend.move_history)
    result = backend.try_move(from_sq, to_sq)
    if not result.legal:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.INVALID_MOVE_FORMAT, msg_type=answers))
        return "illegal"
    if result.promotion_required:
        backend.promote(to_sq, PROMO_TYPE_BY_LETTER[promotion or "q"])
    room.plies_ever += 1
    room.note_history_change(app.state.now(), prev_len)
    room.mark_idle_activity(app.state.now() if room.idle_window() is not None else None)
    room.skillcheck_locks.clear()
    room.annotations_white.clear_marks()
    room.annotations_black.clear_marks()
    if room.first_move_at is None:
        room.first_move_at = app.state.now()
    await clear_resyncing(app, room, color)
    room.draw_offered_by = None
    room.takeback_offered_by = None
    san = backend.move_history[-1].san
    if skill_kind is not None:
        room.skillcheck_log.append(SkillCheckOutcome(
            len(backend.move_history), skill_kind,
            cast(bool, skill_won), san))
    log.info("move applied room=%s mover=%s san=%s", room.room_id, color, san)
    applied = MoveAppliedMessage(
        from_sq=coord_from_square(from_sq), to_sq=coord_from_square(to_sq),
        promotion=promotion, san=san,
        clock=clock_snapshot(backend.clock),
        ply=len(backend.move_history),
        skill_check_kind=skill_kind, skill_check_won=skill_won,
    )
    await broadcast(rooms, connections, room, applied)
    game_result = backend.game_result()
    if game_result in RESULT_REASON_BY_GAME_RESULT:
        reason, winner = RESULT_REASON_BY_GAME_RESULT[game_result]
        await finalize_and_broadcast(rooms, connections, room, reason, winner_color=winner)
        return f"applied+result:{reason}"
    await push_idle_window(rooms, connections, room, app.state.now(), force=True)
    return "applied"


def _shot_payload_ok(pending: PendingSkillCheck, msg: SkillCheckShotMessage) -> bool:
    """
    Check that a skill-check input carries the field its challenge needs -- a
    direction for combo, a board square for whack-a-mole -- before anything is
    judged. An input missing its payload is dropped quietly rather than
    counted against the player

    :param pending: the check currently running in this room.
    :param msg: the input the player sent.
    :returns: True when the input is shaped for this kind of check.
    """
    if pending.kind == SkillCheckKind.COMBO:
        return msg.direction is not None
    if pending.kind == SkillCheckKind.WHACK:
        return msg.target_row is not None and msg.target_col is not None
    return True


def _adjudicate_shot(pending: PendingSkillCheck, msg: SkillCheckShotMessage,
                     elapsed: int) -> tuple[Any, bool, bool]:
    """
    Decide what one skill-check input did: whether it hit, and whether that
    hit finishes the check. Every judgement is made against the challenge the
    server rebuilt from its own seed, never against anything the client
    described about it

    :param pending: the check currently running in this room.
    :param msg: the input the player sent.
    :param elapsed: milliseconds into the check, already clamped server-side.
    :returns: the rebuilt challenge, whether the input hit, and whether the
        check is now won.
    """
    is_whack = pending.kind == SkillCheckKind.WHACK
    challenge = pending.challenge
    hit = online.shot_wins(pending.kind, challenge, elapsed, pending.miss_count,
                           pending.deadline_ms, progress=pending.progress,
                           direction=msg.direction,
                           target=(cast(tuple[float, float],
                                        (msg.target_row, msg.target_col))
                                   if is_whack else None),
                           hole_squares=pending.holes if is_whack else None,
                           last_hit_pop=pending.last_hit_pop,
                           flipped=online.adjudicated_flipped(pending.color))
    won = hit and pending.progress + 1 >= online.hits_required(pending.kind, challenge)
    return challenge, hit, won


async def handle_skill_check_shot(app: FastAPI, websocket: WebSocket, room: Room,
                                  color: str, raw: str) -> str:
    """
    Take one input during a live skill check and act on it: land the move when
    the check is beaten, lock that move out for the turn when it is failed,
    otherwise count the hit or the miss. Only the player facing the check may
    resolve it, the elapsed time they report is clamped to what the server
    timed, inputs that arrive too close together are ignored, and the check is
    re-read after every await in case it was resolved meanwhile

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the input arrived on.
    :param room: the room whose check is running.
    :param color: the side that sent the input, white or black.
    :param raw: the websocket text frame holding the input.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    pending = room.pending_skillcheck
    if room.result is not None or pending is None or color != pending.color:
        return "noop"
    try:
        msg = SkillCheckShotMessage.model_validate_json(raw)
    except ValidationError:
        return "noop"
    if not _shot_payload_ok(pending, msg):
        return "noop"
    is_whack = pending.kind == SkillCheckKind.WHACK
    recv_ms = app.state.now_ms()
    raw_elapsed = recv_ms - pending.start_ms
    elapsed = online.adjudicated_elapsed_ms(msg.client_elapsed_ms, recv_ms, pending.start_ms)
    gap = online.min_inter_input_ms(pending.kind)
    if gap > 0 and pending.last_input_ms >= 0 and elapsed - pending.last_input_ms < gap:
        return "noop"
    challenge, hit, won = _adjudicate_shot(pending, msg, elapsed)
    progress_after = pending.progress + (1 if hit else 0)
    log.debug("skillcheck shot room=%s kind=%s raw_ms=%.1f client=%.1f effective=%d "
              "miss=%d subfloor=%s", room.room_id, pending.kind.value,
              raw_elapsed, msg.client_elapsed_ms, elapsed,
              pending.miss_count, elapsed < online.SKILLCHECK_HUMAN_FLOOR_MS)
    if won:
        room.pending_skillcheck = None
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, SkillCheckSpectateShotMessage(
            elapsed_ms=elapsed, miss_count=pending.miss_count, won=won,
            progress=progress_after, direction=msg.direction,
            target_row=msg.target_row, target_col=msg.target_col))
    if won:
        if room.result is not None:
            return "already_over"
        log.info("skillcheck won room=%s mover=%s kind=%s", room.room_id, color,
                 pending.kind.value)
        return await _apply_move(app, room, color, pending.from_sq, pending.to_sq,
                                 pending.promotion, skill_kind=pending.kind.value,
                                 skill_won=True, answers="skill_check_shot")
    if room.pending_skillcheck is not pending:
        return "noop"
    if hit:
        pending.progress = progress_after
        if is_whack:
            pending.last_hit_pop = challenge.pop_up_at(elapsed)
        pending.last_input_ms = elapsed
        return "skillcheck_hit"
    if pending.kind == SkillCheckKind.WHEEL \
            or online.is_past_deadline(elapsed, pending.deadline_ms) \
            or online.check_expired(pending.kind, challenge, elapsed,
                                    pending.miss_count + 1, pending.progress,
                                    pending.last_hit_pop):
        log.info("skillcheck failed room=%s mover=%s kind=%s", room.room_id, color,
                 pending.kind.value)
        await resolve_skillcheck_fail(rooms, connections, room)
        return "skillcheck_fail"
    pending.miss_count += 1
    pending.last_input_ms = elapsed
    return "skillcheck_miss"


async def handle_resign(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                        raw: str) -> str:
    """
    End the game because the sender gave it up, handing the win to their
    opponent. A game that already has a result is left exactly as it is, so a
    resignation racing a checkmate cannot rewrite the ending

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the resignation arrived on.
    :param room: the room being given up.
    :param color: the side resigning, white or black.
    :param raw: the websocket text frame, unused beyond its type.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is not None:
        return "already_over"
    winner = room.opp_color(color)
    await finalize_and_broadcast(rooms, connections, room, Reason.RESIGNATION,
                                 winner_color=winner)
    return "resigned"


async def handle_draw_offer(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                            raw: str) -> str:
    """
    Offer the opponent a draw, or accept theirs when they have one standing,
    which ends the game as agreed. Each side holds at most one offer at a
    time, and a played move clears it

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the offer arrived on.
    :param room: the room the offer belongs to.
    :param color: the side offering, white or black.
    :param raw: the websocket text frame, unused beyond its type.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is not None or room.backend is None:
        return "noop"
    if room.draw_offered_by is not None and room.draw_offered_by != color:
        log.info("draw accepted room=%s by=%s", room.room_id, color)
        room.draw_offered_by = None
        await finalize_and_broadcast(rooms, connections, room, Reason.DRAW_AGREEMENT)
        return "agreed"
    log.info("draw offered room=%s by=%s", room.room_id, color)
    room.draw_offered_by = color
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    await send(opp_ws, DrawOfferedMessage())
    return "offered"


async def handle_draw_response(app: FastAPI, websocket: WebSocket, room: Room,
                               color: str, raw: str) -> str:
    """
    Answer the opponent's draw offer: accepting ends the game as a draw,
    declining simply clears the offer. A player cannot answer their own offer,
    so the reply always comes from the other side

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the answer arrived on.
    :param room: the room the offer belongs to.
    :param color: the side answering, white or black.
    :param raw: the websocket text frame holding the answer.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is not None or room.draw_offered_by is None:
        return "noop"
    try:
        msg = DrawResponseMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    if room.draw_offered_by == color:
        return "self"
    if msg.accept:
        log.info("draw accepted room=%s by=%s", room.room_id, color)
        room.draw_offered_by = None
        await finalize_and_broadcast(rooms, connections, room, Reason.DRAW_AGREEMENT)
        return "accepted"
    log.info("draw declined room=%s by=%s", room.room_id, color)
    room.draw_offered_by = None
    return "declined"


async def _restart_rematch(app: FastAPI, room: Room, color: str) -> str:
    """
    Start the next game in a room both players have agreed to replay: the room
    is reset with the colours swapped and a fresh skill-check secret, then the
    start is announced to both. A room that can no longer be replayed tells
    the player who asked instead

    :param app: the FastAPI application, source of the shared server state.
    :param room: the finished room being replayed.
    :param color: the side whose acceptance triggered the restart.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if not rooms.reset_for_rematch(room.room_id):
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.REMATCH_UNAVAILABLE,
                                    msg_type="rematch_response"))
        return "unavailable"
    log.info("rematch restart room=%s", room.room_id)
    await broadcast_game_start(connections, room, app.state.now, rematch=True)
    return "restarted"


async def handle_rematch_request(app: FastAPI, websocket: WebSocket, room: Room,
                                 color: str, raw: str) -> str:
    """
    Offer another game once this one has finished, which starts immediately
    when both players have offered. Only a player still sitting on the result
    screen may offer, a second offer from the same player is refused, and the
    offer is looked at again after it has been relayed so one withdrawn while
    the relay was in flight is announced as cancelled

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the offer arrived on.
    :param room: the finished room being offered a replay.
    :param color: the side offering, white or black.
    :param raw: the websocket text frame, unused beyond its type.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is None:
        return "noop"
    slot = room.slot(color)
    if slot is None or not slot.at_result:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.REMATCH_UNAVAILABLE,
                                    msg_type="rematch_request"))
        return "unavailable"
    if color in room.rematch_offered_by:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.REMATCH_ALREADY_PENDING,
                                    msg_type="rematch_request"))
        return "duplicate"
    room.rematch_offered_by.add(color)
    rooms.mark_rematch_activity(room)
    if len(room.rematch_offered_by) == 2:
        return await _restart_rematch(app, room, color)
    log.info("rematch requested room=%s by=%s", room.room_id, color)
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, RematchRequestMessage())
        if color not in room.rematch_offered_by:
            await send(opp_ws, RematchUpdateMessage(event="cancelled"))
    return "offered"


async def handle_rematch_response(app: FastAPI, websocket: WebSocket, room: Room,
                                  color: str, raw: str) -> str:
    """
    Answer a rematch offer: accepting restarts the room with the colours
    swapped, declining tells both players the window is over and closes the
    room for good. Answering one's own offer does nothing

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the answer arrived on.
    :param room: the finished room the offer belongs to.
    :param color: the side answering, white or black.
    :param raw: the websocket text frame holding the answer.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is None:
        return "noop"
    try:
        msg = RematchResponseMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    if not room.rematch_offered_by:
        return "noop"
    if color in room.rematch_offered_by:
        return "noop"
    if msg.accept:
        log.info("rematch accepted room=%s by=%s", room.room_id, color)
        rooms.mark_rematch_activity(room)
        return await _restart_rematch(app, room, color)
    log.info("rematch declined room=%s by=%s", room.room_id, color)
    for slot_color in ("white", "black"):
        ws = connections.get_for_color(room, slot_color)
        if ws is not None:
            await send(ws, RematchUpdateMessage(
                event="window_expired" if slot_color == color else "declined"))
    room.rematch_offered_by.clear()
    rooms.drop_room_now(room.room_id)
    return "declined"


async def handle_left_result(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                             raw: str) -> str:
    """
    Note that a player has walked off the result screen back to the menu,
    which withdraws any rematch offer they had standing and tells the
    opponent. The room itself stays alive a while longer, in case the other
    player is still deciding

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the notice arrived on.
    :param room: the finished room being left.
    :param color: the side leaving, white or black.
    :param raw: the websocket text frame, unused beyond its type.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    if room.result is None:
        return "noop"
    slot = room.slot(color)
    if slot is None:
        return "noop"
    slot.at_result = False
    if color in room.rematch_offered_by:
        room.rematch_offered_by.discard(color)
        opp_ws = connections.get_for_color(room, room.opp_color(color))
        if opp_ws is not None:
            await send(opp_ws, RematchUpdateMessage(event="cancelled"))
    return "left_result"


async def handle_takeback_request(app: FastAPI, websocket: WebSocket, room: Room,
                                  color: str, raw: str) -> str:
    """
    Ask the opponent to let the last move be retracted. Only the player who is
    not to move may ask, there has to be a move to take back, and a running
    skill check blocks the request outright

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the request arrived on.
    :param room: the room the request belongs to.
    :param color: the side asking, white or black.
    :param raw: the websocket text frame, unused beyond its type.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    if room.result is not None or room.backend is None:
        return "noop"
    if room.pending_skillcheck is not None:
        return "pending"
    expected = room.color_to_move()
    if color == expected:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.NOT_YOUR_TURN,
                                    msg_type="takeback_request"))
        return "not_your_turn"
    if not room.backend.move_history:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.NO_TAKEBACK_AVAILABLE,
                                    msg_type="takeback_request"))
        return "no_moves"
    log.info("takeback requested room=%s by=%s", room.room_id, color)
    room.takeback_offered_by = color
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    await send(opp_ws, TakebackOfferedMessage())
    return "offered"


async def handle_takeback_response(app: FastAPI, websocket: WebSocket, room: Room,
                                   color: str, raw: str) -> str:
    """
    Answer a takeback request: accepting rewinds the position on the server
    and sends the rewound state to both boards, declining just clears the
    request. An accepted takeback also drops the skill-check record for the
    ply that was popped, stamps the rewind so a heartbeat still in flight is
    read as such, and restarts the idle countdown

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the answer arrived on.
    :param room: the room the request belongs to.
    :param color: the side answering, white or black.
    :param raw: the websocket text frame holding the answer.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is not None or room.takeback_offered_by is None:
        return "noop"
    if room.pending_skillcheck is not None:
        return "pending"
    try:
        msg = TakebackResponseMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    if room.takeback_offered_by == color:
        return "self"
    if msg.accept:
        log.info("takeback accepted room=%s by=%s", room.room_id, color)
        backend = cast(Backend, room.backend)
        popped_ply = len(backend.move_history)
        backend.undo()
        room.note_history_change(app.state.now(), popped_ply)
        room.skillcheck_log = [e for e in room.skillcheck_log if e.ply < popped_ply]
        room.takeback_offered_by = None
        room.annotations_white.clear_marks()
        room.annotations_black.clear_marks()
        await broadcast(rooms, connections, room, TakebackAppliedMessage(
            fen=export_fen(backend),
            clock=clock_snapshot(backend.clock),
            ply=len(backend.move_history),
        ))
        if room.result is None and room.idle_window() is not None:
            room.mark_idle_activity(app.state.now())
            await push_idle_window(rooms, connections, room, app.state.now(), force=True)
        return "accepted"
    log.info("takeback declined room=%s by=%s", room.room_id, color)
    room.takeback_offered_by = None
    return "declined"


async def handle_give_time(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                           raw: str) -> str:
    """
    Give the opponent clock time as a gift, sized by how long the giver held
    the button down. The hold is turned into whole chunks server-side, so what
    lands on the clock comes from the server's own table rather than from the
    client's arithmetic

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the gift arrived on.
    :param room: the room whose clocks are being changed.
    :param color: the side giving the time, white or black.
    :param raw: the websocket text frame holding the hold duration.
    :returns: short outcome word for the dispatch log.
    """
    rooms = app.state.rooms
    connections = app.state.connections
    if room.result is not None or room.backend is None or room.backend.clock is None:
        return "noop"
    try:
        hold_ms = GiveTimeMessage.model_validate_json(raw).hold_ms
    except ValidationError:
        return "invalid"
    ticks = max(1, hold_ms // GIVE_TIME_TICK_MS)
    opp_color_str = room.opp_color(color)
    opp_piece_color = (
        PieceColor.WHITE if opp_color_str == "white" else PieceColor.BLACK
    )
    added = room.backend.clock.add_time(opp_piece_color, GIVE_TIME_SECONDS * ticks)
    log.info("give_time room=%s by=%s hold_ms=%d ticks=%d added=%.2f",
             room.room_id, color, hold_ms, ticks, added)
    await broadcast(rooms, connections, room, TimeGrantedMessage(
        granted_by=color, seconds_added=added,
        clock=clock_snapshot(room.backend.clock),
    ))
    return "granted" if added > 0 else "capped"


async def _notify_opp_state(connections: ConnectionRegistry, room: Room, color: str,
                            state: Literal["connected", "reconnecting", "resyncing"]
                            ) -> None:
    """
    Tell one player how their opponent's side is doing, which is what turns a
    board that has gone quiet into a visible note instead of an apparent
    walk-out

    :param connections: registry of live sockets for every room.
    :param room: the room both players are in.
    :param color: the side the news is about; it goes to the other one.
    :param state: connected, reconnecting or resyncing.
    """
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, ConnectionStatusMessage(opp_state=state))


def _ply_in_transit(room: Room, ply: int, now: float) -> bool:
    """
    Judge whether a heartbeat reporting an out-of-date ply is simply describing
    the position the client had before a very recent change, rather than a
    board that has drifted. It is direction-aware for free: after a move only a
    client one behind is excused, after a takeback only one ahead

    :param room: the room the heartbeat belongs to.
    :param ply: the ply the client says it is on.
    :param now: monotonic seconds the heartbeat was handled at.
    :returns: True while a recent change still explains that ply.
    """
    return any(
        now - change.at < RESYNC_TRANSIT_GRACE_SECONDS
        and (change.prev_len is None or ply == change.prev_len)
        for change in room.history_changes
    )


def _newest_change_age(room: Room, now: float) -> float:
    """
    Measure how long ago this room's move history last changed, the age quoted
    in the resync log line. Past the game-start gate a room always has at least
    one stamp, and a room without one reads as an age of zero

    :param room: the room whose history stamps are being read.
    :param now: monotonic seconds to measure back from.
    :returns: seconds since the newest stamp, 0.0 when there is none.
    """
    if not room.history_changes:
        return 0.0
    return max(now - room.history_changes[-1].at, 0.0)


async def set_resyncing(connections: ConnectionRegistry, room: Room, color: str,
                        *, client_ply: int, server_ply: int, age_s: float,
                        streak: int) -> None:
    """
    Mark a player as rebuilding their game state and tell the opponent once,
    so the pause reads as a repair in progress. Repeat calls while the same
    spell lasts do nothing, which is also what keeps the one operator line to
    one per lagging spell rather than one per heartbeat

    :param connections: registry of live sockets for every room.
    :param room: the room the lagging player is in.
    :param color: the side that has fallen behind, white or black.
    :param client_ply: the ply the client last reported.
    :param server_ply: the ply the server's own board is on.
    :param age_s: seconds since this room's history last changed.
    :param streak: how many judged heartbeats have mismatched in a row.
    """
    slot = room.slot(color)
    if slot is not None and not slot.desync_active:
        slot.desync_active = True
        log.info("resync directive room=%s color=%s client_ply=%d server_ply=%d "
                 "age_s=%.2f streak=%d",
                 room.room_id, color, client_ply, server_ply, age_s, streak)
        await _notify_opp_state(connections, room, color, "resyncing")


async def clear_resyncing(app: FastAPI, room: Room, color: str) -> None:
    """
    Mark a player as caught up again and tell the opponent, called whenever
    the client proves it is on the right ply. It also lets the notification
    gate reopen shortly, so a fresh lagging spell is reported promptly instead
    of being swallowed by the previous interval

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the recovered player is in.
    :param color: the side that has caught up, white or black.
    """
    slot = room.slot(color)
    if slot is None or not slot.desync_active:
        return
    slot.desync_active = False
    slot.ply_mismatch_streak = 0
    log.info("resync cleared room=%s color=%s", room.room_id, color)
    _resync_gate(app).reopen((room.room_id, color, RESYNC_NOTIFY), app.state.now(),
                             RESYNC_NOTIFY_FLAP_FLOOR_SECONDS)
    await _notify_opp_state(app.state.connections, room, color, "connected")


async def handle_ping(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                      raw: str) -> str:
    """
    Answer the client's heartbeat, and use the ply it reports to spot a board
    that has genuinely drifted behind the server's. Everything that legitimately
    puts the two out of step for a moment is forgiven first: a client not
    sitting on a live online board reports no ply, a game not yet announced is
    never judged, and a ply explained by a very recent move or takeback is read
    as news still in flight. Only a mismatch that survives two judged
    heartbeats earns a repair, and both the order to refetch and the note to
    the opponent are debounced on top

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the heartbeat arrived on.
    :param room: the room the heartbeat belongs to.
    :param color: the side that sent it, white or black.
    :param raw: the websocket text frame holding the client's ply.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    try:
        msg = PingMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid_ping"
    await send(connections.get_for_color(room, color), PongMessage())
    if room.result is not None or room.backend is None:
        return "ping"
    if room.pending_skillcheck is not None:
        return "ping_pending"
    server_ply = len(room.backend.move_history)
    slot = cast(PlayerSlot, room.slot(color))
    if msg.ply == server_ply:
        slot.ply_mismatch_streak = 0
        await clear_resyncing(app, room, color)
        return "ping"
    if not room.game_start_broadcast:
        return "ping_pregame"
    if msg.ply is None:
        return "ping_offboard"
    now = app.state.now()
    if _ply_in_transit(room, msg.ply, now):
        return "ping_inflight"
    slot.ply_mismatch_streak += 1
    if slot.ply_mismatch_streak < RESYNC_STABLE_MISMATCH_HEARTBEATS:
        return "ping_strike"
    gate = _resync_gate(app)
    if (not slot.desync_active
            and gate.allow((room.room_id, color, RESYNC_NOTIFY), now,
                           RESYNC_NOTIFY_MIN_INTERVAL_SECONDS)):
        await set_resyncing(connections, room, color, client_ply=msg.ply,
                            server_ply=server_ply,
                            age_s=_newest_change_age(room, now),
                            streak=slot.ply_mismatch_streak)
    if gate.allow((room.room_id, color, RESYNC_DIRECTIVE), now,
                  RESYNC_DIRECTIVE_MIN_INTERVAL_SECONDS):
        await send(connections.get_for_color(room, color),
                   ResyncDirectiveMessage(server_ply=server_ply))
        return "ping_directed"
    return "ping_gated"


async def _relay_guard(websocket: WebSocket, room: Room, color: str, limiter: Any,
                       msg_type: str) -> str | None:
    """
    Shared gate for the frames players relay to each other, the shared marks
    and the quick chat. Nothing is relayed outside a live game, and each
    player spends an allowance of their own rather than the room's

    :param websocket: the socket the frame arrived on.
    :param room: the room the frame belongs to.
    :param color: the side that sent it, white or black.
    :param limiter: this frame kind's per-player allowance.
    :param msg_type: message type quoted back in a refusal.
    :returns: a short outcome word when the frame is refused, else None.
    """
    if room.backend is None or room.result is not None:
        return "noop"
    if not limiter.hit(cast(PlayerSlot, room.slot(color)).client_uuid):
        await send(websocket, ErrorMessage(reason=Reason.RATE_LIMITED, msg_type=msg_type))
        return "rate_limited"
    return None


async def handle_annotations_state(app: FastAPI, websocket: WebSocket, room: Room,
                                   color: str, raw: str) -> str:
    """
    Take a player's whole set of shared board marks and pass it to the
    opponent, or switch their sharing off. Marks are screened before they
    travel, and a player who has been muted or has run through their screening
    budget has their sharing stopped instead of relayed

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the marks arrived on.
    :param room: the room the marks belong to.
    :param color: the side sharing them, white or black.
    :param raw: the websocket text frame holding the whole mark set.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    verdict = await _relay_guard(websocket, room, color,
                                 app.state.annotation_limiter, "annotations_state")
    if verdict is not None:
        return verdict
    try:
        msg = AnnotationsStateMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    store = room.annotations_for(color)
    if msg.sharing and store.share_muted and app.state.moderation_enabled:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.SHARE_MUTED, msg_type="annotations_state"))
        return "muted"
    if msg.sharing and _over_moderation_budget(app, room, color):
        return await _suppress_sharing(app, room, color, "annotations_state")
    was_sharing = store.sharing
    if msg.sharing != store.sharing:
        log.info("annotations sharing room=%s by=%s on=%s",
                 room.room_id, color, msg.sharing)
    store.sharing = msg.sharing
    if not msg.sharing:
        store.clear_marks()
        await _relay_plain(connections, room, color,
                           AnnotationsStateMessage(sharing=False, highlights=[], arrows=[]))
        return "relayed"
    if not was_sharing:
        store.opp_hidden_notice_sent = False
    store.highlights = set(msg.highlights)
    store.arrows = [(a.from_sq, a.to_sq) for a in msg.arrows]
    return await _relay_or_moderate(app, room, color, msg, None, "annotations_state")


async def handle_annotation_delta(app: FastAPI, websocket: WebSocket, room: Room,
                                  color: str, raw: str) -> str:
    """
    Take a single added or removed mark and pass it on, the cheap path used
    while a player is actually drawing. The stored set is capped, the change
    is screened exactly as a whole set would be, and a delta that names no
    usable square is dropped

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the change arrived on.
    :param room: the room the marks belong to.
    :param color: the side drawing, white or black.
    :param raw: the websocket text frame holding the single change.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    verdict = await _relay_guard(websocket, room, color,
                                 app.state.annotation_limiter, "annotation_delta")
    if verdict is not None:
        return verdict
    try:
        msg = AnnotationDeltaMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    if msg.kind == "highlight":
        if msg.square is None:
            return "invalid"
    else:
        if msg.from_sq is None or msg.to_sq is None or msg.from_sq == msg.to_sq:
            return "invalid"
    store = room.annotations_for(color)
    if store.share_muted and app.state.moderation_enabled:
        await send(connections.get_for_color(room, color),
                     ErrorMessage(reason=Reason.SHARE_MUTED, msg_type="annotation_delta"))
        return "muted"
    if _over_moderation_budget(app, room, color):
        return await _suppress_sharing(app, room, color, "annotation_delta")
    changed: str | tuple[str, str] | None
    if msg.kind == "highlight":
        if msg.action == "add":
            if (msg.square not in store.highlights
                    and len(store.highlights) >= MAX_SHARED_HIGHLIGHTS):
                return "capped"
            store.highlights.add(cast(str, msg.square))
        else:
            store.highlights.discard(cast(str, msg.square))
        changed = msg.square if msg.action == "add" else None
    else:
        pair = cast(tuple[str, str], (msg.from_sq, msg.to_sq))
        if msg.action == "add":
            if pair not in store.arrows:
                if len(store.arrows) >= MAX_SHARED_ARROWS:
                    return "capped"
                store.arrows.append(pair)
        elif pair in store.arrows:
            store.arrows.remove(pair)
        changed = pair if msg.action == "add" else None
    return await _relay_or_moderate(app, room, color, msg, changed, "annotation_delta")


def _last_move_context(room: Room) -> tuple[str, ...]:
    """
    Give the squares of the move just played, which the mark screener treats
    as ink already on the board. Without it, an arrow drawn along the last
    move would read as part of whatever shape is being looked for

    :param room: the room whose board is being screened.
    :returns: origin and destination squares, empty before the first move.
    """
    backend = cast(Backend, room.backend)
    if not backend.move_history:
        return ()
    move = backend.move_history[-1].move
    return (coord_from_square(move.from_sq), coord_from_square(move.to_sq))


def _moderation_inputs(room: Room, color: str) -> tuple[
        list[tuple[str, str]], set[str],
        list[tuple[str, str]], set[str], tuple[str, ...]]:
    """
    Gather everything the mark screener has to judge at once: this player's
    marks, the opponent's, and the last move for context. Both sides are
    needed because a forbidden shape can be drawn half by each player

    :param room: the room whose marks are being screened.
    :param color: the side whose change triggered the screening.
    :returns: own arrows and highlights, the opponent's, and the last-move
        squares.
    """
    store = room.annotations_for(color)
    opp_store = room.annotations_for(room.opp_color(color))
    return (list(store.arrows), set(store.highlights),
            list(opp_store.arrows), set(opp_store.highlights),
            _last_move_context(room))


def _over_moderation_budget(app: FastAPI, room: Room, color: str) -> bool:
    """
    Ask whether this player has already spent their share of screening time.
    Screening is charged in measured CPU seconds against a per-player and a
    per-room allowance, so a flood of marks costs the flooder their sharing
    rather than costing the server its responsiveness

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side that would be screened, white or black.
    :returns: True when the player is over budget and must be cut off.
    """
    if not app.state.moderation_enabled:
        return False
    return _moderation_load(app).over_budget(room.room_id, color, app.state.now())


async def _suppress_sharing(app: FastAPI, room: Room, color: str, msg_type: str) -> str:
    """
    Force a player's mark sharing off because screening it costs too much,
    clearing what they had shared and telling the opponent those marks are
    gone. The player is told their traffic was refused, not that their drawing
    was judged

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side being cut off, white or black.
    :param msg_type: message type quoted back in the refusal.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    store = room.annotations_for(color)
    stopped = store.sharing
    store.sharing = False
    store.clear_marks()
    if stopped:
        log.warning("annotations load-suppressed room=%s color=%s", room.room_id, color)
        await _relay_plain(connections, room, color,
                           AnnotationsStateMessage(sharing=False, highlights=[], arrows=[]))
    await send(connections.get_for_color(room, color),
               ErrorMessage(reason=Reason.RATE_LIMITED, msg_type=msg_type))
    return "load_suppressed"


def _moderate(own_arrows: list[tuple[str, str]], own_highlights: set[str],
              opp_arrows: list[tuple[str, str]], opp_highlights: set[str],
              context: tuple[str, ...],
              changed: str | tuple[str, str] | None
              ) -> tuple[detector.Verdict | None, bool]:
    """
    Run the shared-mark screener and decide what it found: this player's marks
    first, then both sides' marks together, since a forbidden shape can be
    split across the two boards. This is the pure part, safe to run off the
    event loop

    :param own_arrows: arrows the acting player is sharing.
    :param own_highlights: squares the acting player is highlighting.
    :param opp_arrows: arrows the opponent is sharing.
    :param opp_highlights: squares the opponent is highlighting.
    :param context: last-move squares, treated as ink already on the board.
    :param changed: the mark just drawn, to focus the search, or None.
    :returns: the verdict worth acting on plus whether it came from the two
        boards combined, or no verdict at all.
    """
    own = detector.detect(own_arrows, own_highlights, changed=changed, context=context)
    if own.kind == detector.BLOCKED:
        return own, False
    union_arrows, union_highlights = detector.union_sides(
        own_arrows, own_highlights, opp_arrows, opp_highlights)
    union = detector.detect(union_arrows, union_highlights,
                            changed=changed, context=context)
    if union.kind == detector.BLOCKED:
        return union, True
    if own.kind == detector.SUSPECT:
        return own, False
    return None, False


def _own_matched(store: SharedAnnotations, verdict: detector.Verdict
                 ) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Narrow a screener verdict down to the marks this player actually owns, so
    the reply names only their own arrows and highlights even when the shape
    was found across both boards

    :param store: the acting player's shared marks.
    :param verdict: what the screener matched.
    :returns: the matched arrows and highlights belonging to this player.
    """
    arrows = [a for a in verdict.matched_arrows if (a[0], a[1]) in store.arrows]
    highlights = [h for h in verdict.matched_highlights if h in store.highlights]
    return arrows, highlights


async def _relay_or_moderate(app: FastAPI, room: Room, color: str,
                             msg: AnnotationsStateMessage | AnnotationDeltaMessage,
                             changed: str | tuple[str, str] | None, msg_type: str) -> str:
    """
    Send a mark update on to the opponent, screening it first unless screening
    is switched off for this server. Both the whole-set and the single-mark
    handlers come through here, so the two behave identically

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side that made the change, white or black.
    :param msg: the mark update waiting to be relayed.
    :param changed: the mark just drawn, to focus the search, or None.
    :param msg_type: message type quoted back in any refusal.
    :returns: short outcome word for the dispatch log.
    """
    if not app.state.moderation_enabled:
        await _relay_to_opp(app, room, color, msg, msg_type)
        return "relayed"
    return await _moderate_relay(app, room, color, msg, changed, msg_type)


def _timed_moderate(*args: Any) -> tuple[tuple[detector.Verdict | None, bool], float]:
    """
    Run the screener and measure what it cost in CPU time, which is the
    currency the per-player and per-room budgets are charged in. It runs on a
    worker thread, where wall clock would measure waiting rather than work

    :param args: the screener inputs, passed straight through.
    :returns: the screener outcome and the CPU seconds it consumed.
    """
    started = time.thread_time()
    return _moderate(*args), time.thread_time() - started


async def _moderate_relay(app: FastAPI, room: Room, color: str,
                          relay_msg: AnnotationsStateMessage | AnnotationDeltaMessage,
                          changed: str | tuple[str, str] | None, msg_type: str) -> str:
    """
    Screen a mark update off the event loop and act on the verdict: relay it,
    relay it with a warning, or strip the marks it refused. Only a couple of
    screenings run at once and each one's measured cost is charged to the
    player and the room, so anyone who cannot get in or is over budget has
    their sharing stopped. The room is re-read afterwards, because the game
    may have ended while the screening ran

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side that made the change, white or black.
    :param relay_msg: the mark update waiting to be relayed.
    :param changed: the mark just drawn, to focus the search, or None.
    :param msg_type: message type quoted back in any refusal.
    :returns: short outcome word for the dispatch log.
    """
    load = _moderation_load(app)
    inputs = _moderation_inputs(room, color)
    try:
        await asyncio.wait_for(load.semaphore.acquire(), load.admission_timeout)
    except asyncio.TimeoutError:
        log.warning("moderation admission timed out room=%s color=%s", room.room_id, color)
        return await _suppress_sharing(app, room, color, msg_type)
    try:
        outcome, cpu_seconds = await asyncio.to_thread(_timed_moderate, *inputs, changed)
    finally:
        load.semaphore.release()
    load.charge(room.room_id, color, cpu_seconds, app.state.now())
    verdict, is_union = outcome
    if room.backend is None or room.result is not None:
        return "noop"
    if verdict is None:
        await _relay_to_opp(app, room, color, relay_msg, msg_type)
        return "relayed"
    if verdict.kind == detector.BLOCKED:
        await _handle_block(app, room, color, verdict, is_union)
        return "blocked"
    await _relay_to_opp(app, room, color, relay_msg, msg_type)
    log.warning("marks suspect room=%s color=%s pattern=%s",
                room.room_id, color, str(verdict.pattern_id))
    store = room.annotations_for(color)
    sus_arrows, sus_highlights = _own_matched(store, verdict)
    await send(app.state.connections.get_for_color(room, color),
                 AnnotationsBlockedMessage(
                     action="suspect",
                     highlights=sus_highlights,
                     arrows=arrow_wires(sus_arrows),
                     share_muted=store.share_muted))
    return "suspect"


async def _handle_block(app: FastAPI, room: Room, color: str, verdict: detector.Verdict,
                        is_union: bool) -> None:
    """
    Take away the marks the screener refused and put every board back in
    step: the offender's copy is stripped, the opponent's too when the shape
    was only there across both, and enough refusals mute that player's sharing
    for the rest of the game

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side whose marks were refused, white or black.
    :param verdict: what the screener matched.
    :param is_union: True when the shape appears only across both boards.
    """
    connections = app.state.connections
    store = room.annotations_for(color)
    opp_color = room.opp_color(color)
    own_arrows, own_highlights = _own_matched(store, verdict)
    store.strip(verdict.matched_arrows, verdict.matched_highlights)
    if is_union:
        room.annotations_for(opp_color).strip(
            verdict.matched_arrows, verdict.matched_highlights)
    muted = store.register_trip(MODERATION_TRIP_LIMIT)
    log.warning("marks blocked room=%s color=%s pattern=%s trip=%d muted=%s",
                room.room_id, color, str(verdict.pattern_id),
                store.trip_count, muted)
    await _corrective_snapshot(app, room, color)
    if is_union:
        await _corrective_snapshot(app, room, opp_color)
    await send(connections.get_for_color(room, color),
                 AnnotationsBlockedMessage(
                     action="blocked",
                     highlights=own_highlights,
                     arrows=arrow_wires(own_arrows),
                     share_muted=muted))


async def _corrective_snapshot(app: FastAPI, room: Room, source_color: str) -> None:
    """
    Resend one player's whole mark set to the other after marks were stripped,
    so the receiving board cannot keep drawing something the server has since
    removed. Skipped when that player has chosen not to see the opponent's
    marks at all

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param source_color: the side whose marks are being resent.
    """
    connections = app.state.connections
    target_color = room.opp_color(source_color)
    if room.hides_opponent_marks(target_color):
        return
    target_ws = connections.get_for_color(room, target_color)
    if target_ws is None:
        return
    store = room.annotations_for(source_color)
    await send(target_ws, AnnotationsStateMessage(
        sharing=True, highlights=sorted(store.highlights),
        arrows=arrow_wires(store.arrows)))


async def _relay_plain(connections: ConnectionRegistry, room: Room, color: str,
                       msg: AnnotationsStateMessage) -> None:
    """
    Push a mark state straight to the opponent with no screening and no
    visibility check, for the corrective updates the server itself generates
    when it has just cleared somebody's marks

    :param connections: registry of live sockets for every room.
    :param room: the room both players are in.
    :param color: the side the state is about; it goes to the other one.
    :param msg: the mark state to send.
    """
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, msg)


async def _relay_to_opp(app: FastAPI, room: Room, color: str,
                        msg: AnnotationsStateMessage | AnnotationDeltaMessage,
                        msg_type: str) -> None:
    """
    Deliver a mark update to the opponent, unless they have chosen not to see
    the other player's marks. In that case nothing is sent and the sharer is
    told once per sharing spell, so they know their drawing is going nowhere

    :param app: the FastAPI application, source of the shared server state.
    :param room: the room the marks belong to.
    :param color: the side that made the change, white or black.
    :param msg: the mark update to deliver.
    :param msg_type: message type quoted back in the notice.
    """
    connections = app.state.connections
    opp_color = room.opp_color(color)
    if room.hides_opponent_marks(opp_color):
        store = room.annotations_for(color)
        if not store.opp_hidden_notice_sent:
            store.opp_hidden_notice_sent = True
            await send(connections.get_for_color(room, color),
                         ErrorMessage(reason=Reason.OPP_HIDES_MARKS, msg_type=msg_type))
        return
    opp_ws = connections.get_for_color(room, opp_color)
    if opp_ws is not None:
        await send(opp_ws, msg)


async def handle_set_marks_visibility(app: FastAPI, websocket: WebSocket, room: Room,
                                      color: str, raw: str) -> str:
    """
    Switch whether this player sees the opponent's shared marks. Turning them
    back on resends the opponent's current set at once, so the board is right
    again without waiting for their next stroke

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the choice arrived on.
    :param room: the room the marks belong to.
    :param color: the side changing what they see, white or black.
    :param raw: the websocket text frame holding the choice.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    try:
        msg = SetMarksVisibilityMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    slot = room.slot(color)
    if slot is None or slot.hide_opp_marks == msg.hide_opp:
        return "noop"
    slot.hide_opp_marks = msg.hide_opp
    log.info("marks visibility room=%s color=%s hide_opp=%s",
             room.room_id, color, msg.hide_opp)
    if msg.hide_opp:
        return "hidden"
    opp_store = room.annotations_for(room.opp_color(color))
    await send(connections.get_for_color(room, color), AnnotationsStateMessage(
        sharing=opp_store.sharing, highlights=sorted(opp_store.highlights),
        arrows=arrow_wires(opp_store.arrows)))
    return "shown"


async def handle_quick_chat(app: FastAPI, websocket: WebSocket, room: Room, color: str,
                            raw: str) -> str:
    """
    Pass a quick-chat phrase to the opponent as its index in the fixed list.
    It is rate limited like every other relayed frame, and no free text ever
    crosses between players

    :param app: the FastAPI application, source of the shared server state.
    :param websocket: the socket the phrase arrived on.
    :param room: the room the players are in.
    :param color: the side that sent it, white or black.
    :param raw: the websocket text frame holding the phrase index.
    :returns: short outcome word for the dispatch log.
    """
    connections = app.state.connections
    verdict = await _relay_guard(websocket, room, color,
                                 app.state.chat_limiter, "quick_chat")
    if verdict is not None:
        return verdict
    try:
        msg = QuickChatMessage.model_validate_json(raw)
    except ValidationError:
        return "invalid"
    log.info("quick chat room=%s by=%s preset=%d", room.room_id, color, msg.preset)
    opp_ws = connections.get_for_color(room, room.opp_color(color))
    if opp_ws is not None:
        await send(opp_ws, QuickChatReceivedMessage(preset=msg.preset, sender=color))
    return "relayed"


HANDLERS = {
    "move": handle_move,
    "resign": handle_resign,
    "draw_offer": handle_draw_offer,
    "draw_response": handle_draw_response,
    "rematch_request": handle_rematch_request,
    "rematch_response": handle_rematch_response,
    "left_result": handle_left_result,
    "takeback_request": handle_takeback_request,
    "takeback_response": handle_takeback_response,
    "give_time": handle_give_time,
    "ping": handle_ping,
    "skill_check_shot": handle_skill_check_shot,
    "annotations_state": handle_annotations_state,
    "annotation_delta": handle_annotation_delta,
    "set_marks_visibility": handle_set_marks_visibility,
    "quick_chat": handle_quick_chat,
}
