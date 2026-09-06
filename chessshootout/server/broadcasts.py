from collections.abc import Callable
from typing import cast

from chessshootout.backend.backend import Backend
from chessshootout.backend.fen import export_fen
from chessshootout.backend.utils import coord_from_square
from chessshootout.skillcheck.types import SkillCheckOutcome

from chessshootout.server import logging_setup
from chessshootout.server.connections import ConnectionRegistry, broadcast, send
from chessshootout.server.protocol import (
    GameStartMessage, IdleWindowMessage, IdleWindowWire, ResultMessage,
    SkillCheckResultMessage)
from chessshootout.server.rooms import PendingSkillCheck, Room, RoomManager


log = logging_setup.get_logger("chess.server.app")


IDLE_WINDOW_PUSH_MIN_INTERVAL_SECONDS = 2.0


async def finalize_and_broadcast(rooms: RoomManager, connections: ConnectionRegistry,
                                 room: Room, reason: str,
                                 winner_color: str | None = None) -> None:
    """
    End a game once and tell both players how it ended. The room decides whether
    this attempt is the one that lands, and a concurrent caller that lost the
    race broadcasts nothing; the frame that does go out is built from the stored
    result, so both players are always told the same outcome. Every game that
    ends passes through here, so this is also where the one line describing the
    ending is logged, whose duration is seconds since pairing

    :param rooms: room manager that records the result
    :param connections: registry used to reach both players
    :param room: room whose game is ending
    :param reason: why it ended, one of the shared reason codes
    :param winner_color: winning side, or None for a draw or an abort
    """
    applied = rooms.finalize_result(room.room_id, reason, winner_color=winner_color)
    if not applied:
        return
    result_reason, result_winner = cast(tuple[str, str | None], room.result)
    log.info("game finalized room=%s reason=%s winner=%s plies=%d duration_s=%.1f",
             room.room_id, result_reason, result_winner or "none", room.plies_ever,
             cast(float, room.ended_at) - cast(float, room.started_at))
    await broadcast(rooms, connections, room,
                    ResultMessage(reason=result_reason, winner_color=result_winner))


def idle_window_wire(room: Room, now: float) -> IdleWindowWire | None:
    """
    Describe the countdown running against the silent side to move: what happens
    when it reaches zero, whose turn it is and how long is left. A finished game,
    a game with no window armed at this ply, or one with nobody to move has no
    countdown at all

    :param room: room to describe
    :param now: monotonic time in seconds, used for the remaining time
    :returns: the countdown to show, or None when none is running
    """
    color = room.color_to_move()
    if room.result is not None or room.idle_since is None or color is None:
        return None
    window = room.idle_window()
    if window is None:
        return None
    return IdleWindowWire(
        outcome=window.outcome, color=color,
        seconds_remaining=room.idle_remaining(now))


async def push_idle_window(rooms: RoomManager, connections: ConnectionRegistry,
                           room: Room, now: float, *, force: bool = False) -> None:
    """
    Push the idle countdown to both players so the timer on screen agrees with
    the one the server keeps. Repeat pushes are throttled, since the sweep would
    otherwise send one every tick; a caller that has just restarted the window
    forces the send so the players see the reset immediately

    :param rooms: room manager passed through to the broadcast
    :param connections: registry used to reach both players
    :param room: room whose countdown is being pushed
    :param now: monotonic time in seconds, both the remaining time and the
        throttle are measured from it
    :param force: send even when the last push was recent
    """
    wire = idle_window_wire(room, now)
    if wire is None:
        return
    if (not force and room.idle_pushed_at is not None
            and now - room.idle_pushed_at < IDLE_WINDOW_PUSH_MIN_INTERVAL_SECONDS):
        return
    room.note_idle_push(now)
    log.debug("idle window push room=%s outcome=%s", room.room_id, wire.outcome)
    await broadcast(rooms, connections, room, IdleWindowMessage(**wire.model_dump()))


async def resolve_skillcheck_fail(rooms: RoomManager, connections: ConnectionRegistry,
                                  room: Room) -> PendingSkillCheck | None:
    """
    Lose the skill check that is running: the move never lands, that exact pair
    of squares is locked for the rest of the turn, the miss is written into the
    room's shootout log and both boards are told. Every surface that can notice a
    dead check -- the move gate, resume and the sweep -- funnels through here, so
    a check is only ever resolved once

    :param rooms: room manager passed through to the broadcast
    :param connections: registry used to reach both players
    :param room: room whose pending check is being failed
    :returns: the check that was resolved, or None when there was none
    """
    pending = room.pending_skillcheck
    if pending is None:
        return None
    backend = cast(Backend, room.backend)
    room.pending_skillcheck = None
    room.skillcheck_locks.add((pending.from_sq, pending.to_sq))
    room.skillcheck_log.append(SkillCheckOutcome(
        len(backend.move_history) + 1, pending.kind.value, False,
        backend.preview_san(pending.from_sq, pending.to_sq, pending.promotion)))
    await broadcast(rooms, connections, room, SkillCheckResultMessage(
        won=False,
        from_sq=coord_from_square(pending.from_sq),
        to_sq=coord_from_square(pending.to_sq),
    ))
    return pending


async def broadcast_game_start(connections: ConnectionRegistry, room: Room,
                               now: Callable[[], float], rematch: bool = False) -> None:
    """
    Start the game on both screens: each player is sent the opening position,
    both names with their countries and series scores, the time control and
    their own color. The frames are built per player because the color differs,
    and each carries how long ago the game actually started so a client that
    joined late does not run its clock from the wrong instant. The moment is
    stamped as a history change with no previous length, since what a client
    was showing before a game start cannot be known

    :param connections: registry used to reach both players
    :param room: paired room whose game is starting
    :param now: monotonic seconds source, read for the elapsed-since-start value
        and for the history-change stamp
    :param rematch: True when this game follows an accepted rematch offer
    """
    fen = export_fen(cast(Backend, room.backend))
    started_seconds_ago = max(now() - (room.started_at or now()), 0.0)
    sent = []
    for color in ("white", "black"):
        ws = connections.get_for_color(room, color)
        if ws is None:
            continue
        await send(ws, GameStartMessage(
            fen=fen,
            white_name=room.white.nickname if room.white else "",
            black_name=room.black.nickname if room.black else "",
            time_minutes=room.time_minutes,
            increment_seconds=room.increment_seconds,
            your_color=color,
            started_seconds_ago=started_seconds_ago,
            white_score=room.score_for("white"),
            black_score=room.score_for("black"),
            white_country=room.white.country if room.white else None,
            black_country=room.black.country if room.black else None,
            rematch=rematch,
        ))
        sent.append(color)
    room.note_history_change(now(), None)
    room.game_start_broadcast = True
    log.info("game_start broadcast room=%s sent_to=%s elapsed=%.2f",
             room.room_id, sent, started_seconds_ago)
