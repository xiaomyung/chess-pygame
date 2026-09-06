from collections.abc import Callable
from typing import cast

from fastapi import WebSocketDisconnect

from chessshootout.server import logging_setup
from chessshootout.server.broadcasts import finalize_and_broadcast, resolve_skillcheck_fail
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.protocol import (
    ConnectionStatusMessage, ErrorMessage, GRACE_SECONDS,
    QUEUE_MAX_WAIT_SECONDS, Reason, RematchUpdateEvent, RematchUpdateMessage,
)
from chessshootout.server.rooms import (
    REMATCH_ABSOLUTE_CAP_SECONDS, REMATCH_IDLE_SECONDS, POST_GAME_DISCONNECT_GRACE,
    PlayerSlot, Room, RoomManager,
)


log = logging_setup.get_logger("chess.server.app")


PREGAME_CONNECT_GRACE_SECONDS = 5.0
WS_CLOSE_QUEUE_TIMEOUT = 4004


RESULT_REASON_BY_GAME_RESULT = {
    "white_wins": (Reason.CHECKMATE, "white"),
    "black_wins": (Reason.CHECKMATE, "black"),
    "white_wins_on_time": (Reason.TIMEOUT, "white"),
    "black_wins_on_time": (Reason.TIMEOUT, "black"),
    "draw_stalemate": (Reason.DRAW_STALEMATE, None),
    "draw_repetition": (Reason.DRAW_REPETITION, None),
    "draw_fifty_move": (Reason.DRAW_FIFTY_MOVE, None),
    "draw_insufficient_material": (Reason.DRAW_INSUFFICIENT_MATERIAL, None),
}


class Sweep:
    """
    The server's housekeeping pass over every room, run on a short timer:
    clocks, skill-check deadlines, idle countdowns, disconnects, abandoned
    queue slots and finished rooms. It is where everything that has to happen
    without a player doing anything happens
    """

    def __init__(self, rooms: RoomManager, connections: ConnectionRegistry,
                 now_provider: Callable[[], float], now_ms: Callable[[], float]) -> None:
        """
        Bind the sweep to the state it walks and the clocks it reads. Both
        clocks are injected, so tests can drive every timeout deterministically
        instead of waiting real minutes out

        :param rooms: the room manager holding every queued and active room.
        :param connections: registry of live sockets for every room.
        :param now_provider: returns monotonic seconds.
        :param now_ms: returns the same clock in milliseconds.
        """
        self.rooms = rooms
        self.connections = connections
        self._now = now_provider
        self._now_ms = now_ms

    async def step_all(self) -> None:
        """
        Run one whole housekeeping pass in the order the game's rules need it:
        skill-check deadlines, then clocks and idle windows, then the
        connection timeouts, then the queue and the finished rooms
        """
        await self.step_skillcheck_deadline()
        await self.step_clock_and_idle_windows()
        await self.step_heartbeat_timeout()
        await self.step_grace_expired()
        self.step_drop_orphans_pre_game()
        self.step_reap_abandoned_queue()
        await self.step_reap_timed_out_queue()
        await self.step_post_game()
        self.rooms.gc_finished_rooms()

    async def step_skillcheck_deadline(self) -> None:
        """
        Resolve as misses the skill checks that can no longer be won, so a
        player cannot dodge one by going quiet until it is forgotten. It
        applies exactly the same expiry rule the move gate and a resume apply
        """
        now_ms = self._now_ms()
        for room in self.rooms.active_rooms():
            pending = room.pending_skillcheck
            if room.result is not None or pending is None:
                continue
            if pending.is_dead(now_ms):
                log.info("skillcheck deadline room=%s color=%s kind=%s",
                         room.room_id, pending.color, pending.kind.value)
                await resolve_skillcheck_fail(self.rooms, self.connections, room)

    async def step_heartbeat_timeout(self) -> None:
        """
        Treat a player whose heartbeat has stopped as disconnected, and tell
        the opponent the board has gone quiet for a reason. This starts the
        grace period; it does not end anybody's game
        """
        for room, color in list(self.rooms.heartbeat_timed_out_rooms()):
            log.info("heartbeat timeout room=%s color=%s", room.room_id, color)
            self.rooms.mark_disconnected(room.room_id, color)
            opp_ws = self.connections.get_for_color(room, room.opp_color(color))
            if opp_ws is not None:
                await send(opp_ws, ConnectionStatusMessage(opp_state="reconnecting"))

    async def step_clock_and_idle_windows(self) -> None:
        """
        Tick every running clock, end the games that have just run out of
        time, and only then look at the idle countdowns. Flag-fall is settled
        first on purpose, so a clock running out in the same tick as an idle
        window wins and the game ends on time rather than on silence
        """
        for room in self.rooms.active_rooms():
            if room.result is not None:
                continue
            backend = room.backend
            if (backend is not None and backend.clock is not None
                    and room.first_move_at is not None):
                backend.tick_clock()
                game_result = backend.game_result()
                if game_result in RESULT_REASON_BY_GAME_RESULT:
                    reason, winner = RESULT_REASON_BY_GAME_RESULT[game_result]
                    await finalize_and_broadcast(self.rooms, self.connections, room,
                                                 reason, winner_color=winner)
            if room.result is not None:
                continue
            await self._step_idle_timeout(room)

    async def _step_idle_timeout(self, room: Room) -> None:
        """
        End a game whose side to move has stayed silent past their window --
        aborted before the first moves, resigned once the game is properly
        under way. An idle resignation is never awarded to a player who is
        themselves disconnected, since there would be nobody there to win it

        :param room: the active room whose countdown is being checked.
        """
        reason = room.idle_timeout_reason(self._now())
        if reason is None:
            return
        winner = None
        if reason == Reason.RESIGNATION:
            winner = room.opp_color(cast(str, room.color_to_move()))
            winner_slot = room.slot(winner)
            if winner_slot is None or winner_slot.disconnected_at is not None:
                return
        await finalize_and_broadcast(self.rooms, self.connections, room, reason,
                                     winner_color=winner)

    async def step_grace_expired(self) -> None:
        """
        End the games where a disconnected player has used up their grace
        period, giving the win to the one who stayed. Every candidate is
        checked again here, because ending an earlier room in the same pass
        awaits -- in that time a player may have reconnected, or their game
        may have ended some other way
        """
        for room, gone_color in list(self.rooms.grace_expired_rooms()):
            if room.result is not None:
                continue
            slot = room.slot(gone_color)
            if slot is None or slot.disconnected_at is None:
                continue
            if self._now() - slot.disconnected_at < GRACE_SECONDS:
                continue
            winner = room.opp_color(gone_color)
            log.info("grace expired room=%s loser=%s desync=%s",
                     room.room_id, gone_color, slot.desync_active)
            await finalize_and_broadcast(self.rooms, self.connections, room,
                                         Reason.ABANDONMENT, winner_color=winner)

    def step_drop_orphans_pre_game(self) -> None:
        """
        Drop a paired room that both players walked away from before a single
        move was played, once the sockets have had a moment to arrive. No
        result is recorded, because no game ever happened
        """
        now = self._now()
        for room in self.rooms.active_rooms():
            if room.result is not None or room.first_move_at is not None:
                continue
            white_present = (room.white is not None
                             and self.connections.get_for_color(room, "white") is not None)
            black_present = (room.black is not None
                             and self.connections.get_for_color(room, "black") is not None)
            if (not white_present and not black_present
                    and room.started_at is not None
                    and now - room.started_at >= PREGAME_CONNECT_GRACE_SECONDS):
                log.info("drop room=%s reason=both_disconnected_pre_game", room.room_id)
                self.rooms.drop_room_now(room.room_id)

    def step_reap_abandoned_queue(self) -> None:
        """
        Clear out the queue slots left behind by players who are no longer
        connected. Without this a dead slot could be paired with a real
        opponent, who would then sit waiting for somebody who left long ago
        """
        for room in self.rooms.stale_queued_rooms():
            slot = room.white or room.black
            if slot is not None and self.connections.get_for_uuid(
                    room.room_id, slot.client_uuid) is not None:
                continue
            log.info("drop room=%s reason=queue_abandoned", room.room_id)
            self.rooms.drop_queued_room(room)

    async def step_reap_timed_out_queue(self) -> None:
        """
        Give up on the players who have waited far too long for an opponent,
        telling them the queue timed out and closing their socket so the app
        can offer them something else to do
        """
        for room in self.rooms.stale_queued_rooms(QUEUE_MAX_WAIT_SECONDS):
            slot = room.white or room.black
            ws = (None if slot is None
                  else self.connections.get_for_uuid(room.room_id, slot.client_uuid))
            if not self.rooms.drop_queued_room(room):
                continue
            log.info("drop room=%s reason=queue_timeout", room.room_id)
            if ws is None:
                continue
            await send(ws, ErrorMessage(reason=Reason.QUEUE_TIMEOUT))
            try:
                await ws.close(code=WS_CLOSE_QUEUE_TIMEOUT)
            except (RuntimeError, WebSocketDisconnect) as exc:
                log.debug("ws close on queue timeout failed: %s", exc)

    async def _notify_rematch(self, room: Room, color: str,
                              event: RematchUpdateEvent) -> None:
        """
        Tell one player what has become of the post-game rematch window, when
        they are still connected to hear it

        :param room: the finished room.
        :param color: which side to tell, white or black.
        :param event: what happened to the rematch window.
        """
        ws = self.connections.get_for_color(room, color)
        if ws is not None:
            await send(ws, RematchUpdateMessage(event=event))

    async def _notify_both(self, room: Room, event: RematchUpdateEvent) -> None:
        """
        Tell both players the same thing about the rematch window, used just
        before the room is dropped from under them

        :param room: the finished room.
        :param event: what happened to the rematch window.
        """
        await self._notify_rematch(room, "white", event)
        await self._notify_rematch(room, "black", event)

    async def step_post_game(self) -> None:
        """
        Look after the window a finished room stays alive for, so the two
        players can agree a rematch, and close it once there is nothing left
        to wait for: both gone, one gone past the grace period, both back at
        the menu, or the window simply run out
        """
        now = self._now()
        for room in self.rooms.active_rooms():
            if room.result is None:
                continue
            white_present = self.connections.get_for_color(room, "white") is not None
            black_present = self.connections.get_for_color(room, "black") is not None
            if not white_present and not black_present:
                log.info("drop room=%s reason=both_disconnected_post_result", room.room_id)
                self.rooms.drop_room_now(room.room_id)
                continue
            present_color = "white" if white_present else "black"
            if room.ended_at is not None and now - room.ended_at >= REMATCH_ABSOLUTE_CAP_SECONDS:
                await self._notify_both(room, "window_expired")
                log.info("drop room=%s reason=rematch_cap", room.room_id)
                self.rooms.drop_room_now(room.room_id)
                continue
            if not (white_present and black_present):
                gone_color = "black" if white_present else "white"
                gone = room.slot(gone_color)
                if (gone is not None and gone.disconnected_at is not None
                        and now - gone.disconnected_at >= POST_GAME_DISCONNECT_GRACE):
                    await self._notify_rematch(room, present_color, "opponent_left")
                    log.info("drop room=%s reason=rematch_grace_expired gone=%s",
                             room.room_id, gone_color)
                    self.rooms.drop_room_now(room.room_id)
                continue
            if (not cast(PlayerSlot, room.white).at_result
                    and not cast(PlayerSlot, room.black).at_result):
                await self._notify_both(room, "window_expired")
                log.info("drop room=%s reason=both_left_result", room.room_id)
                self.rooms.drop_room_now(room.room_id)
                continue
            last = room.last_rematch_activity_at or room.ended_at
            if last is not None and now - last >= REMATCH_IDLE_SECONDS:
                await self._notify_both(room, "window_expired")
                log.info("drop room=%s reason=rematch_idle", room.room_id)
                self.rooms.drop_room_now(room.room_id)
