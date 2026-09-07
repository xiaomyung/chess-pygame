from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import WebSocketDisconnect

from chessshootout.server import logging_setup
from chessshootout.server.broadcasts import finalize_and_broadcast, resolve_skillcheck_fail
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.protocol import (
    ConnectionStatusMessage, ErrorMessage, GRACE_SECONDS,
    QUEUE_MAX_WAIT_SECONDS, RESULT_REASON_BY_GAME_RESULT, Reason, RematchUpdateEvent,
    RematchUpdateMessage, WS_CLOSE_QUEUE_TIMEOUT,
)
from chessshootout.server.rooms import (
    REMATCH_ABSOLUTE_CAP_SECONDS, REMATCH_IDLE_SECONDS, POST_GAME_DISCONNECT_GRACE,
    PlayerSlot, Room, RoomManager,
)


log = logging_setup.get_logger("chess.server.app")


PREGAME_CONNECT_GRACE_SECONDS = 5.0
SWEEP_STALE_SECONDS = 30.0
SWEEP_ERROR_LOG_INTERVAL_SECONDS = 60.0

STEP_SKILLCHECK_DEADLINE = "skillcheck_deadline"
STEP_CLOCK_AND_IDLE_WINDOWS = "clock_and_idle_windows"
STEP_HEARTBEAT_TIMEOUT = "heartbeat_timeout"
STEP_GRACE_EXPIRED = "grace_expired"
STEP_DROP_ORPHANS_PRE_GAME = "drop_orphans_pre_game"
STEP_REAP_ABANDONED_QUEUE = "reap_abandoned_queue"
STEP_REAP_TIMED_OUT_QUEUE = "reap_timed_out_queue"
STEP_POST_GAME = "post_game"
STEP_GC_FINISHED_ROOMS = "gc_finished_rooms"


class Sweep:
    """
    The server's housekeeping pass over every room, run on a short timer:
    clocks, skill-check deadlines, idle countdowns, disconnects, abandoned
    queue slots and finished rooms. It is where everything that has to happen
    without a player doing anything happens. It also tracks whether its passes
    are completing, which the health check reports as a degraded server
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
        self._running = False
        self._pass_clean = True
        self._last_ok_at = 0.0
        self._last_error_log_at: float | None = None
        self._suppressed = 0
        self._last_failure: BaseException | None = None
        self.failure_count = 0

    def mark_running(self) -> None:
        """
        Note that the timer driving the pass has started. Freshness only begins
        to mean anything from here: before the first tick there is nothing for
        the pass to be late for
        """
        self._running = True
        self._last_ok_at = self._now()

    def note_unhandled(self, exc: BaseException) -> None:
        """
        Record a failure that escaped a whole pass rather than one step of it,
        so a server that cannot even start its housekeeping reports itself as
        degraded exactly as one whose steps keep failing does

        :param exc: the failure raised out of the pass.
        """
        self._note_failure("step_all", exc)

    @property
    def age_s(self) -> float:
        """
        How long ago the last pass that completed without a single failure
        finished. It reads as zero until the timer has started, so a server
        that has only just come up never reports itself as behind

        :returns: seconds since the last clean pass.
        """
        if not self._running:
            return 0.0
        return max(0.0, self._now() - self._last_ok_at)

    @property
    def is_stale(self) -> bool:
        """
        Whether the pass has gone so long without completing cleanly that the
        server should be called degraded. The limit is loose on purpose: one
        slow socket send inside a walk must not flip a server's health

        :returns: True when the last clean pass is older than the limit.
        """
        return self._running and self.age_s > SWEEP_STALE_SECONDS

    def _note_failure(self, step: str, exc: BaseException) -> None:
        """
        The single place a failed step is recorded. It spoils the current pass,
        which is what stops the freshness stamp, and reports the failure at most
        once a minute, so a fault repeating every tick cannot bury the log --
        the failures skipped meanwhile are counted into the next line written

        :param step: name of the step that failed.
        :param exc: the failure that step raised.
        """
        self._pass_clean = False
        self.failure_count += 1
        self._last_failure = exc
        now = self._now()
        if (self._last_error_log_at is not None
                and now - self._last_error_log_at < SWEEP_ERROR_LOG_INTERVAL_SECONDS):
            self._suppressed += 1
            return
        log.error("sweep step failed step=%s suppressed=%d", step, self._suppressed,
                  exc_info=exc)
        self._suppressed = 0
        self._last_error_log_at = now

    async def _guard(self, step: str, run: Callable[[], Awaitable[None] | None]) -> None:
        """
        Run one step of the pass behind its own safety net, so a step that fails
        costs only its own work: every later step in the same pass still runs,
        and the loop driving the pass stays alive. Steps that walk the rooms
        catch their own failures per room; what this net covers is the setup
        around such a walk, and the whole of a step that has no loop at all.
        Both a plain step and an awaitable one are accepted, so no step needs a
        wrapper to be guarded

        :param step: name of the step, reported when it fails.
        :param run: the step to run.
        """
        try:
            outcome = run()
            if outcome is not None:
                await outcome
        except Exception as exc:
            self._note_failure(step, exc)

    async def step_all(self) -> None:
        """
        Run one whole housekeeping pass in the order the game's rules need it:
        skill-check deadlines, then clocks and idle windows, then the
        connection timeouts, then the queue and the finished rooms. Every step
        is run behind its own safety net, and the pass only stamps itself fresh
        when all of them got through without a failure
        """
        self._pass_clean = True
        await self._guard(STEP_SKILLCHECK_DEADLINE, self.step_skillcheck_deadline)
        await self._guard(STEP_CLOCK_AND_IDLE_WINDOWS, self.step_clock_and_idle_windows)
        await self._guard(STEP_HEARTBEAT_TIMEOUT, self.step_heartbeat_timeout)
        await self._guard(STEP_GRACE_EXPIRED, self.step_grace_expired)
        await self._guard(STEP_DROP_ORPHANS_PRE_GAME, self.step_drop_orphans_pre_game)
        await self._guard(STEP_REAP_ABANDONED_QUEUE, self.step_reap_abandoned_queue)
        await self._guard(STEP_REAP_TIMED_OUT_QUEUE, self.step_reap_timed_out_queue)
        await self._guard(STEP_POST_GAME, self.step_post_game)
        await self._guard(STEP_GC_FINISHED_ROOMS, self.rooms.gc_finished_rooms)
        if self._pass_clean:
            self._last_ok_at = self._now()

    async def step_skillcheck_deadline(self) -> None:
        """
        Resolve as misses the skill checks that can no longer be won, so a
        player cannot dodge one by going quiet until it is forgotten. It
        applies exactly the same expiry rule the move gate and a resume apply
        """
        now_ms = self._now_ms()
        for room in self.rooms.active_rooms():
            try:
                pending = room.pending_skillcheck
                if room.result is not None or pending is None:
                    continue
                if pending.is_dead(now_ms):
                    log.info("skillcheck deadline room=%s color=%s kind=%s",
                             room.room_id, pending.color, pending.kind.value)
                    await resolve_skillcheck_fail(self.rooms, self.connections, room)
            except Exception as exc:
                self._note_failure(STEP_SKILLCHECK_DEADLINE, exc)

    async def step_heartbeat_timeout(self) -> None:
        """
        Treat a player whose heartbeat has stopped as disconnected, and tell
        the opponent the board has gone quiet for a reason. This starts the
        grace period; it does not end anybody's game
        """
        for room, color in list(self.rooms.heartbeat_timed_out_rooms()):
            try:
                log.info("heartbeat timeout room=%s color=%s", room.room_id, color)
                self.rooms.mark_disconnected(room.room_id, color)
                opp_ws = self.connections.get_for_color(room, room.opp_color(color))
                if opp_ws is not None:
                    await send(opp_ws, ConnectionStatusMessage(opp_state="reconnecting"))
            except Exception as exc:
                self._note_failure(STEP_HEARTBEAT_TIMEOUT, exc)

    async def step_clock_and_idle_windows(self) -> None:
        """
        Tick every running clock, end the games that have just run out of
        time, and only then look at the idle countdowns. Flag-fall is settled
        first on purpose, so a clock running out in the same tick as an idle
        window wins and the game ends on time rather than on silence
        """
        for room in self.rooms.active_rooms():
            try:
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
            except Exception as exc:
                self._note_failure(STEP_CLOCK_AND_IDLE_WINDOWS, exc)

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
            try:
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
            except Exception as exc:
                self._note_failure(STEP_GRACE_EXPIRED, exc)

    def step_drop_orphans_pre_game(self) -> None:
        """
        Drop a paired room that both players walked away from before a single
        move was played, once the sockets have had a moment to arrive. No
        result is recorded, because no game ever happened
        """
        now = self._now()
        for room in self.rooms.active_rooms():
            try:
                if room.result is not None or room.first_move_at is not None:
                    continue
                white_present = (
                    room.white is not None
                    and self.connections.get_for_color(room, "white") is not None)
                black_present = (
                    room.black is not None
                    and self.connections.get_for_color(room, "black") is not None)
                if (not white_present and not black_present
                        and room.started_at is not None
                        and now - room.started_at >= PREGAME_CONNECT_GRACE_SECONDS):
                    log.info("drop room=%s reason=both_disconnected_pre_game", room.room_id)
                    self.rooms.drop_room_now(room.room_id)
            except Exception as exc:
                self._note_failure(STEP_DROP_ORPHANS_PRE_GAME, exc)

    def step_reap_abandoned_queue(self) -> None:
        """
        Clear out the queue slots left behind by players who are no longer
        connected. Without this a dead slot could be paired with a real
        opponent, who would then sit waiting for somebody who left long ago
        """
        for room in self.rooms.stale_queued_rooms():
            try:
                slot = room.white or room.black
                if slot is not None and self.connections.get_for_uuid(
                        room.room_id, slot.client_uuid) is not None:
                    continue
                log.info("drop room=%s reason=queue_abandoned", room.room_id)
                self.rooms.drop_queued_room(room)
            except Exception as exc:
                self._note_failure(STEP_REAP_ABANDONED_QUEUE, exc)

    async def step_reap_timed_out_queue(self) -> None:
        """
        Give up on the players who have waited far too long for an opponent,
        telling them the queue timed out and closing their socket so the app
        can offer them something else to do
        """
        for room in self.rooms.stale_queued_rooms(QUEUE_MAX_WAIT_SECONDS):
            try:
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
            except Exception as exc:
                self._note_failure(STEP_REAP_TIMED_OUT_QUEUE, exc)

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
            try:
                if room.result is None:
                    continue
                white_present = self.connections.get_for_color(room, "white") is not None
                black_present = self.connections.get_for_color(room, "black") is not None
                if not white_present and not black_present:
                    log.info("drop room=%s reason=both_disconnected_post_result",
                             room.room_id)
                    self.rooms.drop_room_now(room.room_id)
                    continue
                present_color = "white" if white_present else "black"
                if (room.ended_at is not None
                        and now - room.ended_at >= REMATCH_ABSOLUTE_CAP_SECONDS):
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
            except Exception as exc:
                self._note_failure(STEP_POST_GAME, exc)
