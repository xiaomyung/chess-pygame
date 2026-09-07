import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from chessshootout import paths
from chessshootout.infra import crash_log
from chessshootout.online.transport import (
    ClientOutdated, FatalResumeError, HEALTHZ_TIMEOUT_SECONDS, ServerTransport,
    ServerWebSocket, TransportError, TransportHTTPError, SchemaVersionMismatch,
    WsConnectionClosed,
)
from chessshootout.server.protocol import (
    CancelMatchmakeRequest, GRACE_SECONDS, HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_MISS_LIMIT, MatchmakeRequest, Reason, ResumeRequest,
    parse_client_version, version_text,
)


log = logging.getLogger("chess.client")


SERVER_FULL_RETRIES = 3
SERVER_FULL_BACKOFF_SECONDS = 1.5
RECONNECT_TOTAL_SECONDS = GRACE_SECONDS
RECONNECT_INTERVAL_SECONDS = 2
RECONNECT_BACKOFF_MAX_SECONDS = 8
PING_SAMPLE_WINDOW = 5

_ProbeResult = TypeVar("_ProbeResult")


class ClientReason:
    """
    Reason codes the client raises on its own behalf, for the failures the
    server never gets to report: it could not be reached, reconnecting ran out
    of time, or the room is gone although the server is up. They travel in the
    same error events as server reasons, so the UI handles one vocabulary
    """

    SERVER_UNREACHABLE = "server_unreachable"
    RECONNECT_FAILED = "reconnect_failed"
    ROOM_LOST = "room_lost"


@dataclass
class Event:
    """
    One thing that happened online, queued for the game to pick up on its next
    frame. The network runs on its own thread, so everything it learns -- a
    server frame, or a failure the client itself decided on -- reaches the game
    as one of these instead of as a direct call
    """

    type: str
    payload: dict[str, Any]


def _probe(addr: str | None,
           request: Callable[[ServerTransport], _ProbeResult]) -> _ProbeResult | None:
    """
    Run one short blocking question against a server address and treat every
    disappointment as no answer. It backs the startup and settings probes, so a
    missing or unusable address costs nothing more than a None

    :param addr: server address to ask, or empty when none is configured.
    :param request: does the asking, given a transport bound to that address.
    :returns: whatever the request answered, or None when it could not be run.
    """
    if not addr:
        return None
    try:
        transport = ServerTransport(addr)
    except TransportError:
        return None
    return request(transport)


def probe_active_game(addr: str | None, client_uuid: str | None,
                      timeout: float = 2.0) -> dict[str, Any] | None:
    """
    Ask whether this player has a game still waiting for them after the app was
    restarted. A hit is what lights up the Reconnect button on the menu, so it
    stays quiet and quick rather than blocking startup on a slow server

    :param addr: server address to ask, or empty when none is configured.
    :param client_uuid: this installation's stable player id.
    :param timeout: how long to wait before treating the probe as a miss.
    :returns: room and fresh session token as plain data, or None when there
        is nothing to rejoin.
    """
    if not client_uuid:
        return None
    response = _probe(addr, lambda transport: transport.reclaim_blocking(
        client_uuid, timeout=timeout))
    if response is None:
        return None
    return response.model_dump(by_alias=True)


def probe_server_health(addr: str | None,
                        timeout: float = HEALTHZ_TIMEOUT_SECONDS) -> dict[str, Any] | None:
    """
    Ask a server how it is doing, which is what the Options screen reports when
    the player tests a server address. A server that answers without naming a
    protocol version is reported with version None, so an unknown server is
    never mistaken for a matching one

    :param addr: server address to test, or empty when none is configured.
    :param timeout: how long to wait before treating the server as down.
    :returns: the health summary as plain data, or None when unreachable.
    """
    response = _probe(addr, lambda transport: transport.healthz_blocking(timeout=timeout))
    if response is None:
        return None
    payload = response.model_dump()
    if "version" not in response.model_fields_set:
        payload["version"] = None
    return payload


def fetch_resume(addr: str, room_id: str, session_token: str) -> dict[str, Any] | None:
    """
    Fetch the whole state of a game this player is already in, ahead of any
    websocket. The reconnect flow calls it twice: once to check the offer is
    real, and again when the player takes it, so the board is rebuilt from
    fresh state rather than from what was cached a moment ago

    :param addr: server address holding the game.
    :param room_id: room the game is being played in.
    :param session_token: this player's seat in that room.
    :returns: the full game state as plain data, or None when it could not be
        fetched.
    """
    transport = ServerTransport(addr)
    response = transport.resume_blocking(room_id, session_token)
    if response is None:
        return None
    return response.model_dump(by_alias=True)


class OnlineClient:
    """
    The game's connection to one online match, from matchmaking through to the
    final result. It owns a background thread with its own event loop, so the
    frontend never blocks on the network: outgoing actions are queued to that
    loop, and everything coming back is queued as events the game drains once
    per frame. Reconnecting after a drop is handled here too
    """

    ROOM_LOST = object()

    def __init__(self) -> None:
        """
        Build a client that is not connected to anything yet. Connecting is a
        separate step, so one client exists per online session and is dropped
        rather than reused once that session ends
        """
        self._inbound: queue.Queue[Event] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._outbound: asyncio.Queue[tuple[str, tuple[Any, ...]]] | None = None
        self._ws: ServerWebSocket | None = None
        self._transport: ServerTransport | None = None
        self._addr: str | None = None
        self._room_id: str | None = None
        self._session_token: str | None = None
        self.state = "disconnected"
        self.opp_state = "connected"
        self._in_queue = False
        self._game_active = False
        self._post_game = False
        self._ping_samples_ms: deque[float] = deque(maxlen=PING_SAMPLE_WINDOW)
        self._heartbeat_interval = HEARTBEAT_INTERVAL_SECONDS
        self._reconnect_total = RECONNECT_TOTAL_SECONDS
        self._last_ping_sent_at = 0.0
        self._last_server_msg_at = 0.0

    @property
    def room_id(self) -> str | None:
        """
        The room this client holds a seat in, which also serves as the id of
        the current online game for saved PGNs and logs

        :returns: the room id, or None before matchmaking has answered.
        """
        return self._room_id

    def get_ping_ms(self) -> int | None:
        """
        The round trip to the server, averaged over the last few heartbeats so
        the number shown on screen does not jump about

        :returns: average round trip in milliseconds, or None before any
            heartbeat has come back.
        """
        if not self._ping_samples_ms:
            return None
        return int(round(sum(self._ping_samples_ms) / len(self._ping_samples_ms)))

    def connect(self, addr: str, request: dict[str, Any]) -> None:
        """
        Start looking for an opponent: bind to a server, then run matchmaking
        and the whole game session on the background thread. An address that
        cannot even be used queues a server-unreachable error instead

        :param addr: server address to play on.
        :param request: matchmaking request fields -- nickname, player id, time
            control, side preference, country, marks preference.
        """
        self._addr = addr
        try:
            self._transport = ServerTransport(addr)
        except TransportError:
            self._inbound.put(Event("error", {"reason": ClientReason.SERVER_UNREACHABLE}))
            self.state = "disconnected"
            return
        self._spawn_loop_thread(self._async_main, request)

    def reconnect_to_existing(self, addr: str, room_id: str, session_token: str,
                              resume_payload: dict[str, Any]) -> None:
        """
        Rejoin a game that is already under way, skipping matchmaking entirely.
        This is the path taken when the player accepts the Reconnect offer
        after restarting the app

        :param addr: server address holding the game.
        :param room_id: room the game is being played in.
        :param session_token: this player's seat in that room.
        :param resume_payload: game state already fetched, read here only for
            the timings the server wants this client to use.
        """
        self._addr = addr
        try:
            self._transport = ServerTransport(addr)
        except TransportError:
            self._inbound.put(Event("error", {"reason": ClientReason.RECONNECT_FAILED}))
            self.state = "disconnected"
            return
        self._room_id = room_id
        self._session_token = session_token
        self._game_active = True
        self._spawn_loop_thread(self._async_main_resume, resume_payload)

    def _spawn_loop_thread(self, coro_factory: Callable[..., Coroutine[Any, Any, None]],
                           *args: Any) -> None:
        """
        Start the background thread that owns this session's event loop. It is
        a daemon, so a crash or a quit can never leave the game waiting on the
        network to finish

        :param coro_factory: builds the coroutine that runs the whole session.
        :param args: arguments handed to that coroutine.
        """
        self._thread = threading.Thread(
            target=self._run_loop, args=(coro_factory, args), daemon=True,
        )
        self._thread.start()

    def cancel_queue(self) -> None:
        """
        Withdraw from matchmaking when the player cancels the search, telling
        the server to release the seat and then closing the session down. Safe
        to call when the session thread has already finished
        """
        if self._loop is None or not self._loop.is_running():
            self.state = "disconnected"
            return
        cancel = self._cancel_async()
        try:
            asyncio.run_coroutine_threadsafe(cancel, self._loop)
        except RuntimeError:
            log.debug("cancel dropped: loop already closed")
            cancel.close()
            self.state = "disconnected"

    async def _cancel_async(self) -> None:
        """
        Do the cancelling on the session loop: tell the server to drop the
        queued seat, then stop the session and close the socket. Failing to
        reach the server is not worth reporting, since the queue entry times
        out on its own
        """
        if self._in_queue and self._room_id and self._session_token:
            try:
                transport = cast(ServerTransport, self._transport)
                async with transport.make_async_http() as http:
                    body = CancelMatchmakeRequest(
                        room_id=self._room_id,
                        session_token=self._session_token,
                    )
                    await transport.cancel_matchmake_async(body, http)
            except Exception:
                pass
        self._in_queue = False
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    def disconnect(self) -> None:
        """
        End this online session for good: stop the loops and close the socket,
        with no attempt to come back. Used when leaving a game and at app exit
        """
        if self._loop is None or not self._loop.is_running():
            self.state = "disconnected"
            return
        self._stop.set()
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)

    def force_reconnect(self) -> None:
        """
        Drop the socket on purpose while keeping the session alive, so the
        reconnect loop rebuilds it. This is the escalation when the server has
        gone silent or a resync never landed
        """
        if self._loop is None or not self._loop.is_running():
            return
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)

    def send_move(self, from_sq: str, to_sq: str, promotion: str | None = None) -> None:
        """
        Offer a move to the server, which alone decides whether it lands

        :param from_sq: origin square in algebraic form, e.g. e2.
        :param to_sq: destination square in algebraic form.
        :param promotion: piece letter q, r, b or n when a pawn promotes,
            otherwise None.
        """
        self._enqueue("send_move", from_sq, to_sq, promotion)

    def send_resign(self) -> None:
        """
        Give up the game, which ends it for both players
        """
        self._enqueue("send_resign")

    def send_draw_offer(self) -> None:
        """
        Offer the opponent a draw
        """
        self._enqueue("send_draw_offer")

    def send_draw_response(self, accept: bool) -> None:
        """
        Answer the opponent's draw offer

        :param accept: True ends the game as a draw, False clears the offer.
        """
        self._enqueue("send_draw_response", accept)

    def send_rematch_request(self) -> None:
        """
        Offer another game in the same room once this one has finished
        """
        self._enqueue("send_rematch_request")

    def send_rematch_response(self, accept: bool) -> None:
        """
        Answer the opponent's rematch offer

        :param accept: True restarts the room with colours swapped, False
            closes the rematch out.
        """
        self._enqueue("send_rematch_response", accept)

    def send_left_result(self) -> None:
        """
        Say that this player has walked away from the finished game, which
        withdraws any rematch they had offered
        """
        self._enqueue("send_left_result")

    def send_takeback_request(self) -> None:
        """
        Ask the opponent to let the last move be retracted
        """
        self._enqueue("send_takeback_request")

    def send_takeback_response(self, accept: bool) -> None:
        """
        Answer the opponent's takeback request

        :param accept: True retracts the move for both sides, False leaves the
            position alone.
        """
        self._enqueue("send_takeback_response", accept)

    def send_give_time(self, hold_ms: int) -> None:
        """
        Hand the opponent some of the clock, sized by how long the give-time
        button was held

        :param hold_ms: how long the button was held, in milliseconds.
        """
        self._enqueue("send_give_time", hold_ms)

    def send_ping(self, ply: int | None) -> None:
        """
        Send the heartbeat, stamping the moment so the reply can be turned into
        a round-trip reading. It also reports the ply this client is on, which
        is how a board that has fallen behind gets noticed

        :param ply: number of half-moves this client has applied, or None while
            it is not sitting on a live online board and has no ply to claim.
        """
        self._last_ping_sent_at = time.monotonic()
        self._enqueue("send_ping", ply)

    def send_skill_check_shot(self, client_elapsed_ms: float, direction: str | None = None,
                              target_row: float | None = None,
                              target_col: float | None = None) -> None:
        """
        Report one input the player made during a live skill check. Only the
        input travels -- a direction, or a point in board space -- since the
        server judges every shot against geometry it worked out itself

        :param client_elapsed_ms: milliseconds into the check when the input
            was made, as this client measured it.
        :param direction: prompt answered during a combo check, else None.
        :param target_row: aim point row in board space, else None.
        :param target_col: aim point column in board space, else None.
        """
        self._enqueue("send_skill_check_shot", client_elapsed_ms, direction,
                      target_row, target_col)

    def send_annotations_state(self, sharing: bool, highlights: list[str],
                               arrows: list[tuple[str, str]]) -> None:
        """
        Publish this player's whole set of shared board marks at once, sent
        when sharing is switched on or off and whenever the set has to be
        replaced rather than nudged

        :param sharing: whether this player is sharing marks at all.
        :param highlights: highlighted squares in algebraic form.
        :param arrows: arrows as (origin, destination) square pairs.
        """
        self._enqueue("send_annotations_state", sharing, highlights, arrows)

    def send_annotation_delta(self, action: str, kind: str, square: str | None = None,
                              from_sq: str | None = None, to_sq: str | None = None) -> None:
        """
        Publish a single change to this player's shared marks, which is what
        every individual stroke on a shared board sends

        :param action: add or remove.
        :param kind: highlight or arrow.
        :param square: the square involved, for a highlight change.
        :param from_sq: arrow origin square, for an arrow change.
        :param to_sq: arrow destination square, for an arrow change.
        """
        self._enqueue("send_annotation_delta", action, kind, square, from_sq, to_sq)

    def send_quick_chat(self, preset: int) -> None:
        """
        Send one phrase from the fixed quick-chat list

        :param preset: index of the phrase in the quick-chat list.
        """
        self._enqueue("send_quick_chat", preset)

    def send_set_marks_visibility(self, hide_opp: bool) -> None:
        """
        Say whether this player wants the opponent's shared marks delivered at
        all, keeping the server's view in step with the local setting

        :param hide_opp: True to stop receiving the opponent's marks.
        """
        self._enqueue("send_set_marks_visibility", hide_opp)

    def heartbeat_interval(self) -> float:
        """
        How often this client should be sending heartbeats. The server states
        the interval when a game starts or resumes, so both sides agree on what
        counts as silence

        :returns: seconds between heartbeats.
        """
        return self._heartbeat_interval

    def seconds_since_server_msg(self) -> float:
        """
        How long the server has been quiet, counting anything at all that
        arrived. It is the raw measure behind the silence check

        :returns: seconds since the last frame, or 0 before the first one.
        """
        if self._last_server_msg_at == 0.0:
            return 0.0
        return time.monotonic() - self._last_server_msg_at

    def is_connected(self) -> bool:
        """
        Whether the game socket is up right now, which gates sending anything
        that is only meaningful on a live connection

        :returns: True while a session socket is open and connected.
        """
        return self.state == "connected" and self._ws is not None

    def is_server_silent(self) -> bool:
        """
        Whether the server has missed enough heartbeats to be treated as gone.
        The caller escalates to dropping the socket, which the reconnect loop
        then rebuilds rather than waiting on a connection that is already dead

        :returns: True once the silence has run past the miss limit.
        """
        return self.seconds_since_server_msg() > self._heartbeat_interval * HEARTBEAT_MISS_LIMIT

    def request_state_sync(self) -> None:
        """
        Ask for the whole game state again, the repair for a board that has
        drifted out of step. The answer arrives as a game-resumed event like
        any other resume, and nothing happens if there is no live session
        """
        if (self._loop is None or self._loop.is_closed()
                or self._transport is None
                or self._room_id is None or self._session_token is None):
            log.debug("state-sync dropped: loop not running")
            return
        try:
            asyncio.run_coroutine_threadsafe(self._fetch_state_sync(), self._loop)
        except RuntimeError:
            pass

    async def _fetch_state_sync(self) -> None:
        """
        Fetch the full state on the session loop and queue it for the game. A
        refusal that means the game is over or the seat is dead is checked
        against the server's health, so a restarted server reads as a lost room
        rather than as a connection problem
        """
        transport = cast(ServerTransport, self._transport)
        body = ResumeRequest(
            room_id=cast(str, self._room_id),
            session_token=cast(str, self._session_token),
        )
        try:
            async with transport.make_async_http() as http:
                response = await transport.resume_async(body, http)
        except FatalResumeError as exc:
            log.warning("state-sync fatal: %s", exc)
            try:
                async with transport.make_async_http() as http:
                    health = await transport.healthz_async(http)
            except Exception:
                health = None
            reason = (ClientReason.ROOM_LOST if health is not None
                      else ClientReason.RECONNECT_FAILED)
            self._inbound.put(Event("error", {"reason": reason}))
            return
        except Exception as exc:
            log.warning("state-sync failed: %s", exc)
            self._inbound.put(Event("error", {"reason": ClientReason.RECONNECT_FAILED}))
            return
        if response is None:
            self._inbound.put(Event("error", {"reason": ClientReason.RECONNECT_FAILED}))
            return
        log.info("state-sync ok room=%s ply=%d",
                 self._room_id, len(response.move_history))
        resumed = response.model_dump(by_alias=True)
        self._adopt_timing(resumed)
        self._inbound.put(Event("game_resumed", resumed))

    def _enqueue(self, method: str, *args: Any) -> None:
        """
        Hand one outgoing action to the session loop by name, which is how
        every send crosses from the game thread to the network thread. With no
        session running the action is dropped, since there is nowhere to send
        it and the game must not block

        :param method: name of the send method on the game socket.
        :param args: arguments for that send method.
        """
        if self._loop is None or self._loop.is_closed() or self._outbound is None:
            log.debug("send dropped (loop not running): method=%s", method)
            return
        try:
            self._loop.call_soon_threadsafe(
                self._outbound.put_nowait, (method, args),
            )
            log.debug("ws send method=%s", method)
        except RuntimeError:
            pass

    def drain_inbound(self) -> list[Event]:
        """
        Take everything the network has learned since the last frame. The game
        calls this once per frame and is the only reader, so the events arrive
        in order and each one is delivered exactly once

        :returns: the queued events, oldest first, empty when nothing arrived.
        """
        events = []
        while True:
            try:
                events.append(self._inbound.get_nowait())
            except queue.Empty:
                break
        return events

    def _run_loop(self, coro_factory: Callable[..., Coroutine[Any, Any, None]],
                  args: tuple[Any, ...]) -> None:
        """
        The body of the background thread: make an event loop, run the session
        on it, and close it afterwards. A crash here becomes an error event and
        a crash log rather than a silently dead thread

        :param coro_factory: builds the coroutine that runs the whole session.
        :param args: arguments handed to that coroutine.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._outbound = asyncio.Queue()
        try:
            self._loop.run_until_complete(coro_factory(*args))
        except Exception as exc:
            self._inbound.put(Event("error", {"reason": str(exc)}))
            self._dump_crash_log(exc)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def _dump_crash_log(self, exc: Exception) -> None:
        """
        Write a crash log for a session that died, with the online state that
        explains it beside the traceback. Failing to write the log is itself
        logged and otherwise ignored, since the session is already lost

        :param exc: the exception that ended the session.
        """
        try:
            crash_log.write_crash_log(exc, crash_log.get_memory_buffer(), {
                "online_state": self.state,
                "addr": self._addr,
                "room_id": self._room_id,
                "in_queue": self._in_queue,
                "game_active": self._game_active,
            }, root=paths.get_log_dir())
        except Exception:
            log.exception("crash log write failed")

    async def _async_main_resume(self, resume_payload: dict[str, Any]) -> None:
        """
        Run a session that starts by rejoining an existing game: adopt the
        timings the server stated, then go straight into the socket with the
        reconnect loop around it

        :param resume_payload: game state already fetched for this room.
        """
        log.info("reconnect-resume addr=%s room=%s", self._addr, self._room_id)
        if isinstance(resume_payload, dict):
            self._adopt_timing(resume_payload)
        self.state = "connecting"
        await self._run_session_with_reconnects()

    async def _async_main(self, request: dict[str, Any]) -> None:
        """
        Run a session from scratch: queue for an opponent, publish the seat the
        server handed back, then hold the socket for the whole game. A build the
        server will not take -- wrong protocol, or too old -- is reported as its
        own reason, the outdated one carrying the version to update to, so the
        player can be told to update instead of being asked to retry

        :param request: matchmaking request fields for this search.
        """
        log.info("connect addr=%s", self._addr)
        try:
            self.state = "connecting"
            mm = await self._matchmake_with_retries(request)
        except ClientOutdated as exc:
            parsed_minimum = parse_client_version(exc.min_version)
            log.warning("client outdated min_version=%s",
                        "unparseable" if parsed_minimum is None
                        else version_text(parsed_minimum))
            self._inbound.put(Event("error", {"reason": Reason.CLIENT_OUTDATED,
                                              "min_version": exc.min_version}))
            self.state = "disconnected"
            return
        except SchemaVersionMismatch as exc:
            log.warning("schema version mismatch reason=%s", exc)
            self._inbound.put(Event("error", {"reason": Reason.VERSION_MISMATCH}))
            self.state = "disconnected"
            return
        except Exception as exc:
            log.warning("matchmake failed reason=%s", exc)
            self._inbound.put(Event("error", {"reason": str(exc)}))
            self.state = "disconnected"
            return
        self._room_id = mm["room_id"]
        self._session_token = mm["session_token"]
        self._in_queue = True
        log.info("matchmake ok room=%s", self._room_id)
        self._inbound.put(Event("matchmake_response", mm))
        await self._run_session_with_reconnects()

    async def _run_session_with_reconnects(self) -> None:
        """
        Keep the game connected for as long as it is worth keeping: hold the
        socket, and when it drops mid-game, resume and hold it again. It gives
        up only when the game is over, the player left, the room turns out to
        be gone, or the reconnect window runs out -- and each of those is
        reported before the session ends
        """
        try:
            await self._run_ws_session()
            while (not self._stop.is_set() and (self._game_active or self._post_game)
                   and self._session_token is not None):
                log.info("ws dropped mid-game; attempting reconnect")
                self.state = "reconnecting"
                resumed = await self._resume_with_retries()
                if resumed is self.ROOM_LOST:
                    log.warning("reconnect: server alive but room gone")
                    self._inbound.put(Event("error", {"reason": ClientReason.ROOM_LOST}))
                    break
                if not resumed:
                    log.warning("reconnect gave up")
                    self._inbound.put(Event("error", {"reason": ClientReason.RECONNECT_FAILED}))
                    break
                log.info("reconnect succeeded; resuming game")
                self._adopt_timing(resumed)
                self._inbound.put(Event("game_resumed", resumed))
                try:
                    await self._run_ws_session()
                except Exception as exc:
                    self._inbound.put(Event("error", {"reason": str(exc)}))
                    break
        except Exception as exc:
            log.warning("ws session crash: %s", exc)
            self._inbound.put(Event("error", {"reason": str(exc)}))
        finally:
            log.info("session ended state=disconnected")
            self.state = "disconnected"

    async def _resume_with_retries(self) -> Any:
        """
        Keep asking for the game state back until it comes, the window closes,
        or the answer says it never will. Waiting doubles between tries so a
        server coming back up is not hammered, and the whole window is the
        grace period the server allows for a disconnect

        :returns: the game state as plain data, the ROOM_LOST marker when the
            server is up but the game is gone, or None when it gave up.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._reconnect_total
        backoff = RECONNECT_INTERVAL_SECONDS
        transport = cast(ServerTransport, self._transport)
        body = ResumeRequest(
            room_id=cast(str, self._room_id),
            session_token=cast(str, self._session_token),
        )
        async with transport.make_async_http() as http:
            while loop.time() < deadline and not self._stop.is_set():
                try:
                    response = await transport.resume_async(body, http)
                except FatalResumeError:
                    health = await transport.healthz_async(http)
                    if health is not None:
                        return self.ROOM_LOST
                    return None
                if response is not None:
                    return response.model_dump(by_alias=True)
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(backoff, remaining))
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_SECONDS)
        return None

    async def _matchmake_with_retries(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Ask for a seat, retrying while the server is full or unreachable. A
        busy server is worth a few tries, since seats free up constantly, but a
        request the server rejects on its merits -- a protocol mismatch
        included -- is raised at once

        :param request: matchmaking request fields for this search.
        :returns: the room and session token, as plain data.
        """
        last_exc: Exception | None = None
        req = MatchmakeRequest(**request)
        transport = cast(ServerTransport, self._transport)
        async with transport.make_async_http() as http:
            for attempt in range(SERVER_FULL_RETRIES + 1):
                try:
                    return (await transport.matchmake_async(req, http)
                            ).model_dump(by_alias=True)
                except SchemaVersionMismatch:
                    raise
                except TransportHTTPError as exc:
                    if exc.status_code == 503:
                        last_exc = RuntimeError("room_full")
                        await asyncio.sleep(SERVER_FULL_BACKOFF_SECONDS)
                        continue
                    raise RuntimeError(exc.reason or f"http_{exc.status_code}") from exc
                except TransportError as exc:
                    last_exc = exc
                    if attempt < SERVER_FULL_RETRIES:
                        await asyncio.sleep(SERVER_FULL_BACKOFF_SECONDS)
                        continue
                    raise RuntimeError("server_unreachable") from exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("room_full")

    async def _run_ws_session(self) -> None:
        """
        Hold one game socket open, reading and writing at the same time, and
        return once either side of it finishes. Round-trip samples start empty
        for each socket, since readings from a dead connection say nothing
        about the new one
        """
        transport = cast(ServerTransport, self._transport)
        ws = await transport.ws_connect(cast(str, self._room_id),
                                        cast(str, self._session_token))
        self._ws = ws
        self._last_server_msg_at = time.monotonic()
        self.opp_state = "connected"
        self.state = "connected"
        self._ping_samples_ms.clear()
        recv_task = asyncio.create_task(self._recv_loop(ws))
        send_task = asyncio.create_task(self._send_loop(ws))
        try:
            await asyncio.wait({recv_task, send_task},
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            recv_task.cancel()
            send_task.cancel()
            await ws.close()
            self._ws = None

    def _adopt_timing(self, msg: dict[str, Any]) -> None:
        """
        Take the heartbeat interval and reconnect window from the server, which
        states both whenever a game starts or resumes. Following the server
        means an operator can retune them without a new client build

        :param msg: a game-start or resume payload, possibly carrying neither
            value, in which case the current ones stand.
        """
        interval = msg.get("heartbeat_interval_seconds")
        if interval:
            self._heartbeat_interval = float(interval)
        grace = msg.get("grace_seconds")
        if grace:
            self._reconnect_total = float(grace)

    async def _recv_loop(self, ws: ServerWebSocket) -> None:
        """
        Read server frames for as long as the socket lives and queue them for
        the game, while keeping the few facts the client itself needs: the
        round-trip reading, whether a game is running, and how the opponent's
        connection is doing. A closed socket simply ends the loop, which is
        what starts a reconnect

        :param ws: the open game socket to read from.
        """
        try:
            while not self._stop.is_set():
                msg = await ws.recv()
                if msg is None:
                    continue
                self._last_server_msg_at = time.monotonic()
                msg_type = msg.get("type", "")
                log.debug("ws recv type=%s", msg_type)
                if msg_type == "pong":
                    if self._last_ping_sent_at:
                        rtt_ms = (time.monotonic() - self._last_ping_sent_at) * 1000.0
                        self._ping_samples_ms.append(rtt_ms)
                    continue
                if msg_type in ("game_start", "game_resumed"):
                    self._adopt_timing(msg)
                if msg_type == "game_start":
                    self._in_queue = False
                    self._game_active = True
                    self._post_game = False
                if msg_type == "result":
                    self._game_active = False
                    self._post_game = True
                    log.info("ws result reason=%s winner=%s",
                             msg.get("reason"), msg.get("winner_color"))
                if msg_type == "connection_status":
                    self.opp_state = msg.get("opp_state", "connected")
                self._inbound.put(Event(msg_type, msg))
        except (WsConnectionClosed, asyncio.CancelledError):
            pass

    async def _send_loop(self, ws: ServerWebSocket) -> None:
        """
        Take queued player actions and put them on the socket in order. One
        action failing to send is logged and skipped rather than allowed to
        take the connection down, but a closed socket ends the loop and hands
        over to the reconnect

        :param ws: the open game socket to write to.
        """
        try:
            while not self._stop.is_set():
                method, args = await cast(
                    "asyncio.Queue[tuple[str, tuple[Any, ...]]]", self._outbound).get()
                send = getattr(ws, method, None)
                if send is None:
                    log.warning("unknown ws send method=%s", method)
                    continue
                try:
                    await send(*args)
                except (WsConnectionClosed, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    log.warning("ws send failed method=%s: %s", method, exc)
        except (WsConnectionClosed, asyncio.CancelledError):
            pass
