import asyncio
import ipaddress
import os
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chessshootout.backend.backend import Backend
from chessshootout.backend.fen import export_fen
from chessshootout.backend.utils import (
    Move, PROMO_LETTER_BY_TYPE, coord_from_square,
)
from chessshootout.server import logging_setup
from chessshootout.server.broadcasts import (
    broadcast_game_start, finalize_and_broadcast, idle_window_wire,
    resolve_skillcheck_fail)
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.handlers import arrow_wires, clock_snapshot, dispatch
from chessshootout.server.moderation import library
from chessshootout.server.protocol import (
    ANNOTATIONS_PER_SECOND, AnnotationSetWire,
    AuthMessage, CHAT_COOLDOWN_SECONDS, CancelMatchmakeRequest,
    ConnectionStatusMessage, ErrorMessage,
    HealthResponse, HistoryEntryWire, LockWire,
    MatchmakeRequest, MatchmakeResponse,
    PROTOCOL_VERSION, PendingSkillCheckWire, Reason, ReasonEnvelope, ReclaimRequest,
    ReclaimResponse, RematchRequestMessage, RematchUpdateMessage,
    ResultMessage, ResumeRequest, ResumeResponse, SkillCheckOutcomeWire, is_uuid4,
)
from chessshootout.server.rooms import (
    AlreadyInGameError, GameAlreadyStartedError, InvalidTokenError, NotInRoomError,
    PAIRING_WAIT_SECONDS, PlayerSlot, Room, RoomManager, ServerFullError,
    SharedAnnotations,
)
from chessshootout.server.sweep import Sweep


CLOCK_TICK_INTERVAL_SECONDS = 0.1
MAX_INBOUND_MESSAGE_BYTES = 4096
DEFAULT_MAX_ROOMS = 100


def _moderation_enabled() -> bool:
    """
    Say whether this server screens the board marks players share with each
    other. Screening is on by default and only an operator can switch it off,
    through the MODERATION_OFF environment variable, which is what a local test
    rig does to keep its runs cheap

    :returns: True when inbound marks are screened before being relayed
    """
    return os.environ.get("MODERATION_OFF", "").strip().lower() not in (
        "1", "true", "yes", "on")


def app_version() -> str:
    """
    Give the released build of the server, which health checks report so an
    operator can see which image a box is actually running. A source checkout
    has no installed distribution and simply has no version to give

    :returns: the installed package version, or empty when there is none
    """
    try:
        return _pkg_version("chess-shootout")
    except PackageNotFoundError:
        return ""


RECLAIM_PER_UUID_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PRUNE_THRESHOLD = 4096
RECLAIM_WINDOW_SECONDS = 60.0

MATCHMAKE_PER_IP_LIMIT = "60/minute"
RESUME_PER_IP_LIMIT = "60/minute"
RECLAIM_PER_IP_LIMIT = "120/minute"

WS_MESSAGES_PER_SECOND = 30
WS_RATE_WINDOW_SECONDS = 1.0

WS_CLOSE_PAYLOAD_TOO_LARGE = 1009
WS_CLOSE_INVALID_TOKEN = 4000
WS_CLOSE_SERVER_SHUTDOWN = 4002
WS_CLOSE_SUPERSEDED = 4003


DEFAULT_TRUSTED_PROXIES = "127.0.0.1/32"


def _parse_trusted_proxies(raw: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """
    Work out which machines are allowed to speak for somebody else, by turning
    the TRUSTED_PROXIES setting into a list of networks. Only a peer inside one
    of them may hand the server a forwarded client address; a bad entry is
    skipped instead of stopping startup, and an entirely unusable setting leaves
    the list empty so nobody is trusted

    :param raw: comma-separated networks or bare addresses, blanks tolerated
    :returns: the entries that parsed, in the order they were written
    """
    networks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return networks


TRUSTED_PROXIES_RAW = os.environ.get("TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES)
TRUSTED_PROXIES = _parse_trusted_proxies(TRUSTED_PROXIES_RAW)


def log_trusted_proxies(
    raw: str | None = None,
    trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
) -> None:
    """
    Record which proxies this server trusts, once at startup, so an operator can
    see the effective set in the logs instead of guessing at it. A configured
    value that parsed to nothing is a warning rather than an info line, because
    it silently moves every per-IP limit onto the socket peer

    :param raw: the setting as configured; None reads the module's own value
    :param trusted: the parsed networks; None reads the module's own list
    """
    raw = TRUSTED_PROXIES_RAW if raw is None else raw
    trusted = TRUSTED_PROXIES if trusted is None else trusted
    if raw.strip() and not trusted:
        log.warning("trusted proxies unparsable (TRUSTED_PROXIES=%r); "
                    "rate limits key on the socket peer", raw)
        return
    log.info("trusted proxies %s", ",".join(str(net) for net in trusted) or "none")


def _peer_trusted(
    peer: str, trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """
    Say whether the machine on the other end of the socket is one of the
    configured proxies. A peer that is not an address at all -- the in-process
    test client, for one -- is never trusted

    :param peer: socket peer address as text
    :param trusted: networks that may speak for another client
    :returns: True when the peer falls inside one of those networks
    """
    try:
        ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(ip in net for net in trusted)


def _forwarded_ip(raw: str | None) -> str | None:
    """
    Read the client address a proxy claims to be forwarding, accepting it only
    when it really is an IP address. A header that is missing, empty or
    malformed yields nothing, so a junk value can never become a limiter key

    :param raw: header value as received, surrounding whitespace tolerated
    :returns: the normalised address, or None when there is no usable one
    """
    if not raw:
        return None
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        return None


def client_ip_key(
    request: Request,
    trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
) -> str:
    """
    Decide which address a request counts against for the per-IP limits. The
    server sits behind Cloudflare, so the real player address arrives in the
    cf-connecting-ip header -- but that header is believed only when the socket
    peer is itself a trusted proxy and the value parses as an address, otherwise
    anyone could forge a key and slip past their own limits

    :param request: inbound HTTP request, read for its peer and headers
    :param trusted: networks allowed to forward an address; None uses the
        module's configured set
    :returns: the address the limiter counts this call against
    """
    trusted = TRUSTED_PROXIES if trusted is None else trusted
    peer = get_remote_address(request)
    if _peer_trusted(peer, trusted):
        forwarded = _forwarded_ip(request.headers.get("cf-connecting-ip"))
        if forwarded is not None:
            return forwarded
    return peer


log = logging_setup.get_logger("chess.server.app")


class UuidRateLimiter:
    """
    A sliding-window call counter keyed by whoever is calling rather than by
    where they call from, for the places an address is the wrong identity:
    session reclaims, annotation and quick-chat floods, and the message cap on a
    single websocket. Time comes from an injected clock, so tests drive it
    instead of waiting
    """

    def __init__(self, limit_per_minute: int, window_seconds: float,
                 now_provider: Callable[[], float] = time.monotonic) -> None:
        """
        Build one limiter with its own allowance, window and clock. Each place
        that limits by identity keeps its own instance, so their counts never
        interfere

        :param limit_per_minute: how many calls one key may make inside the
            window
        :param window_seconds: length of the sliding window in seconds
        :param now_provider: monotonic seconds source, injected for tests
        """
        self.limit = limit_per_minute
        self.window = window_seconds
        self._now = now_provider
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, cutoff: float) -> None:
        """
        Forget every key whose calls have all aged out, so the table cannot grow
        by one entry for each player id the server has ever seen. It runs only
        once the table is past RATE_LIMIT_PRUNE_THRESHOLD, leaving the ordinary
        call path untouched

        :param cutoff: monotonic timestamp in seconds; calls older than it no
            longer count towards any limit
        """
        for key in list(self._calls.keys()):
            d = self._calls[key]
            while d and d[0] < cutoff:
                d.popleft()
            if not d:
                del self._calls[key]

    def hit(self, key: str) -> bool:
        """
        Count one call for a key and say whether it is allowed. Calls that have
        aged out of the window are forgotten first, so an allowance comes back
        gradually rather than all at once on a boundary, and a refused call is
        not recorded -- being over the limit never extends the block

        :param key: identity being limited, such as a player id
        :returns: True when the call is inside the allowance, False when it is
            over and should be refused
        """
        now = self._now()
        cutoff = now - self.window
        if len(self._calls) > RATE_LIMIT_PRUNE_THRESHOLD:
            self._prune(cutoff)
        d = self._calls[key]
        while d and d[0] < cutoff:
            d.popleft()
        if len(d) >= self.limit:
            return False
        d.append(now)
        return True


def create_app(*, now_provider: Callable[[], float] = time.monotonic,
               max_rooms: int = DEFAULT_MAX_ROOMS) -> FastAPI:
    """
    Build the whole game server in one place: the room manager, the socket
    registry, the limiters, the background sweep and every HTTP and websocket
    route. A process makes one of these; tests make their own with a fake clock,
    which is how grace periods and deadlines are stepped instantly

    :param now_provider: monotonic seconds source shared by rooms, clocks,
        limiters and the sweep, injected so tests can control time
    :param max_rooms: how many rooms may exist at once, queued and running
        together, before new matchmaking is refused as server full
    :returns: the configured application, ready to serve
    """
    rooms = RoomManager(now_provider=now_provider, max_rooms=max_rooms)

    def now_ms() -> float:
        """
        Give the same monotonic clock in milliseconds, the unit every
        skill-check start, deadline and elapsed time is measured in

        :returns: monotonic time in milliseconds
        """
        return now_provider() * 1000.0

    connections = ConnectionRegistry()
    limiter = Limiter(key_func=client_ip_key)
    reclaim_limiter = UuidRateLimiter(
        RECLAIM_PER_UUID_LIMIT_PER_MINUTE, RECLAIM_WINDOW_SECONDS,
        now_provider=now_provider,
    )
    annotation_limiter = UuidRateLimiter(
        ANNOTATIONS_PER_SECOND, 1.0, now_provider=now_provider,
    )
    chat_limiter = UuidRateLimiter(
        1, CHAT_COOLDOWN_SECONDS, now_provider=now_provider,
    )
    started_at = now_provider()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """
        Run the server's startup and shutdown around the serving period: note
        the build and the trusted proxy set in the log, keep the background
        sweep alive for as long as the process serves, and on the way out
        record what was still going on before telling everyone still connected
        that the server is going down and closing their sockets, so nobody is
        left staring at a dead board

        :param app: application being started, carrying the shared state
        :returns: a context that stays open for the server's whole lifetime
        """
        log.info("gameserver v%d release=%s listening (max_rooms=%d)",
                 PROTOCOL_VERSION, app_version() or "dev", max_rooms)
        log_trusted_proxies()
        sweep_task = asyncio.create_task(_sweep_loop(app))
        try:
            yield
        finally:
            log.info("gameserver shutting down uptime_s=%.1f rooms_active=%d "
                     "queue_depth=%d sockets=%d",
                     now_provider() - started_at, rooms.rooms_active,
                     rooms.queue_depth, sum(1 for _ in connections.all_active()))
            sweep_task.cancel()
            shutdown_msg = ResultMessage(reason=Reason.SERVER_SHUTDOWN)
            for _, ws in list(connections.all_active()):
                await send(ws, shutdown_msg)
                try:
                    await ws.close(code=WS_CLOSE_SERVER_SHUTDOWN)
                except (RuntimeError, WebSocketDisconnect) as exc:
                    log.debug("ws close on shutdown failed: %s", exc)

    app = FastAPI(lifespan=lifespan)
    app.state.rooms = rooms
    app.state.connections = connections
    app.state.limiter = limiter
    app.state.now = now_provider
    app.state.now_ms = now_ms
    app.state.started_at = started_at
    app.state.reclaim_limiter = reclaim_limiter
    app.state.annotation_limiter = annotation_limiter
    app.state.chat_limiter = chat_limiter
    app.state.moderation_enabled = _moderation_enabled()
    if app.state.moderation_enabled:
        library.preload()
    app.state.sweep = Sweep(rooms, connections, now_provider, now_ms)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        """
        Answer a caller who has tripped one of the per-IP limits with the shape
        the client already understands: a 429 carrying the shared rate-limited
        reason code, which the client reports as a passing hiccup rather than a
        hard failure. Nothing about the limit itself is disclosed

        :param request: the refused request, not read here
        :param exc: the limiter's exception, not read here
        :returns: a 429 response carrying the reason code
        """
        return JSONResponse(
            status_code=429,
            content={"detail": {"reason": Reason.RATE_LIMITED}},
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        """
        Turn a request body the server could not read into the single reason
        code the client acts on, rather than a wall of field errors. The refusal
        is recorded with the endpoint and the field that failed, never with what
        was sent, so a rejected value cannot be written into the log this way

        :param request: the rejected request, read for the path it was sent to
        :param exc: the failure raised while validating the request body
        :returns: a 422 response carrying the invalid-field reason
        """
        errors = exc.errors()
        first: dict[str, Any] = dict(errors[0]) if errors else {}
        field = ".".join(str(part) for part in first.get("loc", ())) or "body"
        log.warning("request rejected path=%s field=%s error=%s",
                    request.url.path, field, first.get("type", "unknown"))
        return JSONResponse(
            status_code=422,
            content={"detail": {"reason": Reason.INVALID_FIELD}},
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """
        Service manifest for the game server: which protocol version it speaks
        and which endpoints it offers. A useful first call to confirm that a
        client and a server are talking the same protocol

        :returns: the service name, the protocol version and the endpoint list
        """
        return {
            "service": "gameserver",
            "version": PROTOCOL_VERSION,
            "endpoints": ["/healthz", "/matchmake", "/resume", "/reclaim", "/ws/{room_id}"],
        }

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        """
        Answer a browser asking for a site icon. This is a game server rather
        than a website, so the request is closed off with an empty reply instead
        of a not-found error

        :returns: an empty no-content response
        """
        return Response(status_code=204)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """
        Liveness and load check for the server, used both by monitoring and by
        the game's server picker. Reports the protocol and build version, how
        many games are being played, how many players are waiting for an
        opponent and how long the server has been up

        :returns: the current health and load summary
        """
        return HealthResponse(
            app_version=app_version(),
            rooms_active=rooms.rooms_active,
            queue_depth=rooms.queue_depth,
            uptime_s=now_provider() - app.state.started_at,
        )

    @app.post("/matchmake", response_model=MatchmakeResponse,
              responses={422: {"model": ReasonEnvelope}})
    @limiter.limit(MATCHMAKE_PER_IP_LIMIT)
    async def post_matchmake(request: Request, body: MatchmakeRequest) -> MatchmakeResponse:
        """
        Ask to be paired for an online game. Someone already waiting on the same
        time control is matched immediately; otherwise the request holds a place
        until an opponent arrives. Any game or waiting place the same player
        still holds is given up first, so a player is only ever in one game

        :param request: the incoming HTTP request; the endpoint itself reads
            only the body
        :param body: who is asking, the time control wanted and which side they
            would prefer to play
        :returns: the room to open the game connection on, with the session
            token for it
        """
        log.info("matchmake nickname=%s uuid=%s tc=%s+%s side=%s",
                 body.nickname, body.client_uuid[:8],
                 body.time_minutes, body.increment_seconds, body.side_preference)
        prior = rooms.in_progress_room_for(body.client_uuid)
        if prior is not None:
            prior_room, prior_color = prior
            log.info("matchmake abandons room=%s color=%s", prior_room.room_id, prior_color)
            await finalize_and_broadcast(rooms, connections, prior_room, Reason.ABANDONMENT,
                                         winner_color=prior_room.opp_color(prior_color))
            rooms.release_for_new_game(body.client_uuid)
        finished = rooms.finished_room_for(body.client_uuid)
        if finished is not None:
            fin_color = finished.color_of(body.client_uuid)
            if fin_color is not None:
                opp_ws = connections.get_for_color(finished, finished.opp_color(fin_color))
                if opp_ws is not None:
                    await send(opp_ws, RematchUpdateMessage(event="opponent_left"))
            log.info("matchmake leaves finished room=%s", finished.room_id)
            rooms.release_for_new_game(body.client_uuid)
        queued = rooms.queued_room_for(body.client_uuid)
        if queued is not None:
            log.info("matchmake releases queue slot room=%s", queued.room_id)
            rooms.release_for_new_game(body.client_uuid)
        token = RoomManager.make_session_token()
        try:
            room = await rooms.enqueue(
                client_uuid=body.client_uuid, nickname=body.nickname,
                session_token=token, time_minutes=body.time_minutes,
                increment_seconds=body.increment_seconds,
                side_preference=body.side_preference,
                country=body.country,
                hide_opp_marks=body.hide_opp_marks,
            )
        except AlreadyInGameError:
            log.info("matchmake rejected uuid=%s reason=already_in_game", body.client_uuid[:8])
            raise HTTPException(status_code=409, detail={"reason": Reason.ALREADY_IN_GAME})
        except ServerFullError:
            log.warning("matchmake rejected reason=%s rooms_active=%d queue_depth=%d "
                        "max_rooms=%d", Reason.ROOM_FULL, rooms.rooms_active,
                        rooms.queue_depth, max_rooms)
            raise HTTPException(status_code=503, detail={"reason": Reason.ROOM_FULL})
        if room.is_paired():
            log.info("room paired room=%s white=%s black=%s", room.room_id,
                     cast(PlayerSlot, room.white).client_uuid[:8],
                     cast(PlayerSlot, room.black).client_uuid[:8])
        else:
            log.info("room created room=%s uuid=%s", room.room_id, body.client_uuid[:8])
        return MatchmakeResponse(room_id=room.room_id, session_token=token)

    @app.delete("/matchmake", responses={422: {"model": ReasonEnvelope}})
    @limiter.limit(MATCHMAKE_PER_IP_LIMIT)
    async def delete_matchmake(request: Request,
                               body: CancelMatchmakeRequest) -> dict[str, str]:
        """
        Withdraw a player who is still waiting to be paired, freeing their place
        in the queue. A game that has already started cannot be withdrawn from
        this way, and the answer says so rather than failing

        :param request: the incoming HTTP request; the endpoint itself reads
            only the body
        :param body: the room the player was placed in and their session token
        :returns: a status of ok, or already_started when the game had begun
        """
        try:
            await rooms.cancel_wait(body.room_id, body.session_token)
        except NotInRoomError:
            raise HTTPException(status_code=404, detail={"reason": Reason.NOT_IN_ROOM})
        except InvalidTokenError:
            raise HTTPException(status_code=401, detail={"reason": Reason.SESSION_EXPIRED})
        except GameAlreadyStartedError:
            log.info("cancel ignored room=%s reason=game_already_started", body.room_id)
            return {"status": "already_started"}
        log.info("cancel ok room=%s", body.room_id)
        return {"status": "ok"}

    @app.post("/resume", response_model=ResumeResponse,
              responses={422: {"model": ReasonEnvelope}})
    @limiter.limit(RESUME_PER_IP_LIMIT)
    async def post_resume(request: Request, body: ResumeRequest) -> ResumeResponse:
        """
        Fetch the complete current state of a game the caller is playing:
        position and move list, both clocks, both players, anything still in
        flight and the result if there is one. Clients call it after a dropped
        connection so the board is rebuilt exactly rather than guessed at

        :param request: the incoming HTTP request; the endpoint itself reads
            only the body
        :param body: the room and the session token identifying the caller
        :returns: the full game state, told from that player's side
        """
        log.info("resume request room=%s", body.room_id)
        room = rooms.get(body.room_id)
        if room is None:
            log.info("resume rejected room=%s reason=not_in_room", body.room_id)
            raise HTTPException(status_code=404, detail={"reason": Reason.NOT_IN_ROOM})
        color, slot = room.slot_by_token(body.session_token)
        if slot is None:
            log.info("resume rejected room=%s reason=session_expired", body.room_id)
            raise HTTPException(status_code=401, detail={"reason": Reason.SESSION_EXPIRED})
        dead = room.pending_skillcheck
        if dead is not None and dead.is_dead(app.state.now_ms()):
            await resolve_skillcheck_fail(rooms, connections, room)
        if room.backend is not None:
            room.backend.tick_clock()
        history = [
            HistoryEntryWire(
                from_sq=coord_from_square(entry.move.from_sq),
                to_sq=coord_from_square(entry.move.to_sq),
                promotion=_promotion_letter(entry.move),
                san=entry.san,
            )
            for entry in (room.backend.move_history if room.backend else [])
        ]
        pending = _pending_skillcheck_wire(room, app.state.now_ms)
        locks = [LockWire(from_sq=coord_from_square(frm), to_sq=coord_from_square(to))
                 for frm, to in room.skillcheck_locks]
        skillcheck_log = [
            SkillCheckOutcomeWire(ply=e.ply, kind=e.kind, won=e.won, san=e.san)
            for e in room.skillcheck_log]
        white_annotations = _annotation_set_wire(room.annotations_white)
        black_annotations = _annotation_set_wire(room.annotations_black)
        if room.hides_opponent_marks(cast(str, color)):
            if color == "white":
                black_annotations = AnnotationSetWire()
            else:
                white_annotations = AnnotationSetWire()
        response = ResumeResponse(
            fen=export_fen(cast(Backend, room.backend)),
            move_history=history,
            clock=clock_snapshot(cast(Backend, room.backend).clock),
            your_color=color,
            white_name=room.white.nickname if room.white else "",
            black_name=room.black.nickname if room.black else "",
            time_minutes=room.time_minutes,
            increment_seconds=room.increment_seconds,
            white_score=room.score_for("white"),
            black_score=room.score_for("black"),
            white_country=room.white.country if room.white else None,
            black_country=room.black.country if room.black else None,
            pending_skillcheck=pending,
            skillcheck_locks=locks,
            skillcheck_log=skillcheck_log,
            white_annotations=white_annotations,
            black_annotations=black_annotations,
            share_muted=room.annotations_for(cast(str, color)).share_muted,
            hide_opp_marks=slot.hide_opp_marks,
            result_reason=room.result[0] if room.result else None,
            result_winner=room.result[1] if room.result else None,
            idle_window=idle_window_wire(room, app.state.now()),
        )
        log.info("resume served room=%s color=%s ply=%d", body.room_id, color, len(history))
        if (connections.get_for_color(room, cast(str, color)) is not None
                and not slot.desync_active):
            slot.desync_active = True
            slot.ply_mismatch_streak = 0
            opp_ws = connections.get_for_color(
                room, room.opp_color(cast(str, color)))
            if opp_ws is not None:
                await send(opp_ws, ConnectionStatusMessage(opp_state="resyncing"))
        return response

    @app.post("/reclaim", response_model=ReclaimResponse,
              responses={422: {"model": ReasonEnvelope}})
    @limiter.limit(RECLAIM_PER_IP_LIMIT)
    async def post_reclaim(request: Request, body: ReclaimRequest) -> ReclaimResponse:
        """
        Ask whether a game is still waiting for a player, knowing nothing about
        them but their player id. The game does this when the app is started
        again and the session token from the previous run is gone; a fresh token
        is issued for whatever game is found

        :param request: the incoming HTTP request; the endpoint itself reads
            only the body
        :param body: the player id to look up
        :returns: the room to reconnect to, with a newly issued session token
        """
        if not reclaim_limiter.hit(body.client_uuid):
            log.info("reclaim rate-limited uuid=%s", body.client_uuid[:8])
            raise HTTPException(status_code=429, detail={"reason": Reason.RATE_LIMITED})
        log.info("reclaim request uuid=%s", body.client_uuid[:8])
        try:
            room, color, new_token = await rooms.reclaim_session(body.client_uuid)
        except NotInRoomError:
            raise HTTPException(status_code=404, detail={"reason": Reason.NOT_IN_ROOM})
        old_ws = connections.get_for_uuid(room.room_id, body.client_uuid)
        if old_ws is not None:
            try:
                await old_ws.close(code=WS_CLOSE_SUPERSEDED)
            except (RuntimeError, WebSocketDisconnect) as exc:
                log.debug("ws close on reclaim failed: %s", exc)
        log.info("reclaim ok uuid=%s room=%s color=%s",
                 body.client_uuid[:8], room.room_id, color)
        return ReclaimResponse(room_id=room.room_id, session_token=new_token)

    @app.websocket("/ws/{room_id}")
    async def ws_endpoint(websocket: WebSocket, room_id: str) -> None:
        """
        Front door for a game's live connection, where every move, clock update
        and skill-check shot travels. The room id is shape-checked before the
        socket is even accepted, so a malformed path never reaches a room
        lookup; everything past the accept belongs to the session loop, starting
        with the authentication handshake

        :param websocket: the incoming socket, not yet accepted
        :param room_id: room taken from the path, refused unless it is a UUID4
        """
        if not is_uuid4(room_id):
            await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
            return
        await websocket.accept()
        await _ws_session(app, websocket, room_id)

    return app


async def _sweep_loop(app: FastAPI) -> None:
    """
    The server's heartbeat. Every tick it runs the whole sweep -- clocks,
    skill-check deadlines, idle windows, disconnect grace, queue reaping and
    cleanup of finished rooms -- which is what makes a game end on time even
    when nobody sends anything. It starts with the app and only ever stops by
    being cancelled at shutdown

    :param app: application whose sweep and shared state the loop drives
    """
    try:
        while True:
            await asyncio.sleep(CLOCK_TICK_INTERVAL_SECONDS)
            await app.state.sweep.step_all()
    except asyncio.CancelledError:
        pass


def _over_inbound_cap(raw: str) -> bool:
    """
    Say whether one inbound websocket frame is larger than the server accepts.
    Every frame is measured, the authentication handshake included, and the same
    MAX_INBOUND_MESSAGE_BYTES ceiling is handed to the web server as its own
    frame limit, so an oversized frame is refused at both layers

    :param raw: decoded frame text, measured as UTF-8 bytes rather than
        characters
    :returns: True when the frame is over the ceiling and must be refused
    """
    return len(raw.encode("utf-8")) > MAX_INBOUND_MESSAGE_BYTES


def _promotion_letter(move: Move) -> str | None:
    """
    Give the wire letter for a promotion -- q, r, b or n -- when a stored move is
    written back out to a client. A move that promoted nothing has no letter to
    give

    :param move: engine move taken from the room's history
    :returns: the promotion letter, or None when the move was not a promotion
    """
    if move.promoted_to is None:
        return None
    return PROMO_LETTER_BY_TYPE.get(move.promoted_to)


def _annotation_set_wire(store: SharedAnnotations) -> AnnotationSetWire:
    """
    Turn one player's stored board marks into the whole-set form a resuming
    client is sent, so a reconnecting player sees the same drawing they left.
    Highlights go out sorted, which keeps a restored board rebuilding in a
    stable order

    :param store: that player's shared-annotation state held on the room
    :returns: the sharing flag together with every highlight and arrow
    """
    return AnnotationSetWire(
        sharing=store.sharing,
        highlights=sorted(store.highlights),
        arrows=arrow_wires(store.arrows),
    )


def _pending_skillcheck_wire(
    room: Room, now_ms: Callable[[], float],
) -> PendingSkillCheckWire | None:
    """
    Describe the skill check a returning player still has to beat, so they
    rejoin the challenge already in progress instead of starting it afresh. The
    challenge is described from the server's own seed and how long it has been
    running; a check that has already run out is reported as nothing, because
    the caller resolves it as a miss instead

    :param room: room being resumed
    :param now_ms: monotonic clock in milliseconds, read for the elapsed time
    :returns: the pending challenge with its progress, or None when there is
        none or it is already dead
    """
    pending = room.pending_skillcheck
    if pending is None or pending.is_dead(now_ms()):
        return None
    elapsed = max(0.0, now_ms() - pending.start_ms)
    return PendingSkillCheckWire(
        kind=pending.kind.value, seed=pending.seed, value_diff=pending.value_diff,
        deadline_ms=pending.deadline_ms, captured_value=pending.captured_value,
        elapsed_ms=elapsed, miss_count=pending.miss_count, progress=pending.progress,
        last_hit_pop=pending.last_hit_pop,
        from_sq=coord_from_square(pending.from_sq),
        to_sq=coord_from_square(pending.to_sq),
        promotion=pending.promotion, color=pending.color,
    )


async def _authenticate_ws(
    websocket: WebSocket, rooms: RoomManager, room_id: str,
) -> tuple[Room, str, PlayerSlot] | None:
    """
    Run the handshake that has to precede any game traffic at all. The first
    frame must arrive inside the pairing wait, fit the inbound size cap, be an
    auth message on this exact protocol version, and name a session token that
    matches a slot in an existing room. Anything else closes the socket and lets
    nobody in

    :param websocket: accepted socket that has not identified itself yet
    :param rooms: room manager the room id is looked up in
    :param room_id: room from the path, already shape-checked as a UUID4
    :returns: the room, the caller's color and their slot, or None when the
        handshake failed and the socket was closed
    """
    try:
        first_raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=PAIRING_WAIT_SECONDS)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
        return None
    if _over_inbound_cap(first_raw):
        await websocket.close(code=WS_CLOSE_PAYLOAD_TOO_LARGE)
        return None
    try:
        auth_msg = AuthMessage.model_validate_json(first_raw)
    except (ValidationError, ValueError):
        await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
        return None
    if auth_msg.version != PROTOCOL_VERSION:
        await send(websocket, ErrorMessage(reason=Reason.VERSION_MISMATCH))
        await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
        return None
    room = rooms.get(room_id)
    if room is None:
        await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
        return None
    color, slot = room.slot_by_token(auth_msg.session_token)
    if slot is None:
        await websocket.close(code=WS_CLOSE_INVALID_TOKEN)
        return None
    return room, cast(str, color), slot


async def _ws_session(app: FastAPI, websocket: WebSocket, room_id: str) -> None:
    """
    Own one player's live connection for as long as it lasts: authenticate,
    file the socket while closing any older one the same player left behind,
    tell them where the game currently stands, then pump inbound frames through
    the dispatch table until the socket goes away. The player's color is re-read
    from the room on every frame, because a rematch swaps colors underneath a
    connection that never dropped. Oversized frames close the socket outright,
    and a flood is answered with an error frame instead of being processed

    :param app: application holding the rooms, sockets and clock
    :param websocket: accepted socket for this player
    :param room_id: room the socket connected to
    """
    rooms = app.state.rooms
    connections = app.state.connections
    auth = await _authenticate_ws(websocket, rooms, room_id)
    if auth is None:
        return
    room, color, slot = auth
    auth_room = room
    auth_color = color
    auth_uuid = slot.client_uuid
    rooms.mark_connected(room.room_id, color)
    displaced = connections.add(room.room_id, auth_uuid, websocket)
    if displaced is not None:
        try:
            await displaced.close(code=WS_CLOSE_SUPERSEDED)
        except (RuntimeError, WebSocketDisconnect) as exc:
            log.debug("ws close on supersede failed: %s", exc)
    log.info("ws auth ok room=%s uuid=%s tentative_color=%s paired=%s has_both=%s",
             room.room_id, auth_uuid[:8], color, room.is_paired(),
             connections.has_both(room))

    if room.result is not None:
        slot.at_result = True
        reason, winner = room.result
        await send(websocket, ResultMessage(reason=reason, winner_color=winner))
        if room.opp_color(color) in room.rematch_offered_by:
            await send(websocket, RematchRequestMessage())
        opp_ws = connections.get_for_color(room, room.opp_color(color))
        if opp_ws is not None:
            await send(opp_ws, RematchUpdateMessage(event="opponent_returned"))
    elif room.is_paired() and connections.has_both(room) and not room.game_start_broadcast:
        if room.started_at is None:
            room.started_at = app.state.now()
        await broadcast_game_start(connections, room, app.state.now)
    else:
        opp_ws = connections.get_for_color(room, room.opp_color(color))
        if opp_ws is not None:
            await send(opp_ws, ConnectionStatusMessage(opp_state="connected"))
        if room.game_start_broadcast:
            await send(websocket, ConnectionStatusMessage(
                opp_state="connected" if opp_ws is not None else "reconnecting"))

    ws_rate_limiter = UuidRateLimiter(
        WS_MESSAGES_PER_SECOND, WS_RATE_WINDOW_SECONDS,
        now_provider=app.state.now,
    )

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except RuntimeError as exc:
                log.debug("ws recv on superseded/closed socket room=%s color=%s: %s",
                          room.room_id, color, exc)
                break
            except Exception:
                log.exception("ws recv failed room=%s color=%s",
                              room.room_id, room.color_of(auth_uuid) or color)
                break
            if _over_inbound_cap(raw):
                await websocket.close(code=WS_CLOSE_PAYLOAD_TOO_LARGE)
                return
            current_color = room.color_of(auth_uuid)
            if current_color is None:
                break
            rooms.touch_seen(auth_room.room_id, current_color)
            if not ws_rate_limiter.hit(auth_uuid):
                await send(websocket, ErrorMessage(reason=Reason.RATE_LIMITED))
                continue
            t0 = app.state.now()
            msg_type, outcome = await dispatch(app, websocket, room, current_color, raw)
            log.debug("ws dispatch room=%s uuid=%s type=%s latency_ms=%.1f outcome=%s",
                      room.room_id, auth_uuid[:8], msg_type,
                      (app.state.now() - t0) * 1000.0, outcome)
    finally:
        removed = connections.remove(auth_room.room_id, auth_uuid, websocket)
        cur_color = auth_room.color_of(auth_uuid) or auth_color
        if removed or connections.get_for_color(auth_room, cur_color) is None:
            rooms.mark_disconnected(auth_room.room_id, cur_color)
            log.info("ws disconnected room=%s color=%s",
                     auth_room.room_id, cur_color)
            opp_ws = connections.get_for_color(auth_room, auth_room.opp_color(cur_color))
            if opp_ws is not None:
                msg = (ConnectionStatusMessage(opp_state="reconnecting")
                       if auth_room.result is None
                       else RematchUpdateMessage(event="opponent_reconnecting"))
                await send(opp_ws, msg)
