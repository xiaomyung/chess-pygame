import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from chessshootout.server import logging_setup
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.handlers import (
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS,
)
from chessshootout.server.limits import (
    RECLAIM_PER_UUID_LIMIT_PER_MINUTE, RECLAIM_WINDOW_SECONDS, UuidRateLimiter,
    client_ip_key, log_trusted_proxies,
)
from chessshootout.server.moderation import library
from chessshootout.server.protocol import (
    ANNOTATIONS_PER_SECOND, CHAT_COOLDOWN_SECONDS, GRACE_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS, HEARTBEAT_MISS_LIMIT, HEARTBEAT_TIMEOUT_SECONDS,
    PROTOCOL_VERSION, Reason, ResultMessage, WS_CLOSE_SERVER_SHUTDOWN,
)
from chessshootout.server.rooms import RoomManager
from chessshootout.server.routes_http import app_version, build_http_router
from chessshootout.server.sweep import SWEEP_STALE_SECONDS, Sweep
from chessshootout.server.ws_session import ws_router


CLOCK_TICK_INTERVAL_SECONDS = 0.1
DEFAULT_MAX_ROOMS = 100

log = logging_setup.get_logger("chess.server.app")


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
        log.info("tuning grace=%.1f heartbeat=%.1f miss_limit=%d heartbeat_timeout=%.1f "
                 "tick=%.2f sweep_stale=%.1f transit_grace=%.2f stable_heartbeats=%d",
                 GRACE_SECONDS, HEARTBEAT_INTERVAL_SECONDS, HEARTBEAT_MISS_LIMIT,
                 HEARTBEAT_TIMEOUT_SECONDS, CLOCK_TICK_INTERVAL_SECONDS,
                 SWEEP_STALE_SECONDS, RESYNC_TRANSIT_GRACE_SECONDS,
                 RESYNC_STABLE_MISMATCH_HEARTBEATS)
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

    app.include_router(build_http_router(
        limiter=limiter, rooms=rooms, connections=connections, sweep=app.state.sweep,
        now_provider=now_provider, now_ms=now_ms, started_at=started_at,
        reclaim_limiter=reclaim_limiter, max_rooms=max_rooms,
    ))
    app.include_router(ws_router)

    return app


async def _sweep_loop(app: FastAPI) -> None:
    """
    The server's heartbeat. Every tick it runs the whole sweep -- clocks,
    skill-check deadlines, idle windows, disconnect grace, queue reaping and
    cleanup of finished rooms -- which is what makes a game end on time even
    when nobody sends anything. A pass that fails outright is recorded and the
    next tick still runs, so the loop starts with the app and only ever stops by
    being cancelled at shutdown

    :param app: application whose sweep and shared state the loop drives
    """
    sweep: Sweep = app.state.sweep
    sweep.mark_running()
    try:
        while True:
            await asyncio.sleep(CLOCK_TICK_INTERVAL_SECONDS)
            try:
                await sweep.step_all()
            except Exception as exc:
                sweep.note_unhandled(exc)
    except asyncio.CancelledError:
        pass
