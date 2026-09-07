from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, WebSocketDisconnect
from fastapi.responses import Response
from slowapi import Limiter

from chessshootout.backend.backend import Backend
from chessshootout.backend.fen import export_fen
from chessshootout.backend.utils import (
    Move, PROMO_LETTER_BY_TYPE, coord_from_square,
)
from chessshootout.server import logging_setup
from chessshootout.server.broadcasts import (
    arrow_wires, clock_snapshot, finalize_and_broadcast, idle_window_wire,
    resolve_skillcheck_fail,
)
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.limits import (
    MATCHMAKE_PER_IP_LIMIT, RECLAIM_PER_IP_LIMIT, RESUME_PER_IP_LIMIT, UuidRateLimiter,
)
from chessshootout.server.protocol import (
    AnnotationSetWire, CancelMatchmakeRequest, ConnectionStatusMessage,
    HealthResponse, HealthStatus, HistoryEntryWire, LockWire,
    MIN_CLIENT_VERSION, MatchmakeRequest, MatchmakeResponse,
    PROTOCOL_VERSION, PendingSkillCheckWire, Reason, ReasonEnvelope, ReclaimRequest,
    ReclaimResponse, RematchUpdateMessage, ResumeRequest, ResumeResponse,
    SkillCheckOutcomeWire, WS_CLOSE_SUPERSEDED,
    client_version_outdated, parse_client_version, version_text,
)
from chessshootout.server.rooms import (
    AlreadyInGameError, GameAlreadyStartedError, InvalidTokenError, NotInRoomError,
    PlayerSlot, Room, RoomManager, ServerFullError, SharedAnnotations,
)
from chessshootout.server.sweep import Sweep


log = logging_setup.get_logger("chess.server.app")


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
    at_ms = now_ms()
    if pending is None or pending.is_dead(at_ms):
        return None
    elapsed = max(0.0, at_ms - pending.start_ms)
    return PendingSkillCheckWire(
        kind=pending.kind.value, seed=pending.seed, value_diff=pending.value_diff,
        deadline_ms=pending.deadline_ms, captured_value=pending.captured_value,
        elapsed_ms=elapsed, miss_count=pending.miss_count, progress=pending.progress,
        last_hit_pop=pending.last_hit_pop,
        from_sq=coord_from_square(pending.from_sq),
        to_sq=coord_from_square(pending.to_sq),
        promotion=pending.promotion, color=pending.color,
    )


def build_http_router(
    *,
    limiter: Limiter,
    rooms: RoomManager,
    connections: ConnectionRegistry,
    sweep: Sweep,
    now_provider: Callable[[], float],
    now_ms: Callable[[], float],
    started_at: float,
    reclaim_limiter: UuidRateLimiter,
    max_rooms: int,
) -> APIRouter:
    """
    Build the server's whole HTTP surface -- the manifest, the health check,
    matchmaking, resume and reclaim -- against one application's own parts. The
    routes are defined here rather than at import time because each application
    owns its rooms, its clock and its rate limiter, and a shared router would
    make every server in a test run count against the same budgets

    :param limiter: that application's per-IP limiter, bound into the decorators
    :param rooms: room manager the endpoints queue, look up and release in
    :param connections: live socket registry, used to notify an opponent
    :param sweep: background upkeep, read by the health check for its freshness
    :param now_provider: monotonic seconds source shared with the rest of the app
    :param now_ms: the same clock in milliseconds, for skill-check timing
    :param started_at: monotonic timestamp the server began serving at
    :param reclaim_limiter: per-player limiter guarding session reclaims
    :param max_rooms: how many rooms may exist at once before matchmaking is
        refused as server full
    :returns: the router to mount on the application
    """
    router = APIRouter()
    build_version = app_version()

    def _enforce_client_build(body: MatchmakeRequest) -> None:
        """
        Turn a build this server will not play with away before matchmaking
        touches a single room: one that speaks another protocol version, and one
        older than the oldest release still accepted. Both refusals name the
        reason, and the outdated one names the version to update to

        :param body: the matchmaking request, read for its two version fields
        """
        if body.version != PROTOCOL_VERSION:
            log.info("matchmake rejected uuid=%s reason=%s version=%d",
                     body.client_uuid[:8], Reason.VERSION_MISMATCH, body.version)
            raise HTTPException(status_code=426,
                                detail={"reason": Reason.VERSION_MISMATCH})
        if client_version_outdated(body.client_version, MIN_CLIENT_VERSION):
            parsed = parse_client_version(body.client_version)
            log.info("matchmake rejected uuid=%s reason=%s version=%s",
                     body.client_uuid[:8], Reason.CLIENT_OUTDATED,
                     "unparseable" if parsed is None else version_text(parsed))
            raise HTTPException(status_code=426,
                                detail={"reason": Reason.CLIENT_OUTDATED,
                                        "min_version": MIN_CLIENT_VERSION})

    @router.get("/")
    async def root() -> dict[str, Any]:
        """
        Service manifest for the game server: which protocol version it speaks,
        the oldest build of the game it still accepts, and which endpoints it
        offers. A useful first call to confirm that a client and a server are
        talking the same protocol

        :returns: the service name, the protocol version, the oldest accepted
            build version and the endpoint list
        """
        return {
            "service": "gameserver",
            "version": PROTOCOL_VERSION,
            "min_client_version": MIN_CLIENT_VERSION,
            "endpoints": ["/healthz", "/matchmake", "/resume", "/reclaim", "/ws/{room_id}"],
        }

    @router.get("/favicon.ico")
    async def favicon() -> Response:
        """
        Answer a browser asking for a site icon. This is a game server rather
        than a website, so the request is closed off with an empty reply instead
        of a not-found error

        :returns: an empty no-content response
        """
        return Response(status_code=204)

    @router.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """
        Liveness and load check for the server, used both by monitoring and by
        the game's server picker. Reports whether the server is serving
        normally, has no room left for another game or has fallen behind on the
        upkeep it runs on its own, along with the protocol and build version,
        how many games are being played, how many players are waiting for an
        opponent, how long the server has been up and how many seconds it is
        since the last complete upkeep pass

        :returns: the current health and load summary
        """
        status = HealthStatus.OK
        if rooms.rooms_active + rooms.queue_depth >= max_rooms:
            status = HealthStatus.FULL
        elif sweep.is_stale:
            status = HealthStatus.DEGRADED
        return HealthResponse(
            status=status,
            app_version=build_version,
            rooms_active=rooms.rooms_active,
            queue_depth=rooms.queue_depth,
            uptime_s=now_provider() - started_at,
            housekeeping_age_s=sweep.age_s,
        )

    @router.post("/matchmake", response_model=MatchmakeResponse,
                 responses={422: {"model": ReasonEnvelope}})
    @limiter.limit(MATCHMAKE_PER_IP_LIMIT)
    async def post_matchmake(request: Request, body: MatchmakeRequest) -> MatchmakeResponse:
        """
        Ask to be paired for an online game. Someone already waiting on the same
        time control is matched immediately; otherwise the request holds a place
        until an opponent arrives. Any game or waiting place the same player
        still holds is given up first, so a player is only ever in one game. A
        build that speaks another protocol version, or one older than this
        server accepts, is turned away before any of that happens

        :param request: the incoming HTTP request; the endpoint itself reads
            only the body
        :param body: who is asking, which build they run, the time control
            wanted and which side they would prefer to play
        :returns: the room to open the game connection on, with the session
            token for it
        """
        _enforce_client_build(body)
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

    @router.delete("/matchmake", responses={422: {"model": ReasonEnvelope}})
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

    @router.post("/resume", response_model=ResumeResponse,
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
        seat_color = cast(str, color)
        dead = room.pending_skillcheck
        if dead is not None and dead.is_dead(now_ms()):
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
        pending = _pending_skillcheck_wire(room, now_ms)
        locks = [LockWire(from_sq=coord_from_square(frm), to_sq=coord_from_square(to))
                 for frm, to in room.skillcheck_locks]
        skillcheck_log = [
            SkillCheckOutcomeWire(ply=e.ply, kind=e.kind, won=e.won, san=e.san)
            for e in room.skillcheck_log]
        white_annotations = _annotation_set_wire(room.annotations_white)
        black_annotations = _annotation_set_wire(room.annotations_black)
        if room.hides_opponent_marks(seat_color):
            if seat_color == "white":
                black_annotations = AnnotationSetWire()
            else:
                white_annotations = AnnotationSetWire()
        backend = cast(Backend, room.backend)
        response = ResumeResponse(
            fen=export_fen(backend),
            move_history=history,
            clock=clock_snapshot(backend.clock),
            your_color=seat_color,
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
            share_muted=room.annotations_for(seat_color).share_muted,
            hide_opp_marks=slot.hide_opp_marks,
            result_reason=room.result[0] if room.result else None,
            result_winner=room.result[1] if room.result else None,
            idle_window=idle_window_wire(room, now_provider()),
        )
        log.info("resume served room=%s color=%s ply=%d",
                 body.room_id, seat_color, len(history))
        slot.clear_strikes()
        if room.result is None:
            if (connections.get_for_color(room, seat_color) is not None
                    and not slot.desync_active):
                slot.mark_desynced()
                opp_ws = connections.get_for_color(room, room.opp_color(seat_color))
                if opp_ws is not None:
                    await send(opp_ws, ConnectionStatusMessage(opp_state="resyncing"))
        return response

    @router.post("/reclaim", response_model=ReclaimResponse,
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

    return router
