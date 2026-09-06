import asyncio
from typing import cast

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from chessshootout.server import logging_setup
from chessshootout.server.broadcasts import broadcast_game_start
from chessshootout.server.connections import ConnectionRegistry, send
from chessshootout.server.handlers import dispatch
from chessshootout.server.limits import UuidRateLimiter
from chessshootout.server.protocol import (
    AuthMessage, ConnectionStatusMessage, ErrorMessage, PROTOCOL_VERSION, Reason,
    RematchRequestMessage, RematchUpdateMessage, ResultMessage,
    WS_CLOSE_INVALID_TOKEN, WS_CLOSE_PAYLOAD_TOO_LARGE, WS_CLOSE_SUPERSEDED,
    is_uuid4,
)
from chessshootout.server.rooms import PAIRING_WAIT_SECONDS, PlayerSlot, Room, RoomManager


MAX_INBOUND_MESSAGE_BYTES = 4096

WS_MESSAGES_PER_SECOND = 30
WS_RATE_WINDOW_SECONDS = 1.0

log = logging_setup.get_logger("chess.server.app")

ws_router = APIRouter()


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
    rooms: RoomManager = app.state.rooms
    connections: ConnectionRegistry = app.state.connections
    auth = await _authenticate_ws(websocket, rooms, room_id)
    if auth is None:
        return
    room, auth_color, slot = auth
    auth_uuid = slot.client_uuid
    rooms.mark_connected(room.room_id, auth_color)
    displaced = connections.add(room.room_id, auth_uuid, websocket)
    if displaced is not None:
        try:
            await displaced.close(code=WS_CLOSE_SUPERSEDED)
        except (RuntimeError, WebSocketDisconnect) as exc:
            log.debug("ws close on supersede failed: %s", exc)
    log.info("ws auth ok room=%s uuid=%s tentative_color=%s paired=%s has_both=%s",
             room.room_id, auth_uuid[:8], auth_color, room.is_paired(),
             connections.has_both(room))

    if room.result is not None:
        slot.at_result = True
        reason, winner = room.result
        await send(websocket, ResultMessage(reason=reason, winner_color=winner))
        if room.opp_color(auth_color) in room.rematch_offered_by:
            await send(websocket, RematchRequestMessage())
        opp_ws = connections.get_for_color(room, room.opp_color(auth_color))
        if opp_ws is not None:
            await send(opp_ws, RematchUpdateMessage(event="opponent_returned"))
    elif room.is_paired() and connections.has_both(room) and not room.game_start_broadcast:
        if room.started_at is None:
            room.started_at = app.state.now()
        await broadcast_game_start(connections, room, app.state.now)
    else:
        opp_ws = connections.get_for_color(room, room.opp_color(auth_color))
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
                          room.room_id, auth_color, exc)
                break
            except Exception:
                log.exception("ws recv failed room=%s color=%s",
                              room.room_id, room.color_of(auth_uuid) or auth_color)
                break
            if _over_inbound_cap(raw):
                await websocket.close(code=WS_CLOSE_PAYLOAD_TOO_LARGE)
                return
            current_color = room.color_of(auth_uuid)
            if current_color is None:
                break
            rooms.touch_seen(room.room_id, current_color)
            if not ws_rate_limiter.hit(auth_uuid):
                await send(websocket, ErrorMessage(reason=Reason.RATE_LIMITED))
                continue
            t0 = app.state.now()
            msg_type, outcome = await dispatch(app, websocket, room, current_color, raw)
            log.debug("ws dispatch room=%s uuid=%s type=%s latency_ms=%.1f outcome=%s",
                      room.room_id, auth_uuid[:8], msg_type,
                      (app.state.now() - t0) * 1000.0, outcome)
    finally:
        removed = connections.remove(room.room_id, auth_uuid, websocket)
        exit_color = room.color_of(auth_uuid) or auth_color
        if removed or connections.get_for_color(room, exit_color) is None:
            rooms.mark_disconnected(room.room_id, exit_color)
            log.info("ws disconnected room=%s color=%s",
                     room.room_id, exit_color)
            opp_ws = connections.get_for_color(room, room.opp_color(exit_color))
            if opp_ws is not None:
                msg = (ConnectionStatusMessage(opp_state="reconnecting")
                       if room.result is None
                       else RematchUpdateMessage(event="opponent_reconnecting"))
                await send(opp_ws, msg)


@ws_router.websocket("/ws/{room_id}")
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
    await _ws_session(cast(FastAPI, websocket.app), websocket, room_id)
