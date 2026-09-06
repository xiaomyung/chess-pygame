import ipaddress
import json
import logging
import ssl
from collections.abc import Callable, Iterable
from typing import Any, NoReturn, TypeVar, cast

import certifi
import httpx
import websockets
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from chessshootout.server.protocol import (
    AnnotationDeltaMessage, AnnotationsStateMessage, ArrowWire, AuthMessage,
    CancelMatchmakeRequest, DrawResponseMessage, GiveTimeMessage, HealthResponse,
    MatchmakeRequest, MatchmakeResponse, MoveMessage, PingMessage, PROTOCOL_VERSION,
    QuickChatMessage, Reason, ReclaimRequest, ReclaimResponse, RematchRequestMessage,
    RematchResponseMessage, ResumeRequest, ResumeResponse, SetMarksVisibilityMessage,
    SkillCheckShotMessage, TakebackResponseMessage,
)

WsConnectionClosed = ConnectionClosed


log = logging.getLogger("chess.client.transport")


HTTP_TIMEOUT_SECONDS = 5.0
NEWS_TIMEOUT_SECONDS = 5.0
RECLAIM_TIMEOUT_SECONDS = 2.0
HEALTHZ_TIMEOUT_SECONDS = 3.0
WS_PING_INTERVAL_SECONDS = 20
WS_PING_TIMEOUT_SECONDS = 30
NEWS_MAX_BYTES = 512 * 1024


_TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())

_ResponseModel = TypeVar("_ResponseModel", bound=BaseModel)


class TransportError(Exception):
    """
    Base class for every way talking to the game server can fail on this side:
    an unreachable host, an address that cannot be sent to, a refused request,
    a body that will not decode. Callers catch this one type and choose between
    retrying, degrading to offline, and telling the player
    """

    pass


class SchemaVersionMismatch(TransportError):
    """
    Raised when the server speaks a different protocol version than this build.
    Nothing can be played across that gap, so the client stops and asks the
    player to update instead of retrying
    """

    pass


class TransportHTTPError(TransportError):
    """
    A request the server answered with an error status. It carries the status
    and the server's reason code, which is how a caller tells a full server
    worth retrying from a request that was rejected outright
    """

    def __init__(self, status_code: int, reason: str) -> None:
        """
        Record the failed request's status and reason so callers can branch on
        them rather than parse the message text

        :param status_code: HTTP status the server answered with.
        :param reason: server-supplied reason code, or an http_<status>
            stand-in when the body carried nothing usable.
        """
        super().__init__(f"http_{status_code}:{reason}")
        self.status_code = status_code
        self.reason = reason


class FatalResumeError(TransportError):
    """
    Raised when rejoining a game is pointless because the room or the seat is
    gone for good. It tells the reconnect loop to give up at once instead of
    spending the whole reconnect window on a game that no longer exists
    """

    pass


class ResponseTooLarge(TransportError):
    """
    Raised when a streamed body outgrows its byte budget, which is how the news
    fetch refuses to buffer an oversized response. The stream is abandoned on
    the spot, so the rest of the body is never read
    """

    pass


def _reject_json_constant(name: str) -> NoReturn:
    """
    Refuse the non-standard JSON constants NaN, Infinity and -Infinity, which
    Python's decoder otherwise accepts. A server-supplied infinity would flow
    straight into clocks and drawing code, so every decode on this side turns
    one into a decode failure

    :param name: the constant token the decoder ran into, e.g. NaN
    """
    raise ValueError(f"non_standard_json_constant: {name}")


def _loads(raw: str | bytes) -> Any:
    """
    Decode one JSON payload from the server the strict way: exactly like
    json.loads, minus the non-standard constants. Every HTTP body and every
    websocket frame the client reads goes through here

    :param raw: undecoded JSON text or bytes exactly as received.
    :returns: the decoded value, whose shape the caller still has to check.
    """
    return json.loads(raw, parse_constant=_reject_json_constant)


def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """
    Read a streamed body while counting bytes, and abandon it the moment it
    grows past the budget. Keeps a hostile or broken host from filling the
    fetching thread's memory with a response nobody asked for

    :param response: open streaming response, not yet consumed.
    :param max_bytes: hard ceiling; reading stops as soon as it is passed.
    :returns: the whole body, once it is known to fit inside the budget.
    """
    chunks = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLarge(f"response_over_{max_bytes}_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_news(url: str, *, timeout: float = NEWS_TIMEOUT_SECONDS,
               max_bytes: int = NEWS_MAX_BYTES) -> Any:
    """
    Fetch the raw news feed behind the menu's News card. It lives in this
    module because transport is the only client-side module allowed to import
    httpx, and the feed is treated as untrusted throughout: capped while it
    streams, then decoded without the non-standard JSON constants

    :param url: feed address, from CHESS_NEWS_URL or the built-in default.
    :param timeout: per-request timeout in seconds.
    :param max_bytes: largest body accepted before the stream is abandoned.
    :returns: the decoded feed, normally a list of news items.
    """
    try:
        with httpx.stream("GET", url, timeout=timeout, verify=_TLS_CONTEXT) as r:
            if r.status_code != 200:
                raise TransportHTTPError(r.status_code, f"http_{r.status_code}")
            body = _read_capped(r, max_bytes)
    except (httpx.HTTPError, httpx.TimeoutException, httpx.InvalidURL, UnicodeError) as exc:
        raise TransportError(str(exc)) from exc
    try:
        return _loads(body)
    except ValueError as exc:
        raise TransportError(f"invalid_json: {exc}") from exc


def _looks_like_ip_or_localhost(host: str) -> bool:
    """
    Tell whether a host is a bare address rather than a domain name, which is
    the hint that the player is pointing at a LAN or development server. Those
    default to plaintext, since there is no certificate to verify

    :param host: hostname or address from the server setting, port removed.
    :returns: True for localhost or a literal IP address.
    """
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _split_addr(addr: str) -> tuple[str, str, int]:
    """
    Turn whatever the player typed as a server address into the three things
    the client needs: scheme, host and port. An explicit scheme always wins; a
    bare IP or port 8000 means plaintext, and anything else defaults to TLS. A
    port that is not a number is rejected as an unusable address

    :param addr: address as configured, with or without scheme, port or path.
    :returns: websocket scheme (ws or wss), host, and port.
    """
    explicit_scheme = None
    rest = addr
    for prefix in ("ws://", "wss://", "http://", "https://"):
        if addr.startswith(prefix):
            explicit_scheme = prefix
            rest = addr[len(prefix):]
            break
    if "/" in rest:
        rest = rest.split("/", 1)[0]
    if ":" in rest:
        host, port_s = rest.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError as exc:
            raise TransportError(f"invalid_address: {addr}") from exc
    else:
        host = rest
        port = None
    if explicit_scheme:
        ws_scheme = "ws" if explicit_scheme in ("ws://", "http://") else "wss"
    elif _looks_like_ip_or_localhost(host) or port == 8000:
        ws_scheme = "ws"
    else:
        ws_scheme = "wss"
    if port is None:
        port = 8000 if ws_scheme == "ws" else 443
    return ws_scheme, host, port


class _UrlBuilder:
    """
    Resolves one server address once, then hands out full URLs for it, so every
    request and websocket in a session agrees on scheme, host and port
    """

    def __init__(self, addr: str) -> None:
        """
        Parse the configured address a single time and keep the matching HTTP
        and websocket schemes beside it

        :param addr: address as configured, with or without scheme and port.
        """
        ws_scheme, host, port = _split_addr(addr)
        self.host = host
        self.port = port
        self.ws_scheme = ws_scheme
        self.http_scheme = "https" if ws_scheme == "wss" else "http"

    def http(self, path: str) -> str:
        """
        Build the full HTTP URL for one server route

        :param path: route path starting with a slash, e.g. /healthz.
        :returns: absolute URL for that route on this server.
        """
        return f"{self.http_scheme}://{self.host}:{self.port}{path}"

    def ws(self, path: str) -> str:
        """
        Build the full websocket URL for one server route

        :param path: route path starting with a slash, e.g. /ws/<room>.
        :returns: absolute websocket URL for that route on this server.
        """
        return f"{self.ws_scheme}://{self.host}:{self.port}{path}"


def _safe_error_reason(response: httpx.Response) -> str | None:
    """
    Dig the server's reason code out of an error response, accepting both the
    plain reason field and the detail wrapper. An error body is reachable by
    anything sitting between client and server, so anything unreadable yields
    no reason at all and the caller falls back to naming the status

    :param response: the failed response, body not yet decoded.
    :returns: the reason code, or None when the body carries none.
    """
    try:
        body = _loads(response.content)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    if "reason" in body:
        return cast(str, body["reason"])
    detail = body.get("detail")
    if isinstance(detail, dict) and "reason" in detail:
        return cast(str, detail["reason"])
    if isinstance(detail, str):
        return detail
    return None


class ServerTransport:
    """
    The client's whole HTTP surface for one server address: matchmaking,
    resume, reclaim, health and opening the game websocket. With
    ServerWebSocket it is the only client-side module that touches httpx or
    websockets, which keeps the network libraries out of the rest of the app
    """

    def __init__(self, addr: str, *, http_client: httpx.Client | None = None,
                 async_http_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        """
        Bind this transport to one server address, resolved up front so an
        unusable address fails here rather than at the first request

        :param addr: server address as configured by the player.
        :param http_client: blocking client to reuse; None means a fresh
            client per request, and tests inject one to observe calls.
        :param async_http_factory: builds the async client a session shares;
            None means the default one wired to the bundled certificates.
        """
        self.addr = addr
        self._url = _UrlBuilder(addr)
        self._http = http_client
        self._async_http_factory = async_http_factory or (
            lambda: httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, verify=_TLS_CONTEXT)
        )

    def _sync_request(self, method: str, path: str, *,
                      json_body: dict[str, Any] | None = None,
                      timeout: float = HTTP_TIMEOUT_SECONDS) -> httpx.Response:
        """
        Make one blocking request to this server. Every way the attempt can
        fail before an answer exists -- refused, timed out, an address that
        cannot even be sent to -- is turned into TransportError, so the worker
        threads these run on never die on a raw network exception

        :param method: HTTP verb, e.g. POST.
        :param path: route path starting with a slash.
        :param json_body: body to send as JSON, or None to send no body.
        :param timeout: per-request timeout in seconds.
        :returns: the response, whatever status it carries.
        """
        url = self._url.http(path)
        try:
            if self._http is not None:
                r = self._http.request(method, url, json=json_body, timeout=timeout)
            else:
                r = httpx.request(method, url, json=json_body, timeout=timeout, verify=_TLS_CONTEXT)
        except (httpx.HTTPError, httpx.TimeoutException, httpx.InvalidURL, UnicodeError) as exc:
            raise TransportError(str(exc)) from exc
        return r

    def _blocking_request(self, method: str, path: str, response_model: type[_ResponseModel],
                          timeout: float,
                          json_body: dict[str, Any] | None = None) -> _ResponseModel | None:
        """
        Make one blocking request and validate the answer into a typed model,
        collapsing every failure into None. Reclaim, resume and health share
        this path, so an unreachable server, a refusal and a 200 carrying
        something that is not the model all read alike: no usable answer

        :param method: HTTP verb, e.g. POST.
        :param path: route path starting with a slash.
        :param response_model: protocol model the body must validate against.
        :param timeout: per-request timeout in seconds.
        :param json_body: body to send as JSON, or None to send no body.
        :returns: the validated model, or None when there was no usable answer.
        """
        try:
            r = self._sync_request(method, path, json_body=json_body, timeout=timeout)
        except TransportError:
            return None
        if r.status_code != 200:
            return None
        try:
            return response_model.model_validate(_loads(r.content))
        except ValueError:
            return None

    def reclaim_blocking(self, client_uuid: str, *,
                         timeout: float = RECLAIM_TIMEOUT_SECONDS) -> ReclaimResponse | None:
        """
        Ask whether this player still has a game waiting for them, knowing only
        their player id. This runs after an app restart, when the previous
        run's session token is gone, and answers None for anything short of a
        clear yes

        :param client_uuid: this installation's stable player id.
        :param timeout: how long to wait before treating the probe as a miss.
        :returns: the room plus a fresh session token, or None when there is
            nothing to rejoin.
        """
        body = ReclaimRequest(client_uuid=client_uuid).model_dump()
        return self._blocking_request("POST", "/reclaim", ReclaimResponse, timeout,
                                      json_body=body)

    def resume_blocking(self, room_id: str, session_token: str) -> ResumeResponse | None:
        """
        Fetch the full state of a game this player is already in, blocking
        until it arrives. The reconnect path uses it to rebuild the game screen
        before any websocket is opened

        :param room_id: room the game is being played in.
        :param session_token: this player's seat in that room.
        :returns: the whole game state, or None when it could not be fetched.
        """
        body = ResumeRequest(
            room_id=room_id, session_token=session_token,
        ).model_dump()
        return self._blocking_request("POST", "/resume", ResumeResponse,
                                      HTTP_TIMEOUT_SECONDS, json_body=body)

    def healthz_blocking(self, *,
                         timeout: float = HEALTHZ_TIMEOUT_SECONDS) -> HealthResponse | None:
        """
        Ask a server for its liveness and load summary, blocking until it
        answers. This is what the Options screen's server test reports, so a
        proxy answering with an error page has to read as no health rather
        than as a healthy server

        :param timeout: how long to wait before treating the server as down.
        :returns: the health summary, or None when none came back.
        """
        return self._blocking_request("GET", "/healthz", HealthResponse, timeout)

    async def healthz_async(self, http: httpx.AsyncClient) -> HealthResponse | None:
        """
        Ask the server for its liveness summary from inside a running session.
        The reconnect loop uses it to tell a server that is down apart from one
        that is up but no longer holding this game

        :param http: async client shared by this session's requests.
        :returns: the health summary, or None when none came back.
        """
        url = self._url.http("/healthz")
        try:
            r = await http.get(url)
        except (httpx.HTTPError, httpx.TimeoutException):
            return None
        if r.status_code != 200:
            return None
        try:
            return HealthResponse.model_validate(_loads(r.content))
        except ValueError:
            return None

    async def matchmake_async(self, req: MatchmakeRequest,
                              http: httpx.AsyncClient) -> MatchmakeResponse:
        """
        Join the matchmaking queue and take back the room and seat the server
        assigned. A protocol mismatch is raised as its own error so the client
        can tell the player to update, any other refusal keeps its status and
        reason for the retry decision, and a server that cannot be reached at
        all is reported as a transport failure the search retries through

        :param req: the matchmaking request, already validated.
        :param http: async client shared by this session's requests.
        :returns: the room id and session token to open the websocket with.
        """
        url = self._url.http("/matchmake")
        try:
            r = await http.post(url, json=req.model_dump())
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise TransportError(str(exc)) from exc
        if r.status_code >= 400:
            reason = _safe_error_reason(r) or f"http_{r.status_code}"
            if reason == Reason.VERSION_MISMATCH:
                raise SchemaVersionMismatch(reason)
            raise TransportHTTPError(r.status_code, reason)
        return MatchmakeResponse.model_validate(_loads(r.content))

    async def cancel_matchmake_async(self, body: CancelMatchmakeRequest,
                                     http: httpx.AsyncClient) -> None:
        """
        Withdraw from the matchmaking queue when the player cancels the search.
        It is sent on the way out, so the answer is not waited on for anything

        :param body: room and session token naming the queued seat.
        :param http: async client shared by this session's requests.
        """
        url = self._url.http("/matchmake")
        await http.request("DELETE", url, json=body.model_dump())

    async def resume_async(self, body: ResumeRequest,
                           http: httpx.AsyncClient) -> ResumeResponse | None:
        """
        Fetch the full game state again while reconnecting. A passing failure
        answers None so the caller keeps trying inside its window, while a room
        that is gone or a seat that is dead raises the fatal error that stops
        the retrying at once

        :param body: room and session token naming the seat to restore.
        :param http: async client shared by this session's requests.
        :returns: the whole game state, or None when the attempt is worth
            repeating.
        """
        url = self._url.http("/resume")
        try:
            r = await http.post(url, json=body.model_dump())
        except (httpx.HTTPError, httpx.TimeoutException):
            return None
        if r.status_code in (401, 410, 404):
            raise FatalResumeError(_safe_error_reason(r) or f"http_{r.status_code}")
        if r.status_code != 200:
            return None
        return ResumeResponse.model_validate(_loads(r.content))

    def make_async_http(self) -> httpx.AsyncClient:
        """
        Open the async HTTP client that one session's requests share, wired to
        the bundled certificates so packaged builds can still verify TLS

        :returns: a fresh async client, meant to be used as a context manager.
        """
        return self._async_http_factory()

    async def ws_connect(self, room_id: str, session_token: str) -> "ServerWebSocket":
        """
        Open the game websocket for a room and send the auth frame that claims
        the seat, which the server demands before any game traffic. TLS is used
        unless the address resolved to a plaintext LAN or development server

        :param room_id: room to connect to.
        :param session_token: this player's seat in that room.
        :returns: the wrapper every later send and receive goes through.
        """
        url = self._url.ws(f"/ws/{room_id}")
        log.info("ws connecting %s", url)
        tls = _TLS_CONTEXT if self._url.ws_scheme == "wss" else None
        ws = await websockets.connect(url, ssl=tls, ping_interval=WS_PING_INTERVAL_SECONDS,
                                      ping_timeout=WS_PING_TIMEOUT_SECONDS)
        await ws.send(AuthMessage(session_token=session_token).model_dump_json())
        return ServerWebSocket(ws)


class ServerWebSocket:
    """
    The live game connection: one typed send method per action a player can
    take during a game, plus a receive that hands back decoded server frames.
    Wrapping the raw socket keeps the protocol models and the websockets
    library on this side of the boundary, away from the rest of the client
    """

    def __init__(self, ws: Any) -> None:
        """
        Wrap a socket that is already connected and has already authenticated

        :param ws: the open websocket connection this wrapper sends on.
        """
        self._ws = ws

    async def close(self) -> None:
        """
        Close the connection, tolerating a socket that is already gone. Closing
        only ever happens while tearing something down, so it never raises
        """
        try:
            await self._ws.close()
        except Exception:
            log.debug("ws close failed", exc_info=True)

    async def send_move(self, from_sq: str, to_sq: str, promotion: str | None = None) -> None:
        """
        Offer a move to the server, which alone decides whether it lands. The
        local board advances only when the move comes back applied

        :param from_sq: origin square in algebraic form, e.g. e2.
        :param to_sq: destination square in algebraic form.
        :param promotion: piece letter q, r, b or n when a pawn promotes,
            otherwise None.
        """
        msg = MoveMessage(
            **{"from": from_sq, "to": to_sq}, promotion=promotion,
        )
        await self._send(msg)

    async def send_resign(self) -> None:
        """
        Give up the game, which ends it for both players immediately
        """
        await self._send_raw({"type": "resign", "version": PROTOCOL_VERSION})

    async def send_draw_offer(self) -> None:
        """
        Offer the opponent a draw, which they then accept or decline
        """
        await self._send_raw({"type": "draw_offer", "version": PROTOCOL_VERSION})

    async def send_draw_response(self, accept: bool) -> None:
        """
        Answer the opponent's draw offer

        :param accept: True ends the game as a draw, False clears the offer.
        """
        await self._send(DrawResponseMessage(accept=accept))

    async def send_rematch_request(self) -> None:
        """
        Offer another game in the same room once this one has finished
        """
        await self._send(RematchRequestMessage())

    async def send_rematch_response(self, accept: bool) -> None:
        """
        Answer the opponent's rematch offer

        :param accept: True restarts the room with colours swapped, False
            closes the rematch out.
        """
        await self._send(RematchResponseMessage(accept=accept))

    async def send_left_result(self) -> None:
        """
        Say that this player has walked away from the finished game, which
        withdraws any rematch they had offered and lets the room be cleaned up
        once both players have gone
        """
        await self._send_raw({"type": "left_result", "version": PROTOCOL_VERSION})

    async def send_takeback_request(self) -> None:
        """
        Ask the opponent to let the last move be retracted
        """
        await self._send_raw({"type": "takeback_request", "version": PROTOCOL_VERSION})

    async def send_takeback_response(self, accept: bool) -> None:
        """
        Answer the opponent's takeback request

        :param accept: True retracts the move for both sides, False leaves the
            position alone.
        """
        await self._send(TakebackResponseMessage(accept=accept))

    async def send_give_time(self, hold_ms: int) -> None:
        """
        Hand the opponent some of the clock, sized by how long the give-time
        button was held. The server turns the hold into whole seconds and is
        the one that actually moves the time

        :param hold_ms: how long the button was held, in milliseconds.
        """
        await self._send(GiveTimeMessage(hold_ms=hold_ms))

    async def send_ping(self, ply: int | None) -> None:
        """
        Send the heartbeat that keeps the connection alive and reports which
        ply this client thinks it is on, which is how the server notices a
        board that has fallen behind and repairs it

        :param ply: number of half-moves this client has applied, or None while
            it is not sitting on a live online board and has no ply to claim.
        """
        await self._send(PingMessage(ply=ply))

    async def send_skill_check_shot(self, client_elapsed_ms: float,
                                    direction: str | None = None,
                                    target_row: float | None = None,
                                    target_col: float | None = None) -> None:
        """
        Report one input the player made during a live skill check. Only the
        input itself travels -- a direction, or a point in board space -- and
        the server judges it against the geometry it derived for the check

        :param client_elapsed_ms: milliseconds into the check when the input
            was made, as this client measured it.
        :param direction: prompt answered during a combo check, else None.
        :param target_row: aim point row in board space, else None.
        :param target_col: aim point column in board space, else None.
        """
        await self._send(SkillCheckShotMessage(
            client_elapsed_ms=client_elapsed_ms, direction=direction,
            target_row=target_row, target_col=target_col))

    async def send_annotations_state(self, sharing: bool, highlights: Iterable[str],
                                     arrows: Iterable[tuple[str, str]]) -> None:
        """
        Publish this player's whole set of shared board marks at once, which is
        what happens when sharing is switched on or off and whenever the set
        has to be replaced wholesale rather than nudged

        :param sharing: whether this player is sharing marks at all.
        :param highlights: highlighted squares in algebraic form.
        :param arrows: arrows as (origin, destination) square pairs.
        """
        msg = AnnotationsStateMessage(
            sharing=sharing,
            highlights=list(highlights),
            arrows=[ArrowWire(from_sq=from_sq, to_sq=to_sq) for from_sq, to_sq in arrows],
        )
        await self._send(msg)

    async def send_annotation_delta(self, action: str, kind: str, square: str | None = None,
                                    from_sq: str | None = None,
                                    to_sq: str | None = None) -> None:
        """
        Publish a single change to this player's shared marks, which keeps live
        drawing cheap next to resending the whole set on every stroke

        :param action: add or remove.
        :param kind: highlight or arrow.
        :param square: the square involved, for a highlight change.
        :param from_sq: arrow origin square, for an arrow change.
        :param to_sq: arrow destination square, for an arrow change.
        """
        msg = AnnotationDeltaMessage(
            action=action, kind=kind, square=square, from_sq=from_sq, to_sq=to_sq,
        )
        await self._send(msg)

    async def send_quick_chat(self, preset: int) -> None:
        """
        Send one phrase from the fixed quick-chat list; free text never travels
        between players

        :param preset: index of the phrase in the quick-chat list.
        """
        await self._send(QuickChatMessage(preset=preset))

    async def send_set_marks_visibility(self, hide_opp: bool) -> None:
        """
        Say whether this player wants to see the opponent's shared marks. While
        they are hidden the server does not deliver them at all

        :param hide_opp: True to stop receiving the opponent's marks.
        """
        await self._send(SetMarksVisibilityMessage(hide_opp=hide_opp))

    async def _send(self, message: BaseModel) -> None:
        """
        Put one protocol model on the wire, serialised under its wire field
        names so that squares travel as from and to rather than the Python
        spellings the server would reject

        :param message: the protocol model to send.
        """
        await self._ws.send(message.model_dump_json(by_alias=True))

    async def _send_raw(self, payload: dict[str, Any]) -> None:
        """
        Put a hand-built frame on the wire, used by the messages that carry
        nothing beyond their type and the protocol version

        :param payload: the complete frame, type and version included.
        """
        await self._ws.send(json.dumps(payload))

    async def recv(self) -> dict[str, Any] | None:
        """
        Wait for the next frame from the server and decode it. A frame that is
        not readable JSON, or that smuggles in a non-standard constant, is
        dropped rather than handed on to the game

        :returns: the decoded frame, or None when it was discarded.
        """
        raw = await self._ws.recv()
        try:
            return cast("dict[str, Any] | None", _loads(raw))
        except ValueError:
            return None
