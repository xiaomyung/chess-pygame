import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any, cast

import pygame as pg

from chessshootout.backend.utils import Square, coord_from_square, square_from_coord
from chessshootout.domain.pgn.load import time_category_for_minutes
from chessshootout.frontend.modals.match_found import MatchFoundModal
from chessshootout.frontend.modals.reconnecting import ReconnectingModal
from chessshootout.frontend.modals.wait import WaitModal
from chessshootout.frontend.panels.banners import OfferBanners
from chessshootout.frontend.panels.player_strip import AUTO_END_RED_THRESHOLD_SECONDS
from chessshootout.frontend.game.variant import Variant
from chessshootout.frontend.screens.base import Nav
from chessshootout.infra import env
from chessshootout.online.client import (
    ClientReason, Event, OnlineClient, RECONNECT_TOTAL_SECONDS, fetch_resume,
    probe_active_game,
)
from chessshootout.server.protocol import (
    FIRST_MOVE_ABORT_SECONDS, GRACE_SECONDS, Reason,
)
from chessshootout.skillcheck.wheel import SKILLCHECK_DEADLINE_MS


log = logging.getLogger("chess.frontend")

MATCH_FOUND_SECONDS = 3
REMATCH_STATE_TOAST_KEY = "rematch_state"

RESYNC_TIMEOUT_MS = 8000
SKILLCHECK_WATCHDOG_SLACK_MS = 4000
RECONNECT_MODAL_DEBOUNCE_MS = 500

TOAST_REASON_MAX_CHARS = 80
APPLY_FAILED_LABEL = "Couldn't apply the server update"
APPLY_FAILED_TOAST_KEY = "online_apply_failed"

ONLINE_DEFAULT_TIME_MINUTES = 5

RECONNECT_PROBE_INTERVAL_MS = 5000
RECONNECT_PROBE_MAX_ATTEMPTS = 3

OFFER_BANNERS = {
    "draw_offered": ("🤝", "offers a draw", "Accept", "Decline", "send_draw_response"),
    "takeback_offered": ("↩️", "wants a takeback", "Allow", "Deny",
                         "send_takeback_response"),
    "rematch_request": ("⚔️", "wants a rematch", "Accept", "Deny",
                        "send_rematch_response"),
}

MOVE_INVALIDATED_OFFER_KEYS = ("draw_offered", "takeback_offered")

ONLINE_HARD_FAILURE_LABELS = {
    ClientReason.SERVER_UNREACHABLE: "Server unreachable",
    ClientReason.RECONNECT_FAILED: "Could not reconnect",
    Reason.ROOM_FULL: "Server is full",
    Reason.INVALID_FIELD: "Request rejected by server",
}

ONLINE_HARD_FAILURE_REASONS = frozenset(ONLINE_HARD_FAILURE_LABELS)

ONLINE_TRANSIENT_REASON_LABELS = {
    Reason.QUEUE_TIMEOUT: "Matchmaking timed out — try again",
    Reason.RATE_LIMITED: "Slow down a bit",
    Reason.NO_TAKEBACK_AVAILABLE: "Nothing to take back",
    Reason.REMATCH_ALREADY_PENDING: "Rematch already requested",
    Reason.ALREADY_IN_GAME: "Already in a game",
    Reason.GAME_ALREADY_OVER: "This game has ended",
    Reason.SHARE_MUTED: "Mark sharing is muted for this game",
    Reason.OPP_HIDES_MARKS: "Opponent hides shared marks",
}

ONLINE_GAME_STATE_REASONS = {
    Reason.NOT_YOUR_TURN, Reason.INVALID_MOVE_FORMAT, Reason.INVALID_MESSAGE,
    Reason.VERSION_MISMATCH,
}

MOVE_REJECTION_REASONS = {
    Reason.INVALID_MOVE_FORMAT, Reason.NOT_YOUR_TURN, Reason.SKILLCHECK_PENDING,
    Reason.MOVE_LOCKED,
}

NOT_YOUR_TURN_TOASTS = {
    "takeback_request": "Take back is only available right after your move",
}


class OnlineCoordinator:
    """
    The client side of an online match, from the search for an opponent all the
    way to the rematch window after the result. It owns the server connection
    and the waiting, match-found and reconnecting cards, turns every inbound
    server message into a call on the game screen, and keeps the menu's
    Reconnect button honest
    """

    def __init__(self, app: Any) -> None:
        """
        Build the coordinator once at startup, long before any online game
        exists. Nothing connects here: a client is created only when a search
        or a reconnect actually starts

        :param app: the Frontend shell, this coordinator's only route to the
            window, the menu, the game screen, toasts and sound
        """
        self.app = app
        self.client: OnlineClient | None = None
        self.wait_modal = WaitModal(app.window)
        self.match_found_modal = MatchFoundModal(app.window)
        self.reconnecting_modal = ReconnectingModal(app.window)
        self.offer_banners = OfferBanners(app.window)
        self._subscriber: Any = None

        self._online_config: dict[str, Any] | None = None
        self._resyncing = False
        self._resync_buffer: list[tuple[str, dict[str, Any]]] = []
        self._resync_started_at_ms = 0
        self._last_heartbeat_sent_ms = 0
        self._wait_started_at_ms: int | None = None
        self._match_found_at_ms: int | None = None
        self._match_found_started_seconds_ago = 0.0
        self._pending_game_start_payload: dict[str, Any] | None = None
        self._rematch_offered = False
        self._prev_online_state: str | None = None

        self._pending_reconnect: dict[str, Any] | None = None
        self._pending_reconnect_lock = threading.Lock()
        self._last_reconnect_probe_ms = 0
        self._reconnect_probe_inflight = False
        self._reconnect_probe_gen = 0
        self._reconnect_probe_attempts = 0

    def subscribe(self, subscriber: Any) -> None:
        """
        Register the screen that should receive board-level online events, done
        by the game screen as it enters an online match. Exactly one subscriber
        may be registered at a time

        :param subscriber: the active game screen, which implements the
            on_remote_move, on_result, on_takeback and sibling callbacks
        """
        assert self._subscriber is None, "OnlineCoordinator already has a subscriber"
        self._subscriber = subscriber

    def unsubscribe(self, subscriber: Any) -> None:
        """
        Drop the registered screen as it exits, so events raised afterwards
        stop reaching it. Handing over a screen that is not the current
        subscriber does nothing, which makes a double exit safe

        :param subscriber: the screen that is standing down
        """
        if self._subscriber is subscriber:
            self._subscriber = None

    def _forward_board_event(self, method_name: str, payload: dict[str, Any]) -> None:
        """
        Hand a board-level event -- a move, a takeback, a skill check -- to the
        subscribed game screen. Such events only mean anything on a live board,
        so arriving with nobody subscribed is impossible by design and logged

        :param method_name: subscriber method to call, such as on_remote_move
        :param payload: decoded server message for that method
        """
        subscriber = self._subscriber
        if subscriber is None:
            log.error("board-level event %s arrived with no subscriber", method_name)
            return
        getattr(subscriber, method_name)(payload)

    def _forward_screen_event(self, method_name: str, payload: dict[str, Any]) -> None:
        """
        Hand a screen-level event -- result, offer, connection status, idle
        window -- to the subscribed screen, or to the game screen object itself
        when nothing is subscribed, so a result still lands after the player
        walked back to the menu

        :param method_name: subscriber method to call, such as on_result
        :param payload: decoded server message for that method
        """
        target = self._subscriber if self._subscriber is not None else self.app.game
        getattr(target, method_name)(payload)

    def is_connected(self) -> bool:
        """
        Say whether an online session object exists at all, the check every
        online-only branch in the UI makes before sending anything. True while
        searching and while reconnecting, not only during a live game

        :returns: True when an online client is attached
        """
        return self.client is not None

    def opponent_state(self) -> str | None:
        """
        Report how the opponent's own connection is doing, which the player
        strip shows as a disconnect countdown or a resyncing badge

        :returns: connected, reconnecting or resyncing, or None when this
            client has no session
        """
        return self.client.opp_state if self.client is not None else None

    def ping_ms(self) -> int | None:
        """
        Report the smoothed round trip to the server, shown in the optional
        PING readout in the corner of the window

        :returns: average round trip in milliseconds, or None before the first
            answered ping of the session
        """
        return self.client.get_ping_ms() if self.client is not None else None

    def send_local_move(self, from_sq: Square, to_sq: Square,
                        promotion: str | None) -> None:
        """
        Report a move the player just made on their own board to the server,
        translating engine squares into algebraic coordinates on the way. Any
        offer the move makes stale is taken off screen at the same moment

        :param from_sq: square the piece left
        :param to_sq: square the piece arrived on
        :param promotion: promotion piece letter (q, r, b, n), or None when the
            move is not a promotion
        """
        if self.client is not None:
            self.client.send_move(coord_from_square(from_sq), coord_from_square(to_sq), promotion)
            self._dismiss_move_invalidated_offers()

    def send_move(self, from_coord: str, to_coord: str, promo_letter: str | None) -> None:
        """
        Send a move that still has to win a skill check before it lands, handed
        over in algebraic coordinates by the caller. The server decides its
        fate, so nothing changes on the local board here

        :param from_coord: square the piece leaves, spelled as e2
        :param to_coord: square the piece is aimed at, spelled as e4
        :param promo_letter: promotion piece letter, or None when not promoting
        """
        if self.client is not None:
            self.client.send_move(from_coord, to_coord, promo_letter)

    def send_resign(self) -> bool:
        """
        Tell the server the player gives up, sent from the resign confirmation
        on the game screen. The result itself still comes back from the server,
        never from here

        :returns: True when there was a session to send it on
        """
        if self.client is None:
            return False
        self.client.send_resign()
        return True

    def send_draw_offer(self) -> bool:
        """
        Offer the opponent a draw; they answer with a banner of their own and
        the server reports whatever they choose

        :returns: True when there was a session to send it on
        """
        if self.client is None:
            return False
        log.info("draw offer sent")
        self.client.send_draw_offer()
        return True

    def send_takeback_request(self) -> bool:
        """
        Ask the opponent to let the player unmake their last move, which the
        server only entertains right after that move

        :returns: True when there was a session to send it on
        """
        if self.client is None:
            return False
        log.info("takeback requested")
        self.client.send_takeback_request()
        return True

    def send_give_time(self, hold_ms: int) -> None:
        """
        Forward how long the player held the give-time button, which the server
        turns into whole chunks of clock time for the opponent. Online the
        server owns both clocks, so nothing local is added here

        :param hold_ms: length of the button hold in milliseconds
        """
        if self.client is not None:
            self.client.send_give_time(hold_ms)

    def send_skill_check_shot(self, client_elapsed_ms: float, direction: str | None = None,
                              target_row: float | None = None,
                              target_col: float | None = None) -> None:
        """
        Send one input the player made during a live skill check: a direction
        for the combo prompts, or an aim point in board space for whack-a-mole.
        Only timing and aim travel -- the server owns all of the geometry and
        judges every shot itself

        :param client_elapsed_ms: how far into the check the input happened, in
            milliseconds measured on this machine
        :param direction: combo prompt direction, or None for other kinds
        :param target_row: aimed row in board space, or None when not aiming
        :param target_col: aimed column in board space, or None when not aiming
        """
        if self.client is not None:
            self.client.send_skill_check_shot(client_elapsed_ms, direction=direction,
                                              target_row=target_row, target_col=target_col)

    def send_annotations_state(self, sharing: bool, highlights: list[str],
                               arrows: list[tuple[str, str]]) -> None:
        """
        Publish the player's whole set of board marks at once, sent when mark
        sharing is switched on or off and whenever the opponent's view has to
        be rebuilt from scratch

        :param sharing: whether this player is sharing marks at all
        :param highlights: highlighted squares as algebraic names
        :param arrows: arrows as (from, to) pairs of algebraic names
        """
        if self.client is not None:
            self.client.send_annotations_state(sharing, highlights, arrows)

    def send_annotation_delta(self, action: str, kind: str, square: str | None = None,
                              from_sq: str | None = None,
                              to_sq: str | None = None) -> None:
        """
        Publish the single mark the player just drew or erased, which keeps
        live drawing cheap next to resending the whole set on every stroke

        :param action: add or remove
        :param kind: highlight or arrow
        :param square: highlighted square as an algebraic name, for highlights
        :param from_sq: square an arrow starts on, for arrows
        :param to_sq: square an arrow points at, for arrows
        """
        if self.client is not None:
            self.client.send_annotation_delta(action, kind, square, from_sq, to_sq)

    def send_quick_chat(self, preset: int) -> None:
        """
        Send the player's pick from the fixed quick-chat phrase list. Only the
        index of the phrase travels between players, never free text

        :param preset: index into the shared quick-chat phrase list
        """
        if self.client is not None:
            self.client.send_quick_chat(preset)

    def set_marks_visibility(self, hide_opp: bool) -> None:
        """
        Tell the server whether this player wants to see the opponent's shared
        marks, mirroring the Options toggle. While they are hidden the server
        stops delivering them to this client at all

        :param hide_opp: True to stop receiving the opponent's marks
        """
        if self.client is not None:
            self.client.send_set_marks_visibility(hide_opp)

    def _guarded_apply(self, what: str, apply: Callable[[], None]) -> bool:
        """
        Run one piece of work that acts on untrusted server data -- an inbound
        event, a reconnect adoption -- so a bad payload can never take the game
        down. A failure is logged and shown to the player as a single toast

        :param what: short label for the log line, such as event handler
        :param apply: the work to run, called with no arguments
        :returns: True when it ran cleanly, False when it raised
        """
        try:
            apply()
            return True
        except Exception:
            log.exception("online %s failed", what)
            self.app.toast.show(APPLY_FAILED_LABEL, key=APPLY_FAILED_TOAST_KEY)
            return False

    def _drain_online_inbound(self) -> None:
        """
        Take everything the network thread has queued since the last frame and
        apply it, one guarded call per event. It runs first in the coordinator's
        frame, so the rest of the frame already sees the new state
        """
        if self.client is None:
            return
        for event in self.client.drain_inbound():
            self._guarded_apply("event handler", lambda: self._handle_online_event(event))

    def _handle_online_event(self, event: Event) -> None:
        """
        Route one decoded server message to the handler that owns it -- the
        single fan-out point for everything the server can say. Message types
        this client does not know are ignored on purpose

        :param event: queued event carrying a message type and its payload
        """
        if event.type == "game_start":
            self._begin_match_found_transition(event.payload)
        elif event.type == "move_applied":
            self._handle_remote_move_applied(event.payload)
        elif event.type == "result":
            self._handle_online_result(event.payload)
        elif event.type in ("draw_offered", "takeback_offered"):
            self._push_offer_banner(event.type)
        elif event.type == "rematch_request":
            self._handle_rematch_request()
        elif event.type == "rematch_update":
            self._handle_rematch_update(event.payload)
        elif event.type == "takeback_applied":
            self._handle_takeback_applied(event.payload)
        elif event.type == "game_resumed":
            self._handle_game_resumed(event.payload)
        elif event.type == "time_granted":
            self._handle_time_granted(event.payload)
        elif event.type == "annotations_state":
            self._handle_annotations_state(event.payload)
        elif event.type == "annotation_delta":
            self._handle_annotation_delta(event.payload)
        elif event.type == "annotations_blocked":
            self._handle_annotations_blocked(event.payload)
        elif event.type == "quick_chat_received":
            self._forward_board_event("on_quick_chat", event.payload)
        elif event.type == "connection_status":
            self._handle_connection_status(event.payload)
        elif event.type == "idle_window":
            self._handle_idle_window(event.payload)
        elif event.type == "resync_directive":
            self._begin_resync()
        elif event.type == "skill_check_required":
            self._handle_skill_check_required(event.payload)
        elif event.type == "skill_check_result":
            self._handle_skill_check_result(event.payload)
        elif event.type == "skill_check_spectate":
            self._handle_skill_check_spectate(event.payload)
        elif event.type == "skill_check_spectate_shot":
            self._handle_skill_check_spectate_shot(event.payload)
        elif event.type == "error":
            self._handle_online_error(event.payload)

    def _handle_online_error(self, payload: dict[str, Any]) -> None:
        """
        Decide what a rejection or failure from the server should look like to
        the player: silence for ordinary game-state answers, a toast for
        transient ones, a confirm dialog with Retry for the hard failures. A
        reason string the server supplied is truncated before it reaches a toast

        :param payload: error message, keyed by reason plus the message type it
            is answering
        """
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        game = self.app.game
        pending_move = game.skillcheck_session.pending_online_move
        if pending_move is not None and reason in MOVE_REJECTION_REASONS:
            game.skillcheck_session.pending_online_move = None
            game.board.selected_square = None
            if reason in (Reason.INVALID_MOVE_FORMAT, Reason.NOT_YOUR_TURN):
                self._begin_resync()
            return
        if reason == Reason.NOT_YOUR_TURN:
            label = NOT_YOUR_TURN_TOASTS.get(cast(str, payload.get("msg_type")))
            if label is not None:
                self.app.toast.show(label)
            return
        if reason in ONLINE_GAME_STATE_REASONS:
            return
        if reason == Reason.REMATCH_UNAVAILABLE:
            self.app.toast.show("Rematch no longer available", key=REMATCH_STATE_TOAST_KEY)
            self._end_rematch_window()
            return
        if reason == ClientReason.ROOM_LOST:
            log.warning("online room lost — server restarted mid-game")
            self._end_resync(replay=False)
            game.result_flow.auto_save_pgn()
            self.reconnecting_modal.hide()
            self.offer_banners.clear()
            self.app.confirm_modal.show(
                "Server restarted — game ended",
                on_yes=self._restart_online_search,
                on_no=self._abandon_online_game,
                yes_label="New Search", no_label="Cancel",
            )
            return
        if reason == Reason.QUEUE_TIMEOUT:
            log.warning("matchmaking queue timed out")
            self._on_online_cancel()
            self.app.toast.show(ONLINE_TRANSIENT_REASON_LABELS[reason])
            return
        if reason in ONLINE_HARD_FAILURE_REASONS or reason.startswith("http_"):
            log.warning("online hard failure reason=%s", reason)
            self._end_resync(replay=False)
            self.wait_modal.hide()
            self.match_found_modal.hide()
            self.offer_banners.clear()
            label = ONLINE_HARD_FAILURE_LABELS.get(
                reason, ONLINE_HARD_FAILURE_LABELS[ClientReason.SERVER_UNREACHABLE])
            self.app.confirm_modal.show(
                label,
                on_yes=self._restart_online_search,
                on_no=self._on_online_cancel,
                yes_label="New Search", no_label="Cancel",
            )
            return
        if reason:
            label = ONLINE_TRANSIENT_REASON_LABELS.get(
                reason, reason[:TOAST_REASON_MAX_CHARS])
            self.app.toast.show(label)
        else:
            self.app.toast.show("Server error")

    def _opp_name(self) -> str:
        """
        Name the other player the way banners and toasts address them. It is
        read off the game screen, so it follows whichever side this client got

        :returns: the opponent's display nickname
        """
        game = self.app.game
        opp_color = "black" if game._chosen_side == "white" else "white"
        return cast(str, game._name_for_color(opp_color))

    def _push_rematch_banner(self) -> None:
        """
        Put the opponent's rematch offer on screen as a banner with Accept and
        Deny. The whole post-game rematch window belongs to the coordinator,
        this banner included
        """
        self._rematch_offered = True
        icon, verb, yes_label, no_label, _ = OFFER_BANNERS["rematch_request"]
        self.offer_banners.push(
            "rematch_request", icon, self._opp_name(), verb, yes_label, no_label,
            on_yes=self._accept_rematch, on_no=self._decline_rematch)

    def _clear_rematch_offer(self) -> None:
        """
        Take a rematch offer down everywhere it shows -- the banner and the
        result menu's lit Rematch button -- once it has been answered, withdrawn
        or has run out of time
        """
        game = self.app.game
        self.offer_banners.dismiss("rematch_request")
        self._rematch_offered = False
        game.result_menu.set_rematch_offered(False)

    def _handle_rematch_request(self) -> None:
        """
        Greet an incoming rematch offer: a banner over the board, the result
        menu's Rematch button lit up and a chime. Nothing happens once the
        session has already been dropped
        """
        if self.client is None:
            return
        log.info("rematch offer received")
        self._push_rematch_banner()
        self.app.game.on_offer("rematch_request")
        self.app.sound_manager.play_give_time()

    def _reshow_rematch_banner(self) -> None:
        """
        Bring a still-standing rematch offer back on screen after the player
        walked from the finished board to the menu, so leaving the board does
        not quietly throw the offer away
        """
        if self.client is None:
            return
        self._push_rematch_banner()

    def _accept_rematch(self) -> None:
        """
        Say yes to the opponent's rematch from the banner. The server answers
        with a fresh game start, so no new game is set up here
        """
        if self.client is not None:
            log.info("rematch response sent accepted=True")
            self.client.send_rematch_response(True)

    def _decline_rematch(self) -> None:
        """
        Say no to the opponent's rematch from the banner and clear the offer
        locally at once, without waiting for the server to confirm
        """
        self._clear_rematch_offer()
        if self.client is not None:
            log.info("rematch response sent accepted=False")
            self.client.send_rematch_response(False)

    def _end_rematch_window(self) -> None:
        """
        Close the post-game rematch window for good: drop the offer, save the
        game and tear the session down. The player stays on the finished board
        while that is still what they are looking at, otherwise off to the menu
        """
        self._clear_rematch_offer()
        game = self.app.game
        stay_on_result = (self.app.screen is game
                          and game.current_result() is not None)
        self._tear_down_online_session("rematch_window_closed",
                                       navigate=not stay_on_result)
        if not stay_on_result:
            self._return_to_menu_card()

    def _handle_rematch_update(self, payload: dict[str, Any]) -> None:
        """
        Narrate what is happening to a pending rematch -- the opponent dropped,
        came back, withdrew, declined or left -- through one toast that keeps
        replacing itself. Anything final also closes the rematch window

        :param payload: rematch update message, keyed by event
        """
        event = payload.get("event", "")
        log.info("rematch update event=%s", event)
        opp = self._opp_name()
        if event == "opponent_reconnecting":
            self.app.toast.show(f"{opp} disconnected — waiting…", key=REMATCH_STATE_TOAST_KEY)
            return
        if event == "opponent_returned":
            self.app.toast.show(f"{opp} is back", key=REMATCH_STATE_TOAST_KEY)
            return
        if event == "cancelled":
            self._clear_rematch_offer()
            self.app.toast.show(f"{opp} withdrew the rematch", key=REMATCH_STATE_TOAST_KEY)
            return
        labels = {
            "declined": f"{opp} declined the rematch",
            "opponent_left": f"{opp} left the game",
            "window_expired": "Rematch window closed",
        }
        self.app.toast.show(labels.get(event, "Rematch ended"), key=REMATCH_STATE_TOAST_KEY)
        self._end_rematch_window()

    def _push_offer_banner(self, event_type: str) -> None:
        """
        Show an incoming draw or takeback offer as a banner over the board,
        with its two buttons wired straight to the matching answer on the
        client, and chime so the player notices it arrive

        :param event_type: draw_offered or takeback_offered
        """
        if self.client is None:
            return
        icon, verb, yes_label, no_label, send_method = OFFER_BANNERS[event_type]
        opp_name = self._opp_name()
        send_response = getattr(self.client, send_method)
        log.info("offer received type=%s from=%s", event_type, opp_name)
        self.app.game.on_offer(event_type)
        self.app.sound_manager.play_toast()

        def respond(value: bool) -> Callable[[], None]:
            """
            Build the click handler for one of the banner's two buttons, so
            both answers travel the same path

            :param value: True for the accepting button, False for the
                declining one
            :returns: callback the banner runs when that button is clicked
            """
            def fire() -> None:
                """
                Send this button's answer to the opponent when the player
                actually presses it, with a click for feedback
                """
                log.info("offer responded type=%s accepted=%s", event_type, value)
                self.app.sound_manager.play_toast()
                send_response(value)
            return fire

        self.offer_banners.push(
            event_type, icon, opp_name, verb, yes_label, no_label,
            on_yes=respond(True), on_no=respond(False),
        )

    def _begin_resync(self) -> None:
        """
        Start rebuilding the whole game from the server after a suspected
        desync, asking for a fresh snapshot and holding board events back until
        it lands. Asking again while one is already running does nothing
        """
        if self._resyncing:
            return
        self._resyncing = True
        self.offer_banners.clear()
        self._resync_started_at_ms = pg.time.get_ticks()
        if self.client is not None:
            self.client.request_state_sync()

    def _end_resync(self, *, replay: bool = True) -> None:
        """
        Finish a resync and let the game screen see events again, replaying the
        few that were buffered while the snapshot was in flight. Teardowns turn
        the replay off, since there is no board left to replay onto

        :param replay: False to throw the buffered events away instead
        """
        self._resyncing = False
        buffered = self._resync_buffer
        self._resync_buffer = []
        if not replay or self._subscriber is None:
            return
        for method_name, payload in buffered:
            self._forward_board_event(method_name, payload)

    def _handle_game_resumed(self, payload: dict[str, Any]) -> None:
        """
        Adopt a full state snapshot from the server, which is how both a
        reconnect and a resync end. The game screen rebuilds itself from it,
        and the hide-marks preference is re-sent whenever the server disagrees

        :param payload: resume snapshot: move history, clocks, shared marks,
            any pending skill check and the idle window
        """
        game = self.app.game
        if game.variant != Variant.ONLINE:
            log.info("resume ignored — no active online game")
            self._end_resync(replay=False)
            return
        desired = env.get_hide_opp_marks()
        if bool(payload.get("hide_opp_marks")) != desired:
            self.set_marks_visibility(desired)
        self._forward_screen_event("on_resume", payload)
        self._end_resync()

    def _handle_time_granted(self, payload: dict[str, Any]) -> None:
        """
        Pass on the server's word that clock time was given away, so the board
        can snap both clocks and toast who gave how much to whom

        :param payload: time-granted message: the giver, the seconds added and
            the resulting clocks
        """
        self._forward_board_event("on_give_time", payload)

    def _handle_takeback_applied(self, payload: dict[str, Any]) -> None:
        """
        Pass on a takeback the server accepted, which unwinds the last ply on
        the board. Dropped while a resync is in flight, because the snapshot on
        its way already accounts for it

        :param payload: takeback message: resulting ply and clock snapshot
        """
        if self._resyncing:
            return
        self._forward_board_event("on_takeback", payload)

    def _handle_annotations_state(self, payload: dict[str, Any]) -> None:
        """
        Pass on the opponent's whole set of shared marks so the board can
        redraw them. Dropped mid-resync, since the snapshot carries the marks

        :param payload: sharing flag plus the opponent's highlights and arrows
        """
        if self._resyncing:
            return
        self._forward_board_event("on_annotations_state", payload)

    def _handle_annotation_delta(self, payload: dict[str, Any]) -> None:
        """
        Pass on the single mark the opponent just drew or erased, the cheap
        live counterpart to a whole-state update. Dropped mid-resync

        :param payload: delta message: add or remove, highlight or arrow, and
            the squares involved
        """
        if self._resyncing:
            return
        self._forward_board_event("on_annotation_delta", payload)

    def _handle_annotations_blocked(self, payload: dict[str, Any]) -> None:
        """
        Pass on moderation's verdict that some of this player's own marks were
        not shared. This one is buffered rather than dropped during a resync,
        because a snapshot does not carry it and the player has to hear it

        :param payload: blocked message: the offending marks and whether
            sharing was muted for the rest of the game
        """
        if self._resyncing:
            self._resync_buffer.append(("on_annotations_blocked", payload))
            return
        self._forward_board_event("on_annotations_blocked", payload)

    def _handle_skill_check_required(self, payload: dict[str, Any]) -> None:
        """
        Open the timed shootout challenge the server has armed for this
        player's capture or promotion. Dropped mid-resync, because the snapshot
        restores any pending check by itself

        :param payload: skill-check message: kind, seed, deadline and the move
            the check is guarding
        """
        if self._resyncing:
            return
        self._forward_board_event("on_skillcheck_required", payload)

    def _handle_skill_check_result(self, payload: dict[str, Any]) -> None:
        """
        Deliver the server's verdict on a finished skill check to whoever is
        watching it: the mover sees their own miss, the opponent sees it land
        in the read-only mirror they already have open

        :param payload: verdict message carrying the move squares the check
            belongs to
        """
        if self._resyncing:
            return
        game = self.app.game
        from_sq = square_from_coord(payload["from"])
        to_sq = square_from_coord(payload["to"])
        if game._is_my_open_check(from_sq, to_sq):
            self._forward_board_event("on_skillcheck_required", payload)
        elif game.skillcheck_session.online_spectate_kind is not None:
            self._forward_board_event("on_spectate", payload)

    def _handle_skill_check_spectate(self, payload: dict[str, Any]) -> None:
        """
        Open the read-only mirror of the opponent's skill check, so the player
        watches the shot being lined up instead of staring at a frozen board.
        Dropped mid-resync

        :param payload: spectate message: kind, seed and the move it guards
        """
        if self._resyncing:
            return
        self._forward_board_event("on_spectate", payload)

    def _handle_skill_check_spectate_shot(self, payload: dict[str, Any]) -> None:
        """
        Play one shot the opponent fired into the mirror that is already open.
        Ignored when no mirror is showing, which is what a shot arriving after
        the overlay closed looks like

        :param payload: shot message: elapsed time, misses, progress and the
            aim point or direction
        """
        if self._resyncing or self.app.game.skillcheck_session.online_spectate_kind is None:
            return
        self._forward_board_event("on_spectate", payload)

    def _dismiss_move_invalidated_offers(self) -> None:
        """
        Take down the draw and takeback banners a move has just made stale, so
        the player cannot answer an offer that no longer means anything
        """
        for key in MOVE_INVALIDATED_OFFER_KEYS:
            self.offer_banners.dismiss(key)

    def _handle_remote_move_applied(self, payload: dict[str, Any]) -> None:
        """
        Deliver the opponent's move to the board and clear the offers it
        invalidates. Because the coordinator updates before the active screen
        does, a premove waiting for this reply still fires in the same frame

        :param payload: move message: squares, SAN, resulting ply and clocks
        """
        if self._resyncing:
            return
        self._dismiss_move_invalidated_offers()
        self._forward_board_event("on_remote_move", payload)

    def _handle_online_result(self, payload: dict[str, Any]) -> None:
        """
        Deliver the server's final word on the game, the only place an online
        result ever comes from. It reaches the game screen even with nothing
        subscribed, so a game that ended after the player left is still saved

        :param payload: result message: reason and the winning colour
        """
        self._forward_screen_event("on_result", payload)

    def _handle_connection_status(self, payload: dict[str, Any]) -> None:
        """
        Report how the opponent's connection is doing, which the player strip
        turns into a disconnect countdown, plus a toast while they resync

        :param payload: connection message carrying the opponent's state
        """
        self._forward_screen_event("on_connection_status", payload)
        if payload.get("opp_state", "connected") == "resyncing":
            self.app.toast.show("Opponent is resyncing…")

    def _handle_idle_window(self, payload: dict[str, Any]) -> None:
        """
        Pass on the server-pushed countdown that ends a game nobody is playing
        -- an abort before the first moves, an auto-resign after them -- so the
        board can show it ticking down

        :param payload: idle-window message: the outcome, the idle colour and
            the seconds left
        """
        self._forward_screen_event("on_idle_window", payload)

    def _begin_match_found_transition(self, payload: dict[str, Any]) -> None:
        """
        Greet a paired opponent with the match-found card and its countdown,
        holding the real game start back until the card has finished. A second
        pairing message is ignored while one is already pending

        :param payload: game-start message: both names and countries, this
            client's colour and the time control
        """
        if self._pending_game_start_payload is not None:
            return
        self._pending_game_start_payload = payload
        now = pg.time.get_ticks()
        self._match_found_at_ms = now
        try:
            self._match_found_started_seconds_ago = float(
                payload.get("started_seconds_ago", 0.0))
        except (TypeError, ValueError):
            self._match_found_started_seconds_ago = 0.0
        self.wait_modal.hide()
        room_id = self.client.room_id if self.client is not None else None
        log.info("match found room=%s side=%s", room_id, payload.get("your_color"))
        self.match_found_modal.show(
            env.clip_nickname(payload["white_name"]),
            env.clip_nickname(payload["black_name"]), payload["your_color"],
            self._finish_match_found, seconds=MATCH_FOUND_SECONDS,
            white_country=payload.get("white_country") or "",
            black_country=payload.get("black_country") or "",
            rematch=bool(payload.get("rematch")),
        )
        self.app.sound_manager.play_online_game_start()

    def _finish_match_found(self) -> None:
        """
        Start the game once the match-found card has run its countdown, then
        arm the first-move abort countdown with the time the card itself spent
        already taken off it
        """
        payload = self._pending_game_start_payload
        self._pending_game_start_payload = None
        matched_at_ms = self._match_found_at_ms
        self._match_found_at_ms = None
        elapsed = self._match_found_started_seconds_ago
        self._match_found_started_seconds_ago = 0.0
        self._wait_started_at_ms = None
        if payload is None:
            return
        self._start_online_game(payload)
        if matched_at_ms is not None:
            elapsed += (pg.time.get_ticks() - matched_at_ms) / 1000.0
        self._handle_idle_window({
            "outcome": Reason.ABORTED, "color": "white",
            "seconds_remaining": max(FIRST_MOVE_ABORT_SECONDS - elapsed, 0.0),
        })

    def _session_id_for_online(self) -> str:
        """
        Name the session the game screen files this match under, so a resumed
        game keeps writing to the same saved PGN instead of starting a new one

        :returns: the room id when there is one, otherwise a fresh uuid
        """
        if self.client is not None and self.client.room_id:
            return self.client.room_id
        return str(uuid.uuid4())

    def _start_online_game(self, payload: dict[str, Any]) -> None:
        """
        Hand the match to the game screen -- the single door leading from
        searching or reconnecting onto a real online board. The navigation runs
        immediately rather than being queued, so the board exists right away

        :param payload: game-start or resume payload: names, countries, this
            client's colour, time control and series scores
        """
        opp_name = env.clip_nickname(
            payload.get("white_name") if payload.get("your_color") == "black"
            else payload.get("black_name"))
        log.info("game start mode=online side=%s vs=%s tc=%s+%s",
                 payload.get("your_color"), opp_name,
                 payload.get("time_minutes"), payload.get("increment_seconds"))
        self.wait_modal.hide()
        self.app.confirm_modal.hide()
        self.app.menu.hide_play_view()
        nav_payload = {
            "your_color": payload["your_color"],
            "white_name": payload["white_name"],
            "black_name": payload["black_name"],
            "white_country": payload.get("white_country") or "",
            "black_country": payload.get("black_country") or "",
            "time_minutes": payload["time_minutes"],
            "increment_seconds": payload["increment_seconds"],
            "white_score": float(payload.get("white_score", 0.0)),
            "black_score": float(payload.get("black_score", 0.0)),
            "session_id": self._session_id_for_online(),
        }
        self.app.request_nav(Nav("game", nav_payload))
        self.app._execute_pending_nav()

    def _begin_online_flow(self, config: dict[str, Any]) -> None:
        """
        Start looking for an opponent with the settings the player chose on the
        menu. The settings are remembered, so a New Search after a failure can
        repeat exactly the same request

        :param config: play-view settings: nickname, time control and side
            preference
        """
        log.info("online flow begin tc=%s+%s side=%s",
                 config.get("time_minutes"), config.get("increment_seconds"),
                 config.get("side"))
        self._online_config = config
        self.app.menu.hide_play_view()
        self._on_server_addr_connect(env.get_server_addr())

    def _on_server_addr_connect(self, addr: str) -> None:
        """
        Open a fresh connection to the chosen server and queue for a match,
        replacing whatever session was there before. An empty address means
        there is nowhere to play, so the search is cancelled instead

        :param addr: server address to connect to, from the server picker
        """
        log.info("connect to %s", addr)
        if not addr:
            self._on_online_cancel()
            return
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        self._end_resync(replay=False)
        self.offer_banners.dismiss("rematch_request")
        self._rematch_offered = False
        self.client = OnlineClient()
        config = cast("dict[str, Any]", self._online_config)
        request = {
            "nickname": (config.get("nickname") or "").strip() or "Player",
            "client_uuid": env.get_or_create_client_uuid(),
            "time_minutes": config["time_minutes"] or ONLINE_DEFAULT_TIME_MINUTES,
            "increment_seconds": config["increment_seconds"],
            "side_preference": config["side"],
            "country": env.get_country() or None,
            "hide_opp_marks": env.get_hide_opp_marks(),
        }
        self.client.connect(addr, request)
        mode_label, tc_text = self._search_labels()
        self.wait_modal.show(mode_label, tc_text, self._on_online_cancel)
        self._wait_started_at_ms = pg.time.get_ticks()
        self._match_found_at_ms = None
        self._pending_game_start_payload = None

    def _search_labels(self) -> tuple[str, str]:
        """
        Describe the match being searched for the way the waiting card shows
        it, as a speed category over a readable time control

        :returns: the category name (Bullet, Blitz, Rapid) and the minutes plus
            increment as text
        """
        config = cast("dict[str, Any]", self._online_config)
        minutes = config.get("time_minutes") or ONLINE_DEFAULT_TIME_MINUTES
        incr = config.get("increment_seconds", 0) or 0
        return time_category_for_minutes(minutes), f"{minutes} + {incr}"

    def _drop_client(self, *, cancel_queue: bool = False) -> None:
        """
        Let go of the server session, either by leaving the matchmaking queue
        or by closing the socket outright. A resync in flight is abandoned with
        it, since there would be nothing left to apply the answer to

        :param cancel_queue: True while still queued, so the server drops the
            entry instead of leaving it to time out
        """
        self._end_resync(replay=False)
        if self.client is None:
            return
        if cancel_queue:
            self.client.cancel_queue()
        else:
            self.client.disconnect()
        self.client = None

    def _clear_search_state(self) -> None:
        """
        Wipe everything the search put on screen -- the waiting card, the
        match-found card and the pairing it was holding -- so the next search
        starts from nothing
        """
        self.wait_modal.hide()
        self.match_found_modal.hide()
        self._wait_started_at_ms = None
        self._match_found_at_ms = None
        self._pending_game_start_payload = None

    def unbind_game_from_online(self) -> None:
        """
        Turn the game screen back into a plain local board: no side ownership,
        no move forwarding, no online result menu and none of the disconnect or
        idle countdowns. Runs whenever an online session ends, for any reason
        """
        game = self.app.game
        game.variant = Variant.LOCAL
        game.match.local_color = None
        game.match.on_local_move_applied = None
        game.right_menu.set_game_info(None)
        game.result_menu.set_online_mode(False)
        game._idle_window = None
        game._opp_disconnected_at_ms = None
        game._local_disconnected_at_ms = None
        self._prev_online_state = None

    def _on_online_cancel(self) -> None:
        """
        Give the search up when the player cancels it or matchmaking times out:
        leave the queue, unbind the board and put the player back on the menu's
        play card
        """
        log.info("online flow cancel")
        self._drop_client(cancel_queue=True)
        self.unbind_game_from_online()
        self._clear_search_state()
        self.offer_banners.clear()
        self._return_to_menu_card()

    def _on_rematch(self) -> None:
        """
        Act on the result menu's Rematch button, which either accepts the offer
        already on the table or sends one of the player's own and says so in a
        toast while the opponent thinks about it
        """
        if self.client is None:
            return
        game = self.app.game
        if self._rematch_offered:
            self._rematch_offered = False
            game.result_menu.set_rematch_offered(False)
            log.info("rematch response sent accepted=True")
            self.client.send_rematch_response(True)
        else:
            log.info("rematch requested")
            self.client.send_rematch_request()
            self.app.toast.show(f"Rematch sent — waiting for {self._opp_name()}…",
                                key=REMATCH_STATE_TOAST_KEY)

    def _drop_post_game_online_session(self) -> None:
        """
        Close a session that is only still alive for the rematch window, done
        when the player starts a local or FEN game instead. Harmless when there
        is no session left to close
        """
        if self.client is None:
            return
        self._drop_client()
        self.unbind_game_from_online()

    def _tear_down_online_session(self, reason: str = "unspecified",
                                  navigate: bool = True) -> None:
        """
        End an online session for good: save the game, drop the connection,
        clear every online overlay and unbind the board. Normally it also puts
        the player back on the menu's play card, Reconnect probing and all,
        and resets the board for a new game

        :param reason: short label for the log line, such as restart_search
        :param navigate: False to leave the player where they are, used when a
            finished board is still worth looking at
        """
        log.info("online session teardown reason=%s", reason)
        game = self.app.game
        game.result_flow.auto_save_pgn()
        self._drop_client()
        self.reconnecting_modal.hide()
        self._clear_search_state()
        self.offer_banners.clear()
        self.unbind_game_from_online()
        if not navigate:
            return
        self._return_to_menu_card()
        game._reset_to_new_game()

    def _return_to_menu_card(self) -> None:
        """
        Put the player back on the menu's play card, whether they are already
        on the menu or on another screen, and let the Reconnect probe have a
        fresh set of attempts
        """
        if self.app.screen is self.app.menu:
            self.app.menu.show_play_view()
        else:
            self.app.switch_to("menu")
        self._reconnect_probe_attempts = 0

    def _abandon_online_game(self) -> None:
        """
        Walk away from a game that cannot be recovered, offered as Abandon on
        the reconnecting card and used when adopting a resumed game fails
        """
        self._tear_down_online_session("reconnect_cancelled")
        self._return_to_menu_card()

    def _restart_online_search(self) -> None:
        """
        Start a brand new search with the settings from the last one, the New
        Search answer to a hard failure. With no remembered settings there is
        nothing to repeat, so the player just goes back to the menu
        """
        self._tear_down_online_session("restart_search")
        if self._online_config is not None:
            self._begin_online_flow(self._online_config)
        else:
            self._return_to_menu_card()

    def retain_for_rematch(self, keep_online: bool) -> None:
        """
        Decide what becomes of the session when the player leaves a finished
        game: hold it open for the rematch window while telling the server they
        stepped off the board, or close it outright

        :param keep_online: True to keep the session alive for a rematch
        """
        self._end_resync(replay=False)
        if keep_online:
            cast(OnlineClient, self.client).send_left_result()
        elif self.client is not None:
            self.client.disconnect()
            self.client = None

    def _track_local_online_state(self) -> None:
        """
        Watch this client's own connection and stamp the moment it drops, which
        is what the reconnecting card and the local disconnect countdown both
        measure from
        """
        current = self.client.state if self.client is not None else None
        prev = self._prev_online_state
        game = self.app.game
        if current == "reconnecting" and prev != "reconnecting":
            game._local_disconnected_at_ms = pg.time.get_ticks()
        elif current != "reconnecting" and prev == "reconnecting":
            game._local_disconnected_at_ms = None
        self._prev_online_state = current

    def _send_heartbeat_if_due(self) -> None:
        """
        Keep the connection alive on the interval the server asked for, and
        carry this client's ply count in every ping so the server can notice a
        client lagging behind and answer with a resync directive, debounced on
        its side. A server that has gone silent is escalated to a reconnect
        """
        if self.client is None or not self.client.is_connected():
            return
        if self.client.is_server_silent():
            log.warning("server heartbeat silent; escalating to reconnect")
            self.client.force_reconnect()
            return
        game = self.app.game
        if game.skillcheck_session.online_verdict_action is not None:
            return
        now = pg.time.get_ticks()
        if now - self._last_heartbeat_sent_ms >= self.client.heartbeat_interval() * 1000:
            self._last_heartbeat_sent_ms = now
            self.client.send_ping(len(game.match.move_history))

    def _update_heartbeat(self) -> None:
        """
        Drive the heartbeat sound that quickens as the game runs out of road,
        taken from whichever countdown is closest to ending: the clock, a
        disconnect grace period or an idle window. It falls silent whenever the
        board is not live
        """
        game = self.app.game
        clock = game.match.clock
        paused = (self.app.screen is not self.app.game or game.current_result() is not None
                  or clock is None)
        if paused or clock.initial_seconds <= 0:
            fraction = None
        else:
            fraction = clock.remaining(game.match.current_turn()) / clock.initial_seconds
        auto_end_fraction = self._auto_end_heartbeat_fraction()
        if auto_end_fraction is not None:
            fraction = (auto_end_fraction if fraction is None
                        else min(fraction, auto_end_fraction))
        self.app.sound_manager.update_heartbeat(fraction, paused)

    def _auto_end_heartbeat_fraction(self) -> float | None:
        """
        Find the most urgent countdown that could end an online game without
        anyone moving -- the opponent's grace period, this client's own
        reconnect window, an idle window -- and size it for the heartbeat sound

        :returns: share of that window still left, 0.0 once it is inside the
            red threshold, or None when nothing is counting down
        """
        game = self.app.game
        if game.variant != Variant.ONLINE:
            return None
        now = pg.time.get_ticks()
        candidates: list[tuple[float, float]] = []
        for snap_ms, total in (
            (game._opp_disconnected_at_ms, GRACE_SECONDS),
            (game._local_disconnected_at_ms, RECONNECT_TOTAL_SECONDS),
        ):
            if snap_ms is None:
                continue
            remaining = total - (now - snap_ms) / 1000.0
            if remaining <= 0:
                continue
            candidates.append((remaining, total))
        window = game._idle_window
        if window is not None:
            remaining_ms = window.deadline_ms - now
            if remaining_ms > 0:
                candidates.append((remaining_ms / 1000.0, window.total_seconds))
        if not candidates:
            return None
        remaining, total = min(candidates, key=lambda r: r[0])
        if remaining < AUTO_END_RED_THRESHOLD_SECONDS:
            return 0.0
        return remaining / total

    def _tick_skillcheck_watchdog(self) -> None:
        """
        Rescue a skill check whose verdict never came back: once the overlay has
        been open well past the deadline, it is torn down and the whole state
        resynced, so one lost message cannot leave the player stuck
        """
        session = self.app.game.skillcheck_session
        if (session.online_skillcheck is None
                or session.online_skillcheck_opened_ms is None
                or not self.app.game.skillcheck_overlay.is_active()):
            return
        elapsed = pg.time.get_ticks() - session.online_skillcheck_opened_ms
        if elapsed > SKILLCHECK_DEADLINE_MS + SKILLCHECK_WATCHDOG_SLACK_MS:
            log.warning("skillcheck verdict lost; resyncing")
            session.teardown_skillcheck_overlay()
            self._begin_resync()

    def _update_online_phase(self) -> None:
        """
        Advance everything time-based on the online side of a frame: the
        waiting card's elapsed counter, the match-found countdown, the
        heartbeat and the reconnecting card. A resync that has not landed
        within eight seconds heals itself here, by giving up and reconnecting
        """
        now = pg.time.get_ticks()
        if (self.wait_modal.is_visible() and self._match_found_at_ms is None
                and self._wait_started_at_ms is not None):
            self.wait_modal.set_elapsed((now - self._wait_started_at_ms) // 1000)
        self.match_found_modal.update()
        self._track_local_online_state()
        self._send_heartbeat_if_due()
        game = self.app.game
        reconnecting = (game.variant == Variant.ONLINE and game.current_result() is None
                        and self.client is not None
                        and self.client.state == "reconnecting")
        if reconnecting:
            since = game._local_disconnected_at_ms
            if (not self.reconnecting_modal.is_visible() and since is not None
                    and now - since >= RECONNECT_MODAL_DEBOUNCE_MS):
                self.reconnecting_modal.show(
                    since, on_abandon=self._abandon_online_game)
        elif self.reconnecting_modal.is_visible():
            self.reconnecting_modal.hide()
        if self._resyncing:
            if now - self._resync_started_at_ms > RESYNC_TIMEOUT_MS:
                self._end_resync()
                if self.client is not None and self.client.state == "connected":
                    log.warning("resync timed out; escalating to reconnect")
                    self.client.force_reconnect()
            else:
                self.app.toast.show("Resyncing…")
        self._tick_skillcheck_watchdog()

    def _spawn_reconnect_probe(self) -> None:
        """
        Ask the server in the background whether this player has a game waiting
        to be reclaimed, which is what lights the menu's Reconnect button after
        the app was restarted. It gives up after a few fruitless attempts
        """
        addr = env.get_server_addr()
        client_uuid = env.get_or_create_client_uuid()
        if (not addr or not client_uuid or self._reconnect_probe_inflight
                or self._reconnect_probe_attempts >= RECONNECT_PROBE_MAX_ATTEMPTS):
            return
        with self._pending_reconnect_lock:
            if self._pending_reconnect is not None:
                return
        self._last_reconnect_probe_ms = pg.time.get_ticks()
        self._reconnect_probe_inflight = True
        with self._pending_reconnect_lock:
            self._reconnect_probe_gen += 1
            gen = self._reconnect_probe_gen
        log.debug("reclaim probe attempt addr=%s gen=%d", addr, gen)
        thread = threading.Thread(
            target=self._reconnect_probe_worker,
            args=(addr, client_uuid, gen),
            daemon=True,
        )
        thread.start()

    def _reconnect_probe_worker(self, addr: str, client_uuid: str, gen: int) -> None:
        """
        Run one reclaim probe off the main thread and remember any game the
        player may rejoin. A probe whose generation has gone stale -- the server
        target changed, or a reconnect already started -- drops its answer

        :param addr: server address being probed
        :param client_uuid: this installation's stable player identity
        :param gen: probe generation this call was started with
        """
        try:
            reclaim = probe_active_game(addr, client_uuid)
            resume = (fetch_resume(addr, reclaim["room_id"], reclaim["session_token"])
                      if reclaim is not None else None)
            with self._pending_reconnect_lock:
                if gen != self._reconnect_probe_gen:
                    return
                if reclaim is None or resume is None:
                    self._pending_reconnect = None
                    self._reconnect_probe_attempts += 1
                else:
                    became_available = self._pending_reconnect is None
                    self._pending_reconnect = {
                        "addr": addr,
                        "room_id": reclaim["room_id"],
                        "session_token": reclaim["session_token"],
                    }
                    if became_available:
                        log.info("reclaim available room=%s", reclaim["room_id"])
        finally:
            self._reconnect_probe_inflight = False

    def _refresh_reconnect_button(self) -> None:
        """
        Keep the menu's Reconnect button in step with reality, probing the
        server every few seconds while the player sits on the menu with no
        session of their own open
        """
        if (self.app.screen is self.app.menu and self.client is None
                and pg.time.get_ticks() - self._last_reconnect_probe_ms
                >= RECONNECT_PROBE_INTERVAL_MS):
            self._spawn_reconnect_probe()
        self.app.menu.set_reconnect_available(self.reconnect_available())

    def on_server_target_changed(self) -> None:
        """
        React to the player pointing the game at a different server: a game
        found on the old one stops counting, the Reconnect button goes dark,
        and probing starts over against the new address
        """
        with self._pending_reconnect_lock:
            self._reconnect_probe_gen += 1
            dropped = self._pending_reconnect
            self._pending_reconnect = None
        self._reconnect_probe_attempts = 0
        self._last_reconnect_probe_ms = 0
        self.app.menu.set_reconnect_available(False)
        if dropped is not None:
            log.info("reclaim dropped: server target changed")

    def reconnect_available(self) -> bool:
        """
        Say whether a game is sitting there waiting to be rejoined, which is
        what decides if the menu offers a Reconnect button at all

        :returns: True when a probe found a game this player can rejoin
        """
        with self._pending_reconnect_lock:
            return self._pending_reconnect is not None

    def reconnect(self) -> None:
        """
        Rejoin the game the Reconnect button is offering, the entry point the
        menu calls the moment the player presses it
        """
        self._on_reconnect_active_game()

    def _on_reconnect_active_game(self) -> None:
        """
        Rejoin a reclaimed game: fetch a fresh snapshot, rebuild the board from
        it, and only then attach the socket. A failed fetch keeps the offer
        alive behind a Retry dialog, a failed rebuild abandons the game
        """
        with self._pending_reconnect_lock:
            self._reconnect_probe_gen += 1
            pending = self._pending_reconnect
            self._pending_reconnect = None
        if pending is None:
            return
        log.info("reconnect: resume begin room=%s", pending["room_id"])
        self.app.menu.set_reconnect_available(False)
        resume = fetch_resume(
            pending["addr"], pending["room_id"], pending["session_token"],
        )
        if resume is None:
            log.warning("reconnect: fresh /resume failed; restoring pending entry")
            with self._pending_reconnect_lock:
                self._pending_reconnect = pending
            self.app.menu.set_reconnect_available(True)
            self.app.confirm_modal.show(
                ONLINE_HARD_FAILURE_LABELS[ClientReason.RECONNECT_FAILED],
                on_yes=self._on_reconnect_active_game,
                on_no=lambda: None,
                yes_label="Retry", no_label="Cancel",
            )
            return
        log.info("reconnect: resume ok room=%s", pending["room_id"])
        self.client = OnlineClient()
        if not self._guarded_apply("reconnect adoption",
                                   lambda: self._adopt_resumed_game(resume)):
            self._abandon_online_game()
            return
        self.client.reconnect_to_existing(
            pending["addr"], pending["room_id"], pending["session_token"], resume,
        )
        self.app.menu.hide_play_view()

    def _adopt_resumed_game(self, resume: dict[str, Any]) -> None:
        """
        Rebuild a whole game out of a server snapshot -- menu settings, board,
        clocks, marks and any pending skill check -- as one step, so a half
        restored game can never be left on screen

        :param resume: resume snapshot fetched from the server
        """
        self.app.menu.apply_resume_config(resume)
        self._start_online_game(resume)
        self._handle_game_resumed(resume)

    def update(self, now: int) -> None:
        """
        Advance the online side of one frame: drain the server's messages
        first, then the heartbeat, the online cards and the Reconnect button.
        The shell runs this before the active screen updates, so a premove
        answering the opponent's move still fires in the same frame

        :param now: pygame tick count in milliseconds since pygame init
        """
        self._drain_online_inbound()
        self._update_heartbeat()
        self._update_online_phase()
        self._refresh_reconnect_button()

    def on_app_exit(self) -> None:
        """
        Close the connection as the app quits. Every screen has already flushed
        its unsaved game by the time this runs, so nothing is left waiting on
        the socket
        """
        if self.client is not None:
            self.client.disconnect()
