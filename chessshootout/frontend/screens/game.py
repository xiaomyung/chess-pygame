import logging
import math
import random
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import pygame as pg

from chessshootout.backend.fen import apply_fen
from chessshootout.domain import openings
from chessshootout.domain.match import Match, SINGLE_SCREEN, BOT, ONLINE
from chessshootout.domain.capture_summary import captured_by, material_advantage
from chessshootout.domain.pgn.load import format_time_control
from chessshootout.backend.pieces import Piece, PieceColor, PieceType, opponent_of
from chessshootout.backend.pseudo_legal import king_square, piece_square
from chessshootout.backend.utils import (
    BOARD_SIZE, HistoryEntry, PROMO_TYPE_BY_LETTER, Square, coord_from_square,
    square_from_coord,
)
from chessshootout.infra import countries, env
from chessshootout.frontend import signal_controls
from chessshootout.frontend.board import Board
from chessshootout.frontend.board.speech_bubble import SpeechBubble
from chessshootout.frontend.modal_registry import ModalSpec
from chessshootout.frontend.modals.help import HOTKEYS
from chessshootout.frontend.screens.base import Screen
from chessshootout.frontend.skillcheck.overlay import SkillCheckOverlay
from chessshootout.skillcheck.coordinator import SkillCheckCoordinator
from chessshootout.skillcheck.types import SkillCheckKind
from chessshootout.frontend.panels.right import (
    Cap,
    RightMenu,
    CAPS as RIGHT_MENU_CAPS,
    UNTIMED_CAPS as RIGHT_MENU_UNTIMED_CAPS,
)
from chessshootout.frontend.panels.player_strip import (
    PlayerStrip, is_white, top_strip_color, refresh_capture_icons,
)
from chessshootout.frontend.modals.result import ResultMenu
from chessshootout.frontend.focus.arrow import (
    FocusArrow, FOCUS_EDGE_ZONE_PX, FOCUS_ARROW_D, LONG_AGO_MS,
)
from chessshootout.frontend.focus.time_line import TimeLine
from chessshootout.frontend.focus.transition import FocusTransition
from chessshootout.frontend.layout import compute_layout
from chessshootout.frontend.online_coordinator import (
    ONLINE_TRANSIENT_REASON_LABELS, ResyncCause,
)
from chessshootout.frontend.visual import cache
from chessshootout.frontend.visual.colors import Colors
from chessshootout.frontend.visual.backdrop import ArenaBackdrop
from chessshootout.frontend.visual.effects import TAKEOVER_TOTAL_MS
from chessshootout.frontend.game.result_flow import ResultFlow, score_str
from chessshootout.frontend.game.skillcheck_session import CheckContext, SkillCheckSession
from chessshootout.frontend.game.give_time import GiveTimeHold
from chessshootout.frontend.game.variant import MATCH_MODE_BY_VARIANT, Variant
from chessshootout.skillcheck.combo import COMBO_DIRECTIONS, COMBO_PROMPT_COUNT_MAX
from chessshootout.server.protocol import (
    CHAT_COOLDOWN_SECONDS, GRACE_SECONDS, IDLE_SECONDS_BY_OUTCOME, Reason,
    SKILLCHECK_TARGET_MAX,
)
from chessshootout.online.client import RECONNECT_TOTAL_SECONDS


log = logging.getLogger("chess.frontend")

OPPONENT_NAME_FOR_MODE = {
    SINGLE_SCREEN: "Player 2",
    BOT: "Bot",
}

PLACEHOLDER_RATING = "1500"

VARIANT_PILL_LABELS = {
    Variant.LOCAL: "Local",
    Variant.BOT: "Bot",
    Variant.ONLINE: "Online",
}

CHAT_PRESETS = (
    "GG, PARTNER", "HELL OF A SHOT", "THAT STINGS", "REMATCH AT DAWN?",
    "YES", "NO", "HAHAHAH!", "NICE TRY",
)

PROMOTION_KEYS = {
    pg.K_q: PieceType.QUEEN,
    pg.K_r: PieceType.ROOK,
    pg.K_b: PieceType.BISHOP,
    pg.K_n: PieceType.KNIGHT,
}

FOCUS_OFF_LINGER_MS = 2000
FOCUS_EDGE_ARROW_MARGIN = 16
FOCUS_ARROW_OFF_GAP = 6
FOCUS_HINT_MS = 1600
PRESENT_SETTLE_MS = 120

AUTO_FLIP_DELAY_MS = 200
RESULT_FADE_MS = 400
RESULT_MODAL_DELAY_MS = 500
RESULT_TAKEOVER_MS = TAKEOVER_TOTAL_MS
RESULT_FADE_MAX_ALPHA = 140

AUTO_END_GATE_FRACTION = 0.1

CHAT_COOLDOWN_MARGIN_MS = 150

ONLINE_WIN_RESULT_BY_REASON = {
    Reason.CHECKMATE:   ("white_wins",                "black_wins"),
    Reason.TIMEOUT:     ("white_wins_on_time",        "black_wins_on_time"),
    Reason.RESIGNATION: ("white_wins_by_resignation", "black_wins_by_resignation"),
    Reason.ABANDONMENT: ("white_wins_by_abandonment", "black_wins_by_abandonment"),
}
ONLINE_WIN_REASONS = frozenset(ONLINE_WIN_RESULT_BY_REASON)
ONLINE_DRAW_REASONS = {
    Reason.DRAW_AGREEMENT, Reason.DRAW_STALEMATE, Reason.DRAW_REPETITION,
    Reason.DRAW_FIFTY_MOVE, Reason.DRAW_INSUFFICIENT_MATERIAL,
}
ONLINE_STATIC_RESULTS = {
    Reason.ABORTED, Reason.SERVER_SHUTDOWN,
}
IDLE_LABEL_BY_OUTCOME = {Reason.ABORTED: "Abort in", Reason.RESIGNATION: "Resign in"}
IDLE_RESIGN_NOTICE_SECONDS = 30.0

OPPONENT_AIMING_LABEL = "Opponent is lining up a shot…"
RESUME_FEN_FAILED_LABEL = "Couldn't rebuild the position — resyncing"

ANIM_MS_DEFAULT = 180
ANIM_MS_MIN = 140
ANIM_MS_MAX = 280
ANIM_MS_PER_SECOND = 0.5

RESUME_MAX_PLIES = 4096
SPECTATE_PROGRESS_MAX = COMBO_PROMPT_COUNT_MAX
NO_LAST_HIT_POP = -1

_RESULT_FADE_CACHE = cache.new_size_cache()


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    """
    Turn a number that arrived off the wire into a real float, so a NaN, an
    infinity or a stray string can never reach clock maths or drawing code.
    Anything unusable becomes the caller's default

    :param value: raw payload field, whatever type the server sent
    :param default: stand-in for an unusable field; None lets the caller tell
        a missing field apart from a genuine zero
    :returns: the finite float, or the default
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _board_coord(value: Any) -> float | None:
    """
    Clamp one board-space coordinate from a spectate payload into the span the
    board really covers, so a mirrored shot is always drawn on the board.
    These are floats rather than square indices -- shots land between squares

    :param value: raw row or column exactly as the server sent it
    :returns: the clamped coordinate, or None when the field is unusable
    """
    coord = _finite_float(value, None)
    return None if coord is None else min(max(coord, 0.0), SKILLCHECK_TARGET_MAX)


def _spectate_target(payload: dict[str, Any]) -> tuple[float, float] | None:
    """
    Read the board-space point the opponent just shot at, so the spectate
    mirror can show where their shot landed. Both halves have to survive
    clamping, otherwise the shot is replayed without a target

    :param payload: spectate-shot payload, may carry target_row and target_col
    :returns: (row, column) in board space, or None when absent or unusable
    """
    row = _board_coord(payload.get("target_row"))
    col = _board_coord(payload.get("target_col"))
    if row is None or col is None:
        return None
    return (row, col)


def _spectate_direction(payload: dict[str, Any]) -> str | None:
    """
    Read the direction the opponent answered a combo prompt with, keeping only
    the four the combo check knows. Anything else is dropped rather than
    mirrored, since it arrives straight off the wire

    :param payload: spectate-shot payload, may carry a direction field
    :returns: the combo direction, or None when absent or unrecognised
    """
    direction = payload.get("direction")
    return direction if direction in COMBO_DIRECTIONS else None


class IdleWindow(NamedTuple):
    """
    The countdown the server is running against one side that has gone quiet:
    what happens when it expires, whose silence is being timed and when it
    runs out in local tick terms. The player strips show it as an auto-end
    badge, and only the server ever ends the game on it
    """

    outcome: str
    color: PieceColor
    deadline_ms: int
    total_seconds: float


class OnlineCheck(NamedTuple):
    """
    One server-issued skill check, decoded from the wire into everything the
    overlay needs to open. It holds no secret: the seed is fresh per check and
    the server alone decides whether a shot hit
    """

    kind: SkillCheckKind
    seed: str
    value_diff: int
    deadline_ms: float
    from_sq: Square
    to_sq: Square
    promo_type: PieceType | None
    captured_value: int

    def overlay_args(self) -> tuple[
            SkillCheckKind, str, int, float, Square, Square, PieceType | None, int]:
        """
        Lay the check out in the positional order the overlay openers expect,
        so one decoded check can arm either the player's own overlay or the
        passive spectate mirror without either caller unpacking it by hand

        :returns: kind, seed, value difference, deadline, from and to squares,
            promotion type and captured value, in that order
        """
        return (self.kind, self.seed, self.value_diff, self.deadline_ms,
                self.from_sq, self.to_sq, self.promo_type, self.captured_value)


def compute_animation_ms(initial_seconds: float | None) -> int:
    """
    Pick how long a piece takes to slide across the board, scaled to the time
    control so bullet games feel snappy and long games stay readable. Untimed
    games use the default, and the result is always clamped to a sane range

    :param initial_seconds: starting clock per side in seconds, None or zero
        when the game is untimed
    :returns: animation duration in milliseconds
    """
    if initial_seconds is None or initial_seconds <= 0:
        return ANIM_MS_DEFAULT
    return max(ANIM_MS_MIN, min(ANIM_MS_MAX, int(initial_seconds * ANIM_MS_PER_SECOND)))


class GameScreen(Screen):
    """
    The screen a game is played on. It owns the match, the board, the right
    rail, both player strips and the give-time, skill-check and result
    controllers, and the shell reuses this one instance for every game --
    local hotseat, bot or online, picked per entry by the variant
    """

    name = "game"

    def __init__(self, app: Any) -> None:
        """
        Build the whole game screen once at startup: match, board, rail,
        strips, speech bubbles and the controllers they lean on, all wired to
        the shell's shared services. The variant starts local and is decided
        again on every enter

        :param app: the Frontend shell hosting this screen, and this screen's
            only route to the window, sound, toasts, settings, the input
            router and the online coordinator
        """
        super().__init__(app)
        window = app.window

        self.variant = Variant.LOCAL
        self.manual_result: str | None = None
        self.white_name = "Player 1"
        self.black_name = "Player 2"
        self.white_country = ""
        self.black_country = ""
        self._chosen_side = "white"
        self._time_control: tuple[int, int] | None = None
        self._flag_fall_played = False
        self._strip_memo: dict[
            PieceColor, tuple[tuple[Any, ...], list[PieceType], int]] = {}
        self._result_first_seen_at_ms: int | None = None
        self._match_session_id: str | None = None
        self.backdrop = ArenaBackdrop()
        self._last_turn_for_flip: PieceColor | None = None
        self._idle_window: IdleWindow | None = None
        self._opp_disconnected_at_ms: int | None = None
        self._local_disconnected_at_ms: int | None = None
        self._focus_click_consumed = False
        self._custom_start = False
        self._debut = openings.DebutTracker(self._reviewed_sans)
        self._share_marks = False
        self._share_muted = False
        self._board_needs_present = False
        self._opp_sharing = False
        self._chat_cooldown_until_ms = 0
        self._speech_anchor_memo: dict[str, tuple[tuple[Any, ...], Square | None]] = {}
        self._game_info_memo: tuple[tuple[Any, ...], dict[str, Any] | None] | None = None

        self.match = Match()
        self.board = Board(window, self.match,
                           move_landed_callback=self._on_move_landed,
                           on_premove_queued=app.sound_manager.play_premove_queued,
                           shot_callback=self._on_shot_fired,
                           announce_callback=self._on_kill_announced)
        self.board.auto_queen_provider = env.get_auto_queen
        self.skillcheck = SkillCheckCoordinator()
        self.skillcheck_overlay = SkillCheckOverlay()
        self.skillcheck_session = SkillCheckSession(self)
        self.board.skillcheck_gate = self.skillcheck_session.skillcheck_gate
        self.board.skillcheck_armed = lambda: self.skillcheck.enabled
        self.board.locked_targets = self.skillcheck.is_locked

        self.result_flow = ResultFlow(self)
        self.give_time = GiveTimeHold(self)

        self.result_menu = ResultMenu(window, {
            "new_game": app._on_new_game,
            "open_pgn": self.result_flow.on_open_pgn,
            "menu": app._on_back_to_menu,
            "rematch": app.coordinator._on_rematch,
        }, pgn_available_provider=self.result_flow.pgn_available)
        self.right_menu = RightMenu(window, self.match, {
            "undo": self._on_undo,
            "resign": self._on_resign,
            "draw": self._on_draw,
            "flip": self._on_flip,
            "menu": app._on_back_to_menu,
            "help": app._on_help,
            "give_time": self.give_time.on_give_time,
            "share_toggle": self._on_share_toggle,
            "share_blocked": self._on_share_blocked,
            "hide_marks_toggle": self._on_hide_marks_toggle,
            "auto_q_toggle": self._toggle_auto_queen,
            "sound_toggle": lambda: signal_controls.toggle_sound(app),
            "set_volume": lambda value: signal_controls.set_signal_volume(app, value),
            "quick_chat": self._on_quick_chat_send,
        }, board=self.board, buttons_provider=self._right_menu_buttons,
            disabled_keys_provider=self._right_menu_disabled_keys,
            whiffs_provider=self.skillcheck_session.skillcheck_whiffs,
            debut_provider=self._debut_line,
            signals_provider=self._signals_provider,
            chat_visible_provider=lambda: self.variant == Variant.ONLINE,
            chat_presets_provider=lambda: CHAT_PRESETS,
            chat_cooldown_provider=self._chat_buttons_inert,
            sounds=app.sound_manager,
            suppress_click=app.input_router.suppress_click_sound)
        self.player_strip_top = PlayerStrip(window)
        self.player_strip_bottom = PlayerStrip(window)

        self.speech_bubbles = {"white": SpeechBubble(), "black": SpeechBubble()}

        self.focus_mode = False
        self.focus_transition: FocusTransition | None = None
        self.focus_arrow = FocusArrow()
        self.time_line = TimeLine()
        self._focus_panel_hover_ms = LONG_AGO_MS
        self._focus_hint_until_ms = 0

        self.result_flow.probe_games_dir_writable()

    @property
    def window(self) -> pg.Surface:
        """
        The app window this screen paints into, read back through the shell
        every time so a resize that swaps the display surface is picked up
        without anything here rebinding it

        :returns: the current pygame display surface
        """
        return cast(pg.Surface, self.app.window)

    def _compute_layout(self) -> None:
        """
        Ask the shell for a full layout pass, which recomputes the global
        rects and relayouts every screen. Focus mode changes the board's
        footprint, so the whole window has to be measured again, not just here
        """
        self.app._compute_layout()

    def enter(self, **payload: Any) -> None:
        """
        Start a game on this screen, called by the shell once the previous
        screen has exited. The payload picks the variant: side (with an
        optional FEN) opens a local hotseat or bot game, your_color opens an
        online one. Online is the only variant with a server behind it --
        resign, draw, takeback and give-time travel over the wire, mark
        sharing and quick chat exist only there, and the local auto-flip
        between turns is off

        :param payload: navigation payload -- variant, side, your_color,
            nickname, player names and countries, time control, series scores,
            session id and an optional starting FEN
        """
        openings.preload()
        fen = payload.get("fen")
        side = payload.get("side")
        your_color = payload.get("your_color")
        self.variant = payload.get("variant") or (
            Variant.ONLINE if your_color is not None else Variant.LOCAL)
        if side is not None or fen is not None:
            self._chosen_side = "white"
            self.white_name = "Player 1"
            self.black_name = "Player 2"
            self.white_country = env.get_country()
            self.black_country = countries.random_code()
            self._time_control = None
            self._ensure_local_session()
            self.skillcheck.reset(
                enabled=True,
                seed=f"local-{pg.time.get_ticks()}-{random.randint(0, 1 << 30)}")
        if side is not None:
            self._apply_start_config(payload)
        if your_color is not None:
            self._apply_online_start_config(payload)
        self._reset_to_new_game()
        if fen is not None:
            apply_fen(self.match.backend, fen)
            self._custom_start = True
        if your_color is not None:
            self.board.flipped = (your_color == "black")
            self.app.coordinator.subscribe(self)
        self._focus_hint_until_ms = pg.time.get_ticks() + FOCUS_HINT_MS
        log.info("game enter variant=%s tc=%s side=%s",
                 self.variant, self._time_control, self._chosen_side)

    def _apply_start_config(self, payload: dict[str, Any]) -> None:
        """
        Set up an offline game from the menu's choices: who sits on which
        side, the two display names and countries, and the time control. The
        opponent is the stock local name with a random country, since there is
        nobody on the other end to ask

        :param payload: navigation payload carrying side, nickname and the
            optional time_minutes and increment_seconds pair
        """
        self._chosen_side = payload["side"]
        nickname = (payload.get("nickname") or "").strip() or "Player 1"
        opponent_name = OPPONENT_NAME_FOR_MODE[SINGLE_SCREEN]
        self.white_name, self.black_name = (
            (nickname, opponent_name) if self._chosen_side == "white"
            else (opponent_name, nickname)
        )
        my_country = env.get_country()
        opp_country = countries.random_code()
        self.white_country, self.black_country = (
            (my_country, opp_country) if self._chosen_side == "white"
            else (opp_country, my_country)
        )
        time_minutes = payload.get("time_minutes")
        increment_seconds = payload.get("increment_seconds")
        self._time_control = cast(
            "tuple[int, int] | None",
            (time_minutes * 60, increment_seconds) if time_minutes is not None else None
        )

    def _apply_online_start_config(self, payload: dict[str, Any]) -> None:
        """
        Set up a server-backed game from the pairing the server sent: both
        nicknames and countries, the agreed time control, which colour is
        local and the running series score. It also points the match's local
        move hook at the coordinator, which is what puts moves on the wire

        :param payload: game-start or resume payload -- your_color, names,
            countries, time control, series scores and an optional session id
        """
        self._chosen_side = payload["your_color"]
        self.white_name = env.clip_nickname(payload["white_name"])
        self.black_name = env.clip_nickname(payload["black_name"])
        self.white_country = payload.get("white_country") or ""
        self.black_country = payload.get("black_country") or ""
        self._time_control = (payload["time_minutes"] * 60, payload["increment_seconds"])
        self.match.mode = ONLINE
        self.match.local_color = (PieceColor.WHITE if payload["your_color"] == "white"
                                  else PieceColor.BLACK)
        self.match.on_local_move_applied = self._on_local_move_applied
        self._match_session_id = payload.get("session_id") or str(uuid.uuid4())
        self.result_flow.series_scores = {
            self.white_name: float(payload.get("white_score", 0.0)),
            self.black_name: float(payload.get("black_score", 0.0)),
        }
        self._opp_disconnected_at_ms = None
        self._local_disconnected_at_ms = None
        self.skillcheck.reset(enabled=True, seed="online")
        self.result_menu.set_online_mode(True)

    def _on_local_move_applied(self, from_sq: Square, to_sq: Square,
                               promotion: str | None) -> None:
        """
        Put a move this player just made on the wire. The match calls this the
        moment a local ply is applied, which is the one place an online move
        is sent from, so a move can never land locally without being reported

        :param from_sq: square the piece left
        :param to_sq: square it arrived on
        :param promotion: promotion piece letter, or None for an ordinary move
        """
        self.app.coordinator.send_local_move(from_sq, to_sq, promotion)

    def on_app_exit(self) -> None:
        """
        Flush a finished game to disk on the way out, so quitting straight
        from the result screen still leaves a saved PGN. Games still in
        progress are already being written by the continuous autosave
        """
        if self.current_result() is not None:
            self.result_flow.auto_save_pgn()

    def on_offer(self, kind: str) -> None:
        """
        React to an offer the opponent just made. Only the rematch offer
        changes anything here -- it lights up the result menu's rematch button
        so accepting is one click; draw and takeback live on their banners

        :param kind: offer type, one of the coordinator's banner event names
        """
        if kind == "rematch_request":
            self.result_menu.set_rematch_offered(True)

    def on_connection_status(self, payload: dict[str, Any]) -> None:
        """
        Track when the opponent drops so the strips can count down to the
        abandonment win. The first reconnecting report stamps the clock and
        every later one is ignored, so the countdown does not restart

        :param payload: connection-status payload carrying opp_state
        """
        opp_state = payload.get("opp_state", "connected")
        if opp_state == "reconnecting":
            if self._opp_disconnected_at_ms is None:
                self._opp_disconnected_at_ms = pg.time.get_ticks()
        else:
            self._opp_disconnected_at_ms = None

    def on_idle_window(self, payload: dict[str, Any]) -> None:
        """
        Adopt the idle countdown the server pushes when the side to move has
        gone quiet, and show it as an auto-end badge on that player's strip.
        The server owns the decision -- this only mirrors it -- and a resign
        window that is nearly up on the opponent also raises a toast

        :param payload: idle-window payload -- outcome, the colour being timed
            and how many seconds are left
        """
        if self.variant != Variant.ONLINE or self.match.local_color is None:
            return
        outcome = payload.get("outcome")
        raw_color = payload.get("color")
        if outcome not in IDLE_SECONDS_BY_OUTCOME or raw_color not in ("white", "black"):
            return
        total = float(IDLE_SECONDS_BY_OUTCOME[outcome])
        seconds = min(
            max(cast(float, _finite_float(payload.get("seconds_remaining"), 0.0)), 0.0), total)
        color = PieceColor.WHITE if raw_color == "white" else PieceColor.BLACK
        self._idle_window = IdleWindow(
            outcome, color, pg.time.get_ticks() + int(seconds * 1000), total)
        if (outcome == Reason.RESIGNATION and color != self.match.local_color
                and seconds <= IDLE_RESIGN_NOTICE_SECONDS):
            idler = self._name_for_color(color)
            self.app.toast.show(f"{idler} is idle — auto-resign soon", key="idle_resign")

    def _adopt_idle_window(self, payload: Any) -> None:
        """
        Take the idle countdown a resume response reported, dropping whatever
        was showing first. A resume carries no window at all when none is
        armed, so the clear has to happen even when there is nothing to adopt

        :param payload: idle-window block from a resume response, or None
        """
        self._idle_window = None
        if isinstance(payload, dict):
            self.on_idle_window(payload)

    def on_annotations_state(self, payload: dict[str, Any]) -> None:
        """
        Replace the opponent's shared marks with the full set they just sent,
        which is how sharing starts and how it is corrected after a resync.
        A player who has chosen to hide opponent marks ignores this outright

        :param payload: annotations payload -- sharing flag plus the
            highlight squares and arrows, all in algebraic coordinates
        """
        if env.get_hide_opp_marks():
            return
        self._opp_sharing = bool(payload.get("sharing"))
        self.board.annotations.set_opp(
            self._decode_highlight_coords(payload.get("highlights", [])),
            self._decode_arrow_wires(payload.get("arrows", [])))

    def on_annotation_delta(self, payload: dict[str, Any]) -> None:
        """
        Apply a single mark the opponent added or removed, the cheap update
        that keeps shared drawings in step between full states. Anything that
        does not decode to a real square is dropped, since it comes off the
        wire, and a player hiding opponent marks ignores the whole message

        :param payload: delta payload -- action, kind, and either a square or
            a from/to pair in algebraic coordinates
        """
        if env.get_hide_opp_marks():
            return
        action = payload.get("action")
        kind = payload.get("kind")
        if action not in ("add", "remove") or kind not in ("highlight", "arrow"):
            return
        if kind == "highlight":
            try:
                square = square_from_coord(payload["square"])
            except (ValueError, KeyError, TypeError):
                return
            self.board.annotations.apply_opp_delta(action, kind, square=square)
        else:
            try:
                arrow = (square_from_coord(payload["from"]),
                         square_from_coord(payload["to"]))
            except (ValueError, KeyError, TypeError):
                return
            self.board.annotations.apply_opp_delta(action, kind, arrow=arrow)

    def on_annotations_blocked(self, payload: dict[str, Any]) -> None:
        """
        Tell the player that moderation refused some of their own marks: the
        offending shapes are flagged on the board and a toast explains why.
        Repeated offences mute sharing for the rest of the game, and a
        borderline drawing only earns the softer suspect warning

        :param payload: moderation payload -- action, the arrows and
            highlights that were refused, and the share_muted flag
        """
        action = payload.get("action")
        arrows = self._decode_arrow_wires(payload.get("arrows", []))
        highlights = self._decode_highlight_coords(payload.get("highlights", []))
        if action == "blocked":
            self.board.annotations.flag_own(arrows, highlights)
            log.info("marks blocked arrows=%d highlights=%d muted=%s",
                     len(arrows), len(highlights), bool(payload.get("share_muted")))
            self.app.toast.show("Some marks can't be shared", key="marks_blocked")
            if payload.get("share_muted"):
                self._share_muted = True
                self._toast_share_muted()
        elif action == "suspect":
            self.app.toast.show("Those marks look suspicious", key="marks_suspect")

    def apply_opp_marks_shield(self) -> None:
        """
        Wipe the opponent's marks off the board and stop showing them as
        sharing, the moment this player turns the hide-marks option on. The
        server is told separately; this is the immediate local effect
        """
        self.board.annotations.clear_opp()
        self._opp_sharing = False

    def on_quick_chat(self, payload: dict[str, Any]) -> None:
        """
        Show the canned line the opponent sent as a speech bubble over their
        king or queen. Only an index into the preset list travels, never free
        text, so an index outside the list is simply ignored

        :param payload: quick-chat payload -- preset index and sender colour
        """
        preset = int(payload.get("preset", -1))
        if not 0 <= preset < len(CHAT_PRESETS):
            return
        self.show_speech_bubble(payload["sender"], CHAT_PRESETS[preset])
        self.app.sound_manager.play_chat_receive()

    def _chat_buttons_inert(self) -> bool:
        """
        Say whether the quick-chat buttons should be dead right now, which the
        rail reads to grey them out. They rest during the server's send
        cooldown and stay off for good once the game has a result

        :returns: True while quick chat cannot be sent
        """
        return (self.current_result() is not None
                or pg.time.get_ticks() < self._chat_cooldown_until_ms)

    def _on_quick_chat_send(self, index: int) -> None:
        """
        Send one canned line and show it over this player's own piece straight
        away, so the sender sees it without waiting for the round trip. A
        local cooldown a little longer than the server's keeps the buttons
        from being spammed into a rate-limit rejection

        :param index: position in the preset list, from the rail button
        """
        if self.variant != Variant.ONLINE:
            return
        if not 0 <= index < len(CHAT_PRESETS):
            return
        if self._chat_buttons_inert():
            return
        self._chat_cooldown_until_ms = (
            pg.time.get_ticks() + int(CHAT_COOLDOWN_SECONDS * 1000) + CHAT_COOLDOWN_MARGIN_MS)
        self.app.coordinator.send_quick_chat(index)
        self.show_speech_bubble(self._chosen_side, CHAT_PRESETS[index])
        log.info("quick chat sent preset=%d", index)

    @staticmethod
    def _decode_highlight_coords(coords: list[Any]) -> set[Square]:
        """
        Turn a list of algebraic squares from the wire into board squares,
        quietly skipping anything that does not name a real square. Shared
        marks arrive from another player, so a bad entry must not break the
        whole set

        :param coords: algebraic square names as the server relayed them
        :returns: the squares that decoded cleanly
        """
        result = set()
        for coord in coords:
            try:
                result.add(square_from_coord(coord))
            except (ValueError, TypeError):
                continue
        return result

    @staticmethod
    def _decode_arrow_wires(wire_arrows: list[Any]) -> list[tuple[Square, Square]]:
        """
        Turn the wire's arrow records into from/to square pairs, dropping any
        entry that is missing an end or does not name a real square. Draw
        order is kept, since arrows are drawn in the order they were made

        :param wire_arrows: arrow records, each expected to carry from and to
        :returns: the arrows that decoded cleanly, in the order given
        """
        result = []
        for arrow in wire_arrows:
            try:
                result.append((square_from_coord(arrow["from"]),
                               square_from_coord(arrow["to"])))
            except (ValueError, KeyError, TypeError):
                continue
        return result

    def on_give_time(self, payload: dict[str, Any]) -> None:
        """
        Apply the clock the server rewrote after somebody gave time away, then
        tell whichever side this client is playing what happened. The server
        decides how many seconds actually moved, so the snapshot replaces the
        local clocks outright rather than adding to them

        :param payload: time-granted payload -- clock snapshot, seconds_added
            and the colour that gave the time
        """
        self._apply_clock_snap(payload, default_to_existing=False)
        added = float(payload.get("seconds_added", 0))
        granted_by = payload.get("granted_by")
        recipient_color = "black" if granted_by == "white" else "white"
        if granted_by == self._chosen_side:
            self.give_time.toast_for_giver(recipient_color, added)
        else:
            self.give_time.toast_for_receiver(cast(str, granted_by), added)

    def on_resume(self, payload: dict[str, Any]) -> None:
        """
        Rebuild the whole game from the server's resume snapshot after a drop,
        a restart or a resync. The position is rebuilt by starting a new game
        and replaying the SANs -- never by patching the current board -- and
        the FEN is only used as a fallback when a SAN fails to replay

        :param payload: resume payload -- move history, FEN, clock, marks,
            skill-check log and locks, any pending check, result and idle
            window
        """
        self.match.new_game()
        for entry in payload.get("move_history", [])[:RESUME_MAX_PLIES]:
            result = self.match.apply_san(entry["san"])
            if not result.legal:
                log.warning("resume: SAN replay failed at %r", entry.get("san"))
                self._adopt_resumed_fen(payload.get("fen"))
                break
        if self._time_control is not None:
            initial, incr = self._time_control
            self.match.setup_clock(initial, incr)
            self._apply_clock_snap(payload, default_to_existing=False)
        self.board.cancel_animations()
        self.board.selected_square = None
        self.board.clear_premoves()
        self.board.clear_all_annotations()
        self._restore_resumed_annotations(payload)
        self._adopt_resumed_result(payload)
        self.skillcheck_session.apply_resumed_skillcheck_log(payload.get("skillcheck_log", []))
        self._restore_online_skillcheck_state(payload)
        self._adopt_idle_window(payload.get("idle_window"))

    def _adopt_resumed_fen(self, fen: Any) -> None:
        """
        Take the position from the FEN a resume snapshot carries, the fallback
        for a move list that would not replay. A FEN the engine refuses is
        dropped rather than allowed to abandon the rest of the adoption: the
        replayed prefix stays on the board, the player is told, and the resync
        heartbeat converges the two sides again

        :param fen: the snapshot's FEN field exactly as the server sent it,
            which is to say not to be trusted
        """
        try:
            apply_fen(self.match.backend, str(fen))
        except (IndexError, KeyError, ValueError):
            log.warning("resume: FEN fallback refused")
            self.app.toast.show(RESUME_FEN_FAILED_LABEL, key="resume_fen_failed")
            return
        self._custom_start = True
        self.board.forget_position_memos()

    def _restore_resumed_annotations(self, payload: dict[str, Any]) -> None:
        """
        Put both players' marks back after a resume, picking this client's own
        set by the colour it is playing. The blocked-mark flags start clean,
        and the hide-marks option still wins over whatever the opponent had
        been sharing

        :param payload: resume payload carrying white_annotations,
            black_annotations and the share_muted flag
        """
        if self._chosen_side == "white":
            mine_key, opp_key = "white_annotations", "black_annotations"
        else:
            mine_key, opp_key = "black_annotations", "white_annotations"
        mine = payload.get(mine_key) or {}
        opp = payload.get(opp_key) or {}
        self.board.annotations.flagged = set()
        self.board.highlighted_squares = self._decode_highlight_coords(
            mine.get("highlights", []))
        self.board.arrows = self._decode_arrow_wires(mine.get("arrows", []))
        self._share_marks = bool(mine.get("sharing"))
        if env.get_hide_opp_marks():
            self.apply_opp_marks_shield()
        else:
            self.board.annotations.set_opp(
                self._decode_highlight_coords(opp.get("highlights", [])),
                self._decode_arrow_wires(opp.get("arrows", [])))
            self._opp_sharing = bool(opp.get("sharing"))
        self._share_muted = bool(payload.get("share_muted"))

    def on_takeback(self, payload: dict[str, Any]) -> None:
        """
        Undo the last ply because the server says the opponent's takeback was
        agreed, animating the piece back and dropping that ply's skill-check
        entries. A ply count that disagrees with the local history means the
        two sides have drifted, so this asks for a resync instead

        :param payload: takeback-applied payload -- the resulting ply count
            and a clock snapshot
        """
        server_ply = payload.get("ply")
        expected = len(self.match.move_history) - 1
        if server_ply is not None and server_ply != expected:
            log.warning("takeback ply gap server_ply=%s expected=%d", server_ply, expected)
            self.app.coordinator._begin_resync(ResyncCause.TAKEBACK_PLY_GAP)
            return
        self._idle_window = None
        self.board.clear_all_annotations()
        if self.match.move_history:
            self.skillcheck_session.drop_skillcheck_log_from(len(self.match.move_history))
            last = self.match.move_history[-1].move
            self.match.undo()
            self.board.start_undo_animation(last)
            self.app.sound_manager.play_undo()
            log.info("takeback applied ply=%d", len(self.match.move_history))
        self._apply_clock_snap(payload, default_to_existing=True)

    def _apply_clock_snap(self, payload: dict[str, Any], *,
                          default_to_existing: bool) -> None:
        """
        Overwrite the local clocks with the server's, which is the only thing
        that decides remaining time in an online game. Untimed games ignore it
        entirely

        :param payload: any server payload that may carry a clock block with
            white_remaining, black_remaining and running_for
        :param default_to_existing: keep the current readings when the block
            is missing, instead of zeroing them
        """
        clock_snap = payload.get("clock") or {}
        if self.match.clock is None:
            return
        white_default = self.match.clock.white_remaining if default_to_existing else 0.0
        black_default = self.match.clock.black_remaining if default_to_existing else 0.0
        self.match.clock.restore_from_server(
            clock_snap.get("white_remaining", white_default),
            clock_snap.get("black_remaining", black_default),
            clock_snap.get("running_for"),
        )

    def _is_my_open_check(self, from_sq: Square, to_sq: Square) -> bool:
        """
        Say whether the skill check open on screen is this player's own shot
        at the given move, rather than the read-only mirror of the opponent's.
        Only the mover's overlay may resolve into a landed move

        :param from_sq: square the pending capture or promotion starts from
        :param to_sq: square it is aimed at
        :returns: True when the local player owns the open check for that move
        """
        return (self.skillcheck_session.online_skillcheck is not None
                and self.skillcheck_session.online_spectate_kind is None
                and self.skillcheck_session.online_skillcheck[:2] == (from_sq, to_sq))

    def _begin_online_verdict(self, won: bool, action: Callable[[], None]) -> None:
        """
        Play out the server's verdict on an open skill check: the overlay runs
        its win or miss flourish first and the consequence is held back until
        that finishes, so the move never lands before the shot is shown

        :param won: True when the server says the shot hit
        :param action: what to run once the overlay has finished -- apply the
            move, or lock it and play the miss
        """
        self.skillcheck_session.online_skillcheck = None
        self.skillcheck_session.online_spectate_kind = None
        self.skillcheck_session.online_verdict_action = action
        self.skillcheck_overlay.resolve_online(won)

    def _apply_online_fail(self, from_sq: Square, to_sq: Square,
                           aim_victim: Square | None,
                           kind: SkillCheckKind | None) -> None:
        """
        Finish a skill check this player lost: that exact move is locked for
        the rest of the turn, and the missed shot is played out on the board.
        Locking is what stops a missed capture from simply being retried

        :param from_sq: square the failed capture or promotion started from
        :param to_sq: square it was aimed at
        :param aim_victim: square whose piece the overlay was hiding, restored
            unless the mole check ate it, None when nothing was suppressed
        :param kind: which check ran, None when it could not be determined
        """
        self.skillcheck.lock(from_sq, to_sq)
        log.info("skillcheck move locked from=%s to=%s online=True",
                 coord_from_square(from_sq), coord_from_square(to_sq))
        self._apply_spectate_fail(from_sq, to_sq, aim_victim, kind)

    def _apply_spectate_fail(self, from_sq: Square, to_sq: Square,
                             aim_victim: Square | None,
                             kind: SkillCheckKind | None) -> None:
        """
        Play the missed shot on the board -- the muzzle flash, the swearing
        and the piece that survives it. Shared by the mover and the spectator,
        because both sides watch the same miss; only the mover also locks the
        move

        :param from_sq: square the failed capture or promotion started from
        :param to_sq: square it was aimed at
        :param aim_victim: square whose piece the overlay was hiding and which
            now drops back in, None when nothing was suppressed
        :param kind: which check ran; the mole check keeps its victim hidden
        """
        self.board.trigger_skillcheck_fail(
            from_sq, to_sq, on_fire=self.skillcheck_session.on_skillcheck_miss_fire)
        if aim_victim is not None and kind != SkillCheckKind.WHACK:
            self.board.restore_piece(aim_victim)

    def on_skillcheck_required(self, payload: dict[str, Any]) -> None:
        """
        Handle the server's word on this player's own skill check: a payload
        naming a kind opens the overlay, and one without says the check is
        already lost. Both arrive on the same channel, which is why the shape
        of the payload picks the branch

        :param payload: skill-check payload -- either a full check to open or
            the from/to pair of a check that has just been lost
        """
        if "kind" in payload:
            self._open_my_skillcheck(payload)
        else:
            self._resolve_my_skillcheck_failure(payload)

    @staticmethod
    def _decode_squares(payload: dict[str, Any]) -> tuple[Square, Square]:
        """
        Pull the from and to squares out of a server payload as board squares.
        Both keys are required, so a malformed payload raises rather than
        silently addressing the wrong square

        :param payload: any payload carrying from and to in algebraic form
        :returns: the (from, to) pair as board squares
        """
        return square_from_coord(payload["from"]), square_from_coord(payload["to"])

    @staticmethod
    def _decode_check(payload: dict[str, Any]) -> OnlineCheck:
        """
        Decode a skill check the server issued into the record the overlay is
        built from. Nothing is invented here -- the kind, seed and deadline
        are all the server's, and the client never picks its own geometry

        :param payload: skill-check payload -- kind, seed, value_diff,
            deadline_ms, from and to, promotion letter and captured value
        :returns: the decoded check, ready to hand to an overlay opener
        """
        promo = payload.get("promotion")
        from_sq, to_sq = GameScreen._decode_squares(payload)
        return OnlineCheck(
            kind=SkillCheckKind(payload["kind"]),
            seed=payload["seed"],
            value_diff=int(payload["value_diff"]),
            deadline_ms=cast(float, _finite_float(payload["deadline_ms"])),
            from_sq=from_sq,
            to_sq=to_sq,
            promo_type=PROMO_TYPE_BY_LETTER.get(promo) if promo else None,
            captured_value=int(payload.get("captured_value", 0)),
        )

    def _open_my_skillcheck(self, payload: dict[str, Any]) -> None:
        """
        Open the playable skill-check overlay for a capture or promotion this
        player is attempting. The elapsed time the server reports is honoured,
        so a check adopted mid-flight after a reconnect resumes where it
        stood rather than restarting

        :param payload: skill-check payload -- the check itself plus how much
            of its deadline has already gone and how many shots were missed
        """
        check = self._decode_check(payload)
        session = self.skillcheck_session
        session.pending_online_move = None
        session.online_skillcheck = CheckContext(
            check.from_sq, check.to_sq, check.promo_type, check.kind)
        session.online_skillcheck_opened_ms = pg.time.get_ticks()
        session.online_last_hit_pop = NO_LAST_HIT_POP
        session.open_skillcheck_overlay(
            *check.overlay_args(), online=True,
            elapsed_ms=cast(float, _finite_float(payload.get("elapsed_ms", 0.0))),
            miss_count=int(payload.get("miss_count", 0)))

    def _prepare_online_fail(self, payload: dict[str, Any]) -> tuple[
            Square, Square, Square | None, SkillCheckKind | None]:
        """
        Work out everything a lost skill check needs before the overlay is
        torn down: which move failed, which piece the overlay was hiding and
        which check ran. The miss is also written into the log now, so the
        saved PGN carries it even if the game ends immediately after

        :param payload: skill-check payload naming the move that was lost
        :returns: from square, to square, the suppressed victim square if any,
            and the kind of check that was running
        """
        from_sq, to_sq = self._decode_squares(payload)
        pending = self.skillcheck_session.online_skillcheck
        self.skillcheck_session.record_online_fail(pending, from_sq, to_sq)
        aim_victim = self.board.aim_suppressed_square
        kind = pending.kind if pending is not None else None
        return from_sq, to_sq, aim_victim, kind

    def _resolve_my_skillcheck_failure(self, payload: dict[str, Any]) -> None:
        """
        Close out a skill check this player lost: the overlay plays its miss
        and only then is the move locked and the shot played on the board.
        The selection is dropped so the losing piece is not left picked up

        :param payload: skill-check payload naming the move that was lost
        """
        from_sq, to_sq, aim_victim, kind = self._prepare_online_fail(payload)
        self.board.selected_square = None
        self._begin_online_verdict(
            False, lambda: self._apply_online_fail(from_sq, to_sq, aim_victim, kind))

    def on_spectate(self, payload: dict[str, Any]) -> None:
        """
        Handle the read-only mirror of the opponent's skill check, so both
        players watch the same shootout. The payload's shape says which stage
        it is: a kind opens the mirror, a from/to pair ends it in a miss, and
        anything else is one of their shots being replayed

        :param payload: spectate payload -- a full check, a lost move, or a
            single shot with its elapsed time and outcome
        """
        if "kind" in payload:
            self._open_spectated_skillcheck(payload)
        elif "from" in payload:
            self._resolve_spectated_skillcheck_failure(payload)
        else:
            self._spectate_shot(payload)

    def _open_spectated_skillcheck(self, payload: dict[str, Any]) -> None:
        """
        Open the passive mirror of the check the opponent is shooting, so this
        player watches it happen instead of staring at a frozen board. The
        mirror takes no input -- it only replays the shots the server relays

        :param payload: spectate payload carrying the opponent's full check
        """
        check = self._decode_check(payload)
        self.skillcheck_session.online_last_hit_pop = NO_LAST_HIT_POP
        self.skillcheck_session.open_spectate_overlay(*check.overlay_args())
        self.app.toast.show(OPPONENT_AIMING_LABEL)

    def _resolve_spectated_skillcheck_failure(self, payload: dict[str, Any]) -> None:
        """
        Close out the mirror when the opponent misses: they are told about it,
        and the miss is played on the board once the overlay has finished. No
        move lock is applied here -- the lock belongs to the mover's client

        :param payload: spectate payload naming the move they lost
        """
        from_sq, to_sq, aim_victim, kind = self._prepare_online_fail(payload)
        self.app.toast.show("Opponent missed!")
        self._begin_online_verdict(
            False, lambda: self._apply_spectate_fail(from_sq, to_sq, aim_victim, kind))

    def _spectate_shot(self, payload: dict[str, Any]) -> None:
        """
        Replay one shot the opponent just fired inside the passive mirror.
        Everything the server sends is clamped first -- progress, target and
        direction alike -- because it is drawn straight onto this board

        :param payload: spectate-shot payload -- elapsed time, miss count,
            whether it hit, progress so far and the target or direction
        """
        self.skillcheck_overlay.spectate_shot(
            cast(float, _finite_float(payload["elapsed_ms"])), int(payload["miss_count"]),
            bool(payload["won"]),
            progress=min(int(payload.get("progress", 0)), SPECTATE_PROGRESS_MAX),
            direction=_spectate_direction(payload), target=_spectate_target(payload))

    def _clear_online_move_locks(self, from_sq: Square, to_sq: Square) -> None:
        """
        Drop the skill-check move locks once a ply has actually landed, since
        they only ever bind for the turn they were earned on. A move still
        waiting on the server is forgotten too when it is the one that landed

        :param from_sq: square the landed move started from
        :param to_sq: square it arrived on
        """
        if self.variant != Variant.ONLINE:
            return
        self.skillcheck.clear_locks()
        if (self.skillcheck_session.pending_online_move is not None
                and self.skillcheck_session.pending_online_move[:2] == (from_sq, to_sq)):
            self.skillcheck_session.pending_online_move = None

    def on_remote_move(self, payload: dict[str, Any]) -> None:
        """
        Apply a move the server confirmed, whoever made it. When that move won
        a skill check the overlay on screen plays its win first and the move
        lands afterwards, so the shot and the capture read as one action

        :param payload: move-applied payload -- from, to, SAN, ply, clock,
            promotion letter and the skill-check kind and outcome if any
        """
        from_sq = square_from_coord(payload["from"])
        to_sq = square_from_coord(payload["to"])
        if payload.get("skill_check_kind") is not None:
            if self._is_my_open_check(from_sq, to_sq):
                self._begin_online_verdict(
                    True, lambda: self._apply_remote_move_payload(payload))
                return
            if self.skillcheck_session.online_spectate_kind is not None:
                self.app.toast.show("Opponent nailed it!")
                self._begin_online_verdict(
                    True, lambda: self._apply_remote_move_payload(payload))
                return
        self._apply_remote_move_payload(payload)

    def _apply_remote_move_payload(self, payload: dict[str, Any]) -> None:
        """
        Land a server-confirmed move on the local board, animate it and log
        its skill check. A move whose SAN is already the last one played is
        just a clock update, and a ply number that disagrees with the local
        history -- or a move the engine calls illegal -- triggers a resync

        :param payload: move-applied payload -- from, to, SAN, ply, clock,
            promotion letter and the skill-check kind and outcome if any
        """
        from_sq = square_from_coord(payload["from"])
        to_sq = square_from_coord(payload["to"])
        san = payload.get("san")
        last = self.match.move_history[-1] if self.match.move_history else None
        if last is not None and last.san == san:
            self._apply_clock_snap(payload, default_to_existing=True)
            self._clear_online_move_locks(from_sq, to_sq)
            return
        server_ply = payload.get("ply")
        expected = len(self.match.move_history) + 1
        if server_ply is not None and server_ply != expected:
            log.warning("move ply gap server_ply=%s expected=%d", server_ply, expected)
            self.app.coordinator._begin_resync(ResyncCause.MOVE_PLY_GAP)
            return
        self._apply_clock_snap(payload, default_to_existing=True)
        promo = payload.get("promotion")
        promo_type = PROMO_TYPE_BY_LETTER.get(promo) if promo else None
        result = self.match.apply_remote_move(from_sq, to_sq, promo_type)
        if result.legal:
            self._idle_window = None
            self.board.animate_remote_move(from_sq, to_sq)
            self._clear_online_move_locks(from_sq, to_sq)
            kind = payload.get("skill_check_kind")
            if kind is not None:
                self.skillcheck_session.record_skillcheck(
                    kind, bool(payload.get("skill_check_won")),
                    payload.get("ply") or len(self.match.move_history))
        else:
            self.app.coordinator._begin_resync(ResyncCause.MOVE_ILLEGAL)

    def on_result(self, payload: dict[str, Any]) -> None:
        """
        Adopt the server's verdict on how the game ended, which is what closes
        an online game down: banners go, a pending skill-check verdict is
        played out and the overlay is torn down. A result already settled
        locally wins, so a late server message cannot rewrite it

        :param payload: result payload -- the reason code and, for a decisive
            game, the winning colour
        """
        if self.manual_result is not None:
            return
        reason = payload.get("reason", "")
        if not self._apply_online_result(reason, payload.get("winner_color")):
            return
        coordinator = self.app.coordinator
        self._idle_window = None
        self._opp_disconnected_at_ms = None
        self._local_disconnected_at_ms = None
        coordinator.offer_banners.clear()
        self.skillcheck_session.pending_online_move = None
        pending_action = self.skillcheck_session.online_verdict_action
        self.skillcheck_session.online_verdict_action = None
        try:
            if pending_action is not None:
                pending_action()
            self.skillcheck_session.teardown_skillcheck_overlay()
        except Exception:
            log.exception("online result verdict/teardown failed")
            coordinator._begin_resync(ResyncCause.RESULT_APPLY_FAILED)
        if reason == Reason.TIMEOUT and not self._flag_fall_played:
            self._flag_fall_played = True
            self.app.sound_manager.play_flag_fall()

    def _apply_online_result(self, reason: str, winner: str | None) -> bool:
        """
        Translate the server's reason code into the result code this app saves
        and shows, and adopt it. Codes the client does not recognise are
        refused outright rather than guessed at, and an already settled result
        is never overwritten

        :param reason: server reason code for the ending
        :param winner: winning colour for a decisive game, None otherwise
        :returns: True when a result was adopted from this message
        """
        if self.manual_result is not None:
            return False
        if reason in ONLINE_WIN_REASONS:
            white_code, black_code = ONLINE_WIN_RESULT_BY_REASON[reason]
            self.manual_result = white_code if winner == "white" else black_code
        elif reason in ONLINE_DRAW_REASONS or reason in ONLINE_STATIC_RESULTS:
            self.manual_result = reason
        else:
            return False
        self.result_flow.on_result_final(self.manual_result)
        return True

    def _adopt_resumed_result(self, payload: dict[str, Any]) -> None:
        """
        Pick up a result that had already been decided while this client was
        away, so reconnecting into a finished game shows the ending instead of
        a live board. A resume for a game still running carries no reason and
        changes nothing

        :param payload: resume payload, may carry result_reason and
            result_winner
        """
        reason = payload.get("result_reason") or ""
        self._apply_online_result(reason, payload.get("result_winner"))

    def _restore_online_skillcheck_state(self, payload: dict[str, Any]) -> None:
        """
        Rebuild the skill-check side of the game after a resume: any overlay
        on screen is torn down, the locked moves are taken from the server's
        list, and a check still running is reopened. The server is the only
        source here, so nothing local survives the rebuild

        :param payload: resume payload carrying skillcheck_locks and an
            optional pending_skillcheck block
        """
        self.skillcheck_session.teardown_skillcheck_overlay()
        self.skillcheck_session.pending_online_move = None
        self.skillcheck.clear_locks()
        for lock in payload.get("skillcheck_locks", []):
            self.skillcheck.lock(square_from_coord(lock["from"]),
                                 square_from_coord(lock["to"]))
        pending = payload.get("pending_skillcheck")
        if pending is not None:
            self._restore_pending_skillcheck(pending)

    def _restore_pending_skillcheck(self, pending: dict[str, Any]) -> None:
        """
        Reopen a skill check that was still running when this client came
        back, as the player's own overlay or as the passive mirror depending
        on whose turn it is. Elapsed time, misses and progress are all taken
        from the server, so the deadline keeps running down where it was

        :param pending: pending-check block -- the check itself plus the
            colour shooting, elapsed time, miss count, progress and last hit
        """
        check = self._decode_check(pending)
        elapsed_ms = cast(float, _finite_float(pending.get("elapsed_ms", 0.0)))
        miss_count = int(pending.get("miss_count", 0))
        progress = min(int(pending.get("progress", 0)), SPECTATE_PROGRESS_MAX)
        self.skillcheck_session.online_last_hit_pop = int(
            pending.get("last_hit_pop", NO_LAST_HIT_POP))
        if pending["color"] != self._chosen_side:
            self.skillcheck_session.open_spectate_overlay(
                *check.overlay_args(),
                elapsed_ms=elapsed_ms, miss_count=miss_count, progress=progress)
            self.app.toast.show(OPPONENT_AIMING_LABEL)
            return
        self.skillcheck_session.online_skillcheck = CheckContext(
            check.from_sq, check.to_sq, check.promo_type, check.kind)
        self.skillcheck_session.online_skillcheck_opened_ms = pg.time.get_ticks() - int(elapsed_ms)
        self.skillcheck_session.open_skillcheck_overlay(
            *check.overlay_args(), online=True,
            elapsed_ms=elapsed_ms, miss_count=miss_count, progress=progress)

    def exit(self) -> None:
        """
        Hand the window back and cancel everything screen-local, so nothing
        from this game leaks into the next screen: the coordinator
        subscription, focus mode, an in-flight drag, queued premoves, board
        marks and any live skill-check overlay all go. Entering again always
        starts from a clean baseline
        """
        super().exit()
        self.app.coordinator.unsubscribe(self)
        did_focus = self.focus_mode or self.focus_transition is not None
        if did_focus:
            self._force_focus_off_instant()
        was_dragging = self.board.is_dragging()
        self.board.cancel_drag_physics()
        self.board.end_press()
        had_premoves = bool(self.board.premoves)
        self.board.clear_premoves()
        had_annotations = bool(self.board.highlighted_squares or self.board.arrows)
        self.board.clear_all_annotations()
        overlay_active = self.skillcheck_overlay.is_active()
        if overlay_active:
            self.skillcheck_session.teardown_skillcheck_overlay()
        if did_focus or was_dragging or had_premoves or had_annotations or overlay_active:
            log.debug(
                "game exit teardown focus=%s drag=%s premoves=%s annotations=%s skillcheck=%s",
                did_focus, was_dragging, had_premoves, had_annotations, overlay_active)

    def update(self, now: int) -> None:
        """
        Advance the game by one frame before it is drawn: run the clock, play
        the flag fall, and once the board has settled either flip it for the
        next player (local hotseat only) or fire the next queued premove. This
        screen never navigates away on its own, so it returns nothing

        :param now: pygame tick count in milliseconds since pygame init
        """
        if self.skillcheck_overlay.is_active() and self.current_result() is not None:
            self.skillcheck_session.teardown_skillcheck_overlay()
        self._maybe_play_flag_fall()
        if self.game_live():
            self.match.tick_clock()
        post_animation_settled = (
            not self.board.is_animating()
            and not self.board.effects.captures
            and now - self.board.last_animation_completed_at_ms >= AUTO_FLIP_DELAY_MS
        )
        if (self.variant == Variant.LOCAL
                and self.current_result() is None
                and post_animation_settled):
            current = self.match.current_turn()
            if current != self._last_turn_for_flip:
                self.board.flipped = (current == PieceColor.BLACK)
                self._last_turn_for_flip = current
        if (self.game_live()
                and post_animation_settled
                and self.skillcheck_session.pending_online_move is None
                and not self.skillcheck_session.skillcheck_swallows_input()):
            self.board.try_apply_next_premove()

    def _maybe_play_flag_fall(self) -> None:
        """
        Sound the moment a clock runs out, once per game. Online it depends on
        which side flagged -- losing on time and winning on time are not the
        same news -- while offline it is always the plain flag fall
        """
        if self._flag_fall_played:
            return
        clock = self.match.clock
        if clock is None or clock.flagged is None:
            return
        self._flag_fall_played = True
        local = self.match.local_color
        if self.variant == Variant.ONLINE and local is not None and clock.flagged != local:
            self.app.sound_manager.play_you_win()
        else:
            self.app.sound_manager.play_flag_fall()

    def relayout(self, size: tuple[int, int]) -> None:
        """
        Rebuild every rect this screen owns for a new window size, including
        the smaller focus-mode footprint. The shell calls it at startup, on a
        resize and on every screen switch, so it must work while the screen is
        not even showing

        :param size: window width and height in pixels
        """
        window_width, window_height = size
        r = compute_layout(
            window_width, window_height, mode=self.name, focus_mode=self.focus_mode,
            focus_show=self._focus_show(), board_size=BOARD_SIZE)
        self.board.set_rect(r.board_rect, scale=r.scale)
        self.right_menu.set_rect(r.menu_rect, scale=r.scale)
        self.player_strip_top.set_rect(r.top_strip_rect, scale=r.scale)
        self.player_strip_bottom.set_rect(r.bottom_strip_rect, scale=r.scale)
        self.result_menu.set_rect(r.result_rect)
        self._refresh_skillcheck_geometry()
        refresh_capture_icons(self.board, r.strip_height,
                              (self.player_strip_top, self.player_strip_bottom))

    def _refresh_skillcheck_geometry(self) -> None:
        """
        Move a live skill-check overlay onto the board's new geometry after a
        resize or a flip, so the dial stays over the square it belongs to. The
        remembered mole impact point is dropped, since it was in old pixels
        """
        self.skillcheck_session.clear_whack_impact_px()
        target = self.skillcheck_session.skillcheck_target
        if self.skillcheck_overlay.is_active() and target is not None:
            self.skillcheck_overlay.relayout(self.board.cell_rect(target))
            self.skillcheck_overlay.set_board_rect(self.board.rect)

    def draw(self) -> None:
        """
        Paint the game: backdrop, board, strips and rail, plus whatever is on
        top of them -- the focus transition, the checkmate takeover, the fade
        behind the result menu and the result menu itself. Focus mode drops
        the rail and the strips and leaves the board alone on screen
        """
        app = self.app
        now = pg.time.get_ticks()
        self._board_needs_present = self.board.consume_needs_present()
        if self.current_result() is not None and self.focus_mode:
            if self.focus_transition is None:
                self._toggle_focus(False)
            elif self.focus_transition.collapsing:
                self._force_focus_off_instant()
        if self.focus_transition is not None:
            tr = self.focus_transition
            tr.draw(now)
            if self._focus_show() == "line" and tr.cur_board_rect is not None:
                alpha = tr.cur_e if tr.collapsing else 1.0 - tr.cur_e
                self._draw_time_lines(tr.cur_board_rect, alpha)
            if tr.done(now):
                self._finalize_focus_transition()
        elif self.focus_mode:
            show = self._focus_show()
            self._draw_game_scene(
                show_panel=False, show_strips=(show == "strips"),
                arrow_hook=lambda: self._draw_focus_edge_arrow(now),
                after_board=self._draw_time_lines if show == "line" else None)
            self.board.draw_drag_overlay()
            if self.board.effects.has_takeover():
                self.board.effects.draw_takeover(self.window, now)
            self.result_flow.update_result_pending()
        else:
            self._draw_game_scene(
                show_panel=True, show_strips=True,
                arrow_hook=lambda: self._update_focus_arrow_off(now))
            self.board.draw_drag_overlay()
            self.result_flow.update_result_pending()
            if not app.coordinator.match_found_modal.is_visible():
                show_modal = self._result_menu_should_show()
                if not show_modal and self.board.effects.has_takeover():
                    self.board.effects.draw_takeover(self.window, now)
                else:
                    self._draw_result_fade_overlay()
                if show_modal:
                    self.board.effects.clear_takeover()
                    self.result_flow.feed_result_menu()
                    self.result_menu.draw()

    def current_result(self) -> str | None:
        """
        The result code for this game, whether the engine found it or a
        player, a clock or the server settled it. This is the single question
        the whole screen asks about whether the game is over

        :returns: the result code, or None while the game is still running
        """
        return self.result_flow.current_result()

    def game_live(self) -> bool:
        """
        Whether the game is still being played, which gates the clock, the
        premove queue and the online-only signals on the rail

        :returns: True while there is no result yet
        """
        return self.current_result() is None

    def board_interactive(self) -> bool:
        """
        Whether the board still accepts play. Once a result lands the pieces
        stop responding, though browsing the move list stays available

        :returns: True while moves and offers can still be made
        """
        return self.current_result() is None

    def swallows_input(self) -> bool:
        """
        Whether a live skill check is grabbing every event, which it does
        while the player is shooting. The passive spectate mirror does not
        swallow anything -- there is nothing for the watcher to aim

        :returns: True while every event should go to the overlay instead
        """
        return self.skillcheck_session.skillcheck_swallows_input()

    def forward_swallowed_event(self, event: pg.event.Event) -> None:
        """
        Hand a raw event to the live skill-check overlay while it is
        swallowing input, which is how a shot is aimed and fired

        :param event: pygame event, keyboard or mouse
        """
        self.skillcheck_overlay.handle_event(event)

    def modals(self) -> list[ModalSpec]:
        """
        The modals this screen owns. Only the shared help modal, which the
        shell merges after the global ones for Esc, clicks and draw order

        :returns: this screen's modal specs
        """
        return [ModalSpec(self.app.help_modal)]

    def scrollables(self) -> list[RightMenu]:
        """
        Every scrollable widget on this screen, so the shell can cancel all
        scrolling at once when the window is resized

        :returns: the right rail, the only thing here that scrolls
        """
        return [self.right_menu]

    def caption(self) -> str:
        """
        The window title while a game is on. Online games name the opponent so
        the title bar is useful when the window is not in front; every other
        variant keeps the default app title

        :returns: caption text, or empty to keep the default title
        """
        if self.variant == Variant.ONLINE:
            opp_name = self.white_name if self._chosen_side == "black" else self.black_name
            return f"Chess — vs {opp_name}"
        return ""

    def debug_state(self) -> dict[str, Any]:
        """
        Summarise the game for the crash log: how far it has got, how it
        ended, which variant it is and whether the player was browsing or in
        focus mode. Small plain values only, and nothing secret

        :returns: readable key/value summary of this screen's state
        """
        return {
            "move_history_len": len(self.match.move_history),
            "result": self.current_result(),
            "variant": self.variant,
            "review_ply": self.board.review_ply,
            "focus_mode": self.focus_mode,
            "match_session_id": self._match_session_id,
        }

    def _forces_full_redraw(self) -> bool:
        """
        Whether anything on screen is moving in a way partial presents cannot
        follow -- a focus transition, a skill check, the result sequence, a
        drag, board effects or the promotion picker. When it is, the frame is
        presented whole rather than by rects

        :returns: True when this frame must present the entire window
        """
        return (self.focus_transition is not None
                or self.skillcheck_overlay.is_active()
                or self.current_result() is not None
                or self.give_time.holding
                or self.board.is_dragging()
                or self.board.effects.is_active()
                or self.board.is_restoring()
                or self.board.pending_promotion_square is not None)

    def dirty_rects(self) -> list[pg.Rect] | None:
        """
        Report the regions that changed this frame so the shell presents only
        those: the strips and rail always, plus the board while a move is
        animating or settling, the focus arrow, live speech bubbles and the
        focus-mode time lines

        :returns: rects to present, or None to force a full present
        """
        if self._forces_full_redraw():
            return None
        rects = [
            self.player_strip_top.rect,
            self.player_strip_bottom.rect,
            self.right_menu.outer_rect,
        ]
        now = pg.time.get_ticks()
        if (self.board.is_animating()
                or now - self.board.last_animation_completed_at_ms < PRESENT_SETTLE_MS):
            rects.append(self.board.animation_dirty_rect())
        if self.focus_arrow.is_visible():
            rects.append(self.focus_arrow.dirty_rect())
        for bubble in self.speech_bubbles.values():
            if bubble.last_rect is not None:
                rects.append(bubble.last_rect)
        if self._board_needs_present:
            rects.append(self.board.rect)
        if self.focus_mode and self._focus_show() == "line":
            top_line, bottom_line = self.time_line.rects_for(self.board, self.board.rect)
            rects.extend([top_line, bottom_line])
        return rects

    def escape(self) -> bool:
        """
        Handle Esc as a context Back or Cancel; it never closes the window
        from here. The ladder is fixed: a live skill check swallows it, then
        focus mode turns off, then a running game opens the resign confirm,
        and a finished game goes back to the menu

        :returns: True, since Esc is always dealt with on this screen
        """
        if self.swallows_input():
            return True
        if not self.board_interactive():
            self.app._on_back_to_menu()
            return True
        if self.focus_mode or self.focus_transition is not None:
            self._toggle_focus(False)
            return True
        self._on_resign()
        return True

    def active_scrollable(self) -> RightMenu | None:
        """
        Name the widget that should take wheel and drag scrolling right now.
        The move log scrolls during play, but not while the result menu is up
        or while the board is alone in focus mode

        :returns: the right rail, or None when nothing here should scroll
        """
        if self._result_menu_should_show():
            return None
        if self.focus_mode or self.focus_transition is not None:
            return None
        return self.right_menu

    def handle_key(self, event: pg.event.Event) -> bool:
        """
        Run the game hotkeys: promotion picks, focus mode, help, flip, resign,
        draw, undo, give time, the rail sections, and arrows or Home and End
        to browse the moves already played. Browsing happens on the live board
        itself -- an accent border round the frame is the cue that you are off
        live, and the board snaps back on its own when the next move lands

        :param event: pygame KEYDOWN event, key code and unicode both readable
        :returns: True when the key was consumed here
        """
        if self._handle_promotion_key(event):
            return True
        if event.key == pg.K_h:
            self._toggle_focus(not self.focus_mode)
            return True
        if getattr(event, "unicode", "") == "?":
            self.app.help_modal.show(HOTKEYS)
            return True
        if event.key == pg.K_f:
            self._on_flip()
            return True
        if event.key == pg.K_r:
            self._on_resign()
            return True
        if event.key == pg.K_d:
            self._on_draw()
            return True
        if event.key == pg.K_z:
            self._on_undo()
            return True
        if event.key == pg.K_g:
            self.give_time.tap_give_time()
            return True
        if event.key == pg.K_a:
            self.right_menu.toggle_section("actions")
            return True
        if event.key == pg.K_s:
            self.right_menu.toggle_section("signals")
            return True
        if event.key == pg.K_c and self.right_menu.chat_visible_provider():
            self.right_menu.toggle_section("chat")
            return True
        if event.key == pg.K_LEFT:
            self.board.step_review(-1)
            return True
        if event.key == pg.K_RIGHT:
            self.board.step_review(1)
            return True
        if event.key == pg.K_HOME:
            self.board.jump_to_review_ply(0)
            return True
        if event.key == pg.K_END:
            self.board.jump_to_review_ply(None)
            return True
        return False

    def _handle_promotion_key(self, event: pg.event.Event) -> bool:
        """
        Let Q, R, B and N pick the promotion piece while the picker is up.
        It runs before every other hotkey, so those letters mean the promotion
        and nothing else for as long as the choice is pending

        :param event: pygame KEYDOWN event
        :returns: True when the key chose a promotion piece
        """
        if self.board.pending_promotion_square is None:
            return False
        chosen = PROMOTION_KEYS.get(event.key)
        if chosen is None:
            return False
        self.board.pick_promotion(chosen)
        return True

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """
        Route a left click to whatever is under it: the focus arrow, the
        result menu, the rail, the promotion picker or the board. A click
        during the result fade skips straight to the menu, and a click on the
        board clears the drawn marks first and only then snaps back to live,
        so leaving a browse never wipes marks and moves in one go

        :param pos: click position in window pixels
        :returns: True when the click was consumed here
        """
        self._focus_click_consumed = False
        if self.focus_transition is not None:
            return False
        if (self.current_result() is not None
                and self._result_first_seen_at_ms is not None
                and not self._result_menu_should_show()):
            self._skip_result_fade()
            return False
        if (self.focus_arrow.is_visible() and self._focus_arrow_allowed()
                and self.focus_arrow.handle_click(pos)):
            self._toggle_focus(not self.focus_mode)
            self._focus_click_consumed = True
            self.app.input_router.suppress_click_sound()
            return True
        if not self.focus_mode and self.result_menu.handle_click(pos):
            return True
        if not self.focus_mode and self.right_menu.handle_click(pos):
            return True
        if self.current_result() is not None:
            return False
        if self.board.pending_promotion_square is not None:
            self.board.pick_promotion_at(pos)
            self.app.input_router.suppress_click_sound()
            return True
        square = self.board.cell_at(pos)
        if square is None:
            return False
        had_marks = bool(self.board.highlighted_squares or self.board.arrows)
        if self.board.review_ply is None and not self.board.is_square_annotated(square):
            self.board.clear_annotations()
        signal = self.board.handle_click(square)
        self._maybe_sync_marks_cleared(had_marks)
        if signal:
            self.app.input_router.suppress_click_sound()
            if signal == "select":
                self.app.sound_manager.play_pickup()
            return True
        return False

    def handle_press(self, pos: tuple[int, int]) -> bool:
        """
        Begin a left press on the board, which is where picking a piece up
        starts. It is refused once the game is over, while the promotion
        picker is open, during a focus transition, and on the same click that
        just worked the focus arrow

        :param pos: press position in window pixels
        :returns: True when the board took the press
        """
        if (self.current_result() is None
                and self.board.pending_promotion_square is None
                and not self._focus_click_consumed
                and self.focus_transition is None):
            self.board.begin_press(pos)
            return True
        return False

    def handle_motion(self, pos: tuple[int, int]) -> bool:
        """
        Follow the cursor so a dragged piece keeps up with it, which is what
        gives the drag its weight and swing

        :param pos: cursor position in window pixels
        :returns: True, since the board always takes the motion
        """
        self.board.update_drag_motion(pos)
        return True

    def handle_release(self, pos: tuple[int, int]) -> bool:
        """
        Drop a dragged piece. The release is replayed as a board click so a
        drag and a click land a move by exactly the same path, and only a real
        drag in a running game gets the drop sound

        :param pos: release position in window pixels
        :returns: True, since the board always takes the release
        """
        was_dragging = self.board.dragging_from is not None
        if was_dragging and self.current_result() is None:
            self.app.input_router.mouse_left_clicked(pos, ui_click=False)
        self.board.end_press()
        if was_dragging and self.current_result() is None:
            self.app.sound_manager.play_drop()
        return True

    def handle_right_press(self, pos: tuple[int, int]) -> bool:
        """
        Start a right-button gesture on the board, which draws a highlight or
        an arrow. Right-clicking while dragging queues a premove to that
        square instead, so a plan can be lined up during the opponent's turn

        :param pos: press position in window pixels
        :returns: True, since the screen always answers the right press
        """
        if self.current_result() is not None:
            self.app.sound_manager.play_ui_click()
            return True
        sq = self.board.cell_at(pos)
        if sq is None:
            self.app.sound_manager.play_ui_click()
            return True
        if self.board.dragging_from is not None:
            if not self.board.queue_premove_from_drag(sq):
                self.app.sound_manager.play_ui_click()
            return True
        self.board.begin_right_press(pos)
        self.app.sound_manager.play_ui_click()
        return True

    def handle_right_release(self, pos: tuple[int, int]) -> bool:
        """
        Finish a right-button gesture and commit the mark it drew. When
        sharing is on, that one mark is also sent to the opponent as a delta
        rather than resending the whole set

        :param pos: release position in window pixels
        :returns: True, since the screen always answers the right release
        """
        delta = self.board.end_right_press(pos)
        if (delta is not None and delta[-1] is not None
                and self._share_marks and self._share_active()):
            self._send_annotation_delta(delta)
        return True

    def _share_active(self) -> bool:
        """
        Whether marks can travel at all right now. There has to be an opponent
        and a running game, and moderation must not have muted this player's
        sharing for the rest of the game

        :returns: True when a mark may be sent to the opponent
        """
        return (self.variant == Variant.ONLINE and self.current_result() is None
                and not self._share_muted)

    def _toast_share_muted(self) -> None:
        """
        Tell the player their mark sharing has been muted for this game, the
        one explanation for why the share control no longer does anything
        """
        self.app.toast.show(ONLINE_TRANSIENT_REASON_LABELS[Reason.SHARE_MUTED],
                            key="marks_muted")

    def _on_share_blocked(self) -> None:
        """
        Answer a click on the share control while it is switched off. Only a
        moderation mute is worth explaining -- the control is also off simply
        because the game is offline or finished, which needs no toast
        """
        if self._share_muted:
            self._toast_share_muted()

    def _on_share_toggle(self) -> None:
        """
        Turn mark sharing on or off from the rail. Switching it on sends the
        marks already on the board so the opponent sees the whole picture at
        once; switching it off sends an empty set, which clears their copy
        """
        if not self._share_active():
            return
        self._share_marks = not self._share_marks
        log.info("share marks toggled on=%s", self._share_marks)
        coordinator = self.app.coordinator
        if self._share_marks:
            coordinator.send_annotations_state(
                True,
                sorted(coord_from_square(sq) for sq in self.board.highlighted_squares),
                [(coord_from_square(f), coord_from_square(t))
                 for f, t in self.board.arrows])
        else:
            coordinator.send_annotations_state(False, [], [])

    def _send_annotation_delta(self, delta: tuple[Any, ...]) -> None:
        """
        Send the one mark that just changed to the opponent, which is far
        cheaper than resending every mark on the board after each gesture

        :param delta: what the board reported -- either ("highlight", square,
            added) or ("arrow", from, to, added), the last item saying whether
            the mark appeared or was removed
        """
        action = "add" if delta[-1] else "remove"
        coordinator = self.app.coordinator
        if delta[0] == "highlight":
            coordinator.send_annotation_delta(
                action, "highlight", square=coord_from_square(delta[1]))
        else:
            coordinator.send_annotation_delta(
                action, "arrow", from_sq=coord_from_square(delta[1]),
                to_sq=coord_from_square(delta[2]))

    def _maybe_sync_marks_cleared(self, had_marks: bool) -> None:
        """
        Tell the opponent when a board click has just wiped the shared marks,
        since that clearing is not a gesture and produces no delta of its own.
        Without it their copy would keep showing marks that are gone here

        :param had_marks: whether this client had marks drawn before the click
        """
        if (had_marks and self._share_marks and self._share_active()
                and not (self.board.highlighted_squares or self.board.arrows)):
            self.app.coordinator.send_annotations_state(True, [], [])

    def _draw_game_scene(self, *, show_panel: bool, show_strips: bool,
                         arrow_hook: Callable[[], None] | None = None,
                         after_board: Callable[[], None] | None = None) -> None:
        """
        Paint the game itself in back-to-front order -- backdrop, board,
        speech bubbles, strips, rail -- with the pieces a live skill check is
        hiding synced first. The normal view shows everything; focus mode
        drops the rail and, depending on the setting, the strips as well

        :param show_panel: draw the right rail and refresh its game info
        :param show_strips: draw both player strips
        :param arrow_hook: called after the strips to place the focus arrow,
            which differs between focus mode and the normal view
        :param after_board: called straight after the board, used for the
            focus-mode time lines
        """
        self.skillcheck_session.sync_aim_victim()
        self.skillcheck_session.sync_whack_gun()
        self._draw_game_background()
        self.board.draw_board()
        self._draw_speech_bubbles()
        if after_board is not None:
            after_board()
        if show_strips:
            self._update_player_strips()
            self.player_strip_top.draw()
            self.player_strip_bottom.draw()
        if show_panel:
            self._refresh_game_info()
        if arrow_hook is not None:
            arrow_hook()
        if show_panel:
            self.right_menu.draw_menu()

    def show_speech_bubble(self, color: str, text: str) -> None:
        """
        Pop a quick-chat line above one player's pieces for a few seconds.
        Both the sender and the receiver come through here, so a line always
        looks the same on both screens

        :param color: whose bubble this is, "white" or "black"
        :param text: the preset line to show
        """
        self.speech_bubbles[color].show(text, pg.time.get_ticks())

    def _draw_speech_bubbles(self) -> None:
        """
        Draw the live quick-chat bubbles over the board. They are hidden while
        a blocking modal or the result menu is up, and a bubble whose anchor
        piece has been captured simply does not draw
        """
        now = pg.time.get_ticks()
        if self.app._blocking_modal_visible() or self._result_menu_should_show():
            for bubble in self.speech_bubbles.values():
                bubble.last_rect = None
            return
        scale = self.board.scale
        for color, bubble in self.speech_bubbles.items():
            if not bubble.active(now):
                bubble.last_rect = None
                continue
            anchor = self._speech_anchor(color)
            if anchor is None:
                bubble.last_rect = None
                continue
            bubble.draw(self.window, anchor, self.board.rect, now, scale=scale)

    def _speech_anchor(self, color: str) -> pg.Rect | None:
        """
        Find the square a player's speech bubble should sit over, memoised so
        the board is not searched every frame. The memo is keyed on the ply
        and the browse position, both of which move the pieces

        :param color: whose bubble is being placed, "white" or "black"
        :returns: the anchor square's rect in window pixels, or None when that
            player has no piece to anchor to
        """
        key = (color, self.board.review_ply, len(self.match.move_history))
        cached = self._speech_anchor_memo.get(color)
        if cached is None or cached[0] != key:
            cached = (key, self._speech_anchor_square(color))
            self._speech_anchor_memo[color] = cached
        target = cached[1]
        if target is None:
            return None
        return self.board.cell_rect(target)

    def _speech_anchor_square(self, color: str) -> Square | None:
        """
        Pick the piece a player's bubble hangs off: their queen if she is
        still about, otherwise their king. It reads the browsed position while
        the player is looking back, so the bubble sits where the pieces are

        :param color: whose piece is wanted, "white" or "black"
        :returns: the square to anchor to, or None when neither piece is there
        """
        pcolor = PieceColor.WHITE if color == "white" else PieceColor.BLACK
        if self.board.review_ply is not None:
            grid = self.match.position_at(self.board.review_ply)
        else:
            grid = self.match.state
        target = piece_square(grid, PieceType.QUEEN, pcolor)
        if target is None:
            target = king_square(grid, pcolor)
        return target

    def _draw_game_background(self) -> None:
        """
        Lay down the arena backdrop behind everything else, lit around the
        board so the eye is pulled to the middle of the window
        """
        self.backdrop.draw(self.window, self.board.rect)

    def _refresh_game_info(self) -> None:
        """
        Keep the rail's header -- variant pill, time control, round number and
        the series line -- in step with the game, rebuilding it only when one
        of those actually changes rather than every frame
        """
        key = (self.app.screen is self, self.variant, self._time_control,
               self.white_name, self.black_name,
               self.result_flow.series_scores.get(self.white_name, 0.0),
               self.result_flow.series_scores.get(self.black_name, 0.0))
        if self._game_info_memo is not None and self._game_info_memo[0] == key:
            return
        info = self._compute_game_info()
        self._game_info_memo = (key, info)
        if info != self.right_menu.game_info:
            self.right_menu.set_game_info(info)

    def _compute_game_info(self) -> dict[str, Any] | None:
        """
        Build the rail header block for this game: the variant pill, the time
        control, the round number and, online only, the running series score
        between these two players

        :returns: the header block, or None when this screen is not the one
            showing, which clears the rail instead
        """
        if self.app.screen is not self:
            return None
        tc = format_time_control(self._time_control) or "∞"
        rnd = self._current_round()
        pill = VARIANT_PILL_LABELS[self.variant]
        info = {"mode": pill, "time_control": tc, "round": rnd, "lines": []}
        if self.variant == Variant.ONLINE:
            white_score = self.result_flow.series_scores.get(self.white_name, 0.0)
            black_score = self.result_flow.series_scores.get(self.black_name, 0.0)
            info["lines"] = [
                f"{self.white_name}  {score_str(white_score)} – "
                f"{score_str(black_score)}  {self.black_name}",
            ]
        return info

    def _current_round(self) -> int:
        """
        Which game of the series this is, counted from the points already
        shared out between the two players. Offline games have no series, so
        this stays at one

        :returns: the round number, starting at one
        """
        total = (self.result_flow.series_scores.get(self.white_name, 0.0)
                 + self.result_flow.series_scores.get(self.black_name, 0.0))
        return int(total) + 1

    def _result_elapsed_ms(self) -> int | None:
        """
        How long the ending has been on screen, which drives the fade and the
        wait before the result menu appears. The stamp is only taken once the
        last move has visually settled, so the timing starts when the player
        actually sees the finish

        :returns: milliseconds since the result was first shown, or None while
            the game has no result yet
        """
        if self._result_first_seen_at_ms is None:
            return None
        return pg.time.get_ticks() - self._result_first_seen_at_ms

    def _result_menu_should_show(self) -> bool:
        """
        Whether the result menu has waited long enough to come up. A plain
        ending gets a short beat; a checkmate waits for the full takeover
        animation, so the celebration is never cut off by the menu

        :returns: True once the result menu should be drawn and clickable
        """
        elapsed = self._result_elapsed_ms()
        if elapsed is None:
            return False
        delay = RESULT_TAKEOVER_MS if self.board.effects.has_takeover() else RESULT_MODAL_DELAY_MS
        return elapsed >= delay

    def _result_fade_surface(self, size: tuple[int, int]) -> pg.Surface:
        """
        The black sheet the ending fades in behind the result menu, cached per
        window size so a full-window surface is not built every frame

        :param size: window width and height in pixels
        :returns: the shared black surface for that size
        """
        def build() -> pg.Surface:
            """
            Make the black sheet the first time this size is asked for

            :returns: an opaque black surface of the requested size
            """
            surf = pg.Surface(size)
            surf.fill((0, 0, 0))
            return surf
        return cache.memoized_surface(_RESULT_FADE_CACHE, size, build)

    def _draw_result_fade_overlay(self) -> None:
        """
        Dim the board as the game ends, easing in over a fraction of a second
        and stopping short of black so the final position stays readable
        behind the result menu
        """
        elapsed = self._result_elapsed_ms()
        if elapsed is None:
            return
        alpha = min(RESULT_FADE_MAX_ALPHA,
                    int(RESULT_FADE_MAX_ALPHA * elapsed / RESULT_FADE_MS))
        if alpha <= 0:
            return
        overlay = self._result_fade_surface(self.window.get_size())
        overlay.set_alpha(alpha)
        self.window.blit(overlay, (0, 0))

    def _skip_result_fade(self) -> None:
        """
        Let an impatient click cut straight to the result menu by backdating
        the moment the ending appeared and clearing the takeover animation.
        Nothing about the result itself changes
        """
        if self._result_first_seen_at_ms is None:
            return
        self.board.effects.clear_takeover()
        self._result_first_seen_at_ms = pg.time.get_ticks() - RESULT_MODAL_DELAY_MS

    def _trigger_result_effects(self) -> None:
        """
        Celebrate or commiserate the ending: the loser's king raises a white
        flag on a mate or a resignation, a mate also takes the screen over,
        and the sound follows whether this player won. Timeouts and the quiet
        draws get no flourish beyond their own sound
        """
        code = self.current_result()
        if code is None:
            return
        if code.startswith("draw"):
            self.app.sound_manager.play_draw()
            return
        if code.startswith("white_wins"):
            winner, loser = PieceColor.WHITE, PieceColor.BLACK
        elif code.startswith("black_wins"):
            winner, loser = PieceColor.BLACK, PieceColor.WHITE
        else:
            return
        is_mate = code in ("white_wins", "black_wins")
        is_resign = code.endswith("_by_resignation")
        if is_mate or is_resign:
            self.board.show_surrender_flag(loser)
        if is_mate:
            self.board.show_checkmate_takeover(winner.value.upper())
        if is_mate or is_resign:
            if self._local_won(winner):
                self.app.sound_manager.play_you_win()
            elif is_resign:
                self.app.sound_manager.play_resign()
            elif is_mate:
                self.app.sound_manager.play_you_lose()

    def _local_won(self, winner: PieceColor) -> bool:
        """
        Whether the win belongs to the person at this keyboard, which decides
        between the victory and the defeat sound. Offline both sides are
        played here, so any win counts as the local one

        :param winner: colour that won the game
        :returns: True when this client should hear the victory sound
        """
        return self.variant != Variant.ONLINE or self.match.local_color == winner

    def _on_flip(self) -> None:
        """
        Turn the board round so the other colour is at the bottom. Any drag in
        flight is dropped first and a live skill check is moved onto the new
        geometry, so nothing is left pointing at the old orientation
        """
        if self.current_result() is not None:
            return
        self.board.cancel_drag_physics()
        self.board.flipped = not self.board.flipped
        self._refresh_skillcheck_geometry()
        self.app.sound_manager.play_flip()

    def _on_resign(self) -> None:
        """
        Ask before giving the game up, since resigning cannot be undone. This
        is also where Esc lands during a live game, so the confirm is the last
        thing between a stray key and a lost game
        """
        if not self.board_interactive():
            return
        self.app.confirm_modal.show(
            "Tap out?", on_yes=self._perform_resign,
            sub="The pieces are watching.",
            yes_label="I'm done", no_label="Keep fighting", danger=True, emoji="🏳️",
        )

    def _perform_resign(self) -> None:
        """
        Actually resign, once the player has confirmed. Online it goes to the
        server and the ending comes back from there; offline the side to move
        loses on the spot, with a half-finished promotion tidied up first so
        the saved game is not left mid-ply
        """
        if self.current_result() is not None:
            return
        if self.variant == Variant.ONLINE and self.app.coordinator.send_resign():
            return
        loser = self.match.current_turn()
        self._auto_complete_pending_promotion()
        self.manual_result = (
            "black_wins_by_resignation" if loser == PieceColor.WHITE
            else "white_wins_by_resignation"
        )
        self.board.clear_premoves()
        self.board.clear_annotations()
        self.result_flow.on_result_final(self.manual_result)

    def _on_draw(self) -> None:
        """
        Ask before offering a draw. Online the offer goes to the opponent and
        may well be refused; offline it ends the game immediately, which is
        why it is worth a confirm either way
        """
        if not self.board_interactive():
            return
        self.app.confirm_modal.show(
            "Offer a draw?", on_yes=self._perform_draw,
            sub="Propose splitting the point. No shots fired, no glory either.",
            yes_label="Offer draw", no_label="Nevermind", emoji="🤝",
        )

    def _perform_draw(self) -> None:
        """
        Act on a confirmed draw. Online it is only an offer, and the game goes
        on until the opponent answers; offline both sides sit here, so the
        draw is agreed at once
        """
        if self.current_result() is not None:
            return
        if self.variant == Variant.ONLINE and self.app.coordinator.send_draw_offer():
            return
        self._auto_complete_pending_promotion()
        self.manual_result = "draw_agreement"
        self.board.clear_premoves()
        self.board.clear_annotations()
        self.result_flow.on_result_final(self.manual_result)

    def _auto_complete_pending_promotion(self) -> None:
        """
        Settle a promotion the player never picked, so a game that ends with
        the picker open is saved as a whole position. A promotion whose move
        has not been applied yet is simply cancelled; one already on the board
        becomes a queen
        """
        if self.board.pending_promotion_square is None:
            return
        if self.board.cancel_unapplied_promotion():
            return
        self.match.promote(self.board.pending_promotion_square, PieceType.QUEEN)
        self.board.pending_promotion_square = None

    def _on_undo(self) -> None:
        """
        Take the last move back. Online this can only be a request the
        opponent has to allow; offline it happens straight away, dropping the
        selection, the premoves, the move locks, the marks and any browse
        position so the board is honest about where the game now stands
        """
        if not self.board_interactive():
            return
        if self.variant == Variant.ONLINE and self.app.coordinator.send_takeback_request():
            return
        self.board.selected_square = None
        self.board.clear_premoves()
        self.skillcheck.clear_locks()
        self.board.clear_annotations()
        self.board.review_ply = None
        self.board.cancel_animations()
        if not self.match.move_history:
            return
        log.info("undo ply=%d", len(self.match.move_history))
        self.app.sound_manager.play_undo()
        self.skillcheck_session.drop_skillcheck_log_from(len(self.match.move_history))
        move = self.match.move_history[-1].move
        self.match.undo()
        self.board.start_undo_animation(move)

    def _on_move_landed(self, entry: HistoryEntry) -> None:
        """
        React to a ply that has finished landing on the board: play the move,
        castle, check or mate sound, show the gun on a check, flash the
        increment on the mover's clock and drop the skill-check move locks,
        which only ever bound for the turn just played

        :param entry: the history entry for the ply, complete with its SAN and
            its check and mate flags
        """
        self._idle_window = None
        if entry.gives_checkmate:
            self.app.sound_manager.play_checkmate()
        elif entry.move.is_castle:
            self.app.sound_manager.play_castle()
        else:
            self.app.sound_manager.play_move(entry.move.piece.type)
        if entry.gives_check and not entry.gives_checkmate:
            self.app.sound_manager.play_check()
            self.board.show_check_gun(entry)
        self._maybe_flash_increment_for(entry.move.piece.color)
        self.skillcheck.clear_locks()

    def _on_shot_fired(self, entry: HistoryEntry) -> None:
        """
        Fire the capturing piece's weapon sound at the exact frame the board's
        capture choreography pulls the trigger, so the bang lands with the
        muzzle flash rather than with the move

        :param entry: the history entry for the capturing ply
        """
        self.app.sound_manager.play_capture(entry.move.piece.type)

    def _on_kill_announced(self, key: str, victim: Piece | None = None) -> None:
        """
        Voice a kill the board just registered -- the hit thud, or the
        announcer calling a streak. The announcer is kept quiet once the game
        is over, so a mating capture is not talked over by a killing spree

        :param key: what the board registered, "hit" or an announcer line key
        :param victim: piece that was taken, used to pick the hit sound; None
            when the board has none to report
        """
        if key == "hit":
            self.app.sound_manager.play_hit(victim.type if victim is not None else None)
        elif self.current_result() is None:
            self.app.sound_manager.play_announcer(key)

    def _maybe_flash_increment_for(self, mover_color: PieceColor) -> None:
        """
        Flash the mover's clock when their increment is added, so the time
        going back on is visible. Games without an increment, and untimed
        games, flash nothing

        :param mover_color: colour that just completed a move
        """
        clock = self.match.clock
        if clock is None or clock.increment_seconds <= 0:
            return
        self._strip_for_color(mover_color).flash_increment()

    def _name_for_color(self, color: PieceColor | str) -> str:
        """
        The display name of whoever is playing a colour, used by the strips,
        the toasts and the series score

        :param color: the colour to look up, as a piece colour or as "white"
            or "black"
        :returns: that player's display name
        """
        return self.white_name if is_white(color) else self.black_name

    def _country_for_color(self, color: PieceColor | str) -> str:
        """
        The country code flying on a colour's strip

        :param color: the colour to look up, as a piece colour or as "white"
            or "black"
        :returns: the country code, empty when that player has none set
        """
        return self.white_country if is_white(color) else self.black_country

    def _strip_for_color(self, color: PieceColor | str) -> PlayerStrip:
        """
        Find the strip a colour is currently shown on, which depends on how
        the board is turned. Callers that flash a clock or a granted second
        need the strip, not the colour

        :param color: the colour to place, as a piece colour or as "white" or
            "black"
        :returns: the top or bottom strip, whichever that colour occupies
        """
        top_is_white = top_strip_color(self.board.flipped) == PieceColor.WHITE
        return (self.player_strip_top if is_white(color) == top_is_white
                else self.player_strip_bottom)

    def _update_player_strips(self) -> None:
        """
        Push this frame's state into both player strips, working out which
        colour is on top from the board orientation. Everything the strips
        show comes through here -- clocks, captures, connection dot and the
        auto-end countdown
        """
        top_color = top_strip_color(self.board.flipped)
        bottom_color = opponent_of(top_color)
        turn = self.match.current_turn()
        over = self.current_result() is not None
        self.player_strip_top.set_state(**self._strip_state(top_color, turn, over))
        self.player_strip_bottom.set_state(**self._strip_state(bottom_color, turn, over))

    def _reviewed_sans(self) -> list[str]:
        """
        The moves as far as the player is currently looking, in SAN. The
        opening lookup reads this, so browsing back through the game names the
        opening as it stood at that point

        :returns: SAN strings up to the browsed position, or the whole game
            when the board is live
        """
        return [entry.san for entry in self.board.reviewed_history()]

    def _debut_line(self) -> tuple[str, str] | None:
        """
        Name the opening being played, for the line under the rail's shot-log
        header. A game started from a custom position has no opening to name

        :returns: the ECO code and opening name, or None when there is no
            match or the game did not start from the initial position
        """
        if self._custom_start:
            return None
        return self._debut.line((len(self.match.move_history), self.board.review_ply))

    def _strip_capture_summary(self, color: PieceColor) -> tuple[list[PieceType], int]:
        """
        What a player has taken and how far ahead they are on material, for
        their strip. It is memoised per colour and recomputed only when the
        ply or the browsed position moves, since it walks the whole history

        :param color: the capturing side
        :returns: the piece types they have taken and their material lead,
            the lead being zero or less when they are not ahead
        """
        key = (len(self.match.move_history), self.board.review_ply, color)
        cached = self._strip_memo.get(color)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        history = self.board.reviewed_history()
        captured = captured_by(history, color)
        advantage = material_advantage(history, color)
        self._strip_memo[color] = (key, captured, advantage)
        return captured, advantage

    def _strip_state(self, color: PieceColor, turn: PieceColor, over: bool) -> dict[str, Any]:
        """
        Build everything one player's strip shows this frame: name, country,
        clock, captures, material lead, whose turn it is, and the online-only
        extras -- the opponent's connection dot, their sharing pill and any
        auto-end countdown

        :param color: the side this strip belongs to
        :param turn: the side to move, which decides the active highlight
        :param over: whether the game has finished, which stops the highlight
        :returns: keyword state ready to hand straight to the strip
        """
        name = self._name_for_color(color)
        seconds = (self.match.clock.remaining(color)
                   if self.match.clock is not None else None)
        initial_seconds = (self.match.clock.initial_seconds
                           if self.match.clock is not None else None)
        active = (color == turn) and not over
        captured, advantage = self._strip_capture_summary(color)
        connection_state = None
        app = self.app
        if (self.variant == Variant.ONLINE and app.coordinator.is_connected()
                and self.match.local_color is not None
                and color != self.match.local_color):
            connection_state = app.coordinator.opponent_state()
        auto_end_label, auto_end_seconds = self._compute_auto_end(color, over)
        is_bot = self.variant == Variant.BOT and name == OPPONENT_NAME_FOR_MODE[BOT]
        sharing, sharing_color = self._strip_sharing(color)
        return {
            "name": name,
            "player_color": color,
            "is_bot": is_bot,
            "rating": PLACEHOLDER_RATING,
            "clock_seconds": seconds,
            "active": active,
            "captured": captured,
            "advantage": advantage,
            "captured_color": opponent_of(color),
            "ko_count": len(captured),
            "connection_state": connection_state,
            "country": self._country_for_color(color),
            "clock_initial_seconds": initial_seconds,
            "auto_end_label": auto_end_label,
            "auto_end_seconds": auto_end_seconds,
            "sharing": sharing,
            "sharing_color": sharing_color,
        }

    def _strip_sharing(self, color: PieceColor) -> tuple[bool, str | None]:
        """
        Whether a player is broadcasting their marks, and in which colour the
        strip should say so. Each side gets its own accent so the two sets of
        drawings on the board are easy to tell apart

        :param color: the side whose strip is being filled in
        :returns: whether they are sharing and the pill colour to use, the
            colour being None when there is nothing to show
        """
        if self.variant != Variant.ONLINE or self.match.local_color is None:
            return False, None
        if color == self.match.local_color:
            return self._share_marks, Colors.amber
        return self._opp_sharing, Colors.avatar_blue

    def _compute_auto_end(self, color: PieceColor, over: bool) -> tuple[str | None, float | None]:
        """
        Work out the countdown badge on a player's strip when the game is
        heading for an automatic ending: this client reconnecting, the
        opponent's grace period running out, or the server's idle window
        against whoever is sitting on their hands

        :param color: the side whose strip is being filled in
        :param over: whether the game has finished, which hides every badge
        :returns: the badge label and the seconds left, both None when nothing
            is counting down
        """
        if (self.variant != Variant.ONLINE or over or self.match.local_color is None):
            return None, None
        now = pg.time.get_ticks()
        is_local = (color == self.match.local_color)
        if is_local and self._local_disconnected_at_ms is not None:
            return self._auto_end_remaining(
                "Aborting in", self._local_disconnected_at_ms,
                RECONNECT_TOTAL_SECONDS, now,
            )
        if (not is_local
                and self._opp_disconnected_at_ms is not None):
            return self._auto_end_remaining(
                "Abandon in", self._opp_disconnected_at_ms, GRACE_SECONDS, now,
            )
        window = self._idle_window
        if window is not None and color == window.color:
            return self._auto_end_remaining(
                IDLE_LABEL_BY_OUTCOME[window.outcome],
                window.deadline_ms - int(window.total_seconds * 1000),
                window.total_seconds, now,
            )
        return None, None

    def _auto_end_remaining(self, label: str, snap_ms: int, total_seconds: float,
                            now_ms: int) -> tuple[str | None, float | None]:
        """
        Turn a countdown's start stamp into what the badge should read now.
        The first tenth of any window is held back, so a brief blip never
        flashes a scary countdown, and an expired window shows nothing

        :param label: badge wording for this kind of countdown
        :param snap_ms: tick count when the window started
        :param total_seconds: how long the window runs in full
        :param now_ms: current pygame tick count in milliseconds
        :returns: the label and the seconds left, both None while the badge
            should stay hidden
        """
        elapsed = (now_ms - snap_ms) / 1000.0
        if elapsed < AUTO_END_GATE_FRACTION * total_seconds:
            return None, None
        remaining = total_seconds - elapsed
        if remaining <= 0:
            return None, None
        return label, remaining

    def _signals_provider(self) -> dict[str, Any]:
        """
        Report the state of every toggle in the rail's signals section --
        share, hide marks, auto-queen, sound and volume. The two mark toggles
        only mean anything in a running online game, so they report themselves
        as disabled everywhere else

        :returns: the chip states and enabled flags the rail draws from
        """
        sm = self.app.sound_manager
        return {
            "share_on": self._share_marks,
            "share_enabled": (self.variant == Variant.ONLINE and self.game_live()
                              and not self._share_muted),
            "hide_on": env.get_hide_opp_marks(),
            "hide_enabled": self.variant == Variant.ONLINE and self.game_live(),
            "auto_q": env.get_auto_queen(),
            "sound_on": sm.enabled,
            "volume": sm.master_volume,
            "review": False,
        }

    def _toggle_auto_queen(self) -> None:
        """
        Flip the auto-queen setting, which decides whether a promotion opens
        the picker or simply takes a queen. It is a stored preference, so it
        outlives this game
        """
        env.set_auto_queen(not env.get_auto_queen())

    def _on_hide_marks_toggle(self) -> None:
        """
        Flip the hide-opponent-marks setting through the shell, which both
        stores it and tells the server, so the opponent's marks stop being
        sent as well as stop being drawn
        """
        self.app.settings.apply_hide_opp_marks(not env.get_hide_opp_marks())

    def _right_menu_buttons(self) -> list[Cap]:
        """
        Which action buttons the rail should offer. An untimed game drops the
        give-time button, since there is no clock to give from

        :returns: the button definitions for this game
        """
        if self.match.clock is None:
            return RIGHT_MENU_UNTIMED_CAPS
        return RIGHT_MENU_CAPS

    def _right_menu_disabled_keys(self) -> set[str]:
        """
        Which rail buttons should be greyed out right now. A finished game
        disables everything that would change it, and give-time also rests
        during its cooldown and once a clock has flagged

        :returns: the keys of the buttons to disable this frame
        """
        if self.current_result() is not None:
            return {"undo", "resign", "draw", "flip", "give_time"}
        disabled = set()
        clock = self.match.clock
        if clock is None or clock.flagged is not None:
            disabled.add("give_time")
        if self.give_time.on_cooldown():
            disabled.add("give_time")
        return disabled

    def _ensure_local_session(self) -> None:
        """
        Give an offline game an identity the first time it needs one, so its
        saved PGN carries a match id like an online game does. Later games
        keep the same id until an online pairing replaces it
        """
        if self._match_session_id is None:
            self._match_session_id = str(uuid.uuid4())

    def _reset_to_new_game(self) -> None:
        """
        Wipe the screen back to a fresh game: match, clocks, board, rail,
        strips, chat and every skill-check leftover. Every pre-board overlay
        has to be reset here, the result menu included -- a stale outcome once
        left its buttons invisible but still clickable over the new game
        """
        app = self.app
        coordinator = app.coordinator
        self._force_focus_off_instant()
        coordinator.offer_banners.clear()
        coordinator._rematch_offered = False
        self.result_menu.reset()
        app.sound_manager.stop_all()
        self.give_time.cancel_give_time_hold()
        self.manual_result = None
        self._flag_fall_played = False
        self._strip_memo = {}
        self._custom_start = False
        self._debut.reset()
        self._share_marks = False
        self._share_muted = False
        self._opp_sharing = False
        self._chat_cooldown_until_ms = 0
        self._speech_anchor_memo = {}
        self._game_info_memo = None
        self._result_first_seen_at_ms = None
        self._idle_window = None
        self.result_flow.reset_for_new_game()
        self.right_menu.reset_for_new_game()
        self.match.new_game()
        self.match.mode = MATCH_MODE_BY_VARIANT[self.variant]
        if self._time_control is not None:
            initial, incr = self._time_control
            self.match.setup_clock(initial, incr)
            self.board.animation_duration_ms = compute_animation_ms(initial)
        else:
            self.board.animation_duration_ms = ANIM_MS_DEFAULT
        self.board.reset_for_new_game()
        self.skillcheck_overlay.cancel()
        self.skillcheck.clear_locks()
        self.skillcheck_session.clear_online_skillcheck_state()
        self.skillcheck_session.skillcheck_log = []
        app.confirm_modal.hide()
        self._last_turn_for_flip = None
        for bubble in self.speech_bubbles.values():
            bubble.clear()

    def _focus_show(self) -> str:
        """
        What focus mode keeps on screen besides the board, as the player has
        set it: the slim time lines, the full strips, or nothing at all

        :returns: the focus display mode, "line", "strips" or "nothing"
        """
        return env.get_focus_show()

    def _focus_available(self) -> bool:
        """
        Whether focus mode may be entered at this moment. It is barred while
        anything is mid-gesture or demanding attention -- a skill check, a
        drag, the promotion picker, a blocking modal or a finished game

        :returns: True when focus mode can be turned on
        """
        return (self.app.screen is self
                and self.board_interactive()
                and not self.skillcheck_overlay.is_active()
                and not self.board.is_dragging()
                and self.board.pending_promotion_square is None
                and not self.app._blocking_modal_visible())

    def _focus_arrow_allowed(self) -> bool:
        """
        Whether the little focus arrow may show and be clicked. It is a looser
        test than entering focus mode itself, since the arrow also has to be
        available for leaving it

        :returns: True when the arrow should react to the cursor
        """
        return (self.board_interactive()
                and not self.skillcheck_overlay.is_active()
                and not self.app._blocking_modal_visible())

    def _toggle_focus(self, on: bool) -> None:
        """
        Collapse the window down to the board, or expand it back to the full
        game, through an animated transition. Asking for the state it is
        already in does nothing, and a transition already running is left
        alone rather than being restarted

        :param on: True to collapse into focus mode, False to expand out
        """
        if self.focus_transition is not None:
            return
        if self.skillcheck_overlay.is_active() and self.current_result() is None:
            return
        if on == self.focus_mode:
            return
        if on and not self._focus_available():
            return
        log.info("focus mode toggled on=%s", on)
        if on:
            self.app.sound_manager.play_focus_action()
        else:
            self.app.sound_manager.play_focus_action_off()
        self.board.cancel_drag_physics()
        self.focus_transition = FocusTransition(self)
        if on:
            self.focus_transition.begin_collapse(self._focus_show())
        else:
            self.focus_transition.begin_expand(self._focus_show())
        self.focus_arrow.reset()
        self.app._needs_full_present = True

    def _finalize_focus_transition(self) -> None:
        """
        Drop the finished focus transition and ask for one whole frame, since
        the snapshots it was drawing are gone and the real widgets underneath
        have to be painted again
        """
        self.focus_transition = None
        self.app._needs_full_present = True

    def on_resize(self) -> None:
        """
        React to the window surface being replaced, which happens just before
        the relayout pass. A focus transition is animating snapshots of the
        old surface, so it cannot survive and is abandoned
        """
        self._abort_transition_for_resize()

    def _abort_transition_for_resize(self) -> None:
        """
        Abandon a focus transition mid-flight and repaint the window whole.
        The transition draws grabbed pixels of the old window, which mean
        nothing once it has been resized
        """
        if self.focus_transition is not None:
            self.focus_transition.cancel()
            self.focus_transition = None
            self.app._needs_full_present = True

    def _force_focus_off_instant(self) -> None:
        """
        Snap straight back to the full game view with no animation, used when
        the screen is leaving or a new game is starting. Layout is recomputed
        on the spot, so the next screen never inherits the focus geometry
        """
        if not self.focus_mode and self.focus_transition is None:
            return
        if self.focus_transition is not None:
            self.focus_transition.cancel()
            self.focus_transition = None
        self.focus_mode = False
        self.focus_arrow.reset()
        self._focus_panel_hover_ms = LONG_AGO_MS
        self._focus_hint_until_ms = 0
        self.app._compute_layout()
        self.app._needs_full_present = True

    def _focus_edge_zone_rect(self) -> pg.Rect:
        """
        The strip along the right edge of the window that reveals the focus
        arrow while focus mode is on. It is the only hint of the way out, so
        it spans the whole height of the board

        :returns: the hover zone in window pixels
        """
        board_r = self.board.rect
        w = self.window.get_width()
        return pg.Rect(w - FOCUS_EDGE_ZONE_PX, board_r.top, FOCUS_EDGE_ZONE_PX, board_r.height)

    def _focus_arrow_off_anchor(self) -> tuple[int, int]:
        """
        Where the focus arrow sits in the normal view: just left of the rail,
        level with the middle of the board, so it reads as the handle that
        pulls the rail away

        :returns: the arrow's centre in window pixels
        """
        x = self.right_menu.outer_rect.x - FOCUS_ARROW_D // 2 - FOCUS_ARROW_OFF_GAP
        return (x, self.board.rect.centery)

    def _update_focus_arrow_off(self, now: int) -> None:
        """
        Show and draw the focus arrow in the normal view. It appears for a
        moment when the game starts, so the feature is discoverable, and again
        while the cursor is over the rail, then lingers briefly on the way out

        :param now: pygame tick count in milliseconds
        """
        if not self._focus_arrow_allowed():
            self.focus_arrow.reset()
            return
        mp = pg.mouse.get_pos()
        if self.right_menu.outer_rect.collidepoint(mp):
            self._focus_panel_hover_ms = now
        anchor = self._focus_arrow_off_anchor()
        shown = (now < self._focus_hint_until_ms
                 or now - self._focus_panel_hover_ms < FOCUS_OFF_LINGER_MS)
        self.focus_arrow.update(now, shown, anchor, mp, False)
        self.focus_arrow.draw(self.window)

    def _draw_focus_edge_arrow(self, now: int) -> None:
        """
        Show and draw the focus arrow while focus mode is on, pinned to the
        right edge and revealed only when the cursor goes looking for it, so
        it stays out of the way of the bare board

        :param now: pygame tick count in milliseconds
        """
        mp = pg.mouse.get_pos()
        zone = self._focus_edge_zone_rect()
        reveal = self._focus_arrow_allowed() and zone.collidepoint(mp)
        anchor = (self.window.get_width() - FOCUS_EDGE_ARROW_MARGIN - FOCUS_ARROW_D // 2,
                  self.board.rect.centery)
        self.focus_arrow.update(now, reveal, anchor, mp, True)
        self.focus_arrow.draw(self.window)

    def _draw_time_lines(self, board_rect: pg.Rect | None = None,
                         alpha: float = 1.0) -> None:
        """
        Draw the two slim clock bars above and below the board, the whole
        clock display in the leanest focus mode. Untimed games have nothing to
        show, and during a transition they are drawn against the moving board
        rect and faded with it

        :param board_rect: board rect to lay the lines against, None to use
            the board's own current rect
        :param alpha: fade factor from 0 to 1, used during the transition
        """
        if self._focus_show() != "line" or self._time_control is None:
            return
        self.time_line.draw(self.window, self.board, self.match.clock,
                            self.match.current_turn(), board_rect or self.board.rect, alpha)
