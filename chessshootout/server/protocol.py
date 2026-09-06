import os
import re
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, field_validator

from chessshootout.backend.pieces import PIECE_VALUES, PieceType
from chessshootout.backend.utils import BOARD_SIZE


def _env_float(name: str, default: float) -> float:
    """
    Read one server tuning knob out of the process environment as a float, so an
    operator can retune timings per deployment without a rebuild. A missing or
    unparsable value falls back to the compiled-in default instead of failing
    startup

    :param name: environment variable to read.
    :param default: value used when the variable is absent or not a number.
    :returns: the parsed value, or the default.
    """
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """
    Read one server tuning knob out of the process environment as an integer,
    the counting counterpart of the float reader. A missing or unparsable value
    falls back to the compiled-in default instead of failing startup

    :param name: environment variable to read.
    :param default: value used when the variable is absent or not a number.
    :returns: the parsed value, or the default.
    """
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


PROTOCOL_VERSION = 5
MAX_NICKNAME_LEN = 20
GIVE_TIME_SECONDS = 15
GIVE_TIME_TICK_MS = 100
GIVE_TIME_MAX_HOLD_MS = 600_000
FIRST_MOVE_ABORT_SECONDS = 60
IDLE_RESIGN_SECONDS = 60
QUEUE_MAX_WAIT_SECONDS = 600.0
MIN_TIME_MINUTES = 1
MAX_TIME_MINUTES = 180
MIN_INCREMENT_SECONDS = 0
MAX_INCREMENT_SECONDS = 180
GRACE_SECONDS = _env_float("GRACE_SECONDS", 60.0)
HEARTBEAT_INTERVAL_SECONDS = _env_float("HEARTBEAT_INTERVAL_SECONDS", 2.0)
HEARTBEAT_MISS_LIMIT = _env_int("HEARTBEAT_MISS_LIMIT", 3)
HEARTBEAT_TIMEOUT_SECONDS = HEARTBEAT_INTERVAL_SECONDS * HEARTBEAT_MISS_LIMIT
MAX_SHARED_HIGHLIGHTS = 64
MAX_SHARED_ARROWS = 128
CHAT_COOLDOWN_SECONDS = 3.0
CHAT_PRESET_COUNT = 8
ANNOTATIONS_PER_SECOND = 10
MODERATION_TRIP_LIMIT = 3
SKILLCHECK_TARGET_MAX = float(BOARD_SIZE)
SKILLCHECK_MAX_CAPTURED_VALUE = PIECE_VALUES[PieceType.QUEEN]

UUID4_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
)
COORD_RE = re.compile(r"^[a-h][1-8]$")


def is_uuid4(value: object) -> bool:
    """
    Report whether a value is a canonical UUID4 string. Room ids and player ids
    arrive as untrusted text on every wire surface, so this is the shared shape
    gate that the websocket route and the request validators both apply before
    any room is looked up

    :param value: candidate identifier; anything that is not a string fails.
    :returns: True when the value is a well-formed UUID4 string.
    """
    return isinstance(value, str) and bool(UUID4_RE.match(value))


def _validate_uuid4(value: str, name: str) -> str:
    """
    Gate a UUID4-shaped request field, rejecting anything else with a stable
    invalid_<name> code. That code travels back to the client as the rejection
    reason, so it is part of the protocol vocabulary and must not be reworded
    casually

    :param value: candidate identifier taken from the incoming request.
    :param name: field name woven into the error code, e.g. client_uuid.
    :returns: the identifier unchanged once it is a valid UUID4.
    """
    if not is_uuid4(value):
        raise ValueError(f"invalid_{name}")
    return value


def _validate_coord(value: str, name: str) -> str:
    """
    Gate a board-square field so that only algebraic coordinates from a1 to h8
    are ever stored or relayed. Stops a mark or move that points off the board
    from reaching the other player at all

    :param value: candidate square in algebraic form, e.g. e4.
    :param name: field name woven into the error code.
    :returns: the square unchanged once it names a real board square.
    """
    if not (isinstance(value, str) and COORD_RE.match(value)):
        raise ValueError(f"invalid_{name}")
    return value


def _validate_coords(values: list[str]) -> list[str]:
    """
    Gate a whole set of highlighted squares in one go, the list counterpart of
    the single-square gate. A set holding one unusable square is refused whole,
    so a mark that cannot be drawn never reaches the other player

    :param values: candidate squares in algebraic form, e.g. e4.
    :returns: the same list once every entry names a real board square.
    """
    for value in values:
        _validate_coord(value, "coord")
    return values


class Reason:
    """
    The closed vocabulary of reason codes shared by server and client: why an
    action was refused, and why a game ended. Both sides compare against these
    constants rather than against literals, so a code is renamed in one place
    """

    VERSION_MISMATCH = "version_mismatch"
    INVALID_MESSAGE = "invalid_message"
    INVALID_FIELD = "invalid_field"
    INVALID_MOVE_FORMAT = "invalid_move_format"
    NOT_YOUR_TURN = "not_your_turn"
    SKILLCHECK_PENDING = "skillcheck_pending"
    MOVE_LOCKED = "move_locked"
    SESSION_EXPIRED = "session_expired"
    ALREADY_IN_GAME = "already_in_game"
    NOT_IN_ROOM = "not_in_room"
    ROOM_FULL = "room_full"
    QUEUE_TIMEOUT = "queue_timeout"
    RATE_LIMITED = "rate_limited"
    GAME_ALREADY_OVER = "game_already_over"
    REMATCH_UNAVAILABLE = "rematch_unavailable"
    REMATCH_ALREADY_PENDING = "rematch_already_pending"
    NO_TAKEBACK_AVAILABLE = "no_takeback_available"
    SHARE_MUTED = "share_muted"
    OPP_HIDES_MARKS = "opp_hides_marks"

    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    DRAW_AGREEMENT = "draw_agreement"
    DRAW_STALEMATE = "draw_stalemate"
    DRAW_REPETITION = "draw_repetition"
    DRAW_FIFTY_MOVE = "draw_fifty_move"
    DRAW_INSUFFICIENT_MATERIAL = "draw_insufficient_material"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    ABANDONMENT = "abandonment"
    SERVER_SHUTDOWN = "server_shutdown"


class IdleWindowSpec(NamedTuple):
    """
    One row of the idle-window table: what becomes of a game whose side to move
    has gone silent, and how many seconds of silence are allowed first. Paired
    with a ply count in IDLE_WINDOW_BY_PLIES, the single source both the sweep
    and the pushed countdown read
    """

    outcome: str
    seconds: float


IDLE_WINDOW_BY_PLIES = {
    0: IdleWindowSpec(Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
    1: IdleWindowSpec(Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
    2: IdleWindowSpec(Reason.RESIGNATION, IDLE_RESIGN_SECONDS),
}

IDLE_SECONDS_BY_OUTCOME = dict(IDLE_WINDOW_BY_PLIES.values())


def normalize_nickname(raw: str | None) -> str:
    """
    Put a player-supplied nickname into the single canonical form the server
    stores and shows: trimmed, inner whitespace runs collapsed to one space,
    printable ASCII only, no longer than MAX_NICKNAME_LEN. Series scores are
    keyed by nickname, so two spellings of one name must never both exist

    :param raw: nickname as typed by the player; None is rejected outright.
    :returns: the canonical nickname.
    """
    if raw is None:
        raise ValueError("nickname required")
    if not (raw.isascii() and raw.isprintable()):
        raise ValueError("nickname must be printable ASCII")
    collapsed = re.sub(r"\s+", " ", raw.strip())
    if not collapsed:
        raise ValueError("nickname must not be empty")
    if len(collapsed) > MAX_NICKNAME_LEN:
        raise ValueError(f"nickname must be <= {MAX_NICKNAME_LEN} chars")
    return collapsed


def normalize_country(raw: object) -> str | None:
    """
    Coerce a player-supplied country into the upper-case two-letter code that
    drives the flag beside their name, or None when there is nothing usable.
    Never raises -- an unrecognised country costs a flag, not a match

    :param raw: country value as sent, in any case and with stray whitespace.
    :returns: the two-letter code, or None when the value is not one.
    """
    if raw is None:
        return None
    value = str(raw).strip().upper()
    if len(value) == 2 and value.isascii() and value.isalpha():
        return value
    return None


class _Base(BaseModel):
    """
    Common base for every request and websocket frame, stamping the protocol
    version onto each payload so a client built against a different protocol is
    turned away at the handshake instead of desyncing mid-game
    """

    version: int = PROTOCOL_VERSION


class ReasonDetail(BaseModel):
    """
    Why a request was turned down, given as one code from the shared vocabulary
    both the game and the server use, so the app can react to a refusal instead
    of only showing it
    """

    reason: str


class ReasonEnvelope(BaseModel):
    """
    The body sent back with a refused request: one detail object naming the
    reason it was not accepted
    """

    detail: ReasonDetail


class ClockSnapshot(BaseModel):
    """
    Both players' clocks at a single instant: seconds left for each side, and
    which side, if either, is counting down right now
    """

    white_remaining: float
    black_remaining: float
    running_for: Literal["white", "black"] | None


class HistoryEntryWire(BaseModel):
    """
    One played half-move as it travels on the wire: the squares it went between,
    the piece a pawn promoted to if it did, and the move in standard algebraic
    notation
    """

    from_sq: str
    to_sq: str
    promotion: Literal["q", "r", "b", "n"] | None = None
    san: str


class LockWire(BaseModel):
    """
    A move that is unavailable for the rest of the current turn, given as its
    origin and destination squares. The player must choose a different move
    until the turn passes
    """

    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class IdleWindowWire(BaseModel):
    """
    The countdown that runs while the side to move stays silent: what will
    happen when it reaches zero, whose turn it is, and how much time is left
    """

    outcome: Literal["aborted", "resignation"]
    color: Literal["white", "black"]
    seconds_remaining: float = Field(ge=0.0, allow_inf_nan=False)


class IdleWindowMessage(IdleWindowWire, _Base):
    """
    Live update of the idle countdown for the running game, sent to both players
    so the timer on screen agrees with the one the server is keeping
    """

    type: Literal["idle_window"] = "idle_window"


class ArrowWire(BaseModel):
    """
    One board arrow a player has drawn, as the pair of squares it points from
    and to. The shared building block wherever sets of marks are exchanged
    """

    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")

    model_config = {"populate_by_name": True}

    @field_validator("from_sq", "to_sq")
    @classmethod
    def _coord(cls, v: str) -> str:
        """
        Reject an arrow whose endpoints are not real board squares, so no mark
        pointing off the board can be stored or relayed

        :param v: candidate square in algebraic form.
        :returns: the square unchanged once it names a real board square.
        """
        return _validate_coord(v, "coord")


class AnnotationSetWire(BaseModel):
    """
    A player's complete set of shared board marks: whether they are sharing at
    all, which squares are highlighted, and which arrows are drawn. Sent whole
    when a game is restored, so both boards resume from the same picture
    """

    sharing: bool = False
    highlights: list[str] = Field(default_factory=list, max_length=MAX_SHARED_HIGHLIGHTS)
    arrows: list[ArrowWire] = Field(default_factory=list, max_length=MAX_SHARED_ARROWS)

    @field_validator("highlights")
    @classmethod
    def _coords(cls, v: list[str]) -> list[str]:
        """
        Reject a highlight set holding anything that is not a real board square,
        so every stored mark stays drawable on the receiving board

        :param v: highlighted squares in algebraic form.
        :returns: the same list once every entry names a real square.
        """
        return _validate_coords(v)


SkillCheckKindLiteral = Literal["wheel", "aim", "whack", "combo"]
SkillCheckDirectionLiteral = Literal["up", "down", "left", "right"]


class _SkillCheckGeometryBase(BaseModel):
    """
    Shared description of one skill-check challenge: which kind it is, the move
    that set it off, and the parameters that shape how it plays. Reused by every
    frame that has to describe the same challenge to somebody
    """

    kind: SkillCheckKindLiteral
    seed: str
    value_diff: int
    deadline_ms: float = Field(ge=0.0, allow_inf_nan=False)
    captured_value: int = Field(default=0, ge=0, le=SKILLCHECK_MAX_CAPTURED_VALUE)
    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")
    promotion: Literal["q", "r", "b", "n"] | None = None

    model_config = {"populate_by_name": True}


class PendingSkillCheckWire(_SkillCheckGeometryBase):
    """
    A skill check that was already running when the game state was fetched, so a
    returning player rejoins the challenge in progress instead of starting over.
    Carries the challenge itself plus how far through it the player has got
    """

    elapsed_ms: float = Field(ge=0.0, allow_inf_nan=False)
    miss_count: int = 0
    progress: int = 0
    last_hit_pop: int = -1
    color: Literal["white", "black"]


class SkillCheckOutcomeWire(BaseModel):
    """
    The record of one finished skill check: the ply and move it belonged to,
    which kind of challenge it was, and whether the player won it. Replayed on a
    restore so the game keeps its full shootout history
    """

    ply: int
    kind: SkillCheckKindLiteral
    won: bool
    san: str = ""


class MatchmakeRequest(_Base):
    """
    A player's request to be paired for an online game. Says who they are, which
    time control they want to play and which side they would prefer; they are
    paired with an opponent asking for the same time control
    """

    nickname: str
    client_uuid: str
    time_minutes: int = Field(ge=MIN_TIME_MINUTES, le=MAX_TIME_MINUTES)
    increment_seconds: int = Field(ge=MIN_INCREMENT_SECONDS, le=MAX_INCREMENT_SECONDS)
    side_preference: Literal["white", "black", "random"] = "random"
    country: str | None = None
    hide_opp_marks: bool = False

    @field_validator("nickname")
    @classmethod
    def _normalize(cls, v: str) -> str:
        """
        Store the nickname in its canonical display form, so one player always
        appears the same way to opponents and in the series score table, which
        is keyed by nickname

        :param v: nickname as typed by the player.
        :returns: the trimmed, whitespace-collapsed nickname.
        """
        return normalize_nickname(v)

    @field_validator("country", mode="before")
    @classmethod
    def _country(cls, v: object) -> str | None:
        """
        Accept the country in whatever case and spacing the client sent, and
        drop it quietly when it is not a two-letter code -- an unusable country
        means no flag, never a refused match request

        :param v: raw country value, before any field validation has run.
        :returns: the upper-case two-letter code, or None when unusable.
        """
        return normalize_country(v)

    @field_validator("client_uuid")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        """
        Reject a player id that is not a UUID4. The id is how a player is
        recognised across reconnects, so a malformed one is refused before any
        room is created

        :param v: player id supplied with the request.
        :returns: the id unchanged once it is a valid UUID4.
        """
        return _validate_uuid4(v, "client_uuid")


class MatchmakeResponse(_Base):
    """
    Confirms the player now holds a place in a room: the room to open the
    websocket on, and the session token that identifies them on it
    """

    room_id: str
    session_token: str


class CancelMatchmakeRequest(_Base):
    """
    Withdraws a player who is still waiting to be paired, named by the room they
    were placed in and their session token. It has no effect once the game has
    already started
    """

    room_id: str
    session_token: str

    @field_validator("room_id")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        """
        Reject a room id that is not a UUID4, so a malformed id is refused
        before any queue is searched

        :param v: room id supplied with the request.
        :returns: the id unchanged once it is a valid UUID4.
        """
        return _validate_uuid4(v, "room_id")


class ResumeRequest(_Base):
    """
    Asks for the full current state of a game the player is already in, named by
    room and session token. Used after a dropped connection so the client can
    rebuild the game exactly rather than guess at it
    """

    room_id: str
    session_token: str

    @field_validator("room_id")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        """
        Reject a room id that is not a UUID4, so a malformed id is refused
        before any room is looked up

        :param v: room id supplied with the request.
        :returns: the id unchanged once it is a valid UUID4.
        """
        return _validate_uuid4(v, "room_id")


class ResumeResponse(_Base):
    """
    The whole state of a running or just-finished game: position and move list,
    clocks, both players, any challenge or shared marks in flight, and the
    result if one has been recorded. Everything a client needs to rebuild its
    game screen after reconnecting
    """

    fen: str
    move_history: list[HistoryEntryWire]
    clock: ClockSnapshot
    your_color: Literal["white", "black"]
    white_name: str
    black_name: str
    time_minutes: int
    increment_seconds: int
    white_score: float = 0.0
    black_score: float = 0.0
    white_country: str | None = None
    black_country: str | None = None
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    grace_seconds: float = GRACE_SECONDS
    pending_skillcheck: PendingSkillCheckWire | None = None
    skillcheck_locks: list[LockWire] = Field(default_factory=list)
    skillcheck_log: list[SkillCheckOutcomeWire] = Field(default_factory=list)
    white_annotations: AnnotationSetWire = Field(default_factory=AnnotationSetWire)
    black_annotations: AnnotationSetWire = Field(default_factory=AnnotationSetWire)
    share_muted: bool = False
    hide_opp_marks: bool = False
    result_reason: str | None = None
    result_winner: Literal["white", "black"] | None = None
    idle_window: IdleWindowWire | None = None


class ReclaimRequest(_Base):
    """
    Asks whether a player known only by their player id still has a game waiting
    for them. Used after the app has been restarted, when the session token from
    the previous run is gone
    """

    client_uuid: str

    @field_validator("client_uuid")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        """
        Reject a player id that is not a UUID4, so a malformed id is refused
        before any session is handed out

        :param v: player id supplied with the request.
        :returns: the id unchanged once it is a valid UUID4.
        """
        return _validate_uuid4(v, "client_uuid")


class ReclaimResponse(_Base):
    """
    The game the player may rejoin: the room to reconnect to, and a freshly
    issued session token that replaces the one they no longer have
    """

    room_id: str
    session_token: str


class HealthResponse(BaseModel):
    """
    Liveness and load summary for the server: protocol and build version, how
    many games are running, how many players are waiting to be paired, and how
    long the process has been up
    """

    status: str = "ok"
    version: int = PROTOCOL_VERSION
    app_version: str = ""
    rooms_active: int
    queue_depth: int = 0
    uptime_s: float = 0.0


class AuthMessage(_Base):
    """
    The first frame a client sends after opening the websocket, presenting the
    session token for that room. No game traffic flows until it is accepted
    """

    type: Literal["auth"] = "auth"
    session_token: str


class MoveMessage(_Base):
    """
    A player's attempt to play a move: origin and destination squares, plus the
    chosen piece when a pawn reaches the last rank. The move is only real once
    the server accepts it and echoes it back to both sides
    """

    type: Literal["move"] = "move"
    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")
    promotion: Literal["q", "r", "b", "n"] | None = None

    model_config = {"populate_by_name": True}


class DrawResponseMessage(_Base):
    """
    A player's answer to a draw offer -- accepted, which ends the game as a
    draw, or declined, which clears the offer
    """

    type: Literal["draw_response"] = "draw_response"
    accept: bool


class RematchRequestMessage(_Base):
    """
    Offers another game in the same room once the current one has finished. The
    same frame is relayed on to the opponent as the incoming offer, and two
    matching offers start the rematch
    """

    type: Literal["rematch_request"] = "rematch_request"


class RematchResponseMessage(_Base):
    """
    A player's answer to a rematch offer -- accepted, which restarts the room
    with colours swapped, or declined, which closes it out
    """

    type: Literal["rematch_response"] = "rematch_response"
    accept: bool


RematchUpdateEvent = Literal[
    "declined", "cancelled", "window_expired",
    "opponent_left", "opponent_reconnecting", "opponent_returned",
]


class RematchUpdateMessage(_Base):
    """
    Tells a player what became of the rematch they were waiting on: it was
    declined or withdrawn, the window ran out, or the opponent left, came back,
    or is reconnecting
    """

    type: Literal["rematch_update"] = "rematch_update"
    event: RematchUpdateEvent


class TakebackResponseMessage(_Base):
    """
    A player's answer to a takeback request -- accepted, which retracts the last
    move for both sides, or declined, which leaves the position alone
    """

    type: Literal["takeback_response"] = "takeback_response"
    accept: bool


class GameStartMessage(_Base):
    """
    Announces that the game is under way: starting position, both players with
    their countries and running series scores, the time control, and which side
    the recipient plays. Sent to each player separately, since the colour and
    the clock offset differ
    """

    type: Literal["game_start"] = "game_start"
    fen: str
    white_name: str
    black_name: str
    time_minutes: int
    increment_seconds: int
    your_color: Literal["white", "black"]
    started_seconds_ago: float = 0.0
    white_score: float = 0.0
    black_score: float = 0.0
    white_country: str | None = None
    black_country: str | None = None
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    grace_seconds: float = GRACE_SECONDS
    rematch: bool = False


class MoveAppliedMessage(_Base):
    """
    The authoritative record of a move that has landed: the squares played, the
    notation, the clocks afterwards, the new ply count, and the skill check it
    came through if it triggered one. Both boards advance from this frame, not
    from their own guess
    """

    type: Literal["move_applied"] = "move_applied"
    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")
    promotion: Literal["q", "r", "b", "n"] | None = None
    san: str
    clock: ClockSnapshot
    ply: int
    skill_check_kind: SkillCheckKindLiteral | None = None
    skill_check_won: bool | None = None

    model_config = {"populate_by_name": True}


class ResultMessage(_Base):
    """
    Ends the game for both players, naming why it ended and which side won when
    there is a winner. A game ends exactly once, and every client takes its
    outcome from this frame
    """

    type: Literal["result"] = "result"
    reason: str
    winner_color: Literal["white", "black"] | None = None


class DrawOfferedMessage(_Base):
    """
    Tells a player their opponent has offered a draw, which they answer with a
    draw response
    """

    type: Literal["draw_offered"] = "draw_offered"


class TakebackOfferedMessage(_Base):
    """
    Tells a player their opponent has asked to take back their last move, which
    they answer with a takeback response
    """

    type: Literal["takeback_offered"] = "takeback_offered"


class TakebackAppliedMessage(_Base):
    """
    Confirms an agreed takeback: the position once the move was retracted, the
    clocks, and the new ply count. Both boards rewind to exactly this state
    """

    type: Literal["takeback_applied"] = "takeback_applied"
    fen: str
    clock: ClockSnapshot
    ply: int


class GiveTimeMessage(_Base):
    """
    A player's gift of clock time to their opponent, sized by how long they held
    the give-time button. The hold is converted into whole chunks of time, so a
    longer hold gives more
    """

    type: Literal["give_time"] = "give_time"
    hold_ms: int = Field(default=0, ge=0, le=GIVE_TIME_MAX_HOLD_MS)


class TimeGrantedMessage(_Base):
    """
    Tells both players that clock time changed hands: who gave it, how many
    seconds were actually added, and the clocks once it was applied
    """

    type: Literal["time_granted"] = "time_granted"
    granted_by: Literal["white", "black"]
    seconds_added: float
    clock: ClockSnapshot


class AnnotationsStateMessage(_Base):
    """
    A player's whole set of shared board marks at once: whether they are
    sharing, plus every highlight and arrow. Sent when sharing is switched on or
    off, and whenever the opponent's view has to be reset wholesale
    """

    type: Literal["annotations_state"] = "annotations_state"
    sharing: bool
    highlights: list[str] = Field(default_factory=list, max_length=MAX_SHARED_HIGHLIGHTS)
    arrows: list[ArrowWire] = Field(default_factory=list, max_length=MAX_SHARED_ARROWS)

    @field_validator("highlights")
    @classmethod
    def _coords(cls, v: list[str]) -> list[str]:
        """
        Reject a highlight set holding anything that is not a real board square,
        so no mark that cannot be drawn is ever relayed onward

        :param v: highlighted squares in algebraic form.
        :returns: the same list once every entry names a real square.
        """
        return _validate_coords(v)


class AnnotationDeltaMessage(_Base):
    """
    A single change to a player's shared marks: one highlight or one arrow,
    added or removed. Keeps live drawing cheap next to resending the whole set
    on every stroke
    """

    type: Literal["annotation_delta"] = "annotation_delta"
    action: Literal["add", "remove"]
    kind: Literal["highlight", "arrow"]
    square: str | None = None
    from_sq: str | None = Field(default=None, alias="from")
    to_sq: str | None = Field(default=None, alias="to")

    model_config = {"populate_by_name": True}

    @field_validator("square", "from_sq", "to_sq")
    @classmethod
    def _coord(cls, v: str | None) -> str | None:
        """
        Reject a delta that names a square off the board, while letting the
        fields the other mark kind does not use stay absent

        :param v: candidate square in algebraic form, or None when unused.
        :returns: the value unchanged when it is None or a real square.
        """
        if v is None:
            return v
        return _validate_coord(v, "coord")


class AnnotationsBlockedMessage(_Base):
    """
    Tells a player that some of their shared marks were flagged: which
    highlights and arrows were involved, whether they were only suspect or
    removed outright, and whether their sharing is now muted
    """

    type: Literal["annotations_blocked"] = "annotations_blocked"
    action: Literal["blocked", "suspect"]
    highlights: list[str] = Field(default_factory=list, max_length=MAX_SHARED_HIGHLIGHTS)
    arrows: list[ArrowWire] = Field(default_factory=list, max_length=MAX_SHARED_ARROWS)
    share_muted: bool = False


class SetMarksVisibilityMessage(_Base):
    """
    A player's choice about seeing the opponent's shared marks. While they are
    hidden, the opponent's marks are not delivered to this player at all
    """

    type: Literal["set_marks_visibility"] = "set_marks_visibility"
    hide_opp: bool


class QuickChatMessage(_Base):
    """
    A player's pick from the fixed quick-chat phrase list, sent as the index of
    the phrase. Free text never travels between players
    """

    type: Literal["quick_chat"] = "quick_chat"
    preset: int = Field(ge=0, le=CHAT_PRESET_COUNT - 1)


class QuickChatReceivedMessage(_Base):
    """
    A quick-chat phrase arriving from the opponent, as the phrase index plus the
    side that sent it, so the bubble appears over the right player
    """

    type: Literal["quick_chat_received"] = "quick_chat_received"
    preset: int = Field(ge=0, le=CHAT_PRESET_COUNT - 1)
    sender: Literal["white", "black"]


class ConnectionStatusMessage(_Base):
    """
    Reports how the opponent's side is doing -- connected, reconnecting, or
    resyncing -- so a player can tell why the board has gone quiet instead of
    assuming they were abandoned
    """

    type: Literal["connection_status"] = "connection_status"
    opp_state: Literal["connected", "reconnecting", "resyncing"]


class PingMessage(_Base):
    """
    The client's periodic heartbeat, which also reports the ply the client
    believes it is on, so a board that has fallen behind can be spotted and
    repaired rather than drifting
    """

    type: Literal["ping"] = "ping"
    ply: int = 0


class PongMessage(_Base):
    """
    The reply to a heartbeat, proving the connection is alive and giving the
    client its round-trip time
    """

    type: Literal["pong"] = "pong"


class ResyncDirectiveMessage(_Base):
    """
    Tells a client its board has fallen behind and it should fetch the whole
    game state again instead of patching up what it has
    """

    type: Literal["resync_directive"] = "resync_directive"


class SkillCheckRequiredMessage(_SkillCheckGeometryBase, _Base):
    """
    Tells the moving player that their move set off a skill check, and describes
    the challenge they have to beat. The move lands only if they win it
    """

    type: Literal["skill_check_required"] = "skill_check_required"
    miss_count: int = 0


class SkillCheckShotMessage(_Base):
    """
    One input the player makes during a running skill check: a direction for the
    combo prompts, or a board-space row and column for whack-a-mole, together
    with how far into the challenge it was made
    """

    type: Literal["skill_check_shot"] = "skill_check_shot"
    client_elapsed_ms: float = Field(default=0.0, allow_inf_nan=False)
    direction: SkillCheckDirectionLiteral | None = None
    target_row: float | None = Field(default=None, ge=0.0, lt=SKILLCHECK_TARGET_MAX)
    target_col: float | None = Field(default=None, ge=0.0, lt=SKILLCHECK_TARGET_MAX)

    model_config = {"extra": "ignore"}


class SkillCheckResultMessage(_Base):
    """
    Reports how a skill check finished for the move that triggered it, naming
    that move. Sent to both players so their boards agree on whether the move
    landed
    """

    type: Literal["skill_check_result"] = "skill_check_result"
    won: bool
    from_sq: str = Field(alias="from")
    to_sq: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class SkillCheckSpectateMessage(_SkillCheckGeometryBase, _Base):
    """
    The watching opponent's view of a skill check that has just started: the
    same challenge the moving player is facing, so it can be mirrored on screen
    while they play it
    """

    type: Literal["skill_check_spectate"] = "skill_check_spectate"


class SkillCheckSpectateShotMessage(_Base):
    """
    Mirrors each of the moving player's skill-check inputs to the watching
    opponent: when it came, whether it hit, how far the challenge has got, and
    which board square it was aimed at. Keeps both screens telling the same
    story
    """

    type: Literal["skill_check_spectate_shot"] = "skill_check_spectate_shot"
    elapsed_ms: float
    miss_count: int
    won: bool
    progress: int = 0
    direction: SkillCheckDirectionLiteral | None = None
    target_row: float | None = Field(default=None, ge=0.0, lt=SKILLCHECK_TARGET_MAX)
    target_col: float | None = Field(default=None, ge=0.0, lt=SKILLCHECK_TARGET_MAX)


class ErrorMessage(_Base):
    """
    Refuses one inbound frame, naming the reason and the kind of message it
    applies to. It never ends the game -- the sender is expected to correct
    course and carry on
    """

    type: Literal["error"] = "error"
    reason: str
    msg_type: str | None = None
