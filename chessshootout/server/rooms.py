import asyncio
import random
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast
from uuid import uuid4

from chessshootout.backend.backend import Backend
from chessshootout.backend.pieces import PieceColor
from chessshootout.backend.utils import Square
from chessshootout.server.protocol import (
    GRACE_SECONDS, HEARTBEAT_TIMEOUT_SECONDS, IDLE_WINDOW_BY_PLIES, IdleWindowSpec,
    Reason,
)
from chessshootout.skillcheck import online
from chessshootout.skillcheck.types import SkillCheckKind, SkillCheckOutcome
from chessshootout.skillcheck.wheel import SKILLCHECK_DEADLINE_MS


REMATCH_IDLE_SECONDS = 300.0
REMATCH_ABSOLUTE_CAP_SECONDS = 900.0
POST_GAME_DISCONNECT_GRACE = GRACE_SECONDS
PAIRING_WAIT_SECONDS = 30
QUEUE_ABANDON_SECONDS = 120.0
HISTORY_CHANGE_WINDOW = 8


class HistoryChange(NamedTuple):
    """
    One moment at which a room's move history changed, and how long it was just
    before. The heartbeat check reads these to tell a client that is merely a
    frame behind the news from one that has genuinely drifted
    """

    at: float
    prev_len: int | None


class AlreadyInGameError(Exception):
    """
    Raised when a player asks to be matched while they already hold a seat
    somewhere, which the matchmaking route turns into a refusal rather than
    letting one player sit in two games
    """

    pass


class NotInRoomError(Exception):
    """
    Raised when the room or session a request names cannot be found any more,
    usually because that game finished and was swept away
    """

    pass


class InvalidTokenError(Exception):
    """
    Raised when a session token does not match the seat it claims, which stops
    one player cancelling or resuming another player's game
    """

    pass


class ServerFullError(Exception):
    """
    Raised when the server already holds as many games and waiting players as
    it will take, which the matchmaking route turns into a refusal asking the
    player to try again shortly
    """

    pass


class GameAlreadyStartedError(Exception):
    """
    Raised when a player tries to withdraw from a waiting place whose game has
    meanwhile begun, which the cancel route reports as already started rather
    than as a failure
    """

    pass


@dataclass
class PlayerSlot:
    """
    One seat in a room: who the player is, how they are recognised across
    reconnects, and the live state of their connection. A seat outlives any
    single socket, which is what makes a dropped connection recoverable
    """

    client_uuid: str
    nickname: str
    session_token: str
    side_preference: str = "random"
    country: str | None = None
    hide_opp_marks: bool = False
    connected: bool = False
    disconnected_at: float | None = None
    desync_active: bool = False
    ply_mismatch_streak: int = 0
    last_seen: float = 0.0
    at_result: bool = False


@dataclass
class PendingSkillCheck:
    """
    The skill check a room is waiting on, held entirely server-side: the move
    it is gating, the seed the challenge is rebuilt from, and how far the
    player has got. The move lands only once this is beaten, and the deadline
    is absolute so going quiet cannot dodge it
    """

    color: str
    from_sq: Square
    to_sq: Square
    promotion: str | None
    kind: SkillCheckKind
    seed: str
    value_diff: int
    start_ms: float
    expires_at_ms: float
    deadline_ms: float = SKILLCHECK_DEADLINE_MS
    miss_count: int = 0
    captured_value: int = 0
    progress: int = 0
    last_hit_pop: int = -1
    last_input_ms: float = -1.0
    holes: tuple[tuple[int, int], ...] = field(default=(), repr=False, compare=False)
    _challenge: object = field(default=None, repr=False, compare=False)

    @property
    def challenge(self) -> Any:
        """
        The challenge itself, rebuilt from the seed the first time it is asked
        for and kept from then on. It is never stored or sent anywhere --
        every surface that has to judge a shot derives the same challenge from
        the same seed

        :returns: the rebuilt challenge for this check's kind.
        """
        if self._challenge is None:
            self._challenge = online.challenge_from(
                self.kind, self.seed, self.value_diff, self.deadline_ms,
                self.captured_value)
        return self._challenge

    def is_expired(self, now_ms: float) -> bool:
        """
        Tell whether the check has run past its absolute deadline, the
        stopwatch that stops a player stalling a check forever by going quiet

        :param now_ms: current server time in milliseconds.
        :returns: True once the deadline has passed.
        """
        return now_ms > self.expires_at_ms

    def is_dead(self, now_ms: float) -> bool:
        """
        Tell whether this check is finished as a miss, either because the
        deadline passed or because the challenge can no longer be won at all.
        Every surface that has to decide about a stale check -- the move gate,
        a resume and the sweep -- asks exactly this question

        :param now_ms: current server time in milliseconds.
        :returns: True when the check should be resolved as a miss.
        """
        if self.is_expired(now_ms):
            return True
        return online.check_expired(self.kind, self.challenge, now_ms - self.start_ms,
                                    self.miss_count, self.progress, self.last_hit_pop)


@dataclass
class SharedAnnotations:
    """
    One player's shared board marks as the server holds them: whether they are
    sharing at all, the highlights and arrows themselves, and their standing
    with the mark screener
    """

    sharing: bool = False
    highlights: set[str] = field(default_factory=set)
    arrows: list[tuple[str, str]] = field(default_factory=list)
    trip_count: int = 0
    share_muted: bool = False
    opp_hidden_notice_sent: bool = False

    def clear_marks(self) -> None:
        """
        Drop every highlight and arrow while leaving the sharing switch and
        the screener standing alone, which is what a played move does to both
        players' marks
        """
        self.highlights.clear()
        self.arrows.clear()

    def reset(self) -> None:
        """
        Put this player's sharing back to its start-of-game state, forgiving
        any screener refusals along with it. Called when a room is finalised
        or replayed, so a rematch starts clean
        """
        self.clear_marks()
        self.sharing = False
        self.trip_count = 0
        self.share_muted = False
        self.opp_hidden_notice_sent = False

    def register_trip(self, limit: int) -> bool:
        """
        Record that the screener refused this player's marks, and mute their
        sharing once they have collected enough refusals. Muting lasts the
        rest of the game and clears their marks on the spot

        :param limit: refusals allowed before sharing is muted.
        :returns: True when this refusal was the one that muted them.
        """
        self.trip_count += 1
        muted = self.trip_count >= limit
        if muted:
            self.share_muted = True
            self.clear_marks()
        return muted

    def strip(self, arrows: Iterable[tuple[str, str]], highlights: Iterable[str]) -> None:
        """
        Remove exactly the marks the screener refused, leaving the rest of
        this player's drawing untouched

        :param arrows: refused arrows as origin and destination squares.
        :param highlights: refused squares in algebraic form.
        """
        for arrow in arrows:
            pair = (arrow[0], arrow[1])
            while pair in self.arrows:
                self.arrows.remove(pair)
        for highlight in highlights:
            self.highlights.discard(highlight)


@dataclass
class Room:
    """
    One online game and everything the server knows about it: both seats, the
    engine board, the offers in flight, the shared marks and the result once
    there is one. plies_ever counts every ply the room has ever seen and never
    leaves the server, since it is part of what decides which move draws which
    skill check
    """

    room_id: str
    time_minutes: int
    increment_seconds: int
    created_at: float
    white: PlayerSlot | None = None
    black: PlayerSlot | None = None
    backend: Backend | None = None
    started_at: float | None = None
    first_move_at: float | None = None
    ended_at: float | None = None
    last_rematch_activity_at: float | None = None
    game_start_broadcast: bool = False
    draw_offered_by: str | None = None
    takeback_offered_by: str | None = None
    rematch_offered_by: set[str] = field(default_factory=set)
    result: tuple[str, str | None] | None = None
    series_scores: dict[str, float] = field(default_factory=dict)
    skillcheck_secret: str = ""
    skillcheck_locks: set[tuple[Square, Square]] = field(default_factory=set)
    skillcheck_log: list[SkillCheckOutcome] = field(default_factory=list)
    pending_skillcheck: PendingSkillCheck | None = None
    plies_ever: int = 0
    idle_since: float | None = None
    idle_pushed_at: float | None = None
    annotations_white: SharedAnnotations = field(default_factory=SharedAnnotations)
    annotations_black: SharedAnnotations = field(default_factory=SharedAnnotations)
    history_changes: deque[HistoryChange] = field(
        default_factory=lambda: deque(maxlen=HISTORY_CHANGE_WINDOW))

    def note_history_change(self, now: float, prev_len: int | None) -> None:
        """
        Record that the move history just changed, which is what lets a
        heartbeat still describing the previous position be read as news in
        flight rather than as a client that has drifted. A few of the most
        recent changes are kept, so two quick plies cannot hide each other

        :param now: monotonic seconds at which the change landed.
        :param prev_len: how long the history was just before, or None when the
            client's previous state is unknowable, as at a game start.
        """
        self.history_changes.append(HistoryChange(at=now, prev_len=prev_len))

    def score_for(self, color: str) -> float:
        """
        Read a side's running score in this room's series, the number shown
        beside the names. Scores are keyed by nickname, so they survive the
        colour swap a rematch performs

        :param color: white or black.
        :returns: that side's series score, 0.0 while the seat is empty.
        """
        slot = self.slot(color)
        return self.series_scores.get(slot.nickname, 0.0) if slot else 0.0

    def is_paired(self) -> bool:
        """
        Tell whether both seats are filled, which is what separates a room
        still waiting in the queue from a game that can actually start

        :returns: True once both seats hold a player.
        """
        return self.white is not None and self.black is not None

    def slot(self, color: str) -> PlayerSlot | None:
        """
        Get the seat for one colour, which is how nearly everything reaches a
        player's identity and connection state

        :param color: white or black.
        :returns: that seat, or None while nobody holds the colour.
        """
        return self.white if color == "white" else self.black

    def annotations_for(self, color: str) -> SharedAnnotations:
        """
        Get the shared-mark store belonging to one side. The two players'
        marks are kept apart so the screener can judge them separately and
        then together

        :param color: white or black.
        :returns: that side's shared marks.
        """
        return self.annotations_white if color == "white" else self.annotations_black

    def opp_color(self, color: str) -> str:
        """
        Name the other side, the small helper the whole server uses whenever
        it has to address a player's opponent

        :param color: white or black.
        :returns: the opposite colour.
        """
        return "black" if color == "white" else "white"

    def color_to_move(self) -> str | None:
        """
        Say whose turn it is on the server's own board, the authority behind
        every turn check and behind the idle countdown

        :returns: white or black, or None before a board exists.
        """
        if self.backend is None:
            return None
        return "white" if self.backend.current_turn() == PieceColor.WHITE else "black"

    def idle_window(self) -> IdleWindowSpec | None:
        """
        Look up the idle rule in force at this point of the game: silence
        before the first moves aborts the game, silence on the ply after that
        resigns it, and from then on no idle window arms at all

        :returns: the rule for the current ply count, None when none arms.
        """
        return IDLE_WINDOW_BY_PLIES.get(self.plies_ever)

    def idle_remaining(self, now: float) -> float | None:
        """
        How much longer the side to move may stay silent, the countdown that
        is pushed to both clients. Zero means the window is up

        :param now: monotonic seconds, the server's own clock.
        :returns: seconds left, or None when no window is running.
        """
        window = self.idle_window()
        if window is None or self.idle_since is None:
            return None
        return max(window.seconds - (now - self.idle_since), 0.0)

    def idle_timeout_reason(self, now: float) -> str | None:
        """
        Say what should become of this room right now if its idle window has
        run out -- aborted or resigned -- and nothing at all while it has not.
        A room that is not yet paired never times out this way

        :param now: monotonic seconds, the server's own clock.
        :returns: the reason code to finalise with, or None to leave it be.
        """
        if not self.is_paired():
            return None
        remaining = self.idle_remaining(now)
        if remaining is None or remaining > 0.0:
            return None
        return cast(IdleWindowSpec, self.idle_window()).outcome

    def mark_idle_activity(self, now: float | None) -> None:
        """
        Stamp the moment the idle countdown should be measured from, and clear
        the push throttle so the new reading reaches both clients at once.
        Passing None instead switches the countdown off

        :param now: monotonic seconds to count from, or None to stop counting.
        """
        self.idle_since = now
        self.idle_pushed_at = None

    def note_idle_push(self, now: float) -> None:
        """
        Remember when the countdown was last sent out, so routine pushes can
        be throttled rather than following every tick of the sweep

        :param now: monotonic seconds the push went out at.
        """
        self.idle_pushed_at = now

    def hides_opponent_marks(self, color: str) -> bool:
        """
        Tell whether this player has chosen not to see the opponent's shared
        marks, a choice that suppresses delivery rather than the opponent's
        drawing

        :param color: white or black.
        :returns: True when that player wants the opponent's marks hidden.
        """
        slot = self.slot(color)
        return slot is not None and slot.hide_opp_marks

    def color_of(self, client_uuid: str) -> str | None:
        """
        Find which side a player holds in this room. A socket looks this up
        rather than trusting the colour it authenticated with, because a
        rematch swaps the colours underneath it

        :param client_uuid: the player's stable id.
        :returns: white or black, or None when they are not in this room.
        """
        if self.white and self.white.client_uuid == client_uuid:
            return "white"
        if self.black and self.black.client_uuid == client_uuid:
            return "black"
        return None

    def slot_by_token(self, session_token: str) -> tuple[str | None, PlayerSlot | None]:
        """
        Find the seat a session token belongs to, which is how a websocket or
        a resume request proves who it is

        :param session_token: token handed out when the seat was taken.
        :returns: the colour and the seat, or two Nones when nothing matches.
        """
        if self.white and self.white.session_token == session_token:
            return "white", self.white
        if self.black and self.black.session_token == session_token:
            return "black", self.black
        return None, None


class RoomManager:
    """
    The server's whole population of games: the queues of players waiting for
    an opponent, the rooms in progress, and the map from a player id to the
    room holding them. Every room is created, paired, finalised and dropped
    through here, so nothing else has to know how rooms are stored
    """

    def __init__(self, now_provider: Callable[[], float] = time.monotonic,
                 max_rooms: int = 100) -> None:
        """
        Build an empty room manager. The clock is injected so tests can drive
        every timeout deterministically, and the room cap is what lets the
        server refuse new games rather than degrade for everyone

        :param now_provider: returns monotonic seconds; the server's clock.
        :param max_rooms: queued plus active rooms allowed before refusing.
        """
        self._lock = asyncio.Lock()
        self._queue: dict[tuple[int, int], list[Room]] = defaultdict(list)
        self._active: dict[str, Room] = {}
        self._uuid_to_room: dict[str, str] = {}
        self._now = now_provider
        self._max_rooms = max_rooms

    @property
    def rooms_active(self) -> int:
        """
        How many games are in progress, the figure the health endpoint reports
        as this server's load

        :returns: count of active rooms.
        """
        return len(self._active)

    @property
    def queue_depth(self) -> int:
        """
        How many players are waiting to be paired across every time control,
        reported alongside the active room count

        :returns: count of rooms sitting in a queue.
        """
        return sum(len(q) for q in self._queue.values())

    def active_rooms(self) -> list[Room]:
        """
        Snapshot the games in progress for the sweep to walk. It is a copy on
        purpose, since finalising a room during the walk drops it out of the
        live map

        :returns: the active rooms as of this moment.
        """
        return list(self._active.values())

    def get(self, room_id: str) -> Room | None:
        """
        Find a room by id whether it is already playing or still waiting for
        an opponent. The websocket route needs both, because a client connects
        before it has been paired

        :param room_id: the room's id, as handed out at matchmaking.
        :returns: the room, or None when no such room exists.
        """
        if room_id in self._active:
            return self._active[room_id]
        return self._find_in_queue(room_id)

    async def enqueue(self, *, client_uuid: str, nickname: str, session_token: str,
                      time_minutes: int, increment_seconds: int, side_preference: str,
                      country: str | None = None,
                      hide_opp_marks: bool = False) -> Room:
        """
        Seat a player: pair them with the room already waiting on the same
        time control, or open a new one for them to wait in. Pairing settles
        the colours from both players' preferences, builds the board and the
        clocks, draws the room's own skill-check secret and starts the idle
        countdown

        :param client_uuid: the player's stable id; one game at a time.
        :param nickname: display name, and the key their series score uses.
        :param session_token: token this seat will be proved with.
        :param time_minutes: base minutes per side, half of the queue key.
        :param increment_seconds: seconds added per move, the other half.
        :param side_preference: white, black or random.
        :param country: two-letter country code for the flag, or None.
        :param hide_opp_marks: start with the opponent's marks hidden.
        :returns: the room the player now holds a seat in, paired or waiting.
        """
        async with self._lock:
            if client_uuid in self._uuid_to_room:
                raise AlreadyInGameError()
            if len(self._active) + self.queue_depth >= self._max_rooms:
                raise ServerFullError()
            tc = (time_minutes, increment_seconds)
            queue = self._queue[tc]
            if queue:
                room = queue[0]
                self._dequeue(room)
                first_slot = cast(PlayerSlot, room.white or room.black)
                first_pref = first_slot.side_preference
                second_color, first_color_resolved = self._resolve_colors(
                    first_pref, side_preference,
                )
                room.white = None
                room.black = None
                setattr(room, first_color_resolved, first_slot)
                new_slot = PlayerSlot(
                    client_uuid=client_uuid, nickname=nickname,
                    session_token=session_token, side_preference=side_preference,
                    country=country, hide_opp_marks=hide_opp_marks,
                )
                setattr(room, second_color, new_slot)
                room.backend = Backend()
                room.backend.new_game()
                room.backend.setup_clock(
                    time_minutes * 60, increment_seconds, now_provider=self._now,
                )
                room.skillcheck_secret = secrets.token_hex(16)
                room.started_at = self._now()
                room.mark_idle_activity(room.started_at)
                self._uuid_to_room[client_uuid] = room.room_id
                self._active[room.room_id] = room
                return room
            room_id = str(uuid4())
            tentative = (side_preference if side_preference != "random"
                         else random.choice(["white", "black"]))
            slot = PlayerSlot(
                client_uuid=client_uuid, nickname=nickname,
                session_token=session_token, side_preference=side_preference,
                country=country, hide_opp_marks=hide_opp_marks,
            )
            room = Room(
                room_id=room_id, time_minutes=time_minutes,
                increment_seconds=increment_seconds, created_at=self._now(),
            )
            setattr(room, tentative, slot)
            self._uuid_to_room[client_uuid] = room_id
            queue.append(room)
            return room

    @staticmethod
    def _resolve_colors(first_pref: str, second_pref: str) -> tuple[str, str]:
        """
        Settle who plays white when two players' side preferences meet. A
        stated preference wins, the waiting player's preference decides the
        arriving one when only they stated it, and two shrugs are coin-flipped

        :param first_pref: the waiting player's preference.
        :param second_pref: the arriving player's preference.
        :returns: the arriving player's colour, then the waiting player's.
        """
        if second_pref != "random":
            second = second_pref
        elif first_pref != "random":
            second = "black" if first_pref == "white" else "white"
        else:
            second = random.choice(["white", "black"])
        first = "black" if second == "white" else "white"
        return second, first

    async def reclaim_session(self, client_uuid: str) -> tuple[Room, str, str]:
        """
        Hand a returning player a fresh session token for the game they are
        still in, which is how the app finds its way back after a restart has
        lost the old one. The old token stops working immediately

        :param client_uuid: the player's stable id.
        :returns: the room, the colour they hold, and the new session token.
        """
        async with self._lock:
            room_id = self._uuid_to_room.get(client_uuid)
            if room_id is None or room_id not in self._active:
                raise NotInRoomError()
            room = self._active[room_id]
            for color in ("white", "black"):
                slot = room.slot(color)
                if slot is not None and slot.client_uuid == client_uuid:
                    new_token = self.make_session_token()
                    slot.session_token = new_token
                    return room, color, new_token
            raise NotInRoomError()

    async def cancel_wait(self, room_id: str, session_token: str) -> None:
        """
        Take a player out of the waiting queue at their own request. A game
        that has already started cannot be cancelled this way, and the token
        has to match the seat

        :param room_id: the room they were waiting in.
        :param session_token: token proving that seat is theirs.
        """
        async with self._lock:
            room = self._find_in_queue(room_id)
            if room is None:
                if room_id in self._active:
                    raise GameAlreadyStartedError()
                raise NotInRoomError()
            slot = room.white or room.black
            if slot is None or slot.session_token != session_token:
                raise InvalidTokenError()
            self._dequeue(room)
            self._uuid_to_room.pop(slot.client_uuid, None)

    def _find_in_queue(self, room_id: str) -> Room | None:
        """
        Search the waiting queues for a room by id, the slower lookup that
        exists because a waiting room is not in the active map yet

        :param room_id: the room's id.
        :returns: the queued room, or None when it is not waiting.
        """
        for queue in self._queue.values():
            for room in queue:
                if room.room_id == room_id:
                    return room
        return None

    def _dequeue(self, room: Room) -> bool:
        """
        Take one room out of its waiting queue, the single removal path every
        caller goes through. It also drops the time-control key once that
        queue empties, so the queue map cannot grow a bucket for every time
        control anybody has ever asked for

        :param room: the queued room to remove.
        :returns: True when the room was queued and has been removed.
        """
        tc = (room.time_minutes, room.increment_seconds)
        queue = self._queue.get(tc)
        if queue is None or room not in queue:
            return False
        queue.remove(room)
        if not queue:
            del self._queue[tc]
        return True

    def stale_queued_rooms(self, ttl_seconds: float = QUEUE_ABANDON_SECONDS) -> list[Room]:
        """
        List the rooms that have been waiting long enough to be worth a second
        look. The sweep uses it twice: at the shorter age to reap slots whose
        player has vanished, and at the full queue lifetime to give up on
        players nobody turned up for

        :param ttl_seconds: age since creation that counts as stale.
        :returns: the queued rooms older than that.
        """
        cutoff = self._now() - ttl_seconds
        return [room for queue in self._queue.values() for room in queue
                if room.created_at <= cutoff]

    def drop_queued_room(self, room: Room) -> bool:
        """
        Remove a waiting room and free its player to start something else,
        used when the wait was abandoned or has simply run out

        :param room: the queued room to drop.
        :returns: True when the room was still queued and has been dropped.
        """
        if not self._dequeue(room):
            return False
        for slot in (room.white, room.black):
            if slot is not None:
                self._uuid_to_room.pop(slot.client_uuid, None)
        return True

    def mark_connected(self, room_id: str, color: str) -> None:
        """
        Record that a player's socket is live again, clearing the disconnect
        stamp the grace period counts from along with any resync state left
        over from before

        :param room_id: the room the player is in.
        :param color: white or black.
        """
        room = self.get(room_id)
        if room is None:
            return
        slot = room.slot(color)
        if slot is not None:
            slot.connected = True
            slot.disconnected_at = None
            slot.desync_active = False
            slot.ply_mismatch_streak = 0
            slot.last_seen = self._now()

    def mark_disconnected(self, room_id: str, color: str) -> None:
        """
        Record that a player's socket is gone and start the grace period from
        this moment. A player already marked gone keeps their original stamp,
        so reconnect churn cannot quietly extend the grace period

        :param room_id: the room the player is in.
        :param color: white or black.
        """
        room = self.get(room_id)
        if room is None:
            return
        slot = room.slot(color)
        if slot is not None and slot.connected:
            slot.connected = False
            slot.disconnected_at = self._now()

    def touch_seen(self, room_id: str, color: str) -> None:
        """
        Note that a player is still talking to us, which is what keeps the
        heartbeat timeout from firing on them

        :param room_id: the room the player is in.
        :param color: white or black.
        """
        room = self.get(room_id)
        if room is None:
            return
        slot = room.slot(color)
        if slot is not None:
            slot.last_seen = self._now()

    def heartbeat_timed_out_rooms(self) -> Iterator[tuple[Room, str]]:
        """
        Walk the players whose heartbeat has been missing too long in a game
        that is properly under way, so the sweep can treat them as
        disconnected. Finished rooms and games that never started are skipped

        :returns: each silent player as their room and colour.
        """
        now = self._now()
        for room in list(self._active.values()):
            if room.result is not None or room.first_move_at is None:
                continue
            for color in ("white", "black"):
                slot = room.slot(color)
                if (slot is not None and slot.connected
                        and now - slot.last_seen >= HEARTBEAT_TIMEOUT_SECONDS):
                    yield room, color

    def in_progress_room_for(self, client_uuid: str) -> tuple[Room, str] | None:
        """
        Find the live game a player is in, which matchmaking consults before
        letting them start a second one. A room merely waiting for an
        opponent, or one already finished, does not count as live

        :param client_uuid: the player's stable id.
        :returns: the room and their colour, or None when there is no game.
        """
        room_id = self._uuid_to_room.get(client_uuid)
        if room_id is None:
            return None
        room = self._active.get(room_id)
        if room is None or room.result is not None:
            return None
        if not room.is_paired() or room.first_move_at is None:
            return None
        color = room.color_of(client_uuid)
        if color is None:
            return None
        return room, color

    def queued_room_for(self, client_uuid: str) -> Room | None:
        """
        Find the room a player is waiting in, so a fresh matchmake request can
        release the old queue slot instead of colliding with it

        :param client_uuid: the player's stable id.
        :returns: the queued room, or None when they are not waiting.
        """
        room_id = self._uuid_to_room.get(client_uuid)
        if room_id is None or room_id in self._active:
            return None
        return self._find_in_queue(room_id)

    def release_for_new_game(self, client_uuid: str) -> None:
        """
        Let go of whatever room a player is attached to, playing or waiting,
        so they can be seated somewhere new. An active room is dropped
        outright, since the game they walked out of is over for both of them

        :param client_uuid: the player's stable id.
        """
        room_id = self._uuid_to_room.get(client_uuid)
        if room_id is None:
            return
        if room_id in self._active:
            self._drop_room(room_id)
            return
        queued = self._find_in_queue(room_id)
        if queued is not None:
            self._dequeue(queued)
        self._uuid_to_room.pop(client_uuid, None)

    def grace_expired_rooms(self) -> Iterator[tuple[Room, str]]:
        """
        Walk the players who have been disconnected past the grace period, the
        candidates for losing by abandonment. It is only a shortlist -- the
        sweep re-checks each one before acting, because ending an earlier room
        awaits, and in that time a player may have come back

        :returns: each timed-out player as their room and colour.
        """
        now = self._now()
        for room in list(self._active.values()):
            if room.result is not None:
                continue
            for color in ("white", "black"):
                slot = room.slot(color)
                if (slot is not None and slot.disconnected_at is not None
                        and now - slot.disconnected_at >= GRACE_SECONDS):
                    yield room, color

    ZERO_PLY_ABORT_REASONS = (Reason.TIMEOUT, Reason.ABANDONMENT)

    def finalize_result(self, room_id: str, reason: str,
                        winner_color: str | None = None) -> bool:
        """
        Write the result onto a room once and for all, settling everything
        that ends along with the game: the series score, any pending check,
        the idle countdown, the shared marks, and both players moving to the
        result screen. It reports whether this call was the one that applied
        the result, so a finalise that lost a race stays quiet instead of
        announcing a second ending

        :param room_id: the room to finalise.
        :param reason: why the game ended, from the shared reason codes; a
            timeout or abandonment with no ply played is recorded as an abort.
        :param winner_color: the winning side, None for a draw or an abort.
        :returns: True when this call is the one that applied the result.
        """
        room = self._active.get(room_id)
        if room is None or room.result is not None:
            return False
        if room.plies_ever == 0 and reason in self.ZERO_PLY_ABORT_REASONS:
            reason, winner_color = Reason.ABORTED, None
        room.result = (reason, winner_color)
        room.ended_at = self._now()
        for slot in (room.white, room.black):
            if slot is not None and slot.disconnected_at is not None:
                slot.disconnected_at = room.ended_at
        room.last_rematch_activity_at = room.ended_at
        room.pending_skillcheck = None
        room.idle_since = None
        room.idle_pushed_at = None
        room.annotations_white.reset()
        room.annotations_black.reset()
        for slot in (room.white, room.black):
            if slot is not None:
                slot.at_result = True
        self._award_series(room, reason, winner_color)
        return True

    @staticmethod
    def _award_series(room: Room, reason: str, winner_color: str | None) -> None:
        """
        Add this game to the room's running series score: a point to the
        winner, half each for a draw. A game that never really happened --
        aborted, or cut short by the server going down -- scores nothing

        :param room: the room being finalised.
        :param reason: why the game ended.
        :param winner_color: the winning side, None for a draw or an abort.
        """
        if reason in (Reason.ABORTED, Reason.SERVER_SHUTDOWN):
            return
        scores = room.series_scores
        if winner_color in ("white", "black"):
            slot = room.slot(winner_color)
            if slot is not None:
                scores[slot.nickname] = scores.get(slot.nickname, 0.0) + 1.0
        elif reason.startswith("draw"):
            for slot in (room.white, room.black):
                if slot is not None:
                    scores[slot.nickname] = scores.get(slot.nickname, 0.0) + 0.5

    def finished_timed_out(self, room: Room) -> bool:
        """
        Tell whether a finished room has sat around long enough to be cleaned
        up, either idle since the last sign of rematch interest or simply open
        too long however lively it has been

        :param room: a room that already carries a result.
        :returns: True when the room may be dropped.
        """
        if room.result is None or room.ended_at is None:
            return False
        now = self._now()
        last = room.last_rematch_activity_at or room.ended_at
        return (now - last >= REMATCH_IDLE_SECONDS
                or now - room.ended_at >= REMATCH_ABSOLUTE_CAP_SECONDS)

    def gc_finished_rooms(self) -> None:
        """
        Drop the finished rooms whose rematch window has closed, which is what
        stops a server that has been up for weeks from holding on to every
        game it ever ran
        """
        to_drop = [
            room_id for room_id, room in self._active.items()
            if self.finished_timed_out(room)
        ]
        for room_id in to_drop:
            self._drop_room(room_id)

    def mark_rematch_activity(self, room: Room) -> None:
        """
        Note that somebody is still interested in a rematch, which pushes back
        the idle half of the post-game window

        :param room: the finished room being kept alive.
        """
        room.last_rematch_activity_at = self._now()

    def finished_room_for(self, client_uuid: str) -> Room | None:
        """
        Find the finished room a player is still attached to, so leaving for a
        new game can tell their old opponent they have gone

        :param client_uuid: the player's stable id.
        :returns: the finished room, or None when there is none.
        """
        room_id = self._uuid_to_room.get(client_uuid)
        if room_id is None:
            return None
        room = self._active.get(room_id)
        if room is None or room.result is None:
            return None
        return room

    def drop_room_now(self, room_id: str) -> None:
        """
        Drop an active room straight away, for the moments when it has nothing
        left to wait for: a declined rematch, or both players gone

        :param room_id: the room to drop.
        """
        if room_id in self._active:
            self._drop_room(room_id)

    def _drop_room(self, room_id: str) -> None:
        """
        Forget a room and release both of its players, the single place a room
        stops existing. Going through here is what stops a player being left
        pointing at a room that is gone

        :param room_id: the room to forget.
        """
        room = self._active.pop(room_id, None)
        if room is None:
            return
        for slot in (room.white, room.black):
            if slot is not None:
                self._uuid_to_room.pop(slot.client_uuid, None)

    def reset_for_rematch(self, room_id: str) -> bool:
        """
        Turn a finished room back into a fresh game with the colours swapped:
        new board and clocks, a new skill-check secret, cleared marks, offers
        and result, and the ply counter back to zero. The series scores are
        the one thing that survives, since they are the point of a series

        :param room_id: the finished room to replay.
        :returns: True when the room could be replayed and has been reset.
        """
        room = self._active.get(room_id)
        if room is None or room.result is None:
            return False
        old_white, old_black = room.white, room.black
        room.white, room.black = old_black, old_white
        room.backend = Backend()
        room.backend.new_game()
        room.backend.setup_clock(
            room.time_minutes * 60, room.increment_seconds, now_provider=self._now,
        )
        room.skillcheck_secret = secrets.token_hex(16)
        room.skillcheck_locks = set()
        room.skillcheck_log = []
        room.pending_skillcheck = None
        room.plies_ever = 0
        room.annotations_white = SharedAnnotations()
        room.annotations_black = SharedAnnotations()
        room.history_changes.clear()
        room.started_at = self._now()
        room.first_move_at = None
        room.mark_idle_activity(room.started_at)
        room.ended_at = None
        room.last_rematch_activity_at = None
        room.game_start_broadcast = False
        room.draw_offered_by = None
        room.takeback_offered_by = None
        room.rematch_offered_by = set()
        room.result = None
        for slot in (room.white, room.black):
            if slot is not None:
                slot.at_result = False
                slot.ply_mismatch_streak = 0
        return True

    @staticmethod
    def make_session_token() -> str:
        """
        Mint the token for one seat. It is the secret that proves a websocket
        or a resume request belongs to that player, so it comes from the
        cryptographic generator rather than anything guessable

        :returns: a fresh URL-safe session token.
        """
        return secrets.token_urlsafe(24)
