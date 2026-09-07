import ast
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chessshootout.backend.backend import Backend, DEFAULT_CASTLING_RIGHTS
from chessshootout.backend.utils import Square, Move, MoveResult, HistoryEntry
from chessshootout.backend.pieces import Piece, PieceColor, PieceType
from chessshootout.server.protocol import PROTOCOL_VERSION

if TYPE_CHECKING:
    import pygame
    import pytest

    from chessshootout.frontend.frontend import Frontend


WHITE = PieceColor.WHITE
BLACK = PieceColor.BLACK
NO_CASTLING = {"WK": False, "WQ": False, "BK": False, "BQ": False}

K = PieceType.KING
Q = PieceType.QUEEN
R = PieceType.ROOK
B = PieceType.BISHOP
N = PieceType.KNIGHT
P = PieceType.PAWN


def sq(row: int, col: int) -> Square:
    """
    Build a board square from raw indices, the terse spelling fixtures use when
    they lay pieces out by hand. Row 0 is Black's back rank and column 0 is the
    a-file, exactly as the engine numbers them

    :param row: board row, 0 at Black's back rank
    :param col: board column, 0 at the a-file
    :returns: the square at those indices
    """
    return Square(row, col)


def make_backend(piece_map: dict[Square, Piece], turn: PieceColor = WHITE,
                 castling_rights: dict[str, bool] | None = None,
                 ep_target: Square | None = None,
                 halfmove_clock: int = 0) -> Backend:
    """
    Build an engine holding exactly the pieces a test names, skipping new_game()
    so a rules case can be set up in two lines rather than played into being.
    The move history starts empty and the position is counted once for
    repetition, so the board behaves like a freshly loaded one

    :param piece_map: pieces to place, keyed by the square each stands on
    :param turn: side to move in the position being built
    :param castling_rights: flags keyed by WK, WQ, BK and BQ, or None to grant
        all four
    :param ep_target: square a pawn may be captured on en passant, or None
    :param halfmove_clock: plies since the last pawn move or capture, for
        fifty-move cases
    :returns: the ready-to-use engine
    """
    backend = Backend()
    for s, piece in piece_map.items():
        backend.state[s.row][s.col] = piece
    backend.turn = turn
    backend.castling_rights = (
        dict(castling_rights) if castling_rights is not None
        else dict(DEFAULT_CASTLING_RIGHTS)
    )
    backend.en_passant_target = ep_target
    backend.halfmove_clock = halfmove_clock
    backend.move_history = []
    backend.position_counts = Counter()
    backend.position_counts[backend._position_key()] = 1
    return backend


def piece(piece_type: PieceType, color: PieceColor) -> Piece:
    """
    Build one piece for a fixture board. It exists so piece maps read tersely
    beside the single-letter type shorthands this module defines

    :param piece_type: which of the six kinds to build
    :param color: side that owns it
    :returns: the new piece
    """
    return Piece(piece_type, color)


def play_moves(backend: Backend, moves: Iterable[tuple[Square, Square]]) -> MoveResult | None:
    """
    Play a run of moves through the real engine, asserting every one of them is
    legal, so a test can reach an interesting position without checking each ply
    itself. The run stops at the first rejection, naming the move that was
    refused

    :param backend: engine to play on; it is mutated in place
    :param moves: (from, to) square pairs in play order
    :returns: the last move's result, or None when no moves were given
    """
    last = None
    for from_sq, to_sq in moves:
        last = backend.try_move(from_sq, to_sq)
        assert last.legal, f"Move {from_sq} -> {to_sq} was rejected"
    return last


def sq_of(algebraic: str) -> Square:
    """
    Turn an algebraic square name into engine coordinates, so a test can write
    e4 instead of spelling out row and column. Rank 8 is row 0 and file a is
    column 0

    :param algebraic: two-character square name such as e4
    :returns: the matching square
    """
    col = "abcdefgh".index(algebraic[0])
    row = 8 - int(algebraic[1])
    return Square(row, col)


def squares(text: str) -> set[Square]:
    """
    Turn a space-separated list of algebraic square names into a set, the
    compact way move-set expectations are written. An empty string yields the
    empty set, which is how a test states that a piece has no moves at all

    :param text: square names separated by spaces, possibly empty
    :returns: the named squares as a set
    """
    return {sq_of(token) for token in text.split()}


def assert_legal_moves(backend: Backend, from_sq: Square | str,
                       expected: str | Iterable[Square]) -> None:
    """
    Assert that a piece's legal destinations are exactly the expected set, the
    check the movement tests are mostly written out of. Both squares and
    expectations accept the terse algebraic spelling, and a failure reports what
    was missing and what was extra instead of dumping both sets

    :param backend: engine holding the position
    :param from_sq: square the piece stands on, as a name or a Square
    :param expected: the exact destinations, as a space-separated string or as
        Squares
    """
    if isinstance(from_sq, str):
        from_sq = sq_of(from_sq)
    expected = squares(expected) if isinstance(expected, str) else set(expected)
    actual = set(backend.legal_moves_from(from_sq))
    assert actual == expected, (
        f"from {from_sq}: missing={expected - actual} extra={actual - expected}"
    )


class FakeClock:
    """
    A stand-in for time.monotonic that only moves when a test tells it to. It is
    passed wherever the engine or the server takes a now_provider, so clocks,
    grace periods and idle windows can be exercised without waiting in real time
    """

    def __init__(self) -> None:
        """
        Start the fake clock at zero seconds, the baseline every test advances
        away from
        """
        self.t = 0.0

    def __call__(self) -> float:
        """
        Report the fake clock's current reading, which is what lets an instance
        be passed anywhere a now_provider callable is expected

        :returns: current fake time in seconds
        """
        return self.t

    def advance(self, dt: float) -> None:
        """
        Move the fake clock forward, the way a test says that time passed
        between two engine or server steps

        :param dt: seconds to add to the current reading
        """
        self.t += dt

    def set(self, t: float) -> None:
        """
        Jump the fake clock to an absolute reading, for a test that cares about
        a particular timestamp rather than an elapsed amount

        :param t: new reading of the clock in seconds
        """
        self.t = t


def fake_uuid4(seed: int) -> str:
    """
    Build a deterministic UUID4-shaped player id from a small integer, so server
    tests can name players readably and still satisfy the protocol's UUID4
    validation. The version and variant nibbles are forced into range, and one
    seed always produces the same id

    :param seed: integer standing for an identity, such as 1 for Alice
    :returns: a UUID4-shaped string the protocol validators accept
    """
    n = int(seed)
    hex_pad = f"{n:032x}"
    return (
        f"{hex_pad[0:8]}-{hex_pad[8:12]}-4{hex_pad[13:16]}-"
        f"8{hex_pad[17:20]}-{hex_pad[20:32]}"
    )


def make_app(w: int = 1000, h: int = 800, *, mock_sound: bool = True) -> "Frontend":
    """
    Boot the whole app shell on the menu screen at a given window size, the
    starting point nearly every frontend test builds on. No game is started --
    pair it with start_single_screen() for that

    :param w: window width in pixels
    :param h: window height in pixels
    :param mock_sound: True swaps the sound manager for a mock so tests never
        touch the real mixer; the mock carries plausible volume and enabled
        values, so widgets reading them survive a plain draw_frame()
    :returns: the booted app shell
    """
    from unittest.mock import MagicMock

    from chessshootout.frontend.frontend import Frontend

    app = Frontend(w, h)
    if mock_sound:
        app.sound_manager = MagicMock(master_volume=1.0, menu_volume=1.0, enabled=True)
    return app


def start_single_screen(app: "Frontend", *, nickname: str = "alice", side: str = "white",
                        time_minutes: int = 5, increment_seconds: int = 0) -> "Frontend":
    """
    Start a local single-screen game on an app a test already built, driving the
    very start-menu callback a real player triggers rather than poking the game
    screen directly. The navigation intent is queued and run at once, so the
    game is live by the time this returns

    :param app: app shell to start the game on
    :param nickname: name shown for the local player
    :param side: side the local player takes, white or black
    :param time_minutes: starting time per side in minutes
    :param increment_seconds: seconds added to a player's clock after each move
    :returns: the same app, now sitting on a live game screen
    """
    app._on_start_game({
        "mode": "single_screen", "nickname": nickname, "side": side,
        "time_minutes": time_minutes, "increment_seconds": increment_seconds,
    })
    return app


def assert_pixel_color(surface: "pygame.Surface", x: float, y: float,
                       expected: str | Sequence[int], tol: int = 0) -> None:
    """
    Assert that one pixel of a surface carries the expected colour, the
    finest-grained check the pixel tests make. Only the RGB channels are
    compared, so a difference in alpha never fails it

    :param surface: surface to sample; tests only sample surfaces they own
    :param x: pixel column
    :param y: pixel row
    :param expected: hex colour string (a Colors attribute) or an (r, g, b)
        sequence
    :param tol: allowed per-channel difference; 0 demands an exact match
    """
    import pygame as pg

    got = surface.get_at((int(x), int(y)))[:3]
    want = pg.Color(expected)[:3] if isinstance(expected, str) else tuple(expected)[:3]
    if tol == 0:
        assert got == want, f"pixel ({x},{y}) was {got}, expected {want}"
    else:
        assert all(abs(g - w) <= tol for g, w in zip(got, want)), \
            f"pixel ({x},{y}) was {got}, expected ~{want} (tol {tol})"


def rgb(color: str | Sequence[int]) -> tuple[int, int, int]:
    """
    Reduce a colour written either way -- a hex string like the Colors
    attributes, or an (r, g, b) sequence -- to a plain triple. Alpha is dropped,
    which is what lets the pixel comparisons ignore it

    :param color: hex colour string or an (r, g, b) sequence
    :returns: the colour as an (r, g, b) triple
    """
    import pygame as pg

    return pg.Color(color)[:3] if isinstance(color, str) else tuple(color)[:3]


def scan_region(surface: "pygame.Surface", rect: "pygame.Rect",
                color: str | Sequence[int], *, tol: int = 0, step: int = 1,
                count: bool = False, clamp: bool = False) -> bool | int:
    """
    Scan a rectangle of a surface for pixels of a given colour, and answer
    either whether any matched or how many did. It is the one source for the
    region colour scans that used to be copy-pasted per test file, each with its
    own tolerance and stride semantics

    :param surface: surface to sample; tests only sample surfaces they own
    :param rect: region to scan, in surface pixels
    :param color: hex colour string or an (r, g, b) sequence to look for
    :param tol: allowed per-channel difference; 0 demands an exact match
    :param step: stride on both axes, so 2 samples every other pixel
    :param count: False answers whether any pixel matched, True how many did
    :param clamp: True keeps the scanned region inside the surface bounds
    :returns: a bool when count is False, otherwise the number of matches
    """
    want = rgb(color)
    x0, y0, x1, y1 = rect.x, rect.y, rect.right, rect.bottom
    if clamp:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, surface.get_width()), min(y1, surface.get_height())

    def matches(px: "pygame.Color") -> bool:
        """
        Test one sampled pixel against the wanted colour, honouring whatever
        tolerance the scan was asked for

        :param px: sampled pixel, whose alpha channel is ignored
        :returns: True when the pixel counts as a match
        """
        if tol == 0:
            return px[:3] == want
        return all(abs(a - b) <= tol for a, b in zip(px[:3], want))

    hits = (
        matches(surface.get_at((x, y)))
        for x in range(x0, x1, step)
        for y in range(y0, y1, step)
    )
    return sum(1 for hit in hits if hit) if count else any(hits)


def valid_pgn_text() -> str:
    """
    Render a short, legal three-ply game as PGN text, for the menu history and
    navigation views that only need some loadable game on disk. It is always the
    same White win between two fixed names

    :returns: PGN text of the sample game
    """
    from chessshootout.backend.backend import Backend
    from chessshootout.domain.pgn.generate import generate_pgn

    backend = Backend()
    backend.new_game()
    for san in ["e4", "e5", "Nf3"]:
        backend.apply_san(san)
    return generate_pgn(backend.move_history, "white_wins",
                        white_name="alice", black_name="Bob")


def freeze_ticks(monkeypatch: "pytest.MonkeyPatch") -> dict[str, int]:
    """
    Freeze pygame's millisecond tick counter behind a mutable holder, so tween
    and transition tests step animation time deliberately instead of racing the
    real clock. Raise the holder's ms entry to advance the frozen clock

    :param monkeypatch: the test's monkeypatch fixture, which owns the undo
    :returns: the holder whose ms key drives the frozen clock
    """
    import pygame as pg

    holder = {"ms": 100_000}
    monkeypatch.setattr(pg.time, "get_ticks", lambda: holder["ms"])
    return holder


def fire_animation(board: Any) -> None:
    """
    Force every in-flight board animation to its end and draw the final frame,
    so an animated move lands at once in a review or board test instead of over
    the next several frames

    :param board: board whose animations are being finished
    """
    import pygame as pg

    for a in list(board.animations):
        a.start_ms = pg.time.get_ticks() - 10_000
    board._draw_animations()


def key_event(key: int, unicode: str = "") -> "pygame.event.Event":
    """
    Build a KEYDOWN event with no modifier keys held, the shape a test hands to
    a screen's key handler

    :param key: pygame key constant, such as pg.K_LEFT
    :param unicode: character the key produced, for the text-entry cases
    :returns: the pygame event
    """
    import pygame as pg

    return pg.event.Event(pg.KEYDOWN, key=key, mod=0, unicode=unicode)


def click_event(pos: tuple[int, int], button: int = 1) -> "pygame.event.Event":
    """
    Build a MOUSEBUTTONDOWN event at a position, the shape a test hands to the
    input router or straight to a widget

    :param pos: click position in window pixels
    :param button: pygame mouse button number; 1 is the left button
    :returns: the pygame event
    """
    import pygame as pg

    return pg.event.Event(pg.MOUSEBUTTONDOWN, {"button": button, "pos": pos})


def history_entry(san: str) -> HistoryEntry:
    """
    Build a minimal history entry carrying a pawn move and the notation given,
    for the move-list scroll tests that only need a certain number of filler
    rows

    :param san: notation the row should display
    :returns: a history entry safe to append to a move list
    """
    move = Move(Square(6, 0), Square(5, 0), Piece(PieceType.PAWN, PieceColor.WHITE))
    return HistoryEntry(move=move, prev_castling_rights=(), prev_en_passant_target=None,
                        prev_halfmove_clock=0, position_key_added=("k",), san=san)


def draw_strip(strip: Any) -> None:
    """
    Blank a strip's window and redraw just that strip, the setup step the player
    and review strip pixel tests take before they sample any colours

    :param strip: player or review strip to redraw
    """
    strip.window.fill((0, 0, 0))
    strip.draw()


def strip_avatar_pixels(strip: Any) -> tuple["pygame.Color", "pygame.Color"]:
    """
    Sample the top and the bottom of a strip's flat avatar, using the same
    geometry the strip itself places it with, so a test can tell which palette
    was drawn

    :param strip: player or review strip to sample
    :returns: the colours found near the top and near the bottom of the avatar
    """
    pad = max(int(strip.rect.height * 0.16), 4)
    av = strip.rect.height - 2 * pad
    cx = strip.rect.x + pad + av // 2
    top = strip.window.get_at((cx, strip.rect.y + pad + max(3, av // 6)))
    bottom = strip.window.get_at((cx, strip.rect.y + pad + av - max(3, av // 6)))
    return top, bottom


def write_pgn_fixture(tmp_path: Path, name: str, white: str, black: str, result: str,
                      moves: str = "1. e4 e5") -> Path:
    """
    Write a minimal single-game PGN into a games folder under a test's temporary
    directory, shared by the profile and rail-card views that scan that folder
    for stats. The folder is created when it is not there yet

    :param tmp_path: the test's temporary directory
    :param name: filename to write, extension included
    :param white: name for the White player tag
    :param black: name for the Black player tag
    :param result: PGN result tag, which also terminates the move text
    :param moves: move text making up the body of the game
    :returns: path of the file that was written
    """
    games = tmp_path / "games"
    games.mkdir(exist_ok=True)
    path = games / name
    path.write_text(
        f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n\n{moves} {result}\n',
        encoding="utf-8")
    return path


def read_source_without_docstrings(path: str | Path) -> str:
    """
    Read a Python source file with every inert prose statement blanked.
    Parses the file and erases the lines of every expression statement
    whose value is a plain string constant -- module, class, and def
    docstrings plus bare attribute docstrings -- replacing them with
    empty lines so every surviving line keeps its original line number.
    Real code, including string literals used as values, is untouched.
    Guard tests that scan source text call this instead of read_text()
    so prose can never trip a code-shape tripwire

    :param path: path to the .py file to read
    :returns: the docstring-blind source text, line count preserved
    """
    source = Path(path).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = ""
    return "\n".join(lines) + "\n"


def online_start_payload(**overrides: Any) -> dict[str, Any]:
    """
    Build the game-start payload the online coordinator consumes, defaulting to
    White for alice against bob at five minutes with no increment. Pass
    overrides to vary a single field without restating the rest

    :param overrides: payload fields to replace in the default
    :returns: the match-found payload
    """
    payload = {
        "your_color": "white", "white_name": "alice", "black_name": "bob",
        "time_minutes": 5, "increment_seconds": 0,
    }
    payload.update(overrides)
    return payload


def auth_msg(token: str) -> dict[str, Any]:
    """
    Build the handshake frame a client must send first on a game websocket: the
    protocol version the server insists on, plus the session token naming the
    slot being claimed. No game traffic is accepted before it

    :param token: session token handed out by matchmake, resume or reclaim
    :returns: the auth frame, ready to send as JSON
    """
    return {"version": PROTOCOL_VERSION, "type": "auth", "session_token": token}
