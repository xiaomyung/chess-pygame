import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

import chessshootout
from chessshootout.server.app import MAX_INBOUND_MESSAGE_BYTES
from chessshootout.server.protocol import (
    AnnotationDeltaMessage, AnnotationSetWire, AnnotationsStateMessage, ArrowWire,
    AuthMessage, CHAT_PRESET_COUNT, CLIENT_VERSION_MAX_LEN, ClockSnapshot, ErrorMessage,
    FIRST_MOVE_ABORT_SECONDS, GameStartMessage, HealthResponse, HealthStatus,
    IDLE_RESIGN_SECONDS, MIN_CLIENT_VERSION,
    IDLE_SECONDS_BY_OUTCOME, IDLE_WINDOW_BY_PLIES, IdleWindowMessage, IdleWindowWire,
    LockWire, MAX_INCREMENT_SECONDS, MAX_SHARED_ARROWS, MAX_SHARED_HIGHLIGHTS,
    MAX_TIME_MINUTES, MIN_INCREMENT_SECONDS, MIN_TIME_MINUTES, MatchmakeRequest,
    MoveAppliedMessage, MoveMessage, PROTOCOL_VERSION, PendingSkillCheckWire,
    PingMessage, PongMessage, QuickChatMessage, QuickChatReceivedMessage, Reason,
    ResumeResponse, ResyncDirectiveMessage, SkillCheckRequiredMessage,
    SkillCheckResultMessage, SkillCheckOutcomeWire, SkillCheckShotMessage,
    SkillCheckSpectateMessage, SkillCheckSpectateShotMessage, client_version_outdated,
    normalize_country, normalize_nickname, parse_client_version,
)
from tests.helpers import fake_uuid4


U1 = fake_uuid4(1)

GAME_START = GameStartMessage(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    white_name="Alice", black_name="Bob",
    time_minutes=5, increment_seconds=0,
    your_color="white",
)


@pytest.mark.parametrize(
    "msg, expected",
    [
        pytest.param(
            PingMessage(ply=7),
            {"version": PROTOCOL_VERSION, "type": "ping", "ply": 7},
            id="ping",
        ),
        pytest.param(
            ResyncDirectiveMessage(server_ply=12),
            {"version": PROTOCOL_VERSION, "type": "resync_directive", "server_ply": 12},
            id="resync_directive",
        ),
        pytest.param(
            GAME_START,
            {
                "version": PROTOCOL_VERSION,
                "type": "game_start",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "white_name": "Alice",
                "black_name": "Bob",
                "time_minutes": 5,
                "increment_seconds": 0,
                "your_color": "white",
                "started_seconds_ago": 0.0,
                "white_score": 0.0,
                "black_score": 0.0,
                "white_country": None,
                "black_country": None,
                "heartbeat_interval_seconds": 2.0,
                "grace_seconds": 60.0,
                "rematch": False,
            },
            id="game_start",
        ),
    ],
)
def test_message_round_trip_matches_independent_wire_shape(msg, expected):
    """model_dump matches a hand-written wire dict, and the dict re-parses to an equal model."""
    assert msg.model_dump() == expected
    assert type(msg).model_validate(expected) == msg


@pytest.mark.parametrize(
    "msg",
    [
        pytest.param(AuthMessage(session_token="t"), id="auth"),
        pytest.param(ErrorMessage(reason="version_mismatch"), id="error"),
        pytest.param(PingMessage(ply=0), id="ping"),
        pytest.param(PongMessage(), id="pong"),
        pytest.param(ResyncDirectiveMessage(server_ply=0), id="resync_directive"),
    ],
)
def test_messages_carry_protocol_version(msg):
    assert msg.version == PROTOCOL_VERSION


def test_a_heartbeat_without_a_ply_is_a_client_that_is_off_the_board():
    """The default has to be None rather than 0: a heartbeat from a client that
    is not on a live board must claim nothing, and 0 is a real ply that a fresh
    game legitimately sits on."""
    assert PingMessage().ply is None
    assert PingMessage.model_validate_json('{"type": "ping"}').ply is None
    assert PingMessage.model_validate_json('{"type": "ping", "ply": null}').ply is None


def test_a_heartbeat_ply_is_bounded_at_zero():
    """`ply` is client-supplied, so it is bounded by the model rather than by
    hand in the handler -- a negative ply can never match and would otherwise
    walk straight into the strike counter."""
    assert PingMessage.model_validate_json('{"type": "ping", "ply": 0}').ply == 0
    with pytest.raises(ValidationError):
        PingMessage(ply=-1)


def test_a_resync_directive_always_names_the_ply_it_was_written_against():
    """The client drops a directive it has already outrun by comparing this
    against its own ply, so the field is required rather than defaulted -- a
    default would make every directive look like it was about ply 0."""
    with pytest.raises(ValidationError):
        ResyncDirectiveMessage()
    parsed = ResyncDirectiveMessage.model_validate_json(
        '{"type": "resync_directive", "server_ply": 4}')
    assert parsed.server_ply == 4


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param("  alice  ", "alice", id="strips_surrounding"),
        pytest.param("a   b   c", "a b c", id="collapses_internal_runs"),
        pytest.param("Alice 42!", "Alice 42!", id="keeps_printable_ascii"),
    ],
)
def test_normalize_nickname_accepts(raw, expected):
    assert normalize_nickname(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace_only"),
        pytest.param("x" * 21, id="too_long"),
        pytest.param("alice\x00", id="null_byte"),
        pytest.param("alice\n", id="newline"),
    ],
)
def test_normalize_nickname_rejects(raw):
    with pytest.raises(ValueError):
        normalize_nickname(raw)


@pytest.mark.parametrize(
    "kwargs, attr, expected",
    [
        pytest.param(
            {"nickname": "  Alice  "}, "nickname", "Alice",
            id="normalizes_nickname",
        ),
        pytest.param({}, "side_preference", "random", id="default_side_random"),
    ],
)
def test_matchmake_request_accepts(kwargs, attr, expected):
    base = {
        "nickname": "Alice", "client_uuid": U1,
        "time_minutes": 5, "increment_seconds": 0,
    }
    req = MatchmakeRequest(**{**base, **kwargs})
    assert getattr(req, attr) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"time_minutes": -1}, id="negative_time"),
        pytest.param({"side_preference": "middle"}, id="bad_side"),
        pytest.param({"time_minutes": 0}, id="zero_time"),
        pytest.param({"increment_seconds": -1}, id="negative_increment"),
        pytest.param({"time_minutes": MAX_TIME_MINUTES + 1}, id="time_over_cap"),
        pytest.param({"increment_seconds": MAX_INCREMENT_SECONDS + 1},
                     id="increment_over_cap"),
        pytest.param({"time_minutes": 2 ** 40}, id="absurd_time"),
        pytest.param({"increment_seconds": 2 ** 40}, id="absurd_increment"),
    ],
)
def test_matchmake_request_rejects(kwargs):
    """SECURITY: `_queue` is a defaultdict keyed by the client-supplied
    (time_minutes, increment_seconds) pair, so unbounded ints let one client mint
    an unbounded number of never-pairing queue buckets. Both fields are capped at
    the model boundary, which is the same 422 path every other malformed
    matchmake field takes."""
    base = {
        "nickname": "Alice", "client_uuid": U1,
        "time_minutes": 5, "increment_seconds": 0,
    }
    with pytest.raises(ValidationError):
        MatchmakeRequest(**{**base, **kwargs})


@pytest.mark.parametrize(
    "minutes, increment",
    [
        pytest.param(MIN_TIME_MINUTES, MIN_INCREMENT_SECONDS, id="both_at_minimum"),
        pytest.param(MAX_TIME_MINUTES, MAX_INCREMENT_SECONDS, id="both_at_maximum"),
        pytest.param(MAX_TIME_MINUTES, MIN_INCREMENT_SECONDS, id="max_time_no_increment"),
    ],
)
def test_matchmake_request_accepts_the_bounds_themselves(minutes, increment):
    """The caps are inclusive: the boundary values are legal input, only what is
    strictly outside them is refused."""
    req = MatchmakeRequest(nickname="Alice", client_uuid=U1,
                           time_minutes=minutes, increment_seconds=increment)
    assert (req.time_minutes, req.increment_seconds) == (minutes, increment)


CLIENT_MINUTES = (1, 3, 5, 10, 15, 30)
CLIENT_INCREMENTS = (0, 2, 5, 10, 15)
"""Every time control the shipped picker can emit -- CHAMBERS / INCREMENTS in
chessshootout/frontend/menu/time_picker.py (the picker's remaining chamber is
the local-only infinity setting, which the online path substitutes with a real
minute count before it ever reaches the wire)."""


def test_client_time_control_tables_match_the_shipped_picker():
    """Drift guard: the tables above are a hand copy of the picker's, and a copy
    that silently falls behind turns the fence below into a fence around nothing --
    a newly added chamber would never be checked against the wire caps at all. The
    frontend import is deliberate and stays inside the test: the no-pygame guard
    (tests/infra/test_server_no_pygame.py) scans package SOURCE under server/, not
    tests/. The infinity chamber is excluded on purpose -- it carries no minute
    count and the online path substitutes a real one before matchmaking."""
    from chessshootout.frontend.menu.time_picker import CHAMBERS, INCREMENTS

    assert CLIENT_MINUTES == tuple(minutes for minutes, _ in CHAMBERS if minutes is not None)
    assert CLIENT_INCREMENTS == tuple(INCREMENTS)
    assert sum(1 for minutes, _ in CHAMBERS if minutes is None) == 1, \
        "exactly one non-numeric chamber (∞) is expected to be excluded"


@pytest.mark.parametrize("minutes", CLIENT_MINUTES)
@pytest.mark.parametrize("increment", CLIENT_INCREMENTS)
def test_matchmake_request_accepts_every_client_producible_time_control(minutes, increment):
    """Regression fence for the caps: the bounds must sit far outside what the UI
    can actually produce, so tightening them can never lock real players out."""
    req = MatchmakeRequest(nickname="Alice", client_uuid=U1,
                           time_minutes=minutes, increment_seconds=increment)
    assert (req.time_minutes, req.increment_seconds) == (minutes, increment), \
        "the model preserves the picker's values instead of coercing them"
    assert MIN_TIME_MINUTES <= req.time_minutes <= MAX_TIME_MINUTES
    assert MIN_INCREMENT_SECONDS <= req.increment_seconds <= MAX_INCREMENT_SECONDS


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param("US", "US", id="passthrough"),
        pytest.param("ro", "RO", id="uppercases"),
        pytest.param("  gb  ", "GB", id="strips"),
    ],
)
def test_normalize_country_accepts(raw, expected):
    assert normalize_country(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("USA", id="three_letters"),
        pytest.param("U1", id="digit"),
        pytest.param("longstring", id="too_long"),
        pytest.param(123, id="non_string"),
    ],
)
def test_normalize_country_nulls_invalid(raw):
    assert normalize_country(raw) is None


def test_matchmake_request_country_defaults_none():
    req = MatchmakeRequest(nickname="Alice", client_uuid=U1,
                           time_minutes=5, increment_seconds=0)
    assert req.country is None


def test_matchmake_request_normalizes_country():
    req = MatchmakeRequest(nickname="Alice", client_uuid=U1,
                           time_minutes=5, increment_seconds=0, country="  us ")
    assert req.country == "US"


@pytest.mark.parametrize("bad", ["usa", "1", 999, "no-country!"])
def test_matchmake_request_bad_country_nulls_not_rejects(bad):
    """A malformed country must never deny a match — it nulls out instead of raising."""
    req = MatchmakeRequest(nickname="Alice", client_uuid=U1,
                           time_minutes=5, increment_seconds=0, country=bad)
    assert req.country is None


def test_game_start_backward_compat_without_country():
    """An old payload lacking the country keys still parses, defaulting to None."""
    wire = {
        "version": PROTOCOL_VERSION, "type": "game_start",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "white_name": "Alice", "black_name": "Bob",
        "time_minutes": 5, "increment_seconds": 0, "your_color": "white",
    }
    msg = GameStartMessage.model_validate(wire)
    assert msg.white_country is None and msg.black_country is None


def test_game_start_carries_country():
    msg = GameStartMessage(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        white_name="Alice", black_name="Bob", time_minutes=5,
        increment_seconds=0, your_color="white",
        white_country="US", black_country="RO",
    )
    assert (msg.white_country, msg.black_country) == ("US", "RO")


def test_move_message_uses_alias_for_from():
    raw = {"type": "move", "version": PROTOCOL_VERSION, "from": "e2", "to": "e4", "promotion": None}
    msg = MoveMessage.model_validate(raw)
    assert msg.from_sq == "e2"
    assert msg.to_sq == "e4"
    assert msg.promotion is None
    assert msg.model_dump(by_alias=True) == {
        "version": PROTOCOL_VERSION, "type": "move",
        "from": "e2", "to": "e4", "promotion": None,
    }


def test_move_message_promotion_validated():
    with pytest.raises(ValidationError):
        MoveMessage.model_validate({"type": "move", "version": PROTOCOL_VERSION,
                                    "from": "e7", "to": "e8", "promotion": "x"})


def test_clock_snapshot_requires_the_running_side_to_be_stated():
    """running_for carries no default ON PURPOSE. It is decoded from server payloads
    on the client, so a truncated clock -- the key missing entirely -- has to fail
    validation and route into the client's error path. A default of None would make
    that payload parse and silently freeze both clocks instead. Both producers pass
    the field explicitly, so requiring it costs the server nothing."""
    with pytest.raises(ValidationError) as info:
        ClockSnapshot(white_remaining=300.0, black_remaining=180.0)
    assert [err["loc"] for err in info.value.errors()] == [("running_for",)]
    stopped = ClockSnapshot(white_remaining=300.0, black_remaining=180.0, running_for=None)
    assert stopped.model_dump() == {
        "white_remaining": 300.0, "black_remaining": 180.0, "running_for": None,
    }


def test_clock_snapshot_rejects_an_unknown_running_side():
    """Only the two colours parse: a stated side is still a closed Literal, so no
    third value can reach a client clock through this field."""
    with pytest.raises(ValidationError):
        ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="green")


def test_protocol_version_pinned_for_the_version_gate():
    """Bumped to 6 with the matchmake version gate (the resync directive's
    server_ply rides on the same bump). The pin is what makes a protocol change
    a deliberate edit: every pre-gate build is on 5 and is refused at /matchmake,
    and every wire model in this file is stamped with this number."""
    assert PROTOCOL_VERSION == 6


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param("2.13.0", (2, 13, 0), id="plain_release"),
        pytest.param("10.0.1", (10, 0, 1), id="two_digit_major_is_numeric_not_textual"),
        pytest.param("0.0.1", (0, 0, 1), id="zeros_are_versions_too"),
        pytest.param("2.13", None, id="two_parts_is_not_a_version"),
        pytest.param("2.13.0.1", None, id="four_parts_is_not_a_version"),
        pytest.param("v2.13.0", None, id="leading_v_refused"),
        pytest.param("2.13.0-rc1", None, id="prerelease_suffix_refused"),
        pytest.param("", None, id="empty_string_is_not_a_version"),
        pytest.param("garbage", None, id="free_text_refused"),
        pytest.param("٢.١٣.٠", None,
                     id="arabic_indic_digits_refused_because_the_regex_is_ascii"),
        pytest.param("2.13.0\n", None, id="trailing_newline_refused"),
        pytest.param("2.13.0 ", None, id="trailing_space_refused"),
        pytest.param(f"{'9' * 5000}.0.0", None, id="five_thousand_digit_component"),
        pytest.param(None, None, id="none_is_not_a_string"),
        pytest.param(213, None, id="int_is_not_a_string"),
        pytest.param((2, 13, 0), None, id="tuple_is_not_a_string"),
        pytest.param(b"2.13.0", None, id="bytes_is_not_a_string"),
    ],
)
def test_parse_client_version_accepts_only_plain_ascii_three_part_versions(raw, expected):
    """The version a client states is untrusted text off the wire. `\\d` would
    accept Arabic-Indic digits and `$` would accept a trailing newline, so both
    are deliberately avoided; the length cap keeps a huge component from ever
    reaching int(), which raises on very long digit strings."""
    assert parse_client_version(raw) == expected


def test_a_component_at_the_length_cap_still_parses():
    """The cap bounds the field, it does not reject ordinary versions: the
    longest string that fits still reads as three numbers."""
    raw = f"{'9' * (CLIENT_VERSION_MAX_LEN - 4)}.0.0"
    assert len(raw) == CLIENT_VERSION_MAX_LEN
    assert parse_client_version(raw) == (int("9" * (CLIENT_VERSION_MAX_LEN - 4)), 0, 0)
    assert parse_client_version(f"0{raw}") is None


@pytest.mark.parametrize(
    "client_version, minimum, outdated",
    [
        pytest.param("2.12.9", "2.13.0", True, id="older_patch_line_is_outdated"),
        pytest.param("2.9.0", "2.13.0", True,
                     id="compared_as_numbers_not_as_text_9_is_below_13"),
        pytest.param("0.0.1", "2.13.0", True, id="far_older_is_outdated"),
        pytest.param("2.13.0", "2.13.0", False, id="exactly_the_minimum_plays"),
        pytest.param("2.13.1", "2.13.0", False, id="newer_patch_plays"),
        pytest.param("3.0.0", "2.13.0", False, id="newer_major_plays"),
        pytest.param("", "2.13.0", False, id="a_source_run_states_nothing_and_plays"),
        pytest.param("garbage", "2.13.0", True,
                     id="a_version_nobody_can_read_is_not_trusted"),
        pytest.param("٢.١٣.٠", "2.13.0", True, id="unicode_digits_are_not_trusted"),
        pytest.param("2.13.0", "", False, id="an_empty_minimum_turns_nobody_away"),
        pytest.param("2.13.0", "garbage", False,
                     id="an_unreadable_minimum_turns_nobody_away"),
        pytest.param("2.13.0", "2.13", False,
                     id="a_two_part_minimum_turns_nobody_away"),
        pytest.param("0.0.1", "garbage", False,
                     id="a_broken_minimum_never_locks_even_an_ancient_build_out"),
    ],
)
def test_client_version_outdated_compares_versions_as_numbers(
    client_version, minimum, outdated,
):
    """The gate's whole decision. The empty client version is the source-run
    exemption and must stay False -- a checkout has no version.txt, and the
    protocol number is what turns an incompatible build away. A minimum the
    server itself cannot parse must never refuse anyone, or one typo in the
    constant closes the server to every player."""
    assert client_version_outdated(client_version, minimum) is outdated


def test_min_client_version_is_readable_and_not_ahead_of_this_build():
    """MIN_CLIENT_VERSION is a source constant baked into the server image, so a
    typo there is only caught here. A minimum ahead of the version this repo
    ships would refuse the very build released alongside it."""
    repo_root = Path(chessshootout.__file__).resolve().parent.parent
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.is_file(), f"expected the project file at {pyproject}"
    shipped = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    minimum = parse_client_version(MIN_CLIENT_VERSION)
    current = parse_client_version(shipped)
    assert minimum is not None, f"MIN_CLIENT_VERSION {MIN_CLIENT_VERSION!r} does not parse"
    assert current is not None, f"pyproject version {shipped!r} does not parse"
    assert minimum <= current, (
        f"MIN_CLIENT_VERSION {MIN_CLIENT_VERSION} is ahead of the shipped {shipped}")


def test_matchmake_client_version_is_bounded_and_optional():
    """The field is attacker-supplied and reaches a log line, so its length is
    capped at the model boundary rather than in the endpoint. It defaults to
    empty, which is exactly what a source run sends."""
    assert MatchmakeRequest(nickname="A", client_uuid=U1, time_minutes=5,
                            increment_seconds=0).client_version == ""
    at_cap = "9" * CLIENT_VERSION_MAX_LEN
    assert MatchmakeRequest(nickname="A", client_uuid=U1, time_minutes=5,
                            increment_seconds=0,
                            client_version=at_cap).client_version == at_cap
    with pytest.raises(ValidationError):
        MatchmakeRequest(nickname="A", client_uuid=U1, time_minutes=5,
                         increment_seconds=0, client_version="9" * (CLIENT_VERSION_MAX_LEN + 1))


def test_idle_window_table_is_the_single_source_of_policy():
    """Server and client both read this table (the client renders "Abort in" vs
    "Resign in" copy off it), so its exact shape is wire-adjacent: plies 0 and 1
    share the no-winner abort window, ply 2 is the somebody-loses resign window,
    and nothing else ever arms."""
    assert IDLE_WINDOW_BY_PLIES == {
        0: (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
        1: (Reason.ABORTED, FIRST_MOVE_ABORT_SECONDS),
        2: (Reason.RESIGNATION, IDLE_RESIGN_SECONDS),
    }


def test_idle_seconds_by_outcome_loses_no_ply_row_to_the_collapse():
    """IDLE_SECONDS_BY_OUTCOME is built by dropping the ply keys off
    IDLE_WINDOW_BY_PLIES and re-keying by outcome, which silently keeps only the
    LAST duration when two ply rows share an outcome. Today plies 0 and 1 agree on
    the abort window so nothing is lost -- but the client renders its countdown
    label and total off this collapsed table, so the moment a future edit gives one
    of them a different duration the client would draw a window the server is not
    keeping. This is the pin that makes such an edit loud instead of silent."""
    lossless = {}
    for plies, spec in IDLE_WINDOW_BY_PLIES.items():
        assert lossless.setdefault(spec.outcome, spec.seconds) == spec.seconds, \
            f"ply {plies} maps {spec.outcome} to a duration another ply row disagrees with"
    assert IDLE_SECONDS_BY_OUTCOME == lossless, \
        "the shipped table is exactly that lossless mapping, with every outcome present"
    assert set(IDLE_SECONDS_BY_OUTCOME) == {spec.outcome for spec in
                                            IDLE_WINDOW_BY_PLIES.values()}


def test_idle_window_message_round_trips():
    msg = IdleWindowMessage(outcome="resignation", color="white",
                            seconds_remaining=42.5)
    expected = {
        "version": PROTOCOL_VERSION, "type": "idle_window",
        "outcome": "resignation", "color": "white", "seconds_remaining": 42.5,
    }
    assert msg.model_dump() == expected
    assert IdleWindowMessage.model_validate(expected) == msg


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"seconds_remaining": -0.1}, id="negative_seconds"),
        pytest.param({"seconds_remaining": float("nan")}, id="nan_seconds"),
        pytest.param({"seconds_remaining": float("inf")}, id="inf_seconds"),
        pytest.param({"outcome": "timeout"}, id="unknown_outcome"),
        pytest.param({"outcome": "abandonment"}, id="grace_reason_is_not_a_window"),
        pytest.param({"color": "green"}, id="unknown_color"),
    ],
)
def test_idle_window_message_rejects_bad_fields(kwargs):
    """The countdown the client renders is hardened like every other wire float:
    NaN/Infinity would poison the deadline arithmetic and an unknown outcome has
    no label copy, so none of them can ever be minted."""
    base = dict(outcome="aborted", color="black", seconds_remaining=10.0)
    base.update(kwargs)
    with pytest.raises(ValidationError):
        IdleWindowMessage(**base)


def test_idle_window_message_rejects_non_finite_json_seconds():
    """pydantic-core parses bare NaN/Infinity JSON tokens into floats, so
    allow_inf_nan=False must also hold on the model_validate_json path."""
    raw = ('{"type": "idle_window", "outcome": "aborted", "color": "black", '
           '"seconds_remaining": NaN}')
    with pytest.raises(ValidationError):
        IdleWindowMessage.model_validate_json(raw)


def test_resume_response_idle_window_defaults_none_and_carries_the_wire():
    base = dict(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0,
                            running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0)
    assert ResumeResponse(**base).idle_window is None
    armed = ResumeResponse(**base, idle_window=IdleWindowWire(
        outcome="aborted", color="black", seconds_remaining=3.0))
    assert armed.model_dump()["idle_window"] == {
        "outcome": "aborted", "color": "black", "seconds_remaining": 3.0,
    }, "the client reads /resume via plain model_dump() — field-name keys"


def test_skill_check_required_round_trips():
    msg = SkillCheckRequiredMessage(
        kind="wheel", seed="abc", value_diff=8, deadline_ms=5000.0,
        from_sq="e4", to_sq="d5")
    dumped = msg.model_dump(by_alias=True)
    assert dumped["from"] == "e4" and dumped["to"] == "d5"
    assert dumped["promotion"] is None and dumped["miss_count"] == 0
    assert SkillCheckRequiredMessage.model_validate(dumped) == msg


def test_skill_check_required_carries_promotion():
    msg = SkillCheckRequiredMessage(
        kind="wheel", seed="s", value_diff=9, deadline_ms=5000.0,
        from_sq="e7", to_sq="e8", promotion="q")
    assert msg.model_dump(by_alias=True)["promotion"] == "q"


def test_skill_check_required_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        SkillCheckRequiredMessage(kind="duel", seed="x", value_diff=0, deadline_ms=5000.0,
                                  from_sq="e4", to_sq="d5")


def test_skill_check_shot_carries_the_client_rendered_elapsed():
    msg = SkillCheckShotMessage.model_validate(
        {"type": "skill_check_shot", "version": PROTOCOL_VERSION, "client_elapsed_ms": 412.0})
    assert msg.client_elapsed_ms == 412.0
    assert SkillCheckShotMessage().client_elapsed_ms == 0.0, "absent -> 0, clamped to a loss"


def test_skill_check_shot_ignores_injected_client_timestamp():
    msg = SkillCheckShotMessage.model_validate(
        {"type": "skill_check_shot", "version": PROTOCOL_VERSION, "client_fire_ms": 123})
    assert not hasattr(msg, "client_fire_ms")
    assert "client_fire_ms" not in msg.model_dump()


def test_skill_check_result_uses_from_to_aliases():
    msg = SkillCheckResultMessage.model_validate(
        {"type": "skill_check_result", "won": False, "from": "e4", "to": "d5"})
    assert (msg.won, msg.from_sq, msg.to_sq) == (False, "e4", "d5")
    assert msg.model_dump(by_alias=True)["from"] == "e4"


def test_skill_check_spectate_carries_the_full_challenge_with_from_to_aliases():
    msg = SkillCheckSpectateMessage(
        kind="aim", seed="abc", value_diff=3, deadline_ms=5000.0,
        **{"from": "d4", "to": "d5"}, promotion=None)
    assert (msg.kind, msg.seed, msg.value_diff) == ("aim", "abc", 3)
    assert (msg.from_sq, msg.to_sq) == ("d4", "d5")
    dumped = msg.model_dump(by_alias=True)
    assert dumped["from"] == "d4" and dumped["to"] == "d5"


def test_skill_check_spectate_shot_round_trips():
    msg = SkillCheckSpectateShotMessage.model_validate(
        {"type": "skill_check_spectate_shot", "elapsed_ms": 742.0,
         "miss_count": 2, "won": False})
    assert (msg.elapsed_ms, msg.miss_count, msg.won) == (742.0, 2, False)
    assert msg.model_dump()["type"] == "skill_check_spectate_shot"


def test_move_applied_skillcheck_fields_default_none_and_round_trip():
    base = dict(from_sq="e2", to_sq="e4", san="e4",
                clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="black"),
                ply=1)
    quiet = MoveAppliedMessage(**base)
    assert quiet.skill_check_kind is None and quiet.skill_check_won is None
    won = MoveAppliedMessage(**base, skill_check_kind="wheel", skill_check_won=True)
    assert MoveAppliedMessage.model_validate(won.model_dump(by_alias=True)) == won


def test_skillcheck_outcome_wire_uses_plain_field_keys_no_alias():
    wire = SkillCheckOutcomeWire(ply=7, kind="wheel", won=False, san="Rxe5")
    dumped = wire.model_dump()
    assert dumped == {"ply": 7, "kind": "wheel", "won": False, "san": "Rxe5"}
    assert wire.model_dump(by_alias=True) == dumped, "no from/to alias, no version key"


def test_skillcheck_outcome_wire_san_defaults_empty_for_wins():
    assert SkillCheckOutcomeWire(ply=1, kind="aim", won=True).san == ""


def test_resume_response_carries_the_skillcheck_log():
    resp = ResumeResponse(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0,
        skillcheck_log=[
            SkillCheckOutcomeWire(ply=1, kind="wheel", won=True),
            SkillCheckOutcomeWire(ply=2, kind="aim", won=False, san="Qxd5")])
    dumped = resp.model_dump()
    assert dumped["skillcheck_log"] == [
        {"ply": 1, "kind": "wheel", "won": True, "san": ""},
        {"ply": 2, "kind": "aim", "won": False, "san": "Qxd5"}]


def test_resume_response_skillcheck_log_defaults_empty():
    resp = ResumeResponse(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0)
    assert resp.skillcheck_log == []


def test_resume_response_pending_and_locks_default_empty():
    resp = ResumeResponse(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0)
    assert resp.pending_skillcheck is None
    assert resp.skillcheck_locks == []


def test_skillcheck_wire_messages_are_byte_identical_after_the_shared_base_refactor():
    """Pins the exact wire shape of the three skill-check messages that share a
    kind/seed/value_diff/deadline_ms/from/to/promotion geometry base, against
    literals captured from the pre-refactor (independently defined) models --
    so factoring out the shared base can never silently change what goes over
    the wire."""
    pending = PendingSkillCheckWire(
        kind="aim", seed="seed123", value_diff=5, deadline_ms=5000.0, elapsed_ms=1200.0,
        miss_count=2, from_sq="e4", to_sq="d5", promotion="q", color="white")
    required = SkillCheckRequiredMessage(
        kind="wheel", seed="seedreq", value_diff=-3, deadline_ms=4000.0, miss_count=1,
        from_sq="a7", to_sq="b8", promotion="n")
    spectate = SkillCheckSpectateMessage(
        kind="aim", seed="seedspec", value_diff=0, deadline_ms=3000.0,
        from_sq="c2", to_sq="c3", promotion=None)

    assert pending.model_dump(by_alias=True) == {
        "kind": "aim", "seed": "seed123", "value_diff": 5, "deadline_ms": 5000.0,
        "captured_value": 0, "elapsed_ms": 1200.0, "miss_count": 2, "progress": 0,
        "last_hit_pop": -1,
        "from": "e4", "to": "d5", "promotion": "q", "color": "white",
    }
    assert required.model_dump(by_alias=True) == {
        "version": PROTOCOL_VERSION, "type": "skill_check_required",
        "kind": "wheel", "seed": "seedreq", "value_diff": -3, "deadline_ms": 4000.0,
        "captured_value": 0, "miss_count": 1, "from": "a7", "to": "b8", "promotion": "n",
    }
    assert spectate.model_dump(by_alias=True) == {
        "version": PROTOCOL_VERSION, "type": "skill_check_spectate",
        "kind": "aim", "seed": "seedspec", "value_diff": 0, "deadline_ms": 3000.0,
        "captured_value": 0, "from": "c2", "to": "c3", "promotion": None,
    }


def test_resume_response_carries_pending_and_locks_when_set():
    pending = PendingSkillCheckWire(
        kind="aim", seed="s", value_diff=3, deadline_ms=5000.0, elapsed_ms=1800.0,
        miss_count=1, from_sq="e4", to_sq="d5", color="white")
    resp = ResumeResponse(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0,
        pending_skillcheck=pending,
        skillcheck_locks=[LockWire(from_sq="e4", to_sq="d5")])
    dumped = resp.model_dump()
    assert dumped["skillcheck_locks"][0] == {"from_sq": "e4", "to_sq": "d5"}, \
        "the client reads /resume via plain model_dump() — field-name keys, not aliases"
    assert dumped["pending_skillcheck"]["from_sq"] == "e4"
    assert dumped["pending_skillcheck"]["to_sq"] == "d5"
    assert dumped["pending_skillcheck"]["color"] == "white"


@pytest.mark.parametrize("kind", ["wheel", "aim", "whack", "combo"])
def test_skill_check_kind_literal_accepts_all_four_kinds(kind):
    msg = SkillCheckRequiredMessage(
        kind=kind, seed="s", value_diff=0, deadline_ms=5000.0,
        from_sq="e4", to_sq="d5")
    assert msg.kind == kind
    assert SkillCheckRequiredMessage.model_validate(msg.model_dump(by_alias=True)) == msg


def test_geometry_base_bounds_captured_value():
    """captured_value rides the shared geometry base: defaults 0, capped at the
    queen's value 9, and never negative — the whack/combo scaling input can't be
    forged outside real piece values."""
    base = dict(kind="whack", seed="s", value_diff=0, deadline_ms=5000.0,
                from_sq="e4", to_sq="d5")
    assert SkillCheckRequiredMessage(**base).captured_value == 0
    assert SkillCheckRequiredMessage(**base, captured_value=9).captured_value == 9
    for bad in (-1, 10):
        with pytest.raises(ValidationError):
            SkillCheckRequiredMessage(**base, captured_value=bad)


def test_skill_check_shot_wheel_shape_omits_the_whack_and_combo_fields():
    """A wheel/aim shot carries only client_elapsed_ms: direction and target_row/col
    are optional and default to None. This is NOT a v3 compatibility claim — v4 is a
    hard gate (the version field is checked at the handshake, no v3 client can
    connect) — it pins that the two positional kinds' fields stay opt-in so the
    wheel/aim payload never has to send them."""
    msg = SkillCheckShotMessage.model_validate(
        {"type": "skill_check_shot", "version": PROTOCOL_VERSION, "client_elapsed_ms": 412.0})
    assert msg.client_elapsed_ms == 412.0
    assert msg.direction is None
    assert msg.target_row is None and msg.target_col is None


@pytest.mark.parametrize(
    "token",
    [pytest.param("NaN", id="nan"), pytest.param("Infinity", id="inf"),
     pytest.param("-Infinity", id="neg_inf")],
)
def test_skill_check_shot_rejects_non_finite_client_elapsed(token):
    """pydantic-core parses bare NaN/Infinity JSON tokens into floats. A non-finite
    client_elapsed_ms would poison every comparison downstream (NaN loses the
    lag-comp clamp's min/max, the human floor, and the deadline test all at once),
    so allow_inf_nan=False must reject it at the wire; the handler's
    `except ValidationError -> noop` then drops the frame with no state change."""
    raw = ('{"type": "skill_check_shot", "client_elapsed_ms": ' + token + '}')
    with pytest.raises(ValidationError):
        SkillCheckShotMessage.model_validate_json(raw)


def test_skill_check_shot_carries_combo_direction_and_whack_target():
    combo = SkillCheckShotMessage(client_elapsed_ms=500.0, direction="left")
    assert combo.direction == "left"
    whack = SkillCheckShotMessage(client_elapsed_ms=900.0, target_row=3.5, target_col=0.0)
    assert (whack.target_row, whack.target_col) == (3.5, 0.0)


def test_skill_check_shot_rejects_unknown_direction():
    with pytest.raises(ValidationError):
        SkillCheckShotMessage(client_elapsed_ms=500.0, direction="diagonal")


@pytest.mark.parametrize("field", ["target_row", "target_col"])
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(8.0, id="at_upper_bound_exclusive"),
        pytest.param(8.5, id="past_upper_bound"),
        pytest.param(-0.1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="neg_inf"),
    ],
)
def test_skill_check_shot_rejects_out_of_board_target(field, bad):
    with pytest.raises(ValidationError):
        SkillCheckShotMessage(client_elapsed_ms=500.0, **{field: bad})


@pytest.mark.parametrize(
    "token",
    [pytest.param("NaN", id="nan"), pytest.param("Infinity", id="inf"),
     pytest.param("-Infinity", id="neg_inf")],
)
def test_skill_check_shot_rejects_non_finite_json_target(token):
    """pydantic-core parses bare NaN/Infinity JSON tokens, so the ge/lt bounds must
    reject them at the wire — a non-finite target can never reach adjudication."""
    raw = ('{"type": "skill_check_shot", "client_elapsed_ms": 500.0, '
           '"target_row": ' + token + ', "target_col": 1.0}')
    with pytest.raises(ValidationError):
        SkillCheckShotMessage.model_validate_json(raw)


def test_skill_check_spectate_shot_defaults_keep_the_v3_shape_parsing():
    msg = SkillCheckSpectateShotMessage.model_validate(
        {"type": "skill_check_spectate_shot", "elapsed_ms": 742.0,
         "miss_count": 2, "won": False})
    assert msg.progress == 0
    assert msg.direction is None
    assert msg.target_row is None and msg.target_col is None


def test_skill_check_spectate_shot_carries_progress_direction_and_target():
    msg = SkillCheckSpectateShotMessage(
        elapsed_ms=900.0, miss_count=1, won=False, progress=2,
        direction="up", target_row=2.5, target_col=4.5)
    dumped = msg.model_dump()
    assert dumped["progress"] == 2
    assert dumped["direction"] == "up"
    assert (dumped["target_row"], dumped["target_col"]) == (2.5, 4.5)


@pytest.mark.parametrize("field", ["target_row", "target_col"])
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(8.0, id="at_upper_bound_exclusive"),
        pytest.param(8.5, id="past_upper_bound"),
        pytest.param(-0.1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="neg_inf"),
    ],
)
def test_skill_check_spectate_shot_rejects_out_of_board_target(field, bad):
    """The spectate mirror carries the same board bounds as the mover's own
    SkillCheckShotMessage, which is where these coordinates come from -- the server
    copies an already-bounded shot onto the mirror, so an out-of-board value can
    only be a bug on our side. The client clamps defensively when it draws
    (_board_coord in screens/game.py), but the two halves of one wire pair must not
    disagree about what is representable: without the bounds a NaN or an off-board
    row could be minted here and never be minted on the inbound twin."""
    with pytest.raises(ValidationError):
        SkillCheckSpectateShotMessage(elapsed_ms=100.0, miss_count=0, won=False,
                                      **{field: bad})


@pytest.mark.parametrize(
    "token",
    [pytest.param("NaN", id="nan"), pytest.param("Infinity", id="inf"),
     pytest.param("-Infinity", id="neg_inf")],
)
def test_skill_check_spectate_shot_rejects_non_finite_json_target(token):
    """pydantic-core parses bare NaN/Infinity JSON tokens into floats, so the mirror
    has to reject them on the model_validate_json path too -- the same hardening the
    inbound shot already had."""
    raw = ('{"type": "skill_check_spectate_shot", "elapsed_ms": 100.0, '
           '"miss_count": 0, "won": false, "target_row": ' + token +
           ', "target_col": 1.0}')
    with pytest.raises(ValidationError):
        SkillCheckSpectateShotMessage.model_validate_json(raw)


@pytest.mark.parametrize("value", [0.0, 3.5, 7.999])
def test_skill_check_spectate_shot_accepts_every_on_board_target(value):
    """The bounds are the board itself: both edges of a real aim point stay legal,
    so hardening the mirror cannot drop a shot the mover was allowed to fire."""
    msg = SkillCheckSpectateShotMessage(elapsed_ms=100.0, miss_count=0, won=False,
                                        target_row=value, target_col=value)
    assert (msg.target_row, msg.target_col) == (value, value)


def test_spectate_shot_target_bounds_match_the_movers_own_shot_bounds():
    """Drift guard for the pair: whatever a mover is allowed to send as a target
    is exactly what the mirror is allowed to relay. Tightening one side alone would
    make the server unable to mirror a legal shot, or reopen the gap this closed."""
    inbound = SkillCheckShotMessage.model_fields
    mirror = SkillCheckSpectateShotMessage.model_fields
    for name in ("target_row", "target_col"):
        assert [str(m) for m in inbound[name].metadata] == \
            [str(m) for m in mirror[name].metadata], \
            f"{name} carries different constraints on the two halves of the pair"


def test_pending_skillcheck_wire_carries_progress_and_captured_value():
    wire = PendingSkillCheckWire(
        kind="whack", seed="s", value_diff=8, deadline_ms=5000.0, captured_value=1,
        elapsed_ms=1800.0, miss_count=1, progress=2, from_sq="e4", to_sq="d5",
        color="white")
    dumped = wire.model_dump()
    assert (dumped["progress"], dumped["captured_value"]) == (2, 1)
    assert PendingSkillCheckWire(
        kind="whack", seed="s", value_diff=8, deadline_ms=5000.0,
        elapsed_ms=0.0, from_sq="e4", to_sq="d5", color="white").progress == 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(PendingSkillCheckWire(
            kind="whack", seed="s", value_diff=8, deadline_ms=5000.0, captured_value=1,
            elapsed_ms=100.0, progress=1, from_sq="e4", to_sq="d5", color="white"),
            id="pending_wire"),
        pytest.param(SkillCheckRequiredMessage(
            kind="whack", seed="s", value_diff=8, deadline_ms=5000.0, captured_value=1,
            from_sq="e4", to_sq="d5"), id="required"),
        pytest.param(SkillCheckSpectateMessage(
            kind="whack", seed="s", value_diff=8, deadline_ms=5000.0, captured_value=1,
            from_sq="e4", to_sq="d5"), id="spectate"),
        pytest.param(SkillCheckSpectateShotMessage(
            elapsed_ms=900.0, miss_count=1, won=False, progress=1,
            target_row=2.5, target_col=4.5), id="spectate_shot"),
        pytest.param(SkillCheckShotMessage(client_elapsed_ms=100.0), id="shot"),
        pytest.param(SkillCheckResultMessage(won=False, from_sq="e4", to_sq="d5"),
                     id="result"),
    ],
)
def test_server_only_pending_fields_never_appear_on_any_skillcheck_wire_model(model):
    """last_input_ms (the anti-mash gate, paced on the server wall clock) and
    plies_ever (the room's abort counter) are server-side state — no wire model
    may ever dump them, or a crafted client could pace its bursts against the
    gate. last_hit_pop is deliberately NOT in this set: it is mover knowledge
    (which pop the mover already scored), and a reconnecting mover that has to
    guess it re-credits or forfeits a pop, so PendingSkillCheckWire carries it —
    pinned by test_pending_skillcheck_wire_carries_last_hit_pop_for_resume."""
    for dumped in (model.model_dump(), model.model_dump(by_alias=True)):
        assert "last_input_ms" not in dumped
        assert "plies_ever" not in dumped


def test_pending_skillcheck_wire_carries_last_hit_pop_for_resume():
    """The resume wire echoes the pop the mover has already been credited for, so
    a mid-whack reconnect neither re-credits it nor forfeits it. Default -1 = no
    pop scored yet."""
    wire = PendingSkillCheckWire(
        kind="whack", seed="s", value_diff=8, deadline_ms=5000.0, captured_value=1,
        elapsed_ms=1800.0, progress=1, last_hit_pop=2, from_sq="e4", to_sq="d5",
        color="white")
    assert wire.model_dump()["last_hit_pop"] == 2
    assert PendingSkillCheckWire(
        kind="whack", seed="s", value_diff=8, deadline_ms=5000.0,
        elapsed_ms=0.0, from_sq="e4", to_sq="d5", color="white").last_hit_pop == -1


@pytest.mark.parametrize(
    "model_cls, field, extra",
    [
        pytest.param(SkillCheckRequiredMessage, "deadline_ms", {}, id="required_deadline"),
        pytest.param(SkillCheckSpectateMessage, "deadline_ms", {}, id="spectate_deadline"),
        pytest.param(PendingSkillCheckWire, "deadline_ms",
                     {"elapsed_ms": 0.0, "color": "white"}, id="pending_deadline"),
        pytest.param(PendingSkillCheckWire, "elapsed_ms",
                     {"deadline_ms": 5000.0, "color": "white"}, id="pending_elapsed"),
    ],
)
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="neg_inf"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_outbound_skillcheck_millisecond_fields_reject_non_finite_and_negative(
        model_cls, field, extra, bad):
    """The outbound geometry the client rebuilds its challenge from is hardened
    exactly like the inbound client_elapsed_ms: a non-finite or negative
    deadline/elapsed would make every downstream comparison (deadline test,
    schedule refit, resume elapsed) meaningless, so it can never be minted."""
    base = dict(kind="whack", seed="s", value_diff=0, from_sq="e4", to_sq="d5")
    base.update(extra)
    base[field] = bad
    with pytest.raises(ValidationError):
        model_cls(**base)


ALL_SQUARES = [f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"]


def test_annotations_state_accepts_full_valid_payload_with_from_to_aliases():
    raw = {
        "version": PROTOCOL_VERSION, "type": "annotations_state", "sharing": True,
        "highlights": ["e4", "d5"],
        "arrows": [{"from": "e2", "to": "e4"}, {"from": "g1", "to": "f3"}],
    }
    msg = AnnotationsStateMessage.model_validate(raw)
    assert msg.sharing is True
    assert msg.highlights == ["e4", "d5"]
    assert (msg.arrows[0].from_sq, msg.arrows[0].to_sq) == ("e2", "e4")
    assert msg.model_dump(by_alias=True)["arrows"][1] == {"from": "g1", "to": "f3"}


def test_annotations_state_rejects_too_many_highlights():
    with pytest.raises(ValidationError):
        AnnotationsStateMessage(sharing=True, highlights=["a1"] * (MAX_SHARED_HIGHLIGHTS + 1))


def test_annotations_state_rejects_too_many_arrows():
    arrow = {"from": "e2", "to": "e4"}
    with pytest.raises(ValidationError):
        AnnotationsStateMessage(sharing=True, arrows=[arrow] * (MAX_SHARED_ARROWS + 1))


@pytest.mark.parametrize("bad", ["z9", "a0", "aa", "e9", "i1", ""])
def test_annotations_state_rejects_off_board_highlight(bad):
    with pytest.raises(ValidationError):
        AnnotationsStateMessage(sharing=True, highlights=[bad])


@pytest.mark.parametrize("bad", ["z9", "a0", "aa"])
def test_arrow_wire_rejects_off_board_coord(bad):
    with pytest.raises(ValidationError):
        ArrowWire(from_sq=bad, to_sq="e4")
    with pytest.raises(ValidationError):
        ArrowWire(from_sq="e4", to_sq=bad)


def test_annotation_delta_add_highlight_round_trips():
    msg = AnnotationDeltaMessage(action="add", kind="highlight", square="e4")
    assert (msg.action, msg.kind, msg.square) == ("add", "highlight", "e4")
    dumped = msg.model_dump(by_alias=True)
    assert dumped["type"] == "annotation_delta"
    assert dumped["square"] == "e4" and dumped["from"] is None


def test_annotation_delta_arrow_uses_from_to_aliases():
    raw = {"type": "annotation_delta", "action": "remove", "kind": "arrow",
           "from": "e2", "to": "e4"}
    msg = AnnotationDeltaMessage.model_validate(raw)
    assert (msg.from_sq, msg.to_sq) == ("e2", "e4")
    assert msg.model_dump(by_alias=True)["from"] == "e2"


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("action", "toggle", id="bad_action"),
        pytest.param("kind", "circle", id="bad_kind"),
    ],
)
def test_annotation_delta_enforces_literals(field, value):
    base = {"action": "add", "kind": "highlight", "square": "e4"}
    base[field] = value
    with pytest.raises(ValidationError):
        AnnotationDeltaMessage(**base)


def test_annotation_delta_rejects_off_board_square():
    with pytest.raises(ValidationError):
        AnnotationDeltaMessage(action="add", kind="highlight", square="z9")


@pytest.mark.parametrize("preset", list(range(CHAT_PRESET_COUNT)))
def test_quick_chat_accepts_valid_presets(preset):
    msg = QuickChatMessage(preset=preset)
    assert msg.preset == preset
    assert msg.type == "quick_chat"


@pytest.mark.parametrize("preset", [-1, CHAT_PRESET_COUNT])
def test_quick_chat_rejects_out_of_range_preset(preset):
    with pytest.raises(ValidationError):
        QuickChatMessage(preset=preset)


def test_quick_chat_received_carries_sender_and_preset():
    msg = QuickChatReceivedMessage(preset=2, sender="black")
    assert (msg.preset, msg.sender) == (2, "black")
    assert msg.model_dump()["type"] == "quick_chat_received"


def test_quick_chat_received_enforces_sender_literal():
    with pytest.raises(ValidationError):
        QuickChatReceivedMessage(preset=0, sender="green")


def test_resume_response_annotations_default_empty_both_colors():
    resp = ResumeResponse(
        fen="x", move_history=[],
        clock=ClockSnapshot(white_remaining=1.0, black_remaining=1.0, running_for="white"),
        your_color="white", white_name="A", black_name="B",
        time_minutes=5, increment_seconds=0)
    for side in ("white_annotations", "black_annotations"):
        ann = getattr(resp, side)
        assert isinstance(ann, AnnotationSetWire)
        assert ann.sharing is False and ann.highlights == [] and ann.arrows == []


def test_full_annotations_state_serializes_under_inbound_cap():
    highlights = ALL_SQUARES[:MAX_SHARED_HIGHLIGHTS]
    arrows = [ArrowWire(from_sq=ALL_SQUARES[i % 64], to_sq=ALL_SQUARES[(i + 1) % 64])
              for i in range(MAX_SHARED_ARROWS)]
    msg = AnnotationsStateMessage(sharing=True, highlights=highlights, arrows=arrows)
    payload = msg.model_dump_json(by_alias=True)
    assert len(payload.encode()) < MAX_INBOUND_MESSAGE_BYTES


def _public_string_constants(cls):
    """Every shared code a class publishes, read off the class rather than
    listed by hand so a newly added one cannot escape the guards below."""
    return {value for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, str)}


def test_health_status_is_a_closed_set_of_three_lowercase_codes():
    """The health field is a plain string on the wire so an unknown future
    status degrades gracefully in an old client, which makes the set of codes a
    contract rather than a type. Three, spelled the same way every other code in
    the protocol is."""
    statuses = _public_string_constants(HealthStatus)
    assert statuses == {"ok", "full", "degraded"}
    for value in statuses:
        assert value == value.lower() and value.isascii() and value.isidentifier()


def test_no_health_status_collides_with_a_reason_code():
    """Both vocabularies travel as bare strings and the client dispatches on
    them, so a code that meant `game ended` in one and `server state` in the
    other would be read by whichever branch tested first."""
    overlap = _public_string_constants(HealthStatus) & _public_string_constants(Reason)
    assert overlap == set(), f"health status and reason codes share {sorted(overlap)}"


def test_health_response_defaults_describe_a_healthy_just_started_server():
    """rooms_active is the only field a caller has to supply; everything else
    has to default to the honest reading for a server that has just come up, so
    a missing field can never publish a scarier state than the real one."""
    body = HealthResponse(rooms_active=0)
    assert body.status == HealthStatus.OK
    assert body.housekeeping_age_s == 0.0
    assert body.version == PROTOCOL_VERSION
    assert body.queue_depth == 0
    assert body.uptime_s == 0.0
    assert "max_rooms" not in body.model_dump(), "the room cap is not published"
