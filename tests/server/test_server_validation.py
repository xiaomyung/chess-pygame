"""M18a protocol-boundary validation: UUID4 rejection (model + route layers),
the per-uuid /reclaim sliding-window limit, and the expanded /healthz fields.

Two distinct layers are exercised on purpose: the Pydantic models reject
non-UUID4 ids with a ValidationError, and the FastAPI routes map that into an
HTTP 422 — keep both, they verify different seams.
"""
import json
import logging
from types import SimpleNamespace

import pydantic
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from chessshootout.server.app import (
    PROTOCOL_VERSION, RECLAIM_PER_UUID_LIMIT_PER_MINUTE, UuidRateLimiter,
    WS_CLOSE_INVALID_TOKEN, _parse_trusted_proxies, client_ip_key, create_app,
    log_trusted_proxies,
)
from chessshootout.server.protocol import (
    CancelMatchmakeRequest, HealthStatus, MIN_GRACE_SECONDS,
    MIN_HEARTBEAT_INTERVAL_SECONDS, MIN_HEARTBEAT_MISS_LIMIT, MatchmakeRequest,
    Reason, ReclaimRequest, ResumeRequest, _env_float, _env_int, _read_tuning,
    is_uuid4,
)
from tests.helpers import FakeClock, fake_uuid4


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)
ROOM = fake_uuid4(100)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("00000000-0000-4000-8000-000000000000", id="variant_8"),
        pytest.param("12345678-1234-4234-9234-123456789abc", id="variant_9"),
        pytest.param("f47ac10b-58cc-4372-a567-0e02b2c3d479", id="variant_a"),
        pytest.param("abcdef01-2345-4678-bcde-f01234567890", id="variant_b"),
        pytest.param(ALICE, id="fake_uuid4_seed_1"),
        pytest.param(ROOM, id="fake_uuid4_seed_100"),
    ],
)
def test_is_uuid4_accepts_canonical_v4_strings(value):
    assert is_uuid4(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param("alice", id="alpha_word"),
        pytest.param("aaaa", id="too_short"),
        pytest.param("not-a-uuid", id="hyphenated_word"),
        pytest.param("00000000-0000-0000-0000-000000000000", id="version_nibble_not_4"),
        pytest.param("00000000-0000-4000-0000-000000000000", id="variant_nibble_not_8_9_a_b"),
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(["uuid"], id="list"),
    ],
)
def test_is_uuid4_rejects_short_or_malformed_values(value):
    assert not is_uuid4(value)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: MatchmakeRequest(nickname="Alice", client_uuid="alice",
                                     time_minutes=5, increment_seconds=0),
            id="matchmake_client_uuid",
        ),
        pytest.param(lambda: ReclaimRequest(client_uuid="alice"), id="reclaim_client_uuid"),
        pytest.param(
            lambda: ResumeRequest(room_id="my-room", session_token="t"),
            id="resume_room_id",
        ),
        pytest.param(
            lambda: CancelMatchmakeRequest(room_id="my-room", session_token="t"),
            id="cancel_matchmake_room_id",
        ),
    ],
)
def test_request_model_rejects_non_uuid4(build):
    with pytest.raises(pydantic.ValidationError):
        build()


def test_resume_request_accepts_valid_uuid4_room_id():
    req = ResumeRequest(room_id=ROOM, session_token="t")
    assert req.room_id == ROOM


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def client(clock):
    return TestClient(create_app(now_provider=clock, max_rooms=8))


@pytest.mark.parametrize(
    "method, route, payload, field",
    [
        pytest.param(
            "POST", "/matchmake",
            {"version": PROTOCOL_VERSION, "client_uuid": "alice",
             "nickname": "Alice", "time_minutes": 5, "increment_seconds": 0},
            "client_uuid", id="matchmake_garbage_client_uuid",
        ),
        pytest.param(
            "POST", "/resume",
            {"version": PROTOCOL_VERSION, "room_id": "not-a-uuid", "session_token": "x"},
            "room_id", id="resume_garbage_room_id",
        ),
        pytest.param(
            "POST", "/reclaim",
            {"version": PROTOCOL_VERSION, "client_uuid": "u1"},
            "client_uuid", id="reclaim_garbage_client_uuid",
        ),
        pytest.param(
            "DELETE", "/matchmake",
            {"version": PROTOCOL_VERSION, "room_id": "blah", "session_token": "t"},
            "room_id", id="cancel_matchmake_garbage_room_id",
        ),
    ],
)
def test_route_rejects_non_uuid4_payload(client, caplog, method, route, payload, field):
    """All four body routes answer a refused body with the same closed envelope --
    one shared reason code, no field list, no pydantic prose. The list shape they
    used to return was FastAPI's own default: unstable across versions, and it put
    the failure's raw text (and, in `input`, the rejected value itself) into a
    reply anybody can trigger. Which field failed now goes to the operator's log
    instead, which is what the caplog half asserts."""
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        r = client.request(method, route, json=payload)
    assert r.status_code == 422
    assert r.json() == {"detail": {"reason": Reason.INVALID_FIELD}}
    rejected = [rec.getMessage() for rec in caplog.records
                if rec.getMessage().startswith("request rejected")]
    assert rejected == [f"request rejected path={route} field=body.{field} "
                        f"error=value_error"]


def test_the_published_422_matches_the_shape_the_routes_actually_send(client):
    """/openapi.json is the contract a stranger generates a client from, and it
    used to document FastAPI's HTTPValidationError list for a 422 the server never
    sends. Every body route now publishes the envelope it really answers with."""
    spec = client.get("/openapi.json").json()
    body_routes = [("/matchmake", "post"), ("/matchmake", "delete"),
                   ("/resume", "post"), ("/reclaim", "post")]
    for path, method in body_routes:
        schema = spec["paths"][path][method]["responses"]["422"]["content"]
        ref = schema["application/json"]["schema"]["$ref"]
        assert ref.endswith("/ReasonEnvelope"), f"{method.upper()} {path} publishes {ref}"
    envelope = spec["components"]["schemas"]["ReasonEnvelope"]
    assert envelope["properties"]["detail"]["$ref"].endswith("/ReasonDetail")
    assert spec["components"]["schemas"]["ReasonDetail"]["properties"]["reason"][
        "type"] == "string"


def test_a_route_raised_validation_error_is_a_server_error_not_a_422(clock):
    """The deleted `_validation_handler` caught pydantic's own ValidationError
    app-wide. It never fired for request bodies -- FastAPI raises
    RequestValidationError for those -- but it WOULD have dressed a server-side
    modelling bug up as the caller's fault, with a 422 and a leaked pydantic
    message. Uncaught, such a bug is what it actually is: a 500."""
    app = create_app(now_provider=clock, max_rooms=8)

    @app.get("/raises-a-model-error")
    async def raises_a_model_error():
        ResumeRequest(room_id="not-a-uuid", session_token="t")
        return {"unreachable": True}

    quiet = TestClient(app, raise_server_exceptions=False)
    r = quiet.get("/raises-a-model-error")
    assert r.status_code == 500
    assert Reason.INVALID_FIELD not in r.text


def test_ws_closes_with_invalid_token_on_garbage_room_id_path(client):
    """The WS endpoint validates the path param up-front and closes (code 4000)
    before accepting any frame, so the client never gets a successful auth."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/not-a-uuid") as ws:
            ws.send_text(json.dumps({"version": PROTOCOL_VERSION,
                                     "type": "auth", "session_token": "t"}))
            ws.receive_text()
    assert excinfo.value.code == WS_CLOSE_INVALID_TOKEN


def test_reclaim_per_uuid_rate_limited_after_burst(client):
    """The 5th call still resolves (404 NOT_IN_ROOM since the uuid isn't in a
    room); the 6th is short-circuited with 429 rate_limited regardless of room
    state."""
    for _ in range(RECLAIM_PER_UUID_LIMIT_PER_MINUTE):
        r = client.post("/reclaim", json={
            "version": PROTOCOL_VERSION, "client_uuid": ALICE,
        })
        assert r.status_code == 404, r.text
    r = client.post("/reclaim", json={
        "version": PROTOCOL_VERSION, "client_uuid": ALICE,
    })
    assert r.status_code == 429
    assert r.json().get("detail", {}).get("reason") == Reason.RATE_LIMITED


def test_reclaim_limit_is_per_uuid_independent(client):
    """Bursting Alice past the cap leaves Bob's first call unthrottled."""
    for _ in range(RECLAIM_PER_UUID_LIMIT_PER_MINUTE):
        client.post("/reclaim", json={
            "version": PROTOCOL_VERSION, "client_uuid": ALICE,
        })
    r = client.post("/reclaim", json={
        "version": PROTOCOL_VERSION, "client_uuid": BOB,
    })
    assert r.status_code == 404


def test_reclaim_window_slides_releases_capacity(clock):
    """UuidRateLimiter is a sliding 60s window — driven directly with the fake
    clock to verify capacity is released without relying on real time."""
    limiter = UuidRateLimiter(limit_per_minute=5, window_seconds=60.0,
                              now_provider=clock)
    for _ in range(5):
        assert limiter.hit("u")
    assert not limiter.hit("u")
    clock.advance(61)
    assert limiter.hit("u")


def test_uuid_rate_limiter_prunes_stale_buckets(clock):
    """Distinct uuids leave per-key buckets; once their hits age out, pruning
    evicts the empty buckets so a flood of one-off uuids can't grow memory
    unboundedly."""
    limiter = UuidRateLimiter(limit_per_minute=5, window_seconds=60.0,
                              now_provider=clock)
    for i in range(50):
        limiter.hit(f"uuid-{i}")
    assert len(limiter._calls) == 50
    clock.advance(61)
    limiter._prune(clock() - limiter.window)
    assert len(limiter._calls) == 0


def test_healthz_includes_version_and_status_fields(client):
    body = client.get("/healthz").json()
    assert body["version"] == PROTOCOL_VERSION
    assert body["status"] == HealthStatus.OK


def test_healthz_includes_queue_depth_and_uptime(clock, client):
    body = client.get("/healthz").json()
    assert body["queue_depth"] == 0
    assert body["uptime_s"] == pytest.approx(0.0, abs=1e-6)
    assert body["housekeeping_age_s"] == pytest.approx(0.0, abs=1e-6)
    clock.advance(7.5)
    body = client.get("/healthz").json()
    assert body["uptime_s"] == pytest.approx(7.5, abs=1e-3)


TRUSTED = _parse_trusted_proxies("127.0.0.1/32,10.0.0.0/8")


def _request(peer, **headers):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


@pytest.mark.parametrize(
    "header, expected",
    [
        pytest.param("203.0.113.7", "203.0.113.7", id="valid_header_is_still_used"),
        pytest.param("not-an-ip", "10.1.2.3", id="garbage_header_falls_back_to_peer"),
        pytest.param("203.0.113.7, 198.51.100.9", "10.1.2.3",
                     id="address_list_is_not_one_ip_so_falls_back"),
        pytest.param("2001:DB8::0:1", "2001:db8::1",
                     id="ipv6_header_is_canonicalised_to_one_bucket"),
    ],
)
def test_client_ip_key_only_trusts_a_parseable_forwarded_address(header, expected):
    """SECURITY: cf-connecting-ip was returned verbatim, so whatever arrived in
    that header became a rate-limit bucket key -- an unbounded string space, one
    fresh bucket per garbage value. It is now honoured only when it parses as a
    single IP, and the parsed form is what keys the bucket, so two spellings of
    one address cannot split into two allowances.

    (Peer-trust itself -- spoofed header from an untrusted peer, trimming, no
    header at all -- is pinned in test_rate_limit_client_ip.py; this covers only
    the parse gate on top of it.)"""
    assert client_ip_key(_request("10.1.2.3", **{"cf-connecting-ip": header}),
                         trusted=TRUSTED) == expected


def test_client_ip_key_with_no_trusted_proxies_always_uses_the_peer():
    """The behavioural half of the warning below: when TRUSTED_PROXIES parses
    empty the header is ignored outright rather than trusted by accident."""
    assert client_ip_key(_request("10.1.2.3", **{"cf-connecting-ip": "203.0.113.7"}),
                         trusted=[]) == "10.1.2.3"


def test_log_trusted_proxies_warns_when_a_configured_value_parses_empty(caplog):
    """SECURITY (silent failure): a typo'd or drifted TRUSTED_PROXIES parsed to
    [] with no signal at all, which silently flips client_ip_key to the socket
    peer -- behind Cloudflare that is one shared bucket for every player on the
    server, so the per-IP limits stop separating anyone. Config that was set but
    could not be understood has to be loud."""
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        log_trusted_proxies(raw="not-a-cidr", trusted=[])
    records = [r for r in caplog.records if r.name == "chess.server.app"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "not-a-cidr" in records[0].getMessage()


@pytest.mark.parametrize(
    "raw, trusted, expected_fragment",
    [
        pytest.param("10.0.0.0/8", _parse_trusted_proxies("10.0.0.0/8"), "10.0.0.0/8",
                     id="parsed_set_is_reported"),
        pytest.param("", [], "none", id="deliberately_unset_is_not_a_warning"),
    ],
)
def test_log_trusted_proxies_reports_the_effective_set_at_info(
        caplog, raw, trusted, expected_fragment):
    """The set is echoed once at startup so a live server can be checked against
    what the operator meant to deploy. An empty env var is a deliberate 'trust
    nobody', not drift, so it stays INFO."""
    with caplog.at_level(logging.INFO, logger="chess.server.app"):
        log_trusted_proxies(raw=raw, trusted=trusted)
    records = [r for r in caplog.records if r.name == "chess.server.app"]
    assert [r.levelno for r in records] == [logging.INFO]
    assert expected_fragment in records[0].getMessage()


def test_healthz_queue_depth_reflects_pending_room(client):
    """One unpaired matchmake bumps queue_depth to 1; the peer pairs it into a
    room, draining the queue and incrementing rooms_active."""
    r = client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": ALICE,
        "nickname": "Alice", "time_minutes": 5, "increment_seconds": 0,
    })
    assert r.status_code == 200
    body = client.get("/healthz").json()
    assert body["queue_depth"] == 1
    assert body["rooms_active"] == 0
    client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": BOB,
        "nickname": "Bob", "time_minutes": 5, "increment_seconds": 0,
    })
    body = client.get("/healthz").json()
    assert body["queue_depth"] == 0
    assert body["rooms_active"] == 1


TUNING_PROBE = "CHESS_TUNING_PROBE"


@pytest.mark.parametrize(
    "reader, default, minimum",
    [
        pytest.param(_env_float, 60.0, MIN_GRACE_SECONDS, id="float"),
        pytest.param(_env_int, 3, MIN_HEARTBEAT_MISS_LIMIT, id="int"),
    ],
)
def test_a_missing_tuning_variable_is_the_silent_compiled_in_default(
        monkeypatch, caplog, reader, default, minimum):
    """Not setting a knob is the normal case -- every deployment leaves most of
    them alone -- so it must not cost a log line."""
    monkeypatch.delenv(TUNING_PROBE, raising=False)
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        assert reader(TUNING_PROBE, default, minimum=minimum) == default
    assert caplog.records == []


@pytest.mark.parametrize(
    "reader, default, minimum",
    [
        pytest.param(_env_float, 60.0, MIN_GRACE_SECONDS, id="float"),
        pytest.param(_env_int, 3, MIN_HEARTBEAT_MISS_LIMIT, id="int"),
    ],
)
def test_an_unparsable_tuning_value_falls_back_and_says_so(
        monkeypatch, caplog, reader, default, minimum):
    """These used to fall back in total silence, so `GRACE_SECONDS=60s` ran a
    server on the default forever with nothing to show for it. The variable is
    named; the value never is -- an operator-supplied string in a log line is a
    forged-record vector."""
    monkeypatch.setenv(TUNING_PROBE, "sixty seconds\nmatchmake ok room=forged")
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        assert reader(TUNING_PROBE, default, minimum=minimum) == default
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert f"name={TUNING_PROBE}" in messages[0]
    assert f"default={default}" in messages[0]
    assert "sixty seconds" not in messages[0]
    assert "forged" not in messages[0]


@pytest.mark.parametrize(
    "reader, raw, default, minimum",
    [
        pytest.param(_env_float, "0", 60.0, MIN_GRACE_SECONDS, id="float"),
        pytest.param(_env_int, "0", 3, MIN_HEARTBEAT_MISS_LIMIT, id="int"),
        pytest.param(_env_float, "nan", 60.0, MIN_GRACE_SECONDS, id="float_nan"),
        pytest.param(_env_float, "inf", 60.0, MIN_GRACE_SECONDS, id="float_infinity"),
        pytest.param(_env_float, "-inf", 60.0, MIN_GRACE_SECONDS,
                     id="float_negative_infinity"),
    ],
)
def test_a_tuning_value_below_its_floor_is_clamped_and_says_so(
        monkeypatch, caplog, reader, raw, default, minimum):
    """SECURITY-adjacent misconfiguration: a zero heartbeat interval produced a
    zero heartbeat timeout, which disconnects every player the moment they
    connect. A number that would break the server is replaced by the floor
    rather than obeyed.

    nan and the infinities are the ones a bare `value < minimum` misses:
    float() accepts all three spellings, nan compares false against every
    bound, and +inf sails over the floor into a grace period no disconnect ever
    ends. They are not workable settings, so they take the floor as well."""
    monkeypatch.setenv(TUNING_PROBE, raw)
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        assert reader(TUNING_PROBE, default, minimum=minimum) == minimum
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith("env clamped")
    assert f"name={TUNING_PROBE}" in messages[0]
    assert f"minimum={minimum}" in messages[0]


def test_read_tuning_clamps_every_knob_at_its_own_floor(monkeypatch, caplog):
    """The three real variable names together, so the floors are pinned where an
    operator actually sets them. Reading them through one function is what makes
    this testable without reimporting the module."""
    for name in ("GRACE_SECONDS", "HEARTBEAT_INTERVAL_SECONDS", "HEARTBEAT_MISS_LIMIT"):
        monkeypatch.setenv(name, "0")
    with caplog.at_level(logging.WARNING, logger="chess.server.app"):
        grace, interval, miss_limit = _read_tuning()
    assert (grace, interval, miss_limit) == (
        MIN_GRACE_SECONDS, MIN_HEARTBEAT_INTERVAL_SECONDS, MIN_HEARTBEAT_MISS_LIMIT)
    assert interval * miss_limit > 0, "a zero heartbeat timeout is unreachable"
    assert len(caplog.records) == 3, "one warning per clamped knob"


def test_read_tuning_takes_an_operators_values_when_they_are_workable(monkeypatch):
    """The point of the knobs: sane overrides still get through untouched."""
    monkeypatch.setenv("GRACE_SECONDS", "45")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "1.5")
    monkeypatch.setenv("HEARTBEAT_MISS_LIMIT", "5")
    assert _read_tuning() == (45.0, 1.5, 5)
