import ipaddress

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from chessshootout.server import routes_http
from chessshootout.server.app import create_app
from chessshootout.server.limits import (
    RECLAIM_PER_IP_LIMIT, RECLAIM_PER_UUID_LIMIT_PER_MINUTE,
    _parse_trusted_proxies, _peer_trusted, client_ip_key,
)
from chessshootout.server.protocol import PROTOCOL_VERSION, Reason
from tests.helpers import FakeClock, fake_uuid4

TRUSTED = [ipaddress.ip_network("172.28.0.0/16")]
TINY_PER_IP_LIMIT = "2/minute"
TINY_BUDGET = 2


def make_request(peer_ip, headers=None):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {"type": "http", "headers": raw}
    scope["client"] = (peer_ip, 12345) if peer_ip is not None else None
    return Request(scope)


def test_parse_trusted_proxies_parses_cidrs_and_skips_blanks_and_invalid():
    nets = _parse_trusted_proxies("172.28.0.0/16, , 10.0.0.1, not-an-ip, ::1/128")
    assert [str(n) for n in nets] == ["172.28.0.0/16", "10.0.0.1/32", "::1/128"]


def test_parse_trusted_proxies_empty_string_yields_no_networks():
    assert _parse_trusted_proxies("") == []


@pytest.mark.parametrize("peer,expected", [
    ("172.28.0.5", True),
    ("172.29.0.5", False),
    ("8.8.8.8", False),
    ("testclient", False),
])
def test_peer_trusted_matches_only_configured_networks(peer, expected):
    assert _peer_trusted(peer, TRUSTED) is expected


def test_client_ip_key_ignores_spoofed_cf_header_from_untrusted_peer():
    req = make_request("8.8.8.8", {"cf-connecting-ip": "1.2.3.4"})
    assert client_ip_key(req, TRUSTED) == "8.8.8.8"


def test_client_ip_key_honors_cf_header_from_trusted_peer():
    req = make_request("172.28.0.5", {"cf-connecting-ip": "1.2.3.4"})
    assert client_ip_key(req, TRUSTED) == "1.2.3.4"


def test_client_ip_key_falls_back_to_socket_ip_for_trusted_peer_without_cf_header():
    req = make_request("172.28.0.5")
    assert client_ip_key(req, TRUSTED) == "172.28.0.5"


def test_client_ip_key_strips_whitespace_from_cf_header():
    req = make_request("172.28.0.5", {"cf-connecting-ip": "  1.2.3.4  "})
    assert client_ip_key(req, TRUSTED) == "1.2.3.4"


def test_client_ip_key_uses_loopback_when_request_has_no_client():
    req = make_request(None, {"cf-connecting-ip": "1.2.3.4"})
    assert client_ip_key(req, TRUSTED) == "127.0.0.1"


def test_client_ip_key_defaults_to_module_trusted_proxies(monkeypatch):
    import chessshootout.server.limits as limits_module
    monkeypatch.setattr(
        limits_module, "TRUSTED_PROXIES", [ipaddress.ip_network("10.1.0.0/16")],
    )
    trusted_req = make_request("10.1.2.3", {"cf-connecting-ip": "9.9.9.9"})
    untrusted_req = make_request("8.8.8.8", {"cf-connecting-ip": "9.9.9.9"})
    assert client_ip_key(trusted_req) == "9.9.9.9"
    assert client_ip_key(untrusted_req) == "8.8.8.8"


def test_client_ip_key_trusts_cf_header_from_loopback_under_default_config():
    trusted = _parse_trusted_proxies("127.0.0.1/32")
    req = make_request("127.0.0.1", {"cf-connecting-ip": "1.2.3.4"})
    assert client_ip_key(req, trusted) == "1.2.3.4"


def _limit_count(limit):
    return int(limit.split("/")[0])


def _client_on_a_tiny_budget(monkeypatch, *names):
    """An app whose named per-IP limits are TINY_PER_IP_LIMIT instead of the
    shipped ones. The strings are read when build_http_router runs, i.e. once per
    create_app, so patching them here and building the app afterwards proves the
    decorator binding at two requests instead of sixty."""
    for name in names:
        monkeypatch.setattr(routes_http, name, TINY_PER_IP_LIMIT)
    return TestClient(create_app(now_provider=FakeClock(), max_rooms=8))


def test_resume_is_rate_limited_per_ip(monkeypatch):
    """The @limiter.limit decorator on POST /resume had no coverage at all: the
    key function above was tested, the endpoint binding was not. A resume that is
    refused because the room does not exist still SPENDS allowance -- the limiter
    runs in front of the handler -- which is what makes the endpoint a cheap
    unauthenticated way to hammer the room lookup if the decorator ever came off.

    The 404s are the point: every call before the refusal reached the handler."""
    client = _client_on_a_tiny_budget(monkeypatch, "RESUME_PER_IP_LIMIT")
    payload = {"version": PROTOCOL_VERSION, "room_id": fake_uuid4(4242),
               "session_token": "nope"}
    for _ in range(TINY_BUDGET):
        assert client.post("/resume", json=payload).status_code == 404
    limited = client.post("/resume", json=payload)
    assert limited.status_code == 429
    assert limited.json()["detail"]["reason"] == Reason.RATE_LIMITED


def test_reclaim_is_rate_limited_per_ip_beyond_the_per_uuid_limiter(monkeypatch):
    """/reclaim carries TWO limiters -- a per-uuid one inside the handler and the
    per-IP decorator around it -- and only the per-uuid one was covered. Every
    call here uses a DISTINCT uuid so the inner limiter (100/minute per id) can
    never be the thing that answers 429; the refusal can only have come from the
    per-IP decorator.

    That ordering matters: the per-uuid limiter is what a single stolen id costs,
    the per-IP one is what one host costs however many ids it invents -- which is
    why the shipped per-IP cap is the higher of the two, pinned here."""
    assert _limit_count(RECLAIM_PER_IP_LIMIT) > RECLAIM_PER_UUID_LIMIT_PER_MINUTE, \
        "the per-IP cap is the outer one and must sit above the per-uuid cap"
    client = _client_on_a_tiny_budget(monkeypatch, "RECLAIM_PER_IP_LIMIT")
    for i in range(TINY_BUDGET):
        r = client.post("/reclaim", json={"version": PROTOCOL_VERSION,
                                          "client_uuid": fake_uuid4(5000 + i)})
        assert r.status_code == 404, f"call {i} should reach the handler"
    limited = client.post("/reclaim", json={"version": PROTOCOL_VERSION,
                                            "client_uuid": fake_uuid4(5000 + TINY_BUDGET)})
    assert limited.status_code == 429
    assert limited.json()["detail"]["reason"] == Reason.RATE_LIMITED
