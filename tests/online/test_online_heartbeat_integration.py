"""The heartbeat tolerance over a real socket, on a real clock.

Everything else about the ping window is driven through the handlers with a
fake clock, which is the only way to pin the branches exactly. That leaves one
claim untested: that RESYNC_TRANSIT_GRACE_SECONDS is actually long enough on a
real connection. A grace that looked fine against an instantly-advancing fake
clock but was shorter than one real round trip would put the "Resyncing..."
toast straight back.

So this file holds exactly ONE test, and never parametrized: two real
OnlineClients over real HTTP and WebSockets, a heartbeat sent the instant the
opponent's move lands, and then the same heartbeat once the window has really
elapsed. A parametrized version would pay the full pairing handshake per case
for no extra claim.
"""
import time

from chessshootout.online.client import OnlineClient
from chessshootout.server.protocol import (
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS,
)
from tests.helpers import fake_uuid4


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)

QUIET_WINDOW_SECONDS = 0.5


def _wait_for(client, type_name, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ev in client.drain_inbound():
            if ev.type == type_name:
                return ev
        time.sleep(0.02)
    return None


def _collect(client, seconds):
    """Everything that arrives over a real wall-clock window, so an assertion
    that NOTHING arrived is about elapsed time rather than about polling luck."""
    deadline = time.time() + seconds
    seen = []
    while time.time() < deadline:
        seen.extend(client.drain_inbound())
        time.sleep(0.02)
    return seen


def test_a_heartbeat_racing_a_real_move_is_tolerated_then_judged(server_with_app):
    port, app = server_with_app
    addr = f"localhost:{port}"
    a, b = OnlineClient(), OnlineClient()
    a.connect(addr, {"nickname": "Alice", "client_uuid": ALICE, "time_minutes": 5,
                     "increment_seconds": 0, "side_preference": "white"})
    b.connect(addr, {"nickname": "Bob", "client_uuid": BOB, "time_minutes": 5,
                     "increment_seconds": 0, "side_preference": "black"})
    assert _wait_for(a, "game_start") is not None
    assert _wait_for(b, "game_start") is not None
    time.sleep(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)

    a.send_move("e2", "e4")
    assert _wait_for(a, "move_applied") is not None

    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
        b.send_ping(0)
    racing = _collect(b, QUIET_WINDOW_SECONDS)
    assert [ev for ev in racing if ev.type == "resync_directive"] == [], \
        "a real round trip has to fit inside the transit grace"

    time.sleep(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
        b.send_ping(0)

    directive = _wait_for(b, "resync_directive", timeout=5.0)
    assert directive is not None, "a client still behind long afterwards is repaired"
    assert directive.payload["server_ply"] == 1

    a.disconnect()
    b.disconnect()
