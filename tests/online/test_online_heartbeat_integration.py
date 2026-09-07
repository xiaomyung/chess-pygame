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

import pytest

from chessshootout.online.client import OnlineClient
from chessshootout.server.protocol import (
    RESYNC_STABLE_MISMATCH_HEARTBEATS, RESYNC_TRANSIT_GRACE_SECONDS,
)
from tests.helpers import fake_uuid4
from tests.online.online_helpers import collect_for, wait_for


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)

QUIET_WINDOW_SECONDS = 0.5


def test_a_heartbeat_racing_a_real_move_is_tolerated_then_judged(server_with_app):
    """The in-flight excuse over a real socket: black's heartbeat reports the ply
    before white's move because the broadcast has not landed yet, and no
    directive follows.

    The claim only means anything while the whole exchange fit inside the transit
    grace, so a host that stalled past it skips rather than failing on its own
    slowness."""
    port, app = server_with_app
    addr = f"localhost:{port}"
    a, b = OnlineClient(), OnlineClient()
    a.connect(addr, {"nickname": "Alice", "client_uuid": ALICE, "time_minutes": 5,
                     "increment_seconds": 0, "side_preference": "white"})
    b.connect(addr, {"nickname": "Bob", "client_uuid": BOB, "time_minutes": 5,
                     "increment_seconds": 0, "side_preference": "black"})
    assert wait_for(a, "game_start") is not None
    assert wait_for(b, "game_start") is not None
    time.sleep(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)

    t0 = time.time()
    a.send_move("e2", "e4")
    assert wait_for(a, "move_applied") is not None

    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
        b.send_ping(0)
    racing = collect_for(b, QUIET_WINDOW_SECONDS)
    if time.time() - t0 >= RESYNC_TRANSIT_GRACE_SECONDS:
        pytest.skip("host stalled past the transit grace")
    assert [ev for ev in racing if ev.type == "resync_directive"] == [], \
        "a real round trip has to fit inside the transit grace"

    time.sleep(RESYNC_TRANSIT_GRACE_SECONDS + 0.1)
    for _ in range(RESYNC_STABLE_MISMATCH_HEARTBEATS):
        b.send_ping(0)

    directive = wait_for(b, "resync_directive", timeout=5.0)
    assert directive is not None, "a client still behind long afterwards is repaired"
    assert directive.payload["server_ply"] == 1

    a.disconnect()
    b.disconnect()
