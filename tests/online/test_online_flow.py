"""End-to-end test: two clients pair against an in-process server, play
a short game over real WebSockets, and verify state symmetry.

The `server` fixture lives in tests/conftest.py (shared with
test_server_transport.py). Poll timeouts here are deliberately generous:
OnlineClient._matchmake_with_retries can spend ~4.5s retrying before it
gives up, so a tight poll would time out before a transient first attempt
recovers."""
import time

from chessshootout.online.client import OnlineClient
from tests.helpers import fake_uuid4
from tests.online.online_helpers import wait_for


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)
ALICE2 = fake_uuid4(11)
BOB2 = fake_uuid4(12)
ALICE3 = fake_uuid4(13)
BOB3 = fake_uuid4(14)


def _drain(client, timeout=15.0):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        events = client.drain_inbound()
        for ev in events:
            seen.append(ev)
            if ev.type == "game_start":
                return seen
        time.sleep(0.05)
    return seen


def test_two_clients_pair_and_play_a_move(server):
    addr = f"localhost:{server}"
    a = OnlineClient()
    a.connect(addr, {"nickname": "Alice", "client_uuid": ALICE,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "white"})
    b = OnlineClient()
    b.connect(addr, {"nickname": "Bob", "client_uuid": BOB,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "black"})

    a_game = wait_for(a, "game_start")
    b_game = wait_for(b, "game_start")
    assert a_game is not None and b_game is not None
    assert a_game.payload["your_color"] == "white"
    assert b_game.payload["your_color"] == "black"

    a.send_move("e2", "e4")
    a_applied = wait_for(a, "move_applied")
    b_applied = wait_for(b, "move_applied")
    assert a_applied.payload["san"] == "e4"
    assert b_applied.payload["san"] == "e4"

    b.send_move("e7", "e5")
    a_applied2 = wait_for(a, "move_applied")
    b_applied2 = wait_for(b, "move_applied")
    assert a_applied2.payload["san"] == "e5"
    assert b_applied2.payload["san"] == "e5"

    a.disconnect()
    b.disconnect()


def test_resign_broadcasts_result_to_both_clients(server):
    addr = f"localhost:{server}"
    a = OnlineClient()
    a.connect(addr, {"nickname": "Alice", "client_uuid": ALICE2,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "white"})
    b = OnlineClient()
    b.connect(addr, {"nickname": "Bob", "client_uuid": BOB2,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "black"})
    wait_for(a, "game_start")
    wait_for(b, "game_start")

    a.send_move("e2", "e4")
    wait_for(a, "move_applied")
    wait_for(b, "move_applied")
    a.send_resign()
    a_result = wait_for(a, "result")
    b_result = wait_for(b, "result")
    assert a_result.payload["reason"] == "resignation"
    assert a_result.payload["winner_color"] == "black"
    assert b_result.payload["reason"] == "resignation"
    a.disconnect()
    b.disconnect()


def test_online_rematch_swaps_colors(server):
    """On a mutual rematch the server swaps colours: White becomes Black and vice
    versa, and each client's fresh game_start carries the flipped your_color."""
    addr = f"localhost:{server}"
    a = OnlineClient()
    a.connect(addr, {"nickname": "Alice", "client_uuid": ALICE3,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "white"})
    b = OnlineClient()
    b.connect(addr, {"nickname": "Bob", "client_uuid": BOB3,
                      "time_minutes": 5, "increment_seconds": 0,
                      "side_preference": "black"})
    assert wait_for(a, "game_start").payload["your_color"] == "white"
    assert wait_for(b, "game_start").payload["your_color"] == "black"

    a.send_resign()
    wait_for(a, "result")
    wait_for(b, "result")

    a.send_rematch_request()
    b.send_rematch_request()
    a_game2 = wait_for(a, "game_start")
    b_game2 = wait_for(b, "game_start")
    assert a_game2 is not None and b_game2 is not None
    assert a_game2.payload["your_color"] == "black"
    assert b_game2.payload["your_color"] == "white"

    a.disconnect()
    b.disconnect()
