"""The version gate over a real socket, end to end.

REGRESSION. Before the gate, /matchmake accepted a body stamped with any
protocol version at all: the player was queued, the websocket handshake then
refused the same build, and the search card sat there spinning for ever with
nothing to click. Every layer of the fix has its own fast test -- the 426 in
tests/server, the exception mapping in test_server_transport.py, the event in
test_online_client.py, the card in test_online_ux.py -- and each of those fakes
whatever is below it. This file is the one that fakes nothing: real uvicorn,
real HTTP, the real OnlineClient thread and the real coordinator, so an answer
that the stack decodes differently anywhere along the way is caught here.

Exactly ONE test, never parametrized: a second case would pay a whole server
start-up for a claim the layer tests already make.
"""
import time

import pygame as pg

from tests.conftest import pygame_display
from chessshootout.online.client import OnlineClient
from chessshootout.frontend.online_coordinator import UPDATE_REQUIRED_TITLE
from chessshootout.server.protocol import PROTOCOL_VERSION
from tests.helpers import fake_uuid4, make_app


_pygame_init = pygame_display(900, 600)

ALICE = fake_uuid4(1)
SPIN_TIMEOUT_SECONDS = 15.0


def test_a_build_on_the_previous_protocol_ends_the_search_instead_of_spinning(
    server_with_app,
):
    port, _app = server_with_app
    frontend = make_app(900, 600)
    coordinator = frontend.coordinator
    coordinator.client = OnlineClient()
    coordinator.wait_modal.show("Blitz", "5 + 0", coordinator._on_online_cancel)
    coordinator._wait_started_at_ms = pg.time.get_ticks()

    coordinator.client.connect(f"localhost:{port}", {
        "version": PROTOCOL_VERSION - 1,
        "nickname": "Alice", "client_uuid": ALICE,
        "time_minutes": 5, "increment_seconds": 0, "side_preference": "random",
    })

    deadline = time.time() + SPIN_TIMEOUT_SECONDS
    while time.time() < deadline and not frontend.confirm_modal.is_visible():
        coordinator.update(pg.time.get_ticks())
        time.sleep(0.02)

    assert frontend.confirm_modal.is_visible(), \
        "the refusal never reached the player -- this is the eternal spinner"
    assert frontend.confirm_modal.title == UPDATE_REQUIRED_TITLE
    assert not coordinator.wait_modal.is_visible(), \
        "the search card has to come down with it"
    assert coordinator.client.state == "disconnected"
