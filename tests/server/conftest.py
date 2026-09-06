from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chessshootout.server.app import PROTOCOL_VERSION, create_app
from tests.helpers import FakeClock, fake_uuid4


ALICE = fake_uuid4(1)
BOB = fake_uuid4(2)
APP_KEY: pytest.StashKey[FastAPI] = pytest.StashKey()


@pytest.fixture
def clock() -> FakeClock:
    """
    Give each server test its own monotonic clock stand-in, so grace periods,
    skill-check deadlines and idle windows are stepped instantly instead of
    waited out

    :returns: a fresh clock sitting at zero seconds
    """
    return FakeClock()


@pytest.fixture
def app(clock: FakeClock, request: pytest.FixtureRequest) -> FastAPI:
    """
    Build a server application wired to the test clock, with a small room cap so
    the server-full path is reachable without creating a hundred rooms. The
    application is left on the test item as well, so the housekeeping check can
    still reach it after this fixture has been torn down

    :param clock: fake monotonic clock the app reads every timestamp from
    :param request: the running test, whose item carries the application on
    :returns: the application under test
    """
    application = create_app(now_provider=clock, max_rooms=8)
    request.node.stash[APP_KEY] = application
    return application


@pytest.fixture
def allow_sweep_failures() -> bool:
    """
    Opt one test out of the clean-housekeeping teardown check, for the handful
    that break a step on purpose to prove a failure stays contained. Requesting
    it is the whole opt-out; the flag itself is only read as present

    :returns: True, so a test can also assert it asked for the opt-out
    """
    return True


@pytest.fixture(autouse=True)
def clean_sweep(request: pytest.FixtureRequest) -> Iterator[None]:
    """
    Fail any server test that left a swallowed housekeeping failure behind. The
    sweep now contains a failing step instead of dying, which would otherwise
    turn a broken step into a silently passing test in the ~30 tests that drive
    those steps directly. Tests without an app never build one to be checked

    :param request: the running test, read for the fixtures it asked for
    :returns: a context that runs the check after the test body
    """
    yield
    if "allow_sweep_failures" in request.fixturenames:
        return
    app = request.node.stash.get(APP_KEY, None)
    if app is None:
        return
    sweep = app.state.sweep
    if sweep.failure_count:
        raise AssertionError(
            f"housekeeping swallowed {sweep.failure_count} failure(s)"
        ) from sweep._last_failure


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """
    Drive the app in-process, which exercises the real HTTP endpoints and the
    real websocket route without binding a port or starting a server

    :param app: the application under test
    :returns: a test client bound to that application
    """
    return TestClient(app)


def auth_msg(token: str) -> dict[str, Any]:
    """
    Build the handshake frame a client must send first on a game websocket: the
    protocol version the server insists on, plus the session token naming the
    slot being claimed. No game traffic is accepted before it

    :param token: session token handed out by matchmake, resume or reclaim
    :returns: the auth frame, ready to send as JSON
    """
    return {"version": PROTOCOL_VERSION, "type": "auth", "session_token": token}
