"""
Polling helpers shared by the end-to-end online tests. Every e2e file drives
real OnlineClients over real sockets, so each one needs the same two waits:
step forward until a named event shows up, and gather a fixed window
"""
import time

from chessshootout.online.client import Event, OnlineClient


POLL_INTERVAL_SECONDS = 0.02


def wait_for(client: OnlineClient, type_name: str,
             timeout: float = 15.0) -> Event | None:
    """
    Drain a client's inbound queue until an event of the wanted type arrives.
    Events stepped past on the way are discarded, so this is for presence
    assertions only; use collect_for when the events in between matter

    :param client: connected client whose inbound queue is polled
    :param type_name: the event type to stop on
    :param timeout: seconds to keep polling before giving up
    :returns: the first matching event, or None if the timeout ran out
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ev in client.drain_inbound():
            if ev.type == type_name:
                return ev
        time.sleep(POLL_INTERVAL_SECONDS)
    return None


def collect_for(client: OnlineClient, timeout: float) -> list[Event]:
    """
    Drain a client's inbound queue for a fixed wall-clock window and return
    everything seen. An assertion that nothing arrived is then about elapsed
    time rather than about polling luck

    :param client: connected client whose inbound queue is polled
    :param timeout: seconds to keep draining before returning
    :returns: every event that arrived during the window, oldest first
    """
    deadline = time.time() + timeout
    seen: list[Event] = []
    while time.time() < deadline:
        seen.extend(client.drain_inbound())
        time.sleep(POLL_INTERVAL_SECONDS)
    seen.extend(client.drain_inbound())
    return seen
