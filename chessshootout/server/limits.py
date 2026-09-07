import ipaddress
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request
from slowapi.util import get_remote_address

from chessshootout.server import logging_setup


RECLAIM_PER_UUID_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PRUNE_THRESHOLD = 4096
RECLAIM_WINDOW_SECONDS = 60.0

MAX_INBOUND_MESSAGE_BYTES = 4096
WS_MESSAGES_PER_SECOND = 30
WS_RATE_WINDOW_SECONDS = 1.0

MATCHMAKE_PER_IP_LIMIT = "60/minute"
RESUME_PER_IP_LIMIT = "60/minute"
RECLAIM_PER_IP_LIMIT = "120/minute"

DEFAULT_TRUSTED_PROXIES = "127.0.0.1/32"

log = logging_setup.get_logger("chess.server.app")


def _parse_trusted_proxies(raw: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """
    Work out which machines are allowed to speak for somebody else, by turning
    the TRUSTED_PROXIES setting into a list of networks. Only a peer inside one
    of them may hand the server a forwarded client address; a bad entry is
    skipped instead of stopping startup, and an entirely unusable setting leaves
    the list empty so nobody is trusted

    :param raw: comma-separated networks or bare addresses, blanks tolerated
    :returns: the entries that parsed, in the order they were written
    """
    networks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return networks


TRUSTED_PROXIES_RAW = os.environ.get("TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES)
TRUSTED_PROXIES = _parse_trusted_proxies(TRUSTED_PROXIES_RAW)


def log_trusted_proxies(
    raw: str | None = None,
    trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
) -> None:
    """
    Record which proxies this server trusts, once at startup, so an operator can
    see the effective set in the logs instead of guessing at it. A configured
    value that parsed to nothing is a warning rather than an info line, because
    it silently moves every per-IP limit onto the socket peer

    :param raw: the setting as configured; None reads the module's own value
    :param trusted: the parsed networks; None reads the module's own list
    """
    raw = TRUSTED_PROXIES_RAW if raw is None else raw
    trusted = TRUSTED_PROXIES if trusted is None else trusted
    if raw.strip() and not trusted:
        log.warning("trusted proxies unparsable (TRUSTED_PROXIES=%r); "
                    "rate limits key on the socket peer", raw)
        return
    log.info("trusted proxies %s", ",".join(str(net) for net in trusted) or "none")


def _peer_trusted(
    peer: str, trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """
    Say whether the machine on the other end of the socket is one of the
    configured proxies. A peer that is not an address at all -- the in-process
    test client, for one -- is never trusted

    :param peer: socket peer address as text
    :param trusted: networks that may speak for another client
    :returns: True when the peer falls inside one of those networks
    """
    try:
        ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(ip in net for net in trusted)


def _forwarded_ip(raw: str | None) -> str | None:
    """
    Read the client address a proxy claims to be forwarding, accepting it only
    when it really is an IP address. A header that is missing, empty or
    malformed yields nothing, so a junk value can never become a limiter key

    :param raw: header value as received, surrounding whitespace tolerated
    :returns: the normalised address, or None when there is no usable one
    """
    if not raw:
        return None
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        return None


def client_ip_key(
    request: Request,
    trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
) -> str:
    """
    Decide which address a request counts against for the per-IP limits. The
    server sits behind Cloudflare, so the real player address arrives in the
    cf-connecting-ip header -- but that header is believed only when the socket
    peer is itself a trusted proxy and the value parses as an address, otherwise
    anyone could forge a key and slip past their own limits

    :param request: inbound HTTP request, read for its peer and headers
    :param trusted: networks allowed to forward an address; None uses the
        module's configured set
    :returns: the address the limiter counts this call against
    """
    trusted = TRUSTED_PROXIES if trusted is None else trusted
    peer = get_remote_address(request)
    if _peer_trusted(peer, trusted):
        forwarded = _forwarded_ip(request.headers.get("cf-connecting-ip"))
        if forwarded is not None:
            return forwarded
    return peer


class UuidRateLimiter:
    """
    A sliding-window call counter keyed by whoever is calling rather than by
    where they call from, for the places an address is the wrong identity:
    session reclaims, annotation and quick-chat floods, and the message cap on a
    single websocket. Time comes from an injected clock, so tests drive it
    instead of waiting
    """

    def __init__(self, limit_per_minute: int, window_seconds: float,
                 now_provider: Callable[[], float] = time.monotonic) -> None:
        """
        Build one limiter with its own allowance, window and clock. Each place
        that limits by identity keeps its own instance, so their counts never
        interfere

        :param limit_per_minute: how many calls one key may make inside the
            window
        :param window_seconds: length of the sliding window in seconds
        :param now_provider: monotonic seconds source, injected for tests
        """
        self.limit = limit_per_minute
        self.window = window_seconds
        self._now = now_provider
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, cutoff: float) -> None:
        """
        Forget every key whose calls have all aged out, so the table cannot grow
        by one entry for each player id the server has ever seen. It runs only
        once the table is past RATE_LIMIT_PRUNE_THRESHOLD, leaving the ordinary
        call path untouched

        :param cutoff: monotonic timestamp in seconds; calls older than it no
            longer count towards any limit
        """
        for key in list(self._calls.keys()):
            d = self._calls[key]
            while d and d[0] < cutoff:
                d.popleft()
            if not d:
                del self._calls[key]

    def hit(self, key: str) -> bool:
        """
        Count one call for a key and say whether it is allowed. Calls that have
        aged out of the window are forgotten first, so an allowance comes back
        gradually rather than all at once on a boundary, and a refused call is
        not recorded -- being over the limit never extends the block

        :param key: identity being limited, such as a player id
        :returns: True when the call is inside the allowance, False when it is
            over and should be refused
        """
        now = self._now()
        cutoff = now - self.window
        if len(self._calls) > RATE_LIMIT_PRUNE_THRESHOLD:
            self._prune(cutoff)
        d = self._calls[key]
        while d and d[0] < cutoff:
            d.popleft()
        if len(d) >= self.limit:
            return False
        d.append(now)
        return True
