"""End-to-end annotation moderation over two real clients against the in-process
server (moderation ON by default -- the `server` fixture builds create_app with
no MODERATION_OFF).

Two guarantees are proven on the wire, not in the geometry (the concrete
swastika inputs are the detector's own fixtures): a drawer who paints a full
swastika never lands the complete symbol on the receiver -- neither in one
snapshot (the receiver only ever sees the server's cleaned corrective, empty
arrows) nor drawn incrementally delta-by-delta (the realistic attack: the
completing delta is never relayed and the corrective snapshot wipes the leaked
partials, so the receiver's reconstructed view never once contains every
segment and ends empty) -- while the drawer alone gets `annotations_blocked`;
and the hide-opponent-marks shield (carried at matchmake time, flipped
mid-game with `set_marks_visibility`) stops relay to the hider entirely,
notifies the sender exactly once per share session (the second attempt is an
absence assertion, not a wait), and -- on un-hide -- pushes the accumulated
store the hider missed.
"""
import time

from chessshootout.online.client import OnlineClient
from tests.helpers import fake_uuid4
from tests.online.online_helpers import collect_for, wait_for
from tests.server import moderation_helpers as M


MOD_WHITE = fake_uuid4(41)
MOD_BLACK = fake_uuid4(42)
HIDE_WHITE = fake_uuid4(43)
HIDE_BLACK = fake_uuid4(44)
DELTA_WHITE = fake_uuid4(45)
DELTA_BLACK = fake_uuid4(46)

DELTA_PACING_S = 0.12


SWASTIKA = M.arrows_from_segments(
    [(tuple(a), tuple(b)) for a, b in M.SWASTIKA_SCREENSHOTS["v1_hooks_only_pinwheel"]])


def _connect(addr, uuid, nickname, side, hide=False):
    c = OnlineClient()
    c.connect(addr, {"nickname": nickname, "client_uuid": uuid,
                     "time_minutes": 5, "increment_seconds": 0,
                     "side_preference": side, "hide_opp_marks": hide})
    return c


def _pair(server, white_uuid, black_uuid, *, black_hide=False):
    addr = f"localhost:{server}"
    a = _connect(addr, white_uuid, "Alice", "white")
    b = _connect(addr, black_uuid, "Bob", "black", hide=black_hide)
    assert wait_for(a, "game_start").payload["your_color"] == "white"
    assert wait_for(b, "game_start").payload["your_color"] == "black"
    return addr, a, b


def test_swastika_never_fully_reaches_receiver_and_drawer_is_blocked(server):
    _addr, a, b = _pair(server, MOD_WHITE, MOD_BLACK)

    a.send_annotations_state(True, [], SWASTIKA)

    corrective = wait_for(b, "annotations_state")
    assert corrective is not None
    assert corrective.payload["sharing"] is True
    assert corrective.payload["arrows"] == []
    assert corrective.payload["highlights"] == []

    blocked = wait_for(a, "annotations_blocked")
    assert blocked is not None
    assert blocked.payload["action"] == "blocked"
    got = {(arrow["from"], arrow["to"]) for arrow in blocked.payload["arrows"]}
    assert got == set(SWASTIKA)

    a.disconnect()
    b.disconnect()


def test_incremental_deltas_never_land_the_full_symbol_on_the_receiver(server):
    _addr, a, b = _pair(server, DELTA_WHITE, DELTA_BLACK)
    a.send_annotations_state(True, [], [])

    received = set()
    peak = [0]

    def apply_pending():
        for ev in b.drain_inbound():
            if ev.type == "annotation_delta" and ev.payload["action"] == "add":
                received.add((ev.payload["from"], ev.payload["to"]))
            elif ev.type == "annotations_state":
                received.clear()
                received.update((w["from"], w["to"]) for w in ev.payload["arrows"])
            assert not received >= set(SWASTIKA)
            peak[0] = max(peak[0], len(received))

    for fr, to in SWASTIKA:
        a.send_annotation_delta("add", "arrow", from_sq=fr, to_sq=to)
        time.sleep(DELTA_PACING_S)
        apply_pending()

    blocked = wait_for(a, "annotations_blocked")
    assert blocked is not None
    assert blocked.payload["action"] == "blocked"

    deadline = time.time() + 10.0
    while time.time() < deadline:
        apply_pending()
        if received == set():
            break
        time.sleep(0.05)
    assert received == set()
    assert 0 < peak[0] < len(SWASTIKA)

    a.disconnect()
    b.disconnect()


def test_hide_toggle_opts_out_of_relay_via_matchmake_and_midgame(server):
    _addr, a, b = _pair(server, HIDE_WHITE, HIDE_BLACK, black_hide=True)

    a.send_annotations_state(True, ["e4"], [])
    notice = wait_for(a, "error")
    assert notice is not None
    assert notice.payload["reason"] == "opp_hides_marks"
    stray = [ev.type for ev in collect_for(b, 1.0) if ev.type == "annotations_state"]
    assert stray == []

    a.send_annotation_delta("add", "highlight", square="d5")
    repeat_notice = [ev for ev in collect_for(a, 1.0) if ev.type == "error"]
    assert repeat_notice == []
    still = [ev.type for ev in collect_for(b, 1.0)
             if ev.type in ("annotations_state", "annotation_delta")]
    assert still == []

    b.send_set_marks_visibility(False)
    pushed = wait_for(b, "annotations_state")
    assert pushed is not None
    assert pushed.payload["sharing"] is True
    assert set(pushed.payload["highlights"]) == {"e4", "d5"}

    a.disconnect()
    b.disconnect()
