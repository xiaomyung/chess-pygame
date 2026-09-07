"""Server-side annotation moderation wired into the live relay, driven end-to-end
through real websockets with a fake clock.

Detection runs off the event loop (asyncio.to_thread) on a snapshot of both
stores taken before the await, so an adversarial worst-case detect cannot
stall clocks/pings for other rooms; the room's liveness is re-validated after
the await (race discipline), so a finalize landing mid-detection drops the
whole outcome instead of writing to a freshly reset store. Past that gate,
ALL block state writes (strip, trip_count, share_muted, mute-clear) commit
before the first send await. At the mute trip the corrective snapshot
therefore already carries the emptied store and doubles as the clearing
snapshot -- one wire event, routed through the hide-aware corrective path.
The blocked/suspect message to the drawer carries only the MOVER'S OWN subset
of the matched marks: a cross-color union can match up to two full stores
(~256 arrows), which would burst the wire model's MAX_SHARED_ARROWS cap and
tear down the session mid-block, and the client can only red-flag its own
marks anyway. Every "opponent received nothing" claim is proved with a ping
sentinel: the very next frame the peer reads back must be its pong, so a
stray relay would surface ahead of it.

Deltas moderate anchored at the changed mark only on ADD; a remove runs the
full-scan path (changed=None) because the raster stage's local excess-ink
guard means a pattern can sit suppressed by an ADJACENT decoy mark -- delete
the decoy and the now-clean pattern lies outside the removed mark's search
window, so an anchored rescan would miss it for the rest of the game.

The concrete trip inputs (4-arrow knight-pinwheel swastika, the 8-arrow novel
pinwheel that only the stage-4 net flags, and the split-across-colors collusion
swastika) are the detector's own fixtures -- this file asserts the RELAY
consequences, not the geometry.
"""
import asyncio
import json
import os
import random
import threading
import time

import pytest

from chessshootout.server import handlers
from chessshootout.server.app import create_app
from chessshootout.server.handlers import handle_annotation_delta, handle_annotations_state
from chessshootout.server.moderation import detector
from chessshootout.server.moderation.load import (
    BUCKET_PRUNE_THRESHOLD, MAX_CONCURRENT_DETECTS, PLAYER,
    PLAYER_BURST_CPU_SECONDS, PLAYER_REFILL_CPU_SECONDS, ROOM_BURST_CPU_SECONDS,
    ROOM_REFILL_CPU_SECONDS, ModerationLoad,
)
from chessshootout.server.protocol import (
    ANNOTATIONS_PER_SECOND, PROTOCOL_VERSION, Reason)
from tests.helpers import FakeClock
from tests.server import moderation_helpers as M
from tests.server.conftest import ALICE, BOB, auth_msg
from fastapi.testclient import TestClient


SWASTIKA = M.arrows_from_segments(
    [(tuple(a), tuple(b)) for a, b in M.SWASTIKA_SCREENSHOTS["v1_hooks_only_pinwheel"]])
NOVEL = M.arrows_from_segments([(tuple(a), tuple(b)) for a, b in M.NOVEL_PINWHEEL])


def _wire_arrows(arrows):
    return [{"from": f, "to": t} for f, t in arrows]


def _matchmake(client, *, uuid, nickname, side, hide=False, time=5, inc=0):
    return client.post("/matchmake", json={
        "version": PROTOCOL_VERSION, "client_uuid": uuid, "nickname": nickname,
        "time_minutes": time, "increment_seconds": inc, "side_preference": side,
        "hide_opp_marks": hide,
    }).json()


def _send(ws, **fields):
    ws.send_text(json.dumps({"version": PROTOCOL_VERSION, **fields}))


def _recv(ws):
    return json.loads(ws.receive_text())


def _pong(ws, ply=0):
    _send(ws, type="ping", ply=ply)
    return _recv(ws)


def _room(client, a):
    return client.app.state.rooms.get(a["room_id"])


class _Paired:
    def __init__(self, client, ws_w, ws_b, a):
        self.client = client
        self.ws_w = ws_w
        self.ws_b = ws_b
        self.a = a

    def room(self):
        return _room(self.client, self.a)


def _pair(client, *, white_hide=False, black_hide=False):
    random.seed(0)
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white", hide=white_hide)
    b = _matchmake(client, uuid=BOB, nickname="Bob", side="black", hide=black_hide)
    ws_w = client.websocket_connect(f"/ws/{a['room_id']}").__enter__()
    ws_w.send_text(json.dumps(auth_msg(a["session_token"])))
    ws_b = client.websocket_connect(f"/ws/{b['room_id']}").__enter__()
    ws_b.send_text(json.dumps(auth_msg(b["session_token"])))
    ws_w.receive_text()
    ws_b.receive_text()
    return _Paired(client, ws_w, ws_b, a)


def _fresh_client(monkeypatch, *, moderation_off=False):
    if moderation_off:
        monkeypatch.setenv("MODERATION_OFF", "1")
    else:
        monkeypatch.delenv("MODERATION_OFF", raising=False)
    return TestClient(create_app(now_provider=FakeClock(), max_rooms=8))


# --- hard block: strip + corrective + no relay --------------------------------

def test_block_strips_store_corrects_opponent_and_never_relays(client):
    p = _pair(client)
    _send(p.ws_w, type="annotations_state", sharing=True,
          highlights=[], arrows=_wire_arrows(SWASTIKA))

    corrective = _recv(p.ws_b)
    assert corrective["type"] == "annotations_state"
    assert corrective["sharing"] is True
    assert corrective["arrows"] == []
    assert corrective["highlights"] == []

    blocked = _recv(p.ws_w)
    assert blocked["type"] == "annotations_blocked"
    assert blocked["action"] == "blocked"

    room = p.room()
    assert room.annotations_white.arrows == []
    assert room.annotations_white.trip_count == 1
    assert room.annotations_white.share_muted is False

    assert _pong(p.ws_b)["type"] == "pong"


def test_blocked_message_carries_matched_marks_and_mute_state(client):
    p = _pair(client)
    _send(p.ws_w, type="annotations_state", sharing=True,
          highlights=[], arrows=_wire_arrows(SWASTIKA))
    assert _recv(p.ws_b)["type"] == "annotations_state"

    blocked = _recv(p.ws_w)
    assert blocked["action"] == "blocked"
    assert blocked["share_muted"] is False
    got = {(arrow["from"], arrow["to"]) for arrow in blocked["arrows"]}
    assert got == set(SWASTIKA)
    assert blocked["highlights"] == []


# --- three trips -> share muted -----------------------------------------------

def test_three_trips_mute_share_and_muted_attempts_noop(client):
    p = _pair(client)
    swastika = _wire_arrows(SWASTIKA)

    for trip in (1, 2):
        _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
              arrows=swastika)
        assert _recv(p.ws_b)["type"] == "annotations_state"
        blk = _recv(p.ws_w)
        assert blk["type"] == "annotations_blocked" and blk["share_muted"] is False
        assert p.room().annotations_white.trip_count == trip

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
          arrows=swastika)
    clearing = _recv(p.ws_b)
    assert clearing["type"] == "annotations_state"
    assert clearing["sharing"] is True
    assert clearing["arrows"] == [] and clearing["highlights"] == []
    blocked = _recv(p.ws_w)
    assert blocked["type"] == "annotations_blocked"
    assert blocked["share_muted"] is True
    assert p.room().annotations_white.share_muted is True

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=["e4"],
          arrows=[])
    err = _recv(p.ws_w)
    assert err["type"] == "error"
    assert err["reason"] == Reason.SHARE_MUTED
    assert err["msg_type"] == "annotations_state"
    assert _pong(p.ws_b)["type"] == "pong"


def test_muted_sharing_false_snapshot_still_processes(client):
    p = _pair(client)
    room = p.room()
    room.annotations_white.share_muted = True
    room.annotations_white.sharing = True
    room.annotations_white.highlights = {"e4"}

    _send(p.ws_w, type="annotations_state", sharing=False, highlights=[], arrows=[])
    off = _recv(p.ws_b)
    assert off["type"] == "annotations_state"
    assert off["sharing"] is False
    assert off["highlights"] == [] and off["arrows"] == []
    assert room.annotations_white.highlights == set()
    assert room.annotations_white.share_muted is True


def test_muted_delta_noops_with_reason(client):
    p = _pair(client)
    p.room().annotations_white.share_muted = True

    _send(p.ws_w, type="annotation_delta", action="add", kind="highlight", square="e4")
    err = _recv(p.ws_w)
    assert err["type"] == "error"
    assert err["reason"] == Reason.SHARE_MUTED
    assert err["msg_type"] == "annotation_delta"
    assert _pong(p.ws_b)["type"] == "pong"
    assert p.room().annotations_white.highlights == set()


# --- soft flag: relay + suspect notice ----------------------------------------

def test_suspect_relays_and_notifies_without_strip_or_trip(client):
    p = _pair(client)
    _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
          arrows=_wire_arrows(NOVEL))

    relayed = _recv(p.ws_b)
    assert relayed["type"] == "annotations_state"
    assert len(relayed["arrows"]) == len(NOVEL)

    notice = _recv(p.ws_w)
    assert notice["type"] == "annotations_blocked"
    assert notice["action"] == "suspect"

    room = p.room()
    assert len(room.annotations_white.arrows) == len(NOVEL)
    assert room.annotations_white.trip_count == 0


# --- hide opponent marks ------------------------------------------------------

def test_hide_via_matchmake_field_skips_relay_and_notifies_once(client):
    p = _pair(client, black_hide=True)

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=["e4"], arrows=[])
    notice = _recv(p.ws_w)
    assert notice["type"] == "error"
    assert notice["reason"] == Reason.OPP_HIDES_MARKS
    assert notice["msg_type"] == "annotations_state"
    assert _pong(p.ws_b)["type"] == "pong"

    assert p.room().annotations_white.highlights == {"e4"}

    _send(p.ws_w, type="annotation_delta", action="add", kind="highlight", square="d4")
    assert _pong(p.ws_w)["type"] == "pong"
    assert _pong(p.ws_b)["type"] == "pong"
    assert p.room().annotations_white.highlights == {"e4", "d4"}


def test_mid_game_hide_toggle_then_unhide_pushes_snapshot(client):
    p = _pair(client)

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=["e4"], arrows=[])
    assert _recv(p.ws_b)["type"] == "annotations_state"

    _send(p.ws_b, type="set_marks_visibility", hide_opp=True)
    assert _pong(p.ws_b)["type"] == "pong"
    assert p.room().slot("black").hide_opp_marks is True

    _send(p.ws_w, type="annotation_delta", action="add", kind="highlight", square="d5")
    notice = _recv(p.ws_w)
    assert notice["type"] == "error"
    assert notice["reason"] == Reason.OPP_HIDES_MARKS
    assert _pong(p.ws_b)["type"] == "pong"

    _send(p.ws_b, type="set_marks_visibility", hide_opp=False)
    pushed = _recv(p.ws_b)
    assert pushed["type"] == "annotations_state"
    assert pushed["sharing"] is True
    assert set(pushed["highlights"]) == {"e4", "d5"}
    assert p.room().slot("black").hide_opp_marks is False


# --- resume interplay with the shield -----------------------------------------

def test_resume_zeroes_opponent_set_for_a_hidden_requester(client):
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white")
    _matchmake(client, uuid=BOB, nickname="Bob", side="black", hide=True)
    room = _room(client, a)
    room.annotations_white.sharing = True
    room.annotations_white.highlights = {"e4", "d5"}
    room.annotations_white.arrows = [("e2", "e4")]
    room.annotations_black.sharing = True
    room.annotations_black.highlights = {"h5"}

    b_token = room.slot("black").session_token
    r = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": a["room_id"],
        "session_token": b_token,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["your_color"] == "black"
    assert body["hide_opp_marks"] is True

    opp = body["white_annotations"]
    assert opp["sharing"] is False
    assert opp["highlights"] == [] and opp["arrows"] == []

    own = body["black_annotations"]
    assert own["highlights"] == ["h5"]


def test_resume_carries_mute_and_preference(client):
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white")
    _matchmake(client, uuid=BOB, nickname="Bob", side="black")
    room = _room(client, a)
    room.annotations_white.share_muted = True

    r = client.post("/resume", json={
        "version": PROTOCOL_VERSION, "room_id": a["room_id"],
        "session_token": a["session_token"],
    })
    body = r.json()
    assert body["share_muted"] is True
    assert body["hide_opp_marks"] is False
    assert body["black_annotations"]["sharing"] is False


# --- cross-color collusion ----------------------------------------------------

def test_cross_color_collusion_strips_both_and_trips_completing_mover(client):
    p = _pair(client)
    a_side = _wire_arrows(SWASTIKA[:2])
    b_side = _wire_arrows(SWASTIKA[2:])

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=[], arrows=a_side)
    assert _recv(p.ws_b)["type"] == "annotations_state"

    _send(p.ws_b, type="annotations_state", sharing=True, highlights=[], arrows=b_side)
    white_corr = _recv(p.ws_w)
    assert white_corr["type"] == "annotations_state"
    assert white_corr["arrows"] == []
    black_corr = _recv(p.ws_b)
    assert black_corr["type"] == "annotations_state"
    assert black_corr["arrows"] == []
    blocked = _recv(p.ws_b)
    assert blocked["type"] == "annotations_blocked"
    assert blocked["action"] == "blocked"
    # only the completing mover's own contribution comes back on the wire --
    # the full union can exceed MAX_SHARED_ARROWS and the client can only
    # red-flag its own marks
    got = {(arrow["from"], arrow["to"]) for arrow in blocked["arrows"]}
    assert got == set(SWASTIKA[2:])

    room = p.room()
    assert room.annotations_white.arrows == []
    assert room.annotations_black.arrows == []
    assert room.annotations_black.trip_count == 1
    assert room.annotations_white.trip_count == 0


def test_blocked_wire_marks_stay_within_the_arrow_cap(client):
    """A union verdict can match both stores at once (up to ~2x
    MAX_SHARED_ARROWS distinct arrows); building AnnotationsBlockedMessage
    from the raw union raises pydantic ValidationError mid-block and tears
    down the mover's socket with half-applied trip state. The own-store
    filter keeps the wire list at or below the store cap by construction."""
    from chessshootout.server.protocol import AnnotationsBlockedMessage, MAX_SHARED_ARROWS
    from chessshootout.server.rooms import SharedAnnotations

    def coord(i):
        return f"{chr(ord('a') + i % 8)}{i // 8 + 1}"

    union_arrows = [(coord(i), coord(i + off))
                    for off in (1, 2, 7, 8, 9) for i in range(64 - off)]
    union_arrows = union_arrows[:2 * MAX_SHARED_ARROWS]
    store = SharedAnnotations()
    store.arrows = list(union_arrows[:MAX_SHARED_ARROWS])
    verdict = detector.Verdict(detector.BLOCKED, pattern_id="x",
                               matched_arrows=list(union_arrows),
                               matched_highlights=[])
    own_arrows, own_highlights = handlers._own_matched(store, verdict)
    msg = AnnotationsBlockedMessage(
        action="blocked", highlights=own_highlights,
        arrows=[{"from": f, "to": t} for f, t in own_arrows], share_muted=False)
    assert len(msg.arrows) == MAX_SHARED_ARROWS
    assert {(a.from_sq, a.to_sq) for a in msg.arrows} == set(store.arrows)


# --- kill switch --------------------------------------------------------------

def test_moderation_off_relays_the_swastika_untouched(monkeypatch):
    client = _fresh_client(monkeypatch, moderation_off=True)
    p = _pair(client)

    _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
          arrows=_wire_arrows(SWASTIKA))
    relayed = _recv(p.ws_b)
    assert relayed["type"] == "annotations_state"
    assert {(arrow["from"], arrow["to"]) for arrow in relayed["arrows"]} == set(SWASTIKA)

    room = p.room()
    assert len(room.annotations_white.arrows) == len(SWASTIKA)
    assert room.annotations_white.trip_count == 0
    assert _pong(p.ws_w)["type"] == "pong"


# --- lifecycle: result + rematch reset, hide preference survives --------------

def test_result_resets_mod_state_and_rematch_swap_keeps_hide_preference(client):
    a = _matchmake(client, uuid=ALICE, nickname="Alice", side="white", hide=True)
    _matchmake(client, uuid=BOB, nickname="Bob", side="black")
    room = _room(client, a)
    rooms = client.app.state.rooms

    room.annotations_white.trip_count = 2
    room.annotations_white.share_muted = True
    room.annotations_white.opp_hidden_notice_sent = True
    assert room.slot("white").hide_opp_marks is True

    assert rooms.finalize_result(room.room_id, Reason.RESIGNATION, winner_color="black")
    assert room.annotations_white.trip_count == 0
    assert room.annotations_white.share_muted is False
    assert room.annotations_white.opp_hidden_notice_sent is False
    assert room.slot("white").hide_opp_marks is True

    assert rooms.reset_for_rematch(room.room_id)
    assert room.slot("black").hide_opp_marks is True
    assert room.slot("white").hide_opp_marks is False
    assert room.annotations_white.trip_count == 0
    assert room.annotations_white.share_muted is False


# --- race discipline + delta search-window pins -------------------------------

def test_mute_state_commits_before_the_first_block_await(client, monkeypatch):
    """The block path awaits several sends; a finalize can interleave at any of
    them and reset() the annotation stores. Every state write (strip, trip
    count, share_muted, mute-clear) must therefore land BEFORE the first SEND
    await -- a post-send `share_muted = True` would re-mute a freshly reset
    store. (Detection itself awaits to_thread earlier; that window is covered
    by the liveness re-validation pinned below.) Pinned by observing the store
    at the moment each send coroutine fires."""
    p = _pair(client)
    swastika = _wire_arrows(SWASTIKA)
    for _ in (1, 2):
        _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
              arrows=swastika)
        assert _recv(p.ws_b)["type"] == "annotations_state"
        assert _recv(p.ws_w)["type"] == "annotations_blocked"

    room = p.room()
    observed = []
    real_send = handlers.send

    async def spy(ws, message):
        observed.append((type(message).__name__,
                         room.annotations_white.share_muted,
                         sorted(room.annotations_white.highlights),
                         list(room.annotations_white.arrows)))
        return await real_send(ws, message)

    monkeypatch.setattr(handlers, "send", spy)
    _send(p.ws_w, type="annotations_state", sharing=True, highlights=[],
          arrows=swastika)
    assert _recv(p.ws_b)["type"] == "annotations_state"
    assert _recv(p.ws_w)["type"] == "annotations_blocked"

    assert [name for name, _, _, _ in observed] == [
        "AnnotationsStateMessage", "AnnotationsBlockedMessage"]
    assert all(muted for _, muted, _, _ in observed)
    assert all(h == [] and a == [] for _, _, h, a in observed)


def test_finalize_during_detection_drops_the_relay(client, monkeypatch):
    """Detection now happens inside asyncio.to_thread, so a finalize can land
    while the verdict is being computed. Race discipline: the room's liveness
    is re-validated after the await -- without it, a clean verdict computed on
    a live game relays annotations into a finished one (and a blocked verdict
    would strip/trip a freshly reset store). Simulated by finalizing from
    inside the detection callable itself."""
    p = _pair(client)
    room = p.room()
    rooms = client.app.state.rooms
    real = handlers._moderate

    def finalize_then_detect(*args):
        rooms.finalize_result(room.room_id, Reason.RESIGNATION, winner_color="black")
        return real(*args)

    monkeypatch.setattr(handlers, "_moderate", finalize_then_detect)
    _send(p.ws_w, type="annotation_delta", action="add", kind="highlight",
          square="e4")
    assert _pong(p.ws_w)["type"] == "pong"
    assert room.result is not None
    assert _pong(p.ws_b)["type"] == "pong"


def test_add_delta_anchors_search_and_remove_delta_full_scans(client, monkeypatch):
    """Anchored raster search only tries placements overlapping the changed
    mark's bbox, and the raster stage suppresses a pattern via LOCAL excess
    ink -- so an adjacent decoy mark can hold a symbol under the IoU bar.
    Deleting the decoy un-suppresses a pattern that sits outside the removed
    mark's window, and nothing else ever rescans the store, so removals must
    take the full-scan path (changed=None) while adds keep the anchored perf
    path. Both detect calls (own store + cross-color union) get the same
    treatment."""
    p = _pair(client)
    calls = []
    real_detect = detector.detect

    def spy(arrows, highlights, changed=None, context=()):
        calls.append(changed)
        return real_detect(arrows, highlights, changed=changed, context=context)

    monkeypatch.setattr(detector, "detect", spy)

    _send(p.ws_w, type="annotation_delta", action="add", kind="highlight",
          square="e4")
    assert _recv(p.ws_b)["type"] == "annotation_delta"
    assert calls == ["e4", "e4"]

    calls.clear()
    _send(p.ws_w, type="annotation_delta", action="add", kind="arrow",
          **{"from": "a1", "to": "a3"})
    assert _recv(p.ws_b)["type"] == "annotation_delta"
    assert calls == [("a1", "a3"), ("a1", "a3")]

    calls.clear()
    _send(p.ws_w, type="annotation_delta", action="remove", kind="highlight",
          square="e4")
    assert _recv(p.ws_b)["type"] == "annotation_delta"
    assert calls == [None, None]

    calls.clear()
    _send(p.ws_w, type="annotation_delta", action="remove", kind="arrow",
          **{"from": "a1", "to": "a3"})
    assert _recv(p.ws_b)["type"] == "annotation_delta"
    assert calls == [None, None]


# --- worst-case timing pin ----------------------------------------------------

def _clear_detect_memo():
    """detect() memoises on the exact input key, so a resampled identical call
    times the memo instead of the detector."""
    detector._CACHE.clear()
    detector._CACHE_ORDER.clear()


def test_worst_case_delta_moderation_stays_within_budget():
    """Event-loop budget pin: a near-cap store (127 arrows + 63 highlights)
    with an anchored add-update per iteration, plus a full rescan (the remove
    path). Wall-clock time per iteration is not deterministic under xdist -- a
    loaded worker can lose 100 ms+ to scheduler contention with nothing wrong --
    so the pin asserts the BEST of the samples: the fastest of ten anchored
    updates (and three full rescans) must clear 100 ms. Contention inflates
    individual samples but at least one runs uncontended; an order-of-magnitude
    regression in the real per-message cost pushes every sample over the line.
    Each verdict is also asserted well-formed so the loop can't pass by silently
    erroring."""
    from chessshootout.backend.utils import Square, coord_from_square

    def coord(x, y):
        return coord_from_square(Square(row=y, col=x))

    rng = random.Random(7)
    arrows = []
    seen = set()
    while len(arrows) < 127:
        a = (rng.randrange(8), rng.randrange(8))
        b = (rng.randrange(8), rng.randrange(8))
        if a == b:
            continue
        pair = (coord(*a), coord(*b))
        if pair in seen:
            continue
        seen.add(pair)
        arrows.append(pair)
    highlights = [coord(x, y) for x in range(8) for y in range(8)][:63]

    detector.detect(arrows[:5], highlights[:5])

    budget = 0.1
    delta_times = []
    for i in range(10):
        new = (coord(i % 8, (i * 3) % 8), coord((i + 2) % 8, (i * 5 + 1) % 8))
        if new[0] == new[1]:
            new = (new[0], coord((i + 3) % 8, i % 8))
        _clear_detect_memo()
        t0 = time.perf_counter()
        verdict = detector.detect(arrows + [new], highlights, changed=new)
        delta_times.append(time.perf_counter() - t0)
        assert verdict.kind in (detector.CLEAN, detector.SUSPECT, detector.BLOCKED)

    # The rescan samples are byte-identical calls, so without emptying the memo
    # first only sample 1 does any work and the other two time a dict lookup --
    # the min() would then report the cache's cost, not the rescan's.
    full_times = []
    for _ in range(3):
        _clear_detect_memo()
        t0 = time.perf_counter()
        detector.detect(arrows, highlights)
        full_times.append(time.perf_counter() - t0)

    assert min(delta_times) < budget
    assert min(full_times) < budget


# --- moderation load meter ----------------------------------------------------
#
# Detection is CPU-heavy work reachable at ANNOTATIONS_PER_SECOND per player on
# a 1.5-CPU container, and the memo keys on the exact arrow tuple so a shuffled
# resend is always a cache miss. Two self-paired clients spamming the dense
# CLEAN set (moderation_helpers.dense_clean_arrows, tens of ms per detect)
# therefore buy the better part of a CPU-second per second per room, which
# starves the sweep loop through GIL contention rather than merely making
# detection slow: clocks stop, heartbeats time out, games abandon.
#
# The fix is a meter, not a bypass: over budget the server STOPS SHARING (store
# cleared, sharing off, opponent told sharing=False, sender gets the same
# rate_limited frame the annotation limiter sends) rather than publishing marks
# it has not vetted. Skipping moderation under load instead would be a
# filter-evasion hole: flood the room, then draw the symbol.
#
# Leaky-bucket shape (moderation.load): per-player and per-room buckets refill
# at a fixed CPU-seconds-per-second rate with a burst capacity on top, so a
# single expensive-but-legal message can never trip the meter -- only a
# sustained rate above the refill can. That is what keeps honest play identical:
# a human draws at mouse-release speed (one delta per right-drag), a flood runs
# at the limiter's ceiling.


class _RecordingWS:

    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)

    def of_type(self, kind):
        return [m for m in self.sent if m["type"] == kind]


async def _handler_room(app):
    rooms = app.state.rooms
    await rooms.enqueue(client_uuid=ALICE, nickname="Alice", session_token="ta",
                        time_minutes=5, increment_seconds=0, side_preference="white")
    await rooms.enqueue(client_uuid=BOB, nickname="Bob", session_token="tb",
                        time_minutes=5, increment_seconds=0, side_preference="black")
    room = list(rooms._active.values())[0]
    ws_w, ws_b = _RecordingWS(), _RecordingWS()
    app.state.connections.add(room.room_id, room.white.client_uuid, ws_w)
    app.state.connections.add(room.room_id, room.black.client_uuid, ws_b)
    return room, ws_w, ws_b


def _state_raw(arrows, highlights=()):
    return json.dumps({"version": PROTOCOL_VERSION, "type": "annotations_state",
                       "sharing": True, "highlights": list(highlights),
                       "arrows": _wire_arrows(arrows)})


def _delta_raw(square):
    return json.dumps({"version": PROTOCOL_VERSION, "type": "annotation_delta",
                       "action": "add", "kind": "highlight", "square": square})


def _player_debt(load, room, color):
    bucket = load._buckets.get((PLAYER, room.room_id, color))
    return 0.0 if bucket is None else bucket.debt


async def test_over_budget_room_stops_sharing_without_bypassing_moderation(
        app, clock, monkeypatch):
    """Over budget the relay is suppressed, not the moderation: detect() is
    never reached, because nothing reaches the opponent for it to have vetted.
    The store is emptied so no later surface (/resume, un-hide snapshot,
    corrective) can publish the unvetted marks either, and the opponent is told
    sharing stopped exactly the way a manual share-off tells them."""
    room, ws_w, ws_b = await _handler_room(app)
    await handle_annotations_state(app, ws_w, room, "white",
                                   _state_raw([("e2", "e4")], ["e4"]))
    assert ws_b.of_type("annotations_state"), "baseline share must relay"

    load = handlers._moderation_load(app)
    load.charge(room.room_id, "white", ROOM_BURST_CPU_SECONDS, clock())

    detected = []
    real_detect = detector.detect

    def spy(*args, **kwargs):
        detected.append(args)
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(detector, "detect", spy)

    outcome = await handle_annotations_state(app, ws_w, room, "white",
                                             _state_raw(SWASTIKA))

    assert outcome == "load_suppressed"
    assert detected == []
    off = ws_b.of_type("annotations_state")[-1]
    assert off["sharing"] is False
    assert off["arrows"] == [] and off["highlights"] == []
    err = ws_w.of_type("error")[-1]
    assert err["reason"] == Reason.RATE_LIMITED
    assert err["msg_type"] == "annotations_state"
    store = room.annotations_white
    assert store.sharing is False
    assert store.arrows == [] and store.highlights == set()


async def test_over_budget_delta_is_suppressed_and_never_stored(app, clock):
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)
    load.charge(room.room_id, "white", ROOM_BURST_CPU_SECONDS, clock())

    outcome = await handle_annotation_delta(app, ws_w, room, "white", _delta_raw("e4"))

    assert outcome == "load_suppressed"
    assert ws_b.of_type("annotation_delta") == []
    assert room.annotations_white.highlights == set()
    assert ws_w.of_type("error")[-1]["reason"] == Reason.RATE_LIMITED


async def test_suppression_lifts_once_the_bucket_drains(app, clock):
    """The meter is a leak, not a latch -- the room shares again on its own once
    the window has refilled, with no reconnect and no operator action."""
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)
    load.charge(room.room_id, "white", ROOM_BURST_CPU_SECONDS, clock())
    assert await handle_annotations_state(
        app, ws_w, room, "white", _state_raw([("e2", "e4")])) == "load_suppressed"

    clock.advance(ROOM_BURST_CPU_SECONDS / ROOM_REFILL_CPU_SECONDS + 1.0)

    assert await handle_annotations_state(
        app, ws_w, room, "white", _state_raw([("e2", "e4")])) == "relayed"
    assert ws_b.of_type("annotations_state")[-1]["arrows"] == [
        {"from": "e2", "to": "e4"}]


async def test_normal_annotation_traffic_is_metered_but_never_suppressed(app, clock):
    """The honest-play invariant: a full session of ordinary marks relays every
    frame and never comes near the budget. The debt assertion keeps this honest
    -- it proves the meter really is charging these messages (a meter wired to
    nothing would also pass the relay assertions)."""
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)

    marks = [("e2", "e4"), ("g1", "f3"), ("d2", "d4")]
    for step in range(20):
        clock.advance(0.25)
        await handle_annotations_state(app, ws_w, room, "white",
                                       _state_raw(marks[:1 + step % 3], ["e4", "d5"]))
        await handle_annotation_delta(app, ws_b, room, "black", _delta_raw("h5"))

    assert len(ws_b.of_type("annotations_state")) == 20
    assert len(ws_w.of_type("annotation_delta")) == 20
    assert ws_w.of_type("error") == []
    assert ws_b.of_type("error") == []
    assert load.over_budget(room.room_id, "white", clock()) is False
    assert load.over_budget(room.room_id, "black", clock()) is False
    assert _player_debt(load, room, "white") > 0.0


HUMAN_DRAW_RATE = 3


async def _measured_cost(app, room, ws, arrows):
    load = handlers._moderation_load(app)
    before = _player_debt(load, room, "white")
    await handle_annotations_state(app, ws, room, "white", _state_raw(arrows))
    return _player_debt(load, room, "white") - before


async def test_honest_drawing_stays_far_under_the_player_refill(app, clock):
    """The honest-play half of the budget's justification, and the half that can
    run anywhere: an ordinary two-mark message charged at a frantic human draw rate
    has to sit well under the player refill. The headroom here is orders of
    magnitude, so no runner's mood can flip the verdict -- unlike the flood
    comparison below, which is a live CPU measurement and is opt-in for that
    reason."""
    room, ws_w, _ws_b = await _handler_room(app)
    await handle_annotations_state(app, ws_w, room, "white", _state_raw([("e2", "e4")]))

    honest_cost = await _measured_cost(app, room, ws_w, [("e2", "e4"), ("g1", "f3")])

    assert honest_cost > 0.0, "the meter really is charging these messages"
    assert honest_cost * HUMAN_DRAW_RATE < PLAYER_REFILL_CPU_SECONDS, (
        f"honest drawing {honest_cost * HUMAN_DRAW_RATE:.3f} CPU-s/s no longer "
        f"fits the {PLAYER_REFILL_CPU_SECONDS} refill -- honest play would trip")


@pytest.mark.skipif(
    not os.environ.get("CHESS_CHECK_PERF"),
    reason="opt-in CPU-cost pin; set CHESS_CHECK_PERF=1 (shared CI runners are too noisy)",
)
async def test_dense_clean_flood_outruns_the_refill(app, clock):
    """The hostile half: one dense-CLEAN message must cost enough that the
    limiter's ceiling (ANNOTATIONS_PER_SECOND) outruns the refills, so a flood
    always fills the bucket. Both claims are ratios of a LIVE thread_time
    measurement to the shipped constants, and a shared CI runner delivers wildly
    different CPU-seconds for identical code -- the same reason
    test_dense_clean_set_timing_under_its_own_budget is opt-in. Run it where the
    number means something: CHESS_CHECK_PERF=1 locally, before and after touching
    the detector or the budget."""
    room, ws_w, _ws_b = await _handler_room(app)
    await handle_annotations_state(app, ws_w, room, "white", _state_raw([("e2", "e4")]))

    dense_cost = await _measured_cost(app, room, ws_w, M.dense_clean_arrows())

    flood = dense_cost * ANNOTATIONS_PER_SECOND
    assert flood > PLAYER_REFILL_CPU_SECONDS, (
        f"one flooding player {flood:.3f} CPU-s/s no longer outruns the "
        f"{PLAYER_REFILL_CPU_SECONDS} player refill -- re-tune the budget")
    assert 2 * flood > ROOM_REFILL_CPU_SECONDS, (
        f"two colluding floods {2 * flood:.3f} CPU-s/s no longer outrun the "
        f"{ROOM_REFILL_CPU_SECONDS} room refill -- re-tune the budget")


async def test_a_saturated_detect_pool_suppresses_instead_of_queueing(app, clock):
    """SECURITY: CPU is charged only AFTER a detect finishes, so a saturated
    semaphore used to park every further annotation in an unbounded wait queue --
    nothing charged, nothing over budget, and coroutines piling up for as long as
    the flood lasted. Admission is bounded now: a message that cannot get a slot
    inside ModerationLoad.admission_timeout takes the same force-stop-sharing path
    an over-budget one does, so the backlog can never outlive the timeout."""
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)
    load.admission_timeout = 0.01
    for _ in range(MAX_CONCURRENT_DETECTS):
        await load.semaphore.acquire()
    try:
        outcome = await handle_annotations_state(app, ws_w, room, "white",
                                                 _state_raw([("e2", "e4")]))
    finally:
        for _ in range(MAX_CONCURRENT_DETECTS):
            load.semaphore.release()

    assert outcome == "load_suppressed"
    assert ws_w.of_type("error")[-1]["reason"] == Reason.RATE_LIMITED
    assert room.annotations_white.sharing is False, "unvetted marks are never stored"
    assert room.annotations_white.arrows == []
    assert ws_b.of_type("annotations_state")[-1]["sharing"] is False, \
        "the opponent is told sharing stopped, exactly as an over-budget stop tells them"


async def test_a_freed_slot_lets_the_next_annotation_through(app, clock):
    """The bound is a timeout, not a latch: the very next message relays once a
    detect slot is free again."""
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)
    load.admission_timeout = 0.01
    for _ in range(MAX_CONCURRENT_DETECTS):
        await load.semaphore.acquire()
    assert await handle_annotations_state(
        app, ws_w, room, "white", _state_raw([("e2", "e4")])) == "load_suppressed"
    for _ in range(MAX_CONCURRENT_DETECTS):
        load.semaphore.release()

    assert await handle_annotations_state(
        app, ws_w, room, "white", _state_raw([("e2", "e4")])) == "relayed"


async def test_the_shared_room_bucket_suppresses_the_opponent_too(app, clock):
    """Pinning the shared-bucket design AND its cost. The room bucket is charged by
    both colors and read by both, so one player filling it force-stops the
    OPPONENT's sharing as well, even though the opponent spent nothing. That is
    deliberate: the abuse this meter defends against is two players cooperating, and
    a per-player-only budget is trivially beaten by splitting the flood across the
    pair. The price is this collateral mute, which lasts only as long as the room
    bucket stays full -- pinned here so it is a known trade, not a surprise."""
    room, ws_w, ws_b = await _handler_room(app)
    load = handlers._moderation_load(app)
    assert await handle_annotations_state(
        app, ws_b, room, "black", _state_raw([("e7", "e5")])) == "relayed"
    load.charge(room.room_id, "white", ROOM_BURST_CPU_SECONDS, clock())
    assert _player_debt(load, room, "black") < PLAYER_BURST_CPU_SECONDS / 100, \
        "the innocent side's own bucket is nowhere near its own ceiling"

    outcome = await handle_annotation_delta(app, ws_b, room, "black", _delta_raw("h5"))

    assert outcome == "load_suppressed"
    assert ws_b.of_type("error")[-1]["reason"] == Reason.RATE_LIMITED
    assert room.annotations_black.sharing is False
    assert room.annotations_black.highlights == set()
    assert ws_w.of_type("annotations_state")[-1]["sharing"] is False


async def test_detect_concurrency_is_capped_by_the_semaphore(app, monkeypatch):
    """Bounding rooms one by one is not enough: every detect runs in a worker
    thread, and N of them at once contend for the GIL with the event loop that
    ticks clocks and answers heartbeats. The global slot count caps how many can
    ever be in flight, whatever the room count."""
    room, ws_w, _ws_b = await _handler_room(app)
    inflight = {"now": 0, "peak": 0}
    lock = threading.Lock()
    real = handlers._moderate

    def slow(*args):
        with lock:
            inflight["now"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["now"])
        time.sleep(0.05)
        try:
            return real(*args)
        finally:
            with lock:
                inflight["now"] -= 1

    monkeypatch.setattr(handlers, "_moderate", slow)
    await asyncio.gather(*[
        handle_annotations_state(app, ws_w, room, "white", _state_raw([("e2", "e4")]))
        for _ in range(6)])

    assert inflight["peak"] == MAX_CONCURRENT_DETECTS


def test_load_bucket_refills_over_the_window():
    """Leaky-bucket arithmetic in isolation: a burst-sized charge trips the
    meter, and only the passage of time clears it -- proportionally, so a room
    that keeps spending stays suppressed while one that stops recovers."""
    load = ModerationLoad(player_refill=1.0, player_burst=2.0,
                          room_refill=1.0, room_burst=2.0)
    assert load.over_budget("r", "white", 0.0) is False

    load.charge("r", "white", 2.0, 0.0)
    assert load.over_budget("r", "white", 0.0) is True
    assert load.over_budget("r", "black", 0.0) is True, "the room bucket is shared"

    assert load.over_budget("r", "white", 1.5) is False
    assert load.over_budget("other", "white", 0.0) is False


def test_load_meter_prunes_drained_buckets_and_keeps_indebted_ones():
    """Rooms are per-match uuids, so the ledger would grow for the process
    lifetime without a sweep -- and the meter cannot lean on rooms.py to tell it
    a room died. Pruning is self-service: a bucket that has refilled to zero is
    indistinguishable from one that never existed, so it is dropped, while a
    bucket still carrying debt survives the sweep it triggers."""
    load = ModerationLoad(player_refill=1.0, player_burst=2.0,
                          room_refill=1.0, room_burst=2.0)
    for i in range(BUCKET_PRUNE_THRESHOLD):
        now = float(i)
        load.charge("live", "white", 1.5, now)
        load.charge(f"gone-{i}", "white", 0.001, now)

    assert len(load._buckets) < BUCKET_PRUNE_THRESHOLD
    assert load.over_budget("live", "white", float(BUCKET_PRUNE_THRESHOLD)) is True
