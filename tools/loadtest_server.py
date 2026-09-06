# Synthetic load driver for the chess server.
#
# Pairs N games via /matchmake, connects both players over WebSocket, plays a
# scripted legal opening line, then holds every connection open with 2s
# heartbeats while measuring ping round-trip latency. Reports rooms established
# and RTT; watch `docker stats` on the server for CPU / RAM.
#
# LOCAL / THROWAWAY SERVERS ONLY -- never point this at the shared production box.
#
#   python tools/loadtest_server.py --addr localhost:8000 --rooms 200 --hold 60

import argparse
import asyncio
import statistics
import time
import uuid

import httpx

from chessshootout.online.transport import ServerTransport
from chessshootout.server.protocol import MatchmakeRequest

# A legal Ruy Lopez line (algebraic from -> to), white moves first.
SCRIPT = [
    ("e2", "e4"), ("e7", "e5"), ("g1", "f3"), ("b8", "c6"), ("f1", "b5"),
    ("a7", "a6"), ("b5", "a4"), ("g8", "f6"), ("e1", "g1"), ("f8", "e7"),
    ("f1", "e1"), ("b7", "b5"), ("a4", "b3"), ("d7", "d6"), ("c2", "c3"),
    ("e8", "g8"), ("h2", "h3"), ("c6", "a5"), ("b3", "c2"), ("c7", "c5"),
    ("d2", "d4"), ("d8", "c7"),
]

GAME_START_WAIT_ATTEMPTS = 30
CONNECT_TIMEOUT_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 2.0
MOVE_DELAY_SECONDS = 0.6


def _request(label, side, minutes, increment):
    return MatchmakeRequest(
        nickname=label, client_uuid=str(uuid.uuid4()),
        time_minutes=minutes, increment_seconds=increment, side_preference=side,
    )


async def _await_color(ws):
    for _ in range(GAME_START_WAIT_ATTEMPTS):
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception:
            return None
        if msg and msg.get("type") == "game_start":
            return msg.get("your_color")
    return None


async def _setup_game(transport, http, idx, minutes, increment):
    ra = await transport.matchmake_async(_request(f"lt-w{idx}", "white", minutes, increment), http)
    rb = await transport.matchmake_async(_request(f"lt-b{idx}", "black", minutes, increment), http)
    if ra.room_id != rb.room_id:
        raise RuntimeError(f"cross-paired {ra.room_id} != {rb.room_id}")
    ws_a = await transport.ws_connect(ra.room_id, ra.session_token)
    ws_b = await transport.ws_connect(rb.room_id, rb.session_token)
    color_a = await _await_color(ws_a)
    await _await_color(ws_b)
    return (ws_a, ws_b) if color_a == "white" else (ws_b, ws_a)


async def _play_game(white, black, stop, rtts, resyncs):
    pending = {white: None, black: None}
    ply = [0]

    async def drain(ws):
        while not stop.is_set():
            try:
                msg = await ws.recv()
            except Exception:
                return
            if msg and msg.get("type") == "pong" and pending[ws] is not None:
                rtts.append(time.monotonic() - pending[ws])
                pending[ws] = None
            elif msg and msg.get("type") == "resync_directive":
                resyncs.append(1)

    async def beat(ws):
        while not stop.is_set():
            try:
                pending[ws] = time.monotonic()
                await ws.send_ping(ply[0])
            except Exception:
                return
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    sub = [
        asyncio.create_task(drain(white)), asyncio.create_task(drain(black)),
        asyncio.create_task(beat(white)), asyncio.create_task(beat(black)),
    ]
    for i, (frm, to) in enumerate(SCRIPT):
        if stop.is_set():
            break
        ws = white if i % 2 == 0 else black
        try:
            await ws.send_move(frm, to)
        except Exception:
            break
        ply[0] = i + 1
        await asyncio.sleep(MOVE_DELAY_SECONDS)
    await stop.wait()
    for ws in (white, black):
        await ws.close()
    for task in sub:
        task.cancel()


async def _run(args):
    transport = ServerTransport(args.addr)
    rtts = []
    resyncs = []
    stop = asyncio.Event()
    games = []
    errors = 0
    async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_SECONDS) as http:
        for i in range(args.rooms):
            try:
                white, black = await _setup_game(transport, http, i, args.minutes, args.increment)
                games.append(asyncio.create_task(_play_game(white, black, stop, rtts, resyncs)))
            except Exception as exc:
                errors += 1
                if args.verbose:
                    print(f"  setup error #{i}: {exc}")
            if (i + 1) % 50 == 0:
                health = await transport.healthz_async(http)
                active = health.rooms_active if health else "?"
                print(f"... {i + 1}/{args.rooms} set up, server rooms_active={active}")
        print(f"established {len(games)} games ({errors} setup errors); holding {args.hold}s")
        print("WATCH SERVER RESOURCES NOW:  docker stats")
        await asyncio.sleep(args.hold)
        stop.set()
        await asyncio.gather(*games, return_exceptions=True)
    if rtts:
        rtts.sort()
        p50 = statistics.median(rtts)
        p99 = rtts[min(len(rtts) - 1, int(len(rtts) * 0.99))]
        print(f"ping RTT  p50={p50 * 1000:.1f}ms  p99={p99 * 1000:.1f}ms  samples={len(rtts)}")
    else:
        print("no RTT samples collected")
    print(f"resync directives: {len(resyncs)}")


def main():
    parser = argparse.ArgumentParser(description="synthetic load driver (local servers only)")
    parser.add_argument("--addr", default="localhost:8000")
    parser.add_argument("--rooms", type=int, default=50)
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument("--increment", type=int, default=0)
    parser.add_argument("--hold", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(f"!! load test against {args.addr} -- LOCAL/THROWAWAY ONLY, never prod !!")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
