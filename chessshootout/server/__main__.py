import argparse
import os

import uvicorn
from fastapi import FastAPI

from chessshootout.server import logging_setup
from chessshootout.server.app import create_app, DEFAULT_MAX_ROOMS
from chessshootout.server.ws_session import MAX_INBOUND_MESSAGE_BYTES
from chessshootout.infra.log_format import uvicorn_log_config


def _main() -> None:
    """
    Command-line entry point that runs the game server. It reads the address,
    port and room cap from the command line or the matching HOST, PORT and
    MAX_ROOMS environment variables, sets logging up and then serves the app
    under uvicorn. Websocket pings are switched off because the protocol carries
    its own heartbeat, and uvicorn is handed the same inbound frame ceiling the
    app enforces so an oversized frame is refused at both layers
    """
    parser = argparse.ArgumentParser(description="chess multiplayer server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true",
                        help="dev: auto-restart on source change")
    parser.add_argument("--max-rooms", type=int,
                        default=int(os.environ.get("MAX_ROOMS", str(DEFAULT_MAX_ROOMS))))
    args = parser.parse_args()
    logging_setup.configure()
    log_config = uvicorn_log_config()
    if args.reload:
        os.environ.setdefault("CHESS_MAX_ROOMS", str(args.max_rooms))
        uvicorn.run("chessshootout.server.__main__:_app_factory", host=args.host, port=args.port,
                    reload=True, factory=True,
                    ws_ping_interval=None, ws_max_size=MAX_INBOUND_MESSAGE_BYTES,
                    log_config=log_config)
    else:
        app = create_app(max_rooms=args.max_rooms)
        uvicorn.run(app, host=args.host, port=args.port,
                    ws_ping_interval=None, ws_max_size=MAX_INBOUND_MESSAGE_BYTES,
                    log_config=log_config)


def _app_factory() -> FastAPI:
    """
    Build a fresh application for uvicorn's dev reload mode, which re-imports
    this module in a worker process rather than being handed an object. The room
    cap travels through the environment because that is all that survives the
    reload

    :returns: a newly created application, ready to serve
    """
    return create_app(max_rooms=int(os.environ.get("CHESS_MAX_ROOMS", str(DEFAULT_MAX_ROOMS))))


if __name__ == "__main__":
    _main()
