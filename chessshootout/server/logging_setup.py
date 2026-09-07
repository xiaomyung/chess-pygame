import logging
import logging.handlers
import os

from chessshootout.infra.log_format import configure_basic, make_formatter

ROTATING_MAX_BYTES = 5 * 1024 * 1024
ROTATING_BACKUP_COUNT = 3


def configure(level: str | None = None, log_file: str | None = None) -> None:
    """
    Set the server's logging up once at startup: console output in the shared
    project format, plus a rotating file whenever one is asked for. Both
    settings fall back to the LOG_LEVEL and LOG_FILE environment variables,
    which is how the deployed container configures them without a rebuild

    :param level: level name in any case, e.g. info or DEBUG; None reads the
        environment and defaults to INFO
    :param log_file: path to also write to; None reads the environment, and
        with nothing there no file is attached at all
    """
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    configure_basic(level)
    log_file = log_file or os.environ.get("LOG_FILE")
    if log_file:
        attach_rotating_file_handler(log_file, level=level)


def attach_rotating_file_handler(
    path: str, *, level: str | int = "INFO",
    max_bytes: int = ROTATING_MAX_BYTES,
    backup_count: int = ROTATING_BACKUP_COUNT,
) -> logging.handlers.RotatingFileHandler:
    """
    Give the root logger a size-capped log file, so a server that runs for weeks
    keeps its recent history without ever filling the disk. The handler is
    returned rather than just attached, so a caller can flush, inspect or detach
    exactly the one it added

    :param path: file to write to; its directory must already exist
    :param level: minimum level for this handler, a level name or a logging
        constant
    :param max_bytes: size the current file may reach before it is rotated
    :param backup_count: how many rotated files are kept before the oldest one
        is deleted
    :returns: the handler that was attached to the root logger
    """
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    handler.setLevel(level if isinstance(level, int) else level.upper())
    handler.setFormatter(make_formatter())
    logging.getLogger().addHandler(handler)
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Give the named logger that server code writes through, instead of every
    module reaching for the logging package itself. The name is always spelled
    out at the call site: the server modules all ask for the same one, which
    keeps their output under a single readable prefix

    :param name: dotted logger name, e.g. chess.server.app
    :returns: the standard library logger for that name
    """
    return logging.getLogger(name)
