import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from chessshootout import paths
from chessshootout.infra import env
from chessshootout.online.transport import TransportError, fetch_news


log = logging.getLogger("chess.client.news")

CACHE_FILENAME = "news_cache.json"

NEWS_MAX_ITEMS = 60

NEWS_TITLE_MAX_CHARS = 200
NEWS_BODY_MAX_CHARS = 4000
NEWS_DATE_MAX_CHARS = 32

NEWS_FIELD_MAX_CHARS = {
    "title": NEWS_TITLE_MAX_CHARS,
    "body": NEWS_BODY_MAX_CHARS,
    "date": NEWS_DATE_MAX_CHARS,
}

NEWS_DATE_FORMAT = "%Y-%m-%d"


def _cache_path() -> Path:
    """
    Where the last fetched feed is kept between runs, so the menu's News card
    has something to show before -- or instead of -- a successful fetch

    :returns: full path of the cache file inside the config directory.
    """
    return paths.get_config_dir() / CACHE_FILENAME


def _valid_str(value: object) -> bool:
    """
    Whether one feed field is text that can actually be shown. The feed comes
    from outside the app, so a missing, non-text or blank field disqualifies
    the whole item rather than reaching the screen empty

    :param value: raw field value straight out of the decoded feed.
    :returns: True when it is a string with something in it.
    """
    return isinstance(value, str) and value.strip() != ""


def _coerce_item(raw: object) -> dict[str, Any] | None:
    """
    Turn one entry from the feed into a news item fit to display, or refuse it.
    Title, body and date must all be real text; anything else the entry carries
    is kept as it came, and the three shown fields are clamped in length so no
    feed can push an unbounded string into the menu

    :param raw: one decoded entry from the feed, of any shape.
    :returns: the item with its shown fields clamped, or None when unusable.
    """
    if not isinstance(raw, dict):
        return None
    if not (_valid_str(raw.get("title")) and _valid_str(raw.get("body"))
            and _valid_str(raw.get("date"))):
        return None
    item = dict(raw)
    for key, limit in NEWS_FIELD_MAX_CHARS.items():
        item[key] = item[key][:limit]
    return item


def _sort_key(item: dict[str, Any]) -> datetime:
    """
    Order news items by date, treating a date that will not parse as the oldest
    one possible so a malformed entry sinks instead of heading the card

    :param item: a coerced news item, its date already known to be text.
    :returns: the parsed date, or the minimum date when it does not parse.
    """
    try:
        return datetime.strptime(item["date"], NEWS_DATE_FORMAT)
    except ValueError:
        return datetime.min


def parse_news_items(raw: object) -> list[dict[str, Any]]:
    """
    Turn a decoded feed into the list the News card shows: unusable entries
    dropped, newest first, capped at what the card will ever display. Every
    feed goes through here, whether it was fetched, cached, or read from the
    repo while developing

    :param raw: decoded feed, expected to be a list of entries.
    :returns: display-ready items, newest first, empty when none survived.
    """
    if not isinstance(raw, list):
        return []
    items = [item for entry in raw if (item := _coerce_item(entry)) is not None]
    items.sort(key=_sort_key, reverse=True)
    return items[:NEWS_MAX_ITEMS]


def format_news_date(date_str: str) -> str:
    """
    Render an item's date as the short stamp printed on the News card, such as
    JUL 14. A date that does not parse is shown exactly as it came, so a typo
    in the feed is visible rather than silently blank

    :param date_str: item date, normally in year-month-day form.
    :returns: the short display stamp, or the original text unchanged.
    """
    try:
        return datetime.strptime(date_str, NEWS_DATE_FORMAT).strftime("%b %d").upper()
    except ValueError:
        return date_str


def _atomic_write_json(path: Path, data: list[dict[str, Any]]) -> None:
    """
    Save freshly fetched items for the next run, written through a temporary
    file so an interrupted write cannot leave half a cache behind. Failing to
    write costs only the cache, so it is logged and otherwise shrugged off

    :param path: cache file to replace.
    :param data: the items to store, already parsed and clamped.
    """
    try:
        env.atomic_write_text(path, json.dumps(data))
    except OSError:
        log.debug("news cache write failed", exc_info=True)


def _load_cache(path: Path) -> list[dict[str, Any]]:
    """
    Read the items kept from the last run, which is what the News card shows
    the moment the menu appears. A missing or unreadable cache simply means
    there is no news yet

    :param path: cache file to read.
    :returns: the cached items, newest first, empty when there are none.
    """
    return _read_news_file(path) or []


def _read_news_file(path: Path) -> list[dict[str, Any]] | None:
    """
    Read one feed from disk and put it through the same gate a fetched feed
    passes. Shared by the cache and by the repo copy used while developing

    :param path: file holding a JSON feed.
    :returns: the parsed items, or None when the file is absent or unreadable.
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parse_news_items(raw)


def _dev_news_items() -> list[dict[str, Any]] | None:
    """
    Read the feed straight out of the repo when running from source, so a news
    item can be written and looked at without publishing anything. Packaged
    builds and the test suite never take this path

    :returns: the repo's items, or None when this is not a source run.
    """
    if paths.is_frozen() or "pytest" in sys.modules:
        return None
    return _read_news_file(paths.resource_path("news.json"))


class NewsClient:
    """
    Supplies the items behind the menu's News card. Whatever was cached last
    run is available immediately, the live feed is fetched once per run on a
    background thread, and the two are swapped over only when that fetch
    yields something usable; with no items at all the card is not shown
    """

    def __init__(self, url: str | None = None, cache_path: Path | None = None) -> None:
        """
        Build the client and load whatever can be shown right away. Left to its
        own devices it reads the repo feed when running from source and the
        cached feed otherwise; naming either argument, as tests do, keeps it on
        the fetching path

        :param url: feed address, or None to use CHESS_NEWS_URL or the default.
        :param cache_path: cache file to use, or None for the standard one.
        """
        dev = _dev_news_items() if url is None and cache_path is None else None
        self.url = url or env.get_news_url()
        self._cache_path = cache_path or _cache_path()
        self._lock = threading.Lock()
        self._items = dev if dev is not None else _load_cache(self._cache_path)
        self._dev = dev is not None
        self._generation = 0
        self._fetched = False

    def items(self) -> list[dict[str, Any]]:
        """
        The news items to show right now. A copy is taken under the lock, so
        the menu can read them while the fetch thread is replacing them

        :returns: display-ready items, newest first.
        """
        with self._lock:
            return list(self._items)

    def generation(self) -> int:
        """
        A counter that rises each time the items are replaced. The menu
        compares it against the one its News card was built from and rebuilds
        the card only when it has moved

        :returns: how many times the item list has been replaced.
        """
        with self._lock:
            return self._generation

    def fetch_once(self) -> None:
        """
        Start the single fetch this run gets, on a background thread so startup
        never waits on the network. Later calls do nothing, and a source run
        already showing the repo feed does not fetch at all
        """
        with self._lock:
            if self._fetched or self._dev:
                return
            self._fetched = True
        thread = threading.Thread(target=self._fetch_worker, daemon=True)
        thread.start()

    def _fetch_worker(self) -> None:
        """
        Fetch the feed on the background thread and, only when it yields real
        items, replace what is on show and save it for next time. Any failure
        leaves the existing items untouched, so the News card never empties
        because one fetch went wrong
        """
        try:
            raw = fetch_news(self.url)
        except TransportError as exc:
            log.debug("news fetch failed url=%s: %s", self.url, exc)
            return
        items = parse_news_items(raw)
        if not items:
            log.debug("news fetch returned no usable items url=%s", self.url)
            return
        with self._lock:
            self._items = items
            self._generation += 1
        _atomic_write_json(self._cache_path, items)
