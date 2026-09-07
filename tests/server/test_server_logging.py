"""Structured logging surface: RotatingFileHandler wiring + KV log format.

Server log lines follow one shape: a free-form lowercase verb prefix, then
`key=value` pairs (`move applied room=… mover=… san=…`). That shape is what
makes the journal greppable and is documented in CONTRIBUTING.md, but nothing
enforces it — a line written the other way round reads fine in review and is
invisible to every other test.

This file pins the rotating handler's path/rotation policy, and then checks the
shape PER MODULE rather than against one joined blob: a blob only proves that
somebody, somewhere, still writes `room=%s`, which stays true no matter how many
new lines drift off the format.
"""
import ast
import importlib
import inspect
import logging
import logging.handlers
import os
import re

import pytest

import chessshootout
from chessshootout.server import logging_setup
from tests.helpers import read_source_without_docstrings

PACKAGE_ROOT = os.path.dirname(os.path.abspath(chessshootout.__file__))
SERVER_ROOT = os.path.join(PACKAGE_ROOT, "server")

SERVER_LOG_MODULES = ("app", "broadcasts", "connections", "handlers", "limits",
                      "protocol", "routes_http", "sweep", "ws_session")
LOUD_LEVELS = ("info", "warning")
LOUD_LOGGER_NAMES = ("log", "logger")
KV_TOKEN_RE = re.compile(r"(?:^|[\s(])([A-Za-z_][A-Za-z0-9_]*)=(\S+)")

PREFIX_ONLY_TEMPLATES = {
    ("connections", "ws send failed: %s"),
    ("limits", "trusted proxies %s"),
}


@pytest.fixture
def isolated_root_logger():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_attach_rotating_file_handler_writes_to_path(tmp_path,
                                                       isolated_root_logger):
    path = tmp_path / "server.log"
    logging_setup.attach_rotating_file_handler(str(path), level="INFO")
    isolated_root_logger.setLevel(logging.DEBUG)
    log = logging.getLogger("chess.server.test_logging")
    log.setLevel(logging.DEBUG)
    log.info("matchmake ok room=room-x uuid=abcd1234")
    for h in isolated_root_logger.handlers:
        h.flush()
    contents = path.read_text(encoding="utf-8")
    assert "matchmake ok" in contents
    assert "room=room-x" in contents


def test_attach_rotating_file_handler_returns_rotating_handler(tmp_path,
                                                                 isolated_root_logger):
    path = tmp_path / "server.log"
    handler = logging_setup.attach_rotating_file_handler(str(path), level="INFO")
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler in isolated_root_logger.handlers


def test_attach_rotating_file_handler_rotates_and_caps_backups(tmp_path,
                                                                isolated_root_logger):
    """Real rotation: file rolls at max_bytes and never keeps more than
    backup_count backups (no .{backup_count + 1})."""
    path = tmp_path / "server.log"
    max_bytes, backup_count = 200, 2
    handler = logging_setup.attach_rotating_file_handler(
        str(path), level="DEBUG", max_bytes=max_bytes, backup_count=backup_count,
    )
    isolated_root_logger.setLevel(logging.DEBUG)
    log = logging.getLogger("chess.server.rotation")
    log.setLevel(logging.DEBUG)
    for i in range(40):
        log.debug("line %02d room=r-%02d uuid=u-%02d padding=xxxxxxxxxxxxxxxxxxxx",
                  i, i, i)
    handler.flush()

    assert path.exists()
    assert (tmp_path / "server.log.1").exists()
    assert (tmp_path / "server.log.2").exists()
    assert not (tmp_path / "server.log.3").exists()

    rolled = list(tmp_path.iterdir())
    assert len(rolled) == backup_count + 1
    for f in rolled:
        assert f.stat().st_size <= max_bytes


@pytest.mark.parametrize(
    "level, max_bytes, backup_count, expected_level",
    [
        pytest.param("INFO", 1024, 2, logging.INFO, id="info_string_level"),
        pytest.param("DEBUG", 4096, 5, logging.DEBUG, id="debug_string_level"),
        pytest.param(logging.WARNING, 512, 0, logging.WARNING, id="int_level_zero_backups"),
    ],
)
def test_attach_rotating_file_handler_applies_config(tmp_path, isolated_root_logger,
                                                     level, max_bytes, backup_count,
                                                     expected_level):
    path = tmp_path / "server.log"
    handler = logging_setup.attach_rotating_file_handler(
        str(path), level=level, max_bytes=max_bytes, backup_count=backup_count,
    )
    assert handler.maxBytes == max_bytes
    assert handler.backupCount == backup_count
    assert handler.level == expected_level
    assert os.path.abspath(handler.baseFilename) == os.path.abspath(str(path))


@pytest.mark.parametrize(
    "set_env, expected_count",
    [
        pytest.param(False, 0, id="env_unset_skips_file_handler"),
        pytest.param(True, 1, id="env_set_attaches_file_handler"),
    ],
)
def test_configure_file_handler_follows_log_file_env(tmp_path, monkeypatch,
                                                     isolated_root_logger,
                                                     set_env, expected_count):
    path = tmp_path / "from-env.log"
    if set_env:
        monkeypatch.setenv("LOG_FILE", str(path))
    else:
        monkeypatch.delenv("LOG_FILE", raising=False)
    logging_setup.configure(level="INFO")
    rotating = [h for h in isolated_root_logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(rotating) == expected_count
    if expected_count:
        assert os.path.abspath(rotating[0].baseFilename) == os.path.abspath(str(path))


def _loud_templates_in_source(path):
    """Every INFO/WARNING format string in one file, read off the AST so a
    literal inside a docstring or a data table can never be mistaken for one.

    A loud call whose first argument is not a string literal -- an f-string, or a
    message built before the call -- RAISES here rather than being skipped. A
    skip would drop the line out of the shape check silently, which is the exact
    way a line escapes this file's guard."""
    tree = ast.parse(read_source_without_docstrings(path), filename=path)
    templates = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in LOUD_LEVELS:
            continue
        if not (isinstance(node.func.value, ast.Name)
                and node.func.value.id in LOUD_LOGGER_NAMES):
            continue
        first = node.args[0] if node.args else None
        assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
            f"{path}:{node.lineno}: a loud log call takes a literal % template, "
            "never an f-string or a pre-built message"
        )
        templates.append(first.value)
    return templates


def _loud_templates(module_name):
    """The loud templates of one server module, and never an empty list: a module
    that logs through a differently named logger, or only through f-strings,
    would otherwise pass every check below by having nothing to check."""
    module = importlib.import_module(f"chessshootout.server.{module_name}")
    assert module.__file__ is not None
    templates = _loud_templates_in_source(module.__file__)
    assert templates, f"server/{module_name}.py emits no INFO/WARNING lines any more"
    return templates


@pytest.mark.parametrize("module_name", SERVER_LOG_MODULES)
def test_server_info_and_warning_lines_keep_the_kv_shape(module_name):
    """Per module, every operator-facing line: a lowercase verb prefix carrying
    no `=`, then `key=value` pairs. The prefix rule is what keeps a line
    greppable by its subject (`grep 'matchmake rejected'`) and the pairs are what
    keep it parseable by field."""
    for template in _loud_templates(module_name):
        first = KV_TOKEN_RE.search(template)
        if first is None:
            assert (module_name, template) in PREFIX_ONLY_TEMPLATES, (
                f"server/{module_name}.py: {template!r} carries no key=value pair"
            )
            continue
        prefix = template[:first.start()]
        assert prefix.strip(), f"{template!r} starts with a field, not a verb"
        assert "=" not in prefix, f"{template!r} has a key=value pair inside its prefix"
        assert prefix[0].islower(), f"{template!r} does not open in the house voice"
        for key, value in KV_TOKEN_RE.findall(template):
            assert value, f"{template!r} has an empty value for {key}"


def test_the_prefix_only_log_lines_still_exist():
    """The two lines with nothing to key on (one names a config value, one is a
    bare send failure) are listed by hand, so the list has to stay earned --
    otherwise a reworded line would leave a stale exemption widening the guard."""
    for module_name, template in sorted(PREFIX_ONLY_TEMPLATES):
        assert template in _loud_templates(module_name), (
            f"server/{module_name}.py no longer emits {template!r}"
        )


def test_every_server_module_that_logs_loudly_is_covered():
    """SERVER_LOG_MODULES is a hand-written tuple, so a new module with its own
    logger would silently escape the shape check above."""
    assert os.path.isdir(SERVER_ROOT), f"expected a walkable package dir at {SERVER_ROOT}"
    scanned, uncovered = 0, {}
    for root, _, files in os.walk(SERVER_ROOT):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            scanned += 1
            if name[:-3] in SERVER_LOG_MODULES:
                continue
            templates = _loud_templates_in_source(path)
            if templates:
                uncovered[name] = templates
    assert scanned >= 8, f"only scanned {scanned} files, guard root is likely wrong"
    assert uncovered == {}, f"add these modules to SERVER_LOG_MODULES: {sorted(uncovered)}"
    total = sum(len(_loud_templates(n)) for n in SERVER_LOG_MODULES)
    assert total >= 40, f"only {total} INFO/WARNING lines found, the scan is broken"


def _get_logger_calls(path):
    """Every `get_logger(...)` call node in one file, read off the AST."""
    tree = ast.parse(read_source_without_docstrings(path), filename=path)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "get_logger":
            calls.append(node)
    return calls


def test_get_logger_takes_a_name_and_has_no_default():
    """The default was `chess.server`, and nothing ever asked for it -- every
    server module spells `chess.server.app` out. A default is a second logger name
    waiting for the first caller who forgets: its lines would then miss every
    caplog filter in the suite (and the operator's `journalctl` grep) while still
    looking perfectly logged from the call site."""
    params = list(inspect.signature(logging_setup.get_logger).parameters.values())
    assert [p.name for p in params] == ["name"]
    assert params[0].default is inspect.Parameter.empty


def test_every_server_module_names_its_logger_with_a_literal():
    """The one logger name for the whole server package, asserted at the call
    sites rather than at the definition: a computed name (`__name__`, a variable,
    an f-string) would still type-check and still log, but it would split the
    package's output across several loggers and quietly break the caplog filters
    the observability tests are built on."""
    assert os.path.isdir(SERVER_ROOT), f"expected a walkable package dir at {SERVER_ROOT}"
    scanned, names = 0, []
    for root, _, files in os.walk(SERVER_ROOT):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            scanned += 1
            for node in _get_logger_calls(path):
                where = f"{os.path.relpath(path, PACKAGE_ROOT)}:{node.lineno}"
                assert len(node.args) == 1 and not node.keywords, \
                    f"{where}: get_logger takes exactly the logger name"
                arg = node.args[0]
                assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), \
                    f"{where}: the logger name must be a literal, not {ast.dump(arg)}"
                names.append(arg.value)
    assert scanned >= 8, f"only scanned {scanned} files, guard root is likely wrong"
    assert names, "no get_logger call found at all, the scan is broken"
    assert set(names) == {"chess.server.app"}, f"the server logger split: {sorted(set(names))}"
