"""Import layering inside the server package: `app.py` is the assembly point and
nothing under `chessshootout/server/` may import it back.

app.py builds the room manager, the socket registry, the limiters and the sweep
and hands them to everybody else through `app.state`; handlers, rooms, sweep,
broadcasts and connections are the parts it assembles. An import in the other
direction is either a cycle waiting to happen or a shortcut past `app.state`, and
both get much easier to write accidentally once app.py is split into modules.

`__main__.py` is the one legitimate importer -- it is the process entry point,
which is above app.py rather than below it.

AST-based and relative-import aware for the same reason the pygame guard is:
`from .app import send`, `from . import app` and `from ..server import app` all
mean the same thing and a text scan sees three different lines. The probes at the
bottom are the negative control.
"""

import ast
import os

import pytest

import chessshootout
from tests.helpers import read_source_without_docstrings

PACKAGE_ROOT = os.path.dirname(os.path.abspath(chessshootout.__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
SERVER_ROOT = os.path.join(PACKAGE_ROOT, "server")

FORBIDDEN_ROOT = "chessshootout.server.app"
EXEMPT_MODULES = ("chessshootout.server.__main__",)


def _module_name(path):
    rel = os.path.relpath(path, REPO_ROOT)
    dotted = rel[:-3].replace(os.sep, ".")
    if dotted.endswith(".__init__"):
        dotted = dotted[:-len(".__init__")]
    return dotted


def _resolve(module_dotted, level, target):
    if level == 0:
        return target
    parts = module_dotted.split(".")
    base = parts[:-level]
    if target:
        base = base + [target]
    return ".".join(base)


def _is_forbidden(module):
    return module == FORBIDDEN_ROOT or module.startswith(FORBIDDEN_ROOT + ".")


def _forbidden_lines_in_source(source, source_module, filename="<probe>"):
    tree = ast.parse(source, filename=filename)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_forbidden(alias.name) for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(source_module, node.level or 0, node.module)
            if not resolved:
                continue
            if _is_forbidden(resolved) or any(
                    _is_forbidden(f"{resolved}.{alias.name}") for alias in node.names):
                lines.append(node.lineno)
    return lines


def test_nothing_under_server_imports_the_app_module_except_the_entry_point():
    offenders = []
    scanned = 0
    assert os.path.isdir(SERVER_ROOT), f"expected a walkable package dir at {SERVER_ROOT}"
    for root, _, files in os.walk(SERVER_ROOT):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            scanned += 1
            module = _module_name(path)
            if module in EXEMPT_MODULES:
                continue
            rel = os.path.relpath(path, REPO_ROOT)
            offenders += [f"{rel}:{lineno}" for lineno in
                          _forbidden_lines_in_source(
                              read_source_without_docstrings(path), module, filename=path)]
    assert scanned >= 8, f"only scanned {scanned} files, guard root is likely wrong"
    assert offenders == [], (
        "server modules are assembled BY app.py and must not import it back: "
        f"{offenders}"
    )


def test_the_entry_point_really_is_the_exempt_importer():
    """The exemption must stay earned. If __main__ ever stopped importing app.py
    the exemption would be dead weight quietly widening the guard."""
    main_path = os.path.join(SERVER_ROOT, "__main__.py")
    module = _module_name(main_path)
    assert module in EXEMPT_MODULES
    assert _forbidden_lines_in_source(
        read_source_without_docstrings(main_path), module, filename=main_path), \
        "__main__.py is exempt because it imports app.py; it no longer does"


@pytest.mark.parametrize("module, source", [
    pytest.param("chessshootout.server.handlers",
                 "from chessshootout.server.app import send\n", id="absolute_from"),
    pytest.param("chessshootout.server.handlers",
                 "import chessshootout.server.app\n", id="absolute_import"),
    pytest.param("chessshootout.server.handlers",
                 "import chessshootout.server.app as a\n", id="aliased_import"),
    pytest.param("chessshootout.server.handlers", "from .app import send\n",
                 id="relative_from"),
    pytest.param("chessshootout.server.handlers", "from . import app\n",
                 id="app_as_an_imported_name"),
    pytest.param("chessshootout.server.moderation.detector", "from .. import app\n",
                 id="relative_from_a_nested_module"),
])
def test_guard_catches_every_spelling_of_the_back_import(module, source):
    assert _forbidden_lines_in_source(source, module) == [1]


@pytest.mark.parametrize("module, source", [
    pytest.param("chessshootout.server.handlers", "from . import rooms\n",
                 id="relative_sibling"),
    pytest.param("chessshootout.server.handlers",
                 "from chessshootout.server.protocol import Reason\n", id="absolute_sibling"),
    pytest.param("chessshootout.server.handlers", "from fastapi import FastAPI\n",
                 id="third_party"),
    pytest.param("chessshootout.server.handlers", "app = 1\nfrom . import broadcasts\n",
                 id="app_as_a_local_name"),
    pytest.param("chessshootout.server.handlers",
                 "from chessshootout.server.app_helpers import x\n",
                 id="a_module_whose_name_merely_starts_with_app"),
])
def test_guard_leaves_the_legitimate_imports_alone(module, source):
    """A bare local name that happens to read `app`, and a sibling whose dotted
    name merely has `app` as a prefix, must both stay legal -- the predicate
    compares whole dotted segments, not string prefixes."""
    assert _forbidden_lines_in_source(source, module) == []
