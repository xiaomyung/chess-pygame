"""Docstring + annotation guard over the whole production tree.

Since v2.12.2 every def (dunders, properties, nested closures) and every class
in chessshootout/ -- plus the shared test-infra modules -- must carry a reST
docstring in the house shape and a fully annotated signature. These tests are
the enforcement: same walking pattern as the pygame guard in
tests/infra/test_server_no_pygame.py and the transport-import guard in
tests/online/test_server_transport.py (root derived from the package, scanned
floors, collect-then-assert-once).

House shape, pinned here: opening triple quote on its own line; a 2-3 sentence
summary whose last sentence has no trailing period; a blank line before the
field block; one `:param name:` per parameter in exact signature order (self
and cls omitted, `*args`/`**kwargs` under their plain names); `:returns:`
present exactly when the return annotation is neither None nor NoReturn; no
other field tags; physical lines at most 92 columns; ASCII plus the six
allowlisted glyphs. Class docstrings carry no field lists -- constructor
params are documented on __init__. Lambdas are exempt (they cannot hold a
docstring); `@overload` stubs are exempt from the docstring rules only.
"""

import ast
import re
from pathlib import Path

import chessshootout

PACKAGE_ROOT = Path(chessshootout.__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
TEST_INFRA_FILES = (
    REPO_ROOT / "tests" / "helpers.py",
    REPO_ROOT / "tests" / "conftest.py",
    REPO_ROOT / "tests" / "server" / "conftest.py",
    REPO_ROOT / "tests" / "frontend" / "focus_helpers.py",
    REPO_ROOT / "tests" / "online" / "online_helpers.py",
)
MAX_DOC_LINE_WIDTH = 92
ALLOWED_NON_ASCII = set("→←↑↓✓✗")
FIELD_RE = re.compile(r"^:(param|returns)\b")
PARAM_RE = re.compile(r"^:param ([A-Za-z_][A-Za-z0-9_]*):")
RETURNS_RE = re.compile(r"^:returns:")
ANY_TAG_RE = re.compile(r"^:[A-Za-z]+")

_LOADED: dict[Path, tuple[list[str], ast.Module]] = {}


def _target_files():
    assert PACKAGE_ROOT.is_dir(), f"expected a walkable package dir at {PACKAGE_ROOT}"
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    for extra in TEST_INFRA_FILES:
        assert extra.is_file(), f"expected shared test-infra module at {extra}"
    files.extend(TEST_INFRA_FILES)
    assert len(files) > 100, f"only found {len(files)} files, guard root is likely wrong"
    return files


def _load(path):
    if path not in _LOADED:
        source = path.read_text(encoding="utf-8")
        _LOADED[path] = (source.splitlines(), ast.parse(source, filename=str(path)))
    return _LOADED[path]


def _walk_defs(tree):
    stack = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{child.name}"
                yield child, qual
                stack.append((child, qual + "."))
            else:
                stack.append((child, prefix))


def _is_overload(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in node.decorator_list)


def _param_args(node):
    args = list(node.args.posonlyargs) + list(node.args.args)
    if args and args[0].arg in ("self", "cls"):
        args = args[1:]
    args += [a for a in (node.args.vararg,) if a is not None]
    args += list(node.args.kwonlyargs)
    args += [a for a in (node.args.kwarg,) if a is not None]
    return args


def _needs_returns(node):
    ret = node.returns
    if ret is None or (isinstance(ret, ast.Constant) and ret.value is None):
        return False
    name = ret.id if isinstance(ret, ast.Name) else (
        ret.attr if isinstance(ret, ast.Attribute) else "")
    return name not in ("NoReturn", "Never")


def _rel(path):
    return str(path.relative_to(REPO_ROOT))


def test_every_production_def_and_class_has_a_docstring():
    offenders = []
    files = _target_files()
    for path in files:
        _, tree = _load(path)
        for node, qual in _walk_defs(tree):
            if _is_overload(node):
                continue
            if not ast.get_docstring(node):
                offenders.append(f"{_rel(path)}:{node.lineno} {qual}")
    assert len(files) > 100
    assert offenders == [], (
        "every def and class needs a reST docstring in the house shape "
        f"(see CONTRIBUTING.md, Docstrings and types): {offenders}")


def test_every_production_signature_is_fully_annotated():
    offenders = []
    files = _target_files()
    for path in files:
        _, tree = _load(path)
        for node, qual in _walk_defs(tree):
            if isinstance(node, ast.ClassDef):
                continue
            for arg in _param_args(node):
                if arg.annotation is None:
                    offenders.append(
                        f"{_rel(path)}:{node.lineno} {qual} param '{arg.arg}' unannotated")
            if node.returns is None:
                offenders.append(
                    f"{_rel(path)}:{node.lineno} {qual} missing return annotation")
    assert len(files) > 100
    assert offenders == [], (
        "every signature is fully annotated -- every parameter and an explicit "
        f"return type, -> None included: {offenders}")


def test_docstring_fields_match_the_signature():
    offenders = []
    files = _target_files()
    for path in files:
        _, tree = _load(path)
        for node, qual in _walk_defs(tree):
            if isinstance(node, ast.ClassDef) or _is_overload(node):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc:
                continue
            doc_lines = doc.splitlines()
            field_lines = [ln for ln in doc_lines if FIELD_RE.match(ln)]
            doc_params = [m.group(1) for ln in field_lines
                          for m in [PARAM_RE.match(ln)] if m]
            sig_params = [a.arg for a in _param_args(node)]
            where = f"{_rel(path)}:{node.lineno} {qual}"
            if doc_params != sig_params:
                offenders.append(
                    f"{where} :param lines {doc_params} != signature {sig_params}")
            has_returns = any(RETURNS_RE.match(ln) for ln in field_lines)
            if _needs_returns(node) and not has_returns:
                offenders.append(f"{where} missing :returns: for a non-None return")
            if not _needs_returns(node) and has_returns:
                offenders.append(f"{where} :returns: on a None/NoReturn function")
    assert len(files) > 100
    assert offenders == [], (
        "docstring fields mirror the signature -- :param per parameter in "
        f"signature order, :returns: iff the return is not None: {offenders}")


def test_docstring_shape_summary_break_width_and_charset():
    offenders = []
    files = _target_files()
    for path in files:
        lines, tree = _load(path)
        for node, qual in _walk_defs(tree):
            doc = ast.get_docstring(node, clean=True)
            if not doc:
                continue
            where = f"{_rel(path)}:{node.lineno} {qual}"
            body0 = node.body[0]
            raw = lines[body0.lineno - 1:body0.end_lineno]
            if raw[0].strip() not in ('"""', "'''"):
                offenders.append(f"{where} text on the opening quote line")
            for i, ln in enumerate(raw):
                if len(ln) > MAX_DOC_LINE_WIDTH:
                    offenders.append(
                        f"{where} line {body0.lineno + i} over {MAX_DOC_LINE_WIDTH} cols")
            bad = {c for c in doc if ord(c) > 127 and c not in ALLOWED_NON_ASCII}
            if bad:
                offenders.append(f"{where} disallowed non-ASCII {sorted(bad)!r}")
            doc_lines = doc.splitlines()
            field_idx = next(
                (i for i, ln in enumerate(doc_lines) if FIELD_RE.match(ln)), None)
            summary = doc_lines if field_idx is None else doc_lines[:field_idx]
            stripped = [ln for ln in summary if ln.strip()]
            if not stripped:
                offenders.append(f"{where} empty summary")
            elif stripped[-1].rstrip().endswith("."):
                offenders.append(f"{where} trailing period on the summary")
            if field_idx is not None and (field_idx == 0 or summary[-1].strip()):
                offenders.append(f"{where} missing blank line before the field block")
            for ln in doc_lines:
                if ANY_TAG_RE.match(ln) and not FIELD_RE.match(ln):
                    offenders.append(f"{where} disallowed field tag {ln.strip()[:40]!r}")
            if isinstance(node, ast.ClassDef) and field_idx is not None:
                offenders.append(f"{where} class docstring carries a field list")
    assert len(files) > 100
    assert offenders == [], (
        "docstring shape: summary, blank line, fields only, no trailing period, "
        f"<= {MAX_DOC_LINE_WIDTH} cols, ASCII plus {''.join(sorted(ALLOWED_NON_ASCII))}: "
        f"{offenders}")
