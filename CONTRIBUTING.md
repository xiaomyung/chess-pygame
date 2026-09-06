# Contributing to Chess Shootout

Thanks for your interest. A few ground rules before you open a pull request.

## What this project is

Chess Shootout is **source available, non-commercial** — not open source. The
code is under the [PolyForm Noncommercial License 1.0.0](LICENSE) and the
original assets under CC BY-NC 4.0. It may not be sold or used commercially by
anyone other than the author. See the README for the full picture.

## Contributor terms

By submitting a contribution (code, art, sound, docs, or anything else) you
agree to all of the following:

1. **You have the right to contribute it.** You wrote it yourself, or you
   otherwise have the right to submit it, and to your knowledge it does not
   infringe anyone's rights.
2. **You grant the author broad rights.** You keep your copyright, but you grant
   Xiao Myung a perpetual, worldwide, irrevocable, royalty-free, sublicensable,
   and transferable license to use, reproduce, modify, distribute, **relicense,
   and commercially exploit** your contribution as part of Chess Shootout or any
   successor work — including under different license terms and in paid or
   commercial releases.
3. **You won't assert claims that limit that use.** You waive, to the extent the
   law allows, any moral rights or other claims that would prevent the author
   from using your contribution in the ways above.
4. **No ownership stake, no payment.** You receive no ownership share and no
   compensation. Accepted contributors are credited in [CREDITS.md](CREDITS.md).

If you do not agree to these terms, please do not submit a contribution. To
record your agreement, state in your pull request that you have read and accept
this file.

## Docstrings and types

Every def -- dunders, properties, nested closures -- and every class in
`chessshootout/` (plus the shared test infra: `tests/helpers.py`, the root and
server `conftest.py`, `tests/frontend/focus_helpers.py`) carries a reST
docstring and a fully annotated signature. `tests/infra/test_docstring_guard.py`
enforces the shape and `mypy` (strict, gated in CI) enforces the types, so a
regression fails the build.

The house shape:

```python
def my_fun(x: int, z: int | str, w: str | None = None) -> int:
    """
    Two or three plain-language sentences. Lead with what the function does
    for the game or server -- its role in the project -- then the technical
    contract. No trailing period on the last summary sentence

    :param x: what it means, 1-2 sentences with units or ownership
    :param z: same
    :param w: same; say what None means
    :returns: description only, no type; omit the line when returning None
    """
```

- Opening `"""` on its own line; blank line between summary and fields; one
  `:param name:` per parameter in exact signature order (skip `self`/`cls`,
  `*args`/`**kwargs` under their plain names); `:returns:` present exactly when
  the return annotation is neither `None` nor `NoReturn`; no `:raises:` or
  other tags. Class docstrings carry no field lists -- constructor parameters
  are documented on `__init__`.
- Wrap docstring prose at ~80 columns (hard guard ceiling 92); ASCII plus at
  most the arrows and check glyphs already used in the UI. No em dashes -- use
  `--`.
- Summaries must be immediately understandable and non-tautological:
  ":param x: the x" is a review reject.
- Annotations use PEP 604 unions (`int | str`, `X | None`) and builtin
  generics; every parameter and an explicit return type on every def. Do not
  add `from __future__ import annotations`.
- **When you touch a function, re-verify its docstring and typehints still
  match its behavior** -- a stale docstring is treated as a bug.
- Comments remain banned in production code with exactly one exception:
  `# type: ignore[code]`, always error-code-qualified (CI errors on bare
  ignores), permitted only where a third-party library's typing forces it --
  never to silence an error in our own code.

## Logging

Most modules get their own `log = logging.getLogger(__name__)` (or the matching
`chess.*` name already used elsewhere in that package — grep a sibling file
before adding a new logger name). The server is the deliberate exception: every
module under `chessshootout/server/` logs through
`logging_setup.get_logger("chess.server.app")`, one name for the whole package.
Nothing there wants per-module level control, and the server tests filter
`caplog` by that single name — a per-module logger would make its lines
invisible to them. Pick the level by what the line is *for*, not by habit:

- **INFO** — a user action or a state transition: a game starts or ends, a
  move gets undone, an offer is sent/received/resolved, a setting is
  persisted, a room gets created/paired, a result gets finalized. One line
  per event, with the handful of fields that let you reconstruct the story
  (never a value the app also treats as identity or a secret — see below).
- **DEBUG** — per-message wire dispatch and other detail that's only useful
  shoulder-to-shoulder with the code (`ws dispatch`, `ws send`, `ws recv`).
- **WARNING** — degraded but continuing: a save fell back to a second
  directory, a resync kicked in, a reconnect attempt failed and is retrying.
- **ERROR** (`log.exception` inside an `except` block) — a failure the app
  is not going to recover from on its own; always with a traceback.

Never log a skill-check's `room.skillcheck_secret` (the which-fires selector)
or a per-check geometry `seed`, on either the client or the server — logging
either would hand a spectator or opponent the information the anti-cheat
model is built to keep from them. That extends to anything the seed derives:
the whack hole squares and pop schedule, and a combo's prompt sequence, are
the same secret in another shape. Same spirit for nicknames and client
UUIDs: log the *setting key* that changed, never the value.

Nothing logs on a per-frame cadence — no log call belongs directly inside
`draw_frame`, `draw_board`, or any other function that runs every tick,
whether or not something changed. If a per-frame function needs to report a
failure, wrap it in a state flag so it logs once, or emit at the point the
condition changes rather than at the point it's polled.

The server reads `LOG_LEVEL` (default `INFO`) and, when set, `LOG_FILE` (a
rotating file handler). The desktop client keeps every level-DEBUG-and-up
record in an in-memory ring buffer (`infra/crash_log.py`) regardless of what
the console shows, so a crash report is enriched with the INFO breadcrumbs
that led up to it even though nothing was written to disk along the way.

## Versioning

Every pull request must bump `[project].version` in `pyproject.toml` — a CI
job blocks merges where it's unchanged from the base branch. Run `make bump`
(`uv version --bump patch`) to bump the patch version and update `uv.lock`'s
`chess-shootout` entry together in one atomic step. A version-only bump can
also be done by hand: edit both files' version strings to match; no `uv`
required.

`uv.lock` is not a convenience file — every CI job (and every published
binary) installs from it with `uv sync --frozen`, and a `lock-check` job runs
`uv lock --check` to prove it still matches `pyproject.toml`. If your change
touches a dependency, run `uv lock` and commit the result in the same pull
request, or the check fails and nothing installs the version you tested
against.
