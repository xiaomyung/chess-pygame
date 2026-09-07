# Testing conventions

How tests are written in this repo. The goal is a suite that **exercises real
behavior and surfaces regressions** — not one that merely goes green. Every test
should be able to fail for a real reason.

## Running

```bash
.venv/bin/pytest tests -n 8 -q                 # fast local loop
.venv/bin/pytest tests -n auto -q              # match CI before pushing
.venv/bin/pytest tests/backend/test_san.py -v  # one file
.venv/bin/pytest "tests/backend/test_clock.py::test_add_time_caps[at_cap_returns_zero]" -v   # one case
.venv/bin/pytest tests -n auto --cov --cov-report=term:skip-covered          # coverage (informational)
.venv/bin/pylama chessshootout tests                                        # lint (≤100 cols)
```

Do **not** run the suite while a chess server is live on `:8000` (the per-process
reconnect probe floods its rate limit). `perft(3)` must read `8902` — it is the
strongest single regression for the move engine.

## Directory layout

Tests mirror the source layout, five dirs under `tests/`:

- `tests/backend/` — the pure chess engine (`chessshootout/backend/`) and the
  pygame-free `chessshootout/skillcheck/` package (the four capture-check
  engines — wheel/aim geometry plus whack (`test_skillcheck_mole.py`) and combo
  (`test_skillcheck_combo.py`) — triggers, weights, the seeded `rng`
  (`test_skillcheck_rng.py`, which pins the chained blocks that let
  `seeded_floats` serve more than four words), coordinator, the cross-side
  `online` adjudication surface — a peer of `backend/`, imported by both server
  and frontend, not to be confused with the client-side `chessshootout/online`
  package below).
- `tests/frontend/` — pygame UI: the `Frontend` shell and its four screens
  (`menu`/`game`/`history`/`review`, see "Screens" below), board, panels,
  modals, visual/, the `frontend/skillcheck/` view layer (wheel/aim plus the
  whack (`test_skillcheck_mole_view.py`) and combo
  (`test_skillcheck_combo_view.py`) views and the shared juice/particle helpers
  (`test_skillcheck_juice.py`), with `test_skillcheck_session_kinds.py` driving
  all four kinds through a session and `test_effects_whack_gun.py` covering the
  held-gun choreography a whack check drives on the board), focus mode, audio
  dispatch. Flat — no `tests/frontend/board/` subdirs.
- `tests/server/` — the `chessshootout/server/` package (FastAPI app,
  handlers, rooms, sweep, protocol — the wire models' own field bounds live in
  `test_server_protocol.py`, the heartbeat transit window and the two-strike
  resync model in `test_server_ping_tolerance.py` — and `moderation/`: the
  symbol detector plus its library and timing pin in `test_moderation_flow.py` /
  `test_moderation_guards.py`, including the per-room and per-player CPU meter
  that force-stops sharing once a client outspends its budget). The modules
  split out of `app.py` (`limits.py`, `routes_http.py`, `ws_session.py`) kept
  their coverage where it already was — `test_server_app.py`,
  `test_server_validation.py`, `test_rate_limit_client_ip.py` — with the imports
  and monkeypatch targets moved to the new homes.
- `tests/online/` — client-side online multiplayer: the top-level
  `chessshootout/online/` package (`OnlineClient`, `ServerTransport`) plus
  `chessshootout/frontend/online_coordinator.py` and the offer banners.
- `tests/infra/` — cross-cutting app lifecycle/config: `paths`, `env`,
  `countries`, `icons`, `log_format`, `crash_log`, `migration`, `utf8`, plus
  whole-repo static guards (`imports`, `logging_hygiene`,
  `server_no_pygame`, `server_layering`, `docstring_guard`) that scan the source
  tree rather than exercise one module.

**Where does a new test go?** By the primary module under test — what the
asserts verify, not an incidental import. A file that drives a real
`Frontend`/`Board` belongs in `frontend/` even if it also touches
`backend`/`domain` setup; a file that is pygame-free and asserts on
`chessshootout.online.*`/`chessshootout.server.*` belongs in `online/`/
`server/` respectively; a whole-repo static scan (import-boundary guard,
pygame-free guard) belongs in `infra/`.

Stays at `tests/` root (their `__file__`-relative paths depend on it, or they
are meta-guards that scan every dir): `conftest.py`, `helpers.py`,
`test_no_weak_tests.py`, `README.md`, `__init__.py`. Each of the five
subdirs has its own `__init__.py`; `tests/server/conftest.py` additionally
holds the server `clock`/`app`/`client` fixture trio + `ALICE`/`BOB` +
`auth_msg` (server-only — `client`/`app` there are FastAPI-flavored, distinct
from a `Frontend` a client-side test might build under the same names).

A guard that walks the source tree by path (`test_only_transport_module_...`,
`test_server_and_backend_never_import_pygame_or_frontend`, the emoji/log
hygiene scans) must anchor its root off an **imported package's `__file__`**
(`Path(chessshootout.__file__).resolve().parent[.parent]`), never off its own
`__file__` — the nesting depth from `tests/<dir>/test_x.py` to the repo root
differs from the old flat `tests/test_x.py`, and a stale two-`dirname()` walk
silently scans zero files instead of failing loudly. Each such guard also
asserts a minimum scanned-file count, so a future path break fails loud
instead of passing green over an empty walk.

## Screens

`Frontend` is a thin shell around four independent screens
(`chessshootout/frontend/screens/{menu,game,history,review}.py`), each
implementing a common `Screen` contract (`enter`/`exit`/`draw`/`handle_*`/
`escape`/`modals`/...). Navigation is a `Nav(name, payload)` intent run
through `Frontend.switch_to`, which calls a real `exit()` then `enter()` —
no test should hand-roll a screen swap by poking `app.screen` directly.

- `tests/frontend/test_screen_guards.py` — static AST guards on the shell:
  no screen imports a sibling screen, `online_coordinator.py` imports
  nothing from `screens` except `screens.base` (where `Nav` lives),
  `chessshootout.frontend` has no import cycles, `input_router.py` never
  references a game-specific identifier (`board`, `right_menu`,
  `result_menu`, `skillcheck`, `focus_` — it dispatches through the `Screen`
  contract only), and every `Nav(...)` call site imports the canonical `Nav`
  from `screens.base`.
- `tests/frontend/test_screen_lifecycle.py` — the runtime matrix: enter/exit
  idempotence per screen, a table of real nav paths (menu↔game, a
  menu→history→review→history→menu round trip, a FEN start, a game→game
  self-switch rematch, an online match-found handoff), harmless input on an
  exited/inactive screen, a modal opened on one screen not drawing after
  switching away, and `VIDEORESIZE` mid-transition/mid-drag/mid-animation.
- `tests/infra/test_logging_hygiene.py` pins two invariants across every
  screen: `test_scripted_local_session_info_lines_match_the_allowlist` runs
  a full menu→game→resign→history→review→quit session and asserts every
  `chess.*` INFO line matches `INFO_ALLOWLIST_PREFIXES` (add a prefix there
  when a screen gains a new story-beat log line, per CONTRIBUTING.md's
  logging levels); `test_idle_draw_frame_emits_no_log_records_on_every_screen`
  asserts an idle `draw_frame()` emits zero log records on menu, game,
  history, and review alike — a per-frame log leak on any one screen fails it.

**Adding a new screen or a test for one:** tests go in
`tests/frontend/test_<name>_screen.py`, driven through `make_app()` +
`switch_to`/`request_nav` — never construct a screen directly or reach into
`app.screen` by hand. Cover at minimum `enter`/`exit`, `escape()`, and a
`draw_frame()` smoke. A screen that imports a sibling screen module fails
`test_screen_guards.py` by design — route shared state through
`OnlineCoordinator` or `screens.base` instead.

## Naming

`test_<subject>_<scenario>_<expected>` — e.g. `test_en_passant_capture_clears_target`,
`test_add_time_caps_at_initial`. Avoid generic names (`test_smoke`, `test_feature`)
and names that describe the call rather than the outcome.

## When to parametrize

Collapse **2+** tests whose bodies are structurally identical and differ only in
input data and/or expected output into one `@pytest.mark.parametrize`.

Keep tests **separate** when:
- the setup differs (not just the data),
- they assert different *kinds* of things,
- the test is a multi-step narrative (a real game replay, an interaction flow),
- they take the `server` fixture — **never parametrize a real-server test** (it
  changes xdist worker placement and the shared uvicorn lifecycle).

Convert ad-hoc `for x in [...]: assert f(x)` loops into `parametrize` so each case
is its own node and a failure reports which input broke.

## Parametrize ids

Use explicit `pytest.param(..., id="...")`, snake_case, ≤ ~50 chars. The id is where
a stripped rationale comment goes. Wrap to respect the 100-col limit:

```python
@pytest.mark.parametrize(
    "pieces, expected",
    [
        pytest.param(KN_VS_KN, "draw_insufficient_material", id="knight_vs_knight_auto_draws"),
        pytest.param(KBB_VS_K, None, id="two_bishops_can_mate_not_drawn"),
    ],
)
def test_insufficient_material(pieces, expected):
    assert make_backend(pieces).game_result() == expected
```

Keep the case table inline above the test; hoist it to a module-level constant only
when it is long or reused.

## Comments and docstrings

- **No comments** in test bodies — strip them, including `# ---- section ----`
  dividers. Structure comes from grouping and parametrize ids.
- Real rationale (FIDE citations, regression reasons, non-obvious timing) goes in a
  **function docstring** and/or the **parametrize id**. When unsure whether a comment
  is load-bearing, preserve it as a docstring.
- Strip pure-title module docstrings; **keep** module docstrings that document test
  *strategy* or a non-obvious invariant (tighten them).
- `helpers.py` keeps its docstrings.

Production code follows the repo-wide docstring + annotation standard
(CONTRIBUTING.md, "Docstrings and types"), enforced by
`tests/infra/test_docstring_guard.py` and mypy strict in CI. The shared
test-infra modules (`helpers.py`, the conftests, `focus_helpers.py`,
`online_helpers.py`) follow the
same standard; ordinary test functions need docstrings only where rationale
lives (see above). Guards that scan source text read it through
`helpers.read_source_without_docstrings`, so prose in docstrings can never trip
a code-shape tripwire — never scan raw file text in a new guard.

## Fixtures and helpers

- Pure data / builders → `tests/helpers.py` (imported explicitly, e.g. `make_backend`,
  `sq`, `play_*`, `make_app()`/`start_single_screen()`). `make_app()` boots a
  real `Frontend` shell — every screen constructed, sitting on `MenuScreen` —
  never a partial or screen-less double. `start_single_screen()` drives the
  real `switch_to("game", ...)` path into a local hot-seat game (the
  `"single_screen"` game *mode*, unrelated to the `frontend/screens/` package
  — see "Screens" below). Assertion helpers are named `assert_*` / `expect_*`
  (or `_assert_*`) so the weak-test guard recognizes them.
- Stateful resources → fixtures in `tests/conftest.py` (e.g. the real-uvicorn `server`
  / `server_with_app`, `pygame_display(w, h)` — a factory that builds a
  module-scoped autouse pygame-init fixture; assign its result to a module-level
  name, e.g. `_pygame_init = pygame_display(900, 500)`).
- `tests/frontend/focus_helpers.py` is the focus-mode test suite's own shared
  builder (`make_app`, `start_game`, `FakeTicks` — a `pg.time.get_ticks()`
  millisecond stand-in, distinct from `helpers.FakeClock`'s server-side seconds
  clock) — import it as `from tests.frontend.focus_helpers import ...`.
- `tests/server/conftest.py` holds the server-only `clock`/`app`/`client` fixture
  trio, `ALICE`/`BOB`, and the shared server test doubles — `RecordingWS`,
  `pair_room()`, the skill-check factories (`capture_room`, `fire`, `win_elapsed`,
  `move_raw`, `shot_raw`, `seed_for`, `capture_backend`), `KV_TOKEN_RE` and
  `assert_sweep_clean()` — plus a re-export of `helpers.auth_msg`. Auto-scoped to
  `tests/server/` so it can't shadow a client-side test's own `app`/`client`
  names; server test modules import shared helpers from here, never from each
  other.
- `tests/online/online_helpers.py` holds `wait_for` / `collect_for`, the two
  polling waits every end-to-end online test needs.
- `tests/server/moderation_helpers.py` is the moderation suite's own builder set
  (pattern-library readers, canonical/transformed constructions, and the
  worst-case dense-but-clean arrow store the CPU timing pin measures) — import
  it as `from tests.server.moderation_helpers import ...`.
- `tests/frontend/conftest.py` holds a shared `app` fixture (a drawn 1200x900
  `Frontend` with its games folder isolated to a temp dir) for the menu sub-view
  tests; a file with its own `app` fixture overrides it.
- Don't duplicate a builder across files — add it to `helpers.py` (or the
  relevant subdir's `conftest.py`) once.

## Weak-test ban

A test must exercise real behavior: a non-trivial `assert`, a `pytest.raises/warns`,
or a call to an assertion helper. Banned (the `test_no_weak_tests.py` guard fails on
these):
- `assert True` / `assert <literal>`,
- `assert X == X` (identical operands),
- a test with no assertion and no behavior check (a bare `draw()` "doesn't crash").

A render path still worth guarding should assert something real about the result
(rects produced, expected colors/state), not just "didn't raise". There is no
allowlist — the guard fails on any weak test, so strengthen it rather than ship it.

## Determinism (xdist-safe)

The suite runs under `pytest-xdist`, where each worker imports the test modules in
its own process. A `parametrize`'s argvalues must therefore be **identical across
processes**, or collection diverges and the run errors with "Different tests were
collected between gw0 and gwN". Use fixed literals — never build argvalues from
`uuid.uuid4()`, `random`, the clock, or unsorted `set`/`dict` iteration (sort if you
must). Verify a parametrize change with `pytest tests -n auto`, not just `-n0`.

A pixel-colour assertion must sample a **surface the test owns** (`pg.Surface(...)`
it filled and drew onto), never the shared app display surface after driving the
full app. Two tests in a worker share one `pg.display`, so a neighbour can leave it
in a state that makes the sampled pixel wrong — and because xdist bucketing shifts
when files are added, it passes locally and flakes only in CI. Build the minimal
inputs (a stub board, a real `Clock`) and draw the one element under test onto your
own surface.

## Worked examples (before → after)

**Collapse a cluster** — ten near-identical material checks become one table:

```python
# before: 10 functions, each make_backend(...) then assert game_result() == ...
def test_kvk_is_draw():
    bk = make_backend({sq(7, 4): piece(K, WHITE), sq(0, 4): piece(K, BLACK)})
    assert bk.game_result() == "draw_insufficient_material"
# ...nine more...

# after: one parametrized test; each FIDE case keeps its own expected, rationale in the id
@pytest.mark.parametrize("pieces, expected", [
    pytest.param(KV_K, "draw_insufficient_material", id="kvk_draw"),
    pytest.param(KNN_V_K, "draw_insufficient_material", id="knn_v_k_draw_fide_5_2_2"),
    pytest.param(KBB_V_K, None, id="kbb_v_k_mate_exists_not_drawn"),
])
def test_insufficient_material(pieces, expected):
    assert make_backend(pieces).game_result() == expected
```

**Strengthen a render smoke** — assert the output, not just "doesn't crash":

```python
# before: a weak smoke (banned by the guard)
def test_draw_button_smoke_idle(font):
    draw_button(surface, rect, "OK", font)   # no assertion

# after: assert the rendered pixels match the state colour, with an idle-vs-pressed diff
def test_draw_button_idle_and_pressed_render_differently(font):
    draw_button(surface, rect, "OK", font, force_pressed=False)
    idle = surface.get_at((rect.x + 8, rect.centery))[:3]
    draw_button(surface, rect, "OK", font, force_pressed=True)
    pressed = surface.get_at((rect.x + 8, rect.centery))[:3]
    assert idle != pressed
    assert idle == pg.Color(Colors.dark_menu)[:3]
```

To prove a render assertion bites, neutralize the draw **inside the test** with
`monkeypatch` (it auto-reverts) — never edit a source file, even temporarily.
