# Chess Shootout

A full-featured chess game built with [pygame](https://www.pygame.org/):
a clean UI at rest with comedic **gun-fight** chaos in the moments — every
capture is won through a quick **skill-check**, pieces blast each other off the
board, and an FPS-style announcer calls the killstreaks. Local hot-seat play,
premoves, clocks, annotations, PGN auto-save and review, plus an authoritative
server for online two-player matches.

## Features

**Skill-checks — the *Shootout***
- Every capture and promotion triggers a fast skill-check: win it and the move
  lands; miss it and that move is locked for the turn, so you play something
  else — no turn is forfeited. Four checks roll an even split per capture;
  promotions are wheel-only.
- **Timing wheel** — tap when the needle is inside the shrinking sweet-spot; it
  spins faster the more material is at stake.
- **Steady-Aim** — a crosshair auto-traces a figure-8 over the shrinking victim;
  it is multi-shot, and every miss escalates the sway and shrink.
- **Whack-a-Mole** — the captured piece dives into a pit and pops back from a
  ring of glowing holes; land its quota of shots before it ducks away for good —
  three on a pawn, four on a knight, bishop or rook, all five on a queen — and
  three whiffed shots miss it.
- **Combo** — a strip of arrow prompts flies in and stays put; punch each direction in order
  with the Arrows/WASD keys or the on-screen pad, and three wrong inputs miss it.
- Online, the server adjudicates every shot and your opponent watches a live,
  read-only mirror of the same minigame while their own board stays interactive.

**Gameplay**
- Complete rules engine: castling, en passant, promotion, threefold
  repetition, fifty-move rule, insufficient material.
- Drag-and-drop or click-to-move; right-click highlights and arrows.
- chess.com-style premove queueing (pseudo-legal validation, bouncing chains).
- Time controls with increment, board flip, undo, resign, draw agreement.
- Opening names live in the shot log (full ECO book), and an **Auto-queen**
  option that skips the promotion picker — the skill check still fires.
- Start from any FEN, or play two-up on a single screen.
- Captured-piece graveyard with running material balance; master-volume
  slider persisted to `.env`.
- Help modal listing every shortcut (`?` or the right-panel button).

**Presentation**
- Gun-fight captures — muzzle flash, tracer, impact, bullet holes, ragdoll,
  comic blood, and board screen-shake; each piece type fires its own shot.
- FPS-style announcer with killstreaks (FIRST BLOOD → DOUBLE … GODLIKE), a
  checkmate takeover, and a surrender flag.
- Custom borderless window chrome — native drag, edge/corner resize, snap and
  taskbar-aware maximize (Windows Aero-Snap / drag-to-top; handled by the window
  manager on Linux), fullscreen (`F11` or the green title-bar button), minimize,
  and a themed title bar on every screen.
- Animated menu "battle" backdrop and themed result / online screens.
- The menu's **News** card lists every update as an expandable, scrollable feed
  with an unread badge for what you haven't read yet.
- Focus mode (`H`, the board-seam arrow, or `Esc` to exit) collapses the in-game
  UI to just the board, which grows to a centered square. **Options → Focus mode
  → "Show in focus"** picks what stays on the board: a slim per-player **Time
  Line** on the top/bottom edges that depletes with each clock (default), the
  full player strips, or nothing.

**Audio**
- A distinct move sound per piece, and a distinct gun on a won capture check —
  pawn revolver, knight hand-cannon, bishop lever-action, rook shotgun, queen
  blunderbuss, king ray-gun — over a victim "oof" (the queen has her own).
- Skill-check cues: a ding as a check opens, a tick on every good beat — the
  needle in the sweet-spot, the crosshair on target, a mole tagged, a combo
  arrow nailed — and a win/miss sting on resolve. Online, only your own check is
  audible — a spectated opponent's is silent.
- A universal typewriter click on every button, key, and empty-square press;
  moves, pickups, and drops have their own sounds instead.
- A two-stage low-time heartbeat that starts slow near 10% on the clock and
  swaps to fast near 5%, riding a volume ramp as time runs out.
- Announcer killstreaks and result voices (get-ready, you-win, you-lose, draw,
  surrender), plus castle doors, undo rewind, board-flip whoosh, and toast pops.
- Master and menu-gun volume sliders, persisted to `.env`.

**PGN**
- Every game auto-saves to `games/<prefix>-YYYYMMDD-HHMMSS.pgn` (`local`,
  `bot`, or `online`), continuously while you play — an in-progress game is
  kept on disk with `[Result "*"]` and finalized when it ends, so a crash or
  disconnect can't lose it. If the games folder isn't writable you get a
  toast and the save falls back to the OS data directory.
- Load and review past games from the **History** menu; step through with
  the arrow keys.
- Skill-check outcomes (hits and misses) are saved as standard `{comments}` and
  replayed in review; the file still imports cleanly into other chess apps.
- **Open PGN** — from the result modal, or the **Open PGN** button while
  reviewing a game — opens the file in your OS default app.

**Online**
- Authoritative FastAPI + WebSocket server that runs the same engine as the
  client and validates every move.
- Rematch (colors swap; the series score follows the player, not the color),
  takeback, draw offers, and **Give 15 sec** (tops up the opponent's clock,
  capped at the starting time).
- **Shared marks** — flip SHARE in the rail's SIGNALS section to broadcast your
  highlights and arrows live; your opponent's marks arrive in blue.
- Shared drawings are screened for prohibited symbols before they reach your
  opponent; a blocked mark turns red for you with a heads-up toast.
  Flip HIDE in the SIGNALS section (or **Options → Online → Hide opponent's
  marks**) to hide their shared marks entirely — both toggle the same setting.
- **Quick chat** — preset lines only; your queen speaks them in a bubble on the
  opponent's board.
- Live abort / abandon / reconnect countdowns in the player strips: an opening
  nobody plays ends as **aborted** (nobody wins), and going quiet once both
  players have moved is an auto-resign — see [Idle timeouts](#idle-timeouts).
- Matchmaking gives up after 10 minutes in the queue with a toast — hit
  **FIND MATCH** again to keep looking.
- Layered reconnection for WiFi blips, app restarts, and server restarts
  (see [Reconnection](#reconnection)).
- Crash-log capture for easy bug reports.

## Download

No Python needed — grab the file for your OS from the
[**Releases**](https://github.com/xiaomyung/chess-shootout/releases) page.

| OS | File | First run |
|----|------|-----------|
| **Windows 10/11** | `ChessShootout-<version>-Setup.exe` (installer, no admin) or `ChessShootout-<version>-Portable.exe` (portable) | Unsigned, so SmartScreen warns: **More info → Run anyway**. |
| **macOS** (Apple Silicon) | `ChessShootout-<version>.dmg` | Drag **Chess Shootout** to Applications. Unsigned, so first launch is blocked: **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine /Applications/ChessShootout.app`. |
| **Linux** (incl. Arch) | `ChessShootout-<version>-x86_64.AppImage` | `chmod +x` it, then run. No install, no FUSE. |

Games, settings, and logs live in a per-user location (`%APPDATA%`,
`~/Library/Application Support`, `~/.local/share`); change the games folder
anytime from the in-app **Options** (bottom of the left nav rail). The Windows
**portable** build is the exception — it keeps everything in a `data/` folder
beside the executable, so the whole app stays self-contained.

> **Tip for the portable build:** since it creates a `data/` folder next to the
> `.exe`, give it its own folder before the first run — make a new directory
> (e.g. `ChessShootout/`, on a USB stick or anywhere), drop
> `ChessShootout-<version>-Portable.exe` inside, and launch it from there. That
> keeps the app and its data tidy in one place instead of scattering a `data/`
> folder into wherever you downloaded it.

## Play from source

Requires **Python 3.12** — and specifically 3.12: newer pygame wheels drop
`pygame.mixer` (no audio) and the code uses 3.12 syntax. There are no other
runtime dependencies — all audio ships as `.ogg`, so `ffmpeg` is not required.

Once you have 3.12:

```bash
git clone https://github.com/xiaomyung/chess-shootout.git
cd chess-shootout
python3.12 -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
python --version                  # confirm 3.12.x
pip install -e .                  # add ".[dev]" for tests + linting
python -m chessshootout.main      # or just: chess-shootout
```

`-e` is an editable install: pip resolves dependencies but runs against your
working tree, so edits take effect with no reinstall.

### Getting Python 3.12

**Linux** — `pyenv` is the most reliable route, since distro packages drift:

```bash
curl https://pyenv.run | bash     # one-time; follow the shell-rc steps it prints
pyenv install 3.12
pyenv shell 3.12                  # `python3.12` now resolves for the venv step above
```

<details>
<summary>Native distro packages (when they happen to ship 3.12)</summary>

| Distro | Command |
|---|---|
| Arch (rolling) | often already 3.12 — check `python --version`, else use pyenv |
| Ubuntu 24.04+ | `sudo apt install python3.12 python3.12-venv` |
| Ubuntu 22.04 / 23.x | deadsnakes: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install python3.12 python3.12-venv` |
| Debian 12 | enable bookworm-backports, or use pyenv (simpler) |
| Fedora 39+ | `sudo dnf install python3.12` |

</details>

**macOS** — `brew install python@3.12`

**Windows** — install 3.12.x from [python.org](https://www.python.org/downloads/)
(tick *Add Python to PATH*); the `py -3.12` launcher then selects it.

## Hotkeys

| Key | Action |
|---|---|
| `F` | Flip board |
| `H` | Focus mode — collapse the UI to just the board (live game only) |
| `F11` | Toggle fullscreen |
| `R` | Resign / promote to rook (when a promotion is pending) |
| `D` | Offer draw |
| `Q` / `B` / `N` | Promote (queen / bishop / knight) |
| `Space` / Click | Fire the active skill-check (wheel / aim / whack) |
| Arrows / WASD | Combo check input, or click the on-screen pad |
| `Z` | Undo move (`Ctrl+Z` also works; online: takeback request) |
| `G` | Give 15 seconds (hold the **+15** cap to ramp) |
| `A` / `S` / `C` | Collapse or expand rail sections |
| Left / Right | Step through moves (also during live games) |
| `Home` / `End` | Jump to ply 0 / return to live play |
| `?` | Open Help modal |
| `Esc` | Context Back/Cancel — closes the top modal, else exits focus mode, else the quit / resign prompt (never the window) |

## Online play

Two players, one server — authoritative, running the same engine as the client.

### Quick start (local)

Three terminals, copy-paste as-is:

```bash
# 1 — server
HOST=127.0.0.1 PORT=8000 LOG_LEVEL=DEBUG python -m chessshootout.server
```
```bash
# 2 — Alice
CHESS_SERVER_ADDR=localhost:8000 CHESS_DATA_DIR=/tmp/cs-a LOG_LEVEL=DEBUG \
  python -m chessshootout.main --client-uuid alice --nickname Alice
```
```bash
# 3 — Bob
CHESS_SERVER_ADDR=localhost:8000 CHESS_DATA_DIR=/tmp/cs-b LOG_LEVEL=DEBUG \
  python -m chessshootout.main --client-uuid bob --nickname Bob
```

Every variable is set explicitly so nothing inherited can interfere:
`CHESS_SERVER_ADDR` set at launch beats both a saved `.env` and the in-app
Official / Custom picker for that run, `CHESS_DATA_DIR` gives each client its own
settings and PGN folder, and `HOST=127.0.0.1` binds the loopback so a service
already on `0.0.0.0:8000` does not collide.

`--client-uuid alice` is a debug shortcut: any non-UUID4 alias is coerced to
a deterministic UUID4 client-side so the server's validator accepts it. Real
clients auto-generate and persist a UUID4 on first launch.

In both clients pick **Online**, choose the **same time control**, then hit
**FIND MATCH** — both see "Match found!" and the game begins.

### Settings (`.env`)

The client reads a gitignored `.env` at the repo root — copy `.env.example`:

```
CHESS_SERVER_MODE=custom                # "official" or "custom"
CHESS_CUSTOM_SERVER_ADDR=localhost:8000 # the address used in custom mode
CHESS_SERVER_ADDR=localhost:8000        # resolved target; a launch var wins
CHESS_NICKNAME=YourName
CHESS_CLIENT_UUID=                      # auto-generated UUID4 on first launch
CHESS_LAST_MODE=                        # auto-saved
CHESS_MASTER_VOLUME=0.70                # 0.0–1.0; the slider writes here
```

`--client-uuid` and `--nickname` override `.env` for a single run (handy for
two clients on one machine). The games folder is set from in-app **Options**
and persisted as `CHESS_DATA_DIR`.

**Options → Performance** toggles a title-bar diagnostic overlay: FPS, rolling
average / minimum FPS, 1% low (the stutter metric), and per-frame render time in
milliseconds. Each is an independent switch (`CHESS_SHOW_FPS`,
`CHESS_SHOW_FRAME_STATS`, `CHESS_SHOW_1PCT_LOW`, `CHESS_SHOW_FRAMETIME`,
`CHESS_SHOW_PING`).

**Options → Online → Server** picks where online games connect: **Official** —
the public server at `server.chess-shootout.com` — or **Custom**, which reveals
a **Custom address** field for a host (or `host:port`) you run yourself. The
**Connection** row's **Test** button checks the server answers before you queue
and reports the round-trip in milliseconds plus the server's version, or flags a
protocol mismatch when it is too old or too new for your client. Servers can't
be swapped mid-session — leave the online game first.

The picker persists as `CHESS_SERVER_MODE` and `CHESS_CUSTOM_SERVER_ADDR`, and
writes the address it resolves to into `CHESS_SERVER_ADDR`. A downloaded desktop
app starts on **Official**, a source checkout on **Custom** at `localhost:8000`.
Setting `CHESS_SERVER_ADDR` in the environment at launch wins for that run and
bypasses the picker entirely.

### Keeping the client current

The server names the oldest client build it plays with. Anything older is turned
away at matchmaking — before a game exists — so an out-of-date copy never sits
in a search that could not have found it an opponent. Instead you get an
**Update required** card naming both versions (the build you run and the oldest
one the server takes), with a single **OK** that returns you to the play card. A
server speaking a different protocol version than your build shows the same
card.

Offline play is untouched: local and bot games never contact a server. So is
running from source — a source checkout carries no stamped version, and an
unversioned client is always admitted.

### In-game actions

- **Resign** — opponent wins.
- **Draw** (on your turn) — opponent gets Accept/Decline; mutual offers auto-agree.
- **Undo / takeback** — only right after your own move, while the opponent is
  on the clock; on accept, one ply rolls back and the clocks restore.
- **Give 15 sec** — tops up the opponent's clock; **tap** for +15 s or **hold**
  the button to keep adding 15 s every 0.1 s until it reaches the starting time
  control. Online, the total is server-authoritative and the clock reconciles on
  the grant.
- **Rematch** (from the result modal) — the same room restarts with swapped
  colors; the series score follows the players, not the colors.

### Idle timeouts

Online openings can't be left hanging: until the game is properly under way,
whoever is on the move has **60 seconds** to play it.

| Moves played | On the move | Nobody moves for 60 s |
|---|---|---|
| None yet | White | Game **aborts** — no winner |
| White's first, no reply | Black | Game **aborts** — no winner |
| Both first moves in | White | White **resigns** — an ordinary loss |

Once White's second move lands the game is under way and the chess clock is the
only deadline left. Both players watch the same countdown on the idle player's
strip — `Abort in …` or `Resign in …` — and it keeps ticking while that player
is disconnected, so dropping out buys no extra time. An abort has no winner, no
series point, and no saved PGN when it lands before a single move; an
auto-resign is an ordinary resignation, series point and all.

### Reconnection

Layered recovery:

- **WS drops mid-game** (WiFi blip) — the client retries `/resume` until the
  grace window (~60 s, server-configurable) expires. The opponent sees a
  "Reconnecting…" overlay, a red status dot, and an `Abandon in …` countdown;
  you see `Reconnect in …`. Liveness rides an **application-level heartbeat** —
  a small `ping` the client sends every couple of seconds — so a dropped or
  half-open connection is noticed even through a proxy. **Desync** is caught two
  ways: every move/takeback carries a `ply` counter, and the heartbeat reports
  the client's ply, so the server spots a player who has fallen behind and tells
  it to `/resume`. A heartbeat that merely crossed a move or a takeback on the
  wire is not a desync — for a short transit window after each change the server
  still accepts the position the client was on just before it — and a mismatch
  must survive two judged heartbeats before a resync is ordered. The order names
  the ply it was written against, so a client that has already caught up to it
  drops it instead of rebuilding for nothing. While someone resyncs, the other
  player sees an amber status dot and a toast.
- **Client app restart** — on next launch the client probes `POST /reclaim`;
  if the room is still alive, a **Reconnect** button appears in the start menu.
- **Server restart** — when `/resume` fails but `/healthz` is reachable, the
  client shows **"Server restarted — game ended"** with New Search / Cancel;
  New Search re-matches against your previous time control directly.

If a countdown ends with no reconnect, the game **aborts** (no winner) when a
desync was left unresolved, or the waiting player **wins by abandonment** when
the other side simply left; starting a new game while still in one forfeits the
old immediately. A drop pauses nothing else, either: the opening's
[idle timeouts](#idle-timeouts) keep counting down while you are away, so a
reconnect that lands after the window can find the game already over. Rooms are
in-memory only (no DB), so a true server crash loses game state — but New Search
starts a fresh game in one click.

### Crash logs

Unhandled exceptions write `crashlogs/YYYYMMDD-HHMMSS.txt` (traceback, app
state, and the whole-session log buffer). `crashlogs/` is gitignored — attach
the file when reporting a bug.

### Deployment

See [deploy/README.md](deploy/README.md) for the containerized single-VPS setup.
This repo ships one `docker compose` service — the hardened `gameserver`
container — which runs behind a standalone TLS-terminating edge proxy (its own
stack, from its own repo) and Cloudflare in front of that.

## Development

Install the dev extra, then run the same checks CI does:

```bash
pip install -e ".[dev]"
pytest tests -n 8 -q                            # run the test suite
pylama chessshootout tests                      # pycodestyle + pyflakes; exits 0 when clean
```

CI installs from the committed `uv.lock` rather than resolving fresh, so every
job (and every published binary) gets the exact pinned dependency set. To
reproduce that environment locally: `uv sync --extra dev`.

These gate merges to `master` (the `test` and `lint` jobs), alongside two more
required checks: `version-bump` fails any PR that doesn't change
`pyproject.toml`'s `[project].version`, and `lock-check` fails any PR where
`uv.lock` has drifted from `pyproject.toml`. `make bump` moves both together, so
a green local run **plus a version bump** means the PR checks will pass.

### Releasing

Releases are built by CI on a version tag — merging to `master` builds nothing.
The version lives in **one place**, `pyproject.toml`'s `[project].version`; bump
it (every PR does, anyway), then tag to match:

```bash
git tag -a v1.2.0 -m "Chess Shootout 1.2.0"   # v1.2.0-rc1 for a pre-release
git push origin v1.2.0
```

The tagged build runs the test suite, then builds all three OSes and stamps the
version into every artifact filename, the Windows installer, the macOS bundle,
and the in-app footer, before assembling a **draft** GitHub Release. The version
is read from `pyproject.toml`, and the build **fails if the tag isn't
`v<version>`** — so the tag and the package version can't drift. A hyphen marks
a pre-release. Review the four artifacts, then **Publish**.

## License

**Source available, not open source.** Because it forbids commercial use, this
project is *not* an OSI "open source" license — please don't call it that.

- **Code** — [PolyForm Noncommercial License 1.0.0](LICENSE). Read, run, modify,
  fork, and share it for any **non-commercial** purpose. You may **not** sell it
  or use it (or parts of it) commercially.
- **Original assets** (piece art, icons, and any sounds authored for the
  project) — [**CC BY-NC 4.0**](LICENSE-CC-BY-NC-4.0.txt): reuse with credit,
  non-commercial only.
- **Bundled third-party assets** (fonts, sound effects, emoji) keep their own
  licenses — see [ATTRIBUTION.md](ATTRIBUTION.md).
- **v1.0.0** was released under the **MIT License** and stays MIT; the terms
  above apply from **v2.0.0** onward.

"Non-commercial" is meant generously: personal/hobby use, education, clubs,
streaming/videos, and free forks that take donations are all fine — only selling
or charging is off-limits. Running your own server for non-commercial play is
fine; the online service operated by the author is not part of this grant and
access to it is at the author's discretion.

**"Chess Shootout™"** is a trademark of the author — forks and derivatives must
use a different name and not imply endorsement.

Contributions are welcome under the terms in [CONTRIBUTING.md](CONTRIBUTING.md).
