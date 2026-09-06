# Deploying the server (containerized)

The server runs as a single hardened `gameserver` container (uvicorn) managed by
`docker compose`. TLS termination and the public `:80/:443` surface live in a
**standalone edge proxy stack** (Caddy) that ships from its own private repo and runs
separately on the VPS. This repo's compose file joins that stack's external `edge`
docker network under the alias `chess-gameserver`, and the edge proxy reverse-proxies
to it.

```
Player ──wss:443──▶ Cloudflare (orange cloud, Full strict)
                         ▼
        edge proxy stack (Caddy, separate repo)  :80/:443, terminates TLS
                         │  reverse_proxy chess-gameserver:8000  (external `edge` network)
                         ▼
        gameserver container  (uvicorn; 127.0.0.1:8000 published for the healthcheck)
```

The server is **stateless** (in-memory rooms, no DB) — a restart loses in-flight
games. TLS, the Cloudflare origin certificate, and the "only Cloudflare reaches the
origin" firewall are all owned by the edge stack and documented in its own repo — none
of that lives here anymore. This repo ships only the gameserver; the `127.0.0.1:8000`
publish is for the local healthcheck only.

## Prerequisites

- A Debian VPS with Docker installed, running the standalone **edge proxy stack**. That
  stack creates and owns the external `edge` docker network (bridge) and terminates TLS
  in front of this container; deploy it first.
- Read access to the image at `ghcr.io/xiaomyung/chess-shootout-gameserver`: make the GHCR
  package **Public** (repo → Packages → Package settings), or `docker login ghcr.io`
  once on the box.

## One-time setup

### 1. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add yourself to the `docker` group, then log out and back in so it takes effect:

```bash
sudo usermod -aG docker $USER
```

### 2. Clone the project

```bash
sudo mkdir -p /srv/chess-shootout
sudo chown "$USER:$USER" /srv/chess-shootout
git clone https://github.com/xiaomyung/chess-shootout.git /srv/chess-shootout
cd /srv/chess-shootout
```

### 3. Config files

Create `.env` in the project dir — it selects which image runs:

```bash
echo "IMAGE_TAG=latest" > .env
```

Create `gameserver.env` — it is injected into the container:

```bash
cat > gameserver.env <<'EOF'
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
MAX_ROOMS=100
EOF
```

`HOST=0.0.0.0` lets the edge proxy reach the app over the shared `edge` network. Do not
set `LOG_FILE` (logs go to stdout). `TRUSTED_PROXIES` is set in `docker-compose.yml`
(the edge proxy's IP on the `edge` network), not here. Optional tunables you can add to
`gameserver.env`, with the floor each one is held to: `GRACE_SECONDS=60` (min `1.0`),
`HEARTBEAT_INTERVAL_SECONDS=2` (min `0.5`), `HEARTBEAT_MISS_LIMIT=3` (min `1`).

A tunable that does not parse as a number falls back to its default; one that parses
but is below its floor — or is an infinity or a nan — is replaced by the floor. Either
way the server logs a `WARNING` naming the variable and keeps serving; it never starts
with a zero-second heartbeat because of a typo. These particular warnings are emitted
while the module is imported, before the log format is configured, so in
`docker compose logs` they appear as bare stderr lines (`env unparsable name=…` /
`env clamped name=…`) ahead of the normal timestamped output.

The oldest client build the server accepts is **not** an env knob: `MIN_CLIENT_VERSION`
is a source constant in `chessshootout/server/protocol.py` (currently `"2.13.0"`), baked
into the image, and it is reported on the `GET /` manifest. Clients older than it are
refused at `/matchmake` with a 426; a client that reports no version at all — anyone
running from source — is admitted. Raise it only in a release that actually ships a
break, and remember that raising it turns away every older build the moment the new
image starts.

### 4. Start

The external `edge` network must already exist (created by the edge proxy stack). Then:

```bash
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:8000/healthz
```

The compose file attaches the container to the external `edge` network with the alias
`chess-gameserver`, which is how the proxy reaches it. `restart: unless-stopped` brings
the container back automatically after a reboot. For systemd integration (so
`systemctl stop` triggers the graceful client drain), you can optionally install the
bundled unit:

```bash
sudo cp deploy/gameserver-compose.service.example /etc/systemd/system/gameserver-compose.service
sudo systemctl daemon-reload
sudo systemctl enable --now gameserver-compose
```

## Updating

A version-bumped PR merging to master auto-publishes the new image to GHCR
(`docker.yml`). Update to the latest release with one command on the box:

```bash
cd /srv/chess-shootout
./deploy/update.sh
```

To pin a specific version, or to roll back, pass its release tag:

```bash
cd /srv/chess-shootout
./deploy/update.sh v2.1.5
```

The script pulls the matching CI-built image from GHCR, refreshes the compose file from
git, recreates the `gameserver` container with the graceful `server_shutdown` drain, and
reports the installed version before and after (`was <ver>@<digest> -> now
<ver>@<digest>`, read from `/healthz`). Each run is appended to `deploy/update.log` (UTC,
gitignored). It falls back to `sudo` automatically when your shell isn't in the `docker`
group.

A failed pull **aborts the update** and leaves the running container untouched — the
`up` passes `--no-build`, so a missing or unreachable image can never fall through to
building the compose file's `build:` stage from whatever source happens to be checked
out on the box.

### Image scanning is advisory, not a gate

`docker.yml` pushes the image to GHCR **first**, then runs a trivy scan as a
non-blocking reporting step (`continue-on-error`, table output to the job log, no
failing exit code). It flags HIGH/CRITICAL findings that already have fixes available,
but it does **not** hold back a release — a tag with a CRITICAL finding is published and
pullable exactly like any other. Read the scan output in the workflow log when deciding
whether to deploy a build.

## Operations

Run these from `/srv/chess-shootout`.

Status:

```bash
docker compose ps
```

Live logs:

```bash
docker compose logs -f
```

The first lines after a start say exactly what is running and with which settings —
the build and protocol version plus the room cap (`gameserver v6 release=2.13.0
listening (max_rooms=100)`), the trusted proxy set (`trusted proxies …`, a `WARNING`
when `TRUSTED_PROXIES` was configured but parsed to nothing), and every timing knob in
effect on one line (`tuning grace=… heartbeat=… miss_limit=… heartbeat_timeout=…
tick=… sweep_stale=… transit_grace=… stable_heartbeats=…`). Read that line rather than
guessing whether a `gameserver.env` edit took. A clean stop logs the matching
`gameserver shutting down uptime_s=… rooms_active=… queue_depth=… sockets=…` before
the drain, so a log without it means the process was killed rather than stopped.

Restart:

```bash
docker compose restart
```

Stop:

```bash
docker compose down
```

A clean stop, restart, or update lets the server broadcast `server_shutdown` to
connected clients; the compose `stop_grace_period` covers the 10 s drain.

### Health endpoint

`GET /healthz` is what the container healthcheck, `update.sh` and the game's own
connection test all poll:

```bash
curl -s http://127.0.0.1:8000/healthz
```

```json
{
  "status": "ok",
  "version": 6,
  "app_version": "2.13.0",
  "rooms_active": 3,
  "queue_depth": 1,
  "uptime_s": 4210.7,
  "housekeeping_age_s": 0.4
}
```

The verdict is in the body — the endpoint answers **HTTP 200 in every state** this
release, healthy or not:

- `status` — `ok`, `full` or `degraded`. **`full`** means the room cap is reached
  (games in progress plus players waiting have hit `MAX_ROOMS`), so the next
  matchmake request is refused; the server is otherwise fine. **`degraded`** means the
  server's own housekeeping pass — the loop that expires grace and idle windows, ticks
  clocks, times out silent sockets and unanswered skill checks, drops orphaned rooms
  and reaps stale queue entries — has not completed within its staleness threshold. The
  server still answers and still plays games, but it can no longer be trusted to end
  them on time. A step that is actually raising also writes a throttled `ERROR` with a
  traceback, so read the log next.
- `housekeeping_age_s` — seconds since the last housekeeping pass that completed with
  no failures, and the number `degraded` is computed from. It reads `0.0` until the
  loop has started, so a server that has only just come up never calls itself behind.
- `version` is the wire protocol version, `app_version` the build (empty on a source
  run), `rooms_active` / `queue_depth` the current load.

Any uptime monitor that can poll a URL and read a JSON field is enough — point one at
`https://server.chess-shootout.com/healthz` and alert on `status != "ok"`. Nothing in
this deploy is tied to a particular monitoring product.

The Dockerfile `HEALTHCHECK` and `deploy/update.sh` are unchanged by all this: both
only ask whether the endpoint answers 200, so a `degraded` server keeps a healthy
container and an update still reports its before/after versions. Answering non-200 for
`degraded` is deliberately left to the monitoring work that would own the consumer of
it — flipping the status code today would make three client paths and `curl -f` read a
busy-but-working server as unreachable.
