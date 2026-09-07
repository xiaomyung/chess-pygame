.PHONY: build up down logs loadtest update bump

# Build the server image via compose.
build:
	docker compose build

# Bump [project].version (patch) and keep uv.lock in sync; run once per PR.
bump:
	uv version --bump patch

# Start the one service this repo ships, published on 127.0.0.1:8000. TLS lives
# in the standalone edge proxy stack, which is started from its own repo.
up:
	docker compose up -d gameserver

# Stop and remove the gameserver container; the named data volume survives.
down:
	docker compose down

logs:
	docker compose logs -f

# Drive synthetic games at a local/throwaway server. Pass args via ARGS, e.g.
#   make loadtest ARGS="--addr localhost:8000 --rooms 200 --hold 60"
loadtest:
	.venv/bin/python tools/loadtest_server.py $(ARGS)

# Update the running server to the latest release (or REF=v2.1.5) and recreate it.
update:
	./deploy/update.sh $(REF)
