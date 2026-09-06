"""The published HTTP contract: /openapi.json is served to anyone who asks, so
its route table and its prose are both public API.

The (method, path, operationId) triples are pinned as LITERALS, not derived from
the app -- a derived list would happily agree with itself after a route was
renamed, moved or dropped. operationId is included because FastAPI mints it from
the handler's own function name, so renaming a handler silently rewrites a
published identifier that clients and generators key on.

Descriptions are deliberately NOT pinned verbatim: they are handler docstrings
and get reworded as the endpoints gain fields. What IS pinned is that each one
exists and stays public copy -- no internal vocabulary (`guard`, `quarantine`,
`sweep`, `app.state`) leaking into the document a stranger can curl.
"""
import pytest


ROUTES = [
    ("get", "/", "root__get"),
    ("get", "/favicon.ico", "favicon_favicon_ico_get"),
    ("get", "/healthz", "healthz_healthz_get"),
    ("post", "/matchmake", "post_matchmake_matchmake_post"),
    ("delete", "/matchmake", "delete_matchmake_matchmake_delete"),
    ("post", "/resume", "post_resume_resume_post"),
    ("post", "/reclaim", "post_reclaim_reclaim_post"),
]

INTERNAL_WORDS = ("guard", "quarantine", "sweep", "app.state")


@pytest.fixture
def spec(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    return r.json()


def test_openapi_publishes_exactly_the_seven_documented_routes(spec):
    """The whole HTTP surface, as literals. A new route has to be added here
    deliberately, and a removed or renamed one cannot slip out silently."""
    published = sorted(
        (method, path, operation["operationId"])
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
    )
    assert published == sorted(ROUTES)


def test_openapi_documents_no_websocket_route(spec):
    """/ws/{room_id} carries every move but is a websocket, which OpenAPI cannot
    describe -- so the seven above really are the whole documented surface."""
    assert "/ws/{room_id}" not in spec["paths"]


@pytest.mark.parametrize("method, path, operation_id", ROUTES,
                         ids=[f"{m}_{p}" for m, p, _ in ROUTES])
def test_every_route_description_is_public_copy(spec, method, path, operation_id):
    """Handler docstrings ARE the published description, so the house docstring
    rule doubles as an API-copy rule here: every route explains itself, and none
    of them explains the server's insides to the internet."""
    operation = spec["paths"][path][method]
    assert operation["operationId"] == operation_id
    description = operation.get("description", "")
    assert description.strip(), f"{method.upper()} {path} publishes an empty description"
    lowered = description.lower()
    leaked = [word for word in INTERNAL_WORDS if word in lowered]
    assert leaked == [], f"{method.upper()} {path} description leaks internals: {leaked}"
