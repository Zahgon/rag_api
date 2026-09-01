# app/routes/docs_routes.py
"""Interactive API documentation, served as it always was.

``/openapi.json``, ``/docs``, ``/docs/oauth2-redirect`` and ``/redoc`` answer the
documents this API has always published for its clients and code generators. They
live under ``app/api_docs/``: the OpenAPI schema — with the debug-only paths merged
in when those routes are registered — and the Swagger UI / ReDoc pages.

The schema is a document of this API now, not something derived from the routes at
import time, so a new route, or a change to a request or response model, has to be
written into ``app/api_docs/openapi.json`` (or ``openapi_debug_paths.json``) as part
of the same change.
"""

import json
from pathlib import Path

from flask import Blueprint, Response

from app.config import debug_mode

API_DOCS_DIR = Path(__file__).resolve().parent.parent / "api_docs"

bp = Blueprint("docs", __name__)


def _read(name: str) -> str:
    return (API_DOCS_DIR / name).read_text(encoding="utf-8")


@bp.get("/openapi.json")
def openapi_schema():
    schema = json.loads(_read("openapi.json"))
    if debug_mode:
        schema["paths"].update(json.loads(_read("openapi_debug_paths.json")))
    return Response(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        mimetype="application/json",
    )


@bp.get("/docs")
def swagger_ui():
    return Response(_read("swagger_ui.html"), mimetype="text/html")


@bp.get("/docs/oauth2-redirect")
def swagger_ui_oauth2_redirect():
    return Response(_read("swagger_ui_oauth2_redirect.html"), mimetype="text/html")


@bp.get("/redoc")
def redoc():
    return Response(_read("redoc.html"), mimetype="text/html")
