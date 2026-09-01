# app/errors.py
"""JSON error responses.

The routes raise :class:`APIError` where they used to raise the framework's HTTP
exception, and request parsing raises :class:`RequestValidationError`. Both are
rendered here, so the bodies on the wire — ``{"detail": ...}`` for an error and
``{"detail": [...], "message": "Request validation failed"}`` for a 422 — are
unchanged.
"""

import traceback
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from flask import Flask, Response, current_app, jsonify, request
from werkzeug.exceptions import HTTPException, InternalServerError, MethodNotAllowed

from app.config import logger


class APIError(Exception):
    """An error carrying the status code and ``detail`` payload to return."""

    def __init__(self, status_code: int, detail: Any = None) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = (
            detail if detail is not None else HTTPStatus(self.status_code).phrase
        )


class RequestValidationError(Exception):
    """A request did not match the schema its route declares."""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        super().__init__(errors)
        self._errors = errors

    def errors(self) -> List[Dict[str, Any]]:
        return self._errors


def _trailing_slash_redirect() -> Optional[Response]:
    """Redirect to the same path with its trailing slash toggled, if that path exists.

    The previous router answered an unknown ``/ids/`` with a ``307`` to ``/ids`` (and the
    other way round) whenever the toggled path had a route, whatever its methods.
    """
    path = request.path
    if path == "/":
        return None
    other = path.rstrip("/") if path.endswith("/") else path + "/"
    adapter = current_app.url_map.bind_to_environ(request.environ)
    try:
        adapter.match(other, method=request.method)
    except MethodNotAllowed:
        pass  # the path has a route for other methods: still redirected
    except HTTPException:
        return None
    location = request.host_url.rstrip("/") + other
    if request.query_string:
        location += "?" + request.query_string.decode("latin-1")
    response = Response(status=307)
    response.headers["Location"] = location
    del response.headers["Content-Type"]
    return response


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        """Answer routing-level errors — unknown path, wrong method — as JSON.

        Without this the framework renders them as HTML, which no client of a
        JSON API is prepared to parse.
        """
        if exc is request.routing_exception and exc.code == 404:
            redirect = _trailing_slash_redirect()
            if redirect is not None:
                return redirect
        return jsonify({"detail": exc.name}), exc.code

    @app.errorhandler(InternalServerError)
    def handle_unhandled_exception(exc: InternalServerError):
        """Answer an unhandled exception the way the previous server did.

        A plain-text ``Internal Server Error``; in debug mode, the plain-text
        traceback of the exception instead.
        """
        original = getattr(exc, "original_exception", None)
        if current_app.debug and original is not None:
            text = "".join(
                traceback.format_exception(type(original), original, original.__traceback__)
            )
            return Response(text, status=500, mimetype="text/plain")
        return Response("Internal Server Error", status=500, mimetype="text/plain")

    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):
        return jsonify({"detail": exc.detail}), exc.status_code

    @app.errorhandler(RequestValidationError)
    def handle_validation_error(exc: RequestValidationError):
        logger.debug("Validation error: %s", exc.errors())
        return (
            jsonify({"detail": exc.errors(), "message": "Request validation failed"}),
            422,
        )
