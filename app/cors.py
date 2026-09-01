# app/cors.py
"""Cross-origin resource sharing, as this service always answered it.

A WSGI middleware carrying the policy the API has always had — any origin, any
method, any header, credentials allowed — with the header semantics its clients
already see: a simple response carries ``Access-Control-Allow-Origin: *`` unless
the request sends cookies, in which case the origin is echoed back and ``Vary:
Origin`` added; a preflight request is answered directly with a plain-text ``OK``.

It wraps the application rather than hooking into it, so it sits underneath the
authentication layer: a request refused there never reaches this middleware and
carries no CORS headers, which is the order this service has always used.
"""

from typing import Any, Callable, Dict, Iterable, List, Sequence

from werkzeug.datastructures import Headers
from werkzeug.wrappers import Response

ALL_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")
SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}


class CORSMiddleware:
    def __init__(
        self,
        app: Callable,
        allow_origins: Sequence[str] = (),
        allow_methods: Sequence[str] = ("GET",),
        allow_headers: Sequence[str] = (),
        allow_credentials: bool = False,
        expose_headers: Sequence[str] = (),
        max_age: int = 600,
    ) -> None:
        if "*" in allow_methods:
            allow_methods = ALL_METHODS

        allow_all_origins = "*" in allow_origins
        allow_all_headers = "*" in allow_headers
        preflight_explicit_allow_origin = not allow_all_origins or allow_credentials

        simple_headers: Dict[str, str] = {}
        if allow_all_origins:
            simple_headers["Access-Control-Allow-Origin"] = "*"
        if allow_credentials:
            simple_headers["Access-Control-Allow-Credentials"] = "true"
        if expose_headers:
            simple_headers["Access-Control-Expose-Headers"] = ", ".join(expose_headers)

        preflight_headers: Dict[str, str] = {}
        if preflight_explicit_allow_origin:
            # the origin is filled in by _preflight_response() when it is allowed
            preflight_headers["Vary"] = "Origin"
        else:
            preflight_headers["Access-Control-Allow-Origin"] = "*"
        preflight_headers.update(
            {
                "Access-Control-Allow-Methods": ", ".join(allow_methods),
                "Access-Control-Max-Age": str(max_age),
            }
        )
        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
        if allow_headers and not allow_all_headers:
            preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
        if allow_credentials:
            preflight_headers["Access-Control-Allow-Credentials"] = "true"

        self.app = app
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = [header.lower() for header in allow_headers]
        self.allow_all_origins = allow_all_origins
        self.allow_all_headers = allow_all_headers
        self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
        self.simple_headers = simple_headers
        self.preflight_headers = preflight_headers

    def __call__(self, environ: Dict[str, Any], start_response: Callable) -> Iterable[bytes]:
        origin = environ.get("HTTP_ORIGIN")
        if origin is None:
            return self.app(environ, start_response)

        if (
            environ.get("REQUEST_METHOD") == "OPTIONS"
            and "HTTP_ACCESS_CONTROL_REQUEST_METHOD" in environ
        ):
            return self._preflight_response(environ)(environ, start_response)

        return self._simple_response(environ, start_response, origin)

    def is_allowed_origin(self, origin: str) -> bool:
        return self.allow_all_origins or origin in self.allow_origins

    def _preflight_response(self, environ: Dict[str, Any]) -> Response:
        requested_origin = environ["HTTP_ORIGIN"]
        requested_method = environ["HTTP_ACCESS_CONTROL_REQUEST_METHOD"]
        requested_headers = environ.get("HTTP_ACCESS_CONTROL_REQUEST_HEADERS")

        headers = dict(self.preflight_headers)
        failures: List[str] = []

        if self.is_allowed_origin(origin=requested_origin):
            if self.preflight_explicit_allow_origin:
                # the "else" case is already covered by self.preflight_headers,
                # where the value is "*"
                headers["Access-Control-Allow-Origin"] = requested_origin
        else:
            failures.append("origin")

        if requested_method not in self.allow_methods:
            failures.append("method")

        # when every header is allowed, the requested ones are mirrored back
        if self.allow_all_headers and requested_headers is not None:
            headers["Access-Control-Allow-Headers"] = requested_headers
        elif requested_headers is not None:
            for header in [h.lower() for h in requested_headers.split(",")]:
                if header.strip() not in self.allow_headers:
                    failures.append("headers")
                    break

        if failures:
            return Response(
                "Disallowed CORS " + ", ".join(failures),
                status=400,
                headers=headers,
                mimetype="text/plain",
            )
        return Response("OK", status=200, headers=headers, mimetype="text/plain")

    def _simple_response(
        self, environ: Dict[str, Any], start_response: Callable, origin: str
    ) -> Iterable[bytes]:
        has_cookie = "HTTP_COOKIE" in environ

        def _start_response(status: str, response_headers: List, exc_info: Any = None) -> Callable:
            headers = Headers(response_headers)
            for key, value in self.simple_headers.items():
                headers[key] = value
            # a request carrying cookies is answered with its own origin, never "*"
            if self.allow_all_origins and has_cookie:
                self._allow_explicit_origin(headers, origin)
            elif not self.allow_all_origins and self.is_allowed_origin(origin=origin):
                self._allow_explicit_origin(headers, origin)
            return start_response(status, headers.to_wsgi_list(), exc_info)

        return self.app(environ, _start_response)

    @staticmethod
    def _allow_explicit_origin(headers: Headers, origin: str) -> None:
        headers["Access-Control-Allow-Origin"] = origin
        vary = headers.get("Vary")
        headers["Vary"] = f"{vary}, Origin" if vary else "Origin"
