# app/middleware.py
import os
import jwt
from jwt import PyJWTError
from datetime import datetime, timezone
from flask import g, jsonify, request
from app.config import logger


def security_middleware():
    """Authenticate the caller before the view runs.

    Registered with ``Flask.before_request``: returning ``None`` lets the request
    continue to its view, returning a response short-circuits it.
    """
    if request.path in {"/docs", "/openapi.json", "/health"}:
        return None

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warn("JWT_SECRET not found in environment variables")
        return None

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(
            f"Unauthorized request with missing or invalid Authorization header to: {request.path}"
        )
        return (
            jsonify({"detail": "Missing or invalid Authorization header"}),
            401,
        )

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(tz=timezone.utc) > datetime.fromtimestamp(
            exp_timestamp, tz=timezone.utc
        ):
            logger.info(f"Unauthorized request with expired token to: {request.path}")
            return jsonify({"detail": "Token has expired"}), 401

        g.user = payload
        logger.debug(f"{request.path} - {payload}")
    except PyJWTError as e:
        logger.info(
            f"Unauthorized request with invalid token to: {request.path}, reason: {str(e)}"
        )
        return jsonify({"detail": f"Invalid token: {str(e)}"}), 401

    return None


#: WSGI environ key carrying the verified token payload into the application.
USER_ENVIRON_KEY = "rag_api.user"


class AuthenticateBeforeApp:
    """Run :func:`security_middleware` outside the Flask app, ahead of CORS and logging.

    The FastAPI application nested its middleware with the authentication decision
    outermost: a request it refused never reached the CORS or the logging layer, so a
    401 (including the answer to a CORS preflight) carried no CORS headers. Flask's
    request hooks cannot nest like that - every ``after_request`` hook, flask-cors'
    included, still runs on a response a ``before_request`` hook returned - so the
    check wraps the WSGI application instead.
    """

    def __init__(self, app):
        self.app = app
        self.wsgi_app = app.wsgi_app

    def __call__(self, environ, start_response):
        with self.app.request_context(environ):
            rejection = security_middleware()
            if rejection is not None:
                return self.app.make_response(rejection)(environ, start_response)
            environ[USER_ENVIRON_KEY] = g.get("user")
        return self.wsgi_app(environ, start_response)


def restore_user():
    """``before_request`` hook: expose the payload verified above as ``g.user``."""
    user = request.environ.get(USER_ENVIRON_KEY)
    if user is not None:
        g.user = user
    return None
