# app/factory.py
"""Flask application factory.

CORS, the logging hook, the authentication layer, the blueprints, the error handlers
and the shared thread pool are wired here, so a caller — the server entry point, a test —
gets a fully configured application from one call.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_EXTRACT_IMAGES,
    debug_mode,
    log_request,
    logger,
)
from app.cors import CORSMiddleware
from app.errors import register_error_handlers
from app.middleware import AuthenticateBeforeApp, restore_user
from app.routes import docs_routes, document_routes, pgvector_routes
from app.utils.json_provider import JSONProvider


def create_app() -> Flask:
    # No static assets: the API serves JSON only.
    app = Flask(__name__, static_folder=None)
    app.debug = debug_mode
    app.json = JSONProvider(app)
    # Unhandled exceptions are answered by the app's own handler, in debug mode too.
    app.config["PROPAGATE_EXCEPTIONS"] = False

    app.before_request(restore_user)
    app.after_request(log_request)

    # Configuration the routes read.
    app.config["CHUNK_SIZE"] = CHUNK_SIZE
    app.config["CHUNK_OVERLAP"] = CHUNK_OVERLAP
    app.config["PDF_EXTRACT_IMAGES"] = PDF_EXTRACT_IMAGES

    # Bounded thread pool executor based on CPU cores
    max_workers = min(
        int(os.getenv("RAG_THREAD_POOL_SIZE", str(os.cpu_count()))), 8
    )  # Cap at 8
    app.extensions["thread_pool"] = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rag-worker"
    )
    logger.info(
        f"Initialized thread pool with {max_workers} workers (CPU cores: {os.cpu_count()})"
    )

    register_error_handlers(app)

    app.register_blueprint(docs_routes.bp)
    app.register_blueprint(document_routes.bp)
    if debug_mode:
        app.register_blueprint(pgvector_routes.bp)

    app.wsgi_app = CORSMiddleware(
        app.wsgi_app,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Authentication is the outermost layer, as it was: a request it refuses never
    # reaches CORS, logging or routing.
    app.wsgi_app = AuthenticateBeforeApp(app)

    return app
