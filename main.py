# main.py
import signal
from contextlib import contextmanager

from waitress import create_server

from app.async_runner import run_async, runner
from app.config import (
    VectorDBType,
    RAG_HOST,
    RAG_PORT,
    VECTOR_DB_TYPE,
    logger,
    vector_store,
)
from app.factory import create_app
from app.services.database import PSQLDatabase, ensure_vector_indexes
from app.services.vector_store.factory import close_vector_store_connections

app = create_app()


@contextmanager
def lifespan(flask_app):
    """Bracket the serving loop with the application's startup and shutdown.

    WSGI has no lifespan protocol, so the entry point below wraps the call that
    serves requests: the pool and indexes are ready before the first request,
    and every backing resource is drained after the last one.
    """
    # Startup logic goes here
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        run_async(PSQLDatabase.get_pool())  # Initialize the pool
        run_async(ensure_vector_indexes())

    try:
        yield flask_app
    finally:
        # Cleanup logic
        if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
            try:
                logger.info("Closing asyncpg connection pool")
                run_async(PSQLDatabase.close_pool())
                logger.info("asyncpg connection pool closed")
            except Exception as e:
                logger.warning("Failed to close asyncpg pool: %s", e)

        # Drain in-flight work before closing backing resources
        logger.info("Shutting down thread pool")
        flask_app.extensions["thread_pool"].shutdown(wait=True)
        logger.info("Thread pool shutdown complete")

        # Close vector store connections (MongoDB client / SQLAlchemy engine)
        try:
            close_vector_store_connections(vector_store)
        except Exception as e:
            logger.warning("Failed to close vector store connections: %s", e)

        runner.shutdown()


def _serve_until_signalled(flask_app, host: str, port: int) -> None:
    """Serve requests, returning on SIGINT/SIGTERM so cleanup can run.

    Waitress leaves SIGTERM at its default disposition, which would kill the
    process outright and skip the shutdown half of ``lifespan``.
    """
    server = create_server(flask_app, host=host, port=port)

    def stop(signum, _frame):
        logger.info("Received signal %s, shutting down", signum)
        # ``server.run()`` treats SystemExit as its stop signal and closes the
        # listening socket on the way out.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)

    logger.info("Serving on http://%s:%s", host, port)
    server.run()


if __name__ == "__main__":
    with lifespan(app):
        _serve_until_signalled(app, RAG_HOST, RAG_PORT)
