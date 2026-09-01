# app/async_runner.py
"""One persistent event loop for the whole process.

Flask serves each request from a WSGI worker thread that has no event loop of
its own, but this application's storage layer is asyncio and keeps loop-bound
state *between* requests: an asyncpg pool belongs to the loop that created it,
and the batched ingestion pipeline coordinates through ``asyncio`` primitives.
Giving every request a fresh loop would invalidate that state, so the process
keeps a single loop alive on a background thread and dispatches coroutines onto
it.

One loop, shared pools, and blocking work handed to an executor: the model the
asyncio code below is written against, made available to a WSGI server.
"""

import asyncio
import contextvars
import threading
from concurrent.futures import Future
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, TypeVar

T = TypeVar("T")


class AsyncRunner:
    """Runs coroutines on a shared background event loop."""

    def __init__(self, thread_name: str = "rag-event-loop") -> None:
        self._thread_name = thread_name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self.start()
        assert self._loop is not None
        return self._loop

    def start(self) -> None:
        with self._lock:
            if self._loop is not None:
                return

            loop = asyncio.new_event_loop()
            running = threading.Event()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(running.set)
                loop.run_forever()

            thread = threading.Thread(target=_run, name=self._thread_name, daemon=True)
            thread.start()
            running.wait()

            self._loop = loop
            self._thread = thread

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the shared loop and return its result.

        The caller's :mod:`contextvars` context is carried across, so Flask's
        request-scoped globals (``request``, ``g``, ``current_app``) resolve
        inside the coroutine exactly as they do in the worker thread.
        """
        loop = self.loop
        context = contextvars.copy_context()
        result: "Future[T]" = Future()
        result.set_running_or_notify_cancel()

        def _relay(task: "asyncio.Task[T]") -> None:
            try:
                result.set_result(task.result())
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                result.set_exception(exc)

        def _start() -> None:
            loop.create_task(coro, context=context).add_done_callback(_relay)

        loop.call_soon_threadsafe(_start)
        return result.result()

    def shutdown(self) -> None:
        """Stop the loop and join its thread."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None

        if loop is None:
            return

        async def _drain() -> None:
            pending = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not asyncio.current_task(loop)
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await loop.shutdown_asyncgens()

        asyncio.run_coroutine_threadsafe(_drain(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join()
        loop.close()


runner = AsyncRunner()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on the application's shared event loop."""
    return runner.run(coro)


def async_route(view: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Adapt an ``async def`` view so Flask can dispatch it."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_async(view(*args, **kwargs))

    return wrapper
