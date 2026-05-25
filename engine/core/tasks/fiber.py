"""Coroutine / fiber support — thin wrapper around asyncio.

Python's ``async/await`` is essentially cooperative fiber scheduling.
This module provides a :class:`Fiber` wrapper and :class:`FiberScheduler`
to run async tasks on an event loop from synchronous code.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from engine.core.constants import FIBER_JOIN_TIMEOUT
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class Fiber:
    """A lightweight cooperative task wrapping an asyncio coroutine.

    Supports yield/resume semantics via an internal :class:`asyncio.Event`.
    """

    def __init__(self, coro: Coroutine) -> None:
        self._coro = coro
        self._task: Optional[asyncio.Task] = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # start in resumed state
        self._result: Any = None
        self._done = False

    @property
    def done(self) -> bool:
        if self._task is not None:
            return self._task.done()
        return self._done

    @property
    def result(self) -> Any:
        if self._task is not None and self._task.done():
            return self._task.result()
        return self._result

    async def yield_(self) -> None:
        """Suspend this fiber until :meth:`resume` is called."""
        self._resume_event.clear()
        await self._resume_event.wait()

    def resume(self) -> None:
        """Resume a suspended fiber."""
        self._resume_event.set()

    def _bind(self, task: asyncio.Task) -> None:
        self._task = task


class FiberScheduler:
    """Runs :class:`Fiber` instances on a dedicated asyncio event loop.

    The loop runs in a background thread so callers can use it from
    synchronous code.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background event loop thread."""
        if self._running:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="FiberScheduler", daemon=True
        )
        self._running = True
        self._thread.start()
        logger.info("FiberScheduler started")

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """Stop the event loop and join the thread."""
        if not self._running or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=FIBER_JOIN_TIMEOUT)
        self._loop.close()
        self._loop = None
        self._running = False
        logger.info("FiberScheduler stopped")

    def spawn(self, fiber: Fiber) -> Fiber:
        """Schedule *fiber* for execution on the event loop."""
        if self._loop is None:
            self.start()
        assert self._loop is not None

        future = asyncio.run_coroutine_threadsafe(fiber._coro, self._loop)
        # Wrap the concurrent.futures.Future in an asyncio.Task-like interface
        # by storing it on the fiber.
        fiber._task = None  # type: ignore
        fiber._done = False

        def _on_done(f: Any) -> None:
            fiber._done = True
            try:
                fiber._result = f.result()
            except Exception:
                logger.exception("Fiber coroutine raised an exception")

        future.add_done_callback(_on_done)
        return fiber

    def run_sync(self, coro: Coroutine) -> Any:
        """Run a coroutine and block until it completes."""
        if self._loop is None:
            self.start()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
