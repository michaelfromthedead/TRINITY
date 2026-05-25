"""TaskScheduler — parallel job execution with thread pool.

Wraps :class:`concurrent.futures.ThreadPoolExecutor` and the
:mod:`engine.core.tasks.worker` work-stealing pool to provide
submit / wait / parallel_for semantics.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, Future as CFuture
from typing import Any, Callable, List, Optional, Sequence

from engine.core.tasks.sync import Future
from engine.core.tasks.worker import TaskPriority

logger = logging.getLogger(__name__)


class TaskHandle:
    """Opaque handle returned by :meth:`TaskScheduler.submit`."""

    def __init__(self, cf_future: CFuture, priority: TaskPriority = TaskPriority.NORMAL) -> None:
        self._future = cf_future
        self._priority = priority

    @property
    def priority(self) -> TaskPriority:
        return self._priority

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout=timeout)


class TaskScheduler:
    """Manages a thread pool for parallel job execution.

    Parameters
    ----------
    worker_count : int
        Number of worker threads.  ``0`` means auto-detect
        (``os.cpu_count() - 1``, minimum 1).
    """

    def __init__(self, worker_count: int = 0) -> None:
        self._pool: Optional[ThreadPoolExecutor] = None
        self._worker_count = 0
        self._initialized = False
        if worker_count != 0:
            self.initialize(worker_count)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, worker_count: int = 0) -> None:
        """Start the internal thread pool.

        Args:
            worker_count: 0 means auto-detect (cpu_count - 1, min 1).
        """
        if self._initialized:
            return
        if worker_count <= 0:
            cpu = os.cpu_count() or 2
            worker_count = max(1, cpu - 1)
        self._worker_count = worker_count
        self._pool = ThreadPoolExecutor(max_workers=worker_count)
        self._initialized = True
        logger.info("TaskScheduler initialized with %d workers", worker_count)

    def shutdown(self, wait: bool = True) -> None:
        """Stop worker threads and release resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None
        self._initialized = False
        logger.info("TaskScheduler shut down")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self.initialize()
        assert self._pool is not None
        return self._pool

    def submit(
        self,
        func: Callable,
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> TaskHandle:
        """Submit a callable for execution.  Returns a :class:`TaskHandle`."""
        pool = self._ensure_pool()
        cf = pool.submit(func, *args)
        return TaskHandle(cf, priority=priority)

    def submit_after(
        self,
        func: Callable,
        dependencies: Sequence[TaskHandle],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> TaskHandle:
        """Submit *func* to run only after all *dependencies* complete.

        Uses continuation-passing (add_done_callback) instead of blocking
        a pool thread, avoiding thread starvation under load.
        """
        pool = self._ensure_pool()
        if not dependencies:
            cf = pool.submit(func, *args)
            return TaskHandle(cf, priority=priority)

        deps = list(dependencies)
        from threading import Lock
        lock = Lock()
        remaining = [len(deps)]
        cf: CFuture = CFuture()

        def _on_one_done(_: Any) -> None:
            with lock:
                remaining[0] -= 1
                if remaining[0] > 0:
                    return
            try:
                for d in deps:
                    d.result()
                result = func(*args)
                cf.set_result(result)
            except BaseException as exc:
                cf.set_exception(exc)

        for d in deps:
            d._future.add_done_callback(_on_one_done)

        return TaskHandle(cf, priority=priority)

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def wait(self, handle: TaskHandle, timeout: Optional[float] = None) -> Any:
        """Block until *handle* completes and return its result."""
        return handle.result(timeout=timeout)

    def wait_all(
        self, handles: Sequence[TaskHandle], timeout: Optional[float] = None
    ) -> List[Any]:
        """Block until all *handles* complete.  Returns list of results."""
        return [h.result(timeout=timeout) for h in handles]

    def is_complete(self, handle: TaskHandle) -> bool:
        """Return True if *handle* has finished executing."""
        return handle.done()

    # ------------------------------------------------------------------
    # Parallel patterns
    # ------------------------------------------------------------------

    def parallel_for(
        self,
        count: int,
        chunk_size: int,
        func: Callable[[int, int], Any],
    ) -> TaskHandle:
        """Split *count* iterations into chunks and run in parallel.

        *func(start, end)* is called for each chunk where ``start`` is
        inclusive and ``end`` is exclusive.

        Returns a :class:`TaskHandle` that completes when all chunks finish.
        """
        if count <= 0:
            # Return an already-done handle
            pool = self._ensure_pool()
            cf = pool.submit(lambda: None)
            return TaskHandle(cf)

        chunk_size = max(1, chunk_size)
        num_chunks = math.ceil(count / chunk_size)

        handles: list[TaskHandle] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, count)
            handles.append(self.submit(func, start, end))

        # Aggregate handle
        def _wait_all() -> None:
            for h in handles:
                h.result()

        pool = self._ensure_pool()
        cf = pool.submit(_wait_all)
        return TaskHandle(cf)
