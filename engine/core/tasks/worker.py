"""Worker thread management with work-stealing deques.

Provides a WorkerPool that manages N worker threads, each with a local
deque. Workers process their own queue first, then steal from random
peers when idle.
"""

from __future__ import annotations

import enum
import logging

from engine.core.constants import WORKER_IDLE_POLL_INTERVAL
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskPriority(enum.IntEnum):
    """Task execution priority. Lower value == higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


class TaskAffinity(enum.Enum):
    """Which thread(s) may execute a task."""
    ANY = "any"
    MAIN = "main"
    WORKER = "worker"
    IO = "io"


@dataclass(order=True)
class WorkItem:
    """A unit of work submitted to the pool."""
    priority: int
    seq: int = field(compare=True)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    affinity: TaskAffinity = field(default=TaskAffinity.ANY, compare=False)
    future: Any = field(default=None, compare=False)


class WorkerThread:
    """A single worker with a local work-stealing deque."""

    def __init__(self, worker_id: int, pool: WorkerPool) -> None:
        self._id = worker_id
        self._pool = pool
        self._local_queue: deque[WorkItem] = deque()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def worker_id(self) -> int:
        return self._id

    def push(self, item: WorkItem) -> None:
        """Push work to this worker's local deque (owner side)."""
        with self._lock:
            self._local_queue.append(item)

    def pop(self) -> Optional[WorkItem]:
        """Pop from the back (LIFO) — owner side."""
        with self._lock:
            if self._local_queue:
                return self._local_queue.pop()
        return None

    def steal(self) -> Optional[WorkItem]:
        """Steal from the front (FIFO) — thief side."""
        with self._lock:
            if self._local_queue:
                return self._local_queue.popleft()
        return None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"Worker-{self._id}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        logger.debug("Worker-%d started", self._id)
        while not self._pool._shutdown_event.is_set():
            # Clear before checking so new submits can re-set it
            self._pool._work_available.clear()
            item = self._get_work()
            if item is not None:
                self._execute(item)
            else:
                # Wait briefly before retrying
                self._pool._work_available.wait(timeout=WORKER_IDLE_POLL_INTERVAL)

    def _get_work(self) -> Optional[WorkItem]:
        # Try own queue first
        item = self.pop()
        if item is not None:
            return item
        # Steal from a random peer
        return self._pool._steal_from_random(self._id)

    def _execute(self, item: WorkItem) -> None:
        try:
            result = item.func(*item.args, **item.kwargs)
            if item.future is not None:
                item.future._set_result(result)
        except Exception as exc:
            logger.error("Worker-%d task failed: %s", self._id, exc)
            if item.future is not None:
                item.future._set_exception(exc)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class WorkerPool:
    """Manages N worker threads with work-stealing scheduling."""

    def __init__(self, num_workers: int) -> None:
        if num_workers < 1:
            num_workers = 1
        self._workers: list[WorkerThread] = []
        self._shutdown_event = threading.Event()
        self._work_available = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._running = False

        for i in range(num_workers):
            self._workers.append(WorkerThread(i, self))

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start all worker threads."""
        if self._running:
            return
        self._shutdown_event.clear()
        self._running = True
        for w in self._workers:
            w.start()
        logger.info("WorkerPool started with %d workers", len(self._workers))

    def shutdown(self, timeout: float = 2.0) -> None:
        """Signal shutdown and join all workers."""
        if not self._running:
            return
        self._shutdown_event.set()
        self._work_available.set()  # wake sleeping workers
        for w in self._workers:
            w.join(timeout=timeout)
        self._running = False
        logger.info("WorkerPool shut down")

    def submit(self, item: WorkItem) -> None:
        """Submit a work item, distributing to the least-loaded worker."""
        with self._seq_lock:
            item.seq = self._seq
            self._seq += 1

        # Simple round-robin based on seq
        target = item.seq % len(self._workers)
        self._workers[target].push(item)
        self._work_available.set()

    def _steal_from_random(self, exclude_id: int) -> Optional[WorkItem]:
        """Try to steal work from a random worker other than *exclude_id*."""
        if len(self._workers) <= 1:
            return None
        candidates = [w for w in self._workers if w.worker_id != exclude_id]
        random.shuffle(candidates)
        for w in candidates:
            item = w.steal()
            if item is not None:
                return item
        return None
