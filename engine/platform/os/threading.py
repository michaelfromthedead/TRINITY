"""
Threading primitives: threads, mutexes, RW locks, semaphores, barriers.
"""
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Any


class ThreadPriority(Enum):
    """Thread priority levels (best-effort on Linux)."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3


@dataclass(slots=True)
class ThreadConfig:
    """Thread configuration."""
    name: Optional[str] = None
    affinity: Optional[list[int]] = None  # CPU cores
    priority: ThreadPriority = ThreadPriority.NORMAL
    daemon: bool = False


class Thread:
    """Cross-platform thread wrapper with affinity and priority support."""

    def __init__(self, target: Callable, args: tuple = (), config: Optional[ThreadConfig] = None):
        self.config = config or ThreadConfig()
        self._target = target
        self._args = args
        self._thread = threading.Thread(
            target=self._run_wrapper,
            name=self.config.name,
            daemon=self.config.daemon
        )
        self._running = False

    def _run_wrapper(self):
        """Wrapper to apply affinity and priority before running target."""
        # Set CPU affinity (Linux-specific)
        if self.config.affinity is not None:
            try:
                os.sched_setaffinity(0, self.config.affinity)
            except (AttributeError, OSError):
                pass  # Not supported on this platform

        # Priority is best-effort on most systems without elevated permissions
        # Would need setpriority/nice on Linux (requires root for realtime)

        self._running = True
        try:
            self._target(*self._args)
        finally:
            self._running = False

    def start(self):
        """Start the thread."""
        self._thread.start()

    def join(self, timeout: Optional[float] = None):
        """Wait for thread to complete."""
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Check if thread is still running."""
        return self._thread.is_alive()

    @property
    def ident(self) -> Optional[int]:
        """Get thread identifier."""
        return self._thread.ident


class Mutex:
    """Mutual exclusion lock with try_lock support."""

    __slots__ = ('_lock',)

    def __init__(self):
        self._lock = threading.Lock()

    def lock(self):
        """Acquire the lock, blocking if necessary."""
        self._lock.acquire()

    def unlock(self):
        """Release the lock."""
        self._lock.release()

    def try_lock(self) -> bool:
        """Try to acquire lock without blocking."""
        return self._lock.acquire(blocking=False)

    def try_lock_for(self, timeout: float) -> bool:
        """Try to acquire lock with timeout in seconds."""
        return self._lock.acquire(blocking=True, timeout=timeout)

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, *args):
        self.unlock()


class RWLock:
    """Readers-writer lock implementation."""

    __slots__ = ('_readers', '_writers', '_read_ready', '_write_ready', '_lock')

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._write_ready = threading.Condition(threading.Lock())
        self._lock = threading.Lock()

    def acquire_read(self):
        """Acquire read lock. Multiple readers allowed."""
        with self._read_ready:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        """Release read lock."""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        # Notify writers OUTSIDE the read lock to avoid nested lock
        if self._readers == 0:
            with self._write_ready:
                self._write_ready.notify()

    def acquire_write(self):
        """Acquire write lock. Exclusive access."""
        with self._write_ready:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers += 1

    def release_write(self):
        """Release write lock."""
        with self._write_ready:
            self._writers -= 1
            self._write_ready.notify()
            with self._read_ready:
                self._read_ready.notify_all()

    class ReadContext:
        """Context manager for read lock."""
        def __init__(self, rwlock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.acquire_read()
            return self

        def __exit__(self, *args):
            self.rwlock.release_read()

    class WriteContext:
        """Context manager for write lock."""
        def __init__(self, rwlock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.acquire_write()
            return self

        def __exit__(self, *args):
            self.rwlock.release_write()

    def read(self):
        """Get read context manager."""
        return self.ReadContext(self)

    def write(self):
        """Get write context manager."""
        return self.WriteContext(self)


class Semaphore:
    """Counting semaphore."""

    __slots__ = ('_sem',)

    def __init__(self, value: int = 1):
        self._sem = threading.Semaphore(value)

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore."""
        return self._sem.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        """Release semaphore."""
        self._sem.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class CondVar:
    """Condition variable."""

    __slots__ = ('_cond',)

    def __init__(self, lock: Optional[Mutex] = None):
        underlying_lock = lock._lock if lock else None
        self._cond = threading.Condition(underlying_lock)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for notification."""
        return self._cond.wait(timeout=timeout)

    def notify(self, n: int = 1):
        """Wake up n waiting threads."""
        self._cond.notify(n)

    def notify_all(self):
        """Wake up all waiting threads."""
        self._cond.notify_all()

    def __enter__(self):
        self._cond.__enter__()
        return self

    def __exit__(self, *args):
        self._cond.__exit__(*args)


class Barrier:
    """Synchronization barrier for multiple threads."""

    __slots__ = ('_barrier',)

    def __init__(self, parties: int):
        self._barrier = threading.Barrier(parties)

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait at barrier until all parties arrive."""
        return self._barrier.wait(timeout=timeout)

    def reset(self):
        """Reset barrier to initial state."""
        self._barrier.reset()

    def abort(self):
        """Put barrier in broken state."""
        self._barrier.abort()

    @property
    def parties(self) -> int:
        """Number of threads required to trip barrier."""
        return self._barrier.parties

    @property
    def n_waiting(self) -> int:
        """Number of threads currently waiting."""
        return self._barrier.n_waiting

    @property
    def broken(self) -> bool:
        """Whether barrier is in broken state."""
        return self._barrier.broken


class ThreadLocalStorage:
    """Thread-local storage wrapper."""

    __slots__ = ('_local',)

    def __init__(self):
        self._local = threading.local()

    def get(self, key: str, default: Any = None) -> Any:
        """Get thread-local value."""
        return getattr(self._local, key, default)

    def set(self, key: str, value: Any):
        """Set thread-local value."""
        setattr(self._local, key, value)

    def delete(self, key: str):
        """Delete thread-local value."""
        if hasattr(self._local, key):
            delattr(self._local, key)

    def has(self, key: str) -> bool:
        """Check if thread-local value exists."""
        return hasattr(self._local, key)
