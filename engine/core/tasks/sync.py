"""Synchronization primitives for the task system.

Provides TaskCounter, Future/Promise, Latch, and Barrier — all built
on top of ``threading`` primitives.
"""

from __future__ import annotations

import threading
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# TaskCounter
# ---------------------------------------------------------------------------

class TaskCounter:
    """Atomic counter that supports waiting until the value reaches zero."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()
        self._zero_event = threading.Event()
        if initial == 0:
            self._zero_event.set()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def increment(self, n: int = 1) -> None:
        with self._lock:
            self._value += n
            if self._value != 0:
                self._zero_event.clear()

    def decrement(self, n: int = 1) -> None:
        with self._lock:
            self._value -= n
            if self._value <= 0:
                self._value = max(self._value, 0)
                self._zero_event.set()

    def wait_until_zero(self, timeout: Optional[float] = None) -> bool:
        """Block until counter reaches zero. Returns True if reached zero."""
        return self._zero_event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Future / Promise
# ---------------------------------------------------------------------------

class Future(Generic[T]):
    """Read-side of an async result.  Wraps a simple event + value store."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Optional[T] = None
        self._exception: Optional[BaseException] = None
        self._done = False

    def is_ready(self) -> bool:
        return self._done

    def get(self, timeout: Optional[float] = None) -> T:
        """Block until result is available, then return it (or raise)."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Future.get() timed out")
        if self._exception is not None:
            raise self._exception
        return self._result  # type: ignore[return-value]

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout=timeout)

    # -- internal setters (called by Promise / worker) --

    def _set_result(self, value: T) -> None:
        self._result = value
        self._done = True
        self._event.set()

    def _set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._done = True
        self._event.set()


class Promise(Generic[T]):
    """Write-side of an async result.  Owns a :class:`Future`."""

    def __init__(self) -> None:
        self._future: Future[T] = Future()

    @property
    def future(self) -> Future[T]:
        return self._future

    def set_value(self, value: T) -> None:
        self._future._set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        self._future._set_exception(exc)


# ---------------------------------------------------------------------------
# Latch  (one-shot, count-down)
# ---------------------------------------------------------------------------

class Latch:
    """One-shot barrier: count down from *count* and release all waiters."""

    def __init__(self, count: int) -> None:
        if count < 0:
            raise ValueError("Latch count must be >= 0")
        self._count = count
        self._lock = threading.Lock()
        self._event = threading.Event()
        if count == 0:
            self._event.set()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def count_down(self, n: int = 1) -> None:
        with self._lock:
            self._count = max(0, self._count - n)
            if self._count == 0:
                self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until count reaches zero.  Returns True if latch opened."""
        return self._event.wait(timeout=timeout)

    def try_wait(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Barrier  (reusable)
# ---------------------------------------------------------------------------

class Barrier:
    """Reusable barrier for *count* participants."""

    def __init__(self, count: int) -> None:
        if count < 1:
            raise ValueError("Barrier count must be >= 1")
        self._parties = count
        self._count = 0
        self._generation = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def parties(self) -> int:
        return self._parties

    def arrive_and_wait(self, timeout: Optional[float] = None) -> int:
        """Arrive and block until all parties have arrived.

        Returns the arrival index (0-based).
        """
        with self._cond:
            gen = self._generation
            idx = self._count
            self._count += 1
            if self._count >= self._parties:
                # Last to arrive — release everyone and reset
                self._count = 0
                self._generation += 1
                self._cond.notify_all()
                return idx
            # Wait for this generation to complete
            self._cond.wait_for(
                lambda: self._generation != gen, timeout=timeout
            )
            return idx
