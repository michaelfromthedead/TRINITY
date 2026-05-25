"""
Atomic operations for lock-free programming.
Uses locks internally for correctness across Python implementations.
"""
import threading
from typing import TypeVar, Generic, Optional

T = TypeVar('T')


class AtomicInt:
    """Atomic integer with compare-exchange and fetch-add operations."""

    __slots__ = ('_value', '_lock')

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def load(self) -> int:
        """Atomically load value."""
        with self._lock:
            return self._value

    def store(self, value: int):
        """Atomically store value."""
        with self._lock:
            self._value = value

    def exchange(self, value: int) -> int:
        """Atomically exchange value and return old value."""
        with self._lock:
            old = self._value
            self._value = value
            return old

    def compare_exchange(self, expected: int, desired: int) -> tuple[bool, int]:
        """
        Compare-and-swap operation.

        Returns:
            (success, actual_value) tuple
        """
        with self._lock:
            actual = self._value
            if actual == expected:
                self._value = desired
                return (True, desired)
            return (False, actual)

    def fetch_add(self, delta: int) -> int:
        """Atomically add delta and return old value."""
        with self._lock:
            old = self._value
            self._value += delta
            return old

    def fetch_sub(self, delta: int) -> int:
        """Atomically subtract delta and return old value."""
        with self._lock:
            old = self._value
            self._value -= delta
            return old

    def add_fetch(self, delta: int) -> int:
        """Atomically add delta and return new value."""
        with self._lock:
            self._value += delta
            return self._value

    def sub_fetch(self, delta: int) -> int:
        """Atomically subtract delta and return new value."""
        with self._lock:
            self._value -= delta
            return self._value

    def increment(self) -> int:
        """Atomically increment and return new value."""
        return self.add_fetch(1)

    def decrement(self) -> int:
        """Atomically decrement and return new value."""
        return self.sub_fetch(1)


class AtomicFloat:
    """Atomic float operations."""

    __slots__ = ('_value', '_lock')

    def __init__(self, initial: float = 0.0):
        self._value = initial
        self._lock = threading.Lock()

    def load(self) -> float:
        """Atomically load value."""
        with self._lock:
            return self._value

    def store(self, value: float):
        """Atomically store value."""
        with self._lock:
            self._value = value

    def exchange(self, value: float) -> float:
        """Atomically exchange value and return old value."""
        with self._lock:
            old = self._value
            self._value = value
            return old

    def compare_exchange(self, expected: float, desired: float) -> tuple[bool, float]:
        """
        Compare-and-swap operation.

        Returns:
            (success, actual_value) tuple
        """
        with self._lock:
            actual = self._value
            if actual == expected:
                self._value = desired
                return (True, desired)
            return (False, actual)

    def fetch_add(self, delta: float) -> float:
        """Atomically add delta and return old value."""
        with self._lock:
            old = self._value
            self._value += delta
            return old

    def fetch_sub(self, delta: float) -> float:
        """Atomically subtract delta and return old value."""
        with self._lock:
            old = self._value
            self._value -= delta
            return old


class AtomicBool:
    """Atomic boolean operations."""

    __slots__ = ('_value', '_lock')

    def __init__(self, initial: bool = False):
        self._value = initial
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Atomically load value."""
        with self._lock:
            return self._value

    def store(self, value: bool):
        """Atomically store value."""
        with self._lock:
            self._value = value

    def exchange(self, value: bool) -> bool:
        """Atomically exchange value and return old value."""
        with self._lock:
            old = self._value
            self._value = value
            return old

    def compare_exchange(self, expected: bool, desired: bool) -> tuple[bool, bool]:
        """
        Compare-and-swap operation.

        Returns:
            (success, actual_value) tuple
        """
        with self._lock:
            actual = self._value
            if actual == expected:
                self._value = desired
                return (True, desired)
            return (False, actual)

    def test_and_set(self) -> bool:
        """Set to True and return old value."""
        return self.exchange(True)

    def clear(self):
        """Set to False."""
        self.store(False)


class AtomicRef(Generic[T]):
    """Atomic reference to an object."""

    __slots__ = ('_value', '_lock')

    def __init__(self, initial: Optional[T] = None):
        self._value = initial
        self._lock = threading.Lock()

    def load(self) -> Optional[T]:
        """Atomically load reference."""
        with self._lock:
            return self._value

    def store(self, value: Optional[T]):
        """Atomically store reference."""
        with self._lock:
            self._value = value

    def exchange(self, value: Optional[T]) -> Optional[T]:
        """Atomically exchange reference and return old value."""
        with self._lock:
            old = self._value
            self._value = value
            return old

    def compare_exchange(self, expected: Optional[T], desired: Optional[T]) -> tuple[bool, Optional[T]]:
        """
        Compare-and-swap operation using identity comparison.

        Returns:
            (success, actual_value) tuple
        """
        with self._lock:
            actual = self._value
            if actual is expected:
                self._value = desired
                return (True, desired)
            return (False, actual)
