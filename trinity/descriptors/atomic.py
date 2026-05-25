"""
Atomic descriptor - thread-safe field access with compare-and-swap.

Provides thread-safe get/set operations using per-field locks
and an atomic compare-and-swap utility.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class AtomicDescriptor(BaseDescriptor[T]):
    """
    Thread-safe descriptor using per-field locks.

    Every get and set operation acquires a threading.Lock stored
    on the instance, ensuring atomicity for concurrent access.
    """

    __slots__ = ("_lock_key",)

    descriptor_id = "atomic"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._lock_key: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._lock_key = f"_lock_{name}"

    def _get_lock(self, obj: Any) -> threading.Lock:
        """Get or create the lock for this field on the given object.

        Uses ``setdefault`` which is atomic under CPython's GIL,
        eliminating the check-then-create race condition.
        """
        return obj.__dict__.setdefault(self._lock_key, threading.Lock())

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        if obj is None:
            return self  # type: ignore
        # If already inside CAS, skip re-acquiring (CAS holds the lock).
        if getattr(obj, "_atomic_in_cas", False):
            return super().__get__(obj, objtype)
        lock = self._get_lock(obj)
        with lock:
            return super().__get__(obj, objtype)

    def __set__(self, obj: Any, value: T) -> None:
        # If already inside CAS, skip re-acquiring (CAS holds the lock).
        if getattr(obj, "_atomic_in_cas", False):
            super().__set__(obj, value)
            return
        lock = self._get_lock(obj)
        with lock:
            super().__set__(obj, value)

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.INTERCEPT, {"get": "atomic_get", "set": "atomic_set"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["lock_key"] = self._lock_key
        return meta


def compare_and_swap(obj: Any, field: str, expected: Any, new: Any) -> bool:
    """
    Atomic compare-and-swap on a descriptor field.

    Acquires the field lock, checks if current value equals expected,
    and if so sets the new value. Returns True on success, False otherwise.
    """
    # Get the descriptor from the class
    descriptor = None
    for cls in type(obj).__mro__:
        if field in cls.__dict__:
            descriptor = cls.__dict__[field]
            break

    if not isinstance(descriptor, AtomicDescriptor):
        raise TypeError(f"Field '{field}' is not an AtomicDescriptor")

    lock = descriptor._get_lock(obj)
    with lock:
        # Set flag so descriptor.__get__/__set__ skip re-acquiring the lock
        # but still run the full lifecycle chain (pre_get, post_get, etc.).
        obj._atomic_in_cas = True
        try:
            current = descriptor.__get__(obj, type(obj))
            if current == expected:
                descriptor.__set__(obj, new)
                return True
            return False
        finally:
            obj._atomic_in_cas = False


__all__ = ["AtomicDescriptor", "compare_and_swap"]
