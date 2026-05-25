"""
Caching descriptors - cache computed values with optional TTL.

Provides lazy evaluation, caching, and computed fields.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class CachedDescriptor(BaseDescriptor[T]):
    """
    Caches computed values with optional TTL.

    Wraps a computed descriptor to add caching behavior.
    """

    __slots__ = ("_ttl", "_cache_key", "_time_key")

    descriptor_id = "cached"
    accepts_inner = ("computed",)
    accepts_outer = ("*",)
    excludes = ("tracked",)  # Cached values aren't tracked the same way

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        ttl: Optional[float] = None,
        **config: Any,
    ) -> None:
        """
        Initialize caching descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap (should be computed).
            ttl: Time-to-live in seconds (None = forever).
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        self._ttl = ttl
        self._cache_key = ""
        self._time_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Set up cache keys."""
        super().__set_name__(owner, name)
        self._cache_key = f"_cache_{name}"
        self._time_key = f"_cache_time_{name}"

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Get cached value, recomputing if necessary."""
        if obj is None:
            return self  # type: ignore

        now = time.time()

        # Check cache validity
        if self._cache_key in obj.__dict__:
            if self._ttl is None:
                return obj.__dict__[self._cache_key]

            cache_time = obj.__dict__.get(self._time_key, 0)
            if (now - cache_time) < self._ttl:
                return obj.__dict__[self._cache_key]

        # Recompute via inner descriptor
        value = super().__get__(obj, objtype)
        obj.__dict__[self._cache_key] = value
        obj.__dict__[self._time_key] = now
        return value

    def invalidate(self, obj: Any) -> None:
        """Invalidate the cached value."""
        obj.__dict__.pop(self._cache_key, None)
        obj.__dict__.pop(self._time_key, None)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "cache_check"}),
                Step(Op.TAG, {"key": "ttl", "value": self._ttl})]

    def get_metadata(self) -> dict[str, Any]:
        """Return caching configuration."""
        meta = super().get_metadata()
        meta["ttl"] = self._ttl
        return meta


class ComputedDescriptor(BaseDescriptor[T]):
    """
    Computes value from a function (no storage).

    The compute function receives the object instance and returns the value.
    """

    __slots__ = ("_compute",)

    descriptor_id = "computed"
    accepts_inner = ()  # Computed fields don't wrap
    accepts_outer = ("cached",)
    excludes = ("tracked", "networked", "observable")  # No write path

    def __init__(
        self,
        field_type: type = object,
        compute_func: Optional[Callable[[Any], T]] = None,
        **config: Any,
    ) -> None:
        """
        Initialize computed descriptor.

        Args:
            field_type: The type annotation for this field.
            compute_func: Function that computes the value.
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=None, **config)
        self._compute = compute_func

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Compute and return the value."""
        if obj is None:
            return self  # type: ignore

        if self._compute is None:
            raise AttributeError(
                f"Computed field '{self._name}' has no compute function"
            )

        return self._compute(obj)

    def __set__(self, obj: Any, value: T) -> None:
        """Raise error - computed fields are read-only."""
        raise AttributeError(f"Cannot set computed field '{self._name}'")

    def __delete__(self, obj: Any) -> None:
        """Raise error - computed fields cannot be deleted."""
        raise AttributeError(f"Cannot delete computed field '{self._name}'")

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "compute", "set": "deny", "delete": "deny"}),
                Step(Op.TAG, {"key": "computed", "value": True}),
                Step(Op.TAG, {"key": "transient", "value": True})]

    def get_metadata(self) -> dict[str, Any]:
        """Return computed configuration."""
        meta = super().get_metadata()
        meta["has_compute_func"] = self._compute is not None
        return meta
