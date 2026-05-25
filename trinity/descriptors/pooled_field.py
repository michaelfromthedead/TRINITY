"""
Pooled field descriptor - object pooling for field values.

Provides automatic return-to-pool on set/delete and factory-based acquisition.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["PooledDescriptor", "acquire"]

_DEFAULT_MAX_POOL_SIZE = 100


class PooledDescriptor(BaseDescriptor[T]):
    """
    Descriptor that pools field values for reuse.

    Old values are returned to a class-level pool on set/delete.
    New values can be acquired from the pool via the acquire() function.
    """

    __slots__ = ("_pool_factory", "_max_pool_size", "_pool_key")

    descriptor_id = "pooled_field"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        pool_factory: Callable[[], T],
        max_pool_size: int = _DEFAULT_MAX_POOL_SIZE,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._pool_factory = pool_factory
        self._max_pool_size = max_pool_size
        self._pool_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._pool_key = f"_pool_{name}"
        if not hasattr(owner, self._pool_key):
            setattr(owner, self._pool_key, [])

    def _get_pool(self) -> list:
        return getattr(self._owner, self._pool_key)

    def _return_to_pool(self, value: Any) -> None:
        if value is None:
            return
        pool = self._get_pool()
        if len(pool) < self._max_pool_size:
            pool.append(value)

    def __set__(self, obj: Any, value: T) -> None:
        old_value = self._get_stored_safe(obj)
        self._return_to_pool(old_value)
        super().__set__(obj, value)

    def __delete__(self, obj: Any) -> None:
        old_value = self._get_stored_safe(obj)
        self._return_to_pool(old_value)
        super().__delete__(obj)

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"set": "pool_return", "delete": "pool_return"}),
            Step(Op.TAG, {"key": "pooled_field", "value": True}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["max_pool_size"] = self._max_pool_size
        return meta


def acquire(cls: type, field: str) -> Any:
    """Acquire a value from the pool, or create one via the factory."""
    descriptor = cls.__dict__.get(field)
    if not isinstance(descriptor, PooledDescriptor):
        raise TypeError(f"{field} is not a PooledDescriptor on {cls.__name__}")
    pool = descriptor._get_pool()
    if pool:
        return pool.pop()
    return descriptor._pool_factory()
