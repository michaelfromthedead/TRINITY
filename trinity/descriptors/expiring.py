"""
Expiring descriptor - values that expire after a TTL.

Provides time-based expiration with configurable default.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["ExpiringDescriptor"]


class ExpiringDescriptor(BaseDescriptor[T]):
    """
    Descriptor whose values expire after a time-to-live period.

    After the TTL elapses, the stored value is cleared and the
    configured default is returned instead.
    """

    __slots__ = ("_ttl", "_default", "_expire_key")

    descriptor_id = "expiring"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        ttl: float,
        default: Any = None,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._ttl = ttl
        self._default = default
        self._expire_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._expire_key = f"_expire_{name}"

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        if obj is None:
            return self  # type: ignore

        expire_time = obj.__dict__.get(self._expire_key)
        if expire_time is not None and time.time() > expire_time:
            # Expired — clear value and return default
            self._delete_stored(obj)
            obj.__dict__.pop(self._expire_key, None)
            return self.post_get(obj, self._default)

        return super().__get__(obj, objtype)

    def __set__(self, obj: Any, value: T) -> None:
        super().__set__(obj, value)
        obj.__dict__[self._expire_key] = time.time() + self._ttl

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"get": "ttl_check"}),
            Step(Op.TAG, {"key": "ttl", "value": self._ttl}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["ttl"] = self._ttl
        meta["default"] = self._default
        return meta
