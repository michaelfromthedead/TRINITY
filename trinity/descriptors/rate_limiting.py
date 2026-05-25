"""
Rate-limiting descriptor - throttle write frequency.

Enforces a maximum number of writes per second with configurable
exceed policies: raise or drop.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

# Sliding window duration in seconds
_WINDOW_SECONDS = 1.0


class RateLimitExceeded(Exception):
    """Raised when write rate exceeds the configured limit."""
    pass


class _DropWrite(Exception):
    """Internal signal to silently drop a write under the 'drop' policy."""
    pass


class RateLimitedDescriptor(BaseDescriptor[T]):
    """
    Throttles writes to a maximum frequency.

    Tracks timestamps of recent writes and enforces a sliding 1-second
    window limit. When exceeded, applies the configured policy.
    """

    __slots__ = ("_max_writes_per_second", "_on_exceed", "_ts_key")

    descriptor_id = "rate_limited"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ("rate_limited",)

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        max_writes_per_second: float = 10.0,
        on_exceed: str = "raise",
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if on_exceed not in ("raise", "drop"):
            # NOTE: "queue" policy may be implemented in a future version.
            raise ValueError(f"on_exceed must be 'raise' or 'drop', got {on_exceed!r}")
        self._max_writes_per_second = int(math.floor(max_writes_per_second))
        self._on_exceed = on_exceed
        self._ts_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._ts_key = f"_rate_{name}_timestamps"

    def _get_timestamps(self, obj: Any) -> deque:
        ts = obj.__dict__.get(self._ts_key)
        if ts is None:
            ts = deque()
            obj.__dict__[self._ts_key] = ts
        return ts

    def pre_set(self, obj: Any, value: T) -> T:
        """Check rate limit before allowing write."""
        timestamps = self._get_timestamps(obj)
        now = time.time()
        # Trim old entries outside the sliding window
        while timestamps and now - timestamps[0] > _WINDOW_SECONDS:
            timestamps.popleft()

        if len(timestamps) >= self._max_writes_per_second:
            if self._on_exceed == "raise":
                raise RateLimitExceeded(
                    f"Rate limit exceeded for '{self._name}': "
                    f"{self._max_writes_per_second} writes/s"
                )
            elif self._on_exceed == "drop":
                raise _DropWrite()
        return value

    def __set__(self, obj: Any, value: T) -> None:
        """Delegate to base __set__; catch _DropWrite to silently discard."""
        try:
            super().__set__(obj, value)
        except _DropWrite:
            return  # Drop the write silently

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Record timestamp after successful write."""
        timestamps = self._get_timestamps(obj)
        timestamps.append(time.time())

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"set": "rate_check"}),
            Step(Op.VALIDATE, {"constraint": "rate_limit"}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["max_writes_per_second"] = self._max_writes_per_second
        meta["on_exceed"] = self._on_exceed
        return meta



__all__ = ["RateLimitedDescriptor", "RateLimitExceeded"]
