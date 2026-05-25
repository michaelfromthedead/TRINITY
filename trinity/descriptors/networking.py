"""
Networking descriptor - queues network updates on change.

Handles network replication, authority, and interpolation hints.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypeVar

from trinity.constants import (
    DEFAULT_NETWORK_PRIORITY,
    DEFAULT_UPDATE_FREQUENCY,
    DEFAULT_MAX_UPDATES_PER_SECOND,
    DEFAULT_PREDICTION_HISTORY,
    INTERPOLATION_BUFFER_SIZE,
    VALID_NETWORK_AUTHORITIES,
)
from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class NetworkedDescriptor(BaseDescriptor[T]):
    """
    Queues network updates on change.

    When a value changes, adds it to the network queue for replication.
    Supports authority rules and interpolation hints.
    """

    __slots__ = ("_authority", "_interpolated", "_priority", "_update_frequency")

    descriptor_id = "networked"
    accepts_inner = ("tracked", "observable", "validated", "range", "storage")
    accepts_outer = ()  # Usually outermost
    excludes = ("transient", "local_only")

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        authority: str = "server",
        interpolated: bool = False,
        priority: int = DEFAULT_NETWORK_PRIORITY,
        update_frequency: int = DEFAULT_UPDATE_FREQUENCY,
        **config: Any,
    ) -> None:
        """
        Initialize networking descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            authority: Who can write - "server", "client", or "owner".
            interpolated: Whether to interpolate between network updates.
            priority: Network update priority (higher = more important).
            update_frequency: Ticks between updates (0 = every change).
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        if authority not in VALID_NETWORK_AUTHORITIES:
            raise ValueError(
                f"Invalid authority '{authority}'. Must be one of: {sorted(VALID_NETWORK_AUTHORITIES)}"
            )
        if update_frequency < 0:
            raise ValueError(f"update_frequency must be >= 0, got {update_frequency}")
        self._authority = authority
        self._interpolated = interpolated
        self._priority = priority
        self._update_frequency = update_frequency

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [
            Step(Op.TAG, {"key": "networked", "value": True}),
            Step(Op.TAG, {"key": "authority", "value": self._authority}),
            Step(Op.TAG, {"key": "interpolated", "value": self._interpolated}),
            Step(Op.INTERCEPT, {"set": "network_queue"}),
        ]

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Queue network update if value changed."""
        try:
            changed = value != old_value
        except TypeError:
            # For unhashable types, use identity comparison
            changed = value is not old_value

        if changed:
            # Queue for network replication
            if not hasattr(obj, "_network_queue"):
                obj._network_queue = []

            obj._network_queue.append(
                {
                    "field": self._name,
                    "value": value,
                    "old_value": old_value,
                    "priority": self._priority,
                }
            )

    def get_metadata(self) -> dict[str, Any]:
        """Return networking configuration."""
        meta = super().get_metadata()
        meta["network"] = {
            "authority": self._authority,
            "interpolated": self._interpolated,
            "priority": self._priority,
            "update_frequency": self._update_frequency,
        }
        return meta


def get_network_queue(obj: Any) -> list[dict[str, Any]]:
    """Get pending network updates for an object."""
    return getattr(obj, "_network_queue", []).copy()


def clear_network_queue(obj: Any) -> None:
    """Clear the network queue."""
    if hasattr(obj, "_network_queue"):
        obj._network_queue.clear()


def pop_network_updates(obj: Any) -> list[dict[str, Any]]:
    """Get and clear network updates."""
    updates = get_network_queue(obj)
    clear_network_queue(obj)
    return updates


class InterpolatedDescriptor(BaseDescriptor[T]):
    """Smooths between network snapshots for rendering. Modes: linear, hermite."""

    __slots__ = ("_mode", "_buffer_attr", "_t_attr")
    descriptor_id = "interpolated"
    accepts_inner = ("networked", "tracked", "storage", "*")
    accepts_outer = ("*",)
    excludes = ()

    VALID_MODES = frozenset({"linear", "hermite"})

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        mode: str = "linear",
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid interpolation mode '{mode}'. Valid: {sorted(self.VALID_MODES)}"
            )
        self._mode = mode
        self._buffer_attr = ""
        self._t_attr = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._buffer_attr = f"_interp_buffer_{name}"
        self._t_attr = f"_interp_t_{name}"

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Buffer snapshots for interpolation."""
        buf = getattr(obj, self._buffer_attr, None)
        if buf is None:
            buf = []
            object.__setattr__(obj, self._buffer_attr, buf)
        buf.append(value)
        if len(buf) > INTERPOLATION_BUFFER_SIZE:
            buf.pop(0)

    def get_interpolated(self, obj: Any, t: float) -> Optional[T]:
        """Get interpolated value at time t (0..1 between last two snapshots)."""
        buf = getattr(obj, self._buffer_attr, [])
        if len(buf) < 2:
            return buf[-1] if buf else None
        a, b = buf[-2], buf[-1]
        if self._mode == "linear":
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a + (b - a) * t  # type: ignore
            return b  # non-numeric fallback
        elif self._mode == "hermite":
            # Cubic hermite with 0 tangents (Catmull-Rom simplified)
            t2 = t * t
            t3 = t2 * t
            h1 = 2 * t3 - 3 * t2 + 1
            h2 = -2 * t3 + 3 * t2
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return h1 * a + h2 * b  # type: ignore
            return b
        return b

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.INTERCEPT, {"field": self._name, "strategy": "interpolation", "method": self._mode})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["mode"] = self._mode
        return meta


class PredictedDescriptor(BaseDescriptor[T]):
    """Client-side prediction with history buffer and rollback."""

    __slots__ = ("_history_attr", "_max_history", "_predicted_attr")
    descriptor_id = "predicted"
    accepts_inner = ("networked", "tracked", "storage", "*")
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        max_history: int = DEFAULT_PREDICTION_HISTORY,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if max_history <= 0:
            raise ValueError(f"max_history must be > 0, got {max_history}")
        self._max_history = max_history
        self._history_attr = ""
        self._predicted_attr = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._history_attr = f"_pred_history_{name}"
        self._predicted_attr = f"_pred_value_{name}"

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        history = getattr(obj, self._history_attr, None)
        if history is None:
            history = []
            object.__setattr__(obj, self._history_attr, history)
        history.append(value)
        if len(history) > self._max_history:
            history.pop(0)

    def rollback(self, obj: Any, frames: int = 1) -> Any:
        """Roll back to a previous state."""
        history = getattr(obj, self._history_attr, [])
        if frames > 0 and len(history) > frames:
            target = history[-(frames + 1)]
            del history[-frames:]
            # Directly set the value without triggering post_set
            if self._inner is not None:
                self._inner.__set__(obj, target)
            else:
                self._set_stored(obj, target)
            return target
        return None

    def get_history(self, obj: Any) -> list:
        return list(getattr(obj, self._history_attr, []))

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.INTERCEPT, {"field": self._name, "strategy": "prediction"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["max_history"] = self._max_history
        return meta


class ThrottledNetworkDescriptor(BaseDescriptor[T]):
    """Rate-limits network updates per field. Token bucket algorithm."""

    __slots__ = ("_min_interval", "_last_send_attr", "_pending_attr")
    descriptor_id = "throttled_network"
    accepts_inner = ("networked", "tracked", "storage", "*")
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        max_updates_per_second: float = DEFAULT_MAX_UPDATES_PER_SECOND,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if max_updates_per_second <= 0:
            raise ValueError(
                f"max_updates_per_second must be > 0, got {max_updates_per_second}"
            )
        self._min_interval = 1.0 / max_updates_per_second
        self._last_send_attr = ""
        self._pending_attr = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._last_send_attr = f"_throttle_last_{name}"
        self._pending_attr = f"_throttle_pending_{name}"

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        now = time.monotonic()
        last = getattr(obj, self._last_send_attr, 0.0)
        if now - last >= self._min_interval:
            object.__setattr__(obj, self._last_send_attr, now)
            object.__setattr__(obj, self._pending_attr, False)
            # Would trigger actual network send here
        else:
            object.__setattr__(obj, self._pending_attr, True)

    def has_pending(self, obj: Any) -> bool:
        return getattr(obj, self._pending_attr, False)

    def flush(self, obj: Any) -> None:
        """Force send pending update."""
        object.__setattr__(obj, self._last_send_attr, time.monotonic())
        object.__setattr__(obj, self._pending_attr, False)

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.INTERCEPT, {"field": self._name, "strategy": "throttle"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["min_interval"] = self._min_interval
        return meta


__all__ = [
    "NetworkedDescriptor",
    "InterpolatedDescriptor",
    "PredictedDescriptor",
    "ThrottledNetworkDescriptor",
    "get_network_queue",
    "clear_network_queue",
    "pop_network_updates",
]
