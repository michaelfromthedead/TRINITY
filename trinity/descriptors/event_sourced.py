"""
Event-sourced descriptor - records all changes as an event log.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["EventSourcedDescriptor", "get_events", "replay_events"]

_DEFAULT_MAX_EVENTS = 1000


class EventSourcedDescriptor(BaseDescriptor[T]):
    """
    Records all field mutations as timestamped events.

    Events can be retrieved and replayed to reconstruct state.
    """

    __slots__ = ("_max_events", "_events_key")

    descriptor_id = "event_sourced"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        max_events: int = _DEFAULT_MAX_EVENTS,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._max_events = max_events
        self._events_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._events_key = f"_events_{name}"

    def _get_events_list(self, obj: Any) -> list[dict[str, Any]]:
        key = self._events_key
        if key not in obj.__dict__:
            obj.__dict__[key] = []
        return obj.__dict__[key]

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        events = self._get_events_list(obj)
        events.append({
            "type": "set",
            "field": self._name,
            "old": old_value,
            "new": value,
            "timestamp": time.time(),
        })
        # Trim to max_events
        if len(events) > self._max_events:
            del events[: len(events) - self._max_events]

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.TRACK, {"field": self._name}),
            Step(Op.TAG, {"key": "event_sourced", "value": True}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["max_events"] = self._max_events
        return meta


def get_events(
    obj: Any, field: str, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Return the event list for the given field on obj."""
    key = f"_events_{field}"
    events = obj.__dict__.get(key, [])
    if limit is not None:
        return events[-limit:]
    return list(events)


def replay_events(obj: Any, field: str) -> Any:
    """Replay all events to reconstruct the final state."""
    events = get_events(obj, field)
    value = None
    for event in events:
        if event["type"] == "set":
            value = event["new"]
    return value
