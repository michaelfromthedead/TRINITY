"""Simple per-type event bus."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Type

__all__ = ["EventBus"]


class EventBus:
    """Emit events, subscribe to event types, drain per-type queues each frame."""
    __slots__ = ("_queues", "_subscribers")

    def __init__(self) -> None:
        self._queues: dict[Type, list[Any]] = defaultdict(list)
        self._subscribers: dict[Type, list[Callable]] = defaultdict(list)

    def emit(self, event: Any) -> None:
        etype = type(event)
        self._queues[etype].append(event)
        for cb in self._subscribers.get(etype, ()):
            cb(event)

    def subscribe(self, event_type: Type, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: Type, callback: Callable) -> None:
        """Remove *callback* from *event_type* subscribers."""
        subs = self._subscribers.get(event_type)
        if subs is not None:
            try:
                subs.remove(callback)
            except ValueError:
                pass

    def drain(self, event_type: Type) -> list[Any]:
        """Remove and return all queued events of the given type."""
        events = self._queues.pop(event_type, [])
        return events

    def clear(self) -> None:
        """Clear all queued events and all subscribers."""
        self._queues.clear()
        self._subscribers.clear()

    def clear_events(self) -> None:
        """Clear only queued events, preserving subscribers."""
        self._queues.clear()
