"""
Broadcast descriptor - notifies subscribers on field changes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

_log = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["BroadcastDescriptor", "subscribe", "unsubscribe"]


class BroadcastDescriptor(BaseDescriptor[T]):
    """
    Broadcasts field changes to subscribed callbacks.

    Subscribers receive (obj, field_name, old_value, new_value) on each change.
    """

    __slots__ = ("_channel",)

    descriptor_id = "broadcast"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        channel: str = "default",
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._channel = channel

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        subs_key = f"_subscribers_{name}"
        if not hasattr(owner, subs_key):
            setattr(owner, subs_key, [])

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        subs_key = f"_subscribers_{self._name}"
        subscribers = getattr(type(obj), subs_key, [])
        errors: list[Exception] = []
        for callback in subscribers:
            try:
                callback(obj, self._name, old_value, value)
            except Exception as exc:  # noqa: BLE001
                _log.error(
                    "Broadcast subscriber %r for field %r raised: %s",
                    callback,
                    self._name,
                    exc,
                )
                errors.append(exc)
        if errors:
            _log.warning(
                "%d subscriber(s) failed for field %r", len(errors), self._name
            )

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.HOOK, {"event": "on_change", "callback": None}),
            Step(Op.TAG, {"key": "broadcast_channel", "value": self._channel}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["channel"] = self._channel
        return meta


def subscribe(cls: type, field: str, callback: Callable) -> None:
    """Add a callback to the subscribers list for the given field on cls."""
    subs_key = f"_subscribers_{field}"
    if not hasattr(cls, subs_key):
        setattr(cls, subs_key, [])
    getattr(cls, subs_key).append(callback)


def unsubscribe(cls: type, field: str, callback: Callable) -> None:
    """Remove a callback from the subscribers list for the given field on cls."""
    subs_key = f"_subscribers_{field}"
    subs = getattr(cls, subs_key, [])
    if callback in subs:
        subs.remove(callback)
