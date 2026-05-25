"""
Observable descriptor - notifies observers on change.

Provides a simple observer pattern for field changes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

# Type alias for observer callbacks
Observer = Callable[
    [Any, str, Any, Any], None
]  # (obj, field_name, old_value, new_value)


class ObservableDescriptor(BaseDescriptor[T]):
    """
    Notifies observers on change.

    Observers are callbacks that receive (obj, field_name, old_value, new_value).
    """
    __slots__ = ()
    descriptor_id = "observable"
    accepts_inner = ("tracked", "storage", "validated", "range")
    accepts_outer = ("networked", "cached")
    excludes = ("transient",)  # Can't observe non-persisted fields

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.HOOK, {"event": "on_change", "callback": "observer_dispatch"})]

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Notify observers if value changed."""
        if value != old_value:
            observers = getattr(obj, "_observers", {}).get(self._name, [])
            for callback in observers:
                try:
                    callback(obj, self._name, old_value, value)
                except Exception as e:
                    # Log observer errors but don't let them break the setter
                    import warnings
                    warnings.warn(
                        f"Observer callback {callback} failed: {e}",
                        RuntimeWarning,
                        stacklevel=2
                    )


def add_observer(obj: Any, field_name: str, callback: Observer) -> None:
    """Add an observer for a field."""
    if not hasattr(obj, "_observers"):
        obj._observers = {}
    if field_name not in obj._observers:
        obj._observers[field_name] = []
    obj._observers[field_name].append(callback)


def remove_observer(obj: Any, field_name: str, callback: Observer) -> None:
    """Remove an observer for a field."""
    if hasattr(obj, "_observers") and field_name in obj._observers:
        try:
            obj._observers[field_name].remove(callback)
        except ValueError:
            pass


def clear_observers(obj: Any, field_name: Optional[str] = None) -> None:
    """Clear observers for a field or all fields."""
    if not hasattr(obj, "_observers"):
        return

    if field_name is None:
        obj._observers.clear()
    elif field_name in obj._observers:
        obj._observers[field_name].clear()


class BoundDescriptor(BaseDescriptor[T]):
    """Two-way binding to external data source."""
    __slots__ = ("_source", "_getter", "_setter")
    descriptor_id = "bound"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(self, field_type=object, inner=None, source=None, getter=None, setter=None, **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._source = source
        self._getter = getter
        self._setter = setter

    def post_get(self, obj, value):
        if self._getter:
            return self._getter(self._source)
        return value

    def post_set(self, obj, value, old_value):
        if self._setter:
            self._setter(self._source, value)

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.INTERCEPT, {"field": self._name, "strategy": "bound", "has_source": self._source is not None})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["source"] = str(self._source)
        return meta


__all__ = [
    "ObservableDescriptor",
    "BoundDescriptor",
    "Observer",
    "add_observer",
    "remove_observer",
    "clear_observers",
]
