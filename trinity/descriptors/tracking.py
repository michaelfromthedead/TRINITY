"""
Tracking descriptor - tracks field changes via dirty flags.

Used for change detection, serialization optimization, and networking.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class TrackedDescriptor(BaseDescriptor[T]):
    """
    Tracks field changes via dirty flags.

    When a value changes, marks the field as dirty. Supports both:
    - Set-based tracking (field names in a set)
    - Bitmask tracking (field offset as bit position)
    """

    __slots__ = ("_field_offset", "_use_bitmask")

    descriptor_id = "tracked"
    accepts_inner = ("storage", "validated", "range")
    accepts_outer = ("networked", "observable", "cached")
    excludes = ("computed",)  # Can't track computed fields

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        field_offset: int = 0,
        use_bitmask: bool = False,
        **config: Any,
    ) -> None:
        """
        Initialize tracking descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            field_offset: Bit position for bitmask tracking.
            use_bitmask: If True, use bitmask; if False, use set.
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        self._field_offset = field_offset
        self._use_bitmask = use_bitmask

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Mark field as dirty if value changed."""
        if value != old_value:
            # Always track in set for is_dirty() compatibility
            if not hasattr(obj, "_dirty_fields"):
                obj._dirty_fields = set()
            obj._dirty_fields.add(self._name)

            if self._use_bitmask:
                # Also use bitmask for high-performance network delta
                if not hasattr(obj, "_dirty_mask"):
                    obj._dirty_mask = 0
                obj._dirty_mask |= 1 << self._field_offset

            # Foundation integration: notify central tracker
            self._notify_foundation_tracker(obj, old_value, value)

            # EventLog integration: record change if in traced context
            self._notify_eventlog(obj, old_value, value)

    def _notify_foundation_tracker(
        self, obj: Any, old_value: Optional[T], new_value: T
    ) -> None:
        """Notify Foundation's central tracker of the change."""
        try:
            from foundation import tracker
            tracker.mark_dirty(obj, self._name, old_value, new_value)
        except ImportError:
            # Foundation not available - skip integration
            pass

    def _notify_eventlog(
        self, obj: Any, old_value: Optional[T], new_value: T
    ) -> None:
        """
        Record change to EventLog if inside a @traced context.

        Creates a Change object and appends it to the current Event.
        If no traced context is active, does nothing (changes are tracked
        via Foundation's tracker for standalone modifications).
        """
        try:
            from foundation.eventlog import (
                Change,
                add_change_to_current_event,
            )

            # Get entity ID (id attribute or object id as fallback)
            entity_id = getattr(obj, 'id', None)
            if entity_id is None:
                entity_id = id(obj)

            change = Change(
                entity=entity_id,
                field=self._name,
                old_value=old_value,
                new_value=new_value,
            )

            # Add to current event if in traced context
            add_change_to_current_event(change)
        except ImportError:
            # Foundation not available - skip integration
            pass

    @property
    def descriptor_steps(self) -> list["Step"]:
        steps = [Step(Op.TRACK, {"field": self._name})]
        if self._use_bitmask:
            steps.append(Step(Op.TAG, {"key": "track_bitmask", "value": True}))
        return steps

    def get_metadata(self) -> dict[str, Any]:
        """Return tracking configuration."""
        meta = super().get_metadata()
        meta["field_offset"] = self._field_offset
        meta["use_bitmask"] = self._use_bitmask
        return meta


def is_dirty(obj: Any, field_name: str) -> bool:
    """Check if a field is marked as dirty.

    Supports both set-based tracking (_dirty_fields) and bitmask tracking
    (_dirty_mask). For bitmask tracking, the field's descriptor must have
    a field_offset attribute.
    """
    # Check set-based tracking first
    dirty_fields = getattr(obj, "_dirty_fields", set())
    if field_name in dirty_fields:
        return True

    # Check bitmask tracking
    dirty_mask = getattr(obj, "_dirty_mask", 0)
    if dirty_mask:
        # Get the descriptor from the class to find field_offset
        descriptor = getattr(type(obj), field_name, None)
        if descriptor is not None:
            field_offset = getattr(descriptor, "_field_offset", None)
            if field_offset is not None:
                return bool(dirty_mask & (1 << field_offset))

    return False


def get_dirty_fields(obj: Any) -> set[str]:
    """Get all dirty field names."""
    return getattr(obj, "_dirty_fields", set()).copy()


def clear_dirty(obj: Any) -> None:
    """Clear all dirty flags."""
    if hasattr(obj, "_dirty_fields"):
        obj._dirty_fields.clear()
    if hasattr(obj, "_dirty_mask"):
        obj._dirty_mask = 0


def clear_dirty_field(obj: Any, field_name: str) -> None:
    """Clear dirty flag for a specific field."""
    if hasattr(obj, "_dirty_fields"):
        obj._dirty_fields.discard(field_name)


class VersionedDescriptor(BaseDescriptor[T]):
    """
    Per-field version counter.

    Increments on every change. Useful for cache invalidation
    and optimistic concurrency.
    """

    __slots__ = ("_version_attr",)

    descriptor_id = "versioned"
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
        self._version_attr: str = ""  # Set in __set_name__

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._version_attr = f"_version_{name}"

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Increment the version counter for this field."""
        current: int = getattr(obj, self._version_attr, 0)
        object.__setattr__(obj, self._version_attr, current + 1)

    def get_version(self, obj: Any) -> int:
        """Return the current version number for this field on *obj*."""
        return getattr(obj, self._version_attr, 0)

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.TRACK, {"field": self._name, "strategy": "versioned"})]

    def get_metadata(self) -> dict[str, Any]:
        """Return versioning configuration."""
        meta = super().get_metadata()
        meta["version_attr"] = self._version_attr
        return meta


class DiffDescriptor(BaseDescriptor[T]):
    """
    Stores previous value and computes diffs.

    Strategies: shallow, deep, structural, custom.
    """

    __slots__ = ("_prev_attr", "_strategy", "_custom_differ")

    descriptor_id = "diff"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    VALID_STRATEGIES: frozenset[str] = frozenset(
        {"shallow", "deep", "structural", "custom"}
    )

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        strategy: str = "shallow",
        custom_differ: Optional[Callable[[Any, Any], bool]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid diff strategy '{strategy}'. "
                f"Valid: {sorted(self.VALID_STRATEGIES)}"
            )
        if strategy == "custom" and custom_differ is None:
            raise ValueError(
                "custom_differ is required when strategy='custom'"
            )
        self._strategy: str = strategy
        self._custom_differ: Optional[Callable[[Any, Any], bool]] = custom_differ
        self._prev_attr: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._prev_attr = f"_prev_{name}"

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        """Store the old value so it can be retrieved later."""
        object.__setattr__(obj, self._prev_attr, old_value)

    def get_previous(self, obj: Any) -> Optional[T]:
        """Return the previous value of this field on *obj*."""
        return getattr(obj, self._prev_attr, None)

    def has_changed(self, obj: Any) -> bool:
        """Return whether the current value differs from the previous one."""
        current = self._get_stored_safe(obj)
        prev = self.get_previous(obj)
        if self._strategy == "shallow":
            return current != prev
        elif self._strategy == "deep":
            # Deep comparison: compare objects recursively
            # Note: We compare the actual values, not a deepcopy of prev
            if prev is None:
                return current is not None
            if current is None:
                return True
            # For deep comparison, use equality which handles nested structures
            return current != prev
        elif self._strategy == "custom" and self._custom_differ:
            return self._custom_differ(prev, current)
        # Fallback: structural strategy uses shallow comparison
        return current != prev

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.TRACK, {"field": self._name, "strategy": self._strategy})]

    def get_metadata(self) -> dict[str, Any]:
        """Return diff configuration."""
        meta = super().get_metadata()
        meta["strategy"] = self._strategy
        meta["has_custom_differ"] = self._custom_differ is not None
        return meta


__all__ = [
    "TrackedDescriptor",
    "VersionedDescriptor",
    "DiffDescriptor",
    "is_dirty",
    "get_dirty_fields",
    "clear_dirty",
    "clear_dirty_field",
]
