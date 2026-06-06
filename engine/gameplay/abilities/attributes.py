"""
Attribute System.

Provides a flexible attribute system for gameplay with base values, modifiers,
min/max bounds, and derived attributes with formula-based calculations.

Attributes follow proper modifier order of operations:
1. Add Base modifiers
2. Multiply Base modifiers
3. Add Bonus modifiers
4. Multiply Bonus modifiers
5. Override modifiers
6. Clamp to min/max bounds

Foundation Integration:
- TrackedDescriptor wiring via @tracked_attribute decorator
- Automatic dirty flag tracking on attribute changes
- Change subscription via tracker.on_change()
- Batch updates via tracker.begin_batch() / tracker.end_batch()
"""

from __future__ import annotations

import math
import threading
import weakref
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from uuid import UUID, uuid4

from engine.gameplay.abilities.constants import (
    DEFAULT_ATTRIBUTE_MAX,
    DEFAULT_ATTRIBUTE_MIN,
    EPSILON,
    MODIFIER_ORDER_ADD_BASE,
    MODIFIER_ORDER_ADD_BONUS,
    MODIFIER_ORDER_MULTIPLY_BASE,
    MODIFIER_ORDER_MULTIPLY_BONUS,
    MODIFIER_ORDER_OVERRIDE,
    ModifierOperation,
)

# Foundation integration imports
try:
    from foundation import tracker as foundation_tracker
    from foundation import Tracker, Change, Transaction
    FOUNDATION_AVAILABLE = True
except ImportError:
    FOUNDATION_AVAILABLE = False
    foundation_tracker = None
    Tracker = None
    Change = None
    Transaction = None

# Trinity descriptor integration
try:
    from trinity.descriptors import TrackedDescriptor as TrinityTrackedDescriptor
    TRINITY_DESCRIPTORS_AVAILABLE = True
except ImportError:
    TRINITY_DESCRIPTORS_AVAILABLE = False
    TrinityTrackedDescriptor = None

T = TypeVar("T")
AttrT = TypeVar("AttrT", bound=float)

# Type alias for attribute change callbacks
AttributeChangeCallback = Callable[[Any, str, Any, Any], None]


# =============================================================================
# ATTRIBUTE TRACKER (Foundation Integration)
# =============================================================================


class AttributeTracker:
    """
    Centralized tracker for ability attributes with Foundation integration.

    Provides:
    - Dirty flag tracking per attribute
    - Change subscriptions by attribute type or instance
    - Batch update support for atomic changes
    - Integration with Foundation Tracker when available
    """

    __slots__ = (
        "_dirty",
        "_callbacks",
        "_type_callbacks",
        "_batch_mode",
        "_batch_changes",
        "_lock",
        "_version",
    )

    def __init__(self) -> None:
        """Initialize the attribute tracker."""
        self._dirty: Dict[int, Tuple[weakref.ref, Set[str]]] = {}
        self._callbacks: Dict[int, List[AttributeChangeCallback]] = {}
        self._type_callbacks: Dict[str, List[AttributeChangeCallback]] = {}
        self._batch_mode: bool = False
        self._batch_changes: List[Tuple[Any, str, Any, Any]] = []
        self._lock = threading.RLock()
        self._version: int = 0

    def mark_dirty(
        self,
        obj: Any,
        field_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Mark an attribute as dirty and optionally notify callbacks."""
        with self._lock:
            oid = id(obj)
            if oid not in self._dirty:
                self._dirty[oid] = (
                    weakref.ref(obj, lambda _: self._cleanup(oid)),
                    set(),
                )
            self._dirty[oid][1].add(field_name)
            self._version += 1

            if self._batch_mode:
                # Store change for later notification
                self._batch_changes.append((obj, field_name, old_value, new_value))
            else:
                self._notify(obj, field_name, old_value, new_value)

            # Forward to Foundation tracker if available
            if FOUNDATION_AVAILABLE and foundation_tracker is not None:
                foundation_tracker.mark_dirty(obj, field_name, old_value, new_value)

    def mark_clean(self, obj: Any, field_name: Optional[str] = None) -> None:
        """Clear dirty flag(s) for an object."""
        with self._lock:
            oid = id(obj)
            entry = self._dirty.get(oid)
            if entry:
                if field_name is None:
                    entry[1].clear()
                else:
                    entry[1].discard(field_name)

    def is_dirty(self, obj: Any, field_name: Optional[str] = None) -> bool:
        """Check if an object or specific field is dirty."""
        with self._lock:
            oid = id(obj)
            entry = self._dirty.get(oid)
            if not entry:
                return False
            if field_name is None:
                return bool(entry[1])
            return field_name in entry[1]

    def dirty_fields(self, obj: Any) -> Set[str]:
        """Get all dirty field names for an object."""
        with self._lock:
            oid = id(obj)
            entry = self._dirty.get(oid)
            return set(entry[1]) if entry else set()

    def all_dirty(self) -> bool:
        """Check if any tracked object has dirty fields."""
        with self._lock:
            for oid, (ref, fields) in list(self._dirty.items()):
                if ref() is None:
                    del self._dirty[oid]
                    continue
                if fields:
                    return True
            return False

    def get_all_dirty_objects(self) -> List[Any]:
        """Return list of all objects with dirty fields."""
        with self._lock:
            result: List[Any] = []
            dead: List[int] = []
            for oid, (ref, fields) in self._dirty.items():
                obj = ref()
                if obj is None:
                    dead.append(oid)
                elif fields:
                    result.append(obj)
            for oid in dead:
                del self._dirty[oid]
            return result

    def on_change(
        self,
        target: Union[None, Any, str],
        callback: AttributeChangeCallback,
    ) -> None:
        """
        Subscribe to attribute changes.

        Args:
            target: None for global, object instance for object-specific,
                   or string for attribute type subscription
            callback: Function(obj, field, old, new) to call on change
        """
        with self._lock:
            if target is None:
                # Global subscription - use type callback with empty key
                self._type_callbacks.setdefault("", []).append(callback)
            elif isinstance(target, str):
                # Attribute type subscription
                self._type_callbacks.setdefault(target, []).append(callback)
            else:
                # Object-specific subscription
                oid = id(target)
                self._callbacks.setdefault(oid, []).append(callback)

    def off_change(self, callback: AttributeChangeCallback) -> None:
        """Unsubscribe a callback from all subscriptions."""
        with self._lock:
            # Remove from type callbacks
            for cbs in self._type_callbacks.values():
                if callback in cbs:
                    cbs.remove(callback)
            # Remove from object callbacks
            for cbs in self._callbacks.values():
                if callback in cbs:
                    cbs.remove(callback)

    def begin_batch(self) -> None:
        """Begin batch mode - callbacks deferred until end_batch()."""
        with self._lock:
            if self._batch_mode:
                raise RuntimeError("Already in batch mode")
            self._batch_mode = True
            self._batch_changes.clear()

    def end_batch(self) -> None:
        """End batch mode and fire all deferred callbacks."""
        with self._lock:
            if not self._batch_mode:
                raise RuntimeError("Not in batch mode")
            self._batch_mode = False
            # Fire all deferred notifications
            changes = self._batch_changes.copy()
            self._batch_changes.clear()

        # Fire callbacks outside the lock
        for obj, field_name, old_value, new_value in changes:
            self._notify(obj, field_name, old_value, new_value)

    @property
    def in_batch(self) -> bool:
        """Check if currently in batch mode."""
        return self._batch_mode

    @property
    def version(self) -> int:
        """Get the global version counter."""
        return self._version

    def _notify(
        self,
        obj: Any,
        field_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Fire change callbacks."""
        # Global callbacks
        for cb in self._type_callbacks.get("", []):
            try:
                cb(obj, field_name, old_value, new_value)
            except Exception:
                pass

        # Type-specific callbacks
        for cb in self._type_callbacks.get(field_name, []):
            try:
                cb(obj, field_name, old_value, new_value)
            except Exception:
                pass

        # Object-specific callbacks
        for cb in self._callbacks.get(id(obj), []):
            try:
                cb(obj, field_name, old_value, new_value)
            except Exception:
                pass

    def _cleanup(self, oid: int) -> None:
        """Clean up entries for garbage collected objects."""
        with self._lock:
            self._dirty.pop(oid, None)
            self._callbacks.pop(oid, None)


# Global attribute tracker instance
attribute_tracker = AttributeTracker()


# =============================================================================
# TRACKED ATTRIBUTE DESCRIPTOR
# =============================================================================


class TrackedAttributeDescriptor(Generic[AttrT]):
    """
    Descriptor that wraps attribute access with Foundation tracking.

    Provides:
    - Automatic dirty flag on value change
    - Min/max value clamping
    - Integration with AttributeTracker for subscriptions
    - Compatible with Trinity TrackedDescriptor pattern
    """

    __slots__ = (
        "_name",
        "_min_value",
        "_max_value",
        "_default",
        "_storage_attr",
        "_tracker",
    )

    def __init__(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default: float = 0.0,
        tracker: Optional[AttributeTracker] = None,
    ) -> None:
        """
        Initialize tracked attribute descriptor.

        Args:
            name: Attribute name for tracking
            min_value: Minimum allowed value (None for no limit)
            max_value: Maximum allowed value (None for no limit)
            default: Default value when not set
            tracker: AttributeTracker instance (uses global if not provided)
        """
        self._name = name
        self._min_value = min_value
        self._max_value = max_value
        self._default = default
        self._storage_attr = f"_tracked_attr_{name}"
        self._tracker = tracker or attribute_tracker

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when descriptor is assigned to a class attribute."""
        # Register with class for introspection
        if not hasattr(owner, "_tracked_attributes"):
            owner._tracked_attributes = set()
        owner._tracked_attributes.add(self._name)

    @overload
    def __get__(self, obj: None, objtype: type) -> "TrackedAttributeDescriptor[AttrT]":
        ...

    @overload
    def __get__(self, obj: Any, objtype: Optional[type]) -> AttrT:
        ...

    def __get__(
        self, obj: Any, objtype: Optional[type] = None
    ) -> Union["TrackedAttributeDescriptor[AttrT]", AttrT]:
        """Get the tracked attribute value."""
        if obj is None:
            return self
        return getattr(obj, self._storage_attr, self._default)

    def __set__(self, obj: Any, value: AttrT) -> None:
        """Set the tracked attribute value with clamping and dirty tracking."""
        # Get old value
        old_value = getattr(obj, self._storage_attr, self._default)

        # Apply clamping
        clamped_value = self._clamp(value)

        # Only update and track if actually changed
        if old_value != clamped_value:
            setattr(obj, self._storage_attr, clamped_value)
            self._tracker.mark_dirty(obj, self._name, old_value, clamped_value)

    def __delete__(self, obj: Any) -> None:
        """Reset attribute to default."""
        old_value = getattr(obj, self._storage_attr, self._default)
        if hasattr(obj, self._storage_attr):
            delattr(obj, self._storage_attr)
        if old_value != self._default:
            self._tracker.mark_dirty(obj, self._name, old_value, self._default)

    def _clamp(self, value: AttrT) -> AttrT:
        """Clamp value to min/max bounds."""
        result = value
        if self._min_value is not None and result < self._min_value:
            result = self._min_value  # type: ignore
        if self._max_value is not None and result > self._max_value:
            result = self._max_value  # type: ignore
        return result

    @property
    def name(self) -> str:
        """Get attribute name."""
        return self._name

    @property
    def min_value(self) -> Optional[float]:
        """Get minimum value constraint."""
        return self._min_value

    @property
    def max_value(self) -> Optional[float]:
        """Get maximum value constraint."""
        return self._max_value

    @property
    def default(self) -> float:
        """Get default value."""
        return self._default

    def is_dirty(self, obj: Any) -> bool:
        """Check if this attribute is dirty on the given object."""
        return self._tracker.is_dirty(obj, self._name)

    def clear_dirty(self, obj: Any) -> None:
        """Clear the dirty flag for this attribute."""
        self._tracker.mark_clean(obj, self._name)

    def mark_dirty(self, obj: Any) -> None:
        """Manually mark this attribute as dirty."""
        value = self.__get__(obj, type(obj))
        self._tracker.mark_dirty(obj, self._name, value, value)


def tracked_attribute(
    name: str,
    min: Optional[float] = None,
    max: Optional[float] = None,
    default: float = 0.0,
    tracker: Optional[AttributeTracker] = None,
) -> TrackedAttributeDescriptor:
    """
    Create a tracked attribute descriptor.

    This decorator/factory creates a TrackedAttributeDescriptor that:
    - Wraps attribute access with automatic dirty tracking
    - Supports min/max value clamping
    - Integrates with Foundation's Tracker system
    - Enables change subscriptions via tracker.on_change()

    Args:
        name: Attribute name for tracking and subscriptions
        min: Minimum allowed value (None for no minimum)
        max: Maximum allowed value (None for no maximum)
        default: Default value when attribute is not set
        tracker: Custom AttributeTracker (uses global if not provided)

    Returns:
        TrackedAttributeDescriptor instance

    Example:
        class Character:
            health = tracked_attribute("health", min=0, max=100, default=100)
            mana = tracked_attribute("mana", min=0, max=50, default=50)

        char = Character()
        char.health = 80  # Sets dirty flag
        attribute_tracker.is_dirty(char, "health")  # True
    """
    return TrackedAttributeDescriptor(
        name=name,
        min_value=min,
        max_value=max,
        default=default,
        tracker=tracker,
    )


# =============================================================================
# TRACKED ABILITY ATTRIBUTE CLASSES
# =============================================================================


class TrackedAbilityAttribute:
    """
    Base class for ability attributes with Foundation tracking integration.

    Provides automatic dirty tracking, min/max clamping, and change callbacks
    for common ability attributes like Health, Mana, Stamina, and Cooldowns.
    """

    # Class-level tracked attributes - override in subclasses
    _tracked_fields: FrozenSet[str] = frozenset()

    def __init__(self, tracker: Optional[AttributeTracker] = None) -> None:
        """Initialize with optional custom tracker."""
        self._custom_tracker = tracker

    @property
    def tracker(self) -> AttributeTracker:
        """Get the tracker instance for this attribute."""
        return self._custom_tracker or attribute_tracker

    def is_dirty(self) -> bool:
        """Check if any field on this attribute is dirty."""
        return self.tracker.is_dirty(self)

    def clear_dirty(self) -> None:
        """Clear all dirty flags for this attribute."""
        self.tracker.mark_clean(self)


class TrackedVitalAttribute(TrackedAbilityAttribute):
    """
    Tracked vital attribute (Health, Mana, Stamina) with current/max values.
    """

    current = tracked_attribute("current", min=0.0, default=100.0)
    maximum = tracked_attribute("maximum", min=1.0, default=100.0)
    regen_rate = tracked_attribute("regen_rate", min=-100.0, max=1000.0, default=0.0)

    _tracked_fields = frozenset({"current", "maximum", "regen_rate"})

    def __init__(
        self,
        current: float = 100.0,
        maximum: float = 100.0,
        regen_rate: float = 0.0,
        tracker: Optional[AttributeTracker] = None,
    ) -> None:
        """Initialize vital attribute."""
        super().__init__(tracker)
        self.current = min(current, maximum)
        self.maximum = maximum
        self.regen_rate = regen_rate
        # Clear initial dirty state
        self.clear_dirty()

    @property
    def percent(self) -> float:
        """Get current value as percentage of maximum."""
        if self.maximum <= 0:
            return 0.0
        return self.current / self.maximum

    def apply_damage(self, amount: float) -> float:
        """Apply damage, returning actual damage dealt."""
        old = self.current
        self.current = max(0.0, self.current - amount)
        return old - self.current

    def apply_healing(self, amount: float) -> float:
        """Apply healing, returning actual healing done."""
        old = self.current
        self.current = min(self.maximum, self.current + amount)
        return self.current - old

    def regenerate(self, delta_time: float) -> float:
        """Apply regeneration for delta_time seconds."""
        if self.regen_rate == 0:
            return 0.0
        amount = self.regen_rate * delta_time
        if amount > 0:
            return self.apply_healing(amount)
        else:
            return -self.apply_damage(-amount)


class TrackedCooldownAttribute(TrackedAbilityAttribute):
    """
    Tracked cooldown timer attribute.
    """

    remaining = tracked_attribute("remaining", min=0.0, default=0.0)
    duration = tracked_attribute("duration", min=0.0, default=1.0)
    reduction = tracked_attribute("reduction", min=0.0, max=0.75, default=0.0)

    _tracked_fields = frozenset({"remaining", "duration", "reduction"})

    def __init__(
        self,
        duration: float = 1.0,
        reduction: float = 0.0,
        tracker: Optional[AttributeTracker] = None,
    ) -> None:
        """Initialize cooldown attribute."""
        super().__init__(tracker)
        self.duration = duration
        self.reduction = reduction
        self.remaining = 0.0
        self.clear_dirty()

    @property
    def effective_duration(self) -> float:
        """Get duration after cooldown reduction applied."""
        return self.duration * (1.0 - self.reduction)

    @property
    def is_ready(self) -> bool:
        """Check if cooldown is ready (remaining <= 0)."""
        return self.remaining <= EPSILON

    @property
    def progress(self) -> float:
        """Get cooldown progress (0.0 = just started, 1.0 = ready)."""
        eff_dur = self.effective_duration
        if eff_dur <= 0:
            return 1.0
        return 1.0 - (self.remaining / eff_dur)

    def start(self) -> None:
        """Start the cooldown timer."""
        self.remaining = self.effective_duration

    def tick(self, delta_time: float) -> bool:
        """
        Tick the cooldown timer.

        Returns True if cooldown just became ready.
        """
        if self.remaining <= 0:
            return False
        was_on_cooldown = self.remaining > EPSILON
        self.remaining = max(0.0, self.remaining - delta_time)
        return was_on_cooldown and self.remaining <= EPSILON

    def reset(self) -> None:
        """Reset cooldown to ready state."""
        self.remaining = 0.0


# =============================================================================
# ATTRIBUTE MODIFIER
# =============================================================================


@dataclass(slots=True)
class AttributeModifier:
    """
    A modifier that affects an attribute's value.

    Modifiers are applied in order based on their operation type:
    ADD_BASE -> MULTIPLY_BASE -> ADD_BONUS -> MULTIPLY_BONUS -> OVERRIDE
    """

    operation: ModifierOperation
    magnitude: float
    source: Optional[Any] = None
    id: UUID = field(default_factory=uuid4)
    order: int = field(init=False)

    def __post_init__(self) -> None:
        """Set order based on operation type."""
        self.order = {
            ModifierOperation.ADD: MODIFIER_ORDER_ADD_BASE,
            ModifierOperation.MULTIPLY: MODIFIER_ORDER_MULTIPLY_BASE,
            ModifierOperation.OVERRIDE: MODIFIER_ORDER_OVERRIDE,
            ModifierOperation.STACKING: MODIFIER_ORDER_ADD_BONUS,
        }.get(self.operation, MODIFIER_ORDER_ADD_BASE)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AttributeModifier):
            return self.id == other.id
        return False


@dataclass(slots=True)
class AttributeModifierHandle:
    """Handle for tracking and removing a modifier."""

    modifier_id: UUID
    attribute_name: str
    source: Optional[Any] = None


# =============================================================================
# ATTRIBUTE
# =============================================================================


@dataclass
class Attribute:
    """
    A gameplay attribute with base value, modifiers, and bounds.

    The current value is calculated from the base value plus all active
    modifiers, then clamped to the min/max bounds.
    """

    name: str
    base_value: float = 0.0
    min_value: float = DEFAULT_ATTRIBUTE_MIN
    max_value: float = DEFAULT_ATTRIBUTE_MAX
    _modifiers: List[AttributeModifier] = field(default_factory=list)
    _cached_value: Optional[float] = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)
    _on_change: Optional[Callable[[Attribute, float, float], None]] = field(
        default=None, repr=False
    )

    @property
    def current_value(self) -> float:
        """Get the current value after all modifiers are applied."""
        if self._dirty or self._cached_value is None:
            self._recalculate()
        return self._cached_value  # type: ignore

    @property
    def value(self) -> float:
        """Alias for current_value."""
        return self.current_value

    def set_base_value(self, value: float) -> None:
        """Set the base value and mark dirty."""
        old_value = self.current_value
        self.base_value = value
        self._mark_dirty()
        self._notify_change(old_value)

    def add_modifier(self, modifier: AttributeModifier) -> AttributeModifierHandle:
        """Add a modifier and return a handle for removal."""
        old_value = self.current_value
        self._modifiers.append(modifier)
        self._modifiers.sort(key=lambda m: m.order)
        self._mark_dirty()
        self._notify_change(old_value)
        return AttributeModifierHandle(
            modifier_id=modifier.id,
            attribute_name=self.name,
            source=modifier.source,
        )

    def remove_modifier(self, modifier_or_handle: AttributeModifier | AttributeModifierHandle | UUID) -> bool:
        """Remove a modifier by modifier, handle, or ID. Returns True if removed."""
        if isinstance(modifier_or_handle, AttributeModifierHandle):
            modifier_id = modifier_or_handle.modifier_id
        elif isinstance(modifier_or_handle, UUID):
            modifier_id = modifier_or_handle
        else:
            modifier_id = modifier_or_handle.id

        old_value = self.current_value
        for i, mod in enumerate(self._modifiers):
            if mod.id == modifier_id:
                self._modifiers.pop(i)
                self._mark_dirty()
                self._notify_change(old_value)
                return True
        return False

    def remove_modifiers_from_source(self, source: Any) -> int:
        """Remove all modifiers from a source. Returns count removed."""
        old_value = self.current_value
        original_count = len(self._modifiers)
        self._modifiers = [m for m in self._modifiers if m.source != source]
        removed = original_count - len(self._modifiers)
        if removed > 0:
            self._mark_dirty()
            self._notify_change(old_value)
        return removed

    def clear_modifiers(self) -> int:
        """Remove all modifiers. Returns count removed."""
        old_value = self.current_value
        count = len(self._modifiers)
        if count > 0:
            self._modifiers.clear()
            self._mark_dirty()
            self._notify_change(old_value)
        return count

    def get_modifiers(self) -> List[AttributeModifier]:
        """Get a copy of all active modifiers."""
        return list(self._modifiers)

    def get_modifiers_by_operation(
        self, operation: ModifierOperation
    ) -> List[AttributeModifier]:
        """Get all modifiers with the given operation type."""
        return [m for m in self._modifiers if m.operation == operation]

    def _recalculate(self) -> None:
        """Recalculate the current value from base and modifiers."""
        value = self.base_value

        # Group modifiers by operation
        add_base: List[float] = []
        mult_base: List[float] = []
        add_bonus: List[float] = []
        mult_bonus: List[float] = []
        override: Optional[float] = None

        for mod in self._modifiers:
            if mod.operation == ModifierOperation.ADD:
                add_base.append(mod.magnitude)
            elif mod.operation == ModifierOperation.MULTIPLY:
                mult_base.append(mod.magnitude)
            elif mod.operation == ModifierOperation.STACKING:
                add_bonus.append(mod.magnitude)
            elif mod.operation == ModifierOperation.OVERRIDE:
                override = mod.magnitude

        # Apply in order
        # 1. Add base modifiers
        for mag in add_base:
            value += mag

        # 2. Multiply base modifiers (additive stacking: 1 + sum(mults))
        if mult_base:
            multiplier = 1.0 + sum(mult_base)
            value *= multiplier

        # 3. Add bonus modifiers
        for mag in add_bonus:
            value += mag

        # 4. Multiply bonus modifiers
        if mult_bonus:
            multiplier = 1.0 + sum(mult_bonus)
            value *= multiplier

        # 5. Override (last one wins)
        if override is not None:
            value = override

        # 6. Clamp to bounds
        value = max(self.min_value, min(self.max_value, value))

        self._cached_value = value
        self._dirty = False

    def _mark_dirty(self) -> None:
        """Mark the cached value as needing recalculation."""
        self._dirty = True

    def _notify_change(self, old_value: float) -> None:
        """Notify callback if value changed."""
        if self._on_change is not None:
            new_value = self.current_value
            if abs(new_value - old_value) > EPSILON:
                self._on_change(self, old_value, new_value)

    def __float__(self) -> float:
        return self.current_value

    def __int__(self) -> int:
        return int(self.current_value)


# =============================================================================
# DERIVED ATTRIBUTE
# =============================================================================


@dataclass
class DerivedAttribute:
    """
    An attribute whose value is derived from other attributes via a formula.

    The value is cached and only recalculated when dependencies change.
    """

    name: str
    formula: Callable[[Dict[str, float]], float]
    dependencies: FrozenSet[str]
    min_value: float = DEFAULT_ATTRIBUTE_MIN
    max_value: float = DEFAULT_ATTRIBUTE_MAX
    _cached_value: Optional[float] = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        formula: Callable[[Dict[str, float]], float],
        *dependencies: str,
        min_value: float = DEFAULT_ATTRIBUTE_MIN,
        max_value: float = DEFAULT_ATTRIBUTE_MAX,
    ) -> DerivedAttribute:
        """Create a derived attribute with the given formula and dependencies."""
        return cls(
            name=name,
            formula=formula,
            dependencies=frozenset(dependencies),
            min_value=min_value,
            max_value=max_value,
        )

    def calculate(self, attributes: Dict[str, float]) -> float:
        """Calculate the derived value from source attributes."""
        if self._dirty or self._cached_value is None:
            value = self.formula(attributes)
            value = max(self.min_value, min(self.max_value, value))
            self._cached_value = value
            self._dirty = False
        return self._cached_value

    def mark_dirty(self) -> None:
        """Mark the cached value as needing recalculation."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if the cached value needs recalculation."""
        return self._dirty


# =============================================================================
# ATTRIBUTE SET
# =============================================================================


class AttributeSet:
    """
    A collection of attributes for an entity.

    Manages both regular attributes and derived attributes, handling
    dependency tracking and dirty flag propagation.
    """

    def __init__(self) -> None:
        self._attributes: Dict[str, Attribute] = {}
        self._derived: Dict[str, DerivedAttribute] = {}
        self._dependents: Dict[str, Set[str]] = {}  # attribute -> derived attrs that depend on it
        self._on_change: Optional[Callable[[str, float, float], None]] = None

    def define(
        self,
        name: str,
        base_value: float = 0.0,
        min_value: float = DEFAULT_ATTRIBUTE_MIN,
        max_value: float = DEFAULT_ATTRIBUTE_MAX,
    ) -> Attribute:
        """Define a new attribute."""
        if name in self._attributes or name in self._derived:
            raise ValueError(f"Attribute '{name}' already exists")

        attr = Attribute(
            name=name,
            base_value=base_value,
            min_value=min_value,
            max_value=max_value,
            _on_change=self._on_attribute_change,
        )
        self._attributes[name] = attr
        return attr

    def define_derived(
        self,
        name: str,
        formula: Callable[[Dict[str, float]], float],
        *dependencies: str,
        min_value: float = DEFAULT_ATTRIBUTE_MIN,
        max_value: float = DEFAULT_ATTRIBUTE_MAX,
    ) -> DerivedAttribute:
        """Define a new derived attribute."""
        if name in self._attributes or name in self._derived:
            raise ValueError(f"Attribute '{name}' already exists")

        # Validate dependencies exist
        for dep in dependencies:
            if dep not in self._attributes and dep not in self._derived:
                raise ValueError(f"Dependency '{dep}' does not exist")

        derived = DerivedAttribute.create(
            name,
            formula,
            *dependencies,
            min_value=min_value,
            max_value=max_value,
        )
        self._derived[name] = derived

        # Track reverse dependencies
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(name)

        return derived

    def get(self, name: str) -> float:
        """Get the current value of an attribute."""
        if name in self._attributes:
            return self._attributes[name].current_value
        elif name in self._derived:
            return self._derived[name].calculate(self._get_values_dict())
        else:
            raise KeyError(f"Attribute '{name}' not found")

    def get_attribute(self, name: str) -> Attribute:
        """Get an attribute object."""
        if name not in self._attributes:
            raise KeyError(f"Attribute '{name}' not found")
        return self._attributes[name]

    def get_derived(self, name: str) -> DerivedAttribute:
        """Get a derived attribute object."""
        if name not in self._derived:
            raise KeyError(f"Derived attribute '{name}' not found")
        return self._derived[name]

    def set_base(self, name: str, value: float) -> None:
        """Set the base value of an attribute."""
        if name not in self._attributes:
            raise KeyError(f"Attribute '{name}' not found")
        self._attributes[name].set_base_value(value)

    def add_modifier(
        self,
        name: str,
        operation: ModifierOperation,
        magnitude: float,
        source: Optional[Any] = None,
    ) -> AttributeModifierHandle:
        """Add a modifier to an attribute."""
        if name not in self._attributes:
            raise KeyError(f"Attribute '{name}' not found")

        modifier = AttributeModifier(
            operation=operation,
            magnitude=magnitude,
            source=source,
        )
        return self._attributes[name].add_modifier(modifier)

    def remove_modifier(
        self, handle: AttributeModifierHandle
    ) -> bool:
        """Remove a modifier using its handle."""
        if handle.attribute_name not in self._attributes:
            return False
        return self._attributes[handle.attribute_name].remove_modifier(handle)

    def remove_all_from_source(self, source: Any) -> int:
        """Remove all modifiers from a source across all attributes."""
        count = 0
        for attr in self._attributes.values():
            count += attr.remove_modifiers_from_source(source)
        return count

    def has(self, name: str) -> bool:
        """Check if an attribute exists."""
        return name in self._attributes or name in self._derived

    def names(self) -> FrozenSet[str]:
        """Get all attribute names."""
        return frozenset(self._attributes.keys()) | frozenset(self._derived.keys())

    def attribute_names(self) -> FrozenSet[str]:
        """Get non-derived attribute names."""
        return frozenset(self._attributes.keys())

    def derived_names(self) -> FrozenSet[str]:
        """Get derived attribute names."""
        return frozenset(self._derived.keys())

    def _get_values_dict(self) -> Dict[str, float]:
        """Get all attribute values as a dictionary."""
        values = {}
        for name, attr in self._attributes.items():
            values[name] = attr.current_value
        return values

    def _on_attribute_change(
        self, attr: Attribute, old_value: float, new_value: float
    ) -> None:
        """Handle attribute value changes."""
        # Mark dependent derived attributes as dirty
        if attr.name in self._dependents:
            for derived_name in self._dependents[attr.name]:
                self._derived[derived_name].mark_dirty()

        # Notify external callback
        if self._on_change is not None:
            self._on_change(attr.name, old_value, new_value)

    def __getitem__(self, name: str) -> float:
        return self.get(name)

    def __setitem__(self, name: str, value: float) -> None:
        self.set_base(name, value)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._attributes.keys())

    def __len__(self) -> int:
        return len(self._attributes) + len(self._derived)


# =============================================================================
# COMMON ATTRIBUTE DEFINITIONS
# =============================================================================


def create_standard_attributes() -> AttributeSet:
    """Create an attribute set with common gameplay attributes."""
    attrs = AttributeSet()

    # Vital stats
    attrs.define("health", base_value=100.0, min_value=0.0, max_value=10000.0)
    attrs.define("max_health", base_value=100.0, min_value=1.0, max_value=10000.0)
    attrs.define("mana", base_value=100.0, min_value=0.0, max_value=5000.0)
    attrs.define("max_mana", base_value=100.0, min_value=0.0, max_value=5000.0)
    attrs.define("stamina", base_value=100.0, min_value=0.0, max_value=1000.0)
    attrs.define("max_stamina", base_value=100.0, min_value=0.0, max_value=1000.0)

    # Regeneration
    attrs.define("health_regen", base_value=0.0, min_value=-100.0, max_value=1000.0)
    attrs.define("mana_regen", base_value=1.0, min_value=-100.0, max_value=1000.0)
    attrs.define("stamina_regen", base_value=10.0, min_value=-100.0, max_value=1000.0)

    # Combat stats
    attrs.define("damage", base_value=10.0, min_value=0.0, max_value=100000.0)
    attrs.define("armor", base_value=0.0, min_value=0.0, max_value=10000.0)
    attrs.define("attack_speed", base_value=1.0, min_value=0.1, max_value=10.0)
    attrs.define("critical_chance", base_value=0.05, min_value=0.0, max_value=1.0)
    attrs.define("critical_damage", base_value=1.5, min_value=1.0, max_value=10.0)

    # Movement
    attrs.define("movement_speed", base_value=400.0, min_value=0.0, max_value=2000.0)

    # Cooldown reduction
    attrs.define("cooldown_reduction", base_value=0.0, min_value=0.0, max_value=0.75)

    # Derived: effective damage (damage * (1 + crit_chance * (crit_damage - 1)))
    attrs.define_derived(
        "effective_damage",
        lambda v: v["damage"] * (1 + v["critical_chance"] * (v["critical_damage"] - 1)),
        "damage", "critical_chance", "critical_damage",
        min_value=0.0,
        max_value=1000000.0,
    )

    # Derived: health percentage
    attrs.define_derived(
        "health_percent",
        lambda v: (v["health"] / v["max_health"]) if v["max_health"] > 0 else 0.0,
        "health", "max_health",
        min_value=0.0,
        max_value=1.0,
    )

    # Derived: mana percentage
    attrs.define_derived(
        "mana_percent",
        lambda v: (v["mana"] / v["max_mana"]) if v["max_mana"] > 0 else 0.0,
        "mana", "max_mana",
        min_value=0.0,
        max_value=1.0,
    )

    return attrs


# =============================================================================
# TRACKED ATTRIBUTE SET (AttributeSet with Foundation tracking)
# =============================================================================


class TrackedAttributeSet(AttributeSet):
    """
    AttributeSet with Foundation Tracker integration.

    Extends AttributeSet to automatically track all attribute changes
    via the Foundation Tracker system, enabling:
    - all_dirty() checks across the entire set
    - on_change() subscriptions by attribute type
    - Batch updates for atomic changes
    """

    def __init__(self, tracker: Optional[AttributeTracker] = None) -> None:
        """Initialize with optional custom tracker."""
        super().__init__()
        self._tracker = tracker or attribute_tracker

    @property
    def tracker(self) -> AttributeTracker:
        """Get the attribute tracker for this set."""
        return self._tracker

    def all_dirty(self) -> bool:
        """Check if any attribute in the set is dirty."""
        return self._tracker.all_dirty()

    def clear_all_dirty(self) -> None:
        """Clear dirty flags for all attributes."""
        for attr in self._attributes.values():
            self._tracker.mark_clean(attr)
        for derived in self._derived.values():
            self._tracker.mark_clean(derived)

    def on_change(
        self,
        target: Union[None, str, Attribute],
        callback: AttributeChangeCallback,
    ) -> None:
        """
        Subscribe to attribute changes.

        Args:
            target: None for all, attribute name string, or Attribute instance
            callback: Function(obj, field, old, new) to call on change
        """
        if isinstance(target, Attribute):
            self._tracker.on_change(target, callback)
        elif isinstance(target, str):
            # Subscribe to attribute by name
            attr = self._attributes.get(target)
            if attr:
                self._tracker.on_change(attr, callback)
            else:
                # Type-level subscription
                self._tracker.on_change(target, callback)
        else:
            self._tracker.on_change(None, callback)

    def begin_batch(self) -> None:
        """Begin batch mode - defer all callbacks until end_batch()."""
        self._tracker.begin_batch()

    def end_batch(self) -> None:
        """End batch mode and fire all deferred callbacks."""
        self._tracker.end_batch()

    def _on_attribute_change(
        self, attr: Attribute, old_value: float, new_value: float
    ) -> None:
        """Handle attribute value changes with tracking."""
        # Mark dirty via tracker
        self._tracker.mark_dirty(attr, attr.name, old_value, new_value)

        # Call parent implementation
        super()._on_attribute_change(attr, old_value, new_value)


def create_tracked_standard_attributes(
    tracker: Optional[AttributeTracker] = None,
) -> TrackedAttributeSet:
    """Create a tracked attribute set with common gameplay attributes."""
    attrs = TrackedAttributeSet(tracker=tracker)

    # Vital stats
    attrs.define("health", base_value=100.0, min_value=0.0, max_value=10000.0)
    attrs.define("max_health", base_value=100.0, min_value=1.0, max_value=10000.0)
    attrs.define("mana", base_value=100.0, min_value=0.0, max_value=5000.0)
    attrs.define("max_mana", base_value=100.0, min_value=0.0, max_value=5000.0)
    attrs.define("stamina", base_value=100.0, min_value=0.0, max_value=1000.0)
    attrs.define("max_stamina", base_value=100.0, min_value=0.0, max_value=1000.0)

    # Regeneration
    attrs.define("health_regen", base_value=0.0, min_value=-100.0, max_value=1000.0)
    attrs.define("mana_regen", base_value=1.0, min_value=-100.0, max_value=1000.0)
    attrs.define("stamina_regen", base_value=10.0, min_value=-100.0, max_value=1000.0)

    # Combat stats
    attrs.define("damage", base_value=10.0, min_value=0.0, max_value=100000.0)
    attrs.define("armor", base_value=0.0, min_value=0.0, max_value=10000.0)
    attrs.define("attack_speed", base_value=1.0, min_value=0.1, max_value=10.0)
    attrs.define("critical_chance", base_value=0.05, min_value=0.0, max_value=1.0)
    attrs.define("critical_damage", base_value=1.5, min_value=1.0, max_value=10.0)

    # Movement
    attrs.define("movement_speed", base_value=400.0, min_value=0.0, max_value=2000.0)

    # Cooldown reduction
    attrs.define("cooldown_reduction", base_value=0.0, min_value=0.0, max_value=0.75)

    # Derived: effective damage
    attrs.define_derived(
        "effective_damage",
        lambda v: v["damage"] * (1 + v["critical_chance"] * (v["critical_damage"] - 1)),
        "damage", "critical_chance", "critical_damage",
        min_value=0.0,
        max_value=1000000.0,
    )

    # Derived: health percentage
    attrs.define_derived(
        "health_percent",
        lambda v: (v["health"] / v["max_health"]) if v["max_health"] > 0 else 0.0,
        "health", "max_health",
        min_value=0.0,
        max_value=1.0,
    )

    # Derived: mana percentage
    attrs.define_derived(
        "mana_percent",
        lambda v: (v["mana"] / v["max_mana"]) if v["max_mana"] > 0 else 0.0,
        "mana", "max_mana",
        min_value=0.0,
        max_value=1.0,
    )

    return attrs


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core attribute classes
    "Attribute",
    "AttributeModifier",
    "AttributeModifierHandle",
    "AttributeSet",
    "DerivedAttribute",
    "create_standard_attributes",
    # Foundation Tracker integration
    "AttributeTracker",
    "attribute_tracker",
    "TrackedAttributeDescriptor",
    "tracked_attribute",
    "AttributeChangeCallback",
    # Tracked ability attributes
    "TrackedAbilityAttribute",
    "TrackedVitalAttribute",
    "TrackedCooldownAttribute",
    # Tracked attribute set
    "TrackedAttributeSet",
    "create_tracked_standard_attributes",
    # Constants for availability checks
    "FOUNDATION_AVAILABLE",
    "TRINITY_DESCRIPTORS_AVAILABLE",
]
