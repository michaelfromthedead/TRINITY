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
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
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

T = TypeVar("T")

# Check if Foundation is available for tracker integration
try:
    from foundation import Registry, registry
    from foundation import tracker as foundation_tracker
    FOUNDATION_AVAILABLE = True
except ImportError:
    FOUNDATION_AVAILABLE = False
    foundation_tracker = None  # type: ignore


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
# ATTRIBUTE TRACKER (Foundation Integration)
# =============================================================================


# Type alias for attribute change callbacks
AttributeChangeCallback = Callable[[Any, str, float, float], None]


class AttributeTracker:
    """
    Tracks attribute changes across multiple objects.

    Provides dirty flag tracking, callbacks, batch mode, and versioning.
    Integrates with the Foundation framework for change notifications.
    """

    def __init__(self) -> None:
        # _dirty stores {obj_id: (weak_ref, {field_names})} for garbage collection
        self._dirty: Dict[int, tuple] = {}
        self._callbacks: Dict[int, Dict[str, List[AttributeChangeCallback]]] = {}
        self._type_callbacks: Dict[type, List[AttributeChangeCallback]] = {}
        self._batch_mode: bool = False
        self._batch_changes: List[tuple] = []  # [(obj, field, old, new), ...]
        self._version: int = 0

    @property
    def version(self) -> int:
        """Get the current version counter."""
        return self._version

    def _cleanup_dead_refs(self) -> None:
        """Remove entries for garbage-collected objects."""
        dead_ids = [
            obj_id for obj_id, (ref, _) in self._dirty.items()
            if ref() is None
        ]
        for obj_id in dead_ids:
            del self._dirty[obj_id]

    def mark_dirty(
        self,
        obj: Any,
        field: str,
        old_value: float,
        new_value: float,
    ) -> None:
        """Mark a field as dirty and notify callbacks."""
        obj_id = id(obj)

        # Track object reference for get_all_dirty_objects using weak refs
        try:
            ref = weakref.ref(obj)
        except TypeError:
            # Object doesn't support weak references
            ref = lambda: obj  # type: ignore

        # Track dirty state with weak reference
        if obj_id not in self._dirty:
            self._dirty[obj_id] = (ref, set())
        self._dirty[obj_id][1].add(field)

        # Increment version
        self._version += 1

        if self._batch_mode:
            self._batch_changes.append((obj, field, old_value, new_value))
        else:
            self._notify_callbacks(obj, field, old_value, new_value)

    def is_dirty(self, obj: Any, field: Optional[str] = None) -> bool:
        """Check if an object (or specific field) is dirty."""
        obj_id = id(obj)
        if obj_id not in self._dirty:
            return False
        ref, fields = self._dirty[obj_id]
        if field is None:
            return len(fields) > 0
        return field in fields

    def clear_dirty(self, obj: Any, field: Optional[str] = None) -> None:
        """Clear dirty flags for an object or specific field."""
        obj_id = id(obj)
        if obj_id not in self._dirty:
            return
        ref, fields = self._dirty[obj_id]
        if field is None:
            fields.clear()
        else:
            fields.discard(field)

    def mark_clean(self, obj: Any, field: Optional[str] = None) -> None:
        """Alias for clear_dirty - clears dirty flags for an object or field."""
        self.clear_dirty(obj, field)

    def all_dirty(self) -> bool:
        """Check if any object has dirty fields."""
        self._cleanup_dead_refs()
        for obj_id, (ref, fields) in self._dirty.items():
            if ref() is not None and fields:
                return True
        return False

    def get_all_dirty_objects(self) -> List[Any]:
        """Get all objects that have dirty fields."""
        self._cleanup_dead_refs()
        result = []
        for obj_id, (ref, fields) in self._dirty.items():
            obj = ref()
            if obj is not None and fields:
                result.append(obj)
        return result

    def on_change(
        self,
        field_or_obj: Optional[Any],
        callback: AttributeChangeCallback,
    ) -> None:
        """
        Subscribe to change notifications.

        Args:
            field_or_obj: If None, subscribe to all changes.
                         If a string, subscribe to that field name.
                         If an object, subscribe to that object's changes.
            callback: Function to call on change (obj, field, old, new).
        """
        if field_or_obj is None:
            # Subscribe to all changes
            if not hasattr(self, "_global_callbacks"):
                self._global_callbacks: List[AttributeChangeCallback] = []
            self._global_callbacks.append(callback)
        elif isinstance(field_or_obj, str):
            # Subscribe to field name
            if not hasattr(self, "_field_callbacks"):
                self._field_callbacks: Dict[str, List[AttributeChangeCallback]] = {}
            if field_or_obj not in self._field_callbacks:
                self._field_callbacks[field_or_obj] = []
            self._field_callbacks[field_or_obj].append(callback)
        else:
            # Subscribe to object
            self.add_callback(field_or_obj, "*", callback)

    def off_change(
        self,
        callback: AttributeChangeCallback,
        field_or_obj: Optional[Any] = None,
    ) -> bool:
        """
        Unsubscribe from change notifications.

        Args:
            callback: The callback to remove.
            field_or_obj: Optional filter - if None, removes from global callbacks.
                         If a string, removes from that field's callbacks.
                         If an object, removes from that object's callbacks.

        Returns True if callback was removed.
        """
        removed = False

        # Try to remove from global callbacks
        if hasattr(self, "_global_callbacks") and callback in self._global_callbacks:
            self._global_callbacks.remove(callback)
            removed = True

        # Try to remove from field callbacks
        if hasattr(self, "_field_callbacks"):
            for field_name, callbacks in self._field_callbacks.items():
                if callback in callbacks:
                    callbacks.remove(callback)
                    removed = True

        # Try to remove from object callbacks
        for obj_id, field_callbacks in self._callbacks.items():
            for field, callbacks in field_callbacks.items():
                if callback in callbacks:
                    callbacks.remove(callback)
                    removed = True

        return removed

    def add_callback(
        self,
        obj: Any,
        field: str,
        callback: AttributeChangeCallback,
    ) -> None:
        """Add a callback for changes to a specific field."""
        obj_id = id(obj)
        if obj_id not in self._callbacks:
            self._callbacks[obj_id] = {}
        if field not in self._callbacks[obj_id]:
            self._callbacks[obj_id][field] = []
        self._callbacks[obj_id][field].append(callback)

    def remove_callback(
        self,
        obj: Any,
        field: str,
        callback: AttributeChangeCallback,
    ) -> bool:
        """Remove a callback. Returns True if removed."""
        obj_id = id(obj)
        if obj_id not in self._callbacks:
            return False
        if field not in self._callbacks[obj_id]:
            return False
        try:
            self._callbacks[obj_id][field].remove(callback)
            return True
        except ValueError:
            return False

    def add_type_callback(
        self,
        obj_type: type,
        callback: AttributeChangeCallback,
    ) -> None:
        """Add a callback for all objects of a type."""
        if obj_type not in self._type_callbacks:
            self._type_callbacks[obj_type] = []
        self._type_callbacks[obj_type].append(callback)

    @property
    def in_batch(self) -> bool:
        """Check if currently in batch mode."""
        return self._batch_mode

    def begin_batch(self) -> None:
        """Begin batch mode - changes are collected but not notified."""
        if self._batch_mode:
            raise RuntimeError("Already in batch mode")
        self._batch_mode = True
        self._batch_changes.clear()

    def end_batch(self) -> None:
        """End batch mode and notify all collected changes."""
        if not self._batch_mode:
            raise RuntimeError("Not in batch mode")
        self._batch_mode = False
        for obj, field, old_value, new_value in self._batch_changes:
            self._notify_callbacks(obj, field, old_value, new_value)
        self._batch_changes.clear()

    def dirty_fields(self, obj: Any) -> Set[str]:
        """Get the set of dirty field names for an object."""
        obj_id = id(obj)
        if obj_id not in self._dirty:
            return set()
        ref, fields = self._dirty[obj_id]
        return set(fields)

    def _notify_callbacks(
        self,
        obj: Any,
        field: str,
        old_value: float,
        new_value: float,
    ) -> None:
        """Notify all relevant callbacks of a change."""
        obj_id = id(obj)

        def _safe_call(callback: AttributeChangeCallback) -> None:
            """Call a callback safely, catching any exceptions."""
            try:
                callback(obj, field, old_value, new_value)
            except Exception:
                # Silently ignore callback errors to prevent crashes
                pass

        # Global callbacks (on_change(None, callback))
        if hasattr(self, "_global_callbacks"):
            for callback in self._global_callbacks:
                _safe_call(callback)

        # Field-specific callbacks (on_change("field_name", callback))
        if hasattr(self, "_field_callbacks") and field in self._field_callbacks:
            for callback in self._field_callbacks[field]:
                _safe_call(callback)

        # Instance callbacks
        if obj_id in self._callbacks:
            # Check for field-specific callbacks
            if field in self._callbacks[obj_id]:
                for callback in self._callbacks[obj_id][field]:
                    _safe_call(callback)
            # Check for wildcard callbacks (subscribe to all fields for this object)
            if "*" in self._callbacks[obj_id]:
                for callback in self._callbacks[obj_id]["*"]:
                    _safe_call(callback)

        # Type callbacks
        for obj_type, callbacks in self._type_callbacks.items():
            if isinstance(obj, obj_type):
                for callback in callbacks:
                    _safe_call(callback)


# Global default tracker instance
_default_tracker = AttributeTracker()

# Expose the global tracker as attribute_tracker
attribute_tracker = _default_tracker


# =============================================================================
# TRACKED ATTRIBUTE DESCRIPTOR
# =============================================================================


class TrackedAttributeDescriptor:
    """
    Descriptor for tracked attributes that automatically notify changes.

    Usage:
        class Character:
            health = TrackedAttributeDescriptor("health", default=100.0)
    """

    def __init__(
        self,
        name: str,
        default: float = 0.0,
        min_value: float = DEFAULT_ATTRIBUTE_MIN,
        max_value: float = DEFAULT_ATTRIBUTE_MAX,
        tracker: Optional[AttributeTracker] = None,
    ) -> None:
        self.name = name
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        # Use provided tracker or global default
        self._tracker = tracker if tracker is not None else attribute_tracker

    @property
    def tracker(self) -> AttributeTracker:
        return self._tracker

    @property
    def _storage_attr(self) -> str:
        """Get the storage attribute name (for test compatibility)."""
        return f"_tracked_attr_{self.name}"

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr_name = f"_tracked_{name}"
        # Register this attribute on the owner class
        if not hasattr(owner, "_tracked_attributes"):
            owner._tracked_attributes = set()
        owner._tracked_attributes.add(name)

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        if obj is None:
            # When accessed from class, return the descriptor itself
            return self
        return getattr(obj, self.attr_name, self.default)

    def clear_dirty(self, obj: Any) -> None:
        """Clear the dirty flag for this attribute on the given object."""
        self.tracker.mark_clean(obj, self.name)

    def is_dirty(self, obj: Any) -> bool:
        """Check if this attribute is dirty on the given object."""
        return self.tracker.is_dirty(obj, self.name)

    def mark_dirty(self, obj: Any) -> None:
        """Mark this attribute as dirty on the given object."""
        old_value = getattr(obj, self.attr_name, self.default)
        self.tracker.mark_dirty(obj, self.name, old_value, old_value)

    def __delete__(self, obj: Any) -> None:
        """Reset attribute to default value."""
        old_value = getattr(obj, self.attr_name, self.default)
        setattr(obj, self.attr_name, self.default)
        if abs(self.default - old_value) > EPSILON:
            self.tracker.mark_dirty(obj, self.name, old_value, self.default)

    def __set__(self, obj: Any, value: float) -> None:
        old_value = getattr(obj, self.attr_name, self.default)
        # Clamp value
        new_value = max(self.min_value, min(self.max_value, value))
        setattr(obj, self.attr_name, new_value)

        if abs(new_value - old_value) > EPSILON:
            self.tracker.mark_dirty(obj, self.name, old_value, new_value)
            # Also notify Foundation tracker if available
            if FOUNDATION_AVAILABLE and foundation_tracker is not None:
                foundation_tracker.mark_dirty(obj, self.name, old_value, new_value)


_NOT_SET = object()  # Sentinel for detecting unset values


def tracked_attribute(
    name: str,
    default: float = 0.0,
    min_value: Any = _NOT_SET,
    max_value: Any = _NOT_SET,
    tracker: Optional[AttributeTracker] = None,
    # Support alternate naming for min/max
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> TrackedAttributeDescriptor:
    """Create a tracked attribute descriptor."""
    # Determine actual min - use 'min' alias if provided, else min_value if provided
    if min is not None:
        actual_min = min
    elif min_value is not _NOT_SET:
        actual_min = min_value
    else:
        # No min specified - use very low value to allow any negative
        actual_min = -float('inf')

    # Determine actual max - use 'max' alias if provided, else max_value if provided
    if max is not None:
        actual_max = max
    elif max_value is not _NOT_SET:
        actual_max = max_value
    else:
        # No max specified - use very high value
        actual_max = float('inf')

    return TrackedAttributeDescriptor(
        name=name,
        default=default,
        min_value=actual_min,
        max_value=actual_max,
        tracker=tracker,
    )


# =============================================================================
# TRACKED ABILITY ATTRIBUTES
# =============================================================================


class TrackedAbilityAttribute(TrackedAttributeDescriptor):
    """Tracked attribute for ability-related values (cooldowns, charges, etc.)."""

    def __init__(
        self,
        name: str,
        default: float = 0.0,
        min_value: float = 0.0,
        max_value: float = DEFAULT_ATTRIBUTE_MAX,
        ability_id: Optional[str] = None,
    ) -> None:
        super().__init__(name, default, min_value, max_value)
        self.ability_id = ability_id


class TrackedVitalAttribute:
    """
    Tracked attribute for vital stats (health, mana, stamina).

    Can be used as a standalone object with current/maximum values,
    apply_damage, apply_healing, and regenerate methods.
    """

    def __init__(
        self,
        current: float = 100.0,
        maximum: float = 100.0,
        regen_rate: float = 0.0,
        tracker: Optional[AttributeTracker] = None,
        # Legacy params for backwards compatibility
        name: Optional[str] = None,
        default: Optional[float] = None,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> None:
        self._current = min(current, maximum)
        self._maximum = max(maximum, 1.0)  # Ensure at least 1.0
        self._regen_rate = regen_rate
        self._tracker = tracker or _default_tracker
        self._dirty = False

    @property
    def current(self) -> float:
        """Get the current value."""
        return self._current

    @current.setter
    def current(self, value: float) -> None:
        """Set the current value (clamped to 0-maximum)."""
        old = self._current
        self._current = max(0.0, min(value, self._maximum))
        if abs(self._current - old) > EPSILON:
            self._dirty = True

    @property
    def maximum(self) -> float:
        """Get the maximum value."""
        return self._maximum

    @maximum.setter
    def maximum(self, value: float) -> None:
        """Set the maximum value."""
        self._maximum = max(value, 1.0)
        # Clamp current if now exceeds max
        if self._current > self._maximum:
            self._current = self._maximum

    @property
    def regen_rate(self) -> float:
        """Get the regeneration rate per second."""
        return self._regen_rate

    @regen_rate.setter
    def regen_rate(self, value: float) -> None:
        """Set the regeneration rate."""
        self._regen_rate = value

    @property
    def percent(self) -> float:
        """Get current value as a percentage of maximum (0.0-1.0)."""
        if self._maximum <= 0:
            return 0.0
        return self._current / self._maximum

    def apply_damage(self, amount: float) -> float:
        """
        Apply damage, reducing current value.

        Args:
            amount: The damage amount (positive value).

        Returns:
            The actual damage dealt (may be less if current was lower).
        """
        actual = min(amount, self._current)
        self.current = self._current - amount
        return actual

    def apply_healing(self, amount: float) -> float:
        """
        Apply healing, increasing current value.

        Args:
            amount: The healing amount (positive value).

        Returns:
            The actual healing done (may be less if close to max).
        """
        before = self._current
        self.current = self._current + amount
        return self._current - before

    def regenerate(self, delta_time: float) -> float:
        """
        Apply regeneration over time.

        Args:
            delta_time: Time elapsed in seconds.

        Returns:
            The amount regenerated (positive) or degenerated (negative).
        """
        regen_amount = self._regen_rate * delta_time
        if regen_amount >= 0:
            return self.apply_healing(regen_amount)
        else:
            # Negative regen = degeneration
            actual = self.apply_damage(-regen_amount)
            return -actual if actual > 0 else regen_amount

    def is_dirty(self) -> bool:
        """Check if the value has changed since last clear."""
        return self._dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False


class TrackedCooldownAttribute:
    """
    Tracked attribute for cooldown management.

    Manages ability cooldowns with duration, reduction, and tick-based updates.
    """

    def __init__(
        self,
        duration: float = 1.0,
        reduction: float = 0.0,
        tracker: Optional[AttributeTracker] = None,
        # Legacy params for backwards compatibility
        name: Optional[str] = None,
        base_cooldown: Optional[float] = None,
        min_cooldown: float = 0.0,
    ) -> None:
        self._duration = duration
        self._reduction = min(0.75, max(0.0, reduction))  # Clamp 0-75%
        self._remaining = 0.0
        self._tracker = tracker or _default_tracker

    @property
    def duration(self) -> float:
        """Get the base cooldown duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the base cooldown duration."""
        self._duration = max(0.0, value)

    @property
    def reduction(self) -> float:
        """Get the cooldown reduction percentage (0.0-0.75)."""
        return self._reduction

    @reduction.setter
    def reduction(self, value: float) -> None:
        """Set the cooldown reduction percentage."""
        self._reduction = min(0.75, max(0.0, value))

    @property
    def remaining(self) -> float:
        """Get the remaining cooldown time."""
        return self._remaining

    @property
    def effective_duration(self) -> float:
        """Get the effective cooldown duration after reduction."""
        return self._duration * (1.0 - self._reduction)

    @property
    def is_ready(self) -> bool:
        """Check if the cooldown has elapsed (ready to use)."""
        return self._remaining <= 0.0

    @property
    def progress(self) -> float:
        """Get the cooldown progress (0.0 = just started, 1.0 = ready)."""
        eff_dur = self.effective_duration
        if eff_dur <= 0:
            return 1.0
        if self._remaining <= 0:
            return 1.0
        return 1.0 - (self._remaining / eff_dur)

    def start(self) -> None:
        """Start the cooldown (set remaining to effective duration)."""
        self._remaining = self.effective_duration

    def tick(self, delta_time: float) -> bool:
        """
        Update the cooldown by elapsed time.

        Args:
            delta_time: Time elapsed in seconds.

        Returns:
            True if the cooldown just became ready (was active, now ready).
        """
        if self._remaining <= 0:
            return False

        was_active = self._remaining > 0
        self._remaining = max(0.0, self._remaining - delta_time)
        now_ready = self._remaining <= 0

        return was_active and now_ready

    def reset(self) -> None:
        """Reset the cooldown (set remaining to 0, making it ready)."""
        self._remaining = 0.0

    def start_cooldown(self, obj: Any = None, cdr: float = 0.0) -> None:
        """Legacy method - start the cooldown with optional reduction."""
        self._reduction = min(0.75, cdr)
        self.start()


# =============================================================================
# TRACKED ATTRIBUTE SET
# =============================================================================


class TrackedAttributeSet(AttributeSet):
    """
    Attribute set with Foundation tracking integration.

    Automatically tracks changes and notifies the global tracker.
    """

    def __init__(self, tracker: Optional[AttributeTracker] = None) -> None:
        super().__init__()
        self._tracker = tracker or _default_tracker
        self._tracked_owner: Optional[Any] = None

    @property
    def tracker(self) -> AttributeTracker:
        """Get the associated tracker."""
        return self._tracker

    def bind_to(self, owner: Any) -> "TrackedAttributeSet":
        """Bind this attribute set to an owner object for tracking."""
        self._tracked_owner = owner
        return self

    def all_dirty(self) -> bool:
        """Check if any attribute is dirty."""
        return self._tracker.all_dirty()

    def clear_all_dirty(self) -> None:
        """Clear all dirty flags for all tracked attributes."""
        # Clear for tracked owner if set
        if self._tracked_owner is not None:
            self._tracker.mark_clean(self._tracked_owner)
        # Clear for all attributes
        for attr in self._attributes.values():
            self._tracker.mark_clean(attr)

    def on_change(
        self,
        field_or_obj: Optional[Any],
        callback: Callable[[Any, str, float, float], None],
    ) -> None:
        """Subscribe to change notifications."""
        self._tracker.on_change(field_or_obj, callback)

    def begin_batch(self) -> None:
        """Begin batch mode - changes are collected but not notified."""
        self._tracker.begin_batch()

    def end_batch(self) -> None:
        """End batch mode and notify all collected changes."""
        self._tracker.end_batch()

    def _on_attribute_change(
        self, attr: Attribute, old_value: float, new_value: float
    ) -> None:
        """Handle attribute value changes with tracking."""
        # Call parent implementation
        super()._on_attribute_change(attr, old_value, new_value)

        # Mark the attribute itself as dirty
        self._tracker.mark_dirty(
            attr,
            attr.name,
            old_value,
            new_value,
        )

        # Also notify tracker if we have a tracked owner
        if self._tracked_owner is not None:
            self._tracker.mark_dirty(
                self._tracked_owner,
                attr.name,
                old_value,
                new_value,
            )


def create_tracked_standard_attributes(
    owner: Optional[Any] = None,
    tracker: Optional[AttributeTracker] = None,
) -> TrackedAttributeSet:
    """Create a tracked attribute set with common gameplay attributes."""
    attrs = TrackedAttributeSet(tracker)

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

    # Derived attributes
    attrs.define_derived(
        "effective_damage",
        lambda v: v["damage"] * (1 + v["critical_chance"] * (v["critical_damage"] - 1)),
        "damage", "critical_chance", "critical_damage",
        min_value=0.0,
        max_value=1000000.0,
    )

    attrs.define_derived(
        "health_percent",
        lambda v: (v["health"] / v["max_health"]) if v["max_health"] > 0 else 0.0,
        "health", "max_health",
        min_value=0.0,
        max_value=1.0,
    )

    attrs.define_derived(
        "mana_percent",
        lambda v: (v["mana"] / v["max_mana"]) if v["max_mana"] > 0 else 0.0,
        "mana", "max_mana",
        min_value=0.0,
        max_value=1.0,
    )

    if owner is not None:
        attrs.bind_to(owner)

    return attrs


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Attribute",
    "AttributeChangeCallback",
    "AttributeModifier",
    "AttributeModifierHandle",
    "AttributeSet",
    "AttributeTracker",
    "DerivedAttribute",
    "EPSILON",
    "FOUNDATION_AVAILABLE",
    "TrackedAbilityAttribute",
    "TrackedAttributeDescriptor",
    "TrackedAttributeSet",
    "TrackedCooldownAttribute",
    "TrackedVitalAttribute",
    "attribute_tracker",
    "create_standard_attributes",
    "create_tracked_standard_attributes",
    "tracked_attribute",
]
