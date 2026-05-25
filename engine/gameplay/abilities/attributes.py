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
            name=name,
            formula=formula,
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
# EXPORTS
# =============================================================================

__all__ = [
    "Attribute",
    "AttributeModifier",
    "AttributeModifierHandle",
    "AttributeSet",
    "DerivedAttribute",
    "create_standard_attributes",
]
