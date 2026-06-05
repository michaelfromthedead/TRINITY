"""
Stats/Attributes Component - Base values, modifiers, and derived stats.

Provides a flexible attribute system with support for base values,
stacking modifiers, and derived/computed statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, TYPE_CHECKING

from trinity.descriptors import (
    TrackedDescriptor,
    clear_dirty,
    is_dirty,
)

if TYPE_CHECKING:
    from foundation import to_dict, from_dict


class ModifierType(Enum):
    """Type of stat modifier affecting how it stacks."""
    FLAT = auto()           # Added directly to base (+10)
    PERCENT_BASE = auto()   # Percentage of base value (+10%)
    PERCENT_TOTAL = auto()  # Percentage of total after other mods (+10% final)
    OVERRIDE = auto()       # Replaces the value entirely
    MULTIPLY = auto()       # Multiplicative (*1.5)


class ModifierSource(Enum):
    """Source category for modifier tracking."""
    EQUIPMENT = auto()
    BUFF = auto()
    DEBUFF = auto()
    SKILL = auto()
    PASSIVE = auto()
    TEMPORARY = auto()
    ENVIRONMENT = auto()
    LEVEL = auto()
    ITEM = auto()
    OTHER = auto()


@dataclass
class StatModifier:
    """
    A modifier that affects a stat value.

    Modifiers can be stacked and are applied in order:
    1. OVERRIDE (highest priority wins)
    2. FLAT
    3. PERCENT_BASE
    4. MULTIPLY
    5. PERCENT_TOTAL
    """
    value: float
    modifier_type: ModifierType = ModifierType.FLAT
    source: ModifierSource = ModifierSource.OTHER
    source_id: str = ""  # Identifier for the source (equipment ID, buff ID, etc.)
    priority: int = 0    # For ordering same-type modifiers
    duration: float = -1.0  # -1 = permanent
    stacks: int = 1      # Number of stacks (multiplies value)
    max_stacks: int = 1  # Maximum stacks allowed
    tag: str = ""        # Optional tag for filtering

    def get_total_value(self) -> float:
        """Get total value including stacks."""
        return self.value * self.stacks

    def can_add_stack(self) -> bool:
        """Check if more stacks can be added."""
        return self.stacks < self.max_stacks

    def add_stack(self) -> bool:
        """Add a stack if possible. Returns True if successful."""
        if self.can_add_stack():
            self.stacks += 1
            return True
        return False

    def remove_stack(self) -> bool:
        """Remove a stack. Returns True if stacks remain, False if should be removed."""
        self.stacks -= 1
        return self.stacks > 0


@dataclass
class Stat:
    """
    A single stat with base value and modifiers.

    Supports caching of computed values for performance.
    """
    name: str
    base_value: float = 0.0
    min_value: float = float("-inf")
    max_value: float = float("inf")
    modifiers: List[StatModifier] = field(default_factory=list)

    # Cached values
    _cached_value: float = 0.0
    _cache_dirty: bool = True

    @property
    def value(self) -> float:
        """Get the final computed value."""
        if self._cache_dirty:
            self._compute_value()
        return self._cached_value

    def _compute_value(self) -> None:
        """Compute the final value from base and modifiers."""
        # Check for override first
        override_mods = [
            m for m in self.modifiers
            if m.modifier_type == ModifierType.OVERRIDE
        ]
        if override_mods:
            # Use highest priority override
            override_mods.sort(key=lambda m: m.priority, reverse=True)
            self._cached_value = max(self.min_value, min(self.max_value,
                                                         override_mods[0].get_total_value()))
            self._cache_dirty = False
            return

        # Start with base value
        result = self.base_value

        # Apply FLAT modifiers
        for mod in self.modifiers:
            if mod.modifier_type == ModifierType.FLAT:
                result += mod.get_total_value()

        # Apply PERCENT_BASE modifiers (multiplicative)
        percent_base = 1.0
        for mod in self.modifiers:
            if mod.modifier_type == ModifierType.PERCENT_BASE:
                percent_base += mod.get_total_value() / 100.0
        result = self.base_value * percent_base + (result - self.base_value)

        # Apply MULTIPLY modifiers
        for mod in self.modifiers:
            if mod.modifier_type == ModifierType.MULTIPLY:
                result *= mod.get_total_value()

        # Apply PERCENT_TOTAL modifiers
        for mod in self.modifiers:
            if mod.modifier_type == ModifierType.PERCENT_TOTAL:
                result *= (1.0 + mod.get_total_value() / 100.0)

        # Clamp to bounds
        self._cached_value = max(self.min_value, min(self.max_value, result))
        self._cache_dirty = False

    def add_modifier(self, modifier: StatModifier) -> bool:
        """
        Add a modifier to this stat.

        Args:
            modifier: The modifier to add

        Returns:
            True if added, False if stacking on existing modifier
        """
        # Check for existing modifier from same source
        for existing in self.modifiers:
            if (existing.source_id == modifier.source_id and
                existing.modifier_type == modifier.modifier_type and
                modifier.source_id):
                # Stack on existing
                if existing.add_stack():
                    self._cache_dirty = True
                    return False
                # At max stacks, don't add
                return False

        self.modifiers.append(modifier)
        self._cache_dirty = True
        return True

    def remove_modifier(self, source_id: str, modifier_type: Optional[ModifierType] = None) -> bool:
        """
        Remove a modifier by source ID.

        Args:
            source_id: Source identifier
            modifier_type: Optional type filter

        Returns:
            True if a modifier was removed
        """
        original_count = len(self.modifiers)
        self.modifiers = [
            m for m in self.modifiers
            if not (m.source_id == source_id and
                   (modifier_type is None or m.modifier_type == modifier_type))
        ]
        if len(self.modifiers) != original_count:
            self._cache_dirty = True
            return True
        return False

    def remove_modifiers_by_source(self, source: ModifierSource) -> int:
        """Remove all modifiers from a source type. Returns count removed."""
        original_count = len(self.modifiers)
        self.modifiers = [m for m in self.modifiers if m.source != source]
        removed = original_count - len(self.modifiers)
        if removed > 0:
            self._cache_dirty = True
        return removed

    def remove_modifiers_by_tag(self, tag: str) -> int:
        """Remove all modifiers with a specific tag. Returns count removed."""
        original_count = len(self.modifiers)
        self.modifiers = [m for m in self.modifiers if m.tag != tag]
        removed = original_count - len(self.modifiers)
        if removed > 0:
            self._cache_dirty = True
        return removed

    def clear_modifiers(self) -> None:
        """Remove all modifiers."""
        self.modifiers.clear()
        self._cache_dirty = True

    def set_base_value(self, value: float) -> None:
        """Set the base value."""
        self.base_value = value
        self._cache_dirty = True

    def invalidate_cache(self) -> None:
        """Mark cache as dirty (forces recomputation)."""
        self._cache_dirty = True

    def get_modifier_total(self, modifier_type: ModifierType) -> float:
        """Get total value of all modifiers of a type."""
        return sum(
            m.get_total_value()
            for m in self.modifiers
            if m.modifier_type == modifier_type
        )


class StatsComponent:
    """
    Stats/Attributes component with base values and modifiers.

    Features:
    - Dynamic stat registration
    - Multiple modifier types (flat, percent, multiply)
    - Modifier stacking and duration
    - Derived/computed stats
    - Stat change callbacks
    - Serialization support

    Usage:
        stats = StatsComponent()
        stats.register_stat("health", base_value=100, min_value=0)
        stats.register_stat("damage", base_value=10)

        # Add modifier
        stats.add_modifier("damage", StatModifier(5, ModifierType.FLAT))

        # Get value
        damage = stats.get_value("damage")  # 15
    """

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_stats",
        "_derived_stats",
        "_on_stat_changed",
        "_entity_id",
    )

    def __init__(self, entity_id: Optional[str] = None) -> None:
        """
        Initialize the stats component.

        Args:
            entity_id: Optional entity ID for tracking
        """
        self._stats: Dict[str, Stat] = {}
        self._derived_stats: Dict[str, Callable[[], float]] = {}
        self._on_stat_changed: List[Callable[[str, float, float], None]] = []
        self._entity_id = entity_id

    # =========================================================================
    # STAT REGISTRATION
    # =========================================================================

    def register_stat(
        self,
        name: str,
        base_value: float = 0.0,
        min_value: float = float("-inf"),
        max_value: float = float("inf"),
    ) -> Stat:
        """
        Register a new stat.

        Args:
            name: Stat name (case-insensitive)
            base_value: Initial base value
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            The created stat
        """
        name = name.lower()
        stat = Stat(
            name=name,
            base_value=base_value,
            min_value=min_value,
            max_value=max_value,
        )
        self._stats[name] = stat
        return stat

    def unregister_stat(self, name: str) -> bool:
        """Unregister a stat. Returns True if found and removed."""
        name = name.lower()
        if name in self._stats:
            del self._stats[name]
            return True
        return False

    def has_stat(self, name: str) -> bool:
        """Check if a stat exists."""
        return name.lower() in self._stats or name.lower() in self._derived_stats

    def get_stat(self, name: str) -> Optional[Stat]:
        """Get a stat object by name."""
        return self._stats.get(name.lower())

    def get_all_stats(self) -> Dict[str, Stat]:
        """Get all registered stats."""
        return dict(self._stats)

    def get_stat_names(self) -> List[str]:
        """Get all stat names."""
        return list(self._stats.keys())

    # =========================================================================
    # DERIVED STATS
    # =========================================================================

    def register_derived_stat(self, name: str, compute_func: Callable[[], float]) -> None:
        """
        Register a derived stat computed from other stats.

        Args:
            name: Stat name
            compute_func: Function that returns the computed value
        """
        self._derived_stats[name.lower()] = compute_func

    def unregister_derived_stat(self, name: str) -> bool:
        """Unregister a derived stat. Returns True if found and removed."""
        name = name.lower()
        if name in self._derived_stats:
            del self._derived_stats[name]
            return True
        return False

    # =========================================================================
    # VALUE ACCESS
    # =========================================================================

    def get_value(self, name: str, default: float = 0.0) -> float:
        """
        Get the current value of a stat.

        Args:
            name: Stat name
            default: Default value if stat doesn't exist

        Returns:
            Current stat value
        """
        name = name.lower()

        # Check derived stats first
        if name in self._derived_stats:
            return self._derived_stats[name]()

        # Check regular stats
        stat = self._stats.get(name)
        return stat.value if stat else default

    def get_base_value(self, name: str, default: float = 0.0) -> float:
        """Get the base value of a stat (without modifiers)."""
        stat = self._stats.get(name.lower())
        return stat.base_value if stat else default

    def set_base_value(self, name: str, value: float) -> bool:
        """
        Set the base value of a stat.

        Args:
            name: Stat name
            value: New base value

        Returns:
            True if stat exists and was modified
        """
        stat = self._stats.get(name.lower())
        if stat is None:
            return False

        old_value = stat.value
        stat.set_base_value(value)
        new_value = stat.value

        if old_value != new_value:
            self._notify_change(name, old_value, new_value)

        return True

    def modify_base_value(self, name: str, delta: float) -> bool:
        """
        Modify the base value by a delta.

        Args:
            name: Stat name
            delta: Amount to add to base value

        Returns:
            True if stat exists and was modified
        """
        stat = self._stats.get(name.lower())
        if stat is None:
            return False

        return self.set_base_value(name, stat.base_value + delta)

    def __getitem__(self, name: str) -> float:
        """Get stat value using indexing syntax."""
        return self.get_value(name)

    def __setitem__(self, name: str, value: float) -> None:
        """Set base value using indexing syntax."""
        if not self.set_base_value(name, value):
            # Auto-register if doesn't exist
            self.register_stat(name, base_value=value)

    # =========================================================================
    # MODIFIERS
    # =========================================================================

    def add_modifier(self, stat_name: str, modifier: StatModifier) -> bool:
        """
        Add a modifier to a stat.

        Args:
            stat_name: Stat to modify
            modifier: Modifier to add

        Returns:
            True if modifier was added (not stacked)
        """
        stat = self._stats.get(stat_name.lower())
        if stat is None:
            return False

        old_value = stat.value
        result = stat.add_modifier(modifier)
        new_value = stat.value

        if old_value != new_value:
            self._notify_change(stat_name, old_value, new_value)

        return result

    def remove_modifier(
        self,
        stat_name: str,
        source_id: str,
        modifier_type: Optional[ModifierType] = None,
    ) -> bool:
        """
        Remove a modifier from a stat.

        Args:
            stat_name: Stat name
            source_id: Modifier source ID
            modifier_type: Optional type filter

        Returns:
            True if a modifier was removed
        """
        stat = self._stats.get(stat_name.lower())
        if stat is None:
            return False

        old_value = stat.value
        result = stat.remove_modifier(source_id, modifier_type)
        new_value = stat.value

        if old_value != new_value:
            self._notify_change(stat_name, old_value, new_value)

        return result

    def remove_modifiers_by_source(self, source: ModifierSource) -> int:
        """Remove all modifiers from a source type across all stats."""
        total_removed = 0
        for stat in self._stats.values():
            old_value = stat.value
            removed = stat.remove_modifiers_by_source(source)
            if removed > 0:
                total_removed += removed
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat.name, old_value, new_value)
        return total_removed

    def remove_modifiers_by_source_id(self, source_id: str) -> int:
        """Remove all modifiers with a source ID across all stats."""
        total_removed = 0
        for stat in self._stats.values():
            old_value = stat.value
            if stat.remove_modifier(source_id):
                total_removed += 1
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat.name, old_value, new_value)
        return total_removed

    def remove_modifiers_by_tag(self, tag: str) -> int:
        """Remove all modifiers with a tag across all stats."""
        total_removed = 0
        for stat in self._stats.values():
            old_value = stat.value
            removed = stat.remove_modifiers_by_tag(tag)
            if removed > 0:
                total_removed += removed
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat.name, old_value, new_value)
        return total_removed

    def clear_modifiers(self, stat_name: Optional[str] = None) -> None:
        """Clear modifiers for a stat, or all stats if name is None."""
        if stat_name:
            stat = self._stats.get(stat_name.lower())
            if stat:
                old_value = stat.value
                stat.clear_modifiers()
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat_name, old_value, new_value)
        else:
            for stat in self._stats.values():
                old_value = stat.value
                stat.clear_modifiers()
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat.name, old_value, new_value)

    def get_modifiers(self, stat_name: str) -> List[StatModifier]:
        """Get all modifiers for a stat."""
        stat = self._stats.get(stat_name.lower())
        return list(stat.modifiers) if stat else []

    # =========================================================================
    # TIMED MODIFIERS
    # =========================================================================

    def update_timed_modifiers(self, delta_time: float) -> List[tuple[str, StatModifier]]:
        """
        Update timed modifiers, removing expired ones.

        Args:
            delta_time: Time elapsed since last update

        Returns:
            List of (stat_name, modifier) for removed modifiers
        """
        removed = []

        for stat_name, stat in self._stats.items():
            old_value = stat.value
            expired = []

            for modifier in stat.modifiers:
                if modifier.duration > 0:
                    modifier.duration -= delta_time
                    if modifier.duration <= 0:
                        expired.append(modifier)

            for modifier in expired:
                stat.modifiers.remove(modifier)
                removed.append((stat_name, modifier))
                stat._cache_dirty = True

            if expired:
                new_value = stat.value
                if old_value != new_value:
                    self._notify_change(stat_name, old_value, new_value)

        return removed

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_stat_changed(self, callback: Callable[[str, float, float], None]) -> None:
        """Register callback for stat changes (stat_name, old_value, new_value)."""
        self._on_stat_changed.append(callback)

    def off_stat_changed(self, callback: Callable[[str, float, float], None]) -> None:
        """Unregister stat change callback."""
        if callback in self._on_stat_changed:
            self._on_stat_changed.remove(callback)

    def _notify_change(self, stat_name: str, old_value: float, new_value: float) -> None:
        """Notify listeners of a stat change."""
        for callback in self._on_stat_changed:
            callback(stat_name, old_value, new_value)

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    def copy_from(self, other: StatsComponent) -> None:
        """Copy all stats from another component."""
        self._stats.clear()
        for name, stat in other._stats.items():
            new_stat = Stat(
                name=stat.name,
                base_value=stat.base_value,
                min_value=stat.min_value,
                max_value=stat.max_value,
                modifiers=[
                    StatModifier(
                        value=m.value,
                        modifier_type=m.modifier_type,
                        source=m.source,
                        source_id=m.source_id,
                        priority=m.priority,
                        duration=m.duration,
                        stacks=m.stacks,
                        max_stacks=m.max_stacks,
                        tag=m.tag,
                    )
                    for m in stat.modifiers
                ],
            )
            self._stats[name] = new_stat

    def get_snapshot(self) -> Dict[str, float]:
        """Get a snapshot of all current stat values."""
        snapshot = {name: stat.value for name, stat in self._stats.items()}
        for name, compute_func in self._derived_stats.items():
            snapshot[name] = compute_func()
        return snapshot

    # =========================================================================
    # ITERATION
    # =========================================================================

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Iterate over stat name-value pairs."""
        for name, stat in self._stats.items():
            yield name, stat.value
        for name, compute_func in self._derived_stats.items():
            yield name, compute_func()

    def __len__(self) -> int:
        """Get number of stats."""
        return len(self._stats) + len(self._derived_stats)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stats component to dictionary."""
        stats_data = {}
        for name, stat in self._stats.items():
            stats_data[name] = {
                "base_value": stat.base_value,
                "min_value": stat.min_value,
                "max_value": stat.max_value,
                "modifiers": [
                    {
                        "value": m.value,
                        "modifier_type": m.modifier_type.name,
                        "source": m.source.name,
                        "source_id": m.source_id,
                        "priority": m.priority,
                        "duration": m.duration,
                        "stacks": m.stacks,
                        "max_stacks": m.max_stacks,
                        "tag": m.tag,
                    }
                    for m in stat.modifiers
                ],
            }

        return {
            "stats": stats_data,
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StatsComponent:
        """Deserialize stats component from dictionary."""
        component = cls(entity_id=data.get("entity_id"))

        for name, stat_data in data.get("stats", {}).items():
            stat = component.register_stat(
                name,
                base_value=stat_data["base_value"],
                min_value=stat_data.get("min_value", float("-inf")),
                max_value=stat_data.get("max_value", float("inf")),
            )

            for mod_data in stat_data.get("modifiers", []):
                modifier = StatModifier(
                    value=mod_data["value"],
                    modifier_type=ModifierType[mod_data["modifier_type"]],
                    source=ModifierSource[mod_data.get("source", "OTHER")],
                    source_id=mod_data.get("source_id", ""),
                    priority=mod_data.get("priority", 0),
                    duration=mod_data.get("duration", -1.0),
                    stacks=mod_data.get("stacks", 1),
                    max_stacks=mod_data.get("max_stacks", 1),
                    tag=mod_data.get("tag", ""),
                )
                stat.add_modifier(modifier)

        return component

    def __repr__(self) -> str:
        stats_preview = ", ".join(
            f"{name}={stat.value:.1f}"
            for name, stat in list(self._stats.items())[:3]
        )
        if len(self._stats) > 3:
            stats_preview += f"... (+{len(self._stats) - 3} more)"
        return f"StatsComponent({stats_preview})"


__all__ = [
    "StatsComponent",
    "Stat",
    "StatModifier",
    "ModifierType",
    "ModifierSource",
]
