"""
Gameplay Effects System.

Provides effect types (Instant, Duration, Infinite, Periodic) and modifiers
(Add, Multiply, Override, Stacking) for applying changes to game entities.

Effects follow proper order of operations and support various modifier types
for attribute manipulation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeModifierHandle,
    AttributeSet,
)
from engine.gameplay.abilities.constants import (
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_MAGNITUDE,
    DEFAULT_MIN_DURATION,
    DEFAULT_MIN_MAGNITUDE,
    DEFAULT_TICK_RATE,
    EPSILON,
    EffectType,
    ModifierOperation,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer

T = TypeVar("T")


# =============================================================================
# EFFECT CONTEXT
# =============================================================================


@dataclass
class EffectContext:
    """Context information for effect application."""

    source: Optional[Any] = None  # Entity that created the effect
    target: Optional[Any] = None  # Entity receiving the effect
    instigator: Optional[Any] = None  # Entity that triggered the effect
    ability: Optional[Any] = None  # Ability that created the effect
    level: int = 1  # Effect level for scaling
    magnitude_multiplier: float = 1.0  # Global magnitude modifier
    duration_multiplier: float = 1.0  # Global duration modifier
    tags: GameplayTagContainer = field(default_factory=GameplayTagContainer)


# =============================================================================
# EFFECT MODIFIER
# =============================================================================


@dataclass(slots=True)
class EffectModifier:
    """
    Defines how an effect modifies an attribute.

    Supports different operations (Add, Multiply, Override, Stacking) and
    magnitude scaling based on effect level.
    """

    attribute: str
    operation: ModifierOperation
    base_magnitude: float
    level_scaling: float = 0.0  # Additional magnitude per level
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE
    max_magnitude: float = DEFAULT_MAX_MAGNITUDE

    def get_magnitude(self, level: int = 1, multiplier: float = 1.0) -> float:
        """Calculate the final magnitude based on level and multiplier."""
        magnitude = self.base_magnitude + (self.level_scaling * (level - 1))
        magnitude *= multiplier
        return max(self.min_magnitude, min(self.max_magnitude, magnitude))


# =============================================================================
# BASE EFFECT
# =============================================================================


@dataclass
class GameplayEffect(ABC):
    """
    Base class for all gameplay effects.

    Effects modify attributes on entities and can be instant, duration-based,
    infinite, or periodic.
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    effect_type: EffectType = EffectType.INSTANT
    modifiers: List[EffectModifier] = field(default_factory=list)
    granted_tags: List[GameplayTag | str] = field(default_factory=list)
    removed_tags: List[GameplayTag | str] = field(default_factory=list)
    application_tags: List[GameplayTag | str] = field(default_factory=list)
    blocked_by_tags: List[GameplayTag | str] = field(default_factory=list)
    _active_handles: List[AttributeModifierHandle] = field(
        default_factory=list, repr=False
    )
    _is_active: bool = field(default=False, repr=False)

    @property
    def is_active(self) -> bool:
        """Check if effect is currently active."""
        return self._is_active

    def can_apply(
        self, target_tags: GameplayTagContainer, context: Optional[EffectContext] = None
    ) -> bool:
        """Check if effect can be applied based on tags."""
        # Check if blocked by target tags
        for tag in self.blocked_by_tags:
            tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
            if target_tags.has(tag_obj):
                return False

        # Check application requirements
        for tag in self.application_tags:
            tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
            if not target_tags.has(tag_obj):
                return False

        return True

    @abstractmethod
    def apply(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply the effect. Returns True if successfully applied."""
        pass

    @abstractmethod
    def remove(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> bool:
        """Remove the effect. Returns True if successfully removed."""
        pass

    @abstractmethod
    def tick(self, delta_time: float, attributes: AttributeSet) -> bool:
        """
        Update the effect. Returns True if still active.

        For instant effects, always returns False.
        For duration effects, returns False when expired.
        """
        pass

    def _apply_modifiers(
        self,
        attributes: AttributeSet,
        context: Optional[EffectContext] = None,
    ) -> List[AttributeModifierHandle]:
        """Apply all modifiers to attributes."""
        handles = []
        level = context.level if context else 1
        multiplier = context.magnitude_multiplier if context else 1.0
        source = context.source if context else self

        for mod in self.modifiers:
            if attributes.has(mod.attribute):
                magnitude = mod.get_magnitude(level, multiplier)
                handle = attributes.add_modifier(
                    mod.attribute,
                    mod.operation,
                    magnitude,
                    source=source,
                )
                handles.append(handle)

        return handles

    def _remove_modifiers(self, attributes: AttributeSet) -> None:
        """Remove all active modifier handles."""
        for handle in self._active_handles:
            attributes.remove_modifier(handle)
        self._active_handles.clear()

    def _apply_tags(self, tags: GameplayTagContainer) -> None:
        """Apply granted tags and remove removed tags."""
        for tag in self.removed_tags:
            tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
            tags.remove(tag_obj)

        for tag in self.granted_tags:
            tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
            tags.add(tag_obj)

    def _remove_tags(self, tags: GameplayTagContainer) -> None:
        """Remove granted tags (don't restore removed tags)."""
        for tag in self.granted_tags:
            tag_obj = tag if isinstance(tag, GameplayTag) else GameplayTag(tag)
            tags.remove(tag_obj)


# =============================================================================
# INSTANT EFFECT
# =============================================================================


@dataclass
class InstantEffect(GameplayEffect):
    """
    An effect that applies instantly and has no duration.

    Modifiers from instant effects are applied once and remain indefinitely
    unless explicitly removed.
    """

    effect_type: EffectType = field(default=EffectType.INSTANT, init=False)

    def apply(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply instant effect."""
        if tags and not self.can_apply(tags, context):
            return False

        self._active_handles = self._apply_modifiers(attributes, context)

        if tags:
            self._apply_tags(tags)

        self._is_active = True
        return True

    def remove(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> bool:
        """Remove instant effect."""
        if not self._is_active:
            return False

        self._remove_modifiers(attributes)

        if tags:
            self._remove_tags(tags)

        self._is_active = False
        return True

    def tick(self, delta_time: float, attributes: AttributeSet) -> bool:
        """Instant effects don't tick, always returns True (stays active)."""
        return self._is_active


# =============================================================================
# DURATION EFFECT
# =============================================================================


@dataclass
class DurationEffect(GameplayEffect):
    """
    An effect with a limited duration.

    The effect is automatically removed when the duration expires.
    """

    effect_type: EffectType = field(default=EffectType.DURATION, init=False)
    duration: float = 0.0
    _remaining_time: float = field(default=0.0, repr=False)
    _start_time: float = field(default=0.0, repr=False)

    @property
    def remaining_time(self) -> float:
        """Get remaining duration."""
        return self._remaining_time

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since application."""
        if self._is_active:
            return self.duration - self._remaining_time
        return 0.0

    @property
    def progress(self) -> float:
        """Get progress as 0.0 to 1.0."""
        if self.duration <= 0:
            return 1.0
        return 1.0 - (self._remaining_time / self.duration)

    def apply(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply duration effect."""
        if tags and not self.can_apply(tags, context):
            return False

        duration_mult = context.duration_multiplier if context else 1.0
        self._remaining_time = self.duration * duration_mult
        self._start_time = time.monotonic()

        self._active_handles = self._apply_modifiers(attributes, context)

        if tags:
            self._apply_tags(tags)

        self._is_active = True
        return True

    def remove(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> bool:
        """Remove duration effect."""
        if not self._is_active:
            return False

        self._remove_modifiers(attributes)

        if tags:
            self._remove_tags(tags)

        self._is_active = False
        self._remaining_time = 0.0
        return True

    def tick(self, delta_time: float, attributes: AttributeSet) -> bool:
        """Update duration. Returns False when expired."""
        if not self._is_active:
            return False

        self._remaining_time -= delta_time

        if self._remaining_time <= 0:
            self._remaining_time = 0
            return False

        return True

    def extend_duration(self, additional_time: float) -> None:
        """Extend the remaining duration."""
        self._remaining_time += additional_time

    def refresh_duration(self) -> None:
        """Reset duration to full."""
        self._remaining_time = self.duration


# =============================================================================
# INFINITE EFFECT
# =============================================================================


@dataclass
class InfiniteEffect(GameplayEffect):
    """
    An effect that lasts until explicitly removed.

    Similar to duration effects but has no automatic expiration.
    """

    effect_type: EffectType = field(default=EffectType.INFINITE, init=False)

    def apply(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply infinite effect."""
        if tags and not self.can_apply(tags, context):
            return False

        self._active_handles = self._apply_modifiers(attributes, context)

        if tags:
            self._apply_tags(tags)

        self._is_active = True
        return True

    def remove(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> bool:
        """Remove infinite effect."""
        if not self._is_active:
            return False

        self._remove_modifiers(attributes)

        if tags:
            self._remove_tags(tags)

        self._is_active = False
        return True

    def tick(self, delta_time: float, attributes: AttributeSet) -> bool:
        """Infinite effects always remain active."""
        return self._is_active


# =============================================================================
# PERIODIC EFFECT
# =============================================================================


@dataclass
class PeriodicEffect(GameplayEffect):
    """
    An effect that applies periodically over a duration.

    The effect triggers at regular intervals, applying instant effects
    each tick (e.g., damage over time, healing over time).
    """

    effect_type: EffectType = field(default=EffectType.PERIODIC, init=False)
    duration: float = 0.0
    tick_rate: float = DEFAULT_TICK_RATE  # Seconds between ticks
    execute_on_apply: bool = True  # Execute first tick immediately
    execute_on_remove: bool = False  # Execute tick on removal
    _remaining_time: float = field(default=0.0, repr=False)
    _time_since_tick: float = field(default=0.0, repr=False)
    _tick_count: int = field(default=0, repr=False)
    _on_tick: Optional[Callable[[PeriodicEffect, AttributeSet], None]] = field(
        default=None, repr=False
    )

    @property
    def remaining_time(self) -> float:
        """Get remaining duration."""
        return self._remaining_time

    @property
    def tick_count(self) -> int:
        """Get number of ticks that have occurred."""
        return self._tick_count

    @property
    def time_until_next_tick(self) -> float:
        """Get time until next tick."""
        return max(0, self.tick_rate - self._time_since_tick)

    def apply(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply periodic effect."""
        if tags and not self.can_apply(tags, context):
            return False

        duration_mult = context.duration_multiplier if context else 1.0
        self._remaining_time = self.duration * duration_mult
        self._time_since_tick = 0.0
        self._tick_count = 0

        # Store context for tick applications
        self._context = context

        if tags:
            self._apply_tags(tags)

        self._is_active = True

        # Execute first tick if configured
        if self.execute_on_apply:
            self._execute_tick(attributes)

        return True

    def remove(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> bool:
        """Remove periodic effect."""
        if not self._is_active:
            return False

        # Execute final tick if configured
        if self.execute_on_remove:
            self._execute_tick(attributes)

        if tags:
            self._remove_tags(tags)

        self._is_active = False
        self._remaining_time = 0.0
        return True

    def tick(self, delta_time: float, attributes: AttributeSet) -> bool:
        """Update periodic effect. Returns False when expired."""
        if not self._is_active:
            return False

        self._remaining_time -= delta_time
        self._time_since_tick += delta_time

        # Check for tick
        while self._time_since_tick >= self.tick_rate:
            self._time_since_tick -= self.tick_rate
            self._execute_tick(attributes)

        # Check expiration (0 duration = infinite)
        if self.duration > 0 and self._remaining_time <= 0:
            self._remaining_time = 0
            return False

        return True

    def _execute_tick(self, attributes: AttributeSet) -> None:
        """Execute a single tick of the effect."""
        self._tick_count += 1

        # Apply modifiers as instant changes
        context = getattr(self, "_context", None)
        level = context.level if context else 1
        multiplier = context.magnitude_multiplier if context else 1.0

        for mod in self.modifiers:
            if attributes.has(mod.attribute):
                magnitude = mod.get_magnitude(level, multiplier)
                # For periodic effects, we directly modify the base value
                # rather than adding persistent modifiers
                if mod.operation == ModifierOperation.ADD:
                    current = attributes.get(mod.attribute)
                    attr = attributes.get_attribute(mod.attribute)
                    attr.set_base_value(current + magnitude)
                elif mod.operation == ModifierOperation.MULTIPLY:
                    current = attributes.get(mod.attribute)
                    attr = attributes.get_attribute(mod.attribute)
                    attr.set_base_value(current * (1 + magnitude))

        # Call tick callback
        if self._on_tick is not None:
            self._on_tick(self, attributes)


# =============================================================================
# EFFECT CONTAINER
# =============================================================================


class EffectContainer:
    """
    Container for managing active effects on an entity.

    Handles effect lifecycle, stacking, and tick updates.
    """

    def __init__(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> None:
        self._attributes = attributes
        self._tags = tags or GameplayTagContainer()
        self._effects: Dict[UUID, GameplayEffect] = {}
        self._by_name: Dict[str, Set[UUID]] = {}

    @property
    def active_effects(self) -> List[GameplayEffect]:
        """Get all active effects."""
        return list(self._effects.values())

    def apply(
        self,
        effect: GameplayEffect,
        context: Optional[EffectContext] = None,
    ) -> bool:
        """Apply an effect to the container."""
        if effect.apply(self._attributes, self._tags, context):
            self._effects[effect.id] = effect
            if effect.name:
                if effect.name not in self._by_name:
                    self._by_name[effect.name] = set()
                self._by_name[effect.name].add(effect.id)
            return True
        return False

    def remove(self, effect_or_id: GameplayEffect | UUID) -> bool:
        """Remove an effect from the container."""
        if isinstance(effect_or_id, GameplayEffect):
            effect_id = effect_or_id.id
        else:
            effect_id = effect_or_id

        if effect_id not in self._effects:
            return False

        effect = self._effects[effect_id]
        if effect.remove(self._attributes, self._tags):
            del self._effects[effect_id]
            if effect.name and effect.name in self._by_name:
                self._by_name[effect.name].discard(effect_id)
                if not self._by_name[effect.name]:
                    del self._by_name[effect.name]
            return True
        return False

    def remove_by_name(self, name: str) -> int:
        """Remove all effects with the given name. Returns count removed."""
        if name not in self._by_name:
            return 0

        ids_to_remove = list(self._by_name[name])
        count = 0
        for effect_id in ids_to_remove:
            if self.remove(effect_id):
                count += 1
        return count

    def remove_all(self) -> int:
        """Remove all effects. Returns count removed."""
        count = 0
        for effect_id in list(self._effects.keys()):
            if self.remove(effect_id):
                count += 1
        return count

    def has_effect(self, name: str) -> bool:
        """Check if an effect with the given name is active."""
        return name in self._by_name and len(self._by_name[name]) > 0

    def get_effects_by_name(self, name: str) -> List[GameplayEffect]:
        """Get all effects with the given name."""
        if name not in self._by_name:
            return []
        return [self._effects[eid] for eid in self._by_name[name]]

    def tick(self, delta_time: float) -> None:
        """Update all effects. Removes expired effects."""
        expired = []
        for effect_id, effect in self._effects.items():
            if not effect.tick(delta_time, self._attributes):
                expired.append(effect_id)

        for effect_id in expired:
            self.remove(effect_id)


# =============================================================================
# EFFECT FACTORY FUNCTIONS
# =============================================================================


def instant_damage(
    amount: float,
    attribute: str = "health",
    source: Optional[Any] = None,
) -> InstantEffect:
    """Create an instant damage effect."""
    return InstantEffect(
        name="instant_damage",
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=ModifierOperation.ADD,
                base_magnitude=-amount,
            )
        ],
    )


def instant_heal(
    amount: float,
    attribute: str = "health",
    source: Optional[Any] = None,
) -> InstantEffect:
    """Create an instant heal effect."""
    return InstantEffect(
        name="instant_heal",
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=ModifierOperation.ADD,
                base_magnitude=amount,
            )
        ],
    )


def damage_over_time(
    damage_per_tick: float,
    duration: float,
    tick_rate: float = DEFAULT_TICK_RATE,
    attribute: str = "health",
) -> PeriodicEffect:
    """Create a damage over time effect."""
    return PeriodicEffect(
        name="damage_over_time",
        duration=duration,
        tick_rate=tick_rate,
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=ModifierOperation.ADD,
                base_magnitude=-damage_per_tick,
            )
        ],
    )


def heal_over_time(
    heal_per_tick: float,
    duration: float,
    tick_rate: float = DEFAULT_TICK_RATE,
    attribute: str = "health",
) -> PeriodicEffect:
    """Create a heal over time effect."""
    return PeriodicEffect(
        name="heal_over_time",
        duration=duration,
        tick_rate=tick_rate,
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=ModifierOperation.ADD,
                base_magnitude=heal_per_tick,
            )
        ],
    )


def stat_buff(
    attribute: str,
    amount: float,
    duration: float,
    operation: ModifierOperation = ModifierOperation.ADD,
) -> DurationEffect:
    """Create a stat buff effect."""
    return DurationEffect(
        name=f"{attribute}_buff",
        duration=duration,
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=operation,
                base_magnitude=amount,
            )
        ],
    )


def stat_debuff(
    attribute: str,
    amount: float,
    duration: float,
    operation: ModifierOperation = ModifierOperation.ADD,
) -> DurationEffect:
    """
    Create a stat debuff effect.

    For ADD operations, the amount is negated to reduce the attribute.
    For MULTIPLY operations, the amount should be positive and represents
    the percentage reduction (e.g., 0.25 = 25% reduction, applied as -0.25).
    """
    # Debuffs always use negative magnitude to reduce the attribute value
    magnitude = -abs(amount)
    return DurationEffect(
        name=f"{attribute}_debuff",
        duration=duration,
        modifiers=[
            EffectModifier(
                attribute=attribute,
                operation=operation,
                base_magnitude=magnitude,
            )
        ],
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base types
    "EffectContext",
    "EffectModifier",
    "GameplayEffect",
    # Effect types
    "InstantEffect",
    "DurationEffect",
    "InfiniteEffect",
    "PeriodicEffect",
    # Container
    "EffectContainer",
    # Factory functions
    "instant_damage",
    "instant_heal",
    "damage_over_time",
    "heal_over_time",
    "stat_buff",
    "stat_debuff",
]
