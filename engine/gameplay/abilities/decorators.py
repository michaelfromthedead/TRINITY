"""
Ability and Buff Decorators wired to Foundation Registry.

Provides @ability and @buff decorators for registering gameplay abilities
and buffs with the Foundation Registry system, enabling runtime discovery
via registry queries.

Events:
    - AbilityCast: Fired when an ability is cast
    - BuffApplied: Fired when a buff is applied
    - BuffExpired: Fired when a buff expires
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from foundation import Registry, registry, Event, EventLog, get_event_log, get_current_tick


# =============================================================================
# STACKING MODES
# =============================================================================


class StackingMode(str, Enum):
    """
    How buffs stack when reapplied.

    Modes:
        NONE: No stacking, reapplication refreshes duration only
        DURATION: Stack duration additively
        INTENSITY: Stack effect intensity/magnitude
        INDEPENDENT: Each application is tracked separately
    """

    NONE = "none"
    DURATION = "duration"
    INTENSITY = "intensity"
    INDEPENDENT = "independent"


# =============================================================================
# EVENTS
# =============================================================================


@dataclass
class AbilityCast:
    """
    Event fired when an ability is cast.

    Attributes:
        entity_id: The entity casting the ability
        ability_name: Name of the ability being cast
        target_id: Optional target entity ID
        timestamp: Unix timestamp of the cast
    """

    entity_id: int
    ability_name: str
    target_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BuffApplied:
    """
    Event fired when a buff is applied.

    Attributes:
        entity_id: The entity receiving the buff
        buff_name: Name of the buff being applied
        stacks: Current number of stacks
        duration: Remaining duration in seconds
        timestamp: Unix timestamp of the application
    """

    entity_id: int
    buff_name: str
    stacks: int = 1
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BuffExpired:
    """
    Event fired when a buff expires.

    Attributes:
        entity_id: The entity losing the buff
        buff_name: Name of the expired buff
        timestamp: Unix timestamp of the expiration
    """

    entity_id: int
    buff_name: str
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# ABILITY DECORATOR
# =============================================================================


F = TypeVar("F", bound=Callable[..., Any])


def ability(
    name: str,
    cooldown: float = 0.0,
    cost: Optional[Dict[str, float]] = None,
    tags: Optional[List[str]] = None,
    required_tags: Optional[List[str]] = None,
    granted_tags: Optional[List[str]] = None,
    blocked_by_tags: Optional[List[str]] = None,
    description: str = "",
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register an ability class with the Foundation Registry.

    Registers the class with the "ability" tag and stores metadata for
    runtime discovery via Registry.query().

    Args:
        name: Unique ability name for identification
        cooldown: Cooldown duration in seconds
        cost: Resource costs (e.g., {"mana": 25, "stamina": 10})
        tags: Additional tags for categorization (e.g., ["fire", "aoe"])
        required_tags: Tags required on caster to use ability
        granted_tags: Tags granted to caster while ability is active
        blocked_by_tags: Tags on caster that prevent ability use
        description: Human-readable description
        track_instances: Whether to track live instances

    Returns:
        Class decorator that registers the ability

    Example:
        @ability(name="fireball", cooldown=2.0, cost={"mana": 25}, tags=["fire", "projectile"])
        class Fireball:
            def cast(self, caster, target):
                ...
    """
    resolved_cost = cost if cost is not None else {}
    resolved_tags = tags if tags is not None else []
    resolved_required = required_tags if required_tags is not None else []
    resolved_granted = granted_tags if granted_tags is not None else []
    resolved_blocked = blocked_by_tags if blocked_by_tags is not None else []

    def decorator(cls: type) -> type:
        # Register with Foundation Registry
        registry.register(cls, name=name, track_instances=track_instances)

        # Add the primary "ability" tag
        registry.add_tag(cls, "ability")

        # Add additional categorization tags
        for tag in resolved_tags:
            registry.add_tag(cls, tag)

        # Store metadata for querying
        registry.set_metadata(cls, "name", name)
        registry.set_metadata(cls, "cooldown", cooldown)
        registry.set_metadata(cls, "cost", resolved_cost)
        registry.set_metadata(cls, "description", description)
        registry.set_metadata(cls, "required_tags", frozenset(resolved_required))
        registry.set_metadata(cls, "granted_tags", frozenset(resolved_granted))
        registry.set_metadata(cls, "blocked_by_tags", frozenset(resolved_blocked))
        registry.set_metadata(cls, "category_tags", frozenset(resolved_tags))

        # Store metadata on the class for introspection
        cls._ability_name = name
        cls._ability_cooldown = cooldown
        cls._ability_cost = resolved_cost
        cls._ability_description = description
        cls._ability_required_tags = frozenset(resolved_required)
        cls._ability_granted_tags = frozenset(resolved_granted)
        cls._ability_blocked_by_tags = frozenset(resolved_blocked)
        cls._ability_category_tags = frozenset(resolved_tags)

        return cls

    return decorator


def emit_ability_cast(
    entity_id: int,
    ability_name: str,
    target_id: Optional[int] = None,
) -> AbilityCast:
    """
    Emit an AbilityCast event and record it in the EventLog.

    Args:
        entity_id: The entity casting the ability
        ability_name: Name of the ability being cast
        target_id: Optional target entity ID

    Returns:
        The emitted AbilityCast event
    """
    event = AbilityCast(
        entity_id=entity_id,
        ability_name=ability_name,
        target_id=target_id,
        timestamp=time.time(),
    )

    # Record to EventLog
    log = get_event_log()
    log.record(
        Event(
            tick=get_current_tick(),
            operation=f"AbilityCast.{ability_name}",
            operation_args={
                "entity_id": entity_id,
                "ability_name": ability_name,
                "target_id": target_id,
            },
            entity=entity_id,
        )
    )

    return event


# =============================================================================
# BUFF DECORATOR
# =============================================================================


def buff(
    name: str,
    duration: float = 0.0,
    stacking: Union[str, StackingMode] = StackingMode.NONE,
    max_stacks: int = 1,
    tags: Optional[List[str]] = None,
    is_debuff: bool = False,
    tick_rate: float = 0.0,
    description: str = "",
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a buff class with the Foundation Registry.

    Registers the class with the "buff" tag (or "debuff" if is_debuff=True)
    and stores metadata for runtime discovery via Registry.query().

    Args:
        name: Unique buff name for identification
        duration: Duration in seconds (0 = infinite/permanent)
        stacking: How the buff stacks (none, duration, intensity, independent)
        max_stacks: Maximum number of stacks (for intensity/independent modes)
        tags: Additional tags for categorization
        is_debuff: Whether this is a harmful debuff
        tick_rate: Interval for periodic tick effects (0 = no ticking)
        description: Human-readable description
        track_instances: Whether to track live instances

    Returns:
        Class decorator that registers the buff

    Example:
        @buff(name="burning", duration=5.0, stacking="intensity", max_stacks=3, is_debuff=True)
        class Burning:
            def on_tick(self, entity):
                entity.take_damage(10 * self.stacks)
    """
    # Normalize stacking mode
    if isinstance(stacking, str):
        stacking_mode = StackingMode(stacking)
    else:
        stacking_mode = stacking

    resolved_tags = tags if tags is not None else []

    def decorator(cls: type) -> type:
        # Register with Foundation Registry
        registry.register(cls, name=name, track_instances=track_instances)

        # Add the primary "buff" or "debuff" tag
        if is_debuff:
            registry.add_tag(cls, "debuff")
        registry.add_tag(cls, "buff")

        # Add stacking mode as a tag for filtering
        registry.add_tag(cls, f"stacking_{stacking_mode.value}")

        # Add additional categorization tags
        for tag in resolved_tags:
            registry.add_tag(cls, tag)

        # Store metadata for querying
        registry.set_metadata(cls, "name", name)
        registry.set_metadata(cls, "duration", duration)
        registry.set_metadata(cls, "stacking", stacking_mode.value)
        registry.set_metadata(cls, "stacking_mode", stacking_mode)
        registry.set_metadata(cls, "max_stacks", max_stacks)
        registry.set_metadata(cls, "is_debuff", is_debuff)
        registry.set_metadata(cls, "tick_rate", tick_rate)
        registry.set_metadata(cls, "description", description)
        registry.set_metadata(cls, "category_tags", frozenset(resolved_tags))

        # Store metadata on the class for introspection
        cls._buff_name = name
        cls._buff_duration = duration
        cls._buff_stacking = stacking_mode
        cls._buff_max_stacks = max_stacks
        cls._buff_is_debuff = is_debuff
        cls._buff_tick_rate = tick_rate
        cls._buff_description = description
        cls._buff_category_tags = frozenset(resolved_tags)

        return cls

    return decorator


def emit_buff_applied(
    entity_id: int,
    buff_name: str,
    stacks: int = 1,
    duration: float = 0.0,
) -> BuffApplied:
    """
    Emit a BuffApplied event and record it in the EventLog.

    Args:
        entity_id: The entity receiving the buff
        buff_name: Name of the buff being applied
        stacks: Current number of stacks
        duration: Remaining duration in seconds

    Returns:
        The emitted BuffApplied event
    """
    event = BuffApplied(
        entity_id=entity_id,
        buff_name=buff_name,
        stacks=stacks,
        duration=duration,
        timestamp=time.time(),
    )

    # Record to EventLog
    log = get_event_log()
    log.record(
        Event(
            tick=get_current_tick(),
            operation=f"BuffApplied.{buff_name}",
            operation_args={
                "entity_id": entity_id,
                "buff_name": buff_name,
                "stacks": stacks,
                "duration": duration,
            },
            entity=entity_id,
        )
    )

    return event


def emit_buff_expired(
    entity_id: int,
    buff_name: str,
) -> BuffExpired:
    """
    Emit a BuffExpired event and record it in the EventLog.

    Args:
        entity_id: The entity losing the buff
        buff_name: Name of the expired buff

    Returns:
        The emitted BuffExpired event
    """
    event = BuffExpired(
        entity_id=entity_id,
        buff_name=buff_name,
        timestamp=time.time(),
    )

    # Record to EventLog
    log = get_event_log()
    log.record(
        Event(
            tick=get_current_tick(),
            operation=f"BuffExpired.{buff_name}",
            operation_args={
                "entity_id": entity_id,
                "buff_name": buff_name,
            },
            entity=entity_id,
        )
    )

    return event


# =============================================================================
# QUERY HELPERS
# =============================================================================


def get_all_abilities() -> List[type]:
    """
    Get all registered ability classes.

    Returns:
        List of all classes registered with the "ability" tag
    """
    return registry.query(tag="ability")


def get_abilities_by_tag(*tags: str) -> List[type]:
    """
    Get ability classes that have all the specified tags.

    Args:
        *tags: Tags to filter by

    Returns:
        List of ability classes that have all specified tags
    """
    abilities = registry.query(tag="ability")
    result = []
    for cls in abilities:
        cls_tags = registry.get_tags(cls)
        if all(t in cls_tags for t in tags):
            result.append(cls)
    return result


def get_all_buffs() -> List[type]:
    """
    Get all registered buff classes.

    Returns:
        List of all classes registered with the "buff" tag
    """
    return registry.query(tag="buff")


def get_buffs_by_stacking(stacking: Union[str, StackingMode]) -> List[type]:
    """
    Get buff classes with the specified stacking mode.

    Args:
        stacking: The stacking mode to filter by

    Returns:
        List of buff classes with the specified stacking mode
    """
    if isinstance(stacking, StackingMode):
        stacking_str = stacking.value
    else:
        stacking_str = stacking

    return registry.query(tag="buff", stacking=stacking_str)


def get_debuffs() -> List[type]:
    """
    Get all registered debuff classes.

    Returns:
        List of all classes registered with both "buff" and "debuff" tags
    """
    return registry.query(tag="debuff")


def get_ability_metadata(cls: type) -> Dict[str, Any]:
    """
    Get all metadata for a registered ability class.

    Args:
        cls: The ability class to get metadata for

    Returns:
        Dictionary of all metadata for the ability

    Raises:
        ValueError: If the class is not a registered ability
    """
    if not registry.has_tag(cls, "ability"):
        raise ValueError(f"{cls.__name__} is not a registered ability")
    return registry.get_all_metadata(cls)


def get_buff_metadata(cls: type) -> Dict[str, Any]:
    """
    Get all metadata for a registered buff class.

    Args:
        cls: The buff class to get metadata for

    Returns:
        Dictionary of all metadata for the buff

    Raises:
        ValueError: If the class is not a registered buff
    """
    if not registry.has_tag(cls, "buff"):
        raise ValueError(f"{cls.__name__} is not a registered buff")
    return registry.get_all_metadata(cls)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Stacking mode
    "StackingMode",
    # Events
    "AbilityCast",
    "BuffApplied",
    "BuffExpired",
    # Decorators
    "ability",
    "buff",
    # Event emitters
    "emit_ability_cast",
    "emit_buff_applied",
    "emit_buff_expired",
    # Query helpers
    "get_all_abilities",
    "get_abilities_by_tag",
    "get_all_buffs",
    "get_buffs_by_stacking",
    "get_debuffs",
    "get_ability_metadata",
    "get_buff_metadata",
]
