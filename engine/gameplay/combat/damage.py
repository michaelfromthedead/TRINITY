"""
Combat System - Damage Module

Provides the DamageSystem for calculating and applying damage with support for:
- Multiple damage types (physical, elemental, etc.)
- Armor and resistance calculations
- Damage multipliers (critical hits, hitbox zones)
- Damage events and tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING
from enum import Enum
import time

from .constants import (
    DamageType,
    DamageSource,
    DamageConfig,
    HitboxZone,
    CombatEventType,
    DEFAULT_DAMAGE_CONFIG,
    DEFAULT_RESISTANCES,
    ARMOR_CONSTANT,
    MAX_ARMOR_REDUCTION,
    MAX_RESISTANCE,
    MIN_RESISTANCE,
    MINIMUM_DAMAGE,
    MAXIMUM_DAMAGE,
    HITBOX_DAMAGE_MULTIPLIERS,
    CRITICAL_HIT_ZONES,
    PHYSICAL_DAMAGE_TYPES,
    MAX_DAMAGE_HISTORY_SIZE,
)


# =============================================================================
# PROTOCOLS
# =============================================================================


class DamageReceiver(Protocol):
    """Protocol for entities that can receive damage."""

    def get_armor(self) -> float:
        """Get the entity's armor value."""
        ...

    def get_resistance(self, damage_type: DamageType) -> float:
        """Get resistance to a specific damage type."""
        ...

    def apply_damage(self, amount: float, damage_info: "DamageInfo") -> float:
        """Apply damage and return actual damage dealt."""
        ...

    def is_invulnerable(self) -> bool:
        """Check if entity is currently invulnerable."""
        ...


class DamageDealer(Protocol):
    """Protocol for entities that can deal damage."""

    @property
    def entity_id(self) -> int:
        """Get the entity's unique ID."""
        ...


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DamageModifier:
    """A modifier that affects damage calculation."""

    name: str
    multiplier: float = 1.0
    flat_bonus: float = 0.0
    priority: int = 0  # Higher priority applied first
    condition: Optional[Callable[["DamageInfo"], bool]] = None

    def apply(self, damage: float, info: "DamageInfo") -> float:
        """Apply this modifier to the damage value."""
        if self.condition is not None and not self.condition(info):
            return damage
        return (damage + self.flat_bonus) * self.multiplier


@dataclass
class DamageInfo:
    """Complete information about a damage instance."""

    base_damage: float
    damage_type: DamageType
    source: DamageSource = DamageSource.UNKNOWN
    attacker_id: Optional[int] = None
    target_id: Optional[int] = None
    hitbox_zone: HitboxZone = HitboxZone.GENERIC
    is_critical: bool = False
    is_headshot: bool = False
    is_backstab: bool = False
    timestamp: float = field(default_factory=time.time)

    # Calculated values (filled in during processing)
    final_damage: float = 0.0
    damage_mitigated: float = 0.0
    armor_applied: float = 0.0
    resistance_applied: float = 0.0
    multipliers_applied: List[str] = field(default_factory=list)

    # Additional metadata
    weapon_id: Optional[int] = None
    ability_id: Optional[int] = None
    projectile_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and process damage info after initialization."""
        if self.base_damage < 0:
            raise ValueError("base_damage cannot be negative")

        # Auto-detect headshot from hitbox zone
        if self.hitbox_zone == HitboxZone.HEAD:
            self.is_headshot = True

        # Auto-detect backstab
        if self.hitbox_zone == HitboxZone.BACK:
            self.is_backstab = True


@dataclass
class DamageResult:
    """Result of a damage calculation or application."""

    damage_dealt: float
    damage_blocked: float
    damage_resisted: float
    was_lethal: bool
    was_critical: bool
    was_headshot: bool
    overkill_damage: float = 0.0
    damage_info: Optional[DamageInfo] = None

    @property
    def total_mitigated(self) -> float:
        """Total damage mitigated by defenses."""
        return self.damage_blocked + self.damage_resisted


@dataclass
class ResistanceProfile:
    """Complete resistance profile for an entity."""

    resistances: Dict[DamageType, float] = field(default_factory=lambda: dict(DEFAULT_RESISTANCES))
    armor: float = 0.0

    def get_resistance(self, damage_type: DamageType) -> float:
        """Get resistance for a damage type, clamped to valid range."""
        base = self.resistances.get(damage_type, 0.0)
        return max(MIN_RESISTANCE, min(MAX_RESISTANCE, base))

    def set_resistance(self, damage_type: DamageType, value: float) -> None:
        """Set resistance for a damage type."""
        self.resistances[damage_type] = max(MIN_RESISTANCE, min(MAX_RESISTANCE, value))

    def add_resistance(self, damage_type: DamageType, value: float) -> None:
        """Add to existing resistance for a damage type."""
        current = self.resistances.get(damage_type, 0.0)
        self.set_resistance(damage_type, current + value)


# =============================================================================
# DAMAGE SYSTEM
# =============================================================================


class DamageSystem:
    """
    Core damage calculation system.

    Handles damage calculation following the formula:
    final_damage = (base_damage * multipliers - armor_reduction) * (1 - resistance)

    Features:
    - Multiple damage types with type-specific resistances
    - Armor system with diminishing returns
    - Hitbox-based damage multipliers
    - Critical hit detection
    - Damage event callbacks
    - Configurable damage modifiers
    """

    def __init__(self, config: Optional[DamageConfig] = None) -> None:
        """
        Initialize the damage system.

        Args:
            config: Optional damage configuration. Uses defaults if not provided.
        """
        self._config = config or DEFAULT_DAMAGE_CONFIG
        self._global_modifiers: List[DamageModifier] = []
        self._type_modifiers: Dict[DamageType, List[DamageModifier]] = {}
        self._event_handlers: Dict[CombatEventType, List[Callable[[DamageInfo], None]]] = {}
        self._damage_history: List[DamageInfo] = []
        self._max_history_size: int = MAX_DAMAGE_HISTORY_SIZE

    @property
    def config(self) -> DamageConfig:
        """Get the current damage configuration."""
        return self._config

    # =========================================================================
    # DAMAGE CALCULATION
    # =========================================================================

    def calculate_damage(
        self,
        base_damage: float,
        damage_type: DamageType,
        armor: float = 0.0,
        resistance: float = 0.0,
        hitbox_zone: HitboxZone = HitboxZone.GENERIC,
        critical_multiplier: float = 1.0,
        additional_multipliers: Optional[List[float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate final damage after all reductions.

        Args:
            base_damage: Base damage before any modifications
            damage_type: Type of damage being dealt
            armor: Target's armor value
            resistance: Target's resistance to this damage type (0.0-1.0)
            hitbox_zone: Which hitbox zone was hit
            critical_multiplier: Multiplier for critical hits (1.0 = no crit)
            additional_multipliers: List of additional damage multipliers

        Returns:
            Tuple of (final_damage, armor_reduction, resistance_reduction)
        """
        if base_damage <= 0:
            return (0.0, 0.0, 0.0)

        # Apply hitbox multiplier
        hitbox_mult = HITBOX_DAMAGE_MULTIPLIERS.get(hitbox_zone, 1.0)

        # Apply critical multiplier
        damage = base_damage * hitbox_mult * critical_multiplier

        # Apply additional multipliers
        if additional_multipliers:
            for mult in additional_multipliers:
                damage *= mult

        # Calculate armor reduction (only for physical damage types)
        armor_reduction = 0.0
        if damage_type in PHYSICAL_DAMAGE_TYPES and armor > 0:
            armor_reduction = self._calculate_armor_reduction(damage, armor)
            damage -= armor_reduction

        # Calculate resistance reduction
        resistance_reduction = 0.0
        if damage_type != DamageType.TRUE:
            resistance = max(self._config.min_resistance,
                           min(self._config.max_resistance, resistance))
            resistance_reduction = damage * resistance
            damage *= (1.0 - resistance)

        # Clamp to min/max damage
        damage = max(self._config.minimum_damage,
                    min(self._config.maximum_damage, damage))

        return (damage, armor_reduction, resistance_reduction)

    def _calculate_armor_reduction(self, damage: float, armor: float) -> float:
        """
        Calculate damage reduction from armor using diminishing returns.

        Formula: reduction = damage * (armor / (armor + ARMOR_CONSTANT))
        Capped at MAX_ARMOR_REDUCTION.
        """
        if armor <= 0:
            return 0.0

        reduction_ratio = armor / (armor + self._config.armor_constant)
        reduction_ratio = min(reduction_ratio, self._config.max_armor_reduction)

        return damage * reduction_ratio

    def calculate_effective_armor(self, armor: float) -> float:
        """
        Calculate effective damage reduction percentage from armor.

        Args:
            armor: Raw armor value

        Returns:
            Damage reduction as a percentage (0.0-1.0)
        """
        if armor <= 0:
            return 0.0

        reduction = armor / (armor + self._config.armor_constant)
        return min(reduction, self._config.max_armor_reduction)

    def calculate_armor_for_reduction(self, target_reduction: float) -> float:
        """
        Calculate armor needed for a target damage reduction.

        Args:
            target_reduction: Desired damage reduction (0.0-1.0)

        Returns:
            Armor value needed
        """
        if target_reduction <= 0:
            return 0.0
        if target_reduction >= self._config.max_armor_reduction:
            target_reduction = self._config.max_armor_reduction - 0.01

        # Solve: reduction = armor / (armor + constant)
        # armor = constant * reduction / (1 - reduction)
        return self._config.armor_constant * target_reduction / (1.0 - target_reduction)

    # =========================================================================
    # DAMAGE APPLICATION
    # =========================================================================

    def create_damage_info(
        self,
        base_damage: float,
        damage_type: DamageType,
        attacker_id: Optional[int] = None,
        target_id: Optional[int] = None,
        source: DamageSource = DamageSource.UNKNOWN,
        hitbox_zone: HitboxZone = HitboxZone.GENERIC,
        is_critical: bool = False,
        **metadata: Any,
    ) -> DamageInfo:
        """
        Create a DamageInfo object for tracking damage.

        Args:
            base_damage: Base damage amount
            damage_type: Type of damage
            attacker_id: ID of attacking entity
            target_id: ID of target entity
            source: Source category of the damage
            hitbox_zone: Which zone was hit
            is_critical: Whether this is a critical hit
            **metadata: Additional metadata

        Returns:
            Configured DamageInfo object
        """
        return DamageInfo(
            base_damage=base_damage,
            damage_type=damage_type,
            attacker_id=attacker_id,
            target_id=target_id,
            source=source,
            hitbox_zone=hitbox_zone,
            is_critical=is_critical,
            metadata=metadata,
        )

    def process_damage(
        self,
        info: DamageInfo,
        armor: float = 0.0,
        resistance: float = 0.0,
        critical_multiplier: float = 2.0,
    ) -> DamageResult:
        """
        Process a damage instance and calculate results.

        Args:
            info: DamageInfo describing the damage
            armor: Target's armor value
            resistance: Target's resistance to this damage type
            critical_multiplier: Multiplier when critical hit

        Returns:
            DamageResult with calculated values
        """
        # Gather multipliers
        multipliers: List[float] = []

        # Apply global modifiers
        base = info.base_damage
        for mod in sorted(self._global_modifiers, key=lambda m: -m.priority):
            base = mod.apply(base, info)
            if mod.multiplier != 1.0 or mod.flat_bonus != 0.0:
                info.multipliers_applied.append(mod.name)

        # Apply type-specific modifiers
        type_mods = self._type_modifiers.get(info.damage_type, [])
        for mod in sorted(type_mods, key=lambda m: -m.priority):
            base = mod.apply(base, info)
            if mod.multiplier != 1.0 or mod.flat_bonus != 0.0:
                info.multipliers_applied.append(mod.name)

        # Calculate final damage
        crit_mult = critical_multiplier if info.is_critical else 1.0
        final_damage, armor_blocked, resistance_blocked = self.calculate_damage(
            base_damage=base,
            damage_type=info.damage_type,
            armor=armor,
            resistance=resistance,
            hitbox_zone=info.hitbox_zone,
            critical_multiplier=crit_mult,
            additional_multipliers=multipliers if multipliers else None,
        )

        # Update damage info
        info.final_damage = final_damage
        info.damage_mitigated = armor_blocked + resistance_blocked
        info.armor_applied = armor_blocked
        info.resistance_applied = resistance_blocked

        # Record history
        self._record_damage(info)

        # Emit events
        self._emit_event(CombatEventType.DAMAGE_DEALT, info)
        if info.is_headshot:
            self._emit_event(CombatEventType.HEADSHOT, info)
        if info.is_critical:
            self._emit_event(CombatEventType.CRITICAL_HIT, info)

        return DamageResult(
            damage_dealt=final_damage,
            damage_blocked=armor_blocked,
            damage_resisted=resistance_blocked,
            was_lethal=False,  # Caller should determine lethality
            was_critical=info.is_critical,
            was_headshot=info.is_headshot,
            damage_info=info,
        )

    def apply_damage_to_receiver(
        self,
        info: DamageInfo,
        receiver: DamageReceiver,
        critical_multiplier: float = 2.0,
    ) -> DamageResult:
        """
        Apply damage to an entity implementing DamageReceiver.

        Args:
            info: DamageInfo describing the damage
            receiver: Entity receiving the damage
            critical_multiplier: Multiplier for critical hits

        Returns:
            DamageResult with actual damage dealt
        """
        # Check invulnerability
        if receiver.is_invulnerable():
            return DamageResult(
                damage_dealt=0.0,
                damage_blocked=info.base_damage,
                damage_resisted=0.0,
                was_lethal=False,
                was_critical=info.is_critical,
                was_headshot=info.is_headshot,
                damage_info=info,
            )

        # Get defense values
        armor = receiver.get_armor()
        resistance = receiver.get_resistance(info.damage_type)

        # Process damage
        result = self.process_damage(info, armor, resistance, critical_multiplier)

        # Apply to receiver
        actual_damage = receiver.apply_damage(result.damage_dealt, info)

        # Update result if actual differs (e.g., shields, damage cap)
        if actual_damage != result.damage_dealt:
            result = DamageResult(
                damage_dealt=actual_damage,
                damage_blocked=result.damage_blocked + (result.damage_dealt - actual_damage),
                damage_resisted=result.damage_resisted,
                was_lethal=result.was_lethal,
                was_critical=result.was_critical,
                was_headshot=result.was_headshot,
                damage_info=info,
            )

        return result

    # =========================================================================
    # MODIFIERS
    # =========================================================================

    def add_global_modifier(self, modifier: DamageModifier) -> None:
        """Add a modifier that applies to all damage."""
        self._global_modifiers.append(modifier)

    def remove_global_modifier(self, name: str) -> bool:
        """Remove a global modifier by name. Returns True if found."""
        for i, mod in enumerate(self._global_modifiers):
            if mod.name == name:
                del self._global_modifiers[i]
                return True
        return False

    def add_type_modifier(self, damage_type: DamageType, modifier: DamageModifier) -> None:
        """Add a modifier for a specific damage type."""
        if damage_type not in self._type_modifiers:
            self._type_modifiers[damage_type] = []
        self._type_modifiers[damage_type].append(modifier)

    def remove_type_modifier(self, damage_type: DamageType, name: str) -> bool:
        """Remove a type-specific modifier by name. Returns True if found."""
        if damage_type not in self._type_modifiers:
            return False
        mods = self._type_modifiers[damage_type]
        for i, mod in enumerate(mods):
            if mod.name == name:
                del mods[i]
                return True
        return False

    def clear_modifiers(self) -> None:
        """Remove all damage modifiers."""
        self._global_modifiers.clear()
        self._type_modifiers.clear()

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================

    def register_event_handler(
        self,
        event_type: CombatEventType,
        handler: Callable[[DamageInfo], None],
    ) -> None:
        """Register a handler for combat events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def unregister_event_handler(
        self,
        event_type: CombatEventType,
        handler: Callable[[DamageInfo], None],
    ) -> bool:
        """Unregister an event handler. Returns True if found."""
        if event_type not in self._event_handlers:
            return False
        try:
            self._event_handlers[event_type].remove(handler)
            return True
        except ValueError:
            return False

    def _emit_event(self, event_type: CombatEventType, info: DamageInfo) -> None:
        """Emit an event to all registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(info)
            except Exception:
                # Log but don't propagate handler errors
                pass

    # =========================================================================
    # HISTORY
    # =========================================================================

    def _record_damage(self, info: DamageInfo) -> None:
        """Record damage to history."""
        self._damage_history.append(info)
        # Trim history if needed
        if len(self._damage_history) > self._max_history_size:
            self._damage_history = self._damage_history[-self._max_history_size:]

    def get_damage_history(
        self,
        entity_id: Optional[int] = None,
        as_attacker: bool = True,
        as_target: bool = True,
        limit: int = 100,
    ) -> List[DamageInfo]:
        """
        Get damage history, optionally filtered by entity.

        Args:
            entity_id: Filter to specific entity (None for all)
            as_attacker: Include damage dealt by entity
            as_target: Include damage received by entity
            limit: Maximum entries to return

        Returns:
            List of DamageInfo entries
        """
        if entity_id is None:
            return self._damage_history[-limit:]

        filtered = []
        for info in reversed(self._damage_history):
            if len(filtered) >= limit:
                break
            if as_attacker and info.attacker_id == entity_id:
                filtered.append(info)
            elif as_target and info.target_id == entity_id:
                filtered.append(info)

        return list(reversed(filtered))

    def clear_history(self) -> None:
        """Clear damage history."""
        self._damage_history.clear()

    def get_total_damage_dealt(
        self,
        attacker_id: int,
        damage_type: Optional[DamageType] = None,
        time_window: Optional[float] = None,
    ) -> float:
        """
        Get total damage dealt by an entity.

        Args:
            attacker_id: Entity ID to check
            damage_type: Optional filter by damage type
            time_window: Optional time window in seconds (from now)

        Returns:
            Total damage dealt
        """
        total = 0.0
        cutoff_time = time.time() - time_window if time_window else 0

        for info in self._damage_history:
            if info.attacker_id != attacker_id:
                continue
            if time_window and info.timestamp < cutoff_time:
                continue
            if damage_type and info.damage_type != damage_type:
                continue
            total += info.final_damage

        return total


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def calculate_dps(
    base_damage: float,
    attacks_per_second: float,
    crit_chance: float = 0.0,
    crit_multiplier: float = 2.0,
) -> float:
    """
    Calculate damage per second (DPS).

    Args:
        base_damage: Damage per hit
        attacks_per_second: Attack speed
        crit_chance: Critical hit chance (0.0-1.0)
        crit_multiplier: Critical hit damage multiplier

    Returns:
        Average DPS
    """
    avg_damage = base_damage * (1.0 + crit_chance * (crit_multiplier - 1.0))
    return avg_damage * attacks_per_second


def calculate_effective_health(
    health: float,
    armor: float,
    resistance: float = 0.0,
    armor_constant: float = ARMOR_CONSTANT,
) -> float:
    """
    Calculate effective health against physical damage.

    Args:
        health: Base health
        armor: Armor value
        resistance: Physical damage resistance (0.0-1.0)
        armor_constant: Armor formula constant

    Returns:
        Effective health value
    """
    armor_mult = 1.0 + (armor / armor_constant)
    resist_mult = 1.0 / (1.0 - min(resistance, MAX_RESISTANCE))
    return health * armor_mult * resist_mult


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "DamageSystem",
    "DamageInfo",
    "DamageResult",
    "DamageModifier",
    "ResistanceProfile",
    # Protocols
    "DamageReceiver",
    "DamageDealer",
    # Utility functions
    "calculate_dps",
    "calculate_effective_health",
]
