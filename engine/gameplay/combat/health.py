"""
Combat System - Health Module

Provides the HealthComponent for managing entity health with support for:
- Current/max health tracking
- Health regeneration (with delay after damage)
- Invulnerability periods
- Health change events
- Shields and temporary health
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
import time

from .constants import (
    DEFAULT_MAX_HEALTH,
    DEFAULT_CURRENT_HEALTH,
    MINIMUM_MAX_HEALTH,
    DEFAULT_HEALTH_REGEN_RATE,
    MAX_HEALTH_REGEN_RATE,
    REGEN_DELAY_AFTER_DAMAGE,
    OUT_OF_COMBAT_THRESHOLD,
    OUT_OF_COMBAT_REGEN_MULTIPLIER,
    RESPAWN_INVULNERABILITY_DURATION,
    HealthConfig,
    DEFAULT_HEALTH_CONFIG,
    CombatEventType,
)
from .damage import DamageInfo


# =============================================================================
# ENUMS
# =============================================================================


class HealthChangeReason(Enum):
    """Reasons for health changes."""

    DAMAGE = auto()
    HEALING = auto()
    REGENERATION = auto()
    MAX_HEALTH_CHANGE = auto()
    DEATH = auto()
    RESPAWN = auto()
    EFFECT = auto()  # Buff/debuff effect
    DIRECT_SET = auto()  # Direct manipulation
    SHIELD_ABSORBED = auto()


class InvulnerabilityReason(Enum):
    """Reasons for invulnerability."""

    RESPAWN = auto()
    ABILITY = auto()
    BUFF = auto()
    CINEMATIC = auto()
    CUSTOM = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class HealthChangeEvent:
    """Event data for health changes."""

    entity_id: int
    old_health: float
    new_health: float
    max_health: float
    change_amount: float
    reason: HealthChangeReason
    timestamp: float = field(default_factory=time.time)
    source_id: Optional[int] = None
    damage_info: Optional[DamageInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def health_percentage(self) -> float:
        """Current health as a percentage (0.0-1.0)."""
        if self.max_health <= 0:
            return 0.0
        return self.new_health / self.max_health

    @property
    def is_damage(self) -> bool:
        """Whether this was a damaging change."""
        return self.change_amount < 0

    @property
    def is_healing(self) -> bool:
        """Whether this was a healing change."""
        return self.change_amount > 0

    @property
    def is_lethal(self) -> bool:
        """Whether this change resulted in death."""
        return self.old_health > 0 and self.new_health <= 0


@dataclass
class ShieldInfo:
    """Information about a damage shield."""

    name: str
    amount: float
    max_amount: float
    priority: int = 0  # Higher priority absorbs first
    damage_types: Optional[set] = None  # None = absorbs all types
    duration: Optional[float] = None  # None = permanent until depleted
    created_at: float = field(default_factory=time.time)
    source_id: Optional[int] = None

    @property
    def remaining_percentage(self) -> float:
        """Remaining shield as percentage."""
        if self.max_amount <= 0:
            return 0.0
        return self.amount / self.max_amount

    @property
    def is_expired(self) -> bool:
        """Check if shield has timed out."""
        if self.duration is None:
            return False
        return time.time() - self.created_at >= self.duration

    def absorb(self, damage: float, damage_type: Optional[Any] = None) -> Tuple[float, float]:
        """
        Absorb damage with this shield.

        Args:
            damage: Incoming damage
            damage_type: Type of damage (for type-specific shields)

        Returns:
            Tuple of (remaining_damage, absorbed_amount)
        """
        # Check if shield can absorb this damage type
        if self.damage_types is not None and damage_type not in self.damage_types:
            return (damage, 0.0)

        absorbed = min(damage, self.amount)
        self.amount -= absorbed
        return (damage - absorbed, absorbed)


@dataclass
class InvulnerabilityInfo:
    """Information about an invulnerability period."""

    reason: InvulnerabilityReason
    duration: float
    started_at: float = field(default_factory=time.time)
    allow_healing: bool = True
    source_id: Optional[int] = None

    @property
    def remaining_time(self) -> float:
        """Time remaining in invulnerability."""
        elapsed = time.time() - self.started_at
        return max(0.0, self.duration - elapsed)

    @property
    def is_expired(self) -> bool:
        """Check if invulnerability has expired."""
        return self.remaining_time <= 0


# =============================================================================
# HEALTH COMPONENT
# =============================================================================


class HealthComponent:
    """
    Component for managing entity health.

    Features:
    - Current and maximum health tracking
    - Health regeneration with configurable rates
    - Combat state tracking (in/out of combat)
    - Invulnerability periods
    - Damage shields
    - Health change event callbacks
    """

    def __init__(
        self,
        entity_id: int,
        max_health: float = DEFAULT_MAX_HEALTH,
        current_health: Optional[float] = None,
        config: Optional[HealthConfig] = None,
    ) -> None:
        """
        Initialize health component.

        Args:
            entity_id: ID of the owning entity
            max_health: Maximum health value
            current_health: Starting health (defaults to max)
            config: Health configuration
        """
        self._entity_id = entity_id
        self._config = config or DEFAULT_HEALTH_CONFIG
        self._max_health = max(max_health, self._config.minimum_max_health)
        self._current_health = current_health if current_health is not None else self._max_health
        self._current_health = max(0.0, min(self._current_health, self._max_health))

        # Regeneration
        self._regen_rate: float = self._config.default_regen_rate
        self._regen_enabled: bool = True
        self._last_damage_time: float = 0.0

        # Invulnerability
        self._invulnerabilities: List[InvulnerabilityInfo] = []

        # Shields
        self._shields: List[ShieldInfo] = []

        # Event handlers
        self._on_health_changed: List[Callable[[HealthChangeEvent], None]] = []
        self._on_death: List[Callable[[HealthChangeEvent], None]] = []
        self._on_revive: List[Callable[[HealthChangeEvent], None]] = []

        # State
        self._is_dead: bool = False
        self._accumulated_regen: float = 0.0

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def entity_id(self) -> int:
        """Get the owning entity's ID."""
        return self._entity_id

    @property
    def current_health(self) -> float:
        """Get current health."""
        return self._current_health

    @property
    def max_health(self) -> float:
        """Get maximum health."""
        return self._max_health

    @property
    def health_percentage(self) -> float:
        """Get health as a percentage (0.0-1.0)."""
        if self._max_health <= 0:
            return 0.0
        return self._current_health / self._max_health

    @property
    def missing_health(self) -> float:
        """Get amount of missing health."""
        return self._max_health - self._current_health

    @property
    def is_full_health(self) -> bool:
        """Check if at full health."""
        return self._current_health >= self._max_health

    @property
    def is_dead(self) -> bool:
        """Check if entity is dead."""
        return self._is_dead

    @property
    def is_alive(self) -> bool:
        """Check if entity is alive."""
        return not self._is_dead

    @property
    def regen_rate(self) -> float:
        """Get current regeneration rate (per second)."""
        return self._regen_rate

    @regen_rate.setter
    def regen_rate(self, value: float) -> None:
        """Set regeneration rate."""
        self._regen_rate = max(0.0, min(value, self._config.max_regen_rate))

    @property
    def is_invulnerable(self) -> bool:
        """Check if currently invulnerable."""
        self._cleanup_expired_invulnerabilities()
        return len(self._invulnerabilities) > 0

    @property
    def total_shield(self) -> float:
        """Get total shield amount."""
        self._cleanup_expired_shields()
        return sum(s.amount for s in self._shields)

    @property
    def effective_health(self) -> float:
        """Get health including shields."""
        return self._current_health + self.total_shield

    @property
    def is_in_combat(self) -> bool:
        """Check if entity is in combat (recently damaged)."""
        time_since_damage = time.time() - self._last_damage_time
        return time_since_damage < self._config.out_of_combat_threshold

    @property
    def time_since_damage(self) -> float:
        """Get time since last damage received."""
        return time.time() - self._last_damage_time

    # =========================================================================
    # HEALTH MODIFICATION
    # =========================================================================

    def take_damage(
        self,
        amount: float,
        damage_info: Optional[DamageInfo] = None,
        source_id: Optional[int] = None,
        ignore_shields: bool = False,
        ignore_invulnerability: bool = False,
    ) -> float:
        """
        Apply damage to this entity.

        Args:
            amount: Damage amount
            damage_info: Optional detailed damage information
            source_id: ID of damage source entity
            ignore_shields: Whether to bypass shields
            ignore_invulnerability: Whether to bypass invulnerability

        Returns:
            Actual damage dealt (after shields, etc.)
        """
        if amount <= 0 or self._is_dead:
            return 0.0

        # Check invulnerability
        if not ignore_invulnerability and self.is_invulnerable:
            return 0.0

        # Apply shields first
        actual_damage = amount
        shield_absorbed = 0.0
        if not ignore_shields:
            actual_damage, shield_absorbed = self._apply_shields(
                amount,
                damage_info.damage_type if damage_info else None,
            )

        # Apply damage to health
        old_health = self._current_health
        self._current_health = max(0.0, self._current_health - actual_damage)

        # Update combat state
        self._last_damage_time = time.time()

        # Create event
        event = HealthChangeEvent(
            entity_id=self._entity_id,
            old_health=old_health,
            new_health=self._current_health,
            max_health=self._max_health,
            change_amount=-actual_damage,
            reason=HealthChangeReason.DAMAGE,
            source_id=source_id,
            damage_info=damage_info,
            metadata={"shield_absorbed": shield_absorbed},
        )

        # Emit events
        self._emit_health_changed(event)

        # Check for death
        if self._current_health <= 0 and not self._is_dead:
            self._is_dead = True
            self._emit_death(event)

        return actual_damage

    def heal(
        self,
        amount: float,
        source_id: Optional[int] = None,
        reason: HealthChangeReason = HealthChangeReason.HEALING,
        allow_overheal: bool = False,
    ) -> float:
        """
        Heal the entity.

        Args:
            amount: Healing amount
            source_id: ID of healing source
            reason: Reason for the healing
            allow_overheal: Whether to allow healing beyond max health

        Returns:
            Actual amount healed
        """
        if amount <= 0 or self._is_dead:
            return 0.0

        old_health = self._current_health

        if allow_overheal:
            self._current_health += amount
        else:
            self._current_health = min(self._max_health, self._current_health + amount)

        actual_heal = self._current_health - old_health

        if actual_heal > 0:
            event = HealthChangeEvent(
                entity_id=self._entity_id,
                old_health=old_health,
                new_health=self._current_health,
                max_health=self._max_health,
                change_amount=actual_heal,
                reason=reason,
                source_id=source_id,
            )
            self._emit_health_changed(event)

        return actual_heal

    def set_health(
        self,
        value: float,
        source_id: Optional[int] = None,
    ) -> None:
        """
        Directly set health to a value.

        Args:
            value: New health value
            source_id: ID of source making the change
        """
        old_health = self._current_health
        self._current_health = max(0.0, min(value, self._max_health))

        if self._current_health != old_health:
            change = self._current_health - old_health
            event = HealthChangeEvent(
                entity_id=self._entity_id,
                old_health=old_health,
                new_health=self._current_health,
                max_health=self._max_health,
                change_amount=change,
                reason=HealthChangeReason.DIRECT_SET,
                source_id=source_id,
            )
            self._emit_health_changed(event)

            # Check for death from direct set
            if self._current_health <= 0 and not self._is_dead:
                self._is_dead = True
                self._emit_death(event)

    def set_max_health(
        self,
        value: float,
        adjust_current: bool = True,
    ) -> None:
        """
        Set maximum health.

        Args:
            value: New maximum health
            adjust_current: Whether to adjust current health proportionally
        """
        old_max = self._max_health
        self._max_health = max(value, self._config.minimum_max_health)

        if adjust_current and old_max > 0:
            ratio = self._current_health / old_max
            old_current = self._current_health
            self._current_health = self._max_health * ratio

            if self._current_health != old_current:
                event = HealthChangeEvent(
                    entity_id=self._entity_id,
                    old_health=old_current,
                    new_health=self._current_health,
                    max_health=self._max_health,
                    change_amount=self._current_health - old_current,
                    reason=HealthChangeReason.MAX_HEALTH_CHANGE,
                )
                self._emit_health_changed(event)
        else:
            # Clamp current health to new max
            if self._current_health > self._max_health:
                old_current = self._current_health
                self._current_health = self._max_health

                event = HealthChangeEvent(
                    entity_id=self._entity_id,
                    old_health=old_current,
                    new_health=self._current_health,
                    max_health=self._max_health,
                    change_amount=self._current_health - old_current,
                    reason=HealthChangeReason.MAX_HEALTH_CHANGE,
                )
                self._emit_health_changed(event)

    def modify_max_health(self, delta: float, adjust_current: bool = True) -> None:
        """
        Modify maximum health by a delta.

        Args:
            delta: Amount to add (can be negative)
            adjust_current: Whether to adjust current health proportionally
        """
        self.set_max_health(self._max_health + delta, adjust_current)

    # =========================================================================
    # REGENERATION
    # =========================================================================

    def update_regeneration(self, delta_time: float) -> float:
        """
        Update health regeneration.

        Args:
            delta_time: Time elapsed since last update (seconds)

        Returns:
            Amount regenerated
        """
        if not self._regen_enabled or self._is_dead or self._regen_rate <= 0:
            return 0.0

        # Check regen delay after damage
        if self.time_since_damage < self._config.regen_delay_after_damage:
            return 0.0

        # Calculate regen amount
        regen_mult = 1.0
        if not self.is_in_combat:
            regen_mult = self._config.out_of_combat_regen_multiplier

        regen_amount = self._regen_rate * delta_time * regen_mult

        # Apply regeneration
        return self.heal(regen_amount, reason=HealthChangeReason.REGENERATION)

    def enable_regeneration(self) -> None:
        """Enable health regeneration."""
        self._regen_enabled = True

    def disable_regeneration(self) -> None:
        """Disable health regeneration."""
        self._regen_enabled = False

    # =========================================================================
    # INVULNERABILITY
    # =========================================================================

    def add_invulnerability(
        self,
        duration: float,
        reason: InvulnerabilityReason = InvulnerabilityReason.CUSTOM,
        allow_healing: bool = True,
        source_id: Optional[int] = None,
    ) -> InvulnerabilityInfo:
        """
        Add an invulnerability period.

        Args:
            duration: Duration in seconds
            reason: Reason for invulnerability
            allow_healing: Whether healing is allowed during invulnerability
            source_id: Source granting invulnerability

        Returns:
            InvulnerabilityInfo for the new period
        """
        info = InvulnerabilityInfo(
            reason=reason,
            duration=duration,
            allow_healing=allow_healing,
            source_id=source_id,
        )
        self._invulnerabilities.append(info)
        return info

    def remove_invulnerability(self, reason: Optional[InvulnerabilityReason] = None) -> int:
        """
        Remove invulnerability periods.

        Args:
            reason: If provided, only remove invulnerabilities with this reason.
                   If None, removes all.

        Returns:
            Number of invulnerabilities removed
        """
        if reason is None:
            count = len(self._invulnerabilities)
            self._invulnerabilities.clear()
            return count

        original_count = len(self._invulnerabilities)
        self._invulnerabilities = [
            inv for inv in self._invulnerabilities if inv.reason != reason
        ]
        return original_count - len(self._invulnerabilities)

    def _cleanup_expired_invulnerabilities(self) -> None:
        """Remove expired invulnerability periods."""
        self._invulnerabilities = [
            inv for inv in self._invulnerabilities if not inv.is_expired
        ]

    def get_invulnerability_remaining(self) -> float:
        """Get remaining invulnerability time (max of all active)."""
        self._cleanup_expired_invulnerabilities()
        if not self._invulnerabilities:
            return 0.0
        return max(inv.remaining_time for inv in self._invulnerabilities)

    # =========================================================================
    # SHIELDS
    # =========================================================================

    def add_shield(
        self,
        name: str,
        amount: float,
        priority: int = 0,
        damage_types: Optional[set] = None,
        duration: Optional[float] = None,
        source_id: Optional[int] = None,
    ) -> ShieldInfo:
        """
        Add a damage shield.

        Args:
            name: Unique name for the shield
            amount: Shield amount
            priority: Higher priority shields absorb first
            damage_types: Set of damage types this shield blocks (None = all)
            duration: Duration in seconds (None = until depleted)
            source_id: Entity that granted the shield

        Returns:
            ShieldInfo for the new shield
        """
        # Remove existing shield with same name
        self.remove_shield(name)

        shield = ShieldInfo(
            name=name,
            amount=amount,
            max_amount=amount,
            priority=priority,
            damage_types=damage_types,
            duration=duration,
            source_id=source_id,
        )
        self._shields.append(shield)
        # Sort by priority (highest first)
        self._shields.sort(key=lambda s: -s.priority)
        return shield

    def remove_shield(self, name: str) -> bool:
        """
        Remove a shield by name.

        Args:
            name: Shield name to remove

        Returns:
            True if shield was found and removed
        """
        for i, shield in enumerate(self._shields):
            if shield.name == name:
                del self._shields[i]
                return True
        return False

    def get_shield(self, name: str) -> Optional[ShieldInfo]:
        """Get a shield by name."""
        for shield in self._shields:
            if shield.name == name:
                return shield
        return None

    def _apply_shields(
        self,
        damage: float,
        damage_type: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """
        Apply damage through shields.

        Args:
            damage: Incoming damage
            damage_type: Type of damage

        Returns:
            Tuple of (remaining_damage, total_absorbed)
        """
        self._cleanup_expired_shields()

        remaining = damage
        total_absorbed = 0.0

        for shield in self._shields[:]:  # Copy list since we may modify it
            if remaining <= 0:
                break
            remaining, absorbed = shield.absorb(remaining, damage_type)
            total_absorbed += absorbed

            # Remove depleted shields
            if shield.amount <= 0:
                self._shields.remove(shield)

        return (remaining, total_absorbed)

    def _cleanup_expired_shields(self) -> None:
        """Remove expired shields."""
        self._shields = [s for s in self._shields if not s.is_expired]

    # =========================================================================
    # DEATH AND REVIVAL
    # =========================================================================

    def kill(self, source_id: Optional[int] = None) -> bool:
        """
        Instantly kill the entity.

        Args:
            source_id: ID of entity causing the kill

        Returns:
            True if entity was killed (False if already dead)
        """
        if self._is_dead:
            return False

        old_health = self._current_health
        self._current_health = 0.0
        self._is_dead = True

        event = HealthChangeEvent(
            entity_id=self._entity_id,
            old_health=old_health,
            new_health=0.0,
            max_health=self._max_health,
            change_amount=-old_health,
            reason=HealthChangeReason.DEATH,
            source_id=source_id,
        )

        self._emit_health_changed(event)
        self._emit_death(event)

        return True

    def revive(
        self,
        health_percentage: float = 1.0,
        source_id: Optional[int] = None,
        add_invulnerability: bool = True,
        invulnerability_duration: float = RESPAWN_INVULNERABILITY_DURATION,
    ) -> bool:
        """
        Revive the entity from death.

        Args:
            health_percentage: Health to restore (0.0-1.0 of max)
            source_id: ID of reviving entity
            add_invulnerability: Whether to add spawn protection
            invulnerability_duration: Duration of spawn protection

        Returns:
            True if entity was revived (False if not dead)
        """
        if not self._is_dead:
            return False

        self._is_dead = False
        old_health = self._current_health
        self._current_health = self._max_health * max(0.0, min(1.0, health_percentage))

        # Clear shields on revive
        self._shields.clear()

        # Add spawn protection
        if add_invulnerability and invulnerability_duration > 0:
            self.add_invulnerability(
                duration=invulnerability_duration,
                reason=InvulnerabilityReason.RESPAWN,
            )

        event = HealthChangeEvent(
            entity_id=self._entity_id,
            old_health=old_health,
            new_health=self._current_health,
            max_health=self._max_health,
            change_amount=self._current_health - old_health,
            reason=HealthChangeReason.RESPAWN,
            source_id=source_id,
        )

        self._emit_health_changed(event)
        self._emit_revive(event)

        return True

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_health_changed(self, handler: Callable[[HealthChangeEvent], None]) -> None:
        """Register a handler for health changes."""
        self._on_health_changed.append(handler)

    def on_death(self, handler: Callable[[HealthChangeEvent], None]) -> None:
        """Register a handler for death."""
        self._on_death.append(handler)

    def on_revive(self, handler: Callable[[HealthChangeEvent], None]) -> None:
        """Register a handler for revival."""
        self._on_revive.append(handler)

    def _emit_health_changed(self, event: HealthChangeEvent) -> None:
        """Emit health changed event."""
        for handler in self._on_health_changed:
            try:
                handler(event)
            except Exception:
                pass

    def _emit_death(self, event: HealthChangeEvent) -> None:
        """Emit death event."""
        for handler in self._on_death:
            try:
                handler(event)
            except Exception:
                pass

    def _emit_revive(self, event: HealthChangeEvent) -> None:
        """Emit revive event."""
        for handler in self._on_revive:
            try:
                handler(event)
            except Exception:
                pass

    # =========================================================================
    # UTILITY
    # =========================================================================

    def reset(self) -> None:
        """Reset health component to initial state."""
        self._current_health = self._max_health
        self._is_dead = False
        self._shields.clear()
        self._invulnerabilities.clear()
        self._last_damage_time = 0.0
        self._accumulated_regen = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "current_health": self._current_health,
            "max_health": self._max_health,
            "is_dead": self._is_dead,
            "regen_rate": self._regen_rate,
            "regen_enabled": self._regen_enabled,
            "shields": [
                {
                    "name": s.name,
                    "amount": s.amount,
                    "max_amount": s.max_amount,
                    "priority": s.priority,
                }
                for s in self._shields
            ],
        }

    def __repr__(self) -> str:
        return (
            f"HealthComponent(entity_id={self._entity_id}, "
            f"health={self._current_health:.1f}/{self._max_health:.1f}, "
            f"dead={self._is_dead})"
        )


# =============================================================================
# HEALTH POOL (Multiple Entities)
# =============================================================================


class HealthPool:
    """
    Manages health components for multiple entities.

    Useful for systems that need to track and update many entities.
    """

    def __init__(self, config: Optional[HealthConfig] = None) -> None:
        """Initialize the health pool."""
        self._config = config or DEFAULT_HEALTH_CONFIG
        self._components: Dict[int, HealthComponent] = {}

    def create(
        self,
        entity_id: int,
        max_health: float = DEFAULT_MAX_HEALTH,
        current_health: Optional[float] = None,
    ) -> HealthComponent:
        """Create a health component for an entity."""
        component = HealthComponent(
            entity_id=entity_id,
            max_health=max_health,
            current_health=current_health,
            config=self._config,
        )
        self._components[entity_id] = component
        return component

    def get(self, entity_id: int) -> Optional[HealthComponent]:
        """Get health component for an entity."""
        return self._components.get(entity_id)

    def remove(self, entity_id: int) -> bool:
        """Remove health component for an entity."""
        if entity_id in self._components:
            del self._components[entity_id]
            return True
        return False

    def update_all(self, delta_time: float) -> None:
        """Update regeneration for all entities."""
        for component in self._components.values():
            component.update_regeneration(delta_time)

    def get_all_alive(self) -> List[HealthComponent]:
        """Get all alive entities."""
        return [c for c in self._components.values() if c.is_alive]

    def get_all_dead(self) -> List[HealthComponent]:
        """Get all dead entities."""
        return [c for c in self._components.values() if c.is_dead]

    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, entity_id: int) -> bool:
        return entity_id in self._components


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HealthChangeReason",
    "InvulnerabilityReason",
    # Data classes
    "HealthChangeEvent",
    "ShieldInfo",
    "InvulnerabilityInfo",
    # Components
    "HealthComponent",
    "HealthPool",
]
