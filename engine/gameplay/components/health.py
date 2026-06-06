"""
Health Component - Health tracking with damage resistance and regeneration.

Provides health management for entities including current/max health,
damage modification, regeneration, and death handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from trinity.descriptors import (
    TrackedDescriptor,
    RangeDescriptor,
    StorageDescriptor,
    clear_dirty,
    get_dirty_fields,
    is_dirty,
)

from engine.gameplay.components.constants import HealthConstants

if TYPE_CHECKING:
    from foundation import to_dict, from_dict


class DamageType(Enum):
    """Types of damage for resistance calculations."""
    PHYSICAL = auto()
    FIRE = auto()
    ICE = auto()
    LIGHTNING = auto()
    POISON = auto()
    MAGIC = auto()
    TRUE = auto()  # Ignores resistances


class HealthState(Enum):
    """Current health state of an entity."""
    ALIVE = auto()
    DEAD = auto()
    INVULNERABLE = auto()
    DOWNED = auto()  # For revival mechanics


@dataclass
class DamageEvent:
    """Record of a damage event."""
    amount: float
    damage_type: DamageType
    source_id: Optional[str] = None
    timestamp: float = 0.0
    was_lethal: bool = False
    final_damage: float = 0.0  # After resistances


@dataclass
class HealEvent:
    """Record of a healing event."""
    amount: float
    source_id: Optional[str] = None
    timestamp: float = 0.0
    actual_healing: float = 0.0  # After overheal prevention


class HealthComponent:
    """
    Health component with damage resistance and regeneration support.

    Features:
    - Current and maximum health tracking
    - Damage type resistances
    - Health regeneration over time
    - Damage/healing event callbacks
    - Shield/armor integration points
    - Dirty tracking for network sync

    Attributes:
        current_health: Current health value
        max_health: Maximum health value
        regen_rate: Health regeneration per second
        resistances: Damage type resistance multipliers
    """

    # Tracked descriptor for current health with clamping
    current_health = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=0,
    )

    # Max health with range validation
    max_health = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=1,
    )

    # Regeneration rate
    regen_rate = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=2,
    )

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_state",
        "_resistances",
        "_armor",
        "_shield",
        "_shield_max",
        "_invulnerable_timer",
        "_damage_multiplier",
        "_healing_multiplier",
        "_on_damage",
        "_on_heal",
        "_on_death",
        "_on_revive",
        "_entity_id",
        "_damage_history",
        "_heal_history",
    )

    def __init__(
        self,
        max_health: float = HealthConstants.DEFAULT_MAX_HEALTH,
        current_health: Optional[float] = None,
        regen_rate: float = HealthConstants.DEFAULT_REGEN_RATE,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the health component.

        Args:
            max_health: Maximum health value (must be > 0)
            current_health: Starting health (default: max_health)
            regen_rate: Health regeneration per second
            entity_id: Optional entity ID for tracking
        """
        if max_health <= 0:
            raise ValueError("max_health must be greater than 0")

        self._state = HealthState.ALIVE
        self._resistances: Dict[DamageType, float] = {}
        self._armor: float = 0.0
        self._shield: float = 0.0
        self._shield_max: float = 0.0
        self._invulnerable_timer: float = 0.0
        self._damage_multiplier: float = 1.0
        self._healing_multiplier: float = 1.0
        self._entity_id = entity_id

        # Callbacks
        self._on_damage: List[Callable[[DamageEvent], None]] = []
        self._on_heal: List[Callable[[HealEvent], None]] = []
        self._on_death: List[Callable[[HealthComponent], None]] = []
        self._on_revive: List[Callable[[HealthComponent], None]] = []

        # History for replay/analysis
        self._damage_history: List[DamageEvent] = []
        self._heal_history: List[HealEvent] = []

        # Set values (tracked)
        self.max_health = max_health
        self.current_health = current_health if current_health is not None else max_health
        self.regen_rate = regen_rate

        # Clamp current health to max
        if self.current_health > self.max_health:
            self.current_health = self.max_health

        clear_dirty(self)

    # =========================================================================
    # HEALTH PROPERTIES
    # =========================================================================

    @property
    def health_percentage(self) -> float:
        """Get health as a percentage (0.0 to 1.0)."""
        if self.max_health <= 0:
            return 0.0
        return self.current_health / self.max_health

    @property
    def missing_health(self) -> float:
        """Get amount of missing health."""
        return self.max_health - self.current_health

    @property
    def is_full_health(self) -> bool:
        """Check if at full health."""
        return self.current_health >= self.max_health

    @property
    def is_alive(self) -> bool:
        """Check if the entity is alive."""
        return self._state in (HealthState.ALIVE, HealthState.INVULNERABLE)

    @property
    def is_dead(self) -> bool:
        """Check if the entity is dead."""
        return self._state == HealthState.DEAD

    @property
    def is_invulnerable(self) -> bool:
        """Check if currently invulnerable."""
        return self._state == HealthState.INVULNERABLE or self._invulnerable_timer > 0

    @property
    def state(self) -> HealthState:
        """Get current health state."""
        return self._state

    @property
    def effective_health(self) -> float:
        """Get total effective health (health + shield)."""
        return self.current_health + self._shield

    # =========================================================================
    # DAMAGE AND HEALING
    # =========================================================================

    def take_damage(
        self,
        amount: float,
        damage_type: DamageType = DamageType.PHYSICAL,
        source_id: Optional[str] = None,
        timestamp: float = 0.0,
        ignore_armor: bool = False,
        ignore_resistance: bool = False,
    ) -> DamageEvent:
        """
        Apply damage to the entity.

        Args:
            amount: Raw damage amount
            damage_type: Type of damage for resistance calculation
            source_id: ID of the damage source
            timestamp: Time of the damage event
            ignore_armor: If True, bypass armor reduction
            ignore_resistance: If True, bypass damage resistance

        Returns:
            DamageEvent with details of the damage taken
        """
        event = DamageEvent(
            amount=amount,
            damage_type=damage_type,
            source_id=source_id,
            timestamp=timestamp,
        )

        # Check invulnerability - blocks ALL damage including TRUE damage
        if self.is_invulnerable:
            event.final_damage = 0.0
            return event

        # Check if already dead
        if self.is_dead:
            event.final_damage = 0.0
            return event

        # Calculate final damage
        final_damage = amount * self._damage_multiplier

        # Apply resistance
        if not ignore_resistance and damage_type != DamageType.TRUE:
            resistance = self._resistances.get(damage_type, 0.0)
            final_damage *= (1.0 - min(resistance, HealthConstants.MAX_RESISTANCE_CAP))

        # Apply armor
        if not ignore_armor and damage_type != DamageType.TRUE:
            final_damage = max(0, final_damage - self._armor)

        # Absorb with shield first
        if self._shield > 0 and final_damage > 0:
            shield_absorbed = min(self._shield, final_damage)
            self._shield -= shield_absorbed
            final_damage -= shield_absorbed

        # Apply to health
        if final_damage > 0:
            old_health = self.current_health
            self.current_health = max(0.0, self.current_health - final_damage)

            event.final_damage = old_health - self.current_health
            event.was_lethal = self.current_health <= 0

            # Check for death
            if self.current_health <= 0:
                self._die()

        # Record and notify
        self._damage_history.append(event)
        for callback in self._on_damage:
            callback(event)

        return event

    def heal(
        self,
        amount: float,
        source_id: Optional[str] = None,
        timestamp: float = 0.0,
        can_overheal: bool = False,
    ) -> HealEvent:
        """
        Heal the entity.

        Args:
            amount: Healing amount
            source_id: ID of the healing source
            timestamp: Time of the heal event
            can_overheal: If True, can exceed max_health temporarily

        Returns:
            HealEvent with details of the healing
        """
        event = HealEvent(
            amount=amount,
            source_id=source_id,
            timestamp=timestamp,
        )

        if self.is_dead:
            event.actual_healing = 0.0
            return event

        # Apply healing multiplier
        actual_healing = amount * self._healing_multiplier

        # Calculate how much can actually be healed
        if can_overheal:
            max_heal = actual_healing
        else:
            max_heal = self.max_health - self.current_health

        actual_healing = min(actual_healing, max_heal)

        # Apply healing
        if actual_healing > 0:
            self.current_health = self.current_health + actual_healing

        event.actual_healing = actual_healing

        # Record and notify
        self._heal_history.append(event)
        for callback in self._on_heal:
            callback(event)

        return event

    def regenerate(self, delta_time: float) -> float:
        """
        Apply health regeneration over time.

        Args:
            delta_time: Time elapsed since last update

        Returns:
            Amount of health regenerated
        """
        if self.regen_rate <= 0 or self.is_dead or self.is_full_health:
            return 0.0

        regen_amount = self.regen_rate * delta_time
        event = self.heal(regen_amount)
        return event.actual_healing

    # =========================================================================
    # DEATH AND REVIVAL
    # =========================================================================

    def _die(self) -> None:
        """Handle death state transition."""
        if self._state == HealthState.DEAD:
            return

        self._state = HealthState.DEAD
        self.current_health = 0.0

        for callback in self._on_death:
            callback(self)

    def revive(self, health_percentage: float = HealthConstants.DEFAULT_REVIVE_HEALTH_PERCENTAGE) -> bool:
        """
        Revive from death.

        Args:
            health_percentage: Percentage of max health to revive with (0.0 to 1.0)

        Returns:
            True if successfully revived
        """
        if not self.is_dead:
            return False

        self._state = HealthState.ALIVE
        self.current_health = self.max_health * max(HealthConstants.MIN_REVIVE_HEALTH_PERCENTAGE, min(1.0, health_percentage))

        for callback in self._on_revive:
            callback(self)

        return True

    def kill(self) -> None:
        """Instantly kill the entity."""
        self.take_damage(self.current_health + 1, DamageType.TRUE)

    # =========================================================================
    # RESISTANCE AND MODIFIERS
    # =========================================================================

    def set_resistance(self, damage_type: DamageType, value: float) -> None:
        """
        Set resistance for a damage type.

        Args:
            damage_type: Type of damage
            value: Resistance value (0.0 = no resistance, 1.0 = immune)
        """
        self._resistances[damage_type] = max(-1.0, min(1.0, value))

    def get_resistance(self, damage_type: DamageType) -> float:
        """Get resistance for a damage type."""
        return self._resistances.get(damage_type, 0.0)

    def add_resistance(self, damage_type: DamageType, value: float) -> None:
        """Add to existing resistance."""
        current = self.get_resistance(damage_type)
        self.set_resistance(damage_type, current + value)

    def clear_resistances(self) -> None:
        """Clear all resistances."""
        self._resistances.clear()

    @property
    def armor(self) -> float:
        """Get current armor value."""
        return self._armor

    @armor.setter
    def armor(self, value: float) -> None:
        """Set armor value (flat damage reduction)."""
        self._armor = max(0.0, value)

    @property
    def damage_multiplier(self) -> float:
        """Get damage multiplier."""
        return self._damage_multiplier

    @damage_multiplier.setter
    def damage_multiplier(self, value: float) -> None:
        """Set damage multiplier (1.0 = normal, 2.0 = double damage taken)."""
        self._damage_multiplier = max(0.0, value)

    @property
    def healing_multiplier(self) -> float:
        """Get healing multiplier."""
        return self._healing_multiplier

    @healing_multiplier.setter
    def healing_multiplier(self, value: float) -> None:
        """Set healing multiplier (1.0 = normal, 0.5 = half healing)."""
        self._healing_multiplier = max(0.0, value)

    # =========================================================================
    # SHIELD
    # =========================================================================

    @property
    def shield(self) -> float:
        """Get current shield value."""
        return self._shield

    @property
    def shield_max(self) -> float:
        """Get maximum shield value."""
        return self._shield_max

    def set_shield(self, current: float, maximum: Optional[float] = None) -> None:
        """Set shield values."""
        if maximum is not None:
            self._shield_max = max(0.0, maximum)
        self._shield = max(0.0, min(current, self._shield_max))

    def add_shield(self, amount: float) -> float:
        """Add to shield, capped at max. Returns actual amount added."""
        old_shield = self._shield
        self._shield = min(self._shield + amount, self._shield_max)
        return self._shield - old_shield

    # =========================================================================
    # INVULNERABILITY
    # =========================================================================

    def set_invulnerable(self, duration: float = -1.0) -> None:
        """
        Make the entity invulnerable.

        Args:
            duration: Duration in seconds (-1 = permanent)
        """
        if duration < 0:
            self._state = HealthState.INVULNERABLE
            self._invulnerable_timer = -1.0
        else:
            self._invulnerable_timer = duration

    def clear_invulnerability(self) -> None:
        """Remove invulnerability."""
        if self._state == HealthState.INVULNERABLE:
            self._state = HealthState.ALIVE
        self._invulnerable_timer = 0.0

    def update_invulnerability(self, delta_time: float) -> None:
        """Update invulnerability timer."""
        if self._invulnerable_timer > 0:
            self._invulnerable_timer -= delta_time
            if self._invulnerable_timer <= 0:
                self.clear_invulnerability()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_damage(self, callback: Callable[[DamageEvent], None]) -> None:
        """Register a callback for damage events."""
        self._on_damage.append(callback)

    def on_heal(self, callback: Callable[[HealEvent], None]) -> None:
        """Register a callback for heal events."""
        self._on_heal.append(callback)

    def on_death(self, callback: Callable[[HealthComponent], None]) -> None:
        """Register a callback for death."""
        self._on_death.append(callback)

    def on_revive(self, callback: Callable[[HealthComponent], None]) -> None:
        """Register a callback for revival."""
        self._on_revive.append(callback)

    # =========================================================================
    # HISTORY
    # =========================================================================

    def get_damage_history(self, limit: int = HealthConstants.DEFAULT_HISTORY_LIMIT) -> List[DamageEvent]:
        """Get recent damage history."""
        return self._damage_history[-limit:]

    def get_heal_history(self, limit: int = HealthConstants.DEFAULT_HISTORY_LIMIT) -> List[HealEvent]:
        """Get recent heal history."""
        return self._heal_history[-limit:]

    def clear_history(self) -> None:
        """Clear damage and heal history."""
        self._damage_history.clear()
        self._heal_history.clear()

    def get_total_damage_taken(self) -> float:
        """Get total damage taken from history."""
        return sum(e.final_damage for e in self._damage_history)

    def get_total_healing_received(self) -> float:
        """Get total healing received from history."""
        return sum(e.actual_healing for e in self._heal_history)

    # =========================================================================
    # MAX HEALTH MODIFICATION
    # =========================================================================

    def set_max_health(self, value: float, adjust_current: bool = True) -> None:
        """
        Set maximum health.

        Args:
            value: New maximum health
            adjust_current: If True, proportionally adjust current health
        """
        if value <= 0:
            raise ValueError("max_health must be greater than 0")

        if adjust_current:
            ratio = self.health_percentage
            self.max_health = value
            self.current_health = self.max_health * ratio
        else:
            self.max_health = value
            # Clamp current to new max
            if self.current_health > self.max_health:
                self.current_health = self.max_health

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize health component to dictionary."""
        return {
            "current_health": self.current_health,
            "max_health": self.max_health,
            "regen_rate": self.regen_rate,
            "state": self._state.name,
            "resistances": {k.name: v for k, v in self._resistances.items()},
            "armor": self._armor,
            "shield": self._shield,
            "shield_max": self._shield_max,
            "damage_multiplier": self._damage_multiplier,
            "healing_multiplier": self._healing_multiplier,
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HealthComponent:
        """Deserialize health component from dictionary."""
        component = cls(
            max_health=data["max_health"],
            current_health=data["current_health"],
            regen_rate=data.get("regen_rate", 0.0),
            entity_id=data.get("entity_id"),
        )

        component._state = HealthState[data.get("state", "ALIVE")]
        component._armor = data.get("armor", 0.0)
        component._shield = data.get("shield", 0.0)
        component._shield_max = data.get("shield_max", 0.0)
        component._damage_multiplier = data.get("damage_multiplier", 1.0)
        component._healing_multiplier = data.get("healing_multiplier", 1.0)

        for name, value in data.get("resistances", {}).items():
            component._resistances[DamageType[name]] = value

        return component

    def __repr__(self) -> str:
        return (
            f"HealthComponent(current={self.current_health:.1f}, "
            f"max={self.max_health:.1f}, state={self._state.name})"
        )


# Descriptor setup
HealthComponent.current_health.__set_name__(HealthComponent, "current_health")
HealthComponent.max_health.__set_name__(HealthComponent, "max_health")
HealthComponent.regen_rate.__set_name__(HealthComponent, "regen_rate")


__all__ = [
    "HealthComponent",
    "DamageType",
    "HealthState",
    "DamageEvent",
    "HealEvent",
]
