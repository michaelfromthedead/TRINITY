"""
Damage Types and Resistance System.

Defines the damage type enumeration, damage data structures, and resistance
calculations for the destruction system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Dict, Optional, Tuple

import math


class DamageType(IntEnum):
    """
    Enumeration of damage types supported by the destruction system.

    Each damage type can have different effects on materials and may be
    resisted differently by various destructible objects.
    """
    IMPACT = 0
    """Physical impact damage (collisions, projectiles)."""

    EXPLOSIVE = auto()
    """Explosive damage with radial falloff."""

    STRESS = auto()
    """Structural stress damage (weight, tension)."""

    BURN = auto()
    """Fire/heat damage over time."""

    PIERCE = auto()
    """Penetrating damage (narrow, focused)."""

    SLASH = auto()
    """Cutting damage (blades, edges)."""

    CRUSH = auto()
    """Crushing damage (compression, weight)."""

    ELECTRIC = auto()
    """Electrical damage."""

    CORROSIVE = auto()
    """Chemical/acid damage over time."""

    FREEZE = auto()
    """Cold damage that may cause brittleness."""


@dataclass(slots=True)
class Damage:
    """
    Represents a single damage instance applied to a destructible object.

    Attributes:
        amount: Raw damage value before modifiers.
        damage_type: Type of damage being applied.
        position: World-space position where damage is applied.
        direction: Direction of damage application (normalized).
        source_id: Optional identifier of damage source entity.
        impulse: Optional physics impulse magnitude.
        radius: For area damage, the effect radius (0 for point damage).
        falloff: Falloff function for area damage.
        timestamp: Simulation time when damage was created.
    """
    amount: float
    damage_type: DamageType
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    source_id: Optional[int] = None
    impulse: float = 0.0
    radius: float = 0.0
    falloff: str = "linear"  # "linear", "quadratic", "none"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Damage amount cannot be negative")
        if self.radius < 0:
            raise ValueError("Damage radius cannot be negative")
        # Normalize direction
        dx, dy, dz = self.direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            self.direction = (dx/length, dy/length, dz/length)

    def calculate_falloff(self, distance: float) -> float:
        """
        Calculate damage falloff at a given distance from the damage center.

        Args:
            distance: Distance from damage position.

        Returns:
            Falloff multiplier (0.0 to 1.0).
        """
        if self.radius <= 0:
            return 1.0 if distance <= 0 else 0.0

        if distance >= self.radius:
            return 0.0

        normalized = distance / self.radius

        if self.falloff == "none":
            return 1.0
        elif self.falloff == "quadratic":
            return 1.0 - (normalized * normalized)
        else:  # linear
            return 1.0 - normalized

    def with_falloff(self, distance: float) -> Damage:
        """
        Create a new Damage instance with amount modified by falloff.

        Args:
            distance: Distance from damage center.

        Returns:
            New Damage instance with modified amount.
        """
        multiplier = self.calculate_falloff(distance)
        return Damage(
            amount=self.amount * multiplier,
            damage_type=self.damage_type,
            position=self.position,
            direction=self.direction,
            source_id=self.source_id,
            impulse=self.impulse * multiplier,
            radius=0.0,  # Point damage after falloff
            falloff="none",
            timestamp=self.timestamp
        )


@dataclass(slots=True)
class DamageResistance:
    """
    Damage resistance configuration for a destructible object.

    Resistances are stored as multipliers where:
    - 1.0 = normal damage
    - 0.5 = 50% damage reduction (50% resistance)
    - 2.0 = 100% extra damage (weakness)
    - 0.0 = immune

    Attributes:
        resistances: Mapping of damage type to resistance multiplier.
        default_resistance: Resistance for unlisted damage types.
    """
    resistances: Dict[DamageType, float] = field(default_factory=dict)
    default_resistance: float = 1.0

    def __post_init__(self) -> None:
        # Validate resistance values
        for dtype, value in self.resistances.items():
            if value < 0:
                raise ValueError(f"Resistance for {dtype.name} cannot be negative")

    def get_resistance(self, damage_type: DamageType) -> float:
        """
        Get the resistance multiplier for a damage type.

        Args:
            damage_type: The type of damage.

        Returns:
            Resistance multiplier (lower = more resistant).
        """
        return self.resistances.get(damage_type, self.default_resistance)

    def set_resistance(self, damage_type: DamageType, value: float) -> None:
        """
        Set the resistance multiplier for a damage type.

        Args:
            damage_type: The type of damage.
            value: Resistance multiplier (0.0 = immune, 1.0 = normal).
        """
        if value < 0:
            raise ValueError("Resistance value cannot be negative")
        self.resistances[damage_type] = value

    def apply(self, damage: Damage) -> float:
        """
        Apply resistance to a damage instance and return modified amount.

        Args:
            damage: The damage to modify.

        Returns:
            Modified damage amount after resistance.
        """
        multiplier = self.get_resistance(damage.damage_type)
        return damage.amount * multiplier

    def is_immune(self, damage_type: DamageType) -> bool:
        """Check if completely immune to a damage type."""
        return self.get_resistance(damage_type) == 0.0

    def is_vulnerable(self, damage_type: DamageType) -> bool:
        """Check if vulnerable (takes extra damage) to a damage type."""
        return self.get_resistance(damage_type) > 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, float], default: float = 1.0) -> DamageResistance:
        """
        Create DamageResistance from string-keyed dictionary.

        Args:
            data: Dictionary mapping damage type names to multipliers.
            default: Default resistance for unlisted types.

        Returns:
            New DamageResistance instance.
        """
        resistances = {}
        for name, value in data.items():
            try:
                dtype = DamageType[name.upper()]
                resistances[dtype] = value
            except KeyError:
                raise ValueError(f"Unknown damage type: {name}")
        return cls(resistances=resistances, default_resistance=default)


# =============================================================================
# DAMAGE TYPE PROPERTIES
# =============================================================================

@dataclass(frozen=True, slots=True)
class DamageTypeProperties:
    """
    Properties and behavior configuration for a damage type.

    Attributes:
        base_multiplier: Base damage multiplier for this type.
        propagates: Whether damage propagates to neighbors.
        propagation_factor: Propagation strength (0.0-1.0).
        causes_fracture: Whether this damage type can cause fracturing.
        fracture_threshold: Damage threshold for fracture.
        is_area: Whether damage is naturally area-based.
        default_radius: Default radius for area damage.
        is_dot: Whether damage is damage-over-time.
        dot_duration: Duration for DoT damage.
        dot_tick_rate: Tick rate for DoT damage.
    """
    base_multiplier: float = 1.0
    propagates: bool = True
    propagation_factor: float = 0.5
    causes_fracture: bool = True
    fracture_threshold: float = 50.0
    is_area: bool = False
    default_radius: float = 0.0
    is_dot: bool = False
    dot_duration: float = 0.0
    dot_tick_rate: float = 1.0


# Default properties for each damage type
DAMAGE_TYPE_PROPERTIES: Dict[DamageType, DamageTypeProperties] = {
    DamageType.IMPACT: DamageTypeProperties(
        base_multiplier=1.0,
        propagates=True,
        propagation_factor=0.5,
        causes_fracture=True,
        fracture_threshold=50.0
    ),
    DamageType.EXPLOSIVE: DamageTypeProperties(
        base_multiplier=1.5,
        propagates=True,
        propagation_factor=0.8,
        causes_fracture=True,
        fracture_threshold=30.0,
        is_area=True,
        default_radius=5.0
    ),
    DamageType.STRESS: DamageTypeProperties(
        base_multiplier=0.5,
        propagates=True,
        propagation_factor=0.9,
        causes_fracture=True,
        fracture_threshold=100.0
    ),
    DamageType.BURN: DamageTypeProperties(
        base_multiplier=0.8,
        propagates=False,
        causes_fracture=False,
        is_dot=True,
        dot_duration=5.0,
        dot_tick_rate=0.5
    ),
    DamageType.PIERCE: DamageTypeProperties(
        base_multiplier=1.2,
        propagates=False,
        causes_fracture=True,
        fracture_threshold=80.0
    ),
    DamageType.SLASH: DamageTypeProperties(
        base_multiplier=1.0,
        propagates=False,
        causes_fracture=True,
        fracture_threshold=60.0
    ),
    DamageType.CRUSH: DamageTypeProperties(
        base_multiplier=1.3,
        propagates=True,
        propagation_factor=0.6,
        causes_fracture=True,
        fracture_threshold=40.0
    ),
    DamageType.ELECTRIC: DamageTypeProperties(
        base_multiplier=1.0,
        propagates=True,
        propagation_factor=0.7,
        causes_fracture=False
    ),
    DamageType.CORROSIVE: DamageTypeProperties(
        base_multiplier=0.6,
        propagates=False,
        causes_fracture=False,
        is_dot=True,
        dot_duration=10.0,
        dot_tick_rate=1.0
    ),
    DamageType.FREEZE: DamageTypeProperties(
        base_multiplier=0.7,
        propagates=False,
        causes_fracture=True,  # Brittle fracture
        fracture_threshold=25.0  # Lower threshold when frozen
    ),
}


def get_damage_type_properties(damage_type: DamageType) -> DamageTypeProperties:
    """
    Get the properties for a damage type.

    Args:
        damage_type: The damage type to query.

    Returns:
        DamageTypeProperties for the type.
    """
    return DAMAGE_TYPE_PROPERTIES.get(damage_type, DamageTypeProperties())


def apply_damage_modifiers(
    damage: Damage,
    resistance: DamageResistance,
    additional_modifiers: Optional[Dict[str, float]] = None
) -> float:
    """
    Apply all damage modifiers to calculate final damage amount.

    Args:
        damage: The base damage instance.
        resistance: Target's damage resistance.
        additional_modifiers: Optional extra multipliers (e.g., buffs/debuffs).

    Returns:
        Final calculated damage amount.
    """
    # Get base properties for this damage type
    properties = get_damage_type_properties(damage.damage_type)

    # Start with base amount
    final_amount = damage.amount

    # Apply type base multiplier
    final_amount *= properties.base_multiplier

    # Apply resistance
    final_amount = resistance.apply(
        Damage(
            amount=final_amount,
            damage_type=damage.damage_type,
            position=damage.position,
            direction=damage.direction
        )
    )

    # Apply additional modifiers
    if additional_modifiers:
        for modifier in additional_modifiers.values():
            final_amount *= modifier

    return max(0.0, final_amount)


@dataclass(slots=True)
class DamageResult:
    """
    Result of a damage calculation.

    Attributes:
        original_amount: Original damage before modifiers.
        final_amount: Final damage after all modifiers.
        damage_type: Type of damage applied.
        was_resisted: Whether any resistance was applied.
        was_lethal: Whether damage was enough to destroy target.
        caused_fracture: Whether damage triggered fracturing.
        propagated_amount: Amount of damage that propagated to neighbors.
    """
    original_amount: float
    final_amount: float
    damage_type: DamageType
    was_resisted: bool = False
    was_lethal: bool = False
    caused_fracture: bool = False
    propagated_amount: float = 0.0


class DamageAccumulator:
    """
    Accumulates damage over time for a destructible object.

    Supports damage accumulation, decay, and threshold tracking.
    """

    __slots__ = (
        '_total_damage', '_damage_by_type', '_decay_rate',
        '_last_update', '_threshold', '_max_damage'
    )

    def __init__(
        self,
        threshold: float = 100.0,
        decay_rate: float = 0.0,
        max_damage: float = float('inf')
    ) -> None:
        """
        Initialize damage accumulator.

        Args:
            threshold: Damage threshold for destruction.
            decay_rate: Rate of damage decay per second.
            max_damage: Maximum damage cap.
        """
        self._total_damage: float = 0.0
        self._damage_by_type: Dict[DamageType, float] = {}
        self._decay_rate = decay_rate
        self._last_update: float = 0.0
        self._threshold = threshold
        self._max_damage = max_damage

    @property
    def total_damage(self) -> float:
        """Total accumulated damage."""
        return self._total_damage

    @property
    def threshold(self) -> float:
        """Destruction threshold."""
        return self._threshold

    @property
    def remaining_health(self) -> float:
        """Remaining health before destruction."""
        return max(0.0, self._threshold - self._total_damage)

    @property
    def health_percent(self) -> float:
        """Health as a percentage (0.0-1.0)."""
        if self._threshold <= 0:
            return 0.0
        return max(0.0, 1.0 - (self._total_damage / self._threshold))

    @property
    def is_destroyed(self) -> bool:
        """Whether accumulated damage exceeds threshold."""
        return self._total_damage >= self._threshold

    def accumulate(self, damage: float, damage_type: DamageType) -> float:
        """
        Add damage to the accumulator.

        Args:
            damage: Amount of damage to add.
            damage_type: Type of damage.

        Returns:
            New total damage.
        """
        if damage <= 0:
            return self._total_damage

        self._total_damage = min(self._max_damage, self._total_damage + damage)
        self._damage_by_type[damage_type] = (
            self._damage_by_type.get(damage_type, 0.0) + damage
        )

        return self._total_damage

    def update(self, current_time: float) -> None:
        """
        Update accumulator with time-based decay.

        Args:
            current_time: Current simulation time.
        """
        if self._decay_rate <= 0:
            return

        if self._last_update > 0:
            dt = current_time - self._last_update
            decay = self._decay_rate * dt
            self._total_damage = max(0.0, self._total_damage - decay)

        self._last_update = current_time

    def get_damage_by_type(self, damage_type: DamageType) -> float:
        """Get accumulated damage for a specific type."""
        return self._damage_by_type.get(damage_type, 0.0)

    def get_dominant_damage_type(self) -> Optional[DamageType]:
        """Get the damage type with highest accumulated damage."""
        if not self._damage_by_type:
            return None
        return max(self._damage_by_type.keys(), key=lambda k: self._damage_by_type[k])

    def reset(self) -> None:
        """Reset all accumulated damage."""
        self._total_damage = 0.0
        self._damage_by_type.clear()
        self._last_update = 0.0

    def to_dict(self) -> Dict:
        """Serialize accumulator state."""
        return {
            'total_damage': self._total_damage,
            'damage_by_type': {
                dt.name: amount for dt, amount in self._damage_by_type.items()
            },
            'threshold': self._threshold,
            'decay_rate': self._decay_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> DamageAccumulator:
        """Deserialize accumulator state."""
        accumulator = cls(
            threshold=data.get('threshold', 100.0),
            decay_rate=data.get('decay_rate', 0.0)
        )
        accumulator._total_damage = data.get('total_damage', 0.0)
        for name, amount in data.get('damage_by_type', {}).items():
            dtype = DamageType[name]
            accumulator._damage_by_type[dtype] = amount
        return accumulator
