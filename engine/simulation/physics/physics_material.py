"""
Physics Material Module

Defines material properties for physics interactions including
friction, restitution, and density.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum, auto
import math


class CombineMode(Enum):
    """
    Mode for combining material properties between two bodies.
    """
    AVERAGE = auto()    # (a + b) / 2
    MIN = auto()        # min(a, b)
    MAX = auto()        # max(a, b)
    MULTIPLY = auto()   # a * b
    GEOMETRIC = auto()  # sqrt(a * b)


# Default material property values
DEFAULT_STATIC_FRICTION = 0.6
DEFAULT_DYNAMIC_FRICTION = 0.4
DEFAULT_RESTITUTION = 0.0
DEFAULT_DENSITY = 1000.0  # kg/m^3 (water)

# Material property limits
MIN_FRICTION = 0.0
MAX_FRICTION = 2.0  # Can exceed 1.0 for high-grip materials
MIN_RESTITUTION = 0.0
MAX_RESTITUTION = 1.0
MIN_DENSITY = 0.001  # Near vacuum
MAX_DENSITY = 50000.0  # Very dense materials like osmium


@dataclass
class PhysicsMaterial:
    """
    Material properties for physics simulation.

    Controls how objects interact during collisions including
    friction, bounciness, and mass calculation via density.

    Attributes:
        static_friction: Friction when stationary (0-2)
        dynamic_friction: Friction when moving (0-2)
        restitution: Bounciness coefficient (0-1)
        density: Mass per unit volume in kg/m^3
        friction_combine: How to combine friction values
        restitution_combine: How to combine restitution values
        name: Optional material name for debugging
    """

    static_friction: float = DEFAULT_STATIC_FRICTION
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION
    restitution: float = DEFAULT_RESTITUTION
    density: float = DEFAULT_DENSITY
    friction_combine: CombineMode = CombineMode.AVERAGE
    restitution_combine: CombineMode = CombineMode.AVERAGE
    name: Optional[str] = None

    def __post_init__(self):
        """Validate and clamp material properties."""
        self.static_friction = max(MIN_FRICTION, min(MAX_FRICTION, self.static_friction))
        self.dynamic_friction = max(MIN_FRICTION, min(MAX_FRICTION, self.dynamic_friction))
        self.restitution = max(MIN_RESTITUTION, min(MAX_RESTITUTION, self.restitution))
        self.density = max(MIN_DENSITY, min(MAX_DENSITY, self.density))

        # Dynamic friction should not exceed static friction
        if self.dynamic_friction > self.static_friction:
            self.dynamic_friction = self.static_friction

    def set_friction(self, static: float, dynamic: Optional[float] = None) -> None:
        """
        Set friction coefficients.

        Args:
            static: Static friction coefficient
            dynamic: Dynamic friction coefficient (defaults to 0.75 * static)
        """
        self.static_friction = max(MIN_FRICTION, min(MAX_FRICTION, static))
        if dynamic is None:
            dynamic = static * 0.75
        self.dynamic_friction = max(MIN_FRICTION, min(self.static_friction, dynamic))

    def set_bounciness(self, value: float) -> None:
        """
        Set restitution (bounciness).

        Args:
            value: Restitution coefficient (0 = no bounce, 1 = perfect bounce)
        """
        self.restitution = max(MIN_RESTITUTION, min(MAX_RESTITUTION, value))

    def copy(self) -> 'PhysicsMaterial':
        """Create a copy of this material."""
        return PhysicsMaterial(
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction,
            restitution=self.restitution,
            density=self.density,
            friction_combine=self.friction_combine,
            restitution_combine=self.restitution_combine,
            name=self.name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert material to dictionary."""
        return {
            'static_friction': self.static_friction,
            'dynamic_friction': self.dynamic_friction,
            'restitution': self.restitution,
            'density': self.density,
            'friction_combine': self.friction_combine.name,
            'restitution_combine': self.restitution_combine.name,
            'name': self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsMaterial':
        """Create material from dictionary."""
        return cls(
            static_friction=data.get('static_friction', DEFAULT_STATIC_FRICTION),
            dynamic_friction=data.get('dynamic_friction', DEFAULT_DYNAMIC_FRICTION),
            restitution=data.get('restitution', DEFAULT_RESTITUTION),
            density=data.get('density', DEFAULT_DENSITY),
            friction_combine=CombineMode[data.get('friction_combine', 'AVERAGE')],
            restitution_combine=CombineMode[data.get('restitution_combine', 'AVERAGE')],
            name=data.get('name'),
        )

    def __repr__(self) -> str:
        name = f"'{self.name}'" if self.name else "unnamed"
        return (
            f"PhysicsMaterial({name}, "
            f"friction=({self.static_friction:.2f}, {self.dynamic_friction:.2f}), "
            f"restitution={self.restitution:.2f}, "
            f"density={self.density:.1f})"
        )


def combine_values(a: float, b: float, mode: CombineMode) -> float:
    """
    Combine two material values using the specified mode.

    Args:
        a: First value
        b: Second value
        mode: Combination mode

    Returns:
        Combined value
    """
    if mode == CombineMode.AVERAGE:
        return (a + b) * 0.5
    elif mode == CombineMode.MIN:
        return min(a, b)
    elif mode == CombineMode.MAX:
        return max(a, b)
    elif mode == CombineMode.MULTIPLY:
        return a * b
    elif mode == CombineMode.GEOMETRIC:
        return math.sqrt(abs(a * b))
    else:
        return (a + b) * 0.5  # Default to average


def combine_materials(
    material_a: PhysicsMaterial,
    material_b: PhysicsMaterial
) -> tuple[float, float, float]:
    """
    Combine two materials for collision response.

    Uses each material's combine mode preferences. When materials have
    different combine modes, uses the higher priority mode:
    MULTIPLY > MAX > AVERAGE > MIN

    Args:
        material_a: First material
        material_b: Second material

    Returns:
        Tuple of (static_friction, dynamic_friction, restitution)
    """
    # Priority order for combine modes
    priority = {
        CombineMode.MIN: 0,
        CombineMode.AVERAGE: 1,
        CombineMode.GEOMETRIC: 2,
        CombineMode.MAX: 3,
        CombineMode.MULTIPLY: 4,
    }

    # Select friction combine mode (use higher priority)
    if priority[material_a.friction_combine] >= priority[material_b.friction_combine]:
        friction_mode = material_a.friction_combine
    else:
        friction_mode = material_b.friction_combine

    # Select restitution combine mode (use higher priority)
    if priority[material_a.restitution_combine] >= priority[material_b.restitution_combine]:
        restitution_mode = material_a.restitution_combine
    else:
        restitution_mode = material_b.restitution_combine

    # Combine values
    static_friction = combine_values(
        material_a.static_friction,
        material_b.static_friction,
        friction_mode
    )

    dynamic_friction = combine_values(
        material_a.dynamic_friction,
        material_b.dynamic_friction,
        friction_mode
    )

    restitution = combine_values(
        material_a.restitution,
        material_b.restitution,
        restitution_mode
    )

    # Ensure constraints
    static_friction = max(MIN_FRICTION, min(MAX_FRICTION, static_friction))
    dynamic_friction = max(MIN_FRICTION, min(static_friction, dynamic_friction))
    restitution = max(MIN_RESTITUTION, min(MAX_RESTITUTION, restitution))

    return static_friction, dynamic_friction, restitution


# =============================================================================
# Preset Materials
# =============================================================================

class MaterialPresets:
    """Collection of common material presets."""

    @staticmethod
    def default() -> PhysicsMaterial:
        """Default material with moderate friction and no bounce."""
        return PhysicsMaterial(name="default")

    @staticmethod
    def rubber() -> PhysicsMaterial:
        """High friction, high bounce rubber material."""
        return PhysicsMaterial(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.8,
            density=1100.0,
            name="rubber",
        )

    @staticmethod
    def ice() -> PhysicsMaterial:
        """Very low friction ice material."""
        return PhysicsMaterial(
            static_friction=0.05,
            dynamic_friction=0.02,
            restitution=0.1,
            density=917.0,
            name="ice",
        )

    @staticmethod
    def metal() -> PhysicsMaterial:
        """Metal material with moderate friction."""
        return PhysicsMaterial(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.3,
            density=7800.0,  # Steel density
            name="metal",
        )

    @staticmethod
    def wood() -> PhysicsMaterial:
        """Wood material."""
        return PhysicsMaterial(
            static_friction=0.5,
            dynamic_friction=0.35,
            restitution=0.2,
            density=600.0,
            name="wood",
        )

    @staticmethod
    def concrete() -> PhysicsMaterial:
        """Concrete material."""
        return PhysicsMaterial(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
            density=2400.0,
            name="concrete",
        )

    @staticmethod
    def glass() -> PhysicsMaterial:
        """Glass material."""
        return PhysicsMaterial(
            static_friction=0.4,
            dynamic_friction=0.3,
            restitution=0.5,
            density=2500.0,
            name="glass",
        )

    @staticmethod
    def bouncy_ball() -> PhysicsMaterial:
        """Super bouncy ball material."""
        return PhysicsMaterial(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.95,
            density=800.0,
            name="bouncy_ball",
        )

    @staticmethod
    def frictionless() -> PhysicsMaterial:
        """Completely frictionless material."""
        return PhysicsMaterial(
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
            density=1000.0,
            name="frictionless",
        )

    @staticmethod
    def sticky() -> PhysicsMaterial:
        """Very high friction sticky material."""
        return PhysicsMaterial(
            static_friction=2.0,
            dynamic_friction=1.8,
            restitution=0.0,
            density=1200.0,
            friction_combine=CombineMode.MAX,
            name="sticky",
        )

    @staticmethod
    def sand() -> PhysicsMaterial:
        """Sand/granular material."""
        return PhysicsMaterial(
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.05,
            density=1600.0,
            name="sand",
        )

    @staticmethod
    def foam() -> PhysicsMaterial:
        """Soft foam material."""
        return PhysicsMaterial(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.1,
            density=50.0,
            name="foam",
        )

    @staticmethod
    def plastic() -> PhysicsMaterial:
        """Plastic material."""
        return PhysicsMaterial(
            static_friction=0.4,
            dynamic_friction=0.3,
            restitution=0.4,
            density=1200.0,
            name="plastic",
        )

    @staticmethod
    def leather() -> PhysicsMaterial:
        """Leather material."""
        return PhysicsMaterial(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.3,
            density=900.0,
            name="leather",
        )

    @staticmethod
    def fabric() -> PhysicsMaterial:
        """Fabric/cloth material."""
        return PhysicsMaterial(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.1,
            density=300.0,
            name="fabric",
        )


# Material lookup table
MATERIAL_PRESETS: Dict[str, PhysicsMaterial] = {
    'default': MaterialPresets.default(),
    'rubber': MaterialPresets.rubber(),
    'ice': MaterialPresets.ice(),
    'metal': MaterialPresets.metal(),
    'wood': MaterialPresets.wood(),
    'concrete': MaterialPresets.concrete(),
    'glass': MaterialPresets.glass(),
    'bouncy_ball': MaterialPresets.bouncy_ball(),
    'frictionless': MaterialPresets.frictionless(),
    'sticky': MaterialPresets.sticky(),
    'sand': MaterialPresets.sand(),
    'foam': MaterialPresets.foam(),
    'plastic': MaterialPresets.plastic(),
    'leather': MaterialPresets.leather(),
    'fabric': MaterialPresets.fabric(),
}


def get_material(name: str) -> PhysicsMaterial:
    """
    Get a preset material by name.

    Args:
        name: Material preset name

    Returns:
        Copy of the preset material

    Raises:
        KeyError: If material name not found
    """
    if name not in MATERIAL_PRESETS:
        raise KeyError(f"Unknown material preset: {name}. "
                      f"Available: {list(MATERIAL_PRESETS.keys())}")
    return MATERIAL_PRESETS[name].copy()
