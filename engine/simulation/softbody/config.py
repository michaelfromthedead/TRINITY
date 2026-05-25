"""Soft body simulation configuration constants.

This module defines default parameters for soft body physics simulation
including material properties, solver settings, and constraints.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Material Properties
# =============================================================================

# Young's Modulus (stiffness) in Pascals
# Typical values: rubber ~1e6, muscle ~1e4, fat ~1e3
DEFAULT_YOUNG_MODULUS: float = 10000.0

# Poisson's Ratio (incompressibility)
# 0.0 = fully compressible, 0.5 = incompressible
# Typical values: rubber ~0.49, muscle ~0.4, fat ~0.45
DEFAULT_POISSON_RATIO: float = 0.3

# Volume preservation stiffness for PBD
VOLUME_STIFFNESS: float = 1.0

# Shape matching stiffness (0 = no shape preservation, 1 = rigid)
SHAPE_MATCHING_STIFFNESS: float = 0.5

# Maximum allowed deformation ratio (1.3 = 30% stretch/compression)
MAX_DEFORMATION: float = 0.3

# Number of substeps for soft body simulation per physics step
SOFTBODY_SUBSTEPS: int = 4


# =============================================================================
# Solver Configuration
# =============================================================================

# PBD solver iterations per substep
PBD_ITERATIONS: int = 4

# FEM solver tolerance for convergence
FEM_TOLERANCE: float = 1e-6

# FEM maximum iterations
FEM_MAX_ITERATIONS: int = 20

# Damping coefficient for velocity damping
DEFAULT_DAMPING: float = 0.99

# Minimum tetrahedron volume before degeneracy handling
MIN_TET_VOLUME: float = 1e-8

# Collision margin for soft body collision detection
COLLISION_MARGIN: float = 0.01

# FEM Singularity Handling
FEM_MIN_JACOBIAN: float = 0.001  # Minimum Jacobian determinant to prevent singularity
FEM_INVERSION_HANDLING: str = "clamp"  # "clamp", "reflect", or "penalty"

# Shape Matching Singularity Handling
SVD_MIN_SINGULAR_VALUE: float = 1e-6  # Minimum singular value to prevent degeneracy
SVD_REGULARIZATION: float = 1e-8  # Regularization for Aqq matrix inversion

# Boundary Damping (was hardcoded in sph.py)
BOUNDARY_VELOCITY_DAMPING: float = 0.3

# Muscle Simulation Constants
MUSCLE_FORCE_LENGTH_WIDTH: float = 0.45  # Width parameter for force-length Gaussian
MUSCLE_ECCENTRIC_FORCE_MAX: float = 1.8  # Maximum eccentric force multiplier
MUSCLE_CONCENTRIC_THRESHOLD: float = 0.25  # Velocity threshold for concentric force
MUSCLE_VOLUME_STIFFNESS: float = 10.0  # Stiffness for volume preservation


# =============================================================================
# Material Presets
# =============================================================================

class MaterialPreset(Enum):
    """Predefined material types with realistic parameters."""
    RUBBER = auto()
    MUSCLE = auto()
    FAT = auto()
    JELLY = auto()
    SKIN = auto()
    FOAM = auto()
    CLAY = auto()


@dataclass
class SoftBodyMaterial:
    """Material properties for soft body simulation.

    Attributes:
        young_modulus: Stiffness in Pascals
        poisson_ratio: Incompressibility (0-0.5)
        density: Mass per unit volume in kg/m^3
        damping: Velocity damping factor
        plasticity: Permanent deformation threshold (0 = elastic)
        max_stretch: Maximum allowed stretch ratio
        max_compress: Maximum allowed compression ratio
    """
    young_modulus: float = DEFAULT_YOUNG_MODULUS
    poisson_ratio: float = DEFAULT_POISSON_RATIO
    density: float = 1000.0
    damping: float = DEFAULT_DAMPING
    plasticity: float = 0.0
    max_stretch: float = 1.0 + MAX_DEFORMATION
    max_compress: float = 1.0 - MAX_DEFORMATION * 0.5

    @classmethod
    def from_preset(cls, preset: MaterialPreset) -> "SoftBodyMaterial":
        """Create material from preset."""
        presets = {
            MaterialPreset.RUBBER: cls(
                young_modulus=1e6,
                poisson_ratio=0.49,
                density=1100.0,
                damping=0.98,
                max_stretch=2.0,
            ),
            MaterialPreset.MUSCLE: cls(
                young_modulus=1e4,
                poisson_ratio=0.4,
                density=1060.0,
                damping=0.95,
            ),
            MaterialPreset.FAT: cls(
                young_modulus=1e3,
                poisson_ratio=0.45,
                density=920.0,
                damping=0.9,
            ),
            MaterialPreset.JELLY: cls(
                young_modulus=500.0,
                poisson_ratio=0.48,
                density=1020.0,
                damping=0.85,
                max_stretch=1.5,
            ),
            MaterialPreset.SKIN: cls(
                young_modulus=5e5,
                poisson_ratio=0.35,
                density=1100.0,
                damping=0.97,
            ),
            MaterialPreset.FOAM: cls(
                young_modulus=100.0,
                poisson_ratio=0.3,
                density=50.0,
                damping=0.8,
                max_compress=0.3,
            ),
            MaterialPreset.CLAY: cls(
                young_modulus=2e4,
                poisson_ratio=0.35,
                density=1800.0,
                damping=0.7,
                plasticity=0.1,
            ),
        }
        return presets.get(preset, cls())

    def compute_lame_parameters(self) -> tuple[float, float]:
        """Compute Lame parameters (lambda, mu) from Young's modulus and Poisson's ratio.

        Returns:
            Tuple of (lambda, mu) where:
            - lambda: First Lame parameter
            - mu: Second Lame parameter (shear modulus)
        """
        E = self.young_modulus
        nu = self.poisson_ratio

        # Clamp Poisson's ratio to avoid division by zero
        nu = min(nu, 0.4999)

        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return lam, mu


# =============================================================================
# Solver Types
# =============================================================================

class SoftBodySolverType(Enum):
    """Available soft body solver types."""
    FEM = auto()           # Finite Element Method (accurate)
    COROTATIONAL = auto()  # Corotational FEM (handles large rotations)
    SHAPE_MATCHING = auto() # Shape matching (fast, geometric)
    PBD = auto()           # Position Based Dynamics (stable, interactive)
    XPBD = auto()          # Extended PBD (physically accurate)


@dataclass
class SolverConfig:
    """Configuration for soft body solvers.

    Attributes:
        solver_type: Type of solver to use
        substeps: Number of substeps per physics step
        iterations: Solver iterations per substep
        tolerance: Convergence tolerance
        damping: Global damping factor
        gravity: Enable gravity
    """
    solver_type: SoftBodySolverType = SoftBodySolverType.PBD
    substeps: int = SOFTBODY_SUBSTEPS
    iterations: int = PBD_ITERATIONS
    tolerance: float = FEM_TOLERANCE
    damping: float = DEFAULT_DAMPING
    gravity: bool = True


# =============================================================================
# Constraint Configuration
# =============================================================================

@dataclass
class ConstraintConfig:
    """Configuration for soft body constraints.

    Attributes:
        volume_stiffness: Volume preservation strength
        shape_stiffness: Shape matching strength
        edge_stiffness: Edge length constraint strength
        collision_stiffness: Collision response strength
        friction: Collision friction coefficient
    """
    volume_stiffness: float = VOLUME_STIFFNESS
    shape_stiffness: float = SHAPE_MATCHING_STIFFNESS
    edge_stiffness: float = 1.0
    collision_stiffness: float = 1.0
    friction: float = 0.5


# =============================================================================
# Collision Configuration
# =============================================================================

@dataclass
class CollisionConfig:
    """Configuration for soft body collision detection.

    Attributes:
        enabled: Enable collision detection
        self_collision: Enable self-collision
        margin: Collision margin distance
        friction: Friction coefficient
        restitution: Bounce coefficient
        iterations: Collision solving iterations
    """
    enabled: bool = True
    self_collision: bool = False
    margin: float = COLLISION_MARGIN
    friction: float = 0.5
    restitution: float = 0.0
    iterations: int = 2
