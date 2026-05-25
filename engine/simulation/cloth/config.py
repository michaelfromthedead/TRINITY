"""
Cloth simulation configuration constants.

This module defines default parameters for Position-Based Dynamics (PBD) cloth simulation,
including stiffness coefficients, timestep parameters, and collision settings.
"""

from typing import Final

# =============================================================================
# Stiffness Constants
# =============================================================================

# Stretch stiffness controls resistance to edge length changes (0-1)
# Higher values make cloth more resistant to stretching
DEFAULT_STRETCH_STIFFNESS: Final[float] = 1.0

# Bend stiffness controls resistance to folding/bending (0-1)
# Lower values create more flowing, drapey cloth
DEFAULT_BEND_STIFFNESS: Final[float] = 0.1

# Shear stiffness controls resistance to diagonal distortion (0-1)
# Higher values prevent parallelogram-like deformation
DEFAULT_SHEAR_STIFFNESS: Final[float] = 0.5

# =============================================================================
# Simulation Parameters
# =============================================================================

# Fixed timestep for cloth simulation (seconds)
# 1/120 = 120 Hz simulation rate for stable behavior
CLOTH_TIMESTEP: Final[float] = 1.0 / 120.0

# Number of constraint solver substeps per simulation step
# More substeps = more stable but slower
CLOTH_SUBSTEPS: Final[int] = 4

# Number of constraint solver iterations per substep
CLOTH_SOLVER_ITERATIONS: Final[int] = 4

# Velocity damping factor (0-1)
# Higher values reduce oscillation but may look unnatural
CLOTH_DAMPING: Final[float] = 0.99

# =============================================================================
# Collision Parameters
# =============================================================================

# Thickness offset for self-collision detection (meters)
# Prevents interpenetration of cloth with itself
SELF_COLLISION_THICKNESS: Final[float] = 0.02

# Minimum distance between cloth and colliders (meters)
COLLISION_MARGIN: Final[float] = 0.01

# Friction coefficient for cloth-collider interaction (0-1)
COLLISION_FRICTION: Final[float] = 0.3

# Restitution (bounciness) for collision response (0-1)
COLLISION_RESTITUTION: Final[float] = 0.0

# =============================================================================
# Particle Limits
# =============================================================================

# Maximum number of particles in a single cloth mesh
MAX_CLOTH_PARTICLES: Final[int] = 10000

# Maximum number of edges (distance constraints)
MAX_CLOTH_EDGES: Final[int] = 30000

# Maximum number of triangles
MAX_CLOTH_TRIANGLES: Final[int] = 20000

# Maximum number of cloth objects in a scene
MAX_CLOTH_OBJECTS: Final[int] = 100

# =============================================================================
# Spatial Hashing Parameters (for self-collision)
# =============================================================================

# Cell size for spatial hashing (relative to self-collision thickness)
SPATIAL_HASH_CELL_SIZE: Final[float] = 0.05

# Initial hash table size
SPATIAL_HASH_TABLE_SIZE: Final[int] = 8192

# Maximum neighbors to check per particle
MAX_COLLISION_NEIGHBORS: Final[int] = 32

# =============================================================================
# Long-Range Attachment Parameters
# =============================================================================

# Maximum distance ratio before long-range attachments activate
LONG_RANGE_MAX_RATIO: Final[float] = 1.5

# Stiffness of long-range attachments
LONG_RANGE_STIFFNESS: Final[float] = 0.8

# =============================================================================
# Numerical Stability Parameters
# =============================================================================

# Epsilon for division-by-zero protection and numerical stability
NUMERICAL_EPSILON: Final[float] = 1e-8

# Minimum timestep for velocity calculations to prevent instability
MIN_VELOCITY_TIMESTEP: Final[float] = 1e-6

# Self-collision correction factor (0.5 = half correction per iteration)
SELF_COLLISION_CORRECTION_FACTOR: Final[float] = 0.5

# Bending constraint correction factor
BENDING_CORRECTION_FACTOR: Final[float] = 0.25

# =============================================================================
# Wind Parameters
# =============================================================================

# Wind drag coefficient for cloth triangles
WIND_DRAG_COEFFICIENT: Final[float] = 0.5

# Wind lift coefficient for cloth triangles
WIND_LIFT_COEFFICIENT: Final[float] = 0.2

# Wind turbulence strength
WIND_TURBULENCE_STRENGTH: Final[float] = 0.3

# Wind turbulence frequency
WIND_TURBULENCE_FREQUENCY: Final[float] = 2.0

# Wind turbulence octaves
WIND_TURBULENCE_OCTAVES: Final[int] = 3

# =============================================================================
# Quality/Performance Presets
# =============================================================================


class ClothQualityPreset:
    """Predefined quality settings for different use cases."""

    HIGH = {
        "substeps": 8,
        "solver_iterations": 8,
        "self_collision": True,
        "max_particles": 10000,
    }

    MEDIUM = {
        "substeps": 4,
        "solver_iterations": 4,
        "self_collision": True,
        "max_particles": 5000,
    }

    LOW = {
        "substeps": 2,
        "solver_iterations": 2,
        "self_collision": False,
        "max_particles": 2000,
    }

    MOBILE = {
        "substeps": 1,
        "solver_iterations": 2,
        "self_collision": False,
        "max_particles": 1000,
    }
