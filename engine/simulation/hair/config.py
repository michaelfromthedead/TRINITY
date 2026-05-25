"""
Hair simulation configuration constants.

This module defines default parameters for hair simulation including
strand parameters, stiffness values, and LOD settings.
"""

from typing import Final

# =============================================================================
# Strand Parameters
# =============================================================================

# Default number of segments per hair strand
DEFAULT_STRAND_SEGMENTS: Final[int] = 16

# Minimum segments for LOD (at far distances)
MIN_STRAND_SEGMENTS: Final[int] = 4

# Maximum segments for high detail
MAX_STRAND_SEGMENTS: Final[int] = 32

# Default hair strand thickness (meters)
DEFAULT_HAIR_THICKNESS: Final[float] = 0.0003  # 0.3mm

# Default hair length (meters)
DEFAULT_HAIR_LENGTH: Final[float] = 0.3  # 30cm

# =============================================================================
# Simulation Parameters
# =============================================================================

# Fixed timestep for hair simulation (seconds)
HAIR_TIMESTEP: Final[float] = 1.0 / 120.0

# Number of constraint solver iterations
HAIR_SOLVER_ITERATIONS: Final[int] = 4

# Velocity damping factor
HAIR_DAMPING: Final[float] = 0.95

# Air resistance coefficient
HAIR_AIR_RESISTANCE: Final[float] = 0.1

# =============================================================================
# Stiffness Parameters
# =============================================================================

# Length constraint stiffness (inextensibility)
LENGTH_STIFFNESS: Final[float] = 1.0

# Global shape matching stiffness (tendency to return to rest pose)
SHAPE_STIFFNESS: Final[float] = 0.5

# Local shape stiffness (relative angle preservation)
LOCAL_SHAPE_STIFFNESS: Final[float] = 0.3

# Root attachment stiffness
ROOT_STIFFNESS: Final[float] = 1.0

# Collision stiffness
COLLISION_STIFFNESS: Final[float] = 0.8

# =============================================================================
# Guide Hair Parameters
# =============================================================================

# Maximum number of guide hairs
MAX_GUIDE_HAIRS: Final[int] = 1000

# Minimum guide hairs for sparse representation
MIN_GUIDE_HAIRS: Final[int] = 100

# Interpolation ratio (render hairs per guide hair)
INTERPOLATION_RATIO: Final[int] = 10

# Maximum interpolated hairs
MAX_INTERPOLATED_HAIRS: Final[int] = 10000

# =============================================================================
# Collision Parameters
# =============================================================================

# Collision margin for hair-body interaction
HAIR_COLLISION_MARGIN: Final[float] = 0.002  # 2mm

# Self-collision detection radius
SELF_COLLISION_RADIUS: Final[float] = 0.005  # 5mm

# Maximum collision iterations per frame
MAX_COLLISION_ITERATIONS: Final[int] = 4

# Density threshold for self-collision (too many hairs in same cell)
SELF_COLLISION_DENSITY_THRESHOLD: Final[float] = 2.0

# Self-collision push strength multiplier
SELF_COLLISION_PUSH_STRENGTH: Final[float] = 0.01

# =============================================================================
# Inertia and Wind Parameters
# =============================================================================

# Inertia coefficient for head motion transfer
HEAD_INERTIA_COEFFICIENT: Final[float] = 0.5

# Wind influence multiplier for hair strands
WIND_INFLUENCE_MULTIPLIER: Final[float] = 0.1

# =============================================================================
# Numerical Stability Parameters
# =============================================================================

# Epsilon for division-by-zero protection
NUMERICAL_EPSILON: Final[float] = 1e-8

# Minimum timestep for velocity calculations
MIN_VELOCITY_TIMESTEP: Final[float] = 1e-6

# Local shape constraint correction factor
LOCAL_SHAPE_CORRECTION_FACTOR: Final[float] = 0.1

# Gravity droop factor for natural hair hang
GRAVITY_DROOP_FACTOR: Final[float] = 0.3

# LOD interpolated hair offset radius (for random positioning)
LOD_INTERPOLATION_OFFSET: Final[float] = 0.005

# =============================================================================
# LOD Parameters
# =============================================================================

# Distance thresholds for LOD levels (meters)
LOD_DISTANCE_HIGH: Final[float] = 2.0
LOD_DISTANCE_MEDIUM: Final[float] = 5.0
LOD_DISTANCE_LOW: Final[float] = 10.0
LOD_DISTANCE_SHELL: Final[float] = 20.0

# Guide hair reduction factors per LOD
LOD_GUIDE_FACTOR_HIGH: Final[float] = 1.0
LOD_GUIDE_FACTOR_MEDIUM: Final[float] = 0.5
LOD_GUIDE_FACTOR_LOW: Final[float] = 0.25
LOD_GUIDE_FACTOR_SHELL: Final[float] = 0.0  # Use shell rendering

# Segment reduction per LOD
LOD_SEGMENT_FACTOR_HIGH: Final[float] = 1.0
LOD_SEGMENT_FACTOR_MEDIUM: Final[float] = 0.75
LOD_SEGMENT_FACTOR_LOW: Final[float] = 0.5

# =============================================================================
# Physics Quality Presets
# =============================================================================


class HairQualityPreset:
    """Predefined quality settings for hair simulation."""

    ULTRA = {
        "guide_hairs": 1000,
        "segments": 32,
        "interpolation_ratio": 20,
        "solver_iterations": 8,
        "self_collision": True,
        "wind_enabled": True,
    }

    HIGH = {
        "guide_hairs": 500,
        "segments": 16,
        "interpolation_ratio": 10,
        "solver_iterations": 4,
        "self_collision": True,
        "wind_enabled": True,
    }

    MEDIUM = {
        "guide_hairs": 250,
        "segments": 12,
        "interpolation_ratio": 8,
        "solver_iterations": 3,
        "self_collision": False,
        "wind_enabled": True,
    }

    LOW = {
        "guide_hairs": 100,
        "segments": 8,
        "interpolation_ratio": 4,
        "solver_iterations": 2,
        "self_collision": False,
        "wind_enabled": False,
    }

    MOBILE = {
        "guide_hairs": 50,
        "segments": 4,
        "interpolation_ratio": 2,
        "solver_iterations": 1,
        "self_collision": False,
        "wind_enabled": False,
    }


# =============================================================================
# Material Properties (for rendering hints)
# =============================================================================

# Hair color multiplier range
HAIR_COLOR_VARIATION: Final[float] = 0.1

# Hair thickness variation
HAIR_THICKNESS_VARIATION: Final[float] = 0.2

# Hair curl frequency range
HAIR_CURL_FREQUENCY_MIN: Final[float] = 0.0
HAIR_CURL_FREQUENCY_MAX: Final[float] = 10.0

# Hair clumping factor range
HAIR_CLUMPING_MIN: Final[float] = 0.0
HAIR_CLUMPING_MAX: Final[float] = 1.0
