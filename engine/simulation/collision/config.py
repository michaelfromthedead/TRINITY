"""
Collision Detection Configuration Constants.

This module defines all configuration parameters for the collision detection system,
including broadphase margins, contact tolerances, CCD thresholds, and spatial
partitioning parameters.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


# =============================================================================
# Broadphase Configuration
# =============================================================================

# Margin added to AABBs during broadphase to account for motion
BROADPHASE_MARGIN: Final[float] = 0.05

# Size of cells in the spatial hash grid (world units)
SPATIAL_HASH_CELL_SIZE: Final[float] = 2.0

# Maximum depth for octree subdivision
OCTREE_MAX_DEPTH: Final[int] = 8

# Maximum objects per octree leaf before subdivision
OCTREE_MAX_OBJECTS_PER_LEAF: Final[int] = 16

# BVH rebalancing threshold (percentage of insertions/removals before rebuild)
BVH_REBALANCE_THRESHOLD: Final[float] = 0.3


# =============================================================================
# Narrowphase Configuration
# =============================================================================

# Tolerance for contact point generation
CONTACT_TOLERANCE: Final[float] = 0.01

# Maximum number of contact points per manifold
MAX_CONTACT_POINTS: Final[int] = 4

# GJK algorithm maximum iterations
GJK_MAX_ITERATIONS: Final[int] = 64

# EPA algorithm maximum iterations
EPA_MAX_ITERATIONS: Final[int] = 64

# EPA tolerance for termination
EPA_TOLERANCE: Final[float] = 0.0001

# SAT edge bias to handle parallel edges
SAT_EDGE_BIAS: Final[float] = 0.00001

# Numerical epsilon for zero-length vector checks
NUMERICAL_EPSILON: Final[float] = 1e-10

# Threshold for determining parallel/perpendicular vectors
PARALLEL_THRESHOLD: Final[float] = 0.9


# =============================================================================
# Continuous Collision Detection (CCD) Configuration
# =============================================================================

# Velocity threshold above which CCD is enabled (units/second)
CCD_THRESHOLD_VELOCITY: Final[float] = 10.0

# Maximum iterations for TOI calculation
MAX_CCD_ITERATIONS: Final[int] = 10

# CCD time step subdivision factor
CCD_TIME_STEP_FRACTION: Final[float] = 0.05

# Speculative margin for expanded AABBs
CCD_SPECULATIVE_MARGIN: Final[float] = 0.1

# Minimum time of impact considered (avoids division by zero)
CCD_MIN_TOI: Final[float] = 0.0001

# Safety factor for conservative advancement (prevents overshooting)
CCD_SAFETY_FACTOR: Final[float] = 0.9


# =============================================================================
# Contact Manifold Configuration
# =============================================================================

# Maximum age (frames) for persistent contacts before removal
CONTACT_MAX_AGE: Final[int] = 3

# Distance threshold for contact point matching during persistence
CONTACT_MATCH_THRESHOLD: Final[float] = 0.02

# Warm starting impulse retention factor
WARM_START_FACTOR: Final[float] = 0.8


# =============================================================================
# Collision Filtering Configuration
# =============================================================================

# Number of collision layers (32-bit mask)
NUM_COLLISION_LAYERS: Final[int] = 32


# =============================================================================
# Performance Tuning
# =============================================================================

# Initial capacity for collision pair lists
INITIAL_PAIR_CAPACITY: Final[int] = 1024

# SAP axis to use for initial sort (0=X, 1=Y, 2=Z)
SAP_PRIMARY_AXIS: Final[int] = 0

# Batch size for parallel narrowphase processing
NARROWPHASE_BATCH_SIZE: Final[int] = 64


# =============================================================================
# Configuration Profiles
# =============================================================================


class CollisionQuality(Enum):
    """Preset quality levels for collision detection."""

    LOW = auto()      # Fast, less accurate
    MEDIUM = auto()   # Balanced (default)
    HIGH = auto()     # Accurate, slower
    ULTRA = auto()    # Maximum accuracy


@dataclass(frozen=True)
class CollisionConfig:
    """Complete collision detection configuration."""

    # Broadphase
    broadphase_margin: float = BROADPHASE_MARGIN
    spatial_hash_cell_size: float = SPATIAL_HASH_CELL_SIZE

    # Narrowphase
    contact_tolerance: float = CONTACT_TOLERANCE
    max_contact_points: int = MAX_CONTACT_POINTS
    gjk_max_iterations: int = GJK_MAX_ITERATIONS
    epa_max_iterations: int = EPA_MAX_ITERATIONS

    # CCD
    ccd_threshold_velocity: float = CCD_THRESHOLD_VELOCITY
    max_ccd_iterations: int = MAX_CCD_ITERATIONS
    ccd_speculative_margin: float = CCD_SPECULATIVE_MARGIN

    # Contact Manifold
    contact_max_age: int = CONTACT_MAX_AGE
    warm_start_factor: float = WARM_START_FACTOR

    @classmethod
    def from_quality(cls, quality: CollisionQuality) -> "CollisionConfig":
        """Create configuration from quality preset."""
        if quality == CollisionQuality.LOW:
            return cls(
                broadphase_margin=0.1,
                contact_tolerance=0.05,
                max_contact_points=2,
                gjk_max_iterations=32,
                epa_max_iterations=32,
                max_ccd_iterations=5,
            )
        elif quality == CollisionQuality.MEDIUM:
            return cls()  # Use defaults
        elif quality == CollisionQuality.HIGH:
            return cls(
                broadphase_margin=0.02,
                contact_tolerance=0.005,
                max_contact_points=8,
                gjk_max_iterations=128,
                epa_max_iterations=128,
                max_ccd_iterations=20,
            )
        elif quality == CollisionQuality.ULTRA:
            return cls(
                broadphase_margin=0.01,
                contact_tolerance=0.001,
                max_contact_points=16,
                gjk_max_iterations=256,
                epa_max_iterations=256,
                max_ccd_iterations=50,
            )
        return cls()


# Default configuration instance
DEFAULT_CONFIG: Final[CollisionConfig] = CollisionConfig()
