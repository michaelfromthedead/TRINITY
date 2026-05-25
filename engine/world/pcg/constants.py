"""
PCG System Constants.

Centralizes all magic numbers and default values for the PCG system.
This improves maintainability and allows easy tuning of PCG parameters.
"""

from __future__ import annotations

# ============================================================================
# NOISE GENERATION CONSTANTS
# ============================================================================

# Default noise settings
DEFAULT_NOISE_FREQUENCY = 1.0
DEFAULT_NOISE_OCTAVES = 4
DEFAULT_NOISE_LACUNARITY = 2.0
DEFAULT_NOISE_PERSISTENCE = 0.5
DEFAULT_NOISE_AMPLITUDE = 1.0

# Noise algorithm constants
PERLIN_PERMUTATION_SIZE = 256

# Simplex noise skewing factors (mathematical constants)
SIMPLEX_F2 = 0.3660254037844386  # 0.5 * (sqrt(3) - 1)
SIMPLEX_G2 = 0.21132486540518713  # (3 - sqrt(3)) / 6
SIMPLEX_F3 = 0.3333333333333333  # 1/3
SIMPLEX_G3 = 0.16666666666666666  # 1/6

# Simplex noise output scaling factors
SIMPLEX_2D_SCALE = 70.0  # Scales 2D simplex to approximately [-1, 1]
SIMPLEX_3D_SCALE = 32.0  # Scales 3D simplex to approximately [-1, 1]

# Worley noise normalization factors
WORLEY_F1_SCALE = 2.0  # Scale for F1 return type
WORLEY_F2_F1_SCALE = 4.0  # Scale for F2-F1 return type

# Value noise table size
VALUE_TABLE_SIZE = 256
VALUE_TABLE_PRIME = 7919  # Prime multiplier for mixing

# White noise precision multiplier
WHITE_NOISE_PRECISION = 1000000  # Convert float to int with 6 decimal places

# 3D sample perturbation factors (for 2D noise extended to 3D)
SAMPLE_3D_PERTURBATION_X = 0.7654321
SAMPLE_3D_PERTURBATION_Y = 0.5678

# ============================================================================
# LCG (Linear Congruential Generator) CONSTANTS
# ============================================================================

# Standard LCG parameters (Numerical Recipes / glibc)
LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 0x7FFFFFFF  # 2^31 - 1

# MINSTD LCG parameters (used by RandomStream)
MINSTD_MULTIPLIER = 48271
MINSTD_MODULUS = 2147483647  # 2^31 - 1

# FNV-1a hash parameters
FNV_PRIME = 0x01000193
FNV_OFFSET_BASIS = 0x9e3779b9  # Golden ratio hash

# ============================================================================
# SCATTER PLACEMENT CONSTANTS
# ============================================================================

# Default scatter settings
DEFAULT_SCATTER_DENSITY = 1.0
DEFAULT_MIN_SPACING = 1.0
DEFAULT_JITTER = 0.0
DEFAULT_CLUSTER_SIZE = 5
DEFAULT_CLUSTER_RADIUS = 10.0

# Poisson disk sampling
POISSON_MAX_ATTEMPTS = 30  # Default attempts before giving up on point
POISSON_CELL_FACTOR = 1.4142135623730951  # sqrt(2), for cell size calculation
POISSON_NEIGHBORHOOD_RADIUS = 2  # Grid cells to check in each direction

# Jitter limits
MAX_JITTER = 0.5  # Maximum jitter as fraction of spacing

# ============================================================================
# FILTER DEFAULT RANGES
# ============================================================================

# Slope filter defaults (degrees)
DEFAULT_MIN_SLOPE = 0.0
DEFAULT_MAX_SLOPE = 90.0

# Height filter defaults (world units)
DEFAULT_MIN_HEIGHT = -1000.0
DEFAULT_MAX_HEIGHT = 1000.0

# Noise filter defaults
DEFAULT_NOISE_THRESHOLD = 0.5

# Exclusion zone defaults
DEFAULT_EXCLUSION_RADIUS = 10.0

# ============================================================================
# TRANSFORM RULE DEFAULTS
# ============================================================================

# Scale range
DEFAULT_SCALE_MIN = 0.8
DEFAULT_SCALE_MAX = 1.2

# Rotation range (degrees)
DEFAULT_ROTATION_MIN = 0.0
DEFAULT_ROTATION_MAX = 360.0

# Offset range
DEFAULT_OFFSET_MIN = 0.0
DEFAULT_OFFSET_MAX = 0.0

# ============================================================================
# SEED MANAGEMENT CONSTANTS
# ============================================================================

# Seed offsets for different purposes
CHUNK_SEED_OFFSET = 0x12345678
LAYER_SEED_OFFSET = 0x87654321

# Maximum seed value (31-bit positive integer)
MAX_SEED_VALUE = 0x7FFFFFFF

# Circle/sphere sampling
MAX_CIRCLE_REJECTION_ATTEMPTS = 100  # Max rejection sampling attempts

# Gaussian sampling
GAUSSIAN_EPSILON = 1e-10  # Minimum value to avoid log(0)

# ============================================================================
# NOISE MAP CONSTANTS
# ============================================================================

# Normalization
NORMALIZATION_EPSILON = 1e-10  # Minimum range to avoid division by zero

# Bounds clamping (for get_value interpolation)
BOUNDS_EPSILON = 1.001  # Small offset to prevent boundary issues

# ============================================================================
# VALIDATION LIMITS
# ============================================================================

# These define hard limits for validation
MIN_FREQUENCY = 0.0  # Exclusive
MIN_OCTAVES = 1
MIN_LACUNARITY = 0.0  # Exclusive
MIN_PERSISTENCE = 0.0  # Exclusive
MAX_PERSISTENCE = 1.0  # Inclusive
MIN_AMPLITUDE = 0.0  # Exclusive
MIN_DENSITY = 0.0  # Exclusive
MIN_SPACING = 0.0  # Exclusive
MIN_CLUSTER_SIZE = 1
MIN_CLUSTER_RADIUS = 0.0  # Exclusive
MIN_RADIUS = 0.0  # Exclusive
MIN_SCALE = 0.0  # Exclusive

__all__ = [
    # Noise constants
    "DEFAULT_NOISE_FREQUENCY",
    "DEFAULT_NOISE_OCTAVES",
    "DEFAULT_NOISE_LACUNARITY",
    "DEFAULT_NOISE_PERSISTENCE",
    "DEFAULT_NOISE_AMPLITUDE",
    "PERLIN_PERMUTATION_SIZE",
    "SIMPLEX_F2",
    "SIMPLEX_G2",
    "SIMPLEX_F3",
    "SIMPLEX_G3",
    "SIMPLEX_2D_SCALE",
    "SIMPLEX_3D_SCALE",
    "WORLEY_F1_SCALE",
    "WORLEY_F2_F1_SCALE",
    "VALUE_TABLE_SIZE",
    "VALUE_TABLE_PRIME",
    "WHITE_NOISE_PRECISION",
    "SAMPLE_3D_PERTURBATION_X",
    "SAMPLE_3D_PERTURBATION_Y",
    # LCG constants
    "LCG_MULTIPLIER",
    "LCG_INCREMENT",
    "LCG_MODULUS",
    "MINSTD_MULTIPLIER",
    "MINSTD_MODULUS",
    "FNV_PRIME",
    "FNV_OFFSET_BASIS",
    # Scatter constants
    "DEFAULT_SCATTER_DENSITY",
    "DEFAULT_MIN_SPACING",
    "DEFAULT_JITTER",
    "DEFAULT_CLUSTER_SIZE",
    "DEFAULT_CLUSTER_RADIUS",
    "POISSON_MAX_ATTEMPTS",
    "POISSON_CELL_FACTOR",
    "POISSON_NEIGHBORHOOD_RADIUS",
    "MAX_JITTER",
    # Filter constants
    "DEFAULT_MIN_SLOPE",
    "DEFAULT_MAX_SLOPE",
    "DEFAULT_MIN_HEIGHT",
    "DEFAULT_MAX_HEIGHT",
    "DEFAULT_NOISE_THRESHOLD",
    "DEFAULT_EXCLUSION_RADIUS",
    # Transform constants
    "DEFAULT_SCALE_MIN",
    "DEFAULT_SCALE_MAX",
    "DEFAULT_ROTATION_MIN",
    "DEFAULT_ROTATION_MAX",
    "DEFAULT_OFFSET_MIN",
    "DEFAULT_OFFSET_MAX",
    # Seed constants
    "CHUNK_SEED_OFFSET",
    "LAYER_SEED_OFFSET",
    "MAX_SEED_VALUE",
    "MAX_CIRCLE_REJECTION_ATTEMPTS",
    "GAUSSIAN_EPSILON",
    # Noise map constants
    "NORMALIZATION_EPSILON",
    "BOUNDS_EPSILON",
    # Validation limits
    "MIN_FREQUENCY",
    "MIN_OCTAVES",
    "MIN_LACUNARITY",
    "MIN_PERSISTENCE",
    "MAX_PERSISTENCE",
    "MIN_AMPLITUDE",
    "MIN_DENSITY",
    "MIN_SPACING",
    "MIN_CLUSTER_SIZE",
    "MIN_CLUSTER_RADIUS",
    "MIN_RADIUS",
    "MIN_SCALE",
]
