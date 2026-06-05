"""
Constants for the terrain system.

This module centralizes all magic numbers and configuration values used
across the terrain system to improve maintainability and clarity.
"""

# =============================================================================
# Heightfield Constants
# =============================================================================

# Default heightfield configuration
DEFAULT_RESOLUTION = 65  # Samples per edge (64+1 for stitching)
DEFAULT_HEIGHT_MIN = -500.0
DEFAULT_HEIGHT_MAX = 500.0
DEFAULT_SCALE = 1.0  # World units per sample

# Height range as tuple for default HeightfieldConfig
DEFAULT_HEIGHT_RANGE = (DEFAULT_HEIGHT_MIN, DEFAULT_HEIGHT_MAX)

# Minimum valid resolution (must have at least 2 samples to form a surface)
MIN_RESOLUTION = 2

# =============================================================================
# Precision Constants
# =============================================================================

# 16-bit quantization
BITS_16_MAX_VALUE = 65535  # 2^16 - 1

# Float comparison epsilon for height equality checks
HEIGHT_EPSILON = 1e-6

# Epsilon for vector normalization (avoid division by near-zero)
NORMAL_EPSILON = 1e-10

# =============================================================================
# Normal Calculation Constants
# =============================================================================

# Multiplier for delta in central difference gradient calculation
# delta = scale * NORMAL_DELTA_MULTIPLIER
NORMAL_DELTA_MULTIPLIER = 0.5

# Divisor for gradient: gradient = (h_right - h_left) / (GRADIENT_DIVISOR * delta)
# Effectively 2.0 for central difference method
GRADIENT_DIVISOR = 2.0

# =============================================================================
# Compression Constants
# =============================================================================

# zlib compression level (0-9, higher = better compression, slower)
ZLIB_COMPRESSION_LEVEL = 6

# Minimum size of compressed header in bytes:
# 4 (resolution) + 1 (precision) + 8 (min_h) + 8 (max_h) + 8 (scale) = 29
COMPRESSED_HEADER_SIZE = 29

# =============================================================================
# LOD Constants
# =============================================================================

# Default number of LOD levels
DEFAULT_LOD_LEVELS = 6

# Default LOD distance thresholds (world units)
DEFAULT_LOD_DISTANCES = (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)

# Minimum LOD levels (must have at least one)
MIN_LOD_LEVELS = 1

# =============================================================================
# Component Constants
# =============================================================================

# Default component bounds (unit cube)
DEFAULT_BOUNDS = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

# LOD bias range
LOD_BIAS_MIN = -1.0
LOD_BIAS_MAX = 1.0

# LOD bias multiplier (how many LOD levels to adjust per 1.0 of bias)
LOD_BIAS_MULTIPLIER = 2

# =============================================================================
# Physics Constants
# =============================================================================

# Default friction coefficient (0.0 = no friction, 1.0 = maximum friction)
DEFAULT_FRICTION = 0.6

# Default restitution coefficient (bounciness, 0.0 = no bounce, 1.0 = full bounce)
DEFAULT_RESTITUTION = 0.1

# Physics coefficient range
PHYSICS_COEFF_MIN = 0.0
PHYSICS_COEFF_MAX = 1.0

# =============================================================================
# Raycast Constants
# =============================================================================

# Default maximum raycast distance (world units)
DEFAULT_RAYCAST_MAX_DISTANCE = 10000.0

# Ray direction zero check epsilon
RAY_DIRECTION_EPSILON = 1e-10

# Step size multiplier for ray marching (relative to heightfield scale)
RAYCAST_STEP_MULTIPLIER = 0.5

# Number of binary search iterations for raycast refinement
RAYCAST_BINARY_SEARCH_ITERATIONS = 8


# =============================================================================
# Sculpting Constants (from sculpting.py)
# =============================================================================

# Brush defaults
DEFAULT_BRUSH_SIZE: float = 10.0
DEFAULT_BRUSH_RADIUS: float = 10.0  # Alias for brush size
DEFAULT_BRUSH_STRENGTH: float = 0.5
DEFAULT_BRUSH_FALLOFF: float = 0.5

# Smooth tool
DEFAULT_SMOOTH_KERNEL_SIZE: int = 3
SMOOTH_GAUSSIAN_SIGMA_FACTOR: float = 0.5

# Erosion simulation
DEFAULT_EROSION_ITERATIONS: int = 10
DEFAULT_SEDIMENT_CAPACITY: float = 0.1
DEFAULT_DEPOSITION_RATE: float = 0.3
DEFAULT_EROSION_RATE: float = 0.3

# Noise tool
DEFAULT_NOISE_SCALE: float = 0.1
DEFAULT_NOISE_OCTAVES: int = 4
DEFAULT_NOISE_PERSISTENCE: float = 0.5
DEFAULT_NOISE_SEED: int = 42

# Undo/redo system
DEFAULT_MAX_UNDO_LEVELS: int = 50

# Numerical thresholds for sculpting
SCULPT_HEIGHT_CHANGE_EPSILON: float = 1e-6
SCULPT_RAMP_LENGTH_EPSILON: float = 1e-6


# =============================================================================
# Materials Constants (from materials.py)
# =============================================================================

# Height blending
DEFAULT_BLEND_SHARPNESS: float = 8.0

# Weight threshold for considering a layer active
WEIGHT_ACTIVE_THRESHOLD: float = 0.001

# Material weight comparison epsilon
MATERIAL_WEIGHT_EPSILON: float = 1e-6


# =============================================================================
# LOD Constants (from lod.py)
# =============================================================================

# Distance thresholds
MIN_CAMERA_DISTANCE: float = 1.0

# Quadtree defaults
DEFAULT_QUADTREE_MAX_DEPTH: int = 8
DEFAULT_QUADTREE_BASE_ERROR: float = 100.0

# LOD system defaults
DEFAULT_PATCH_SIZE: float = 64.0
DEFAULT_MAX_LOD_LEVEL: int = 6
DEFAULT_ERROR_THRESHOLD: float = 4.0

# Clipmap ring defaults
DEFAULT_CLIPMAP_RESOLUTION: int = 64

# Skirt rendering
DEFAULT_SKIRT_DEPTH: float = 10.0

# Vertex morphing
LOD_MORPH_START_RATIO: float = 0.8
LOD_MORPH_TRANSITION_SPEED: float = 0.1


# =============================================================================
# Features Constants (from features.py)
# =============================================================================

# Terrain holes
DEFAULT_HOLE_RADIUS: float = 10.0
DEFAULT_HOLE_MASK_RESOLUTION: int = 32

# Spline points
DEFAULT_SPLINE_WIDTH: float = 10.0
DEFAULT_SPLINE_SEGMENT_LENGTH: float = 5.0

# Spline evaluation
SPLINE_TANGENT_EPSILON: float = 0.001
SPLINE_LENGTH_SEGMENTS: int = 100
SPLINE_SEARCH_SEGMENTS: int = 50
SPLINE_DISTANCE_SEGMENTS: int = 100

# Road spline
DEFAULT_ROAD_BANK_ANGLE: float = 5.0
DEFAULT_ROAD_DEPTH: float = 0.1
DEFAULT_ROAD_BLEND_WIDTH: float = 5.0

# River spline
DEFAULT_RIVER_DEPTH: float = 2.0
DEFAULT_RIVER_BANK_SLOPE: float = 45.0

# Terrain deformation
DEFAULT_SMOOTHING_STRENGTH: float = 0.5
DEFAULT_PLATEAU_BLEND_WIDTH: float = 10.0

# Collision/raycast for features
FEATURES_RAYCAST_MAX_DISTANCE: float = 1000.0
FEATURES_RAYCAST_STEP_FACTOR: float = 0.5
