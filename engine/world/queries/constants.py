"""
Constants for the world queries system.

This module centralizes all magic numbers and configuration values
used throughout the spatial, terrain, and navigation query systems.

Categories:
    - Spatial Query Constants
    - Terrain Query Constants
    - Navigation Query Constants
    - Epsilon/Tolerance Values
"""

from __future__ import annotations


# =============================================================================
# EPSILON / TOLERANCE VALUES
# =============================================================================

# Minimum length for vector normalization
EPSILON_NORMALIZE = 1e-10

# Minimum length for ray direction validation
EPSILON_RAY_DIRECTION = 1e-10

# Minimum height difference for terrain crossing detection
EPSILON_HEIGHT = 1e-6

# Tolerance for distance comparisons
EPSILON_DISTANCE = 1e-6

# Tolerance for hit detection in sweeps
EPSILON_HIT_DETECTION = 1e-10


# =============================================================================
# SPATIAL QUERY CONSTANTS
# =============================================================================

# Default direction when zero vector provided for Ray
DEFAULT_RAY_DIRECTION = (0.0, 0.0, 1.0)

# Default max distance for rays when not specified
DEFAULT_RAY_MAX_DISTANCE = float("inf")

# Default max hits for multi-raycast queries
DEFAULT_MAX_HITS = 10

# Default search distance for closest point queries
DEFAULT_CLOSEST_POINT_DISTANCE = 100.0

# Sweep step multiplier - controls accuracy vs performance tradeoff
# Smaller = more accurate but slower
SWEEP_STEP_MULTIPLIER_SPHERE = 0.5
SWEEP_STEP_MULTIPLIER_BOX = 1.0
SWEEP_STEP_MULTIPLIER_CAPSULE = 0.5

# Maximum adaptive step multiplier for sweeps
SWEEP_MAX_ADAPTIVE_MULTIPLIER = 4.0

# Default shape parameters
DEFAULT_SPHERE_RADIUS = 0.5
DEFAULT_BOX_HALF_EXTENT = 0.5
DEFAULT_CAPSULE_RADIUS = 0.5
DEFAULT_CAPSULE_HALF_HEIGHT = 1.0
DEFAULT_OVERLAP_RADIUS = 1.0


# =============================================================================
# TERRAIN QUERY CONSTANTS
# =============================================================================

# Default max distance for terrain raycasts
DEFAULT_TERRAIN_RAYCAST_MAX_DISTANCE = 1000.0

# Default step size for terrain line trace
DEFAULT_LINE_TRACE_STEP_SIZE = 1.0

# Terrain raycast step multiplier
DEFAULT_TERRAIN_STEP_MULTIPLIER = 1.0

# Terrain raycast base step factor (multiplied by cell_size)
TERRAIN_RAYCAST_BASE_STEP_FACTOR = 0.5

# Terrain raycast adaptive step minimum factor
TERRAIN_RAYCAST_MIN_ADAPTIVE_FACTOR = 0.5

# Terrain raycast adaptive step maximum multiplier
TERRAIN_RAYCAST_MAX_ADAPTIVE_MULTIPLIER = 4.0

# Binary search iterations for terrain intersection refinement
TERRAIN_INTERSECTION_BINARY_SEARCH_ITERATIONS = 10

# Default terrain area query resolution (samples per axis)
DEFAULT_AREA_QUERY_RESOLUTION = 10

# Higher resolution for min/max height queries
MIN_MAX_HEIGHT_RESOLUTION = 20

# Default max slope for flat area detection (degrees)
DEFAULT_FLAT_AREA_MAX_SLOPE = 15.0

# Default minimum size for flat area detection
DEFAULT_FLAT_AREA_MIN_SIZE = 10.0

# Normal calculation epsilon factor (multiplied by cell_size)
NORMAL_CALCULATION_EPSILON_FACTOR = 0.5


# =============================================================================
# NAVIGATION QUERY CONSTANTS
# =============================================================================

# Default agent dimensions
DEFAULT_AGENT_RADIUS = 0.5
DEFAULT_AGENT_HEIGHT = 2.0

# Default path configuration
DEFAULT_MAX_PATH_NODES = 2048
DEFAULT_ALLOW_PARTIAL_PATH = True

# Default search radius for navmesh projection
DEFAULT_NAVMESH_SEARCH_RADIUS = 5.0

# Default area cost (multiplier)
DEFAULT_AREA_COST = 1.0

# Path cache settings
DEFAULT_PATH_CACHE_SIZE = 100

# Default max distance for reachable area estimation
DEFAULT_REACHABLE_AREA_MAX_DISTANCE = 100.0

# Default sample count for reachable area estimation
DEFAULT_REACHABLE_AREA_SAMPLE_COUNT = 100

# Random point search max attempts
RANDOM_POINT_MAX_ATTEMPTS = 100

# Navmesh raycast step factor (multiplied by cell_size)
NAVMESH_RAYCAST_STEP_FACTOR = 0.5


# =============================================================================
# STUB NAVMESH CONSTANTS (for testing)
# =============================================================================

# Default bounds for stub navmesh
DEFAULT_STUB_BOUNDS = (-100.0, -100.0, 100.0, 100.0)

# Default cell size for stub navmesh
DEFAULT_STUB_CELL_SIZE = 1.0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Epsilon values
    "EPSILON_NORMALIZE",
    "EPSILON_RAY_DIRECTION",
    "EPSILON_HEIGHT",
    "EPSILON_DISTANCE",
    "EPSILON_HIT_DETECTION",
    # Spatial constants
    "DEFAULT_RAY_DIRECTION",
    "DEFAULT_RAY_MAX_DISTANCE",
    "DEFAULT_MAX_HITS",
    "DEFAULT_CLOSEST_POINT_DISTANCE",
    "SWEEP_STEP_MULTIPLIER_SPHERE",
    "SWEEP_STEP_MULTIPLIER_BOX",
    "SWEEP_STEP_MULTIPLIER_CAPSULE",
    "SWEEP_MAX_ADAPTIVE_MULTIPLIER",
    "DEFAULT_SPHERE_RADIUS",
    "DEFAULT_BOX_HALF_EXTENT",
    "DEFAULT_CAPSULE_RADIUS",
    "DEFAULT_CAPSULE_HALF_HEIGHT",
    "DEFAULT_OVERLAP_RADIUS",
    # Terrain constants
    "DEFAULT_TERRAIN_RAYCAST_MAX_DISTANCE",
    "DEFAULT_LINE_TRACE_STEP_SIZE",
    "DEFAULT_TERRAIN_STEP_MULTIPLIER",
    "TERRAIN_RAYCAST_BASE_STEP_FACTOR",
    "TERRAIN_RAYCAST_MIN_ADAPTIVE_FACTOR",
    "TERRAIN_RAYCAST_MAX_ADAPTIVE_MULTIPLIER",
    "TERRAIN_INTERSECTION_BINARY_SEARCH_ITERATIONS",
    "DEFAULT_AREA_QUERY_RESOLUTION",
    "MIN_MAX_HEIGHT_RESOLUTION",
    "DEFAULT_FLAT_AREA_MAX_SLOPE",
    "DEFAULT_FLAT_AREA_MIN_SIZE",
    "NORMAL_CALCULATION_EPSILON_FACTOR",
    # Navigation constants
    "DEFAULT_AGENT_RADIUS",
    "DEFAULT_AGENT_HEIGHT",
    "DEFAULT_MAX_PATH_NODES",
    "DEFAULT_ALLOW_PARTIAL_PATH",
    "DEFAULT_NAVMESH_SEARCH_RADIUS",
    "DEFAULT_AREA_COST",
    "DEFAULT_PATH_CACHE_SIZE",
    "DEFAULT_REACHABLE_AREA_MAX_DISTANCE",
    "DEFAULT_REACHABLE_AREA_SAMPLE_COUNT",
    "RANDOM_POINT_MAX_ATTEMPTS",
    "NAVMESH_RAYCAST_STEP_FACTOR",
    # Stub constants
    "DEFAULT_STUB_BOUNDS",
    "DEFAULT_STUB_CELL_SIZE",
]
