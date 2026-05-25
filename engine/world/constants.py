"""
Constants for the World Layer.

Centralizes magic numbers and configuration values used across level
and world composition systems.
"""

# =============================================================================
# Level Bounds Constants
# =============================================================================

# Multiplier for calculating center/extents (half)
HALF_MULTIPLIER: float = 0.5

# Default vertical bounds for tile calculations (essentially infinite)
TILE_VERTICAL_MIN: float = -1e6
TILE_VERTICAL_MAX: float = 1e6


# =============================================================================
# Streaming Level Defaults
# =============================================================================

# Distance at which streaming levels begin loading (world units)
DEFAULT_STREAMING_LOAD_DISTANCE: float = 5000.0

# Distance at which streaming levels unload (world units)
DEFAULT_STREAMING_UNLOAD_DISTANCE: float = 6000.0

# Buffer distance to prevent load/unload thrashing
DEFAULT_STREAMING_HYSTERESIS: float = 1000.0


# =============================================================================
# World Composition Defaults
# =============================================================================

# Distance from origin before rebasing is triggered (floating point precision)
DEFAULT_ORIGIN_SHIFT_THRESHOLD: float = 10000.0

# Default tile size for large world organization (world units)
DEFAULT_TILE_SIZE: float = 2048.0

# Overlap between adjacent tiles (world units)
DEFAULT_TILE_OVERLAP: float = 128.0


# =============================================================================
# Actor Transform Defaults
# =============================================================================

# Default quaternion representing no rotation (identity)
DEFAULT_ROTATION_QUATERNION: tuple = (0.0, 0.0, 0.0, 1.0)

# Default uniform scale
DEFAULT_SCALE: float = 1.0
