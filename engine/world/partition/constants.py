"""
Constants for the World Partition system.

Centralizes magic numbers and configuration values used across grid,
cell, streaming, and data layer systems.
"""

# =============================================================================
# Grid Defaults
# =============================================================================

# Default grid dimensions (cells)
DEFAULT_GRID_WIDTH: int = 100
DEFAULT_GRID_HEIGHT: int = 100

# Default cell size in world units
DEFAULT_CELL_SIZE: float = 256.0

# Default load radius in cell units (not world units)
DEFAULT_GRID_LOAD_RADIUS: float = 5.0


# =============================================================================
# Cell Bounds Defaults
# =============================================================================

# Default cell height bounds (essentially infinite for 2D grid in 3D world)
CELL_VERTICAL_MIN: float = -1e6
CELL_VERTICAL_MAX: float = 1e6

# Default cell bounds maximum (when not calculated from grid)
DEFAULT_CELL_BOUNDS_MAX: float = 256.0


# =============================================================================
# Geometry Constants
# =============================================================================

# Approximation of sqrt(2)/2 for AABB diagonal distance calculations
SQRT2_OVER_2: float = 0.7071067811865476

# Multiplier for calculating center point (half)
HALF_MULTIPLIER: float = 0.5


# =============================================================================
# Memory Estimation Constants (bytes)
# =============================================================================

# Base overhead per cell
CELL_BASE_MEMORY_BYTES: int = 256

# Estimated memory per actor reference
ACTOR_MEMORY_ESTIMATE_BYTES: int = 128

# Estimated memory per foliage instance
FOLIAGE_MEMORY_ESTIMATE_BYTES: int = 64


# =============================================================================
# Streaming Source Defaults
# =============================================================================

# Default load radius for player streaming source (world units)
DEFAULT_PLAYER_LOAD_RADIUS: float = 5000.0

# Default load radius for camera streaming source (world units)
DEFAULT_CAMERA_LOAD_RADIUS: float = 3000.0

# Default forward bias for camera (loads further in view direction)
DEFAULT_CAMERA_FORWARD_BIAS: float = 1.5

# Default load radius for custom streaming sources (world units)
DEFAULT_CUSTOM_LOAD_RADIUS: float = 2000.0


# =============================================================================
# Streaming Configuration Defaults
# =============================================================================

# Distance at which cells load (world units)
DEFAULT_STREAMING_LOAD_DISTANCE: float = 5000.0

# Distance at which cells unload (world units)
DEFAULT_STREAMING_UNLOAD_DISTANCE: float = 6000.0

# Buffer to prevent load/unload thrashing (world units)
DEFAULT_STREAMING_HYSTERESIS: float = 1000.0

# Multiplier for priority calculations
DEFAULT_PRIORITY_SCALE: float = 1.0

# Maximum simultaneous cell load operations
DEFAULT_MAX_CONCURRENT_LOADS: int = 4

# Maximum simultaneous cell unload operations
DEFAULT_MAX_CONCURRENT_UNLOADS: int = 2

# Distance for preload hints (world units)
DEFAULT_PRELOAD_DISTANCE: float = 7000.0


# =============================================================================
# Streaming Budget Defaults
# =============================================================================

# Maximum memory budget for streaming content (MB)
DEFAULT_MEMORY_BUDGET_MB: float = 1024.0

# Maximum IO bandwidth (MB/s)
DEFAULT_IO_BANDWIDTH_MBPS: float = 100.0

# Maximum frame time budget for streaming operations (ms)
DEFAULT_FRAME_TIME_BUDGET_MS: float = 2.0


# =============================================================================
# Streaming Priority Calculation Constants
# =============================================================================

# Base priority adjustment for distance calculations
PRIORITY_DISTANCE_BASE: int = 1000

# Divisor for converting distance to priority (smaller = higher priority)
PRIORITY_DISTANCE_DIVISOR: float = 10.0

# Divisor for unload priority calculations
UNLOAD_PRIORITY_DIVISOR: float = 100.0


# =============================================================================
# Streaming Volume Defaults
# =============================================================================

# Default volume bounds size
DEFAULT_VOLUME_SIZE: float = 100.0


# =============================================================================
# Data Layer Defaults
# =============================================================================

# Default load distance for streamed layers (world units)
DEFAULT_LAYER_LOAD_DISTANCE: float = 5000.0


# =============================================================================
# Data Layer Priorities (higher = loads first)
# =============================================================================

LAYER_PRIORITY_RUNTIME: int = 100
LAYER_PRIORITY_LANDSCAPE: int = 90
LAYER_PRIORITY_GAMEPLAY: int = 80
LAYER_PRIORITY_LIGHTING: int = 70
LAYER_PRIORITY_NAVIGATION: int = 60
LAYER_PRIORITY_FOLIAGE: int = 50
LAYER_PRIORITY_AUDIO: int = 40
LAYER_PRIORITY_VFX: int = 30


# =============================================================================
# Data Layer Load Distances (world units)
# =============================================================================

LAYER_DISTANCE_GAMEPLAY: float = 5000.0
LAYER_DISTANCE_LIGHTING: float = 4000.0
LAYER_DISTANCE_FOLIAGE: float = 3000.0
LAYER_DISTANCE_VFX: float = 2500.0
LAYER_DISTANCE_AUDIO: float = 2000.0


# =============================================================================
# Validation Constants
# =============================================================================

# Minimum allowed cell size to prevent division by zero
MIN_CELL_SIZE: float = 0.001

# Minimum allowed load radius
MIN_LOAD_RADIUS: float = 0.001
