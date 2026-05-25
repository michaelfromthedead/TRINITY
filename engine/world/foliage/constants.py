"""
Constants for the foliage system.

Centralizes all magic numbers, default values, and configuration constants
to improve maintainability and prevent duplication.
"""

# =============================================================================
# LOD (Level of Detail) Distance Thresholds
# =============================================================================

# Default LOD transition distances in world units
DEFAULT_LOD_DISTANCES = [50.0, 150.0, 500.0]

# LOD distances by foliage category
LOD_DISTANCE_TREE = [100.0, 300.0, 1000.0]
LOD_DISTANCE_SHRUB = [50.0, 150.0, 500.0]
LOD_DISTANCE_GRASS = [25.0, 50.0, 75.0]
LOD_DISTANCE_ROCK = [75.0, 250.0, 800.0]
LOD_DISTANCE_DEBRIS = [25.0, 75.0, 200.0]


# =============================================================================
# Cull Distances
# =============================================================================

# Default cull distance for foliage types (world units)
DEFAULT_CULL_DISTANCE = 2000.0

# Category-specific cull distances
CULL_DISTANCE_TREE = 3000.0
CULL_DISTANCE_SHRUB = 1500.0
CULL_DISTANCE_GRASS = 100.0
CULL_DISTANCE_ROCK = 2500.0
CULL_DISTANCE_DEBRIS = 500.0


# =============================================================================
# Density Settings
# =============================================================================

# Default instances per square unit
DEFAULT_DENSITY = 1.0

# Category-specific densities
DENSITY_GRASS = 50.0
DENSITY_DEBRIS = 2.0

# Density calculation limits
MIN_DENSITY = 0.0
MAX_DENSITY = 1000.0  # Safety cap to prevent memory issues


# =============================================================================
# Minimum Spacing
# =============================================================================

# Default minimum distance between instances
DEFAULT_MIN_SPACING = 1.0

# Category-specific minimum spacing
MIN_SPACING_TREE = 5.0
MIN_SPACING_SHRUB = 2.0
MIN_SPACING_GRASS = 0.1
MIN_SPACING_ROCK = 3.0
MIN_SPACING_DEBRIS = 0.5


# =============================================================================
# Scale Ranges
# =============================================================================

# Default scale variation (min, max)
DEFAULT_SCALE_RANGE = (0.8, 1.2)

# Category-specific scale ranges
SCALE_RANGE_TREE = (0.7, 1.3)
SCALE_RANGE_ROCK = (0.5, 2.0)


# =============================================================================
# Grass Blade Dimensions
# =============================================================================

# Default grass blade dimensions
GRASS_BLADE_WIDTH = 0.05
GRASS_BLADE_HEIGHT = 0.3
GRASS_BLADE_CURVE = 0.2
GRASS_BLADE_BEND = 0.5

# Grass blade variation ranges (multipliers)
GRASS_HEIGHT_VARIATION_MIN = 0.7
GRASS_HEIGHT_VARIATION_MAX = 1.3
GRASS_WIDTH_VARIATION_MIN = 0.7
GRASS_WIDTH_VARIATION_MAX = 1.3
GRASS_BEND_VARIATION_MIN = 0.3
GRASS_BEND_VARIATION_MAX = 1.0

# Default blades per grass instance
DEFAULT_BLADES_PER_INSTANCE = 8


# =============================================================================
# Wind Sway Parameters
# =============================================================================

# Default wind weight (intensity of wind effect)
DEFAULT_WIND_WEIGHT = 1.0

# Category-specific wind weights
WIND_WEIGHT_GRASS = 1.5
WIND_WEIGHT_SHRUB = 0.8
WIND_WEIGHT_TREE_CANOPY = 1.0
WIND_WEIGHT_ROCK = 0.0  # Rocks don't sway

# Grass wind settings
GRASS_WIND_SWAY_AMOUNT = 1.0
GRASS_WIND_SWAY_SPEED = 1.0


# =============================================================================
# Noise and Procedural Generation
# =============================================================================

# Default noise scale for placement
DEFAULT_NOISE_SCALE = 10.0

# Noise threshold for grass placement (0-1)
GRASS_NOISE_REJECTION_THRESHOLD = 0.3

# Grass slope rejection threshold (Y component of normal)
# Values below this are too steep for grass (approximately 45 degrees)
GRASS_SLOPE_THRESHOLD = 0.7


# =============================================================================
# Chunk and Cluster Sizes
# =============================================================================

# Default cluster size for HISM spatial organization
DEFAULT_CLUSTER_SIZE = 50.0

# Default grass chunk size for streaming
DEFAULT_GRASS_CHUNK_SIZE = 32.0


# =============================================================================
# View Distances and Fade
# =============================================================================

# Default grass view distance
DEFAULT_GRASS_VIEW_DISTANCE = 100.0

# Default fade distance (distance over which foliage fades out)
DEFAULT_FADE_DISTANCE = 20.0

# Alpha cutoff for grass transparency
DEFAULT_ALPHA_CUTOFF = 0.5


# =============================================================================
# Placement Algorithm Limits
# =============================================================================

# Maximum iterations for Poisson disk sampling to prevent infinite loops
POISSON_DISK_MAX_ITERATIONS = 30

# Maximum candidates to try per Poisson disk sample point
POISSON_DISK_MAX_CANDIDATES = 30

# Minimum area in square units to prevent division by zero
MIN_PLACEMENT_AREA = 0.0001

# Jitter factor for grid-based placement (fraction of spacing)
PLACEMENT_JITTER_FACTOR = 0.5


# =============================================================================
# Instance Buffer Limits
# =============================================================================

# Maximum instances per buffer to prevent GPU memory overflow
MAX_INSTANCES_PER_BUFFER = 1_000_000

# Warning threshold for instance count
INSTANCE_COUNT_WARNING_THRESHOLD = 500_000


# =============================================================================
# Slope Ranges (degrees)
# =============================================================================

# Default slope range (min, max in degrees)
DEFAULT_SLOPE_RANGE = (0.0, 90.0)

# Maximum slope for grass growth (degrees)
MAX_GRASS_SLOPE = 45.0


# =============================================================================
# Color Defaults
# =============================================================================

# Default grass colors (RGB, 0-1 range)
GRASS_COLOR_BASE = (0.1, 0.3, 0.05)
GRASS_COLOR_TIP = (0.2, 0.5, 0.1)

# Default berry color
BERRY_COLOR_DEFAULT = (1.0, 0.0, 0.0)

# Default moss color
MOSS_COLOR_DEFAULT = (0.1, 0.3, 0.1)


# =============================================================================
# Color Variation
# =============================================================================

# Default color variation (0-1)
DEFAULT_COLOR_VARIATION = 0.1


# =============================================================================
# Tree-Specific Defaults
# =============================================================================

# Tree branch detail distance
TREE_BRANCH_DETAIL_DISTANCE = 100.0


# =============================================================================
# Debris-Specific Defaults
# =============================================================================

# Default scatter radius for debris
DEBRIS_SCATTER_RADIUS = 0.5


# =============================================================================
# Hash/Noise Precision
# =============================================================================

# Position quantization factor for hash-based randomness
POSITION_QUANTIZATION_FACTOR = 100

# Hash divisor for normalizing to 0-1 range
HASH_NORMALIZE_DIVISOR = 0xFFFFFFFF
