"""
Navigation subsystem constants.

Defines default values and configuration constants for the navigation system,
including agent parameters, pathfinding limits, and steering behavior weights.
"""

from enum import Enum, auto
from typing import Final

# =============================================================================
# Agent Parameters
# =============================================================================

# Default agent dimensions (in world units, typically meters)
DEFAULT_AGENT_RADIUS: Final[float] = 0.5
DEFAULT_AGENT_HEIGHT: Final[float] = 2.0
DEFAULT_STEP_HEIGHT: Final[float] = 0.4
DEFAULT_MAX_SLOPE: Final[float] = 45.0  # degrees

# Agent size constraints
MIN_AGENT_RADIUS: Final[float] = 0.1
MAX_AGENT_RADIUS: Final[float] = 10.0
MIN_AGENT_HEIGHT: Final[float] = 0.5
MAX_AGENT_HEIGHT: Final[float] = 50.0
MIN_STEP_HEIGHT: Final[float] = 0.0
MAX_STEP_HEIGHT: Final[float] = 5.0
MIN_MAX_SLOPE: Final[float] = 0.0
MAX_MAX_SLOPE: Final[float] = 89.0  # Must be < 90 degrees

# =============================================================================
# NavMesh Generation Parameters
# =============================================================================

# Voxelization defaults
DEFAULT_CELL_SIZE: Final[float] = 0.3
DEFAULT_CELL_HEIGHT: Final[float] = 0.2
MIN_CELL_SIZE: Final[float] = 0.01
MAX_CELL_SIZE: Final[float] = 10.0

# Region building
DEFAULT_MIN_REGION_AREA: Final[float] = 8.0
DEFAULT_MERGE_REGION_AREA: Final[float] = 20.0

# Contour tracing
DEFAULT_MAX_CONTOUR_ERROR: Final[float] = 1.3
DEFAULT_MAX_EDGE_LENGTH: Final[float] = 12.0

# Mesh building
DEFAULT_MAX_VERTICES_PER_POLY: Final[int] = 6
MIN_VERTICES_PER_POLY: Final[int] = 3
MAX_VERTICES_PER_POLY: Final[int] = 8

# Tile-based NavMesh
DEFAULT_TILE_SIZE: Final[float] = 48.0
MIN_TILE_SIZE: Final[float] = 8.0
MAX_TILE_SIZE: Final[float] = 1024.0

# =============================================================================
# Pathfinding Parameters
# =============================================================================

# Maximum path length and iterations
DEFAULT_MAX_PATH_LENGTH: Final[int] = 256
DEFAULT_MAX_SEARCH_NODES: Final[int] = 65536
DEFAULT_MAX_ITERATIONS: Final[int] = 100000

# Heuristic weights
DEFAULT_HEURISTIC_WEIGHT: Final[float] = 1.0
MIN_HEURISTIC_WEIGHT: Final[float] = 0.0
MAX_HEURISTIC_WEIGHT: Final[float] = 10.0

# Path costs
DEFAULT_STRAIGHT_COST: Final[float] = 1.0
DEFAULT_DIAGONAL_COST: Final[float] = 1.414  # sqrt(2)

# =============================================================================
# Path Modification Parameters
# =============================================================================

# Path smoothing
DEFAULT_SMOOTH_ITERATIONS: Final[int] = 3
DEFAULT_SMOOTH_FACTOR: Final[float] = 0.5

# Funnel/String-pulling
DEFAULT_FUNNEL_EPSILON: Final[float] = 0.001

# Corridor width
DEFAULT_CORRIDOR_WIDTH: Final[float] = 0.5
MIN_CORRIDOR_WIDTH: Final[float] = 0.1
MAX_CORRIDOR_WIDTH: Final[float] = 10.0

# =============================================================================
# Steering Behavior Parameters
# =============================================================================

# Default steering weights
DEFAULT_SEEK_WEIGHT: Final[float] = 1.0
DEFAULT_FLEE_WEIGHT: Final[float] = 1.0
DEFAULT_ARRIVE_WEIGHT: Final[float] = 1.0
DEFAULT_PURSUE_WEIGHT: Final[float] = 1.0
DEFAULT_EVADE_WEIGHT: Final[float] = 1.0
DEFAULT_WANDER_WEIGHT: Final[float] = 0.5
DEFAULT_SEPARATION_WEIGHT: Final[float] = 1.5
DEFAULT_ALIGNMENT_WEIGHT: Final[float] = 1.0
DEFAULT_COHESION_WEIGHT: Final[float] = 1.0

# Steering behavior thresholds
DEFAULT_ARRIVE_SLOW_RADIUS: Final[float] = 3.0
DEFAULT_ARRIVE_STOP_RADIUS: Final[float] = 0.5
DEFAULT_WANDER_RADIUS: Final[float] = 2.0
DEFAULT_WANDER_DISTANCE: Final[float] = 4.0
DEFAULT_WANDER_JITTER: Final[float] = 40.0

# Flocking parameters
DEFAULT_NEIGHBOR_DISTANCE: Final[float] = 10.0
DEFAULT_SEPARATION_DISTANCE: Final[float] = 2.0

# Maximum forces and speeds
DEFAULT_MAX_FORCE: Final[float] = 10.0
DEFAULT_MAX_SPEED: Final[float] = 5.0
DEFAULT_MAX_ACCELERATION: Final[float] = 20.0

# =============================================================================
# Local Avoidance Parameters
# =============================================================================

# RVO/ORCA parameters
DEFAULT_RVO_TIME_HORIZON: Final[float] = 2.0
DEFAULT_RVO_TIME_HORIZON_OBSTACLES: Final[float] = 0.5
DEFAULT_RVO_NEIGHBOR_DISTANCE: Final[float] = 15.0
DEFAULT_RVO_MAX_NEIGHBORS: Final[int] = 10

# Force-based avoidance
DEFAULT_AVOIDANCE_FORCE: Final[float] = 100.0
DEFAULT_AVOIDANCE_DISTANCE: Final[float] = 3.0

# =============================================================================
# Navigation Link Parameters
# =============================================================================

# Jump link defaults
DEFAULT_JUMP_HEIGHT: Final[float] = 2.0
DEFAULT_JUMP_DISTANCE: Final[float] = 4.0
MIN_JUMP_HEIGHT: Final[float] = 0.1
MAX_JUMP_HEIGHT: Final[float] = 10.0

# Drop link defaults
DEFAULT_DROP_HEIGHT: Final[float] = 3.0
MIN_DROP_HEIGHT: Final[float] = 0.5
MAX_DROP_HEIGHT: Final[float] = 20.0

# Climb link defaults
DEFAULT_CLIMB_HEIGHT: Final[float] = 2.5
DEFAULT_CLIMB_ANGLE: Final[float] = 80.0  # degrees
MIN_CLIMB_HEIGHT: Final[float] = 0.5
MAX_CLIMB_HEIGHT: Final[float] = 10.0

# Teleport link defaults (no physical constraints)
DEFAULT_TELEPORT_COOLDOWN: Final[float] = 1.0

# Link traversal animation durations
DEFAULT_JUMP_DURATION: Final[float] = 0.5
DEFAULT_DROP_DURATION: Final[float] = 0.3
DEFAULT_CLIMB_DURATION: Final[float] = 1.5
DEFAULT_TELEPORT_DURATION: Final[float] = 0.2

# =============================================================================
# Smart Object Parameters
# =============================================================================

# Interaction points
DEFAULT_INTERACTION_RADIUS: Final[float] = 1.0
DEFAULT_INTERACTION_ANGLE: Final[float] = 90.0  # degrees

# Slot reservation
DEFAULT_RESERVATION_TIMEOUT: Final[float] = 30.0
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 10

# =============================================================================
# Enumerations
# =============================================================================


class NavMeshBuildMode(Enum):
    """NavMesh generation mode."""
    STATIC = auto()      # Pre-built, unchanging
    DYNAMIC = auto()     # Runtime updates allowed
    TILED = auto()       # Streaming tiles
    HYBRID = auto()      # Static base with dynamic obstacles


class ObstacleType(Enum):
    """Dynamic obstacle types for NavMesh carving."""
    BOX = auto()
    CYLINDER = auto()
    CONVEX = auto()
    CUSTOM = auto()


class PathfindingAlgorithm(Enum):
    """Available pathfinding algorithms."""
    A_STAR = auto()
    JUMP_POINT_SEARCH = auto()
    THETA_STAR = auto()
    HPA_STAR = auto()
    DIJKSTRA = auto()


class HeuristicType(Enum):
    """Heuristic functions for pathfinding."""
    MANHATTAN = auto()
    EUCLIDEAN = auto()
    OCTILE = auto()
    CHEBYSHEV = auto()
    ZERO = auto()  # For Dijkstra


class SteeringBehavior(Enum):
    """Available steering behaviors."""
    SEEK = auto()
    FLEE = auto()
    ARRIVE = auto()
    PURSUE = auto()
    EVADE = auto()
    WANDER = auto()
    SEPARATION = auto()
    ALIGNMENT = auto()
    COHESION = auto()
    OBSTACLE_AVOIDANCE = auto()
    WALL_FOLLOWING = auto()
    PATH_FOLLOWING = auto()
    FLOCKING = auto()


class AvoidanceMode(Enum):
    """Local avoidance algorithms."""
    RVO = auto()        # Reciprocal Velocity Obstacles
    ORCA = auto()       # Optimal Reciprocal Collision Avoidance
    FORCE_BASED = auto()
    NONE = auto()


class NavLinkType(Enum):
    """Types of navigation links."""
    JUMP = auto()
    DROP = auto()
    CLIMB = auto()
    TELEPORT = auto()
    CUSTOM = auto()


class NavLinkDirection(Enum):
    """Navigation link traversal direction."""
    ONE_WAY = auto()
    TWO_WAY = auto()


class SlotState(Enum):
    """Smart object slot states."""
    AVAILABLE = auto()
    RESERVED = auto()
    OCCUPIED = auto()
    DISABLED = auto()


class QueryType(Enum):
    """NavMesh query types."""
    NEAREST_POINT = auto()
    RAYCAST = auto()
    POLYGON = auto()
    RANDOM_POINT = auto()
    PATH = auto()


# =============================================================================
# Numeric Precision Constants
# =============================================================================

# Epsilon for float comparisons (used in Vector3 equality, normalization threshold)
FLOAT_EPSILON: Final[float] = 1e-9

# Threshold for considering a segment/vector as zero-length
ZERO_LENGTH_THRESHOLD: Final[float] = 1e-9

# =============================================================================
# Pathfinding Internal Constants
# =============================================================================

# Edge vertex matching threshold (fraction of cell_size for shared edge detection)
EDGE_VERTEX_THRESHOLD_FACTOR: Final[float] = 0.5

# =============================================================================
# Steering Behavior Internal Constants
# =============================================================================

# Braking weight for obstacle avoidance deceleration
OBSTACLE_AVOIDANCE_BRAKING_WEIGHT: Final[float] = 0.2

# Default weight for obstacle avoidance behavior in SteeringManager
DEFAULT_OBSTACLE_AVOIDANCE_WEIGHT: Final[float] = 2.0

# Default weight for wall following behavior in SteeringManager
DEFAULT_WALL_FOLLOWING_WEIGHT: Final[float] = 1.0

# Default weight for path following behavior in SteeringManager
DEFAULT_PATH_FOLLOWING_WEIGHT: Final[float] = 1.0

# Default weight for flocking behavior in SteeringManager
DEFAULT_FLOCKING_WEIGHT: Final[float] = 1.0

# =============================================================================
# RVO/ORCA Sampling Constants
# =============================================================================

# Number of velocity samples for RVO collision-free velocity search
RVO_VELOCITY_SAMPLES: Final[int] = 250

# Speed factors for RVO velocity sampling (fractions of max_speed)
RVO_SPEED_SAMPLE_FACTORS: Final[tuple] = (0.25, 0.5, 0.75, 1.0)

# =============================================================================
# Spatial Indexing Constants
# =============================================================================

# Default cell size for spatial hash grid (NavLinkManager, SmartObjectManager)
DEFAULT_SPATIAL_CELL_SIZE: Final[float] = 10.0

# =============================================================================
# Frozen Sets for Validation
# =============================================================================

VALID_BUILD_MODES = frozenset(NavMeshBuildMode)
VALID_OBSTACLE_TYPES = frozenset(ObstacleType)
VALID_ALGORITHMS = frozenset(PathfindingAlgorithm)
VALID_HEURISTICS = frozenset(HeuristicType)
VALID_STEERING_BEHAVIORS = frozenset(SteeringBehavior)
VALID_AVOIDANCE_MODES = frozenset(AvoidanceMode)
VALID_LINK_TYPES = frozenset(NavLinkType)
VALID_LINK_DIRECTIONS = frozenset(NavLinkDirection)
VALID_SLOT_STATES = frozenset(SlotState)
VALID_QUERY_TYPES = frozenset(QueryType)
