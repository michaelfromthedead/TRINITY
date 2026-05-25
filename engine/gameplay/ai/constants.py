"""
AI Systems Constants - All magic numbers and configuration values.

This module centralizes all AI-related constants to avoid magic numbers
scattered throughout the codebase.
"""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Final

# =============================================================================
# Behavior Tree Constants
# =============================================================================

class BTStatus(Enum):
    """Behavior tree node execution status."""
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BTNodeType(Enum):
    """Behavior tree node types."""
    # Composite nodes
    SEQUENCE = auto()
    SELECTOR = auto()
    PARALLEL = auto()
    # Decorator nodes
    INVERT = auto()
    REPEAT = auto()
    TIMEOUT = auto()
    COOLDOWN = auto()
    RETRY = auto()
    FORCE_SUCCESS = auto()
    FORCE_FAILURE = auto()
    # Leaf nodes
    ACTION = auto()
    CONDITION = auto()


class ParallelPolicy(Enum):
    """Policy for parallel node success/failure determination."""
    REQUIRE_ALL = auto()      # All children must succeed
    REQUIRE_ONE = auto()      # Any one child succeeding is enough
    REQUIRE_MAJORITY = auto() # Majority must succeed


# Default tick interval for behavior trees (seconds)
BT_DEFAULT_TICK_INTERVAL: Final[float] = 0.1

# Maximum depth for behavior tree traversal (prevents infinite loops)
BT_MAX_DEPTH: Final[int] = 100

# Default timeout for timeout decorator (seconds)
BT_DEFAULT_TIMEOUT: Final[float] = 5.0

# Default cooldown for cooldown decorator (seconds)
BT_DEFAULT_COOLDOWN: Final[float] = 1.0

# Default repeat count for repeat decorator
BT_DEFAULT_REPEAT_COUNT: Final[int] = 1

# Maximum repeat count (infinite represented by -1)
BT_INFINITE_REPEAT: Final[int] = -1

# Default retry count for retry decorator
BT_DEFAULT_RETRY_COUNT: Final[int] = 3


# =============================================================================
# Blackboard Constants
# =============================================================================

# Maximum number of observers per key
BLACKBOARD_MAX_OBSERVERS: Final[int] = 100

# Default namespace for blackboard keys
BLACKBOARD_DEFAULT_NAMESPACE: Final[str] = "global"

# Key separator for hierarchical keys
BLACKBOARD_KEY_SEPARATOR: Final[str] = "."


# =============================================================================
# Utility AI Constants
# =============================================================================

class ResponseCurveType(Enum):
    """Types of response curves for utility scoring."""
    LINEAR = auto()
    QUADRATIC = auto()
    EXPONENTIAL = auto()
    LOGISTIC = auto()
    SINE = auto()
    INVERSE = auto()
    STEP = auto()
    SMOOTHSTEP = auto()
    CUSTOM = auto()


# Default update rate for utility AI (seconds)
UTILITY_DEFAULT_UPDATE_RATE: Final[float] = 0.5

# Minimum score threshold for action selection
UTILITY_MIN_SCORE_THRESHOLD: Final[float] = 0.01

# Score epsilon for floating point comparisons
UTILITY_SCORE_EPSILON: Final[float] = 1e-6

# Maximum considerations per action
UTILITY_MAX_CONSIDERATIONS: Final[int] = 50

# Default weight for considerations
UTILITY_DEFAULT_WEIGHT: Final[float] = 1.0

# Momentum factor for action switching (0 = no momentum, 1 = full momentum)
UTILITY_DEFAULT_MOMENTUM: Final[float] = 0.1

# Logistic curve parameters for sigmoid response
UTILITY_LOGISTIC_CENTER: Final[float] = 0.5
UTILITY_LOGISTIC_STEEPNESS: Final[float] = 10.0

# Smoothstep polynomial coefficients (3x^2 - 2x^3)
UTILITY_SMOOTHSTEP_COEFF_A: Final[int] = 3
UTILITY_SMOOTHSTEP_COEFF_B: Final[int] = 2

# Action history size for tracking recent actions
UTILITY_ACTION_HISTORY_SIZE: Final[int] = 10


# =============================================================================
# GOAP Constants
# =============================================================================

# Maximum planning iterations (prevents infinite loops)
GOAP_MAX_ITERATIONS: Final[int] = 1000

# Maximum plan length
GOAP_MAX_PLAN_LENGTH: Final[int] = 50

# Default action cost if not specified
GOAP_DEFAULT_ACTION_COST: Final[float] = 1.0

# A* heuristic weight
GOAP_HEURISTIC_WEIGHT: Final[float] = 1.0

# Cache size for frequently used plans
GOAP_PLAN_CACHE_SIZE: Final[int] = 100

# Plan validity timeout (seconds)
GOAP_PLAN_CACHE_TTL: Final[float] = 5.0


# =============================================================================
# Perception Constants
# =============================================================================

class SenseType(Enum):
    """Types of perception senses."""
    SIGHT = auto()
    HEARING = auto()
    DAMAGE = auto()
    SQUAD = auto()
    TOUCH = auto()
    SMELL = auto()


# Default field of view (degrees)
PERCEPTION_DEFAULT_FOV: Final[float] = 90.0

# Default perception range (units)
PERCEPTION_DEFAULT_RANGE: Final[float] = 50.0

# Alias for sight-specific range (same as default range)
PERCEPTION_DEFAULT_SIGHT_RANGE: Final[float] = PERCEPTION_DEFAULT_RANGE

# Default hearing range (units)
PERCEPTION_DEFAULT_HEARING_RANGE: Final[float] = 30.0

# Memory decay time (seconds until forgotten)
PERCEPTION_MEMORY_DECAY_TIME: Final[float] = 30.0

# Memory update interval (seconds)
PERCEPTION_MEMORY_UPDATE_INTERVAL: Final[float] = 0.1

# Maximum remembered targets
PERCEPTION_MAX_REMEMBERED_TARGETS: Final[int] = 50

# Line of sight check interval (seconds)
PERCEPTION_LOS_CHECK_INTERVAL: Final[float] = 0.05

# Sound occlusion factor (0-1, how much walls block sound)
PERCEPTION_SOUND_OCCLUSION_FACTOR: Final[float] = 0.7

# Sound falloff exponent (for distance-based attenuation)
PERCEPTION_SOUND_FALLOFF_EXPONENT: Final[float] = 2.0

# Damage awareness radius (additional radius for damage perception)
PERCEPTION_DAMAGE_AWARENESS_RADIUS: Final[float] = 5.0

# Squad communication range
PERCEPTION_SQUAD_COMM_RANGE: Final[float] = 100.0


# =============================================================================
# Knowledge/World State Constants
# =============================================================================

# Influence map grid cell size (units)
INFLUENCE_MAP_CELL_SIZE: Final[float] = 5.0

# Maximum influence value
INFLUENCE_MAX_VALUE: Final[float] = 1.0

# Minimum influence value (below this is considered zero)
INFLUENCE_MIN_VALUE: Final[float] = 0.001

# Default influence decay rate (per second)
INFLUENCE_DECAY_RATE: Final[float] = 0.1

# Default influence propagation rate (per second)
INFLUENCE_PROPAGATION_RATE: Final[float] = 0.5

# Maximum propagation distance (cells)
INFLUENCE_MAX_PROPAGATION_DISTANCE: Final[int] = 10

# Fact expiry time (seconds, 0 = never expires)
KNOWLEDGE_FACT_DEFAULT_EXPIRY: Final[float] = 0.0

# Maximum facts stored
KNOWLEDGE_MAX_FACTS: Final[int] = 1000


# =============================================================================
# Combat AI Constants
# =============================================================================

class CombatBehavior(Enum):
    """Types of combat behaviors."""
    ATTACK = auto()
    DEFEND = auto()
    FLANK = auto()
    RETREAT = auto()
    SUPPORT = auto()
    COVER = auto()
    SUPPRESS = auto()
    ADVANCE = auto()
    HOLD_POSITION = auto()


class TargetPriority(Enum):
    """Target selection priority modes."""
    NEAREST = auto()
    WEAKEST = auto()
    STRONGEST = auto()
    HIGHEST_THREAT = auto()
    LOWEST_THREAT = auto()
    MOST_DAMAGED = auto()
    LEAST_DAMAGED = auto()
    RANDOM = auto()


# Default attack range (units)
COMBAT_DEFAULT_ATTACK_RANGE: Final[float] = 10.0

# Default retreat health threshold (0-1)
COMBAT_RETREAT_HEALTH_THRESHOLD: Final[float] = 0.25

# Default flank angle (degrees from forward)
COMBAT_FLANK_ANGLE: Final[float] = 45.0

# Minimum distance for flanking maneuver (units)
COMBAT_FLANK_MIN_DISTANCE: Final[float] = 5.0

# Cover evaluation radius (units)
COMBAT_COVER_EVAL_RADIUS: Final[float] = 20.0

# Suppression fire duration (seconds)
COMBAT_SUPPRESSION_DURATION: Final[float] = 3.0

# Target re-evaluation interval (seconds)
COMBAT_TARGET_EVAL_INTERVAL: Final[float] = 0.5

# Threat level decay rate (per second)
COMBAT_THREAT_DECAY_RATE: Final[float] = 0.1

# Maximum threat level
COMBAT_MAX_THREAT_LEVEL: Final[float] = 100.0


# =============================================================================
# Social AI Constants
# =============================================================================

class RelationshipType(Enum):
    """Types of relationships between entities."""
    ALLY = auto()
    ENEMY = auto()
    NEUTRAL = auto()
    FRIENDLY = auto()
    HOSTILE = auto()
    AFRAID = auto()
    TRUSTED = auto()


class FactionStanding(IntEnum):
    """Faction standing levels."""
    HATED = -100
    HOSTILE = -50
    UNFRIENDLY = -25
    NEUTRAL = 0
    FRIENDLY = 25
    HONORED = 50
    EXALTED = 100


# Default reputation change rate
SOCIAL_REPUTATION_CHANGE_RATE: Final[float] = 1.0

# Reputation decay rate (per second)
SOCIAL_REPUTATION_DECAY_RATE: Final[float] = 0.01

# Minimum reputation value
SOCIAL_MIN_REPUTATION: Final[int] = -100

# Maximum reputation value
SOCIAL_MAX_REPUTATION: Final[int] = 100

# Relationship strength decay rate (per second)
SOCIAL_RELATIONSHIP_DECAY_RATE: Final[float] = 0.001

# Maximum relationships per entity
SOCIAL_MAX_RELATIONSHIPS: Final[int] = 100

# Faction alliance threshold
SOCIAL_ALLIANCE_THRESHOLD: Final[int] = 50

# Faction war threshold
SOCIAL_WAR_THRESHOLD: Final[int] = -50


# =============================================================================
# General AI Constants
# =============================================================================

# Default AI update rate (ticks per second)
AI_DEFAULT_UPDATE_RATE: Final[float] = 10.0

# Maximum concurrent AI agents
AI_MAX_CONCURRENT_AGENTS: Final[int] = 1000

# AI LOD distances (units) for different detail levels
AI_LOD_FULL_DISTANCE: Final[float] = 50.0
AI_LOD_MEDIUM_DISTANCE: Final[float] = 100.0
AI_LOD_LOW_DISTANCE: Final[float] = 200.0

# AI time budget per frame (milliseconds)
AI_FRAME_TIME_BUDGET_MS: Final[float] = 5.0


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "BTStatus",
    "BTNodeType",
    "ParallelPolicy",
    "ResponseCurveType",
    "SenseType",
    "CombatBehavior",
    "TargetPriority",
    "RelationshipType",
    "FactionStanding",
    # BT Constants
    "BT_DEFAULT_TICK_INTERVAL",
    "BT_MAX_DEPTH",
    "BT_DEFAULT_TIMEOUT",
    "BT_DEFAULT_COOLDOWN",
    "BT_DEFAULT_REPEAT_COUNT",
    "BT_DEFAULT_RETRY_COUNT",
    "BT_INFINITE_REPEAT",
    # Blackboard Constants
    "BLACKBOARD_MAX_OBSERVERS",
    "BLACKBOARD_DEFAULT_NAMESPACE",
    "BLACKBOARD_KEY_SEPARATOR",
    # Utility Constants
    "UTILITY_DEFAULT_UPDATE_RATE",
    "UTILITY_MIN_SCORE_THRESHOLD",
    "UTILITY_SCORE_EPSILON",
    "UTILITY_MAX_CONSIDERATIONS",
    "UTILITY_DEFAULT_WEIGHT",
    "UTILITY_DEFAULT_MOMENTUM",
    "UTILITY_LOGISTIC_CENTER",
    "UTILITY_LOGISTIC_STEEPNESS",
    "UTILITY_SMOOTHSTEP_COEFF_A",
    "UTILITY_SMOOTHSTEP_COEFF_B",
    "UTILITY_ACTION_HISTORY_SIZE",
    # GOAP Constants
    "GOAP_MAX_ITERATIONS",
    "GOAP_MAX_PLAN_LENGTH",
    "GOAP_DEFAULT_ACTION_COST",
    "GOAP_HEURISTIC_WEIGHT",
    "GOAP_PLAN_CACHE_SIZE",
    "GOAP_PLAN_CACHE_TTL",
    # Perception Constants
    "PERCEPTION_DEFAULT_FOV",
    "PERCEPTION_DEFAULT_RANGE",
    "PERCEPTION_DEFAULT_SIGHT_RANGE",
    "PERCEPTION_DEFAULT_HEARING_RANGE",
    "PERCEPTION_MEMORY_DECAY_TIME",
    "PERCEPTION_MEMORY_UPDATE_INTERVAL",
    "PERCEPTION_MAX_REMEMBERED_TARGETS",
    "PERCEPTION_LOS_CHECK_INTERVAL",
    "PERCEPTION_SOUND_OCCLUSION_FACTOR",
    "PERCEPTION_SOUND_FALLOFF_EXPONENT",
    "PERCEPTION_DAMAGE_AWARENESS_RADIUS",
    "PERCEPTION_SQUAD_COMM_RANGE",
    # Knowledge Constants
    "INFLUENCE_MAP_CELL_SIZE",
    "INFLUENCE_MAX_VALUE",
    "INFLUENCE_MIN_VALUE",
    "INFLUENCE_DECAY_RATE",
    "INFLUENCE_PROPAGATION_RATE",
    "INFLUENCE_MAX_PROPAGATION_DISTANCE",
    "KNOWLEDGE_FACT_DEFAULT_EXPIRY",
    "KNOWLEDGE_MAX_FACTS",
    # Combat Constants
    "COMBAT_DEFAULT_ATTACK_RANGE",
    "COMBAT_RETREAT_HEALTH_THRESHOLD",
    "COMBAT_FLANK_ANGLE",
    "COMBAT_FLANK_MIN_DISTANCE",
    "COMBAT_COVER_EVAL_RADIUS",
    "COMBAT_SUPPRESSION_DURATION",
    "COMBAT_TARGET_EVAL_INTERVAL",
    "COMBAT_THREAT_DECAY_RATE",
    "COMBAT_MAX_THREAT_LEVEL",
    # Social Constants
    "SOCIAL_REPUTATION_CHANGE_RATE",
    "SOCIAL_REPUTATION_DECAY_RATE",
    "SOCIAL_MIN_REPUTATION",
    "SOCIAL_MAX_REPUTATION",
    "SOCIAL_RELATIONSHIP_DECAY_RATE",
    "SOCIAL_MAX_RELATIONSHIPS",
    "SOCIAL_ALLIANCE_THRESHOLD",
    "SOCIAL_WAR_THRESHOLD",
    # General Constants
    "AI_DEFAULT_UPDATE_RATE",
    "AI_MAX_CONCURRENT_AGENTS",
    "AI_LOD_FULL_DISTANCE",
    "AI_LOD_MEDIUM_DISTANCE",
    "AI_LOD_LOW_DISTANCE",
    "AI_FRAME_TIME_BUDGET_MS",
]
