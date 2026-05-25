"""
Entity & Object Model - Centralized Constants
==============================================
All magic numbers and configuration values for the entity system in one place.
"""
from __future__ import annotations

from enum import Enum, auto

# =============================================================================
# ENTITY LIFECYCLE STATES
# =============================================================================


class LifecycleState(Enum):
    """Entity lifecycle states (ordered progression)."""

    UNINITIALIZED = auto()  # Created but not yet initialized
    CREATED = auto()        # Constructor called, initial setup
    INITIALIZING = auto()   # Running initialization hooks
    INITIALIZED = auto()    # All initialization complete
    BEGINNING_PLAY = auto() # begin_play hook running
    ACTIVE = auto()         # Fully active and ticking
    DEACTIVATING = auto()   # end_play hook running
    DEACTIVATED = auto()    # No longer active but not destroyed
    DESTROYING = auto()     # Destruction in progress
    DESTROYED = auto()      # Fully destroyed, ready for cleanup


# State transition mappings - which states can transition to which
VALID_LIFECYCLE_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.UNINITIALIZED: frozenset({LifecycleState.CREATED}),
    LifecycleState.CREATED: frozenset({LifecycleState.INITIALIZING, LifecycleState.DESTROYING}),
    LifecycleState.INITIALIZING: frozenset({LifecycleState.INITIALIZED, LifecycleState.DESTROYING}),
    LifecycleState.INITIALIZED: frozenset({LifecycleState.BEGINNING_PLAY, LifecycleState.DESTROYING}),
    LifecycleState.BEGINNING_PLAY: frozenset({LifecycleState.ACTIVE, LifecycleState.DESTROYING}),
    LifecycleState.ACTIVE: frozenset({LifecycleState.DEACTIVATING, LifecycleState.DESTROYING}),
    LifecycleState.DEACTIVATING: frozenset({LifecycleState.DEACTIVATED, LifecycleState.DESTROYING}),
    LifecycleState.DEACTIVATED: frozenset({LifecycleState.BEGINNING_PLAY, LifecycleState.DESTROYING}),
    LifecycleState.DESTROYING: frozenset({LifecycleState.DESTROYED}),
    LifecycleState.DESTROYED: frozenset(),  # Terminal state
}


# =============================================================================
# ACTOR TYPES
# =============================================================================


class ActorType(Enum):
    """Actor classification types."""

    STATIC = auto()     # Non-moving, no physics
    DYNAMIC = auto()    # Physics-enabled, can move
    PAWN = auto()       # Can be possessed by controllers
    CHARACTER = auto()  # Humanoid pawn with movement


# =============================================================================
# CONTROLLER TYPES
# =============================================================================


class ControllerType(Enum):
    """Controller classification types."""

    PLAYER = auto()     # Human player controller
    AI = auto()         # AI-driven controller
    REMOTE = auto()     # Network-replicated controller
    SPECTATOR = auto()  # Observer-only controller


# =============================================================================
# SPAWNER CONFIGURATION
# =============================================================================

# Default pool sizes
DEFAULT_ENTITY_POOL_SIZE: int = 64
DEFAULT_ENTITY_POOL_MAX_SIZE: int = 1024
DEFAULT_ENTITY_POOL_GROW_FACTOR: float = 2.0

# Spawner configuration
DEFAULT_SPAWN_QUEUE_SIZE: int = 256
DEFAULT_DESTROY_QUEUE_SIZE: int = 256
DEFAULT_SPAWN_BATCH_SIZE: int = 32
DEFAULT_DESTROY_BATCH_SIZE: int = 32

# Deferred operations
DEFERRED_SPAWN_PRIORITY: int = 0
DEFERRED_DESTROY_PRIORITY: int = 100


# =============================================================================
# ENTITY IDENTIFICATION
# =============================================================================

# Entity ID generation
ENTITY_ID_INVALID: int = 0
ENTITY_ID_START: int = 1

# Entity name constraints
ENTITY_NAME_MAX_LENGTH: int = 256
ENTITY_TAG_MAX_COUNT: int = 64


# =============================================================================
# COMPONENT CONFIGURATION
# =============================================================================

# Maximum components per entity
MAX_COMPONENTS_PER_ENTITY: int = 64

# Component slot allocation
DEFAULT_COMPONENT_SLOT_COUNT: int = 16


# =============================================================================
# PREFAB CONFIGURATION
# =============================================================================

# Prefab instantiation
DEFAULT_PREFAB_CACHE_SIZE: int = 128
PREFAB_INSTANCE_BATCH_SIZE: int = 16

# Prefab inheritance depth limit (to prevent infinite recursion)
MAX_PREFAB_INHERITANCE_DEPTH: int = 16


# =============================================================================
# POSSESSION CONFIGURATION
# =============================================================================

# Possession transition timing
POSSESSION_TRANSITION_TIMEOUT_MS: int = 100
UNPOSSESS_CLEANUP_DELAY_MS: int = 16  # One frame at 60 FPS
POSSESSION_HISTORY_MAX_LENGTH: int = 10

# AI movement defaults
DEFAULT_AI_MOVE_SPEED: float = 5.0
DEFAULT_ACCEPTANCE_RADIUS: float = 50.0


# =============================================================================
# CHARACTER MOVEMENT CONFIGURATION
# =============================================================================

# Character movement defaults
DEFAULT_MAX_WALK_SPEED: float = 6.0
DEFAULT_MAX_RUN_SPEED: float = 12.0
DEFAULT_JUMP_VELOCITY: float = 8.0
DEFAULT_JUMP_FORCE: float = 10.0  # For __init__.py Character class
DEFAULT_CHARACTER_HEALTH: float = 100.0
DEFAULT_CHARACTER_MAX_HEALTH: float = 100.0

# Character movement multipliers
CROUCH_SPEED_MULTIPLIER: float = 0.5
SPRINT_SPEED_MULTIPLIER: float = 1.5


# =============================================================================
# TICK CONFIGURATION
# =============================================================================

# Tick groups (execution order)
class TickGroup(Enum):
    """Standard tick execution groups."""

    PRE_PHYSICS = 0
    PHYSICS = 10
    POST_PHYSICS = 20
    PRE_UPDATE = 30
    UPDATE = 40
    POST_UPDATE = 50
    LATE_UPDATE = 60


# Default tick rates
DEFAULT_TICK_RATE_HZ: int = 60
DEFAULT_FIXED_TICK_RATE_HZ: int = 60
MIN_TICK_DELTA_SECONDS: float = 0.0001  # Prevent division by zero
MAX_TICK_DELTA_SECONDS: float = 0.25    # Cap to prevent spiral of death


# =============================================================================
# MEMORY ALIGNMENT
# =============================================================================

# Cache-friendly alignment
ENTITY_CACHE_LINE_BYTES: int = 64
ENTITY_ALIGNMENT_BYTES: int = 16


# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

# Debug visualization
DEBUG_ACTOR_BOUNDS_COLOR: tuple[int, int, int, int] = (0, 255, 0, 128)
DEBUG_PAWN_INDICATOR_COLOR: tuple[int, int, int, int] = (255, 255, 0, 128)
DEBUG_POSSESSED_INDICATOR_COLOR: tuple[int, int, int, int] = (0, 255, 255, 128)


__all__ = [
    # Enums
    "LifecycleState",
    "ActorType",
    "ControllerType",
    "TickGroup",
    # Lifecycle
    "VALID_LIFECYCLE_TRANSITIONS",
    # Pool sizes
    "DEFAULT_ENTITY_POOL_SIZE",
    "DEFAULT_ENTITY_POOL_MAX_SIZE",
    "DEFAULT_ENTITY_POOL_GROW_FACTOR",
    # Spawner
    "DEFAULT_SPAWN_QUEUE_SIZE",
    "DEFAULT_DESTROY_QUEUE_SIZE",
    "DEFAULT_SPAWN_BATCH_SIZE",
    "DEFAULT_DESTROY_BATCH_SIZE",
    "DEFERRED_SPAWN_PRIORITY",
    "DEFERRED_DESTROY_PRIORITY",
    # Entity ID
    "ENTITY_ID_INVALID",
    "ENTITY_ID_START",
    "ENTITY_NAME_MAX_LENGTH",
    "ENTITY_TAG_MAX_COUNT",
    # Components
    "MAX_COMPONENTS_PER_ENTITY",
    "DEFAULT_COMPONENT_SLOT_COUNT",
    # Prefabs
    "DEFAULT_PREFAB_CACHE_SIZE",
    "PREFAB_INSTANCE_BATCH_SIZE",
    "MAX_PREFAB_INHERITANCE_DEPTH",
    # Possession
    "POSSESSION_TRANSITION_TIMEOUT_MS",
    "UNPOSSESS_CLEANUP_DELAY_MS",
    "POSSESSION_HISTORY_MAX_LENGTH",
    "DEFAULT_AI_MOVE_SPEED",
    "DEFAULT_ACCEPTANCE_RADIUS",
    # Character movement
    "DEFAULT_MAX_WALK_SPEED",
    "DEFAULT_MAX_RUN_SPEED",
    "DEFAULT_JUMP_VELOCITY",
    "DEFAULT_JUMP_FORCE",
    "DEFAULT_CHARACTER_HEALTH",
    "DEFAULT_CHARACTER_MAX_HEALTH",
    "CROUCH_SPEED_MULTIPLIER",
    "SPRINT_SPEED_MULTIPLIER",
    # Tick
    "DEFAULT_TICK_RATE_HZ",
    "DEFAULT_FIXED_TICK_RATE_HZ",
    "MIN_TICK_DELTA_SECONDS",
    "MAX_TICK_DELTA_SECONDS",
    # Memory
    "ENTITY_CACHE_LINE_BYTES",
    "ENTITY_ALIGNMENT_BYTES",
    # Debug
    "DEBUG_ACTOR_BOUNDS_COLOR",
    "DEBUG_PAWN_INDICATOR_COLOR",
    "DEBUG_POSSESSED_INDICATOR_COLOR",
]
