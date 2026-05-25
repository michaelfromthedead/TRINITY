"""
Character Physics Configuration Constants.

All configuration values for character physics including capsule dimensions,
movement parameters, ground detection settings, and physics constraints.
"""

from enum import Enum, IntEnum
from typing import Final


# =============================================================================
# Capsule Configuration
# =============================================================================

DEFAULT_CAPSULE_RADIUS: Final[float] = 0.35
DEFAULT_CAPSULE_HEIGHT: Final[float] = 1.8
DEFAULT_CROUCHED_HEIGHT: Final[float] = 1.2
DEFAULT_PRONE_HEIGHT: Final[float] = 0.6


# =============================================================================
# Step and Slope Configuration
# =============================================================================

DEFAULT_STEP_HEIGHT: Final[float] = 0.35
MAX_SLOPE_ANGLE: Final[float] = 45.0  # degrees
STEEP_SLOPE_ANGLE: Final[float] = 60.0  # degrees - slides down
MIN_SLOPE_ANGLE: Final[float] = 5.0  # degrees - considered flat below this


# =============================================================================
# Collision Detection Configuration
# =============================================================================

SKIN_WIDTH: Final[float] = 0.02
GROUND_PROBE_DISTANCE: Final[float] = 0.1
GROUND_SPHERE_PROBE_RADIUS: Final[float] = 0.25
MAX_DEPENETRATION_VELOCITY: Final[float] = 10.0
MIN_MOVE_DISTANCE: Final[float] = 0.001
MAX_COLLISION_ITERATIONS: Final[int] = 4
PUSH_POWER: Final[float] = 2.0


# =============================================================================
# Movement Configuration
# =============================================================================

AIR_CONTROL: Final[float] = 0.3
DEFAULT_JUMP_VELOCITY: Final[float] = 5.0
MAX_FALL_VELOCITY: Final[float] = 55.0
DEFAULT_GRAVITY: Final[float] = -9.81


# =============================================================================
# Ground Detection Configuration
# =============================================================================

COYOTE_TIME_MS: Final[float] = 150.0  # Grace period for jumping after leaving ground
JUMP_BUFFER_TIME_MS: Final[float] = 100.0  # Pre-landing jump buffer
LEDGE_GRAB_DISTANCE: Final[float] = 0.5
LEDGE_DETECTION_HEIGHT: Final[float] = 2.2  # Head + reach height


# =============================================================================
# Movement Mode Speeds (units per second)
# =============================================================================

class MovementSpeed(float, Enum):
    """Default speeds for each movement mode."""
    WALKING = 2.5
    RUNNING = 6.0
    SPRINTING = 9.0
    CROUCHING = 1.5
    PRONE = 0.5
    SWIMMING = 3.0
    CLIMBING = 2.0
    FLYING = 10.0
    LADDERING = 2.5


WALKING_SPEED: Final[float] = MovementSpeed.WALKING.value
RUNNING_SPEED: Final[float] = MovementSpeed.RUNNING.value
SPRINTING_SPEED: Final[float] = MovementSpeed.SPRINTING.value
CROUCHING_SPEED: Final[float] = MovementSpeed.CROUCHING.value
PRONE_SPEED: Final[float] = MovementSpeed.PRONE.value
SWIMMING_SPEED: Final[float] = MovementSpeed.SWIMMING.value
CLIMBING_SPEED: Final[float] = MovementSpeed.CLIMBING.value
FLYING_SPEED: Final[float] = MovementSpeed.FLYING.value


# =============================================================================
# Acceleration and Deceleration
# =============================================================================

GROUND_ACCELERATION: Final[float] = 20.0
GROUND_DECELERATION: Final[float] = 15.0
AIR_ACCELERATION: Final[float] = 8.0
AIR_DECELERATION: Final[float] = 5.0
TURN_ACCELERATION: Final[float] = 10.0


# =============================================================================
# Platform Handling
# =============================================================================

PLATFORM_STICK_FORCE: Final[float] = 10.0
PLATFORM_DETACH_THRESHOLD: Final[float] = 5.0  # Velocity to detach
PLATFORM_UPDATE_PRIORITY: Final[int] = -100  # Execute before character update
MAX_PLATFORM_VELOCITY: Final[float] = 50.0


# =============================================================================
# Ragdoll Configuration
# =============================================================================

RAGDOLL_BLEND_TIME_MS: Final[float] = 200.0
RAGDOLL_RECOVERY_TIME_MS: Final[float] = 500.0
RAGDOLL_MIN_VELOCITY: Final[float] = 0.5  # Threshold for settled detection
RAGDOLL_SETTLED_TIME_MS: Final[float] = 1000.0  # Time to wait before recovery


# =============================================================================
# Active Ragdoll PD Controller
# =============================================================================

DEFAULT_PD_KP: Final[float] = 300.0  # Proportional gain
DEFAULT_PD_KD: Final[float] = 30.0   # Derivative gain
MAX_TORQUE: Final[float] = 500.0
BALANCE_THRESHOLD: Final[float] = 0.3  # Maximum allowed COM offset


# =============================================================================
# Physics Animation Blending
# =============================================================================

class BlendMode(str, Enum):
    """Modes for blending physics and animation."""
    POSE = "pose"           # Direct pose replacement
    ADDITIVE = "additive"   # Add physics delta to animation
    CHAIN = "chain"         # Physics controls chain hierarchy


BLEND_POSE: Final[str] = BlendMode.POSE.value
BLEND_ADDITIVE: Final[str] = BlendMode.ADDITIVE.value
BLEND_CHAIN: Final[str] = BlendMode.CHAIN.value

DEFAULT_BLEND_WEIGHT: Final[float] = 1.0
HIT_REACTION_BLEND_IN_MS: Final[float] = 50.0
HIT_REACTION_BLEND_OUT_MS: Final[float] = 300.0


# =============================================================================
# Character Interaction
# =============================================================================

PUSH_FORCE: Final[float] = 500.0
GRAB_DISTANCE: Final[float] = 1.5
CARRY_MASS_LIMIT: Final[float] = 50.0  # kg
THROW_FORCE_MULTIPLIER: Final[float] = 10.0
VAULT_MAX_HEIGHT: Final[float] = 1.2
CLIMB_MAX_HEIGHT: Final[float] = 2.5


# =============================================================================
# Collision Layers
# =============================================================================

class CollisionLayer(IntEnum):
    """Collision layer definitions for character physics."""
    DEFAULT = 0
    STATIC = 1
    DYNAMIC = 2
    CHARACTER = 3
    PROJECTILE = 4
    TRIGGER = 5
    PLATFORM = 6
    RAGDOLL = 7
    VEHICLE = 8
    DEBRIS = 9
    WATER = 10


LAYER_DEFAULT: Final[int] = CollisionLayer.DEFAULT
LAYER_STATIC: Final[int] = CollisionLayer.STATIC
LAYER_DYNAMIC: Final[int] = CollisionLayer.DYNAMIC
LAYER_CHARACTER: Final[int] = CollisionLayer.CHARACTER
LAYER_PLATFORM: Final[int] = CollisionLayer.PLATFORM
LAYER_RAGDOLL: Final[int] = CollisionLayer.RAGDOLL


# =============================================================================
# Collision Masks
# =============================================================================

MASK_CHARACTER_MOVEMENT: Final[int] = (
    (1 << CollisionLayer.STATIC) |
    (1 << CollisionLayer.DYNAMIC) |
    (1 << CollisionLayer.PLATFORM)
)

MASK_GROUND_DETECTION: Final[int] = (
    (1 << CollisionLayer.STATIC) |
    (1 << CollisionLayer.DYNAMIC) |
    (1 << CollisionLayer.PLATFORM)
)

MASK_RAGDOLL: Final[int] = (
    (1 << CollisionLayer.STATIC) |
    (1 << CollisionLayer.DYNAMIC) |
    (1 << CollisionLayer.RAGDOLL)
)


# =============================================================================
# Material Friction Coefficients
# =============================================================================

class SurfaceMaterial(str, Enum):
    """Surface material types for physics interactions."""
    DEFAULT = "default"
    CONCRETE = "concrete"
    METAL = "metal"
    WOOD = "wood"
    GRASS = "grass"
    SAND = "sand"
    ICE = "ice"
    MUD = "mud"
    WATER = "water"


FRICTION_DEFAULT: Final[float] = 0.6
FRICTION_CONCRETE: Final[float] = 0.8
FRICTION_METAL: Final[float] = 0.4
FRICTION_WOOD: Final[float] = 0.5
FRICTION_GRASS: Final[float] = 0.5
FRICTION_SAND: Final[float] = 0.4
FRICTION_ICE: Final[float] = 0.05
FRICTION_MUD: Final[float] = 0.3

SURFACE_FRICTION: Final[dict[str, float]] = {
    SurfaceMaterial.DEFAULT.value: FRICTION_DEFAULT,
    SurfaceMaterial.CONCRETE.value: FRICTION_CONCRETE,
    SurfaceMaterial.METAL.value: FRICTION_METAL,
    SurfaceMaterial.WOOD.value: FRICTION_WOOD,
    SurfaceMaterial.GRASS.value: FRICTION_GRASS,
    SurfaceMaterial.SAND.value: FRICTION_SAND,
    SurfaceMaterial.ICE.value: FRICTION_ICE,
    SurfaceMaterial.MUD.value: FRICTION_MUD,
}


# =============================================================================
# Performance Limits
# =============================================================================

MAX_CHARACTERS_PER_FRAME: Final[int] = 128
MAX_RAGDOLL_BODIES: Final[int] = 32
MAX_ACTIVE_RAGDOLLS: Final[int] = 16
MAX_PLATFORMS_PER_CHARACTER: Final[int] = 4
