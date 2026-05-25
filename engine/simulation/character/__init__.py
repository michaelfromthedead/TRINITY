"""
Character Physics Module.

Provides comprehensive character physics functionality including:
- Character controller with move-and-slide collision
- Ground detection with coyote time support
- Movement modes (walking, running, crouching, etc.)
- Slope handling and step up/down
- Moving platform handling
- Ragdoll physics
- Active ragdoll with balance control
- Physics-animation blending
- Character interactions (push, grab, climb, vault)
"""

from .config import (
    # Capsule
    DEFAULT_CAPSULE_HEIGHT,
    DEFAULT_CAPSULE_RADIUS,
    DEFAULT_CROUCHED_HEIGHT,
    DEFAULT_PRONE_HEIGHT,
    # Step/Slope
    DEFAULT_STEP_HEIGHT,
    MAX_SLOPE_ANGLE,
    MIN_SLOPE_ANGLE,
    STEEP_SLOPE_ANGLE,
    # Collision
    GROUND_PROBE_DISTANCE,
    GROUND_SPHERE_PROBE_RADIUS,
    MAX_COLLISION_ITERATIONS,
    MAX_DEPENETRATION_VELOCITY,
    MIN_MOVE_DISTANCE,
    SKIN_WIDTH,
    # Movement
    AIR_CONTROL,
    DEFAULT_GRAVITY,
    DEFAULT_JUMP_VELOCITY,
    MAX_FALL_VELOCITY,
    # Ground Detection
    COYOTE_TIME_MS,
    JUMP_BUFFER_TIME_MS,
    LEDGE_DETECTION_HEIGHT,
    LEDGE_GRAB_DISTANCE,
    # Movement Speeds
    CLIMBING_SPEED,
    CROUCHING_SPEED,
    FLYING_SPEED,
    PRONE_SPEED,
    RUNNING_SPEED,
    SPRINTING_SPEED,
    SWIMMING_SPEED,
    WALKING_SPEED,
    MovementSpeed,
    # Acceleration
    AIR_ACCELERATION,
    AIR_DECELERATION,
    GROUND_ACCELERATION,
    GROUND_DECELERATION,
    TURN_ACCELERATION,
    # Platform
    MAX_PLATFORM_VELOCITY,
    PLATFORM_DETACH_THRESHOLD,
    PLATFORM_STICK_FORCE,
    # Ragdoll
    RAGDOLL_BLEND_TIME_MS,
    RAGDOLL_MIN_VELOCITY,
    RAGDOLL_RECOVERY_TIME_MS,
    RAGDOLL_SETTLED_TIME_MS,
    # Active Ragdoll
    BALANCE_THRESHOLD,
    DEFAULT_PD_KD,
    DEFAULT_PD_KP,
    MAX_TORQUE,
    # Blend
    BLEND_ADDITIVE,
    BLEND_CHAIN,
    BLEND_POSE,
    DEFAULT_BLEND_WEIGHT,
    HIT_REACTION_BLEND_IN_MS,
    HIT_REACTION_BLEND_OUT_MS,
    BlendMode,
    # Interaction
    CARRY_MASS_LIMIT,
    CLIMB_MAX_HEIGHT,
    GRAB_DISTANCE,
    PUSH_FORCE,
    THROW_FORCE_MULTIPLIER,
    VAULT_MAX_HEIGHT,
    # Collision Layers
    CollisionLayer,
    LAYER_CHARACTER,
    LAYER_DEFAULT,
    LAYER_DYNAMIC,
    LAYER_PLATFORM,
    LAYER_RAGDOLL,
    LAYER_STATIC,
    MASK_CHARACTER_MOVEMENT,
    MASK_GROUND_DETECTION,
    MASK_RAGDOLL,
    # Materials
    FRICTION_CONCRETE,
    FRICTION_DEFAULT,
    FRICTION_GRASS,
    FRICTION_ICE,
    FRICTION_METAL,
    FRICTION_MUD,
    FRICTION_SAND,
    FRICTION_WOOD,
    SURFACE_FRICTION,
    SurfaceMaterial,
    # Performance
    MAX_ACTIVE_RAGDOLLS,
    MAX_CHARACTERS_PER_FRAME,
    MAX_PLATFORMS_PER_CHARACTER,
    MAX_RAGDOLL_BODIES,
)

from .character_controller import (
    CharacterController,
    CharacterControllerConfig,
    CollisionHit,
    ControllerCollision,
    ControllerType,
    PhysicsWorldInterface,
    Quaternion,
    SweepResult,
    Transform,
    Vector3,
)

from .ground_detection import (
    GroundDetector,
    GroundInfo,
    GroundType,
    LedgeInfo,
)

from .movement_modes import (
    DEFAULT_MODE_PARAMS,
    MovementContext,
    MovementMode,
    MovementModeManager,
    MovementModeParams,
    MovementState,
    TRANSITION_RULES,
    TransitionRule,
)

from .slope_handling import (
    SlopeHandler,
    SlopeInfo,
    StepInfo,
)

from .platform_handling import (
    AttachmentMode,
    PlatformAttachment,
    PlatformData,
    PlatformHandler,
    PlatformProvider,
    PlatformType,
)

from .ragdoll import (
    BodyPartType,
    Ragdoll,
    RagdollBodyDef,
    RagdollBodyState,
    RagdollJointDef,
    RagdollPhysicsInterface,
    RagdollPose,
    RagdollSetup,
    RagdollState,
    SkeletonInterface,
    create_default_humanoid_setup,
)

from .active_ragdoll import (
    ActiveRagdoll,
    ActiveRagdollState,
    BalanceConfig,
    JointController,
    PDController,
    RecoveryBehavior,
)

from .physics_animation_blend import (
    BlendLayer,
    BonePose,
    HitReaction,
    PhysicsAnimationBlender,
    SkeletonPose,
)

from .character_interaction import (
    CharacterInteractionManager,
    ClimbInfo,
    GrabInfo,
    GrabState,
    InteractionTarget,
    InteractionType,
    VaultInfo,
)


__all__ = [
    # Config - Capsule
    "DEFAULT_CAPSULE_HEIGHT",
    "DEFAULT_CAPSULE_RADIUS",
    "DEFAULT_CROUCHED_HEIGHT",
    "DEFAULT_PRONE_HEIGHT",
    # Config - Step/Slope
    "DEFAULT_STEP_HEIGHT",
    "MAX_SLOPE_ANGLE",
    "MIN_SLOPE_ANGLE",
    "STEEP_SLOPE_ANGLE",
    # Config - Collision
    "GROUND_PROBE_DISTANCE",
    "GROUND_SPHERE_PROBE_RADIUS",
    "MAX_COLLISION_ITERATIONS",
    "MAX_DEPENETRATION_VELOCITY",
    "MIN_MOVE_DISTANCE",
    "SKIN_WIDTH",
    # Config - Movement
    "AIR_CONTROL",
    "DEFAULT_GRAVITY",
    "DEFAULT_JUMP_VELOCITY",
    "MAX_FALL_VELOCITY",
    # Config - Ground Detection
    "COYOTE_TIME_MS",
    "JUMP_BUFFER_TIME_MS",
    "LEDGE_DETECTION_HEIGHT",
    "LEDGE_GRAB_DISTANCE",
    # Config - Speeds
    "CLIMBING_SPEED",
    "CROUCHING_SPEED",
    "FLYING_SPEED",
    "MovementSpeed",
    "PRONE_SPEED",
    "RUNNING_SPEED",
    "SPRINTING_SPEED",
    "SWIMMING_SPEED",
    "WALKING_SPEED",
    # Config - Acceleration
    "AIR_ACCELERATION",
    "AIR_DECELERATION",
    "GROUND_ACCELERATION",
    "GROUND_DECELERATION",
    "TURN_ACCELERATION",
    # Config - Platform
    "MAX_PLATFORM_VELOCITY",
    "PLATFORM_DETACH_THRESHOLD",
    "PLATFORM_STICK_FORCE",
    # Config - Ragdoll
    "RAGDOLL_BLEND_TIME_MS",
    "RAGDOLL_MIN_VELOCITY",
    "RAGDOLL_RECOVERY_TIME_MS",
    "RAGDOLL_SETTLED_TIME_MS",
    # Config - Active Ragdoll
    "BALANCE_THRESHOLD",
    "DEFAULT_PD_KD",
    "DEFAULT_PD_KP",
    "MAX_TORQUE",
    # Config - Blend
    "BLEND_ADDITIVE",
    "BLEND_CHAIN",
    "BLEND_POSE",
    "BlendMode",
    "DEFAULT_BLEND_WEIGHT",
    "HIT_REACTION_BLEND_IN_MS",
    "HIT_REACTION_BLEND_OUT_MS",
    # Config - Interaction
    "CARRY_MASS_LIMIT",
    "CLIMB_MAX_HEIGHT",
    "GRAB_DISTANCE",
    "PUSH_FORCE",
    "THROW_FORCE_MULTIPLIER",
    "VAULT_MAX_HEIGHT",
    # Config - Collision Layers
    "CollisionLayer",
    "LAYER_CHARACTER",
    "LAYER_DEFAULT",
    "LAYER_DYNAMIC",
    "LAYER_PLATFORM",
    "LAYER_RAGDOLL",
    "LAYER_STATIC",
    "MASK_CHARACTER_MOVEMENT",
    "MASK_GROUND_DETECTION",
    "MASK_RAGDOLL",
    # Config - Materials
    "FRICTION_CONCRETE",
    "FRICTION_DEFAULT",
    "FRICTION_GRASS",
    "FRICTION_ICE",
    "FRICTION_METAL",
    "FRICTION_MUD",
    "FRICTION_SAND",
    "FRICTION_WOOD",
    "SURFACE_FRICTION",
    "SurfaceMaterial",
    # Config - Performance
    "MAX_ACTIVE_RAGDOLLS",
    "MAX_CHARACTERS_PER_FRAME",
    "MAX_PLATFORMS_PER_CHARACTER",
    "MAX_RAGDOLL_BODIES",
    # Character Controller
    "CharacterController",
    "CharacterControllerConfig",
    "CollisionHit",
    "ControllerCollision",
    "ControllerType",
    "PhysicsWorldInterface",
    "Quaternion",
    "SweepResult",
    "Transform",
    "Vector3",
    # Ground Detection
    "GroundDetector",
    "GroundInfo",
    "GroundType",
    "LedgeInfo",
    # Movement Modes
    "DEFAULT_MODE_PARAMS",
    "MovementContext",
    "MovementMode",
    "MovementModeManager",
    "MovementModeParams",
    "MovementState",
    "TRANSITION_RULES",
    "TransitionRule",
    # Slope Handling
    "SlopeHandler",
    "SlopeInfo",
    "StepInfo",
    # Platform Handling
    "AttachmentMode",
    "PlatformAttachment",
    "PlatformData",
    "PlatformHandler",
    "PlatformProvider",
    "PlatformType",
    # Ragdoll
    "BodyPartType",
    "Ragdoll",
    "RagdollBodyDef",
    "RagdollBodyState",
    "RagdollJointDef",
    "RagdollPhysicsInterface",
    "RagdollPose",
    "RagdollSetup",
    "RagdollState",
    "SkeletonInterface",
    "create_default_humanoid_setup",
    # Active Ragdoll
    "ActiveRagdoll",
    "ActiveRagdollState",
    "BalanceConfig",
    "JointController",
    "PDController",
    "RecoveryBehavior",
    # Physics Animation Blend
    "BlendLayer",
    "BonePose",
    "HitReaction",
    "PhysicsAnimationBlender",
    "SkeletonPose",
    # Character Interaction
    "CharacterInteractionManager",
    "ClimbInfo",
    "GrabInfo",
    "GrabState",
    "InteractionTarget",
    "InteractionType",
    "VaultInfo",
]
