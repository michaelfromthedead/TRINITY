"""Animation ECS systems: animation graph, IK, procedural, skinning, motion matching, facial, crowd."""

from .animation_graph_system import (
    # System decorator
    system,
    # SoA format
    BoneTransformSoA,
    # Dirty tracking
    DirtyFlags,
    AnimationDirtyState,
    # State machine output
    StateMachineOutput,
    # Component
    AnimationGraphComponent,
    # Re-exports
    GraphParameter,
    ParameterType,
    # Evaluators
    ClipSampler,
    BlendTreeEvaluator,
    # Main system
    AnimationGraphSystem,
)
from .ik_system import (
    # System decorator (also in animation_graph_system)
    system as ik_system_decorator,
    # Enums
    IKSolverType,
    IKHintType,
    # Data structures
    IKGoal,
    IKChainBone,
    IKSolveResult,
    # Component
    IKComponent,
    # System
    IKSystem,
    # Stats
    IKSystemStats,
)
from .procedural_system import (
    # Enumerations
    ProceduralEffectType,
    ControllerType,
    # Bone mask for per-bone control
    BoneMask,
    # Components and controllers
    ProceduralComponent,
    ProceduralController,
    SpringController,
    LookAtController,
    TwistController,
    RagdollBlendController,
    SwayController,
    BreathingController,
    # System
    ProceduralSystem,
    # Factory functions
    create_spring_chain_controller,
    create_lookat_controller,
    create_twist_controller,
    create_ragdoll_blend_controller,
)
from .skinning_system import (
    # Enums
    SkinningMethod,
    SkinningBackend,
    LODInfluenceLevel,
    # Data structures
    BoneInfluence,
    VertexSkinData,
    SkinningData,
    MeshData,
    GPUDispatchConfig,
    SkinningDispatch,
    SkinningBatch,
    SkinningStats,
    LODComponent,
    GPUBufferHandle,
    GPUCapabilities,
    # Components
    SkinnedMeshComponent,
    # Dispatcher and fallback
    GPUSkinningDispatcher,
    CPUSkinningFallback,
    # System
    SkinningSystem,
)
from .motion_matching_system import (
    # System decorator
    system as mm_system_decorator,
    # Enums
    FallbackReason,
    MotionMatchingMode,
    # Configuration
    MotionMatchingConfig,
    MotionMatchingStatistics,
    # Components
    MotionMatchingComponent,
    MotionMatchingInput,
    TrajectoryState,
    # Legacy compatibility
    MotionInput,
    MotionFeature,
    # System
    MotionMatchingSystem,
    # Protocols
    PoseProvider,
)
from .facial_system import (
    # System decorator
    system as facial_system_decorator,
    # Enumerations
    EmotionState,
    LipSyncPhoneme,
    FacialRegion,
    FacialLayerPriority,
    BlendMode,
    # Data structures
    Expression,
    FacialLayer,
    FaceRig,
    LipSyncState,
    EyeState,
    FACSState,
    AudioSyncData,
    # Component
    FacialComponent,
    # System
    FacialSystem,
    # Factory functions
    create_default_face_rig,
    create_facial_component,
)
from .crowd_system import (
    # System decorator
    system as crowd_system_decorator,
    # Enums
    SteeringMode,
    CullingMode,
    AnimationBakeMode,
    # Frustum culling
    Plane,
    Frustum,
    # RVO/ORCA steering
    VelocityObstacle,
    ORCALine,
    RVOConfig,
    RVOSteering,
    # Instance buffer
    CrowdInstanceData,
    CrowdInstanceBuffer,
    # Components
    CrowdComponent,
    # System
    CrowdSystem,
)

__all__ = [
    # Animation Graph System
    "system",
    "BoneTransformSoA",
    "DirtyFlags",
    "AnimationDirtyState",
    "StateMachineOutput",
    "AnimationGraphComponent",
    "AnimationGraphSystem",
    "GraphParameter",
    "ParameterType",
    "ClipSampler",
    "BlendTreeEvaluator",
    # IK System
    "IKSolverType",
    "IKHintType",
    "IKGoal",
    "IKChainBone",
    "IKSolveResult",
    "IKComponent",
    "IKSystem",
    "IKSystemStats",
    # Procedural System
    "ProceduralEffectType",
    "ControllerType",
    "BoneMask",
    "ProceduralComponent",
    "ProceduralController",
    "SpringController",
    "LookAtController",
    "TwistController",
    "RagdollBlendController",
    "SwayController",
    "BreathingController",
    "ProceduralSystem",
    "create_spring_chain_controller",
    "create_lookat_controller",
    "create_twist_controller",
    "create_ragdoll_blend_controller",
    # Skinning System
    "SkinningMethod",
    "SkinningBackend",
    "LODInfluenceLevel",
    "BoneInfluence",
    "VertexSkinData",
    "SkinningData",
    "MeshData",
    "GPUDispatchConfig",
    "SkinningDispatch",
    "SkinningBatch",
    "SkinningStats",
    "LODComponent",
    "GPUBufferHandle",
    "GPUCapabilities",
    "SkinnedMeshComponent",
    "GPUSkinningDispatcher",
    "CPUSkinningFallback",
    "SkinningSystem",
    # Motion Matching System
    "FallbackReason",
    "MotionMatchingMode",
    "MotionMatchingConfig",
    "MotionMatchingStatistics",
    "MotionMatchingComponent",
    "MotionMatchingInput",
    "TrajectoryState",
    "MotionInput",
    "MotionFeature",
    "MotionMatchingSystem",
    "PoseProvider",
    # Facial System
    "EmotionState",
    "LipSyncPhoneme",
    "FacialRegion",
    "FacialLayerPriority",
    "BlendMode",
    "Expression",
    "FacialLayer",
    "FaceRig",
    "LipSyncState",
    "EyeState",
    "FACSState",
    "AudioSyncData",
    "FacialComponent",
    "FacialSystem",
    "create_default_face_rig",
    "create_facial_component",
    # Crowd System
    "SteeringMode",
    "CullingMode",
    "AnimationBakeMode",
    "Plane",
    "Frustum",
    "VelocityObstacle",
    "ORCALine",
    "RVOConfig",
    "RVOSteering",
    "CrowdInstanceData",
    "CrowdInstanceBuffer",
    "CrowdComponent",
    "CrowdSystem",
]
