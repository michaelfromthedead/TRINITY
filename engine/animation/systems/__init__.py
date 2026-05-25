"""Animation ECS systems: animation graph, IK, procedural, skinning, motion matching, facial, crowd."""

from .animation_graph_system import (
    AnimationGraphComponent,
    AnimationGraphSystem,
    GraphParameter,
    ParameterType,
)
from .ik_system import (
    IKComponent,
    IKGoal,
    IKSolverType,
    IKSystem,
)
from .procedural_system import (
    ProceduralComponent,
    ProceduralController,
    SpringController,
    LookAtController,
    SwayController,
    BreathingController,
    ProceduralSystem,
)
from .skinning_system import (
    SkinnedMeshComponent,
    SkinningMethod,
    SkinningSystem,
    BoneInfluence,
)
from .motion_matching_system import (
    MotionMatchingComponent,
    MotionMatchingSystem,
    MotionInput,
    MotionFeature,
)
from .facial_system import (
    FacialComponent,
    FacialSystem,
    Expression,
    LipSyncPhoneme,
    EmotionState,
)
from .crowd_system import (
    CrowdComponent,
    CrowdSystem,
)

__all__ = [
    # Animation Graph System
    "AnimationGraphComponent",
    "AnimationGraphSystem",
    "GraphParameter",
    "ParameterType",
    # IK System
    "IKComponent",
    "IKGoal",
    "IKSolverType",
    "IKSystem",
    # Procedural System
    "ProceduralComponent",
    "ProceduralController",
    "SpringController",
    "LookAtController",
    "SwayController",
    "BreathingController",
    "ProceduralSystem",
    # Skinning System
    "SkinnedMeshComponent",
    "SkinningMethod",
    "SkinningSystem",
    "BoneInfluence",
    # Motion Matching System
    "MotionMatchingComponent",
    "MotionMatchingSystem",
    "MotionInput",
    "MotionFeature",
    # Facial System
    "FacialComponent",
    "FacialSystem",
    "Expression",
    "LipSyncPhoneme",
    "EmotionState",
    # Crowd System
    "CrowdComponent",
    "CrowdSystem",
]
