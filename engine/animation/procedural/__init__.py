"""
Procedural Animation Subsystem.

This module provides procedural animation components including:
- Spring/jiggle physics for bones (hair, cloth, accessories)
- Look-at/aim controllers for head and eye tracking
- Twist bone distribution for realistic joint deformation
- Ragdoll physics integration for physics-driven animation
- Procedural locomotion for walk/run cycle generation
- Breathing animation for natural character motion
- Secondary motion effects (delay, oscillation, noise, impulse)

Usage:
    from engine.animation.procedural import (
        SpringBone, SpringChain,
        LookAtController,
        TwistBone,
        Ragdoll, RagdollConfig, RagdollBody, RagdollJoint,
        ProceduralLocomotion, GaitConfig,
        BreathingController,
        SecondaryMotion, DelayedMotion, OscillatingMotion, NoiseMotion, ImpulseResponse
    )
"""

from engine.animation.procedural.spring_bone import (
    SpringBone,
    SpringChain,
    CollisionSphere,
    CollisionCapsule,
    WindForce,
)

from engine.animation.procedural.lookat import (
    LookAtController,
    InterestPoint,
    SaccadeGenerator,
)

from engine.animation.procedural.twist import (
    TwistBone,
    TwistDistribution,
)

from engine.animation.procedural.ragdoll import (
    Ragdoll,
    RagdollConfig,
    RagdollBody,
    RagdollJoint,
    JointLimits,
    JointMotor,
    CollisionGroup,
)

from engine.animation.procedural.locomotion import (
    ProceduralLocomotion,
    GaitConfig,
    FootTrajectory,
    BodyDynamics,
)

from engine.animation.procedural.breathing import (
    BreathingController,
    BreathPhase,
    ExertionLevel,
)

from engine.animation.procedural.secondary_motion import (
    SecondaryMotion,
    DelayedMotion,
    OscillatingMotion,
    NoiseMotion,
    ImpulseResponse,
    MotionComposer,
)

from engine.animation.procedural.config import (
    ProceduralConfig,
    SpringPhysicsConfig,
    LookAtConfig,
    LocomotionConfig,
    BreathingConfig,
)

__all__ = [
    # Spring bone
    "SpringBone",
    "SpringChain",
    "CollisionSphere",
    "CollisionCapsule",
    "WindForce",
    # Look-at
    "LookAtController",
    "InterestPoint",
    "SaccadeGenerator",
    # Twist
    "TwistBone",
    "TwistDistribution",
    # Ragdoll
    "Ragdoll",
    "RagdollConfig",
    "RagdollBody",
    "RagdollJoint",
    "JointLimits",
    "JointMotor",
    "CollisionGroup",
    # Locomotion
    "ProceduralLocomotion",
    "GaitConfig",
    "FootTrajectory",
    "BodyDynamics",
    # Breathing
    "BreathingController",
    "BreathPhase",
    "ExertionLevel",
    # Secondary motion
    "SecondaryMotion",
    "DelayedMotion",
    "OscillatingMotion",
    "NoiseMotion",
    "ImpulseResponse",
    "MotionComposer",
    # Configuration
    "ProceduralConfig",
    "SpringPhysicsConfig",
    "LookAtConfig",
    "LocomotionConfig",
    "BreathingConfig",
]
