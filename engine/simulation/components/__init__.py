"""
Simulation Components Module.

Provides physics-related components for the entity-component system including:
- Rigid body physics
- Collision shapes (sphere, box, capsule, mesh)
- Physics joints/constraints
- Cloth simulation
- Vehicle physics
- Character controllers
- Destruction physics
- Fluid volumes
"""

from .rigid_body_component import (
    ActivationState,
    CollisionEvent,
    RigidBodyComponent,
    RigidBodyConfig,
    RigidBodyType,
)

from .collider_components import (
    BoxCollider,
    CapsuleCollider,
    ColliderComponent,
    ColliderType,
    MeshCollider,
    PhysicsMaterial,
    SphereCollider,
)

from .joint_component import (
    JointComponent,
    JointDrive,
    JointLimits,
    JointMotion,
    JointType,
)

from .cloth_component import (
    ClothComponent,
    ClothConfig,
    ClothConstraint,
    ClothParticle,
    ClothSolverType,
    CollisionMode,
)

from .vehicle_component import (
    DriveType,
    EngineConfig,
    GearboxConfig,
    VehicleComponent,
    VehicleType,
    WheelConfig,
    WheelState,
)

from .character_component import (
    CharacterComponentConfig,
    CharacterControllerComponent,
)

from .destruction_component import (
    DamageInfo,
    DamageType,
    DestructibleComponent,
    DestructionConfig,
    DestructionType,
    FractureChunk,
)

from .fluid_component import (
    FlowConfig,
    FlowType,
    FluidConfig,
    FluidType,
    FluidVolumeComponent,
    SubmergedObject,
)


__all__ = [
    # Rigid Body
    "ActivationState",
    "CollisionEvent",
    "RigidBodyComponent",
    "RigidBodyConfig",
    "RigidBodyType",
    # Colliders
    "BoxCollider",
    "CapsuleCollider",
    "ColliderComponent",
    "ColliderType",
    "MeshCollider",
    "PhysicsMaterial",
    "SphereCollider",
    # Joints
    "JointComponent",
    "JointDrive",
    "JointLimits",
    "JointMotion",
    "JointType",
    # Cloth
    "ClothComponent",
    "ClothConfig",
    "ClothConstraint",
    "ClothParticle",
    "ClothSolverType",
    "CollisionMode",
    # Vehicle
    "DriveType",
    "EngineConfig",
    "GearboxConfig",
    "VehicleComponent",
    "VehicleType",
    "WheelConfig",
    "WheelState",
    # Character
    "CharacterComponentConfig",
    "CharacterControllerComponent",
    # Destruction
    "DamageInfo",
    "DamageType",
    "DestructibleComponent",
    "DestructionConfig",
    "DestructionType",
    "FractureChunk",
    # Fluid
    "FlowConfig",
    "FlowType",
    "FluidConfig",
    "FluidType",
    "FluidVolumeComponent",
    "SubmergedObject",
]
