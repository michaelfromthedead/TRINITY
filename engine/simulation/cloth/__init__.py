"""
Cloth simulation module.

Provides Position-Based Dynamics (PBD) cloth simulation with:
- Particle-based cloth representation
- Distance, bending, and shear constraints
- Collision with primitives and meshes
- Self-collision detection
- Aerodynamic wind forces
- GPU acceleration stubs
"""

from .cloth_collision import (
    BoxCollider,
    CapsuleCollider,
    ClothCollisionHandler,
    CollisionResult,
    MeshCollider,
    SDFCollider,
    SpatialHash,
    SphereCollider,
    collide_with_box,
    collide_with_capsule,
    collide_with_mesh,
    collide_with_sdf,
    collide_with_sphere,
    handle_self_collision,
)
from .cloth_constraints import (
    AnchorConstraint,
    BendingConstraint,
    DistanceConstraint,
    LongRangeAttachment,
    ShearConstraint,
    TetherConstraint,
    create_bend_constraints,
    create_long_range_attachments,
)
from .cloth_simulation import (
    ClothEdge,
    ClothMesh,
    ClothParticle,
    ClothSimulation,
    ClothSimulationConfig,
    ClothState,
    ClothTriangle,
    create_cloth_from_mesh,
    create_cloth_grid,
)
from .cloth_wind import (
    DirectionalWind,
    PointWind,
    VortexWind,
    WindForce,
    WindSettings,
    WindSystem,
)
from .config import (
    CLOTH_DAMPING,
    CLOTH_SOLVER_ITERATIONS,
    CLOTH_SUBSTEPS,
    CLOTH_TIMESTEP,
    COLLISION_FRICTION,
    COLLISION_MARGIN,
    DEFAULT_BEND_STIFFNESS,
    DEFAULT_SHEAR_STIFFNESS,
    DEFAULT_STRETCH_STIFFNESS,
    MAX_CLOTH_PARTICLES,
    SELF_COLLISION_THICKNESS,
    ClothQualityPreset,
)
from .gpu_cloth import (
    GPUBuffer,
    GPUBufferAccess,
    GPUBufferUsage,
    GPUClothBuffers,
    GPUClothPipelines,
    GPUClothSolver,
    GPUClothSolverStub,
    GPUComputePipeline,
    GPUDevice,
    calculate_workgroups,
    get_shader_templates,
)

__all__ = [
    # Config
    "CLOTH_DAMPING",
    "CLOTH_SOLVER_ITERATIONS",
    "CLOTH_SUBSTEPS",
    "CLOTH_TIMESTEP",
    "COLLISION_FRICTION",
    "COLLISION_MARGIN",
    "DEFAULT_BEND_STIFFNESS",
    "DEFAULT_SHEAR_STIFFNESS",
    "DEFAULT_STRETCH_STIFFNESS",
    "MAX_CLOTH_PARTICLES",
    "SELF_COLLISION_THICKNESS",
    "ClothQualityPreset",
    # Simulation
    "ClothEdge",
    "ClothMesh",
    "ClothParticle",
    "ClothSimulation",
    "ClothSimulationConfig",
    "ClothState",
    "ClothTriangle",
    "create_cloth_from_mesh",
    "create_cloth_grid",
    # Constraints
    "AnchorConstraint",
    "BendingConstraint",
    "DistanceConstraint",
    "LongRangeAttachment",
    "ShearConstraint",
    "TetherConstraint",
    "create_bend_constraints",
    "create_long_range_attachments",
    # Collision
    "BoxCollider",
    "CapsuleCollider",
    "ClothCollisionHandler",
    "CollisionResult",
    "MeshCollider",
    "SDFCollider",
    "SpatialHash",
    "SphereCollider",
    "collide_with_box",
    "collide_with_capsule",
    "collide_with_mesh",
    "collide_with_sdf",
    "collide_with_sphere",
    "handle_self_collision",
    # Wind
    "DirectionalWind",
    "PointWind",
    "VortexWind",
    "WindForce",
    "WindSettings",
    "WindSystem",
    # GPU
    "GPUBuffer",
    "GPUBufferAccess",
    "GPUBufferUsage",
    "GPUClothBuffers",
    "GPUClothPipelines",
    "GPUClothSolver",
    "GPUClothSolverStub",
    "GPUComputePipeline",
    "GPUDevice",
    "calculate_workgroups",
    "get_shader_templates",
]
