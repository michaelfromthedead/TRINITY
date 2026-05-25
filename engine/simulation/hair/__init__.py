"""
Hair simulation module.

Provides hair simulation using Follow-The-Leader (FTL) and Position-Based Dynamics:
- Guide hair simulation
- Interpolated hair generation
- Body collision handling
- Self-collision via density field
- Level of detail management
"""

from .config import (
    COLLISION_STIFFNESS,
    DEFAULT_HAIR_LENGTH,
    DEFAULT_HAIR_THICKNESS,
    DEFAULT_STRAND_SEGMENTS,
    HAIR_AIR_RESISTANCE,
    HAIR_COLLISION_MARGIN,
    HAIR_DAMPING,
    HAIR_SOLVER_ITERATIONS,
    HAIR_TIMESTEP,
    INTERPOLATION_RATIO,
    LENGTH_STIFFNESS,
    LOCAL_SHAPE_STIFFNESS,
    LOD_DISTANCE_HIGH,
    LOD_DISTANCE_LOW,
    LOD_DISTANCE_MEDIUM,
    LOD_DISTANCE_SHELL,
    MAX_GUIDE_HAIRS,
    MAX_INTERPOLATED_HAIRS,
    ROOT_STIFFNESS,
    SELF_COLLISION_RADIUS,
    SHAPE_STIFFNESS,
    HairQualityPreset,
)
from .hair_collision import (
    CapsuleCollider,
    HairCollisionResult,
    HairCollisionSystem,
    HairDensityField,
    SphereCollider,
    collide_point_with_capsule,
    collide_point_with_sphere,
    collide_strands,
    collide_with_sdf,
)
from .hair_constraints import (
    CollisionConstraint,
    GlobalShapeConstraint,
    LengthConstraint,
    LocalShapeConstraint,
    RootConstraint,
    create_length_constraints,
    create_local_shape_constraints,
    solve_global_shape_constraint,
    solve_length_constraint,
    solve_local_shape_constraint,
)
from .hair_lod import (
    HairLODLevel,
    HairLODSystem,
    LODSettings,
    LODState,
    LODTransition,
    create_lod_interpolated_hairs,
)
from .hair_simulation import (
    GuideHair,
    HairControlPoint,
    HairSimulation,
    HairSimulationConfig,
    HairState,
    HairStrand,
    InterpolatedHair,
    create_hair_from_scalp,
    create_hair_strand,
    create_interpolated_hairs,
)

__all__ = [
    # Config
    "COLLISION_STIFFNESS",
    "DEFAULT_HAIR_LENGTH",
    "DEFAULT_HAIR_THICKNESS",
    "DEFAULT_STRAND_SEGMENTS",
    "HAIR_AIR_RESISTANCE",
    "HAIR_COLLISION_MARGIN",
    "HAIR_DAMPING",
    "HAIR_SOLVER_ITERATIONS",
    "HAIR_TIMESTEP",
    "INTERPOLATION_RATIO",
    "LENGTH_STIFFNESS",
    "LOCAL_SHAPE_STIFFNESS",
    "LOD_DISTANCE_HIGH",
    "LOD_DISTANCE_LOW",
    "LOD_DISTANCE_MEDIUM",
    "LOD_DISTANCE_SHELL",
    "MAX_GUIDE_HAIRS",
    "MAX_INTERPOLATED_HAIRS",
    "ROOT_STIFFNESS",
    "SELF_COLLISION_RADIUS",
    "SHAPE_STIFFNESS",
    "HairQualityPreset",
    # Simulation
    "GuideHair",
    "HairControlPoint",
    "HairSimulation",
    "HairSimulationConfig",
    "HairState",
    "HairStrand",
    "InterpolatedHair",
    "create_hair_from_scalp",
    "create_hair_strand",
    "create_interpolated_hairs",
    # Constraints
    "CollisionConstraint",
    "GlobalShapeConstraint",
    "LengthConstraint",
    "LocalShapeConstraint",
    "RootConstraint",
    "create_length_constraints",
    "create_local_shape_constraints",
    "solve_global_shape_constraint",
    "solve_length_constraint",
    "solve_local_shape_constraint",
    # Collision
    "CapsuleCollider",
    "HairCollisionResult",
    "HairCollisionSystem",
    "HairDensityField",
    "SphereCollider",
    "collide_point_with_capsule",
    "collide_point_with_sphere",
    "collide_strands",
    "collide_with_sdf",
    # LOD
    "HairLODLevel",
    "HairLODSystem",
    "LODSettings",
    "LODState",
    "LODTransition",
    "create_lod_interpolated_hairs",
]
