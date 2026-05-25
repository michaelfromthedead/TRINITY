"""Soft body and deformable simulation module.

This module provides:
- FEM (Finite Element Method) solver for accurate deformations
- Shape matching for fast geometric deformations
- Position-based soft body dynamics
- Muscle simulation with contraction
- Deformable mesh handling for rendering
"""

from .config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    VOLUME_STIFFNESS,
    SHAPE_MATCHING_STIFFNESS,
    MAX_DEFORMATION,
    SOFTBODY_SUBSTEPS,
)
from .fem_solver import (
    FEMSolver,
    TetrahedralMesh,
    MaterialModel,
    NeoHookeanMaterial,
    CorotationalMaterial,
)
from .shape_matching import (
    ShapeMatchingSolver,
    ShapeMatchingCluster,
    ClusterConfig,
)
from .soft_body_pbd import (
    PBDSoftBody,
    VolumeConstraint,
    StrainLimitConstraint,
    EdgeLengthConstraint,
    CollisionConstraint,
)
from .muscle import (
    Muscle,
    MuscleAttachment,
    MuscleGroup,
    MuscleFiber,
)
from .deformable_mesh import (
    DeformableMesh,
    EmbeddedSurface,
    TetSkinning,
)

__all__ = [
    # Config
    "DEFAULT_YOUNG_MODULUS",
    "DEFAULT_POISSON_RATIO",
    "VOLUME_STIFFNESS",
    "SHAPE_MATCHING_STIFFNESS",
    "MAX_DEFORMATION",
    "SOFTBODY_SUBSTEPS",
    # FEM
    "FEMSolver",
    "TetrahedralMesh",
    "MaterialModel",
    "NeoHookeanMaterial",
    "CorotationalMaterial",
    # Shape Matching
    "ShapeMatchingSolver",
    "ShapeMatchingCluster",
    "ClusterConfig",
    # PBD
    "PBDSoftBody",
    "VolumeConstraint",
    "StrainLimitConstraint",
    "EdgeLengthConstraint",
    "CollisionConstraint",
    # Muscle
    "Muscle",
    "MuscleAttachment",
    "MuscleGroup",
    "MuscleFiber",
    # Deformable Mesh
    "DeformableMesh",
    "EmbeddedSurface",
    "TetSkinning",
]
