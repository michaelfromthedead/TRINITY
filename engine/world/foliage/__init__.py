"""
Foliage system for the game engine World Layer.

Provides foliage type definitions, procedural and manual placement,
hierarchical instanced mesh management, and specialized grass rendering.

Main Components:
- types: Foliage type definitions and registry
- placement: Procedural and manual placement systems
- instances: HISM pattern for efficient instance management
- grass: Specialized grass rendering and streaming

Example Usage:
    from engine.world.foliage import (
        FoliageType,
        FoliageCategory,
        GrassType,
        FoliageTypeRegistry,
        FoliagePlacement,
        PlacementRule,
        FoliageManager,
        LandscapeGrass,
        GrassSettings,
    )

    # Register a foliage type
    registry = FoliageTypeRegistry()
    tree = FoliageType(
        type_id="oak_tree",
        category=FoliageCategory.TREE,
        mesh_id="meshes/oak_tree",
        has_collision=True,
    )
    registry.register(tree)

    # Setup grass system
    settings = GrassSettings(
        density_scale=1.0,
        wind_sway_amount=1.5,
        cull_distance=100.0,
    )
    landscape_grass = LandscapeGrass(settings)
"""

# Types
from .types import (
    CollisionType,
    DebrisType,
    FoliageCategory,
    FoliageType,
    FoliageTypeRegistry,
    GrassType,
    RockType,
    ShrubType,
    TreeType,
    foliage_type,
    get_global_registry,
)

# Placement
from .placement import (
    Bounds,
    FoliagePlacement,
    ManualPlacement,
    NoiseGenerator,
    PlacementResult,
    PlacementRule,
    ProceduralPlacer,
    TerrainInterface,
    procedural_placement,
)

# Instances
from .instances import (
    BatchedDescriptor,
    FoliageCluster,
    FoliageInstance,
    FoliageManager,
    Frustum,
    HierarchicalInstancedMesh,
)

# Grass
from .grass import (
    GrassChunk,
    GrassInstance,
    GrassRenderer,
    GrassSettings,
    LandscapeGrass,
    ProceduralGrass,
)

__all__ = [
    # Types
    "FoliageCategory",
    "CollisionType",
    "FoliageType",
    "TreeType",
    "ShrubType",
    "GrassType",
    "RockType",
    "DebrisType",
    "FoliageTypeRegistry",
    "get_global_registry",
    "foliage_type",
    # Placement
    "PlacementRule",
    "PlacementResult",
    "Bounds",
    "TerrainInterface",
    "NoiseGenerator",
    "ProceduralPlacer",
    "FoliagePlacement",
    "ManualPlacement",
    "procedural_placement",
    # Instances
    "FoliageInstance",
    "Frustum",
    "FoliageCluster",
    "HierarchicalInstancedMesh",
    "FoliageManager",
    "BatchedDescriptor",
    # Grass
    "GrassSettings",
    "GrassInstance",
    "GrassChunk",
    "ProceduralGrass",
    "LandscapeGrass",
    "GrassRenderer",
]
