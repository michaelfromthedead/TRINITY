"""
Terrain Core system for the game engine World Layer.

This module provides heightfield-based terrain with LOD support:

Heightfield (heightfield.py):
- HeightfieldPrecision: Precision modes (16-bit, 32-bit)
- HeightfieldConfig: Configuration dataclass
- Heightfield: 2D height data with bilinear interpolation

Patches (patch.py):
- TerrainPatch: Single terrain patch with LOD support

Components (component.py):
- LandscapeComponent: Render unit for terrain
- TerrainSection: LOD unit within a component
- TerrainProxy: Collision unit for terrain
- RaycastHit: Raycast result data
- TerrainActor: High-level terrain manager

Sculpting (sculpting.py):
- SculptTool: Enum of available sculpting tools
- BrushSettings: Configuration for sculpting brushes
- TerrainBrush: Brush for sculpting operations
- RaiseTool, LowerTool, SmoothTool, etc.: Tool implementations
- SculptingSession: Manages undo/redo for sculpting

Materials (materials.py):
- TerrainLayer: Material layer configuration
- WeightMap: Stores blend weights for layers
- AutoLayerRule: Rule for automatic layer application
- TerrainMaterial: Manages layers and blending

LOD (lod.py):
- TerrainLODMethod: LOD methods (QUADTREE, CLIPMAPS, etc.)
- TerrainChunk: Chunk with LOD information
- TerrainQuadtree: Quadtree-based LOD selection
- TerrainLODSystem: Main LOD management system

Features (features.py):
- TerrainHole: Defines holes in terrain
- TerrainSpline: Base spline class
- RoadSpline, RiverSpline: Specialized splines
- TerrainDeformer: Applies terrain deformations
- TerrainCollision: Collision detection and queries

Example Usage:
    from engine.world.terrain import (
        Heightfield,
        HeightfieldConfig,
        HeightfieldPrecision,
        TerrainPatch,
        LandscapeComponent,
        TerrainActor,
    )

    # Create a heightfield
    config = HeightfieldConfig(resolution=65, scale=1.0)
    heightfield = Heightfield(config)

    # Set some heights
    for x in range(65):
        for z in range(65):
            heightfield.set_height_at(x, z, x * 0.1 + z * 0.1)

    # Create a terrain patch
    patch = TerrainPatch(patch_x=0, patch_y=0, heightfield=heightfield)

    # Create a landscape component
    component = LandscapeComponent(
        bounds=patch.get_world_bounds(),
        patch=patch,
        material_id="grass"
    )

    # Create terrain actor and add component
    actor = TerrainActor()
    actor.add_component(component)

    # Query terrain height
    height = actor.get_height_at(32.0, 32.0)
"""

from .heightfield import (
    HeightfieldPrecision,
    HeightfieldConfig,
    Heightfield,
)

from .patch import (
    TerrainPatch,
    AABB,
)

from .component import (
    LandscapeComponent,
    TerrainSection,
    TerrainProxy,
    RaycastHit,
    TerrainActor,
)

from .sculpting import (
    SculptTool,
    BrushShape,
    BrushSettings,
    TerrainBrush,
    HeightDelta,
    BaseSculptTool,
    RaiseTool,
    LowerTool,
    SmoothTool,
    FlattenTool,
    ErosionTool,
    NoiseTool,
    RampTool,
    SculptingSession,
    create_tool,
)

from .materials import (
    TerrainLayerType,
    BlendTechnique,
    TerrainLayer,
    WeightMap,
    AutoLayerRule,
    TerrainMaterial,
    MaterialPalette,
)

from .lod import (
    TerrainLODMethod,
    LODStitchMethod,
    BoundingBox,
    TerrainChunk,
    QuadtreeNode,
    TerrainQuadtree,
    ClipmapRing,
    TerrainLODSystem,
)

from .constants import (
    # Heightfield constants
    DEFAULT_RESOLUTION,
    DEFAULT_HEIGHT_RANGE,
    DEFAULT_SCALE,
    MIN_RESOLUTION,
    BITS_16_MAX_VALUE,
    HEIGHT_EPSILON,
    NORMAL_EPSILON,
    ZLIB_COMPRESSION_LEVEL,
    COMPRESSED_HEADER_SIZE,
    # LOD constants
    DEFAULT_LOD_LEVELS,
    DEFAULT_LOD_DISTANCES,
    MIN_LOD_LEVELS,
    # Component constants
    DEFAULT_BOUNDS,
    LOD_BIAS_MIN,
    LOD_BIAS_MAX,
    # Physics constants
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    PHYSICS_COEFF_MIN,
    PHYSICS_COEFF_MAX,
    # Raycast constants
    DEFAULT_RAYCAST_MAX_DISTANCE,
    RAY_DIRECTION_EPSILON,
    RAYCAST_STEP_MULTIPLIER,
    RAYCAST_BINARY_SEARCH_ITERATIONS,
)

from .features import (
    TerrainHole,
    TerrainHoleManager,
    SplinePoint,
    TerrainSpline,
    RoadSpline,
    RiverSpline,
    DeformationSettings,
    TerrainDeformer,
    PhysicalMaterialMapping,
    HitResult,
    TerrainCollision,
)

__all__ = [
    # Constants
    'DEFAULT_RESOLUTION',
    'DEFAULT_HEIGHT_RANGE',
    'DEFAULT_SCALE',
    'MIN_RESOLUTION',
    'BITS_16_MAX_VALUE',
    'HEIGHT_EPSILON',
    'NORMAL_EPSILON',
    'ZLIB_COMPRESSION_LEVEL',
    'COMPRESSED_HEADER_SIZE',
    'DEFAULT_LOD_LEVELS',
    'DEFAULT_LOD_DISTANCES',
    'MIN_LOD_LEVELS',
    'DEFAULT_BOUNDS',
    'LOD_BIAS_MIN',
    'LOD_BIAS_MAX',
    'DEFAULT_FRICTION',
    'DEFAULT_RESTITUTION',
    'PHYSICS_COEFF_MIN',
    'PHYSICS_COEFF_MAX',
    'DEFAULT_RAYCAST_MAX_DISTANCE',
    'RAY_DIRECTION_EPSILON',
    'RAYCAST_STEP_MULTIPLIER',
    'RAYCAST_BINARY_SEARCH_ITERATIONS',
    # Heightfield
    'HeightfieldPrecision',
    'HeightfieldConfig',
    'Heightfield',
    # Patch
    'TerrainPatch',
    'AABB',
    # Components
    'LandscapeComponent',
    'TerrainSection',
    'TerrainProxy',
    'RaycastHit',
    'TerrainActor',
    # Sculpting
    'SculptTool',
    'BrushShape',
    'BrushSettings',
    'TerrainBrush',
    'HeightDelta',
    'BaseSculptTool',
    'RaiseTool',
    'LowerTool',
    'SmoothTool',
    'FlattenTool',
    'ErosionTool',
    'NoiseTool',
    'RampTool',
    'SculptingSession',
    'create_tool',
    # Materials
    'TerrainLayerType',
    'BlendTechnique',
    'TerrainLayer',
    'WeightMap',
    'AutoLayerRule',
    'TerrainMaterial',
    'MaterialPalette',
    # LOD
    'TerrainLODMethod',
    'LODStitchMethod',
    'BoundingBox',
    'TerrainChunk',
    'QuadtreeNode',
    'TerrainQuadtree',
    'ClipmapRing',
    'TerrainLODSystem',
    # Features
    'TerrainHole',
    'TerrainHoleManager',
    'SplinePoint',
    'TerrainSpline',
    'RoadSpline',
    'RiverSpline',
    'DeformationSettings',
    'TerrainDeformer',
    'PhysicalMaterialMapping',
    'HitResult',
    'TerrainCollision',
]
