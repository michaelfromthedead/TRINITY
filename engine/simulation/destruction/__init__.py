"""
Destruction System Module.

Provides comprehensive destruction simulation including:
- Damage types and resistance systems
- Multiple fracture patterns (Voronoi, radial, slice)
- Support structure analysis
- Debris management with pooling

Example usage:
    from engine.simulation.destruction import (
        DestructionSystem,
        Damage,
        DamageType,
        DamageResistance,
        FracturePattern,
    )

    # Create destruction system
    system = DestructionSystem()

    # Register a destructible object
    destructible_id = system.register_destructible(
        vertices=mesh_vertices,
        triangles=mesh_triangles,
        health=100.0,
        resistance=DamageResistance.from_dict({
            'EXPLOSIVE': 0.5,  # 50% resistance to explosives
            'IMPACT': 1.0,    # Normal impact damage
        }),
        fracture_pattern=FracturePattern.VORONOI
    )

    # Apply damage
    damage = Damage(
        amount=50.0,
        damage_type=DamageType.IMPACT,
        position=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0)
    )
    result = system.apply_damage(destructible_id, damage, immediate=True)

    # Update system each frame
    system.update(dt=0.016, camera_position=(0, 0, 10))
"""

# Configuration
from .config import (
    # Constants
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    MIN_VORONOI_SITES,
    MAX_VORONOI_SITES,
    DEBRIS_LIFETIME,
    MAX_ACTIVE_DEBRIS,
    DAMAGE_PROPAGATION_FACTOR,
    SUPPORT_STRESS_THRESHOLD,
    DEBRIS_ANGULAR_VELOCITY_MIN,
    DEBRIS_ANGULAR_VELOCITY_MAX,
    DEBRIS_IMPORTANCE_VOLUME_MULTIPLIER,
    DEBRIS_LOD_DISTANCE_FULL,
    DEBRIS_LOD_DISTANCE_REDUCED,
    DEBRIS_LOD_DISTANCE_SIMPLE,
    DEBRIS_MIN_VELOCITY,
    DEBRIS_SLEEP_TIME,
    FRACTURE_VELOCITY_MULTIPLIER,
    FRACTURE_SPREAD_MULTIPLIER,
    SURFACE_SAMPLE_ITERATIONS,
    DEGENERATE_TRIANGLE_AREA_THRESHOLD,
    # Enums
    FracturePattern,
    DebrisState,
    SupportType,
    # Config dataclasses
    FractureConfig,
    DebrisConfig,
    DamageConfig,
    SupportConfig,
    DestructionSystemConfig,
    DEFAULT_CONFIG,
)

# Damage types
from .damage_types import (
    DamageType,
    Damage,
    DamageResistance,
    DamageTypeProperties,
    DamageResult,
    DamageAccumulator,
    DAMAGE_TYPE_PROPERTIES,
    get_damage_type_properties,
    apply_damage_modifiers,
)

# Voronoi fracture
from .fracture_voronoi import (
    # Types
    Vec3,
    Triangle,
    Plane,
    BoundingBox,
    Chunk,
    VoronoiCell,
    SiteDistribution,
    # Classes
    VoronoiFracture,
    TetrahedralVoronoiFracture,
    # Utilities
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_dot,
    vec3_cross,
    vec3_length,
    vec3_normalize,
    vec3_distance,
    vec3_lerp,
    triangle_area,
    is_degenerate_triangle,
)

# Radial fracture
from .fracture_radial import (
    RadialSlice,
    ConcentricRing,
    RadialChunk,
    RadialFracture,
    ConcentricRadialFracture,
    SpiderWebFracture,
)

# Slice fracture
from .fracture_slice import (
    SliceResult,
    CappedMesh,
    SliceFracture,
    AdaptiveSliceFracture,
    HierarchicalSliceFracture,
)

# Support graph
from .support_graph import (
    SupportNode,
    SupportEdge,
    SupportGraph,
    build_support_graph_from_chunks,
)

# Debris
from .debris import (
    DebrisLOD,
    Debris,
    DebrisSpawnParams,
    DebrisPool,
    DebrisManager,
    spawn_debris_from_fracture,
)

# Main system
from .destruction_system import (
    DestructibleState,
    Destructible,
    FractureRequest,
    DamageEvent,
    FractureEvent,
    DestructionSystem,
)


__all__ = [
    # Configuration
    'DEFAULT_FRACTURE_SEED',
    'MIN_CHUNK_VOLUME',
    'MAX_CHUNKS_PER_OBJECT',
    'MIN_VORONOI_SITES',
    'MAX_VORONOI_SITES',
    'DEBRIS_LIFETIME',
    'MAX_ACTIVE_DEBRIS',
    'DAMAGE_PROPAGATION_FACTOR',
    'SUPPORT_STRESS_THRESHOLD',
    'DEBRIS_ANGULAR_VELOCITY_MIN',
    'DEBRIS_ANGULAR_VELOCITY_MAX',
    'DEBRIS_IMPORTANCE_VOLUME_MULTIPLIER',
    'DEBRIS_LOD_DISTANCE_FULL',
    'DEBRIS_LOD_DISTANCE_REDUCED',
    'DEBRIS_LOD_DISTANCE_SIMPLE',
    'DEBRIS_MIN_VELOCITY',
    'DEBRIS_SLEEP_TIME',
    'FRACTURE_VELOCITY_MULTIPLIER',
    'FRACTURE_SPREAD_MULTIPLIER',
    'SURFACE_SAMPLE_ITERATIONS',
    'DEGENERATE_TRIANGLE_AREA_THRESHOLD',
    'FracturePattern',
    'DebrisState',
    'SupportType',
    'FractureConfig',
    'DebrisConfig',
    'DamageConfig',
    'SupportConfig',
    'DestructionSystemConfig',
    'DEFAULT_CONFIG',

    # Damage types
    'DamageType',
    'Damage',
    'DamageResistance',
    'DamageTypeProperties',
    'DamageResult',
    'DamageAccumulator',
    'DAMAGE_TYPE_PROPERTIES',
    'get_damage_type_properties',
    'apply_damage_modifiers',

    # Voronoi fracture
    'Vec3',
    'Triangle',
    'Plane',
    'BoundingBox',
    'Chunk',
    'VoronoiCell',
    'SiteDistribution',
    'VoronoiFracture',
    'TetrahedralVoronoiFracture',
    'vec3_add',
    'vec3_sub',
    'vec3_mul',
    'vec3_dot',
    'vec3_cross',
    'vec3_length',
    'vec3_normalize',
    'vec3_distance',
    'vec3_lerp',
    'triangle_area',
    'is_degenerate_triangle',

    # Radial fracture
    'RadialSlice',
    'ConcentricRing',
    'RadialChunk',
    'RadialFracture',
    'ConcentricRadialFracture',
    'SpiderWebFracture',

    # Slice fracture
    'SliceResult',
    'CappedMesh',
    'SliceFracture',
    'AdaptiveSliceFracture',
    'HierarchicalSliceFracture',

    # Support graph
    'SupportNode',
    'SupportEdge',
    'SupportGraph',
    'build_support_graph_from_chunks',

    # Debris
    'DebrisLOD',
    'Debris',
    'DebrisSpawnParams',
    'DebrisPool',
    'DebrisManager',
    'spawn_debris_from_fracture',

    # Main system
    'DestructibleState',
    'Destructible',
    'FractureRequest',
    'DamageEvent',
    'FractureEvent',
    'DestructionSystem',
]
