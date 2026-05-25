"""
Collision Detection Module.

This module provides a complete collision detection system for the game engine,
including broadphase culling, narrowphase algorithms, contact management,
continuous collision detection, and event handling.

Submodules:
    - config: Configuration constants and quality presets
    - broadphase: Spatial partitioning structures (SAP, BVH, Grid, Octree)
    - narrowphase: Precise collision algorithms (GJK, EPA, SAT)
    - contact_manifold: Contact point management and persistence
    - ccd: Continuous collision detection for fast-moving objects
    - collision_filter: Layer-based collision filtering
    - collision_events: Event system for collision callbacks

Example usage:
    from engine.simulation.collision import (
        DynamicBVH, BroadphaseType, create_broadphase,
        Sphere, Box, test_collision,
        ContactManifold, ManifoldCache,
        CCDManager, CCDMode,
        CollisionFilter, CollisionLayer,
        CollisionEventDispatcher, CollisionEventType,
    )

    # Create broadphase
    broadphase = create_broadphase(BroadphaseType.BVH)

    # Insert objects
    id1 = broadphase.insert(AABB(Vec3(-1,-1,-1), Vec3(1,1,1)))
    id2 = broadphase.insert(AABB(Vec3(0,0,0), Vec3(2,2,2)))

    # Query overlaps
    pairs = broadphase.query_overlaps()

    # Narrowphase test
    sphere = Sphere(center=Vec3(0,0,0), radius=1.0)
    box = Box(center=Vec3(1,0,0), half_extents=Vec3(0.5,0.5,0.5))
    result = test_collision(sphere, box)

    if result.colliding:
        print(f"Collision! Depth: {result.depth}, Normal: {result.normal}")
"""

# Configuration
from .config import (
    BROADPHASE_MARGIN,
    CONTACT_TOLERANCE,
    MAX_CONTACT_POINTS,
    CCD_THRESHOLD_VELOCITY,
    MAX_CCD_ITERATIONS,
    SPATIAL_HASH_CELL_SIZE,
    GJK_MAX_ITERATIONS,
    EPA_MAX_ITERATIONS,
    EPA_TOLERANCE,
    CONTACT_MAX_AGE,
    WARM_START_FACTOR,
    NUMERICAL_EPSILON,
    PARALLEL_THRESHOLD,
    CCD_SAFETY_FACTOR,
    CollisionQuality,
    CollisionConfig,
    DEFAULT_CONFIG,
)

# Broadphase
from .broadphase import (
    Vec3,
    AABB,
    Ray,
    CollisionPair,
    RaycastHit,
    BroadphaseType,
    Broadphase,
    SweepAndPrune,
    DynamicBVH,
    SpatialHashGrid,
    Octree,
    create_broadphase,
)

# Narrowphase
from .narrowphase import (
    NarrowphaseAlgorithm,
    ShapeType,
    ContactResult,
    Sphere,
    Capsule,
    Box,
    ConvexHull,
    gjk_distance,
    epa_penetration,
    sat_test,
    sphere_sphere,
    sphere_capsule,
    capsule_capsule,
    box_box,
    sphere_box,
    capsule_box,
    collide_shapes,
)

# Contact Manifold
from .contact_manifold import (
    ContactPoint,
    ManifoldKey,
    ContactManifold,
    ManifoldCache,
    ContactPair,
    create_contact_pairs,
)

# Continuous Collision Detection
from .ccd import (
    CCDMode,
    CCDResult,
    MotionState,
    linear_sweep_sphere,
    linear_sweep_capsule,
    linear_sweep_box,
    linear_sweep_test,
    time_of_impact,
    time_of_impact_sphere_sphere,
    conservative_advancement,
    speculative_contacts,
    CCDManager,
)

# Collision Filtering
from .collision_filter import (
    CollisionLayer,
    CollisionMask,
    CollisionFilter,
    should_collide,
    create_layer_matrix,
    CollisionFilterManager,
    FilterPresets,
)

# Collision Events
from .collision_events import (
    CollisionEventType,
    CollisionEvent,
    CollisionCallback,
    EventFilterCallback,
    CollisionEventDispatcher,
    CollisionListener,
    CollisionEventProcessor,
)


__all__ = [
    # Config
    "BROADPHASE_MARGIN",
    "CONTACT_TOLERANCE",
    "MAX_CONTACT_POINTS",
    "CCD_THRESHOLD_VELOCITY",
    "MAX_CCD_ITERATIONS",
    "SPATIAL_HASH_CELL_SIZE",
    "GJK_MAX_ITERATIONS",
    "EPA_MAX_ITERATIONS",
    "EPA_TOLERANCE",
    "CONTACT_MAX_AGE",
    "WARM_START_FACTOR",
    "NUMERICAL_EPSILON",
    "PARALLEL_THRESHOLD",
    "CCD_SAFETY_FACTOR",
    "CollisionQuality",
    "CollisionConfig",
    "DEFAULT_CONFIG",
    # Broadphase
    "Vec3",
    "AABB",
    "Ray",
    "CollisionPair",
    "RaycastHit",
    "BroadphaseType",
    "Broadphase",
    "SweepAndPrune",
    "DynamicBVH",
    "SpatialHashGrid",
    "Octree",
    "create_broadphase",
    # Narrowphase
    "NarrowphaseAlgorithm",
    "ShapeType",
    "ContactResult",
    "Sphere",
    "Capsule",
    "Box",
    "ConvexHull",
    "gjk_distance",
    "epa_penetration",
    "sat_test",
    "sphere_sphere",
    "sphere_capsule",
    "capsule_capsule",
    "box_box",
    "sphere_box",
    "capsule_box",
    "collide_shapes",
    # Contact Manifold
    "ContactPoint",
    "ManifoldKey",
    "ContactManifold",
    "ManifoldCache",
    "ContactPair",
    "create_contact_pairs",
    # CCD
    "CCDMode",
    "CCDResult",
    "MotionState",
    "linear_sweep_sphere",
    "linear_sweep_capsule",
    "linear_sweep_box",
    "linear_sweep_test",
    "time_of_impact",
    "time_of_impact_sphere_sphere",
    "conservative_advancement",
    "speculative_contacts",
    "CCDManager",
    # Collision Filtering
    "CollisionLayer",
    "CollisionMask",
    "CollisionFilter",
    "should_collide",
    "create_layer_matrix",
    "CollisionFilterManager",
    "FilterPresets",
    # Collision Events
    "CollisionEventType",
    "CollisionEvent",
    "CollisionCallback",
    "EventFilterCallback",
    "CollisionEventDispatcher",
    "CollisionListener",
    "CollisionEventProcessor",
]
