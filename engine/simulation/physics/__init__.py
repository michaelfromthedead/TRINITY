"""
Physics Core Module

A complete physics simulation system for game engines including:
- Rigid body dynamics
- Collision detection and response
- Sleep optimization
- Physics queries (raycasting, overlap, sweep)
- Material properties

Usage:
    from engine.simulation.physics import (
        PhysicsWorld, PhysicsConfig,
        RigidBody, BodyType,
        SphereShape, BoxShape, CapsuleShape,
        PhysicsMaterial, MaterialPresets,
    )

    # Create world
    world = PhysicsWorld()
    world.gravity = (0, -9.81, 0)

    # Create a dynamic body
    body = RigidBody(
        body_type=BodyType.DYNAMIC,
        position=(0, 10, 0),
        mass=1.0,
        shape=SphereShape(radius=0.5),
        material=MaterialPresets.rubber(),
    )
    world.add_body(body)

    # Simulate
    world.start()
    world.step(1/60)
"""

# Configuration
from .config import (
    PhysicsConfig,
    PhysicsBackend,
    BroadphaseType,
    NarrowphaseType,
    SolverType,
    # Constants
    DEFAULT_GRAVITY,
    DEFAULT_TIMESTEP,
    MIN_TIMESTEP,
    MAX_SUBSTEPS,
    SLEEP_THRESHOLD_LINEAR,
    SLEEP_THRESHOLD_ANGULAR,
    SLEEP_TIME_THRESHOLD,
    MAX_BODIES,
    SOLVER_ITERATIONS,
    POSITION_ITERATIONS,
    MIN_MASS,
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    COLLISION_EPSILON,
    FLOAT_COMPARISON_EPSILON,
    # Presets
    PRESET_HIGH_QUALITY,
    PRESET_PERFORMANCE,
    PRESET_MOBILE,
    PRESET_DETERMINISTIC,
)

# Body flags
from .body_flags import (
    BodyFlags,
    BodyFlagBits,
)

# Physics materials
from .physics_material import (
    PhysicsMaterial,
    CombineMode,
    combine_materials,
    MaterialPresets,
    MATERIAL_PRESETS,
    get_material,
)

# Collision shapes
from .collision_shapes import (
    ShapeType,
    CollisionShape,
    SphereShape,
    BoxShape,
    CapsuleShape,
    CylinderShape,
    ConvexHullShape,
    MeshShape,
    CompoundShape,
    CompoundChild,
    AABB,
    MassProperties,
    create_shape,
)

# Rigid body
from .rigid_body import (
    RigidBody,
    BodyType,
    BodyState,
)

# Sleep management
from .sleeping import (
    SleepManager,
    Island,
    IslandState,
)

# Physics queries
from .queries import (
    CollisionFilter,
    QueryFlags,
    RaycastHit,
    OverlapResult,
    SweepResult,
    raycast_single,
    raycast_all,
    overlap_sphere,
    overlap_box,
    overlap_capsule,
    sweep_sphere,
    sweep_box,
    sweep_capsule,
    point_inside,
    closest_point_on_body,
    distance_to_body,
)

# Physics world
from .physics_world import (
    PhysicsWorld,
    Contact,
    ContactManifold,
    SimulationState,
)

__all__ = [
    # Config
    'PhysicsConfig',
    'PhysicsBackend',
    'BroadphaseType',
    'NarrowphaseType',
    'SolverType',
    'DEFAULT_GRAVITY',
    'DEFAULT_TIMESTEP',
    'MIN_TIMESTEP',
    'MAX_SUBSTEPS',
    'SLEEP_THRESHOLD_LINEAR',
    'SLEEP_THRESHOLD_ANGULAR',
    'SLEEP_TIME_THRESHOLD',
    'MAX_BODIES',
    'SOLVER_ITERATIONS',
    'POSITION_ITERATIONS',
    'MIN_MASS',
    'MAX_LINEAR_VELOCITY',
    'MAX_ANGULAR_VELOCITY',
    'COLLISION_EPSILON',
    'FLOAT_COMPARISON_EPSILON',
    'PRESET_HIGH_QUALITY',
    'PRESET_PERFORMANCE',
    'PRESET_MOBILE',
    'PRESET_DETERMINISTIC',

    # Flags
    'BodyFlags',
    'BodyFlagBits',

    # Materials
    'PhysicsMaterial',
    'CombineMode',
    'combine_materials',
    'MaterialPresets',
    'MATERIAL_PRESETS',
    'get_material',

    # Shapes
    'ShapeType',
    'CollisionShape',
    'SphereShape',
    'BoxShape',
    'CapsuleShape',
    'CylinderShape',
    'ConvexHullShape',
    'MeshShape',
    'CompoundShape',
    'CompoundChild',
    'AABB',
    'MassProperties',
    'create_shape',

    # Rigid body
    'RigidBody',
    'BodyType',
    'BodyState',

    # Sleeping
    'SleepManager',
    'Island',
    'IslandState',

    # Queries
    'CollisionFilter',
    'QueryFlags',
    'RaycastHit',
    'OverlapResult',
    'SweepResult',
    'raycast_single',
    'raycast_all',
    'overlap_sphere',
    'overlap_box',
    'overlap_capsule',
    'sweep_sphere',
    'sweep_box',
    'sweep_capsule',
    'point_inside',
    'closest_point_on_body',
    'distance_to_body',

    # World
    'PhysicsWorld',
    'Contact',
    'ContactManifold',
    'SimulationState',
]

__version__ = '1.0.0'
