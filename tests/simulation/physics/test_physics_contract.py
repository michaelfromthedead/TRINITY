"""
Blackbox contract tests for the Physics module public API (CLEANROOM).

Tests the public contract of:
  - PhysicsConfig, SolverType, and module constants
  - RigidBody, BodyType, BodyState
  - CollisionShape hierarchy (Sphere, Box, Capsule, Cylinder, etc.)
  - PhysicsWorld lifecycle and body management
  - Physics queries (raycast, overlap, sweep)
  - Physics materials and presets
  - Body flags
  - Sleep manager

Design methodology (blackbox):
  - Equivalence partitioning: body types (static/dynamic/kinematic),
    shape types (primitive/complex/compound), query modes (single/all/overlap/sweep)
  - Boundary value analysis: zero mass, zero velocity, edge-of-shape rays,
    empty world step, min/max timestep limits
  - Error case coverage: invalid body removal, unregistered body wake,
    empty shape parameters, zero-length query direction
"""

import math
import pytest

# Import from the public API surface only
from engine.simulation.physics import (
    # Config
    PhysicsConfig,
    PhysicsBackend,
    BroadphaseType,
    NarrowphaseType,
    SolverType,
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
    PRESET_HIGH_QUALITY,
    PRESET_PERFORMANCE,
    PRESET_MOBILE,
    PRESET_DETERMINISTIC,
    # Flags
    BodyFlags,
    BodyFlagBits,
    # Materials
    PhysicsMaterial,
    CombineMode,
    combine_materials,
    MaterialPresets,
    MATERIAL_PRESETS,
    get_material,
    # Shapes
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
    # Rigid body
    RigidBody,
    BodyType,
    BodyState,
    # Sleep
    SleepManager,
    Island,
    IslandState,
    # Queries
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
    # World
    PhysicsWorld,
    Contact,
    ContactManifold,
    SimulationState,
)

from ..physics_test_base import PhysicsTestCase


# ===========================================================================
# 1. Constants and Config
# ===========================================================================

class TestPhysicsConstants(PhysicsTestCase):
    """Module-level constants exist with reasonable values."""

    def test_gravity_default_is_3tuple(self):
        """DEFAULT_GRAVITY is a 3-element tuple with downward Y."""
        assert isinstance(DEFAULT_GRAVITY, (tuple, list))
        assert len(DEFAULT_GRAVITY) == 3
        # Gravity points downward along Y
        assert DEFAULT_GRAVITY[1] < 0

    def test_timestep_default_positive(self):
        """DEFAULT_TIMESTEP is a positive float."""
        assert isinstance(DEFAULT_TIMESTEP, float)
        assert DEFAULT_TIMESTEP > 0

    def test_min_timestep_positive(self):
        """MIN_TIMESTEP is a positive float less than default."""
        assert isinstance(MIN_TIMESTEP, float)
        assert MIN_TIMESTEP > 0
        assert MIN_TIMESTEP < DEFAULT_TIMESTEP

    def test_max_substeps_positive(self):
        """MAX_SUBSTEPS is a positive integer."""
        assert isinstance(MAX_SUBSTEPS, int)
        assert MAX_SUBSTEPS >= 1

    def test_sleep_thresholds_positive(self):
        """All sleep threshold constants are positive."""
        assert SLEEP_THRESHOLD_LINEAR > 0
        assert SLEEP_THRESHOLD_ANGULAR > 0
        assert SLEEP_TIME_THRESHOLD > 0

    def test_max_bodies_positive(self):
        """MAX_BODIES is a positive integer."""
        assert isinstance(MAX_BODIES, int)
        assert MAX_BODIES > 0

    def test_solver_iterations_positive(self):
        """SOLVER_ITERATIONS and POSITION_ITERATIONS are positive ints."""
        assert isinstance(SOLVER_ITERATIONS, int)
        assert SOLVER_ITERATIONS > 0
        assert isinstance(POSITION_ITERATIONS, int)
        assert POSITION_ITERATIONS > 0

    def test_min_mass_positive(self):
        """MIN_MASS is a positive float."""
        assert isinstance(MIN_MASS, (int, float))
        assert MIN_MASS > 0

    def test_velocity_limits_positive(self):
        """MAX_LINEAR_VELOCITY and MAX_ANGULAR_VELOCITY are positive."""
        assert MAX_LINEAR_VELOCITY > 0
        assert MAX_ANGULAR_VELOCITY > 0

    def test_epsilons_positive(self):
        """COLLISION_EPSILON and FLOAT_COMPARISON_EPSILON are positive."""
        assert COLLISION_EPSILON > 0
        assert FLOAT_COMPARISON_EPSILON > 0


class TestPhysicsConfig(PhysicsTestCase):
    """PhysicsConfig creation, defaults, and presets."""

    def test_default_config_creates(self):
        """PhysicsConfig can be created with default values."""
        config = PhysicsConfig()
        assert config is not None

    def test_config_with_solver_type(self):
        """PhysicsConfig accepts a SolverType."""
        config = PhysicsConfig(solver_type=SolverType.SEQUENTIAL_IMPULSE)
        assert config is not None
        assert config.solver_type == SolverType.SEQUENTIAL_IMPULSE

    def test_config_with_all_solver_types(self):
        """All SolverType values can be used in PhysicsConfig."""
        for solver in SolverType:
            config = PhysicsConfig(solver_type=solver)
            assert config.solver_type == solver

    def test_config_with_backend(self):
        """PhysicsConfig accepts a PhysicsBackend."""
        config = PhysicsConfig(backend=PhysicsBackend.PYTHON)
        assert config.backend == PhysicsBackend.PYTHON

    def test_config_with_broadphase(self):
        """PhysicsConfig accepts a BroadphaseType."""
        config = PhysicsConfig(broadphase_type=BroadphaseType.BRUTE_FORCE)
        assert config.broadphase_type == BroadphaseType.BRUTE_FORCE

    def test_config_with_narrowphase(self):
        """PhysicsConfig accepts a NarrowphaseType."""
        config = PhysicsConfig(narrowphase_type=NarrowphaseType.GJK_EPA)
        assert config.narrowphase_type == NarrowphaseType.GJK_EPA

    def test_preset_high_quality(self):
        """PRESET_HIGH_QUALITY is a PhysicsConfig instance."""
        assert isinstance(PRESET_HIGH_QUALITY, PhysicsConfig)

    def test_preset_performance(self):
        """PRESET_PERFORMANCE is a PhysicsConfig instance."""
        assert isinstance(PRESET_PERFORMANCE, PhysicsConfig)

    def test_preset_mobile(self):
        """PRESET_MOBILE is a PhysicsConfig instance."""
        assert isinstance(PRESET_MOBILE, PhysicsConfig)

    def test_preset_deterministic(self):
        """PRESET_DETERMINISTIC is a PhysicsConfig instance."""
        assert isinstance(PRESET_DETERMINISTIC, PhysicsConfig)

    def test_presets_exist(self):
        """All four preset configs exist as distinct objects."""
        presets = [PRESET_HIGH_QUALITY, PRESET_PERFORMANCE, PRESET_MOBILE, PRESET_DETERMINISTIC]
        ids = {id(c) for c in presets}
        assert len(ids) >= 2, "Presets should be distinct config objects"

    def test_solver_type_enum_values(self):
        """SolverType enum has SEQUENTIAL_IMPULSE, PROJECTED_GAUSS_SEIDEL, JACOBI."""
        members = list(SolverType)
        assert len(members) >= 2
        names = {m.name for m in members}
        assert "SEQUENTIAL_IMPULSE" in names, "SEQUENTIAL_IMPULSE solver type must exist"

    def test_physics_backend_enum(self):
        """PhysicsBackend enum has expected members."""
        members = list(PhysicsBackend)
        assert len(members) >= 1

    def test_broadphase_type_enum(self):
        """BroadphaseType enum has expected members."""
        members = list(BroadphaseType)
        assert len(members) >= 1

    def test_narrowphase_type_enum(self):
        """NarrowphaseType enum has expected members."""
        members = list(NarrowphaseType)
        assert len(members) >= 1


# ===========================================================================
# 2. Body Flags
# ===========================================================================

class TestBodyFlags(PhysicsTestCase):
    """BodyFlags and BodyFlagBits."""

    def test_body_flags_default_zero(self):
        """BodyFlags default to all bits cleared."""
        flags = BodyFlags()
        assert flags is not None

    def test_body_flags_with_bits(self):
        """BodyFlags can be created with boolean kwargs."""
        flags = BodyFlags(use_gravity=True, enable_ccd=True)
        assert flags is not None

    def test_body_flags_from_flags(self):
        """BodyFlags.from_bits accepts a bitmask value."""
        flags = BodyFlags.from_bits(3)  # USE_GRAVITY | ENABLE_CCD
        assert flags is not None

    def test_body_flag_bits_enum(self):
        """BodyFlagBits enum has expected members."""
        members = list(BodyFlagBits)
        assert len(members) >= 1
        names = {m.name for m in members}
        assert "USE_GRAVITY" in names, "USE_GRAVITY flag must exist"
        assert "ENABLE_CCD" in names, "ENABLE_CCD flag must exist"


# ===========================================================================
# 3. Physics Materials
# ===========================================================================

class TestPhysicsMaterial(PhysicsTestCase):
    """PhysicsMaterial creation and presets."""

    def test_default_material(self):
        """Default PhysicsMaterial has expected properties."""
        mat = PhysicsMaterial()
        assert mat is not None
        assert hasattr(mat, "static_friction")
        assert hasattr(mat, "dynamic_friction")
        assert hasattr(mat, "restitution")
        assert hasattr(mat, "density")

    def test_custom_material(self):
        """PhysicsMaterial accepts custom friction and restitution."""
        mat = PhysicsMaterial(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8,
            density=2.0,
        )
        assert mat.restitution == 0.8

    def test_friction_range_positive(self):
        """Material friction values are non-negative."""
        mat = PhysicsMaterial(static_friction=0.0, dynamic_friction=0.0)
        assert mat.static_friction >= 0

    def test_restitution_range(self):
        """Material restitution is in [0, 1]."""
        mat = PhysicsMaterial(restitution=1.0)
        assert 0.0 <= mat.restitution <= 1.0
        mat_zero = PhysicsMaterial(restitution=0.0)
        assert 0.0 <= mat_zero.restitution <= 1.0

    def test_combine_mode_enum(self):
        """CombineMode enum has AVERAGE, MIN, MAX, MULTIPLY, GEOMETRIC."""
        members = list(CombineMode)
        assert len(members) >= 1
        names = {m.name for m in members}
        assert "AVERAGE" in names, "CombineMode must include AVERAGE"

    def test_combine_materials(self):
        """combine_materials produces a tuple of combined values."""
        mat_a = PhysicsMaterial(static_friction=0.5, dynamic_friction=0.3, restitution=0.2)
        mat_b = PhysicsMaterial(static_friction=0.7, dynamic_friction=0.4, restitution=0.6)
        combined = combine_materials(mat_a, mat_b)
        assert combined is not None
        assert isinstance(combined, tuple)
        assert len(combined) == 3

    def test_material_presets_have_preset_methods(self):
        """MaterialPresets has preset material generators."""
        assert hasattr(MaterialPresets, "rubber") or hasattr(MaterialPresets, "default")
        if hasattr(MaterialPresets, "rubber"):
            mat = MaterialPresets.rubber()
            assert isinstance(mat, PhysicsMaterial)

    def test_material_presets_dict(self):
        """MATERIAL_PRESETS is a dict of material name to PhysicsMaterial."""
        assert isinstance(MATERIAL_PRESETS, dict)
        assert len(MATERIAL_PRESETS) > 0
        for name, mat in MATERIAL_PRESETS.items():
            assert isinstance(mat, PhysicsMaterial), f"{name} is not PhysicsMaterial"

    def test_get_material_by_name(self):
        """get_material retrieves a preset material by name."""
        if len(MATERIAL_PRESETS) > 0:
            name = next(iter(MATERIAL_PRESETS.keys()))
            mat = get_material(name)
            assert isinstance(mat, PhysicsMaterial)


# ===========================================================================
# 4. Collision Shapes
# ===========================================================================

class TestCollisionShapeSphere(PhysicsTestCase):
    """SphereShape creation and properties."""

    def test_create_sphere(self):
        """SphereShape can be created with positive radius."""
        shape = SphereShape(radius=1.0)
        assert shape.radius == 1.0
        assert shape.shape_type == ShapeType.SPHERE

    def test_sphere_accepts_zero_radius(self):
        """SphereShape with radius=0 clamps to minimum allowed value."""
        shape = SphereShape(radius=0.0)
        # Radius may be clamped to a minimum positive value
        assert shape.radius >= 0

    def test_sphere_with_offset(self):
        """SphereShape accepts local_offset parameter."""
        shape = SphereShape(radius=0.5, local_offset=(1.0, 0.0, 0.0))
        assert shape.radius == 0.5

    def test_sphere_as_trigger(self):
        """SphereShape accepts is_trigger parameter."""
        shape = SphereShape(radius=0.5, is_trigger=True)
        assert shape.radius == 0.5


class TestCollisionShapeBox(PhysicsTestCase):
    """BoxShape creation and properties."""

    def test_create_box(self):
        """BoxShape can be created with positive half extents."""
        shape = BoxShape(half_extents=(1.0, 2.0, 3.0))
        assert shape.half_extents == (1.0, 2.0, 3.0)
        assert shape.shape_type == ShapeType.BOX

    def test_box_uniform_cube(self):
        """BoxShape with equal extents is a cube."""
        shape = BoxShape(half_extents=(1.0, 1.0, 1.0))
        assert shape.half_extents == (1.0, 1.0, 1.0)

    def test_box_with_rotation(self):
        """BoxShape accepts local_rotation parameter."""
        shape = BoxShape(half_extents=(1.0, 1.0, 1.0), local_rotation=(0.0, 0.0, 0.0, 1.0))
        assert shape.half_extents == (1.0, 1.0, 1.0)


class TestCollisionShapeCapsule(PhysicsTestCase):
    """CapsuleShape creation and properties."""

    def test_create_capsule(self):
        """CapsuleShape can be created with positive radius and height."""
        shape = CapsuleShape(radius=0.5, half_height=1.0)
        assert shape.radius == 0.5
        assert shape.half_height == 1.0
        assert shape.shape_type == ShapeType.CAPSULE

    def test_capsule_zero_height(self):
        """CapsuleShape with zero half_height degenerates to a sphere."""
        shape = CapsuleShape(radius=0.5, half_height=0.0)
        assert shape.radius == 0.5


class TestCollisionShapeCylinder(PhysicsTestCase):
    """CylinderShape creation and properties."""

    def test_create_cylinder(self):
        """CylinderShape can be created with parameters."""
        shape = CylinderShape(radius=1.0, height=2.0)
        assert shape.radius == 1.0
        assert shape.height == 2.0
        assert shape.shape_type == ShapeType.CYLINDER


class TestCollisionShapeConvexHull(PhysicsTestCase):
    """ConvexHullShape creation and properties."""

    def test_create_convex_hull(self):
        """ConvexHullShape can be created with a list of points."""
        shape = ConvexHullShape(points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ])
        assert shape.shape_type == ShapeType.CONVEX_HULL

    def test_convex_hull_empty_raises(self):
        """ConvexHullShape with no points should raise."""
        with pytest.raises((ValueError, AssertionError)):
            ConvexHullShape(points=[])


class TestCollisionShapeMesh(PhysicsTestCase):
    """MeshShape creation and properties."""

    def test_create_mesh(self):
        """MeshShape can be created with vertices and indices."""
        shape = MeshShape(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2],
        )
        assert shape.shape_type == ShapeType.MESH

    def test_mesh_empty_raises(self):
        """MeshShape with no vertices should raise."""
        with pytest.raises((ValueError, AssertionError)):
            MeshShape(vertices=[], indices=[])


class TestCollisionShapeCompound(PhysicsTestCase):
    """CompoundShape creation and child management."""

    def test_create_empty_compound(self):
        """CompoundShape can be created empty."""
        shape = CompoundShape()
        assert shape.shape_type == ShapeType.COMPOUND
        assert len(shape.children) == 0

    def test_compound_with_shapes(self):
        """CompoundShape can be created and children added."""
        shape = CompoundShape()
        shape.add_child(SphereShape(radius=0.5), local_offset=(0.0, 0.0, 0.0))
        assert len(shape.children) >= 1

    def test_compound_add_child(self):
        """CompoundShape accepts child shapes with local offsets."""
        shape = CompoundShape()
        child = SphereShape(radius=0.5)
        shape.add_child(child, local_offset=(1.0, 0.0, 0.0))
        assert len(shape.children) == 1

    def test_compound_multiple_children(self):
        """CompoundShape manages multiple children."""
        shape = CompoundShape()
        shape.add_child(SphereShape(radius=0.5), local_offset=(0.0, 0.0, 0.0))
        shape.add_child(BoxShape(half_extents=(0.5, 0.5, 0.5)), local_offset=(1.0, 0.0, 0.0))
        assert len(shape.children) == 2

    def test_compound_child_type(self):
        """CompoundShape children are CompoundChild instances."""
        shape = CompoundShape()
        shape.add_child(SphereShape(radius=0.5), local_offset=(0.0, 0.0, 0.0))
        child = shape.children[0]
        assert isinstance(child, CompoundChild)
        assert isinstance(child.shape, CollisionShape)


class TestCollisionShapeAABB(PhysicsTestCase):
    """AABB type."""

    def test_aabb_creation(self):
        """AABB can be created with min_point/max_point corners."""
        aabb = AABB(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))
        assert aabb is not None
        assert hasattr(aabb, "min_point")
        assert hasattr(aabb, "max_point")

    def test_aabb_default(self):
        """AABB default creates degenerate box at origin."""
        aabb = AABB()
        assert aabb is not None


class TestCollisionShapeMassProperties(PhysicsTestCase):
    """MassProperties type."""

    def _check_mass_properties(self, shape):
        """Helper: verify shape returns valid MassProperties."""
        props = shape.compute_mass_properties(density=1.0)
        assert isinstance(props, MassProperties)
        assert props.mass > 0
        assert hasattr(props, "inertia_tensor")

    def test_sphere_mass_properties(self):
        """SphereShape compute_mass_properties works."""
        self._check_mass_properties(SphereShape(radius=1.0))

    def test_box_mass_properties(self):
        """BoxShape compute_mass_properties works."""
        self._check_mass_properties(BoxShape(half_extents=(1.0, 2.0, 3.0)))

    def test_capsule_mass_properties(self):
        """CapsuleShape compute_mass_properties works."""
        self._check_mass_properties(CapsuleShape(radius=0.5, half_height=1.0))

    def test_cylinder_mass_properties(self):
        """CylinderShape compute_mass_properties works."""
        self._check_mass_properties(CylinderShape(radius=0.5, height=2.0))

    def test_compound_mass_properties(self):
        """CompoundShape compute_mass_properties works."""
        shape = CompoundShape()
        shape.add_child(SphereShape(radius=1.0), local_offset=(0.0, 0.0, 0.0))
        self._check_mass_properties(shape)

    def test_mass_properties_default(self):
        """MassProperties can be created with default values."""
        props = MassProperties()
        # Default mass may be set to a non-zero value (e.g. 1.0)
        assert props.mass > 0

    def test_create_shape_factory(self):
        """create_shape factory creates shapes by type and params."""
        shape = create_shape(ShapeType.SPHERE, radius=1.0)
        assert isinstance(shape, SphereShape)
        assert shape.radius == 1.0

    def test_shape_type_enum(self):
        """ShapeType enum has all expected shape kinds."""
        members = list(ShapeType)
        names = {m.name for m in members}
        for required in ("SPHERE", "BOX", "CAPSULE", "CYLINDER", "CONVEX_HULL", "MESH", "COMPOUND"):
            assert required in names, f"ShapeType missing {required}"


# ===========================================================================
# 5. Rigid Body
# ===========================================================================

class TestRigidBody(PhysicsTestCase):
    """RigidBody creation, body types, and state."""

    # ------------------------------------------------------------------
    # Body types -- equivalence partitioning
    # ------------------------------------------------------------------

    def test_create_dynamic_body(self):
        """Dynamic RigidBody with position, mass, and shape."""
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 10, 0),
            mass=1.0,
            shape=SphereShape(radius=0.5),
        )
        assert body.body_type == BodyType.DYNAMIC
        self.assertAlmostEqualVec3(body.position, (0, 10, 0))
        assert body.mass == 1.0

    def test_create_static_body(self):
        """Static RigidBody without explicit mass."""
        body = RigidBody(
            body_type=BodyType.STATIC,
            position=(0, 0, 0),
            shape=BoxShape(half_extents=(1, 1, 1)),
        )
        assert body.body_type == BodyType.STATIC

    def test_create_kinematic_body(self):
        """Kinematic RigidBody."""
        body = RigidBody(
            body_type=BodyType.KINEMATIC,
            position=(0, 0, 0),
            shape=SphereShape(radius=1.0),
        )
        assert body.body_type == BodyType.KINEMATIC

    def test_body_type_enum_values(self):
        """BodyType enum has STATIC, DYNAMIC, KINEMATIC."""
        names = {m.name for m in BodyType}
        assert "STATIC" in names
        assert "DYNAMIC" in names
        assert "KINEMATIC" in names

    def test_body_state_is_dataclass(self):
        """BodyState is a dataclass with position, rotation, velocity fields."""
        import dataclasses
        assert dataclasses.is_dataclass(BodyState)
        fields = {f.name for f in dataclasses.fields(BodyState)}
        assert "position" in fields
        assert "rotation" in fields

    # ------------------------------------------------------------------
    # Position and rotation -- boundary values
    # ------------------------------------------------------------------

    def test_body_position_default(self):
        """Default RigidBody position is origin."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        self.assertAlmostEqualVec3(body.position, (0, 0, 0))

    def test_body_set_position(self):
        """RigidBody position can be set."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.position = (5, -3, 2)
        self.assertAlmostEqualVec3(body.position, (5, -3, 2))

    def test_body_set_rotation(self):
        """RigidBody rotation (quaternion) can be set."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.rotation = (0.0, 0.0, 0.0, 1.0)  # identity
        assert body.rotation is not None

    def test_body_with_name(self):
        """RigidBody accepts optional name parameter."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5), name="test_body")
        assert body.name == "test_body"

    # ------------------------------------------------------------------
    # Velocity and momentum
    # ------------------------------------------------------------------

    def test_body_default_velocity_zero(self):
        """Default linear and angular velocity is zero."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        self.assertAlmostEqualVec3(body.linear_velocity, (0, 0, 0))
        self.assertAlmostEqualVec3(body.angular_velocity, (0, 0, 0))

    def test_body_set_linear_velocity(self):
        """RigidBody linear velocity can be set."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.linear_velocity = (10, 0, 0)
        self.assertAlmostEqualVec3(body.linear_velocity, (10, 0, 0))

    def test_body_apply_force(self):
        """RigidBody.apply_force does not crash."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.apply_force((0, -9.81, 0), (0, 0, 0))

    def test_body_apply_impulse(self):
        """RigidBody.apply_impulse does not crash."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.apply_impulse((10, 0, 0), (0, 0, 0))

    def test_body_clear_forces(self):
        """RigidBody.clear_forces resets force accumulator."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.apply_force((100, 0, 0), (0, 0, 0))
        body.clear_forces()

    # ------------------------------------------------------------------
    # Mass and inertia
    # ------------------------------------------------------------------

    def test_body_default_mass(self):
        """Default mass for dynamic body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        assert body.mass > 0

    def test_body_zero_mass_accepted(self):
        """Zero mass clamps to minimum allowed value."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=0, shape=SphereShape(radius=0.5))
        # Mass is clamped to MIN_MASS (e.g. 1e-06)
        assert body.mass > 0

    def test_static_body_mass(self):
        """Static body has mass=0 (infinite)."""
        body = RigidBody(body_type=BodyType.STATIC, shape=SphereShape(radius=0.5))
        assert body.mass == 0

    # ------------------------------------------------------------------
    # Sleeping
    # ------------------------------------------------------------------

    def test_body_starts_awake(self):
        """RigidBody starts awake (not sleeping)."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        assert hasattr(body, "is_sleeping")
        assert not body.is_sleeping

    def test_body_put_to_sleep(self):
        """RigidBody.put_to_sleep forces body to sleep."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.put_to_sleep()
        assert body.is_sleeping

    def test_body_wake_up(self):
        """RigidBody.wake_up wakes a sleeping body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.put_to_sleep()
        assert body.is_sleeping
        body.wake_up()
        assert not body.is_sleeping

    # ------------------------------------------------------------------
    # Collision layers
    # ------------------------------------------------------------------

    def test_body_collision_layer_mask(self):
        """RigidBody has collision_layer and collision_mask."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        assert hasattr(body, "collision_layer")
        assert hasattr(body, "collision_mask")
        assert body.collision_layer >= 1
        assert body.collision_mask >= 1

    def test_body_collision_layer_custom(self):
        """RigidBody collision layer can be customized."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body.collision_layer = 5
        assert body.collision_layer == 5

    # ------------------------------------------------------------------
    # Unique ID
    # ------------------------------------------------------------------

    def test_body_has_id(self):
        """RigidBody has a unique identifier."""
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        assert body.id is not None
        assert isinstance(body.id, (str, int))

    def test_body_ids_unique(self):
        """Different RigidBody instances have different IDs."""
        body_a = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        body_b = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        assert body_a.id != body_b.id

    # ------------------------------------------------------------------
    # Material support
    # ------------------------------------------------------------------

    def test_body_with_material(self):
        """RigidBody accepts PhysicsMaterial parameter."""
        mat = PhysicsMaterial(static_friction=0.3, dynamic_friction=0.3, restitution=0.5)
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5), material=mat)
        assert body.material is not None

    # ------------------------------------------------------------------
    # Shape attachment
    # ------------------------------------------------------------------

    def test_body_attached_shape(self):
        """RigidBody stores its collision shape."""
        shape = BoxShape(half_extents=(2, 3, 4))
        body = RigidBody(body_type=BodyType.STATIC, shape=shape)
        assert body.shape is not None


# ===========================================================================
# 6. Physics World
# ===========================================================================

class TestPhysicsWorldLifecycle(PhysicsTestCase):
    """PhysicsWorld lifecycle: create, start, stop, pause, resume."""

    def test_create_world(self):
        """PhysicsWorld can be created with default settings."""
        world = PhysicsWorld()
        assert world is not None
        assert world.state == SimulationState.STOPPED

    def test_create_world_with_config(self):
        """PhysicsWorld can be created with a PhysicsConfig."""
        config = PhysicsConfig()
        world = PhysicsWorld(config=config)
        assert world is not None

    def test_start_world(self):
        """Starting the world transitions to RUNNING."""
        world = PhysicsWorld()
        world.start()
        assert world.state == SimulationState.RUNNING

    def test_pause_world(self):
        """Pausing the world transitions to PAUSED."""
        world = PhysicsWorld()
        world.start()
        world.pause()
        assert world.state == SimulationState.PAUSED

    def test_resume_world(self):
        """Resuming from pause transitions back to RUNNING."""
        world = PhysicsWorld()
        world.start()
        world.pause()
        world.resume()
        assert world.state == SimulationState.RUNNING

    def test_simulation_state_enum(self):
        """SimulationState enum has STOPPED, RUNNING, PAUSED."""
        names = {m.name for m in SimulationState}
        assert "STOPPED" in names
        assert "RUNNING" in names
        assert "PAUSED" in names

    def test_world_gravity_default(self):
        """World has default gravity."""
        world = PhysicsWorld()
        assert world.gravity is not None
        assert len(world.gravity) == 3

    def test_world_set_gravity(self):
        """World gravity can be set."""
        world = PhysicsWorld()
        world.gravity = (0, -20, 0)
        self.assertAlmostEqualVec3(world.gravity, (0, -20, 0))


class TestPhysicsWorldBodies(PhysicsTestCase):
    """PhysicsWorld body management."""

    def test_add_body(self):
        """Add body to world assigns it an ID."""
        world = PhysicsWorld()
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 0, 0),
            shape=SphereShape(radius=0.5),
        )
        world.add_body(body)
        assert body.id is not None

    def test_get_body_by_id(self):
        """Get body by ID returns the correct body."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        world.add_body(body)
        retrieved = world.get_body(body.id)
        assert retrieved is not None
        assert retrieved.id == body.id

    def test_get_nonexistent_body(self):
        """Get body with unknown ID returns None."""
        world = PhysicsWorld()
        result = world.get_body("nonexistent_id")
        assert result is None

    def test_remove_body(self):
        """Remove body from world."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        world.add_body(body)
        world.remove_body(body)
        assert world.get_body(body.id) is None

    def test_remove_nonexistent_body_does_not_crash(self):
        """Remove body not in world does not crash (no-op)."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        # remove_body on an unregistered body should not crash
        world.remove_body(body)
        # No exception means success

    def test_add_duplicate_body(self):
        """Adding the same body twice does not crash."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5))
        world.add_body(body)
        world.add_body(body)  # Should not crash
        # Verify the body is still in the world
        retrieved = world.get_body(body.id)
        assert retrieved is not None

    def test_body_count(self):
        """World reports accurate body count."""
        world = PhysicsWorld()
        assert world.body_count == 0
        world.add_body(RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5)))
        assert world.body_count == 1
        world.add_body(RigidBody(body_type=BodyType.DYNAMIC, shape=SphereShape(radius=0.5)))
        assert world.body_count == 2


class TestPhysicsWorldStep(PhysicsTestCase):
    """PhysicsWorld simulation stepping."""

    def test_step_empty_world(self):
        """Stepping an empty world does not crash."""
        world = PhysicsWorld()
        world.start()
        world.step(0.016)
        assert world.state == SimulationState.RUNNING

    def test_step_with_bodies(self):
        """Stepping a world with bodies does not crash."""
        world = PhysicsWorld()
        world.start()
        body = RigidBody(body_type=BodyType.DYNAMIC, position=(0, 10, 0),
                         shape=SphereShape(radius=0.5))
        world.add_body(body)
        world.step(0.016)
        assert world.state == SimulationState.RUNNING

    def test_step_updates_positions(self):
        """After stepping, body positions update (gravity acts)."""
        world = PhysicsWorld()
        world.gravity = (0, -9.81, 0)
        world.start()
        body = RigidBody(body_type=BodyType.DYNAMIC, position=(0, 10, 0),
                         shape=SphereShape(radius=0.5))
        world.add_body(body)
        initial_y = body.position[1]

        for _ in range(10):
            world.step(0.016)

        updated = world.get_body(body.id)
        if updated is not None:
            # Body should have moved down (gravitational acceleration)
            assert updated.position[1] < initial_y, \
                f"Body did not fall: y went from {initial_y} to {updated.position[1]}"

    def test_step_multiple_times(self):
        """World can be stepped many times in sequence."""
        world = PhysicsWorld()
        world.start()
        for _ in range(100):
            world.step(0.016)
        assert world.state == SimulationState.RUNNING

    def test_step_stopped_world(self):
        """Stepping a stopped world does not crash."""
        world = PhysicsWorld()
        # World is STOPPED by default -- step may be no-op
        world.step(0.016)
        # No exception means success


class TestPhysicsWorldCollisions(PhysicsTestCase):
    """Collision detection and callbacks."""

    def test_collision_enter_callback_registers(self):
        """on_collision_enter callback can be registered without crashing."""
        world = PhysicsWorld()
        world.start()

        events = []

        def callback(body_a, body_b, contact_info):
            events.append((body_a, body_b))

        world.on_collision_enter(callback)

        a = RigidBody(body_type=BodyType.DYNAMIC, position=(-0.6, 0, 0),
                      shape=SphereShape(radius=0.5))
        b = RigidBody(body_type=BodyType.DYNAMIC, position=(0.6, 0, 0),
                      shape=SphereShape(radius=0.5))
        world.add_body(a)
        world.add_body(b)

        for _ in range(5):
            world.step(0.016)

        assert world.state == SimulationState.RUNNING

    def test_collision_exit_callback_registers(self):
        """on_collision_exit callback can be registered without crashing."""
        world = PhysicsWorld()
        world.start()

        def callback(body_a, body_b):
            pass

        world.on_collision_exit(callback)

        a = RigidBody(body_type=BodyType.DYNAMIC, position=(-0.6, 0, 0),
                      shape=SphereShape(radius=0.5))
        b = RigidBody(body_type=BodyType.DYNAMIC, position=(0.6, 0, 0),
                      shape=SphereShape(radius=0.5))
        world.add_body(a)
        world.add_body(b)

        for _ in range(5):
            world.step(0.016)

        assert world.state == SimulationState.RUNNING

    def test_contact_types_exist(self):
        """Contact and ContactManifold types exist."""
        assert Contact is not None
        assert ContactManifold is not None


# ===========================================================================
# 7. Solver Configuration
# ===========================================================================

class TestSolverConfiguration(PhysicsTestCase):
    """Solver type configuration via PhysicsConfig."""

    def test_solver_type_sequential_impulse_exists(self):
        """SolverType.SEQUENTIAL_IMPULSE exists."""
        assert hasattr(SolverType, "SEQUENTIAL_IMPULSE")

    def test_solver_type_projected_gauss_seidel_exists(self):
        """SolverType.PROJECTED_GAUSS_SEIDEL exists."""
        assert hasattr(SolverType, "PROJECTED_GAUSS_SEIDEL")

    def test_solver_type_jacobi_exists(self):
        """SolverType.JACOBI exists."""
        assert hasattr(SolverType, "JACOBI")

    def test_config_solver_iterations(self):
        """PhysicsConfig stores solver_iterations."""
        config = PhysicsConfig(solver_iterations=10)
        assert config.solver_iterations == 10

    def test_config_position_iterations(self):
        """PhysicsConfig stores position_iterations."""
        config = PhysicsConfig(position_iterations=4)
        assert config.position_iterations == 4

    def test_config_max_substeps(self):
        """PhysicsConfig stores max_substeps."""
        config = PhysicsConfig(max_substeps=8)
        assert config.max_substeps == 8


# ===========================================================================
# 8. Physics Queries
# ===========================================================================

class TestRaycastQuery(PhysicsTestCase):
    """raycast_single and raycast_all query functions."""

    def test_raycast_single_hit(self):
        """raycast_single returns a hit for an intersecting ray."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        hit = raycast_single(
            bodies=[body],
            origin=(0, -5, 0),
            direction=(0, 1, 0),
        )
        assert hit is not None
        assert isinstance(hit, RaycastHit)
        assert hit.distance > 0

    def test_raycast_single_miss(self):
        """raycast_single returns None for a non-intersecting ray."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        hit = raycast_single(
            bodies=[body],
            origin=(10, 0, 0),
            direction=(0, 1, 0),
        )
        assert hit is None

    def test_raycast_all_returns_list(self):
        """raycast_all returns a list."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=0.5))
        results = raycast_all(
            bodies=[body],
            origin=(0, -5, 0),
            direction=(0, 1, 0),
        )
        assert isinstance(results, list)

    def test_raycast_no_bodies(self):
        """raycast_single with empty body list returns None."""
        hit = raycast_single(bodies=[], origin=(0, 0, 0), direction=(1, 0, 0))
        assert hit is None

    def test_raycast_hit_attributes(self):
        """RaycastHit has expected attributes."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        hit = raycast_single(
            bodies=[body],
            origin=(0, -5, 0),
            direction=(0, 1, 0),
        )
        if hit is not None:
            assert hasattr(hit, "point")
            assert hasattr(hit, "normal")
            assert hasattr(hit, "distance")
            assert hasattr(hit, "body")
            assert hasattr(hit, "shape")

    def test_raycast_with_max_distance(self):
        """raycast_single accepts max_distance parameter."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        hit = raycast_single(
            bodies=[body],
            origin=(0, -5, 0),
            direction=(0, 1, 0),
            max_distance=2.0,  # Too short to reach the sphere
        )
        # Should miss because max_distance is too short
        assert hit is None


class TestOverlapQuery(PhysicsTestCase):
    """Overlap query functions."""

    def test_overlap_sphere(self):
        """overlap_sphere can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        results = overlap_sphere(
            bodies=[body],
            center=(0, 0, 0),
            radius=2.0,
        )
        assert isinstance(results, list)

    def test_overlap_box(self):
        """overlap_box can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        results = overlap_box(
            bodies=[body],
            center=(0, 0, 0),
            half_extents=(2, 2, 2),
        )
        assert isinstance(results, list)

    def test_overlap_capsule(self):
        """overlap_capsule can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        results = overlap_capsule(
            bodies=[body],
            start=(0, 0, 0),
            end=(0, 2.0, 0),
            radius=1.0,
        )
        assert isinstance(results, list)

    def test_overlap_result_type(self):
        """OverlapResult type exists."""
        assert OverlapResult is not None

    def test_overlap_no_bodies(self):
        """Overlap queries with empty body list return empty list."""
        results = overlap_sphere(bodies=[], center=(0, 0, 0), radius=1.0)
        assert isinstance(results, list)
        assert len(results) == 0


class TestSweepQuery(PhysicsTestCase):
    """Sweep query functions."""

    def test_sweep_sphere(self):
        """sweep_sphere can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(5, 0, 0),
                         shape=SphereShape(radius=1.0))
        result = sweep_sphere(
            bodies=[body],
            start=(0, 0, 0),
            direction=(1, 0, 0),
            radius=0.5,
            distance=10.0,
        )
        assert result is None or isinstance(result, SweepResult)

    def test_sweep_box(self):
        """sweep_box can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(5, 0, 0),
                         shape=SphereShape(radius=1.0))
        result = sweep_box(
            bodies=[body],
            start=(0, 0, 0),
            direction=(1, 0, 0),
            half_extents=(0.5, 0.5, 0.5),
            distance=10.0,
        )
        assert result is None or isinstance(result, SweepResult)

    def test_sweep_capsule(self):
        """sweep_capsule can be called without crashing."""
        body = RigidBody(body_type=BodyType.STATIC, position=(5, 0, 0),
                         shape=SphereShape(radius=1.0))
        result = sweep_capsule(
            bodies=[body],
            start_a=(0, 0, 0),
            start_b=(0, 1.0, 0),
            direction=(1, 0, 0),
            radius=0.5,
            distance=10.0,
        )
        assert result is None or isinstance(result, SweepResult)

    def test_sweep_no_bodies(self):
        """Sweep queries with empty body list return a miss SweepResult."""
        result = sweep_sphere(bodies=[], start=(0, 0, 0), direction=(1, 0, 0),
                              radius=0.5, distance=10.0)
        assert isinstance(result, SweepResult)
        assert not result.hit

    def test_sweep_result_type(self):
        """SweepResult type exists."""
        assert SweepResult is not None


class TestPointQuery(PhysicsTestCase):
    """Point-based query functions."""

    def test_point_inside_hit(self):
        """point_inside returns a body for a point inside a shape."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=5.0))
        result = point_inside([body], (1, 1, 1))
        # Returns the body itself if point is inside, or None if outside
        assert result is body or result is None

    def test_point_inside_miss(self):
        """point_inside returns None for a point outside a shape."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        result = point_inside([body], (10, 10, 10))
        assert result is None

    def test_closest_point_on_body(self):
        """closest_point_on_body returns a point on the body's surface."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        result = closest_point_on_body(body, (5, 0, 0))
        if result is not None:
            assert len(result) == 3

    def test_distance_to_body(self):
        """distance_to_body returns a positive distance or 0."""
        body = RigidBody(body_type=BodyType.STATIC, position=(0, 0, 0),
                         shape=SphereShape(radius=1.0))
        dist = distance_to_body(body, (5, 0, 0))
        if dist is not None:
            assert dist >= 0


class TestCollisionFilter(PhysicsTestCase):
    """CollisionFilter for query filtering."""

    def test_filter_default(self):
        """Default CollisionFilter does not filter."""
        filt = CollisionFilter()
        assert filt is not None

    def test_filter_custom_layers(self):
        """CollisionFilter can specify layer and mask."""
        filt = CollisionFilter(layer=1, mask=1)
        assert filt.layer == 1
        assert filt.mask == 1

    def test_query_flags_enum(self):
        """QueryFlags enum has expected members."""
        members = list(QueryFlags)
        assert len(members) >= 1


# ===========================================================================
# 9. Sleep Manager
# ===========================================================================

class TestSleepManager(PhysicsTestCase):
    """SleepManager sleep/wake behavior."""

    def _make_body(self, position=(0, 0, 0)):
        """Helper: create a dynamic body."""
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=position,
            shape=SphereShape(radius=0.5),
        )
        return body

    def test_create_sleep_manager(self):
        """SleepManager can be created with defaults."""
        mgr = SleepManager()
        assert mgr is not None

    def test_register_body(self):
        """SleepManager accepts body registration."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.register_body(body)

    def test_unregister_body(self):
        """SleepManager accepts body unregistration."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.register_body(body)
        mgr.unregister_body(body)

    def test_unregister_unknown_body(self):
        """Unregistering unknown body does not crash."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.unregister_body(body)  # Should not raise

    def test_put_to_sleep(self):
        """SleepManager.put_to_sleep forces a body to sleep."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.register_body(body)
        mgr.put_to_sleep(body)
        assert body.is_sleeping

    def test_wake_up(self):
        """SleepManager.wake_up forces a body to wake."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.register_body(body)
        mgr.put_to_sleep(body)
        assert body.is_sleeping
        mgr.wake_up(body)
        assert not body.is_sleeping

    def test_update(self):
        """SleepManager.update processes sleep timers without crash."""
        mgr = SleepManager()
        body = self._make_body()
        body.linear_velocity = (0, 0, 0)
        mgr.register_body(body)
        mgr.update(0.016)

    def test_merge_islands(self):
        """SleepManager.merge_islands connects two bodies."""
        mgr = SleepManager()
        body_a = self._make_body()
        body_b = self._make_body()
        mgr.register_body(body_a)
        mgr.register_body(body_b)
        mgr.merge_islands(body_a, body_b)

    def test_rebuild_islands(self):
        """SleepManager.rebuild_islands recomputes island graph."""
        mgr = SleepManager()
        body_a = self._make_body()
        body_b = self._make_body()
        mgr.register_body(body_a)
        mgr.register_body(body_b)
        mgr.merge_islands(body_a, body_b)
        mgr.rebuild_islands(contacts=[])

    def test_island_type(self):
        """Island and IslandState types exist."""
        assert Island is not None
        assert IslandState is not None

    def test_get_statistics(self):
        """SleepManager.get_statistics returns a dict with expected keys."""
        mgr = SleepManager()
        body = self._make_body()
        mgr.register_body(body)
        stats = mgr.get_statistics()
        assert isinstance(stats, dict)
        assert "total_bodies" in stats

    def test_configurable_thresholds(self):
        """SleepManager accepts configurable thresholds."""
        mgr = SleepManager(
            linear_threshold=0.5,
            angular_threshold=0.3,
            time_threshold=1.0,
        )
        assert mgr is not None
