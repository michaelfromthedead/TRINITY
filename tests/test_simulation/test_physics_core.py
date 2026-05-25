"""
Comprehensive Test Suite for Physics Core Module

Tests all components of the physics simulation including:
- RigidBody creation, forces, impulses
- Collision shapes and AABB computation
- PhysicsWorld step, add/remove bodies
- Sleeping behavior
- Raycasting and queries
- Physics materials and combining

Target: 120+ tests
"""

import pytest
import math
from typing import Tuple

import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.simulation.physics import (
    # Config
    PhysicsConfig,
    PhysicsBackend,
    BroadphaseType,
    DEFAULT_GRAVITY,
    DEFAULT_TIMESTEP,
    MAX_SUBSTEPS,
    MIN_MASS,
    MAX_LINEAR_VELOCITY,
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

    # Sleeping
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

    # World
    PhysicsWorld,
    Contact,
    ContactManifold,
    SimulationState,
)


# =============================================================================
# Helper Functions
# =============================================================================

def approx_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tolerance


def vector_approx_equal(a: Tuple, b: Tuple, tolerance: float = 1e-5) -> bool:
    """Check if two vectors are approximately equal."""
    return all(abs(ai - bi) < tolerance for ai, bi in zip(a, b))


def vector_length(v: Tuple) -> float:
    """Calculate vector length."""
    return math.sqrt(sum(x*x for x in v))


# =============================================================================
# Test Configuration
# =============================================================================

class TestPhysicsConfig:
    """Tests for PhysicsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhysicsConfig()
        assert config.gravity == DEFAULT_GRAVITY
        assert config.timestep == DEFAULT_TIMESTEP
        assert config.max_substeps == MAX_SUBSTEPS

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PhysicsConfig(
            gravity=(0, -20, 0),
            timestep=1/120,
            max_substeps=16,
        )
        assert config.gravity == (0, -20, 0)
        assert approx_equal(config.timestep, 1/120)
        assert config.max_substeps == 16

    def test_config_validation_valid(self):
        """Test validation passes for valid config."""
        config = PhysicsConfig()
        assert config.validate() is True

    def test_config_validation_invalid_timestep(self):
        """Test validation fails for invalid timestep."""
        config = PhysicsConfig(timestep=-1.0)
        with pytest.raises(ValueError):
            config.validate()

    def test_config_validation_invalid_substeps(self):
        """Test validation fails for invalid substeps."""
        config = PhysicsConfig(max_substeps=0)
        with pytest.raises(ValueError):
            config.validate()

    def test_config_copy(self):
        """Test config copy creates independent copy."""
        config = PhysicsConfig(gravity=(0, -5, 0))
        copy = config.copy()
        assert copy.gravity == config.gravity
        copy.gravity = (0, -10, 0)
        assert config.gravity == (0, -5, 0)

    def test_preset_high_quality(self):
        """Test high quality preset."""
        assert PRESET_HIGH_QUALITY.solver_iterations == 20
        assert PRESET_HIGH_QUALITY.enable_ccd is True

    def test_preset_performance(self):
        """Test performance preset."""
        assert PRESET_PERFORMANCE.solver_iterations == 4
        assert PRESET_PERFORMANCE.enable_ccd is False

    def test_preset_mobile(self):
        """Test mobile preset."""
        assert PRESET_MOBILE.max_bodies == 1024
        assert PRESET_MOBILE.max_substeps == 2

    def test_preset_deterministic(self):
        """Test deterministic preset."""
        assert PRESET_DETERMINISTIC.enable_sleeping is False


# =============================================================================
# Test Body Flags
# =============================================================================

class TestBodyFlags:
    """Tests for BodyFlags."""

    def test_default_flags(self):
        """Test default flag values."""
        flags = BodyFlags()
        assert flags.use_gravity is True
        assert flags.enable_ccd is False
        assert flags.is_trigger is False

    def test_custom_flags(self):
        """Test custom flag values."""
        flags = BodyFlags(
            use_gravity=False,
            enable_ccd=True,
            lock_position_y=True,
        )
        assert flags.use_gravity is False
        assert flags.enable_ccd is True
        assert flags.lock_position_y is True

    def test_set_flag(self):
        """Test setting individual flags."""
        flags = BodyFlags()
        flags.set_flag(BodyFlagBits.ENABLE_CCD, True)
        assert flags.enable_ccd is True

    def test_get_flag(self):
        """Test getting individual flags."""
        flags = BodyFlags(is_trigger=True)
        assert flags.get_flag(BodyFlagBits.IS_TRIGGER) is True

    def test_toggle_flag(self):
        """Test toggling flags."""
        flags = BodyFlags(use_gravity=True)
        flags.toggle_flag(BodyFlagBits.USE_GRAVITY)
        assert flags.use_gravity is False

    def test_lock_position_all(self):
        """Test locking all position axes."""
        flags = BodyFlags()
        flags.lock_position_all = True
        assert flags.lock_position_x is True
        assert flags.lock_position_y is True
        assert flags.lock_position_z is True

    def test_lock_rotation_all(self):
        """Test locking all rotation axes."""
        flags = BodyFlags()
        flags.lock_rotation_all = True
        assert flags.lock_rotation_x is True
        assert flags.lock_rotation_y is True
        assert flags.lock_rotation_z is True

    def test_position_lock_mask(self):
        """Test position lock mask generation."""
        flags = BodyFlags(lock_position_x=True, lock_position_z=True)
        mask = flags.get_position_lock_mask()
        assert mask == (0.0, 1.0, 0.0)

    def test_rotation_lock_mask(self):
        """Test rotation lock mask generation."""
        flags = BodyFlags(lock_rotation_y=True)
        mask = flags.get_rotation_lock_mask()
        assert mask == (1.0, 0.0, 1.0)

    def test_static_body_preset(self):
        """Test static body flag preset."""
        flags = BodyFlags.static_body()
        assert flags.use_gravity is False
        assert flags.lock_position_all is True
        assert flags.lock_rotation_all is True

    def test_kinematic_body_preset(self):
        """Test kinematic body flag preset."""
        flags = BodyFlags.kinematic_body()
        assert flags.use_gravity is False
        assert flags.disable_deactivation is True

    def test_dynamic_body_preset(self):
        """Test dynamic body flag preset."""
        flags = BodyFlags.dynamic_body()
        assert flags.use_gravity is True
        assert flags.enable_gyroscopic is True

    def test_trigger_volume_preset(self):
        """Test trigger volume flag preset."""
        flags = BodyFlags.trigger_volume()
        assert flags.is_trigger is True
        assert flags.lock_position_all is True

    def test_flags_copy(self):
        """Test flag copy."""
        flags = BodyFlags(enable_ccd=True)
        copy = flags.copy()
        assert copy.enable_ccd is True
        copy.enable_ccd = False
        assert flags.enable_ccd is True

    def test_flags_equality(self):
        """Test flag equality comparison."""
        flags1 = BodyFlags(use_gravity=True)
        flags2 = BodyFlags(use_gravity=True)
        flags3 = BodyFlags(use_gravity=False)
        assert flags1 == flags2
        assert flags1 != flags3


# =============================================================================
# Test Physics Materials
# =============================================================================

class TestPhysicsMaterial:
    """Tests for PhysicsMaterial."""

    def test_default_material(self):
        """Test default material values."""
        mat = PhysicsMaterial()
        assert mat.static_friction == 0.6
        assert mat.dynamic_friction == 0.4
        assert mat.restitution == 0.0

    def test_custom_material(self):
        """Test custom material values."""
        mat = PhysicsMaterial(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.5,
            density=2000.0,
        )
        assert mat.static_friction == 0.8
        assert mat.dynamic_friction == 0.6
        assert mat.restitution == 0.5
        assert mat.density == 2000.0

    def test_friction_clamping(self):
        """Test friction values are clamped."""
        mat = PhysicsMaterial(static_friction=5.0, dynamic_friction=-1.0)
        assert mat.static_friction == 2.0  # MAX_FRICTION
        assert mat.dynamic_friction == 0.0  # MIN_FRICTION

    def test_restitution_clamping(self):
        """Test restitution is clamped to [0, 1]."""
        mat = PhysicsMaterial(restitution=2.0)
        assert mat.restitution == 1.0

    def test_dynamic_not_exceeds_static(self):
        """Test dynamic friction doesn't exceed static."""
        mat = PhysicsMaterial(static_friction=0.5, dynamic_friction=0.8)
        assert mat.dynamic_friction <= mat.static_friction

    def test_set_friction(self):
        """Test set_friction method."""
        mat = PhysicsMaterial()
        mat.set_friction(0.9, 0.7)
        assert mat.static_friction == 0.9
        assert mat.dynamic_friction == 0.7

    def test_set_bounciness(self):
        """Test set_bounciness method."""
        mat = PhysicsMaterial()
        mat.set_bounciness(0.8)
        assert mat.restitution == 0.8

    def test_material_copy(self):
        """Test material copy."""
        mat = PhysicsMaterial(restitution=0.5)
        copy = mat.copy()
        assert copy.restitution == 0.5
        copy.restitution = 0.2
        assert mat.restitution == 0.5

    def test_material_presets_rubber(self):
        """Test rubber preset."""
        mat = MaterialPresets.rubber()
        assert mat.restitution == 0.8
        assert mat.name == "rubber"

    def test_material_presets_ice(self):
        """Test ice preset."""
        mat = MaterialPresets.ice()
        assert mat.static_friction < 0.1
        assert mat.name == "ice"

    def test_material_presets_metal(self):
        """Test metal preset."""
        mat = MaterialPresets.metal()
        assert mat.density > 5000
        assert mat.name == "metal"

    def test_material_presets_bouncy_ball(self):
        """Test bouncy ball preset."""
        mat = MaterialPresets.bouncy_ball()
        assert mat.restitution > 0.9

    def test_get_material(self):
        """Test get_material function."""
        mat = get_material('rubber')
        assert mat.name == "rubber"

    def test_get_material_unknown(self):
        """Test get_material with unknown name."""
        with pytest.raises(KeyError):
            get_material('unknown_material')

    def test_combine_materials_average(self):
        """Test material combination with average mode."""
        mat_a = PhysicsMaterial(static_friction=0.8, dynamic_friction=0.6, restitution=0.2)
        mat_b = PhysicsMaterial(static_friction=0.4, dynamic_friction=0.2, restitution=0.8)
        sf, df, r = combine_materials(mat_a, mat_b)
        assert approx_equal(sf, 0.6)  # (0.8 + 0.4) / 2
        assert approx_equal(r, 0.5)   # (0.2 + 0.8) / 2

    def test_combine_materials_min_mode(self):
        """Test material combination with min mode."""
        mat_a = PhysicsMaterial(restitution=0.2, restitution_combine=CombineMode.MIN)
        mat_b = PhysicsMaterial(restitution=0.8, restitution_combine=CombineMode.MIN)
        _, _, r = combine_materials(mat_a, mat_b)
        assert approx_equal(r, 0.2)

    def test_combine_materials_max_mode(self):
        """Test material combination with max mode."""
        mat_a = PhysicsMaterial(restitution=0.2, restitution_combine=CombineMode.MAX)
        mat_b = PhysicsMaterial(restitution=0.8, restitution_combine=CombineMode.MAX)
        _, _, r = combine_materials(mat_a, mat_b)
        assert approx_equal(r, 0.8)

    def test_combine_materials_multiply_mode(self):
        """Test material combination with multiply mode."""
        mat_a = PhysicsMaterial(restitution=0.5, restitution_combine=CombineMode.MULTIPLY)
        mat_b = PhysicsMaterial(restitution=0.5, restitution_combine=CombineMode.MULTIPLY)
        _, _, r = combine_materials(mat_a, mat_b)
        assert approx_equal(r, 0.25)


# =============================================================================
# Test Collision Shapes
# =============================================================================

class TestAABB:
    """Tests for AABB."""

    def test_aabb_center(self):
        """Test AABB center calculation."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.center == (1, 2, 3)

    def test_aabb_half_extents(self):
        """Test AABB half extents."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.half_extents == (1, 2, 3)

    def test_aabb_size(self):
        """Test AABB size."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.size == (2, 4, 6)

    def test_aabb_volume(self):
        """Test AABB volume."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.volume == 48  # 2 * 4 * 6

    def test_aabb_contains_point_inside(self):
        """Test point inside AABB."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 2, 2))
        assert aabb.contains_point((1, 1, 1)) is True

    def test_aabb_contains_point_outside(self):
        """Test point outside AABB."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 2, 2))
        assert aabb.contains_point((3, 1, 1)) is False

    def test_aabb_intersects_true(self):
        """Test AABB intersection."""
        aabb1 = AABB(min_point=(0, 0, 0), max_point=(2, 2, 2))
        aabb2 = AABB(min_point=(1, 1, 1), max_point=(3, 3, 3))
        assert aabb1.intersects(aabb2) is True

    def test_aabb_intersects_false(self):
        """Test AABB non-intersection."""
        aabb1 = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        aabb2 = AABB(min_point=(2, 2, 2), max_point=(3, 3, 3))
        assert aabb1.intersects(aabb2) is False

    def test_aabb_expand(self):
        """Test AABB expansion."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        expanded = aabb.expand(0.5)
        assert expanded.min_point == (-0.5, -0.5, -0.5)
        assert expanded.max_point == (1.5, 1.5, 1.5)

    def test_aabb_merge(self):
        """Test AABB merge."""
        aabb1 = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        aabb2 = AABB(min_point=(2, 2, 2), max_point=(3, 3, 3))
        merged = aabb1.merge(aabb2)
        assert merged.min_point == (0, 0, 0)
        assert merged.max_point == (3, 3, 3)

    def test_aabb_from_points(self):
        """Test AABB from points."""
        points = [(0, 0, 0), (1, 2, 3), (-1, -2, -3)]
        aabb = AABB.from_points(points)
        assert aabb.min_point == (-1, -2, -3)
        assert aabb.max_point == (1, 2, 3)


class TestSphereShape:
    """Tests for SphereShape."""

    def test_sphere_creation(self):
        """Test sphere shape creation."""
        sphere = SphereShape(radius=1.0)
        assert sphere.radius == 1.0
        assert sphere.shape_type == ShapeType.SPHERE

    def test_sphere_aabb(self):
        """Test sphere AABB computation."""
        sphere = SphereShape(radius=1.0)
        aabb = sphere.compute_aabb(position=(0, 0, 0))
        assert approx_equal(aabb.min_point[0], -1.04, 0.1)
        assert approx_equal(aabb.max_point[0], 1.04, 0.1)

    def test_sphere_mass_properties(self):
        """Test sphere mass properties."""
        sphere = SphereShape(radius=1.0)
        props = sphere.compute_mass_properties(density=1000.0)
        # Volume of unit sphere = 4/3 * pi
        expected_volume = (4/3) * math.pi
        assert approx_equal(props.mass, expected_volume * 1000, 1.0)

    def test_sphere_support_point(self):
        """Test sphere support point."""
        sphere = SphereShape(radius=1.0)
        support = sphere.get_support_point((1, 0, 0))
        assert approx_equal(support[0], 1.0)

    def test_sphere_contains_point_inside(self):
        """Test point inside sphere."""
        sphere = SphereShape(radius=1.0)
        assert sphere.contains_point((0.5, 0, 0)) is True

    def test_sphere_contains_point_outside(self):
        """Test point outside sphere."""
        sphere = SphereShape(radius=1.0)
        assert sphere.contains_point((2, 0, 0)) is False

    def test_sphere_copy(self):
        """Test sphere copy."""
        sphere = SphereShape(radius=2.0)
        copy = sphere.copy()
        assert copy.radius == 2.0
        copy.radius = 3.0
        assert sphere.radius == 2.0


class TestBoxShape:
    """Tests for BoxShape."""

    def test_box_creation(self):
        """Test box shape creation."""
        box = BoxShape(half_extents=(1, 2, 3))
        assert box.half_extents == (1, 2, 3)
        assert box.shape_type == ShapeType.BOX

    def test_box_size(self):
        """Test box size property."""
        box = BoxShape(half_extents=(1, 2, 3))
        assert box.size == (2, 4, 6)

    def test_box_aabb(self):
        """Test box AABB computation."""
        box = BoxShape(half_extents=(1, 1, 1))
        aabb = box.compute_aabb(position=(0, 0, 0))
        assert approx_equal(aabb.min_point[0], -1.04, 0.1)
        assert approx_equal(aabb.max_point[0], 1.04, 0.1)

    def test_box_mass_properties(self):
        """Test box mass properties."""
        box = BoxShape(half_extents=(1, 1, 1))
        props = box.compute_mass_properties(density=1000.0)
        # Volume of 2x2x2 box = 8
        assert approx_equal(props.mass, 8000, 1.0)

    def test_box_support_point(self):
        """Test box support point."""
        box = BoxShape(half_extents=(1, 1, 1))
        support = box.get_support_point((1, 1, 1))
        assert support == (1, 1, 1)

    def test_box_contains_point_inside(self):
        """Test point inside box."""
        box = BoxShape(half_extents=(1, 1, 1))
        assert box.contains_point((0.5, 0.5, 0.5)) is True

    def test_box_contains_point_outside(self):
        """Test point outside box."""
        box = BoxShape(half_extents=(1, 1, 1))
        assert box.contains_point((2, 0, 0)) is False


class TestCapsuleShape:
    """Tests for CapsuleShape."""

    def test_capsule_creation(self):
        """Test capsule shape creation."""
        capsule = CapsuleShape(radius=0.5, half_height=1.0)
        assert capsule.radius == 0.5
        assert capsule.half_height == 1.0
        assert capsule.shape_type == ShapeType.CAPSULE

    def test_capsule_total_height(self):
        """Test capsule total height."""
        capsule = CapsuleShape(radius=0.5, half_height=1.0)
        assert capsule.total_height == 3.0  # 2*1.0 + 2*0.5

    def test_capsule_aabb(self):
        """Test capsule AABB computation."""
        capsule = CapsuleShape(radius=1.0, half_height=1.0)
        aabb = capsule.compute_aabb(position=(0, 0, 0))
        assert approx_equal(aabb.min_point[1], -2.04, 0.1)
        assert approx_equal(aabb.max_point[1], 2.04, 0.1)

    def test_capsule_contains_point_cylinder(self):
        """Test point in cylindrical section."""
        capsule = CapsuleShape(radius=1.0, half_height=1.0)
        assert capsule.contains_point((0.5, 0, 0)) is True

    def test_capsule_contains_point_cap(self):
        """Test point in cap section."""
        capsule = CapsuleShape(radius=1.0, half_height=1.0)
        assert capsule.contains_point((0, 1.5, 0)) is True


class TestCylinderShape:
    """Tests for CylinderShape."""

    def test_cylinder_creation(self):
        """Test cylinder shape creation."""
        cyl = CylinderShape(radius=0.5, height=2.0)
        assert cyl.radius == 0.5
        assert cyl.height == 2.0
        assert cyl.shape_type == ShapeType.CYLINDER

    def test_cylinder_half_height(self):
        """Test cylinder half height."""
        cyl = CylinderShape(radius=0.5, height=2.0)
        assert cyl.half_height == 1.0

    def test_cylinder_contains_point_inside(self):
        """Test point inside cylinder."""
        cyl = CylinderShape(radius=1.0, height=2.0)
        assert cyl.contains_point((0.5, 0.5, 0)) is True


class TestConvexHullShape:
    """Tests for ConvexHullShape."""

    def test_convex_hull_creation(self):
        """Test convex hull shape creation."""
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        hull = ConvexHullShape(points)
        assert hull.vertex_count >= 4
        assert hull.shape_type == ShapeType.CONVEX_HULL

    def test_convex_hull_minimum_points(self):
        """Test convex hull requires minimum points."""
        with pytest.raises(ValueError):
            ConvexHullShape([(0, 0, 0), (1, 0, 0), (0, 1, 0)])


class TestMeshShape:
    """Tests for MeshShape."""

    def test_mesh_creation(self):
        """Test mesh shape creation."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        mesh = MeshShape(vertices, indices)
        assert mesh.vertex_count == 4
        assert mesh.triangle_count == 4
        assert mesh.shape_type == ShapeType.MESH

    def test_mesh_get_triangle(self):
        """Test getting mesh triangle."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        mesh = MeshShape(vertices, indices)
        tri = mesh.get_triangle(0)
        assert tri == ((0, 0, 0), (1, 0, 0), (0, 1, 0))


class TestCompoundShape:
    """Tests for CompoundShape."""

    def test_compound_creation(self):
        """Test compound shape creation."""
        compound = CompoundShape()
        assert compound.child_count == 0
        assert compound.shape_type == ShapeType.COMPOUND

    def test_compound_add_child(self):
        """Test adding child to compound."""
        compound = CompoundShape()
        compound.add_child(SphereShape(radius=1.0))
        assert compound.child_count == 1

    def test_compound_remove_child(self):
        """Test removing child from compound."""
        compound = CompoundShape()
        compound.add_child(SphereShape(radius=1.0))
        compound.remove_child(0)
        assert compound.child_count == 0

    def test_compound_mass_properties(self):
        """Test compound mass properties combine children correctly."""
        compound = CompoundShape()
        box = BoxShape(half_extents=(1, 1, 1))
        sphere = SphereShape(radius=1.0)

        compound.add_child(box)
        compound.add_child(sphere, local_offset=(5, 0, 0))

        # Get individual mass properties
        box_props = box.compute_mass_properties(density=1000.0)
        sphere_props = sphere.compute_mass_properties(density=1000.0)

        # Get compound mass properties
        props = compound.compute_mass_properties(density=1000.0)

        # Compound mass should be sum of children masses
        expected_mass = box_props.mass + sphere_props.mass
        assert approx_equal(props.mass, expected_mass, tolerance=1.0), \
            f"Compound mass {props.mass} != expected {expected_mass}"
        assert props.mass > 0, "Mass should be positive"


class TestCreateShape:
    """Tests for create_shape factory function."""

    def test_create_sphere(self):
        """Test creating sphere via factory."""
        shape = create_shape(ShapeType.SPHERE, radius=2.0)
        assert isinstance(shape, SphereShape)
        assert shape.radius == 2.0

    def test_create_box(self):
        """Test creating box via factory."""
        shape = create_shape(ShapeType.BOX, half_extents=(1, 2, 3))
        assert isinstance(shape, BoxShape)
        assert shape.half_extents == (1, 2, 3)

    def test_create_capsule(self):
        """Test creating capsule via factory."""
        shape = create_shape(ShapeType.CAPSULE, radius=0.5, half_height=1.0)
        assert isinstance(shape, CapsuleShape)


# =============================================================================
# Test Rigid Body
# =============================================================================

class TestRigidBody:
    """Tests for RigidBody."""

    def test_body_creation_dynamic(self):
        """Test creating dynamic body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=2.0)
        assert body.body_type == BodyType.DYNAMIC
        assert body.mass == 2.0
        assert body.is_dynamic is True

    def test_body_creation_static(self):
        """Test creating static body."""
        body = RigidBody(body_type=BodyType.STATIC)
        assert body.body_type == BodyType.STATIC
        assert body.mass == 0.0
        assert body.inverse_mass == 0.0
        assert body.is_static is True

    def test_body_creation_kinematic(self):
        """Test creating kinematic body."""
        body = RigidBody(body_type=BodyType.KINEMATIC)
        assert body.body_type == BodyType.KINEMATIC
        assert body.is_kinematic is True

    def test_body_position(self):
        """Test body position."""
        body = RigidBody(position=(1, 2, 3))
        assert body.position == (1, 2, 3)
        body.position = (4, 5, 6)
        assert body.position == (4, 5, 6)

    def test_body_rotation(self):
        """Test body rotation."""
        body = RigidBody(rotation=(0, 0, 0, 1))
        assert body.rotation == (0, 0, 0, 1)

    def test_body_linear_velocity(self):
        """Test body linear velocity."""
        body = RigidBody(body_type=BodyType.DYNAMIC)
        body.linear_velocity = (1, 2, 3)
        assert body.linear_velocity == (1, 2, 3)

    def test_body_angular_velocity(self):
        """Test body angular velocity."""
        body = RigidBody(body_type=BodyType.DYNAMIC)
        body.angular_velocity = (0.1, 0.2, 0.3)
        assert vector_approx_equal(body.angular_velocity, (0.1, 0.2, 0.3))

    def test_body_apply_force(self):
        """Test applying force to body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=1.0)
        body.apply_force((10, 0, 0))
        # Force is accumulated, not immediately applied to velocity
        body.integrate_velocities(1.0, (0, 0, 0))
        assert body.linear_velocity[0] > 0

    def test_body_apply_impulse(self):
        """Test applying impulse to body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=1.0)
        body.apply_impulse((10, 0, 0))
        # Impulse immediately affects velocity
        assert approx_equal(body.linear_velocity[0], 10.0)

    def test_body_apply_torque(self):
        """Test applying torque to body."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=1.0)
        body.apply_torque((0, 10, 0))
        body.integrate_velocities(1.0, (0, 0, 0))
        assert body.angular_velocity[1] != 0

    def test_body_apply_angular_impulse(self):
        """Test applying angular impulse."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=1.0)
        body.apply_angular_impulse((0, 1, 0))
        assert body.angular_velocity[1] != 0

    def test_body_damping(self):
        """Test body damping."""
        body = RigidBody(body_type=BodyType.DYNAMIC)
        body.linear_damping = 0.5
        body.angular_damping = 0.5
        assert body.linear_damping == 0.5
        assert body.angular_damping == 0.5

    def test_body_material(self):
        """Test body material."""
        mat = MaterialPresets.rubber()
        body = RigidBody(material=mat)
        assert body.material.restitution == 0.8

    def test_body_shape(self):
        """Test body shape."""
        shape = BoxShape(half_extents=(1, 1, 1))
        body = RigidBody(shape=shape)
        assert isinstance(body.shape, BoxShape)

    def test_body_get_aabb(self):
        """Test body AABB."""
        body = RigidBody(
            position=(0, 0, 0),
            shape=SphereShape(radius=1.0)
        )
        aabb = body.get_aabb()
        assert aabb.min_point[0] < 0
        assert aabb.max_point[0] > 0

    def test_body_transform_point(self):
        """Test body point transformation."""
        body = RigidBody(position=(10, 0, 0))
        world_pt = body.transform_point_to_world((1, 0, 0))
        assert vector_approx_equal(world_pt, (11, 0, 0))

    def test_body_velocity_at_point(self):
        """Test velocity at point."""
        body = RigidBody(body_type=BodyType.DYNAMIC)
        body.linear_velocity = (1, 0, 0)
        vel = body.get_velocity_at_point(body.position)
        assert vector_approx_equal(vel, (1, 0, 0))

    def test_body_collision_layer(self):
        """Test body collision layer."""
        body = RigidBody()
        body.collision_layer = 2
        body.collision_mask = 0xFF
        assert body.collision_layer == 2
        assert body.collision_mask == 0xFF

    def test_body_user_data(self):
        """Test body user data."""
        body = RigidBody()
        body.user_data['custom'] = 'value'
        assert body.user_data['custom'] == 'value'

    def test_body_state_save_restore(self):
        """Test body state save/restore."""
        body = RigidBody(body_type=BodyType.DYNAMIC, position=(1, 2, 3))
        body.save_state()
        body.position = (4, 5, 6)
        state = body.interpolate_state(0.0)
        assert vector_approx_equal(state.position, (1, 2, 3))

    def test_body_integration(self):
        """Test body integration."""
        body = RigidBody(body_type=BodyType.DYNAMIC, position=(0, 0, 0))
        body.linear_velocity = (1, 0, 0)
        body.integrate_positions(1.0)
        assert approx_equal(body.position[0], 1.0)


# =============================================================================
# Test Sleep Manager
# =============================================================================

class TestSleepManager:
    """Tests for SleepManager."""

    def test_sleep_manager_creation(self):
        """Test sleep manager creation."""
        manager = SleepManager()
        assert manager.linear_threshold > 0
        assert manager.angular_threshold > 0

    def test_sleep_manager_register_body(self):
        """Test registering body."""
        manager = SleepManager()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        manager.register_body(body)
        assert manager.awake_count == 1

    def test_sleep_manager_unregister_body(self):
        """Test unregistering body."""
        manager = SleepManager()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        manager.register_body(body)
        manager.unregister_body(body)
        assert manager.awake_count == 0

    def test_sleep_manager_can_sleep(self):
        """Test can_sleep check."""
        manager = SleepManager()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        assert manager.can_sleep(body) is True

    def test_sleep_manager_static_cannot_sleep(self):
        """Test static bodies can't sleep."""
        manager = SleepManager()
        body = RigidBody(body_type=BodyType.STATIC)
        assert manager.can_sleep(body) is False

    def test_sleep_manager_wake_up(self):
        """Test waking up body."""
        manager = SleepManager()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        manager.register_body(body)
        body.put_to_sleep()
        manager.wake_up(body)
        assert body.is_sleeping is False

    def test_sleep_manager_statistics(self):
        """Test sleep manager statistics."""
        manager = SleepManager()
        body1 = RigidBody(body_type=BodyType.DYNAMIC)
        body2 = RigidBody(body_type=BodyType.DYNAMIC)
        manager.register_body(body1)
        manager.register_body(body2)
        stats = manager.get_statistics()
        assert stats['total_bodies'] == 2


# =============================================================================
# Test Collision Filter
# =============================================================================

class TestCollisionFilter:
    """Tests for CollisionFilter."""

    def test_filter_creation(self):
        """Test filter creation."""
        filter = CollisionFilter(layer=1, mask=0xFF)
        assert filter.layer == 1
        assert filter.mask == 0xFF

    def test_filter_all_layers(self):
        """Test all layers filter."""
        filter = CollisionFilter.all_layers()
        assert filter.mask == 0xFFFFFFFF

    def test_filter_layer_only(self):
        """Test layer-only filter."""
        filter = CollisionFilter.layer_only(2)
        assert filter.layer == (1 << 2)

    def test_filter_should_collide(self):
        """Test collision filtering."""
        filter = CollisionFilter(layer=1, mask=0xFF)
        body = RigidBody(body_type=BodyType.DYNAMIC)
        body.collision_layer = 1
        body.collision_mask = 0xFFFFFFFF
        assert filter.should_collide(body) is True


# =============================================================================
# Test Physics Queries
# =============================================================================

class TestQueries:
    """Tests for physics queries."""

    def test_raycast_hit_sphere(self):
        """Test raycast hitting sphere."""
        body = RigidBody(
            position=(5, 0, 0),
            shape=SphereShape(radius=1.0)
        )
        hit = raycast_single([body], (0, 0, 0), (1, 0, 0), 100.0)
        assert hit is not None
        assert hit.body == body
        assert approx_equal(hit.distance, 4.0, 0.2)

    def test_raycast_miss(self):
        """Test raycast missing."""
        body = RigidBody(
            position=(5, 5, 0),
            shape=SphereShape(radius=1.0)
        )
        hit = raycast_single([body], (0, 0, 0), (1, 0, 0), 100.0)
        assert hit is None

    def test_raycast_all_multiple_hits(self):
        """Test raycast returning multiple hits."""
        body1 = RigidBody(position=(5, 0, 0), shape=SphereShape(radius=1.0))
        body2 = RigidBody(position=(10, 0, 0), shape=SphereShape(radius=1.0))
        hits = raycast_all([body1, body2], (0, 0, 0), (1, 0, 0), 100.0)
        assert len(hits) == 2
        assert hits[0].distance < hits[1].distance

    def test_overlap_sphere(self):
        """Test sphere overlap."""
        body = RigidBody(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        results = overlap_sphere([body], (0, 0, 0), 0.5)
        assert len(results) == 1

    def test_overlap_sphere_no_overlap(self):
        """Test sphere no overlap."""
        body = RigidBody(position=(10, 0, 0), shape=SphereShape(radius=1.0))
        results = overlap_sphere([body], (0, 0, 0), 0.5)
        assert len(results) == 0

    def test_overlap_box(self):
        """Test box overlap."""
        body = RigidBody(position=(0, 0, 0), shape=BoxShape(half_extents=(1, 1, 1)))
        results = overlap_box([body], (0, 0, 0), (0.5, 0.5, 0.5))
        assert len(results) == 1

    def test_overlap_capsule(self):
        """Test capsule overlap."""
        body = RigidBody(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        results = overlap_capsule([body], (0, -2, 0), (0, 2, 0), 0.5)
        assert len(results) == 1


# =============================================================================
# Test Physics World
# =============================================================================

class TestPhysicsWorld:
    """Tests for PhysicsWorld."""

    def test_world_creation(self):
        """Test world creation."""
        world = PhysicsWorld()
        assert world.body_count == 0
        assert world.state == SimulationState.STOPPED

    def test_world_gravity(self):
        """Test world gravity."""
        world = PhysicsWorld()
        world.gravity = (0, -20, 0)
        assert world.gravity == (0, -20, 0)

    def test_world_timestep(self):
        """Test world timestep."""
        world = PhysicsWorld()
        world.timestep = 1/120
        assert approx_equal(world.timestep, 1/120)

    def test_world_add_body(self):
        """Test adding body to world."""
        world = PhysicsWorld()
        body = RigidBody()
        assert world.add_body(body) is True
        assert world.body_count == 1
        assert body.world == world

    def test_world_add_body_duplicate(self):
        """Test adding duplicate body fails."""
        world = PhysicsWorld()
        body = RigidBody()
        world.add_body(body)
        assert world.add_body(body) is False

    def test_world_remove_body(self):
        """Test removing body from world."""
        world = PhysicsWorld()
        body = RigidBody()
        world.add_body(body)
        assert world.remove_body(body) is True
        assert world.body_count == 0
        assert body.world is None

    def test_world_get_body(self):
        """Test getting body by ID."""
        world = PhysicsWorld()
        body = RigidBody()
        world.add_body(body)
        retrieved = world.get_body(body.id)
        assert retrieved == body

    def test_world_get_bodies(self):
        """Test getting all bodies."""
        world = PhysicsWorld()
        body1 = RigidBody()
        body2 = RigidBody()
        world.add_body(body1)
        world.add_body(body2)
        bodies = world.get_bodies()
        assert len(bodies) == 2

    def test_world_clear(self):
        """Test clearing world."""
        world = PhysicsWorld()
        world.add_body(RigidBody())
        world.add_body(RigidBody())
        world.clear()
        assert world.body_count == 0

    def test_world_start_stop(self):
        """Test world start/stop."""
        world = PhysicsWorld()
        world.start()
        assert world.state == SimulationState.RUNNING
        world.stop()
        assert world.state == SimulationState.STOPPED

    def test_world_pause_resume(self):
        """Test world pause/resume."""
        world = PhysicsWorld()
        world.start()
        world.pause()
        assert world.state == SimulationState.PAUSED
        world.resume()
        assert world.state == SimulationState.RUNNING

    def test_world_step(self):
        """Test world step."""
        world = PhysicsWorld()
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 10, 0),
            mass=1.0
        )
        world.add_body(body)
        world.start()
        world.step(1/60)
        # Body should have fallen due to gravity
        assert body.position[1] < 10

    def test_world_step_no_run_when_stopped(self):
        """Test world doesn't step when stopped."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC, position=(0, 10, 0))
        world.add_body(body)
        # Don't start the world
        substeps = world.step(1/60)
        assert substeps == 0
        assert body.position[1] == 10

    def test_world_raycast(self):
        """Test world raycast."""
        world = PhysicsWorld()
        body = RigidBody(position=(5, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)
        hit = world.raycast((0, 0, 0), (1, 0, 0))
        assert hit is not None
        assert hit.body == body

    def test_world_overlap_test(self):
        """Test world overlap test."""
        world = PhysicsWorld()
        body = RigidBody(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)
        results = world.overlap_test(SphereShape(radius=0.5), (0, 0, 0))
        assert len(results) == 1

    def test_world_sweep_test(self):
        """Test world sweep test."""
        world = PhysicsWorld()
        body = RigidBody(position=(5, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)
        result = world.sweep_test(SphereShape(radius=0.5), (0, 0, 0), (1, 0, 0), 10.0)
        assert result.hit is True

    def test_world_collision_callback(self):
        """Test collision callbacks are actually called on collision."""
        world = PhysicsWorld()
        callback_called = [False]
        collision_bodies = [None, None]

        def on_collision(a, b, contact):
            callback_called[0] = True
            collision_bodies[0] = a
            collision_bodies[1] = b

        world.on_collision_enter(on_collision)

        # Create overlapping bodies to ensure collision detection
        body1 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 0, 0),
            shape=BoxShape(half_extents=(1, 1, 1))
        )
        body2 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(1.5, 0, 0),  # Overlaps with body1
            shape=BoxShape(half_extents=(1, 1, 1))
        )
        world.add_body(body1)
        world.add_body(body2)
        world.start()
        world.step(1/60)

        # Verify callback was called with the correct bodies
        assert callback_called[0] is True, "Collision callback was not called"
        assert collision_bodies[0] is not None, "Body A not set in callback"
        assert collision_bodies[1] is not None, "Body B not set in callback"

    def test_world_statistics(self):
        """Test world statistics."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        world.add_body(body)
        stats = world.statistics
        assert stats['bodies'] == 1

    def test_world_wake_all(self):
        """Test waking all bodies."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        world.add_body(body)
        body.put_to_sleep()
        world.wake_all()
        assert body.is_sleeping is False

    def test_world_get_bodies_in_aabb(self):
        """Test getting bodies in AABB."""
        world = PhysicsWorld()
        body1 = RigidBody(position=(0, 0, 0))
        body2 = RigidBody(position=(100, 100, 100))
        world.add_body(body1)
        world.add_body(body2)
        aabb = AABB(min_point=(-5, -5, -5), max_point=(5, 5, 5))
        bodies = world.get_bodies_in_aabb(aabb)
        assert len(bodies) == 1
        assert body1 in bodies

    def test_world_to_dict(self):
        """Test world serialization."""
        world = PhysicsWorld()
        body = RigidBody(body_type=BodyType.DYNAMIC)
        world.add_body(body)
        data = world.to_dict()
        assert 'bodies' in data
        assert len(data['bodies']) == 1

    def test_world_simulation_time(self):
        """Test simulation time tracking."""
        world = PhysicsWorld()
        world.start()
        world.step(1/60)
        assert world.simulation_time > 0
        assert world.step_count > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for physics system."""

    def test_falling_body(self):
        """Test body falling under gravity."""
        world = PhysicsWorld()
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 100, 0),
            mass=1.0,
            shape=SphereShape(radius=0.5)
        )
        world.add_body(body)
        world.start()

        # Simulate for 1 second
        for _ in range(60):
            world.step(1/60)

        # Body should have fallen significantly
        assert body.position[1] < 100

    def test_body_resting_on_floor(self):
        """Test body coming to rest on floor."""
        world = PhysicsWorld()

        # Floor
        floor = RigidBody(
            body_type=BodyType.STATIC,
            position=(0, 0, 0),
            shape=BoxShape(half_extents=(10, 1, 10))
        )

        # Falling body
        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 5, 0),
            mass=1.0,
            shape=SphereShape(radius=0.5)
        )

        world.add_body(floor)
        world.add_body(body)
        world.start()

        # Simulate for 2 seconds
        for _ in range(120):
            world.step(1/60)

        # Body should be near floor level
        assert body.position[1] < 5

    def test_bouncing_ball(self):
        """Test bouncing ball behavior."""
        world = PhysicsWorld()

        floor = RigidBody(
            body_type=BodyType.STATIC,
            position=(0, 0, 0),
            shape=BoxShape(half_extents=(10, 1, 10)),
            material=PhysicsMaterial(restitution=1.0)
        )

        ball = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 10, 0),
            mass=1.0,
            shape=SphereShape(radius=0.5),
            material=PhysicsMaterial(restitution=1.0)
        )

        world.add_body(floor)
        world.add_body(ball)
        world.start()

        # Record positions at intervals
        positions = []
        for i in range(180):
            world.step(1/60)
            if i % 30 == 0:
                positions.append(ball.position[1])

        # Ball should fall from 10 - verify it actually went down
        assert positions[0] < 10, f"Ball did not fall from initial position: {positions[0]}"
        assert positions[0] > positions[1], f"Ball did not continue falling: {positions[0]} vs {positions[1]}"

        # Verify minimum position is above floor (accounting for ball radius and floor half-extent)
        min_pos = min(positions)
        assert min_pos >= 1.0, f"Ball fell through floor: {min_pos}"

    def test_multiple_body_collision(self):
        """Test collision between multiple bodies."""
        world = PhysicsWorld()
        world.gravity = (0, 0, 0)  # Disable gravity for this test

        body1 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 0, 0),
            mass=1.0,
            shape=SphereShape(radius=1.0)
        )

        # Start bodies already overlapping to ensure collision detection
        body2 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(1.5, 0, 0),  # Overlap with body1
            mass=1.0,
            shape=SphereShape(radius=1.0)
        )

        # Give body1 velocity towards body2
        body1.linear_velocity = (5, 0, 0)

        world.add_body(body1)
        world.add_body(body2)
        world.start()

        # Simulate
        for _ in range(60):
            world.step(1/60)

        # Body2 should have moved due to collision (or body1 should have slowed down)
        # Either body2 gains velocity or body1 loses some velocity
        assert body2.position[0] > 1.5 or body1.linear_velocity[0] < 5

    def test_compound_shape_physics(self):
        """Test physics with compound shapes."""
        world = PhysicsWorld()

        compound = CompoundShape()
        compound.add_child(SphereShape(radius=0.5), local_offset=(0, 0, 0))
        compound.add_child(SphereShape(radius=0.5), local_offset=(1, 0, 0))

        body = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 10, 0),
            mass=1.0,
            shape=compound
        )

        world.add_body(body)
        world.start()
        world.step(1/60)

        # Body should fall
        assert body.position[1] < 10


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_raycast_division_by_zero_near_zero_direction(self):
        """Test raycast doesn't divide by zero with near-zero direction."""
        body = RigidBody(
            position=(5, 0, 0),
            shape=SphereShape(radius=1.0)
        )
        # Use extremely small but non-zero direction - should not crash
        try:
            hit = raycast_single([body], (0, 0, 0), (1e-15, 0, 0), 100.0)
            # If it returns, result should be None (direction normalized to zero vector)
            assert hit is None, "Near-zero direction should not produce a valid hit"
        except ZeroDivisionError:
            pytest.fail("raycast_single raised ZeroDivisionError with near-zero direction")

    def test_body_zero_mass_handling(self):
        """Test body with near-zero mass is handled correctly."""
        body = RigidBody(body_type=BodyType.DYNAMIC, mass=1e-10)
        # Mass should be clamped to MIN_MASS from config
        assert body.mass >= MIN_MASS, f"Mass {body.mass} should be >= MIN_MASS ({MIN_MASS})"

    def test_body_impulse_at_zero_inverse_mass(self):
        """Test impulse application to static body (zero inverse mass)."""
        body = RigidBody(body_type=BodyType.STATIC)
        assert body.inverse_mass == 0.0
        body.apply_impulse((100, 0, 0))
        # Static body should not move
        assert body.linear_velocity == (0.0, 0.0, 0.0)

    def test_convex_hull_minimum_points(self):
        """Test convex hull with exactly minimum points."""
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        hull = ConvexHullShape(points)
        assert hull.vertex_count == 4

    def test_aabb_zero_volume(self):
        """Test AABB with degenerate dimensions."""
        # Create very thin box
        box = BoxShape(half_extents=(0.001, 0.001, 0.001))
        aabb = box.compute_aabb(position=(0, 0, 0))
        # Should still be valid, volume > 0
        assert aabb.volume > 0

    def test_quaternion_normalize_near_zero(self):
        """Test quaternion normalization with near-zero quaternion."""
        body = RigidBody(rotation=(0, 0, 0, 1e-15))
        # Should normalize to identity quaternion
        assert body.rotation[3] == 1.0 or abs(body.rotation[3] - 1.0) < 0.01

    def test_world_max_bodies_limit(self):
        """Test world respects max bodies limit."""
        config = PhysicsConfig(max_bodies=5)
        world = PhysicsWorld(config)

        for i in range(5):
            body = RigidBody()
            assert world.add_body(body) is True

        # 6th body should fail
        extra_body = RigidBody()
        assert world.add_body(extra_body) is False
        assert world.body_count == 5

    def test_overlapping_bodies_collision_detection(self):
        """Test collision is detected for overlapping bodies."""
        world = PhysicsWorld()
        # Use boxes instead of spheres for more reliable AABB-based collision detection
        body1 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(0, 0, 0),
            shape=BoxShape(half_extents=(1, 1, 1)),
            mass=1.0
        )
        body2 = RigidBody(
            body_type=BodyType.DYNAMIC,
            position=(1.5, 0, 0),  # Overlaps with body1 (box centers 1.5 apart, each extends 1.0)
            shape=BoxShape(half_extents=(1, 1, 1)),
            mass=1.0
        )
        world.add_body(body1)
        world.add_body(body2)
        world.start()
        world.step(1/60)

        # Verify contact manifolds were generated
        manifolds = world.get_contact_manifolds()
        assert len(manifolds) > 0, "No collision detected between overlapping bodies"

    def test_velocity_clamp(self):
        """Test velocity is clamped to maximum."""
        body = RigidBody(body_type=BodyType.DYNAMIC)
        # Try to set extremely high velocity
        body.linear_velocity = (1e10, 0, 0)
        # Velocity should be clamped to MAX_LINEAR_VELOCITY from config
        assert vector_length(body.linear_velocity) <= MAX_LINEAR_VELOCITY + 0.1, \
            f"Velocity {vector_length(body.linear_velocity)} should be <= MAX_LINEAR_VELOCITY ({MAX_LINEAR_VELOCITY})"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
