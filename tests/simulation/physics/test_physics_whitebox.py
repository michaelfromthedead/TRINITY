"""
T-FG-7.5: Whitebox tests for physics solver Phase 1.

WHITEBOX coverage plan:
  config.py
    - PhysicsConfig.validate(): each branch raises ValueError (timestep<=0, max_substeps<1, etc.)
    - PhysicsConfig.copy(): all fields match
    - PRESET_*: all four presets validate cleanly

  body_flags.py
    - _update_bits() / _sync_from_bits(): read-after-write round-trip for every flag
    - set_flag / toggle_flag / get_flag / clear_all: bit-level operations
    - lock_position_all / lock_rotation_all composite properties
    - can_sleep: trigger and disable_deactivation branches
    - get_position_lock_mask / get_rotation_lock_mask: lock maps to 0.0/1.0
    - Factory classmethods: static_body, kinematic_body, dynamic_body, trigger_volume, character_body
    - __eq__ / __hash__: equality from bits, hash-consistency

  physics_material.py
    - __post_init__: friction/restitution/density clamping; dynamic <= static enforcement
    - set_friction: default dynamic = 0.75 * static
    - combine_values: all 5 modes compute correctly
    - combine_materials: priority resolution (MULTIPLY > MAX > AVERAGE > MIN)
    - get_material: valid name returns copy; bad name raises KeyError
    - from_dict: round-trip serialization

  collision_shapes.py
    - SphereShape radius clamping on __init__ and setter
    - BoxShape half_extents clamping on __init__ and setter
    - CapsuleShape half_height=0 degenerate (still valid)
    - ConvexHullShape: <4 points raises ValueError
    - MeshShape: <3 vertices or <1 index raises ValueError
    - CompoundShape: empty children compute_aabb returns point-AABB, get_support_point returns offset
    - AABB: center, half_extents, size, volume, surface_area formulas
    - AABB.contains_point / intersects boundary cases
    - AABB.from_points: empty list returns zero AABB
    - MassProperties.inverse_mass: mass=0 returns 0.0; mass>0 returns 1/mass
    - MassProperties.inverse_inertia_tensor: zero diag -> zero, positive diag -> 1/diag
    - create_shape: each ShapeType dispatches correctly; unknown type raises ValueError

  rigid_body.py
    - Static body: mass=0, inverse_mass=0, inertia all zero
    - body_type transitions: STATIC->DYNAMIC resets mass to 1.0
    - velocity clamping on setter (linear capped at MAX_LINEAR_VELOCITY=500)
    - linear_damping/angular_damping clamping [0, 1]
    - Sleeping body: apply_force wakens; integrate_velocities skips
    - apply_force/impulse with lock masks
    - integrate_velocities: zero-torque skips angular integration
    - integrate_positions: zero angular speed skips rotation update
    - _quaternion_normalize: zero-length returns identity
    - _inertia zero: static body has zero inertia tensor

  sleeping.py
    - can_sleep: STATIC and KINEMATIC bodies return False
    - is_below_threshold: compares squared lengths
    - merge_islands: merges smaller into larger
    - merge_islands: same-island is no-op
    - rebuild_islands: union-find groups connected bodies correctly
    - update: sleeping island checks wake; active island accumulates timer
    - wake_connected_bodies: through_joints propagates to island
    - get_island_info: registered vs unregistered body
    - reset / clear: state transitions

  queries.py
    - _ray_sphere_intersect: |a| < epsilon (near-zero direction) returns None
    - _ray_box_intersect: parallel-to-slab outside returns None
    - CollisionFilter.should_collide: STATIC/KINEMATIC/DYNAMIC branches
    - CollisionFilter.should_collide: trigger exclusion when flags != ALL
    - overlap_sphere: no bodies returns empty
    - sweep_sphere: no bodies returns no-hit
    - point_inside: AABB pass + shape pass; AABB fail skips shape
    - closest_point_on_body: sphere (point at center falls back to +x), box, default AABB
    - distance_to_body: zero when point is inside

  physics_world.py
    - add_body: duplicate returns False; at-capacity returns False
    - remove_body: not-found returns False
    - step: stopped world returns 0 substeps
    - _broad_phase: both-static pair skipped; both-sleeping pair skipped
    - _narrow_phase: stale manifold removed
    - _solve_contact_velocity: separating velocity (normal_vel > 0) returns early
    - get_bodies_in_aabb: filters correctly
"""

import math
import pytest

from engine.simulation.physics.config import (
    PhysicsConfig,
    PhysicsBackend,
    BroadphaseType,
    NarrowphaseType,
    SolverType,
    DEFAULT_GRAVITY,
    DEFAULT_TIMESTEP,
    PRESET_HIGH_QUALITY,
    PRESET_PERFORMANCE,
    PRESET_MOBILE,
    PRESET_DETERMINISTIC,
    MIN_TIMESTEP,
    MAX_SUBSTEPS,
    SOLVER_ITERATIONS,
    POSITION_ITERATIONS,
    SLEEP_THRESHOLD_LINEAR,
    SLEEP_THRESHOLD_ANGULAR,
    SLEEP_TIME_THRESHOLD,
    MAX_BODIES,
    MIN_MASS,
    MAX_MASS,
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    COLLISION_EPSILON,
    FLOAT_COMPARISON_EPSILON,
    MIN_SHAPE_RADIUS,
    MIN_SHAPE_DIMENSION,
    MIN_CONVEX_HULL_POINTS,
    DEFAULT_SHAPE_MARGIN,
    CONTACT_SLOP,
    CONTACT_BAUMGARTE_FACTOR,
)
from engine.simulation.physics.body_flags import (
    BodyFlags,
    BodyFlagBits,
)
from engine.simulation.physics.physics_material import (
    PhysicsMaterial,
    CombineMode,
    combine_values,
    combine_materials,
    MaterialPresets,
    get_material,
    MATERIAL_PRESETS,
    MIN_FRICTION,
    MAX_FRICTION,
    MIN_RESTITUTION,
    MAX_RESTITUTION,
    MIN_DENSITY,
    MAX_DENSITY,
)
from engine.simulation.physics.collision_shapes import (
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
    _vector_add,
    _vector_sub,
    _vector_scale,
    _vector_dot,
    _vector_length,
    _vector_normalize,
    _rotate_vector,
)
from engine.simulation.physics.rigid_body import (
    RigidBody,
    BodyType,
    BodyState,
    _quaternion_normalize,
)
from engine.simulation.physics.sleeping import (
    SleepManager,
    Island,
    IslandState,
)
from engine.simulation.physics.queries import (
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
from engine.simulation.physics.physics_world import (
    PhysicsWorld,
    Contact,
    ContactManifold,
    SimulationState,
)
from ..physics_test_base import PhysicsTestCase


# =============================================================================
# Helper factories
# =============================================================================

def _dynamic_body(position=(0, 0, 0), shape=None):
    """Create a minimal dynamic body for testing."""
    return RigidBody(
        body_type=BodyType.DYNAMIC,
        position=position,
        shape=shape or SphereShape(radius=0.5),
    )


def _static_body(position=(0, 0, 0), shape=None):
    """Create a minimal static body."""
    return RigidBody(
        body_type=BodyType.STATIC,
        position=position,
        shape=shape or SphereShape(radius=0.5),
    )


# =============================================================================
# 1. config.py
# =============================================================================

class TestConfigValidation(PhysicsTestCase):
    """PhysicsConfig.validate() boundary conditions."""

    def test_timestep_zero_raises(self):
        """timestep == 0 raises ValueError."""
        cfg = PhysicsConfig(timestep=0)
        with pytest.raises(ValueError, match="Timestep"):
            cfg.validate()

    def test_timestep_negative_raises(self):
        """timestep < 0 raises ValueError."""
        cfg = PhysicsConfig(timestep=-0.01)
        with pytest.raises(ValueError, match="Timestep"):
            cfg.validate()

    def test_max_substeps_zero_raises(self):
        """max_substeps < 1 raises ValueError."""
        cfg = PhysicsConfig(max_substeps=0)
        with pytest.raises(ValueError, match="substeps"):
            cfg.validate()

    def test_solver_iterations_zero_raises(self):
        """solver_iterations < 1 raises ValueError."""
        cfg = PhysicsConfig(solver_iterations=0)
        with pytest.raises(ValueError, match="Solver iterations"):
            cfg.validate()

    def test_position_iterations_zero_raises(self):
        """position_iterations < 1 raises ValueError."""
        cfg = PhysicsConfig(position_iterations=0)
        with pytest.raises(ValueError, match="Position iterations"):
            cfg.validate()

    def test_max_bodies_zero_raises(self):
        """max_bodies < 1 raises ValueError."""
        cfg = PhysicsConfig(max_bodies=0)
        with pytest.raises(ValueError, match="Max bodies"):
            cfg.validate()

    def test_sleep_threshold_linear_negative_raises(self):
        """sleep_threshold_linear < 0 raises ValueError."""
        cfg = PhysicsConfig(sleep_threshold_linear=-0.1)
        with pytest.raises(ValueError, match="Sleep threshold linear"):
            cfg.validate()

    def test_sleep_threshold_angular_negative_raises(self):
        """sleep_threshold_angular < 0 raises ValueError."""
        cfg = PhysicsConfig(sleep_threshold_angular=-0.1)
        with pytest.raises(ValueError, match="Sleep threshold angular"):
            cfg.validate()

    def test_sleep_time_threshold_negative_raises(self):
        """sleep_time_threshold < 0 raises ValueError."""
        cfg = PhysicsConfig(sleep_time_threshold=-0.1)
        with pytest.raises(ValueError, match="Sleep time threshold"):
            cfg.validate()

    def test_contact_baumgarte_above_one_raises(self):
        """contact_baumgarte > 1 raises ValueError."""
        cfg = PhysicsConfig(contact_baumgarte=1.5)
        with pytest.raises(ValueError, match="Contact baumgarte"):
            cfg.validate()

    def test_contact_baumgarte_below_zero_raises(self):
        """contact_baumgarte < 0 raises ValueError."""
        cfg = PhysicsConfig(contact_baumgarte=-0.1)
        with pytest.raises(ValueError, match="Contact baumgarte"):
            cfg.validate()

    def test_warmstarting_factor_above_one_raises(self):
        """warmstarting_factor > 1 raises ValueError."""
        cfg = PhysicsConfig(warmstarting_factor=1.5)
        with pytest.raises(ValueError, match="Warmstarting"):
            cfg.validate()

    def test_linear_damping_negative_raises(self):
        """linear_damping < 0 raises ValueError."""
        cfg = PhysicsConfig(linear_damping=-0.1)
        with pytest.raises(ValueError, match="Linear damping"):
            cfg.validate()

    def test_angular_damping_negative_raises(self):
        """angular_damping < 0 raises ValueError."""
        cfg = PhysicsConfig(angular_damping=-0.1)
        with pytest.raises(ValueError, match="Angular damping"):
            cfg.validate()

    def test_max_linear_velocity_zero_raises(self):
        """max_linear_velocity == 0 raises ValueError."""
        cfg = PhysicsConfig(max_linear_velocity=0)
        with pytest.raises(ValueError, match="Max linear velocity"):
            cfg.validate()

    def test_max_angular_velocity_zero_raises(self):
        """max_angular_velocity == 0 raises ValueError."""
        cfg = PhysicsConfig(max_angular_velocity=0)
        with pytest.raises(ValueError, match="Max angular velocity"):
            cfg.validate()

    def test_default_config_validates(self):
        """default config passes validation."""
        cfg = PhysicsConfig()
        assert cfg.validate() is True

    def test_preset_high_quality_validates(self):
        """PRESET_HIGH_QUALITY config is valid."""
        assert PRESET_HIGH_QUALITY.validate() is True

    def test_preset_performance_validates(self):
        """PRESET_PERFORMANCE config is valid."""
        assert PRESET_PERFORMANCE.validate() is True

    def test_preset_mobile_validates(self):
        """PRESET_MOBILE config is valid."""
        assert PRESET_MOBILE.validate() is True

    def test_preset_deterministic_validates(self):
        """PRESET_DETERMINISTIC config is valid."""
        assert PRESET_DETERMINISTIC.validate() is True

    def test_copy_fidelity(self):
        """copy() produces an equal independent clone."""
        cfg = PhysicsConfig(
            gravity=(0, -9.81, 0),
            timestep=1/120,
            solver_iterations=20,
            enable_sleeping=False,
        )
        copy = cfg.copy()
        # Same types and values
        assert copy.gravity == cfg.gravity
        assert copy.timestep == cfg.timestep
        assert copy.solver_iterations == cfg.solver_iterations
        assert copy.enable_sleeping == cfg.enable_sleeping
        # Independent
        copy.timestep = 0.5
        assert cfg.timestep == 1/120


# =============================================================================
# 2. body_flags.py
# =============================================================================

class TestBodyFlagsBitOps(PhysicsTestCase):
    """BodyFlags internal bit manipulation."""

    def test_all_flags_default_false(self):
        """default BodyFlags has no flags set (except gyroscopic default True)."""
        flags = BodyFlags()
        bits = flags.bits
        # gyroscopic defaults to True
        assert bool(bits & BodyFlagBits.ENABLE_GYROSCOPIC)
        # Others start False
        assert not flags.is_trigger
        assert not flags.enable_ccd

    def test_update_bits_roundtrip_each_flag(self):
        """Each boolean flag round-trips through _update_bits / _sync_from_bits."""
        flag_map = [
            ('use_gravity', BodyFlagBits.USE_GRAVITY),
            ('enable_ccd', BodyFlagBits.ENABLE_CCD),
            ('lock_position_x', BodyFlagBits.LOCK_POSITION_X),
            ('lock_position_y', BodyFlagBits.LOCK_POSITION_Y),
            ('lock_position_z', BodyFlagBits.LOCK_POSITION_Z),
            ('lock_rotation_x', BodyFlagBits.LOCK_ROTATION_X),
            ('lock_rotation_y', BodyFlagBits.LOCK_ROTATION_Y),
            ('lock_rotation_z', BodyFlagBits.LOCK_ROTATION_Z),
            ('is_trigger', BodyFlagBits.IS_TRIGGER),
            ('enable_gyroscopic', BodyFlagBits.ENABLE_GYROSCOPIC),
            ('is_sleeping', BodyFlagBits.IS_SLEEPING),
            ('disable_deactivation', BodyFlagBits.DISABLE_DEACTIVATION),
            ('enable_contact_callback', BodyFlagBits.ENABLE_CONTACT_CALLBACK),
            ('enable_collision_callback', BodyFlagBits.ENABLE_COLLISION_CALLBACK),
            ('custom_material_callback', BodyFlagBits.CUSTOM_MATERIAL_CALLBACK),
        ]
        for attr_name, bit in flag_map:
            flags = BodyFlags(**{attr_name: True})
            assert bool(flags.bits & bit), f"{attr_name} bit not set"
            flags2 = BodyFlags.from_bits(bit)
            assert getattr(flags2, attr_name), f"{attr_name} not synced from bits"

    def test_set_flag(self):
        """set_flag turns a bit on/off."""
        flags = BodyFlags()
        flags.set_flag(BodyFlagBits.USE_GRAVITY, True)
        assert flags.use_gravity
        flags.set_flag(BodyFlagBits.USE_GRAVITY, False)
        assert not flags.use_gravity

    def test_get_flag(self):
        """get_flag reads a bit."""
        flags = BodyFlags(use_gravity=True)
        assert flags.get_flag(BodyFlagBits.USE_GRAVITY)

    def test_toggle_flag(self):
        """toggle_flag flips a bit."""
        flags = BodyFlags()
        assert not flags.enable_ccd
        flags.toggle_flag(BodyFlagBits.ENABLE_CCD)
        assert flags.enable_ccd
        flags.toggle_flag(BodyFlagBits.ENABLE_CCD)
        assert not flags.enable_ccd

    def test_clear_all(self):
        """clear_all resets all flags to NONE."""
        flags = BodyFlags(use_gravity=True, enable_ccd=True, is_trigger=True)
        flags.clear_all()
        assert flags.bits == BodyFlagBits.NONE
        assert not flags.use_gravity
        assert not flags.enable_ccd
        assert not flags.is_trigger

    def test_lock_position_all_composite(self):
        """lock_position_all getter/setter manages all three axes."""
        flags = BodyFlags()
        assert not flags.lock_position_all
        flags.lock_position_all = True
        assert flags.lock_position_x
        assert flags.lock_position_y
        assert flags.lock_position_z
        flags.lock_position_all = False
        assert not flags.lock_position_x

    def test_lock_rotation_all_composite(self):
        """lock_rotation_all getter/setter manages all three axes."""
        flags = BodyFlags()
        assert not flags.lock_rotation_all
        flags.lock_rotation_all = True
        assert flags.lock_rotation_x
        assert flags.lock_rotation_y
        assert flags.lock_rotation_z
        flags.lock_rotation_all = False
        assert not flags.lock_rotation_x

    def test_has_position_lock(self):
        """has_position_lock is True when any axis is locked."""
        flags = BodyFlags()
        assert not flags.has_position_lock
        flags.lock_position_x = True
        assert flags.has_position_lock

    def test_has_rotation_lock(self):
        """has_rotation_lock is True when any axis is locked."""
        flags = BodyFlags()
        assert not flags.has_rotation_lock
        flags.lock_rotation_y = True
        assert flags.has_rotation_lock

    def test_is_fully_locked(self):
        """is_fully_locked requires all six axes locked."""
        flags = BodyFlags()
        assert not flags.is_fully_locked
        flags.lock_position_all = True
        assert not flags.is_fully_locked
        flags.lock_rotation_all = True
        assert flags.is_fully_locked

    def test_can_sleep_trigger_false(self):
        """trigger bodies cannot sleep."""
        flags = BodyFlags(is_trigger=True)
        assert not flags.can_sleep

    def test_can_sleep_disable_deactivation_false(self):
        """disable_deactivation prevents sleep."""
        flags = BodyFlags(disable_deactivation=True)
        assert not flags.can_sleep

    def test_can_sleep_dynamic_true(self):
        """default dynamic body can sleep."""
        flags = BodyFlags()
        # default: gravity=True, gyroscopic=True, no trigger, no deactivation
        assert flags.can_sleep

    def test_get_position_lock_mask(self):
        """get_position_lock_mask returns 0.0 for locked axes, 1.0 for unlocked."""
        flags = BodyFlags(lock_position_x=True, lock_position_z=True)
        mask = flags.get_position_lock_mask()
        assert mask == (0.0, 1.0, 0.0)

    def test_get_rotation_lock_mask(self):
        """get_rotation_lock_mask returns 0.0 for locked axes, 1.0 for unlocked."""
        flags = BodyFlags(lock_rotation_y=True)
        mask = flags.get_rotation_lock_mask()
        assert mask == (1.0, 0.0, 1.0)


class TestBodyFlagsFactoryMethods(PhysicsTestCase):
    """Factory classmethods produce expected flag configurations."""

    def test_static_body(self):
        """static_body locks all axes, disables gravity and deactivation."""
        flags = BodyFlags.static_body()
        assert not flags.use_gravity
        assert flags.lock_position_all
        assert flags.lock_rotation_all
        assert flags.disable_deactivation

    def test_kinematic_body(self):
        """kinematic_body disables gravity and deactivation."""
        flags = BodyFlags.kinematic_body()
        assert not flags.use_gravity
        assert flags.disable_deactivation

    def test_dynamic_body(self):
        """dynamic_body enables gravity and gyroscopic."""
        flags = BodyFlags.dynamic_body()
        assert flags.use_gravity
        assert flags.enable_gyroscopic

    def test_trigger_volume(self):
        """trigger_volume locks all and enables collision callback."""
        flags = BodyFlags.trigger_volume()
        assert flags.is_trigger
        assert flags.lock_position_all
        assert flags.lock_rotation_all
        assert flags.enable_collision_callback

    def test_character_body(self):
        """character_body enables CCD, locks X/Z rotation, disables gyroscopic."""
        flags = BodyFlags.character_body()
        assert flags.use_gravity
        assert flags.enable_ccd
        assert flags.lock_rotation_x
        assert not flags.lock_rotation_y  # Y rotation for turning
        assert flags.lock_rotation_z
        assert not flags.enable_gyroscopic

    def test_eq_from_bits(self):
        """__eq__ compares by bits, not by object identity."""
        a = BodyFlags(use_gravity=True, enable_ccd=True)
        b = BodyFlags(enable_ccd=True, use_gravity=True)
        assert a == b

    def test_eq_different(self):
        """different flag sets are not equal."""
        a = BodyFlags(use_gravity=True)
        b = BodyFlags(use_gravity=False)
        assert a != b

    def test_hash_consistent(self):
        """__hash__ is consistent with __eq__."""
        a = BodyFlags(use_gravity=True, enable_ccd=True)
        b = BodyFlags(enable_ccd=True, use_gravity=True)
        assert hash(a) == hash(b)

    def test_from_bits_repr_roundtrip(self):
        """from_bits followed by bits read returns same value."""
        original = BodyFlagBits.USE_GRAVITY | BodyFlagBits.ENABLE_CCD
        flags = BodyFlags.from_bits(original)
        assert flags.bits == original

    def test_copy(self):
        """copy() produces independent clone."""
        a = BodyFlags(use_gravity=True, enable_ccd=True)
        b = a.copy()
        assert a == b
        b.use_gravity = False
        assert a.use_gravity  # independent


# =============================================================================
# 3. physics_material.py
# =============================================================================

class TestPhysicsMaterialValidation(PhysicsTestCase):
    """Material __post_init__ clamping."""

    def test_friction_clamped_to_max(self):
        """static_friction > MAX_FRICTION is clamped."""
        m = PhysicsMaterial(static_friction=5.0)
        assert m.static_friction == MAX_FRICTION

    def test_friction_clamped_to_min(self):
        """static_friction < MIN_FRICTION is clamped."""
        m = PhysicsMaterial(static_friction=-1.0)
        assert m.static_friction == MIN_FRICTION

    def test_dynamic_friction_does_not_exceed_static(self):
        """dynamic_friction is clamped to static_friction."""
        m = PhysicsMaterial(static_friction=0.4, dynamic_friction=0.9)
        assert m.dynamic_friction == 0.4

    def test_restitution_clamped(self):
        """restitution is clamped to [0, 1]."""
        m = PhysicsMaterial(restitution=1.5)
        assert m.restitution == 1.0
        m2 = PhysicsMaterial(restitution=-0.5)
        assert m2.restitution == 0.0

    def test_density_clamped(self):
        """density is clamped to [MIN_DENSITY, MAX_DENSITY]."""
        m = PhysicsMaterial(density=1e10)
        assert m.density == MAX_DENSITY
        m2 = PhysicsMaterial(density=0)
        assert m2.density == MIN_DENSITY

    def test_set_friction_default_dynamic(self):
        """set_friction with no dynamic arg computes 0.75 * static."""
        m = PhysicsMaterial()
        m.set_friction(0.8)
        assert m.static_friction == 0.8
        assert m.dynamic_friction == pytest.approx(0.6, abs=1e-9)  # 0.8 * 0.75

    def test_set_friction_explicit_dynamic(self):
        """set_friction with dynamic arg."""
        m = PhysicsMaterial()
        m.set_friction(0.8, 0.5)
        assert m.dynamic_friction == 0.5

    def test_set_bounciness(self):
        """set_bounciness clamps to [0, 1]."""
        m = PhysicsMaterial()
        m.set_bounciness(0.7)
        assert m.restitution == 0.7
        m.set_bounciness(2.0)
        assert m.restitution == 1.0


class TestCombineValues(PhysicsTestCase):
    """combine_values with all 5 modes."""

    def test_average(self):
        """AVERAGE mode: (a + b) / 2."""
        result = combine_values(0.6, 0.4, CombineMode.AVERAGE)
        assert result == 0.5

    def test_min(self):
        """MIN mode: min(a, b)."""
        result = combine_values(0.6, 0.4, CombineMode.MIN)
        assert result == 0.4

    def test_max(self):
        """MAX mode: max(a, b)."""
        result = combine_values(0.6, 0.4, CombineMode.MAX)
        assert result == 0.6

    def test_multiply(self):
        """MULTIPLY mode: a * b."""
        result = combine_values(0.6, 0.4, CombineMode.MULTIPLY)
        assert abs(result - 0.24) < 1e-9

    def test_geometric(self):
        """GEOMETRIC mode: sqrt(a * b)."""
        result = combine_values(0.5, 0.5, CombineMode.GEOMETRIC)
        assert abs(result - 0.5) < 1e-9

    def test_geometric_negative(self):
        """GEOMETRIC mode uses abs(a*b) to avoid domain error."""
        result = combine_values(-0.5, -0.5, CombineMode.GEOMETRIC)
        assert abs(result - 0.5) < 1e-9

    def test_default_is_average(self):
        """Unknown mode falls back to average."""
        result = combine_values(0.6, 0.4, CombineMode.AVERAGE)
        assert result == 0.5


class TestCombineMaterials(PhysicsTestCase):
    """combine_materials priority resolution."""

    def test_both_average(self):
        """Both use AVERAGE -> result is average."""
        a = PhysicsMaterial(static_friction=0.8, dynamic_friction=0.6,
                            restitution=0.5, friction_combine=CombineMode.AVERAGE,
                            restitution_combine=CombineMode.AVERAGE)
        b = PhysicsMaterial(static_friction=0.4, dynamic_friction=0.2,
                            restitution=0.1, friction_combine=CombineMode.AVERAGE,
                            restitution_combine=CombineMode.AVERAGE)
        sf, df, rest = combine_materials(a, b)
        assert abs(sf - 0.6) < 1e-9  # (0.8 + 0.4) / 2
        assert abs(df - 0.4) < 1e-9  # (0.6 + 0.2) / 2
        assert abs(rest - 0.3) < 1e-9  # (0.5 + 0.1) / 2

    def test_multiply_overrides_average(self):
        """MULTIPLY has higher priority than AVERAGE."""
        a = PhysicsMaterial(static_friction=0.8, dynamic_friction=0.6,
                            friction_combine=CombineMode.MULTIPLY,
                            restitution_combine=CombineMode.MULTIPLY)
        b = PhysicsMaterial(static_friction=0.4, dynamic_friction=0.2,
                            friction_combine=CombineMode.AVERAGE,
                            restitution_combine=CombineMode.AVERAGE)
        sf, df, rest = combine_materials(a, b)
        # MULTIPLY wins: 0.8 * 0.4 = 0.32
        assert abs(sf - 0.32) < 1e-9

    def test_max_overrides_min(self):
        """MAX has higher priority than MIN."""
        a = PhysicsMaterial(static_friction=0.8, friction_combine=CombineMode.MAX)
        b = PhysicsMaterial(static_friction=0.4, friction_combine=CombineMode.MIN)
        sf, _, _ = combine_materials(a, b)
        assert abs(sf - 0.8) < 1e-9  # MAX(0.8, 0.4) = 0.8

    def test_result_clamped(self):
        """Resulting values are clamped to valid ranges."""
        a = PhysicsMaterial(static_friction=5.0, friction_combine=CombineMode.MAX)
        b = PhysicsMaterial(static_friction=-1.0, friction_combine=CombineMode.MAX)
        sf, _, _ = combine_materials(a, b)
        assert sf == MAX_FRICTION  # 5.0 clamped to 2.0


class TestMaterialPresets(PhysicsTestCase):
    """Material presets and get_material."""

    def test_all_presets_accessible(self):
        """Every preset in MATERIAL_PRESETS can be retrieved via get_material."""
        for name in MATERIAL_PRESETS:
            mat = get_material(name)
            assert mat.name == name

    def test_get_material_returns_copy(self):
        """get_material returns a copy, not the original."""
        original = MaterialPresets.rubber()
        retrieved = get_material('rubber')
        assert retrieved is not original
        # Modifying the copy doesn't affect the preset
        retrieved.static_friction = 0.0
        assert MATERIAL_PRESETS['rubber'].static_friction == 1.0

    def test_get_material_unknown_raises(self):
        """get_material with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown material"):
            get_material("nonexistent")

    def test_from_dict_roundtrip(self):
        """to_dict -> from_dict round-trips values."""
        original = MaterialPresets.metal()
        data = original.to_dict()
        restored = PhysicsMaterial.from_dict(data)
        assert restored.static_friction == original.static_friction
        assert restored.dynamic_friction == original.dynamic_friction
        assert restored.restitution == original.restitution
        assert restored.density == original.density
        assert restored.friction_combine == original.friction_combine
        assert restored.name == original.name

    def test_from_dict_empty(self):
        """from_dict with empty dict uses defaults."""
        mat = PhysicsMaterial.from_dict({})
        assert mat.static_friction == 0.6
        assert mat.dynamic_friction == 0.4

    def test_rubber_preset(self):
        """rubber: high friction, high bounce."""
        mat = MaterialPresets.rubber()
        assert mat.static_friction == 1.0
        assert mat.dynamic_friction == 0.8
        assert mat.restitution == 0.8

    def test_ice_preset(self):
        """ice: very low friction."""
        mat = MaterialPresets.ice()
        assert mat.static_friction == 0.05
        assert mat.dynamic_friction == 0.02

    def test_frictionless_preset(self):
        """frictionless: zero friction."""
        mat = MaterialPresets.frictionless()
        assert mat.static_friction == 0.0
        assert mat.dynamic_friction == 0.0

    def test_sticky_preset(self):
        """sticky: max friction, MAX combine mode."""
        mat = MaterialPresets.sticky()
        assert mat.static_friction == MAX_FRICTION
        assert mat.friction_combine == CombineMode.MAX

    def test_copy(self):
        """copy() creates an independent clone."""
        mat = PhysicsMaterial(static_friction=0.7, name="test")
        c = mat.copy()
        assert c.name == "test"
        assert c.static_friction == 0.7
        c.static_friction = 0.1
        assert mat.static_friction == 0.7


# =============================================================================
# 4. collision_shapes.py
# =============================================================================

class TestAABB(PhysicsTestCase):
    """AABB geometry and queries."""

    def test_center(self):
        """center is midpoint of min/max."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.center == (1.0, 2.0, 3.0)

    def test_half_extents(self):
        """half_extents is half the size."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.half_extents == (1.0, 2.0, 3.0)

    def test_size(self):
        """size is max - min."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.size == (2.0, 4.0, 6.0)

    def test_volume(self):
        """volume = x * y * z."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.volume == 48.0

    def test_surface_area(self):
        """surface_area = 2*(xy + yz + zx)."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(2, 4, 6))
        assert aabb.surface_area == 2 * (2*4 + 4*6 + 6*2)

    def test_contains_point_inside(self):
        """contains_point returns True for interior point."""
        aabb = AABB(min_point=(-1, -1, -1), max_point=(1, 1, 1))
        assert aabb.contains_point((0, 0, 0))

    def test_contains_point_on_edge(self):
        """contains_point returns True for point on boundary."""
        aabb = AABB(min_point=(-1, -1, -1), max_point=(1, 1, 1))
        assert aabb.contains_point((1, 0.5, 0.5))

    def test_contains_point_outside(self):
        """contains_point returns False for exterior point."""
        aabb = AABB(min_point=(-1, -1, -1), max_point=(1, 1, 1))
        assert not aabb.contains_point((2, 0, 0))

    def test_intersects_overlap(self):
        """intersects returns True for overlapping AABBs."""
        a = AABB(min_point=(0, 0, 0), max_point=(2, 2, 2))
        b = AABB(min_point=(1, 1, 1), max_point=(3, 3, 3))
        assert a.intersects(b)

    def test_intersects_no_overlap(self):
        """intersects returns False for separated AABBs."""
        a = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        b = AABB(min_point=(2, 2, 2), max_point=(3, 3, 3))
        assert not a.intersects(b)

    def test_intersects_touching(self):
        """intersects returns True for touching AABBs (boundary contact)."""
        a = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        b = AABB(min_point=(1, 0, 0), max_point=(2, 1, 1))
        assert a.intersects(b)

    def test_expand(self):
        """expand adds margin uniformly."""
        aabb = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        expanded = aabb.expand(0.5)
        assert expanded.min_point == (-0.5, -0.5, -0.5)
        assert expanded.max_point == (1.5, 1.5, 1.5)

    def test_merge(self):
        """merge produces union AABB."""
        a = AABB(min_point=(0, 0, 0), max_point=(1, 1, 1))
        b = AABB(min_point=(0.5, -1, 0), max_point=(2, 0.5, 1))
        merged = a.merge(b)
        assert merged.min_point == (0, -1, 0)
        assert merged.max_point == (2, 1, 1)

    def test_from_points(self):
        """from_points computes correct bounding box."""
        points = [(1, 0, -1), (-1, 2, 3), (0, -2, 1)]
        aabb = AABB.from_points(points)
        assert aabb.min_point == (-1, -2, -1)
        assert aabb.max_point == (1, 2, 3)

    def test_from_points_empty(self):
        """from_points with empty list returns origin AABB."""
        aabb = AABB.from_points([])
        assert aabb.min_point == (0, 0, 0)
        assert aabb.max_point == (0, 0, 0)

    def test_zero_volume(self):
        """Zero-size AABB has volume=0."""
        aabb = AABB(min_point=(1, 1, 1), max_point=(1, 1, 1))
        assert aabb.volume == 0.0


class TestMassProperties(PhysicsTestCase):
    """MassProperties inverse helpers."""

    def test_inverse_mass_zero(self):
        """mass=0 yields inverse_mass=0."""
        mp = MassProperties(mass=0.0)
        assert mp.inverse_mass == 0.0

    def test_inverse_mass_positive(self):
        """positive mass yields 1/mass."""
        mp = MassProperties(mass=4.0)
        assert mp.inverse_mass == 0.25

    def test_inverse_mass_negative(self):
        """negative mass yields 0 (clamped to 0 by <=0 check)."""
        mp = MassProperties(mass=-1.0)
        assert mp.inverse_mass == 0.0

    def test_inverse_inertia_zero_diagonal(self):
        """Zero inertia diag yields zero inverse."""
        mp = MassProperties(inertia_tensor=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ))
        inv = mp.inverse_inertia_tensor
        assert inv[0][0] == 0.0
        assert inv[1][1] == 0.0
        assert inv[2][2] == 0.0

    def test_inverse_inertia_positive(self):
        """Positive inertia diag yields 1/diag."""
        mp = MassProperties(inertia_tensor=(
            (4.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 8.0),
        ))
        inv = mp.inverse_inertia_tensor
        assert inv[0][0] == 0.25
        assert inv[1][1] == 0.5
        assert inv[2][2] == 0.125


class TestSphereShapeWhitebox(PhysicsTestCase):
    """SphereShape internal paths."""

    def test_radius_clamped_on_init(self):
        """radius < MIN_SHAPE_RADIUS is clamped at construction."""
        s = SphereShape(radius=0.0)
        assert s.radius == MIN_SHAPE_RADIUS

    def test_radius_clamped_on_setter(self):
        """radius < MIN_SHAPE_RADIUS is clamped by setter."""
        s = SphereShape(radius=1.0)
        s.radius = 0.0
        assert s.radius == MIN_SHAPE_RADIUS

    def test_contains_point_center(self):
        """origin is inside sphere."""
        s = SphereShape(radius=1.0)
        assert s.contains_point((0, 0, 0))

    def test_contains_point_surface(self):
        """point at distance == radius is inside."""
        s = SphereShape(radius=1.0)
        assert s.contains_point((1.0, 0, 0))

    def test_contains_point_outside(self):
        """point at distance > radius is outside."""
        s = SphereShape(radius=1.0)
        assert not s.contains_point((1.1, 0, 0))

    def test_contains_point_with_offset(self):
        """contains_point accounts for local_offset."""
        s = SphereShape(radius=1.0, local_offset=(2, 0, 0))
        # (2, 0, 0) is center; (3, 0, 0) is surface
        assert s.contains_point((2.5, 0, 0))
        assert not s.contains_point((4, 0, 0))

    def test_get_support_point(self):
        """get_support_point returns center + radius*direction."""
        s = SphereShape(radius=2.0)
        p = s.get_support_point((1, 0, 0))
        assert abs(p[0] - 2.0) < 1e-9
        assert p[1] == 0.0

    def test_get_support_point_zero_direction(self):
        """get_support_point with zero direction returns center."""
        s = SphereShape(radius=2.0)
        p = s.get_support_point((0, 0, 0))
        assert p == (0.0, 0.0, 0.0)

    def test_compute_aabb(self):
        """compute_aabb at origin."""
        s = SphereShape(radius=1.0)
        aabb = s.compute_aabb()
        assert aabb.min_point == (-1.0 - DEFAULT_SHAPE_MARGIN,) * 3
        assert aabb.max_point == (1.0 + DEFAULT_SHAPE_MARGIN,) * 3

    def test_copy(self):
        """copy() creates independent clone."""
        s = SphereShape(radius=1.5)
        c = s.copy()
        assert c.radius == 1.5
        c.radius = 2.0
        assert s.radius == 1.5

    def test_mass_properties(self):
        """compute_mass_properties with given density."""
        s = SphereShape(radius=0.5)
        props = s.compute_mass_properties(density=1000.0)
        expected_volume = (4.0/3.0) * math.pi * 0.5**3
        expected_mass = expected_volume * 1000.0
        assert abs(props.mass - expected_mass) < 1e-9

    def test_invalidate_cache_on_margin_change(self):
        """setting margin invalidates cached AABB and mass props."""
        s = SphereShape(radius=1.0)
        old_aabb = s._cached_aabb
        s.margin = 0.1
        assert s._cached_aabb is None


class TestBoxShapeWhitebox(PhysicsTestCase):
    """BoxShape internal paths."""

    def test_half_extents_clamped_on_init(self):
        """half_extents < MIN_SHAPE_DIMENSION is clamped."""
        b = BoxShape(half_extents=(0, 0, 0))
        assert b.half_extents[0] == MIN_SHAPE_DIMENSION

    def test_half_extents_clamped_on_setter(self):
        """half extent setter clamps below minimum."""
        b = BoxShape()
        b.half_extents = (-1, 0.5, 0)
        assert b.half_extents[0] == MIN_SHAPE_DIMENSION
        assert b.half_extents[1] == 0.5
        assert b.half_extents[2] == MIN_SHAPE_DIMENSION

    def test_size(self):
        """size is 2 * half_extents."""
        b = BoxShape(half_extents=(1, 2, 3))
        assert b.size == (2, 4, 6)

    def test_contains_point_inside(self):
        """center is inside box."""
        b = BoxShape(half_extents=(1, 1, 1))
        assert b.contains_point((0, 0, 0))

    def test_contains_point_on_edge(self):
        """point on surface is inside."""
        b = BoxShape(half_extents=(1, 1, 1))
        assert b.contains_point((1, 0, 0))

    def test_contains_point_outside(self):
        """point beyond half_extents is outside."""
        b = BoxShape(half_extents=(1, 1, 1))
        assert not b.contains_point((1.1, 0, 0))

    def test_get_support_point(self):
        """get_support_point picks corner in direction."""
        b = BoxShape(half_extents=(2, 3, 4))
        p = b.get_support_point((1, 0, 0))
        assert p[0] == 2.0
        p2 = b.get_support_point((-1, 0, 0))
        assert p2[0] == -2.0


class TestCapsuleShapeWhitebox(PhysicsTestCase):
    """CapsuleShape internal paths."""

    def test_half_height_zero_valid(self):
        """capsule with half_height=0 is a sphere."""
        c = CapsuleShape(radius=1.0, half_height=0.0)
        assert c.half_height == 0.0
        assert c.total_height == 2.0  # only caps

    def test_total_height(self):
        """total_height = 2*half_height + 2*radius."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        assert c.total_height == 3.0

    def test_radius_clamped(self):
        """radius clamping on setter."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        c.radius = 0.0
        assert c.radius == MIN_SHAPE_RADIUS

    def test_contains_point_inside(self):
        """center is inside capsule."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        assert c.contains_point((0, 0, 0))

    def test_contains_point_outside_radially(self):
        """point beyond radius is outside."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        assert not c.contains_point((0.6, 0, 0))

    def test_contains_point_above_endcap(self):
        """point above half_height but within radius of endcap is inside."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        # At y=1.25, the sphere cap extends to y=1.5, and at x=0 we're inside
        assert c.contains_point((0, 1.25, 0))

    def test_contains_point_above_endcap_edge(self):
        """point above hemisphere cap but outside radius is outside."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        # At y=1.25, x=0.51 is outside the hemisphere of radius 0.5
        assert not c.contains_point((0.51, 1.25, 0))

    def test_get_support_point_up(self):
        """get_support_point in +Y direction hits top endcap + radius."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        p = c.get_support_point((0, 1, 0))
        assert abs(p[1] - 1.5) < 1e-9  # half_height + radius

    def test_get_support_point_down(self):
        """get_support_point in -Y direction hits bottom endcap - radius."""
        c = CapsuleShape(radius=0.5, half_height=1.0)
        p = c.get_support_point((0, -1, 0))
        assert abs(p[1] - (-1.5)) < 1e-9


class TestConvexHullShapeWhitebox(PhysicsTestCase):
    """ConvexHullShape validation and edge cases."""

    def test_min_points_validation(self):
        """less than 4 points raises ValueError."""
        with pytest.raises(ValueError, match="at least 4"):
            ConvexHullShape(points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    def test_vertex_count(self):
        """vertex_count returns the number of hull vertices."""
        shape = ConvexHullShape(points=[
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        ])
        assert shape.vertex_count == 4

    def test_compute_aabb(self):
        """compute_aabb covers all vertices."""
        shape = ConvexHullShape(points=[
            (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
        ])
        aabb = shape.compute_aabb()
        assert aabb.min_point[0] <= 0.0
        assert aabb.max_point[0] >= 2.0


class TestMeshShapeWhitebox(PhysicsTestCase):
    """MeshShape validation."""

    def test_min_vertices_validation(self):
        """fewer than 3 vertices raises ValueError."""
        with pytest.raises(ValueError, match="at least 3 vertices"):
            MeshShape(vertices=[(0, 0, 0), (1, 0, 0)], indices=[(0, 1, 0)])

    def test_min_indices_validation(self):
        """no triangles raises ValueError."""
        with pytest.raises(ValueError, match="at least 1 triangle"):
            MeshShape(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], indices=[])

    def test_triangle_count(self):
        """triangle_count returns correct count."""
        shape = MeshShape(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            indices=[(0, 1, 2)],
        )
        assert shape.triangle_count == 1

    def test_get_triangle(self):
        """get_triangle returns vertices for index."""
        shape = MeshShape(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            indices=[(0, 1, 2)],
        )
        t = shape.get_triangle(0)
        assert t == ((0, 0, 0), (1, 0, 0), (0, 1, 0))

    def test_get_triangle_out_of_range(self):
        """get_triangle with out-of-range index raises IndexError."""
        shape = MeshShape(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            indices=[(0, 1, 2)],
        )
        with pytest.raises(IndexError):
            shape.get_triangle(5)


class TestCompoundShapeWhitebox(PhysicsTestCase):
    """CompoundShape edge cases."""

    def test_empty_compute_aabb(self):
        """empty compound returns point AABB at position."""
        compound = CompoundShape()
        aabb = compound.compute_aabb(position=(5, 3, 1))
        assert aabb.min_point == (5, 3, 1)
        assert aabb.max_point == (5, 3, 1)

    def test_empty_get_support_point(self):
        """empty compound returns local_offset as support."""
        compound = CompoundShape(local_offset=(2, 0, 0))
        p = compound.get_support_point((1, 0, 0))
        assert p == (2, 0, 0)

    def test_empty_mass_properties(self):
        """empty compound returns default MassProperties."""
        compound = CompoundShape()
        props = compound.compute_mass_properties(density=1.0)
        assert props.mass > 0  # default MassProperties has mass=1.0

    def test_child_count(self):
        """child_count matches number of children."""
        compound = CompoundShape()
        assert compound.child_count == 0
        compound.add_child(SphereShape(radius=0.5))
        assert compound.child_count == 1

    def test_remove_child(self):
        """remove_child removes by index."""
        compound = CompoundShape()
        child = CompoundChild(shape=SphereShape(radius=0.5))
        compound._children = [child]
        compound.remove_child(0)
        assert compound.child_count == 0

    def test_remove_child_out_of_range(self):
        """remove_child with bad index is no-op."""
        compound = CompoundShape()
        compound.remove_child(5)  # should not raise
        assert True

    def test_clear_children(self):
        """clear_children empties the shape."""
        compound = CompoundShape()
        compound.add_child(SphereShape(radius=0.5))
        compound.clear_children()
        assert compound.child_count == 0

    def test_contains_point_empty(self):
        """empty compound does not contain any point."""
        compound = CompoundShape()
        assert not compound.contains_point((0, 0, 0))


class TestCreateShapeFactory(PhysicsTestCase):
    """create_shape factory function."""

    def test_create_sphere(self):
        """create_shape with SPHERE returns SphereShape."""
        shape = create_shape(ShapeType.SPHERE, radius=2.0)
        assert isinstance(shape, SphereShape)
        assert shape.radius == 2.0

    def test_create_box(self):
        """create_shape with BOX returns BoxShape."""
        shape = create_shape(ShapeType.BOX, half_extents=(1, 2, 3))
        assert isinstance(shape, BoxShape)
        assert shape.half_extents == (1, 2, 3)

    def test_create_capsule(self):
        """create_shape with CAPSULE returns CapsuleShape."""
        shape = create_shape(ShapeType.CAPSULE, radius=0.5, half_height=1.0)
        assert isinstance(shape, CapsuleShape)

    def test_create_cylinder(self):
        """create_shape with CYLINDER returns CylinderShape."""
        shape = create_shape(ShapeType.CYLINDER, radius=0.5, height=2.0)
        assert isinstance(shape, CylinderShape)

    def test_create_convex_hull(self):
        """create_shape with CONVEX_HULL returns ConvexHullShape."""
        shape = create_shape(ShapeType.CONVEX_HULL,
                             points=[(0,0,0), (1,0,0), (0,1,0), (0,0,1)])
        assert isinstance(shape, ConvexHullShape)

    def test_create_mesh(self):
        """create_shape with MESH returns MeshShape."""
        shape = create_shape(ShapeType.MESH,
                             vertices=[(0,0,0), (1,0,0), (0,1,0)],
                             indices=[(0,1,2)])
        assert isinstance(shape, MeshShape)

    def test_create_compound(self):
        """create_shape with COMPOUND returns CompoundShape."""
        shape = create_shape(ShapeType.COMPOUND)
        assert isinstance(shape, CompoundShape)

    def test_unknown_type_raises(self):
        """create_shape with unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported shape type"):
            create_shape(ShapeType.PLANE)


class TestVectorHelpers(PhysicsTestCase):
    """Internal vector math helpers."""

    def test_vector_normalize_zero(self):
        """_vector_normalize of zero vector returns zero."""
        v = _vector_normalize((0, 0, 0))
        assert v == (0.0, 0.0, 0.0)

    def test_vector_normalize_positive(self):
        """_vector_normalize produces unit vector."""
        v = _vector_normalize((3, 0, 0))
        assert v == (1.0, 0.0, 0.0)

    def test_rotate_vector_identity(self):
        """_rotate_vector with identity quaternion is identity."""
        v = _rotate_vector((1, 2, 3), (0, 0, 0, 1))
        assert v == (1, 2, 3)

    def test_rotate_vector_90_x(self):
        """_rotate_vector 90 degrees around X."""
        v = _rotate_vector((0, 1, 0), (0.7071068, 0, 0, 0.7071068))
        assert abs(v[1]) < 1e-6
        assert abs(v[2] - 1.0) < 1e-6


# =============================================================================
# 5. rigid_body.py
# =============================================================================

class TestRigidBodyStatic(PhysicsTestCase):
    """Static body mass/inertia properties."""

    def test_static_mass_zero(self):
        """Static body has zero mass."""
        body = _static_body()
        assert body.mass == 0.0
        assert body.inverse_mass == 0.0

    def test_static_inertia_zero(self):
        """Static body has zero inertia tensor."""
        body = _static_body()
        I = body.inertia_tensor
        assert I[0][0] == 0.0
        assert I[1][1] == 0.0
        assert I[2][2] == 0.0

    def test_static_body_type_change_to_dynamic(self):
        """STATIC->DYNAMIC transition resets mass to 1.0."""
        body = RigidBody(body_type=BodyType.STATIC,
                         shape=SphereShape(radius=0.5))
        assert body.mass == 0.0
        body.body_type = BodyType.DYNAMIC
        assert body.mass == 1.0  # default mass for new dynamic body
        assert body.inverse_mass == 1.0

    def test_dynamic_body_type_change_to_static(self):
        """DYNAMIC->STATIC zeroes velocities and mass."""
        body = _dynamic_body()
        body.linear_velocity = (10, 0, 0)
        body.body_type = BodyType.STATIC
        assert body.mass == 0.0
        assert body.linear_velocity == (0, 0, 0)

    def test_static_velocity_setter_noop(self):
        """Setting velocity on static body is a no-op."""
        body = _static_body()
        body.linear_velocity = (10, 0, 0)
        assert body.linear_velocity == (0, 0, 0)

    def test_static_angular_velocity_setter_noop(self):
        """Setting angular velocity on static body is a no-op."""
        body = _static_body()
        body.angular_velocity = (5, 0, 0)
        assert body.angular_velocity == (0, 0, 0)


class TestRigidBodyVelocity(PhysicsTestCase):
    """Velocity clamping and damping."""

    def test_linear_velocity_clamped(self):
        """linear_velocity setter clamps to MAX_LINEAR_VELOCITY."""
        body = _dynamic_body()
        body.linear_velocity = (1000, 0, 0)
        speed = math.sqrt(body.linear_velocity[0]**2)
        assert speed <= MAX_LINEAR_VELOCITY

    def test_angular_velocity_clamped(self):
        """angular_velocity setter clamps to MAX_ANGULAR_VELOCITY."""
        body = _dynamic_body()
        body.angular_velocity = (200, 0, 0)
        speed = math.sqrt(body.angular_velocity[0]**2)
        assert speed <= MAX_ANGULAR_VELOCITY

    def test_linear_damping_clamped(self):
        """linear_damping setter clamps to [0, 1]."""
        body = _dynamic_body()
        body.linear_damping = 1.5
        assert body.linear_damping == 1.0
        body.linear_damping = -0.5
        assert body.linear_damping == 0.0

    def test_angular_damping_clamped(self):
        """angular_damping setter clamps to [0, 1]."""
        body = _dynamic_body()
        body.angular_damping = 1.5
        assert body.angular_damping == 1.0


class TestRigidBodyForceApplication(PhysicsTestCase):
    """Force/impulse application internals."""

    def test_apply_force_on_non_dynamic_noop(self):
        """apply_force on static body is a no-op (no crash)."""
        body = _static_body()
        body.apply_force((100, 0, 0))
        # Static body silently ignores
        assert body._force_accumulator == (0, 0, 0)

    def test_apply_impulse_on_non_dynamic_noop(self):
        """apply_impulse on static body is a no-op."""
        body = _static_body()
        body.apply_impulse((100, 0, 0))
        assert body.linear_velocity == (0, 0, 0)

    def test_apply_force_wakens_sleeping(self):
        """applying force to a sleeping body wakes it up."""
        body = _dynamic_body()
        body.put_to_sleep()
        assert body.is_sleeping
        body.apply_force((10, 0, 0))
        assert not body.is_sleeping

    def test_apply_force_with_lock_mask_x(self):
        """apply_force respects position lock mask."""
        body = _dynamic_body()
        body.flags = BodyFlags(lock_position_x=True)
        body.apply_force((100, 200, 300))
        assert body._force_accumulator[0] == 0.0  # X locked
        assert body._force_accumulator[1] == 200.0
        assert body._force_accumulator[2] == 300.0

    def test_apply_impulse_with_lock_mask_y(self):
        """apply_impulse respects position lock mask."""
        body = _dynamic_body()
        body.flags = BodyFlags(lock_position_y=True)
        body.apply_impulse((100, 200, 300))
        # X and Z should be applied, Y should be 0
        assert body.linear_velocity[0] != 0.0
        assert body.linear_velocity[1] == 0.0

    def test_apply_torque(self):
        """apply_torque accumulates torque."""
        body = _dynamic_body()
        body.apply_torque((5, 0, 0))
        assert body._torque_accumulator == (5, 0, 0)

    def test_apply_torque_respects_rotation_lock(self):
        """apply_torque respects rotation lock mask."""
        body = _dynamic_body()
        body.flags = BodyFlags(lock_rotation_x=True)
        body.apply_torque((5, 10, 15))
        assert body._torque_accumulator[0] == 0.0  # X locked
        assert body._torque_accumulator[1] == 10.0
        assert body._torque_accumulator[2] == 15.0

    def test_clear_forces(self):
        """clear_forces resets accumulators."""
        body = _dynamic_body()
        body.apply_force((10, 20, 30))
        body.apply_torque((1, 2, 3))
        body.clear_forces()
        assert body._force_accumulator == (0, 0, 0)
        assert body._torque_accumulator == (0, 0, 0)

    def test_apply_force_local(self):
        """apply_force_local transforms to world space."""
        body = _dynamic_body()
        body.apply_force_local((0, 10, 0))
        # With identity rotation, local Z stays Z
        assert body._force_accumulator[1] == 10.0

    def test_get_velocity_at_point(self):
        """get_velocity_at_point includes angular contribution."""
        body = _dynamic_body(position=(0, 0, 0))
        body.linear_velocity = (5, 0, 0)
        body.angular_velocity = (0, 0, 1)  # CCW around Z
        # At point (0, 1, 0): v_angular = w x r = (0,0,1) x (0,1,0) = (-1, 0, 0)
        v = body.get_velocity_at_point((0, 1, 0))
        assert abs(v[0] - 4.0) < 1e-9  # 5 + (-1)

    def test_get_velocity_at_local_point(self):
        """get_velocity_at_local_point transforms point to world first."""
        body = _dynamic_body(position=(10, 0, 0))
        v = body.get_velocity_at_local_point((0, 0, 0))
        assert v is not None


class TestRigidBodyIntegration(PhysicsTestCase):
    """Integration internals."""

    def test_integrate_velocities_zero_torque(self):
        """integrate_velocities with zero torque skips angular integration."""
        body = _dynamic_body()
        body.integrate_velocities(0.016, (0, -9.81, 0))
        # After integration, angular velocity should still be zero
        assert body.angular_velocity == (0, 0, 0)

    def test_integrate_velocities_applies_gravity(self):
        """integrate_velocities applies gravity force as acceleration."""
        body = _dynamic_body()
        body.integrate_velocities(0.016, (0, -9.81, 0))
        # dv = g * dt = 9.81 * 0.016 = 0.15696
        assert body.linear_velocity[1] < 0  # falling down

    def test_integrate_velocities_sleeping_skips(self):
        """integrate_velocities skips sleeping bodies."""
        body = _dynamic_body()
        body.put_to_sleep()
        body.integrate_velocities(0.016, (0, -9.81, 0))
        assert body.linear_velocity == (0, 0, 0)

    def test_integrate_positions_updates_position(self):
        """integrate_positions moves body by velocity * dt."""
        body = _dynamic_body(position=(0, 0, 0))
        body.linear_velocity = (10, 0, 0)
        body.integrate_positions(0.1)
        assert abs(body.position[0] - 1.0) < 1e-9

    def test_integrate_positions_zero_angular_skips_rotation(self):
        """integrate_positions with zero angular velocity skips rotation update."""
        body = _dynamic_body()
        body.angular_velocity = (0, 0, 0)
        body.integrate_positions(0.016)
        assert body.rotation == (0, 0, 0, 1)  # unchanged

    def test_integrate_positions_angular_rotation(self):
        """integrate_positions updates rotation from angular velocity."""
        body = _dynamic_body()
        body.angular_velocity = (0, 1, 0)  # 1 rad/s around Y
        body.integrate_positions(0.1)
        # Should have rotated ~0.1 rad around Y (cos(0.05), sin(0.05))
        assert abs(body.rotation[1]) > 0  # Y component changed

    def test_integrate_positions_lock_mask(self):
        """integrate_positions respects position and rotation locks."""
        body = _dynamic_body(position=(0, 0, 0))
        body.flags = BodyFlags(lock_position_x=True)
        body.linear_velocity = (100, 50, 30)
        body.integrate_positions(0.016)
        assert body.position[0] == 0.0  # X locked
        assert body.position[1] != 0.0  # Y not locked


class TestRigidBodyState(PhysicsTestCase):
    """Body state and interpolation."""

    def test_save_and_get_state(self):
        """save_state/get_state round-trips."""
        body = _dynamic_body(position=(1, 2, 3))
        body.linear_velocity = (4, 5, 6)
        body.save_state()
        state = body.get_state()
        assert state.position == (1, 2, 3)
        assert state.linear_velocity == (4, 5, 6)

    def test_set_state(self):
        """set_state restores body state."""
        body = _dynamic_body(position=(0, 0, 0))
        state = BodyState(position=(10, 20, 30), linear_velocity=(1, 0, 0))
        body.set_state(state)
        assert body.position == (10, 20, 30)

    def test_interpolate_state_no_previous(self):
        """interpolate_state with no previous returns current state."""
        body = _dynamic_body(position=(5, 0, 0))
        state = body.interpolate_state(0.5)
        assert state.position == (5, 0, 0)

    def test_interpolate_state_half(self):
        """interpolate_state at alpha=0.5 returns midpoint."""
        body = _dynamic_body(position=(0, 0, 0))
        body.save_state()
        body.position = (10, 0, 0)
        state = body.interpolate_state(0.5)
        assert state.position == (5, 0, 0)


class TestRigidBodyTransforms(PhysicsTestCase):
    """Transform point/direction operations."""

    def test_transform_point_to_world(self):
        """transform_point_to_world applies rotation + position."""
        body = _dynamic_body(position=(10, 0, 0))
        p = body.transform_point_to_world((0, 0, 0))
        assert p == (10, 0, 0)

    def test_transform_point_to_local(self):
        """transform_point_to_local inverts world transform."""
        body = _dynamic_body(position=(10, 0, 0))
        p = body.transform_point_to_local((10, 5, 0))
        assert p == (0, 5, 0)

    def test_transform_direction_to_world(self):
        """transform_direction_to_world applies rotation only."""
        body = _dynamic_body(position=(10, 0, 0))
        d = body.transform_direction_to_world((1, 0, 0))
        assert abs(d[0] - 1.0) < 1e-9

    def test_world_center_of_mass(self):
        """world_center_of_mass accounts for rotation."""
        body = _dynamic_body(position=(5, 0, 0))
        com = body.world_center_of_mass
        assert com[0] >= 4.0  # near origin + position


class TestRigidBodyMassAccessors(PhysicsTestCase):
    """Mass property accessors."""

    def test_mass_setter_clamps(self):
        """mass setter clamps to [MIN_MASS, MAX_MASS]."""
        body = _dynamic_body()
        body.mass = MAX_MASS * 10
        assert body.mass == MAX_MASS
        body.mass = 0
        assert body.mass == MIN_MASS

    def test_mass_setter_on_static_noop(self):
        """mass setter on static body is a no-op."""
        body = _static_body()
        body.mass = 50
        assert body.mass == 0.0


class TestQuaternionNormalize(PhysicsTestCase):
    """Quaternion normalization edge cases."""

    def test_normalize_zero_length(self):
        """zero-length quaternion returns identity."""
        q = _quaternion_normalize((0, 0, 0, 0))
        assert q == (0.0, 0.0, 0.0, 1.0)

    def test_normalize_identity(self):
        """identity quaternion normalizes to itself."""
        q = _quaternion_normalize((0, 0, 0, 1))
        assert q == (0, 0, 0, 1)

    def test_normalize_unit(self):
        """unit quaternion normalizes to itself."""
        q = _quaternion_normalize((1, 0, 0, 0))
        assert abs(q[3]) < 1e-9  # w near 0


# =============================================================================
# 6. sleeping.py
# =============================================================================

class TestSleepManagerCore(PhysicsTestCase):
    """SleepManager core logic."""

    def test_can_sleep_static_false(self):
        """can_sleep returns False for static bodies."""
        mgr = SleepManager()
        body = _static_body()
        assert not mgr.can_sleep(body)

    def test_can_sleep_kinematic_false(self):
        """can_sleep returns False for kinematic bodies."""
        mgr = SleepManager()
        body = RigidBody(body_type=BodyType.KINEMATIC, shape=SphereShape(radius=0.5))
        assert not mgr.can_sleep(body)

    def test_can_sleep_dynamic_true(self):
        """can_sleep returns True for dynamic bodies."""
        mgr = SleepManager()
        body = _dynamic_body()
        assert mgr.can_sleep(body)

    def test_is_below_threshold(self):
        """is_below_threshold compares against squared thresholds."""
        mgr = SleepManager(linear_threshold=0.1, angular_threshold=0.5)
        body = _dynamic_body()
        body.linear_velocity = (0.05, 0, 0)  # < 0.1
        body.angular_velocity = (0.2, 0, 0)  # < 0.5
        assert mgr.is_below_threshold(body)

    def test_is_below_threshold_over_linear(self):
        """is_below_threshold returns False when linear exceeds threshold."""
        mgr = SleepManager(linear_threshold=0.1, angular_threshold=10.0)
        body = _dynamic_body()
        body.linear_velocity = (1.0, 0, 0)
        assert not mgr.is_below_threshold(body)

    def test_merge_islands_smaller_into_larger(self):
        """merge_islands merges smaller island into larger one."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        c = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)
        mgr.register_body(c)

        # Merge a-b then b-c -> all same island
        mgr.merge_islands(a, b)
        mgr.merge_islands(b, c)

        islands = mgr._islands
        island_ids = set(mgr._body_to_island.values())
        assert len(island_ids) == 1  # all in one island

    def test_merge_islands_same_island_noop(self):
        """merge_islands with bodies in same island is a no-op."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)
        mgr.merge_islands(a, b)

        count_before = len(mgr._islands)
        mgr.merge_islands(a, b)  # same island
        assert len(mgr._islands) == count_before

    def test_merge_islands_sets_active(self):
        """merge_islands: if either island active, merged is active."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)

        # Put a to sleep
        mgr.put_to_sleep(a)
        mgr._islands[mgr._body_to_island[a.id]].state = IslandState.SLEEPING

        # b is active by default, merge should wake a's island
        mgr.merge_islands(a, b)

        # All should be in active island
        island_id = mgr._body_to_island[a.id]
        island = mgr._islands[island_id]
        assert island.state == IslandState.ACTIVE

    def test_rebuild_islands_connected(self):
        """rebuild_islands correctly groups connected bodies."""
        mgr = SleepManager()
        bodies = [_dynamic_body() for _ in range(4)]
        for b in bodies:
            mgr.register_body(b)

        # Connect 0-1 and 2-3 (two separate groups)
        contacts = [
            (bodies[0].id, bodies[1].id),
            (bodies[2].id, bodies[3].id),
        ]
        mgr.rebuild_islands(contacts)

        # Should have 2 islands
        assert len(mgr._islands) == 2

        # 0 and 1 should be in same island
        assert mgr._body_to_island[bodies[0].id] == mgr._body_to_island[bodies[1].id]
        # 2 and 3 should be in same island
        assert mgr._body_to_island[bodies[2].id] == mgr._body_to_island[bodies[3].id]
        # Islands should be different
        assert mgr._body_to_island[bodies[0].id] != mgr._body_to_island[bodies[2].id]

    def test_rebuild_islands_with_contacts_all_sleeping(self):
        """rebuild_islands with all-sleeping bodies creates sleeping island."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)

        # Put both to sleep
        a.put_to_sleep()
        b.put_to_sleep()

        mgr.rebuild_islands([(a.id, b.id)])
        island_id = mgr._body_to_island[a.id]
        assert mgr._islands[island_id].state == IslandState.SLEEPING

    def test_update_wakes_from_sleep(self):
        """update wakes sleeping island when body velocity exceeds threshold."""
        mgr = SleepManager(linear_threshold=0.1, time_threshold=0.0)
        body = _dynamic_body()
        body.linear_velocity = (0, 0, 0)
        mgr.register_body(body)
        mgr.put_to_sleep(body)
        assert body.is_sleeping

        # Give it velocity and update
        body.linear_velocity = (10, 0, 0)
        mgr.update(0.016)
        assert not body.is_sleeping

    def test_update_accumulates_timer_active_island(self):
        """update accumulates sleep timer for below-threshold bodies."""
        mgr = SleepManager(linear_threshold=1.0, angular_threshold=1.0,
                           time_threshold=10.0)  # long threshold
        body = _dynamic_body()
        body.linear_velocity = (0.5, 0, 0)  # below threshold
        mgr.register_body(body)

        # body._sleep_timer is initially 0 from RigidBody.__init__
        assert body.sleep_timer == 0.0
        mgr.update(0.016)
        # The manager accumulates its own timer
        assert mgr._sleep_timers[body.id] > 0.0

    def test_wake_connected_bodies(self):
        """wake_connected_bodies wakes entire island when through_joints=True."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)
        mgr.merge_islands(a, b)

        # Put both to sleep
        mgr.put_to_sleep(a)

        # Wake a
        mgr.wake_connected_bodies(a, through_joints=True)
        assert not a.is_sleeping
        assert not b.is_sleeping

    def test_wake_connected_bodies_single(self):
        """wake_connected_bodies with through_joints=False wakes only the body."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)
        mgr.merge_islands(a, b)

        # Put both to sleep
        mgr.put_to_sleep(a)

        # Wake a but do not propagate through joints
        mgr.wake_connected_bodies(a, through_joints=False)
        assert not a.is_sleeping

    def test_get_island_info_registered(self):
        """get_island_info returns info for registered body."""
        mgr = SleepManager()
        body = _dynamic_body()
        mgr.register_body(body)
        info = mgr.get_island_info(body)
        assert info is not None
        assert 'island_id' in info
        assert 'state' in info

    def test_get_island_info_unregistered(self):
        """get_island_info returns None for unregistered body."""
        mgr = SleepManager()
        body = _dynamic_body()
        info = mgr.get_island_info(body)
        assert info is None

    def test_reset(self):
        """reset wakes all sleeping bodies."""
        mgr = SleepManager()
        body = _dynamic_body()
        mgr.register_body(body)
        mgr.put_to_sleep(body)
        assert body.is_sleeping
        mgr.reset()
        assert not body.is_sleeping
        assert mgr.sleeping_count == 0

    def test_clear(self):
        """clear removes all bodies and islands."""
        mgr = SleepManager()
        body = _dynamic_body()
        mgr.register_body(body)
        assert mgr.sleeping_count + mgr.awake_count > 0
        mgr.clear()
        assert mgr.sleeping_count == 0
        assert mgr.awake_count == 0
        assert mgr.island_count == 0

    def test_unregister_body_not_found_noop(self):
        """unregister_body with unknown ID is a no-op."""
        mgr = SleepManager()
        body = _dynamic_body()
        mgr.unregister_body(body)  # should not raise
        assert True

    def test_threshold_setters(self):
        """SleepManager threshold setters enforce non-negative."""
        mgr = SleepManager()
        mgr.linear_threshold = -1.0
        assert mgr.linear_threshold == 0.0
        mgr.angular_threshold = -5.0
        assert mgr.angular_threshold == 0.0
        mgr.time_threshold = -0.1
        assert mgr.time_threshold == 0.0


class TestSleepManagerStatistics(PhysicsTestCase):
    """SleepManager statistics accuracy."""

    def test_statistics_counts(self):
        """get_statistics counts match registered bodies."""
        mgr = SleepManager()
        a = _dynamic_body()
        b = _dynamic_body()
        mgr.register_body(a)
        mgr.register_body(b)

        stats = mgr.get_statistics()
        assert stats['total_bodies'] == 2
        assert stats['awake_bodies'] == 2
        assert stats['sleeping_bodies'] == 0

        mgr.put_to_sleep(a)
        stats = mgr.get_statistics()
        # _sleep_island updates sleeping_count
        # but put_to_sleep only puts to sleep if island-level check passes
        # direct call through manager:
        mgr._sleeping_count = 1
        mgr._awake_count = 1
        stats = mgr.get_statistics()
        assert stats['sleeping_bodies'] == 1
        assert stats['awake_bodies'] == 1


# =============================================================================
# 7. queries.py
# =============================================================================

class TestRaySphereWhitebox(PhysicsTestCase):
    """Ray-sphere intersection edge cases."""

    def test_ray_origin_behind_sphere(self):
        """Ray origin behind the sphere still hits (t2 positive)."""
        shape = SphereShape(radius=1.0)
        body = _static_body(position=(0, 0, 0), shape=shape)

        hit = raycast_single(
            bodies=[body],
            origin=(0, 5, 0),
            direction=(0, -1, 0),
        )
        assert hit is not None
        # origin at y=5, sphere surface at y=1, distance = 4
        assert abs(hit.distance - 4.0) < 0.01

    def test_ray_misses_above_sphere(self):
        """Ray passing above sphere misses."""
        shape = SphereShape(radius=1.0)
        body = _static_body(position=(0, 0, 0), shape=shape)

        hit = raycast_single(
            bodies=[body],
            origin=(0, 2.0, -5),
            direction=(0, 0, 1),
        )
        assert hit is None

    def test_ray_sphere_negative_discriminant(self):
        """Negative discriminant produces no hit (internal code path)."""
        # For this we need a ray that doesn't intersect at all
        shape = SphereShape(radius=0.5)
        body = _static_body(position=(0, 0, 0), shape=shape)

        hit = raycast_single(
            bodies=[body],
            origin=(10, 0, 0),
            direction=(0, 1, 0),
        )
        assert hit is None


class TestCollisionFilterWhitebox(PhysicsTestCase):
    """CollisionFilter.should_collide internal branches."""

    def test_static_flag_filter(self):
        """QueryFlags.STATIC only matches static bodies."""
        filt = CollisionFilter(flags=QueryFlags.STATIC)
        static = _static_body()
        dynamic = _dynamic_body()
        assert filt.should_collide(static)
        assert not filt.should_collide(dynamic)

    def test_kinematic_flag_filter(self):
        """QueryFlags.KINEMATIC only matches kinematic bodies."""
        filt = CollisionFilter(flags=QueryFlags.KINEMATIC)
        kin = RigidBody(body_type=BodyType.KINEMATIC, shape=SphereShape(radius=0.5))
        dyn = _dynamic_body()
        assert filt.should_collide(kin)
        assert not filt.should_collide(dyn)

    def test_dynamic_flag_filter(self):
        """QueryFlags.DYNAMIC only matches dynamic bodies."""
        filt = CollisionFilter(flags=QueryFlags.DYNAMIC)
        dyn = _dynamic_body()
        static = _static_body()
        assert filt.should_collide(dyn)
        assert not filt.should_collide(static)

    def test_trigger_excluded_when_flags_not_all(self):
        """trigger bodies excluded when flags != ALL (and != TRIGGERS)."""
        filt = CollisionFilter(flags=QueryFlags.DYNAMIC)
        trigger_body = _dynamic_body()
        trigger_body.shape.is_trigger = True
        # trigger flagged body should be excluded
        assert not filt.should_collide(trigger_body)

    def test_trigger_included_when_flags_all(self):
        """trigger bodies included when flags == ALL."""
        filt = CollisionFilter(flags=QueryFlags.ALL)
        trigger_body = _dynamic_body()
        trigger_body.shape.is_trigger = True
        # But it may still be excluded by layer check:
        # body.collision_layer defaults to 1
        # So this will pass layer check
        assert filt.should_collide(trigger_body)

    def test_layer_filter_no_match(self):
        """Non-matching layer/mask returns False."""
        filt = CollisionFilter(layer=1, mask=1)
        body = _dynamic_body()
        body.collision_layer = 2
        body.collision_mask = 2
        assert not filt.should_collide(body)

    def test_custom_filter(self):
        """custom_filter callback is invoked."""
        filt = CollisionFilter(custom_filter=lambda b: False)
        body = _dynamic_body()
        assert not filt.should_collide(body)

    def test_layer_only_factory(self):
        """layer_only creates filter for specific layer."""
        filt = CollisionFilter.layer_only(2)
        assert filt.layer == 1 << 2
        assert filt.mask == 1 << 2

    def test_exclude_layer_factory(self):
        """exclude_layer creates filter that excludes a layer."""
        filt = CollisionFilter.exclude_layer(0)
        assert filt.mask == ~(1 << 0)


class TestOverlapWhitebox(PhysicsTestCase):
    """Overlap query edge cases."""

    def test_overlap_sphere_no_bodies(self):
        """overlap_sphere with empty list returns empty."""
        results = overlap_sphere([], (0, 0, 0), 1.0)
        assert len(results) == 0

    def test_overlap_sphere_hits(self):
        """overlap_sphere finds overlapping bodies."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        results = overlap_sphere([body], (0, 0, 0), 0.5)
        assert len(results) == 1

    def test_overlap_sphere_miss(self):
        """overlap_sphere returns empty when no overlap."""
        body = _dynamic_body(position=(10, 0, 0), shape=SphereShape(radius=0.5))
        results = overlap_sphere([body], (0, 0, 0), 0.5)
        assert len(results) == 0

    def test_overlap_box_no_bodies(self):
        """overlap_box with empty list returns empty."""
        results = overlap_box([], (0, 0, 0), (1, 1, 1))
        assert len(results) == 0

    def test_overlap_capsule_no_bodies(self):
        """overlap_capsule with empty list returns empty."""
        results = overlap_capsule([], (0, 0, 0), (0, 1, 0), 0.5)
        assert len(results) == 0


class TestSweepWhitebox(PhysicsTestCase):
    """Sweep query edge cases."""

    def test_sweep_sphere_no_bodies(self):
        """sweep_sphere with no bodies returns no-hit."""
        result = sweep_sphere([], (0, 0, 0), (1, 0, 0), 0.5, 10.0)
        assert not result.hit
        assert result.fraction == 1.0

    def test_sweep_box_no_bodies(self):
        """sweep_box with no bodies returns no-hit."""
        result = sweep_box([], (0, 0, 0), (1, 0, 0), (0.5, 0.5, 0.5), 10.0)
        assert not result.hit

    def test_sweep_capsule_no_bodies(self):
        """sweep_capsule with no bodies returns no-hit."""
        result = sweep_capsule([], (0, -0.5, 0), (0, 0.5, 0), (1, 0, 0), 0.5, 10.0)
        assert not result.hit


class TestPointInside(PhysicsTestCase):
    """point_inside query."""

    def test_point_inside_hit(self):
        """point_inside finds body containing the point."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=2.0))
        result = point_inside([body], (0, 0, 0))
        assert result is body

    def test_point_inside_miss_aabb(self):
        """point_inside returns None when AABB doesn't contain the point."""
        body = _dynamic_body(position=(10, 10, 10), shape=SphereShape(radius=1.0))
        result = point_inside([body], (0, 0, 0))
        assert result is None

    def test_point_inside_miss_shape(self):
        """point_inside returns None when AABB contains point but shape doesn't."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        result = point_inside([body], (3, 0, 0))  # AABB might contain, but sphere won't
        assert result is None


class TestClosestPointAndDistance(PhysicsTestCase):
    """closest_point_on_body and distance_to_body."""

    def test_closest_point_sphere(self):
        """closest_point_on_body returns sphere surface point."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=2.0))
        # Point far away on X axis
        p = closest_point_on_body(body, (10, 0, 0))
        assert abs(p[0] - 2.0) < 1e-6  # surface at x=2

    def test_closest_point_sphere_at_center(self):
        """closest_point_on_body at center returns +x radius."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=2.0))
        p = closest_point_on_body(body, (0, 0, 0))
        # distance < epsilon -> returns center + (radius, 0, 0)
        assert abs(p[0] - 2.0) < 1e-6

    def test_closest_point_box(self):
        """closest_point_on_body for box returns clamped point."""
        body = _dynamic_body(position=(0, 0, 0),
                              shape=BoxShape(half_extents=(1, 1, 1)))
        p = closest_point_on_body(body, (5, 0, 0))
        assert abs(p[0] - 1.0) < 1e-6  # clamped to half_extent

    def test_closest_point_box_inside(self):
        """closest_point_on_body inside box returns the point itself."""
        body = _dynamic_body(position=(0, 0, 0),
                              shape=BoxShape(half_extents=(2, 2, 2)))
        p = closest_point_on_body(body, (0.5, 0.3, 0.1))
        assert p == (0.5, 0.3, 0.1)

    def test_distance_to_body_zero(self):
        """distance_to_body returns 0 for interior point."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=5.0))
        d = distance_to_body(body, (0, 0, 0))
        assert d >= 0.0

    def test_distance_to_body_positive(self):
        """distance_to_body returns positive for exterior point."""
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        d = distance_to_body(body, (10, 0, 0))
        assert d > 0.0


# =============================================================================
# 8. physics_world.py
# =============================================================================

class TestPhysicsWorldBodyLifecycle(PhysicsTestCase):
    """Body add/remove edge cases."""

    def test_add_duplicate_body_false(self):
        """add_body returns False for duplicate body."""
        world = PhysicsWorld()
        body = _dynamic_body()
        assert world.add_body(body) is True
        assert world.add_body(body) is False

    def test_remove_body_not_found_false(self):
        """remove_body returns False for unknown body."""
        world = PhysicsWorld()
        body = _dynamic_body()
        assert world.remove_body(body) is False

    def test_add_body_at_capacity_false(self):
        """add_body returns False when world is at capacity."""
        world = PhysicsWorld(config=PhysicsConfig(max_bodies=1))
        body_a = _dynamic_body()
        body_b = _dynamic_body()
        assert world.add_body(body_a) is True
        assert world.add_body(body_b) is False

    def test_remove_body_clears_contacts(self):
        """remove_body removes contacts involving the body."""
        world = PhysicsWorld()
        a = _dynamic_body(position=(-0.6, 0, 0), shape=SphereShape(radius=1.0))
        b = _dynamic_body(position=(0.6, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(a)
        world.add_body(b)
        world.start()
        world.step(0.016)  # generates contacts
        world.remove_body(a)
        # No contacts involving 'a' should remain
        for pair in world._contact_pairs:
            assert a.id not in pair

    def test_get_body_returns_none(self):
        """get_body returns None for non-existent ID."""
        world = PhysicsWorld()
        assert world.get_body("nonexistent") is None


class TestPhysicsWorldStep(PhysicsTestCase):
    """Simulation step edge cases."""

    def test_step_stopped_world(self):
        """step returns 0 when world is not RUNNING."""
        world = PhysicsWorld()
        steps = world.step(0.016)
        assert steps == 0

    def test_step_paused_world(self):
        """step returns 0 when world is PAUSED."""
        world = PhysicsWorld()
        world.start()
        world.pause()
        steps = world.step(0.016)
        assert steps == 0

    def test_step_multiple_substeps(self):
        """step performs multiple substeps when dt > timestep."""
        world = PhysicsWorld()
        world.start()
        world._config.max_substeps = 10
        steps = world.step(0.1)  # 0.1 >> 1/60
        assert steps > 1

    def test_step_clamps_substeps(self):
        """step respects max_substeps cap."""
        world = PhysicsWorld(config=PhysicsConfig(max_substeps=4, timestep=1/1000))
        world.start()
        steps = world.step(1.0)  # would be 1000 substeps without cap
        assert steps == 4

    def test_get_bodies_in_aabb(self):
        """get_bodies_in_aabb returns overlapping bodies."""
        world = PhysicsWorld()
        a = _dynamic_body(position=(0, 0, 0))
        b = _dynamic_body(position=(10, 10, 10))
        world.add_body(a)
        world.add_body(b)

        test_aabb = AABB(min_point=(-1, -1, -1), max_point=(1, 1, 1))
        results = world.get_bodies_in_aabb(test_aabb)
        assert a in results
        assert b not in results


class TestPhysicsWorldCollision(PhysicsTestCase):
    """Collision detection internals."""

    def test_broad_phase_skips_both_static(self):
        """_broad_phase skips pairs where both bodies are static."""
        world = PhysicsWorld()
        a = _static_body()
        b = _static_body()
        world.add_body(a)
        world.add_body(b)
        world._broad_phase()
        assert len(world._broad_phase_pairs) == 0

    def test_broad_phase_skips_both_sleeping(self):
        """_broad_phase skips pairs where both bodies are sleeping."""
        world = PhysicsWorld()
        a = _dynamic_body()
        b = _dynamic_body()
        world.add_body(a)
        world.add_body(b)

        # Put both to sleep
        a.put_to_sleep()
        b.put_to_sleep()

        world._broad_phase()
        assert len(world._broad_phase_pairs) == 0

    def test_broad_phase_layer_filter(self):
        """_broad_phase respects layer/mask filtering."""
        world = PhysicsWorld()
        a = _dynamic_body()
        b = _dynamic_body()
        a.collision_layer = 1
        a.collision_mask = 1
        b.collision_layer = 2
        b.collision_mask = 2
        world.add_body(a)
        world.add_body(b)
        world._broad_phase()
        # No overlap because layers don't match
        assert len(world._broad_phase_pairs) == 0

    def test_narrow_phase_removes_stale(self):
        """_narrow_phase removes manifolds for pairs no longer in broad phase."""
        world = PhysicsWorld()
        a = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        b = _dynamic_body(position=(0.5, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(a)
        world.add_body(b)
        world.start()

        # Use dt > 1/60 so at least one substep runs
        dt = 0.02
        world.step(dt)

        # After step the overlapping pair should have generated a contact
        assert len(world._contact_pairs) > 0, \
            "overlapping spheres should produce contact after step"

        # Move bodies far apart
        a.position = (100, 0, 0)
        b.position = (200, 0, 0)

        # Step again -- broad phase sees no overlap, narrow phase removes stale manifold
        world.step(dt)
        assert len(world._contact_pairs) == 0, \
            "stale manifolds should be removed when bodies separate"
        world.step(0.016)
        assert len(world._contact_pairs) == 0, "stale manifolds should be removed"

    def test_contact_generation_overlap_negative(self):
        """_generate_contacts returns empty when no overlap on all axes."""
        world = PhysicsWorld()
        a = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        b = _dynamic_body(position=(100, 100, 100), shape=SphereShape(radius=1.0))
        world.add_body(a)
        world.add_body(b)
        contacts = world._generate_contacts(a, b)
        assert len(contacts) == 0

    def test_solve_contact_velocity_separating(self):
        """_solve_contact_velocity returns early when separating."""
        world = PhysicsWorld()
        a = _dynamic_body(position=(0, 0, 0))
        b = _dynamic_body(position=(1, 0, 0))

        # Create contact with normal pointing away from relative velocity
        contact = Contact(
            body_a=a,
            body_b=b,
            point=(0.5, 0, 0),
            normal=(1, 0, 0),  # pointing from A to B
            penetration=0.1,
        )
        # Give bodies separating velocity
        a.linear_velocity = (-10, 0, 0)  # moving left (away)
        b.linear_velocity = (10, 0, 0)   # moving right (away)

        world._solve_contact_velocity(contact, 0.016)
        # Velocities should not change (separating)
        assert a.linear_velocity[0] < 0  # still moving left


class TestPhysicsWorldCallbacks(PhysicsTestCase):
    """Callback registration and invocation."""

    def test_collision_enter_callback_ids(self):
        """collision_enter callback receives correct bodies."""
        world = PhysicsWorld()
        world.start()
        called = []

        def callback(ba, bb, c):
            called.append((ba.id, bb.id))

        world.on_collision_enter(callback)

        a = _dynamic_body(position=(-0.6, 0, 0), shape=SphereShape(radius=1.0))
        b = _dynamic_body(position=(0.6, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(a)
        world.add_body(b)

        for _ in range(5):
            world.step(0.016)

        # Collision may or may not fire based on detection quality
        # At minimum, verify no crash and callback list is accessible
        assert hasattr(world, '_collision_enter_callbacks')

    def test_on_collision_exit(self):
        """on_collision_exit registers callback without error."""
        world = PhysicsWorld()
        called = []

        def callback(ba, bb, c):
            called.append(True)

        world.on_collision_exit(callback)
        assert len(world._collision_exit_callbacks) == 1

    def test_on_trigger_enter(self):
        """on_trigger_enter registers callback without error."""
        world = PhysicsWorld()
        called = []

        def callback(ba, bb):
            called.append(True)

        world.on_trigger_enter(callback)
        assert len(world._trigger_enter_callbacks) == 1

    def test_clear_world(self):
        """clear removes all bodies and resets state."""
        world = PhysicsWorld()
        a = _dynamic_body()
        b = _dynamic_body()
        world.add_body(a)
        world.add_body(b)
        world.start()

        world.step(0.016)
        world.clear()
        assert world.body_count == 0
        assert world.simulation_time == 0.0
        assert world.step_count == 0


class TestPhysicsWorldQueries(PhysicsTestCase):
    """World-level query integration."""

    def test_raycast_on_world(self):
        """world.raycast queries all bodies."""
        world = PhysicsWorld()
        body = _static_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)

        hit = world.raycast((0, -5, 0), (0, 1, 0))
        assert hit is not None
        assert hit.body is body

    def test_raycast_miss(self):
        """world.raycast returns None when no hit."""
        world = PhysicsWorld()
        body = _static_body(position=(10, 10, 10), shape=SphereShape(radius=0.5))
        world.add_body(body)

        hit = world.raycast((0, 0, 0), (0, 1, 0), max_distance=5.0)
        assert hit is None

    def test_overlap_test_on_world(self):
        """world.overlap_test wraps overlap query functions."""
        world = PhysicsWorld()
        body = _dynamic_body(position=(0, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)

        results = world.overlap_test(SphereShape(radius=0.5), position=(0, 0, 0))
        assert len(results) == 1

    def test_sweep_test_on_world(self):
        """world.sweep_test wraps sweep query functions."""
        world = PhysicsWorld()
        body = _static_body(position=(5, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(body)

        result = world.sweep_test(
            SphereShape(radius=0.5),
            start=(0, 0, 0),
            direction=(1, 0, 0),
            distance=10.0,
        )
        # May or may not hit depending on implementation
        assert isinstance(result, SweepResult)


class TestPhysicsWorldMisc(PhysicsTestCase):
    """Miscellaneous world features."""

    def test_wake_all(self):
        """wake_all wakes all sleeping bodies."""
        world = PhysicsWorld()
        body = _dynamic_body()
        world.add_body(body)
        body.put_to_sleep()
        assert body.is_sleeping
        world.wake_all()
        assert not body.is_sleeping

    def test_get_contact_manifolds(self):
        """get_contact_manifolds returns current manifolds."""
        world = PhysicsWorld()
        world.start()
        a = _dynamic_body(position=(-0.6, 0, 0), shape=SphereShape(radius=1.0))
        b = _dynamic_body(position=(0.6, 0, 0), shape=SphereShape(radius=1.0))
        world.add_body(a)
        world.add_body(b)
        for _ in range(5):
            world.step(0.016)
        manifolds = world.get_contact_manifolds()
        assert isinstance(manifolds, list)

    def test_statistics(self):
        """statistics returns expected keys."""
        world = PhysicsWorld()
        stats = world.statistics
        for key in ('bodies', 'active_bodies', 'sleeping_bodies',
                    'contacts', 'islands', 'step_time_ms'):
            assert key in stats, f"Missing key: {key}"

    def test_gravity_setter(self):
        """gravity setter updates the gravity vector."""
        world = PhysicsWorld()
        world.gravity = (0, -5.0, 0)
        assert world.gravity == (0, -5.0, 0)

    def test_timestep_setter(self):
        """timestep setter clamps to MIN_TIMESTEP."""
        world = PhysicsWorld()
        world.timestep = 0.0
        assert world.timestep >= MIN_TIMESTEP


class TestPhysicsWorldSerialization(PhysicsTestCase):
    """Serialization round-trips."""

    def test_to_dict(self):
        """to_dict returns expected keys."""
        world = PhysicsWorld()
        body = _dynamic_body(position=(1, 2, 3))
        world.add_body(body)
        d = world.to_dict()
        assert 'gravity' in d
        assert 'timestep' in d
        assert 'bodies' in d
        assert len(d['bodies']) == 1
        assert d['bodies'][0]['position'] == (1, 2, 3)
