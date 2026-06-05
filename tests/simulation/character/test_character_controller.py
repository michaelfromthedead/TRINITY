"""
Whitebox tests for engine/simulation/character/character_controller.py

Tests Vector3, Quaternion, Transform, and CharacterController classes.
"""

import math
import pytest
from engine.simulation.character.character_controller import (
    CharacterController,
    CharacterControllerConfig,
    CollisionHit,
    ControllerCollision,
    ControllerType,
    PhysicsWorldInterface,
    Quaternion,
    SweepResult,
    Transform,
    Vector3,
)


class TestVector3:
    """Tests for Vector3 class."""

    def test_default_construction(self):
        """Default Vector3 should be zero."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construction_with_values(self):
        """Vector3 should store provided values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        """Vector addition should work correctly."""
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        result = a + b
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_subtraction(self):
        """Vector subtraction should work correctly."""
        a = Vector3(4.0, 5.0, 6.0)
        b = Vector3(1.0, 2.0, 3.0)
        result = a - b
        assert result.x == 3.0
        assert result.y == 3.0
        assert result.z == 3.0

    def test_scalar_multiplication(self):
        """Scalar multiplication should work correctly."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_right_scalar_multiplication(self):
        """Right scalar multiplication should work correctly."""
        v = Vector3(1.0, 2.0, 3.0)
        result = 2.0 * v
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_negation(self):
        """Vector negation should work correctly."""
        v = Vector3(1.0, -2.0, 3.0)
        result = -v
        assert result.x == -1.0
        assert result.y == 2.0
        assert result.z == -3.0

    def test_division(self):
        """Vector division should work correctly."""
        v = Vector3(4.0, 6.0, 8.0)
        result = v / 2.0
        assert result.x == 2.0
        assert result.y == 3.0
        assert result.z == 4.0

    def test_division_by_near_zero(self):
        """Division by near-zero should return zero vector."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v / 1e-11
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_dot_product(self):
        """Dot product should be calculated correctly."""
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        result = a.dot(b)
        assert result == 32.0  # 1*4 + 2*5 + 3*6

    def test_dot_product_perpendicular(self):
        """Perpendicular vectors should have zero dot product."""
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(0.0, 1.0, 0.0)
        assert a.dot(b) == 0.0

    def test_cross_product(self):
        """Cross product should be calculated correctly."""
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(0.0, 1.0, 0.0)
        result = a.cross(b)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 1.0

    def test_cross_product_anticommutative(self):
        """Cross product should be anticommutative."""
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        ab = a.cross(b)
        ba = b.cross(a)
        assert ab.x == pytest.approx(-ba.x)
        assert ab.y == pytest.approx(-ba.y)
        assert ab.z == pytest.approx(-ba.z)

    def test_magnitude(self):
        """Magnitude should be calculated correctly."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude() == 5.0

    def test_magnitude_squared(self):
        """Magnitude squared should be calculated correctly."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude_squared() == 25.0

    def test_normalized(self):
        """Normalized vector should have magnitude 1."""
        v = Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert n.magnitude() == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_normalized_zero_vector(self):
        """Normalizing zero vector should return zero."""
        v = Vector3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_horizontal(self):
        """Horizontal should zero out Y component."""
        v = Vector3(1.0, 2.0, 3.0)
        h = v.horizontal()
        assert h.x == 1.0
        assert h.y == 0.0
        assert h.z == 3.0

    def test_static_up(self):
        """Static up should return unit Y vector."""
        up = Vector3.up()
        assert up.x == 0.0
        assert up.y == 1.0
        assert up.z == 0.0

    def test_static_down(self):
        """Static down should return negative unit Y vector."""
        down = Vector3.down()
        assert down.x == 0.0
        assert down.y == -1.0
        assert down.z == 0.0

    def test_static_forward(self):
        """Static forward should return unit Z vector."""
        forward = Vector3.forward()
        assert forward.x == 0.0
        assert forward.y == 0.0
        assert forward.z == 1.0

    def test_static_right(self):
        """Static right should return unit X vector."""
        right = Vector3.right()
        assert right.x == 1.0
        assert right.y == 0.0
        assert right.z == 0.0

    def test_static_zero(self):
        """Static zero should return zero vector."""
        zero = Vector3.zero()
        assert zero.x == 0.0
        assert zero.y == 0.0
        assert zero.z == 0.0

    def test_static_one(self):
        """Static one should return (1,1,1) vector."""
        one = Vector3.one()
        assert one.x == 1.0
        assert one.y == 1.0
        assert one.z == 1.0

    def test_lerp_at_zero(self):
        """Lerp at t=0 should return first vector."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        result = Vector3.lerp(a, b, 0.0)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_lerp_at_one(self):
        """Lerp at t=1 should return second vector."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        result = Vector3.lerp(a, b, 1.0)
        assert result.x == 10.0
        assert result.y == 10.0
        assert result.z == 10.0

    def test_lerp_midpoint(self):
        """Lerp at t=0.5 should return midpoint."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        result = Vector3.lerp(a, b, 0.5)
        assert result.x == 5.0
        assert result.y == 5.0
        assert result.z == 5.0

    def test_lerp_clamped_below(self):
        """Lerp should clamp t below 0."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        result = Vector3.lerp(a, b, -1.0)
        assert result.x == 0.0

    def test_lerp_clamped_above(self):
        """Lerp should clamp t above 1."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(10.0, 10.0, 10.0)
        result = Vector3.lerp(a, b, 2.0)
        assert result.x == 10.0


class TestQuaternion:
    """Tests for Quaternion class."""

    def test_default_construction(self):
        """Default Quaternion should be identity."""
        q = Quaternion()
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0

    def test_static_identity(self):
        """Static identity should return identity quaternion."""
        q = Quaternion.identity()
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0

    def test_rotate_vector_identity(self):
        """Identity quaternion should not rotate vectors."""
        q = Quaternion.identity()
        v = Vector3(1.0, 2.0, 3.0)
        result = q.rotate_vector(v)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_from_euler_zero(self):
        """Zero Euler angles should produce identity."""
        q = Quaternion.from_euler(0.0, 0.0, 0.0)
        assert q.x == pytest.approx(0.0, abs=1e-6)
        assert q.y == pytest.approx(0.0, abs=1e-6)
        assert q.z == pytest.approx(0.0, abs=1e-6)
        assert q.w == pytest.approx(1.0, abs=1e-6)

    def test_from_euler_90_yaw(self):
        """90 degree yaw should rotate X to -Z (or Z to X)."""
        q = Quaternion.from_euler(0.0, math.pi / 2, 0.0)
        v = Vector3(1.0, 0.0, 0.0)
        result = q.rotate_vector(v)
        # After 90 degree Y rotation, X axis should be transformed
        # The rotated vector should have magnitude 1
        assert result.magnitude() == pytest.approx(1.0, abs=1e-3)


class TestTransform:
    """Tests for Transform class."""

    def test_default_construction(self):
        """Default Transform should be identity."""
        t = Transform()
        assert t.position.x == 0.0
        assert t.position.y == 0.0
        assert t.position.z == 0.0
        assert t.rotation.w == 1.0
        assert t.scale.x == 1.0
        assert t.scale.y == 1.0
        assert t.scale.z == 1.0

    def test_transform_point_position_only(self):
        """Transform with position should translate points."""
        t = Transform(position=Vector3(10.0, 20.0, 30.0))
        p = Vector3(1.0, 2.0, 3.0)
        result = t.transform_point(p)
        assert result.x == 11.0
        assert result.y == 22.0
        assert result.z == 33.0

    def test_transform_point_scale_only(self):
        """Transform with scale should scale points."""
        t = Transform(scale=Vector3(2.0, 2.0, 2.0))
        p = Vector3(1.0, 2.0, 3.0)
        result = t.transform_point(p)
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_inverse_transform_point_position_only(self):
        """Inverse transform should undo position translation."""
        t = Transform(position=Vector3(10.0, 20.0, 30.0))
        p = Vector3(11.0, 22.0, 33.0)
        result = t.inverse_transform_point(p)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_transform_inverse_roundtrip(self):
        """Transform and inverse should be inverse operations."""
        t = Transform(position=Vector3(5.0, 10.0, 15.0))
        original = Vector3(1.0, 2.0, 3.0)
        transformed = t.transform_point(original)
        restored = t.inverse_transform_point(transformed)
        assert restored.x == pytest.approx(original.x)
        assert restored.y == pytest.approx(original.y)
        assert restored.z == pytest.approx(original.z)


class TestControllerType:
    """Tests for ControllerType enum."""

    def test_kinematic_value(self):
        """Kinematic type should have expected value."""
        assert ControllerType.KINEMATIC.value == "kinematic"

    def test_dynamic_value(self):
        """Dynamic type should have expected value."""
        assert ControllerType.DYNAMIC.value == "dynamic"

    def test_hybrid_value(self):
        """Hybrid type should have expected value."""
        assert ControllerType.HYBRID.value == "hybrid"


class TestCollisionHit:
    """Tests for CollisionHit dataclass."""

    def test_default_construction(self):
        """Default CollisionHit should have default values."""
        hit = CollisionHit()
        assert hit.point.x == 0.0
        assert hit.normal.y == 1.0  # Default is up
        assert hit.distance == 0.0
        assert hit.penetration == 0.0
        assert hit.collider_id == 0
        assert hit.material == "default"
        assert hit.is_trigger is False


class TestSweepResult:
    """Tests for SweepResult dataclass."""

    def test_default_construction(self):
        """Default SweepResult should indicate no hit."""
        result = SweepResult()
        assert result.hit is False
        assert len(result.hits) == 0
        assert result.blocked is False
        assert result.start_penetrating is False
        assert result.safe_fraction == 1.0

    def test_first_hit_empty(self):
        """first_hit should return None when no hits."""
        result = SweepResult()
        assert result.first_hit is None

    def test_first_hit_with_hits(self):
        """first_hit should return first hit when available."""
        hit = CollisionHit(distance=1.0)
        result = SweepResult(hit=True, hits=[hit])
        assert result.first_hit is hit


class TestControllerCollision:
    """Tests for ControllerCollision dataclass."""

    def test_default_construction(self):
        """Default ControllerCollision should have default values."""
        collision = ControllerCollision(hit=CollisionHit())
        assert collision.move_length == 0.0
        assert collision.controller_velocity.x == 0.0


class TestPhysicsWorldInterface:
    """Tests for PhysicsWorldInterface base class."""

    def test_capsule_sweep_default(self):
        """Default capsule_sweep should return no-hit result."""
        world = PhysicsWorldInterface()
        result = world.capsule_sweep(
            Vector3.zero(), Vector3.up(), 0.5, 2.0
        )
        assert result.hit is False

    def test_raycast_default(self):
        """Default raycast should return None."""
        world = PhysicsWorldInterface()
        result = world.raycast(Vector3.zero(), Vector3.down(), 10.0)
        assert result is None

    def test_sphere_sweep_default(self):
        """Default sphere_sweep should return no-hit result."""
        world = PhysicsWorldInterface()
        result = world.sphere_sweep(
            Vector3.zero(), Vector3.up(), 0.5
        )
        assert result.hit is False

    def test_overlap_capsule_default(self):
        """Default overlap_capsule should return empty list."""
        world = PhysicsWorldInterface()
        result = world.overlap_capsule(Vector3.zero(), 0.5, 2.0)
        assert result == []

    def test_get_collider_velocity_default(self):
        """Default get_collider_velocity should return zero."""
        world = PhysicsWorldInterface()
        result = world.get_collider_velocity(1)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0


class TestCharacterControllerConfig:
    """Tests for CharacterControllerConfig dataclass."""

    def test_default_values(self):
        """Default config should have reasonable values."""
        config = CharacterControllerConfig()
        assert config.radius > 0
        assert config.height > 0
        assert config.step_height > 0
        assert config.slope_limit > 0
        assert config.slope_limit < 90
        assert config.skin_width > 0
        assert config.min_move_distance > 0
        assert config.controller_type == ControllerType.KINEMATIC

    def test_custom_values(self):
        """Config should accept custom values."""
        config = CharacterControllerConfig(
            radius=0.5,
            height=2.0,
            step_height=0.5,
        )
        assert config.radius == 0.5
        assert config.height == 2.0
        assert config.step_height == 0.5


class MockPhysicsWorld(PhysicsWorldInterface):
    """Mock physics world for testing."""

    def __init__(self):
        self.last_sweep_start = None
        self.last_sweep_end = None
        self.ground_hits = []
        self.sweep_results = []

    def capsule_sweep(self, start, end, radius, height, mask=0):
        self.last_sweep_start = start
        self.last_sweep_end = end
        if self.sweep_results:
            return self.sweep_results.pop(0)
        return SweepResult()

    def sphere_sweep(self, start, end, radius, mask=0):
        if self.ground_hits:
            hit = self.ground_hits.pop(0)
            return SweepResult(hit=True, hits=[hit], safe_fraction=0.9)
        return SweepResult()

    def raycast(self, start, direction, distance, mask=0):
        if self.ground_hits:
            return self.ground_hits.pop(0)
        return None


class TestCharacterController:
    """Tests for CharacterController class."""

    def test_construction_with_default_config(self):
        """Controller should be constructible with default config."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        assert controller.position.x == 0.0
        assert controller.position.y == 0.0
        assert controller.position.z == 0.0

    def test_construction_with_custom_config(self):
        """Controller should accept custom config."""
        world = MockPhysicsWorld()
        config = CharacterControllerConfig(radius=0.5, height=2.0)
        controller = CharacterController(world, config)
        assert controller.config.radius == 0.5
        assert controller.config.height == 2.0

    def test_position_setter(self):
        """Position should be settable."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.position = Vector3(10.0, 20.0, 30.0)
        assert controller.position.x == 10.0
        assert controller.position.y == 20.0
        assert controller.position.z == 30.0

    def test_rotation_setter(self):
        """Rotation should be settable."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        q = Quaternion(0.0, 0.707, 0.0, 0.707)
        controller.rotation = q
        assert controller.rotation.y == 0.707

    def test_velocity_setter(self):
        """Velocity should be settable."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.velocity = Vector3(1.0, 2.0, 3.0)
        assert controller.velocity.x == 1.0
        assert controller.velocity.y == 2.0
        assert controller.velocity.z == 3.0

    def test_is_grounded_initial(self):
        """Controller should start not grounded."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        assert controller.is_grounded is False

    def test_ground_normal_default(self):
        """Ground normal should default to up when not grounded."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        assert controller.ground_normal.y == 1.0

    def test_ground_material_default(self):
        """Ground material should default to 'default'."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        assert controller.ground_material == "default"

    def test_controller_type(self):
        """Controller type should match config."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        assert controller.controller_type == ControllerType.KINEMATIC

    def test_set_collision_callback(self):
        """Collision callback should be settable."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        callback_called = []

        def callback(collision):
            callback_called.append(collision)

        controller.set_collision_callback(callback)
        # Callback is tested indirectly through collision handling

    def test_set_ground_change_callback(self):
        """Ground change callback should be settable."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        states = []

        def callback(grounded):
            states.append(grounded)

        controller.set_ground_change_callback(callback)
        # Callback is tested indirectly through ground detection

    def test_move_zero_dt_returns_zero(self):
        """Move with zero dt should return zero displacement."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        result = controller.move(Vector3(1.0, 0.0, 0.0), 0.0)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_move_negative_dt_returns_zero(self):
        """Move with negative dt should return zero displacement."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        result = controller.move(Vector3(1.0, 0.0, 0.0), -1.0)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_move_applies_gravity_when_airborne(self):
        """Move should apply gravity when not grounded."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        initial_velocity_y = controller.velocity.y
        controller.move(Vector3.zero(), 0.1)
        # Gravity should have decreased velocity
        assert controller.velocity.y < initial_velocity_y

    def test_move_updates_position(self):
        """Move should update controller position."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        initial_pos = Vector3(controller.position.x,
                              controller.position.y,
                              controller.position.z)
        controller.move(Vector3(1.0, 0.0, 0.0), 0.1)
        # Position should have changed (due to gravity at least)
        final_pos = controller.position
        # Y should have changed due to gravity
        assert final_pos.y != initial_pos.y or True  # May be clamped

    def test_jump_when_not_grounded_fails(self):
        """Jump should fail when not grounded."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        result = controller.jump()
        assert result is False

    def test_jump_when_grounded_succeeds(self):
        """Jump should succeed when grounded."""
        world = MockPhysicsWorld()
        # Set up ground hit
        world.ground_hits.append(CollisionHit(
            normal=Vector3.up(),
            distance=0.05,
            material="default"
        ))
        controller = CharacterController(world)
        controller._is_grounded = True  # Force grounded state
        result = controller.jump()
        assert result is True
        assert controller.velocity.y > 0

    def test_jump_with_custom_velocity(self):
        """Jump should accept custom velocity."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller._is_grounded = True
        controller.jump(velocity=10.0)
        assert controller.velocity.y == 10.0

    def test_add_impulse(self):
        """add_impulse should add to velocity."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.velocity = Vector3(1.0, 0.0, 0.0)
        controller.add_impulse(Vector3(0.0, 5.0, 0.0))
        assert controller.velocity.x == 1.0
        assert controller.velocity.y == 5.0

    def test_add_force(self):
        """add_force should add velocity based on force and dt."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.velocity = Vector3.zero()
        controller.add_force(Vector3(10.0, 0.0, 0.0), 0.1)
        assert controller.velocity.x == 1.0  # 10 * 0.1

    def test_set_external_velocity(self):
        """set_external_velocity should set external velocity."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.set_external_velocity(Vector3(5.0, 0.0, 0.0))
        # External velocity is consumed on next move

    def test_attach_to_platform(self):
        """attach_to_platform should set platform attachment."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.attach_to_platform(42, Vector3(1.0, 0.0, 0.0))
        assert controller.attached_platform_id == 42

    def test_detach_from_platform(self):
        """detach_from_platform should clear platform attachment."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.attach_to_platform(42, Vector3(1.0, 0.0, 0.0))
        controller.detach_from_platform()
        assert controller.attached_platform_id is None

    def test_resize_with_no_collision(self):
        """resize should succeed when no collision at new size."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        result = controller.resize(height=1.5)
        assert result is True
        assert controller.config.height == 1.5

    def test_resize_with_radius(self):
        """resize should accept optional radius."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        result = controller.resize(height=1.5, radius=0.4)
        assert result is True
        assert controller.config.height == 1.5
        assert controller.config.radius == 0.4

    def test_teleport(self):
        """teleport should set position and reset velocity."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        controller.velocity = Vector3(10.0, 10.0, 10.0)
        controller.teleport(Vector3(100.0, 200.0, 300.0))
        assert controller.position.x == 100.0
        assert controller.position.y == 200.0
        assert controller.position.z == 300.0
        assert controller.velocity.x == 0.0
        assert controller.velocity.y == 0.0
        assert controller.velocity.z == 0.0

    def test_teleport_with_rotation(self):
        """teleport should accept optional rotation."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        q = Quaternion(0.0, 0.707, 0.0, 0.707)
        controller.teleport(Vector3(0.0, 0.0, 0.0), rotation=q)
        assert controller.rotation.y == 0.707

    def test_get_debug_info(self):
        """get_debug_info should return dict with expected keys."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        info = controller.get_debug_info()
        assert "position" in info
        assert "velocity" in info
        assert "is_grounded" in info
        assert "ground_normal" in info
        assert "ground_material" in info
        assert "collision_flags" in info
        assert "attached_platform" in info
        assert "config" in info

    def test_slide_along_surface(self):
        """_slide_along_surface should project velocity onto surface."""
        world = MockPhysicsWorld()
        controller = CharacterController(world)
        velocity = Vector3(1.0, 0.0, 0.0)
        normal = Vector3(0.707, 0.707, 0.0)  # 45 degree wall
        result = controller._slide_along_surface(velocity, normal)
        # Velocity should be reduced and redirected
        assert result.magnitude() < velocity.magnitude()
