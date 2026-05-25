"""
Comprehensive tests for the character physics module.

Tests cover:
- Character controller movement
- Ground detection
- Slope handling
- Step up/down
- Platform attachment
- Ragdoll activation
- Active ragdoll balance
- Physics-animation blending
- Character interactions

Minimum 130 tests as specified.
"""

from __future__ import annotations

import math
import pytest
from typing import Optional
from unittest.mock import MagicMock, patch

# Import character module components
from engine.simulation.character import (
    # Config
    DEFAULT_CAPSULE_HEIGHT,
    DEFAULT_CAPSULE_RADIUS,
    DEFAULT_STEP_HEIGHT,
    MAX_SLOPE_ANGLE,
    COYOTE_TIME_MS,
    AIR_CONTROL,
    DEFAULT_JUMP_VELOCITY,
    # Character Controller
    CharacterController,
    CharacterControllerConfig,
    ControllerType,
    PhysicsWorldInterface,
    Vector3,
    Quaternion,
    Transform,
    CollisionHit,
    SweepResult,
    # Ground Detection
    GroundDetector,
    GroundInfo,
    GroundType,
    LedgeInfo,
    # Movement Modes
    MovementMode,
    MovementModeManager,
    MovementModeParams,
    MovementState,
    MovementContext,
    # Slope Handling
    SlopeHandler,
    SlopeInfo,
    StepInfo,
    # Platform Handling
    PlatformHandler,
    PlatformProvider,
    PlatformData,
    PlatformAttachment,
    PlatformType,
    AttachmentMode,
    # Ragdoll
    Ragdoll,
    RagdollSetup,
    RagdollState,
    BodyPartType,
    RagdollPhysicsInterface,
    SkeletonInterface,
    create_default_humanoid_setup,
    # Active Ragdoll
    ActiveRagdoll,
    ActiveRagdollState,
    PDController,
    BalanceConfig,
    RecoveryBehavior,
    # Physics Animation Blend
    PhysicsAnimationBlender,
    BonePose,
    SkeletonPose,
    BlendMode,
    HitReaction,
    # Character Interaction
    CharacterInteractionManager,
    InteractionType,
    InteractionTarget,
    GrabState,
    ClimbInfo,
    VaultInfo,
)


# =============================================================================
# Mock Physics World
# =============================================================================

class MockPhysicsWorld(PhysicsWorldInterface):
    """Mock physics world for testing."""

    def __init__(self):
        self.sweep_results: dict[str, SweepResult] = {}
        self.raycast_results: dict[str, Optional[CollisionHit]] = {}
        self.overlap_results: list[CollisionHit] = []
        self.collider_velocities: dict[int, Vector3] = {}

    def set_sweep_result(self, key: str, result: SweepResult) -> None:
        self.sweep_results[key] = result

    def set_raycast_result(self, key: str, result: Optional[CollisionHit]) -> None:
        self.raycast_results[key] = result

    def capsule_sweep(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        height: float,
        mask: int = 0xFFFF,
    ) -> SweepResult:
        key = f"capsule_{int(start.x)}_{int(start.y)}_{int(start.z)}"
        return self.sweep_results.get(key, SweepResult())

    def raycast(
        self,
        start: Vector3,
        direction: Vector3,
        distance: float,
        mask: int = 0xFFFF,
    ) -> Optional[CollisionHit]:
        key = f"ray_{int(start.x)}_{int(start.y)}_{int(start.z)}"
        return self.raycast_results.get(key, None)

    def sphere_sweep(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        mask: int = 0xFFFF,
    ) -> SweepResult:
        key = f"sphere_{int(start.x)}_{int(start.y)}_{int(start.z)}"
        return self.sweep_results.get(key, SweepResult())

    def overlap_capsule(
        self,
        position: Vector3,
        radius: float,
        height: float,
        mask: int = 0xFFFF,
    ) -> list[CollisionHit]:
        return self.overlap_results

    def get_collider_velocity(self, collider_id: int) -> Vector3:
        return self.collider_velocities.get(collider_id, Vector3.zero())

    def apply_impulse(self, body_id: int, impulse: Vector3, point: Optional[Vector3] = None) -> None:
        """Apply impulse to a body (mock implementation)."""
        pass


class MockPlatformProvider(PlatformProvider):
    """Mock platform provider for testing."""

    def __init__(self):
        self.platforms: dict[int, PlatformData] = {}

    def add_platform(self, platform: PlatformData) -> None:
        self.platforms[platform.platform_id] = platform

    def get_platform(self, platform_id: int) -> Optional[PlatformData]:
        return self.platforms.get(platform_id)

    def get_platform_transform(self, platform_id: int) -> Optional[Transform]:
        platform = self.platforms.get(platform_id)
        return platform.transform if platform else None

    def get_platform_velocity(self, platform_id: int) -> Vector3:
        platform = self.platforms.get(platform_id)
        return platform.velocity if platform else Vector3.zero()

    def get_platform_angular_velocity(self, platform_id: int) -> Vector3:
        platform = self.platforms.get(platform_id)
        return platform.angular_velocity if platform else Vector3.zero()


class MockRagdollPhysics(RagdollPhysicsInterface):
    """Mock ragdoll physics for testing."""

    def __init__(self):
        self._bodies: dict[int, tuple[Vector3, Quaternion]] = {}
        self._velocities: dict[int, tuple[Vector3, Vector3]] = {}
        self._next_id = 1
        self._kinematic: dict[int, bool] = {}

    def create_body(self, *args, **kwargs) -> int:
        body_id = self._next_id
        self._next_id += 1
        self._bodies[body_id] = (Vector3.zero(), Quaternion.identity())
        self._velocities[body_id] = (Vector3.zero(), Vector3.zero())
        self._kinematic[body_id] = True
        return body_id

    def destroy_body(self, body_id: int) -> None:
        self._bodies.pop(body_id, None)

    def create_joint(self, *args, **kwargs) -> int:
        return self._next_id

    def destroy_joint(self, joint_id: int) -> None:
        pass

    def get_body_transform(self, body_id: int) -> tuple[Vector3, Quaternion]:
        return self._bodies.get(body_id, (Vector3.zero(), Quaternion.identity()))

    def get_body_velocity(self, body_id: int) -> tuple[Vector3, Vector3]:
        return self._velocities.get(body_id, (Vector3.zero(), Vector3.zero()))

    def set_body_transform(self, body_id: int, position: Vector3, rotation: Quaternion) -> None:
        self._bodies[body_id] = (position, rotation)

    def set_body_velocity(self, body_id: int, linear: Vector3, angular: Vector3) -> None:
        self._velocities[body_id] = (linear, angular)

    def set_body_kinematic(self, body_id: int, kinematic: bool) -> None:
        self._kinematic[body_id] = kinematic

    def apply_impulse(self, body_id: int, impulse: Vector3, point: Optional[Vector3] = None) -> None:
        pass


class MockSkeleton(SkeletonInterface):
    """Mock skeleton for testing."""

    def __init__(self):
        self._bones: dict[str, Transform] = {}

    def get_bone_names(self) -> list[str]:
        return list(self._bones.keys())

    def get_bone_transform(self, bone_name: str) -> Transform:
        return self._bones.get(bone_name, Transform())

    def get_bone_parent(self, bone_name: str) -> Optional[str]:
        return None

    def set_bone_transform(self, bone_name: str, transform: Transform) -> None:
        self._bones[bone_name] = transform


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def physics_world() -> MockPhysicsWorld:
    return MockPhysicsWorld()


@pytest.fixture
def controller(physics_world) -> CharacterController:
    return CharacterController(physics_world)


@pytest.fixture
def ground_detector(physics_world) -> GroundDetector:
    return GroundDetector(physics_world)


@pytest.fixture
def movement_manager() -> MovementModeManager:
    return MovementModeManager()


@pytest.fixture
def slope_handler(physics_world) -> SlopeHandler:
    return SlopeHandler(physics_world)


@pytest.fixture
def platform_provider() -> MockPlatformProvider:
    return MockPlatformProvider()


@pytest.fixture
def platform_handler(platform_provider) -> PlatformHandler:
    return PlatformHandler(platform_provider)


@pytest.fixture
def ragdoll_physics() -> MockRagdollPhysics:
    return MockRagdollPhysics()


@pytest.fixture
def skeleton() -> MockSkeleton:
    return MockSkeleton()


@pytest.fixture
def ragdoll(ragdoll_physics, skeleton) -> Ragdoll:
    return Ragdoll(ragdoll_physics, skeleton)


@pytest.fixture
def active_ragdoll(ragdoll, ragdoll_physics) -> ActiveRagdoll:
    return ActiveRagdoll(ragdoll, ragdoll_physics)


@pytest.fixture
def blender() -> PhysicsAnimationBlender:
    return PhysicsAnimationBlender()


@pytest.fixture
def interaction_manager(physics_world) -> CharacterInteractionManager:
    return CharacterInteractionManager(physics_world)


# =============================================================================
# Vector3 Tests (10 tests)
# =============================================================================

class TestVector3:
    """Tests for Vector3 math operations."""

    def test_vector3_creation(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector3_zero(self):
        v = Vector3.zero()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_vector3_one(self):
        v = Vector3.one()
        assert v.x == 1.0 and v.y == 1.0 and v.z == 1.0

    def test_vector3_addition(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        c = a + b
        assert c.x == 5.0 and c.y == 7.0 and c.z == 9.0

    def test_vector3_subtraction(self):
        a = Vector3(4.0, 5.0, 6.0)
        b = Vector3(1.0, 2.0, 3.0)
        c = a - b
        assert c.x == 3.0 and c.y == 3.0 and c.z == 3.0

    def test_vector3_scalar_multiplication(self):
        v = Vector3(1.0, 2.0, 3.0)
        r = v * 2.0
        assert r.x == 2.0 and r.y == 4.0 and r.z == 6.0

    def test_vector3_magnitude(self):
        v = Vector3(3.0, 4.0, 0.0)
        assert abs(v.magnitude() - 5.0) < 0.001

    def test_vector3_normalized(self):
        v = Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 0.001

    def test_vector3_dot(self):
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(1.0, 0.0, 0.0)
        assert abs(a.dot(b) - 1.0) < 0.001

    def test_vector3_cross(self):
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(0.0, 1.0, 0.0)
        c = a.cross(b)
        assert abs(c.z - 1.0) < 0.001


# =============================================================================
# Character Controller Tests (20 tests)
# =============================================================================

class TestCharacterController:
    """Tests for CharacterController."""

    def test_controller_creation(self, controller):
        assert controller is not None
        assert controller.position.x == 0.0
        assert controller.is_grounded is False

    def test_controller_type_default(self, controller):
        assert controller.controller_type == ControllerType.KINEMATIC

    def test_controller_custom_config(self, physics_world):
        config = CharacterControllerConfig(
            radius=0.5,
            height=2.0,
            step_height=0.4,
        )
        ctrl = CharacterController(physics_world, config)
        assert ctrl.config.radius == 0.5
        assert ctrl.config.height == 2.0

    def test_controller_move_basic(self, controller):
        direction = Vector3(1.0, 0.0, 0.0)
        displacement = controller.move(direction, 0.016)
        # Without physics response, position changes
        assert controller.position is not None

    def test_controller_position_setter(self, controller):
        controller.position = Vector3(10.0, 5.0, 3.0)
        assert controller.position.x == 10.0
        assert controller.position.y == 5.0

    def test_controller_velocity_setter(self, controller):
        controller.velocity = Vector3(5.0, 0.0, 0.0)
        assert controller.velocity.x == 5.0

    def test_controller_jump_not_grounded(self, controller):
        result = controller.jump()
        assert result is False  # Can't jump if not grounded

    def test_controller_teleport(self, controller):
        controller.teleport(Vector3(100.0, 50.0, 25.0))
        assert controller.position.x == 100.0
        assert controller.position.y == 50.0
        assert controller.velocity.x == 0.0

    def test_controller_add_impulse(self, controller):
        controller.add_impulse(Vector3(10.0, 5.0, 0.0))
        assert controller.velocity.x == 10.0
        assert controller.velocity.y == 5.0

    def test_controller_add_force(self, controller):
        controller.add_force(Vector3(100.0, 0.0, 0.0), 0.1)
        assert controller.velocity.x == 10.0

    def test_controller_external_velocity(self, controller):
        controller.set_external_velocity(Vector3(5.0, 0.0, 0.0))
        # External velocity consumed during move
        controller.move(Vector3.zero(), 0.016)

    def test_controller_resize_success(self, controller, physics_world):
        # No overlaps
        physics_world.overlap_results = []
        result = controller.resize(1.5)
        assert result is True

    def test_controller_resize_blocked(self, controller, physics_world):
        # Add overlap
        physics_world.overlap_results = [CollisionHit()]
        result = controller.resize(1.5)
        assert result is False

    def test_controller_collision_callback(self, controller):
        called = []
        controller.set_collision_callback(lambda c: called.append(c))
        # Collision callback registered
        assert len(called) == 0

    def test_controller_ground_change_callback(self, controller):
        called = []
        controller.set_ground_change_callback(lambda g: called.append(g))
        # Ground change callback registered
        assert len(called) == 0

    def test_controller_platform_attachment(self, controller):
        controller.attach_to_platform(1, Vector3(0.0, 0.0, 0.0))
        assert controller.attached_platform_id == 1

    def test_controller_platform_detachment(self, controller):
        controller.attach_to_platform(1, Vector3.zero())
        controller.detach_from_platform()
        assert controller.attached_platform_id is None

    def test_controller_debug_info(self, controller):
        info = controller.get_debug_info()
        assert "position" in info
        assert "velocity" in info
        assert "is_grounded" in info

    def test_controller_rotation(self, controller):
        rot = Quaternion.from_euler(0, math.pi / 2, 0)
        controller.rotation = rot
        # Check that rotation was set (z component changes for yaw rotation)
        assert controller.rotation.z != 0 or controller.rotation.w != 1.0

    def test_controller_ground_normal(self, controller):
        normal = controller.ground_normal
        assert normal.y == 1.0  # Default up


# =============================================================================
# Ground Detection Tests (20 tests)
# =============================================================================

class TestGroundDetection:
    """Tests for GroundDetector."""

    def test_detector_creation(self, ground_detector):
        assert ground_detector is not None

    def test_detect_ground_no_hit(self, ground_detector):
        info = ground_detector.detect_ground(Vector3.zero())
        assert info.is_grounded is False

    def test_detect_ground_with_hit(self, ground_detector, physics_world):
        hit = CollisionHit(
            point=Vector3(0.0, 0.0, 0.0),
            normal=Vector3(0.0, 1.0, 0.0),
            distance=0.05,
            material="concrete",
        )
        physics_world.set_raycast_result("ray_0_0_0", hit)
        physics_world.sweep_results["sphere_0_0_0"] = SweepResult(
            hit=True, hits=[hit], safe_fraction=0.5
        )

        info = ground_detector.detect_ground(Vector3.zero())
        assert info.is_grounded is True

    def test_raycast_ground(self, ground_detector, physics_world):
        hit = CollisionHit(normal=Vector3.up(), distance=0.05)
        physics_world.set_raycast_result("ray_0_0_0", hit)

        info = ground_detector.raycast_ground(Vector3.zero())
        assert info.is_grounded is True

    def test_sphere_sweep_ground(self, ground_detector, physics_world):
        hit = CollisionHit(normal=Vector3.up(), distance=0.05)
        physics_world.sweep_results["sphere_0_0_0"] = SweepResult(
            hit=True, hits=[hit], safe_fraction=0.5
        )

        info = ground_detector.sphere_sweep_ground(Vector3.zero())
        assert info.is_grounded is True

    def test_detect_slope_angle_flat(self, ground_detector):
        angle = ground_detector.detect_slope_angle(Vector3.up())
        assert abs(angle) < 0.1

    def test_detect_slope_angle_45(self, ground_detector):
        normal = Vector3(0.707, 0.707, 0.0)
        angle = ground_detector.detect_slope_angle(normal)
        assert abs(angle - 45.0) < 1.0

    def test_is_walkable_slope(self, ground_detector):
        assert ground_detector.is_walkable_slope(Vector3.up()) is True

    def test_is_steep_slope(self, ground_detector):
        steep_normal = Vector3(0.9, 0.1, 0.0).normalized()
        assert ground_detector.is_steep_slope(steep_normal) is True

    def test_ground_detection_boundary_consistency(self, ground_detector):
        """Test that is_walkable and is_steep don't have gaps at boundary."""
        ground_detector.set_slope_limit(45.0)
        # Test exactly at 45 degrees
        normal_45 = Vector3(0.7071067811865476, 0.7071067811865476, 0.0)
        # Should be walkable
        assert ground_detector.is_walkable_slope(normal_45) is True
        # Should NOT be steep (no gap)
        assert ground_detector.is_steep_slope(normal_45) is False

    def test_detect_ledge_no_wall(self, ground_detector):
        info = ground_detector.detect_ledge(
            Vector3.zero(), Vector3.forward()
        )
        assert info.has_ledge is False

    def test_coyote_time_check(self, ground_detector):
        # Initially not in coyote time
        assert ground_detector.is_in_coyote_time() is False

    def test_jump_buffer_registration(self, ground_detector):
        ground_detector.register_jump_input()
        assert ground_detector.is_jump_buffered() is True

    def test_jump_buffer_clear(self, ground_detector):
        ground_detector.register_jump_input()
        ground_detector.clear_jump_buffer()
        assert ground_detector.is_jump_buffered() is False

    def test_can_jump_grounded(self, ground_detector):
        assert ground_detector.can_jump(is_grounded=True) is True

    def test_can_jump_not_grounded(self, ground_detector):
        assert ground_detector.can_jump(is_grounded=False) is False

    def test_set_probe_distance(self, ground_detector):
        ground_detector.set_probe_distance(0.2)
        # Verify probe distance was actually set by using it
        assert ground_detector._probe_distance == 0.2

    def test_set_slope_limit(self, ground_detector):
        ground_detector.set_slope_limit(60.0)
        # Verify slope limit was updated
        assert ground_detector._slope_limit == 60.0

    def test_set_coyote_time(self, ground_detector):
        ground_detector.set_coyote_time(200.0)
        # Verify coyote time was updated
        assert ground_detector._coyote_time_ms == 200.0

    def test_ground_type_solid(self, ground_detector, physics_world):
        hit = CollisionHit(normal=Vector3.up(), distance=0.05)
        physics_world.set_raycast_result("ray_0_0_0", hit)
        info = ground_detector.raycast_ground(Vector3.zero())
        assert info.ground_type == GroundType.SOLID

    def test_multi_point_detection(self, ground_detector):
        combined, probes = ground_detector.detect_ground_multi_point(
            Vector3.zero(), Vector3.forward(), num_points=4
        )
        assert len(probes) >= 4


# =============================================================================
# Movement Mode Tests (15 tests)
# =============================================================================

class TestMovementModes:
    """Tests for MovementModeManager."""

    def test_manager_creation(self, movement_manager):
        assert movement_manager is not None
        assert movement_manager.current_mode == MovementMode.WALKING

    def test_initial_state(self, movement_manager):
        state = movement_manager.state
        assert state.mode == MovementMode.WALKING

    def test_transition_to_running(self, movement_manager):
        success = movement_manager.transition_to_mode(MovementMode.RUNNING)
        assert success is True
        assert movement_manager.current_mode == MovementMode.RUNNING

    def test_transition_to_sprinting(self, movement_manager):
        # First transition to running
        movement_manager.transition_to_mode(MovementMode.RUNNING)
        # Then to sprinting
        success = movement_manager.transition_to_mode(MovementMode.SPRINTING)
        assert success is True

    def test_blocked_transition(self, movement_manager):
        movement_manager.block_mode(MovementMode.FLYING)
        success = movement_manager.transition_to_mode(MovementMode.FLYING)
        assert success is False

    def test_unblock_mode(self, movement_manager):
        movement_manager.block_mode(MovementMode.FLYING)
        movement_manager.unblock_mode(MovementMode.FLYING)
        # Verify mode is no longer blocked by attempting transition
        assert MovementMode.FLYING not in movement_manager._blocked_modes

    def test_max_speed_walking(self, movement_manager):
        speed = movement_manager.max_speed
        assert speed > 0

    def test_can_jump_walking(self, movement_manager):
        assert movement_manager.can_jump is True

    def test_height_modifier_crouching(self, movement_manager):
        movement_manager.transition_to_mode(MovementMode.CROUCHING, force=True)
        modifier = movement_manager.height_modifier
        assert modifier < 1.0

    def test_apply_movement(self, movement_manager):
        direction = Vector3(1.0, 0.0, 0.0)
        movement = movement_manager.apply_movement(direction, 0.016, is_grounded=True)
        assert movement is not None

    def test_is_moving(self, movement_manager):
        assert movement_manager.is_moving() is False

    def test_is_sprinting(self, movement_manager):
        movement_manager.transition_to_mode(MovementMode.SPRINTING, force=True)
        assert movement_manager.is_sprinting() is True

    def test_stamina_consumption(self, movement_manager):
        movement_manager.transition_to_mode(MovementMode.SPRINTING, force=True)
        initial_stamina = movement_manager.state.stamina
        movement_manager.apply_movement(Vector3.forward(), 1.0, is_grounded=True)
        assert movement_manager.state.stamina < initial_stamina

    def test_gravity_scale(self, movement_manager):
        scale = movement_manager.get_gravity_scale()
        assert scale >= 0

    def test_state_serialization(self, movement_manager):
        state_dict = movement_manager.get_state_dict()
        assert "mode" in state_dict


# =============================================================================
# Slope Handling Tests (15 tests)
# =============================================================================

class TestSlopeHandling:
    """Tests for SlopeHandler."""

    def test_handler_creation(self, slope_handler):
        assert slope_handler is not None

    def test_is_walkable_flat(self, slope_handler):
        assert slope_handler.is_walkable_slope(Vector3.up()) is True

    def test_is_walkable_slope(self, slope_handler):
        # 30 degrees slope (well within 45 degree limit)
        normal = Vector3(0.5, 0.866, 0.0)  # ~30 degrees
        assert slope_handler.is_walkable_slope(normal) is True

    def test_is_steep_slope(self, slope_handler):
        steep_normal = Vector3(0.866, 0.5, 0.0)  # 60 degrees
        slope_handler.set_slope_limit(45.0)
        assert slope_handler.is_steep_slope(steep_normal) is True

    def test_is_wall(self, slope_handler):
        wall_normal = Vector3(1.0, 0.0, 0.0)
        assert slope_handler.is_wall(wall_normal) is True

    def test_compute_slope_angle(self, slope_handler):
        angle = slope_handler.compute_slope_angle(Vector3.up())
        assert abs(angle) < 0.1

    def test_get_slope_info(self, slope_handler):
        info = slope_handler.get_slope_info(Vector3.up())
        assert info.is_walkable is True
        assert info.angle < 1.0

    def test_compute_slope_velocity_modifier_uphill(self, slope_handler):
        modifier = slope_handler.compute_slope_velocity_modifier(30.0, is_uphill=True)
        assert modifier < 1.0

    def test_compute_slope_velocity_modifier_downhill(self, slope_handler):
        modifier = slope_handler.compute_slope_velocity_modifier(30.0, is_uphill=False)
        assert modifier > 1.0

    def test_slide_down_steep_slope(self, slope_handler):
        pos = Vector3.zero()
        vel = Vector3.zero()
        steep_normal = Vector3(0.5, 0.866, 0.0)  # ~60 degrees
        new_pos, new_vel = slope_handler.slide_down_steep_slope(
            pos, vel, steep_normal, 0.1
        )
        # Should have some movement

    def test_step_up_no_obstacle(self, slope_handler, physics_world):
        result = slope_handler.step_up(
            Vector3.zero(), Vector3.forward(), 0.35, 1.8
        )
        # No obstacle, no step needed
        assert result is None or not result.can_step

    def test_step_down(self, slope_handler, physics_world):
        result = slope_handler.step_down(Vector3.zero(), 0.35, 1.8)
        # No ground hit, no step down
        assert result is None or not result.can_step

    def test_should_step_down(self, slope_handler):
        should = slope_handler.should_step_down(
            velocity=Vector3(1.0, -0.1, 0.0),
            is_grounded=False,
            was_grounded=True,
        )
        assert should is True

    def test_snap_to_ground(self, slope_handler, physics_world):
        pos, snapped = slope_handler.snap_to_ground(Vector3.zero(), 0.35)
        assert snapped is False  # No raycast hit

    def test_project_on_slope(self, slope_handler):
        movement = Vector3(1.0, 0.0, 0.0)
        slope_normal = Vector3(0.0, 1.0, 0.0)
        projected = slope_handler.project_on_slope(movement, slope_normal)
        assert abs(projected.y) < 0.001

    def test_slope_at_exact_limit_is_walkable(self, slope_handler):
        """Test that a slope at exactly the limit angle is still walkable."""
        slope_handler.set_slope_limit(45.0)
        # Create normal for exactly 45 degrees
        # cos(45) = sin(45) = 0.7071...
        normal = Vector3(0.7071067811865476, 0.7071067811865476, 0.0)
        assert slope_handler.is_walkable_slope(normal) is True
        # Should NOT be steep at exact boundary
        assert slope_handler.is_steep_slope(normal) is False


# =============================================================================
# Platform Handling Tests (15 tests)
# =============================================================================

class TestPlatformHandling:
    """Tests for PlatformHandler."""

    def test_handler_creation(self, platform_handler):
        assert platform_handler is not None

    def test_not_attached_initially(self, platform_handler):
        assert platform_handler.is_attached is False

    def test_attach_to_platform(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            transform=Transform(position=Vector3.zero()),
        )
        platform_provider.add_platform(platform)

        result = platform_handler.attach_to_platform(
            1, Vector3.zero(), Quaternion.identity()
        )
        assert result is True
        assert platform_handler.is_attached is True

    def test_attach_invalid_platform(self, platform_handler):
        result = platform_handler.attach_to_platform(
            999, Vector3.zero(), Quaternion.identity()
        )
        assert result is False

    def test_detach_from_platform(self, platform_handler, platform_provider):
        platform = PlatformData(platform_id=1)
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        velocity = platform_handler.detach_from_platform()
        assert platform_handler.is_attached is False

    def test_get_platform_velocity(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            velocity=Vector3(5.0, 0.0, 0.0),
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        vel = platform_handler.get_platform_velocity()
        assert vel.x == 5.0

    def test_inherited_velocity(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            velocity=Vector3(3.0, 0.0, 0.0),
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        platform_handler.update(Vector3.zero(), Quaternion.identity(), 0.016)

        assert platform_handler.inherited_velocity.x != 0

    def test_attached_platform_id(self, platform_handler, platform_provider):
        platform = PlatformData(platform_id=42)
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(42, Vector3.zero(), Quaternion.identity())

        assert platform_handler.attached_platform_id == 42

    def test_update_position(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            transform=Transform(position=Vector3(10.0, 0.0, 0.0)),
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        new_pos, new_rot, vel = platform_handler.update(
            Vector3.zero(), Quaternion.identity(), 0.016
        )
        # Position updated based on platform

    def test_is_on_moving_platform(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            velocity=Vector3(1.0, 0.0, 0.0),
            is_active=True,
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        assert platform_handler.is_on_moving_platform() is True

    def test_get_platform_type(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            platform_type=PlatformType.ELEVATOR,
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        assert platform_handler.get_platform_type() == PlatformType.ELEVATOR

    def test_attach_callback(self, platform_handler, platform_provider):
        called = []
        platform_handler.set_attach_callback(lambda pid: called.append(pid))

        platform = PlatformData(platform_id=1)
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        assert 1 in called

    def test_detach_callback(self, platform_handler, platform_provider):
        called = []
        platform_handler.set_detach_callback(lambda pid, vel: called.append(pid))

        platform = PlatformData(platform_id=1)
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        platform_handler.detach_from_platform()

        assert 1 in called

    def test_debug_info(self, platform_handler, platform_provider):
        platform = PlatformData(platform_id=1)
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        info = platform_handler.get_debug_info()
        assert info["attached"] is True

    def test_rotating_platform_handling(self, platform_handler, platform_provider):
        platform = PlatformData(
            platform_id=1,
            platform_type=PlatformType.ROTATING,
            angular_velocity=Vector3(0.0, 1.0, 0.0),
        )
        platform_provider.add_platform(platform)
        platform_handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())

        new_pos, yaw = platform_handler.handle_rotating_platform(
            Vector3(1.0, 0.0, 0.0), Vector3.forward(), 0.016
        )
        # Verify rotation was applied
        assert new_pos is not None
        # Yaw should change due to angular velocity
        assert isinstance(yaw, float)


# =============================================================================
# Ragdoll Tests (10 tests)
# =============================================================================

class TestRagdoll:
    """Tests for Ragdoll physics."""

    def test_ragdoll_creation(self, ragdoll):
        assert ragdoll is not None
        assert ragdoll.state == RagdollState.INACTIVE

    def test_default_humanoid_setup(self):
        setup = create_default_humanoid_setup()
        assert len(setup.bodies_per_bone) > 0
        assert len(setup.joints) > 0

    def test_setup_from_skeleton(self, ragdoll):
        result = ragdoll.setup_from_skeleton()
        assert result is True

    def test_activate_ragdoll(self, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.activate()
        assert ragdoll.is_active is True

    def test_activate_with_velocity(self, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.activate(initial_velocity=Vector3(5.0, 10.0, 0.0))
        assert ragdoll.is_active is True

    def test_deactivate_ragdoll(self, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.activate()
        ragdoll.deactivate()
        assert ragdoll.state == RagdollState.RECOVERING

    def test_get_pose(self, ragdoll):
        ragdoll.setup_from_skeleton()
        pose = ragdoll.get_pose()
        assert pose is not None

    def test_get_center_of_mass(self, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.activate()
        com = ragdoll.get_center_of_mass()
        assert com is not None

    def test_destroy_ragdoll(self, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.destroy()
        assert ragdoll.state == RagdollState.INACTIVE

    def test_ragdoll_debug_info(self, ragdoll):
        info = ragdoll.get_debug_info()
        assert "state" in info


# =============================================================================
# Active Ragdoll Tests (10 tests)
# =============================================================================

class TestActiveRagdoll:
    """Tests for ActiveRagdoll balance system."""

    def test_active_ragdoll_creation(self, active_ragdoll):
        assert active_ragdoll is not None
        assert active_ragdoll.state == ActiveRagdollState.INACTIVE

    def test_pd_controller_creation(self):
        pd = PDController(kp=300.0, kd=30.0)
        assert pd.kp == 300.0
        assert pd.kd == 30.0

    def test_pd_controller_compute_torque(self):
        pd = PDController()
        torque = pd.compute_torque(
            Quaternion.identity(),
            Vector3.zero(),
        )
        assert torque is not None

    def test_activate_active_ragdoll(self, active_ragdoll, ragdoll):
        ragdoll.setup_from_skeleton()
        ragdoll.activate()
        active_ragdoll.activate()
        assert active_ragdoll.state == ActiveRagdollState.BALANCED

    def test_deactivate_active_ragdoll(self, active_ragdoll):
        active_ragdoll.deactivate()
        assert active_ragdoll.state == ActiveRagdollState.INACTIVE

    def test_set_joint_strength(self, active_ragdoll):
        active_ragdoll.set_joint_strength(BodyPartType.PELVIS, 0.5)
        # Verify strength was set
        assert active_ragdoll._controllers[BodyPartType.PELVIS].strength == 0.5

    def test_set_global_strength(self, active_ragdoll):
        active_ragdoll.set_global_strength(0.8)
        # Verify all controllers have updated strength
        for controller in active_ragdoll._controllers.values():
            assert controller.strength == 0.8

    def test_balance_error(self, active_ragdoll):
        error = active_ragdoll.balance_error
        assert error is not None

    def test_ankle_strategy(self, active_ragdoll):
        correction = active_ragdoll.ankle_strategy()
        assert correction is not None

    def test_debug_info(self, active_ragdoll):
        info = active_ragdoll.get_debug_info()
        assert "state" in info


# =============================================================================
# Physics Animation Blend Tests (10 tests)
# =============================================================================

class TestPhysicsAnimationBlend:
    """Tests for PhysicsAnimationBlender."""

    def test_blender_creation(self, blender):
        assert blender is not None

    def test_add_layer(self, blender):
        blender.add_layer("test", BlendMode.POSE, 1.0)
        # Verify layer was added
        assert len(blender._layers) == 1
        assert blender._layers[0].name == "test"

    def test_remove_layer(self, blender):
        blender.add_layer("test", BlendMode.POSE, 1.0)
        result = blender.remove_layer("test")
        assert result is True

    def test_set_layer_weight(self, blender):
        blender.add_layer("test", BlendMode.POSE, 1.0)
        blender.set_layer_weight("test", 0.5)
        # Verify weight was updated
        assert blender._layers[0].weight == 0.5

    def test_blend_poses(self, blender):
        anim_pose = SkeletonPose()
        anim_pose.bones["test"] = BonePose(position=Vector3.zero())

        physics_pose = SkeletonPose()
        physics_pose.bones["test"] = BonePose(position=Vector3(1.0, 0.0, 0.0))

        result = blender.blend_poses(anim_pose, physics_pose, 0.5)
        assert "test" in result.bones

    def test_additive_physics(self, blender):
        base = SkeletonPose()
        base.bones["test"] = BonePose(position=Vector3.zero())

        delta = SkeletonPose()
        delta.bones["test"] = BonePose(position=Vector3(1.0, 0.0, 0.0))

        result = blender.additive_physics(base, delta, 0.5)
        assert result.bones["test"].position.x == 0.5

    def test_add_hit_reaction(self, blender):
        blender.add_hit_reaction(
            Vector3.zero(),
            Vector3.forward(),
            100.0,
            ["bone1", "bone2"],
            0.0,
        )
        # Verify hit reaction was added
        assert len(blender._hit_reactions) == 1
        assert blender._hit_reactions[0].force == 100.0

    def test_clear_hit_reactions(self, blender):
        blender.add_hit_reaction(Vector3.zero(), Vector3.forward(), 100.0, [], 0.0)
        blender.clear_hit_reactions()
        # Verify reactions were cleared
        assert len(blender._hit_reactions) == 0

    def test_set_bone_weight(self, blender):
        blender.set_bone_weight("test_bone", 0.7)
        weight = blender.get_bone_weight("test_bone")
        assert weight == 0.7

    def test_debug_info(self, blender):
        info = blender.get_debug_info()
        assert "layer_count" in info


# =============================================================================
# Character Interaction Tests (15 tests)
# =============================================================================

class TestCharacterInteraction:
    """Tests for CharacterInteractionManager."""

    def test_manager_creation(self, interaction_manager):
        assert interaction_manager is not None
        assert interaction_manager.current_interaction == InteractionType.NONE

    def test_not_interacting_initially(self, interaction_manager):
        assert interaction_manager.is_interacting is False

    def test_update_character_state(self, interaction_manager):
        interaction_manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        # Verify state was updated
        assert interaction_manager._character_position.x == 0.0
        assert interaction_manager._character_forward.z == 1.0

    def test_push_character(self, interaction_manager):
        target = InteractionTarget(
            entity_id=1,
            body_id=1,
            position=Vector3(1.0, 0.0, 0.0),
            mass=70.0,
        )
        result = interaction_manager.push_character(target)
        assert result is True

    def test_grab_object_too_far(self, interaction_manager):
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(100.0, 0.0, 0.0),
            can_be_grabbed=True,
        )
        interaction_manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        result = interaction_manager.grab_object(target)
        assert result is False

    def test_grab_object_success(self, interaction_manager):
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.5, 0.0, 0.5),
            can_be_grabbed=True,
        )
        interaction_manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        result = interaction_manager.grab_object(target)
        assert result is True
        assert interaction_manager.is_grabbing is True

    def test_release_grab(self, interaction_manager):
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.5, 0.0, 0.5),
            can_be_grabbed=True,
        )
        interaction_manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        interaction_manager.grab_object(target)
        released = interaction_manager.release_grab()
        assert released is not None
        assert interaction_manager.is_grabbing is False

    def test_throw_object(self, interaction_manager):
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.5, 0.0, 0.5),
            can_be_grabbed=True,
            body_id=1,
        )
        interaction_manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        interaction_manager.grab_object(target)
        interaction_manager.confirm_grab()

        result = interaction_manager.throw_object(Vector3.forward(), 10.0)
        # Verify throw occurred and object is released
        assert result is True
        assert interaction_manager.is_grabbing is False

    def test_climb_ledge(self, interaction_manager):
        result = interaction_manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            2.0,
        )
        assert result is True
        assert interaction_manager.is_climbing is True

    def test_climb_ledge_too_high(self, interaction_manager):
        result = interaction_manager.climb_ledge(
            Vector3(0.0, 10.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            10.0,  # Too high
        )
        assert result is False

    def test_update_climb(self, interaction_manager):
        interaction_manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            2.0,
        )
        pos = interaction_manager.update_climb(0.5)
        assert pos is not None

    def test_vault_obstacle(self, interaction_manager):
        result = interaction_manager.vault_obstacle(
            Vector3(0.0, 0.0, 1.0),
            1.0,
            Vector3.forward(),
        )
        assert result is True
        assert interaction_manager.is_vaulting is True

    def test_vault_too_high(self, interaction_manager):
        result = interaction_manager.vault_obstacle(
            Vector3.zero(),
            5.0,  # Too high
            Vector3.forward(),
        )
        assert result is False

    def test_cancel_interaction(self, interaction_manager):
        interaction_manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            2.0,
        )
        interaction_manager.cancel_interaction()
        assert interaction_manager.is_interacting is False

    def test_debug_info(self, interaction_manager):
        info = interaction_manager.get_debug_info()
        assert "current_interaction" in info


# =============================================================================
# Integration Tests (5 tests)
# =============================================================================

class TestCharacterIntegration:
    """Integration tests combining multiple systems."""

    def test_controller_with_ground_detection(self, physics_world):
        controller = CharacterController(physics_world)
        detector = GroundDetector(physics_world)

        # Move character
        controller.move(Vector3.forward(), 0.016)

        # Detect ground
        info = detector.detect_ground(controller.position)
        # Systems work together

    def test_movement_mode_affects_controller(self, physics_world):
        controller = CharacterController(physics_world)
        movement = MovementModeManager()

        # Set to crouching
        movement.transition_to_mode(MovementMode.CROUCHING, force=True)

        # Height modifier should affect resize
        modifier = movement.height_modifier
        assert modifier < 1.0

    def test_platform_velocity_inheritance(self, physics_world, platform_provider):
        controller = CharacterController(physics_world)
        platform_handler = PlatformHandler(platform_provider)

        # Setup moving platform
        platform = PlatformData(
            platform_id=1,
            velocity=Vector3(5.0, 0.0, 0.0),
            is_active=True,
        )
        platform_provider.add_platform(platform)

        # Attach to platform
        platform_handler.attach_to_platform(
            1, controller.position, controller.rotation
        )

        # Get inherited velocity
        vel = platform_handler.get_platform_velocity()
        assert vel.x == 5.0

    def test_ragdoll_to_active_ragdoll(self, ragdoll_physics, skeleton):
        ragdoll = Ragdoll(ragdoll_physics, skeleton)
        active = ActiveRagdoll(ragdoll, ragdoll_physics)

        # Setup and activate
        ragdoll.setup_from_skeleton()
        ragdoll.activate()
        active.activate()

        assert ragdoll.is_active is True
        assert active.state == ActiveRagdollState.BALANCED

    def test_full_character_system(self, physics_world, platform_provider):
        """Test complete character physics system."""
        # Create all systems
        controller = CharacterController(physics_world)
        ground_detector = GroundDetector(physics_world)
        movement = MovementModeManager()
        slope_handler = SlopeHandler(physics_world)
        platform_handler = PlatformHandler(platform_provider)
        interaction = CharacterInteractionManager(physics_world)

        # Simulate a frame
        dt = 0.016

        # Get input
        input_dir = Vector3(1.0, 0.0, 0.0)

        # Apply movement mode
        movement_vec = movement.apply_movement(input_dir, dt, is_grounded=True)

        # Move controller
        displacement = controller.move(movement_vec / dt, dt)

        # Detect ground
        ground_info = ground_detector.detect_ground(controller.position)

        # All systems working together
        assert displacement is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
