"""
Comprehensive tests for MovementComponent.

Tests cover:
- Movement modes (walking, running, flying, swimming)
- Speed modifiers
- Acceleration/deceleration
- Gravity application
- Jump mechanics
- Ground detection
- Slope handling
- Movement input processing
- Movement prediction
- Root motion
"""

import math
import pytest
from typing import List

from engine.core.math.vec import Vec3
from engine.gameplay.components.movement import (
    MovementComponent,
    MovementMode,
    MovementState,
    MovementSettings,
    MovementSnapshot,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def movement():
    """Create a default movement component."""
    return MovementComponent()


@pytest.fixture
def walking_movement():
    """Create a movement component in walking mode."""
    return MovementComponent(max_speed=5.0, movement_mode=MovementMode.WALKING)


@pytest.fixture
def running_movement():
    """Create a movement component in running mode."""
    return MovementComponent(movement_mode=MovementMode.RUNNING)


@pytest.fixture
def flying_movement():
    """Create a movement component in flying mode."""
    return MovementComponent(movement_mode=MovementMode.FLYING)


@pytest.fixture
def swimming_movement():
    """Create a movement component in swimming mode."""
    return MovementComponent(movement_mode=MovementMode.SWIMMING)


@pytest.fixture
def grounded_movement():
    """Create a grounded movement component with velocity."""
    m = MovementComponent()
    m.set_grounded(True)
    m.velocity = Vec3(5, 0, 0)
    return m


@pytest.fixture
def airborne_movement():
    """Create an airborne movement component."""
    m = MovementComponent()
    m.set_grounded(False)
    m.velocity = Vec3(5, 10, 0)  # Moving with upward velocity
    return m


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestMovementInitialization:
    """Tests for MovementComponent initialization."""

    def test_default_initialization(self, movement):
        """Test default movement values."""
        assert movement.velocity == Vec3.zero()
        assert movement.input_direction == Vec3.zero()
        assert movement.movement_mode == MovementMode.WALKING
        assert movement.movement_state == MovementState.IDLE

    def test_initialization_with_max_speed(self):
        """Test initialization with custom max speed."""
        m = MovementComponent(max_speed=10.0)
        assert m.max_speed == 10.0

    def test_initialization_with_mode(self):
        """Test initialization with movement mode."""
        m = MovementComponent(movement_mode=MovementMode.FLYING)
        assert m.movement_mode == MovementMode.FLYING

    def test_initialization_with_entity_id(self):
        """Test initialization with entity ID."""
        m = MovementComponent(entity_id="entity_123")
        assert m._entity_id == "entity_123"

    def test_initial_grounded_state(self, movement):
        """Test initial grounded state."""
        assert movement.is_grounded is True

    def test_initial_jumps_remaining(self, movement):
        """Test initial jumps remaining."""
        assert movement.jumps_remaining == 1

    def test_initial_facing_direction(self, movement):
        """Test initial facing direction."""
        # Should be forward direction
        assert movement.facing_direction == Vec3.forward()

    def test_initial_movement_enabled(self, movement):
        """Test movement is initially enabled."""
        assert movement.movement_enabled is True


# =============================================================================
# MOVEMENT MODE TESTS
# =============================================================================


class TestMovementModes:
    """Tests for movement mode system."""

    def test_walking_mode(self, walking_movement):
        """Test walking mode settings."""
        assert walking_movement.movement_mode == MovementMode.WALKING
        settings = walking_movement.current_settings
        assert settings.can_jump is True

    def test_running_mode(self, running_movement):
        """Test running mode settings."""
        assert running_movement.movement_mode == MovementMode.RUNNING
        assert running_movement.max_speed > 4.0  # Faster than walking

    def test_sprinting_mode(self):
        """Test sprinting mode settings."""
        m = MovementComponent(movement_mode=MovementMode.SPRINTING)
        assert m.movement_mode == MovementMode.SPRINTING
        assert m.max_speed > 7.0  # Faster than running

    def test_crouching_mode(self):
        """Test crouching mode settings."""
        m = MovementComponent(movement_mode=MovementMode.CROUCHING)
        assert m.movement_mode == MovementMode.CROUCHING
        assert m.max_speed < 4.0  # Slower than walking
        assert m.current_settings.can_jump is False

    def test_swimming_mode(self, swimming_movement):
        """Test swimming mode settings."""
        assert swimming_movement.movement_mode == MovementMode.SWIMMING
        assert swimming_movement.current_settings.gravity_scale < 1.0

    def test_flying_mode(self, flying_movement):
        """Test flying mode settings."""
        assert flying_movement.movement_mode == MovementMode.FLYING
        assert flying_movement.current_settings.gravity_scale == 0.0
        assert flying_movement.current_settings.air_control == 1.0

    def test_falling_mode(self):
        """Test falling mode settings."""
        m = MovementComponent(movement_mode=MovementMode.FALLING)
        assert m.movement_mode == MovementMode.FALLING
        assert m.current_settings.acceleration == 0.0

    def test_climbing_mode(self):
        """Test climbing mode settings."""
        m = MovementComponent(movement_mode=MovementMode.CLIMBING)
        assert m.movement_mode == MovementMode.CLIMBING
        assert m.current_settings.gravity_scale == 0.0

    def test_sliding_mode(self):
        """Test sliding mode settings."""
        m = MovementComponent(movement_mode=MovementMode.SLIDING)
        assert m.movement_mode == MovementMode.SLIDING
        assert m.current_settings.deceleration < m.current_settings.acceleration

    def test_custom_mode(self):
        """Test custom mode settings."""
        m = MovementComponent(movement_mode=MovementMode.CUSTOM)
        assert m.movement_mode == MovementMode.CUSTOM

    def test_change_movement_mode(self, movement):
        """Test changing movement mode."""
        movement.movement_mode = MovementMode.RUNNING
        assert movement.movement_mode == MovementMode.RUNNING

    def test_mode_change_callback(self, movement):
        """Test mode change callback."""
        changes = []
        movement.on_mode_changed(lambda old, new: changes.append((old, new)))
        movement.movement_mode = MovementMode.RUNNING
        assert len(changes) == 1
        assert changes[0] == (MovementMode.WALKING, MovementMode.RUNNING)

    def test_mode_change_same_mode_no_callback(self, movement):
        """Test setting same mode doesn't trigger callback."""
        changes = []
        movement.on_mode_changed(lambda old, new: changes.append((old, new)))
        movement.movement_mode = MovementMode.WALKING  # Same as current
        assert len(changes) == 0

    def test_get_mode_settings(self, movement):
        """Test getting settings for specific mode."""
        settings = movement.get_mode_settings(MovementMode.RUNNING)
        assert settings.max_speed > 0

    def test_set_mode_settings(self, movement):
        """Test setting custom mode settings."""
        custom_settings = MovementSettings(max_speed=15.0, acceleration=30.0)
        movement.set_mode_settings(MovementMode.WALKING, custom_settings)
        assert movement.get_mode_settings(MovementMode.WALKING).max_speed == 15.0


# =============================================================================
# MOVEMENT STATE TESTS
# =============================================================================


class TestMovementState:
    """Tests for movement state system."""

    def test_idle_state(self, movement):
        """Test idle state."""
        assert movement.movement_state == MovementState.IDLE

    def test_moving_state(self, movement):
        """Test moving state."""
        movement.movement_state = MovementState.MOVING
        assert movement.movement_state == MovementState.MOVING

    def test_jumping_state(self, movement):
        """Test jumping state."""
        movement.movement_state = MovementState.JUMPING
        assert movement.movement_state == MovementState.JUMPING

    def test_airborne_state(self, movement):
        """Test airborne state."""
        movement.movement_state = MovementState.AIRBORNE
        assert movement.movement_state == MovementState.AIRBORNE

    def test_landed_state(self, movement):
        """Test landed state."""
        movement.movement_state = MovementState.LANDED
        assert movement.movement_state == MovementState.LANDED

    def test_disabled_state(self, movement):
        """Test disabled state."""
        movement.movement_state = MovementState.DISABLED
        assert movement.movement_state == MovementState.DISABLED

    def test_state_change_callback(self, movement):
        """Test state change callback."""
        changes = []
        movement.on_state_changed(lambda old, new: changes.append((old, new)))
        movement.movement_state = MovementState.MOVING
        assert len(changes) == 1
        assert changes[0] == (MovementState.IDLE, MovementState.MOVING)

    def test_state_change_same_state_no_callback(self, movement):
        """Test setting same state doesn't trigger callback."""
        changes = []
        movement.on_state_changed(lambda old, new: changes.append((old, new)))
        movement.movement_state = MovementState.IDLE  # Same as current
        assert len(changes) == 0


# =============================================================================
# VELOCITY AND SPEED TESTS
# =============================================================================


class TestVelocityAndSpeed:
    """Tests for velocity and speed properties."""

    def test_set_velocity(self, movement):
        """Test setting velocity."""
        movement.set_velocity(Vec3(5, 0, 0))
        assert movement.velocity == Vec3(5, 0, 0)

    def test_add_velocity(self, movement):
        """Test adding velocity."""
        movement.velocity = Vec3(5, 0, 0)
        movement.add_velocity(Vec3(0, 5, 0))
        assert movement.velocity == Vec3(5, 5, 0)

    def test_add_impulse(self, movement):
        """Test adding impulse."""
        movement.add_impulse(Vec3(10, 10, 0))
        assert movement.velocity == Vec3(10, 10, 0)

    def test_speed_horizontal(self, grounded_movement):
        """Test horizontal speed calculation."""
        grounded_movement.velocity = Vec3(3, 0, 4)
        assert grounded_movement.speed == 5.0  # sqrt(9 + 16)

    def test_speed_3d(self, movement):
        """Test 3D speed calculation."""
        movement.velocity = Vec3(2, 2, 1)
        assert movement.speed_3d == 3.0  # sqrt(4 + 4 + 1)

    def test_max_speed_walking(self, walking_movement):
        """Test max speed for walking mode."""
        assert walking_movement.max_speed == 5.0

    def test_max_speed_with_multiplier(self, walking_movement):
        """Test max speed with speed multiplier."""
        walking_movement.speed_multiplier = 2.0
        assert walking_movement.max_speed == 10.0

    def test_speed_percentage(self, walking_movement):
        """Test speed percentage calculation."""
        walking_movement.velocity = Vec3(2.5, 0, 0)
        assert walking_movement.speed_percentage == 0.5

    def test_speed_percentage_over_max(self, walking_movement):
        """Test speed percentage can exceed 1.0."""
        walking_movement.velocity = Vec3(10, 0, 0)
        assert walking_movement.speed_percentage == 2.0

    def test_is_moving_true(self, grounded_movement):
        """Test is_moving when moving."""
        assert grounded_movement.is_moving is True

    def test_is_moving_false(self, movement):
        """Test is_moving when stationary."""
        assert movement.is_moving is False

    def test_horizontal_velocity(self, movement):
        """Test horizontal velocity extraction."""
        movement.velocity = Vec3(3, 10, 4)
        horizontal = movement.horizontal_velocity
        assert horizontal == Vec3(3, 0, 4)

    def test_vertical_velocity(self, movement):
        """Test vertical velocity extraction."""
        movement.velocity = Vec3(3, 10, 4)
        assert movement.vertical_velocity == 10


# =============================================================================
# SPEED MULTIPLIER TESTS
# =============================================================================


class TestSpeedMultiplier:
    """Tests for speed multiplier system."""

    def test_default_speed_multiplier(self, movement):
        """Test default speed multiplier."""
        assert movement.speed_multiplier == 1.0

    def test_set_speed_multiplier(self, movement):
        """Test setting speed multiplier."""
        movement.speed_multiplier = 1.5
        assert movement.speed_multiplier == 1.5

    def test_speed_multiplier_affects_max_speed(self, walking_movement):
        """Test speed multiplier affects max speed."""
        base_speed = walking_movement.max_speed
        walking_movement.speed_multiplier = 2.0
        assert walking_movement.max_speed == base_speed * 2.0

    def test_speed_multiplier_zero(self, walking_movement):
        """Test zero speed multiplier."""
        walking_movement.speed_multiplier = 0.0
        assert walking_movement.max_speed == 0.0

    def test_speed_multiplier_cannot_be_negative(self, movement):
        """Test speed multiplier cannot be negative."""
        movement.speed_multiplier = -1.0
        assert movement.speed_multiplier == 0.0


# =============================================================================
# GROUND STATE TESTS
# =============================================================================


class TestGroundState:
    """Tests for ground detection system."""

    def test_is_grounded_default(self, movement):
        """Test default grounded state."""
        assert movement.is_grounded is True

    def test_set_grounded_false(self, movement):
        """Test setting grounded to false."""
        movement.set_grounded(False)
        assert movement.is_grounded is False

    def test_ground_normal_default(self, movement):
        """Test default ground normal."""
        assert movement.ground_normal == Vec3.up()

    def test_ground_normal_custom(self, movement):
        """Test custom ground normal."""
        normal = Vec3(0.5, 0.866, 0).normalized()
        movement.set_grounded(True, normal=normal)
        assert movement.ground_normal.y > 0.8

    def test_landing_callback(self, movement):
        """Test landing callback."""
        landed = [False]
        movement.on_landed(lambda m: landed.__setitem__(0, True))
        movement.set_grounded(False)
        movement.set_grounded(True)
        assert landed[0] is True

    def test_landing_sets_state(self, movement):
        """Test landing sets LANDED state."""
        movement.set_grounded(False)
        movement.set_grounded(True)
        assert movement.movement_state == MovementState.LANDED

    def test_leaving_ground_sets_airborne(self, movement):
        """Test leaving ground sets AIRBORNE state."""
        movement.set_grounded(False)
        assert movement.movement_state == MovementState.AIRBORNE

    def test_jumps_reset_on_landing(self, movement):
        """Test jumps remaining reset on landing."""
        movement._jumps_remaining = 0
        movement.set_grounded(False)
        movement.set_grounded(True)
        assert movement.jumps_remaining == movement.max_jumps


# =============================================================================
# JUMP MECHANICS TESTS
# =============================================================================


class TestJumpMechanics:
    """Tests for jump mechanics."""

    def test_can_jump_on_ground(self, movement):
        """Test can jump when grounded."""
        assert movement.can_jump is True

    def test_can_jump_in_air_no_jumps(self, movement):
        """Test cannot jump in air with no jumps remaining."""
        movement.set_grounded(False)
        movement._jumps_remaining = 0
        assert movement.can_jump is False

    def test_can_jump_crouching_disabled(self):
        """Test cannot jump while crouching."""
        m = MovementComponent(movement_mode=MovementMode.CROUCHING)
        assert m.can_jump is False

    def test_request_jump_on_ground(self, movement):
        """Test requesting jump on ground."""
        success = movement.request_jump()
        assert success is True
        assert movement.movement_state == MovementState.JUMPING
        assert movement.velocity.y > 0

    def test_request_jump_in_air_fails(self, movement):
        """Test requesting jump in air without jumps."""
        movement.set_grounded(False)
        movement._jumps_remaining = 0
        success = movement.request_jump()
        assert success is False

    def test_jumps_remaining_decremented(self, movement):
        """Test jumps remaining is decremented."""
        initial_jumps = movement.jumps_remaining
        movement.request_jump()
        assert movement.jumps_remaining == initial_jumps - 1

    def test_jump_callback(self, movement):
        """Test jump callback."""
        jumped = [False]
        movement.on_jumped(lambda m: jumped.__setitem__(0, True))
        movement.request_jump()
        assert jumped[0] is True

    def test_max_jumps_setting(self, movement):
        """Test max jumps setting."""
        movement.max_jumps = 3
        assert movement.max_jumps == 3

    def test_double_jump(self, movement):
        """Test double jump."""
        movement.max_jumps = 2
        movement._jumps_remaining = 2
        movement.request_jump()
        movement.set_grounded(False)
        assert movement.jumps_remaining == 1
        success = movement.request_jump()
        assert success is True

    def test_cancel_jump(self, movement):
        """Test canceling a jump."""
        movement.request_jump()
        initial_y_vel = movement.velocity.y
        movement.cancel_jump()
        assert movement.velocity.y == initial_y_vel * 0.5

    def test_cancel_jump_only_when_rising(self, movement):
        """Test cancel jump only works when rising."""
        movement.request_jump()
        movement.velocity = Vec3(0, -5, 0)  # Falling
        initial_y_vel = movement.velocity.y
        movement.cancel_jump()
        assert movement.velocity.y == initial_y_vel  # Unchanged


# =============================================================================
# COYOTE TIME TESTS
# =============================================================================


class TestCoyoteTime:
    """Tests for coyote time (grace period for jumping after leaving ground)."""

    def test_coyote_time_allows_jump(self, movement):
        """Test coyote time allows jumping after leaving ground."""
        movement.set_grounded(True, current_time=0.0)
        movement.set_grounded(False, current_time=0.1)
        # Within coyote time (0.15s default)
        assert movement.can_use_coyote_time(0.2) is True

    def test_coyote_time_expired(self, movement):
        """Test coyote time expires."""
        movement.set_grounded(True, current_time=0.0)
        movement.set_grounded(False, current_time=0.1)
        # After coyote time
        assert movement.can_use_coyote_time(0.5) is False

    def test_jump_with_coyote_time(self, movement):
        """Test jumping with coyote time."""
        movement.set_grounded(True, current_time=0.0)
        movement.set_grounded(False, current_time=0.1)
        success = movement.request_jump(current_time=0.15)
        assert success is True


# =============================================================================
# JUMP BUFFERING TESTS
# =============================================================================


class TestJumpBuffering:
    """Tests for jump buffering (pre-landing jump requests)."""

    def test_buffered_jump_executes_on_landing(self, movement):
        """Test buffered jump executes on landing."""
        movement.set_grounded(False, current_time=0.0)
        movement._jumps_remaining = 0
        movement.request_jump(current_time=0.05)  # Buffer jump
        movement.set_grounded(True, current_time=0.1)  # Land within buffer time
        assert movement.movement_state == MovementState.JUMPING

    def test_expired_buffer_ignored(self, movement):
        """Test expired buffer is ignored."""
        movement.set_grounded(False, current_time=0.0)
        movement._jumps_remaining = 0
        movement.request_jump(current_time=0.01)  # Buffer jump
        movement.set_grounded(True, current_time=0.5)  # Land after buffer expires
        assert movement.movement_state != MovementState.JUMPING


# =============================================================================
# INPUT PROCESSING TESTS
# =============================================================================


class TestInputProcessing:
    """Tests for movement input processing."""

    def test_set_input_direction(self, movement):
        """Test setting input direction."""
        movement.set_input_direction(Vec3(1, 0, 0))
        assert movement.input_direction == Vec3(1, 0, 0)

    def test_input_normalized(self, movement):
        """Test input is normalized if too large."""
        movement.set_input_direction(Vec3(10, 0, 0))
        assert movement.input_direction.length() == pytest.approx(1.0, abs=0.01)

    def test_clear_input(self, movement):
        """Test clearing input."""
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.clear_input()
        assert movement.input_direction == Vec3.zero()

    def test_has_input_true(self, movement):
        """Test has_input when there is input."""
        movement.set_input_direction(Vec3(1, 0, 0))
        assert movement.has_input is True

    def test_has_input_false(self, movement):
        """Test has_input when no input."""
        assert movement.has_input is False

    def test_facing_direction(self, movement):
        """Test facing direction."""
        movement.facing_direction = Vec3(1, 0, 0)
        assert movement.facing_direction.x > 0.9

    def test_facing_direction_normalized(self, movement):
        """Test facing direction is normalized."""
        movement.facing_direction = Vec3(10, 0, 0)
        assert movement.facing_direction.length() == pytest.approx(1.0, abs=0.01)

    def test_facing_direction_zero_rejected(self, movement):
        """Test zero facing direction is rejected."""
        initial_facing = movement.facing_direction
        movement.facing_direction = Vec3.zero()
        assert movement.facing_direction == initial_facing


# =============================================================================
# MOVEMENT UPDATE TESTS
# =============================================================================


class TestMovementUpdate:
    """Tests for movement update processing."""

    def test_update_idle_to_moving(self, movement):
        """Test update transitions from idle to moving."""
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.update(0.1)
        assert movement.movement_state == MovementState.MOVING

    def test_update_moving_to_idle(self, movement):
        """Test update transitions from moving to idle."""
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.update(1.0)  # Accelerate
        movement.clear_input()
        movement.update(10.0)  # Decelerate to stop
        assert movement.movement_state == MovementState.IDLE

    def test_update_accelerates_toward_input(self, walking_movement):
        """Test update accelerates toward input direction."""
        walking_movement.set_input_direction(Vec3(1, 0, 0))
        walking_movement.update(0.5)
        assert walking_movement.velocity.x > 0

    def test_update_decelerates_without_input(self, grounded_movement):
        """Test update decelerates without input."""
        initial_speed = grounded_movement.speed
        grounded_movement.update(0.5)
        assert grounded_movement.speed < initial_speed

    def test_update_respects_max_speed(self, walking_movement):
        """Test update respects max speed."""
        walking_movement.set_input_direction(Vec3(1, 0, 0))
        walking_movement.update(10.0)  # Long time to reach max
        assert walking_movement.speed <= walking_movement.max_speed + 0.1

    def test_update_air_control(self, airborne_movement):
        """Test update uses air control in air."""
        airborne_movement.set_grounded(False)
        airborne_movement.set_input_direction(Vec3(0, 0, 1))
        initial_vel = Vec3(airborne_movement.velocity.x, 0, airborne_movement.velocity.z)
        airborne_movement.update(0.1)
        # Air control is reduced, so change should be smaller
        settings = airborne_movement.current_settings
        assert settings.air_control < 1.0

    def test_update_disabled_does_nothing(self, movement):
        """Test update does nothing when disabled."""
        movement.movement_enabled = False
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.update(1.0)
        assert movement.velocity == Vec3.zero()

    def test_update_updates_facing_direction(self, movement):
        """Test update updates facing direction."""
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.update(1.0)
        assert movement.facing_direction.x > 0


# =============================================================================
# GRAVITY TESTS
# =============================================================================


class TestGravity:
    """Tests for gravity application."""

    def test_apply_gravity_in_air(self, airborne_movement):
        """Test gravity applies when in air."""
        initial_y_vel = airborne_movement.velocity.y
        airborne_movement.set_grounded(False)
        airborne_movement.apply_gravity(10.0, 0.5)
        assert airborne_movement.velocity.y < initial_y_vel

    def test_no_gravity_on_ground(self, grounded_movement):
        """Test gravity doesn't apply on ground."""
        initial_y_vel = grounded_movement.velocity.y
        grounded_movement.apply_gravity(10.0, 0.5)
        assert grounded_movement.velocity.y == initial_y_vel

    def test_gravity_scale(self, movement):
        """Test gravity scale affects gravity."""
        movement.set_grounded(False)
        movement.velocity = Vec3(0, 0, 0)

        # Swimming has reduced gravity
        movement.movement_mode = MovementMode.SWIMMING
        movement.apply_gravity(10.0, 1.0)
        swimming_vel = movement.velocity.y

        # Reset
        movement.velocity = Vec3(0, 0, 0)
        movement.movement_mode = MovementMode.WALKING
        movement.apply_gravity(10.0, 1.0)
        walking_vel = movement.velocity.y

        assert swimming_vel > walking_vel  # Less negative

    def test_gravity_flying_no_effect(self, flying_movement):
        """Test gravity has no effect in flying mode."""
        flying_movement.set_grounded(False)
        flying_movement.velocity = Vec3(0, 0, 0)
        flying_movement.apply_gravity(10.0, 1.0)
        assert flying_movement.velocity.y == 0


# =============================================================================
# MOVEMENT CONTROL TESTS
# =============================================================================


class TestMovementControl:
    """Tests for movement control (enable/disable, stop, freeze)."""

    def test_disable_movement(self, movement):
        """Test disabling movement."""
        movement.movement_enabled = False
        assert movement.movement_enabled is False
        assert movement.movement_state == MovementState.DISABLED

    def test_enable_movement(self, movement):
        """Test enabling movement."""
        movement.movement_enabled = False
        movement.movement_enabled = True
        assert movement.movement_enabled is True

    def test_cannot_jump_when_disabled(self, movement):
        """Test cannot jump when movement disabled."""
        movement.movement_enabled = False
        assert movement.can_jump is False

    def test_stop(self, grounded_movement):
        """Test stop method."""
        grounded_movement.set_input_direction(Vec3(1, 0, 0))
        grounded_movement.stop()
        assert grounded_movement.velocity == Vec3.zero()
        assert grounded_movement.input_direction == Vec3.zero()
        assert grounded_movement.movement_state == MovementState.IDLE

    def test_freeze(self, grounded_movement):
        """Test freeze method."""
        grounded_movement.freeze()
        assert grounded_movement.movement_enabled is False
        assert grounded_movement.velocity == Vec3.zero()

    def test_unfreeze(self, movement):
        """Test unfreeze method."""
        movement.freeze()
        movement.unfreeze()
        assert movement.movement_enabled is True
        assert movement.movement_state == MovementState.IDLE


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================


class TestSnapshots:
    """Tests for movement snapshots."""

    def test_create_snapshot(self, movement):
        """Test creating a snapshot."""
        movement.velocity = Vec3(5, 0, 0)
        movement.movement_mode = MovementMode.RUNNING
        snapshot = movement.create_snapshot(Vec3(10, 20, 30), timestamp=1.0)
        assert snapshot.position == Vec3(10, 20, 30)
        assert snapshot.velocity == Vec3(5, 0, 0)
        assert snapshot.mode == MovementMode.RUNNING
        assert snapshot.timestamp == 1.0

    def test_apply_snapshot(self, movement):
        """Test applying a snapshot."""
        snapshot = MovementSnapshot(
            position=Vec3(10, 20, 30),
            velocity=Vec3(5, 5, 5),
            mode=MovementMode.FLYING,
            state=MovementState.MOVING,
            timestamp=0.5,
        )
        movement.apply_snapshot(snapshot)
        assert movement.velocity == Vec3(5, 5, 5)
        assert movement.movement_mode == MovementMode.FLYING
        assert movement.movement_state == MovementState.MOVING


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for movement component serialization."""

    def test_to_dict(self, movement):
        """Test serialization to dictionary."""
        movement.velocity = Vec3(1, 2, 3)
        movement.movement_mode = MovementMode.RUNNING
        data = movement.to_dict()
        assert "velocity" in data
        assert "movement_mode" in data
        assert "movement_state" in data
        assert "is_grounded" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "velocity": [5, 0, 0],
            "input_direction": [1, 0, 0],
            "movement_mode": "RUNNING",
            "movement_state": "MOVING",
            "is_grounded": True,
            "jumps_remaining": 2,
            "max_jumps": 2,
            "speed_multiplier": 1.5,
            "movement_enabled": True,
            "facing_direction": [0, 0, -1],
        }
        m = MovementComponent.from_dict(data)
        assert m.velocity == Vec3(5, 0, 0)
        assert m.movement_mode == MovementMode.RUNNING
        assert m.jumps_remaining == 2
        assert m.speed_multiplier == 1.5

    def test_round_trip(self, movement):
        """Test serialization round trip."""
        movement.velocity = Vec3(3, 4, 5)
        movement.movement_mode = MovementMode.SWIMMING
        movement.max_jumps = 3
        data = movement.to_dict()
        restored = MovementComponent.from_dict(data)
        assert restored.velocity == movement.velocity
        assert restored.movement_mode == movement.movement_mode
        assert restored.max_jumps == movement.max_jumps

    def test_repr(self, movement):
        """Test string representation."""
        rep = repr(movement)
        assert "MovementComponent" in rep
        assert "WALKING" in rep


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for movement callbacks."""

    def test_multiple_mode_callbacks(self, movement):
        """Test multiple mode change callbacks."""
        count = [0]
        movement.on_mode_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        movement.on_mode_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        movement.movement_mode = MovementMode.RUNNING
        assert count[0] == 2

    def test_multiple_state_callbacks(self, movement):
        """Test multiple state change callbacks."""
        count = [0]
        movement.on_state_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        movement.on_state_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        movement.movement_state = MovementState.MOVING
        assert count[0] == 2

    def test_multiple_landed_callbacks(self, movement):
        """Test multiple landed callbacks."""
        count = [0]
        movement.on_landed(lambda m: count.__setitem__(0, count[0] + 1))
        movement.on_landed(lambda m: count.__setitem__(0, count[0] + 1))
        movement.set_grounded(False)
        movement.set_grounded(True)
        assert count[0] == 2

    def test_multiple_jumped_callbacks(self, movement):
        """Test multiple jumped callbacks."""
        count = [0]
        movement.on_jumped(lambda m: count.__setitem__(0, count[0] + 1))
        movement.on_jumped(lambda m: count.__setitem__(0, count[0] + 1))
        movement.request_jump()
        assert count[0] == 2


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_high_velocity(self, movement):
        """Test very high velocity values."""
        movement.velocity = Vec3(1e10, 1e10, 1e10)
        assert movement.speed_3d > 0

    def test_very_small_delta_time(self, movement):
        """Test update with very small delta time."""
        movement.set_input_direction(Vec3(1, 0, 0))
        movement.update(0.0001)
        # Should not crash

    def test_zero_delta_time(self, movement):
        """Test update with zero delta time."""
        movement.set_input_direction(Vec3(1, 0, 0))
        initial_vel = Vec3(movement.velocity.x, movement.velocity.y, movement.velocity.z)
        movement.update(0.0)
        # Should not change significantly

    def test_large_delta_time(self, walking_movement):
        """Test update with large delta time."""
        walking_movement.set_input_direction(Vec3(1, 0, 0))
        walking_movement.update(100.0)
        # Should reach max speed without issues
        assert walking_movement.speed <= walking_movement.max_speed + 0.1

    def test_diagonal_input(self, walking_movement):
        """Test diagonal input normalization."""
        walking_movement.set_input_direction(Vec3(1, 0, 1))
        walking_movement.update(10.0)
        # Speed should not exceed max
        assert walking_movement.speed <= walking_movement.max_speed + 0.1

    def test_rapid_mode_switching(self, movement):
        """Test rapid movement mode switching."""
        for _ in range(100):
            movement.movement_mode = MovementMode.WALKING
            movement.movement_mode = MovementMode.RUNNING
            movement.movement_mode = MovementMode.FLYING
        # Should not crash

    def test_rapid_jump_requests(self, movement):
        """Test rapid jump requests."""
        movement.max_jumps = 10
        movement._jumps_remaining = 10
        for _ in range(10):
            movement.request_jump()
        assert movement.jumps_remaining == 0

    def test_movement_with_negative_velocity(self, movement):
        """Test movement with negative velocity."""
        movement.velocity = Vec3(-5, -10, -5)
        assert movement.speed > 0  # Speed should be positive
        assert movement.is_moving is True
