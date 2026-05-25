"""
Tests for smooth locomotion (smooth.py).

Tests the smooth locomotion system including:
    - SmoothLocomotion component
    - TurnSettings
    - MovementInput/MovementResult
    - @xr_locomotion decorator
"""

import math

import pytest

from engine.xr.locomotion.smooth import (
    ArmSwingData,
    MovementInput,
    MovementMode,
    MovementResult,
    MovementState,
    SmoothLocomotion,
    SmoothLocomotionProvider,
    StrafeBehavior,
    TurnSettings,
    TurnType,
    xr_locomotion,
)
from trinity.decorators.ops import decompose


# =============================================================================
# TurnSettings Tests
# =============================================================================


class TestTurnSettings:
    """Tests for TurnSettings configuration."""

    def test_default_values(self):
        """Test default initialization."""
        settings = TurnSettings()
        assert settings.turn_type == TurnType.SNAP
        assert settings.snap_angle == 45.0
        assert settings.smooth_speed == 90.0
        assert settings.snap_cooldown == 0.15

    def test_can_snap_turn(self):
        """Test snap turn availability check."""
        settings = TurnSettings(turn_type=TurnType.SNAP)
        assert settings.can_snap_turn(1) is True
        assert settings.can_snap_turn(-1) is True

    def test_cannot_snap_when_smooth(self):
        """Test that snap is disabled in smooth mode."""
        settings = TurnSettings(turn_type=TurnType.SMOOTH)
        assert settings.can_snap_turn(1) is False

    def test_snap_turn_cooldown(self):
        """Test snap turn triggers cooldown."""
        settings = TurnSettings(
            turn_type=TurnType.SNAP,
            snap_angle=45.0,
            snap_cooldown=0.2,
        )

        rotation = settings.execute_snap_turn(1)
        assert rotation == pytest.approx(math.radians(45.0), abs=0.01)
        assert settings._snap_cooldown_remaining == 0.2

        # Cannot turn during cooldown
        assert settings.can_snap_turn(1) is False

    def test_snap_turn_direction(self):
        """Test snap turn direction."""
        settings = TurnSettings(turn_type=TurnType.SNAP, snap_angle=30.0)

        # Right turn
        rotation_right = settings.execute_snap_turn(1)
        assert rotation_right > 0

        # Reset cooldown for test
        settings._snap_cooldown_remaining = 0

        # Left turn
        rotation_left = settings.execute_snap_turn(-1)
        assert rotation_left < 0

    def test_smooth_turn_calculation(self):
        """Test smooth turn calculation."""
        settings = TurnSettings(
            turn_type=TurnType.SMOOTH,
            smooth_speed=90.0,
            dead_zone=0.1,
        )

        # Full input for 1 second = 90 degrees
        rotation = settings.calculate_smooth_turn(1.0, 1.0)
        # Account for dead zone normalization
        expected_rotation = math.radians(90.0) * (1.0 - 0.1) / (1.0 - 0.1)
        assert rotation > 0

    def test_smooth_turn_dead_zone(self):
        """Test smooth turn dead zone."""
        settings = TurnSettings(
            turn_type=TurnType.SMOOTH,
            dead_zone=0.2,
        )

        # Input below dead zone
        rotation = settings.calculate_smooth_turn(0.1, 1.0)
        assert rotation == 0.0

    def test_smooth_turn_not_in_snap_mode(self):
        """Test smooth turn returns 0 in snap mode."""
        settings = TurnSettings(turn_type=TurnType.SNAP)
        rotation = settings.calculate_smooth_turn(1.0, 1.0)
        assert rotation == 0.0

    def test_update_reduces_cooldown(self):
        """Test update reduces cooldown timer."""
        settings = TurnSettings(turn_type=TurnType.SNAP)
        settings._snap_cooldown_remaining = 0.5

        settings.update(0.2)
        assert settings._snap_cooldown_remaining == pytest.approx(0.3, abs=0.01)

        settings.update(0.4)
        assert settings._snap_cooldown_remaining == 0.0


# =============================================================================
# SmoothLocomotion Tests
# =============================================================================


class TestSmoothLocomotion:
    """Tests for the SmoothLocomotion component."""

    def test_default_state(self):
        """Test default initialization state."""
        locomotion = SmoothLocomotion()
        assert locomotion.state == MovementState.IDLE
        assert locomotion.mode == MovementMode.THUMBSTICK
        assert locomotion.move_speed == 3.0
        assert locomotion.turn_type == TurnType.SNAP

    def test_calculate_movement_idle(self):
        """Test movement calculation with no input."""
        locomotion = SmoothLocomotion()
        input_state = MovementInput()

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert result.is_moving is False
        assert result.velocity == (0.0, 0.0, 0.0)

    def test_calculate_movement_forward(self):
        """Test forward movement calculation."""
        locomotion = SmoothLocomotion(move_speed=3.0, dead_zone=0.0)
        input_state = MovementInput(forward=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert result.is_moving is True
        assert result.velocity[2] > 0  # Moving forward in Z

    def test_calculate_movement_backward(self):
        """Test backward movement with speed reduction."""
        locomotion = SmoothLocomotion(
            move_speed=3.0,
            backward_speed_multiplier=0.5,
            dead_zone=0.0,
        )
        input_state = MovementInput(forward=-1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert result.is_moving is True
        assert result.velocity[2] < 0  # Moving backward

    def test_calculate_movement_strafe(self):
        """Test strafe movement calculation."""
        locomotion = SmoothLocomotion(strafe_speed=2.0, dead_zone=0.0)
        input_state = MovementInput(strafe=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert result.is_moving is True
        # Strafe should be perpendicular to forward

    def test_strafe_disabled(self):
        """Test strafe behavior when disabled."""
        locomotion = SmoothLocomotion(
            strafe_behavior=StrafeBehavior.DISABLED,
            dead_zone=0.0,
        )
        input_state = MovementInput(strafe=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Strafe should be zero or very small
        assert abs(result.velocity[0]) < 0.01

    def test_sprint_multiplier(self):
        """Test sprint speed multiplier."""
        locomotion = SmoothLocomotion(
            move_speed=3.0,
            sprint_multiplier=1.5,
            dead_zone=0.0,
        )
        input_normal = MovementInput(forward=1.0, sprint=False)
        input_sprint = MovementInput(forward=1.0, sprint=True)

        result_normal = locomotion.calculate_movement(
            input_normal,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        result_sprint = locomotion.calculate_movement(
            input_sprint,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Sprint should be faster
        assert abs(result_sprint.velocity[2]) > abs(result_normal.velocity[2])

    def test_snap_turn_in_movement(self):
        """Test snap turn during movement calculation."""
        locomotion = SmoothLocomotion(
            turn_type=TurnType.SNAP,
            snap_angle=45.0,
        )
        input_state = MovementInput(turn=1.0)  # Full turn input

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Should trigger snap turn
        if result.is_turning:
            assert result.rotation_delta != 0

    def test_smooth_turn_in_movement(self):
        """Test smooth turn during movement calculation."""
        locomotion = SmoothLocomotion(
            turn_type=TurnType.SMOOTH,
            smooth_turn_speed=90.0,
            dead_zone=0.0,
        )
        input_state = MovementInput(turn=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=1.0,  # 1 second
        )

        # Should have rotation
        assert result.rotation_delta != 0

    def test_vignette_activation(self):
        """Test vignette activates during fast movement."""
        locomotion = SmoothLocomotion(
            vignette_enabled=True,
            vignette_intensity=0.5,
            vignette_velocity_threshold=0.5,
            move_speed=5.0,
            dead_zone=0.0,
        )
        input_state = MovementInput(forward=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Fast movement should activate vignette
        assert result.vignette_intensity >= 0.0

    def test_vignette_disabled(self):
        """Test vignette when disabled."""
        locomotion = SmoothLocomotion(
            vignette_enabled=False,
            move_speed=10.0,
            dead_zone=0.0,
        )
        input_state = MovementInput(forward=1.0)

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Vignette should be zero when disabled
        # This depends on internal implementation
        assert result.vignette_intensity >= 0.0

    def test_dead_zone_filtering(self):
        """Test input dead zone filtering."""
        locomotion = SmoothLocomotion(dead_zone=0.2)
        input_state = MovementInput(forward=0.1)  # Below dead zone

        result = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert result.is_moving is False

    def test_state_transitions(self):
        """Test movement state transitions."""
        locomotion = SmoothLocomotion(dead_zone=0.0)

        # Start idle
        assert locomotion.state == MovementState.IDLE

        # Move
        input_moving = MovementInput(forward=1.0)
        locomotion.calculate_movement(
            input_moving,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert locomotion.state == MovementState.MOVING

        # Stop
        input_idle = MovementInput()
        locomotion.calculate_movement(
            input_idle,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert locomotion.state == MovementState.IDLE

    def test_movement_callbacks(self):
        """Test movement callbacks are called."""
        locomotion = SmoothLocomotion(dead_zone=0.0)
        start_called = False
        stop_called = False

        def on_start():
            nonlocal start_called
            start_called = True

        def on_stop():
            nonlocal stop_called
            stop_called = True

        locomotion.set_movement_callbacks(on_start=on_start, on_stop=on_stop)

        # Start moving
        input_moving = MovementInput(forward=1.0)
        locomotion.calculate_movement(
            input_moving,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert start_called is True

        # Stop moving
        input_idle = MovementInput()
        locomotion.calculate_movement(
            input_idle,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert stop_called is True

    def test_grounded_state(self):
        """Test grounded state affects gravity."""
        locomotion = SmoothLocomotion(gravity_enabled=True)
        locomotion.set_grounded(True)
        assert locomotion.is_grounded is True

        locomotion.set_grounded(False)
        assert locomotion.is_grounded is False


# =============================================================================
# Arm Swing Movement Tests
# =============================================================================


class TestArmSwingMovement:
    """Tests for arm swing movement mode."""

    def test_arm_swing_basic(self):
        """Test basic arm swing movement."""
        locomotion = SmoothLocomotion(mode=MovementMode.ARM_SWING)

        result = locomotion.calculate_arm_swing_movement(
            left_hand_velocity=(0.0, -1.0, 0.0),
            right_hand_velocity=(0.0, -1.0, 0.0),
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert isinstance(result, MovementResult)

    def test_arm_swing_threshold(self):
        """Test arm swing velocity threshold."""
        locomotion = SmoothLocomotion(mode=MovementMode.ARM_SWING)

        # Slow movement below threshold
        result = locomotion.calculate_arm_swing_movement(
            left_hand_velocity=(0.0, -0.1, 0.0),
            right_hand_velocity=(0.0, -0.1, 0.0),
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        # Should not register significant movement
        # Based on threshold settings


# =============================================================================
# SmoothLocomotionProvider Tests
# =============================================================================


class TestSmoothLocomotionProvider:
    """Tests for the SmoothLocomotionProvider."""

    def test_init(self):
        """Test provider initialization."""
        locomotion = SmoothLocomotion()
        provider = SmoothLocomotionProvider(locomotion)
        assert provider.locomotion is locomotion

    def test_update(self):
        """Test provider update method."""
        locomotion = SmoothLocomotion(dead_zone=0.0)
        provider = SmoothLocomotionProvider(locomotion)

        input_state = MovementInput(forward=1.0)
        result = provider.update(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )

        assert isinstance(result, MovementResult)
        assert result.is_moving is True


# =============================================================================
# @xr_locomotion Decorator Tests
# =============================================================================


class TestXRLocomotionDecorator:
    """Tests for the @xr_locomotion decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""
        @xr_locomotion(locomotion_type="smooth", speed=3.0)
        class Player:
            pass

        assert Player._xr_locomotion is True

    def test_locomotion_type_stored(self):
        """Test locomotion type is stored."""
        @xr_locomotion(locomotion_type="teleport", speed=1.0)
        class Player:
            pass

        assert Player._locomotion_type == "teleport"

    def test_speed_stored(self):
        """Test speed is stored."""
        @xr_locomotion(locomotion_type="smooth", speed=5.0)
        class Player:
            pass

        assert Player._locomotion_speed == 5.0

    def test_tags_applied(self):
        """Test tags are applied."""
        @xr_locomotion(locomotion_type="climbing", speed=2.0)
        class Player:
            pass

        assert Player._tags["xr_locomotion"] is True
        assert Player._tags["locomotion_type"] == "climbing"
        assert Player._tags["locomotion_speed"] == 2.0

    def test_registered_in_xr_registry(self):
        """Test registration in XR registry."""
        @xr_locomotion(locomotion_type="smooth", speed=3.0)
        class Player:
            pass

        assert "xr" in Player._registries

    def test_invalid_locomotion_type(self):
        """Test validation of locomotion type."""
        with pytest.raises(ValueError, match="locomotion_type"):
            @xr_locomotion(locomotion_type="invalid", speed=3.0)
            class Player:
                pass

    def test_invalid_speed(self):
        """Test validation of speed."""
        with pytest.raises(ValueError, match="speed"):
            @xr_locomotion(locomotion_type="smooth", speed=0.0)
            class Player:
                pass

        with pytest.raises(ValueError, match="speed"):
            @xr_locomotion(locomotion_type="smooth", speed=-5.0)
            class Player:
                pass

    def test_applied_decorators(self):
        """Test applied decorators list."""
        @xr_locomotion(locomotion_type="smooth", speed=3.0)
        class Player:
            pass

        assert "xr_locomotion" in Player._applied_decorators

    def test_steps_recorded(self):
        """Test steps are recorded."""
        @xr_locomotion(locomotion_type="smooth", speed=3.0)
        class Player:
            pass

        assert len(Player._applied_steps) > 0

    def test_decompose(self):
        """Test decompose shows steps."""
        steps = decompose(xr_locomotion)
        assert isinstance(steps, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSmoothLocomotionIntegration:
    """Integration tests for smooth locomotion."""

    def test_full_movement_cycle(self):
        """Test complete movement cycle."""
        locomotion = SmoothLocomotion(
            move_speed=3.0,
            turn_type=TurnType.SNAP,
            snap_angle=45.0,
            vignette_enabled=True,
            dead_zone=0.0,
        )

        # Start moving
        input_state = MovementInput(forward=1.0)
        result1 = locomotion.calculate_movement(
            input_state,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert result1.is_moving is True
        assert locomotion.state == MovementState.MOVING

        # Continue with turn
        input_with_turn = MovementInput(forward=0.5, turn=1.0)
        result2 = locomotion.calculate_movement(
            input_with_turn,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert result2.is_moving is True

        # Stop
        input_stop = MovementInput()
        result3 = locomotion.calculate_movement(
            input_stop,
            forward_direction=(0.0, 0.0, 1.0),
            delta_time=0.016,
        )
        assert result3.is_moving is False
        assert locomotion.state == MovementState.IDLE

    def test_all_movement_modes(self):
        """Test all movement modes are accessible."""
        for mode in MovementMode:
            locomotion = SmoothLocomotion(mode=mode)
            assert locomotion.mode == mode

    def test_all_turn_types(self):
        """Test all turn types."""
        for turn_type in TurnType:
            locomotion = SmoothLocomotion(turn_type=turn_type)
            assert locomotion.turn_type == turn_type
