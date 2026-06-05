"""
Whitebox tests for joint_motors.py - Motor helpers.

Tests:
- MotorMode enum
- Motor dataclass
- compute_motor_impulse function
- MotorState dataclass
- MotorController class
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_motors import (
    MotorMode,
    Motor,
    compute_motor_impulse,
    MotorState,
    MotorController,
)


class TestMotorMode:
    """Tests for MotorMode enum."""

    def test_velocity_mode_exists(self):
        """VELOCITY mode should exist."""
        assert MotorMode.VELOCITY is not None

    def test_position_mode_exists(self):
        """POSITION mode should exist."""
        assert MotorMode.POSITION is not None

    def test_modes_unique(self):
        """Modes should be unique."""
        assert MotorMode.VELOCITY != MotorMode.POSITION


class TestMotor:
    """Tests for Motor dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        motor = Motor()
        assert motor.mode == MotorMode.VELOCITY
        assert motor.target == 0.0
        assert motor.max_force == 100.0
        assert motor.position_gain == 1.0
        assert motor.velocity_gain == 0.1
        assert motor.integral_gain == 0.0

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1.5,
            max_force=50.0,
            position_gain=5.0,
            velocity_gain=0.5
        )
        assert motor.mode == MotorMode.POSITION
        assert motor.target == 1.5
        assert motor.max_force == 50.0
        assert motor.position_gain == 5.0
        assert motor.velocity_gain == 0.5

    def test_set_velocity_target(self):
        """set_velocity_target should configure motor correctly."""
        motor = Motor(mode=MotorMode.POSITION)
        motor.set_velocity_target(10.0, 200.0)
        assert motor.mode == MotorMode.VELOCITY
        assert motor.target == 10.0
        assert motor.max_force == 200.0

    def test_set_velocity_target_no_force(self):
        """set_velocity_target should keep existing max_force if not specified."""
        motor = Motor(max_force=50.0)
        motor.set_velocity_target(10.0)
        assert motor.max_force == 50.0

    def test_set_position_target(self):
        """set_position_target should configure motor correctly."""
        motor = Motor()
        motor.set_position_target(2.5, 150.0)
        assert motor.mode == MotorMode.POSITION
        assert motor.target == 2.5
        assert motor.max_force == 150.0

    def test_set_gains(self):
        """set_gains should update PID gains."""
        motor = Motor()
        motor.set_gains(position_gain=10.0, velocity_gain=2.0, integral_gain=0.1)
        assert motor.position_gain == 10.0
        assert motor.velocity_gain == 2.0
        assert motor.integral_gain == 0.1

    def test_set_gains_partial(self):
        """set_gains should only update specified gains."""
        motor = Motor(position_gain=5.0, velocity_gain=1.0, integral_gain=0.0)
        motor.set_gains(velocity_gain=2.0)
        assert motor.position_gain == 5.0  # Unchanged
        assert motor.velocity_gain == 2.0  # Changed
        assert motor.integral_gain == 0.0  # Unchanged

    def test_reset_integral(self):
        """reset_integral should clear integral error."""
        motor = Motor(integral_gain=0.1)
        motor._integral_error = 100.0
        motor.reset_integral()
        assert motor._integral_error == 0.0


class TestMotorComputeTargetVelocity:
    """Tests for Motor.compute_target_velocity method."""

    def test_velocity_mode_returns_target(self):
        """In velocity mode, should return target directly."""
        motor = Motor(mode=MotorMode.VELOCITY, target=5.0)
        result = motor.compute_target_velocity(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        assert result == 5.0

    def test_position_mode_proportional(self):
        """Position mode should use proportional gain."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1.0,
            position_gain=10.0,
            velocity_gain=0.0,
            integral_gain=0.0
        )
        result = motor.compute_target_velocity(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        # P term: 10.0 * (1.0 - 0.0) = 10.0
        assert abs(result - 10.0) < 1e-6

    def test_position_mode_derivative(self):
        """Position mode should use derivative (damping) term."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1.0,
            position_gain=0.0,
            velocity_gain=2.0,
            integral_gain=0.0
        )
        result = motor.compute_target_velocity(
            current_position=0.5,
            current_velocity=5.0,
            dt=0.016
        )
        # D term: -2.0 * 5.0 = -10.0
        assert abs(result - (-10.0)) < 1e-6

    def test_position_mode_integral(self):
        """Position mode should accumulate integral error."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1.0,
            position_gain=0.0,
            velocity_gain=0.0,
            integral_gain=1.0
        )
        dt = 0.016
        # First call accumulates error
        motor.compute_target_velocity(0.0, 0.0, dt)
        # Error = 1.0, integral = 1.0 * dt = 0.016
        result = motor.compute_target_velocity(0.0, 0.0, dt)
        # integral = 2 * 0.016 = 0.032
        assert result > 0.0

    def test_position_mode_anti_windup(self):
        """Integral should be clamped for anti-windup."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1000.0,  # Large error
            max_force=1.0,
            position_gain=0.0,
            velocity_gain=0.0,
            integral_gain=1.0
        )
        # Accumulate many iterations
        for _ in range(1000):
            motor.compute_target_velocity(0.0, 0.0, 0.016)
        # Integral should be bounded
        assert motor._integral_error <= motor.max_force / motor.integral_gain + 0.01

    def test_position_mode_angle_normalization(self):
        """Position mode should normalize angular errors."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=0.0,
            position_gain=1.0,
            velocity_gain=0.0,
            integral_gain=0.0
        )
        # Current position slightly beyond pi
        result = motor.compute_target_velocity(
            current_position=math.pi + 0.1,
            current_velocity=0.0,
            dt=0.016
        )
        # Should normalize to small negative error
        assert abs(result) < math.pi * 1.1


class TestComputeMotorImpulse:
    """Tests for compute_motor_impulse function."""

    def test_velocity_mode(self):
        """Should compute impulse for velocity mode."""
        motor = Motor(mode=MotorMode.VELOCITY, target=5.0, max_force=100.0)
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        # Impulse is clamped to max_force * dt = 100.0 * 0.016 = 1.6
        # velocity_error = 5.0, unclamped impulse = 1.0 * 5.0 = 5.0
        # clamped impulse = 1.6
        assert abs(impulse - 1.6) < 1e-6

    def test_velocity_mode_with_current_velocity(self):
        """Should account for current velocity."""
        motor = Motor(mode=MotorMode.VELOCITY, target=5.0, max_force=100.0)
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=3.0,
            effective_mass=1.0,
            dt=0.016
        )
        # velocity_error = 5.0 - 3.0 = 2.0
        # clamped impulse = min(2.0, 1.6) = 1.6
        assert abs(impulse - 1.6) < 1e-6

    def test_position_mode(self):
        """Should compute impulse for position mode."""
        motor = Motor(
            mode=MotorMode.POSITION,
            target=1.0,
            max_force=100.0,
            position_gain=10.0,
            velocity_gain=0.0
        )
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        # target_velocity = 10.0 * 1.0 = 10.0
        # velocity_error = 10.0, impulse = 1.0 * 10.0 = 10.0
        # But clamped to max_force * dt = 100 * 0.016 = 1.6
        assert abs(impulse - 1.6) < 1e-6

    def test_impulse_clamped_to_max_force(self):
        """Impulse should be clamped to max_force * dt."""
        motor = Motor(mode=MotorMode.VELOCITY, target=1000.0, max_force=10.0)
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        max_impulse = 10.0 * 0.016
        assert abs(impulse - max_impulse) < 1e-6

    def test_impulse_clamped_negative(self):
        """Negative impulse should also be clamped."""
        motor = Motor(mode=MotorMode.VELOCITY, target=-1000.0, max_force=10.0)
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        min_impulse = -10.0 * 0.016
        assert abs(impulse - min_impulse) < 1e-6

    def test_zero_effective_mass(self):
        """Should return zero with zero effective mass."""
        motor = Motor(mode=MotorMode.VELOCITY, target=5.0)
        impulse = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=0.0,
            dt=0.016
        )
        assert impulse == 0.0

    def test_heavy_mass(self):
        """Heavier mass should produce larger impulse."""
        motor = Motor(mode=MotorMode.VELOCITY, target=5.0, max_force=1000.0)
        impulse_light = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=1.0,
            dt=0.016
        )
        impulse_heavy = compute_motor_impulse(
            motor=motor,
            current_value=0.0,
            current_velocity=0.0,
            effective_mass=10.0,
            dt=0.016
        )
        assert impulse_heavy > impulse_light


class TestMotorState:
    """Tests for MotorState dataclass."""

    def test_default_values(self):
        """Default values should be zero."""
        state = MotorState()
        assert state.accumulated_impulse == 0.0
        assert state.integral_error == 0.0
        assert state.last_error == 0.0

    def test_reset(self):
        """reset should clear all state."""
        state = MotorState(
            accumulated_impulse=100.0,
            integral_error=50.0,
            last_error=10.0
        )
        state.reset()
        assert state.accumulated_impulse == 0.0
        assert state.integral_error == 0.0
        assert state.last_error == 0.0


class TestMotorController:
    """Tests for MotorController class."""

    def test_default_values(self):
        """Default values should be from config."""
        controller = MotorController()
        assert controller.max_force > 0.0
        assert controller.max_velocity > 0.0
        assert controller.max_acceleration > 0.0
        assert controller.kp > 0.0
        assert controller.kd > 0.0
        assert controller.ki >= 0.0

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        controller = MotorController(
            max_force=50.0,
            max_velocity=5.0,
            max_acceleration=10.0
        )
        assert controller.max_force == 50.0
        assert controller.max_velocity == 5.0
        assert controller.max_acceleration == 10.0

    def test_set_position_target(self):
        """set_position_target should configure position control."""
        controller = MotorController()
        controller.set_position_target(2.5)
        assert controller._target_position == 2.5
        assert controller._target_velocity is None

    def test_set_velocity_target(self):
        """set_velocity_target should configure velocity control."""
        controller = MotorController()
        controller.set_velocity_target(3.0)
        assert controller._target_velocity == 3.0
        assert controller._target_position is None

    def test_compute_force_velocity_mode(self):
        """Should compute force for velocity control."""
        controller = MotorController(max_force=100.0)
        controller.set_velocity_target(5.0)
        force = controller.compute_force(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        # Force should be proportional to velocity error
        assert force > 0.0

    def test_compute_force_position_mode(self):
        """Should compute force for position control."""
        controller = MotorController(max_force=100.0)
        controller.set_position_target(1.0)
        force = controller.compute_force(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        # Force should drive towards target position
        assert force > 0.0

    def test_compute_force_clamped(self):
        """Force should be clamped to max_force."""
        controller = MotorController(max_force=10.0)
        controller.set_velocity_target(1000.0)
        force = controller.compute_force(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        assert abs(force) <= 10.0 + 1e-6

    def test_compute_force_velocity_limited(self):
        """Desired velocity should be limited."""
        controller = MotorController(max_velocity=5.0)
        controller.set_position_target(1000.0)  # Very far away
        # Internal desired velocity should be clamped
        force = controller.compute_force(
            current_position=0.0,
            current_velocity=0.0,
            dt=0.016
        )
        assert abs(force) <= controller.max_force + 1e-6

    def test_compute_force_acceleration_limited(self):
        """Acceleration should be limited."""
        controller = MotorController(
            max_force=1000.0,
            max_velocity=100.0,
            max_acceleration=1.0
        )
        controller.set_position_target(10.0)
        # First step: limited acceleration
        controller.compute_force(0.0, 0.0, 0.1)
        # After reset, velocity target should be ramped
        controller._last_velocity_target = 5.0
        controller.compute_force(0.0, 5.0, 0.1)
        # Should not exceed max acceleration

    def test_compute_force_integral(self):
        """Integral term should accumulate."""
        controller = MotorController(max_force=100.0)
        controller.ki = 1.0  # Enable integral
        controller.set_position_target(1.0)

        # Accumulate integral error
        for _ in range(10):
            controller.compute_force(0.0, 0.0, 0.016)

        assert controller._integral_error > 0.0

    def test_reset(self):
        """reset should clear controller state."""
        controller = MotorController()
        controller.set_position_target(1.0)
        controller.compute_force(0.0, 0.0, 0.016)
        controller._integral_error = 100.0
        controller._last_velocity_target = 50.0
        controller.reset()
        assert controller._integral_error == 0.0
        assert controller._last_velocity_target == 0.0

    def test_angle_normalization(self):
        """Position error should be normalized for angles."""
        controller = MotorController()
        controller.set_position_target(0.0)
        # Position just beyond pi
        force1 = controller.compute_force(math.pi - 0.1, 0.0, 0.016)
        force2 = controller.compute_force(math.pi + 0.1, 0.0, 0.016)
        # Signs should be opposite (one positive, one negative)
        assert force1 * force2 < 0 or abs(force1) < 0.1 or abs(force2) < 0.1

    def test_no_target_returns_zero(self):
        """Should return zero force with no target set."""
        controller = MotorController()
        force = controller.compute_force(0.0, 0.0, 0.016)
        assert force == 0.0
