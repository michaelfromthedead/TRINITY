"""
Motor Helpers for Joint Constraints.

Provides motor functionality for joints that support motorized
movement (velocity targets or position targets).
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import math

from ..solver.config import (
    MOTOR_DEFAULT_KP,
    MOTOR_DEFAULT_KD,
    MOTOR_DEFAULT_KI,
    MOTOR_DEFAULT_MAX_FORCE,
    MOTOR_DEFAULT_MAX_VELOCITY,
    MOTOR_DEFAULT_MAX_ACCELERATION,
)


class MotorMode(Enum):
    """Motor operating mode."""
    VELOCITY = auto()  # Motor targets a velocity
    POSITION = auto()  # Motor targets a position (servo mode)


@dataclass
class Motor:
    """
    Motor configuration for joints.

    Motors apply forces to achieve either a target velocity or
    target position. The force is clamped to max_force.

    Attributes:
        mode: VELOCITY or POSITION mode.
        target: Target velocity (rad/s or m/s) or position (rad or m).
        max_force: Maximum force/torque the motor can apply.
        position_gain: P gain for position mode (servo stiffness).
        velocity_gain: D gain for position mode (servo damping).
        integral_gain: I gain for position mode (error accumulation).
    """
    mode: MotorMode = MotorMode.VELOCITY
    target: float = 0.0
    max_force: float = 100.0
    position_gain: float = 1.0
    velocity_gain: float = 0.1
    integral_gain: float = 0.0

    # Internal state for position mode
    _integral_error: float = field(default=0.0, repr=False)
    _last_error: float = field(default=0.0, repr=False)

    def set_velocity_target(self, velocity: float, max_force: float = None) -> None:
        """
        Set motor to velocity mode.

        Args:
            velocity: Target velocity.
            max_force: Maximum force (uses existing if None).
        """
        self.mode = MotorMode.VELOCITY
        self.target = velocity
        if max_force is not None:
            self.max_force = max_force

    def set_position_target(self, position: float, max_force: float = None) -> None:
        """
        Set motor to position mode.

        Args:
            position: Target position.
            max_force: Maximum force (uses existing if None).
        """
        self.mode = MotorMode.POSITION
        self.target = position
        if max_force is not None:
            self.max_force = max_force

    def set_gains(
        self,
        position_gain: float = None,
        velocity_gain: float = None,
        integral_gain: float = None
    ) -> None:
        """
        Set PID gains for position mode.

        Args:
            position_gain: P gain (proportional).
            velocity_gain: D gain (derivative).
            integral_gain: I gain (integral).
        """
        if position_gain is not None:
            self.position_gain = position_gain
        if velocity_gain is not None:
            self.velocity_gain = velocity_gain
        if integral_gain is not None:
            self.integral_gain = integral_gain

    def reset_integral(self) -> None:
        """Reset integral error accumulator."""
        self._integral_error = 0.0

    def compute_target_velocity(
        self,
        current_position: float,
        current_velocity: float,
        dt: float
    ) -> float:
        """
        Compute target velocity for position mode.

        Uses PID control to compute a velocity that will drive
        the joint towards the target position.

        Args:
            current_position: Current joint position.
            current_velocity: Current joint velocity.
            dt: Time step.

        Returns:
            Target velocity for constraint solver.
        """
        if self.mode == MotorMode.VELOCITY:
            return self.target

        # Position error
        error = self.target - current_position

        # Normalize angular errors to [-pi, pi]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        # P term
        p_term = self.position_gain * error

        # D term (derivative of error = -velocity when target is constant)
        d_term = -self.velocity_gain * current_velocity

        # I term
        if self.integral_gain > 0:
            self._integral_error += error * dt
            # Anti-windup: clamp integral
            max_integral = self.max_force / (self.integral_gain + 1e-10)
            self._integral_error = max(-max_integral, min(max_integral, self._integral_error))

        i_term = self.integral_gain * self._integral_error

        return p_term + d_term + i_term


def compute_motor_impulse(
    motor: Motor,
    current_value: float,
    current_velocity: float,
    effective_mass: float,
    dt: float
) -> float:
    """
    Compute motor impulse for constraint solver.

    Args:
        motor: Motor configuration.
        current_value: Current position/angle.
        current_velocity: Current velocity/angular velocity.
        effective_mass: Effective mass of the constraint.
        dt: Time step.

    Returns:
        Motor impulse (clamped to max force * dt).
    """
    if effective_mass == 0:
        return 0.0

    # Compute target velocity
    if motor.mode == MotorMode.VELOCITY:
        target_velocity = motor.target
    else:
        target_velocity = motor.compute_target_velocity(
            current_value, current_velocity, dt
        )

    # Velocity error
    velocity_error = target_velocity - current_velocity

    # Compute impulse
    impulse = effective_mass * velocity_error

    # Clamp to max force
    max_impulse = motor.max_force * dt
    impulse = max(-max_impulse, min(max_impulse, impulse))

    return impulse


@dataclass
class MotorState:
    """
    Runtime state for a motor.

    Tracks accumulated impulse and integral error for
    motors across simulation steps.
    """
    accumulated_impulse: float = 0.0
    integral_error: float = 0.0
    last_error: float = 0.0

    def reset(self) -> None:
        """Reset motor state."""
        self.accumulated_impulse = 0.0
        self.integral_error = 0.0
        self.last_error = 0.0


class MotorController:
    """
    Advanced motor controller with multiple control modes.

    Provides more sophisticated motor control including:
    - PID control for position
    - Feed-forward for velocity
    - Acceleration limits
    - Jerk limits
    """

    def __init__(
        self,
        max_force: float = MOTOR_DEFAULT_MAX_FORCE,
        max_velocity: float = MOTOR_DEFAULT_MAX_VELOCITY,
        max_acceleration: float = MOTOR_DEFAULT_MAX_ACCELERATION
    ):
        """
        Initialize motor controller.

        Args:
            max_force: Maximum motor force.
            max_velocity: Maximum velocity limit.
            max_acceleration: Maximum acceleration limit.
        """
        self.max_force = max_force
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # PID gains (from config for tunability)
        self.kp = MOTOR_DEFAULT_KP  # Position gain
        self.kd = MOTOR_DEFAULT_KD  # Velocity gain (damping)
        self.ki = MOTOR_DEFAULT_KI  # Integral gain

        # State
        self._target_position: Optional[float] = None
        self._target_velocity: Optional[float] = None
        self._integral_error = 0.0
        self._last_velocity_target = 0.0

    def set_position_target(self, position: float) -> None:
        """Set position target (position control mode)."""
        self._target_position = position
        self._target_velocity = None

    def set_velocity_target(self, velocity: float) -> None:
        """Set velocity target (velocity control mode)."""
        self._target_velocity = velocity
        self._target_position = None

    def compute_force(
        self,
        current_position: float,
        current_velocity: float,
        dt: float
    ) -> float:
        """
        Compute motor force for current state.

        Args:
            current_position: Current joint position.
            current_velocity: Current joint velocity.
            dt: Time step.

        Returns:
            Force to apply (clamped to max_force).
        """
        force = 0.0

        if self._target_velocity is not None:
            # Velocity control mode
            velocity_error = self._target_velocity - current_velocity
            force = self.kd * velocity_error

        elif self._target_position is not None:
            # Position control mode
            position_error = self._target_position - current_position

            # Normalize for angular
            while position_error > math.pi:
                position_error -= 2 * math.pi
            while position_error < -math.pi:
                position_error += 2 * math.pi

            # Compute desired velocity (with limits)
            desired_velocity = self.kp * position_error
            desired_velocity = max(-self.max_velocity, min(self.max_velocity, desired_velocity))

            # Apply acceleration limit
            velocity_change = desired_velocity - self._last_velocity_target
            max_change = self.max_acceleration * dt
            velocity_change = max(-max_change, min(max_change, velocity_change))
            desired_velocity = self._last_velocity_target + velocity_change
            self._last_velocity_target = desired_velocity

            # Compute force
            velocity_error = desired_velocity - current_velocity
            force = self.kd * velocity_error

            # Integral term
            if self.ki > 0:
                self._integral_error += position_error * dt
                max_integral = self.max_force / (self.ki + 1e-10)
                self._integral_error = max(-max_integral, min(max_integral, self._integral_error))
                force += self.ki * self._integral_error

        # Clamp force
        force = max(-self.max_force, min(self.max_force, force))

        return force

    def reset(self) -> None:
        """Reset controller state."""
        self._integral_error = 0.0
        self._last_velocity_target = 0.0
