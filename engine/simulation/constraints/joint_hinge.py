"""
Hinge Joint (Revolute Joint) Implementation.

A hinge joint allows rotation around a single axis while constraining
all other degrees of freedom. It supports optional angle limits and motors.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

from .joint_base import Joint, JointState
from .joint_motors import Motor, MotorMode, compute_motor_impulse
from .joint_limits import AngularLimit, LimitState, compute_limit_impulse
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig, POSITION_CORRECTION_FACTOR


class HingeJoint(Joint):
    """
    Hinge joint (revolute joint) allowing rotation around a single axis.

    The joint constrains 5 DOF, leaving only rotation around the hinge axis free.
    Optionally supports:
    - Angle limits (min/max rotation)
    - Motor (velocity or position target)

    Attributes:
        local_axis_a: Hinge axis in body A's local space.
        local_axis_b: Hinge axis in body B's local space.
        reference_angle: Reference angle for measuring rotation.
        limits_enabled: Whether angle limits are active.
        min_angle: Minimum angle (radians).
        max_angle: Maximum angle (radians).
        motor_enabled: Whether motor is active.
        motor: Motor configuration.
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        local_axis_a: Vec3 = None,
        local_axis_b: Vec3 = None,
        break_force: float = 0.0,
        break_torque: float = 0.0
    ):
        """
        Initialize hinge joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            local_axis_a: Hinge axis in body A's local space.
            local_axis_b: Hinge axis in body B's local space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, break_torque
        )

        # Hinge axis (default: Y axis)
        self._local_axis_a = local_axis_a or Vec3.unit_y()
        self._local_axis_b = local_axis_b or Vec3.unit_y()

        # Normalize axes
        self._local_axis_a = self._local_axis_a.normalized()
        self._local_axis_b = self._local_axis_b.normalized()

        # Compute perpendicular axes for constraint
        self._local_perp_a = self._compute_perpendicular(self._local_axis_a)
        self._local_perp_b = self._compute_perpendicular(self._local_axis_b)

        # Store reference angle
        self._reference_angle = self._compute_current_angle()

        # Limits
        self._limits_enabled = False
        self._min_angle = -math.pi
        self._max_angle = math.pi
        self._limit_state = LimitState.INACTIVE
        self._limit = AngularLimit(self._min_angle, self._max_angle)

        # Motor
        self._motor_enabled = False
        self._motor = Motor(MotorMode.VELOCITY, 0.0, 0.0)

        # Initialize constraint storage
        # 5 DOF constraints (3 linear + 2 angular) + optional limit + motor
        max_rows = 8
        self._jacobians = [Jacobian() for _ in range(max_rows)]
        self._effective_masses = [0.0] * max_rows
        self._biases = [0.0] * max_rows
        self._accumulated_impulse = [0.0] * max_rows
        self._warm_start_impulse = [0.0] * max_rows
        self._lower_limits = [float("-inf")] * max_rows
        self._upper_limits = [float("inf")] * max_rows

        self._active_constraint_count = 5

    @property
    def local_axis_a(self) -> Vec3:
        """Get hinge axis in body A's local space."""
        return self._local_axis_a

    @local_axis_a.setter
    def local_axis_a(self, value: Vec3) -> None:
        """Set hinge axis in body A's local space."""
        self._local_axis_a = value.normalized()
        self._local_perp_a = self._compute_perpendicular(self._local_axis_a)

    @property
    def local_axis_b(self) -> Vec3:
        """Get hinge axis in body B's local space."""
        return self._local_axis_b

    @local_axis_b.setter
    def local_axis_b(self, value: Vec3) -> None:
        """Set hinge axis in body B's local space."""
        self._local_axis_b = value.normalized()
        self._local_perp_b = self._compute_perpendicular(self._local_axis_b)

    @property
    def limits_enabled(self) -> bool:
        """Check if angle limits are enabled."""
        return self._limits_enabled

    @limits_enabled.setter
    def limits_enabled(self, value: bool) -> None:
        """Enable/disable angle limits."""
        self._limits_enabled = value

    @property
    def min_angle(self) -> float:
        """Get minimum angle limit (radians)."""
        return self._min_angle

    @min_angle.setter
    def min_angle(self, value: float) -> None:
        """Set minimum angle limit."""
        self._min_angle = value
        self._limit.lower = value

    @property
    def max_angle(self) -> float:
        """Get maximum angle limit (radians)."""
        return self._max_angle

    @max_angle.setter
    def max_angle(self, value: float) -> None:
        """Set maximum angle limit."""
        self._max_angle = value
        self._limit.upper = value

    @property
    def motor_enabled(self) -> bool:
        """Check if motor is enabled."""
        return self._motor_enabled

    @motor_enabled.setter
    def motor_enabled(self, value: bool) -> None:
        """Enable/disable motor."""
        self._motor_enabled = value

    @property
    def motor(self) -> Motor:
        """Get motor configuration."""
        return self._motor

    def get_current_angle(self) -> float:
        """Get current joint angle relative to reference."""
        return self._compute_current_angle() - self._reference_angle

    def get_joint_speed(self) -> float:
        """Get current angular speed around hinge axis."""
        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)

        rel_angular_vel = self._body_a.angular_velocity
        if self._body_b is not None:
            rel_angular_vel = rel_angular_vel - self._body_b.angular_velocity

        return rel_angular_vel.dot(axis_a)

    def set_limits(self, min_angle: float, max_angle: float) -> None:
        """
        Set angle limits.

        Args:
            min_angle: Minimum angle (radians).
            max_angle: Maximum angle (radians).
        """
        self._min_angle = min_angle
        self._max_angle = max_angle
        self._limit = AngularLimit(min_angle, max_angle)
        self._limits_enabled = True

    def set_motor_speed(self, speed: float, max_torque: float) -> None:
        """
        Set motor to velocity mode.

        Args:
            speed: Target angular velocity (rad/s).
            max_torque: Maximum motor torque.
        """
        self._motor = Motor(MotorMode.VELOCITY, speed, max_torque)
        self._motor_enabled = True

    def set_motor_position(self, angle: float, max_torque: float) -> None:
        """
        Set motor to position mode.

        Args:
            angle: Target angle (radians).
            max_torque: Maximum motor torque.
        """
        self._motor = Motor(MotorMode.POSITION, angle, max_torque)
        self._motor_enabled = True

    def get_constraint_count(self) -> int:
        """Get number of active constraint rows."""
        return self._active_constraint_count

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare hinge joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Transform to world space
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)
        perp1_a = self._body_a.local_to_world_direction(self._local_perp_a)
        perp2_a = axis_a.cross(perp1_a)

        # Position error
        position_error = anchor_b - anchor_a

        # ============ LINEAR CONSTRAINTS (3 rows) ============
        axes = [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]
        for i, axis in enumerate(axes):
            self._jacobians[i] = Jacobian(
                linear_a=-axis,
                angular_a=-r_a.cross(axis),
                linear_b=axis,
                angular_b=r_b.cross(axis)
            )
            self._effective_masses[i] = self._compute_effective_mass(self._jacobians[i])
            error = position_error.dot(axis)
            self._biases[i] = config.baumgarte_factor * error / dt

        # ============ ANGULAR CONSTRAINTS (2 rows) ============
        # Keep the two axes perpendicular to hinge axis aligned
        if self._body_b is not None:
            axis_b = self._body_b.local_to_world_direction(self._local_axis_b)
        else:
            axis_b = self._local_axis_b

        # Error axes are cross products
        error1 = perp1_a.cross(axis_b)
        error2 = perp2_a.cross(axis_b)

        self._jacobians[3] = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-perp1_a,
            linear_b=Vec3.zero(),
            angular_b=perp1_a
        )
        self._effective_masses[3] = self._compute_effective_mass(self._jacobians[3])
        self._biases[3] = config.baumgarte_factor * error1.dot(axis_a) / dt

        self._jacobians[4] = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-perp2_a,
            linear_b=Vec3.zero(),
            angular_b=perp2_a
        )
        self._effective_masses[4] = self._compute_effective_mass(self._jacobians[4])
        self._biases[4] = config.baumgarte_factor * error2.dot(axis_a) / dt

        self._active_constraint_count = 5

        # ============ LIMIT CONSTRAINT ============
        if self._limits_enabled:
            angle = self.get_current_angle()
            self._limit_state = LimitState.INACTIVE

            # Check lower limit
            if angle <= self._min_angle:
                self._limit_state = LimitState.AT_LOWER
                self._jacobians[5] = Jacobian(
                    linear_a=Vec3.zero(),
                    angular_a=-axis_a,
                    linear_b=Vec3.zero(),
                    angular_b=axis_a
                )
                self._effective_masses[5] = self._compute_effective_mass(self._jacobians[5])
                self._biases[5] = config.baumgarte_factor * (self._min_angle - angle) / dt
                self._lower_limits[5] = 0.0
                self._upper_limits[5] = float("inf")
                self._active_constraint_count = 6

            # Check upper limit
            elif angle >= self._max_angle:
                self._limit_state = LimitState.AT_UPPER
                self._jacobians[5] = Jacobian(
                    linear_a=Vec3.zero(),
                    angular_a=axis_a,
                    linear_b=Vec3.zero(),
                    angular_b=-axis_a
                )
                self._effective_masses[5] = self._compute_effective_mass(self._jacobians[5])
                self._biases[5] = config.baumgarte_factor * (angle - self._max_angle) / dt
                self._lower_limits[5] = 0.0
                self._upper_limits[5] = float("inf")
                self._active_constraint_count = 6

        # ============ MOTOR CONSTRAINT ============
        if self._motor_enabled:
            motor_idx = self._active_constraint_count

            self._jacobians[motor_idx] = Jacobian(
                linear_a=Vec3.zero(),
                angular_a=-axis_a,
                linear_b=Vec3.zero(),
                angular_b=axis_a
            )
            self._effective_masses[motor_idx] = self._compute_effective_mass(
                self._jacobians[motor_idx]
            )

            if self._motor.mode == MotorMode.VELOCITY:
                # Velocity motor
                current_speed = self.get_joint_speed()
                self._biases[motor_idx] = self._motor.target - current_speed
            else:
                # Position motor (servo)
                current_angle = self.get_current_angle()
                angle_error = self._motor.target - current_angle
                self._biases[motor_idx] = angle_error / dt

            # Motor impulse limits
            max_impulse = self._motor.max_force * dt
            self._lower_limits[motor_idx] = -max_impulse
            self._upper_limits[motor_idx] = max_impulse

            self._active_constraint_count += 1

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for hinge joint."""
        max_error = 0.0

        # Position error
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        position_error = anchor_b - anchor_a
        max_error = max(max_error, position_error.length())

        # Angular alignment error
        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)
        if self._body_b is not None:
            axis_b = self._body_b.local_to_world_direction(self._local_axis_b)
        else:
            axis_b = self._local_axis_b

        axis_error = axis_a.cross(axis_b)
        max_error = max(max_error, axis_error.length())

        # Solve position
        self._solve_position_rows(position_error, axis_error, max_correction)

        return max_error

    def _solve_position_rows(
        self,
        position_error: Vec3,
        axis_error: Vec3,
        max_correction: float
    ) -> None:
        """Apply position corrections."""
        r_a, r_b = self._get_r_vectors()

        # Linear corrections
        errors = [position_error.x, position_error.y, position_error.z]
        axes = [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]

        for i, (error, axis) in enumerate(zip(errors, axes)):
            if abs(error) < 1e-6:
                continue

            jacobian = self._jacobians[i]
            k = self._effective_masses[i]
            if k == 0:
                continue

            c = max(-max_correction, min(max_correction, error))
            impulse = -k * c * POSITION_CORRECTION_FACTOR

            if not self._body_a.is_static:
                self._body_a.position = self._body_a.position + jacobian.linear_a * (
                    self._body_a.inv_mass * impulse
                )

            if self._body_b is not None and not self._body_b.is_static:
                self._body_b.position = self._body_b.position + jacobian.linear_b * (
                    self._body_b.inv_mass * impulse
                )

    def _compute_current_angle(self) -> float:
        """Compute current angle between bodies around hinge axis."""
        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)
        perp_a = self._body_a.local_to_world_direction(self._local_perp_a)

        if self._body_b is not None:
            perp_b = self._body_b.local_to_world_direction(self._local_perp_b)
        else:
            perp_b = self._local_perp_b

        # Angle between perpendicular vectors
        cos_angle = perp_a.dot(perp_b)
        sin_angle = axis_a.dot(perp_a.cross(perp_b))

        return math.atan2(sin_angle, cos_angle)

    def _compute_perpendicular(self, axis: Vec3) -> Vec3:
        """Compute a vector perpendicular to axis."""
        if abs(axis.x) < 0.9:
            return axis.cross(Vec3.unit_x()).normalized()
        else:
            return axis.cross(Vec3.unit_y()).normalized()

    @classmethod
    def create_at_point(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3,
        world_axis: Vec3,
        min_angle: float = None,
        max_angle: float = None
    ) -> "HingeJoint":
        """
        Create hinge joint at a world position.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor: Anchor position in world space.
            world_axis: Hinge axis in world space.
            min_angle: Optional minimum angle.
            max_angle: Optional maximum angle.

        Returns:
            Configured HingeJoint.
        """
        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)
        local_axis_a = body_a.world_to_local_direction(world_axis)
        local_axis_b = body_b.world_to_local_direction(world_axis)

        joint = cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            local_axis_a=local_axis_a,
            local_axis_b=local_axis_b
        )

        if min_angle is not None and max_angle is not None:
            joint.set_limits(min_angle, max_angle)

        return joint
