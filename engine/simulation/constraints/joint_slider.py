"""
Slider Joint (Prismatic Joint) Implementation.

A slider joint allows translation along a single axis while constraining
all other degrees of freedom including all rotations. It supports optional
position limits and motors.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

from .joint_base import Joint, JointState
from .joint_motors import Motor, MotorMode
from .joint_limits import LinearLimit, LimitState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig, POSITION_CORRECTION_FACTOR


class SliderJoint(Joint):
    """
    Slider joint (prismatic joint) allowing translation along a single axis.

    The joint constrains 5 DOF: 2 linear (perpendicular to axis) and 3 angular.
    Only translation along the slider axis is allowed.

    Optionally supports:
    - Position limits (min/max translation)
    - Motor (velocity or position target)

    Attributes:
        local_axis_a: Slider axis in body A's local space.
        reference_position: Reference position for measuring translation.
        limits_enabled: Whether position limits are active.
        min_distance: Minimum translation.
        max_distance: Maximum translation.
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
        break_force: float = 0.0,
        break_torque: float = 0.0
    ):
        """
        Initialize slider joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            local_axis_a: Slider axis in body A's local space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, break_torque
        )

        # Slider axis (default: X axis)
        self._local_axis_a = (local_axis_a or Vec3.unit_x()).normalized()

        # Compute perpendicular axes
        self._local_perp1_a, self._local_perp2_a = self._compute_perpendiculars(
            self._local_axis_a
        )

        # Store reference orientation and position
        if body_b is not None:
            self._reference_orientation = (
                body_b.orientation.conjugate() * body_a.orientation
            )
        else:
            self._reference_orientation = body_a.orientation

        self._reference_position = self._compute_current_translation()

        # Limits
        self._limits_enabled = False
        self._min_distance = -1.0
        self._max_distance = 1.0
        self._limit_state = LimitState.INACTIVE
        self._limit = LinearLimit(self._min_distance, self._max_distance)

        # Motor
        self._motor_enabled = False
        self._motor = Motor(MotorMode.VELOCITY, 0.0, 0.0)

        # Initialize constraint storage
        # 5 DOF constraints (2 linear + 3 angular) + optional limit + motor
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
        """Get slider axis in body A's local space."""
        return self._local_axis_a

    @local_axis_a.setter
    def local_axis_a(self, value: Vec3) -> None:
        """Set slider axis in body A's local space."""
        self._local_axis_a = value.normalized()
        self._local_perp1_a, self._local_perp2_a = self._compute_perpendiculars(
            self._local_axis_a
        )

    @property
    def limits_enabled(self) -> bool:
        """Check if position limits are enabled."""
        return self._limits_enabled

    @limits_enabled.setter
    def limits_enabled(self, value: bool) -> None:
        """Enable/disable position limits."""
        self._limits_enabled = value

    @property
    def min_distance(self) -> float:
        """Get minimum translation limit."""
        return self._min_distance

    @min_distance.setter
    def min_distance(self, value: float) -> None:
        """Set minimum translation limit."""
        self._min_distance = value
        self._limit.lower = value

    @property
    def max_distance(self) -> float:
        """Get maximum translation limit."""
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        """Set maximum translation limit."""
        self._max_distance = value
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

    def get_current_translation(self) -> float:
        """Get current joint translation relative to reference."""
        return self._compute_current_translation() - self._reference_position

    def get_joint_speed(self) -> float:
        """Get current linear speed along slider axis."""
        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)

        rel_vel = self._body_a.velocity
        if self._body_b is not None:
            rel_vel = rel_vel - self._body_b.velocity

        return rel_vel.dot(axis_a)

    def set_limits(self, min_distance: float, max_distance: float) -> None:
        """
        Set position limits.

        Args:
            min_distance: Minimum translation.
            max_distance: Maximum translation.
        """
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._limit = LinearLimit(min_distance, max_distance)
        self._limits_enabled = True

    def set_motor_speed(self, speed: float, max_force: float) -> None:
        """
        Set motor to velocity mode.

        Args:
            speed: Target linear velocity.
            max_force: Maximum motor force.
        """
        self._motor = Motor(MotorMode.VELOCITY, speed, max_force)
        self._motor_enabled = True

    def set_motor_position(self, position: float, max_force: float) -> None:
        """
        Set motor to position mode.

        Args:
            position: Target position.
            max_force: Maximum motor force.
        """
        self._motor = Motor(MotorMode.POSITION, position, max_force)
        self._motor_enabled = True

    def get_constraint_count(self) -> int:
        """Get number of active constraint rows."""
        return self._active_constraint_count

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare slider joint for solving."""
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
        perp1_a = self._body_a.local_to_world_direction(self._local_perp1_a)
        perp2_a = self._body_a.local_to_world_direction(self._local_perp2_a)

        # Position error vector
        d = anchor_b - anchor_a

        # ============ LINEAR CONSTRAINTS (2 rows) ============
        # Constrain motion perpendicular to slider axis
        for i, perp in enumerate([perp1_a, perp2_a]):
            self._jacobians[i] = Jacobian(
                linear_a=-perp,
                angular_a=-r_a.cross(perp),
                linear_b=perp,
                angular_b=r_b.cross(perp)
            )
            self._effective_masses[i] = self._compute_effective_mass(self._jacobians[i])
            error = d.dot(perp)
            self._biases[i] = config.baumgarte_factor * error / dt

        # ============ ANGULAR CONSTRAINTS (3 rows) ============
        # Lock all rotations
        if self._body_b is not None:
            q_current = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_current = self._body_a.orientation

        q_error = q_current * self._reference_orientation.conjugate()
        if q_error.w < 0:
            q_error = Quaternion(-q_error.x, -q_error.y, -q_error.z, -q_error.w)

        angular_error = Vec3(q_error.x, q_error.y, q_error.z) * 2.0

        axes = [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]
        angular_errors = [angular_error.x, angular_error.y, angular_error.z]

        for i, (axis, error) in enumerate(zip(axes, angular_errors)):
            idx = i + 2
            self._jacobians[idx] = Jacobian(
                linear_a=Vec3.zero(),
                angular_a=-axis,
                linear_b=Vec3.zero(),
                angular_b=axis
            )
            self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
            self._biases[idx] = config.baumgarte_factor * error / dt

        self._active_constraint_count = 5

        # ============ LIMIT CONSTRAINT ============
        if self._limits_enabled:
            translation = self.get_current_translation()
            self._limit_state = LimitState.INACTIVE

            # Check lower limit
            if translation <= self._min_distance:
                self._limit_state = LimitState.AT_LOWER
                self._jacobians[5] = Jacobian(
                    linear_a=-axis_a,
                    angular_a=-r_a.cross(axis_a),
                    linear_b=axis_a,
                    angular_b=r_b.cross(axis_a)
                )
                self._effective_masses[5] = self._compute_effective_mass(self._jacobians[5])
                self._biases[5] = config.baumgarte_factor * (self._min_distance - translation) / dt
                self._lower_limits[5] = 0.0
                self._upper_limits[5] = float("inf")
                self._active_constraint_count = 6

            # Check upper limit
            elif translation >= self._max_distance:
                self._limit_state = LimitState.AT_UPPER
                self._jacobians[5] = Jacobian(
                    linear_a=axis_a,
                    angular_a=r_a.cross(axis_a),
                    linear_b=-axis_a,
                    angular_b=-r_b.cross(axis_a)
                )
                self._effective_masses[5] = self._compute_effective_mass(self._jacobians[5])
                self._biases[5] = config.baumgarte_factor * (translation - self._max_distance) / dt
                self._lower_limits[5] = 0.0
                self._upper_limits[5] = float("inf")
                self._active_constraint_count = 6

        # ============ MOTOR CONSTRAINT ============
        if self._motor_enabled:
            motor_idx = self._active_constraint_count

            self._jacobians[motor_idx] = Jacobian(
                linear_a=-axis_a,
                angular_a=-r_a.cross(axis_a),
                linear_b=axis_a,
                angular_b=r_b.cross(axis_a)
            )
            self._effective_masses[motor_idx] = self._compute_effective_mass(
                self._jacobians[motor_idx]
            )

            if self._motor.mode == MotorMode.VELOCITY:
                current_speed = self.get_joint_speed()
                self._biases[motor_idx] = self._motor.target - current_speed
            else:
                current_pos = self.get_current_translation()
                pos_error = self._motor.target - current_pos
                self._biases[motor_idx] = pos_error / dt

            max_impulse = self._motor.max_force * dt
            self._lower_limits[motor_idx] = -max_impulse
            self._upper_limits[motor_idx] = max_impulse

            self._active_constraint_count += 1

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for slider joint."""
        max_error = 0.0

        # Linear error (perpendicular to axis)
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        d = anchor_b - anchor_a

        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)
        perp1_a = self._body_a.local_to_world_direction(self._local_perp1_a)
        perp2_a = self._body_a.local_to_world_direction(self._local_perp2_a)

        error1 = d.dot(perp1_a)
        error2 = d.dot(perp2_a)
        max_error = max(max_error, abs(error1), abs(error2))

        # Angular error
        if self._body_b is not None:
            q_current = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_current = self._body_a.orientation

        q_error = q_current * self._reference_orientation.conjugate()
        if q_error.w < 0:
            q_error = Quaternion(-q_error.x, -q_error.y, -q_error.z, -q_error.w)

        angular_error = Vec3(q_error.x, q_error.y, q_error.z) * 2.0
        max_error = max(max_error, angular_error.length())

        # Apply corrections
        self._apply_position_corrections(
            [error1, error2],
            [perp1_a, perp2_a],
            angular_error,
            max_correction
        )

        return max_error

    def _apply_position_corrections(
        self,
        linear_errors: List[float],
        linear_axes: List[Vec3],
        angular_error: Vec3,
        max_correction: float
    ) -> None:
        """Apply position corrections."""
        r_a, r_b = self._get_r_vectors()

        # Linear corrections
        for i, (error, axis) in enumerate(zip(linear_errors, linear_axes)):
            if abs(error) < 1e-6:
                continue

            k = self._effective_masses[i]
            if k == 0:
                continue

            c = max(-max_correction, min(max_correction, error))
            impulse = -k * c * POSITION_CORRECTION_FACTOR

            if not self._body_a.is_static:
                self._body_a.position = self._body_a.position - axis * (
                    self._body_a.inv_mass * impulse
                )

            if self._body_b is not None and not self._body_b.is_static:
                self._body_b.position = self._body_b.position + axis * (
                    self._body_b.inv_mass * impulse
                )

        # Angular corrections (simplified)
        if angular_error.length() > 1e-6:
            for i, (axis_val, axis) in enumerate(zip(
                [angular_error.x, angular_error.y, angular_error.z],
                [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]
            )):
                if abs(axis_val) < 1e-6:
                    continue

                idx = i + 2
                k = self._effective_masses[idx]
                if k == 0:
                    continue

                c = max(-max_correction, min(max_correction, axis_val))
                impulse = -k * c * POSITION_CORRECTION_FACTOR

                if not self._body_a.is_static:
                    delta_ang = self._body_a.inv_inertia_world * (axis * (-impulse))
                    angle = delta_ang.length()
                    if angle > 1e-10:
                        rot_axis = delta_ang / angle
                        delta_q = Quaternion.from_axis_angle(rot_axis, angle)
                        self._body_a.orientation = (
                            delta_q * self._body_a.orientation
                        ).normalized()

                if self._body_b is not None and not self._body_b.is_static:
                    delta_ang = self._body_b.inv_inertia_world * (axis * impulse)
                    angle = delta_ang.length()
                    if angle > 1e-10:
                        rot_axis = delta_ang / angle
                        delta_q = Quaternion.from_axis_angle(rot_axis, angle)
                        self._body_b.orientation = (
                            delta_q * self._body_b.orientation
                        ).normalized()

    def _compute_current_translation(self) -> float:
        """Compute current translation along slider axis."""
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        axis_a = self._body_a.local_to_world_direction(self._local_axis_a)

        d = anchor_b - anchor_a
        return d.dot(axis_a)

    def _compute_perpendiculars(self, axis: Vec3) -> tuple:
        """Compute two perpendicular axes."""
        if abs(axis.x) < 0.9:
            perp1 = axis.cross(Vec3.unit_x()).normalized()
        else:
            perp1 = axis.cross(Vec3.unit_y()).normalized()

        perp2 = axis.cross(perp1).normalized()
        return perp1, perp2

    @classmethod
    def create_at_point(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3,
        world_axis: Vec3,
        min_distance: float = None,
        max_distance: float = None
    ) -> "SliderJoint":
        """
        Create slider joint at a world position.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor: Anchor position in world space.
            world_axis: Slider axis in world space.
            min_distance: Optional minimum translation.
            max_distance: Optional maximum translation.

        Returns:
            Configured SliderJoint.
        """
        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)
        local_axis_a = body_a.world_to_local_direction(world_axis)

        joint = cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            local_axis_a=local_axis_a
        )

        if min_distance is not None and max_distance is not None:
            joint.set_limits(min_distance, max_distance)

        return joint
