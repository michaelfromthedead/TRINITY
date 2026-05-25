"""
Ball Joint (Spherical Joint) Implementation.

A ball joint allows rotation around all three axes while constraining
the anchor points to coincide. It supports optional cone limits.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

from .joint_base import Joint, JointState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig, POSITION_CORRECTION_FACTOR


class BallJoint(Joint):
    """
    Ball joint (spherical joint) allowing rotation around all axes.

    The joint constrains 3 DOF (linear), leaving all 3 rotational DOF free.
    Useful for shoulder joints, ragdoll connections, etc.

    Optionally supports:
    - Cone limit: restricts the angle between body axes
    - Twist limit: restricts rotation around the cone axis

    Attributes:
        cone_limit_enabled: Whether cone limit is active.
        cone_limit_angle: Maximum angle from reference (radians).
        twist_limit_enabled: Whether twist limit is active.
        min_twist_angle: Minimum twist angle (radians).
        max_twist_angle: Maximum twist angle (radians).
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        break_force: float = 0.0,
        break_torque: float = 0.0
    ):
        """
        Initialize ball joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, break_torque
        )

        # Reference axes for limits
        self._local_twist_axis_a = Vec3.unit_x()
        self._local_twist_axis_b = Vec3.unit_x()

        # Cone limit
        self._cone_limit_enabled = False
        self._cone_limit_angle = math.pi / 4  # 45 degrees

        # Twist limit
        self._twist_limit_enabled = False
        self._min_twist_angle = -math.pi
        self._max_twist_angle = math.pi

        # Initialize constraint storage
        # 3 linear + optional cone limit + optional twist limit
        max_rows = 5
        self._jacobians = [Jacobian() for _ in range(max_rows)]
        self._effective_masses = [0.0] * max_rows
        self._biases = [0.0] * max_rows
        self._accumulated_impulse = [0.0] * max_rows
        self._warm_start_impulse = [0.0] * max_rows
        self._lower_limits = [float("-inf")] * max_rows
        self._upper_limits = [float("inf")] * max_rows

        self._active_constraint_count = 3

    @property
    def cone_limit_enabled(self) -> bool:
        """Check if cone limit is enabled."""
        return self._cone_limit_enabled

    @cone_limit_enabled.setter
    def cone_limit_enabled(self, value: bool) -> None:
        """Enable/disable cone limit."""
        self._cone_limit_enabled = value

    @property
    def cone_limit_angle(self) -> float:
        """Get cone limit angle (radians)."""
        return self._cone_limit_angle

    @cone_limit_angle.setter
    def cone_limit_angle(self, value: float) -> None:
        """Set cone limit angle."""
        self._cone_limit_angle = max(0.0, value)

    @property
    def twist_limit_enabled(self) -> bool:
        """Check if twist limit is enabled."""
        return self._twist_limit_enabled

    @twist_limit_enabled.setter
    def twist_limit_enabled(self, value: bool) -> None:
        """Enable/disable twist limit."""
        self._twist_limit_enabled = value

    @property
    def min_twist_angle(self) -> float:
        """Get minimum twist angle."""
        return self._min_twist_angle

    @min_twist_angle.setter
    def min_twist_angle(self, value: float) -> None:
        """Set minimum twist angle."""
        self._min_twist_angle = value

    @property
    def max_twist_angle(self) -> float:
        """Get maximum twist angle."""
        return self._max_twist_angle

    @max_twist_angle.setter
    def max_twist_angle(self, value: float) -> None:
        """Set maximum twist angle."""
        self._max_twist_angle = value

    def set_cone_limit(self, angle: float) -> None:
        """
        Set cone limit.

        Args:
            angle: Maximum angle from reference axis (radians).
        """
        self._cone_limit_angle = max(0.0, angle)
        self._cone_limit_enabled = True

    def set_twist_limits(self, min_angle: float, max_angle: float) -> None:
        """
        Set twist limits.

        Args:
            min_angle: Minimum twist angle (radians).
            max_angle: Maximum twist angle (radians).
        """
        self._min_twist_angle = min_angle
        self._max_twist_angle = max_angle
        self._twist_limit_enabled = True

    def set_twist_axis(self, local_axis_a: Vec3, local_axis_b: Vec3 = None) -> None:
        """
        Set the twist axis for limits.

        Args:
            local_axis_a: Twist axis in body A's local space.
            local_axis_b: Twist axis in body B's local space (defaults to same as A).
        """
        self._local_twist_axis_a = local_axis_a.normalized()
        self._local_twist_axis_b = (local_axis_b or local_axis_a).normalized()

    def get_swing_angle(self) -> float:
        """Get current swing angle (angle between twist axes)."""
        axis_a = self._body_a.local_to_world_direction(self._local_twist_axis_a)

        if self._body_b is not None:
            axis_b = self._body_b.local_to_world_direction(self._local_twist_axis_b)
        else:
            axis_b = self._local_twist_axis_b

        cos_angle = axis_a.dot(axis_b)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    def get_twist_angle(self) -> float:
        """Get current twist angle around twist axis."""
        # Project relative rotation onto twist axis
        if self._body_b is not None:
            q_rel = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_rel = self._body_a.orientation

        # Decompose into swing and twist
        twist_axis = self._local_twist_axis_a

        # Twist component: project quaternion onto twist axis
        twist_component = Vec3(q_rel.x, q_rel.y, q_rel.z).dot(twist_axis)
        twist_w = q_rel.w

        # Compute twist angle
        twist_angle = 2.0 * math.atan2(twist_component, twist_w)

        # Normalize to [-pi, pi]
        if twist_angle > math.pi:
            twist_angle -= 2.0 * math.pi
        elif twist_angle < -math.pi:
            twist_angle += 2.0 * math.pi

        return twist_angle

    def get_constraint_count(self) -> int:
        """Get number of active constraint rows."""
        return self._active_constraint_count

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare ball joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Get anchor positions and r vectors
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        # Position error
        position_error = anchor_b - anchor_a

        # ============ LINEAR CONSTRAINTS (3 rows) ============
        axes = [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]
        errors = [position_error.x, position_error.y, position_error.z]

        for i, (axis, error) in enumerate(zip(axes, errors)):
            self._jacobians[i] = Jacobian(
                linear_a=-axis,
                angular_a=-r_a.cross(axis),
                linear_b=axis,
                angular_b=r_b.cross(axis)
            )
            self._effective_masses[i] = self._compute_effective_mass(self._jacobians[i])
            self._biases[i] = config.baumgarte_factor * error / dt

        self._active_constraint_count = 3

        # ============ CONE LIMIT ============
        if self._cone_limit_enabled:
            swing_angle = self.get_swing_angle()

            if swing_angle > self._cone_limit_angle:
                # Get constraint axis (perpendicular to both twist axes)
                axis_a = self._body_a.local_to_world_direction(self._local_twist_axis_a)

                if self._body_b is not None:
                    axis_b = self._body_b.local_to_world_direction(self._local_twist_axis_b)
                else:
                    axis_b = self._local_twist_axis_b

                # Constraint axis is the swing axis
                swing_axis = axis_a.cross(axis_b)
                swing_length = swing_axis.length()

                if swing_length > 1e-6:
                    swing_axis = swing_axis / swing_length

                    self._jacobians[3] = Jacobian(
                        linear_a=Vec3.zero(),
                        angular_a=-swing_axis,
                        linear_b=Vec3.zero(),
                        angular_b=swing_axis
                    )
                    self._effective_masses[3] = self._compute_effective_mass(
                        self._jacobians[3]
                    )

                    error = swing_angle - self._cone_limit_angle
                    self._biases[3] = config.baumgarte_factor * error / dt
                    self._lower_limits[3] = 0.0
                    self._upper_limits[3] = float("inf")

                    self._active_constraint_count = 4

        # ============ TWIST LIMIT ============
        if self._twist_limit_enabled:
            twist_angle = self.get_twist_angle()
            twist_axis = self._body_a.local_to_world_direction(self._local_twist_axis_a)

            twist_idx = self._active_constraint_count

            # Check lower limit
            if twist_angle < self._min_twist_angle:
                self._jacobians[twist_idx] = Jacobian(
                    linear_a=Vec3.zero(),
                    angular_a=-twist_axis,
                    linear_b=Vec3.zero(),
                    angular_b=twist_axis
                )
                self._effective_masses[twist_idx] = self._compute_effective_mass(
                    self._jacobians[twist_idx]
                )

                error = self._min_twist_angle - twist_angle
                self._biases[twist_idx] = config.baumgarte_factor * error / dt
                self._lower_limits[twist_idx] = 0.0
                self._upper_limits[twist_idx] = float("inf")

                self._active_constraint_count += 1

            # Check upper limit
            elif twist_angle > self._max_twist_angle:
                self._jacobians[twist_idx] = Jacobian(
                    linear_a=Vec3.zero(),
                    angular_a=twist_axis,
                    linear_b=Vec3.zero(),
                    angular_b=-twist_axis
                )
                self._effective_masses[twist_idx] = self._compute_effective_mass(
                    self._jacobians[twist_idx]
                )

                error = twist_angle - self._max_twist_angle
                self._biases[twist_idx] = config.baumgarte_factor * error / dt
                self._lower_limits[twist_idx] = 0.0
                self._upper_limits[twist_idx] = float("inf")

                self._active_constraint_count += 1

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for ball joint."""
        max_error = 0.0

        # Position error
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        position_error = anchor_b - anchor_a
        max_error = position_error.length()

        # Apply corrections
        self._apply_position_corrections(position_error, max_correction)

        return max_error

    def _apply_position_corrections(
        self,
        position_error: Vec3,
        max_correction: float
    ) -> None:
        """Apply position corrections."""
        r_a, r_b = self._get_r_vectors()

        errors = [position_error.x, position_error.y, position_error.z]
        axes = [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]

        for i, (error, axis) in enumerate(zip(errors, axes)):
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

    @classmethod
    def create_at_point(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3,
        cone_limit: float = None,
        twist_limits: tuple = None
    ) -> "BallJoint":
        """
        Create ball joint at a world position.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor: Anchor position in world space.
            cone_limit: Optional cone limit angle (radians).
            twist_limits: Optional (min, max) twist limits (radians).

        Returns:
            Configured BallJoint.
        """
        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)

        joint = cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b
        )

        if cone_limit is not None:
            joint.set_cone_limit(cone_limit)

        if twist_limits is not None:
            joint.set_twist_limits(twist_limits[0], twist_limits[1])

        return joint

    @classmethod
    def create_ragdoll_shoulder(
        cls,
        torso: RigidBody,
        arm: RigidBody,
        world_anchor: Vec3,
        forward_axis: Vec3 = None
    ) -> "BallJoint":
        """
        Create a ball joint configured for a ragdoll shoulder.

        Args:
            torso: Torso rigid body.
            arm: Arm rigid body.
            world_anchor: Shoulder position in world space.
            forward_axis: Forward direction for the character.

        Returns:
            Configured BallJoint with typical shoulder limits.
        """
        joint = cls.create_at_point(torso, arm, world_anchor)

        # Set up twist axis (along the arm)
        arm_direction = (arm.position - world_anchor).normalized()
        local_axis_torso = torso.world_to_local_direction(arm_direction)
        local_axis_arm = arm.world_to_local_direction(arm_direction)

        joint.set_twist_axis(local_axis_torso, local_axis_arm)

        # Typical shoulder limits
        joint.set_cone_limit(math.pi * 0.5)  # 90 degrees cone
        joint.set_twist_limits(-math.pi * 0.5, math.pi * 0.5)  # +/- 90 degrees twist

        return joint
