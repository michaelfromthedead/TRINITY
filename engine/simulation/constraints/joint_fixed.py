"""
Fixed Joint (Weld Joint) Implementation.

A fixed joint locks all 6 degrees of freedom between two bodies,
effectively welding them together. Both position and orientation
are constrained.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

from .joint_base import Joint, JointState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


class FixedJoint(Joint):
    """
    Fixed joint that locks all 6 DOF between two bodies.

    The joint maintains both the relative position and orientation
    of the two bodies. Useful for creating rigid structures from
    multiple bodies.

    Attributes:
        reference_orientation: Relative orientation to maintain.
        stiffness: Optional softness (0 = infinitely stiff).
        damping: Damping coefficient for soft joints.
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        break_force: float = 0.0,
        break_torque: float = 0.0,
        stiffness: float = 0.0,
        damping: float = 0.0
    ):
        """
        Initialize fixed joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body (or None for world attachment).
            local_anchor_a: Anchor point in body A's local space.
            local_anchor_b: Anchor point in body B's local space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.
            stiffness: Softness parameter (0 = rigid).
            damping: Damping coefficient.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, break_torque
        )

        self.stiffness = stiffness
        self.damping = damping

        # Store reference relative orientation: q_rel = q_b^-1 * q_a
        if body_b is not None:
            self._reference_orientation = (
                body_b.orientation.conjugate() * body_a.orientation
            )
        else:
            self._reference_orientation = body_a.orientation

        # Initialize constraint storage (6 DOF: 3 linear + 3 angular)
        self._jacobians = [Jacobian() for _ in range(6)]
        self._effective_masses = [0.0] * 6
        self._biases = [0.0] * 6
        self._accumulated_impulse = [0.0] * 6
        self._warm_start_impulse = [0.0] * 6
        self._lower_limits = [float("-inf")] * 6
        self._upper_limits = [float("inf")] * 6

        # Compliance for soft joints
        self._gamma = 0.0
        self._softness = 0.0

    @property
    def reference_orientation(self) -> Quaternion:
        """Get reference relative orientation."""
        return self._reference_orientation

    @reference_orientation.setter
    def reference_orientation(self, value: Quaternion) -> None:
        """Set reference relative orientation."""
        self._reference_orientation = value

    def get_constraint_count(self) -> int:
        """Get number of constraint rows (6 for fixed joint)."""
        return 6

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare fixed joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        # Update world inertia tensors
        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Compute compliance for soft joints
        if self.stiffness > 0:
            omega = math.sqrt(self.stiffness)
            d = 2.0 * self.damping * omega
            self._gamma = 1.0 / (dt * (d + dt * self.stiffness))
            self._softness = self._gamma / (dt * self.stiffness)
        else:
            self._gamma = 0.0
            self._softness = 0.0

        # Get anchor positions and r vectors
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        # Position error
        position_error = anchor_b - anchor_a

        # Angular error: compute relative orientation error
        if self._body_b is not None:
            q_current = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_current = self._body_a.orientation

        # Error quaternion: q_error = q_current * q_ref^-1
        q_error = q_current * self._reference_orientation.conjugate()

        # Ensure shortest path
        if q_error.w < 0:
            q_error = Quaternion(-q_error.x, -q_error.y, -q_error.z, -q_error.w)

        # Extract rotation vector from quaternion (angle * axis for small angles)
        # For small angles: theta * axis ~= 2 * q.xyz
        angular_error = Vec3(q_error.x, q_error.y, q_error.z) * 2.0

        # Set up Jacobians for linear constraints (3 rows)
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

            if self._gamma > 0:
                self._biases[i] += self._gamma * self._accumulated_impulse[i]
                self._effective_masses[i] = 1.0 / (
                    1.0 / self._effective_masses[i] + self._softness
                )

        # Set up Jacobians for angular constraints (3 rows)
        angular_errors = [angular_error.x, angular_error.y, angular_error.z]

        for i, (axis, error) in enumerate(zip(axes, angular_errors)):
            idx = i + 3
            self._jacobians[idx] = Jacobian(
                linear_a=Vec3.zero(),
                angular_a=-axis,
                linear_b=Vec3.zero(),
                angular_b=axis
            )
            self._effective_masses[idx] = self._compute_effective_mass(self._jacobians[idx])
            self._biases[idx] = config.baumgarte_factor * error / dt

            if self._gamma > 0:
                self._biases[idx] += self._gamma * self._accumulated_impulse[idx]
                self._effective_masses[idx] = 1.0 / (
                    1.0 / self._effective_masses[idx] + self._softness
                )

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for fixed joint."""
        if self._state != JointState.ACTIVE:
            return 0.0

        max_error = 0.0

        # Position error
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        position_error = anchor_b - anchor_a

        # Angular error
        if self._body_b is not None:
            q_current = self._body_b.orientation.conjugate() * self._body_a.orientation
        else:
            q_current = self._body_a.orientation

        q_error = q_current * self._reference_orientation.conjugate()
        if q_error.w < 0:
            q_error = Quaternion(-q_error.x, -q_error.y, -q_error.z, -q_error.w)

        angular_error = Vec3(q_error.x, q_error.y, q_error.z) * 2.0

        # Solve linear position constraints
        position_errors = [position_error.x, position_error.y, position_error.z]
        for i, error in enumerate(position_errors):
            if abs(error) > max_error:
                max_error = abs(error)

            correction = self._solve_position_row(i, error, max_correction)

        # Solve angular position constraints
        angular_errors = [angular_error.x, angular_error.y, angular_error.z]
        for i, error in enumerate(angular_errors):
            idx = i + 3
            if abs(error) > max_error:
                max_error = abs(error)

            correction = self._solve_position_row(idx, error, max_correction)

        return max_error

    def _solve_position_row(
        self,
        index: int,
        error: float,
        max_correction: float
    ) -> float:
        """
        Solve a single position constraint row.

        Args:
            index: Row index.
            error: Position error.
            max_correction: Maximum correction.

        Returns:
            Applied correction.
        """
        jacobian = self._jacobians[index]
        k = self._effective_masses[index]

        if k == 0:
            return 0.0

        # Clamp error
        c = max(-max_correction, min(max_correction, error))

        # Compute impulse
        impulse = -k * c * 0.2  # Position correction factor

        # Apply to body A
        if not self._body_a.is_static:
            self._body_a.position = self._body_a.position + jacobian.linear_a * (
                self._body_a.inv_mass * impulse
            )

            if jacobian.angular_a.length_squared() > 1e-10:
                delta_ang = self._body_a.inv_inertia_world * (jacobian.angular_a * impulse)
                angle = delta_ang.length()
                if angle > 1e-10:
                    axis = delta_ang / angle
                    delta_q = Quaternion.from_axis_angle(axis, angle)
                    self._body_a.orientation = (delta_q * self._body_a.orientation).normalized()

        # Apply to body B
        if self._body_b is not None and not self._body_b.is_static:
            self._body_b.position = self._body_b.position + jacobian.linear_b * (
                self._body_b.inv_mass * impulse
            )

            if jacobian.angular_b.length_squared() > 1e-10:
                delta_ang = self._body_b.inv_inertia_world * (jacobian.angular_b * impulse)
                angle = delta_ang.length()
                if angle > 1e-10:
                    axis = delta_ang / angle
                    delta_q = Quaternion.from_axis_angle(axis, angle)
                    self._body_b.orientation = (delta_q * self._body_b.orientation).normalized()

        return impulse

    def set_reference_from_current(self) -> None:
        """Set reference orientation from current body orientations."""
        if self._body_b is not None:
            self._reference_orientation = (
                self._body_b.orientation.conjugate() * self._body_a.orientation
            )
        else:
            self._reference_orientation = self._body_a.orientation

    def set_reference_angle(self, angle: float, axis: Vec3) -> None:
        """
        Set reference orientation from angle-axis.

        Args:
            angle: Rotation angle in radians.
            axis: Rotation axis (normalized).
        """
        self._reference_orientation = Quaternion.from_axis_angle(axis, angle)

    @classmethod
    def create_weld(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3 = None,
        break_force: float = 0.0,
        break_torque: float = 0.0
    ) -> "FixedJoint":
        """
        Create a weld joint at a world position.

        The joint anchors are computed from the world anchor position.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor: Anchor position in world space.
            break_force: Force threshold for breaking.
            break_torque: Torque threshold for breaking.

        Returns:
            Configured FixedJoint.
        """
        if world_anchor is None:
            # Use midpoint between body centers
            world_anchor = (body_a.position + body_b.position) * 0.5

        local_anchor_a = body_a.world_to_local(world_anchor)
        local_anchor_b = body_b.world_to_local(world_anchor)

        return cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            break_force=break_force,
            break_torque=break_torque
        )

    @classmethod
    def create_soft_weld(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor: Vec3 = None,
        stiffness: float = 1000.0,
        damping: float = 10.0
    ) -> "FixedJoint":
        """
        Create a soft weld joint.

        Soft welds allow some flexibility, useful for deformable structures.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor: Anchor position in world space.
            stiffness: Joint stiffness.
            damping: Joint damping.

        Returns:
            Configured soft FixedJoint.
        """
        joint = cls.create_weld(body_a, body_b, world_anchor)
        joint.stiffness = stiffness
        joint.damping = damping
        return joint
