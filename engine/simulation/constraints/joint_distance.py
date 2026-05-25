"""
Distance Joint Implementation.

A distance joint maintains a fixed distance between two anchor points.
Unlike a spring joint, it's a hard constraint with no compliance.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .joint_base import Joint, JointState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


class DistanceJoint(Joint):
    """
    Distance joint maintaining fixed distance between anchors.

    This is a hard constraint that keeps the distance between two
    anchor points constant. Can be configured with min/max distance
    for rope-like behavior.

    Attributes:
        rest_length: Target distance to maintain.
        min_distance: Minimum allowed distance (for rope behavior).
        max_distance: Maximum allowed distance (for rod behavior).
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        rest_length: float = None,
        break_force: float = 0.0
    ):
        """
        Initialize distance joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            rest_length: Target distance (computed from initial positions if None).
            break_force: Force threshold for breaking.
        """
        super().__init__(
            body_a, body_b,
            local_anchor_a, local_anchor_b,
            break_force, 0.0
        )

        # Compute rest length from initial configuration if not specified
        if rest_length is None:
            anchor_a = self.get_world_anchor_a()
            anchor_b = self.get_world_anchor_b()
            self._rest_length = (anchor_b - anchor_a).length()
        else:
            self._rest_length = rest_length

        # Optional min/max for inequality constraint behavior
        self._min_distance: Optional[float] = None
        self._max_distance: Optional[float] = None

        # Constraint mode
        self._constraint_mode = "equality"  # "equality", "min", "max", "range"

        # Initialize constraint storage
        self._jacobians = [Jacobian()]
        self._effective_masses = [0.0]
        self._biases = [0.0]
        self._accumulated_impulse = [0.0]
        self._warm_start_impulse = [0.0]
        self._lower_limits = [float("-inf")]
        self._upper_limits = [float("inf")]

        # Internal state
        self._current_length = 0.0
        self._is_active_constraint = True

    @property
    def rest_length(self) -> float:
        """Get target distance."""
        return self._rest_length

    @rest_length.setter
    def rest_length(self, value: float) -> None:
        """Set target distance."""
        self._rest_length = max(0.0, value)
        self._update_constraint_mode()

    @property
    def min_distance(self) -> Optional[float]:
        """Get minimum distance."""
        return self._min_distance

    @min_distance.setter
    def min_distance(self, value: Optional[float]) -> None:
        """Set minimum distance."""
        self._min_distance = value
        self._update_constraint_mode()

    @property
    def max_distance(self) -> Optional[float]:
        """Get maximum distance."""
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: Optional[float]) -> None:
        """Set maximum distance."""
        self._max_distance = value
        self._update_constraint_mode()

    @property
    def current_length(self) -> float:
        """Get current distance between anchors."""
        return self._current_length

    def _update_constraint_mode(self) -> None:
        """Update constraint mode based on configured limits."""
        if self._min_distance is not None and self._max_distance is not None:
            self._constraint_mode = "range"
        elif self._min_distance is not None:
            self._constraint_mode = "min"
        elif self._max_distance is not None:
            self._constraint_mode = "max"
        else:
            self._constraint_mode = "equality"

    def set_as_rope(self, max_length: float) -> None:
        """
        Configure as a rope (only constrains maximum distance).

        Args:
            max_length: Maximum rope length.
        """
        self._rest_length = max_length
        self._min_distance = None
        self._max_distance = max_length
        self._constraint_mode = "max"

    def set_as_rod(self, length: float) -> None:
        """
        Configure as a rigid rod (fixed distance).

        Args:
            length: Rod length.
        """
        self._rest_length = length
        self._min_distance = None
        self._max_distance = None
        self._constraint_mode = "equality"

    def set_range(self, min_distance: float, max_distance: float) -> None:
        """
        Configure with distance range.

        Args:
            min_distance: Minimum allowed distance.
            max_distance: Maximum allowed distance.
        """
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._rest_length = (min_distance + max_distance) / 2
        self._constraint_mode = "range"

    def get_constraint_count(self) -> int:
        """Get number of constraint rows."""
        return 1

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare distance joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Get anchor positions
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        # Compute direction and length
        d = anchor_b - anchor_a
        self._current_length = d.length()

        if self._current_length < 1e-6:
            direction = Vec3.unit_x()
            self._current_length = 0.0
        else:
            direction = d / self._current_length

        # Determine if constraint is active and compute error
        self._is_active_constraint = True
        error = 0.0

        if self._constraint_mode == "equality":
            error = self._current_length - self._rest_length
            self._lower_limits[0] = float("-inf")
            self._upper_limits[0] = float("inf")

        elif self._constraint_mode == "max":
            if self._current_length <= self._max_distance:
                self._is_active_constraint = False
                error = 0.0
            else:
                error = self._current_length - self._max_distance
            self._lower_limits[0] = 0.0
            self._upper_limits[0] = float("inf")

        elif self._constraint_mode == "min":
            if self._current_length >= self._min_distance:
                self._is_active_constraint = False
                error = 0.0
            else:
                error = self._min_distance - self._current_length
                direction = -direction  # Push apart
            self._lower_limits[0] = 0.0
            self._upper_limits[0] = float("inf")

        elif self._constraint_mode == "range":
            if self._current_length > self._max_distance:
                error = self._current_length - self._max_distance
                self._lower_limits[0] = 0.0
                self._upper_limits[0] = float("inf")
            elif self._current_length < self._min_distance:
                error = self._min_distance - self._current_length
                direction = -direction
                self._lower_limits[0] = 0.0
                self._upper_limits[0] = float("inf")
            else:
                self._is_active_constraint = False
                error = 0.0

        if not self._is_active_constraint:
            self._effective_masses[0] = 0.0
            self._biases[0] = 0.0
            return

        # Set up Jacobian
        self._jacobians[0] = Jacobian(
            linear_a=-direction,
            angular_a=-r_a.cross(direction),
            linear_b=direction,
            angular_b=r_b.cross(direction)
        )

        # Compute effective mass
        self._effective_masses[0] = self._compute_effective_mass(self._jacobians[0])

        # Baumgarte stabilization bias
        baumgarte = config.baumgarte_factor
        slop = config.slop

        # Apply slop for inequality constraints
        if self._constraint_mode != "equality":
            error = max(0.0, error - slop)

        self._biases[0] = baumgarte * error / dt

    def solve_velocity(self) -> float:
        """Solve velocity constraint."""
        if self._state != JointState.ACTIVE or not self._is_active_constraint:
            return 0.0

        if self._effective_masses[0] == 0:
            return 0.0

        jacobian = self._jacobians[0]

        # Get velocities
        vel_b = self._body_b.velocity if self._body_b else Vec3.zero()
        ang_vel_b = self._body_b.angular_velocity if self._body_b else Vec3.zero()

        # Compute constraint velocity
        cdot = (
            jacobian.linear_a.dot(self._body_a.velocity) +
            jacobian.angular_a.dot(self._body_a.angular_velocity) +
            jacobian.linear_b.dot(vel_b) +
            jacobian.angular_b.dot(ang_vel_b)
        )

        # Compute impulse
        impulse = -self._effective_masses[0] * (cdot + self._biases[0])

        # Clamp for inequality constraints
        old_accumulated = self._accumulated_impulse[0]
        new_accumulated = old_accumulated + impulse
        new_accumulated = max(self._lower_limits[0], min(self._upper_limits[0], new_accumulated))
        impulse = new_accumulated - old_accumulated
        self._accumulated_impulse[0] = new_accumulated

        # Apply impulse
        if not self._body_a.is_static:
            self._body_a.velocity = self._body_a.velocity + jacobian.linear_a * (
                self._body_a.inv_mass * impulse
            )
            self._body_a.angular_velocity = self._body_a.angular_velocity + (
                self._body_a.inv_inertia_world * (jacobian.angular_a * impulse)
            )

        if self._body_b is not None and not self._body_b.is_static:
            self._body_b.velocity = self._body_b.velocity + jacobian.linear_b * (
                self._body_b.inv_mass * impulse
            )
            self._body_b.angular_velocity = self._body_b.angular_velocity + (
                self._body_b.inv_inertia_world * (jacobian.angular_b * impulse)
            )

        return impulse

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraint."""
        if self._state != JointState.ACTIVE:
            return 0.0

        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()

        d = anchor_b - anchor_a
        length = d.length()

        if length < 1e-6:
            return 0.0

        direction = d / length

        # Compute position error
        error = 0.0
        target_length = self._rest_length

        if self._constraint_mode == "equality":
            error = length - target_length
        elif self._constraint_mode == "max":
            if length > self._max_distance:
                error = length - self._max_distance
        elif self._constraint_mode == "min":
            if length < self._min_distance:
                error = self._min_distance - length
                direction = -direction
        elif self._constraint_mode == "range":
            if length > self._max_distance:
                error = length - self._max_distance
            elif length < self._min_distance:
                error = self._min_distance - length
                direction = -direction

        if abs(error) < 1e-6:
            return 0.0

        # Clamp correction
        correction = max(-max_correction, min(max_correction, error))

        # Compute effective mass
        total_inv_mass = 0.0
        if not self._body_a.is_static:
            total_inv_mass += self._body_a.inv_mass
        if self._body_b is not None and not self._body_b.is_static:
            total_inv_mass += self._body_b.inv_mass

        if total_inv_mass < 1e-10:
            return abs(error)

        # Apply correction
        if not self._body_a.is_static:
            self._body_a.position = self._body_a.position - direction * (
                correction * self._body_a.inv_mass / total_inv_mass
            )

        if self._body_b is not None and not self._body_b.is_static:
            self._body_b.position = self._body_b.position + direction * (
                correction * self._body_b.inv_mass / total_inv_mass
            )

        return abs(error)

    @classmethod
    def create_at_points(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        length: float = None
    ) -> "DistanceJoint":
        """
        Create distance joint between two world points.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor_a: Anchor on body A in world space.
            world_anchor_b: Anchor on body B in world space.
            length: Target distance (defaults to current distance).

        Returns:
            Configured DistanceJoint.
        """
        local_anchor_a = body_a.world_to_local(world_anchor_a)
        local_anchor_b = body_b.world_to_local(world_anchor_b)

        return cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            rest_length=length
        )

    @classmethod
    def create_rope(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        max_length: float
    ) -> "DistanceJoint":
        """
        Create a rope constraint.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor_a: Anchor on body A.
            world_anchor_b: Anchor on body B.
            max_length: Maximum rope length.

        Returns:
            DistanceJoint configured as rope.
        """
        joint = cls.create_at_points(body_a, body_b, world_anchor_a, world_anchor_b)
        joint.set_as_rope(max_length)
        return joint
