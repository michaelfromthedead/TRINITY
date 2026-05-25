"""
Spring Joint Implementation.

A spring joint creates a soft distance constraint between two bodies
using spring-damper dynamics. It applies forces based on displacement
from rest length and relative velocity.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .joint_base import Joint, JointState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


class SpringJoint(Joint):
    """
    Spring joint with configurable stiffness and damping.

    Creates a soft distance constraint using Hooke's law:
    F = -k * (x - rest_length) - c * v

    Where k is stiffness, c is damping, x is current length,
    and v is relative velocity along the spring axis.

    Attributes:
        rest_length: Natural length of the spring.
        stiffness: Spring constant (force per unit displacement).
        damping: Damping coefficient (force per unit velocity).
        min_length: Minimum allowed length (optional).
        max_length: Maximum allowed length (optional).
        frequency: Natural frequency (alternative to stiffness).
        damping_ratio: Damping ratio (alternative to damping).
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        rest_length: float = None,
        stiffness: float = 100.0,
        damping: float = 1.0,
        break_force: float = 0.0
    ):
        """
        Initialize spring joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body.
            local_anchor_a: Anchor in body A's local space.
            local_anchor_b: Anchor in body B's local space.
            rest_length: Rest length (computed from initial positions if None).
            stiffness: Spring stiffness.
            damping: Damping coefficient.
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

        self._stiffness = stiffness
        self._damping = damping

        # Optional length limits
        self._min_length: Optional[float] = None
        self._max_length: Optional[float] = None

        # Alternative parameters (frequency and damping ratio)
        self._use_frequency_params = False
        self._frequency = 0.0
        self._damping_ratio = 0.0

        # Soft constraint type
        self._constraint_type = ConstraintType.SOFT

        # Initialize constraint storage (1 row for spring)
        self._jacobians = [Jacobian()]
        self._effective_masses = [0.0]
        self._biases = [0.0]
        self._accumulated_impulse = [0.0]
        self._warm_start_impulse = [0.0]
        self._lower_limits = [float("-inf")]
        self._upper_limits = [float("inf")]

        # Internal state
        self._gamma = 0.0
        self._beta = 0.0
        self._current_length = 0.0
        self._spring_force = 0.0

    @property
    def rest_length(self) -> float:
        """Get rest length."""
        return self._rest_length

    @rest_length.setter
    def rest_length(self, value: float) -> None:
        """Set rest length."""
        self._rest_length = max(0.0, value)

    @property
    def stiffness(self) -> float:
        """Get spring stiffness."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: float) -> None:
        """Set spring stiffness."""
        self._stiffness = max(0.0, value)
        self._use_frequency_params = False

    @property
    def damping(self) -> float:
        """Get damping coefficient."""
        return self._damping

    @damping.setter
    def damping(self, value: float) -> None:
        """Set damping coefficient."""
        self._damping = max(0.0, value)
        self._use_frequency_params = False

    @property
    def min_length(self) -> Optional[float]:
        """Get minimum length limit."""
        return self._min_length

    @min_length.setter
    def min_length(self, value: Optional[float]) -> None:
        """Set minimum length limit."""
        self._min_length = value

    @property
    def max_length(self) -> Optional[float]:
        """Get maximum length limit."""
        return self._max_length

    @max_length.setter
    def max_length(self, value: Optional[float]) -> None:
        """Set maximum length limit."""
        self._max_length = value

    @property
    def frequency(self) -> float:
        """Get natural frequency (Hz)."""
        return self._frequency

    @property
    def damping_ratio(self) -> float:
        """Get damping ratio (0 = undamped, 1 = critically damped)."""
        return self._damping_ratio

    @property
    def current_length(self) -> float:
        """Get current spring length."""
        return self._current_length

    @property
    def spring_force(self) -> float:
        """Get current spring force magnitude."""
        return self._spring_force

    def set_frequency_damping(
        self,
        frequency: float,
        damping_ratio: float
    ) -> None:
        """
        Set spring parameters using frequency and damping ratio.

        Args:
            frequency: Natural frequency in Hz.
            damping_ratio: Damping ratio (0-1, where 1 is critical damping).
        """
        self._frequency = max(0.0, frequency)
        self._damping_ratio = max(0.0, min(1.0, damping_ratio))
        self._use_frequency_params = True

        # Convert to stiffness and damping
        # omega = 2 * pi * frequency
        # k = m * omega^2
        # c = 2 * m * damping_ratio * omega
        # Since we don't know effective mass yet, store params and compute later

    def set_length_limits(
        self,
        min_length: float = None,
        max_length: float = None
    ) -> None:
        """
        Set optional length limits.

        Args:
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.
        """
        self._min_length = min_length
        self._max_length = max_length

    def get_constraint_count(self) -> int:
        """Get number of constraint rows (1 for spring)."""
        return 1

    def get_displacement(self) -> float:
        """Get current displacement from rest length."""
        return self._current_length - self._rest_length

    def get_relative_velocity(self) -> float:
        """Get relative velocity along spring axis."""
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()

        d = anchor_b - anchor_a
        length = d.length()

        if length < 1e-6:
            return 0.0

        direction = d / length

        vel_a = self._body_a.get_velocity_at_point(anchor_a)
        if self._body_b is not None:
            vel_b = self._body_b.get_velocity_at_point(anchor_b)
        else:
            vel_b = Vec3.zero()

        rel_vel = vel_b - vel_a
        return rel_vel.dot(direction)

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare spring joint for solving."""
        if self._state != JointState.ACTIVE:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        # Get anchor positions
        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        r_a, r_b = self._get_r_vectors()

        # Compute spring direction and length
        d = anchor_b - anchor_a
        self._current_length = d.length()

        if self._current_length < 1e-6:
            # Bodies are coincident, use arbitrary direction
            direction = Vec3.unit_x()
            self._current_length = 0.0
        else:
            direction = d / self._current_length

        # Apply length limits
        target_length = self._rest_length
        if self._min_length is not None and self._current_length < self._min_length:
            target_length = self._min_length
        elif self._max_length is not None and self._current_length > self._max_length:
            target_length = self._max_length

        # Set up Jacobian
        self._jacobians[0] = Jacobian(
            linear_a=-direction,
            angular_a=-r_a.cross(direction),
            linear_b=direction,
            angular_b=r_b.cross(direction)
        )

        # Compute effective mass
        k = 0.0
        if not self._body_a.is_static:
            k += self._body_a.inv_mass
            ang_contrib = self._body_a.inv_inertia_world * self._jacobians[0].angular_a
            k += self._jacobians[0].angular_a.dot(ang_contrib)

        if self._body_b is not None and not self._body_b.is_static:
            k += self._body_b.inv_mass
            ang_contrib = self._body_b.inv_inertia_world * self._jacobians[0].angular_b
            k += self._jacobians[0].angular_b.dot(ang_contrib)

        if k < 1e-10:
            self._effective_masses[0] = 0.0
            return

        effective_mass = 1.0 / k

        # Compute spring parameters
        if self._use_frequency_params:
            omega = 2.0 * math.pi * self._frequency
            stiffness = effective_mass * omega * omega
            damping = 2.0 * effective_mass * self._damping_ratio * omega
        else:
            stiffness = self._stiffness
            damping = self._damping

        # Soft constraint coefficients
        # gamma = 1 / (c + dt * k)
        # beta = dt * k / (c + dt * k)
        c_plus_dtk = damping + dt * stiffness
        if c_plus_dtk > 1e-10:
            self._gamma = 1.0 / c_plus_dtk
            self._beta = dt * stiffness * self._gamma
        else:
            self._gamma = 0.0
            self._beta = 0.0

        # Position error
        displacement = self._current_length - target_length

        # Bias velocity
        self._biases[0] = self._beta * displacement / dt

        # Modified effective mass for soft constraint
        self._effective_masses[0] = 1.0 / (k + self._gamma)

        # Track spring force
        self._spring_force = stiffness * displacement

    def solve_velocity(self) -> float:
        """Solve spring constraint."""
        if self._state != JointState.ACTIVE:
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

        # Soft constraint impulse with gamma term
        impulse = -self._effective_masses[0] * (
            cdot + self._biases[0] + self._gamma * self._accumulated_impulse[0]
        )

        self._accumulated_impulse[0] += impulse

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
        """Position solving for spring joints (minimal)."""
        # Spring joints are primarily velocity-based
        # Only apply position correction if at length limits
        if self._min_length is None and self._max_length is None:
            return 0.0

        anchor_a = self.get_world_anchor_a()
        anchor_b = self.get_world_anchor_b()
        d = anchor_b - anchor_a
        length = d.length()

        error = 0.0
        if self._min_length is not None and length < self._min_length:
            error = self._min_length - length
        elif self._max_length is not None and length > self._max_length:
            error = length - self._max_length

        if abs(error) < 1e-6:
            return 0.0

        if length < 1e-6:
            return error

        direction = d / length

        # Compute correction
        correction = min(max_correction, abs(error)) * (1.0 if error > 0 else -1.0)

        # Apply correction
        total_inv_mass = self._body_a.inv_mass
        if self._body_b is not None:
            total_inv_mass += self._body_b.inv_mass

        if total_inv_mass < 1e-10:
            return error

        if not self._body_a.is_static:
            self._body_a.position = self._body_a.position - direction * (
                correction * self._body_a.inv_mass / total_inv_mass
            )

        if self._body_b is not None and not self._body_b.is_static:
            self._body_b.position = self._body_b.position + direction * (
                correction * self._body_b.inv_mass / total_inv_mass
            )

        return error

    @classmethod
    def create_between_points(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        stiffness: float = 100.0,
        damping: float = 1.0
    ) -> "SpringJoint":
        """
        Create spring joint between two world points.

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor_a: Anchor point on body A in world space.
            world_anchor_b: Anchor point on body B in world space.
            stiffness: Spring stiffness.
            damping: Damping coefficient.

        Returns:
            Configured SpringJoint.
        """
        local_anchor_a = body_a.world_to_local(world_anchor_a)
        local_anchor_b = body_b.world_to_local(world_anchor_b)

        return cls(
            body_a=body_a,
            body_b=body_b,
            local_anchor_a=local_anchor_a,
            local_anchor_b=local_anchor_b,
            stiffness=stiffness,
            damping=damping
        )

    @classmethod
    def create_bungee(
        cls,
        body_a: RigidBody,
        body_b: RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        stiffness: float = 100.0,
        damping: float = 1.0
    ) -> "SpringJoint":
        """
        Create a bungee-style spring (only pulls, no push).

        Args:
            body_a: First body.
            body_b: Second body.
            world_anchor_a: Anchor point on body A.
            world_anchor_b: Anchor point on body B.
            stiffness: Spring stiffness.
            damping: Damping coefficient.

        Returns:
            Configured SpringJoint that only applies tension.
        """
        joint = cls.create_between_points(
            body_a, body_b,
            world_anchor_a, world_anchor_b,
            stiffness, damping
        )
        joint.min_length = joint.rest_length
        return joint
