"""
Base Joint Class for Constraint System.

Provides the abstract base class and common functionality for all joint types.
Joints constrain the relative motion between two rigid bodies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from enum import Enum, auto
import math

# Import from solver module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


class JointState(Enum):
    """State of a joint."""
    ACTIVE = auto()      # Joint is active and constraining
    DISABLED = auto()    # Joint is temporarily disabled
    BROKEN = auto()      # Joint has been broken (exceeded break force)


@dataclass
class JointBreakEvent:
    """Event data when a joint breaks."""
    joint: "Joint"
    break_force: float
    break_torque: float
    applied_force: float
    applied_torque: float
    timestamp: float = 0.0


class Joint(ABC):
    """
    Abstract base class for all joint types.

    A joint constrains the relative motion between two rigid bodies.
    It can have limits, motors, and can break under excessive force.

    Attributes:
        body_a: First rigid body.
        body_b: Second rigid body (or None for world attachment).
        local_anchor_a: Anchor point in body A's local space.
        local_anchor_b: Anchor point in body B's local space (or world space).
        break_force: Force threshold for breaking (0 = unbreakable).
        break_torque: Torque threshold for breaking (0 = unbreakable).
        is_broken: Whether the joint has been broken.
        state: Current state of the joint.
        warm_start_impulse: Cached impulse for warm starting.
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
        Initialize base joint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body (or None for world attachment).
            local_anchor_a: Anchor point in body A's local space.
            local_anchor_b: Anchor point in body B's local space.
            break_force: Force threshold for breaking (0 = unbreakable).
            break_torque: Torque threshold for breaking (0 = unbreakable).
        """
        self._body_a = body_a
        self._body_b = body_b
        self._local_anchor_a = local_anchor_a or Vec3.zero()
        self._local_anchor_b = local_anchor_b or Vec3.zero()
        self._break_force = break_force
        self._break_torque = break_torque
        self._is_broken = False
        self._state = JointState.ACTIVE
        self._warm_start_impulse: List[float] = []
        self._accumulated_impulse: List[float] = []

        # Internal state for solving
        self._jacobians: List[Jacobian] = []
        self._effective_masses: List[float] = []
        self._biases: List[float] = []
        self._lower_limits: List[float] = []
        self._upper_limits: List[float] = []

        # Callbacks
        self._on_break: Optional[Callable[[JointBreakEvent], None]] = None

        # Applied force tracking for break detection
        self._last_applied_force: float = 0.0
        self._last_applied_torque: float = 0.0

    @property
    def body_a(self) -> RigidBody:
        """Get first body."""
        return self._body_a

    @property
    def body_b(self) -> Optional[RigidBody]:
        """Get second body."""
        return self._body_b

    @property
    def local_anchor_a(self) -> Vec3:
        """Get local anchor point on body A."""
        return self._local_anchor_a

    @local_anchor_a.setter
    def local_anchor_a(self, value: Vec3) -> None:
        """Set local anchor point on body A."""
        self._local_anchor_a = value

    @property
    def local_anchor_b(self) -> Vec3:
        """Get local anchor point on body B."""
        return self._local_anchor_b

    @local_anchor_b.setter
    def local_anchor_b(self, value: Vec3) -> None:
        """Set local anchor point on body B."""
        self._local_anchor_b = value

    @property
    def break_force(self) -> float:
        """Get break force threshold."""
        return self._break_force

    @break_force.setter
    def break_force(self, value: float) -> None:
        """Set break force threshold."""
        self._break_force = max(0.0, value)

    @property
    def break_torque(self) -> float:
        """Get break torque threshold."""
        return self._break_torque

    @break_torque.setter
    def break_torque(self, value: float) -> None:
        """Set break torque threshold."""
        self._break_torque = max(0.0, value)

    @property
    def is_broken(self) -> bool:
        """Check if joint is broken."""
        return self._is_broken

    @property
    def state(self) -> JointState:
        """Get joint state."""
        return self._state

    @property
    def constraint_type(self) -> ConstraintType:
        """Get constraint type."""
        return ConstraintType.EQUALITY

    def set_break_callback(
        self,
        callback: Callable[[JointBreakEvent], None]
    ) -> None:
        """Set callback for when joint breaks."""
        self._on_break = callback

    def enable(self) -> None:
        """Enable the joint."""
        if not self._is_broken:
            self._state = JointState.ACTIVE

    def disable(self) -> None:
        """Disable the joint temporarily."""
        self._state = JointState.DISABLED

    def is_enabled(self) -> bool:
        """Check if joint is enabled."""
        return self._state == JointState.ACTIVE

    def get_world_anchor_a(self) -> Vec3:
        """Get anchor point A in world space."""
        return self._body_a.local_to_world(self._local_anchor_a)

    def get_world_anchor_b(self) -> Vec3:
        """Get anchor point B in world space."""
        if self._body_b is not None:
            return self._body_b.local_to_world(self._local_anchor_b)
        return self._local_anchor_b  # World space anchor

    def get_reaction_force(self, inv_dt: float) -> Vec3:
        """
        Get reaction force applied by the joint.

        Args:
            inv_dt: Inverse time step (1/dt).

        Returns:
            Reaction force in world coordinates.
        """
        if not self._accumulated_impulse:
            return Vec3.zero()

        # Sum linear impulses from all constraint rows
        force = Vec3.zero()
        for i, jacobian in enumerate(self._jacobians):
            if i < len(self._accumulated_impulse):
                force = force + jacobian.linear_a * (-self._accumulated_impulse[i] * inv_dt)

        return force

    def get_reaction_torque(self, inv_dt: float) -> Vec3:
        """
        Get reaction torque applied by the joint.

        Args:
            inv_dt: Inverse time step (1/dt).

        Returns:
            Reaction torque in world coordinates.
        """
        if not self._accumulated_impulse:
            return Vec3.zero()

        # Sum angular impulses from all constraint rows
        torque = Vec3.zero()
        for i, jacobian in enumerate(self._jacobians):
            if i < len(self._accumulated_impulse):
                torque = torque + jacobian.angular_a * (-self._accumulated_impulse[i] * inv_dt)

        return torque

    @abstractmethod
    def get_constraint_count(self) -> int:
        """
        Get number of constraint rows.

        Returns:
            Number of scalar constraints.
        """
        pass

    @abstractmethod
    def prepare(self, dt: float, config: SolverConfig) -> None:
        """
        Prepare joint for solving.

        Called once per physics step before iterations.

        Args:
            dt: Time step.
            config: Solver configuration.
        """
        pass

    def warm_start(self, factor: float) -> None:
        """
        Apply warm starting impulse.

        Args:
            factor: Warm start scale factor.
        """
        if not self._warm_start_impulse or not self._jacobians:
            return

        for i, (jacobian, impulse) in enumerate(
            zip(self._jacobians, self._warm_start_impulse)
        ):
            if abs(impulse) < 1e-10:
                continue

            scaled_impulse = impulse * factor

            # Apply to body A
            if not self._body_a.is_static:
                self._body_a.velocity = self._body_a.velocity + jacobian.linear_a * (
                    self._body_a.inv_mass * scaled_impulse
                )
                self._body_a.angular_velocity = self._body_a.angular_velocity + (
                    self._body_a.inv_inertia_world * (jacobian.angular_a * scaled_impulse)
                )

            # Apply to body B
            if self._body_b is not None and not self._body_b.is_static:
                self._body_b.velocity = self._body_b.velocity + jacobian.linear_b * (
                    self._body_b.inv_mass * scaled_impulse
                )
                self._body_b.angular_velocity = self._body_b.angular_velocity + (
                    self._body_b.inv_inertia_world * (jacobian.angular_b * scaled_impulse)
                )

    def solve_velocity(self) -> float:
        """
        Solve velocity constraints.

        Returns:
            Maximum impulse applied.
        """
        if self._state != JointState.ACTIVE:
            return 0.0

        max_impulse = 0.0

        for i in range(len(self._jacobians)):
            impulse = self._solve_velocity_row(i)
            max_impulse = max(max_impulse, abs(impulse))

        # Track applied forces for break detection
        self._update_applied_forces()

        return max_impulse

    def _solve_velocity_row(self, index: int) -> float:
        """
        Solve a single velocity constraint row.

        Args:
            index: Row index.

        Returns:
            Impulse applied.
        """
        jacobian = self._jacobians[index]
        effective_mass = self._effective_masses[index]
        bias = self._biases[index]

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
        impulse = -effective_mass * (cdot + bias)

        # Clamp for inequality constraints
        old_accumulated = self._accumulated_impulse[index]
        new_accumulated = old_accumulated + impulse

        lower = self._lower_limits[index] if index < len(self._lower_limits) else float("-inf")
        upper = self._upper_limits[index] if index < len(self._upper_limits) else float("inf")

        new_accumulated = max(lower, min(upper, new_accumulated))
        impulse = new_accumulated - old_accumulated
        self._accumulated_impulse[index] = new_accumulated

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

    def solve_position(self, max_correction: float) -> float:
        """
        Solve position constraints.

        Args:
            max_correction: Maximum position correction.

        Returns:
            Maximum position error.
        """
        if self._state != JointState.ACTIVE:
            return 0.0

        return self._solve_position_internal(max_correction)

    @abstractmethod
    def _solve_position_internal(self, max_correction: float) -> float:
        """
        Internal position solving implementation.

        Args:
            max_correction: Maximum position correction.

        Returns:
            Maximum position error.
        """
        pass

    def get_cached_impulse(self) -> float:
        """Get total cached impulse magnitude."""
        if not self._accumulated_impulse:
            return 0.0
        return sum(abs(i) for i in self._accumulated_impulse)

    def set_cached_impulse(self, impulse: float) -> None:
        """Set cached impulse (for all rows)."""
        if self._accumulated_impulse:
            self._accumulated_impulse[0] = impulse

    def store_impulses_for_warm_start(self) -> None:
        """Store current impulses for next frame's warm start."""
        self._warm_start_impulse = list(self._accumulated_impulse)

    def _update_applied_forces(self) -> None:
        """Update tracked applied forces for break detection."""
        force = Vec3.zero()
        torque = Vec3.zero()

        for i, jacobian in enumerate(self._jacobians):
            if i < len(self._accumulated_impulse):
                force = force + jacobian.linear_a * self._accumulated_impulse[i]
                torque = torque + jacobian.angular_a * self._accumulated_impulse[i]

        self._last_applied_force = force.length()
        self._last_applied_torque = torque.length()

    def check_break_condition(self, inv_dt: float) -> bool:
        """
        Check if joint should break.

        Args:
            inv_dt: Inverse time step.

        Returns:
            True if joint should break.
        """
        if self._is_broken:
            return False

        force = self._last_applied_force * inv_dt
        torque = self._last_applied_torque * inv_dt

        should_break = False

        if self._break_force > 0 and force > self._break_force:
            should_break = True

        if self._break_torque > 0 and torque > self._break_torque:
            should_break = True

        if should_break:
            self._break(force, torque)

        return should_break

    def _break(self, applied_force: float, applied_torque: float) -> None:
        """
        Break the joint.

        Args:
            applied_force: Force that caused the break.
            applied_torque: Torque that caused the break.
        """
        self._is_broken = True
        self._state = JointState.BROKEN

        # Fire callback
        if self._on_break is not None:
            event = JointBreakEvent(
                joint=self,
                break_force=self._break_force,
                break_torque=self._break_torque,
                applied_force=applied_force,
                applied_torque=applied_torque
            )
            self._on_break(event)

    def reset(self) -> None:
        """Reset joint to initial state."""
        self._is_broken = False
        self._state = JointState.ACTIVE
        self._accumulated_impulse = [0.0] * len(self._accumulated_impulse)
        self._warm_start_impulse = [0.0] * len(self._warm_start_impulse)
        self._last_applied_force = 0.0
        self._last_applied_torque = 0.0

    def _compute_effective_mass(
        self,
        jacobian: Jacobian
    ) -> float:
        """
        Compute effective mass for a Jacobian.

        Args:
            jacobian: Constraint Jacobian.

        Returns:
            Effective mass (inverse of K).
        """
        k = 0.0

        # Body A contribution
        if not self._body_a.is_static:
            k += self._body_a.inv_mass * jacobian.linear_a.dot(jacobian.linear_a)
            ang_contrib = self._body_a.inv_inertia_world * jacobian.angular_a
            k += jacobian.angular_a.dot(ang_contrib)

        # Body B contribution
        if self._body_b is not None and not self._body_b.is_static:
            k += self._body_b.inv_mass * jacobian.linear_b.dot(jacobian.linear_b)
            ang_contrib = self._body_b.inv_inertia_world * jacobian.angular_b
            k += jacobian.angular_b.dot(ang_contrib)

        if k < 1e-10:
            return 0.0

        return 1.0 / k

    def _get_r_vectors(self) -> Tuple[Vec3, Vec3]:
        """
        Get r vectors from body centers to anchors in world space.

        Returns:
            Tuple of (r_a, r_b).
        """
        anchor_a = self.get_world_anchor_a()
        r_a = anchor_a - self._body_a.position

        if self._body_b is not None:
            anchor_b = self.get_world_anchor_b()
            r_b = anchor_b - self._body_b.position
        else:
            r_b = Vec3.zero()

        return r_a, r_b
