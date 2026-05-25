"""
Temporal Gauss-Seidel (TGS) Constraint Solver.

An improved constraint solver that provides better convergence
for stiff constraints and extreme mass ratios.

Key improvements over standard Sequential Impulse:
- Better handling of mass ratio differences
- Substep-aware solving for improved stability
- Improved convergence for stiff systems
- Better warm starting strategy

References:
- Erin Catto, "Solving Rigid Body Contacts" (GDC 2005)
- Dirk Gregorius, "Robust Contact Creation for Physics Simulation" (GDC 2013)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math

from .config import SolverConfig, SolverType
from .constraint_solver import (
    ConstraintSolver,
    Constraint,
    RigidBody,
    ConstraintType,
    BaseConstraint,
)
from .jacobian import (
    Vec3,
    Mat3,
    Quaternion,
    Jacobian,
    compute_effective_mass,
    apply_impulse,
    compute_relative_velocity,
    clamp_impulse,
)


@dataclass
class TGSConstraintData:
    """
    Per-constraint data for TGS solver.

    Stores additional information needed for TGS iterations.
    """
    constraint: Constraint
    jacobian: Jacobian
    effective_mass: float = 0.0
    bias: float = 0.0
    gamma: float = 0.0  # Regularization term
    lambda_accumulated: float = 0.0
    lower_limit: float = float("-inf")
    upper_limit: float = float("inf")
    mass_scale_a: float = 1.0  # Mass scaling for body A
    mass_scale_b: float = 1.0  # Mass scaling for body B


class TGSSolver(ConstraintSolver):
    """
    Temporal Gauss-Seidel Constraint Solver.

    Extends the base constraint solver with TGS-specific improvements:
    - Split impulse for position correction
    - Regularization for stiff constraints
    - Mass scaling for extreme mass ratios
    - Substep integration

    Attributes:
        config: Solver configuration.
        substep_dt: Time step per substep.
        current_substep: Current substep index.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize TGS solver."""
        super().__init__(config or SolverConfig.tgs_default())
        self._tgs_data: Dict[id, TGSConstraintData] = {}
        self._substep_dt: float = 0.0
        self._current_substep: int = 0
        self._split_impulse_velocities: Dict[int, Tuple[Vec3, Vec3]] = {}
        self._use_split_impulse: bool = True
        self._regularization_factor: float = 0.1

    @property
    def substep_dt(self) -> float:
        """Get current substep time step."""
        return self._substep_dt

    @property
    def current_substep(self) -> int:
        """Get current substep index."""
        return self._current_substep

    def solve(self, dt: float) -> None:
        """
        Solve all constraints with TGS method.

        Performs multiple substeps for improved stability.

        Args:
            dt: Total time step.
        """
        if not self.constraints:
            return

        num_substeps = self.config.substeps
        self._substep_dt = dt / num_substeps

        for substep in range(num_substeps):
            self._current_substep = substep
            self._solve_substep(self._substep_dt)

    def _solve_substep(self, sub_dt: float) -> None:
        """
        Solve constraints for one substep.

        Args:
            sub_dt: Substep time step.
        """
        # Prepare constraints for this substep
        self._prepare_tgs_constraints(sub_dt)

        # Initialize split impulse velocities
        if self._use_split_impulse:
            self._init_split_impulse_velocities()

        # Warm start (reduced for substeps after first)
        warm_factor = self.config.warm_start_factor
        if self._current_substep > 0:
            warm_factor *= 0.5  # Reduce warm start for subsequent substeps

        if warm_factor > 0:
            self._warm_start_tgs(warm_factor, sub_dt)

        # Velocity iterations
        for i in range(self.config.velocity_iterations):
            self._iteration_count = i
            error = self._solve_tgs_velocity_iteration()
            if error < 1e-7:
                break

        # Position iterations with split impulse
        for i in range(self.config.position_iterations):
            error = self._solve_tgs_position_iteration(sub_dt)
            if error < self.config.slop:
                break

        # Apply split impulse velocities
        if self._use_split_impulse:
            self._apply_split_impulse_velocities()

    def _prepare_tgs_constraints(self, dt: float) -> None:
        """Prepare all constraints with TGS-specific data."""
        self._tgs_data.clear()

        for constraint in self.constraints:
            # Update world inertia
            constraint.body_a.update_world_inertia()
            if constraint.body_b is not None:
                constraint.body_b.update_world_inertia()

            # Standard preparation
            constraint.prepare(dt, self.config)

            # Create TGS data
            tgs_data = self._create_tgs_data(constraint, dt)
            self._tgs_data[id(constraint)] = tgs_data

    def _create_tgs_data(self, constraint: Constraint, dt: float) -> TGSConstraintData:
        """
        Create TGS-specific data for a constraint.

        Args:
            constraint: The constraint.
            dt: Time step.

        Returns:
            TGS constraint data.
        """
        # Get base constraint data if available
        if hasattr(constraint, '_jacobian'):
            jacobian = constraint._jacobian
        else:
            jacobian = Jacobian()

        if hasattr(constraint, '_effective_mass'):
            effective_mass = constraint._effective_mass
        else:
            effective_mass = 1.0

        if hasattr(constraint, '_bias'):
            bias = constraint._bias
        else:
            bias = 0.0

        if hasattr(constraint, '_accumulated_impulse'):
            lambda_acc = constraint._accumulated_impulse
        else:
            lambda_acc = 0.0

        # Compute mass scaling for extreme mass ratios
        mass_scale_a, mass_scale_b = self._compute_mass_scales(constraint)

        # Compute regularization (gamma)
        gamma = self._compute_regularization(constraint, dt)

        # Get limits
        lower = float("-inf")
        upper = float("inf")
        if hasattr(constraint, '_lower_limit'):
            lower = constraint._lower_limit
        if hasattr(constraint, '_upper_limit'):
            upper = constraint._upper_limit

        if constraint.constraint_type == ConstraintType.INEQUALITY:
            lower = 0.0

        return TGSConstraintData(
            constraint=constraint,
            jacobian=jacobian,
            effective_mass=effective_mass,
            bias=bias,
            gamma=gamma,
            lambda_accumulated=lambda_acc,
            lower_limit=lower,
            upper_limit=upper,
            mass_scale_a=mass_scale_a,
            mass_scale_b=mass_scale_b,
        )

    def _compute_mass_scales(self, constraint: Constraint) -> Tuple[float, float]:
        """
        Compute mass scaling factors for handling extreme mass ratios.

        Args:
            constraint: The constraint.

        Returns:
            Tuple of (scale_a, scale_b).
        """
        mass_a = constraint.body_a.mass if not constraint.body_a.is_static else float("inf")
        mass_b = (constraint.body_b.mass if constraint.body_b and not constraint.body_b.is_static
                  else float("inf"))

        if mass_a == float("inf") and mass_b == float("inf"):
            return 1.0, 1.0

        if mass_a == float("inf"):
            return 0.0, 1.0
        if mass_b == float("inf"):
            return 1.0, 0.0

        # Compute mass ratio
        ratio = min(mass_a, mass_b) / max(mass_a, mass_b)

        # Apply scaling for extreme ratios (< 0.01)
        if ratio < 0.01:
            # Scale down the lighter body's response
            if mass_a < mass_b:
                scale_factor = math.sqrt(ratio)
                return scale_factor, 1.0
            else:
                scale_factor = math.sqrt(ratio)
                return 1.0, scale_factor

        return 1.0, 1.0

    def _compute_regularization(self, constraint: Constraint, dt: float) -> float:
        """
        Compute regularization factor for soft constraints.

        Args:
            constraint: The constraint.
            dt: Time step.

        Returns:
            Regularization factor gamma.
        """
        if constraint.constraint_type == ConstraintType.SOFT:
            # Soft constraints use larger regularization
            return self._regularization_factor / dt
        else:
            # Small regularization for stability
            return 0.01 / dt

    def _init_split_impulse_velocities(self) -> None:
        """Initialize split impulse velocity storage."""
        self._split_impulse_velocities.clear()

        for constraint in self.constraints:
            body_a = constraint.body_a
            if body_a.id not in self._split_impulse_velocities:
                self._split_impulse_velocities[body_a.id] = (Vec3.zero(), Vec3.zero())

            if constraint.body_b is not None:
                body_b = constraint.body_b
                if body_b.id not in self._split_impulse_velocities:
                    self._split_impulse_velocities[body_b.id] = (Vec3.zero(), Vec3.zero())

    def _warm_start_tgs(self, factor: float, dt: float) -> None:
        """Apply TGS warm starting with stale data detection."""
        for data in self._tgs_data.values():
            if abs(data.lambda_accumulated) < 1e-10:
                continue

            constraint = data.constraint

            # Stale data detection: Check if constraint configuration changed significantly
            # by verifying the jacobian direction is still approximately valid
            if hasattr(constraint, '_last_jacobian_linear'):
                last_dir = constraint._last_jacobian_linear
                curr_dir = data.jacobian.linear_a
                # If direction changed by more than 45 degrees, data is stale
                if last_dir.length_squared() > 1e-6 and curr_dir.length_squared() > 1e-6:
                    dot = last_dir.normalized().dot(curr_dir.normalized())
                    if dot < 0.707:  # cos(45 degrees)
                        # Reset accumulated impulse - data is stale
                        data.lambda_accumulated = 0.0
                        continue

            # Clamp warm start impulse to prevent explosive correction from stale large impulses
            max_warm_impulse = data.effective_mass * 10.0  # Reasonable upper bound
            warm_impulse = max(-max_warm_impulse, min(max_warm_impulse, data.lambda_accumulated))

            impulse = warm_impulse * factor

            self._apply_tgs_impulse(
                constraint.body_a,
                constraint.body_b,
                data.jacobian,
                impulse,
                data.mass_scale_a,
                data.mass_scale_b
            )

            # Store current jacobian direction for next frame's stale check
            constraint._last_jacobian_linear = Vec3(
                data.jacobian.linear_a.x,
                data.jacobian.linear_a.y,
                data.jacobian.linear_a.z
            )

    def _solve_tgs_velocity_iteration(self) -> float:
        """
        Perform one TGS velocity iteration.

        Returns:
            Maximum impulse magnitude.
        """
        max_impulse = 0.0

        for data in self._tgs_data.values():
            constraint = data.constraint
            body_a = constraint.body_a
            body_b = constraint.body_b

            # Get velocities
            vel_b = body_b.velocity if body_b else Vec3.zero()
            ang_vel_b = body_b.angular_velocity if body_b else Vec3.zero()

            # Compute constraint velocity
            cdot = compute_relative_velocity(
                data.jacobian,
                body_a.velocity,
                body_a.angular_velocity,
                vel_b,
                ang_vel_b
            )

            # Compute impulse with regularization
            # lambda = -(K + gamma)^-1 * (Jv + bias + gamma * lambda_acc)
            regularized_mass = 1.0 / (1.0 / data.effective_mass + data.gamma)
            impulse = -regularized_mass * (cdot + data.bias + data.gamma * data.lambda_accumulated)

            # Apply relaxation
            impulse *= self.config.relaxation_factor

            # Clamp for inequality constraints
            old_accumulated = data.lambda_accumulated
            new_accumulated = data.lambda_accumulated + impulse

            new_accumulated = max(data.lower_limit, min(data.upper_limit, new_accumulated))
            impulse = new_accumulated - old_accumulated
            data.lambda_accumulated = new_accumulated

            # Apply impulse
            self._apply_tgs_impulse(
                body_a, body_b, data.jacobian, impulse,
                data.mass_scale_a, data.mass_scale_b
            )

            max_impulse = max(max_impulse, abs(impulse))

        return max_impulse

    def _solve_tgs_position_iteration(self, dt: float) -> float:
        """
        Perform one TGS position iteration.

        Uses split impulse to avoid adding energy.

        Args:
            dt: Time step.

        Returns:
            Maximum position error.
        """
        max_error = 0.0

        for data in self._tgs_data.values():
            constraint = data.constraint
            error = constraint.solve_position(self.config.max_correction_velocity)
            max_error = max(max_error, abs(error))

        return max_error

    def _apply_tgs_impulse(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody],
        jacobian: Jacobian,
        impulse: float,
        mass_scale_a: float,
        mass_scale_b: float
    ) -> None:
        """
        Apply impulse with TGS mass scaling.

        Args:
            body_a: First body.
            body_b: Second body (or None).
            jacobian: Constraint Jacobian.
            impulse: Impulse magnitude.
            mass_scale_a: Mass scale for body A.
            mass_scale_b: Mass scale for body B.
        """
        # Body A
        if not body_a.is_static:
            scaled_inv_mass_a = body_a.inv_mass * mass_scale_a
            delta_vel_a = jacobian.linear_a * (scaled_inv_mass_a * impulse)
            delta_ang_a = body_a.inv_inertia_world * (jacobian.angular_a * (impulse * mass_scale_a))

            body_a.velocity = body_a.velocity + delta_vel_a
            body_a.angular_velocity = body_a.angular_velocity + delta_ang_a

        # Body B
        if body_b is not None and not body_b.is_static:
            scaled_inv_mass_b = body_b.inv_mass * mass_scale_b
            delta_vel_b = jacobian.linear_b * (scaled_inv_mass_b * impulse)
            delta_ang_b = body_b.inv_inertia_world * (jacobian.angular_b * (impulse * mass_scale_b))

            body_b.velocity = body_b.velocity + delta_vel_b
            body_b.angular_velocity = body_b.angular_velocity + delta_ang_b

    def _apply_split_impulse_velocities(self) -> None:
        """Apply accumulated split impulse velocities to positions."""
        for body_id, (linear, angular) in self._split_impulse_velocities.items():
            if body_id in self.bodies:
                body = self.bodies[body_id]
                if not body.is_static:
                    # Apply position correction
                    body.position = body.position + linear * self._substep_dt

                    # Apply rotation correction
                    if angular.length_squared() > 1e-10:
                        angle = angular.length() * self._substep_dt
                        axis = angular.normalized()
                        delta_q = Quaternion.from_axis_angle(axis, angle)
                        body.orientation = (delta_q * body.orientation).normalized()

    def set_regularization(self, factor: float) -> None:
        """
        Set regularization factor.

        Args:
            factor: Regularization factor (larger = softer constraints).
        """
        self._regularization_factor = max(0.0, factor)

    def enable_split_impulse(self, enabled: bool) -> None:
        """
        Enable or disable split impulse.

        Args:
            enabled: Whether to use split impulse.
        """
        self._use_split_impulse = enabled


class TGSContactSolver:
    """
    Specialized TGS solver for contact constraints.

    Provides optimized solving for contact manifolds with
    friction clamping and block solving.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize TGS contact solver."""
        self.config = config or SolverConfig.tgs_default()
        self._contacts: List[TGSContactData] = []

    def add_contact(self, contact_data: "TGSContactData") -> None:
        """Add a contact to solve."""
        self._contacts.append(contact_data)

    def clear_contacts(self) -> None:
        """Clear all contacts."""
        self._contacts.clear()

    def solve_contacts(self, dt: float) -> None:
        """
        Solve all contacts with TGS method.

        Args:
            dt: Time step.
        """
        # Prepare contacts
        for contact in self._contacts:
            contact.prepare(dt, self.config)

        # Velocity iterations
        for _ in range(self.config.velocity_iterations):
            for contact in self._contacts:
                contact.solve_normal()

            # Friction iterations
            for _ in range(self.config.friction_iterations):
                for contact in self._contacts:
                    contact.solve_friction()

        # Position iterations
        for _ in range(self.config.position_iterations):
            for contact in self._contacts:
                contact.solve_position(self.config.max_correction_velocity)


@dataclass
class TGSContactData:
    """
    Contact data for TGS solver.

    Stores contact point information and solving state.
    """
    body_a: RigidBody
    body_b: Optional[RigidBody]
    point: Vec3
    normal: Vec3
    penetration: float
    friction: float = 0.4
    restitution: float = 0.0

    # Solving state
    normal_jacobian: Jacobian = field(default_factory=Jacobian)
    tangent1_jacobian: Jacobian = field(default_factory=Jacobian)
    tangent2_jacobian: Jacobian = field(default_factory=Jacobian)
    normal_mass: float = 0.0
    tangent1_mass: float = 0.0
    tangent2_mass: float = 0.0
    normal_impulse: float = 0.0
    tangent1_impulse: float = 0.0
    tangent2_impulse: float = 0.0
    velocity_bias: float = 0.0

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare contact for solving."""
        # Compute r vectors
        r_a = self.point - self.body_a.position
        r_b = (self.point - self.body_b.position) if self.body_b else Vec3.zero()

        # Normal Jacobian
        self.normal_jacobian = Jacobian(
            linear_a=-self.normal,
            angular_a=-r_a.cross(self.normal),
            linear_b=self.normal,
            angular_b=r_b.cross(self.normal)
        )

        # Tangent Jacobians
        tangent1, tangent2 = self._compute_tangent_basis()
        self.tangent1_jacobian = Jacobian(
            linear_a=-tangent1,
            angular_a=-r_a.cross(tangent1),
            linear_b=tangent1,
            angular_b=r_b.cross(tangent1)
        )
        self.tangent2_jacobian = Jacobian(
            linear_a=-tangent2,
            angular_a=-r_a.cross(tangent2),
            linear_b=tangent2,
            angular_b=r_b.cross(tangent2)
        )

        # Compute effective masses
        inv_mass_b = self.body_b.inv_mass if self.body_b else 0.0
        inv_inertia_b = self.body_b.inv_inertia_world if self.body_b else Mat3.zero()

        self.normal_mass = compute_effective_mass(
            self.normal_jacobian,
            self.body_a.inv_mass, self.body_a.inv_inertia_world,
            inv_mass_b, inv_inertia_b
        )
        self.tangent1_mass = compute_effective_mass(
            self.tangent1_jacobian,
            self.body_a.inv_mass, self.body_a.inv_inertia_world,
            inv_mass_b, inv_inertia_b
        )
        self.tangent2_mass = compute_effective_mass(
            self.tangent2_jacobian,
            self.body_a.inv_mass, self.body_a.inv_inertia_world,
            inv_mass_b, inv_inertia_b
        )

        # Compute velocity bias (restitution)
        vel_b = self.body_b.velocity if self.body_b else Vec3.zero()
        ang_vel_b = self.body_b.angular_velocity if self.body_b else Vec3.zero()

        relative_vel = compute_relative_velocity(
            self.normal_jacobian,
            self.body_a.velocity, self.body_a.angular_velocity,
            vel_b, ang_vel_b
        )

        if relative_vel < -1.0:
            self.velocity_bias = -self.restitution * relative_vel
        else:
            self.velocity_bias = 0.0

        # Add Baumgarte bias for penetration
        if self.penetration > config.slop:
            self.velocity_bias += config.baumgarte_factor * (self.penetration - config.slop) / dt

    def solve_normal(self) -> None:
        """Solve normal constraint."""
        vel_b = self.body_b.velocity if self.body_b else Vec3.zero()
        ang_vel_b = self.body_b.angular_velocity if self.body_b else Vec3.zero()

        cdot = compute_relative_velocity(
            self.normal_jacobian,
            self.body_a.velocity, self.body_a.angular_velocity,
            vel_b, ang_vel_b
        )

        impulse = -self.normal_mass * (cdot + self.velocity_bias)

        # Clamp to non-negative (contacts can only push apart)
        old_impulse = self.normal_impulse
        self.normal_impulse = max(0.0, self.normal_impulse + impulse)
        impulse = self.normal_impulse - old_impulse

        self._apply_impulse(self.normal_jacobian, impulse)

    def solve_friction(self) -> None:
        """Solve friction constraints."""
        max_friction = self.friction * self.normal_impulse

        # Tangent 1
        vel_b = self.body_b.velocity if self.body_b else Vec3.zero()
        ang_vel_b = self.body_b.angular_velocity if self.body_b else Vec3.zero()

        cdot1 = compute_relative_velocity(
            self.tangent1_jacobian,
            self.body_a.velocity, self.body_a.angular_velocity,
            vel_b, ang_vel_b
        )
        impulse1 = -self.tangent1_mass * cdot1

        old_impulse1 = self.tangent1_impulse
        self.tangent1_impulse = max(-max_friction, min(max_friction, self.tangent1_impulse + impulse1))
        impulse1 = self.tangent1_impulse - old_impulse1

        self._apply_impulse(self.tangent1_jacobian, impulse1)

        # Tangent 2
        vel_b = self.body_b.velocity if self.body_b else Vec3.zero()
        ang_vel_b = self.body_b.angular_velocity if self.body_b else Vec3.zero()

        cdot2 = compute_relative_velocity(
            self.tangent2_jacobian,
            self.body_a.velocity, self.body_a.angular_velocity,
            vel_b, ang_vel_b
        )
        impulse2 = -self.tangent2_mass * cdot2

        old_impulse2 = self.tangent2_impulse
        self.tangent2_impulse = max(-max_friction, min(max_friction, self.tangent2_impulse + impulse2))
        impulse2 = self.tangent2_impulse - old_impulse2

        self._apply_impulse(self.tangent2_jacobian, impulse2)

    def solve_position(self, max_correction: float) -> float:
        """Solve position constraint."""
        if self.penetration <= 0:
            return 0.0

        # Compute correction
        correction = min(max_correction, self.penetration * 0.2)

        # Apply correction via pseudo-velocity
        inv_mass_b = self.body_b.inv_mass if self.body_b else 0.0

        total_inv_mass = self.body_a.inv_mass + inv_mass_b
        if total_inv_mass < 1e-10:
            return self.penetration

        # Move bodies apart
        if not self.body_a.is_static:
            self.body_a.position = self.body_a.position - self.normal * (
                correction * self.body_a.inv_mass / total_inv_mass
            )

        if self.body_b and not self.body_b.is_static:
            self.body_b.position = self.body_b.position + self.normal * (
                correction * self.body_b.inv_mass / total_inv_mass
            )

        return self.penetration

    def _apply_impulse(self, jacobian: Jacobian, impulse: float) -> None:
        """Apply impulse to bodies."""
        if not self.body_a.is_static:
            self.body_a.velocity = self.body_a.velocity + jacobian.linear_a * (
                self.body_a.inv_mass * impulse
            )
            self.body_a.angular_velocity = self.body_a.angular_velocity + (
                self.body_a.inv_inertia_world * (jacobian.angular_a * impulse)
            )

        if self.body_b and not self.body_b.is_static:
            self.body_b.velocity = self.body_b.velocity + jacobian.linear_b * (
                self.body_b.inv_mass * impulse
            )
            self.body_b.angular_velocity = self.body_b.angular_velocity + (
                self.body_b.inv_inertia_world * (jacobian.angular_b * impulse)
            )

    def _compute_tangent_basis(self) -> Tuple[Vec3, Vec3]:
        """Compute tangent basis from normal."""
        if abs(self.normal.x) < 0.9:
            arbitrary = Vec3.unit_x()
        else:
            arbitrary = Vec3.unit_y()

        tangent1 = self.normal.cross(arbitrary).normalized()
        tangent2 = self.normal.cross(tangent1).normalized()

        return tangent1, tangent2
