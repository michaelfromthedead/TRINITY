"""
Sequential Impulse Constraint Solver.

This module implements the Sequential Impulse (SI) method for solving
physics constraints. It iteratively applies impulses to satisfy velocity
and position constraints.

References:
- Erin Catto, "Iterative Dynamics with Temporal Coherence" (GDC 2005)
- Box2D source code
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Protocol, runtime_checkable
from enum import Enum, auto
import math

from .config import (
    SolverConfig,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    BAUMGARTE_FACTOR,
    SLOP,
    WARM_START_FACTOR,
    MAX_CORRECTION_VELOCITY,
    RELAXATION_FACTOR,
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


class ConstraintType(Enum):
    """Types of constraints."""
    EQUALITY = auto()      # Must be exactly satisfied (e.g., joints)
    INEQUALITY = auto()    # One-sided (e.g., contact non-penetration)
    SOFT = auto()          # Soft constraint with compliance


@dataclass
class RigidBody:
    """
    Represents a rigid body in the physics simulation.

    Attributes:
        id: Unique identifier.
        position: World position of center of mass.
        orientation: Rotation quaternion.
        velocity: Linear velocity.
        angular_velocity: Angular velocity.
        mass: Total mass (0 for static bodies).
        inv_mass: Inverse mass.
        local_inertia: Inertia tensor in local space.
        inv_inertia_local: Inverse inertia in local space.
        inv_inertia_world: Inverse inertia in world space.
        is_static: Whether body is static (immovable).
        is_sleeping: Whether body is sleeping.
        island_id: ID of the island this body belongs to.
    """
    id: int
    position: Vec3 = field(default_factory=Vec3.zero)
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    velocity: Vec3 = field(default_factory=Vec3.zero)
    angular_velocity: Vec3 = field(default_factory=Vec3.zero)
    mass: float = 1.0
    inv_mass: float = 1.0
    local_inertia: Mat3 = field(default_factory=Mat3.identity)
    inv_inertia_local: Mat3 = field(default_factory=Mat3.identity)
    inv_inertia_world: Mat3 = field(default_factory=Mat3.identity)
    is_static: bool = False
    is_sleeping: bool = False
    island_id: int = -1

    def __post_init__(self):
        if self.is_static:
            self.inv_mass = 0.0
            self.inv_inertia_local = Mat3.zero()
            self.inv_inertia_world = Mat3.zero()
        else:
            if self.mass > 0:
                self.inv_mass = 1.0 / self.mass
            self.update_world_inertia()

    def update_world_inertia(self) -> None:
        """Update world-space inverse inertia tensor."""
        if self.is_static:
            self.inv_inertia_world = Mat3.zero()
            return

        rot = self.orientation.to_matrix()
        rot_t = rot.transpose()
        # I_world^-1 = R * I_local^-1 * R^T
        self.inv_inertia_world = rot.mat_mul(self.inv_inertia_local).mat_mul(rot_t)

    def apply_impulse(
        self,
        linear_impulse: Vec3,
        angular_impulse: Vec3
    ) -> None:
        """Apply impulse to body."""
        if self.is_static:
            return

        self.velocity = self.velocity + linear_impulse * self.inv_mass
        self.angular_velocity = self.angular_velocity + self.inv_inertia_world * angular_impulse

    def local_to_world(self, local_point: Vec3) -> Vec3:
        """Transform point from local to world space."""
        return self.position + self.orientation.rotate_vector(local_point)

    def world_to_local(self, world_point: Vec3) -> Vec3:
        """Transform point from world to local space."""
        return self.orientation.inverse_rotate_vector(world_point - self.position)

    def local_to_world_direction(self, local_dir: Vec3) -> Vec3:
        """Transform direction from local to world space."""
        return self.orientation.rotate_vector(local_dir)

    def world_to_local_direction(self, world_dir: Vec3) -> Vec3:
        """Transform direction from world to local space."""
        return self.orientation.inverse_rotate_vector(world_dir)

    def get_velocity_at_point(self, world_point: Vec3) -> Vec3:
        """Get velocity at a world point on the body."""
        r = world_point - self.position
        return self.velocity + self.angular_velocity.cross(r)

    @classmethod
    def create_static(cls, id: int, position: Vec3 = None, orientation: Quaternion = None) -> "RigidBody":
        """Create a static (immovable) body."""
        return cls(
            id=id,
            position=position or Vec3.zero(),
            orientation=orientation or Quaternion.identity(),
            mass=0.0,
            inv_mass=0.0,
            is_static=True
        )

    @classmethod
    def create_dynamic(
        cls,
        id: int,
        position: Vec3 = None,
        orientation: Quaternion = None,
        mass: float = 1.0,
        inertia: Mat3 = None
    ) -> "RigidBody":
        """Create a dynamic body."""
        body = cls(
            id=id,
            position=position or Vec3.zero(),
            orientation=orientation or Quaternion.identity(),
            mass=mass,
            is_static=False
        )
        if inertia is not None:
            body.local_inertia = inertia
            body.inv_inertia_local = inertia.inverse()
            body.update_world_inertia()
        return body


@runtime_checkable
class Constraint(Protocol):
    """
    Protocol defining the constraint interface.

    All constraint types must implement this interface to work with
    the constraint solver.
    """

    @property
    def body_a(self) -> RigidBody:
        """First body in the constraint."""
        ...

    @property
    def body_b(self) -> Optional[RigidBody]:
        """Second body in the constraint (None for world constraints)."""
        ...

    @property
    def constraint_type(self) -> ConstraintType:
        """Type of constraint."""
        ...

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """
        Prepare constraint for solving.

        Called once per physics step before iterations.
        Computes Jacobians, effective masses, bias velocities, etc.

        Args:
            dt: Time step.
            config: Solver configuration.
        """
        ...

    def warm_start(self, factor: float) -> None:
        """
        Apply warm starting impulse.

        Uses cached impulse from previous frame scaled by factor.

        Args:
            factor: Warm start scale factor (0-1).
        """
        ...

    def solve_velocity(self) -> float:
        """
        Solve velocity constraint.

        Returns:
            Impulse magnitude applied.
        """
        ...

    def solve_position(self, max_correction: float) -> float:
        """
        Solve position constraint.

        Args:
            max_correction: Maximum position correction.

        Returns:
            Position error magnitude.
        """
        ...

    def get_cached_impulse(self) -> float:
        """Get cached impulse for warm starting."""
        ...

    def set_cached_impulse(self, impulse: float) -> None:
        """Set cached impulse for warm starting."""
        ...


@dataclass
class ConstraintInfo:
    """
    Information about a constraint for the solver.

    Holds precomputed data needed during constraint solving iterations.
    """
    constraint: Constraint
    jacobian: Jacobian
    effective_mass: float = 0.0
    bias: float = 0.0
    lower_limit: float = float("-inf")
    upper_limit: float = float("inf")
    accumulated_impulse: float = 0.0


class ConstraintSolver:
    """
    Sequential Impulse Constraint Solver.

    Solves constraints using iterative impulse application:
    1. Warm start: Apply previous frame's impulses
    2. Velocity solve: Iterate to satisfy velocity constraints
    3. Position solve: Iterate to correct position errors

    Attributes:
        config: Solver configuration.
        constraints: List of active constraints.
        bodies: Dictionary of bodies by ID.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize constraint solver.

        Args:
            config: Solver configuration (uses defaults if None).
        """
        self.config = config or SolverConfig.default()
        self.constraints: List[Constraint] = []
        self.bodies: Dict[int, RigidBody] = {}
        self._constraint_infos: List[ConstraintInfo] = []
        self._iteration_count: int = 0
        self._last_error: float = 0.0

    def add_body(self, body: RigidBody) -> None:
        """Add a body to the solver."""
        self.bodies[body.id] = body

    def remove_body(self, body_id: int) -> None:
        """Remove a body from the solver."""
        self.bodies.pop(body_id, None)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the solver."""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        """Remove a constraint from the solver."""
        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self.constraints.clear()
        self._constraint_infos.clear()

    def solve(self, dt: float) -> None:
        """
        Solve all constraints for one physics step.

        Args:
            dt: Time step.
        """
        if not self.constraints:
            return

        # Prepare all constraints
        self._prepare_constraints(dt)

        # Warm start
        if self.config.warm_start_factor > 0:
            self._warm_start()

        # Velocity iterations
        for i in range(self.config.velocity_iterations):
            self._iteration_count = i
            error = self._solve_velocity_iteration()
            if error < 1e-6:
                break

        # Position iterations
        for i in range(self.config.position_iterations):
            error = self._solve_position_iteration()
            self._last_error = error
            if error < self.config.slop:
                break

    def _prepare_constraints(self, dt: float) -> None:
        """Prepare all constraints for solving."""
        self._constraint_infos.clear()

        for constraint in self.constraints:
            # Update world inertia for constraint bodies
            constraint.body_a.update_world_inertia()
            if constraint.body_b is not None:
                constraint.body_b.update_world_inertia()

            # Let constraint prepare itself
            constraint.prepare(dt, self.config)

    def _warm_start(self) -> None:
        """Apply warm starting impulses."""
        factor = self.config.warm_start_factor

        for constraint in self.constraints:
            constraint.warm_start(factor)

    def _solve_velocity_iteration(self) -> float:
        """
        Perform one velocity constraint iteration.

        Returns:
            Maximum impulse magnitude applied.
        """
        max_impulse = 0.0

        for constraint in self.constraints:
            impulse = constraint.solve_velocity()
            max_impulse = max(max_impulse, abs(impulse))

        return max_impulse

    def _solve_position_iteration(self) -> float:
        """
        Perform one position constraint iteration.

        Returns:
            Maximum position error.
        """
        max_error = 0.0
        max_correction = self.config.max_correction_velocity

        for constraint in self.constraints:
            error = constraint.solve_position(max_correction)
            max_error = max(max_error, abs(error))

        return max_error

    def solve_velocity_constraints(self, iterations: Optional[int] = None) -> float:
        """
        Solve only velocity constraints.

        Args:
            iterations: Number of iterations (uses config default if None).

        Returns:
            Final maximum impulse.
        """
        num_iterations = iterations or self.config.velocity_iterations
        max_impulse = 0.0

        for i in range(num_iterations):
            self._iteration_count = i
            max_impulse = self._solve_velocity_iteration()
            if max_impulse < 1e-6:
                break

        return max_impulse

    def solve_position_constraints(self, iterations: Optional[int] = None) -> float:
        """
        Solve only position constraints.

        Args:
            iterations: Number of iterations (uses config default if None).

        Returns:
            Final maximum error.
        """
        num_iterations = iterations or self.config.position_iterations
        max_error = 0.0

        for i in range(num_iterations):
            max_error = self._solve_position_iteration()
            if max_error < self.config.slop:
                break

        return max_error

    def get_iteration_count(self) -> int:
        """Get number of iterations from last solve."""
        return self._iteration_count

    def get_last_error(self) -> float:
        """Get final error from last solve."""
        return self._last_error


@dataclass
class BaseConstraint:
    """
    Base implementation for constraints.

    Provides common functionality for all constraint types.
    Subclasses should override prepare() and solve methods.
    """
    _body_a: RigidBody
    _body_b: Optional[RigidBody] = None
    _constraint_type: ConstraintType = ConstraintType.EQUALITY
    _jacobian: Jacobian = field(default_factory=Jacobian)
    _effective_mass: float = 0.0
    _bias: float = 0.0
    _accumulated_impulse: float = 0.0
    _lower_limit: float = float("-inf")
    _upper_limit: float = float("inf")

    @property
    def body_a(self) -> RigidBody:
        return self._body_a

    @property
    def body_b(self) -> Optional[RigidBody]:
        return self._body_b

    @property
    def constraint_type(self) -> ConstraintType:
        return self._constraint_type

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare constraint for solving. Override in subclasses."""
        pass

    def warm_start(self, factor: float) -> None:
        """Apply warm starting impulse."""
        if abs(self._accumulated_impulse) < 1e-10:
            return

        impulse = self._accumulated_impulse * factor

        body_b = self._body_b or self._get_world_body()

        delta_vel_a, delta_ang_a, delta_vel_b, delta_ang_b = apply_impulse(
            self._jacobian,
            impulse,
            self._body_a.inv_mass,
            self._body_a.inv_inertia_world,
            body_b.inv_mass,
            body_b.inv_inertia_world
        )

        self._body_a.velocity = self._body_a.velocity + delta_vel_a
        self._body_a.angular_velocity = self._body_a.angular_velocity + delta_ang_a

        if self._body_b is not None:
            self._body_b.velocity = self._body_b.velocity + delta_vel_b
            self._body_b.angular_velocity = self._body_b.angular_velocity + delta_ang_b

    def solve_velocity(self) -> float:
        """Solve velocity constraint."""
        body_b = self._body_b or self._get_world_body()

        # Compute relative velocity
        cdot = compute_relative_velocity(
            self._jacobian,
            self._body_a.velocity,
            self._body_a.angular_velocity,
            body_b.velocity,
            body_b.angular_velocity
        )

        # Compute impulse: lambda = -K * (Jv + b)
        impulse = -self._effective_mass * (cdot + self._bias)

        # Clamp impulse for inequality constraints
        if self._constraint_type == ConstraintType.INEQUALITY:
            impulse, self._accumulated_impulse = clamp_impulse(
                impulse,
                self._accumulated_impulse,
                self._lower_limit,
                self._upper_limit
            )
        else:
            self._accumulated_impulse += impulse

        # Apply impulse
        delta_vel_a, delta_ang_a, delta_vel_b, delta_ang_b = apply_impulse(
            self._jacobian,
            impulse,
            self._body_a.inv_mass,
            self._body_a.inv_inertia_world,
            body_b.inv_mass,
            body_b.inv_inertia_world
        )

        self._body_a.velocity = self._body_a.velocity + delta_vel_a
        self._body_a.angular_velocity = self._body_a.angular_velocity + delta_ang_a

        if self._body_b is not None:
            self._body_b.velocity = self._body_b.velocity + delta_vel_b
            self._body_b.angular_velocity = self._body_b.angular_velocity + delta_ang_b

        return impulse

    def solve_position(self, max_correction: float) -> float:
        """Solve position constraint. Override in subclasses."""
        return 0.0

    def get_cached_impulse(self) -> float:
        return self._accumulated_impulse

    def set_cached_impulse(self, impulse: float) -> None:
        self._accumulated_impulse = impulse

    def _get_world_body(self) -> RigidBody:
        """Get a static world body for single-body constraints."""
        return RigidBody.create_static(id=-1)

    def _compute_effective_mass(self) -> float:
        """Compute effective mass for the constraint."""
        body_b = self._body_b or self._get_world_body()

        return compute_effective_mass(
            self._jacobian,
            self._body_a.inv_mass,
            self._body_a.inv_inertia_world,
            body_b.inv_mass,
            body_b.inv_inertia_world
        )


@dataclass
class PointConstraint(BaseConstraint):
    """
    Point-to-point constraint (distance constraint).

    Constrains two anchor points to maintain zero distance.
    """
    local_anchor_a: Vec3 = field(default_factory=Vec3.zero)
    local_anchor_b: Vec3 = field(default_factory=Vec3.zero)

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare point constraint."""
        # Transform anchors to world space
        anchor_a = self._body_a.local_to_world(self.local_anchor_a)

        if self._body_b is not None:
            anchor_b = self._body_b.local_to_world(self.local_anchor_b)
        else:
            anchor_b = self.local_anchor_b  # World space anchor

        # Compute direction
        direction = anchor_b - anchor_a
        distance = direction.length()

        if distance < 1e-10:
            direction = Vec3.unit_x()
        else:
            direction = direction / distance

        # Compute r vectors (from body center to anchor in world space)
        r_a = anchor_a - self._body_a.position
        if self._body_b is not None:
            r_b = anchor_b - self._body_b.position
        else:
            r_b = Vec3.zero()

        # Set up Jacobian
        self._jacobian = Jacobian(
            linear_a=-direction,
            angular_a=-r_a.cross(direction),
            linear_b=direction,
            angular_b=r_b.cross(direction)
        )

        # Compute effective mass
        self._effective_mass = self._compute_effective_mass()

        # Baumgarte stabilization bias
        self._bias = config.baumgarte_factor * max(0.0, distance - config.slop) / dt


@dataclass
class AxisConstraint(BaseConstraint):
    """
    Axis constraint.

    Constrains motion along a specific axis.
    """
    local_anchor_a: Vec3 = field(default_factory=Vec3.zero)
    local_anchor_b: Vec3 = field(default_factory=Vec3.zero)
    local_axis_a: Vec3 = field(default_factory=Vec3.unit_x)

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare axis constraint."""
        # Transform to world space
        anchor_a = self._body_a.local_to_world(self.local_anchor_a)
        axis = self._body_a.local_to_world_direction(self.local_axis_a)

        if self._body_b is not None:
            anchor_b = self._body_b.local_to_world(self.local_anchor_b)
        else:
            anchor_b = self.local_anchor_b

        # Compute error
        error = anchor_b - anchor_a
        position_error = error.dot(axis)

        # Compute r vectors
        r_a = anchor_a - self._body_a.position
        if self._body_b is not None:
            r_b = anchor_b - self._body_b.position
        else:
            r_b = Vec3.zero()

        # Set up Jacobian
        self._jacobian = Jacobian(
            linear_a=-axis,
            angular_a=-r_a.cross(axis),
            linear_b=axis,
            angular_b=r_b.cross(axis)
        )

        # Compute effective mass
        self._effective_mass = self._compute_effective_mass()

        # Baumgarte stabilization bias
        self._bias = config.baumgarte_factor * position_error / dt


@dataclass
class AngularConstraint(BaseConstraint):
    """
    Angular constraint.

    Constrains rotation around a specific axis.
    """
    local_axis_a: Vec3 = field(default_factory=Vec3.unit_x)
    local_axis_b: Vec3 = field(default_factory=Vec3.unit_x)
    target_angle: float = 0.0

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare angular constraint."""
        # Transform axes to world space
        axis_a = self._body_a.local_to_world_direction(self.local_axis_a)

        if self._body_b is not None:
            axis_b = self._body_b.local_to_world_direction(self.local_axis_b)
        else:
            axis_b = self.local_axis_b

        # Compute angular error (cross product gives error direction)
        error_axis = axis_a.cross(axis_b)
        error_magnitude = error_axis.length()

        if error_magnitude < 1e-10:
            error_axis = Vec3.unit_x()
        else:
            error_axis = error_axis / error_magnitude

        # Compute angular error
        angle_error = math.asin(min(1.0, error_magnitude))

        # Set up Jacobian (angular only)
        self._jacobian = Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-error_axis,
            linear_b=Vec3.zero(),
            angular_b=error_axis
        )

        # Compute effective mass
        self._effective_mass = self._compute_effective_mass()

        # Baumgarte stabilization bias
        self._bias = config.baumgarte_factor * angle_error / dt
