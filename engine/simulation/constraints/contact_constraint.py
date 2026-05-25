"""
Contact Constraint Implementation.

Handles contact constraints for collision response including:
- Normal constraint (non-penetration)
- Friction constraint (Coulomb model)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

from .joint_base import Joint, JointState
from ..solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from ..solver.constraint_solver import RigidBody, ConstraintType
from ..solver.config import SolverConfig


@dataclass
class ContactPoint:
    """
    Single contact point between two bodies.

    Attributes:
        position: Contact position in world space.
        normal: Contact normal (from B to A).
        penetration: Penetration depth (positive = penetrating).
        local_point_a: Contact point in body A's local space.
        local_point_b: Contact point in body B's local space.
        normal_impulse: Accumulated normal impulse.
        tangent_impulse_1: Accumulated first tangent impulse.
        tangent_impulse_2: Accumulated second tangent impulse.
        combined_friction: Combined friction coefficient.
        combined_restitution: Combined restitution coefficient.
    """
    position: Vec3
    normal: Vec3
    penetration: float
    local_point_a: Vec3 = field(default_factory=Vec3.zero)
    local_point_b: Vec3 = field(default_factory=Vec3.zero)
    normal_impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0
    combined_friction: float = 0.4
    combined_restitution: float = 0.0
    id: int = 0


@dataclass
class ContactManifold:
    """
    Collection of contact points between two bodies.

    A manifold contains up to 4 contact points for 3D simulation.

    Attributes:
        body_a: First rigid body.
        body_b: Second rigid body.
        points: List of contact points.
        normal: Average contact normal.
        friction: Combined friction coefficient.
        restitution: Combined restitution coefficient.
    """
    body_a: RigidBody
    body_b: Optional[RigidBody]
    points: List[ContactPoint] = field(default_factory=list)
    normal: Vec3 = field(default_factory=Vec3.unit_y)
    friction: float = 0.4
    restitution: float = 0.0

    def add_point(self, point: ContactPoint) -> None:
        """Add a contact point to the manifold."""
        # Limit to 4 points
        if len(self.points) >= 4:
            # Replace furthest point from new point
            max_dist_sq = 0.0
            max_idx = 0
            for i, p in enumerate(self.points):
                dist_sq = (p.position - point.position).length_squared()
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    max_idx = i
            self.points[max_idx] = point
        else:
            self.points.append(point)

    def clear(self) -> None:
        """Clear all contact points."""
        self.points.clear()

    @property
    def point_count(self) -> int:
        """Get number of contact points."""
        return len(self.points)

    def get_average_position(self) -> Vec3:
        """Get average contact position."""
        if not self.points:
            return Vec3.zero()

        avg = Vec3.zero()
        for p in self.points:
            avg = avg + p.position
        return avg * (1.0 / len(self.points))

    def get_max_penetration(self) -> float:
        """Get maximum penetration depth."""
        if not self.points:
            return 0.0
        return max(p.penetration for p in self.points)


class ContactConstraint(Joint):
    """
    Contact constraint for collision response.

    Implements:
    - Normal constraint: prevents penetration (inequality)
    - Friction constraint: Coulomb friction model

    Attributes:
        manifold: Contact manifold with contact points.
        friction_coefficient: Combined friction coefficient.
        restitution_coefficient: Combined restitution coefficient.
        contact_point: Primary contact point for single-point contacts.
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody] = None,
        manifold: ContactManifold = None
    ):
        """
        Initialize contact constraint.

        Args:
            body_a: First body (the colliding body).
            body_b: Second body (or None for world collision).
            manifold: Contact manifold (created if None).
        """
        super().__init__(body_a, body_b)

        self._constraint_type = ConstraintType.INEQUALITY

        if manifold is None:
            self._manifold = ContactManifold(body_a, body_b)
        else:
            self._manifold = manifold

        self._friction_coefficient = 0.4
        self._restitution_coefficient = 0.0

        # Per-point constraint data
        self._point_data: List[ContactPointData] = []

    @property
    def manifold(self) -> ContactManifold:
        """Get contact manifold."""
        return self._manifold

    @property
    def friction_coefficient(self) -> float:
        """Get friction coefficient."""
        return self._friction_coefficient

    @friction_coefficient.setter
    def friction_coefficient(self, value: float) -> None:
        """Set friction coefficient."""
        self._friction_coefficient = max(0.0, value)

    @property
    def restitution_coefficient(self) -> float:
        """Get restitution coefficient."""
        return self._restitution_coefficient

    @restitution_coefficient.setter
    def restitution_coefficient(self, value: float) -> None:
        """Set restitution coefficient."""
        self._restitution_coefficient = max(0.0, min(1.0, value))

    def set_contact_point(
        self,
        position: Vec3,
        normal: Vec3,
        penetration: float
    ) -> None:
        """
        Set a single contact point.

        Args:
            position: Contact position in world space.
            normal: Contact normal (from B to A).
            penetration: Penetration depth.
        """
        self._manifold.clear()
        self._manifold.normal = normal

        point = ContactPoint(
            position=position,
            normal=normal,
            penetration=penetration,
            combined_friction=self._friction_coefficient,
            combined_restitution=self._restitution_coefficient
        )

        # Compute local points
        point.local_point_a = self._body_a.world_to_local(position)
        if self._body_b is not None:
            point.local_point_b = self._body_b.world_to_local(position)
        else:
            point.local_point_b = position

        self._manifold.add_point(point)

    def get_constraint_count(self) -> int:
        """Get number of constraint rows (3 per contact point)."""
        return len(self._manifold.points) * 3

    def prepare(self, dt: float, config: SolverConfig) -> None:
        """Prepare contact constraint for solving."""
        if self._state != JointState.ACTIVE:
            return

        if not self._manifold.points:
            return

        self._body_a.update_world_inertia()
        if self._body_b is not None:
            self._body_b.update_world_inertia()

        self._point_data.clear()

        for point in self._manifold.points:
            point_data = self._prepare_contact_point(point, dt, config)
            self._point_data.append(point_data)

    def _prepare_contact_point(
        self,
        point: ContactPoint,
        dt: float,
        config: SolverConfig
    ) -> "ContactPointData":
        """Prepare a single contact point."""
        # Recompute world-space contact position
        world_point_a = self._body_a.local_to_world(point.local_point_a)
        if self._body_b is not None:
            world_point_b = self._body_b.local_to_world(point.local_point_b)
        else:
            world_point_b = point.local_point_b

        # R vectors from body centers to contact points
        r_a = world_point_a - self._body_a.position
        if self._body_b is not None:
            r_b = world_point_b - self._body_b.position
        else:
            r_b = Vec3.zero()

        normal = point.normal

        # Compute tangent basis
        tangent1, tangent2 = self._compute_tangent_basis(normal)

        # ============ NORMAL CONSTRAINT ============
        normal_jacobian = Jacobian(
            linear_a=-normal,
            angular_a=-r_a.cross(normal),
            linear_b=normal,
            angular_b=r_b.cross(normal)
        )
        normal_mass = self._compute_effective_mass(normal_jacobian)

        # Compute velocity bias for restitution
        vel_a = self._body_a.get_velocity_at_point(world_point_a)
        if self._body_b is not None:
            vel_b = self._body_b.get_velocity_at_point(world_point_b)
        else:
            vel_b = Vec3.zero()

        rel_vel = vel_b - vel_a
        normal_vel = rel_vel.dot(normal)

        # Restitution bias (bounce)
        restitution_bias = 0.0
        velocity_threshold = 1.0
        if normal_vel < -velocity_threshold:
            restitution_bias = -point.combined_restitution * normal_vel

        # Baumgarte bias for penetration correction
        penetration_bias = 0.0
        if point.penetration > config.slop:
            penetration_bias = config.baumgarte_factor * (
                point.penetration - config.slop
            ) / dt

        normal_bias = restitution_bias + penetration_bias

        # ============ FRICTION CONSTRAINTS ============
        friction_jacobian_1 = Jacobian(
            linear_a=-tangent1,
            angular_a=-r_a.cross(tangent1),
            linear_b=tangent1,
            angular_b=r_b.cross(tangent1)
        )
        friction_mass_1 = self._compute_effective_mass(friction_jacobian_1)

        friction_jacobian_2 = Jacobian(
            linear_a=-tangent2,
            angular_a=-r_a.cross(tangent2),
            linear_b=tangent2,
            angular_b=r_b.cross(tangent2)
        )
        friction_mass_2 = self._compute_effective_mass(friction_jacobian_2)

        return ContactPointData(
            point=point,
            normal_jacobian=normal_jacobian,
            friction_jacobian_1=friction_jacobian_1,
            friction_jacobian_2=friction_jacobian_2,
            normal_mass=normal_mass,
            friction_mass_1=friction_mass_1,
            friction_mass_2=friction_mass_2,
            normal_bias=normal_bias,
            r_a=r_a,
            r_b=r_b,
            tangent1=tangent1,
            tangent2=tangent2,
        )

    def _compute_tangent_basis(self, normal: Vec3) -> Tuple[Vec3, Vec3]:
        """Compute tangent basis for friction."""
        if abs(normal.x) < 0.9:
            arbitrary = Vec3.unit_x()
        else:
            arbitrary = Vec3.unit_y()

        tangent1 = normal.cross(arbitrary).normalized()
        tangent2 = normal.cross(tangent1).normalized()

        return tangent1, tangent2

    def warm_start(self, factor: float) -> None:
        """Apply warm starting impulse."""
        for point_data in self._point_data:
            self._apply_warm_start(point_data, factor)

    def _apply_warm_start(self, point_data: "ContactPointData", factor: float) -> None:
        """Apply warm start to a single contact point."""
        point = point_data.point

        # Normal impulse
        normal_impulse = point.normal_impulse * factor
        self._apply_impulse(point_data.normal_jacobian, normal_impulse)

        # Friction impulses
        tangent_impulse_1 = point.tangent_impulse_1 * factor
        self._apply_impulse(point_data.friction_jacobian_1, tangent_impulse_1)

        tangent_impulse_2 = point.tangent_impulse_2 * factor
        self._apply_impulse(point_data.friction_jacobian_2, tangent_impulse_2)

    def solve_velocity(self) -> float:
        """Solve velocity constraints for all contact points."""
        if self._state != JointState.ACTIVE:
            return 0.0

        max_impulse = 0.0

        for point_data in self._point_data:
            # Solve normal constraint first
            normal_impulse = self._solve_normal(point_data)
            max_impulse = max(max_impulse, abs(normal_impulse))

            # Solve friction constraints
            friction_impulse_1, friction_impulse_2 = self._solve_friction(point_data)
            max_impulse = max(max_impulse, abs(friction_impulse_1), abs(friction_impulse_2))

        return max_impulse

    def _solve_normal(self, point_data: "ContactPointData") -> float:
        """Solve normal constraint for a contact point."""
        point = point_data.point
        jacobian = point_data.normal_jacobian

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
        impulse = -point_data.normal_mass * (cdot + point_data.normal_bias)

        # Clamp to non-negative (contact can only push apart)
        old_impulse = point.normal_impulse
        point.normal_impulse = max(0.0, point.normal_impulse + impulse)
        impulse = point.normal_impulse - old_impulse

        # Apply impulse
        self._apply_impulse(jacobian, impulse)

        return impulse

    def _solve_friction(self, point_data: "ContactPointData") -> Tuple[float, float]:
        """Solve friction constraints for a contact point."""
        point = point_data.point

        # Maximum friction impulse based on normal impulse
        max_friction = self._friction_coefficient * point.normal_impulse

        # ============ FRICTION 1 ============
        jacobian_1 = point_data.friction_jacobian_1

        vel_b = self._body_b.velocity if self._body_b else Vec3.zero()
        ang_vel_b = self._body_b.angular_velocity if self._body_b else Vec3.zero()

        cdot_1 = (
            jacobian_1.linear_a.dot(self._body_a.velocity) +
            jacobian_1.angular_a.dot(self._body_a.angular_velocity) +
            jacobian_1.linear_b.dot(vel_b) +
            jacobian_1.angular_b.dot(ang_vel_b)
        )

        impulse_1 = -point_data.friction_mass_1 * cdot_1

        old_impulse_1 = point.tangent_impulse_1
        point.tangent_impulse_1 = max(-max_friction, min(max_friction, point.tangent_impulse_1 + impulse_1))
        impulse_1 = point.tangent_impulse_1 - old_impulse_1

        self._apply_impulse(jacobian_1, impulse_1)

        # ============ FRICTION 2 ============
        jacobian_2 = point_data.friction_jacobian_2

        vel_b = self._body_b.velocity if self._body_b else Vec3.zero()
        ang_vel_b = self._body_b.angular_velocity if self._body_b else Vec3.zero()

        cdot_2 = (
            jacobian_2.linear_a.dot(self._body_a.velocity) +
            jacobian_2.angular_a.dot(self._body_a.angular_velocity) +
            jacobian_2.linear_b.dot(vel_b) +
            jacobian_2.angular_b.dot(ang_vel_b)
        )

        impulse_2 = -point_data.friction_mass_2 * cdot_2

        old_impulse_2 = point.tangent_impulse_2
        point.tangent_impulse_2 = max(-max_friction, min(max_friction, point.tangent_impulse_2 + impulse_2))
        impulse_2 = point.tangent_impulse_2 - old_impulse_2

        self._apply_impulse(jacobian_2, impulse_2)

        return impulse_1, impulse_2

    def _apply_impulse(self, jacobian: Jacobian, impulse: float) -> None:
        """Apply impulse to bodies."""
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

    def _solve_position_internal(self, max_correction: float) -> float:
        """Solve position constraints for contact."""
        if not self._manifold.points:
            return 0.0

        max_error = 0.0

        for point in self._manifold.points:
            # Recompute world positions
            world_point_a = self._body_a.local_to_world(point.local_point_a)
            if self._body_b is not None:
                world_point_b = self._body_b.local_to_world(point.local_point_b)
            else:
                world_point_b = point.local_point_b

            # Compute separation
            d = world_point_b - world_point_a
            separation = d.dot(point.normal) - point.penetration

            max_error = max(max_error, -separation)

            if separation >= 0:
                continue

            # Apply position correction
            correction = min(max_correction, -separation)
            correction *= 0.2  # Position correction factor

            total_inv_mass = self._body_a.inv_mass
            if self._body_b is not None:
                total_inv_mass += self._body_b.inv_mass

            if total_inv_mass < 1e-10:
                continue

            if not self._body_a.is_static:
                self._body_a.position = self._body_a.position - point.normal * (
                    correction * self._body_a.inv_mass / total_inv_mass
                )

            if self._body_b is not None and not self._body_b.is_static:
                self._body_b.position = self._body_b.position + point.normal * (
                    correction * self._body_b.inv_mass / total_inv_mass
                )

        return max_error


@dataclass
class ContactPointData:
    """Per-contact-point solving data."""
    point: ContactPoint
    normal_jacobian: Jacobian
    friction_jacobian_1: Jacobian
    friction_jacobian_2: Jacobian
    normal_mass: float
    friction_mass_1: float
    friction_mass_2: float
    normal_bias: float
    r_a: Vec3
    r_b: Vec3
    tangent1: Vec3
    tangent2: Vec3


def compute_contact_jacobian(
    contact_point: Vec3,
    normal: Vec3,
    body_a: RigidBody,
    body_b: Optional[RigidBody]
) -> Jacobian:
    """
    Compute Jacobian for a contact constraint.

    Args:
        contact_point: Contact position in world space.
        normal: Contact normal (from B to A).
        body_a: First body.
        body_b: Second body (or None).

    Returns:
        Contact Jacobian.
    """
    r_a = contact_point - body_a.position
    r_b = (contact_point - body_b.position) if body_b else Vec3.zero()

    return Jacobian(
        linear_a=-normal,
        angular_a=-r_a.cross(normal),
        linear_b=normal,
        angular_b=r_b.cross(normal)
    )


def combine_friction(friction_a: float, friction_b: float) -> float:
    """
    Combine friction coefficients from two bodies.

    Uses geometric mean as is common in physics engines.

    Args:
        friction_a: Friction coefficient of body A.
        friction_b: Friction coefficient of body B.

    Returns:
        Combined friction coefficient.
    """
    return math.sqrt(friction_a * friction_b)


def combine_restitution(
    restitution_a: float,
    restitution_b: float
) -> float:
    """
    Combine restitution coefficients from two bodies.

    Uses maximum as is common in physics engines.

    Args:
        restitution_a: Restitution of body A.
        restitution_b: Restitution of body B.

    Returns:
        Combined restitution coefficient.
    """
    return max(restitution_a, restitution_b)
