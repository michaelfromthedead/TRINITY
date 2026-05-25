"""
Extended Position Based Dynamics (XPBD) Solver.

Implements XPBD for constraint solving with compliance parameters.
XPBD provides excellent stability for soft constraints and is
particularly well-suited for cloth, soft bodies, and deformable objects.

Key features:
- Compliance parameter (inverse stiffness)
- Damping support
- Position-based solving with Lagrange multipliers
- Substep integration for stability

References:
- Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (2016)
- Bender et al., "A Survey on Position Based Dynamics, 2017" (2017)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Protocol, runtime_checkable
from enum import Enum, auto
import math

from .config import SolverConfig, SolverType, XPBD_VELOCITY_DAMPING
from .constraint_solver import ConstraintSolver, Constraint, RigidBody, ConstraintType
from .jacobian import Vec3, Mat3, Quaternion


@dataclass
class XPBDParticle:
    """
    Particle representation for XPBD.

    XPBD works with particles that have position and velocity.
    Rigid bodies can be represented as particles with additional
    orientation state.
    """
    id: int
    position: Vec3 = field(default_factory=Vec3.zero)
    previous_position: Vec3 = field(default_factory=Vec3.zero)
    velocity: Vec3 = field(default_factory=Vec3.zero)
    inv_mass: float = 1.0
    is_static: bool = False

    # For rigid body representation
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    previous_orientation: Quaternion = field(default_factory=Quaternion.identity)
    angular_velocity: Vec3 = field(default_factory=Vec3.zero)
    inv_inertia: Mat3 = field(default_factory=Mat3.identity)

    def store_state(self) -> None:
        """Store current state as previous state."""
        self.previous_position = Vec3(self.position.x, self.position.y, self.position.z)
        self.previous_orientation = Quaternion(
            self.orientation.x, self.orientation.y,
            self.orientation.z, self.orientation.w
        )

    def predict_position(self, dt: float, gravity: Vec3 = None) -> None:
        """Predict new position based on velocity."""
        if self.is_static:
            return

        # Apply gravity to velocity
        if gravity is not None:
            self.velocity = self.velocity + gravity * dt

        # Predict position
        self.position = self.position + self.velocity * dt

        # Predict orientation
        if self.angular_velocity.length_squared() > 1e-10:
            angle = self.angular_velocity.length() * dt
            axis = self.angular_velocity.normalized()
            delta_q = Quaternion.from_axis_angle(axis, angle)
            self.orientation = (delta_q * self.orientation).normalized()

    def update_velocity(self, dt: float) -> None:
        """Update velocity from position change."""
        if self.is_static or dt < 1e-10:
            return

        inv_dt = 1.0 / dt

        # Linear velocity
        self.velocity = (self.position - self.previous_position) * inv_dt

        # Angular velocity from quaternion difference
        q_diff = self.orientation * self.previous_orientation.conjugate()
        if q_diff.w < 0:
            q_diff = Quaternion(-q_diff.x, -q_diff.y, -q_diff.z, -q_diff.w)

        # Extract angular velocity: w = 2 * q_diff.xyz / dt (for small angles)
        self.angular_velocity = Vec3(q_diff.x, q_diff.y, q_diff.z) * (2.0 * inv_dt)


@runtime_checkable
class XPBDConstraint(Protocol):
    """
    Protocol for XPBD constraints.

    XPBD constraints compute position corrections using compliance
    and Lagrange multipliers.
    """

    @property
    def compliance(self) -> float:
        """Compliance (inverse stiffness). 0 = infinitely stiff."""
        ...

    @property
    def damping(self) -> float:
        """Damping coefficient."""
        ...

    def compute_constraint(self) -> float:
        """
        Compute constraint value C.

        Returns:
            Constraint value (0 when satisfied).
        """
        ...

    def compute_gradients(self) -> List[Tuple[XPBDParticle, Vec3]]:
        """
        Compute constraint gradients for each particle.

        Returns:
            List of (particle, gradient) tuples.
        """
        ...

    def solve(self, dt: float, lambda_accumulated: float) -> float:
        """
        Solve constraint and update positions.

        Args:
            dt: Time step.
            lambda_accumulated: Accumulated Lagrange multiplier.

        Returns:
            New accumulated Lagrange multiplier.
        """
        ...


@dataclass
class XPBDDistanceConstraint:
    """
    Distance constraint for XPBD.

    Maintains a fixed distance between two particles.
    """
    particle_a: XPBDParticle
    particle_b: XPBDParticle
    rest_length: float = 1.0
    compliance: float = 0.0  # 0 = rigid
    damping: float = 0.0
    _lambda: float = 0.0

    def compute_constraint(self) -> float:
        """Compute constraint value C = |p2 - p1| - rest_length."""
        diff = self.particle_b.position - self.particle_a.position
        return diff.length() - self.rest_length

    def compute_gradients(self) -> List[Tuple[XPBDParticle, Vec3]]:
        """Compute gradients: grad_C = normalized direction."""
        diff = self.particle_b.position - self.particle_a.position
        length = diff.length()

        if length < 1e-10:
            direction = Vec3.unit_x()
        else:
            direction = diff / length

        return [
            (self.particle_a, -direction),
            (self.particle_b, direction)
        ]

    def solve(self, dt: float) -> float:
        """
        Solve distance constraint.

        Args:
            dt: Time step.

        Returns:
            Position correction magnitude.
        """
        # Compute constraint value
        c = self.compute_constraint()

        if abs(c) < 1e-10:
            return 0.0

        # Compute gradients
        diff = self.particle_b.position - self.particle_a.position
        length = diff.length()

        if length < 1e-10:
            return 0.0

        direction = diff / length

        # Compute effective mass: w = sum(w_i * |grad_i|^2)
        w = 0.0
        if not self.particle_a.is_static:
            w += self.particle_a.inv_mass
        if not self.particle_b.is_static:
            w += self.particle_b.inv_mass

        if w < 1e-10:
            return 0.0

        # XPBD: alpha = compliance / dt^2
        alpha = self.compliance / (dt * dt)

        # Compute Lagrange multiplier update
        # delta_lambda = (-C - alpha * lambda) / (w + alpha)
        delta_lambda = (-c - alpha * self._lambda) / (w + alpha)

        # Apply damping
        if self.damping > 0:
            # Compute relative velocity along constraint
            rel_vel = (self.particle_b.velocity - self.particle_a.velocity).dot(direction)
            damping_force = self.damping * rel_vel / (w + alpha)
            delta_lambda -= damping_force * dt

        self._lambda += delta_lambda

        # Apply position corrections
        if not self.particle_a.is_static:
            self.particle_a.position = self.particle_a.position - direction * (
                delta_lambda * self.particle_a.inv_mass
            )

        if not self.particle_b.is_static:
            self.particle_b.position = self.particle_b.position + direction * (
                delta_lambda * self.particle_b.inv_mass
            )

        return abs(c)

    def reset_lambda(self) -> None:
        """Reset Lagrange multiplier for new frame."""
        self._lambda = 0.0


@dataclass
class XPBDBendingConstraint:
    """
    Bending constraint for XPBD.

    Maintains angle between three particles.
    """
    particle_a: XPBDParticle  # End particle 1
    particle_b: XPBDParticle  # Center particle
    particle_c: XPBDParticle  # End particle 2
    rest_angle: float = math.pi  # Default: straight
    compliance: float = 0.001  # Bending is usually soft
    damping: float = 0.0
    _lambda: float = 0.0

    def compute_constraint(self) -> float:
        """Compute constraint value C = angle - rest_angle."""
        ab = self.particle_a.position - self.particle_b.position
        cb = self.particle_c.position - self.particle_b.position

        ab_len = ab.length()
        cb_len = cb.length()

        if ab_len < 1e-10 or cb_len < 1e-10:
            return 0.0

        cos_angle = ab.dot(cb) / (ab_len * cb_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)

        return angle - self.rest_angle

    def solve(self, dt: float) -> float:
        """Solve bending constraint."""
        c = self.compute_constraint()

        if abs(c) < 1e-10:
            return 0.0

        ab = self.particle_a.position - self.particle_b.position
        cb = self.particle_c.position - self.particle_b.position

        ab_len = ab.length()
        cb_len = cb.length()

        if ab_len < 1e-10 or cb_len < 1e-10:
            return 0.0

        ab_norm = ab / ab_len
        cb_norm = cb / cb_len

        # Compute gradients
        # For angle constraint, gradients are perpendicular to the edges
        cos_angle = ab.dot(cb) / (ab_len * cb_len)
        sin_angle = math.sqrt(max(0.0, 1.0 - cos_angle * cos_angle))

        if sin_angle < 1e-10:
            return 0.0

        # Gradient for particle A
        grad_a = (cb_norm - ab_norm * cos_angle) / (ab_len * sin_angle)

        # Gradient for particle C
        grad_c = (ab_norm - cb_norm * cos_angle) / (cb_len * sin_angle)

        # Gradient for particle B
        grad_b = -(grad_a + grad_c)

        # Compute effective mass
        w = 0.0
        if not self.particle_a.is_static:
            w += self.particle_a.inv_mass * grad_a.dot(grad_a)
        if not self.particle_b.is_static:
            w += self.particle_b.inv_mass * grad_b.dot(grad_b)
        if not self.particle_c.is_static:
            w += self.particle_c.inv_mass * grad_c.dot(grad_c)

        if w < 1e-10:
            return 0.0

        # XPBD
        alpha = self.compliance / (dt * dt)
        delta_lambda = (-c - alpha * self._lambda) / (w + alpha)
        self._lambda += delta_lambda

        # Apply corrections
        if not self.particle_a.is_static:
            self.particle_a.position = self.particle_a.position + grad_a * (
                delta_lambda * self.particle_a.inv_mass
            )

        if not self.particle_b.is_static:
            self.particle_b.position = self.particle_b.position + grad_b * (
                delta_lambda * self.particle_b.inv_mass
            )

        if not self.particle_c.is_static:
            self.particle_c.position = self.particle_c.position + grad_c * (
                delta_lambda * self.particle_c.inv_mass
            )

        return abs(c)

    def reset_lambda(self) -> None:
        """Reset Lagrange multiplier."""
        self._lambda = 0.0


@dataclass
class XPBDVolumeConstraint:
    """
    Volume preservation constraint for tetrahedra.

    Maintains the volume of a tetrahedron formed by four particles.
    """
    particles: List[XPBDParticle] = field(default_factory=list)
    rest_volume: float = 1.0
    compliance: float = 0.0
    _lambda: float = 0.0

    def __post_init__(self):
        if len(self.particles) != 4:
            raise ValueError("Volume constraint requires exactly 4 particles")

    def compute_volume(self) -> float:
        """Compute current tetrahedron volume."""
        p0 = self.particles[0].position
        p1 = self.particles[1].position
        p2 = self.particles[2].position
        p3 = self.particles[3].position

        # Volume = (1/6) * |((p1-p0) x (p2-p0)) . (p3-p0)|
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0

        return abs(v1.cross(v2).dot(v3)) / 6.0

    def compute_constraint(self) -> float:
        """Compute constraint value C = volume - rest_volume."""
        return self.compute_volume() - self.rest_volume

    def solve(self, dt: float) -> float:
        """Solve volume constraint."""
        c = self.compute_constraint()

        if abs(c) < 1e-10:
            return 0.0

        p0 = self.particles[0].position
        p1 = self.particles[1].position
        p2 = self.particles[2].position
        p3 = self.particles[3].position

        # Compute gradients (derivative of volume w.r.t each particle)
        # grad_0 = (p2-p1) x (p3-p1) / 6
        # grad_1 = (p3-p0) x (p2-p0) / 6
        # grad_2 = (p1-p0) x (p3-p0) / 6
        # grad_3 = (p2-p0) x (p1-p0) / 6

        grad_0 = (p2 - p1).cross(p3 - p1) / 6.0
        grad_1 = (p3 - p0).cross(p2 - p0) / 6.0
        grad_2 = (p1 - p0).cross(p3 - p0) / 6.0
        grad_3 = (p2 - p0).cross(p1 - p0) / 6.0

        grads = [grad_0, grad_1, grad_2, grad_3]

        # Compute effective mass
        w = 0.0
        for i, particle in enumerate(self.particles):
            if not particle.is_static:
                w += particle.inv_mass * grads[i].dot(grads[i])

        if w < 1e-10:
            return 0.0

        # XPBD
        alpha = self.compliance / (dt * dt)
        delta_lambda = (-c - alpha * self._lambda) / (w + alpha)
        self._lambda += delta_lambda

        # Apply corrections
        for i, particle in enumerate(self.particles):
            if not particle.is_static:
                particle.position = particle.position + grads[i] * (
                    delta_lambda * particle.inv_mass
                )

        return abs(c)

    def reset_lambda(self) -> None:
        """Reset Lagrange multiplier."""
        self._lambda = 0.0


@dataclass
class XPBDCollisionConstraint:
    """
    Collision constraint for XPBD.

    Handles particle-plane or particle-particle collisions.

    contact_normal convention:
        The normal is an inward-pointing surface normal (from surface
        toward the particle interior).  A positive position correction
        along ``+contact_normal`` therefore pushes the particle away
        from the contact surface.

        Constraint function:  C(x) = separation = -penetration.
        When the particle overlaps the surface (penetration > 0),
        C < 0, so the XPBD solver produces a positive delta-lambda
        that pushes the particle outward.
    """
    particle: XPBDParticle
    contact_point: Vec3 = field(default_factory=Vec3.zero)
    contact_normal: Vec3 = field(default_factory=Vec3.unit_y)
    penetration: float = 0.0
    friction: float = 0.4
    compliance: float = 0.0  # Rigid contacts
    _lambda_n: float = 0.0
    _lambda_t: float = 0.0

    def solve(self, dt: float) -> float:
        """Solve collision constraint."""
        if self.penetration <= 0:
            return 0.0

        if self.particle.is_static:
            return 0.0

        # Normal constraint
        # C(x) = -penetration  (separation, positive = non-overlapping)
        c = -self.penetration
        w = self.particle.inv_mass

        if w < 1e-10:
            return 0.0

        alpha = self.compliance / (dt * dt)
        delta_lambda_n = (-c - alpha * self._lambda_n) / (w + alpha)

        # Contact can only push (lambda >= 0)
        new_lambda = max(0.0, self._lambda_n + delta_lambda_n)
        delta_lambda_n = new_lambda - self._lambda_n
        self._lambda_n = new_lambda

        # Apply normal correction
        self.particle.position = self.particle.position + self.contact_normal * (
            delta_lambda_n * self.particle.inv_mass
        )

        # Friction
        if self.friction > 0 and self._lambda_n > 0:
            # Compute tangential velocity
            vel = self.particle.velocity
            vel_n = vel.dot(self.contact_normal)
            vel_t = vel - self.contact_normal * vel_n

            vel_t_len = vel_t.length()
            if vel_t_len > 1e-10:
                tangent = vel_t / vel_t_len

                # Friction impulse magnitude (Coulomb model)
                #   max_friction = mu * lambda_n  (Coulomb cone)
                #   stopping impulse = m * |v_t| = |v_t| / inv_mass
                # Scaled by dt so friction is consistent across substep sizes.
                max_friction = self.friction * self._lambda_n
                friction_impulse = min(
                    max_friction,
                    vel_t_len / self.particle.inv_mass * dt,
                )

                # Apply friction: delta_v = J * inv_mass
                self.particle.velocity = self.particle.velocity - tangent * (
                    friction_impulse * self.particle.inv_mass
                )

        return abs(c)  # Consistent with other constraints (solver convergence)

    def reset_lambda(self) -> None:
        """Reset Lagrange multipliers."""
        self._lambda_n = 0.0
        self._lambda_t = 0.0


class XPBDSolver:
    """
    Extended Position Based Dynamics Solver.

    Solves constraints using position-based approach with
    compliance parameters for soft constraints.

    Attributes:
        config: Solver configuration.
        particles: List of particles.
        constraints: List of XPBD constraints.
        gravity: Gravity acceleration.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize XPBD solver."""
        self.config = config or SolverConfig.xpbd_default()
        self.particles: List[XPBDParticle] = []
        self.constraints: List = []  # Any type with solve() method
        self.gravity: Vec3 = Vec3(0, -9.81, 0)
        self._substep_dt: float = 0.0

    def add_particle(self, particle: XPBDParticle) -> None:
        """Add a particle to the simulation."""
        self.particles.append(particle)

    def remove_particle(self, particle: XPBDParticle) -> None:
        """Remove a particle from the simulation."""
        if particle in self.particles:
            self.particles.remove(particle)

    def add_constraint(self, constraint) -> None:
        """Add a constraint to the solver."""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint) -> None:
        """Remove a constraint from the solver."""
        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def clear(self) -> None:
        """Clear all particles and constraints."""
        self.particles.clear()
        self.constraints.clear()

    def solve(self, dt: float) -> None:
        """
        Solve all constraints for one physics step.

        Args:
            dt: Time step.
        """
        num_substeps = self.config.substeps
        self._substep_dt = dt / num_substeps

        for _ in range(num_substeps):
            self._solve_substep(self._substep_dt)

    def _solve_substep(self, sub_dt: float) -> None:
        """Solve one substep."""
        # Store previous positions
        for particle in self.particles:
            particle.store_state()

        # Predict positions (apply external forces)
        for particle in self.particles:
            particle.predict_position(sub_dt, self.gravity)

        # Reset Lagrange multipliers
        for constraint in self.constraints:
            if hasattr(constraint, 'reset_lambda'):
                constraint.reset_lambda()

        # Solve constraints iteratively
        for _ in range(self.config.position_iterations):
            max_error = 0.0
            for constraint in self.constraints:
                error = constraint.solve(sub_dt)
                max_error = max(max_error, error)

            if max_error < 1e-6:
                break

        # Update velocities from position changes
        for particle in self.particles:
            particle.update_velocity(sub_dt)

        # Velocity damping
        self._apply_velocity_damping(sub_dt)

    def _apply_velocity_damping(self, dt: float) -> None:
        """Apply global velocity damping."""
        damping = XPBD_VELOCITY_DAMPING  # From config for tunability

        for particle in self.particles:
            if not particle.is_static:
                particle.velocity = particle.velocity * damping
                particle.angular_velocity = particle.angular_velocity * damping

    def solve_position(self, iterations: Optional[int] = None) -> float:
        """
        Solve only position constraints.

        Args:
            iterations: Number of iterations.

        Returns:
            Maximum constraint error.
        """
        num_iterations = iterations or self.config.position_iterations
        max_error = 0.0

        for _ in range(num_iterations):
            max_error = 0.0
            for constraint in self.constraints:
                error = constraint.solve(self._substep_dt)
                max_error = max(max_error, error)

            if max_error < 1e-6:
                break

        return max_error

    def get_kinetic_energy(self) -> float:
        """Compute total kinetic energy of the system."""
        energy = 0.0

        for particle in self.particles:
            if not particle.is_static and particle.inv_mass > 0:
                mass = 1.0 / particle.inv_mass
                vel_sq = particle.velocity.length_squared()
                energy += 0.5 * mass * vel_sq

        return energy


class XPBDRigidBodyConstraint:
    """
    Rigid body constraint for XPBD.

    Adapts rigid body constraints to work with XPBD solver.
    """

    def __init__(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody],
        local_anchor_a: Vec3 = None,
        local_anchor_b: Vec3 = None,
        compliance: float = 0.0,
        damping: float = 0.0
    ):
        """
        Initialize rigid body constraint.

        Args:
            body_a: First rigid body.
            body_b: Second rigid body (or None for world constraint).
            local_anchor_a: Anchor point in body A's local space.
            local_anchor_b: Anchor point in body B's local space (or world space if body_b is None).
            compliance: Compliance (inverse stiffness).
            damping: Damping coefficient.
        """
        self.body_a = body_a
        self.body_b = body_b
        self.local_anchor_a = local_anchor_a or Vec3.zero()
        self.local_anchor_b = local_anchor_b or Vec3.zero()
        self.compliance = compliance
        self.damping = damping
        self._lambda: float = 0.0

    def solve(self, dt: float) -> float:
        """
        Solve the rigid body constraint.

        Args:
            dt: Time step.

        Returns:
            Constraint error.
        """
        # Transform anchors to world space
        anchor_a = self.body_a.local_to_world(self.local_anchor_a)

        if self.body_b is not None:
            anchor_b = self.body_b.local_to_world(self.local_anchor_b)
        else:
            anchor_b = self.local_anchor_b  # World space

        # Compute error
        diff = anchor_b - anchor_a
        c = diff.length()

        if c < 1e-10:
            return 0.0

        direction = diff / c

        # Compute r vectors
        r_a = anchor_a - self.body_a.position
        r_b = (anchor_b - self.body_b.position) if self.body_b else Vec3.zero()

        # Compute effective mass
        w = 0.0

        if not self.body_a.is_static:
            w += self.body_a.inv_mass
            # Angular contribution
            r_cross_n_a = r_a.cross(direction)
            w += r_cross_n_a.dot(self.body_a.inv_inertia_world * r_cross_n_a)

        if self.body_b is not None and not self.body_b.is_static:
            w += self.body_b.inv_mass
            r_cross_n_b = r_b.cross(direction)
            w += r_cross_n_b.dot(self.body_b.inv_inertia_world * r_cross_n_b)

        if w < 1e-10:
            return c

        # XPBD
        alpha = self.compliance / (dt * dt)
        delta_lambda = (-c - alpha * self._lambda) / (w + alpha)
        self._lambda += delta_lambda

        # Apply position corrections
        if not self.body_a.is_static:
            # Linear correction
            self.body_a.position = self.body_a.position - direction * (
                delta_lambda * self.body_a.inv_mass
            )

            # Angular correction
            angular_impulse = r_a.cross(direction) * delta_lambda
            delta_q = self._angular_impulse_to_rotation(
                angular_impulse,
                self.body_a.inv_inertia_world
            )
            self.body_a.orientation = (delta_q * self.body_a.orientation).normalized()

        if self.body_b is not None and not self.body_b.is_static:
            # Linear correction
            self.body_b.position = self.body_b.position + direction * (
                delta_lambda * self.body_b.inv_mass
            )

            # Angular correction
            angular_impulse = r_b.cross(direction) * (-delta_lambda)
            delta_q = self._angular_impulse_to_rotation(
                angular_impulse,
                self.body_b.inv_inertia_world
            )
            self.body_b.orientation = (delta_q * self.body_b.orientation).normalized()

        return c

    def _angular_impulse_to_rotation(
        self,
        impulse: Vec3,
        inv_inertia: Mat3
    ) -> Quaternion:
        """Convert angular impulse to quaternion rotation."""
        delta_omega = inv_inertia * impulse
        angle = delta_omega.length()

        if angle < 1e-10:
            return Quaternion.identity()

        axis = delta_omega / angle
        return Quaternion.from_axis_angle(axis, angle)

    def reset_lambda(self) -> None:
        """Reset Lagrange multiplier."""
        self._lambda = 0.0
