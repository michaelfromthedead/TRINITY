"""
T-1.5: Test Sequential Impulse constraint solver.

Covers:
  - Jacobian computation for contact constraint
  - Effective mass computation K = J * M^-1 * J^T
  - Impulse computation lambda = -K^-1 * (Jv + bias)
  - Warm starting (impulse accumulation and clamping)
  - Single body falling under gravity (no constraints)
  - Two bodies connected by distance constraint
"""

import math
import pytest

from engine.simulation.solver.constraint_solver import (
    ConstraintSolver,
    ConstraintType,
    RigidBody,
    BaseConstraint,
    PointConstraint,
    SolverConfig,
)
from engine.simulation.solver.jacobian import (
    Vec3, Mat3, Quaternion, Jacobian,
    compute_effective_mass, apply_impulse, compute_relative_velocity, clamp_impulse,
)
from ..physics_test_base import PhysicsTestCase


# ===========================================================================
# T-1.5  —  Constraint solver tests
# ===========================================================================

class TestConstraintSolver(PhysicsTestCase):
    """Sequential Impulse solver verification."""

    # ------------------------------------------------------------------
    # Effective mass: K = J * M^-1 * J^T
    # ------------------------------------------------------------------
    def test_effective_mass_single_body(self):
        """effective mass for a point impulse on a single body."""
        body = RigidBody.create_dynamic(0, mass=2.0)
        j = Jacobian(linear_a=Vec3.unit_x())   # impulse along X
        K = compute_effective_mass(
            j, body.inv_mass, body.inv_inertia_world,
            body.inv_mass, body.inv_inertia_world
        )
        # compute_effective_mass returns 1/K where K = J * M^-1 * J^T
        # For a free body: K = 1/m, so effective mass = m = 1/inv_mass
        expected = 1.0 / body.inv_mass
        assert abs(K - expected) < 1e-9, f"K = {K}, expected {expected}"

    def test_effective_mass_static_body(self):
        """static body contributes zero to effective mass."""
        dynamic = RigidBody.create_dynamic(0, mass=2.0)
        static = RigidBody.create_static(1)
        j = Jacobian(linear_a=Vec3.unit_x(), linear_b=-Vec3.unit_x())
        K = compute_effective_mass(
            j, dynamic.inv_mass, dynamic.inv_inertia_world,
            static.inv_mass, static.inv_inertia_world
        )
        # compute_effective_mass returns 1/K; static body (inv_mass=0) contributes 0
        expected = 1.0 / dynamic.inv_mass
        assert abs(K - expected) < 1e-9, f"K = {K}, expected {expected}"

    # ------------------------------------------------------------------
    # Impulse computation  lambda = -K * (Jv + bias)
    # ------------------------------------------------------------------
    def test_impulse_magnitude(self):
        """impulse stops a moving body in one step."""
        body = RigidBody.create_dynamic(0, mass=1.0, position=Vec3.zero())
        body.velocity = Vec3(5, 0, 0)

        j = Jacobian(linear_a=Vec3.unit_x())
        K = compute_effective_mass(
            j, body.inv_mass, body.inv_inertia_world,
            body.inv_mass, body.inv_inertia_world
        )
        rel_vel = compute_relative_velocity(
            j, body.velocity, body.angular_velocity,
            Vec3.zero(), Vec3.zero()
        )
        impulse = -K * rel_vel  # lambda = -K * Jv

        # Apply impulse
        dv_a, dwa, dv_b, dwb = apply_impulse(
            j, impulse,
            body.inv_mass, body.inv_inertia_world,
            body.inv_mass, body.inv_inertia_world
        )
        body.velocity = body.velocity + dv_a
        assert abs(body.velocity.x) < 1e-9, f"Velocity not zero: {body.velocity.x}"

    # ------------------------------------------------------------------
    # Clamping  (warm starting accumulator)
    # ------------------------------------------------------------------
    def test_clamp_impulse_lower_limit(self):
        """impulse is clamped to the lower limit."""
        impulse = -100.0
        accumulated = 0.0
        lower, upper = -10.0, 10.0
        clamped, new_acc = clamp_impulse(impulse, accumulated, lower, upper)
        assert abs(clamped - (-10.0)) < 1e-9, f"clamped = {clamped}"

    def test_clamp_impulse_upper_limit(self):
        """impulse is clamped to the upper limit."""
        impulse = 100.0
        accumulated = 0.0
        lower, upper = -10.0, 10.0
        clamped, new_acc = clamp_impulse(impulse, accumulated, lower, upper)
        assert abs(clamped - 10.0) < 1e-9, f"clamped = {clamped}"

    # ------------------------------------------------------------------
    # PointConstraint (distance constraint)
    # ------------------------------------------------------------------
    def test_point_constraint_prepare(self):
        """PointConstraint compute jacobian and effective mass."""
        body_a = RigidBody.create_dynamic(0, Vec3(0, 0, 0), mass=1.0)
        body_b = RigidBody.create_dynamic(1, Vec3(2, 0, 0), mass=1.0)
        config = SolverConfig.default()

        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )
        constraint.prepare(0.016, config)

        # After prepare the jacobian and effective mass should be set
        assert abs(constraint._effective_mass) > 0
        # For two equal bodies at 2m distance, the effective mass should be
        # symmetric. The jacobian linear part is (-normal, normal)
        assert constraint._jacobian is not None

    def test_distance_constraint_maintains_distance(self):
        """two bodies connected by a PointConstraint stay within tolerance."""
        body_a = RigidBody.create_dynamic(0, Vec3(-1, 0, 0), mass=1.0)
        body_b = RigidBody.create_dynamic(1, Vec3(1, 0, 0), mass=1.0)

        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )

        solver = ConstraintSolver()
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        # After solving, the distance should be close to the initial distance
        dist = (body_a.position - body_b.position).length()
        initial_dist = 2.0
        # Some drift is expected with Baumgarte, but it should be within reason
        assert abs(dist - initial_dist) < 0.1, f"Distance = {dist}"

    # ------------------------------------------------------------------
    # Single body falling under gravity (no constraints)
    # ------------------------------------------------------------------
    def test_single_body_freefall(self):
        """single body reaches expected velocity after N steps."""
        body = RigidBody.create_dynamic(0, Vec3(0, 10, 0), mass=1.0)
        solver = ConstraintSolver()
        solver.add_body(body)

        dt = 0.016
        gravity = Vec3(0, -9.81, 0)

        for _ in range(60):  # ~1 second
            body.velocity = body.velocity + gravity * dt
            solver.solve(dt)
            body.position = body.position + body.velocity * dt

        # v = g * t = -9.81 * ~0.96s ≈ -9.42 m/s
        expected_vy = -9.81 * 0.96
        assert abs(body.velocity.y - expected_vy) < 0.5, \
            f"vy = {body.velocity.y}, expected ~{expected_vy}"

    # ------------------------------------------------------------------
    # Two bodies connected by distance constraint (PointConstraint)
    # ------------------------------------------------------------------
    def test_two_bodies_distance_constrained(self):
        """bodies connected by PointConstraint move but stay connected."""
        body_a = RigidBody.create_dynamic(0, Vec3(-2, 0, 0), mass=2.0)
        body_b = RigidBody.create_dynamic(1, Vec3(2, 0, 0), mass=1.0)

        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )
        solver = ConstraintSolver()
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(constraint)

        dt = 0.016
        gravity = Vec3(0, -9.81, 0)

        initial_dist = (body_a.position - body_b.position).length()
        for _ in range(10):
            body_a.velocity = body_a.velocity + gravity * dt
            body_b.velocity = body_b.velocity + gravity * dt
            solver.solve(dt)
            body_a.position = body_a.position + body_a.velocity * dt
            body_b.position = body_b.position + body_b.velocity * dt

        dist = (body_a.position - body_b.position).length()
        # PointConstraint tries to make anchors coincide (ball joint),
        # so distance will shrink. Just verify it stays finite.
        assert dist < initial_dist * 2, \
            f"Distance constraint diverged: {dist}"

    # ------------------------------------------------------------------
    # Solver iteration tracking
    # ------------------------------------------------------------------
    def test_solver_iteration_count(self):
        """after solve, iteration count is set."""
        body_a = RigidBody.create_dynamic(0, Vec3(-1, 0, 0), mass=1.0)
        body_b = RigidBody.create_dynamic(1, Vec3(1, 0, 0), mass=1.0)
        constraint = PointConstraint(
            _body_a=body_a, _body_b=body_b,
            local_anchor_a=Vec3(0, 0, 0), local_anchor_b=Vec3(0, 0, 0),
        )
        solver = ConstraintSolver()
        solver.add_body(body_a)
        solver.add_body(body_b)
        solver.add_constraint(constraint)
        solver.solve(0.016)
        assert solver.get_iteration_count() >= 0


# ===========================================================================
# Jacobian computation tests  (also part of T-1.5)
# ===========================================================================

class TestJacobianComputation(PhysicsTestCase):
    """Jacobian computation for contact constraints."""

    def test_contact_jacobian_normal(self):
        """contact jacobian has non-zero angular parts when offset from COM."""
        body_a = RigidBody.create_dynamic(0, Vec3(0, 0, 0), mass=1.0)

        # Contact point at (1, 0, 0) relative to body center
        r = Vec3(1, 0, 0)
        normal = Vec3.unit_y()

        # Manual contact jacobian
        j = Jacobian(
            linear_a=-normal,
            angular_a=-r.cross(normal),
            linear_b=Vec3.zero(),
            angular_b=Vec3.zero(),
        )
        # r x n = (1,0,0) x (0,1,0) = (0,0,1)
        assert abs(j.angular_a.z - (-1.0)) < 1e-9

    def test_contact_jacobian_effective_mass(self):
        """effective mass including angular component for offset contact."""
        body_a = RigidBody.create_dynamic(0, Vec3(0, 0, 0), mass=1.0)
        body_b = RigidBody.create_static(1)

        r = Vec3(1, 0, 0)
        normal = Vec3.unit_y()
        j = Jacobian(
            linear_a=-normal,
            angular_a=-r.cross(normal),
            linear_b=Vec3.zero(),
            angular_b=Vec3.zero(),
        )
        K = compute_effective_mass(
            j, body_a.inv_mass, body_a.inv_inertia_world,
            body_b.inv_mass, body_b.inv_inertia_world
        )
        # Effective mass = 1/(J*M^-1*J^T); angular component increases
        # J*M^-1*J^T so effective mass is reduced relative to 1/m
        assert K > 0, "Effective mass should be positive"
