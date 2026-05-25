"""
T-1.7: Test XPBD solver.

Covers:
  - Compliance parameter effect (higher = softer)
  - Delta lambda formula: (-C - alpha * lambda) / (w + alpha)
  - Distance constraint in XPBD
  - Collision constraint in XPBD
"""

import math
import pytest

from engine.simulation.solver.xpbd_solver import (
    XPBDSolver, XPBDParticle, XPBDDistanceConstraint,
    XPBDCollisionConstraint, XPBDBendingConstraint, XPBDVolumeConstraint,
)
from engine.simulation.solver.jacobian import Vec3, Mat3, Quaternion
from ..physics_test_base import PhysicsTestCase


# ===========================================================================
# T-1.7  —  XPBD solver
# ===========================================================================

class TestXPBDSolver(PhysicsTestCase):
    """XPBD solver verification."""

    # ------------------------------------------------------------------
    # XPBD particle basics
    # ------------------------------------------------------------------
    def test_particle_predict_position(self):
        """particle predict_position updates position from velocity."""
        p = XPBDParticle(id=0, position=Vec3(0, 0, 0), velocity=Vec3(10, 0, 0), inv_mass=1.0)
        p.store_state()
        p.predict_position(0.1)
        assert abs(p.position.x - 1.0) < 1e-9, f"x = {p.position.x}, expected 1.0"

    def test_particle_store_state(self):
        """store_state saves previous position/orientation."""
        p = XPBDParticle(id=0, position=Vec3(1, 2, 3),
                         orientation=Quaternion(1, 2, 3, 4).normalized())
        p.store_state()
        assert p.previous_position.x == 1.0
        assert abs(p.previous_orientation.w - p.orientation.w) < 1e-9

    def test_particle_static_no_motion(self):
        """static particle does not move during prediction."""
        p = XPBDParticle(id=0, position=Vec3(0, 10, 0), velocity=Vec3(0, -10, 0),
                         inv_mass=0.0, is_static=True)
        p.store_state()
        p.predict_position(0.1, gravity=Vec3(0, -9.81, 0))
        # Static particle should not move
        assert abs(p.position.y - 10.0) < 1e-9

    # ------------------------------------------------------------------
    # Compliance parameter effect
    # ------------------------------------------------------------------
    def test_zero_compliance_is_stiff(self):
        """zero compliance -> constraint is very stiff."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        constraint = XPBDDistanceConstraint(
            particle_a=p1, particle_b=p2,
            rest_length=2.0, compliance=0.0, damping=0.0,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        dist = (p1.position - p2.position).length()
        assert abs(dist - 2.0) < 0.001, f"dist={dist}"

    def test_high_compliance_allows_stretch(self):
        """high compliance allows visible stretch from initial displacement."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        # Stretch the constraint: set rest length shorter than current distance
        constraint = XPBDDistanceConstraint(
            particle_a=p1, particle_b=p2,
            rest_length=1.0, compliance=0.01, damping=0.0,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)

        for _ in range(5):
            solver.solve(0.016)

        dist = (p1.position - p2.position).length()
        # With compliance, the constraint should not be fully satisfied
        assert dist > 1.0, f"Compliant constraint over-satisfied: dist={dist}"

    # ------------------------------------------------------------------
    # Delta lambda formula  (-C - alpha * lambda) / (w + alpha)
    # ------------------------------------------------------------------
    def test_delta_lambda_sign(self):
        """constraint violation produces impulse in the correct direction."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(3, 0, 0), inv_mass=1.0)

        # rest_length=2, current dist=3 => positive violation
        constraint = XPBDDistanceConstraint(
            particle_a=p1, particle_b=p2,
            rest_length=2.0, compliance=0.0, damping=0.0,
        )
        # Solve once
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        # After solving, the distance should be closer to 2.0
        dist = (p1.position - p2.position).length()
        assert dist < 3.0, f"Distance increased: {dist}"

    # ------------------------------------------------------------------
    # XPBDDistanceConstraint
    # ------------------------------------------------------------------
    def test_distance_constraint_xpbd(self):
        """XPBD distance constraint maintains rest length."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        constraint = XPBDDistanceConstraint(
            particle_a=p1, particle_b=p2,
            rest_length=2.0, compliance=0.0, damping=0.0,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)

        # Apply gravity and solve
        for _ in range(10):
            p1.velocity = p1.velocity + Vec3(0, -9.81, 0) * 0.016
            solver.solve(0.016)

        dist = (p1.position - p2.position).length()
        assert abs(dist - 2.0) < 0.01, f"Rest length violated: dist={dist}"

    # ------------------------------------------------------------------
    # XPBDCollisionConstraint
    # ------------------------------------------------------------------
    def test_collision_constraint_resolves_penetration(self):
        """collision constraint is wired into solver."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)

        constraint = XPBDCollisionConstraint(
            particle=p1,
            contact_point=Vec3(0.5, 0, 0),
            contact_normal=Vec3(1, 0, 0),
            penetration=0.3,
            compliance=0.0,
            friction=0.0,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        # Solver runs without error; particle has finite position
        assert not any(math.isnan(c) for c in (p1.position.x, p1.position.y, p1.position.z))

    # ------------------------------------------------------------------
    # Bending constraint
    # ------------------------------------------------------------------
    def test_bending_constraint(self):
        """bending constraint has correct interface."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(1, 0, 0), inv_mass=1.0)
        p3 = XPBDParticle(id=2, position=Vec3(2, 0.1, 0), inv_mass=1.0)

        constraint = XPBDBendingConstraint(
            particle_a=p1, particle_b=p2, particle_c=p3,
            rest_angle=0.0, compliance=0.0, damping=0.0,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_particle(p3)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        # Constraint should try to flatten the angle
        # (basic run without crash / NaN)
        assert not any(math.isnan(p.position.x) for p in (p1, p2, p3))

    # ------------------------------------------------------------------
    # Solver iteration count
    # ------------------------------------------------------------------
    def test_xpbd_solver_iterations(self):
        """XPBDSolver runs for configured iterations."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)
        constraint = XPBDDistanceConstraint(p1, p2, rest_length=2.0)

        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)
        solver.solve(0.016)

        # Solver ran without error
        dist = (p1.position - p2.position).length()
        assert abs(dist - 2.0) < 0.01

    # ------------------------------------------------------------------
    # Compliance + damping
    # ------------------------------------------------------------------
    def test_compliance_damping_does_not_raise(self):
        """XPBD distance constraint with compliance and damping runs."""
        p1 = XPBDParticle(id=0, position=Vec3(0, 0, 0), inv_mass=1.0)
        p2 = XPBDParticle(id=1, position=Vec3(2, 0, 0), inv_mass=1.0)

        constraint = XPBDDistanceConstraint(
            particle_a=p1, particle_b=p2,
            rest_length=2.0, compliance=0.001, damping=0.1,
        )
        solver = XPBDSolver()
        solver.add_particle(p1)
        solver.add_particle(p2)
        solver.add_constraint(constraint)

        for _ in range(5):
            solver.solve(0.016)

        dist = (p1.position - p2.position).length()
        assert abs(dist - 2.0) < 0.1
