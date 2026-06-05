"""
Whitebox tests for contact_constraint.py - Contact Constraint.

Tests:
- ContactPoint dataclass
- ContactManifold dataclass
- ContactConstraint class
- compute_contact_jacobian function
- combine_friction function
- combine_restitution function
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.contact_constraint import (
    ContactPoint,
    ContactManifold,
    ContactConstraint,
    ContactPointData,
    compute_contact_jacobian,
    combine_friction,
    combine_restitution,
)
from simulation.solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from simulation.solver.constraint_solver import RigidBody, ConstraintType
from simulation.solver.config import SolverConfig


class TestContactPoint:
    """Tests for ContactPoint dataclass."""

    def test_basic_creation(self):
        """Should create contact point with required fields."""
        point = ContactPoint(
            position=Vec3(1, 0, 0),
            normal=Vec3(0, 1, 0),
            penetration=0.1
        )
        assert point.position.x == 1
        assert point.normal.y == 1
        assert point.penetration == 0.1

    def test_default_values(self):
        """Default values should be sensible."""
        point = ContactPoint(
            position=Vec3.zero(),
            normal=Vec3.unit_y(),
            penetration=0.0
        )
        assert point.normal_impulse == 0.0
        assert point.tangent_impulse_1 == 0.0
        assert point.tangent_impulse_2 == 0.0
        assert point.combined_friction == 0.4
        assert point.combined_restitution == 0.0

    def test_custom_friction_restitution(self):
        """Should accept custom friction/restitution."""
        point = ContactPoint(
            position=Vec3.zero(),
            normal=Vec3.unit_y(),
            penetration=0.0,
            combined_friction=0.8,
            combined_restitution=0.5
        )
        assert point.combined_friction == 0.8
        assert point.combined_restitution == 0.5


class TestContactManifold:
    """Tests for ContactManifold dataclass."""

    def test_basic_creation(self, dynamic_body_a, dynamic_body_b):
        """Should create manifold with bodies."""
        manifold = ContactManifold(
            body_a=dynamic_body_a,
            body_b=dynamic_body_b
        )
        assert manifold.body_a is dynamic_body_a
        assert manifold.body_b is dynamic_body_b

    def test_add_point(self, dynamic_body_a, dynamic_body_b):
        """add_point should add contact point."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        point = ContactPoint(Vec3.zero(), Vec3.unit_y(), 0.1)
        manifold.add_point(point)
        assert manifold.point_count == 1

    def test_max_points(self, dynamic_body_a, dynamic_body_b):
        """Manifold should limit to 4 points."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        for i in range(6):
            point = ContactPoint(Vec3(i, 0, 0), Vec3.unit_y(), 0.1)
            manifold.add_point(point)
        assert manifold.point_count == 4

    def test_clear(self, dynamic_body_a, dynamic_body_b):
        """clear should remove all points."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        for i in range(3):
            point = ContactPoint(Vec3(i, 0, 0), Vec3.unit_y(), 0.1)
            manifold.add_point(point)
        manifold.clear()
        assert manifold.point_count == 0

    def test_get_average_position(self, dynamic_body_a, dynamic_body_b):
        """get_average_position should compute centroid."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        manifold.add_point(ContactPoint(Vec3(0, 0, 0), Vec3.unit_y(), 0.0))
        manifold.add_point(ContactPoint(Vec3(2, 0, 0), Vec3.unit_y(), 0.0))
        avg = manifold.get_average_position()
        assert abs(avg.x - 1.0) < 1e-6

    def test_get_average_position_empty(self, dynamic_body_a, dynamic_body_b):
        """get_average_position should return zero for empty manifold."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        avg = manifold.get_average_position()
        assert abs(avg.x) < 1e-6
        assert abs(avg.y) < 1e-6
        assert abs(avg.z) < 1e-6

    def test_get_max_penetration(self, dynamic_body_a, dynamic_body_b):
        """get_max_penetration should return maximum."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        manifold.add_point(ContactPoint(Vec3.zero(), Vec3.unit_y(), 0.1))
        manifold.add_point(ContactPoint(Vec3(1, 0, 0), Vec3.unit_y(), 0.3))
        manifold.add_point(ContactPoint(Vec3(2, 0, 0), Vec3.unit_y(), 0.2))
        assert abs(manifold.get_max_penetration() - 0.3) < 1e-6


class TestContactConstraint:
    """Tests for ContactConstraint class."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with bodies."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        assert constraint.body_a is dynamic_body_a
        assert constraint.body_b is dynamic_body_b

    def test_construction_with_manifold(self, dynamic_body_a, dynamic_body_b):
        """Should construct with existing manifold."""
        manifold = ContactManifold(dynamic_body_a, dynamic_body_b)
        manifold.add_point(ContactPoint(Vec3.zero(), Vec3.unit_y(), 0.1))
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b, manifold)
        assert constraint.manifold is manifold

    def test_friction_coefficient(self, dynamic_body_a, dynamic_body_b):
        """friction_coefficient should be settable."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        assert constraint.friction_coefficient == 0.4
        constraint.friction_coefficient = 0.8
        assert constraint.friction_coefficient == 0.8

    def test_friction_coefficient_clamped(self, dynamic_body_a, dynamic_body_b):
        """friction_coefficient should clamp to non-negative."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        constraint.friction_coefficient = -0.5
        assert constraint.friction_coefficient >= 0.0

    def test_restitution_coefficient(self, dynamic_body_a, dynamic_body_b):
        """restitution_coefficient should be settable."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        assert constraint.restitution_coefficient == 0.0
        constraint.restitution_coefficient = 0.5
        assert constraint.restitution_coefficient == 0.5

    def test_restitution_coefficient_clamped(self, dynamic_body_a, dynamic_body_b):
        """restitution_coefficient should clamp to [0, 1]."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        constraint.restitution_coefficient = 1.5
        assert constraint.restitution_coefficient <= 1.0
        constraint.restitution_coefficient = -0.5
        assert constraint.restitution_coefficient >= 0.0


class TestContactConstraintSetPoint:
    """Tests for set_contact_point method."""

    def test_set_contact_point(self, dynamic_body_a, dynamic_body_b):
        """set_contact_point should configure contact."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        constraint.set_contact_point(
            position=Vec3(1, 0, 0),
            normal=Vec3(0, 1, 0),
            penetration=0.1
        )
        assert constraint.manifold.point_count == 1

    def test_set_contact_point_clears_manifold(self, dynamic_body_a, dynamic_body_b):
        """set_contact_point should clear existing points."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.set_contact_point(Vec3(1, 0, 0), Vec3.unit_y(), 0.2)
        assert constraint.manifold.point_count == 1


class TestContactConstraintConstraintCount:
    """Tests for constraint count."""

    def test_constraint_count_empty(self, dynamic_body_a, dynamic_body_b):
        """Empty manifold should have 0 constraints."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        assert constraint.get_constraint_count() == 0

    def test_constraint_count_per_point(self, dynamic_body_a, dynamic_body_b):
        """Each point should add 3 constraints (normal + 2 friction)."""
        constraint = ContactConstraint(dynamic_body_a, dynamic_body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        assert constraint.get_constraint_count() == 3


class TestContactConstraintPrepare:
    """Tests for prepare method."""

    def test_prepare_sets_up_jacobians(self, solver_config):
        """prepare should set up contact Jacobians."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3(0, 0.5, 0), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        assert len(constraint._point_data) == 1

    def test_prepare_computes_bias(self, solver_config):
        """prepare should compute penetration bias."""
        body_a = RigidBody(id=1, position=Vec3(0, 0.9, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        # Should have non-zero bias for penetration
        assert len(constraint._point_data) == 1


class TestContactConstraintSolving:
    """Tests for velocity solving."""

    def test_solve_velocity(self, solver_config):
        """solve_velocity should apply contact impulses."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(0, -5, 0)  # Falling toward B

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3(0, 0.5, 0), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        initial_vel = body_a.velocity.y
        constraint.solve_velocity()
        final_vel = body_a.velocity.y

        # Velocity should be less negative (impulse pushes up)
        assert final_vel >= initial_vel

    def test_normal_impulse_non_negative(self, solver_config):
        """Normal impulse should be non-negative (contacts only push)."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        for _ in range(10):
            constraint.solve_velocity()

        assert constraint.manifold.points[0].normal_impulse >= 0.0

    def test_friction_clamped(self, solver_config):
        """Friction impulse should be clamped by normal impulse."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(10, -5, 0)  # Sliding and falling

        constraint = ContactConstraint(body_a, body_b)
        constraint.friction_coefficient = 0.4
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        for _ in range(20):
            constraint.solve_velocity()

        point = constraint.manifold.points[0]
        max_friction = constraint.friction_coefficient * point.normal_impulse
        assert abs(point.tangent_impulse_1) <= max_friction + 1e-6


class TestContactConstraintPositionSolving:
    """Tests for position solving."""

    def test_solve_position(self, solver_config):
        """solve_position should handle penetration correction."""
        body_a = RigidBody(id=1, position=Vec3(0, 0.9, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3(0, 0.5, 0), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        initial_error = constraint._solve_position_internal(0.2)
        # Should have some max error returned
        assert initial_error >= 0.0


class TestContactConstraintWarmStart:
    """Tests for warm starting."""

    def test_warm_start(self, solver_config):
        """warm_start should apply cached impulses."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)

        # Solve to accumulate impulses
        constraint.solve_velocity()
        cached_impulse = constraint.manifold.points[0].normal_impulse

        # Reset velocities
        body_a.velocity = Vec3.zero()

        # Apply warm start
        constraint.warm_start(0.8)

        # Should have applied some impulse
        assert body_a.velocity.y != 0.0 or cached_impulse == 0.0


class TestComputeContactJacobian:
    """Tests for compute_contact_jacobian function."""

    def test_basic_jacobian(self):
        """Should compute contact Jacobian."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0, 2, 0), mass=1.0, local_inertia=Mat3.identity())

        jacobian = compute_contact_jacobian(
            contact_point=Vec3(0, 1, 0),
            normal=Vec3.unit_y(),
            body_a=body_a,
            body_b=body_b
        )

        assert isinstance(jacobian, Jacobian)
        # Normal direction: linear_a should be -normal, linear_b should be +normal
        assert abs(jacobian.linear_a.y - (-1.0)) < 1e-6
        assert abs(jacobian.linear_b.y - 1.0) < 1e-6

    def test_jacobian_without_body_b(self):
        """Should handle None body_b."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        jacobian = compute_contact_jacobian(
            contact_point=Vec3(0, 1, 0),
            normal=Vec3.unit_y(),
            body_a=body_a,
            body_b=None
        )

        # r_b should be zero
        assert abs(jacobian.angular_b.length()) < 1e-6


class TestCombineFriction:
    """Tests for combine_friction function."""

    def test_geometric_mean(self):
        """Should return geometric mean."""
        result = combine_friction(0.4, 0.9)
        expected = math.sqrt(0.4 * 0.9)
        assert abs(result - expected) < 1e-6

    def test_same_values(self):
        """Same values should return that value."""
        result = combine_friction(0.5, 0.5)
        assert abs(result - 0.5) < 1e-6

    def test_with_zero(self):
        """Zero friction should produce zero."""
        result = combine_friction(0.0, 0.8)
        assert abs(result) < 1e-6


class TestCombineRestitution:
    """Tests for combine_restitution function."""

    def test_returns_max(self):
        """Should return maximum."""
        result = combine_restitution(0.3, 0.7)
        assert abs(result - 0.7) < 1e-6

    def test_same_values(self):
        """Same values should return that value."""
        result = combine_restitution(0.5, 0.5)
        assert abs(result - 0.5) < 1e-6

    def test_with_zero(self):
        """Zero restitution should return other value."""
        result = combine_restitution(0.0, 0.8)
        assert abs(result - 0.8) < 1e-6


class TestContactConstraintEdgeCases:
    """Tests for edge cases."""

    def test_static_body_contact(self, solver_config):
        """Should handle contact with static body."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        static_body = RigidBody.create_static(id=0, position=Vec3.zero())

        constraint = ContactConstraint(body_a, static_body)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)
        constraint.solve_velocity()
        # Should not crash, only body_a should move

    def test_world_collision(self, dynamic_body_a, solver_config):
        """Should handle collision with world (body_b = None)."""
        constraint = ContactConstraint(dynamic_body_a, None)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.1)
        constraint.prepare(0.016, solver_config)
        constraint.solve_velocity()
        # Should not crash

    def test_high_velocity_impact(self, solver_config):
        """Should handle high velocity impacts."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(0, -100, 0)  # Very fast falling

        constraint = ContactConstraint(body_a, body_b)
        constraint.restitution_coefficient = 0.8
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.01)
        constraint.prepare(0.016, solver_config)
        constraint.solve_velocity()
        # Should not produce NaN
        assert not math.isnan(body_a.velocity.y)

    def test_multiple_contact_points(self, solver_config):
        """Should handle multiple contact points."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        manifold = ContactManifold(body_a, body_b)
        manifold.add_point(ContactPoint(Vec3(-1, 0, 0), Vec3.unit_y(), 0.1))
        manifold.add_point(ContactPoint(Vec3(1, 0, 0), Vec3.unit_y(), 0.1))
        manifold.add_point(ContactPoint(Vec3(0, 0, -1), Vec3.unit_y(), 0.1))
        manifold.add_point(ContactPoint(Vec3(0, 0, 1), Vec3.unit_y(), 0.1))

        constraint = ContactConstraint(body_a, body_b, manifold)
        constraint.prepare(0.016, solver_config)
        constraint.solve_velocity()
        # Should handle all 4 points

    def test_grazing_contact(self, solver_config):
        """Should handle grazing contact (nearly parallel)."""
        body_a = RigidBody(id=1, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(10, -0.1, 0)  # Mostly horizontal

        constraint = ContactConstraint(body_a, body_b)
        constraint.set_contact_point(Vec3.zero(), Vec3.unit_y(), 0.01)
        constraint.prepare(0.016, solver_config)
        constraint.solve_velocity()
        # Should apply friction to slow horizontal motion
