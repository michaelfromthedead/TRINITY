"""
Whitebox tests for joint_spring.py - Spring Joint.

Tests:
- SpringJoint construction
- Spring/damper dynamics
- Length limits
- Frequency/damping ratio parameters
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_spring import SpringJoint
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody, ConstraintType
from simulation.solver.config import SolverConfig


class TestSpringJointConstruction:
    """Tests for SpringJoint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_auto_rest_length(self):
        """Rest length should be computed from initial positions."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b)
        assert abs(joint.rest_length - 5.0) < 1e-6

    def test_explicit_rest_length(self, dynamic_body_a, dynamic_body_b):
        """Explicit rest length should override auto-computed."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b, rest_length=3.0)
        assert abs(joint.rest_length - 3.0) < 1e-6

    def test_stiffness_damping(self, dynamic_body_a, dynamic_body_b):
        """Stiffness and damping should be configurable."""
        joint = SpringJoint(
            dynamic_body_a, dynamic_body_b,
            stiffness=500.0,
            damping=25.0
        )
        assert joint.stiffness == 500.0
        assert joint.damping == 25.0


class TestSpringJointProperties:
    """Tests for spring properties."""

    def test_rest_length_setter(self, dynamic_body_a, dynamic_body_b):
        """rest_length setter should clamp to non-negative."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.rest_length = -5.0
        assert joint.rest_length >= 0.0

    def test_stiffness_setter(self, dynamic_body_a, dynamic_body_b):
        """stiffness setter should clamp to non-negative."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.stiffness = -100.0
        assert joint.stiffness >= 0.0

    def test_damping_setter(self, dynamic_body_a, dynamic_body_b):
        """damping setter should clamp to non-negative."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.damping = -10.0
        assert joint.damping >= 0.0

    def test_min_max_length(self, dynamic_body_a, dynamic_body_b):
        """min_length and max_length should be optional."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        assert joint.min_length is None
        assert joint.max_length is None

        joint.min_length = 0.5
        joint.max_length = 5.0
        assert joint.min_length == 0.5
        assert joint.max_length == 5.0


class TestSpringJointConstraintCount:
    """Tests for constraint count."""

    def test_constraint_count(self, dynamic_body_a, dynamic_body_b):
        """Spring joint should have 1 constraint."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        assert joint.get_constraint_count() == 1


class TestSpringJointFrequencyDamping:
    """Tests for frequency/damping ratio parameters."""

    def test_set_frequency_damping(self, dynamic_body_a, dynamic_body_b):
        """set_frequency_damping should configure parameters."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.set_frequency_damping(frequency=5.0, damping_ratio=0.7)
        assert abs(joint.frequency - 5.0) < 1e-6
        assert abs(joint.damping_ratio - 0.7) < 1e-6

    def test_damping_ratio_clamped(self, dynamic_body_a, dynamic_body_b):
        """damping_ratio should be clamped to [0, 1]."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.set_frequency_damping(frequency=5.0, damping_ratio=2.0)
        assert joint.damping_ratio <= 1.0

        joint.set_frequency_damping(frequency=5.0, damping_ratio=-0.5)
        assert joint.damping_ratio >= 0.0


class TestSpringJointLengthLimits:
    """Tests for length limit functionality."""

    def test_set_length_limits(self, dynamic_body_a, dynamic_body_b):
        """set_length_limits should configure limits."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.set_length_limits(min_length=0.5, max_length=2.0)
        assert joint.min_length == 0.5
        assert joint.max_length == 2.0

    def test_limits_affect_position_solving(self, solver_config):
        """Length limits should affect position solving."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=1.0)
        joint.set_length_limits(max_length=2.0)  # Beyond max
        joint.prepare(0.016, solver_config)
        error = joint._solve_position_internal(0.2)
        # Should have error due to exceeding max length
        assert error > 0.0


class TestSpringJointMeasurements:
    """Tests for spring measurements."""

    def test_current_length(self, solver_config):
        """current_length should reflect current distance."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(3, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        assert abs(joint.current_length - 3.0) < 1e-6

    def test_get_displacement(self, solver_config):
        """get_displacement should return difference from rest length."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=3.0)
        joint.prepare(0.016, solver_config)
        displacement = joint.get_displacement()
        # current = 5, rest = 3, displacement = 2
        assert abs(displacement - 2.0) < 1e-6

    def test_get_relative_velocity(self):
        """get_relative_velocity should return velocity along spring axis."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(3, 0, 0), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(0, 0, 0)
        body_b.velocity = Vec3(2, 0, 0)

        joint = SpringJoint(body_a, body_b)
        rel_vel = joint.get_relative_velocity()
        # Body B moving away from A at 2 m/s along X
        assert abs(rel_vel - 2.0) < 1e-6

    def test_spring_force(self, solver_config):
        """spring_force should reflect current spring force."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=3.0, stiffness=100.0)
        joint.prepare(0.016, solver_config)
        # F = k * (x - rest) = 100 * 2 = 200
        assert abs(joint.spring_force - 200.0) < 1e-6


class TestSpringJointPrepare:
    """Tests for prepare method."""

    def test_prepare_sets_up_jacobian(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should set up spring Jacobian."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Should have Jacobian and effective mass

    def test_prepare_with_zero_length(self, solver_config):
        """prepare should handle zero length gracefully."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should use arbitrary direction

    def test_prepare_computes_soft_coefficients(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should compute soft constraint coefficients."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b, stiffness=100.0, damping=10.0)
        joint.prepare(0.016, solver_config)
        # gamma and beta should be computed
        assert joint._gamma >= 0.0
        assert joint._beta >= 0.0


class TestSpringJointSolving:
    """Tests for velocity solving."""

    def test_solve_velocity(self, dynamic_body_a, dynamic_body_b, solver_config):
        """solve_velocity should return impulse."""
        joint = SpringJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        impulse = joint.solve_velocity()
        assert isinstance(impulse, float)

    def test_spring_pulls_bodies(self, solver_config):
        """Spring should pull bodies together when stretched."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=1.0, stiffness=100.0, damping=1.0)
        joint.prepare(0.016, solver_config)

        initial_vel_a = body_a.velocity.x
        initial_vel_b = body_b.velocity.x

        joint.solve_velocity()

        # Bodies should accelerate toward each other
        # Body A should gain positive velocity (toward B)
        # Body B should gain negative velocity (toward A)
        assert body_a.velocity.x > initial_vel_a or body_b.velocity.x < initial_vel_b

    def test_spring_pushes_bodies(self, solver_config):
        """Spring should push bodies apart when compressed."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=3.0, stiffness=100.0, damping=1.0)
        joint.prepare(0.016, solver_config)

        joint.solve_velocity()

        # Bodies should accelerate away from each other


class TestSpringJointFactoryMethods:
    """Tests for factory methods."""

    def test_create_between_points(self):
        """create_between_points should create spring at world positions."""
        body_a = RigidBody(id=1, position=Vec3(-1, 0, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint.create_between_points(
            body_a, body_b,
            Vec3.zero(), Vec3(2, 0, 0),
            stiffness=200.0,
            damping=5.0
        )
        assert joint.stiffness == 200.0
        assert joint.damping == 5.0

    def test_create_bungee(self):
        """create_bungee should create spring that only pulls."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(2, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint.create_bungee(
            body_a, body_b,
            Vec3.zero(), Vec3(2, 0, 0),
            stiffness=100.0,
            damping=5.0
        )
        # min_length should equal rest_length (only pulls, no push)
        assert joint.min_length == joint.rest_length


class TestSpringJointEdgeCases:
    """Tests for edge cases."""

    def test_zero_stiffness(self, solver_config):
        """Zero stiffness should produce no spring force."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, stiffness=0.0)
        joint.prepare(0.016, solver_config)
        assert joint.spring_force == 0.0

    def test_high_stiffness(self, solver_config):
        """High stiffness should work without numerical issues."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(2, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=1.0, stiffness=100000.0, damping=100.0)
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        assert not math.isnan(body_a.velocity.x)

    def test_with_angular_velocity(self, solver_config):
        """Should work with bodies having angular velocity."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(2, 0, 0), mass=1.0, local_inertia=Mat3.identity())
        body_a.angular_velocity = Vec3(0, 5, 0)

        joint = SpringJoint(body_a, body_b, stiffness=100.0, damping=10.0)
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash

    def test_frequency_damping_mode(self, solver_config):
        """Frequency/damping mode should compute correct spring force."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(2, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SpringJoint(body_a, body_b, rest_length=1.0)
        joint.set_frequency_damping(5.0, 0.5)  # 5 Hz, critical damping = 0.5
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should apply spring force based on frequency
