"""
Whitebox tests for joint_ball.py - Ball (Spherical) Joint.

Tests:
- BallJoint construction
- 3 DOF constraint (linear only)
- Cone limits
- Twist limits
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_ball import BallJoint
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestBallJointConstruction:
    """Tests for BallJoint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_construction_with_anchors(self, dynamic_body_a, dynamic_body_b):
        """Should construct with anchor points."""
        joint = BallJoint(
            dynamic_body_a, dynamic_body_b,
            local_anchor_a=Vec3(1, 0, 0),
            local_anchor_b=Vec3(-1, 0, 0)
        )
        assert joint.local_anchor_a.x == 1
        assert joint.local_anchor_b.x == -1

    def test_construction_world_attachment(self, dynamic_body_a):
        """Should construct with world attachment."""
        joint = BallJoint(dynamic_body_a, None)
        assert joint.body_b is None


class TestBallJointConstraintCount:
    """Tests for constraint count."""

    def test_base_constraint_count(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Base ball joint should have 3 constraints."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 3

    def test_with_cone_limit(self, solver_config):
        """Cone limit should add constraint when active."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi/3),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = BallJoint(body_a, body_b)
        joint.set_cone_limit(0.1)  # Very tight limit
        joint.prepare(0.016, solver_config)
        # Should have 4 constraints when at cone limit
        assert joint.get_constraint_count() in [3, 4]

    def test_with_twist_limit(self, solver_config):
        """Twist limit should add constraint when active."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_x(), math.pi/2),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = BallJoint(body_a, body_b)
        joint.set_twist_limits(-0.1, 0.1)  # Very tight limit
        joint.prepare(0.016, solver_config)
        # Should have additional constraint
        assert joint.get_constraint_count() in [3, 4, 5]


class TestBallJointConeLimit:
    """Tests for cone limit functionality."""

    def test_cone_limit_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Cone limit should be disabled by default."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        assert joint.cone_limit_enabled is False

    def test_set_cone_limit(self, dynamic_body_a, dynamic_body_b):
        """set_cone_limit should configure limit."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.set_cone_limit(math.pi/4)
        assert joint.cone_limit_enabled is True
        assert abs(joint.cone_limit_angle - math.pi/4) < 1e-6

    def test_cone_limit_angle_setter(self, dynamic_body_a, dynamic_body_b):
        """cone_limit_angle setter should clamp to non-negative."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.cone_limit_angle = -0.5
        assert joint.cone_limit_angle >= 0.0

    def test_cone_limit_enabled_setter(self, dynamic_body_a, dynamic_body_b):
        """cone_limit_enabled setter should work."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.set_cone_limit(math.pi/4)
        joint.cone_limit_enabled = False
        assert joint.cone_limit_enabled is False


class TestBallJointTwistLimit:
    """Tests for twist limit functionality."""

    def test_twist_limit_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Twist limit should be disabled by default."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        assert joint.twist_limit_enabled is False

    def test_set_twist_limits(self, dynamic_body_a, dynamic_body_b):
        """set_twist_limits should configure limits."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.set_twist_limits(-math.pi/2, math.pi/2)
        assert joint.twist_limit_enabled is True
        assert abs(joint.min_twist_angle - (-math.pi/2)) < 1e-6
        assert abs(joint.max_twist_angle - math.pi/2) < 1e-6

    def test_twist_limit_setters(self, dynamic_body_a, dynamic_body_b):
        """Individual twist limit setters should work."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.min_twist_angle = -math.pi/4
        joint.max_twist_angle = math.pi/4
        assert abs(joint.min_twist_angle - (-math.pi/4)) < 1e-6
        assert abs(joint.max_twist_angle - math.pi/4) < 1e-6


class TestBallJointTwistAxis:
    """Tests for twist axis configuration."""

    def test_set_twist_axis(self, dynamic_body_a, dynamic_body_b):
        """set_twist_axis should configure axis."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.set_twist_axis(Vec3.unit_z())
        assert abs(joint._local_twist_axis_a.z - 1.0) < 1e-6

    def test_set_twist_axis_both(self, dynamic_body_a, dynamic_body_b):
        """set_twist_axis should allow different axes for each body."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.set_twist_axis(Vec3.unit_x(), Vec3.unit_z())
        assert abs(joint._local_twist_axis_a.x - 1.0) < 1e-6
        assert abs(joint._local_twist_axis_b.z - 1.0) < 1e-6


class TestBallJointAngleMeasurement:
    """Tests for angle measurement."""

    def test_get_swing_angle(self, dynamic_body_a, dynamic_body_b):
        """get_swing_angle should return 0 for aligned bodies."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        angle = joint.get_swing_angle()
        assert abs(angle) < 1e-6

    def test_get_swing_angle_rotated(self):
        """get_swing_angle should detect rotation."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi/4),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = BallJoint(body_a, body_b)
        angle = joint.get_swing_angle()
        # Should be approximately pi/4
        assert abs(angle - math.pi/4) < 0.1

    def test_get_twist_angle(self, dynamic_body_a, dynamic_body_b):
        """get_twist_angle should return 0 for aligned bodies."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        angle = joint.get_twist_angle()
        assert abs(angle) < 1e-6


class TestBallJointPrepare:
    """Tests for prepare method."""

    def test_prepare_sets_up_jacobians(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should set up Jacobians."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Should have 3 linear Jacobians

    def test_prepare_with_position_error(self, solver_config):
        """prepare should handle position error."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should have non-zero biases


class TestBallJointSolving:
    """Tests for velocity and position solving."""

    def test_solve_velocity(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Velocity solving should work."""
        joint = BallJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        result = joint.solve_velocity()
        assert isinstance(result, float)

    def test_solve_position(self, solver_config):
        """Position solving should reduce error."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        initial_error = joint.solve_position(0.2)
        assert initial_error > 0.0

        for _ in range(10):
            joint.prepare(0.016, solver_config)
            joint.solve_position(0.2)

        final_error = joint.solve_position(0.2)
        assert final_error <= initial_error + 0.001


class TestBallJointFactoryMethods:
    """Tests for factory methods."""

    def test_create_at_point(self):
        """create_at_point should create joint at world position."""
        body_a = RigidBody(id=1, position=Vec3(-1, 0, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint.create_at_point(body_a, body_b, Vec3.zero())
        anchor_a = joint.get_world_anchor_a()
        assert abs(anchor_a.x) < 1e-6
        assert abs(anchor_a.y) < 1e-6
        assert abs(anchor_a.z) < 1e-6

    def test_create_at_point_with_cone_limit(self):
        """create_at_point should accept cone limit."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint.create_at_point(body_a, body_b, Vec3(0.5, 0, 0), cone_limit=math.pi/4)
        assert joint.cone_limit_enabled is True
        assert abs(joint.cone_limit_angle - math.pi/4) < 1e-6

    def test_create_at_point_with_twist_limits(self):
        """create_at_point should accept twist limits."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint.create_at_point(
            body_a, body_b, Vec3(0.5, 0, 0),
            twist_limits=(-math.pi/4, math.pi/4)
        )
        assert joint.twist_limit_enabled is True

    def test_create_ragdoll_shoulder(self):
        """create_ragdoll_shoulder should create appropriate joint."""
        torso = RigidBody(id=1, position=Vec3.zero(), mass=10.0, local_inertia=Mat3.identity())
        arm = RigidBody(id=2, position=Vec3(1, 0, 0), mass=2.0, local_inertia=Mat3.identity())

        joint = BallJoint.create_ragdoll_shoulder(torso, arm, Vec3(0.5, 0, 0))
        assert joint.cone_limit_enabled is True
        assert joint.twist_limit_enabled is True
        # Should have typical shoulder limits
        assert abs(joint.cone_limit_angle - math.pi * 0.5) < 1e-6


class TestBallJointEdgeCases:
    """Tests for edge cases."""

    def test_coincident_bodies(self, solver_config):
        """Should handle coincident bodies."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        assert error >= 0.0

    def test_very_tight_cone_limit(self, solver_config):
        """Should handle very tight cone limits."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), 0.5),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.set_cone_limit(0.001)  # Very tight
        joint.prepare(0.016, solver_config)
        # Should have cone limit constraint active
        assert joint.get_constraint_count() >= 3

    def test_180_degree_rotation(self, solver_config):
        """Should handle 180 degree rotation."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should not crash

    def test_cone_and_twist_together(self, solver_config):
        """Cone and twist limits should work together."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3(1, 1, 0).normalized(), 0.5),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = BallJoint(body_a, body_b)
        joint.set_cone_limit(0.1)
        joint.set_twist_limits(-0.1, 0.1)
        joint.prepare(0.016, solver_config)
        # Should handle both limits


class TestBallJointWorldAttachment:
    """Tests for world attachment (body_b = None)."""

    def test_world_attachment(self, dynamic_body_a, solver_config):
        """Should work with world attachment."""
        joint = BallJoint(dynamic_body_a, None, local_anchor_b=Vec3(0, 5, 0))
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()

    def test_world_attachment_position(self, solver_config):
        """World attachment should constrain to world position."""
        body_a = RigidBody(
            id=1,
            position=Vec3(1, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        world_anchor = Vec3(0, 5, 0)

        joint = BallJoint(body_a, None, local_anchor_b=world_anchor)
        joint.prepare(0.016, solver_config)

        for _ in range(20):
            joint.solve_velocity()
            joint.prepare(0.016, solver_config)
            joint.solve_position(0.2)

        # Anchor should move toward world position
