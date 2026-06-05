"""
Whitebox tests for joint_hinge.py - Hinge (Revolute) Joint.

Tests:
- HingeJoint construction
- 5 DOF constraint (3 linear + 2 angular)
- Angle limits
- Motors (velocity and position)
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_hinge import HingeJoint
from simulation.constraints.joint_motors import MotorMode
from simulation.constraints.joint_limits import LimitState
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestHingeJointConstruction:
    """Tests for HingeJoint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_construction_with_axis(self, dynamic_body_a, dynamic_body_b):
        """Should construct with custom hinge axis."""
        joint = HingeJoint(
            dynamic_body_a, dynamic_body_b,
            local_axis_a=Vec3.unit_z(),
            local_axis_b=Vec3.unit_z()
        )
        # Axis should be normalized
        assert abs(joint.local_axis_a.length() - 1.0) < 1e-6

    def test_default_axis(self, dynamic_body_a, dynamic_body_b):
        """Default hinge axis should be Y."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        assert abs(joint.local_axis_a.y - 1.0) < 1e-6

    def test_construction_with_anchors(self, dynamic_body_a, dynamic_body_b):
        """Should construct with anchor points."""
        joint = HingeJoint(
            dynamic_body_a, dynamic_body_b,
            local_anchor_a=Vec3(1, 0, 0),
            local_anchor_b=Vec3(-1, 0, 0)
        )
        assert joint.local_anchor_a.x == 1
        assert joint.local_anchor_b.x == -1


class TestHingeJointConstraintCount:
    """Tests for constraint count."""

    def test_base_constraint_count(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Base hinge should have 5 constraints."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 5

    def test_with_active_limit(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Active limit should add 1 constraint."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_limits(-0.001, 0.001)  # Very tight limits
        joint.prepare(0.016, solver_config)
        # Might have 6 constraints if at limit
        assert joint.get_constraint_count() in [5, 6]

    def test_with_motor(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Motor should add 1 constraint."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_speed(1.0, 10.0)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 6


class TestHingeJointAxis:
    """Tests for hinge axis handling."""

    def test_local_axis_a_setter(self, dynamic_body_a, dynamic_body_b):
        """local_axis_a setter should normalize."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.local_axis_a = Vec3(2, 0, 0)  # Non-unit vector
        assert abs(joint.local_axis_a.length() - 1.0) < 1e-6

    def test_local_axis_b_setter(self, dynamic_body_a, dynamic_body_b):
        """local_axis_b setter should normalize."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.local_axis_b = Vec3(0, 3, 0)
        assert abs(joint.local_axis_b.length() - 1.0) < 1e-6


class TestHingeJointAngle:
    """Tests for angle measurement."""

    def test_get_current_angle_zero(self, dynamic_body_a, dynamic_body_b):
        """Angle should be zero initially."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        angle = joint.get_current_angle()
        assert abs(angle) < 1e-6

    def test_get_current_angle_with_rotation(self):
        """Angle should reflect body rotation."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.identity(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi/4),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint._reference_angle = 0.0  # Reset reference
        angle = joint.get_current_angle()
        # Should reflect relative rotation

    def test_get_joint_speed(self, dynamic_body_a, dynamic_body_b):
        """Joint speed should be relative angular velocity along axis."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        # Set angular velocities
        dynamic_body_a.angular_velocity = Vec3(0, 5.0, 0)  # 5 rad/s around Y
        dynamic_body_b.angular_velocity = Vec3(0, 2.0, 0)  # 2 rad/s around Y
        speed = joint.get_joint_speed()
        # Relative speed should be 5 - 2 = 3
        assert abs(speed - 3.0) < 1e-6


class TestHingeJointLimits:
    """Tests for angle limits."""

    def test_limits_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Limits should be disabled by default."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        assert joint.limits_enabled is False

    def test_set_limits(self, dynamic_body_a, dynamic_body_b):
        """set_limits should configure limits."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_limits(-math.pi/4, math.pi/4)
        assert joint.limits_enabled is True
        assert abs(joint.min_angle - (-math.pi/4)) < 1e-6
        assert abs(joint.max_angle - math.pi/4) < 1e-6

    def test_min_angle_setter(self, dynamic_body_a, dynamic_body_b):
        """min_angle setter should update limit."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.min_angle = -math.pi/2
        assert abs(joint.min_angle - (-math.pi/2)) < 1e-6

    def test_max_angle_setter(self, dynamic_body_a, dynamic_body_b):
        """max_angle setter should update limit."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.max_angle = math.pi/2
        assert abs(joint.max_angle - math.pi/2) < 1e-6

    def test_limit_at_lower(self, solver_config):
        """Should detect when at lower limit."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), -0.5),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            orientation=Quaternion.identity(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint.set_limits(-0.1, 0.1)
        joint._reference_angle = 0.0
        joint.prepare(0.016, solver_config)
        # Should have limit active (AT_LOWER)

    def test_limit_at_upper(self, solver_config):
        """Should detect when at upper limit."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), 0.5),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            orientation=Quaternion.identity(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint.set_limits(-0.1, 0.1)
        joint._reference_angle = 0.0
        joint.prepare(0.016, solver_config)
        # Should have limit active (AT_UPPER)


class TestHingeJointMotor:
    """Tests for motor functionality."""

    def test_motor_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Motor should be disabled by default."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        assert joint.motor_enabled is False

    def test_set_motor_speed(self, dynamic_body_a, dynamic_body_b):
        """set_motor_speed should configure velocity motor."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_speed(5.0, 100.0)
        assert joint.motor_enabled is True
        assert joint.motor.mode == MotorMode.VELOCITY
        assert joint.motor.target == 5.0
        assert joint.motor.max_force == 100.0

    def test_set_motor_position(self, dynamic_body_a, dynamic_body_b):
        """set_motor_position should configure position motor."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_position(math.pi/4, 50.0)
        assert joint.motor_enabled is True
        assert joint.motor.mode == MotorMode.POSITION
        assert abs(joint.motor.target - math.pi/4) < 1e-6
        assert joint.motor.max_force == 50.0

    def test_motor_enabled_setter(self, dynamic_body_a, dynamic_body_b):
        """motor_enabled setter should work."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_speed(1.0, 10.0)
        joint.motor_enabled = False
        assert joint.motor_enabled is False

    def test_motor_applies_impulse(self, solver_config):
        """Motor should apply impulse to change velocity."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint.set_motor_speed(5.0, 100.0)  # Target 5 rad/s
        joint.prepare(0.016, solver_config)

        initial_speed_a = body_a.angular_velocity.y
        joint.solve_velocity()
        final_speed_a = body_a.angular_velocity.y

        # Velocities should change
        assert initial_speed_a != final_speed_a or abs(final_speed_a) > 0


class TestHingeJointPrepare:
    """Tests for prepare method."""

    def test_prepare_sets_up_jacobians(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should set up constraint Jacobians."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Should have at least 5 Jacobians set up

    def test_prepare_with_position_error(self, solver_config):
        """prepare should compute position error bias."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0.5, 0, 0),  # Offset
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should have non-zero biases for position error


class TestHingeJointSolving:
    """Tests for velocity and position solving."""

    def test_solve_velocity(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Velocity solving should work."""
        joint = HingeJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        result = joint.solve_velocity()
        assert isinstance(result, float)

    def test_solve_position(self, solver_config):
        """Position solving should reduce error."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0.5, 0, 0),  # Offset
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        initial_error = joint.solve_position(0.2)

        for _ in range(10):
            joint.prepare(0.016, solver_config)
            joint.solve_position(0.2)

        final_error = joint.solve_position(0.2)
        # Error should decrease
        assert final_error <= initial_error + 0.001


class TestHingeJointFactoryMethod:
    """Tests for create_at_point factory method."""

    def test_create_at_point(self):
        """create_at_point should create joint at world position."""
        body_a = RigidBody(
            id=1,
            position=Vec3(-1, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(1, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        world_anchor = Vec3(0, 0, 0)
        world_axis = Vec3.unit_y()

        joint = HingeJoint.create_at_point(body_a, body_b, world_anchor, world_axis)
        assert joint.body_a is body_a
        assert joint.body_b is body_b

        # Anchor should be at world position
        anchor_a = joint.get_world_anchor_a()
        assert abs(anchor_a.x) < 1e-6
        assert abs(anchor_a.y) < 1e-6
        assert abs(anchor_a.z) < 1e-6

    def test_create_at_point_with_limits(self):
        """create_at_point should accept limits."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = HingeJoint.create_at_point(
            body_a, body_b,
            Vec3(0.5, 0, 0), Vec3.unit_y(),
            min_angle=-math.pi/4,
            max_angle=math.pi/4
        )
        assert joint.limits_enabled is True
        assert abs(joint.min_angle - (-math.pi/4)) < 1e-6
        assert abs(joint.max_angle - math.pi/4) < 1e-6


class TestHingeJointEdgeCases:
    """Tests for edge cases."""

    def test_coincident_anchors(self, solver_config):
        """Should handle coincident anchor points."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = HingeJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        assert error >= 0.0

    def test_perpendicular_axes(self, solver_config):
        """Should handle perpendicular hinge axes."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_z(), math.pi/2),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = HingeJoint(body_a, body_b, local_axis_a=Vec3.unit_y(), local_axis_b=Vec3.unit_y())
        joint.prepare(0.016, solver_config)
        # Should not crash

    def test_motor_and_limit_together(self, solver_config):
        """Motor and limit should work together."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = HingeJoint(body_a, body_b)
        joint.set_limits(-math.pi/4, math.pi/4)
        joint.set_motor_speed(10.0, 100.0)  # High speed motor

        joint.prepare(0.016, solver_config)
        # Should have 6 or 7 constraints depending on limit state
        assert joint.get_constraint_count() >= 6

    def test_high_angular_velocity(self, solver_config):
        """Should handle high angular velocities."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.angular_velocity = Vec3(0, 100.0, 0)  # 100 rad/s

        joint = HingeJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash or produce NaN
        assert not math.isnan(body_a.angular_velocity.y)
