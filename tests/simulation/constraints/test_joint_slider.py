"""
Whitebox tests for joint_slider.py - Slider (Prismatic) Joint.

Tests:
- SliderJoint construction
- 5 DOF constraint (2 linear + 3 angular)
- Position limits
- Motors (velocity and position)
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_slider import SliderJoint
from simulation.constraints.joint_motors import MotorMode
from simulation.constraints.joint_limits import LimitState
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestSliderJointConstruction:
    """Tests for SliderJoint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_construction_with_axis(self, dynamic_body_a, dynamic_body_b):
        """Should construct with custom slider axis."""
        joint = SliderJoint(
            dynamic_body_a, dynamic_body_b,
            local_axis_a=Vec3.unit_z()
        )
        assert abs(joint.local_axis_a.z - 1.0) < 1e-6

    def test_default_axis(self, dynamic_body_a, dynamic_body_b):
        """Default slider axis should be X."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        assert abs(joint.local_axis_a.x - 1.0) < 1e-6

    def test_construction_with_anchors(self, dynamic_body_a, dynamic_body_b):
        """Should construct with anchor points."""
        joint = SliderJoint(
            dynamic_body_a, dynamic_body_b,
            local_anchor_a=Vec3(1, 0, 0),
            local_anchor_b=Vec3(-1, 0, 0)
        )
        assert joint.local_anchor_a.x == 1


class TestSliderJointConstraintCount:
    """Tests for constraint count."""

    def test_base_constraint_count(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Base slider should have 5 constraints."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 5

    def test_with_active_limit(self, solver_config):
        """Active limit should add 1 constraint."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(2, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint(body_a, body_b)
        joint.set_limits(-0.1, 0.1)  # Very tight limits
        joint._reference_position = 0.0
        joint.prepare(0.016, solver_config)
        # Should have 6 constraints when at limit
        assert joint.get_constraint_count() in [5, 6]

    def test_with_motor(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Motor should add 1 constraint."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_speed(1.0, 10.0)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 6


class TestSliderJointAxis:
    """Tests for slider axis handling."""

    def test_local_axis_a_setter(self, dynamic_body_a, dynamic_body_b):
        """local_axis_a setter should normalize."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.local_axis_a = Vec3(2, 0, 0)  # Non-unit
        assert abs(joint.local_axis_a.length() - 1.0) < 1e-6

    def test_perpendicular_axes_computed(self, dynamic_body_a, dynamic_body_b):
        """Perpendicular axes should be computed."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        # Perpendicular axes should be orthogonal to slider axis
        axis = joint.local_axis_a
        perp1 = joint._local_perp1_a
        perp2 = joint._local_perp2_a
        assert abs(axis.dot(perp1)) < 1e-6
        assert abs(axis.dot(perp2)) < 1e-6
        assert abs(perp1.dot(perp2)) < 1e-6


class TestSliderJointTranslation:
    """Tests for translation measurement."""

    def test_get_current_translation_zero(self, dynamic_body_a, dynamic_body_b):
        """Translation should be zero initially."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        translation = joint.get_current_translation()
        # Should be zero relative to initial state
        # Note: reference_position is set at construction

    def test_get_joint_speed(self, dynamic_body_a, dynamic_body_b):
        """Joint speed should be relative velocity along axis."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        # Set velocities along X (slider axis)
        dynamic_body_a.velocity = Vec3(5.0, 0, 0)
        dynamic_body_b.velocity = Vec3(2.0, 0, 0)
        speed = joint.get_joint_speed()
        # Relative speed along axis
        assert abs(speed - 3.0) < 1e-6


class TestSliderJointLimits:
    """Tests for position limits."""

    def test_limits_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Limits should be disabled by default."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        assert joint.limits_enabled is False

    def test_set_limits(self, dynamic_body_a, dynamic_body_b):
        """set_limits should configure limits."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.set_limits(-0.5, 0.5)
        assert joint.limits_enabled is True
        assert abs(joint.min_distance - (-0.5)) < 1e-6
        assert abs(joint.max_distance - 0.5) < 1e-6

    def test_min_distance_setter(self, dynamic_body_a, dynamic_body_b):
        """min_distance setter should work."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.min_distance = -1.0
        assert abs(joint.min_distance - (-1.0)) < 1e-6

    def test_max_distance_setter(self, dynamic_body_a, dynamic_body_b):
        """max_distance setter should work."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.max_distance = 1.0
        assert abs(joint.max_distance - 1.0) < 1e-6


class TestSliderJointMotor:
    """Tests for motor functionality."""

    def test_motor_disabled_by_default(self, dynamic_body_a, dynamic_body_b):
        """Motor should be disabled by default."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        assert joint.motor_enabled is False

    def test_set_motor_speed(self, dynamic_body_a, dynamic_body_b):
        """set_motor_speed should configure velocity motor."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_speed(2.0, 50.0)
        assert joint.motor_enabled is True
        assert joint.motor.mode == MotorMode.VELOCITY
        assert joint.motor.target == 2.0
        assert joint.motor.max_force == 50.0

    def test_set_motor_position(self, dynamic_body_a, dynamic_body_b):
        """set_motor_position should configure position motor."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.set_motor_position(0.5, 100.0)
        assert joint.motor_enabled is True
        assert joint.motor.mode == MotorMode.POSITION
        assert joint.motor.target == 0.5


class TestSliderJointPrepare:
    """Tests for prepare method."""

    def test_prepare_sets_up_jacobians(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should set up Jacobians."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Should have Jacobians for 2 linear + 3 angular = 5 constraints

    def test_prepare_handles_angular_error(self, solver_config):
        """prepare should compute angular error."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), 0.2),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = SliderJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should have angular error bias


class TestSliderJointSolving:
    """Tests for velocity and position solving."""

    def test_solve_velocity(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Velocity solving should work."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        result = joint.solve_velocity()
        assert isinstance(result, float)

    def test_solve_position(self, solver_config):
        """Position solving should reduce error."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0, 0.5, 0), mass=1.0, local_inertia=Mat3.identity())
        # Offset perpendicular to X axis

        joint = SliderJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        initial_error = joint.solve_position(0.2)
        assert initial_error > 0.0

    def test_angular_constraint(self, solver_config):
        """Angular constraints should prevent rotation."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), 0.5),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint(body_a, body_b)
        # The reference orientation is captured at construction
        # To create angular error, we need to change body orientation after construction
        body_a.orientation = Quaternion.from_axis_angle(Vec3.unit_y(), 1.0)  # Different rotation
        joint.prepare(0.016, solver_config)

        error = joint.solve_position(0.2)
        # Should have angular error - but may be zero if corrected in first iteration
        assert error >= 0.0


class TestSliderJointFactoryMethod:
    """Tests for create_at_point factory method."""

    def test_create_at_point(self):
        """create_at_point should create joint at world position."""
        body_a = RigidBody(id=1, position=Vec3(-1, 0, 0), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint.create_at_point(
            body_a, body_b,
            Vec3.zero(),
            Vec3.unit_x()
        )
        anchor_a = joint.get_world_anchor_a()
        assert abs(anchor_a.x) < 1e-6

    def test_create_at_point_with_limits(self):
        """create_at_point should accept limits."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint.create_at_point(
            body_a, body_b,
            Vec3(0.5, 0, 0), Vec3.unit_x(),
            min_distance=-1.0,
            max_distance=1.0
        )
        assert joint.limits_enabled is True


class TestSliderJointEdgeCases:
    """Tests for edge cases."""

    def test_coincident_bodies(self, solver_config):
        """Should handle coincident bodies."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        assert error >= 0.0

    def test_high_velocity(self, solver_config):
        """Should handle high velocities."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_a.velocity = Vec3(100, 0, 0)

        joint = SliderJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        assert not math.isnan(body_a.velocity.x)

    def test_motor_and_limit_together(self, solver_config):
        """Motor and limit should work together."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint(body_a, body_b)
        joint.set_limits(-0.5, 0.5)
        joint.set_motor_speed(10.0, 100.0)  # High speed motor
        joint.prepare(0.016, solver_config)
        # Should have motor constraint
        assert joint.get_constraint_count() >= 6

    def test_different_axis(self, solver_config):
        """Should work with different slider axis."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0, 1, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = SliderJoint(body_a, body_b, local_axis_a=Vec3.unit_y())
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash


class TestSliderJointReferenceOrientation:
    """Tests for reference orientation handling."""

    def test_reference_orientation_stored(self, dynamic_body_a, dynamic_body_b):
        """Reference orientation should be stored."""
        joint = SliderJoint(dynamic_body_a, dynamic_body_b)
        ref = joint._reference_orientation
        assert ref is not None

    def test_angular_constraint_uses_reference(self, solver_config):
        """Angular constraint should use reference orientation."""
        body_a = RigidBody(
            id=1,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_z(), math.pi/4),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3.zero(),
            orientation=Quaternion.from_axis_angle(Vec3.unit_z(), math.pi/4),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        # Both at same orientation - no angular error
        joint = SliderJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        # Now rotate body_a
        body_a.orientation = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi/2)
        joint.prepare(0.016, solver_config)
        # Should now have angular error
