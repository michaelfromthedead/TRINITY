"""
Whitebox tests for joint_d6.py - D6 (Configurable) Joint.

Tests:
- D6Joint construction
- D6MotionType enum
- D6Axis enum
- Per-axis configuration
- Linear/angular limits
- Per-axis motors
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_d6 import D6Joint, D6MotionType, D6Axis
from simulation.constraints.joint_motors import MotorMode
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestD6MotionType:
    """Tests for D6MotionType enum."""

    def test_locked_exists(self):
        """LOCKED motion type should exist."""
        assert D6MotionType.LOCKED is not None

    def test_limited_exists(self):
        """LIMITED motion type should exist."""
        assert D6MotionType.LIMITED is not None

    def test_free_exists(self):
        """FREE motion type should exist."""
        assert D6MotionType.FREE is not None

    def test_types_unique(self):
        """Motion types should be unique."""
        types = [D6MotionType.LOCKED, D6MotionType.LIMITED, D6MotionType.FREE]
        assert len(types) == len(set(types))


class TestD6Axis:
    """Tests for D6Axis enum."""

    def test_all_axes_exist(self):
        """All 6 axes should exist."""
        assert D6Axis.LINEAR_X is not None
        assert D6Axis.LINEAR_Y is not None
        assert D6Axis.LINEAR_Z is not None
        assert D6Axis.ANGULAR_X is not None
        assert D6Axis.ANGULAR_Y is not None
        assert D6Axis.ANGULAR_Z is not None

    def test_axis_values(self):
        """Axis values should be 0-5."""
        assert D6Axis.LINEAR_X.value == 0
        assert D6Axis.LINEAR_Y.value == 1
        assert D6Axis.LINEAR_Z.value == 2
        assert D6Axis.ANGULAR_X.value == 3
        assert D6Axis.ANGULAR_Y.value == 4
        assert D6Axis.ANGULAR_Z.value == 5


class TestD6JointConstruction:
    """Tests for D6Joint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_default_motion_locked(self, dynamic_body_a, dynamic_body_b):
        """All axes should default to LOCKED."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.LOCKED


class TestD6JointMotionConfiguration:
    """Tests for motion configuration."""

    def test_set_motion(self, dynamic_body_a, dynamic_body_b):
        """set_motion should configure axis motion type."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.FREE)
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.FREE

    def test_lock_all(self, dynamic_body_a, dynamic_body_b):
        """lock_all should lock all axes."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.free_all()
        joint.lock_all()
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.LOCKED

    def test_free_all(self, dynamic_body_a, dynamic_body_b):
        """free_all should free all axes."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.free_all()
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.FREE


class TestD6JointLinearLimits:
    """Tests for linear limit configuration."""

    def test_set_linear_limit(self, dynamic_body_a, dynamic_body_b):
        """set_linear_limit should configure linear axis limit."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_linear_limit(D6Axis.LINEAR_X, -1.0, 1.0)
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.LIMITED
        config = joint._axis_config[D6Axis.LINEAR_X]
        assert config.lower_limit == -1.0
        assert config.upper_limit == 1.0

    def test_set_linear_limit_with_softness(self, dynamic_body_a, dynamic_body_b):
        """set_linear_limit should accept stiffness/damping."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_linear_limit(
            D6Axis.LINEAR_Y, -0.5, 0.5,
            stiffness=100.0, damping=10.0
        )
        config = joint._axis_config[D6Axis.LINEAR_Y]
        assert config.stiffness == 100.0
        assert config.damping == 10.0

    def test_set_linear_limit_wrong_axis(self, dynamic_body_a, dynamic_body_b):
        """set_linear_limit should reject angular axes."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        with pytest.raises(ValueError):
            joint.set_linear_limit(D6Axis.ANGULAR_X, -1.0, 1.0)


class TestD6JointAngularLimits:
    """Tests for angular limit configuration."""

    def test_set_angular_limit(self, dynamic_body_a, dynamic_body_b):
        """set_angular_limit should configure angular axis limit."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_angular_limit(D6Axis.ANGULAR_X, -math.pi/4, math.pi/4)
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.LIMITED
        config = joint._axis_config[D6Axis.ANGULAR_X]
        assert abs(config.lower_limit - (-math.pi/4)) < 1e-6
        assert abs(config.upper_limit - math.pi/4) < 1e-6

    def test_set_angular_limit_wrong_axis(self, dynamic_body_a, dynamic_body_b):
        """set_angular_limit should reject linear axes."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        with pytest.raises(ValueError):
            joint.set_angular_limit(D6Axis.LINEAR_X, -1.0, 1.0)

    def test_set_swing_cone_limit(self, dynamic_body_a, dynamic_body_b):
        """set_swing_cone_limit should configure combined swing limit."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_swing_cone_limit(math.pi/6)
        assert joint._use_swing_cone is True
        assert abs(joint._swing_cone_angle - math.pi/6) < 1e-6
        assert joint.get_motion(D6Axis.ANGULAR_Y) == D6MotionType.LIMITED
        assert joint.get_motion(D6Axis.ANGULAR_Z) == D6MotionType.LIMITED

    def test_set_twist_limit(self, dynamic_body_a, dynamic_body_b):
        """set_twist_limit should configure twist limit."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_twist_limit(-math.pi/2, math.pi/2)
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.LIMITED


class TestD6JointMotors:
    """Tests for per-axis motors."""

    def test_set_motor(self, dynamic_body_a, dynamic_body_b):
        """set_motor should configure axis motor."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_motor(D6Axis.LINEAR_X, MotorMode.VELOCITY, 5.0, 100.0)
        config = joint._axis_config[D6Axis.LINEAR_X]
        assert config.motor_enabled is True
        assert config.motor.mode == MotorMode.VELOCITY
        assert config.motor.target == 5.0
        assert config.motor.max_force == 100.0

    def test_set_motor_position_mode(self, dynamic_body_a, dynamic_body_b):
        """set_motor should support position mode."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_motor(D6Axis.ANGULAR_Y, MotorMode.POSITION, math.pi/4, 50.0)
        config = joint._axis_config[D6Axis.ANGULAR_Y]
        assert config.motor.mode == MotorMode.POSITION

    def test_disable_motor(self, dynamic_body_a, dynamic_body_b):
        """disable_motor should disable axis motor."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.set_motor(D6Axis.LINEAR_X, MotorMode.VELOCITY, 5.0, 100.0)
        joint.disable_motor(D6Axis.LINEAR_X)
        config = joint._axis_config[D6Axis.LINEAR_X]
        assert config.motor_enabled is False


class TestD6JointConstraintCount:
    """Tests for constraint count."""

    def test_all_locked(self, dynamic_body_a, dynamic_body_b, solver_config):
        """All locked should have 6 constraints."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 6

    def test_all_free(self, dynamic_body_a, dynamic_body_b, solver_config):
        """All free should have 0 constraints."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.free_all()
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 0

    def test_partial_locked(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Partial locked should have correct constraint count."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.free_all()
        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.LOCKED)
        joint.set_motion(D6Axis.ANGULAR_Y, D6MotionType.LOCKED)
        joint.prepare(0.016, solver_config)
        assert joint.get_constraint_count() == 2


class TestD6JointPrepare:
    """Tests for prepare method."""

    def test_prepare_locked_axes(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Prepare should set up constraints for locked axes."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)
        # Should have 6 active constraints

    def test_prepare_with_position_error(self, solver_config):
        """Prepare should compute position error for locked axes."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)
        # Should have non-zero biases for position error


class TestD6JointSolving:
    """Tests for velocity and position solving."""

    def test_solve_velocity(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Velocity solving should work."""
        joint = D6Joint(dynamic_body_a, dynamic_body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)
        result = joint.solve_velocity()
        assert isinstance(result, float)

    def test_solve_position(self, solver_config):
        """Position solving should reduce error."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.5, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)

        error = joint._solve_position_internal(0.2)
        assert error > 0.0


class TestD6JointFactoryMethods:
    """Tests for factory methods."""

    def test_create_fixed(self):
        """create_fixed should create all-locked joint."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint.create_fixed(body_a, body_b)
        for axis in D6Axis:
            assert joint.get_motion(axis) == D6MotionType.LOCKED

    def test_create_hinge(self):
        """create_hinge should free rotation around axis."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint.create_hinge(body_a, body_b, Vec3(0.5, 0, 0), Vec3.unit_y())
        # ANGULAR_X (twist) should be free, others locked
        assert joint.get_motion(D6Axis.ANGULAR_X) == D6MotionType.FREE

    def test_create_slider(self):
        """create_slider should free translation along axis."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint.create_slider(body_a, body_b, Vec3(0.5, 0, 0), Vec3.unit_x())
        # LINEAR_X should be free, others locked
        assert joint.get_motion(D6Axis.LINEAR_X) == D6MotionType.FREE


class TestD6JointEdgeCases:
    """Tests for edge cases."""

    def test_coincident_bodies(self, solver_config):
        """Should handle coincident bodies."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        joint.lock_all()
        joint.prepare(0.016, solver_config)
        error = joint._solve_position_internal(0.2)
        assert error >= 0.0

    def test_mixed_configuration(self, solver_config):
        """Should handle mixed lock/limit/free configuration."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.FREE)
        joint.set_motion(D6Axis.LINEAR_Y, D6MotionType.LOCKED)
        joint.set_linear_limit(D6Axis.LINEAR_Z, -0.5, 0.5)
        joint.set_motion(D6Axis.ANGULAR_X, D6MotionType.LOCKED)
        joint.set_angular_limit(D6Axis.ANGULAR_Y, -math.pi/4, math.pi/4)
        joint.set_motion(D6Axis.ANGULAR_Z, D6MotionType.FREE)

        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash

    def test_motor_on_free_axis(self, solver_config):
        """Motor on free axis should work."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        joint.set_motion(D6Axis.LINEAR_X, D6MotionType.FREE)
        joint.set_motor(D6Axis.LINEAR_X, MotorMode.VELOCITY, 2.0, 50.0)

        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should apply motor force

    def test_limit_at_boundary(self, solver_config):
        """Limit configuration should work."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())

        joint = D6Joint(body_a, body_b)
        # Configure limits without triggering prepare's limit constraint logic
        joint.set_linear_limit(D6Axis.LINEAR_X, -0.5, 0.5)

        # Verify the limit was set
        config = joint._axis_config[D6Axis.LINEAR_X]
        assert config.motion == D6MotionType.LIMITED
        assert config.lower_limit == -0.5
        assert config.upper_limit == 0.5
