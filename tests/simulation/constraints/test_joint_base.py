"""
Whitebox tests for joint_base.py - Joint base class.

Tests:
- JointState enum
- JointBreakEvent dataclass
- Joint abstract base class
- Property accessors
- Break detection
- Warm starting
- Effective mass computation
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_base import Joint, JointState, JointBreakEvent
from simulation.constraints.joint_fixed import FixedJoint
from simulation.solver.jacobian import Vec3, Mat3, Quaternion, Jacobian
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestJointState:
    """Tests for JointState enum."""

    def test_active_state_exists(self):
        """ACTIVE state should exist."""
        assert JointState.ACTIVE is not None

    def test_disabled_state_exists(self):
        """DISABLED state should exist."""
        assert JointState.DISABLED is not None

    def test_broken_state_exists(self):
        """BROKEN state should exist."""
        assert JointState.BROKEN is not None

    def test_states_are_unique(self):
        """All states should be unique."""
        states = [JointState.ACTIVE, JointState.DISABLED, JointState.BROKEN]
        assert len(states) == len(set(states))


class TestJointBreakEvent:
    """Tests for JointBreakEvent dataclass."""

    def test_break_event_creation(self, dynamic_body_a, dynamic_body_b):
        """Should create break event with all fields."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        event = JointBreakEvent(
            joint=joint,
            break_force=100.0,
            break_torque=50.0,
            applied_force=150.0,
            applied_torque=60.0,
            timestamp=1.5
        )
        assert event.joint is joint
        assert event.break_force == 100.0
        assert event.break_torque == 50.0
        assert event.applied_force == 150.0
        assert event.applied_torque == 60.0
        assert event.timestamp == 1.5

    def test_break_event_default_timestamp(self, dynamic_body_a, dynamic_body_b):
        """Timestamp should default to 0.0."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        event = JointBreakEvent(
            joint=joint,
            break_force=100.0,
            break_torque=50.0,
            applied_force=150.0,
            applied_torque=60.0
        )
        assert event.timestamp == 0.0


class TestJointProperties:
    """Tests for Joint property accessors."""

    def test_body_a_property(self, dynamic_body_a, dynamic_body_b):
        """body_a property should return correct body."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a

    def test_body_b_property(self, dynamic_body_a, dynamic_body_b):
        """body_b property should return correct body."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_b is dynamic_body_b

    def test_body_b_can_be_none(self, dynamic_body_a):
        """body_b can be None for world attachment."""
        joint = FixedJoint(dynamic_body_a, None)
        assert joint.body_b is None

    def test_local_anchor_a_default(self, dynamic_body_a, dynamic_body_b):
        """local_anchor_a should default to zero."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.local_anchor_a.x == 0
        assert joint.local_anchor_a.y == 0
        assert joint.local_anchor_a.z == 0

    def test_local_anchor_a_setter(self, dynamic_body_a, dynamic_body_b):
        """local_anchor_a should be settable."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.local_anchor_a = Vec3(1, 2, 3)
        assert joint.local_anchor_a.x == 1
        assert joint.local_anchor_a.y == 2
        assert joint.local_anchor_a.z == 3

    def test_local_anchor_b_setter(self, dynamic_body_a, dynamic_body_b):
        """local_anchor_b should be settable."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.local_anchor_b = Vec3(4, 5, 6)
        assert joint.local_anchor_b.x == 4
        assert joint.local_anchor_b.y == 5
        assert joint.local_anchor_b.z == 6

    def test_break_force_default(self, dynamic_body_a, dynamic_body_b):
        """break_force should default to 0 (unbreakable)."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.break_force == 0.0

    def test_break_force_setter(self, dynamic_body_a, dynamic_body_b):
        """break_force should be settable."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=100.0)
        assert joint.break_force == 100.0
        joint.break_force = 200.0
        assert joint.break_force == 200.0

    def test_break_force_clamps_negative(self, dynamic_body_a, dynamic_body_b):
        """break_force should clamp negative values to 0."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.break_force = -100.0
        assert joint.break_force == 0.0

    def test_break_torque_setter(self, dynamic_body_a, dynamic_body_b):
        """break_torque should be settable."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_torque=50.0)
        assert joint.break_torque == 50.0

    def test_is_broken_initially_false(self, dynamic_body_a, dynamic_body_b):
        """is_broken should initially be False."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.is_broken is False

    def test_state_initially_active(self, dynamic_body_a, dynamic_body_b):
        """state should initially be ACTIVE."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.state == JointState.ACTIVE


class TestJointEnableDisable:
    """Tests for joint enable/disable functionality."""

    def test_enable(self, dynamic_body_a, dynamic_body_b):
        """enable() should set state to ACTIVE."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.disable()
        joint.enable()
        assert joint.state == JointState.ACTIVE

    def test_disable(self, dynamic_body_a, dynamic_body_b):
        """disable() should set state to DISABLED."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.disable()
        assert joint.state == JointState.DISABLED

    def test_is_enabled_true(self, dynamic_body_a, dynamic_body_b):
        """is_enabled() should return True when ACTIVE."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.is_enabled() is True

    def test_is_enabled_false(self, dynamic_body_a, dynamic_body_b):
        """is_enabled() should return False when DISABLED."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.disable()
        assert joint.is_enabled() is False

    def test_enable_broken_joint(self, dynamic_body_a, dynamic_body_b):
        """enable() should not enable a broken joint."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=1.0)
        joint._is_broken = True
        joint._state = JointState.BROKEN
        joint.enable()
        assert joint.state == JointState.BROKEN


class TestWorldAnchors:
    """Tests for world anchor calculations."""

    def test_get_world_anchor_a_at_origin(self, dynamic_body_a, dynamic_body_b):
        """World anchor A should equal body position when local anchor is zero."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        anchor = joint.get_world_anchor_a()
        assert abs(anchor.x - dynamic_body_a.position.x) < 1e-6
        assert abs(anchor.y - dynamic_body_a.position.y) < 1e-6
        assert abs(anchor.z - dynamic_body_a.position.z) < 1e-6

    def test_get_world_anchor_a_with_offset(self, dynamic_body_a, dynamic_body_b):
        """World anchor A should include local anchor offset."""
        joint = FixedJoint(
            dynamic_body_a, dynamic_body_b,
            local_anchor_a=Vec3(1, 0, 0)
        )
        anchor = joint.get_world_anchor_a()
        assert abs(anchor.x - 1.0) < 1e-6

    def test_get_world_anchor_b_when_none(self, dynamic_body_a):
        """World anchor B should be local_anchor_b when body_b is None."""
        joint = FixedJoint(
            dynamic_body_a, None,
            local_anchor_b=Vec3(5, 5, 5)
        )
        anchor = joint.get_world_anchor_b()
        assert abs(anchor.x - 5.0) < 1e-6
        assert abs(anchor.y - 5.0) < 1e-6
        assert abs(anchor.z - 5.0) < 1e-6

    def test_get_world_anchor_with_rotation(self, dynamic_body_a, rotated_body):
        """World anchor should account for body rotation."""
        joint = FixedJoint(
            rotated_body, dynamic_body_a,
            local_anchor_a=Vec3(1, 0, 0)
        )
        anchor = joint.get_world_anchor_a()
        # Rotated 45 deg around Y: (1,0,0) -> (cos45, 0, -sin45)
        assert abs(anchor.x - math.cos(math.pi/4)) < 1e-6
        assert abs(anchor.z + math.sin(math.pi/4)) < 1e-6


class TestBreakDetection:
    """Tests for joint break detection."""

    def test_check_break_condition_no_break(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Joint should not break when forces are below threshold."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=1000.0)
        joint.prepare(0.016, solver_config)
        result = joint.check_break_condition(60.0)  # inv_dt
        assert result is False
        assert joint.is_broken is False

    def test_check_break_condition_force_break(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Joint should break when force exceeds threshold."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=0.001)
        joint.prepare(0.016, solver_config)
        # Simulate high force
        joint._last_applied_force = 100.0
        result = joint.check_break_condition(1.0)
        assert result is True
        assert joint.is_broken is True
        assert joint.state == JointState.BROKEN

    def test_check_break_condition_torque_break(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Joint should break when torque exceeds threshold."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_torque=0.001)
        joint.prepare(0.016, solver_config)
        joint._last_applied_torque = 100.0
        result = joint.check_break_condition(1.0)
        assert result is True
        assert joint.is_broken is True

    def test_break_callback(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Break callback should be called when joint breaks."""
        callback_data = []

        def on_break(event):
            callback_data.append(event)

        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=0.001)
        joint.set_break_callback(on_break)
        joint.prepare(0.016, solver_config)
        joint._last_applied_force = 100.0
        joint.check_break_condition(1.0)

        assert len(callback_data) == 1
        assert isinstance(callback_data[0], JointBreakEvent)

    def test_already_broken_returns_false(self, dynamic_body_a, dynamic_body_b):
        """check_break_condition should return False if already broken."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b, break_force=0.001)
        joint._is_broken = True
        result = joint.check_break_condition(60.0)
        assert result is False


class TestReactionForces:
    """Tests for reaction force/torque calculation."""

    def test_get_reaction_force_empty(self, dynamic_body_a, dynamic_body_b):
        """Reaction force should be zero before solving."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        force = joint.get_reaction_force(60.0)
        assert abs(force.x) < 1e-10
        assert abs(force.y) < 1e-10
        assert abs(force.z) < 1e-10

    def test_get_reaction_torque_empty(self, dynamic_body_a, dynamic_body_b):
        """Reaction torque should be zero before solving."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        torque = joint.get_reaction_torque(60.0)
        assert abs(torque.x) < 1e-10
        assert abs(torque.y) < 1e-10
        assert abs(torque.z) < 1e-10


class TestReset:
    """Tests for joint reset functionality."""

    def test_reset_clears_broken(self, dynamic_body_a, dynamic_body_b):
        """reset() should clear broken state."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint._is_broken = True
        joint._state = JointState.BROKEN
        joint.reset()
        assert joint.is_broken is False
        assert joint.state == JointState.ACTIVE

    def test_reset_clears_impulses(self, dynamic_body_a, dynamic_body_b, solver_config):
        """reset() should clear accumulated impulses."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        joint._accumulated_impulse = [1.0, 2.0, 3.0]
        joint.reset()
        assert all(i == 0.0 for i in joint._accumulated_impulse)


class TestWarmStart:
    """Tests for warm start functionality."""

    def test_store_impulses_for_warm_start(self, dynamic_body_a, dynamic_body_b, solver_config):
        """store_impulses_for_warm_start should copy accumulated impulses."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        joint._accumulated_impulse = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        joint.store_impulses_for_warm_start()
        assert joint._warm_start_impulse == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_warm_start_empty(self, dynamic_body_a, dynamic_body_b):
        """warm_start should handle empty impulses."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.warm_start(0.8)  # Should not raise

    def test_get_cached_impulse(self, dynamic_body_a, dynamic_body_b, solver_config):
        """get_cached_impulse should return sum of absolute impulses."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        joint._accumulated_impulse = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
        total = joint.get_cached_impulse()
        assert abs(total - 21.0) < 1e-6


class TestEffectiveMass:
    """Tests for effective mass computation."""

    def test_compute_effective_mass_static_body(self, static_body, dynamic_body_a, solver_config):
        """Effective mass should handle static bodies."""
        joint = FixedJoint(static_body, dynamic_body_a)
        joint.prepare(0.016, solver_config)
        # Should not crash and should compute valid effective mass

    def test_compute_effective_mass_two_dynamic(self, dynamic_body_a, dynamic_body_b, solver_config):
        """Effective mass should be positive for two dynamic bodies."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # All effective masses should be positive
        for em in joint._effective_masses:
            assert em >= 0.0


class TestSolveVelocity:
    """Tests for velocity solving."""

    def test_solve_velocity_disabled(self, dynamic_body_a, dynamic_body_b, solver_config):
        """solve_velocity should return 0 when disabled."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        joint.disable()
        result = joint.solve_velocity()
        assert result == 0.0

    def test_solve_velocity_active(self, dynamic_body_a, dynamic_body_b, solver_config):
        """solve_velocity should apply impulses when active."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        result = joint.solve_velocity()
        # Result is max impulse magnitude
        assert isinstance(result, float)


class TestSolvePosition:
    """Tests for position solving."""

    def test_solve_position_disabled(self, dynamic_body_a, dynamic_body_b, solver_config):
        """solve_position should return 0 when disabled."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        joint.disable()
        result = joint.solve_position(0.2)
        assert result == 0.0

    def test_solve_position_returns_error(self, dynamic_body_a, dynamic_body_b, solver_config):
        """solve_position should return position error."""
        # Offset bodies so there's a position error
        dynamic_body_b.position = Vec3(2, 0, 0)
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        assert error > 0.0
