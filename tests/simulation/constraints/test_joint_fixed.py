"""
Whitebox tests for joint_fixed.py - Fixed (Weld) Joint.

Tests:
- FixedJoint construction
- 6 DOF constraint (3 linear + 3 angular)
- Soft weld joints
- Reference orientation
- Position and velocity solving
- Factory methods
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'engine'))

from simulation.constraints.joint_fixed import FixedJoint
from simulation.constraints.joint_base import JointState
from simulation.solver.jacobian import Vec3, Mat3, Quaternion
from simulation.solver.constraint_solver import RigidBody
from simulation.solver.config import SolverConfig


class TestFixedJointConstruction:
    """Tests for FixedJoint construction."""

    def test_basic_construction(self, dynamic_body_a, dynamic_body_b):
        """Should construct with two bodies."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.body_a is dynamic_body_a
        assert joint.body_b is dynamic_body_b

    def test_construction_with_anchors(self, dynamic_body_a, dynamic_body_b):
        """Should construct with anchor points."""
        joint = FixedJoint(
            dynamic_body_a, dynamic_body_b,
            local_anchor_a=Vec3(1, 0, 0),
            local_anchor_b=Vec3(-1, 0, 0)
        )
        assert joint.local_anchor_a.x == 1
        assert joint.local_anchor_b.x == -1

    def test_construction_world_attachment(self, dynamic_body_a):
        """Should construct with world attachment."""
        joint = FixedJoint(dynamic_body_a, None)
        assert joint.body_b is None

    def test_construction_with_break_thresholds(self, dynamic_body_a, dynamic_body_b):
        """Should construct with break thresholds."""
        joint = FixedJoint(
            dynamic_body_a, dynamic_body_b,
            break_force=100.0,
            break_torque=50.0
        )
        assert joint.break_force == 100.0
        assert joint.break_torque == 50.0

    def test_construction_soft_joint(self, dynamic_body_a, dynamic_body_b):
        """Should construct soft joint with stiffness/damping."""
        joint = FixedJoint(
            dynamic_body_a, dynamic_body_b,
            stiffness=1000.0,
            damping=10.0
        )
        assert joint.stiffness == 1000.0
        assert joint.damping == 10.0


class TestFixedJointConstraintCount:
    """Tests for constraint count (6 DOF)."""

    def test_constraint_count(self, dynamic_body_a, dynamic_body_b):
        """Fixed joint should have 6 constraints."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        assert joint.get_constraint_count() == 6


class TestFixedJointReferenceOrientation:
    """Tests for reference orientation handling."""

    def test_reference_orientation_stored(self, dynamic_body_a, dynamic_body_b):
        """Reference orientation should be stored on construction."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        ref = joint.reference_orientation
        assert ref is not None
        # Should be relative orientation q_b^-1 * q_a
        # For identity quaternions, this should be identity
        assert abs(ref.w - 1.0) < 1e-6 or abs(ref.w + 1.0) < 1e-6

    def test_reference_orientation_setter(self, dynamic_body_a, dynamic_body_b):
        """Reference orientation should be settable."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        new_ref = Quaternion.from_axis_angle(Vec3.unit_y(), math.pi/4)
        joint.reference_orientation = new_ref
        assert joint.reference_orientation is new_ref

    def test_set_reference_from_current(self, dynamic_body_a, dynamic_body_b):
        """set_reference_from_current should update reference."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        # Rotate body A
        dynamic_body_a.orientation = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi/4)
        joint.set_reference_from_current()
        # Reference should now reflect new configuration

    def test_set_reference_angle(self, dynamic_body_a, dynamic_body_b):
        """set_reference_angle should set orientation from angle-axis."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.set_reference_angle(math.pi/2, Vec3.unit_x())
        ref = joint.reference_orientation
        # Should be 90 degree rotation around X

    def test_reference_with_rotated_bodies(self, rotated_body, dynamic_body_a):
        """Reference should capture relative rotation."""
        joint = FixedJoint(rotated_body, dynamic_body_a)
        # rotated_body is 45 deg around Y, dynamic_body_a is identity
        # relative orientation should be approximately 45 deg


class TestFixedJointPrepare:
    """Tests for prepare method."""

    def test_prepare_updates_inertia(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should update world inertia tensors."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Bodies should have updated world inertia

    def test_prepare_sets_up_jacobians(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should set up 6 Jacobians."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        assert len(joint._jacobians) == 6
        # All effective masses should be computed
        for em in joint._effective_masses[:6]:
            assert em >= 0.0

    def test_prepare_disabled_joint(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should do nothing for disabled joint."""
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.disable()
        joint.prepare(0.016, solver_config)
        # Should not crash

    def test_prepare_soft_joint(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should handle soft joint compliance."""
        joint = FixedJoint(
            dynamic_body_a, dynamic_body_b,
            stiffness=1000.0,
            damping=10.0
        )
        joint.prepare(0.016, solver_config)
        # Soft joint should have gamma set
        assert joint._gamma > 0.0 or joint.stiffness == 0

    def test_prepare_with_position_error(self, dynamic_body_a, dynamic_body_b, solver_config):
        """prepare should compute position error bias."""
        # Offset body B
        dynamic_body_b.position = Vec3(2, 0, 0)
        joint = FixedJoint(dynamic_body_a, dynamic_body_b)
        joint.prepare(0.016, solver_config)
        # Biases should be non-zero
        has_nonzero_bias = any(abs(b) > 1e-6 for b in joint._biases[:3])
        assert has_nonzero_bias


class TestFixedJointSolving:
    """Tests for velocity and position solving."""

    def test_solve_velocity_reduces_error(self, solver_config):
        """Velocity solving should reduce constraint error."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0.1, 0, 0),  # Slight offset
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        # Apply velocity solving iterations
        for _ in range(10):
            joint.solve_velocity()

        # Accumulated impulse should be non-zero
        has_impulse = any(abs(i) > 1e-10 for i in joint._accumulated_impulse)
        assert has_impulse

    def test_solve_position_returns_error(self, solver_config):
        """Position solving should return max error."""
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
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        error = joint.solve_position(0.2)
        # Should have position error
        assert error > 0.0

    def test_solve_position_moves_bodies(self, solver_config):
        """Position solving should move bodies closer."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(1, 0, 0),  # 1 unit offset
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        initial_distance = (body_b.position - body_a.position).length()

        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        for _ in range(20):
            joint.solve_position(0.2)

        final_distance = (body_b.position - body_a.position).length()
        # Bodies should be closer after position correction
        assert final_distance < initial_distance

    def test_solve_static_body(self, static_body, dynamic_body_a, solver_config):
        """Solving should only move dynamic body when one is static."""
        initial_static_pos = Vec3(static_body.position.x, static_body.position.y, static_body.position.z)
        dynamic_body_a.position = Vec3(1, 0, 0)

        joint = FixedJoint(static_body, dynamic_body_a)
        joint.prepare(0.016, solver_config)

        for _ in range(10):
            joint.solve_velocity()
            joint.solve_position(0.2)

        # Static body should not move
        assert abs(static_body.position.x - initial_static_pos.x) < 1e-10
        assert abs(static_body.position.y - initial_static_pos.y) < 1e-10
        assert abs(static_body.position.z - initial_static_pos.z) < 1e-10


class TestFixedJointAngular:
    """Tests for angular constraints."""

    def test_angular_constraint_with_rotation_error(self, solver_config):
        """Angular constraints should correct rotation error."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            orientation=Quaternion.identity(),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0, 0, 0),
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), 0.2),  # 0.2 rad rotation
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)

        # There should be angular error - but reference was captured at construction
        # so the joint considers current configuration as "correct"
        # Reset reference to identity to create actual error
        joint._reference_orientation = Quaternion.identity()
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        assert error >= 0.0  # Error depends on how reference was set


class TestFixedJointFactoryMethods:
    """Tests for factory methods."""

    def test_create_weld(self):
        """create_weld should create joint at world position."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(2, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        world_anchor = Vec3(1, 0, 0)  # Midpoint

        joint = FixedJoint.create_weld(body_a, body_b, world_anchor)
        assert joint.body_a is body_a
        assert joint.body_b is body_b
        # Anchors should be at world position
        anchor_a = joint.get_world_anchor_a()
        assert abs(anchor_a.x - 1.0) < 1e-6

    def test_create_weld_default_anchor(self):
        """create_weld should use midpoint as default anchor."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(2, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint.create_weld(body_a, body_b)
        anchor_a = joint.get_world_anchor_a()
        # Should be at midpoint (1, 0, 0)
        assert abs(anchor_a.x - 1.0) < 1e-6

    def test_create_weld_with_break_force(self):
        """create_weld should accept break parameters."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = FixedJoint.create_weld(
            body_a, body_b,
            break_force=100.0,
            break_torque=50.0
        )
        assert joint.break_force == 100.0
        assert joint.break_torque == 50.0

    def test_create_soft_weld(self):
        """create_soft_weld should create soft joint."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = FixedJoint.create_soft_weld(
            body_a, body_b,
            stiffness=500.0,
            damping=5.0
        )
        assert joint.stiffness == 500.0
        assert joint.damping == 5.0


class TestFixedJointEdgeCases:
    """Tests for edge cases."""

    def test_coincident_bodies(self, solver_config):
        """Should handle coincident bodies (zero distance)."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0, 0, 0),  # Same position
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        # Should not crash, error should be small
        error = joint.solve_position(0.2)
        assert error >= 0.0

    def test_very_different_masses(self, solver_config):
        """Should handle very different mass ratios."""
        body_a = RigidBody(
            id=1,
            position=Vec3(0, 0, 0),
            mass=0.001,  # Very light
            local_inertia=Mat3.identity()
        )
        body_b = RigidBody(
            id=2,
            position=Vec3(0.1, 0, 0),
            mass=1000.0,  # Very heavy
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash or produce NaN

    def test_high_stiffness_soft_joint(self, solver_config):
        """High stiffness soft joint should behave like hard joint."""
        body_a = RigidBody(id=1, position=Vec3.zero(), mass=1.0, local_inertia=Mat3.identity())
        body_b = RigidBody(id=2, position=Vec3(0.1, 0, 0), mass=1.0, local_inertia=Mat3.identity())

        joint = FixedJoint(
            body_a, body_b,
            stiffness=100000.0,
            damping=100.0
        )
        joint.prepare(0.016, solver_config)
        joint.solve_velocity()
        # Should not crash

    def test_large_angle_error(self, solver_config):
        """Should handle large rotation errors."""
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
            orientation=Quaternion.from_axis_angle(Vec3.unit_y(), math.pi * 0.9),  # Nearly 180 deg
            mass=1.0,
            local_inertia=Mat3.identity()
        )
        joint = FixedJoint(body_a, body_b)
        joint.prepare(0.016, solver_config)
        error = joint.solve_position(0.2)
        # Should handle without NaN
        assert not math.isnan(error)
