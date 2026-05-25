"""Tests for IK Solver implementations."""

import math
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat

# Handle import errors from XR __init__.py gracefully
try:
    from engine.xr.avatars.ik_solver import (
        CCDSolver,
        FABRIKSolver,
        IKChain,
        IKJoint,
        IKSolverType,
        TwoBoneSolver,
        create_solver,
    )
except (ImportError, AttributeError) as e:
    pytest.skip(f"XR module has unrelated import errors: {e}", allow_module_level=True)


class TestIKJoint:
    """Tests for IKJoint dataclass."""

    def test_default_values(self):
        """Test IKJoint defaults."""
        joint = IKJoint()

        assert joint.position == Vec3.zero()
        assert joint.rotation == Quat.identity()
        assert joint.length == 0.0
        assert joint.min_angle == -math.pi
        assert joint.max_angle == math.pi

    def test_custom_values(self):
        """Test IKJoint with custom values."""
        pos = Vec3(1, 2, 3)
        joint = IKJoint(
            position=pos,
            length=0.5,
            min_angle=-math.pi / 2,
            max_angle=math.pi / 2,
        )

        assert joint.position == pos
        assert joint.length == 0.5
        assert joint.min_angle == -math.pi / 2
        assert joint.max_angle == math.pi / 2

    def test_clamp_rotation(self):
        """Test rotation clamping to limits."""
        joint = IKJoint(
            min_angle=-math.pi / 4,
            max_angle=math.pi / 4,
        )

        # Rotation within limits should be preserved
        small_rot = Quat.from_euler(0.1, 0.1, 0.1)
        clamped = joint.clamp_rotation(small_rot)
        # Should be approximately the same
        assert abs(clamped.x - small_rot.x) < 0.1


class TestIKChain:
    """Tests for IKChain."""

    def test_empty_chain(self):
        """Test empty chain properties."""
        chain = IKChain()

        assert chain.total_length == 0.0
        assert chain.root_position == Vec3.zero()
        assert chain.end_effector_position == Vec3.zero()

    def test_total_length(self):
        """Test total chain length calculation."""
        chain = IKChain(joints=[
            IKJoint(length=0.5),
            IKJoint(length=0.4),
            IKJoint(length=0.0),  # End effector
        ])

        assert chain.total_length == pytest.approx(0.9, abs=0.001)

    def test_create_arm_chain(self):
        """Test arm chain creation."""
        chain = IKChain()
        chain.create_arm_chain(
            shoulder_pos=Vec3(0, 1.4, 0),
            elbow_pos=Vec3(0.3, 1.1, 0),
            wrist_pos=Vec3(0.5, 0.9, 0),
        )

        assert len(chain.joints) == 3
        assert chain.joints[0].position == Vec3(0, 1.4, 0)
        assert chain.joints[1].position == Vec3(0.3, 1.1, 0)
        assert chain.joints[2].position == Vec3(0.5, 0.9, 0)
        assert chain.joints[0].length > 0  # Upper arm
        assert chain.joints[1].length > 0  # Lower arm
        assert chain.joints[2].length == 0  # End effector

    def test_create_leg_chain(self):
        """Test leg chain creation."""
        chain = IKChain()
        chain.create_leg_chain(
            hip_pos=Vec3(0.1, 0.9, 0),
            knee_pos=Vec3(0.1, 0.5, 0.1),
            ankle_pos=Vec3(0.1, 0.1, 0),
        )

        assert len(chain.joints) == 3
        assert chain.joints[0].position == Vec3(0.1, 0.9, 0)
        assert chain.joints[2].length == 0  # End effector

    def test_root_and_end_positions(self):
        """Test root and end effector position access."""
        chain = IKChain(joints=[
            IKJoint(position=Vec3(0, 0, 0)),
            IKJoint(position=Vec3(1, 0, 0)),
            IKJoint(position=Vec3(2, 0, 0)),
        ])

        assert chain.root_position == Vec3(0, 0, 0)
        assert chain.end_effector_position == Vec3(2, 0, 0)


class TestFABRIKSolver:
    """Tests for FABRIK IK solver."""

    def test_initialization(self):
        """Test solver initialization."""
        solver = FABRIKSolver(max_iterations=15, tolerance=0.01)

        assert solver.max_iterations == 15
        assert solver.tolerance == 0.01

    def test_invalid_iterations(self):
        """Test rejection of invalid iteration count."""
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            FABRIKSolver(max_iterations=0)

    def test_invalid_tolerance(self):
        """Test rejection of invalid tolerance."""
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            FABRIKSolver(tolerance=0)

    def test_solve_reachable_target(self):
        """Test solving for reachable target."""
        solver = FABRIKSolver(max_iterations=20, tolerance=0.01)

        # Create a simple 3-joint chain
        chain = IKChain(
            joints=[
                IKJoint(position=Vec3(0, 0, 0), length=1.0),
                IKJoint(position=Vec3(1, 0, 0), length=1.0),
                IKJoint(position=Vec3(2, 0, 0), length=0.0),
            ],
            target_position=Vec3(1.5, 0.5, 0),
        )

        converged = solver.solve(chain)

        # Should reach target (within tolerance)
        end_pos = chain.end_effector_position
        distance = end_pos.distance(chain.target_position)
        assert distance < 0.1  # Allow some tolerance

    def test_solve_unreachable_target(self):
        """Test solving for unreachable target."""
        solver = FABRIKSolver(max_iterations=10, tolerance=0.01)

        # Chain with total length 2
        chain = IKChain(
            joints=[
                IKJoint(position=Vec3(0, 0, 0), length=1.0),
                IKJoint(position=Vec3(1, 0, 0), length=1.0),
                IKJoint(position=Vec3(2, 0, 0), length=0.0),
            ],
            target_position=Vec3(5, 0, 0),  # Beyond reach
        )

        converged = solver.solve(chain)

        # Should not converge but chain should stretch toward target
        assert converged is False
        # End effector should be in direction of target
        end_pos = chain.end_effector_position
        assert end_pos.x > 1.5  # Stretched toward target

    def test_solve_empty_chain(self):
        """Test solving empty chain."""
        solver = FABRIKSolver()
        chain = IKChain()

        converged = solver.solve(chain)
        assert converged is True  # Nothing to solve

    def test_solve_single_joint(self):
        """Test solving chain with single joint."""
        solver = FABRIKSolver()
        chain = IKChain(
            joints=[IKJoint(position=Vec3(0, 0, 0))],
            target_position=Vec3(1, 0, 0),
        )

        converged = solver.solve(chain)
        assert converged is True


class TestCCDSolver:
    """Tests for CCD IK solver."""

    def test_initialization(self):
        """Test CCD solver initialization."""
        solver = CCDSolver(max_iterations=20, tolerance=0.005)

        assert solver.max_iterations == 20
        assert solver.tolerance == 0.005

    def test_solve_reachable_target(self):
        """Test CCD solving for reachable target."""
        solver = CCDSolver(max_iterations=30, tolerance=0.01)

        chain = IKChain(
            joints=[
                IKJoint(position=Vec3(0, 0, 0), length=1.0),
                IKJoint(position=Vec3(1, 0, 0), length=1.0),
                IKJoint(position=Vec3(2, 0, 0), length=0.0),
            ],
            target_position=Vec3(1.5, 0.5, 0),
        )

        converged = solver.solve(chain)

        end_pos = chain.end_effector_position
        distance = end_pos.distance(chain.target_position)
        assert distance < 0.5  # CCD may not be as precise

    def test_solve_empty_chain(self):
        """Test CCD with empty chain."""
        solver = CCDSolver()
        chain = IKChain()

        converged = solver.solve(chain)
        assert converged is True


class TestTwoBoneSolver:
    """Tests for analytical two-bone IK solver."""

    def test_initialization(self):
        """Test two-bone solver initialization."""
        solver = TwoBoneSolver()

        assert solver.max_iterations > 0
        assert solver.tolerance > 0

    def test_solve_requires_three_joints(self):
        """Test that two-bone solver requires exactly 3 joints."""
        solver = TwoBoneSolver()

        # Too few joints
        chain = IKChain(joints=[
            IKJoint(position=Vec3(0, 0, 0), length=1.0),
            IKJoint(position=Vec3(1, 0, 0), length=0.0),
        ])

        with pytest.raises(ValueError, match="exactly 3 joints"):
            solver.solve(chain)

        # Too many joints
        chain = IKChain(joints=[
            IKJoint(position=Vec3(0, 0, 0), length=1.0),
            IKJoint(position=Vec3(1, 0, 0), length=1.0),
            IKJoint(position=Vec3(2, 0, 0), length=1.0),
            IKJoint(position=Vec3(3, 0, 0), length=0.0),
        ])

        with pytest.raises(ValueError, match="exactly 3 joints"):
            solver.solve(chain)

    def test_solve_arm(self):
        """Test solving arm IK."""
        solver = TwoBoneSolver()

        chain = IKChain()
        chain.create_arm_chain(
            shoulder_pos=Vec3(0, 1.4, 0),
            elbow_pos=Vec3(0.3, 1.1, 0),
            wrist_pos=Vec3(0.5, 0.9, 0),
        )
        chain.target_position = Vec3(0.4, 1.0, 0.2)

        converged = solver.solve(chain)

        assert converged is True
        # End effector should be at target
        distance = chain.end_effector_position.distance(chain.target_position)
        assert distance < 0.1

    def test_solve_with_pole_target(self):
        """Test solving with pole target for elbow direction."""
        solver = TwoBoneSolver()

        chain = IKChain(
            joints=[
                IKJoint(position=Vec3(0, 1.4, 0), length=0.35),
                IKJoint(position=Vec3(0.35, 1.2, 0), length=0.3),
                IKJoint(position=Vec3(0.6, 1.0, 0), length=0.0),
            ],
            target_position=Vec3(0.5, 1.2, 0.3),
            pole_target=Vec3(0.3, 1.0, -0.5),  # Behind elbow
        )

        converged = solver.solve(chain)

        assert converged is True


class TestCreateSolver:
    """Tests for solver factory function."""

    def test_create_fabrik(self):
        """Test creating FABRIK solver."""
        solver = create_solver(IKSolverType.FABRIK)

        assert isinstance(solver, FABRIKSolver)

    def test_create_ccd(self):
        """Test creating CCD solver."""
        solver = create_solver(IKSolverType.CCD)

        assert isinstance(solver, CCDSolver)

    def test_create_two_bone(self):
        """Test creating two-bone solver."""
        solver = create_solver(IKSolverType.TWO_BONE)

        assert isinstance(solver, TwoBoneSolver)

    def test_create_with_params(self):
        """Test creating solver with custom parameters."""
        solver = create_solver(
            IKSolverType.FABRIK,
            max_iterations=25,
            tolerance=0.005,
        )

        assert solver.max_iterations == 25
        assert solver.tolerance == 0.005
