"""IK Solver implementations for avatar body estimation.

Provides FABRIK, CCD, and TwoBone IK solvers for computing
joint positions from end effector targets.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import RigidTransform
from engine.core.constants import MATH_EPSILON
from engine.xr.config import XR_CONFIG


class IKSolverType(Enum):
    """Available IK solver algorithms."""
    FABRIK = auto()    # Forward And Backward Reaching Inverse Kinematics
    CCD = auto()       # Cyclic Coordinate Descent
    TWO_BONE = auto()  # Analytical two-bone solver (arm/leg)


@dataclass(slots=True)
class IKJoint:
    """Represents a joint in an IK chain.

    Attributes:
        position: World position of joint
        rotation: World rotation of joint
        length: Length to next joint (0 for end effector)
        min_angle: Minimum rotation angle (radians)
        max_angle: Maximum rotation angle (radians)
        twist_axis: Axis for twist rotation
        swing_axis: Axis for swing rotation
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    length: float = 0.0
    min_angle: float = -math.pi
    max_angle: float = math.pi
    twist_axis: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    swing_axis: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))

    def clamp_rotation(self, rotation: Quat) -> Quat:
        """Clamp rotation to joint limits.

        Args:
            rotation: Proposed rotation

        Returns:
            Clamped rotation
        """
        pitch, yaw, roll = rotation.to_euler()

        # Clamp all angles to limits
        pitch = max(self.min_angle, min(self.max_angle, pitch))
        yaw = max(self.min_angle, min(self.max_angle, yaw))
        roll = max(self.min_angle, min(self.max_angle, roll))

        return Quat.from_euler(pitch, yaw, roll)


@dataclass(slots=True)
class IKChain:
    """Chain of joints for IK solving.

    Attributes:
        joints: List of joints from root to end effector
        target_position: Desired end effector position
        target_rotation: Desired end effector rotation
        pole_target: Optional pole vector for elbow/knee direction
    """
    joints: list[IKJoint] = field(default_factory=list)
    target_position: Vec3 = field(default_factory=Vec3.zero)
    target_rotation: Quat = field(default_factory=Quat.identity)
    pole_target: Optional[Vec3] = None

    @property
    def total_length(self) -> float:
        """Get total length of the chain."""
        return sum(joint.length for joint in self.joints)

    @property
    def root_position(self) -> Vec3:
        """Get root joint position."""
        if self.joints:
            return self.joints[0].position
        return Vec3.zero()

    @property
    def end_effector_position(self) -> Vec3:
        """Get end effector position."""
        if self.joints:
            return self.joints[-1].position
        return Vec3.zero()

    def create_arm_chain(
        self,
        shoulder_pos: Vec3,
        elbow_pos: Vec3,
        wrist_pos: Vec3,
    ) -> None:
        """Create a 3-joint arm chain.

        Args:
            shoulder_pos: Shoulder joint position
            elbow_pos: Elbow joint position
            wrist_pos: Wrist joint position
        """
        upper_arm_length = shoulder_pos.distance(elbow_pos)
        lower_arm_length = elbow_pos.distance(wrist_pos)

        self.joints = [
            IKJoint(
                position=shoulder_pos,
                length=upper_arm_length,
                min_angle=-math.pi * 0.9,
                max_angle=math.pi * 0.9,
            ),
            IKJoint(
                position=elbow_pos,
                length=lower_arm_length,
                min_angle=0.0,  # Elbow can't bend backwards
                max_angle=math.pi * 0.85,
            ),
            IKJoint(
                position=wrist_pos,
                length=0.0,  # End effector
            ),
        ]

    def create_leg_chain(
        self,
        hip_pos: Vec3,
        knee_pos: Vec3,
        ankle_pos: Vec3,
    ) -> None:
        """Create a 3-joint leg chain.

        Args:
            hip_pos: Hip joint position
            knee_pos: Knee joint position
            ankle_pos: Ankle joint position
        """
        thigh_length = hip_pos.distance(knee_pos)
        shin_length = knee_pos.distance(ankle_pos)

        self.joints = [
            IKJoint(
                position=hip_pos,
                length=thigh_length,
                min_angle=-math.pi * 0.5,
                max_angle=math.pi * 0.5,
            ),
            IKJoint(
                position=knee_pos,
                length=shin_length,
                min_angle=-math.pi * 0.85,  # Knee bends backwards
                max_angle=0.0,
            ),
            IKJoint(
                position=ankle_pos,
                length=0.0,
            ),
        ]


class IKSolver(ABC):
    """Abstract base class for IK solvers.

    Subclasses implement specific algorithms for computing
    joint positions from end effector targets.
    """

    def __init__(self, max_iterations: int = XR_CONFIG.avatar.IK_MAX_ITERATIONS, tolerance: float = XR_CONFIG.avatar.IK_TOLERANCE):
        """Initialize solver.

        Args:
            max_iterations: Maximum solver iterations
            tolerance: Position tolerance for convergence
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        self._max_iterations = max_iterations
        self._tolerance = tolerance

    @property
    def max_iterations(self) -> int:
        """Get maximum iterations."""
        return self._max_iterations

    @property
    def tolerance(self) -> float:
        """Get convergence tolerance."""
        return self._tolerance

    @abstractmethod
    def solve(self, chain: IKChain) -> bool:
        """Solve IK for the given chain.

        Args:
            chain: IK chain to solve

        Returns:
            True if converged within tolerance
        """
        pass

    def _is_reachable(self, chain: IKChain) -> bool:
        """Check if target is reachable by the chain.

        Args:
            chain: IK chain

        Returns:
            True if target is within reach
        """
        root_to_target = chain.target_position - chain.root_position
        distance = root_to_target.length()
        return distance <= chain.total_length + self._tolerance


class FABRIKSolver(IKSolver):
    """Forward And Backward Reaching Inverse Kinematics solver.

    FABRIK is a fast, iterative solver that works by alternating
    forward and backward passes along the chain. It handles
    arbitrary chain lengths and is computationally efficient.
    """

    def solve(self, chain: IKChain) -> bool:
        """Solve IK using FABRIK algorithm.

        Args:
            chain: IK chain to solve

        Returns:
            True if converged within tolerance
        """
        if len(chain.joints) < 2:
            return True

        # Store original root position (it should not move)
        root_pos = chain.joints[0].position

        # Check reachability
        if not self._is_reachable(chain):
            # Target unreachable - stretch toward it
            direction = (chain.target_position - root_pos).normalized()
            current_pos = root_pos

            for i, joint in enumerate(chain.joints):
                joint.position = current_pos
                if i < len(chain.joints) - 1:
                    current_pos = current_pos + direction * joint.length

            return False

        # Iterate forward and backward passes
        for iteration in range(self._max_iterations):
            # Check convergence
            distance_to_target = chain.end_effector_position.distance(
                chain.target_position
            )
            if distance_to_target <= self._tolerance:
                return True

            # Forward pass (from end effector to root)
            self._forward_pass(chain)

            # Backward pass (from root to end effector)
            self._backward_pass(chain, root_pos)

        # Did not converge within max iterations
        return False

    def _forward_pass(self, chain: IKChain) -> None:
        """Forward reaching: move end effector to target, pull chain."""
        # Set end effector to target
        chain.joints[-1].position = chain.target_position

        # Work backwards to root
        for i in range(len(chain.joints) - 2, -1, -1):
            current = chain.joints[i]
            next_joint = chain.joints[i + 1]

            # Direction from next joint to current
            direction = (current.position - next_joint.position).normalized()

            # Place current joint at proper distance from next
            current.position = next_joint.position + direction * current.length

    def _backward_pass(self, chain: IKChain, root_pos: Vec3) -> None:
        """Backward reaching: restore root, push chain forward."""
        # Restore root position
        chain.joints[0].position = root_pos

        # Work forwards to end effector
        for i in range(len(chain.joints) - 1):
            current = chain.joints[i]
            next_joint = chain.joints[i + 1]

            # Direction from current to next joint
            direction = (next_joint.position - current.position).normalized()

            # Place next joint at proper distance from current
            next_joint.position = current.position + direction * current.length


class CCDSolver(IKSolver):
    """Cyclic Coordinate Descent IK solver.

    CCD iteratively adjusts each joint rotation to point toward
    the target. It handles joint limits naturally and is good
    for chains with constrained joints.
    """

    def solve(self, chain: IKChain) -> bool:
        """Solve IK using CCD algorithm.

        Args:
            chain: IK chain to solve

        Returns:
            True if converged within tolerance
        """
        if len(chain.joints) < 2:
            return True

        for iteration in range(self._max_iterations):
            # Check convergence
            distance_to_target = chain.end_effector_position.distance(
                chain.target_position
            )
            if distance_to_target <= self._tolerance:
                return True

            # Iterate through joints from end effector toward root
            for i in range(len(chain.joints) - 2, -1, -1):
                joint = chain.joints[i]
                end_effector = chain.joints[-1]

                # Vector from joint to end effector
                to_end = end_effector.position - joint.position

                # Vector from joint to target
                to_target = chain.target_position - joint.position

                # Skip if vectors are too short
                if to_end.length() < MATH_EPSILON or to_target.length() < MATH_EPSILON:
                    continue

                # Normalize
                to_end = to_end.normalized()
                to_target = to_target.normalized()

                # Calculate rotation axis and angle
                dot = to_end.dot(to_target)
                dot = max(-1.0, min(1.0, dot))  # Clamp for acos
                angle = math.acos(dot)

                if angle < MATH_EPSILON:
                    continue

                # Rotation axis
                axis = to_end.cross(to_target)
                if axis.length() < MATH_EPSILON:
                    continue
                axis = axis.normalized()

                # Create rotation
                rotation = Quat.from_axis_angle(axis, angle)

                # Apply joint limits to resulting joint rotation
                new_rotation = rotation * joint.rotation
                joint.rotation = joint.clamp_rotation(new_rotation)

                # Apply rotation to this joint and all children
                self._apply_rotation(chain, i, joint.position, rotation)

        return False

    def _apply_rotation(
        self,
        chain: IKChain,
        joint_index: int,
        pivot: Vec3,
        rotation: Quat,
    ) -> None:
        """Apply rotation to joint and all its children.

        Args:
            chain: IK chain
            joint_index: Index of joint to rotate
            pivot: Pivot point for rotation
            rotation: Rotation to apply
        """
        for i in range(joint_index + 1, len(chain.joints)):
            joint = chain.joints[i]

            # Rotate position around pivot
            relative_pos = joint.position - pivot
            rotated_pos = rotation.rotate_vector(relative_pos)
            joint.position = pivot + rotated_pos

            # Update joint rotation
            joint.rotation = rotation * joint.rotation


class TwoBoneSolver(IKSolver):
    """Analytical two-bone IK solver.

    Uses trigonometry to solve IK for exactly two bones (e.g., arm or leg).
    This is faster than iterative methods and produces exact solutions.
    Supports pole targets for controlling elbow/knee direction.
    """

    def solve(self, chain: IKChain) -> bool:
        """Solve IK using analytical two-bone method.

        Args:
            chain: IK chain (must have exactly 3 joints)

        Returns:
            True if solved successfully
        """
        if len(chain.joints) != 3:
            raise ValueError("TwoBoneSolver requires exactly 3 joints")

        root = chain.joints[0]
        mid = chain.joints[1]
        end = chain.joints[2]

        # Bone lengths
        upper_length = root.length
        lower_length = mid.length

        # Vector from root to target
        root_to_target = chain.target_position - root.position
        target_distance = root_to_target.length()

        # Clamp target distance to reachable range
        max_reach = upper_length + lower_length - MATH_EPSILON
        min_reach = abs(upper_length - lower_length) + MATH_EPSILON

        target_distance = max(min_reach, min(max_reach, target_distance))

        # Calculate mid joint angle using law of cosines
        # c^2 = a^2 + b^2 - 2ab*cos(C)
        # cos(C) = (a^2 + b^2 - c^2) / (2ab)
        cos_angle = (
            upper_length * upper_length
            + lower_length * lower_length
            - target_distance * target_distance
        ) / (2.0 * upper_length * lower_length)

        cos_angle = max(-1.0, min(1.0, cos_angle))
        mid_angle = math.acos(cos_angle)

        # Calculate root joint angle
        # Use law of sines or geometry
        cos_root_angle = (
            upper_length * upper_length
            + target_distance * target_distance
            - lower_length * lower_length
        ) / (2.0 * upper_length * target_distance)

        cos_root_angle = max(-1.0, min(1.0, cos_root_angle))
        root_angle = math.acos(cos_root_angle)

        # Direction to target
        target_dir = root_to_target.normalized() if target_distance > MATH_EPSILON else Vec3.forward()

        # Calculate pole plane normal
        if chain.pole_target is not None:
            # Use pole target to determine bend direction
            pole_dir = (chain.pole_target - root.position).normalized()
            plane_normal = target_dir.cross(pole_dir)
            if plane_normal.length() < MATH_EPSILON:
                plane_normal = Vec3.up()
            else:
                plane_normal = plane_normal.normalized()
        else:
            # Default: bend forward (negative Z)
            plane_normal = target_dir.cross(Vec3.up())
            if plane_normal.length() < MATH_EPSILON:
                plane_normal = Vec3.right()
            else:
                plane_normal = plane_normal.normalized()

        # Perpendicular direction in the bend plane
        bend_dir = plane_normal.cross(target_dir).normalized()

        # Position the mid joint
        # It lies on a plane perpendicular to the target direction
        # at a distance determined by the root angle
        mid_offset_along_target = math.cos(root_angle) * upper_length
        mid_offset_perpendicular = math.sin(root_angle) * upper_length

        mid.position = (
            root.position
            + target_dir * mid_offset_along_target
            + bend_dir * mid_offset_perpendicular
        )

        # Position the end effector
        end.position = chain.target_position

        # Calculate rotations
        # Root rotation: point toward mid joint
        root_forward = (mid.position - root.position).normalized()
        root.rotation = self._look_rotation(root_forward, Vec3.up())

        # Mid rotation: point toward end effector
        mid_forward = (end.position - mid.position).normalized()
        mid.rotation = self._look_rotation(mid_forward, Vec3.up())

        # End rotation: use target rotation
        end.rotation = chain.target_rotation

        return True

    def _look_rotation(self, forward: Vec3, up: Vec3) -> Quat:
        """Create rotation that looks along forward direction.

        Args:
            forward: Forward direction
            up: Up reference direction

        Returns:
            Rotation quaternion
        """
        if forward.length() < MATH_EPSILON:
            return Quat.identity()

        forward = forward.normalized()

        # Ensure up is not parallel to forward
        dot = abs(forward.dot(up))
        if dot > 0.999:
            up = Vec3.right() if abs(forward.dot(Vec3.right())) < 0.9 else Vec3.forward()

        right = up.cross(forward).normalized()
        actual_up = forward.cross(right).normalized()

        # Build rotation matrix and convert to quaternion
        # This is a simplified implementation
        m00, m01, m02 = right.x, actual_up.x, forward.x
        m10, m11, m12 = right.y, actual_up.y, forward.y
        m20, m21, m22 = right.z, actual_up.z, forward.z

        trace = m00 + m11 + m22

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        return Quat(x, y, z, w).normalized()


def create_solver(
    solver_type: IKSolverType,
    max_iterations: int = XR_CONFIG.avatar.IK_MAX_ITERATIONS,
    tolerance: float = XR_CONFIG.avatar.IK_TOLERANCE,
) -> IKSolver:
    """Factory function to create an IK solver.

    Args:
        solver_type: Type of solver to create
        max_iterations: Maximum iterations for iterative solvers
        tolerance: Position tolerance for convergence

    Returns:
        IK solver instance
    """
    if solver_type == IKSolverType.FABRIK:
        return FABRIKSolver(max_iterations, tolerance)
    elif solver_type == IKSolverType.CCD:
        return CCDSolver(max_iterations, tolerance)
    elif solver_type == IKSolverType.TWO_BONE:
        return TwoBoneSolver(max_iterations, tolerance)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
