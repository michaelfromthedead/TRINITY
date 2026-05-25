"""FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.

FABRIK is an iterative IK solver that works by alternating between forward
and backward passes through the chain, adjusting bone positions to satisfy
length constraints while moving toward the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Callable

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FABRIK_DEFAULT_MAX_ITERATIONS,
    FABRIK_MULTI_CHAIN_MAX_ITERATIONS,
    JOINT_DEFAULT_CONE_ANGLE,
)


class JointConstraintType(Enum):
    """Types of joint constraints for FABRIK."""

    NONE = auto()
    """No constraint - free rotation."""

    HINGE = auto()
    """Hinge joint - rotation around single axis."""

    BALL_SOCKET = auto()
    """Ball socket - rotation within cone."""

    TWIST_LIMIT = auto()
    """Twist constraint - limited rotation around bone axis."""


@dataclass
class JointConstraint:
    """Constraint for a joint in the FABRIK chain.

    Attributes:
        constraint_type: Type of constraint
        axis: Constraint axis (for hinge joints)
        min_angle: Minimum angle (radians)
        max_angle: Maximum angle (radians)
        cone_angle: Half-angle of cone (for ball-socket)
        twist_min: Minimum twist (radians)
        twist_max: Maximum twist (radians)
    """

    constraint_type: JointConstraintType = JointConstraintType.NONE
    axis: Vec3 = field(default_factory=lambda: Vec3.unit_y())
    min_angle: float = -math.pi
    max_angle: float = math.pi
    cone_angle: float = JOINT_DEFAULT_CONE_ANGLE
    twist_min: float = -math.pi
    twist_max: float = math.pi

    def apply(self, direction: Vec3, parent_rotation: Quat) -> Vec3:
        """Apply constraint to a direction vector.

        Args:
            direction: Desired direction (normalized)
            parent_rotation: Parent bone rotation for reference frame

        Returns:
            Constrained direction.
        """
        if self.constraint_type == JointConstraintType.NONE:
            return direction

        if self.constraint_type == JointConstraintType.HINGE:
            return self._apply_hinge(direction, parent_rotation)

        if self.constraint_type == JointConstraintType.BALL_SOCKET:
            return self._apply_ball_socket(direction, parent_rotation)

        return direction

    def _apply_hinge(self, direction: Vec3, parent_rotation: Quat) -> Vec3:
        """Apply hinge constraint.

        Args:
            direction: Desired direction
            parent_rotation: Parent rotation for reference

        Returns:
            Constrained direction on hinge plane.
        """
        # Transform axis to world space
        world_axis = parent_rotation.rotate_vector(self.axis)

        # Project direction onto plane perpendicular to axis
        projected = direction - world_axis * direction.dot(world_axis)

        if projected.length_squared() < MATH_EPSILON:
            # Direction parallel to axis - use forward
            forward = parent_rotation.rotate_vector(Vec3(0, 0, 1))
            return forward

        return projected.normalized()

    def _apply_ball_socket(self, direction: Vec3, parent_rotation: Quat) -> Vec3:
        """Apply ball-socket cone constraint.

        Args:
            direction: Desired direction
            parent_rotation: Parent rotation for cone axis

        Returns:
            Constrained direction within cone.
        """
        # Reference direction (rest pose direction)
        ref_dir = parent_rotation.rotate_vector(Vec3(0, 1, 0))

        # Angle between desired and reference
        dot = direction.dot(ref_dir)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)

        if angle <= self.cone_angle:
            return direction

        # Clamp to cone surface
        axis = ref_dir.cross(direction)
        if axis.length_squared() < MATH_EPSILON:
            return ref_dir

        axis = axis.normalized()
        rotation = Quat.from_axis_angle(axis, self.cone_angle)
        return rotation.rotate_vector(ref_dir)


@dataclass
class FABRIKResult:
    """Result from FABRIK solve operation.

    Attributes:
        success: Whether solve converged
        iterations: Number of iterations used
        final_error: Distance from end effector to target
        positions: Final joint positions
        rotations: Computed joint rotations
    """

    success: bool
    iterations: int = 0
    final_error: float = float('inf')
    positions: List[Vec3] = field(default_factory=list)
    rotations: List[Quat] = field(default_factory=list)


class FABRIKChain:
    """FABRIK inverse kinematics chain solver.

    FABRIK works by:
    1. Forward pass: Move end effector to target, adjust each joint
       backward maintaining bone lengths
    2. Backward pass: Move root to original position, adjust each
       joint forward maintaining bone lengths
    3. Repeat until convergence or max iterations

    Attributes:
        bone_indices: Indices of bones in the chain (root to tip)
        tolerance: Distance threshold for convergence
        max_iterations: Maximum solver iterations
    """

    def __init__(
        self,
        bone_indices: List[int],
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = FABRIK_DEFAULT_MAX_ITERATIONS
    ) -> None:
        """Initialize the FABRIK chain.

        Args:
            bone_indices: List of bone indices from root to end effector
            tolerance: Convergence distance threshold
            max_iterations: Maximum iterations before giving up

        Raises:
            ValueError: If chain has fewer than 2 bones
        """
        if len(bone_indices) < 2:
            raise ValueError("FABRIK chain requires at least 2 bones")

        self.bone_indices = list(bone_indices)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Joint constraints (one per bone)
        self._constraints: List[JointConstraint] = [
            JointConstraint() for _ in bone_indices
        ]

        # Cached bone lengths
        self._bone_lengths: List[float] = []
        self._total_length: float = 0.0
        self._lengths_cached: bool = False

    @property
    def chain_length(self) -> int:
        """Number of bones in the chain."""
        return len(self.bone_indices)

    @property
    def root_index(self) -> int:
        """Index of root bone."""
        return self.bone_indices[0]

    @property
    def end_index(self) -> int:
        """Index of end effector bone."""
        return self.bone_indices[-1]

    def set_constraint(self, joint_index: int, constraint: JointConstraint) -> None:
        """Set constraint for a specific joint.

        Args:
            joint_index: Index within chain (0 = root)
            constraint: Constraint to apply
        """
        if 0 <= joint_index < len(self._constraints):
            self._constraints[joint_index] = constraint

    def _cache_bone_lengths(self, positions: List[Vec3]) -> None:
        """Cache bone lengths from positions.

        Args:
            positions: World positions of all joints
        """
        self._bone_lengths = []
        self._total_length = 0.0

        for i in range(len(positions) - 1):
            length = (positions[i + 1] - positions[i]).length()
            self._bone_lengths.append(length)
            self._total_length += length

        self._lengths_cached = True

    def solve(
        self,
        positions: List[Vec3],
        target: Vec3,
        rotations: Optional[List[Quat]] = None
    ) -> FABRIKResult:
        """Solve the FABRIK chain.

        Args:
            positions: Current world positions of all joints in chain
            target: Target position for end effector
            rotations: Optional current rotations (for constraints)

        Returns:
            FABRIKResult with new positions and rotations.
        """
        if len(positions) != self.chain_length:
            raise ValueError(
                f"Expected {self.chain_length} positions, got {len(positions)}"
            )

        # Cache bone lengths
        if not self._lengths_cached:
            self._cache_bone_lengths(positions)

        # Copy positions for modification
        pos = [Vec3(p.x, p.y, p.z) for p in positions]
        rots = rotations if rotations else [Quat.identity() for _ in positions]
        rots = [Quat(r.x, r.y, r.z, r.w) for r in rots]

        # Store original root position
        root_pos = Vec3(pos[0].x, pos[0].y, pos[0].z)

        # Check if target is reachable
        dist_to_target = (target - root_pos).length()

        if dist_to_target > self._total_length:
            # Target unreachable - extend chain toward target
            direction = (target - root_pos).normalized()
            pos[0] = root_pos

            for i in range(len(self._bone_lengths)):
                pos[i + 1] = pos[i] + direction * self._bone_lengths[i]

            return FABRIKResult(
                success=False,
                iterations=0,
                final_error=dist_to_target - self._total_length,
                positions=pos,
                rotations=self._compute_rotations(pos, rots)
            )

        # Iterative solve
        for iteration in range(self.max_iterations):
            # Check convergence
            end_pos = pos[-1]
            error = (target - end_pos).length()

            if error <= self.tolerance:
                return FABRIKResult(
                    success=True,
                    iterations=iteration + 1,
                    final_error=error,
                    positions=pos,
                    rotations=self._compute_rotations(pos, rots)
                )

            # Forward pass (end to root)
            pos = self._forward_pass(pos, target, rots)

            # Backward pass (root to end)
            pos = self._backward_pass(pos, root_pos, rots)

        # Compute final error
        final_error = (target - pos[-1]).length()

        return FABRIKResult(
            success=final_error <= self.tolerance,
            iterations=self.max_iterations,
            final_error=final_error,
            positions=pos,
            rotations=self._compute_rotations(pos, rots)
        )

    def _forward_pass(
        self,
        positions: List[Vec3],
        target: Vec3,
        rotations: List[Quat]
    ) -> List[Vec3]:
        """Forward pass: end effector to root.

        Sets end effector to target and works backward, maintaining
        bone lengths.

        Args:
            positions: Current joint positions
            target: Target position
            rotations: Current rotations (for constraints)

        Returns:
            Updated positions after forward pass.
        """
        pos = list(positions)

        # Set end effector to target
        pos[-1] = Vec3(target.x, target.y, target.z)

        # Work backward
        for i in range(len(pos) - 2, -1, -1):
            # Direction from next joint to current
            direction = pos[i] - pos[i + 1]
            length = direction.length()

            if length < MATH_EPSILON:
                # Joints coincident - use arbitrary direction
                direction = Vec3(0, 1, 0)
            else:
                direction = direction / length

            # Apply constraint
            constraint = self._constraints[i]
            parent_rot = rotations[max(0, i - 1)] if i > 0 else Quat.identity()
            direction = constraint.apply(direction, parent_rot)

            # Place joint at bone length from next joint
            bone_length = self._bone_lengths[i]
            pos[i] = pos[i + 1] + direction * bone_length

        return pos

    def _backward_pass(
        self,
        positions: List[Vec3],
        root_pos: Vec3,
        rotations: List[Quat]
    ) -> List[Vec3]:
        """Backward pass: root to end effector.

        Sets root to original position and works forward, maintaining
        bone lengths.

        Args:
            positions: Current joint positions
            root_pos: Original root position
            rotations: Current rotations (for constraints)

        Returns:
            Updated positions after backward pass.
        """
        pos = list(positions)

        # Set root to original position
        pos[0] = Vec3(root_pos.x, root_pos.y, root_pos.z)

        # Work forward
        for i in range(len(pos) - 1):
            # Direction from current to next
            direction = pos[i + 1] - pos[i]
            length = direction.length()

            if length < MATH_EPSILON:
                direction = Vec3(0, 1, 0)
            else:
                direction = direction / length

            # Apply constraint
            constraint = self._constraints[i]
            parent_rot = rotations[max(0, i - 1)] if i > 0 else Quat.identity()
            direction = constraint.apply(direction, parent_rot)

            # Place next joint at bone length from current
            bone_length = self._bone_lengths[i]
            pos[i + 1] = pos[i] + direction * bone_length

        return pos

    def _compute_rotations(
        self,
        positions: List[Vec3],
        original_rotations: List[Quat]
    ) -> List[Quat]:
        """Compute rotations from new positions.

        Args:
            positions: Final joint positions
            original_rotations: Original rotations for reference

        Returns:
            List of rotations for each joint.
        """
        rotations = []

        for i in range(len(positions) - 1):
            # Direction to next joint
            direction = (positions[i + 1] - positions[i]).normalized()

            # Compute rotation that aligns Y-axis with direction
            rotation = self._rotation_to_direction(direction, original_rotations[i])
            rotations.append(rotation)

        # End effector keeps its original rotation
        rotations.append(original_rotations[-1])

        return rotations

    def _rotation_to_direction(self, direction: Vec3, current: Quat) -> Quat:
        """Compute rotation to point at direction.

        Args:
            direction: Desired forward direction
            current: Current rotation for reference

        Returns:
            Rotation quaternion.
        """
        # Default bone direction is Y-up
        default_dir = Vec3(0, 1, 0)

        # Rotation from default to target
        dot = default_dir.dot(direction)

        if dot > 1.0 - MATH_EPSILON:
            return Quat.identity()

        if dot < -1.0 + MATH_EPSILON:
            # Opposite direction - rotate 180 around perpendicular
            axis = Vec3.unit_x()
            return Quat.from_axis_angle(axis, math.pi)

        axis = default_dir.cross(direction).normalized()
        angle = math.acos(max(-1.0, min(1.0, dot)))

        return Quat.from_axis_angle(axis, angle)

    def solve_with_transforms(
        self,
        transforms: List[Transform],
        target: Vec3
    ) -> List[Transform]:
        """Solve using transforms and return modified transforms.

        Args:
            transforms: All bone transforms (world space)
            target: Target position

        Returns:
            Copy of transforms with IK chain modified.
        """
        # Extract chain positions
        positions = [transforms[idx].translation for idx in self.bone_indices]
        rotations = [transforms[idx].rotation for idx in self.bone_indices]

        # Solve
        result = self.solve(positions, target, rotations)

        # Create copy of transforms
        new_transforms = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        # Apply results
        for i, idx in enumerate(self.bone_indices):
            new_transforms[idx].translation = result.positions[i]
            new_transforms[idx].rotation = result.rotations[i]

        return new_transforms

    def reset_cached_lengths(self) -> None:
        """Reset cached bone lengths."""
        self._lengths_cached = False
        self._bone_lengths = []
        self._total_length = 0.0


class FABRIKMultiChain:
    """Multi-chain FABRIK solver for connected chains.

    Handles multiple IK chains that share joints, solving them
    together to maintain consistency.
    """

    def __init__(self) -> None:
        """Initialize the multi-chain solver."""
        self._chains: List[FABRIKChain] = []
        self._chain_targets: List[Vec3] = []
        self._shared_joints: dict[int, List[int]] = {}  # joint_idx -> chain indices

    def add_chain(self, chain: FABRIKChain, target: Vec3) -> int:
        """Add a chain to the solver.

        Args:
            chain: FABRIK chain to add
            target: Target for this chain

        Returns:
            Index of added chain.
        """
        chain_idx = len(self._chains)
        self._chains.append(chain)
        self._chain_targets.append(target)

        # Track shared joints
        for bone_idx in chain.bone_indices:
            if bone_idx not in self._shared_joints:
                self._shared_joints[bone_idx] = []
            self._shared_joints[bone_idx].append(chain_idx)

        return chain_idx

    def set_target(self, chain_idx: int, target: Vec3) -> None:
        """Update target for a chain.

        Args:
            chain_idx: Index of chain
            target: New target position
        """
        if 0 <= chain_idx < len(self._chain_targets):
            self._chain_targets[chain_idx] = target

    def solve(
        self,
        all_positions: List[Vec3],
        max_iterations: int = FABRIK_MULTI_CHAIN_MAX_ITERATIONS
    ) -> List[Vec3]:
        """Solve all chains simultaneously.

        Args:
            all_positions: Positions of all joints in skeleton
            max_iterations: Maximum iterations

        Returns:
            Modified positions.
        """
        positions = [Vec3(p.x, p.y, p.z) for p in all_positions]

        for _ in range(max_iterations):
            converged = True

            for chain_idx, chain in enumerate(self._chains):
                target = self._chain_targets[chain_idx]

                # Extract chain positions
                chain_positions = [positions[idx] for idx in chain.bone_indices]

                # Solve single chain
                result = chain.solve(chain_positions, target)

                if not result.success:
                    converged = False

                # Update shared positions (averaging)
                for i, bone_idx in enumerate(chain.bone_indices):
                    chains_using = self._shared_joints.get(bone_idx, [])
                    if len(chains_using) > 1:
                        # Average with current position
                        positions[bone_idx] = positions[bone_idx].lerp(
                            result.positions[i], 1.0 / len(chains_using)
                        )
                    else:
                        positions[bone_idx] = result.positions[i]

            if converged:
                break

        return positions
