"""Cyclic Coordinate Descent (CCD) IK solver.

CCD is an iterative IK solver that works by adjusting each joint in turn,
starting from the end effector and working toward the root. Each joint
is rotated to minimize the distance from the end effector to the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Callable

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    CCD_DEFAULT_MAX_ITERATIONS,
    CCD_DEFAULT_DAMPING,
)


class CCDRotationOrder(Enum):
    """Order of joint rotation during CCD iteration."""

    END_TO_ROOT = auto()
    """Standard CCD: start from end effector, work toward root."""

    ROOT_TO_END = auto()
    """Reverse CCD: start from root, work toward end."""

    ALTERNATING = auto()
    """Alternate between end-to-root and root-to-end passes."""


@dataclass
class RotationLimit:
    """Rotation limits for a joint.

    Attributes:
        enabled: Whether limits are active
        axis: Axis for single-axis rotation (hinge joints)
        min_angles: Minimum rotation per axis (radians)
        max_angles: Maximum rotation per axis (radians)
        is_hinge: If True, only rotate around specified axis
    """

    enabled: bool = False
    axis: Vec3 = field(default_factory=lambda: Vec3.unit_y())
    min_angles: Vec3 = field(default_factory=lambda: Vec3(-math.pi, -math.pi, -math.pi))
    max_angles: Vec3 = field(default_factory=lambda: Vec3(math.pi, math.pi, math.pi))
    is_hinge: bool = False

    def clamp_rotation(self, rotation: Quat) -> Quat:
        """Clamp rotation to within limits.

        Args:
            rotation: Input rotation

        Returns:
            Clamped rotation.
        """
        if not self.enabled:
            return rotation

        if self.is_hinge:
            return self._clamp_hinge(rotation)

        return self._clamp_euler(rotation)

    def _clamp_hinge(self, rotation: Quat) -> Quat:
        """Clamp to hinge rotation.

        Args:
            rotation: Input rotation

        Returns:
            Rotation constrained to hinge axis.
        """
        # Project rotation onto hinge axis
        axis = self.axis.normalized()

        # Get rotation axis and angle
        # For a quaternion q = (sin(a/2)*axis, cos(a/2))
        quat_axis = Vec3(rotation.x, rotation.y, rotation.z)
        sin_half = quat_axis.length()

        if sin_half < MATH_EPSILON:
            return Quat.identity()

        cos_half = rotation.w
        angle = 2.0 * math.atan2(sin_half, cos_half)

        # Project axis
        rot_axis = quat_axis.normalized()
        projected_axis = axis * rot_axis.dot(axis)

        if projected_axis.length_squared() < MATH_EPSILON:
            return Quat.identity()

        # Sign of angle depends on axis alignment
        if projected_axis.dot(axis) < 0:
            angle = -angle
            projected_axis = -projected_axis

        # Clamp angle
        angle = max(self.min_angles.y, min(angle, self.max_angles.y))

        return Quat.from_axis_angle(axis, angle)

    def _clamp_euler(self, rotation: Quat) -> Quat:
        """Clamp using Euler angle limits.

        Args:
            rotation: Input rotation

        Returns:
            Euler-clamped rotation.
        """
        pitch, yaw, roll = rotation.to_euler()

        pitch = max(self.min_angles.x, min(pitch, self.max_angles.x))
        yaw = max(self.min_angles.y, min(yaw, self.max_angles.y))
        roll = max(self.min_angles.z, min(roll, self.max_angles.z))

        return Quat.from_euler(pitch, yaw, roll)


@dataclass
class CCDResult:
    """Result from CCD solve operation.

    Attributes:
        success: Whether solve converged
        iterations: Number of iterations used
        final_error: Distance from end effector to target
        rotations: Computed rotations for each joint
        positions: Final joint positions
    """

    success: bool
    iterations: int = 0
    final_error: float = float('inf')
    rotations: List[Quat] = field(default_factory=list)
    positions: List[Vec3] = field(default_factory=list)


class CCDSolver:
    """Cyclic Coordinate Descent IK solver.

    CCD iteratively rotates each joint to minimize the distance from
    the end effector to the target. It's simple, robust, and handles
    joint limits well.

    Advantages:
    - Simple and fast
    - Naturally handles joint limits
    - Stable convergence

    Disadvantages:
    - Can produce unnatural poses
    - May not find optimal solution
    - End effector often overreaches

    Attributes:
        bone_indices: Indices of bones in the chain
        rotation_limits: Per-joint rotation limits
        damping: Damping factor for stability (0-1)
    """

    def __init__(
        self,
        bone_indices: List[int],
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = CCD_DEFAULT_MAX_ITERATIONS,
        damping: float = CCD_DEFAULT_DAMPING
    ) -> None:
        """Initialize the CCD solver.

        Args:
            bone_indices: List of bone indices from root to end effector
            tolerance: Convergence distance threshold
            max_iterations: Maximum iterations
            damping: Damping factor (0-1, lower = more damping)

        Raises:
            ValueError: If chain has fewer than 2 bones
        """
        if len(bone_indices) < 2:
            raise ValueError("CCD chain requires at least 2 bones")

        if damping <= 0 or damping > 1:
            raise ValueError("Damping must be in range (0, 1]")

        self.bone_indices = list(bone_indices)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.damping = damping

        # Per-joint rotation limits
        self._rotation_limits: List[RotationLimit] = [
            RotationLimit() for _ in bone_indices
        ]

        # Rotation order
        self._rotation_order = CCDRotationOrder.END_TO_ROOT

        # Cached data
        self._bone_lengths: List[float] = []
        self._lengths_cached: bool = False

    @property
    def chain_length(self) -> int:
        """Number of joints in chain."""
        return len(self.bone_indices)

    def set_rotation_limit(self, joint_index: int, limit: RotationLimit) -> None:
        """Set rotation limit for a joint.

        Args:
            joint_index: Index within chain (0 = root)
            limit: Rotation limit to apply
        """
        if 0 <= joint_index < len(self._rotation_limits):
            self._rotation_limits[joint_index] = limit

    def set_rotation_order(self, order: CCDRotationOrder) -> None:
        """Set the order of joint rotation.

        Args:
            order: Rotation order enum
        """
        self._rotation_order = order

    def _cache_bone_lengths(self, positions: List[Vec3]) -> None:
        """Cache bone lengths from positions."""
        self._bone_lengths = []
        for i in range(len(positions) - 1):
            length = (positions[i + 1] - positions[i]).length()
            self._bone_lengths.append(length)
        self._lengths_cached = True

    def solve(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> CCDResult:
        """Solve the CCD chain.

        Args:
            positions: Current world positions of all joints
            rotations: Current world rotations of all joints
            target: Target position for end effector

        Returns:
            CCDResult with computed rotations.
        """
        if len(positions) != self.chain_length:
            raise ValueError(
                f"Expected {self.chain_length} positions, got {len(positions)}"
            )

        # Cache bone lengths
        if not self._lengths_cached:
            self._cache_bone_lengths(positions)

        # Copy for modification
        pos = [Vec3(p.x, p.y, p.z) for p in positions]
        rots = [Quat(r.x, r.y, r.z, r.w) for r in rotations]

        for iteration in range(self.max_iterations):
            # Check convergence
            end_pos = pos[-1]
            error = (target - end_pos).length()

            if error <= self.tolerance:
                return CCDResult(
                    success=True,
                    iterations=iteration + 1,
                    final_error=error,
                    rotations=rots,
                    positions=pos
                )

            # Perform CCD iteration
            if self._rotation_order == CCDRotationOrder.END_TO_ROOT:
                self._iterate_end_to_root(pos, rots, target)
            elif self._rotation_order == CCDRotationOrder.ROOT_TO_END:
                self._iterate_root_to_end(pos, rots, target)
            else:  # ALTERNATING
                if iteration % 2 == 0:
                    self._iterate_end_to_root(pos, rots, target)
                else:
                    self._iterate_root_to_end(pos, rots, target)

        # Final error
        final_error = (target - pos[-1]).length()

        return CCDResult(
            success=final_error <= self.tolerance,
            iterations=self.max_iterations,
            final_error=final_error,
            rotations=rots,
            positions=pos
        )

    def _iterate_end_to_root(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> None:
        """Perform one CCD iteration from end to root.

        Args:
            positions: Joint positions (modified in place)
            rotations: Joint rotations (modified in place)
            target: Target position
        """
        # Iterate from second-to-last joint to root
        # (Don't rotate end effector - it has no children)
        for i in range(len(positions) - 2, -1, -1):
            self._rotate_joint(i, positions, rotations, target)

    def _iterate_root_to_end(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> None:
        """Perform one CCD iteration from root to end.

        Args:
            positions: Joint positions (modified in place)
            rotations: Joint rotations (modified in place)
            target: Target position
        """
        for i in range(len(positions) - 1):
            self._rotate_joint(i, positions, rotations, target)

    def _rotate_joint(
        self,
        joint_idx: int,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> None:
        """Rotate a single joint toward the target.

        Args:
            joint_idx: Index of joint to rotate
            positions: All joint positions
            rotations: All joint rotations
            target: Target position
        """
        joint_pos = positions[joint_idx]
        end_pos = positions[-1]

        # Vector from joint to end effector
        to_end = end_pos - joint_pos
        to_end_len = to_end.length()

        if to_end_len < MATH_EPSILON:
            return

        to_end = to_end / to_end_len

        # Vector from joint to target
        to_target = target - joint_pos
        to_target_len = to_target.length()

        if to_target_len < MATH_EPSILON:
            return

        to_target = to_target / to_target_len

        # Compute rotation from current to desired direction
        dot = to_end.dot(to_target)
        dot = max(-1.0, min(1.0, dot))

        if dot > 1.0 - MATH_EPSILON:
            # Already aligned
            return

        # Rotation axis
        axis = to_end.cross(to_target)
        if axis.length_squared() < MATH_EPSILON:
            # Parallel - use perpendicular axis
            axis = Vec3.unit_x().cross(to_end)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(to_end)
        axis = axis.normalized()

        # Rotation angle
        angle = math.acos(dot) * self.damping

        # Create rotation
        rotation = Quat.from_axis_angle(axis, angle)

        # Apply rotation limits
        combined = rotation * rotations[joint_idx]
        limit = self._rotation_limits[joint_idx]
        combined = limit.clamp_rotation(combined)

        # Compute the actual rotation applied
        actual_rotation = combined * rotations[joint_idx].inverse()
        rotations[joint_idx] = combined

        # Update positions of all joints after this one
        self._update_chain_positions(joint_idx, positions, rotations, actual_rotation)

    def _update_chain_positions(
        self,
        from_idx: int,
        positions: List[Vec3],
        rotations: List[Quat],
        rotation: Quat
    ) -> None:
        """Update positions after rotating a joint.

        Args:
            from_idx: Index of rotated joint
            positions: Positions to update
            rotations: Current rotations
            rotation: Rotation that was applied
        """
        pivot = positions[from_idx]

        for i in range(from_idx + 1, len(positions)):
            # Rotate position around pivot
            offset = positions[i] - pivot
            new_offset = rotation.rotate_vector(offset)
            positions[i] = pivot + new_offset

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
        # Extract chain data
        positions = [transforms[idx].translation for idx in self.bone_indices]
        rotations = [transforms[idx].rotation for idx in self.bone_indices]

        # Solve
        result = self.solve(positions, rotations, target)

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


class CCDSolverWithWeights(CCDSolver):
    """CCD solver with per-joint weights.

    Extends the basic CCD solver to support different weights
    for each joint, controlling how much each joint contributes
    to the solution.
    """

    def __init__(
        self,
        bone_indices: List[int],
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = CCD_DEFAULT_MAX_ITERATIONS,
        damping: float = CCD_DEFAULT_DAMPING,
        weights: Optional[List[float]] = None
    ) -> None:
        """Initialize weighted CCD solver.

        Args:
            bone_indices: Chain bone indices
            tolerance: Convergence threshold
            max_iterations: Maximum iterations
            damping: Base damping factor
            weights: Per-joint weights (0-1)
        """
        super().__init__(bone_indices, tolerance, max_iterations, damping)

        if weights is None:
            self._weights = [1.0] * len(bone_indices)
        else:
            if len(weights) != len(bone_indices):
                raise ValueError("Weights must match number of joints")
            self._weights = [max(0.0, min(1.0, w)) for w in weights]

    def set_weight(self, joint_idx: int, weight: float) -> None:
        """Set weight for a specific joint.

        Args:
            joint_idx: Index within chain
            weight: Weight value (0-1)
        """
        if 0 <= joint_idx < len(self._weights):
            self._weights[joint_idx] = max(0.0, min(1.0, weight))

    def _rotate_joint(
        self,
        joint_idx: int,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> None:
        """Rotate joint with weight applied.

        Args:
            joint_idx: Joint index
            positions: All positions
            rotations: All rotations
            target: Target position
        """
        weight = self._weights[joint_idx]

        if weight < MATH_EPSILON:
            return

        joint_pos = positions[joint_idx]
        end_pos = positions[-1]

        to_end_raw = end_pos - joint_pos
        to_target_raw = target - joint_pos

        # Check for zero-length vectors before normalizing
        if to_end_raw.length_squared() < MATH_EPSILON or to_target_raw.length_squared() < MATH_EPSILON:
            return

        to_end = to_end_raw.normalized()
        to_target = to_target_raw.normalized()

        dot = max(-1.0, min(1.0, to_end.dot(to_target)))

        if dot > 1.0 - MATH_EPSILON:
            return

        axis = to_end.cross(to_target)
        if axis.length_squared() < MATH_EPSILON:
            axis = Vec3.unit_x().cross(to_end)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(to_end)
        axis = axis.normalized()

        # Apply weight to angle
        angle = math.acos(dot) * self.damping * weight

        rotation = Quat.from_axis_angle(axis, angle)
        combined = rotation * rotations[joint_idx]

        limit = self._rotation_limits[joint_idx]
        combined = limit.clamp_rotation(combined)

        actual_rotation = combined * rotations[joint_idx].inverse()
        rotations[joint_idx] = combined

        self._update_chain_positions(joint_idx, positions, rotations, actual_rotation)


class ConstrainedCCDSolver(CCDSolver):
    """CCD solver with advanced constraint support.

    Provides additional constraint types beyond basic rotation limits,
    including custom constraint functions and priority-based solving.
    """

    ConstraintFunc = Callable[[Quat, int], Quat]

    def __init__(
        self,
        bone_indices: List[int],
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = CCD_DEFAULT_MAX_ITERATIONS,
        damping: float = CCD_DEFAULT_DAMPING
    ) -> None:
        """Initialize constrained CCD solver."""
        super().__init__(bone_indices, tolerance, max_iterations, damping)

        self._custom_constraints: List[Optional[ConstrainedCCDSolver.ConstraintFunc]] = [
            None for _ in bone_indices
        ]

    def set_custom_constraint(
        self,
        joint_idx: int,
        constraint_func: ConstraintFunc
    ) -> None:
        """Set a custom constraint function for a joint.

        Args:
            joint_idx: Joint index
            constraint_func: Function taking (rotation, joint_idx) -> constrained rotation
        """
        if 0 <= joint_idx < len(self._custom_constraints):
            self._custom_constraints[joint_idx] = constraint_func

    def _apply_constraints(self, joint_idx: int, rotation: Quat) -> Quat:
        """Apply all constraints to a rotation.

        Args:
            joint_idx: Joint index
            rotation: Input rotation

        Returns:
            Constrained rotation.
        """
        # Apply standard limit
        rotation = self._rotation_limits[joint_idx].clamp_rotation(rotation)

        # Apply custom constraint
        custom = self._custom_constraints[joint_idx]
        if custom is not None:
            rotation = custom(rotation, joint_idx)

        return rotation

    def _rotate_joint(
        self,
        joint_idx: int,
        positions: List[Vec3],
        rotations: List[Quat],
        target: Vec3
    ) -> None:
        """Rotate joint with custom constraints.

        Args:
            joint_idx: Joint index
            positions: All positions
            rotations: All rotations
            target: Target position
        """
        joint_pos = positions[joint_idx]
        end_pos = positions[-1]

        to_end = end_pos - joint_pos
        to_end_len = to_end.length()
        if to_end_len < MATH_EPSILON:
            return
        to_end = to_end / to_end_len

        to_target = target - joint_pos
        to_target_len = to_target.length()
        if to_target_len < MATH_EPSILON:
            return
        to_target = to_target / to_target_len

        dot = max(-1.0, min(1.0, to_end.dot(to_target)))
        if dot > 1.0 - MATH_EPSILON:
            return

        axis = to_end.cross(to_target)
        if axis.length_squared() < MATH_EPSILON:
            axis = Vec3.unit_x().cross(to_end)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(to_end)
        axis = axis.normalized()

        angle = math.acos(dot) * self.damping
        rotation = Quat.from_axis_angle(axis, angle)

        combined = rotation * rotations[joint_idx]
        combined = self._apply_constraints(joint_idx, combined)

        actual_rotation = combined * rotations[joint_idx].inverse()
        rotations[joint_idx] = combined

        self._update_chain_positions(joint_idx, positions, rotations, actual_rotation)
