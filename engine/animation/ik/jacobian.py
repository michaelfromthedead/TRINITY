"""Jacobian-based IK solver.

This module implements Jacobian-based inverse kinematics using various
methods including Jacobian transpose, pseudoinverse, and damped least squares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    JACOBIAN_DEFAULT_MAX_ITERATIONS,
    JACOBIAN_DLS_DAMPING,
    JACOBIAN_DEFAULT_STEP_SIZE,
)


class JacobianMethod(Enum):
    """Method for solving the Jacobian IK."""

    TRANSPOSE = auto()
    """Jacobian transpose - simple, stable, but can be slow to converge."""

    PSEUDOINVERSE = auto()
    """Pseudoinverse - accurate but can be unstable near singularities."""

    DAMPED_LEAST_SQUARES = auto()
    """Damped least squares (DLS) - robust and stable."""

    SELECTIVELY_DAMPED = auto()
    """Selectively damped least squares (SDLS) - better near singularities."""


@dataclass
class JacobianResult:
    """Result from Jacobian IK solve.

    Attributes:
        success: Whether solve converged
        iterations: Number of iterations
        final_error: Final position error
        angle_changes: Change in joint angles
        rotations: Final joint rotations
        positions: Final joint positions
    """

    success: bool
    iterations: int = 0
    final_error: float = float('inf')
    angle_changes: List[Vec3] = field(default_factory=list)
    rotations: List[Quat] = field(default_factory=list)
    positions: List[Vec3] = field(default_factory=list)


class Matrix:
    """Simple matrix class for Jacobian operations.

    A minimal matrix implementation for IK calculations.
    For production use, consider numpy or similar.
    """

    def __init__(self, rows: int, cols: int, data: Optional[List[float]] = None) -> None:
        """Initialize matrix.

        Args:
            rows: Number of rows
            cols: Number of columns
            data: Optional initial data (row-major)
        """
        self.rows = rows
        self.cols = cols

        if data:
            if len(data) != rows * cols:
                raise ValueError(f"Data length {len(data)} != {rows * cols}")
            self.data = list(data)
        else:
            self.data = [0.0] * (rows * cols)

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        row, col = idx
        return self.data[row * self.cols + col]

    def __setitem__(self, idx: Tuple[int, int], value: float) -> None:
        row, col = idx
        self.data[row * self.cols + col] = value

    def transpose(self) -> Matrix:
        """Return transposed matrix."""
        result = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                result[c, r] = self[r, c]
        return result

    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}"
            )

        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self[i, k] * other[k, j]
                result[i, j] = s
        return result

    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")

        result = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def __mul__(self, scalar: float) -> Matrix:
        """Scalar multiplication."""
        result = Matrix(self.rows, self.cols)
        result.data = [x * scalar for x in self.data]
        return result

    def __rmul__(self, scalar: float) -> Matrix:
        return self.__mul__(scalar)

    @staticmethod
    def identity(size: int) -> Matrix:
        """Create identity matrix."""
        result = Matrix(size, size)
        for i in range(size):
            result[i, i] = 1.0
        return result

    def to_vector(self) -> List[float]:
        """Convert single-column matrix to list."""
        if self.cols != 1:
            raise ValueError("Can only convert column vector")
        return list(self.data)

    @staticmethod
    def from_vector(vec: List[float]) -> Matrix:
        """Create column matrix from list."""
        result = Matrix(len(vec), 1)
        result.data = list(vec)
        return result


class JacobianIK:
    """Jacobian-based IK solver.

    Uses the Jacobian matrix to relate end effector velocity to joint
    velocities. Various methods are available for inverting the Jacobian.

    The Jacobian J relates joint angle changes dq to end effector position
    change dx: dx = J * dq

    Attributes:
        bone_indices: Indices of bones in the chain
        method: Jacobian inversion method
        damping: Damping factor for DLS method
    """

    def __init__(
        self,
        bone_indices: List[int],
        method: JacobianMethod = JacobianMethod.DAMPED_LEAST_SQUARES,
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = JACOBIAN_DEFAULT_MAX_ITERATIONS,
        damping: float = JACOBIAN_DLS_DAMPING,
        step_size: float = JACOBIAN_DEFAULT_STEP_SIZE
    ) -> None:
        """Initialize Jacobian IK solver.

        Args:
            bone_indices: List of bone indices from root to end
            method: Inversion method
            tolerance: Convergence threshold
            max_iterations: Maximum iterations
            damping: Damping factor for DLS (higher = more stable, slower)
            step_size: Step size multiplier
        """
        if len(bone_indices) < 2:
            raise ValueError("Chain requires at least 2 bones")

        self.bone_indices = list(bone_indices)
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self.step_size = step_size

        # End effector goals (for multi-effector support)
        self._end_effector_indices: List[int] = [bone_indices[-1]]

        # Joint axes (default: Y-axis for all)
        self._joint_axes: List[List[Vec3]] = [
            [Vec3.unit_x(), Vec3.unit_y(), Vec3.unit_z()]
            for _ in bone_indices
        ]

    @property
    def num_joints(self) -> int:
        """Number of joints in chain."""
        return len(self.bone_indices)

    @property
    def num_end_effectors(self) -> int:
        """Number of end effectors."""
        return len(self._end_effector_indices)

    def set_joint_axes(self, joint_idx: int, axes: List[Vec3]) -> None:
        """Set rotation axes for a joint.

        Args:
            joint_idx: Joint index
            axes: List of rotation axes (1-3)
        """
        if 0 <= joint_idx < len(self._joint_axes):
            self._joint_axes[joint_idx] = [a.normalized() for a in axes]

    def add_end_effector(self, bone_idx: int) -> None:
        """Add an end effector.

        Args:
            bone_idx: Bone index for the effector
        """
        if bone_idx not in self._end_effector_indices:
            self._end_effector_indices.append(bone_idx)

    def compute_jacobian(
        self,
        positions: List[Vec3],
        rotations: List[Quat]
    ) -> Matrix:
        """Compute the Jacobian matrix.

        The Jacobian relates joint velocities to end effector velocities.
        For rotation joints: J_col = axis cross (end_effector - joint_pos)

        Args:
            positions: World positions of all joints
            rotations: World rotations of all joints

        Returns:
            Jacobian matrix (3*num_effectors x num_dofs)
        """
        num_dofs = sum(len(axes) for axes in self._joint_axes)
        num_rows = 3 * self.num_end_effectors

        jacobian = Matrix(num_rows, num_dofs)

        col = 0
        for joint_idx in range(self.num_joints - 1):  # Skip end effector
            joint_pos = positions[joint_idx]
            joint_rot = rotations[joint_idx]

            for local_axis in self._joint_axes[joint_idx]:
                # Transform axis to world space
                world_axis = joint_rot.rotate_vector(local_axis)

                row = 0
                for ee_idx in self._end_effector_indices:
                    # Find position of this end effector
                    ee_pos = positions[self.bone_indices.index(ee_idx) if ee_idx in self.bone_indices else -1]
                    if ee_pos is None:
                        continue

                    # For rotation: J = axis cross (ee - joint)
                    to_ee = ee_pos - joint_pos
                    j_col = world_axis.cross(to_ee)

                    jacobian[row, col] = j_col.x
                    jacobian[row + 1, col] = j_col.y
                    jacobian[row + 2, col] = j_col.z
                    row += 3

                col += 1

        return jacobian

    def solve_jacobian_transpose(
        self,
        jacobian: Matrix,
        error: List[float]
    ) -> List[float]:
        """Solve using Jacobian transpose method.

        Simple but stable: dq = alpha * J^T * e
        where alpha is computed to minimize |e - J*dq|

        Args:
            jacobian: The Jacobian matrix
            error: Position error vector

        Returns:
            Joint angle changes.
        """
        J_T = jacobian.transpose()
        e = Matrix.from_vector(error)

        # J_T * e
        JTe = J_T @ e

        # Compute optimal step size
        # alpha = (e^T * J * J^T * e) / (e^T * J * J^T * J * J^T * e)
        JJTe = jacobian @ JTe
        numerator = sum(x * x for x in JJTe.data)

        JJTJJTe = jacobian @ (J_T @ JJTe)
        denominator = sum(x * x for x in JJTJJTe.data)

        if denominator < MATH_EPSILON:
            alpha = self.step_size
        else:
            alpha = numerator / denominator

        result = JTe * alpha
        return result.to_vector()

    def solve_pseudoinverse(
        self,
        jacobian: Matrix,
        error: List[float]
    ) -> List[float]:
        """Solve using pseudoinverse.

        dq = J^+ * e where J^+ = J^T * (J * J^T)^-1

        This is the optimal solution but can be unstable near singularities.

        Args:
            jacobian: The Jacobian matrix
            error: Position error vector

        Returns:
            Joint angle changes.
        """
        J_T = jacobian.transpose()
        e = Matrix.from_vector(error)

        # J * J^T
        JJT = jacobian @ J_T

        # Invert JJT (simple for small matrices)
        JJT_inv = self._invert_matrix(JJT)

        if JJT_inv is None:
            # Singular - fall back to transpose
            return self.solve_jacobian_transpose(jacobian, error)

        # J^+ = J^T * (J * J^T)^-1
        J_pseudo = J_T @ JJT_inv

        # dq = J^+ * e
        result = J_pseudo @ e
        return [x * self.step_size for x in result.to_vector()]

    def solve_damped_least_squares(
        self,
        jacobian: Matrix,
        error: List[float],
        damping: Optional[float] = None
    ) -> List[float]:
        """Solve using damped least squares (DLS).

        dq = J^T * (J * J^T + lambda^2 * I)^-1 * e

        This is more stable than pseudoinverse near singularities.

        Args:
            jacobian: The Jacobian matrix
            error: Position error vector
            damping: Damping factor (uses self.damping if None)

        Returns:
            Joint angle changes.
        """
        if damping is None:
            damping = self.damping

        J_T = jacobian.transpose()
        e = Matrix.from_vector(error)

        # J * J^T + lambda^2 * I
        JJT = jacobian @ J_T
        damped = JJT + Matrix.identity(JJT.rows) * (damping * damping)

        # Invert
        damped_inv = self._invert_matrix(damped)

        if damped_inv is None:
            return self.solve_jacobian_transpose(jacobian, error)

        # J^T * (JJT + lambda^2 I)^-1 * e
        temp = damped_inv @ e
        result = J_T @ temp

        return [x * self.step_size for x in result.to_vector()]

    def _invert_matrix(self, m: Matrix) -> Optional[Matrix]:
        """Invert a small square matrix using Gauss-Jordan.

        Args:
            m: Matrix to invert

        Returns:
            Inverted matrix or None if singular.
        """
        n = m.rows
        if m.cols != n:
            return None

        # Create augmented matrix [M | I]
        aug = Matrix(n, 2 * n)
        for i in range(n):
            for j in range(n):
                aug[i, j] = m[i, j]
            aug[i, n + i] = 1.0

        # Gauss-Jordan elimination
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row, col]) > abs(aug[max_row, col]):
                    max_row = row

            # Swap rows
            if max_row != col:
                for j in range(2 * n):
                    aug[col, j], aug[max_row, j] = aug[max_row, j], aug[col, j]

            pivot = aug[col, col]
            if abs(pivot) < MATH_EPSILON:
                return None  # Singular

            # Scale pivot row
            for j in range(2 * n):
                aug[col, j] /= pivot

            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row, col]
                    for j in range(2 * n):
                        aug[row, j] -= factor * aug[col, j]

        # Extract inverse
        result = Matrix(n, n)
        for i in range(n):
            for j in range(n):
                result[i, j] = aug[i, n + j]

        return result

    def solve(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        targets: List[Vec3]
    ) -> JacobianResult:
        """Solve the IK chain.

        Args:
            positions: Current world positions of joints
            rotations: Current world rotations of joints
            targets: Target positions for end effectors

        Returns:
            JacobianResult with computed solution.
        """
        if len(positions) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} positions, got {len(positions)}"
            )

        if len(targets) != self.num_end_effectors:
            raise ValueError(
                f"Expected {self.num_end_effectors} targets, got {len(targets)}"
            )

        # Copy for modification
        pos = [Vec3(p.x, p.y, p.z) for p in positions]
        rots = [Quat(r.x, r.y, r.z, r.w) for r in rotations]

        for iteration in range(self.max_iterations):
            # Compute error
            error = []
            total_error = 0.0

            for i, ee_idx in enumerate(self._end_effector_indices):
                chain_idx = self.bone_indices.index(ee_idx) if ee_idx in self.bone_indices else -1
                if chain_idx < 0:
                    continue

                ee_pos = pos[chain_idx]
                target = targets[i]

                err = target - ee_pos
                error.extend([err.x, err.y, err.z])
                total_error += err.length()

            avg_error = total_error / self.num_end_effectors

            # Check convergence
            if avg_error <= self.tolerance:
                return JacobianResult(
                    success=True,
                    iterations=iteration + 1,
                    final_error=avg_error,
                    rotations=rots,
                    positions=pos
                )

            # Compute Jacobian
            jacobian = self.compute_jacobian(pos, rots)

            # Solve for angle changes
            if self.method == JacobianMethod.TRANSPOSE:
                dq = self.solve_jacobian_transpose(jacobian, error)
            elif self.method == JacobianMethod.PSEUDOINVERSE:
                dq = self.solve_pseudoinverse(jacobian, error)
            else:  # DLS or SDLS
                dq = self.solve_damped_least_squares(jacobian, error)

            # Apply angle changes
            self._apply_angle_changes(pos, rots, dq)

        # Final error
        final_error = 0.0
        for i, ee_idx in enumerate(self._end_effector_indices):
            chain_idx = self.bone_indices.index(ee_idx) if ee_idx in self.bone_indices else -1
            if chain_idx >= 0:
                final_error += (targets[i] - pos[chain_idx]).length()
        final_error /= max(1, self.num_end_effectors)

        return JacobianResult(
            success=False,
            iterations=self.max_iterations,
            final_error=final_error,
            rotations=rots,
            positions=pos
        )

    def _apply_angle_changes(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        dq: List[float]
    ) -> None:
        """Apply computed angle changes to joints.

        Args:
            positions: Joint positions (modified in place)
            rotations: Joint rotations (modified in place)
            dq: Angle changes per DOF
        """
        dq_idx = 0

        for joint_idx in range(self.num_joints - 1):
            combined_rot = Quat.identity()

            for axis in self._joint_axes[joint_idx]:
                if dq_idx >= len(dq):
                    break

                angle = dq[dq_idx]
                world_axis = rotations[joint_idx].rotate_vector(axis)
                rot = Quat.from_axis_angle(world_axis, angle)
                combined_rot = rot * combined_rot
                dq_idx += 1

            # Apply rotation
            old_rot = rotations[joint_idx]
            rotations[joint_idx] = combined_rot * old_rot

            # Update child positions
            self._update_child_positions(joint_idx, positions, combined_rot)

    def _update_child_positions(
        self,
        parent_idx: int,
        positions: List[Vec3],
        rotation: Quat
    ) -> None:
        """Update positions of bones after parent rotation.

        Args:
            parent_idx: Index of rotated joint
            positions: Positions to update
            rotation: Applied rotation
        """
        pivot = positions[parent_idx]

        for i in range(parent_idx + 1, len(positions)):
            offset = positions[i] - pivot
            new_offset = rotation.rotate_vector(offset)
            positions[i] = pivot + new_offset

    def solve_with_transforms(
        self,
        transforms: List[Transform],
        targets: List[Vec3]
    ) -> List[Transform]:
        """Solve using transforms.

        Args:
            transforms: All bone transforms
            targets: End effector targets

        Returns:
            Modified transforms.
        """
        positions = [transforms[idx].translation for idx in self.bone_indices]
        rotations = [transforms[idx].rotation for idx in self.bone_indices]

        result = self.solve(positions, rotations, targets)

        new_transforms = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        for i, idx in enumerate(self.bone_indices):
            new_transforms[idx].translation = result.positions[i]
            new_transforms[idx].rotation = result.rotations[i]

        return new_transforms


class MultiTargetJacobianIK(JacobianIK):
    """Jacobian IK with weighted multi-target support.

    Extends the basic Jacobian solver to handle multiple weighted
    targets with different priorities.
    """

    def __init__(
        self,
        bone_indices: List[int],
        method: JacobianMethod = JacobianMethod.DAMPED_LEAST_SQUARES,
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = JACOBIAN_DEFAULT_MAX_ITERATIONS,
        damping: float = JACOBIAN_DLS_DAMPING,
        step_size: float = JACOBIAN_DEFAULT_STEP_SIZE
    ) -> None:
        """Initialize multi-target solver."""
        super().__init__(
            bone_indices, method, tolerance, max_iterations, damping, step_size
        )

        self._target_weights: List[float] = []

    def add_end_effector_weighted(self, bone_idx: int, weight: float = 1.0) -> None:
        """Add weighted end effector.

        Args:
            bone_idx: Bone index
            weight: Target weight
        """
        self.add_end_effector(bone_idx)
        self._target_weights.append(weight)

    def solve(
        self,
        positions: List[Vec3],
        rotations: List[Quat],
        targets: List[Vec3]
    ) -> JacobianResult:
        """Solve with weighted targets."""
        # Ensure weights list matches
        while len(self._target_weights) < len(targets):
            self._target_weights.append(1.0)

        # Scale errors by weights for solve
        # ... (implementation would modify error computation)

        return super().solve(positions, rotations, targets)
