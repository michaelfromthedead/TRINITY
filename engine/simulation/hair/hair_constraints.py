"""
Hair constraint implementations.

Includes:
- LengthConstraint: Maintains segment length (inextensibility)
- GlobalShapeConstraint: Matches rest pose
- LocalShapeConstraint: Preserves relative angles
- RootConstraint: Scalp attachment
- CollisionConstraint: Body collision response
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .config import LOCAL_SHAPE_CORRECTION_FACTOR, NUMERICAL_EPSILON

if TYPE_CHECKING:
    from .hair_simulation import GuideHair, HairControlPoint, HairStrand


def solve_length_constraint(
    cp0: HairControlPoint,
    cp1: HairControlPoint,
    rest_length: float,
    stiffness: float = 1.0,
) -> float:
    """
    Solve length constraint between two control points.

    Uses Follow-The-Leader (FTL) style: the parent point is fixed
    and only the child point moves.

    Args:
        cp0: Parent control point (toward root)
        cp1: Child control point (toward tip)
        rest_length: Target distance
        stiffness: Constraint stiffness (0-1)

    Returns:
        Constraint error
    """
    delta = cp1.position - cp0.position
    current_length = float(np.linalg.norm(delta))

    if current_length < NUMERICAL_EPSILON:
        return rest_length

    error = current_length - rest_length

    if abs(error) < NUMERICAL_EPSILON:
        return 0.0

    # FTL: Only move cp1 (child point)
    # The correction moves cp1 toward cp0 by the full error amount (scaled by stiffness)
    if cp1.inv_mass > 0:
        direction = delta / current_length  # Normalized direction from cp0 to cp1
        # Correction is the error (overshoot) scaled by stiffness
        correction = error * stiffness
        cp1.position -= direction * correction

    return error


def solve_global_shape_constraint(
    hair: HairStrand,
    head_position: NDArray[np.float32],
    head_rotation: NDArray[np.float32],
    stiffness: float = 0.5,
) -> None:
    """
    Solve global shape matching constraint.

    Pulls hair toward its rest pose in world space.

    Args:
        hair: The hair strand
        head_position: Current head position
        head_rotation: Current head rotation (3x3)
        stiffness: Constraint stiffness (0-1)
    """
    root_world = head_position + np.dot(head_rotation, hair.root_position)

    for i, cp in enumerate(hair.control_points):
        if cp.is_root:
            continue

        # Transform rest position to world space
        rest_world = root_world + np.dot(head_rotation, cp.rest_position)

        # Blend toward rest position
        delta = rest_world - cp.position
        cp.position += delta * stiffness * cp.inv_mass


def solve_local_shape_constraint(
    hair: HairStrand,
    stiffness: float = 0.3,
) -> None:
    """
    Solve local shape constraint.

    Preserves relative angles between consecutive segments.

    Args:
        hair: The hair strand
        stiffness: Constraint stiffness (0-1)
    """
    cps = hair.control_points

    if len(cps) < 3:
        return

    for i in range(1, len(cps) - 1):
        cp_prev = cps[i - 1]
        cp_curr = cps[i]
        cp_next = cps[i + 1]

        if cp_curr.inv_mass == 0 or cp_next.inv_mass == 0:
            continue

        # Current edges
        edge0 = cp_curr.position - cp_prev.position
        edge1 = cp_next.position - cp_curr.position

        edge0_len = np.linalg.norm(edge0)
        edge1_len = np.linalg.norm(edge1)

        if edge0_len < NUMERICAL_EPSILON or edge1_len < NUMERICAL_EPSILON:
            continue

        edge0_dir = edge0 / edge0_len
        edge1_dir = edge1 / edge1_len

        # Rest pose edges (approximate - assume straight)
        rest_edge0 = cp_curr.rest_position - cps[i - 1].rest_position
        rest_edge1 = cp_next.rest_position - cp_curr.rest_position

        rest_edge0_len = np.linalg.norm(rest_edge0)
        rest_edge1_len = np.linalg.norm(rest_edge1)

        if rest_edge0_len < NUMERICAL_EPSILON or rest_edge1_len < NUMERICAL_EPSILON:
            continue

        rest_edge0_dir = rest_edge0 / rest_edge0_len
        rest_edge1_dir = rest_edge1 / rest_edge1_len

        # Compute angle error
        current_cos = float(np.dot(edge0_dir, edge1_dir))
        rest_cos = float(np.dot(rest_edge0_dir, rest_edge1_dir))

        angle_diff = current_cos - rest_cos

        if abs(angle_diff) < NUMERICAL_EPSILON:
            continue

        # Compute correction (simplified - rotate edge1 toward rest angle)
        # This is an approximation; full angular constraint is more complex
        correction_axis = np.cross(edge0_dir, edge1_dir)
        correction_axis_len = np.linalg.norm(correction_axis)

        if correction_axis_len > NUMERICAL_EPSILON:
            correction_axis /= correction_axis_len

            # Rotate cp_next around cp_curr
            # Using config constant for tunable correction strength
            correction_angle = -angle_diff * stiffness * LOCAL_SHAPE_CORRECTION_FACTOR
            cos_a = math.cos(correction_angle)
            sin_a = math.sin(correction_angle)

            # Rodrigues' rotation
            edge1_rotated = (
                edge1_dir * cos_a
                + np.cross(correction_axis, edge1_dir) * sin_a
                + correction_axis * np.dot(correction_axis, edge1_dir) * (1 - cos_a)
            )

            target_pos = cp_curr.position + edge1_rotated * edge1_len
            cp_next.position += (target_pos - cp_next.position) * stiffness


@dataclass
class LengthConstraint:
    """
    Distance constraint for hair segments.

    Maintains the rest length between consecutive control points.
    """

    cp0_index: int  # Index of parent control point
    cp1_index: int  # Index of child control point
    rest_length: float
    stiffness: float = 1.0

    def solve(
        self,
        control_points: List[HairControlPoint],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the length constraint.

        Args:
            control_points: List of all control points in the strand
            stiffness_override: Optional stiffness override

        Returns:
            Constraint error
        """
        stiff = stiffness_override if stiffness_override is not None else self.stiffness
        return solve_length_constraint(
            control_points[self.cp0_index],
            control_points[self.cp1_index],
            self.rest_length,
            stiff,
        )


@dataclass
class GlobalShapeConstraint:
    """
    Global shape matching constraint.

    Pulls the entire strand toward its rest pose.
    """

    rest_positions: NDArray[np.float32]  # Rest positions for all control points
    stiffness: float = 0.5

    def solve(
        self,
        control_points: List[HairControlPoint],
        root_position: NDArray[np.float32],
        root_rotation: NDArray[np.float32],
        stiffness_override: Optional[float] = None,
    ) -> None:
        """
        Solve the global shape constraint.

        Args:
            control_points: List of all control points
            root_position: World position of root
            root_rotation: Rotation matrix at root
            stiffness_override: Optional stiffness override
        """
        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        for i, cp in enumerate(control_points):
            if cp.is_root:
                continue

            # Transform rest position to world space
            rest_world = root_position + np.dot(root_rotation, self.rest_positions[i])

            # Blend toward rest
            delta = rest_world - cp.position
            cp.position += delta * stiff * cp.inv_mass


@dataclass
class LocalShapeConstraint:
    """
    Local shape constraint preserving relative angles.

    Maintains the angle between consecutive segments.
    """

    rest_angles: NDArray[np.float32]  # Rest angles at each joint
    stiffness: float = 0.3

    def solve(
        self,
        control_points: List[HairControlPoint],
        stiffness_override: Optional[float] = None,
    ) -> None:
        """
        Solve local shape constraints.

        Args:
            control_points: List of all control points
            stiffness_override: Optional stiffness override
        """
        if len(control_points) < 3:
            return

        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        for i in range(1, len(control_points) - 1):
            if i - 1 >= len(self.rest_angles):
                break

            cp_prev = control_points[i - 1]
            cp_curr = control_points[i]
            cp_next = control_points[i + 1]

            if cp_curr.inv_mass == 0 or cp_next.inv_mass == 0:
                continue

            # Current angle
            edge0 = cp_curr.position - cp_prev.position
            edge1 = cp_next.position - cp_curr.position

            edge0_len = np.linalg.norm(edge0)
            edge1_len = np.linalg.norm(edge1)

            if edge0_len < NUMERICAL_EPSILON or edge1_len < NUMERICAL_EPSILON:
                continue

            current_cos = np.dot(edge0, edge1) / (edge0_len * edge1_len)
            current_angle = math.acos(np.clip(current_cos, -1.0, 1.0))

            # Target angle
            rest_angle = self.rest_angles[i - 1]
            angle_error = current_angle - rest_angle

            if abs(angle_error) < NUMERICAL_EPSILON:
                continue

            # Apply correction (simplified)
            # Using config constant for tunable correction strength
            correction = angle_error * stiff * LOCAL_SHAPE_CORRECTION_FACTOR
            edge1_dir = edge1 / edge1_len

            # Rotate edge1 around its cross product with edge0
            axis = np.cross(edge0 / edge0_len, edge1_dir)
            axis_len = np.linalg.norm(axis)

            if axis_len > NUMERICAL_EPSILON:
                axis /= axis_len

                cos_c = math.cos(correction)
                sin_c = math.sin(correction)

                edge1_rotated = (
                    edge1_dir * cos_c
                    + np.cross(axis, edge1_dir) * sin_c
                    + axis * np.dot(axis, edge1_dir) * (1 - cos_c)
                )

                target = cp_curr.position + edge1_rotated * edge1_len
                cp_next.position += (target - cp_next.position) * stiff


@dataclass
class RootConstraint:
    """
    Root attachment constraint.

    Fixes the root control point to the scalp position.
    """

    scalp_position: NDArray[np.float32]
    scalp_normal: NDArray[np.float32]
    stiffness: float = 1.0

    def solve(
        self,
        control_points: List[HairControlPoint],
        head_position: NDArray[np.float32],
        head_rotation: NDArray[np.float32],
    ) -> None:
        """
        Solve root constraint.

        Args:
            control_points: List of all control points
            head_position: Current head position
            head_rotation: Current head rotation matrix
        """
        if len(control_points) == 0:
            return

        root = control_points[0]

        # Transform scalp position to world space
        world_pos = head_position + np.dot(head_rotation, self.scalp_position)

        root.position[:] = world_pos
        root.prev_position[:] = world_pos
        root.velocity[:] = 0.0

    def update_scalp_position(
        self,
        position: NDArray[np.float32],
        normal: NDArray[np.float32],
    ) -> None:
        """Update the scalp attachment point."""
        self.scalp_position = position.copy()
        self.scalp_normal = normal.copy()


@dataclass
class CollisionConstraint:
    """
    Collision response constraint.

    Handles collision with the body (head, shoulders, etc.).
    """

    collision_radius: float = 0.002  # 2mm
    friction: float = 0.3
    stiffness: float = 0.8

    def solve_capsule_collision(
        self,
        cp: HairControlPoint,
        capsule_a: NDArray[np.float32],
        capsule_b: NDArray[np.float32],
        capsule_radius: float,
    ) -> bool:
        """
        Solve collision with a capsule.

        Args:
            cp: Control point to test
            capsule_a: Capsule start point
            capsule_b: Capsule end point
            capsule_radius: Capsule radius

        Returns:
            True if collision occurred
        """
        if cp.inv_mass == 0:
            return False

        # Find closest point on capsule axis
        axis = capsule_b - capsule_a
        axis_len_sq = float(np.dot(axis, axis))

        if axis_len_sq < NUMERICAL_EPSILON:
            return False

        t = np.dot(cp.position - capsule_a, axis) / axis_len_sq
        t = float(np.clip(t, 0.0, 1.0))

        closest = capsule_a + t * axis
        delta = cp.position - closest
        distance = float(np.linalg.norm(delta))

        min_dist = capsule_radius + self.collision_radius

        if distance >= min_dist:
            return False

        if distance < NUMERICAL_EPSILON:
            # Point on axis - push in arbitrary perpendicular direction
            perp = np.cross(axis, np.array([1, 0, 0]))
            if np.linalg.norm(perp) < NUMERICAL_EPSILON:
                perp = np.cross(axis, np.array([0, 1, 0]))
            normal = (perp / np.linalg.norm(perp)).astype(np.float32)
        else:
            normal = (delta / distance).astype(np.float32)

        # Push out
        penetration = min_dist - distance
        cp.position += normal * penetration * self.stiffness

        # Apply friction
        if self.friction > 0:
            velocity = cp.position - cp.prev_position
            tangent = velocity - np.dot(velocity, normal) * normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(self.friction * penetration, tangent_len)
                cp.position -= (tangent / tangent_len) * friction_force

        return True


def create_length_constraints(
    strand: HairStrand,
    stiffness: float = 1.0,
) -> List[LengthConstraint]:
    """
    Create length constraints for all segments in a strand.

    Args:
        strand: The hair strand
        stiffness: Constraint stiffness

    Returns:
        List of length constraints
    """
    constraints = []

    for i in range(len(strand.control_points) - 1):
        constraints.append(
            LengthConstraint(
                cp0_index=i,
                cp1_index=i + 1,
                rest_length=strand.rest_lengths[i],
                stiffness=stiffness,
            )
        )

    return constraints


def create_local_shape_constraints(
    strand: HairStrand,
    stiffness: float = 0.3,
) -> Optional[LocalShapeConstraint]:
    """
    Create local shape constraint for a strand.

    Args:
        strand: The hair strand
        stiffness: Constraint stiffness

    Returns:
        LocalShapeConstraint or None if strand too short
    """
    cps = strand.control_points

    if len(cps) < 3:
        return None

    # Compute rest angles
    rest_angles = []

    for i in range(1, len(cps) - 1):
        rest0 = cps[i].rest_position - cps[i - 1].rest_position
        rest1 = cps[i + 1].rest_position - cps[i].rest_position

        len0 = np.linalg.norm(rest0)
        len1 = np.linalg.norm(rest1)

        if len0 < NUMERICAL_EPSILON or len1 < NUMERICAL_EPSILON:
            rest_angles.append(0.0)
        else:
            cos_angle = np.dot(rest0, rest1) / (len0 * len1)
            rest_angles.append(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    return LocalShapeConstraint(
        rest_angles=np.array(rest_angles, dtype=np.float32),
        stiffness=stiffness,
    )
