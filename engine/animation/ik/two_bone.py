"""Analytical two-bone IK solver.

This module implements an efficient analytical solution for two-bone IK chains,
commonly used for arm and leg IK. The solver uses the law of cosines to
compute joint angles directly without iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    SOFT_IK_FALLOFF_RATE,
    JOINT_MIN_BEND_ANGLE,
    JOINT_MAX_BEND_ANGLE,
)


@dataclass
class TwoBoneIKResult:
    """Result from a two-bone IK solve operation.

    Attributes:
        success: Whether the solve succeeded
        root_rotation: Computed rotation for root bone
        mid_rotation: Computed rotation for middle bone
        end_rotation: Computed rotation for end bone (if rotation goal)
        target_reached: Whether target was reached exactly
        extension_ratio: How extended the chain is (0=folded, 1=fully extended)
    """

    success: bool
    root_rotation: Quat = field(default_factory=Quat.identity)
    mid_rotation: Quat = field(default_factory=Quat.identity)
    end_rotation: Quat = field(default_factory=Quat.identity)
    target_reached: bool = False
    extension_ratio: float = 0.0


class TwoBoneIK:
    """Analytical two-bone IK solver using law of cosines.

    This solver computes exact joint rotations for a two-bone chain
    (like an arm or leg) to reach a target position. It uses an
    analytical solution based on the law of cosines, making it very fast.

    The chain consists of:
    - Root bone (e.g., upper arm/thigh)
    - Mid bone (e.g., forearm/shin)
    - End effector (e.g., hand/foot)

    Attributes:
        root_bone: Index of the root bone in the skeleton
        mid_bone: Index of the middle bone (elbow/knee)
        end_bone: Index of the end bone (wrist/ankle)
        soft_ik_ratio: Ratio for soft IK falloff (0 = hard limits)
        soft_ik_blend: How much soft IK affects final position
    """

    def __init__(
        self,
        root_bone: int,
        mid_bone: int,
        end_bone: int,
        soft_ik_ratio: float = 0.0,
        soft_ik_blend: float = 1.0
    ) -> None:
        """Initialize the two-bone IK solver.

        Args:
            root_bone: Index of root bone (upper arm/thigh)
            mid_bone: Index of middle bone (elbow/knee)
            end_bone: Index of end bone (wrist/ankle)
            soft_ik_ratio: Start softening at this ratio of max reach (0-1)
            soft_ik_blend: Blend factor for soft IK effect (0-1)
        """
        if root_bone < 0 or mid_bone < 0 or end_bone < 0:
            raise ValueError("Bone indices must be non-negative")

        self.root_bone = root_bone
        self.mid_bone = mid_bone
        self.end_bone = end_bone
        self.soft_ik_ratio = max(0.0, min(1.0, soft_ik_ratio))
        self.soft_ik_blend = max(0.0, min(1.0, soft_ik_blend))

        # Cached bone lengths (computed on first solve)
        self._upper_length: float = 0.0
        self._lower_length: float = 0.0
        self._total_length: float = 0.0
        self._lengths_cached: bool = False

    def _cache_bone_lengths(
        self,
        root_pos: Vec3,
        mid_pos: Vec3,
        end_pos: Vec3
    ) -> None:
        """Cache bone lengths from current positions.

        Args:
            root_pos: Root bone world position
            mid_pos: Middle bone world position
            end_pos: End bone world position
        """
        self._upper_length = (mid_pos - root_pos).length()
        self._lower_length = (end_pos - mid_pos).length()
        self._total_length = self._upper_length + self._lower_length
        self._lengths_cached = True

    def _apply_soft_ik(self, target_distance: float) -> float:
        """Apply soft IK to smoothly handle reach limits.

        When the target is beyond the chain's reach, soft IK creates
        a smooth falloff instead of hard clamping.

        Args:
            target_distance: Distance to target

        Returns:
            Modified distance with soft IK applied.
        """
        if self.soft_ik_ratio <= 0.0 or self.soft_ik_blend <= 0.0:
            return target_distance

        if self._total_length < MATH_EPSILON:
            return target_distance

        soft_start = self._total_length * self.soft_ik_ratio

        if target_distance <= soft_start:
            return target_distance

        # Exponential falloff for smooth limit
        # d_soft = d_start + (d_max - d_start) * (1 - e^(-k * (d - d_start)))
        overshoot = target_distance - soft_start
        max_overshoot = self._total_length - soft_start

        if max_overshoot < MATH_EPSILON:
            return self._total_length

        # Falloff rate
        k = SOFT_IK_FALLOFF_RATE / max_overshoot
        soft_overshoot = max_overshoot * (1.0 - math.exp(-k * overshoot))

        soft_distance = soft_start + soft_overshoot

        # Blend between hard and soft
        return target_distance * (1.0 - self.soft_ik_blend) + soft_distance * self.soft_ik_blend

    def solve(
        self,
        root_transform: Transform,
        mid_transform: Transform,
        end_transform: Transform,
        target_position: Vec3,
        pole_vector: Optional[Vec3] = None,
        target_rotation: Optional[Quat] = None
    ) -> TwoBoneIKResult:
        """Solve the two-bone IK chain.

        Computes rotations for root and mid bones to place the end
        effector at the target position, with optional pole vector
        control for joint direction.

        Args:
            root_transform: World transform of root bone
            mid_transform: World transform of middle bone
            end_transform: World transform of end bone
            target_position: World position to reach
            pole_vector: Optional world position for pole (controls bend direction)
            target_rotation: Optional rotation for end effector

        Returns:
            TwoBoneIKResult with computed rotations and status.
        """
        # Get current positions
        root_pos = root_transform.translation
        mid_pos = mid_transform.translation
        end_pos = end_transform.translation

        # Cache or update bone lengths
        if not self._lengths_cached:
            self._cache_bone_lengths(root_pos, mid_pos, end_pos)

        upper_len = self._upper_length
        lower_len = self._lower_length

        # Handle degenerate cases
        if upper_len < MATH_EPSILON or lower_len < MATH_EPSILON:
            return TwoBoneIKResult(success=False)

        # Vector from root to target
        to_target = target_position - root_pos
        target_dist = to_target.length()

        if target_dist < MATH_EPSILON:
            # Target at root - fully fold the chain
            return TwoBoneIKResult(
                success=True,
                root_rotation=root_transform.rotation,
                mid_rotation=mid_transform.rotation,
                end_rotation=end_transform.rotation if target_rotation is None else target_rotation,
                target_reached=True,
                extension_ratio=0.0
            )

        # Apply soft IK
        original_dist = target_dist
        target_dist = self._apply_soft_ik(target_dist)

        # Clamp to reachable distance
        min_dist = abs(upper_len - lower_len)
        max_dist = upper_len + lower_len

        target_reached = min_dist <= original_dist <= max_dist

        target_dist = max(min_dist + MATH_EPSILON, min(target_dist, max_dist - MATH_EPSILON))
        extension_ratio = (target_dist - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 1.0

        # Direction to target
        target_dir = to_target.normalized()

        # Compute mid joint angle using law of cosines
        # cos(angle) = (a^2 + b^2 - c^2) / (2ab)
        # where a = upper_len, b = lower_len, c = target_dist
        cos_mid = (upper_len * upper_len + lower_len * lower_len - target_dist * target_dist)
        cos_mid /= (2.0 * upper_len * lower_len)
        cos_mid = max(-1.0, min(1.0, cos_mid))  # Clamp for numerical stability

        mid_angle = math.pi - math.acos(cos_mid)

        # Compute root angle
        cos_root = (upper_len * upper_len + target_dist * target_dist - lower_len * lower_len)
        cos_root /= (2.0 * upper_len * target_dist)
        cos_root = max(-1.0, min(1.0, cos_root))

        root_angle = math.acos(cos_root)

        # Compute pole plane orientation
        # Default: bend backward (negative Z in local space)
        if pole_vector is None:
            # Use current bend direction
            current_mid_dir = (mid_pos - root_pos).normalized()
            current_end_dir = (end_pos - mid_pos).normalized()

            # Cross of current chain direction gives bend normal
            chain_normal = current_mid_dir.cross(current_end_dir)
            if chain_normal.length_squared() < MATH_EPSILON:
                # Chain is straight - use a default
                chain_normal = target_dir.cross(Vec3.up())
                if chain_normal.length_squared() < MATH_EPSILON:
                    chain_normal = Vec3.right()
            chain_normal = chain_normal.normalized()
        else:
            # Use pole vector to define bend plane
            to_pole = pole_vector - root_pos
            # Project to plane perpendicular to target direction
            pole_on_plane = to_pole - target_dir * to_pole.dot(target_dir)

            if pole_on_plane.length_squared() < MATH_EPSILON:
                chain_normal = target_dir.cross(Vec3.up())
                if chain_normal.length_squared() < MATH_EPSILON:
                    chain_normal = Vec3.right()
                chain_normal = chain_normal.normalized()
            else:
                # Normal to the bend plane
                chain_normal = target_dir.cross(pole_on_plane.normalized())
                chain_normal = chain_normal.normalized()

        # Build rotation for root bone
        # First rotate to point at target
        current_to_mid = (mid_pos - root_pos).normalized()

        # Desired direction after rotation (accounting for root angle)
        # Rotate target_dir by root_angle around the bend normal
        root_rot = Quat.from_axis_angle(chain_normal, -root_angle)
        desired_mid_dir = root_rot.rotate_vector(target_dir)

        # Rotation from current to desired
        root_rotation = self._rotation_between_vectors(current_to_mid, desired_mid_dir)
        final_root_rotation = root_rotation * root_transform.rotation

        # Compute world position of mid after root rotation
        new_mid_pos = root_pos + final_root_rotation.rotate_vector(
            root_transform.rotation.inverse().rotate_vector(mid_pos - root_pos)
        )

        # Build rotation for mid bone
        current_to_end = (end_pos - mid_pos).normalized()

        # Desired direction: toward target
        # The mid bone needs to rotate by (pi - mid_angle) around its local axis
        desired_end_dir = (target_position - new_mid_pos).normalized()

        mid_rotation = self._rotation_between_vectors(current_to_end, desired_end_dir)
        final_mid_rotation = mid_rotation * mid_transform.rotation

        # End rotation
        final_end_rotation = target_rotation if target_rotation else end_transform.rotation

        return TwoBoneIKResult(
            success=True,
            root_rotation=final_root_rotation,
            mid_rotation=final_mid_rotation,
            end_rotation=final_end_rotation,
            target_reached=target_reached,
            extension_ratio=extension_ratio
        )

    def _rotation_between_vectors(self, from_vec: Vec3, to_vec: Vec3) -> Quat:
        """Compute rotation quaternion between two vectors.

        Args:
            from_vec: Starting direction (normalized)
            to_vec: Target direction (normalized)

        Returns:
            Quaternion rotating from_vec to to_vec.
        """
        dot = from_vec.dot(to_vec)

        if dot > 1.0 - MATH_EPSILON:
            # Vectors are parallel
            return Quat.identity()

        if dot < -1.0 + MATH_EPSILON:
            # Vectors are opposite
            # Find perpendicular axis
            axis = Vec3.unit_x().cross(from_vec)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(from_vec)
            axis = axis.normalized()
            return Quat.from_axis_angle(axis, math.pi)

        axis = from_vec.cross(to_vec).normalized()
        angle = math.acos(max(-1.0, min(1.0, dot)))

        return Quat.from_axis_angle(axis, angle)

    def solve_with_pose(
        self,
        bone_transforms: list[Transform],
        target_position: Vec3,
        pole_vector: Optional[Vec3] = None,
        target_rotation: Optional[Quat] = None
    ) -> list[Transform]:
        """Solve and return modified transforms.

        Convenience method that takes a list of all bone transforms
        and returns modified transforms after IK.

        Args:
            bone_transforms: List of all bone transforms (world space)
            target_position: World position to reach
            pole_vector: Optional pole vector position
            target_rotation: Optional end effector rotation

        Returns:
            Copy of transforms with IK bones modified.
        """
        if len(bone_transforms) <= max(self.root_bone, self.mid_bone, self.end_bone):
            raise ValueError("Not enough bones in transform list")

        result = self.solve(
            bone_transforms[self.root_bone],
            bone_transforms[self.mid_bone],
            bone_transforms[self.end_bone],
            target_position,
            pole_vector,
            target_rotation
        )

        if not result.success:
            return bone_transforms

        # Create copy with modified transforms
        new_transforms = [Transform(t.translation, t.rotation, t.scale) for t in bone_transforms]

        new_transforms[self.root_bone].rotation = result.root_rotation
        new_transforms[self.mid_bone].rotation = result.mid_rotation
        new_transforms[self.end_bone].rotation = result.end_rotation

        # Recompute positions based on new rotations
        self._update_chain_positions(new_transforms)

        return new_transforms

    def _update_chain_positions(self, transforms: list[Transform]) -> None:
        """Update positions of chain bones after rotation changes.

        Args:
            transforms: List of transforms to update in place.
        """
        # Update mid position based on root rotation
        root = transforms[self.root_bone]
        mid = transforms[self.mid_bone]
        end = transforms[self.end_bone]

        # Compute new mid position
        local_offset = Vec3(0, self._upper_length, 0)  # Assumes Y-up bone axis
        mid.translation = root.translation + root.rotation.rotate_vector(local_offset)

        # Compute new end position
        local_offset = Vec3(0, self._lower_length, 0)
        end.translation = mid.translation + mid.rotation.rotate_vector(local_offset)

    def reset_cached_lengths(self) -> None:
        """Reset cached bone lengths.

        Call this when bone lengths may have changed (e.g., scaling).
        """
        self._lengths_cached = False
        self._upper_length = 0.0
        self._lower_length = 0.0
        self._total_length = 0.0

    @property
    def max_reach(self) -> float:
        """Get maximum reach distance of the chain."""
        return self._total_length

    @property
    def min_reach(self) -> float:
        """Get minimum reach distance (fully bent)."""
        return abs(self._upper_length - self._lower_length)


class TwoBoneIKConstraint:
    """Constraint wrapper for two-bone IK.

    Provides additional constraint options like angle limits
    and twist constraints for the two-bone chain.
    """

    def __init__(
        self,
        solver: TwoBoneIK,
        min_bend_angle: float = JOINT_MIN_BEND_ANGLE,
        max_bend_angle: float = JOINT_MAX_BEND_ANGLE,
        twist_axis: Vec3 = Vec3.unit_y(),
        min_twist: float = -math.pi,
        max_twist: float = math.pi
    ) -> None:
        """Initialize the constraint.

        Args:
            solver: The TwoBoneIK solver to wrap
            min_bend_angle: Minimum bend angle (radians)
            max_bend_angle: Maximum bend angle (radians)
            twist_axis: Axis for twist rotation
            min_twist: Minimum twist angle (radians)
            max_twist: Maximum twist angle (radians)
        """
        self.solver = solver
        self.min_bend_angle = min_bend_angle
        self.max_bend_angle = max_bend_angle
        self.twist_axis = twist_axis.normalized()
        self.min_twist = min_twist
        self.max_twist = max_twist

    def apply_constraints(self, result: TwoBoneIKResult) -> TwoBoneIKResult:
        """Apply constraints to IK result.

        Args:
            result: Raw IK result

        Returns:
            Constrained IK result.
        """
        if not result.success:
            return result

        # Compute actual bend angle from extension ratio
        # extension_ratio of 1.0 = straight, 0.0 = fully bent
        bend_angle = math.pi * (1.0 - result.extension_ratio)

        # Clamp bend angle
        clamped_bend = max(self.min_bend_angle, min(bend_angle, self.max_bend_angle))

        if abs(clamped_bend - bend_angle) > MATH_EPSILON:
            # Would need to recompute - for now just return original
            # A full implementation would re-solve with constrained angle
            pass

        return result
