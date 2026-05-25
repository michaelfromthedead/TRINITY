"""Foot IK for terrain adaptation.

This module implements foot placement IK for adapting character feet
to uneven terrain, including pelvis height adjustment and toe alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Callable, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON

from engine.animation.ik.config import (
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
    FOOT_PLACEMENT_MAX_PELVIS_DROP,
    FOOT_PLACEMENT_MAX_PELVIS_RAISE,
    FOOT_PLACEMENT_TOE_ALIGN_WEIGHT,
    FOOT_PLACEMENT_REACH_SAFETY_MARGIN,
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    MULTI_LEG_MAX_PELVIS_DROP,
)
from engine.animation.ik.two_bone import TwoBoneIK


# Type alias for raycast callback
# Returns (hit, position, normal) given (origin, direction, max_distance)
RaycastCallback = Callable[[Vec3, Vec3, float], Tuple[bool, Vec3, Vec3]]


class FootState(Enum):
    """State of a foot during locomotion."""

    PLANTED = auto()
    """Foot is on ground, weight-bearing."""

    LIFTING = auto()
    """Foot is lifting off ground."""

    AIRBORNE = auto()
    """Foot is in air during swing phase."""

    LANDING = auto()
    """Foot is about to contact ground."""


@dataclass
class FootData:
    """Data for a single foot.

    Attributes:
        upper_leg: Upper leg bone index (hip/thigh)
        lower_leg: Lower leg bone index (knee/shin)
        foot: Foot bone index (ankle)
        toe: Optional toe bone index
        state: Current foot state
        target_position: IK target position
        target_normal: Ground normal at foot
        blend_weight: IK blend weight (0-1)
        height_offset: Additional height offset
    """

    upper_leg: int
    lower_leg: int
    foot: int
    toe: Optional[int] = None
    state: FootState = FootState.PLANTED
    target_position: Vec3 = field(default_factory=Vec3.zero)
    target_normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    blend_weight: float = 1.0
    height_offset: float = 0.0


@dataclass
class FootPlacementResult:
    """Result from foot placement solve.

    Attributes:
        success: Whether solve completed successfully
        transforms: Modified transforms
        pelvis_offset: How much pelvis was adjusted
        left_foot_planted: Whether left foot is planted
        right_foot_planted: Whether right foot is planted
        terrain_slope: Estimated terrain slope angle
    """

    success: bool
    transforms: List[Transform] = field(default_factory=list)
    pelvis_offset: Vec3 = field(default_factory=Vec3.zero)
    left_foot_planted: bool = True
    right_foot_planted: bool = True
    terrain_slope: float = 0.0


class FootPlacement:
    """Foot IK for terrain adaptation.

    Adapts character feet to uneven terrain using raycasting and IK.
    Handles:
    - Ground height detection
    - Pelvis height adjustment
    - Foot rotation to match terrain slope
    - Toe alignment
    - Smooth transitions when terrain changes

    Attributes:
        left_foot: Left foot bone data
        right_foot: Right foot bone data
        pelvis: Pelvis bone index
        raycast: Raycast callback function
    """

    def __init__(
        self,
        left_foot: FootData,
        right_foot: FootData,
        pelvis: int,
        raycast_callback: Optional[RaycastCallback] = None
    ) -> None:
        """Initialize foot placement.

        Args:
            left_foot: Left foot configuration
            right_foot: Right foot configuration
            pelvis: Pelvis bone index
            raycast_callback: Function for ground detection
        """
        self.left_foot = left_foot
        self.right_foot = right_foot
        self.pelvis = pelvis
        self._raycast = raycast_callback

        # Create leg IK solvers
        self._left_leg_ik = TwoBoneIK(
            left_foot.upper_leg,
            left_foot.lower_leg,
            left_foot.foot,
            soft_ik_ratio=SOFT_IK_DEFAULT_RATIO,
            soft_ik_blend=SOFT_IK_DEFAULT_BLEND
        )

        self._right_leg_ik = TwoBoneIK(
            right_foot.upper_leg,
            right_foot.lower_leg,
            right_foot.foot,
            soft_ik_ratio=SOFT_IK_DEFAULT_RATIO,
            soft_ik_blend=SOFT_IK_DEFAULT_BLEND
        )

        # Configuration
        self.ray_length = FOOT_PLACEMENT_RAY_LENGTH
        self.foot_height = FOOT_PLACEMENT_FOOT_HEIGHT
        self.blend_speed = FOOT_PLACEMENT_BLEND_SPEED
        self.max_pelvis_drop = FOOT_PLACEMENT_MAX_PELVIS_DROP
        self.max_pelvis_raise = FOOT_PLACEMENT_MAX_PELVIS_RAISE
        self.toe_align_weight = FOOT_PLACEMENT_TOE_ALIGN_WEIGHT

        # State tracking
        self._prev_left_target = Vec3.zero()
        self._prev_right_target = Vec3.zero()
        self._prev_pelvis_offset = 0.0

    def set_raycast_callback(self, callback: RaycastCallback) -> None:
        """Set the raycast callback function.

        Args:
            callback: Raycast function
        """
        self._raycast = callback

    def solve(
        self,
        transforms: List[Transform],
        character_position: Vec3,
        dt: float = 1.0 / 60.0
    ) -> FootPlacementResult:
        """Solve foot placement IK.

        Args:
            transforms: Current bone transforms (world space)
            character_position: Character root position
            dt: Delta time for blending

        Returns:
            FootPlacementResult with modified transforms.
        """
        if self._raycast is None:
            return FootPlacementResult(
                success=False,
                transforms=transforms
            )

        # Copy transforms
        result_transforms = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        # Get current foot positions
        left_pos = transforms[self.left_foot.foot].translation
        right_pos = transforms[self.right_foot.foot].translation
        pelvis_pos = transforms[self.pelvis].translation

        # Raycast for both feet
        left_hit, left_ground, left_normal = self._raycast_foot(left_pos)
        right_hit, right_ground, right_normal = self._raycast_foot(right_pos)

        # Determine foot states and targets
        left_target = left_pos
        right_target = right_pos
        left_planted = False
        right_planted = False

        if left_hit:
            left_target = Vec3(
                left_ground.x,
                left_ground.y + self.foot_height + self.left_foot.height_offset,
                left_ground.z
            )
            self.left_foot.target_normal = left_normal
            left_planted = True

        if right_hit:
            right_target = Vec3(
                right_ground.x,
                right_ground.y + self.foot_height + self.right_foot.height_offset,
                right_ground.z
            )
            self.right_foot.target_normal = right_normal
            right_planted = True

        # Smooth transitions
        blend_factor = min(1.0, self.blend_speed * dt)
        left_target = self._prev_left_target.lerp(left_target, blend_factor)
        right_target = self._prev_right_target.lerp(right_target, blend_factor)

        self._prev_left_target = left_target
        self._prev_right_target = right_target

        # Calculate pelvis adjustment
        pelvis_offset = self._calculate_pelvis_offset(
            pelvis_pos,
            left_target, left_planted,
            right_target, right_planted,
            dt
        )

        # Apply pelvis offset
        pelvis_adjust = Vec3(0, pelvis_offset, 0)
        result_transforms[self.pelvis].translation = pelvis_pos + pelvis_adjust

        # Adjust targets relative to pelvis movement
        if pelvis_offset != 0:
            # Don't adjust targets - let IK handle the difference
            pass

        # Solve left leg IK
        if left_planted and self.left_foot.blend_weight > 0:
            self._solve_leg_ik(
                result_transforms,
                self._left_leg_ik,
                self.left_foot,
                left_target
            )

        # Solve right leg IK
        if right_planted and self.right_foot.blend_weight > 0:
            self._solve_leg_ik(
                result_transforms,
                self._right_leg_ik,
                self.right_foot,
                right_target
            )

        # Align feet to terrain
        if left_planted:
            self._align_foot_to_terrain(
                result_transforms,
                self.left_foot,
                self.left_foot.target_normal
            )

        if right_planted:
            self._align_foot_to_terrain(
                result_transforms,
                self.right_foot,
                self.right_foot.target_normal
            )

        # Compute terrain slope
        terrain_slope = 0.0
        if left_planted and right_planted:
            height_diff = abs(left_target.y - right_target.y)
            horiz_dist = Vec3(
                left_target.x - right_target.x,
                0,
                left_target.z - right_target.z
            ).length()
            if horiz_dist > MATH_EPSILON:
                terrain_slope = math.atan(height_diff / horiz_dist)

        return FootPlacementResult(
            success=True,
            transforms=result_transforms,
            pelvis_offset=pelvis_adjust,
            left_foot_planted=left_planted,
            right_foot_planted=right_planted,
            terrain_slope=terrain_slope
        )

    def _raycast_foot(self, foot_pos: Vec3) -> Tuple[bool, Vec3, Vec3]:
        """Raycast from foot position to find ground.

        Args:
            foot_pos: Current foot position

        Returns:
            Tuple of (hit, ground_position, normal).
        """
        # Cast ray downward from slightly above foot
        origin = Vec3(foot_pos.x, foot_pos.y + self.ray_length * 0.5, foot_pos.z)
        direction = Vec3(0, -1, 0)

        hit, position, normal = self._raycast(origin, direction, self.ray_length)

        return hit, position, normal

    def _calculate_pelvis_offset(
        self,
        pelvis_pos: Vec3,
        left_target: Vec3,
        left_planted: bool,
        right_target: Vec3,
        right_planted: bool,
        dt: float
    ) -> float:
        """Calculate required pelvis height offset.

        The pelvis needs to drop when feet are on lower ground and
        raise when on higher ground, within limits.

        Args:
            pelvis_pos: Current pelvis position
            left_target: Left foot target
            left_planted: Whether left foot is planted
            right_target: Right foot target
            right_planted: Whether right foot is planted
            dt: Delta time

        Returns:
            Pelvis Y offset.
        """
        if not left_planted and not right_planted:
            # No ground contact - return to neutral
            blend = min(1.0, self.blend_speed * dt)
            return self._prev_pelvis_offset * (1.0 - blend)

        # Calculate how far each leg needs to reach
        left_reach = 0.0
        right_reach = 0.0

        if left_planted:
            to_target = left_target - pelvis_pos
            left_reach = to_target.length() - self._left_leg_ik.max_reach * FOOT_PLACEMENT_REACH_SAFETY_MARGIN

        if right_planted:
            to_target = right_target - pelvis_pos
            right_reach = to_target.length() - self._right_leg_ik.max_reach * FOOT_PLACEMENT_REACH_SAFETY_MARGIN

        # Need to drop pelvis by the maximum overshoot
        required_drop = max(0, max(left_reach, right_reach))
        required_drop = min(required_drop, self.max_pelvis_drop)

        # Or raise if both feet are higher
        required_raise = 0.0
        if left_planted and right_planted:
            avg_target_y = (left_target.y + right_target.y) / 2
            # Estimate expected foot height when legs are relaxed (70% extension)
            LEG_RELAXED_RATIO = 0.7
            current_expected = pelvis_pos.y - self._left_leg_ik.max_reach * LEG_RELAXED_RATIO
            if avg_target_y > current_expected:
                required_raise = min(
                    avg_target_y - current_expected,
                    self.max_pelvis_raise
                )

        target_offset = -required_drop + required_raise

        # Smooth transition
        blend = min(1.0, self.blend_speed * dt)
        offset = self._prev_pelvis_offset + (target_offset - self._prev_pelvis_offset) * blend
        self._prev_pelvis_offset = offset

        return offset

    def _solve_leg_ik(
        self,
        transforms: List[Transform],
        ik_solver: TwoBoneIK,
        foot_data: FootData,
        target: Vec3
    ) -> None:
        """Solve IK for a single leg.

        Args:
            transforms: Transforms to modify
            ik_solver: Two-bone IK solver for this leg
            foot_data: Foot configuration
            target: Target foot position
        """
        # Get current transforms
        upper = transforms[foot_data.upper_leg]
        lower = transforms[foot_data.lower_leg]
        foot = transforms[foot_data.foot]

        # Solve IK
        result = ik_solver.solve(
            upper, lower, foot,
            target,
            None,  # Pole vector - could use knee hint
            None   # No rotation target
        )

        if result.success:
            # Blend based on weight
            weight = foot_data.blend_weight

            if weight >= 1.0 - MATH_EPSILON:
                transforms[foot_data.upper_leg].rotation = result.root_rotation
                transforms[foot_data.lower_leg].rotation = result.mid_rotation
                transforms[foot_data.foot].rotation = result.end_rotation
            else:
                # Slerp blend
                transforms[foot_data.upper_leg].rotation = (
                    upper.rotation.slerp(result.root_rotation, weight)
                )
                transforms[foot_data.lower_leg].rotation = (
                    lower.rotation.slerp(result.mid_rotation, weight)
                )
                transforms[foot_data.foot].rotation = (
                    foot.rotation.slerp(result.end_rotation, weight)
                )

    def _align_foot_to_terrain(
        self,
        transforms: List[Transform],
        foot_data: FootData,
        ground_normal: Vec3
    ) -> None:
        """Align foot rotation to terrain slope.

        Args:
            transforms: Transforms to modify
            foot_data: Foot configuration
            ground_normal: Ground surface normal
        """
        foot_transform = transforms[foot_data.foot]

        # Current foot up direction (Y axis in local space)
        foot_up = foot_transform.rotation.rotate_vector(Vec3(0, 1, 0))

        # Rotation to align foot up with ground normal
        alignment_rotation = self._rotation_between_vectors(foot_up, ground_normal)

        # Apply weighted alignment
        weight = self.toe_align_weight * foot_data.blend_weight
        scaled_rotation = self._scale_rotation(alignment_rotation, weight)

        foot_transform.rotation = scaled_rotation * foot_transform.rotation

        # Also align toe if present
        if foot_data.toe is not None and foot_data.toe < len(transforms):
            toe_transform = transforms[foot_data.toe]
            toe_up = toe_transform.rotation.rotate_vector(Vec3(0, 1, 0))
            toe_alignment = self._rotation_between_vectors(toe_up, ground_normal)
            toe_scaled = self._scale_rotation(toe_alignment, weight * 0.5)
            toe_transform.rotation = toe_scaled * toe_transform.rotation

    def _rotation_between_vectors(self, from_vec: Vec3, to_vec: Vec3) -> Quat:
        """Compute rotation between two vectors."""
        from_n = from_vec.normalized()
        to_n = to_vec.normalized()

        dot = from_n.dot(to_n)
        dot = max(-1.0, min(1.0, dot))

        if dot > 1.0 - MATH_EPSILON:
            return Quat.identity()

        if dot < -1.0 + MATH_EPSILON:
            axis = Vec3.unit_x().cross(from_n)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(from_n)
            return Quat.from_axis_angle(axis.normalized(), math.pi)

        axis = from_n.cross(to_n).normalized()
        angle = math.acos(dot)
        return Quat.from_axis_angle(axis, angle)

    def _scale_rotation(self, rotation: Quat, scale: float) -> Quat:
        """Scale a rotation quaternion."""
        if scale <= MATH_EPSILON:
            return Quat.identity()

        if scale >= 1.0 - MATH_EPSILON:
            return rotation

        # Use slerp from identity
        return Quat.identity().slerp(rotation, scale)


class FootPlacementAnimated:
    """Foot placement with animation curve support.

    Extends basic foot placement with support for animation-driven
    foot height and timing.
    """

    def __init__(self, base_placement: FootPlacement) -> None:
        """Initialize animated foot placement.

        Args:
            base_placement: Base foot placement solver
        """
        self.base = base_placement

        # Animation curves (time -> value)
        self._left_height_curve: Optional[Callable[[float], float]] = None
        self._right_height_curve: Optional[Callable[[float], float]] = None

        self._animation_time = 0.0

    def set_height_curves(
        self,
        left_curve: Callable[[float], float],
        right_curve: Callable[[float], float]
    ) -> None:
        """Set foot height animation curves.

        Args:
            left_curve: Function returning left foot height offset
            right_curve: Function returning right foot height offset
        """
        self._left_height_curve = left_curve
        self._right_height_curve = right_curve

    def update(self, dt: float) -> None:
        """Update animation time.

        Args:
            dt: Delta time
        """
        self._animation_time += dt

    def solve(
        self,
        transforms: List[Transform],
        character_position: Vec3,
        dt: float = 1.0 / 60.0
    ) -> FootPlacementResult:
        """Solve with animated height offsets.

        Args:
            transforms: Bone transforms
            character_position: Character position
            dt: Delta time

        Returns:
            FootPlacementResult.
        """
        # Apply animation curves to height offsets
        if self._left_height_curve:
            self.base.left_foot.height_offset = self._left_height_curve(self._animation_time)

        if self._right_height_curve:
            self.base.right_foot.height_offset = self._right_height_curve(self._animation_time)

        return self.base.solve(transforms, character_position, dt)


class MultiLegFootPlacement:
    """Foot placement for multi-legged characters.

    Handles foot placement for characters with more than two legs
    (spiders, centaurs, etc.).
    """

    def __init__(
        self,
        feet: List[FootData],
        pelvis: int,
        raycast_callback: Optional[RaycastCallback] = None
    ) -> None:
        """Initialize multi-leg foot placement.

        Args:
            feet: List of foot configurations
            pelvis: Pelvis bone index
            raycast_callback: Raycast function
        """
        self.feet = feet
        self.pelvis = pelvis
        self._raycast = raycast_callback

        # Create IK solvers for each leg
        self._leg_iks: List[TwoBoneIK] = []
        for foot in feet:
            ik = TwoBoneIK(
                foot.upper_leg,
                foot.lower_leg,
                foot.foot,
                soft_ik_ratio=SOFT_IK_DEFAULT_RATIO,
                soft_ik_blend=SOFT_IK_DEFAULT_BLEND
            )
            self._leg_iks.append(ik)

        # Configuration
        self.ray_length = FOOT_PLACEMENT_RAY_LENGTH
        self.foot_height = FOOT_PLACEMENT_FOOT_HEIGHT
        self.blend_speed = FOOT_PLACEMENT_BLEND_SPEED

    def solve(
        self,
        transforms: List[Transform],
        character_position: Vec3,
        dt: float = 1.0 / 60.0
    ) -> List[Transform]:
        """Solve foot placement for all legs.

        Args:
            transforms: Bone transforms
            character_position: Character position
            dt: Delta time

        Returns:
            Modified transforms.
        """
        if self._raycast is None:
            return transforms

        result = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        # Process each foot
        targets = []
        normals = []
        planted = []

        for foot in self.feet:
            foot_pos = transforms[foot.foot].translation

            # Raycast
            origin = Vec3(foot_pos.x, foot_pos.y + self.ray_length * 0.5, foot_pos.z)
            hit, position, normal = self._raycast(origin, Vec3(0, -1, 0), self.ray_length)

            if hit:
                target = Vec3(
                    position.x,
                    position.y + self.foot_height + foot.height_offset,
                    position.z
                )
                targets.append(target)
                normals.append(normal)
                planted.append(True)
            else:
                targets.append(foot_pos)
                normals.append(Vec3(0, 1, 0))
                planted.append(False)

        # Calculate pelvis adjustment
        pelvis_offset = self._calculate_multi_pelvis_offset(
            transforms[self.pelvis].translation,
            targets,
            planted
        )

        result[self.pelvis].translation = (
            transforms[self.pelvis].translation + Vec3(0, pelvis_offset, 0)
        )

        # Solve IK for each leg
        for i, (foot, ik) in enumerate(zip(self.feet, self._leg_iks)):
            if planted[i] and foot.blend_weight > 0:
                upper = result[foot.upper_leg]
                lower = result[foot.lower_leg]
                foot_t = result[foot.foot]

                ik_result = ik.solve(upper, lower, foot_t, targets[i])

                if ik_result.success:
                    result[foot.upper_leg].rotation = ik_result.root_rotation
                    result[foot.lower_leg].rotation = ik_result.mid_rotation
                    result[foot.foot].rotation = ik_result.end_rotation

        return result

    def _calculate_multi_pelvis_offset(
        self,
        pelvis_pos: Vec3,
        targets: List[Vec3],
        planted: List[bool]
    ) -> float:
        """Calculate pelvis offset for multiple legs.

        Args:
            pelvis_pos: Current pelvis position
            targets: Target positions for all feet
            planted: Which feet are planted

        Returns:
            Pelvis Y offset.
        """
        max_overshoot = 0.0

        for i, (target, is_planted) in enumerate(zip(targets, planted)):
            if not is_planted:
                continue

            ik = self._leg_iks[i]
            to_target = target - pelvis_pos
            overshoot = to_target.length() - ik.max_reach * FOOT_PLACEMENT_REACH_SAFETY_MARGIN

            if overshoot > max_overshoot:
                max_overshoot = overshoot

        return -min(max_overshoot, MULTI_LEG_MAX_PELVIS_DROP)
