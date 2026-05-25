"""Full body IK solver.

This module implements a full body inverse kinematics solver that handles
multiple end effectors simultaneously while maintaining balance and
natural poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Callable, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON

from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
    FULLBODY_SPINE_STIFFNESS,
    POLYGON_EDGE_MIN_LENGTH,
    LOOK_AT_MAX_ANGLE,
    LOOK_AT_HEAD_WEIGHT,
    LOOK_AT_NECK_WEIGHT,
    LOOK_AT_SPINE_WEIGHT,
)
from engine.animation.ik.ik_goal import (
    IKGoal, IKGoalType, PositionGoal, RotationGoal,
    LookAtGoal, PositionRotationGoal, ChainGoal
)
from engine.animation.ik.fabrik import FABRIKChain
from engine.animation.ik.two_bone import TwoBoneIK


class BodyPart(Enum):
    """Standard body parts for full body IK."""

    PELVIS = auto()
    SPINE = auto()
    CHEST = auto()
    NECK = auto()
    HEAD = auto()

    LEFT_SHOULDER = auto()
    LEFT_UPPER_ARM = auto()
    LEFT_LOWER_ARM = auto()
    LEFT_HAND = auto()

    RIGHT_SHOULDER = auto()
    RIGHT_UPPER_ARM = auto()
    RIGHT_LOWER_ARM = auto()
    RIGHT_HAND = auto()

    LEFT_UPPER_LEG = auto()
    LEFT_LOWER_LEG = auto()
    LEFT_FOOT = auto()
    LEFT_TOE = auto()

    RIGHT_UPPER_LEG = auto()
    RIGHT_LOWER_LEG = auto()
    RIGHT_FOOT = auto()
    RIGHT_TOE = auto()


@dataclass
class SkeletonMapping:
    """Mapping of body parts to bone indices.

    Attributes:
        bone_map: Dictionary mapping BodyPart to bone index
        spine_chain: List of spine bone indices (pelvis to head)
        left_arm_chain: Left arm bone indices
        right_arm_chain: Right arm bone indices
        left_leg_chain: Left leg bone indices
        right_leg_chain: Right leg bone indices
    """

    bone_map: Dict[BodyPart, int] = field(default_factory=dict)
    spine_chain: List[int] = field(default_factory=list)
    left_arm_chain: List[int] = field(default_factory=list)
    right_arm_chain: List[int] = field(default_factory=list)
    left_leg_chain: List[int] = field(default_factory=list)
    right_leg_chain: List[int] = field(default_factory=list)

    def get_bone(self, part: BodyPart) -> int:
        """Get bone index for a body part.

        Args:
            part: Body part enum

        Returns:
            Bone index or -1 if not mapped.
        """
        return self.bone_map.get(part, -1)

    def set_bone(self, part: BodyPart, bone_idx: int) -> None:
        """Set bone index for a body part.

        Args:
            part: Body part
            bone_idx: Bone index
        """
        self.bone_map[part] = bone_idx


@dataclass
class FullBodyIKGoal:
    """Goal for full body IK solver.

    Extends the basic IKGoal with full body specific options.

    Attributes:
        bone_index: Target bone index
        target_position: World space position target
        target_rotation: Optional rotation target
        position_weight: Weight for position component
        rotation_weight: Weight for rotation component
        priority: Goal priority (higher = more important)
        enabled: Whether goal is active
        chain_type: Which limb chain this affects
    """

    bone_index: int
    target_position: Optional[Vec3] = None
    target_rotation: Optional[Quat] = None
    position_weight: float = 1.0
    rotation_weight: float = 0.0
    priority: int = 0
    enabled: bool = True
    chain_type: Optional[str] = None  # "left_arm", "right_arm", "left_leg", "right_leg", "spine"

    def has_position(self) -> bool:
        """Check if goal has position target."""
        return self.target_position is not None and self.position_weight > 0

    def has_rotation(self) -> bool:
        """Check if goal has rotation target."""
        return self.target_rotation is not None and self.rotation_weight > 0


@dataclass
class FullBodyIKResult:
    """Result from full body IK solve.

    Attributes:
        success: Whether solve completed
        transforms: Modified transforms
        goals_achieved: Which goals were achieved
        final_errors: Position errors per goal
        pelvis_adjustment: How much pelvis was moved
    """

    success: bool
    transforms: List[Transform] = field(default_factory=list)
    goals_achieved: Dict[int, bool] = field(default_factory=dict)
    final_errors: Dict[int, float] = field(default_factory=dict)
    pelvis_adjustment: Vec3 = field(default_factory=Vec3.zero)


class FullBodyIK:
    """Full body inverse kinematics solver.

    Solves IK for multiple end effectors while maintaining balance
    and natural-looking poses. Uses a combination of analytical
    and iterative methods.

    The solver processes goals in priority order and uses:
    - Two-bone IK for arms and legs
    - FABRIK for spine
    - Balance/COM maintenance
    - Pelvis height adjustment

    Attributes:
        skeleton: Skeleton mapping for body parts
        maintain_balance: Whether to maintain center of mass
        pelvis_height_adjust: Whether to adjust pelvis height
    """

    def __init__(
        self,
        skeleton_mapping: SkeletonMapping,
        tolerance: float = IK_DEFAULT_TOLERANCE,
        max_iterations: int = FULLBODY_DEFAULT_MAX_ITERATIONS
    ) -> None:
        """Initialize full body IK solver.

        Args:
            skeleton_mapping: Mapping of body parts to bones
            tolerance: Convergence threshold
            max_iterations: Maximum iterations
        """
        self.skeleton = skeleton_mapping
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.maintain_balance = True
        self.pelvis_height_adjust = True
        self.spine_stiffness = FULLBODY_SPINE_STIFFNESS

        # Arm and leg solvers
        self._left_arm_ik: Optional[TwoBoneIK] = None
        self._right_arm_ik: Optional[TwoBoneIK] = None
        self._left_leg_ik: Optional[TwoBoneIK] = None
        self._right_leg_ik: Optional[TwoBoneIK] = None

        # Spine solver
        self._spine_ik: Optional[FABRIKChain] = None

        # Initialize solvers from skeleton
        self._initialize_solvers()

        # COM tracking
        self._bone_masses: Dict[int, float] = {}
        self._support_polygon: List[Vec3] = []

    def _initialize_solvers(self) -> None:
        """Initialize limb solvers from skeleton mapping."""
        # Left arm
        if self.skeleton.left_arm_chain and len(self.skeleton.left_arm_chain) >= 3:
            chain = self.skeleton.left_arm_chain
            self._left_arm_ik = TwoBoneIK(chain[0], chain[1], chain[2])

        # Right arm
        if self.skeleton.right_arm_chain and len(self.skeleton.right_arm_chain) >= 3:
            chain = self.skeleton.right_arm_chain
            self._right_arm_ik = TwoBoneIK(chain[0], chain[1], chain[2])

        # Left leg
        if self.skeleton.left_leg_chain and len(self.skeleton.left_leg_chain) >= 3:
            chain = self.skeleton.left_leg_chain
            self._left_leg_ik = TwoBoneIK(chain[0], chain[1], chain[2])

        # Right leg
        if self.skeleton.right_leg_chain and len(self.skeleton.right_leg_chain) >= 3:
            chain = self.skeleton.right_leg_chain
            self._right_leg_ik = TwoBoneIK(chain[0], chain[1], chain[2])

        # Spine
        if self.skeleton.spine_chain and len(self.skeleton.spine_chain) >= 2:
            self._spine_ik = FABRIKChain(
                self.skeleton.spine_chain,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations
            )

    def set_bone_mass(self, bone_idx: int, mass: float) -> None:
        """Set mass for a bone (used for COM calculation).

        Args:
            bone_idx: Bone index
            mass: Mass value
        """
        self._bone_masses[bone_idx] = mass

    def set_support_polygon(self, vertices: List[Vec3]) -> None:
        """Set support polygon for balance checking.

        Args:
            vertices: Polygon vertices (typically foot positions)
        """
        self._support_polygon = list(vertices)

    def solve(
        self,
        transforms: List[Transform],
        goals: List[FullBodyIKGoal]
    ) -> FullBodyIKResult:
        """Solve full body IK.

        Args:
            transforms: Current bone transforms (world space)
            goals: List of IK goals to satisfy

        Returns:
            FullBodyIKResult with modified transforms.
        """
        # Copy transforms
        result_transforms = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        # Sort goals by priority
        sorted_goals = sorted(goals, key=lambda g: -g.priority)

        # Filter enabled goals
        active_goals = [g for g in sorted_goals if g.enabled]

        goals_achieved = {}
        final_errors = {}
        pelvis_adjust = Vec3.zero()

        # Phase 1: Adjust pelvis height based on leg goals
        if self.pelvis_height_adjust:
            pelvis_adjust = self._adjust_pelvis_height(result_transforms, active_goals)

        # Phase 2: Solve spine for look-at and reach
        spine_goals = [g for g in active_goals if g.chain_type == "spine"]
        if spine_goals and self._spine_ik:
            self._solve_spine(result_transforms, spine_goals)

        # Phase 3: Solve limbs
        for goal in active_goals:
            achieved, error = self._solve_goal(result_transforms, goal)
            goals_achieved[goal.bone_index] = achieved
            final_errors[goal.bone_index] = error

        # Phase 4: Balance adjustment
        if self.maintain_balance:
            self._maintain_balance(result_transforms)

        # Check overall success
        all_achieved = all(
            err <= self.tolerance
            for err in final_errors.values()
        )

        return FullBodyIKResult(
            success=all_achieved,
            transforms=result_transforms,
            goals_achieved=goals_achieved,
            final_errors=final_errors,
            pelvis_adjustment=pelvis_adjust
        )

    def _adjust_pelvis_height(
        self,
        transforms: List[Transform],
        goals: List[FullBodyIKGoal]
    ) -> Vec3:
        """Adjust pelvis height based on foot positions.

        When feet are on uneven terrain, the pelvis needs to lower
        so both feet can reach.

        Args:
            transforms: Transforms to modify
            goals: Active goals

        Returns:
            Pelvis adjustment vector.
        """
        pelvis_idx = self.skeleton.get_bone(BodyPart.PELVIS)
        if pelvis_idx < 0:
            return Vec3.zero()

        # Find leg goals
        left_leg_goal = None
        right_leg_goal = None

        for goal in goals:
            if goal.chain_type == "left_leg" and goal.has_position():
                left_leg_goal = goal
            elif goal.chain_type == "right_leg" and goal.has_position():
                right_leg_goal = goal

        if not left_leg_goal and not right_leg_goal:
            return Vec3.zero()

        # Compute required pelvis drop
        pelvis_pos = transforms[pelvis_idx].translation
        required_drops = []

        if left_leg_goal and self._left_leg_ik:
            max_reach = self._left_leg_ik.max_reach
            target = left_leg_goal.target_position
            current_dist = (target - pelvis_pos).length()

            if current_dist > max_reach * 0.95:
                drop = current_dist - max_reach * 0.95
                required_drops.append(drop)

        if right_leg_goal and self._right_leg_ik:
            max_reach = self._right_leg_ik.max_reach
            target = right_leg_goal.target_position
            current_dist = (target - pelvis_pos).length()

            if current_dist > max_reach * 0.95:
                drop = current_dist - max_reach * 0.95
                required_drops.append(drop)

        if not required_drops:
            return Vec3.zero()

        # Use maximum required drop
        drop = max(required_drops)
        adjustment = Vec3(0, -drop, 0)

        # Apply to pelvis and all children
        self._translate_hierarchy(transforms, pelvis_idx, adjustment)

        return adjustment

    def _translate_hierarchy(
        self,
        transforms: List[Transform],
        root_idx: int,
        offset: Vec3
    ) -> None:
        """Translate a bone and all its children.

        Args:
            transforms: All transforms
            root_idx: Root of subtree to move
            offset: Translation offset
        """
        # In a full implementation, this would traverse the hierarchy
        # For now, translate all bones (simplified)
        for t in transforms:
            t.translation = t.translation + offset

    def _solve_spine(
        self,
        transforms: List[Transform],
        goals: List[FullBodyIKGoal]
    ) -> None:
        """Solve spine IK for look-at and reach.

        Args:
            transforms: Transforms to modify
            goals: Spine goals
        """
        if not self._spine_ik or not self.skeleton.spine_chain:
            return

        # Get spine positions
        spine_positions = [
            transforms[idx].translation
            for idx in self.skeleton.spine_chain
        ]

        # Find primary target (highest priority goal)
        primary_goal = max(goals, key=lambda g: g.priority) if goals else None

        if primary_goal and primary_goal.has_position():
            # Solve spine to reach target
            target = primary_goal.target_position

            # Apply stiffness - blend between original and solved
            result = self._spine_ik.solve(spine_positions, target)

            for i, idx in enumerate(self.skeleton.spine_chain):
                original_pos = transforms[idx].translation
                solved_pos = result.positions[i]

                # Blend based on stiffness
                blended_pos = original_pos.lerp(
                    solved_pos,
                    1.0 - self.spine_stiffness
                )

                transforms[idx].translation = blended_pos
                transforms[idx].rotation = result.rotations[i]

    def _solve_goal(
        self,
        transforms: List[Transform],
        goal: FullBodyIKGoal
    ) -> Tuple[bool, float]:
        """Solve a single IK goal.

        Args:
            transforms: Transforms to modify
            goal: Goal to solve

        Returns:
            Tuple of (achieved, error).
        """
        if not goal.has_position():
            return True, 0.0

        # Route to appropriate solver based on chain type
        solver = None
        chain = None

        if goal.chain_type == "left_arm":
            solver = self._left_arm_ik
            chain = self.skeleton.left_arm_chain
        elif goal.chain_type == "right_arm":
            solver = self._right_arm_ik
            chain = self.skeleton.right_arm_chain
        elif goal.chain_type == "left_leg":
            solver = self._left_leg_ik
            chain = self.skeleton.left_leg_chain
        elif goal.chain_type == "right_leg":
            solver = self._right_leg_ik
            chain = self.skeleton.right_leg_chain

        if solver is None or chain is None:
            # No solver for this chain - compute error only
            if goal.bone_index < len(transforms):
                error = (
                    goal.target_position - transforms[goal.bone_index].translation
                ).length()
                return error <= self.tolerance, error
            return False, float('inf')

        # Get chain transforms
        root_t = transforms[chain[0]]
        mid_t = transforms[chain[1]]
        end_t = transforms[chain[2]]

        # Solve
        result = solver.solve(
            root_t, mid_t, end_t,
            goal.target_position,
            None,  # No pole vector
            goal.target_rotation
        )

        if result.success:
            transforms[chain[0]].rotation = result.root_rotation
            transforms[chain[1]].rotation = result.mid_rotation
            transforms[chain[2]].rotation = result.end_rotation

        # Compute final error
        final_pos = transforms[goal.bone_index].translation
        error = (goal.target_position - final_pos).length()

        return error <= self.tolerance, error

    def _maintain_balance(self, transforms: List[Transform]) -> None:
        """Adjust pose to maintain balance.

        Ensures center of mass stays within support polygon.

        Args:
            transforms: Transforms to modify
        """
        if not self._bone_masses or len(self._support_polygon) < 3:
            return

        # Compute current COM
        total_mass = 0.0
        com = Vec3.zero()

        for bone_idx, mass in self._bone_masses.items():
            if bone_idx < len(transforms):
                pos = transforms[bone_idx].translation
                com = com + pos * mass
                total_mass += mass

        if total_mass < MATH_EPSILON:
            return

        com = com / total_mass

        # Check if COM is in support polygon
        if self._point_in_polygon(com):
            return

        # Find closest point on polygon edge
        closest = self._closest_point_on_polygon(com)

        # Compute required adjustment
        adjustment = closest - Vec3(com.x, closest.y, com.z)
        adjustment = adjustment * 0.5  # Don't over-correct

        # Apply subtle pelvis shift
        pelvis_idx = self.skeleton.get_bone(BodyPart.PELVIS)
        if pelvis_idx >= 0:
            transforms[pelvis_idx].translation = (
                transforms[pelvis_idx].translation + adjustment
            )

    def _point_in_polygon(self, point: Vec3) -> bool:
        """Check if point is inside support polygon (XZ plane).

        Args:
            point: Point to check

        Returns:
            True if inside polygon.
        """
        if len(self._support_polygon) < 3:
            return True

        n = len(self._support_polygon)
        inside = False

        j = n - 1
        for i in range(n):
            pi = self._support_polygon[i]
            pj = self._support_polygon[j]

            dz = pj.z - pi.z
            if abs(dz) < MATH_EPSILON:
                # Edge is horizontal, skip this edge for the ray cast test
                j = i
                continue
            if ((pi.z > point.z) != (pj.z > point.z)) and \
               (point.x < (pj.x - pi.x) * (point.z - pi.z) / dz + pi.x):
                inside = not inside
            j = i

        return inside

    def _closest_point_on_polygon(self, point: Vec3) -> Vec3:
        """Find closest point on polygon edge.

        Args:
            point: Query point

        Returns:
            Closest point on polygon boundary.
        """
        if len(self._support_polygon) < 2:
            return point

        closest = self._support_polygon[0]
        min_dist = float('inf')

        n = len(self._support_polygon)
        for i in range(n):
            a = self._support_polygon[i]
            b = self._support_polygon[(i + 1) % n]

            # Project point onto edge
            edge = b - a
            edge_len = edge.length()

            if edge_len < POLYGON_EDGE_MIN_LENGTH:
                continue

            edge_dir = edge / edge_len
            to_point = point - a

            t = to_point.dot(edge_dir)
            t = max(0.0, min(t, edge_len))

            projected = a + edge_dir * t
            dist = (point - projected).length()

            if dist < min_dist:
                min_dist = dist
                closest = projected

        return closest


class LookAtSolver:
    """Solver for look-at constraints on the spine.

    Makes the head/eyes track a target point while distributing
    rotation across the spine for natural movement.
    """

    def __init__(
        self,
        head_bone: int,
        neck_bone: int,
        spine_bones: List[int],
        head_weight: float = LOOK_AT_HEAD_WEIGHT,
        neck_weight: float = LOOK_AT_NECK_WEIGHT,
        spine_weight: float = LOOK_AT_SPINE_WEIGHT
    ) -> None:
        """Initialize look-at solver.

        Args:
            head_bone: Head bone index
            neck_bone: Neck bone index
            spine_bones: List of spine bone indices
            head_weight: Weight of head in rotation
            neck_weight: Weight of neck
            spine_weight: Weight distributed across spine
        """
        self.head_bone = head_bone
        self.neck_bone = neck_bone
        self.spine_bones = list(spine_bones)

        self.head_weight = head_weight
        self.neck_weight = neck_weight
        self.spine_weight = spine_weight

        self.max_angle = LOOK_AT_MAX_ANGLE

    def solve(
        self,
        transforms: List[Transform],
        target: Vec3,
        forward_axis: Vec3 = Vec3(0, 0, 1)
    ) -> List[Transform]:
        """Solve look-at constraint.

        Args:
            transforms: Bone transforms
            target: Point to look at
            forward_axis: Local forward axis of head

        Returns:
            Modified transforms.
        """
        result = [
            Transform(t.translation, t.rotation, t.scale)
            for t in transforms
        ]

        head_pos = result[self.head_bone].translation
        to_target = (target - head_pos).normalized()

        # Current forward direction
        current_forward = result[self.head_bone].rotation.rotate_vector(forward_axis)

        # Total rotation needed
        total_rotation = self._rotation_between(current_forward, to_target)

        # Clamp total angle
        axis, angle = self._quat_to_axis_angle(total_rotation)
        if angle > self.max_angle:
            angle = self.max_angle
            total_rotation = Quat.from_axis_angle(axis, angle)

        # Distribute rotation
        head_rot = self._scale_rotation(total_rotation, self.head_weight)
        neck_rot = self._scale_rotation(total_rotation, self.neck_weight)

        spine_per_bone = self.spine_weight / max(1, len(self.spine_bones))

        # Apply head rotation
        result[self.head_bone].rotation = head_rot * result[self.head_bone].rotation

        # Apply neck rotation
        result[self.neck_bone].rotation = neck_rot * result[self.neck_bone].rotation

        # Apply spine rotations
        for bone_idx in self.spine_bones:
            spine_rot = self._scale_rotation(total_rotation, spine_per_bone)
            result[bone_idx].rotation = spine_rot * result[bone_idx].rotation

        return result

    def _rotation_between(self, from_vec: Vec3, to_vec: Vec3) -> Quat:
        """Compute rotation between two vectors."""
        dot = from_vec.dot(to_vec)
        dot = max(-1.0, min(1.0, dot))

        if dot > 1.0 - MATH_EPSILON:
            return Quat.identity()

        if dot < -1.0 + MATH_EPSILON:
            axis = Vec3.unit_x().cross(from_vec)
            if axis.length_squared() < MATH_EPSILON:
                axis = Vec3.unit_y().cross(from_vec)
            return Quat.from_axis_angle(axis.normalized(), math.pi)

        axis = from_vec.cross(to_vec).normalized()
        angle = math.acos(dot)
        return Quat.from_axis_angle(axis, angle)

    def _scale_rotation(self, rotation: Quat, scale: float) -> Quat:
        """Scale a rotation by a factor."""
        axis, angle = self._quat_to_axis_angle(rotation)
        return Quat.from_axis_angle(axis, angle * scale)

    def _quat_to_axis_angle(self, q: Quat) -> Tuple[Vec3, float]:
        """Extract axis and angle from quaternion."""
        angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))

        s = math.sqrt(1.0 - q.w * q.w)
        if s < MATH_EPSILON:
            return Vec3.unit_y(), 0.0

        axis = Vec3(q.x / s, q.y / s, q.z / s)
        return axis, angle
