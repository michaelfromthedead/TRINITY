"""Full body IK solver.

This module implements a full body inverse kinematics solver that handles
multiple end effectors simultaneously while maintaining balance and
natural poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Callable, Tuple, Set

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


@dataclass
class COMCalculator:
    """Center of Mass calculator for skeleton.

    Calculates the weighted center of mass from bone positions,
    using configurable per-bone masses. Useful for balance checking
    and physics-aware animation.

    Attributes:
        bone_masses: Dictionary mapping bone name to mass value.
            Bones not in this dict default to mass 1.0.
        default_mass: Default mass for bones not explicitly configured.
    """

    bone_masses: Dict[str, float] = field(default_factory=dict)
    default_mass: float = 1.0

    def set_bone_mass(self, bone_name: str, mass: float) -> None:
        """Set mass for a bone.

        Args:
            bone_name: Name of the bone.
            mass: Mass value (must be non-negative).

        Raises:
            ValueError: If mass is negative.
        """
        if mass < 0:
            raise ValueError(f"Mass cannot be negative: {mass}")
        self.bone_masses[bone_name] = mass

    def set_bone_masses(self, masses: Dict[str, float]) -> None:
        """Set masses for multiple bones at once.

        Args:
            masses: Dictionary mapping bone names to mass values.

        Raises:
            ValueError: If any mass is negative.
        """
        for bone_name, mass in masses.items():
            self.set_bone_mass(bone_name, mass)

    def get_bone_mass(self, bone_name: str) -> float:
        """Get mass for a bone.

        Args:
            bone_name: Name of the bone.

        Returns:
            Mass value, or default_mass if not configured.
        """
        return self.bone_masses.get(bone_name, self.default_mass)

    def calculate(self, bone_positions: Dict[str, Vec3]) -> Vec3:
        """Calculate weighted center of mass from bone positions.

        COM = sum(mass_i * pos_i) / sum(mass_i)

        Args:
            bone_positions: Dictionary mapping bone names to world positions.

        Returns:
            Center of mass position, or Vec3.zero() if total mass is zero.
        """
        total_mass = 0.0
        weighted_sum = Vec3.zero()

        for bone_name, pos in bone_positions.items():
            mass = self.bone_masses.get(bone_name, self.default_mass)
            total_mass += mass
            weighted_sum = weighted_sum + pos * mass

        if total_mass <= 0.0:
            return Vec3.zero()

        return weighted_sum / total_mass

    def calculate_partial(
        self,
        bone_positions: Dict[str, Vec3],
        bone_subset: Set[str]
    ) -> Vec3:
        """Calculate center of mass for a subset of bones.

        Useful for calculating COM of specific body parts (e.g., upper body,
        left arm chain) rather than the full skeleton.

        Args:
            bone_positions: Dictionary mapping bone names to world positions.
            bone_subset: Set of bone names to include in calculation.

        Returns:
            Center of mass for the subset, or Vec3.zero() if total mass is zero.
        """
        total_mass = 0.0
        weighted_sum = Vec3.zero()

        for bone_name in bone_subset:
            if bone_name not in bone_positions:
                continue
            pos = bone_positions[bone_name]
            mass = self.bone_masses.get(bone_name, self.default_mass)
            total_mass += mass
            weighted_sum = weighted_sum + pos * mass

        if total_mass <= 0.0:
            return Vec3.zero()

        return weighted_sum / total_mass

    def calculate_from_transforms(
        self,
        transforms: List[Transform],
        bone_names: List[str]
    ) -> Vec3:
        """Calculate COM from a list of transforms with corresponding bone names.

        Convenience method when bone positions are stored as transforms
        rather than a dictionary.

        Args:
            transforms: List of bone transforms (world space).
            bone_names: List of bone names corresponding to each transform.

        Returns:
            Center of mass position.

        Raises:
            ValueError: If transforms and bone_names have different lengths.
        """
        if len(transforms) != len(bone_names):
            raise ValueError(
                f"transforms ({len(transforms)}) and bone_names ({len(bone_names)}) "
                "must have the same length"
            )

        bone_positions = {
            name: t.translation
            for name, t in zip(bone_names, transforms)
        }
        return self.calculate(bone_positions)

    def total_mass(self, bone_names: Optional[Set[str]] = None) -> float:
        """Calculate total mass for given bones.

        Args:
            bone_names: Set of bone names to sum, or None to use all
                configured bones (falling back to 0 if none configured).

        Returns:
            Total mass.
        """
        if bone_names is None:
            return sum(self.bone_masses.values())

        return sum(
            self.bone_masses.get(name, self.default_mass)
            for name in bone_names
        )


@dataclass
class SupportPolygon:
    """Support polygon for balance checking.

    The support polygon is the convex hull of foot contact points,
    projected onto the ground plane (XZ). Used to check if center
    of mass is within stable support region.

    Attributes:
        vertices: XZ plane vertices defining the polygon boundary.
    """

    vertices: List[Vec3] = field(default_factory=list)

    @classmethod
    def from_foot_positions(cls, positions: List[Vec3]) -> 'SupportPolygon':
        """Build support polygon from foot contact positions.

        Projects positions to XZ plane (y=0). For basic case, uses
        the positions directly without computing convex hull.

        Args:
            positions: List of foot contact positions in world space.

        Returns:
            SupportPolygon with vertices projected to ground plane.
        """
        projected = [Vec3(p.x, 0.0, p.z) for p in positions]
        return cls(vertices=projected)

    def contains_point(self, point: Vec3) -> bool:
        """Check if point is inside polygon using ray casting.

        Projects point to XZ plane and uses ray casting algorithm:
        Cast ray from point, count edge crossings. Odd = inside.

        Handles horizontal edges correctly by skipping them in the
        crossing calculation (they don't contribute to the count).

        Args:
            point: Point to test (will be projected to XZ plane).

        Returns:
            True if point is inside the polygon, False otherwise.
        """
        if len(self.vertices) < 3:
            return False

        # Project to XZ (use x and z as 2D coordinates)
        px, pz = point.x, point.z
        n = len(self.vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, zi = self.vertices[i].x, self.vertices[i].z
            xj, zj = self.vertices[j].x, self.vertices[j].z

            # Skip horizontal edges (both endpoints have same z)
            dz = zj - zi
            if abs(dz) < 1e-10:
                j = i
                continue

            # Ray casting: count crossings of horizontal ray from point
            if ((zi > pz) != (zj > pz)) and \
               (px < (xj - xi) * (pz - zi) / dz + xi):
                inside = not inside
            j = i

        return inside

    def project_to_ground(self, point: Vec3) -> Vec3:
        """Project a point onto the ground plane (y=0).

        Args:
            point: Point to project.

        Returns:
            Point with y coordinate set to 0.
        """
        return Vec3(point.x, 0.0, point.z)

    @staticmethod
    def closest_point_on_segment(point: Vec3, seg_start: Vec3, seg_end: Vec3) -> Vec3:
        """Find closest point on line segment to a given point.

        Projects point onto line defined by segment, clamped to segment bounds.
        Works in XZ plane (ignores Y).

        Args:
            point: Query point
            seg_start: Segment start point
            seg_end: Segment end point

        Returns:
            Closest point on segment to query point
        """
        # Direction vector of segment
        dx = seg_end.x - seg_start.x
        dz = seg_end.z - seg_start.z

        # Length squared
        len_sq = dx * dx + dz * dz
        if len_sq < 1e-10:
            return seg_start  # Degenerate segment

        # Parameter t for projection onto line (0 = start, 1 = end)
        t = ((point.x - seg_start.x) * dx + (point.z - seg_start.z) * dz) / len_sq
        t = max(0.0, min(1.0, t))  # Clamp to segment

        return Vec3(
            seg_start.x + t * dx,
            0.0,  # Ground plane
            seg_start.z + t * dz
        )

    def closest_point_on_boundary(self, point: Vec3) -> Vec3:
        """Find the closest point on polygon boundary to given point.

        Iterates through all edges and finds the closest point.
        Works in XZ plane.

        Args:
            point: Query point (projected to XZ plane)

        Returns:
            Closest point on polygon boundary
        """
        if len(self.vertices) < 2:
            return self.vertices[0] if self.vertices else Vec3.zero()

        closest = None
        min_dist_sq = float('inf')

        # Iterate through all edges
        n = len(self.vertices)
        for i in range(n):
            seg_start = self.vertices[i]
            seg_end = self.vertices[(i + 1) % n]

            candidate = self.closest_point_on_segment(point, seg_start, seg_end)

            # Distance squared in XZ plane
            dx = candidate.x - point.x
            dz = candidate.z - point.z
            dist_sq = dx * dx + dz * dz

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest = candidate

        return closest if closest else Vec3.zero()

    def correction_vector(self, point: Vec3) -> Vec3:
        """Get vector to move point back onto polygon boundary.

        If point is inside polygon, returns zero vector.
        If outside, returns vector from point to closest boundary point.
        Used for COM correction.

        Args:
            point: Query point (typically COM)

        Returns:
            Correction vector (zero if inside, direction to boundary if outside)
        """
        if self.contains_point(point):
            return Vec3.zero()

        closest = self.closest_point_on_boundary(point)
        return Vec3(
            closest.x - point.x,
            0.0,
            closest.z - point.z
        )


@dataclass
class BalanceController:
    """Controller for maintaining balance by adjusting pelvis/spine.

    Uses center of mass and support polygon to detect instability
    and apply corrective adjustments. The controller calculates
    whether the character's COM is within the support polygon and
    provides correction vectors when it falls outside.

    Attributes:
        com_calculator: Calculator for center of mass.
        support_polygon: Current support polygon defining stable region.
        correction_strength: How strongly to correct imbalance (0-1).
            Higher values result in more aggressive corrections.
        pelvis_weight: Weight of pelvis adjustment relative to spine.
            The spine receives (1 - pelvis_weight) of the correction.
    """

    com_calculator: COMCalculator = field(default_factory=COMCalculator)
    support_polygon: SupportPolygon = field(default_factory=SupportPolygon)
    correction_strength: float = 0.5
    pelvis_weight: float = 0.7

    def is_balanced(self, bone_positions: Dict[str, Vec3]) -> bool:
        """Check if current pose is balanced (COM in support polygon).

        Args:
            bone_positions: Dictionary mapping bone names to world positions.

        Returns:
            True if center of mass is within support polygon, False otherwise.
        """
        com = self.com_calculator.calculate(bone_positions)
        return self.support_polygon.contains_point(com)

    def get_correction(self, bone_positions: Dict[str, Vec3]) -> Vec3:
        """Get correction vector to restore balance.

        Calculates the vector needed to move the center of mass back
        into the support polygon, scaled by correction_strength.

        Args:
            bone_positions: Dictionary mapping bone names to world positions.

        Returns:
            Correction vector scaled by strength. Returns zero vector
            if already balanced.
        """
        com = self.com_calculator.calculate(bone_positions)
        raw_correction = self.support_polygon.correction_vector(com)
        return raw_correction * self.correction_strength

    def apply_correction(
        self,
        bone_positions: Dict[str, Vec3],
        pelvis_name: str = "pelvis",
        spine_name: str = "spine"
    ) -> Dict[str, Vec3]:
        """Apply balance correction to bone positions.

        Distributes the correction between pelvis and spine based on
        pelvis_weight. Only adjusts X and Z coordinates to maintain
        ground contact.

        Args:
            bone_positions: Current bone positions. A copy is made
                and modified rather than mutating the original.
            pelvis_name: Name of the pelvis bone to adjust.
            spine_name: Name of the spine bone to adjust.

        Returns:
            New dictionary with corrected bone positions.
        """
        correction = self.get_correction(bone_positions)

        # Skip if correction is negligible
        if correction.length_squared() < 1e-10:
            return bone_positions

        # Distribute correction between pelvis and spine
        pelvis_correction = correction * self.pelvis_weight
        spine_correction = correction * (1.0 - self.pelvis_weight)

        # Create result with corrections applied
        result = dict(bone_positions)

        if pelvis_name in result:
            pos = result[pelvis_name]
            result[pelvis_name] = Vec3(
                pos.x + pelvis_correction.x,
                pos.y,  # Keep Y unchanged to maintain ground contact
                pos.z + pelvis_correction.z
            )

        if spine_name in result:
            pos = result[spine_name]
            result[spine_name] = Vec3(
                pos.x + spine_correction.x,
                pos.y,  # Keep Y unchanged
                pos.z + spine_correction.z
            )

        return result

    def set_correction_strength(self, strength: float) -> None:
        """Set correction strength, clamped to valid range.

        Args:
            strength: Desired strength value. Will be clamped to [0, 1].
        """
        self.correction_strength = max(0.0, min(1.0, strength))

    def update_support_polygon(self, foot_positions: List[Vec3]) -> None:
        """Update support polygon from current foot positions.

        Rebuilds the support polygon based on new foot contact points.
        Call this when feet move to update the stable region.

        Args:
            foot_positions: List of foot contact positions in world space.
        """
        self.support_polygon = SupportPolygon.from_foot_positions(foot_positions)


class IKSolverType(Enum):
    """Types of IK solvers that can be used for a chain."""
    TWO_BONE = auto()
    FABRIK = auto()
    CCD = auto()
    JACOBIAN = auto()


@dataclass
class IKChain:
    """Definition of an IK chain for full body solving.

    An IK chain represents a series of bones from root to effector
    that are solved together using a specific IK algorithm.

    Attributes:
        name: Unique identifier for this chain
        root_bone: Name of the root bone (e.g., "shoulder")
        joint_bones: Names of intermediate bones (e.g., ["upper_arm", "lower_arm"])
        effector_bone: Name of the end effector bone (e.g., "hand")
        solver_type: Which IK algorithm to use
        weight: Influence of this chain (0-1)
        priority: Solve order (higher = solved first)
        enabled: Whether this chain is active
    """

    name: str
    root_bone: str
    joint_bones: List[str] = field(default_factory=list)
    effector_bone: str = ""
    solver_type: IKSolverType = IKSolverType.TWO_BONE
    weight: float = 1.0
    priority: int = 0
    enabled: bool = True

    def __post_init__(self):
        self.weight = max(0.0, min(1.0, self.weight))

    @property
    def bone_count(self) -> int:
        """Total number of bones in chain (root + joints + effector)."""
        return 2 + len(self.joint_bones)  # root, joints, effector

    @property
    def all_bones(self) -> List[str]:
        """Get all bone names in order from root to effector."""
        return [self.root_bone] + self.joint_bones + [self.effector_bone]

    def set_weight(self, weight: float) -> None:
        """Set chain weight (clamped to 0-1)."""
        self.weight = max(0.0, min(1.0, weight))

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this chain."""
        self.enabled = enabled

    @classmethod
    def arm_chain(cls, side: str = "left") -> 'IKChain':
        """Create a standard arm chain."""
        prefix = "l_" if side == "left" else "r_"
        return cls(
            name=f"{side}_arm",
            root_bone=f"{prefix}shoulder",
            joint_bones=[f"{prefix}upper_arm", f"{prefix}lower_arm"],
            effector_bone=f"{prefix}hand",
            solver_type=IKSolverType.TWO_BONE,
            priority=10
        )

    @classmethod
    def leg_chain(cls, side: str = "left") -> 'IKChain':
        """Create a standard leg chain."""
        prefix = "l_" if side == "left" else "r_"
        return cls(
            name=f"{side}_leg",
            root_bone=f"{prefix}thigh",
            joint_bones=[f"{prefix}shin"],
            effector_bone=f"{prefix}foot",
            solver_type=IKSolverType.TWO_BONE,
            priority=20  # Legs solved before arms
        )


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


@dataclass
class PelvisAdjustmentConfig:
    """Configuration for pelvis height adjustment.

    Controls how aggressively and smoothly the pelvis adjusts
    to allow legs to reach their targets.

    Attributes:
        safety_margin: Maximum reach safety factor (0-1).
            At 0.95, legs will only extend to 95% of max reach.
        max_drop: Maximum pelvis drop distance in world units.
            Prevents extreme crouching.
        smooth_speed: Smoothing speed in units/second.
            Higher values = faster adjustment, less smooth.
    """

    safety_margin: float = 0.95
    max_drop: float = 0.5
    smooth_speed: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.safety_margin <= 1.0:
            raise ValueError(
                f"safety_margin must be in (0, 1], got {self.safety_margin}"
            )
        if self.max_drop < 0.0:
            raise ValueError(
                f"max_drop must be non-negative, got {self.max_drop}"
            )
        if self.smooth_speed <= 0.0:
            raise ValueError(
                f"smooth_speed must be positive, got {self.smooth_speed}"
            )


class PelvisHeightAdjuster:
    """Standalone pelvis height adjustment for IK.

    Calculates and applies smooth pelvis height adjustments to ensure
    legs can reach their target positions. Useful for foot placement
    on uneven terrain.

    The adjuster tracks internal state for smooth transitions and
    provides configurable safety margins and limits.

    Example:
        adjuster = PelvisHeightAdjuster(PelvisAdjustmentConfig(
            safety_margin=0.95,
            max_drop=0.5,
            smooth_speed=5.0
        ))

        # Each frame:
        adjustment = adjuster.adjust(
            transforms, pelvis_idx, leg_targets, max_leg_reach, dt
        )
    """

    def __init__(self, config: Optional[PelvisAdjustmentConfig] = None) -> None:
        """Initialize the pelvis height adjuster.

        Args:
            config: Configuration for adjustment behavior.
                Uses defaults if None.
        """
        self._config = config if config is not None else PelvisAdjustmentConfig()
        self._current_offset: float = 0.0

    @property
    def config(self) -> PelvisAdjustmentConfig:
        """Get the current configuration."""
        return self._config

    @property
    def current_offset(self) -> float:
        """Get the current vertical offset being applied."""
        return self._current_offset

    def calculate_required_drop(
        self,
        pelvis_pos: Vec3,
        leg_targets: List[Vec3],
        max_leg_reach: float
    ) -> float:
        """Calculate raw drop needed for legs to reach targets.

        Computes how much the pelvis needs to drop vertically so that
        all leg targets are within reach, accounting for the safety margin.

        Args:
            pelvis_pos: Current pelvis world position.
            leg_targets: List of target positions for legs.
                Empty targets are skipped.
            max_leg_reach: Maximum leg reach distance.
                Same value used for all legs.

        Returns:
            Required drop distance (positive = down).
            Returns 0 if all targets are reachable.
        """
        if not leg_targets or max_leg_reach <= 0.0:
            return 0.0

        safe_reach = max_leg_reach * self._config.safety_margin
        max_required_drop = 0.0

        for target in leg_targets:
            if target is None:
                continue

            # Distance from pelvis to target
            to_target = target - pelvis_pos
            current_dist = to_target.length()

            # If beyond safe reach, calculate required drop
            if current_dist > safe_reach:
                # The drop required to bring distance within safe reach
                # Using vertical projection approximation
                excess = current_dist - safe_reach

                # Calculate the vertical component of adjustment
                # If target is below pelvis, dropping helps directly
                # If target is at same height, horizontal distance dominates
                if current_dist > MATH_EPSILON:
                    # Use geometry: after dropping by d, new distance
                    # sqrt((dist_xz)^2 + (dist_y - d)^2) <= safe_reach
                    # Solve for minimum d
                    dist_xz_sq = to_target.x ** 2 + to_target.z ** 2
                    dist_y = to_target.y

                    # Quadratic to find minimum drop
                    # We want: dist_xz_sq + (dist_y - d)^2 <= safe_reach^2
                    # (dist_y - d)^2 <= safe_reach^2 - dist_xz_sq
                    target_y_sq = safe_reach ** 2 - dist_xz_sq

                    if target_y_sq < 0:
                        # Target is too far horizontally, drop won't help
                        # Use excess as approximation
                        required_drop = excess
                    else:
                        # Required vertical distance after adjustment
                        required_y_dist = math.sqrt(target_y_sq)
                        # If target is below pelvis, dist_y is negative
                        # required_drop = -dist_y - required_y_dist (for below)
                        # or = dist_y - required_y_dist (for above, but usually negative)
                        if dist_y < 0:
                            # Target below pelvis - typical case
                            required_drop = -dist_y - required_y_dist
                        else:
                            # Target at or above pelvis - unusual case
                            required_drop = dist_y - required_y_dist

                        required_drop = max(0.0, required_drop)
                else:
                    required_drop = 0.0

                max_required_drop = max(max_required_drop, required_drop)

        return max_required_drop

    def adjust(
        self,
        transforms: List[Transform],
        pelvis_idx: int,
        leg_targets: List[Vec3],
        max_leg_reach: float,
        dt: float
    ) -> Vec3:
        """Calculate and apply smoothed pelvis adjustment.

        Main method: calculates required drop, smooths it over time,
        and applies the adjustment to the pelvis transform.

        Args:
            transforms: List of transforms to modify.
                The pelvis transform at pelvis_idx will be adjusted.
            pelvis_idx: Index of pelvis in transforms list.
            leg_targets: List of world positions for leg targets.
                None entries are skipped.
            max_leg_reach: Maximum leg reach distance.
            dt: Delta time in seconds for smoothing.

        Returns:
            The adjustment vector applied (vertical only).
            Returns zero vector if no adjustment needed.
        """
        if pelvis_idx < 0 or pelvis_idx >= len(transforms):
            return Vec3.zero()

        pelvis_transform = transforms[pelvis_idx]
        pelvis_pos = pelvis_transform.translation

        # Calculate target drop
        target_drop = self.calculate_required_drop(
            pelvis_pos, leg_targets, max_leg_reach
        )

        # Clamp to max drop
        target_drop = min(target_drop, self._config.max_drop)

        # Smooth adjustment using lerp
        # current += (target - current) * min(1.0, smooth_speed * dt)
        alpha = min(1.0, self._config.smooth_speed * dt)
        self._current_offset += (target_drop - self._current_offset) * alpha

        # Create adjustment vector (negative Y = downward)
        adjustment = Vec3(0.0, -self._current_offset, 0.0)

        # Apply to pelvis - we need delta from last frame
        # Since transforms may reset each frame, we apply full offset
        # The caller typically provides fresh transforms each frame
        pelvis_transform.translation = pelvis_pos + Vec3(
            0.0, -self._current_offset, 0.0
        )

        return adjustment

    def reset(self) -> None:
        """Reset internal state.

        Call this when the character teleports or on animation reset
        to prevent jarring smooth transitions from old positions.
        """
        self._current_offset = 0.0

    def set_config(self, config: PelvisAdjustmentConfig) -> None:
        """Update the configuration.

        Args:
            config: New configuration to use.
        """
        self._config = config

    def get_target_offset(
        self,
        pelvis_pos: Vec3,
        leg_targets: List[Vec3],
        max_leg_reach: float
    ) -> float:
        """Get the target offset without applying smoothing.

        Useful for debugging or preview purposes.

        Args:
            pelvis_pos: Current pelvis world position.
            leg_targets: List of target positions for legs.
            max_leg_reach: Maximum leg reach distance.

        Returns:
            Target offset (clamped to max_drop).
        """
        target_drop = self.calculate_required_drop(
            pelvis_pos, leg_targets, max_leg_reach
        )
        return min(target_drop, self._config.max_drop)


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
