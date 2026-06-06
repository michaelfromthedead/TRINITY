"""IK System Integration (T-AN-9.4).

This module implements the IK solver dispatch system that runs AFTER the
animation graph evaluation (order=0). It processes IK goals in priority order
and dispatches to appropriate solvers based on chain type.

Key Features:
- @system(phase="animation", order=1) annotation for ECS scheduling
- Priority-ordered goal processing (higher priority first)
- Solver dispatch: two-bone, FABRIK, CCD, Jacobian, full-body
- Chain weight blending (0-1)
- Chain enable/disable support
- Multi-chain interaction handling

Dependencies:
- Phase 4 IK solvers: TwoBoneIK, FABRIKChain, CCDSolver, JacobianIK, FullBodyIK
- Phase 4 IK goals: IKGoal, PositionGoal, RotationGoal, ChainGoal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Quat, Transform
from engine.animation.config import IK_CONFIG, ANIMATION_SYSTEM_CONFIG

# Import IK solvers from Phase 4
from engine.animation.ik import (
    # Solvers
    TwoBoneIK,
    TwoBoneIKResult,
    FABRIKChain,
    FABRIKResult,
    CCDSolver,
    CCDResult,
    JacobianIK,
    JacobianResult,
    JacobianMethod,
    FullBodyIK,
    FullBodyIKGoal,
    FullBodyIKResult,
    SkeletonMapping,
    # Goals
    IKGoal as BaseIKGoal,
    IKGoalType,
    PositionGoal,
    RotationGoal,
    LookAtGoal,
    PositionRotationGoal,
    PoleVectorGoal,
    ChainGoal,
    CenterOfMassGoal,
    IKGoalBlender,
    # Config
    IK_DEFAULT_TOLERANCE,
    FABRIK_DEFAULT_MAX_ITERATIONS,
    CCD_DEFAULT_MAX_ITERATIONS,
    JACOBIAN_DEFAULT_MAX_ITERATIONS,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    priority: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier). 0 = animation graph, 1 = IK
        priority: Legacy priority field (deprecated, use order instead)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_priority = priority if priority else order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# SOLVER TYPE ENUM
# =============================================================================


class IKSolverType(Enum):
    """Type of IK solver to use for a chain."""

    TWO_BONE = auto()
    """Analytical two-bone solver (arms/legs). Fast, exact for 2-bone chains."""

    FABRIK = auto()
    """Forward And Backward Reaching IK. Good for spines and long chains."""

    CCD = auto()
    """Cyclic Coordinate Descent. Good for chains with joint constraints."""

    JACOBIAN = auto()
    """Jacobian-based solver. Best for complex multi-target scenarios."""

    FULL_BODY = auto()
    """Full body solver with balance/COM. For multi-effector coordination."""

    AUTO = auto()
    """Automatically select solver based on chain length and type."""


class IKHintType(Enum):
    """Type of IK hint/pole vector for controlling chain plane."""

    NONE = auto()
    """No hint - solver uses default bending direction."""

    POSITION = auto()
    """Pole vector as world position (e.g., knee points toward pole)."""

    DIRECTION = auto()
    """Pole vector as direction vector."""


# =============================================================================
# IK GOAL DATA STRUCTURES
# =============================================================================


@dataclass
class IKGoal:
    """IK goal definition for the IK system.

    Defines a target for an IK chain with solver configuration
    and blending parameters.

    Attributes:
        target_bone: Index of end effector bone
        target_position: Target world position (None if rotation-only)
        target_rotation: Target world rotation (optional)
        weight: Blend weight (0-1), controls how much IK affects final pose
        priority: Processing priority (higher = processed first)
        chain_length: Number of bones in the chain
        solver_type: Type of solver to use
        hint_type: Type of pole vector hint
        hint_value: Pole vector value
        position_tolerance: Position error tolerance for convergence
        rotation_tolerance: Rotation error tolerance for convergence
        max_iterations: Maximum solver iterations (for iterative solvers)
        enabled: Whether this goal is currently active
        chain_name: Optional name for the chain (for debugging/lookup)
    """

    target_bone: int = -1
    target_position: Optional[Vec3] = None
    target_rotation: Optional[Quat] = None
    weight: float = 1.0
    priority: int = 0
    chain_length: int = 2
    solver_type: IKSolverType = IKSolverType.AUTO
    hint_type: IKHintType = IKHintType.NONE
    hint_value: Vec3 = field(default_factory=Vec3.zero)
    position_tolerance: float = IK_CONFIG.DEFAULT_POSITION_TOLERANCE
    rotation_tolerance: float = IK_CONFIG.DEFAULT_ROTATION_TOLERANCE
    max_iterations: int = IK_CONFIG.DEFAULT_MAX_ITERATIONS
    enabled: bool = True
    chain_name: str = ""

    def set_target(self, position: Optional[Vec3] = None, rotation: Optional[Quat] = None) -> None:
        """Set target position and/or rotation.

        Args:
            position: World-space target position
            rotation: World-space target rotation
        """
        if position is not None:
            self.target_position = position
        if rotation is not None:
            self.target_rotation = rotation

    def has_position_target(self) -> bool:
        """Check if this goal has a position target."""
        return self.target_position is not None

    def has_rotation_target(self) -> bool:
        """Check if this goal has a rotation target."""
        return self.target_rotation is not None

    def is_valid(self) -> bool:
        """Validate goal configuration."""
        if self.target_bone < 0:
            return False
        if self.chain_length < 1:
            return False
        if self.weight < 0.0 or self.weight > 1.0:
            return False
        if not self.has_position_target() and not self.has_rotation_target():
            return False
        return True


@dataclass
class IKChainBone:
    """Bone in an IK chain with cached transform data.

    Attributes:
        bone_index: Index in skeleton
        length: Length of this bone segment
        local_transform: Local-space transform (relative to parent)
        world_position: Cached world-space position
        world_rotation: Cached world-space rotation
    """

    bone_index: int
    length: float = 0.0
    local_transform: Transform = field(default_factory=Transform.identity)
    world_position: Vec3 = field(default_factory=Vec3.zero)
    world_rotation: Quat = field(default_factory=Quat.identity)


@dataclass
class IKSolveResult:
    """Result from an IK solve operation.

    Attributes:
        converged: Whether solver converged within tolerance
        iterations: Number of iterations taken
        final_error: Final position error distance
        bone_transforms: Modified transforms for affected bones
        solver_type: Which solver was used
    """

    converged: bool = False
    iterations: int = 0
    final_error: float = float('inf')
    bone_transforms: Dict[int, Transform] = field(default_factory=dict)
    solver_type: IKSolverType = IKSolverType.AUTO


# =============================================================================
# IK COMPONENT
# =============================================================================


@dataclass
class IKComponent:
    """Component for entities with IK needs.

    Attached to entities that require inverse kinematics processing.

    Attributes:
        goals: List of IK goals to process
        enabled: Master enable/disable for all IK
        blend_to_animation: Blend factor (0 = full IK, 1 = full animation)
        skeleton_mapping: Optional skeleton mapping for full-body IK
    """

    goals: List[IKGoal] = field(default_factory=list)
    enabled: bool = True
    blend_to_animation: float = 0.0
    skeleton_mapping: Optional[SkeletonMapping] = None

    def add_goal(self, goal: IKGoal) -> int:
        """Add an IK goal to this component.

        Args:
            goal: The IK goal to add

        Returns:
            Index of the added goal
        """
        self.goals.append(goal)
        return len(self.goals) - 1

    def remove_goal(self, index: int) -> bool:
        """Remove a goal by index.

        Args:
            index: Index of goal to remove

        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self.goals):
            self.goals.pop(index)
            return True
        return False

    def get_goal(self, index: int) -> Optional[IKGoal]:
        """Get a goal by index.

        Args:
            index: Goal index

        Returns:
            The goal or None if invalid index
        """
        if 0 <= index < len(self.goals):
            return self.goals[index]
        return None

    def get_goal_by_name(self, name: str) -> Optional[IKGoal]:
        """Get a goal by chain name.

        Args:
            name: Chain name to search for

        Returns:
            The goal or None if not found
        """
        for goal in self.goals:
            if goal.chain_name == name:
                return goal
        return None

    def set_goal_enabled(self, index: int, enabled: bool) -> bool:
        """Enable or disable a goal.

        Args:
            index: Goal index
            enabled: New enabled state

        Returns:
            True if set, False if invalid index
        """
        goal = self.get_goal(index)
        if goal:
            goal.enabled = enabled
            return True
        return False

    def set_goal_weight(self, index: int, weight: float) -> bool:
        """Set a goal's weight.

        Args:
            index: Goal index
            weight: New weight (0-1)

        Returns:
            True if set, False if invalid index
        """
        goal = self.get_goal(index)
        if goal:
            goal.weight = max(0.0, min(1.0, weight))
            return True
        return False

    def set_all_weights(self, weight: float) -> None:
        """Set weight for all goals.

        Args:
            weight: Weight to apply (0-1)
        """
        weight = max(0.0, min(1.0, weight))
        for goal in self.goals:
            goal.weight = weight

    def get_enabled_goals_sorted(self) -> List[IKGoal]:
        """Get enabled goals sorted by priority (descending).

        Returns:
            List of enabled goals, highest priority first
        """
        return sorted(
            [g for g in self.goals if g.enabled and g.weight > 0],
            key=lambda g: g.priority,
            reverse=True
        )


# =============================================================================
# IK SYSTEM
# =============================================================================


@system(phase="animation", order=1, reads=("IKComponent",), writes=("Pose",))
class IKSystem:
    """ECS system for inverse kinematics.

    Runs AFTER the animation graph system (order=0) to apply IK modifications
    to the base pose. Processes goals in priority order and dispatches to
    the appropriate solver based on chain type.

    The system:
    1. Reads IK goals from IKComponent
    2. Sorts goals by priority (higher first)
    3. For each goal, identifies chain type and dispatches to solver
    4. Applies results to working pose with weight blending
    5. Outputs modified pose for procedural/skinning systems
    """

    def __init__(self) -> None:
        """Initialize the IK system."""
        # Skeleton data (set per entity)
        self._bone_hierarchy: Dict[int, int] = {}  # bone -> parent
        self._bone_lengths: Dict[int, float] = {}
        self._world_transforms: Dict[int, Transform] = {}

        # Solver instances (cached for performance)
        self._two_bone_solvers: Dict[Tuple[int, int, int], TwoBoneIK] = {}
        self._fabrik_chains: Dict[Tuple[int, ...], FABRIKChain] = {}
        self._ccd_solvers: Dict[Tuple[int, ...], CCDSolver] = {}
        self._jacobian_solvers: Dict[int, JacobianIK] = {}
        self._fullbody_solver: Optional[FullBodyIK] = None

        # Goal blender for smooth transitions
        self._goal_blender = IKGoalBlender()

        # Performance stats
        self._stats = IKSystemStats()

    def set_skeleton_data(
        self,
        hierarchy: Dict[int, int],
        bone_lengths: Dict[int, float]
    ) -> None:
        """Set skeleton data for IK solving.

        Args:
            hierarchy: Bone index -> parent index mapping (-1 for root)
            bone_lengths: Bone index -> bone length mapping
        """
        self._bone_hierarchy = dict(hierarchy)
        self._bone_lengths = dict(bone_lengths)
        # Clear cached solvers as skeleton changed
        self._clear_solver_cache()

    def _clear_solver_cache(self) -> None:
        """Clear all cached solver instances."""
        self._two_bone_solvers.clear()
        self._fabrik_chains.clear()
        self._ccd_solvers.clear()
        self._jacobian_solvers.clear()
        self._fullbody_solver = None

    def update(
        self,
        world: Any,
        entity_components: List[Tuple[Any, IKComponent]],
        pose_data: Dict[Any, Dict[int, Transform]],
        dt: float = 1.0 / 60.0
    ) -> Dict[Any, Dict[int, Transform]]:
        """Update all IK components.

        Main entry point called by the animation scheduler.

        Args:
            world: ECS world reference
            entity_components: List of (entity, IKComponent) tuples
            pose_data: Current poses for entities (entity -> bone transforms)
            dt: Delta time for blending

        Returns:
            Updated pose data with IK applied
        """
        self._stats.reset_frame()
        result: Dict[Any, Dict[int, Transform]] = {}

        for entity, component in entity_components:
            if not component.enabled:
                # Pass through unchanged pose
                result[entity] = dict(pose_data.get(entity, {}))
                continue

            # Get current pose for this entity
            entity_pose = dict(pose_data.get(entity, {}))

            # Compute world transforms from local
            self._world_transforms = self._compute_world_transforms(entity_pose)

            # Get goals sorted by priority (highest first)
            sorted_goals = component.get_enabled_goals_sorted()

            # Process each goal in priority order
            for goal in sorted_goals:
                if not goal.is_valid():
                    self._stats.invalid_goals += 1
                    continue

                # Solve this goal
                solve_result = self._solve_goal(goal, entity_pose, component, dt)

                if solve_result.converged or solve_result.iterations > 0:
                    # Apply results with weight blending
                    blend_weight = goal.weight * (1.0 - component.blend_to_animation)

                    for bone_idx, ik_transform in solve_result.bone_transforms.items():
                        if bone_idx in entity_pose:
                            original = entity_pose[bone_idx]
                            entity_pose[bone_idx] = self._blend_transforms(
                                original, ik_transform, blend_weight
                            )
                        else:
                            entity_pose[bone_idx] = ik_transform

                    # Update world transforms after applying this goal
                    self._world_transforms = self._compute_world_transforms(entity_pose)

                self._stats.goals_processed += 1

            result[entity] = entity_pose
            self._stats.entities_processed += 1

        return result

    def _solve_goal(
        self,
        goal: IKGoal,
        pose: Dict[int, Transform],
        component: IKComponent,
        dt: float
    ) -> IKSolveResult:
        """Solve a single IK goal.

        Determines the appropriate solver and dispatches to it.

        Args:
            goal: The IK goal to solve
            pose: Current pose transforms
            component: Parent IK component (for skeleton mapping)
            dt: Delta time

        Returns:
            IKSolveResult with modified transforms
        """
        # Determine solver type (auto-detect if AUTO)
        solver_type = self._determine_solver_type(goal)

        # Dispatch to appropriate solver
        if solver_type == IKSolverType.TWO_BONE:
            return self._solve_two_bone(goal, pose)
        elif solver_type == IKSolverType.FABRIK:
            return self._solve_fabrik(goal, pose)
        elif solver_type == IKSolverType.CCD:
            return self._solve_ccd(goal, pose)
        elif solver_type == IKSolverType.JACOBIAN:
            return self._solve_jacobian(goal, pose)
        elif solver_type == IKSolverType.FULL_BODY:
            return self._solve_fullbody(goal, pose, component)
        else:
            return IKSolveResult()

    def _determine_solver_type(self, goal: IKGoal) -> IKSolverType:
        """Determine the best solver type for a goal.

        Args:
            goal: The IK goal

        Returns:
            Appropriate solver type
        """
        if goal.solver_type != IKSolverType.AUTO:
            return goal.solver_type

        # Auto-detect based on chain length and characteristics
        if goal.chain_length == 2:
            # Two-bone chains (arms, legs) use analytical solver
            return IKSolverType.TWO_BONE
        elif goal.chain_length <= 5:
            # Medium chains (spines) use FABRIK
            return IKSolverType.FABRIK
        elif goal.chain_length <= 10:
            # Longer chains with potential constraints use CCD
            return IKSolverType.CCD
        else:
            # Very long or complex chains use Jacobian
            return IKSolverType.JACOBIAN

    def _solve_two_bone(self, goal: IKGoal, pose: Dict[int, Transform]) -> IKSolveResult:
        """Solve using analytical two-bone IK.

        Args:
            goal: IK goal (must be 2-bone chain)
            pose: Current pose

        Returns:
            IKSolveResult with rotations for the two bones
        """
        result = IKSolveResult(solver_type=IKSolverType.TWO_BONE)

        # Get chain bones
        chain = self._get_chain(goal.target_bone, 2)
        if len(chain) < 2:
            return result

        end_bone, mid_bone = chain[0], chain[1]
        root_bone = self._bone_hierarchy.get(mid_bone, -1)
        if root_bone < 0:
            return result

        # Get world positions
        root_tf = self._world_transforms.get(root_bone, Transform.identity())
        mid_tf = self._world_transforms.get(mid_bone, Transform.identity())
        end_tf = self._world_transforms.get(end_bone, Transform.identity())

        root_pos = root_tf.translation
        mid_pos = mid_tf.translation
        end_pos = end_tf.translation

        target = goal.target_position if goal.target_position else end_pos

        # Get or create solver
        solver_key = (root_bone, mid_bone, end_bone)
        if solver_key not in self._two_bone_solvers:
            self._two_bone_solvers[solver_key] = TwoBoneIK(
                root_bone=root_bone,
                mid_bone=mid_bone,
                end_bone=end_bone
            )

        solver = self._two_bone_solvers[solver_key]

        # Bone lengths
        upper_length = root_pos.distance(mid_pos)
        lower_length = mid_pos.distance(end_pos)
        total_length = upper_length + lower_length

        # Guard against zero-length bones
        min_length = IK_CONFIG.MIN_BONE_LENGTH if hasattr(IK_CONFIG, 'MIN_BONE_LENGTH') else 0.001
        if upper_length < min_length or lower_length < min_length or total_length < min_length * 2:
            # Cannot solve with zero-length bones, return identity
            return result

        # Vector from root to target
        to_target = target - root_pos
        target_dist = to_target.length()

        if target_dist < IK_CONFIG.MIN_TARGET_DISTANCE:
            return result

        # Clamp to reachable range
        target_dist = max(
            abs(upper_length - lower_length) + 0.001,
            min(total_length - 0.001, target_dist)
        )

        # Calculate joint angle using law of cosines
        denominator = 2 * upper_length * lower_length
        if abs(denominator) < min_length:
            return result  # Cannot compute with zero denominator

        cos_angle = (upper_length ** 2 + lower_length ** 2 - target_dist ** 2)
        cos_angle /= denominator
        cos_angle = max(-1.0, min(1.0, cos_angle))
        joint_angle = math.acos(cos_angle)

        # Direction to target
        if to_target.length_squared() < min_length * min_length:
            return result  # Target too close
        target_dir = to_target.normalized()

        # Pole vector handling
        pole_dir = Vec3(0, 1, 0)  # Default up
        if goal.hint_type == IKHintType.POSITION:
            pole_to_root = goal.hint_value - root_pos
            if pole_to_root.length_squared() > min_length * min_length:
                pole_dir = pole_to_root.normalized()
        elif goal.hint_type == IKHintType.DIRECTION:
            if goal.hint_value.length_squared() > min_length * min_length:
                pole_dir = goal.hint_value.normalized()

        # Calculate plane normal
        plane_normal = target_dir.cross(pole_dir)
        if plane_normal.length_squared() < 0.001:
            plane_normal = target_dir.cross(Vec3(0, 0, 1))
            if plane_normal.length_squared() < 0.001:
                plane_normal = target_dir.cross(Vec3(1, 0, 0))
        if plane_normal.length_squared() < min_length * min_length:
            return result  # Cannot determine plane
        plane_normal = plane_normal.normalized()

        # Calculate upper bone angle
        denominator_upper = 2 * upper_length * target_dist
        if abs(denominator_upper) < min_length:
            return result

        cos_upper = (upper_length ** 2 + target_dist ** 2 - lower_length ** 2)
        cos_upper /= denominator_upper
        cos_upper = max(-1.0, min(1.0, cos_upper))
        upper_angle = math.acos(cos_upper)

        # Calculate new positions
        upper_rot = Quat.from_axis_angle(plane_normal, upper_angle)
        upper_dir = upper_rot.rotate_vector(target_dir)
        new_mid_pos = root_pos + upper_dir * upper_length

        lower_dir = (target - new_mid_pos).normalized()
        new_end_pos = new_mid_pos + lower_dir * lower_length

        # Get local transforms
        root_local = pose.get(root_bone, Transform.identity())
        mid_local = pose.get(mid_bone, Transform.identity())
        end_local = pose.get(end_bone, Transform.identity())

        # Calculate rotations
        result.bone_transforms[mid_bone] = Transform(
            translation=mid_local.translation,
            rotation=self._rotation_to_direction(upper_dir, root_local.rotation),
            scale=mid_local.scale,
        )

        result.bone_transforms[end_bone] = Transform(
            translation=end_local.translation,
            rotation=self._rotation_to_direction(lower_dir, mid_local.rotation),
            scale=end_local.scale,
        )

        # Apply end effector rotation if specified
        if goal.target_rotation is not None:
            result.bone_transforms[end_bone] = Transform(
                translation=end_local.translation,
                rotation=goal.target_rotation,
                scale=end_local.scale,
            )

        result.converged = True
        result.iterations = 1
        result.final_error = new_end_pos.distance(target)

        self._stats.two_bone_solves += 1
        return result

    def _solve_fabrik(self, goal: IKGoal, pose: Dict[int, Transform]) -> IKSolveResult:
        """Solve using FABRIK algorithm.

        Args:
            goal: IK goal
            pose: Current pose

        Returns:
            IKSolveResult
        """
        result = IKSolveResult(solver_type=IKSolverType.FABRIK)

        chain = self._get_chain(goal.target_bone, goal.chain_length)
        if not chain:
            return result

        # Get chain world positions
        positions = [
            self._world_transforms.get(bone, Transform.identity()).translation
            for bone in chain
        ]
        lengths = [
            positions[i].distance(positions[i + 1])
            for i in range(len(positions) - 1)
        ]

        target = goal.target_position if goal.target_position else positions[0]
        root = positions[-1]

        # Check reachability
        total_length = sum(lengths)
        if root.distance(target) > total_length:
            # Unreachable - stretch toward target
            direction = (target - root).normalized()
            for i in range(len(positions) - 1, 0, -1):
                positions[i - 1] = positions[i] + direction * lengths[i - 1]
            result.converged = False
        else:
            # FABRIK iterations
            max_iters = goal.max_iterations or FABRIK_DEFAULT_MAX_ITERATIONS
            for iteration in range(max_iters):
                # Forward pass (end to root)
                positions[0] = target
                for i in range(len(positions) - 1):
                    direction = (positions[i + 1] - positions[i]).normalized()
                    positions[i + 1] = positions[i] + direction * lengths[i]

                # Backward pass (root to end)
                positions[-1] = root
                for i in range(len(positions) - 2, -1, -1):
                    direction = (positions[i] - positions[i + 1]).normalized()
                    positions[i] = positions[i + 1] + direction * lengths[i]

                # Check convergence
                error = positions[0].distance(target)
                if error < goal.position_tolerance:
                    result.converged = True
                    result.final_error = error
                    break

                result.iterations = iteration + 1

        # Convert positions to local transforms
        for i, bone in enumerate(chain):
            if i < len(chain) - 1:
                direction = (positions[i] - positions[i + 1]).normalized()
                rotation = self._rotation_to_direction(direction, Quat.identity())
            else:
                rotation = pose.get(bone, Transform.identity()).rotation

            local = pose.get(bone, Transform.identity())
            result.bone_transforms[bone] = Transform(
                translation=local.translation,
                rotation=rotation,
                scale=local.scale,
            )

        self._stats.fabrik_solves += 1
        return result

    def _solve_ccd(self, goal: IKGoal, pose: Dict[int, Transform]) -> IKSolveResult:
        """Solve using CCD algorithm.

        Args:
            goal: IK goal
            pose: Current pose

        Returns:
            IKSolveResult
        """
        result = IKSolveResult(solver_type=IKSolverType.CCD)

        chain = self._get_chain(goal.target_bone, goal.chain_length)
        if not chain:
            return result

        positions = [
            self._world_transforms.get(bone, Transform.identity()).translation
            for bone in chain
        ]
        target = goal.target_position if goal.target_position else positions[0]

        max_iters = goal.max_iterations or CCD_DEFAULT_MAX_ITERATIONS
        for iteration in range(max_iters):
            # Iterate from end effector toward root
            for i in range(len(chain) - 1):
                bone = chain[i + 1]  # Skip end effector
                bone_pos = positions[i + 1]
                end_pos = positions[0]

                # Vector from bone to end effector
                to_end = end_pos - bone_pos
                # Vector from bone to target
                to_target = target - bone_pos

                if to_end.length_squared() < 0.0001 or to_target.length_squared() < 0.0001:
                    continue

                to_end = to_end.normalized()
                to_target = to_target.normalized()

                # Calculate rotation
                dot = max(-1.0, min(1.0, to_end.dot(to_target)))
                angle = math.acos(dot)

                if abs(angle) > 0.0001:
                    axis = to_end.cross(to_target)
                    if axis.length_squared() > 0.0001:
                        axis = axis.normalized()
                        rotation = Quat.from_axis_angle(axis, angle)

                        # Rotate all positions from this bone to end
                        for j in range(i + 1):
                            relative = positions[j] - bone_pos
                            positions[j] = bone_pos + rotation.rotate_vector(relative)

            # Check convergence
            error = positions[0].distance(target)
            result.final_error = error
            result.iterations = iteration + 1

            if error < goal.position_tolerance:
                result.converged = True
                break

        # Convert to transforms
        for i, bone in enumerate(chain):
            local = pose.get(bone, Transform.identity())
            direction = Vec3(0, 0, 1)
            if i < len(chain) - 1:
                direction = (positions[i] - positions[i + 1]).normalized()

            result.bone_transforms[bone] = Transform(
                translation=local.translation,
                rotation=self._rotation_to_direction(direction, local.rotation),
                scale=local.scale,
            )

        self._stats.ccd_solves += 1
        return result

    def _solve_jacobian(self, goal: IKGoal, pose: Dict[int, Transform]) -> IKSolveResult:
        """Solve using Jacobian-based IK.

        Args:
            goal: IK goal
            pose: Current pose

        Returns:
            IKSolveResult
        """
        result = IKSolveResult(solver_type=IKSolverType.JACOBIAN)

        chain = self._get_chain(goal.target_bone, goal.chain_length)
        if len(chain) < 2:
            return result

        # Get or create solver
        chain_key = goal.target_bone
        if chain_key not in self._jacobian_solvers:
            self._jacobian_solvers[chain_key] = JacobianIK(
                chain_length=len(chain),
                tolerance=goal.position_tolerance,
                max_iterations=goal.max_iterations or JACOBIAN_DEFAULT_MAX_ITERATIONS,
                method=JacobianMethod.DAMPED_LEAST_SQUARES
            )

        solver = self._jacobian_solvers[chain_key]

        # Get current positions
        positions = [
            self._world_transforms.get(bone, Transform.identity()).translation
            for bone in chain
        ]

        target = goal.target_position if goal.target_position else positions[0]

        # Solve using Jacobian
        jacobian_result = solver.solve(
            positions=positions,
            target_position=target,
            bone_lengths=[
                positions[i].distance(positions[i + 1])
                for i in range(len(positions) - 1)
            ]
        )

        result.converged = jacobian_result.success
        result.iterations = jacobian_result.iterations
        result.final_error = jacobian_result.final_error

        # Convert rotations to transforms
        for i, bone in enumerate(chain):
            local = pose.get(bone, Transform.identity())
            if i < len(jacobian_result.rotations):
                result.bone_transforms[bone] = Transform(
                    translation=local.translation,
                    rotation=jacobian_result.rotations[i],
                    scale=local.scale,
                )
            else:
                result.bone_transforms[bone] = local

        self._stats.jacobian_solves += 1
        return result

    def _solve_fullbody(
        self,
        goal: IKGoal,
        pose: Dict[int, Transform],
        component: IKComponent
    ) -> IKSolveResult:
        """Solve using full-body IK.

        Args:
            goal: IK goal
            pose: Current pose
            component: IK component with skeleton mapping

        Returns:
            IKSolveResult
        """
        result = IKSolveResult(solver_type=IKSolverType.FULL_BODY)

        if component.skeleton_mapping is None:
            return result

        # Create or get full-body solver
        if self._fullbody_solver is None:
            self._fullbody_solver = FullBodyIK(
                skeleton_mapping=component.skeleton_mapping,
                tolerance=goal.position_tolerance,
                max_iterations=goal.max_iterations or FULLBODY_DEFAULT_MAX_ITERATIONS
            )

        # Create full-body goal
        fb_goal = FullBodyIKGoal(
            bone_index=goal.target_bone,
            target_position=goal.target_position,
            target_rotation=goal.target_rotation,
            position_weight=1.0 if goal.target_position else 0.0,
            rotation_weight=1.0 if goal.target_rotation else 0.0,
            priority=goal.priority,
            chain_type=goal.chain_name if goal.chain_name else None
        )

        # Convert pose to transform list
        max_bone = max(pose.keys()) if pose else 0
        transforms = [
            pose.get(i, Transform.identity())
            for i in range(max_bone + 1)
        ]

        # Solve
        fb_result = self._fullbody_solver.solve(transforms, [fb_goal])

        result.converged = fb_result.success
        result.final_error = fb_result.final_errors.get(goal.target_bone, float('inf'))

        # Convert back to dict
        for i, transform in enumerate(fb_result.transforms):
            if i in pose:
                result.bone_transforms[i] = transform

        self._stats.fullbody_solves += 1
        return result

    def _get_chain(self, end_bone: int, length: int) -> List[int]:
        """Get chain of bone indices from end to root.

        Args:
            end_bone: End effector bone index
            length: Desired chain length

        Returns:
            List of bone indices [end, ..., root]
        """
        chain = [end_bone]
        current = end_bone

        for _ in range(length - 1):
            parent = self._bone_hierarchy.get(current, -1)
            if parent < 0:
                break
            chain.append(parent)
            current = parent

        return chain

    def _compute_world_transforms(
        self,
        local_transforms: Dict[int, Transform]
    ) -> Dict[int, Transform]:
        """Compute world transforms from local transforms.

        Args:
            local_transforms: Bone index -> local Transform mapping

        Returns:
            Bone index -> world Transform mapping
        """
        world: Dict[int, Transform] = {}

        # Process bones in order (assumes parents have lower indices)
        for bone in sorted(local_transforms.keys()):
            local = local_transforms[bone]
            parent = self._bone_hierarchy.get(bone, -1)

            if parent < 0 or parent not in world:
                world[bone] = local
            else:
                parent_world = world[parent]
                world[bone] = Transform(
                    translation=parent_world.transform_point(local.translation),
                    rotation=parent_world.rotation * local.rotation,
                    scale=Vec3(
                        parent_world.scale.x * local.scale.x,
                        parent_world.scale.y * local.scale.y,
                        parent_world.scale.z * local.scale.z,
                    ),
                )

        return world

    def _blend_transforms(
        self,
        a: Transform,
        b: Transform,
        t: float
    ) -> Transform:
        """Blend between two transforms.

        Args:
            a: First transform
            b: Second transform
            t: Blend factor (0 = a, 1 = b)

        Returns:
            Blended transform
        """
        t = max(0.0, min(1.0, t))

        return Transform(
            translation=a.translation.lerp(b.translation, t),
            rotation=a.rotation.slerp(b.rotation, t),
            scale=a.scale.lerp(b.scale, t),
        )

    def _rotation_to_direction(self, direction: Vec3, base_rotation: Quat) -> Quat:
        """Calculate rotation to align with a direction.

        Args:
            direction: Target direction (will be normalized)
            base_rotation: Base rotation to combine with

        Returns:
            Rotation quaternion
        """
        forward = Vec3(0, 0, 1)
        dir_normalized = direction.normalized()

        dot = forward.dot(dir_normalized)
        if dot > 0.9999:
            return base_rotation
        if dot < -0.9999:
            return Quat.from_axis_angle(Vec3(0, 1, 0), math.pi) * base_rotation

        axis = forward.cross(dir_normalized)
        if axis.length_squared() > 0.0001:
            axis = axis.normalized()
            angle = math.acos(max(-1.0, min(1.0, dot)))
            return Quat.from_axis_angle(axis, angle) * base_rotation

        return base_rotation

    def get_stats(self) -> "IKSystemStats":
        """Get current frame statistics.

        Returns:
            IKSystemStats with solve counts and timing
        """
        return self._stats


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class IKSystemStats:
    """Statistics for the IK system.

    Tracks solve counts and performance metrics per frame.
    """

    entities_processed: int = 0
    goals_processed: int = 0
    invalid_goals: int = 0
    two_bone_solves: int = 0
    fabrik_solves: int = 0
    ccd_solves: int = 0
    jacobian_solves: int = 0
    fullbody_solves: int = 0
    total_time_ms: float = 0.0

    def reset_frame(self) -> None:
        """Reset all counters for new frame."""
        self.entities_processed = 0
        self.goals_processed = 0
        self.invalid_goals = 0
        self.two_bone_solves = 0
        self.fabrik_solves = 0
        self.ccd_solves = 0
        self.jacobian_solves = 0
        self.fullbody_solves = 0
        self.total_time_ms = 0.0

    @property
    def total_solves(self) -> int:
        """Total number of solves across all solver types."""
        return (
            self.two_bone_solves +
            self.fabrik_solves +
            self.ccd_solves +
            self.jacobian_solves +
            self.fullbody_solves
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # System decorator
    "system",
    # Enums
    "IKSolverType",
    "IKHintType",
    # Data structures
    "IKGoal",
    "IKChainBone",
    "IKSolveResult",
    # Component
    "IKComponent",
    # System
    "IKSystem",
    # Stats
    "IKSystemStats",
]
