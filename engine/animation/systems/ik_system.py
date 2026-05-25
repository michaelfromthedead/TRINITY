"""ECS system for Inverse Kinematics.

Processes entities with IK components, solving IK chains to match goals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import IK_CONFIG


class IKSolverType(Enum):
    """Type of IK solver to use."""
    TWO_BONE = auto()  # Simple two-bone (limb) solver
    FABRIK = auto()  # Forward And Backward Reaching IK
    CCD = auto()  # Cyclic Coordinate Descent
    JACOBIAN = auto()  # Jacobian-based (for complex chains)


class IKHintType(Enum):
    """Type of IK hint/pole vector."""
    NONE = auto()
    POSITION = auto()  # Pole vector as position
    DIRECTION = auto()  # Pole vector as direction


@dataclass
class IKGoal:
    """IK goal definition.

    Attributes:
        target_bone: Index of end effector bone
        target_position: Target world position
        target_rotation: Target world rotation (optional)
        weight: Blend weight (0-1)
        chain_length: Number of bones in chain
        solver_type: Type of solver to use
        hint_type: Type of pole vector hint
        hint_value: Pole vector value
        position_tolerance: Position error tolerance
        rotation_tolerance: Rotation error tolerance
        max_iterations: Maximum solver iterations
    """
    target_bone: int = -1
    target_position: Vec3 = field(default_factory=Vec3.zero)
    target_rotation: Quat | None = None
    weight: float = 1.0
    chain_length: int = 2
    solver_type: IKSolverType = IKSolverType.TWO_BONE
    hint_type: IKHintType = IKHintType.NONE
    hint_value: Vec3 = field(default_factory=Vec3.zero)
    position_tolerance: float = IK_CONFIG.DEFAULT_POSITION_TOLERANCE
    rotation_tolerance: float = IK_CONFIG.DEFAULT_ROTATION_TOLERANCE
    max_iterations: int = IK_CONFIG.DEFAULT_MAX_ITERATIONS
    enabled: bool = True

    def set_target(self, position: Vec3, rotation: Quat | None = None) -> None:
        """Set target position and optional rotation."""
        self.target_position = position
        self.target_rotation = rotation


@dataclass
class IKChainBone:
    """Bone in an IK chain."""
    bone_index: int
    length: float
    local_transform: Transform
    world_position: Vec3 = field(default_factory=Vec3.zero)


@dataclass
class IKSolveResult:
    """Result of IK solve operation."""
    converged: bool = False
    iterations: int = 0
    final_error: float = 0.0
    bone_transforms: dict[int, Transform] = field(default_factory=dict)


@dataclass
class IKComponent:
    """Component for entities with IK needs.

    Attributes:
        goals: List of IK goals
        enabled: Whether IK is enabled
        blend_to_animation: Blend factor between IK and animation (0 = full IK, 1 = full anim)
    """
    goals: list[IKGoal] = field(default_factory=list)
    enabled: bool = True
    blend_to_animation: float = 0.0

    def add_goal(self, goal: IKGoal) -> int:
        """Add IK goal, returns index."""
        self.goals.append(goal)
        return len(self.goals) - 1

    def remove_goal(self, index: int) -> bool:
        """Remove goal by index."""
        if 0 <= index < len(self.goals):
            self.goals.pop(index)
            return True
        return False

    def get_goal(self, index: int) -> IKGoal | None:
        """Get goal by index."""
        if 0 <= index < len(self.goals):
            return self.goals[index]
        return None

    def set_goal_target(self, index: int, position: Vec3, rotation: Quat | None = None) -> bool:
        """Set goal target position/rotation."""
        goal = self.get_goal(index)
        if goal:
            goal.set_target(position, rotation)
            return True
        return False


class IKSystem:
    """ECS system for solving IK.

    Processes entities with IKComponent after animation graph evaluation.
    """

    def __init__(self):
        self._bone_hierarchy: dict[int, int] = {}  # bone -> parent
        self._bone_lengths: dict[int, float] = {}
        self._world_transforms: dict[int, Transform] = {}

    def set_skeleton_data(
        self,
        hierarchy: dict[int, int],
        bone_lengths: dict[int, float]
    ) -> None:
        """Set skeleton data for IK solving.

        Args:
            hierarchy: Bone index -> parent index mapping
            bone_lengths: Bone index -> bone length mapping
        """
        self._bone_hierarchy = hierarchy
        self._bone_lengths = bone_lengths

    def update(
        self,
        world: World,
        entity_components: list[tuple[Entity, IKComponent]],
        pose_data: dict[Entity, dict[int, Transform]]
    ) -> dict[Entity, dict[int, Transform]]:
        """Update all IK components.

        Args:
            world: ECS world
            entity_components: List of (entity, component) tuples
            pose_data: Current poses for entities (entity -> bone transforms)

        Returns:
            Updated pose data with IK applied
        """
        result = {}

        for entity, component in entity_components:
            if not component.enabled:
                result[entity] = pose_data.get(entity, {})
                continue

            entity_pose = pose_data.get(entity, {})
            self._world_transforms = self._compute_world_transforms(entity_pose)

            for goal in component.goals:
                if not goal.enabled or goal.weight <= 0:
                    continue

                solve_result = self._solve_goal(goal, entity_pose)

                if solve_result.converged or solve_result.iterations > 0:
                    # Blend IK result with original pose
                    blend_weight = goal.weight * (1.0 - component.blend_to_animation)
                    for bone_idx, ik_transform in solve_result.bone_transforms.items():
                        original = entity_pose.get(bone_idx, Transform.identity())
                        entity_pose[bone_idx] = original.lerp(ik_transform, blend_weight)

            result[entity] = entity_pose

        return result

    def _solve_goal(self, goal: IKGoal, pose: dict[int, Transform]) -> IKSolveResult:
        """Solve single IK goal."""
        if goal.solver_type == IKSolverType.TWO_BONE:
            return self._solve_two_bone(goal, pose)
        elif goal.solver_type == IKSolverType.FABRIK:
            return self._solve_fabrik(goal, pose)
        elif goal.solver_type == IKSolverType.CCD:
            return self._solve_ccd(goal, pose)
        else:
            return IKSolveResult()

    def _solve_two_bone(self, goal: IKGoal, pose: dict[int, Transform]) -> IKSolveResult:
        """Solve two-bone IK (e.g., arm, leg)."""
        result = IKSolveResult()

        # Get chain bones
        chain = self._get_chain(goal.target_bone, 2)
        if len(chain) < 2:
            return result

        end_bone, mid_bone = chain[0], chain[1]
        root_bone = self._bone_hierarchy.get(mid_bone, -1)
        if root_bone < 0:
            return result

        # Get world positions
        root_pos = self._world_transforms.get(root_bone, Transform.identity()).translation
        mid_pos = self._world_transforms.get(mid_bone, Transform.identity()).translation
        end_pos = self._world_transforms.get(end_bone, Transform.identity()).translation

        target = goal.target_position

        # Bone lengths
        upper_length = root_pos.distance(mid_pos)
        lower_length = mid_pos.distance(end_pos)
        total_length = upper_length + lower_length

        # Vector from root to target
        to_target = target - root_pos
        target_dist = to_target.length()

        if target_dist < IK_CONFIG.MIN_TARGET_DISTANCE:
            return result

        # Clamp to reachable range
        target_dist = max(abs(upper_length - lower_length) + 0.001,
                         min(total_length - 0.001, target_dist))

        # Calculate joint angle using law of cosines
        # cos(angle) = (a^2 + b^2 - c^2) / (2ab)
        cos_angle = (upper_length**2 + lower_length**2 - target_dist**2) / (2 * upper_length * lower_length)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        joint_angle = math.acos(cos_angle)

        # Direction to target
        target_dir = to_target.normalized()

        # Calculate pole vector influence
        pole_dir = Vec3.up()
        if goal.hint_type == IKHintType.POSITION:
            pole_dir = (goal.hint_value - root_pos).normalized()
        elif goal.hint_type == IKHintType.DIRECTION:
            pole_dir = goal.hint_value.normalized()

        # Calculate plane normal (perpendicular to target direction and pole)
        plane_normal = target_dir.cross(pole_dir).normalized()
        if plane_normal.length_squared() < 0.001:
            # Fallback if parallel
            plane_normal = target_dir.cross(Vec3.forward())
            if plane_normal.length_squared() < 0.001:
                plane_normal = target_dir.cross(Vec3.right())
            plane_normal = plane_normal.normalized()

        # Calculate upper bone angle
        cos_upper = (upper_length**2 + target_dist**2 - lower_length**2) / (2 * upper_length * target_dist)
        cos_upper = max(-1.0, min(1.0, cos_upper))
        upper_angle = math.acos(cos_upper)

        # Calculate new positions
        # Rotate target_dir around plane_normal by upper_angle
        upper_rot = Quat.from_axis_angle(plane_normal, upper_angle)
        upper_dir = upper_rot.rotate_vector(target_dir)
        new_mid_pos = root_pos + upper_dir * upper_length

        # Lower bone points toward target
        lower_dir = (target - new_mid_pos).normalized()
        new_end_pos = new_mid_pos + lower_dir * lower_length

        # Convert to local transforms
        root_local = pose.get(root_bone, Transform.identity())
        mid_local = pose.get(mid_bone, Transform.identity())
        end_local = pose.get(end_bone, Transform.identity())

        # Calculate rotations to achieve positions
        # This is simplified - proper implementation would compute rotation deltas
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

        return result

    def _solve_fabrik(self, goal: IKGoal, pose: dict[int, Transform]) -> IKSolveResult:
        """Solve IK using FABRIK algorithm."""
        result = IKSolveResult()

        chain = self._get_chain(goal.target_bone, goal.chain_length)
        if not chain:
            return result

        # Get chain world positions
        positions = [self._world_transforms.get(bone, Transform.identity()).translation for bone in chain]
        lengths = [positions[i].distance(positions[i+1]) for i in range(len(positions)-1)]

        target = goal.target_position
        root = positions[-1]

        # Check reachability
        total_length = sum(lengths)
        if root.distance(target) > total_length:
            # Unreachable - stretch toward target
            direction = (target - root).normalized()
            for i in range(len(positions) - 1, 0, -1):
                positions[i-1] = positions[i] + direction * lengths[i-1]
            result.converged = False
        else:
            # FABRIK iterations
            for iteration in range(goal.max_iterations):
                # Forward pass
                positions[0] = target
                for i in range(len(positions) - 1):
                    direction = (positions[i+1] - positions[i]).normalized()
                    positions[i+1] = positions[i] + direction * lengths[i]

                # Backward pass
                positions[-1] = root
                for i in range(len(positions) - 2, -1, -1):
                    direction = (positions[i] - positions[i+1]).normalized()
                    positions[i] = positions[i+1] + direction * lengths[i]

                # Check convergence
                error = positions[0].distance(target)
                if error < goal.position_tolerance:
                    result.converged = True
                    result.final_error = error
                    break

                result.iterations = iteration + 1

        # Convert positions to transforms
        for i, bone in enumerate(chain):
            if i < len(chain) - 1:
                direction = (positions[i] - positions[i+1]).normalized()
                rotation = self._rotation_to_direction(direction, Quat.identity())
            else:
                rotation = pose.get(bone, Transform.identity()).rotation

            local = pose.get(bone, Transform.identity())
            result.bone_transforms[bone] = Transform(
                translation=local.translation,
                rotation=rotation,
                scale=local.scale,
            )

        return result

    def _solve_ccd(self, goal: IKGoal, pose: dict[int, Transform]) -> IKSolveResult:
        """Solve IK using CCD algorithm."""
        result = IKSolveResult()

        chain = self._get_chain(goal.target_bone, goal.chain_length)
        if not chain:
            return result

        positions = [self._world_transforms.get(bone, Transform.identity()).translation for bone in chain]
        target = goal.target_position

        for iteration in range(goal.max_iterations):
            # Iterate from end effector to root
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

                # Calculate rotation axis and angle
                dot = to_end.dot(to_target)
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)

                if abs(angle) > 0.0001:
                    axis = to_end.cross(to_target).normalized()
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
            direction = Vec3.forward()
            if i < len(chain) - 1:
                direction = (positions[i] - positions[i+1]).normalized()

            result.bone_transforms[bone] = Transform(
                translation=local.translation,
                rotation=self._rotation_to_direction(direction, local.rotation),
                scale=local.scale,
            )

        return result

    def _get_chain(self, end_bone: int, length: int) -> list[int]:
        """Get chain of bone indices from end to root."""
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
        local_transforms: dict[int, Transform]
    ) -> dict[int, Transform]:
        """Compute world transforms from local transforms."""
        world = {}

        # Simple implementation - assumes bones are ordered parent-first
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

    def _rotation_to_direction(self, direction: Vec3, base_rotation: Quat) -> Quat:
        """Calculate rotation to align with direction."""
        forward = Vec3.forward()
        dir_normalized = direction.normalized()

        dot = forward.dot(dir_normalized)
        if dot > 0.9999:
            return base_rotation
        if dot < -0.9999:
            return Quat.from_axis_angle(Vec3.up(), math.pi) * base_rotation

        axis = forward.cross(dir_normalized).normalized()
        angle = math.acos(max(-1.0, min(1.0, dot)))

        return Quat.from_axis_angle(axis, angle) * base_rotation
