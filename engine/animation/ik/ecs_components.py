"""ECS Components for Animation IK Systems.

This module provides ECS components for integrating IK systems with the
Trinity entity-component-system architecture. Components are designed as
dataclasses for easy serialization and runtime modification.

Components:
    - FullBodyIKController: Full body IK with balance and reach
    - AnimationGraphController: Animation graph with IK layers
    - LookAtTarget: Look-at constraint specification
    - FootPlacementController: Terrain-adaptive foot IK

Example usage:

    from engine.animation.ik.ecs_components import (
        FullBodyIKController,
        AnimationGraphController,
        LookAtTarget,
        FootPlacementController,
    )

    # Create entity with IK components
    entity = world.create_entity()
    world.add_component(entity, FullBodyIKController(
        enabled=True,
        weight=1.0,
        foot_placement_enabled=True,
        look_at_enabled=True,
    ))

    # Set look-at target at runtime
    look_at = world.get_component(entity, LookAtTarget)
    look_at.target_position = enemy.position
    look_at.blend_weight = 0.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from trinity.decorators import component

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform

if TYPE_CHECKING:
    from engine.animation.ik.fullbody import FullBodyIK, LookAtSolver
    from engine.animation.ik.foot_placement import FootPlacement
    from engine.animation.ik.ik_layer import IKLayer, IKGoalContext
    from engine.animation.ik.graph_integration import AnimationIKController


# =============================================================================
# FULL BODY IK CONTROLLER
# =============================================================================


@component
@dataclass
class FullBodyIKController:
    """ECS component for full body IK control.

    Manages full body inverse kinematics including limb chains, spine,
    foot placement, and look-at constraints. This component wraps the
    FullBodyIK solver and provides runtime control over IK behavior.

    The component supports multiple IK subsystems:
    - Full body solver for multi-effector IK
    - Foot placement for terrain adaptation
    - Look-at for head/eye tracking

    Attributes:
        solver: FullBodyIK solver instance (created on demand).
        enabled: Master enable flag for all IK processing.
        weight: Global IK blend weight (0-1).
        foot_placement: FootPlacement solver instance.
        foot_placement_enabled: Enable terrain-adaptive foot IK.
        look_at_solver: LookAtSolver for head tracking.
        look_at_target: World position to look at.
        look_at_enabled: Enable look-at constraint.
        look_at_weight: Look-at blend weight (0-1).
        ik_goals: Goal context for dynamic targets.
        maintain_balance: Enable COM-based balance maintenance.
        pelvis_adjust_enabled: Enable automatic pelvis height adjustment.
        max_pelvis_drop: Maximum pelvis drop distance.
        solve_order: Order of IK solve phases.

    Example:
        controller = FullBodyIKController(
            enabled=True,
            weight=1.0,
            foot_placement_enabled=True,
            look_at_enabled=True,
            look_at_weight=0.7,
        )
        entity.add_component(controller)

        # Each frame:
        controller.look_at_target = enemy.head_position
    """

    # Class-level attribute for Trinity component system
    _component_name: str = "FullBodyIKController"

    # Core IK solver
    solver: Optional['FullBodyIK'] = None
    enabled: bool = True
    weight: float = 1.0

    # Foot placement subsystem
    foot_placement: Optional['FootPlacement'] = None
    foot_placement_enabled: bool = True
    foot_height_offset: float = 0.0

    # Look-at subsystem
    look_at_solver: Optional['LookAtSolver'] = None
    look_at_target: Optional[Vec3] = None
    look_at_enabled: bool = False
    look_at_weight: float = 1.0
    look_at_max_angle: float = 90.0  # degrees

    # Goals
    ik_goals: Optional['IKGoalContext'] = None

    # Balance and pelvis
    maintain_balance: bool = True
    pelvis_adjust_enabled: bool = True
    max_pelvis_drop: float = 0.5

    # Solve configuration
    solve_order: List[str] = field(default_factory=lambda: [
        "foot_placement", "spine", "arms", "look_at"
    ])

    # Runtime state (not serialized)
    _last_solve_time: float = 0.0
    _pelvis_offset: Vec3 = field(default_factory=Vec3.zero)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all IK processing.

        Args:
            enabled: Whether IK should be active.
        """
        self.enabled = enabled

    def set_weight(self, weight: float) -> None:
        """Set global IK blend weight.

        Args:
            weight: Weight value (clamped to 0-1).
        """
        self.weight = max(0.0, min(1.0, weight))

    def set_look_at_target(
        self,
        position: Optional[Vec3],
        weight: float = 1.0,
        immediate: bool = False
    ) -> None:
        """Set the look-at target position.

        Args:
            position: World position to look at, or None to clear.
            weight: Look-at blend weight (0-1).
            immediate: If True, skip blending to new target.
        """
        self.look_at_target = position
        if immediate:
            self.look_at_weight = max(0.0, min(1.0, weight))
        else:
            # Weight will be blended by the system
            self.look_at_weight = max(0.0, min(1.0, weight))

    def clear_look_at(self) -> None:
        """Clear the look-at target."""
        self.look_at_target = None
        self.look_at_enabled = False

    def set_foot_placement_enabled(self, enabled: bool) -> None:
        """Enable or disable foot placement IK.

        Args:
            enabled: Whether foot placement should be active.
        """
        self.foot_placement_enabled = enabled

    def reset(self) -> None:
        """Reset component to initial state."""
        self.weight = 1.0
        self.look_at_target = None
        self.look_at_weight = 1.0
        self.look_at_enabled = False
        self._last_solve_time = 0.0
        self._pelvis_offset = Vec3.zero()


# =============================================================================
# ANIMATION GRAPH CONTROLLER
# =============================================================================


@component
@dataclass
class AnimationGraphController:
    """ECS component for animation graph with IK integration.

    Manages the animation graph evaluation pipeline with integrated
    IK layer support. This component bridges the animation graph system
    with IK post-processing.

    Pipeline:
        1. Animation graph evaluates base pose
        2. IK layers apply corrections in order
        3. Final pose is output for skinning

    Attributes:
        controller: AnimationIKController instance.
        enabled: Master enable flag.
        graph: Reference to animation graph (external).
        layers: Ordered list of IK layers to apply.
        current_state: Current animation state name.
        blend_time: Time remaining in current blend.
        output_transforms: Final bone transforms after IK.
        ik_weight: Global IK weight multiplier.
        solve_order: IK solve order mode.
        custom_solve_order: Custom layer order (if using custom mode).

    Example:
        controller = AnimationGraphController(
            enabled=True,
            ik_weight=1.0,
        )
        entity.add_component(controller)

        # Add IK layers
        controller.add_layer(foot_ik_layer)
        controller.add_layer(hand_ik_layer)

        # Each frame (handled by system):
        transforms = controller.output_transforms
    """

    # Class-level attribute for Trinity component system
    _component_name: str = "AnimationGraphController"

    # Core controller
    controller: Optional['AnimationIKController'] = None
    enabled: bool = True

    # Animation graph reference (external, not owned)
    graph: Optional[Any] = None  # AnimationGraph

    # IK layer stack
    layers: List['IKLayer'] = field(default_factory=list)

    # Animation state
    current_state: str = ""
    blend_time: float = 0.0
    time_scale: float = 1.0

    # Output
    output_transforms: List[Transform] = field(default_factory=list)

    # IK configuration
    ik_weight: float = 1.0
    solve_order: str = "foot_first"  # "foot_first", "fullbody_first", "parallel", "custom"
    custom_solve_order: List[str] = field(default_factory=list)

    # Goal sources
    goal_source_names: List[str] = field(default_factory=list)

    # Runtime state
    _frame_count: int = 0
    _last_update_time: float = 0.0

    def add_layer(self, layer: 'IKLayer') -> int:
        """Add an IK layer to the stack.

        Args:
            layer: IK layer to add.

        Returns:
            Index where layer was added.
        """
        self.layers.append(layer)
        return len(self.layers) - 1

    def remove_layer(self, name: str) -> bool:
        """Remove an IK layer by name.

        Args:
            name: Name of the layer to remove.

        Returns:
            True if layer was found and removed.
        """
        for i, layer in enumerate(self.layers):
            if layer.name == name:
                self.layers.pop(i)
                return True
        return False

    def get_layer(self, name: str) -> Optional['IKLayer']:
        """Get an IK layer by name.

        Args:
            name: Layer name.

        Returns:
            Layer or None if not found.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def set_layer_weight(self, name: str, weight: float) -> bool:
        """Set weight for a specific layer.

        Args:
            name: Layer name.
            weight: New weight (0-1).

        Returns:
            True if layer was found.
        """
        layer = self.get_layer(name)
        if layer:
            layer.set_weight(weight)
            return True
        return False

    def set_layer_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a specific layer.

        Args:
            name: Layer name.
            enabled: Whether layer should be active.

        Returns:
            True if layer was found.
        """
        layer = self.get_layer(name)
        if layer:
            layer.set_enabled(enabled)
            return True
        return False

    def set_ik_weight(self, weight: float) -> None:
        """Set global IK weight.

        Args:
            weight: Weight value (clamped to 0-1).
        """
        self.ik_weight = max(0.0, min(1.0, weight))

    def set_solve_order(self, order: str, custom_order: Optional[List[str]] = None) -> None:
        """Set the IK solve order.

        Args:
            order: Solve order mode ("foot_first", "fullbody_first", "parallel", "custom").
            custom_order: Layer names for custom order mode.
        """
        self.solve_order = order
        if custom_order is not None:
            self.custom_solve_order = list(custom_order)

    def layer_count(self) -> int:
        """Get the number of IK layers.

        Returns:
            Number of layers.
        """
        return len(self.layers)

    def reset(self) -> None:
        """Reset component to initial state."""
        self.layers.clear()
        self.output_transforms.clear()
        self.current_state = ""
        self.blend_time = 0.0
        self.ik_weight = 1.0
        self._frame_count = 0
        self._last_update_time = 0.0


# =============================================================================
# LOOK-AT TARGET
# =============================================================================


@component
@dataclass
class LookAtTarget:
    """ECS component for look-at target specification.

    Defines the target for look-at constraints. Can specify either
    a world position or another entity to track. Supports spine
    rotation distribution for natural head movement.

    Attributes:
        target_position: World position to look at.
        target_entity: Entity ID to track (alternative to position).
        blend_weight: Overall look-at blend weight (0-1).
        max_rotation: Maximum rotation angle in degrees.
        spine_distribution: How rotation is distributed across spine bones.
            Sum should equal 1.0. [spine1, spine2, neck, head]
        horizontal_limit: Maximum horizontal rotation (degrees).
        vertical_limit: Maximum vertical rotation (degrees).
        blend_speed: Speed of weight transitions (per second).
        enabled: Whether look-at is active.
        priority: Priority for competing look-at targets.

    Example:
        target = LookAtTarget(
            target_position=enemy.head_position,
            blend_weight=0.8,
            max_rotation=70.0,
            spine_distribution=[0.1, 0.2, 0.3, 0.4],
        )
        entity.add_component(target)

        # Track another entity instead
        target.target_entity = enemy.entity_id
        target.target_position = None
    """

    # Class-level attribute for Trinity component system
    _component_name: str = "LookAtTarget"

    # Target specification
    target_position: Optional[Vec3] = None
    target_entity: Optional[int] = None  # Entity ID

    # Blend control
    blend_weight: float = 1.0
    enabled: bool = True
    priority: int = 0

    # Rotation limits
    max_rotation: float = 90.0  # degrees
    horizontal_limit: float = 120.0  # degrees
    vertical_limit: float = 60.0  # degrees

    # Spine distribution: [lower_spine, upper_spine, neck, head]
    # Sum should equal 1.0
    spine_distribution: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])

    # Smoothing
    blend_speed: float = 5.0  # weight units per second

    # Runtime state
    _current_weight: float = 0.0
    _last_target: Optional[Vec3] = None

    def set_target_position(self, position: Vec3, weight: float = 1.0) -> None:
        """Set a world position to look at.

        Args:
            position: World position.
            weight: Look-at weight (0-1).
        """
        self.target_position = position
        self.target_entity = None
        self.blend_weight = max(0.0, min(1.0, weight))
        self.enabled = True

    def set_target_entity(self, entity_id: int, weight: float = 1.0) -> None:
        """Set an entity to track.

        Args:
            entity_id: Entity ID to track.
            weight: Look-at weight (0-1).
        """
        self.target_entity = entity_id
        self.target_position = None
        self.blend_weight = max(0.0, min(1.0, weight))
        self.enabled = True

    def clear_target(self) -> None:
        """Clear the look-at target."""
        self.target_position = None
        self.target_entity = None
        self.enabled = False

    def set_spine_distribution(self, distribution: List[float]) -> None:
        """Set spine rotation distribution.

        The distribution defines how much rotation each spine segment
        contributes to the look-at. The list should have 3-4 elements
        (spine bones + neck + head) and sum to 1.0.

        Args:
            distribution: Weight per spine segment.

        Raises:
            ValueError: If distribution doesn't sum to approximately 1.0.
        """
        total = sum(distribution)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Spine distribution must sum to 1.0, got {total}"
            )
        self.spine_distribution = list(distribution)

    def has_target(self) -> bool:
        """Check if a target is set.

        Returns:
            True if position or entity target is set.
        """
        return self.target_position is not None or self.target_entity is not None


# =============================================================================
# FOOT PLACEMENT CONTROLLER
# =============================================================================


@component
@dataclass
class FootPlacementController:
    """ECS component for foot placement IK.

    Controls terrain-adaptive foot IK including ground detection,
    pelvis adjustment, and foot rotation alignment.

    Attributes:
        placement: FootPlacement solver instance.
        enabled: Whether foot placement is active.
        terrain_layer_mask: Physics layer mask for raycasting.
        max_step_height: Maximum height difference for feet.
        raycast_offset: Vertical offset for raycast origin.
        foot_height: Height of foot above ground.
        blend_speed: Smoothing speed for transitions.
        pelvis_adjust_enabled: Enable pelvis height adjustment.
        max_pelvis_drop: Maximum pelvis drop distance.
        max_pelvis_raise: Maximum pelvis raise distance.
        toe_align_weight: Weight for toe-to-terrain alignment.
        left_foot_offset: Additional height offset for left foot.
        right_foot_offset: Additional height offset for right foot.

    Example:
        controller = FootPlacementController(
            enabled=True,
            terrain_layer_mask=1,
            max_step_height=0.5,
            toe_align_weight=0.8,
        )
        entity.add_component(controller)
    """

    # Class-level attribute for Trinity component system
    _component_name: str = "FootPlacementController"

    # Core solver
    placement: Optional['FootPlacement'] = None
    enabled: bool = True

    # Raycast configuration
    terrain_layer_mask: int = 1
    raycast_offset: float = 1.0
    raycast_length: float = 2.0

    # Step configuration
    max_step_height: float = 0.5
    foot_height: float = 0.05

    # Smoothing
    blend_speed: float = 10.0

    # Pelvis adjustment
    pelvis_adjust_enabled: bool = True
    max_pelvis_drop: float = 0.5
    max_pelvis_raise: float = 0.3

    # Toe alignment
    toe_align_weight: float = 1.0

    # Per-foot offsets
    left_foot_offset: float = 0.0
    right_foot_offset: float = 0.0

    # Raycast callback (set by system)
    _raycast_callback: Optional[Callable] = None

    # Runtime state
    _left_planted: bool = True
    _right_planted: bool = True
    _current_pelvis_offset: float = 0.0
    _terrain_slope: float = 0.0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable foot placement.

        Args:
            enabled: Whether foot placement should be active.
        """
        self.enabled = enabled

    def set_foot_offset(self, foot: str, offset: float) -> None:
        """Set height offset for a specific foot.

        Args:
            foot: "left" or "right".
            offset: Height offset value.
        """
        if foot == "left":
            self.left_foot_offset = offset
        elif foot == "right":
            self.right_foot_offset = offset

    def set_terrain_layer_mask(self, mask: int) -> None:
        """Set the physics layer mask for terrain raycasting.

        Args:
            mask: Bit mask for terrain layers.
        """
        self.terrain_layer_mask = mask

    def set_blend_speed(self, speed: float) -> None:
        """Set transition smoothing speed.

        Args:
            speed: Blend speed (clamped to minimum 0.1).
        """
        self.blend_speed = max(0.1, speed)

    def get_terrain_slope(self) -> float:
        """Get the estimated terrain slope under the character.

        Returns:
            Slope angle in radians.
        """
        return self._terrain_slope

    def is_left_foot_planted(self) -> bool:
        """Check if left foot is on ground.

        Returns:
            True if left foot has ground contact.
        """
        return self._left_planted

    def is_right_foot_planted(self) -> bool:
        """Check if right foot is on ground.

        Returns:
            True if right foot has ground contact.
        """
        return self._right_planted

    def reset(self) -> None:
        """Reset component to initial state."""
        self._left_planted = True
        self._right_planted = True
        self._current_pelvis_offset = 0.0
        self._terrain_slope = 0.0


# =============================================================================
# IK TARGET COMPONENT
# =============================================================================


@component
@dataclass
class IKTargetComponent:
    """ECS component for specifying IK targets.

    Generic component for entities that serve as IK targets (e.g.,
    objects to grab, positions to reach). Can be attached to world
    objects that characters should interact with.

    Attributes:
        position_goals: Target positions keyed by bone name.
        rotation_goals: Target rotations keyed by bone name.
        weights: Per-goal weights keyed by bone name.
        pole_vectors: Pole vector positions for limb IK.
        active: Whether targets are active.
        priority: Priority when multiple targets compete.

    Example:
        # Door handle that character reaches for
        handle_target = IKTargetComponent()
        handle_target.set_position_goal("RightHand", handle_position, 1.0)
        door_entity.add_component(handle_target)
    """

    # Class-level attribute for Trinity component system
    _component_name: str = "IKTargetComponent"

    # Goals by bone name
    position_goals: Dict[str, Vec3] = field(default_factory=dict)
    rotation_goals: Dict[str, Quat] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    pole_vectors: Dict[str, Vec3] = field(default_factory=dict)

    # State
    active: bool = True
    priority: int = 0

    # Chain assignments
    chain_assignments: Dict[str, str] = field(default_factory=dict)

    def set_position_goal(
        self,
        bone_name: str,
        position: Vec3,
        weight: float = 1.0,
        chain_type: Optional[str] = None
    ) -> None:
        """Set a position goal for a bone.

        Args:
            bone_name: Target bone name (e.g., "LeftHand").
            position: World position target.
            weight: Goal weight (0-1).
            chain_type: Optional chain assignment (e.g., "left_arm").
        """
        self.position_goals[bone_name] = position
        self.weights[bone_name] = max(0.0, min(1.0, weight))
        if chain_type:
            self.chain_assignments[bone_name] = chain_type

    def set_rotation_goal(
        self,
        bone_name: str,
        rotation: Quat,
        weight: float = 1.0
    ) -> None:
        """Set a rotation goal for a bone.

        Args:
            bone_name: Target bone name.
            rotation: Target rotation.
            weight: Goal weight (0-1).
        """
        self.rotation_goals[bone_name] = rotation
        self.weights[bone_name] = max(0.0, min(1.0, weight))

    def set_pole_vector(self, bone_name: str, position: Vec3) -> None:
        """Set a pole vector for limb IK.

        Args:
            bone_name: Effector bone name.
            position: Pole vector world position.
        """
        self.pole_vectors[bone_name] = position

    def remove_goal(self, bone_name: str) -> bool:
        """Remove all goals for a bone.

        Args:
            bone_name: Bone to clear goals for.

        Returns:
            True if any goals were removed.
        """
        removed = False
        if bone_name in self.position_goals:
            del self.position_goals[bone_name]
            removed = True
        if bone_name in self.rotation_goals:
            del self.rotation_goals[bone_name]
            removed = True
        if bone_name in self.weights:
            del self.weights[bone_name]
        if bone_name in self.pole_vectors:
            del self.pole_vectors[bone_name]
        if bone_name in self.chain_assignments:
            del self.chain_assignments[bone_name]
        return removed

    def clear(self) -> None:
        """Clear all goals."""
        self.position_goals.clear()
        self.rotation_goals.clear()
        self.weights.clear()
        self.pole_vectors.clear()
        self.chain_assignments.clear()

    def has_goals(self) -> bool:
        """Check if any goals are set.

        Returns:
            True if position or rotation goals exist.
        """
        return bool(self.position_goals) or bool(self.rotation_goals)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Main controllers
    "FullBodyIKController",
    "AnimationGraphController",
    # Target specifications
    "LookAtTarget",
    "FootPlacementController",
    "IKTargetComponent",
]
