"""IK Layer for Animation System Integration.

This module provides the IKLayer class that integrates IK solving into
the animation layer system. It allows IK to be applied as a post-process
after animation graph evaluation, with support for various blend modes.

Example usage:

    from engine.animation.ik import FullBodyIK, IKLayer, IKBlendMode, IKGoalContext
    from engine.core.math import Vec3

    # Create IK layer with a solver
    ik_layer = IKLayer(
        name="foot_ik",
        solver=foot_placement_solver,
        blend_mode=IKBlendMode.BLEND,
        weight=1.0
    )

    # Update goals each frame
    context = IKGoalContext()
    context.position_goals["LeftFoot"] = ground_position
    context.weights["LeftFoot"] = 1.0
    ik_layer.update_goals(context)

    # Apply to animation pose
    result_transforms = ik_layer.apply(input_transforms, dt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Union

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform

if TYPE_CHECKING:
    from engine.animation.ik.fullbody import FullBodyIK, FullBodyIKGoal, FullBodyIKResult
    from engine.animation.ik.foot_placement import FootPlacement, FootPlacementResult
    from engine.animation.ik.two_bone import TwoBoneIK, TwoBoneIKResult
    from engine.animation.ik.fabrik import FABRIKChain, FABRIKResult


# =============================================================================
# IK BLEND MODE
# =============================================================================


class IKBlendMode(Enum):
    """How IK results blend with the input animation pose.

    Attributes:
        OVERRIDE: Replace animation pose with IK result entirely.
        ADDITIVE: Add IK correction delta to the animation pose.
        BLEND: Lerp between animation pose and IK result using weight.
    """

    OVERRIDE = auto()   # Replace animation pose
    ADDITIVE = auto()   # Add IK correction to animation
    BLEND = auto()      # Blend between animation and IK result


# =============================================================================
# IK GOAL CONTEXT
# =============================================================================


@dataclass
class IKGoalContext:
    """Context for updating IK goals each frame.

    Provides a frame-by-frame interface for setting IK targets without
    needing to reconstruct goal objects. Goals are keyed by bone name
    for easy lookup and modification.

    Attributes:
        position_goals: Target positions keyed by bone name.
        rotation_goals: Target rotations keyed by bone name.
        weights: Per-goal weights keyed by bone name (0-1).
        pole_vectors: Optional pole vector positions for limb IK.
        chain_assignments: Maps bone names to chain types (e.g., "left_arm").
    """

    position_goals: Dict[str, Vec3] = field(default_factory=dict)
    rotation_goals: Dict[str, Quat] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    pole_vectors: Dict[str, Vec3] = field(default_factory=dict)
    chain_assignments: Dict[str, str] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all goals."""
        self.position_goals.clear()
        self.rotation_goals.clear()
        self.weights.clear()
        self.pole_vectors.clear()
        self.chain_assignments.clear()

    def set_position_goal(
        self,
        bone_name: str,
        position: Vec3,
        weight: float = 1.0,
        chain_type: Optional[str] = None
    ) -> None:
        """Set a position goal for a bone.

        Args:
            bone_name: Name of the target bone.
            position: World space target position.
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
            bone_name: Name of the target bone.
            rotation: Target rotation quaternion.
            weight: Goal weight (0-1).
        """
        self.rotation_goals[bone_name] = rotation
        self.weights[bone_name] = max(0.0, min(1.0, weight))

    def set_pole_vector(self, bone_name: str, pole_position: Vec3) -> None:
        """Set a pole vector for a limb IK chain.

        Args:
            bone_name: Name of the effector bone.
            pole_position: World space pole vector position.
        """
        self.pole_vectors[bone_name] = pole_position

    def get_weight(self, bone_name: str, default: float = 1.0) -> float:
        """Get the weight for a bone goal.

        Args:
            bone_name: Name of the bone.
            default: Default weight if not specified.

        Returns:
            Goal weight (0-1).
        """
        return self.weights.get(bone_name, default)

    def has_goal(self, bone_name: str) -> bool:
        """Check if a goal exists for a bone.

        Args:
            bone_name: Name of the bone.

        Returns:
            True if position or rotation goal exists.
        """
        return bone_name in self.position_goals or bone_name in self.rotation_goals


# =============================================================================
# IK LAYER RESULT
# =============================================================================


@dataclass
class IKLayerResult:
    """Result from IK layer application.

    Attributes:
        transforms: Modified transforms after IK solve.
        success: Whether IK solve was successful.
        errors: Per-bone position errors (if available).
        blend_weight: Actual blend weight used.
    """

    transforms: List[Transform] = field(default_factory=list)
    success: bool = True
    errors: Dict[str, float] = field(default_factory=dict)
    blend_weight: float = 1.0


# =============================================================================
# IK SOLVER PROTOCOL
# =============================================================================


class IKSolverProtocol(Protocol):
    """Protocol defining the interface for IK solvers usable with IKLayer.

    Any solver implementing this protocol can be used with IKLayer.
    """

    def solve(
        self,
        transforms: List[Transform],
        *args,
        **kwargs
    ) -> object:
        """Solve IK and return result with transforms."""
        ...


# =============================================================================
# IK LAYER
# =============================================================================


class IKLayer:
    """Animation layer that applies IK after graph evaluation.

    IKLayer integrates IK solving into the animation layer system,
    allowing IK corrections to be applied as a post-process after
    animation graph evaluation. Supports multiple blend modes for
    combining IK results with animation.

    The layer can work with various IK solvers (FullBodyIK, FootPlacement,
    TwoBoneIK, etc.) through a common interface.

    Attributes:
        name: Layer identifier.
        solver: IK solver instance (FullBodyIK, FootPlacement, etc.).
        blend_mode: How IK results blend with animation.
        weight: Overall layer weight (0-1).
        enabled: Whether the layer is active.

    Example:
        # Create foot placement layer
        foot_layer = IKLayer(
            name="foot_ik",
            solver=foot_placement,
            blend_mode=IKBlendMode.BLEND,
            weight=1.0
        )

        # Update goals from gameplay
        context = IKGoalContext()
        context.set_position_goal("LeftFoot", ground_pos, weight=1.0)
        foot_layer.update_goals(context)

        # Apply to animation
        result = foot_layer.apply(animation_transforms, dt=0.016)
    """

    def __init__(
        self,
        name: str,
        solver: Optional[Union['FullBodyIK', 'FootPlacement', 'TwoBoneIK', 'FABRIKChain']] = None,
        blend_mode: IKBlendMode = IKBlendMode.BLEND,
        weight: float = 1.0,
        enabled: bool = True
    ) -> None:
        """Initialize IK layer.

        Args:
            name: Layer identifier.
            solver: IK solver instance.
            blend_mode: How IK results blend with animation.
            weight: Overall layer weight (0-1).
            enabled: Whether the layer is active.
        """
        self.name = name
        self.solver = solver
        self.blend_mode = blend_mode
        self.weight = max(0.0, min(1.0, weight))
        self.enabled = enabled

        self._goal_context = IKGoalContext()
        self._bone_name_to_index: Dict[str, int] = {}
        self._cached_goals: List['FullBodyIKGoal'] = []
        self._last_result: Optional[IKLayerResult] = None

        # Smoothing for gradual weight transitions
        self._target_weight = weight
        self._weight_blend_speed = 5.0  # Weight blend speed per second

    def set_solver(self, solver: Union['FullBodyIK', 'FootPlacement', 'TwoBoneIK', 'FABRIKChain']) -> None:
        """Set or replace the IK solver.

        Args:
            solver: New IK solver instance.
        """
        self.solver = solver
        self._cached_goals.clear()

    def get_solver(self) -> Optional[Union['FullBodyIK', 'FootPlacement', 'TwoBoneIK', 'FABRIKChain']]:
        """Get the current IK solver.

        Returns:
            Current solver or None if not set.
        """
        return self.solver

    def update_goals(self, context: IKGoalContext) -> None:
        """Update IK goals from context.

        Copies goals from the provided context to the layer's internal
        goal state. Call this each frame with updated target positions.

        Args:
            context: Goal context with updated targets.
        """
        self._goal_context.position_goals = dict(context.position_goals)
        self._goal_context.rotation_goals = dict(context.rotation_goals)
        self._goal_context.weights = dict(context.weights)
        self._goal_context.pole_vectors = dict(context.pole_vectors)
        self._goal_context.chain_assignments = dict(context.chain_assignments)

    def clear_goals(self) -> None:
        """Clear all IK goals."""
        self._goal_context.clear()

    def set_position_goal(
        self,
        bone_name: str,
        position: Vec3,
        weight: float = 1.0,
        chain_type: Optional[str] = None
    ) -> None:
        """Set a position goal directly.

        Convenience method to set a single position goal without
        constructing a full IKGoalContext.

        Args:
            bone_name: Target bone name.
            position: World space target position.
            weight: Goal weight (0-1).
            chain_type: Optional chain assignment.
        """
        self._goal_context.set_position_goal(bone_name, position, weight, chain_type)

    def set_rotation_goal(
        self,
        bone_name: str,
        rotation: Quat,
        weight: float = 1.0
    ) -> None:
        """Set a rotation goal directly.

        Args:
            bone_name: Target bone name.
            rotation: Target rotation.
            weight: Goal weight (0-1).
        """
        self._goal_context.set_rotation_goal(bone_name, rotation, weight)

    def set_bone_mapping(self, bone_name_to_index: Dict[str, int]) -> None:
        """Set bone name to index mapping.

        Required for converting named goals to indexed goals
        when using FullBodyIK.

        Args:
            bone_name_to_index: Mapping from bone names to indices.
        """
        self._bone_name_to_index = dict(bone_name_to_index)

    def apply(
        self,
        transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply IK to pose transforms.

        Main entry point for applying IK. This method:
        1. Updates goals from context
        2. Calls the solver's solve method
        3. Blends result based on blend mode

        Args:
            transforms: Input bone transforms (world space).
            dt: Delta time for smoothing.

        Returns:
            Modified transforms with IK applied.
        """
        if not self.enabled or not self.solver:
            return transforms

        # Smooth weight transitions
        self._update_weight(dt)

        effective_weight = self.weight
        if effective_weight <= 0.0:
            return transforms

        # Solve IK
        ik_result = self._solve_ik(transforms, dt)

        if ik_result is None or not ik_result.transforms:
            return transforms

        # Apply blend mode
        result = self._blend_transforms(
            transforms,
            ik_result.transforms,
            effective_weight
        )

        # Cache result for debugging
        self._last_result = IKLayerResult(
            transforms=result,
            success=ik_result.success if hasattr(ik_result, 'success') else True,
            blend_weight=effective_weight
        )

        return result

    def _update_weight(self, dt: float) -> None:
        """Smoothly update weight towards target.

        Args:
            dt: Delta time.
        """
        if abs(self.weight - self._target_weight) > 0.001:
            blend = min(1.0, self._weight_blend_speed * dt)
            self.weight = self.weight + (self._target_weight - self.weight) * blend

    def _solve_ik(
        self,
        transforms: List[Transform],
        dt: float
    ) -> Optional[object]:
        """Solve IK using the configured solver.

        Args:
            transforms: Input transforms.
            dt: Delta time.

        Returns:
            Solver result or None if solve failed.
        """
        if self.solver is None:
            return None

        # Handle different solver types
        solver_type = type(self.solver).__name__

        if solver_type == 'FullBodyIK':
            return self._solve_fullbody(transforms)
        elif solver_type == 'FootPlacement':
            return self._solve_foot_placement(transforms, dt)
        elif solver_type == 'TwoBoneIK':
            return self._solve_two_bone(transforms)
        elif solver_type == 'FABRIKChain':
            return self._solve_fabrik(transforms)
        else:
            # Try generic solve interface
            if hasattr(self.solver, 'solve'):
                try:
                    return self.solver.solve(transforms)
                except TypeError:
                    # Solver may need different arguments
                    return None
            return None

    def _solve_fullbody(self, transforms: List[Transform]) -> Optional['FullBodyIKResult']:
        """Solve using FullBodyIK.

        Args:
            transforms: Input transforms.

        Returns:
            FullBodyIKResult or None.
        """
        from engine.animation.ik.fullbody import FullBodyIKGoal

        # Convert goal context to FullBodyIKGoals
        goals = []

        for bone_name, position in self._goal_context.position_goals.items():
            bone_idx = self._bone_name_to_index.get(bone_name, -1)
            if bone_idx < 0:
                continue

            weight = self._goal_context.get_weight(bone_name)
            chain_type = self._goal_context.chain_assignments.get(bone_name)
            rotation = self._goal_context.rotation_goals.get(bone_name)

            goal = FullBodyIKGoal(
                bone_index=bone_idx,
                target_position=position,
                target_rotation=rotation,
                position_weight=weight,
                rotation_weight=weight if rotation else 0.0,
                chain_type=chain_type,
                enabled=True
            )
            goals.append(goal)

        if not goals:
            return None

        return self.solver.solve(transforms, goals)

    def _solve_foot_placement(
        self,
        transforms: List[Transform],
        dt: float
    ) -> Optional['FootPlacementResult']:
        """Solve using FootPlacement.

        Args:
            transforms: Input transforms.
            dt: Delta time for smoothing.

        Returns:
            FootPlacementResult or None.
        """
        # FootPlacement has its own goal management
        # Just call solve with the transforms and dt
        if hasattr(self.solver, 'solve'):
            return self.solver.solve(transforms, dt)
        return None

    def _solve_two_bone(self, transforms: List[Transform]) -> Optional['TwoBoneIKResult']:
        """Solve using TwoBoneIK.

        Args:
            transforms: Input transforms.

        Returns:
            TwoBoneIKResult or None.
        """
        # Get first position goal as target
        if not self._goal_context.position_goals:
            return None

        bone_name = next(iter(self._goal_context.position_goals))
        target = self._goal_context.position_goals[bone_name]
        pole = self._goal_context.pole_vectors.get(bone_name)

        # TwoBoneIK expects specific bone transforms
        # This is a simplified implementation
        if hasattr(self.solver, 'solve'):
            if len(transforms) >= 3:
                return self.solver.solve(
                    transforms[0],
                    transforms[1],
                    transforms[2],
                    target,
                    pole
                )
        return None

    def _solve_fabrik(self, transforms: List[Transform]) -> Optional['FABRIKResult']:
        """Solve using FABRIK.

        Args:
            transforms: Input transforms.

        Returns:
            FABRIKResult or None.
        """
        # Get first position goal as target
        if not self._goal_context.position_goals:
            return None

        bone_name = next(iter(self._goal_context.position_goals))
        target = self._goal_context.position_goals[bone_name]

        # Extract positions from transforms
        positions = [t.translation for t in transforms]

        if hasattr(self.solver, 'solve'):
            return self.solver.solve(positions, target)
        return None

    def _blend_transforms(
        self,
        input_transforms: List[Transform],
        ik_transforms: List[Transform],
        weight: float
    ) -> List[Transform]:
        """Blend input transforms with IK result based on blend mode.

        Args:
            input_transforms: Original animation transforms.
            ik_transforms: IK solver output transforms.
            weight: Blend weight (0-1).

        Returns:
            Blended transforms.
        """
        if weight >= 1.0 and self.blend_mode == IKBlendMode.OVERRIDE:
            return ik_transforms

        result = []
        count = min(len(input_transforms), len(ik_transforms))

        for i in range(count):
            input_tf = input_transforms[i]
            ik_tf = ik_transforms[i]

            if self.blend_mode == IKBlendMode.OVERRIDE:
                # Lerp from input to IK result
                blended = input_tf.lerp(ik_tf, weight)

            elif self.blend_mode == IKBlendMode.ADDITIVE:
                # Compute delta and add to input
                delta = self._compute_additive_delta(input_tf, ik_tf)
                blended = self._apply_additive_delta(input_tf, delta, weight)

            else:  # BLEND
                # Standard lerp blend
                blended = input_tf.lerp(ik_tf, weight)

            result.append(blended)

        # Append remaining input transforms if IK result is shorter
        if len(input_transforms) > count:
            result.extend(input_transforms[count:])

        return result

    def _compute_additive_delta(
        self,
        input_tf: Transform,
        ik_tf: Transform
    ) -> Transform:
        """Compute additive delta between input and IK transforms.

        Args:
            input_tf: Original transform.
            ik_tf: IK result transform.

        Returns:
            Delta transform representing the IK correction.
        """
        # Position delta
        pos_delta = ik_tf.translation - input_tf.translation

        # Rotation delta: ik_rotation = input_rotation * delta
        # delta = input_rotation^-1 * ik_rotation
        inv_input_rot = input_tf.rotation.conjugate()
        rot_delta = inv_input_rot * ik_tf.rotation

        # Scale ratio (multiplicative)
        scale_delta = Vec3(
            ik_tf.scale.x / input_tf.scale.x if input_tf.scale.x != 0 else 1.0,
            ik_tf.scale.y / input_tf.scale.y if input_tf.scale.y != 0 else 1.0,
            ik_tf.scale.z / input_tf.scale.z if input_tf.scale.z != 0 else 1.0
        )

        return Transform(pos_delta, rot_delta, scale_delta)

    def _apply_additive_delta(
        self,
        input_tf: Transform,
        delta: Transform,
        weight: float
    ) -> Transform:
        """Apply weighted additive delta to input transform.

        Args:
            input_tf: Original transform.
            delta: Additive delta from IK.
            weight: Blend weight (0-1).

        Returns:
            Transform with additive correction applied.
        """
        # Scale delta by weight
        weighted_pos = delta.translation * weight

        # Slerp rotation delta towards identity
        identity_rot = Quat.identity()
        weighted_rot = identity_rot.slerp(delta.rotation, weight)

        # Scale delta interpolation
        weighted_scale = Vec3(
            1.0 + (delta.scale.x - 1.0) * weight,
            1.0 + (delta.scale.y - 1.0) * weight,
            1.0 + (delta.scale.z - 1.0) * weight
        )

        # Apply to input
        new_pos = input_tf.translation + weighted_pos
        new_rot = input_tf.rotation * weighted_rot
        new_scale = Vec3(
            input_tf.scale.x * weighted_scale.x,
            input_tf.scale.y * weighted_scale.y,
            input_tf.scale.z * weighted_scale.z
        )

        return Transform(new_pos, new_rot.normalized(), new_scale)

    def set_weight(self, weight: float, immediate: bool = False) -> None:
        """Set the layer weight.

        Args:
            weight: New weight value (0-1).
            immediate: If True, set weight immediately without smoothing.
        """
        clamped_weight = max(0.0, min(1.0, weight))
        self._target_weight = clamped_weight
        if immediate:
            self.weight = clamped_weight

    def get_weight(self) -> float:
        """Get the current layer weight.

        Returns:
            Current weight (0-1).
        """
        return self.weight

    def set_blend_mode(self, mode: IKBlendMode) -> None:
        """Set the blend mode.

        Args:
            mode: New blend mode.
        """
        self.blend_mode = mode

    def get_blend_mode(self) -> IKBlendMode:
        """Get the current blend mode.

        Returns:
            Current blend mode.
        """
        return self.blend_mode

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the layer.

        Args:
            enabled: Whether the layer should be active.
        """
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if the layer is enabled.

        Returns:
            True if layer is enabled.
        """
        return self.enabled

    def set_weight_blend_speed(self, speed: float) -> None:
        """Set the weight blend speed.

        Args:
            speed: Blend speed in units per second.
        """
        self._weight_blend_speed = max(0.1, speed)

    def get_last_result(self) -> Optional[IKLayerResult]:
        """Get the last IK layer result.

        Useful for debugging and performance monitoring.

        Returns:
            Last result or None if layer hasn't been applied.
        """
        return self._last_result

    def get_goal_context(self) -> IKGoalContext:
        """Get the current goal context.

        Returns:
            Current goal context (read-only reference).
        """
        return self._goal_context


# =============================================================================
# IK LAYER STACK
# =============================================================================


class IKLayerStack:
    """Ordered collection of IK layers.

    Allows multiple IK layers to be applied in sequence, with each
    layer processing the output of the previous layer.

    Example:
        stack = IKLayerStack()
        stack.add_layer(foot_ik_layer)
        stack.add_layer(hand_ik_layer)

        # Apply all layers
        result = stack.apply(transforms, dt)
    """

    def __init__(self) -> None:
        """Initialize empty layer stack."""
        self._layers: List[IKLayer] = []
        self._layer_by_name: Dict[str, IKLayer] = {}

    def add_layer(self, layer: IKLayer, index: Optional[int] = None) -> int:
        """Add a layer to the stack.

        Args:
            layer: IK layer to add.
            index: Optional position (appends if None).

        Returns:
            Index where layer was added.

        Raises:
            ValueError: If layer with same name exists.
        """
        if layer.name in self._layer_by_name:
            raise ValueError(f"Layer '{layer.name}' already exists in stack")

        if index is None:
            self._layers.append(layer)
            actual_index = len(self._layers) - 1
        else:
            index = max(0, min(len(self._layers), index))
            self._layers.insert(index, layer)
            actual_index = index

        self._layer_by_name[layer.name] = layer
        return actual_index

    def remove_layer(self, name: str) -> bool:
        """Remove a layer by name.

        Args:
            name: Layer name.

        Returns:
            True if layer was removed.
        """
        layer = self._layer_by_name.get(name)
        if layer:
            self._layers.remove(layer)
            del self._layer_by_name[name]
            return True
        return False

    def get_layer(self, name: str) -> Optional[IKLayer]:
        """Get a layer by name.

        Args:
            name: Layer name.

        Returns:
            Layer or None if not found.
        """
        return self._layer_by_name.get(name)

    def get_layer_by_index(self, index: int) -> Optional[IKLayer]:
        """Get a layer by index.

        Args:
            index: Layer index.

        Returns:
            Layer or None if index out of range.
        """
        if 0 <= index < len(self._layers):
            return self._layers[index]
        return None

    def layer_count(self) -> int:
        """Get the number of layers.

        Returns:
            Number of layers in stack.
        """
        return len(self._layers)

    def apply(self, transforms: List[Transform], dt: float) -> List[Transform]:
        """Apply all layers in order.

        Args:
            transforms: Input bone transforms.
            dt: Delta time.

        Returns:
            Transforms with all IK layers applied.
        """
        result = transforms
        for layer in self._layers:
            if layer.enabled:
                result = layer.apply(result, dt)
        return result

    def set_all_weights(self, weight: float, immediate: bool = False) -> None:
        """Set weight for all layers.

        Args:
            weight: Weight value (0-1).
            immediate: If True, set immediately without smoothing.
        """
        for layer in self._layers:
            layer.set_weight(weight, immediate)

    def disable_all(self) -> None:
        """Disable all layers."""
        for layer in self._layers:
            layer.set_enabled(False)

    def enable_all(self) -> None:
        """Enable all layers."""
        for layer in self._layers:
            layer.set_enabled(True)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "IKBlendMode",
    # Data classes
    "IKGoalContext",
    "IKLayerResult",
    # Layer
    "IKLayer",
    "IKLayerStack",
]
