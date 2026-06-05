"""Graph + IK Integration Layer.

This module provides the integration layer between AnimationGraph output
and IK solving. It manages goal sources, solve order, and result combination.

Pipeline:
    1. AnimationGraph produces base pose (external)
    2. Goal sources provide IK targets (IKGoalSource)
    3. IK layers solve in order (IKSolveOrder)
    4. Results are combined into final pose (AnimationIKController)

Example usage:

    from engine.animation.ik.graph_integration import (
        AnimationIKController,
        IKGoalSource,
        IKSolveOrder,
    )
    from engine.animation.ik import IKLayer, IKGoalContext

    # Create controller
    controller = AnimationIKController()

    # Add IK layers
    controller.add_ik_layer(foot_ik_layer)
    controller.add_ik_layer(hand_ik_layer)

    # Add goal sources
    gameplay_goals = IKGoalSource(
        name="gameplay",
        priority=10,
        get_goals=lambda: get_current_gameplay_goals()
    )
    controller.add_goal_source(gameplay_goals)

    # Set solve order
    controller.set_solve_order(IKSolveOrder.FOOT_FIRST)

    # Update each frame
    final_transforms = controller.update(animation_transforms, dt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from engine.core.math.transform import Transform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat

from engine.animation.ik.ik_layer import (
    IKBlendMode,
    IKGoalContext,
    IKLayer,
    IKLayerResult,
    IKLayerStack,
)


# =============================================================================
# IK SOLVE ORDER
# =============================================================================


class IKSolveOrder(Enum):
    """Order in which IK systems are solved.

    The solve order affects how IK corrections accumulate:

    - FOOT_FIRST: Foot placement adjusts pelvis, then full body IK works
      from the adjusted pose. Good for grounded characters.

    - FULLBODY_FIRST: Full body IK solves first to achieve goals,
      foot placement corrects feet after. Good for reaching tasks.

    - PARALLEL: Solve independently and blend results. Each solver
      works from the original animation pose.

    - CUSTOM: User-defined order via layer names list.

    Attributes:
        FOOT_FIRST: Solve foot placement before other IK.
        FULLBODY_FIRST: Solve full body IK before foot placement.
        PARALLEL: Solve all IK independently, blend results.
        CUSTOM: User-defined order via custom_order list.
    """

    FOOT_FIRST = auto()
    FULLBODY_FIRST = auto()
    PARALLEL = auto()
    CUSTOM = auto()


# =============================================================================
# IK GOAL SOURCE
# =============================================================================


@dataclass
class IKGoalSource:
    """Source for IK goal data.

    A goal source provides IK targets for the controller. Multiple sources
    can be active simultaneously, with priority determining which source
    wins when goals conflict.

    Higher priority sources override lower priority sources when they
    specify goals for the same bones.

    Attributes:
        name: Unique identifier for this source.
        priority: Priority level (higher = more important, wins conflicts).
        enabled: Whether this source is currently active.
        get_goals: Callable that returns the current IKGoalContext.

    Example:
        # Create a goal source from gameplay
        def get_gameplay_goals() -> IKGoalContext:
            ctx = IKGoalContext()
            ctx.set_position_goal("LeftHand", weapon_grip_pos, 1.0)
            return ctx

        source = IKGoalSource(
            name="weapon_grip",
            priority=100,
            get_goals=get_gameplay_goals
        )
    """

    name: str
    priority: int = 0
    enabled: bool = True
    get_goals: Optional[Callable[[], IKGoalContext]] = None

    def __post_init__(self) -> None:
        """Validate source configuration."""
        if not self.name:
            raise ValueError("IKGoalSource requires a non-empty name")

    def fetch_goals(self) -> Optional[IKGoalContext]:
        """Fetch goals from this source.

        Returns:
            IKGoalContext with current goals, or None if disabled or
            no get_goals callable is set.
        """
        if not self.enabled or self.get_goals is None:
            return None

        try:
            return self.get_goals()
        except Exception:
            # Silently fail if goal getter raises
            # Production code should log this
            return None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this goal source.

        Args:
            enabled: Whether the source should be active.
        """
        self.enabled = enabled

    def set_priority(self, priority: int) -> None:
        """Set the priority level.

        Args:
            priority: New priority value.
        """
        self.priority = priority


# =============================================================================
# ANIMATION IK RESULT
# =============================================================================


@dataclass
class AnimationIKResult:
    """Result of animation + IK processing.

    Contains the final transforms after all IK processing, along with
    metadata about which layers and goals contributed to the result.

    Attributes:
        transforms: Final bone transforms after IK application.
        animation_weight: Weight of the original animation in result.
        ik_weight: Overall IK weight applied.
        layers_applied: Names of IK layers that contributed.
        goals_used: Number of goals that were processed.
        errors: Per-bone position errors (bone_name -> error distance).
        success: Whether all IK solves converged successfully.
    """

    transforms: List[Transform] = field(default_factory=list)
    animation_weight: float = 1.0
    ik_weight: float = 1.0
    layers_applied: List[str] = field(default_factory=list)
    goals_used: int = 0
    errors: Dict[str, float] = field(default_factory=dict)
    success: bool = True

    def add_layer_applied(self, layer_name: str) -> None:
        """Record that a layer was applied.

        Args:
            layer_name: Name of the applied layer.
        """
        if layer_name not in self.layers_applied:
            self.layers_applied.append(layer_name)

    def set_error(self, bone_name: str, error: float) -> None:
        """Record an error for a bone.

        Args:
            bone_name: Name of the bone.
            error: Position error distance.
        """
        self.errors[bone_name] = error

    def total_error(self) -> float:
        """Calculate total error across all bones.

        Returns:
            Sum of all bone errors.
        """
        return sum(self.errors.values())

    def average_error(self) -> float:
        """Calculate average error across all bones.

        Returns:
            Average error, or 0 if no errors recorded.
        """
        if not self.errors:
            return 0.0
        return sum(self.errors.values()) / len(self.errors)


# =============================================================================
# ANIMATION IK CONTROLLER
# =============================================================================


class AnimationIKController:
    """Integrates AnimationGraph output with IK solving.

    The controller manages the pipeline from animation output to final pose:

    Pipeline:
        1. AnimationGraph produces base pose (passed to update())
        2. Goal sources provide IK targets (collected from all sources)
        3. IK layers solve in order (determined by solve_order)
        4. Results are combined into final pose (returned from update())

    Goal Conflict Resolution:
        When multiple goal sources specify targets for the same bone,
        the source with higher priority wins. Goals are merged by iterating
        sources from lowest to highest priority, with later sources
        overwriting earlier ones.

    Solve Order:
        - FOOT_FIRST: Foot placement IK runs first, adjusting pelvis.
          Full body IK then solves from the adjusted pose.
        - FULLBODY_FIRST: Full body IK solves first to reach targets.
          Foot placement then corrects feet to terrain.
        - PARALLEL: All IK runs independently from original pose,
          results are blended together.
        - CUSTOM: User specifies exact layer order via set_solve_order().

    Attributes:
        solve_order: Current solve order mode.
        ik_weight: Global IK weight multiplier (0-1).
        enabled: Whether IK processing is active.

    Example:
        controller = AnimationIKController()

        # Add layers
        controller.add_ik_layer(foot_layer)
        controller.add_ik_layer(fullbody_layer)

        # Add goal sources
        controller.add_goal_source(IKGoalSource(
            name="combat",
            priority=100,
            get_goals=get_combat_goals
        ))

        # Configure
        controller.set_solve_order(IKSolveOrder.FOOT_FIRST)

        # Update each frame
        final_pose = controller.update(animation_transforms, dt)
    """

    # Layer names for built-in solve orders
    FOOT_LAYER_NAMES = {"foot_ik", "foot_placement", "feet", "legs"}
    FULLBODY_LAYER_NAMES = {"fullbody_ik", "fullbody", "body", "torso"}

    def __init__(self) -> None:
        """Initialize the Animation IK Controller."""
        self._ik_stack: IKLayerStack = IKLayerStack()
        self._goal_sources: List[IKGoalSource] = []
        self._sources_by_name: Dict[str, IKGoalSource] = {}
        self._solve_order: IKSolveOrder = IKSolveOrder.FOOT_FIRST
        self._custom_order: List[str] = []

        # Global settings
        self.ik_weight: float = 1.0
        self.enabled: bool = True

        # Cached state
        self._merged_context: IKGoalContext = IKGoalContext()
        self._last_result: Optional[AnimationIKResult] = None

        # Layer categorization cache
        self._layer_categories: Dict[str, str] = {}  # layer_name -> "foot"/"fullbody"/"other"

    # -------------------------------------------------------------------------
    # Goal Source Management
    # -------------------------------------------------------------------------

    def add_goal_source(self, source: IKGoalSource) -> None:
        """Add a goal source to the controller.

        Args:
            source: Goal source to add.

        Raises:
            ValueError: If source with same name already exists.
        """
        if source.name in self._sources_by_name:
            raise ValueError(f"Goal source '{source.name}' already exists")

        self._goal_sources.append(source)
        self._sources_by_name[source.name] = source

        # Keep sorted by priority (ascending - we iterate low to high)
        self._goal_sources.sort(key=lambda s: s.priority)

    def remove_goal_source(self, name: str) -> bool:
        """Remove a goal source by name.

        Args:
            name: Name of the source to remove.

        Returns:
            True if source was removed, False if not found.
        """
        source = self._sources_by_name.get(name)
        if source:
            self._goal_sources.remove(source)
            del self._sources_by_name[name]
            return True
        return False

    def get_goal_source(self, name: str) -> Optional[IKGoalSource]:
        """Get a goal source by name.

        Args:
            name: Name of the source.

        Returns:
            Goal source or None if not found.
        """
        return self._sources_by_name.get(name)

    def set_goal_source_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a goal source.

        Args:
            name: Name of the source.
            enabled: Whether the source should be active.

        Returns:
            True if source was found and updated.
        """
        source = self._sources_by_name.get(name)
        if source:
            source.set_enabled(enabled)
            return True
        return False

    def set_goal_source_priority(self, name: str, priority: int) -> bool:
        """Set the priority of a goal source.

        Args:
            name: Name of the source.
            priority: New priority value.

        Returns:
            True if source was found and updated.
        """
        source = self._sources_by_name.get(name)
        if source:
            source.set_priority(priority)
            # Re-sort by priority
            self._goal_sources.sort(key=lambda s: s.priority)
            return True
        return False

    def goal_source_count(self) -> int:
        """Get the number of goal sources.

        Returns:
            Number of registered goal sources.
        """
        return len(self._goal_sources)

    def get_goal_source_names(self) -> List[str]:
        """Get names of all goal sources.

        Returns:
            List of source names in priority order.
        """
        return [s.name for s in self._goal_sources]

    # -------------------------------------------------------------------------
    # IK Layer Management
    # -------------------------------------------------------------------------

    def add_ik_layer(self, layer: IKLayer, category: Optional[str] = None) -> int:
        """Add an IK layer to the stack.

        Args:
            layer: IK layer to add.
            category: Optional category hint ("foot", "fullbody", "other").
                     If not provided, category is inferred from layer name.

        Returns:
            Index where layer was added.
        """
        index = self._ik_stack.add_layer(layer)

        # Categorize layer for solve order
        if category:
            self._layer_categories[layer.name] = category
        else:
            self._layer_categories[layer.name] = self._infer_layer_category(layer.name)

        return index

    def remove_ik_layer(self, name: str) -> bool:
        """Remove an IK layer by name.

        Args:
            name: Name of the layer to remove.

        Returns:
            True if layer was removed.
        """
        if self._ik_stack.remove_layer(name):
            self._layer_categories.pop(name, None)
            return True
        return False

    def get_ik_layer(self, name: str) -> Optional[IKLayer]:
        """Get an IK layer by name.

        Args:
            name: Name of the layer.

        Returns:
            IK layer or None if not found.
        """
        return self._ik_stack.get_layer(name)

    def ik_layer_count(self) -> int:
        """Get the number of IK layers.

        Returns:
            Number of layers in the stack.
        """
        return self._ik_stack.layer_count()

    def get_ik_layer_names(self) -> List[str]:
        """Get names of all IK layers.

        Returns:
            List of layer names in stack order.
        """
        return [
            self._ik_stack.get_layer_by_index(i).name
            for i in range(self._ik_stack.layer_count())
            if self._ik_stack.get_layer_by_index(i) is not None
        ]

    def set_ik_layer_weight(self, name: str, weight: float, immediate: bool = False) -> bool:
        """Set the weight of an IK layer.

        Args:
            name: Name of the layer.
            weight: New weight value (0-1).
            immediate: If True, set immediately without smoothing.

        Returns:
            True if layer was found and updated.
        """
        layer = self._ik_stack.get_layer(name)
        if layer:
            layer.set_weight(weight, immediate)
            return True
        return False

    def set_ik_layer_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable an IK layer.

        Args:
            name: Name of the layer.
            enabled: Whether the layer should be active.

        Returns:
            True if layer was found and updated.
        """
        layer = self._ik_stack.get_layer(name)
        if layer:
            layer.set_enabled(enabled)
            return True
        return False

    def set_layer_category(self, name: str, category: str) -> bool:
        """Set the category for an IK layer.

        Args:
            name: Name of the layer.
            category: Category ("foot", "fullbody", "other").

        Returns:
            True if layer exists.
        """
        if self._ik_stack.get_layer(name):
            self._layer_categories[name] = category
            return True
        return False

    def _infer_layer_category(self, layer_name: str) -> str:
        """Infer layer category from name.

        Args:
            layer_name: Name of the layer.

        Returns:
            Category string ("foot", "fullbody", or "other").
        """
        name_lower = layer_name.lower()

        for foot_name in self.FOOT_LAYER_NAMES:
            if foot_name in name_lower:
                return "foot"

        for body_name in self.FULLBODY_LAYER_NAMES:
            if body_name in name_lower:
                return "fullbody"

        return "other"

    # -------------------------------------------------------------------------
    # Solve Order Configuration
    # -------------------------------------------------------------------------

    def set_solve_order(
        self,
        order: IKSolveOrder,
        custom_order: Optional[List[str]] = None
    ) -> None:
        """Set the IK solve order.

        Args:
            order: Solve order mode.
            custom_order: For CUSTOM mode, list of layer names in solve order.

        Raises:
            ValueError: If CUSTOM order specified without custom_order list.
        """
        self._solve_order = order

        if order == IKSolveOrder.CUSTOM:
            if custom_order is None:
                raise ValueError("CUSTOM solve order requires custom_order list")
            self._custom_order = list(custom_order)
        else:
            self._custom_order = []

    def get_solve_order(self) -> IKSolveOrder:
        """Get the current solve order.

        Returns:
            Current solve order mode.
        """
        return self._solve_order

    def get_custom_order(self) -> List[str]:
        """Get the custom layer order.

        Returns:
            List of layer names for CUSTOM mode, empty otherwise.
        """
        return list(self._custom_order)

    # -------------------------------------------------------------------------
    # Main Update Method
    # -------------------------------------------------------------------------

    def update(
        self,
        base_transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply IK to animation output.

        Main entry point for each frame. This method:
        1. Collects goals from all sources (sorted by priority)
        2. Merges goals into combined context (higher priority wins)
        3. Applies IK layers in solve order
        4. Returns final transforms

        Args:
            base_transforms: Transforms from animation graph (world space).
            dt: Delta time for smoothing and IK solves.

        Returns:
            Final transforms with IK applied.
        """
        if not self.enabled or self.ik_weight <= 0.0:
            return base_transforms

        # Collect and merge goals from all sources
        merged_context = self._collect_goals()

        # Update all layers with merged goals
        self._distribute_goals(merged_context)

        # Apply IK in solve order
        result_transforms = self._apply_solve_order(base_transforms, merged_context, dt)

        # Cache result for debugging
        self._cache_result(result_transforms, merged_context)

        return result_transforms

    def _collect_goals(self) -> IKGoalContext:
        """Gather goals from all enabled sources, higher priority wins.

        Iterates through sources in priority order (low to high).
        Later sources overwrite earlier ones for conflicting goals.

        Returns:
            Merged goal context with all active goals.
        """
        merged = IKGoalContext()

        # Sources are sorted by priority ascending
        # Iterate in order so higher priority overwrites lower
        for source in self._goal_sources:
            if not source.enabled:
                continue

            source_context = source.fetch_goals()
            if source_context is None:
                continue

            # Merge position goals
            for bone_name, position in source_context.position_goals.items():
                merged.position_goals[bone_name] = position
                # Also copy weight if present
                if bone_name in source_context.weights:
                    merged.weights[bone_name] = source_context.weights[bone_name]

            # Merge rotation goals
            for bone_name, rotation in source_context.rotation_goals.items():
                merged.rotation_goals[bone_name] = rotation
                if bone_name in source_context.weights:
                    merged.weights[bone_name] = source_context.weights[bone_name]

            # Merge pole vectors
            for bone_name, pole in source_context.pole_vectors.items():
                merged.pole_vectors[bone_name] = pole

            # Merge chain assignments
            for bone_name, chain in source_context.chain_assignments.items():
                merged.chain_assignments[bone_name] = chain

        self._merged_context = merged
        return merged

    def _distribute_goals(self, context: IKGoalContext) -> None:
        """Distribute merged goals to IK layers.

        Args:
            context: Merged goal context.
        """
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer:
                layer.update_goals(context)

    def _apply_solve_order(
        self,
        transforms: List[Transform],
        context: IKGoalContext,
        dt: float
    ) -> List[Transform]:
        """Apply IK layers according to solve order.

        Args:
            transforms: Input transforms from animation.
            context: Merged goal context.
            dt: Delta time.

        Returns:
            Transforms with IK applied.
        """
        if self._solve_order == IKSolveOrder.PARALLEL:
            return self._apply_parallel(transforms, dt)
        elif self._solve_order == IKSolveOrder.CUSTOM:
            return self._apply_custom_order(transforms, dt)
        elif self._solve_order == IKSolveOrder.FOOT_FIRST:
            return self._apply_foot_first(transforms, dt)
        else:  # FULLBODY_FIRST
            return self._apply_fullbody_first(transforms, dt)

    def _apply_foot_first(
        self,
        transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply foot IK first, then other IK.

        Args:
            transforms: Input transforms.
            dt: Delta time.

        Returns:
            Transforms with IK applied.
        """
        result = transforms

        # First pass: foot layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "foot":
                    result = layer.apply(result, dt)

        # Second pass: fullbody layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "fullbody":
                    result = layer.apply(result, dt)

        # Third pass: other layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "other":
                    result = layer.apply(result, dt)

        return result

    def _apply_fullbody_first(
        self,
        transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply fullbody IK first, then foot IK.

        Args:
            transforms: Input transforms.
            dt: Delta time.

        Returns:
            Transforms with IK applied.
        """
        result = transforms

        # First pass: fullbody layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "fullbody":
                    result = layer.apply(result, dt)

        # Second pass: foot layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "foot":
                    result = layer.apply(result, dt)

        # Third pass: other layers
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                category = self._layer_categories.get(layer.name, "other")
                if category == "other":
                    result = layer.apply(result, dt)

        return result

    def _apply_parallel(
        self,
        transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply all IK independently and blend results.

        Each layer receives the original animation pose.
        Results are blended together using layer weights.

        Args:
            transforms: Input transforms.
            dt: Delta time.

        Returns:
            Blended result transforms.
        """
        if self._ik_stack.layer_count() == 0:
            return transforms

        # Collect results from all enabled layers
        layer_results: List[Tuple[IKLayer, List[Transform]]] = []
        total_weight = 0.0

        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled and layer.weight > 0:
                # Each layer gets original transforms
                layer_output = layer.apply(list(transforms), dt)
                layer_results.append((layer, layer_output))
                total_weight += layer.weight

        if not layer_results or total_weight <= 0:
            return transforms

        # Blend all results together
        result = self._blend_parallel_results(transforms, layer_results, total_weight)

        return result

    def _blend_parallel_results(
        self,
        base_transforms: List[Transform],
        layer_results: List[Tuple[IKLayer, List[Transform]]],
        total_weight: float
    ) -> List[Transform]:
        """Blend parallel IK results.

        Args:
            base_transforms: Original animation transforms.
            layer_results: List of (layer, transforms) tuples.
            total_weight: Sum of all layer weights.

        Returns:
            Blended transforms.
        """
        if len(base_transforms) == 0:
            return []

        # Start with base as fallback
        result = list(base_transforms)

        # Calculate normalized weights
        weights = [(layer.weight / total_weight) for layer, _ in layer_results]

        # Blend each bone
        for bone_idx in range(len(base_transforms)):
            # Collect all transforms for this bone with their weights
            weighted_transforms: List[Tuple[Transform, float]] = []

            for (layer, layer_tfs), weight in zip(layer_results, weights):
                if bone_idx < len(layer_tfs):
                    weighted_transforms.append((layer_tfs[bone_idx], weight))

            if weighted_transforms:
                result[bone_idx] = self._weighted_blend_transform(weighted_transforms)

        return result

    def _weighted_blend_transform(
        self,
        weighted_transforms: List[Tuple[Transform, float]]
    ) -> Transform:
        """Blend transforms with weights.

        Args:
            weighted_transforms: List of (transform, weight) tuples.

        Returns:
            Blended transform.
        """
        if not weighted_transforms:
            return Transform.identity()

        if len(weighted_transforms) == 1:
            return weighted_transforms[0][0]

        # Blend position (weighted average)
        pos_x = sum(tf.translation.x * w for tf, w in weighted_transforms)
        pos_y = sum(tf.translation.y * w for tf, w in weighted_transforms)
        pos_z = sum(tf.translation.z * w for tf, w in weighted_transforms)
        blended_pos = Vec3(pos_x, pos_y, pos_z)

        # Blend scale (weighted average)
        scale_x = sum(tf.scale.x * w for tf, w in weighted_transforms)
        scale_y = sum(tf.scale.y * w for tf, w in weighted_transforms)
        scale_z = sum(tf.scale.z * w for tf, w in weighted_transforms)
        blended_scale = Vec3(scale_x, scale_y, scale_z)

        # Blend rotation (iterative slerp)
        blended_rot = weighted_transforms[0][0].rotation
        accumulated_weight = weighted_transforms[0][1]

        for tf, weight in weighted_transforms[1:]:
            # Calculate relative blend factor
            blend = weight / (accumulated_weight + weight)
            blended_rot = blended_rot.slerp(tf.rotation, blend)
            accumulated_weight += weight

        return Transform(blended_pos, blended_rot.normalized(), blended_scale)

    def _apply_custom_order(
        self,
        transforms: List[Transform],
        dt: float
    ) -> List[Transform]:
        """Apply layers in custom order.

        Args:
            transforms: Input transforms.
            dt: Delta time.

        Returns:
            Transforms with IK applied.
        """
        result = transforms

        # Apply layers in specified order
        for layer_name in self._custom_order:
            layer = self._ik_stack.get_layer(layer_name)
            if layer and layer.enabled:
                result = layer.apply(result, dt)

        # Apply any layers not in custom order (in stack order)
        custom_set = set(self._custom_order)
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled and layer.name not in custom_set:
                result = layer.apply(result, dt)

        return result

    def _cache_result(
        self,
        transforms: List[Transform],
        context: IKGoalContext
    ) -> None:
        """Cache the result for debugging.

        Args:
            transforms: Result transforms.
            context: Merged goal context.
        """
        result = AnimationIKResult(
            transforms=transforms,
            animation_weight=1.0 - self.ik_weight,
            ik_weight=self.ik_weight,
            goals_used=len(context.position_goals) + len(context.rotation_goals),
            success=True
        )

        # Record which layers were applied
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer and layer.enabled:
                result.add_layer_applied(layer.name)

        self._last_result = result

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_last_result(self) -> Optional[AnimationIKResult]:
        """Get the last animation IK result.

        Returns:
            Last result or None if update hasn't been called.
        """
        return self._last_result

    def get_merged_context(self) -> IKGoalContext:
        """Get the last merged goal context.

        Returns:
            Last merged context (may be empty if update not called).
        """
        return self._merged_context

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the entire controller.

        Args:
            enabled: Whether IK processing should be active.
        """
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if the controller is enabled.

        Returns:
            True if controller is enabled.
        """
        return self.enabled

    def set_ik_weight(self, weight: float) -> None:
        """Set the global IK weight.

        Args:
            weight: Global IK weight (0-1).
        """
        self.ik_weight = max(0.0, min(1.0, weight))

    def get_ik_weight(self) -> float:
        """Get the global IK weight.

        Returns:
            Global IK weight (0-1).
        """
        return self.ik_weight

    def clear_all_goals(self) -> None:
        """Clear goals from all layers."""
        for i in range(self._ik_stack.layer_count()):
            layer = self._ik_stack.get_layer_by_index(i)
            if layer:
                layer.clear_goals()

    def disable_all_layers(self) -> None:
        """Disable all IK layers."""
        self._ik_stack.disable_all()

    def enable_all_layers(self) -> None:
        """Enable all IK layers."""
        self._ik_stack.enable_all()

    def reset(self) -> None:
        """Reset the controller to initial state."""
        self._goal_sources.clear()
        self._sources_by_name.clear()
        self._layer_categories.clear()
        self._custom_order.clear()
        self._merged_context = IKGoalContext()
        self._last_result = None
        self._solve_order = IKSolveOrder.FOOT_FIRST
        self.ik_weight = 1.0
        self.enabled = True

        # Clear layer stack
        while self._ik_stack.layer_count() > 0:
            layer = self._ik_stack.get_layer_by_index(0)
            if layer:
                self._ik_stack.remove_layer(layer.name)


# =============================================================================
# GOAL SOURCE BUILDERS
# =============================================================================


class ComponentGoalSource(IKGoalSource):
    """Goal source that reads goals from an ECS component.

    Provides a convenient way to create goal sources that read from
    game objects or entities with IK target components.

    Example:
        source = ComponentGoalSource(
            name="hand_targets",
            priority=50,
            component_getter=lambda: entity.get_component(IKTargetComponent)
        )
    """

    def __init__(
        self,
        name: str,
        priority: int = 0,
        component_getter: Optional[Callable[[], object]] = None
    ) -> None:
        """Initialize component goal source.

        Args:
            name: Source name.
            priority: Priority level.
            component_getter: Callable that returns an IK target component.
        """
        super().__init__(name=name, priority=priority, enabled=True)
        self._component_getter = component_getter

    def set_component_getter(self, getter: Callable[[], object]) -> None:
        """Set the component getter.

        Args:
            getter: Callable returning an IK target component.
        """
        self._component_getter = getter

    def fetch_goals(self) -> Optional[IKGoalContext]:
        """Fetch goals from the component.

        Returns:
            Goal context or None.
        """
        if not self.enabled or self._component_getter is None:
            return None

        try:
            component = self._component_getter()
            if component is None:
                return None

            context = IKGoalContext()

            # Try to read standard attributes from component
            if hasattr(component, 'position_goals'):
                for bone_name, pos in component.position_goals.items():
                    context.position_goals[bone_name] = pos

            if hasattr(component, 'rotation_goals'):
                for bone_name, rot in component.rotation_goals.items():
                    context.rotation_goals[bone_name] = rot

            if hasattr(component, 'weights'):
                for bone_name, weight in component.weights.items():
                    context.weights[bone_name] = weight

            if hasattr(component, 'pole_vectors'):
                for bone_name, pole in component.pole_vectors.items():
                    context.pole_vectors[bone_name] = pole

            return context

        except Exception:
            return None


class CallbackGoalSource(IKGoalSource):
    """Goal source that calls a function to get goals.

    Simplest way to create a dynamic goal source.

    Example:
        def get_look_at_goals():
            ctx = IKGoalContext()
            ctx.set_position_goal("Head", target_pos, 1.0)
            return ctx

        source = CallbackGoalSource("look_at", 100, get_look_at_goals)
    """

    def __init__(
        self,
        name: str,
        priority: int = 0,
        callback: Optional[Callable[[], IKGoalContext]] = None
    ) -> None:
        """Initialize callback goal source.

        Args:
            name: Source name.
            priority: Priority level.
            callback: Callable returning an IKGoalContext.
        """
        super().__init__(name=name, priority=priority, enabled=True, get_goals=callback)

    def set_callback(self, callback: Callable[[], IKGoalContext]) -> None:
        """Set the callback function.

        Args:
            callback: Callable returning an IKGoalContext.
        """
        self.get_goals = callback


class StaticGoalSource(IKGoalSource):
    """Goal source with manually set static goals.

    Useful for debugging or simple cases where goals don't change.

    Example:
        source = StaticGoalSource("debug_goals", priority=1000)
        source.set_position_goal("LeftHand", Vec3(0, 1, 0))
        source.set_position_goal("RightHand", Vec3(0, 1, 1))
    """

    def __init__(self, name: str, priority: int = 0) -> None:
        """Initialize static goal source.

        Args:
            name: Source name.
            priority: Priority level.
        """
        self._static_context = IKGoalContext()
        super().__init__(
            name=name,
            priority=priority,
            enabled=True,
            get_goals=lambda: self._static_context
        )

    def set_position_goal(
        self,
        bone_name: str,
        position: Vec3,
        weight: float = 1.0,
        chain_type: Optional[str] = None
    ) -> None:
        """Set a static position goal.

        Args:
            bone_name: Target bone name.
            position: Target position.
            weight: Goal weight.
            chain_type: Optional chain type.
        """
        self._static_context.set_position_goal(bone_name, position, weight, chain_type)

    def set_rotation_goal(
        self,
        bone_name: str,
        rotation: Quat,
        weight: float = 1.0
    ) -> None:
        """Set a static rotation goal.

        Args:
            bone_name: Target bone name.
            rotation: Target rotation.
            weight: Goal weight.
        """
        self._static_context.set_rotation_goal(bone_name, rotation, weight)

    def set_pole_vector(self, bone_name: str, pole_position: Vec3) -> None:
        """Set a static pole vector.

        Args:
            bone_name: Effector bone name.
            pole_position: Pole position.
        """
        self._static_context.set_pole_vector(bone_name, pole_position)

    def clear(self) -> None:
        """Clear all static goals."""
        self._static_context.clear()

    def get_context(self) -> IKGoalContext:
        """Get the static context (for modification).

        Returns:
            The static goal context.
        """
        return self._static_context


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "IKSolveOrder",
    # Data classes
    "IKGoalSource",
    "AnimationIKResult",
    # Controller
    "AnimationIKController",
    # Goal source builders
    "ComponentGoalSource",
    "CallbackGoalSource",
    "StaticGoalSource",
]
