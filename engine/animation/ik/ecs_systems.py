"""ECS Systems for Animation IK Processing.

This module provides ECS systems for integrating animation graph evaluation
with IK solving in the Trinity entity-component-system architecture.

Systems are executed in the following order:
    1. AnimationGraphSystem (phase: animation) - Evaluates animation graphs
    2. FootPlacementSystem (phase: animation_late) - Terrain-adaptive foot IK
    3. FullBodyIKSystem (phase: animation_late, after FootPlacement) - Full body IK
    4. LookAtSystem (phase: animation_late, after FullBodyIK) - Look-at constraints

Pipeline:
    Animation Graph -> Foot Placement -> Full Body IK -> Look-At -> Final Pose

Example usage:

    from engine.animation.ik.ecs_systems import (
        AnimationGraphIKSystem,
        FootPlacementSystem,
        FullBodyIKSystem,
        LookAtSystem,
        register_animation_ik_systems,
    )

    # Register all systems
    register_animation_ik_systems(world)

    # Or register individually with custom order
    world.add_system(AnimationGraphIKSystem())
    world.add_system(FootPlacementSystem())
    world.add_system(FullBodyIKSystem())
    world.add_system(LookAtSystem())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)

from trinity.decorators import system

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World
    from engine.animation.ik.ecs_components import (
        FullBodyIKController,
        AnimationGraphController,
        FootPlacementController,
        LookAtTarget,
    )
    from engine.animation.ik.fullbody import FullBodyIK, LookAtSolver
    from engine.animation.ik.foot_placement import FootPlacement, RaycastHit


# =============================================================================
# QUERY TYPE (Protocol for type hints)
# =============================================================================


class Query(Protocol):
    """Protocol for ECS query interface."""

    def __iter__(self):
        """Iterate over (entity, *components) tuples."""
        ...


# =============================================================================
# SYSTEM STATISTICS
# =============================================================================


@dataclass
class AnimationIKSystemStats:
    """Performance statistics for animation IK systems.

    Attributes:
        entities_processed: Number of entities processed this frame.
        graph_evaluations: Number of animation graphs evaluated.
        foot_placement_solves: Number of foot placement IK solves.
        fullbody_solves: Number of full body IK solves.
        lookat_solves: Number of look-at solves.
        total_time_ms: Total processing time in milliseconds.
        average_error: Average IK position error.
    """

    entities_processed: int = 0
    graph_evaluations: int = 0
    foot_placement_solves: int = 0
    fullbody_solves: int = 0
    lookat_solves: int = 0
    total_time_ms: float = 0.0
    average_error: float = 0.0

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.entities_processed = 0
        self.graph_evaluations = 0
        self.foot_placement_solves = 0
        self.fullbody_solves = 0
        self.lookat_solves = 0
        self.total_time_ms = 0.0
        self.average_error = 0.0


# =============================================================================
# ANIMATION GRAPH IK SYSTEM
# =============================================================================


@system(phase="animation", priority=90)
class AnimationGraphIKSystem:
    """Evaluates animation graphs and produces base poses for IK processing.

    This system runs in the 'animation' phase before IK systems. It:
    1. Updates animation graph state from controllers
    2. Samples current pose from active animations
    3. Stores output transforms for downstream IK systems

    The output_transforms field on AnimationGraphController contains the
    base pose that IK systems will modify.

    Attributes:
        enabled: Whether the system is active.
        parallel_threshold: Minimum entities for parallel processing.
    """

    def __init__(self) -> None:
        """Initialize the animation graph IK system."""
        self._stats = AnimationIKSystemStats()
        self.enabled: bool = True
        self.parallel_threshold: int = 8
        self._frame_count: int = 0

    def get_stats(self) -> AnimationIKSystemStats:
        """Get statistics from last update.

        Returns:
            Copy of current statistics.
        """
        return AnimationIKSystemStats(
            entities_processed=self._stats.entities_processed,
            graph_evaluations=self._stats.graph_evaluations,
            total_time_ms=self._stats.total_time_ms,
        )

    def update(
        self,
        dt: float,
        entities: List[Tuple[Any, 'AnimationGraphController']],
    ) -> None:
        """Update all animation graph controllers.

        For each entity with AnimationGraphController:
        1. Check if controller is enabled
        2. Update animation graph state
        3. Sample current pose from graph
        4. Store output_transforms for IK systems

        Args:
            dt: Delta time in seconds.
            entities: List of (entity, controller) tuples from query.
        """
        import time
        start_time = time.perf_counter()

        self._stats.reset()
        self._frame_count += 1

        if not self.enabled:
            return

        for entity, controller in entities:
            self._stats.entities_processed += 1

            if not controller.enabled:
                continue

            self._update_controller(controller, dt)
            self._stats.graph_evaluations += 1

        elapsed = time.perf_counter() - start_time
        self._stats.total_time_ms = elapsed * 1000.0

    def _update_controller(
        self,
        controller: 'AnimationGraphController',
        dt: float,
    ) -> None:
        """Update a single animation graph controller.

        Args:
            controller: The AnimationGraphController to update.
            dt: Delta time in seconds.
        """
        # Update frame tracking
        controller._frame_count += 1
        controller._last_update_time = dt

        # Get the AnimationIKController if available
        ik_controller = controller.controller
        if ik_controller is None:
            # No IK controller, just update time-based state
            return

        # If graph is set, evaluate it
        graph = controller.graph
        if graph is not None:
            # Evaluate the animation graph to get base pose
            base_transforms = self._evaluate_graph(graph, controller, dt)

            if base_transforms:
                # Apply IK layers through the controller
                result_transforms = ik_controller.update(base_transforms, dt)
                controller.output_transforms = result_transforms
            else:
                # No transforms from graph, clear output
                controller.output_transforms = []
        else:
            # No graph, check if we have existing transforms to process
            if controller.output_transforms:
                # Apply IK layers to existing transforms
                result_transforms = ik_controller.update(
                    controller.output_transforms, dt
                )
                controller.output_transforms = result_transforms

    def _evaluate_graph(
        self,
        graph: Any,
        controller: 'AnimationGraphController',
        dt: float,
    ) -> List[Transform]:
        """Evaluate animation graph to produce base pose.

        Args:
            graph: The animation graph to evaluate.
            controller: The controller containing graph state.
            dt: Delta time in seconds.

        Returns:
            List of bone transforms from graph evaluation.
        """
        # Update blend time
        if controller.blend_time > 0:
            controller.blend_time = max(0.0, controller.blend_time - dt)

        # Check if graph has an evaluate method
        if hasattr(graph, 'evaluate'):
            # Graph provides pose evaluation
            pose = graph.evaluate(dt * controller.time_scale)
            if hasattr(pose, 'transforms'):
                return list(pose.transforms)
            return []

        # Check if graph has a sample method
        if hasattr(graph, 'sample'):
            pose = graph.sample(dt * controller.time_scale)
            if hasattr(pose, 'transforms'):
                return list(pose.transforms)
            return []

        # Check if graph has current_pose
        if hasattr(graph, 'current_pose'):
            pose = graph.current_pose
            if hasattr(pose, 'transforms'):
                return list(pose.transforms)
            if isinstance(pose, list):
                return pose
            return []

        return []


# =============================================================================
# FOOT PLACEMENT SYSTEM
# =============================================================================


@system(phase="animation_late", priority=10)
class FootPlacementSystem:
    """Processes foot placement IK after animation graph evaluation.

    This system runs before FullBodyIKSystem to ensure pelvis adjustments
    are applied before full body IK solving. It:
    1. Gets animation output transforms from graph system
    2. Raycasts for terrain contact points
    3. Adjusts foot positions to match terrain
    4. Adjusts pelvis height for reachability

    Raycast Interface:
        The system uses a raycast callback provided by the physics system.
        If no callback is set, foot placement is skipped.

    Attributes:
        enabled: Whether the system is active.
        default_raycast_offset: Default vertical offset for raycasts.
        default_raycast_length: Default raycast length.
    """

    def __init__(self) -> None:
        """Initialize the foot placement system."""
        self._stats = AnimationIKSystemStats()
        self.enabled: bool = True
        self.default_raycast_offset: float = 1.0
        self.default_raycast_length: float = 2.0
        self._raycast_callback: Optional[Callable[[Vec3, Vec3], Any]] = None

    def set_raycast_callback(
        self,
        callback: Callable[[Vec3, Vec3], Any],
    ) -> None:
        """Set the raycast callback for terrain detection.

        Args:
            callback: Function(origin, direction) -> RaycastHit
        """
        self._raycast_callback = callback

    def get_stats(self) -> AnimationIKSystemStats:
        """Get statistics from last update.

        Returns:
            Copy of current statistics.
        """
        return AnimationIKSystemStats(
            entities_processed=self._stats.entities_processed,
            foot_placement_solves=self._stats.foot_placement_solves,
            total_time_ms=self._stats.total_time_ms,
            average_error=self._stats.average_error,
        )

    def update(
        self,
        dt: float,
        entities: List[Tuple[Any, 'FootPlacementController', 'AnimationGraphController']],
    ) -> None:
        """Update all foot placement controllers.

        For each entity with FootPlacementController and AnimationGraphController:
        1. Get animation output transforms
        2. Raycast for terrain contact under each foot
        3. Adjust foot positions to match terrain
        4. Adjust pelvis height so feet can reach targets

        Args:
            dt: Delta time in seconds.
            entities: List of (entity, foot_controller, graph_controller) tuples.
        """
        import time
        start_time = time.perf_counter()

        self._stats.reset()

        if not self.enabled:
            return

        total_error = 0.0
        error_count = 0

        for entity, foot_controller, graph_controller in entities:
            self._stats.entities_processed += 1

            if not foot_controller.enabled:
                continue

            result = self._update_foot_placement(
                foot_controller, graph_controller, dt
            )

            if result is not None:
                self._stats.foot_placement_solves += 1
                if result > 0:
                    total_error += result
                    error_count += 1

        if error_count > 0:
            self._stats.average_error = total_error / error_count

        elapsed = time.perf_counter() - start_time
        self._stats.total_time_ms = elapsed * 1000.0

    def _update_foot_placement(
        self,
        foot_controller: 'FootPlacementController',
        graph_controller: 'AnimationGraphController',
        dt: float,
    ) -> Optional[float]:
        """Update foot placement for a single entity.

        Args:
            foot_controller: The FootPlacementController component.
            graph_controller: The AnimationGraphController with base pose.
            dt: Delta time in seconds.

        Returns:
            Position error if solved, None if skipped.
        """
        # Get the foot placement solver
        placement = foot_controller.placement
        if placement is None:
            return None

        # Get current transforms from graph controller
        transforms = graph_controller.output_transforms
        if not transforms:
            return None

        # Set raycast callback if available
        if self._raycast_callback is not None:
            if hasattr(placement, 'set_raycast_callback'):
                placement.set_raycast_callback(self._raycast_callback)

        # Use controller's raycast callback if set
        if foot_controller._raycast_callback is not None:
            if hasattr(placement, 'set_raycast_callback'):
                placement.set_raycast_callback(foot_controller._raycast_callback)

        # Solve foot placement
        if hasattr(placement, 'solve'):
            result = placement.solve(transforms, dt)

            # Apply result to transforms
            if result is not None and hasattr(result, 'transforms'):
                result_transforms = result.transforms
                for i, transform in enumerate(result_transforms):
                    if i < len(graph_controller.output_transforms):
                        graph_controller.output_transforms[i] = transform

                # Update foot placement state
                if hasattr(result, 'left_planted'):
                    foot_controller._left_planted = result.left_planted
                if hasattr(result, 'right_planted'):
                    foot_controller._right_planted = result.right_planted
                if hasattr(result, 'pelvis_offset'):
                    foot_controller._current_pelvis_offset = result.pelvis_offset
                if hasattr(result, 'terrain_slope'):
                    foot_controller._terrain_slope = result.terrain_slope

                # Return error if available
                if hasattr(result, 'error'):
                    return result.error

            return 0.0

        return None


# =============================================================================
# FULL BODY IK SYSTEM
# =============================================================================


@system(phase="animation_late", priority=20)
class FullBodyIKSystem:
    """Processes full body IK after foot placement.

    This system runs after FootPlacementSystem to combine all IK corrections.
    It handles multi-effector IK with balance maintenance:
    1. Gets transforms with foot placement adjustments
    2. Processes IK goals from component
    3. Solves full body IK chains
    4. Applies balance corrections

    The system respects solve order configured in the controller:
    - legs -> spine -> arms -> look_at

    Attributes:
        enabled: Whether the system is active.
        max_iterations: Maximum IK solver iterations.
        position_tolerance: Position error tolerance for convergence.
    """

    def __init__(self) -> None:
        """Initialize the full body IK system."""
        self._stats = AnimationIKSystemStats()
        self.enabled: bool = True
        self.max_iterations: int = 10
        self.position_tolerance: float = 0.001

    def get_stats(self) -> AnimationIKSystemStats:
        """Get statistics from last update.

        Returns:
            Copy of current statistics.
        """
        return AnimationIKSystemStats(
            entities_processed=self._stats.entities_processed,
            fullbody_solves=self._stats.fullbody_solves,
            total_time_ms=self._stats.total_time_ms,
            average_error=self._stats.average_error,
        )

    def update(
        self,
        dt: float,
        entities: List[Tuple[Any, 'FullBodyIKController', 'AnimationGraphController']],
    ) -> None:
        """Update all full body IK controllers.

        For each entity with FullBodyIKController and AnimationGraphController:
        1. Get transforms (with foot placement adjustments)
        2. Gather IK goals from controller
        3. Solve full body IK chains
        4. Apply balance corrections if enabled

        Args:
            dt: Delta time in seconds.
            entities: List of (entity, ik_controller, graph_controller) tuples.
        """
        import time
        start_time = time.perf_counter()

        self._stats.reset()

        if not self.enabled:
            return

        total_error = 0.0
        error_count = 0

        for entity, ik_controller, graph_controller in entities:
            self._stats.entities_processed += 1

            if not ik_controller.enabled:
                continue

            if ik_controller.weight <= 0.0:
                continue

            result = self._update_fullbody_ik(
                ik_controller, graph_controller, dt
            )

            if result is not None:
                self._stats.fullbody_solves += 1
                if result > 0:
                    total_error += result
                    error_count += 1

        if error_count > 0:
            self._stats.average_error = total_error / error_count

        elapsed = time.perf_counter() - start_time
        self._stats.total_time_ms = elapsed * 1000.0

    def _update_fullbody_ik(
        self,
        ik_controller: 'FullBodyIKController',
        graph_controller: 'AnimationGraphController',
        dt: float,
    ) -> Optional[float]:
        """Update full body IK for a single entity.

        Args:
            ik_controller: The FullBodyIKController component.
            graph_controller: The AnimationGraphController with current pose.
            dt: Delta time in seconds.

        Returns:
            Position error if solved, None if skipped.
        """
        # Get the full body IK solver
        solver = ik_controller.solver
        if solver is None:
            return None

        # Get current transforms
        transforms = graph_controller.output_transforms
        if not transforms:
            return None

        # Build goals from controller configuration
        goals = self._build_goals(ik_controller)

        # Solve full body IK
        if hasattr(solver, 'solve'):
            result = solver.solve(goals)

            if result is not None:
                # Apply result transforms
                if hasattr(result, 'transforms'):
                    result_transforms = result.transforms
                    weight = ik_controller.weight

                    for i, ik_transform in enumerate(result_transforms):
                        if i < len(transforms):
                            if weight >= 1.0:
                                transforms[i] = ik_transform
                            else:
                                # Blend between original and IK result
                                transforms[i] = self._blend_transform(
                                    transforms[i], ik_transform, weight
                                )

                    graph_controller.output_transforms = transforms

                # Update pelvis offset
                if hasattr(result, 'pelvis_offset'):
                    ik_controller._pelvis_offset = result.pelvis_offset

                # Return error if available
                if hasattr(result, 'error'):
                    return result.error
                if hasattr(result, 'final_error'):
                    return result.final_error

            return 0.0

        return None

    def _build_goals(
        self,
        ik_controller: 'FullBodyIKController',
    ) -> List[Any]:
        """Build IK goals from controller configuration.

        Args:
            ik_controller: The controller with goal configuration.

        Returns:
            List of IK goals for the solver.
        """
        goals = []

        # Get goals from IK goal context
        if ik_controller.ik_goals is not None:
            context = ik_controller.ik_goals

            # Add position goals
            for bone_name, position in context.position_goals.items():
                weight = context.weights.get(bone_name, 1.0)
                goals.append({
                    'bone_name': bone_name,
                    'position': position,
                    'weight': weight,
                })

            # Add rotation goals
            for bone_name, rotation in context.rotation_goals.items():
                weight = context.weights.get(bone_name, 1.0)
                goals.append({
                    'bone_name': bone_name,
                    'rotation': rotation,
                    'weight': weight,
                })

        return goals

    def _blend_transform(
        self,
        original: Transform,
        target: Transform,
        weight: float,
    ) -> Transform:
        """Blend between two transforms.

        Args:
            original: Original transform.
            target: Target transform.
            weight: Blend weight (0 = original, 1 = target).

        Returns:
            Blended transform.
        """
        if weight <= 0.0:
            return original
        if weight >= 1.0:
            return target

        # Lerp position
        pos = Vec3(
            original.translation.x + (target.translation.x - original.translation.x) * weight,
            original.translation.y + (target.translation.y - original.translation.y) * weight,
            original.translation.z + (target.translation.z - original.translation.z) * weight,
        )

        # Slerp rotation
        rot = original.rotation.slerp(target.rotation, weight)

        # Lerp scale
        scale = Vec3(
            original.scale.x + (target.scale.x - original.scale.x) * weight,
            original.scale.y + (target.scale.y - original.scale.y) * weight,
            original.scale.z + (target.scale.z - original.scale.z) * weight,
        )

        return Transform(pos, rot, scale)


# =============================================================================
# LOOK-AT SYSTEM
# =============================================================================


@system(phase="animation_late", priority=30)
class LookAtSystem:
    """Processes look-at constraints after full body IK.

    This system applies head/eye tracking by rotating spine and head bones
    toward a target. It can run standalone or integrate with FullBodyIKSystem.

    Look-at processing:
    1. Get target position (from component or entity tracking)
    2. Calculate required rotation to face target
    3. Distribute rotation across spine bones using weights
    4. Apply final head rotation

    Spine distribution ensures natural head movement by involving the
    upper spine in the rotation, not just the head bone.

    Attributes:
        enabled: Whether the system is active.
        default_blend_speed: Default speed for weight transitions.
    """

    def __init__(self) -> None:
        """Initialize the look-at system."""
        self._stats = AnimationIKSystemStats()
        self.enabled: bool = True
        self.default_blend_speed: float = 5.0

    def get_stats(self) -> AnimationIKSystemStats:
        """Get statistics from last update.

        Returns:
            Copy of current statistics.
        """
        return AnimationIKSystemStats(
            entities_processed=self._stats.entities_processed,
            lookat_solves=self._stats.lookat_solves,
            total_time_ms=self._stats.total_time_ms,
        )

    def update(
        self,
        dt: float,
        entities: List[Tuple[Any, 'LookAtTarget', 'FullBodyIKController']],
    ) -> None:
        """Update all look-at targets.

        For each entity with LookAtTarget and FullBodyIKController:
        1. Get target position (from component or entity)
        2. Calculate head/spine rotations
        3. Apply with distribution weights

        Args:
            dt: Delta time in seconds.
            entities: List of (entity, look_at, ik_controller) tuples.
        """
        import time
        start_time = time.perf_counter()

        self._stats.reset()

        if not self.enabled:
            return

        for entity, look_at, ik_controller in entities:
            self._stats.entities_processed += 1

            if not look_at.enabled:
                continue

            if not look_at.has_target():
                # Blend out
                self._blend_out(look_at, dt)
                continue

            self._update_look_at(look_at, ik_controller, dt)
            self._stats.lookat_solves += 1

        elapsed = time.perf_counter() - start_time
        self._stats.total_time_ms = elapsed * 1000.0

    def _update_look_at(
        self,
        look_at: 'LookAtTarget',
        ik_controller: 'FullBodyIKController',
        dt: float,
    ) -> None:
        """Update look-at for a single entity.

        Args:
            look_at: The LookAtTarget component.
            ik_controller: The FullBodyIKController with look-at solver.
            dt: Delta time in seconds.
        """
        # Smooth weight transition
        target_weight = look_at.blend_weight if look_at.enabled else 0.0
        blend_speed = look_at.blend_speed

        weight_diff = target_weight - look_at._current_weight
        if abs(weight_diff) > 0.001:
            max_change = blend_speed * dt
            if abs(weight_diff) <= max_change:
                look_at._current_weight = target_weight
            else:
                look_at._current_weight += max_change if weight_diff > 0 else -max_change

        # Skip if weight is effectively zero
        if look_at._current_weight < 0.001:
            return

        # Get target position
        target_pos = look_at.target_position
        if target_pos is None:
            # Would need to resolve target_entity here
            # For now, skip if no direct position
            return

        # Store last target for smoothing
        look_at._last_target = target_pos

        # Use look-at solver from IK controller if available
        look_at_solver = ik_controller.look_at_solver
        if look_at_solver is not None and hasattr(look_at_solver, 'solve'):
            # Solve using the dedicated solver
            rotations = look_at_solver.solve(
                target_pos,
                look_at._current_weight,
            )

            # Apply rotations to IK controller's look-at target
            if ik_controller.look_at_enabled:
                ik_controller.look_at_target = target_pos
                ik_controller.look_at_weight = look_at._current_weight
        else:
            # Direct application through controller
            if ik_controller.look_at_enabled or look_at.enabled:
                ik_controller.look_at_target = target_pos
                ik_controller.look_at_weight = look_at._current_weight
                ik_controller.look_at_enabled = True

    def _blend_out(
        self,
        look_at: 'LookAtTarget',
        dt: float,
    ) -> None:
        """Blend out look-at weight when no target.

        Args:
            look_at: The LookAtTarget component.
            dt: Delta time in seconds.
        """
        if look_at._current_weight > 0.001:
            blend_speed = look_at.blend_speed
            look_at._current_weight = max(
                0.0,
                look_at._current_weight - blend_speed * dt
            )


# =============================================================================
# COMPOSITE SYSTEM (Alternative single-system approach)
# =============================================================================


@system(phase="animation_late", priority=0)
class AnimationIKCompositeSystem:
    """Composite system that runs all IK processing in one pass.

    This system provides an alternative to running separate systems,
    which can be more efficient for simple use cases. It processes:
    1. Foot placement
    2. Full body IK
    3. Look-at

    All processing happens in a single system update, avoiding the
    overhead of multiple system dispatches.

    Attributes:
        enabled: Whether the system is active.
        foot_placement_enabled: Enable foot placement processing.
        fullbody_ik_enabled: Enable full body IK processing.
        look_at_enabled: Enable look-at processing.
    """

    def __init__(self) -> None:
        """Initialize the composite system."""
        self._stats = AnimationIKSystemStats()
        self.enabled: bool = True
        self.foot_placement_enabled: bool = True
        self.fullbody_ik_enabled: bool = True
        self.look_at_enabled: bool = True
        self._raycast_callback: Optional[Callable[[Vec3, Vec3], Any]] = None

    def set_raycast_callback(
        self,
        callback: Callable[[Vec3, Vec3], Any],
    ) -> None:
        """Set the raycast callback for foot placement.

        Args:
            callback: Function(origin, direction) -> RaycastHit
        """
        self._raycast_callback = callback

    def get_stats(self) -> AnimationIKSystemStats:
        """Get statistics from last update.

        Returns:
            Copy of current statistics.
        """
        return AnimationIKSystemStats(
            entities_processed=self._stats.entities_processed,
            foot_placement_solves=self._stats.foot_placement_solves,
            fullbody_solves=self._stats.fullbody_solves,
            lookat_solves=self._stats.lookat_solves,
            total_time_ms=self._stats.total_time_ms,
            average_error=self._stats.average_error,
        )

    def update(
        self,
        dt: float,
        entities: List[Tuple[
            Any,
            'AnimationGraphController',
            Optional['FootPlacementController'],
            Optional['FullBodyIKController'],
            Optional['LookAtTarget'],
        ]],
    ) -> None:
        """Update all IK processing in one pass.

        Processes entities with AnimationGraphController and optional
        IK components in the correct order.

        Args:
            dt: Delta time in seconds.
            entities: List of (entity, graph, foot, fullbody, lookat) tuples.
        """
        import time
        start_time = time.perf_counter()

        self._stats.reset()

        if not self.enabled:
            return

        for entity, graph_controller, foot_controller, ik_controller, look_at in entities:
            self._stats.entities_processed += 1

            if not graph_controller.enabled:
                continue

            # Get base transforms
            transforms = graph_controller.output_transforms
            if not transforms:
                continue

            # 1. Foot placement
            if (
                self.foot_placement_enabled
                and foot_controller is not None
                and foot_controller.enabled
            ):
                self._process_foot_placement(foot_controller, transforms, dt)
                self._stats.foot_placement_solves += 1

            # 2. Full body IK
            if (
                self.fullbody_ik_enabled
                and ik_controller is not None
                and ik_controller.enabled
                and ik_controller.weight > 0
            ):
                self._process_fullbody_ik(ik_controller, transforms, dt)
                self._stats.fullbody_solves += 1

            # 3. Look-at
            if (
                self.look_at_enabled
                and look_at is not None
                and look_at.enabled
                and look_at.has_target()
                and ik_controller is not None
            ):
                self._process_look_at(look_at, ik_controller, dt)
                self._stats.lookat_solves += 1

            # Store final transforms
            graph_controller.output_transforms = transforms

        elapsed = time.perf_counter() - start_time
        self._stats.total_time_ms = elapsed * 1000.0

    def _process_foot_placement(
        self,
        controller: 'FootPlacementController',
        transforms: List[Transform],
        dt: float,
    ) -> None:
        """Process foot placement IK.

        Args:
            controller: The FootPlacementController component.
            transforms: Current bone transforms to modify.
            dt: Delta time in seconds.
        """
        placement = controller.placement
        if placement is None:
            return

        # Set raycast callback
        callback = controller._raycast_callback or self._raycast_callback
        if callback is not None and hasattr(placement, 'set_raycast_callback'):
            placement.set_raycast_callback(callback)

        # Solve
        if hasattr(placement, 'solve'):
            result = placement.solve(transforms, dt)
            if result is not None and hasattr(result, 'transforms'):
                for i, t in enumerate(result.transforms):
                    if i < len(transforms):
                        transforms[i] = t

    def _process_fullbody_ik(
        self,
        controller: 'FullBodyIKController',
        transforms: List[Transform],
        dt: float,
    ) -> None:
        """Process full body IK.

        Args:
            controller: The FullBodyIKController component.
            transforms: Current bone transforms to modify.
            dt: Delta time in seconds.
        """
        solver = controller.solver
        if solver is None:
            return

        # Build and solve
        goals = []
        if controller.ik_goals is not None:
            for bone_name, position in controller.ik_goals.position_goals.items():
                weight = controller.ik_goals.weights.get(bone_name, 1.0)
                goals.append({
                    'bone_name': bone_name,
                    'position': position,
                    'weight': weight,
                })

        if hasattr(solver, 'solve'):
            result = solver.solve(goals)
            if result is not None and hasattr(result, 'transforms'):
                weight = controller.weight
                for i, t in enumerate(result.transforms):
                    if i < len(transforms):
                        if weight >= 1.0:
                            transforms[i] = t
                        else:
                            # Simple lerp for position
                            orig = transforms[i]
                            transforms[i] = Transform(
                                Vec3(
                                    orig.translation.x + (t.translation.x - orig.translation.x) * weight,
                                    orig.translation.y + (t.translation.y - orig.translation.y) * weight,
                                    orig.translation.z + (t.translation.z - orig.translation.z) * weight,
                                ),
                                orig.rotation.slerp(t.rotation, weight),
                                orig.scale,
                            )

    def _process_look_at(
        self,
        look_at: 'LookAtTarget',
        ik_controller: 'FullBodyIKController',
        dt: float,
    ) -> None:
        """Process look-at constraint.

        Args:
            look_at: The LookAtTarget component.
            ik_controller: The FullBodyIKController to update.
            dt: Delta time in seconds.
        """
        # Update weight
        target_weight = look_at.blend_weight
        weight_diff = target_weight - look_at._current_weight
        if abs(weight_diff) > 0.001:
            max_change = look_at.blend_speed * dt
            if abs(weight_diff) <= max_change:
                look_at._current_weight = target_weight
            else:
                look_at._current_weight += max_change if weight_diff > 0 else -max_change

        # Apply to controller
        if look_at.target_position is not None:
            ik_controller.look_at_target = look_at.target_position
            ik_controller.look_at_weight = look_at._current_weight
            ik_controller.look_at_enabled = True


# =============================================================================
# REGISTRATION HELPER
# =============================================================================


def register_animation_ik_systems(world: Any) -> None:
    """Register all animation and IK systems with proper ordering.

    This function registers the individual systems (not the composite)
    in the correct execution order:
    1. AnimationGraphIKSystem (phase: animation)
    2. FootPlacementSystem (phase: animation_late)
    3. FullBodyIKSystem (phase: animation_late, after foot)
    4. LookAtSystem (phase: animation_late, after fullbody)

    Args:
        world: The ECS World to register systems with.
    """
    # Create system instances
    graph_system = AnimationGraphIKSystem()
    foot_system = FootPlacementSystem()
    fullbody_system = FullBodyIKSystem()
    lookat_system = LookAtSystem()

    # Register in order
    if hasattr(world, 'add_system'):
        world.add_system(graph_system)
        world.add_system(foot_system)
        world.add_system(fullbody_system)
        world.add_system(lookat_system)
    elif hasattr(world, 'register_system'):
        world.register_system(graph_system)
        world.register_system(foot_system)
        world.register_system(fullbody_system)
        world.register_system(lookat_system)


def register_composite_system(world: Any) -> AnimationIKCompositeSystem:
    """Register the composite IK system.

    Use this for simpler setups where all IK processing can happen
    in a single system update.

    Args:
        world: The ECS World to register with.

    Returns:
        The registered composite system instance.
    """
    composite = AnimationIKCompositeSystem()

    if hasattr(world, 'add_system'):
        world.add_system(composite)
    elif hasattr(world, 'register_system'):
        world.register_system(composite)

    return composite


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Statistics
    "AnimationIKSystemStats",
    # Individual systems
    "AnimationGraphIKSystem",
    "FootPlacementSystem",
    "FullBodyIKSystem",
    "LookAtSystem",
    # Composite system
    "AnimationIKCompositeSystem",
    # Registration helpers
    "register_animation_ik_systems",
    "register_composite_system",
]
