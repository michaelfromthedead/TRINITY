"""Animation Graph System — Presentation-phase animation evaluation (T-AN-9.3).

This system implements the presentation-phase animation graph evaluation pipeline.
It reads deterministic state machine outputs from simulation (Phase 5) and performs
actual playback: sampling clips, blending, and evaluating blend trees.

Key Features:
- @system(phase="animation") annotation for ECS scheduling
- Dirty-flag driven selective evaluation
- Task-parallel per-entity evaluation via Foundation TaskScheduler
- SoA output format for efficient skinning pipeline integration
- Full blend tree support (1D, 2D, Direct)
- Animation clip sampling with interpolation
- State machine transition blending

Dependencies:
- engine.animation.graph: AnimationGraph, StateMachine, BlendTree, ClipNode
- engine.core.tasks: TaskScheduler for parallel execution
- foundation.tracker: Dirty flag tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from engine.animation.config import ANIMATION_SYSTEM_CONFIG
from engine.animation.graph import (
    AnimationClip,
    AnimationGraph,
    AnimationNode,
    BlendTree,
    BlendTree1D,
    BlendTree2D,
    BlendTreeDirect,
    ClipNode,
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Skeleton,
    StateMachine,
    Transform,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World
    from engine.core.tasks.scheduler import TaskScheduler, TaskHandle


# =============================================================================
# SYSTEM DECORATOR (phase annotation)
# =============================================================================


def system(
    phase: str = "update",
    priority: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        priority: Execution priority within phase (lower = earlier)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_priority = priority
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# SoA BONE TRANSFORM OUTPUT
# =============================================================================


@dataclass
class BoneTransformSoA:
    """Structure-of-Arrays layout for bone transforms.

    Optimized for SIMD processing and GPU upload. Each array holds data
    for all bones of an entity, laid out contiguously.

    Attributes:
        positions_x: X coordinates of all bone positions
        positions_y: Y coordinates of all bone positions
        positions_z: Z coordinates of all bone positions
        rotations_x: X component of quaternion rotations
        rotations_y: Y component of quaternion rotations
        rotations_z: Z component of quaternion rotations
        rotations_w: W component of quaternion rotations
        scales_x: X scale factors
        scales_y: Y scale factors
        scales_z: Z scale factors
        bone_count: Number of bones in this transform set
    """
    positions_x: List[float] = field(default_factory=list)
    positions_y: List[float] = field(default_factory=list)
    positions_z: List[float] = field(default_factory=list)
    rotations_x: List[float] = field(default_factory=list)
    rotations_y: List[float] = field(default_factory=list)
    rotations_z: List[float] = field(default_factory=list)
    rotations_w: List[float] = field(default_factory=list)
    scales_x: List[float] = field(default_factory=list)
    scales_y: List[float] = field(default_factory=list)
    scales_z: List[float] = field(default_factory=list)
    bone_count: int = 0

    @classmethod
    def from_pose(cls, pose: Pose) -> "BoneTransformSoA":
        """Convert AoS Pose to SoA format."""
        soa = cls(bone_count=pose.bone_count())

        for transform in pose.transforms:
            px, py, pz = transform.position
            soa.positions_x.append(px)
            soa.positions_y.append(py)
            soa.positions_z.append(pz)

            rx, ry, rz, rw = transform.rotation
            soa.rotations_x.append(rx)
            soa.rotations_y.append(ry)
            soa.rotations_z.append(rz)
            soa.rotations_w.append(rw)

            sx, sy, sz = transform.scale
            soa.scales_x.append(sx)
            soa.scales_y.append(sy)
            soa.scales_z.append(sz)

        return soa

    def to_pose(self) -> Pose:
        """Convert back to AoS Pose format."""
        transforms = []
        for i in range(self.bone_count):
            t = Transform(
                position=(
                    self.positions_x[i],
                    self.positions_y[i],
                    self.positions_z[i],
                ),
                rotation=(
                    self.rotations_x[i],
                    self.rotations_y[i],
                    self.rotations_z[i],
                    self.rotations_w[i],
                ),
                scale=(
                    self.scales_x[i],
                    self.scales_y[i],
                    self.scales_z[i],
                ),
            )
            transforms.append(t)
        return Pose(transforms=transforms)

    def clear(self) -> None:
        """Clear all arrays."""
        self.positions_x.clear()
        self.positions_y.clear()
        self.positions_z.clear()
        self.rotations_x.clear()
        self.rotations_y.clear()
        self.rotations_z.clear()
        self.rotations_w.clear()
        self.scales_x.clear()
        self.scales_y.clear()
        self.scales_z.clear()
        self.bone_count = 0

    def get_flat_positions(self) -> List[float]:
        """Get positions as flat [x0, y0, z0, x1, y1, z1, ...] array."""
        result = []
        for i in range(self.bone_count):
            result.extend([
                self.positions_x[i],
                self.positions_y[i],
                self.positions_z[i],
            ])
        return result

    def get_flat_rotations(self) -> List[float]:
        """Get rotations as flat [x0, y0, z0, w0, x1, y1, z1, w1, ...] array."""
        result = []
        for i in range(self.bone_count):
            result.extend([
                self.rotations_x[i],
                self.rotations_y[i],
                self.rotations_z[i],
                self.rotations_w[i],
            ])
        return result

    def get_flat_scales(self) -> List[float]:
        """Get scales as flat [x0, y0, z0, x1, y1, z1, ...] array."""
        result = []
        for i in range(self.bone_count):
            result.extend([
                self.scales_x[i],
                self.scales_y[i],
                self.scales_z[i],
            ])
        return result


# =============================================================================
# DIRTY FLAG TRACKING
# =============================================================================


class DirtyFlags(Enum):
    """Flags indicating which parts of animation state need re-evaluation."""
    NONE = 0
    PARAMETERS = 1 << 0
    STATE = 1 << 1
    GRAPH = 1 << 2
    CLIP = 1 << 3
    TIME = 1 << 4
    ALL = (1 << 5) - 1


@dataclass
class AnimationDirtyState:
    """Tracks dirty state for animation evaluation optimization.

    Attributes:
        flags: Bitmask of DirtyFlags
        last_parameters: Hash of last parameter values
        last_state_name: Last active state machine state
        last_clip_time: Last sampled clip time
        last_eval_frame: Frame number of last evaluation
        parameter_versions: Per-parameter change counters
    """
    flags: int = DirtyFlags.ALL.value
    last_parameters: int = 0
    last_state_name: str = ""
    last_clip_time: float = 0.0
    last_eval_frame: int = -1
    parameter_versions: Dict[str, int] = field(default_factory=dict)

    def is_dirty(self, flag: DirtyFlags = DirtyFlags.ALL) -> bool:
        """Check if any of the specified flags are set."""
        return (self.flags & flag.value) != 0

    def mark_dirty(self, flag: DirtyFlags) -> None:
        """Set dirty flag(s)."""
        self.flags |= flag.value

    def mark_clean(self, flag: DirtyFlags = DirtyFlags.ALL) -> None:
        """Clear dirty flag(s)."""
        self.flags &= ~flag.value

    def clear(self) -> None:
        """Clear all dirty flags."""
        self.flags = DirtyFlags.NONE.value

    def mark_all_dirty(self) -> None:
        """Set all dirty flags."""
        self.flags = DirtyFlags.ALL.value


# =============================================================================
# STATE MACHINE OUTPUT (from Phase 5)
# =============================================================================


@dataclass
class StateMachineOutput:
    """Deterministic output from simulation-phase state machine evaluation.

    This represents the state determined during Phase 5 (simulation) and
    consumed by Phase 9 (presentation) for actual animation playback.

    Attributes:
        current_state: Name of the active state
        target_state: Name of transition target (if transitioning)
        transition_progress: Normalized progress through transition (0-1)
        transition_duration: Total transition duration in seconds
        is_transitioning: Whether a transition is in progress
        state_time: Time spent in current state (seconds)
        normalized_time: Normalized time in current state (0-1)
        parameters: Snapshot of parameter values
    """
    current_state: str = ""
    target_state: str = ""
    transition_progress: float = 0.0
    transition_duration: float = 0.0
    is_transitioning: bool = False
    state_time: float = 0.0
    normalized_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state_machine(cls, sm: StateMachine) -> "StateMachineOutput":
        """Extract output from a state machine instance."""
        output = cls()
        if sm.current_state:
            output.current_state = sm.current_state.name
            output.state_time = sm.current_state.time_in_state
            output.normalized_time = sm.current_state.normalized_time

        if sm.active_transition:
            output.is_transitioning = True
            output.target_state = sm.active_transition.target_state.name
            output.transition_progress = sm.active_transition.blend_weight
            output.transition_duration = sm.active_transition.duration

        return output


# =============================================================================
# ANIMATION GRAPH COMPONENT
# =============================================================================


@dataclass
class AnimationGraphComponent:
    """Component for entities with animation graphs.

    This component stores the animation graph definition, runtime state,
    and output data for the skinning pipeline.

    Attributes:
        graph: The animation graph (may contain StateMachine, BlendTrees, etc.)
        skeleton: Reference to the skeleton for this entity
        output_pose: Last evaluated pose (AoS format)
        output_soa: Last evaluated pose (SoA format for skinning)
        enabled: Whether animation evaluation is enabled
        time_scale: Playback speed multiplier
        dirty_state: Dirty flag tracking for optimization
        state_machine_output: Output from Phase 5 state machine evaluation
        parameter_bindings: Map parameter names to gameplay properties
        clips: Registered animation clips by name
        context_cache: Cached evaluation context
        root_motion: Accumulated root motion transform
        root_motion_enabled: Whether to extract and accumulate root motion
    """
    graph: Optional[AnimationGraph] = None
    skeleton: Optional[Skeleton] = None
    output_pose: Pose = field(default_factory=Pose)
    output_soa: BoneTransformSoA = field(default_factory=BoneTransformSoA)
    enabled: bool = True
    time_scale: float = 1.0
    dirty_state: AnimationDirtyState = field(default_factory=AnimationDirtyState)
    state_machine_output: Optional[StateMachineOutput] = None
    parameter_bindings: Dict[str, str] = field(default_factory=dict)
    clips: Dict[str, AnimationClip] = field(default_factory=dict)
    context_cache: Optional[GraphContext] = None
    root_motion: Transform = field(default_factory=Transform.identity)
    root_motion_enabled: bool = False

    # Runtime state
    _current_time: float = 0.0
    _frame_count: int = 0

    def register_clip(self, name: str, clip: AnimationClip) -> None:
        """Register an animation clip by name."""
        self.clips[name] = clip
        self.dirty_state.mark_dirty(DirtyFlags.CLIP)

    def get_clip(self, name: str) -> Optional[AnimationClip]:
        """Get a registered clip by name."""
        return self.clips.get(name)

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a graph parameter value."""
        if self.graph:
            result = self.graph.set_parameter(name, value)
            if result:
                self.dirty_state.mark_dirty(DirtyFlags.PARAMETERS)
            return result
        return False

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get a graph parameter value."""
        if self.graph:
            return self.graph.get_parameter(name)
        return None

    def update_time(self, dt: float) -> None:
        """Advance animation time."""
        self._current_time += dt * self.time_scale
        self._frame_count += 1
        self.dirty_state.mark_dirty(DirtyFlags.TIME)

    def invalidate(self) -> None:
        """Force full re-evaluation on next update."""
        self.dirty_state.mark_all_dirty()
        if self.graph:
            self.graph.invalidate()


# =============================================================================
# CLIP SAMPLER
# =============================================================================


class ClipSampler:
    """Samples animation clips with interpolation.

    Handles clip playback including looping, time scaling, and keyframe
    interpolation. Caches recently sampled clips for efficiency.
    """

    def __init__(self, cache_size: int = 16):
        self._cache: Dict[Tuple[str, float], Pose] = {}
        self._cache_size = cache_size
        self._cache_order: List[Tuple[str, float]] = []

    def sample(
        self,
        clip: AnimationClip,
        time: float,
        bone_count: int,
        use_cache: bool = True,
    ) -> Pose:
        """Sample a clip at the given time.

        Args:
            clip: Animation clip to sample
            time: Sample time in seconds
            bone_count: Number of bones in the skeleton
            use_cache: Whether to use/update the pose cache

        Returns:
            Sampled pose at the given time.
        """
        cache_key = (clip.name, round(time, 4))

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        pose = clip.sample(time, bone_count)

        if use_cache:
            self._update_cache(cache_key, pose)

        return pose

    def sample_blended(
        self,
        clip_a: AnimationClip,
        time_a: float,
        clip_b: AnimationClip,
        time_b: float,
        weight: float,
        bone_count: int,
    ) -> Pose:
        """Sample two clips and blend them.

        Args:
            clip_a: First clip
            time_a: Sample time for first clip
            clip_b: Second clip
            time_b: Sample time for second clip
            weight: Blend weight (0 = clip_a, 1 = clip_b)
            bone_count: Number of bones

        Returns:
            Blended pose.
        """
        pose_a = self.sample(clip_a, time_a, bone_count)
        pose_b = self.sample(clip_b, time_b, bone_count)
        return pose_a.lerp(pose_b, weight)

    def _update_cache(self, key: Tuple[str, float], pose: Pose) -> None:
        """Update the pose cache with LRU eviction."""
        if key in self._cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
        else:
            if len(self._cache) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[key] = pose
            self._cache_order.append(key)

    def clear_cache(self) -> None:
        """Clear the pose cache."""
        self._cache.clear()
        self._cache_order.clear()


# =============================================================================
# BLEND TREE EVALUATOR
# =============================================================================


class BlendTreeEvaluator:
    """Evaluates blend trees for parametric animation blending.

    Handles 1D, 2D, and Direct blend trees, computing weights and
    blending child poses based on parameter values.
    """

    def __init__(self, clip_sampler: ClipSampler):
        self._sampler = clip_sampler

    def evaluate(
        self,
        tree: BlendTree,
        context: GraphContext,
        clips: Dict[str, AnimationClip],
    ) -> Pose:
        """Evaluate a blend tree and return the blended pose.

        Args:
            tree: Blend tree to evaluate
            context: Evaluation context with parameters
            clips: Dictionary of available animation clips

        Returns:
            Blended pose from the tree evaluation.
        """
        if isinstance(tree, BlendTree1D):
            return self._evaluate_1d(tree, context, clips)
        elif isinstance(tree, BlendTree2D):
            return self._evaluate_2d(tree, context, clips)
        elif isinstance(tree, BlendTreeDirect):
            return self._evaluate_direct(tree, context, clips)
        else:
            # Generic blend tree - use standard evaluation
            return tree.evaluate(context)

    def _evaluate_1d(
        self,
        tree: BlendTree1D,
        context: GraphContext,
        clips: Dict[str, AnimationClip],
    ) -> Pose:
        """Evaluate a 1D blend tree."""
        if not tree.entries:
            return Pose()

        weights = tree.get_weights(context)
        bone_count = context.skeleton.bone_count() if context.skeleton else 64

        # Blend poses based on weights
        result_pose: Optional[Pose] = None
        total_weight = 0.0

        for idx, weight in weights.items():
            if weight <= 0:
                continue

            entry = tree.entries[idx]
            child_pose = entry.node.evaluate(context)

            if result_pose is None:
                result_pose = Pose.identity(child_pose.bone_count())

            # Weighted accumulation
            for i in range(min(result_pose.bone_count(), child_pose.bone_count())):
                if total_weight == 0:
                    result_pose.transforms[i] = child_pose.transforms[i]
                else:
                    # Incremental blending
                    blend_factor = weight / (total_weight + weight)
                    result_pose.transforms[i] = result_pose.transforms[i].lerp(
                        child_pose.transforms[i], blend_factor
                    )

            total_weight += weight

        return result_pose or Pose()

    def _evaluate_2d(
        self,
        tree: BlendTree2D,
        context: GraphContext,
        clips: Dict[str, AnimationClip],
    ) -> Pose:
        """Evaluate a 2D blend tree."""
        # Use the tree's built-in evaluation which handles triangulation
        return tree.evaluate(context)

    def _evaluate_direct(
        self,
        tree: BlendTreeDirect,
        context: GraphContext,
        clips: Dict[str, AnimationClip],
    ) -> Pose:
        """Evaluate a direct weight blend tree."""
        return tree.evaluate(context)


# =============================================================================
# ANIMATION GRAPH SYSTEM
# =============================================================================


@system(
    phase="animation",
    priority=ANIMATION_SYSTEM_CONFIG.PRIORITY_ANIMATION_GRAPH,
    reads=("AnimationGraphComponent", "StateMachineOutput"),
    writes=("AnimationGraphComponent",),
)
class AnimationGraphSystem:
    """Presentation-phase animation graph evaluation system (T-AN-9.3).

    This system evaluates animation graphs for all entities with animation
    components. It reads state machine outputs from Phase 5 (simulation),
    samples clips, evaluates blend trees, and outputs bone transforms in
    SoA format for the skinning pipeline.

    Features:
    - Dirty-flag driven selective evaluation
    - Task-parallel per-entity evaluation
    - SoA output for efficient GPU upload
    - Clip sampling with caching
    - Blend tree evaluation (1D, 2D, Direct)
    - State machine transition blending

    Usage:
        system = AnimationGraphSystem()
        system.set_task_scheduler(scheduler)

        # Each frame:
        system.update(world, dt, entity_components)

        # For each entity, output is in component.output_soa
    """

    def __init__(self):
        self._clip_sampler = ClipSampler()
        self._blend_tree_evaluator = BlendTreeEvaluator(self._clip_sampler)
        self._task_scheduler: Optional["TaskScheduler"] = None
        self._parallel_threshold: int = 4  # Min entities for parallel dispatch
        self._current_frame: int = 0

        # Animation provider for external clip sources
        self._animation_provider: Optional[Callable[[str, float], Pose]] = None

        # Statistics
        self._entities_evaluated: int = 0
        self._entities_skipped: int = 0
        self._parallel_batches: int = 0

    def set_task_scheduler(self, scheduler: "TaskScheduler") -> None:
        """Set the task scheduler for parallel execution.

        Args:
            scheduler: TaskScheduler instance from engine.core.tasks
        """
        self._task_scheduler = scheduler

    def set_animation_provider(
        self,
        provider: Callable[[str, float], Pose],
    ) -> None:
        """Set function that provides animation poses from external sources.

        Args:
            provider: Function(clip_name, time) -> Pose
        """
        self._animation_provider = provider

    def set_parallel_threshold(self, threshold: int) -> None:
        """Set minimum entity count for parallel dispatch.

        Args:
            threshold: Minimum entities to trigger parallel evaluation
        """
        self._parallel_threshold = max(1, threshold)

    def update(
        self,
        world: "World",
        dt: float,
        entity_components: List[Tuple["Entity", AnimationGraphComponent]],
    ) -> None:
        """Update all animation graph components.

        Args:
            world: ECS world
            dt: Delta time in seconds
            entity_components: List of (entity, component) tuples
        """
        self._current_frame += 1
        self._entities_evaluated = 0
        self._entities_skipped = 0

        if not entity_components:
            return

        # Filter enabled components
        enabled = [
            (entity, comp) for entity, comp in entity_components
            if comp.enabled
        ]

        if not enabled:
            return

        # Choose serial or parallel execution
        if (self._task_scheduler is not None and
                len(enabled) >= self._parallel_threshold):
            self._update_parallel(enabled, dt)
        else:
            self._update_serial(enabled, dt)

    def _update_serial(
        self,
        entity_components: List[Tuple["Entity", AnimationGraphComponent]],
        dt: float,
    ) -> None:
        """Serial evaluation of all components."""
        for entity, component in entity_components:
            self._evaluate_entity(entity, component, dt)

    def _update_parallel(
        self,
        entity_components: List[Tuple["Entity", AnimationGraphComponent]],
        dt: float,
    ) -> None:
        """Parallel evaluation using task scheduler."""
        assert self._task_scheduler is not None

        # Submit tasks for each entity
        handles: List["TaskHandle"] = []

        for entity, component in entity_components:
            handle = self._task_scheduler.submit(
                self._evaluate_entity, entity, component, dt
            )
            handles.append(handle)

        # Wait for all tasks to complete
        self._task_scheduler.wait_all(handles)
        self._parallel_batches += 1

    def _evaluate_entity(
        self,
        entity: "Entity",
        component: AnimationGraphComponent,
        dt: float,
    ) -> None:
        """Evaluate animation for a single entity.

        This is the core evaluation method that:
        1. Checks dirty flags for optimization
        2. Reads state machine output from Phase 5
        3. Evaluates the animation graph
        4. Converts output to SoA format

        Args:
            entity: The entity being evaluated
            component: Animation graph component
            dt: Delta time in seconds
        """
        # Update time tracking
        component.update_time(dt)

        # Check if evaluation can be skipped (clean state)
        if not self._should_evaluate(component):
            self._entities_skipped += 1
            return

        self._entities_evaluated += 1

        # Get or create evaluation context
        context = self._create_context(component, dt)

        # Evaluate the graph
        if component.graph:
            pose = self._evaluate_graph(component, context)
        elif component.state_machine_output:
            # Allow clip-based evaluation without a full graph
            pose = self._evaluate_with_state_machine_output(component, context)
        else:
            pose = Pose()

        # Store output in both formats
        component.output_pose = pose
        component.output_soa = BoneTransformSoA.from_pose(pose)

        # Extract root motion if enabled
        if component.root_motion_enabled and pose.root_motion:
            component.root_motion = pose.root_motion

        # Clear dirty flags
        component.dirty_state.clear()
        component.dirty_state.last_eval_frame = self._current_frame

    def _should_evaluate(self, component: AnimationGraphComponent) -> bool:
        """Check if component needs evaluation based on dirty flags.

        Returns True if any of the following conditions are met:
        - First evaluation (no previous frame)
        - Any dirty flag is set
        - State machine state has changed
        - Parameters have changed

        Args:
            component: Component to check

        Returns:
            True if evaluation is needed.
        """
        dirty_state = component.dirty_state

        # Always evaluate on first frame
        if dirty_state.last_eval_frame < 0:
            return True

        # Check dirty flags
        if dirty_state.is_dirty():
            return True

        # Check state machine output changes
        if component.state_machine_output:
            current_state = component.state_machine_output.current_state
            if current_state != dirty_state.last_state_name:
                dirty_state.last_state_name = current_state
                return True

        # Check parameter hash changes
        if component.graph:
            param_hash = self._compute_parameter_hash(component.graph)
            if param_hash != dirty_state.last_parameters:
                dirty_state.last_parameters = param_hash
                return True

        return False

    def _compute_parameter_hash(self, graph: AnimationGraph) -> int:
        """Compute a hash of all parameter values for change detection."""
        values = []
        for name, param in sorted(graph.parameters.items()):
            values.append((name, param.value))
        return hash(tuple(values))

    def _create_context(
        self,
        component: AnimationGraphComponent,
        dt: float,
    ) -> GraphContext:
        """Create or update evaluation context for a component."""
        context = GraphContext(
            parameters=component.graph.parameters if component.graph else {},
            dt=dt * component.time_scale,
            skeleton=component.skeleton,
            current_time=component._current_time,
            tick=component._frame_count,
        )
        component.context_cache = context
        return context

    def _evaluate_graph(
        self,
        component: AnimationGraphComponent,
        context: GraphContext,
    ) -> Pose:
        """Evaluate the animation graph and return the output pose.

        Handles:
        - State machine evaluation with Phase 5 output
        - Blend tree evaluation
        - Direct graph evaluation

        Args:
            component: Animation component with graph
            context: Evaluation context

        Returns:
            Evaluated pose.
        """
        graph = component.graph
        if not graph:
            return Pose()

        # If there's state machine output from Phase 5, use it
        if component.state_machine_output:
            return self._evaluate_with_state_machine_output(
                component, context
            )

        # Otherwise, evaluate the graph normally
        return graph.evaluate(context)

    def _evaluate_with_state_machine_output(
        self,
        component: AnimationGraphComponent,
        context: GraphContext,
    ) -> Pose:
        """Evaluate using pre-computed state machine output from Phase 5.

        This method reads the deterministic state from simulation phase
        and performs the actual animation playback.

        Args:
            component: Animation component
            context: Evaluation context

        Returns:
            Evaluated pose.
        """
        sm_output = component.state_machine_output
        if not sm_output:
            return Pose()

        bone_count = context.skeleton.bone_count() if context.skeleton else 64

        # Get current state animation
        current_clip = component.get_clip(sm_output.current_state)
        if not current_clip:
            # Try to get pose from graph evaluation
            if component.graph:
                return component.graph.evaluate(context)
            return Pose()

        # Sample current state
        current_pose = self._clip_sampler.sample(
            current_clip,
            sm_output.state_time,
            bone_count,
        )

        # Handle transitions
        if sm_output.is_transitioning and sm_output.target_state:
            target_clip = component.get_clip(sm_output.target_state)
            if target_clip:
                # Sample target state (starting from beginning)
                target_time = sm_output.transition_progress * sm_output.transition_duration
                target_pose = self._clip_sampler.sample(
                    target_clip,
                    target_time,
                    bone_count,
                )
                # Blend based on transition progress
                return current_pose.lerp(target_pose, sm_output.transition_progress)

        return current_pose

    def sample_animation(
        self,
        clip_name: str,
        time: float,
        component: Optional[AnimationGraphComponent] = None,
    ) -> Pose:
        """Sample an animation clip at a given time.

        Public API for sampling clips directly.

        Args:
            clip_name: Name of the clip to sample
            time: Time in seconds
            component: Optional component to get clip from

        Returns:
            Sampled pose, or empty pose if clip not found.
        """
        # Try external provider first
        if self._animation_provider:
            return self._animation_provider(clip_name, time)

        # Try component clips
        if component:
            clip = component.get_clip(clip_name)
            if clip:
                bone_count = (
                    component.skeleton.bone_count()
                    if component.skeleton else 64
                )
                return self._clip_sampler.sample(clip, time, bone_count)

        return Pose()

    def sync_parameters_from_gameplay(
        self,
        component: AnimationGraphComponent,
        gameplay_data: Dict[str, Any],
    ) -> int:
        """Sync graph parameters from gameplay data.

        Args:
            component: Animation graph component
            gameplay_data: Dictionary of gameplay properties

        Returns:
            Number of parameters updated.
        """
        updated = 0
        for param_name, gameplay_prop in component.parameter_bindings.items():
            if gameplay_prop in gameplay_data:
                if component.set_parameter(param_name, gameplay_data[gameplay_prop]):
                    updated += 1
        return updated

    def trigger_transition(
        self,
        component: AnimationGraphComponent,
        target_state: str,
        duration: Optional[float] = None,
    ) -> bool:
        """Manually trigger a state transition.

        Args:
            component: Animation graph component
            target_state: Target state name
            duration: Optional override for transition duration

        Returns:
            True if transition was started.
        """
        if duration is None:
            duration = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION

        # Update state machine output for next evaluation
        if component.state_machine_output:
            current = component.state_machine_output.current_state
            component.state_machine_output = StateMachineOutput(
                current_state=current,
                target_state=target_state,
                transition_progress=0.0,
                transition_duration=duration,
                is_transitioning=True,
                state_time=0.0,
                normalized_time=0.0,
            )
            component.dirty_state.mark_dirty(DirtyFlags.STATE)
            return True

        return False

    def force_state(
        self,
        component: AnimationGraphComponent,
        state_name: str,
    ) -> bool:
        """Force immediate transition to a state (no blend).

        Args:
            component: Animation graph component
            state_name: State to switch to

        Returns:
            True if state was changed.
        """
        component.state_machine_output = StateMachineOutput(
            current_state=state_name,
            is_transitioning=False,
            state_time=0.0,
            normalized_time=0.0,
        )
        component.dirty_state.mark_dirty(DirtyFlags.STATE)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics.

        Returns:
            Dictionary with evaluation statistics.
        """
        return {
            "current_frame": self._current_frame,
            "entities_evaluated": self._entities_evaluated,
            "entities_skipped": self._entities_skipped,
            "parallel_batches": self._parallel_batches,
            "clip_cache_size": len(self._clip_sampler._cache),
        }

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._clip_sampler.clear_cache()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # System decorator
    "system",
    # SoA format
    "BoneTransformSoA",
    # Dirty tracking
    "DirtyFlags",
    "AnimationDirtyState",
    # State machine output
    "StateMachineOutput",
    # Component
    "AnimationGraphComponent",
    # Re-export from graph module for convenience
    "GraphParameter",
    "ParameterType",
    # Evaluators
    "ClipSampler",
    "BlendTreeEvaluator",
    # Main system
    "AnimationGraphSystem",
]
