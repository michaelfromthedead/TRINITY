"""ECS system for animation graphs.

Processes entities with animation graph components, evaluating state machines
and blend trees to produce poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import ANIMATION_SYSTEM_CONFIG

if TYPE_CHECKING:
    from engine.animation.graph import Pose as GraphPose, AnimationClip, BlendTree1D, BlendTree2D, GraphContext, Skeleton


# =============================================================================
# DIRTY FLAGS
# =============================================================================


class DirtyFlags(Enum):
    """Flags indicating which parts of animation state need re-evaluation."""
    NONE = 0
    PARAMETERS = 1 << 0
    STATE = 1 << 1
    GRAPH = 1 << 2
    CLIP = 1 << 3
    TIME = 1 << 4
    POSE = 1 << 5
    BLEND_WEIGHTS = 1 << 6
    ALL = (1 << 7) - 1


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
# STATE MACHINE OUTPUT
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


# =============================================================================
# BONE TRANSFORM SOA
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
    def from_pose(cls, pose: Any) -> "BoneTransformSoA":
        """Convert AoS Pose to SoA format.

        Accepts either AnimationPose (dict-keyed) or graph Pose (list-based).
        """
        # Check if it's a graph Pose with transforms list
        if hasattr(pose, 'transforms') and isinstance(pose.transforms, list):
            return cls._from_graph_pose(pose)

        # Otherwise it's our AnimationPose with bone_transforms dict
        if hasattr(pose, 'bone_transforms'):
            return cls._from_animation_pose(pose)

        # Empty pose
        return cls()

    @classmethod
    def _from_graph_pose(cls, pose: Any) -> "BoneTransformSoA":
        """Convert graph Pose (list-based transforms) to SoA."""
        transforms = pose.transforms
        soa = cls(bone_count=len(transforms))

        for transform in transforms:
            # Get position (tuple)
            pos = getattr(transform, 'position', (0.0, 0.0, 0.0))
            soa.positions_x.append(pos[0])
            soa.positions_y.append(pos[1])
            soa.positions_z.append(pos[2])

            # Get rotation (tuple: x, y, z, w)
            rot = getattr(transform, 'rotation', (0.0, 0.0, 0.0, 1.0))
            soa.rotations_x.append(rot[0])
            soa.rotations_y.append(rot[1])
            soa.rotations_z.append(rot[2])
            soa.rotations_w.append(rot[3])

            # Get scale (tuple)
            scale = getattr(transform, 'scale', (1.0, 1.0, 1.0))
            soa.scales_x.append(scale[0])
            soa.scales_y.append(scale[1])
            soa.scales_z.append(scale[2])

        return soa

    @classmethod
    def _from_animation_pose(cls, pose: "AnimationPose") -> "BoneTransformSoA":
        """Convert AnimationPose (dict-keyed) to SoA format."""
        soa = cls(bone_count=len(pose.bone_transforms))

        for bone_idx in sorted(pose.bone_transforms.keys()):
            transform = pose.bone_transforms[bone_idx]
            soa.positions_x.append(transform.position.x)
            soa.positions_y.append(transform.position.y)
            soa.positions_z.append(transform.position.z)

            soa.rotations_x.append(transform.rotation.x)
            soa.rotations_y.append(transform.rotation.y)
            soa.rotations_z.append(transform.rotation.z)
            soa.rotations_w.append(transform.rotation.w)

            soa.scales_x.append(transform.scale.x)
            soa.scales_y.append(transform.scale.y)
            soa.scales_z.append(transform.scale.z)

        return soa

    @classmethod
    def from_animation_pose(cls, pose: "AnimationPose") -> "BoneTransformSoA":
        """Convert AnimationPose to SoA format (explicit method)."""
        return cls._from_animation_pose(pose)

    def to_pose(self) -> Any:
        """Convert SoA back to AoS Pose format.

        Returns:
            Pose object with list of Transform objects
        """
        from engine.animation.graph import Pose, Transform

        transforms = []
        for i in range(self.bone_count):
            transform = Transform(
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
            transforms.append(transform)

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
# CLIP SAMPLER
# =============================================================================


class ClipSampler:
    """Samples animation clips with interpolation.

    Handles clip playback including looping, time scaling, and keyframe
    interpolation. Caches recently sampled clips for efficiency.
    """

    def __init__(self, cache_size: int = 16):
        self._cache: Dict[Tuple[str, float], Any] = {}
        self._cache_size = cache_size
        self._cache_order: List[Tuple[str, float]] = []

    def sample(
        self,
        clip: Any,
        time: float,
        bone_count: int,
        use_cache: bool = True,
    ) -> Any:
        """Sample a clip at the given time.

        Args:
            clip: AnimationClip to sample
            time: Sample time in seconds
            bone_count: Number of bones to produce
            use_cache: Whether to use/update the pose cache

        Returns:
            Sampled Pose at the given time.
        """
        from engine.animation.graph import Pose, Transform

        # Get clip name for cache key
        clip_name = getattr(clip, 'name', id(clip))
        cache_key = (clip_name, round(time, 4))

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Sample the clip to produce a pose
        pose = self._sample_clip(clip, time, bone_count)

        if use_cache:
            self._update_cache(cache_key, pose)

        return pose

    def _sample_clip(self, clip: Any, time: float, bone_count: int) -> Any:
        """Sample animation clip at given time.

        Args:
            clip: AnimationClip to sample
            time: Time in seconds
            bone_count: Number of bones

        Returns:
            Sampled Pose
        """
        from engine.animation.graph import Pose, Transform

        # Use the clip's sample method if available
        if hasattr(clip, 'sample'):
            return clip.sample(time, bone_count)

        # Fallback: manual sampling
        transforms = []

        # Handle looping
        duration = getattr(clip, 'duration', 1.0)
        loop_mode = getattr(clip, 'loop_mode', None)

        if loop_mode is not None and duration > 0:
            # Check loop mode name
            mode_name = getattr(loop_mode, 'name', str(loop_mode))
            if 'LOOP' in mode_name.upper():
                time = time % duration

        # Get tracks from clip
        tracks = getattr(clip, 'tracks', {})

        for bone_idx in range(bone_count):
            track = tracks.get(bone_idx)
            if track and hasattr(track, 'sample'):
                transform = track.sample(time)
            elif track:
                transform = self._sample_track(track, time)
            else:
                transform = Transform()
            transforms.append(transform)

        return Pose(transforms=transforms)

    def _sample_track(self, track: Any, time: float) -> Any:
        """Sample a single animation track at given time.

        Args:
            track: AnimationTrack to sample
            time: Time in seconds

        Returns:
            Interpolated Transform
        """
        from engine.animation.graph import Transform

        keyframes = getattr(track, 'keyframes', [])
        if not keyframes:
            return Transform()

        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf in keyframes:
            kf_time = getattr(kf, 'time', 0.0)
            if kf_time <= time:
                prev_kf = kf
            if kf_time >= time and next_kf is None:
                next_kf = kf

        if prev_kf is None:
            prev_kf = keyframes[0]
        if next_kf is None:
            next_kf = keyframes[-1]

        # Same keyframe or at exact keyframe time
        prev_time = getattr(prev_kf, 'time', 0.0)
        next_time = getattr(next_kf, 'time', 0.0)

        if prev_kf is next_kf or prev_time == next_time:
            return getattr(prev_kf, 'value', Transform())

        # Interpolate between keyframes
        t = (time - prev_time) / (next_time - prev_time)
        t = max(0.0, min(1.0, t))

        prev_value = getattr(prev_kf, 'value', Transform())
        next_value = getattr(next_kf, 'value', Transform())

        if hasattr(prev_value, 'lerp'):
            return prev_value.lerp(next_value, t)

        return prev_value

    def sample_blended(
        self,
        clip1: Any,
        time1: float,
        clip2: Any,
        time2: float,
        weight: float,
        bone_count: int,
    ) -> Any:
        """Sample and blend two clips.

        Args:
            clip1: First AnimationClip
            time1: Time for first clip
            clip2: Second AnimationClip
            time2: Time for second clip
            weight: Blend weight (0 = clip1, 1 = clip2)
            bone_count: Number of bones

        Returns:
            Blended Pose
        """
        pose1 = self.sample(clip1, time1, bone_count, use_cache=False)
        pose2 = self.sample(clip2, time2, bone_count, use_cache=False)

        if hasattr(pose1, 'lerp'):
            return pose1.lerp(pose2, weight)
        elif hasattr(pose1, 'blend'):
            return pose1.blend(pose2, weight)

        return pose1

    def _update_cache(self, key: Tuple[str, float], pose: Any) -> None:
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
# PARAMETER TYPE
# =============================================================================


class ParameterType(Enum):
    """Animation graph parameter type."""
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    TRIGGER = auto()


@dataclass
class GraphParameter:
    """Animation graph parameter definition.

    Attributes:
        name: Parameter name
        param_type: Parameter type
        value: Current value
        default_value: Default value
        min_value: Minimum value (for float/int)
        max_value: Maximum value (for float/int)
    """
    name: str = ""
    param_type: ParameterType = ParameterType.FLOAT
    value: Any = 0.0
    default_value: Any = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def set_value(self, new_value: Any) -> None:
        """Set parameter value with type checking."""
        if self.param_type == ParameterType.FLOAT:
            val = float(new_value)
            if self.min_value is not None:
                val = max(self.min_value, val)
            if self.max_value is not None:
                val = min(self.max_value, val)
            self.value = val
        elif self.param_type == ParameterType.INT:
            val = int(new_value)
            if self.min_value is not None:
                val = max(int(self.min_value), val)
            if self.max_value is not None:
                val = min(int(self.max_value), val)
            self.value = val
        elif self.param_type == ParameterType.BOOL:
            self.value = bool(new_value)
        elif self.param_type == ParameterType.TRIGGER:
            self.value = bool(new_value)

    def reset(self) -> None:
        """Reset to default value."""
        self.value = self.default_value

    def consume_trigger(self) -> bool:
        """Consume trigger if set, returns previous value."""
        if self.param_type == ParameterType.TRIGGER:
            was_set = bool(self.value)
            self.value = False
            return was_set
        return False


@dataclass
class AnimationState:
    """State in animation state machine."""
    name: str = ""
    animation_clip: str = ""
    speed: float = 1.0
    loop: bool = True
    blend_time: float = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION


@dataclass
class StateTransition:
    """Transition between animation states."""
    from_state: str = ""
    to_state: str = ""
    condition: Callable[[dict[str, GraphParameter]], bool] | None = None
    duration: float = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION
    has_exit_time: bool = False
    exit_time: float = 1.0


@dataclass
class AnimationGraphInstance:
    """Runtime instance of an animation graph.

    Attributes:
        current_state: Current state name
        states: Available states
        transitions: State transitions
        parameters: Graph parameters
        state_time: Time in current state
        transition_time: Current transition time
        transitioning: Whether currently transitioning
        transition_target: Target state during transition
    """
    current_state: str = ""
    states: dict[str, AnimationState] = field(default_factory=dict)
    transitions: list[StateTransition] = field(default_factory=list)
    parameters: dict[str, GraphParameter] = field(default_factory=dict)
    state_time: float = 0.0
    transition_time: float = 0.0
    transitioning: bool = False
    transition_target: str = ""
    transition_duration: float = 0.0

    def add_state(self, state: AnimationState) -> None:
        """Add a state to the graph."""
        self.states[state.name] = state
        if not self.current_state:
            self.current_state = state.name

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition to the graph."""
        self.transitions.append(transition)

    def add_parameter(self, param: GraphParameter) -> None:
        """Add a parameter to the graph."""
        self.parameters[param.name] = param

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set parameter value.

        Returns:
            True if parameter exists
        """
        param = self.parameters.get(name)
        if param:
            param.set_value(value)
            return True
        return False

    def get_parameter(self, name: str) -> GraphParameter | None:
        """Get parameter by name."""
        return self.parameters.get(name)

    def get_current_animation(self) -> str:
        """Get current animation clip name."""
        state = self.states.get(self.current_state)
        return state.animation_clip if state else ""

    def get_blend_weight(self) -> float:
        """Get current transition blend weight (0 = from, 1 = to)."""
        if not self.transitioning:
            return 1.0
        if self.transition_duration <= 0:
            return 1.0
        return min(1.0, self.transition_time / self.transition_duration)


@dataclass
class AnimationPose:
    """Output pose from animation evaluation."""
    bone_transforms: dict[int, Transform] = field(default_factory=dict)
    root_motion: Transform = field(default_factory=Transform.identity)

    def bone_count(self) -> int:
        """Return the number of bones in this pose."""
        return len(self.bone_transforms)

    def get_bone_transform(self, bone_index: int) -> Transform:
        """Get transform for bone, identity if not present."""
        return self.bone_transforms.get(bone_index, Transform.identity())

    def blend_with(self, other: AnimationPose, weight: float) -> AnimationPose:
        """Blend this pose with another."""
        result = AnimationPose()

        all_bones = set(self.bone_transforms.keys()) | set(other.bone_transforms.keys())
        for bone_idx in all_bones:
            t1 = self.get_bone_transform(bone_idx)
            t2 = other.get_bone_transform(bone_idx)
            result.bone_transforms[bone_idx] = t1.lerp(t2, weight)

        result.root_motion = self.root_motion.lerp(other.root_motion, weight)
        return result


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "animation",
    priority: int = 100,
    reads: Tuple[str, ...] = (),
    writes: Tuple[str, ...] = (),
) -> Callable:
    """Decorator to mark a class or function as an ECS system.

    Args:
        phase: The execution phase for this system
        priority: Priority within the phase (higher = runs earlier)
        reads: Tuple of component types this system reads
        writes: Tuple of component types this system writes

    Returns:
        Decorated class/function with system metadata
    """
    def decorator(cls_or_func: Callable) -> Callable:
        cls_or_func._is_system = True
        cls_or_func._system_phase = phase
        cls_or_func._system_priority = priority
        cls_or_func._system_reads = reads
        cls_or_func._system_writes = writes
        return cls_or_func
    return decorator


# =============================================================================
# ANIMATION GRAPH COMPONENT
# =============================================================================


@dataclass
class AnimationGraphComponent:
    """Component for entities with animation graphs.

    Attributes:
        graph: Animation graph instance
        output_pose: Evaluated pose output
        enabled: Whether animation is enabled
        time_scale: Animation time scale
        skeleton: Reference to skeleton
        clips: Registered animation clips
        dirty_state: Dirty flag tracking
        state_machine_output: State machine evaluation output
        output_soa: SoA format output for GPU
        root_motion_enabled: Whether root motion extraction is enabled
    """
    graph: AnimationGraphInstance | Any | None = field(default_factory=AnimationGraphInstance)
    output_pose: AnimationPose | Any = field(default_factory=AnimationPose)
    enabled: bool = True
    time_scale: float = 1.0

    # For syncing with gameplay
    parameter_bindings: dict[str, str] = field(default_factory=dict)  # param_name -> gameplay_property

    # Extended fields for tests
    skeleton: Any | None = None
    clips: dict[str, Any] = field(default_factory=dict)
    dirty_state: AnimationDirtyState = field(default_factory=AnimationDirtyState)
    state_machine_output: StateMachineOutput | None = None
    output_soa: BoneTransformSoA = field(default_factory=BoneTransformSoA)
    root_motion_enabled: bool = False

    # Private state
    _current_time: float = 0.0
    _frame_count: int = 0

    def register_clip(self, name: str, clip: Any) -> None:
        """Register an animation clip.

        Args:
            name: Name to register clip under
            clip: AnimationClip to register
        """
        self.clips[name] = clip
        self.dirty_state.mark_dirty(DirtyFlags.CLIP)

    def get_clip(self, name: str) -> Any | None:
        """Get a registered clip by name.

        Args:
            name: Clip name

        Returns:
            AnimationClip or None if not found
        """
        return self.clips.get(name)

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a graph parameter value.

        Args:
            name: Parameter name
            value: New value

        Returns:
            True if parameter was set
        """
        if self.graph is None:
            return False

        # Try graph's set_parameter method
        if hasattr(self.graph, 'set_parameter'):
            result = self.graph.set_parameter(name, value)
            if result:
                self.dirty_state.mark_dirty(DirtyFlags.PARAMETERS)
            return result

        # Try direct parameter access
        if hasattr(self.graph, 'parameters'):
            param = self.graph.parameters.get(name)
            if param:
                if hasattr(param, 'set_value'):
                    param.set_value(value)
                else:
                    self.graph.parameters[name] = value
                self.dirty_state.mark_dirty(DirtyFlags.PARAMETERS)
                return True

        return False

    def get_parameter(self, name: str, default: float = 0.0) -> Any:
        """Get a graph parameter value.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value
        """
        if self.graph is None:
            return default

        if hasattr(self.graph, 'get_parameter'):
            param = self.graph.get_parameter(name)
            if param is not None:
                if hasattr(param, 'value'):
                    return param.value
                return param

        if hasattr(self.graph, 'parameters'):
            param = self.graph.parameters.get(name)
            if param is not None:
                if hasattr(param, 'value'):
                    return param.value
                return param

        return default

    def update_time(self, dt: float) -> None:
        """Update animation time.

        Args:
            dt: Delta time in seconds
        """
        self._current_time += dt * self.time_scale
        self._frame_count += 1
        self.dirty_state.mark_dirty(DirtyFlags.TIME)

    def invalidate(self) -> None:
        """Mark all state as dirty."""
        self.dirty_state.mark_all_dirty()


@system(
    phase="animation",
    priority=100,
    reads=("AnimationGraphComponent",),
    writes=("AnimationGraphComponent",),
)
class AnimationGraphSystem:
    """ECS system for evaluating animation graphs.

    Processes entities with AnimationGraphComponent, evaluates state machines,
    and produces poses.
    """

    def __init__(self):
        self._pose_cache: dict[str, AnimationPose] = {}
        self._animation_provider: Callable[[str, float], AnimationPose] | None = None
        self._clip_sampler: ClipSampler = ClipSampler()
        self._parallel_threshold: int = 4
        self._task_scheduler: Any = None
        self._current_frame: int = 0
        self._entities_evaluated: int = 0
        self._entities_skipped: int = 0
        self._parallel_batches: int = 0

    def set_animation_provider(
        self,
        provider: Callable[[str, float], AnimationPose]
    ) -> None:
        """Set function that provides animation poses.

        Args:
            provider: Function(clip_name, time) -> AnimationPose
        """
        self._animation_provider = provider

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, AnimationGraphComponent]]
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
        self._parallel_batches = 0

        enabled_entities = [(e, c) for e, c in entity_components if c.enabled]
        self._entities_skipped = len(entity_components) - len(enabled_entities)

        # Check if we should use parallel execution
        if (
            self._task_scheduler is not None
            and len(enabled_entities) >= self._parallel_threshold
        ):
            self._update_parallel(enabled_entities, dt)
        else:
            for entity, component in enabled_entities:
                self._update_single_entity(component, dt)
                self._entities_evaluated += 1

    def _update_single_entity(self, component: AnimationGraphComponent, dt: float) -> None:
        """Update a single entity's animation."""
        # Update component time
        scaled_dt = dt * component.time_scale
        component._current_time = getattr(component, '_current_time', 0.0) + scaled_dt
        component._frame_count = getattr(component, '_frame_count', 0) + 1

        # Mark TIME dirty
        if hasattr(component, 'dirty_state'):
            component.dirty_state.mark_dirty(DirtyFlags.TIME)

        self._update_graph(component, scaled_dt)

    def _update_parallel(
        self,
        entities: list[tuple[Entity, AnimationGraphComponent]],
        dt: float,
    ) -> None:
        """Update entities in parallel using the task scheduler."""
        if self._task_scheduler is None:
            return

        handles = []
        for entity, component in entities:
            handle = self._task_scheduler.submit(
                self._update_single_entity, component, dt
            )
            handles.append(handle)

        self._task_scheduler.wait_all(handles)
        self._parallel_batches = 1
        self._entities_evaluated = len(entities)

    def _update_graph(self, component: AnimationGraphComponent, dt: float) -> None:
        """Update single animation graph."""
        # Check if we have state_machine_output (from simulation phase)
        if hasattr(component, 'state_machine_output') and component.state_machine_output is not None:
            pose = self._evaluate_from_state_machine_output(component)
            if pose is not None:
                component.output_pose = pose
                # Convert to SoA
                if hasattr(component, 'output_soa'):
                    component.output_soa = BoneTransformSoA.from_pose(pose)
                return

        # Fallback to graph-based evaluation
        graph = component.graph
        if graph is None:
            return

        # Update state time
        graph.state_time += dt

        # Check for transitions
        if graph.transitioning:
            self._update_transition(graph, dt)
        else:
            self._check_transitions(graph)

        # Evaluate pose
        component.output_pose = self._evaluate_pose(graph)

    def _evaluate_from_state_machine_output(
        self,
        component: AnimationGraphComponent,
    ) -> Any:
        """Evaluate animation from state machine output.

        Args:
            component: Animation graph component with state_machine_output

        Returns:
            Evaluated Pose or None
        """
        from engine.animation.graph import Pose

        smo = component.state_machine_output
        current_state = smo.current_state

        # Get clip for current state
        clip = component.get_clip(current_state)
        if clip is None:
            return None

        # Determine bone count from skeleton or clip
        bone_count = 4  # Default
        if hasattr(component, 'skeleton') and component.skeleton is not None:
            bone_count = getattr(component.skeleton, 'bone_count', lambda: 4)()
        elif hasattr(clip, 'tracks'):
            bone_count = max(clip.tracks.keys(), default=-1) + 1

        # Sample the clip
        time = smo.state_time
        current_pose = self._clip_sampler.sample(clip, time, bone_count)

        # Handle transition blending
        if smo.is_transitioning and smo.target_state:
            target_clip = component.get_clip(smo.target_state)
            if target_clip:
                target_pose = self._clip_sampler.sample(target_clip, smo.transition_progress * smo.transition_duration, bone_count)
                weight = smo.transition_progress
                if hasattr(current_pose, 'lerp'):
                    current_pose = current_pose.lerp(target_pose, weight)
                elif hasattr(current_pose, 'blend'):
                    current_pose = current_pose.blend(target_pose, weight)

        return current_pose

    def _check_transitions(self, graph: AnimationGraphInstance) -> None:
        """Check if any transition conditions are met."""
        current_state = graph.states.get(graph.current_state)
        if not current_state:
            return

        for transition in graph.transitions:
            if transition.from_state != graph.current_state:
                continue

            # Check exit time
            if transition.has_exit_time:
                # Assume normalized time for looping
                if graph.state_time < transition.exit_time:
                    continue

            # Check condition
            if transition.condition is not None:
                if not transition.condition(graph.parameters):
                    continue

            # Start transition
            self._start_transition(graph, transition)
            break

    def _start_transition(
        self,
        graph: AnimationGraphInstance,
        transition: StateTransition
    ) -> None:
        """Start a state transition."""
        graph.transitioning = True
        graph.transition_target = transition.to_state
        graph.transition_time = 0.0
        graph.transition_duration = transition.duration

    def _update_transition(self, graph: AnimationGraphInstance, dt: float) -> None:
        """Update ongoing transition."""
        graph.transition_time += dt

        if graph.transition_time >= graph.transition_duration:
            # Complete transition
            graph.current_state = graph.transition_target
            graph.transitioning = False
            graph.state_time = 0.0
            graph.transition_target = ""

            # Consume any triggers
            for param in graph.parameters.values():
                param.consume_trigger()

    def _evaluate_pose(self, graph: AnimationGraphInstance) -> AnimationPose:
        """Evaluate current pose from graph state."""
        current_state = graph.states.get(graph.current_state)
        if not current_state:
            return AnimationPose()

        # Get current animation pose
        current_pose = self._sample_animation(
            current_state.animation_clip,
            graph.state_time * current_state.speed,
            current_state.loop,
        )

        if not graph.transitioning:
            return current_pose

        # Blend with target state
        target_state = graph.states.get(graph.transition_target)
        if not target_state:
            return current_pose

        target_pose = self._sample_animation(
            target_state.animation_clip,
            graph.transition_time * target_state.speed,
            target_state.loop,
        )

        blend_weight = graph.get_blend_weight()
        return current_pose.blend_with(target_pose, blend_weight)

    def _sample_animation(
        self,
        clip_name: str,
        time: float,
        loop: bool = True
    ) -> AnimationPose:
        """Sample animation at given time."""
        if self._animation_provider:
            return self._animation_provider(clip_name, time)

        # Default empty pose
        return AnimationPose()

    def sync_parameters_from_gameplay(
        self,
        component: AnimationGraphComponent,
        gameplay_data: dict[str, Any]
    ) -> int:
        """Sync graph parameters from gameplay data.

        Args:
            component: Animation graph component
            gameplay_data: Dictionary of gameplay properties

        Returns:
            Number of parameters updated
        """
        updated = 0
        for param_name, gameplay_prop in component.parameter_bindings.items():
            if gameplay_prop in gameplay_data:
                component.graph.set_parameter(param_name, gameplay_data[gameplay_prop])
                updated += 1
        return updated

    def trigger_transition(
        self,
        component: AnimationGraphComponent,
        target_state: str,
        duration: float | None = None,
    ) -> bool:
        """Manually trigger transition to target state.

        Args:
            component: Animation graph component
            target_state: Target state name
            duration: Optional transition duration override

        Returns:
            True if transition started
        """
        # Use state_machine_output if available
        if hasattr(component, 'state_machine_output'):
            smo = component.state_machine_output
            smo.is_transitioning = True
            smo.target_state = target_state
            smo.transition_progress = 0.0
            smo.transition_duration = duration if duration is not None else ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION
            return True

        graph = component.graph
        if target_state not in graph.states:
            return False

        if graph.transitioning:
            return False

        # Find matching transition
        for transition in graph.transitions:
            if transition.from_state == graph.current_state and transition.to_state == target_state:
                self._start_transition(graph, transition)
                return True

        # Force transition if no explicit transition exists
        graph.transitioning = True
        graph.transition_target = target_state
        graph.transition_time = 0.0
        graph.transition_duration = duration if duration is not None else ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION
        return True

    def force_state(
        self,
        component: AnimationGraphComponent,
        state_name: str,
    ) -> bool:
        """Force immediate state change without transition.

        Args:
            component: Animation graph component
            state_name: Target state name

        Returns:
            True if state was changed
        """
        if hasattr(component, 'state_machine_output'):
            smo = component.state_machine_output
            smo.current_state = state_name
            smo.is_transitioning = False
            smo.transition_progress = 0.0
            smo.target_state = ""
            smo.state_time = 0.0
            return True

        if hasattr(component, 'graph') and component.graph:
            component.graph.current_state = state_name
            component.graph.transitioning = False
            component.graph.state_time = 0.0
            return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics.

        Returns:
            Dictionary with performance statistics
        """
        return {
            "current_frame": self._current_frame,
            "entities_evaluated": self._entities_evaluated,
            "entities_skipped": self._entities_skipped,
            "parallel_batches": self._parallel_batches,
            "parallel_threshold": self._parallel_threshold,
        }

    def set_parallel_threshold(self, threshold: int) -> None:
        """Set the minimum entity count for parallel execution.

        Args:
            threshold: Minimum entity count (clamped to at least 1)
        """
        self._parallel_threshold = max(1, threshold)

    def set_task_scheduler(self, scheduler: Any) -> None:
        """Set the task scheduler for parallel execution.

        Args:
            scheduler: Task scheduler with submit() and wait_all() methods
        """
        self._task_scheduler = scheduler

    def sample_animation(self, clip_name: str, time: float) -> Any:
        """Sample animation at given time using the provider.

        Args:
            clip_name: Name of the clip to sample
            time: Time in seconds

        Returns:
            Sampled pose
        """
        if self._animation_provider:
            return self._animation_provider(clip_name, time)
        return None

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._pose_cache.clear()
        self._clip_sampler.clear_cache()


# =============================================================================
# BLEND TREE EVALUATOR
# =============================================================================


class BlendTreeEvaluator:
    """Evaluates blend trees to produce animation poses.

    Supports 1D and 2D blend trees with caching for efficiency.

    Attributes:
        _sampler: ClipSampler used for sampling animations
        _cache: Evaluation cache
    """

    def __init__(self, sampler: ClipSampler | None = None):
        """Initialize blend tree evaluator.

        Args:
            sampler: Optional ClipSampler to use (creates one if not provided)
        """
        self._sampler = sampler or ClipSampler()
        self._cache: Dict[int, Any] = {}

    def evaluate(
        self,
        tree: Any,
        context: Any,
        additional_params: Dict[str, float] | None = None,
    ) -> Any:
        """Evaluate a blend tree to produce a pose.

        Args:
            tree: Blend tree to evaluate (BlendTree1D or BlendTree2D)
            context: GraphContext with parameters and skeleton
            additional_params: Additional parameters to use

        Returns:
            Evaluated pose
        """
        from engine.animation.graph import Pose

        if tree is None:
            return Pose()

        tree_type = type(tree).__name__
        params = {}

        # Extract parameters from context
        if hasattr(context, 'parameters'):
            for name, param in context.parameters.items():
                if hasattr(param, 'value'):
                    params[name] = param.value
                else:
                    params[name] = param

        # Add additional params
        if additional_params:
            params.update(additional_params)

        if 'BlendTree1D' in tree_type:
            return self.evaluate_1d(tree, context, params)
        elif 'BlendTree2D' in tree_type:
            return self.evaluate_2d(tree, context, params)

        return Pose()

    def evaluate_1d(
        self,
        tree: Any,
        context: Any,
        params: Dict[str, float],
    ) -> Any:
        """Evaluate 1D blend tree.

        Args:
            tree: BlendTree1D to evaluate
            context: GraphContext
            params: Parameter values

        Returns:
            Blended pose
        """
        from engine.animation.graph import Pose

        # Get the parameter value - BlendTree1D uses 'parameter' not 'parameter_name'
        param_name = getattr(tree, 'parameter', None) or getattr(tree, 'parameter_name', 'blend')
        param_value = params.get(param_name, 0.0)

        # Get entries
        entries = getattr(tree, 'entries', [])
        if not entries:
            return Pose()

        # Find surrounding entries for blending
        lower_entry = None
        upper_entry = None

        for entry in sorted(entries, key=lambda e: e.threshold):
            if entry.threshold <= param_value:
                lower_entry = entry
            if entry.threshold >= param_value and upper_entry is None:
                upper_entry = entry

        # Single entry case
        if lower_entry is None:
            lower_entry = entries[0] if entries else None
        if upper_entry is None:
            upper_entry = entries[-1] if entries else None

        if lower_entry is None or upper_entry is None:
            return Pose()

        # Same entry - no blending needed, evaluate only once
        if lower_entry is upper_entry or lower_entry.threshold == upper_entry.threshold:
            return lower_entry.node.evaluate(context) if hasattr(lower_entry, 'node') else Pose()

        # Evaluate both nodes for blending
        lower_pose = lower_entry.node.evaluate(context) if hasattr(lower_entry, 'node') else Pose()
        upper_pose = upper_entry.node.evaluate(context) if hasattr(upper_entry, 'node') else Pose()

        # Calculate blend weight
        weight = (param_value - lower_entry.threshold) / (upper_entry.threshold - lower_entry.threshold)
        weight = max(0.0, min(1.0, weight))

        return lower_pose.lerp(upper_pose, weight) if hasattr(lower_pose, 'lerp') else lower_pose

    def evaluate_2d(
        self,
        tree: Any,
        context: Any,
        params: Dict[str, float],
    ) -> Any:
        """Evaluate 2D blend tree.

        Args:
            tree: BlendTree2D to evaluate
            context: GraphContext
            params: Parameter values

        Returns:
            Blended pose
        """
        from engine.animation.graph import Pose

        # Get parameter values
        param_x_name = getattr(tree, 'parameter_x_name', 'x')
        param_y_name = getattr(tree, 'parameter_y_name', 'y')
        x = params.get(param_x_name, 0.0)
        y = params.get(param_y_name, 0.0)

        # Get samples
        samples = getattr(tree, 'samples', [])
        if not samples:
            return Pose()

        # Simple barycentric blending for 2D
        # For now, just find closest sample
        closest = None
        closest_dist = float('inf')

        for sample in samples:
            sx = getattr(sample, 'x', 0.0)
            sy = getattr(sample, 'y', 0.0)
            dist = (x - sx) ** 2 + (y - sy) ** 2
            if dist < closest_dist:
                closest_dist = dist
                closest = sample

        if closest and hasattr(closest, 'node'):
            return closest.node.evaluate(context)

        return Pose()

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self._sampler.clear_cache()
