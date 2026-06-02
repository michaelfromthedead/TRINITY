"""
Individual blend nodes for animation graphs.

Provides the building blocks for animation processing:
- ClipNode: Plays a single animation clip
- BlendNode: Blends two inputs by alpha
- AdditiveNode: Applies additive animation on base
- LayerNode: Multiple layers with masks
- MirrorNode: Mirrors animation left/right
- TimeScaleNode: Modifies playback speed
- LoopNode: Controls loop behavior (once, repeat, ping-pong)
- SubGraphNode: Evaluates nested animation graphs

Each node implements evaluate(context) -> Pose and can be combined
in animation graphs for complex animation behaviors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .animation_graph import (
    AnimationGraph,
    AnimationNode,
    BoneMask,
    GraphContext,
    GraphParameter,
    Pose,
    Skeleton,
    Transform,
)


# =============================================================================
# ANIMATION CLIP (Simplified representation)
# =============================================================================


@dataclass
class AnimationKeyframe:
    """A keyframe in an animation track."""

    time: float
    value: Transform
    interpolation: str = "linear"  # "linear", "step", "cubic"


@dataclass
class AnimationTrack:
    """An animation track for a single bone."""

    bone_index: int
    keyframes: List[AnimationKeyframe] = field(default_factory=list)

    def sample(self, time: float) -> Transform:
        """Sample the track at a given time."""
        if not self.keyframes:
            return Transform.identity()

        if len(self.keyframes) == 1:
            return self.keyframes[0].value.copy()

        # Find bracketing keyframes
        lower_kf = self.keyframes[0]
        upper_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.time <= time:
                lower_kf = kf
            if kf.time >= time:
                upper_kf = kf
                break

        if lower_kf == upper_kf:
            return lower_kf.value.copy()

        # Interpolate
        t = (time - lower_kf.time) / (upper_kf.time - lower_kf.time)
        t = max(0.0, min(1.0, t))

        if lower_kf.interpolation == "step":
            return lower_kf.value.copy()
        else:
            return lower_kf.value.lerp(upper_kf.value, t)


class LoopMode(Enum):
    """Animation loop modes."""

    ONCE = auto()
    LOOP = auto()
    PING_PONG = auto()
    CLAMP = auto()


@dataclass
class AnimationClip:
    """
    A container for animation data.

    An animation clip contains tracks for each bone with keyframes.
    It can be sampled at any time to produce a pose.
    """

    name: str
    duration: float = 0.0
    frame_rate: float = 30.0
    tracks: Dict[int, AnimationTrack] = field(default_factory=dict)
    loop_mode: LoopMode = LoopMode.LOOP
    root_motion: bool = False

    # Events at specific times
    events: List[Tuple[float, str]] = field(default_factory=list)

    def add_track(self, bone_index: int) -> AnimationTrack:
        """Add or get a track for a bone."""
        if bone_index not in self.tracks:
            self.tracks[bone_index] = AnimationTrack(bone_index=bone_index)
        return self.tracks[bone_index]

    def add_keyframe(self, bone_index: int, time: float,
                     transform: Transform, interpolation: str = "linear") -> None:
        """Add a keyframe to a bone's track."""
        track = self.add_track(bone_index)
        kf = AnimationKeyframe(time=time, value=transform, interpolation=interpolation)
        track.keyframes.append(kf)
        track.keyframes.sort(key=lambda k: k.time)

        # Update duration
        if time > self.duration:
            self.duration = time

    def sample(self, time: float, bone_count: int) -> Pose:
        """Sample the clip at a given time."""
        # Apply loop mode
        if self.duration > 0:
            if self.loop_mode == LoopMode.LOOP:
                time = time % self.duration
            elif self.loop_mode == LoopMode.PING_PONG:
                cycle = int(time / self.duration)
                time = time % self.duration
                if cycle % 2 == 1:
                    time = self.duration - time
            elif self.loop_mode == LoopMode.CLAMP:
                time = max(0, min(self.duration, time))
            elif self.loop_mode == LoopMode.ONCE:
                if time > self.duration:
                    time = self.duration

        # Sample all tracks
        pose = Pose.identity(bone_count)

        for bone_index, track in self.tracks.items():
            if bone_index < bone_count:
                pose.transforms[bone_index] = track.sample(time)

        return pose

    def get_normalized_time(self, time: float) -> float:
        """Get normalized time (0-1) for the given time."""
        if self.duration <= 0:
            return 0.0
        return (time % self.duration) / self.duration if self.loop_mode == LoopMode.LOOP \
            else min(1.0, time / self.duration)

    def get_events_in_range(self, start_time: float, end_time: float) -> List[str]:
        """Get events that occur within a time range."""
        events = []
        for event_time, event_name in self.events:
            if start_time <= event_time < end_time:
                events.append(event_name)
        return events


# =============================================================================
# CLIP NODE
# =============================================================================


class ClipNode(AnimationNode):
    """
    A node that plays a single animation clip.

    The clip node handles playback control including time advancement,
    looping, and event triggering.
    """

    _abstract = False

    def __init__(self, node_id: str, clip: Optional[AnimationClip] = None) -> None:
        super().__init__(node_id)
        self.clip = clip
        self.current_time: float = 0.0
        self.play_rate: float = 1.0
        self.is_playing: bool = True

        # Callbacks
        self.on_event: Optional[Callable[[str], None]] = None
        self.on_loop: Optional[Callable[[], None]] = None
        self.on_finish: Optional[Callable[[], None]] = None

    def set_clip(self, clip: AnimationClip) -> None:
        """Set the animation clip."""
        self.clip = clip
        self.current_time = 0.0

    def play(self) -> None:
        """Start playback."""
        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def stop(self) -> None:
        """Stop playback and reset."""
        self.is_playing = False
        self.current_time = 0.0

    def seek(self, time: float) -> None:
        """Seek to a specific time."""
        self.current_time = time

    def seek_normalized(self, normalized_time: float) -> None:
        """Seek to a normalized time (0-1)."""
        if self.clip:
            self.current_time = normalized_time * self.clip.duration

    @property
    def normalized_time(self) -> float:
        """Get the current normalized time (0-1)."""
        if self.clip and self.clip.duration > 0:
            return self.current_time / self.clip.duration
        return 0.0

    @property
    def duration(self) -> float:
        """Get the clip duration."""
        return self.clip.duration if self.clip else 0.0

    @property
    def is_finished(self) -> bool:
        """Check if the clip has finished (for non-looping clips)."""
        if not self.clip:
            return True
        if self.clip.loop_mode in (LoopMode.LOOP, LoopMode.PING_PONG):
            return False
        return self.current_time >= self.clip.duration

    def advance(self, dt: float) -> None:
        """Advance playback time."""
        if not self.is_playing or not self.clip:
            return

        old_time = self.current_time
        self.current_time += dt * self.play_rate

        # Check for events
        if self.clip and self.on_event:
            events = self.clip.get_events_in_range(old_time, self.current_time)
            for event in events:
                self.on_event(event)

        # Check for loop/finish
        if self.clip:
            if self.clip.loop_mode == LoopMode.LOOP:
                if self.current_time >= self.clip.duration:
                    self.current_time = self.current_time % self.clip.duration
                    if self.on_loop:
                        self.on_loop()
            elif self.clip.loop_mode == LoopMode.ONCE:
                if self.current_time >= self.clip.duration and old_time < self.clip.duration:
                    if self.on_finish:
                        self.on_finish()
                    self.is_playing = False

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the clip node and return the current pose."""
        # Advance time based on context dt
        self.advance(context.dt)

        if not self.clip:
            return Pose()

        # Determine bone count
        bone_count = 0
        if context.skeleton:
            bone_count = context.skeleton.bone_count()
        else:
            bone_count = max(self.clip.tracks.keys(), default=0) + 1

        return self.clip.sample(self.current_time, bone_count)


# =============================================================================
# BLEND NODE
# =============================================================================


class BlendNode(AnimationNode):
    """
    A node that blends two input poses.

    The blend weight can be a fixed value or driven by a parameter.
    """

    _abstract = False

    def __init__(self, node_id: str, alpha: float = 0.5,
                 alpha_parameter: Optional[str] = None) -> None:
        super().__init__(node_id)
        self.alpha = alpha
        self.alpha_parameter = alpha_parameter

    def set_inputs(self, input_a: AnimationNode, input_b: AnimationNode) -> None:
        """Set the two input nodes."""
        self.inputs["a"] = input_a
        self.inputs["b"] = input_b

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate by blending two input poses."""
        pose_a = self.evaluate_input("a", context)
        pose_b = self.evaluate_input("b", context)

        if not pose_a:
            return pose_b or Pose()
        if not pose_b:
            return pose_a

        # Get blend weight
        alpha = self.alpha
        if self.alpha_parameter:
            alpha = context.get_parameter_float(self.alpha_parameter, self.alpha)

        alpha = max(0.0, min(1.0, alpha))

        return pose_a.lerp(pose_b, alpha)


# =============================================================================
# ADDITIVE NODE
# =============================================================================


class AdditiveNode(AnimationNode):
    """
    A node that applies an additive animation on top of a base.

    Additive animations are stored as deltas from a reference pose
    and are layered on top of the base animation.
    """

    _abstract = False

    def __init__(self, node_id: str, weight: float = 1.0,
                 weight_parameter: Optional[str] = None) -> None:
        super().__init__(node_id)
        self.weight = weight
        self.weight_parameter = weight_parameter

    def set_inputs(self, base: AnimationNode, additive: AnimationNode) -> None:
        """Set the base and additive input nodes."""
        self.inputs["base"] = base
        self.inputs["additive"] = additive

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate by applying additive pose on base."""
        base_pose = self.evaluate_input("base", context)
        additive_pose = self.evaluate_input("additive", context)

        if not base_pose:
            return Pose()
        if not additive_pose:
            return base_pose

        # Get weight
        weight = self.weight
        if self.weight_parameter:
            weight = context.get_parameter_float(self.weight_parameter, self.weight)

        weight = max(0.0, min(1.0, weight))

        return base_pose.additive_blend(additive_pose, weight)


# =============================================================================
# LAYER NODE
# =============================================================================


class LayerBlendMode(Enum):
    """Blend modes for animation layers."""

    OVERRIDE = auto()  # Replace base with layer
    ADDITIVE = auto()  # Add layer on top of base
    MULTIPLY = auto()  # Multiply base by layer (component-wise)


@dataclass
class AnimationLayerInput:
    """An input layer for the layer node."""

    node: AnimationNode
    weight: float = 1.0
    weight_parameter: Optional[str] = None
    mask: Optional[BoneMask] = None
    blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE


class LayerNode(AnimationNode):
    """
    A node that combines multiple layers with masks.

    Layers are evaluated and blended in order, with each layer
    optionally applying only to specific bones via a mask.
    """

    _abstract = False

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.layers: List[AnimationLayerInput] = []

    def add_layer(self, node: AnimationNode, weight: float = 1.0,
                  weight_parameter: Optional[str] = None,
                  mask: Optional[BoneMask] = None,
                  blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE) -> int:
        """Add a layer to the node."""
        layer = AnimationLayerInput(
            node=node,
            weight=weight,
            weight_parameter=weight_parameter,
            mask=mask,
            blend_mode=blend_mode,
        )
        self.layers.append(layer)
        return len(self.layers) - 1

    def remove_layer(self, index: int) -> bool:
        """Remove a layer by index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            return True
        return False

    def set_base(self, node: AnimationNode) -> None:
        """Set the base input (first layer with no mask)."""
        self.inputs["base"] = node

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate all layers and combine them."""
        # Start with base pose
        base_pose = self.evaluate_input("base", context)
        if not base_pose:
            base_pose = Pose()

        result_pose = base_pose.copy()

        # Apply each layer
        for layer in self.layers:
            # Get layer weight
            weight = layer.weight
            if layer.weight_parameter:
                weight = context.get_parameter_float(layer.weight_parameter, layer.weight)

            if weight <= 0:
                continue

            # Evaluate layer
            layer_pose = layer.node.evaluate(context)
            if not layer_pose:
                continue

            # Apply with mask
            if layer.mask:
                # Apply only to masked bones
                for i in range(min(result_pose.bone_count(), layer_pose.bone_count())):
                    mask_weight = layer.mask.get_weight(i) * weight
                    if mask_weight <= 0:
                        continue

                    if layer.blend_mode == LayerBlendMode.ADDITIVE:
                        # Additive blend
                        additive = Transform.identity().lerp(layer_pose.transforms[i], mask_weight)
                        result_pose.transforms[i] = result_pose.transforms[i] + additive
                    else:
                        # Override blend
                        result_pose.transforms[i] = result_pose.transforms[i].lerp(
                            layer_pose.transforms[i], mask_weight
                        )
            else:
                # Apply to all bones
                if layer.blend_mode == LayerBlendMode.ADDITIVE:
                    result_pose = result_pose.additive_blend(layer_pose, weight)
                else:
                    result_pose = result_pose.lerp(layer_pose, weight)

        return result_pose


# =============================================================================
# MIRROR NODE
# =============================================================================


@dataclass
class BoneMirrorPair:
    """A pair of bones that should be mirrored."""

    left_index: int
    right_index: int


class MirrorNode(AnimationNode):
    """
    A node that mirrors an animation left/right.

    Useful for creating right-handed versions of left-handed animations
    or vice versa.
    """

    _abstract = False

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.mirror_pairs: List[BoneMirrorPair] = []
        self.mirror_axis: int = 0  # 0=X, 1=Y, 2=Z

    def add_mirror_pair(self, left_index: int, right_index: int) -> None:
        """Add a bone mirror pair."""
        self.mirror_pairs.append(BoneMirrorPair(left_index, right_index))

    def set_mirror_pairs_from_skeleton(
        self, skeleton: Skeleton,
        left_prefix: str = "Left",
        right_prefix: str = "Right"
    ) -> None:
        """Automatically detect mirror pairs from bone names."""
        self.mirror_pairs.clear()

        for bone in skeleton.bones:
            if bone.name.startswith(left_prefix):
                right_name = right_prefix + bone.name[len(left_prefix):]
                right_bone = skeleton.get_bone_by_name(right_name)
                if right_bone:
                    self.mirror_pairs.append(
                        BoneMirrorPair(bone.index, right_bone.index)
                    )

    def set_input(self, node: AnimationNode) -> None:
        """Set the input node."""
        self.inputs["input"] = node

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate by mirroring the input pose."""
        input_pose = self.evaluate_input("input", context)
        if not input_pose:
            return Pose()

        result_pose = input_pose.copy()

        # Swap mirror pairs
        for pair in self.mirror_pairs:
            if (pair.left_index < len(result_pose.transforms) and
                    pair.right_index < len(result_pose.transforms)):
                # Swap transforms
                left_transform = input_pose.transforms[pair.left_index].copy()
                right_transform = input_pose.transforms[pair.right_index].copy()

                # Mirror the transforms
                result_pose.transforms[pair.left_index] = self._mirror_transform(right_transform)
                result_pose.transforms[pair.right_index] = self._mirror_transform(left_transform)

        # Mirror center bones
        for i, transform in enumerate(result_pose.transforms):
            is_paired = any(
                pair.left_index == i or pair.right_index == i
                for pair in self.mirror_pairs
            )
            if not is_paired:
                result_pose.transforms[i] = self._mirror_transform(transform)

        return result_pose

    def _mirror_transform(self, transform: Transform) -> Transform:
        """Mirror a transform along the mirror axis."""
        pos = list(transform.position)
        rot = list(transform.rotation)

        # Mirror position
        pos[self.mirror_axis] = -pos[self.mirror_axis]

        # Mirror rotation
        # For quaternion mirroring, negate appropriate components
        if self.mirror_axis == 0:  # X-axis
            rot[1] = -rot[1]  # Negate Y
            rot[2] = -rot[2]  # Negate Z
        elif self.mirror_axis == 1:  # Y-axis
            rot[0] = -rot[0]  # Negate X
            rot[2] = -rot[2]  # Negate Z
        else:  # Z-axis
            rot[0] = -rot[0]  # Negate X
            rot[1] = -rot[1]  # Negate Y

        return Transform(
            position=tuple(pos),
            rotation=tuple(rot),
            scale=transform.scale,
        )


# =============================================================================
# TIME SCALE NODE
# =============================================================================


class TimeScaleNode(AnimationNode):
    """
    A node that modifies the playback speed of its input.

    Can use a fixed scale or a parameter-driven scale.
    """

    _abstract = False

    def __init__(self, node_id: str, scale: float = 1.0,
                 scale_parameter: Optional[str] = None) -> None:
        super().__init__(node_id)
        self.scale = scale
        self.scale_parameter = scale_parameter

    def set_input(self, node: AnimationNode) -> None:
        """Set the input node."""
        self.inputs["input"] = node

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate with modified time scale."""
        # Get scale
        scale = self.scale
        if self.scale_parameter:
            scale = context.get_parameter_float(self.scale_parameter, self.scale)

        scale = max(0.0, scale)  # Prevent negative scale

        # Create modified context with scaled dt
        scaled_context = GraphContext(
            parameters=context.parameters,
            dt=context.dt * scale,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=context.normalized_time,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
        )

        return self.evaluate_input("input", scaled_context) or Pose()


# =============================================================================
# POSE CACHE NODE
# =============================================================================


class PoseCacheNode(AnimationNode):
    """
    A node that caches its input pose for a specified duration.

    Useful for reducing evaluation cost of expensive nodes or for
    implementing pose hold/freeze functionality.
    """

    _abstract = False

    def __init__(self, node_id: str, cache_duration: float = 0.0) -> None:
        super().__init__(node_id)
        self.cache_duration = cache_duration
        self._cached_pose: Optional[Pose] = None
        self._cache_time: float = 0.0

    def set_input(self, node: AnimationNode) -> None:
        """Set the input node."""
        self.inputs["input"] = node

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate with caching."""
        self._cache_time += context.dt

        # Check if cache is valid
        if (self._cached_pose is not None and
                self.cache_duration > 0 and
                self._cache_time < self.cache_duration):
            return self._cached_pose

        # Evaluate input and cache
        pose = self.evaluate_input("input", context)
        if pose:
            self._cached_pose = pose.copy()
            self._cache_time = 0.0
            return pose

        return self._cached_pose or Pose()

    def invalidate_cache(self) -> None:
        """Invalidate the cached pose."""
        super().invalidate_cache()
        self._cached_pose = None
        self._cache_time = self.cache_duration + 1  # Force re-evaluation


# =============================================================================
# SELECT NODE
# =============================================================================


class SelectNode(AnimationNode):
    """
    A node that selects between multiple inputs based on a parameter.

    The selection can be integer-based (direct index) or enum-based.
    """

    _abstract = False

    def __init__(self, node_id: str, selector_parameter: str) -> None:
        super().__init__(node_id)
        self.selector_parameter = selector_parameter
        self.options: List[AnimationNode] = []

    def add_option(self, node: AnimationNode) -> int:
        """Add an option to select from."""
        self.options.append(node)
        return len(self.options) - 1

    def remove_option(self, index: int) -> bool:
        """Remove an option by index."""
        if 0 <= index < len(self.options):
            self.options.pop(index)
            return True
        return False

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the selected input."""
        if not self.options:
            return Pose()

        # Get selector value
        selector = context.get_parameter_int(self.selector_parameter, 0)
        selector = max(0, min(len(self.options) - 1, selector))

        return self.options[selector].evaluate(context)


# =============================================================================
# LOOP NODE
# =============================================================================


class LoopControlMode(Enum):
    """Loop control modes for LoopNode."""

    ONCE = auto()      # Play once and stop
    REPEAT = auto()    # Loop indefinitely
    PING_PONG = auto() # Play forward, then backward, repeat


class LoopNode(AnimationNode):
    """
    A node that controls loop behavior for animations.

    LoopNode wraps a child node and manages loop timing, including:
    - Start and end time within the animation
    - Loop count (finite or infinite)
    - Loop mode (once, repeat, ping-pong)

    The evaluate() method returns the pose from the child at the
    correct loop-adjusted time.
    """

    _abstract = False

    def __init__(
        self,
        node_id: str,
        loop_mode: LoopControlMode = LoopControlMode.REPEAT,
        loop_count: int = -1,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> None:
        """
        Initialize the loop node.

        Args:
            node_id: Unique identifier for this node.
            loop_mode: How the animation should loop.
            loop_count: Number of loops (-1 for infinite).
            start_time: Start time within the animation range.
            end_time: End time within the animation range (None = use full duration).
        """
        super().__init__(node_id)
        self.loop_mode = loop_mode
        self.loop_count = loop_count
        self.start_time = start_time
        self.end_time = end_time

        # Internal state
        self._current_time: float = 0.0
        self._current_loop: int = 0
        self._is_forward: bool = True  # For ping-pong
        self._is_finished: bool = False

        # Callbacks
        self.on_loop_complete: Optional[Callable[[int], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None

    def set_input(self, node: AnimationNode) -> None:
        """Set the child animation node."""
        self.inputs["input"] = node

    def reset(self) -> None:
        """Reset the loop to the beginning."""
        self._current_time = self.start_time
        self._current_loop = 0
        self._is_forward = True
        self._is_finished = False

    @property
    def is_finished(self) -> bool:
        """Check if the loop has completed all iterations."""
        return self._is_finished

    @property
    def current_loop(self) -> int:
        """Get the current loop iteration (0-based)."""
        return self._current_loop

    @property
    def normalized_loop_time(self) -> float:
        """Get the normalized time within the current loop (0-1)."""
        duration = self._get_loop_duration()
        if duration <= 0:
            return 0.0
        return (self._current_time - self.start_time) / duration

    def _get_loop_duration(self) -> float:
        """Get the duration of one loop iteration."""
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        # If no end time, use input node's duration if available
        input_node = self.inputs.get("input")
        if input_node and hasattr(input_node, "duration"):
            return max(0.0, input_node.duration - self.start_time)
        return 1.0  # Default duration

    def _advance_time(self, dt: float) -> float:
        """
        Advance time and handle loop transitions.

        Returns the effective time to sample the child node.
        """
        if self._is_finished:
            return self._current_time

        duration = self._get_loop_duration()
        if duration <= 0:
            return self.start_time

        # Advance time based on direction
        if self._is_forward:
            self._current_time += dt
        else:
            self._current_time -= dt

        effective_time = self._current_time

        # Handle loop boundary
        if self.loop_mode == LoopControlMode.ONCE:
            # Play once and stop
            if self._current_time >= self.start_time + duration:
                self._current_time = self.start_time + duration
                effective_time = self._current_time
                self._is_finished = True
                if self.on_finished:
                    self.on_finished()

        elif self.loop_mode == LoopControlMode.REPEAT:
            # Loop indefinitely or count times
            if self._current_time >= self.start_time + duration:
                self._current_loop += 1
                if self.on_loop_complete:
                    self.on_loop_complete(self._current_loop)

                if self.loop_count > 0 and self._current_loop >= self.loop_count:
                    self._current_time = self.start_time + duration
                    effective_time = self._current_time
                    self._is_finished = True
                    if self.on_finished:
                        self.on_finished()
                else:
                    # Wrap around
                    overflow = self._current_time - (self.start_time + duration)
                    self._current_time = self.start_time + (overflow % duration)
                    effective_time = self._current_time

        elif self.loop_mode == LoopControlMode.PING_PONG:
            # Play forward then backward
            if self._is_forward and self._current_time >= self.start_time + duration:
                self._current_time = self.start_time + duration
                self._is_forward = False
                effective_time = self._current_time
            elif not self._is_forward and self._current_time <= self.start_time:
                self._current_time = self.start_time
                self._is_forward = True
                self._current_loop += 1
                if self.on_loop_complete:
                    self.on_loop_complete(self._current_loop)

                if self.loop_count > 0 and self._current_loop >= self.loop_count:
                    self._is_finished = True
                    if self.on_finished:
                        self.on_finished()

                effective_time = self._current_time

        return effective_time

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the loop node and return the pose at the current loop time."""
        # Advance time
        effective_time = self._advance_time(context.dt)

        # Get input node
        input_node = self.inputs.get("input")
        if not input_node:
            return Pose()

        # Create context with adjusted normalized time
        duration = self._get_loop_duration()
        if duration > 0:
            normalized = (effective_time - self.start_time) / duration
        else:
            normalized = 0.0

        loop_context = GraphContext(
            parameters=context.parameters,
            dt=0.0,  # Child doesn't advance time; we control it
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=normalized,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
        )

        # If the input is a ClipNode, seek to the effective time
        if hasattr(input_node, "seek"):
            input_node.seek(effective_time)

        return input_node.evaluate(loop_context)


# =============================================================================
# SUBGRAPH NODE
# =============================================================================


class SubGraphNode(AnimationNode):
    """
    A node that evaluates a nested animation graph.

    SubGraphNode enables hierarchical graph composition by containing
    a reference to another AnimationGraph. Parameters can be passed
    from the parent context to the nested graph.

    This is useful for:
    - Reusable animation behaviors
    - Complex state machines as subgraphs
    - Modular animation systems
    """

    _abstract = False

    def __init__(
        self,
        node_id: str,
        graph: Optional["AnimationGraph"] = None,
        parameter_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the subgraph node.

        Args:
            node_id: Unique identifier for this node.
            graph: The nested animation graph to evaluate.
            parameter_mapping: Maps parent parameter names to child parameter names.
                              Format: {child_param_name: parent_param_name}
        """
        super().__init__(node_id)
        self._graph: Optional["AnimationGraph"] = graph
        self.parameter_mapping: Dict[str, str] = parameter_mapping or {}

        # Optional input overrides for specific child graph inputs
        self.input_overrides: Dict[str, AnimationNode] = {}

    @property
    def graph(self) -> Optional["AnimationGraph"]:
        """Get the nested animation graph."""
        return self._graph

    @graph.setter
    def graph(self, value: Optional["AnimationGraph"]) -> None:
        """Set the nested animation graph."""
        self._graph = value
        self.invalidate_cache()

    def set_graph(self, graph: "AnimationGraph") -> None:
        """Set the nested animation graph."""
        self._graph = graph
        self.invalidate_cache()

    def map_parameter(self, child_param: str, parent_param: str) -> None:
        """Map a parent parameter to a child graph parameter.

        Args:
            child_param: Parameter name in the child graph.
            parent_param: Parameter name in the parent context.
        """
        self.parameter_mapping[child_param] = parent_param

    def unmap_parameter(self, child_param: str) -> None:
        """Remove a parameter mapping."""
        self.parameter_mapping.pop(child_param, None)

    def set_input_override(self, input_name: str, node: AnimationNode) -> None:
        """Override a specific input in the child graph with an external node.

        This allows connecting parent graph nodes to specific inputs
        within the child graph.
        """
        self.input_overrides[input_name] = node

    def clear_input_override(self, input_name: str) -> None:
        """Clear an input override."""
        self.input_overrides.pop(input_name, None)

    def _build_child_parameters(
        self, parent_context: GraphContext
    ) -> Dict[str, GraphParameter]:
        """Build parameter dict for child graph by applying mappings."""
        if not self._graph:
            return {}

        # Start with the child graph's own parameters
        child_params: Dict[str, GraphParameter] = {}
        for name, param in self._graph.parameters.items():
            # Check if this parameter is mapped to a parent parameter
            if name in self.parameter_mapping:
                parent_name = self.parameter_mapping[name]
                parent_param = parent_context.parameters.get(parent_name)
                if parent_param is not None:
                    # Create a copy with the parent's current value
                    child_params[name] = GraphParameter(
                        name=name,
                        param_type=param.param_type,
                        default_value=param.default_value,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        enum_values=param.enum_values,
                    )
                    child_params[name].value = parent_param.value
                else:
                    child_params[name] = param
            else:
                child_params[name] = param

        return child_params

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the nested graph and return its output pose."""
        if not self._graph:
            return Pose()

        # Build child context with mapped parameters
        child_params = self._build_child_parameters(context)

        child_context = GraphContext(
            parameters=child_params,
            dt=context.dt,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=context.normalized_time,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
        )

        # Apply input overrides: inject external nodes into child graph
        # This temporarily replaces inputs in the child graph
        original_inputs: Dict[str, Dict[str, Optional[AnimationNode]]] = {}
        for input_name, override_node in self.input_overrides.items():
            # Find nodes in child graph that might use this input
            for node in self._graph.nodes.values():
                if input_name in node.inputs:
                    if node.node_id not in original_inputs:
                        original_inputs[node.node_id] = {}
                    original_inputs[node.node_id][input_name] = node.inputs[input_name]
                    node.inputs[input_name] = override_node

        try:
            # Evaluate the child graph
            return self._graph.evaluate(child_context)
        finally:
            # Restore original inputs
            for node_id, inputs_dict in original_inputs.items():
                node = self._graph.nodes.get(node_id)
                if node:
                    for input_name, original_node in inputs_dict.items():
                        node.inputs[input_name] = original_node


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Clip
    "AnimationKeyframe",
    "AnimationTrack",
    "LoopMode",
    "AnimationClip",
    # Nodes
    "ClipNode",
    "BlendNode",
    "AdditiveNode",
    "LayerBlendMode",
    "AnimationLayerInput",
    "LayerNode",
    "BoneMirrorPair",
    "MirrorNode",
    "TimeScaleNode",
    "PoseCacheNode",
    "SelectNode",
    "LoopControlMode",
    "LoopNode",
    "SubGraphNode",
]
