"""
Facial Motion Capture Playback Module.

Provides playback and retargeting of facial motion capture data.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Sequence, Tuple


# =============================================================================
# Animation Curves
# =============================================================================


class InterpolationMode(Enum):
    """Interpolation mode for keyframes."""
    LINEAR = auto()
    STEP = auto()
    CUBIC = auto()
    HERMITE = auto()


@dataclass
class Keyframe:
    """
    A single keyframe in an animation curve.

    Attributes:
        time: Time in seconds
        value: Keyframe value
        in_tangent: Incoming tangent (for cubic/hermite)
        out_tangent: Outgoing tangent (for cubic/hermite)
    """
    time: float
    value: float
    in_tangent: float = 0.0
    out_tangent: float = 0.0


@dataclass
class AnimationCurve:
    """
    An animation curve with keyframes.

    Attributes:
        name: Curve name (typically blend shape name)
        keyframes: List of keyframes sorted by time
        interpolation: Interpolation mode
    """
    name: str
    keyframes: list[Keyframe] = field(default_factory=list)
    interpolation: InterpolationMode = InterpolationMode.LINEAR

    def __post_init__(self) -> None:
        """Sort keyframes by time."""
        self.keyframes.sort(key=lambda k: k.time)

    @property
    def duration(self) -> float:
        """Get curve duration."""
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time

    @property
    def keyframe_count(self) -> int:
        """Get number of keyframes."""
        return len(self.keyframes)

    def add_keyframe(
        self,
        time: float,
        value: float,
        in_tangent: float = 0.0,
        out_tangent: float = 0.0,
    ) -> None:
        """
        Add a keyframe to the curve.

        Args:
            time: Keyframe time
            value: Keyframe value
            in_tangent: Incoming tangent
            out_tangent: Outgoing tangent
        """
        keyframe = Keyframe(time, value, in_tangent, out_tangent)

        # Insert in sorted order
        times = [k.time for k in self.keyframes]
        idx = bisect.bisect_right(times, time)
        self.keyframes.insert(idx, keyframe)

    def remove_keyframe(self, index: int) -> bool:
        """
        Remove a keyframe by index.

        Args:
            index: Keyframe index

        Returns:
            True if removed
        """
        if 0 <= index < len(self.keyframes):
            del self.keyframes[index]
            return True
        return False

    def sample(self, time: float) -> float:
        """
        Sample the curve at a specific time.

        Args:
            time: Time to sample

        Returns:
            Interpolated value
        """
        if not self.keyframes:
            return 0.0

        # Before first keyframe
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value

        # After last keyframe
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value

        # Find surrounding keyframes
        times = [k.time for k in self.keyframes]
        idx = bisect.bisect_right(times, time) - 1
        idx = max(0, min(idx, len(self.keyframes) - 2))

        k0 = self.keyframes[idx]
        k1 = self.keyframes[idx + 1]

        # Calculate interpolation factor
        dt = k1.time - k0.time
        if dt <= 0:
            return k0.value

        t = (time - k0.time) / dt

        # Interpolate based on mode
        if self.interpolation == InterpolationMode.STEP:
            return k0.value

        elif self.interpolation == InterpolationMode.LINEAR:
            return k0.value + (k1.value - k0.value) * t

        elif self.interpolation == InterpolationMode.CUBIC:
            # Cubic Hermite interpolation
            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            return (
                h00 * k0.value +
                h10 * dt * k0.out_tangent +
                h01 * k1.value +
                h11 * dt * k1.in_tangent
            )

        elif self.interpolation == InterpolationMode.HERMITE:
            # Catmull-Rom style
            # Use adjacent keyframes for tangent calculation
            if idx > 0:
                k_prev = self.keyframes[idx - 1]
                m0 = (k1.value - k_prev.value) / (k1.time - k_prev.time) * dt
            else:
                m0 = (k1.value - k0.value)

            if idx < len(self.keyframes) - 2:
                k_next = self.keyframes[idx + 2]
                m1 = (k_next.value - k0.value) / (k_next.time - k0.time) * dt
            else:
                m1 = (k1.value - k0.value)

            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            return h00 * k0.value + h10 * m0 + h01 * k1.value + h11 * m1

        return k0.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "interpolation": self.interpolation.name,
            "keyframes": [
                {
                    "time": k.time,
                    "value": k.value,
                    "in_tangent": k.in_tangent,
                    "out_tangent": k.out_tangent,
                }
                for k in self.keyframes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnimationCurve:
        """Create from dictionary."""
        curve = cls(
            name=data["name"],
            interpolation=InterpolationMode[data.get("interpolation", "LINEAR")],
        )
        for kf_data in data.get("keyframes", []):
            curve.add_keyframe(
                time=kf_data["time"],
                value=kf_data["value"],
                in_tangent=kf_data.get("in_tangent", 0.0),
                out_tangent=kf_data.get("out_tangent", 0.0),
            )
        return curve


# =============================================================================
# Face Capture Clip
# =============================================================================


@dataclass
class FaceCaptureClip:
    """
    A facial motion capture clip containing animation curves.

    Attributes:
        name: Clip name
        curves: Dictionary of blend shape curves
        frame_rate: Original capture frame rate
        duration: Clip duration in seconds
        metadata: Optional metadata (actor, session, etc.)
    """
    name: str
    curves: dict[str, AnimationCurve] = field(default_factory=dict)
    frame_rate: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get clip duration."""
        if not self.curves:
            return 0.0
        return max(curve.duration for curve in self.curves.values())

    @property
    def shape_names(self) -> list[str]:
        """Get list of animated blend shape names."""
        return list(self.curves.keys())

    @property
    def curve_count(self) -> int:
        """Get number of curves."""
        return len(self.curves)

    def add_curve(self, curve: AnimationCurve) -> None:
        """
        Add an animation curve.

        Args:
            curve: The curve to add
        """
        self.curves[curve.name] = curve

    def remove_curve(self, name: str) -> bool:
        """
        Remove a curve by name.

        Args:
            name: Curve name

        Returns:
            True if removed
        """
        if name in self.curves:
            del self.curves[name]
            return True
        return False

    def get_curve(self, name: str) -> Optional[AnimationCurve]:
        """Get a curve by name."""
        return self.curves.get(name)

    def sample(self, time: float) -> dict[str, float]:
        """
        Sample all curves at a specific time.

        Args:
            time: Time to sample

        Returns:
            Dictionary of blend shape weights
        """
        return {
            name: curve.sample(time)
            for name, curve in self.curves.items()
        }

    def sample_range(
        self,
        start_time: float,
        end_time: float,
        sample_rate: float = 30.0,
    ) -> list[Tuple[float, dict[str, float]]]:
        """
        Sample curves over a time range.

        Args:
            start_time: Start time
            end_time: End time
            sample_rate: Samples per second

        Returns:
            List of (time, weights) tuples
        """
        samples = []
        dt = 1.0 / sample_rate
        time = start_time

        while time <= end_time:
            samples.append((time, self.sample(time)))
            time += dt

        return samples

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "frame_rate": self.frame_rate,
            "metadata": self.metadata,
            "curves": {name: curve.to_dict() for name, curve in self.curves.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceCaptureClip:
        """Create from dictionary."""
        clip = cls(
            name=data["name"],
            frame_rate=data.get("frame_rate", 30.0),
            metadata=data.get("metadata", {}),
        )
        for name, curve_data in data.get("curves", {}).items():
            curve = AnimationCurve.from_dict(curve_data)
            clip.add_curve(curve)
        return clip


# =============================================================================
# Face Capture Player
# =============================================================================


class PlaybackState(Enum):
    """Playback state."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


class FaceCapturePlayer:
    """
    Player for facial motion capture clips.

    Handles playback, looping, and time control.
    """

    def __init__(
        self,
        clip: Optional[FaceCaptureClip] = None,
        on_weights_changed: Optional[Callable[[dict[str, float]], None]] = None,
        on_playback_finished: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize the player.

        Args:
            clip: Initial clip to load
            on_weights_changed: Callback when weights change
            on_playback_finished: Callback when playback finishes
        """
        self._clip = clip
        self._on_weights_changed = on_weights_changed
        self._on_playback_finished = on_playback_finished

        # Playback state
        self._state = PlaybackState.STOPPED
        self._time: float = 0.0
        self._speed: float = 1.0
        self._loop: bool = False

        # Current weights
        self._current_weights: dict[str, float] = {}

        # Blend settings for transitions
        self._blend_in_time: float = 0.0
        self._blend_out_time: float = 0.0
        self._blend_weight: float = 1.0

    @property
    def clip(self) -> Optional[FaceCaptureClip]:
        """Get current clip."""
        return self._clip

    @property
    def state(self) -> PlaybackState:
        """Get playback state."""
        return self._state

    @property
    def time(self) -> float:
        """Get current playback time."""
        return self._time

    @property
    def duration(self) -> float:
        """Get clip duration."""
        return self._clip.duration if self._clip else 0.0

    @property
    def progress(self) -> float:
        """Get playback progress (0-1)."""
        if not self._clip or self._clip.duration <= 0:
            return 0.0
        return self._time / self._clip.duration

    @property
    def is_playing(self) -> bool:
        """Check if playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def speed(self) -> float:
        """Get playback speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set playback speed."""
        self._speed = value

    @property
    def loop(self) -> bool:
        """Get loop setting."""
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        """Set loop setting."""
        self._loop = value

    def set_clip(self, clip: FaceCaptureClip) -> None:
        """
        Set the clip to play.

        Args:
            clip: The clip to load
        """
        self._clip = clip
        self._time = 0.0
        self._state = PlaybackState.STOPPED
        self._current_weights.clear()

    def play(self) -> None:
        """Start or resume playback."""
        if not self._clip:
            return
        self._state = PlaybackState.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self._state = PlaybackState.STOPPED
        self._time = 0.0
        self._current_weights.clear()
        if self._on_weights_changed:
            self._on_weights_changed({})

    def seek(self, time: float) -> None:
        """
        Seek to a specific time.

        Args:
            time: Time in seconds
        """
        if not self._clip:
            return

        self._time = max(0.0, min(time, self._clip.duration))
        self._update_weights()

    def seek_progress(self, progress: float) -> None:
        """
        Seek to a progress value.

        Args:
            progress: Progress (0-1)
        """
        if not self._clip:
            return

        self.seek(progress * self._clip.duration)

    def set_blend_times(
        self,
        blend_in: float = 0.0,
        blend_out: float = 0.0,
    ) -> None:
        """
        Set blend in/out times.

        Args:
            blend_in: Time to blend in at start
            blend_out: Time to blend out at end
        """
        self._blend_in_time = max(0.0, blend_in)
        self._blend_out_time = max(0.0, blend_out)

    def sample(self, time: float) -> dict[str, float]:
        """
        Sample the clip at a specific time.

        Args:
            time: Time to sample

        Returns:
            Blend shape weights
        """
        if not self._clip:
            return {}
        return self._clip.sample(time)

    def update(self, dt: float) -> dict[str, float]:
        """
        Update playback.

        Args:
            dt: Delta time in seconds

        Returns:
            Current blend shape weights
        """
        if self._state != PlaybackState.PLAYING or not self._clip:
            return self._current_weights

        # Advance time
        self._time += dt * self._speed

        # Handle end of clip
        if self._time >= self._clip.duration:
            if self._loop:
                self._time = self._time % self._clip.duration
            else:
                self._time = self._clip.duration
                self._state = PlaybackState.STOPPED
                if self._on_playback_finished:
                    self._on_playback_finished()

        # Handle reverse playback
        if self._time < 0:
            if self._loop:
                self._time = self._clip.duration + self._time
            else:
                self._time = 0
                self._state = PlaybackState.STOPPED
                if self._on_playback_finished:
                    self._on_playback_finished()

        self._update_weights()
        return self._current_weights

    def _update_weights(self) -> None:
        """Update current weights from clip."""
        if not self._clip:
            return

        # Sample clip
        weights = self._clip.sample(self._time)

        # Apply blend weight
        self._blend_weight = self._calculate_blend_weight()
        if self._blend_weight < 1.0:
            weights = {k: v * self._blend_weight for k, v in weights.items()}

        self._current_weights = weights

        if self._on_weights_changed:
            self._on_weights_changed(weights)

    def _calculate_blend_weight(self) -> float:
        """Calculate blend weight for current time."""
        if not self._clip:
            return 1.0

        duration = self._clip.duration
        weight = 1.0

        # Blend in
        if self._blend_in_time > 0 and self._time < self._blend_in_time:
            weight = min(weight, self._time / self._blend_in_time)

        # Blend out
        if self._blend_out_time > 0:
            blend_out_start = duration - self._blend_out_time
            if self._time > blend_out_start:
                weight = min(weight, (duration - self._time) / self._blend_out_time)

        return max(0.0, min(1.0, weight))

    def get_current_weights(self) -> dict[str, float]:
        """Get current blend shape weights."""
        return self._current_weights.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "state": self._state.name,
            "time": self._time,
            "speed": self._speed,
            "loop": self._loop,
            "clip_name": self._clip.name if self._clip else None,
        }


# =============================================================================
# Retargeting
# =============================================================================


@dataclass
class RetargetMapping:
    """
    Mapping for retargeting between blend shape sets.

    Attributes:
        source_name: Source blend shape name
        target_name: Target blend shape name
        scale: Value scale factor
        offset: Value offset
    """
    source_name: str
    target_name: str
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, value: float) -> float:
        """
        Apply the mapping to a value.

        Args:
            value: Source value

        Returns:
            Mapped value
        """
        return value * self.scale + self.offset


class FaceCaptureRetargeter:
    """
    Retargets face capture data between different blend shape sets.

    Handles name mapping, scaling, and value transformations.

    Supports:
    - Single source -> single target mappings
    - Single source -> multiple targets (one-to-many)
    - Multiple sources -> single target (many-to-one, accumulates)
    """

    def __init__(
        self,
        mappings: Optional[Sequence[RetargetMapping]] = None,
    ) -> None:
        """
        Initialize the retargeter.

        Args:
            mappings: Initial retarget mappings
        """
        # Store mappings as list per source for one-to-many support
        self._mappings: dict[str, list[RetargetMapping]] = {}
        if mappings:
            for mapping in mappings:
                self.add_mapping(
                    mapping.source_name,
                    mapping.target_name,
                    mapping.scale,
                    mapping.offset,
                )

        # Unmapped shape behavior
        self._pass_through_unmapped: bool = True
        self._unmapped_scale: float = 1.0

    @property
    def mapping_count(self) -> int:
        """Get total number of mappings."""
        return sum(len(mappings) for mappings in self._mappings.values())

    def add_mapping(
        self,
        source_name: str,
        target_name: str,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        """
        Add a retarget mapping.

        Args:
            source_name: Source blend shape name
            target_name: Target blend shape name
            scale: Value scale factor (default 1.0)
            offset: Value offset (default 0.0)
        """
        mapping = RetargetMapping(
            source_name=source_name,
            target_name=target_name,
            scale=scale,
            offset=offset,
        )
        if source_name not in self._mappings:
            self._mappings[source_name] = []
        self._mappings[source_name].append(mapping)

    def remove_mapping(
        self,
        source_name: str,
        target_name: Optional[str] = None,
    ) -> bool:
        """
        Remove a mapping or all mappings for a source.

        Args:
            source_name: Source shape name
            target_name: Optional target name. If None, removes all mappings
                for the source.

        Returns:
            True if any mapping was removed
        """
        if source_name not in self._mappings:
            return False

        if target_name is None:
            # Remove all mappings for this source
            del self._mappings[source_name]
            return True
        else:
            # Remove specific source->target mapping
            mappings = self._mappings[source_name]
            original_len = len(mappings)
            self._mappings[source_name] = [
                m for m in mappings if m.target_name != target_name
            ]
            # Clean up empty list
            if not self._mappings[source_name]:
                del self._mappings[source_name]
            return len(self._mappings.get(source_name, [])) < original_len

    def clear_mappings(self) -> None:
        """Remove all mappings."""
        self._mappings.clear()

    def get_mapping(self, source_name: str) -> Optional[RetargetMapping]:
        """Get first mapping for a source shape (legacy compatibility)."""
        mappings = self._mappings.get(source_name)
        return mappings[0] if mappings else None

    def get_mappings(self, source_name: str) -> list[RetargetMapping]:
        """Get all mappings for a source shape."""
        return self._mappings.get(source_name, []).copy()

    def set_pass_through(self, enabled: bool, scale: float = 1.0) -> None:
        """
        Set behavior for unmapped shapes.

        Args:
            enabled: Whether to pass through unmapped shapes
            scale: Scale factor for unmapped shapes
        """
        self._pass_through_unmapped = enabled
        self._unmapped_scale = scale

    def retarget(
        self,
        source_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Retarget blend shape weights.

        Applies all configured mappings:
        - Single source -> single target: target = source * scale + offset
        - Single source -> multiple targets: each gets source * scale + offset
        - Multiple sources -> single target: accumulates (sums contributions)
        - Missing source shapes are skipped silently
        - Results are clamped to [0, 1]

        Args:
            source_weights: Source blend shape weights

        Returns:
            Retargeted weights dictionary
        """
        result: dict[str, float] = {}

        for source_name, value in source_weights.items():
            mappings = self._mappings.get(source_name)

            if mappings:
                # Apply all mappings for this source (one-to-many)
                for mapping in mappings:
                    target_name = mapping.target_name
                    target_value = mapping.apply(value)

                    # Accumulate (for many-to-one mappings)
                    if target_name in result:
                        result[target_name] += target_value
                    else:
                        result[target_name] = target_value

            elif self._pass_through_unmapped:
                # Pass through with optional scaling
                result[source_name] = value * self._unmapped_scale

        # Clamp all values to [0, 1]
        for name in result:
            result[name] = max(0.0, min(1.0, result[name]))

        return result

    def retarget_weights(
        self,
        source_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Retarget blend shape weights (legacy alias for retarget).

        Args:
            source_weights: Source weights

        Returns:
            Retargeted weights
        """
        return self.retarget(source_weights)

    def retarget_clip(
        self,
        source_clip: FaceCaptureClip,
        target_name: Optional[str] = None,
    ) -> FaceCaptureClip:
        """
        Retarget an entire clip.

        Args:
            source_clip: Source clip
            target_name: Name for the new clip (default: source name + "_retargeted")

        Returns:
            Retargeted clip
        """
        if target_name is None:
            target_name = f"{source_clip.name}_retargeted"

        target_clip = FaceCaptureClip(
            name=target_name,
            frame_rate=source_clip.frame_rate,
            metadata=source_clip.metadata.copy(),
        )

        # Process each curve
        for source_name, source_curve in source_clip.curves.items():
            mappings = self._mappings.get(source_name)

            if mappings:
                # Apply all mappings (one-to-many support)
                for mapping in mappings:
                    target_curve_name = mapping.target_name
                    scale = mapping.scale
                    offset = mapping.offset

                    self._apply_curve_mapping(
                        source_curve,
                        target_clip,
                        target_curve_name,
                        scale,
                        offset,
                    )
            elif self._pass_through_unmapped:
                # Pass through unmapped
                self._apply_curve_mapping(
                    source_curve,
                    target_clip,
                    source_name,
                    self._unmapped_scale,
                    0.0,
                )

        return target_clip

    def _apply_curve_mapping(
        self,
        source_curve: AnimationCurve,
        target_clip: FaceCaptureClip,
        target_curve_name: str,
        scale: float,
        offset: float,
    ) -> None:
        """
        Apply a mapping from source curve to target clip.

        Args:
            source_curve: Source animation curve
            target_clip: Target clip to add curve to
            target_curve_name: Name for target curve
            scale: Scale factor
            offset: Offset value
        """
        # Create or get target curve
        if target_curve_name not in target_clip.curves:
            target_curve = AnimationCurve(
                name=target_curve_name,
                interpolation=source_curve.interpolation,
            )
            target_clip.add_curve(target_curve)
        else:
            target_curve = target_clip.curves[target_curve_name]

        # Copy and transform keyframes
        for kf in source_curve.keyframes:
            target_curve.add_keyframe(
                time=kf.time,
                value=max(0.0, min(1.0, kf.value * scale + offset)),
                in_tangent=kf.in_tangent * scale,
                out_tangent=kf.out_tangent * scale,
            )

    def create_identity_mappings(self, shape_names: Sequence[str]) -> None:
        """
        Create identity mappings for a list of shape names.

        Args:
            shape_names: List of shape names
        """
        for name in shape_names:
            self.add_mapping(name, name, scale=1.0, offset=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        all_mappings = []
        for mappings in self._mappings.values():
            for m in mappings:
                all_mappings.append({
                    "source_name": m.source_name,
                    "target_name": m.target_name,
                    "scale": m.scale,
                    "offset": m.offset,
                })
        return {
            "mappings": all_mappings,
            "pass_through_unmapped": self._pass_through_unmapped,
            "unmapped_scale": self._unmapped_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceCaptureRetargeter:
        """Create from dictionary."""
        mappings = [
            RetargetMapping(
                source_name=m["source_name"],
                target_name=m["target_name"],
                scale=m.get("scale", 1.0),
                offset=m.get("offset", 0.0),
            )
            for m in data.get("mappings", [])
        ]
        retargeter = cls(mappings)
        retargeter._pass_through_unmapped = data.get("pass_through_unmapped", True)
        retargeter._unmapped_scale = data.get("unmapped_scale", 1.0)
        return retargeter


# =============================================================================
# Utility Functions
# =============================================================================


def create_clip_from_samples(
    name: str,
    samples: Sequence[Tuple[float, dict[str, float]]],
    frame_rate: float = 30.0,
    interpolation: InterpolationMode = InterpolationMode.LINEAR,
) -> FaceCaptureClip:
    """
    Create a face capture clip from time-stamped samples.

    Args:
        name: Clip name
        samples: List of (time, weights) tuples
        frame_rate: Frame rate
        interpolation: Interpolation mode

    Returns:
        Face capture clip
    """
    clip = FaceCaptureClip(name=name, frame_rate=frame_rate)

    # Collect all shape names
    shape_names = set()
    for _, weights in samples:
        shape_names.update(weights.keys())

    # Create curves for each shape
    for shape_name in shape_names:
        curve = AnimationCurve(name=shape_name, interpolation=interpolation)

        for time, weights in samples:
            value = weights.get(shape_name, 0.0)
            curve.add_keyframe(time, value)

        clip.add_curve(curve)

    return clip


def merge_clips(
    clips: Sequence[FaceCaptureClip],
    name: str = "merged",
    gap_time: float = 0.0,
) -> FaceCaptureClip:
    """
    Merge multiple clips sequentially.

    Args:
        clips: Clips to merge
        name: Name for merged clip
        gap_time: Gap between clips in seconds

    Returns:
        Merged clip
    """
    merged = FaceCaptureClip(name=name)
    current_time = 0.0

    for clip in clips:
        for curve_name, curve in clip.curves.items():
            if curve_name not in merged.curves:
                merged.curves[curve_name] = AnimationCurve(
                    name=curve_name,
                    interpolation=curve.interpolation,
                )

            target_curve = merged.curves[curve_name]

            for kf in curve.keyframes:
                target_curve.add_keyframe(
                    time=kf.time + current_time,
                    value=kf.value,
                    in_tangent=kf.in_tangent,
                    out_tangent=kf.out_tangent,
                )

        current_time += clip.duration + gap_time

    return merged
