"""Animation clip data.

Animation clips contain keyframe data for bones, organized into tracks
with different interpolation modes. Clips support looping, root motion,
and event triggers.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from engine.core.math import Quat, Vec3
from engine.core.math.interpolation import lerp

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton
    from engine.animation.skeletal.pose import Pose, BoneTransform


# =============================================================================
# Configuration Constants
# =============================================================================

# Time tolerance for keyframe matching (seconds)
KEYFRAME_TIME_TOLERANCE = 1e-9

# Default tolerance for removing keyframes by time
KEYFRAME_REMOVE_TOLERANCE = 1e-6

# Minimum time delta for interpolation (prevents division by near-zero)
INTERPOLATION_TIME_EPSILON = 1e-9

# Tolerance for scale division to prevent division by near-zero
SCALE_EPSILON = 1e-9

# Default framerate for animation clips (frames per second)
DEFAULT_FRAMERATE = 30.0


class InterpolationType(Enum):
    """Interpolation mode for animation curves."""

    STEP = auto()  # No interpolation, use previous keyframe value
    LINEAR = auto()  # Linear interpolation between keyframes
    CUBIC = auto()  # Cubic/Hermite interpolation with tangents


def animation_data(cls):
    """Decorator for animation data classes."""
    cls._animation_data = True
    cls._animation_type = cls.__name__
    return cls


@animation_data
@dataclass
class Keyframe:
    """A single keyframe in an animation curve.

    Attributes:
        time: Time of keyframe in seconds.
        value: Value at this keyframe (float, Vec3, or Quat).
        in_tangent: Incoming tangent for cubic interpolation.
        out_tangent: Outgoing tangent for cubic interpolation.
    """

    time: float
    value: Any
    in_tangent: Optional[Any] = None
    out_tangent: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Keyframe time must be >= 0, got {self.time}")

    def copy(self) -> Keyframe:
        """Create a copy of this keyframe."""
        return Keyframe(
            time=self.time,
            value=self._copy_value(self.value),
            in_tangent=self._copy_value(self.in_tangent),
            out_tangent=self._copy_value(self.out_tangent),
        )

    def _copy_value(self, val: Any) -> Any:
        """Copy a value, handling different types."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, Vec3):
            return Vec3(val.x, val.y, val.z)
        if isinstance(val, Quat):
            return Quat(val.x, val.y, val.z, val.w)
        return val

    def __repr__(self) -> str:
        return f"Keyframe(t={self.time:.3f}, v={self.value})"


@animation_data
class AnimationCurve:
    """A curve of keyframes for a single property.

    Curves can be sampled at any time, with interpolation between keyframes.

    Attributes:
        keyframes: Sorted list of keyframes.
        interpolation: How to interpolate between keyframes.
    """

    def __init__(
        self,
        keyframes: Optional[List[Keyframe]] = None,
        interpolation: InterpolationType = InterpolationType.LINEAR,
    ) -> None:
        """Initialize animation curve.

        Args:
            keyframes: List of keyframes (will be sorted by time).
            interpolation: Interpolation mode.
        """
        self._keyframes: List[Keyframe] = []
        self._interpolation = interpolation

        if keyframes:
            for kf in keyframes:
                self.add_keyframe(kf)

    @property
    def keyframes(self) -> List[Keyframe]:
        """Get keyframes (read-only copy)."""
        return [kf.copy() for kf in self._keyframes]

    @property
    def interpolation(self) -> InterpolationType:
        """Get interpolation type."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: InterpolationType) -> None:
        """Set interpolation type."""
        self._interpolation = value

    @property
    def keyframe_count(self) -> int:
        """Get number of keyframes."""
        return len(self._keyframes)

    @property
    def duration(self) -> float:
        """Get curve duration (time of last keyframe)."""
        if not self._keyframes:
            return 0.0
        return self._keyframes[-1].time

    @property
    def start_time(self) -> float:
        """Get time of first keyframe."""
        if not self._keyframes:
            return 0.0
        return self._keyframes[0].time

    def add_keyframe(self, keyframe: Keyframe) -> None:
        """Add a keyframe to the curve.

        Keyframes are kept sorted by time. If a keyframe at the same
        time exists, it is replaced.

        Args:
            keyframe: Keyframe to add.
        """
        # Find insertion point
        times = [kf.time for kf in self._keyframes]
        idx = bisect.bisect_left(times, keyframe.time)

        # Check if replacing existing keyframe
        if idx < len(self._keyframes) and abs(self._keyframes[idx].time - keyframe.time) < KEYFRAME_TIME_TOLERANCE:
            self._keyframes[idx] = keyframe.copy()
        else:
            self._keyframes.insert(idx, keyframe.copy())

    def remove_keyframe(self, index: int) -> None:
        """Remove keyframe at index.

        Args:
            index: Index of keyframe to remove.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self._keyframes):
            raise IndexError(f"Keyframe index {index} out of range")
        self._keyframes.pop(index)

    def remove_keyframe_at_time(self, time: float, tolerance: float = KEYFRAME_REMOVE_TOLERANCE) -> bool:
        """Remove keyframe at specified time.

        Args:
            time: Time to search for.
            tolerance: Time tolerance for matching.

        Returns:
            True if keyframe was removed, False if not found.
        """
        for i, kf in enumerate(self._keyframes):
            if abs(kf.time - time) <= tolerance:
                self._keyframes.pop(i)
                return True
        return False

    def get_keyframe(self, index: int) -> Keyframe:
        """Get keyframe at index.

        Args:
            index: Keyframe index.

        Returns:
            Copy of keyframe.

        Raises:
            IndexError: If index out of range.
        """
        if index < 0 or index >= len(self._keyframes):
            raise IndexError(f"Keyframe index {index} out of range")
        return self._keyframes[index].copy()

    def find_keyframe_indices(self, time: float) -> Tuple[int, int]:
        """Find indices of keyframes surrounding a time.

        Args:
            time: Time to search for.

        Returns:
            Tuple of (prev_index, next_index). Both may be same for exact matches
            or at boundaries.
        """
        if not self._keyframes:
            return (-1, -1)

        # Binary search for position
        times = [kf.time for kf in self._keyframes]
        idx = bisect.bisect_right(times, time)

        if idx == 0:
            return (0, 0)
        if idx >= len(self._keyframes):
            return (len(self._keyframes) - 1, len(self._keyframes) - 1)

        return (idx - 1, idx)

    def sample(self, time: float) -> Any:
        """Sample the curve at a specific time.

        Args:
            time: Time to sample at.

        Returns:
            Interpolated value at time.

        Raises:
            ValueError: If curve has no keyframes.
        """
        if not self._keyframes:
            raise ValueError("Cannot sample empty curve")

        # Handle boundary cases
        if time <= self._keyframes[0].time:
            return self._copy_value(self._keyframes[0].value)
        if time >= self._keyframes[-1].time:
            return self._copy_value(self._keyframes[-1].value)

        # Find surrounding keyframes
        prev_idx, next_idx = self.find_keyframe_indices(time)
        prev_kf = self._keyframes[prev_idx]
        next_kf = self._keyframes[next_idx]

        if prev_idx == next_idx:
            return self._copy_value(prev_kf.value)

        # Calculate interpolation factor
        dt = next_kf.time - prev_kf.time
        if dt < INTERPOLATION_TIME_EPSILON:
            return self._copy_value(prev_kf.value)

        t = (time - prev_kf.time) / dt

        # Interpolate based on mode
        if self._interpolation == InterpolationType.STEP:
            return self._copy_value(prev_kf.value)
        elif self._interpolation == InterpolationType.LINEAR:
            return self._interpolate_linear(prev_kf.value, next_kf.value, t)
        elif self._interpolation == InterpolationType.CUBIC:
            return self._interpolate_cubic(prev_kf, next_kf, t)
        else:
            return self._interpolate_linear(prev_kf.value, next_kf.value, t)

    def _interpolate_linear(self, a: Any, b: Any, t: float) -> Any:
        """Linear interpolation between two values."""
        if isinstance(a, (int, float)):
            return lerp(float(a), float(b), t)
        if isinstance(a, Vec3):
            return a.lerp(b, t)
        if isinstance(a, Quat):
            return a.slerp(b, t)
        # Fallback for unknown types
        return a if t < 0.5 else b

    def _interpolate_cubic(
        self, prev_kf: Keyframe, next_kf: Keyframe, t: float
    ) -> Any:
        """Cubic/Hermite interpolation between keyframes."""
        a = prev_kf.value
        b = next_kf.value

        # Hermite basis functions
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        dt = next_kf.time - prev_kf.time

        if isinstance(a, (int, float)):
            # Get tangents (default to 0 if not specified)
            out_tan = prev_kf.out_tangent if prev_kf.out_tangent is not None else 0.0
            in_tan = next_kf.in_tangent if next_kf.in_tangent is not None else 0.0
            return (
                h00 * float(a)
                + h10 * dt * float(out_tan)
                + h01 * float(b)
                + h11 * dt * float(in_tan)
            )

        if isinstance(a, Vec3):
            # For Vec3, use tangent vectors or compute from values
            out_tan = prev_kf.out_tangent if prev_kf.out_tangent is not None else Vec3.zero()
            in_tan = next_kf.in_tangent if next_kf.in_tangent is not None else Vec3.zero()
            return Vec3(
                h00 * a.x + h10 * dt * out_tan.x + h01 * b.x + h11 * dt * in_tan.x,
                h00 * a.y + h10 * dt * out_tan.y + h01 * b.y + h11 * dt * in_tan.y,
                h00 * a.z + h10 * dt * out_tan.z + h01 * b.z + h11 * dt * in_tan.z,
            )

        if isinstance(a, Quat):
            # For quaternions, fall back to slerp (cubic quat interpolation is complex)
            return a.slerp(b, t)

        # Fallback
        return self._interpolate_linear(a, b, t)

    def _copy_value(self, val: Any) -> Any:
        """Copy a value."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, Vec3):
            return Vec3(val.x, val.y, val.z)
        if isinstance(val, Quat):
            return Quat(val.x, val.y, val.z, val.w)
        return val

    def copy(self) -> AnimationCurve:
        """Create a deep copy of this curve."""
        return AnimationCurve(
            keyframes=[kf.copy() for kf in self._keyframes],
            interpolation=self._interpolation,
        )

    def __repr__(self) -> str:
        return (
            f"AnimationCurve(keyframes={len(self._keyframes)}, "
            f"interp={self._interpolation.name}, duration={self.duration:.3f})"
        )


@animation_data
@dataclass
class AnimationEvent:
    """An event triggered during animation playback.

    Attributes:
        time: Time when event fires (seconds).
        name: Event name/identifier.
        data: Optional event payload data.
    """

    time: float
    name: str
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Event time must be >= 0, got {self.time}")
        if not self.name:
            raise ValueError("Event name cannot be empty")

    def copy(self) -> AnimationEvent:
        """Create a copy of this event."""
        return AnimationEvent(
            time=self.time,
            name=self.name,
            data=dict(self.data) if self.data else None,
        )

    def __repr__(self) -> str:
        return f"AnimationEvent(t={self.time:.3f}, name='{self.name}')"


@animation_data
class BoneTrack:
    """Animation track for a single bone.

    Contains separate curves for position, rotation, and scale.

    Attributes:
        bone_index: Index of the bone this track animates.
        position_curve: Curve for bone position (Vec3).
        rotation_curve: Curve for bone rotation (Quat).
        scale_curve: Curve for bone scale (Vec3).
    """

    def __init__(
        self,
        bone_index: int,
        position_curve: Optional[AnimationCurve] = None,
        rotation_curve: Optional[AnimationCurve] = None,
        scale_curve: Optional[AnimationCurve] = None,
    ) -> None:
        """Initialize bone track.

        Args:
            bone_index: Index of animated bone.
            position_curve: Optional position curve.
            rotation_curve: Optional rotation curve.
            scale_curve: Optional scale curve.
        """
        if bone_index < 0:
            raise ValueError(f"Bone index must be >= 0, got {bone_index}")

        self._bone_index = bone_index
        self._position_curve = position_curve
        self._rotation_curve = rotation_curve
        self._scale_curve = scale_curve

    @property
    def bone_index(self) -> int:
        """Get bone index."""
        return self._bone_index

    @property
    def position_curve(self) -> Optional[AnimationCurve]:
        """Get position curve."""
        return self._position_curve

    @position_curve.setter
    def position_curve(self, value: Optional[AnimationCurve]) -> None:
        """Set position curve."""
        self._position_curve = value

    @property
    def rotation_curve(self) -> Optional[AnimationCurve]:
        """Get rotation curve."""
        return self._rotation_curve

    @rotation_curve.setter
    def rotation_curve(self, value: Optional[AnimationCurve]) -> None:
        """Set rotation curve."""
        self._rotation_curve = value

    @property
    def scale_curve(self) -> Optional[AnimationCurve]:
        """Get scale curve."""
        return self._scale_curve

    @scale_curve.setter
    def scale_curve(self, value: Optional[AnimationCurve]) -> None:
        """Set scale curve."""
        self._scale_curve = value

    @property
    def duration(self) -> float:
        """Get duration of this track."""
        durations = []
        if self._position_curve:
            durations.append(self._position_curve.duration)
        if self._rotation_curve:
            durations.append(self._rotation_curve.duration)
        if self._scale_curve:
            durations.append(self._scale_curve.duration)
        return max(durations) if durations else 0.0

    def has_position(self) -> bool:
        """Check if track has position data."""
        return self._position_curve is not None and self._position_curve.keyframe_count > 0

    def has_rotation(self) -> bool:
        """Check if track has rotation data."""
        return self._rotation_curve is not None and self._rotation_curve.keyframe_count > 0

    def has_scale(self) -> bool:
        """Check if track has scale data."""
        return self._scale_curve is not None and self._scale_curve.keyframe_count > 0

    def sample_position(self, time: float, default: Optional[Vec3] = None) -> Vec3:
        """Sample position at time.

        Args:
            time: Time to sample.
            default: Default value if no curve.

        Returns:
            Position at time.
        """
        if self._position_curve and self._position_curve.keyframe_count > 0:
            return self._position_curve.sample(time)
        return default if default is not None else Vec3.zero()

    def sample_rotation(self, time: float, default: Optional[Quat] = None) -> Quat:
        """Sample rotation at time.

        Args:
            time: Time to sample.
            default: Default value if no curve.

        Returns:
            Rotation at time.
        """
        if self._rotation_curve and self._rotation_curve.keyframe_count > 0:
            return self._rotation_curve.sample(time)
        return default if default is not None else Quat.identity()

    def sample_scale(self, time: float, default: Optional[Vec3] = None) -> Vec3:
        """Sample scale at time.

        Args:
            time: Time to sample.
            default: Default value if no curve.

        Returns:
            Scale at time.
        """
        if self._scale_curve and self._scale_curve.keyframe_count > 0:
            return self._scale_curve.sample(time)
        return default if default is not None else Vec3.one()

    def copy(self) -> BoneTrack:
        """Create a deep copy of this track."""
        return BoneTrack(
            bone_index=self._bone_index,
            position_curve=self._position_curve.copy() if self._position_curve else None,
            rotation_curve=self._rotation_curve.copy() if self._rotation_curve else None,
            scale_curve=self._scale_curve.copy() if self._scale_curve else None,
        )

    def __repr__(self) -> str:
        parts = []
        if self.has_position():
            parts.append("pos")
        if self.has_rotation():
            parts.append("rot")
        if self.has_scale():
            parts.append("scale")
        return f"BoneTrack(bone={self._bone_index}, curves=[{', '.join(parts)}])"


@animation_data
class AnimationClip:
    """A complete animation clip.

    Contains tracks for multiple bones, events, and metadata.

    Attributes:
        name: Clip name.
        duration: Total duration in seconds.
        framerate: Original framerate.
        bone_tracks: Dictionary of bone tracks by bone index.
        events: List of animation events.
        looping: Whether clip should loop.
        root_motion: Whether clip has root motion to extract.
    """

    def __init__(
        self,
        name: str,
        duration: float = 0.0,
        framerate: float = DEFAULT_FRAMERATE,
        bone_tracks: Optional[Dict[int, BoneTrack]] = None,
        events: Optional[List[AnimationEvent]] = None,
        looping: bool = False,
        root_motion: bool = False,
    ) -> None:
        """Initialize animation clip.

        Args:
            name: Clip name.
            duration: Duration in seconds (0 = auto from tracks).
            framerate: Original framerate.
            bone_tracks: Dictionary of bone index -> BoneTrack.
            events: List of animation events.
            looping: Whether clip loops.
            root_motion: Whether to extract root motion.
        """
        if not name:
            raise ValueError("Clip name cannot be empty")
        if framerate <= 0:
            raise ValueError(f"Framerate must be > 0, got {framerate}")

        self._name = name
        self._duration = max(0.0, duration)
        self._framerate = framerate
        self._bone_tracks: Dict[int, BoneTrack] = {}
        self._events: List[AnimationEvent] = []
        self._looping = looping
        self._root_motion = root_motion
        self._root_bone_index = 0  # Index of root bone for motion extraction

        if bone_tracks:
            for bone_idx, track in bone_tracks.items():
                self.add_bone_track(track)

        if events:
            for event in events:
                self.add_event(event)

        # Auto-calculate duration if not specified
        if self._duration == 0.0:
            self._update_duration()

    @property
    def name(self) -> str:
        """Get clip name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set clip name."""
        if not value:
            raise ValueError("Clip name cannot be empty")
        self._name = value

    @property
    def duration(self) -> float:
        """Get clip duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set clip duration."""
        self._duration = max(0.0, value)

    @property
    def framerate(self) -> float:
        """Get original framerate."""
        return self._framerate

    @property
    def bone_tracks(self) -> Dict[int, BoneTrack]:
        """Get bone tracks (read-only view)."""
        return dict(self._bone_tracks)

    @property
    def events(self) -> List[AnimationEvent]:
        """Get events (read-only copy)."""
        return [e.copy() for e in self._events]

    @property
    def looping(self) -> bool:
        """Get looping flag."""
        return self._looping

    @looping.setter
    def looping(self, value: bool) -> None:
        """Set looping flag."""
        self._looping = value

    @property
    def root_motion(self) -> bool:
        """Get root motion flag."""
        return self._root_motion

    @root_motion.setter
    def root_motion(self, value: bool) -> None:
        """Set root motion flag."""
        self._root_motion = value

    @property
    def root_bone_index(self) -> int:
        """Get root bone index for motion extraction."""
        return self._root_bone_index

    @root_bone_index.setter
    def root_bone_index(self, value: int) -> None:
        """Set root bone index."""
        self._root_bone_index = value

    @property
    def frame_count(self) -> int:
        """Get total frame count based on duration and framerate."""
        return int(self._duration * self._framerate) + 1

    @property
    def track_count(self) -> int:
        """Get number of bone tracks."""
        return len(self._bone_tracks)

    @property
    def event_count(self) -> int:
        """Get number of events."""
        return len(self._events)

    def add_bone_track(self, track: BoneTrack) -> None:
        """Add or replace a bone track.

        Args:
            track: The bone track to add.
        """
        self._bone_tracks[track.bone_index] = track.copy()
        self._update_duration()

    def remove_bone_track(self, bone_index: int) -> bool:
        """Remove a bone track.

        Args:
            bone_index: Index of bone track to remove.

        Returns:
            True if removed, False if not found.
        """
        if bone_index in self._bone_tracks:
            del self._bone_tracks[bone_index]
            self._update_duration()
            return True
        return False

    def get_bone_track(self, bone_index: int) -> Optional[BoneTrack]:
        """Get a bone track.

        Args:
            bone_index: Index of bone.

        Returns:
            Copy of track, or None if not found.
        """
        track = self._bone_tracks.get(bone_index)
        return track.copy() if track else None

    def has_bone_track(self, bone_index: int) -> bool:
        """Check if clip has track for bone.

        Args:
            bone_index: Bone index.

        Returns:
            True if track exists.
        """
        return bone_index in self._bone_tracks

    def add_event(self, event: AnimationEvent) -> None:
        """Add an animation event.

        Events are kept sorted by time.

        Args:
            event: Event to add.
        """
        event_copy = event.copy()
        times = [e.time for e in self._events]
        idx = bisect.bisect_left(times, event_copy.time)
        self._events.insert(idx, event_copy)

    def remove_event(self, index: int) -> None:
        """Remove event at index.

        Args:
            index: Event index.

        Raises:
            IndexError: If index out of range.
        """
        if index < 0 or index >= len(self._events):
            raise IndexError(f"Event index {index} out of range")
        self._events.pop(index)

    def get_events_in_range(
        self, start_time: float, end_time: float
    ) -> List[AnimationEvent]:
        """Get events that fire within a time range.

        Args:
            start_time: Start of range (exclusive).
            end_time: End of range (inclusive).

        Returns:
            List of events in range.
        """
        return [
            e.copy()
            for e in self._events
            if start_time < e.time <= end_time
        ]

    def _update_duration(self) -> None:
        """Update duration based on track data."""
        if self._duration > 0:
            return  # Manual duration takes precedence

        max_duration = 0.0
        for track in self._bone_tracks.values():
            max_duration = max(max_duration, track.duration)

        if self._events:
            max_duration = max(max_duration, self._events[-1].time)

        self._duration = max_duration

    def sample_bone(
        self,
        bone_index: int,
        time: float,
        default_position: Optional[Vec3] = None,
        default_rotation: Optional[Quat] = None,
        default_scale: Optional[Vec3] = None,
    ) -> Tuple[Vec3, Quat, Vec3]:
        """Sample bone transform at time.

        Args:
            bone_index: Bone to sample.
            time: Time to sample at.
            default_position: Default position if no track.
            default_rotation: Default rotation if no track.
            default_scale: Default scale if no track.

        Returns:
            Tuple of (position, rotation, scale).
        """
        track = self._bone_tracks.get(bone_index)

        if track is None:
            return (
                default_position or Vec3.zero(),
                default_rotation or Quat.identity(),
                default_scale or Vec3.one(),
            )

        return (
            track.sample_position(time, default_position),
            track.sample_rotation(time, default_rotation),
            track.sample_scale(time, default_scale),
        )

    def sample_pose(
        self,
        skeleton: Skeleton,
        time: float,
    ) -> Pose:
        """Sample full pose at time.

        Args:
            skeleton: Skeleton to sample for.
            time: Time to sample at.

        Returns:
            Pose at the given time.
        """
        from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace

        transforms = []

        for bone in skeleton:
            # Get default from bind pose
            bind = bone.local_bind_pose
            pos, rot, scale = self.sample_bone(
                bone.index,
                time,
                default_position=bind.translation,
                default_rotation=bind.rotation,
                default_scale=bind.scale,
            )

            transforms.append(
                BoneTransform(
                    translation=pos,
                    rotation=rot,
                    scale=scale,
                )
            )

        return Pose(skeleton, PoseSpace.LOCAL, transforms)

    def extract_root_motion(
        self, start_time: float, end_time: float
    ) -> Tuple[Vec3, Quat]:
        """Extract root motion between two times.

        Args:
            start_time: Start time.
            end_time: End time.

        Returns:
            Tuple of (translation_delta, rotation_delta).
        """
        if not self._root_motion:
            return (Vec3.zero(), Quat.identity())

        track = self._bone_tracks.get(self._root_bone_index)
        if track is None:
            return (Vec3.zero(), Quat.identity())

        start_pos = track.sample_position(start_time, Vec3.zero())
        end_pos = track.sample_position(end_time, Vec3.zero())

        start_rot = track.sample_rotation(start_time, Quat.identity())
        end_rot = track.sample_rotation(end_time, Quat.identity())

        # Calculate deltas
        translation_delta = Vec3(
            end_pos.x - start_pos.x,
            end_pos.y - start_pos.y,
            end_pos.z - start_pos.z,
        )

        rotation_delta = start_rot.inverse() * end_rot

        return (translation_delta, rotation_delta)

    def create_additive_clip(
        self,
        reference_clip: AnimationClip,
    ) -> AnimationClip:
        """Create an additive version of this clip.

        The result represents the delta from reference_clip to this clip.

        Args:
            reference_clip: The reference clip to subtract.

        Returns:
            New additive clip.
        """
        additive = AnimationClip(
            name=f"{self._name}_additive",
            duration=self._duration,
            framerate=self._framerate,
            looping=self._looping,
        )

        # Sample at regular intervals
        sample_count = max(2, int(self._duration * self._framerate))
        dt = self._duration / (sample_count - 1) if sample_count > 1 else 0.0

        for bone_index in self._bone_tracks:
            pos_keyframes = []
            rot_keyframes = []
            scale_keyframes = []

            for i in range(sample_count):
                time = i * dt

                # Get this clip's values
                this_pos, this_rot, this_scale = self.sample_bone(bone_index, time)

                # Get reference values
                ref_pos, ref_rot, ref_scale = reference_clip.sample_bone(bone_index, time)

                # Compute deltas
                delta_pos = Vec3(
                    this_pos.x - ref_pos.x,
                    this_pos.y - ref_pos.y,
                    this_pos.z - ref_pos.z,
                )
                delta_rot = ref_rot.inverse() * this_rot
                delta_scale = Vec3(
                    this_scale.x / ref_scale.x if abs(ref_scale.x) > SCALE_EPSILON else 1.0,
                    this_scale.y / ref_scale.y if abs(ref_scale.y) > SCALE_EPSILON else 1.0,
                    this_scale.z / ref_scale.z if abs(ref_scale.z) > SCALE_EPSILON else 1.0,
                )

                pos_keyframes.append(Keyframe(time, delta_pos))
                rot_keyframes.append(Keyframe(time, delta_rot))
                scale_keyframes.append(Keyframe(time, delta_scale))

            track = BoneTrack(
                bone_index=bone_index,
                position_curve=AnimationCurve(pos_keyframes, InterpolationType.LINEAR),
                rotation_curve=AnimationCurve(rot_keyframes, InterpolationType.LINEAR),
                scale_curve=AnimationCurve(scale_keyframes, InterpolationType.LINEAR),
            )
            additive.add_bone_track(track)

        return additive

    def copy(self) -> AnimationClip:
        """Create a deep copy of this clip."""
        return AnimationClip(
            name=self._name,
            duration=self._duration,
            framerate=self._framerate,
            bone_tracks={idx: track.copy() for idx, track in self._bone_tracks.items()},
            events=[e.copy() for e in self._events],
            looping=self._looping,
            root_motion=self._root_motion,
        )

    def validate(self, skeleton: Optional[Skeleton] = None) -> List[str]:
        """Validate clip integrity.

        Args:
            skeleton: Optional skeleton to validate against.

        Returns:
            List of validation errors, empty if valid.
        """
        errors = []

        if self._duration < 0:
            errors.append(f"Invalid duration: {self._duration}")

        if self._framerate <= 0:
            errors.append(f"Invalid framerate: {self._framerate}")

        for event in self._events:
            if event.time > self._duration:
                errors.append(
                    f"Event '{event.name}' at {event.time} exceeds duration {self._duration}"
                )

        if skeleton:
            for bone_index in self._bone_tracks:
                if bone_index >= skeleton.bone_count:
                    errors.append(
                        f"Bone track index {bone_index} exceeds skeleton bone count"
                    )

        return errors

    def __repr__(self) -> str:
        return (
            f"AnimationClip('{self._name}', duration={self._duration:.3f}, "
            f"tracks={len(self._bone_tracks)}, events={len(self._events)}, "
            f"loop={self._looping})"
        )


def create_simple_clip(
    name: str,
    duration: float,
    bone_index: int,
    start_position: Vec3,
    end_position: Vec3,
    start_rotation: Optional[Quat] = None,
    end_rotation: Optional[Quat] = None,
) -> AnimationClip:
    """Helper to create a simple animation clip.

    Args:
        name: Clip name.
        duration: Duration in seconds.
        bone_index: Index of bone to animate.
        start_position: Starting position.
        end_position: Ending position.
        start_rotation: Starting rotation (optional).
        end_rotation: Ending rotation (optional).

    Returns:
        New animation clip.
    """
    pos_curve = AnimationCurve(
        [
            Keyframe(0.0, start_position),
            Keyframe(duration, end_position),
        ],
        InterpolationType.LINEAR,
    )

    rot_curve = None
    if start_rotation and end_rotation:
        rot_curve = AnimationCurve(
            [
                Keyframe(0.0, start_rotation),
                Keyframe(duration, end_rotation),
            ],
            InterpolationType.LINEAR,
        )

    track = BoneTrack(
        bone_index=bone_index,
        position_curve=pos_curve,
        rotation_curve=rot_curve,
    )

    return AnimationClip(
        name=name,
        duration=duration,
        bone_tracks={bone_index: track},
    )


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# AnimationTrack is the legacy name for BoneTrack
AnimationTrack = BoneTrack

# AnimationKeyframe is the legacy name for Keyframe
AnimationKeyframe = Keyframe
