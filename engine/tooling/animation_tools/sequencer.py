"""Animation sequencer with tracks, keyframes, and timeline.

The sequencer provides timeline-based animation editing with support for
multiple track types including transform, skeletal, camera, event, audio,
and property tracks. It integrates with Foundation for undo/redo support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.core.math import Quat, Transform, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class TrackType(Enum):
    """Types of animation tracks."""

    TRANSFORM = auto()
    SKELETAL = auto()
    CAMERA = auto()
    EVENT = auto()
    AUDIO = auto()
    PROPERTY = auto()


class PlaybackMode(Enum):
    """Playback modes for the sequencer."""

    ONCE = auto()
    LOOP = auto()
    PING_PONG = auto()
    CLAMP = auto()


# =============================================================================
# KEYFRAMES
# =============================================================================


@dataclass
class Keyframe:
    """Base class for keyframes in an animation track.

    Attributes:
        time: Time position in seconds
        value: Value at this keyframe
        interpolation: Interpolation mode to next keyframe
        tangent_in: Incoming tangent for curve interpolation
        tangent_out: Outgoing tangent for curve interpolation
    """

    time: float
    value: Any = None
    interpolation: str = "linear"
    tangent_in: float = 0.0
    tangent_out: float = 0.0

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Keyframe time must be >= 0, got {self.time}")

    def copy(self) -> Keyframe:
        """Create a copy of this keyframe."""
        return Keyframe(
            time=self.time,
            value=self.value,
            interpolation=self.interpolation,
            tangent_in=self.tangent_in,
            tangent_out=self.tangent_out,
        )


@dataclass
class TransformKeyframe(Keyframe):
    """Keyframe for transform data (position, rotation, scale)."""

    position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    rotation: Quat = field(default_factory=Quat.identity)
    scale: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.value = Transform(self.position, self.rotation, self.scale)

    def to_transform(self) -> Transform:
        """Convert to a Transform object."""
        return Transform(self.position, self.rotation, self.scale)

    def copy(self) -> TransformKeyframe:
        """Create a copy of this keyframe."""
        return TransformKeyframe(
            time=self.time,
            interpolation=self.interpolation,
            tangent_in=self.tangent_in,
            tangent_out=self.tangent_out,
            position=Vec3(self.position.x, self.position.y, self.position.z),
            rotation=Quat(
                self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w
            ),
            scale=Vec3(self.scale.x, self.scale.y, self.scale.z),
        )


@dataclass
class EventKeyframe(Keyframe):
    """Keyframe for event triggers."""

    event_name: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.value = {"name": self.event_name, "data": self.event_data}

    def copy(self) -> EventKeyframe:
        """Create a copy of this keyframe."""
        return EventKeyframe(
            time=self.time,
            event_name=self.event_name,
            event_data=dict(self.event_data),
        )


@dataclass
class PropertyKeyframe(Keyframe):
    """Keyframe for arbitrary property values."""

    property_name: str = ""
    property_value: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.value = self.property_value

    def copy(self) -> PropertyKeyframe:
        """Create a copy of this keyframe."""
        return PropertyKeyframe(
            time=self.time,
            interpolation=self.interpolation,
            tangent_in=self.tangent_in,
            tangent_out=self.tangent_out,
            property_name=self.property_name,
            property_value=self.property_value,
        )


# =============================================================================
# TIMELINE
# =============================================================================


@dataclass
class TimelineRange:
    """A range on the timeline.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        color: Display color for the range
        label: Optional label for the range
    """

    start: float
    end: float
    color: Tuple[int, int, int] = (100, 100, 100)
    label: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Range start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"Range end must be >= start, got {self.end} < {self.start}")

    @property
    def duration(self) -> float:
        """Get the duration of the range."""
        return self.end - self.start

    def contains(self, time: float) -> bool:
        """Check if a time is within the range."""
        return self.start <= time <= self.end

    def overlaps(self, other: TimelineRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end


class Timeline:
    """Timeline for managing time and ranges.

    The timeline provides time management, range markers, and snapping
    functionality for the sequencer.

    Attributes:
        duration: Total duration in seconds
        frame_rate: Frames per second
    """

    def __init__(
        self,
        duration: float = 10.0,
        frame_rate: float = 30.0,
    ) -> None:
        if duration <= 0:
            raise ValueError(f"Duration must be > 0, got {duration}")
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be > 0, got {frame_rate}")

        self._duration = duration
        self._frame_rate = frame_rate
        self._ranges: List[TimelineRange] = []
        self._markers: Dict[str, float] = {}
        self._snap_interval: float = 0.0
        self._loop_range: Optional[TimelineRange] = None

    @property
    def duration(self) -> float:
        """Get the timeline duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the timeline duration."""
        if value <= 0:
            raise ValueError(f"Duration must be > 0, got {value}")
        self._duration = value

    @property
    def frame_rate(self) -> float:
        """Get the frame rate."""
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value: float) -> None:
        """Set the frame rate."""
        if value <= 0:
            raise ValueError(f"Frame rate must be > 0, got {value}")
        self._frame_rate = value

    @property
    def frame_duration(self) -> float:
        """Get the duration of one frame."""
        return 1.0 / self._frame_rate

    @property
    def total_frames(self) -> int:
        """Get the total number of frames."""
        return int(self._duration * self._frame_rate)

    @property
    def ranges(self) -> List[TimelineRange]:
        """Get all ranges."""
        return list(self._ranges)

    @property
    def markers(self) -> Dict[str, float]:
        """Get all markers."""
        return dict(self._markers)

    def time_to_frame(self, time: float) -> int:
        """Convert time to frame number."""
        return int(time * self._frame_rate)

    def frame_to_time(self, frame: int) -> float:
        """Convert frame number to time."""
        return frame / self._frame_rate

    def snap_time(self, time: float) -> float:
        """Snap time to the nearest snap interval."""
        if self._snap_interval <= 0:
            return time
        return round(time / self._snap_interval) * self._snap_interval

    def set_snap_interval(self, interval: float) -> None:
        """Set the snap interval."""
        self._snap_interval = max(0.0, interval)

    def snap_to_frame(self, time: float) -> float:
        """Snap time to the nearest frame."""
        frame = round(time * self._frame_rate)
        return frame / self._frame_rate

    def add_range(self, range_obj: TimelineRange) -> None:
        """Add a range to the timeline."""
        self._ranges.append(range_obj)

    def remove_range(self, range_obj: TimelineRange) -> bool:
        """Remove a range from the timeline."""
        if range_obj in self._ranges:
            self._ranges.remove(range_obj)
            return True
        return False

    def get_ranges_at(self, time: float) -> List[TimelineRange]:
        """Get all ranges that contain the given time."""
        return [r for r in self._ranges if r.contains(time)]

    def add_marker(self, name: str, time: float) -> None:
        """Add a named marker."""
        if time < 0 or time > self._duration:
            raise ValueError(f"Marker time {time} out of range [0, {self._duration}]")
        self._markers[name] = time

    def remove_marker(self, name: str) -> bool:
        """Remove a marker by name."""
        if name in self._markers:
            del self._markers[name]
            return True
        return False

    def get_marker(self, name: str) -> Optional[float]:
        """Get marker time by name."""
        return self._markers.get(name)

    def set_loop_range(self, start: float, end: float) -> None:
        """Set the loop range."""
        self._loop_range = TimelineRange(start, end)

    def clear_loop_range(self) -> None:
        """Clear the loop range."""
        self._loop_range = None

    @property
    def loop_range(self) -> Optional[TimelineRange]:
        """Get the loop range."""
        return self._loop_range

    def clamp_time(self, time: float) -> float:
        """Clamp time to valid range."""
        return max(0.0, min(time, self._duration))


# =============================================================================
# TRACKS
# =============================================================================


T = TypeVar("T", bound=Keyframe)


class AnimationTrack(ABC, Generic[T]):
    """Base class for animation tracks.

    A track contains a sequence of keyframes that define animation data
    over time. Different track types support different keyframe types.

    Attributes:
        name: Track name
        track_type: Type of track
        muted: Whether the track is muted
        locked: Whether the track is locked for editing
    """

    def __init__(
        self,
        name: str,
        track_type: TrackType,
    ) -> None:
        if not name:
            raise ValueError("Track name cannot be empty")

        self._name = name
        self._track_type = track_type
        self._keyframes: List[T] = []
        self._muted = False
        self._locked = False
        self._color: Tuple[int, int, int] = (200, 200, 200)
        self._expanded = True

    @property
    def name(self) -> str:
        """Get track name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set track name."""
        if not value:
            raise ValueError("Track name cannot be empty")
        self._name = value

    @property
    def track_type(self) -> TrackType:
        """Get track type."""
        return self._track_type

    @property
    def keyframes(self) -> List[T]:
        """Get all keyframes (read-only)."""
        return list(self._keyframes)

    @property
    def keyframe_count(self) -> int:
        """Get number of keyframes."""
        return len(self._keyframes)

    @property
    def muted(self) -> bool:
        """Check if track is muted."""
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set muted state."""
        self._muted = value

    @property
    def locked(self) -> bool:
        """Check if track is locked."""
        return self._locked

    @locked.setter
    def locked(self, value: bool) -> None:
        """Set locked state."""
        self._locked = value

    @property
    def color(self) -> Tuple[int, int, int]:
        """Get track color."""
        return self._color

    @color.setter
    def color(self, value: Tuple[int, int, int]) -> None:
        """Set track color."""
        self._color = value

    @property
    def expanded(self) -> bool:
        """Check if track is expanded in UI."""
        return self._expanded

    @expanded.setter
    def expanded(self, value: bool) -> None:
        """Set expanded state."""
        self._expanded = value

    def add_keyframe(self, keyframe: T) -> None:
        """Add a keyframe to the track."""
        if self._locked:
            raise RuntimeError("Cannot add keyframe to locked track")

        # Insert in sorted order
        index = 0
        for i, kf in enumerate(self._keyframes):
            if kf.time > keyframe.time:
                break
            index = i + 1

        self._keyframes.insert(index, keyframe)

    def remove_keyframe(self, keyframe: T) -> bool:
        """Remove a keyframe from the track."""
        if self._locked:
            raise RuntimeError("Cannot remove keyframe from locked track")

        if keyframe in self._keyframes:
            self._keyframes.remove(keyframe)
            return True
        return False

    def remove_keyframe_at(self, time: float, tolerance: float = 0.001) -> bool:
        """Remove keyframe at the specified time."""
        if self._locked:
            raise RuntimeError("Cannot remove keyframe from locked track")

        for i, kf in enumerate(self._keyframes):
            if abs(kf.time - time) <= tolerance:
                self._keyframes.pop(i)
                return True
        return False

    def get_keyframe_at(self, time: float, tolerance: float = 0.001) -> Optional[T]:
        """Get keyframe at the specified time."""
        for kf in self._keyframes:
            if abs(kf.time - time) <= tolerance:
                return kf
        return None

    def get_keyframes_in_range(self, start: float, end: float) -> List[T]:
        """Get all keyframes within a time range."""
        return [kf for kf in self._keyframes if start <= kf.time <= end]

    def get_surrounding_keyframes(self, time: float) -> Tuple[Optional[T], Optional[T]]:
        """Get keyframes before and after the specified time."""
        before: Optional[T] = None
        after: Optional[T] = None

        for kf in self._keyframes:
            if kf.time <= time:
                before = kf
            elif after is None:
                after = kf
                break

        return before, after

    def move_keyframe(self, keyframe: T, new_time: float) -> bool:
        """Move a keyframe to a new time."""
        if self._locked:
            raise RuntimeError("Cannot move keyframe on locked track")

        if keyframe not in self._keyframes:
            return False

        self._keyframes.remove(keyframe)
        keyframe.time = new_time
        self.add_keyframe(keyframe)
        return True

    def clear_keyframes(self) -> None:
        """Remove all keyframes."""
        if self._locked:
            raise RuntimeError("Cannot clear keyframes on locked track")
        self._keyframes.clear()

    @abstractmethod
    def evaluate(self, time: float) -> Any:
        """Evaluate the track at the specified time.

        Args:
            time: Time to evaluate at

        Returns:
            Interpolated value at the time
        """
        pass

    @property
    def duration(self) -> float:
        """Get track duration based on keyframes."""
        if not self._keyframes:
            return 0.0
        return self._keyframes[-1].time

    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range of keyframes."""
        if not self._keyframes:
            return (0.0, 0.0)
        return (self._keyframes[0].time, self._keyframes[-1].time)


class TransformTrack(AnimationTrack[TransformKeyframe]):
    """Track for transform animation data."""

    def __init__(self, name: str) -> None:
        super().__init__(name, TrackType.TRANSFORM)
        self._color = (100, 200, 100)

    def evaluate(self, time: float) -> Transform:
        """Evaluate transform at time."""
        if not self._keyframes:
            return Transform.identity()

        before, after = self.get_surrounding_keyframes(time)

        if before is None:
            return self._keyframes[0].to_transform()

        if after is None:
            return before.to_transform()

        # Interpolate
        t = (time - before.time) / (after.time - before.time)

        pos = before.position.lerp(after.position, t)
        rot = before.rotation.slerp(after.rotation, t)
        scale = before.scale.lerp(after.scale, t)

        return Transform(pos, rot, scale)


class SkeletalTrack(AnimationTrack[TransformKeyframe]):
    """Track for skeletal animation with bone-specific data."""

    def __init__(self, name: str, bone_name: str) -> None:
        super().__init__(name, TrackType.SKELETAL)
        self._bone_name = bone_name
        self._color = (200, 150, 100)

    @property
    def bone_name(self) -> str:
        """Get bone name."""
        return self._bone_name

    def evaluate(self, time: float) -> Transform:
        """Evaluate bone transform at time."""
        if not self._keyframes:
            return Transform.identity()

        before, after = self.get_surrounding_keyframes(time)

        if before is None:
            return self._keyframes[0].to_transform()

        if after is None:
            return before.to_transform()

        t = (time - before.time) / (after.time - before.time)

        pos = before.position.lerp(after.position, t)
        rot = before.rotation.slerp(after.rotation, t)
        scale = before.scale.lerp(after.scale, t)

        return Transform(pos, rot, scale)


class CameraTrack(AnimationTrack[TransformKeyframe]):
    """Track for camera animation."""

    def __init__(self, name: str) -> None:
        super().__init__(name, TrackType.CAMERA)
        self._fov_keyframes: List[Keyframe] = []
        self._color = (100, 100, 200)

    def add_fov_keyframe(self, time: float, fov: float) -> None:
        """Add a FOV keyframe."""
        kf = Keyframe(time=time, value=fov)
        # Insert in sorted order
        index = 0
        for i, k in enumerate(self._fov_keyframes):
            if k.time > time:
                break
            index = i + 1
        self._fov_keyframes.insert(index, kf)

    def evaluate_fov(self, time: float) -> float:
        """Evaluate FOV at time."""
        if not self._fov_keyframes:
            return 60.0

        before: Optional[Keyframe] = None
        after: Optional[Keyframe] = None

        for kf in self._fov_keyframes:
            if kf.time <= time:
                before = kf
            elif after is None:
                after = kf
                break

        if before is None:
            return self._fov_keyframes[0].value

        if after is None:
            return before.value

        t = (time - before.time) / (after.time - before.time)
        return before.value + (after.value - before.value) * t

    def evaluate(self, time: float) -> Dict[str, Any]:
        """Evaluate camera data at time."""
        transform = Transform.identity()
        if self._keyframes:
            before, after = self.get_surrounding_keyframes(time)
            if before is None:
                transform = self._keyframes[0].to_transform()
            elif after is None:
                transform = before.to_transform()
            else:
                t = (time - before.time) / (after.time - before.time)
                pos = before.position.lerp(after.position, t)
                rot = before.rotation.slerp(after.rotation, t)
                scale = before.scale.lerp(after.scale, t)
                transform = Transform(pos, rot, scale)

        return {
            "transform": transform,
            "fov": self.evaluate_fov(time),
        }


class EventTrack(AnimationTrack[EventKeyframe]):
    """Track for event triggers."""

    def __init__(self, name: str) -> None:
        super().__init__(name, TrackType.EVENT)
        self._color = (200, 100, 100)

    def evaluate(self, time: float) -> Optional[Dict[str, Any]]:
        """Get event at time (if any)."""
        for kf in self._keyframes:
            if abs(kf.time - time) < 0.001:
                return kf.value
        return None

    def get_events_in_range(self, start: float, end: float) -> List[EventKeyframe]:
        """Get all events within a time range."""
        return [kf for kf in self._keyframes if start <= kf.time <= end]


class AudioTrack(AnimationTrack[Keyframe]):
    """Track for audio playback."""

    def __init__(self, name: str) -> None:
        super().__init__(name, TrackType.AUDIO)
        self._audio_clips: List[Dict[str, Any]] = []
        self._color = (100, 200, 200)

    def add_audio_clip(
        self,
        asset_path: str,
        start_time: float,
        duration: float,
        volume: float = 1.0,
    ) -> None:
        """Add an audio clip to the track."""
        self._audio_clips.append({
            "asset_path": asset_path,
            "start_time": start_time,
            "duration": duration,
            "volume": volume,
        })

    def get_clips_at(self, time: float) -> List[Dict[str, Any]]:
        """Get audio clips playing at time."""
        return [
            clip for clip in self._audio_clips
            if clip["start_time"] <= time < clip["start_time"] + clip["duration"]
        ]

    def evaluate(self, time: float) -> List[Dict[str, Any]]:
        """Get active audio clips at time."""
        return self.get_clips_at(time)


class PropertyTrack(AnimationTrack[PropertyKeyframe]):
    """Track for animating arbitrary properties."""

    def __init__(self, name: str, target_path: str, property_name: str) -> None:
        super().__init__(name, TrackType.PROPERTY)
        self._target_path = target_path
        self._property_name = property_name
        self._color = (200, 200, 100)

    @property
    def target_path(self) -> str:
        """Get target object path."""
        return self._target_path

    @property
    def property_name(self) -> str:
        """Get property name."""
        return self._property_name

    def evaluate(self, time: float) -> Any:
        """Evaluate property value at time."""
        if not self._keyframes:
            return None

        before, after = self.get_surrounding_keyframes(time)

        if before is None:
            return self._keyframes[0].property_value

        if after is None:
            return before.property_value

        # Try to interpolate if numeric
        try:
            t = (time - before.time) / (after.time - before.time)
            return before.property_value + (after.property_value - before.property_value) * t
        except (TypeError, ValueError):
            # Non-numeric, return before value
            return before.property_value


# =============================================================================
# PLAYBACK
# =============================================================================


@dataclass
class SequencerPlayback:
    """Manages playback state for the sequencer.

    Attributes:
        current_time: Current playback time
        is_playing: Whether playback is active
        playback_speed: Playback speed multiplier
        mode: Playback mode
    """

    current_time: float = 0.0
    is_playing: bool = False
    playback_speed: float = 1.0
    mode: PlaybackMode = PlaybackMode.ONCE
    _direction: int = 1

    def update(self, dt: float, duration: float, loop_range: Optional[TimelineRange] = None) -> List[Tuple[float, float]]:
        """Update playback and return time ranges traversed.

        Args:
            dt: Delta time
            duration: Total duration
            loop_range: Optional loop range

        Returns:
            List of (start, end) time ranges traversed
        """
        if not self.is_playing:
            return []

        old_time = self.current_time
        self.current_time += dt * self.playback_speed * self._direction

        # Handle loop range
        effective_start = 0.0
        effective_end = duration

        if loop_range:
            effective_start = loop_range.start
            effective_end = loop_range.end

        traversed: List[Tuple[float, float]] = []

        if self.mode == PlaybackMode.ONCE:
            if self.current_time >= effective_end:
                traversed.append((old_time, effective_end))
                self.current_time = effective_end
                self.is_playing = False
            else:
                traversed.append((old_time, self.current_time))

        elif self.mode == PlaybackMode.LOOP:
            if self.current_time >= effective_end:
                traversed.append((old_time, effective_end))
                self.current_time = effective_start + (self.current_time - effective_end)
                traversed.append((effective_start, self.current_time))
            else:
                traversed.append((old_time, self.current_time))

        elif self.mode == PlaybackMode.PING_PONG:
            if self._direction > 0 and self.current_time >= effective_end:
                traversed.append((old_time, effective_end))
                self._direction = -1
                self.current_time = effective_end - (self.current_time - effective_end)
            elif self._direction < 0 and self.current_time <= effective_start:
                traversed.append((self.current_time, old_time))
                self._direction = 1
                self.current_time = effective_start + (effective_start - self.current_time)
            else:
                if self._direction > 0:
                    traversed.append((old_time, self.current_time))
                else:
                    traversed.append((self.current_time, old_time))

        elif self.mode == PlaybackMode.CLAMP:
            self.current_time = max(effective_start, min(effective_end, self.current_time))
            traversed.append((old_time, self.current_time))

        return traversed

    def play(self) -> None:
        """Start playback."""
        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def stop(self) -> None:
        """Stop and reset playback."""
        self.is_playing = False
        self.current_time = 0.0
        self._direction = 1

    def seek(self, time: float) -> None:
        """Seek to a specific time."""
        self.current_time = max(0.0, time)


# =============================================================================
# SEQUENCER
# =============================================================================


class AnimationSequencer:
    """Main animation sequencer for timeline-based editing.

    The sequencer manages multiple tracks and provides playback, editing,
    and export functionality.

    Attributes:
        name: Sequencer name
        timeline: Timeline for time management
        playback: Playback state
    """

    def __init__(self, name: str = "Sequence") -> None:
        self._name = name
        self._timeline = Timeline()
        self._tracks: List[AnimationTrack] = []
        self._playback = SequencerPlayback()
        self._selected_tracks: Set[int] = set()
        self._selected_keyframes: List[Tuple[int, int]] = []  # (track_idx, kf_idx)
        self._event_callbacks: Dict[str, List[Callable]] = {}

    @property
    def name(self) -> str:
        """Get sequencer name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set sequencer name."""
        self._name = value

    @property
    def timeline(self) -> Timeline:
        """Get the timeline."""
        return self._timeline

    @property
    def playback(self) -> SequencerPlayback:
        """Get playback state."""
        return self._playback

    @property
    def tracks(self) -> List[AnimationTrack]:
        """Get all tracks."""
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        """Get number of tracks."""
        return len(self._tracks)

    @property
    def duration(self) -> float:
        """Get sequence duration."""
        return self._timeline.duration

    def add_track(self, track: AnimationTrack) -> int:
        """Add a track to the sequencer."""
        self._tracks.append(track)
        return len(self._tracks) - 1

    def remove_track(self, track: AnimationTrack) -> bool:
        """Remove a track from the sequencer."""
        if track in self._tracks:
            idx = self._tracks.index(track)
            self._tracks.remove(track)
            self._selected_tracks.discard(idx)
            return True
        return False

    def remove_track_at(self, index: int) -> bool:
        """Remove track at index."""
        if 0 <= index < len(self._tracks):
            self._tracks.pop(index)
            self._selected_tracks.discard(index)
            return True
        return False

    def get_track(self, index: int) -> Optional[AnimationTrack]:
        """Get track by index."""
        if 0 <= index < len(self._tracks):
            return self._tracks[index]
        return None

    def get_track_by_name(self, name: str) -> Optional[AnimationTrack]:
        """Get track by name."""
        for track in self._tracks:
            if track.name == name:
                return track
        return None

    def move_track(self, from_index: int, to_index: int) -> bool:
        """Move a track to a new position."""
        if not (0 <= from_index < len(self._tracks)):
            return False
        if not (0 <= to_index < len(self._tracks)):
            return False

        track = self._tracks.pop(from_index)
        self._tracks.insert(to_index, track)
        return True

    def select_track(self, index: int, add_to_selection: bool = False) -> None:
        """Select a track."""
        if not add_to_selection:
            self._selected_tracks.clear()
        if 0 <= index < len(self._tracks):
            self._selected_tracks.add(index)

    def deselect_track(self, index: int) -> None:
        """Deselect a track."""
        self._selected_tracks.discard(index)

    def get_selected_tracks(self) -> List[int]:
        """Get selected track indices."""
        return list(self._selected_tracks)

    def clear_selection(self) -> None:
        """Clear all selections."""
        self._selected_tracks.clear()
        self._selected_keyframes.clear()

    def evaluate(self, time: float) -> Dict[str, Any]:
        """Evaluate all tracks at time.

        Returns:
            Dictionary of track name to evaluated value
        """
        result = {}
        for track in self._tracks:
            if not track.muted:
                result[track.name] = track.evaluate(time)
        return result

    def update(self, dt: float) -> Dict[str, List[Dict[str, Any]]]:
        """Update playback and get triggered events.

        Returns:
            Dictionary mapping event names to event data
        """
        time_ranges = self._playback.update(
            dt,
            self._timeline.duration,
            self._timeline.loop_range,
        )

        events: Dict[str, List[Dict[str, Any]]] = {}

        for start, end in time_ranges:
            for track in self._tracks:
                if track.muted or not isinstance(track, EventTrack):
                    continue

                for event in track.get_events_in_range(start, end):
                    name = event.event_name
                    if name not in events:
                        events[name] = []
                    events[name].append({
                        "time": event.time,
                        "data": event.event_data,
                    })

        # Fire callbacks
        for event_name, event_list in events.items():
            if event_name in self._event_callbacks:
                for callback in self._event_callbacks[event_name]:
                    for event_data in event_list:
                        callback(event_data)

        return events

    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def unregister_event_callback(self, event_name: str, callback: Callable) -> bool:
        """Unregister an event callback."""
        if event_name in self._event_callbacks:
            try:
                self._event_callbacks[event_name].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def get_tracks_by_type(self, track_type: TrackType) -> List[AnimationTrack]:
        """Get all tracks of a specific type."""
        return [t for t in self._tracks if t.track_type == track_type]

    def create_transform_track(self, name: str) -> TransformTrack:
        """Create and add a transform track."""
        track = TransformTrack(name)
        self.add_track(track)
        return track

    def create_skeletal_track(self, name: str, bone_name: str) -> SkeletalTrack:
        """Create and add a skeletal track."""
        track = SkeletalTrack(name, bone_name)
        self.add_track(track)
        return track

    def create_camera_track(self, name: str) -> CameraTrack:
        """Create and add a camera track."""
        track = CameraTrack(name)
        self.add_track(track)
        return track

    def create_event_track(self, name: str) -> EventTrack:
        """Create and add an event track."""
        track = EventTrack(name)
        self.add_track(track)
        return track

    def create_audio_track(self, name: str) -> AudioTrack:
        """Create and add an audio track."""
        track = AudioTrack(name)
        self.add_track(track)
        return track

    def create_property_track(
        self,
        name: str,
        target_path: str,
        property_name: str,
    ) -> PropertyTrack:
        """Create and add a property track."""
        track = PropertyTrack(name, target_path, property_name)
        self.add_track(track)
        return track

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sequencer to dictionary."""
        return {
            "name": self._name,
            "duration": self._timeline.duration,
            "frame_rate": self._timeline.frame_rate,
            "track_count": len(self._tracks),
        }


__all__ = [
    "TrackType",
    "PlaybackMode",
    "Keyframe",
    "TransformKeyframe",
    "EventKeyframe",
    "PropertyKeyframe",
    "TimelineRange",
    "Timeline",
    "AnimationTrack",
    "TransformTrack",
    "SkeletalTrack",
    "CameraTrack",
    "EventTrack",
    "AudioTrack",
    "PropertyTrack",
    "SequencerPlayback",
    "AnimationSequencer",
]
