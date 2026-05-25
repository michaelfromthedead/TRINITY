"""Animation notify events with timing.

Provides tools for creating and managing animation notifies including
sound, particle, custom event, and footstep notifies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# ENUMS
# =============================================================================


class NotifyType(Enum):
    """Types of animation notifies."""

    INSTANT = auto()     # Single-frame notify
    STATE = auto()       # Duration-based notify state
    SOUND = auto()       # Sound playback
    PARTICLE = auto()    # Particle effect
    CUSTOM = auto()      # Custom event
    FOOTSTEP = auto()    # Footstep event
    TRAIL = auto()       # Trail effect
    CAMERA = auto()      # Camera effect


# =============================================================================
# NOTIFY PAYLOAD
# =============================================================================


@dataclass
class NotifyPayload:
    """Payload data for a notify event.

    Attributes:
        notify_type: Type of notify
        data: Notify-specific data
        bone_name: Optional bone for location
    """

    notify_type: NotifyType
    data: Dict[str, Any] = field(default_factory=dict)
    bone_name: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get data value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set data value."""
        self.data[key] = value


# =============================================================================
# BASE NOTIFY
# =============================================================================


class AnimNotify(ABC):
    """Base class for animation notifies.

    Notifies are events that trigger at specific times during animation
    playback.
    """

    def __init__(
        self,
        name: str,
        time: float,
        notify_type: NotifyType = NotifyType.INSTANT,
    ) -> None:
        if not name:
            raise ValueError("Notify name cannot be empty")
        if time < 0:
            raise ValueError(f"Notify time must be >= 0, got {time}")

        self._name = name
        self._time = time
        self._notify_type = notify_type
        self._enabled = True
        self._color: Tuple[int, int, int] = (200, 100, 100)
        self._track_index: int = 0

    @property
    def name(self) -> str:
        """Get notify name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set notify name."""
        if not value:
            raise ValueError("Notify name cannot be empty")
        self._name = value

    @property
    def time(self) -> float:
        """Get notify time."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        """Set notify time."""
        if value < 0:
            raise ValueError(f"Notify time must be >= 0, got {value}")
        self._time = value

    @property
    def notify_type(self) -> NotifyType:
        """Get notify type."""
        return self._notify_type

    @property
    def enabled(self) -> bool:
        """Check if notify is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._enabled = value

    @property
    def color(self) -> Tuple[int, int, int]:
        """Get display color."""
        return self._color

    @color.setter
    def color(self, value: Tuple[int, int, int]) -> None:
        """Set display color."""
        self._color = value

    @property
    def track_index(self) -> int:
        """Get track index for display."""
        return self._track_index

    @track_index.setter
    def track_index(self, value: int) -> None:
        """Set track index."""
        self._track_index = value

    @abstractmethod
    def get_payload(self) -> NotifyPayload:
        """Get notify payload for event dispatch."""
        pass

    @abstractmethod
    def copy(self) -> AnimNotify:
        """Create a copy of this notify."""
        pass


# =============================================================================
# NOTIFY STATE
# =============================================================================


class AnimNotifyState(AnimNotify):
    """A notify with duration (state notify).

    State notifies have a begin, tick, and end phase.
    """

    def __init__(
        self,
        name: str,
        time: float,
        duration: float,
    ) -> None:
        super().__init__(name, time, NotifyType.STATE)
        if duration < 0:
            raise ValueError(f"Duration must be >= 0, got {duration}")
        self._duration = duration
        self._color = (100, 100, 200)

    @property
    def duration(self) -> float:
        """Get duration."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set duration."""
        if value < 0:
            raise ValueError(f"Duration must be >= 0, got {value}")
        self._duration = value

    @property
    def end_time(self) -> float:
        """Get end time."""
        return self._time + self._duration

    def contains_time(self, time: float) -> bool:
        """Check if time is within notify duration."""
        return self._time <= time <= self.end_time

    def get_payload(self) -> NotifyPayload:
        """Get notify payload."""
        return NotifyPayload(
            notify_type=self._notify_type,
            data={
                "name": self._name,
                "duration": self._duration,
            },
        )

    def copy(self) -> AnimNotifyState:
        """Create a copy."""
        notify = AnimNotifyState(self._name, self._time, self._duration)
        notify._enabled = self._enabled
        notify._color = self._color
        notify._track_index = self._track_index
        return notify


# =============================================================================
# SOUND NOTIFY
# =============================================================================


class SoundNotify(AnimNotify):
    """Notify for playing sounds."""

    def __init__(
        self,
        name: str,
        time: float,
        sound_asset: str,
        volume: float = 1.0,
        pitch: float = 1.0,
    ) -> None:
        super().__init__(name, time, NotifyType.SOUND)
        self._sound_asset = sound_asset
        self._volume = volume
        self._pitch = pitch
        self._attach_to_bone: Optional[str] = None
        self._color = (100, 200, 100)

    @property
    def sound_asset(self) -> str:
        """Get sound asset path."""
        return self._sound_asset

    @sound_asset.setter
    def sound_asset(self, value: str) -> None:
        """Set sound asset path."""
        self._sound_asset = value

    @property
    def volume(self) -> float:
        """Get volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set volume."""
        self._volume = max(0.0, min(1.0, value))

    @property
    def pitch(self) -> float:
        """Get pitch."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set pitch."""
        self._pitch = max(0.1, min(3.0, value))

    @property
    def attach_to_bone(self) -> Optional[str]:
        """Get bone to attach sound to."""
        return self._attach_to_bone

    @attach_to_bone.setter
    def attach_to_bone(self, value: Optional[str]) -> None:
        """Set bone to attach sound to."""
        self._attach_to_bone = value

    def get_payload(self) -> NotifyPayload:
        """Get notify payload."""
        return NotifyPayload(
            notify_type=self._notify_type,
            data={
                "name": self._name,
                "sound_asset": self._sound_asset,
                "volume": self._volume,
                "pitch": self._pitch,
            },
            bone_name=self._attach_to_bone,
        )

    def copy(self) -> SoundNotify:
        """Create a copy."""
        notify = SoundNotify(
            self._name,
            self._time,
            self._sound_asset,
            self._volume,
            self._pitch,
        )
        notify._enabled = self._enabled
        notify._color = self._color
        notify._track_index = self._track_index
        notify._attach_to_bone = self._attach_to_bone
        return notify


# =============================================================================
# PARTICLE NOTIFY
# =============================================================================


class ParticleNotify(AnimNotify):
    """Notify for spawning particles."""

    def __init__(
        self,
        name: str,
        time: float,
        particle_asset: str,
        socket_name: Optional[str] = None,
        attached: bool = True,
    ) -> None:
        super().__init__(name, time, NotifyType.PARTICLE)
        self._particle_asset = particle_asset
        self._socket_name = socket_name
        self._attached = attached
        self._scale = 1.0
        self._location_offset = (0.0, 0.0, 0.0)
        self._rotation_offset = (0.0, 0.0, 0.0)
        self._color = (200, 150, 100)

    @property
    def particle_asset(self) -> str:
        """Get particle asset path."""
        return self._particle_asset

    @particle_asset.setter
    def particle_asset(self, value: str) -> None:
        """Set particle asset path."""
        self._particle_asset = value

    @property
    def socket_name(self) -> Optional[str]:
        """Get socket to attach particles to."""
        return self._socket_name

    @socket_name.setter
    def socket_name(self, value: Optional[str]) -> None:
        """Set socket to attach particles to."""
        self._socket_name = value

    @property
    def attached(self) -> bool:
        """Check if particles are attached to socket."""
        return self._attached

    @attached.setter
    def attached(self, value: bool) -> None:
        """Set attached state."""
        self._attached = value

    @property
    def scale(self) -> float:
        """Get particle scale."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        """Set particle scale."""
        self._scale = max(0.0, value)

    @property
    def location_offset(self) -> Tuple[float, float, float]:
        """Get location offset."""
        return self._location_offset

    @location_offset.setter
    def location_offset(self, value: Tuple[float, float, float]) -> None:
        """Set location offset."""
        self._location_offset = value

    @property
    def rotation_offset(self) -> Tuple[float, float, float]:
        """Get rotation offset (euler angles)."""
        return self._rotation_offset

    @rotation_offset.setter
    def rotation_offset(self, value: Tuple[float, float, float]) -> None:
        """Set rotation offset."""
        self._rotation_offset = value

    def get_payload(self) -> NotifyPayload:
        """Get notify payload."""
        return NotifyPayload(
            notify_type=self._notify_type,
            data={
                "name": self._name,
                "particle_asset": self._particle_asset,
                "attached": self._attached,
                "scale": self._scale,
                "location_offset": self._location_offset,
                "rotation_offset": self._rotation_offset,
            },
            bone_name=self._socket_name,
        )

    def copy(self) -> ParticleNotify:
        """Create a copy."""
        notify = ParticleNotify(
            self._name,
            self._time,
            self._particle_asset,
            self._socket_name,
            self._attached,
        )
        notify._enabled = self._enabled
        notify._color = self._color
        notify._track_index = self._track_index
        notify._scale = self._scale
        notify._location_offset = self._location_offset
        notify._rotation_offset = self._rotation_offset
        return notify


# =============================================================================
# CUSTOM EVENT NOTIFY
# =============================================================================


class CustomEventNotify(AnimNotify):
    """Notify for custom events."""

    def __init__(
        self,
        name: str,
        time: float,
        event_name: str,
    ) -> None:
        super().__init__(name, time, NotifyType.CUSTOM)
        self._event_name = event_name
        self._parameters: Dict[str, Any] = {}
        self._color = (200, 200, 100)

    @property
    def event_name(self) -> str:
        """Get event name."""
        return self._event_name

    @event_name.setter
    def event_name(self, value: str) -> None:
        """Set event name."""
        self._event_name = value

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get event parameters."""
        return dict(self._parameters)

    def set_parameter(self, key: str, value: Any) -> None:
        """Set an event parameter."""
        self._parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get an event parameter."""
        return self._parameters.get(key, default)

    def remove_parameter(self, key: str) -> bool:
        """Remove an event parameter."""
        if key in self._parameters:
            del self._parameters[key]
            return True
        return False

    def get_payload(self) -> NotifyPayload:
        """Get notify payload."""
        return NotifyPayload(
            notify_type=self._notify_type,
            data={
                "name": self._name,
                "event_name": self._event_name,
                "parameters": dict(self._parameters),
            },
        )

    def copy(self) -> CustomEventNotify:
        """Create a copy."""
        notify = CustomEventNotify(
            self._name,
            self._time,
            self._event_name,
        )
        notify._enabled = self._enabled
        notify._color = self._color
        notify._track_index = self._track_index
        notify._parameters = dict(self._parameters)
        return notify


# =============================================================================
# FOOTSTEP NOTIFY
# =============================================================================


class FootstepNotify(AnimNotify):
    """Notify for footstep events."""

    def __init__(
        self,
        name: str,
        time: float,
        foot: str = "left",  # "left" or "right"
    ) -> None:
        super().__init__(name, time, NotifyType.FOOTSTEP)
        self._foot = foot
        self._surface_type: Optional[str] = None
        self._foot_bone: Optional[str] = None
        self._color = (150, 100, 200)

    @property
    def foot(self) -> str:
        """Get foot (left/right)."""
        return self._foot

    @foot.setter
    def foot(self, value: str) -> None:
        """Set foot."""
        if value not in ("left", "right"):
            raise ValueError(f"Foot must be 'left' or 'right', got '{value}'")
        self._foot = value

    @property
    def surface_type(self) -> Optional[str]:
        """Get surface type override."""
        return self._surface_type

    @surface_type.setter
    def surface_type(self, value: Optional[str]) -> None:
        """Set surface type override."""
        self._surface_type = value

    @property
    def foot_bone(self) -> Optional[str]:
        """Get foot bone for location."""
        return self._foot_bone

    @foot_bone.setter
    def foot_bone(self, value: Optional[str]) -> None:
        """Set foot bone."""
        self._foot_bone = value

    def get_payload(self) -> NotifyPayload:
        """Get notify payload."""
        return NotifyPayload(
            notify_type=self._notify_type,
            data={
                "name": self._name,
                "foot": self._foot,
                "surface_type": self._surface_type,
            },
            bone_name=self._foot_bone,
        )

    def copy(self) -> FootstepNotify:
        """Create a copy."""
        notify = FootstepNotify(
            self._name,
            self._time,
            self._foot,
        )
        notify._enabled = self._enabled
        notify._color = self._color
        notify._track_index = self._track_index
        notify._surface_type = self._surface_type
        notify._foot_bone = self._foot_bone
        return notify


# =============================================================================
# NOTIFY TIMING
# =============================================================================


@dataclass
class NotifyTiming:
    """Timing configuration for notifies.

    Attributes:
        trigger_in_editor: Whether to trigger in editor preview
        trigger_weight_threshold: Minimum blend weight to trigger
        montage_tick_type: When to tick in montages
    """

    trigger_in_editor: bool = True
    trigger_weight_threshold: float = 0.0
    montage_tick_type: str = "queued"  # queued, branching_point


# =============================================================================
# NOTIFY TRACK
# =============================================================================


class NotifyTrack:
    """A track for organizing notifies.

    Attributes:
        name: Track name
        notifies: List of notifies on this track
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Track name cannot be empty")
        self._name = name
        self._notifies: List[AnimNotify] = []
        self._color: Tuple[int, int, int] = (128, 128, 128)

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
    def notifies(self) -> List[AnimNotify]:
        """Get all notifies."""
        return list(self._notifies)

    @property
    def notify_count(self) -> int:
        """Get number of notifies."""
        return len(self._notifies)

    @property
    def color(self) -> Tuple[int, int, int]:
        """Get track color."""
        return self._color

    @color.setter
    def color(self, value: Tuple[int, int, int]) -> None:
        """Set track color."""
        self._color = value

    def add_notify(self, notify: AnimNotify) -> None:
        """Add a notify to the track."""
        self._notifies.append(notify)
        self._notifies.sort(key=lambda n: n.time)

    def remove_notify(self, notify: AnimNotify) -> bool:
        """Remove a notify from the track."""
        if notify in self._notifies:
            self._notifies.remove(notify)
            return True
        return False

    def get_notifies_at(self, time: float, tolerance: float = 0.001) -> List[AnimNotify]:
        """Get notifies at a specific time."""
        return [n for n in self._notifies if abs(n.time - time) <= tolerance]

    def get_notifies_in_range(self, start: float, end: float) -> List[AnimNotify]:
        """Get notifies within a time range."""
        return [n for n in self._notifies if start <= n.time <= end]


# =============================================================================
# NOTIFIES EDITOR
# =============================================================================


class NotifiesEditor:
    """Editor for animation notifies.

    Provides functionality for creating and managing animation notifies
    with timing and track organization.
    """

    def __init__(self) -> None:
        self._tracks: List[NotifyTrack] = []
        self._selected_notify: Optional[AnimNotify] = None
        self._selected_track: int = -1
        self._timing = NotifyTiming()
        self._on_change_callbacks: List[Callable[[], None]] = []

        # Create default track
        self._tracks.append(NotifyTrack("Default"))

    @property
    def tracks(self) -> List[NotifyTrack]:
        """Get all tracks."""
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        """Get number of tracks."""
        return len(self._tracks)

    @property
    def selected_notify(self) -> Optional[AnimNotify]:
        """Get selected notify."""
        return self._selected_notify

    @property
    def selected_track(self) -> int:
        """Get selected track index."""
        return self._selected_track

    @property
    def timing(self) -> NotifyTiming:
        """Get timing configuration."""
        return self._timing

    def add_track(self, name: str) -> NotifyTrack:
        """Add a new track."""
        track = NotifyTrack(name)
        self._tracks.append(track)
        self._notify_change()
        return track

    def remove_track(self, index: int) -> bool:
        """Remove a track (cannot remove last track)."""
        if len(self._tracks) <= 1:
            return False
        if 0 <= index < len(self._tracks):
            self._tracks.pop(index)
            if self._selected_track >= len(self._tracks):
                self._selected_track = len(self._tracks) - 1
            self._notify_change()
            return True
        return False

    def get_track(self, index: int) -> Optional[NotifyTrack]:
        """Get track by index."""
        if 0 <= index < len(self._tracks):
            return self._tracks[index]
        return None

    def select_track(self, index: int) -> None:
        """Select a track."""
        if 0 <= index < len(self._tracks):
            self._selected_track = index

    def add_sound_notify(
        self,
        track_index: int,
        name: str,
        time: float,
        sound_asset: str,
        volume: float = 1.0,
    ) -> Optional[SoundNotify]:
        """Add a sound notify."""
        track = self.get_track(track_index)
        if track is None:
            return None

        notify = SoundNotify(name, time, sound_asset, volume)
        notify.track_index = track_index
        track.add_notify(notify)
        self._notify_change()
        return notify

    def add_particle_notify(
        self,
        track_index: int,
        name: str,
        time: float,
        particle_asset: str,
        socket_name: Optional[str] = None,
    ) -> Optional[ParticleNotify]:
        """Add a particle notify."""
        track = self.get_track(track_index)
        if track is None:
            return None

        notify = ParticleNotify(name, time, particle_asset, socket_name)
        notify.track_index = track_index
        track.add_notify(notify)
        self._notify_change()
        return notify

    def add_custom_notify(
        self,
        track_index: int,
        name: str,
        time: float,
        event_name: str,
    ) -> Optional[CustomEventNotify]:
        """Add a custom event notify."""
        track = self.get_track(track_index)
        if track is None:
            return None

        notify = CustomEventNotify(name, time, event_name)
        notify.track_index = track_index
        track.add_notify(notify)
        self._notify_change()
        return notify

    def add_footstep_notify(
        self,
        track_index: int,
        name: str,
        time: float,
        foot: str = "left",
    ) -> Optional[FootstepNotify]:
        """Add a footstep notify."""
        track = self.get_track(track_index)
        if track is None:
            return None

        notify = FootstepNotify(name, time, foot)
        notify.track_index = track_index
        track.add_notify(notify)
        self._notify_change()
        return notify

    def add_notify_state(
        self,
        track_index: int,
        name: str,
        time: float,
        duration: float,
    ) -> Optional[AnimNotifyState]:
        """Add a notify state."""
        track = self.get_track(track_index)
        if track is None:
            return None

        notify = AnimNotifyState(name, time, duration)
        notify.track_index = track_index
        track.add_notify(notify)
        self._notify_change()
        return notify

    def remove_notify(self, notify: AnimNotify) -> bool:
        """Remove a notify."""
        for track in self._tracks:
            if track.remove_notify(notify):
                if self._selected_notify == notify:
                    self._selected_notify = None
                self._notify_change()
                return True
        return False

    def move_notify(self, notify: AnimNotify, new_time: float) -> None:
        """Move a notify to a new time."""
        old_track_index = notify.track_index
        track = self.get_track(old_track_index)
        if track:
            track.remove_notify(notify)
            notify.time = new_time
            track.add_notify(notify)
            self._notify_change()

    def move_notify_to_track(self, notify: AnimNotify, track_index: int) -> bool:
        """Move a notify to a different track."""
        old_track = self.get_track(notify.track_index)
        new_track = self.get_track(track_index)

        if old_track is None or new_track is None:
            return False

        old_track.remove_notify(notify)
        notify.track_index = track_index
        new_track.add_notify(notify)
        self._notify_change()
        return True

    def select_notify(self, notify: Optional[AnimNotify]) -> None:
        """Select a notify."""
        self._selected_notify = notify

    def duplicate_notify(self, notify: AnimNotify, time_offset: float = 0.0) -> Optional[AnimNotify]:
        """Duplicate a notify."""
        track = self.get_track(notify.track_index)
        if track is None:
            return None

        new_notify = notify.copy()
        new_notify.time = notify.time + time_offset
        new_notify.name = f"{notify.name}_copy"
        track.add_notify(new_notify)
        self._notify_change()
        return new_notify

    def get_all_notifies(self) -> List[AnimNotify]:
        """Get all notifies from all tracks."""
        all_notifies = []
        for track in self._tracks:
            all_notifies.extend(track.notifies)
        return sorted(all_notifies, key=lambda n: n.time)

    def get_notifies_at(self, time: float, tolerance: float = 0.001) -> List[AnimNotify]:
        """Get all notifies at a specific time."""
        result = []
        for track in self._tracks:
            result.extend(track.get_notifies_at(time, tolerance))
        return result

    def get_notifies_in_range(self, start: float, end: float) -> List[AnimNotify]:
        """Get all notifies within a time range."""
        result = []
        for track in self._tracks:
            result.extend(track.get_notifies_in_range(start, end))
        return sorted(result, key=lambda n: n.time)

    def clear_all(self) -> None:
        """Clear all notifies from all tracks."""
        for track in self._tracks:
            track._notifies.clear()
        self._selected_notify = None
        self._notify_change()

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "NotifyType",
    "NotifyPayload",
    "AnimNotify",
    "AnimNotifyState",
    "SoundNotify",
    "ParticleNotify",
    "CustomEventNotify",
    "FootstepNotify",
    "NotifyTiming",
    "NotifyTrack",
    "NotifiesEditor",
]
