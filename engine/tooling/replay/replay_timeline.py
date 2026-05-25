"""
Replay Timeline - Timeline visualization with markers and events.

Provides timeline representation of replays including markers,
events, segments, and visualization data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional
import bisect


class MarkerType(Enum):
    """Types of timeline markers."""
    BOOKMARK = auto()  # User-placed bookmark
    EVENT = auto()  # Game event
    KEYFRAME = auto()  # State keyframe
    HIGHLIGHT = auto()  # Notable moment
    ERROR = auto()  # Error/issue
    CHECKPOINT = auto()  # Save checkpoint
    CUSTOM = auto()  # Custom marker


@dataclass
class TimelineMarker:
    """A marker on the timeline."""
    id: str
    frame: int
    timestamp: float
    marker_type: MarkerType
    label: str
    description: str = ""
    color: Optional[str] = None  # Hex color code
    icon: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'frame': self.frame,
            'timestamp': self.timestamp,
            'type': self.marker_type.name,
            'label': self.label,
            'description': self.description,
            'color': self.color,
            'icon': self.icon,
            'data': self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TimelineMarker':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            frame=data['frame'],
            timestamp=data['timestamp'],
            marker_type=MarkerType[data['type']],
            label=data['label'],
            description=data.get('description', ''),
            color=data.get('color'),
            icon=data.get('icon'),
            data=data.get('data', {}),
        )


@dataclass
class TimelineEvent:
    """An event that occurred during the replay."""
    id: str
    frame: int
    timestamp: float
    event_type: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0  # Duration if event spans time

    @property
    def end_timestamp(self) -> float:
        """Get end timestamp for events with duration."""
        return self.timestamp + self.duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'frame': self.frame,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'name': self.name,
            'data': self.data,
            'duration': self.duration,
        }


@dataclass
class TimelineSegment:
    """A segment/region of the timeline."""
    id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    label: str
    segment_type: str = "default"
    color: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        """Get number of frames in segment."""
        return self.end_frame - self.start_frame

    def contains_frame(self, frame: int) -> bool:
        """Check if segment contains frame."""
        return self.start_frame <= frame <= self.end_frame

    def contains_time(self, time: float) -> bool:
        """Check if segment contains timestamp."""
        return self.start_time <= time <= self.end_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'label': self.label,
            'segment_type': self.segment_type,
            'color': self.color,
            'data': self.data,
        }


@dataclass
class TimelineTrack:
    """A track containing timeline items."""
    id: str
    name: str
    track_type: str  # "markers", "events", "segments", "data"
    visible: bool = True
    color: Optional[str] = None
    height: int = 24  # Track height in pixels


class ReplayTimeline:
    """Timeline visualization for replay data.

    Provides a structured representation of replay data including
    markers, events, segments, and timing information.
    """
    __slots__ = (
        '_total_frames', '_total_duration', '_fps',
        '_markers', '_events', '_segments', '_tracks',
        '_marker_index', '_next_id', '_callbacks'
    )

    def __init__(
        self,
        total_frames: int = 0,
        total_duration: float = 0.0,
        fps: float = 60.0
    ):
        """Initialize timeline.

        Args:
            total_frames: Total frame count
            total_duration: Total duration in seconds
            fps: Frames per second
        """
        self._total_frames = total_frames
        self._total_duration = total_duration
        self._fps = fps

        self._markers: list[TimelineMarker] = []
        self._events: list[TimelineEvent] = []
        self._segments: list[TimelineSegment] = []
        self._tracks: dict[str, TimelineTrack] = {}

        # Index for fast marker lookup by frame
        self._marker_index: list[int] = []  # Sorted frame numbers

        self._next_id = 0
        self._callbacks: dict[str, list[Callable]] = {
            'marker_added': [],
            'marker_removed': [],
            'event_added': [],
            'segment_added': [],
        }

    @property
    def total_frames(self) -> int:
        """Get total frame count."""
        return self._total_frames

    @total_frames.setter
    def total_frames(self, value: int) -> None:
        """Set total frame count."""
        self._total_frames = value

    @property
    def total_duration(self) -> float:
        """Get total duration in seconds."""
        return self._total_duration

    @total_duration.setter
    def total_duration(self, value: float) -> None:
        """Set total duration."""
        self._total_duration = value

    @property
    def fps(self) -> float:
        """Get frames per second."""
        return self._fps

    @property
    def marker_count(self) -> int:
        """Get number of markers."""
        return len(self._markers)

    @property
    def event_count(self) -> int:
        """Get number of events."""
        return len(self._events)

    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self._segments)

    def add_marker(
        self,
        frame: int,
        label: str,
        marker_type: MarkerType = MarkerType.BOOKMARK,
        description: str = "",
        color: Optional[str] = None,
        data: Optional[dict] = None
    ) -> TimelineMarker:
        """Add a marker to the timeline.

        Args:
            frame: Frame number
            label: Marker label
            marker_type: Type of marker
            description: Optional description
            color: Optional color code
            data: Optional extra data

        Returns:
            The created marker
        """
        marker = TimelineMarker(
            id=self._generate_id('marker'),
            frame=frame,
            timestamp=self._frame_to_time(frame),
            marker_type=marker_type,
            label=label,
            description=description,
            color=color,
            data=data or {}
        )

        # Insert in sorted order
        idx = bisect.bisect_left(self._marker_index, frame)
        self._marker_index.insert(idx, frame)
        self._markers.insert(idx, marker)

        self._notify('marker_added', marker)
        return marker

    def remove_marker(self, marker_id: str) -> bool:
        """Remove a marker by ID.

        Args:
            marker_id: Marker ID

        Returns:
            True if marker was removed
        """
        for i, marker in enumerate(self._markers):
            if marker.id == marker_id:
                self._markers.pop(i)
                self._marker_index.pop(i)
                self._notify('marker_removed', marker)
                return True
        return False

    def get_marker(self, marker_id: str) -> Optional[TimelineMarker]:
        """Get a marker by ID.

        Args:
            marker_id: Marker ID

        Returns:
            Marker if found
        """
        for marker in self._markers:
            if marker.id == marker_id:
                return marker
        return None

    def get_markers_at_frame(self, frame: int) -> list[TimelineMarker]:
        """Get all markers at a specific frame.

        Args:
            frame: Frame number

        Returns:
            List of markers at that frame
        """
        return [m for m in self._markers if m.frame == frame]

    def get_markers_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> list[TimelineMarker]:
        """Get markers within frame range.

        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)

        Returns:
            List of markers in range
        """
        return [
            m for m in self._markers
            if start_frame <= m.frame <= end_frame
        ]

    def get_markers_by_type(self, marker_type: MarkerType) -> list[TimelineMarker]:
        """Get all markers of a specific type.

        Args:
            marker_type: Marker type

        Returns:
            List of markers of that type
        """
        return [m for m in self._markers if m.marker_type == marker_type]

    def get_nearest_marker(
        self,
        frame: int,
        direction: int = 0
    ) -> Optional[TimelineMarker]:
        """Get nearest marker to a frame.

        Args:
            frame: Reference frame
            direction: 0=any, -1=previous, 1=next

        Returns:
            Nearest marker, or None if not found
        """
        if not self._markers:
            return None

        if direction == 0:
            # Find closest in either direction
            idx = bisect.bisect_left(self._marker_index, frame)
            candidates = []
            if idx > 0:
                candidates.append(self._markers[idx - 1])
            if idx < len(self._markers):
                candidates.append(self._markers[idx])
            if candidates:
                return min(candidates, key=lambda m: abs(m.frame - frame))
        elif direction < 0:
            # Find previous
            idx = bisect.bisect_left(self._marker_index, frame)
            if idx > 0:
                return self._markers[idx - 1]
        else:
            # Find next
            idx = bisect.bisect_right(self._marker_index, frame)
            if idx < len(self._markers):
                return self._markers[idx]

        return None

    def add_event(
        self,
        frame: int,
        event_type: str,
        name: str,
        data: Optional[dict] = None,
        duration: float = 0.0
    ) -> TimelineEvent:
        """Add an event to the timeline.

        Args:
            frame: Frame number
            event_type: Type of event
            name: Event name
            data: Optional event data
            duration: Event duration (if applicable)

        Returns:
            The created event
        """
        event = TimelineEvent(
            id=self._generate_id('event'),
            frame=frame,
            timestamp=self._frame_to_time(frame),
            event_type=event_type,
            name=name,
            data=data or {},
            duration=duration
        )

        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

        self._notify('event_added', event)
        return event

    def get_events_at_frame(self, frame: int) -> list[TimelineEvent]:
        """Get all events at a specific frame.

        Args:
            frame: Frame number

        Returns:
            List of events at that frame
        """
        return [e for e in self._events if e.frame == frame]

    def get_events_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> list[TimelineEvent]:
        """Get events within frame range.

        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)

        Returns:
            List of events in range
        """
        return [
            e for e in self._events
            if start_frame <= e.frame <= end_frame
        ]

    def get_events_by_type(self, event_type: str) -> list[TimelineEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Event type string

        Returns:
            List of events of that type
        """
        return [e for e in self._events if e.event_type == event_type]

    def add_segment(
        self,
        start_frame: int,
        end_frame: int,
        label: str,
        segment_type: str = "default",
        color: Optional[str] = None,
        data: Optional[dict] = None
    ) -> TimelineSegment:
        """Add a segment to the timeline.

        Args:
            start_frame: Start frame
            end_frame: End frame
            label: Segment label
            segment_type: Type of segment
            color: Optional color code
            data: Optional extra data

        Returns:
            The created segment
        """
        segment = TimelineSegment(
            id=self._generate_id('segment'),
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=self._frame_to_time(start_frame),
            end_time=self._frame_to_time(end_frame),
            label=label,
            segment_type=segment_type,
            color=color,
            data=data or {}
        )

        self._segments.append(segment)
        self._segments.sort(key=lambda s: s.start_frame)

        self._notify('segment_added', segment)
        return segment

    def get_segment_at_frame(self, frame: int) -> Optional[TimelineSegment]:
        """Get segment containing a frame.

        Args:
            frame: Frame number

        Returns:
            Segment containing frame, or None
        """
        for segment in self._segments:
            if segment.contains_frame(frame):
                return segment
        return None

    def get_segments_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> list[TimelineSegment]:
        """Get segments overlapping frame range.

        Args:
            start_frame: Start frame
            end_frame: End frame

        Returns:
            List of overlapping segments
        """
        return [
            s for s in self._segments
            if s.start_frame <= end_frame and s.end_frame >= start_frame
        ]

    def add_track(
        self,
        track_id: str,
        name: str,
        track_type: str,
        color: Optional[str] = None
    ) -> TimelineTrack:
        """Add a display track.

        Args:
            track_id: Track identifier
            name: Track name
            track_type: Type of track
            color: Optional track color

        Returns:
            The created track
        """
        track = TimelineTrack(
            id=track_id,
            name=name,
            track_type=track_type,
            color=color
        )
        self._tracks[track_id] = track
        return track

    def get_track(self, track_id: str) -> Optional[TimelineTrack]:
        """Get a track by ID."""
        return self._tracks.get(track_id)

    def iter_markers(self) -> Iterator[TimelineMarker]:
        """Iterate over all markers."""
        yield from self._markers

    def iter_events(self) -> Iterator[TimelineEvent]:
        """Iterate over all events."""
        yield from self._events

    def iter_segments(self) -> Iterator[TimelineSegment]:
        """Iterate over all segments."""
        yield from self._segments

    def get_visible_items(
        self,
        start_frame: int,
        end_frame: int,
        include_markers: bool = True,
        include_events: bool = True,
        include_segments: bool = True
    ) -> dict[str, list]:
        """Get all visible timeline items in range.

        Args:
            start_frame: Start frame
            end_frame: End frame
            include_markers: Include markers
            include_events: Include events
            include_segments: Include segments

        Returns:
            Dictionary with 'markers', 'events', 'segments' lists
        """
        result = {}

        if include_markers:
            result['markers'] = self.get_markers_in_range(start_frame, end_frame)

        if include_events:
            result['events'] = self.get_events_in_range(start_frame, end_frame)

        if include_segments:
            result['segments'] = self.get_segments_in_range(start_frame, end_frame)

        return result

    def frame_to_time(self, frame: int) -> float:
        """Convert frame number to timestamp.

        Args:
            frame: Frame number

        Returns:
            Timestamp in seconds
        """
        return self._frame_to_time(frame)

    def time_to_frame(self, time: float) -> int:
        """Convert timestamp to frame number.

        Args:
            time: Timestamp in seconds

        Returns:
            Frame number
        """
        return int(time * self._fps)

    def frame_to_percentage(self, frame: int) -> float:
        """Convert frame to percentage of total.

        Args:
            frame: Frame number

        Returns:
            Percentage (0.0 to 1.0)
        """
        if self._total_frames <= 0:
            return 0.0
        return frame / self._total_frames

    def percentage_to_frame(self, percentage: float) -> int:
        """Convert percentage to frame number.

        Args:
            percentage: Percentage (0.0 to 1.0)

        Returns:
            Frame number
        """
        return int(percentage * self._total_frames)

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            try:
                self._callbacks[event].remove(callback)
            except ValueError:
                pass

    def clear(self) -> None:
        """Clear all timeline data."""
        self._markers.clear()
        self._events.clear()
        self._segments.clear()
        self._marker_index.clear()
        self._tracks.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert timeline to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'total_frames': self._total_frames,
            'total_duration': self._total_duration,
            'fps': self._fps,
            'markers': [m.to_dict() for m in self._markers],
            'events': [e.to_dict() for e in self._events],
            'segments': [s.to_dict() for s in self._segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ReplayTimeline':
        """Create timeline from dictionary.

        Args:
            data: Dictionary data

        Returns:
            ReplayTimeline instance
        """
        timeline = cls(
            total_frames=data.get('total_frames', 0),
            total_duration=data.get('total_duration', 0.0),
            fps=data.get('fps', 60.0)
        )

        for m_data in data.get('markers', []):
            marker = TimelineMarker.from_dict(m_data)
            timeline._markers.append(marker)
            bisect.insort(timeline._marker_index, marker.frame)

        for e_data in data.get('events', []):
            event = TimelineEvent(
                id=e_data['id'],
                frame=e_data['frame'],
                timestamp=e_data['timestamp'],
                event_type=e_data['event_type'],
                name=e_data['name'],
                data=e_data.get('data', {}),
                duration=e_data.get('duration', 0.0)
            )
            timeline._events.append(event)

        for s_data in data.get('segments', []):
            segment = TimelineSegment(
                id=s_data['id'],
                start_frame=s_data['start_frame'],
                end_frame=s_data['end_frame'],
                start_time=s_data['start_time'],
                end_time=s_data['end_time'],
                label=s_data['label'],
                segment_type=s_data.get('segment_type', 'default'),
                color=s_data.get('color'),
                data=s_data.get('data', {})
            )
            timeline._segments.append(segment)

        return timeline

    def _frame_to_time(self, frame: int) -> float:
        """Internal frame to time conversion."""
        if self._fps <= 0:
            return 0.0
        return frame / self._fps

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    def _notify(self, event: str, data: Any) -> None:
        """Notify event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception:
                pass
