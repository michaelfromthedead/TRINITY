"""
Tests for replay_timeline.py - Timeline visualization with markers and events.
"""

import pytest

from engine.tooling.replay.replay_timeline import (
    ReplayTimeline,
    TimelineMarker,
    TimelineEvent,
    TimelineSegment,
    TimelineTrack,
    MarkerType,
)


class TestMarkerType:
    """Tests for MarkerType enum."""

    def test_marker_types_exist(self):
        """Test all marker types exist."""
        assert MarkerType.BOOKMARK
        assert MarkerType.EVENT
        assert MarkerType.KEYFRAME
        assert MarkerType.HIGHLIGHT
        assert MarkerType.ERROR
        assert MarkerType.CHECKPOINT
        assert MarkerType.CUSTOM


class TestTimelineMarker:
    """Tests for TimelineMarker dataclass."""

    def test_create_marker(self):
        """Test creating a marker."""
        marker = TimelineMarker(
            id="marker_1",
            frame=100,
            timestamp=1.67,
            marker_type=MarkerType.BOOKMARK,
            label="Important Moment",
            description="Something happened here"
        )
        assert marker.id == "marker_1"
        assert marker.frame == 100
        assert marker.marker_type == MarkerType.BOOKMARK

    def test_marker_with_color(self):
        """Test marker with color."""
        marker = TimelineMarker(
            id="marker_1",
            frame=50,
            timestamp=0.83,
            marker_type=MarkerType.HIGHLIGHT,
            label="Highlight",
            color="#FF0000"
        )
        assert marker.color == "#FF0000"

    def test_to_dict(self):
        """Test converting marker to dictionary."""
        marker = TimelineMarker(
            id="test",
            frame=0,
            timestamp=0.0,
            marker_type=MarkerType.BOOKMARK,
            label="Test"
        )
        data = marker.to_dict()

        assert data['id'] == "test"
        assert data['type'] == "BOOKMARK"

    def test_from_dict(self):
        """Test creating marker from dictionary."""
        data = {
            'id': 'marker_1',
            'frame': 100,
            'timestamp': 1.67,
            'type': 'EVENT',
            'label': 'Event Label',
            'description': 'Description',
            'data': {'extra': 'info'}
        }
        marker = TimelineMarker.from_dict(data)

        assert marker.id == 'marker_1'
        assert marker.frame == 100
        assert marker.marker_type == MarkerType.EVENT


class TestTimelineEvent:
    """Tests for TimelineEvent dataclass."""

    def test_create_event(self):
        """Test creating an event."""
        event = TimelineEvent(
            id="event_1",
            frame=200,
            timestamp=3.33,
            event_type="player_death",
            name="Player Died"
        )
        assert event.id == "event_1"
        assert event.event_type == "player_death"

    def test_event_with_duration(self):
        """Test event with duration."""
        event = TimelineEvent(
            id="event_1",
            frame=100,
            timestamp=1.67,
            event_type="ability",
            name="Special Attack",
            duration=2.5
        )
        assert event.duration == 2.5
        assert event.end_timestamp == 4.17

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = TimelineEvent(
            id="test",
            frame=0,
            timestamp=0.0,
            event_type="test_type",
            name="Test Event"
        )
        data = event.to_dict()

        assert data['id'] == "test"
        assert data['event_type'] == "test_type"


class TestTimelineSegment:
    """Tests for TimelineSegment dataclass."""

    def test_create_segment(self):
        """Test creating a segment."""
        segment = TimelineSegment(
            id="segment_1",
            start_frame=0,
            end_frame=100,
            start_time=0.0,
            end_time=1.67,
            label="Introduction"
        )
        assert segment.id == "segment_1"
        assert segment.duration == 1.67
        assert segment.frame_count == 100

    def test_contains_frame(self):
        """Test frame containment check."""
        segment = TimelineSegment(
            id="test",
            start_frame=50,
            end_frame=150,
            start_time=0.83,
            end_time=2.5,
            label="Test"
        )
        assert segment.contains_frame(50)
        assert segment.contains_frame(100)
        assert segment.contains_frame(150)
        assert not segment.contains_frame(49)
        assert not segment.contains_frame(151)

    def test_contains_time(self):
        """Test time containment check."""
        segment = TimelineSegment(
            id="test",
            start_frame=0,
            end_frame=100,
            start_time=1.0,
            end_time=3.0,
            label="Test"
        )
        assert segment.contains_time(1.0)
        assert segment.contains_time(2.0)
        assert segment.contains_time(3.0)
        assert not segment.contains_time(0.5)
        assert not segment.contains_time(3.5)

    def test_to_dict(self):
        """Test converting segment to dictionary."""
        segment = TimelineSegment(
            id="test",
            start_frame=0,
            end_frame=100,
            start_time=0.0,
            end_time=1.67,
            label="Test"
        )
        data = segment.to_dict()

        assert data['id'] == "test"
        assert data['start_frame'] == 0
        assert data['end_frame'] == 100


class TestReplayTimeline:
    """Tests for ReplayTimeline class."""

    def test_create_timeline(self):
        """Test creating a timeline."""
        timeline = ReplayTimeline(
            total_frames=6000,
            total_duration=100.0,
            fps=60.0
        )
        assert timeline.total_frames == 6000
        assert timeline.total_duration == 100.0
        assert timeline.fps == 60.0

    def test_add_marker(self):
        """Test adding a marker."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        marker = timeline.add_marker(
            frame=500,
            label="Midpoint",
            marker_type=MarkerType.BOOKMARK
        )

        assert marker is not None
        assert marker.frame == 500
        assert timeline.marker_count == 1

    def test_remove_marker(self):
        """Test removing a marker."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        marker = timeline.add_marker(frame=500, label="Test")
        assert timeline.marker_count == 1

        removed = timeline.remove_marker(marker.id)
        assert removed
        assert timeline.marker_count == 0

    def test_get_marker(self):
        """Test getting a marker by ID."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        added = timeline.add_marker(frame=100, label="Test")
        found = timeline.get_marker(added.id)

        assert found is not None
        assert found.id == added.id

    def test_get_markers_at_frame(self):
        """Test getting markers at specific frame."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="A")
        timeline.add_marker(frame=100, label="B")
        timeline.add_marker(frame=200, label="C")

        markers_at_100 = timeline.get_markers_at_frame(100)
        assert len(markers_at_100) == 2

    def test_get_markers_in_range(self):
        """Test getting markers in frame range."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=50, label="A")
        timeline.add_marker(frame=150, label="B")
        timeline.add_marker(frame=250, label="C")

        markers = timeline.get_markers_in_range(100, 200)
        assert len(markers) == 1
        assert markers[0].label == "B"

    def test_get_markers_by_type(self):
        """Test getting markers by type."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="A", marker_type=MarkerType.BOOKMARK)
        timeline.add_marker(frame=200, label="B", marker_type=MarkerType.EVENT)
        timeline.add_marker(frame=300, label="C", marker_type=MarkerType.BOOKMARK)

        bookmarks = timeline.get_markers_by_type(MarkerType.BOOKMARK)
        assert len(bookmarks) == 2

    def test_get_nearest_marker(self):
        """Test getting nearest marker."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="A")
        timeline.add_marker(frame=200, label="B")
        timeline.add_marker(frame=300, label="C")

        nearest = timeline.get_nearest_marker(150)
        assert nearest is not None
        # Should be either 100 or 200

    def test_get_nearest_marker_direction(self):
        """Test getting nearest marker with direction."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="A")
        timeline.add_marker(frame=200, label="B")
        timeline.add_marker(frame=300, label="C")

        prev = timeline.get_nearest_marker(250, direction=-1)
        assert prev is not None
        assert prev.frame == 200

        next_marker = timeline.get_nearest_marker(250, direction=1)
        assert next_marker is not None
        assert next_marker.frame == 300

    def test_add_event(self):
        """Test adding an event."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        event = timeline.add_event(
            frame=500,
            event_type="player_action",
            name="Jump"
        )

        assert event is not None
        assert timeline.event_count == 1

    def test_get_events_at_frame(self):
        """Test getting events at frame."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_event(frame=100, event_type="a", name="A")
        timeline.add_event(frame=100, event_type="b", name="B")

        events = timeline.get_events_at_frame(100)
        assert len(events) == 2

    def test_get_events_by_type(self):
        """Test getting events by type."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_event(frame=100, event_type="attack", name="A")
        timeline.add_event(frame=200, event_type="defend", name="B")
        timeline.add_event(frame=300, event_type="attack", name="C")

        attacks = timeline.get_events_by_type("attack")
        assert len(attacks) == 2

    def test_add_segment(self):
        """Test adding a segment."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        segment = timeline.add_segment(
            start_frame=0,
            end_frame=300,
            label="Phase 1"
        )

        assert segment is not None
        assert timeline.segment_count == 1

    def test_get_segment_at_frame(self):
        """Test getting segment at frame."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_segment(start_frame=0, end_frame=300, label="Phase 1")
        timeline.add_segment(start_frame=300, end_frame=600, label="Phase 2")

        segment = timeline.get_segment_at_frame(150)
        assert segment is not None
        assert segment.label == "Phase 1"

        segment = timeline.get_segment_at_frame(450)
        assert segment is not None
        assert segment.label == "Phase 2"

    def test_get_segments_in_range(self):
        """Test getting segments in range."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_segment(start_frame=0, end_frame=200, label="A")
        timeline.add_segment(start_frame=300, end_frame=500, label="B")
        timeline.add_segment(start_frame=600, end_frame=800, label="C")

        segments = timeline.get_segments_in_range(250, 650)
        assert len(segments) == 2  # B and C overlap

    def test_add_track(self):
        """Test adding a track."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        track = timeline.add_track(
            track_id="markers",
            name="Markers",
            track_type="markers",
            color="#FF0000"
        )

        assert track is not None
        assert track.id == "markers"

        found = timeline.get_track("markers")
        assert found is not None

    def test_frame_to_time(self):
        """Test frame to time conversion."""
        timeline = ReplayTimeline(total_frames=6000, total_duration=100.0, fps=60.0)

        time = timeline.frame_to_time(600)
        assert time == pytest.approx(10.0)

    def test_time_to_frame(self):
        """Test time to frame conversion."""
        timeline = ReplayTimeline(total_frames=6000, total_duration=100.0, fps=60.0)

        frame = timeline.time_to_frame(10.0)
        assert frame == 600

    def test_frame_to_percentage(self):
        """Test frame to percentage conversion."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        percentage = timeline.frame_to_percentage(500)
        assert percentage == 0.5

    def test_percentage_to_frame(self):
        """Test percentage to frame conversion."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        frame = timeline.percentage_to_frame(0.25)
        assert frame == 250

    def test_get_visible_items(self):
        """Test getting visible items in range."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="M1")
        timeline.add_marker(frame=500, label="M2")
        timeline.add_event(frame=200, event_type="e", name="E1")
        timeline.add_segment(start_frame=50, end_frame=150, label="S1")

        items = timeline.get_visible_items(0, 300)
        assert len(items['markers']) == 1
        assert len(items['events']) == 1
        assert len(items['segments']) == 1

    def test_event_callbacks(self):
        """Test event callbacks."""
        added_markers = []

        def on_marker_added(marker):
            added_markers.append(marker)

        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)
        timeline.on('marker_added', on_marker_added)

        timeline.add_marker(frame=100, label="Test")

        assert len(added_markers) == 1

    def test_clear(self):
        """Test clearing timeline."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="M")
        timeline.add_event(frame=200, event_type="e", name="E")
        timeline.add_segment(start_frame=0, end_frame=100, label="S")

        assert timeline.marker_count > 0

        timeline.clear()

        assert timeline.marker_count == 0
        assert timeline.event_count == 0
        assert timeline.segment_count == 0

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67, fps=60.0)

        timeline.add_marker(frame=100, label="M1")
        timeline.add_event(frame=200, event_type="e", name="E1")
        timeline.add_segment(start_frame=0, end_frame=300, label="S1")

        data = timeline.to_dict()
        restored = ReplayTimeline.from_dict(data)

        assert restored.total_frames == 1000
        assert restored.marker_count == 1
        assert restored.event_count == 1
        assert restored.segment_count == 1

    def test_iter_markers(self):
        """Test iterating over markers."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_marker(frame=100, label="A")
        timeline.add_marker(frame=200, label="B")

        markers = list(timeline.iter_markers())
        assert len(markers) == 2

    def test_iter_events(self):
        """Test iterating over events."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_event(frame=100, event_type="e", name="A")
        timeline.add_event(frame=200, event_type="e", name="B")

        events = list(timeline.iter_events())
        assert len(events) == 2

    def test_iter_segments(self):
        """Test iterating over segments."""
        timeline = ReplayTimeline(total_frames=1000, total_duration=16.67)

        timeline.add_segment(start_frame=0, end_frame=100, label="A")
        timeline.add_segment(start_frame=100, end_frame=200, label="B")

        segments = list(timeline.iter_segments())
        assert len(segments) == 2
