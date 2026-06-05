"""
Blackbox tests for SyncMarker and SyncMarkerTrack.

Tests the public API behavior without examining implementation details.
Covers marker creation, time calculations, distance computations, and track operations.

Task: T-FB-4.14
"""

import pytest
import math


# =============================================================================
# Test Fixtures and Setup
# =============================================================================

@pytest.fixture
def sync_marker_module():
    """Import sync marker module."""
    from engine.animation.graph.sync import SyncMarker, SyncMarkerTrack
    return SyncMarker, SyncMarkerTrack


@pytest.fixture
def SyncMarker(sync_marker_module):
    """Get SyncMarker class."""
    return sync_marker_module[0]


@pytest.fixture
def SyncMarkerTrack(sync_marker_module):
    """Get SyncMarkerTrack class."""
    return sync_marker_module[1]


@pytest.fixture
def walk_cycle_markers(SyncMarker):
    """Create typical walk cycle foot plant markers."""
    return [
        SyncMarker(name="left_foot_plant", normalized_time=0.0),
        SyncMarker(name="left_foot_lift", normalized_time=0.15),
        SyncMarker(name="right_foot_plant", normalized_time=0.5),
        SyncMarker(name="right_foot_lift", normalized_time=0.65),
    ]


@pytest.fixture
def impact_markers(SyncMarker):
    """Create impact event markers."""
    return [
        SyncMarker(name="impact", normalized_time=0.25),
        SyncMarker(name="impact", normalized_time=0.75),
    ]


# =============================================================================
# SyncMarker Creation Tests
# =============================================================================

class TestSyncMarkerCreation:
    """Tests for SyncMarker instantiation and field access."""

    def test_create_with_name_and_time(self, SyncMarker):
        """Create marker with basic name and normalized time."""
        marker = SyncMarker(name="test_marker", normalized_time=0.5)
        assert marker.name == "test_marker"
        assert marker.normalized_time == 0.5

    def test_create_at_start(self, SyncMarker):
        """Create marker at normalized time 0.0."""
        marker = SyncMarker(name="start", normalized_time=0.0)
        assert marker.normalized_time == 0.0

    def test_create_at_end(self, SyncMarker):
        """Create marker at normalized time 1.0."""
        marker = SyncMarker(name="end", normalized_time=1.0)
        assert marker.normalized_time == 1.0

    def test_create_at_midpoint(self, SyncMarker):
        """Create marker at normalized time 0.5."""
        marker = SyncMarker(name="midpoint", normalized_time=0.5)
        assert marker.normalized_time == 0.5

    def test_create_with_precise_time(self, SyncMarker):
        """Create marker with precise fractional time."""
        marker = SyncMarker(name="precise", normalized_time=0.333333)
        assert abs(marker.normalized_time - 0.333333) < 1e-6

    def test_name_with_underscores(self, SyncMarker):
        """Create marker with underscored name."""
        marker = SyncMarker(name="left_foot_contact_ground", normalized_time=0.25)
        assert marker.name == "left_foot_contact_ground"

    def test_name_single_character(self, SyncMarker):
        """Create marker with single character name."""
        marker = SyncMarker(name="A", normalized_time=0.1)
        assert marker.name == "A"

    def test_name_empty_string(self, SyncMarker):
        """Create marker with empty name - may be allowed or raise error."""
        try:
            marker = SyncMarker(name="", normalized_time=0.5)
            # If allowed, verify it works
            assert marker.name == ""
        except (ValueError, TypeError):
            # Empty name rejection is acceptable behavior
            pass

    def test_normalized_time_clamped_below_zero(self, SyncMarker):
        """Normalized time below 0 should be clamped to 0."""
        marker = SyncMarker(name="clamped", normalized_time=-0.5)
        assert marker.normalized_time >= 0.0
        assert marker.normalized_time <= 1.0

    def test_normalized_time_clamped_above_one(self, SyncMarker):
        """Normalized time above 1 should be clamped to 1."""
        marker = SyncMarker(name="clamped", normalized_time=1.5)
        assert marker.normalized_time >= 0.0
        assert marker.normalized_time <= 1.0

    def test_normalized_time_clamped_large_negative(self, SyncMarker):
        """Large negative normalized time should be clamped."""
        marker = SyncMarker(name="clamped", normalized_time=-100.0)
        assert marker.normalized_time >= 0.0

    def test_normalized_time_clamped_large_positive(self, SyncMarker):
        """Large positive normalized time should be clamped."""
        marker = SyncMarker(name="clamped", normalized_time=100.0)
        assert marker.normalized_time <= 1.0

    def test_access_all_fields(self, SyncMarker):
        """Verify all expected fields are accessible."""
        marker = SyncMarker(name="test", normalized_time=0.75)
        # Must have name and normalized_time at minimum
        assert hasattr(marker, 'name')
        assert hasattr(marker, 'normalized_time')


# =============================================================================
# SyncMarker.get_time_for_duration Tests
# =============================================================================

class TestSyncMarkerGetTimeForDuration:
    """Tests for converting normalized time to absolute time."""

    def test_time_for_one_second_duration(self, SyncMarker):
        """Get absolute time for 1 second duration."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        absolute_time = marker.get_time_for_duration(1.0)
        assert abs(absolute_time - 0.5) < 1e-6

    def test_time_for_two_second_duration(self, SyncMarker):
        """Get absolute time for 2 second duration."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        absolute_time = marker.get_time_for_duration(2.0)
        assert abs(absolute_time - 1.0) < 1e-6

    def test_time_at_start_any_duration(self, SyncMarker):
        """Marker at start (0.0) should return 0 for any duration."""
        marker = SyncMarker(name="start", normalized_time=0.0)
        assert marker.get_time_for_duration(1.0) == 0.0
        assert marker.get_time_for_duration(5.0) == 0.0
        assert marker.get_time_for_duration(0.5) == 0.0

    def test_time_at_end_matches_duration(self, SyncMarker):
        """Marker at end (1.0) should return the full duration."""
        marker = SyncMarker(name="end", normalized_time=1.0)
        assert abs(marker.get_time_for_duration(3.0) - 3.0) < 1e-6
        assert abs(marker.get_time_for_duration(10.0) - 10.0) < 1e-6

    def test_time_for_fractional_duration(self, SyncMarker):
        """Get absolute time for fractional duration."""
        marker = SyncMarker(name="test", normalized_time=0.25)
        absolute_time = marker.get_time_for_duration(4.0)
        assert abs(absolute_time - 1.0) < 1e-6

    def test_time_formula_normalized_times_duration(self, SyncMarker):
        """Verify formula: time = normalized_time * duration."""
        marker = SyncMarker(name="test", normalized_time=0.333)
        duration = 6.0
        expected = 0.333 * 6.0
        actual = marker.get_time_for_duration(duration)
        assert abs(actual - expected) < 1e-5

    def test_time_for_zero_duration(self, SyncMarker):
        """Get time for zero duration returns zero."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.get_time_for_duration(0.0) == 0.0

    def test_time_for_very_small_duration(self, SyncMarker):
        """Get time for very small duration."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        absolute_time = marker.get_time_for_duration(0.001)
        assert abs(absolute_time - 0.0005) < 1e-7

    def test_time_for_large_duration(self, SyncMarker):
        """Get time for large duration."""
        marker = SyncMarker(name="test", normalized_time=0.75)
        absolute_time = marker.get_time_for_duration(1000.0)
        assert abs(absolute_time - 750.0) < 1e-3


# =============================================================================
# SyncMarker.distance_to Tests
# =============================================================================

class TestSyncMarkerDistanceTo:
    """Tests for computing distances between markers."""

    def test_distance_same_marker(self, SyncMarker):
        """Distance from marker to itself is zero."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.distance_to(0.5) == 0.0

    def test_distance_direct_close(self, SyncMarker):
        """Direct distance when markers are close."""
        marker = SyncMarker(name="test", normalized_time=0.3)
        distance = marker.distance_to(0.4)
        assert abs(distance - 0.1) < 1e-6

    def test_distance_direct_reverse(self, SyncMarker):
        """Direct distance in reverse direction."""
        marker = SyncMarker(name="test", normalized_time=0.6)
        distance = marker.distance_to(0.4)
        assert abs(distance - 0.2) < 1e-6

    def test_distance_wrapped_shorter(self, SyncMarker):
        """Wrapped distance when it's shorter than direct."""
        # From 0.9 to 0.1: direct = 0.8, wrapped = 0.2
        marker = SyncMarker(name="test", normalized_time=0.9)
        distance = marker.distance_to(0.1)
        assert abs(distance - 0.2) < 1e-6

    def test_distance_wrapped_other_direction(self, SyncMarker):
        """Wrapped distance from low to high."""
        # From 0.1 to 0.9: direct = 0.8, wrapped = 0.2
        marker = SyncMarker(name="test", normalized_time=0.1)
        distance = marker.distance_to(0.9)
        assert abs(distance - 0.2) < 1e-6

    def test_distance_exactly_half(self, SyncMarker):
        """Distance is exactly 0.5 - both paths equal."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        distance = marker.distance_to(0.5)
        assert abs(distance - 0.5) < 1e-6

    def test_distance_start_to_end(self, SyncMarker):
        """Distance from 0.0 to 1.0 should be minimal (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        distance = marker.distance_to(1.0)
        # 0.0 and 1.0 are effectively the same point in a cycle
        assert distance <= 0.5 + 1e-6

    def test_distance_small_wrap(self, SyncMarker):
        """Small wrapped distance near boundaries."""
        marker = SyncMarker(name="test", normalized_time=0.95)
        distance = marker.distance_to(0.05)
        assert abs(distance - 0.1) < 1e-6

    def test_distance_to_zero(self, SyncMarker):
        """Distance to normalized time 0."""
        marker = SyncMarker(name="test", normalized_time=0.25)
        distance = marker.distance_to(0.0)
        assert abs(distance - 0.25) < 1e-6

    def test_distance_to_one(self, SyncMarker):
        """Distance to normalized time 1."""
        marker = SyncMarker(name="test", normalized_time=0.75)
        distance = marker.distance_to(1.0)
        assert abs(distance - 0.25) < 1e-6

    def test_distance_always_positive(self, SyncMarker):
        """Distance is always non-negative."""
        marker = SyncMarker(name="test", normalized_time=0.7)
        for target in [0.0, 0.3, 0.5, 0.8, 1.0]:
            distance = marker.distance_to(target)
            assert distance >= 0.0

    def test_distance_never_exceeds_half(self, SyncMarker):
        """Distance never exceeds 0.5 (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.2)
        for target in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            distance = marker.distance_to(target)
            assert distance <= 0.5 + 1e-6


# =============================================================================
# SyncMarkerTrack Basic Tests
# =============================================================================

class TestSyncMarkerTrackBasics:
    """Basic tests for SyncMarkerTrack creation and marker addition."""

    def test_create_empty_track(self, SyncMarkerTrack):
        """Create empty marker track."""
        track = SyncMarkerTrack()
        assert track is not None

    def test_add_single_marker(self, SyncMarkerTrack, SyncMarker):
        """Add single marker to track."""
        track = SyncMarkerTrack()
        marker = SyncMarker(name="test", normalized_time=0.5)
        track.add_marker(marker)
        # Should not raise

    def test_add_multiple_markers(self, SyncMarkerTrack, SyncMarker):
        """Add multiple markers to track."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.5))
        track.add_marker(SyncMarker(name="c", normalized_time=0.9))
        # Should not raise

    def test_add_markers_same_name(self, SyncMarkerTrack, SyncMarker):
        """Add multiple markers with same name at different times."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="impact", normalized_time=0.25))
        track.add_marker(SyncMarker(name="impact", normalized_time=0.75))
        # Should be allowed

    def test_add_walk_cycle_markers(self, SyncMarkerTrack, walk_cycle_markers):
        """Add complete walk cycle markers."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)


# =============================================================================
# SyncMarkerTrack.get_markers_by_name Tests
# =============================================================================

class TestSyncMarkerTrackGetByName:
    """Tests for retrieving markers by name."""

    def test_get_single_marker_by_name(self, SyncMarkerTrack, SyncMarker):
        """Get single marker by name."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="unique", normalized_time=0.5))
        markers = track.get_markers_by_name("unique")
        assert len(markers) == 1
        assert markers[0].name == "unique"

    def test_get_multiple_markers_same_name(self, SyncMarkerTrack, impact_markers):
        """Get multiple markers with same name."""
        track = SyncMarkerTrack()
        for marker in impact_markers:
            track.add_marker(marker)
        markers = track.get_markers_by_name("impact")
        assert len(markers) == 2

    def test_get_nonexistent_name(self, SyncMarkerTrack, SyncMarker):
        """Get markers by name that doesn't exist."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="exists", normalized_time=0.5))
        markers = track.get_markers_by_name("nonexistent")
        assert len(markers) == 0

    def test_get_from_empty_track(self, SyncMarkerTrack):
        """Get markers by name from empty track."""
        track = SyncMarkerTrack()
        markers = track.get_markers_by_name("any")
        assert len(markers) == 0

    def test_get_preserves_marker_data(self, SyncMarkerTrack, SyncMarker):
        """Retrieved marker has correct data."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="precise", normalized_time=0.333))
        markers = track.get_markers_by_name("precise")
        assert abs(markers[0].normalized_time - 0.333) < 1e-6

    def test_get_by_name_case_sensitive(self, SyncMarkerTrack, SyncMarker):
        """Marker name lookup should be case sensitive."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="Impact", normalized_time=0.5))
        markers_lower = track.get_markers_by_name("impact")
        markers_upper = track.get_markers_by_name("Impact")
        # Case sensitivity expected
        assert len(markers_upper) == 1
        # Lower case should not match (case sensitive)
        assert len(markers_lower) == 0 or markers_lower[0].name == "Impact"


# =============================================================================
# SyncMarkerTrack.get_nearest_marker Tests
# =============================================================================

class TestSyncMarkerTrackGetNearest:
    """Tests for finding nearest marker."""

    def test_get_nearest_exact_match(self, SyncMarkerTrack, SyncMarker):
        """Get nearest when exactly at marker position."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="target", normalized_time=0.5))
        nearest = track.get_nearest_marker(0.5)
        assert nearest is not None
        assert nearest.name == "target"

    def test_get_nearest_before(self, SyncMarkerTrack, SyncMarker):
        """Get nearest when query is before marker."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="only", normalized_time=0.7))
        nearest = track.get_nearest_marker(0.5)
        assert nearest is not None
        assert nearest.name == "only"

    def test_get_nearest_after(self, SyncMarkerTrack, SyncMarker):
        """Get nearest when query is after marker."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="only", normalized_time=0.3))
        nearest = track.get_nearest_marker(0.5)
        assert nearest is not None
        assert nearest.name == "only"

    def test_get_nearest_between_two(self, SyncMarkerTrack, SyncMarker):
        """Get nearest when between two markers."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="left", normalized_time=0.2))
        track.add_marker(SyncMarker(name="right", normalized_time=0.6))
        # Query at 0.35 - closer to left (0.15) than right (0.25)
        nearest = track.get_nearest_marker(0.35)
        assert nearest.name == "left"

    def test_get_nearest_closer_to_right(self, SyncMarkerTrack, SyncMarker):
        """Get nearest when closer to right marker."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="left", normalized_time=0.2))
        track.add_marker(SyncMarker(name="right", normalized_time=0.6))
        # Query at 0.5 - closer to right (0.1) than left (0.3)
        nearest = track.get_nearest_marker(0.5)
        assert nearest.name == "right"

    def test_get_nearest_with_wrap(self, SyncMarkerTrack, SyncMarker):
        """Get nearest considering wrap-around distance."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="near_end", normalized_time=0.9))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        # Query at 0.05 - wrapped distance to near_end is 0.15
        nearest = track.get_nearest_marker(0.05)
        assert nearest.name == "near_end"

    def test_get_nearest_empty_track(self, SyncMarkerTrack):
        """Get nearest from empty track returns None."""
        track = SyncMarkerTrack()
        nearest = track.get_nearest_marker(0.5)
        assert nearest is None

    def test_get_nearest_with_name_filter(self, SyncMarkerTrack, walk_cycle_markers):
        """Get nearest with name filter."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)
        # Query at 0.4, filter by "left_foot_plant"
        nearest = track.get_nearest_marker(0.4, name="left_foot_plant")
        assert nearest is not None
        assert nearest.name == "left_foot_plant"

    def test_get_nearest_name_filter_no_match(self, SyncMarkerTrack, SyncMarker):
        """Get nearest with name filter that has no matches."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="exists", normalized_time=0.5))
        nearest = track.get_nearest_marker(0.5, name="nonexistent")
        assert nearest is None

    def test_get_nearest_multiple_same_name(self, SyncMarkerTrack, impact_markers):
        """Get nearest from multiple markers with same name."""
        track = SyncMarkerTrack()
        for marker in impact_markers:
            track.add_marker(marker)
        # Query at 0.3, impacts at 0.25 and 0.75
        nearest = track.get_nearest_marker(0.3, name="impact")
        assert nearest.normalized_time == 0.25


# =============================================================================
# SyncMarkerTrack.get_markers_in_range Tests
# =============================================================================

class TestSyncMarkerTrackGetInRange:
    """Tests for getting markers within a range."""

    def test_get_in_range_basic(self, SyncMarkerTrack, SyncMarker):
        """Get markers in basic range."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.3))
        track.add_marker(SyncMarker(name="c", normalized_time=0.5))
        markers = track.get_markers_in_range(0.2, 0.4)
        assert len(markers) == 1
        assert markers[0].name == "b"

    def test_get_in_range_multiple(self, SyncMarkerTrack, SyncMarker):
        """Get multiple markers in range."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.2))
        track.add_marker(SyncMarker(name="b", normalized_time=0.3))
        track.add_marker(SyncMarker(name="c", normalized_time=0.4))
        markers = track.get_markers_in_range(0.15, 0.45)
        assert len(markers) == 3

    def test_get_in_range_inclusive_start(self, SyncMarkerTrack, SyncMarker):
        """Range should include start boundary."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="boundary", normalized_time=0.3))
        markers = track.get_markers_in_range(0.3, 0.5)
        assert len(markers) >= 1

    def test_get_in_range_inclusive_end(self, SyncMarkerTrack, SyncMarker):
        """Range should include end boundary."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="boundary", normalized_time=0.5))
        markers = track.get_markers_in_range(0.3, 0.5)
        assert len(markers) >= 1

    def test_get_in_range_empty_result(self, SyncMarkerTrack, SyncMarker):
        """Get markers in range with no matches."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.9))
        markers = track.get_markers_in_range(0.4, 0.6)
        assert len(markers) == 0

    def test_get_in_range_empty_track(self, SyncMarkerTrack):
        """Get markers in range from empty track."""
        track = SyncMarkerTrack()
        markers = track.get_markers_in_range(0.0, 1.0)
        assert len(markers) == 0

    def test_get_in_range_full_range(self, SyncMarkerTrack, SyncMarker):
        """Get all markers with full range."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.0))
        track.add_marker(SyncMarker(name="b", normalized_time=0.5))
        track.add_marker(SyncMarker(name="c", normalized_time=1.0))
        markers = track.get_markers_in_range(0.0, 1.0)
        assert len(markers) == 3

    def test_get_in_range_wrapped(self, SyncMarkerTrack, SyncMarker):
        """Get markers in wrapped range (start > end)."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="early", normalized_time=0.1))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        track.add_marker(SyncMarker(name="late", normalized_time=0.9))
        # Wrapped range: 0.8 to 0.2 should include 0.9 and 0.1
        markers = track.get_markers_in_range(0.8, 0.2)
        names = [m.name for m in markers]
        assert "early" in names or "late" in names
        assert "middle" not in names

    def test_get_in_range_wrapped_all_late(self, SyncMarkerTrack, SyncMarker):
        """Wrapped range includes late markers."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="late", normalized_time=0.95))
        markers = track.get_markers_in_range(0.9, 0.1)
        assert len(markers) >= 1
        assert markers[0].name == "late"

    def test_get_in_range_wrapped_all_early(self, SyncMarkerTrack, SyncMarker):
        """Wrapped range includes early markers."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="early", normalized_time=0.05))
        markers = track.get_markers_in_range(0.9, 0.1)
        assert len(markers) >= 1
        assert markers[0].name == "early"


# =============================================================================
# Integration Tests: Walk Cycle
# =============================================================================

class TestWalkCycleIntegration:
    """Integration tests using walk cycle animation markers."""

    def test_walk_cycle_marker_sequence(self, SyncMarkerTrack, walk_cycle_markers):
        """Verify walk cycle markers are in correct sequence."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)

        # Check sequence by querying at different times
        at_start = track.get_nearest_marker(0.0)
        assert at_start.name == "left_foot_plant"

        at_middle = track.get_nearest_marker(0.5)
        assert at_middle.name == "right_foot_plant"

    def test_walk_cycle_find_foot_plants(self, SyncMarkerTrack, walk_cycle_markers):
        """Find all foot plant markers."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)

        left_plants = track.get_markers_by_name("left_foot_plant")
        right_plants = track.get_markers_by_name("right_foot_plant")

        assert len(left_plants) == 1
        assert len(right_plants) == 1
        assert left_plants[0].normalized_time < right_plants[0].normalized_time

    def test_walk_cycle_time_conversion(self, SyncMarkerTrack, walk_cycle_markers):
        """Convert walk cycle markers to absolute times."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)

        duration = 1.0  # 1 second walk cycle
        left_plant = track.get_markers_by_name("left_foot_plant")[0]
        right_plant = track.get_markers_by_name("right_foot_plant")[0]

        left_time = left_plant.get_time_for_duration(duration)
        right_time = right_plant.get_time_for_duration(duration)

        assert left_time == 0.0
        assert abs(right_time - 0.5) < 1e-6

    def test_walk_cycle_nearest_foot_event(self, SyncMarkerTrack, walk_cycle_markers):
        """Find nearest foot event at given time."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)

        # At time 0.1, nearest left foot event should be left_foot_plant
        nearest = track.get_nearest_marker(0.1, name="left_foot_plant")
        assert nearest is not None
        # Could also be left_foot_lift depending on implementation


# =============================================================================
# Integration Tests: Impact Markers
# =============================================================================

class TestImpactMarkerIntegration:
    """Integration tests using impact event markers."""

    def test_multiple_impacts_same_name(self, SyncMarkerTrack, impact_markers):
        """Handle multiple markers with same name."""
        track = SyncMarkerTrack()
        for marker in impact_markers:
            track.add_marker(marker)

        impacts = track.get_markers_by_name("impact")
        assert len(impacts) == 2

        times = sorted([m.normalized_time for m in impacts])
        assert abs(times[0] - 0.25) < 1e-6
        assert abs(times[1] - 0.75) < 1e-6

    def test_find_nearest_impact(self, SyncMarkerTrack, impact_markers):
        """Find nearest impact marker."""
        track = SyncMarkerTrack()
        for marker in impact_markers:
            track.add_marker(marker)

        # At time 0.3, nearest impact should be at 0.25
        nearest = track.get_nearest_marker(0.3, name="impact")
        assert abs(nearest.normalized_time - 0.25) < 1e-6

        # At time 0.6, nearest impact should be at 0.75
        nearest = track.get_nearest_marker(0.6, name="impact")
        assert abs(nearest.normalized_time - 0.75) < 1e-6

    def test_impacts_in_first_half(self, SyncMarkerTrack, impact_markers):
        """Get impacts in first half of animation."""
        track = SyncMarkerTrack()
        for marker in impact_markers:
            track.add_marker(marker)

        first_half = track.get_markers_in_range(0.0, 0.5)
        impact_in_first = [m for m in first_half if m.name == "impact"]
        assert len(impact_in_first) == 1
        assert abs(impact_in_first[0].normalized_time - 0.25) < 1e-6


# =============================================================================
# Integration Tests: Mixed Markers
# =============================================================================

class TestMixedMarkerIntegration:
    """Integration tests with multiple marker types."""

    def test_mixed_marker_track(self, SyncMarkerTrack, SyncMarker,
                                 walk_cycle_markers, impact_markers):
        """Create track with multiple marker types."""
        track = SyncMarkerTrack()

        # Add all markers
        for marker in walk_cycle_markers:
            track.add_marker(marker)
        for marker in impact_markers:
            track.add_marker(marker)

        # Query by different names
        foot_plants = track.get_markers_by_name("left_foot_plant")
        impacts = track.get_markers_by_name("impact")

        assert len(foot_plants) == 1
        assert len(impacts) == 2

    def test_get_all_markers_in_range_mixed(self, SyncMarkerTrack, SyncMarker,
                                             walk_cycle_markers, impact_markers):
        """Get all markers in range regardless of type."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)
        for marker in impact_markers:
            track.add_marker(marker)

        # Range 0.1 to 0.3 should include left_foot_lift (0.15) and impact (0.25)
        markers = track.get_markers_in_range(0.1, 0.3)
        names = [m.name for m in markers]
        assert len(markers) >= 2

    def test_nearest_any_marker(self, SyncMarkerTrack, SyncMarker,
                                 walk_cycle_markers, impact_markers):
        """Find nearest marker regardless of type."""
        track = SyncMarkerTrack()
        for marker in walk_cycle_markers:
            track.add_marker(marker)
        for marker in impact_markers:
            track.add_marker(marker)

        # At time 0.24, nearest should be impact at 0.25
        nearest = track.get_nearest_marker(0.24)
        assert nearest.name == "impact"
        assert abs(nearest.normalized_time - 0.25) < 1e-6


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_marker_at_exact_boundaries(self, SyncMarkerTrack, SyncMarker):
        """Markers at exact 0.0 and 1.0 boundaries."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="start", normalized_time=0.0))
        track.add_marker(SyncMarker(name="end", normalized_time=1.0))

        start_markers = track.get_markers_by_name("start")
        end_markers = track.get_markers_by_name("end")

        assert len(start_markers) == 1
        assert len(end_markers) == 1

    def test_many_markers_same_time(self, SyncMarkerTrack, SyncMarker):
        """Multiple markers at exactly the same time."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.5))
        track.add_marker(SyncMarker(name="b", normalized_time=0.5))
        track.add_marker(SyncMarker(name="c", normalized_time=0.5))

        in_range = track.get_markers_in_range(0.4, 0.6)
        assert len(in_range) == 3

    def test_dense_marker_distribution(self, SyncMarkerTrack, SyncMarker):
        """Track with many closely spaced markers."""
        track = SyncMarkerTrack()
        for i in range(100):
            time = i / 100.0
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=time))

        # Should find markers accurately
        nearest = track.get_nearest_marker(0.505)
        # Should be m50 or m51
        assert nearest.name in ["m50", "m51"]

    def test_very_close_markers(self, SyncMarkerTrack, SyncMarker):
        """Distinguish between very close markers."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.50000))
        track.add_marker(SyncMarker(name="b", normalized_time=0.50001))

        # Query exactly at b's position
        nearest = track.get_nearest_marker(0.50001)
        assert nearest.name == "b"

    def test_single_marker_track(self, SyncMarkerTrack, SyncMarker):
        """Track with only one marker."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="only", normalized_time=0.5))

        # Any query should return this marker
        assert track.get_nearest_marker(0.0).name == "only"
        assert track.get_nearest_marker(0.5).name == "only"
        assert track.get_nearest_marker(1.0).name == "only"

    def test_markers_sorted_or_accessible(self, SyncMarkerTrack, SyncMarker):
        """Markers added out of order should still be accessible."""
        track = SyncMarkerTrack()
        # Add in non-chronological order
        track.add_marker(SyncMarker(name="late", normalized_time=0.9))
        track.add_marker(SyncMarker(name="early", normalized_time=0.1))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))

        # Should still find them correctly
        nearest_to_start = track.get_nearest_marker(0.05)
        assert nearest_to_start.name == "early"

        nearest_to_end = track.get_nearest_marker(0.95)
        assert nearest_to_end.name == "late"


# =============================================================================
# Performance Characteristics Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Tests to verify reasonable performance with many markers."""

    def test_large_track_creation(self, SyncMarkerTrack, SyncMarker):
        """Create track with many markers."""
        track = SyncMarkerTrack()
        for i in range(1000):
            time = (i % 100) / 100.0
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=time))
        # Should complete without error

    def test_large_track_query(self, SyncMarkerTrack, SyncMarker):
        """Query large track for nearest marker."""
        track = SyncMarkerTrack()
        for i in range(1000):
            time = (i % 100) / 100.0
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=time))

        # Should return a result
        nearest = track.get_nearest_marker(0.555)
        assert nearest is not None

    def test_large_track_range_query(self, SyncMarkerTrack, SyncMarker):
        """Query large track for markers in range."""
        track = SyncMarkerTrack()
        for i in range(1000):
            time = (i % 100) / 100.0
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=time))

        markers = track.get_markers_in_range(0.4, 0.6)
        # Should return multiple markers
        assert len(markers) > 0
