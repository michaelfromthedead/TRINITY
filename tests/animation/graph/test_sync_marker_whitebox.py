"""WHITEBOX tests for engine/animation/graph/sync.py SyncMarker and SyncMarkerTrack.

Tests for T-FB-4.14 (SyncMarker and MarkerTrack).

WHITEBOX coverage plan:
  [SyncMarker dataclass fields]
    Path A1:  name field stores string correctly
    Path A2:  normalized_time field stores float
    Path A3:  bone_index default is None
    Path A4:  bone_index accepts int values
    Path A5:  metadata default is empty dict
    Path A6:  metadata accepts custom dict

  [SyncMarker.__post_init__ clamping]
    Path B1:  normalized_time = 0.0 stays at 0.0
    Path B2:  normalized_time = 1.0 stays at 1.0
    Path B3:  normalized_time = 0.5 stays at 0.5
    Path B4:  negative normalized_time clamped to 0.0
    Path B5:  large negative clamped to 0.0
    Path B6:  normalized_time > 1.0 clamped to 1.0
    Path B7:  normalized_time = 2.5 clamped to 1.0
    Path B8:  very small positive value preserved

  [SyncMarker.get_time_for_duration]
    Path C1:  duration = 1.0 returns normalized_time
    Path C2:  duration = 2.0 doubles the time
    Path C3:  duration = 0.5 halves the time
    Path C4:  duration = 0.0 returns 0.0
    Path C5:  normalized_time = 0.0 returns 0.0 for any duration
    Path C6:  normalized_time = 1.0 returns duration
    Path C7:  fractional normalized_time with fractional duration

  [SyncMarker.distance_to direct distance]
    Path D1:  same normalized_time returns 0.0
    Path D2:  distance from 0.0 to 0.5 = 0.5
    Path D3:  distance from 0.5 to 1.0 = 0.5
    Path D4:  distance from 0.2 to 0.8 = 0.4 (wrapped is shorter)
    Path D5:  distance from 0.3 to 0.4 = 0.1

  [SyncMarker.distance_to wrap-around]
    Path E1:  distance from 0.9 to 0.1 = 0.2 (wrapped)
    Path E2:  distance from 0.1 to 0.9 = 0.2 (wrapped)
    Path E3:  distance from 0.95 to 0.05 = 0.1 (wrapped)
    Path E4:  distance from 0.0 to 1.0 = 0.0 (endpoints)
    Path E5:  distance from 0.8 to 0.2 = 0.4 (wrapped)
    Path E6:  distance from 0.75 to 0.25 = 0.5 (tie case)

  [SyncMarkerTrack dataclass fields]
    Path F1:  markers default is empty list
    Path F2:  markers initialized with list preserves order
    Path F3:  empty track has len 0

  [SyncMarkerTrack.add_marker basic]
    Path G1:  add single marker to empty track
    Path G2:  add marker increases length
    Path G3:  multiple adds accumulate
    Path G4:  same normalized_time allowed

  [SyncMarkerTrack.add_marker sorting]
    Path H1:  markers sorted by normalized_time
    Path H2:  add at end maintains sort
    Path H3:  add at beginning maintains sort
    Path H4:  add in middle maintains sort
    Path H5:  add multiple out of order results in sorted list

  [SyncMarkerTrack.get_markers_by_name]
    Path I1:  empty track returns empty list
    Path I2:  no matching name returns empty list
    Path I3:  single match returns list of one
    Path I4:  multiple matches returns all
    Path I5:  case sensitive matching

  [SyncMarkerTrack.get_nearest_marker without filter]
    Path J1:  empty track returns None
    Path J2:  single marker returns that marker
    Path J3:  exact match returns that marker
    Path J4:  nearest by direct distance
    Path J5:  nearest considering wrap-around
    Path J6:  multiple equidistant returns one (deterministic)

  [SyncMarkerTrack.get_nearest_marker with name filter]
    Path K1:  no markers with name returns None
    Path K2:  filters to only matching names
    Path K3:  nearest among filtered set
    Path K4:  filter excludes closer non-matching markers

  [SyncMarkerTrack.get_markers_in_range normal range]
    Path L1:  empty track returns empty list
    Path L2:  no markers in range returns empty
    Path L3:  single marker in range
    Path L4:  multiple markers in range
    Path L5:  marker at start boundary included
    Path L6:  marker at end boundary included
    Path L7:  markers outside range excluded

  [SyncMarkerTrack.get_markers_in_range wrapped range]
    Path M1:  wrapped range 0.9 to 0.1 includes 0.95
    Path M2:  wrapped range 0.9 to 0.1 includes 0.05
    Path M3:  wrapped range excludes middle (e.g., 0.5)
    Path M4:  wrapped range at boundaries
    Path M5:  wrapped range includes 0.0 endpoint
    Path M6:  wrapped range includes 1.0 equivalent markers
"""

import math
import pytest
from dataclasses import fields

from engine.animation.graph.sync import SyncMarker, SyncMarkerTrack


# =============================================================================
# SyncMarker DATACLASS FIELDS
# =============================================================================


class TestSyncMarkerFields:
    """Tests for SyncMarker dataclass field access and defaults."""

    def test_A1_name_field_stores_string(self):
        """Path A1: name field stores string correctly."""
        marker = SyncMarker(name="foot_plant", normalized_time=0.5)
        assert marker.name == "foot_plant"

    def test_A1_name_field_empty_string(self):
        """Path A1: name field accepts empty string."""
        marker = SyncMarker(name="", normalized_time=0.5)
        assert marker.name == ""

    def test_A1_name_field_unicode(self):
        """Path A1: name field accepts unicode strings."""
        marker = SyncMarker(name="sync_α_point", normalized_time=0.5)
        assert marker.name == "sync_α_point"

    def test_A2_normalized_time_stores_float(self):
        """Path A2: normalized_time field stores float."""
        marker = SyncMarker(name="test", normalized_time=0.75)
        assert marker.normalized_time == 0.75
        assert isinstance(marker.normalized_time, float)

    def test_A2_normalized_time_accepts_int(self):
        """Path A2: normalized_time accepts int and converts."""
        marker = SyncMarker(name="test", normalized_time=1)
        # After clamping, should be 1.0
        assert marker.normalized_time == 1.0

    def test_A3_bone_index_default_none(self):
        """Path A3: bone_index default is None."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.bone_index is None

    def test_A4_bone_index_accepts_int(self):
        """Path A4: bone_index accepts int values."""
        marker = SyncMarker(name="test", normalized_time=0.5, bone_index=5)
        assert marker.bone_index == 5

    def test_A4_bone_index_zero(self):
        """Path A4: bone_index accepts zero."""
        marker = SyncMarker(name="test", normalized_time=0.5, bone_index=0)
        assert marker.bone_index == 0

    def test_A4_bone_index_negative(self):
        """Path A4: bone_index accepts negative (no validation)."""
        marker = SyncMarker(name="test", normalized_time=0.5, bone_index=-1)
        assert marker.bone_index == -1

    def test_A5_metadata_default_empty_dict(self):
        """Path A5: metadata default is empty dict."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.metadata == {}
        assert isinstance(marker.metadata, dict)

    def test_A5_metadata_default_is_unique_per_instance(self):
        """Path A5: metadata default factory creates unique dicts."""
        marker1 = SyncMarker(name="test", normalized_time=0.5)
        marker2 = SyncMarker(name="test", normalized_time=0.5)
        marker1.metadata["key"] = "value"
        assert "key" not in marker2.metadata

    def test_A6_metadata_accepts_custom_dict(self):
        """Path A6: metadata accepts custom dict."""
        meta = {"impact_force": 1.5, "sound": "footstep"}
        marker = SyncMarker(name="test", normalized_time=0.5, metadata=meta)
        assert marker.metadata == meta
        assert marker.metadata["impact_force"] == 1.5


# =============================================================================
# SyncMarker.__post_init__ CLAMPING
# =============================================================================


class TestSyncMarkerPostInit:
    """Tests for SyncMarker.__post_init__ clamping behavior."""

    def test_B1_normalized_time_zero_unchanged(self):
        """Path B1: normalized_time = 0.0 stays at 0.0."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        assert marker.normalized_time == 0.0

    def test_B2_normalized_time_one_unchanged(self):
        """Path B2: normalized_time = 1.0 stays at 1.0."""
        marker = SyncMarker(name="test", normalized_time=1.0)
        assert marker.normalized_time == 1.0

    def test_B3_normalized_time_half_unchanged(self):
        """Path B3: normalized_time = 0.5 stays at 0.5."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.normalized_time == 0.5

    def test_B4_negative_clamped_to_zero(self):
        """Path B4: negative normalized_time clamped to 0.0."""
        marker = SyncMarker(name="test", normalized_time=-0.1)
        assert marker.normalized_time == 0.0

    def test_B5_large_negative_clamped_to_zero(self):
        """Path B5: large negative clamped to 0.0."""
        marker = SyncMarker(name="test", normalized_time=-100.0)
        assert marker.normalized_time == 0.0

    def test_B6_above_one_clamped(self):
        """Path B6: normalized_time > 1.0 clamped to 1.0."""
        marker = SyncMarker(name="test", normalized_time=1.1)
        assert marker.normalized_time == 1.0

    def test_B7_large_positive_clamped(self):
        """Path B7: normalized_time = 2.5 clamped to 1.0."""
        marker = SyncMarker(name="test", normalized_time=2.5)
        assert marker.normalized_time == 1.0

    def test_B8_very_small_positive_preserved(self):
        """Path B8: very small positive value preserved."""
        marker = SyncMarker(name="test", normalized_time=0.001)
        assert marker.normalized_time == 0.001

    def test_B8_epsilon_preserved(self):
        """Path B8: epsilon-sized values preserved."""
        marker = SyncMarker(name="test", normalized_time=1e-10)
        assert marker.normalized_time == 1e-10

    def test_B8_near_one_preserved(self):
        """Path B8: values near 1.0 preserved."""
        marker = SyncMarker(name="test", normalized_time=0.999999)
        assert marker.normalized_time == 0.999999


# =============================================================================
# SyncMarker.get_time_for_duration
# =============================================================================


class TestSyncMarkerGetTimeForDuration:
    """Tests for SyncMarker.get_time_for_duration method."""

    def test_C1_duration_one_returns_normalized_time(self):
        """Path C1: duration = 1.0 returns normalized_time."""
        marker = SyncMarker(name="test", normalized_time=0.25)
        assert marker.get_time_for_duration(1.0) == 0.25

    def test_C2_duration_two_doubles_time(self):
        """Path C2: duration = 2.0 doubles the time."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.get_time_for_duration(2.0) == 1.0

    def test_C3_duration_half_halves_time(self):
        """Path C3: duration = 0.5 halves the time."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.get_time_for_duration(0.5) == 0.25

    def test_C4_duration_zero_returns_zero(self):
        """Path C4: duration = 0.0 returns 0.0."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.get_time_for_duration(0.0) == 0.0

    def test_C5_normalized_time_zero_returns_zero(self):
        """Path C5: normalized_time = 0.0 returns 0.0 for any duration."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        assert marker.get_time_for_duration(100.0) == 0.0

    def test_C6_normalized_time_one_returns_duration(self):
        """Path C6: normalized_time = 1.0 returns duration."""
        marker = SyncMarker(name="test", normalized_time=1.0)
        assert marker.get_time_for_duration(3.0) == 3.0

    def test_C7_fractional_values(self):
        """Path C7: fractional normalized_time with fractional duration."""
        marker = SyncMarker(name="test", normalized_time=0.25)
        result = marker.get_time_for_duration(0.8)
        assert math.isclose(result, 0.2)

    def test_C7_arbitrary_calculation(self):
        """Path C7: arbitrary normalized_time and duration."""
        marker = SyncMarker(name="test", normalized_time=0.33)
        result = marker.get_time_for_duration(3.0)
        assert math.isclose(result, 0.99)


# =============================================================================
# SyncMarker.distance_to DIRECT DISTANCE
# =============================================================================


class TestSyncMarkerDistanceToDirect:
    """Tests for SyncMarker.distance_to method - direct distance cases."""

    def test_D1_same_time_returns_zero(self):
        """Path D1: same normalized_time returns 0.0."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.distance_to(0.5) == 0.0

    def test_D1_both_zero(self):
        """Path D1: distance from 0.0 to 0.0 is 0.0."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        assert marker.distance_to(0.0) == 0.0

    def test_D2_distance_zero_to_half(self):
        """Path D2: distance from 0.0 to 0.5 = 0.5."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        assert marker.distance_to(0.5) == 0.5

    def test_D3_distance_half_to_one(self):
        """Path D3: distance from 0.5 to 1.0 = 0.5."""
        marker = SyncMarker(name="test", normalized_time=0.5)
        assert marker.distance_to(1.0) == 0.5

    def test_D4_distance_wrapped_shorter(self):
        """Path D4: distance from 0.2 to 0.8 = 0.4 (wrapped is shorter)."""
        marker = SyncMarker(name="test", normalized_time=0.2)
        # Direct = 0.6, wrapped = 0.4, min = 0.4
        assert math.isclose(marker.distance_to(0.8), 0.4)

    def test_D5_direct_distance_small(self):
        """Path D5: distance from 0.3 to 0.4 = 0.1."""
        marker = SyncMarker(name="test", normalized_time=0.3)
        assert math.isclose(marker.distance_to(0.4), 0.1)

    def test_D5_direct_distance_adjacent(self):
        """Path D5: small adjacent distances."""
        marker = SyncMarker(name="test", normalized_time=0.1)
        assert math.isclose(marker.distance_to(0.15), 0.05)


# =============================================================================
# SyncMarker.distance_to WRAP-AROUND
# =============================================================================


class TestSyncMarkerDistanceToWrapAround:
    """Tests for SyncMarker.distance_to method - wrap-around cases."""

    def test_E1_wrap_from_high_to_low(self):
        """Path E1: distance from 0.9 to 0.1 = 0.2 (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.9)
        # Direct = 0.8, wrapped = 0.2, min = 0.2
        assert math.isclose(marker.distance_to(0.1), 0.2)

    def test_E2_wrap_from_low_to_high(self):
        """Path E2: distance from 0.1 to 0.9 = 0.2 (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.1)
        assert math.isclose(marker.distance_to(0.9), 0.2)

    def test_E3_wrap_near_endpoints(self):
        """Path E3: distance from 0.95 to 0.05 = 0.1 (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.95)
        # Direct = 0.9, wrapped = 0.1, min = 0.1
        assert math.isclose(marker.distance_to(0.05), 0.1)

    def test_E4_endpoints_zero_distance(self):
        """Path E4: distance from 0.0 to 1.0 = 0.0 (endpoints wrap)."""
        marker = SyncMarker(name="test", normalized_time=0.0)
        # Direct = 1.0, wrapped = 0.0, min = 0.0
        assert marker.distance_to(1.0) == 0.0

    def test_E4_endpoints_one_to_zero(self):
        """Path E4: distance from 1.0 to 0.0 = 0.0 (endpoints wrap)."""
        marker = SyncMarker(name="test", normalized_time=1.0)
        assert marker.distance_to(0.0) == 0.0

    def test_E5_wrap_larger_gap(self):
        """Path E5: distance from 0.8 to 0.2 = 0.4 (wrapped)."""
        marker = SyncMarker(name="test", normalized_time=0.8)
        # Direct = 0.6, wrapped = 0.4, min = 0.4
        assert math.isclose(marker.distance_to(0.2), 0.4)

    def test_E6_tie_case(self):
        """Path E6: distance from 0.75 to 0.25 = 0.5 (tie case)."""
        marker = SyncMarker(name="test", normalized_time=0.75)
        # Direct = 0.5, wrapped = 0.5, tie
        assert marker.distance_to(0.25) == 0.5

    def test_E6_symmetry(self):
        """Path E6: distance is symmetric."""
        marker1 = SyncMarker(name="test", normalized_time=0.3)
        marker2 = SyncMarker(name="test", normalized_time=0.7)
        assert marker1.distance_to(0.7) == marker2.distance_to(0.3)


# =============================================================================
# SyncMarkerTrack DATACLASS FIELDS
# =============================================================================


class TestSyncMarkerTrackFields:
    """Tests for SyncMarkerTrack dataclass fields."""

    def test_F1_markers_default_empty_list(self):
        """Path F1: markers default is empty list."""
        track = SyncMarkerTrack()
        assert track.markers == []
        assert isinstance(track.markers, list)

    def test_F2_markers_initialized_with_list(self):
        """Path F2: markers initialized with list preserves order."""
        m1 = SyncMarker(name="a", normalized_time=0.1)
        m2 = SyncMarker(name="b", normalized_time=0.2)
        track = SyncMarkerTrack(markers=[m1, m2])
        assert track.markers[0] == m1
        assert track.markers[1] == m2

    def test_F3_empty_track_len_zero(self):
        """Path F3: empty track has len 0."""
        track = SyncMarkerTrack()
        assert len(track.markers) == 0


# =============================================================================
# SyncMarkerTrack.add_marker BASIC
# =============================================================================


class TestSyncMarkerTrackAddMarkerBasic:
    """Tests for SyncMarkerTrack.add_marker basic functionality."""

    def test_G1_add_single_marker_to_empty_track(self):
        """Path G1: add single marker to empty track."""
        track = SyncMarkerTrack()
        marker = SyncMarker(name="test", normalized_time=0.5)
        track.add_marker(marker)
        assert len(track.markers) == 1
        assert track.markers[0] == marker

    def test_G2_add_marker_increases_length(self):
        """Path G2: add marker increases length."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        assert len(track.markers) == 1
        track.add_marker(SyncMarker(name="b", normalized_time=0.2))
        assert len(track.markers) == 2

    def test_G3_multiple_adds_accumulate(self):
        """Path G3: multiple adds accumulate."""
        track = SyncMarkerTrack()
        for i in range(5):
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=i * 0.1))
        assert len(track.markers) == 5

    def test_G4_same_normalized_time_allowed(self):
        """Path G4: same normalized_time allowed."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.5))
        track.add_marker(SyncMarker(name="b", normalized_time=0.5))
        assert len(track.markers) == 2


# =============================================================================
# SyncMarkerTrack.add_marker SORTING
# =============================================================================


class TestSyncMarkerTrackAddMarkerSorting:
    """Tests for SyncMarkerTrack.add_marker sorting behavior."""

    def test_H1_markers_sorted_by_normalized_time(self):
        """Path H1: markers sorted by normalized_time."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="c", normalized_time=0.3))
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.2))
        times = [m.normalized_time for m in track.markers]
        assert times == [0.1, 0.2, 0.3]

    def test_H2_add_at_end_maintains_sort(self):
        """Path H2: add at end maintains sort."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.2))
        track.add_marker(SyncMarker(name="c", normalized_time=0.3))
        times = [m.normalized_time for m in track.markers]
        assert times == [0.1, 0.2, 0.3]

    def test_H3_add_at_beginning_maintains_sort(self):
        """Path H3: add at beginning maintains sort."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="c", normalized_time=0.3))
        track.add_marker(SyncMarker(name="b", normalized_time=0.2))
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        assert track.markers[0].name == "a"
        assert track.markers[0].normalized_time == 0.1

    def test_H4_add_in_middle_maintains_sort(self):
        """Path H4: add in middle maintains sort."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="c", normalized_time=0.3))
        track.add_marker(SyncMarker(name="b", normalized_time=0.2))
        assert track.markers[1].name == "b"
        assert track.markers[1].normalized_time == 0.2

    def test_H5_multiple_out_of_order(self):
        """Path H5: add multiple out of order results in sorted list."""
        track = SyncMarkerTrack()
        times_to_add = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0]
        for t in times_to_add:
            track.add_marker(SyncMarker(name=f"m_{t}", normalized_time=t))
        times = [m.normalized_time for m in track.markers]
        assert times == sorted(times_to_add)


# =============================================================================
# SyncMarkerTrack.get_markers_by_name
# =============================================================================


class TestSyncMarkerTrackGetMarkersByName:
    """Tests for SyncMarkerTrack.get_markers_by_name method."""

    def test_I1_empty_track_returns_empty(self):
        """Path I1: empty track returns empty list."""
        track = SyncMarkerTrack()
        result = track.get_markers_by_name("test")
        assert result == []

    def test_I2_no_matching_name_returns_empty(self):
        """Path I2: no matching name returns empty list."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="foo", normalized_time=0.5))
        result = track.get_markers_by_name("bar")
        assert result == []

    def test_I3_single_match_returns_list_of_one(self):
        """Path I3: single match returns list of one."""
        track = SyncMarkerTrack()
        marker = SyncMarker(name="target", normalized_time=0.5)
        track.add_marker(marker)
        track.add_marker(SyncMarker(name="other", normalized_time=0.3))
        result = track.get_markers_by_name("target")
        assert len(result) == 1
        assert result[0] == marker

    def test_I4_multiple_matches_returns_all(self):
        """Path I4: multiple matches returns all."""
        track = SyncMarkerTrack()
        m1 = SyncMarker(name="foot", normalized_time=0.0)
        m2 = SyncMarker(name="foot", normalized_time=0.5)
        m3 = SyncMarker(name="hand", normalized_time=0.25)
        track.add_marker(m1)
        track.add_marker(m2)
        track.add_marker(m3)
        result = track.get_markers_by_name("foot")
        assert len(result) == 2
        assert m1 in result
        assert m2 in result

    def test_I5_case_sensitive_matching(self):
        """Path I5: case sensitive matching."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="Foot", normalized_time=0.5))
        track.add_marker(SyncMarker(name="foot", normalized_time=0.3))
        result_upper = track.get_markers_by_name("Foot")
        result_lower = track.get_markers_by_name("foot")
        assert len(result_upper) == 1
        assert len(result_lower) == 1
        assert result_upper[0].normalized_time == 0.5
        assert result_lower[0].normalized_time == 0.3


# =============================================================================
# SyncMarkerTrack.get_nearest_marker WITHOUT FILTER
# =============================================================================


class TestSyncMarkerTrackGetNearestMarkerNoFilter:
    """Tests for SyncMarkerTrack.get_nearest_marker without name filter."""

    def test_J1_empty_track_returns_none(self):
        """Path J1: empty track returns None."""
        track = SyncMarkerTrack()
        assert track.get_nearest_marker(0.5) is None

    def test_J2_single_marker_returns_that_marker(self):
        """Path J2: single marker returns that marker."""
        track = SyncMarkerTrack()
        marker = SyncMarker(name="only", normalized_time=0.5)
        track.add_marker(marker)
        result = track.get_nearest_marker(0.0)
        assert result == marker

    def test_J3_exact_match_returns_that_marker(self):
        """Path J3: exact match returns that marker."""
        track = SyncMarkerTrack()
        m1 = SyncMarker(name="a", normalized_time=0.3)
        m2 = SyncMarker(name="b", normalized_time=0.5)
        m3 = SyncMarker(name="c", normalized_time=0.7)
        track.add_marker(m1)
        track.add_marker(m2)
        track.add_marker(m3)
        result = track.get_nearest_marker(0.5)
        assert result == m2

    def test_J4_nearest_by_direct_distance(self):
        """Path J4: nearest by direct distance."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.2))
        track.add_marker(SyncMarker(name="b", normalized_time=0.4))
        track.add_marker(SyncMarker(name="c", normalized_time=0.8))
        result = track.get_nearest_marker(0.35)
        # 0.35 to 0.2 = 0.15, 0.35 to 0.4 = 0.05, 0.35 to 0.8 = 0.45
        assert result.normalized_time == 0.4

    def test_J5_nearest_considering_wrap_around(self):
        """Path J5: nearest considering wrap-around."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.9))
        # Query at 0.95: to 0.1 wrapped = 0.15, to 0.9 = 0.05
        result = track.get_nearest_marker(0.95)
        assert result.normalized_time == 0.9
        # Query at 0.05: to 0.1 = 0.05, to 0.9 wrapped = 0.15
        result2 = track.get_nearest_marker(0.05)
        assert result2.normalized_time == 0.1

    def test_J5_wrap_around_across_boundary(self):
        """Path J5: wrap-around selects marker across 0/1 boundary."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="near_end", normalized_time=0.95))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        # Query at 0.02: to 0.95 wrapped = 0.07, to 0.5 = 0.48
        result = track.get_nearest_marker(0.02)
        assert result.normalized_time == 0.95

    def test_J6_equidistant_returns_one(self):
        """Path J6: multiple equidistant returns one (deterministic)."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.25))
        track.add_marker(SyncMarker(name="b", normalized_time=0.75))
        # Query at 0.5: both are 0.25 away
        result = track.get_nearest_marker(0.5)
        # min() returns first minimum found, which depends on iteration order
        assert result is not None
        assert result.normalized_time in [0.25, 0.75]


# =============================================================================
# SyncMarkerTrack.get_nearest_marker WITH NAME FILTER
# =============================================================================


class TestSyncMarkerTrackGetNearestMarkerWithFilter:
    """Tests for SyncMarkerTrack.get_nearest_marker with name filter."""

    def test_K1_no_markers_with_name_returns_none(self):
        """Path K1: no markers with name returns None."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="foot", normalized_time=0.5))
        result = track.get_nearest_marker(0.5, name="hand")
        assert result is None

    def test_K2_filters_to_only_matching_names(self):
        """Path K2: filters to only matching names."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="foot", normalized_time=0.0))
        track.add_marker(SyncMarker(name="hand", normalized_time=0.3))
        track.add_marker(SyncMarker(name="foot", normalized_time=0.5))
        result = track.get_nearest_marker(0.4, name="foot")
        # foot markers at 0.0 and 0.5; 0.4 is closer to 0.5
        assert result.name == "foot"
        assert result.normalized_time == 0.5

    def test_K3_nearest_among_filtered_set(self):
        """Path K3: nearest among filtered set."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="sync", normalized_time=0.1))
        track.add_marker(SyncMarker(name="sync", normalized_time=0.4))
        track.add_marker(SyncMarker(name="sync", normalized_time=0.9))
        result = track.get_nearest_marker(0.35, name="sync")
        assert result.normalized_time == 0.4

    def test_K4_filter_excludes_closer_non_matching(self):
        """Path K4: filter excludes closer non-matching markers."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="target", normalized_time=0.8))
        track.add_marker(SyncMarker(name="other", normalized_time=0.5))  # closer!
        result = track.get_nearest_marker(0.5, name="target")
        # Even though "other" at 0.5 is exact match, we filter to "target"
        assert result.name == "target"
        assert result.normalized_time == 0.8


# =============================================================================
# SyncMarkerTrack.get_markers_in_range NORMAL RANGE
# =============================================================================


class TestSyncMarkerTrackGetMarkersInRangeNormal:
    """Tests for SyncMarkerTrack.get_markers_in_range with normal (non-wrapped) ranges."""

    def test_L1_empty_track_returns_empty(self):
        """Path L1: empty track returns empty list."""
        track = SyncMarkerTrack()
        result = track.get_markers_in_range(0.0, 0.5)
        assert result == []

    def test_L2_no_markers_in_range(self):
        """Path L2: no markers in range returns empty."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.1))
        track.add_marker(SyncMarker(name="b", normalized_time=0.9))
        result = track.get_markers_in_range(0.4, 0.6)
        assert result == []

    def test_L3_single_marker_in_range(self):
        """Path L3: single marker in range."""
        track = SyncMarkerTrack()
        m = SyncMarker(name="mid", normalized_time=0.5)
        track.add_marker(m)
        track.add_marker(SyncMarker(name="early", normalized_time=0.1))
        track.add_marker(SyncMarker(name="late", normalized_time=0.9))
        result = track.get_markers_in_range(0.4, 0.6)
        assert len(result) == 1
        assert result[0] == m

    def test_L4_multiple_markers_in_range(self):
        """Path L4: multiple markers in range."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.3))
        track.add_marker(SyncMarker(name="b", normalized_time=0.5))
        track.add_marker(SyncMarker(name="c", normalized_time=0.7))
        track.add_marker(SyncMarker(name="d", normalized_time=0.9))
        result = track.get_markers_in_range(0.2, 0.75)
        assert len(result) == 3
        times = [m.normalized_time for m in result]
        assert 0.3 in times
        assert 0.5 in times
        assert 0.7 in times

    def test_L5_marker_at_start_boundary_included(self):
        """Path L5: marker at start boundary included."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="boundary", normalized_time=0.3))
        result = track.get_markers_in_range(0.3, 0.5)
        assert len(result) == 1
        assert result[0].normalized_time == 0.3

    def test_L6_marker_at_end_boundary_included(self):
        """Path L6: marker at end boundary included."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="boundary", normalized_time=0.5))
        result = track.get_markers_in_range(0.3, 0.5)
        assert len(result) == 1
        assert result[0].normalized_time == 0.5

    def test_L7_markers_outside_range_excluded(self):
        """Path L7: markers outside range excluded."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="before", normalized_time=0.2))
        track.add_marker(SyncMarker(name="inside", normalized_time=0.5))
        track.add_marker(SyncMarker(name="after", normalized_time=0.8))
        result = track.get_markers_in_range(0.4, 0.6)
        assert len(result) == 1
        assert result[0].name == "inside"


# =============================================================================
# SyncMarkerTrack.get_markers_in_range WRAPPED RANGE
# =============================================================================


class TestSyncMarkerTrackGetMarkersInRangeWrapped:
    """Tests for SyncMarkerTrack.get_markers_in_range with wrapped ranges."""

    def test_M1_wrapped_range_includes_near_end(self):
        """Path M1: wrapped range 0.9 to 0.1 includes 0.95."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="near_end", normalized_time=0.95))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        result = track.get_markers_in_range(0.9, 0.1)
        assert len(result) == 1
        assert result[0].normalized_time == 0.95

    def test_M2_wrapped_range_includes_near_start(self):
        """Path M2: wrapped range 0.9 to 0.1 includes 0.05."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="near_start", normalized_time=0.05))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        result = track.get_markers_in_range(0.9, 0.1)
        assert len(result) == 1
        assert result[0].normalized_time == 0.05

    def test_M2_wrapped_range_includes_both_ends(self):
        """Path M2: wrapped range includes markers at both ends."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="a", normalized_time=0.05))
        track.add_marker(SyncMarker(name="b", normalized_time=0.95))
        result = track.get_markers_in_range(0.9, 0.1)
        assert len(result) == 2

    def test_M3_wrapped_range_excludes_middle(self):
        """Path M3: wrapped range excludes middle (e.g., 0.5)."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        track.add_marker(SyncMarker(name="near_end", normalized_time=0.95))
        result = track.get_markers_in_range(0.9, 0.1)
        assert len(result) == 1
        assert result[0].name == "near_end"

    def test_M4_wrapped_range_at_boundaries(self):
        """Path M4: wrapped range at boundaries."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="at_start", normalized_time=0.0))
        track.add_marker(SyncMarker(name="at_end", normalized_time=0.9))
        # Range 0.9 to 0.0: includes 0.9 (>= start) and 0.0 (<= end)
        result = track.get_markers_in_range(0.9, 0.0)
        assert len(result) == 2

    def test_M5_wrapped_range_includes_zero(self):
        """Path M5: wrapped range includes 0.0 endpoint."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="zero", normalized_time=0.0))
        track.add_marker(SyncMarker(name="half", normalized_time=0.5))
        result = track.get_markers_in_range(0.8, 0.2)
        # 0.0 <= 0.2, so included
        assert len(result) == 1
        assert result[0].normalized_time == 0.0

    def test_M6_wrapped_range_near_one(self):
        """Path M6: wrapped range with marker near 1.0."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(name="near_one", normalized_time=0.99))
        track.add_marker(SyncMarker(name="middle", normalized_time=0.5))
        result = track.get_markers_in_range(0.8, 0.2)
        # 0.99 >= 0.8, included
        assert len(result) == 1
        assert result[0].normalized_time == 0.99


# =============================================================================
# INTEGRATION / EDGE CASE TESTS
# =============================================================================


class TestSyncMarkerIntegration:
    """Integration tests combining multiple SyncMarker and SyncMarkerTrack operations."""

    def test_multiple_markers_same_name_different_times(self):
        """Multiple markers with same name at different times."""
        track = SyncMarkerTrack()
        for i in range(10):
            track.add_marker(SyncMarker(name="beat", normalized_time=i * 0.1))

        beats = track.get_markers_by_name("beat")
        assert len(beats) == 10

        # Get nearest beat to 0.55 (closer to 0.5 than 0.6)
        nearest = track.get_nearest_marker(0.55, name="beat")
        assert nearest.normalized_time == 0.5

    def test_markers_with_metadata(self):
        """Markers with metadata work correctly in track operations."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker(
            name="impact",
            normalized_time=0.5,
            bone_index=5,
            metadata={"force": 100, "sound": "hit.wav"}
        ))

        result = track.get_nearest_marker(0.5)
        assert result.metadata["force"] == 100
        assert result.bone_index == 5

    def test_large_track_performance(self):
        """Large number of markers handled efficiently."""
        track = SyncMarkerTrack()
        # Add 1000 markers
        import random
        random.seed(42)
        for i in range(1000):
            track.add_marker(SyncMarker(
                name=f"marker_{i % 10}",
                normalized_time=random.random()
            ))

        assert len(track.markers) == 1000

        # Should still find markers efficiently
        result = track.get_nearest_marker(0.5)
        assert result is not None

        # Range query
        in_range = track.get_markers_in_range(0.4, 0.6)
        assert len(in_range) > 0

    def test_distance_symmetry_comprehensive(self):
        """Distance calculation is symmetric for all marker pairs."""
        times = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for t1 in times:
            for t2 in times:
                m1 = SyncMarker(name="a", normalized_time=t1)
                m2 = SyncMarker(name="b", normalized_time=t2)
                assert math.isclose(m1.distance_to(t2), m2.distance_to(t1))

    def test_track_sorted_after_many_operations(self):
        """Track remains sorted after many add operations."""
        track = SyncMarkerTrack()
        import random
        random.seed(123)

        for _ in range(100):
            t = random.random()
            track.add_marker(SyncMarker(name="test", normalized_time=t))

        times = [m.normalized_time for m in track.markers]
        assert times == sorted(times)

    def test_get_markers_in_range_full_track(self):
        """Range 0.0 to 1.0 includes all markers."""
        track = SyncMarkerTrack()
        for i in range(10):
            track.add_marker(SyncMarker(name=f"m{i}", normalized_time=i * 0.1))

        result = track.get_markers_in_range(0.0, 1.0)
        assert len(result) == 10

    def test_wrapped_range_almost_full(self):
        """Wrapped range 0.1 to 0.0 is almost the full track."""
        track = SyncMarkerTrack()
        # Markers at 0.0, 0.25, 0.5, 0.75, 1.0 (clamped)
        track.add_marker(SyncMarker(name="a", normalized_time=0.0))
        track.add_marker(SyncMarker(name="b", normalized_time=0.25))
        track.add_marker(SyncMarker(name="c", normalized_time=0.5))
        track.add_marker(SyncMarker(name="d", normalized_time=0.75))

        # Range 0.1 to 0.0 (wrapped): includes 0.25, 0.5, 0.75, and 0.0
        result = track.get_markers_in_range(0.1, 0.0)
        assert len(result) == 4

    def test_get_time_for_various_durations(self):
        """get_time_for_duration works for various animation lengths."""
        marker = SyncMarker(name="foot_plant", normalized_time=0.25)

        # 4-second animation: foot plant at 1.0s
        assert math.isclose(marker.get_time_for_duration(4.0), 1.0)

        # 0.5-second animation: foot plant at 0.125s
        assert math.isclose(marker.get_time_for_duration(0.5), 0.125)

        # 10-second animation: foot plant at 2.5s
        assert math.isclose(marker.get_time_for_duration(10.0), 2.5)
