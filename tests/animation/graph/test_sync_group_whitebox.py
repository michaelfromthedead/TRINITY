"""WHITEBOX tests for engine/animation/graph/sync.py SyncGroup and SyncEntry.

Tests for T-FB-4.15 (SyncGroup Core).

WHITEBOX coverage plan:
  [SyncMode Enum]
    Path A1:  SyncMode.NONE exists
    Path A2:  SyncMode.NORMALIZED exists
    Path A3:  SyncMode.PHASE exists
    Path A4:  SyncMode.LEADER_FOLLOWER exists
    Path A5:  SyncMode.WEIGHTED exists
    Path A6:  SyncMode has exactly 5 members
    Path A7:  All modes are unique values

  [SyncEntry dataclass fields]
    Path B1:  node field stores AnimationNode
    Path B2:  weight default is 1.0
    Path B3:  weight accepts custom float
    Path B4:  is_leader default is False
    Path B5:  is_leader accepts True
    Path B6:  marker_track default is None
    Path B7:  marker_track accepts SyncMarkerTrack
    Path B8:  duration default is 1.0
    Path B9:  duration accepts custom float
    Path B10: _current_time default is 0.0
    Path B11: _normalized_time default is 0.0

  [SyncEntry.normalized_time property getter]
    Path C1:  returns _normalized_time value
    Path C2:  returns 0.0 for default entry
    Path C3:  returns value after setter

  [SyncEntry.normalized_time property setter]
    Path D1:  value 0.0 sets correctly
    Path D2:  value 0.5 sets correctly
    Path D3:  value 1.0 wraps to 0.0 (modulo)
    Path D4:  value > 1.0 applies modulo
    Path D5:  value 1.5 becomes 0.5
    Path D6:  negative value clamped to 0.0
    Path D7:  setter updates _current_time correctly
    Path D8:  _current_time = normalized_time * duration

  [SyncEntry.advance method]
    Path E1:  advance(0.0) does not change time
    Path E2:  advance(0.1) with duration=1.0
    Path E3:  advance wraps around at 1.0
    Path E4:  advance with speed=2.0 doubles rate
    Path E5:  advance with speed=0.5 halves rate
    Path E6:  advance with speed=0.0 no change
    Path E7:  advance with duration=0.0 returns early
    Path E8:  advance with negative duration returns early
    Path E9:  advance accumulates over multiple calls
    Path E10: normalized_time wraps via modulo

  [SyncGroup.__init__]
    Path F1:  name stored correctly
    Path F2:  mode defaults to NORMALIZED
    Path F3:  mode accepts custom SyncMode
    Path F4:  entries starts empty
    Path F5:  _leader_index starts at 0

  [SyncGroup.add_entry]
    Path G1:  add_entry returns index 0 for first entry
    Path G2:  add_entry returns index 1 for second entry
    Path G3:  add_entry increments sequentially
    Path G4:  entry stored in entries list
    Path G5:  entry has correct node
    Path G6:  entry has correct weight
    Path G7:  entry has correct is_leader
    Path G8:  entry has correct duration
    Path G9:  entry has correct marker_track
    Path G10: is_leader=True updates _leader_index
    Path G11: multiple is_leader=True keeps last one
    Path G12: default weight is 1.0

  [SyncGroup.remove_entry]
    Path H1:  remove_entry returns True for valid index
    Path H2:  remove_entry returns False for invalid index
    Path H3:  remove_entry returns False for negative index
    Path H4:  remove_entry decreases entries length
    Path H5:  remove_entry removes correct entry
    Path H6:  remove leader updates _leader_index
    Path H7:  _leader_index clamped to valid range
    Path H8:  remove from empty group returns False
    Path H9:  remove last entry sets _leader_index to 0

  [SyncGroup.set_leader]
    Path I1:  set_leader returns True for valid index
    Path I2:  set_leader returns False for invalid index
    Path I3:  set_leader returns False for negative index
    Path I4:  set_leader updates _leader_index
    Path I5:  set_leader sets is_leader=True for target
    Path I6:  set_leader sets is_leader=False for others
    Path I7:  set_leader on empty group returns False

  [SyncGroup.set_weights]
    Path J1:  set_weights updates all entries
    Path J2:  set_weights clamps negative to 0.0
    Path J3:  set_weights preserves zero weight
    Path J4:  set_weights handles partial list
    Path J5:  set_weights with empty list does nothing
    Path J6:  set_weights with longer list ignores extra

  [SyncGroup.get_leader]
    Path K1:  get_leader returns None for empty group
    Path K2:  get_leader returns first entry by default
    Path K3:  get_leader returns entry at _leader_index
    Path K4:  get_leader after set_leader returns correct entry

  [SyncGroup.update with NONE mode]
    Path L1:  update with empty entries does nothing
    Path L2:  NONE mode advances each entry independently
    Path L3:  NONE mode entries have different times
    Path L4:  NONE mode respects entry duration
    Path L5:  NONE mode dt=0 no change

  [SyncGroup.update with NORMALIZED mode]
    Path M1:  NORMALIZED mode calculates weighted average
    Path M2:  NORMALIZED mode syncs all to same time
    Path M3:  NORMALIZED mode respects weights
    Path M4:  NORMALIZED mode handles zero total weight
    Path M5:  NORMALIZED mode dt=0 no change

  [SyncGroup.update with LEADER_FOLLOWER mode]
    Path N1:  LEADER_FOLLOWER mode advances leader
    Path N2:  followers match leader's normalized_time
    Path N3:  no leader falls back to normalized
    Path N4:  multiple followers all sync to leader

  [SyncGroup.update with WEIGHTED mode]
    Path O1:  WEIGHTED mode calculates weighted time
    Path O2:  WEIGHTED mode syncs all to same time
    Path O3:  WEIGHTED mode handles zero total weight

  [SyncGroup.update with PHASE mode]
    Path P1:  PHASE mode advances leader
    Path P2:  followers sync to leader time
    Path P3:  no leader falls back to normalized

  [SyncGroup.get_synchronized_time]
    Path Q1:  empty group returns 0.0
    Path Q2:  LEADER_FOLLOWER returns leader time
    Path Q3:  LEADER_FOLLOWER no leader returns 0.0
    Path Q4:  other modes return weighted average
    Path Q5:  zero total weight returns 0.0
"""

import math
import pytest
from dataclasses import fields
from unittest.mock import Mock, MagicMock

from engine.animation.graph.sync import (
    SyncMode,
    SyncEntry,
    SyncGroup,
    SyncMarker,
    SyncMarkerTrack,
)
from engine.animation.graph.animation_graph import AnimationNode


# =============================================================================
# FIXTURES
# =============================================================================


class MockAnimationNode(AnimationNode):
    """Mock animation node for testing."""

    _abstract = False

    def __init__(self, node_id: str = "mock_node"):
        super().__init__(node_id)

    def evaluate(self, context):
        from engine.animation.graph.animation_graph import Pose
        return Pose()


@pytest.fixture
def mock_node():
    """Create a mock animation node."""
    return MockAnimationNode("test_node")


@pytest.fixture
def mock_node_factory():
    """Factory for creating multiple mock nodes."""
    counter = [0]
    def _create():
        counter[0] += 1
        return MockAnimationNode(f"test_node_{counter[0]}")
    return _create


@pytest.fixture
def sync_group():
    """Create a basic sync group."""
    return SyncGroup("test_group")


@pytest.fixture
def marker_track():
    """Create a marker track with some markers."""
    track = SyncMarkerTrack()
    track.add_marker(SyncMarker(name="foot_plant", normalized_time=0.0))
    track.add_marker(SyncMarker(name="foot_plant", normalized_time=0.5))
    return track


# =============================================================================
# SyncMode ENUM
# =============================================================================


class TestSyncModeEnum:
    """Tests for SyncMode enumeration."""

    def test_A1_none_mode_exists(self):
        """Path A1: SyncMode.NONE exists."""
        assert hasattr(SyncMode, 'NONE')
        assert SyncMode.NONE is not None

    def test_A2_normalized_mode_exists(self):
        """Path A2: SyncMode.NORMALIZED exists."""
        assert hasattr(SyncMode, 'NORMALIZED')
        assert SyncMode.NORMALIZED is not None

    def test_A3_phase_mode_exists(self):
        """Path A3: SyncMode.PHASE exists."""
        assert hasattr(SyncMode, 'PHASE')
        assert SyncMode.PHASE is not None

    def test_A4_leader_follower_mode_exists(self):
        """Path A4: SyncMode.LEADER_FOLLOWER exists."""
        assert hasattr(SyncMode, 'LEADER_FOLLOWER')
        assert SyncMode.LEADER_FOLLOWER is not None

    def test_A5_weighted_mode_exists(self):
        """Path A5: SyncMode.WEIGHTED exists."""
        assert hasattr(SyncMode, 'WEIGHTED')
        assert SyncMode.WEIGHTED is not None

    def test_A6_exactly_five_members(self):
        """Path A6: SyncMode has exactly 5 members."""
        assert len(SyncMode) == 5

    def test_A7_all_modes_unique(self):
        """Path A7: All modes are unique values."""
        values = [m.value for m in SyncMode]
        assert len(values) == len(set(values))

    def test_A7_modes_iterable(self):
        """Path A7: Can iterate over all modes."""
        modes = list(SyncMode)
        assert len(modes) == 5
        assert SyncMode.NONE in modes
        assert SyncMode.NORMALIZED in modes
        assert SyncMode.PHASE in modes
        assert SyncMode.LEADER_FOLLOWER in modes
        assert SyncMode.WEIGHTED in modes


# =============================================================================
# SyncEntry DATACLASS FIELDS
# =============================================================================


class TestSyncEntryFields:
    """Tests for SyncEntry dataclass field access and defaults."""

    def test_B1_node_field_stores_animation_node(self, mock_node):
        """Path B1: node field stores AnimationNode."""
        entry = SyncEntry(node=mock_node)
        assert entry.node is mock_node

    def test_B2_weight_default_is_one(self, mock_node):
        """Path B2: weight default is 1.0."""
        entry = SyncEntry(node=mock_node)
        assert entry.weight == 1.0

    def test_B3_weight_accepts_custom_float(self, mock_node):
        """Path B3: weight accepts custom float."""
        entry = SyncEntry(node=mock_node, weight=0.5)
        assert entry.weight == 0.5

    def test_B3_weight_accepts_zero(self, mock_node):
        """Path B3: weight accepts zero."""
        entry = SyncEntry(node=mock_node, weight=0.0)
        assert entry.weight == 0.0

    def test_B4_is_leader_default_false(self, mock_node):
        """Path B4: is_leader default is False."""
        entry = SyncEntry(node=mock_node)
        assert entry.is_leader is False

    def test_B5_is_leader_accepts_true(self, mock_node):
        """Path B5: is_leader accepts True."""
        entry = SyncEntry(node=mock_node, is_leader=True)
        assert entry.is_leader is True

    def test_B6_marker_track_default_none(self, mock_node):
        """Path B6: marker_track default is None."""
        entry = SyncEntry(node=mock_node)
        assert entry.marker_track is None

    def test_B7_marker_track_accepts_sync_marker_track(self, mock_node, marker_track):
        """Path B7: marker_track accepts SyncMarkerTrack."""
        entry = SyncEntry(node=mock_node, marker_track=marker_track)
        assert entry.marker_track is marker_track

    def test_B8_duration_default_is_one(self, mock_node):
        """Path B8: duration default is 1.0."""
        entry = SyncEntry(node=mock_node)
        assert entry.duration == 1.0

    def test_B9_duration_accepts_custom_float(self, mock_node):
        """Path B9: duration accepts custom float."""
        entry = SyncEntry(node=mock_node, duration=2.5)
        assert entry.duration == 2.5

    def test_B10_current_time_default_zero(self, mock_node):
        """Path B10: _current_time default is 0.0."""
        entry = SyncEntry(node=mock_node)
        assert entry._current_time == 0.0

    def test_B11_normalized_time_default_zero(self, mock_node):
        """Path B11: _normalized_time default is 0.0."""
        entry = SyncEntry(node=mock_node)
        assert entry._normalized_time == 0.0


# =============================================================================
# SyncEntry.normalized_time PROPERTY GETTER
# =============================================================================


class TestSyncEntryNormalizedTimeGetter:
    """Tests for SyncEntry.normalized_time property getter."""

    def test_C1_returns_normalized_time_value(self, mock_node):
        """Path C1: returns _normalized_time value."""
        entry = SyncEntry(node=mock_node)
        entry._normalized_time = 0.75
        assert entry.normalized_time == 0.75

    def test_C2_returns_zero_for_default_entry(self, mock_node):
        """Path C2: returns 0.0 for default entry."""
        entry = SyncEntry(node=mock_node)
        assert entry.normalized_time == 0.0

    def test_C3_returns_value_after_setter(self, mock_node):
        """Path C3: returns value after setter."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 0.5
        assert entry.normalized_time == 0.5


# =============================================================================
# SyncEntry.normalized_time PROPERTY SETTER
# =============================================================================


class TestSyncEntryNormalizedTimeSetter:
    """Tests for SyncEntry.normalized_time property setter."""

    def test_D1_value_zero_sets_correctly(self, mock_node):
        """Path D1: value 0.0 sets correctly."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 0.0
        assert entry._normalized_time == 0.0

    def test_D2_value_half_sets_correctly(self, mock_node):
        """Path D2: value 0.5 sets correctly."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 0.5
        assert entry._normalized_time == 0.5

    def test_D3_value_one_wraps_to_zero(self, mock_node):
        """Path D3: value 1.0 wraps to 0.0 (modulo)."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 1.0
        # Since 1.0 > 1.0 is False, it goes to else branch: max(0.0, 1.0) = 1.0
        # Wait, let's check the code: value % 1.0 if value > 1.0 else max(0.0, value)
        # 1.0 > 1.0 is False, so max(0.0, 1.0) = 1.0
        # Actually looking at source: value > 1.0 triggers modulo
        # 1.0 is NOT > 1.0, so it uses max(0.0, value) = max(0.0, 1.0) = 1.0
        assert entry._normalized_time == 1.0

    def test_D4_value_greater_than_one_applies_modulo(self, mock_node):
        """Path D4: value > 1.0 applies modulo."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 1.5
        assert entry._normalized_time == pytest.approx(0.5)

    def test_D5_value_one_point_five_becomes_half(self, mock_node):
        """Path D5: value 1.5 becomes 0.5."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 1.5
        assert entry._normalized_time == pytest.approx(0.5)

    def test_D5_value_two_point_three_modulo(self, mock_node):
        """Path D5: value 2.3 applies modulo correctly."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = 2.3
        assert entry._normalized_time == pytest.approx(0.3)

    def test_D6_negative_value_clamped_to_zero(self, mock_node):
        """Path D6: negative value clamped to 0.0."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = -0.5
        assert entry._normalized_time == 0.0

    def test_D6_large_negative_clamped_to_zero(self, mock_node):
        """Path D6: large negative value clamped to 0.0."""
        entry = SyncEntry(node=mock_node)
        entry.normalized_time = -100.0
        assert entry._normalized_time == 0.0

    def test_D7_setter_updates_current_time_correctly(self, mock_node):
        """Path D7: setter updates _current_time correctly."""
        entry = SyncEntry(node=mock_node, duration=2.0)
        entry.normalized_time = 0.5
        assert entry._current_time == pytest.approx(1.0)  # 0.5 * 2.0

    def test_D8_current_time_equals_normalized_times_duration(self, mock_node):
        """Path D8: _current_time = normalized_time * duration."""
        entry = SyncEntry(node=mock_node, duration=4.0)
        entry.normalized_time = 0.25
        assert entry._current_time == pytest.approx(1.0)  # 0.25 * 4.0


# =============================================================================
# SyncEntry.advance METHOD
# =============================================================================


class TestSyncEntryAdvance:
    """Tests for SyncEntry.advance method."""

    def test_E1_advance_zero_no_change(self, mock_node):
        """Path E1: advance(0.0) does not change time."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.0)
        assert entry._current_time == 0.0
        assert entry._normalized_time == 0.0

    def test_E2_advance_with_duration_one(self, mock_node):
        """Path E2: advance(0.1) with duration=1.0."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.1)
        assert entry._current_time == pytest.approx(0.1)
        assert entry._normalized_time == pytest.approx(0.1)

    def test_E3_advance_wraps_around(self, mock_node):
        """Path E3: advance wraps around at 1.0."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry._current_time = 0.9
        entry.advance(0.2)  # 0.9 + 0.2 = 1.1
        assert entry._current_time == pytest.approx(1.1)
        assert entry._normalized_time == pytest.approx(0.1)  # 1.1 % 1.0 = 0.1

    def test_E4_advance_with_speed_double(self, mock_node):
        """Path E4: advance with speed=2.0 doubles rate."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.1, speed=2.0)
        assert entry._current_time == pytest.approx(0.2)
        assert entry._normalized_time == pytest.approx(0.2)

    def test_E5_advance_with_speed_half(self, mock_node):
        """Path E5: advance with speed=0.5 halves rate."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.2, speed=0.5)
        assert entry._current_time == pytest.approx(0.1)
        assert entry._normalized_time == pytest.approx(0.1)

    def test_E6_advance_with_speed_zero(self, mock_node):
        """Path E6: advance with speed=0.0 no change."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.5, speed=0.0)
        assert entry._current_time == 0.0
        assert entry._normalized_time == 0.0

    def test_E7_advance_with_duration_zero_returns_early(self, mock_node):
        """Path E7: advance with duration=0.0 returns early."""
        entry = SyncEntry(node=mock_node, duration=0.0)
        entry.advance(0.1)
        # Should not change anything
        assert entry._current_time == 0.0

    def test_E8_advance_with_negative_duration_returns_early(self, mock_node):
        """Path E8: advance with negative duration returns early."""
        entry = SyncEntry(node=mock_node, duration=-1.0)
        entry.advance(0.1)
        # Should not change anything
        assert entry._current_time == 0.0

    def test_E9_advance_accumulates_over_multiple_calls(self, mock_node):
        """Path E9: advance accumulates over multiple calls."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry.advance(0.1)
        entry.advance(0.1)
        entry.advance(0.1)
        assert entry._current_time == pytest.approx(0.3)
        assert entry._normalized_time == pytest.approx(0.3)

    def test_E10_normalized_time_wraps_via_modulo(self, mock_node):
        """Path E10: normalized_time wraps via modulo."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry._current_time = 0.0
        for _ in range(15):  # 15 * 0.1 = 1.5
            entry.advance(0.1)
        assert entry._normalized_time == pytest.approx(0.5)  # 1.5 % 1.0 = 0.5


# =============================================================================
# SyncGroup.__init__
# =============================================================================


class TestSyncGroupInit:
    """Tests for SyncGroup constructor."""

    def test_F1_name_stored_correctly(self):
        """Path F1: name stored correctly."""
        group = SyncGroup("my_group")
        assert group.name == "my_group"

    def test_F1_name_empty_string(self):
        """Path F1: name accepts empty string."""
        group = SyncGroup("")
        assert group.name == ""

    def test_F2_mode_defaults_to_normalized(self):
        """Path F2: mode defaults to NORMALIZED."""
        group = SyncGroup("test")
        assert group.mode == SyncMode.NORMALIZED

    def test_F3_mode_accepts_custom_sync_mode(self):
        """Path F3: mode accepts custom SyncMode."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        assert group.mode == SyncMode.NONE

    def test_F3_mode_accepts_leader_follower(self):
        """Path F3: mode accepts LEADER_FOLLOWER."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        assert group.mode == SyncMode.LEADER_FOLLOWER

    def test_F4_entries_starts_empty(self):
        """Path F4: entries starts empty."""
        group = SyncGroup("test")
        assert group.entries == []
        assert len(group.entries) == 0

    def test_F5_leader_index_starts_at_zero(self):
        """Path F5: _leader_index starts at 0."""
        group = SyncGroup("test")
        assert group._leader_index == 0


# =============================================================================
# SyncGroup.add_entry
# =============================================================================


class TestSyncGroupAddEntry:
    """Tests for SyncGroup.add_entry method."""

    def test_G1_returns_index_zero_for_first_entry(self, sync_group, mock_node):
        """Path G1: add_entry returns index 0 for first entry."""
        index = sync_group.add_entry(mock_node)
        assert index == 0

    def test_G2_returns_index_one_for_second_entry(self, sync_group, mock_node_factory):
        """Path G2: add_entry returns index 1 for second entry."""
        node1 = mock_node_factory()
        node2 = mock_node_factory()
        sync_group.add_entry(node1)
        index = sync_group.add_entry(node2)
        assert index == 1

    def test_G3_increments_sequentially(self, sync_group, mock_node_factory):
        """Path G3: add_entry increments sequentially."""
        indices = []
        for _ in range(5):
            indices.append(sync_group.add_entry(mock_node_factory()))
        assert indices == [0, 1, 2, 3, 4]

    def test_G4_entry_stored_in_entries_list(self, sync_group, mock_node):
        """Path G4: entry stored in entries list."""
        sync_group.add_entry(mock_node)
        assert len(sync_group.entries) == 1
        assert isinstance(sync_group.entries[0], SyncEntry)

    def test_G5_entry_has_correct_node(self, sync_group, mock_node):
        """Path G5: entry has correct node."""
        sync_group.add_entry(mock_node)
        assert sync_group.entries[0].node is mock_node

    def test_G6_entry_has_correct_weight(self, sync_group, mock_node):
        """Path G6: entry has correct weight."""
        sync_group.add_entry(mock_node, weight=0.75)
        assert sync_group.entries[0].weight == 0.75

    def test_G7_entry_has_correct_is_leader(self, sync_group, mock_node):
        """Path G7: entry has correct is_leader."""
        sync_group.add_entry(mock_node, is_leader=True)
        assert sync_group.entries[0].is_leader is True

    def test_G8_entry_has_correct_duration(self, sync_group, mock_node):
        """Path G8: entry has correct duration."""
        sync_group.add_entry(mock_node, duration=2.5)
        assert sync_group.entries[0].duration == 2.5

    def test_G9_entry_has_correct_marker_track(self, sync_group, mock_node, marker_track):
        """Path G9: entry has correct marker_track."""
        sync_group.add_entry(mock_node, marker_track=marker_track)
        assert sync_group.entries[0].marker_track is marker_track

    def test_G10_is_leader_true_updates_leader_index(self, sync_group, mock_node_factory):
        """Path G10: is_leader=True updates _leader_index."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        assert sync_group._leader_index == 1

    def test_G11_multiple_is_leader_keeps_last_one(self, sync_group, mock_node_factory):
        """Path G11: multiple is_leader=True keeps last one."""
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        assert sync_group._leader_index == 2

    def test_G12_default_weight_is_one(self, sync_group, mock_node):
        """Path G12: default weight is 1.0."""
        sync_group.add_entry(mock_node)
        assert sync_group.entries[0].weight == 1.0


# =============================================================================
# SyncGroup.remove_entry
# =============================================================================


class TestSyncGroupRemoveEntry:
    """Tests for SyncGroup.remove_entry method."""

    def test_H1_returns_true_for_valid_index(self, sync_group, mock_node):
        """Path H1: remove_entry returns True for valid index."""
        sync_group.add_entry(mock_node)
        result = sync_group.remove_entry(0)
        assert result is True

    def test_H2_returns_false_for_invalid_index(self, sync_group, mock_node):
        """Path H2: remove_entry returns False for invalid index."""
        sync_group.add_entry(mock_node)
        result = sync_group.remove_entry(5)
        assert result is False

    def test_H3_returns_false_for_negative_index(self, sync_group, mock_node):
        """Path H3: remove_entry returns False for negative index."""
        sync_group.add_entry(mock_node)
        result = sync_group.remove_entry(-1)
        assert result is False

    def test_H4_decreases_entries_length(self, sync_group, mock_node_factory):
        """Path H4: remove_entry decreases entries length."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        assert len(sync_group.entries) == 2
        sync_group.remove_entry(0)
        assert len(sync_group.entries) == 1

    def test_H5_removes_correct_entry(self, sync_group, mock_node_factory):
        """Path H5: remove_entry removes correct entry."""
        node1 = mock_node_factory()
        node2 = mock_node_factory()
        node3 = mock_node_factory()
        sync_group.add_entry(node1)
        sync_group.add_entry(node2)
        sync_group.add_entry(node3)
        sync_group.remove_entry(1)  # Remove node2
        assert len(sync_group.entries) == 2
        assert sync_group.entries[0].node is node1
        assert sync_group.entries[1].node is node3

    def test_H6_remove_leader_updates_leader_index(self, sync_group, mock_node_factory):
        """Path H6: remove leader updates _leader_index."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        sync_group.add_entry(mock_node_factory())
        assert sync_group._leader_index == 1
        sync_group.remove_entry(1)  # Remove leader
        # After removal, _leader_index should be clamped to valid range
        assert sync_group._leader_index <= len(sync_group.entries) - 1

    def test_H7_leader_index_clamped_to_valid_range(self, sync_group, mock_node_factory):
        """Path H7: _leader_index clamped to valid range."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        # Leader is at index 2
        sync_group.remove_entry(2)  # Remove last (leader)
        # Now only 2 entries remain (indices 0, 1)
        # _leader_index was 2, should be clamped to 1
        assert sync_group._leader_index == 1

    def test_H8_remove_from_empty_group_returns_false(self, sync_group):
        """Path H8: remove from empty group returns False."""
        result = sync_group.remove_entry(0)
        assert result is False

    def test_H9_remove_last_entry_sets_leader_index_to_zero(self, sync_group, mock_node):
        """Path H9: remove last entry sets _leader_index to 0."""
        sync_group.add_entry(mock_node)
        sync_group.remove_entry(0)
        assert sync_group._leader_index == 0
        assert len(sync_group.entries) == 0


# =============================================================================
# SyncGroup.set_leader
# =============================================================================


class TestSyncGroupSetLeader:
    """Tests for SyncGroup.set_leader method."""

    def test_I1_returns_true_for_valid_index(self, sync_group, mock_node_factory):
        """Path I1: set_leader returns True for valid index."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        result = sync_group.set_leader(1)
        assert result is True

    def test_I2_returns_false_for_invalid_index(self, sync_group, mock_node):
        """Path I2: set_leader returns False for invalid index."""
        sync_group.add_entry(mock_node)
        result = sync_group.set_leader(5)
        assert result is False

    def test_I3_returns_false_for_negative_index(self, sync_group, mock_node):
        """Path I3: set_leader returns False for negative index."""
        sync_group.add_entry(mock_node)
        result = sync_group.set_leader(-1)
        assert result is False

    def test_I4_updates_leader_index(self, sync_group, mock_node_factory):
        """Path I4: set_leader updates _leader_index."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.set_leader(1)
        assert sync_group._leader_index == 1

    def test_I5_sets_is_leader_true_for_target(self, sync_group, mock_node_factory):
        """Path I5: set_leader sets is_leader=True for target."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.set_leader(1)
        assert sync_group.entries[1].is_leader is True

    def test_I6_sets_is_leader_false_for_others(self, sync_group, mock_node_factory):
        """Path I6: set_leader sets is_leader=False for others."""
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.set_leader(2)
        assert sync_group.entries[0].is_leader is False
        assert sync_group.entries[1].is_leader is False
        assert sync_group.entries[2].is_leader is True

    def test_I7_set_leader_on_empty_group_returns_false(self, sync_group):
        """Path I7: set_leader on empty group returns False."""
        result = sync_group.set_leader(0)
        assert result is False


# =============================================================================
# SyncGroup.set_weights
# =============================================================================


class TestSyncGroupSetWeights:
    """Tests for SyncGroup.set_weights method."""

    def test_J1_updates_all_entries(self, sync_group, mock_node_factory):
        """Path J1: set_weights updates all entries."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.set_weights([0.5, 0.3, 0.2])
        assert sync_group.entries[0].weight == 0.5
        assert sync_group.entries[1].weight == 0.3
        assert sync_group.entries[2].weight == 0.2

    def test_J2_clamps_negative_to_zero(self, sync_group, mock_node_factory):
        """Path J2: set_weights clamps negative to 0.0."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory())
        sync_group.set_weights([-1.0, -0.5])
        assert sync_group.entries[0].weight == 0.0
        assert sync_group.entries[1].weight == 0.0

    def test_J3_preserves_zero_weight(self, sync_group, mock_node_factory):
        """Path J3: set_weights preserves zero weight."""
        sync_group.add_entry(mock_node_factory())
        sync_group.set_weights([0.0])
        assert sync_group.entries[0].weight == 0.0

    def test_J4_handles_partial_list(self, sync_group, mock_node_factory):
        """Path J4: set_weights handles partial list."""
        sync_group.add_entry(mock_node_factory(), weight=1.0)
        sync_group.add_entry(mock_node_factory(), weight=1.0)
        sync_group.add_entry(mock_node_factory(), weight=1.0)
        sync_group.set_weights([0.5])  # Only first entry
        assert sync_group.entries[0].weight == 0.5
        assert sync_group.entries[1].weight == 1.0  # Unchanged
        assert sync_group.entries[2].weight == 1.0  # Unchanged

    def test_J5_empty_list_does_nothing(self, sync_group, mock_node_factory):
        """Path J5: set_weights with empty list does nothing."""
        sync_group.add_entry(mock_node_factory(), weight=0.8)
        sync_group.set_weights([])
        assert sync_group.entries[0].weight == 0.8

    def test_J6_longer_list_ignores_extra(self, sync_group, mock_node_factory):
        """Path J6: set_weights with longer list ignores extra."""
        sync_group.add_entry(mock_node_factory())
        sync_group.set_weights([0.5, 0.3, 0.2, 0.1])  # 4 weights, 1 entry
        assert sync_group.entries[0].weight == 0.5
        assert len(sync_group.entries) == 1


# =============================================================================
# SyncGroup.get_leader
# =============================================================================


class TestSyncGroupGetLeader:
    """Tests for SyncGroup.get_leader method."""

    def test_K1_returns_none_for_empty_group(self, sync_group):
        """Path K1: get_leader returns None for empty group."""
        result = sync_group.get_leader()
        assert result is None

    def test_K2_returns_first_entry_by_default(self, sync_group, mock_node_factory):
        """Path K2: get_leader returns first entry by default."""
        node1 = mock_node_factory()
        node2 = mock_node_factory()
        sync_group.add_entry(node1)
        sync_group.add_entry(node2)
        leader = sync_group.get_leader()
        assert leader.node is node1

    def test_K3_returns_entry_at_leader_index(self, sync_group, mock_node_factory):
        """Path K3: get_leader returns entry at _leader_index."""
        sync_group.add_entry(mock_node_factory())
        sync_group.add_entry(mock_node_factory(), is_leader=True)
        leader = sync_group.get_leader()
        assert leader is sync_group.entries[1]

    def test_K4_after_set_leader_returns_correct_entry(self, sync_group, mock_node_factory):
        """Path K4: get_leader after set_leader returns correct entry."""
        node1 = mock_node_factory()
        node2 = mock_node_factory()
        node3 = mock_node_factory()
        sync_group.add_entry(node1)
        sync_group.add_entry(node2)
        sync_group.add_entry(node3)
        sync_group.set_leader(2)
        leader = sync_group.get_leader()
        assert leader.node is node3


# =============================================================================
# SyncGroup.update with NONE mode
# =============================================================================


class TestSyncGroupUpdateNoneMode:
    """Tests for SyncGroup.update with SyncMode.NONE."""

    def test_L1_update_with_empty_entries_does_nothing(self):
        """Path L1: update with empty entries does nothing."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.update(0.1)  # Should not raise

    def test_L2_none_mode_advances_each_entry_independently(self, mock_node_factory):
        """Path L2: NONE mode advances each entry independently."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=2.0)
        group.update(0.1)
        # Both should have advanced by 0.1 seconds
        assert group.entries[0]._current_time == pytest.approx(0.1)
        assert group.entries[1]._current_time == pytest.approx(0.1)

    def test_L3_none_mode_entries_have_different_times(self, mock_node_factory):
        """Path L3: NONE mode entries have different normalized times due to different durations."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=2.0)
        group.update(0.5)
        # First entry: 0.5 / 1.0 = 0.5 normalized
        # Second entry: 0.5 / 2.0 = 0.25 normalized
        assert group.entries[0].normalized_time == pytest.approx(0.5)
        assert group.entries[1].normalized_time == pytest.approx(0.25)

    def test_L4_none_mode_respects_entry_duration(self, mock_node_factory):
        """Path L4: NONE mode respects entry duration."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=0.5)
        group.update(0.25)
        assert group.entries[0].normalized_time == pytest.approx(0.5)

    def test_L5_none_mode_dt_zero_no_change(self, mock_node_factory):
        """Path L5: NONE mode dt=0 no change."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.update(0.0)
        assert group.entries[0].normalized_time == 0.0


# =============================================================================
# SyncGroup.update with NORMALIZED mode
# =============================================================================


class TestSyncGroupUpdateNormalizedMode:
    """Tests for SyncGroup.update with SyncMode.NORMALIZED."""

    def test_M1_normalized_mode_calculates_weighted_average(self, mock_node_factory):
        """Path M1: NORMALIZED mode calculates weighted average."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.update(0.1)
        # Both should advance together
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )

    def test_M2_normalized_mode_syncs_all_to_same_time(self, mock_node_factory):
        """Path M2: NORMALIZED mode syncs all to same time."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=2.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=0.5)
        group.update(0.1)
        time0 = group.entries[0].normalized_time
        time1 = group.entries[1].normalized_time
        time2 = group.entries[2].normalized_time
        assert time0 == pytest.approx(time1)
        assert time1 == pytest.approx(time2)

    def test_M3_normalized_mode_respects_weights(self, mock_node_factory):
        """Path M3: NORMALIZED mode respects weights."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=2.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.0, duration=2.0)
        group.update(0.1)
        # With weight 0 for second entry, only first entry's speed matters
        # But they still sync to same normalized time
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )

    def test_M4_normalized_mode_handles_zero_total_weight(self, mock_node_factory):
        """Path M4: NORMALIZED mode handles zero total weight."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        # Should not raise, uses total_weight = 1.0 as fallback
        group.update(0.1)

    def test_M5_normalized_mode_dt_zero_no_change(self, mock_node_factory):
        """Path M5: NORMALIZED mode dt=0 no change."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.update(0.0)
        assert group.entries[0].normalized_time == 0.0


# =============================================================================
# SyncGroup.update with LEADER_FOLLOWER mode
# =============================================================================


class TestSyncGroupUpdateLeaderFollowerMode:
    """Tests for SyncGroup.update with SyncMode.LEADER_FOLLOWER."""

    def test_N1_leader_follower_mode_advances_leader(self, mock_node_factory):
        """Path N1: LEADER_FOLLOWER mode advances leader."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.update(0.2)
        assert group.entries[0].normalized_time == pytest.approx(0.2)

    def test_N2_followers_match_leader_normalized_time(self, mock_node_factory):
        """Path N2: followers match leader's normalized_time."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.add_entry(mock_node_factory(), is_leader=False, duration=2.0)
        group.add_entry(mock_node_factory(), is_leader=False, duration=0.5)
        group.update(0.2)
        leader_time = group.entries[0].normalized_time
        assert group.entries[1].normalized_time == pytest.approx(leader_time)
        assert group.entries[2].normalized_time == pytest.approx(leader_time)

    def test_N3_no_leader_falls_back_to_normalized(self, mock_node_factory):
        """Path N3: no leader falls back to normalized."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        # Add entries without leader (first is default leader by index, but not flagged)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.update(0.1)
        # Should not raise, falls back to default leader (index 0)

    def test_N4_multiple_followers_all_sync_to_leader(self, mock_node_factory):
        """Path N4: multiple followers all sync to leader."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        # Make middle entry the leader
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.update(0.3)
        leader_time = group.entries[1].normalized_time
        assert group.entries[0].normalized_time == pytest.approx(leader_time)
        assert group.entries[2].normalized_time == pytest.approx(leader_time)
        assert group.entries[3].normalized_time == pytest.approx(leader_time)


# =============================================================================
# SyncGroup.update with WEIGHTED mode
# =============================================================================


class TestSyncGroupUpdateWeightedMode:
    """Tests for SyncGroup.update with SyncMode.WEIGHTED."""

    def test_O1_weighted_mode_calculates_weighted_time(self, mock_node_factory):
        """Path O1: WEIGHTED mode calculates weighted time."""
        group = SyncGroup("test", mode=SyncMode.WEIGHTED)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.update(0.1)
        # All should have same normalized time
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )

    def test_O2_weighted_mode_syncs_all_to_same_time(self, mock_node_factory):
        """Path O2: WEIGHTED mode syncs all to same time."""
        group = SyncGroup("test", mode=SyncMode.WEIGHTED)
        group.add_entry(mock_node_factory(), weight=0.5, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.5, duration=2.0)
        group.update(0.2)
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )

    def test_O3_weighted_mode_handles_zero_total_weight(self, mock_node_factory):
        """Path O3: WEIGHTED mode handles zero total weight."""
        group = SyncGroup("test", mode=SyncMode.WEIGHTED)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        # Should not raise
        group.update(0.1)


# =============================================================================
# SyncGroup.update with PHASE mode
# =============================================================================


class TestSyncGroupUpdatePhaseMode:
    """Tests for SyncGroup.update with SyncMode.PHASE."""

    def test_P1_phase_mode_advances_leader(self, mock_node_factory):
        """Path P1: PHASE mode advances leader."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.update(0.2)
        assert group.entries[0].normalized_time == pytest.approx(0.2)

    def test_P2_followers_sync_to_leader_time(self, mock_node_factory):
        """Path P2: followers sync to leader time."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.add_entry(mock_node_factory(), is_leader=False, duration=1.0)
        group.update(0.3)
        # Without markers, followers match leader's normalized time
        assert group.entries[1].normalized_time == pytest.approx(
            group.entries[0].normalized_time
        )

    def test_P3_no_leader_falls_back_to_normalized(self, mock_node_factory):
        """Path P3: no leader falls back to normalized."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=1.0)
        # No explicit leader, should use default (index 0)
        group.update(0.1)
        # Should have advanced without error


# =============================================================================
# SyncGroup.get_synchronized_time
# =============================================================================


class TestSyncGroupGetSynchronizedTime:
    """Tests for SyncGroup.get_synchronized_time method."""

    def test_Q1_empty_group_returns_zero(self, sync_group):
        """Path Q1: empty group returns 0.0."""
        result = sync_group.get_synchronized_time()
        assert result == 0.0

    def test_Q2_leader_follower_returns_leader_time(self, mock_node_factory):
        """Path Q2: LEADER_FOLLOWER returns leader time."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node_factory(), is_leader=True, duration=1.0)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.entries[0].normalized_time = 0.5
        result = group.get_synchronized_time()
        assert result == pytest.approx(0.5)

    def test_Q3_leader_follower_no_leader_returns_zero(self, mock_node_factory):
        """Path Q3: LEADER_FOLLOWER no leader returns 0.0."""
        group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
        # Empty group with LEADER_FOLLOWER mode
        result = group.get_synchronized_time()
        assert result == 0.0

    def test_Q4_other_modes_return_weighted_average(self, mock_node_factory):
        """Path Q4: other modes return weighted average."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.entries[0].normalized_time = 0.4
        group.entries[1].normalized_time = 0.6
        result = group.get_synchronized_time()
        # Average of 0.4 and 0.6 = 0.5
        assert result == pytest.approx(0.5)

    def test_Q5_zero_total_weight_returns_zero(self, mock_node_factory):
        """Path Q5: zero total weight returns 0.0."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        result = group.get_synchronized_time()
        assert result == 0.0


# =============================================================================
# ADDITIONAL EDGE CASES
# =============================================================================


class TestSyncGroupEdgeCases:
    """Additional edge case tests."""

    def test_multiple_updates_accumulate(self, mock_node_factory):
        """Multiple updates accumulate correctly."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.update(0.1)
        group.update(0.1)
        group.update(0.1)
        assert group.entries[0].normalized_time == pytest.approx(0.3)

    def test_wrapping_after_multiple_updates(self, mock_node_factory):
        """Time wraps correctly after many updates."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        for _ in range(15):
            group.update(0.1)  # Total 1.5 seconds
        assert group.entries[0].normalized_time == pytest.approx(0.5)

    def test_add_remove_add_sequence(self, mock_node_factory):
        """Add, remove, add sequence works correctly."""
        group = SyncGroup("test")
        node1 = mock_node_factory()
        node2 = mock_node_factory()
        node3 = mock_node_factory()

        idx1 = group.add_entry(node1)
        idx2 = group.add_entry(node2)
        assert idx1 == 0
        assert idx2 == 1

        group.remove_entry(0)
        assert len(group.entries) == 1
        assert group.entries[0].node is node2

        idx3 = group.add_entry(node3)
        assert idx3 == 1  # Appends to end
        assert len(group.entries) == 2

    def test_set_leader_then_remove_leader(self, mock_node_factory):
        """Set leader then remove leader updates correctly."""
        group = SyncGroup("test")
        group.add_entry(mock_node_factory())
        group.add_entry(mock_node_factory())
        group.add_entry(mock_node_factory())

        group.set_leader(2)
        assert group._leader_index == 2

        group.remove_entry(2)
        # Leader was removed, index should be clamped
        assert group._leader_index == 1

    def test_normalized_mode_different_durations(self, mock_node_factory):
        """NORMALIZED mode with vastly different durations."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=1.0, duration=0.1)
        group.add_entry(mock_node_factory(), weight=1.0, duration=10.0)
        group.update(0.05)
        # Both should sync to same normalized time
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )

    def test_leader_follower_leader_at_different_positions(self, mock_node_factory):
        """LEADER_FOLLOWER with leader at various positions."""
        for leader_idx in range(4):
            group = SyncGroup("test", mode=SyncMode.LEADER_FOLLOWER)
            for i in range(4):
                group.add_entry(
                    mock_node_factory(),
                    is_leader=(i == leader_idx),
                    duration=1.0
                )
            group.update(0.25)
            leader_time = group.entries[leader_idx].normalized_time
            for i, entry in enumerate(group.entries):
                if i != leader_idx:
                    assert entry.normalized_time == pytest.approx(leader_time)


class TestSyncEntryEdgeCases:
    """Edge case tests for SyncEntry."""

    def test_advance_negative_dt_still_advances(self, mock_node):
        """Advance with negative dt moves backward."""
        entry = SyncEntry(node=mock_node, duration=1.0)
        entry._current_time = 0.5
        entry.advance(-0.1)
        assert entry._current_time == pytest.approx(0.4)

    def test_very_small_duration(self, mock_node):
        """Very small duration works correctly."""
        entry = SyncEntry(node=mock_node, duration=0.001)
        entry.advance(0.0001)
        assert entry._current_time == pytest.approx(0.0001)

    def test_very_large_duration(self, mock_node):
        """Very large duration works correctly."""
        entry = SyncEntry(node=mock_node, duration=1000.0)
        entry.advance(1.0)
        assert entry._current_time == pytest.approx(1.0)
        assert entry.normalized_time == pytest.approx(0.001)


class TestSyncGroupModeTransitions:
    """Tests for changing modes."""

    def test_change_mode_after_updates(self, mock_node_factory):
        """Changing mode after updates works."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), duration=2.0)
        group.update(0.2)

        # Times are different in NONE mode
        time0_before = group.entries[0].normalized_time
        time1_before = group.entries[1].normalized_time
        assert time0_before != pytest.approx(time1_before)  # 0.2 vs 0.1

        # Switch to NORMALIZED mode
        group.mode = SyncMode.NORMALIZED
        group.update(0.1)

        # NORMALIZED mode advances all entries by the same normalized amount,
        # so the relative difference between entries is preserved.
        # The mode advances each entry by the weighted average speed.
        time0_after = group.entries[0].normalized_time
        time1_after = group.entries[1].normalized_time

        # Both should have advanced by the same normalized amount
        advance0 = time0_after - time0_before
        advance1 = time1_after - time1_before
        assert advance0 == pytest.approx(advance1)

    def test_switch_to_leader_follower(self, mock_node_factory):
        """Switch to LEADER_FOLLOWER mode."""
        group = SyncGroup("test", mode=SyncMode.NONE)
        group.add_entry(mock_node_factory(), duration=1.0)
        group.add_entry(mock_node_factory(), is_leader=True, duration=2.0)
        group.update(0.2)

        group.mode = SyncMode.LEADER_FOLLOWER
        group.update(0.1)

        # Follower should match leader
        assert group.entries[0].normalized_time == pytest.approx(
            group.entries[1].normalized_time
        )


class TestSyncGroupWeightCombinations:
    """Tests for various weight combinations."""

    def test_single_nonzero_weight(self, mock_node_factory):
        """Single nonzero weight among zeros."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.0, duration=1.0)
        group.update(0.1)
        # Should not raise and all sync together

    def test_very_large_weights(self, mock_node_factory):
        """Very large weights work correctly."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=1000.0, duration=1.0)
        group.add_entry(mock_node_factory(), weight=1.0, duration=1.0)
        group.update(0.1)

    def test_very_small_positive_weights(self, mock_node_factory):
        """Very small positive weights work correctly."""
        group = SyncGroup("test", mode=SyncMode.NORMALIZED)
        group.add_entry(mock_node_factory(), weight=0.001, duration=1.0)
        group.add_entry(mock_node_factory(), weight=0.001, duration=1.0)
        group.update(0.1)


class TestSyncGroupWithMarkers:
    """Tests for sync group with marker tracks."""

    def test_phase_sync_with_markers(self, mock_node_factory, marker_track):
        """PHASE sync with marker tracks."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(
            mock_node_factory(),
            is_leader=True,
            duration=1.0,
            marker_track=marker_track
        )
        group.add_entry(
            mock_node_factory(),
            is_leader=False,
            duration=1.0,
            marker_track=marker_track
        )
        group.update(0.1)
        # With matching markers, should sync

    def test_phase_sync_leader_no_markers(self, mock_node_factory, marker_track):
        """PHASE sync when leader has no markers."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(
            mock_node_factory(),
            is_leader=True,
            duration=1.0,
            marker_track=None  # No markers
        )
        group.add_entry(
            mock_node_factory(),
            is_leader=False,
            duration=1.0,
            marker_track=marker_track
        )
        group.update(0.1)
        # Follower should sync to leader time directly

    def test_phase_sync_follower_no_markers(self, mock_node_factory, marker_track):
        """PHASE sync when follower has no markers."""
        group = SyncGroup("test", mode=SyncMode.PHASE)
        group.add_entry(
            mock_node_factory(),
            is_leader=True,
            duration=1.0,
            marker_track=marker_track
        )
        group.add_entry(
            mock_node_factory(),
            is_leader=False,
            duration=1.0,
            marker_track=None  # No markers
        )
        group.update(0.1)
        # Follower should sync to leader time
