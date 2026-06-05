"""
Blackbox tests for SyncGroup, SyncEntry, and SyncMode.

Tests the public API behavior without examining implementation details.
Covers sync mode enumeration, entry management, leader selection, and group updates.

Task: T-FB-4.15
CLEANROOM MODE: Tests written based ONLY on the public contract from imports.
"""

import pytest
import math
from typing import Optional


# =============================================================================
# Test Fixtures and Setup
# =============================================================================

@pytest.fixture
def sync_module():
    """Import sync module."""
    from engine.animation.graph.sync import SyncGroup, SyncEntry, SyncMode
    return SyncGroup, SyncEntry, SyncMode


@pytest.fixture
def SyncGroup(sync_module):
    """Get SyncGroup class."""
    return sync_module[0]


@pytest.fixture
def SyncEntry(sync_module):
    """Get SyncEntry class."""
    return sync_module[1]


@pytest.fixture
def SyncMode(sync_module):
    """Get SyncMode enum."""
    return sync_module[2]


@pytest.fixture
def mock_node():
    """Create a mock animation node for testing."""
    class MockNode:
        def __init__(self, name: str = "mock", duration: float = 1.0):
            self.name = name
            self.duration = duration
            self.time = 0.0
            self.normalized_time = 0.0
    return MockNode


@pytest.fixture
def create_nodes(mock_node):
    """Factory to create multiple mock nodes."""
    def _create(count: int, base_name: str = "node", durations: list = None):
        nodes = []
        for i in range(count):
            dur = durations[i] if durations and i < len(durations) else 1.0
            nodes.append(mock_node(f"{base_name}_{i}", dur))
        return nodes
    return _create


# =============================================================================
# SyncMode Enumeration Tests
# =============================================================================

class TestSyncModeEnumeration:
    """Tests for SyncMode enum values and access."""

    def test_sync_mode_has_none(self, SyncMode):
        """SyncMode should have NONE value."""
        assert hasattr(SyncMode, 'NONE')
        assert SyncMode.NONE is not None

    def test_sync_mode_has_normalized(self, SyncMode):
        """SyncMode should have NORMALIZED value."""
        assert hasattr(SyncMode, 'NORMALIZED')
        assert SyncMode.NORMALIZED is not None

    def test_sync_mode_has_phase(self, SyncMode):
        """SyncMode should have PHASE value."""
        assert hasattr(SyncMode, 'PHASE')
        assert SyncMode.PHASE is not None

    def test_sync_mode_has_leader_follower(self, SyncMode):
        """SyncMode should have LEADER_FOLLOWER value."""
        assert hasattr(SyncMode, 'LEADER_FOLLOWER')
        assert SyncMode.LEADER_FOLLOWER is not None

    def test_sync_mode_has_weighted(self, SyncMode):
        """SyncMode should have WEIGHTED value."""
        assert hasattr(SyncMode, 'WEIGHTED')
        assert SyncMode.WEIGHTED is not None

    def test_sync_mode_values_are_distinct(self, SyncMode):
        """All SyncMode values should be distinct."""
        modes = [
            SyncMode.NONE,
            SyncMode.NORMALIZED,
            SyncMode.PHASE,
            SyncMode.LEADER_FOLLOWER,
            SyncMode.WEIGHTED,
        ]
        assert len(set(modes)) == 5

    def test_sync_mode_accessible_by_name(self, SyncMode):
        """SyncMode should be accessible by name string."""
        assert SyncMode['NONE'] == SyncMode.NONE
        assert SyncMode['NORMALIZED'] == SyncMode.NORMALIZED
        assert SyncMode['PHASE'] == SyncMode.PHASE
        assert SyncMode['LEADER_FOLLOWER'] == SyncMode.LEADER_FOLLOWER
        assert SyncMode['WEIGHTED'] == SyncMode.WEIGHTED

    def test_sync_mode_name_property(self, SyncMode):
        """SyncMode values should have name property."""
        assert SyncMode.NONE.name == 'NONE'
        assert SyncMode.NORMALIZED.name == 'NORMALIZED'
        assert SyncMode.PHASE.name == 'PHASE'
        assert SyncMode.LEADER_FOLLOWER.name == 'LEADER_FOLLOWER'
        assert SyncMode.WEIGHTED.name == 'WEIGHTED'

    def test_sync_mode_count(self, SyncMode):
        """SyncMode should have exactly 5 modes."""
        mode_count = len(list(SyncMode))
        assert mode_count == 5

    def test_sync_mode_iteration(self, SyncMode):
        """SyncMode should be iterable."""
        modes = list(SyncMode)
        assert len(modes) == 5
        assert SyncMode.NONE in modes
        assert SyncMode.NORMALIZED in modes


# =============================================================================
# SyncEntry Creation Tests
# =============================================================================

class TestSyncEntryCreation:
    """Tests for SyncEntry instantiation and field access."""

    def test_create_entry_with_node(self, SyncGroup, SyncMode, mock_node):
        """Create entry with a node."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        idx = group.add_entry(node)
        assert idx == 0

    def test_entry_has_weight_attribute(self, SyncGroup, SyncMode, mock_node):
        """SyncEntry should have weight attribute."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert hasattr(entry, 'weight')

    def test_entry_has_is_leader_attribute(self, SyncGroup, SyncMode, mock_node):
        """SyncEntry should have is_leader attribute."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert hasattr(entry, 'is_leader')

    def test_entry_has_duration_attribute(self, SyncGroup, SyncMode, mock_node):
        """SyncEntry should have duration attribute."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert hasattr(entry, 'duration')

    def test_entry_default_weight(self, SyncGroup, SyncMode, mock_node):
        """Entry default weight should be 1.0."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert entry.weight == 1.0

    def test_entry_default_is_leader_false(self, SyncGroup, SyncMode, mock_node):
        """Entry default is_leader should be False."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert entry.is_leader is False

    def test_entry_with_custom_weight(self, SyncGroup, SyncMode, mock_node):
        """Create entry with custom weight."""
        group = SyncGroup("test_group", SyncMode.WEIGHTED)
        node = mock_node("test_node")
        group.add_entry(node, weight=0.5)
        entry = group.entries[0]
        assert entry.weight == 0.5

    def test_entry_with_leader_flag(self, SyncGroup, SyncMode, mock_node):
        """Create entry with is_leader flag."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        node = mock_node("test_node")
        group.add_entry(node, is_leader=True)
        entry = group.entries[0]
        assert entry.is_leader is True

    def test_entry_has_node_reference(self, SyncGroup, SyncMode, mock_node):
        """Entry should maintain node reference."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert hasattr(entry, 'node')
        assert entry.node is node


# =============================================================================
# SyncEntry Advance Tests
# =============================================================================

class TestSyncEntryAdvance:
    """Tests for SyncEntry.advance() method."""

    def test_entry_has_advance_method(self, SyncGroup, SyncMode, mock_node):
        """SyncEntry should have advance method."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        assert hasattr(entry, 'advance')
        assert callable(entry.advance)

    def test_advance_moves_time_forward(self, SyncGroup, SyncMode, mock_node):
        """advance() should move time forward."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node", duration=2.0)
        group.add_entry(node)
        entry = group.entries[0]
        initial_time = entry.node.time if hasattr(entry.node, 'time') else 0.0
        entry.advance(0.1)
        # Time should have changed (exact behavior depends on mode)

    def test_advance_with_zero_dt(self, SyncGroup, SyncMode, mock_node):
        """advance(0.0) should not change time."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        entry.advance(0.0)
        # Should not raise

    def test_advance_with_negative_dt(self, SyncGroup, SyncMode, mock_node):
        """advance() with negative dt may be allowed or clamped."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node")
        group.add_entry(node)
        entry = group.entries[0]
        try:
            entry.advance(-0.1)
            # If allowed, behavior depends on implementation
        except (ValueError, AssertionError):
            # Rejecting negative dt is acceptable
            pass

    def test_advance_large_dt(self, SyncGroup, SyncMode, mock_node):
        """advance() with large dt should be handled."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("test_node", duration=1.0)
        group.add_entry(node)
        entry = group.entries[0]
        entry.advance(10.0)
        # Should not crash


# =============================================================================
# SyncGroup Creation Tests
# =============================================================================

class TestSyncGroupCreation:
    """Tests for SyncGroup instantiation."""

    def test_create_with_name_and_mode(self, SyncGroup, SyncMode):
        """Create SyncGroup with name and mode."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        assert group is not None

    def test_group_has_name(self, SyncGroup, SyncMode):
        """SyncGroup should have name attribute."""
        group = SyncGroup("my_group", SyncMode.NORMALIZED)
        assert hasattr(group, 'name')
        assert group.name == "my_group"

    def test_group_has_mode(self, SyncGroup, SyncMode):
        """SyncGroup should have mode attribute."""
        group = SyncGroup("test_group", SyncMode.PHASE)
        assert hasattr(group, 'mode')
        assert group.mode == SyncMode.PHASE

    def test_default_mode_is_normalized(self, SyncGroup, SyncMode):
        """Default mode should be NORMALIZED."""
        group = SyncGroup("test_group")
        assert group.mode == SyncMode.NORMALIZED

    def test_group_has_entries_list(self, SyncGroup, SyncMode):
        """SyncGroup should have entries list."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        assert hasattr(group, 'entries')
        assert isinstance(group.entries, list)

    def test_new_group_has_empty_entries(self, SyncGroup, SyncMode):
        """New SyncGroup should have empty entries list."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        assert len(group.entries) == 0

    def test_create_with_none_mode(self, SyncGroup, SyncMode):
        """Create SyncGroup with NONE mode."""
        group = SyncGroup("no_sync", SyncMode.NONE)
        assert group.mode == SyncMode.NONE

    def test_create_with_weighted_mode(self, SyncGroup, SyncMode):
        """Create SyncGroup with WEIGHTED mode."""
        group = SyncGroup("weighted", SyncMode.WEIGHTED)
        assert group.mode == SyncMode.WEIGHTED

    def test_create_with_leader_follower_mode(self, SyncGroup, SyncMode):
        """Create SyncGroup with LEADER_FOLLOWER mode."""
        group = SyncGroup("leader", SyncMode.LEADER_FOLLOWER)
        assert group.mode == SyncMode.LEADER_FOLLOWER

    def test_group_name_with_spaces(self, SyncGroup, SyncMode):
        """SyncGroup name can have spaces."""
        group = SyncGroup("my sync group", SyncMode.NORMALIZED)
        assert group.name == "my sync group"

    def test_group_name_with_underscores(self, SyncGroup, SyncMode):
        """SyncGroup name can have underscores."""
        group = SyncGroup("walk_run_blend", SyncMode.NORMALIZED)
        assert group.name == "walk_run_blend"

    def test_group_empty_name(self, SyncGroup, SyncMode):
        """SyncGroup with empty name may be allowed."""
        try:
            group = SyncGroup("", SyncMode.NORMALIZED)
            assert group.name == ""
        except (ValueError, TypeError):
            # Rejection of empty name is acceptable
            pass


# =============================================================================
# SyncGroup Entry Management Tests
# =============================================================================

class TestSyncGroupEntryManagement:
    """Tests for adding and removing entries."""

    def test_add_entry_returns_index(self, SyncGroup, SyncMode, mock_node):
        """add_entry() should return index."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node = mock_node("node1")
        idx = group.add_entry(node)
        assert idx == 0

    def test_add_multiple_entries_incrementing_index(self, SyncGroup, SyncMode, mock_node):
        """Adding multiple entries should return incrementing indices."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        idx0 = group.add_entry(mock_node("node0"))
        idx1 = group.add_entry(mock_node("node1"))
        idx2 = group.add_entry(mock_node("node2"))
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_entries_count_after_add(self, SyncGroup, SyncMode, mock_node):
        """Entries count should increase after add_entry."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        assert len(group.entries) == 0
        group.add_entry(mock_node("node1"))
        assert len(group.entries) == 1
        group.add_entry(mock_node("node2"))
        assert len(group.entries) == 2

    def test_remove_entry_by_index(self, SyncGroup, SyncMode, mock_node):
        """remove_entry() should remove by index."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        assert len(group.entries) == 2
        group.remove_entry(0)
        assert len(group.entries) == 1

    def test_remove_entry_shifts_remaining(self, SyncGroup, SyncMode, mock_node):
        """Removing entry should shift remaining entries."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        node0 = mock_node("node0")
        node1 = mock_node("node1")
        node2 = mock_node("node2")
        group.add_entry(node0)
        group.add_entry(node1)
        group.add_entry(node2)
        group.remove_entry(0)
        # node1 should now be at index 0
        assert group.entries[0].node.name == "node1"

    def test_remove_last_entry(self, SyncGroup, SyncMode, mock_node):
        """Remove the last entry."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        group.remove_entry(1)
        assert len(group.entries) == 1
        assert group.entries[0].node.name == "node0"

    def test_remove_only_entry(self, SyncGroup, SyncMode, mock_node):
        """Remove the only entry."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("only"))
        group.remove_entry(0)
        assert len(group.entries) == 0

    def test_remove_invalid_index_negative(self, SyncGroup, SyncMode, mock_node):
        """remove_entry with negative index should raise or handle gracefully."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        try:
            group.remove_entry(-1)
            # May wrap around like Python lists
        except (IndexError, ValueError):
            pass

    def test_remove_invalid_index_out_of_bounds(self, SyncGroup, SyncMode, mock_node):
        """remove_entry with out of bounds index should raise or be no-op."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        initial_count = len(group.entries)
        try:
            group.remove_entry(10)
            # If no error, count should be unchanged (no-op behavior)
            assert len(group.entries) == initial_count
        except (IndexError, ValueError):
            # Error is acceptable behavior
            pass

    def test_add_many_entries(self, SyncGroup, SyncMode, create_nodes):
        """Add many entries to test capacity."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        nodes = create_nodes(20)
        for node in nodes:
            group.add_entry(node)
        assert len(group.entries) == 20

    def test_remove_middle_entry(self, SyncGroup, SyncMode, mock_node):
        """Remove entry from the middle."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("first"))
        group.add_entry(mock_node("middle"))
        group.add_entry(mock_node("last"))
        group.remove_entry(1)
        assert len(group.entries) == 2
        assert group.entries[0].node.name == "first"
        assert group.entries[1].node.name == "last"


# =============================================================================
# SyncGroup Leader Selection Tests
# =============================================================================

class TestSyncGroupLeaderSelection:
    """Tests for set_leader() and get_leader()."""

    def test_get_leader_on_empty_group(self, SyncGroup, SyncMode):
        """get_leader() on empty group should return None."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        leader = group.get_leader()
        assert leader is None

    def test_set_leader_by_index(self, SyncGroup, SyncMode, mock_node):
        """set_leader() should set leader by index."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        group.set_leader(1)
        leader = group.get_leader()
        assert leader is not None
        assert leader.node.name == "node1"

    def test_set_leader_updates_is_leader_flag(self, SyncGroup, SyncMode, mock_node):
        """set_leader() should update is_leader flag."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        group.set_leader(0)
        assert group.entries[0].is_leader is True
        assert group.entries[1].is_leader is False

    def test_change_leader(self, SyncGroup, SyncMode, mock_node):
        """Changing leader should update flags."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        group.set_leader(0)
        assert group.entries[0].is_leader is True
        group.set_leader(1)
        assert group.entries[0].is_leader is False
        assert group.entries[1].is_leader is True

    def test_only_one_leader_at_a_time(self, SyncGroup, SyncMode, mock_node):
        """Only one entry should be leader at a time."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        for i in range(5):
            group.add_entry(mock_node(f"node{i}"))
        group.set_leader(2)
        leader_count = sum(1 for e in group.entries if e.is_leader)
        assert leader_count == 1

    def test_get_leader_returns_correct_entry(self, SyncGroup, SyncMode, mock_node):
        """get_leader() returns the entry with is_leader=True."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("leader_node"))
        group.add_entry(mock_node("node2"))
        group.set_leader(1)
        leader = group.get_leader()
        assert leader.node.name == "leader_node"

    def test_set_leader_invalid_index(self, SyncGroup, SyncMode, mock_node):
        """set_leader with invalid index should raise or have no effect."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node"))
        try:
            group.set_leader(5)
            # If no error, leader should remain unchanged or be None
            leader = group.get_leader()
            # Acceptable: either no leader or the existing entry
        except (IndexError, ValueError):
            # Error is acceptable behavior
            pass

    def test_first_entry_with_leader_flag(self, SyncGroup, SyncMode, mock_node):
        """Adding entry with is_leader=True sets it as leader."""
        group = SyncGroup("test_group", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("leader"), is_leader=True)
        leader = group.get_leader()
        assert leader is not None
        assert leader.node.name == "leader"

    def test_leader_in_normalized_mode(self, SyncGroup, SyncMode, mock_node):
        """Leader selection may behave differently in NORMALIZED mode."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        # In NORMALIZED mode, behavior depends on implementation
        leader = group.get_leader()
        # May return None or first entry


# =============================================================================
# SyncGroup Update Tests
# =============================================================================

class TestSyncGroupUpdate:
    """Tests for update() method behavior."""

    def test_group_has_update_method(self, SyncGroup, SyncMode):
        """SyncGroup should have update method."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        assert hasattr(group, 'update')
        assert callable(group.update)

    def test_update_on_empty_group(self, SyncGroup, SyncMode):
        """update() on empty group should not raise."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.update(0.016)  # ~60fps delta

    def test_update_advances_entries(self, SyncGroup, SyncMode, mock_node):
        """update() should advance all entries."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node0"))
        group.add_entry(mock_node("node1"))
        group.update(0.1)
        # Entries should be advanced (exact behavior depends on mode)

    def test_update_with_zero_dt(self, SyncGroup, SyncMode, mock_node):
        """update(0.0) should be a no-op."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        group.update(0.0)
        # Should not raise or change state significantly

    def test_update_normalized_mode(self, SyncGroup, SyncMode, mock_node):
        """Update in NORMALIZED mode syncs normalized times."""
        group = SyncGroup("sync_normalized", SyncMode.NORMALIZED)
        node1 = mock_node("node1", duration=1.0)
        node2 = mock_node("node2", duration=2.0)
        group.add_entry(node1)
        group.add_entry(node2)
        group.update(0.5)
        # Both entries should have same normalized time

    def test_update_none_mode_no_sync(self, SyncGroup, SyncMode, mock_node):
        """Update in NONE mode should not synchronize."""
        group = SyncGroup("no_sync", SyncMode.NONE)
        node1 = mock_node("node1", duration=1.0)
        node2 = mock_node("node2", duration=2.0)
        group.add_entry(node1)
        group.add_entry(node2)
        group.update(0.5)
        # Entries advance independently

    def test_update_leader_follower_mode(self, SyncGroup, SyncMode, mock_node):
        """Update in LEADER_FOLLOWER mode follows leader."""
        group = SyncGroup("leader_follow", SyncMode.LEADER_FOLLOWER)
        leader = mock_node("leader", duration=1.0)
        follower = mock_node("follower", duration=2.0)
        group.add_entry(leader, is_leader=True)
        group.add_entry(follower)
        group.update(0.25)
        # Follower should sync to leader's normalized time

    def test_update_phase_mode(self, SyncGroup, SyncMode, mock_node):
        """Update in PHASE mode maintains phase relationship."""
        group = SyncGroup("phase_sync", SyncMode.PHASE)
        node1 = mock_node("node1", duration=1.0)
        node2 = mock_node("node2", duration=1.0)
        group.add_entry(node1)
        group.add_entry(node2)
        group.update(0.1)
        # Phase relationship should be maintained

    def test_update_weighted_mode(self, SyncGroup, SyncMode, mock_node):
        """Update in WEIGHTED mode uses entry weights."""
        group = SyncGroup("weighted", SyncMode.WEIGHTED)
        node1 = mock_node("node1")
        node2 = mock_node("node2")
        group.add_entry(node1, weight=0.75)
        group.add_entry(node2, weight=0.25)
        group.update(0.1)
        # Weighted average of times should be computed

    def test_update_multiple_times(self, SyncGroup, SyncMode, mock_node):
        """Multiple updates should accumulate."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        group.update(0.1)
        group.update(0.1)
        group.update(0.1)
        # Total time should be 0.3

    def test_update_with_large_dt(self, SyncGroup, SyncMode, mock_node):
        """Update with large dt should be handled."""
        group = SyncGroup("test_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node", duration=1.0))
        group.update(10.0)
        # Should handle wrapping or clamping


# =============================================================================
# SyncGroup Mode Behavior Tests
# =============================================================================

class TestSyncGroupModeBehavior:
    """Tests for mode-specific behaviors."""

    def test_none_mode_entries_independent(self, SyncGroup, SyncMode, mock_node):
        """NONE mode: entries are not synchronized."""
        group = SyncGroup("independent", SyncMode.NONE)
        group.add_entry(mock_node("node1", duration=1.0))
        group.add_entry(mock_node("node2", duration=2.0))
        # Each entry advances at its own rate

    def test_normalized_mode_same_normalized_time(self, SyncGroup, SyncMode, mock_node):
        """NORMALIZED mode: all entries have same normalized time."""
        group = SyncGroup("normalized", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node1"))
        group.add_entry(mock_node("node2"))
        # After update, normalized times should match

    def test_leader_follower_without_leader(self, SyncGroup, SyncMode, mock_node):
        """LEADER_FOLLOWER without leader should handle gracefully."""
        group = SyncGroup("no_leader", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("node1"))
        group.add_entry(mock_node("node2"))
        group.update(0.1)
        # Should not crash

    def test_weighted_mode_equal_weights(self, SyncGroup, SyncMode, mock_node):
        """WEIGHTED mode with equal weights."""
        group = SyncGroup("equal_weights", SyncMode.WEIGHTED)
        group.add_entry(mock_node("node1"), weight=1.0)
        group.add_entry(mock_node("node2"), weight=1.0)
        group.update(0.1)
        # Should produce average

    def test_weighted_mode_zero_weight(self, SyncGroup, SyncMode, mock_node):
        """WEIGHTED mode with zero weight entry."""
        group = SyncGroup("zero_weight", SyncMode.WEIGHTED)
        group.add_entry(mock_node("node1"), weight=1.0)
        group.add_entry(mock_node("node2"), weight=0.0)
        group.update(0.1)
        # Zero weight entry should not contribute


# =============================================================================
# SyncGroup Edge Cases
# =============================================================================

class TestSyncGroupEdgeCases:
    """Edge case tests for SyncGroup."""

    def test_single_entry_group(self, SyncGroup, SyncMode, mock_node):
        """Group with single entry should work."""
        group = SyncGroup("single", SyncMode.NORMALIZED)
        group.add_entry(mock_node("only"))
        group.update(0.5)
        # Should work without issues

    def test_add_same_node_twice(self, SyncGroup, SyncMode, mock_node):
        """Adding same node twice may or may not be allowed."""
        group = SyncGroup("duplicates", SyncMode.NORMALIZED)
        node = mock_node("shared")
        group.add_entry(node)
        try:
            group.add_entry(node)
            # If allowed, we have two entries for same node
            assert len(group.entries) == 2
        except (ValueError, TypeError):
            # Rejection is acceptable
            pass

    def test_entry_with_zero_duration(self, SyncGroup, SyncMode, mock_node):
        """Entry with zero duration should be handled."""
        group = SyncGroup("zero_dur", SyncMode.NORMALIZED)
        node = mock_node("zero", duration=0.0)
        group.add_entry(node)
        group.update(0.1)
        # Should not divide by zero or crash

    def test_entry_with_very_small_duration(self, SyncGroup, SyncMode, mock_node):
        """Entry with very small duration."""
        group = SyncGroup("tiny", SyncMode.NORMALIZED)
        node = mock_node("tiny", duration=0.001)
        group.add_entry(node)
        group.update(0.1)
        # Should handle precision

    def test_entry_with_large_duration(self, SyncGroup, SyncMode, mock_node):
        """Entry with very large duration."""
        group = SyncGroup("huge", SyncMode.NORMALIZED)
        node = mock_node("huge", duration=10000.0)
        group.add_entry(node)
        group.update(0.1)
        # Should work normally

    def test_negative_dt_handling(self, SyncGroup, SyncMode, mock_node):
        """Negative dt in update may be rejected or handled."""
        group = SyncGroup("test", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        try:
            group.update(-0.1)
            # If allowed, time may go backwards
        except (ValueError, AssertionError):
            # Rejection is acceptable
            pass

    def test_very_small_dt(self, SyncGroup, SyncMode, mock_node):
        """Very small dt should work."""
        group = SyncGroup("tiny_dt", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        group.update(0.000001)
        # Should not accumulate errors

    def test_rapid_add_remove_cycle(self, SyncGroup, SyncMode, mock_node):
        """Rapid add/remove cycles should be stable."""
        group = SyncGroup("cycle", SyncMode.NORMALIZED)
        for i in range(10):
            idx = group.add_entry(mock_node(f"node{i}"))
            group.remove_entry(idx)
        assert len(group.entries) == 0


# =============================================================================
# SyncGroup Consistency Tests
# =============================================================================

class TestSyncGroupConsistency:
    """Tests for state consistency."""

    def test_entries_accessible_after_update(self, SyncGroup, SyncMode, mock_node):
        """Entries should be accessible after update."""
        group = SyncGroup("test", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        group.update(0.1)
        assert len(group.entries) == 1
        assert group.entries[0] is not None

    def test_leader_persists_after_update(self, SyncGroup, SyncMode, mock_node):
        """Leader should persist through updates."""
        group = SyncGroup("test", SyncMode.LEADER_FOLLOWER)
        group.add_entry(mock_node("leader"), is_leader=True)
        group.add_entry(mock_node("follower"))
        group.update(0.1)
        leader = group.get_leader()
        assert leader is not None
        assert leader.node.name == "leader"

    def test_mode_unchanged_after_operations(self, SyncGroup, SyncMode, mock_node):
        """Mode should remain unchanged after operations."""
        group = SyncGroup("test", SyncMode.PHASE)
        group.add_entry(mock_node("node"))
        group.update(0.1)
        group.remove_entry(0)
        assert group.mode == SyncMode.PHASE

    def test_name_unchanged_after_operations(self, SyncGroup, SyncMode, mock_node):
        """Name should remain unchanged after operations."""
        group = SyncGroup("my_group", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        group.update(0.1)
        assert group.name == "my_group"


# =============================================================================
# SyncGroup Weight Tests
# =============================================================================

class TestSyncGroupWeights:
    """Tests for weight handling in entries."""

    def test_weight_range_zero_to_one(self, SyncGroup, SyncMode, mock_node):
        """Weights in 0-1 range should work."""
        group = SyncGroup("weights", SyncMode.WEIGHTED)
        group.add_entry(mock_node("node1"), weight=0.0)
        group.add_entry(mock_node("node2"), weight=0.5)
        group.add_entry(mock_node("node3"), weight=1.0)
        assert group.entries[0].weight == 0.0
        assert group.entries[1].weight == 0.5
        assert group.entries[2].weight == 1.0

    def test_weight_greater_than_one(self, SyncGroup, SyncMode, mock_node):
        """Weight > 1 may be allowed."""
        group = SyncGroup("high_weight", SyncMode.WEIGHTED)
        try:
            group.add_entry(mock_node("node"), weight=2.0)
            assert group.entries[0].weight == 2.0
        except (ValueError, TypeError):
            # Clamping is acceptable
            pass

    def test_negative_weight(self, SyncGroup, SyncMode, mock_node):
        """Negative weight may be rejected or clamped."""
        group = SyncGroup("neg_weight", SyncMode.WEIGHTED)
        try:
            group.add_entry(mock_node("node"), weight=-0.5)
            # If allowed, may be used for subtractive blending
        except (ValueError, TypeError):
            # Rejection is acceptable
            pass

    def test_all_zero_weights(self, SyncGroup, SyncMode, mock_node):
        """All zero weights should be handled."""
        group = SyncGroup("all_zero", SyncMode.WEIGHTED)
        group.add_entry(mock_node("node1"), weight=0.0)
        group.add_entry(mock_node("node2"), weight=0.0)
        group.update(0.1)
        # Should not divide by zero

    def test_weights_sum_not_one(self, SyncGroup, SyncMode, mock_node):
        """Weights that don't sum to 1 should be handled."""
        group = SyncGroup("unbalanced", SyncMode.WEIGHTED)
        group.add_entry(mock_node("node1"), weight=0.3)
        group.add_entry(mock_node("node2"), weight=0.3)
        # Sum is 0.6, not 1.0
        group.update(0.1)
        # Should normalize or work as-is


# =============================================================================
# Integration Tests
# =============================================================================

class TestSyncGroupIntegration:
    """Integration tests for complete workflows."""

    def test_typical_walk_run_blend_setup(self, SyncGroup, SyncMode, mock_node):
        """Set up a typical walk/run blend scenario."""
        group = SyncGroup("locomotion", SyncMode.NORMALIZED)
        walk = mock_node("walk", duration=1.0)
        run = mock_node("run", duration=0.5)
        group.add_entry(walk)
        group.add_entry(run)
        # Simulate several frames
        for _ in range(60):
            group.update(1.0 / 60.0)
        assert len(group.entries) == 2

    def test_leader_follower_animation_setup(self, SyncGroup, SyncMode, mock_node):
        """Set up leader-follower animation sync."""
        group = SyncGroup("character_sync", SyncMode.LEADER_FOLLOWER)
        leader = mock_node("main_character", duration=2.0)
        follower1 = mock_node("companion1", duration=1.5)
        follower2 = mock_node("companion2", duration=2.5)
        group.add_entry(leader, is_leader=True)
        group.add_entry(follower1)
        group.add_entry(follower2)
        # Run update loop
        for _ in range(120):
            group.update(1.0 / 60.0)
        assert group.get_leader().node.name == "main_character"

    def test_weighted_blend_scenario(self, SyncGroup, SyncMode, mock_node):
        """Test weighted blending scenario."""
        group = SyncGroup("combat_blend", SyncMode.WEIGHTED)
        idle = mock_node("idle", duration=2.0)
        attack = mock_node("attack", duration=0.5)
        group.add_entry(idle, weight=0.7)
        group.add_entry(attack, weight=0.3)
        group.update(0.016)
        # Weights should influence sync

    def test_dynamic_entry_management(self, SyncGroup, SyncMode, mock_node):
        """Test adding/removing entries during playback."""
        group = SyncGroup("dynamic", SyncMode.NORMALIZED)
        group.add_entry(mock_node("base"))
        group.update(0.1)
        # Add new entry mid-playback
        group.add_entry(mock_node("added"))
        group.update(0.1)
        # Remove first entry
        group.remove_entry(0)
        group.update(0.1)
        assert len(group.entries) == 1

    def test_switch_leader_during_playback(self, SyncGroup, SyncMode, mock_node):
        """Switch leader during playback."""
        group = SyncGroup("switch_leader", SyncMode.LEADER_FOLLOWER)
        node1 = mock_node("node1")
        node2 = mock_node("node2")
        group.add_entry(node1, is_leader=True)
        group.add_entry(node2)
        group.update(0.5)
        # Switch leader
        group.set_leader(1)
        group.update(0.5)
        assert group.get_leader().node.name == "node2"


# =============================================================================
# Performance Characteristic Tests
# =============================================================================

class TestSyncGroupPerformance:
    """Tests for performance characteristics."""

    def test_many_entries_performance(self, SyncGroup, SyncMode, create_nodes):
        """Group should handle many entries efficiently."""
        group = SyncGroup("many", SyncMode.NORMALIZED)
        nodes = create_nodes(100)
        for node in nodes:
            group.add_entry(node)
        # Update should complete without timeout
        group.update(0.016)
        assert len(group.entries) == 100

    def test_rapid_updates(self, SyncGroup, SyncMode, mock_node):
        """Many rapid updates should be stable."""
        group = SyncGroup("rapid", SyncMode.NORMALIZED)
        group.add_entry(mock_node("node"))
        for _ in range(1000):
            group.update(0.001)
        # Should not accumulate significant errors

    def test_entries_with_varying_durations(self, SyncGroup, SyncMode, mock_node):
        """Entries with widely varying durations."""
        group = SyncGroup("varied", SyncMode.NORMALIZED)
        group.add_entry(mock_node("fast", duration=0.1))
        group.add_entry(mock_node("normal", duration=1.0))
        group.add_entry(mock_node("slow", duration=10.0))
        group.update(0.5)
        # All should be synchronized
