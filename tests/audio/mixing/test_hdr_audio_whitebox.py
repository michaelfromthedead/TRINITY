"""Whitebox tests for HDR Audio system.

Tests internal implementation of:
- MixWindow sliding audibility range
- AudioSource registration and tracking
- Priority-weighted loudness analysis
- Window center adaptation
- Protected source handling
- Gain adjustment calculations
"""

import time
from unittest.mock import MagicMock

import pytest

from engine.audio.mixing.config import (
    HDR_ACTIVE_THRESHOLD_DB,
    HDR_ADAPTATION_SPEED,
    HDR_CEILING_DB,
    HDR_DEFAULT_CENTER_DB,
    HDR_FLOOR_DB,
    HDR_PRIORITY_CRITICAL,
    HDR_PRIORITY_HIGH,
    HDR_PRIORITY_LOW,
    HDR_PRIORITY_NORMAL,
    HDR_WINDOW_DB,
    HDR_WINDOW_MAX_DB,
    HDR_WINDOW_MIN_DB,
    MIN_VOLUME_DB,
    db_to_linear,
)
from engine.audio.mixing.hdr_audio import (
    AudioSource,
    HDRAudioManager,
    HDRPriority,
    MixWindow,
)
from engine.audio.mixing.mix_bus import BusType, MixBus


# =============================================================================
# HDRPriority Tests
# =============================================================================


class TestHDRPriority:
    """Test HDRPriority enum."""

    def test_priority_values(self):
        """Priority values are correct."""
        assert HDRPriority.CRITICAL.value == HDR_PRIORITY_CRITICAL
        assert HDRPriority.HIGH.value == HDR_PRIORITY_HIGH
        assert HDRPriority.NORMAL.value == HDR_PRIORITY_NORMAL
        assert HDRPriority.LOW.value == HDR_PRIORITY_LOW

    def test_priority_ordering(self):
        """Priority values order correctly."""
        assert HDRPriority.CRITICAL.value > HDRPriority.HIGH.value
        assert HDRPriority.HIGH.value > HDRPriority.NORMAL.value
        assert HDRPriority.NORMAL.value > HDRPriority.LOW.value


# =============================================================================
# AudioSource Tests
# =============================================================================


class TestAudioSource:
    """Test AudioSource dataclass."""

    def test_default_values(self):
        """AudioSource has correct defaults."""
        source = AudioSource()
        assert source.name == ""
        assert source.bus is None
        assert source.priority == HDR_PRIORITY_NORMAL
        assert source.loudness_db == MIN_VOLUME_DB
        assert source.target_loudness_db == MIN_VOLUME_DB
        assert source.is_active is False
        assert source.is_protected is False
        assert source.id is not None

    def test_unique_ids(self):
        """Each AudioSource gets unique ID."""
        source1 = AudioSource()
        source2 = AudioSource()
        assert source1.id != source2.id

    def test_copy(self):
        """copy creates independent copy."""
        bus = MixBus("sfx", BusType.CATEGORY)
        source = AudioSource(
            name="test_source",
            bus=bus,
            priority=HDR_PRIORITY_HIGH,
            loudness_db=-20.0,
            target_loudness_db=-15.0,
            is_active=True,
            is_protected=True,
        )

        copy = source.copy()

        assert copy.id == source.id
        assert copy.name == "test_source"
        assert copy.bus is bus
        assert copy.priority == HDR_PRIORITY_HIGH
        assert copy.loudness_db == -20.0
        assert copy.target_loudness_db == -15.0
        assert copy.is_active is True
        assert copy.is_protected is True


# =============================================================================
# MixWindow Tests
# =============================================================================


class TestMixWindow:
    """Test MixWindow audibility range."""

    def test_default_values(self):
        """MixWindow has correct defaults."""
        window = MixWindow()
        assert window.floor_db == HDR_FLOOR_DB
        assert window.ceiling_db == HDR_CEILING_DB
        assert window.window_db == HDR_WINDOW_DB
        assert window.center_db == HDR_DEFAULT_CENTER_DB

    def test_window_floor_property(self):
        """window_floor computes from center."""
        window = MixWindow(center_db=-30.0, window_db=24.0)

        # -30 - 24/2 = -42
        assert window.window_floor == pytest.approx(-42.0, rel=1e-6)

    def test_window_ceiling_property(self):
        """window_ceiling computes from center."""
        window = MixWindow(center_db=-30.0, window_db=24.0)

        # -30 + 24/2 = -18
        assert window.window_ceiling == pytest.approx(-18.0, rel=1e-6)

    def test_map_level_below_window(self):
        """Levels below window floor map to silence."""
        window = MixWindow(center_db=-30.0, window_db=24.0)

        # Floor is -42, level at -50 should be silent
        result = window.map_level(-50.0)
        assert result == MIN_VOLUME_DB

    def test_map_level_above_window(self):
        """Levels above window ceiling map to ceiling."""
        window = MixWindow(center_db=-30.0, window_db=24.0)

        # Ceiling is -18, level at -10 should map to output ceiling
        result = window.map_level(-10.0)
        assert result == HDR_CEILING_DB

    def test_map_level_within_window(self):
        """Levels within window map linearly."""
        window = MixWindow(
            center_db=-30.0,
            window_db=24.0,
            floor_db=-60.0,
            ceiling_db=0.0,
        )

        # Window: -42 to -18 (24dB range)
        # Output: -60 to 0 (60dB range)
        # Level at center should map to middle of output

        result = window.map_level(-30.0)
        # Position = (-30 - (-42)) / 24 = 12/24 = 0.5
        # Output = -60 + 0.5 * 60 = -30
        assert result == pytest.approx(-30.0, rel=0.1)

    def test_map_level_at_floor(self):
        """Level at window floor maps to output floor or silence."""
        window = MixWindow(
            center_db=-30.0,
            window_db=24.0,
            floor_db=-60.0,
            ceiling_db=0.0,
        )

        result = window.map_level(-42.0)  # At window floor
        # May map to output floor or MIN_VOLUME_DB depending on implementation
        assert result <= -60.0

    def test_contains(self):
        """contains checks if level is in window."""
        window = MixWindow(center_db=-30.0, window_db=24.0)

        # Window: -42 to -18
        assert window.contains(-30.0) is True
        assert window.contains(-42.0) is True
        assert window.contains(-18.0) is True
        assert window.contains(-50.0) is False
        assert window.contains(-10.0) is False

    def test_copy(self):
        """copy creates independent copy."""
        window = MixWindow(
            floor_db=-70.0,
            ceiling_db=5.0,
            window_db=30.0,
            center_db=-25.0,
        )

        copy = window.copy()

        assert copy.floor_db == -70.0
        assert copy.ceiling_db == 5.0
        assert copy.window_db == 30.0
        assert copy.center_db == -25.0

        # Modify copy
        copy.center_db = -35.0
        assert window.center_db == -25.0


# =============================================================================
# HDRAudioManager Source Management Tests
# =============================================================================


class TestHDRAudioManagerSourceManagement:
    """Test HDRAudioManager source management."""

    def test_register_source(self):
        """Register audio source."""
        manager = HDRAudioManager()

        source = manager.register_source(
            name="sfx_source",
            priority=HDR_PRIORITY_HIGH,
            protected=True,
        )

        assert source.name == "sfx_source"
        assert source.priority == HDR_PRIORITY_HIGH
        assert source.is_protected is True

    def test_register_source_with_bus(self):
        """Register source with associated bus."""
        manager = HDRAudioManager()
        bus = MixBus("sfx", BusType.CATEGORY)

        source = manager.register_source(name="sfx_source", bus=bus)

        assert source.bus is bus

    def test_unregister_source(self):
        """Unregister audio source."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")

        result = manager.unregister_source(source.id)

        assert result is True
        assert manager.get_source(source.id) is None

    def test_unregister_source_not_found(self):
        """unregister_source returns False if not found."""
        manager = HDRAudioManager()

        result = manager.unregister_source("nonexistent")
        assert result is False

    def test_get_source(self):
        """Get source by ID."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")

        found = manager.get_source(source.id)

        assert found is not None
        assert found.id == source.id
        assert found.name == "test"

    def test_get_source_returns_copy(self):
        """get_source returns a copy."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")
        found = manager.get_source(source.id)

        found.name = "modified"

        original = manager.get_source(source.id)
        assert original.name == "test"

    def test_get_source_not_found(self):
        """get_source returns None if not found."""
        manager = HDRAudioManager()

        result = manager.get_source("nonexistent")
        assert result is None

    def test_set_source_loudness(self):
        """Set source loudness."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")

        manager.set_source_loudness(source.id, -20.0, is_active=True)

        found = manager.get_source(source.id)
        assert found.target_loudness_db == -20.0
        assert found.is_active is True

    def test_set_source_priority(self):
        """Set source priority."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test", priority=HDR_PRIORITY_LOW)

        manager.set_source_priority(source.id, HDR_PRIORITY_CRITICAL)

        found = manager.get_source(source.id)
        assert found.priority == HDR_PRIORITY_CRITICAL


# =============================================================================
# HDRAudioManager Window Management Tests
# =============================================================================


class TestHDRAudioManagerWindowManagement:
    """Test HDRAudioManager window management."""

    def test_window_property(self):
        """window property returns copy."""
        manager = HDRAudioManager()

        window = manager.window

        assert isinstance(window, MixWindow)

        # Modify returned copy
        window.center_db = -50.0

        # Original unchanged
        assert manager.window.center_db == HDR_DEFAULT_CENTER_DB

    def test_set_window_size(self):
        """Set window size in dB."""
        manager = HDRAudioManager()

        manager.set_window_size(30.0)

        assert manager.window.window_db == 30.0

    def test_set_window_size_clamped(self):
        """Window size is clamped to valid range."""
        manager = HDRAudioManager()

        manager.set_window_size(1.0)  # Below minimum
        assert manager.window.window_db == HDR_WINDOW_MIN_DB

        manager.set_window_size(100.0)  # Above maximum
        assert manager.window.window_db == HDR_WINDOW_MAX_DB

    def test_set_adaptation_speed(self):
        """Set adaptation speed."""
        manager = HDRAudioManager()

        manager.set_adaptation_speed(0.25)

        # Check internal value
        assert manager._adaptation_speed == 0.25

    def test_set_adaptation_speed_minimum(self):
        """Adaptation speed has minimum."""
        manager = HDRAudioManager()

        manager.set_adaptation_speed(0.0)

        assert manager._adaptation_speed >= 0.01

    def test_force_window_center(self):
        """Force window to specific center."""
        manager = HDRAudioManager()

        manager.force_window_center(-25.0)

        assert manager.window.center_db == -25.0


# =============================================================================
# HDRAudioManager Enable/Disable Tests
# =============================================================================


class TestHDRAudioManagerEnableDisable:
    """Test HDRAudioManager enable/disable."""

    def test_enabled_default(self):
        """HDR is enabled by default."""
        manager = HDRAudioManager()
        assert manager.enabled is True

    def test_enable_disable(self):
        """Enable/disable HDR."""
        manager = HDRAudioManager()

        manager.enabled = False
        assert manager.enabled is False

        manager.enabled = True
        assert manager.enabled is True


# =============================================================================
# HDRAudioManager Update Tests
# =============================================================================


class TestHDRAudioManagerUpdate:
    """Test HDRAudioManager update logic."""

    def test_update_disabled_does_nothing(self):
        """Update with HDR disabled does nothing."""
        manager = HDRAudioManager()
        manager.enabled = False

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -10.0, is_active=True)

        initial_center = manager.window.center_db

        manager.update(0.1)

        # Window should not have changed
        assert manager.window.center_db == initial_center

    def test_update_no_active_sources(self):
        """Update with no active sources does nothing."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -10.0, is_active=False)

        initial_center = manager.window.center_db

        manager.update(0.1)

        assert manager.window.center_db == initial_center

    def test_update_adapts_window(self):
        """Update adapts window center to loudest sources."""
        manager = HDRAudioManager()
        manager.set_adaptation_speed(0.1)  # Fast adaptation for testing

        source = manager.register_source(name="test", priority=HDR_PRIORITY_HIGH)
        manager.set_source_loudness(source.id, -10.0, is_active=True)

        initial_center = manager.window.center_db

        # Update multiple times
        for _ in range(10):
            manager.update(0.1)

        # Window should have shifted towards the source
        assert manager.window.center_db != initial_center

    def test_update_priority_weighted(self):
        """Higher priority sources influence window more."""
        manager = HDRAudioManager()
        manager.set_adaptation_speed(0.1)

        high = manager.register_source(name="high", priority=HDR_PRIORITY_HIGH)
        low = manager.register_source(name="low", priority=HDR_PRIORITY_LOW)

        # High priority is louder but low priority has same level
        manager.set_source_loudness(high.id, -20.0, is_active=True)
        manager.set_source_loudness(low.id, -20.0, is_active=True)

        for _ in range(10):
            manager.update(0.1)

        # Window should adapt to both, but high priority has more influence

    def test_update_protected_sources_unchanged(self):
        """Protected sources bypass HDR mapping."""
        manager = HDRAudioManager()

        source = manager.register_source(name="protected", protected=True)
        manager.set_source_loudness(source.id, -10.0, is_active=True)

        manager.update(0.1)

        found = manager.get_source(source.id)
        # Protected source should have loudness = target
        assert found.loudness_db == found.target_loudness_db

    def test_update_calls_window_change_callback(self):
        """Update calls window change callback."""
        manager = HDRAudioManager()
        manager.set_adaptation_speed(0.01)

        callback = MagicMock()
        manager.on_window_change(callback)

        source = manager.register_source(name="test", priority=HDR_PRIORITY_HIGH)
        manager.set_source_loudness(source.id, -10.0, is_active=True)

        manager.update(0.1)

        callback.assert_called()


# =============================================================================
# HDRAudioManager Level Queries Tests
# =============================================================================


class TestHDRAudioManagerLevelQueries:
    """Test HDRAudioManager level query methods."""

    def test_get_output_level(self):
        """Get HDR-processed output level."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -20.0, is_active=True)

        manager.update(0.1)

        level = manager.get_output_level(source.id)

        # Should be processed through window
        assert level != MIN_VOLUME_DB  # Not silent

    def test_get_output_level_not_found(self):
        """get_output_level returns MIN_VOLUME_DB if not found."""
        manager = HDRAudioManager()

        level = manager.get_output_level("nonexistent")
        assert level == MIN_VOLUME_DB

    def test_get_gain_adjustment(self):
        """Get gain adjustment from HDR."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -20.0, is_active=True)

        manager.update(0.1)

        gain = manager.get_gain_adjustment(source.id)

        # Gain adjustment is output - target
        found = manager.get_source(source.id)
        expected = found.loudness_db - found.target_loudness_db
        assert gain == pytest.approx(expected, rel=1e-6)

    def test_get_gain_adjustment_not_found(self):
        """get_gain_adjustment returns 0 if not found."""
        manager = HDRAudioManager()

        gain = manager.get_gain_adjustment("nonexistent")
        assert gain == 0.0


# =============================================================================
# HDRAudioManager Batch Updates Tests
# =============================================================================


class TestHDRAudioManagerBatchUpdates:
    """Test HDRAudioManager batch update methods."""

    def test_update_source_levels(self):
        """Update multiple source levels at once."""
        manager = HDRAudioManager()

        source1 = manager.register_source(name="s1")
        source2 = manager.register_source(name="s2")

        manager.update_source_levels({
            source1.id: -10.0,
            source2.id: -20.0,
        })

        found1 = manager.get_source(source1.id)
        found2 = manager.get_source(source2.id)

        assert found1.target_loudness_db == -10.0
        assert found2.target_loudness_db == -20.0
        assert found1.is_active is True  # Above threshold
        assert found2.is_active is True

    def test_update_source_levels_inactive(self):
        """Levels below threshold marked inactive."""
        manager = HDRAudioManager()

        source = manager.register_source(name="quiet")

        manager.update_source_levels({
            source.id: -80.0,  # Below HDR_ACTIVE_THRESHOLD_DB
        })

        found = manager.get_source(source.id)
        assert found.is_active is False

    def test_analyze_bus_levels(self):
        """Update sources based on bus levels."""
        manager = HDRAudioManager()
        bus = MixBus("sfx", BusType.CATEGORY)

        source = manager.register_source(name="sfx_source", bus=bus)

        manager.analyze_bus_levels({
            bus.id: -15.0,
        })

        found = manager.get_source(source.id)
        assert found.target_loudness_db == -15.0
        assert found.is_active is True


# =============================================================================
# HDRAudioManager Query Methods Tests
# =============================================================================


class TestHDRAudioManagerQueries:
    """Test HDRAudioManager query methods."""

    def test_get_active_sources(self):
        """Get active sources sorted by priority."""
        manager = HDRAudioManager()

        high = manager.register_source(name="high", priority=HDR_PRIORITY_HIGH)
        low = manager.register_source(name="low", priority=HDR_PRIORITY_LOW)
        inactive = manager.register_source(name="inactive")

        manager.set_source_loudness(high.id, -10.0, is_active=True)
        manager.set_source_loudness(low.id, -20.0, is_active=True)
        manager.set_source_loudness(inactive.id, -30.0, is_active=False)

        active = manager.get_active_sources()

        assert len(active) == 2
        assert active[0].name == "high"  # Higher priority first
        assert active[1].name == "low"

    def test_get_loudest_source(self):
        """Get loudest active source."""
        manager = HDRAudioManager()

        loud = manager.register_source(name="loud")
        quiet = manager.register_source(name="quiet")

        manager.set_source_loudness(loud.id, -5.0, is_active=True)
        manager.set_source_loudness(quiet.id, -30.0, is_active=True)

        loudest = manager.get_loudest_source()

        assert loudest.name == "loud"
        assert loudest.target_loudness_db == -5.0

    def test_get_loudest_source_none_active(self):
        """get_loudest_source returns None if no active sources."""
        manager = HDRAudioManager()

        source = manager.register_source(name="inactive")
        manager.set_source_loudness(source.id, -10.0, is_active=False)

        loudest = manager.get_loudest_source()
        assert loudest is None

    def test_is_in_window(self):
        """Check if level is in current window."""
        manager = HDRAudioManager()
        manager.force_window_center(-30.0)
        manager.set_window_size(24.0)

        # Window: -42 to -18
        assert manager.is_in_window(-30.0) is True
        assert manager.is_in_window(-50.0) is False
        assert manager.is_in_window(-10.0) is False


# =============================================================================
# HDRAudioManager Callbacks Tests
# =============================================================================


class TestHDRAudioManagerCallbacks:
    """Test HDRAudioManager callback management."""

    def test_on_window_change(self):
        """Register window change callback."""
        manager = HDRAudioManager()
        callback = MagicMock()

        manager.on_window_change(callback)

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -10.0, is_active=True)
        manager.set_adaptation_speed(0.01)

        manager.update(0.1)

        callback.assert_called()

    def test_remove_callback(self):
        """Remove window change callback."""
        manager = HDRAudioManager()
        callback = MagicMock()

        manager.on_window_change(callback)
        result = manager.remove_callback(callback)

        assert result is True

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -10.0, is_active=True)
        manager.update(0.1)

        callback.assert_not_called()

    def test_remove_callback_not_found(self):
        """remove_callback returns False if not found."""
        manager = HDRAudioManager()
        callback = MagicMock()

        result = manager.remove_callback(callback)
        assert result is False


# =============================================================================
# HDRAudioManager State Management Tests
# =============================================================================


class TestHDRAudioManagerState:
    """Test HDRAudioManager state management."""

    def test_reset(self):
        """Reset HDR system to defaults."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")
        manager.set_source_loudness(source.id, -10.0, is_active=True)
        manager.force_window_center(-25.0)

        manager.reset()

        # Window should be reset
        assert manager.window.center_db == HDR_DEFAULT_CENTER_DB

        # Sources should be inactive
        found = manager.get_source(source.id)
        assert found.is_active is False

    def test_clear(self):
        """Clear all sources and reset."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test")

        manager.clear()

        assert manager.get_source(source.id) is None

    def test_get_state(self):
        """Get HDR state for debugging."""
        manager = HDRAudioManager()

        source = manager.register_source(name="test_source", priority=HDR_PRIORITY_HIGH)
        manager.set_source_loudness(source.id, -20.0, is_active=True)

        state = manager.get_state()

        assert "enabled" in state
        assert "window" in state
        assert "sources" in state
        assert "adaptation_speed" in state

        assert state["enabled"] is True
        assert "center_db" in state["window"]
        assert source.id in state["sources"]
        assert state["sources"][source.id]["name"] == "test_source"

    def test_repr(self):
        """repr shows useful info."""
        manager = HDRAudioManager()

        source = manager.register_source(name="active")
        manager.set_source_loudness(source.id, -10.0, is_active=True)

        repr_str = repr(manager)

        assert "HDRAudioManager" in repr_str
        assert "sources=" in repr_str
        assert "active=" in repr_str
        assert "window_center=" in repr_str
