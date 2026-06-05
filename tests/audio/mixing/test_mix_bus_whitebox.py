"""Whitebox tests for MixBus audio routing hierarchy.

Tests internal implementation of:
- Volume dB/linear conversions
- Bus hierarchy and cycle detection
- Filter state management
- Audio processing pipeline
- Thread safety mechanisms
- Callback notifications
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.audio.mixing.config import (
    DEFAULT_BUS_VOLUME,
    DEFAULT_HIGH_PASS,
    DEFAULT_LOW_PASS,
    DEFAULT_PITCH,
    FILTER_Q,
    MAX_FILTER_FREQ,
    MAX_PITCH,
    MAX_VOLUME_DB,
    MIN_FILTER_FREQ,
    MIN_PITCH,
    MIN_VOLUME_DB,
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    clamp,
    db_to_linear,
    linear_to_db,
)
from engine.audio.mixing.mix_bus import (
    BusState,
    BusType,
    FilterState,
    MixBus,
    create_default_hierarchy,
)


# =============================================================================
# dB/Linear Conversion Tests
# =============================================================================


class TestDbLinearConversion:
    """Test dB to linear and linear to dB conversions."""

    def test_db_to_linear_0db(self):
        """0 dB should equal 1.0 linear."""
        assert db_to_linear(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_db_to_linear_minus_6db(self):
        """~6 dB should equal ~0.5 linear."""
        assert db_to_linear(-6.0206) == pytest.approx(0.5, rel=1e-3)

    def test_db_to_linear_plus_6db(self):
        """+6 dB should equal ~2.0 linear."""
        assert db_to_linear(6.0206) == pytest.approx(2.0, rel=1e-3)

    def test_db_to_linear_silence_threshold(self):
        """Below silence threshold should return 0."""
        result = db_to_linear(-80.1)
        assert result == 0.0

    def test_db_to_linear_at_threshold(self):
        """At silence threshold should return small value."""
        result = db_to_linear(-80.0)
        # At -80dB, linear = 10^(-80/20) = 10^-4 = 0.0001
        # But implementation may return 0 due to threshold check
        assert result >= 0.0 and result <= 0.001

    def test_linear_to_db_1(self):
        """Linear 1.0 should equal 0 dB."""
        assert linear_to_db(1.0) == pytest.approx(0.0, rel=1e-6)

    def test_linear_to_db_0_5(self):
        """Linear 0.5 should equal ~-6 dB."""
        assert linear_to_db(0.5) == pytest.approx(-6.0206, rel=1e-3)

    def test_linear_to_db_2(self):
        """Linear 2.0 should equal ~+6 dB."""
        assert linear_to_db(2.0) == pytest.approx(6.0206, rel=1e-3)

    def test_linear_to_db_zero(self):
        """Zero linear should return MIN_VOLUME_DB."""
        assert linear_to_db(0.0) == MIN_VOLUME_DB

    def test_linear_to_db_negative(self):
        """Negative linear should return MIN_VOLUME_DB."""
        assert linear_to_db(-0.5) == MIN_VOLUME_DB

    def test_roundtrip_db_linear(self):
        """dB -> linear -> dB should round-trip."""
        for db in [-60, -40, -20, -6, 0, 6, 12]:
            result = linear_to_db(db_to_linear(db))
            assert result == pytest.approx(db, rel=1e-3)

    def test_roundtrip_linear_db(self):
        """linear -> dB -> linear should round-trip."""
        for linear in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
            result = db_to_linear(linear_to_db(linear))
            assert result == pytest.approx(linear, rel=1e-3)


# =============================================================================
# FilterState Tests
# =============================================================================


class TestFilterState:
    """Test FilterState dataclass."""

    def test_default_values(self):
        """FilterState should have correct defaults."""
        fs = FilterState()
        assert fs.low_pass_freq == DEFAULT_LOW_PASS
        assert fs.high_pass_freq == DEFAULT_HIGH_PASS
        assert fs.low_pass_q == FILTER_Q
        assert fs.high_pass_q == FILTER_Q
        assert fs.low_pass_enabled is False
        assert fs.high_pass_enabled is False

    def test_reset(self):
        """Reset should restore defaults."""
        fs = FilterState()
        fs.low_pass_freq = 5000.0
        fs.high_pass_freq = 500.0
        fs.low_pass_enabled = True
        fs.high_pass_enabled = True
        fs.low_pass_q = 2.0
        fs.high_pass_q = 2.0

        fs.reset()

        assert fs.low_pass_freq == DEFAULT_LOW_PASS
        assert fs.high_pass_freq == DEFAULT_HIGH_PASS
        assert fs.low_pass_enabled is False
        assert fs.high_pass_enabled is False
        assert fs.low_pass_q == FILTER_Q
        assert fs.high_pass_q == FILTER_Q

    def test_copy(self):
        """Copy should create independent copy."""
        fs = FilterState()
        fs.low_pass_freq = 5000.0
        fs.low_pass_enabled = True

        copy = fs.copy()

        assert copy.low_pass_freq == 5000.0
        assert copy.low_pass_enabled is True

        # Modify original
        fs.low_pass_freq = 10000.0
        assert copy.low_pass_freq == 5000.0  # Copy unchanged


# =============================================================================
# BusState Tests
# =============================================================================


class TestBusState:
    """Test BusState dataclass."""

    def test_default_values(self):
        """BusState should have correct defaults."""
        bs = BusState()
        assert bs.volume_linear == DEFAULT_BUS_VOLUME
        assert bs.pitch == DEFAULT_PITCH
        assert bs.muted is False
        assert bs.soloed is False
        assert isinstance(bs.filters, FilterState)

    def test_copy(self):
        """Copy should create deep copy including filters."""
        bs = BusState()
        bs.volume_linear = 0.5
        bs.pitch = 1.5
        bs.muted = True
        bs.filters.low_pass_freq = 5000.0

        copy = bs.copy()

        assert copy.volume_linear == 0.5
        assert copy.pitch == 1.5
        assert copy.muted is True
        assert copy.filters.low_pass_freq == 5000.0

        # Modify original
        bs.volume_linear = 0.8
        bs.filters.low_pass_freq = 10000.0

        # Copy should be unchanged
        assert copy.volume_linear == 0.5
        assert copy.filters.low_pass_freq == 5000.0


# =============================================================================
# MixBus Core Tests
# =============================================================================


class TestMixBusCreation:
    """Test MixBus initialization."""

    def test_default_creation(self):
        """Default bus creation."""
        bus = MixBus("test")
        assert bus.name == "test"
        assert bus.bus_type == BusType.SUB
        assert bus.volume == DEFAULT_BUS_VOLUME
        assert bus.pitch == DEFAULT_PITCH
        assert bus.parent is None
        assert bus.children == []

    def test_creation_with_type(self):
        """Bus creation with specific type."""
        master = MixBus("master", BusType.MASTER)
        assert master.bus_type == BusType.MASTER

        category = MixBus("sfx", BusType.CATEGORY)
        assert category.bus_type == BusType.CATEGORY

        aux = MixBus("reverb", BusType.AUX)
        assert aux.bus_type == BusType.AUX

    def test_creation_with_parent(self):
        """Bus creation with parent."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)

        assert sfx.parent is master
        assert sfx in master.children

    def test_creation_with_volume_pitch(self):
        """Bus creation with custom volume and pitch."""
        bus = MixBus("test", volume=0.5, pitch=2.0)
        assert bus.volume == 0.5
        assert bus.pitch == 2.0

    def test_volume_clamped_on_creation(self):
        """Volume should be clamped on creation."""
        bus = MixBus("test", volume=100.0)  # Way above max
        max_linear = db_to_linear(MAX_VOLUME_DB)
        assert bus.volume <= max_linear

    def test_pitch_clamped_on_creation(self):
        """Pitch should be clamped on creation."""
        bus = MixBus("test", pitch=100.0)
        assert bus.pitch == MAX_PITCH

        bus2 = MixBus("test2", pitch=0.001)
        assert bus2.pitch == MIN_PITCH

    def test_unique_id(self):
        """Each bus should have unique ID."""
        bus1 = MixBus("test1")
        bus2 = MixBus("test2")
        assert bus1.id != bus2.id


# =============================================================================
# MixBus Volume Tests
# =============================================================================


class TestMixBusVolume:
    """Test MixBus volume controls."""

    def test_volume_property(self):
        """Volume property get/set."""
        bus = MixBus("test")
        bus.volume = 0.75
        assert bus.volume == 0.75

    def test_volume_clamping_low(self):
        """Volume should clamp at 0."""
        bus = MixBus("test")
        bus.volume = -1.0
        assert bus.volume == 0.0

    def test_volume_clamping_high(self):
        """Volume should clamp at max."""
        bus = MixBus("test")
        bus.volume = 100.0
        max_linear = db_to_linear(MAX_VOLUME_DB)
        assert bus.volume == pytest.approx(max_linear, rel=1e-3)

    def test_volume_db_property(self):
        """Volume dB property get/set."""
        bus = MixBus("test")
        bus.volume_db = -6.0
        assert bus.volume_db == pytest.approx(-6.0, rel=1e-2)
        assert bus.volume == pytest.approx(0.5012, rel=1e-3)

    def test_volume_db_clamping(self):
        """Volume dB should clamp to valid range."""
        bus = MixBus("test")
        bus.volume_db = -100.0
        assert bus.volume_db == MIN_VOLUME_DB

        bus.volume_db = 100.0
        assert bus.volume_db == MAX_VOLUME_DB

    def test_set_volume_db_method(self):
        """set_volume_db method."""
        bus = MixBus("test")
        bus.set_volume_db(-12.0)
        assert bus.volume_db == pytest.approx(-12.0, rel=1e-2)

    def test_set_volume_linear_method(self):
        """set_volume_linear method."""
        bus = MixBus("test")
        bus.set_volume_linear(0.25)
        assert bus.volume == 0.25

    def test_effective_volume_no_parent(self):
        """Effective volume without parent."""
        bus = MixBus("test")
        bus.volume = 0.5
        assert bus.get_effective_volume() == 0.5

    def test_effective_volume_with_parent(self):
        """Effective volume multiplies through hierarchy."""
        master = MixBus("master", BusType.MASTER)
        master.volume = 0.8
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        sfx.volume = 0.5

        assert sfx.get_effective_volume() == pytest.approx(0.4, rel=1e-6)

    def test_effective_volume_deep_hierarchy(self):
        """Effective volume through multiple levels."""
        master = MixBus("master", BusType.MASTER)
        master.volume = 0.9
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        sfx.volume = 0.8
        weapons = MixBus("weapons", BusType.SUB, parent=sfx)
        weapons.volume = 0.7

        expected = 0.9 * 0.8 * 0.7
        assert weapons.get_effective_volume() == pytest.approx(expected, rel=1e-6)

    def test_effective_volume_muted(self):
        """Effective volume is 0 when muted."""
        bus = MixBus("test")
        bus.volume = 0.5
        bus.muted = True
        assert bus.get_effective_volume() == 0.0

    def test_effective_volume_parent_muted(self):
        """Effective volume is 0 when parent is muted."""
        master = MixBus("master", BusType.MASTER)
        master.muted = True
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        sfx.volume = 0.5

        assert sfx.get_effective_volume() == 0.0


# =============================================================================
# MixBus Pitch Tests
# =============================================================================


class TestMixBusPitch:
    """Test MixBus pitch controls."""

    def test_pitch_property(self):
        """Pitch property get/set."""
        bus = MixBus("test")
        bus.pitch = 1.5
        assert bus.pitch == 1.5

    def test_pitch_clamping(self):
        """Pitch should clamp to valid range."""
        bus = MixBus("test")
        bus.pitch = 0.001
        assert bus.pitch == MIN_PITCH

        bus.pitch = 100.0
        assert bus.pitch == MAX_PITCH

    def test_effective_pitch_no_parent(self):
        """Effective pitch without parent."""
        bus = MixBus("test")
        bus.pitch = 1.5
        assert bus.get_effective_pitch() == 1.5

    def test_effective_pitch_with_parent(self):
        """Effective pitch multiplies through hierarchy."""
        master = MixBus("master", BusType.MASTER)
        master.pitch = 1.5
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        sfx.pitch = 1.2

        expected = min(1.5 * 1.2, MAX_PITCH)
        assert sfx.get_effective_pitch() == pytest.approx(expected, rel=1e-6)

    def test_effective_pitch_clamped(self):
        """Effective pitch is clamped to MAX_PITCH."""
        master = MixBus("master", BusType.MASTER)
        master.pitch = 3.0
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        sfx.pitch = 3.0

        assert sfx.get_effective_pitch() == MAX_PITCH


# =============================================================================
# MixBus Mute/Solo Tests
# =============================================================================


class TestMixBusMuteSolo:
    """Test MixBus mute and solo controls."""

    def test_muted_default(self):
        """Default mute state is False."""
        bus = MixBus("test")
        assert bus.muted is False

    def test_muted_set(self):
        """Mute state can be set."""
        bus = MixBus("test")
        bus.muted = True
        assert bus.muted is True

    def test_toggle_mute(self):
        """Toggle mute returns new state."""
        bus = MixBus("test")
        result = bus.toggle_mute()
        assert result is True
        assert bus.muted is True

        result = bus.toggle_mute()
        assert result is False
        assert bus.muted is False

    def test_soloed_default(self):
        """Default solo state is False."""
        bus = MixBus("test")
        assert bus.soloed is False

    def test_soloed_set(self):
        """Solo state can be set."""
        bus = MixBus("test")
        bus.soloed = True
        assert bus.soloed is True

    def test_toggle_solo(self):
        """Toggle solo returns new state."""
        bus = MixBus("test")
        result = bus.toggle_solo()
        assert result is True
        assert bus.soloed is True

        result = bus.toggle_solo()
        assert result is False
        assert bus.soloed is False


# =============================================================================
# MixBus Filter Tests
# =============================================================================


class TestMixBusFilters:
    """Test MixBus filter controls."""

    def test_filters_property(self):
        """Filters property returns copy."""
        bus = MixBus("test")
        filters = bus.filters
        assert isinstance(filters, FilterState)

        # Modifying returned copy shouldn't affect bus
        filters.low_pass_freq = 5000.0
        assert bus.filters.low_pass_freq == DEFAULT_LOW_PASS

    def test_set_low_pass(self):
        """Set low-pass filter parameters."""
        bus = MixBus("test")
        bus.set_low_pass(5000.0, q=2.0, enabled=True)

        filters = bus.filters
        assert filters.low_pass_freq == 5000.0
        assert filters.low_pass_q == 2.0
        assert filters.low_pass_enabled is True

    def test_set_low_pass_clamping(self):
        """Low-pass frequency is clamped."""
        bus = MixBus("test")
        bus.set_low_pass(1.0)  # Below MIN_FILTER_FREQ
        assert bus.filters.low_pass_freq == MIN_FILTER_FREQ

        bus.set_low_pass(50000.0)  # Above MAX_FILTER_FREQ
        assert bus.filters.low_pass_freq == MAX_FILTER_FREQ

    def test_set_low_pass_q_clamping(self):
        """Low-pass Q is clamped at minimum."""
        bus = MixBus("test")
        bus.set_low_pass(5000.0, q=0.0)
        assert bus.filters.low_pass_q >= 0.1

    def test_set_high_pass(self):
        """Set high-pass filter parameters."""
        bus = MixBus("test")
        bus.set_high_pass(500.0, q=1.5, enabled=True)

        filters = bus.filters
        assert filters.high_pass_freq == 500.0
        assert filters.high_pass_q == 1.5
        assert filters.high_pass_enabled is True

    def test_set_high_pass_clamping(self):
        """High-pass frequency is clamped."""
        bus = MixBus("test")
        bus.set_high_pass(1.0)
        assert bus.filters.high_pass_freq == MIN_FILTER_FREQ

        bus.set_high_pass(50000.0)
        assert bus.filters.high_pass_freq == MAX_FILTER_FREQ

    def test_reset_filters(self):
        """Reset filters to defaults."""
        bus = MixBus("test")
        bus.set_low_pass(5000.0, enabled=True)
        bus.set_high_pass(500.0, enabled=True)

        bus.reset_filters()

        filters = bus.filters
        assert filters.low_pass_freq == DEFAULT_LOW_PASS
        assert filters.high_pass_freq == DEFAULT_HIGH_PASS
        assert filters.low_pass_enabled is False
        assert filters.high_pass_enabled is False


# =============================================================================
# MixBus Hierarchy Tests
# =============================================================================


class TestMixBusHierarchy:
    """Test MixBus parent-child hierarchy."""

    def test_set_parent(self):
        """Set parent updates both parent and child."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY)

        sfx.set_parent(master)

        assert sfx.parent is master
        assert sfx in master.children

    def test_set_parent_none(self):
        """Set parent to None removes from hierarchy."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)

        sfx.set_parent(None)

        assert sfx.parent is None
        assert sfx not in master.children

    def test_set_parent_changes_hierarchy(self):
        """Changing parent updates old and new parent."""
        master = MixBus("master", BusType.MASTER)
        alt_master = MixBus("alt_master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)

        sfx.set_parent(alt_master)

        assert sfx.parent is alt_master
        assert sfx in alt_master.children
        assert sfx not in master.children

    def test_set_parent_self_raises(self):
        """Cannot set bus as its own parent."""
        bus = MixBus("test")
        with pytest.raises(ValueError, match="own parent"):
            bus.set_parent(bus)

    def test_set_parent_cycle_raises(self):
        """Cannot create cycle in hierarchy."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        weapons = MixBus("weapons", BusType.SUB, parent=sfx)

        with pytest.raises(ValueError, match="cycle"):
            master.set_parent(weapons)

    def test_add_child(self):
        """Add child updates both parent and child."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY)

        master.add_child(sfx)

        assert sfx.parent is master
        assert sfx in master.children

    def test_remove_child(self):
        """Remove child updates both parent and child."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)

        result = master.remove_child(sfx)

        assert result is True
        assert sfx.parent is None
        assert sfx not in master.children

    def test_remove_child_not_found(self):
        """Remove child returns False if not found."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY)

        result = master.remove_child(sfx)
        assert result is False

    def test_children_returns_copy(self):
        """Children property returns a copy."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)

        children = master.children
        children.append(MixBus("fake"))

        assert len(master.children) == 1

    def test_get_ancestors(self):
        """Get ancestors returns parent chain."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        weapons = MixBus("weapons", BusType.SUB, parent=sfx)

        ancestors = weapons.get_ancestors()

        assert len(ancestors) == 2
        assert ancestors[0] is sfx
        assert ancestors[1] is master

    def test_get_ancestors_no_parent(self):
        """Get ancestors returns empty list for root."""
        master = MixBus("master", BusType.MASTER)
        assert master.get_ancestors() == []

    def test_get_descendants(self):
        """Get descendants returns all children recursively."""
        master = MixBus("master", BusType.MASTER)
        sfx = MixBus("sfx", BusType.CATEGORY, parent=master)
        weapons = MixBus("weapons", BusType.SUB, parent=sfx)
        music = MixBus("music", BusType.CATEGORY, parent=master)

        descendants = master.get_descendants()

        assert len(descendants) == 3
        assert sfx in descendants
        assert weapons in descendants
        assert music in descendants

    def test_get_descendants_no_children(self):
        """Get descendants returns empty list for leaf."""
        bus = MixBus("test")
        assert bus.get_descendants() == []


# =============================================================================
# MixBus Audio Processing Tests
# =============================================================================


class TestMixBusAudioProcessing:
    """Test MixBus audio buffer processing."""

    def test_clear_acc_buffer(self):
        """Clear accumulation buffer allocates and zeros."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        buffer = bus.read_acc_buffer(512)
        assert buffer.shape == (MIXER_NUM_CHANNELS, 512)
        assert np.all(buffer == 0.0)

    def test_accumulate_stereo(self):
        """Accumulate stereo samples."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32) * 0.5
        bus.accumulate(samples, 512)

        buffer = bus.read_acc_buffer(512)
        assert np.allclose(buffer, 0.5)

    def test_accumulate_mono_broadcast(self):
        """Mono samples broadcast to stereo."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones((1, 512), dtype=np.float32) * 0.5
        bus.accumulate(samples, 512)

        buffer = bus.read_acc_buffer(512)
        assert buffer.shape == (MIXER_NUM_CHANNELS, 512)
        assert np.allclose(buffer, 0.5)

    def test_accumulate_1d_mono(self):
        """1D mono samples reshaped and broadcast."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones(512, dtype=np.float32) * 0.3
        bus.accumulate(samples, 512)

        buffer = bus.read_acc_buffer(512)
        assert np.allclose(buffer, 0.3)

    def test_accumulate_multiple_adds(self):
        """Multiple accumulations add together."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32) * 0.25
        bus.accumulate(samples, 512)
        bus.accumulate(samples, 512)

        buffer = bus.read_acc_buffer(512)
        assert np.allclose(buffer, 0.5)

    def test_write_output_delegates_to_accumulate(self):
        """write_output delegates to accumulate."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32) * 0.5
        bus.write_output(samples, 512)

        buffer = bus.read_acc_buffer(512)
        assert np.allclose(buffer, 0.5)

    def test_read_acc_buffer_returns_copy(self):
        """read_acc_buffer returns a copy."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32) * 0.5
        bus.accumulate(samples, 512)

        buffer1 = bus.read_acc_buffer(512)
        buffer2 = bus.read_acc_buffer(512)

        buffer1[:] = 0.0
        assert np.allclose(buffer2, 0.5)

    def test_read_acc_buffer_uninitialized(self):
        """read_acc_buffer returns zeros if uninitialized."""
        bus = MixBus("test")
        buffer = bus.read_acc_buffer(512)

        assert buffer.shape == (MIXER_NUM_CHANNELS, 512)
        assert np.all(buffer == 0.0)

    def test_process_audio_applies_volume(self):
        """process_audio applies volume scaling."""
        bus = MixBus("test")
        bus.volume = 0.5
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32)
        bus.accumulate(samples, 512)

        output = bus.process_audio(512)
        assert np.allclose(output, 0.5, rtol=1e-3)

    def test_process_audio_muted_returns_silence(self):
        """process_audio returns silence when muted."""
        bus = MixBus("test")
        bus.muted = True
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32)
        bus.accumulate(samples, 512)

        output = bus.process_audio(512)
        assert np.all(output == 0.0)

    def test_process_audio_clips_output(self):
        """process_audio clips to [-1, 1]."""
        bus = MixBus("test")
        bus.volume = db_to_linear(12.0)  # +12 dB
        bus.clear_acc_buffer(512)

        samples = np.ones((2, 512), dtype=np.float32)
        bus.accumulate(samples, 512)

        output = bus.process_audio(512)
        assert np.all(output <= 1.0)
        assert np.all(output >= -1.0)


# =============================================================================
# MixBus State Management Tests
# =============================================================================


class TestMixBusStateManagement:
    """Test MixBus state get/set."""

    def test_get_state(self):
        """get_state returns copy of current state."""
        bus = MixBus("test")
        bus.volume = 0.5
        bus.pitch = 1.5
        bus.muted = True

        state = bus.get_state()

        assert state.volume_linear == 0.5
        assert state.pitch == 1.5
        assert state.muted is True

        # Modifying returned state doesn't affect bus
        state.volume_linear = 0.9
        assert bus.volume == 0.5

    def test_set_state(self):
        """set_state applies new state."""
        bus = MixBus("test")

        state = BusState(volume_linear=0.5, pitch=1.5, muted=True)
        bus.set_state(state)

        assert bus.volume == 0.5
        assert bus.pitch == 1.5
        assert bus.muted is True

    def test_reset(self):
        """reset returns bus to default state."""
        bus = MixBus("test")
        bus.volume = 0.5
        bus.pitch = 1.5
        bus.muted = True
        bus.set_low_pass(5000.0, enabled=True)

        bus.reset()

        assert bus.volume == DEFAULT_BUS_VOLUME
        assert bus.pitch == DEFAULT_PITCH
        assert bus.muted is False
        assert bus.soloed is False


# =============================================================================
# MixBus Callback Tests
# =============================================================================


class TestMixBusCallbacks:
    """Test MixBus change callbacks."""

    def test_on_change_callback(self):
        """Callback called on state change."""
        bus = MixBus("test")
        callback = MagicMock()

        bus.on_change(callback)
        bus.volume = 0.5

        callback.assert_called_once_with(bus)

    def test_multiple_callbacks(self):
        """Multiple callbacks all called."""
        bus = MixBus("test")
        callback1 = MagicMock()
        callback2 = MagicMock()

        bus.on_change(callback1)
        bus.on_change(callback2)
        bus.volume = 0.5

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_remove_callback(self):
        """Remove callback stops notifications."""
        bus = MixBus("test")
        callback = MagicMock()

        bus.on_change(callback)
        result = bus.remove_callback(callback)

        assert result is True
        bus.volume = 0.5
        callback.assert_not_called()

    def test_remove_callback_not_found(self):
        """Remove callback returns False if not found."""
        bus = MixBus("test")
        callback = MagicMock()

        result = bus.remove_callback(callback)
        assert result is False

    def test_callback_exception_ignored(self):
        """Exceptions in callbacks are silently ignored."""
        bus = MixBus("test")
        bad_callback = MagicMock(side_effect=Exception("Test error"))
        good_callback = MagicMock()

        bus.on_change(bad_callback)
        bus.on_change(good_callback)

        # Should not raise, good_callback should still be called
        bus.volume = 0.5
        good_callback.assert_called_once()

    def test_callback_on_mute_change(self):
        """Callback called on mute change."""
        bus = MixBus("test")
        callback = MagicMock()
        bus.on_change(callback)

        bus.muted = True
        callback.assert_called()

    def test_callback_on_filter_change(self):
        """Callback called on filter change."""
        bus = MixBus("test")
        callback = MagicMock()
        bus.on_change(callback)

        bus.set_low_pass(5000.0)
        callback.assert_called()


# =============================================================================
# MixBus Thread Safety Tests
# =============================================================================


class TestMixBusThreadSafety:
    """Test MixBus thread safety."""

    def test_concurrent_volume_changes(self):
        """Concurrent volume changes don't corrupt state."""
        bus = MixBus("test")

        def modify_volume(value):
            for _ in range(100):
                bus.volume = value

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(modify_volume, 0.25),
                executor.submit(modify_volume, 0.5),
                executor.submit(modify_volume, 0.75),
                executor.submit(modify_volume, 1.0),
            ]
            for f in futures:
                f.result()

        # Volume should be one of the valid values
        assert 0.0 <= bus.volume <= db_to_linear(MAX_VOLUME_DB)

    def test_concurrent_accumulate(self):
        """Concurrent accumulate operations don't corrupt buffer."""
        bus = MixBus("test")
        bus.clear_acc_buffer(512)

        def accumulate_samples():
            for _ in range(50):
                samples = np.ones((2, 512), dtype=np.float32) * 0.01
                bus.accumulate(samples, 512)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(accumulate_samples) for _ in range(4)]
            for f in futures:
                f.result()

        buffer = bus.read_acc_buffer(512)
        # 4 threads * 50 iterations * 0.01 = 2.0
        expected = 4 * 50 * 0.01
        assert np.allclose(buffer, expected, rtol=0.01)

    def test_concurrent_hierarchy_changes(self):
        """Concurrent hierarchy changes don't corrupt structure."""
        master = MixBus("master", BusType.MASTER)
        buses = [MixBus(f"bus_{i}", BusType.SUB) for i in range(10)]

        def toggle_parent(bus):
            for _ in range(20):
                bus.set_parent(master)
                bus.set_parent(None)
                bus.set_parent(master)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(toggle_parent, bus) for bus in buses]
            for f in futures:
                f.result()

        # All buses should have valid parent state
        for bus in buses:
            assert bus.parent is None or bus.parent is master


# =============================================================================
# Default Hierarchy Tests
# =============================================================================


class TestDefaultHierarchy:
    """Test create_default_hierarchy function."""

    def test_creates_all_expected_buses(self):
        """Default hierarchy creates all expected buses."""
        buses = create_default_hierarchy()

        expected = [
            "master", "sfx", "music", "vo", "ambient", "ui",
            "footsteps", "weapons", "impacts",
            "combat", "exploration",
            "dialogue", "barks",
        ]

        for name in expected:
            assert name in buses

    def test_master_has_no_parent(self):
        """Master bus has no parent."""
        buses = create_default_hierarchy()
        assert buses["master"].parent is None

    def test_categories_parent_to_master(self):
        """Category buses parent to master."""
        buses = create_default_hierarchy()

        for category in ["sfx", "music", "vo", "ambient", "ui"]:
            assert buses[category].parent is buses["master"]

    def test_subcategories_parent_correctly(self):
        """Subcategory buses parent to their category."""
        buses = create_default_hierarchy()

        assert buses["footsteps"].parent is buses["sfx"]
        assert buses["weapons"].parent is buses["sfx"]
        assert buses["impacts"].parent is buses["sfx"]

        assert buses["combat"].parent is buses["music"]
        assert buses["exploration"].parent is buses["music"]

        assert buses["dialogue"].parent is buses["vo"]
        assert buses["barks"].parent is buses["vo"]

    def test_bus_types_correct(self):
        """Buses have correct types."""
        buses = create_default_hierarchy()

        assert buses["master"].bus_type == BusType.MASTER

        for category in ["sfx", "music", "vo", "ambient", "ui"]:
            assert buses[category].bus_type == BusType.CATEGORY

        for sub in ["footsteps", "weapons", "impacts", "combat",
                    "exploration", "dialogue", "barks"]:
            assert buses[sub].bus_type == BusType.SUB


# =============================================================================
# MixBus String Representation Tests
# =============================================================================


class TestMixBusStringRepresentation:
    """Test MixBus __repr__ and __str__."""

    def test_repr(self):
        """repr shows useful info."""
        bus = MixBus("test_bus", BusType.CATEGORY)
        bus.volume = 0.5
        bus.muted = True

        repr_str = repr(bus)
        assert "test_bus" in repr_str
        assert "category" in repr_str
        assert "0.50" in repr_str  # volume
        assert "muted=True" in repr_str

    def test_str(self):
        """str shows type:name format."""
        bus = MixBus("sfx", BusType.CATEGORY)
        assert str(bus) == "category:sfx"

        master = MixBus("master", BusType.MASTER)
        assert str(master) == "master:master"
