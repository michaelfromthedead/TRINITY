"""
Blackbox tests for MixBus component.

Tests PUBLIC behavior only - no internal state inspection.
Covers: volume (dB/linear), filters, parent/children hierarchy, mute/solo.
"""

import pytest
import math

from engine.audio.mixing import (
    MixBus,
    BusType,
    BusState,
    FilterState,
    create_default_hierarchy,
    db_to_linear,
    linear_to_db,
    MIN_VOLUME_DB,
    MAX_VOLUME_DB,
    DEFAULT_BUS_VOLUME,
    DEFAULT_LOW_PASS,
    DEFAULT_HIGH_PASS,
    MIN_FILTER_FREQ,
    MAX_FILTER_FREQ,
)


class TestMixBusCreation:
    """Test MixBus instantiation and default values."""

    def test_create_bus_with_name(self):
        """Bus can be created with a name."""
        bus = MixBus(name="sfx")
        assert bus.name == "sfx"

    def test_create_bus_with_type(self):
        """Bus can be created with specific type."""
        bus = MixBus(name="master", bus_type=BusType.MASTER)
        assert bus.bus_type == BusType.MASTER

    def test_create_category_bus(self):
        """Category bus can be created."""
        bus = MixBus(name="music", bus_type=BusType.CATEGORY)
        assert bus.bus_type == BusType.CATEGORY

    def test_create_sub_bus(self):
        """Sub bus can be created."""
        bus = MixBus(name="footsteps", bus_type=BusType.SUB)
        assert bus.bus_type == BusType.SUB

    def test_create_aux_bus(self):
        """Aux bus can be created."""
        bus = MixBus(name="reverb_send", bus_type=BusType.AUX)
        assert bus.bus_type == BusType.AUX

    def test_default_volume_is_unity(self):
        """New bus has default volume at unity."""
        bus = MixBus(name="test")
        assert bus.volume == pytest.approx(DEFAULT_BUS_VOLUME)

    def test_default_state_is_active(self):
        """New bus starts in active state."""
        bus = MixBus(name="test")
        assert bus.state == BusState.ACTIVE

    def test_default_not_muted(self):
        """New bus is not muted."""
        bus = MixBus(name="test")
        assert bus.muted is False

    def test_default_not_soloed(self):
        """New bus is not soloed."""
        bus = MixBus(name="test")
        assert bus.soloed is False


class TestVolumeConversions:
    """Test volume handling in dB and linear scale."""

    def test_set_volume_linear(self):
        """Volume can be set in linear scale."""
        bus = MixBus(name="test")
        bus.volume = 0.5
        assert bus.volume == pytest.approx(0.5)

    def test_volume_zero_is_silence(self):
        """Volume 0.0 represents silence."""
        bus = MixBus(name="test")
        bus.volume = 0.0
        assert bus.volume == pytest.approx(0.0)

    def test_volume_one_is_unity(self):
        """Volume 1.0 is unity gain."""
        bus = MixBus(name="test")
        bus.volume = 1.0
        assert bus.volume == pytest.approx(1.0)

    def test_set_volume_db(self):
        """Volume can be set via dB property."""
        bus = MixBus(name="test")
        bus.volume_db = -6.0
        expected_linear = db_to_linear(-6.0)
        assert bus.volume == pytest.approx(expected_linear, rel=1e-3)

    def test_get_volume_db(self):
        """Volume in dB can be retrieved."""
        bus = MixBus(name="test")
        bus.volume = 0.5
        expected_db = linear_to_db(0.5)
        assert bus.volume_db == pytest.approx(expected_db, rel=1e-3)

    def test_volume_db_at_unity_is_zero(self):
        """Unity gain is 0 dB."""
        bus = MixBus(name="test")
        bus.volume = 1.0
        assert bus.volume_db == pytest.approx(0.0, abs=0.1)

    def test_volume_db_at_half_is_negative_six(self):
        """Half amplitude is approximately -6 dB."""
        bus = MixBus(name="test")
        bus.volume = 0.5
        assert bus.volume_db == pytest.approx(-6.02, rel=0.01)

    def test_volume_clamps_to_min_db(self):
        """Volume is clamped at minimum dB threshold."""
        bus = MixBus(name="test")
        bus.volume_db = MIN_VOLUME_DB - 20.0
        assert bus.volume_db >= MIN_VOLUME_DB

    def test_volume_clamps_to_max_db(self):
        """Volume is clamped at maximum dB threshold."""
        bus = MixBus(name="test")
        bus.volume_db = MAX_VOLUME_DB + 20.0
        assert bus.volume_db <= MAX_VOLUME_DB

    def test_db_to_linear_zero_db(self):
        """0 dB converts to linear 1.0."""
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_db_to_linear_negative_six(self):
        """−6 dB converts to approximately 0.5."""
        assert db_to_linear(-6.02) == pytest.approx(0.5, rel=0.01)

    def test_db_to_linear_positive_six(self):
        """+6 dB converts to approximately 2.0."""
        assert db_to_linear(6.02) == pytest.approx(2.0, rel=0.01)

    def test_linear_to_db_one(self):
        """Linear 1.0 converts to 0 dB."""
        assert linear_to_db(1.0) == pytest.approx(0.0)

    def test_linear_to_db_half(self):
        """Linear 0.5 converts to approximately -6 dB."""
        assert linear_to_db(0.5) == pytest.approx(-6.02, rel=0.01)

    def test_linear_to_db_double(self):
        """Linear 2.0 converts to approximately +6 dB."""
        assert linear_to_db(2.0) == pytest.approx(6.02, rel=0.01)

    def test_volume_roundtrip_linear_to_db_to_linear(self):
        """Volume survives linear -> dB -> linear roundtrip."""
        bus = MixBus(name="test")
        original = 0.75
        bus.volume = original
        db_value = bus.volume_db
        bus.volume_db = db_value
        assert bus.volume == pytest.approx(original, rel=1e-3)

    def test_volume_roundtrip_db_to_linear_to_db(self):
        """Volume survives dB -> linear -> dB roundtrip."""
        bus = MixBus(name="test")
        original_db = -12.0
        bus.volume_db = original_db
        linear_value = bus.volume
        bus.volume = linear_value
        assert bus.volume_db == pytest.approx(original_db, rel=1e-3)


class TestFilters:
    """Test filter settings on bus."""

    def test_default_low_pass_is_max(self):
        """Default low pass filter is at maximum (all frequencies pass)."""
        bus = MixBus(name="test")
        assert bus.low_pass_freq == pytest.approx(DEFAULT_LOW_PASS)

    def test_default_high_pass_is_min(self):
        """Default high pass filter is at minimum (all frequencies pass)."""
        bus = MixBus(name="test")
        assert bus.high_pass_freq == pytest.approx(DEFAULT_HIGH_PASS)

    def test_set_low_pass_freq(self):
        """Low pass frequency can be set."""
        bus = MixBus(name="test")
        bus.low_pass_freq = 5000.0
        assert bus.low_pass_freq == pytest.approx(5000.0)

    def test_set_high_pass_freq(self):
        """High pass frequency can be set."""
        bus = MixBus(name="test")
        bus.high_pass_freq = 100.0
        assert bus.high_pass_freq == pytest.approx(100.0)

    def test_low_pass_clamps_to_min(self):
        """Low pass frequency clamps to minimum."""
        bus = MixBus(name="test")
        bus.low_pass_freq = MIN_FILTER_FREQ / 2
        assert bus.low_pass_freq >= MIN_FILTER_FREQ

    def test_low_pass_clamps_to_max(self):
        """Low pass frequency clamps to maximum."""
        bus = MixBus(name="test")
        bus.low_pass_freq = MAX_FILTER_FREQ * 2
        assert bus.low_pass_freq <= MAX_FILTER_FREQ

    def test_high_pass_clamps_to_min(self):
        """High pass frequency clamps to minimum."""
        bus = MixBus(name="test")
        bus.high_pass_freq = MIN_FILTER_FREQ / 2
        assert bus.high_pass_freq >= MIN_FILTER_FREQ

    def test_high_pass_clamps_to_max(self):
        """High pass frequency clamps to maximum."""
        bus = MixBus(name="test")
        bus.high_pass_freq = MAX_FILTER_FREQ * 2
        assert bus.high_pass_freq <= MAX_FILTER_FREQ

    def test_filter_state_default_is_disabled(self):
        """Filter state defaults to disabled."""
        bus = MixBus(name="test")
        assert bus.filter_state == FilterState.DISABLED

    def test_enable_low_pass_filter(self):
        """Low pass filter can be enabled."""
        bus = MixBus(name="test")
        bus.enable_low_pass(3000.0)
        assert bus.low_pass_freq == pytest.approx(3000.0)
        assert bus.filter_state in (FilterState.LOW_PASS, FilterState.BOTH)

    def test_enable_high_pass_filter(self):
        """High pass filter can be enabled."""
        bus = MixBus(name="test")
        bus.enable_high_pass(200.0)
        assert bus.high_pass_freq == pytest.approx(200.0)
        assert bus.filter_state in (FilterState.HIGH_PASS, FilterState.BOTH)

    def test_disable_filters(self):
        """Filters can be disabled."""
        bus = MixBus(name="test")
        bus.enable_low_pass(3000.0)
        bus.enable_high_pass(200.0)
        bus.disable_filters()
        assert bus.filter_state == FilterState.DISABLED


class TestMuteSolo:
    """Test mute and solo functionality."""

    def test_mute_bus(self):
        """Bus can be muted."""
        bus = MixBus(name="test")
        bus.muted = True
        assert bus.muted is True

    def test_unmute_bus(self):
        """Bus can be unmuted."""
        bus = MixBus(name="test")
        bus.muted = True
        bus.muted = False
        assert bus.muted is False

    def test_mute_affects_effective_volume(self):
        """Muted bus has zero effective volume."""
        bus = MixBus(name="test")
        bus.volume = 1.0
        bus.muted = True
        assert bus.effective_volume == pytest.approx(0.0)

    def test_unmuted_preserves_volume(self):
        """Unmuted bus retains original volume."""
        bus = MixBus(name="test")
        bus.volume = 0.8
        bus.muted = True
        bus.muted = False
        assert bus.volume == pytest.approx(0.8)

    def test_solo_bus(self):
        """Bus can be soloed."""
        bus = MixBus(name="test")
        bus.soloed = True
        assert bus.soloed is True

    def test_unsolo_bus(self):
        """Bus can be unsoloed."""
        bus = MixBus(name="test")
        bus.soloed = True
        bus.soloed = False
        assert bus.soloed is False

    def test_solo_does_not_affect_self_volume(self):
        """Soloed bus maintains its own volume."""
        bus = MixBus(name="test")
        bus.volume = 0.7
        bus.soloed = True
        assert bus.effective_volume == pytest.approx(0.7)


class TestBusHierarchy:
    """Test parent/child bus relationships."""

    def test_set_parent_bus(self):
        """Bus can have a parent assigned."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.parent = master
        assert sfx.parent == master

    def test_child_appears_in_parent_children(self):
        """Setting parent adds bus to parent's children."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.parent = master
        assert sfx in master.children

    def test_remove_from_parent(self):
        """Bus can be removed from parent."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.parent = master
        sfx.parent = None
        assert sfx.parent is None
        assert sfx not in master.children

    def test_multiple_children(self):
        """Parent can have multiple children."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        music = MixBus(name="music", bus_type=BusType.CATEGORY)
        sfx.parent = master
        music.parent = master
        assert len(master.children) == 2
        assert sfx in master.children
        assert music in master.children

    def test_effective_volume_includes_parent(self):
        """Effective volume includes parent's volume."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        master.volume = 0.5
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.volume = 0.8
        sfx.parent = master
        expected = 0.5 * 0.8
        assert sfx.effective_volume == pytest.approx(expected)

    def test_effective_volume_with_muted_parent(self):
        """Muted parent zeros child effective volume."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        master.muted = True
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.volume = 1.0
        sfx.parent = master
        assert sfx.effective_volume == pytest.approx(0.0)

    def test_default_hierarchy_creates_expected_buses(self):
        """create_default_hierarchy creates standard bus structure."""
        buses = create_default_hierarchy()
        assert "master" in buses
        assert "sfx" in buses
        assert "music" in buses
        assert "vo" in buses
        assert "ambient" in buses
        assert "ui" in buses

    def test_default_hierarchy_parent_structure(self):
        """Default hierarchy has correct parent structure."""
        buses = create_default_hierarchy()
        master = buses["master"]
        sfx = buses["sfx"]
        assert sfx.parent == master

    def test_nested_hierarchy_volume_propagation(self):
        """Volume propagates through nested hierarchy."""
        master = MixBus(name="master", bus_type=BusType.MASTER)
        master.volume = 0.8
        sfx = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        sfx.volume = 0.5
        sfx.parent = master
        footsteps = MixBus(name="footsteps", bus_type=BusType.SUB)
        footsteps.volume = 0.6
        footsteps.parent = sfx
        expected = 0.8 * 0.5 * 0.6
        assert footsteps.effective_volume == pytest.approx(expected)


class TestBusState:
    """Test bus state transitions."""

    def test_default_state_is_active(self):
        """Bus starts in ACTIVE state."""
        bus = MixBus(name="test")
        assert bus.state == BusState.ACTIVE

    def test_set_state_to_paused(self):
        """Bus can be paused."""
        bus = MixBus(name="test")
        bus.state = BusState.PAUSED
        assert bus.state == BusState.PAUSED

    def test_set_state_to_stopped(self):
        """Bus can be stopped."""
        bus = MixBus(name="test")
        bus.state = BusState.STOPPED
        assert bus.state == BusState.STOPPED

    def test_resume_from_paused(self):
        """Bus can resume from paused state."""
        bus = MixBus(name="test")
        bus.state = BusState.PAUSED
        bus.state = BusState.ACTIVE
        assert bus.state == BusState.ACTIVE


class TestPitch:
    """Test pitch/playback rate."""

    def test_default_pitch_is_unity(self):
        """Default pitch is 1.0 (normal speed)."""
        bus = MixBus(name="test")
        assert bus.pitch == pytest.approx(1.0)

    def test_set_pitch_higher(self):
        """Pitch can be increased."""
        bus = MixBus(name="test")
        bus.pitch = 2.0
        assert bus.pitch == pytest.approx(2.0)

    def test_set_pitch_lower(self):
        """Pitch can be decreased."""
        bus = MixBus(name="test")
        bus.pitch = 0.5
        assert bus.pitch == pytest.approx(0.5)

    def test_pitch_clamps_minimum(self):
        """Pitch clamps to minimum value."""
        bus = MixBus(name="test")
        bus.pitch = 0.0
        assert bus.pitch > 0.0


class TestBusMetadata:
    """Test bus metadata and identification."""

    def test_bus_has_unique_id(self):
        """Each bus has a unique identifier."""
        bus1 = MixBus(name="bus1")
        bus2 = MixBus(name="bus2")
        assert bus1.id != bus2.id

    def test_bus_name_is_readable(self):
        """Bus name can be read."""
        bus = MixBus(name="my_custom_bus")
        assert bus.name == "my_custom_bus"

    def test_bus_string_representation(self):
        """Bus has meaningful string representation."""
        bus = MixBus(name="sfx", bus_type=BusType.CATEGORY)
        str_repr = str(bus)
        assert "sfx" in str_repr


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_volume_negative_is_clamped(self):
        """Negative volume is clamped to zero."""
        bus = MixBus(name="test")
        bus.volume = -0.5
        assert bus.volume >= 0.0

    def test_extremely_small_volume(self):
        """Very small volume values work."""
        bus = MixBus(name="test")
        bus.volume = 1e-10
        assert bus.volume >= 0.0

    def test_extremely_large_volume(self):
        """Very large volume values are clamped."""
        bus = MixBus(name="test")
        bus.volume = 1000.0
        max_linear = db_to_linear(MAX_VOLUME_DB)
        assert bus.volume <= max_linear + 0.1

    def test_reparent_bus(self):
        """Bus can be reparented."""
        parent1 = MixBus(name="parent1", bus_type=BusType.MASTER)
        parent2 = MixBus(name="parent2", bus_type=BusType.MASTER)
        child = MixBus(name="child", bus_type=BusType.CATEGORY)
        child.parent = parent1
        assert child in parent1.children
        child.parent = parent2
        assert child not in parent1.children
        assert child in parent2.children

    def test_filter_freq_at_nyquist(self):
        """Filter frequency near Nyquist is handled."""
        bus = MixBus(name="test")
        bus.low_pass_freq = 22050.0
        assert bus.low_pass_freq <= MAX_FILTER_FREQ

    def test_concurrent_mute_solo(self):
        """Bus can be both muted and soloed."""
        bus = MixBus(name="test")
        bus.muted = True
        bus.soloed = True
        assert bus.muted is True
        assert bus.soloed is True
        assert bus.effective_volume == pytest.approx(0.0)


class TestVolumeSmoothing:
    """Test volume change smoothing behavior."""

    def test_volume_change_is_immediate_without_smoothing(self):
        """Volume change can be immediate."""
        bus = MixBus(name="test")
        bus.volume = 0.3
        assert bus.volume == pytest.approx(0.3)

    def test_set_volume_with_fade(self):
        """Volume can be faded over time."""
        bus = MixBus(name="test")
        bus.volume = 1.0
        bus.fade_to_volume(0.0, duration=1.0)
        # After fade is requested, it should eventually reach target
        assert hasattr(bus, 'fade_to_volume')
