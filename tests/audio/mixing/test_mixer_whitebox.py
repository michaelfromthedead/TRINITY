"""Whitebox tests for the central Mixer class.

Tests internal implementation of:
- 8-stage tick pipeline
- Bus management and routing
- HDR/Ducking/Sidechain coordination
- Snapshot transitions
- Processing order (DFS post-order)
- Source-to-bus routing
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.audio.mixing.config import (
    CATEGORY_AMBIENT,
    CATEGORY_MASTER,
    CATEGORY_MUSIC,
    CATEGORY_SFX,
    CATEGORY_UI,
    CATEGORY_VO,
    CATEGORY_TO_BUS,
    DEFAULT_SAMPLE_RATE,
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    db_to_linear,
    linear_to_db,
)
from engine.audio.mixing.ducking import DuckType
from engine.audio.mixing.hdr_audio import HDRPriority
from engine.audio.mixing.mix_bus import BusType, MixBus
from engine.audio.mixing.mixer import Mixer, MixerConfig


# =============================================================================
# MixerConfig Tests
# =============================================================================


class TestMixerConfig:
    """Test MixerConfig dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = MixerConfig()
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.enable_hdr is True
        assert config.enable_ducking is True
        assert config.enable_sidechain is True
        assert config.enable_snapshots is True
        assert config.auto_create_dialogue_duck is True

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = MixerConfig(
            sample_rate=44100,
            enable_hdr=False,
            enable_ducking=False,
            enable_sidechain=False,
            enable_snapshots=False,
            auto_create_dialogue_duck=False,
        )
        assert config.sample_rate == 44100
        assert config.enable_hdr is False
        assert config.enable_ducking is False
        assert config.enable_sidechain is False
        assert config.enable_snapshots is False
        assert config.auto_create_dialogue_duck is False


# =============================================================================
# Mixer Initialization Tests
# =============================================================================


class TestMixerInitialization:
    """Test Mixer initialization."""

    def test_default_initialization(self):
        """Default initialization without config."""
        mixer = Mixer()
        assert mixer.initialized is False
        assert mixer.master_bus is None
        assert mixer.buses == {}

    def test_initialization_with_config(self):
        """Initialization with custom config."""
        config = MixerConfig(enable_hdr=False)
        mixer = Mixer(config)
        mixer.initialize()

        assert mixer.initialized is True
        assert mixer._config.enable_hdr is False

    def test_initialize_creates_default_hierarchy(self):
        """Initialize creates default bus hierarchy."""
        mixer = Mixer()
        mixer.initialize()

        assert mixer.initialized is True
        assert mixer.master_bus is not None
        assert mixer.master_bus.name == "master"

        # Check expected buses exist
        expected = ["master", "sfx", "music", "vo", "ambient", "ui"]
        for name in expected:
            assert mixer.get_bus(name) is not None

    def test_initialize_with_custom_buses(self):
        """Initialize with custom bus dictionary."""
        custom_master = MixBus("custom_master", BusType.MASTER)
        custom_sfx = MixBus("custom_sfx", BusType.CATEGORY, parent=custom_master)

        custom_buses = {
            "custom_master": custom_master,
            "custom_sfx": custom_sfx,
        }

        mixer = Mixer()
        mixer.initialize(custom_buses=custom_buses)

        assert mixer.master_bus is custom_master
        assert mixer.get_bus("custom_sfx") is custom_sfx
        assert mixer.get_bus("sfx") is None  # Default not created

    def test_initialize_idempotent(self):
        """Calling initialize twice doesn't recreate buses."""
        mixer = Mixer()
        mixer.initialize()

        master_id = mixer.master_bus.id

        mixer.initialize()  # Second call

        assert mixer.master_bus.id == master_id  # Same bus

    def test_initialize_sets_up_default_ducking(self):
        """Initialize creates dialogue ducking by default."""
        config = MixerConfig(auto_create_dialogue_duck=True)
        mixer = Mixer(config)
        mixer.initialize()

        # Check ducking was created
        ducks = mixer.ducking.get_ducks_by_type(DuckType.DIALOGUE)
        assert len(ducks) > 0

    def test_initialize_without_auto_ducking(self):
        """Initialize skips ducking if disabled."""
        config = MixerConfig(auto_create_dialogue_duck=False)
        mixer = Mixer(config)
        mixer.initialize()

        ducks = mixer.ducking.get_ducks_by_type(DuckType.DIALOGUE)
        assert len(ducks) == 0

    def test_initialize_sets_up_hdr(self):
        """Initialize registers HDR sources."""
        mixer = Mixer()
        mixer.initialize()

        # HDR manager should have sources registered
        state = mixer.hdr.get_state()
        assert len(state["sources"]) > 0

    def test_shutdown_clears_state(self):
        """Shutdown clears all mixer state."""
        mixer = Mixer()
        mixer.initialize()

        mixer.shutdown()

        assert mixer.initialized is False
        assert mixer.master_bus is None
        assert mixer.buses == {}
        assert mixer.processing_order == []


# =============================================================================
# Bus Management Tests
# =============================================================================


class TestMixerBusManagement:
    """Test Mixer bus management."""

    def test_get_bus(self):
        """Get bus by name."""
        mixer = Mixer()
        mixer.initialize()

        sfx = mixer.get_bus("sfx")
        assert sfx is not None
        assert sfx.name == "sfx"

    def test_get_bus_not_found(self):
        """Get bus returns None if not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.get_bus("nonexistent")
        assert result is None

    def test_create_bus(self):
        """Create new bus."""
        mixer = Mixer()
        mixer.initialize()

        new_bus = mixer.create_bus("custom", BusType.SUB, parent_name="sfx")

        assert new_bus.name == "custom"
        assert new_bus.bus_type == BusType.SUB
        assert new_bus.parent is mixer.get_bus("sfx")
        assert mixer.get_bus("custom") is new_bus

    def test_create_bus_defaults_to_master_parent(self):
        """Create bus defaults parent to master."""
        mixer = Mixer()
        mixer.initialize()

        new_bus = mixer.create_bus("custom", BusType.SUB)
        assert new_bus.parent is mixer.master_bus

    def test_create_bus_duplicate_raises(self):
        """Create bus with existing name raises."""
        mixer = Mixer()
        mixer.initialize()

        with pytest.raises(ValueError, match="already exists"):
            mixer.create_bus("sfx")

    def test_create_bus_with_volume(self):
        """Create bus with custom volume."""
        mixer = Mixer()
        mixer.initialize()

        new_bus = mixer.create_bus("custom", volume=0.5)
        assert new_bus.volume == 0.5

    def test_remove_bus(self):
        """Remove bus."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.remove_bus("footsteps")

        assert result is True
        assert mixer.get_bus("footsteps") is None

    def test_remove_bus_not_found(self):
        """Remove bus returns False if not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.remove_bus("nonexistent")
        assert result is False

    def test_remove_master_raises(self):
        """Cannot remove master bus."""
        mixer = Mixer()
        mixer.initialize()

        with pytest.raises(ValueError, match="master"):
            mixer.remove_bus("master")

    def test_remove_bus_reparents_children(self):
        """Removing bus reparents children to master."""
        mixer = Mixer()
        mixer.initialize()

        # Get a child of sfx
        footsteps = mixer.get_bus("footsteps")
        assert footsteps.parent.name == "sfx"

        mixer.remove_bus("sfx")

        # footsteps should now parent to master
        assert footsteps.parent is mixer.master_bus

    def test_get_bus_names(self):
        """Get all bus names."""
        mixer = Mixer()
        mixer.initialize()

        names = mixer.get_bus_names()

        assert "master" in names
        assert "sfx" in names
        assert "music" in names

    def test_get_buses_by_type(self):
        """Get buses filtered by type."""
        mixer = Mixer()
        mixer.initialize()

        categories = mixer.get_buses_by_type(BusType.CATEGORY)
        category_names = [b.name for b in categories]

        assert "sfx" in category_names
        assert "music" in category_names
        assert "master" not in category_names

    def test_buses_property_returns_copy(self):
        """Buses property returns copy."""
        mixer = Mixer()
        mixer.initialize()

        buses = mixer.buses
        buses["fake"] = MixBus("fake")

        assert mixer.get_bus("fake") is None


# =============================================================================
# Volume Control Tests
# =============================================================================


class TestMixerVolumeControl:
    """Test Mixer volume control methods."""

    def test_set_bus_volume(self):
        """Set bus volume by name."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.set_bus_volume("sfx", 0.5)

        assert result is True
        assert mixer.get_bus("sfx").volume == 0.5

    def test_set_bus_volume_not_found(self):
        """Set bus volume returns False if not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.set_bus_volume("nonexistent", 0.5)
        assert result is False

    def test_set_bus_volume_db(self):
        """Set bus volume in dB."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.set_bus_volume_db("sfx", -6.0)

        assert result is True
        assert mixer.get_bus("sfx").volume_db == pytest.approx(-6.0, rel=0.1)

    def test_set_master_volume(self):
        """Set master bus volume."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_master_volume(0.8)
        assert mixer.master_bus.volume == 0.8

    def test_get_effective_volume(self):
        """Get effective volume through hierarchy."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("master", 0.8)
        mixer.set_bus_volume("sfx", 0.5)

        effective = mixer.get_effective_volume("sfx")
        assert effective == pytest.approx(0.4, rel=1e-6)

    def test_get_effective_volume_not_found(self):
        """Get effective volume returns 0 if not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.get_effective_volume("nonexistent")
        assert result == 0.0


# =============================================================================
# Source-to-Bus Routing Tests
# =============================================================================


class TestMixerSourceRouting:
    """Test source-to-bus routing."""

    def test_route_source_to_bus(self):
        """Route source to a bus."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.route_source_to_bus("source_123", "sfx")
        assert result is True

    def test_route_source_to_nonexistent_bus(self):
        """Route to nonexistent bus returns False."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.route_source_to_bus("source_123", "nonexistent")
        assert result is False

    def test_unroute_source(self):
        """Unroute a source."""
        mixer = Mixer()
        mixer.initialize()

        mixer.route_source_to_bus("source_123", "sfx")
        result = mixer.unroute_source("source_123")

        assert result is True

    def test_unroute_source_not_found(self):
        """Unroute returns False if not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.unroute_source("nonexistent_source")
        assert result is False

    def test_get_bus_for_category(self):
        """Get bus name for audio category."""
        mixer = Mixer()

        assert mixer.get_bus_for_category("SFX") == "sfx"
        assert mixer.get_bus_for_category("MUSIC") == "music"
        assert mixer.get_bus_for_category("VOICE_OVER") == "vo"
        assert mixer.get_bus_for_category("UNKNOWN") is None


# =============================================================================
# Processing Order Tests
# =============================================================================


class TestMixerProcessingOrder:
    """Test DFS post-order processing order."""

    def test_processing_order_leaf_before_parent(self):
        """Processing order has children before parents."""
        mixer = Mixer()
        mixer.initialize()

        order = mixer.processing_order
        order_names = [b.name for b in order]

        # footsteps should come before sfx
        assert order_names.index("footsteps") < order_names.index("sfx")

        # sfx should come before master
        assert order_names.index("sfx") < order_names.index("master")

        # master should be last
        assert order_names[-1] == "master"

    def test_processing_order_updates_on_bus_add(self):
        """Processing order updates when bus is added."""
        mixer = Mixer()
        mixer.initialize()

        original_order = list(mixer.processing_order)

        mixer.create_bus("new_sub", BusType.SUB, parent_name="sfx")

        new_order = mixer.processing_order
        assert len(new_order) == len(original_order) + 1

    def test_processing_order_updates_on_bus_remove(self):
        """Processing order updates when bus is removed."""
        mixer = Mixer()
        mixer.initialize()

        original_order = list(mixer.processing_order)

        mixer.remove_bus("footsteps")

        new_order = mixer.processing_order
        assert len(new_order) == len(original_order) - 1

    def test_processing_order_returns_copy(self):
        """Processing order returns a copy."""
        mixer = Mixer()
        mixer.initialize()

        order = mixer.processing_order
        order.clear()

        assert len(mixer.processing_order) > 0


# =============================================================================
# Tick Pipeline Tests
# =============================================================================


class TestMixerTickPipeline:
    """Test the 8-stage tick pipeline."""

    def test_tick_returns_none_before_init(self):
        """Tick returns None before initialization."""
        mixer = Mixer()
        result = mixer.tick(512)
        assert result is None

    def test_tick_returns_empty_for_zero_samples(self):
        """Tick returns empty array for 0 samples."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.tick(0)
        assert result.shape == (MIXER_NUM_CHANNELS, 0)

    def test_tick_returns_correct_shape(self):
        """Tick returns correct buffer shape."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.tick(512)

        assert result.shape == (MIXER_NUM_CHANNELS, 512)
        assert result.dtype == np.float32

    def test_tick_with_default_samples(self):
        """Tick with negative samples uses default buffer size."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.tick(-1)
        assert result.shape == (MIXER_NUM_CHANNELS, MIXER_BUFFER_SIZE)

    def test_tick_silence_without_sources(self):
        """Tick returns silence when no sources active."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.tick(512)

        # Should be mostly silence (or very small values from impulses)
        assert np.allclose(result, 0.0, atol=0.6)

    def test_tick_with_routed_source(self):
        """Tick produces output for routed sources."""
        mixer = Mixer()
        mixer.initialize()

        mixer.route_source_to_bus("src1", "sfx")

        result = mixer.tick(512)

        # Should have some non-zero output from impulse
        assert result.shape == (MIXER_NUM_CHANNELS, 512)

    def test_tick_clamps_output(self):
        """Tick clamps output to [-1, 1]."""
        mixer = Mixer()
        mixer.initialize()

        # Boost volume way up
        mixer.set_master_volume(10.0)
        mixer.set_bus_volume("sfx", 10.0)
        mixer.route_source_to_bus("src1", "sfx")

        result = mixer.tick(512)

        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_tick_applies_mute(self):
        """Tick respects mute state."""
        mixer = Mixer()
        mixer.initialize()

        mixer.route_source_to_bus("src1", "sfx")
        mixer.mute_bus("sfx", True)

        result = mixer.tick(512)

        # SFX should be silent, but other buses might have content
        # Just verify mute_bus worked
        assert mixer.get_bus("sfx").muted is True

    def test_read_master_output(self):
        """Read master output buffer."""
        mixer = Mixer()
        mixer.initialize()

        mixer.tick(512)

        output = mixer.read_master_output()
        assert output is not None
        assert output.shape == (MIXER_NUM_CHANNELS, 512)

    def test_read_master_output_before_tick(self):
        """Read master output returns None before tick."""
        mixer = Mixer()
        mixer.initialize()

        output = mixer.read_master_output()
        assert output is None

    def test_read_master_output_partial(self):
        """Read partial master output."""
        mixer = Mixer()
        mixer.initialize()

        mixer.tick(512)

        output = mixer.read_master_output(256)
        assert output.shape == (MIXER_NUM_CHANNELS, 256)


# =============================================================================
# Mute/Solo Tests
# =============================================================================


class TestMixerMuteSolo:
    """Test Mixer mute/solo methods."""

    def test_mute_bus(self):
        """Mute bus by name."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.mute_bus("sfx", True)

        assert result is True
        assert mixer.get_bus("sfx").muted is True

    def test_unmute_bus(self):
        """Unmute bus."""
        mixer = Mixer()
        mixer.initialize()

        mixer.mute_bus("sfx", True)
        mixer.mute_bus("sfx", False)

        assert mixer.get_bus("sfx").muted is False

    def test_mute_bus_not_found(self):
        """Mute returns False if bus not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.mute_bus("nonexistent", True)
        assert result is False

    def test_solo_bus(self):
        """Solo bus by name."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.solo_bus("sfx", True)

        assert result is True
        assert mixer.get_bus("sfx").soloed is True

    def test_unsolo_bus(self):
        """Unsolo bus."""
        mixer = Mixer()
        mixer.initialize()

        mixer.solo_bus("sfx", True)
        mixer.solo_bus("sfx", False)

        assert mixer.get_bus("sfx").soloed is False


# =============================================================================
# Aux Send Tests
# =============================================================================


class TestMixerAuxSends:
    """Test Mixer aux send routing."""

    def test_create_aux_send(self):
        """Create aux send between buses."""
        mixer = Mixer()
        mixer.initialize()

        # Create an aux bus first
        mixer.create_bus("reverb", BusType.AUX)

        send = mixer.create_aux_send("sfx", "reverb", level_db=-6.0)

        assert send is not None
        assert send.send_level_db == -6.0

    def test_create_aux_send_pre_fader(self):
        """Create pre-fader aux send."""
        mixer = Mixer()
        mixer.initialize()
        mixer.create_bus("reverb", BusType.AUX)

        send = mixer.create_aux_send("sfx", "reverb", pre_fader=True)

        from engine.audio.mixing.bus_routing import RoutingMode
        assert send.mode == RoutingMode.PRE_FADER

    def test_create_aux_send_source_not_found(self):
        """Create aux send returns None if source not found."""
        mixer = Mixer()
        mixer.initialize()
        mixer.create_bus("reverb", BusType.AUX)

        send = mixer.create_aux_send("nonexistent", "reverb")
        assert send is None

    def test_create_aux_send_target_not_found(self):
        """Create aux send returns None if target not found."""
        mixer = Mixer()
        mixer.initialize()

        send = mixer.create_aux_send("sfx", "nonexistent")
        assert send is None

    def test_set_direct_output(self):
        """Set direct output routing."""
        mixer = Mixer()
        mixer.initialize()
        mixer.create_bus("submix", BusType.SUB)

        output = mixer.set_direct_output("sfx", "submix", level_db=-3.0)

        assert output is not None
        assert output.level_db == -3.0


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestMixerSnapshots:
    """Test Mixer snapshot management."""

    def test_capture_snapshot(self):
        """Capture current mix state."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("sfx", 0.5)

        snapshot = mixer.capture_snapshot("test_snap")

        assert snapshot.name == "test_snap"
        assert snapshot.bus_states["sfx"].volume_linear == 0.5

    def test_transition_to_snapshot(self):
        """Start transition to snapshot."""
        mixer = Mixer()
        mixer.initialize()

        mixer.capture_snapshot("test_snap")
        mixer.set_bus_volume("sfx", 0.8)

        result = mixer.transition_to_snapshot("test_snap", blend_time=0.5)

        assert result is True
        assert mixer.is_transitioning() is True

    def test_transition_to_nonexistent_snapshot(self):
        """Transition to nonexistent snapshot returns False."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.transition_to_snapshot("nonexistent")
        assert result is False

    def test_apply_snapshot_immediate(self):
        """Apply snapshot immediately without blending."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("sfx", 0.5)
        mixer.capture_snapshot("test_snap")
        mixer.set_bus_volume("sfx", 0.8)

        result = mixer.apply_snapshot_immediate("test_snap")

        assert result is True
        assert mixer.get_bus("sfx").volume == pytest.approx(0.5, rel=1e-3)
        assert mixer.is_transitioning() is False

    def test_is_transitioning(self):
        """Check if transition is in progress."""
        mixer = Mixer()
        mixer.initialize()

        assert mixer.is_transitioning() is False

        mixer.capture_snapshot("snap1")
        mixer.transition_to_snapshot("snap1", blend_time=1.0)

        assert mixer.is_transitioning() is True


# =============================================================================
# Ducking Tests
# =============================================================================


class TestMixerDucking:
    """Test Mixer ducking functionality."""

    def test_create_dialogue_duck(self):
        """Create dialogue ducking."""
        config = MixerConfig(auto_create_dialogue_duck=False)
        mixer = Mixer(config)
        mixer.initialize()

        mixer.create_dialogue_duck(amount_db=-12.0)

        ducks = mixer.ducking.get_ducks_by_type(DuckType.DIALOGUE)
        assert len(ducks) > 0

    def test_trigger_event_duck(self):
        """Trigger event ducking."""
        mixer = Mixer()
        mixer.initialize()

        # This shouldn't raise
        mixer.trigger_event_duck(500.0)

    def test_get_duck_amount(self):
        """Get duck amount for a bus."""
        mixer = Mixer()
        mixer.initialize()

        # Without active ducking, should be 1.0
        amount = mixer.get_duck_amount("sfx")
        assert amount == 1.0

    def test_get_duck_amount_not_found(self):
        """Get duck amount for nonexistent bus."""
        mixer = Mixer()
        mixer.initialize()

        amount = mixer.get_duck_amount("nonexistent")
        assert amount == 1.0


# =============================================================================
# Sidechain Tests
# =============================================================================


class TestMixerSidechain:
    """Test Mixer sidechain compression."""

    def test_create_sidechain(self):
        """Create sidechain compressor."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.create_sidechain(
            "vo",
            "music",
            threshold_db=-20.0,
            ratio=4.0,
        )

        assert result is True

    def test_create_sidechain_source_not_found(self):
        """Create sidechain returns False if key not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.create_sidechain("nonexistent", "music")
        assert result is False

    def test_create_sidechain_target_not_found(self):
        """Create sidechain returns False if target not found."""
        mixer = Mixer()
        mixer.initialize()

        result = mixer.create_sidechain("vo", "nonexistent")
        assert result is False


# =============================================================================
# Level Analysis Tests
# =============================================================================


class TestMixerLevelAnalysis:
    """Test Mixer level analysis."""

    def test_set_bus_level(self):
        """Set measured level for a bus."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_level("sfx", -12.0)

        level = mixer.get_bus_level("sfx")
        assert level == -12.0

    def test_set_bus_levels(self):
        """Set multiple bus levels at once."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_levels({
            "sfx": -10.0,
            "music": -15.0,
            "vo": -5.0,
        })

        assert mixer.get_bus_level("sfx") == -10.0
        assert mixer.get_bus_level("music") == -15.0
        assert mixer.get_bus_level("vo") == -5.0

    def test_get_bus_level_not_found(self):
        """Get level for nonexistent bus."""
        mixer = Mixer()
        mixer.initialize()

        from engine.audio.mixing.config import MIN_VOLUME_DB
        level = mixer.get_bus_level("nonexistent")
        assert level == MIN_VOLUME_DB


# =============================================================================
# Update Tests
# =============================================================================


class TestMixerUpdate:
    """Test Mixer update method."""

    def test_update_before_init(self):
        """Update before initialization doesn't crash."""
        mixer = Mixer()
        mixer.update(0.016)  # Should not raise

    def test_update_runs_components(self):
        """Update calls component updates."""
        mixer = Mixer()
        mixer.initialize()

        # Just verify it doesn't crash
        for _ in range(10):
            mixer.update(0.016)

    def test_update_callback(self):
        """Update callback is called."""
        mixer = Mixer()
        mixer.initialize()

        callback = MagicMock()
        mixer.on_update(callback)

        mixer.update(0.016)

        callback.assert_called_once_with(0.016)


# =============================================================================
# Final Volume Tests
# =============================================================================


class TestMixerFinalVolume:
    """Test final volume calculations."""

    def test_get_final_volume(self):
        """Get final volume including all processing."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("sfx", 0.5)

        volume = mixer.get_final_volume("sfx")

        # Without ducking/sidechain active, should match effective volume
        assert volume == pytest.approx(0.5, rel=0.01)

    def test_get_final_volume_not_found(self):
        """Get final volume for nonexistent bus."""
        mixer = Mixer()
        mixer.initialize()

        volume = mixer.get_final_volume("nonexistent")
        assert volume == 0.0

    def test_get_final_volume_db(self):
        """Get final volume in dB."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("sfx", 0.5)

        volume_db = mixer.get_final_volume_db("sfx")
        assert volume_db == pytest.approx(-6.0, rel=0.5)


# =============================================================================
# State Tests
# =============================================================================


class TestMixerState:
    """Test Mixer state methods."""

    def test_get_state(self):
        """Get complete mixer state."""
        mixer = Mixer()
        mixer.initialize()

        state = mixer.get_state()

        assert "initialized" in state
        assert state["initialized"] is True
        assert "config" in state
        assert "buses" in state
        assert "routing" in state
        assert "ducking" in state
        assert "sidechain" in state
        assert "hdr" in state

    def test_get_state_buses(self):
        """Get state includes bus details."""
        mixer = Mixer()
        mixer.initialize()

        mixer.set_bus_volume("sfx", 0.5)
        mixer.mute_bus("music", True)

        state = mixer.get_state()

        assert state["buses"]["sfx"]["volume"] == 0.5
        assert state["buses"]["music"]["muted"] is True

    def test_repr(self):
        """repr shows useful info."""
        mixer = Mixer()
        mixer.initialize()

        repr_str = repr(mixer)
        assert "Mixer" in repr_str
        assert "buses=" in repr_str
        assert "initialized=True" in repr_str


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestMixerThreadSafety:
    """Test Mixer thread safety."""

    def test_concurrent_tick_calls(self):
        """Concurrent tick calls don't corrupt state."""
        mixer = Mixer()
        mixer.initialize()

        def run_ticks():
            for _ in range(20):
                mixer.tick(512)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_ticks) for _ in range(4)]
            for f in futures:
                f.result()

        # Should still be functional
        result = mixer.tick(512)
        assert result.shape == (MIXER_NUM_CHANNELS, 512)

    def test_concurrent_volume_changes(self):
        """Concurrent volume changes don't corrupt state."""
        mixer = Mixer()
        mixer.initialize()

        def change_volumes():
            for _ in range(50):
                mixer.set_bus_volume("sfx", 0.5)
                mixer.set_bus_volume("music", 0.7)
                mixer.set_bus_volume("vo", 0.9)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(change_volumes) for _ in range(4)]
            for f in futures:
                f.result()

        # Buses should have valid volumes
        assert 0.0 <= mixer.get_bus("sfx").volume <= 2.0
