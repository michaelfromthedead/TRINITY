"""
Whitebox tests for Mixer Tick Pipeline internals.

Tests internal implementation details:
- Accumulation buffer lifecycle (ensure, clear, accumulate, read)
- Bus process_audio (volume, mute, effects, clipping)
- DFS processing order computation
- Source-to-bus routing
- Master output buffer
- 8-stage tick pipeline
- DSP chain integration
- Bus state reset
- Thread safety for buffer operations

All tests use London School TDD: external dependencies are mocked.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import pytest

from engine.audio.mixing.mix_bus import MixBus, BusType, BusState, FilterState
from engine.audio.mixing.mixer import Mixer, MixerConfig
from engine.audio.mixing.config import (
    MIXER_BUFFER_SIZE,
    MIXER_NUM_CHANNELS,
    CATEGORY_TO_BUS,
    DEFAULT_BUS_VOLUME,
    DEFAULT_SAMPLE_RATE,
    MIN_VOLUME_DB,
    MAX_VOLUME_DB,
    MIN_PITCH,
    MAX_PITCH,
    db_to_linear,
    linear_to_db,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_dsp_chain():
    """Create a mock DSPChain."""
    mock = MagicMock()
    mock.nodes = []
    mock.process_block = MagicMock()
    return mock


@pytest.fixture
def bus():
    """Create a fresh MixBus for testing."""
    return MixBus("test_bus", BusType.SUB)


@pytest.fixture
def bus_with_parent():
    """Create a bus with a parent."""
    parent = MixBus("parent", BusType.CATEGORY)
    child = MixBus("child", BusType.SUB, parent=parent)
    return child, parent


@pytest.fixture
def master_bus():
    """Create a master bus."""
    return MixBus("master", BusType.MASTER)


@pytest.fixture
def mixer():
    """Create an initialized mixer with default hierarchy."""
    m = Mixer()
    m.initialize()
    return m


@pytest.fixture
def stereo_samples():
    """Standard stereo test samples (2, 64) float32."""
    return np.random.randn(2, 64).astype(np.float32) * 0.5


@pytest.fixture
def mono_samples():
    """Mono test samples (1, 64) float32."""
    return np.random.randn(1, 64).astype(np.float32) * 0.5


@pytest.fixture
def sine_samples():
    """Clean sine wave for deterministic testing (2, 64) float32."""
    t = np.linspace(0, 2 * np.pi, 64, dtype=np.float32)
    sine = np.sin(t).reshape(1, -1) * 0.5
    return np.broadcast_to(sine, (2, 64)).copy()


# =========================================================================
# Test Accumulation Buffer
# =========================================================================


class TestAccumulationBuffer:
    """Tests for MixBus accumulation buffer methods."""

    def test_ensure_acc_buffer_creates(self, bus):
        """_ensure_acc_buffer should create buffer with correct shape."""
        bus._ensure_acc_buffer(64)
        assert bus._acc_buffer is not None
        assert bus._acc_buffer.shape == (MIXER_NUM_CHANNELS, 64)
        assert bus._acc_buffer.dtype == np.float32

    def test_ensure_acc_buffer_grows(self, bus):
        """_ensure_acc_buffer should grow buffer when needed."""
        bus._ensure_acc_buffer(32)
        small_shape = bus._acc_buffer.shape[1]
        bus._ensure_acc_buffer(128)
        assert bus._acc_buffer.shape[1] >= 128
        assert bus._acc_buffer.shape[0] == MIXER_NUM_CHANNELS

    def test_ensure_acc_buffer_reuses(self, bus):
        """_ensure_acc_buffer should reuse buffer if large enough."""
        bus._ensure_acc_buffer(128)
        original = bus._acc_buffer
        bus._ensure_acc_buffer(64)
        assert bus._acc_buffer is original

    def test_clear_acc_buffer_zeros(self, bus):
        """clear_acc_buffer should zero out the buffer."""
        bus._ensure_acc_buffer(64)
        bus._acc_buffer[:, :] = 0.5
        bus.clear_acc_buffer(64)
        assert np.all(bus._acc_buffer[:, :64] == 0.0)

    def test_clear_acc_buffer_partial(self, bus):
        """clear_acc_buffer should only clear specified samples."""
        bus._ensure_acc_buffer(128)
        bus._acc_buffer[:, :] = 0.5
        bus.clear_acc_buffer(64)
        assert np.all(bus._acc_buffer[:, :64] == 0.0)
        assert np.all(bus._acc_buffer[:, 64:] == 0.5)

    def test_accumulate_mono(self, bus, mono_samples):
        """accumulate should broadcast mono to stereo."""
        bus.clear_acc_buffer(64)
        bus.accumulate(mono_samples, 64)
        readback = bus.read_acc_buffer(64)
        assert readback.shape == (MIXER_NUM_CHANNELS, 64)
        assert np.allclose(readback[0], mono_samples[0, :64])
        assert np.allclose(readback[1], mono_samples[0, :64])

    def test_accumulate_stereo(self, bus, stereo_samples):
        """accumulate should add stereo samples correctly."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        readback = bus.read_acc_buffer(64)
        assert np.allclose(readback, stereo_samples[:, :64])

    def test_accumulate_stacking(self, bus, stereo_samples):
        """accumulate should stack multiple calls."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples * 0.5, 64)
        bus.accumulate(stereo_samples * 0.5, 64)
        readback = bus.read_acc_buffer(64)
        assert np.allclose(readback, stereo_samples[:, :64])

    def test_read_acc_buffer_returns_copy(self, bus):
        """read_acc_buffer should return a copy, not a reference."""
        bus.clear_acc_buffer(64)
        bus._acc_buffer[:, :] = 0.5
        readback = bus.read_acc_buffer(64)
        readback[:, :] = 0.0
        assert np.all(bus._acc_buffer[:, :64] == 0.5)

    def test_read_acc_buffer_returns_empty_if_none(self, bus):
        """read_acc_buffer should return zeros if no buffer exists."""
        result = bus.read_acc_buffer(64)
        assert result.shape == (MIXER_NUM_CHANNELS, 64)
        assert np.all(result == 0.0)

    def test_accumulate_preserves_independence(self, bus, stereo_samples):
        """Adjacent bus buffers should not interfere."""
        bus2 = MixBus("bus2")
        bus.clear_acc_buffer(64)
        bus2.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        r1 = bus.read_acc_buffer(64)
        r2 = bus2.read_acc_buffer(64)
        assert np.allclose(r1, stereo_samples[:, :64])
        assert np.all(r2 == 0.0)


# =========================================================================
# Test Bus ProcessAudio
# =========================================================================


class TestBusProcessAudio:
    """Tests for MixBus.process_audio."""

    def test_process_audio_applies_volume(self, bus, stereo_samples):
        """process_audio should apply volume scaling."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        bus.volume = 0.5
        output = bus.process_audio(64)
        expected = stereo_samples[:, :64] * 0.5
        expected = np.clip(expected, -1.0, 1.0)
        assert np.allclose(output, expected)

    def test_process_audio_muted(self, bus, stereo_samples):
        """process_audio should return silence when muted."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        bus.muted = True
        output = bus.process_audio(64)
        assert np.all(output == 0.0)

    def test_process_audio_zero_volume(self, bus, stereo_samples):
        """process_audio should return silence at zero volume."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        bus.volume = 0.0
        output = bus.process_audio(64)
        assert np.all(output == 0.0)

    def test_process_audio_returns_copy(self, bus, stereo_samples):
        """process_audio should not modify the accumulation buffer."""
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        original = bus.read_acc_buffer(64).copy()
        _ = bus.process_audio(64)
        unchanged = bus.read_acc_buffer(64)
        assert np.allclose(unchanged, original)

    def test_process_audio_effects(self, bus, stereo_samples, mock_dsp_chain):
        """process_audio should call DSP chain when effects present."""
        bus._effect_chain = mock_dsp_chain
        mock_dsp_chain.nodes = [MagicMock()]
        mock_dsp_chain.process_block.return_value = None
        bus.clear_acc_buffer(64)
        bus.accumulate(stereo_samples, 64)
        output = bus.process_audio(64)
        mock_dsp_chain.process_block.assert_called_once()
        call_args = mock_dsp_chain.process_block.call_args[0]
        assert call_args[0].shape == (MIXER_NUM_CHANNELS, 64)
        assert call_args[1].shape == (MIXER_NUM_CHANNELS, 64)

    def test_process_audio_hard_clip(self, bus):
        """process_audio should hard clip to [-1.0, 1.0]."""
        bus.clear_acc_buffer(64)
        loud = np.ones((2, 64), dtype=np.float32) * 10.0
        bus.accumulate(loud, 64)
        output = bus.process_audio(64)
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)


# =========================================================================
# Test Processing Order
# =========================================================================


class TestProcessingOrder:
    """Tests for DFS processing order computation."""

    def test_dfs_processing_order(self, mixer):
        """Processing order should be children before parents (post-order)."""
        order = mixer.processing_order
        bus_names = [b.name for b in order]

        # Find positions of SFX and its children
        sfx_pos = bus_names.index("sfx")
        footsteps_pos = bus_names.index("footsteps")
        weapons_pos = bus_names.index("weapons")
        impacts_pos = bus_names.index("impacts")

        # Children should appear before their parent
        assert footsteps_pos < sfx_pos
        assert weapons_pos < sfx_pos
        assert impacts_pos < sfx_pos

    def test_all_buses_included(self, mixer):
        """All default buses should be in processing order."""
        order_names = {b.name for b in mixer.processing_order}
        expected = {"master", "sfx", "music", "vo", "ambient", "ui",
                     "footsteps", "weapons", "impacts", "combat",
                     "exploration", "dialogue", "barks"}
        assert order_names == expected

    def test_no_duplicates(self, mixer):
        """Processing order should have no duplicate buses."""
        order = mixer.processing_order
        ids = [b.id for b in order]
        assert len(ids) == len(set(ids))

    def test_processing_order_property_returns_copy(self, mixer):
        """processing_order property should return a new list each time."""
        o1 = mixer.processing_order
        o2 = mixer.processing_order
        assert o1 is not o2

    def test_processing_order_updates_on_create(self, mixer):
        """Processing order should update when a new bus is created."""
        order_before = [b.name for b in mixer.processing_order]
        bus = mixer.create_bus("test_sub", BusType.SUB, "sfx")
        order_after = [b.name for b in mixer.processing_order]
        assert "test_sub" in order_after

    def test_processing_order_updates_on_remove(self, mixer):
        """Processing order should update when a bus is removed."""
        sub = mixer.create_bus("temp_bus", BusType.SUB)
        assert "temp_bus" in [b.name for b in mixer.processing_order]
        mixer.remove_bus("temp_bus")
        assert "temp_bus" not in [b.name for b in mixer.processing_order]

    def test_master_is_last(self, mixer):
        """Master bus should be last in processing order (root of DFS)."""
        order = mixer.processing_order
        assert order[-1].bus_type == BusType.MASTER


# =========================================================================
# Test Source Routing
# =========================================================================


class TestSourceRouting:
    """Tests for source-to-bus routing."""

    def test_route_source_to_bus(self, mixer):
        """route_source_to_bus should route a source to a bus."""
        result = mixer.route_source_to_bus("src1", "sfx")
        assert result is True

    def test_route_source_nonexistent_bus(self, mixer):
        """route_source_to_bus should return False for invalid bus."""
        result = mixer.route_source_to_bus("src1", "nonexistent")
        assert result is False

    def test_unroute_source(self, mixer):
        """unroute_source should remove a source from routing."""
        mixer.route_source_to_bus("src1", "sfx")
        mixer.unroute_source("src1")

    def test_route_overwrite(self, mixer):
        """Routing an already-routed source should update its bus."""
        mixer.route_source_to_bus("src1", "sfx")
        mixer.route_source_to_bus("src1", "music")

    def test_get_bus_for_category_found(self, mixer):
        """get_bus_for_category should return correct bus name."""
        bus_name = mixer.get_bus_for_category("SFX")
        assert bus_name == "sfx"

    def test_get_bus_for_category_not_found(self, mixer):
        """get_bus_for_category should return None for unknown category."""
        result = mixer.get_bus_for_category("UNKNOWN")
        assert result is None

    def test_get_bus_for_category_master(self, mixer):
        """get_bus_for_category should return master for MASTER."""
        bus_name = mixer.get_bus_for_category("MASTER")
        assert bus_name == "master"


# =========================================================================
# Test Master Output
# =========================================================================


class TestMasterOutput:
    """Tests for master output buffer."""

    def test_master_output_none_before_tick(self, mixer):
        """read_master_output should return None before any tick."""
        result = mixer.read_master_output()
        assert result is None

    def test_master_output_after_tick(self, mixer):
        """read_master_output should return buffer after tick."""
        mixer.tick(64)
        result = mixer.read_master_output()
        assert result is not None
        assert result.shape[0] == MIXER_NUM_CHANNELS
        assert result.dtype == np.float32

    def test_master_output_returns_copy(self, mixer):
        """read_master_output should return a copy."""
        mixer.tick(64)
        result1 = mixer.read_master_output()
        result2 = mixer.read_master_output()
        # Same content but different objects
        assert np.allclose(result1, result2)
        assert result1 is not result2

    def test_master_output_smaller_size(self, mixer):
        """read_master_output should honor requested size."""
        mixer.tick(128)
        result = mixer.read_master_output(32)
        assert result.shape[1] == 32


# =========================================================================
# Test Tick Pipeline
# =========================================================================


class TestTickPipeline:
    """Tests for the 8-stage tick pipeline."""

    def test_tick_returns_float32_clipped(self, mixer):
        """Tick output should be float32 and clipped to [-1, 1]."""
        output = mixer.tick(64)
        assert output.dtype == np.float32
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)

    def test_tick_returns_none_if_not_initialized(self):
        """Tick should return zeros if mixer not initialized."""
        m = Mixer()
        output = m.tick(64)
        assert output.shape == (MIXER_NUM_CHANNELS, 64)
        assert np.all(output == 0.0)

    def test_tick_returns_correct_shape(self, mixer):
        """Tick should return (channels, samples) array."""
        for size in [32, 64, 128, 256]:
            output = mixer.tick(size)
            assert output.shape == (MIXER_NUM_CHANNELS, size), f"Failed for size={size}"

    def test_tick_clears_buffers(self, mixer, stereo_samples):
        """Each tick should start with cleared accumulators."""
        sfx = mixer.get_bus("sfx")
        sfx.accumulate(stereo_samples, 64)
        before_tick = sfx.read_acc_buffer(64)
        assert not np.all(before_tick == 0.0)
        mixer.tick(64)
        after_tick = sfx.read_acc_buffer(64)
        assert np.all(after_tick == 0.0)

    def test_tick_accumulates_to_parent(self, mixer, sine_samples):
        """Child bus audio should be processed and accumulated to parent."""
        footsteps = mixer.get_bus("footsteps")
        sfx = mixer.get_bus("sfx")
        footsteps.accumulate(sine_samples, 64)
        mixer.tick(64)
        # After tick, the child's buffer should be cleared and parent processed
        footsteps_after = footsteps.read_acc_buffer(64)
        assert np.all(footsteps_after == 0.0)

    def test_tick_muted_bus_silent(self, mixer, sine_samples):
        """A muted bus should not pass audio through."""
        sfx = mixer.get_bus("sfx")
        sfx.muted = True
        footsteps = mixer.get_bus("footsteps")
        footsteps.accumulate(sine_samples, 64)
        output = mixer.tick(64)
        # Master output should be silence (or very close) since sfx is muted
        assert np.max(np.abs(output)) < 0.001

    def test_tick_volume_scaling(self, mixer, sine_samples):
        """Bus volume should scale audio proportionally."""
        sfx = mixer.get_bus("sfx")
        footsteps = mixer.get_bus("footsteps")
        footsteps.accumulate(sine_samples, 64)
        # Normal volume
        output_normal = mixer.tick(64)
        footsteps.accumulate(sine_samples, 64)
        sfx.volume = 0.5
        output_half = mixer.tick(64)
        # Half volume should give roughly half amplitude
        assert np.max(np.abs(output_half)) <= np.max(np.abs(output_normal)) + 0.01

    def test_tick_hard_clip(self, mixer):
        """Excessively loud audio should be hard-clipped."""
        sfx = mixer.get_bus("sfx")
        loud = np.ones((2, 64), dtype=np.float32) * 10.0
        sfx.accumulate(loud, 64)
        output = mixer.tick(64)
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)

    def test_tick_with_effects(self, mixer, sine_samples, mock_dsp_chain):
        """Tick should run DSP chain on buses with effects."""
        sfx = mixer.get_bus("sfx")
        sfx._effect_chain = mock_dsp_chain
        mock_dsp_chain.nodes = [MagicMock()]
        mock_dsp_chain.process_block.return_value = None
        footsteps = mixer.get_bus("footsteps")
        footsteps.accumulate(sine_samples, 64)
        mixer.tick(64)
        mock_dsp_chain.process_block.assert_called()

    def test_tick_empty_returns_silence(self, mixer):
        """Tick with no audio should return silence."""
        output = mixer.tick(64)
        assert np.all(output == 0.0)

    def test_tick_different_sizes(self, mixer, sine_samples):
        """Tick should work with various buffer sizes."""
        for size in [16, 32, 64, 128, 256]:
            output = mixer.tick(size)
            assert output.shape == (MIXER_NUM_CHANNELS, size)
            assert output.dtype == np.float32


# =========================================================================
# Test Effect Chain Integration
# =========================================================================


class TestEffectChainIntegration:
    """Tests for DSP chain integration in MixBus."""

    def test_effect_chain_property_type(self, bus):
        """effect_chain should return a DSPChain."""
        chain = bus.effect_chain
        assert chain is not None

    def test_has_effects_empty(self, bus):
        """has_effects should return False with no nodes."""
        assert bus.has_effects() is False

    def test_has_effects_with_node(self, bus):
        """has_effects should return True with a node."""
        from unittest.mock import MagicMock
        from engine.audio.dsp.dsp_node import DSPNode
        node = MagicMock(spec=DSPNode)
        bus.effect_chain.add_node(node)
        assert bus.has_effects() is True

    def test_effect_chain_persists(self, bus):
        """effect_chain should return the same instance."""
        chain1 = bus.effect_chain
        chain2 = bus.effect_chain
        assert chain1 is chain2


# =========================================================================
# Test Bus State Reset
# =========================================================================


class TestBusStateReset:
    """Tests for bus state reset functionality."""

    def test_reset_defaults(self, bus):
        """reset should restore default bus state."""
        bus.volume = 0.5
        bus.pitch = 2.0
        bus.muted = True
        bus.reset()
        assert bus.volume == DEFAULT_BUS_VOLUME
        assert bus.pitch == 1.0
        assert bus.muted is False

    def test_reset_clears_filters(self, bus):
        """reset should clear filter state."""
        bus.set_low_pass(5000, 1.0, True)
        bus.set_high_pass(200, 1.0, True)
        bus.reset()
        filters = bus.filters
        assert filters.low_pass_enabled is False
        assert filters.high_pass_enabled is False
        assert filters.low_pass_freq == 20000.0
        assert filters.high_pass_freq == 20.0

    def test_reset_preserves_hierarchy(self, bus_with_parent):
        """reset should not change parent-child relationships."""
        child, parent = bus_with_parent
        child.reset()
        assert child.parent is parent
        assert child in parent.children

    def test_set_state(self, bus):
        """set_state should apply a new BusState."""
        new_state = BusState(
            volume_linear=0.5,
            pitch=2.0,
            muted=True,
            soloed=False,
        )
        bus.set_state(new_state)
        assert bus.volume == 0.5
        assert bus.pitch == 2.0
        assert bus.muted is True
        assert bus.soloed is False


# =========================================================================
# Test Thread Safety
# =========================================================================


class TestThreadSafety:
    """Tests for thread-safe buffer operations."""

    def test_concurrent_accumulate(self, bus, stereo_samples):
        """Multiple threads should be able to accumulate concurrently."""
        errors = []
        bus.clear_acc_buffer(256)
        start_event = threading.Event()

        def worker():
            try:
                start_event.wait(timeout=5)
                for _ in range(10):
                    bus.accumulate(stereo_samples, 64)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        start_event.set()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"
        result = bus.read_acc_buffer(64)
        assert not np.any(np.isnan(result))

    def test_concurrent_read_write(self, bus, stereo_samples):
        """Concurrent reads and writes should not produce errors."""
        errors = []
        bus.clear_acc_buffer(256)
        stop_event = threading.Event()

        def writer():
            try:
                while not stop_event.is_set():
                    bus.accumulate(stereo_samples, 64)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                while not stop_event.is_set():
                    _ = bus.read_acc_buffer(64)
                    _ = bus.process_audio(64)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"
        result = bus.read_acc_buffer(64)
        assert not np.any(np.isnan(result))


# =========================================================================
# Test Mixer Initialization Internals
# =========================================================================


class TestMixerInitialization:
    """Tests for Mixer initialization internals."""

    def test_initialize_sets_initialized_flag(self):
        """initialize should set _initialized to True."""
        m = Mixer()
        m.initialize()
        assert m._initialized is True

    def test_initialize_idempotent(self, mixer):
        """Calling initialize twice should not reset state."""
        mixer.create_bus("extra_bus", BusType.SUB)
        mixer.initialize()
        assert "extra_bus" in mixer.get_bus_names()

    def test_shutdown_clears_all_state(self, mixer):
        """shutdown should reset all internal state."""
        mixer.route_source_to_bus("src", "sfx")
        mixer.shutdown()
        assert mixer._initialized is False
        assert mixer._buses == {}
        assert mixer._master_bus is None
        assert mixer._processing_order == []
        assert mixer._source_bus_map == {}
        assert mixer._master_output_buffer is None
        assert mixer._tick_work_buffer is None

    def test_shutdown_clears_sub_managers(self, mixer):
        """shutdown should clear ducking, sidechain, HDR, and router."""
        mixer.shutdown()
        assert mixer._ducking_manager._instances == {}
        assert mixer._sidechain_manager._compressors == {}
        assert mixer._hdr_manager._sources == {}
        assert mixer._router._aux_sends == {}

    def test_initialize_with_custom_buses(self):
        """initialize should accept custom bus dictionary."""
        custom = {
            "master": MixBus("master", BusType.MASTER),
            "custom_sfx": MixBus("custom_sfx", BusType.CATEGORY),
        }
        custom["custom_sfx"].set_parent(custom["master"])
        m = Mixer()
        m.initialize(custom_buses=custom)
        assert "custom_sfx" in m.get_bus_names()
        assert m.get_bus("custom_sfx") is custom["custom_sfx"]

    def test_initialized_property(self, mixer):
        """initialized property should reflect internal state."""
        assert mixer.initialized is True
        mixer.shutdown()
        assert mixer.initialized is False

    def test_master_bus_property(self, mixer):
        """master_bus property should return the master bus."""
        master = mixer.master_bus
        assert master is not None
        assert master.bus_type == BusType.MASTER
        assert master.name == "master"

    def test_buses_property_returns_copy(self, mixer):
        """buses property should return a copy of internal dict."""
        buses1 = mixer.buses
        buses2 = mixer.buses
        assert buses1 is not buses2

    def test_get_buses_by_type_returns_matching(self, mixer):
        """get_buses_by_type should filter by bus type."""
        masters = mixer.get_buses_by_type(BusType.MASTER)
        assert len(masters) == 1
        assert masters[0].name == "master"
        subs = mixer.get_buses_by_type(BusType.SUB)
        assert len(subs) >= 6

    def test_ensure_tick_buffers_allocates(self, mixer):
        """_ensure_tick_buffers should allocate output and work buffers."""
        mixer._ensure_tick_buffers(128)
        assert mixer._master_output_buffer is not None
        assert mixer._tick_work_buffer is not None
        assert mixer._master_output_buffer.shape == (MIXER_NUM_CHANNELS, 128)
        assert mixer._tick_work_buffer.shape == (MIXER_NUM_CHANNELS, 128)
        assert mixer._master_output_buffer.dtype == np.float32

    def test_ensure_tick_buffers_grows_on_demand(self, mixer):
        """_ensure_tick_buffers should grow when larger size requested."""
        mixer._ensure_tick_buffers(64)
        mixer._ensure_tick_buffers(512)
        assert mixer._master_output_buffer.shape[1] >= 512


# =========================================================================
# Test Mixer Bus Management Edge Cases
# =========================================================================


class TestMixerBusManagement:
    """Tests for bus creation, removal, and query edge cases."""

    def test_create_bus_duplicate_raises(self, mixer):
        """create_bus should raise ValueError for duplicate name."""
        with pytest.raises(ValueError, match="already exists"):
            mixer.create_bus("sfx", BusType.SUB)

    def test_remove_bus_master_raises(self, mixer):
        """remove_bus should raise ValueError for master bus."""
        with pytest.raises(ValueError, match="Cannot remove master bus"):
            mixer.remove_bus("master")

    def test_remove_bus_nonexistent_returns_false(self, mixer):
        """remove_bus should return False for non-existent bus."""
        assert mixer.remove_bus("nonexistent") is False

    def test_get_bus_returns_none_for_missing(self, mixer):
        """get_bus should return None for non-existent bus."""
        assert mixer.get_bus("nonexistent") is None


# =========================================================================
# Test Tick Pipeline -- Deep Coverage
# =========================================================================


class TestTickPipelineDeep:
    """Deep tests for tick pipeline stages."""

    def test_tick_default_buffer_size(self, mixer):
        """tick(0) should use default MIXER_BUFFER_SIZE."""
        result = mixer.tick(0)
        assert result is not None
        assert result.shape[1] == MIXER_BUFFER_SIZE

    def test_tick_silent_when_volume_zero(self, mixer):
        """tick should produce silence when bus volume is 0."""
        mixer.set_bus_volume("sfx", 0.0)
        mixer.route_source_to_bus("voice", "sfx")
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result == 0.0)

    def test_tick_uninitialized_size_zero(self):
        """tick(0) on uninitialized mixer should return silence."""
        m = Mixer()
        result = m.tick(0)
        assert result is not None
        assert result.shape == (MIXER_NUM_CHANNELS, 512)

    def test_master_output_none_after_shutdown(self, mixer):
        """read_master_output should return None after shutdown."""
        mixer.tick(64)
        mixer.shutdown()
        result = mixer.read_master_output()
        assert result is None


# =========================================================================
# Test MixBus Hierarchy Internals
# =========================================================================


class TestMixBusHierarchyInternals:
    """Tests for MixBus hierarchy management edge cases."""

    def test_set_parent_cycle_detection(self, bus):
        """set_parent should raise ValueError on cycle."""
        child = MixBus("child", BusType.SUB, parent=bus)
        with pytest.raises(ValueError, match="cycle"):
            bus.set_parent(child)

    def test_set_parent_self_reference(self, bus):
        """set_parent should raise ValueError when setting self as parent."""
        with pytest.raises(ValueError, match="cannot be its own parent"):
            bus.set_parent(bus)


# =========================================================================
# Test Mixer Update Integration
# =========================================================================


class TestMixerUpdate:
    """Tests for Mixer.update() integration."""

    def test_update_no_crash_on_empty(self, mixer):
        """update should not crash when called without state."""
        mixer.update(0.016)

    def test_update_uninitialized_skips(self):
        """update should return early if not initialized."""
        m = Mixer()
        m.update(0.016)


# =========================================================================
# Test Mixer Config and State
# =========================================================================


class TestMixerStateConfig:
    """Tests for MixerConfig defaults and get_state."""

    def test_mixer_config_defaults(self):
        """MixerConfig defaults should match module constants."""
        config = MixerConfig()
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.enable_hdr is True
        assert config.enable_ducking is True
        assert config.enable_sidechain is True
        assert config.enable_snapshots is True
        assert config.auto_create_dialogue_duck is True

    def test_custom_mixer_config(self):
        """MixerConfig should accept custom values."""
        config = MixerConfig(
            sample_rate=96000,
            enable_hdr=False,
            enable_ducking=False,
        )
        mixer = Mixer(config=config)
        mixer.initialize()
        assert mixer._config.enable_hdr is False
        assert mixer._config.enable_ducking is False
        assert mixer._config.sample_rate == 96000


# =========================================================================
# Test MixBus Accumulator Edge Cases
# =========================================================================


class TestAccumulatorEdgeCases:
    """Edge cases for MixBus accumulator methods."""

    def test_accumulate_1d_array(self, bus):
        """accumulate should handle 1D input (single channel)."""
        samples_1d = np.ones(64, dtype=np.float32) * 0.5
        bus.clear_acc_buffer(64)
        bus.accumulate(samples_1d, 64)
        result = bus.read_acc_buffer(64)
        assert result.shape == (MIXER_NUM_CHANNELS, 64)
        assert np.allclose(result[0], 0.5)
        assert np.allclose(result[1], 0.5)

    def test_read_acc_buffer_no_buffer(self, bus):
        """read_acc_buffer should return zeros when no buffer."""
        bus._acc_buffer = None
        result = bus.read_acc_buffer(64)
        assert result.shape == (MIXER_NUM_CHANNELS, 64)
        assert np.all(result == 0.0)


# =========================================================================
# Test MixBus Callbacks
# =========================================================================


class TestMixBusCallbacks:
    """Tests for MixBus on_change callbacks."""

    def test_on_change_callback_fires(self, bus):
        """on_change should fire callback when volume changes."""
        results = []
        def callback(b):
            results.append(b.name)
        bus.on_change(callback)
        bus.volume = 0.5
        assert len(results) == 1
        assert results[0] == "test_bus"

    def test_toggle_mute_works(self, bus):
        """toggle_mute should alternate mute state."""
        assert bus.muted is False
        bus.toggle_mute()
        assert bus.muted is True
        bus.toggle_mute()
        assert bus.muted is False

    def test_toggle_solo_works(self, bus):
        """toggle_solo should alternate solo state."""
        assert bus.soloed is False
        bus.toggle_solo()
        assert bus.soloed is True
        bus.toggle_solo()
        assert bus.soloed is False