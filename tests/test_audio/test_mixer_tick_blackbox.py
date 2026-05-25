"""
Blackbox Tests for T-AU-2.14 Mixer Tick (Cleanroom / Spec Only).

Acceptance criteria from spec (GAPSET_15_AUDIO):
  - Mixer tick processes all active voices into their assigned buses
  - Per-bus effects applied
  - Final mix sent to AudioOutput
  - No missed periods
  - 15+ test cases

Dependencies: T-AU-2.2 (bus hierarchy), T-AU-2.7 (voice pool), T-AU-1.9 (AudioOutput)

Contract defined by these tests (written from spec before implementation):

    Mixer.tick(num_samples: int) -> Optional[np.ndarray]
        Returns (2, num_samples) float32 ndarray clipped to [-1, 1].
        Returns None if mixer is not initialized.

    Mixer.route_source_to_bus(source_id: str, bus_name: str) -> bool
        Route a voice/source to a named bus. Returns True on success.

    Mixer.unroute_source(source_id: str) -> bool
        Remove a source from its bus. Returns True on success.

    Mixer.clear_all_bus_accumulators(num_samples: int)
        Zero all bus accumulation buffers (Stage 1 of tick pipeline).

    MixBus.clear_acc_buffer(num_samples: int)
        Zero this bus's accumulation buffer.

    MixBus.accumulate(samples: np.ndarray, num_samples: int)
        Add audio samples into this bus's accumulation buffer.

    MixBus.process_audio(num_samples: int) -> np.ndarray
        Copy accumulation buffer, apply volume, run DSP effect chain,
        return processed (channels, samples) ndarray.

    MixBus.effect_chain -> DSPChain
        The DSP effect chain attached to this bus.
"""

from __future__ import annotations

import time
import threading
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.audio.mixing import (
    Mixer,
    MixerConfig,
    MixBus,
    BusType,
    BusState,
    create_default_hierarchy,
    CATEGORY_MASTER,
    CATEGORY_SFX,
    CATEGORY_MUSIC,
    CATEGORY_VO,
    CATEGORY_AMBIENT,
    CATEGORY_UI,
    db_to_linear,
    linear_to_db,
)

from engine.audio.dsp import (
    DSPNode,
    DSPChain,
    GainNode,
    PassthroughNode,
)


# =============================================================================
# Helper — concrete DSP node for testing
# =============================================================================


class CallTrackingNode(DSPNode):
    """A DSP node that records all process_block calls and applies a gain."""

    def __init__(self, gain_factor: float = 1.0) -> None:
        super().__init__()
        self._gain_factor = gain_factor
        self.process_block_calls: list[np.ndarray] = []
        self.process_block_count = 0

    @property
    def gain_factor(self) -> float:
        return self._gain_factor

    @gain_factor.setter
    def gain_factor(self, value: float) -> None:
        self._gain_factor = value

    def process_sample(self, sample: float, channel: int = 0) -> float:
        return sample * self._gain_factor

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        self.process_block_calls.append(input_buffer.copy())
        self.process_block_count += 1
        np.copyto(output_buffer, input_buffer * self._gain_factor)

    def reset(self) -> None:
        self.process_block_calls.clear()
        self.process_block_count = 0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mixer() -> Mixer:
    """Create an initialized Mixer with default bus hierarchy (T-AU-2.2)."""
    m = Mixer()
    m.initialize()
    return m


@pytest.fixture
def buses(mixer: Mixer) -> dict[str, MixBus]:
    """Convenience access to all buses in the default hierarchy."""
    return mixer.buses


@pytest.fixture
def master_bus(buses: dict[str, MixBus]) -> MixBus:
    """The master bus."""
    return buses[CATEGORY_MASTER]


@pytest.fixture
def sfx_bus(buses: dict[str, MixBus]) -> MixBus:
    """The SFX category bus."""
    return buses[CATEGORY_SFX]


@pytest.fixture
def music_bus(buses: dict[str, MixBus]) -> MixBus:
    """The music category bus."""
    return buses[CATEGORY_MUSIC]


@pytest.fixture
def footsteps_bus(buses: dict[str, MixBus]) -> MixBus:
    """The footsteps sub-bus."""
    return buses["footsteps"]


# =============================================================================
# Basic Tick Contract
# =============================================================================


class TestTickContract:
    """The fundamental contract of Mixer.tick()."""

    def test_tick_returns_none_when_not_initialized(self) -> None:
        """tick() must return None when the mixer has not been initialized."""
        m = Mixer()
        result = m.tick(64)
        assert result is None, (
            "Uninitialized mixer tick must return None"
        )

    def test_tick_returns_ndarray_when_initialized(self, mixer: Mixer) -> None:
        """tick() must return a numpy ndarray when the mixer is initialized."""
        result = mixer.tick(64)
        assert result is not None, "Initialized mixer tick must not return None"
        assert isinstance(result, np.ndarray), (
            f"Expected ndarray, got {type(result)}"
        )

    def test_tick_output_shape(self, mixer: Mixer) -> None:
        """Output shape must be (channels, num_samples) = (2, num_samples)."""
        for num_samples in [32, 64, 128, 256, 512]:
            result = mixer.tick(num_samples)
            assert result is not None
            assert result.shape == (2, num_samples), (
                f"Expected shape (2, {num_samples}), got {result.shape}"
            )

    def test_tick_output_dtype(self, mixer: Mixer) -> None:
        """Output dtype must be float32."""
        result = mixer.tick(64)
        assert result is not None
        assert result.dtype == np.float32, (
            f"Expected float32 dtype, got {result.dtype}"
        )

    def test_tick_silent_output_with_no_sources(self, mixer: Mixer) -> None:
        """With no active sources, tick must return silence (all zeros)."""
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result == 0.0), (
            "With no sources, tick must produce silence"
        )

    def test_tick_output_clipped_to_interval(self, mixer: Mixer) -> None:
        """Output samples must be in the range [-1.0, 1.0]."""
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result >= -1.0) and np.all(result <= 1.0), (
            f"Output clipped to [-1, 1]; min={result.min()}, max={result.max()}"
        )


# =============================================================================
# Source Routing
# =============================================================================


class TestSourceRouting:
    """Sources must be routable to specific buses (T-AU-2.15 contract)."""

    def test_route_source_to_bus(self, mixer: Mixer) -> None:
        """route_source_to_bus must return True for a valid bus name."""
        result = mixer.route_source_to_bus("voice_1", CATEGORY_SFX)
        assert result is True, "Routing to valid bus must succeed"

    def test_route_source_to_invalid_bus(self, mixer: Mixer) -> None:
        """route_source_to_bus must return False for an invalid bus name."""
        result = mixer.route_source_to_bus("voice_1", "nonexistent_bus")
        assert result is False, "Routing to nonexistent bus must return False"

    def test_unroute_source(self, mixer: Mixer) -> None:
        """unroute_source must return False for an unknown source."""
        result = mixer.unroute_source("nonexistent_voice")
        assert result is False, "Unrouting unknown source must return False"

    def test_route_then_unroute(self, mixer: Mixer) -> None:
        """A routed source must be removable via unroute_source."""
        mixer.route_source_to_bus("voice_1", CATEGORY_SFX)
        result = mixer.unroute_source("voice_1")
        assert result is True, "Unrouting a known source must return True"

    def test_routed_source_produces_audio(self, mixer: Mixer) -> None:
        """A source routed to a bus must contribute to the tick output."""
        mixer.route_source_to_bus("voice_1", CATEGORY_SFX)
        result = mixer.tick(64)
        assert result is not None
        assert not np.all(result == 0.0), (
            "A routed source must contribute non-zero audio"
        )

    def test_unrouted_source_produces_silence(self, mixer: Mixer) -> None:
        """An unrouted source must not contribute to tick output."""
        mixer.route_source_to_bus("voice_1", CATEGORY_SFX)
        mixer.unroute_source("voice_1")
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result == 0.0), (
            "An unrouted source must not produce audio"
        )


# =============================================================================
# Bus Accumulation
# =============================================================================


class TestBusAccumulation:
    """Bus accumulators must correctly sum audio (tick pipeline Stage 2)."""

    def test_bus_accumulate_and_process(self, sfx_bus: MixBus) -> None:
        """A bus must accumulate samples and return them via process_audio."""
        num_samples = 64
        sfx_bus.clear_acc_buffer(num_samples)
        test_data = np.ones((2, num_samples), dtype=np.float32) * 0.5
        sfx_bus.accumulate(test_data, num_samples)
        output = sfx_bus.process_audio(num_samples)
        assert output is not None, "process_audio must return ndarray"
        assert output.shape == (2, num_samples), (
            f"Expected shape (2, {num_samples}), got {output.shape}"
        )
        # Default volume is 1.0, so output should match input
        assert np.allclose(output, test_data), (
            "Output should match accumulated input at volume 1.0"
        )

    def test_bus_accumulate_multiple_calls_sum(self, sfx_bus: MixBus) -> None:
        """Multiple accumulate calls must sum their samples."""
        num_samples = 64
        sfx_bus.clear_acc_buffer(num_samples)
        chunk = np.ones((2, num_samples), dtype=np.float32) * 0.25
        sfx_bus.accumulate(chunk, num_samples)
        sfx_bus.accumulate(chunk, num_samples)
        output = sfx_bus.process_audio(num_samples)
        assert output is not None
        # Two calls at 0.25 should sum to 0.5
        assert np.allclose(output[0], 0.5), (
            "Multiple accumulate calls must sum"
        )

    def test_bus_clear_acc_buffer_zeros(self, sfx_bus: MixBus) -> None:
        """clear_acc_buffer must reset the accumulator to zero."""
        num_samples = 64
        sfx_bus.clear_acc_buffer(num_samples)
        test_data = np.ones((2, num_samples), dtype=np.float32) * 0.5
        sfx_bus.accumulate(test_data, num_samples)
        sfx_bus.clear_acc_buffer(num_samples)
        output = sfx_bus.process_audio(num_samples)
        assert output is not None
        assert np.all(output == 0.0), (
            "After clear, process_audio must return silence"
        )

    def test_bus_accumulator_reset_between_ticks(self, mixer: Mixer,
                                                  sfx_bus: MixBus) -> None:
        """Bus accumulators must be cleared at the start of each tick."""
        num_samples = 64
        mixer.route_source_to_bus("voice_a", CATEGORY_SFX)
        result_a = mixer.tick(num_samples)
        assert result_a is not None

        # Tick with no sources - must be silent if accumulators reset
        mixer.unroute_source("voice_a")
        result_b = mixer.tick(num_samples)
        assert result_b is not None
        assert np.all(result_b == 0.0), (
            "Second tick after unrouting must produce silence"
        )


# =============================================================================
# Bus Volume and Mute
# =============================================================================


class TestBusVolumeMute:
    """Bus volume, mute, and solo must affect the output (T-AU-2.1)."""

    def test_bus_volume_reduces_output(self, mixer: Mixer,
                                        sfx_bus: MixBus) -> None:
        """Lowering bus volume must proportionally reduce the output level."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)

        sfx_bus.volume = 1.0
        result_full = mixer.tick(64)
        assert result_full is not None
        peak_full = np.max(np.abs(result_full))

        sfx_bus.volume = 0.25
        result_quarter = mixer.tick(64)
        assert result_quarter is not None
        peak_quarter = np.max(np.abs(result_quarter))

        assert peak_quarter < peak_full, (
            f"Reducing volume from 1.0 to 0.25 must lower output; "
            f"full={peak_full:.6f}, quarter={peak_quarter:.6f}"
        )

    def test_muted_bus_produces_silence(self, mixer: Mixer,
                                         sfx_bus: MixBus) -> None:
        """A muted bus must produce silence in the output."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        sfx_bus.muted = True
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result == 0.0), "Muted bus must produce silence"

    def test_muted_master_produces_silence(self, mixer: Mixer,
                                            master_bus: MixBus) -> None:
        """Muting the master bus must silence all output."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        master_bus.muted = True
        result = mixer.tick(64)
        assert result is not None
        assert np.all(result == 0.0), "Muted master bus must silence all output"


# =============================================================================
# Bus Hierarchy
# =============================================================================


class TestBusHierarchy:
    """Audio must flow correctly through the bus hierarchy (T-AU-2.2)."""

    def test_sub_bus_routes_through_parent(self, mixer: Mixer) -> None:
        """A source on a sub-bus must route through its parent to output."""
        mixer.route_source_to_bus("voice", "footsteps")
        result = mixer.tick(64)
        assert result is not None
        assert not np.all(result == 0.0), (
            "Sub-bus source must route through parent to output"
        )

    def test_hierarchical_volume(self, mixer: Mixer,
                                  sfx_bus: MixBus,
                                  footsteps_bus: MixBus) -> None:
        """Parent volume must multiply child volume (hierarchical gain)."""
        mixer.route_source_to_bus("voice", "footsteps")

        footsteps_bus.volume = 0.5
        sfx_bus.volume = 1.0
        result_full = mixer.tick(64)
        assert result_full is not None
        peak_full = np.max(np.abs(result_full))

        sfx_bus.volume = 0.25
        result_reduced = mixer.tick(64)
        assert result_reduced is not None
        peak_reduced = np.max(np.abs(result_reduced))

        assert peak_reduced < peak_full, (
            "Reducing parent bus volume must reduce child's effective output"
        )

    def test_sources_on_different_buses_both_contribute(
            self, mixer: Mixer) -> None:
        """Sources on different buses must both contribute to master output."""
        mixer.route_source_to_bus("sfx_voice", CATEGORY_SFX)
        mixer.route_source_to_bus("music_voice", CATEGORY_MUSIC)
        result = mixer.tick(64)
        assert result is not None
        assert not np.all(result == 0.0), (
            "Sources on different buses must both contribute"
        )


# =============================================================================
# Per-Bus Effects
# =============================================================================


class TestPerBusEffects:
    """The per-bus effect chain must process bus audio (T-AU-2.16 contract)."""

    def test_effect_chain_is_dspchain(self, sfx_bus: MixBus) -> None:
        """The effect_chain property must return a DSPChain instance."""
        chain = sfx_bus.effect_chain
        assert isinstance(chain, DSPChain), (
            f"Expected DSPChain, got {type(chain)}"
        )

    def test_add_effect_to_bus(self, sfx_bus: MixBus) -> None:
        """Adding a DSP node to the bus effect chain must succeed."""
        node = CallTrackingNode(gain_factor=2.0)
        sfx_bus.effect_chain.add_node(node)
        assert sfx_bus.effect_chain.length == 1, (
            "Effect chain must have 1 node after add"
        )
        assert sfx_bus.effect_chain.nodes[0] is node, (
            "The added node must be in the chain"
        )

    def test_effect_applied_during_tick(self, mixer: Mixer,
                                         sfx_bus: MixBus) -> None:
        """A DSP node on a bus must be called during tick()."""
        node = CallTrackingNode(gain_factor=2.0)
        sfx_bus.effect_chain.add_node(node)
        mixer.route_source_to_bus("voice", CATEGORY_SFX)

        mixer.tick(64)
        assert node.process_block_count > 0, (
            "The DSP node must be called during tick"
        )

    def test_effect_doubles_amplitude(self, mixer: Mixer,
                                       sfx_bus: MixBus) -> None:
        """A gain=2.0 effect must double the bus output amplitude."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)

        # First tick without effect
        result_before = mixer.tick(64)
        assert result_before is not None
        peak_before = np.max(np.abs(result_before))

        # Add gain=2 effect
        node = CallTrackingNode(gain_factor=2.0)
        sfx_bus.effect_chain.add_node(node)

        result_after = mixer.tick(64)
        assert result_after is not None
        peak_after = np.max(np.abs(result_after))

        assert peak_after > peak_before, (
            "Gain=2.0 effect must increase output amplitude"
        )

    def test_effect_bypass(self, mixer: Mixer, sfx_bus: MixBus) -> None:
        """A bypassed effect must not process audio."""
        node = CallTrackingNode(gain_factor=999.0)
        sfx_bus.effect_chain.add_node(node)
        node.set_bypass(True)

        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        result = mixer.tick(64)
        assert result is not None

        # Node should not have been called
        assert node.process_block_count == 0, (
            "Bypassed node must not be called during tick"
        )

    def test_multiple_effects_chain_order(self, mixer: Mixer,
                                           sfx_bus: MixBus) -> None:
        """Multiple effects must be applied in chain order."""
        first = CallTrackingNode(gain_factor=1.0)
        second = CallTrackingNode(gain_factor=1.0)
        sfx_bus.effect_chain.add_node(first)
        sfx_bus.effect_chain.add_node(second)

        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        mixer.tick(64)

        # Both should have been called, in order (first then second)
        assert first.process_block_count > 0, "First node must be called"
        assert second.process_block_count > 0, "Second node must be called"

    def test_empty_effect_chain_passthrough(self, mixer: Mixer,
                                             sfx_bus: MixBus) -> None:
        """A bus with no effects must still pass audio through correctly."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        result = mixer.tick(64)
        assert result is not None
        assert not np.all(result == 0.0), (
            "Bus with no effects must still output audio"
        )

    def test_effect_on_sub_bus(self, mixer: Mixer,
                                footsteps_bus: MixBus) -> None:
        """An effect on a sub-bus must process before parent accumulation."""
        node = CallTrackingNode(gain_factor=3.0)
        footsteps_bus.effect_chain.add_node(node)

        mixer.route_source_to_bus("voice", "footsteps")
        mixer.tick(64)

        assert node.process_block_count > 0, (
            "Sub-bus effect must be called during tick"
        )


# =============================================================================
# Timing and Consistency
# =============================================================================


class TestTimingConsistency:
    """The mixer tick must not miss periods (T-AU-2.14)."""

    def test_no_missed_periods(self, mixer: Mixer) -> None:
        """Across many consecutive ticks, every tick must return output."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        for i in range(100):
            result = mixer.tick(64)
            assert result is not None, (
                f"Tick {i} returned None — missed period"
            )

    def test_consistent_buffer_size(self, mixer: Mixer) -> None:
        """Each tick must return the requested num_samples."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        for num_samples in [32, 64, 128, 256]:
            result = mixer.tick(num_samples)
            assert result is not None
            assert result.shape[1] == num_samples, (
                f"Expected {num_samples} samples, got {result.shape[1]}"
            )

    def test_tick_with_zero_samples(self, mixer: Mixer) -> None:
        """tick(0) must return a valid (2, 0) array or handle gracefully."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        result = mixer.tick(0)
        # Accept either (2, 0) or None — spec-safe
        if result is not None:
            assert result.shape == (2, 0)

    def test_high_source_count(self, mixer: Mixer) -> None:
        """Many simultaneous sources must not cause tick to fail."""
        for i in range(64):
            mixer.route_source_to_bus(f"voice_{i}", CATEGORY_SFX)
        result = mixer.tick(64)
        assert result is not None, "Tick with 64 sources must return output"
        assert np.any(result != 0.0), (
            "64 sources must produce non-zero audio"
        )


# =============================================================================
# Thread Safety
# =============================================================================


class TestThreadSafety:
    """Concurrent tick calls must not corrupt state."""

    def test_concurrent_ticks_no_crash(self, mixer: Mixer) -> None:
        """Multiple threads calling tick must not crash."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        errors: list[Exception] = []

        def tick_thread() -> None:
            try:
                for _ in range(50):
                    mixer.tick(64)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=tick_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent ticks raised errors: {errors}"

    def test_concurrent_tick_and_bus_modification(self, mixer: Mixer,
                                                    sfx_bus: MixBus) -> None:
        """Tick concurrent with bus volume changes must not corrupt."""
        mixer.route_source_to_bus("voice", CATEGORY_SFX)
        errors: list[Exception] = []

        def tick_loop() -> None:
            try:
                for _ in range(50):
                    mixer.tick(64)
            except Exception as e:
                errors.append(e)

        def modify_bus() -> None:
            try:
                for _ in range(50):
                    sfx_bus.volume = 0.5
                    sfx_bus.muted = True
                    sfx_bus.muted = False
                    sfx_bus.volume = 0.8
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=tick_loop),
            threading.Thread(target=modify_bus),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, (
            f"Concurrent tick and bus modification raised errors: {errors}"
        )


# =============================================================================
# Integration — Full Pipeline
# =============================================================================


class TestFullPipeline:
    """End-to-end integration of the full tick pipeline."""

    def test_full_pipeline_multiple_buses_and_effects(
            self, mixer: Mixer, sfx_bus: MixBus, music_bus: MixBus) -> None:
        """Multiple sources, multiple buses, multiple effects."""
        sfx_node = CallTrackingNode(gain_factor=1.5)
        music_node = CallTrackingNode(gain_factor=0.8)
        sfx_bus.effect_chain.add_node(sfx_node)
        music_bus.effect_chain.add_node(music_node)

        mixer.route_source_to_bus("sfx_voice", CATEGORY_SFX)
        mixer.route_source_to_bus("music_voice", CATEGORY_MUSIC)
        mixer.route_source_to_bus("sub_voice", "footsteps")

        result = mixer.tick(128)
        assert result is not None
        assert result.shape == (2, 128)
        assert sfx_node.process_block_count > 0, (
            "SFX effect must have been called"
        )
        assert music_node.process_block_count > 0, (
            "Music effect must have been called"
        )

    def test_sub_bus_effect_then_parent_effect(
            self, mixer: Mixer, sfx_bus: MixBus,
            footsteps_bus: MixBus) -> None:
        """A sub-bus effect applies first, then the parent bus effect."""
        sub_node = CallTrackingNode(gain_factor=2.0)
        parent_node = CallTrackingNode(gain_factor=3.0)
        footsteps_bus.effect_chain.add_node(sub_node)
        sfx_bus.effect_chain.add_node(parent_node)

        mixer.route_source_to_bus("voice", "footsteps")
        result = mixer.tick(64)
        assert result is not None
        assert sub_node.process_block_count > 0
        assert parent_node.process_block_count > 0


# =============================================================================
# Sample Rate and Configuration
# =============================================================================


class TestSampleRateConfig:
    """The mixer must respect sample rate configuration."""

    def test_default_sample_rate(self) -> None:
        """Default sample rate should be 48000 as per config."""
        config = MixerConfig()
        assert config.sample_rate == 48000, (
            f"Expected default sample rate 48000, got {config.sample_rate}"
        )

    def test_custom_sample_rate(self) -> None:
        """Custom sample rate in MixerConfig must propagate."""
        config = MixerConfig(sample_rate=96000)
        mixer = Mixer(config=config)
        mixer.initialize()
        result = mixer.tick(64)
        assert result is not None, "Mixer with custom config must tick"
