"""
Whitebox tests for MixBus (T-FG-7.5 / engine_audio_mixing_spatial P1 T1.1).

Tests cover every branch, error path, edge-case input, and internal invariant
in MixBus. The three methods added by DEV (set_volume_db, set_volume_linear,
write_output) are tested alongside all pre-existing functionality.

WHITEBOX coverage plan:
  DEV-ADDED:
    - set_volume_db: normal, clamped-below-min, clamped-above-max,
                     boundary-at-min, boundary-at-max, callback fires
    - set_volume_linear: normal, zero, clamped-above-max, callback fires
    - write_output: stereo, mono-to-stereo-broadcast, 1D-reshape,
                    buffer-reallocation, zero-samples

  CONSTRUCTOR:
    - master (no parent)
    - child (with parent)
    - volume-clamp-above-max, pitch-clamp-above-max, pitch-clamp-below-min

  VOLUME:
    - volume getter, setter normal, setter clamp-low, setter clamp-high
    - volume_db getter normal, volume_db getter near-silence
    - volume_db setter normal, setter clamp-low, setter clamp-high
    - get_effective_volume: unmuted, muted, parent-muted, multi-level-hierarchy

  PITCH:
    - pitch getter/setter normal, setter clamp-low, setter clamp-high
    - get_effective_pitch: no-parent, with-parents, final-clamp

  MUTE / SOLO:
    - mute getter/setter, toggle_mute toggles-and-returns
    - solo getter/setter, toggle_solo toggles-and-returns

  FILTERS:
    - filters returns independent copy
    - set_low_pass: normal, freq-clamp-low, freq-clamp-high, q-clamp
    - set_high_pass: normal, freq-clamp-low, freq-clamp-high, q-clamp
    - reset_filters restores defaults

  DSP CHAIN:
    - effect_chain returns internal object
    - has_effects: empty, with-node

  AUDIO PROCESSING:
    - clear_acc_buffer: first-call-allocates, subsequent-reuses
    - accumulate: stereo, mono-broadcast, 1D-reshape, channel-truncation
    - read_acc_buffer: buffer-exists, buffer-none
    - process_audio: muted-returns-silence, volume-applied,
      lowpass-enabled-lazy-init, lowpass-enabled-reuse,
      highpass-enabled-lazy-init, both-filters,
      dsp-chain-applied, dsp-chain-error-handled, hard-clip

  HIERARCHY:
    - set_parent: normal, remove, self-error, cycle-error, re-parent
    - add_child, remove_child found, remove_child not-found
    - get_ancestors: none, one, many
    - get_descendants: none, direct, nested

  STATE MANAGEMENT:
    - get_state returns independent copy
    - set_state replaces, reset restores defaults

  CALLBACKS:
    - on_change registers, remove_callback found/not-found
    - _notify_change: callback-invoked, exception-silenced

  DATA CLASSES:
    - BusState.copy, FilterState.reset, FilterState.copy

  MODULE-LEVEL:
    - create_default_hierarchy: master-exists, category-children,
      sub-buses, parent-pointers
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.audio.mixing.mix_bus import (
    MixBus,
    BusType,
    BusState,
    FilterState,
    create_default_hierarchy,
)
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
    MIXER_NUM_CHANNELS,
    db_to_linear,
    linear_to_db,
    clamp,
)


# =============================================================================
# Helpers
# =============================================================================

_MAX_LINEAR = db_to_linear(MAX_VOLUME_DB)  # ~3.981


def _make_sine_buffer(
    num_samples: int = 128,
    num_channels: int = MIXER_NUM_CHANNELS,
    amplitude: float = 0.5,
    freq_hz: float = 440.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Create a simple sine-wave buffer for audio processing tests."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * math.pi * freq_hz * t)
    buf = np.zeros((num_channels, num_samples), dtype=np.float32)
    for ch in range(min(num_channels, MIXER_NUM_CHANNELS)):
        buf[ch, :] = wave
    return buf


# =============================================================================
# Constructor
# =============================================================================


class TestConstructor:
    """MixBus.__init__ paths."""

    def test_master_no_parent(self) -> None:
        """Master bus created without a parent."""
        bus = MixBus("master", BusType.MASTER)
        assert bus._parent is None
        assert bus._bus_type == BusType.MASTER
        assert bus._name == "master"
        assert bus.parent is None
        assert bus.children == []

    def test_child_with_parent(self) -> None:
        """Child bus attached to a parent at construction."""
        parent = MixBus("parent", BusType.CATEGORY)
        child = MixBus("child", BusType.SUB, parent=parent)
        assert child.parent is parent
        assert child in parent.children

    def test_volume_clamped_above_max_at_init(self) -> None:
        """Volume exceeding db_to_linear(MAX_VOLUME_DB) is clamped."""
        excessive = _MAX_LINEAR * 10.0
        bus = MixBus("test", volume=excessive)
        assert bus.volume == pytest.approx(_MAX_LINEAR)

    def test_pitch_clamped_above_max_at_init(self) -> None:
        """Pitch above MAX_PITCH is clamped."""
        bus = MixBus("test", pitch=MAX_PITCH * 2)
        assert bus.pitch == pytest.approx(MAX_PITCH)

    def test_pitch_clamped_below_min_at_init(self) -> None:
        """Pitch below MIN_PITCH is clamped."""
        bus = MixBus("test", pitch=0.0)
        assert bus.pitch == pytest.approx(MIN_PITCH)


# =============================================================================
# Volume
# =============================================================================


class TestVolume:
    """Volume property, setter, volume_db, and get_effective_volume."""

    def test_volume_getter(self) -> None:
        bus = MixBus("test")
        assert bus.volume == pytest.approx(DEFAULT_BUS_VOLUME)

    def test_volume_setter_normal(self) -> None:
        bus = MixBus("test")
        bus.volume = 0.5
        assert bus.volume == pytest.approx(0.5)

    def test_volume_setter_clamp_below_zero(self) -> None:
        bus = MixBus("test")
        bus.volume = -1.0
        assert bus.volume == pytest.approx(0.0)

    def test_volume_setter_clamp_above_max(self) -> None:
        bus = MixBus("test")
        bus.volume = _MAX_LINEAR * 10.0
        assert bus.volume == pytest.approx(_MAX_LINEAR)

    def test_volume_db_getter_normal(self) -> None:
        bus = MixBus("test")
        bus.volume = 1.0
        assert bus.volume_db == pytest.approx(0.0)

    def test_volume_db_getter_near_silence(self) -> None:
        """When linear volume is below threshold, volume_db returns MIN_VOLUME_DB."""
        bus = MixBus("test")
        bus.volume = 0.0
        assert bus.volume_db == pytest.approx(MIN_VOLUME_DB)

    def test_volume_db_setter_normal(self) -> None:
        bus = MixBus("test")
        bus.volume_db = -6.0
        expected_lin = db_to_linear(-6.0)
        assert bus.volume == pytest.approx(expected_lin)

    def test_volume_db_setter_clamp_below_min(self) -> None:
        bus = MixBus("test")
        bus.volume_db = MIN_VOLUME_DB - 20.0
        assert bus.volume_db == pytest.approx(MIN_VOLUME_DB)

    def test_volume_db_setter_clamp_above_max(self) -> None:
        bus = MixBus("test")
        bus.volume_db = MAX_VOLUME_DB + 10.0
        assert bus.volume_db == pytest.approx(MAX_VOLUME_DB)

    # --- DEV-added: set_volume_db ---

    def test_set_volume_db_normal(self) -> None:
        """set_volume_db delegates to volume_db setter with correct clamping and
        conversion."""
        bus = MixBus("test")
        bus.set_volume_db(-6.0)
        assert bus.volume_db == pytest.approx(-6.0)
        assert bus.volume == pytest.approx(db_to_linear(-6.0))

    def test_set_volume_db_clamp_below_min(self) -> None:
        bus = MixBus("test")
        bus.set_volume_db(MIN_VOLUME_DB - 999.0)
        assert bus.volume_db == pytest.approx(MIN_VOLUME_DB)
        assert bus.volume == pytest.approx(0.0)

    def test_set_volume_db_clamp_above_max(self) -> None:
        bus = MixBus("test")
        bus.set_volume_db(MAX_VOLUME_DB + 999.0)
        assert bus.volume_db == pytest.approx(MAX_VOLUME_DB)
        assert bus.volume == pytest.approx(_MAX_LINEAR)

    def test_set_volume_db_boundary_min(self) -> None:
        """Boundary: MIN_VOLUME_DB exactly."""
        bus = MixBus("test")
        bus.set_volume_db(MIN_VOLUME_DB)
        assert bus.volume_db == pytest.approx(MIN_VOLUME_DB)
        assert bus.volume == pytest.approx(0.0)

    def test_set_volume_db_boundary_max(self) -> None:
        """Boundary: MAX_VOLUME_DB exactly."""
        bus = MixBus("test")
        bus.set_volume_db(MAX_VOLUME_DB)
        assert bus.volume_db == pytest.approx(MAX_VOLUME_DB)
        assert bus.volume == pytest.approx(_MAX_LINEAR)

    def test_set_volume_db_fires_change_callback(self) -> None:
        """set_volume_db triggers a change notification."""
        bus = MixBus("test")
        fired = []
        bus.on_change(lambda b: fired.append(True))
        bus.set_volume_db(-3.0)
        assert len(fired) == 1

    # --- DEV-added: set_volume_linear ---

    def test_set_volume_linear_normal(self) -> None:
        """set_volume_linear delegates to volume setter."""
        bus = MixBus("test")
        bus.set_volume_linear(0.75)
        assert bus.volume == pytest.approx(0.75)

    def test_set_volume_linear_zero(self) -> None:
        """set_volume_linear with 0.0 sets volume to 0."""
        bus = MixBus("test")
        bus.set_volume_linear(0.0)
        assert bus.volume == pytest.approx(0.0)

    def test_set_volume_linear_clamp_above_max(self) -> None:
        """set_volume_linear with excessive value is clamped."""
        bus = MixBus("test")
        bus.set_volume_linear(_MAX_LINEAR * 10.0)
        assert bus.volume == pytest.approx(_MAX_LINEAR)

    def test_set_volume_linear_fires_change_callback(self) -> None:
        """set_volume_linear triggers a change notification."""
        bus = MixBus("test")
        fired = []
        bus.on_change(lambda b: fired.append(True))
        bus.set_volume_linear(0.5)
        assert len(fired) == 1

    # --- get_effective_volume ---

    def test_get_effective_volume_unmuted_no_parent(self) -> None:
        bus = MixBus("test", volume=0.8)
        assert bus.get_effective_volume() == pytest.approx(0.8)

    def test_get_effective_volume_muted(self) -> None:
        """Muted bus returns 0.0 regardless of volume."""
        bus = MixBus("test", volume=1.0)
        bus.muted = True
        assert bus.get_effective_volume() == pytest.approx(0.0)

    def test_get_effective_volume_parent_muted(self) -> None:
        """When parent is muted, effective volume is 0.0."""
        parent = MixBus("parent", volume=1.0)
        child = MixBus("child", volume=1.0, parent=parent)
        parent.muted = True
        assert child.get_effective_volume() == pytest.approx(0.0)

    def test_get_effective_volume_multi_level_hierarchy(self) -> None:
        """Effective volume multiplies through all unmuted ancestors."""
        grandparent = MixBus("gp", volume=0.8)
        parent = MixBus("p", volume=0.5, parent=grandparent)
        child = MixBus("c", volume=0.5, parent=parent)
        expected = 0.8 * 0.5 * 0.5
        assert child.get_effective_volume() == pytest.approx(expected)


# =============================================================================
# Pitch
# =============================================================================


class TestPitch:
    """Pitch property, setter, and get_effective_pitch."""

    def test_pitch_getter_default(self) -> None:
        bus = MixBus("test")
        assert bus.pitch == pytest.approx(DEFAULT_PITCH)

    def test_pitch_setter_normal(self) -> None:
        bus = MixBus("test")
        bus.pitch = 2.0
        assert bus.pitch == pytest.approx(2.0)

    def test_pitch_setter_clamp_below_min(self) -> None:
        bus = MixBus("test")
        bus.pitch = 0.0
        assert bus.pitch == pytest.approx(MIN_PITCH)

    def test_pitch_setter_clamp_above_max(self) -> None:
        bus = MixBus("test")
        bus.pitch = 10.0
        assert bus.pitch == pytest.approx(MAX_PITCH)

    def test_get_effective_pitch_no_parent(self) -> None:
        bus = MixBus("test", pitch=2.0)
        assert bus.get_effective_pitch() == pytest.approx(2.0)

    def test_get_effective_pitch_with_parents(self) -> None:
        parent = MixBus("parent", pitch=2.0)
        child = MixBus("child", pitch=1.5, parent=parent)
        expected = 2.0 * 1.5
        assert child.get_effective_pitch() == pytest.approx(expected)

    def test_get_effective_pitch_final_clamp(self) -> None:
        """Extreme combined pitch is clamped to [MIN_PITCH, MAX_PITCH]."""
        parent = MixBus("parent", pitch=4.0)
        child = MixBus("child", pitch=4.0, parent=parent)
        assert child.get_effective_pitch() == pytest.approx(MAX_PITCH)


# =============================================================================
# Mute / Solo
# =============================================================================


class TestMuteSolo:
    """Mute and solo state transitions."""

    def test_mute_default_false(self) -> None:
        bus = MixBus("test")
        assert bus.muted is False

    def test_mute_set_true(self) -> None:
        bus = MixBus("test")
        bus.muted = True
        assert bus.muted is True

    def test_mute_set_false(self) -> None:
        bus = MixBus("test")
        bus.muted = True
        bus.muted = False
        assert bus.muted is False

    def test_toggle_mute_returns_new_state(self) -> None:
        bus = MixBus("test")
        assert bus.toggle_mute() is True
        assert bus.toggle_mute() is False

    def test_solo_default_false(self) -> None:
        bus = MixBus("test")
        assert bus.soloed is False

    def test_solo_set_true(self) -> None:
        bus = MixBus("test")
        bus.soloed = True
        assert bus.soloed is True

    def test_solo_set_false(self) -> None:
        bus = MixBus("test")
        bus.soloed = True
        bus.soloed = False
        assert bus.soloed is False

    def test_toggle_solo_returns_new_state(self) -> None:
        bus = MixBus("test")
        assert bus.toggle_solo() is True
        assert bus.toggle_solo() is False


# =============================================================================
# Filters
# =============================================================================


class TestFilters:
    """Filter state get/set/reset."""

    def test_filters_property_returns_copy(self) -> None:
        """filters returns an independent copy; mutating it does not affect
        the bus."""
        bus = MixBus("test")
        fs = bus.filters
        fs.low_pass_freq = 500.0
        # Original bus state unchanged
        assert bus.filters.low_pass_freq == pytest.approx(DEFAULT_LOW_PASS)

    def test_set_low_pass_normal(self) -> None:
        bus = MixBus("test")
        bus.set_low_pass(500.0, q=1.0, enabled=True)
        f = bus.filters
        assert f.low_pass_freq == pytest.approx(500.0)
        assert f.low_pass_q == pytest.approx(1.0)
        assert f.low_pass_enabled is True

    def test_set_low_pass_freq_clamp_low(self) -> None:
        bus = MixBus("test")
        bus.set_low_pass(1.0)  # below MIN_FILTER_FREQ (20.0)
        assert bus.filters.low_pass_freq == pytest.approx(MIN_FILTER_FREQ)

    def test_set_low_pass_freq_clamp_high(self) -> None:
        bus = MixBus("test")
        bus.set_low_pass(99999.0)  # above MAX_FILTER_FREQ (20000.0)
        assert bus.filters.low_pass_freq == pytest.approx(MAX_FILTER_FREQ)

    def test_set_low_pass_q_clamp(self) -> None:
        """Q is clamped to >= 0.1."""
        bus = MixBus("test")
        bus.set_low_pass(1000.0, q=-1.0)
        assert bus.filters.low_pass_q == pytest.approx(0.1)

    def test_set_high_pass_normal(self) -> None:
        bus = MixBus("test")
        bus.set_high_pass(80.0, q=2.0, enabled=True)
        f = bus.filters
        assert f.high_pass_freq == pytest.approx(80.0)
        assert f.high_pass_q == pytest.approx(2.0)
        assert f.high_pass_enabled is True

    def test_set_high_pass_freq_clamp_low(self) -> None:
        bus = MixBus("test")
        bus.set_high_pass(1.0)
        assert bus.filters.high_pass_freq == pytest.approx(MIN_FILTER_FREQ)

    def test_set_high_pass_freq_clamp_high(self) -> None:
        bus = MixBus("test")
        bus.set_high_pass(99999.0)
        assert bus.filters.high_pass_freq == pytest.approx(MAX_FILTER_FREQ)

    def test_set_high_pass_q_clamp(self) -> None:
        bus = MixBus("test")
        bus.set_high_pass(1000.0, q=0.0)
        assert bus.filters.high_pass_q == pytest.approx(0.1)

    def test_reset_filters(self) -> None:
        """reset_filters restores both filters to defaults."""
        bus = MixBus("test")
        bus.set_low_pass(500.0, enabled=True)
        bus.set_high_pass(100.0, enabled=True)
        bus.reset_filters()
        f = bus.filters
        assert f.low_pass_freq == pytest.approx(DEFAULT_LOW_PASS)
        assert f.high_pass_freq == pytest.approx(DEFAULT_HIGH_PASS)
        assert f.low_pass_enabled is False
        assert f.high_pass_enabled is False


# =============================================================================
# DSP Chain
# =============================================================================


class TestDSPChain:
    """Effect chain access."""

    def test_effect_chain_property(self) -> None:
        bus = MixBus("test")
        # The same internal DSPChain object is returned
        assert bus.effect_chain is bus._effect_chain

    def test_has_effects_empty(self) -> None:
        bus = MixBus("test")
        assert bus.has_effects() is False

    def test_has_effects_with_node(self) -> None:
        bus = MixBus("test")
        # Add a passthrough node to the DSP chain
        from engine.audio.dsp.dsp_node import PassthroughNode

        bus.effect_chain.add_node(PassthroughNode())
        assert bus.has_effects() is True


# =============================================================================
# Audio Processing
# =============================================================================


class TestAudioProcessing:
    """_ensure_acc_buffer, clear_acc_buffer, accumulate, write_output,
    read_acc_buffer, process_audio."""

    # --- _ensure_acc_buffer ---

    def test_ensure_acc_buffer_first_call(self) -> None:
        """First call allocates buffer of correct shape."""
        bus = MixBus("test")
        assert bus._acc_buffer is None
        bus._ensure_acc_buffer(256)
        assert bus._acc_buffer is not None
        assert bus._acc_buffer.shape == (MIXER_NUM_CHANNELS, 256)
        assert bus._acc_buffer.dtype == np.float32

    def test_ensure_acc_buffer_reallocate_larger(self) -> None:
        """If requested num_samples > existing buffer width, reallocate."""
        bus = MixBus("test")
        bus._ensure_acc_buffer(128)
        original = bus._acc_buffer
        bus._ensure_acc_buffer(512)
        assert bus._acc_buffer.shape[1] >= 512
        # Different array object after reallocation
        assert bus._acc_buffer is not original

    def test_ensure_acc_buffer_reuse_when_sufficient(self) -> None:
        """If existing buffer is large enough, no reallocation."""
        bus = MixBus("test")
        bus._ensure_acc_buffer(512)
        original = bus._acc_buffer
        bus._ensure_acc_buffer(128)
        assert bus._acc_buffer is original

    # --- clear_acc_buffer ---

    def test_clear_acc_buffer_zeros(self) -> None:
        bus = MixBus("test")
        bus._ensure_acc_buffer(64)
        bus._acc_buffer[:, :] = 1.0
        bus.clear_acc_buffer(64)
        assert np.all(bus._acc_buffer[:, :64] == 0.0)

    def test_clear_acc_buffer_allocates_if_needed(self) -> None:
        """Calling clear_acc_buffer on a bus with no buffer allocates."""
        bus = MixBus("test")
        assert bus._acc_buffer is None
        bus.clear_acc_buffer(64)
        assert bus._acc_buffer is not None
        assert bus._acc_buffer.shape == (MIXER_NUM_CHANNELS, 64)

    # --- accumulate ---

    def test_accumulate_stereo(self) -> None:
        """Stereo (2, N) samples are accumulated correctly."""
        bus = MixBus("test")
        ns = 64
        data = np.ones((2, ns), dtype=np.float32) * 0.5
        bus.clear_acc_buffer(ns)
        bus.accumulate(data, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result[:, :ns], 0.5)

    def test_accumulate_mono_broadcast_to_stereo(self) -> None:
        """Mono (1, N) input is broadcast to both channels."""
        bus = MixBus("test")
        ns = 64
        mono = np.ones((1, ns), dtype=np.float32) * 0.3
        bus.clear_acc_buffer(ns)
        bus.accumulate(mono, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result[0, :ns], 0.3)
        assert np.allclose(result[1, :ns], 0.3)

    def test_accumulate_1d_input_reshaped(self) -> None:
        """1D input array is reshaped to (1, N) then broadcast."""
        bus = MixBus("test")
        ns = 64
        flat = np.ones(ns, dtype=np.float32) * 0.4
        bus.clear_acc_buffer(ns)
        bus.accumulate(flat, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result[0, :ns], 0.4)

    def test_accumulate_channel_truncation(self) -> None:
        """Input with more channels than MIXER_NUM_CHANNELS is truncated."""
        bus = MixBus("test")
        ns = 64
        many = np.ones((8, ns), dtype=np.float32) * 0.2
        bus.clear_acc_buffer(ns)
        bus.accumulate(many, ns)
        result = bus.read_acc_buffer(ns)
        assert result.shape[0] == MIXER_NUM_CHANNELS
        assert np.allclose(result[:, :ns], 0.2)

    def test_accumulate_additive(self) -> None:
        """Multiple accumulates sum, not overwrite."""
        bus = MixBus("test")
        ns = 64
        data = np.ones((2, ns), dtype=np.float32) * 0.25
        bus.clear_acc_buffer(ns)
        bus.accumulate(data, ns)
        bus.accumulate(data, ns)
        bus.accumulate(data, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result, 0.75)

    # --- DEV-added: write_output ---

    def test_write_output_stereo(self) -> None:
        """write_output delegates to accumulate with stereo data."""
        bus = MixBus("test")
        ns = 64
        data = np.ones((2, ns), dtype=np.float32) * 0.5
        bus.clear_acc_buffer(ns)
        bus.write_output(data, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result, 0.5)

    def test_write_output_mono(self) -> None:
        """write_output handles mono input via accumulate's broadcast."""
        bus = MixBus("test")
        ns = 64
        mono = np.ones((1, ns), dtype=np.float32) * 0.7
        bus.clear_acc_buffer(ns)
        bus.write_output(mono, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result[0, :ns], 0.7)
        assert np.allclose(result[1, :ns], 0.7)

    def test_write_output_1d(self) -> None:
        """write_output handles 1D input through accumulate's reshape path."""
        bus = MixBus("test")
        ns = 64
        flat = np.ones(ns, dtype=np.float32) * 0.6
        bus.clear_acc_buffer(ns)
        bus.write_output(flat, ns)
        result = bus.read_acc_buffer(ns)
        assert np.allclose(result[0, :ns], 0.6)

    def test_write_output_triggers_buffer_alloc(self) -> None:
        """write_output on a bus with no allocation triggers _ensure_acc_buffer."""
        bus = MixBus("test")
        assert bus._acc_buffer is None
        data = np.ones((2, 32), dtype=np.float32) * 0.1
        bus.write_output(data, 32)
        assert bus._acc_buffer is not None

    def test_write_output_zero_samples(self) -> None:
        """write_output with num_samples=0 does nothing (no crash)."""
        bus = MixBus("test")
        bus.clear_acc_buffer(64)
        empty = np.zeros((2, 0), dtype=np.float32)
        bus.write_output(empty, 0)
        result = bus.read_acc_buffer(0)
        assert result.shape == (MIXER_NUM_CHANNELS, 0)

    # --- read_acc_buffer ---

    def test_read_acc_buffer_returns_copy(self) -> None:
        """read_acc_buffer returns a copy, not a view."""
        bus = MixBus("test")
        bus.clear_acc_buffer(64)
        data = np.ones((2, 64), dtype=np.float32) * 0.5
        bus.accumulate(data, 64)
        result = bus.read_acc_buffer(64)
        # Mutate the copy; original unchanged
        result[0, 0] = 999.0
        original = bus.read_acc_buffer(64)
        assert original[0, 0] == pytest.approx(0.5)

    def test_read_acc_buffer_buffer_none(self) -> None:
        """When _acc_buffer is None, returns zeros."""
        bus = MixBus("test")
        assert bus._acc_buffer is None
        result = bus.read_acc_buffer(64)
        assert np.all(result == 0.0)
        assert result.shape == (MIXER_NUM_CHANNELS, 64)

    # --- process_audio ---

    def test_process_audio_muted_returns_silence(self) -> None:
        """process_audio returns zeros when bus is muted."""
        bus = MixBus("test")
        ns = 64
        bus.clear_acc_buffer(ns)
        data = np.ones((2, ns), dtype=np.float32) * 0.9
        bus.accumulate(data, ns)
        bus.muted = True
        output = bus.process_audio(ns)
        assert np.all(output == 0.0)

    def test_process_audio_applies_volume(self) -> None:
        """process_audio scales by volume."""
        bus = MixBus("test")
        ns = 64
        bus.volume = 0.5
        bus.clear_acc_buffer(ns)
        data = np.ones((2, ns), dtype=np.float32) * 0.8
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        expected = 0.8 * 0.5
        assert np.allclose(output, expected)

    def test_process_audio_lowpass_lazy_init(self) -> None:
        """process_audio creates a LowPassFilter on first use when low-pass
        is enabled (lazy init path)."""
        bus = MixBus("test")
        ns = 64
        bus.set_low_pass(500.0, enabled=True)
        bus.clear_acc_buffer(ns)
        data = _make_sine_buffer(ns, amplitude=0.5)
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        assert bus._low_pass_filter is not None
        assert output.shape == (MIXER_NUM_CHANNELS, ns)
        # Filtering should have modified the signal
        assert not np.allclose(output, data * bus.volume)

    def test_process_audio_lowpass_reuse(self) -> None:
        """Subsequent process_audio calls reuse the existing LowPassFilter
        (no re-creation)."""
        bus = MixBus("test")
        ns = 64
        bus.set_low_pass(500.0, enabled=True)
        bus.clear_acc_buffer(ns)
        bus.accumulate(_make_sine_buffer(ns, amplitude=0.5), ns)
        bus.process_audio(ns)
        first_filter = bus._low_pass_filter
        bus.clear_acc_buffer(ns)
        bus.accumulate(_make_sine_buffer(ns, amplitude=0.5), ns)
        bus.process_audio(ns)
        assert bus._low_pass_filter is first_filter

    def test_process_audio_highpass_lazy_init(self) -> None:
        """process_audio creates a HighPassFilter on first use when high-pass
        is enabled."""
        bus = MixBus("test")
        ns = 64
        bus.set_high_pass(80.0, enabled=True)
        bus.clear_acc_buffer(ns)
        data = _make_sine_buffer(ns, amplitude=0.5)
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        assert bus._high_pass_filter is not None
        assert output.shape == (MIXER_NUM_CHANNELS, ns)

    def test_process_audio_both_filters(self) -> None:
        """Both low-pass and high-pass filters are applied in sequence."""
        bus = MixBus("test")
        ns = 128
        bus.set_low_pass(500.0, enabled=True)
        bus.set_high_pass(80.0, enabled=True)
        bus.clear_acc_buffer(ns)
        data = _make_sine_buffer(ns, amplitude=0.5)
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        assert bus._low_pass_filter is not None
        assert bus._high_pass_filter is not None
        # Output should differ from unfiltered
        raw = data * bus.volume
        assert not np.allclose(output, raw)

    def test_process_audio_dsp_chain_applied(self) -> None:
        """DSP chain effects are applied when has_effects() is True."""
        from engine.audio.dsp.dsp_node import PassthroughNode

        bus = MixBus("test")
        ns = 64
        bus.effect_chain.add_node(PassthroughNode())
        bus.clear_acc_buffer(ns)
        data = _make_sine_buffer(ns, amplitude=0.5)
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        # PassthroughNode does not modify, so output should equal volume-scaled
        expected = data * bus.volume
        assert np.allclose(output, expected)

    def test_process_audio_dsp_chain_error_handled(self) -> None:
        """When the DSP chain raises, process_audio silently falls through
        and returns the raw output."""
        bus = MixBus("test")
        ns = 64
        mock_node = MagicMock()
        mock_node.process_block.side_effect = RuntimeError("DSP fail")
        bus.effect_chain.add_node(mock_node)
        bus.clear_acc_buffer(ns)
        data = _make_sine_buffer(ns, amplitude=0.5)
        bus.accumulate(data, ns)
        # Should not raise
        output = bus.process_audio(ns)
        # Output should be the volume-scaled raw data (fallback path)
        expected = np.clip(data * bus.volume, -1.0, 1.0)
        assert np.allclose(output, expected)

    def test_process_audio_hard_clip(self) -> None:
        """Output is hard-clipped to [-1.0, 1.0]."""
        bus = MixBus("test")
        ns = 64
        bus.volume = 3.0
        bus.clear_acc_buffer(ns)
        data = np.ones((2, ns), dtype=np.float32) * 0.5  # 0.5 * 3.0 = 1.5 -> clip
        bus.accumulate(data, ns)
        output = bus.process_audio(ns)
        # Values above 1.0 should be clipped to 1.0
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)
        assert np.any(output == 1.0)

    def test_process_audio_preserves_acc_buffer(self) -> None:
        """process_audio reads from acc buffer but does not clear it."""
        bus = MixBus("test")
        ns = 64
        bus.clear_acc_buffer(ns)
        bus.accumulate(np.ones((2, ns), dtype=np.float32) * 0.5, ns)
        bus.process_audio(ns)
        # acc buffer should still have the data
        remaining = bus.read_acc_buffer(ns)
        assert np.allclose(remaining, 0.5)


# =============================================================================
# Hierarchy
# =============================================================================


class TestHierarchy:
    """set_parent, add_child, remove_child, get_ancestors, get_descendants."""

    def test_set_parent_normal(self) -> None:
        parent = MixBus("parent", BusType.CATEGORY)
        child = MixBus("child", BusType.SUB)
        child.set_parent(parent)
        assert child.parent is parent
        assert child in parent.children

    def test_set_parent_remove(self) -> None:
        """Setting parent to None removes the bus from its old parent."""
        parent = MixBus("parent")
        child = MixBus("child")
        child.set_parent(parent)
        child.set_parent(None)
        assert child.parent is None
        assert child not in parent.children

    def test_set_parent_self_error(self) -> None:
        bus = MixBus("bus")
        with pytest.raises(ValueError, match="cannot be its own parent"):
            bus.set_parent(bus)

    def test_set_parent_cycle_error(self) -> None:
        """Setting parent that would create a cycle raises ValueError."""
        grandparent = MixBus("gp", BusType.CATEGORY)
        parent = MixBus("p", BusType.CATEGORY, parent=grandparent)
        child = MixBus("c", BusType.SUB, parent=parent)
        # Attempting to make grandparent a child of child creates cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            grandparent.set_parent(child)

    def test_re_parent(self) -> None:
        """Moving a child from one parent to another updates both."""
        old_parent = MixBus("old", BusType.CATEGORY)
        new_parent = MixBus("new", BusType.CATEGORY)
        child = MixBus("child", BusType.SUB, parent=old_parent)
        child.set_parent(new_parent)
        assert child.parent is new_parent
        assert child not in old_parent.children
        assert child in new_parent.children

    def test_add_child(self) -> None:
        parent = MixBus("parent")
        child = MixBus("child")
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children

    def test_remove_child_found(self) -> None:
        parent = MixBus("parent")
        child = MixBus("child", parent=parent)
        result = parent.remove_child(child)
        assert result is True
        assert child.parent is None
        assert child not in parent.children

    def test_remove_child_not_found(self) -> None:
        parent = MixBus("parent")
        stranger = MixBus("stranger")
        result = parent.remove_child(stranger)
        assert result is False

    def test_get_ancestors_no_parent(self) -> None:
        bus = MixBus("master", BusType.MASTER)
        assert bus.get_ancestors() == []

    def test_get_ancestors_one_parent(self) -> None:
        parent = MixBus("parent")
        child = MixBus("child", parent=parent)
        assert child.get_ancestors() == [parent]

    def test_get_ancestors_many(self) -> None:
        gp = MixBus("gp", BusType.CATEGORY)
        p = MixBus("p", BusType.CATEGORY, parent=gp)
        c = MixBus("c", BusType.SUB, parent=p)
        ancestors = c.get_ancestors()
        assert ancestors == [p, gp]

    def test_get_descendants_no_children(self) -> None:
        bus = MixBus("leaf")
        assert bus.get_descendants() == []

    def test_get_descendants_direct(self) -> None:
        parent = MixBus("parent")
        child = MixBus("child", parent=parent)
        assert parent.get_descendants() == [child]

    def test_get_descendants_nested(self) -> None:
        gp = MixBus("gp", BusType.CATEGORY)
        p = MixBus("p", BusType.CATEGORY, parent=gp)
        c = MixBus("c", BusType.SUB, parent=p)
        descendants = gp.get_descendants()
        assert descendants == [p, c]


# =============================================================================
# State Management
# =============================================================================


class TestStateManagement:
    """get_state, set_state, reset."""

    def test_get_state_returns_copy(self) -> None:
        bus = MixBus("test", volume=0.5)
        state = bus.get_state()
        state.volume_linear = 0.9
        assert bus.volume == pytest.approx(0.5)

    def test_set_state_replaces(self) -> None:
        bus = MixBus("test")
        new_state = BusState(
            volume_linear=0.3,
            pitch=2.0,
            muted=True,
            soloed=True,
        )
        bus.set_state(new_state)
        assert bus.volume == pytest.approx(0.3)
        assert bus.pitch == pytest.approx(2.0)
        assert bus.muted is True
        assert bus.soloed is True

    def test_reset_restores_defaults(self) -> None:
        bus = MixBus("test", volume=0.5, pitch=2.0)
        bus.muted = True
        bus.soloed = True
        bus.reset()
        assert bus.volume == pytest.approx(DEFAULT_BUS_VOLUME)
        assert bus.pitch == pytest.approx(DEFAULT_PITCH)
        assert bus.muted is False
        assert bus.soloed is False


# =============================================================================
# Callbacks
# =============================================================================


class TestCallbacks:
    """on_change, remove_callback, _notify_change."""

    def test_on_change_registers(self) -> None:
        bus = MixBus("test")
        cb = MagicMock()
        bus.on_change(cb)
        bus.volume = 0.5
        cb.assert_called_once_with(bus)

    def test_remove_callback_found(self) -> None:
        bus = MixBus("test")
        cb = MagicMock()
        bus.on_change(cb)
        result = bus.remove_callback(cb)
        assert result is True
        bus.volume = 0.5
        cb.assert_not_called()

    def test_remove_callback_not_found(self) -> None:
        bus = MixBus("test")
        cb = MagicMock()
        result = bus.remove_callback(cb)
        assert result is False

    def test_notify_change_exception_silenced(self) -> None:
        """A callback that raises does not propagate and does not prevent
        other callbacks from firing."""
        bus = MixBus("test")
        good = MagicMock()
        bad = MagicMock(side_effect=RuntimeError("cb fail"))

        bus.on_change(bad)
        bus.on_change(good)
        # Should not raise
        bus.volume = 0.5
        good.assert_called_once_with(bus)


# =============================================================================
# String Representation
# =============================================================================


class TestRepr:
    def test_repr(self) -> None:
        bus = MixBus("test_bus", BusType.SUB)
        r = repr(bus)
        assert "test_bus" in r
        assert "sub" in r

    def test_str(self) -> None:
        bus = MixBus("test_bus", BusType.SUB)
        s = str(bus)
        assert "sub:test_bus" in s


# =============================================================================
# Data Classes
# =============================================================================


class TestDataClasses:
    """BusState.copy, FilterState.reset, FilterState.copy."""

    def test_bus_state_copy_is_independent(self) -> None:
        orig = BusState(volume_linear=0.5, pitch=2.0)
        cpy = orig.copy()
        cpy.volume_linear = 0.1
        assert orig.volume_linear == pytest.approx(0.5)

    def test_filter_state_copy_is_independent(self) -> None:
        orig = FilterState(low_pass_freq=500.0, low_pass_enabled=True)
        cpy = orig.copy()
        cpy.low_pass_freq = 1000.0
        assert orig.low_pass_freq == pytest.approx(500.0)

    def test_filter_state_reset_defaults(self) -> None:
        fs = FilterState(
            low_pass_freq=500.0,
            high_pass_freq=100.0,
            low_pass_enabled=True,
            high_pass_enabled=True,
        )
        fs.reset()
        assert fs.low_pass_freq == pytest.approx(DEFAULT_LOW_PASS)
        assert fs.high_pass_freq == pytest.approx(DEFAULT_HIGH_PASS)
        assert fs.low_pass_enabled is False
        assert fs.high_pass_enabled is False


# =============================================================================
# create_default_hierarchy
# =============================================================================


class TestCreateDefaultHierarchy:
    """create_default_hierarchy module-level function."""

    def test_master_exists(self) -> None:
        buses = create_default_hierarchy()
        assert "master" in buses
        assert buses["master"].bus_type == BusType.MASTER
        assert buses["master"].parent is None

    def test_category_buses_exist(self) -> None:
        buses = create_default_hierarchy()
        for cat in ["sfx", "music", "vo", "ambient", "ui"]:
            assert cat in buses
            assert buses[cat].bus_type == BusType.CATEGORY
            assert buses[cat].parent is buses["master"]

    def test_sub_buses_exist(self) -> None:
        buses = create_default_hierarchy()
        assert buses["footsteps"].parent is buses["sfx"]
        assert buses["weapons"].parent is buses["sfx"]
        assert buses["impacts"].parent is buses["sfx"]
        assert buses["combat"].parent is buses["music"]
        assert buses["exploration"].parent is buses["music"]
        assert buses["dialogue"].parent is buses["vo"]
        assert buses["barks"].parent is buses["vo"]

    def test_all_sub_bus_types_are_sub(self) -> None:
        buses = create_default_hierarchy()
        sub_names = [
            "footsteps", "weapons", "impacts",
            "combat", "exploration",
            "dialogue", "barks",
        ]
        for name in sub_names:
            assert buses[name].bus_type == BusType.SUB

    def test_descendants_count(self) -> None:
        """Verify the full tree has the expected number of descendants from
        master."""
        buses = create_default_hierarchy()
        master = buses["master"]
        descendants = master.get_descendants()
        assert len(descendants) == 12  # 5 categories + 7 sub-buses
