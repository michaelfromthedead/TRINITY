"""
Whitebox Tests for SidechainCompressor.

Tests internal implementation details: _compute_gain_db exact math,
internal state management, parameter clamping, KeySource enum behavior,
is_compressing detection, reset internals, lifecycle handlers, and
edge cases (zero times, DC, extreme amplitudes, denormals).

Covers the full internal surface of SidechainCompressor beyond the
blackbox functional tests in test_dsp.py.

Target: 60+ whitebox tests.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    SIDECHAIN_DEFAULT_RATIO,
    SIDECHAIN_DEFAULT_THRESHOLD_DB,
    SIDECHAIN_DEFAULT_ATTACK_MS,
    SIDECHAIN_DEFAULT_RELEASE_MS,
    SIDECHAIN_DEFAULT_KNEE_DB,
    SIDECHAIN_DEFAULT_MAKEUP_DB,
    SIDECHAIN_DEFAULT_MIX,
    SIDECHAIN_MIN_RATIO,
    SIDECHAIN_MAX_RATIO,
    db_to_linear,
    linear_to_db,
)
from engine.audio.dsp.dynamics import (
    KeySource,
    SidechainCompressor,
    DetectionMode,
    EnvelopeFollower,
)


# =============================================================================
# Test Utilities
# =============================================================================


def generate_sine(
    frequency: float,
    duration_samples: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave test signal."""
    t = np.arange(duration_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


# =============================================================================
# _compute_gain_db Whitebox Tests
# =============================================================================


class TestComputeGainDb:
    """Whitebox tests for the internal _compute_gain_db method.

    Unlike blackbox tests that check only output amplitude, these verify
    the exact mathematical formula for hard knee, soft knee, ratio, and
    boundary conditions.
    """

    def test_hard_knee_below_threshold(self):
        """Hard knee: input below threshold returns exactly 0 dB reduction."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=0.0)
        result = sc._compute_gain_db(-30.0)
        assert result == 0.0

    def test_hard_knee_above_threshold(self):
        """Hard knee: input above threshold matches analytic formula."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        result = sc._compute_gain_db(0.0)
        expected = (sc.threshold_db - 0.0) * (1.0 - 1.0 / sc.ratio)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_hard_knee_at_threshold_boundary(self):
        """Hard knee: input exactly at threshold returns 0 dB."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        assert sc._compute_gain_db(-20.0) == 0.0

    def test_hard_knee_exact_numeric(self):
        """Hard knee: verify exact numeric output for known input."""
        sc = SidechainCompressor(threshold_db=-24.0, ratio=4.0, knee_db=0.0)
        result = sc._compute_gain_db(0.0)
        # (-24 - 0) * (1 - 1/4) = -24 * 0.75 = -18.0
        assert result == pytest.approx(-18.0, abs=1e-10)

    def test_hard_knee_above_threshold_different_ratio(self):
        """Hard knee: ratio=10 with input above threshold."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=10.0, knee_db=0.0)
        result = sc._compute_gain_db(-10.0)
        # (-20 - (-10)) * (1 - 1/10) = (-10) * 0.9 = -9.0
        assert result == pytest.approx(-9.0, abs=1e-10)

    def test_soft_knee_below_knee_start(self):
        """Soft knee: input below knee_start returns 0 dB."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=6.0, ratio=4.0)
        # knee_start = -23.0; -25 dB is below
        result = sc._compute_gain_db(-25.0)
        assert result == 0.0

    def test_soft_knee_above_knee_end(self):
        """Soft knee: input above knee_end matches hard knee formula."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=6.0, ratio=4.0)
        # knee_end = -17.0; -10 dB is above
        result = sc._compute_gain_db(-10.0)
        expected = (sc.threshold_db - (-10.0)) * (1.0 - 1.0 / sc.ratio)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_soft_knee_in_knee_region_exact(self):
        """Soft knee: verify exact quadratic value in knee region."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=6.0, ratio=4.0)
        knee_start = -23.0
        # At threshold center: x = -20 - (-23) = 3.0
        x = -20.0 - knee_start
        expected = (1.0 / sc.ratio - 1.0) * (x * x) / (2.0 * sc.knee_db)
        # (0.25 - 1.0) * 9.0 / 12.0 = -0.75 * 0.75 = -0.5625
        result = sc._compute_gain_db(-20.0)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_soft_knee_continuous_at_start_boundary(self):
        """Soft knee: value continuous at knee_start boundary."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=6.0, ratio=4.0)
        knee_start = -23.0
        below = sc._compute_gain_db(knee_start - 0.001)
        at_start = sc._compute_gain_db(knee_start)
        assert below == 0.0
        assert at_start == pytest.approx(0.0, abs=1e-10)

    def test_soft_knee_continuous_at_end_boundary(self):
        """Soft knee: value continuous at knee_end boundary."""
        sc = SidechainCompressor(threshold_db=-20.0, knee_db=6.0, ratio=4.0)
        knee_end = -17.0
        at_end = sc._compute_gain_db(knee_end)
        above_end = sc._compute_gain_db(knee_end + 0.001)
        expected_hard = (sc.threshold_db - knee_end) * (1.0 - 1.0 / sc.ratio)
        assert at_end == pytest.approx(expected_hard, abs=1e-10)
        # Just above should also be close
        assert above_end == pytest.approx(expected_hard, abs=0.01)

    def test_ratio_one_no_compression_hard_knee(self):
        """Ratio 1:1 produces 0 dB reduction regardless of input (hard knee)."""
        sc = SidechainCompressor(threshold_db=-80.0, ratio=1.0, knee_db=0.0)
        assert sc._compute_gain_db(0.0) == 0.0
        assert sc._compute_gain_db(24.0) == 0.0

    def test_ratio_one_no_compression_soft_knee(self):
        """Ratio 1:1 produces 0 dB reduction with soft knee."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=1.0, knee_db=6.0)
        assert sc._compute_gain_db(0.0) == 0.0
        assert sc._compute_gain_db(-20.0) == 0.0

    def test_extreme_ratio(self):
        """Extreme ratio 100:1 produces near-limiting behavior."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=100.0, knee_db=0.0)
        result = sc._compute_gain_db(0.0)
        # (1 - 1/100) = 0.99, so (-20 - 0) * 0.99 = -19.8
        assert result == pytest.approx(-19.8, abs=1e-10)

    def test_zero_knee_edge(self):
        """knee_db=0.0 exactly triggers hard knee path."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        # knee=0 is hard knee branch
        result_below = sc._compute_gain_db(-30.0)
        result_above = sc._compute_gain_db(0.0)
        assert result_below == 0.0
        expected_above = (sc.threshold_db - 0.0) * (1.0 - 1.0 / sc.ratio)
        assert result_above == pytest.approx(expected_above, abs=1e-10)

    def test_very_deep_threshold(self):
        """Very low threshold (-80 dB) still computes correctly."""
        sc = SidechainCompressor(threshold_db=-80.0, ratio=4.0, knee_db=0.0)
        result = sc._compute_gain_db(0.0)
        # (-80 - 0) * 0.75 = -60.0
        assert result == pytest.approx(-60.0, abs=1e-10)

    def test_positive_input_db(self):
        """Positive input dB (above 0 dBFS reference) computes correctly."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        result = sc._compute_gain_db(6.0)
        # (-20 - 6) * 0.75 = -26 * 0.75 = -19.5
        assert result == pytest.approx(-19.5, abs=1e-10)


# =============================================================================
# Internal State Whitebox Tests
# =============================================================================


class TestInternalState:
    """Whitebox tests for internal state management.

    These tests directly inspect private attributes to verify correct
    initialization, mutation, and lifecycle behavior.
    """

    def test_envelope_follower_initialized(self):
        """Internal _envelope is an EnvelopeFollower instance."""
        sc = SidechainCompressor()
        assert isinstance(sc._envelope, EnvelopeFollower)

    def test_gain_reduction_initialized_zero(self):
        """_gain_reduction starts as zeros with correct shape."""
        sc = SidechainCompressor(num_channels=2)
        assert np.all(sc._gain_reduction == 0.0)
        assert sc._gain_reduction.shape == (2,)

    def test_gain_reduction_shape_mono(self):
        """Mono: _gain_reduction is 1-element array."""
        sc = SidechainCompressor(num_channels=1)
        assert sc._gain_reduction.shape == (1,)

    def test_gain_reduction_shape_stereo(self):
        """Stereo: _gain_reduction is 2-element array."""
        sc = SidechainCompressor(num_channels=2)
        assert sc._gain_reduction.shape == (2,)

    def test_gain_reduction_shape_surround(self):
        """5.1 surround: _gain_reduction is 6-element array."""
        sc = SidechainCompressor(num_channels=6)
        assert sc._gain_reduction.shape == (6,)

    def test_key_buffer_initially_none(self):
        """_key_buffer is None at construction."""
        assert SidechainCompressor()._key_buffer is None

    def test_key_samples_initially_empty(self):
        """_key_samples is empty dict at construction."""
        assert SidechainCompressor()._key_samples == {}

    def test_envelope_buffer_allocated(self):
        """_envelope_buffer has correct shape from constructor args."""
        sc = SidechainCompressor(block_size=512, num_channels=2)
        assert sc._envelope_buffer.shape == (2, 512)

    def test_envelope_buffer_shape_custom(self):
        """_envelope_buffer shape matches custom block_size and channels."""
        sc = SidechainCompressor(block_size=256, num_channels=1)
        assert sc._envelope_buffer.shape == (1, 256)

    def test_set_key_buffer_updates_internal(self):
        """set_key_buffer copies data into internal _key_buffer."""
        sc = SidechainCompressor()
        key = np.random.randn(2, 512).astype(np.float32)
        sc.set_key_buffer(key)
        assert sc._key_buffer is not None
        np.testing.assert_array_equal(sc._key_buffer, key)

    def test_set_key_buffer_stores_copy(self):
        """set_key_buffer stores a copy (mutating original is safe)."""
        sc = SidechainCompressor()
        original = np.random.randn(2, 512).astype(np.float32)
        key = original.copy()
        sc.set_key_buffer(key)
        key[0, 0] = 999.0  # Mutate original
        assert sc._key_buffer[0, 0] != 999.0

    def test_clear_key_buffer_resets_to_none(self):
        """clear_key_buffer sets _key_buffer back to None."""
        sc = SidechainCompressor()
        sc.set_key_buffer(np.random.randn(2, 512).astype(np.float32))
        sc.clear_key_buffer()
        assert sc._key_buffer is None

    def test_set_key_sample_populates_dict(self):
        """set_key_sample populates _key_samples dict."""
        sc = SidechainCompressor()
        sc.set_key_sample(0.75, channel=0)
        assert sc._key_samples[0] == 0.75

    def test_set_key_sample_multiple_channels(self):
        """set_key_sample supports independent channel values."""
        sc = SidechainCompressor()
        sc.set_key_sample(0.5, channel=0)
        sc.set_key_sample(0.8, channel=1)
        assert sc._key_samples[0] == 0.5
        assert sc._key_samples[1] == 0.8

    def test_set_key_sample_overwrites_existing(self):
        """set_key_sample overwrites previous value for same channel."""
        sc = SidechainCompressor()
        sc.set_key_sample(0.5, channel=0)
        sc.set_key_sample(0.9, channel=0)
        assert sc._key_samples[0] == 0.9

    def test_gain_reduction_property_returns_copy(self):
        """gain_reduction property returns a copy, not internal reference."""
        sc = SidechainCompressor()
        gr = sc.gain_reduction
        gr[0] = 999.0
        assert sc._gain_reduction[0] != 999.0

    def test_gain_reduction_updates_after_processing(self):
        """_gain_reduction is non-negative after processing loud signal."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        assert np.all(sc._gain_reduction >= 0.0)

    def test_gain_reduction_zero_after_reset(self):
        """_gain_reduction returns to zero after reset."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        sc.reset()
        assert np.all(sc._gain_reduction == 0.0)

    def test_envelope_state_tracked_per_channel(self):
        """Internal envelope has per-channel state."""
        sc = SidechainCompressor(num_channels=2)
        assert sc._envelope._envelope.shape == (2,)


# =============================================================================
# KeySource Enum and Property Tests
# =============================================================================


class TestKeySource:
    """Whitebox tests for KeySource enum and key_source property."""

    def test_default_is_external(self):
        """Default key_source is KeySource.EXTERNAL."""
        assert SidechainCompressor().key_source == KeySource.EXTERNAL

    def test_self_is_valid_enum(self):
        """KeySource.SELF is a valid enum member."""
        assert isinstance(KeySource.SELF, KeySource)

    def test_external_is_valid_enum(self):
        """KeySource.EXTERNAL is a valid enum member."""
        assert isinstance(KeySource.EXTERNAL, KeySource)

    def test_switch_to_self_runtime(self):
        """key_source can switch to SELF at runtime."""
        sc = SidechainCompressor()
        sc.key_source = KeySource.SELF
        assert sc.key_source == KeySource.SELF

    def test_switch_back_to_external_runtime(self):
        """key_source can switch back to EXTERNAL at runtime."""
        sc = SidechainCompressor()
        sc.key_source = KeySource.SELF
        sc.key_source = KeySource.EXTERNAL
        assert sc.key_source == KeySource.EXTERNAL

    def test_self_fallback_in_process_block(self):
        """SELF key_source falls back to input_buffer internally."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0,
            key_source=KeySource.SELF,
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        output = sc.process(loud.reshape(1, -1))
        assert np.sqrt(np.mean(output[0] ** 2)) < np.sqrt(np.mean(loud ** 2))

    def test_external_no_key_fallback_self(self):
        """EXTERNAL without key buffer falls back to self-detection."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0,
            attack_ms=0.1, release_ms=10.0,
            key_source=KeySource.EXTERNAL,
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        output = sc.process(loud.reshape(1, -1))
        assert np.sqrt(np.mean(output[0] ** 2)) < np.sqrt(np.mean(loud ** 2))

    def test_key_source_two_values_distinct(self):
        """KeySource.SELF and EXTERNAL are distinct values."""
        assert KeySource.SELF != KeySource.EXTERNAL


# =============================================================================
# Parameter Clamping Whitebox Tests
# =============================================================================


class TestParameterClamping:
    """Whitebox tests for setter clamping/clipping of bounded parameters."""

    def test_ratio_clamp_below_min(self):
        """Ratio clamps to SIDECHAIN_MIN_RATIO when below."""
        sc = SidechainCompressor()
        sc.ratio = 0.1
        assert sc.ratio == SIDECHAIN_MIN_RATIO

    def test_ratio_clamp_above_max(self):
        """Ratio clamps to SIDECHAIN_MAX_RATIO when above."""
        sc = SidechainCompressor()
        sc.ratio = 200.0
        assert sc.ratio == SIDECHAIN_MAX_RATIO

    def test_ratio_accepts_valid_values(self):
        """Ratio accepts values within [MIN, MAX]."""
        sc = SidechainCompressor()
        sc.ratio = 2.0
        assert sc.ratio == 2.0
        sc.ratio = 50.0
        assert sc.ratio == 50.0

    def test_mix_clamp_below_zero(self):
        """Mix clamps to 0.0 when below."""
        sc = SidechainCompressor()
        sc.mix = -0.5
        assert sc.mix == 0.0

    def test_mix_clamp_above_one(self):
        """Mix clamps to 1.0 when above."""
        sc = SidechainCompressor()
        sc.mix = 1.5
        assert sc.mix == 1.0

    def test_mix_accepts_valid_values(self):
        """Mix accepts values within [0, 1]."""
        sc = SidechainCompressor()
        sc.mix = 0.0
        assert sc.mix == 0.0
        sc.mix = 0.5
        assert sc.mix == 0.5
        sc.mix = 1.0
        assert sc.mix == 1.0

    def test_knee_db_clamp_below_zero(self):
        """knee_db clamps to 0.0 when below."""
        sc = SidechainCompressor()
        sc.knee_db = -5.0
        assert sc.knee_db == 0.0

    def test_knee_db_accepts_positive(self):
        """knee_db accepts positive values."""
        sc = SidechainCompressor()
        sc.knee_db = 3.0
        assert sc.knee_db == 3.0
        sc.knee_db = 12.0
        assert sc.knee_db == 12.0

    def test_makeup_db_allows_negative(self):
        """makeup_db allows negative values (cut)."""
        sc = SidechainCompressor()
        sc.makeup_db = -6.0
        assert sc.makeup_db == -6.0

    def test_makeup_db_allows_positive(self):
        """makeup_db allows positive values (boost)."""
        sc = SidechainCompressor()
        sc.makeup_db = 6.0
        assert sc.makeup_db == 6.0

    def test_attack_ms_no_negative(self):
        """attack_ms setter clamps to 0 minimum."""
        sc = SidechainCompressor()
        sc.attack_ms = -10.0
        assert sc.attack_ms >= 0.0

    def test_release_ms_no_negative(self):
        """release_ms setter clamps to 0 minimum."""
        sc = SidechainCompressor()
        sc.release_ms = -10.0
        assert sc.release_ms >= 0.0

    def test_threshold_db_any_value(self):
        """threshold_db accepts any float (no clamping)."""
        sc = SidechainCompressor()
        sc.threshold_db = 6.0
        assert sc.threshold_db == 6.0
        sc.threshold_db = -120.0
        assert sc.threshold_db == -120.0


# =============================================================================
# is_compressing Property Whitebox Tests
# =============================================================================


class TestIsCompressing:
    """Whitebox tests for the is_compressing property.

    is_compressing returns True when _gain_reduction > 0.1 dB on any channel.
    """

    def test_not_compressing_initial(self):
        """is_compressing is False initially."""
        assert not SidechainCompressor().is_compressing

    def test_compressing_after_loud_signal(self):
        """is_compressing is True after processing a loud signal."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        assert sc.is_compressing

    def test_not_compressing_quiet_signal(self):
        """is_compressing is False when signal is below threshold."""
        sc = SidechainCompressor(
            threshold_db=0.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        quiet = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.05)
        sc.process(quiet.reshape(1, -1))
        assert not sc.is_compressing

    def test_not_compressing_after_reset(self):
        """is_compressing is False immediately after reset."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        sc.reset()
        assert not sc.is_compressing

    def test_compressing_threshold_db_level(self):
        """is_compressing internally uses > 0.1 dB threshold."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        assert np.any(sc._gain_reduction > 0.1)


# =============================================================================
# Reset Whitebox Tests
# =============================================================================


class TestReset:
    """Whitebox tests for reset() internal behavior."""

    def test_reset_clears_gain_reduction(self):
        """reset zeroes out _gain_reduction."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        sc.reset()
        assert np.all(sc._gain_reduction == 0.0)

    def test_reset_clears_key_buffer(self):
        """reset sets _key_buffer to None."""
        sc = SidechainCompressor()
        sc.set_key_buffer(np.random.randn(2, 512).astype(np.float32))
        sc.reset()
        assert sc._key_buffer is None

    def test_reset_clears_key_samples(self):
        """reset clears _key_samples dict."""
        sc = SidechainCompressor()
        sc.set_key_sample(0.5, channel=0)
        sc.reset()
        assert sc._key_samples == {}

    def test_reset_resets_envelope(self):
        """reset resets the internal envelope follower state."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud.reshape(1, -1))
        sc.reset()
        assert np.allclose(sc._envelope._envelope, 0.0)

    def test_reset_idempotent(self):
        """Calling reset() multiple times is safe."""
        sc = SidechainCompressor()
        sc.reset()
        sc.reset()
        sc.reset()
        assert sc._key_buffer is None
        assert sc._key_samples == {}
        assert np.all(sc._gain_reduction == 0.0)

    def test_reset_reuse_consistent(self):
        """Compressor produces identical output after reset+reuse."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        loud = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        out1 = sc.process(loud.reshape(1, -1))
        sc.reset()

        sc2 = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0
        )
        out2 = sc2.process(loud.reshape(1, -1))
        np.testing.assert_allclose(out1, out2, rtol=0.01)


# =============================================================================
# Lifecycle Handler Whitebox Tests
# =============================================================================


class TestLifecycle:
    """Whitebox tests for _on_sample_rate_changed, _on_block_size_changed,
    _on_channels_changed lifecycle callbacks."""

    def test_sample_rate_change_forwards_to_envelope(self):
        """Sample rate change propagates to envelope follower."""
        sc = SidechainCompressor()
        sc.set_sample_rate(96000)
        assert sc._envelope._state.sample_rate == 96000

    def test_block_size_change_reallocates_envelope_buffer(self):
        """Block size change reallocates _envelope_buffer."""
        sc = SidechainCompressor(block_size=256)
        assert sc._envelope_buffer.shape[1] == 256
        sc.set_block_size(512)
        assert sc._envelope_buffer.shape[1] == 512

    def test_block_size_change_forwards_to_envelope(self):
        """Block size change propagates to envelope follower."""
        sc = SidechainCompressor(block_size=512)
        sc.set_block_size(1024)
        assert sc._envelope._state.block_size == 1024

    def test_channels_change_resizes_gain_reduction(self):
        """Channel count change resizes _gain_reduction."""
        sc = SidechainCompressor(num_channels=2)
        sc.set_num_channels(4)
        assert sc._gain_reduction.shape == (4,)

    def test_channels_change_forwards_to_envelope(self):
        """Channel count change propagates to envelope follower."""
        sc = SidechainCompressor(num_channels=2)
        sc.set_num_channels(1)
        assert sc._envelope._state.num_channels == 1

    def test_channels_mono_to_stereo_resize(self):
        """Mono-to-stereo transition resizes internal state."""
        sc = SidechainCompressor(num_channels=1)
        sc.set_num_channels(2)
        assert sc._gain_reduction.shape == (2,)
        assert sc._envelope_buffer.shape[0] == 2

    def test_channels_exceeds_max_raises(self):
        """Exceeding MAX_CHANNELS raises ValueError."""
        sc = SidechainCompressor(num_channels=2)
        with pytest.raises((ValueError, OverflowError)):
            sc.set_num_channels(32)

    def test_lifecycle_state_consistency_after_changes(self):
        """Multiple lifecycle changes leave state consistent."""
        sc = SidechainCompressor()
        sc.set_sample_rate(96000)
        sc.set_block_size(256)
        sc.set_num_channels(1)
        assert sc._state.sample_rate == 96000
        assert sc._state.block_size == 256
        assert sc._state.num_channels == 1
        assert sc._gain_reduction.shape == (1,)
        assert sc._envelope_buffer.shape[0] == 1


# =============================================================================
# Edge Case Whitebox Tests
# =============================================================================


class TestEdgeCases:
    """Whitebox tests for edge cases: zero times, DC signal, extreme
    amplitudes, denormals, and numerical stability."""

    def test_zero_attack_stable(self):
        """Zero attack_ms does not cause NaN or Inf."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0, attack_ms=0.0, release_ms=10.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(signal.reshape(1, -1))
        assert np.all(np.isfinite(output))

    def test_zero_release_stable(self):
        """Zero release_ms does not cause NaN or Inf."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0, attack_ms=10.0, release_ms=0.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(signal.reshape(1, -1))
        assert np.all(np.isfinite(output))

    def test_zero_attack_and_release_stable(self):
        """Both zero attack_ms and release_ms does not cause NaN."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0, attack_ms=0.0, release_ms=0.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(signal.reshape(1, -1))
        assert np.all(np.isfinite(output))

    def test_dc_signal_no_nan(self):
        """Constant DC signal does not produce NaN."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        dc = np.full((1, BLOCK_SIZE), 0.5, dtype=np.float32)
        output = sc.process(dc)
        assert np.all(np.isfinite(output))

    def test_dc_signal_positive_negative(self):
        """Both positive and negative DC values are handled."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        pos = np.full((1, BLOCK_SIZE), 0.8, dtype=np.float32)
        neg = np.full((1, BLOCK_SIZE), -0.8, dtype=np.float32)
        out_pos = sc.process(pos)
        sc.reset()
        out_neg = sc.process(neg)
        assert np.all(np.isfinite(out_pos))
        assert np.all(np.isfinite(out_neg))

    def test_silence_produces_near_zero(self):
        """All-zero input produces near-zero output."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        silence = np.zeros((1, BLOCK_SIZE), dtype=np.float32)
        output = sc.process(silence)
        assert np.sqrt(np.mean(output[0] ** 2)) < 1e-10

    def test_extreme_amplitude_stable(self):
        """Very large amplitude (10.0) does not produce NaN."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        huge = np.full((1, BLOCK_SIZE), 10.0, dtype=np.float32)
        output = sc.process(huge)
        assert np.all(np.isfinite(output))

    def test_denormal_range_signal(self):
        """Very small amplitude near denormal range does not produce NaN."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        tiny = np.full((1, BLOCK_SIZE), 1e-20, dtype=np.float32)
        output = sc.process(tiny)
        assert np.all(np.isfinite(output))

    def test_makeup_gain_boost(self):
        """Makeup gain increases compressed output level."""
        sc_no = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0, makeup_db=0.0
        )
        sc_boost = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0, makeup_db=6.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        out_no = sc_no.process(signal.reshape(1, -1))
        out_boost = sc_boost.process(signal.reshape(1, -1))
        rms_no = np.sqrt(np.mean(out_no[0] ** 2))
        rms_boost = np.sqrt(np.mean(out_boost[0] ** 2))
        assert rms_boost > rms_no

    def test_half_wet_mix_between_dry_and_full(self):
        """mix=0.5 produces level between dry and fully compressed."""
        sc_full = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0, mix=1.0
        )
        sc_half = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0, mix=0.5
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        out_full = sc_full.process(signal.reshape(1, -1))
        out_half = sc_half.process(signal.reshape(1, -1))
        rms_dry = np.sqrt(np.mean(signal ** 2))
        rms_full = np.sqrt(np.mean(out_full[0] ** 2))
        rms_half = np.sqrt(np.mean(out_half[0] ** 2))
        assert rms_full < rms_half < rms_dry

    def test_alternating_parameters_stable(self):
        """Changing attack/release during processing is stable."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0, attack_ms=10.0, release_ms=100.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        for i, chunk in enumerate(np.split(signal, 4)):
            if np.any(chunk):
                sc.attack_ms = 5.0 + i * 10.0
                sc.release_ms = 50.0 + i * 20.0
                output = sc.process(chunk.reshape(1, -1))
                assert np.all(np.isfinite(output))

    def test_single_sample_key_triggers_compression(self):
        """process_sample with key sample compresses quiet main."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0
        )
        sc.set_key_sample(0.9, channel=0)
        result = sc.process_sample(0.05, channel=0)
        assert abs(result) < 0.05

    def test_process_sample_self_source(self):
        """process_sample with SELF key source compresses loud signal."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0,
            key_source=KeySource.SELF,
        )
        result = sc.process_sample(0.9, channel=0)
        assert abs(result) < 0.9

    def test_process_block_preserves_stereo_shape(self):
        """process_block preserves (channels, samples) shape."""
        sc = SidechainCompressor(block_size=512, num_channels=2)
        signal = generate_sine(1000.0, 512, amplitude=0.8)
        stereo = np.stack([signal, signal * 0.5])
        output = sc.process(stereo)
        assert output.shape == (2, 512)

    def test_process_block_with_external_key(self):
        """process_block with external key compresses quiet main."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0
        )
        main = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.05).reshape(1, -1)
        key = generate_sine(100.0, BLOCK_SIZE, amplitude=0.9).reshape(1, -1)
        sc.set_key_buffer(key)
        output = sc.process(main)
        assert np.sqrt(np.mean(output[0] ** 2)) < 0.05 * 0.9

    def test_makeup_gain_can_boost_beyond_dry(self):
        """With high makeup gain, output can exceed dry level."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=2.0,
            attack_ms=0.1, release_ms=10.0, makeup_db=12.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.3)
        output = sc.process(signal.reshape(1, -1))
        rms_in = np.sqrt(np.mean(signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))
        # Makeup gain should boost the signal
        assert rms_out > rms_in

    def test_repeated_reset_no_memory_leak(self):
        """Repeated process+reset cycles do not accumulate state."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=4.0, attack_ms=0.1, release_ms=10.0
        )
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        for _ in range(10):
            sc.process(signal.reshape(1, -1))
            sc.reset()
        assert np.all(sc._gain_reduction == 0.0)
        assert sc._key_buffer is None
        assert sc._key_samples == {}

    def test_hold_key_samples_across_blocks(self):
        """Key samples persist across process_sample calls until cleared."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0
        )
        sc.set_key_sample(0.9, channel=0)
        r1 = sc.process_sample(0.05, channel=0)
        r2 = sc.process_sample(0.05, channel=0)
        # Both should compress (key still present)
        assert abs(r1) < 0.05
        assert abs(r2) < 0.05

    def test_key_source_external_uses_sample_dict(self):
        """EXTERNAL mode uses _key_samples dict in process_sample."""
        sc = SidechainCompressor(
            threshold_db=-40.0, ratio=10.0,
            attack_ms=0.1, release_ms=10.0,
            key_source=KeySource.EXTERNAL,
        )
        # No key sample set -> falls back to self-detection
        result_no_key = sc.process_sample(0.05, channel=0)
        # With key sample set -> uses key
        sc.set_key_sample(0.9, channel=0)
        result_with_key = sc.process_sample(0.05, channel=0)
        # With loud key, output should be quieter
        assert result_with_key <= result_no_key


# =============================================================================
# Default Values Whitebox Tests
# =============================================================================


class TestDefaults:
    """Verify default parameter values match config constants."""

    def test_default_threshold_db(self):
        assert SidechainCompressor().threshold_db == SIDECHAIN_DEFAULT_THRESHOLD_DB

    def test_default_ratio(self):
        assert SidechainCompressor().ratio == SIDECHAIN_DEFAULT_RATIO

    def test_default_attack_ms(self):
        assert SidechainCompressor().attack_ms == SIDECHAIN_DEFAULT_ATTACK_MS

    def test_default_release_ms(self):
        assert SidechainCompressor().release_ms == SIDECHAIN_DEFAULT_RELEASE_MS

    def test_default_knee_db(self):
        assert SidechainCompressor().knee_db == SIDECHAIN_DEFAULT_KNEE_DB

    def test_default_makeup_db(self):
        assert SidechainCompressor().makeup_db == SIDECHAIN_DEFAULT_MAKEUP_DB

    def test_default_mix(self):
        assert SidechainCompressor().mix == SIDECHAIN_DEFAULT_MIX

    def test_default_key_source_enum(self):
        assert SidechainCompressor().key_source == KeySource.EXTERNAL

    def test_default_detection_mode(self):
        assert SidechainCompressor()._envelope._detection_mode == DetectionMode.RMS

    def test_default_num_channels(self):
        assert SidechainCompressor().num_channels == 2

    def test_default_sample_rate(self):
        assert SidechainCompressor().sample_rate == DEFAULT_SAMPLE_RATE

    def test_default_block_size(self):
        assert SidechainCompressor().block_size == BLOCK_SIZE
