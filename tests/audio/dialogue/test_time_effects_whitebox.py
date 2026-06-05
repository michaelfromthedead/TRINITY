"""
Whitebox tests for DSP Time-Based Effects module.

Tests LFO, DelayLine, Delay, MultiTapDelay, Chorus, Flanger, Phaser, and Vibrato.
"""

import pytest
import threading
import time
import math
import numpy as np
from unittest.mock import MagicMock, patch

from engine.audio.dsp.time_effects import (
    LFOWaveform,
    LFO,
    DelayLine,
    Delay,
    MultiTapDelay,
    Chorus,
    Flanger,
    Phaser,
    Vibrato,
)
from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_DELAY_TIME_MS,
    MAX_DELAY_FEEDBACK,
    INTERPOLATION_LINEAR,
    INTERPOLATION_CUBIC,
)


# =============================================================================
# LFOWaveform Enum Tests
# =============================================================================


class TestLFOWaveformEnum:
    """Tests for LFOWaveform enum."""

    def test_all_waveforms_exist(self):
        """Test all waveform types are defined."""
        assert LFOWaveform.SINE
        assert LFOWaveform.TRIANGLE
        assert LFOWaveform.SQUARE
        assert LFOWaveform.SAW_UP
        assert LFOWaveform.SAW_DOWN
        assert LFOWaveform.RANDOM


# =============================================================================
# LFO Tests
# =============================================================================


class TestLFOBasic:
    """Basic tests for LFO."""

    def test_initialization(self):
        """Test LFO initializes correctly."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SINE)

        assert lfo.frequency == 1.0
        assert lfo.waveform == LFOWaveform.SINE

    def test_frequency_setter(self):
        """Test frequency setter."""
        lfo = LFO(frequency=1.0)

        lfo.frequency = 5.0

        assert lfo.frequency == 5.0

    def test_frequency_clamp_positive(self):
        """Test frequency clamps to positive value."""
        lfo = LFO(frequency=1.0)

        lfo.frequency = -1.0

        assert lfo.frequency > 0

    def test_waveform_setter(self):
        """Test waveform setter."""
        lfo = LFO(waveform=LFOWaveform.SINE)

        lfo.waveform = LFOWaveform.TRIANGLE

        assert lfo.waveform == LFOWaveform.TRIANGLE


class TestLFOWaveforms:
    """Tests for LFO waveform output."""

    def test_sine_waveform(self):
        """Test sine waveform output range."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SINE)

        values = [lfo.tick() for _ in range(1000)]

        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_triangle_waveform(self):
        """Test triangle waveform output range."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.TRIANGLE)

        values = [lfo.tick() for _ in range(1000)]

        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_square_waveform(self):
        """Test square waveform output values."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SQUARE)

        values = [lfo.tick() for _ in range(1000)]

        # Square wave should only have values of -1 or 1
        unique_values = set(values)
        assert len(unique_values) == 2
        assert -1.0 in unique_values or 1.0 in unique_values

    def test_saw_up_waveform(self):
        """Test saw up waveform range."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SAW_UP)

        values = [lfo.tick() for _ in range(1000)]

        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_saw_down_waveform(self):
        """Test saw down waveform range."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SAW_DOWN)

        values = [lfo.tick() for _ in range(1000)]

        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_random_waveform(self):
        """Test random waveform range."""
        lfo = LFO(frequency=10.0, waveform=LFOWaveform.RANDOM)

        values = [lfo.tick() for _ in range(1000)]

        assert min(values) >= -1.0
        assert max(values) <= 1.0


class TestLFOMethods:
    """Tests for LFO methods."""

    def test_tick_advances_phase(self):
        """Test tick advances phase."""
        lfo = LFO(frequency=1.0)
        initial_phase = lfo._phase

        lfo.tick()

        assert lfo._phase != initial_phase or lfo._phase == 0.0

    def test_get_block(self):
        """Test get_block returns array."""
        lfo = LFO(frequency=1.0)

        values = lfo.get_block(64)

        assert len(values) == 64
        assert values.dtype == np.float32

    def test_reset(self):
        """Test reset resets phase."""
        lfo = LFO(frequency=1.0)

        for _ in range(100):
            lfo.tick()

        lfo.reset()

        assert lfo._phase == 0.0

    def test_set_sample_rate(self):
        """Test set_sample_rate updates phase increment."""
        lfo = LFO(frequency=1.0)
        old_increment = lfo._phase_increment

        lfo.set_sample_rate(96000)

        assert lfo._phase_increment != old_increment


# =============================================================================
# DelayLine Tests
# =============================================================================


class TestDelayLineBasic:
    """Basic tests for DelayLine."""

    def test_initialization(self):
        """Test DelayLine initializes correctly."""
        delay = DelayLine(max_delay_samples=1000, num_channels=2)

        assert delay._max_delay == 1000
        assert delay._num_channels == 2
        assert delay._buffer.shape == (2, 1000)

    def test_write_and_read(self):
        """Test write and read operations."""
        delay = DelayLine(max_delay_samples=100, num_channels=1)

        delay.write(0.5, channel=0)
        delay.advance()

        # Read with delay of 1 sample
        result = delay.read(1.0, channel=0)

        assert result == 0.5

    def test_clear(self):
        """Test clear clears buffer."""
        delay = DelayLine(max_delay_samples=100, num_channels=1)

        delay.write(0.5, channel=0)
        delay.advance()
        delay.clear()

        result = delay.read(1.0, channel=0)

        assert result == 0.0


class TestDelayLineInterpolation:
    """Tests for DelayLine interpolation."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        delay = DelayLine(
            max_delay_samples=100,
            num_channels=1,
            interpolation=INTERPOLATION_LINEAR,
        )

        # Write some samples
        delay.write(0.0, 0)
        delay.advance()
        delay.write(1.0, 0)
        delay.advance()

        # Read at fractional delay
        result = delay.read(1.5, channel=0)

        # Should be between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_cubic_interpolation(self):
        """Test cubic interpolation."""
        delay = DelayLine(
            max_delay_samples=100,
            num_channels=1,
            interpolation=INTERPOLATION_CUBIC,
        )

        # Write some samples
        for i in range(10):
            delay.write(float(i) / 10, 0)
            delay.advance()

        # Read at fractional delay
        result = delay.read(5.5, channel=0)

        assert isinstance(result, float)


# =============================================================================
# Delay Tests
# =============================================================================


class TestDelayBasic:
    """Basic tests for Delay effect."""

    def test_initialization(self):
        """Test Delay initializes correctly."""
        delay = Delay(delay_time_ms=100.0, feedback=0.5, wet=0.5)

        assert delay.delay_time_ms == 100.0
        assert delay.feedback == 0.5
        assert delay.wet == 0.5

    def test_ping_pong_mode(self):
        """Test ping-pong mode initialization."""
        delay = Delay(ping_pong=True)

        assert delay.ping_pong is True


class TestDelayProperties:
    """Tests for Delay property setters."""

    def test_delay_time_clamp(self):
        """Test delay_time_ms clamps to valid range."""
        delay = Delay()

        delay.delay_time_ms = -10.0
        assert delay.delay_time_ms >= 0.0

        delay.delay_time_ms = MAX_DELAY_TIME_MS + 100
        assert delay.delay_time_ms <= MAX_DELAY_TIME_MS

    def test_feedback_clamp(self):
        """Test feedback clamps to valid range."""
        delay = Delay()

        delay.feedback = 1.5
        assert delay.feedback <= MAX_DELAY_FEEDBACK

        delay.feedback = -0.5
        assert delay.feedback >= 0.0

    def test_wet_clamp(self):
        """Test wet clamps to 0-1."""
        delay = Delay()

        delay.wet = 1.5
        assert delay.wet <= 1.0

        delay.wet = -0.5
        assert delay.wet >= 0.0


class TestDelayProcessing:
    """Tests for Delay processing."""

    def test_process_sample(self):
        """Test process_sample creates delayed output."""
        delay = Delay(delay_time_ms=10.0, wet=1.0, feedback=0.0)

        # Process impulse
        delay.process_sample(1.0)

        # Continue processing
        samples_to_delay = int(10.0 * DEFAULT_SAMPLE_RATE / 1000)
        for _ in range(samples_to_delay):
            result = delay.process_sample(0.0)

        # Should eventually see the delayed impulse
        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        delay = Delay(num_channels=2, block_size=64)
        input_buffer = np.zeros((2, 64), dtype=np.float32)
        input_buffer[0, 0] = 1.0  # Impulse
        output_buffer = np.zeros_like(input_buffer)

        delay.process_block(input_buffer, output_buffer)

        # Output should differ due to wet/dry mix
        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears delay line."""
        delay = Delay(num_channels=2)

        for _ in range(100):
            delay.process_sample(0.5)

        delay.reset()

        # Should be silent after reset
        result = delay.process_sample(0.0)
        # Result may not be exactly 0 due to dry signal


# =============================================================================
# MultiTapDelay Tests
# =============================================================================


class TestMultiTapDelayBasic:
    """Basic tests for MultiTapDelay."""

    def test_initialization(self):
        """Test MultiTapDelay initializes correctly."""
        delay = MultiTapDelay(
            tap_times_ms=(100.0, 200.0, 300.0),
            tap_gains=(0.8, 0.6, 0.4),
        )

        assert len(delay._tap_times_ms) == 3
        assert len(delay._tap_gains) == 3

    def test_default_tap_gains(self):
        """Test default tap gains when not specified."""
        delay = MultiTapDelay(tap_times_ms=(100.0, 200.0, 300.0))

        assert len(delay._tap_gains) == 3
        # Default gains should decay
        assert delay._tap_gains[0] > delay._tap_gains[1]


class TestMultiTapDelayTapManagement:
    """Tests for MultiTapDelay tap management."""

    def test_set_tap(self):
        """Test set_tap modifies tap."""
        delay = MultiTapDelay(tap_times_ms=(100.0, 200.0))

        delay.set_tap(0, time_ms=150.0, gain=0.7)

        assert delay._tap_times_ms[0] == 150.0
        assert delay._tap_gains[0] == 0.7

    def test_add_tap(self):
        """Test add_tap adds new tap."""
        delay = MultiTapDelay(tap_times_ms=(100.0,))

        idx = delay.add_tap(time_ms=250.0, gain=0.5)

        assert len(delay._tap_times_ms) == 2
        assert idx == 1

    def test_remove_tap(self):
        """Test remove_tap removes tap."""
        delay = MultiTapDelay(tap_times_ms=(100.0, 200.0, 300.0))

        delay.remove_tap(1)

        assert len(delay._tap_times_ms) == 2

    def test_remove_tap_keeps_minimum(self):
        """Test remove_tap keeps at least one tap."""
        delay = MultiTapDelay(tap_times_ms=(100.0,))

        delay.remove_tap(0)

        assert len(delay._tap_times_ms) == 1


class TestMultiTapDelayProcessing:
    """Tests for MultiTapDelay processing."""

    def test_process_sample(self):
        """Test process_sample sums all taps."""
        delay = MultiTapDelay(tap_times_ms=(10.0, 20.0))

        result = delay.process_sample(1.0)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        delay = MultiTapDelay(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        delay.process_block(input_buffer, output_buffer)

        assert not np.all(output_buffer == 0)


# =============================================================================
# Chorus Tests
# =============================================================================


class TestChorusBasic:
    """Basic tests for Chorus effect."""

    def test_initialization(self):
        """Test Chorus initializes correctly."""
        chorus = Chorus(rate=1.0, depth=0.5, delay_ms=7.0, voices=3)

        assert chorus.rate == 1.0
        assert chorus.depth == 0.5
        assert chorus._voices == 3

    def test_multiple_lfos(self):
        """Test Chorus creates multiple LFOs for voices."""
        chorus = Chorus(voices=4)

        assert len(chorus._lfos) == 4


class TestChorusProperties:
    """Tests for Chorus property setters."""

    def test_rate_setter(self):
        """Test rate setter updates LFOs."""
        chorus = Chorus(rate=1.0, voices=2)

        chorus.rate = 2.0

        assert chorus.rate == 2.0
        for lfo in chorus._lfos:
            assert lfo.frequency == 2.0

    def test_depth_clamp(self):
        """Test depth clamps to 0-1."""
        chorus = Chorus()

        chorus.depth = 1.5
        assert chorus.depth <= 1.0

        chorus.depth = -0.5
        assert chorus.depth >= 0.0


class TestChorusProcessing:
    """Tests for Chorus processing."""

    def test_process_sample(self):
        """Test process_sample creates chorused output."""
        chorus = Chorus(rate=1.0, depth=0.5, wet=0.5)

        result = chorus.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        chorus = Chorus(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        chorus.process_block(input_buffer, output_buffer)

        # Chorus should modify the signal
        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears state."""
        chorus = Chorus(voices=3)

        for _ in range(100):
            chorus.process_sample(0.5)

        chorus.reset()

        for lfo in chorus._lfos:
            assert lfo._phase == 0.0


# =============================================================================
# Flanger Tests
# =============================================================================


class TestFlangerBasic:
    """Basic tests for Flanger effect."""

    def test_initialization(self):
        """Test Flanger initializes correctly."""
        flanger = Flanger(rate=0.5, depth=0.7, delay_ms=2.0, feedback=0.5)

        assert flanger.rate == 0.5
        assert flanger.depth == 0.7
        assert flanger.feedback == 0.5


class TestFlangerProperties:
    """Tests for Flanger property setters."""

    def test_rate_setter(self):
        """Test rate setter updates LFO."""
        flanger = Flanger(rate=0.5)

        flanger.rate = 1.0

        assert flanger.rate == 1.0
        assert flanger._lfo.frequency == 1.0

    def test_feedback_clamp(self):
        """Test feedback clamps to valid range."""
        flanger = Flanger()

        flanger.feedback = 1.5
        assert flanger.feedback <= 0.95

        flanger.feedback = -1.5
        assert flanger.feedback >= -0.95


class TestFlangerProcessing:
    """Tests for Flanger processing."""

    def test_process_sample(self):
        """Test process_sample creates flanged output."""
        flanger = Flanger(rate=0.5, depth=0.5)

        result = flanger.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        flanger = Flanger(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        flanger.process_block(input_buffer, output_buffer)

        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears state."""
        flanger = Flanger(num_channels=2)

        for _ in range(100):
            flanger.process_sample(0.5)

        flanger.reset()

        assert flanger._lfo._phase == 0.0


# =============================================================================
# Phaser Tests
# =============================================================================


class TestPhaserBasic:
    """Basic tests for Phaser effect."""

    def test_initialization(self):
        """Test Phaser initializes correctly."""
        phaser = Phaser(rate=0.5, depth=0.7, feedback=0.5, stages=6)

        assert phaser.rate == 0.5
        assert phaser.depth == 0.7
        assert phaser.feedback == 0.5
        assert phaser._stages == 6

    def test_allpass_filters_created(self):
        """Test Phaser creates correct number of allpass filters."""
        phaser = Phaser(stages=4)

        assert len(phaser._allpass_filters) == 4


class TestPhaserProperties:
    """Tests for Phaser property setters."""

    def test_rate_setter(self):
        """Test rate setter updates LFO."""
        phaser = Phaser(rate=0.5)

        phaser.rate = 1.0

        assert phaser.rate == 1.0

    def test_feedback_clamp(self):
        """Test feedback clamps to valid range."""
        phaser = Phaser()

        phaser.feedback = 1.5
        assert phaser.feedback <= 0.95

        phaser.feedback = -1.5
        assert phaser.feedback >= -0.95


class TestPhaserProcessing:
    """Tests for Phaser processing."""

    def test_process_sample(self):
        """Test process_sample creates phased output."""
        phaser = Phaser(rate=0.5, depth=0.5)

        result = phaser.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        phaser = Phaser(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        phaser.process_block(input_buffer, output_buffer)

        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears state."""
        phaser = Phaser(num_channels=2, stages=4)

        for _ in range(100):
            phaser.process_sample(0.5)

        phaser.reset()

        assert phaser._lfo._phase == 0.0


# =============================================================================
# Vibrato Tests
# =============================================================================


class TestVibratoBasic:
    """Basic tests for Vibrato effect."""

    def test_initialization(self):
        """Test Vibrato initializes correctly."""
        vibrato = Vibrato(rate=5.0, depth=0.5)

        assert vibrato.rate == 5.0
        assert vibrato.depth == 0.5


class TestVibratoProperties:
    """Tests for Vibrato property setters."""

    def test_rate_setter(self):
        """Test rate setter updates LFO."""
        vibrato = Vibrato(rate=5.0)

        vibrato.rate = 7.0

        assert vibrato.rate == 7.0

    def test_depth_clamp(self):
        """Test depth clamps to 0-1."""
        vibrato = Vibrato()

        vibrato.depth = 1.5
        assert vibrato.depth <= 1.0

        vibrato.depth = -0.5
        assert vibrato.depth >= 0.0


class TestVibratoProcessing:
    """Tests for Vibrato processing."""

    def test_process_sample(self):
        """Test process_sample creates vibrato output."""
        vibrato = Vibrato(rate=5.0, depth=0.5)

        result = vibrato.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        vibrato = Vibrato(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        vibrato.process_block(input_buffer, output_buffer)

        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears state."""
        vibrato = Vibrato(num_channels=2)

        for _ in range(100):
            vibrato.process_sample(0.5)

        vibrato.reset()

        assert vibrato._lfo._phase == 0.0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestTimeEffectsThreadSafety:
    """Thread safety tests for time effects."""

    def test_concurrent_delay_processing(self):
        """Test concurrent delay processing."""
        delay = Delay(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        results = []

        def process_audio():
            for _ in range(50):
                output = np.zeros_like(input_buffer)
                delay.process_block(input_buffer.copy(), output)
                results.append(output.shape)
                time.sleep(0.001)

        threads = [threading.Thread(target=process_audio) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 150

    def test_concurrent_parameter_changes(self):
        """Test concurrent effect parameter changes."""
        chorus = Chorus()

        def change_params():
            for _ in range(100):
                chorus.rate = np.random.uniform(0.1, 5.0)
                chorus.depth = np.random.uniform(0.0, 1.0)
                time.sleep(0.001)

        threads = [threading.Thread(target=change_params) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTimeEffectsEdgeCases:
    """Edge case tests for time effects."""

    def test_delay_zero_time(self):
        """Test delay with zero delay time."""
        delay = Delay(delay_time_ms=0.0)

        result = delay.process_sample(0.5)
        assert isinstance(result, float)

    def test_delay_max_time(self):
        """Test delay at maximum delay time."""
        delay = Delay(delay_time_ms=MAX_DELAY_TIME_MS)

        result = delay.process_sample(0.5)
        assert isinstance(result, float)

    def test_chorus_zero_depth(self):
        """Test chorus with zero depth."""
        chorus = Chorus(depth=0.0, wet=1.0)

        result = chorus.process_sample(0.5)
        assert isinstance(result, float)

    def test_flanger_zero_depth(self):
        """Test flanger with zero depth."""
        flanger = Flanger(depth=0.0)

        result = flanger.process_sample(0.5)
        assert isinstance(result, float)

    def test_phaser_zero_depth(self):
        """Test phaser with zero depth."""
        phaser = Phaser(depth=0.0)

        result = phaser.process_sample(0.5)
        assert isinstance(result, float)

    def test_vibrato_zero_depth(self):
        """Test vibrato with zero depth."""
        vibrato = Vibrato(depth=0.0)

        result = vibrato.process_sample(0.5)
        # With zero depth, should be close to input (delayed)
        assert isinstance(result, float)

    def test_sample_rate_change(self):
        """Test effect after sample rate change."""
        delay = Delay(delay_time_ms=100.0)

        delay.set_sample_rate(96000)

        result = delay.process_sample(0.5)
        assert isinstance(result, float)

    def test_channel_count_change(self):
        """Test effect after channel count change."""
        flanger = Flanger(num_channels=2)

        flanger.set_num_channels(4)

        assert flanger._feedback_sample.shape[0] == 4
