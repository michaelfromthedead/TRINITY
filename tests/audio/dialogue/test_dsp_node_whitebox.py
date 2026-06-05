"""
Whitebox tests for DSP Node base classes.

Tests SmoothedParameter, DSPNode, DSPNodeState, PassthroughNode, GainNode, and MixNode.
"""

import pytest
import threading
import time
import numpy as np
from unittest.mock import MagicMock, patch

from engine.audio.dsp.dsp_node import (
    ProcessingMode,
    BypassMode,
    DSPNodeState,
    SmoothedParameter,
    DSPNode,
    PassthroughNode,
    GainNode,
    MixNode,
)
from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_CHANNELS,
    PARAMETER_SMOOTHING_DEFAULT_MS,
)


# =============================================================================
# SmoothedParameter Tests
# =============================================================================


class TestSmoothedParameterBasic:
    """Basic tests for SmoothedParameter."""

    def test_initialization(self):
        """Test SmoothedParameter initializes correctly."""
        param = SmoothedParameter(initial_value=0.5)

        assert param.value == 0.5
        assert param.target == 0.5

    def test_custom_smoothing(self):
        """Test SmoothedParameter with custom smoothing time."""
        param = SmoothedParameter(
            initial_value=1.0,
            smoothing_ms=100.0,
            sample_rate=48000,
        )

        assert param.value == 1.0

    def test_set_value_smooth(self):
        """Test set_value starts smooth transition."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)

        param.set_value(1.0)

        assert param.target == 1.0
        assert param.value == 0.0  # Not immediately changed

    def test_set_value_immediate(self):
        """Test set_value with immediate flag."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)

        param.set_value(1.0, immediate=True)

        assert param.target == 1.0
        assert param.value == 1.0  # Immediately changed


class TestSmoothedParameterAdvance:
    """Tests for SmoothedParameter advance methods."""

    def test_advance_single_sample(self):
        """Test advance moves toward target."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)
        param.set_value(1.0)

        initial = param.value
        param.advance()

        assert param.value > initial
        assert param.value < 1.0

    def test_advance_converges(self):
        """Test advance converges to target."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=1.0)
        param.set_value(1.0)

        for _ in range(1000):
            param.advance()

        assert abs(param.value - 1.0) < 0.001

    def test_advance_block(self):
        """Test advance_block returns array."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)
        param.set_value(1.0)

        values = param.advance_block(64)

        assert len(values) == 64
        assert values[0] < values[-1]  # Should be increasing

    def test_advance_block_no_smoothing(self):
        """Test advance_block with no smoothing."""
        param = SmoothedParameter(initial_value=0.5, smoothing_ms=0.0)

        values = param.advance_block(64)

        assert np.allclose(values, 0.5)


class TestSmoothedParameterState:
    """Tests for SmoothedParameter state methods."""

    def test_is_smoothing_true(self):
        """Test is_smoothing returns True during smoothing."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)
        param.set_value(1.0)

        assert param.is_smoothing() is True

    def test_is_smoothing_false(self):
        """Test is_smoothing returns False after convergence."""
        param = SmoothedParameter(initial_value=0.5, smoothing_ms=0.0)
        param.set_value(0.5)

        assert param.is_smoothing() is False

    def test_set_smoothing_time(self):
        """Test set_smoothing_time updates coefficient."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)
        old_coeff = param._coefficient

        param.set_smoothing_time(50.0)

        assert param._coefficient != old_coeff
        assert param._smoothing_ms == 50.0

    def test_set_sample_rate(self):
        """Test set_sample_rate updates coefficient."""
        param = SmoothedParameter(initial_value=0.0, smoothing_ms=10.0)
        old_coeff = param._coefficient

        param.set_sample_rate(96000)

        assert param._coefficient != old_coeff


class TestSmoothedParameterThreadSafety:
    """Thread safety tests for SmoothedParameter."""

    def test_concurrent_set_value(self):
        """Test concurrent set_value calls."""
        param = SmoothedParameter(initial_value=0.0)

        def set_values():
            for i in range(100):
                param.set_value(float(i) / 100)
                time.sleep(0.001)

        threads = [threading.Thread(target=set_values) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


# =============================================================================
# DSPNodeState Tests
# =============================================================================


class TestDSPNodeState:
    """Tests for DSPNodeState dataclass."""

    def test_defaults(self):
        """Test DSPNodeState default values."""
        state = DSPNodeState()

        assert state.is_active is True
        assert state.is_bypassed is False
        assert state.sample_rate == DEFAULT_SAMPLE_RATE
        assert state.block_size == BLOCK_SIZE
        assert state.num_channels == 2
        assert state.latency_samples == 0
        assert state.samples_processed == 0
        assert state.blocks_processed == 0
        assert state.peak_cpu_usage == 0.0

    def test_custom_values(self):
        """Test DSPNodeState with custom values."""
        state = DSPNodeState(
            is_active=False,
            sample_rate=96000,
            num_channels=8,
        )

        assert state.is_active is False
        assert state.sample_rate == 96000
        assert state.num_channels == 8


# =============================================================================
# PassthroughNode Tests (concrete DSPNode implementation)
# =============================================================================


class TestPassthroughNodeBasic:
    """Basic tests for PassthroughNode."""

    def test_initialization(self):
        """Test PassthroughNode initializes correctly."""
        node = PassthroughNode()

        assert node.sample_rate == DEFAULT_SAMPLE_RATE
        assert node.block_size == BLOCK_SIZE
        assert node.num_channels == 2
        assert node.is_active is True
        assert node.is_bypassed is False

    def test_custom_initialization(self):
        """Test PassthroughNode with custom values."""
        node = PassthroughNode(
            sample_rate=96000,
            block_size=128,
            num_channels=4,
        )

        assert node.sample_rate == 96000
        assert node.block_size == 128
        assert node.num_channels == 4


class TestPassthroughNodeProcessing:
    """Tests for PassthroughNode processing."""

    def test_process_sample(self):
        """Test process_sample passes through."""
        node = PassthroughNode()

        result = node.process_sample(0.5, channel=0)

        assert result == 0.5

    def test_process_block(self):
        """Test process_block copies input to output."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        node.process_block(input_buffer, output_buffer)

        np.testing.assert_array_equal(output_buffer, input_buffer)

    def test_process_mono(self):
        """Test process handles mono input."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.random.randn(64).astype(np.float32)

        output = node.process(input_buffer)

        assert output.shape == (64,)
        np.testing.assert_array_almost_equal(output, input_buffer, decimal=5)

    def test_process_stereo(self):
        """Test process handles stereo input."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)

        output = node.process(input_buffer)

        assert output.shape == (2, 64)

    def test_reset(self):
        """Test reset does not crash."""
        node = PassthroughNode()

        node.reset()  # Should not raise


# =============================================================================
# DSPNode Bypass Tests
# =============================================================================


class TestDSPNodeBypass:
    """Tests for DSPNode bypass functionality."""

    def test_set_bypass_hard(self):
        """Test hard bypass mode."""
        node = PassthroughNode()

        node.set_bypass(True, BypassMode.HARD)

        assert node.is_bypassed is True
        assert node._bypass_gain.value == 0.0  # Immediate

    def test_set_bypass_soft(self):
        """Test soft bypass mode (smooth transition)."""
        node = PassthroughNode()

        node.set_bypass(True, BypassMode.SOFT)

        assert node.is_bypassed is True
        assert node._bypass_gain.target == 0.0
        # Value may not be 0 immediately

    def test_bypass_processing_hard(self):
        """Test processing with hard bypass."""
        node = PassthroughNode(num_channels=2, block_size=64)
        node.set_bypass(True, BypassMode.HARD)
        input_buffer = np.random.randn(2, 64).astype(np.float32)

        output = node.process(input_buffer)

        np.testing.assert_array_equal(output, input_buffer)

    def test_inactive_processing(self):
        """Test processing when inactive."""
        node = PassthroughNode(num_channels=2, block_size=64)
        node.set_active(False)
        input_buffer = np.random.randn(2, 64).astype(np.float32)

        output = node.process(input_buffer)

        np.testing.assert_array_equal(output, input_buffer)


# =============================================================================
# DSPNode Parameter Tests
# =============================================================================


class TestDSPNodeParameters:
    """Tests for DSPNode parameter management."""

    def test_add_parameter(self):
        """Test add_parameter adds smoothed parameter."""
        node = PassthroughNode()

        param = node.add_parameter("test_param", 0.5)

        assert param.value == 0.5
        assert node.get_parameter("test_param") is param

    def test_get_parameter_not_found(self):
        """Test get_parameter returns None for missing."""
        node = PassthroughNode()

        result = node.get_parameter("missing")

        assert result is None

    def test_set_parameter(self):
        """Test set_parameter sets value."""
        node = PassthroughNode()
        node.add_parameter("test_param", 0.5)

        node.set_parameter("test_param", 0.8)

        assert node.get_parameter("test_param").target == 0.8

    def test_set_parameter_immediate(self):
        """Test set_parameter with immediate flag."""
        node = PassthroughNode()
        node.add_parameter("test_param", 0.5)

        node.set_parameter("test_param", 0.8, immediate=True)

        assert node.get_parameter("test_param").value == 0.8


# =============================================================================
# DSPNode Configuration Tests
# =============================================================================


class TestDSPNodeConfiguration:
    """Tests for DSPNode configuration methods."""

    def test_set_sample_rate(self):
        """Test set_sample_rate updates state."""
        node = PassthroughNode()

        node.set_sample_rate(96000)

        assert node.sample_rate == 96000

    def test_set_block_size(self):
        """Test set_block_size updates state and buffers."""
        node = PassthroughNode(block_size=64)

        node.set_block_size(128)

        assert node.block_size == 128
        assert node._work_buffer.shape[1] == 128

    def test_set_num_channels(self):
        """Test set_num_channels updates state."""
        node = PassthroughNode(num_channels=2)

        node.set_num_channels(4)

        assert node.num_channels == 4
        assert node._work_buffer.shape[0] == 4

    def test_set_num_channels_exceeds_max(self):
        """Test set_num_channels raises for exceeding max."""
        node = PassthroughNode()

        with pytest.raises(ValueError):
            node.set_num_channels(MAX_CHANNELS + 1)


# =============================================================================
# DSPNode State Serialization Tests
# =============================================================================


class TestDSPNodeStateSerialization:
    """Tests for DSPNode state serialization."""

    def test_get_state(self):
        """Test get_state returns state dict."""
        node = PassthroughNode()
        node.add_parameter("test_param", 0.75)

        state = node.get_state()

        assert state["is_active"] is True
        assert state["is_bypassed"] is False
        assert state["sample_rate"] == DEFAULT_SAMPLE_RATE
        assert state["parameters"]["test_param"] == 0.75

    def test_set_state(self):
        """Test set_state restores state."""
        node = PassthroughNode()
        node.add_parameter("test_param", 0.5)
        state = {
            "is_active": False,
            "is_bypassed": True,
            "sample_rate": 96000,
            "parameters": {"test_param": 0.9},
        }

        node.set_state(state)

        assert node.is_active is False
        assert node.is_bypassed is True
        assert node.sample_rate == 96000
        assert node.get_parameter("test_param").target == 0.9


# =============================================================================
# DSPNode Statistics Tests
# =============================================================================


class TestDSPNodeStatistics:
    """Tests for DSPNode processing statistics."""

    def test_samples_processed(self):
        """Test samples_processed counter."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.zeros((2, 64), dtype=np.float32)

        node.process(input_buffer)
        node.process(input_buffer)

        assert node._state.samples_processed == 128

    def test_blocks_processed(self):
        """Test blocks_processed counter."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.zeros((2, 64), dtype=np.float32)

        node.process(input_buffer)
        node.process(input_buffer)
        node.process(input_buffer)

        assert node._state.blocks_processed == 3


# =============================================================================
# GainNode Tests
# =============================================================================


class TestGainNodeBasic:
    """Basic tests for GainNode."""

    def test_initialization_default(self):
        """Test GainNode initializes with 0dB gain."""
        node = GainNode()

        assert node.gain_db == 0.0

    def test_initialization_custom(self):
        """Test GainNode with custom gain."""
        node = GainNode(gain_db=-6.0)

        assert node.gain_db == -6.0

    def test_gain_db_setter(self):
        """Test gain_db setter."""
        node = GainNode(gain_db=0.0)

        node.gain_db = -12.0

        assert node.gain_db == -12.0


class TestGainNodeProcessing:
    """Tests for GainNode processing."""

    def test_process_sample_unity(self):
        """Test process_sample at unity gain."""
        node = GainNode(gain_db=0.0)
        # Let parameter settle
        for _ in range(100):
            node._gain.advance()

        result = node.process_sample(0.5)

        assert abs(result - 0.5) < 0.01

    def test_process_sample_negative_gain(self):
        """Test process_sample with negative gain."""
        node = GainNode(gain_db=-6.0)
        # Let parameter settle
        for _ in range(1000):
            node._gain.advance()

        result = node.process_sample(1.0)

        # -6dB is approximately 0.5 linear
        assert 0.4 < result < 0.6

    def test_process_block(self):
        """Test process_block applies gain."""
        node = GainNode(gain_db=0.0, num_channels=2, block_size=64)
        input_buffer = np.ones((2, 64), dtype=np.float32)
        output_buffer = np.zeros_like(input_buffer)

        node.process_block(input_buffer, output_buffer)

        # Output should be close to input at 0dB
        assert np.mean(output_buffer) > 0.5


# =============================================================================
# MixNode Tests
# =============================================================================


class TestMixNodeBasic:
    """Basic tests for MixNode."""

    def test_initialization_default(self):
        """Test MixNode initializes with 50% mix."""
        node = MixNode()

        assert node.wet == 0.5

    def test_initialization_custom(self):
        """Test MixNode with custom wet amount."""
        node = MixNode(wet=0.75)

        assert node.wet == 0.75

    def test_wet_setter_clamp(self):
        """Test wet setter clamps to 0-1."""
        node = MixNode()

        node.wet = 1.5
        assert node.wet == 1.0

        node.wet = -0.5
        assert node.wet == 0.0

    def test_set_dry_signal(self):
        """Test set_dry_signal stores buffer."""
        node = MixNode(num_channels=2, block_size=64)
        dry = np.random.randn(2, 64).astype(np.float32)

        node.set_dry_signal(dry)

        assert node._dry_buffer is not None

    def test_reset(self):
        """Test reset clears dry buffer."""
        node = MixNode(num_channels=2, block_size=64)
        node.set_dry_signal(np.zeros((2, 64), dtype=np.float32))

        node.reset()

        assert node._dry_buffer is None


class TestMixNodeProcessing:
    """Tests for MixNode processing."""

    def test_process_sample_full_wet(self):
        """Test process_sample at 100% wet."""
        node = MixNode(wet=1.0)
        node._wet.set_value(1.0, immediate=True)

        result = node.process_sample(0.8)

        assert abs(result - 0.8) < 0.01

    def test_process_sample_full_dry(self):
        """Test process_sample at 0% wet (dry)."""
        node = MixNode(wet=0.0)
        node._wet.set_value(0.0, immediate=True)
        # Set up dry buffer
        node._dry_buffer = np.array([[0.3]])

        result = node.process_sample(0.8, channel=0)

        assert abs(result - 0.3) < 0.01

    def test_process_block_mix(self):
        """Test process_block mixes wet and dry."""
        node = MixNode(wet=0.5, num_channels=2, block_size=64)
        wet_buffer = np.ones((2, 64), dtype=np.float32)
        dry_buffer = np.zeros((2, 64), dtype=np.float32)
        output_buffer = np.zeros_like(wet_buffer)

        node.set_dry_signal(dry_buffer)
        node.process_block(wet_buffer, output_buffer)

        # Should be ~0.5 (mix of 1.0 and 0.0)
        assert 0.4 < np.mean(output_buffer) < 0.6


# =============================================================================
# DSPNode Buffer Allocation Tests
# =============================================================================


class TestDSPNodeBufferAllocation:
    """Tests for DSPNode buffer allocation."""

    def test_allocate_aligned_buffer(self):
        """Test _allocate_aligned_buffer returns correct shape."""
        buffer = PassthroughNode._allocate_aligned_buffer(64, 2)

        assert buffer.shape == (2, 64)
        assert buffer.dtype == np.float32

    def test_work_buffer_initial_allocation(self):
        """Test work buffers allocated at init."""
        node = PassthroughNode(block_size=64, num_channels=2)

        assert node._work_buffer.shape == (2, 64)
        assert node._bypass_buffer.shape == (2, 64)

    def test_buffer_resize_on_block_change(self):
        """Test buffers resize on block size change."""
        node = PassthroughNode(block_size=64, num_channels=2)

        node.set_block_size(128)

        assert node._work_buffer.shape == (2, 128)

    def test_buffer_resize_on_channel_change(self):
        """Test buffers resize on channel count change."""
        node = PassthroughNode(block_size=64, num_channels=2)

        node.set_num_channels(4)

        assert node._work_buffer.shape == (4, 64)


# =============================================================================
# Enum Tests
# =============================================================================


class TestProcessingModeEnum:
    """Tests for ProcessingMode enum."""

    def test_values(self):
        """Test ProcessingMode values exist."""
        assert ProcessingMode.SAMPLE
        assert ProcessingMode.BLOCK
        assert ProcessingMode.SIMD


class TestBypassModeEnum:
    """Tests for BypassMode enum."""

    def test_values(self):
        """Test BypassMode values exist."""
        assert BypassMode.HARD
        assert BypassMode.SOFT
        assert BypassMode.LATENCY_COMP


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestDSPNodeThreadSafety:
    """Thread safety tests for DSPNode."""

    def test_concurrent_parameter_updates(self):
        """Test concurrent parameter updates."""
        node = PassthroughNode()
        node.add_parameter("gain", 0.5)

        def update_params():
            for i in range(100):
                node.set_parameter("gain", float(i) / 100)
                time.sleep(0.001)

        threads = [threading.Thread(target=update_params) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors

    def test_concurrent_processing(self):
        """Test concurrent processing calls."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        results = []

        def process_audio():
            for _ in range(50):
                output = node.process(input_buffer.copy())
                results.append(output.shape)
                time.sleep(0.001)

        threads = [threading.Thread(target=process_audio) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 150


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDSPNodeEdgeCases:
    """Edge case tests for DSPNode."""

    def test_process_empty_buffer(self):
        """Test processing empty buffer."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.zeros((2, 0), dtype=np.float32)

        # Should not crash
        output = node.process(input_buffer)
        assert output.shape == (2, 0)

    def test_process_different_size_buffer(self):
        """Test processing buffer of different size."""
        node = PassthroughNode(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 128).astype(np.float32)

        output = node.process(input_buffer)

        assert output.shape == (2, 128)

    def test_latency_samples_property(self):
        """Test latency_samples property."""
        node = PassthroughNode()
        node._state.latency_samples = 256

        assert node.latency_samples == 256
