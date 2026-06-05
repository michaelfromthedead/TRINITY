"""
Tests for SSR Temporal Reprojection System

Comprehensive whitebox and blackbox tests for:
- TemporalSample: Sample data structure and operations
- TemporalBuffer: Single buffer management
- TemporalBufferSet: Ping-pong buffer pair
- SSRTemporalConfig: Configuration validation
- TemporalStats: Statistics tracking
- SSRTemporalReprojection: Main reprojection pass

Requirements tested:
- Temporal reprojection eliminates flickering
- Disocclusion properly rejects invalid history
- Converges to stable result after 8 accumulated frames
- No ghosting on fast-moving objects
"""

import math
import pytest

from engine.rendering.reflections.ssr_temporal import (
    DisocclusionMode,
    SSRTemporalConfig,
    SSRTemporalReprojection,
    TemporalBuffer,
    TemporalBufferSet,
    TemporalQuality,
    TemporalSample,
    TemporalStats,
    _QUALITY_PARAMS,
)


# =============================================================================
# TemporalSample Tests
# =============================================================================


class TestTemporalSample:
    """Tests for TemporalSample data structure."""

    def test_default_construction(self):
        """Test default sample construction."""
        sample = TemporalSample()

        assert sample.color == (0.0, 0.0, 0.0)
        assert sample.alpha == 0.0
        assert sample.depth == 0.0
        assert sample.normal == (0.0, 1.0, 0.0)
        assert sample.velocity == (0.0, 0.0)
        assert sample.confidence == 0.0
        assert sample.frame_count == 0

    def test_custom_construction(self):
        """Test sample with custom values."""
        sample = TemporalSample(
            color=(1.0, 0.5, 0.25),
            alpha=0.9,
            depth=10.0,
            normal=(0.0, 0.0, 1.0),
            velocity=(0.01, -0.02),
            confidence=0.8,
            frame_count=5,
        )

        assert sample.color == (1.0, 0.5, 0.25)
        assert sample.alpha == 0.9
        assert sample.depth == 10.0
        assert sample.normal == (0.0, 0.0, 1.0)
        assert sample.velocity == (0.01, -0.02)
        assert sample.confidence == 0.8
        assert sample.frame_count == 5

    def test_is_valid_with_confidence_and_depth(self):
        """Test validity check requires confidence and depth."""
        valid_sample = TemporalSample(confidence=0.5, depth=5.0)
        assert valid_sample.is_valid()

        zero_confidence = TemporalSample(confidence=0.0, depth=5.0)
        assert not zero_confidence.is_valid()

        zero_depth = TemporalSample(confidence=0.5, depth=0.0)
        assert not zero_depth.is_valid()

    def test_luminance_calculation(self):
        """Test luminance using rec.709 coefficients."""
        # Pure white
        white = TemporalSample(color=(1.0, 1.0, 1.0))
        assert pytest.approx(white.luminance(), rel=1e-4) == 1.0

        # Pure black
        black = TemporalSample(color=(0.0, 0.0, 0.0))
        assert black.luminance() == 0.0

        # rec.709: 0.2126*R + 0.7152*G + 0.0722*B
        mixed = TemporalSample(color=(0.5, 0.3, 0.8))
        expected = 0.2126 * 0.5 + 0.7152 * 0.3 + 0.0722 * 0.8
        assert pytest.approx(mixed.luminance(), rel=1e-4) == expected

    def test_velocity_magnitude_zero(self):
        """Test velocity magnitude for stationary pixel."""
        sample = TemporalSample(velocity=(0.0, 0.0))
        assert sample.velocity_magnitude() == 0.0

    def test_velocity_magnitude_nonzero(self):
        """Test velocity magnitude calculation."""
        sample = TemporalSample(velocity=(3.0, 4.0))
        assert pytest.approx(sample.velocity_magnitude(), rel=1e-6) == 5.0

    def test_velocity_magnitude_negative(self):
        """Test velocity magnitude with negative components."""
        sample = TemporalSample(velocity=(-0.6, -0.8))
        assert pytest.approx(sample.velocity_magnitude(), rel=1e-6) == 1.0

    def test_blend_with_zero_weight(self):
        """Test blending with zero weight keeps original."""
        sample_a = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0)
        sample_b = TemporalSample(color=(0.0, 1.0, 0.0), alpha=0.5)

        result = sample_a.blend_with(sample_b, 0.0)

        assert result.color == (1.0, 0.0, 0.0)
        assert result.alpha == 1.0

    def test_blend_with_full_weight(self):
        """Test blending with weight 1.0 uses other sample."""
        sample_a = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0)
        sample_b = TemporalSample(color=(0.0, 1.0, 0.0), alpha=0.5)

        result = sample_a.blend_with(sample_b, 1.0)

        assert result.color == (0.0, 1.0, 0.0)
        assert result.alpha == 0.5

    def test_blend_with_half_weight(self):
        """Test blending at 50%."""
        sample_a = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0)
        sample_b = TemporalSample(color=(0.0, 1.0, 0.0), alpha=0.0)

        result = sample_a.blend_with(sample_b, 0.5)

        assert pytest.approx(result.color[0], rel=1e-6) == 0.5
        assert pytest.approx(result.color[1], rel=1e-6) == 0.5
        assert pytest.approx(result.color[2], rel=1e-6) == 0.0
        assert pytest.approx(result.alpha, rel=1e-6) == 0.5

    def test_blend_increments_frame_count(self):
        """Test that blending increments frame count."""
        sample_a = TemporalSample(frame_count=3)
        sample_b = TemporalSample(frame_count=5)

        result = sample_a.blend_with(sample_b, 0.5)

        # Should be max + 1
        assert result.frame_count == 6


# =============================================================================
# TemporalBuffer Tests
# =============================================================================


class TestTemporalBuffer:
    """Tests for TemporalBuffer single buffer management."""

    def test_default_construction(self):
        """Test default buffer state."""
        buf = TemporalBuffer()

        assert buf.texture is None
        assert buf.width == 0
        assert buf.height == 0
        assert buf.frame_index == -1
        assert buf.valid is False
        assert buf.format == "rgba16f"

    def test_is_allocated_when_no_texture(self):
        """Test is_allocated returns False without texture."""
        buf = TemporalBuffer()
        assert not buf.is_allocated()

    def test_matches_dimensions_exact(self):
        """Test dimension matching."""
        buf = TemporalBuffer(width=1920, height=1080)

        assert buf.matches_dimensions(1920, 1080)
        assert not buf.matches_dimensions(1920, 720)
        assert not buf.matches_dimensions(1280, 1080)

    def test_invalidate_resets_state(self):
        """Test invalidation resets validity and frame index."""
        buf = TemporalBuffer(valid=True, frame_index=10)

        buf.invalidate()

        assert buf.valid is False
        assert buf.frame_index == -1

    def test_mark_written_sets_state(self):
        """Test mark_written updates validity and frame."""
        buf = TemporalBuffer()

        buf.mark_written(42)

        assert buf.valid is True
        assert buf.frame_index == 42

    def test_age_returns_negative_when_invalid(self):
        """Test age returns -1 for invalid buffer."""
        buf = TemporalBuffer(valid=False)
        assert buf.age(100) == -1

    def test_age_calculation(self):
        """Test age calculation from frame index."""
        buf = TemporalBuffer(valid=True, frame_index=10)

        assert buf.age(10) == 0
        assert buf.age(15) == 5
        assert buf.age(100) == 90


# =============================================================================
# TemporalBufferSet Tests
# =============================================================================


class TestTemporalBufferSet:
    """Tests for TemporalBufferSet ping-pong management."""

    def test_default_construction(self):
        """Test default buffer set state."""
        buf_set = TemporalBufferSet()

        assert buf_set.read_index == 0
        assert buf_set.frame_count == 0

    def test_history_buffer_when_read_index_zero(self):
        """Test history buffer selection."""
        buf_set = TemporalBufferSet()
        buf_set.buffer_a.width = 100
        buf_set.buffer_b.width = 200

        # read_index=0 -> history is buffer_a
        assert buf_set.history_buffer.width == 100
        assert buf_set.current_buffer.width == 200

    def test_swap_alternates_buffers(self):
        """Test swap alternates read and write buffers."""
        buf_set = TemporalBufferSet()
        buf_set.buffer_a.width = 100
        buf_set.buffer_b.width = 200

        # Initial: history=a, current=b
        assert buf_set.history_buffer.width == 100

        buf_set.swap()

        # After swap: history=b, current=a
        assert buf_set.history_buffer.width == 200
        assert buf_set.read_index == 1
        assert buf_set.frame_count == 1

    def test_swap_multiple_times(self):
        """Test multiple swaps cycle correctly."""
        buf_set = TemporalBufferSet()

        for i in range(10):
            expected_index = i % 2
            assert buf_set.read_index == expected_index
            assert buf_set.frame_count == i
            buf_set.swap()

    def test_invalidate_all_resets_state(self):
        """Test invalidate_all resets both buffers."""
        buf_set = TemporalBufferSet()
        buf_set.buffer_a.mark_written(5)
        buf_set.buffer_b.mark_written(6)
        buf_set.frame_count = 10

        buf_set.invalidate_all()

        assert not buf_set.buffer_a.valid
        assert not buf_set.buffer_b.valid
        assert buf_set.frame_count == 0

    def test_needs_resize_when_different(self):
        """Test needs_resize detects dimension changes."""
        buf_set = TemporalBufferSet()
        buf_set.buffer_a.width = 1920
        buf_set.buffer_a.height = 1080
        buf_set.buffer_b.width = 1920
        buf_set.buffer_b.height = 1080

        assert not buf_set.needs_resize(1920, 1080)
        assert buf_set.needs_resize(1280, 720)

    def test_convergence_progress_zero_frames(self):
        """Test convergence at zero frames."""
        buf_set = TemporalBufferSet()
        assert buf_set.get_convergence_progress(8) == 0.0

    def test_convergence_progress_partial(self):
        """Test convergence progress calculation."""
        buf_set = TemporalBufferSet()
        buf_set.frame_count = 4

        assert buf_set.get_convergence_progress(8) == 0.5

    def test_convergence_progress_full(self):
        """Test convergence progress caps at 1.0."""
        buf_set = TemporalBufferSet()
        buf_set.frame_count = 16

        assert buf_set.get_convergence_progress(8) == 1.0

    def test_is_converged_true_at_target(self):
        """Test is_converged at target frame count."""
        buf_set = TemporalBufferSet()
        buf_set.frame_count = 8

        assert buf_set.is_converged(8)

    def test_is_converged_false_below_target(self):
        """Test is_converged below target."""
        buf_set = TemporalBufferSet()
        buf_set.frame_count = 7

        assert not buf_set.is_converged(8)


# =============================================================================
# SSRTemporalConfig Tests
# =============================================================================


class TestSSRTemporalConfig:
    """Tests for SSRTemporalConfig validation and presets."""

    def test_default_construction(self):
        """Test default configuration values."""
        config = SSRTemporalConfig()

        assert config.quality == TemporalQuality.HIGH
        assert config.disocclusion_mode == DisocclusionMode.COMBINED
        assert config.depth_threshold == 0.01
        assert config.normal_threshold == 0.9
        assert config.velocity_threshold == 0.02
        assert config.neighborhood_size == 3
        assert config.use_variance_clipping is True
        assert config.use_ycocg_space is True
        assert config.anti_flicker is True

    def test_invalid_neighborhood_size_raises(self):
        """Test invalid neighborhood size raises ValueError."""
        with pytest.raises(ValueError, match="neighborhood_size must be 3, 5, or 7"):
            SSRTemporalConfig(neighborhood_size=4)

    def test_valid_neighborhood_sizes(self):
        """Test valid neighborhood sizes accepted."""
        for size in (3, 5, 7):
            config = SSRTemporalConfig(neighborhood_size=size)
            assert config.neighborhood_size == size

    def test_invalid_depth_threshold_raises(self):
        """Test non-positive depth threshold raises."""
        with pytest.raises(ValueError, match="depth_threshold must be positive"):
            SSRTemporalConfig(depth_threshold=0.0)

        with pytest.raises(ValueError, match="depth_threshold must be positive"):
            SSRTemporalConfig(depth_threshold=-0.1)

    def test_invalid_normal_threshold_raises(self):
        """Test out-of-range normal threshold raises."""
        with pytest.raises(ValueError, match="normal_threshold must be in"):
            SSRTemporalConfig(normal_threshold=-0.1)

        with pytest.raises(ValueError, match="normal_threshold must be in"):
            SSRTemporalConfig(normal_threshold=1.5)

    def test_invalid_velocity_threshold_raises(self):
        """Test negative velocity threshold raises."""
        with pytest.raises(ValueError, match="velocity_threshold must be non-negative"):
            SSRTemporalConfig(velocity_threshold=-0.01)

    def test_get_history_weight_from_preset(self):
        """Test history weight from quality preset."""
        for quality, (expected_weight, _, _) in _QUALITY_PARAMS.items():
            config = SSRTemporalConfig(quality=quality, history_weight=None)
            assert config.get_history_weight() == expected_weight

    def test_get_history_weight_override(self):
        """Test history weight override."""
        config = SSRTemporalConfig(history_weight=0.75)
        assert config.get_history_weight() == 0.75

    def test_get_history_weight_clamped(self):
        """Test history weight is clamped to [0, 1]."""
        config = SSRTemporalConfig(history_weight=1.5)
        assert config.get_history_weight() == 1.0

        config = SSRTemporalConfig(history_weight=-0.5)
        assert config.get_history_weight() == 0.0

    def test_get_frames_to_converge_from_preset(self):
        """Test frames from quality preset."""
        for quality, (_, expected_frames, _) in _QUALITY_PARAMS.items():
            config = SSRTemporalConfig(quality=quality, frames_to_converge=None)
            assert config.get_frames_to_converge() == expected_frames

    def test_get_frames_to_converge_override(self):
        """Test frames override."""
        config = SSRTemporalConfig(frames_to_converge=12)
        assert config.get_frames_to_converge() == 12

    def test_get_variance_gamma_from_preset(self):
        """Test variance gamma from quality preset."""
        for quality, (_, _, expected_gamma) in _QUALITY_PARAMS.items():
            config = SSRTemporalConfig(quality=quality, variance_gamma=None)
            assert config.get_variance_gamma() == expected_gamma

    def test_quality_presets_have_expected_values(self):
        """Test quality presets have sensible progression."""
        # Higher quality = higher history weight, more frames
        low = _QUALITY_PARAMS[TemporalQuality.LOW]
        medium = _QUALITY_PARAMS[TemporalQuality.MEDIUM]
        high = _QUALITY_PARAMS[TemporalQuality.HIGH]
        ultra = _QUALITY_PARAMS[TemporalQuality.ULTRA]

        # History weight increases
        assert low[0] < medium[0] < high[0] < ultra[0]

        # Frames to converge increases
        assert low[1] < medium[1] < high[1] < ultra[1]

        # Variance gamma decreases (tighter clipping)
        assert low[2] > medium[2] > high[2] > ultra[2]


# =============================================================================
# TemporalStats Tests
# =============================================================================


class TestTemporalStats:
    """Tests for TemporalStats tracking."""

    def test_default_construction(self):
        """Test default stats are zero."""
        stats = TemporalStats()

        assert stats.total_pixels == 0
        assert stats.valid_history_pixels == 0
        assert stats.rejected_pixels == 0
        assert stats.average_confidence == 0.0

    def test_rejection_rate_zero_pixels(self):
        """Test rejection rate with no pixels."""
        stats = TemporalStats()
        assert stats.rejection_rate == 0.0

    def test_rejection_rate_calculation(self):
        """Test rejection rate calculation."""
        stats = TemporalStats(total_pixels=100, rejected_pixels=25)
        assert stats.rejection_rate == 0.25

    def test_convergence_rate_calculation(self):
        """Test convergence rate calculation."""
        stats = TemporalStats(total_pixels=100, converged_pixels=80)
        assert stats.convergence_rate == 0.8

    def test_history_usage_rate_calculation(self):
        """Test history usage rate calculation."""
        stats = TemporalStats(total_pixels=100, valid_history_pixels=90)
        assert stats.history_usage_rate == 0.9

    def test_reset_clears_all(self):
        """Test reset clears all statistics."""
        stats = TemporalStats(
            total_pixels=100,
            valid_history_pixels=90,
            rejected_pixels=10,
            average_confidence=0.5,
            max_velocity=1.0,
        )

        stats.reset()

        assert stats.total_pixels == 0
        assert stats.valid_history_pixels == 0
        assert stats.rejected_pixels == 0
        assert stats.average_confidence == 0.0
        assert stats.max_velocity == 0.0


# =============================================================================
# SSRTemporalReprojection Tests - Setup
# =============================================================================


class TestSSRTemporalReprojectionSetup:
    """Tests for SSRTemporalReprojection initialization and setup."""

    def test_default_construction(self):
        """Test default construction without device."""
        reprojection = SSRTemporalReprojection()

        assert reprojection.device is None
        assert reprojection.config.quality == TemporalQuality.HIGH
        assert not reprojection.is_initialized
        assert reprojection.frame_index == 0

    def test_construction_with_config(self):
        """Test construction with custom config."""
        config = SSRTemporalConfig(quality=TemporalQuality.LOW)
        reprojection = SSRTemporalReprojection(config=config)

        assert reprojection.config.quality == TemporalQuality.LOW

    def test_setup_initializes_buffers(self):
        """Test setup creates buffers with correct dimensions."""
        reprojection = SSRTemporalReprojection()

        reprojection.setup(1920, 1080)

        assert reprojection.is_initialized
        assert reprojection.width == 1920
        assert reprojection.height == 1080
        assert reprojection.buffers.buffer_a.width == 1920
        assert reprojection.buffers.buffer_b.height == 1080

    def test_setup_invalid_dimensions_raises(self):
        """Test setup with invalid dimensions raises."""
        reprojection = SSRTemporalReprojection()

        with pytest.raises(ValueError, match="Invalid dimensions"):
            reprojection.setup(0, 1080)

        with pytest.raises(ValueError, match="Invalid dimensions"):
            reprojection.setup(1920, -1)

    def test_setup_resize_invalidates_history(self):
        """Test resizing invalidates history buffers."""
        reprojection = SSRTemporalReprojection()

        reprojection.setup(1920, 1080)
        reprojection.buffers.buffer_a.mark_written(5)
        reprojection.buffers.frame_count = 10

        reprojection.setup(1280, 720)

        assert reprojection.buffers.frame_count == 0
        assert not reprojection.buffers.buffer_a.valid

    def test_invalidate_history(self):
        """Test manual history invalidation."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)
        reprojection.buffers.frame_count = 10

        reprojection.invalidate_history()

        assert reprojection.buffers.frame_count == 0

    def test_convergence_properties(self):
        """Test convergence progress and is_converged."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)

        # Initial state
        assert reprojection.convergence_progress == 0.0
        assert not reprojection.is_converged

        # Simulate frame accumulation
        for _ in range(8):
            reprojection.buffers.swap()

        assert reprojection.convergence_progress == 1.0
        assert reprojection.is_converged


# =============================================================================
# SSRTemporalReprojection Tests - Disocclusion
# =============================================================================


class TestSSRTemporalReprojectionDisocclusion:
    """Tests for disocclusion detection methods."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance for tests."""
        config = SSRTemporalConfig(
            depth_threshold=0.01,
            normal_threshold=0.9,
            velocity_threshold=0.02,
        )
        return SSRTemporalReprojection(config=config)

    def test_depth_rejection_same_depth(self, reprojection):
        """Test depth rejection with matching depths."""
        weight = reprojection.compute_depth_rejection(10.0, 10.0)
        assert weight == 1.0

    def test_depth_rejection_small_difference(self, reprojection):
        """Test depth rejection with small difference."""
        weight = reprojection.compute_depth_rejection(10.0, 10.05)
        assert 0.0 < weight < 1.0

    def test_depth_rejection_large_difference(self, reprojection):
        """Test depth rejection with large difference."""
        weight = reprojection.compute_depth_rejection(10.0, 20.0)
        assert weight == 0.0

    def test_depth_rejection_invalid_depth(self, reprojection):
        """Test depth rejection with zero/negative depth."""
        assert reprojection.compute_depth_rejection(0.0, 10.0) == 0.0
        assert reprojection.compute_depth_rejection(10.0, 0.0) == 0.0

    def test_normal_rejection_same_normal(self, reprojection):
        """Test normal rejection with matching normals."""
        normal = (0.0, 1.0, 0.0)
        weight = reprojection.compute_normal_rejection(normal, normal)
        assert pytest.approx(weight, rel=1e-4) == 1.0

    def test_normal_rejection_orthogonal(self, reprojection):
        """Test normal rejection with perpendicular normals."""
        n1 = (0.0, 1.0, 0.0)
        n2 = (1.0, 0.0, 0.0)
        weight = reprojection.compute_normal_rejection(n1, n2)
        assert weight == 0.0

    def test_normal_rejection_slight_difference(self, reprojection):
        """Test normal rejection with slight difference."""
        n1 = (0.0, 1.0, 0.0)
        # Small tilt
        n2 = (0.1, 0.995, 0.0)  # Roughly 6 degrees
        weight = reprojection.compute_normal_rejection(n1, n2)
        assert 0.0 < weight < 1.0

    def test_velocity_rejection_stationary(self, reprojection):
        """Test velocity rejection for stationary pixel."""
        weight = reprojection.compute_velocity_rejection((0.0, 0.0))
        assert weight == 1.0

    def test_velocity_rejection_slow_motion(self, reprojection):
        """Test velocity rejection for slow motion."""
        weight = reprojection.compute_velocity_rejection((0.01, 0.0))
        assert weight == 1.0

    def test_velocity_rejection_fast_motion(self, reprojection):
        """Test velocity rejection for fast motion reduces weight."""
        weight = reprojection.compute_velocity_rejection((0.1, 0.1))
        assert 0.0 <= weight < 1.0

    def test_combined_disocclusion_all_pass(self, reprojection):
        """Test combined disocclusion with all criteria passing."""
        reprojection.config.disocclusion_mode = DisocclusionMode.COMBINED

        weight = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
        )

        assert pytest.approx(weight, rel=1e-4) == 1.0

    def test_combined_disocclusion_depth_fails(self, reprojection):
        """Test combined disocclusion when depth fails."""
        reprojection.config.disocclusion_mode = DisocclusionMode.COMBINED

        weight = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=20.0,  # Large difference
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
        )

        assert weight == 0.0

    def test_disocclusion_mode_depth_only(self, reprojection):
        """Test DEPTH_ONLY mode ignores normal and velocity."""
        reprojection.config.disocclusion_mode = DisocclusionMode.DEPTH_ONLY

        # Depth passes, normal and velocity would fail
        weight = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(1.0, 0.0, 0.0),  # Orthogonal
            velocity=(1.0, 1.0),  # Fast
        )

        assert weight == 1.0

    def test_disocclusion_mode_normal_only(self, reprojection):
        """Test NORMAL_ONLY mode ignores depth and velocity."""
        reprojection.config.disocclusion_mode = DisocclusionMode.NORMAL_ONLY

        weight = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=50.0,  # Would fail depth
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(1.0, 1.0),  # Would fail velocity
        )

        assert pytest.approx(weight, rel=1e-4) == 1.0

    def test_disocclusion_mode_velocity_only(self, reprojection):
        """Test VELOCITY_ONLY mode ignores depth and normal."""
        reprojection.config.disocclusion_mode = DisocclusionMode.VELOCITY_ONLY

        weight = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=50.0,  # Would fail depth
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(1.0, 0.0, 0.0),  # Would fail normal
            velocity=(0.0, 0.0),  # Passes
        )

        assert weight == 1.0

    def test_disocclusion_mode_adaptive(self, reprojection):
        """Test ADAPTIVE mode weights criteria by velocity."""
        reprojection.config.disocclusion_mode = DisocclusionMode.ADAPTIVE

        # Low velocity - geometry matters more
        weight_low = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.001, 0.0),
        )

        # High velocity - velocity matters more
        weight_high = reprojection.compute_disocclusion_weight(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.1, 0.1),
        )

        assert weight_low >= weight_high


# =============================================================================
# SSRTemporalReprojection Tests - Color Space
# =============================================================================


class TestSSRTemporalReprojectionColorSpace:
    """Tests for color space conversion methods."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        return SSRTemporalReprojection()

    def test_rgb_to_ycocg_black(self, reprojection):
        """Test RGB to YCoCg for black."""
        y, co, cg = reprojection.rgb_to_ycocg((0.0, 0.0, 0.0))
        assert y == 0.0
        assert co == 0.0
        assert cg == 0.0

    def test_rgb_to_ycocg_white(self, reprojection):
        """Test RGB to YCoCg for white."""
        y, co, cg = reprojection.rgb_to_ycocg((1.0, 1.0, 1.0))
        assert pytest.approx(y, rel=1e-6) == 1.0
        assert pytest.approx(co, rel=1e-6) == 0.0
        assert pytest.approx(cg, rel=1e-6) == 0.0

    def test_rgb_to_ycocg_red(self, reprojection):
        """Test RGB to YCoCg for red."""
        y, co, cg = reprojection.rgb_to_ycocg((1.0, 0.0, 0.0))
        assert pytest.approx(y, rel=1e-6) == 0.25
        assert pytest.approx(co, rel=1e-6) == 0.5
        assert pytest.approx(cg, rel=1e-6) == -0.25

    def test_ycocg_to_rgb_roundtrip(self, reprojection):
        """Test RGB -> YCoCg -> RGB roundtrip."""
        original = (0.3, 0.6, 0.9)
        ycocg = reprojection.rgb_to_ycocg(original)
        recovered = reprojection.ycocg_to_rgb(ycocg)

        assert pytest.approx(recovered[0], rel=1e-6) == original[0]
        assert pytest.approx(recovered[1], rel=1e-6) == original[1]
        assert pytest.approx(recovered[2], rel=1e-6) == original[2]

    def test_ycocg_roundtrip_various_colors(self, reprojection):
        """Test roundtrip for various colors."""
        colors = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.25, 0.75),
        ]

        for original in colors:
            ycocg = reprojection.rgb_to_ycocg(original)
            recovered = reprojection.ycocg_to_rgb(ycocg)

            assert pytest.approx(recovered[0], rel=1e-5) == original[0]
            assert pytest.approx(recovered[1], rel=1e-5) == original[1]
            assert pytest.approx(recovered[2], rel=1e-5) == original[2]


# =============================================================================
# SSRTemporalReprojection Tests - Variance Clipping
# =============================================================================


class TestSSRTemporalReprojectionVarianceClipping:
    """Tests for variance clipping methods."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        config = SSRTemporalConfig(
            use_variance_clipping=True,
            use_ycocg_space=True,
            variance_gamma=1.0,
        )
        return SSRTemporalReprojection(config=config)

    def test_neighborhood_stats_empty(self, reprojection):
        """Test stats with empty sample list."""
        mean, std = reprojection.compute_neighborhood_stats([])
        assert mean == (0.0, 0.0, 0.0)
        assert std == (0.0, 0.0, 0.0)

    def test_neighborhood_stats_single_sample(self, reprojection):
        """Test stats with single sample."""
        samples = [(0.5, 0.5, 0.5)]
        mean, std = reprojection.compute_neighborhood_stats(samples)

        assert mean == (0.5, 0.5, 0.5)
        assert std == (0.0, 0.0, 0.0)

    def test_neighborhood_stats_uniform_samples(self, reprojection):
        """Test stats with identical samples."""
        samples = [(0.5, 0.5, 0.5)] * 9
        mean, std = reprojection.compute_neighborhood_stats(samples)

        assert mean == (0.5, 0.5, 0.5)
        assert std == (0.0, 0.0, 0.0)

    def test_neighborhood_stats_varied_samples(self, reprojection):
        """Test stats with varied samples."""
        samples = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        mean, std = reprojection.compute_neighborhood_stats(samples)

        assert pytest.approx(mean[0], rel=1e-6) == 0.5
        assert std[0] > 0.0

    def test_variance_clip_disabled(self, reprojection):
        """Test variance clipping when disabled."""
        reprojection.config.use_variance_clipping = False

        history = (10.0, 10.0, 10.0)  # Way outside any reasonable range
        samples = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1)]

        result, was_clipped = reprojection.variance_clip(history, samples)

        assert result == history
        assert not was_clipped

    def test_variance_clip_within_range(self, reprojection):
        """Test history within neighborhood range is not clipped."""
        samples = [
            (0.4, 0.4, 0.4),
            (0.5, 0.5, 0.5),
            (0.6, 0.6, 0.6),
        ]
        history = (0.5, 0.5, 0.5)

        result, was_clipped = reprojection.variance_clip(history, samples)

        # Should not be clipped much
        assert pytest.approx(result[0], abs=0.1) == history[0]

    def test_variance_clip_outside_range(self, reprojection):
        """Test history outside neighborhood is clipped."""
        samples = [
            (0.0, 0.0, 0.0),
            (0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2),
        ]
        history = (1.0, 1.0, 1.0)  # Way outside

        result, was_clipped = reprojection.variance_clip(history, samples)

        # Should be clipped toward neighborhood
        assert result[0] < history[0]
        assert was_clipped

    def test_variance_clip_empty_samples(self, reprojection):
        """Test variance clip with no samples."""
        history = (0.5, 0.5, 0.5)

        result, was_clipped = reprojection.variance_clip(history, [])

        assert result == history
        assert not was_clipped


# =============================================================================
# SSRTemporalReprojection Tests - Reprojection
# =============================================================================


class TestSSRTemporalReprojectionUV:
    """Tests for UV reprojection methods."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        return SSRTemporalReprojection()

    def test_reproject_uv_zero_velocity(self, reprojection):
        """Test reprojection with zero velocity."""
        uv = (0.5, 0.5)
        velocity = (0.0, 0.0)

        result = reprojection.reproject_uv(uv, velocity)

        assert result == (0.5, 0.5)

    def test_reproject_uv_with_velocity(self, reprojection):
        """Test reprojection subtracts velocity."""
        uv = (0.5, 0.5)
        velocity = (0.1, 0.2)

        result = reprojection.reproject_uv(uv, velocity)

        assert pytest.approx(result[0], rel=1e-6) == 0.4
        assert pytest.approx(result[1], rel=1e-6) == 0.3

    def test_is_uv_valid_in_range(self, reprojection):
        """Test UV validation for valid coordinates."""
        assert reprojection.is_uv_valid((0.0, 0.0))
        assert reprojection.is_uv_valid((1.0, 1.0))
        assert reprojection.is_uv_valid((0.5, 0.5))

    def test_is_uv_valid_out_of_range(self, reprojection):
        """Test UV validation rejects out-of-range."""
        assert not reprojection.is_uv_valid((-0.1, 0.5))
        assert not reprojection.is_uv_valid((0.5, 1.1))
        assert not reprojection.is_uv_valid((1.5, 0.5))


# =============================================================================
# SSRTemporalReprojection Tests - Blending
# =============================================================================


class TestSSRTemporalReprojectionBlending:
    """Tests for temporal blending methods."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        config = SSRTemporalConfig(
            quality=TemporalQuality.HIGH,
            motion_weight_scale=1.0,
            luminance_weight=True,
        )
        return SSRTemporalReprojection(config=config)

    def test_compute_blend_weight_no_disocclusion(self, reprojection):
        """Test blend weight with perfect history."""
        weight = reprojection.compute_blend_weight(
            disocclusion_weight=1.0,
            velocity=(0.0, 0.0),
            history_confidence=1.0,
        )

        # Should be close to history_weight (0.93 for HIGH)
        assert 0.8 <= weight <= 0.95

    def test_compute_blend_weight_disoccluded(self, reprojection):
        """Test blend weight with disocclusion."""
        weight = reprojection.compute_blend_weight(
            disocclusion_weight=0.0,
            velocity=(0.0, 0.0),
            history_confidence=1.0,
        )

        assert weight == 0.0

    def test_compute_blend_weight_high_velocity(self, reprojection):
        """Test blend weight reduced by high velocity."""
        weight_slow = reprojection.compute_blend_weight(
            disocclusion_weight=1.0,
            velocity=(0.0, 0.0),
            history_confidence=1.0,
        )

        weight_fast = reprojection.compute_blend_weight(
            disocclusion_weight=1.0,
            velocity=(0.5, 0.5),
            history_confidence=1.0,
        )

        assert weight_fast < weight_slow

    def test_blend_samples_zero_weight(self, reprojection):
        """Test blend with zero history weight."""
        current = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0, depth=10.0)
        history = TemporalSample(
            color=(0.0, 1.0, 0.0),
            alpha=0.5,
            depth=10.0,
            confidence=0.9,
            frame_count=5,
        )

        result = reprojection.blend_samples(current, history, 0.0)

        assert result.color == current.color
        assert result.frame_count == 1

    def test_blend_samples_invalid_history(self, reprojection):
        """Test blend with invalid history sample."""
        current = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0, depth=10.0)
        history = TemporalSample(
            color=(0.0, 1.0, 0.0),
            confidence=0.0,  # Invalid
            depth=0.0,
        )

        result = reprojection.blend_samples(current, history, 0.9)

        # Should use current only due to invalid history
        assert result.color == current.color

    def test_blend_samples_valid_history(self, reprojection):
        """Test blend with valid history."""
        current = TemporalSample(color=(1.0, 0.0, 0.0), alpha=1.0, depth=10.0)
        history = TemporalSample(
            color=(0.0, 1.0, 0.0),
            alpha=0.5,
            depth=10.0,
            confidence=0.9,
            frame_count=5,
        )

        result = reprojection.blend_samples(current, history, 0.5)

        # Blended color
        assert pytest.approx(result.color[0], rel=1e-6) == 0.5
        assert pytest.approx(result.color[1], rel=1e-6) == 0.5
        assert result.frame_count > history.frame_count


# =============================================================================
# SSRTemporalReprojection Tests - Anti-Flicker
# =============================================================================


class TestSSRTemporalReprojectionAntiFlicker:
    """Tests for anti-flicker filtering."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        return SSRTemporalReprojection(config=SSRTemporalConfig(anti_flicker=True))

    def test_anti_flicker_disabled(self, reprojection):
        """Test anti-flicker pass-through when disabled."""
        reprojection.config.anti_flicker = False

        current = (0.0, 0.0, 0.0)
        history = (1.0, 1.0, 1.0)
        blended = (0.5, 0.5, 0.5)

        result = reprojection.apply_anti_flicker(current, history, blended)

        assert result == blended

    def test_anti_flicker_no_change_needed(self, reprojection):
        """Test anti-flicker with similar luminances."""
        current = (0.5, 0.5, 0.5)
        history = (0.5, 0.5, 0.5)
        blended = (0.5, 0.5, 0.5)

        result = reprojection.apply_anti_flicker(current, history, blended)

        # Should be essentially unchanged
        assert pytest.approx(result[0], abs=0.01) == blended[0]

    def test_anti_flicker_biases_toward_history(self, reprojection):
        """Test anti-flicker biases toward history on flicker."""
        current = (0.0, 0.0, 0.0)  # Dark
        history = (1.0, 1.0, 1.0)  # Bright
        blended = (0.1, 0.1, 0.1)  # Mostly current

        result = reprojection.apply_anti_flicker(current, history, blended)

        # Should be slightly brighter (toward history)
        assert result[0] > blended[0]


# =============================================================================
# SSRTemporalReprojection Tests - Full Pipeline
# =============================================================================


class TestSSRTemporalReprojectionPipeline:
    """Tests for full pixel processing pipeline."""

    @pytest.fixture
    def reprojection(self):
        """Create reprojection instance."""
        config = SSRTemporalConfig(
            quality=TemporalQuality.HIGH,
            disocclusion_mode=DisocclusionMode.COMBINED,
            use_variance_clipping=True,
            anti_flicker=True,
        )
        rp = SSRTemporalReprojection(config=config)
        rp.setup(1920, 1080)
        return rp

    def test_process_pixel_no_history(self, reprojection):
        """Test processing pixel with no valid history."""
        result = reprojection.process_pixel(
            uv=(0.5, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
            history_color=(0.0, 1.0, 0.0),
            history_alpha=0.5,
            history_depth=0.0,  # Invalid
            history_normal=(0.0, 1.0, 0.0),
            history_confidence=0.0,  # No confidence
            history_frame_count=0,
            neighborhood_samples=[(1.0, 0.0, 0.0)],
        )

        # Should use current only
        assert result.color == (1.0, 0.0, 0.0)
        assert result.frame_count == 1

    def test_process_pixel_uv_out_of_bounds(self, reprojection):
        """Test processing with reprojected UV out of bounds."""
        result = reprojection.process_pixel(
            uv=(0.05, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.1, 0.0),  # Would put history UV at -0.05
            history_color=(0.0, 1.0, 0.0),
            history_alpha=0.5,
            history_depth=10.0,
            history_normal=(0.0, 1.0, 0.0),
            history_confidence=0.9,
            history_frame_count=5,
            neighborhood_samples=[(1.0, 0.0, 0.0)],
        )

        # Should use current only
        assert result.color == (1.0, 0.0, 0.0)

    def test_process_pixel_disoccluded(self, reprojection):
        """Test processing with disocclusion detection."""
        result = reprojection.process_pixel(
            uv=(0.5, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
            history_color=(0.0, 1.0, 0.0),
            history_alpha=0.5,
            history_depth=50.0,  # Very different depth
            history_normal=(0.0, 1.0, 0.0),
            history_confidence=0.9,
            history_frame_count=5,
            neighborhood_samples=[(1.0, 0.0, 0.0)],
        )

        # Should reject history due to depth difference
        assert result.color == (1.0, 0.0, 0.0)

    def test_process_pixel_valid_history(self, reprojection):
        """Test processing with valid history accumulation."""
        result = reprojection.process_pixel(
            uv=(0.5, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
            history_color=(0.0, 0.0, 1.0),
            history_alpha=0.5,
            history_depth=10.0,  # Same depth
            history_normal=(0.0, 1.0, 0.0),  # Same normal
            history_confidence=0.9,
            history_frame_count=5,
            neighborhood_samples=[(0.5, 0.0, 0.5)] * 9,
        )

        # Should blend with history
        assert result.color[0] < 1.0  # Some history contribution
        assert result.frame_count > 1
        assert result.confidence > 0.0

    def test_process_pixel_high_velocity_reduces_history(self, reprojection):
        """Test high velocity reduces history contribution."""
        # Low velocity
        result_slow = reprojection.process_pixel(
            uv=(0.5, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
            history_color=(0.0, 0.0, 1.0),
            history_alpha=0.5,
            history_depth=10.0,
            history_normal=(0.0, 1.0, 0.0),
            history_confidence=0.9,
            history_frame_count=5,
            neighborhood_samples=[(0.5, 0.0, 0.5)] * 9,
        )

        # High velocity
        result_fast = reprojection.process_pixel(
            uv=(0.5, 0.5),
            current_color=(1.0, 0.0, 0.0),
            current_alpha=1.0,
            current_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            velocity=(0.5, 0.5),  # Fast motion
            history_color=(0.0, 0.0, 1.0),
            history_alpha=0.5,
            history_depth=10.0,
            history_normal=(0.0, 1.0, 0.0),
            history_confidence=0.9,
            history_frame_count=5,
            neighborhood_samples=[(0.5, 0.0, 0.5)] * 9,
        )

        # Fast motion should have more current (red) contribution
        assert result_fast.color[0] >= result_slow.color[0]


# =============================================================================
# SSRTemporalReprojection Tests - Execution
# =============================================================================


class TestSSRTemporalReprojectionExecution:
    """Tests for execute method and shader uniforms."""

    @pytest.fixture
    def reprojection(self):
        """Create initialized reprojection instance."""
        rp = SSRTemporalReprojection()
        rp.setup(1920, 1080)
        return rp

    def test_execute_not_initialized_raises(self):
        """Test execute raises if not initialized."""
        reprojection = SSRTemporalReprojection()

        with pytest.raises(RuntimeError, match="not initialized"):
            reprojection.execute(None, None, None, None, None)

    def test_execute_swaps_buffers(self, reprojection):
        """Test execute swaps ping-pong buffers."""
        initial_index = reprojection.buffers.read_index

        reprojection.execute(None, None, None, None, None)

        assert reprojection.buffers.read_index != initial_index

    def test_execute_increments_frame_index(self, reprojection):
        """Test execute increments frame index."""
        initial = reprojection.frame_index

        reprojection.execute(None, None, None, None, None)

        assert reprojection.frame_index == initial + 1

    def test_execute_resets_stats(self, reprojection):
        """Test execute resets statistics."""
        reprojection.stats.total_pixels = 100

        reprojection.execute(None, None, None, None, None)

        assert reprojection.stats.total_pixels == 0

    def test_get_shader_uniforms(self, reprojection):
        """Test shader uniform generation."""
        uniforms = reprojection.get_shader_uniforms()

        assert "history_weight" in uniforms
        assert "variance_gamma" in uniforms
        assert "depth_threshold" in uniforms
        assert "normal_threshold" in uniforms
        assert "velocity_threshold" in uniforms
        assert "disocclusion_mode" in uniforms
        assert "frame_index" in uniforms
        assert "resolution" in uniforms

        assert uniforms["resolution"] == (1920, 1080)

    def test_shader_uniforms_match_config(self, reprojection):
        """Test shader uniforms match configuration."""
        config = SSRTemporalConfig(
            depth_threshold=0.05,
            normal_threshold=0.8,
            velocity_threshold=0.03,
            disocclusion_mode=DisocclusionMode.DEPTH_ONLY,
        )
        reprojection.config = config

        uniforms = reprojection.get_shader_uniforms()

        assert uniforms["depth_threshold"] == 0.05
        assert uniforms["normal_threshold"] == 0.8
        assert uniforms["velocity_threshold"] == 0.03
        assert uniforms["disocclusion_mode"] == int(DisocclusionMode.DEPTH_ONLY)


# =============================================================================
# Convergence Tests
# =============================================================================


class TestSSRTemporalReprojectionConvergence:
    """Tests for temporal convergence behavior."""

    def test_convergence_after_8_frames(self):
        """Test that system converges after 8 frames (acceptance criteria)."""
        config = SSRTemporalConfig(quality=TemporalQuality.HIGH)
        reprojection = SSRTemporalReprojection(config=config)
        reprojection.setup(1920, 1080)

        # Simulate 8 frame swaps
        for _ in range(8):
            reprojection.execute(None, None, None, None, None)

        assert reprojection.is_converged

    def test_convergence_progress_increments(self):
        """Test convergence progress increases each frame."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)

        progress_values = []
        for _ in range(10):
            progress_values.append(reprojection.convergence_progress)
            reprojection.execute(None, None, None, None, None)

        # Progress should be monotonically increasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]

    def test_invalidate_resets_convergence(self):
        """Test invalidate_history resets convergence."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)

        # Build up convergence
        for _ in range(10):
            reprojection.execute(None, None, None, None, None)

        assert reprojection.is_converged

        reprojection.invalidate_history()

        assert not reprojection.is_converged
        assert reprojection.convergence_progress == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestSSRTemporalReprojectionIntegration:
    """Integration tests for full system behavior."""

    def test_full_workflow(self):
        """Test complete temporal reprojection workflow."""
        # Create and configure
        config = SSRTemporalConfig(
            quality=TemporalQuality.HIGH,
            disocclusion_mode=DisocclusionMode.COMBINED,
            use_variance_clipping=True,
            anti_flicker=True,
        )
        reprojection = SSRTemporalReprojection(config=config)

        # Setup
        reprojection.setup(1920, 1080)
        assert reprojection.is_initialized

        # Initial state
        assert reprojection.frame_index == 0
        assert not reprojection.is_converged

        # Simulate frames
        for frame in range(12):
            reprojection.execute(None, None, None, None, None)

        # Verify state
        assert reprojection.frame_index == 12
        assert reprojection.is_converged
        assert reprojection.buffers.frame_count == 12

    def test_resolution_change_workflow(self):
        """Test handling resolution change mid-stream."""
        reprojection = SSRTemporalReprojection()

        # Start at one resolution
        reprojection.setup(1920, 1080)
        for _ in range(10):
            reprojection.execute(None, None, None, None, None)

        assert reprojection.is_converged

        # Change resolution
        reprojection.setup(1280, 720)

        # History should be invalidated
        assert not reprojection.is_converged
        assert reprojection.width == 1280
        assert reprojection.height == 720

    def test_config_change_preserves_history(self):
        """Test config change does not invalidate history."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)

        for _ in range(10):
            reprojection.execute(None, None, None, None, None)

        assert reprojection.is_converged

        # Change config
        reprojection.config = SSRTemporalConfig(quality=TemporalQuality.LOW)

        # History should still be valid
        assert reprojection.buffers.frame_count > 0

    def test_destroy_cleanup(self):
        """Test destroy properly cleans up."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1920, 1080)

        reprojection.destroy()

        assert not reprojection.is_initialized
        assert reprojection.buffers.buffer_a.texture is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestSSRTemporalReprojectionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_resolution(self):
        """Test with minimal resolution."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(1, 1)

        assert reprojection.is_initialized
        assert reprojection.width == 1
        assert reprojection.height == 1

    def test_large_resolution(self):
        """Test with large resolution."""
        reprojection = SSRTemporalReprojection()
        reprojection.setup(7680, 4320)  # 8K

        assert reprojection.is_initialized
        assert reprojection.width == 7680

    def test_extreme_depth_values(self):
        """Test with extreme depth values."""
        reprojection = SSRTemporalReprojection()

        # Very small depth
        weight = reprojection.compute_depth_rejection(0.001, 0.001)
        assert weight == 1.0

        # Very large depth
        weight = reprojection.compute_depth_rejection(10000.0, 10000.0)
        assert weight == 1.0

    def test_extreme_velocity_values(self):
        """Test with extreme velocity values."""
        reprojection = SSRTemporalReprojection()

        # Huge velocity
        weight = reprojection.compute_velocity_rejection((100.0, 100.0))
        assert weight == 0.0

    def test_normalized_normal_vectors(self):
        """Test normal rejection with non-unit normals."""
        reprojection = SSRTemporalReprojection()

        # Non-unit normals (should still work)
        n1 = (0.0, 2.0, 0.0)  # Scaled up
        n2 = (0.0, 2.0, 0.0)

        # Dot product will be 4, but comparison works
        weight = reprojection.compute_normal_rejection(n1, n2)
        assert weight > 0.0
