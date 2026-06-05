"""
Tests for Temporal Denoiser with Variance-Guided Accumulation

Comprehensive tests for the temporal denoising system:
- TemporalQuality enum values and behavior
- TemporalTarget enum values
- ClampingMode and DisocclusionMode enums
- YCoCgConverter color space conversions
- VarianceEstimate data class
- VarianceGuided weight calculator
- HistoryEntry and HistoryTracker
- NeighbourhoodClamper (AABB, variance, YCoCg)
- Reprojector and ReprojectionResult
- EMABlender temporal blending
- TemporalDenoiseConfig validation
- TemporalBuffer and TemporalBufferSet
- TemporalGBuffer validation
- TemporalDenoiseStats calculations
- TemporalDenoiser lifecycle and denoising
- Factory functions for signal-specific denoisers
"""

import math
import pytest
from unittest.mock import MagicMock, PropertyMock

from engine.rendering.denoise.temporal_denoiser import (
    # Core Denoiser
    TemporalDenoiser,
    # Configuration
    TemporalDenoiseConfig,
    TemporalQuality,
    TemporalTarget,
    ClampingMode,
    DisocclusionMode,
    QualityPreset,
    QUALITY_PRESETS,
    # Components
    VarianceGuided,
    VarianceEstimate,
    HistoryTracker,
    HistoryEntry,
    NeighbourhoodClamper,
    Reprojector,
    ReprojectionResult,
    EMABlender,
    # Color Space
    YCoCgConverter,
    # Buffers
    TemporalBuffer,
    TemporalBufferSet,
    TemporalGBuffer,
    # Statistics
    TemporalDenoiseStats,
    # Constants
    MIN_HISTORY_LENGTH,
    MAX_HISTORY_LENGTH,
    DEFAULT_CONVERGENCE_FRAMES,
    DEFAULT_VARIANCE_GAMMA,
    DEFAULT_EMA_ALPHA,
    EPSILON,
    # Factory Functions
    create_gi_temporal_denoiser,
    create_reflection_temporal_denoiser,
    create_shadow_temporal_denoiser,
    create_fast_temporal_denoiser,
    create_quality_temporal_denoiser,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_device():
    """Create a mock RHI device."""
    device = MagicMock()

    def create_texture_impl(desc):
        texture = MagicMock()
        texture.desc = desc
        texture.is_valid.return_value = True
        return texture

    device.create_texture.side_effect = create_texture_impl
    return device


@pytest.fixture
def mock_texture():
    """Create a mock texture with valid state."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_output_texture():
    """Create a mock output texture."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_g_buffer():
    """Create a mock TemporalGBuffer with valid textures."""
    depth = MagicMock()
    depth.is_valid.return_value = True

    normal = MagicMock()
    normal.is_valid.return_value = True

    velocity = MagicMock()
    velocity.is_valid.return_value = True

    albedo = MagicMock()
    albedo.is_valid.return_value = True

    return TemporalGBuffer(
        depth=depth, normal=normal, velocity=velocity, albedo=albedo
    )


# =============================================================================
# TemporalQuality Tests
# =============================================================================


class TestTemporalQuality:
    """Test TemporalQuality enum."""

    def test_quality_low_value(self):
        """Test LOW quality value."""
        assert TemporalQuality.LOW == 1

    def test_quality_medium_value(self):
        """Test MEDIUM quality value."""
        assert TemporalQuality.MEDIUM == 2

    def test_quality_high_value(self):
        """Test HIGH quality value."""
        assert TemporalQuality.HIGH == 3

    def test_quality_ultra_value(self):
        """Test ULTRA quality value."""
        assert TemporalQuality.ULTRA == 4

    def test_quality_is_int_enum(self):
        """Test that quality values can be used as integers."""
        assert int(TemporalQuality.LOW) == 1
        assert int(TemporalQuality.MEDIUM) == 2

    def test_quality_comparison(self):
        """Test quality level comparisons."""
        assert TemporalQuality.LOW < TemporalQuality.MEDIUM
        assert TemporalQuality.MEDIUM < TemporalQuality.HIGH
        assert TemporalQuality.HIGH < TemporalQuality.ULTRA

    def test_all_qualities_have_presets(self):
        """Test all quality levels have preset parameters."""
        for quality in TemporalQuality:
            assert quality in QUALITY_PRESETS


# =============================================================================
# TemporalTarget Tests
# =============================================================================


class TestTemporalTarget:
    """Test TemporalTarget enum."""

    def test_target_gi_exists(self):
        """Test GI target exists."""
        assert TemporalTarget.GI is not None

    def test_target_reflections_exists(self):
        """Test REFLECTIONS target exists."""
        assert TemporalTarget.REFLECTIONS is not None

    def test_target_shadows_exists(self):
        """Test SHADOWS target exists."""
        assert TemporalTarget.SHADOWS is not None

    def test_target_ao_exists(self):
        """Test AO target exists."""
        assert TemporalTarget.AO is not None

    def test_target_combined_exists(self):
        """Test COMBINED target exists."""
        assert TemporalTarget.COMBINED is not None

    def test_target_custom_exists(self):
        """Test CUSTOM target exists."""
        assert TemporalTarget.CUSTOM is not None

    def test_all_targets_unique(self):
        """Test all targets are unique."""
        targets = list(TemporalTarget)
        assert len(targets) == len(set(targets))


# =============================================================================
# ClampingMode Tests
# =============================================================================


class TestClampingMode:
    """Test ClampingMode enum."""

    def test_mode_none_value(self):
        """Test NONE mode value."""
        assert ClampingMode.NONE == 0

    def test_mode_aabb_value(self):
        """Test AABB mode value."""
        assert ClampingMode.AABB == 1

    def test_mode_variance_value(self):
        """Test VARIANCE mode value."""
        assert ClampingMode.VARIANCE == 2

    def test_mode_ycocg_aabb_value(self):
        """Test YCOCG_AABB mode value."""
        assert ClampingMode.YCOCG_AABB == 3

    def test_mode_ycocg_variance_value(self):
        """Test YCOCG_VARIANCE mode value."""
        assert ClampingMode.YCOCG_VARIANCE == 4


# =============================================================================
# DisocclusionMode Tests
# =============================================================================


class TestDisocclusionMode:
    """Test DisocclusionMode enum."""

    def test_mode_depth_only(self):
        """Test DEPTH_ONLY mode."""
        assert DisocclusionMode.DEPTH_ONLY == 0

    def test_mode_normal_only(self):
        """Test NORMAL_ONLY mode."""
        assert DisocclusionMode.NORMAL_ONLY == 1

    def test_mode_velocity(self):
        """Test VELOCITY mode."""
        assert DisocclusionMode.VELOCITY == 2

    def test_mode_combined(self):
        """Test COMBINED mode."""
        assert DisocclusionMode.COMBINED == 3

    def test_mode_adaptive(self):
        """Test ADAPTIVE mode."""
        assert DisocclusionMode.ADAPTIVE == 4


# =============================================================================
# YCoCgConverter Tests
# =============================================================================


class TestYCoCgConverter:
    """Test YCoCg color space conversion."""

    def test_rgb_to_ycocg_white(self):
        """Test white RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 1.0, 1.0)
        assert abs(y - 1.0) < EPSILON
        assert abs(co) < EPSILON
        assert abs(cg) < EPSILON

    def test_rgb_to_ycocg_black(self):
        """Test black RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 0.0, 0.0)
        assert abs(y) < EPSILON
        assert abs(co) < EPSILON
        assert abs(cg) < EPSILON

    def test_rgb_to_ycocg_red(self):
        """Test red RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 0.0, 0.0)
        assert abs(y - 0.25) < EPSILON
        assert abs(co - 0.5) < EPSILON
        assert abs(cg - (-0.25)) < EPSILON

    def test_ycocg_to_rgb_roundtrip(self):
        """Test RGB -> YCoCg -> RGB roundtrip."""
        original = (0.5, 0.7, 0.3)
        ycocg = YCoCgConverter.rgb_to_ycocg(*original)
        result = YCoCgConverter.ycocg_to_rgb(*ycocg)

        assert abs(result[0] - original[0]) < EPSILON
        assert abs(result[1] - original[1]) < EPSILON
        assert abs(result[2] - original[2]) < EPSILON

    def test_luminance_extraction(self):
        """Test luminance extraction."""
        r, g, b = 0.6, 0.8, 0.4
        y, _, _ = YCoCgConverter.rgb_to_ycocg(r, g, b)
        lum = YCoCgConverter.luminance(r, g, b)
        assert abs(y - lum) < EPSILON


# =============================================================================
# VarianceEstimate Tests
# =============================================================================


class TestVarianceEstimate:
    """Test VarianceEstimate data class."""

    def test_default_values(self):
        """Test default variance estimate values."""
        v = VarianceEstimate()
        assert v.mean == (0.0, 0.0, 0.0)
        assert v.variance == (0.0, 0.0, 0.0)
        assert v.sample_count == 0

    def test_is_valid_empty(self):
        """Test invalid when sample_count is 0."""
        v = VarianceEstimate()
        assert not v.is_valid()

    def test_is_valid_with_samples(self):
        """Test valid when sample_count > 0."""
        v = VarianceEstimate(sample_count=5)
        assert v.is_valid()

    def test_variance_magnitude(self):
        """Test variance magnitude calculation."""
        v = VarianceEstimate(variance=(0.1, 0.2, 0.3))
        assert abs(v.get_variance_magnitude() - 0.6) < EPSILON

    def test_std_dev_calculation(self):
        """Test standard deviation calculation."""
        v = VarianceEstimate(variance=(0.04, 0.09, 0.16))
        std = v.get_std_dev()
        assert abs(std[0] - 0.2) < EPSILON
        assert abs(std[1] - 0.3) < EPSILON
        assert abs(std[2] - 0.4) < EPSILON


# =============================================================================
# VarianceGuided Tests
# =============================================================================


class TestVarianceGuided:
    """Test variance-guided weight calculator."""

    def test_default_initialization(self):
        """Test default initialization."""
        vg = VarianceGuided()
        assert vg.base_weight == 0.9
        assert vg.variance_scale == 4.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        vg = VarianceGuided(base_weight=0.8, variance_scale=2.0)
        assert vg.base_weight == 0.8
        assert vg.variance_scale == 2.0

    def test_invalid_base_weight(self):
        """Test invalid base_weight raises error."""
        with pytest.raises(ValueError):
            VarianceGuided(base_weight=1.5)

    def test_invalid_variance_scale(self):
        """Test invalid variance_scale raises error."""
        with pytest.raises(ValueError):
            VarianceGuided(variance_scale=-1.0)

    def test_weight_zero_variance(self):
        """Test weight calculation with zero variance."""
        vg = VarianceGuided(base_weight=0.9)
        weight = vg.calculate_weight(0.0)
        assert abs(weight - 0.9) < EPSILON

    def test_weight_high_variance(self):
        """Test weight decreases with high variance."""
        vg = VarianceGuided()
        w_low = vg.calculate_weight(0.01)
        w_high = vg.calculate_weight(0.5)
        assert w_high < w_low

    def test_weight_clamped(self):
        """Test weight is clamped to valid range."""
        vg = VarianceGuided(min_weight=0.5, max_weight=0.95)
        weight = vg.calculate_weight(100.0)  # Very high variance
        assert weight >= 0.5
        assert weight <= 0.95

    def test_shader_params(self):
        """Test shader parameter export."""
        vg = VarianceGuided()
        params = vg.get_shader_params()
        assert "base_weight" in params
        assert "variance_scale" in params


# =============================================================================
# HistoryEntry Tests
# =============================================================================


class TestHistoryEntry:
    """Test HistoryEntry data class."""

    def test_default_values(self):
        """Test default entry values."""
        e = HistoryEntry()
        assert e.frame_count == 0
        assert e.confidence == 0.0
        assert not e.is_valid()

    def test_is_valid_with_frames(self):
        """Test valid when frame_count > 0."""
        e = HistoryEntry(frame_count=5)
        assert e.is_valid()

    def test_reset(self):
        """Test reset clears all values."""
        e = HistoryEntry(frame_count=10, confidence=0.8)
        e.reset()
        assert e.frame_count == 0
        assert e.confidence == 0.0


# =============================================================================
# HistoryTracker Tests
# =============================================================================


class TestHistoryTracker:
    """Test HistoryTracker class."""

    def test_default_initialization(self):
        """Test default initialization."""
        tracker = HistoryTracker()
        assert tracker.max_history_length == MAX_HISTORY_LENGTH
        assert tracker.target_convergence == DEFAULT_CONVERGENCE_FRAMES

    def test_custom_initialization(self):
        """Test custom initialization."""
        tracker = HistoryTracker(max_history_length=32, target_convergence=8)
        assert tracker.max_history_length == 32
        assert tracker.target_convergence == 8

    def test_invalid_max_history(self):
        """Test invalid max_history_length raises error."""
        with pytest.raises(ValueError):
            HistoryTracker(max_history_length=0)
        with pytest.raises(ValueError):
            HistoryTracker(max_history_length=100)  # > 64

    def test_get_entry_creates_new(self):
        """Test get_entry creates new entry if needed."""
        tracker = HistoryTracker()
        entry = tracker.get_entry(10, 20)
        assert entry.frame_count == 0

    def test_update_entry_increments(self):
        """Test update_entry increments frame count."""
        tracker = HistoryTracker()
        tracker.update_entry(0, 0, valid=True)
        entry = tracker.get_entry(0, 0)
        assert entry.frame_count == 1

    def test_update_entry_max_limit(self):
        """Test frame count is clamped to max."""
        tracker = HistoryTracker(max_history_length=10)
        for _ in range(20):
            tracker.update_entry(0, 0, valid=True)
        entry = tracker.get_entry(0, 0)
        assert entry.frame_count == 10

    def test_update_entry_rejection_resets(self):
        """Test rejection resets frame count."""
        tracker = HistoryTracker()
        for _ in range(10):
            tracker.update_entry(0, 0, valid=True)
        tracker.update_entry(0, 0, valid=False)
        entry = tracker.get_entry(0, 0)
        assert entry.frame_count == 1

    def test_advance_frame(self):
        """Test frame advancement."""
        tracker = HistoryTracker()
        assert tracker.current_frame == 0
        tracker.advance_frame()
        assert tracker.current_frame == 1

    def test_reset_clears_history(self):
        """Test reset clears all history."""
        tracker = HistoryTracker()
        tracker.update_entry(0, 0, valid=True)
        tracker.reset()
        entry = tracker.get_entry(0, 0)
        assert entry.frame_count == 0

    def test_is_converged(self):
        """Test convergence check."""
        tracker = HistoryTracker(target_convergence=5)
        for _ in range(5):
            tracker.update_entry(0, 0, valid=True)
        assert tracker.is_converged(0, 0)

    def test_get_stats(self):
        """Test statistics retrieval."""
        tracker = HistoryTracker()
        stats = tracker.get_stats()
        assert "total_pixels" in stats
        assert "avg_history_length" in stats


# =============================================================================
# NeighbourhoodClamper Tests
# =============================================================================


class TestNeighbourhoodClamper:
    """Test NeighbourhoodClamper class."""

    def test_default_initialization(self):
        """Test default initialization."""
        clamper = NeighbourhoodClamper()
        assert clamper.mode == ClampingMode.YCOCG_VARIANCE
        assert clamper.gamma == DEFAULT_VARIANCE_GAMMA

    def test_custom_initialization(self):
        """Test custom initialization."""
        clamper = NeighbourhoodClamper(
            mode=ClampingMode.AABB, gamma=1.5, neighbourhood_size=5
        )
        assert clamper.mode == ClampingMode.AABB
        assert clamper.gamma == 1.5
        assert clamper.neighbourhood_size == 5

    def test_invalid_neighbourhood_size(self):
        """Test invalid neighbourhood_size raises error."""
        with pytest.raises(ValueError):
            NeighbourhoodClamper(neighbourhood_size=4)

    def test_invalid_gamma(self):
        """Test invalid gamma raises error."""
        with pytest.raises(ValueError):
            NeighbourhoodClamper(gamma=0.0)

    def test_clamp_aabb(self):
        """Test AABB clamping."""
        clamper = NeighbourhoodClamper()
        result = clamper.clamp_aabb(
            history=(1.5, 0.5, -0.5),
            min_color=(0.0, 0.0, 0.0),
            max_color=(1.0, 1.0, 1.0),
        )
        assert result[0] == 1.0  # Clamped
        assert result[1] == 0.5  # Unchanged
        assert result[2] == 0.0  # Clamped

    def test_clamp_variance(self):
        """Test variance-based clamping."""
        clamper = NeighbourhoodClamper(gamma=1.0)
        result = clamper.clamp_variance(
            history=(0.8, 0.5, 0.2),
            mean=(0.5, 0.5, 0.5),
            std_dev=(0.1, 0.1, 0.1),
        )
        assert result[0] == 0.6  # Clamped to mean + gamma * std
        assert result[1] == 0.5  # Within bounds
        assert result[2] == 0.4  # Clamped to mean - gamma * std

    def test_clamp_none_mode(self):
        """Test NONE mode returns original."""
        clamper = NeighbourhoodClamper(mode=ClampingMode.NONE)
        variance = VarianceEstimate(sample_count=1)
        result = clamper.clamp((0.5, 0.6, 0.7), variance)
        assert result == (0.5, 0.6, 0.7)

    def test_compute_clamp_amount_no_change(self):
        """Test clamp amount when no change."""
        clamper = NeighbourhoodClamper()
        amount = clamper.compute_clamp_amount(
            original=(0.5, 0.5, 0.5),
            clamped=(0.5, 0.5, 0.5),
        )
        assert amount == 0.0

    def test_compute_clamp_amount_with_change(self):
        """Test clamp amount when changed."""
        clamper = NeighbourhoodClamper()
        amount = clamper.compute_clamp_amount(
            original=(1.0, 0.0, 0.0),
            clamped=(0.5, 0.0, 0.0),
        )
        assert amount > 0.0


# =============================================================================
# Reprojector Tests
# =============================================================================


class TestReprojector:
    """Test Reprojector class."""

    def test_default_initialization(self):
        """Test default initialization."""
        r = Reprojector()
        assert r.disocclusion_mode == DisocclusionMode.COMBINED
        assert r.depth_threshold == 0.01

    def test_custom_initialization(self):
        """Test custom initialization."""
        r = Reprojector(
            disocclusion_mode=DisocclusionMode.DEPTH_ONLY,
            depth_threshold=0.02,
        )
        assert r.disocclusion_mode == DisocclusionMode.DEPTH_ONLY
        assert r.depth_threshold == 0.02

    def test_invalid_depth_threshold(self):
        """Test invalid depth_threshold raises error."""
        with pytest.raises(ValueError):
            Reprojector(depth_threshold=0.0)

    def test_invalid_normal_threshold(self):
        """Test invalid normal_threshold raises error."""
        with pytest.raises(ValueError):
            Reprojector(normal_threshold=1.5)

    def test_reproject_inside_frame(self):
        """Test reprojection inside frame."""
        r = Reprojector()
        result = r.reproject(uv=(0.5, 0.5), velocity=(0.1, 0.1))
        assert result.valid
        assert result.uv == (0.4, 0.4)

    def test_reproject_outside_frame(self):
        """Test reprojection outside frame."""
        r = Reprojector()
        result = r.reproject(uv=(0.1, 0.1), velocity=(0.2, 0.2))
        assert not result.valid

    def test_check_depth_disocclusion_similar(self):
        """Test depth check with similar depths."""
        r = Reprojector(depth_threshold=0.01)
        # Depth diff = 0.02, relative diff = 0.02/10.0 = 0.002 < 0.01 threshold
        weight = r.check_depth_disocclusion(10.0, 10.02)
        assert weight > 0.5

    def test_check_depth_disocclusion_different(self):
        """Test depth check with different depths."""
        r = Reprojector(depth_threshold=0.01)
        weight = r.check_depth_disocclusion(10.0, 11.0)
        assert weight == 0.0

    def test_check_normal_disocclusion_similar(self):
        """Test normal check with similar normals."""
        r = Reprojector(normal_threshold=0.9)
        weight = r.check_normal_disocclusion(
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 0.99, 0.1),
        )
        assert weight > 0.0

    def test_check_normal_disocclusion_different(self):
        """Test normal check with different normals."""
        r = Reprojector(normal_threshold=0.9)
        weight = r.check_normal_disocclusion(
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(1.0, 0.0, 0.0),
        )
        assert weight == 0.0

    def test_check_velocity_disocclusion_slow(self):
        """Test velocity check with slow motion."""
        r = Reprojector(velocity_threshold=0.05)
        weight = r.check_velocity_disocclusion(velocity=(0.01, 0.01))
        assert weight == 1.0

    def test_check_velocity_disocclusion_fast(self):
        """Test velocity check with fast motion."""
        r = Reprojector(velocity_threshold=0.05)
        weight = r.check_velocity_disocclusion(velocity=(0.2, 0.2))
        assert weight < 1.0

    def test_check_disocclusion_combined(self):
        """Test combined disocclusion check."""
        r = Reprojector(disocclusion_mode=DisocclusionMode.COMBINED)
        weight = r.check_disocclusion(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
        )
        assert weight > 0.9


# =============================================================================
# EMABlender Tests
# =============================================================================


class TestEMABlender:
    """Test EMABlender class."""

    def test_default_initialization(self):
        """Test default initialization."""
        b = EMABlender()
        assert b.alpha == DEFAULT_EMA_ALPHA

    def test_custom_initialization(self):
        """Test custom initialization."""
        b = EMABlender(alpha=0.2, variance_modulation=False)
        assert b.alpha == 0.2
        assert not b.variance_modulation

    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with pytest.raises(ValueError):
            EMABlender(alpha=0.0)
        with pytest.raises(ValueError):
            EMABlender(alpha=1.0)

    def test_blend_equal_weights(self):
        """Test blend with alpha=0.5."""
        b = EMABlender(alpha=0.5)
        result = b.blend(
            current=(1.0, 0.0, 0.0),
            history=(0.0, 1.0, 0.0),
        )
        assert abs(result[0] - 0.5) < EPSILON
        assert abs(result[1] - 0.5) < EPSILON

    def test_blend_favor_history(self):
        """Test blend with low alpha (favor history)."""
        b = EMABlender(alpha=0.1)
        result = b.blend(
            current=(1.0, 1.0, 1.0),
            history=(0.0, 0.0, 0.0),
        )
        assert result[0] < 0.2

    def test_compute_alpha_with_variance(self):
        """Test alpha increases with variance."""
        b = EMABlender(alpha=0.1, variance_modulation=True)
        alpha_low = b.compute_alpha(variance=0.01)
        alpha_high = b.compute_alpha(variance=0.5)
        assert alpha_high > alpha_low

    def test_compute_alpha_with_motion(self):
        """Test alpha increases with motion."""
        b = EMABlender(alpha=0.1, motion_modulation=True)
        alpha_static = b.compute_alpha(velocity_magnitude=0.0)
        alpha_moving = b.compute_alpha(velocity_magnitude=0.1)
        assert alpha_moving > alpha_static

    def test_blend_with_variance(self):
        """Test blend_with_variance returns alpha used."""
        b = EMABlender(alpha=0.1)
        result, alpha_used = b.blend_with_variance(
            current=(1.0, 1.0, 1.0),
            history=(0.0, 0.0, 0.0),
            variance=0.01,
        )
        assert 0.0 < alpha_used < 1.0


# =============================================================================
# TemporalDenoiseConfig Tests
# =============================================================================


class TestTemporalDenoiseConfig:
    """Test TemporalDenoiseConfig data class."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = TemporalDenoiseConfig()
        assert config.quality == TemporalQuality.HIGH
        assert config.target == TemporalTarget.GI

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = TemporalDenoiseConfig(
            quality=TemporalQuality.ULTRA,
            target=TemporalTarget.REFLECTIONS,
            ema_alpha=0.15,
        )
        assert config.quality == TemporalQuality.ULTRA
        assert config.target == TemporalTarget.REFLECTIONS
        assert config.ema_alpha == 0.15

    def test_invalid_neighbourhood_size(self):
        """Test invalid neighbourhood_size raises error."""
        with pytest.raises(ValueError):
            TemporalDenoiseConfig(neighbourhood_size=4)

    def test_invalid_max_history(self):
        """Test invalid max_history raises error."""
        with pytest.raises(ValueError):
            TemporalDenoiseConfig(max_history=100)

    def test_invalid_ema_alpha(self):
        """Test invalid ema_alpha raises error."""
        with pytest.raises(ValueError):
            TemporalDenoiseConfig(ema_alpha=1.5)

    def test_get_preset(self):
        """Test get_preset returns correct preset."""
        config = TemporalDenoiseConfig(quality=TemporalQuality.HIGH)
        preset = config.get_preset()
        assert preset == QUALITY_PRESETS[TemporalQuality.HIGH]

    def test_get_ema_alpha_from_preset(self):
        """Test get_ema_alpha uses preset if not set."""
        config = TemporalDenoiseConfig(quality=TemporalQuality.HIGH)
        assert config.get_ema_alpha() == QUALITY_PRESETS[TemporalQuality.HIGH].ema_alpha

    def test_get_ema_alpha_from_config(self):
        """Test get_ema_alpha uses config value if set."""
        config = TemporalDenoiseConfig(ema_alpha=0.15)
        assert config.get_ema_alpha() == 0.15

    def test_get_target_convergence(self):
        """Test get_target_convergence returns correct value."""
        config = TemporalDenoiseConfig(target_convergence=20)
        assert config.get_target_convergence() == 20


# =============================================================================
# TemporalBuffer Tests
# =============================================================================


class TestTemporalBuffer:
    """Test TemporalBuffer data class."""

    def test_default_values(self):
        """Test default buffer values."""
        buf = TemporalBuffer()
        assert buf.texture is None
        assert buf.width == 0
        assert not buf.valid

    def test_is_allocated_empty(self):
        """Test is_allocated when empty."""
        buf = TemporalBuffer()
        assert not buf.is_allocated()

    def test_is_allocated_with_texture(self):
        """Test is_allocated with texture."""
        buf = TemporalBuffer(texture=MagicMock())
        assert buf.is_allocated()

    def test_matches_dimensions(self):
        """Test dimension matching."""
        buf = TemporalBuffer(width=100, height=200)
        assert buf.matches_dimensions(100, 200)
        assert not buf.matches_dimensions(200, 100)

    def test_invalidate(self):
        """Test invalidate clears validity."""
        buf = TemporalBuffer(valid=True, frame_index=10)
        buf.invalidate()
        assert not buf.valid
        assert buf.frame_index == -1

    def test_mark_written(self):
        """Test mark_written sets validity."""
        buf = TemporalBuffer()
        buf.mark_written(5)
        assert buf.valid
        assert buf.frame_index == 5


# =============================================================================
# TemporalBufferSet Tests
# =============================================================================


class TestTemporalBufferSet:
    """Test TemporalBufferSet data class."""

    def test_default_initialization(self):
        """Test default initialization."""
        buffers = TemporalBufferSet()
        assert buffers.read_index == 0
        assert buffers.frame_count == 0

    def test_history_buffer_initial(self):
        """Test history buffer is buffer_a initially."""
        buffers = TemporalBufferSet()
        assert buffers.history_buffer is buffers.buffer_a

    def test_current_buffer_initial(self):
        """Test current buffer is buffer_b initially."""
        buffers = TemporalBufferSet()
        assert buffers.current_buffer is buffers.buffer_b

    def test_swap(self):
        """Test swap swaps read index."""
        buffers = TemporalBufferSet()
        buffers.swap()
        assert buffers.read_index == 1
        assert buffers.history_buffer is buffers.buffer_b
        assert buffers.current_buffer is buffers.buffer_a

    def test_swap_increments_frame_count(self):
        """Test swap increments frame count."""
        buffers = TemporalBufferSet()
        buffers.swap()
        assert buffers.frame_count == 1

    def test_invalidate_all(self):
        """Test invalidate_all clears both buffers."""
        buffers = TemporalBufferSet()
        buffers.buffer_a.valid = True
        buffers.buffer_b.valid = True
        buffers.frame_count = 10
        buffers.invalidate_all()
        assert not buffers.buffer_a.valid
        assert not buffers.buffer_b.valid
        assert buffers.frame_count == 0

    def test_convergence_progress(self):
        """Test convergence progress calculation."""
        buffers = TemporalBufferSet()
        buffers.frame_count = 8
        assert buffers.get_convergence_progress(16) == 0.5

    def test_is_converged(self):
        """Test convergence check."""
        buffers = TemporalBufferSet()
        buffers.frame_count = 16
        assert buffers.is_converged(16)
        assert not buffers.is_converged(32)


# =============================================================================
# TemporalGBuffer Tests
# =============================================================================


class TestTemporalGBuffer:
    """Test TemporalGBuffer data class."""

    def test_is_valid_complete(self, mock_g_buffer):
        """Test is_valid with complete g-buffer."""
        assert mock_g_buffer.is_valid()

    def test_is_valid_missing_velocity(self):
        """Test is_valid fails without velocity."""
        depth = MagicMock()
        depth.is_valid.return_value = True
        normal = MagicMock()
        normal.is_valid.return_value = True

        g_buffer = TemporalGBuffer(depth=depth, normal=normal, velocity=None)
        assert not g_buffer.is_valid()

    def test_has_albedo(self, mock_g_buffer):
        """Test has_albedo with albedo present."""
        assert mock_g_buffer.has_albedo()


# =============================================================================
# TemporalDenoiseStats Tests
# =============================================================================


class TestTemporalDenoiseStats:
    """Test TemporalDenoiseStats data class."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = TemporalDenoiseStats()
        assert stats.total_pixels == 0
        assert stats.avg_alpha == 0.0

    def test_rejection_rate(self):
        """Test rejection rate calculation."""
        stats = TemporalDenoiseStats(total_pixels=100, rejected_pixels=25)
        assert stats.rejection_rate == 0.25

    def test_convergence_rate(self):
        """Test convergence rate calculation."""
        stats = TemporalDenoiseStats(total_pixels=100, converged_pixels=80)
        assert stats.convergence_rate == 0.8

    def test_history_usage_rate(self):
        """Test history usage rate calculation."""
        stats = TemporalDenoiseStats(total_pixels=100, valid_history_pixels=90)
        assert stats.history_usage_rate == 0.9

    def test_clamp_rate(self):
        """Test clamp rate calculation."""
        stats = TemporalDenoiseStats(total_pixels=100, clamped_pixels=30)
        assert stats.clamp_rate == 0.3

    def test_reset(self):
        """Test reset clears all values."""
        stats = TemporalDenoiseStats(
            total_pixels=100, rejected_pixels=10, avg_alpha=0.5
        )
        stats.reset()
        assert stats.total_pixels == 0
        assert stats.rejected_pixels == 0
        assert stats.avg_alpha == 0.0


# =============================================================================
# TemporalDenoiser Tests
# =============================================================================


class TestTemporalDenoiser:
    """Test TemporalDenoiser class."""

    def test_default_initialization(self):
        """Test default initialization."""
        denoiser = TemporalDenoiser()
        assert not denoiser.is_initialized
        assert denoiser.frame_index == 0

    def test_initialization_with_device(self, mock_device):
        """Test initialization with device."""
        denoiser = TemporalDenoiser(device=mock_device)
        assert denoiser.device is mock_device

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = TemporalDenoiseConfig(quality=TemporalQuality.ULTRA)
        denoiser = TemporalDenoiser(config=config)
        assert denoiser.config.quality == TemporalQuality.ULTRA

    def test_setup(self):
        """Test setup initializes buffers."""
        denoiser = TemporalDenoiser()
        denoiser.setup(1920, 1080)
        assert denoiser.is_initialized
        assert denoiser.width == 1920
        assert denoiser.height == 1080

    def test_setup_invalid_dimensions(self):
        """Test setup with invalid dimensions raises error."""
        denoiser = TemporalDenoiser()
        with pytest.raises(ValueError):
            denoiser.setup(0, 100)

    def test_invalidate_history(self):
        """Test invalidate_history clears state."""
        denoiser = TemporalDenoiser()
        denoiser.setup(100, 100)
        denoiser.invalidate_history()
        assert not denoiser.buffers.buffer_a.valid
        assert not denoiser.buffers.buffer_b.valid

    def test_convergence_progress_initial(self):
        """Test initial convergence progress."""
        denoiser = TemporalDenoiser()
        assert denoiser.convergence_progress == 0.0

    def test_is_converged_initial(self):
        """Test initial convergence state."""
        denoiser = TemporalDenoiser()
        assert not denoiser.is_converged

    def test_config_setter_updates_components(self):
        """Test setting config updates components."""
        denoiser = TemporalDenoiser()
        new_config = TemporalDenoiseConfig(
            ema_alpha=0.2,
            clamping_mode=ClampingMode.AABB,
        )
        denoiser.config = new_config
        assert denoiser.blender.alpha == 0.2
        assert denoiser.clamper.mode == ClampingMode.AABB

    def test_denoise_validates_inputs(self, mock_texture, mock_g_buffer, mock_output_texture):
        """Test denoise validates inputs."""
        denoiser = TemporalDenoiser()
        denoiser.setup(1920, 1080)

        # Should not raise
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
        assert stats.total_pixels == 1920 * 1080

    def test_denoise_invalid_input(self, mock_g_buffer, mock_output_texture):
        """Test denoise raises on invalid input."""
        denoiser = TemporalDenoiser()
        denoiser.setup(1920, 1080)

        invalid_texture = MagicMock()
        invalid_texture.is_valid.return_value = False

        with pytest.raises(ValueError):
            denoiser.denoise(invalid_texture, mock_g_buffer, mock_output_texture)

    def test_denoise_invalid_g_buffer(self, mock_texture, mock_output_texture):
        """Test denoise raises on invalid g_buffer."""
        denoiser = TemporalDenoiser()
        denoiser.setup(1920, 1080)

        depth = MagicMock()
        depth.is_valid.return_value = True
        # Missing velocity
        invalid_g_buffer = TemporalGBuffer(depth=depth, normal=depth, velocity=None)

        with pytest.raises(ValueError):
            denoiser.denoise(mock_texture, invalid_g_buffer, mock_output_texture)

    def test_denoise_increments_frame(self, mock_texture, mock_g_buffer, mock_output_texture):
        """Test denoise increments frame index."""
        denoiser = TemporalDenoiser()
        denoiser.setup(1920, 1080)

        denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
        assert denoiser.frame_index == 1

        denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
        assert denoiser.frame_index == 2

    def test_destroy_clears_state(self):
        """Test destroy releases resources."""
        denoiser = TemporalDenoiser()
        denoiser.setup(100, 100)
        denoiser.destroy()
        assert not denoiser.is_initialized


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for creating denoisers."""

    def test_create_gi_temporal_denoiser(self):
        """Test GI denoiser creation."""
        denoiser = create_gi_temporal_denoiser()
        assert denoiser.config.target == TemporalTarget.GI
        assert denoiser.config.quality == TemporalQuality.HIGH

    def test_create_reflection_temporal_denoiser(self):
        """Test reflection denoiser creation."""
        denoiser = create_reflection_temporal_denoiser()
        assert denoiser.config.target == TemporalTarget.REFLECTIONS
        assert denoiser.config.quality == TemporalQuality.MEDIUM

    def test_create_shadow_temporal_denoiser(self):
        """Test shadow denoiser creation."""
        denoiser = create_shadow_temporal_denoiser()
        assert denoiser.config.target == TemporalTarget.SHADOWS

    def test_create_fast_temporal_denoiser(self):
        """Test fast denoiser creation."""
        denoiser = create_fast_temporal_denoiser()
        assert denoiser.config.quality == TemporalQuality.LOW
        assert denoiser.config.clamping_mode == ClampingMode.AABB

    def test_create_quality_temporal_denoiser(self):
        """Test quality denoiser creation."""
        denoiser = create_quality_temporal_denoiser()
        assert denoiser.config.quality == TemporalQuality.ULTRA
        assert denoiser.config.max_history == 64


# =============================================================================
# Quality Preset Tests
# =============================================================================


class TestQualityPresets:
    """Test quality preset parameters."""

    def test_low_preset_fast(self):
        """Test LOW preset is fast."""
        preset = QUALITY_PRESETS[TemporalQuality.LOW]
        assert preset.ema_alpha >= 0.15  # Fast blending
        assert preset.target_frames <= 10

    def test_ultra_preset_stable(self):
        """Test ULTRA preset is stable."""
        preset = QUALITY_PRESETS[TemporalQuality.ULTRA]
        assert preset.ema_alpha <= 0.1  # Slow blending
        assert preset.target_frames >= 20

    def test_preset_ema_ordering(self):
        """Test EMA alpha decreases with quality."""
        low = QUALITY_PRESETS[TemporalQuality.LOW].ema_alpha
        medium = QUALITY_PRESETS[TemporalQuality.MEDIUM].ema_alpha
        high = QUALITY_PRESETS[TemporalQuality.HIGH].ema_alpha
        ultra = QUALITY_PRESETS[TemporalQuality.ULTRA].ema_alpha

        assert low >= medium >= high >= ultra

    def test_preset_target_frames_ordering(self):
        """Test target_frames increases with quality."""
        low = QUALITY_PRESETS[TemporalQuality.LOW].target_frames
        medium = QUALITY_PRESETS[TemporalQuality.MEDIUM].target_frames
        high = QUALITY_PRESETS[TemporalQuality.HIGH].target_frames
        ultra = QUALITY_PRESETS[TemporalQuality.ULTRA].target_frames

        assert low <= medium <= high <= ultra


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for temporal denoising workflow."""

    def test_full_workflow(self, mock_device, mock_texture, mock_g_buffer, mock_output_texture):
        """Test complete denoising workflow."""
        # Create denoiser
        config = TemporalDenoiseConfig(
            quality=TemporalQuality.HIGH,
            target=TemporalTarget.GI,
        )
        denoiser = TemporalDenoiser(device=mock_device, config=config)

        # Setup
        denoiser.setup(1920, 1080)
        assert denoiser.is_initialized

        # Run multiple frames
        for i in range(16):
            stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
            assert stats.total_pixels == 1920 * 1080

        # Check convergence
        assert denoiser.is_converged
        assert denoiser.frame_index == 16

    def test_resize_invalidates_history(self, mock_device):
        """Test resize invalidates history."""
        denoiser = TemporalDenoiser(device=mock_device)
        denoiser.setup(1920, 1080)

        # Simulate some history
        denoiser.buffers.frame_count = 10

        # Resize
        denoiser.setup(1280, 720)

        # History should be invalidated
        assert denoiser.buffers.frame_count == 0

    def test_camera_cut_invalidation(self, mock_device, mock_texture, mock_g_buffer, mock_output_texture):
        """Test camera cut invalidates history."""
        denoiser = TemporalDenoiser(device=mock_device)
        denoiser.setup(1920, 1080)

        # Run some frames
        for _ in range(8):
            denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        # Simulate camera cut
        denoiser.invalidate_history()

        # Progress should reset
        assert denoiser.convergence_progress == 0.0
