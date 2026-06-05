"""Tests for SSGI temporal accumulation (T-GIR-P3.2).

Tests cover:
    - TemporalConfig validation and adaptive blend
    - ClampMode and ResetCondition enums
    - NeighbourhoodStats computation
    - Clamping functions (AABB, variance, clip)
    - Reprojection utilities
    - TemporalHistory buffer management
    - TemporalUniforms GPU struct
    - SSGITemporalAccumulator lifecycle
    - Temporal blend computation
    - Luminance calculations
    - VarianceTracker convergence
    - Quality presets
"""

import math
import pytest
from engine.rendering.gi.ssgi_temporal import (
    # Constants
    DEFAULT_BLEND_FACTOR,
    DEFAULT_VELOCITY_THRESHOLD,
    DEFAULT_CLAMP_EXPAND,
    DEFAULT_MIN_VARIANCE,
    MAX_VARIANCE_FRAMES,
    WORKGROUP_SIZE,
    # Enums
    ClampMode,
    ResetCondition,
    # Config
    TemporalConfig,
    # Statistics
    NeighbourhoodStats,
    compute_neighbourhood_stats,
    # Clamping
    clamp_color,
    clip_color_to_aabb,
    # Reprojection
    reproject_uv,
    velocity_magnitude,
    is_valid_uv,
    # History
    TemporalHistory,
    TemporalUniforms,
    # Main class
    SSGITemporalAccumulator,
    # Blend computation
    compute_temporal_blend,
    compute_luminance,
    compute_luminance_weight,
    # Variance
    VarianceTracker,
    # Presets
    create_low_quality_config,
    create_medium_quality_config,
    create_high_quality_config,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_blend_factor_in_valid_range(self) -> None:
        """Default blend factor should be in [0, 1]."""
        assert 0.0 <= DEFAULT_BLEND_FACTOR <= 1.0

    def test_default_blend_factor_value(self) -> None:
        """Default blend factor should be 0.9."""
        assert DEFAULT_BLEND_FACTOR == 0.9

    def test_default_velocity_threshold_positive(self) -> None:
        """Default velocity threshold should be positive."""
        assert DEFAULT_VELOCITY_THRESHOLD > 0.0

    def test_default_clamp_expand_minimum(self) -> None:
        """Default clamp expand should be >= 1.0."""
        assert DEFAULT_CLAMP_EXPAND >= 1.0

    def test_default_min_variance_small(self) -> None:
        """Default min variance should be small positive."""
        assert 0.0 < DEFAULT_MIN_VARIANCE < 0.01

    def test_max_variance_frames_reasonable(self) -> None:
        """Max variance frames should be reasonable for tracking."""
        assert 8 <= MAX_VARIANCE_FRAMES <= 64

    def test_workgroup_size(self) -> None:
        """Workgroup size should be 8 for compute shader."""
        assert WORKGROUP_SIZE == 8


# =============================================================================
# Enum Tests
# =============================================================================


class TestClampMode:
    """Tests for ClampMode enumeration."""

    def test_enum_values(self) -> None:
        """Enum should have expected values."""
        assert ClampMode.NONE.value == 0
        assert ClampMode.AABB.value == 1
        assert ClampMode.VARIANCE.value == 2
        assert ClampMode.CLIPPED.value == 3

    def test_enum_count(self) -> None:
        """Should have exactly 4 clamp modes."""
        assert len(ClampMode) == 4

    def test_enum_is_int(self) -> None:
        """Enum values should be integers."""
        for mode in ClampMode:
            assert isinstance(mode.value, int)


class TestResetCondition:
    """Tests for ResetCondition enumeration."""

    def test_enum_values(self) -> None:
        """Enum should have expected values."""
        assert ResetCondition.NEVER.value == 0
        assert ResetCondition.VELOCITY.value == 1
        assert ResetCondition.DEPTH_DISCONTINUITY.value == 2
        assert ResetCondition.BOTH.value == 3

    def test_enum_count(self) -> None:
        """Should have exactly 4 reset conditions."""
        assert len(ResetCondition) == 4


# =============================================================================
# TemporalConfig Tests
# =============================================================================


class TestTemporalConfig:
    """Tests for TemporalConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have expected values."""
        config = TemporalConfig()
        assert config.enabled is True
        assert config.blend_factor == DEFAULT_BLEND_FACTOR
        assert config.velocity_threshold == DEFAULT_VELOCITY_THRESHOLD
        assert config.clamp_mode == ClampMode.VARIANCE
        assert config.clamp_expand == DEFAULT_CLAMP_EXPAND
        assert config.reset_condition == ResetCondition.VELOCITY

    def test_valid_blend_factor_zero(self) -> None:
        """Blend factor of 0.0 should be valid."""
        config = TemporalConfig(blend_factor=0.0)
        assert config.blend_factor == 0.0

    def test_valid_blend_factor_one(self) -> None:
        """Blend factor of 1.0 should be valid."""
        config = TemporalConfig(blend_factor=1.0)
        assert config.blend_factor == 1.0

    def test_invalid_blend_factor_negative(self) -> None:
        """Negative blend factor should raise ValueError."""
        with pytest.raises(ValueError, match="blend_factor"):
            TemporalConfig(blend_factor=-0.1)

    def test_invalid_blend_factor_over_one(self) -> None:
        """Blend factor > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="blend_factor"):
            TemporalConfig(blend_factor=1.1)

    def test_invalid_velocity_threshold_negative(self) -> None:
        """Negative velocity threshold should raise ValueError."""
        with pytest.raises(ValueError, match="velocity_threshold"):
            TemporalConfig(velocity_threshold=-0.01)

    def test_valid_velocity_threshold_zero(self) -> None:
        """Zero velocity threshold should be valid."""
        config = TemporalConfig(velocity_threshold=0.0)
        assert config.velocity_threshold == 0.0

    def test_invalid_clamp_expand_below_one(self) -> None:
        """Clamp expand < 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="clamp_expand"):
            TemporalConfig(clamp_expand=0.5)

    def test_invalid_depth_threshold_zero(self) -> None:
        """Zero depth threshold should raise ValueError."""
        with pytest.raises(ValueError, match="depth_threshold"):
            TemporalConfig(depth_threshold=0.0)

    def test_invalid_depth_threshold_negative(self) -> None:
        """Negative depth threshold should raise ValueError."""
        with pytest.raises(ValueError, match="depth_threshold"):
            TemporalConfig(depth_threshold=-0.01)

    def test_invalid_variance_gamma_zero(self) -> None:
        """Zero variance gamma should raise ValueError."""
        with pytest.raises(ValueError, match="variance_gamma"):
            TemporalConfig(variance_gamma=0.0)

    def test_invalid_variance_gamma_over_one(self) -> None:
        """Variance gamma > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="variance_gamma"):
            TemporalConfig(variance_gamma=1.5)

    def test_invalid_min_history_weight_negative(self) -> None:
        """Negative min history weight should raise ValueError."""
        with pytest.raises(ValueError, match="min_history_weight"):
            TemporalConfig(min_history_weight=-0.1)

    def test_invalid_max_history_weight_over_one(self) -> None:
        """Max history weight > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_history_weight"):
            TemporalConfig(max_history_weight=1.5)

    def test_invalid_min_greater_than_max(self) -> None:
        """Min weight > max weight should raise ValueError."""
        with pytest.raises(ValueError, match="min_history_weight"):
            TemporalConfig(min_history_weight=0.9, max_history_weight=0.5)


class TestTemporalConfigAdaptiveBlend:
    """Tests for TemporalConfig.get_adaptive_blend."""

    def test_zero_velocity_returns_max(self) -> None:
        """Zero velocity should return max weight."""
        config = TemporalConfig(min_history_weight=0.5, max_history_weight=0.98)
        assert config.get_adaptive_blend(0.0) == 0.98

    def test_high_velocity_returns_min(self) -> None:
        """High velocity should return near min weight."""
        config = TemporalConfig(
            velocity_threshold=0.01,
            min_history_weight=0.5,
            max_history_weight=0.98,
        )
        result = config.get_adaptive_blend(0.02)  # 2x threshold
        assert abs(result - 0.5) < 0.01

    def test_threshold_velocity_interpolates(self) -> None:
        """Velocity at threshold should interpolate."""
        config = TemporalConfig(
            velocity_threshold=0.01,
            min_history_weight=0.5,
            max_history_weight=1.0,
        )
        result = config.get_adaptive_blend(0.005)  # 50% of threshold
        assert 0.5 < result < 1.0

    def test_negative_velocity_treated_as_zero(self) -> None:
        """Negative velocity should be treated as zero."""
        config = TemporalConfig(max_history_weight=0.95)
        assert config.get_adaptive_blend(-1.0) == 0.95


class TestTemporalConfigShouldReset:
    """Tests for TemporalConfig.should_reset."""

    def test_never_reset(self) -> None:
        """ResetCondition.NEVER should never reset."""
        config = TemporalConfig(reset_condition=ResetCondition.NEVER)
        assert config.should_reset(100.0, 100.0) is False

    def test_velocity_reset_triggers(self) -> None:
        """Velocity condition should trigger on high velocity."""
        config = TemporalConfig(
            reset_condition=ResetCondition.VELOCITY,
            velocity_threshold=0.01,
        )
        assert config.should_reset(0.02, 0.0) is True

    def test_velocity_reset_not_triggered(self) -> None:
        """Velocity condition should not trigger below threshold."""
        config = TemporalConfig(
            reset_condition=ResetCondition.VELOCITY,
            velocity_threshold=0.01,
        )
        assert config.should_reset(0.005, 0.0) is False

    def test_depth_reset_triggers(self) -> None:
        """Depth condition should trigger on large depth diff."""
        config = TemporalConfig(
            reset_condition=ResetCondition.DEPTH_DISCONTINUITY,
            depth_threshold=0.05,
        )
        assert config.should_reset(0.0, 0.1) is True

    def test_depth_reset_not_triggered(self) -> None:
        """Depth condition should not trigger below threshold."""
        config = TemporalConfig(
            reset_condition=ResetCondition.DEPTH_DISCONTINUITY,
            depth_threshold=0.05,
        )
        assert config.should_reset(0.0, 0.02) is False

    def test_both_condition_velocity(self) -> None:
        """BOTH condition should trigger on velocity."""
        config = TemporalConfig(
            reset_condition=ResetCondition.BOTH,
            velocity_threshold=0.01,
            depth_threshold=0.05,
        )
        assert config.should_reset(0.02, 0.01) is True

    def test_both_condition_depth(self) -> None:
        """BOTH condition should trigger on depth."""
        config = TemporalConfig(
            reset_condition=ResetCondition.BOTH,
            velocity_threshold=0.01,
            depth_threshold=0.05,
        )
        assert config.should_reset(0.001, 0.1) is True

    def test_both_condition_neither(self) -> None:
        """BOTH condition should not trigger when neither exceeds."""
        config = TemporalConfig(
            reset_condition=ResetCondition.BOTH,
            velocity_threshold=0.01,
            depth_threshold=0.05,
        )
        assert config.should_reset(0.005, 0.02) is False


class TestTemporalConfigWithMethods:
    """Tests for TemporalConfig.with_* methods."""

    def test_with_blend_factor(self) -> None:
        """with_blend_factor should create modified copy."""
        original = TemporalConfig(blend_factor=0.9)
        modified = original.with_blend_factor(0.8)
        assert original.blend_factor == 0.9
        assert modified.blend_factor == 0.8
        assert modified.clamp_mode == original.clamp_mode

    def test_with_clamp_mode(self) -> None:
        """with_clamp_mode should create modified copy."""
        original = TemporalConfig(clamp_mode=ClampMode.VARIANCE)
        modified = original.with_clamp_mode(ClampMode.AABB)
        assert original.clamp_mode == ClampMode.VARIANCE
        assert modified.clamp_mode == ClampMode.AABB
        assert modified.blend_factor == original.blend_factor


# =============================================================================
# NeighbourhoodStats Tests
# =============================================================================


class TestNeighbourhoodStats:
    """Tests for NeighbourhoodStats dataclass."""

    def test_default_values(self) -> None:
        """Default stats should be zeros."""
        stats = NeighbourhoodStats()
        assert stats.mean == (0.0, 0.0, 0.0, 0.0)
        assert stats.variance == (0.0, 0.0, 0.0, 0.0)
        assert stats.sample_count == 0

    def test_aabb_min_returns_min_val(self) -> None:
        """aabb_min should return min_val."""
        stats = NeighbourhoodStats(min_val=(0.1, 0.2, 0.3, 0.4))
        assert stats.aabb_min() == (0.1, 0.2, 0.3, 0.4)

    def test_aabb_max_returns_max_val(self) -> None:
        """aabb_max should return max_val."""
        stats = NeighbourhoodStats(max_val=(0.9, 0.8, 0.7, 0.6))
        assert stats.aabb_max() == (0.9, 0.8, 0.7, 0.6)

    def test_variance_min_calculation(self) -> None:
        """variance_min should return mean - expand * sqrt(variance)."""
        stats = NeighbourhoodStats(
            mean=(0.5, 0.5, 0.5, 0.5),
            variance=(0.01, 0.01, 0.01, 0.01),
        )
        result = stats.variance_min(expand=1.0)
        expected = 0.5 - math.sqrt(0.01)
        for i in range(4):
            assert abs(result[i] - expected) < 1e-6

    def test_variance_max_calculation(self) -> None:
        """variance_max should return mean + expand * sqrt(variance)."""
        stats = NeighbourhoodStats(
            mean=(0.5, 0.5, 0.5, 0.5),
            variance=(0.01, 0.01, 0.01, 0.01),
        )
        result = stats.variance_max(expand=1.0)
        expected = 0.5 + math.sqrt(0.01)
        for i in range(4):
            assert abs(result[i] - expected) < 1e-6

    def test_variance_min_with_expand(self) -> None:
        """variance_min should respect expand factor."""
        stats = NeighbourhoodStats(
            mean=(0.5, 0.5, 0.5, 0.5),
            variance=(0.04, 0.04, 0.04, 0.04),
        )
        result = stats.variance_min(expand=2.0)
        expected = 0.5 - 2.0 * math.sqrt(0.04)  # 0.5 - 0.4 = 0.1
        for i in range(4):
            assert abs(result[i] - expected) < 1e-6

    def test_variance_handles_zero_variance(self) -> None:
        """variance_min/max should handle zero variance."""
        stats = NeighbourhoodStats(
            mean=(0.5, 0.5, 0.5, 0.5),
            variance=(0.0, 0.0, 0.0, 0.0),
        )
        assert stats.variance_min() == stats.mean
        assert stats.variance_max() == stats.mean


class TestComputeNeighbourhoodStats:
    """Tests for compute_neighbourhood_stats function."""

    def test_empty_samples(self) -> None:
        """Empty samples should return default stats."""
        stats = compute_neighbourhood_stats([])
        assert stats.sample_count == 0

    def test_single_sample(self) -> None:
        """Single sample stats should equal the sample."""
        sample = (0.5, 0.6, 0.7, 0.8)
        stats = compute_neighbourhood_stats([sample])
        assert stats.sample_count == 1
        assert stats.mean == sample
        assert stats.variance == (0.0, 0.0, 0.0, 0.0)
        assert stats.min_val == sample
        assert stats.max_val == sample

    def test_uniform_samples(self) -> None:
        """Uniform samples should have zero variance."""
        sample = (0.5, 0.5, 0.5, 0.5)
        stats = compute_neighbourhood_stats([sample, sample, sample])
        assert stats.sample_count == 3
        assert stats.mean == sample
        for v in stats.variance:
            assert abs(v) < 1e-6

    def test_two_samples(self) -> None:
        """Two samples should compute correct mean."""
        samples = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)]
        stats = compute_neighbourhood_stats(samples)
        assert stats.sample_count == 2
        for i in range(4):
            assert abs(stats.mean[i] - 0.5) < 1e-6

    def test_min_max_computation(self) -> None:
        """Should correctly compute min and max."""
        samples = [
            (0.1, 0.2, 0.3, 0.4),
            (0.5, 0.6, 0.7, 0.8),
            (0.3, 0.4, 0.5, 0.6),
        ]
        stats = compute_neighbourhood_stats(samples)
        assert stats.min_val == (0.1, 0.2, 0.3, 0.4)
        assert stats.max_val == (0.5, 0.6, 0.7, 0.8)

    def test_variance_computation(self) -> None:
        """Should correctly compute variance."""
        samples = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)]
        stats = compute_neighbourhood_stats(samples)
        # Variance of [0, 1] with mean 0.5 is 0.25
        for i in range(4):
            assert abs(stats.variance[i] - 0.25) < 1e-6


# =============================================================================
# Clamping Function Tests
# =============================================================================


class TestClampColor:
    """Tests for clamp_color function."""

    def test_no_clamp_needed(self) -> None:
        """Color within bounds should not be clamped."""
        color = (0.5, 0.5, 0.5, 0.5)
        min_val = (0.0, 0.0, 0.0, 0.0)
        max_val = (1.0, 1.0, 1.0, 1.0)
        result = clamp_color(color, min_val, max_val)
        assert result == color

    def test_clamp_to_min(self) -> None:
        """Color below min should be clamped up."""
        color = (-0.5, -0.5, -0.5, -0.5)
        min_val = (0.0, 0.0, 0.0, 0.0)
        max_val = (1.0, 1.0, 1.0, 1.0)
        result = clamp_color(color, min_val, max_val)
        assert result == min_val

    def test_clamp_to_max(self) -> None:
        """Color above max should be clamped down."""
        color = (1.5, 1.5, 1.5, 1.5)
        min_val = (0.0, 0.0, 0.0, 0.0)
        max_val = (1.0, 1.0, 1.0, 1.0)
        result = clamp_color(color, min_val, max_val)
        assert result == max_val

    def test_clamp_per_channel(self) -> None:
        """Each channel should be clamped independently."""
        color = (-0.5, 0.5, 1.5, 0.5)
        min_val = (0.0, 0.0, 0.0, 0.0)
        max_val = (1.0, 1.0, 1.0, 1.0)
        result = clamp_color(color, min_val, max_val)
        assert result == (0.0, 0.5, 1.0, 0.5)


class TestClipColorToAABB:
    """Tests for clip_color_to_aabb function."""

    def test_inside_aabb(self) -> None:
        """Color inside AABB should not change."""
        color = (0.5, 0.5, 0.5, 0.5)
        center = (0.5, 0.5, 0.5, 0.5)
        half_extent = (0.5, 0.5, 0.5, 0.5)
        result = clip_color_to_aabb(color, center, half_extent)
        assert result == color

    def test_outside_clips_towards_center(self) -> None:
        """Color outside should be clipped on ray to center."""
        color = (2.0, 0.5, 0.5, 0.5)  # Far right of AABB
        center = (0.5, 0.5, 0.5, 0.5)
        half_extent = (0.5, 0.5, 0.5, 0.5)  # AABB is [0, 1]^4
        result = clip_color_to_aabb(color, center, half_extent)
        # Should be clipped to right edge (1.0, 0.5, 0.5, 0.5)
        assert abs(result[0] - 1.0) < 0.01
        for i in range(1, 4):
            assert abs(result[i] - 0.5) < 0.01

    def test_at_center_returns_center(self) -> None:
        """Color at center should return center."""
        center = (0.5, 0.5, 0.5, 0.5)
        half_extent = (0.5, 0.5, 0.5, 0.5)
        result = clip_color_to_aabb(center, center, half_extent)
        assert result == center


# =============================================================================
# Reprojection Tests
# =============================================================================


class TestReprojectUV:
    """Tests for reproject_uv function."""

    def test_zero_velocity(self) -> None:
        """Zero velocity should return same UV."""
        uv = (0.5, 0.5)
        velocity = (0.0, 0.0)
        result = reproject_uv(uv, velocity)
        assert result == uv

    def test_positive_velocity(self) -> None:
        """Positive velocity should move UV backwards."""
        uv = (0.5, 0.5)
        velocity = (0.1, 0.1)
        result = reproject_uv(uv, velocity)
        assert result == (0.4, 0.4)

    def test_negative_velocity(self) -> None:
        """Negative velocity should move UV forwards."""
        uv = (0.5, 0.5)
        velocity = (-0.1, -0.1)
        result = reproject_uv(uv, velocity)
        assert result == (0.6, 0.6)


class TestVelocityMagnitude:
    """Tests for velocity_magnitude function."""

    def test_zero_velocity(self) -> None:
        """Zero velocity should have zero magnitude."""
        assert velocity_magnitude((0.0, 0.0)) == 0.0

    def test_unit_x(self) -> None:
        """Unit X velocity should have magnitude 1."""
        assert abs(velocity_magnitude((1.0, 0.0)) - 1.0) < 1e-6

    def test_unit_y(self) -> None:
        """Unit Y velocity should have magnitude 1."""
        assert abs(velocity_magnitude((0.0, 1.0)) - 1.0) < 1e-6

    def test_diagonal(self) -> None:
        """Diagonal velocity should have correct magnitude."""
        mag = velocity_magnitude((1.0, 1.0))
        expected = math.sqrt(2.0)
        assert abs(mag - expected) < 1e-6

    def test_small_velocity(self) -> None:
        """Small velocity should compute correctly."""
        mag = velocity_magnitude((0.001, 0.001))
        expected = math.sqrt(0.001**2 + 0.001**2)
        assert abs(mag - expected) < 1e-9


class TestIsValidUV:
    """Tests for is_valid_uv function."""

    def test_center_valid(self) -> None:
        """Center UV should be valid."""
        assert is_valid_uv((0.5, 0.5)) is True

    def test_corners_valid(self) -> None:
        """Corner UVs should be valid."""
        assert is_valid_uv((0.0, 0.0)) is True
        assert is_valid_uv((1.0, 0.0)) is True
        assert is_valid_uv((0.0, 1.0)) is True
        assert is_valid_uv((1.0, 1.0)) is True

    def test_negative_u_invalid(self) -> None:
        """Negative U should be invalid."""
        assert is_valid_uv((-0.01, 0.5)) is False

    def test_negative_v_invalid(self) -> None:
        """Negative V should be invalid."""
        assert is_valid_uv((0.5, -0.01)) is False

    def test_over_one_u_invalid(self) -> None:
        """U > 1 should be invalid."""
        assert is_valid_uv((1.01, 0.5)) is False

    def test_over_one_v_invalid(self) -> None:
        """V > 1 should be invalid."""
        assert is_valid_uv((0.5, 1.01)) is False


# =============================================================================
# TemporalHistory Tests
# =============================================================================


class TestTemporalHistory:
    """Tests for TemporalHistory dataclass."""

    def test_default_state(self) -> None:
        """Default history should be invalid and empty."""
        history = TemporalHistory()
        assert history.width == 0
        assert history.height == 0
        assert history.frame_count == 0
        assert history.is_valid is False

    def test_invalidate_resets_state(self) -> None:
        """invalidate should reset valid state and frame count."""
        history = TemporalHistory(frame_count=10, is_valid=True)
        history.invalidate()
        assert history.is_valid is False
        assert history.frame_count == 0

    def test_swap_buffers_increments_frame(self) -> None:
        """swap_buffers should increment frame count."""
        history = TemporalHistory()
        initial_count = history.frame_count
        history.swap_buffers()
        assert history.frame_count == initial_count + 1

    def test_convergence_ratio_invalid_history(self) -> None:
        """Invalid history should have zero convergence."""
        history = TemporalHistory(is_valid=False)
        assert history.get_convergence_ratio() == 0.0

    def test_convergence_ratio_increases(self) -> None:
        """Convergence ratio should increase with frame count."""
        history = TemporalHistory(is_valid=True, frame_count=0)
        r0 = history.get_convergence_ratio()

        history.frame_count = 8
        r8 = history.get_convergence_ratio()

        history.frame_count = 16
        r16 = history.get_convergence_ratio()

        assert r0 == 0.0
        assert r8 > r0
        assert r16 > r8
        assert r16 < 1.0  # Should approach but not reach 1.0

    def test_convergence_ratio_asymptotic(self) -> None:
        """Convergence ratio should approach 1.0 asymptotically."""
        history = TemporalHistory(is_valid=True, frame_count=100)
        ratio = history.get_convergence_ratio()
        assert 0.99 < ratio < 1.0


# =============================================================================
# TemporalUniforms Tests
# =============================================================================


class TestTemporalUniforms:
    """Tests for TemporalUniforms dataclass."""

    def test_default_values(self) -> None:
        """Default uniforms should match default config."""
        uniforms = TemporalUniforms()
        assert uniforms.blend_factor == DEFAULT_BLEND_FACTOR
        assert uniforms.velocity_threshold == DEFAULT_VELOCITY_THRESHOLD
        assert uniforms.clamp_mode == ClampMode.VARIANCE.value

    def test_from_config(self) -> None:
        """from_config should correctly convert TemporalConfig."""
        config = TemporalConfig(
            blend_factor=0.85,
            clamp_mode=ClampMode.AABB,
            anti_flicker=True,
            luminance_weight=False,
        )
        uniforms = TemporalUniforms.from_config(config, frame_index=42)
        assert uniforms.blend_factor == 0.85
        assert uniforms.clamp_mode == ClampMode.AABB.value
        assert uniforms.frame_index == 42
        # anti_flicker = bit 0, luminance_weight = bit 1
        assert uniforms.flags == 1  # only anti_flicker set

    def test_from_config_both_flags(self) -> None:
        """from_config should set both flags correctly."""
        config = TemporalConfig(anti_flicker=True, luminance_weight=True)
        uniforms = TemporalUniforms.from_config(config, frame_index=0)
        assert uniforms.flags == 3  # both bits set

    def test_from_config_no_flags(self) -> None:
        """from_config should set no flags when both disabled."""
        config = TemporalConfig(anti_flicker=False, luminance_weight=False)
        uniforms = TemporalUniforms.from_config(config, frame_index=0)
        assert uniforms.flags == 0

    def test_to_bytes_size(self) -> None:
        """to_bytes should return 48 bytes."""
        uniforms = TemporalUniforms()
        data = uniforms.to_bytes()
        assert len(data) == 48


# =============================================================================
# SSGITemporalAccumulator Tests
# =============================================================================


class TestSSGITemporalAccumulator:
    """Tests for SSGITemporalAccumulator class."""

    def test_default_construction(self) -> None:
        """Default construction should be valid."""
        accumulator = SSGITemporalAccumulator()
        assert accumulator.device is None
        assert accumulator.is_initialized is False
        assert accumulator.is_history_valid is False
        assert accumulator.frame_index == 0

    def test_config_property(self) -> None:
        """config property should work correctly."""
        accumulator = SSGITemporalAccumulator()
        new_config = TemporalConfig(blend_factor=0.8)
        accumulator.config = new_config
        assert accumulator.config.blend_factor == 0.8

    def test_setup_validates_width(self) -> None:
        """setup should reject invalid width."""
        accumulator = SSGITemporalAccumulator()
        with pytest.raises(ValueError, match="width"):
            accumulator.setup(0, 100)

    def test_setup_validates_height(self) -> None:
        """setup should reject invalid height."""
        accumulator = SSGITemporalAccumulator()
        with pytest.raises(ValueError, match="height"):
            accumulator.setup(100, -1)

    def test_setup_marks_initialized(self) -> None:
        """setup should mark accumulator as initialized."""
        accumulator = SSGITemporalAccumulator()
        accumulator.setup(1920, 1080)
        assert accumulator.is_initialized is True

    def test_reset_history(self) -> None:
        """reset_history should invalidate and reset frame counter."""
        accumulator = SSGITemporalAccumulator()
        accumulator.setup(100, 100)
        accumulator._history.is_valid = True
        accumulator._history.frame_count = 10
        accumulator._frame_index = 50

        accumulator.reset_history()

        assert accumulator.is_history_valid is False
        assert accumulator.frame_index == 0

    def test_get_convergence(self) -> None:
        """get_convergence should delegate to history."""
        accumulator = SSGITemporalAccumulator()
        accumulator._history = TemporalHistory(is_valid=True, frame_count=16)
        convergence = accumulator.get_convergence()
        assert convergence > 0.5

    def test_get_variance_reduction_first_frame(self) -> None:
        """Variance reduction for 0 frames should be 1.0."""
        accumulator = SSGITemporalAccumulator()
        assert accumulator.get_variance_reduction(0) == 1.0

    def test_get_variance_reduction_decreases(self) -> None:
        """Variance reduction should decrease with more frames."""
        accumulator = SSGITemporalAccumulator()
        r1 = accumulator.get_variance_reduction(1)
        r8 = accumulator.get_variance_reduction(8)
        assert r1 > r8

    def test_destroy(self) -> None:
        """destroy should mark as uninitialized."""
        accumulator = SSGITemporalAccumulator()
        accumulator.setup(100, 100)
        accumulator.destroy()
        assert accumulator.is_initialized is False


# =============================================================================
# Temporal Blend Computation Tests
# =============================================================================


class TestComputeTemporalBlend:
    """Tests for compute_temporal_blend function."""

    def test_blend_factor_zero_returns_current(self) -> None:
        """Blend factor 0 should return current frame."""
        current = (1.0, 0.0, 0.0, 1.0)
        history = (0.0, 1.0, 0.0, 1.0)
        result = compute_temporal_blend(current, history, 0.0)
        assert result == current

    def test_blend_factor_one_returns_history(self) -> None:
        """Blend factor 1 should return history."""
        current = (1.0, 0.0, 0.0, 1.0)
        history = (0.0, 1.0, 0.0, 1.0)
        result = compute_temporal_blend(current, history, 1.0)
        assert result == history

    def test_blend_factor_half_averages(self) -> None:
        """Blend factor 0.5 should average current and history."""
        current = (0.0, 0.0, 0.0, 0.0)
        history = (1.0, 1.0, 1.0, 1.0)
        result = compute_temporal_blend(current, history, 0.5)
        for i in range(4):
            assert abs(result[i] - 0.5) < 1e-6

    def test_typical_blend_factor(self) -> None:
        """Typical blend factor (0.9) should weight history heavily."""
        current = (1.0, 1.0, 1.0, 1.0)
        history = (0.0, 0.0, 0.0, 0.0)
        result = compute_temporal_blend(current, history, 0.9)
        for i in range(4):
            # result = 0.0 * 0.9 + 1.0 * 0.1 = 0.1
            assert abs(result[i] - 0.1) < 1e-6


# =============================================================================
# Luminance Tests
# =============================================================================


class TestComputeLuminance:
    """Tests for compute_luminance function."""

    def test_black_luminance(self) -> None:
        """Black should have zero luminance."""
        assert compute_luminance((0.0, 0.0, 0.0, 1.0)) == 0.0

    def test_white_luminance(self) -> None:
        """White should have luminance ~1.0."""
        lum = compute_luminance((1.0, 1.0, 1.0, 1.0))
        # 0.2126 + 0.7152 + 0.0722 = 1.0
        assert abs(lum - 1.0) < 1e-6

    def test_red_luminance(self) -> None:
        """Red should have luminance 0.2126."""
        lum = compute_luminance((1.0, 0.0, 0.0, 1.0))
        assert abs(lum - 0.2126) < 1e-6

    def test_green_luminance(self) -> None:
        """Green should have luminance 0.7152."""
        lum = compute_luminance((0.0, 1.0, 0.0, 1.0))
        assert abs(lum - 0.7152) < 1e-6

    def test_blue_luminance(self) -> None:
        """Blue should have luminance 0.0722."""
        lum = compute_luminance((0.0, 0.0, 1.0, 1.0))
        assert abs(lum - 0.0722) < 1e-6

    def test_alpha_ignored(self) -> None:
        """Alpha should not affect luminance."""
        lum1 = compute_luminance((0.5, 0.5, 0.5, 0.0))
        lum2 = compute_luminance((0.5, 0.5, 0.5, 1.0))
        assert lum1 == lum2


class TestComputeLuminanceWeight:
    """Tests for compute_luminance_weight function."""

    def test_identical_colors(self) -> None:
        """Identical colors should have weight 1.0."""
        color = (0.5, 0.5, 0.5, 1.0)
        weight = compute_luminance_weight(color, color)
        assert abs(weight - 1.0) < 1e-6

    def test_large_difference_reduces_weight(self) -> None:
        """Large luminance difference should reduce weight."""
        current = (1.0, 1.0, 1.0, 1.0)
        history = (0.1, 0.1, 0.1, 1.0)
        weight = compute_luminance_weight(current, history)
        assert weight < 0.5

    def test_zero_history_luminance(self) -> None:
        """Zero history luminance should return 1.0."""
        current = (1.0, 1.0, 1.0, 1.0)
        history = (0.0, 0.0, 0.0, 1.0)
        weight = compute_luminance_weight(current, history)
        assert weight == 1.0

    def test_weight_in_valid_range(self) -> None:
        """Weight should always be in [0, 1]."""
        for _ in range(100):
            import random
            current = tuple(random.random() for _ in range(4))
            history = tuple(random.random() for _ in range(4))
            weight = compute_luminance_weight(current, history)
            assert 0.0 <= weight <= 1.0


# =============================================================================
# VarianceTracker Tests
# =============================================================================


class TestVarianceTracker:
    """Tests for VarianceTracker class."""

    def test_initial_state(self) -> None:
        """Initial tracker should have zero variance."""
        tracker = VarianceTracker()
        assert tracker.mean == 0.0
        assert tracker.variance == 0.0
        assert tracker.sample_count == 0

    def test_first_sample(self) -> None:
        """First sample should set mean directly."""
        tracker = VarianceTracker()
        tracker.update(5.0)
        assert tracker.mean == 5.0
        assert tracker.variance == 0.0
        assert tracker.sample_count == 1

    def test_update_mean(self) -> None:
        """Update should track mean correctly."""
        tracker = VarianceTracker(gamma=0.5)
        tracker.update(0.0)
        tracker.update(1.0)
        # Mean should move towards 1.0
        assert 0.0 < tracker.mean < 1.0

    def test_variance_increases_with_spread(self) -> None:
        """Variance should increase with data spread."""
        tracker = VarianceTracker(gamma=0.5)
        tracker.update(0.0)
        tracker.update(1.0)
        assert tracker.variance > 0.0

    def test_standard_deviation(self) -> None:
        """get_standard_deviation should return sqrt of variance."""
        tracker = VarianceTracker()
        tracker.variance = 0.25
        assert abs(tracker.get_standard_deviation() - 0.5) < 1e-6

    def test_standard_deviation_zero_variance(self) -> None:
        """Standard deviation should be 0 for zero variance."""
        tracker = VarianceTracker()
        tracker.variance = 0.0
        assert tracker.get_standard_deviation() == 0.0

    def test_is_converged_not_enough_samples(self) -> None:
        """Should not be converged with few samples."""
        tracker = VarianceTracker()
        tracker.sample_count = 3
        tracker.variance = 0.001
        assert tracker.is_converged(threshold=0.01) is False

    def test_is_converged_high_variance(self) -> None:
        """Should not be converged with high variance."""
        tracker = VarianceTracker()
        tracker.sample_count = 16
        tracker.variance = 0.1
        assert tracker.is_converged(threshold=0.01) is False

    def test_is_converged_true(self) -> None:
        """Should be converged with enough samples and low variance."""
        tracker = VarianceTracker()
        tracker.sample_count = 16
        tracker.variance = 0.001
        assert tracker.is_converged(threshold=0.01) is True

    def test_reset(self) -> None:
        """reset should clear all state."""
        tracker = VarianceTracker()
        tracker.update(1.0)
        tracker.update(2.0)
        tracker.reset()
        assert tracker.mean == 0.0
        assert tracker.variance == 0.0
        assert tracker.sample_count == 0


# =============================================================================
# Quality Preset Tests
# =============================================================================


class TestQualityPresets:
    """Tests for quality preset functions."""

    def test_low_quality_config(self) -> None:
        """Low quality should prioritize performance."""
        config = create_low_quality_config()
        assert config.enabled is True
        assert config.clamp_mode == ClampMode.AABB  # Fastest clamp
        assert config.anti_flicker is False
        assert config.luminance_weight is False

    def test_medium_quality_config(self) -> None:
        """Medium quality should be balanced."""
        config = create_medium_quality_config()
        assert config.enabled is True
        assert config.clamp_mode == ClampMode.VARIANCE
        assert config.anti_flicker is True
        assert config.luminance_weight is True

    def test_high_quality_config(self) -> None:
        """High quality should prioritize quality."""
        config = create_high_quality_config()
        assert config.enabled is True
        assert config.clamp_mode == ClampMode.CLIPPED  # Most accurate
        assert config.anti_flicker is True
        assert config.luminance_weight is True
        assert config.blend_factor > 0.9  # More history for stability

    def test_preset_blend_factor_ordering(self) -> None:
        """Higher quality should have higher blend factor."""
        low = create_low_quality_config()
        medium = create_medium_quality_config()
        high = create_high_quality_config()
        assert low.blend_factor <= medium.blend_factor <= high.blend_factor

    def test_all_presets_valid(self) -> None:
        """All presets should create valid configs."""
        configs = [
            create_low_quality_config(),
            create_medium_quality_config(),
            create_high_quality_config(),
        ]
        for config in configs:
            assert config.enabled is True
            assert 0.0 <= config.blend_factor <= 1.0


# =============================================================================
# Integration / Scenario Tests
# =============================================================================


class TestTemporalAccumulationScenarios:
    """Integration tests for temporal accumulation scenarios."""

    def test_static_scene_converges(self) -> None:
        """Static scene should converge over multiple frames."""
        tracker = VarianceTracker(gamma=0.1)

        # Simulate static scene with some noise
        import random
        random.seed(42)

        for _ in range(32):
            value = 0.5 + (random.random() - 0.5) * 0.1  # Noisy signal
            tracker.update(value)

        # Should converge to low variance
        assert tracker.variance < 0.01

    def test_variance_reduces_over_8_frames(self) -> None:
        """Variance should reduce over 8+ frames on static scene."""
        accumulator = SSGITemporalAccumulator()
        accumulator.config = create_medium_quality_config()

        reduction_1 = accumulator.get_variance_reduction(1)
        reduction_8 = accumulator.get_variance_reduction(8)

        # Variance reduction should be significant after 8 frames
        assert reduction_8 < reduction_1 * 0.5

    def test_disocclusion_resets_on_high_velocity(self) -> None:
        """High velocity should trigger disocclusion reset."""
        config = TemporalConfig(
            reset_condition=ResetCondition.VELOCITY,
            velocity_threshold=0.01,
        )

        # Normal motion - no reset
        assert config.should_reset(0.005, 0.0) is False

        # Fast motion - reset
        assert config.should_reset(0.02, 0.0) is True

    def test_reprojection_with_velocity(self) -> None:
        """Reprojection should correctly offset UVs by velocity."""
        uv = (0.5, 0.5)
        velocity = (0.05, -0.03)

        reprojected = reproject_uv(uv, velocity)

        # Check reprojected is valid
        assert is_valid_uv(reprojected)

        # Check direction is correct (opposite of velocity)
        assert reprojected[0] < uv[0]  # Moved left due to positive vx
        assert reprojected[1] > uv[1]  # Moved down due to negative vy

    def test_clamping_prevents_ghosting(self) -> None:
        """Clamping should constrain history to neighbourhood bounds."""
        # Simulated neighbourhood with narrow range
        samples = [
            (0.45, 0.45, 0.45, 1.0),
            (0.55, 0.55, 0.55, 1.0),
            (0.50, 0.50, 0.50, 1.0),
        ]
        stats = compute_neighbourhood_stats(samples)

        # Ghost history (outside neighbourhood)
        ghost = (0.9, 0.9, 0.9, 1.0)

        # Clamp to neighbourhood
        clamped = clamp_color(ghost, stats.aabb_min(), stats.aabb_max())

        # Should be within bounds
        assert all(
            stats.aabb_min()[i] <= clamped[i] <= stats.aabb_max()[i]
            for i in range(4)
        )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_velocity_threshold(self) -> None:
        """Config with tiny velocity threshold should work."""
        config = TemporalConfig(velocity_threshold=1e-6)
        assert config.velocity_threshold == 1e-6
        # Should trigger reset on any motion
        assert config.should_reset(0.0001, 0.0) is True

    def test_blend_factor_extremes(self) -> None:
        """Blend factor at 0 and 1 should work correctly."""
        current = (1.0, 1.0, 1.0, 1.0)
        history = (0.0, 0.0, 0.0, 0.0)

        result_0 = compute_temporal_blend(current, history, 0.0)
        assert result_0 == current

        result_1 = compute_temporal_blend(current, history, 1.0)
        assert result_1 == history

    def test_negative_color_values(self) -> None:
        """Negative color values should be handled."""
        samples = [(-0.5, -0.5, -0.5, 1.0), (0.5, 0.5, 0.5, 1.0)]
        stats = compute_neighbourhood_stats(samples)
        assert stats.min_val == (-0.5, -0.5, -0.5, 1.0)

    def test_hdr_color_values(self) -> None:
        """HDR (>1.0) color values should be handled."""
        samples = [(2.0, 2.0, 2.0, 1.0), (10.0, 10.0, 10.0, 1.0)]
        stats = compute_neighbourhood_stats(samples)
        assert stats.max_val == (10.0, 10.0, 10.0, 1.0)

    def test_nan_variance_handled(self) -> None:
        """Negative variance input should not crash."""
        stats = NeighbourhoodStats(variance=(-0.01, -0.01, -0.01, -0.01))
        # Should handle negative variance (clamp to 0 in sqrt)
        result = stats.variance_min()
        # Result should be the mean (sqrt(0) = 0)
        for i in range(4):
            assert math.isfinite(result[i])

    def test_uniforms_bytes_alignment(self) -> None:
        """Uniforms byte representation should be 48 bytes."""
        uniforms = TemporalUniforms()
        data = uniforms.to_bytes()
        assert len(data) == 48
        # Should be parseable
        import struct
        values = struct.unpack("<ffffffff II II", data)
        assert len(values) == 12
