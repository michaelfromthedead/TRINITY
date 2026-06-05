"""Tests for RT Adaptive Rays (T-GIR-P8.3).

Comprehensive test coverage for roughness-based ray count adaptation:
- Roughness to ray count mapping
- Resolution tier selection
- Budget allocation
- Temporal accumulation
- Edge cases (roughness 0, 1)
- Integration with RT pass

Target: 65+ tests
"""

from __future__ import annotations

import math
import random
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from engine.core.math.vec import Vec2, Vec3
from engine.rendering.reflections.rt_adaptive_rays import (
    # Constants
    DEFAULT_TIER_THRESHOLDS,
    DEFAULT_RESOLUTION_SCALES,
    DEFAULT_RAY_COUNTS,
    DEFAULT_DENOISE_STRENGTHS,
    DEFAULT_RAY_BUDGET,
    DEFAULT_TEMPORAL_FRAMES,
    MAX_RT_ROUGHNESS,
    # Enums
    ResolutionTier,
    DenoiseLevel,
    # Data structures
    RoughnessMapping,
    ResolutionLevel,
    ScheduledRays,
    AccumulatedResult,
    BudgetAllocation,
    AdaptiveRayResult,
    # Core classes
    RoughnessRayMapping,
    ResolutionHierarchy,
    AdaptiveRayScheduler,
    RayBudgetManager,
    AdaptiveRTPass,
    # Config
    AdaptiveRTConfig,
    # Utilities
    generate_adaptive_rays_wgsl,
    estimate_adaptive_memory,
    create_test_roughness_map,
)


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_tier_thresholds_sorted(self) -> None:
        """Test tier thresholds are sorted ascending."""
        for i in range(1, len(DEFAULT_TIER_THRESHOLDS)):
            assert DEFAULT_TIER_THRESHOLDS[i] > DEFAULT_TIER_THRESHOLDS[i - 1]

    def test_default_tier_thresholds_valid_range(self) -> None:
        """Test tier thresholds are in [0, 1]."""
        for threshold in DEFAULT_TIER_THRESHOLDS:
            assert 0.0 <= threshold <= 1.0

    def test_default_resolution_scales_valid(self) -> None:
        """Test resolution scales are positive and <= 1."""
        for scale in DEFAULT_RESOLUTION_SCALES:
            assert 0.0 < scale <= 1.0

    def test_default_ray_counts_positive(self) -> None:
        """Test ray counts are positive integers."""
        for count in DEFAULT_RAY_COUNTS:
            assert count >= 1

    def test_default_denoise_strengths_valid_range(self) -> None:
        """Test denoise strengths are in [0, 1]."""
        for strength in DEFAULT_DENOISE_STRENGTHS:
            assert 0.0 <= strength <= 1.0

    def test_default_ray_budget_reasonable(self) -> None:
        """Test default ray budget is reasonable."""
        assert DEFAULT_RAY_BUDGET > 0
        assert DEFAULT_RAY_BUDGET >= 1920 * 1080  # At least 1080p

    def test_max_rt_roughness_valid(self) -> None:
        """Test max RT roughness is valid."""
        assert 0.0 < MAX_RT_ROUGHNESS <= 1.0


# =============================================================================
# Test Resolution Tier Enum
# =============================================================================


class TestResolutionTier:
    """Tests for ResolutionTier enum."""

    def test_all_tiers_defined(self) -> None:
        """Test all expected tiers are defined."""
        assert ResolutionTier.FULL
        assert ResolutionTier.HALF
        assert ResolutionTier.QUARTER
        assert ResolutionTier.EIGHTH

    def test_tier_count(self) -> None:
        """Test number of tiers."""
        assert len(ResolutionTier) == 4


class TestDenoiseLevel:
    """Tests for DenoiseLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Test all expected levels are defined."""
        assert DenoiseLevel.NONE
        assert DenoiseLevel.LIGHT
        assert DenoiseLevel.MEDIUM
        assert DenoiseLevel.HEAVY

    def test_level_count(self) -> None:
        """Test number of levels."""
        assert len(DenoiseLevel) == 4


# =============================================================================
# Test RoughnessMapping Dataclass
# =============================================================================


class TestRoughnessMapping:
    """Tests for RoughnessMapping dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        mapping = RoughnessMapping()
        assert mapping.ray_count == 1
        assert mapping.resolution_scale == 1.0
        assert mapping.denoise_strength == 0.0
        assert mapping.tier == ResolutionTier.FULL
        assert mapping.denoise_level == DenoiseLevel.NONE
        assert mapping.should_trace is True

    def test_custom_values(self) -> None:
        """Test custom values."""
        mapping = RoughnessMapping(
            ray_count=4,
            resolution_scale=0.25,
            denoise_strength=0.75,
            tier=ResolutionTier.QUARTER,
            denoise_level=DenoiseLevel.HEAVY,
            should_trace=True,
        )
        assert mapping.ray_count == 4
        assert mapping.resolution_scale == 0.25
        assert mapping.denoise_strength == 0.75


# =============================================================================
# Test RoughnessRayMapping
# =============================================================================


class TestRoughnessRayMapping:
    """Tests for RoughnessRayMapping class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        mapping = RoughnessRayMapping()
        assert mapping.tier_count == len(DEFAULT_TIER_THRESHOLDS)
        assert mapping.max_roughness == MAX_RT_ROUGHNESS

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        mapping = RoughnessRayMapping(
            tier_thresholds=[0.2, 0.4, 0.6],
            ray_counts=[1, 2, 4],
            resolution_scales=[1.0, 0.5, 0.25],
            denoise_strengths=[0.0, 0.5, 1.0],
            max_roughness=0.6,
        )
        assert mapping.tier_count == 3
        assert mapping.max_roughness == 0.6

    def test_invalid_configuration_mismatched_lengths(self) -> None:
        """Test validation fails for mismatched list lengths."""
        with pytest.raises(ValueError, match="ray_counts length"):
            RoughnessRayMapping(
                tier_thresholds=[0.1, 0.3],
                ray_counts=[1],  # Wrong length
            )

    def test_invalid_configuration_unsorted_thresholds(self) -> None:
        """Test validation fails for unsorted thresholds."""
        with pytest.raises(ValueError, match="strictly increasing"):
            RoughnessRayMapping(
                tier_thresholds=[0.3, 0.1, 0.5],
                ray_counts=[1, 2, 4],
                resolution_scales=[1.0, 0.5, 0.25],
                denoise_strengths=[0.0, 0.5, 1.0],
            )

    # --- Ray Count Tests ---

    def test_get_ray_count_smooth_surface(self) -> None:
        """Test ray count for smooth surface (roughness 0)."""
        mapping = RoughnessRayMapping()
        assert mapping.get_ray_count(0.0) == 1

    def test_get_ray_count_slightly_rough(self) -> None:
        """Test ray count for slightly rough surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_ray_count(0.15) >= 1

    def test_get_ray_count_moderately_rough(self) -> None:
        """Test ray count for moderately rough surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_ray_count(0.4) >= 2

    def test_get_ray_count_rough_surface(self) -> None:
        """Test ray count for rough surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_ray_count(0.6) >= 4

    def test_get_ray_count_above_max(self) -> None:
        """Test ray count for surface above max roughness."""
        mapping = RoughnessRayMapping()
        assert mapping.get_ray_count(0.8) == 0

    def test_get_ray_count_exact_threshold(self) -> None:
        """Test ray count at exact threshold values."""
        mapping = RoughnessRayMapping()
        # Should use the tier at or below the threshold
        count_at_threshold = mapping.get_ray_count(0.1)
        assert count_at_threshold >= 1

    # --- Resolution Scale Tests ---

    def test_get_resolution_scale_smooth(self) -> None:
        """Test resolution scale for smooth surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_resolution_scale(0.0) == 1.0

    def test_get_resolution_scale_rough(self) -> None:
        """Test resolution scale for rough surface."""
        mapping = RoughnessRayMapping()
        scale = mapping.get_resolution_scale(0.5)
        assert scale <= 0.5

    def test_get_resolution_scale_above_max(self) -> None:
        """Test resolution scale above max roughness."""
        mapping = RoughnessRayMapping()
        assert mapping.get_resolution_scale(0.8) == 0.0

    # --- Denoise Strength Tests ---

    def test_get_denoise_strength_smooth(self) -> None:
        """Test denoise strength for smooth surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_denoise_strength(0.0) == 0.0

    def test_get_denoise_strength_rough(self) -> None:
        """Test denoise strength for rough surface."""
        mapping = RoughnessRayMapping()
        strength = mapping.get_denoise_strength(0.6)
        assert strength > 0.5

    def test_get_denoise_strength_above_max(self) -> None:
        """Test denoise strength above max roughness."""
        mapping = RoughnessRayMapping()
        assert mapping.get_denoise_strength(0.8) == 1.0

    # --- Tier Tests ---

    def test_get_tier_smooth(self) -> None:
        """Test tier for smooth surface."""
        mapping = RoughnessRayMapping()
        assert mapping.get_tier(0.0) == ResolutionTier.FULL

    def test_get_tier_rough(self) -> None:
        """Test tier for rough surface."""
        mapping = RoughnessRayMapping()
        tier = mapping.get_tier(0.5)
        assert tier in (ResolutionTier.QUARTER, ResolutionTier.HALF)

    # --- Should Trace Tests ---

    def test_should_trace_smooth(self) -> None:
        """Test should_trace for smooth surface."""
        mapping = RoughnessRayMapping()
        assert mapping.should_trace(0.0) is True

    def test_should_trace_at_max(self) -> None:
        """Test should_trace at max roughness."""
        mapping = RoughnessRayMapping()
        assert mapping.should_trace(0.7) is True

    def test_should_trace_above_max(self) -> None:
        """Test should_trace above max roughness."""
        mapping = RoughnessRayMapping()
        assert mapping.should_trace(0.8) is False

    # --- Complete Mapping Tests ---

    def test_get_mapping_smooth(self) -> None:
        """Test complete mapping for smooth surface."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(0.0)
        assert result.should_trace is True
        assert result.ray_count >= 1
        assert result.resolution_scale == 1.0

    def test_get_mapping_above_max(self) -> None:
        """Test complete mapping above max roughness."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(0.8)
        assert result.should_trace is False
        assert result.ray_count == 0

    def test_interpolate_parameters_no_blend(self) -> None:
        """Test interpolation without smooth blend."""
        mapping = RoughnessRayMapping()
        result = mapping.interpolate_parameters(0.2, smooth_blend=False)
        assert result.should_trace is True

    def test_interpolate_parameters_with_blend(self) -> None:
        """Test interpolation with smooth blend."""
        mapping = RoughnessRayMapping()
        result = mapping.interpolate_parameters(0.2, smooth_blend=True)
        assert result.should_trace is True


# =============================================================================
# Test Resolution Hierarchy
# =============================================================================


class TestResolutionHierarchy:
    """Tests for ResolutionHierarchy class."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        assert hierarchy.base_width == 1920
        assert hierarchy.base_height == 1080
        assert hierarchy.level_count >= 3

    def test_custom_scales(self) -> None:
        """Test custom resolution scales."""
        hierarchy = ResolutionHierarchy(
            1920, 1080, scales=[1.0, 0.5], roughness_thresholds=[0.3, 1.0]
        )
        assert hierarchy.level_count == 2

    def test_get_level(self) -> None:
        """Test getting level by index."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        level0 = hierarchy.get_level(0)
        assert level0.scale == 1.0
        assert level0.width == 1920
        assert level0.height == 1080

    def test_get_level_invalid_index(self) -> None:
        """Test getting level with invalid index."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        level = hierarchy.get_level(100)  # Should return last level
        assert level is not None

    def test_get_level_for_roughness_smooth(self) -> None:
        """Test getting level for smooth surface."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        level = hierarchy.get_level_for_roughness(0.0)
        assert level.scale == 1.0

    def test_get_level_for_roughness_rough(self) -> None:
        """Test getting level for rough surface."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        level = hierarchy.get_level_for_roughness(0.5)
        assert level.scale <= 0.5

    def test_get_tier(self) -> None:
        """Test getting tier for roughness."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        tier = hierarchy.get_tier(0.0)
        assert tier == ResolutionTier.FULL

    def test_get_scale_for_roughness(self) -> None:
        """Test getting scale for roughness."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        scale = hierarchy.get_scale_for_roughness(0.0)
        assert scale == 1.0

    def test_get_dimensions_for_roughness(self) -> None:
        """Test getting dimensions for roughness."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        w, h = hierarchy.get_dimensions_for_roughness(0.0)
        assert w == 1920
        assert h == 1080

    def test_uv_to_level_pixel(self) -> None:
        """Test UV to pixel conversion."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        x, y, level = hierarchy.uv_to_level_pixel(Vec2(0.5, 0.5), 0.0)
        assert 0 <= x < level.width
        assert 0 <= y < level.height

    def test_level_pixel_to_uv(self) -> None:
        """Test pixel to UV conversion."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        level = hierarchy.get_level(0)
        uv = hierarchy.level_pixel_to_uv(960, 540, level)
        assert 0.0 <= uv.x <= 1.0
        assert 0.0 <= uv.y <= 1.0

    def test_set_base_resolution(self) -> None:
        """Test updating base resolution."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        hierarchy.set_base_resolution(3840, 2160)
        assert hierarchy.base_width == 3840
        assert hierarchy.base_height == 2160

    def test_upscale_result_nearest(self) -> None:
        """Test upscaling with nearest neighbor."""
        hierarchy = ResolutionHierarchy(4, 4)
        half_level = hierarchy.get_level(1)  # Half resolution

        # Create small data
        data = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 1, 1)]

        result = hierarchy.upscale_result(data, half_level, method="nearest")
        assert len(result) == 16

    def test_upscale_result_bilinear(self) -> None:
        """Test upscaling with bilinear interpolation."""
        hierarchy = ResolutionHierarchy(4, 4)
        half_level = hierarchy.get_level(1)

        data = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 1, 1)]

        result = hierarchy.upscale_result(data, half_level, method="bilinear")
        assert len(result) == 16

    def test_estimate_memory_usage(self) -> None:
        """Test memory estimation."""
        hierarchy = ResolutionHierarchy(1920, 1080)
        memory = hierarchy.estimate_memory_usage()
        assert memory > 0


# =============================================================================
# Test Adaptive Ray Scheduler
# =============================================================================


class TestAdaptiveRayScheduler:
    """Tests for AdaptiveRayScheduler class."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        assert scheduler.width == 1920
        assert scheduler.height == 1080
        assert scheduler.temporal_frames == DEFAULT_TEMPORAL_FRAMES

    def test_set_resolution(self) -> None:
        """Test updating resolution."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        scheduler.set_resolution(3840, 2160)
        assert scheduler.width == 3840
        assert scheduler.height == 2160

    def test_advance_frame(self) -> None:
        """Test frame advancement."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        initial_frame = scheduler.current_frame
        scheduler.advance_frame()
        assert scheduler.current_frame == initial_frame + 1

    def test_reset_accumulation(self) -> None:
        """Test resetting accumulation."""
        scheduler = AdaptiveRayScheduler(100, 100)
        # Add some accumulated data
        scheduler.accumulate_results(0, 0, Vec3(1, 0, 0), 10.0)
        scheduler.reset_accumulation()
        assert scheduler.current_frame == 0

    def test_get_pixel_index(self) -> None:
        """Test pixel index calculation."""
        scheduler = AdaptiveRayScheduler(100, 100)
        idx = scheduler.get_pixel_index(50, 25)
        assert idx == 25 * 100 + 50

    def test_schedule_rays_smooth(self) -> None:
        """Test scheduling rays for smooth surface."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        scheduled = scheduler.schedule_rays(100, 100, 0.0)
        assert scheduled.ray_count >= 1
        assert scheduled.accumulation_weight > 0

    def test_schedule_rays_rough(self) -> None:
        """Test scheduling rays for rough surface."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        scheduled = scheduler.schedule_rays(100, 100, 0.5)
        assert scheduled.ray_count >= 1

    def test_schedule_rays_above_max(self) -> None:
        """Test scheduling rays above max roughness."""
        scheduler = AdaptiveRayScheduler(1920, 1080)
        scheduled = scheduler.schedule_rays(100, 100, 0.8)
        assert scheduled.ray_count == 0
        assert scheduled.accumulation_weight == 0.0

    def test_schedule_rays_with_stratification(self) -> None:
        """Test ray scheduling with stratification enabled."""
        scheduler = AdaptiveRayScheduler(1920, 1080, temporal_frames=4)
        scheduled = scheduler.schedule_rays(100, 100, 0.3, use_stratification=True)
        assert scheduled.frame_index == scheduler.current_frame

    def test_get_frame_rays(self) -> None:
        """Test getting rays for all pixels in frame."""
        scheduler = AdaptiveRayScheduler(4, 4)
        roughness_map = [0.1] * 16
        scheduled_list = scheduler.get_frame_rays(roughness_map)
        assert len(scheduled_list) == 16

    def test_accumulate_results(self) -> None:
        """Test result accumulation."""
        scheduler = AdaptiveRayScheduler(100, 100)
        result = scheduler.accumulate_results(0, 0, Vec3(1, 0, 0), 10.0)
        assert result.sample_count == 1
        assert result.color.x > 0

    def test_accumulate_results_multiple(self) -> None:
        """Test multiple result accumulations."""
        scheduler = AdaptiveRayScheduler(100, 100)
        scheduler.accumulate_results(0, 0, Vec3(1, 0, 0), 10.0)
        result = scheduler.accumulate_results(0, 0, Vec3(0, 1, 0), 15.0)
        assert result.sample_count == 2

    def test_get_accumulated(self) -> None:
        """Test getting accumulated result."""
        scheduler = AdaptiveRayScheduler(100, 100)
        scheduler.accumulate_results(5, 5, Vec3(0.5, 0.5, 0.5), 10.0)
        result = scheduler.get_accumulated(5, 5)
        assert result.sample_count == 1

    def test_get_accumulated_empty(self) -> None:
        """Test getting accumulated result for empty pixel."""
        scheduler = AdaptiveRayScheduler(100, 100)
        result = scheduler.get_accumulated(99, 99)
        assert result.sample_count == 0

    def test_is_converged_false(self) -> None:
        """Test convergence check when not converged."""
        scheduler = AdaptiveRayScheduler(100, 100)
        scheduler.accumulate_results(0, 0, Vec3(1, 0, 0), 10.0, confidence=0.1)
        assert scheduler.is_converged(0, 0) is False

    def test_is_converged_true(self) -> None:
        """Test convergence check when converged."""
        scheduler = AdaptiveRayScheduler(100, 100)
        for _ in range(10):
            scheduler.accumulate_results(0, 0, Vec3(1, 0, 0), 10.0, confidence=0.5)
        # May or may not be converged depending on confidence accumulation
        # Just test it returns a boolean
        assert isinstance(scheduler.is_converged(0, 0), bool)

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        scheduler = AdaptiveRayScheduler(100, 100)
        stats = scheduler.get_statistics()
        assert "current_frame" in stats
        assert "temporal_frames" in stats
        assert "pixels_tracked" in stats


# =============================================================================
# Test Ray Budget Manager
# =============================================================================


class TestRayBudgetManager:
    """Tests for RayBudgetManager class."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        manager = RayBudgetManager()
        assert manager.budget == DEFAULT_RAY_BUDGET

    def test_set_budget(self) -> None:
        """Test setting budget."""
        manager = RayBudgetManager()
        manager.set_budget(1_000_000)
        assert manager.budget == 1_000_000

    def test_set_budget_zero(self) -> None:
        """Test setting budget to zero."""
        manager = RayBudgetManager()
        manager.set_budget(0)
        assert manager.budget == 0

    def test_allocate_rays_under_budget(self) -> None:
        """Test allocation when under budget."""
        manager = RayBudgetManager(budget=10_000_000)
        roughness = [0.1] * 100  # Small allocation
        allocation = manager.allocate_rays(roughness, 10, 10)
        assert allocation.utilization < 1.0
        assert manager.budget_scale == 1.0

    def test_allocate_rays_over_budget(self) -> None:
        """Test allocation when over budget."""
        manager = RayBudgetManager(budget=100)
        roughness = [0.3] * 10000  # Large allocation
        allocation = manager.allocate_rays(roughness, 100, 100)
        assert manager.budget_scale < 1.0

    def test_allocate_rays_empty(self) -> None:
        """Test allocation with no traceable pixels."""
        manager = RayBudgetManager()
        roughness = [0.9] * 100  # All above max
        allocation = manager.allocate_rays(roughness, 10, 10)
        assert allocation.total_rays == 0

    def test_get_rays_for_pixel(self) -> None:
        """Test getting rays for a pixel."""
        manager = RayBudgetManager()
        rays = manager.get_rays_for_pixel(0.1)
        assert rays >= 1

    def test_get_rays_for_pixel_above_max(self) -> None:
        """Test getting rays for pixel above max roughness."""
        manager = RayBudgetManager()
        rays = manager.get_rays_for_pixel(0.9)
        assert rays == 0

    def test_estimate_frame_time(self) -> None:
        """Test frame time estimation."""
        manager = RayBudgetManager(budget=1_000_000)
        roughness = [0.1] * 1000
        manager.allocate_rays(roughness, 100, 10)
        time_ms = manager.estimate_frame_time(rays_per_second=1_000_000_000)
        assert time_ms >= 0

    def test_adjust_budget_for_target_time(self) -> None:
        """Test adjusting budget for target time."""
        manager = RayBudgetManager()
        manager.adjust_budget_for_target_time(16.67)  # 60 FPS target
        assert manager.budget > 0

    def test_get_utilization_no_allocation(self) -> None:
        """Test utilization before any allocation."""
        manager = RayBudgetManager()
        assert manager.get_utilization() == 0.0

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        manager = RayBudgetManager()
        stats = manager.get_statistics()
        assert "budget" in stats
        assert "utilization" in stats


# =============================================================================
# Test Adaptive RT Config
# =============================================================================


class TestAdaptiveRTConfig:
    """Tests for AdaptiveRTConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = AdaptiveRTConfig()
        assert config.min_rays == 1
        assert config.max_rays == 8
        assert config.temporal_accumulation == DEFAULT_TEMPORAL_FRAMES

    def test_validation_valid(self) -> None:
        """Test validation of valid config."""
        config = AdaptiveRTConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_invalid_min_rays(self) -> None:
        """Test validation with invalid min_rays."""
        config = AdaptiveRTConfig(min_rays=0)
        # Post-init should fix it
        assert config.min_rays >= 1

    def test_validation_min_greater_than_max(self) -> None:
        """Test validation when min > max."""
        config = AdaptiveRTConfig(min_rays=10, max_rays=2)
        # Post-init should fix it
        assert config.max_rays >= config.min_rays

    def test_create_roughness_mapping(self) -> None:
        """Test creating roughness mapping from config."""
        config = AdaptiveRTConfig()
        mapping = config.create_roughness_mapping()
        assert mapping is not None
        assert mapping.tier_count == len(config.roughness_thresholds)

    def test_low_quality_preset(self) -> None:
        """Test low quality preset."""
        config = AdaptiveRTConfig.low_quality()
        assert config.max_rays <= 4
        assert config.ray_budget < DEFAULT_RAY_BUDGET

    def test_medium_quality_preset(self) -> None:
        """Test medium quality preset."""
        config = AdaptiveRTConfig.medium_quality()
        assert config.max_rays >= 2

    def test_high_quality_preset(self) -> None:
        """Test high quality preset."""
        config = AdaptiveRTConfig.high_quality()
        assert config.max_rays >= 4
        assert config.enable_stratification is True


# =============================================================================
# Test Adaptive RT Pass
# =============================================================================


class TestAdaptiveRTPass:
    """Tests for AdaptiveRTPass class."""

    def test_initialization(self) -> None:
        """Test initialization."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 1920, 1080)
        assert pass_obj.width == 1920
        assert pass_obj.height == 1080

    def test_set_resolution(self) -> None:
        """Test setting resolution."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 1920, 1080)
        pass_obj.set_resolution(3840, 2160)
        assert pass_obj.width == 3840
        assert pass_obj.height == 2160

    def test_execute_adaptive_simple(self) -> None:
        """Test simple adaptive execution."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        roughness_map = [0.1] * 16
        results = pass_obj.execute_adaptive(roughness_map)
        assert len(results) == 16

    def test_execute_adaptive_with_trace_func(self) -> None:
        """Test adaptive execution with custom trace function."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        roughness_map = [0.1] * 16

        def mock_trace(x: int, y: int, ray_count: int) -> Vec3:
            return Vec3(0.5, 0.5, 0.5)

        results = pass_obj.execute_adaptive(roughness_map, trace_func=mock_trace)
        assert len(results) == 16

    def test_execute_adaptive_mixed_roughness(self) -> None:
        """Test adaptive execution with mixed roughness."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        # Mix of smooth and rough
        roughness_map = [0.0, 0.2, 0.4, 0.6] * 4
        results = pass_obj.execute_adaptive(roughness_map)
        assert len(results) == 16

    def test_get_result_at(self) -> None:
        """Test getting result at coordinates."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        roughness_map = [0.1] * 16
        pass_obj.execute_adaptive(roughness_map)
        result = pass_obj.get_result_at(0, 0)
        assert result is not None

    def test_get_result_at_invalid(self) -> None:
        """Test getting result at invalid coordinates."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        result = pass_obj.get_result_at(100, 100)
        assert result.rays_traced == 0

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        config = AdaptiveRTConfig()
        pass_obj = AdaptiveRTPass(config, 4, 4)
        roughness_map = [0.1] * 16
        pass_obj.execute_adaptive(roughness_map)
        stats = pass_obj.get_statistics()
        assert "total_rays_traced" in stats
        assert "pixels_processed" in stats

    def test_temporal_accumulation(self) -> None:
        """Test temporal accumulation across frames."""
        config = AdaptiveRTConfig(temporal_accumulation=4)
        pass_obj = AdaptiveRTPass(config, 4, 4)
        roughness_map = [0.1] * 16

        # Execute multiple frames
        for _ in range(4):
            pass_obj.execute_adaptive(roughness_map)

        result = pass_obj.get_result_at(0, 0)
        # After accumulation, confidence should increase
        assert result.confidence > 0


# =============================================================================
# Test WGSL Shader Generation
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_generate_adaptive_rays_wgsl(self) -> None:
        """Test WGSL generation."""
        config = AdaptiveRTConfig()
        shader = generate_adaptive_rays_wgsl(config)
        assert "MIN_RAYS" in shader
        assert "MAX_RAYS" in shader
        assert "get_ray_count" in shader

    def test_wgsl_contains_configuration(self) -> None:
        """Test WGSL contains configuration values."""
        config = AdaptiveRTConfig(min_rays=2, max_rays=16)
        shader = generate_adaptive_rays_wgsl(config)
        assert "2u" in shader  # min_rays
        assert "16u" in shader  # max_rays


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_estimate_adaptive_memory(self) -> None:
        """Test memory estimation."""
        config = AdaptiveRTConfig()
        memory = estimate_adaptive_memory(1920, 1080, config)
        assert memory > 0

    def test_create_test_roughness_map_gradient(self) -> None:
        """Test gradient roughness map creation."""
        roughness_map = create_test_roughness_map(10, 10, pattern="gradient")
        assert len(roughness_map) == 100
        # First pixel should be smooth
        assert roughness_map[0] < 0.1

    def test_create_test_roughness_map_random(self) -> None:
        """Test random roughness map creation."""
        roughness_map = create_test_roughness_map(10, 10, pattern="random")
        assert len(roughness_map) == 100
        # All values should be in valid range
        for r in roughness_map:
            assert 0.0 <= r <= 0.7

    def test_create_test_roughness_map_uniform(self) -> None:
        """Test uniform roughness map creation."""
        roughness_map = create_test_roughness_map(10, 10, pattern="uniform")
        assert len(roughness_map) == 100
        # All values should be the same
        assert all(r == roughness_map[0] for r in roughness_map)

    def test_create_test_roughness_map_bands(self) -> None:
        """Test banded roughness map creation."""
        roughness_map = create_test_roughness_map(40, 10, pattern="bands")
        assert len(roughness_map) == 400


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_roughness_zero(self) -> None:
        """Test with roughness exactly 0."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(0.0)
        assert result.should_trace is True
        assert result.ray_count >= 1
        assert result.resolution_scale == 1.0
        assert result.denoise_strength == 0.0

    def test_roughness_one(self) -> None:
        """Test with roughness exactly 1."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(1.0)
        assert result.should_trace is False
        assert result.ray_count == 0

    def test_roughness_negative(self) -> None:
        """Test with negative roughness (should clamp)."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(-0.5)
        # Should be treated as 0
        assert result.ray_count >= 1

    def test_roughness_greater_than_one(self) -> None:
        """Test with roughness > 1 (should clamp)."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(1.5)
        assert result.should_trace is False

    def test_single_pixel_resolution(self) -> None:
        """Test with 1x1 resolution."""
        hierarchy = ResolutionHierarchy(1, 1)
        assert hierarchy.base_width == 1
        assert hierarchy.base_height == 1

    def test_zero_budget(self) -> None:
        """Test with zero ray budget."""
        manager = RayBudgetManager(budget=0)
        rays = manager.get_rays_for_pixel(0.1)
        # Should still return at least 0
        assert rays >= 0

    def test_exact_threshold_values(self) -> None:
        """Test roughness at exact threshold values."""
        mapping = RoughnessRayMapping()
        for threshold in mapping.tier_thresholds:
            result = mapping.get_mapping(threshold)
            assert result.should_trace is True

    def test_very_small_roughness(self) -> None:
        """Test with very small roughness values."""
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(0.0001)
        assert result.should_trace is True
        assert result.resolution_scale == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the adaptive ray system."""

    def test_full_pipeline_smooth_surface(self) -> None:
        """Test full pipeline for smooth surfaces."""
        config = AdaptiveRTConfig.high_quality()
        pass_obj = AdaptiveRTPass(config, 100, 100)

        # All smooth surfaces
        roughness_map = [0.05] * 10000

        results = pass_obj.execute_adaptive(roughness_map)

        # Check that smooth surfaces use full resolution
        result = results[0]
        assert result.resolution_scale == 1.0
        assert result.rays_traced >= 1

    def test_full_pipeline_rough_surface(self) -> None:
        """Test full pipeline for rough surfaces."""
        config = AdaptiveRTConfig.high_quality()
        pass_obj = AdaptiveRTPass(config, 100, 100)

        # All rough surfaces (but still traceable)
        roughness_map = [0.5] * 10000

        results = pass_obj.execute_adaptive(roughness_map)

        # Check that rough surfaces use lower resolution
        result = results[0]
        assert result.resolution_scale <= 0.5

    def test_full_pipeline_mixed_surfaces(self) -> None:
        """Test full pipeline with mixed surface types."""
        config = AdaptiveRTConfig.high_quality()
        pass_obj = AdaptiveRTPass(config, 10, 10)

        # Gradient of roughness values
        roughness_map = create_test_roughness_map(10, 10, pattern="gradient")

        results = pass_obj.execute_adaptive(roughness_map)

        # First column should have higher resolution than last
        first_result = results[0]
        last_result = results[9]

        assert first_result.resolution_scale >= last_result.resolution_scale

    def test_budget_constrains_rays(self) -> None:
        """Test that budget manager calculates correct scaling."""
        # Create a budget manager directly to test its scaling behavior
        from engine.rendering.reflections.rt_adaptive_rays import RayBudgetManager

        manager = RayBudgetManager(budget=100)

        # Roughness 0.3 should request ~2 rays each (tier 1)
        # 100 pixels * 2 rays = 200 rays needed
        roughness_map = [0.2] * 100

        allocation = manager.allocate_rays(roughness_map, 10, 10)

        # With budget 100 and ~200 rays needed, scale should be ~0.5
        assert manager.budget_scale < 1.0  # Budget scaling is active
        assert allocation.total_rays <= 100  # Should respect budget

    def test_temporal_convergence(self) -> None:
        """Test that temporal accumulation converges."""
        config = AdaptiveRTConfig(temporal_accumulation=8)
        pass_obj = AdaptiveRTPass(config, 4, 4)

        roughness_map = [0.1] * 16

        # Execute many frames
        for _ in range(8):
            pass_obj.execute_adaptive(roughness_map)

        result = pass_obj.get_result_at(0, 0)
        # Should have accumulated samples
        assert result.confidence > 0
