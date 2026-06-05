"""
Tests for TSR Adaptive Sharpening (T-PP-6.4)

Tests the adaptive sharpening functionality added to TSRLanczosUpscaler,
including local contrast measurement, adaptive strength calculation,
and quality presets.
"""

from __future__ import annotations

import pytest
from typing import List, Tuple

from engine.rendering.postprocess.upscaling import (
    measure_local_contrast,
    get_adaptive_sharpening_for_quality,
    TSRLanczosSettings,
    TSRLanczosUpscaler,
    LanczosKernel,
)


# =============================================================================
# Test measure_local_contrast
# =============================================================================


class TestMeasureLocalContrast:
    """Tests for the measure_local_contrast function."""

    def test_contrast_measurement_uniform_returns_zero(self) -> None:
        """Uniform area (all same color) should have zero contrast."""
        center = (0.5, 0.5, 0.5)
        neighbors = [(0.5, 0.5, 0.5)] * 8  # All same as center

        contrast = measure_local_contrast(center, neighbors)

        assert contrast == pytest.approx(0.0, abs=1e-6)

    def test_contrast_measurement_edge_returns_high(self) -> None:
        """Sharp edge (black/white) should have high contrast."""
        center = (1.0, 1.0, 1.0)  # White
        neighbors = [(0.0, 0.0, 0.0)] * 8  # All black

        contrast = measure_local_contrast(center, neighbors)

        # Contrast should be clamped to 1.0 for maximum difference
        assert contrast == pytest.approx(1.0, abs=1e-6)

    def test_contrast_measurement_gradient_intermediate(self) -> None:
        """Gradient should have intermediate contrast value."""
        # Center is brighter than the average of its neighbors
        center = (0.7, 0.7, 0.7)
        # Neighbors average to about 0.45, creating contrast with 0.7 center
        neighbors = [
            (0.3, 0.3, 0.3),
            (0.4, 0.4, 0.4),
            (0.5, 0.5, 0.5),
            (0.6, 0.6, 0.6),
        ]
        # avg = (0.3 + 0.4 + 0.5 + 0.6) / 4 = 0.45
        # contrast = |0.7 - 0.45| * 2 = 0.5

        contrast = measure_local_contrast(center, neighbors)

        # Should be between 0 and 1
        assert 0.0 < contrast < 1.0

    def test_contrast_empty_neighbors_returns_zero(self) -> None:
        """Empty neighbors list should return zero contrast."""
        center = (0.5, 0.5, 0.5)
        neighbors: List[Tuple[float, float, float]] = []

        contrast = measure_local_contrast(center, neighbors)

        assert contrast == 0.0

    def test_contrast_single_neighbor(self) -> None:
        """Single neighbor should still compute valid contrast."""
        center = (0.8, 0.8, 0.8)
        neighbors = [(0.2, 0.2, 0.2)]

        contrast = measure_local_contrast(center, neighbors)

        assert contrast > 0.0
        assert contrast <= 1.0

    def test_contrast_uses_luminance_not_rgb_average(self) -> None:
        """Contrast should use proper luminance weighting (Rec. 709)."""
        # Green contributes more to luminance than red or blue
        center = (0.0, 0.5, 0.0)  # Pure green
        neighbors_red = [(0.5, 0.0, 0.0)] * 4  # Pure red
        neighbors_blue = [(0.0, 0.0, 0.5)] * 4  # Pure blue

        contrast_red = measure_local_contrast(center, neighbors_red)
        contrast_blue = measure_local_contrast(center, neighbors_blue)

        # Both should be non-zero and likely different due to luminance weights
        assert contrast_red > 0.0
        assert contrast_blue > 0.0

    def test_contrast_clamped_to_one(self) -> None:
        """Contrast should never exceed 1.0 even for extreme differences."""
        center = (1.0, 1.0, 1.0)
        neighbors = [(0.0, 0.0, 0.0)] * 8

        contrast = measure_local_contrast(center, neighbors)

        assert contrast == 1.0

    def test_contrast_symmetric_around_center(self) -> None:
        """Contrast should be symmetric (doesn't matter if center is lighter or darker)."""
        center_light = (0.8, 0.8, 0.8)
        center_dark = (0.2, 0.2, 0.2)
        neighbors_mid = [(0.5, 0.5, 0.5)] * 4

        contrast_light = measure_local_contrast(center_light, neighbors_mid)
        contrast_dark = measure_local_contrast(center_dark, neighbors_mid)

        assert contrast_light == pytest.approx(contrast_dark, abs=1e-6)


# =============================================================================
# Test get_adaptive_sharpening_for_quality
# =============================================================================


class TestGetAdaptiveSharpeningForQuality:
    """Tests for the get_adaptive_sharpening_for_quality function."""

    def test_quality_preset_ultra_returns_correct_range(self) -> None:
        """Ultra quality should return (0.4, 0.9)."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("ultra")

        assert min_sharp == 0.4
        assert max_sharp == 0.9

    def test_quality_preset_high_returns_correct_range(self) -> None:
        """High quality should return (0.3, 0.8)."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("high")

        assert min_sharp == 0.3
        assert max_sharp == 0.8

    def test_quality_preset_medium_returns_correct_range(self) -> None:
        """Medium quality should return (0.2, 0.6)."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("medium")

        assert min_sharp == 0.2
        assert max_sharp == 0.6

    def test_quality_preset_low_returns_correct_range(self) -> None:
        """Low quality should return (0.1, 0.4)."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("low")

        assert min_sharp == 0.1
        assert max_sharp == 0.4

    def test_quality_preset_case_insensitive(self) -> None:
        """Quality names should be case-insensitive."""
        assert get_adaptive_sharpening_for_quality("ULTRA") == (0.4, 0.9)
        assert get_adaptive_sharpening_for_quality("High") == (0.3, 0.8)
        assert get_adaptive_sharpening_for_quality("MEDIUM") == (0.2, 0.6)
        assert get_adaptive_sharpening_for_quality("LoW") == (0.1, 0.4)

    def test_quality_unknown_returns_default(self) -> None:
        """Unknown quality should return default (high) values."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("unknown")

        assert min_sharp == 0.3
        assert max_sharp == 0.8

    def test_quality_empty_string_returns_default(self) -> None:
        """Empty string should return default values."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("")

        assert min_sharp == 0.3
        assert max_sharp == 0.8


# =============================================================================
# Test TSRLanczosSettings adaptive sharpening fields
# =============================================================================


class TestTSRLanczosSettingsAdaptive:
    """Tests for TSRLanczosSettings adaptive sharpening configuration."""

    def test_default_adaptive_sharpening_enabled(self) -> None:
        """Adaptive sharpening should be enabled by default."""
        settings = TSRLanczosSettings()

        assert settings.adaptive_sharpening is True

    def test_default_sharpening_min(self) -> None:
        """Default minimum sharpening should be 0.3."""
        settings = TSRLanczosSettings()

        assert settings.sharpening_min == 0.3

    def test_default_sharpening_max(self) -> None:
        """Default maximum sharpening should be 0.8."""
        settings = TSRLanczosSettings()

        assert settings.sharpening_max == 0.8

    def test_default_contrast_threshold(self) -> None:
        """Default contrast threshold should be 0.1."""
        settings = TSRLanczosSettings()

        assert settings.contrast_threshold == 0.1

    def test_custom_adaptive_settings(self) -> None:
        """Custom adaptive sharpening settings should be respected."""
        settings = TSRLanczosSettings(
            adaptive_sharpening=True,
            sharpening_min=0.1,
            sharpening_max=0.5,
            contrast_threshold=0.2,
        )

        assert settings.sharpening_min == 0.1
        assert settings.sharpening_max == 0.5
        assert settings.contrast_threshold == 0.2


# =============================================================================
# Test TSRLanczosUpscaler.apply_sharpening (adaptive behavior)
# =============================================================================


class TestApplySharpeningAdaptive:
    """Tests for TSRLanczosUpscaler.apply_sharpening with adaptive behavior."""

    def test_adaptive_strength_below_threshold_uses_min(self) -> None:
        """Low contrast (below threshold) should use minimum sharpening."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=True,
            sharpening_min=0.3,
            sharpening_max=0.8,
            contrast_threshold=0.1,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Uniform area: very low contrast
        center = (0.5, 0.5, 0.5)
        neighbors = [(0.5, 0.5, 0.5)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # With zero contrast and zero difference, output should equal input
        # (no sharpening effect when center == avg)
        assert result == center

    def test_adaptive_strength_above_threshold_interpolates(self) -> None:
        """Moderate contrast should interpolate between min and max."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=True,
            sharpening_min=0.3,
            sharpening_max=0.8,
            contrast_threshold=0.1,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Create moderate contrast scenario
        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # Result should be brighter than center (sharpening enhances difference)
        assert result[0] > center[0]
        assert result[1] > center[1]
        assert result[2] > center[2]

    def test_adaptive_strength_high_contrast_uses_max(self) -> None:
        """High contrast (approaching 1.0) should use maximum sharpening."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=True,
            sharpening_min=0.3,
            sharpening_max=0.8,
            contrast_threshold=0.1,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # High contrast: white center, black neighbors
        center = (0.9, 0.9, 0.9)
        neighbors = [(0.1, 0.1, 0.1)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # With max sharpening, effect should be strong
        # Sharpened = center + 0.8 * (center - avg)
        # center=0.9, avg=0.1, diff=0.8, result = 0.9 + 0.8*0.8 = 1.54 -> clamped to 1.0
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_sharpening_no_overshoot_white(self) -> None:
        """Sharpening white pixels should not exceed 1.0."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=1.0,
            adaptive_sharpening=True,
            sharpening_max=1.0,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (1.0, 1.0, 1.0)
        neighbors = [(0.5, 0.5, 0.5)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        assert result[0] <= 1.0
        assert result[1] <= 1.0
        assert result[2] <= 1.0

    def test_sharpening_no_overshoot_black(self) -> None:
        """Sharpening black pixels should not go below 0.0."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=1.0,
            adaptive_sharpening=True,
            sharpening_max=1.0,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.0, 0.0, 0.0)
        neighbors = [(0.5, 0.5, 0.5)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        assert result[0] >= 0.0
        assert result[1] >= 0.0
        assert result[2] >= 0.0

    def test_disabled_adaptive_uses_fixed_strength(self) -> None:
        """With adaptive_sharpening=False, should use fixed sharpness value."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=False,  # Disabled
            sharpening_min=0.1,
            sharpening_max=0.9,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # Should use fixed sharpness=0.5
        # Expected: 0.6 + 0.5 * (0.6 - 0.4) = 0.6 + 0.1 = 0.7
        assert result[0] == pytest.approx(0.7, abs=1e-6)
        assert result[1] == pytest.approx(0.7, abs=1e-6)
        assert result[2] == pytest.approx(0.7, abs=1e-6)

    def test_sharpening_disabled_returns_original(self) -> None:
        """When sharpening is disabled, return original color unchanged."""
        settings = TSRLanczosSettings(
            sharpening=False,
            sharpness=0.5,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        assert result == center

    def test_sharpness_zero_returns_original(self) -> None:
        """When sharpness is zero, return original color unchanged."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.0,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        assert result == center

    def test_empty_neighbors_returns_original(self) -> None:
        """Empty neighbors list should return original color."""
        settings = TSRLanczosSettings(sharpening=True, sharpness=0.5)
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.6, 0.6, 0.6)
        neighbors: List[Tuple[float, float, float]] = []

        result = upscaler.apply_sharpening(center, neighbors)

        assert result == center


# =============================================================================
# Edge cases and boundary tests
# =============================================================================


class TestSharpeningEdgeCases:
    """Edge case tests for sharpening behavior."""

    def test_very_small_contrast_threshold(self) -> None:
        """Very small threshold should still work correctly."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            contrast_threshold=0.001,
            sharpening_min=0.2,
            sharpening_max=0.9,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.5, 0.5, 0.5)
        neighbors = [(0.49, 0.49, 0.49)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # Should still produce valid output
        assert 0.0 <= result[0] <= 1.0

    def test_contrast_threshold_at_one(self) -> None:
        """Threshold of 1.0 should always use minimum sharpening."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            contrast_threshold=1.0,  # Impossible to exceed
            sharpening_min=0.1,
            sharpening_max=0.9,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Use a case where we can verify min sharpening is used
        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # With sharpening_min=0.1: 0.6 + 0.1 * (0.6 - 0.4) = 0.62
        assert result[0] == pytest.approx(0.62, abs=1e-6)

    def test_min_equals_max_sharpening(self) -> None:
        """When min equals max, sharpening should be constant."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            sharpening_min=0.5,
            sharpening_max=0.5,  # Same as min
            contrast_threshold=0.1,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Different contrast levels should produce same sharpening strength
        center1 = (0.5, 0.5, 0.5)
        neighbors1 = [(0.4, 0.4, 0.4)] * 4  # Low contrast

        center2 = (0.9, 0.9, 0.9)
        neighbors2 = [(0.1, 0.1, 0.1)] * 4  # High contrast

        result1 = upscaler.apply_sharpening(center1, neighbors1)
        result2 = upscaler.apply_sharpening(center2, neighbors2)

        # Both should use strength=0.5
        # result1: 0.5 + 0.5 * (0.5 - 0.4) = 0.55
        # result2: 0.9 + 0.5 * (0.9 - 0.1) = 1.3 -> clamped to 1.0
        assert result1[0] == pytest.approx(0.55, abs=1e-6)
        assert result2[0] == pytest.approx(1.0, abs=1e-6)

    def test_single_neighbor_sharpening(self) -> None:
        """Sharpening with single neighbor should still work."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=False,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.6, 0.6, 0.6)
        neighbors = [(0.4, 0.4, 0.4)]  # Just one neighbor

        result = upscaler.apply_sharpening(center, neighbors)

        # 0.6 + 0.5 * (0.6 - 0.4) = 0.7
        assert result[0] == pytest.approx(0.7, abs=1e-6)

    def test_many_neighbors_sharpening(self) -> None:
        """Sharpening with many neighbors should average them all."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=False,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.5, 0.5, 0.5)
        # 16 neighbors averaging to 0.4
        neighbors = [(0.3, 0.3, 0.3)] * 8 + [(0.5, 0.5, 0.5)] * 8

        result = upscaler.apply_sharpening(center, neighbors)

        # avg = (0.3*8 + 0.5*8) / 16 = 0.4
        # result = 0.5 + 0.5 * (0.5 - 0.4) = 0.55
        assert result[0] == pytest.approx(0.55, abs=1e-6)

    def test_negative_difference_darkens(self) -> None:
        """When center is darker than neighbors, sharpening should darken further."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=False,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.3, 0.3, 0.3)
        neighbors = [(0.6, 0.6, 0.6)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # 0.3 + 0.5 * (0.3 - 0.6) = 0.3 - 0.15 = 0.15
        assert result[0] == pytest.approx(0.15, abs=1e-6)

    def test_rgb_independent_channels(self) -> None:
        """Each RGB channel should be sharpened independently."""
        settings = TSRLanczosSettings(
            sharpening=True,
            sharpness=0.5,
            adaptive_sharpening=False,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.8, 0.5, 0.2)  # Different per channel
        neighbors = [(0.6, 0.5, 0.4)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # R: 0.8 + 0.5 * (0.8 - 0.6) = 0.9
        # G: 0.5 + 0.5 * (0.5 - 0.5) = 0.5
        # B: 0.2 + 0.5 * (0.2 - 0.4) = 0.1
        assert result[0] == pytest.approx(0.9, abs=1e-6)
        assert result[1] == pytest.approx(0.5, abs=1e-6)
        assert result[2] == pytest.approx(0.1, abs=1e-6)


# =============================================================================
# Integration tests
# =============================================================================


class TestSharpeningIntegration:
    """Integration tests combining multiple sharpening features."""

    def test_upscaler_with_quality_preset(self) -> None:
        """Upscaler should work with quality preset sharpening values."""
        min_sharp, max_sharp = get_adaptive_sharpening_for_quality("high")

        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            sharpening_min=min_sharp,
            sharpening_max=max_sharp,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.7, 0.7, 0.7)
        neighbors = [(0.3, 0.3, 0.3)] * 4

        result = upscaler.apply_sharpening(center, neighbors)

        # Should produce valid sharpened output
        assert 0.0 <= result[0] <= 1.0
        assert result[0] > center[0]  # Should be brighter (center > avg)

    def test_full_image_simulation(self) -> None:
        """Simulate sharpening across different image regions."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            sharpening_min=0.3,
            sharpening_max=0.8,
            contrast_threshold=0.1,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Flat region (low contrast)
        flat_center = (0.5, 0.5, 0.5)
        flat_neighbors = [(0.5, 0.5, 0.5)] * 4
        flat_result = upscaler.apply_sharpening(flat_center, flat_neighbors)

        # Edge region (high contrast)
        edge_center = (0.9, 0.9, 0.9)
        edge_neighbors = [(0.1, 0.1, 0.1)] * 4
        edge_result = upscaler.apply_sharpening(edge_center, edge_neighbors)

        # Flat should be unchanged (no difference to sharpen)
        assert flat_result == flat_center

        # Edge should be clamped to 1.0 due to strong sharpening
        assert edge_result[0] == pytest.approx(1.0, abs=1e-6)

    def test_settings_modification_affects_output(self) -> None:
        """Changing settings should affect sharpening output."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            sharpening_min=0.2,
            sharpening_max=0.4,
        )
        upscaler = TSRLanczosUpscaler(settings)

        center = (0.7, 0.7, 0.7)
        neighbors = [(0.5, 0.5, 0.5)] * 4

        result_low = upscaler.apply_sharpening(center, neighbors)

        # Increase sharpening range
        settings.sharpening_min = 0.6
        settings.sharpening_max = 0.9

        result_high = upscaler.apply_sharpening(center, neighbors)

        # Higher settings should produce more sharpening
        assert result_high[0] > result_low[0]
