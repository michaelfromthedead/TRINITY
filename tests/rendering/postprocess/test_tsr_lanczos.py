"""Tests for TSR Lanczos Upsampling (T-PP-6.3).

Tests the native Lanczos-based temporal super-resolution fallback
when DLSS/FSR2/XeSS are unavailable.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from engine.rendering.postprocess.upscaling import (
    LanczosKernel,
    TSRLanczosSettings,
    TSRLanczosUpscaler,
    create_tsr_lanczos,
    generate_lanczos_weights,
    lanczos_kernel,
)


class TestLanczosKernel:
    """Tests for lanczos_kernel function."""

    def test_kernel_center_is_one(self) -> None:
        """Kernel value at center (x=0) should be 1.0."""
        assert lanczos_kernel(0.0, a=2) == 1.0
        assert lanczos_kernel(0.0, a=3) == 1.0

    def test_kernel_at_boundary_is_zero(self) -> None:
        """Kernel value at boundary (|x|>=a) should be 0.0."""
        # Lanczos-2: zero at |x| >= 2
        assert lanczos_kernel(2.0, a=2) == 0.0
        assert lanczos_kernel(-2.0, a=2) == 0.0
        assert lanczos_kernel(3.0, a=2) == 0.0

        # Lanczos-3: zero at |x| >= 3
        assert lanczos_kernel(3.0, a=3) == 0.0
        assert lanczos_kernel(-3.0, a=3) == 0.0
        assert lanczos_kernel(4.0, a=3) == 0.0

    def test_kernel_decays_from_center(self) -> None:
        """Kernel values should decay as distance from center increases.

        Note: The Lanczos kernel crosses zero at integer positions (1, 2, ...)
        and has negative lobes between them. We verify the magnitude
        decreases moving away from center.
        """
        k0 = lanczos_kernel(0.0, a=2)
        k025 = lanczos_kernel(0.25, a=2)
        k05 = lanczos_kernel(0.5, a=2)
        k075 = lanczos_kernel(0.75, a=2)

        # Kernel decays from center towards the first zero crossing
        assert k0 > k025 > k05 > k075 > 0

    def test_kernel_is_symmetric(self) -> None:
        """Kernel should be symmetric around zero."""
        for x in [0.1, 0.5, 1.0, 1.5]:
            assert lanczos_kernel(x, a=2) == pytest.approx(
                lanczos_kernel(-x, a=2), rel=1e-10
            )
            assert lanczos_kernel(x, a=3) == pytest.approx(
                lanczos_kernel(-x, a=3), rel=1e-10
            )

    def test_kernel_has_negative_lobes(self) -> None:
        """Lanczos kernel should have negative lobes (causes ringing)."""
        # Between 1 and 2, the kernel goes negative for Lanczos-2
        k15 = lanczos_kernel(1.5, a=2)
        assert k15 < 0


class TestGenerateLanczosWeights:
    """Tests for generate_lanczos_weights function."""

    def test_weights_are_normalized(self) -> None:
        """Weights should sum to 1.0."""
        weights = generate_lanczos_weights(0.5, a=2)
        total = sum(w for _, w in weights)
        assert total == pytest.approx(1.0, rel=1e-6)

        weights = generate_lanczos_weights(0.33, a=3)
        total = sum(w for _, w in weights)
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_weights_have_correct_count(self) -> None:
        """Number of weights depends on scale and kernel size."""
        weights = generate_lanczos_weights(0.5, a=2)
        # For scale=0.5, radius = ceil(2/0.5) = 4, so -4 to +4 = 9 samples max
        assert len(weights) > 0
        assert len(weights) <= 9

    def test_weights_include_center(self) -> None:
        """Weights should include offset 0 (center pixel)."""
        weights = generate_lanczos_weights(0.5, a=2)
        offsets = [o for o, _ in weights]
        assert 0 in offsets

    def test_weights_filter_small_values(self) -> None:
        """Weights smaller than threshold should be filtered."""
        weights = generate_lanczos_weights(0.5, a=2)
        for _, w in weights:
            assert w > 0.0001 or w < -0.0001  # After normalization, may be negative

    def test_different_scales_produce_different_weights(self) -> None:
        """Different scale factors should produce different weight distributions."""
        weights_2x = generate_lanczos_weights(0.5, a=2)  # 2x upscale
        weights_3x = generate_lanczos_weights(0.33, a=2)  # 3x upscale

        # Different number of samples expected
        assert len(weights_2x) != len(weights_3x) or weights_2x != weights_3x


class TestTSRLanczosSettings:
    """Tests for TSRLanczosSettings dataclass."""

    def test_default_settings(self) -> None:
        """Default settings should have expected values."""
        settings = TSRLanczosSettings()

        assert settings.enabled is True
        assert settings.kernel == LanczosKernel.LANCZOS2
        assert settings.scale_factor == 2.0
        assert settings.sharpness == 0.5
        assert settings.temporal_blend == 0.1
        assert settings.separable is True

    def test_custom_settings(self) -> None:
        """Custom settings should override defaults."""
        settings = TSRLanczosSettings(
            kernel=LanczosKernel.LANCZOS3,
            scale_factor=3.0,
            sharpness=0.8,
            separable=False,
        )

        assert settings.kernel == LanczosKernel.LANCZOS3
        assert settings.scale_factor == 3.0
        assert settings.sharpness == 0.8
        assert settings.separable is False


class TestTSRLanczosUpscaler:
    """Tests for TSRLanczosUpscaler class."""

    def test_is_always_available(self) -> None:
        """TSR Lanczos should always be available."""
        assert TSRLanczosUpscaler.is_available() is True

    def test_default_initialization(self) -> None:
        """Upscaler should initialize with default settings."""
        upscaler = TSRLanczosUpscaler()

        assert upscaler.settings.enabled is True
        assert upscaler.kernel_radius == 2
        assert upscaler.output_scale == (2.0, 2.0)

    def test_custom_initialization(self) -> None:
        """Upscaler should respect custom settings."""
        settings = TSRLanczosSettings(
            kernel=LanczosKernel.LANCZOS3,
            scale_factor=1.5,
        )
        upscaler = TSRLanczosUpscaler(settings)

        assert upscaler.kernel_radius == 3
        assert upscaler.output_scale == (1.5, 1.5)

    def test_jitter_offset_changes_per_frame(self) -> None:
        """Jitter offset should follow Halton sequence."""
        upscaler = TSRLanczosUpscaler()

        offsets = []
        for _ in range(8):
            offsets.append(upscaler.get_jitter_offset())
            upscaler.advance_frame()

        # All 8 offsets should be different
        unique_offsets = set(offsets)
        assert len(unique_offsets) == 8

    def test_jitter_offset_cycles(self) -> None:
        """Jitter offset should cycle after sequence length."""
        upscaler = TSRLanczosUpscaler()

        first_cycle = []
        for _ in range(8):
            first_cycle.append(upscaler.get_jitter_offset())
            upscaler.advance_frame()

        second_cycle = []
        for _ in range(8):
            second_cycle.append(upscaler.get_jitter_offset())
            upscaler.advance_frame()

        assert first_cycle == second_cycle

    def test_jitter_offset_centered(self) -> None:
        """Jitter offsets should be centered around 0."""
        upscaler = TSRLanczosUpscaler()

        for _ in range(8):
            jx, jy = upscaler.get_jitter_offset()
            assert -0.5 <= jx < 0.5
            assert -0.5 <= jy < 0.5
            upscaler.advance_frame()

    def test_halton_16_sequence(self) -> None:
        """Halton_16 jitter sequence should have 16 unique positions."""
        settings = TSRLanczosSettings(jitter_sequence="halton_16")
        upscaler = TSRLanczosUpscaler(settings)

        offsets = []
        for _ in range(16):
            offsets.append(upscaler.get_jitter_offset())
            upscaler.advance_frame()

        unique_offsets = set(offsets)
        assert len(unique_offsets) == 16


class TestTSRLanczosUpscalerSampling:
    """Tests for TSRLanczosUpscaler sampling methods."""

    @pytest.fixture
    def simple_image(self) -> List[List[Tuple[float, float, float]]]:
        """Create a simple 4x4 test image."""
        return [
            [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
            [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
        ]

    @pytest.fixture
    def uniform_image(self) -> List[List[Tuple[float, float, float]]]:
        """Create a uniform gray image."""
        gray = (0.5, 0.5, 0.5)
        return [[gray for _ in range(4)] for _ in range(4)]

    def test_separable_sampling_returns_color(
        self, simple_image: List[List[Tuple[float, float, float]]]
    ) -> None:
        """Separable sampling should return valid RGB color."""
        upscaler = TSRLanczosUpscaler()
        color = upscaler.sample_lanczos(simple_image, 0.5, 0.5)

        assert len(color) == 3
        assert all(isinstance(c, float) for c in color)

    def test_2d_sampling_returns_color(
        self, simple_image: List[List[Tuple[float, float, float]]]
    ) -> None:
        """2D sampling should return valid RGB color."""
        settings = TSRLanczosSettings(separable=False)
        upscaler = TSRLanczosUpscaler(settings)
        color = upscaler.sample_lanczos(simple_image, 0.5, 0.5)

        assert len(color) == 3
        assert all(isinstance(c, float) for c in color)

    def test_sampling_uniform_preserves_value(
        self, uniform_image: List[List[Tuple[float, float, float]]]
    ) -> None:
        """Sampling a uniform image should preserve the uniform value."""
        upscaler = TSRLanczosUpscaler()
        color = upscaler.sample_lanczos(uniform_image, 1.5, 1.5)

        assert color[0] == pytest.approx(0.5, rel=0.1)
        assert color[1] == pytest.approx(0.5, rel=0.1)
        assert color[2] == pytest.approx(0.5, rel=0.1)

    def test_separable_and_2d_produce_similar_results(
        self, simple_image: List[List[Tuple[float, float, float]]]
    ) -> None:
        """Separable and 2D sampling should produce similar results."""
        settings_sep = TSRLanczosSettings(separable=True)
        settings_2d = TSRLanczosSettings(separable=False)

        upscaler_sep = TSRLanczosUpscaler(settings_sep)
        upscaler_2d = TSRLanczosUpscaler(settings_2d)

        # Sample at multiple positions
        for x, y in [(1.0, 1.0), (2.0, 2.0), (1.5, 2.5)]:
            color_sep = upscaler_sep.sample_lanczos(simple_image, x, y)
            color_2d = upscaler_2d.sample_lanczos(simple_image, x, y)

            # Results should be reasonably close (separable is an approximation)
            # Note: They won't be identical due to algorithmic differences
            for i in range(3):
                assert abs(color_sep[i] - color_2d[i]) < 0.5


class TestTSRLanczosUpscalerSharpening:
    """Tests for TSRLanczosUpscaler sharpening."""

    def test_sharpening_disabled_returns_original(self) -> None:
        """Sharpening with sharpness=0 should return original color."""
        settings = TSRLanczosSettings(sharpness=0.0)
        upscaler = TSRLanczosUpscaler(settings)

        color = (0.5, 0.5, 0.5)
        neighbors = [(0.4, 0.4, 0.4), (0.6, 0.6, 0.6)]

        result = upscaler.apply_sharpening(color, neighbors)
        assert result == color

    def test_sharpening_increases_contrast(self) -> None:
        """Sharpening should increase contrast vs neighbors."""
        settings = TSRLanczosSettings(sharpness=0.5)
        upscaler = TSRLanczosUpscaler(settings)

        # Center is brighter than average of neighbors
        color = (0.7, 0.7, 0.7)
        neighbors = [(0.4, 0.4, 0.4), (0.4, 0.4, 0.4)]

        result = upscaler.apply_sharpening(color, neighbors)

        # Sharpened result should be even brighter (away from neighbor avg)
        assert result[0] > color[0]
        assert result[1] > color[1]
        assert result[2] > color[2]

    def test_sharpening_clamps_negative(self) -> None:
        """Sharpening should clamp negative values to 0."""
        settings = TSRLanczosSettings(sharpness=1.0)
        upscaler = TSRLanczosUpscaler(settings)

        # Center is darker than neighbors
        color = (0.1, 0.1, 0.1)
        neighbors = [(0.9, 0.9, 0.9), (0.9, 0.9, 0.9)]

        result = upscaler.apply_sharpening(color, neighbors)

        # Result might try to go negative but should be clamped
        assert result[0] >= 0.0
        assert result[1] >= 0.0
        assert result[2] >= 0.0

    def test_sharpening_with_empty_neighbors_returns_original(self) -> None:
        """Sharpening with no neighbors should return original."""
        settings = TSRLanczosSettings(sharpness=0.5)
        upscaler = TSRLanczosUpscaler(settings)

        color = (0.5, 0.5, 0.5)
        result = upscaler.apply_sharpening(color, [])

        assert result == color


class TestTSRLanczosUpscalerBudget:
    """Tests for TSRLanczosUpscaler performance budget estimation."""

    def test_separable_cheaper_than_2d(self) -> None:
        """Separable filter should have lower estimated cost."""
        settings_sep = TSRLanczosSettings(separable=True)
        settings_2d = TSRLanczosSettings(separable=False)

        upscaler_sep = TSRLanczosUpscaler(settings_sep)
        upscaler_2d = TSRLanczosUpscaler(settings_2d)

        assert upscaler_sep.get_budget_ms() < upscaler_2d.get_budget_ms()

    def test_larger_kernel_costs_more(self) -> None:
        """Lanczos-3 should cost more than Lanczos-2."""
        settings_l2 = TSRLanczosSettings(kernel=LanczosKernel.LANCZOS2)
        settings_l3 = TSRLanczosSettings(kernel=LanczosKernel.LANCZOS3)

        upscaler_l2 = TSRLanczosUpscaler(settings_l2)
        upscaler_l3 = TSRLanczosUpscaler(settings_l3)

        assert upscaler_l2.get_budget_ms() < upscaler_l3.get_budget_ms()

    def test_higher_scale_costs_more(self) -> None:
        """Higher scale factor should cost more."""
        settings_2x = TSRLanczosSettings(scale_factor=2.0)
        settings_3x = TSRLanczosSettings(scale_factor=3.0)

        upscaler_2x = TSRLanczosUpscaler(settings_2x)
        upscaler_3x = TSRLanczosUpscaler(settings_3x)

        assert upscaler_2x.get_budget_ms() < upscaler_3x.get_budget_ms()


class TestCreateTsrLanczos:
    """Tests for create_tsr_lanczos factory function."""

    def test_default_factory_settings(self) -> None:
        """Factory with defaults should create 2x Lanczos-2 with temporal."""
        upscaler = create_tsr_lanczos()

        assert upscaler.settings.scale_factor == 2.0
        assert upscaler.settings.kernel == LanczosKernel.LANCZOS2
        assert upscaler.settings.temporal_blend == 0.1

    def test_factory_custom_scale(self) -> None:
        """Factory should accept custom scale factor."""
        upscaler = create_tsr_lanczos(scale=1.5)
        assert upscaler.settings.scale_factor == 1.5

    def test_factory_custom_kernel(self) -> None:
        """Factory should accept custom kernel size."""
        upscaler = create_tsr_lanczos(kernel=LanczosKernel.LANCZOS3)
        assert upscaler.settings.kernel == LanczosKernel.LANCZOS3

    def test_factory_no_temporal(self) -> None:
        """Factory with temporal=False should disable temporal blending."""
        upscaler = create_tsr_lanczos(temporal=False)
        assert upscaler.settings.temporal_blend == 0.0


class TestLanczosKernelEnum:
    """Tests for LanczosKernel enum."""

    def test_lanczos2_value(self) -> None:
        """Lanczos-2 should have value 2."""
        assert LanczosKernel.LANCZOS2.value == 2

    def test_lanczos3_value(self) -> None:
        """Lanczos-3 should have value 3."""
        assert LanczosKernel.LANCZOS3.value == 3

    def test_enum_members(self) -> None:
        """Enum should have exactly 2 members."""
        assert len(LanczosKernel) == 2


class TestTSRLanczosReset:
    """Tests for TSRLanczosUpscaler reset functionality."""

    def test_reset_clears_frame_index(self) -> None:
        """Reset should clear frame index."""
        upscaler = TSRLanczosUpscaler()

        # Advance several frames
        for _ in range(10):
            upscaler.advance_frame()

        # Get jitter before reset
        jitter_before = upscaler.get_jitter_offset()

        upscaler.reset()

        # After reset, frame index should be 0, same as fresh upscaler
        fresh = TSRLanczosUpscaler()
        assert upscaler.get_jitter_offset() == fresh.get_jitter_offset()
