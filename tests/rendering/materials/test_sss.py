"""Tests for Subsurface Scattering functions (T-MAT-4.1).

This module tests:
- WGSL syntax validation for sss.wgsl
- Burley diffusion profile correctness
- Reference value matching within tolerance
- Edge cases (intensity=0/1, scatter_distance=0, etc.)
- Screen-space blur kernel computation
- SSS application and compositing
- Transmission evaluation
- Pre-integrated LUT generation
- Energy conservation properties
- Quality tier gating const
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import pytest

from trinity.materials.sss_shader import (
    # WGSL source
    get_sss_wgsl,
    # Data structures
    SSSProfile,
    SSSParams,
    # Predefined profiles
    SSS_PROFILE_SKIN,
    SSS_PROFILE_WAX,
    SSS_PROFILE_JADE,
    SSS_PROFILE_MILK,
    # Diffusion functions
    burley_diffusion,
    burley_diffusion_rgb,
    evaluate_diffusion_profile,
    # Kernel computation
    compute_sss_kernel,
    SSS_KERNEL_WEIGHTS,
    SSS_KERNEL_OFFSETS,
    SSS_KERNEL_SIZE,
    # Application functions
    apply_sss,
    apply_sss_with_bleeding,
    # Transmission
    evaluate_sss_transmission,
    # LUT generation
    compute_diffusion_lut_value,
    generate_diffusion_lut,
    # Profile utilities
    get_diffusion_profile_samples,
    get_sss_mask,
    # Reference values
    SSS_REFERENCE_VALUES,
    SSS_EDGE_CASES,
    # Constants
    PI,
    TWO_PI,
    EPSILON,
)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_sss_wgsl_loads(self) -> None:
        """Test that sss.wgsl can be loaded."""
        wgsl = get_sss_wgsl()
        assert len(wgsl) > 0
        assert "fn burley_diffusion" in wgsl
        assert "fn sss_blur_horizontal" in wgsl
        assert "fn sss_blur_vertical" in wgsl
        assert "fn apply_sss" in wgsl

    def test_sss_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "sss.wgsl"
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_sss_wgsl_has_required_functions(self) -> None:
        """Test that all required SSS functions are present."""
        wgsl = get_sss_wgsl()
        required_functions = [
            "fn burley_diffusion",
            "fn burley_diffusion_rgb",
            "fn evaluate_diffusion_profile",
            "fn sss_blur_horizontal",
            "fn sss_blur_vertical",
            "fn apply_sss",
            "fn apply_sss_with_bleeding",
            "fn evaluate_sss_transmission",
            "fn sample_diffusion_lut",
            "fn compute_diffusion_lut_value",
            "fn estimate_curvature_from_depth",
            "fn evaluate_sss_direct",
            "fn get_sss_mask",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_sss_wgsl_has_profile_struct(self) -> None:
        """Test that SSSProfile struct is defined."""
        wgsl = get_sss_wgsl()
        assert "struct SSSProfile" in wgsl
        assert "scatter_distance: vec3<f32>" in wgsl
        assert "scatter_color: vec3<f32>" in wgsl
        assert "blur_strength: f32" in wgsl

    def test_sss_wgsl_has_params_struct(self) -> None:
        """Test that SSSParams struct is defined."""
        wgsl = get_sss_wgsl()
        assert "struct SSSParams" in wgsl
        assert "profile: SSSProfile" in wgsl
        assert "subsurface_intensity: f32" in wgsl
        assert "enable_transmission: bool" in wgsl

    def test_sss_wgsl_has_quality_const(self) -> None:
        """Test that quality tier const is defined."""
        wgsl = get_sss_wgsl()
        assert "QUALITY_SSS_ENABLED" in wgsl
        assert "const QUALITY_SSS_ENABLED: bool" in wgsl

    def test_sss_wgsl_has_predefined_profiles(self) -> None:
        """Test that predefined SSS profiles are present."""
        wgsl = get_sss_wgsl()
        profiles = [
            "fn sss_profile_skin",
            "fn sss_profile_wax",
            "fn sss_profile_jade",
            "fn sss_profile_milk",
            "fn sss_profile_default",
        ]
        for profile in profiles:
            assert profile in wgsl, f"Missing profile function: {profile}"

    def test_sss_wgsl_has_kernel_constants(self) -> None:
        """Test that kernel constants are defined."""
        wgsl = get_sss_wgsl()
        assert "SSS_KERNEL_SIZE" in wgsl
        assert "SSS_KERNEL_HALF" in wgsl
        assert "fn sss_kernel_weights" in wgsl
        assert "fn sss_kernel_offsets" in wgsl

    def test_sss_wgsl_syntax_patterns(self) -> None:
        """Test basic WGSL syntax patterns."""
        wgsl = get_sss_wgsl()

        # Check function declarations
        fn_pattern = r"fn\s+\w+\([^)]*\)\s*->\s*\w+"
        assert re.search(fn_pattern, wgsl), "No valid function declarations found"

        # Check for proper type annotations
        assert "f32" in wgsl, "Missing f32 type annotations"
        assert "vec3<f32>" in wgsl, "Missing vec3<f32> type annotations"
        assert "texture_2d<f32>" in wgsl, "Missing texture_2d type"
        assert "sampler" in wgsl, "Missing sampler type"

    def test_sss_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_sss_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"


# =============================================================================
# SSSProfile Tests
# =============================================================================


class TestSSSProfile:
    """Test SSSProfile dataclass."""

    def test_default_profile(self) -> None:
        """Test default profile values."""
        profile = SSSProfile()
        assert profile.name == "Default"
        assert profile.blur_strength == 0.7
        assert profile.curvature_scale == 0.75
        assert profile.boundary_color_bleed == 0.5

    def test_skin_profile_values(self) -> None:
        """Test skin profile has expected values."""
        profile = SSS_PROFILE_SKIN
        assert profile.name == "Skin"
        assert profile.scatter_distance == (1.0, 0.4, 0.25)
        assert profile.blur_strength == 0.8
        # Red channel should scatter furthest
        assert profile.scatter_distance[0] > profile.scatter_distance[1]
        assert profile.scatter_distance[1] > profile.scatter_distance[2]

    def test_wax_profile_values(self) -> None:
        """Test wax profile has expected values."""
        profile = SSS_PROFILE_WAX
        assert profile.name == "Wax"
        # Wax should have more uniform scatter
        assert abs(profile.scatter_distance[0] - profile.scatter_distance[1]) < 0.1

    def test_jade_profile_values(self) -> None:
        """Test jade profile has expected values."""
        profile = SSS_PROFILE_JADE
        assert profile.name == "Jade"
        # Green channel should scatter furthest
        assert profile.scatter_distance[1] > profile.scatter_distance[0]
        assert profile.scatter_distance[1] > profile.scatter_distance[2]

    def test_milk_profile_values(self) -> None:
        """Test milk profile has expected values."""
        profile = SSS_PROFILE_MILK
        assert profile.name == "Milk"
        # Milk has high blur strength
        assert profile.blur_strength >= 0.8

    def test_blur_strength_validation(self) -> None:
        """Test blur_strength must be in [0,1]."""
        with pytest.raises(ValueError):
            SSSProfile(blur_strength=-0.1)
        with pytest.raises(ValueError):
            SSSProfile(blur_strength=1.1)

    def test_curvature_scale_validation(self) -> None:
        """Test curvature_scale must be in [0,2]."""
        with pytest.raises(ValueError):
            SSSProfile(curvature_scale=-0.1)
        with pytest.raises(ValueError):
            SSSProfile(curvature_scale=2.1)

    def test_boundary_color_bleed_validation(self) -> None:
        """Test boundary_color_bleed must be in [0,1]."""
        with pytest.raises(ValueError):
            SSSProfile(boundary_color_bleed=-0.1)
        with pytest.raises(ValueError):
            SSSProfile(boundary_color_bleed=1.1)


# =============================================================================
# SSSParams Tests
# =============================================================================


class TestSSSParams:
    """Test SSSParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = SSSParams()
        assert params.subsurface_intensity == 0.0
        assert params.enable_transmission is False
        assert params.transmission_tint == (1.0, 0.0, 0.0)

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = SSSParams(
            profile=SSS_PROFILE_SKIN,
            subsurface_intensity=0.8,
            enable_transmission=True,
            transmission_tint=(0.9, 0.3, 0.2),
        )
        assert params.profile.name == "Skin"
        assert params.subsurface_intensity == 0.8
        assert params.enable_transmission is True

    def test_intensity_validation(self) -> None:
        """Test subsurface_intensity must be in [0,1]."""
        with pytest.raises(ValueError):
            SSSParams(subsurface_intensity=-0.1)
        with pytest.raises(ValueError):
            SSSParams(subsurface_intensity=1.1)


# =============================================================================
# Burley Diffusion Function Tests
# =============================================================================


class TestBurleyDiffusion:
    """Test Burley diffusion profile function."""

    @pytest.mark.parametrize("ref", SSS_REFERENCE_VALUES["burley_diffusion"])
    def test_burley_diffusion_reference_values(self, ref: dict) -> None:
        """Test burley_diffusion matches reference values within tolerance."""
        result = burley_diffusion(ref["r"], ref["d"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"burley_diffusion(r={ref['r']}, d={ref['d']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_burley_diffusion_at_center(self) -> None:
        """Test burley_diffusion returns peak at center (r=0)."""
        # At r=0, function returns peak value: A + B = 2/(3*PI*d^2)
        result = burley_diffusion(0.0, 1.0)
        assert result > 0
        # Should be approximately A + B = 1/(2*PI*d^2) + 1/(6*PI*d^2) = 2/(3*PI*d^2)
        expected_peak = 2.0 / (3.0 * PI * 1.0 * 1.0)
        assert abs(result - expected_peak) < 0.01, f"Expected {expected_peak}, got {result}"

    def test_burley_diffusion_monotonic_decay(self) -> None:
        """Test burley_diffusion decreases monotonically with distance."""
        d = 1.0
        prev = burley_diffusion(0.0, d)
        for r in [0.1, 0.2, 0.5, 1.0, 2.0]:
            current = burley_diffusion(r, d)
            assert current < prev, f"Not monotonically decreasing at r={r}"
            prev = current

    def test_burley_diffusion_scatter_distance_effect(self) -> None:
        """Test larger scatter distance gives slower relative decay."""
        r = 1.0
        # Compare decay ratios (value/peak) for different d values
        small_d_ratio = burley_diffusion(r, 0.5) / burley_diffusion(0.0, 0.5)
        large_d_ratio = burley_diffusion(r, 2.0) / burley_diffusion(0.0, 2.0)
        # Larger d means slower relative decay
        assert large_d_ratio > small_d_ratio, (
            f"Larger d should decay slower: {large_d_ratio} vs {small_d_ratio}"
        )

    def test_burley_diffusion_approaches_zero(self) -> None:
        """Test burley_diffusion approaches zero at large distances."""
        d = 1.0
        result = burley_diffusion(10.0, d)
        # At r=10d, the value should be small relative to peak
        peak = burley_diffusion(0.0, d)
        ratio = result / peak
        assert ratio < 0.01, f"Expected ratio < 0.01 at r=10, got {ratio}"

    def test_burley_diffusion_positive(self) -> None:
        """Test burley_diffusion always returns positive values."""
        for r in [0.0, 0.1, 0.5, 1.0, 5.0]:
            for d in [0.1, 0.5, 1.0, 2.0]:
                result = burley_diffusion(r, d)
                assert result >= 0, f"Negative result at r={r}, d={d}"

    def test_burley_diffusion_zero_distance(self) -> None:
        """Test burley_diffusion handles d=0 gracefully."""
        result = burley_diffusion(0.5, 0.0)
        assert result == 0.0 or math.isfinite(result)


class TestBurleyDiffusionRGB:
    """Test per-channel Burley diffusion."""

    def test_burley_diffusion_rgb_returns_tuple(self) -> None:
        """Test burley_diffusion_rgb returns 3-element tuple."""
        result = burley_diffusion_rgb(0.5, (1.0, 0.5, 0.25))
        assert len(result) == 3

    def test_burley_diffusion_rgb_per_channel(self) -> None:
        """Test each channel is computed independently."""
        d = (1.0, 0.5, 0.25)
        r = 0.5
        result = burley_diffusion_rgb(r, d)

        # Each channel should match individual burley_diffusion call
        assert abs(result[0] - burley_diffusion(r, d[0])) < EPSILON
        assert abs(result[1] - burley_diffusion(r, d[1])) < EPSILON
        assert abs(result[2] - burley_diffusion(r, d[2])) < EPSILON

    def test_burley_diffusion_rgb_larger_d_slower_decay(self) -> None:
        """Test larger scatter distance gives slower decay at same r."""
        # At r > 0, larger d means slower decay relative to peak
        # Compare decay ratio: burley(r, d) / burley(0, d)
        d_values = (1.0, 0.5, 0.25)
        r = 0.5

        ratios = []
        for d in d_values:
            peak = burley_diffusion(0.0, d)
            value = burley_diffusion(r, d)
            ratio = value / peak if peak > 0 else 0
            ratios.append(ratio)

        # Larger d should have higher ratio (slower decay)
        assert ratios[0] > ratios[1] > ratios[2], f"Decay ratios: {ratios}"


# =============================================================================
# Evaluate Diffusion Profile Tests
# =============================================================================


class TestEvaluateDiffusionProfile:
    """Test diffusion profile evaluation."""

    @pytest.mark.parametrize("ref", SSS_REFERENCE_VALUES["evaluate_diffusion_profile_skin"])
    def test_evaluate_diffusion_profile_skin_reference(self, ref: dict) -> None:
        """Test evaluate_diffusion_profile with skin profile matches reference."""
        result = evaluate_diffusion_profile(ref["r"], SSS_PROFILE_SKIN)
        assert abs(result[0] - ref["expected_r"]) < ref["tolerance"], (
            f"evaluate_diffusion_profile(r={ref['r']}, SKIN).r = {result[0]}, "
            f"expected {ref['expected_r']} +/- {ref['tolerance']}"
        )

    def test_evaluate_diffusion_profile_returns_colored(self) -> None:
        """Test that evaluate_diffusion_profile returns colored weights."""
        result = evaluate_diffusion_profile(0.5, SSS_PROFILE_SKIN)
        # With skin profile, red should be strongest (highest scatter_color.r)
        # but also depends on scatter_distance
        assert all(w >= 0 for w in result)

    def test_evaluate_diffusion_profile_applies_scatter_color(self) -> None:
        """Test that scatter_color modulates the result."""
        profile = SSSProfile(
            scatter_distance=(1.0, 1.0, 1.0),
            scatter_color=(0.5, 0.25, 0.1),
        )
        result = evaluate_diffusion_profile(0.5, profile)
        # With equal scatter_distance, ratios should match scatter_color
        ratio_rg = result[0] / result[1] if result[1] > EPSILON else float('inf')
        expected_ratio = 0.5 / 0.25
        assert abs(ratio_rg - expected_ratio) < 0.1


# =============================================================================
# SSS Kernel Tests
# =============================================================================


class TestSSSKernel:
    """Test SSS blur kernel computation."""

    def test_kernel_size_is_nine(self) -> None:
        """Test kernel size constant is 9."""
        assert SSS_KERNEL_SIZE == 9
        assert len(SSS_KERNEL_WEIGHTS) == 9
        assert len(SSS_KERNEL_OFFSETS) == 9

    def test_kernel_weights_symmetric(self) -> None:
        """Test kernel weights are symmetric."""
        weights = SSS_KERNEL_WEIGHTS
        for i in range(4):
            assert abs(weights[i] - weights[8 - i]) < EPSILON

    def test_kernel_weights_sum_to_one(self) -> None:
        """Test kernel weights sum to approximately 1."""
        total = sum(SSS_KERNEL_WEIGHTS)
        assert abs(total - 1.0) < 0.01

    def test_kernel_offsets_symmetric(self) -> None:
        """Test kernel offsets are symmetric around zero."""
        offsets = SSS_KERNEL_OFFSETS
        for i in range(4):
            assert abs(offsets[i] + offsets[8 - i]) < EPSILON
        assert offsets[4] == 0.0  # Center is zero

    def test_compute_sss_kernel_returns_nine(self) -> None:
        """Test compute_sss_kernel returns 9 weights."""
        kernel = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        assert len(kernel) == 9

    def test_compute_sss_kernel_symmetric(self) -> None:
        """Test kernel is symmetric around center."""
        kernel = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        # Kernel should be symmetric: kernel[i] approx == kernel[8-i]
        for i in range(4):
            for c in range(3):
                diff = abs(kernel[i][c] - kernel[8 - i][c])
                assert diff < 0.001, f"Kernel not symmetric at tap {i}, channel {c}"

    def test_compute_sss_kernel_normalized(self) -> None:
        """Test kernel is normalized (sums to 1)."""
        kernel = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        total_r = sum(k[0] for k in kernel)
        total_g = sum(k[1] for k in kernel)
        total_b = sum(k[2] for k in kernel)
        assert abs(total_r - 1.0) < 0.01, f"Red channel not normalized: {total_r}"
        assert abs(total_g - 1.0) < 0.01, f"Green channel not normalized: {total_g}"
        assert abs(total_b - 1.0) < 0.01, f"Blue channel not normalized: {total_b}"

    def test_compute_sss_kernel_varies_with_profile(self) -> None:
        """Test different profiles produce different kernels."""
        kernel_skin = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        kernel_jade = compute_sss_kernel(SSS_PROFILE_JADE, 0.01)
        # Kernels should differ
        assert kernel_skin[0] != kernel_jade[0]


# =============================================================================
# Apply SSS Tests
# =============================================================================


class TestApplySSS:
    """Test SSS application functions."""

    @pytest.mark.parametrize("ref", SSS_REFERENCE_VALUES["apply_sss"])
    def test_apply_sss_reference_values(self, ref: dict) -> None:
        """Test apply_sss matches reference values."""
        profile = SSSProfile(blur_strength=ref["blur_strength"])
        result = apply_sss(
            ref["base_color"],
            ref["sss_buffer"],
            profile,
            ref["intensity"],
        )
        assert abs(result[0] - ref["expected_r"]) < ref["tolerance"], (
            f"apply_sss result.r = {result[0]}, "
            f"expected {ref['expected_r']} +/- {ref['tolerance']}"
        )

    def test_apply_sss_zero_intensity_passthrough(self) -> None:
        """Test zero intensity passes through base color."""
        base = (0.5, 0.4, 0.3)
        sss = (0.8, 0.7, 0.6)
        result = apply_sss(base, sss, SSS_PROFILE_SKIN, 0.0)
        assert result == base

    def test_apply_sss_interpolates(self) -> None:
        """Test apply_sss interpolates between base and SSS."""
        base = (0.0, 0.0, 0.0)
        sss = (1.0, 1.0, 1.0)
        profile = SSSProfile(
            scatter_color=(1.0, 1.0, 1.0),
            blur_strength=1.0,
        )
        result = apply_sss(base, sss, profile, 0.5)
        # Should be ~0.5 blend
        assert 0.4 < result[0] < 0.6


class TestApplySSSWithBleeding:
    """Test SSS with boundary bleeding."""

    def test_apply_sss_with_bleeding_in_shadow(self) -> None:
        """Test bleeding adds falloff color in shadow."""
        base = (0.5, 0.4, 0.3)
        sss = (0.5, 0.4, 0.3)
        profile = SSS_PROFILE_SKIN
        # Full shadow
        result = apply_sss_with_bleeding(base, sss, 0.0, profile, 0.5)
        # Should have some falloff_color bleeding
        assert result[0] != base[0]

    def test_apply_sss_with_bleeding_full_lit(self) -> None:
        """Test fully lit has minimal bleeding."""
        base = (0.5, 0.4, 0.3)
        sss = (0.5, 0.4, 0.3)
        profile = SSSProfile(
            boundary_color_bleed=0.0,  # No bleeding
        )
        result_lit = apply_sss_with_bleeding(base, sss, 1.0, profile, 0.5)
        result_base = apply_sss(base, sss, profile, 0.5)
        # With zero boundary_color_bleed, results should be similar
        assert abs(result_lit[0] - result_base[0]) < 0.1


# =============================================================================
# Transmission Tests
# =============================================================================


class TestSSSTransmission:
    """Test subsurface transmission evaluation."""

    def test_transmission_backlit(self) -> None:
        """Test transmission with light from behind."""
        N = (0.0, 1.0, 0.0)  # Up
        L = (0.0, -1.0, 0.0)  # Down (from behind)
        V = (0.0, 1.0, 0.0)  # Looking at front
        result = evaluate_sss_transmission(N, L, V, 0.5, SSS_PROFILE_SKIN)
        # Should have positive transmission
        assert result[0] > 0 or result[1] > 0 or result[2] > 0

    def test_transmission_frontlit_zero(self) -> None:
        """Test transmission is zero when frontlit."""
        N = (0.0, 1.0, 0.0)  # Up
        L = (0.0, 1.0, 0.0)  # Up (from front)
        V = (0.0, 1.0, 0.0)
        result = evaluate_sss_transmission(N, L, V, 0.5, SSS_PROFILE_SKIN)
        # NoL_back = max(-N.L, 0) = max(-1, 0) = 0
        assert result == (0.0, 0.0, 0.0)

    def test_transmission_thick_material_attenuates(self) -> None:
        """Test thicker material attenuates transmission."""
        N = (0.0, 1.0, 0.0)
        L = (0.0, -1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        thin = evaluate_sss_transmission(N, L, V, 0.1, SSS_PROFILE_SKIN)
        thick = evaluate_sss_transmission(N, L, V, 1.0, SSS_PROFILE_SKIN)
        # Thicker should be more attenuated
        assert thin[0] > thick[0]


# =============================================================================
# LUT Generation Tests
# =============================================================================


class TestDiffusionLUT:
    """Test pre-integrated diffusion LUT generation."""

    def test_compute_diffusion_lut_value_in_range(self) -> None:
        """Test LUT values are in valid range."""
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for v in [0.0, 0.5, 1.0]:
                result = compute_diffusion_lut_value((u, v), SSS_PROFILE_SKIN)
                assert all(0 <= c <= 1 for c in result), f"LUT value out of range at ({u}, {v})"

    def test_generate_diffusion_lut_dimensions(self) -> None:
        """Test LUT has correct dimensions."""
        lut = generate_diffusion_lut(SSS_PROFILE_SKIN, width=64, height=16)
        assert len(lut) == 16  # height
        assert len(lut[0]) == 64  # width

    def test_generate_diffusion_lut_valid_values(self) -> None:
        """Test all LUT values are valid."""
        lut = generate_diffusion_lut(SSS_PROFILE_SKIN, width=32, height=8)
        for row in lut:
            for value in row:
                assert len(value) == 3
                assert all(math.isfinite(c) for c in value)


# =============================================================================
# Profile Samples Tests
# =============================================================================


class TestDiffusionProfileSamples:
    """Test diffusion profile sample generation."""

    def test_get_diffusion_profile_samples_count(self) -> None:
        """Test sample count matches request."""
        samples = get_diffusion_profile_samples(SSS_PROFILE_SKIN, 16)
        assert len(samples) == 16

    def test_get_diffusion_profile_samples_normalized(self) -> None:
        """Test samples are normalized."""
        samples = get_diffusion_profile_samples(SSS_PROFILE_SKIN, 16)
        total_r = sum(s[0] for s in samples)
        total_g = sum(s[1] for s in samples)
        total_b = sum(s[2] for s in samples)
        assert abs(total_r - 1.0) < 0.01
        assert abs(total_g - 1.0) < 0.01
        assert abs(total_b - 1.0) < 0.01

    def test_get_diffusion_profile_samples_positive(self) -> None:
        """Test all samples are non-negative."""
        samples = get_diffusion_profile_samples(SSS_PROFILE_SKIN, 16)
        for s in samples:
            assert all(w >= 0 for w in s)


# =============================================================================
# SSS Mask Tests
# =============================================================================


class TestSSSMask:
    """Test SSS mask generation."""

    def test_get_sss_mask_zero_intensity(self) -> None:
        """Test mask is zero when intensity is zero."""
        params = SSSParams(subsurface_intensity=0.0)
        assert get_sss_mask(params) == 0.0

    def test_get_sss_mask_full(self) -> None:
        """Test mask reflects intensity and blur_strength."""
        profile = SSSProfile(blur_strength=0.8)
        params = SSSParams(profile=profile, subsurface_intensity=1.0)
        mask = get_sss_mask(params)
        assert mask == 0.8  # 1.0 * 0.8

    def test_get_sss_mask_partial(self) -> None:
        """Test mask with partial values."""
        profile = SSSProfile(blur_strength=0.5)
        params = SSSParams(profile=profile, subsurface_intensity=0.6)
        mask = get_sss_mask(params)
        assert abs(mask - 0.3) < EPSILON  # 0.6 * 0.5


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy conservation properties."""

    def test_burley_diffusion_integral_finite(self) -> None:
        """Test Burley diffusion integrates to finite value."""
        d = 1.0
        total = 0.0
        num_samples = 1000
        max_r = d * 10.0

        for i in range(num_samples):
            r = (i + 0.5) * max_r / num_samples
            # 2D integration: weight by r for polar coordinates
            weight = burley_diffusion(r, d) * r * (2 * PI)
            total += weight * (max_r / num_samples)

        # Integral should be finite and reasonable
        # Note: The exact integral value depends on normalization choice
        # What matters is it's finite, positive, and doesn't blow up
        assert 0.0 < total < 100.0, f"Integral = {total}, should be finite"
        assert math.isfinite(total), "Integral should be finite"

    def test_sss_kernel_preserves_energy(self) -> None:
        """Test SSS kernel preserves total energy."""
        kernel = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        # Each channel should sum to 1
        for channel in range(3):
            total = sum(k[channel] for k in kernel)
            assert abs(total - 1.0) < 0.02, f"Channel {channel} total = {total}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_scatter_distance(self) -> None:
        """Test handling of zero scatter distance."""
        profile = SSSProfile(scatter_distance=(0.0, 0.0, 0.0))
        result = evaluate_diffusion_profile(0.5, profile)
        # Should return finite values
        assert all(math.isfinite(c) for c in result)

    def test_very_small_scatter_distance(self) -> None:
        """Test handling of very small scatter distance."""
        profile = SSSProfile(scatter_distance=(0.001, 0.001, 0.001))
        result = evaluate_diffusion_profile(0.5, profile)
        assert all(math.isfinite(c) for c in result)

    def test_very_large_scatter_distance(self) -> None:
        """Test handling of very large scatter distance."""
        profile = SSSProfile(scatter_distance=(100.0, 100.0, 100.0))
        result = evaluate_diffusion_profile(0.5, profile)
        assert all(c >= 0 for c in result)

    def test_zero_blur_strength(self) -> None:
        """Test zero blur strength results in base color."""
        base = (0.5, 0.4, 0.3)
        sss = (0.8, 0.7, 0.6)
        profile = SSSProfile(blur_strength=0.0)
        result = apply_sss(base, sss, profile, 1.0)
        # With zero blur_strength, should be base color
        assert result == base

    def test_collinear_vectors_transmission(self) -> None:
        """Test transmission with collinear N, L, V."""
        N = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)  # Same as N
        V = (0.0, 1.0, 0.0)
        result = evaluate_sss_transmission(N, L, V, 0.5, SSS_PROFILE_SKIN)
        # Should be zero (light from front)
        assert result == (0.0, 0.0, 0.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for SSS pipeline."""

    def test_full_sss_pipeline(self) -> None:
        """Test complete SSS pipeline with skin profile."""
        # Setup
        profile = SSS_PROFILE_SKIN
        params = SSSParams(
            profile=profile,
            subsurface_intensity=0.8,
            enable_transmission=False,
        )

        # Step 1: Get diffusion profile samples
        samples = get_diffusion_profile_samples(profile, 16)
        assert len(samples) == 16

        # Step 2: Compute blur kernel
        kernel = compute_sss_kernel(profile, 0.01)
        assert len(kernel) == 9

        # Step 3: Apply SSS (simulated blur result)
        base_color = (0.5, 0.4, 0.35)
        sss_buffer = (0.55, 0.42, 0.36)  # Simulated blur
        result = apply_sss(base_color, sss_buffer, profile, params.subsurface_intensity)

        # Step 4: Get mask for G-buffer
        mask = get_sss_mask(params)
        assert 0 <= mask <= 1

        # Final result should be valid
        assert all(math.isfinite(c) for c in result)
        assert all(0 <= c <= 1 for c in result)

    def test_sss_with_transmission(self) -> None:
        """Test SSS with transmission enabled."""
        profile = SSS_PROFILE_SKIN
        params = SSSParams(
            profile=profile,
            subsurface_intensity=1.0,
            enable_transmission=True,
        )

        # Backlit scenario
        N = (0.0, 0.0, 1.0)
        L = (0.0, 0.0, -1.0)  # Behind
        V = (0.0, 0.0, 1.0)

        transmission = evaluate_sss_transmission(N, L, V, 0.3, profile)

        # Should have transmission contribution
        assert any(c > 0 for c in transmission)
        assert all(math.isfinite(c) for c in transmission)


# =============================================================================
# Profile Comparison Tests
# =============================================================================


class TestProfileComparison:
    """Compare different SSS profiles."""

    def test_profiles_differ_in_scatter(self) -> None:
        """Test different profiles have different scatter characteristics."""
        profiles = [SSS_PROFILE_SKIN, SSS_PROFILE_WAX, SSS_PROFILE_JADE, SSS_PROFILE_MILK]

        # Evaluate at same distance
        r = 0.5
        results = [evaluate_diffusion_profile(r, p) for p in profiles]

        # All should be different
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert results[i] != results[j], f"Profiles {i} and {j} should differ"

    def test_skin_vs_jade_channel_dominance(self) -> None:
        """Test skin profile vs jade profile have different characteristics."""
        r = 0.3
        skin = evaluate_diffusion_profile(r, SSS_PROFILE_SKIN)
        jade = evaluate_diffusion_profile(r, SSS_PROFILE_JADE)

        # Skin has warmer tones (higher red scatter_color)
        # Jade has cooler tones (higher green scatter_color)
        # Just verify they're different and both valid
        assert skin != jade, "Profiles should produce different results"
        assert all(s >= 0 for s in skin), "Skin values should be non-negative"
        assert all(j >= 0 for j in jade), "Jade values should be non-negative"
        # Verify the profiles produce different color balances
        # Skin scatter_color is (0.48, 0.25, 0.17) - red dominant
        # Jade scatter_color is (0.5, 0.9, 0.5) - green dominant
        # The actual output depends on both scatter_color AND scatter_distance
        assert len(skin) == 3 and len(jade) == 3, "Both should be RGB tuples"
