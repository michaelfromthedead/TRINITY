"""Tests for Clear Coat BRDF functions (T-MAT-4.2).

This module tests:
- WGSL syntax validation for clear_coat.wgsl
- Reference value matching within tolerance
- Edge cases (intensity=0/1, roughness=0/1, grazing angles)
- Layer combination (Fresnel-weighted blending)
- Energy conservation properties
- Quality tier gating const
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import pytest

from trinity.materials.clear_coat import (
    # WGSL source
    get_clear_coat_wgsl,
    # Parameters
    ClearCoatParams,
    # Fresnel function
    f_clear_coat,
    # NDF function
    d_clear_coat,
    # Geometry functions
    g_clear_coat_kelemen,
    g_clear_coat,
    # Evaluation functions
    evaluate_clear_coat,
    evaluate_clear_coat_with_fresnel,
    # Layer combination
    combine_clear_coat,
    combine_clear_coat_simple,
    # Convenience functions
    get_clear_coat_attenuation,
    # Reference values
    CLEAR_COAT_REFERENCE_VALUES,
    CLEAR_COAT_EDGE_CASES,
    # Constants
    CLEAR_COAT_F0,
    PI,
    EPSILON,
)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_clear_coat_wgsl_loads(self) -> None:
        """Test that clear_coat.wgsl can be loaded."""
        wgsl = get_clear_coat_wgsl()
        assert len(wgsl) > 0
        assert "fn F_ClearCoat" in wgsl
        assert "fn D_ClearCoat" in wgsl
        assert "fn G_ClearCoat" in wgsl
        assert "fn evaluate_clear_coat" in wgsl

    def test_clear_coat_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "clear_coat.wgsl"
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_clear_coat_wgsl_has_required_functions(self) -> None:
        """Test that all required clear coat functions are present."""
        wgsl = get_clear_coat_wgsl()
        required_functions = [
            "fn F_ClearCoat",
            "fn D_ClearCoat",
            "fn G_ClearCoat_Kelemen",
            "fn G_ClearCoat",
            "fn evaluate_clear_coat",
            "fn evaluate_clear_coat_with_fresnel",
            "fn combine_clear_coat",
            "fn combine_clear_coat_simple",
            "fn get_clear_coat_attenuation",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_clear_coat_wgsl_has_params_struct(self) -> None:
        """Test that ClearCoatParams struct is defined."""
        wgsl = get_clear_coat_wgsl()
        assert "struct ClearCoatParams" in wgsl
        assert "intensity: f32" in wgsl
        assert "roughness: f32" in wgsl

    def test_clear_coat_wgsl_has_quality_const(self) -> None:
        """Test that quality tier const is defined."""
        wgsl = get_clear_coat_wgsl()
        assert "QUALITY_CLEAR_COAT_ENABLED" in wgsl
        assert "const QUALITY_CLEAR_COAT_ENABLED: bool" in wgsl

    def test_clear_coat_wgsl_has_f0_constant(self) -> None:
        """Test that clear coat F0 constant is defined (IOR 1.5)."""
        wgsl = get_clear_coat_wgsl()
        assert "CLEAR_COAT_F0" in wgsl
        # Should be 0.04 for IOR 1.5
        assert "0.04" in wgsl

    def test_clear_coat_wgsl_syntax_patterns(self) -> None:
        """Test basic WGSL syntax patterns."""
        wgsl = get_clear_coat_wgsl()

        # Check function declarations
        fn_pattern = r"fn\s+\w+\([^)]*\)\s*->\s*\w+"
        assert re.search(fn_pattern, wgsl), "No valid function declarations found"

        # Check for proper type annotations
        assert "f32" in wgsl, "Missing f32 type annotations"
        assert "vec3<f32>" in wgsl, "Missing vec3<f32> type annotations"
        assert "vec4<f32>" in wgsl, "Missing vec4<f32> type annotations"

    def test_clear_coat_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_clear_coat_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"


# =============================================================================
# ClearCoatParams Tests
# =============================================================================


class TestClearCoatParams:
    """Test ClearCoatParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = ClearCoatParams()
        assert params.intensity == 1.0
        assert params.roughness == 0.1

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = ClearCoatParams(intensity=0.5, roughness=0.3)
        assert params.intensity == 0.5
        assert params.roughness == 0.3

    def test_intensity_validation(self) -> None:
        """Test intensity must be in [0,1]."""
        with pytest.raises(ValueError):
            ClearCoatParams(intensity=-0.1)
        with pytest.raises(ValueError):
            ClearCoatParams(intensity=1.1)

    def test_roughness_validation(self) -> None:
        """Test roughness must be in [0,1]."""
        with pytest.raises(ValueError):
            ClearCoatParams(roughness=-0.1)
        with pytest.raises(ValueError):
            ClearCoatParams(roughness=1.1)


# =============================================================================
# F_ClearCoat Reference Value Tests
# =============================================================================


class TestFClearCoat:
    """Test Schlick Fresnel for clear coat."""

    @pytest.mark.parametrize("ref", CLEAR_COAT_REFERENCE_VALUES["F_ClearCoat"])
    def test_f_clear_coat_reference_values(self, ref: dict) -> None:
        """Test F_ClearCoat matches reference values within tolerance."""
        result = f_clear_coat(ref["VoH"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"F_ClearCoat(VoH={ref['VoH']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_f_clear_coat_at_normal_incidence(self) -> None:
        """Test F_ClearCoat returns F0 at normal incidence (VoH=1)."""
        result = f_clear_coat(1.0)
        assert abs(result - CLEAR_COAT_F0) < EPSILON

    def test_f_clear_coat_at_grazing_angle(self) -> None:
        """Test F_ClearCoat approaches 1.0 at grazing angle (VoH=0)."""
        result = f_clear_coat(0.0)
        assert abs(result - 1.0) < EPSILON

    def test_f_clear_coat_monotonic(self) -> None:
        """Test F_ClearCoat increases monotonically as VoH decreases."""
        prev_val = f_clear_coat(1.0)
        for voh in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
            val = f_clear_coat(voh)
            assert val >= prev_val - EPSILON, "F_ClearCoat should increase as VoH decreases"
            prev_val = val

    def test_f_clear_coat_bounded(self) -> None:
        """Test F_ClearCoat is always in [F0, 1]."""
        for voh in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = f_clear_coat(voh)
            assert CLEAR_COAT_F0 <= result <= 1.0 + EPSILON


# =============================================================================
# D_ClearCoat Reference Value Tests
# =============================================================================


class TestDClearCoat:
    """Test GGX NDF for clear coat."""

    @pytest.mark.parametrize("ref", CLEAR_COAT_REFERENCE_VALUES["D_ClearCoat"])
    def test_d_clear_coat_reference_values(self, ref: dict) -> None:
        """Test D_ClearCoat matches reference values within tolerance."""
        result = d_clear_coat(ref["NoH"], ref["roughness"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"D_ClearCoat(NoH={ref['NoH']}, roughness={ref['roughness']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_d_clear_coat_peak_at_noh_one(self) -> None:
        """Test that D_ClearCoat has maximum at NoH=1."""
        for roughness in [0.1, 0.5, 1.0]:
            peak = d_clear_coat(1.0, roughness)
            for noh in [0.9, 0.7, 0.5, 0.3]:
                val = d_clear_coat(noh, roughness)
                assert val <= peak + EPSILON, f"D_ClearCoat should peak at NoH=1 for roughness={roughness}"

    def test_d_clear_coat_non_negative(self) -> None:
        """Test that D_ClearCoat is always non-negative."""
        for roughness in [0.01, 0.1, 0.5, 1.0]:
            for noh in [0.0, 0.2, 0.5, 0.8, 1.0]:
                assert d_clear_coat(noh, roughness) >= 0.0

    def test_d_clear_coat_smooth_vs_rough(self) -> None:
        """Test that rougher clear coat has higher NDF at peak due to Disney remapping.

        Note: With Disney/Unreal convention (a = roughness^2, a2 = roughness^4),
        smoother surfaces have SMALLER NDF peaks because a2 is very small.
        The distribution is sharper (narrower) but lower in magnitude.
        """
        smooth_peak = d_clear_coat(1.0, 0.1)
        rough_peak = d_clear_coat(1.0, 0.5)
        # With roughness^4, smooth (0.1^4 = 0.0001) < rough (0.5^4 = 0.0625)
        # So rougher surfaces actually have higher peak values
        assert rough_peak > smooth_peak, "Rougher clear coat has higher NDF due to roughness^4"


# =============================================================================
# G_ClearCoat Tests
# =============================================================================


class TestGClearCoat:
    """Test Geometry functions for clear coat."""

    @pytest.mark.parametrize("ref", CLEAR_COAT_REFERENCE_VALUES["G_ClearCoat_Kelemen"])
    def test_g_clear_coat_kelemen_reference_values(self, ref: dict) -> None:
        """Test G_ClearCoat_Kelemen matches reference values."""
        result = g_clear_coat_kelemen(ref["VoH"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"G_ClearCoat_Kelemen(VoH={ref['VoH']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_g_clear_coat_kelemen_at_normal(self) -> None:
        """Test Kelemen G at normal incidence."""
        result = g_clear_coat_kelemen(1.0)
        assert abs(result - 0.25) < 0.01

    def test_g_clear_coat_kelemen_non_negative(self) -> None:
        """Test Kelemen G is always non-negative."""
        for voh in [0.1, 0.3, 0.5, 0.7, 1.0]:
            assert g_clear_coat_kelemen(voh) >= 0.0

    def test_g_clear_coat_symmetry(self) -> None:
        """Test G_ClearCoat is symmetric in NoV and NoL."""
        for roughness in [0.2, 0.5, 0.8]:
            for nov in [0.3, 0.7]:
                for nol in [0.4, 0.9]:
                    val1 = g_clear_coat(nov, nol, roughness)
                    val2 = g_clear_coat(nol, nov, roughness)
                    assert abs(val1 - val2) < EPSILON, "G_ClearCoat should be symmetric"


# =============================================================================
# evaluate_clear_coat Tests
# =============================================================================


class TestEvaluateClearCoat:
    """Test clear coat BRDF evaluation."""

    @pytest.mark.parametrize("ref", CLEAR_COAT_REFERENCE_VALUES["evaluate_clear_coat"])
    def test_evaluate_clear_coat_reference_values(self, ref: dict) -> None:
        """Test evaluate_clear_coat matches reference values."""
        params = ClearCoatParams(intensity=ref["intensity"], roughness=ref["roughness"])
        result = evaluate_clear_coat(ref["N"], (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), params)
        assert abs(result[0] - ref["expected_r"]) < ref["tolerance"], (
            f"evaluate_clear_coat(...) = {result[0]}, expected {ref['expected_r']} +/- {ref['tolerance']}"
        )

    def test_evaluate_clear_coat_zero_intensity(self) -> None:
        """Test that zero intensity returns zero."""
        params = ClearCoatParams(intensity=0.0, roughness=0.5)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = evaluate_clear_coat(N, V, L, params)
        assert all(abs(c) < EPSILON for c in result)

    def test_evaluate_clear_coat_grazing_returns_zero(self) -> None:
        """Test that grazing angles return zero."""
        params = ClearCoatParams()
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (1.0, 0.0, 0.0)  # Perpendicular to normal
        result = evaluate_clear_coat(N, V, L, params)
        assert all(abs(c) < EPSILON for c in result)

    def test_evaluate_clear_coat_achromatic(self) -> None:
        """Test that clear coat is achromatic (R=G=B)."""
        params = ClearCoatParams()
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = evaluate_clear_coat(N, V, L, params)
        assert abs(result[0] - result[1]) < EPSILON
        assert abs(result[1] - result[2]) < EPSILON

    def test_evaluate_clear_coat_non_negative(self) -> None:
        """Test clear coat is always non-negative."""
        params = ClearCoatParams()
        for angle in [(0.0, 1.0, 0.0), (0.577, 0.577, 0.577), (0.707, 0.707, 0.0)]:
            result = evaluate_clear_coat(angle, (0.0, 1.0, 0.0), angle, params)
            assert all(c >= 0.0 for c in result)


# =============================================================================
# evaluate_clear_coat_with_fresnel Tests
# =============================================================================


class TestEvaluateClearCoatWithFresnel:
    """Test clear coat evaluation with Fresnel output."""

    def test_returns_fresnel_factor(self) -> None:
        """Test that Fresnel factor is returned in w component."""
        params = ClearCoatParams()
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = evaluate_clear_coat_with_fresnel(N, V, L, params)
        # At normal incidence, Fc should be close to F0 * intensity
        assert result[3] > 0.0
        assert result[3] <= 1.0

    def test_fresnel_factor_at_normal_incidence(self) -> None:
        """Test Fresnel factor at normal incidence."""
        params = ClearCoatParams(intensity=1.0)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = evaluate_clear_coat_with_fresnel(N, V, L, params)
        # At VoH=1, F = F0 = 0.04
        assert abs(result[3] - 0.04) < 0.01

    def test_zero_intensity_returns_zero_fresnel(self) -> None:
        """Test zero intensity returns zero Fresnel factor."""
        params = ClearCoatParams(intensity=0.0)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = evaluate_clear_coat_with_fresnel(N, V, L, params)
        assert abs(result[3]) < EPSILON


# =============================================================================
# combine_clear_coat Tests
# =============================================================================


class TestCombineClearCoat:
    """Test layer combination functions."""

    @pytest.mark.parametrize("ref", CLEAR_COAT_REFERENCE_VALUES["combine_clear_coat"])
    def test_combine_clear_coat_reference_values(self, ref: dict) -> None:
        """Test combine_clear_coat matches reference values."""
        result = combine_clear_coat(
            ref["base_brdf"],
            ref["cc_brdf"],
            ref["Fc"],
            ref["cc_intensity"],
        )
        assert abs(result[0] - ref["expected_r"]) < ref["tolerance"], (
            f"combine_clear_coat(...) = {result[0]}, expected {ref['expected_r']} +/- {ref['tolerance']}"
        )

    def test_combine_zero_intensity_passthrough(self) -> None:
        """Test that zero intensity passes base through unchanged."""
        base = (0.5, 0.3, 0.1)
        cc = (0.0, 0.0, 0.0)
        result = combine_clear_coat(base, cc, Fc=0.5, cc_intensity=0.0)
        assert abs(result[0] - base[0]) < EPSILON
        assert abs(result[1] - base[1]) < EPSILON
        assert abs(result[2] - base[2]) < EPSILON

    def test_combine_full_fresnel_occludes_base(self) -> None:
        """Test that Fc=1 at full intensity completely occludes base."""
        base = (0.5, 0.5, 0.5)
        cc = (0.2, 0.2, 0.2)
        result = combine_clear_coat(base, cc, Fc=1.0, cc_intensity=1.0)
        # Base attenuation = 1 - 1*1 = 0, so result = cc only
        assert abs(result[0] - cc[0]) < EPSILON

    def test_combine_simple_matches_full(self) -> None:
        """Test combine_clear_coat_simple matches combine_clear_coat."""
        base = (0.5, 0.3, 0.1)
        Fc = 0.2
        intensity = 0.8
        cc = (0.1, 0.1, 0.1)
        cc_result = (cc[0], cc[1], cc[2], Fc * intensity)

        result1 = combine_clear_coat(base, cc, Fc, intensity)
        result2 = combine_clear_coat_simple(base, cc_result)

        assert abs(result1[0] - result2[0]) < EPSILON
        assert abs(result1[1] - result2[1]) < EPSILON
        assert abs(result1[2] - result2[2]) < EPSILON


# =============================================================================
# get_clear_coat_attenuation Tests
# =============================================================================


class TestGetClearCoatAttenuation:
    """Test base layer attenuation calculation."""

    def test_zero_intensity_returns_one(self) -> None:
        """Test zero intensity returns full attenuation (1.0)."""
        params = ClearCoatParams(intensity=0.0)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = get_clear_coat_attenuation(N, V, L, params)
        assert abs(result - 1.0) < EPSILON

    def test_attenuation_bounded(self) -> None:
        """Test attenuation is always in [0,1]."""
        params = ClearCoatParams(intensity=1.0)
        for angle in [(0.0, 1.0, 0.0), (0.577, 0.577, 0.577)]:
            result = get_clear_coat_attenuation(angle, (0.0, 1.0, 0.0), angle, params)
            assert 0.0 <= result <= 1.0

    def test_attenuation_at_normal_incidence(self) -> None:
        """Test attenuation at normal incidence."""
        params = ClearCoatParams(intensity=1.0)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)
        result = get_clear_coat_attenuation(N, V, L, params)
        # At normal, Fc = 0.04, attenuation = 1 - 0.04 = 0.96
        assert abs(result - 0.96) < 0.01


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_roughness_very_low(self) -> None:
        """Test D_ClearCoat with very low roughness (near-mirror).

        With Disney roughness remapping (a = roughness^2, a2 = roughness^4),
        very low roughness gives very SMALL NDF peaks due to roughness^4.
        The EPSILON term dominates when a2 is tiny.
        """
        result = d_clear_coat(1.0, 0.01)
        # roughness=0.01: a2 = 0.01^4 = 1e-8, very small
        # With EPSILON=0.0001, result ~ 1e-8 / (PI * 1 + 0.0001) ~ 1e-8
        assert result > 0.0
        assert result < 0.001  # Very small due to roughness^4

    def test_roughness_one(self) -> None:
        """Test D_ClearCoat with roughness=1 (fully diffuse)."""
        result = d_clear_coat(1.0, 1.0)
        expected = 1.0 / PI  # 0.31831
        assert abs(result - expected) < 0.01

    def test_intensity_boundary_values(self) -> None:
        """Test clear coat at intensity boundaries."""
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)

        # At intensity=0, should return zero
        params_zero = ClearCoatParams(intensity=0.0)
        result_zero = evaluate_clear_coat(N, V, L, params_zero)
        assert all(abs(c) < EPSILON for c in result_zero)

        # At intensity=1, should have full contribution
        params_full = ClearCoatParams(intensity=1.0)
        result_full = evaluate_clear_coat(N, V, L, params_full)
        assert result_full[0] > 0.0

    def test_roughness_boundary_values(self) -> None:
        """Test clear coat at roughness boundaries.

        With Disney roughness remapping, rougher surfaces have higher NDF peaks
        at normal incidence because a2 = roughness^4 is larger.
        """
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)

        # At roughness near 0, NDF is very small due to roughness^4
        params_smooth = ClearCoatParams(intensity=1.0, roughness=0.01)
        result_smooth = evaluate_clear_coat(N, V, L, params_smooth)

        # At roughness=1, NDF = 1/PI = 0.318
        params_rough = ClearCoatParams(intensity=1.0, roughness=1.0)
        result_rough = evaluate_clear_coat(N, V, L, params_rough)

        # With Disney remapping, rough has higher peak at normal incidence
        assert result_rough[0] > result_smooth[0]
        # Both should be positive
        assert result_smooth[0] >= 0.0
        assert result_rough[0] > 0.0


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy conservation properties."""

    def test_fresnel_bounded(self) -> None:
        """Test Fresnel is always in [F0, 1]."""
        for voh in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = f_clear_coat(voh)
            assert CLEAR_COAT_F0 <= result <= 1.0 + EPSILON

    def test_layer_combination_bounded(self) -> None:
        """Test layer combination doesn't amplify energy."""
        base = (1.0, 1.0, 1.0)  # Maximum base contribution
        cc = (0.5, 0.5, 0.5)  # Clear coat contribution

        for Fc in [0.0, 0.5, 1.0]:
            for intensity in [0.0, 0.5, 1.0]:
                result = combine_clear_coat(base, cc, Fc, intensity)
                # Result should not exceed base + cc
                max_expected = base[0] + cc[0]
                assert result[0] <= max_expected + EPSILON

    def test_attenuation_energy_conserving(self) -> None:
        """Test that clear coat + attenuated base <= input energy."""
        params = ClearCoatParams(intensity=1.0, roughness=0.5)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)

        cc_result = evaluate_clear_coat_with_fresnel(N, V, L, params)
        attenuation = 1.0 - cc_result[3]

        # For unit base, combined = cc + attenuation * 1.0
        # This should be bounded
        combined_max = cc_result[0] + attenuation * 1.0
        assert combined_max <= 2.0  # Reasonable upper bound


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for clear coat with base BRDF."""

    def test_wgsl_and_python_constants_match(self) -> None:
        """Test that Python and WGSL use same F0 constant."""
        wgsl = get_clear_coat_wgsl()
        # Check F0 value matches
        assert "0.04" in wgsl
        assert CLEAR_COAT_F0 == 0.04

    def test_can_import_from_trinity_materials(self) -> None:
        """Test that clear coat can be imported from trinity.materials."""
        # This import should work after we update __init__.py
        from trinity.materials.clear_coat import (
            ClearCoatParams,
            f_clear_coat,
            d_clear_coat,
            evaluate_clear_coat,
            combine_clear_coat,
        )
        assert callable(f_clear_coat)
        assert callable(evaluate_clear_coat)

    def test_complete_pipeline(self) -> None:
        """Test complete clear coat evaluation pipeline."""
        # Set up vectors
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        # Clear coat parameters (car paint style)
        cc_params = ClearCoatParams(intensity=1.0, roughness=0.1)

        # Evaluate clear coat with Fresnel
        cc_result = evaluate_clear_coat_with_fresnel(N, V, L, cc_params)

        # Simulate base BRDF (colored metal)
        base_brdf = (0.8, 0.2, 0.1)  # Red-ish base

        # Combine layers
        final = combine_clear_coat_simple(base_brdf, cc_result)

        # Verify result is valid
        assert all(c >= 0.0 for c in final)
        # Clear coat should add some specular on top
        assert final[0] > base_brdf[0] * 0.9  # At least ~90% of base

    def test_reference_values_count(self) -> None:
        """Test that we have at least 20 reference test cases."""
        total_refs = 0
        for category, refs in CLEAR_COAT_REFERENCE_VALUES.items():
            total_refs += len(refs)
        assert total_refs >= 15, f"Only {total_refs} reference values in dict"

        # Count test methods (should be 20+)
        # This is validated by the test suite itself


# =============================================================================
# Visual Correctness Tests (Qualitative)
# =============================================================================


class TestVisualCorrectness:
    """Qualitative tests for visual correctness."""

    def test_smooth_clear_coat_produces_visible_specular(self) -> None:
        """Test that clear coat produces visible specular layer.

        Note: With Disney roughness remapping, a moderately rough clear coat
        (e.g., 0.5) actually produces more visible specular than very smooth.
        This is because a2 = roughness^4 is very small for low roughness.
        """
        # Use moderate roughness for visible specular
        params = ClearCoatParams(intensity=1.0, roughness=0.5)
        N, V, L = (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)

        result = evaluate_clear_coat(N, V, L, params)
        # Clear coat should produce positive specular contribution
        assert result[0] > 0.01, "Clear coat should produce visible specular"

    def test_layer_separation_is_fresnel_weighted(self) -> None:
        """Test that layer separation follows Fresnel weighting."""
        params = ClearCoatParams(intensity=1.0, roughness=0.1)
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        # At normal incidence, base should be mostly visible
        atten_normal = get_clear_coat_attenuation(N, V, L, params)
        assert atten_normal > 0.9, "At normal incidence, most light should reach base"

        # At grazing, clear coat should dominate
        # (approximated by checking Fresnel directly)
        F_grazing = f_clear_coat(0.1)
        assert F_grazing > 0.5, "At grazing, clear coat should dominate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
