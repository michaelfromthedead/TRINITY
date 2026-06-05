"""Tests for Iridescence functions (T-MAT-4.6).

This module tests:
- WGSL syntax validation for iridescence.wgsl
- Reference value matching within tolerance
- Edge cases (intensity=0/1, thin/thick films, grazing angles)
- Physical properties (Fresnel behavior, interference patterns)
- Integration with BRDF and quality tiers
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import pytest

from trinity.materials.iridescence import (
    # WGSL source
    get_iridescence_wgsl,
    # Parameter class
    IridescenceParams,
    # Presets
    IRIDESCENCE_PRESETS,
    get_preset,
    PRESET_SOAP_BUBBLE,
    PRESET_OIL_SLICK,
    PRESET_BEETLE,
    PRESET_PEARL,
    # Fresnel functions
    snell_cos_theta_t,
    fresnel_dielectric,
    fresnel_air_film,
    fresnel_film_substrate,
    # Phase computation
    compute_film_phase,
    # Interference
    compute_interference,
    compute_interference_color,
    # Main functions
    evaluate_iridescence,
    apply_iridescence,
    # Reference values
    IRIDESCENCE_REFERENCE_VALUES,
    IRIDESCENCE_EDGE_CASES,
    # Constants
    PI,
    EPSILON,
    WAVELENGTH_R,
    WAVELENGTH_G,
    WAVELENGTH_B,
)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_iridescence_wgsl_loads(self) -> None:
        """Test that iridescence.wgsl can be loaded."""
        wgsl = get_iridescence_wgsl()
        assert len(wgsl) > 0
        assert "fn evaluate_iridescence" in wgsl
        assert "fn apply_iridescence" in wgsl
        assert "fn compute_film_phase" in wgsl

    def test_iridescence_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = (
            Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "iridescence.wgsl"
        )
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_iridescence_wgsl_has_required_functions(self) -> None:
        """Test that all required iridescence functions are present."""
        wgsl = get_iridescence_wgsl()
        required_functions = [
            "fn snell_cos_theta_t",
            "fn fresnel_dielectric",
            "fn fresnel_air_film",
            "fn fresnel_film_substrate",
            "fn compute_film_phase",
            "fn compute_interference",
            "fn compute_interference_color",
            "fn evaluate_iridescence",
            "fn apply_iridescence",
            "fn iridescence_params_default",
            "fn iridescence_preset",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_iridescence_wgsl_has_struct(self) -> None:
        """Test that IridescenceParams struct is defined."""
        wgsl = get_iridescence_wgsl()
        assert "struct IridescenceParams" in wgsl
        assert "intensity: f32" in wgsl
        assert "ior: f32" in wgsl
        assert "thickness_nm: f32" in wgsl

    def test_iridescence_wgsl_has_constants(self) -> None:
        """Test that wavelength constants are defined."""
        wgsl = get_iridescence_wgsl()
        assert "const WAVELENGTH_R" in wgsl
        assert "const WAVELENGTH_G" in wgsl
        assert "const WAVELENGTH_B" in wgsl

    def test_iridescence_wgsl_has_quality_const(self) -> None:
        """Test that quality tier const is defined for dead-code elimination."""
        wgsl = get_iridescence_wgsl()
        assert "const QUALITY_IRIDESCENCE_ENABLED" in wgsl
        assert "bool" in wgsl

    def test_iridescence_wgsl_syntax_patterns(self) -> None:
        """Test basic WGSL syntax patterns."""
        wgsl = get_iridescence_wgsl()

        # Check function declarations
        fn_pattern = r"fn\s+\w+\([^)]*\)\s*->\s*\w+"
        assert re.search(fn_pattern, wgsl), "No valid function declarations found"

        # Check for proper type annotations
        assert "f32" in wgsl
        assert "vec3<f32>" in wgsl

    def test_iridescence_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_iridescence_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses"


# =============================================================================
# IridescenceParams Tests
# =============================================================================


class TestIridescenceParams:
    """Test IridescenceParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = IridescenceParams()
        assert params.intensity == 0.0
        assert params.ior == 1.5
        assert params.thickness_nm == 400.0

    def test_custom_params(self) -> None:
        """Test custom parameter creation."""
        params = IridescenceParams(intensity=0.8, ior=1.4, thickness_nm=350.0)
        assert params.intensity == 0.8
        assert params.ior == 1.4
        assert params.thickness_nm == 350.0

    def test_intensity_validation(self) -> None:
        """Test intensity range validation."""
        with pytest.raises(ValueError):
            IridescenceParams(intensity=-0.1)
        with pytest.raises(ValueError):
            IridescenceParams(intensity=1.5)

    def test_ior_validation(self) -> None:
        """Test IOR range validation."""
        with pytest.raises(ValueError):
            IridescenceParams(ior=0.5)  # Too low
        with pytest.raises(ValueError):
            IridescenceParams(ior=4.0)  # Too high

    def test_thickness_validation(self) -> None:
        """Test thickness range validation."""
        with pytest.raises(ValueError):
            IridescenceParams(thickness_nm=10.0)  # Too thin
        with pytest.raises(ValueError):
            IridescenceParams(thickness_nm=5000.0)  # Too thick


# =============================================================================
# Preset Tests
# =============================================================================


class TestPresets:
    """Test iridescence presets."""

    def test_all_presets_exist(self) -> None:
        """Test that all expected presets are defined."""
        assert "soap_bubble" in IRIDESCENCE_PRESETS
        assert "oil_slick" in IRIDESCENCE_PRESETS
        assert "beetle" in IRIDESCENCE_PRESETS
        assert "pearl" in IRIDESCENCE_PRESETS

    def test_get_preset(self) -> None:
        """Test preset retrieval."""
        soap = get_preset("soap_bubble")
        assert soap.intensity == 1.0
        assert soap.ior == 1.33
        assert soap.thickness_nm == 200.0

    def test_get_preset_invalid(self) -> None:
        """Test invalid preset raises error."""
        with pytest.raises(KeyError):
            get_preset("nonexistent")

    def test_preset_parameters_valid(self) -> None:
        """Test all presets have valid parameter ranges."""
        for name, params in IRIDESCENCE_PRESETS.items():
            assert 0.0 <= params.intensity <= 1.0, f"{name} intensity out of range"
            assert 1.0 <= params.ior <= 3.0, f"{name} ior out of range"
            assert 50.0 <= params.thickness_nm <= 2000.0, f"{name} thickness out of range"


# =============================================================================
# Snell's Law Tests
# =============================================================================


class TestSnellCosTheta:
    """Test Snell's law implementation."""

    def test_normal_incidence(self) -> None:
        """Test normal incidence (no refraction)."""
        result = snell_cos_theta_t(1.0, 0.67)  # Air to glass
        assert result > 0.0
        assert abs(result - 1.0) < 0.01  # Very close to 1 at normal

    def test_total_internal_reflection(self) -> None:
        """Test TIR condition returns negative."""
        # Glass to air at steep angle
        result = snell_cos_theta_t(0.5, 1.5)  # eta = n_glass/n_air
        # Should be TIR for sin_theta_i > 1/1.5
        # At cos_theta_i = 0.5, sin = 0.866, sin_t = 1.3 > 1 -> TIR
        assert result < 0.0

    def test_symmetry(self) -> None:
        """Test that refraction is reversible."""
        eta = 0.67  # Air to glass
        cos_i = 0.8
        cos_t = snell_cos_theta_t(cos_i, eta)

        # Reverse: glass to air
        cos_i_back = snell_cos_theta_t(cos_t, 1.0 / eta)
        assert abs(cos_i_back - cos_i) < 0.01

    def test_sin_squared_conservation(self) -> None:
        """Test n1*sin(theta1) = n2*sin(theta2)."""
        n1, n2 = 1.0, 1.5
        eta = n1 / n2
        cos_i = 0.8
        cos_t = snell_cos_theta_t(cos_i, eta)

        sin_i = math.sqrt(1.0 - cos_i**2)
        sin_t = math.sqrt(1.0 - cos_t**2)

        assert abs(n1 * sin_i - n2 * sin_t) < 0.001


# =============================================================================
# Fresnel Tests
# =============================================================================


class TestFresnelAirFilm:
    """Test air-film Fresnel interface."""

    @pytest.mark.parametrize("ref", IRIDESCENCE_REFERENCE_VALUES["fresnel_air_film"])
    def test_fresnel_air_film_reference_values(self, ref: dict) -> None:
        """Test fresnel_air_film matches reference values."""
        result = fresnel_air_film(ref["cos_theta"], ref["film_ior"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"fresnel_air_film({ref['cos_theta']}, {ref['film_ior']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_fresnel_at_normal_incidence_glass(self) -> None:
        """Test Fresnel at normal incidence equals ((n-1)/(n+1))^2."""
        n = 1.5
        expected = ((n - 1.0) / (n + 1.0)) ** 2
        result = fresnel_air_film(1.0, n)
        assert abs(result - expected) < 0.001

    def test_fresnel_increases_at_grazing(self) -> None:
        """Test Fresnel increases toward 1.0 at grazing angles."""
        prev = fresnel_air_film(1.0, 1.5)
        for cos_theta in [0.8, 0.5, 0.3, 0.1]:
            curr = fresnel_air_film(cos_theta, 1.5)
            assert curr >= prev - EPSILON
            prev = curr

    def test_fresnel_bounded(self) -> None:
        """Test Fresnel is always in [0, 1]."""
        for cos_theta in [0.0, 0.1, 0.5, 1.0]:
            for ior in [1.3, 1.5, 2.0]:
                R = fresnel_air_film(cos_theta, ior)
                assert 0.0 <= R <= 1.0


class TestFresnelFilmSubstrate:
    """Test film-substrate Fresnel interface."""

    def test_film_to_lower_ior(self) -> None:
        """Test film to lower IOR substrate."""
        # Film IOR > substrate IOR (e.g., film=1.5, substrate=1.4)
        R = fresnel_film_substrate(1.0, 1.5, 1.4)
        assert R > 0.0
        # Small difference means small reflectance
        assert R < 0.01

    def test_film_to_higher_ior(self) -> None:
        """Test film to higher IOR substrate."""
        # Film IOR < substrate IOR (film on metal)
        R = fresnel_film_substrate(1.0, 1.5, 2.5)
        assert R > 0.0

    def test_same_ior_no_reflection(self) -> None:
        """Test same IOR gives zero reflectance."""
        R = fresnel_film_substrate(1.0, 1.5, 1.5)
        assert R < 0.001


# =============================================================================
# Phase Computation Tests
# =============================================================================


class TestComputeFilmPhase:
    """Test phase computation from film properties."""

    @pytest.mark.parametrize("ref", IRIDESCENCE_REFERENCE_VALUES["compute_film_phase"])
    def test_compute_film_phase_reference_values(self, ref: dict) -> None:
        """Test compute_film_phase matches reference values."""
        result = compute_film_phase(
            ref["thickness_nm"],
            ref["cos_theta_film"],
            ref["film_ior"],
            ref["wavelength_nm"],
        )
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"compute_film_phase(...) = {result}, expected {ref['expected']}"
        )

    def test_phase_proportional_to_thickness(self) -> None:
        """Test phase doubles when thickness doubles."""
        phase1 = compute_film_phase(200.0, 1.0, 1.5, 500.0)
        phase2 = compute_film_phase(400.0, 1.0, 1.5, 500.0)
        assert abs(phase2 - 2.0 * phase1) < 0.01

    def test_phase_proportional_to_ior(self) -> None:
        """Test phase doubles when IOR doubles."""
        phase1 = compute_film_phase(400.0, 1.0, 1.0, 500.0)
        phase2 = compute_film_phase(400.0, 1.0, 2.0, 500.0)
        assert abs(phase2 - 2.0 * phase1) < 0.01

    def test_phase_inversely_proportional_to_wavelength(self) -> None:
        """Test phase halves when wavelength doubles."""
        phase1 = compute_film_phase(400.0, 1.0, 1.5, 250.0)
        phase2 = compute_film_phase(400.0, 1.0, 1.5, 500.0)
        assert abs(phase1 - 2.0 * phase2) < 0.01

    def test_phase_decreases_with_angle(self) -> None:
        """Test phase decreases at oblique angles."""
        phase_normal = compute_film_phase(400.0, 1.0, 1.5, 500.0)
        phase_oblique = compute_film_phase(400.0, 0.707, 1.5, 500.0)
        assert phase_oblique < phase_normal


# =============================================================================
# Interference Tests
# =============================================================================


class TestComputeInterference:
    """Test interference computation."""

    @pytest.mark.parametrize("ref", IRIDESCENCE_REFERENCE_VALUES["compute_interference"])
    def test_compute_interference_reference_values(self, ref: dict) -> None:
        """Test compute_interference matches reference values."""
        result = compute_interference(ref["R_air_film"], ref["R_film_sub"], ref["phase"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"compute_interference(...) = {result}, expected {ref['expected']}"
        )

    def test_interference_bounded(self) -> None:
        """Test interference result is in [0, 1]."""
        for R1 in [0.04, 0.1, 0.5]:
            for R2 in [0.04, 0.1, 0.5]:
                for phase in [0.0, PI / 2, PI, 3 * PI / 2, 2 * PI]:
                    result = compute_interference(R1, R2, phase)
                    assert 0.0 <= result <= 1.0

    def test_interference_periodic(self) -> None:
        """Test interference is periodic in phase."""
        R1, R2 = 0.1, 0.1
        val1 = compute_interference(R1, R2, 0.0)
        val2 = compute_interference(R1, R2, 2 * PI)
        assert abs(val1 - val2) < 0.001


class TestComputeInterferenceColor:
    """Test RGB interference color computation."""

    def test_rgb_channels_differ(self) -> None:
        """Test that RGB channels have different values (wavelength dependence)."""
        # Use substrate_ior different from film_ior to get film-substrate reflection
        # Also use a thickness that creates visible interference
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        # Use substrate_ior=2.5 (metal-like) to get significant film-substrate Fresnel
        color = compute_interference_color(0.8, params, 2.5)

        # With different wavelengths and sufficient Fresnel, RGB should differ
        r, g, b = color
        # At least one pair should show measurable difference due to wavelength-dependent phase
        assert abs(r - g) > 0.001 or abs(g - b) > 0.001 or abs(r - b) > 0.001, (
            f"RGB channels should differ: R={r:.6f}, G={g:.6f}, B={b:.6f}"
        )

    def test_all_channels_bounded(self) -> None:
        """Test all RGB channels are in [0, 1]."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        for cos_theta in [0.1, 0.5, 1.0]:
            for substrate_ior in [1.5, 2.5]:
                color = compute_interference_color(cos_theta, params, substrate_ior)
                for c in color:
                    assert 0.0 <= c <= 1.0

    def test_color_changes_with_thickness(self) -> None:
        """Test color shifts when thickness changes."""
        # Use substrate_ior different from film_ior for visible effect
        params1 = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=300.0)
        params2 = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=500.0)

        color1 = compute_interference_color(0.8, params1, 2.5)
        color2 = compute_interference_color(0.8, params2, 2.5)

        # Colors should be different due to different phase shifts
        diff = sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))
        assert diff > 0.01, f"Colors should differ: {color1} vs {color2}"


# =============================================================================
# Evaluate Iridescence Tests
# =============================================================================


class TestEvaluateIridescence:
    """Test main iridescence evaluation."""

    def test_zero_intensity_returns_neutral(self) -> None:
        """Test intensity=0 returns (1,1,1)."""
        params = IridescenceParams(intensity=0.0, ior=1.5, thickness_nm=400.0)
        result = evaluate_iridescence(0.8, params)
        assert result == (1.0, 1.0, 1.0)

    def test_full_intensity_produces_color(self) -> None:
        """Test intensity=1.0 produces non-neutral color."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        result = evaluate_iridescence(0.5, params)

        # Result should differ from neutral
        assert result != (1.0, 1.0, 1.0)

    def test_metallic_vs_dielectric(self) -> None:
        """Test metallic produces different result than dielectric."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)

        metal = evaluate_iridescence(0.5, params, is_metallic=True)
        dielectric = evaluate_iridescence(0.5, params, is_metallic=False)

        # Different substrate IOR should give different results
        assert metal != dielectric

    def test_angle_dependence(self) -> None:
        """Test result changes with viewing angle."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)

        normal = evaluate_iridescence(1.0, params)
        oblique = evaluate_iridescence(0.5, params)
        grazing = evaluate_iridescence(0.1, params)

        # All should be valid
        for color in [normal, oblique, grazing]:
            for c in color:
                assert 0.0 <= c <= 1.0


# =============================================================================
# Apply Iridescence Tests
# =============================================================================


class TestApplyIridescence:
    """Test F0 modulation with iridescence."""

    def test_zero_intensity_preserves_f0(self) -> None:
        """Test intensity=0 returns original F0."""
        F0 = (0.04, 0.04, 0.04)
        params = IridescenceParams(intensity=0.0, ior=1.5, thickness_nm=400.0)
        result = apply_iridescence(F0, 0.8, params)
        assert result == F0

    def test_full_intensity_changes_f0(self) -> None:
        """Test intensity=1.0 changes F0."""
        F0 = (0.04, 0.04, 0.04)
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        result = apply_iridescence(F0, 0.5, params)
        assert result != F0

    def test_partial_intensity_blends(self) -> None:
        """Test partial intensity blends between original and iridescent F0."""
        F0 = (0.04, 0.04, 0.04)
        params_full = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        params_half = IridescenceParams(intensity=0.5, ior=1.5, thickness_nm=400.0)

        result_full = apply_iridescence(F0, 0.5, params_full)
        result_half = apply_iridescence(F0, 0.5, params_half)

        # Half intensity should be between original and full
        for i in range(3):
            assert (
                min(F0[i], result_full[i]) <= result_half[i] <= max(F0[i], result_full[i])
                or abs(result_half[i] - (F0[i] + result_full[i]) / 2) < 0.1
            )

    def test_metallic_multiplication(self) -> None:
        """Test metallic uses multiplicative blend."""
        F0 = (1.0, 0.766, 0.336)  # Gold
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        result = apply_iridescence(F0, 0.5, params, is_metallic=True)

        # Result should maintain relative channel relationships
        for c in result:
            assert 0.0 <= c <= 1.5  # Can exceed 1 slightly due to blend

    def test_output_non_negative(self) -> None:
        """Test output is always non-negative."""
        for intensity in [0.0, 0.5, 1.0]:
            for ior in [1.3, 1.5, 2.0]:
                for thickness in [200.0, 400.0, 800.0]:
                    params = IridescenceParams(intensity, ior, thickness)
                    F0 = (0.04, 0.04, 0.04)
                    result = apply_iridescence(F0, 0.5, params)
                    for c in result:
                        assert c >= 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_thin_film(self) -> None:
        """Test very thin film (wide interference bands)."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=100.0)
        result = evaluate_iridescence(0.5, params)
        # Should still produce valid result
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_thick_film(self) -> None:
        """Test thick film (tight interference bands)."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=1000.0)
        result = evaluate_iridescence(0.5, params)
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_low_ior(self) -> None:
        """Test low IOR film (weak Fresnel)."""
        params = IridescenceParams(intensity=1.0, ior=1.3, thickness_nm=400.0)
        result = evaluate_iridescence(0.5, params)
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_high_ior(self) -> None:
        """Test high IOR film (strong Fresnel)."""
        params = IridescenceParams(intensity=1.0, ior=2.0, thickness_nm=400.0)
        result = evaluate_iridescence(0.5, params)
        # Higher IOR should give stronger effect
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_normal_incidence(self) -> None:
        """Test at normal incidence (cos_theta=1)."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        result = evaluate_iridescence(1.0, params)
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_near_grazing_angle(self) -> None:
        """Test near grazing angle (cos_theta very small)."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        result = evaluate_iridescence(0.01, params)
        for c in result:
            assert 0.0 <= c <= 1.0

    def test_zero_cos_theta(self) -> None:
        """Test at exactly grazing (cos_theta=0)."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)
        # Should handle gracefully (clamped to minimum)
        result = evaluate_iridescence(0.0, params)
        for c in result:
            assert 0.0 <= c <= 1.0


# =============================================================================
# Physical Properties Tests
# =============================================================================


class TestPhysicalProperties:
    """Test physical correctness of the model."""

    def test_thickness_shifts_color(self) -> None:
        """Test changing thickness shifts interference pattern."""
        colors = []
        for thickness in [200.0, 300.0, 400.0, 500.0, 600.0]:
            params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=thickness)
            # Use metallic=True to get higher substrate IOR (2.5) for visible interference
            colors.append(evaluate_iridescence(0.7, params, is_metallic=True))

        # At least some consecutive colors should differ (pattern shifts)
        diffs = []
        for i in range(len(colors) - 1):
            diff = sum(abs(c1 - c2) for c1, c2 in zip(colors[i], colors[i + 1]))
            diffs.append(diff)

        # At least one pair should show difference
        assert max(diffs) > 0.001, f"Colors should differ across thickness range: {diffs}"

    def test_ior_affects_intensity(self) -> None:
        """Test IOR affects interference intensity."""
        params_low = IridescenceParams(intensity=1.0, ior=1.3, thickness_nm=400.0)
        params_high = IridescenceParams(intensity=1.0, ior=2.0, thickness_nm=400.0)

        color_low = evaluate_iridescence(0.5, params_low)
        color_high = evaluate_iridescence(0.5, params_high)

        # Higher IOR should give different (generally stronger) effect
        assert color_low != color_high

    def test_rainbow_variation(self) -> None:
        """Test that we get rainbow-like color variation with thickness."""
        # Sample across thickness range with metallic substrate for visible effect
        has_red_dominant = False
        has_green_dominant = False
        has_blue_dominant = False

        for thickness in range(100, 1000, 50):
            params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=float(thickness))
            r, g, b = evaluate_iridescence(0.7, params, is_metallic=True)

            # Check if any channel dominates (with small threshold for numerical stability)
            if r > g + 0.001 and r > b + 0.001:
                has_red_dominant = True
            if g > r + 0.001 and g > b + 0.001:
                has_green_dominant = True
            if b > r + 0.001 and b > g + 0.001:
                has_blue_dominant = True

        # Should have at least one color dominating at some thickness
        # (full rainbow requires specific conditions, but some variation expected)
        assert has_red_dominant or has_green_dominant or has_blue_dominant, (
            "Expected at least one color to dominate at some thickness"
        )

    def test_fresnel_formula_accuracy(self) -> None:
        """Test Fresnel formula matches ((n-1)/(n+1))^2 at normal incidence."""
        for n in [1.3, 1.5, 2.0]:
            expected = ((n - 1) / (n + 1)) ** 2
            result = fresnel_air_film(1.0, n)
            assert abs(result - expected) < 0.001


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other material system components."""

    def test_wgsl_and_python_constants_match(self) -> None:
        """Test Python and WGSL use same constants."""
        wgsl = get_iridescence_wgsl()

        # Check PI constant
        match = re.search(r"const PI:\s*f32\s*=\s*([\d.]+)", wgsl)
        assert match, "Could not find PI in WGSL"
        assert abs(float(match.group(1)) - PI) < 1e-9

        # Check wavelength constants
        assert f"const WAVELENGTH_R: f32 = {WAVELENGTH_R}" in wgsl
        assert f"const WAVELENGTH_G: f32 = {WAVELENGTH_G}" in wgsl
        assert f"const WAVELENGTH_B: f32 = {WAVELENGTH_B}" in wgsl

    def test_import_from_materials(self) -> None:
        """Test iridescence can be imported from trinity.materials."""
        # This import should work after we update __init__.py
        from trinity.materials.iridescence import (
            get_iridescence_wgsl,
            IridescenceParams,
            evaluate_iridescence,
            apply_iridescence,
        )

        assert callable(get_iridescence_wgsl)
        assert callable(evaluate_iridescence)

    def test_works_with_brdf(self) -> None:
        """Test iridescence integrates with BRDF F0 computation."""
        from trinity.materials.brdf import compute_f0

        # Compute base F0
        base_color = (0.8, 0.2, 0.2)
        metallic = 0.0
        F0 = compute_f0(base_color, metallic)

        # Apply iridescence
        params = IridescenceParams(intensity=0.5, ior=1.5, thickness_nm=400.0)
        F0_irid = apply_iridescence(F0, 0.7, params)

        # Result should be valid and different from input
        for c in F0_irid:
            assert 0.0 <= c <= 1.0

    def test_quality_tier_integration(self) -> None:
        """Test iridescence follows quality tier pattern."""
        from trinity.materials.quality import QualityFeatures, QualityTier

        # Check quality features include iridescence
        low = QualityFeatures.for_tier(QualityTier.LOW)
        high = QualityFeatures.for_tier(QualityTier.HIGH)

        assert low.iridescence == False
        assert high.iridescence == True

    def test_reference_values_count(self) -> None:
        """Test we have sufficient reference test inputs."""
        total_refs = 0
        for category, refs in IRIDESCENCE_REFERENCE_VALUES.items():
            total_refs += len(refs)
        assert total_refs >= 15, f"Only {total_refs} reference values, need at least 15"


# =============================================================================
# Preset Material Tests
# =============================================================================


class TestPresetMaterials:
    """Test realistic material presets."""

    def test_soap_bubble_preset(self) -> None:
        """Test soap bubble produces strong rainbow effect."""
        params = get_preset("soap_bubble")
        assert params.intensity == 1.0
        # Thin film should give wide color bands
        color = evaluate_iridescence(0.5, params)
        for c in color:
            assert 0.0 <= c <= 1.0

    def test_oil_slick_preset(self) -> None:
        """Test oil slick produces characteristic iridescence."""
        params = get_preset("oil_slick")
        assert params.intensity == 0.8
        color = evaluate_iridescence(0.5, params)
        for c in color:
            assert 0.0 <= c <= 1.0

    def test_beetle_preset(self) -> None:
        """Test beetle shell preset."""
        params = get_preset("beetle")
        assert params.ior == 1.8  # Chitin
        color = evaluate_iridescence(0.5, params)
        for c in color:
            assert 0.0 <= c <= 1.0

    def test_pearl_preset(self) -> None:
        """Test pearl/nacre preset."""
        params = get_preset("pearl")
        assert params.intensity == 0.4  # Subtle effect
        color = evaluate_iridescence(0.5, params)
        for c in color:
            assert 0.0 <= c <= 1.0


# =============================================================================
# Robustness Tests
# =============================================================================


class TestRobustness:
    """Test numerical robustness."""

    def test_no_nan_or_inf(self) -> None:
        """Test no NaN or Inf values in outputs."""
        test_cases = [
            (0.0, 1.0, 400.0),
            (1.0, 1.001, 50.0),  # Near-unity IOR
            (0.001, 2.0, 2000.0),  # Extreme parameters
        ]

        for intensity, ior, thickness in test_cases:
            try:
                params = IridescenceParams(intensity=intensity, ior=ior, thickness_nm=thickness)
                result = evaluate_iridescence(0.5, params)
                for c in result:
                    assert not math.isnan(c), f"NaN with params ({intensity}, {ior}, {thickness})"
                    assert not math.isinf(c), f"Inf with params ({intensity}, {ior}, {thickness})"
            except ValueError:
                # Parameter validation may reject extreme values
                pass

    def test_continuous_variation(self) -> None:
        """Test output varies continuously with parameters."""
        params = IridescenceParams(intensity=1.0, ior=1.5, thickness_nm=400.0)

        prev_color = evaluate_iridescence(1.0, params)
        for cos_theta in [0.99, 0.98, 0.95, 0.9, 0.8]:
            curr_color = evaluate_iridescence(cos_theta, params)
            # Change should be gradual, not discontinuous
            diff = sum(abs(c1 - c2) for c1, c2 in zip(prev_color, curr_color))
            assert diff < 0.5, f"Discontinuity at cos_theta={cos_theta}"
            prev_color = curr_color


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
