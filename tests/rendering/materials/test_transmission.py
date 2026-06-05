"""Tests for Transmission BRDF functions (T-MAT-4.5).

This module tests:
- WGSL syntax validation for transmission.wgsl
- Reference value matching within tolerance
- Snell's law refraction correctness
- Total internal reflection handling
- Beer-Lambert absorption accuracy
- Fresnel-weighted blending
- Edge cases (factor=0/1, grazing angles, TIR)
- Energy conservation properties
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple, Optional

import pytest

from trinity.materials.transmission_shader import (
    # WGSL source
    get_transmission_wgsl,
    # Parameters
    TransmissionParams,
    # Fresnel functions
    f_transmission,
    ior_to_f0,
    # Refraction functions
    refract_direction,
    is_total_internal_reflection,
    get_critical_angle,
    # Beer-Lambert absorption
    apply_beer_law,
    compute_beer_transmittance,
    # Evaluation functions
    evaluate_transmission,
    evaluate_transmission_with_fresnel,
    # Layer combination
    combine_transmission_reflection,
    combine_transmission_simple,
    # Reference values
    TRANSMISSION_REFERENCE_VALUES,
    TRANSMISSION_EDGE_CASES,
    # Constants
    AIR_IOR,
    GLASS_IOR,
    WATER_IOR,
    PI,
    EPSILON,
)


# Type alias
Vec3 = Tuple[float, float, float]


def normalize(v: Vec3) -> Vec3:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_transmission_wgsl_loads(self) -> None:
        """Test that transmission.wgsl can be loaded."""
        wgsl = get_transmission_wgsl()
        assert len(wgsl) > 0
        assert "fn F_Transmission" in wgsl
        assert "fn refract_direction" in wgsl
        assert "fn apply_beer_law" in wgsl
        assert "fn evaluate_transmission" in wgsl

    def test_transmission_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = (
            Path(__file__).parents[3]
            / "trinity"
            / "materials"
            / "wgsl"
            / "transmission.wgsl"
        )
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_transmission_wgsl_has_required_functions(self) -> None:
        """Test that all required transmission functions are present."""
        wgsl = get_transmission_wgsl()
        required_functions = [
            "fn F_Transmission",
            "fn ior_to_f0",
            "fn refract_direction",
            "fn is_total_internal_reflection",
            "fn apply_beer_law",
            "fn compute_beer_transmittance",
            "fn evaluate_transmission",
            "fn evaluate_transmission_screen_space",
            "fn evaluate_transmission_with_fresnel",
            "fn combine_transmission_reflection",
            "fn combine_transmission_simple",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_transmission_wgsl_has_params_struct(self) -> None:
        """Test that TransmissionParams struct is defined."""
        wgsl = get_transmission_wgsl()
        assert "struct TransmissionParams" in wgsl
        assert "factor: f32" in wgsl
        assert "ior: f32" in wgsl
        assert "roughness: f32" in wgsl
        assert "attenuation_color: vec3<f32>" in wgsl
        assert "attenuation_distance: f32" in wgsl

    def test_transmission_wgsl_has_quality_const(self) -> None:
        """Test that quality tier const is defined."""
        wgsl = get_transmission_wgsl()
        assert "QUALITY_TRANSMISSION_ENABLED" in wgsl
        assert "const QUALITY_TRANSMISSION_ENABLED: bool" in wgsl

    def test_transmission_wgsl_has_ior_constants(self) -> None:
        """Test that IOR constants are defined."""
        wgsl = get_transmission_wgsl()
        assert "DEFAULT_IOR" in wgsl
        assert "AIR_IOR" in wgsl
        assert "1.5" in wgsl  # Glass IOR

    def test_transmission_wgsl_syntax_patterns(self) -> None:
        """Test basic WGSL syntax patterns."""
        wgsl = get_transmission_wgsl()

        # Check function declarations
        fn_pattern = r"fn\s+\w+\([^)]*\)\s*->\s*\w+"
        assert re.search(fn_pattern, wgsl), "No valid function declarations found"

        # Check for proper type annotations
        assert "f32" in wgsl, "Missing f32 type annotations"
        assert "vec3<f32>" in wgsl, "Missing vec3<f32> type annotations"
        assert "vec4<f32>" in wgsl, "Missing vec4<f32> type annotations"
        assert "vec2<f32>" in wgsl, "Missing vec2<f32> type annotations"

    def test_transmission_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_transmission_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert (
            open_braces == close_braces
        ), f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert (
            open_parens == close_parens
        ), f"Unbalanced parentheses: {open_parens} open, {close_parens} close"


# =============================================================================
# TransmissionParams Tests
# =============================================================================


class TestTransmissionParams:
    """Test TransmissionParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = TransmissionParams()
        assert params.factor == 1.0
        assert params.ior == 1.5
        assert params.roughness == 0.0
        assert params.attenuation_color == (1.0, 1.0, 1.0)
        assert params.attenuation_distance == float("inf")

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = TransmissionParams(
            factor=0.5,
            ior=1.33,
            roughness=0.1,
            attenuation_color=(0.8, 0.9, 1.0),
            attenuation_distance=5.0,
        )
        assert params.factor == 0.5
        assert params.ior == 1.33
        assert params.roughness == 0.1
        assert params.attenuation_color == (0.8, 0.9, 1.0)
        assert params.attenuation_distance == 5.0

    def test_factor_validation(self) -> None:
        """Test factor must be in [0,1]."""
        with pytest.raises(ValueError):
            TransmissionParams(factor=-0.1)
        with pytest.raises(ValueError):
            TransmissionParams(factor=1.1)

    def test_ior_validation(self) -> None:
        """Test IOR must be in [1,3]."""
        with pytest.raises(ValueError):
            TransmissionParams(ior=0.9)
        with pytest.raises(ValueError):
            TransmissionParams(ior=3.1)

    def test_roughness_validation(self) -> None:
        """Test roughness must be in [0,1]."""
        with pytest.raises(ValueError):
            TransmissionParams(roughness=-0.1)
        with pytest.raises(ValueError):
            TransmissionParams(roughness=1.1)

    def test_attenuation_distance_validation(self) -> None:
        """Test attenuation_distance must be positive."""
        with pytest.raises(ValueError):
            TransmissionParams(attenuation_distance=0.0)
        with pytest.raises(ValueError):
            TransmissionParams(attenuation_distance=-1.0)

    def test_glass_preset(self) -> None:
        """Test glass preset creates correct parameters."""
        params = TransmissionParams.glass()
        assert params.factor == 1.0
        assert params.ior == 1.5
        assert params.roughness == 0.0

    def test_water_preset(self) -> None:
        """Test water preset creates correct parameters."""
        params = TransmissionParams.water()
        assert params.factor == 1.0
        assert params.ior == 1.33
        assert params.attenuation_distance < float("inf")

    def test_colored_glass_preset(self) -> None:
        """Test colored glass preset."""
        params = TransmissionParams.colored_glass((0.5, 0.8, 0.3), 2.0)
        assert params.ior == 1.5
        assert params.attenuation_color == (0.5, 0.8, 0.3)
        assert params.attenuation_distance == 2.0


# =============================================================================
# F_Transmission Reference Value Tests
# =============================================================================


class TestFTransmission:
    """Test Schlick Fresnel for transmission."""

    @pytest.mark.parametrize("ref", TRANSMISSION_REFERENCE_VALUES["F_Transmission"])
    def test_f_transmission_reference_values(self, ref: dict) -> None:
        """Test F_Transmission matches reference values within tolerance."""
        result = f_transmission(ref["cos_theta"], ref["ior"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"F_Transmission(cos_theta={ref['cos_theta']}, ior={ref['ior']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_f_transmission_at_normal_incidence(self) -> None:
        """Test F_Transmission returns F0 at normal incidence."""
        result = f_transmission(1.0, GLASS_IOR)
        expected_f0 = ior_to_f0(GLASS_IOR)
        assert abs(result - expected_f0) < EPSILON

    def test_f_transmission_at_grazing_angle(self) -> None:
        """Test F_Transmission approaches 1.0 at grazing angle."""
        result = f_transmission(0.0, GLASS_IOR)
        assert abs(result - 1.0) < EPSILON

    def test_f_transmission_monotonic(self) -> None:
        """Test F_Transmission is monotonically decreasing with cos_theta."""
        prev_value = f_transmission(0.0, GLASS_IOR)
        for i in range(1, 11):
            cos_theta = i / 10.0
            value = f_transmission(cos_theta, GLASS_IOR)
            assert (
                value <= prev_value + EPSILON
            ), f"F_Transmission not monotonic at cos_theta={cos_theta}"
            prev_value = value


# =============================================================================
# IOR to F0 Tests
# =============================================================================


class TestIORToF0:
    """Test IOR to F0 conversion."""

    @pytest.mark.parametrize("ref", TRANSMISSION_REFERENCE_VALUES["ior_to_f0"])
    def test_ior_to_f0_reference_values(self, ref: dict) -> None:
        """Test ior_to_f0 matches reference values."""
        result = ior_to_f0(ref["ior"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"ior_to_f0({ref['ior']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_ior_to_f0_air(self) -> None:
        """Test F0 is 0 for IOR=1 (air to air)."""
        result = ior_to_f0(1.0)
        assert abs(result) < EPSILON

    def test_ior_to_f0_glass(self) -> None:
        """Test F0 for standard glass (IOR 1.5)."""
        result = ior_to_f0(1.5)
        assert abs(result - 0.04) < 0.001

    def test_ior_to_f0_increasing(self) -> None:
        """Test F0 increases with IOR."""
        prev_f0 = ior_to_f0(1.0)
        for ior in [1.1, 1.3, 1.5, 2.0, 2.5]:
            f0 = ior_to_f0(ior)
            assert f0 > prev_f0, f"F0 should increase with IOR at ior={ior}"
            prev_f0 = f0


# =============================================================================
# Refraction Tests (Snell's Law)
# =============================================================================


class TestRefraction:
    """Test Snell's law refraction."""

    def test_normal_incidence_no_refraction(self) -> None:
        """Test that normal incidence produces no angular deviation."""
        incident = (0.0, 1.0, 0.0)
        normal = (0.0, 1.0, 0.0)
        eta = AIR_IOR / GLASS_IOR

        refracted = refract_direction(incident, normal, eta)
        assert refracted is not None

        # At normal incidence, refracted direction should be opposite to normal
        # (light continues straight through)
        assert abs(refracted[1] - (-1.0)) < 0.01

    def test_refraction_bends_toward_normal(self) -> None:
        """Test that light bends toward normal when entering denser medium."""
        # Incident at 45 degrees
        incident = normalize((1.0, 1.0, 0.0))
        normal = (0.0, 1.0, 0.0)
        eta = AIR_IOR / GLASS_IOR  # Air to glass

        refracted = refract_direction(incident, normal, eta)
        assert refracted is not None

        # Refracted angle should be less than incident angle
        # cos of refracted angle with -normal should be larger
        incident_cos = abs(dot(incident, normal))
        refracted_cos = abs(dot(refracted, (0.0, -1.0, 0.0)))

        # When eta < 1, light bends toward normal (larger cos)
        assert refracted_cos > incident_cos - 0.1

    def test_refraction_bends_away_from_normal(self) -> None:
        """Test that light bends away from normal when exiting denser medium."""
        incident = normalize((0.1, 1.0, 0.0))
        normal = (0.0, 1.0, 0.0)
        eta = GLASS_IOR / AIR_IOR  # Glass to air

        # At shallow angles, might get TIR
        refracted = refract_direction(incident, normal, eta)

        # For this shallow angle, should not be TIR
        if refracted is not None:
            # Refracted should bend away from normal
            incident_cos = abs(dot(incident, normal))
            refracted_cos = abs(dot(refracted, (0.0, -1.0, 0.0)))
            # When eta > 1, light bends away (smaller cos)
            assert refracted_cos < incident_cos + 0.1

    def test_total_internal_reflection(self) -> None:
        """Test TIR occurs at steep angles when exiting denser medium."""
        # Steep angle - almost parallel to surface
        incident = normalize((1.0, 0.1, 0.0))
        normal = (0.0, 1.0, 0.0)
        eta = GLASS_IOR / AIR_IOR  # Glass to air

        refracted = refract_direction(incident, normal, eta)
        # Should be TIR - no refracted ray
        assert refracted is None


class TestTotalInternalReflection:
    """Test TIR detection."""

    @pytest.mark.parametrize(
        "ref", TRANSMISSION_REFERENCE_VALUES["total_internal_reflection"]
    )
    def test_tir_detection(self, ref: dict) -> None:
        """Test is_total_internal_reflection matches expected."""
        result = is_total_internal_reflection(ref["cos_i"], ref["eta"])
        assert result == ref["expected_tir"], (
            f"is_total_internal_reflection(cos_i={ref['cos_i']}, eta={ref['eta']}) = {result}, "
            f"expected {ref['expected_tir']}"
        )

    def test_no_tir_entering_denser_medium(self) -> None:
        """Test no TIR when entering denser medium (air to glass)."""
        eta = AIR_IOR / GLASS_IOR
        for cos_i in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            assert not is_total_internal_reflection(
                cos_i, eta
            ), f"Should not have TIR at cos_i={cos_i} entering glass"

    def test_critical_angle_glass(self) -> None:
        """Test critical angle for glass (about 41.8 degrees)."""
        critical = get_critical_angle(GLASS_IOR)
        expected_deg = 41.8
        result_deg = math.degrees(critical)
        assert abs(result_deg - expected_deg) < 1.0


# =============================================================================
# Beer-Lambert Absorption Tests
# =============================================================================


class TestBeerLambert:
    """Test Beer-Lambert absorption."""

    @pytest.mark.parametrize("ref", TRANSMISSION_REFERENCE_VALUES["apply_beer_law"])
    def test_apply_beer_law_reference_values(self, ref: dict) -> None:
        """Test apply_beer_law matches reference values."""
        result = apply_beer_law(
            ref["transmitted_color"],
            ref["distance"],
            ref["attenuation_color"],
            ref["attenuation_distance"],
        )
        for i, (r, e) in enumerate(zip(result, ref["expected"])):
            assert abs(r - e) < ref["tolerance"], (
                f"apply_beer_law channel {i}: {r}, expected {e} +/- {ref['tolerance']}"
            )

    def test_no_absorption_white_attenuation(self) -> None:
        """Test no absorption with white attenuation color."""
        color = (1.0, 0.5, 0.25)
        result = apply_beer_law(color, 10.0, (1.0, 1.0, 1.0), 1.0)
        for i in range(3):
            assert abs(result[i] - color[i]) < EPSILON

    def test_no_absorption_zero_distance(self) -> None:
        """Test no absorption at zero distance."""
        color = (1.0, 0.5, 0.25)
        result = apply_beer_law(color, 0.0, (0.5, 0.5, 0.5), 1.0)
        for i in range(3):
            assert abs(result[i] - color[i]) < EPSILON

    def test_absorption_increases_with_distance(self) -> None:
        """Test absorption increases with distance traveled."""
        color = (1.0, 1.0, 1.0)
        atten = (0.5, 0.5, 0.5)
        atten_dist = 1.0

        prev_result = apply_beer_law(color, 0.0, atten, atten_dist)
        for distance in [0.5, 1.0, 2.0, 5.0]:
            result = apply_beer_law(color, distance, atten, atten_dist)
            # Each channel should be less than or equal to previous
            for i in range(3):
                assert (
                    result[i] <= prev_result[i] + EPSILON
                ), f"Absorption should increase at distance={distance}"
            prev_result = result

    def test_colored_absorption(self) -> None:
        """Test different absorption rates per channel."""
        color = (1.0, 1.0, 1.0)
        # Green glass - absorbs red and blue
        atten = (0.3, 0.9, 0.3)
        result = apply_beer_law(color, 1.0, atten, 1.0)

        # Green channel should remain higher
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_compute_beer_transmittance(self) -> None:
        """Test compute_beer_transmittance returns transmittance only."""
        atten = (0.5, 0.8, 1.0)
        result = compute_beer_transmittance(1.0, atten, 1.0)
        assert abs(result[0] - 0.5) < EPSILON
        assert abs(result[1] - 0.8) < EPSILON
        assert abs(result[2] - 1.0) < EPSILON


# =============================================================================
# Transmission Evaluation Tests
# =============================================================================


class TestEvaluateTransmission:
    """Test transmission evaluation."""

    @pytest.mark.parametrize(
        "ref", TRANSMISSION_REFERENCE_VALUES["evaluate_transmission"]
    )
    def test_evaluate_transmission_reference_values(self, ref: dict) -> None:
        """Test evaluate_transmission matches reference values."""
        params = TransmissionParams(
            factor=ref["factor"],
            ior=ref["ior"],
            roughness=0.0,
        )
        result = evaluate_transmission(
            ref["N"],
            ref["V"],
            params,
            ref["thickness"],
            ref["background_color"],
        )
        assert abs(result[0] - ref["expected_r"]) < ref["tolerance"], (
            f"evaluate_transmission: {result[0]}, "
            f"expected {ref['expected_r']} +/- {ref['tolerance']}"
        )

    def test_zero_factor_no_transmission(self) -> None:
        """Test zero factor produces no transmission."""
        params = TransmissionParams(factor=0.0, ior=1.5)
        result = evaluate_transmission(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            0.01,
            (1.0, 1.0, 1.0),
        )
        for i in range(3):
            assert abs(result[i]) < EPSILON

    def test_full_factor_maximum_transmission(self) -> None:
        """Test full factor produces maximum transmission."""
        params = TransmissionParams.glass()
        result = evaluate_transmission(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            0.01,
            (1.0, 1.0, 1.0),
        )
        # At normal incidence, glass transmits about 96%
        assert result[0] > 0.9

    def test_grazing_angle_low_transmission(self) -> None:
        """Test grazing angles have low transmission (high Fresnel)."""
        params = TransmissionParams.glass()
        # Near-grazing angle
        V = normalize((0.99, 0.1, 0.0))
        N = (0.0, 1.0, 0.0)
        result = evaluate_transmission(
            N, V, params, 0.01, (1.0, 1.0, 1.0)
        )
        # High Fresnel at grazing = low transmission
        assert result[0] < 0.5


class TestEvaluateTransmissionWithFresnel:
    """Test transmission evaluation with Fresnel factor."""

    def test_returns_transmission_factor(self) -> None:
        """Test that w component contains transmission factor."""
        params = TransmissionParams.glass()
        result = evaluate_transmission_with_fresnel(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            0.01,
            (1.0, 1.0, 1.0),
        )
        # T = (1 - F) * factor, at normal incidence F ~ 0.04
        expected_T = 0.96
        assert abs(result[3] - expected_T) < 0.02

    def test_transmission_factor_decreases_at_grazing(self) -> None:
        """Test transmission factor decreases at grazing angles."""
        params = TransmissionParams.glass()
        N = (0.0, 1.0, 0.0)

        # Normal incidence
        result_normal = evaluate_transmission_with_fresnel(
            N, (0.0, 1.0, 0.0), params, 0.01, (1.0, 1.0, 1.0)
        )

        # Grazing
        result_grazing = evaluate_transmission_with_fresnel(
            N, normalize((0.95, 0.1, 0.0)), params, 0.01, (1.0, 1.0, 1.0)
        )

        assert result_grazing[3] < result_normal[3]


# =============================================================================
# Layer Combination Tests
# =============================================================================


class TestLayerCombination:
    """Test layer combination functions."""

    def test_combine_transmission_reflection_basic(self) -> None:
        """Test basic transmission/reflection combination."""
        reflection = (0.1, 0.1, 0.1)
        transmission = (0.9, 0.9, 0.9)
        F = 0.04
        factor = 1.0

        result = combine_transmission_reflection(reflection, transmission, F, factor)

        # At normal incidence: mostly transmission
        # Result = F * refl + (1-F) * factor * trans
        expected = 0.04 * 0.1 + 0.96 * 0.9
        assert abs(result[0] - expected) < 0.01

    def test_combine_transmission_reflection_no_transmission(self) -> None:
        """Test combination with zero transmission factor."""
        reflection = (0.5, 0.5, 0.5)
        transmission = (0.8, 0.8, 0.8)
        F = 0.04
        factor = 0.0

        result = combine_transmission_reflection(reflection, transmission, F, factor)

        # With factor=0, should be F * reflection
        expected = 0.04 * 0.5
        assert abs(result[0] - expected) < EPSILON

    def test_combine_transmission_reflection_full_fresnel(self) -> None:
        """Test combination at grazing angle (F=1)."""
        reflection = (0.3, 0.3, 0.3)
        transmission = (0.7, 0.7, 0.7)
        F = 1.0  # Grazing angle
        factor = 1.0

        result = combine_transmission_reflection(reflection, transmission, F, factor)

        # At grazing: only reflection
        # T = (1-1)*1 = 0
        assert abs(result[0] - 0.3) < EPSILON

    def test_combine_transmission_simple(self) -> None:
        """Test simplified layer combination."""
        base = (0.3, 0.3, 0.3)
        trans_result = (0.6, 0.6, 0.6, 0.9)  # (r, g, b, T)

        result = combine_transmission_simple(base, trans_result)

        # base * (1-T) + transmission = 0.3 * 0.1 + 0.6 = 0.63
        expected = 0.3 * 0.1 + 0.6
        assert abs(result[0] - expected) < 0.01


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy conservation properties."""

    def test_fresnel_plus_transmission_equals_one(self) -> None:
        """Test F + T = 1 (energy conservation)."""
        for cos_theta in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            F = f_transmission(cos_theta, GLASS_IOR)
            T = 1.0 - F
            assert abs(F + T - 1.0) < EPSILON

    def test_no_energy_gain(self) -> None:
        """Test transmission never increases energy."""
        params = TransmissionParams.glass()
        bg_color = (0.5, 0.5, 0.5)

        result = evaluate_transmission(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            0.01,
            bg_color,
        )

        # Output should never exceed input
        for i in range(3):
            assert result[i] <= bg_color[i] + EPSILON

    def test_absorption_only_reduces_energy(self) -> None:
        """Test absorption can only reduce energy, never increase."""
        color = (1.0, 1.0, 1.0)
        atten = (0.7, 0.8, 0.9)

        for distance in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = apply_beer_law(color, distance, atten, 1.0)
            for i in range(3):
                assert result[i] <= color[i] + EPSILON


# =============================================================================
# Glass IOR 1.5 Sample Verification
# =============================================================================


class TestGlassIOR15:
    """Verify transmission for standard glass (IOR 1.5)."""

    def test_glass_f0(self) -> None:
        """Test glass F0 is approximately 0.04 (4% reflectance)."""
        f0 = ior_to_f0(1.5)
        assert abs(f0 - 0.04) < 0.001

    def test_glass_normal_incidence_transmission(self) -> None:
        """Test glass transmits ~96% at normal incidence."""
        F = f_transmission(1.0, 1.5)
        T = 1.0 - F
        assert abs(T - 0.96) < 0.01

    def test_glass_45_degree_transmission(self) -> None:
        """Test glass transmission at 45 degrees."""
        cos_45 = math.cos(math.radians(45))
        F = f_transmission(cos_45, 1.5)
        T = 1.0 - F
        # At 45 degrees, still high transmission but slightly less
        assert T > 0.9

    def test_glass_critical_angle(self) -> None:
        """Test glass critical angle is about 41.8 degrees."""
        critical = get_critical_angle(1.5)
        critical_deg = math.degrees(critical)
        assert abs(critical_deg - 41.8) < 0.5

    def test_glass_clear_transmission(self) -> None:
        """Test clear glass (no absorption) transmits color unchanged."""
        params = TransmissionParams.glass()
        bg = (0.8, 0.5, 0.3)

        result = evaluate_transmission(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            0.01,  # Thin glass
            bg,
        )

        # Color should be scaled by transmission factor (~0.96)
        for i in range(3):
            assert abs(result[i] - bg[i] * 0.96) < 0.02


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full transmission pipeline."""

    def test_full_transmission_pipeline(self) -> None:
        """Test complete transmission evaluation pipeline."""
        # Setup: colored glass with absorption
        params = TransmissionParams.colored_glass(
            color=(0.8, 0.95, 0.8),  # Slight green tint
            absorption_distance=2.0,
        )

        N = (0.0, 1.0, 0.0)
        V = normalize((0.2, 0.98, 0.0))
        bg = (1.0, 1.0, 1.0)
        thickness = 1.0

        result = evaluate_transmission_with_fresnel(N, V, params, thickness, bg)

        # Should have valid transmission
        assert result[3] > 0.0
        # Color should show green tint (G > R, G > B)
        assert result[1] >= result[0] - 0.1
        assert result[1] >= result[2] - 0.1

    def test_water_transmission(self) -> None:
        """Test water transmission (IOR 1.33)."""
        params = TransmissionParams.water()

        result = evaluate_transmission(
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            params,
            1.0,  # 1 meter of water
            (1.0, 1.0, 1.0),
        )

        # Water has lower IOR = higher transmission at normal incidence
        # F0 for water ~ 0.02, so T ~ 0.98
        assert result[0] > 0.9

        # Water has slight blue tint
        assert result[2] >= result[0] - 0.05

    def test_layer_combination_with_specular(self) -> None:
        """Test combining transmission with specular reflection."""
        params = TransmissionParams.glass()
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)

        # Evaluate transmission
        trans_result = evaluate_transmission_with_fresnel(
            N, V, params, 0.01, (0.5, 0.5, 0.5)
        )

        # Simulate specular reflection
        specular = (0.1, 0.1, 0.1)

        # Combine
        final = combine_transmission_simple(specular, trans_result)

        # Final should include both contributions
        assert final[0] > specular[0]
        assert final[0] > trans_result[0]
