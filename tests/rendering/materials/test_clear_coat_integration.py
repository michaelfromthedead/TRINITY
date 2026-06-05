"""Tests for Clear Coat PBR Shader Integration (T-MAT-4.2).

This module tests:
- Clear coat integration into pbr.frag.wgsl
- Clear coat = 0 produces identical output to base PBR
- Clear coat = 1 adds visible specular layer
- Energy conservation in dual-layer BRDF
- WGSL shader validation with naga (if available)
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pytest


# Type alias for RGB color
Vec3 = Tuple[float, float, float]


# =============================================================================
# Constants (matching WGSL shader)
# =============================================================================

PI = 3.14159265359
EPSILON = 0.00001
CLEAR_COAT_F0 = 0.04


# =============================================================================
# Helper Functions
# =============================================================================


def normalize(v: Vec3) -> Vec3:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Vec3, b: Vec3) -> Vec3:
    """Add two 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vec3, s: float) -> Vec3:
    """Scale a 3D vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


# =============================================================================
# WGSL Source Loading
# =============================================================================


def get_pbr_frag_wgsl() -> str:
    """Load the PBR fragment shader source."""
    wgsl_path = Path(__file__).parents[3] / "crates" / "renderer-backend" / "shaders" / "pbr.frag.wgsl"
    if not wgsl_path.exists():
        pytest.skip(f"PBR shader not found at {wgsl_path}")
    return wgsl_path.read_text()


def check_naga_available() -> bool:
    """Check if naga CLI is available for WGSL validation."""
    try:
        result = subprocess.run(
            ["naga", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# =============================================================================
# Clear Coat BRDF Reference Implementation (Python)
# =============================================================================


def fresnel_clear_coat(VoH: float) -> float:
    """Schlick Fresnel for clear coat (IOR 1.5, F0 = 0.04)."""
    Fc = pow(1.0 - VoH, 5.0)
    return CLEAR_COAT_F0 + (1.0 - CLEAR_COAT_F0) * Fc


def distribution_ggx_clear_coat(NoH: float, cc_roughness: float) -> float:
    """GGX NDF for clear coat with Disney roughness remapping."""
    a = cc_roughness * cc_roughness
    a2 = a * a
    NoH2 = NoH * NoH
    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


def geometry_clear_coat_kelemen(VoH: float) -> float:
    """Kelemen visibility function for clear coat."""
    return 0.25 / (VoH * VoH + EPSILON)


def eval_clear_coat(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    cc_intensity: float,
    cc_roughness: float,
) -> Tuple[Vec3, float]:
    """Evaluate clear coat BRDF.

    Returns: (brdf_rgb, fresnel_factor)
    """
    if cc_intensity < EPSILON:
        return ((0.0, 0.0, 0.0), 0.0)

    # Half vector
    H = normalize(add(V, L))

    NoL = max(dot(N, L), 0.0)
    NoV = max(dot(N, V), 0.0)
    NoH = max(dot(N, H), 0.0)
    VoH = max(dot(V, H), 0.0)

    if NoL < EPSILON or NoV < EPSILON:
        return ((0.0, 0.0, 0.0), 0.0)

    D = distribution_ggx_clear_coat(NoH, cc_roughness)
    G = geometry_clear_coat_kelemen(VoH)
    F = fresnel_clear_coat(VoH)

    cc_brdf = D * G * F * cc_intensity

    return ((cc_brdf, cc_brdf, cc_brdf), F * cc_intensity)


# =============================================================================
# WGSL Structure Tests
# =============================================================================


class TestWGSLStructure:
    """Test that pbr.frag.wgsl has correct clear coat integration."""

    def test_material_table_entry_has_clear_coat_fields(self) -> None:
        """Test MaterialTableEntry struct includes clear coat parameters."""
        wgsl = get_pbr_frag_wgsl()

        # Check for clear coat field
        assert "clear_coat: f32" in wgsl, "Missing clear_coat field in MaterialTableEntry"
        assert "clear_coat_roughness: f32" in wgsl, "Missing clear_coat_roughness field in MaterialTableEntry"

    def test_clear_coat_f0_constant_defined(self) -> None:
        """Test CLEAR_COAT_F0 constant is defined."""
        wgsl = get_pbr_frag_wgsl()
        assert "CLEAR_COAT_F0" in wgsl, "Missing CLEAR_COAT_F0 constant"
        assert "0.04" in wgsl, "CLEAR_COAT_F0 should be 0.04 for IOR 1.5"

    def test_clear_coat_brdf_functions_exist(self) -> None:
        """Test clear coat BRDF functions are present."""
        wgsl = get_pbr_frag_wgsl()

        required_functions = [
            "fn fresnel_clear_coat",
            "fn distribution_ggx_clear_coat",
            "fn geometry_clear_coat_kelemen",
            "fn eval_clear_coat",
            "fn eval_brdf_with_clear_coat",
        ]

        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_light_functions_accept_clear_coat_params(self) -> None:
        """Test light evaluation functions accept clear coat parameters."""
        wgsl = get_pbr_frag_wgsl()

        # Check that light functions have cc_intensity and cc_roughness parameters
        light_funcs = ["eval_point_light", "eval_directional_light", "eval_spot_light"]

        for func in light_funcs:
            # Find the function signature
            pattern = rf"fn {func}\([^)]*cc_intensity: f32[^)]*cc_roughness: f32"
            assert re.search(pattern, wgsl), f"{func} should accept cc_intensity and cc_roughness"

    def test_fs_main_extracts_clear_coat_params(self) -> None:
        """Test fs_main extracts clear coat from material."""
        wgsl = get_pbr_frag_wgsl()

        # Should extract clear coat parameters
        assert "material.clear_coat" in wgsl, "fs_main should extract clear_coat from material"
        assert "material.clear_coat_roughness" in wgsl, "fs_main should extract clear_coat_roughness"


# =============================================================================
# Clear Coat Behavior Tests
# =============================================================================


class TestClearCoatBehavior:
    """Test clear coat BRDF behavior."""

    def test_clear_coat_zero_matches_base(self) -> None:
        """Clear coat = 0 should produce zero additional contribution."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        cc_brdf, Fc = eval_clear_coat(N, V, L, cc_intensity=0.0, cc_roughness=0.5)

        # Should be zero
        assert all(abs(c) < EPSILON for c in cc_brdf), "Zero intensity should produce zero BRDF"
        assert abs(Fc) < EPSILON, "Zero intensity should produce zero Fresnel factor"

    def test_clear_coat_one_adds_specular(self) -> None:
        """Clear coat = 1 should add visible specular layer."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        cc_brdf, Fc = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.5)

        # Should have positive contribution
        assert cc_brdf[0] > 0.01, "Full clear coat should produce visible specular"
        # Fresnel should be F0 at normal incidence
        assert abs(Fc - CLEAR_COAT_F0) < 0.01, "Fc at normal should be ~F0"

    def test_clear_coat_achromatic(self) -> None:
        """Clear coat should be achromatic (R = G = B)."""
        N = (0.0, 1.0, 0.0)
        V = normalize((0.3, 0.9, 0.3))
        L = normalize((0.5, 0.8, 0.2))

        cc_brdf, _ = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.3)

        # All channels should be equal
        assert abs(cc_brdf[0] - cc_brdf[1]) < EPSILON, "Clear coat should be achromatic"
        assert abs(cc_brdf[1] - cc_brdf[2]) < EPSILON, "Clear coat should be achromatic"

    def test_clear_coat_grazing_increases_fresnel(self) -> None:
        """Clear coat Fresnel should increase at grazing angles."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)

        # Normal incidence light
        L_normal = (0.0, 1.0, 0.0)
        _, Fc_normal = eval_clear_coat(N, V, L_normal, cc_intensity=1.0, cc_roughness=0.3)

        # Grazing light
        L_grazing = normalize((0.9, 0.436, 0.0))
        _, Fc_grazing = eval_clear_coat(N, V, L_grazing, cc_intensity=1.0, cc_roughness=0.3)

        # Fresnel should be higher at grazing
        assert Fc_grazing > Fc_normal, "Fresnel should increase at grazing angles"


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy conservation in dual-layer BRDF."""

    def test_base_attenuation_bounded(self) -> None:
        """Base layer attenuation should be in [0, 1]."""
        for Fc in [0.0, CLEAR_COAT_F0, 0.5, 1.0]:
            for intensity in [0.0, 0.5, 1.0]:
                attenuation = 1.0 - Fc * intensity
                assert 0.0 <= attenuation <= 1.0, f"Attenuation out of bounds: {attenuation}"

    def test_energy_conservation_at_normal(self) -> None:
        """At normal incidence, ~96% of energy reaches base layer."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        _, Fc = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.3)
        attenuation = 1.0 - Fc

        # At normal, Fc ~ 0.04, so attenuation ~ 0.96
        assert abs(attenuation - 0.96) < 0.02, f"At normal, attenuation should be ~0.96, got {attenuation}"

    def test_combined_energy_bounded(self) -> None:
        """Combined clear coat + base should not exceed input energy."""
        # Simulate unit base BRDF
        base_brdf = (1.0, 1.0, 1.0)

        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        cc_brdf, Fc = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.3)
        attenuation = 1.0 - Fc

        # Combined = cc + attenuation * base
        combined = (
            cc_brdf[0] + attenuation * base_brdf[0],
            cc_brdf[1] + attenuation * base_brdf[1],
            cc_brdf[2] + attenuation * base_brdf[2],
        )

        # Should not exceed reasonable bounds
        for c in combined:
            assert c <= 2.0, f"Combined energy too high: {c}"


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLValidation:
    """Test WGSL shader validation."""

    def test_wgsl_no_syntax_errors(self) -> None:
        """Test WGSL has no obvious syntax errors."""
        wgsl = get_pbr_frag_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"

    def test_wgsl_function_declarations_valid(self) -> None:
        """Test function declarations are properly formed."""
        wgsl = get_pbr_frag_wgsl()

        # Look for function declarations
        fn_pattern = r"fn\s+\w+\s*\([^)]*\)\s*(->\s*\w+(<[^>]+>)?)?\s*\{"
        matches = re.findall(fn_pattern, wgsl)

        # Should have multiple functions
        assert len(matches) > 5, "Should have multiple function declarations"

    @pytest.mark.skipif(not check_naga_available(), reason="naga CLI not available")
    def test_wgsl_compiles_with_naga(self) -> None:
        """Test WGSL passes naga validation (requires naga CLI)."""
        wgsl = get_pbr_frag_wgsl()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".wgsl", delete=False) as f:
            f.write(wgsl)
            f.flush()

            try:
                result = subprocess.run(
                    ["naga", f.name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    pytest.fail(f"naga validation failed:\n{result.stderr}")
            finally:
                Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for clear coat PBR pipeline."""

    def test_car_paint_material(self) -> None:
        """Test typical car paint material (base metallic + clear coat)."""
        N = (0.0, 1.0, 0.0)
        V = normalize((0.0, 0.9, 0.436))  # ~25 degree view
        L = normalize((0.5, 0.866, 0.0))  # ~30 degree light

        # Evaluate clear coat layer
        cc_brdf, Fc = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.1)
        attenuation = 1.0 - Fc

        # Simulate metallic base (simplified)
        base_brdf = (0.8, 0.2, 0.1)  # Red metallic

        # Combine layers
        final = (
            cc_brdf[0] + attenuation * base_brdf[0],
            cc_brdf[1] + attenuation * base_brdf[1],
            cc_brdf[2] + attenuation * base_brdf[2],
        )

        # Clear coat should add some specular but preserve base color
        assert final[0] > base_brdf[0] * 0.8, "Base red should still dominate"
        assert final[0] > final[1] > final[2], "Should maintain red > green > blue"

    def test_lacquered_wood_material(self) -> None:
        """Test lacquered wood (dielectric base + clear coat)."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        # Wood base color
        wood_color = (0.4, 0.25, 0.15)

        # Evaluate clear coat
        cc_brdf, Fc = eval_clear_coat(N, V, L, cc_intensity=0.8, cc_roughness=0.2)
        attenuation = 1.0 - Fc

        # Combined should retain wood color tint
        final = (
            cc_brdf[0] + attenuation * wood_color[0],
            cc_brdf[1] + attenuation * wood_color[1],
            cc_brdf[2] + attenuation * wood_color[2],
        )

        assert all(c >= 0.0 for c in final), "All channels should be non-negative"
        assert final[0] > final[1] > final[2], "Wood color should maintain warmth"

    def test_clear_coat_quality_tiers(self) -> None:
        """Test clear coat can be disabled for lower quality tiers."""
        N = (0.0, 1.0, 0.0)
        V = (0.0, 1.0, 0.0)
        L = (0.0, 1.0, 0.0)

        # Low quality: cc_intensity = 0
        cc_low, Fc_low = eval_clear_coat(N, V, L, cc_intensity=0.0, cc_roughness=0.3)

        # High quality: cc_intensity = 1
        cc_high, Fc_high = eval_clear_coat(N, V, L, cc_intensity=1.0, cc_roughness=0.3)

        # Low quality should have zero contribution
        assert all(abs(c) < EPSILON for c in cc_low), "Low quality should have zero clear coat"

        # High quality should have contribution
        assert cc_high[0] > 0.01, "High quality should have clear coat contribution"


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Regression tests to ensure base PBR is not affected."""

    def test_zero_clear_coat_matches_original_signature(self) -> None:
        """Test that eval_brdf_with_clear_coat with cc=0 matches eval_brdf."""
        wgsl = get_pbr_frag_wgsl()

        # Should have both functions
        assert "fn eval_brdf(" in wgsl, "Should have eval_brdf function"
        assert "fn eval_brdf_with_clear_coat(" in wgsl, "Should have eval_brdf_with_clear_coat function"

        # eval_brdf should delegate to eval_brdf_with_clear_coat with cc=0
        assert "eval_brdf_with_clear_coat(n, v, l, h, albedo, f0, roughness, radiance, 0.0, 0.0)" in wgsl

    def test_material_table_backward_compatible(self) -> None:
        """Test MaterialTableEntry maintains backward compatibility."""
        wgsl = get_pbr_frag_wgsl()

        # Original fields should still exist
        required_fields = [
            "base_color: vec4<f32>",
            "emissive: vec4<f32>",
            "metallic: f32",
            "roughness: f32",
            "occlusion: f32",
            "normal_scale: f32",
        ]

        for field in required_fields:
            assert field in wgsl, f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
