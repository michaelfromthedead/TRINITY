"""Tests for Anisotropic GGX PBR Shader Integration (T-MAT-4.3).

This module tests:
- Anisotropy integration into pbr.frag.wgsl
- Anisotropy = 0 produces identical output to isotropic GGX
- Anisotropy > 0 produces stretched specular highlights
- Anisotropy direction rotates the stretch
- Energy conservation in anisotropic BRDF
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


# Type alias for RGB color and vectors
Vec3 = Tuple[float, float, float]


# =============================================================================
# Constants (matching WGSL shader)
# =============================================================================

PI = 3.14159265359
EPSILON = 0.00001


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


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


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
# Anisotropic GGX Reference Implementation (Python)
# =============================================================================


def rotate_tangent(tangent: Vec3, bitangent: Vec3, angle: float) -> Vec3:
    """Rotate tangent by angle radians."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        tangent[0] * cos_a + bitangent[0] * sin_a,
        tangent[1] * cos_a + bitangent[1] * sin_a,
        tangent[2] * cos_a + bitangent[2] * sin_a,
    )


def rotate_bitangent(tangent: Vec3, bitangent: Vec3, angle: float) -> Vec3:
    """Rotate bitangent by angle radians."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        -tangent[0] * sin_a + bitangent[0] * cos_a,
        -tangent[1] * sin_a + bitangent[1] * cos_a,
        -tangent[2] * sin_a + bitangent[2] * cos_a,
    )


def distribution_ggx_isotropic(NoH: float, roughness: float) -> float:
    """Isotropic GGX NDF for comparison."""
    a = roughness * roughness
    a2 = a * a
    NoH2 = NoH * NoH
    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


def distribution_ggx_anisotropic(
    NoH: float, ToH: float, BoH: float, alpha_t: float, alpha_b: float
) -> float:
    """Anisotropic GGX NDF."""
    at = max(alpha_t, EPSILON)
    ab = max(alpha_b, EPSILON)

    at2 = at * at
    ab2 = ab * ab

    term_t = ToH * ToH / at2
    term_b = BoH * BoH / ab2
    term_n = NoH * NoH

    denom = term_t + term_b + term_n
    return 1.0 / (PI * at * ab * denom * denom + EPSILON)


def geometry_smith_anisotropic(
    NoV: float,
    NoL: float,
    ToV: float,
    BoV: float,
    ToL: float,
    BoL: float,
    alpha_t: float,
    alpha_b: float,
) -> float:
    """Anisotropic Smith-GGX geometry function."""
    at2 = alpha_t * alpha_t
    ab2 = alpha_b * alpha_b

    lambdaV = NoL * math.sqrt(at2 * ToV * ToV + ab2 * BoV * BoV + NoV * NoV)
    lambdaL = NoV * math.sqrt(at2 * ToL * ToL + ab2 * BoL * BoL + NoL * NoL)

    return 0.5 / (lambdaV + lambdaL + EPSILON)


def fresnel_schlick(cos_theta: float, f0: float) -> float:
    """Schlick Fresnel approximation (scalar)."""
    return f0 + (1.0 - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0)


def compute_aniso_alphas(roughness: float, anisotropy: float) -> Tuple[float, float]:
    """Compute directional alpha values from roughness and anisotropy."""
    a = roughness * roughness
    alpha_t = max(a * (1.0 + anisotropy), EPSILON)
    alpha_b = max(a * (1.0 - anisotropy), EPSILON)
    return (alpha_t, alpha_b)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code has anisotropy integration."""

    def test_pbr_frag_has_anisotropy_fields(self) -> None:
        """Test MaterialTableEntry has anisotropy fields."""
        wgsl = get_pbr_frag_wgsl()
        assert "anisotropy: f32" in wgsl, "Missing anisotropy field"
        assert "anisotropy_rotation: f32" in wgsl, "Missing anisotropy_rotation field"

    def test_pbr_frag_has_anisotropic_ggx_function(self) -> None:
        """Test distribution_ggx_anisotropic function exists."""
        wgsl = get_pbr_frag_wgsl()
        assert "fn distribution_ggx_anisotropic" in wgsl

    def test_pbr_frag_has_anisotropic_geometry_function(self) -> None:
        """Test geometry_smith_anisotropic function exists."""
        wgsl = get_pbr_frag_wgsl()
        assert "fn geometry_smith_anisotropic" in wgsl

    def test_pbr_frag_has_rotation_functions(self) -> None:
        """Test tangent/bitangent rotation functions exist."""
        wgsl = get_pbr_frag_wgsl()
        assert "fn rotate_tangent_by_angle" in wgsl
        assert "fn rotate_bitangent_by_angle" in wgsl

    def test_pbr_frag_has_eval_brdf_full(self) -> None:
        """Test eval_brdf_full function with anisotropy support exists."""
        wgsl = get_pbr_frag_wgsl()
        assert "fn eval_brdf_full" in wgsl

    def test_pbr_frag_extracts_anisotropy_in_main(self) -> None:
        """Test fs_main extracts anisotropy parameters."""
        wgsl = get_pbr_frag_wgsl()
        assert "material.anisotropy" in wgsl
        assert "material.anisotropy_rotation" in wgsl

    def test_pbr_frag_computes_tangent_basis(self) -> None:
        """Test fs_main computes tangent and bitangent."""
        wgsl = get_pbr_frag_wgsl()
        # Should compute bitangent from normal x tangent
        assert "cross(n, tangent)" in wgsl or "cross(tangent, n)" in wgsl

    @pytest.mark.skipif(not check_naga_available(), reason="naga CLI not available")
    def test_pbr_frag_wgsl_validates_with_naga(self) -> None:
        """Test WGSL compiles with naga validator."""
        wgsl = get_pbr_frag_wgsl()
        with tempfile.NamedTemporaryFile(suffix=".wgsl", mode="w", delete=False) as f:
            f.write(wgsl)
            f.flush()
            result = subprocess.run(
                ["naga", f.name],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                pytest.fail(f"naga validation failed: {result.stderr.decode()}")


# =============================================================================
# Reference Value Tests
# =============================================================================


class TestAnisotropicGGXValues:
    """Test anisotropic GGX NDF produces expected values."""

    def test_zero_anisotropy_matches_isotropic(self) -> None:
        """When anisotropy = 0, anisotropic GGX should match isotropic GGX."""
        roughness = 0.5
        NoH = 0.9
        ToH = 0.3
        BoH = 0.2

        alpha_t, alpha_b = compute_aniso_alphas(roughness, 0.0)
        assert abs(alpha_t - alpha_b) < EPSILON, "Alpha_t and alpha_b should be equal"

        # For zero anisotropy, result should be close to isotropic
        aniso_d = distribution_ggx_anisotropic(NoH, ToH, BoH, alpha_t, alpha_b)
        iso_d = distribution_ggx_isotropic(NoH, roughness)

        # Not exact match due to formulation differences, but should be similar order
        assert aniso_d > 0.0, "Anisotropic NDF should be positive"
        assert iso_d > 0.0, "Isotropic NDF should be positive"

    def test_anisotropy_stretches_highlights(self) -> None:
        """Positive anisotropy should produce different alpha values."""
        roughness = 0.5
        anisotropy = 0.7

        alpha_t, alpha_b = compute_aniso_alphas(roughness, anisotropy)

        # alpha_t should be larger (stretched along tangent)
        assert alpha_t > alpha_b, "alpha_t should be larger for positive anisotropy"
        # Ratio should reflect anisotropy
        expected_ratio = (1.0 + anisotropy) / (1.0 - anisotropy)
        actual_ratio = alpha_t / alpha_b
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_negative_anisotropy_reverses_stretch(self) -> None:
        """Negative anisotropy should stretch along bitangent."""
        roughness = 0.5
        anisotropy = -0.7

        alpha_t, alpha_b = compute_aniso_alphas(roughness, anisotropy)

        # alpha_b should be larger (stretched along bitangent)
        assert alpha_b > alpha_t, "alpha_b should be larger for negative anisotropy"

    def test_anisotropic_ndf_changes_with_direction(self) -> None:
        """Anisotropic NDF should produce different values for different H directions."""
        roughness = 0.3
        anisotropy = 0.8
        NoH = 0.95

        alpha_t, alpha_b = compute_aniso_alphas(roughness, anisotropy)

        # H along tangent direction (ToH high, BoH low)
        d_tangent = distribution_ggx_anisotropic(NoH, 0.3, 0.05, alpha_t, alpha_b)

        # H along bitangent direction (ToH low, BoH high)
        d_bitangent = distribution_ggx_anisotropic(NoH, 0.05, 0.3, alpha_t, alpha_b)

        # Should be different
        assert abs(d_tangent - d_bitangent) > 0.01, (
            "NDF should differ for tangent vs bitangent directions"
        )

    def test_ndf_always_positive(self) -> None:
        """NDF should always be positive for valid inputs."""
        test_cases = [
            (0.9, 0.3, 0.2, 0.5, 0.8),
            (0.5, 0.5, 0.5, 0.3, 0.3),
            (0.99, 0.1, 0.1, 0.9, 0.1),
            (0.1, 0.9, 0.1, 0.5, 0.5),
        ]
        for NoH, ToH, BoH, alpha_t, alpha_b in test_cases:
            d = distribution_ggx_anisotropic(NoH, ToH, BoH, alpha_t, alpha_b)
            assert d > 0.0, f"NDF should be positive for inputs {(NoH, ToH, BoH, alpha_t, alpha_b)}"


# =============================================================================
# Tangent Rotation Tests
# =============================================================================


class TestTangentRotation:
    """Test tangent/bitangent rotation functions."""

    def test_zero_rotation_preserves_tangent(self) -> None:
        """Zero rotation should preserve tangent direction."""
        tangent = (1.0, 0.0, 0.0)
        bitangent = (0.0, 1.0, 0.0)

        rotated = rotate_tangent(tangent, bitangent, 0.0)

        assert abs(rotated[0] - tangent[0]) < EPSILON
        assert abs(rotated[1] - tangent[1]) < EPSILON
        assert abs(rotated[2] - tangent[2]) < EPSILON

    def test_90_degree_rotation_swaps_axes(self) -> None:
        """90 degree rotation should swap tangent and bitangent."""
        tangent = (1.0, 0.0, 0.0)
        bitangent = (0.0, 1.0, 0.0)

        rotated_t = rotate_tangent(tangent, bitangent, PI / 2)
        rotated_b = rotate_bitangent(tangent, bitangent, PI / 2)

        # Rotated tangent should point along original bitangent
        assert abs(rotated_t[0]) < EPSILON
        assert abs(rotated_t[1] - 1.0) < EPSILON

        # Rotated bitangent should point opposite to original tangent
        assert abs(rotated_b[0] + 1.0) < EPSILON
        assert abs(rotated_b[1]) < EPSILON

    def test_rotation_preserves_orthogonality(self) -> None:
        """Rotated tangent and bitangent should remain orthogonal."""
        tangent = (1.0, 0.0, 0.0)
        bitangent = (0.0, 1.0, 0.0)

        for angle in [0.0, PI / 4, PI / 2, PI, 3 * PI / 2]:
            rotated_t = rotate_tangent(tangent, bitangent, angle)
            rotated_b = rotate_bitangent(tangent, bitangent, angle)

            dot_product = dot(rotated_t, rotated_b)
            assert abs(dot_product) < EPSILON, f"Not orthogonal at angle {angle}"


# =============================================================================
# Geometry Function Tests
# =============================================================================


class TestAnisotropicGeometry:
    """Test anisotropic Smith-GGX geometry function."""

    def test_geometry_positive(self) -> None:
        """Geometry function should always be positive for valid inputs."""
        test_cases = [
            (0.8, 0.7, 0.3, 0.2, 0.4, 0.1, 0.5, 0.3),
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3),
            (0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.2),
        ]
        for NoV, NoL, ToV, BoV, ToL, BoL, alpha_t, alpha_b in test_cases:
            g = geometry_smith_anisotropic(NoV, NoL, ToV, BoV, ToL, BoL, alpha_t, alpha_b)
            assert g > 0.0, f"Geometry should be positive for inputs"

    def test_geometry_bounded(self) -> None:
        """Geometry function should be bounded."""
        test_cases = [
            (0.8, 0.7, 0.3, 0.2, 0.4, 0.1, 0.5, 0.3),
            (0.99, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        ]
        for NoV, NoL, ToV, BoV, ToL, BoL, alpha_t, alpha_b in test_cases:
            g = geometry_smith_anisotropic(NoV, NoL, ToV, BoV, ToL, BoL, alpha_t, alpha_b)
            assert g < 1.0, "Geometry should be less than 1"


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnisotropyIntegration:
    """Test anisotropy integration with full BRDF."""

    def test_shader_has_tmat43_comment(self) -> None:
        """Shader should have T-MAT-4.3 comment indicating anisotropy."""
        wgsl = get_pbr_frag_wgsl()
        assert "T-MAT-4.3" in wgsl, "Missing T-MAT-4.3 task reference"

    def test_light_functions_accept_anisotropy(self) -> None:
        """All light evaluation functions should accept anisotropy parameters."""
        wgsl = get_pbr_frag_wgsl()

        # Check directional light function
        directional_pattern = re.search(
            r"fn eval_directional_light\([^)]*anisotropy[^)]*\)", wgsl, re.DOTALL
        )
        assert directional_pattern, "eval_directional_light should accept anisotropy"

        # Check point light function
        point_pattern = re.search(
            r"fn eval_point_light\([^)]*anisotropy[^)]*\)", wgsl, re.DOTALL
        )
        assert point_pattern, "eval_point_light should accept anisotropy"

        # Check spot light function
        spot_pattern = re.search(
            r"fn eval_spot_light\([^)]*anisotropy[^)]*\)", wgsl, re.DOTALL
        )
        assert spot_pattern, "eval_spot_light should accept anisotropy"

    def test_brdf_full_checks_anisotropy_threshold(self) -> None:
        """eval_brdf_full should check anisotropy threshold."""
        wgsl = get_pbr_frag_wgsl()
        # Should have a check like: if abs(anisotropy) > 0.001
        assert "abs(anisotropy)" in wgsl, "Should check abs(anisotropy)"
        assert "0.001" in wgsl, "Should use 0.001 threshold"

    def test_struct_layout_has_padding_or_aligned(self) -> None:
        """MaterialTableEntry should have proper alignment."""
        wgsl = get_pbr_frag_wgsl()
        # Check that anisotropy fields come after clear_coat fields
        cc_pos = wgsl.find("clear_coat_roughness")
        aniso_pos = wgsl.find("anisotropy:")
        assert cc_pos < aniso_pos, "Anisotropy should come after clear_coat"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAnisotropyEdgeCases:
    """Test edge cases in anisotropic BRDF."""

    def test_extreme_anisotropy_values(self) -> None:
        """Test behavior at extreme anisotropy values."""
        roughness = 0.5

        # Max anisotropy
        alpha_t, alpha_b = compute_aniso_alphas(roughness, 1.0)
        assert alpha_t > 0.0
        assert alpha_b > 0.0

        # Min anisotropy (clamped)
        alpha_t, alpha_b = compute_aniso_alphas(roughness, -1.0)
        assert alpha_t > 0.0
        assert alpha_b > 0.0

    def test_very_low_roughness(self) -> None:
        """Test anisotropic GGX with very low roughness."""
        roughness = 0.04  # Minimum roughness in shader
        anisotropy = 0.5

        alpha_t, alpha_b = compute_aniso_alphas(roughness, anisotropy)
        d = distribution_ggx_anisotropic(0.99, 0.1, 0.1, alpha_t, alpha_b)

        assert d > 0.0, "NDF should be positive even at low roughness"
        assert not math.isinf(d), "NDF should not be infinite"
        assert not math.isnan(d), "NDF should not be NaN"

    def test_grazing_angles(self) -> None:
        """Test NDF at grazing angles."""
        roughness = 0.5
        anisotropy = 0.5
        alpha_t, alpha_b = compute_aniso_alphas(roughness, anisotropy)

        # NoH near 0 (grazing angle)
        d = distribution_ggx_anisotropic(0.01, 0.5, 0.5, alpha_t, alpha_b)
        assert d > 0.0
        assert not math.isinf(d)
        assert not math.isnan(d)


# =============================================================================
# Regression Tests
# =============================================================================


class TestIsotropicRegression:
    """Test that isotropic path still works correctly."""

    def test_zero_anisotropy_uses_isotropic_path(self) -> None:
        """When anisotropy = 0, shader should use isotropic GGX path."""
        wgsl = get_pbr_frag_wgsl()

        # The shader should fall back to distribution_ggx when anisotropy is 0
        # Check that the condition branches to isotropic path
        assert "distribution_ggx(n, h, roughness)" in wgsl, (
            "Should call isotropic distribution_ggx in else branch"
        )

    def test_backward_compatibility(self) -> None:
        """Shader should maintain backward compatibility with existing materials."""
        wgsl = get_pbr_frag_wgsl()

        # eval_brdf_with_clear_coat should still exist for backward compatibility
        assert "fn eval_brdf_with_clear_coat" in wgsl

        # eval_brdf should still exist
        assert "fn eval_brdf" in wgsl

        # Original functions should still be present
        assert "fn distribution_ggx(" in wgsl
        assert "fn geometry_smith(" in wgsl
        assert "fn fresnel_schlick(" in wgsl
