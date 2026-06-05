"""
Whitebox tests for SDF Specular Lighting (T-DEMO-3.8).

Tests the implementation-aware specular lighting models:
  - Blinn-Phong specular with half-vector formulation
  - GGX/Cook-Torrance microfacet BRDF

Implementation (engine/rendering/demoscene/sdf_lighting.py):
  - calculate_specular_blinn_phong(): Point light Blinn-Phong specular
  - calculate_specular_ggx(): GGX microfacet specular
  - fresnel_schlick(): Fresnel approximation
  - distribution_ggx(): GGX normal distribution
  - geometry_smith(): Smith geometry term

WHITEBOX coverage plan for Blinn-Phong:
  Path 1: N.H = 1 (perfect reflection) -> max specular
  Path 2: N.H = 0 (perpendicular) -> zero specular
  Path 3: Roughness controls shininess exponent
  Path 4: Shadow factor attenuates specular
  Path 5: Half-vector calculation: H = normalize(V + L)

WHITEBOX coverage plan for GGX:
  Path 6: Fresnel-Schlick at normal incidence -> F0
  Path 7: Fresnel-Schlick at grazing angle -> 1.0
  Path 8: GGX NDF at perfect alignment -> high value
  Path 9: Metallic affects F0 (dielectric=0.04, metal=albedo)
  Path 10: Geometry term Smith = G1(V) * G1(L)
"""

from __future__ import annotations

import math

import pytest

from engine.rendering.demoscene.sdf_lighting import (
    calculate_specular_blinn_phong,
    calculate_specular_blinn_phong_directional,
    calculate_specular_ggx,
    calculate_specular_ggx_directional,
    roughness_to_shininess,
    fresnel_schlick,
    distribution_ggx,
    geometry_schlick_ggx,
    geometry_smith,
    vec3_normalize,
    vec3_add,
    vec3_dot,
)


# =============================================================================
# Tolerance constants
# =============================================================================

TOL = 1e-10
TOL_FLOAT = 1e-6


# =============================================================================
# Path 1: N.H = 1 (perfect reflection) -> max specular
# =============================================================================


class TestBlinnPhongPerfectReflection:
    """Perfect view-light alignment gives maximum specular."""

    def test_perfect_reflection_low_roughness(self):
        """N.H = 1 with low roughness gives high specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        # View and light both directly above -> H = (0,1,0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        roughness = 0.1  # Very smooth

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=roughness
        )

        # With N.H = 1 and low roughness, specular should be high
        # Shininess = (2/0.1^4) - 2 = 2/0.0001 - 2 = 19998
        # pow(1, 19998) = 1
        assert result[0] == pytest.approx(1.0, abs=TOL_FLOAT)

    def test_perfect_reflection_medium_roughness(self):
        """N.H = 1 with medium roughness still gives max specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        roughness = 0.5

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=roughness
        )

        # pow(1, shininess) = 1 for any shininess
        assert result[0] == pytest.approx(1.0, abs=TOL_FLOAT)


# =============================================================================
# Path 2: N.H = 0 (perpendicular) -> zero specular
# =============================================================================


class TestBlinnPhongPerpendicular:
    """Perpendicular half-vector gives zero specular."""

    def test_perpendicular_half_vector(self):
        """H perpendicular to N gives zero specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        # View from +X, light from -X -> H along X, perpendicular to N
        view_dir = (1.0, 0.0, 0.0)
        light_pos = (-10.0, 0.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3
        )

        # N.H = 0 -> pow(0, shininess) = 0
        assert result[0] == pytest.approx(0.0, abs=TOL_FLOAT)


# =============================================================================
# Path 3: Roughness controls shininess exponent
# =============================================================================


class TestRoughnessToShininess:
    """Verify roughness to shininess conversion formula."""

    def test_low_roughness_high_shininess(self):
        """Low roughness -> high shininess (sharp highlights)."""
        shininess = roughness_to_shininess(0.1)
        # shininess = (2/0.1^4) - 2 = 2/0.0001 - 2 = 19998
        expected = (2.0 / (0.1 ** 4)) - 2.0
        assert shininess == pytest.approx(expected, rel=TOL_FLOAT)
        assert shininess > 1000

    def test_medium_roughness(self):
        """Medium roughness -> moderate shininess."""
        shininess = roughness_to_shininess(0.5)
        expected = (2.0 / (0.5 ** 4)) - 2.0  # 2/0.0625 - 2 = 32 - 2 = 30
        assert shininess == pytest.approx(expected, rel=TOL_FLOAT)
        assert 20 < shininess < 50

    def test_high_roughness_low_shininess(self):
        """High roughness -> low shininess (broad highlights)."""
        shininess = roughness_to_shininess(1.0)
        # shininess = (2/1) - 2 = 0
        assert shininess == pytest.approx(0.0, abs=TOL_FLOAT)

    def test_clamped_roughness(self):
        """Very low roughness is clamped to avoid division by zero."""
        shininess = roughness_to_shininess(0.0)  # Would be division by zero
        # Clamped to 0.01
        expected = (2.0 / (0.01 ** 4)) - 2.0
        assert shininess == pytest.approx(expected, rel=TOL_FLOAT)


# =============================================================================
# Path 4: Shadow factor attenuates specular
# =============================================================================


class TestBlinnPhongShadow:
    """Shadow factor attenuates specular correctly."""

    def test_full_shadow_zero_specular(self):
        """Shadow=0 gives zero specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3, shadow_factor=0.0
        )

        assert result[0] == pytest.approx(0.0, abs=TOL)

    def test_partial_shadow(self):
        """Partial shadow proportionally reduces specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        full = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3, shadow_factor=1.0
        )
        half = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3, shadow_factor=0.5
        )

        assert half[0] == pytest.approx(0.5 * full[0], rel=TOL_FLOAT)


# =============================================================================
# Path 5: Half-vector calculation
# =============================================================================


class TestHalfVectorCalculation:
    """Verify half-vector H = normalize(V + L)."""

    def test_half_vector_symmetry(self):
        """H is symmetric: swapping V and L gives same result."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        # Case 1: view from one side, light from other
        v1 = vec3_normalize((1.0, 1.0, 0.0))
        l1_pos = (-5.0, 5.0, 0.0)
        # Case 2: swapped
        v2 = vec3_normalize((-1.0, 1.0, 0.0))
        l2_pos = (5.0, 5.0, 0.0)

        r1 = calculate_specular_blinn_phong(
            p, n, v1, l1_pos, (1.0, 1.0, 1.0), 1.0, 0.3
        )
        r2 = calculate_specular_blinn_phong(
            p, n, v2, l2_pos, (1.0, 1.0, 1.0), 1.0, 0.3
        )

        # Due to symmetry of half-vector, results should be similar
        # (not exact due to different attenuation distances)
        assert abs(r1[0] - r2[0]) < 0.1

    def test_half_vector_45_degree_view(self):
        """45-degree view angle test."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        # View at 45 degrees
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        view_dir = (inv_sqrt2, inv_sqrt2, 0.0)
        # Light directly above
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color, 1.0, 0.3
        )

        # Half-vector should be between view and light direction
        # H = normalize(view + (0,1,0)) = normalize((inv_sqrt2, inv_sqrt2+1, 0))
        # N.H = (0,1,0).(normalized) > 0
        assert result[0] > 0


# =============================================================================
# Path 6: Fresnel-Schlick at normal incidence -> F0
# =============================================================================


class TestFresnelSchlick:
    """Test Fresnel-Schlick approximation."""

    def test_normal_incidence_returns_f0(self):
        """At cos_theta=1 (normal incidence), F = F0."""
        f0 = (0.04, 0.04, 0.04)
        result = fresnel_schlick(1.0, f0)
        assert result[0] == pytest.approx(0.04, abs=TOL)
        assert result[1] == pytest.approx(0.04, abs=TOL)
        assert result[2] == pytest.approx(0.04, abs=TOL)

    def test_colored_f0(self):
        """Colored F0 (metallic) at normal incidence."""
        f0 = (0.8, 0.5, 0.2)  # Gold-like
        result = fresnel_schlick(1.0, f0)
        assert result[0] == pytest.approx(0.8, abs=TOL)
        assert result[1] == pytest.approx(0.5, abs=TOL)
        assert result[2] == pytest.approx(0.2, abs=TOL)


# =============================================================================
# Path 7: Fresnel-Schlick at grazing angle -> 1.0
# =============================================================================


class TestFresnelGrazing:
    """Fresnel approaches 1.0 at grazing angles."""

    def test_grazing_angle_approaches_one(self):
        """At cos_theta=0 (grazing), F approaches 1."""
        f0 = (0.04, 0.04, 0.04)
        result = fresnel_schlick(0.0, f0)
        # (1-0)^5 = 1, so F = f0 + (1-f0)*1 = 1
        assert result[0] == pytest.approx(1.0, abs=TOL)

    def test_near_grazing(self):
        """Near grazing angle has high Fresnel."""
        f0 = (0.04, 0.04, 0.04)
        result = fresnel_schlick(0.1, f0)
        # (1-0.1)^5 = 0.9^5 ~= 0.59
        # F = 0.04 + 0.96 * 0.59 ~= 0.61
        assert result[0] > 0.5


# =============================================================================
# Path 8: GGX NDF at perfect alignment
# =============================================================================


class TestGGXDistribution:
    """Test GGX normal distribution function."""

    def test_perfect_alignment_high_ndf(self):
        """N = H gives maximum NDF value."""
        n = (0.0, 1.0, 0.0)
        h = (0.0, 1.0, 0.0)  # Perfect alignment
        roughness = 0.3

        ndf = distribution_ggx(n, h, roughness)

        # NDF should be high at perfect alignment
        # D = a^2 / (pi * ((n.h)^2 * (a^2-1) + 1)^2)
        # With n.h = 1: D = a^2 / (pi * (a^2)^2) = 1 / (pi * a^2)
        a = roughness * roughness
        expected = 1.0 / (math.pi * a * a)
        assert ndf == pytest.approx(expected, rel=TOL_FLOAT)

    def test_perpendicular_low_ndf(self):
        """N perpendicular to H gives low NDF."""
        n = (0.0, 1.0, 0.0)
        h = (1.0, 0.0, 0.0)  # Perpendicular
        roughness = 0.3

        ndf = distribution_ggx(n, h, roughness)

        # N.H = 0, so NDF should be very small
        # D = a^2 / (pi * (0 * (a^2-1) + 1)^2) = a^2 / pi
        a = roughness * roughness
        expected = a * a / math.pi
        assert ndf == pytest.approx(expected, rel=TOL_FLOAT)

    def test_rougher_surface_broader_distribution(self):
        """Higher roughness spreads the distribution."""
        n = (0.0, 1.0, 0.0)
        h = (0.0, 1.0, 0.0)

        ndf_smooth = distribution_ggx(n, h, 0.1)
        ndf_rough = distribution_ggx(n, h, 0.9)

        # Smoother surface has narrower, taller peak
        assert ndf_smooth > ndf_rough


# =============================================================================
# Path 9: Metallic affects F0
# =============================================================================


class TestGGXMetallic:
    """Test metallic parameter affects specular."""

    def test_dielectric_f0_is_004(self):
        """Dielectric (metallic=0) uses F0 = 0.04."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        albedo = (1.0, 0.0, 0.0)  # Red

        result = calculate_specular_ggx(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3, metallic=0.0, albedo=albedo
        )

        # With metallic=0, F0 = 0.04 (achromatic)
        # So specular should be grayish, not red
        # At normal incidence, Fresnel = F0 = 0.04
        # All channels should be similar
        assert abs(result[0] - result[1]) < 0.1
        assert abs(result[1] - result[2]) < 0.1

    def test_metallic_f0_is_albedo(self):
        """Metal (metallic=1) uses F0 = albedo."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        albedo = (1.0, 0.5, 0.0)  # Orange

        result = calculate_specular_ggx(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.3, metallic=1.0, albedo=albedo
        )

        # With metallic=1, F0 = albedo (orange)
        # Specular should have color tint
        # Red > Green > Blue (like albedo ratio)
        assert result[0] > result[1]
        assert result[1] > result[2]


# =============================================================================
# Path 10: Geometry term Smith = G1(V) * G1(L)
# =============================================================================


class TestGeometrySmith:
    """Test Smith geometry function."""

    def test_geometry_schlick_ggx_formula(self):
        """Verify Schlick-GGX geometry formula."""
        n_dot_v = 0.8
        roughness = 0.5
        r = roughness + 1.0
        k = (r * r) / 8.0

        result = geometry_schlick_ggx(n_dot_v, roughness)
        expected = n_dot_v / (n_dot_v * (1.0 - k) + k)

        assert result == pytest.approx(expected, rel=TOL_FLOAT)

    def test_geometry_smith_is_product(self):
        """Smith = G1(N.V) * G1(N.L)."""
        n = (0.0, 1.0, 0.0)
        v = vec3_normalize((0.5, 1.0, 0.0))
        l = vec3_normalize((0.0, 1.0, 0.5))
        roughness = 0.4

        n_dot_v = vec3_dot(n, v)
        n_dot_l = vec3_dot(n, l)

        g1_v = geometry_schlick_ggx(max(0.0, n_dot_v), roughness)
        g1_l = geometry_schlick_ggx(max(0.0, n_dot_l), roughness)
        expected = g1_v * g1_l

        result = geometry_smith(n, v, l, roughness)

        assert result == pytest.approx(expected, rel=TOL_FLOAT)

    def test_geometry_perfect_alignment(self):
        """Perfect alignment (N=V=L) gives maximum geometry."""
        n = (0.0, 1.0, 0.0)
        v = (0.0, 1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        roughness = 0.3

        result = geometry_smith(n, v, l, roughness)

        # With N.V = N.L = 1:
        # G1 = 1 / (1*(1-k) + k) = 1/(1-k+k) = 1
        # Smith = 1 * 1 = 1
        assert result == pytest.approx(1.0, abs=TOL_FLOAT)


# =============================================================================
# Directional light variants
# =============================================================================


class TestDirectionalSpecular:
    """Test directional light specular variants."""

    def test_blinn_phong_directional(self):
        """Blinn-Phong with directional light."""
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_dir = (0.0, 1.0, 0.0)  # Light from above
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong_directional(
            n, view_dir, light_dir, light_color,
            intensity=1.0, roughness=0.3
        )

        # N = V = L = H -> N.H = 1 -> max specular
        assert result[0] == pytest.approx(1.0, abs=TOL_FLOAT)

    def test_ggx_directional(self):
        """GGX with directional light."""
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_dir = (0.0, 1.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        albedo = (0.5, 0.5, 0.5)

        result = calculate_specular_ggx_directional(
            n, view_dir, light_dir, light_color,
            intensity=1.0, roughness=0.3, metallic=0.0, albedo=albedo
        )

        # Should produce specular output
        assert result[0] > 0


# =============================================================================
# Edge cases
# =============================================================================


class TestSpecularEdgeCases:
    """Edge cases for specular calculations."""

    def test_very_rough_surface(self):
        """Very rough surface (roughness=1) produces some specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=1.0
        )

        # Shininess = 0, so pow(n_dot_h, 0) = 1 for n_dot_h > 0
        assert result[0] > 0

    def test_very_smooth_surface(self):
        """Very smooth surface produces narrow highlight."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=1.0, roughness=0.01
        )

        # Perfect alignment should still give high specular
        assert result[0] == pytest.approx(1.0, abs=TOL_FLOAT)

    def test_zero_intensity(self):
        """Zero intensity produces no specular."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        view_dir = (0.0, 1.0, 0.0)
        light_pos = (0.0, 10.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_specular_blinn_phong(
            p, n, view_dir, light_pos, light_color,
            intensity=0.0, roughness=0.3
        )

        assert result == (0.0, 0.0, 0.0)
