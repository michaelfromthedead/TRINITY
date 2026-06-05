"""
Whitebox tests for SDF Diffuse Lighting (T-DEMO-3.7).

Tests the implementation-aware Lambertian diffuse lighting model including:
  - Single light diffuse calculation
  - Multi-light accumulation
  - Shadow integration
  - Directional vs point light handling

Implementation (engine/rendering/demoscene/sdf_lighting.py):
  - calculate_diffuse(): Point light Lambertian diffuse
  - calculate_diffuse_directional(): Directional light diffuse
  - calculate_all_diffuse(): Multi-light accumulation

WHITEBOX coverage plan:
  Path 1: N.L = 1 (surface facing light directly) -> full diffuse
  Path 2: N.L = 0 (surface perpendicular to light) -> zero diffuse
  Path 3: N.L < 0 (surface facing away) -> clamped to zero
  Path 4: N.L = 0.5 (45 degree angle) -> half diffuse
  Path 5: Shadow factor = 0 -> zero diffuse
  Path 6: Shadow factor = 0.5 -> half diffuse
  Path 7: Attenuation = 0 -> zero diffuse
  Path 8: Attenuation = 0.5 -> half diffuse
  Path 9: Multi-light additive accumulation
  Path 10: Directional light direction handling
"""

from __future__ import annotations

import math

import pytest

from engine.rendering.demoscene.sdf_lighting import (
    calculate_diffuse,
    calculate_diffuse_directional,
    calculate_all_diffuse,
    calculate_attenuation_inverse_square,
    LightParams,
    LightType,
    vec3_normalize,
    vec3_dot,
    vec3_length,
)


# =============================================================================
# Tolerance constants
# =============================================================================

TOL = 1e-10
TOL_FLOAT = 1e-6


# =============================================================================
# Path 1: N.L = 1 (surface facing light directly)
# =============================================================================


class TestDiffuseDirectFacing:
    """Surface normal pointing directly at light -> max diffuse."""

    def test_direct_facing_white_light(self):
        """N.L=1 with white light intensity=1 -> full diffuse."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)  # Normal pointing up
        light_pos = (0.0, 5.0, 0.0)  # Light directly above
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # N.L = 1.0, so diffuse = 1.0 * color * intensity
        assert result[0] == pytest.approx(1.0, abs=TOL)
        assert result[1] == pytest.approx(1.0, abs=TOL)
        assert result[2] == pytest.approx(1.0, abs=TOL)

    def test_direct_facing_colored_light(self):
        """N.L=1 with colored light."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 0.5, 0.2)
        intensity = 2.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # diffuse = N.L * color * intensity = 1.0 * (1, 0.5, 0.2) * 2.0
        assert result[0] == pytest.approx(2.0, abs=TOL)
        assert result[1] == pytest.approx(1.0, abs=TOL)
        assert result[2] == pytest.approx(0.4, abs=TOL)

    def test_direct_facing_different_axis(self):
        """N.L=1 along X axis."""
        p = (0.0, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)  # Normal pointing +X
        light_pos = (5.0, 0.0, 0.0)  # Light in +X direction
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        assert result[0] == pytest.approx(1.0, abs=TOL)
        assert result[1] == pytest.approx(1.0, abs=TOL)
        assert result[2] == pytest.approx(1.0, abs=TOL)


# =============================================================================
# Path 2: N.L = 0 (surface perpendicular to light)
# =============================================================================


class TestDiffusePerpendicular:
    """Surface normal perpendicular to light direction -> zero diffuse."""

    def test_perpendicular_y_normal_x_light(self):
        """Normal up, light from +X -> N.L = 0."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)  # Normal pointing up
        light_pos = (5.0, 0.0, 0.0)  # Light in +X
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # N.L = 0
        assert result[0] == pytest.approx(0.0, abs=TOL)
        assert result[1] == pytest.approx(0.0, abs=TOL)
        assert result[2] == pytest.approx(0.0, abs=TOL)

    def test_perpendicular_z_normal_y_light(self):
        """Normal +Z, light from +Y -> N.L = 0."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 0.0, 1.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 2.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        assert result[0] == pytest.approx(0.0, abs=TOL)
        assert result[1] == pytest.approx(0.0, abs=TOL)
        assert result[2] == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Path 3: N.L < 0 (surface facing away)
# =============================================================================


class TestDiffuseBackfacing:
    """Surface normal facing away from light -> clamped to zero."""

    def test_backfacing_opposite_direction(self):
        """Normal +Y, light from -Y -> N.L < 0, clamped to 0."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)  # Normal pointing up
        light_pos = (0.0, -5.0, 0.0)  # Light below
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # N.L = -1, clamped to 0
        assert result[0] == pytest.approx(0.0, abs=TOL)
        assert result[1] == pytest.approx(0.0, abs=TOL)
        assert result[2] == pytest.approx(0.0, abs=TOL)

    def test_backfacing_diagonal(self):
        """Diagonal backfacing case."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (1.0, -5.0, 1.0)  # Light mostly below
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # Light direction is mostly downward, N.L < 0
        assert result[0] == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Path 4: N.L = 0.5 (45 degree angle)
# =============================================================================


class TestDiffuseAngled:
    """Surface at angle to light -> partial diffuse."""

    def test_45_degree_angle(self):
        """Normal at 45 degrees to light direction."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)  # Normal up
        # Light at 45 degrees: equal X and Y components
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        light_pos = (5.0 * inv_sqrt2, 5.0 * inv_sqrt2, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, light_pos, light_color, intensity)

        # Light direction = normalize(light_pos - p) = (inv_sqrt2, inv_sqrt2, 0)
        # N.L = (0,1,0) . (inv_sqrt2, inv_sqrt2, 0) = inv_sqrt2 ~= 0.707
        expected_ndotl = inv_sqrt2
        assert result[0] == pytest.approx(expected_ndotl, abs=TOL_FLOAT)

    def test_60_degree_angle(self):
        """Normal at 60 degrees to light direction."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        # cos(60) = 0.5, so light at angle where N.L = 0.5
        # Light direction (sin60, cos60, 0) = (sqrt3/2, 0.5, 0)
        light_pos = (math.sqrt(3.0) * 5.0, 2.5, 0.0)
        light_pos_norm = vec3_normalize(light_pos)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(p, n, (10.0, 5.0 / math.sqrt(3.0), 0.0), light_color, intensity)

        # Verify the angle gives expected N.L
        light_dir = vec3_normalize((10.0, 5.0 / math.sqrt(3.0), 0.0))
        expected_ndotl = vec3_dot(n, light_dir)
        assert result[0] == pytest.approx(expected_ndotl, abs=TOL_FLOAT)


# =============================================================================
# Path 5: Shadow factor = 0
# =============================================================================


class TestDiffuseShadowZero:
    """Shadow factor = 0 -> zero diffuse (fully shadowed)."""

    def test_full_shadow_direct_facing(self):
        """Direct facing but fully shadowed."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 2.0

        result = calculate_diffuse(
            p, n, light_pos, light_color, intensity,
            shadow_factor=0.0
        )

        assert result[0] == pytest.approx(0.0, abs=TOL)
        assert result[1] == pytest.approx(0.0, abs=TOL)
        assert result[2] == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Path 6: Shadow factor = 0.5
# =============================================================================


class TestDiffuseShadowPartial:
    """Shadow factor = 0.5 -> half diffuse (partial shadow)."""

    def test_half_shadow_direct_facing(self):
        """Direct facing with half shadow."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(
            p, n, light_pos, light_color, intensity,
            shadow_factor=0.5
        )

        # N.L = 1, shadow = 0.5, result = 0.5
        assert result[0] == pytest.approx(0.5, abs=TOL)

    def test_quarter_shadow(self):
        """25% lit (75% shadow)."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(
            p, n, light_pos, light_color, intensity,
            shadow_factor=0.25
        )

        assert result[0] == pytest.approx(0.25, abs=TOL)


# =============================================================================
# Path 7: Attenuation = 0
# =============================================================================


class TestDiffuseAttenuationZero:
    """Attenuation = 0 -> zero diffuse (light too far)."""

    def test_zero_attenuation(self):
        """Zero attenuation means no light contribution."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 2.0

        result = calculate_diffuse(
            p, n, light_pos, light_color, intensity,
            attenuation=0.0
        )

        assert result[0] == pytest.approx(0.0, abs=TOL)

    def test_attenuation_beyond_radius(self):
        """Verify attenuation function returns 0 beyond radius."""
        atten = calculate_attenuation_inverse_square(15.0, 10.0)
        assert atten == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Path 8: Attenuation = 0.5
# =============================================================================


class TestDiffuseAttenuationPartial:
    """Partial attenuation -> reduced diffuse."""

    def test_half_attenuation(self):
        """Half attenuation."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse(
            p, n, light_pos, light_color, intensity,
            attenuation=0.5
        )

        assert result[0] == pytest.approx(0.5, abs=TOL)

    def test_attenuation_at_half_radius(self):
        """Test attenuation function at half radius."""
        # At half radius, attenuation should be significant but not zero
        atten = calculate_attenuation_inverse_square(5.0, 10.0)
        # (1 - (0.5)^2)^2 * 1/(25+1) = (1 - 0.25)^2 / 26 = 0.75^2 / 26 ~= 0.0216
        assert atten > 0.0
        assert atten < 1.0


# =============================================================================
# Path 9: Multi-light additive accumulation
# =============================================================================


class TestMultiLightAccumulation:
    """Multiple lights accumulate additively."""

    def test_two_equal_lights(self):
        """Two equal lights double the contribution."""
        lights = [
            LightParams(position=(0.0, 5.0, 0.0), color=(1.0, 0.0, 0.0), intensity=1.0, radius=100.0),
            LightParams(position=(0.0, 5.0, 0.0), color=(0.0, 1.0, 0.0), intensity=1.0, radius=100.0),
        ]
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # Two lights add independently: red + green
        # With attenuation from calculate_attenuation_inverse_square
        assert result[0] > 0.0  # Red contribution
        assert result[1] > 0.0  # Green contribution
        assert result[2] == pytest.approx(0.0, abs=TOL)  # No blue

    def test_three_lights_rgb(self):
        """Three lights with RGB colors accumulate to white-ish."""
        lights = [
            LightParams(position=(0.0, 5.0, 0.0), color=(1.0, 0.0, 0.0), intensity=1.0, radius=100.0),
            LightParams(position=(0.0, 5.0, 0.0), color=(0.0, 1.0, 0.0), intensity=1.0, radius=100.0),
            LightParams(position=(0.0, 5.0, 0.0), color=(0.0, 0.0, 1.0), intensity=1.0, radius=100.0),
        ]
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # All three components should be equal (grayscale)
        assert result[0] == pytest.approx(result[1], rel=TOL_FLOAT)
        assert result[1] == pytest.approx(result[2], rel=TOL_FLOAT)

    def test_lights_from_different_directions(self):
        """Lights from different directions contribute based on angle."""
        lights = [
            LightParams(position=(0.0, 5.0, 0.0), color=(1.0, 1.0, 1.0), intensity=1.0, radius=100.0),  # Above
            LightParams(position=(5.0, 0.0, 0.0), color=(1.0, 1.0, 1.0), intensity=1.0, radius=100.0),  # Side
        ]
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)  # Facing up

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # Only the light above contributes (N.L > 0 for that one)
        # Side light has N.L = 0
        assert result[0] > 0.0


# =============================================================================
# Path 10: Directional light direction handling
# =============================================================================


class TestDirectionalLightDiffuse:
    """Directional light uses direction instead of position."""

    def test_directional_down(self):
        """Directional light pointing down on upward-facing surface."""
        n = (0.0, 1.0, 0.0)
        # Direction stored as "where light shines", so (0, -1, 0) means light shines downward
        # We negate it to get direction TO the light
        light_dir = (0.0, 1.0, 0.0)  # Already pointing toward light (inverted by caller)
        light_color = (1.0, 1.0, 1.0)
        intensity = 2.0

        result = calculate_diffuse_directional(
            n, light_dir, light_color, intensity
        )

        # N.L = 1, result = intensity
        assert result[0] == pytest.approx(2.0, abs=TOL)

    def test_directional_angled(self):
        """Directional light at 45 degree angle."""
        n = (0.0, 1.0, 0.0)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        light_dir = (inv_sqrt2, inv_sqrt2, 0.0)  # 45 degrees from vertical
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse_directional(
            n, light_dir, light_color, intensity
        )

        # N.L = inv_sqrt2
        assert result[0] == pytest.approx(inv_sqrt2, abs=TOL_FLOAT)

    def test_directional_with_shadow(self):
        """Directional light with shadow factor."""
        n = (0.0, 1.0, 0.0)
        light_dir = (0.0, 1.0, 0.0)
        light_color = (1.0, 1.0, 1.0)
        intensity = 1.0

        result = calculate_diffuse_directional(
            n, light_dir, light_color, intensity,
            shadow_factor=0.3
        )

        assert result[0] == pytest.approx(0.3, abs=TOL)

    def test_directional_in_multi_light(self):
        """Directional light in multi-light accumulation."""
        lights = [
            LightParams(
                position=(0.0, 0.0, 0.0),  # Position ignored for directional
                direction=(0.0, -1.0, 0.0),  # Points down, will be inverted
                color=(1.0, 0.0, 0.0),
                intensity=1.0,
                light_type=LightType.DIRECTIONAL,
            ),
        ]
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # Red light from above
        assert result[0] > 0.0
        assert result[1] == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Edge cases
# =============================================================================


class TestDiffuseEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_intensity(self):
        """Zero intensity light produces no diffuse."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_diffuse(p, n, light_pos, light_color, intensity=0.0)

        assert result[0] == pytest.approx(0.0, abs=TOL)

    def test_black_light_color(self):
        """Black light color produces no diffuse."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 5.0, 0.0)
        light_color = (0.0, 0.0, 0.0)

        result = calculate_diffuse(p, n, light_pos, light_color, intensity=10.0)

        assert result[0] == pytest.approx(0.0, abs=TOL)
        assert result[1] == pytest.approx(0.0, abs=TOL)
        assert result[2] == pytest.approx(0.0, abs=TOL)

    def test_very_close_light(self):
        """Light very close to surface."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
        light_pos = (0.0, 0.01, 0.0)  # Very close
        light_color = (1.0, 1.0, 1.0)

        result = calculate_diffuse(p, n, light_pos, light_color, intensity=1.0)

        # Should still work, N.L = 1
        assert result[0] == pytest.approx(1.0, abs=TOL)

    def test_empty_lights_list(self):
        """Empty lights list returns zero."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)

        result = calculate_all_diffuse(p, n, [], enable_shadows=False)

        assert result == (0.0, 0.0, 0.0)

    def test_unnormalized_normal_still_works(self):
        """Even unnormalized normal produces reasonable result."""
        p = (0.0, 0.0, 0.0)
        n = (0.0, 2.0, 0.0)  # Not normalized
        light_pos = (0.0, 5.0, 0.0)
        light_color = (1.0, 1.0, 1.0)

        result = calculate_diffuse(p, n, light_pos, light_color, intensity=1.0)

        # N.L = 2 (because not normalized), but still works
        # This is a known behavior - caller should normalize
        assert result[0] > 0.0
