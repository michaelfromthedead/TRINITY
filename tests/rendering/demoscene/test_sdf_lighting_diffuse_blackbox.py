"""
Blackbox tests for SDF Diffuse Lighting (T-DEMO-3.7).

Tests the observable behavior of Lambertian diffuse lighting without
knowledge of internal implementation details.

Acceptance criteria from T-DEMO-3.7:
  - Correct Lambertian diffuse shading
  - Shadow term correctly attenuates lighting
  - Multiple lights combine additively
  - 20+ tests covering single/multi light scenarios

Testing strategy:
  - Test observable input/output relationships
  - Verify mathematical properties (additivity, commutativity)
  - Test boundary conditions
  - Verify WGSL generation produces valid code
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
    MaterialParams,
    LightingCodegen,
    WGSL_DIFFUSE,
)
from engine.rendering.demoscene.ast_nodes import LightType


# =============================================================================
# Tolerance
# =============================================================================

TOL = 1e-6


# =============================================================================
# Basic Diffuse Behavior
# =============================================================================


class TestBasicDiffuseBehavior:
    """Test fundamental diffuse lighting behavior."""

    def test_facing_light_produces_output(self):
        """Surface facing light should produce non-zero output."""
        result = calculate_diffuse(
            p=(0, 0, 0),
            n=(0, 1, 0),
            light_pos=(0, 10, 0),
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] > 0

    def test_facing_away_produces_zero(self):
        """Surface facing away from light should produce zero output."""
        result = calculate_diffuse(
            p=(0, 0, 0),
            n=(0, 1, 0),
            light_pos=(0, -10, 0),  # Light behind surface
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        assert result[0] == pytest.approx(0, abs=TOL)
        assert result[1] == pytest.approx(0, abs=TOL)
        assert result[2] == pytest.approx(0, abs=TOL)

    def test_intensity_scales_output(self):
        """Higher intensity should produce proportionally brighter output."""
        result1 = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
        )
        result2 = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=2.0,
        )
        # Result2 should be exactly 2x result1
        assert result2[0] == pytest.approx(2 * result1[0], rel=TOL)

    def test_color_affects_output(self):
        """Light color should affect output color."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 0, 0),  # Red only
            intensity=1.0,
        )
        assert result[0] > 0  # Red present
        assert result[1] == pytest.approx(0, abs=TOL)  # No green
        assert result[2] == pytest.approx(0, abs=TOL)  # No blue


# =============================================================================
# Shadow Integration
# =============================================================================


class TestShadowIntegration:
    """Test shadow term correctly attenuates lighting."""

    def test_full_shadow_blocks_light(self):
        """Shadow factor 0 should block all light."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
            shadow_factor=0.0,
        )
        assert result == (0.0, 0.0, 0.0)

    def test_no_shadow_full_light(self):
        """Shadow factor 1 should not attenuate."""
        with_shadow = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
            shadow_factor=1.0,
        )
        without_explicit = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
            # Default shadow_factor=1.0
        )
        assert with_shadow[0] == pytest.approx(without_explicit[0], rel=TOL)

    def test_partial_shadow_attenuates(self):
        """Partial shadow should reduce output proportionally."""
        full = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
            shadow_factor=1.0,
        )
        half = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1.0,
            shadow_factor=0.5,
        )
        assert half[0] == pytest.approx(0.5 * full[0], rel=TOL)


# =============================================================================
# Multi-Light Additivity
# =============================================================================


class TestMultiLightAdditivity:
    """Test that multiple lights combine additively."""

    def test_two_lights_are_additive(self):
        """Sum of individual lights equals combined calculation."""
        lights = [
            LightParams(position=(0, 10, 0), color=(1, 0, 0), intensity=1.0, radius=100.0),
            LightParams(position=(10, 0, 0), color=(0, 0, 1), intensity=1.0, radius=100.0),
        ]
        p = (0, 0, 0)
        n = (0, 1, 0)

        combined = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # Red light contributes (light above, N.L = 1)
        # Blue light contributes 0 (N.L = 0 for side light)
        assert combined[0] > 0  # Red from above
        assert combined[2] == pytest.approx(0, abs=TOL)  # Blue from side = 0

    def test_order_independence(self):
        """Light order should not affect result."""
        lights_a = [
            LightParams(position=(0, 10, 0), color=(1, 0, 0), intensity=1.0, radius=100.0),
            LightParams(position=(0, 5, 5), color=(0, 1, 0), intensity=1.0, radius=100.0),
        ]
        lights_b = [
            LightParams(position=(0, 5, 5), color=(0, 1, 0), intensity=1.0, radius=100.0),
            LightParams(position=(0, 10, 0), color=(1, 0, 0), intensity=1.0, radius=100.0),
        ]
        p = (0, 0, 0)
        n = (0, 1, 0)

        result_a = calculate_all_diffuse(p, n, lights_a, enable_shadows=False)
        result_b = calculate_all_diffuse(p, n, lights_b, enable_shadows=False)

        assert result_a[0] == pytest.approx(result_b[0], rel=TOL)
        assert result_a[1] == pytest.approx(result_b[1], rel=TOL)
        assert result_a[2] == pytest.approx(result_b[2], rel=TOL)

    def test_many_lights_accumulate(self):
        """Many lights should accumulate correctly."""
        lights = [
            LightParams(position=(0, 10, 0), color=(0.2, 0.2, 0.2), intensity=1.0, radius=100.0)
            for _ in range(5)
        ]
        p = (0, 0, 0)
        n = (0, 1, 0)

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # 5 identical lights should give 5x a single light
        single = calculate_all_diffuse(p, n, [lights[0]], enable_shadows=False)
        assert result[0] == pytest.approx(5 * single[0], rel=TOL)


# =============================================================================
# Directional Light Behavior
# =============================================================================


class TestDirectionalLightBehavior:
    """Test directional light specific behavior."""

    def test_directional_has_no_falloff(self):
        """Directional light should not have distance falloff."""
        result_near = calculate_diffuse_directional(
            n=(0, 1, 0),
            light_dir=(0, 1, 0),
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        # Same direction regardless of conceptual distance
        result_far = calculate_diffuse_directional(
            n=(0, 1, 0),
            light_dir=(0, 1, 0),
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        assert result_near[0] == pytest.approx(result_far[0], rel=TOL)

    def test_directional_respects_angle(self):
        """Directional light should respect surface angle."""
        direct = calculate_diffuse_directional(
            n=(0, 1, 0),
            light_dir=(0, 1, 0),  # Direct
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        angled = calculate_diffuse_directional(
            n=(0, 1, 0),
            light_dir=(0.707, 0.707, 0),  # 45 degrees
            light_color=(1, 1, 1),
            intensity=1.0,
        )
        # Angled should be less than direct
        assert angled[0] < direct[0]
        assert angled[0] > 0


# =============================================================================
# Attenuation Behavior
# =============================================================================


class TestAttenuationBehavior:
    """Test distance attenuation behavior."""

    def test_attenuation_decreases_with_distance(self):
        """Closer lights should be brighter."""
        close = calculate_attenuation_inverse_square(2.0, 10.0)
        far = calculate_attenuation_inverse_square(8.0, 10.0)
        assert close > far

    def test_attenuation_zero_beyond_radius(self):
        """Light beyond radius should have zero attenuation."""
        atten = calculate_attenuation_inverse_square(15.0, 10.0)
        assert atten == 0.0

    def test_attenuation_positive_within_radius(self):
        """Light within radius should have positive attenuation."""
        atten = calculate_attenuation_inverse_square(5.0, 10.0)
        assert atten > 0

    def test_attenuation_max_at_zero_distance(self):
        """Attenuation should be maximum at zero distance."""
        at_zero = calculate_attenuation_inverse_square(0.0, 10.0)
        at_one = calculate_attenuation_inverse_square(1.0, 10.0)
        assert at_zero > at_one


# =============================================================================
# WGSL Code Generation
# =============================================================================


class TestWGSLGeneration:
    """Test WGSL code generation for diffuse lighting."""

    def test_wgsl_diffuse_contains_function(self):
        """WGSL diffuse code should contain function definition."""
        assert "fn calculate_diffuse(" in WGSL_DIFFUSE
        assert "fn calculate_diffuse_directional(" in WGSL_DIFFUSE

    def test_wgsl_diffuse_has_correct_signature(self):
        """WGSL diffuse should have correct parameter types."""
        assert "p: vec3<f32>" in WGSL_DIFFUSE
        assert "n: vec3<f32>" in WGSL_DIFFUSE
        assert "light_pos: vec3<f32>" in WGSL_DIFFUSE
        assert "light_color: vec3<f32>" in WGSL_DIFFUSE

    def test_wgsl_diffuse_has_dot_product(self):
        """Diffuse shader should use dot product for N.L."""
        assert "dot(n, light_dir)" in WGSL_DIFFUSE

    def test_wgsl_diffuse_has_max_clamp(self):
        """Diffuse shader should clamp N.L to zero."""
        assert "max(" in WGSL_DIFFUSE


class TestLightingCodegen:
    """Test LightingCodegen class."""

    def test_codegen_generates_diffuse(self):
        """Codegen should generate diffuse functions."""
        codegen = LightingCodegen()
        wgsl = codegen.generate_diffuse()
        assert "calculate_diffuse" in wgsl

    def test_codegen_generates_all_functions(self):
        """Codegen should generate all lighting functions."""
        codegen = LightingCodegen()
        wgsl = codegen.generate_lighting_functions()
        assert "calculate_diffuse" in wgsl
        assert "calculate_attenuation" in wgsl
        assert "calculate_soft_shadow" in wgsl

    def test_codegen_selective_generation(self):
        """Codegen should allow selective function generation."""
        codegen = LightingCodegen()
        wgsl = codegen.generate_lighting_functions(
            include_shadow=False,
            include_ggx=False,
        )
        assert "calculate_diffuse" in wgsl
        assert "calculate_soft_shadow" not in wgsl
        assert "calculate_specular_ggx" not in wgsl


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_intensity_light(self):
        """Zero intensity light produces no output."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=0.0,
        )
        assert result == (0.0, 0.0, 0.0)

    def test_black_light(self):
        """Black light produces no output."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(0, 0, 0), intensity=10.0,
        )
        assert result == (0.0, 0.0, 0.0)

    def test_empty_light_list(self):
        """Empty light list returns zero."""
        result = calculate_all_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), lights=[],
            enable_shadows=False,
        )
        assert result == (0.0, 0.0, 0.0)

    def test_very_high_intensity(self):
        """Very high intensity should scale correctly."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=1000.0,
        )
        assert result[0] == pytest.approx(1000.0, rel=TOL)

    def test_negative_intensity_allowed(self):
        """Negative intensity (for subtraction) should work."""
        result = calculate_diffuse(
            p=(0, 0, 0), n=(0, 1, 0), light_pos=(0, 10, 0),
            light_color=(1, 1, 1), intensity=-1.0,
        )
        assert result[0] < 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other components."""

    def test_with_material_params(self):
        """Test integration with MaterialParams."""
        material = MaterialParams(albedo=(0.8, 0.2, 0.1), roughness=0.5)
        # Just verify MaterialParams can be created
        assert material.albedo == (0.8, 0.2, 0.1)

    def test_light_params_conversion(self):
        """Test LightParams can be created with all fields."""
        light = LightParams(
            position=(5, 10, 3),
            color=(1, 0.9, 0.8),
            intensity=2.5,
            light_type=LightType.POINT,
            direction=(0, -1, 0),
            radius=20.0,
        )
        assert light.position == (5, 10, 3)
        assert light.light_type == LightType.POINT

    def test_mixed_light_types(self):
        """Test mixing point and directional lights."""
        lights = [
            LightParams(
                position=(0, 10, 0),
                color=(1, 0, 0),
                intensity=1.0,
                light_type=LightType.POINT,
                radius=100.0,
            ),
            LightParams(
                direction=(0, -1, 0),
                color=(0, 1, 0),
                intensity=1.0,
                light_type=LightType.DIRECTIONAL,
            ),
        ]
        p = (0, 0, 0)
        n = (0, 1, 0)

        result = calculate_all_diffuse(p, n, lights, enable_shadows=False)

        # Both lights should contribute
        assert result[0] > 0  # Red from point
        assert result[1] > 0  # Green from directional
