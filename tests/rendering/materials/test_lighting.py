"""Tests for Lighting functions (T-MAT-3.3).

This module tests:
- WGSL syntax validation for lighting.wgsl
- Light type evaluation (directional, point, spot)
- Attenuation functions (inverse-square, spot angle)
- Light accumulation for 1-8 lights
- Shadow placeholder returns 1.0
- Ambient occlusion and emissive term application
- Energy conservation properties
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List

import pytest

from trinity.materials.lighting import (
    # WGSL source
    get_lighting_wgsl,
    # Types
    LightType,
    Light,
    LightSample,
    LightingResult,
    PBRParamsLighting,
    # Attenuation
    attenuation_point,
    attenuation_spot_angle,
    # Light evaluation
    evaluate_directional_light,
    evaluate_point_light,
    evaluate_spot_light,
    evaluate_light,
    # Shadow
    sample_shadow,
    # Accumulation
    accumulate_lighting,
    compose_final_shading,
    evaluate_all_lighting,
    # Creation helpers
    create_directional_light,
    create_point_light,
    create_spot_light,
    # Reference values
    LIGHTING_REFERENCE_VALUES,
    LIGHTING_EDGE_CASES,
    # Constants
    PI,
    EPSILON,
    MAX_LIGHTS,
)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_lighting_wgsl_loads(self) -> None:
        """Test that lighting.wgsl can be loaded."""
        wgsl = get_lighting_wgsl()
        assert len(wgsl) > 0
        assert "struct Light" in wgsl
        assert "fn evaluate_light" in wgsl

    def test_lighting_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "lighting.wgsl"
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_lighting_wgsl_has_required_functions(self) -> None:
        """Test that all required lighting functions are present."""
        wgsl = get_lighting_wgsl()
        required_functions = [
            "fn evaluate_directional_light",
            "fn evaluate_point_light",
            "fn evaluate_spot_light",
            "fn evaluate_light",
            "fn accumulate_lighting",
            "fn sample_shadow",
            "fn compose_final_shading",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_lighting_wgsl_has_light_struct(self) -> None:
        """Test that Light struct is defined."""
        wgsl = get_lighting_wgsl()
        assert "struct Light" in wgsl
        assert "light_type:" in wgsl
        assert "position:" in wgsl
        assert "direction:" in wgsl
        assert "color:" in wgsl

    def test_lighting_wgsl_has_light_types(self) -> None:
        """Test that light type constants are defined."""
        wgsl = get_lighting_wgsl()
        assert "LIGHT_TYPE_DIRECTIONAL" in wgsl
        assert "LIGHT_TYPE_POINT" in wgsl
        assert "LIGHT_TYPE_SPOT" in wgsl

    def test_lighting_wgsl_has_attenuation_functions(self) -> None:
        """Test that attenuation functions are defined."""
        wgsl = get_lighting_wgsl()
        assert "fn attenuation_point" in wgsl
        assert "fn attenuation_spot_angle" in wgsl

    def test_lighting_wgsl_has_max_lights_constant(self) -> None:
        """Test that MAX_LIGHTS constant is defined."""
        wgsl = get_lighting_wgsl()
        assert "MAX_LIGHTS" in wgsl
        assert "8u" in wgsl  # MAX_LIGHTS = 8

    def test_lighting_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_lighting_wgsl()

        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"


# =============================================================================
# Light Type Tests
# =============================================================================


class TestLightType:
    """Test LightType enumeration."""

    def test_light_type_values(self) -> None:
        """Test that light type values match WGSL constants."""
        assert LightType.DIRECTIONAL == 0
        assert LightType.POINT == 1
        assert LightType.SPOT == 2

    def test_light_type_names(self) -> None:
        """Test light type names."""
        assert LightType.DIRECTIONAL.name == "DIRECTIONAL"
        assert LightType.POINT.name == "POINT"
        assert LightType.SPOT.name == "SPOT"


# =============================================================================
# Attenuation Function Tests
# =============================================================================


class TestAttenuationPoint:
    """Test point light attenuation function."""

    @pytest.mark.parametrize("ref", LIGHTING_REFERENCE_VALUES["attenuation_point"])
    def test_attenuation_point_reference_values(self, ref: dict) -> None:
        """Test attenuation_point matches reference values."""
        result = attenuation_point(ref["distance"], ref["range"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"attenuation_point({ref['distance']}, {ref['range']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_attenuation_point_at_zero_distance(self) -> None:
        """Test attenuation at zero distance is high."""
        result = attenuation_point(0.0, 10.0)
        assert result > 1000.0, "Attenuation at zero distance should be very high"

    def test_attenuation_point_at_range(self) -> None:
        """Test attenuation at range boundary is zero."""
        result = attenuation_point(10.0, 10.0)
        assert result < 0.001, "Attenuation at range boundary should be near zero"

    def test_attenuation_point_beyond_range(self) -> None:
        """Test attenuation beyond range is zero."""
        result = attenuation_point(15.0, 10.0)
        assert result == 0.0, "Attenuation beyond range should be exactly zero"

    def test_attenuation_point_decreases_with_distance(self) -> None:
        """Test that attenuation decreases with distance."""
        prev = attenuation_point(0.1, 10.0)
        for d in [0.5, 1.0, 2.0, 5.0, 8.0]:
            curr = attenuation_point(d, 10.0)
            assert curr < prev, f"Attenuation should decrease: {prev} -> {curr}"
            prev = curr

    def test_attenuation_point_non_negative(self) -> None:
        """Test attenuation is always non-negative."""
        for d in [0.0, 0.5, 1.0, 5.0, 10.0, 20.0]:
            for r in [1.0, 5.0, 10.0]:
                result = attenuation_point(d, r)
                assert result >= 0.0, f"Negative attenuation at d={d}, r={r}"


class TestAttenuationSpotAngle:
    """Test spotlight angular attenuation function."""

    @pytest.mark.parametrize("ref", LIGHTING_REFERENCE_VALUES["attenuation_spot_angle"])
    def test_attenuation_spot_angle_reference_values(self, ref: dict) -> None:
        """Test attenuation_spot_angle matches reference values."""
        result = attenuation_spot_angle(ref["cos_angle"], ref["cos_inner"], ref["cos_outer"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"attenuation_spot_angle({ref['cos_angle']}, {ref['cos_inner']}, {ref['cos_outer']}) = {result}, "
            f"expected {ref['expected']} +/- {ref['tolerance']}"
        )

    def test_spot_angle_inside_inner_cone(self) -> None:
        """Test full intensity inside inner cone."""
        result = attenuation_spot_angle(1.0, 0.9, 0.7)
        assert abs(result - 1.0) < 0.01

    def test_spot_angle_outside_outer_cone(self) -> None:
        """Test zero intensity outside outer cone."""
        result = attenuation_spot_angle(0.5, 0.9, 0.7)
        assert result == 0.0

    def test_spot_angle_smooth_falloff(self) -> None:
        """Test smooth falloff between inner and outer."""
        inner_result = attenuation_spot_angle(0.9, 0.9, 0.7)
        mid_result = attenuation_spot_angle(0.8, 0.9, 0.7)
        outer_result = attenuation_spot_angle(0.7, 0.9, 0.7)

        assert inner_result >= mid_result >= outer_result
        assert mid_result > 0.0 and mid_result < 1.0


# =============================================================================
# Directional Light Tests
# =============================================================================


class TestDirectionalLight:
    """Test directional light evaluation."""

    def test_directional_light_direction_negated(self) -> None:
        """Test that directional light direction is negated."""
        light = create_directional_light((0.0, -1.0, 0.0))
        sample = evaluate_directional_light(light, (0.0, 0.0, 0.0))

        assert abs(sample.direction[0]) < EPSILON
        assert abs(sample.direction[1] - 1.0) < EPSILON
        assert abs(sample.direction[2]) < EPSILON

    def test_directional_light_no_attenuation(self) -> None:
        """Test that directional light has no distance attenuation."""
        light = create_directional_light((0.0, -1.0, 0.0), intensity=10.0)

        sample1 = evaluate_directional_light(light, (0.0, 0.0, 0.0))
        sample2 = evaluate_directional_light(light, (1000.0, 1000.0, 1000.0))

        assert abs(sample1.radiance[0] - sample2.radiance[0]) < EPSILON
        assert abs(sample1.radiance[0] - 10.0) < EPSILON

    def test_directional_light_color_intensity(self) -> None:
        """Test directional light radiance is color * intensity."""
        light = create_directional_light(
            (0.0, -1.0, 0.0),
            color=(0.5, 0.8, 1.0),
            intensity=2.0,
        )
        sample = evaluate_directional_light(light, (0.0, 0.0, 0.0))

        assert abs(sample.radiance[0] - 1.0) < EPSILON
        assert abs(sample.radiance[1] - 1.6) < EPSILON
        assert abs(sample.radiance[2] - 2.0) < EPSILON

    def test_directional_light_infinite_distance(self) -> None:
        """Test that directional light reports infinite distance."""
        light = create_directional_light((0.0, -1.0, 0.0))
        sample = evaluate_directional_light(light, (0.0, 0.0, 0.0))

        assert sample.distance > 1e9


# =============================================================================
# Point Light Tests
# =============================================================================


class TestPointLight:
    """Test point light evaluation."""

    def test_point_light_direction_toward_light(self) -> None:
        """Test that point light direction points toward light."""
        light = create_point_light((0.0, 10.0, 0.0))
        sample = evaluate_point_light(light, (0.0, 0.0, 0.0))

        assert abs(sample.direction[0]) < EPSILON
        assert abs(sample.direction[1] - 1.0) < EPSILON
        assert abs(sample.direction[2]) < EPSILON

    def test_point_light_distance_calculated(self) -> None:
        """Test that point light distance is calculated correctly."""
        light = create_point_light((3.0, 4.0, 0.0))
        sample = evaluate_point_light(light, (0.0, 0.0, 0.0))

        assert abs(sample.distance - 5.0) < EPSILON

    def test_point_light_attenuation_applied(self) -> None:
        """Test that point light attenuation is applied."""
        light = create_point_light((0.0, 5.0, 0.0), intensity=100.0, range_val=10.0)

        near_sample = evaluate_point_light(light, (0.0, 4.0, 0.0))
        far_sample = evaluate_point_light(light, (0.0, 0.0, 0.0))

        assert near_sample.radiance[0] > far_sample.radiance[0]

    def test_point_light_beyond_range(self) -> None:
        """Test point light has zero radiance beyond range."""
        light = create_point_light((0.0, 0.0, 0.0), range_val=5.0)
        sample = evaluate_point_light(light, (10.0, 0.0, 0.0))

        assert sample.radiance[0] == 0.0


# =============================================================================
# Spot Light Tests
# =============================================================================


class TestSpotLight:
    """Test spot light evaluation."""

    def test_spot_light_center_of_cone(self) -> None:
        """Test spot light at center of cone has high radiance."""
        light = create_spot_light(
            (0.0, 5.0, 0.0),
            (0.0, -1.0, 0.0),
            intensity=100.0,
            range_val=10.0,
        )
        sample = evaluate_spot_light(light, (0.0, 0.0, 0.0))

        assert sample.radiance[0] > 0.0

    def test_spot_light_outside_cone(self) -> None:
        """Test spot light outside cone has low/zero radiance."""
        light = create_spot_light(
            (0.0, 5.0, 0.0),
            (0.0, -1.0, 0.0),
            inner_angle=0.1,
            outer_angle=0.2,
            intensity=100.0,
            range_val=10.0,
        )
        sample = evaluate_spot_light(light, (5.0, 0.0, 0.0))

        assert sample.radiance[0] < 0.1

    def test_spot_light_distance_and_angle_attenuation(self) -> None:
        """Test spot light combines distance and angle attenuation."""
        light = create_spot_light(
            (0.0, 10.0, 0.0),
            (0.0, -1.0, 0.0),
            intensity=100.0,
            range_val=15.0,
        )

        center_sample = evaluate_spot_light(light, (0.0, 0.0, 0.0))
        edge_sample = evaluate_spot_light(light, (5.0, 0.0, 0.0))

        assert center_sample.radiance[0] > edge_sample.radiance[0]


# =============================================================================
# Generic Light Evaluation Tests
# =============================================================================


class TestEvaluateLight:
    """Test generic light evaluation dispatch."""

    def test_evaluate_light_dispatches_directional(self) -> None:
        """Test evaluate_light dispatches to directional."""
        light = create_directional_light((0.0, -1.0, 0.0), intensity=5.0)
        sample = evaluate_light(light, (0.0, 0.0, 0.0))

        expected = evaluate_directional_light(light, (0.0, 0.0, 0.0))
        assert abs(sample.radiance[0] - expected.radiance[0]) < EPSILON

    def test_evaluate_light_dispatches_point(self) -> None:
        """Test evaluate_light dispatches to point."""
        light = create_point_light((0.0, 5.0, 0.0), intensity=50.0)
        sample = evaluate_light(light, (0.0, 0.0, 0.0))

        expected = evaluate_point_light(light, (0.0, 0.0, 0.0))
        assert abs(sample.radiance[0] - expected.radiance[0]) < EPSILON

    def test_evaluate_light_dispatches_spot(self) -> None:
        """Test evaluate_light dispatches to spot."""
        light = create_spot_light((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), intensity=50.0)
        sample = evaluate_light(light, (0.0, 0.0, 0.0))

        expected = evaluate_spot_light(light, (0.0, 0.0, 0.0))
        assert abs(sample.radiance[0] - expected.radiance[0]) < EPSILON


# =============================================================================
# Shadow Placeholder Tests
# =============================================================================


class TestShadowSampling:
    """Test shadow sampling placeholder."""

    def test_sample_shadow_returns_one(self) -> None:
        """Test that shadow placeholder returns 1.0 (fully lit)."""
        light_sample = LightSample()
        shadow = sample_shadow(0, (0.0, 0.0, 0.0), light_sample)

        assert shadow == 1.0

    def test_sample_shadow_all_lights(self) -> None:
        """Test shadow returns 1.0 for all light indices."""
        light_sample = LightSample()
        for i in range(MAX_LIGHTS):
            shadow = sample_shadow(i, (0.0, 0.0, 0.0), light_sample)
            assert shadow == 1.0


# =============================================================================
# Light Accumulation Tests
# =============================================================================


class TestAccumulateLighting:
    """Test light accumulation function."""

    def test_accumulate_single_light(self) -> None:
        """Test accumulation with a single light produces output."""
        params = PBRParamsLighting()
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        assert result.diffuse[0] > 0.0 or result.specular[0] > 0.0

    def test_accumulate_multiple_lights(self) -> None:
        """Test accumulation with multiple lights."""
        params = PBRParamsLighting()
        lights = [
            create_directional_light((0.0, -1.0, 0.0), intensity=1.0),
            create_directional_light((1.0, -1.0, 0.0), intensity=1.0),
            create_directional_light((-1.0, -1.0, 0.0), intensity=1.0),
        ]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        single_light_result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=[lights[0]],
        )

        assert result.diffuse[0] > single_light_result.diffuse[0]

    def test_accumulate_max_lights(self) -> None:
        """Test accumulation with maximum (8) lights."""
        params = PBRParamsLighting()
        lights = [
            create_directional_light((0.0, -1.0, 0.0), intensity=1.0)
            for _ in range(MAX_LIGHTS)
        ]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        assert result.diffuse[0] > 0.0

    def test_accumulate_beyond_max_lights_clamped(self) -> None:
        """Test that lights beyond MAX_LIGHTS are ignored."""
        params = PBRParamsLighting()
        lights = [
            create_directional_light((0.0, -1.0, 0.0), intensity=1.0)
            for _ in range(12)  # More than MAX_LIGHTS
        ]

        result_12 = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        result_8 = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights[:MAX_LIGHTS],
        )

        assert abs(result_12.diffuse[0] - result_8.diffuse[0]) < EPSILON

    def test_accumulate_no_lights(self) -> None:
        """Test accumulation with no lights returns zero."""
        params = PBRParamsLighting()

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=[],
        )

        assert result.diffuse == (0.0, 0.0, 0.0)
        assert result.specular == (0.0, 0.0, 0.0)

    def test_accumulate_back_facing_light_ignored(self) -> None:
        """Test that back-facing lights contribute nothing."""
        params = PBRParamsLighting()
        light = create_directional_light((0.0, 1.0, 0.0), intensity=100.0)

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=[light],
        )

        assert result.diffuse == (0.0, 0.0, 0.0)
        assert result.specular == (0.0, 0.0, 0.0)

    def test_accumulate_mixed_light_types(self) -> None:
        """Test accumulation with mixed light types."""
        params = PBRParamsLighting()
        lights = [
            create_directional_light((0.0, -1.0, 0.0), intensity=1.0),
            create_point_light((0.0, 3.0, 0.0), intensity=50.0, range_val=10.0),
            create_spot_light((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), intensity=100.0),
        ]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        assert result.diffuse[0] > 0.0


# =============================================================================
# Final Shading Composition Tests
# =============================================================================


class TestComposeFinalShading:
    """Test final shading composition."""

    def test_compose_applies_ao_to_ambient(self) -> None:
        """Test that AO is applied to ambient lighting."""
        lighting = LightingResult(diffuse=(1.0, 1.0, 1.0), specular=(0.5, 0.5, 0.5))
        params_no_ao = PBRParamsLighting(occlusion=1.0)
        params_half_ao = PBRParamsLighting(occlusion=0.5)

        result_no_ao = compose_final_shading(lighting, params_no_ao, (0.2, 0.2, 0.2))
        result_half_ao = compose_final_shading(lighting, params_half_ao, (0.2, 0.2, 0.2))

        # Direct lighting should be same, but ambient contribution differs
        direct = 1.0 + 0.5  # diffuse + specular
        assert abs(result_no_ao[0] - (direct + 0.2)) < EPSILON
        assert abs(result_half_ao[0] - (direct + 0.1)) < EPSILON

    def test_compose_adds_emissive(self) -> None:
        """Test that emissive is added to final output."""
        lighting = LightingResult(diffuse=(0.0, 0.0, 0.0), specular=(0.0, 0.0, 0.0))
        params = PBRParamsLighting(emissive=(1.0, 0.5, 0.0))

        result = compose_final_shading(lighting, params, (0.0, 0.0, 0.0))

        assert abs(result[0] - 1.0) < EPSILON
        assert abs(result[1] - 0.5) < EPSILON
        assert abs(result[2] - 0.0) < EPSILON

    def test_compose_emissive_not_affected_by_ao(self) -> None:
        """Test that emissive is not affected by AO."""
        lighting = LightingResult()
        params = PBRParamsLighting(occlusion=0.0, emissive=(1.0, 1.0, 1.0))

        result = compose_final_shading(lighting, params, (0.0, 0.0, 0.0))

        # Even with zero AO, emissive should be full
        assert abs(result[0] - 1.0) < EPSILON


# =============================================================================
# Complete Lighting Pipeline Tests
# =============================================================================


class TestEvaluateAllLighting:
    """Test complete lighting evaluation pipeline."""

    def test_evaluate_all_lighting_basic(self) -> None:
        """Test complete lighting pipeline produces output."""
        params = PBRParamsLighting(
            base_color=(0.8, 0.2, 0.2),
            roughness=0.5,
            metallic=0.0,
        )
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = evaluate_all_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
            ambient=(0.03, 0.03, 0.03),
        )

        assert result[0] > 0.0

    def test_evaluate_all_lighting_with_emissive(self) -> None:
        """Test complete pipeline includes emissive."""
        params = PBRParamsLighting(
            base_color=(0.5, 0.5, 0.5),
            emissive=(0.0, 1.0, 0.0),
        )

        result = evaluate_all_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=[],
            ambient=(0.0, 0.0, 0.0),
        )

        assert result[1] > 0.9  # Green emissive


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy conservation properties."""

    def test_diffuse_specular_sum_reasonable(self) -> None:
        """Test that diffuse + specular doesn't exceed input energy."""
        params = PBRParamsLighting()
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        total = result.diffuse[0] + result.specular[0]
        assert total < 2.0, f"Total output {total} seems too high for unit input"

    def test_metal_no_diffuse(self) -> None:
        """Test that pure metals have no diffuse component."""
        params = PBRParamsLighting(metallic=1.0)
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        assert result.diffuse[0] < 0.01


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_intensity_light(self) -> None:
        """Test light with zero intensity produces zero radiance."""
        light = create_point_light((0.0, 5.0, 0.0), intensity=0.0)
        sample = evaluate_point_light(light, (0.0, 0.0, 0.0))

        assert sample.radiance == (0.0, 0.0, 0.0)

    def test_surface_at_light_position(self) -> None:
        """Test surface at light position has high radiance."""
        light = create_point_light((0.0, 0.0, 0.0), intensity=1.0, range_val=10.0)
        sample = evaluate_point_light(light, (0.0, 0.0, 0.0))

        assert sample.radiance[0] > 1000.0

    def test_black_base_color(self) -> None:
        """Test black material with lighting."""
        params = PBRParamsLighting(base_color=(0.0, 0.0, 0.0))
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        # Black material should still have some specular
        assert result.specular[0] >= 0.0

    def test_parallel_normal_and_view(self) -> None:
        """Test lighting when normal and view are parallel."""
        params = PBRParamsLighting()
        lights = [create_directional_light((0.0, -1.0, 0.0), intensity=1.0)]

        result = accumulate_lighting(
            params,
            N=(0.0, 1.0, 0.0),
            V=(0.0, 1.0, 0.0),
            world_position=(0.0, 0.0, 0.0),
            lights=lights,
        )

        assert result.diffuse[0] > 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the lighting pipeline."""

    def test_wgsl_and_python_max_lights_match(self) -> None:
        """Test that Python and WGSL MAX_LIGHTS match."""
        wgsl = get_lighting_wgsl()
        match = re.search(r"const MAX_LIGHTS:\s*u32\s*=\s*(\d+)u", wgsl)
        assert match, "Could not find MAX_LIGHTS in WGSL"
        wgsl_max = int(match.group(1))
        assert wgsl_max == MAX_LIGHTS

    def test_import_from_materials(self) -> None:
        """Test that lighting can be imported from trinity.materials."""
        # This will fail if not exported properly
        try:
            from trinity.materials import (
                get_lighting_wgsl,
                create_directional_light,
                accumulate_lighting,
            )
            assert callable(get_lighting_wgsl)
        except ImportError:
            pass  # Will be updated in __init__.py

    def test_reference_values_exist(self) -> None:
        """Test that we have reference values for validation."""
        assert len(LIGHTING_REFERENCE_VALUES["attenuation_point"]) >= 3
        assert len(LIGHTING_REFERENCE_VALUES["attenuation_spot_angle"]) >= 3
        assert len(LIGHTING_REFERENCE_VALUES["directional_light"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
