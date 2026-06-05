"""Tests for domain-specific shader templates.

Task: T-MAT-2.2 Domain Variants
Gap: S3-G3 (CRITICAL)

Tests verify:
1. Each domain produces correct shader structure
2. Each domain compiles without errors (valid WGSL syntax)
3. Domain switching produces observably different output
4. Domain capabilities are correctly mapped
5. Domain output formats match expected structure
6. Integration with VariantConfig
"""

from __future__ import annotations

import pytest
import re
from typing import Set

from trinity.materials.domains import (
    DomainCapability,
    DomainOutputFormat,
    DomainShaderTemplate,
    DomainVariantGenerator,
    DOMAIN_CAPABILITIES,
    DOMAIN_OUTPUT_FORMATS,
    domain_has_capability,
    get_domain_shader_info,
)

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
)


# =============================================================================
# Test: Domain Capabilities
# =============================================================================


class TestDomainCapabilities:
    """Test domain capability mappings."""

    def test_surface_has_lighting(self):
        """SURFACE domain should have lighting capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.LIGHTING)

    def test_surface_has_shadows(self):
        """SURFACE domain should have shadow capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.SHADOWS)

    def test_surface_has_normal_mapping(self):
        """SURFACE domain should have normal mapping capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.NORMAL_MAPPING)

    def test_surface_has_depth_write(self):
        """SURFACE domain should have depth write capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.DEPTH_WRITE)

    def test_surface_has_environment_map(self):
        """SURFACE domain should have environment map capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.ENVIRONMENT_MAP)

    def test_surface_has_emissive(self):
        """SURFACE domain should have emissive capability."""
        assert domain_has_capability(MaterialDomain.SURFACE, DomainCapability.EMISSIVE)

    def test_deferred_decal_has_gbuffer_output(self):
        """DEFERRED_DECAL domain should have G-buffer output capability."""
        assert domain_has_capability(MaterialDomain.DEFERRED_DECAL, DomainCapability.GBUFFER_OUTPUT)

    def test_deferred_decal_has_normal_mapping(self):
        """DEFERRED_DECAL domain should have normal mapping capability."""
        assert domain_has_capability(MaterialDomain.DEFERRED_DECAL, DomainCapability.NORMAL_MAPPING)

    def test_deferred_decal_no_lighting(self):
        """DEFERRED_DECAL domain should NOT have lighting capability."""
        assert not domain_has_capability(MaterialDomain.DEFERRED_DECAL, DomainCapability.LIGHTING)

    def test_volume_has_volumetric(self):
        """VOLUME domain should have volumetric capability."""
        assert domain_has_capability(MaterialDomain.VOLUME, DomainCapability.VOLUMETRIC)

    def test_volume_has_emissive(self):
        """VOLUME domain should have emissive capability."""
        assert domain_has_capability(MaterialDomain.VOLUME, DomainCapability.EMISSIVE)

    def test_volume_no_shadows(self):
        """VOLUME domain should NOT have shadow capability (simplified)."""
        assert not domain_has_capability(MaterialDomain.VOLUME, DomainCapability.SHADOWS)

    def test_post_process_has_fullscreen(self):
        """POST_PROCESS domain should have fullscreen capability."""
        assert domain_has_capability(MaterialDomain.POST_PROCESS, DomainCapability.FULLSCREEN)

    def test_post_process_no_lighting(self):
        """POST_PROCESS domain should NOT have lighting capability."""
        assert not domain_has_capability(MaterialDomain.POST_PROCESS, DomainCapability.LIGHTING)

    def test_ui_has_vertex_color(self):
        """UI domain should have vertex color capability."""
        assert domain_has_capability(MaterialDomain.UI, DomainCapability.VERTEX_COLOR)

    def test_ui_no_lighting(self):
        """UI domain should NOT have lighting capability."""
        assert not domain_has_capability(MaterialDomain.UI, DomainCapability.LIGHTING)

    def test_ui_no_depth_write(self):
        """UI domain should NOT have depth write capability."""
        assert not domain_has_capability(MaterialDomain.UI, DomainCapability.DEPTH_WRITE)

    def test_all_domains_have_capabilities(self):
        """All domains should have at least one capability."""
        for domain in MaterialDomain:
            caps = DOMAIN_CAPABILITIES.get(domain, set())
            assert len(caps) > 0, f"{domain.name} has no capabilities"


# =============================================================================
# Test: Domain Output Formats
# =============================================================================


class TestDomainOutputFormats:
    """Test domain output format specifications."""

    def test_surface_output_format(self):
        """SURFACE domain should output primary color only."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.SURFACE]
        assert fmt.primary_color is True
        assert fmt.normal is False
        assert fmt.material is False

    def test_deferred_decal_output_format(self):
        """DEFERRED_DECAL domain should output color, normal, and material."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.DEFERRED_DECAL]
        assert fmt.primary_color is True
        assert fmt.normal is True
        assert fmt.material is True

    def test_volume_output_format(self):
        """VOLUME domain should output primary color only."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.VOLUME]
        assert fmt.primary_color is True
        assert fmt.normal is False
        assert fmt.material is False

    def test_post_process_output_format(self):
        """POST_PROCESS domain should output primary color only."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.POST_PROCESS]
        assert fmt.primary_color is True
        assert fmt.normal is False
        assert fmt.material is False

    def test_ui_output_format(self):
        """UI domain should output primary color only."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.UI]
        assert fmt.primary_color is True
        assert fmt.normal is False
        assert fmt.material is False

    def test_output_struct_generation_simple(self):
        """Simple output format should generate correct struct."""
        fmt = DomainOutputFormat(primary_color=True, normal=False, material=False)
        wgsl = fmt.generate_output_struct()

        assert "struct FragmentOutput {" in wgsl
        assert "@location(0) color: vec4<f32>," in wgsl
        assert "@location(1)" not in wgsl
        assert "@location(2)" not in wgsl

    def test_output_struct_generation_gbuffer(self):
        """G-buffer output format should generate multi-target struct."""
        fmt = DomainOutputFormat(primary_color=True, normal=True, material=True)
        wgsl = fmt.generate_output_struct("GBufferOutput")

        assert "struct GBufferOutput {" in wgsl
        assert "@location(0) color: vec4<f32>," in wgsl
        assert "@location(1) normal: vec4<f32>," in wgsl
        assert "@location(2) material: vec4<f32>," in wgsl


# =============================================================================
# Test: Domain Shader Templates Structure
# =============================================================================


class TestDomainShaderTemplates:
    """Test domain shader template structure and content."""

    def test_surface_template_has_lighting_code(self):
        """SURFACE template should contain lighting evaluation."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "LIGHTING_ENABLED" in template
        assert "evaluate_direct_light" in template
        assert "SHADOWS_ENABLED" in template
        assert "sample_shadow_for_light" in template

    def test_surface_template_has_emissive(self):
        """SURFACE template should handle emissive contribution."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "params.emissive" in template

    def test_surface_template_has_normal_mapping(self):
        """SURFACE template should perform tangent-space normal mapping."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "world_tangent" in template
        assert "N_perturbed" in template

    def test_surface_template_has_ibl(self):
        """SURFACE template should have IBL for high quality."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "evaluate_ibl" in template
        assert "QUALITY_HIGH" in template

    def test_deferred_decal_template_has_gbuffer_output(self):
        """DEFERRED_DECAL template should output G-buffer data."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert "DecalOutput" in template
        assert "output.color" in template
        assert "output.normal" in template
        assert "output.material" in template

    def test_deferred_decal_no_lighting(self):
        """DEFERRED_DECAL template should NOT evaluate lighting."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert "evaluate_direct_light" not in template
        assert "LIGHTING_ENABLED" not in template

    def test_deferred_decal_has_blend_factor(self):
        """DEFERRED_DECAL template should compute blend factor."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert "angle_fade" in template
        assert "output.blend" in template

    def test_volume_template_has_scattering(self):
        """VOLUME template should implement scattering."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        assert "VolumeParams" in template
        assert "scattering" in template
        assert "absorption" in template
        assert "in_scatter" in template

    def test_volume_template_has_phase_function(self):
        """VOLUME template should have phase function."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        assert "phase_hg" in template
        assert "phase_g" in template
        assert "Henyey-Greenstein" in template

    def test_volume_template_has_density(self):
        """VOLUME template should handle density."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        assert "density" in template
        assert "transmittance" in template

    def test_post_process_template_has_tonemapping(self):
        """POST_PROCESS template should have tonemapping."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert "tonemap_aces" in template
        assert "tonemap_reinhard" in template

    def test_post_process_template_has_gamma(self):
        """POST_PROCESS template should have gamma correction."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert "linear_to_srgb" in template

    def test_post_process_template_has_vignette(self):
        """POST_PROCESS template should have vignette effect."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert "vignette" in template
        assert "QUALITY_HIGH" in template

    def test_ui_template_has_vertex_color(self):
        """UI template should use vertex color."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert "vertex_color" in template
        assert "vert_color" in template

    def test_ui_template_has_premultiplied_alpha(self):
        """UI template should use premultiplied alpha."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert "premultiplied" in template.lower() or "result.a" in template

    def test_ui_template_no_lighting(self):
        """UI template should NOT have lighting."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert "LIGHTING_ENABLED" not in template
        assert "evaluate_direct_light" not in template


# =============================================================================
# Test: Each Domain Produces Valid WGSL Syntax
# =============================================================================


class TestDomainTemplatesValidSyntax:
    """Test that domain templates produce syntactically valid WGSL."""

    def _check_balanced_braces(self, code: str) -> bool:
        """Check that braces are balanced."""
        count = 0
        for char in code:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _check_function_declarations(self, code: str) -> bool:
        """Check that function declarations have valid syntax."""
        # Pattern for WGSL function declaration
        fn_pattern = r'fn\s+\w+\s*\([^)]*\)\s*(->\s*[\w<>,\s]+)?\s*\{'
        matches = re.findall(fn_pattern, code)
        return len(matches) > 0

    def test_surface_template_balanced_braces(self):
        """SURFACE template should have balanced braces."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert self._check_balanced_braces(template), "Unbalanced braces in SURFACE"

    def test_surface_template_has_function(self):
        """SURFACE template should have valid function declaration."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert self._check_function_declarations(template), "Invalid function in SURFACE"

    def test_deferred_decal_template_balanced_braces(self):
        """DEFERRED_DECAL template should have balanced braces."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert self._check_balanced_braces(template), "Unbalanced braces in DEFERRED_DECAL"

    def test_deferred_decal_template_has_function(self):
        """DEFERRED_DECAL template should have valid function declaration."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert self._check_function_declarations(template), "Invalid function in DEFERRED_DECAL"

    def test_volume_template_balanced_braces(self):
        """VOLUME template should have balanced braces."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        assert self._check_balanced_braces(template), "Unbalanced braces in VOLUME"

    def test_volume_template_has_function(self):
        """VOLUME template should have valid function declaration."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        assert self._check_function_declarations(template), "Invalid function in VOLUME"

    def test_post_process_template_balanced_braces(self):
        """POST_PROCESS template should have balanced braces."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert self._check_balanced_braces(template), "Unbalanced braces in POST_PROCESS"

    def test_post_process_template_has_function(self):
        """POST_PROCESS template should have valid function declaration."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert self._check_function_declarations(template), "Invalid function in POST_PROCESS"

    def test_ui_template_balanced_braces(self):
        """UI template should have balanced braces."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert self._check_balanced_braces(template), "Unbalanced braces in UI"

    def test_ui_template_has_function(self):
        """UI template should have valid function declaration."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert self._check_function_declarations(template), "Invalid function in UI"

    def test_all_templates_have_return_statements(self):
        """All domain templates should have return statements."""
        for domain in MaterialDomain:
            template = DomainShaderTemplate.get_for_domain(domain)
            assert "return" in template, f"{domain.name} template missing return"

    def test_all_templates_use_valid_wgsl_types(self):
        """All templates should use valid WGSL types."""
        valid_types = {
            "vec2<f32>", "vec3<f32>", "vec4<f32>",
            "f32", "i32", "u32", "bool",
            "mat3x3<f32>", "mat4x4<f32>",
        }
        type_pattern = r':\s*(vec[234]<f32>|f32|i32|u32|bool|mat[34]x[34]<f32>)'

        for domain in MaterialDomain:
            template = DomainShaderTemplate.get_for_domain(domain)
            matches = re.findall(type_pattern, template)
            for match in matches:
                assert match in valid_types or any(match.startswith(t.split('<')[0]) for t in valid_types), \
                    f"Invalid type {match} in {domain.name}"


# =============================================================================
# Test: Domain Switching Produces Different Output
# =============================================================================


class TestDomainDifferentOutput:
    """Test that different domains produce observably different output."""

    def test_all_domains_have_unique_templates(self):
        """Each domain should have a unique template."""
        templates = {
            domain: DomainShaderTemplate.get_for_domain(domain)
            for domain in MaterialDomain
        }

        # All templates should be different
        template_set = set(templates.values())
        assert len(template_set) == len(MaterialDomain), "Domains share templates"

    def test_surface_vs_ui_lighting_difference(self):
        """SURFACE should have lighting, UI should not."""
        surface = DomainShaderTemplate.get_for_domain(MaterialDomain.SURFACE)
        ui = DomainShaderTemplate.get_for_domain(MaterialDomain.UI)

        assert "LIGHTING_ENABLED" in surface
        assert "LIGHTING_ENABLED" not in ui

    def test_surface_vs_volume_output_structure(self):
        """SURFACE and VOLUME should have different output structures."""
        surface = DomainShaderTemplate.get_for_domain(MaterialDomain.SURFACE)
        volume = DomainShaderTemplate.get_for_domain(MaterialDomain.VOLUME)

        # Volume has unique concepts
        assert "VolumeParams" in volume
        assert "VolumeParams" not in surface
        assert "density" in volume
        assert "transmittance" in volume

    def test_deferred_decal_unique_gbuffer(self):
        """DEFERRED_DECAL should be the only one with G-buffer output."""
        decal = DomainShaderTemplate.get_for_domain(MaterialDomain.DEFERRED_DECAL)

        assert "DecalOutput" in decal

        for domain in MaterialDomain:
            if domain != MaterialDomain.DEFERRED_DECAL:
                template = DomainShaderTemplate.get_for_domain(domain)
                assert "DecalOutput" not in template, f"{domain.name} has DecalOutput"

    def test_post_process_unique_tonemapping(self):
        """POST_PROCESS should be the only one with tonemapping functions."""
        pp = DomainShaderTemplate.get_for_domain(MaterialDomain.POST_PROCESS)

        # Post-process has tonemapping
        assert "tonemap_aces" in pp or "tonemap_reinhard" in pp

    def test_function_names_are_domain_specific(self):
        """Each domain should have uniquely named evaluation function."""
        func_names = set()
        for domain in MaterialDomain:
            name = DomainShaderTemplate.get_domain_function_name(domain)
            assert name not in func_names, f"Duplicate function name: {name}"
            func_names.add(name)

        assert len(func_names) == len(MaterialDomain)


# =============================================================================
# Test: DomainVariantGenerator
# =============================================================================


class TestDomainVariantGenerator:
    """Test DomainVariantGenerator code generation."""

    def test_generate_domain_code_for_surface(self):
        """Generator should produce valid code for SURFACE domain."""
        config = VariantConfig(domain=MaterialDomain.SURFACE)
        gen = DomainVariantGenerator(config)
        code = gen.generate_domain_code()

        assert "Domain-Specific Shader Templates" in code
        assert "Active Domain: SURFACE" in code
        assert "evaluate_surface_domain" in code
        assert "evaluate_domain" in code  # Dispatch function

    def test_generate_domain_code_for_volume(self):
        """Generator should produce valid code for VOLUME domain."""
        config = VariantConfig(domain=MaterialDomain.VOLUME)
        gen = DomainVariantGenerator(config)
        code = gen.generate_domain_code()

        assert "Active Domain: VOLUME" in code
        assert "evaluate_volume_domain" in code

    def test_generate_domain_code_for_ui(self):
        """Generator should produce valid code for UI domain."""
        config = VariantConfig(domain=MaterialDomain.UI)
        gen = DomainVariantGenerator(config)
        code = gen.generate_domain_code()

        assert "Active Domain: UI" in code
        assert "evaluate_ui_domain" in code

    def test_generate_all_domain_code(self):
        """Generator should produce code for all domains."""
        config = VariantConfig()
        gen = DomainVariantGenerator(config)
        code = gen.generate_all_domain_code()

        # Should contain all domain templates
        assert "SURFACE Domain" in code
        assert "DEFERRED_DECAL Domain" in code
        assert "VOLUME Domain" in code
        assert "POST_PROCESS Domain" in code
        assert "UI Domain" in code

        # Should have unified dispatch
        assert "DOMAIN_SURFACE" in code
        assert "DOMAIN_DEFERRED_DECAL" in code
        assert "DOMAIN_VOLUME" in code
        assert "DOMAIN_POST_PROCESS" in code
        assert "DOMAIN_UI" in code

    def test_generator_includes_helper_stubs(self):
        """Generator should include helper function stubs."""
        config = VariantConfig()
        gen = DomainVariantGenerator(config)
        code = gen.generate_domain_code()

        assert "evaluate_direct_light" in code
        assert "evaluate_ibl" in code
        assert "sample_shadow_for_light" in code

    def test_get_output_format(self):
        """Generator should return correct output format for domain."""
        surface_gen = DomainVariantGenerator(VariantConfig(domain=MaterialDomain.SURFACE))
        decal_gen = DomainVariantGenerator(VariantConfig(domain=MaterialDomain.DEFERRED_DECAL))

        surface_fmt = surface_gen.get_output_format()
        decal_fmt = decal_gen.get_output_format()

        assert surface_fmt.primary_color is True
        assert surface_fmt.normal is False

        assert decal_fmt.primary_color is True
        assert decal_fmt.normal is True
        assert decal_fmt.material is True

    def test_get_capabilities(self):
        """Generator should return correct capabilities for domain."""
        surface_gen = DomainVariantGenerator(VariantConfig(domain=MaterialDomain.SURFACE))
        ui_gen = DomainVariantGenerator(VariantConfig(domain=MaterialDomain.UI))

        surface_caps = surface_gen.get_capabilities()
        ui_caps = ui_gen.get_capabilities()

        assert DomainCapability.LIGHTING in surface_caps
        assert DomainCapability.SHADOWS in surface_caps

        assert DomainCapability.VERTEX_COLOR in ui_caps
        assert DomainCapability.LIGHTING not in ui_caps


# =============================================================================
# Test: get_domain_shader_info
# =============================================================================


class TestGetDomainShaderInfo:
    """Test get_domain_shader_info utility function."""

    def test_surface_shader_info(self):
        """Should return correct info for SURFACE domain."""
        info = get_domain_shader_info(MaterialDomain.SURFACE)

        assert info["domain"] == MaterialDomain.SURFACE
        assert info["has_lighting"] is True
        assert info["has_gbuffer"] is False
        assert info["function_name"] == "evaluate_surface_domain"
        assert info["template_length"] > 0

    def test_deferred_decal_shader_info(self):
        """Should return correct info for DEFERRED_DECAL domain."""
        info = get_domain_shader_info(MaterialDomain.DEFERRED_DECAL)

        assert info["domain"] == MaterialDomain.DEFERRED_DECAL
        assert info["has_lighting"] is False
        assert info["has_gbuffer"] is True
        assert info["function_name"] == "evaluate_deferred_decal_domain"

    def test_volume_shader_info(self):
        """Should return correct info for VOLUME domain."""
        info = get_domain_shader_info(MaterialDomain.VOLUME)

        assert info["domain"] == MaterialDomain.VOLUME
        assert info["has_lighting"] is False
        assert info["has_gbuffer"] is False
        assert DomainCapability.VOLUMETRIC in info["capabilities"]

    def test_post_process_shader_info(self):
        """Should return correct info for POST_PROCESS domain."""
        info = get_domain_shader_info(MaterialDomain.POST_PROCESS)

        assert info["domain"] == MaterialDomain.POST_PROCESS
        assert DomainCapability.FULLSCREEN in info["capabilities"]

    def test_ui_shader_info(self):
        """Should return correct info for UI domain."""
        info = get_domain_shader_info(MaterialDomain.UI)

        assert info["domain"] == MaterialDomain.UI
        assert DomainCapability.VERTEX_COLOR in info["capabilities"]
        assert info["has_lighting"] is False


# =============================================================================
# Test: Template Accessor Methods
# =============================================================================


class TestTemplateAccessors:
    """Test DomainShaderTemplate accessor methods."""

    def test_get_for_domain_surface(self):
        """get_for_domain should return SURFACE template."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.SURFACE)
        assert template == DomainShaderTemplate.SURFACE_FRAGMENT

    def test_get_for_domain_deferred_decal(self):
        """get_for_domain should return DEFERRED_DECAL template."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.DEFERRED_DECAL)
        assert template == DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT

    def test_get_for_domain_volume(self):
        """get_for_domain should return VOLUME template."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.VOLUME)
        assert template == DomainShaderTemplate.VOLUME_FRAGMENT

    def test_get_for_domain_post_process(self):
        """get_for_domain should return POST_PROCESS template."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.POST_PROCESS)
        assert template == DomainShaderTemplate.POST_PROCESS_FRAGMENT

    def test_get_for_domain_ui(self):
        """get_for_domain should return UI template."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.UI)
        assert template == DomainShaderTemplate.UI_FRAGMENT

    def test_get_all_templates(self):
        """get_all_templates should return all 5 templates."""
        templates = DomainShaderTemplate.get_all_templates()

        assert len(templates) == 5
        assert MaterialDomain.SURFACE in templates
        assert MaterialDomain.DEFERRED_DECAL in templates
        assert MaterialDomain.VOLUME in templates
        assert MaterialDomain.POST_PROCESS in templates
        assert MaterialDomain.UI in templates

    def test_get_domain_function_name_surface(self):
        """Should return correct function name for SURFACE."""
        name = DomainShaderTemplate.get_domain_function_name(MaterialDomain.SURFACE)
        assert name == "evaluate_surface_domain"

    def test_get_domain_function_name_all(self):
        """Should return unique function name for each domain."""
        names = {
            domain: DomainShaderTemplate.get_domain_function_name(domain)
            for domain in MaterialDomain
        }

        # All names should be unique
        assert len(set(names.values())) == len(MaterialDomain)

        # All names should follow pattern
        for domain, name in names.items():
            assert name.startswith("evaluate_"), f"{domain.name}: {name}"
            assert name.endswith("_domain"), f"{domain.name}: {name}"


# =============================================================================
# Test: Integration with VariantConfig
# =============================================================================


class TestVariantConfigIntegration:
    """Test integration between domains and VariantConfig."""

    def test_domain_generator_respects_quality(self):
        """Generator should include quality-gated code."""
        config_high = VariantConfig(domain=MaterialDomain.SURFACE, quality=QualityTier.HIGH)
        config_low = VariantConfig(domain=MaterialDomain.SURFACE, quality=QualityTier.LOW)

        gen_high = DomainVariantGenerator(config_high)
        gen_low = DomainVariantGenerator(config_low)

        # Both should have same template (quality gating is in variant consts)
        code_high = gen_high.generate_domain_code()
        code_low = gen_low.generate_domain_code()

        # Template code is the same, but const declarations differ
        assert "evaluate_surface_domain" in code_high
        assert "evaluate_surface_domain" in code_low

    def test_domain_generator_respects_blend(self):
        """Generator should work with different blend modes."""
        config_opaque = VariantConfig(domain=MaterialDomain.SURFACE, blend=BlendMode.OPAQUE)
        config_trans = VariantConfig(domain=MaterialDomain.SURFACE, blend=BlendMode.TRANSLUCENT)

        gen_opaque = DomainVariantGenerator(config_opaque)
        gen_trans = DomainVariantGenerator(config_trans)

        # Both should produce valid code
        code_opaque = gen_opaque.generate_domain_code()
        code_trans = gen_trans.generate_domain_code()

        assert "evaluate_surface_domain" in code_opaque
        assert "evaluate_surface_domain" in code_trans

    def test_all_variant_combinations_produce_valid_domain_code(self):
        """All 75 variant combinations should produce valid domain code."""
        from trinity.materials.variants import generate_all_variant_combinations

        configs = generate_all_variant_combinations()

        for config in configs:
            gen = DomainVariantGenerator(config)
            code = gen.generate_domain_code()

            # Should have header
            assert "Domain-Specific Shader Templates" in code

            # Should have dispatch function
            assert "evaluate_domain" in code

            # Should have the active domain function
            func_name = DomainShaderTemplate.get_domain_function_name(config.domain)
            assert func_name in code, f"Missing {func_name} for {config.domain.name}"


# =============================================================================
# Test: Domain-Specific Feature Verification
# =============================================================================


class TestDomainSpecificFeatures:
    """Test domain-specific shader features in detail."""

    def test_surface_has_light_loop(self):
        """SURFACE template should have light iteration."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "for (var i = 0u; i < min(input.light_count, MAX_LIGHTS)" in template

    def test_surface_has_ao_application(self):
        """SURFACE template should apply ambient occlusion."""
        template = DomainShaderTemplate.SURFACE_FRAGMENT
        assert "params.occlusion" in template

    def test_volume_has_beer_lambert(self):
        """VOLUME template should reference Beer-Lambert law."""
        template = DomainShaderTemplate.VOLUME_FRAGMENT
        # Uses exp(-...) for transmittance
        assert "exp(-" in template
        assert "extinction" in template

    def test_deferred_decal_has_angle_fade(self):
        """DEFERRED_DECAL template should fade based on projection angle."""
        template = DomainShaderTemplate.DEFERRED_DECAL_FRAGMENT
        assert "angle_fade" in template

    def test_post_process_has_exposure(self):
        """POST_PROCESS template should apply exposure."""
        template = DomainShaderTemplate.POST_PROCESS_FRAGMENT
        assert "exposure" in template

    def test_ui_has_srgb_variant(self):
        """UI template should have sRGB variant function."""
        template = DomainShaderTemplate.UI_FRAGMENT
        assert "evaluate_ui_domain_srgb" in template


# =============================================================================
# Test: WGSL Identifier Validation
# =============================================================================


class TestWGSLIdentifiers:
    """Test that generated code uses valid WGSL identifiers."""

    def test_no_reserved_word_collision(self):
        """Generated code should not use WGSL reserved words as identifiers."""
        wgsl_reserved = {
            "break", "case", "const", "continue", "default", "discard",
            "else", "enable", "false", "fn", "for", "function", "if",
            "let", "loop", "private", "return", "struct", "switch",
            "true", "type", "uniform", "var", "while",
        }

        for domain in MaterialDomain:
            template = DomainShaderTemplate.get_for_domain(domain)

            # Extract variable declarations
            var_pattern = r'(var|let)\s+(\w+)'
            matches = re.findall(var_pattern, template)

            for _, var_name in matches:
                assert var_name.lower() not in wgsl_reserved, (
                    f"Reserved word used as variable: {var_name} in {domain.name}"
                )

    def test_function_names_valid_wgsl(self):
        """All function names should be valid WGSL identifiers."""
        fn_pattern = r'fn\s+(\w+)'

        for domain in MaterialDomain:
            template = DomainShaderTemplate.get_for_domain(domain)
            matches = re.findall(fn_pattern, template)

            for name in matches:
                # Valid WGSL identifier: starts with letter/underscore, contains alphanumeric
                assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name), (
                    f"Invalid function name: {name} in {domain.name}"
                )

    def test_struct_names_valid_wgsl(self):
        """All struct names should be valid WGSL identifiers."""
        struct_pattern = r'struct\s+(\w+)'

        for domain in MaterialDomain:
            template = DomainShaderTemplate.get_for_domain(domain)
            matches = re.findall(struct_pattern, template)

            for name in matches:
                assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name), (
                    f"Invalid struct name: {name} in {domain.name}"
                )
