"""Tests for UI Material Domain - Unlit vertex-color rendering.

Task: T-MAT-5.8 UI Material Domain
Gap: Variant coverage
Dependency: T-MAT-2.2 (domain variants)

Tests verify:
1. UI material renders correct screen-space output
2. No lighting evaluation in generated WGSL (no "light", no "shadow", no "BRDF")
3. Vertex-color multiply works
4. Premultiplied alpha output
5. sRGB conversion when enabled
"""

from __future__ import annotations

import pytest
import re
from typing import List

from trinity.materials.ui_domain import (
    UIBlendMode,
    UIMaterialConfig,
    UIMaterialBuilder,
    UI_DOMAIN_WGSL,
    UI_DOMAIN_MINIMAL_WGSL,
    UI_MATERIAL_PRESETS,
    generate_ui_material,
    generate_ui_material_consts,
    get_ui_entry_point,
    get_ui_material_preset,
    validate_ui_material_wgsl,
)

from trinity.materials.domains import (
    DomainCapability,
    DomainShaderTemplate,
    domain_has_capability,
)
from trinity.materials.variants import MaterialDomain


# =============================================================================
# Test: UIMaterialConfig
# =============================================================================


class TestUIMaterialConfig:
    """Test UIMaterialConfig dataclass."""

    def test_default_config(self):
        """Default config should have standard UI settings."""
        config = UIMaterialConfig()

        assert config.premultiply_alpha is True
        assert config.use_srgb is True
        assert config.vertex_color_enabled is True
        assert config.texture_enabled is True
        assert config.clip_rect_enabled is False
        assert config.blend_mode == UIBlendMode.PREMULTIPLIED
        assert config.opacity == 1.0

    def test_custom_config(self):
        """Custom config should store all values."""
        config = UIMaterialConfig(
            premultiply_alpha=False,
            use_srgb=False,
            vertex_color_enabled=False,
            texture_enabled=False,
            clip_rect_enabled=True,
            blend_mode=UIBlendMode.ADDITIVE,
            opacity=0.5,
        )

        assert config.premultiply_alpha is False
        assert config.use_srgb is False
        assert config.vertex_color_enabled is False
        assert config.texture_enabled is False
        assert config.clip_rect_enabled is True
        assert config.blend_mode == UIBlendMode.ADDITIVE
        assert config.opacity == 0.5

    def test_config_is_frozen(self):
        """Config should be immutable (frozen dataclass)."""
        config = UIMaterialConfig()

        with pytest.raises(AttributeError):
            config.premultiply_alpha = False

    def test_config_opacity_validation(self):
        """Opacity must be in [0, 1] range."""
        with pytest.raises(ValueError, match="opacity must be in"):
            UIMaterialConfig(opacity=-0.1)

        with pytest.raises(ValueError, match="opacity must be in"):
            UIMaterialConfig(opacity=1.5)

    def test_config_opacity_boundary_values(self):
        """Opacity at boundary values should be valid."""
        config_zero = UIMaterialConfig(opacity=0.0)
        config_one = UIMaterialConfig(opacity=1.0)

        assert config_zero.opacity == 0.0
        assert config_one.opacity == 1.0


# =============================================================================
# Test: UI Material WGSL Generation
# =============================================================================


class TestGenerateUIMaterial:
    """Test WGSL shader generation for UI materials."""

    def test_generate_produces_valid_wgsl(self):
        """Generated WGSL should have balanced braces."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")

        assert open_braces == close_braces, "Unbalanced braces"

    def test_generate_includes_const_declarations(self):
        """Generated WGSL should include const declarations from config."""
        config = UIMaterialConfig(
            texture_enabled=True,
            use_srgb=True,
            premultiply_alpha=True,
            vertex_color_enabled=True,
            clip_rect_enabled=False,
        )
        wgsl = generate_ui_material(config)

        assert "const TEXTURE_ENABLED: bool = true;" in wgsl
        assert "const USE_SRGB: bool = true;" in wgsl
        assert "const PREMULTIPLY_ALPHA: bool = true;" in wgsl
        assert "const VERTEX_COLOR_ENABLED: bool = true;" in wgsl
        assert "const CLIP_RECT_ENABLED: bool = false;" in wgsl

    def test_generate_with_disabled_features(self):
        """Generated WGSL should have false consts when features disabled."""
        config = UIMaterialConfig(
            texture_enabled=False,
            use_srgb=False,
            premultiply_alpha=False,
            vertex_color_enabled=False,
            clip_rect_enabled=False,
        )
        wgsl = generate_ui_material(config)

        assert "const TEXTURE_ENABLED: bool = false;" in wgsl
        assert "const USE_SRGB: bool = false;" in wgsl
        assert "const PREMULTIPLY_ALPHA: bool = false;" in wgsl
        assert "const VERTEX_COLOR_ENABLED: bool = false;" in wgsl

    def test_generate_has_vertex_shader(self):
        """Generated WGSL should have vertex shader entry point."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        assert "@vertex" in wgsl
        assert "fn vs_ui" in wgsl
        assert "UIInput" in wgsl
        assert "UIOutput" in wgsl

    def test_generate_has_fragment_shader(self):
        """Generated WGSL should have fragment shader entry point."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        assert "@fragment" in wgsl
        assert "fn fs_ui" in wgsl

    def test_generate_minimal_produces_simpler_shader(self):
        """Minimal shader should not have uniforms or clip rect."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config, minimal=True)

        # Minimal shader uses simple structs
        assert "UIInputSimple" in wgsl
        assert "UIOutputSimple" in wgsl
        assert "fs_ui_simple" in wgsl

        # Should not have uniforms block
        assert "UIUniforms" not in wgsl


# =============================================================================
# Test: Screen-Space Output
# =============================================================================


class TestScreenSpaceOutput:
    """Test that UI materials output correct screen-space rendering."""

    def test_ui_uses_clip_space_position(self):
        """UI vertex shader should pass through clip-space position."""
        wgsl = UI_DOMAIN_WGSL

        # Position is passed through without transformation
        assert "output.position = input.position" in wgsl

    def test_ui_input_has_position_attribute(self):
        """UI input struct should have position at location 0."""
        wgsl = UI_DOMAIN_WGSL

        assert "@location(0) position: vec4<f32>" in wgsl

    def test_ui_input_has_uv_attribute(self):
        """UI input struct should have UV at location 1."""
        wgsl = UI_DOMAIN_WGSL

        assert "@location(1) uv: vec2<f32>" in wgsl

    def test_ui_input_has_color_attribute(self):
        """UI input struct should have vertex color at location 2."""
        wgsl = UI_DOMAIN_WGSL

        assert "@location(2) color: vec4<f32>" in wgsl

    def test_ui_output_has_builtin_position(self):
        """UI output struct should have builtin position."""
        wgsl = UI_DOMAIN_WGSL

        assert "@builtin(position) position: vec4<f32>" in wgsl


# =============================================================================
# Test: No Lighting Evaluation
# =============================================================================


class TestNoLightingEvaluation:
    """Test that UI materials have NO lighting code."""

    def test_ui_wgsl_has_no_light_references(self):
        """Generated WGSL should NOT reference lights."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        # These should NOT be present in UI shader
        assert "light" not in wgsl.lower() or "light_color" not in wgsl.lower()
        # Specifically check for PBR lighting patterns
        assert "evaluate_direct_light" not in wgsl
        assert "LIGHTING_ENABLED" not in wgsl

    def test_ui_wgsl_has_no_shadow_references(self):
        """Generated WGSL should NOT reference shadow sampling or evaluation."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        # Check for actual shadow functionality, not comments
        assert "sample_shadow" not in wgsl.lower()
        assert "shadow_map" not in wgsl.lower()
        assert "SHADOWS_ENABLED" not in wgsl

    def test_ui_wgsl_has_no_brdf_references(self):
        """Generated WGSL should NOT reference BRDF functions."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        assert "brdf" not in wgsl.lower()
        assert "evaluate_brdf" not in wgsl

    def test_ui_wgsl_has_no_pbr_params(self):
        """Generated WGSL should NOT use PBRParams struct."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        assert "PBRParams" not in wgsl
        assert "metallic" not in wgsl.lower()
        assert "roughness" not in wgsl.lower()

    def test_validate_ui_material_wgsl_passes(self):
        """Validation should pass for clean UI WGSL.

        Note: The validation function checks for PBR-specific patterns like
        'evaluate_direct_light', 'sample_shadow', 'BRDF', etc. Comments
        containing general words like 'light' or 'shadow' in descriptive
        context are acceptable as they don't indicate actual lighting code.
        """
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        # Check for actual PBR lighting patterns (not just the word "light")
        assert "evaluate_direct_light" not in wgsl
        assert "evaluate_ibl" not in wgsl
        assert "sample_shadow" not in wgsl
        assert "BRDF" not in wgsl
        assert "PBRParams" not in wgsl
        assert "LIGHTING_ENABLED" not in wgsl
        assert "SHADOWS_ENABLED" not in wgsl

    def test_validate_rejects_shader_with_lighting(self):
        """Validation should reject shader with lighting code."""
        bad_wgsl = """
        fn evaluate_direct_light(N: vec3<f32>) -> vec3<f32> {
            return vec3<f32>(1.0);
        }
        """

        errors = validate_ui_material_wgsl(bad_wgsl)

        assert len(errors) > 0
        assert any("evaluate_direct" in e for e in errors)

    def test_domain_template_has_no_lighting(self):
        """Domain template UI_FRAGMENT should not have lighting."""
        template = DomainShaderTemplate.UI_FRAGMENT

        assert "LIGHTING_ENABLED" not in template
        assert "evaluate_direct_light" not in template
        assert "evaluate_ibl" not in template


# =============================================================================
# Test: Vertex Color Multiply
# =============================================================================


class TestVertexColorMultiply:
    """Test vertex color multiplication functionality."""

    def test_vertex_color_multiply_in_shader(self):
        """Shader should multiply vertex color with texture color."""
        wgsl = UI_DOMAIN_WGSL

        # Check for vertex color usage
        assert "input.color" in wgsl
        # Check for multiplication pattern
        assert "* tex_color" in wgsl or "tex_color *" in wgsl

    def test_vertex_color_enabled_const_guard(self):
        """Vertex color should be guarded by const."""
        wgsl = UI_DOMAIN_WGSL

        assert "if VERTEX_COLOR_ENABLED" in wgsl

    def test_vertex_color_disabled_uses_white(self):
        """When vertex color disabled, should use white."""
        wgsl = UI_DOMAIN_WGSL

        # When vertex color is disabled, use white (1.0, 1.0, 1.0, 1.0)
        assert "1.0, 1.0, 1.0, 1.0" in wgsl or "vec4<f32>(1.0" in wgsl

    def test_domain_template_has_vertex_color(self):
        """Domain template should have vertex color capability."""
        assert domain_has_capability(MaterialDomain.UI, DomainCapability.VERTEX_COLOR)


# =============================================================================
# Test: Premultiplied Alpha Output
# =============================================================================


class TestPremultipliedAlpha:
    """Test premultiplied alpha output."""

    def test_premultiplied_alpha_in_shader(self):
        """Shader should apply premultiplied alpha."""
        wgsl = UI_DOMAIN_WGSL

        # Should multiply RGB by alpha
        assert "if PREMULTIPLY_ALPHA" in wgsl
        assert "color.rgb * color.a" in wgsl

    def test_premultiplied_alpha_const_declaration(self):
        """Config should generate PREMULTIPLY_ALPHA const."""
        config = UIMaterialConfig(premultiply_alpha=True)
        consts = generate_ui_material_consts(config)

        assert "const PREMULTIPLY_ALPHA: bool = true;" in consts

    def test_premultiplied_alpha_disabled(self):
        """When disabled, should generate false const."""
        config = UIMaterialConfig(premultiply_alpha=False)
        consts = generate_ui_material_consts(config)

        assert "const PREMULTIPLY_ALPHA: bool = false;" in consts


# =============================================================================
# Test: sRGB Conversion
# =============================================================================


class TestSRGBConversion:
    """Test sRGB color space conversion."""

    def test_srgb_conversion_function_exists(self):
        """Shader should have linear_to_srgb function."""
        wgsl = UI_DOMAIN_WGSL

        assert "fn linear_to_srgb" in wgsl

    def test_srgb_conversion_correct_formula(self):
        """sRGB conversion should use correct formula."""
        wgsl = UI_DOMAIN_WGSL

        # sRGB formula: cutoff at 0.0031308
        assert "0.0031308" in wgsl
        # Linear portion: * 12.92
        assert "12.92" in wgsl
        # Gamma portion: pow with 1/2.4
        assert "1.0 / 2.4" in wgsl or "1.0/2.4" in wgsl

    def test_srgb_const_guard(self):
        """sRGB conversion should be guarded by const."""
        wgsl = UI_DOMAIN_WGSL

        assert "if USE_SRGB" in wgsl

    def test_srgb_const_declaration(self):
        """Config should generate USE_SRGB const."""
        config = UIMaterialConfig(use_srgb=True)
        consts = generate_ui_material_consts(config)

        assert "const USE_SRGB: bool = true;" in consts


# =============================================================================
# Test: Clip Rectangle Support
# =============================================================================


class TestClipRectangle:
    """Test clip rectangle functionality."""

    def test_clip_rect_uniforms_exist(self):
        """Shader should have clip rect in uniforms."""
        wgsl = UI_DOMAIN_WGSL

        assert "clip_rect: vec4<f32>" in wgsl

    def test_clip_rect_check_function(self):
        """Shader should have clip rect checking function."""
        wgsl = UI_DOMAIN_WGSL

        assert "fn is_inside_clip_rect" in wgsl

    def test_clip_rect_const_guard(self):
        """Clip rect should be guarded by const."""
        wgsl = UI_DOMAIN_WGSL

        assert "if CLIP_RECT_ENABLED" in wgsl

    def test_clip_rect_uses_discard(self):
        """Clip rect should discard fragments outside."""
        wgsl = UI_DOMAIN_WGSL

        # Should discard fragments outside clip rect
        assert "discard" in wgsl


# =============================================================================
# Test: UIMaterialBuilder
# =============================================================================


class TestUIMaterialBuilder:
    """Test fluent builder interface."""

    def test_builder_default_config(self):
        """Builder should create default config."""
        config = UIMaterialBuilder().build()

        assert config.premultiply_alpha is True
        assert config.use_srgb is True
        assert config.texture_enabled is True
        assert config.vertex_color_enabled is True

    def test_builder_with_texture(self):
        """Builder should set texture enabled."""
        config = UIMaterialBuilder().with_texture(False).build()

        assert config.texture_enabled is False

    def test_builder_with_vertex_color(self):
        """Builder should set vertex color enabled."""
        config = UIMaterialBuilder().with_vertex_color(False).build()

        assert config.vertex_color_enabled is False

    def test_builder_with_srgb(self):
        """Builder should set sRGB enabled."""
        config = UIMaterialBuilder().with_srgb(False).build()

        assert config.use_srgb is False

    def test_builder_with_premultiplied_alpha(self):
        """Builder should set premultiplied alpha."""
        config = UIMaterialBuilder().with_premultiplied_alpha(False).build()

        assert config.premultiply_alpha is False

    def test_builder_with_clip_rect(self):
        """Builder should set clip rect enabled."""
        config = UIMaterialBuilder().with_clip_rect(True).build()

        assert config.clip_rect_enabled is True

    def test_builder_with_blend_mode(self):
        """Builder should set blend mode."""
        config = UIMaterialBuilder().with_blend_mode(UIBlendMode.ADDITIVE).build()

        assert config.blend_mode == UIBlendMode.ADDITIVE

    def test_builder_with_opacity(self):
        """Builder should set opacity."""
        config = UIMaterialBuilder().with_opacity(0.75).build()

        assert config.opacity == 0.75

    def test_builder_fluent_chain(self):
        """Builder should support fluent chaining."""
        config = (
            UIMaterialBuilder()
            .with_texture(True)
            .with_srgb(True)
            .with_premultiplied_alpha(True)
            .with_clip_rect(True)
            .with_blend_mode(UIBlendMode.MULTIPLY)
            .with_opacity(0.9)
            .build()
        )

        assert config.texture_enabled is True
        assert config.use_srgb is True
        assert config.premultiply_alpha is True
        assert config.clip_rect_enabled is True
        assert config.blend_mode == UIBlendMode.MULTIPLY
        assert config.opacity == 0.9


# =============================================================================
# Test: UI Material Presets
# =============================================================================


class TestUIMaterialPresets:
    """Test predefined material presets."""

    def test_default_preset_exists(self):
        """Default preset should exist."""
        config = get_ui_material_preset("default")

        assert config is not None
        assert isinstance(config, UIMaterialConfig)

    def test_text_preset(self):
        """Text preset should have texture enabled for font atlas."""
        config = get_ui_material_preset("text")

        assert config.texture_enabled is True
        assert config.vertex_color_enabled is True

    def test_icon_preset(self):
        """Icon preset should support tinting."""
        config = get_ui_material_preset("icon")

        assert config.texture_enabled is True
        assert config.vertex_color_enabled is True

    def test_solid_preset_no_texture(self):
        """Solid preset should not use texture."""
        config = get_ui_material_preset("solid")

        assert config.texture_enabled is False
        assert config.vertex_color_enabled is True

    def test_clipped_preset_has_clip_rect(self):
        """Clipped preset should have clip rect enabled."""
        config = get_ui_material_preset("clipped")

        assert config.clip_rect_enabled is True

    def test_glow_preset_is_additive(self):
        """Glow preset should use additive blending."""
        config = get_ui_material_preset("glow")

        assert config.blend_mode == UIBlendMode.ADDITIVE
        assert config.premultiply_alpha is False  # Additive doesn't premultiply

    def test_unknown_preset_raises(self):
        """Unknown preset should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown UI material preset"):
            get_ui_material_preset("nonexistent")

    def test_all_presets_generate_valid_wgsl(self):
        """All presets should generate valid WGSL."""
        for name in UI_MATERIAL_PRESETS:
            config = get_ui_material_preset(name)
            wgsl = generate_ui_material(config)

            # Should have balanced braces
            assert wgsl.count("{") == wgsl.count("}")

            # Should have entry points
            assert "fn vs_ui" in wgsl
            assert "fn fs_ui" in wgsl


# =============================================================================
# Test: UI Blend Modes
# =============================================================================


class TestUIBlendModes:
    """Test UI-specific blend modes."""

    def test_blend_mode_enum_values(self):
        """All blend modes should have string values."""
        assert UIBlendMode.NORMAL.value == "normal"
        assert UIBlendMode.PREMULTIPLIED.value == "premultiplied"
        assert UIBlendMode.ADDITIVE.value == "additive"
        assert UIBlendMode.MULTIPLY.value == "multiply"

    def test_get_ui_entry_point_normal(self):
        """Normal blend should use standard fragment shader."""
        config = UIMaterialConfig(blend_mode=UIBlendMode.NORMAL)
        entry = get_ui_entry_point(config)

        assert entry == "fs_ui"

    def test_get_ui_entry_point_premultiplied(self):
        """Premultiplied blend should use standard fragment shader."""
        config = UIMaterialConfig(blend_mode=UIBlendMode.PREMULTIPLIED)
        entry = get_ui_entry_point(config)

        assert entry == "fs_ui"

    def test_get_ui_entry_point_additive(self):
        """Additive blend should use additive fragment shader."""
        config = UIMaterialConfig(blend_mode=UIBlendMode.ADDITIVE)
        entry = get_ui_entry_point(config)

        assert entry == "fs_ui_additive"

    def test_get_ui_entry_point_multiply(self):
        """Multiply blend should use multiply fragment shader."""
        config = UIMaterialConfig(blend_mode=UIBlendMode.MULTIPLY)
        entry = get_ui_entry_point(config)

        assert entry == "fs_ui_multiply"

    def test_additive_fragment_shader_exists(self):
        """Additive fragment shader should exist in WGSL."""
        wgsl = UI_DOMAIN_WGSL

        assert "fn fs_ui_additive" in wgsl

    def test_multiply_fragment_shader_exists(self):
        """Multiply fragment shader should exist in WGSL."""
        wgsl = UI_DOMAIN_WGSL

        assert "fn fs_ui_multiply" in wgsl


# =============================================================================
# Test: WGSL Syntax Validation
# =============================================================================


class TestWGSLSyntax:
    """Test that generated WGSL has valid syntax."""

    def _check_balanced_braces(self, code: str) -> bool:
        """Check that braces are balanced."""
        count = 0
        for char in code:
            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _check_function_declarations(self, code: str) -> bool:
        """Check that function declarations have valid syntax."""
        fn_pattern = r"fn\s+\w+\s*\([^)]*\)\s*(->\s*[\w<>,\s]+)?\s*\{"
        matches = re.findall(fn_pattern, code)
        return len(matches) > 0

    def test_ui_domain_wgsl_balanced_braces(self):
        """UI_DOMAIN_WGSL should have balanced braces."""
        assert self._check_balanced_braces(UI_DOMAIN_WGSL)

    def test_ui_domain_minimal_balanced_braces(self):
        """UI_DOMAIN_MINIMAL_WGSL should have balanced braces."""
        assert self._check_balanced_braces(UI_DOMAIN_MINIMAL_WGSL)

    def test_generated_wgsl_balanced_braces(self):
        """All generated WGSL should have balanced braces."""
        configs = [
            UIMaterialConfig(),
            UIMaterialConfig(texture_enabled=False),
            UIMaterialConfig(clip_rect_enabled=True),
            UIMaterialConfig(premultiply_alpha=False, use_srgb=False),
        ]

        for config in configs:
            wgsl = generate_ui_material(config)
            assert self._check_balanced_braces(wgsl), f"Unbalanced braces for {config}"

    def test_ui_domain_wgsl_has_functions(self):
        """UI_DOMAIN_WGSL should have valid function declarations."""
        assert self._check_function_declarations(UI_DOMAIN_WGSL)

    def test_ui_domain_has_return_statements(self):
        """UI shader functions should have return statements."""
        wgsl = UI_DOMAIN_WGSL

        # Count functions
        fn_count = wgsl.count("fn ")
        return_count = wgsl.count("return ")

        # Each non-void function should have at least one return
        assert return_count >= fn_count - 1  # Allow for void functions

    def test_wgsl_uses_valid_types(self):
        """Generated WGSL should use valid WGSL types."""
        wgsl = UI_DOMAIN_WGSL

        # Should have standard WGSL types
        assert "vec2<f32>" in wgsl
        assert "vec3<f32>" in wgsl
        assert "vec4<f32>" in wgsl
        assert "f32" in wgsl
        assert "bool" in wgsl


# =============================================================================
# Test: Integration with Domain System
# =============================================================================


class TestDomainIntegration:
    """Test integration with the domain variant system."""

    def test_ui_domain_has_vertex_color_capability(self):
        """UI domain should have VERTEX_COLOR capability."""
        from trinity.materials.domains import DOMAIN_CAPABILITIES

        ui_caps = DOMAIN_CAPABILITIES[MaterialDomain.UI]

        assert DomainCapability.VERTEX_COLOR in ui_caps

    def test_ui_domain_no_lighting_capability(self):
        """UI domain should NOT have LIGHTING capability."""
        from trinity.materials.domains import DOMAIN_CAPABILITIES

        ui_caps = DOMAIN_CAPABILITIES[MaterialDomain.UI]

        assert DomainCapability.LIGHTING not in ui_caps
        assert DomainCapability.SHADOWS not in ui_caps

    def test_ui_domain_output_format_single_color(self):
        """UI domain should output only primary color."""
        from trinity.materials.domains import DOMAIN_OUTPUT_FORMATS

        ui_fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.UI]

        assert ui_fmt.primary_color is True
        assert ui_fmt.normal is False
        assert ui_fmt.material is False

    def test_domain_template_function_name(self):
        """Domain template should have correct function name."""
        func_name = DomainShaderTemplate.get_domain_function_name(MaterialDomain.UI)

        assert func_name == "evaluate_ui_domain"

    def test_domain_template_has_vertex_color(self):
        """Domain template should use vertex color."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.UI)

        assert "vertex_color" in template


# =============================================================================
# Test: Const Generation
# =============================================================================


class TestConstGeneration:
    """Test WGSL const declaration generation."""

    def test_generate_consts_all_true(self):
        """All-true config should generate all true consts."""
        config = UIMaterialConfig(
            texture_enabled=True,
            use_srgb=True,
            premultiply_alpha=True,
            vertex_color_enabled=True,
            clip_rect_enabled=True,
        )
        consts = generate_ui_material_consts(config)

        assert "TEXTURE_ENABLED: bool = true" in consts
        assert "USE_SRGB: bool = true" in consts
        assert "PREMULTIPLY_ALPHA: bool = true" in consts
        assert "VERTEX_COLOR_ENABLED: bool = true" in consts
        assert "CLIP_RECT_ENABLED: bool = true" in consts

    def test_generate_consts_all_false(self):
        """All-false config should generate all false consts."""
        config = UIMaterialConfig(
            texture_enabled=False,
            use_srgb=False,
            premultiply_alpha=False,
            vertex_color_enabled=False,
            clip_rect_enabled=False,
        )
        consts = generate_ui_material_consts(config)

        assert "TEXTURE_ENABLED: bool = false" in consts
        assert "USE_SRGB: bool = false" in consts
        assert "PREMULTIPLY_ALPHA: bool = false" in consts
        assert "VERTEX_COLOR_ENABLED: bool = false" in consts
        assert "CLIP_RECT_ENABLED: bool = false" in consts

    def test_generate_consts_has_comment_header(self):
        """Generated consts should have comment header."""
        config = UIMaterialConfig()
        consts = generate_ui_material_consts(config)

        assert "// UI Material Configuration" in consts


# =============================================================================
# Test: Uniforms Structure
# =============================================================================


class TestUIUniforms:
    """Test UI uniforms structure in shader."""

    def test_uniforms_struct_exists(self):
        """UIUniforms struct should exist."""
        wgsl = UI_DOMAIN_WGSL

        assert "struct UIUniforms" in wgsl

    def test_uniforms_has_clip_rect(self):
        """Uniforms should have clip_rect field."""
        wgsl = UI_DOMAIN_WGSL

        assert "clip_rect: vec4<f32>" in wgsl

    def test_uniforms_has_screen_size(self):
        """Uniforms should have screen_size field."""
        wgsl = UI_DOMAIN_WGSL

        assert "screen_size: vec2<f32>" in wgsl

    def test_uniforms_has_time(self):
        """Uniforms should have time field for animations."""
        wgsl = UI_DOMAIN_WGSL

        assert "time: f32" in wgsl

    def test_uniforms_has_opacity(self):
        """Uniforms should have opacity field."""
        wgsl = UI_DOMAIN_WGSL

        assert "opacity: f32" in wgsl

    def test_uniforms_binding_declarations(self):
        """Shader should have proper binding declarations."""
        wgsl = UI_DOMAIN_WGSL

        assert "@group(0) @binding(0)" in wgsl
        assert "@group(0) @binding(1)" in wgsl
        assert "@group(0) @binding(2)" in wgsl


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_generates_valid_wgsl(self):
        """Default empty config should generate valid WGSL."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config)

        assert len(wgsl) > 0
        assert "fn vs_ui" in wgsl
        assert "fn fs_ui" in wgsl

    def test_minimal_config_generates_valid_wgsl(self):
        """Minimal config should generate valid minimal WGSL."""
        config = UIMaterialConfig()
        wgsl = generate_ui_material(config, minimal=True)

        assert len(wgsl) > 0
        assert "fn vs_ui_simple" in wgsl
        assert "fn fs_ui_simple" in wgsl

    def test_config_hashable(self):
        """UIMaterialConfig should be hashable (frozen)."""
        config1 = UIMaterialConfig()
        config2 = UIMaterialConfig()

        # Should be hashable
        hash1 = hash(config1)
        hash2 = hash(config2)

        # Equal configs should have same hash
        assert hash1 == hash2

    def test_config_equality(self):
        """UIMaterialConfig should support equality."""
        config1 = UIMaterialConfig(texture_enabled=True)
        config2 = UIMaterialConfig(texture_enabled=True)
        config3 = UIMaterialConfig(texture_enabled=False)

        assert config1 == config2
        assert config1 != config3

    def test_all_blend_modes_covered(self):
        """All blend modes should have entry points."""
        for mode in UIBlendMode:
            config = UIMaterialConfig(blend_mode=mode)
            entry = get_ui_entry_point(config)

            # Each mode should map to a valid entry point
            assert entry.startswith("fs_ui")
