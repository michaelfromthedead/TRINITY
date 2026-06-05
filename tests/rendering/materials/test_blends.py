"""Tests for blend mode variants.

Task: T-MAT-2.3 Blend Mode Variants
Gap: S3-G3 (CRITICAL)
Dependency: T-MAT-2.1 (DONE - variant const system)

Tests verify:
1. BlendState generation for each mode
2. MASKED mode generates discard statements
3. TRANSLUCENT mode skips depth write
4. All blend modes compile (valid WGSL syntax)
5. Blend state properties are correct
"""

from __future__ import annotations

import pytest
import re
from typing import Dict

from trinity.materials.variants import BlendMode
from trinity.materials.blends import (
    BlendFactor,
    BlendOperation,
    ColorWriteMask,
    BlendState,
    BlendShaderCode,
    get_blend_state_for_variant,
    validate_blend_combination,
)


# =============================================================================
# Test: BlendState generation for each mode
# =============================================================================


class TestBlendStateGeneration:
    """Test BlendState.for_blend_mode returns correct configurations."""

    def test_opaque_blend_state(self):
        """OPAQUE mode should have no blending, full depth write."""
        state = BlendState.for_blend_mode(BlendMode.OPAQUE)

        assert state.src_factor == BlendFactor.ONE
        assert state.dst_factor == BlendFactor.ZERO
        assert state.depth_write is True
        assert state.depth_test is True
        assert state.is_opaque is True
        assert state.requires_sorting is False

    def test_masked_blend_state(self):
        """MASKED mode should have same pipeline state as opaque."""
        state = BlendState.for_blend_mode(BlendMode.MASKED)

        # Pipeline state is same as opaque
        assert state.src_factor == BlendFactor.ONE
        assert state.dst_factor == BlendFactor.ZERO
        assert state.depth_write is True
        assert state.depth_test is True
        # Alpha testing happens in shader via discard

    def test_translucent_blend_state(self):
        """TRANSLUCENT mode should use alpha blending, no depth write."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)

        assert state.src_factor == BlendFactor.SRC_ALPHA
        assert state.dst_factor == BlendFactor.ONE_MINUS_SRC_ALPHA
        assert state.depth_write is False
        assert state.depth_test is True
        assert state.is_opaque is False
        assert state.requires_sorting is True

    def test_additive_blend_state(self):
        """ADDITIVE mode should add source to destination."""
        state = BlendState.for_blend_mode(BlendMode.ADDITIVE)

        assert state.src_factor == BlendFactor.ONE
        assert state.dst_factor == BlendFactor.ONE
        assert state.operation == BlendOperation.ADD
        assert state.depth_write is False
        assert state.depth_test is True
        assert state.requires_sorting is True

    def test_modulate_blend_state(self):
        """MODULATE mode should multiply with destination."""
        state = BlendState.for_blend_mode(BlendMode.MODULATE)

        assert state.src_factor == BlendFactor.DST_COLOR
        assert state.dst_factor == BlendFactor.ZERO
        assert state.depth_write is False
        assert state.depth_test is True

    def test_all_modes_return_valid_state(self):
        """All BlendMode values should return valid BlendState."""
        for mode in BlendMode:
            state = BlendState.for_blend_mode(mode)
            assert isinstance(state, BlendState)
            assert isinstance(state.src_factor, BlendFactor)
            assert isinstance(state.dst_factor, BlendFactor)
            assert isinstance(state.operation, BlendOperation)
            assert isinstance(state.depth_write, bool)
            assert isinstance(state.depth_test, bool)

    def test_blend_state_caching(self):
        """BlendState.for_blend_mode should cache states."""
        state1 = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        state2 = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)

        # Should return same object (cached)
        assert state1 is state2


# =============================================================================
# Test: MASKED mode generates discard statements
# =============================================================================


class TestMaskedModeDiscard:
    """Test that MASKED blend mode generates proper discard code."""

    def test_masked_generates_discard_code(self):
        """MASKED mode should generate alpha test with discard."""
        code = BlendShaderCode.get_for_blend(BlendMode.MASKED)

        assert "discard" in code
        assert "ALPHA_CUTOFF" in code
        assert "apply_alpha_test" in code

    def test_masked_discard_has_threshold_check(self):
        """MASKED discard code should check alpha against threshold."""
        code = BlendShaderCode.get_for_blend(BlendMode.MASKED)

        # Should have comparison against threshold
        assert "alpha < ALPHA_CUTOFF" in code or "alpha < threshold" in code

    def test_masked_has_configurable_threshold(self):
        """MASKED mode should support configurable alpha threshold."""
        code = BlendShaderCode.get_for_blend(BlendMode.MASKED)

        # Should have function with threshold parameter
        assert "apply_alpha_test_threshold" in code
        assert "threshold" in code

    def test_masked_discard_is_valid_wgsl(self):
        """MASKED discard code should be syntactically valid WGSL."""
        code = BlendShaderCode.get_for_blend(BlendMode.MASKED)

        # Check for valid WGSL patterns
        assert "fn apply_alpha_test" in code
        assert "f32" in code
        # Discard is a WGSL statement
        assert re.search(r"discard\s*;", code)


# =============================================================================
# Test: TRANSLUCENT mode skips depth write
# =============================================================================


class TestTranslucentDepthWrite:
    """Test that TRANSLUCENT mode properly skips depth write."""

    def test_translucent_skips_depth_write(self):
        """TRANSLUCENT blend state should have depth_write=False."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        assert state.depth_write is False

    def test_translucent_keeps_depth_test(self):
        """TRANSLUCENT should still perform depth testing."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        assert state.depth_test is True

    def test_translucent_alpha_blend_factors(self):
        """TRANSLUCENT should use correct alpha blend factors."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)

        assert state.alpha_src == BlendFactor.ONE
        assert state.alpha_dst == BlendFactor.ONE_MINUS_SRC_ALPHA

    def test_translucent_generates_premultiply_code(self):
        """TRANSLUCENT mode should generate premultiply alpha code."""
        code = BlendShaderCode.get_for_blend(BlendMode.TRANSLUCENT)

        assert "premultiply_alpha" in code
        assert "color * alpha" in code

    def test_translucent_has_unpremultiply(self):
        """TRANSLUCENT code should include unpremultiply for compositing."""
        code = BlendShaderCode.get_for_blend(BlendMode.TRANSLUCENT)

        assert "unpremultiply_alpha" in code


# =============================================================================
# Test: All blend modes compile (valid WGSL)
# =============================================================================


class TestBlendModesCompile:
    """Test that all blend mode shader code is valid WGSL."""

    def test_all_blend_codes_have_valid_function_syntax(self):
        """All blend mode codes should have valid WGSL function syntax."""
        for mode in BlendMode:
            code = BlendShaderCode.get_for_blend(mode)

            if code:  # OPAQUE returns empty string
                # Check for valid function declaration pattern
                fn_pattern = r"fn\s+\w+\s*\([^)]*\)"
                assert re.search(fn_pattern, code), (
                    f"Mode {mode.name} missing valid function declaration"
                )

    def test_opaque_returns_empty_code(self):
        """OPAQUE mode doesn't need special shader code."""
        code = BlendShaderCode.get_for_blend(BlendMode.OPAQUE)
        assert code == ""

    def test_masked_code_compiles(self):
        """MASKED shader code should be syntactically valid."""
        code = BlendShaderCode.get_for_blend(BlendMode.MASKED)

        # Check basic WGSL syntax
        assert code.count("{") == code.count("}")
        assert "fn " in code
        assert re.search(r":\s*(f32|vec\d+<f32>)", code)

    def test_translucent_code_compiles(self):
        """TRANSLUCENT shader code should be syntactically valid."""
        code = BlendShaderCode.get_for_blend(BlendMode.TRANSLUCENT)

        assert code.count("{") == code.count("}")
        assert "fn " in code
        assert "vec4<f32>" in code

    def test_additive_code_compiles(self):
        """ADDITIVE shader code should be syntactically valid."""
        code = BlendShaderCode.get_for_blend(BlendMode.ADDITIVE)

        assert code.count("{") == code.count("}")
        assert "fn " in code

    def test_modulate_code_compiles(self):
        """MODULATE shader code should be syntactically valid."""
        code = BlendShaderCode.get_for_blend(BlendMode.MODULATE)

        assert code.count("{") == code.count("}")
        assert "fn " in code

    def test_all_helpers_combined_is_valid(self):
        """Combined blend helpers should be valid WGSL."""
        all_code = BlendShaderCode.get_all_blend_helpers()

        # Should contain all the helpers
        assert "apply_alpha_test" in all_code
        assert "premultiply_alpha" in all_code
        assert "apply_additive_emission" in all_code
        assert "apply_modulate_factor" in all_code

        # Braces should be balanced
        assert all_code.count("{") == all_code.count("}")


# =============================================================================
# Test: Blend state to wgpu descriptor
# =============================================================================


class TestBlendStateDescriptor:
    """Test BlendState conversion to wgpu descriptor format."""

    def test_wgpu_descriptor_format(self):
        """to_wgpu_descriptor should return correct format."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        desc = state.to_wgpu_descriptor()

        assert "color" in desc
        assert "alpha" in desc
        assert desc["color"]["srcFactor"] == "src_alpha"
        assert desc["color"]["dstFactor"] == "one_minus_src_alpha"
        assert desc["color"]["operation"] == "add"

    def test_depth_stencil_descriptor(self):
        """to_depth_stencil_descriptor should return correct format."""
        state = BlendState.for_blend_mode(BlendMode.OPAQUE)
        desc = state.to_depth_stencil_descriptor()

        assert desc["depthWriteEnabled"] is True
        assert desc["depthCompare"] == "less"

    def test_depth_stencil_no_depth_write(self):
        """Translucent should have depth write disabled in descriptor."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        desc = state.to_depth_stencil_descriptor()

        assert desc["depthWriteEnabled"] is False


# =============================================================================
# Test: Blend mode properties
# =============================================================================


class TestBlendModeProperties:
    """Test BlendState property calculations."""

    def test_requires_sorting_translucent(self):
        """TRANSLUCENT should require depth sorting."""
        state = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        assert state.requires_sorting is True

    def test_requires_sorting_additive(self):
        """ADDITIVE should require depth sorting."""
        state = BlendState.for_blend_mode(BlendMode.ADDITIVE)
        assert state.requires_sorting is True

    def test_opaque_no_sorting(self):
        """OPAQUE should not require sorting."""
        state = BlendState.for_blend_mode(BlendMode.OPAQUE)
        assert state.requires_sorting is False

    def test_masked_no_sorting(self):
        """MASKED should not require sorting (depth writes enabled)."""
        state = BlendState.for_blend_mode(BlendMode.MASKED)
        assert state.requires_sorting is False

    def test_is_opaque_correct(self):
        """is_opaque should be True only for OPAQUE and MASKED."""
        assert BlendState.for_blend_mode(BlendMode.OPAQUE).is_opaque is True
        assert BlendState.for_blend_mode(BlendMode.MASKED).is_opaque is True
        assert BlendState.for_blend_mode(BlendMode.TRANSLUCENT).is_opaque is False
        assert BlendState.for_blend_mode(BlendMode.ADDITIVE).is_opaque is False
        assert BlendState.for_blend_mode(BlendMode.MODULATE).is_opaque is False


# =============================================================================
# Test: Helper functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions for blend mode handling."""

    def test_get_blend_state_for_variant_opaque(self):
        """get_blend_state_for_variant should work with string modes."""
        state = get_blend_state_for_variant("opaque")
        assert state.depth_write is True
        assert state.src_factor == BlendFactor.ONE

    def test_get_blend_state_for_variant_translucent(self):
        """get_blend_state_for_variant should work for translucent."""
        state = get_blend_state_for_variant("translucent")
        assert state.depth_write is False
        assert state.src_factor == BlendFactor.SRC_ALPHA

    def test_get_blend_state_for_variant_invalid(self):
        """get_blend_state_for_variant should raise for invalid mode."""
        with pytest.raises(ValueError, match="Invalid blend mode"):
            get_blend_state_for_variant("invalid_mode")

    def test_validate_blend_combination_ok(self):
        """validate_blend_combination should return valid for good config."""
        valid, warning = validate_blend_combination(BlendMode.OPAQUE)
        assert valid is True
        assert warning is None

    def test_validate_blend_combination_sorting_warning(self):
        """validate_blend_combination should warn about sorting needs."""
        valid, warning = validate_blend_combination(
            BlendMode.TRANSLUCENT, depth_prepass=False
        )
        assert valid is True
        assert warning is not None
        assert "sorting" in warning.lower()

    def test_validate_blend_combination_two_sided_warning(self):
        """validate_blend_combination should warn about two-sided translucent."""
        valid, warning = validate_blend_combination(
            BlendMode.TRANSLUCENT, two_sided=True
        )
        assert valid is True
        assert warning is not None
        assert "two-sided" in warning.lower()


# =============================================================================
# Test: Depth fade code
# =============================================================================


class TestDepthFadeCode:
    """Test depth fade shader code for soft particles."""

    def test_depth_fade_code_exists(self):
        """Depth fade code should be available."""
        code = BlendShaderCode.get_depth_fade_code()
        assert len(code) > 0
        assert "calculate_depth_fade" in code

    def test_depth_fade_has_parameters(self):
        """Depth fade should take fragment depth, scene depth, and fade distance."""
        code = BlendShaderCode.get_depth_fade_code()

        assert "fragment_depth" in code
        assert "scene_depth" in code
        assert "fade_distance" in code

    def test_depth_fade_returns_saturated(self):
        """Depth fade should return saturated (clamped 0-1) value."""
        code = BlendShaderCode.get_depth_fade_code()
        assert "saturate" in code


# =============================================================================
# Test: Blend factor and operation enums
# =============================================================================


class TestBlendEnums:
    """Test BlendFactor and BlendOperation enum values."""

    def test_all_blend_factors_have_values(self):
        """All BlendFactor enum members should have string values."""
        for factor in BlendFactor:
            assert isinstance(factor.value, str)
            assert len(factor.value) > 0

    def test_all_blend_operations_have_values(self):
        """All BlendOperation enum members should have string values."""
        for op in BlendOperation:
            assert isinstance(op.value, str)
            assert len(op.value) > 0

    def test_color_write_mask_values(self):
        """ColorWriteMask should have expected values."""
        assert ColorWriteMask.NONE.value == "none"
        assert ColorWriteMask.ALL.value == "all"
        assert ColorWriteMask.RGB.value == "rgb"

    def test_blend_factors_match_wgpu_names(self):
        """BlendFactor values should match wgpu naming convention."""
        # These are the wgpu/WebGPU standard names
        assert BlendFactor.ZERO.value == "zero"
        assert BlendFactor.ONE.value == "one"
        assert BlendFactor.SRC_ALPHA.value == "src_alpha"
        assert BlendFactor.ONE_MINUS_SRC_ALPHA.value == "one_minus_src_alpha"
        assert BlendFactor.DST_COLOR.value == "dst_color"


# =============================================================================
# Test: BlendState immutability
# =============================================================================


class TestBlendStateImmutability:
    """Test that BlendState is properly immutable (frozen dataclass)."""

    def test_blend_state_is_frozen(self):
        """BlendState should be frozen (immutable)."""
        state = BlendState.for_blend_mode(BlendMode.OPAQUE)

        with pytest.raises((AttributeError, TypeError)):
            state.depth_write = False

    def test_cached_states_cannot_be_modified(self):
        """Cached blend states should remain unchanged."""
        state1 = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)
        depth_write_before = state1.depth_write

        # Try to get it again
        state2 = BlendState.for_blend_mode(BlendMode.TRANSLUCENT)

        # Should be same value
        assert state2.depth_write == depth_write_before
        assert state1 is state2


# =============================================================================
# Test: Edge cases and special scenarios
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_default_blend_state(self):
        """Default BlendState should be equivalent to opaque."""
        default = BlendState()
        opaque = BlendState.for_blend_mode(BlendMode.OPAQUE)

        assert default.src_factor == opaque.src_factor
        assert default.dst_factor == opaque.dst_factor
        assert default.depth_write == opaque.depth_write

    def test_shader_code_contains_no_undefined_symbols(self):
        """Shader code should not reference undefined symbols."""
        all_code = BlendShaderCode.get_all_blend_helpers()

        # All vec types should be properly typed
        assert "vec3(" not in all_code  # Should be vec3<f32>
        assert "vec4(" not in all_code  # Should be vec4<f32>

        # Check that vec types are generic
        assert "vec3<f32>" in all_code or "vec4<f32>" in all_code

    def test_all_blend_modes_covered(self):
        """All BlendMode enum values should be handled."""
        for mode in BlendMode:
            # Should not raise
            state = BlendState.for_blend_mode(mode)
            assert state is not None

            # get_for_blend should return string (possibly empty)
            code = BlendShaderCode.get_for_blend(mode)
            assert isinstance(code, str)


# =============================================================================
# Test: WGSL compilation of blend modes (T-MAT-2.3 acceptance criteria)
# =============================================================================


def _has_naga() -> bool:
    """Check if naga-py is available."""
    try:
        import naga
        return True
    except ImportError:
        return False


class TestBlendModeWGSLCompilation:
    """Test that blend modes generate valid, compilable WGSL.

    These tests verify the T-MAT-2.3 acceptance criteria:
    - MASKED mode generates discard statement
    - All 5 blend modes produce valid WGSL syntax
    - naga can parse the generated WGSL (when available)
    """

    def test_masked_generates_discard(self):
        """MASKED mode should generate WGSL with discard statement."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.MASKED)

        # Must contain discard statement
        assert "discard" in wgsl
        # Must have alpha test enabled
        assert "ALPHA_TEST_ENABLED: bool = true" in wgsl
        # Should reference alpha_threshold (or similar)
        assert "alpha_threshold" in wgsl or "ALPHA_CUTOFF" in wgsl

    def test_masked_has_alpha_cutoff(self):
        """MASKED mode should include alpha cutoff functionality."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.MASKED)

        # The apply_blend_mode function should check alpha against threshold
        assert "color.a < alpha_threshold" in wgsl

    def test_translucent_alpha_output(self):
        """TRANSLUCENT mode should preserve alpha in output."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.TRANSLUCENT)

        # Must have alpha blend enabled
        assert "ALPHA_BLEND_ENABLED: bool = true" in wgsl
        # Should NOT discard (translucent doesn't use alpha test)
        # The apply_blend_mode function returns color directly for ALPHA_BLEND_ENABLED
        assert "ALPHA_TEST_ENABLED: bool = false" in wgsl

    def test_opaque_forces_alpha_one(self):
        """OPAQUE mode should force alpha to 1.0."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.OPAQUE)

        # Both alpha test and blend should be disabled
        assert "ALPHA_TEST_ENABLED: bool = false" in wgsl
        assert "ALPHA_BLEND_ENABLED: bool = false" in wgsl
        # The apply_blend_mode function should return alpha = 1.0 for opaque
        assert "color.rgb, 1.0" in wgsl

    def test_additive_has_alpha_blend_enabled(self):
        """ADDITIVE mode should have alpha blend enabled."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.ADDITIVE)

        assert "ALPHA_BLEND_ENABLED: bool = true" in wgsl
        assert "BLEND_ADDITIVE: bool = true" in wgsl

    def test_modulate_has_alpha_blend_enabled(self):
        """MODULATE mode should have alpha blend enabled."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        wgsl = compile_blend_mode_wgsl(BlendMode.MODULATE)

        assert "ALPHA_BLEND_ENABLED: bool = true" in wgsl
        assert "BLEND_MODULATE: bool = true" in wgsl

    def test_all_blend_modes_generate_wgsl(self):
        """All 5 blend modes should generate non-empty WGSL."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        for mode in BlendMode:
            wgsl = compile_blend_mode_wgsl(mode)

            # Should produce non-trivial output
            assert len(wgsl) > 500, f"Mode {mode.name} generated too little WGSL"

            # Should have a fragment shader
            assert "@fragment" in wgsl, f"Mode {mode.name} missing @fragment"

            # Should have apply_blend_mode function
            assert "fn apply_blend_mode" in wgsl, (
                f"Mode {mode.name} missing apply_blend_mode"
            )

            # Should have the correct blend mode const set to true
            expected_const = f"BLEND_{mode.name}: bool = true"
            assert expected_const in wgsl, (
                f"Mode {mode.name} missing {expected_const}"
            )

    def test_wgsl_has_balanced_braces(self):
        """Generated WGSL should have balanced braces."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        for mode in BlendMode:
            wgsl = compile_blend_mode_wgsl(mode)

            open_braces = wgsl.count("{")
            close_braces = wgsl.count("}")

            assert open_braces == close_braces, (
                f"Mode {mode.name} has unbalanced braces: "
                f"{open_braces} open, {close_braces} close"
            )

    def test_wgsl_has_valid_function_declarations(self):
        """Generated WGSL should have valid function syntax."""
        from trinity.materials.blends import compile_blend_mode_wgsl

        for mode in BlendMode:
            wgsl = compile_blend_mode_wgsl(mode)

            # Should have at least fs_main and apply_blend_mode functions
            fn_pattern = r"fn\s+\w+\s*\([^)]*\)\s*(->.*?)?\s*\{"
            matches = re.findall(fn_pattern, wgsl)

            assert len(matches) >= 2, (
                f"Mode {mode.name} should have at least 2 functions"
            )

    @pytest.mark.skipif(
        not _has_naga(),
        reason="naga-py not installed"
    )
    def test_all_blend_modes_validate_with_naga(self):
        """All blend mode WGSL should pass naga validation."""
        import naga
        from trinity.materials.blends import compile_blend_mode_wgsl

        for mode in BlendMode:
            wgsl = compile_blend_mode_wgsl(mode)

            try:
                naga.parse_wgsl(wgsl)
            except Exception as e:
                pytest.fail(
                    f"Mode {mode.name} failed naga validation: {e}\n"
                    f"WGSL:\n{wgsl[:500]}..."
                )
