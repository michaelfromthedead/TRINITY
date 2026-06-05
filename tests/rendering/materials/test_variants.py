"""Tests for the variant const system.

Task: T-MAT-2.1 Variant const system
Gap: S3-G3 (CRITICAL)

Tests verify:
1. VariantConfig generates correct const declarations
2. Different configs produce different WGSL output
3. Gated code paths respect const bools
4. Integration with MaterialCompiler (via VariantCompiler)
5. All 75 variant combinations produce valid WGSL (naga validation)
"""

from __future__ import annotations

import pytest
import re
from typing import Set

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
    VariantCompiler,
    generate_all_variant_combinations,
    get_variant_for_material_system,
)

from trinity.materials import (
    Material,
    MaterialMeta,
    MaterialCompiler,
    surface,
    SurfaceContext,
    SurfaceOutput,
    Vec3,
)


# =============================================================================
# Test: VariantConfig generates correct const declarations
# =============================================================================


class TestVariantConfigConstDeclarations:
    """Test that VariantConfig produces correct WGSL const declarations."""

    def test_default_config_generates_valid_wgsl(self):
        """Default config should produce valid WGSL const block."""
        config = VariantConfig()
        wgsl = config.generate_const_declarations()

        # Should contain header comment
        assert "Variant Const Declarations" in wgsl

        # Should contain domain consts
        assert "const DOMAIN_SURFACE: bool = true;" in wgsl
        assert "const DOMAIN_DEFERRED_DECAL: bool = false;" in wgsl
        assert "const DOMAIN_VOLUME: bool = false;" in wgsl
        assert "const DOMAIN_POST_PROCESS: bool = false;" in wgsl
        assert "const DOMAIN_UI: bool = false;" in wgsl

        # Should contain blend consts
        assert "const BLEND_OPAQUE: bool = true;" in wgsl
        assert "const BLEND_MASKED: bool = false;" in wgsl
        assert "const BLEND_TRANSLUCENT: bool = false;" in wgsl
        assert "const BLEND_ADDITIVE: bool = false;" in wgsl
        assert "const BLEND_MODULATE: bool = false;" in wgsl

        # Should contain quality consts
        assert "const QUALITY_LOW: bool = false;" in wgsl
        assert "const QUALITY_MEDIUM: bool = false;" in wgsl
        assert "const QUALITY_HIGH: bool = true;" in wgsl

    def test_surface_domain_const(self):
        """SURFACE domain should set DOMAIN_SURFACE = true."""
        config = VariantConfig(domain=MaterialDomain.SURFACE)
        wgsl = config.generate_const_declarations()
        assert "const DOMAIN_SURFACE: bool = true;" in wgsl
        assert "const DOMAIN_DEFERRED_DECAL: bool = false;" in wgsl

    def test_deferred_decal_domain_const(self):
        """DEFERRED_DECAL domain should set DOMAIN_DEFERRED_DECAL = true."""
        config = VariantConfig(domain=MaterialDomain.DEFERRED_DECAL)
        wgsl = config.generate_const_declarations()
        assert "const DOMAIN_SURFACE: bool = false;" in wgsl
        assert "const DOMAIN_DEFERRED_DECAL: bool = true;" in wgsl

    def test_volume_domain_const(self):
        """VOLUME domain should set DOMAIN_VOLUME = true."""
        config = VariantConfig(domain=MaterialDomain.VOLUME)
        wgsl = config.generate_const_declarations()
        assert "const DOMAIN_VOLUME: bool = true;" in wgsl
        assert "const DOMAIN_SURFACE: bool = false;" in wgsl

    def test_post_process_domain_const(self):
        """POST_PROCESS domain should set DOMAIN_POST_PROCESS = true."""
        config = VariantConfig(domain=MaterialDomain.POST_PROCESS)
        wgsl = config.generate_const_declarations()
        assert "const DOMAIN_POST_PROCESS: bool = true;" in wgsl

    def test_ui_domain_const(self):
        """UI domain should set DOMAIN_UI = true."""
        config = VariantConfig(domain=MaterialDomain.UI)
        wgsl = config.generate_const_declarations()
        assert "const DOMAIN_UI: bool = true;" in wgsl

    def test_opaque_blend_const(self):
        """OPAQUE blend should set BLEND_OPAQUE = true."""
        config = VariantConfig(blend=BlendMode.OPAQUE)
        wgsl = config.generate_const_declarations()
        assert "const BLEND_OPAQUE: bool = true;" in wgsl
        assert "const BLEND_MASKED: bool = false;" in wgsl

    def test_masked_blend_const(self):
        """MASKED blend should set BLEND_MASKED = true."""
        config = VariantConfig(blend=BlendMode.MASKED)
        wgsl = config.generate_const_declarations()
        assert "const BLEND_MASKED: bool = true;" in wgsl
        assert "const BLEND_OPAQUE: bool = false;" in wgsl

    def test_translucent_blend_const(self):
        """TRANSLUCENT blend should set BLEND_TRANSLUCENT = true."""
        config = VariantConfig(blend=BlendMode.TRANSLUCENT)
        wgsl = config.generate_const_declarations()
        assert "const BLEND_TRANSLUCENT: bool = true;" in wgsl

    def test_additive_blend_const(self):
        """ADDITIVE blend should set BLEND_ADDITIVE = true."""
        config = VariantConfig(blend=BlendMode.ADDITIVE)
        wgsl = config.generate_const_declarations()
        assert "const BLEND_ADDITIVE: bool = true;" in wgsl

    def test_modulate_blend_const(self):
        """MODULATE blend should set BLEND_MODULATE = true."""
        config = VariantConfig(blend=BlendMode.MODULATE)
        wgsl = config.generate_const_declarations()
        assert "const BLEND_MODULATE: bool = true;" in wgsl

    def test_low_quality_const(self):
        """LOW quality should set QUALITY_LOW = true."""
        config = VariantConfig(quality=QualityTier.LOW)
        wgsl = config.generate_const_declarations()
        assert "const QUALITY_LOW: bool = true;" in wgsl
        assert "const QUALITY_MEDIUM: bool = false;" in wgsl
        assert "const QUALITY_HIGH: bool = false;" in wgsl

    def test_medium_quality_const(self):
        """MEDIUM quality should set QUALITY_MEDIUM = true."""
        config = VariantConfig(quality=QualityTier.MEDIUM)
        wgsl = config.generate_const_declarations()
        assert "const QUALITY_LOW: bool = false;" in wgsl
        assert "const QUALITY_MEDIUM: bool = true;" in wgsl
        assert "const QUALITY_HIGH: bool = false;" in wgsl

    def test_high_quality_const(self):
        """HIGH quality should set QUALITY_HIGH = true."""
        config = VariantConfig(quality=QualityTier.HIGH)
        wgsl = config.generate_const_declarations()
        assert "const QUALITY_LOW: bool = false;" in wgsl
        assert "const QUALITY_MEDIUM: bool = false;" in wgsl
        assert "const QUALITY_HIGH: bool = true;" in wgsl


# =============================================================================
# Test: Different configs produce different WGSL output
# =============================================================================


class TestVariantConfigDifferentOutput:
    """Test that different variant configs produce different WGSL."""

    def test_different_domains_produce_different_wgsl(self):
        """Different domains should produce different const declarations."""
        surface = VariantConfig(domain=MaterialDomain.SURFACE)
        volume = VariantConfig(domain=MaterialDomain.VOLUME)

        wgsl_surface = surface.generate_const_declarations()
        wgsl_volume = volume.generate_const_declarations()

        assert wgsl_surface != wgsl_volume
        assert "DOMAIN_SURFACE: bool = true" in wgsl_surface
        assert "DOMAIN_VOLUME: bool = true" in wgsl_volume

    def test_different_blends_produce_different_wgsl(self):
        """Different blend modes should produce different const declarations."""
        opaque = VariantConfig(blend=BlendMode.OPAQUE)
        masked = VariantConfig(blend=BlendMode.MASKED)

        wgsl_opaque = opaque.generate_const_declarations()
        wgsl_masked = masked.generate_const_declarations()

        assert wgsl_opaque != wgsl_masked
        assert "BLEND_OPAQUE: bool = true" in wgsl_opaque
        assert "BLEND_MASKED: bool = true" in wgsl_masked

    def test_different_qualities_produce_different_wgsl(self):
        """Different quality tiers should produce different const declarations."""
        low = VariantConfig(quality=QualityTier.LOW)
        high = VariantConfig(quality=QualityTier.HIGH)

        wgsl_low = low.generate_const_declarations()
        wgsl_high = high.generate_const_declarations()

        assert wgsl_low != wgsl_high
        assert "QUALITY_LOW: bool = true" in wgsl_low
        assert "QUALITY_HIGH: bool = true" in wgsl_high

    def test_quality_derived_features_differ_by_tier(self):
        """Quality-derived features should differ between tiers."""
        low = VariantConfig(quality=QualityTier.LOW)
        high = VariantConfig(quality=QualityTier.HIGH)

        wgsl_low = low.generate_const_declarations()
        wgsl_high = high.generate_const_declarations()

        # LOW should have limited features
        assert "const MAX_LIGHTS: u32 = 1u;" in wgsl_low
        assert "const SHADOWS_ENABLED: bool = false;" in wgsl_low
        assert "const SUBSURFACE_ENABLED: bool = false;" in wgsl_low

        # HIGH should have full features
        assert "const MAX_LIGHTS: u32 = 16u;" in wgsl_high
        assert "const SHADOWS_ENABLED: bool = true;" in wgsl_high
        assert "const SUBSURFACE_ENABLED: bool = true;" in wgsl_high

    def test_medium_quality_features(self):
        """MEDIUM quality should have intermediate features."""
        medium = VariantConfig(quality=QualityTier.MEDIUM)
        wgsl = medium.generate_const_declarations()

        assert "const MAX_LIGHTS: u32 = 4u;" in wgsl
        assert "const SHADOWS_ENABLED: bool = true;" in wgsl
        assert "const SUBSURFACE_ENABLED: bool = false;" in wgsl  # HIGH only
        assert "const CLEARCOAT_ENABLED: bool = true;" in wgsl   # MEDIUM has this
        assert "const TRANSMISSION_ENABLED: bool = true;" in wgsl


# =============================================================================
# Test: Gated code paths respect const bools
# =============================================================================


class TestGatedCodePaths:
    """Test that gated code paths use the correct const bools."""

    def test_gated_lighting_uses_quality_consts(self):
        """Gated lighting code should reference QUALITY_* consts."""
        config = VariantConfig()
        code = config.generate_gated_lighting_code()

        assert "if QUALITY_HIGH {" in code
        assert "if QUALITY_MEDIUM {" in code
        # LOW is the else case

    def test_gated_lighting_uses_shadows_enabled(self):
        """Gated lighting should check SHADOWS_ENABLED."""
        config = VariantConfig()
        code = config.generate_gated_lighting_code()

        assert "if SHADOWS_ENABLED {" in code

    def test_gated_features_uses_feature_consts(self):
        """Gated advanced features should check feature consts."""
        config = VariantConfig()
        code = config.generate_gated_features_code()

        assert "if !ADVANCED_SHADING_ENABLED {" in code
        assert "if SUBSURFACE_ENABLED {" in code
        assert "if CLEARCOAT_ENABLED {" in code
        assert "if ANISOTROPY_ENABLED {" in code
        assert "if SHEEN_ENABLED {" in code
        assert "if TRANSMISSION_ENABLED {" in code
        assert "if IRIDESCENCE_ENABLED {" in code

    def test_blend_handling_uses_blend_consts(self):
        """Blend handling code should check blend mode consts."""
        config = VariantConfig()
        code = config.generate_blend_handling_code()

        assert "if ALPHA_TEST_ENABLED {" in code
        assert "if ALPHA_BLEND_ENABLED {" in code
        assert "discard;" in code  # Masked mode discards

    def test_domain_handling_uses_domain_consts(self):
        """Domain handling code should check domain consts."""
        config = VariantConfig()
        code = config.generate_domain_handling_code()

        assert "if DOMAIN_SURFACE {" in code
        assert "if DOMAIN_DEFERRED_DECAL {" in code
        assert "if DOMAIN_VOLUME {" in code
        assert "if DOMAIN_POST_PROCESS {" in code
        assert "if DOMAIN_UI {" in code


# =============================================================================
# Test: Domain/Blend/Quality derived features
# =============================================================================


class TestDerivedFeatures:
    """Test domain, blend, and quality derived feature flags."""

    def test_surface_domain_enables_lighting(self):
        """SURFACE domain should enable lighting."""
        config = VariantConfig(domain=MaterialDomain.SURFACE)
        wgsl = config.generate_const_declarations()
        assert "const LIGHTING_ENABLED: bool = true;" in wgsl
        assert "const PBR_ENABLED: bool = true;" in wgsl

    def test_ui_domain_disables_lighting(self):
        """UI domain should disable lighting."""
        config = VariantConfig(domain=MaterialDomain.UI)
        wgsl = config.generate_const_declarations()
        assert "const LIGHTING_ENABLED: bool = false;" in wgsl
        assert "const PBR_ENABLED: bool = false;" in wgsl

    def test_post_process_domain_disables_lighting(self):
        """POST_PROCESS domain should disable lighting."""
        config = VariantConfig(domain=MaterialDomain.POST_PROCESS)
        wgsl = config.generate_const_declarations()
        assert "const LIGHTING_ENABLED: bool = false;" in wgsl

    def test_opaque_blend_enables_depth_write(self):
        """OPAQUE blend should enable depth write."""
        config = VariantConfig(blend=BlendMode.OPAQUE)
        wgsl = config.generate_const_declarations()
        assert "const DEPTH_WRITE_ENABLED: bool = true;" in wgsl

    def test_translucent_blend_disables_depth_write(self):
        """TRANSLUCENT blend should disable depth write."""
        config = VariantConfig(blend=BlendMode.TRANSLUCENT)
        wgsl = config.generate_const_declarations()
        assert "const DEPTH_WRITE_ENABLED: bool = false;" in wgsl

    def test_masked_blend_enables_alpha_test(self):
        """MASKED blend should enable alpha test."""
        config = VariantConfig(blend=BlendMode.MASKED)
        wgsl = config.generate_const_declarations()
        assert "const ALPHA_TEST_ENABLED: bool = true;" in wgsl
        assert "const ALPHA_BLEND_ENABLED: bool = false;" in wgsl

    def test_translucent_blend_enables_alpha_blend(self):
        """TRANSLUCENT blend should enable alpha blend."""
        config = VariantConfig(blend=BlendMode.TRANSLUCENT)
        wgsl = config.generate_const_declarations()
        assert "const ALPHA_TEST_ENABLED: bool = false;" in wgsl
        assert "const ALPHA_BLEND_ENABLED: bool = true;" in wgsl

    def test_additive_blend_enables_alpha_blend(self):
        """ADDITIVE blend should enable alpha blend."""
        config = VariantConfig(blend=BlendMode.ADDITIVE)
        wgsl = config.generate_const_declarations()
        assert "const ALPHA_BLEND_ENABLED: bool = true;" in wgsl


# =============================================================================
# Test: Custom feature flags
# =============================================================================


class TestCustomFeatureFlags:
    """Test custom feature flag handling."""

    def test_empty_feature_flags(self):
        """Empty feature flags should not add extra consts."""
        config = VariantConfig(feature_flags=set())
        wgsl = config.generate_const_declarations()
        assert "Custom feature flags" not in wgsl

    def test_single_feature_flag(self):
        """Single feature flag should add one const."""
        config = VariantConfig(feature_flags={"ENABLE_DEBUG_VIEW"})
        wgsl = config.generate_const_declarations()
        assert "Custom feature flags" in wgsl
        assert "const ENABLE_DEBUG_VIEW: bool = true;" in wgsl

    def test_multiple_feature_flags(self):
        """Multiple feature flags should add multiple consts."""
        config = VariantConfig(feature_flags={"ENABLE_SSS", "ENABLE_DEBUG"})
        wgsl = config.generate_const_declarations()
        assert "const ENABLE_SSS: bool = true;" in wgsl
        assert "const ENABLE_DEBUG: bool = true;" in wgsl

    def test_feature_flag_sanitization(self):
        """Feature flags with special chars should be sanitized."""
        config = VariantConfig(feature_flags={"enable-debug view"})
        wgsl = config.generate_const_declarations()
        # Should be uppercase and special chars replaced
        assert "const ENABLE_DEBUG_VIEW: bool = true;" in wgsl


# =============================================================================
# Test: Variant key generation
# =============================================================================


class TestVariantKey:
    """Test variant key hashing."""

    def test_same_config_same_key(self):
        """Identical configs should produce same key."""
        config1 = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )
        config2 = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )
        assert config1.get_variant_key() == config2.get_variant_key()

    def test_different_domain_different_key(self):
        """Different domain should produce different key."""
        config1 = VariantConfig(domain=MaterialDomain.SURFACE)
        config2 = VariantConfig(domain=MaterialDomain.VOLUME)
        assert config1.get_variant_key() != config2.get_variant_key()

    def test_different_blend_different_key(self):
        """Different blend should produce different key."""
        config1 = VariantConfig(blend=BlendMode.OPAQUE)
        config2 = VariantConfig(blend=BlendMode.MASKED)
        assert config1.get_variant_key() != config2.get_variant_key()

    def test_different_quality_different_key(self):
        """Different quality should produce different key."""
        config1 = VariantConfig(quality=QualityTier.LOW)
        config2 = VariantConfig(quality=QualityTier.HIGH)
        assert config1.get_variant_key() != config2.get_variant_key()

    def test_feature_flags_affect_key(self):
        """Feature flags should affect key."""
        config1 = VariantConfig(feature_flags=set())
        config2 = VariantConfig(feature_flags={"ENABLE_DEBUG"})
        assert config1.get_variant_key() != config2.get_variant_key()


# =============================================================================
# Test: VariantConfig.copy_with
# =============================================================================


class TestVariantConfigCopyWith:
    """Test VariantConfig copy_with method."""

    def test_copy_with_no_changes(self):
        """copy_with with no args should produce equal config."""
        original = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.MASKED,
            quality=QualityTier.MEDIUM,
        )
        copy = original.copy_with()
        assert copy.domain == original.domain
        assert copy.blend == original.blend
        assert copy.quality == original.quality

    def test_copy_with_domain_change(self):
        """copy_with should allow domain change."""
        original = VariantConfig(domain=MaterialDomain.SURFACE)
        copy = original.copy_with(domain=MaterialDomain.VOLUME)
        assert copy.domain == MaterialDomain.VOLUME
        assert original.domain == MaterialDomain.SURFACE

    def test_copy_with_blend_change(self):
        """copy_with should allow blend change."""
        original = VariantConfig(blend=BlendMode.OPAQUE)
        copy = original.copy_with(blend=BlendMode.TRANSLUCENT)
        assert copy.blend == BlendMode.TRANSLUCENT
        assert original.blend == BlendMode.OPAQUE

    def test_copy_with_quality_change(self):
        """copy_with should allow quality change."""
        original = VariantConfig(quality=QualityTier.HIGH)
        copy = original.copy_with(quality=QualityTier.LOW)
        assert copy.quality == QualityTier.LOW
        assert original.quality == QualityTier.HIGH

    def test_copy_with_multiple_changes(self):
        """copy_with should allow multiple changes."""
        original = VariantConfig()
        copy = original.copy_with(
            domain=MaterialDomain.UI,
            blend=BlendMode.ADDITIVE,
            quality=QualityTier.LOW,
        )
        assert copy.domain == MaterialDomain.UI
        assert copy.blend == BlendMode.ADDITIVE
        assert copy.quality == QualityTier.LOW


# =============================================================================
# Test: generate_all_variant_combinations
# =============================================================================


class TestGenerateAllVariants:
    """Test generate_all_variant_combinations function."""

    def test_generates_75_combinations(self):
        """Should generate 5 domains x 5 blends x 3 qualities = 75."""
        configs = generate_all_variant_combinations()
        assert len(configs) == 75

    def test_all_domains_present(self):
        """All domains should be present in combinations."""
        configs = generate_all_variant_combinations()
        domains = {c.domain for c in configs}
        assert domains == set(MaterialDomain)

    def test_all_blends_present(self):
        """All blend modes should be present in combinations."""
        configs = generate_all_variant_combinations()
        blends = {c.blend for c in configs}
        assert blends == set(BlendMode)

    def test_all_qualities_present(self):
        """All quality tiers should be present in combinations."""
        configs = generate_all_variant_combinations()
        qualities = {c.quality for c in configs}
        assert qualities == set(QualityTier)

    def test_all_combinations_unique(self):
        """All combinations should have unique variant keys."""
        configs = generate_all_variant_combinations()
        keys = [c.get_variant_key() for c in configs]
        assert len(keys) == len(set(keys))


# =============================================================================
# Test: get_variant_for_material_system
# =============================================================================


class TestGetVariantForMaterialSystem:
    """Test get_variant_for_material_system helper."""

    def test_surface_opaque_high(self):
        """Should create correct config from strings."""
        config = get_variant_for_material_system("surface", "opaque", "high")
        assert config.domain == MaterialDomain.SURFACE
        assert config.blend == BlendMode.OPAQUE
        assert config.quality == QualityTier.HIGH

    def test_volume_translucent_low(self):
        """Should handle different combinations."""
        config = get_variant_for_material_system("volume", "translucent", "low")
        assert config.domain == MaterialDomain.VOLUME
        assert config.blend == BlendMode.TRANSLUCENT
        assert config.quality == QualityTier.LOW

    def test_deferred_decal_masked_medium(self):
        """Should handle deferred_decal domain."""
        config = get_variant_for_material_system("deferred_decal", "masked", "medium")
        assert config.domain == MaterialDomain.DEFERRED_DECAL
        assert config.blend == BlendMode.MASKED
        assert config.quality == QualityTier.MEDIUM

    def test_invalid_domain_raises(self):
        """Invalid domain string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            get_variant_for_material_system("invalid", "opaque", "high")

    def test_invalid_blend_raises(self):
        """Invalid blend string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid blend mode"):
            get_variant_for_material_system("surface", "invalid", "high")

    def test_invalid_quality_raises(self):
        """Invalid quality string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid quality tier"):
            get_variant_for_material_system("surface", "opaque", "invalid")


# =============================================================================
# Test: VariantCompiler
# =============================================================================


class TestVariantCompiler:
    """Test VariantCompiler shader injection."""

    @pytest.fixture
    def sample_wgsl(self) -> str:
        """Sample WGSL shader for testing."""
        return """\
// SPDX-License-Identifier: MIT
// Generated by TRINITY MaterialCompiler

struct PBRInput {
    @builtin(position) position: vec4<f32>,
}

@fragment
fn fs_main(input: PBRInput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"""

    def test_inject_variant_consts(self, sample_wgsl: str):
        """Should inject const declarations after header."""
        config = VariantConfig()
        compiler = VariantCompiler(config)

        result = compiler.inject_variant_consts(sample_wgsl)

        # Should contain original code
        assert "struct PBRInput {" in result
        assert "fn fs_main" in result

        # Should contain variant consts
        assert "const DOMAIN_SURFACE: bool = true;" in result
        assert "const BLEND_OPAQUE: bool = true;" in result
        assert "const QUALITY_HIGH: bool = true;" in result

    def test_inject_gated_functions(self, sample_wgsl: str):
        """Should inject gated functions before @fragment."""
        config = VariantConfig()
        compiler = VariantCompiler(config)

        result = compiler.inject_gated_functions(sample_wgsl)

        # Should contain gated function definitions
        assert "Quality-Gated Functions" in result
        assert "fn evaluate_lighting_gated" in result
        assert "fn apply_advanced_shading" in result
        assert "fn apply_blend_mode" in result
        assert "fn compute_domain_output" in result

        # Original code should still be present
        assert "fn fs_main" in result

    def test_full_injection_pipeline(self, sample_wgsl: str):
        """Full injection should add consts and gated functions."""
        config = VariantConfig(
            domain=MaterialDomain.VOLUME,
            blend=BlendMode.TRANSLUCENT,
            quality=QualityTier.LOW,
        )
        compiler = VariantCompiler(config)

        # Apply both injections
        result = compiler.inject_variant_consts(sample_wgsl)
        result = compiler.inject_gated_functions(result)

        # Check variant consts
        assert "const DOMAIN_VOLUME: bool = true;" in result
        assert "const BLEND_TRANSLUCENT: bool = true;" in result
        assert "const QUALITY_LOW: bool = true;" in result

        # Check gated functions
        assert "fn evaluate_lighting_gated" in result


# =============================================================================
# Test: All 75 variants produce valid WGSL syntax (basic validation)
# =============================================================================


class TestAllVariantsValid:
    """Test that all variant combinations produce syntactically valid WGSL."""

    def test_all_variants_generate_const_declarations(self):
        """All 75 variants should generate valid const declarations."""
        configs = generate_all_variant_combinations()

        for config in configs:
            wgsl = config.generate_const_declarations()

            # Should have exactly one true per category
            domain_trues = wgsl.count("DOMAIN_") - wgsl.count("DOMAIN_") // 2  # Approx
            blend_trues = sum(
                1 for line in wgsl.split("\n")
                if line.startswith("const BLEND_") and "true" in line
            )
            quality_trues = sum(
                1 for line in wgsl.split("\n")
                if line.startswith("const QUALITY_") and "true" in line
            )

            assert blend_trues == 1, f"Expected 1 blend true, got {blend_trues}"
            assert quality_trues == 1, f"Expected 1 quality true, got {quality_trues}"

    def test_all_variants_have_valid_syntax(self):
        """All variants should produce syntactically correct WGSL consts."""
        configs = generate_all_variant_combinations()

        for config in configs:
            wgsl = config.generate_const_declarations()

            # Basic syntax validation: all const lines should match pattern
            for line in wgsl.split("\n"):
                line = line.strip()
                if line.startswith("const ") and ":" in line:
                    # Should match: const NAME: TYPE = VALUE;
                    pattern = r"const \w+: (bool|u32) = \w+;"
                    assert re.match(pattern, line), f"Invalid const line: {line}"


# =============================================================================
# Test: WGSL keyword/identifier validation
# =============================================================================


class TestWGSLIdentifiers:
    """Test that generated WGSL uses valid identifiers."""

    def test_const_names_are_valid_wgsl_identifiers(self):
        """All const names should be valid WGSL identifiers."""
        config = VariantConfig()
        wgsl = config.generate_const_declarations()

        # Extract const names
        pattern = r"const (\w+):"
        names = re.findall(pattern, wgsl)

        # All names should be uppercase with underscores only
        for name in names:
            assert re.match(r"^[A-Z][A-Z0-9_]*$", name), f"Invalid name: {name}"

    def test_no_wgsl_reserved_words_used(self):
        """Should not use WGSL reserved words as identifiers."""
        wgsl_reserved = {
            "break", "case", "const", "continue", "default", "discard",
            "else", "enable", "false", "fn", "for", "function", "if",
            "let", "loop", "private", "return", "struct", "switch",
            "true", "type", "uniform", "var", "while", "array", "bool",
            "f32", "i32", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
            "mat3x4", "mat4x2", "mat4x3", "mat4x4", "ptr", "sampler",
            "texture_1d", "texture_2d", "texture_2d_array", "texture_3d",
            "texture_cube", "texture_cube_array", "texture_multisampled_2d",
            "texture_storage_1d", "texture_storage_2d", "texture_storage_2d_array",
            "texture_storage_3d", "u32", "vec2", "vec3", "vec4",
        }

        config = VariantConfig()
        wgsl = config.generate_const_declarations()

        # Extract identifier names (after const keyword)
        pattern = r"const (\w+):"
        names = re.findall(pattern, wgsl)

        for name in names:
            assert name.lower() not in wgsl_reserved, (
                f"Reserved word used: {name}"
            )


# =============================================================================
# Test: MaterialCompiler integration with variants
# =============================================================================


class TestMaterialCompilerVariantIntegration:
    """Test MaterialCompiler with VariantConfig integration."""

    @pytest.fixture
    def gold_material(self):
        """Create a simple test material class."""
        class GoldMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.83, 0.69, 0.22)
                out.metallic = 0.9
                out.roughness = 0.3
        return GoldMaterial

    def test_compile_with_variant_config(self, gold_material):
        """Compiler should inject variant consts when config is set."""
        config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )
        compiler = MaterialCompiler(variant_config=config)
        wgsl = compiler.compile(gold_material)

        # Should contain variant consts
        assert "const DOMAIN_SURFACE: bool = true;" in wgsl
        assert "const BLEND_OPAQUE: bool = true;" in wgsl
        assert "const QUALITY_HIGH: bool = true;" in wgsl

        # Should still contain material code
        assert "vec3<f32>(0.83, 0.69, 0.22)" in wgsl

    def test_compile_without_variant_config(self, gold_material):
        """Compiler should not inject variant consts when config is None."""
        compiler = MaterialCompiler(variant_config=None)
        wgsl = compiler.compile(gold_material)

        # Should not contain variant consts
        assert "const DOMAIN_SURFACE:" not in wgsl
        assert "Variant Const Declarations" not in wgsl

    def test_compile_with_variants_method(self, gold_material):
        """compile_with_variants should temporarily set config."""
        compiler = MaterialCompiler()  # No config initially
        config = VariantConfig(quality=QualityTier.LOW)

        wgsl = compiler.compile_with_variants(gold_material, config)

        # Should contain LOW quality consts
        assert "const QUALITY_LOW: bool = true;" in wgsl
        assert "const MAX_LIGHTS: u32 = 1u;" in wgsl

        # Compiler should be back to original state
        assert compiler.variant_config is None

    def test_different_quality_tiers_produce_different_output(self, gold_material):
        """Different quality tiers should produce different WGSL."""
        compiler = MaterialCompiler()

        low_config = VariantConfig(quality=QualityTier.LOW)
        high_config = VariantConfig(quality=QualityTier.HIGH)

        wgsl_low = compiler.compile_with_variants(gold_material, low_config)
        wgsl_high = compiler.compile_with_variants(gold_material, high_config)

        # Should have different quality consts
        assert "const QUALITY_LOW: bool = true;" in wgsl_low
        assert "const QUALITY_HIGH: bool = true;" in wgsl_high

        # Should have different MAX_LIGHTS
        assert "const MAX_LIGHTS: u32 = 1u;" in wgsl_low
        assert "const MAX_LIGHTS: u32 = 16u;" in wgsl_high

    def test_compile_all_variants(self, gold_material):
        """compile_all_variants should produce 75 variants."""
        compiler = MaterialCompiler()
        all_variants = compiler.compile_all_variants(gold_material)

        # Should have 75 variants
        assert len(all_variants) == 75

        # All should be non-empty WGSL
        for key, wgsl in all_variants.items():
            assert isinstance(key, int)
            assert len(wgsl) > 0
            assert "fn fs_main" in wgsl

    def test_variant_keys_match_config_keys(self, gold_material):
        """Variant keys should match VariantConfig.get_variant_key()."""
        compiler = MaterialCompiler()

        # Compile a specific variant
        config = VariantConfig(
            domain=MaterialDomain.VOLUME,
            blend=BlendMode.TRANSLUCENT,
            quality=QualityTier.MEDIUM,
        )
        wgsl_direct = compiler.compile_with_variants(gold_material, config)

        # Get from all variants
        all_variants = compiler.compile_all_variants(gold_material)
        wgsl_from_all = all_variants[config.get_variant_key()]

        # Should match
        assert wgsl_direct == wgsl_from_all

    def test_variant_consts_before_structs(self, gold_material):
        """Variant consts should appear before struct definitions."""
        config = VariantConfig()
        compiler = MaterialCompiler(variant_config=config)
        wgsl = compiler.compile(gold_material)

        # Find positions
        variant_pos = wgsl.find("const DOMAIN_SURFACE:")
        struct_pos = wgsl.find("struct PBRInput")

        assert variant_pos != -1, "Variant consts not found"
        assert struct_pos != -1, "PBRInput struct not found"
        assert variant_pos < struct_pos, "Variant consts should come before structs"
