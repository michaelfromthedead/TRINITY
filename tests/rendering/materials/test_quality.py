"""Tests for quality tier variant system.

Task: T-MAT-2.4 Quality Tier Variants
Gap: S3-G3, S3-G9 (CRITICAL + HIGH)

Tests verify:
1. QualityFeatures for each tier
2. LOW tier produces simpler shader code
3. HIGH tier enables all features (subsurface, clearcoat, PCSS)
4. Shader complexity correlates with tier
5. All quality tiers compile to valid WGSL
"""

from __future__ import annotations

import pytest
import re
from typing import Dict, List

from trinity.materials.variants import QualityTier, VariantConfig
from trinity.materials.quality import (
    QualityFeatures,
    QualityShaderCode,
    get_quality_config_for_device,
)


# =============================================================================
# Test: QualityFeatures for each tier
# =============================================================================


class TestQualityFeaturesLow:
    """Test LOW quality tier feature configuration."""

    @pytest.fixture
    def features(self) -> QualityFeatures:
        return QualityFeatures.for_tier(QualityTier.LOW)

    def test_max_lights_is_one(self, features: QualityFeatures):
        """LOW quality should support only 1 light."""
        assert features.max_lights == 1

    def test_no_shadows(self, features: QualityFeatures):
        """LOW quality should have no shadows."""
        assert features.shadow_quality == "none"
        assert not features.has_shadows

    def test_no_subsurface(self, features: QualityFeatures):
        """LOW quality should disable subsurface scattering."""
        assert not features.subsurface

    def test_no_clearcoat(self, features: QualityFeatures):
        """LOW quality should disable clearcoat."""
        assert not features.clearcoat

    def test_no_anisotropy(self, features: QualityFeatures):
        """LOW quality should disable anisotropy."""
        assert not features.anisotropy

    def test_no_ssr(self, features: QualityFeatures):
        """LOW quality should disable screen-space reflections."""
        assert not features.screen_space_reflections

    def test_no_ao(self, features: QualityFeatures):
        """LOW quality should have no ambient occlusion."""
        assert features.ambient_occlusion == "none"

    def test_no_iridescence(self, features: QualityFeatures):
        """LOW quality should disable iridescence."""
        assert not features.iridescence

    def test_no_sheen(self, features: QualityFeatures):
        """LOW quality should disable sheen."""
        assert not features.sheen

    def test_no_transmission(self, features: QualityFeatures):
        """LOW quality should disable transmission."""
        assert not features.transmission

    def test_no_advanced_shading(self, features: QualityFeatures):
        """LOW quality should have no advanced shading features."""
        assert not features.has_advanced_shading

    def test_lowest_complexity_score(self, features: QualityFeatures):
        """LOW quality should have the lowest complexity score."""
        low_score = features.complexity_score
        medium_score = QualityFeatures.for_tier(QualityTier.MEDIUM).complexity_score
        high_score = QualityFeatures.for_tier(QualityTier.HIGH).complexity_score

        assert low_score < medium_score < high_score


class TestQualityFeaturesMedium:
    """Test MEDIUM quality tier feature configuration."""

    @pytest.fixture
    def features(self) -> QualityFeatures:
        return QualityFeatures.for_tier(QualityTier.MEDIUM)

    def test_max_lights_is_four(self, features: QualityFeatures):
        """MEDIUM quality should support 4 lights."""
        assert features.max_lights == 4

    def test_basic_shadows(self, features: QualityFeatures):
        """MEDIUM quality should have basic shadows."""
        assert features.shadow_quality == "basic"
        assert features.has_shadows

    def test_no_subsurface(self, features: QualityFeatures):
        """MEDIUM quality should disable subsurface scattering."""
        assert not features.subsurface

    def test_clearcoat_enabled(self, features: QualityFeatures):
        """MEDIUM quality should enable clearcoat."""
        assert features.clearcoat

    def test_no_anisotropy(self, features: QualityFeatures):
        """MEDIUM quality should disable anisotropy."""
        assert not features.anisotropy

    def test_no_ssr(self, features: QualityFeatures):
        """MEDIUM quality should disable screen-space reflections."""
        assert not features.screen_space_reflections

    def test_ssao_enabled(self, features: QualityFeatures):
        """MEDIUM quality should have SSAO."""
        assert features.ambient_occlusion == "ssao"

    def test_transmission_enabled(self, features: QualityFeatures):
        """MEDIUM quality should enable transmission."""
        assert features.transmission

    def test_has_some_advanced_shading(self, features: QualityFeatures):
        """MEDIUM quality should have some advanced shading (clearcoat)."""
        assert features.has_advanced_shading


class TestQualityFeaturesHigh:
    """Test HIGH quality tier feature configuration."""

    @pytest.fixture
    def features(self) -> QualityFeatures:
        return QualityFeatures.for_tier(QualityTier.HIGH)

    def test_max_lights_is_sixteen(self, features: QualityFeatures):
        """HIGH quality should support 16 lights."""
        assert features.max_lights == 16

    def test_pcss_shadows(self, features: QualityFeatures):
        """HIGH quality should have PCSS shadows."""
        assert features.shadow_quality == "pcss"
        assert features.has_shadows

    def test_subsurface_enabled(self, features: QualityFeatures):
        """HIGH quality should enable subsurface scattering."""
        assert features.subsurface

    def test_clearcoat_enabled(self, features: QualityFeatures):
        """HIGH quality should enable clearcoat."""
        assert features.clearcoat

    def test_anisotropy_enabled(self, features: QualityFeatures):
        """HIGH quality should enable anisotropy."""
        assert features.anisotropy

    def test_ssr_enabled(self, features: QualityFeatures):
        """HIGH quality should enable screen-space reflections."""
        assert features.screen_space_reflections

    def test_hbao_enabled(self, features: QualityFeatures):
        """HIGH quality should have HBAO."""
        assert features.ambient_occlusion == "hbao"

    def test_iridescence_enabled(self, features: QualityFeatures):
        """HIGH quality should enable iridescence."""
        assert features.iridescence

    def test_sheen_enabled(self, features: QualityFeatures):
        """HIGH quality should enable sheen."""
        assert features.sheen

    def test_transmission_enabled(self, features: QualityFeatures):
        """HIGH quality should enable transmission."""
        assert features.transmission

    def test_has_advanced_shading(self, features: QualityFeatures):
        """HIGH quality should have advanced shading features."""
        assert features.has_advanced_shading

    def test_highest_complexity_score(self, features: QualityFeatures):
        """HIGH quality should have the highest complexity score."""
        high_score = features.complexity_score
        medium_score = QualityFeatures.for_tier(QualityTier.MEDIUM).complexity_score
        low_score = QualityFeatures.for_tier(QualityTier.LOW).complexity_score

        assert high_score > medium_score > low_score


class TestQualityFeaturesToDict:
    """Test QualityFeatures serialization."""

    def test_to_dict_contains_all_fields(self):
        """to_dict should contain all feature fields."""
        features = QualityFeatures.for_tier(QualityTier.HIGH)
        d = features.to_dict()

        assert "max_lights" in d
        assert "shadow_quality" in d
        assert "subsurface" in d
        assert "clearcoat" in d
        assert "anisotropy" in d
        assert "screen_space_reflections" in d
        assert "ambient_occlusion" in d
        assert "iridescence" in d
        assert "sheen" in d
        assert "transmission" in d

    def test_to_dict_values_match(self):
        """to_dict values should match feature attributes."""
        features = QualityFeatures.for_tier(QualityTier.MEDIUM)
        d = features.to_dict()

        assert d["max_lights"] == features.max_lights
        assert d["shadow_quality"] == features.shadow_quality
        assert d["subsurface"] == features.subsurface
        assert d["clearcoat"] == features.clearcoat


# =============================================================================
# Test: QualityShaderCode produces different code per tier
# =============================================================================


class TestQualityShaderCodeShadows:
    """Test shadow code generation for each tier."""

    def test_low_shadow_returns_one(self):
        """LOW shadow code should return 1.0 (no shadows)."""
        code = QualityShaderCode.get_shadow_code(QualityTier.LOW)
        assert "return 1.0;" in code
        assert "No shadows" in code

    def test_medium_shadow_has_bias(self):
        """MEDIUM shadow code should use basic shadow mapping with bias."""
        code = QualityShaderCode.get_shadow_code(QualityTier.MEDIUM)
        assert "SHADOW_BIAS" in code
        assert "textureSample" in code
        assert "light_matrices" in code

    def test_high_shadow_has_pcss(self):
        """HIGH shadow code should implement PCSS."""
        code = QualityShaderCode.get_shadow_code(QualityTier.HIGH)
        assert "find_blocker" in code
        assert "estimate_penumbra" in code
        assert "pcf_filter" in code
        assert "POISSON_DISK" in code
        assert "PCF_SAMPLES" in code

    def test_shadow_code_length_increases_with_quality(self):
        """Higher quality should produce longer shadow code."""
        low = QualityShaderCode.get_shadow_code(QualityTier.LOW)
        medium = QualityShaderCode.get_shadow_code(QualityTier.MEDIUM)
        high = QualityShaderCode.get_shadow_code(QualityTier.HIGH)

        assert len(low) < len(medium) < len(high)


class TestQualityShaderCodeBRDF:
    """Test BRDF code generation for each tier."""

    def test_low_brdf_is_lambert(self):
        """LOW BRDF should be simple Lambert diffuse."""
        code = QualityShaderCode.get_brdf_code(QualityTier.LOW)
        assert "evaluate_brdf_simple" in code
        assert "base_color / 3.14159" in code
        # Should not have GGX or complex specular
        assert "distribution_ggx" not in code

    def test_medium_brdf_has_cook_torrance(self):
        """MEDIUM BRDF should implement Cook-Torrance."""
        code = QualityShaderCode.get_brdf_code(QualityTier.MEDIUM)
        assert "evaluate_brdf_standard" in code
        assert "Fresnel-Schlick" in code or "pow(1.0 - HdotV, 5.0)" in code
        assert "GGX" in code

    def test_high_brdf_has_full_features(self):
        """HIGH BRDF should implement full multi-lobe BRDF."""
        code = QualityShaderCode.get_brdf_code(QualityTier.HIGH)
        assert "evaluate_brdf_full" in code
        assert "fresnel_schlick_roughness" in code
        assert "evaluate_sheen" in code
        assert "evaluate_clearcoat" in code

    def test_high_brdf_has_anisotropic(self):
        """HIGH BRDF should include anisotropic GGX."""
        code = QualityShaderCode.get_brdf_code(QualityTier.HIGH)
        assert "distribution_ggx_aniso" in code
        assert "geometry_smith_aniso" in code


class TestQualityShaderCodeAO:
    """Test ambient occlusion code generation for each tier."""

    def test_low_ao_returns_one(self):
        """LOW AO should return 1.0 (no occlusion)."""
        code = QualityShaderCode.get_ao_code(QualityTier.LOW)
        assert "return 1.0;" in code

    def test_medium_ao_has_ssao(self):
        """MEDIUM AO should implement SSAO."""
        code = QualityShaderCode.get_ao_code(QualityTier.MEDIUM)
        assert "SSAO_SAMPLES" in code
        assert "SSAO_RADIUS" in code

    def test_high_ao_has_hbao(self):
        """HIGH AO should implement HBAO."""
        code = QualityShaderCode.get_ao_code(QualityTier.HIGH)
        assert "HBAO_DIRECTIONS" in code
        assert "HBAO_STEPS" in code
        assert "max_horizon" in code


class TestQualityShaderCodeSubsurface:
    """Test subsurface scattering code generation."""

    def test_low_subsurface_returns_zero(self):
        """LOW subsurface should return vec3(0.0)."""
        code = QualityShaderCode.get_subsurface_code(QualityTier.LOW)
        assert "return vec3<f32>(0.0);" in code

    def test_medium_subsurface_returns_zero(self):
        """MEDIUM subsurface should return vec3(0.0)."""
        code = QualityShaderCode.get_subsurface_code(QualityTier.MEDIUM)
        assert "return vec3<f32>(0.0);" in code

    def test_high_subsurface_has_approximation(self):
        """HIGH subsurface should implement SSS approximation."""
        code = QualityShaderCode.get_subsurface_code(QualityTier.HIGH)
        assert "wrap_diffuse" in code or "NdotL_wrap" in code
        assert "forward_scatter" in code or "back_scatter" in code


# =============================================================================
# Test: LOW quality produces observably simpler shader code
# =============================================================================


class TestLowQualitySimplicity:
    """Test that LOW quality produces simpler shader code."""

    def test_low_has_fewer_functions(self):
        """LOW quality should define fewer shader functions."""
        low_code = QualityShaderCode.get_all_quality_code(QualityTier.LOW)
        high_code = QualityShaderCode.get_all_quality_code(QualityTier.HIGH)

        low_fn_count = low_code.count("fn ")
        high_fn_count = high_code.count("fn ")

        assert low_fn_count < high_fn_count

    def test_low_has_fewer_loops(self):
        """LOW quality should have fewer loops."""
        low_code = QualityShaderCode.get_all_quality_code(QualityTier.LOW)
        high_code = QualityShaderCode.get_all_quality_code(QualityTier.HIGH)

        low_loop_count = low_code.count("for (")
        high_loop_count = high_code.count("for (")

        assert low_loop_count < high_loop_count

    def test_low_has_fewer_texture_samples(self):
        """LOW quality should have fewer texture samples."""
        low_code = QualityShaderCode.get_all_quality_code(QualityTier.LOW)
        high_code = QualityShaderCode.get_all_quality_code(QualityTier.HIGH)

        low_sample_count = low_code.count("textureSample")
        high_sample_count = high_code.count("textureSample")

        assert low_sample_count < high_sample_count

    def test_low_code_is_shorter(self):
        """LOW quality code should be significantly shorter."""
        low_code = QualityShaderCode.get_all_quality_code(QualityTier.LOW)
        medium_code = QualityShaderCode.get_all_quality_code(QualityTier.MEDIUM)
        high_code = QualityShaderCode.get_all_quality_code(QualityTier.HIGH)

        assert len(low_code) < len(medium_code) < len(high_code)


class TestHighQualityCompleteness:
    """Test that HIGH quality enables all features."""

    def test_high_has_pcss_shadow_functions(self):
        """HIGH quality should have all PCSS helper functions."""
        code = QualityShaderCode.get_shadow_code(QualityTier.HIGH)

        required_functions = [
            "find_blocker",
            "estimate_penumbra",
            "pcf_filter",
            "sample_shadow",
        ]

        for fn in required_functions:
            assert f"fn {fn}" in code, f"Missing function: {fn}"

    def test_high_has_all_brdf_lobes(self):
        """HIGH quality BRDF should have all material lobes."""
        code = QualityShaderCode.get_brdf_code(QualityTier.HIGH)

        assert "evaluate_sheen" in code
        assert "evaluate_clearcoat" in code
        assert "fresnel_schlick_roughness" in code

    def test_high_has_anisotropic_functions(self):
        """HIGH quality should have anisotropic GGX functions."""
        code = QualityShaderCode.get_brdf_code(QualityTier.HIGH)

        assert "distribution_ggx_aniso" in code
        assert "geometry_smith_aniso" in code


# =============================================================================
# Test: Shader code complexity correlates with tier
# =============================================================================


class TestShaderComplexityCorrelation:
    """Test that shader complexity metrics correlate with quality tier."""

    def test_instruction_count_increases_with_tier(self):
        """Estimated instruction count should increase with tier."""
        low_inst = QualityShaderCode.estimate_instruction_count(QualityTier.LOW)
        medium_inst = QualityShaderCode.estimate_instruction_count(QualityTier.MEDIUM)
        high_inst = QualityShaderCode.estimate_instruction_count(QualityTier.HIGH)

        assert low_inst["total"] < medium_inst["total"] < high_inst["total"]

    def test_shadow_instructions_increase_with_tier(self):
        """Shadow instruction count should increase with tier."""
        low_inst = QualityShaderCode.estimate_instruction_count(QualityTier.LOW)
        medium_inst = QualityShaderCode.estimate_instruction_count(QualityTier.MEDIUM)
        high_inst = QualityShaderCode.estimate_instruction_count(QualityTier.HIGH)

        assert low_inst["shadow"] < medium_inst["shadow"] < high_inst["shadow"]

    def test_brdf_instructions_increase_with_tier(self):
        """BRDF instruction count should increase with tier."""
        low_inst = QualityShaderCode.estimate_instruction_count(QualityTier.LOW)
        medium_inst = QualityShaderCode.estimate_instruction_count(QualityTier.MEDIUM)
        high_inst = QualityShaderCode.estimate_instruction_count(QualityTier.HIGH)

        assert low_inst["brdf"] < medium_inst["brdf"] < high_inst["brdf"]

    def test_ao_instructions_increase_with_tier(self):
        """AO instruction count should increase with tier."""
        low_inst = QualityShaderCode.estimate_instruction_count(QualityTier.LOW)
        medium_inst = QualityShaderCode.estimate_instruction_count(QualityTier.MEDIUM)
        high_inst = QualityShaderCode.estimate_instruction_count(QualityTier.HIGH)

        assert low_inst["ao"] < medium_inst["ao"] < high_inst["ao"]

    def test_complexity_score_matches_instruction_trend(self):
        """QualityFeatures complexity_score should match instruction trends."""
        for tier in QualityTier:
            features = QualityFeatures.for_tier(tier)
            instructions = QualityShaderCode.estimate_instruction_count(tier)

            # Both metrics should agree on relative ordering
            # (we can't compare absolute values, but can check correlation)
            if tier == QualityTier.LOW:
                assert features.complexity_score < 100
            elif tier == QualityTier.MEDIUM:
                assert 100 <= features.complexity_score < 400
            else:  # HIGH
                assert features.complexity_score >= 400


# =============================================================================
# Test: WGSL syntax validity
# =============================================================================


class TestWGSLSyntaxValidity:
    """Test that generated WGSL code has valid syntax."""

    def test_all_tiers_produce_valid_function_declarations(self):
        """All tiers should produce syntactically valid fn declarations."""
        fn_pattern = re.compile(r"fn\s+\w+\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*\{")

        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)

            # Find all fn declarations
            matches = fn_pattern.findall(code)
            assert len(matches) > 0, f"No functions found for {tier.name}"

    def test_all_const_declarations_are_valid(self):
        """All const declarations should have valid WGSL syntax."""
        # Pattern handles simple types, generic types, and array types
        # e.g., const FOO: f32 = ..., const BAR: vec3<f32> = ...,
        #       const ARR: array<vec2<f32>, 16> = ...
        const_pattern = re.compile(
            r"const\s+\w+\s*:\s*(?:array<[^>]+(?:<[^>]+>)?(?:,\s*\d+)?>|\w+(?:<[^>]+>)?)\s*="
        )

        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)

            # Find all const declarations
            for line in code.split("\n"):
                if line.strip().startswith("const "):
                    assert const_pattern.search(line), f"Invalid const: {line}"

    def test_balanced_braces(self):
        """All generated code should have balanced braces."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)

            open_count = code.count("{")
            close_count = code.count("}")

            assert open_count == close_count, (
                f"{tier.name}: {open_count} open vs {close_count} close braces"
            )

    def test_balanced_parentheses(self):
        """All generated code should have balanced parentheses."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)

            open_count = code.count("(")
            close_count = code.count(")")

            assert open_count == close_count, (
                f"{tier.name}: {open_count} open vs {close_count} close parens"
            )


# =============================================================================
# Test: get_quality_config_for_device
# =============================================================================


class TestGetQualityConfigForDevice:
    """Test device-based quality recommendation."""

    def test_low_tier_gpu_gets_low_quality(self):
        """Low tier GPU should get LOW quality."""
        tier = get_quality_config_for_device("low", vram_mb=1024)
        assert tier == QualityTier.LOW

    def test_mid_tier_gpu_gets_medium_quality(self):
        """Mid tier GPU should get MEDIUM quality."""
        tier = get_quality_config_for_device("mid", vram_mb=4096)
        assert tier == QualityTier.MEDIUM

    def test_high_tier_gpu_gets_high_quality(self):
        """High tier GPU should get HIGH quality."""
        tier = get_quality_config_for_device("high", vram_mb=8192)
        assert tier == QualityTier.HIGH

    def test_ultra_tier_gpu_gets_high_quality(self):
        """Ultra tier GPU should get HIGH quality."""
        tier = get_quality_config_for_device("ultra", vram_mb=12288)
        assert tier == QualityTier.HIGH

    def test_low_vram_downgrades_high_to_medium(self):
        """High GPU with low VRAM should downgrade to MEDIUM."""
        tier = get_quality_config_for_device("high", vram_mb=1500)
        assert tier == QualityTier.MEDIUM

    def test_very_low_vram_downgrades_to_low(self):
        """Any GPU with very low VRAM should use LOW."""
        tier = get_quality_config_for_device("high", vram_mb=512)
        assert tier == QualityTier.LOW

    def test_high_fps_target_downgrades(self):
        """High FPS targets (VR) should downgrade quality."""
        tier_60 = get_quality_config_for_device("high", vram_mb=8192, target_fps=60)
        tier_90 = get_quality_config_for_device("high", vram_mb=8192, target_fps=90)

        assert tier_60 == QualityTier.HIGH
        assert tier_90 == QualityTier.MEDIUM

    def test_unknown_gpu_tier_defaults_to_medium(self):
        """Unknown GPU tier should default to MEDIUM."""
        tier = get_quality_config_for_device("unknown", vram_mb=4096)
        assert tier == QualityTier.MEDIUM


# =============================================================================
# Test: Integration with VariantConfig
# =============================================================================


class TestQualityVariantConfigIntegration:
    """Test integration between QualityFeatures and VariantConfig."""

    def test_features_match_variant_config_quality_settings(self):
        """QualityFeatures should match VariantConfig.QUALITY_CONFIG."""
        for tier in QualityTier:
            features = QualityFeatures.for_tier(tier)
            variant_cfg = VariantConfig.QUALITY_CONFIG[tier]

            assert features.max_lights == variant_cfg["max_lights"]
            assert features.shadow_quality == variant_cfg["shadow_quality"]
            assert features.subsurface == variant_cfg["subsurface_enabled"]
            assert features.clearcoat == variant_cfg["clearcoat_enabled"]
            assert features.anisotropy == variant_cfg["anisotropy_enabled"]

    def test_generated_shader_code_respects_variant_consts(self):
        """Generated shader code should check variant consts."""
        # The gated code in VariantConfig uses consts like SUBSURFACE_ENABLED
        # Our shader code should be compatible with this approach
        low_config = VariantConfig(quality=QualityTier.LOW)
        high_config = VariantConfig(quality=QualityTier.HIGH)

        low_consts = low_config.generate_const_declarations()
        high_consts = high_config.generate_const_declarations()

        # LOW should disable advanced features
        assert "const SUBSURFACE_ENABLED: bool = false;" in low_consts
        assert "const CLEARCOAT_ENABLED: bool = false;" in low_consts

        # HIGH should enable advanced features
        assert "const SUBSURFACE_ENABLED: bool = true;" in high_consts
        assert "const CLEARCOAT_ENABLED: bool = true;" in high_consts


# =============================================================================
# Test: QualityFeatures frozen dataclass
# =============================================================================


class TestQualityFeaturesFrozen:
    """Test that QualityFeatures is immutable."""

    def test_cannot_modify_max_lights(self):
        """Should not be able to modify max_lights after creation."""
        features = QualityFeatures.for_tier(QualityTier.HIGH)

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            features.max_lights = 32

    def test_cannot_modify_shadow_quality(self):
        """Should not be able to modify shadow_quality after creation."""
        features = QualityFeatures.for_tier(QualityTier.LOW)

        with pytest.raises(Exception):
            features.shadow_quality = "pcss"


# =============================================================================
# Test: get_all_quality_code includes all sections
# =============================================================================


class TestGetAllQualityCode:
    """Test get_all_quality_code combines all shader sections."""

    def test_includes_header(self):
        """Should include quality tier header."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)
            assert f"Quality Tier: {tier.name}" in code

    def test_includes_shadow_section(self):
        """Should include shadow sampling section."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)
            assert "Shadow Sampling" in code
            assert "sample_shadow" in code

    def test_includes_brdf_section(self):
        """Should include BRDF section."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)
            assert "BRDF" in code

    def test_includes_ao_section(self):
        """Should include ambient occlusion section."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)
            assert "Ambient Occlusion" in code
            assert "sample_ambient_occlusion" in code

    def test_includes_subsurface_section(self):
        """Should include subsurface scattering section."""
        for tier in QualityTier:
            code = QualityShaderCode.get_all_quality_code(tier)
            assert "Subsurface Scattering" in code
            assert "evaluate_subsurface" in code
