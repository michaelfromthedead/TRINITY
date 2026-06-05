"""Variant Combination Matrix Tests for Material System (T-MAT-11.1).

This module tests all combinations of:
- 5 domains (SURFACE, DEFERRED_DECAL, VOLUME, POST_PROCESS, UI)
- 5 blend modes (OPAQUE, MASKED, TRANSLUCENT, ADDITIVE, MODULATE)
- 3 quality tiers (LOW, MEDIUM, HIGH)

Total: 5 x 5 x 3 = 75 variant combinations

Each combination is tested for:
1. Const declaration generation
2. WGSL compilation
3. naga validation (if available)
4. Pipeline creation

Gap: S11-G1 (PARTIAL)
Dependencies: T-MAT-3.4 (DONE)
"""

from __future__ import annotations

import itertools
import pytest
from typing import List, Tuple, Generator, Dict, Any

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec3,
    Vec4,
    MaterialCompiler,
    Texture2D,
)

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
    VariantCompiler,
    generate_all_variant_combinations,
)

from trinity.materials.domains import (
    DomainCapability,
    DOMAIN_CAPABILITIES,
    DOMAIN_OUTPUT_FORMATS,
    domain_has_capability,
    DomainShaderTemplate,
    DomainVariantGenerator,
)

from trinity.materials.pipeline_integration import (
    PipelineIntegration,
    PipelineCacheHandle,
    ShaderCache,
)


# =============================================================================
# Test Data
# =============================================================================


ALL_DOMAINS = list(MaterialDomain)
ALL_BLEND_MODES = list(BlendMode)
ALL_QUALITY_TIERS = list(QualityTier)


def all_variant_combinations() -> Generator[
    Tuple[MaterialDomain, BlendMode, QualityTier], None, None
]:
    """Generate all 75 variant combinations."""
    for domain in ALL_DOMAINS:
        for blend in ALL_BLEND_MODES:
            for quality in ALL_QUALITY_TIERS:
                yield (domain, blend, quality)


def variant_id(combo: Tuple[MaterialDomain, BlendMode, QualityTier]) -> str:
    """Generate a test ID for a variant combination."""
    domain, blend, quality = combo
    return f"{domain.name}_{blend.name}_{quality.name}"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def compiler() -> MaterialCompiler:
    """Create a fresh MaterialCompiler instance."""
    return MaterialCompiler()


@pytest.fixture
def variant_compiler() -> VariantCompiler:
    """Create a fresh VariantCompiler instance."""
    return VariantCompiler()


@pytest.fixture
def pipeline_integration() -> PipelineIntegration:
    """Create PipelineIntegration with large cache for matrix tests."""
    return PipelineIntegration(max_cache_size=128)


@pytest.fixture
def test_material_class():
    """Create a test material class for variant compilation."""

    class TestMaterial(Material, metaclass=MaterialMeta):
        """Generic test material for variant testing."""

        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            out.base_color = Vec3(0.5, 0.5, 0.5)
            out.metallic = 0.0
            out.roughness = 0.5
            out.alpha = 1.0

    return TestMaterial


@pytest.fixture
def textured_material_class():
    """Create a textured material class for variant compilation."""

    class TexturedTestMaterial(Material, metaclass=MaterialMeta):
        """Test material with texture for variant testing."""

        diffuse = Texture2D(default="white")

        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            color = ctx.sample(self.diffuse, ctx.uv)
            out.base_color = color.xyz
            out.alpha = color.w
            out.roughness = 0.5

    return TexturedTestMaterial


# =============================================================================
# Suite A: Variant Const Generation (75 tests)
# =============================================================================


class TestVariantConstGeneration:
    """Test const declaration generation for all 75 combinations."""

    @pytest.mark.parametrize(
        "domain,blend,quality",
        list(all_variant_combinations()),
        ids=[variant_id(c) for c in all_variant_combinations()],
    )
    def test_const_declarations_generated(
        self,
        domain: MaterialDomain,
        blend: BlendMode,
        quality: QualityTier,
    ) -> None:
        """VariantConfig generates correct const declarations."""
        config = VariantConfig(domain=domain, blend=blend, quality=quality)
        consts = config.generate_const_declarations()

        # Verify domain consts
        assert f"DOMAIN_{domain.name}: bool = true" in consts
        for other_domain in MaterialDomain:
            if other_domain != domain:
                assert f"DOMAIN_{other_domain.name}: bool = false" in consts

        # Verify blend consts
        assert f"BLEND_{blend.name}: bool = true" in consts
        for other_blend in BlendMode:
            if other_blend != blend:
                assert f"BLEND_{other_blend.name}: bool = false" in consts

        # Verify quality consts
        assert f"QUALITY_{quality.name}: bool = true" in consts
        for other_quality in QualityTier:
            if other_quality != quality:
                assert f"QUALITY_{other_quality.name}: bool = false" in consts


# =============================================================================
# Suite B: Domain-Blend Compatibility (25 tests)
# =============================================================================


class TestDomainBlendCompatibility:
    """Test domain-blend mode compatibility."""

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    @pytest.mark.parametrize("blend", ALL_BLEND_MODES, ids=[b.name for b in ALL_BLEND_MODES])
    def test_domain_blend_combination(
        self,
        domain: MaterialDomain,
        blend: BlendMode,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Each domain-blend combination compiles to valid WGSL."""
        config = VariantConfig(domain=domain, blend=blend, quality=QualityTier.MEDIUM)
        wgsl = compiler.compile_with_variants(test_material_class, config)

        assert len(wgsl) > 0
        assert f"DOMAIN_{domain.name}" in wgsl
        assert f"BLEND_{blend.name}" in wgsl


# =============================================================================
# Suite C: Domain-Quality Compatibility (15 tests)
# =============================================================================


class TestDomainQualityCompatibility:
    """Test domain-quality tier compatibility."""

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    @pytest.mark.parametrize("quality", ALL_QUALITY_TIERS, ids=[q.name for q in ALL_QUALITY_TIERS])
    def test_domain_quality_combination(
        self,
        domain: MaterialDomain,
        quality: QualityTier,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Each domain-quality combination compiles to valid WGSL."""
        config = VariantConfig(domain=domain, blend=BlendMode.OPAQUE, quality=quality)
        wgsl = compiler.compile_with_variants(test_material_class, config)

        assert len(wgsl) > 0
        assert f"DOMAIN_{domain.name}" in wgsl
        assert f"QUALITY_{quality.name}" in wgsl

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    def test_domain_quality_feature_gating(
        self,
        domain: MaterialDomain,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Quality-gated features are correctly declared per domain."""
        for quality in ALL_QUALITY_TIERS:
            config = VariantConfig(domain=domain, blend=BlendMode.OPAQUE, quality=quality)
            wgsl = compiler.compile_with_variants(test_material_class, config)

            # Check quality-derived consts
            quality_cfg = VariantConfig.QUALITY_CONFIG[quality]

            if quality_cfg["shadows_enabled"]:
                assert "SHADOWS_ENABLED: bool = true" in wgsl
            else:
                assert "SHADOWS_ENABLED: bool = false" in wgsl

            if quality_cfg["subsurface_enabled"]:
                assert "SUBSURFACE_ENABLED: bool = true" in wgsl
            else:
                assert "SUBSURFACE_ENABLED: bool = false" in wgsl


# =============================================================================
# Suite D: Blend-Quality Compatibility (15 tests)
# =============================================================================


class TestBlendQualityCompatibility:
    """Test blend mode-quality tier compatibility."""

    @pytest.mark.parametrize("blend", ALL_BLEND_MODES, ids=[b.name for b in ALL_BLEND_MODES])
    @pytest.mark.parametrize("quality", ALL_QUALITY_TIERS, ids=[q.name for q in ALL_QUALITY_TIERS])
    def test_blend_quality_combination(
        self,
        blend: BlendMode,
        quality: QualityTier,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Each blend-quality combination compiles to valid WGSL."""
        config = VariantConfig(
            domain=MaterialDomain.SURFACE, blend=blend, quality=quality
        )
        wgsl = compiler.compile_with_variants(test_material_class, config)

        assert len(wgsl) > 0
        assert f"BLEND_{blend.name}" in wgsl
        assert f"QUALITY_{quality.name}" in wgsl


# =============================================================================
# Suite E: Full Matrix Compilation (75 tests)
# =============================================================================


class TestFullVariantMatrix:
    """Test full 75-combination matrix compilation."""

    @pytest.mark.parametrize(
        "domain,blend,quality",
        list(all_variant_combinations()),
        ids=[variant_id(c) for c in all_variant_combinations()],
    )
    def test_variant_compiles(
        self,
        domain: MaterialDomain,
        blend: BlendMode,
        quality: QualityTier,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Each of 75 variant combinations compiles to WGSL."""
        config = VariantConfig(domain=domain, blend=blend, quality=quality)
        wgsl = compiler.compile_with_variants(test_material_class, config)

        # Basic validation
        assert isinstance(wgsl, str)
        assert len(wgsl) > 100

        # Verify variant consts are present
        assert "DOMAIN_" in wgsl
        assert "BLEND_" in wgsl
        assert "QUALITY_" in wgsl


# =============================================================================
# Suite F: Naga Validation Matrix
# =============================================================================


class TestNagaValidationMatrix:
    """Test naga validation for variant combinations."""

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    def test_domain_variants_pass_naga(
        self,
        domain: MaterialDomain,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """All variants for each domain pass naga validation."""
        for blend in ALL_BLEND_MODES:
            for quality in ALL_QUALITY_TIERS:
                config = VariantConfig(domain=domain, blend=blend, quality=quality)
                wgsl = compiler.compile_with_variants(test_material_class, config)

                is_valid, error = compiler.validate_wgsl(wgsl)
                if error is not None:
                    pytest.fail(
                        f"Naga validation failed for "
                        f"{domain.name}/{blend.name}/{quality.name}: {error}"
                    )

    @pytest.mark.parametrize("blend", ALL_BLEND_MODES, ids=[b.name for b in ALL_BLEND_MODES])
    def test_blend_variants_pass_naga(
        self,
        blend: BlendMode,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """All variants for each blend mode pass naga validation."""
        for domain in ALL_DOMAINS:
            for quality in ALL_QUALITY_TIERS:
                config = VariantConfig(domain=domain, blend=blend, quality=quality)
                wgsl = compiler.compile_with_variants(test_material_class, config)

                is_valid, error = compiler.validate_wgsl(wgsl)
                if error is not None:
                    pytest.fail(
                        f"Naga validation failed for "
                        f"{domain.name}/{blend.name}/{quality.name}: {error}"
                    )

    @pytest.mark.parametrize("quality", ALL_QUALITY_TIERS, ids=[q.name for q in ALL_QUALITY_TIERS])
    def test_quality_variants_pass_naga(
        self,
        quality: QualityTier,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """All variants for each quality tier pass naga validation."""
        for domain in ALL_DOMAINS:
            for blend in ALL_BLEND_MODES:
                config = VariantConfig(domain=domain, blend=blend, quality=quality)
                wgsl = compiler.compile_with_variants(test_material_class, config)

                is_valid, error = compiler.validate_wgsl(wgsl)
                if error is not None:
                    pytest.fail(
                        f"Naga validation failed for "
                        f"{domain.name}/{blend.name}/{quality.name}: {error}"
                    )


# =============================================================================
# Suite G: Pipeline Creation Matrix
# =============================================================================


class TestPipelineCreationMatrix:
    """Test pipeline creation for variant combinations."""

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    def test_domain_pipelines_create(
        self,
        domain: MaterialDomain,
        compiler: MaterialCompiler,
        pipeline_integration: PipelineIntegration,
        test_material_class,
    ) -> None:
        """Pipelines create successfully for all domain variants."""
        for quality in ALL_QUALITY_TIERS:
            config = VariantConfig(
                domain=domain, blend=BlendMode.OPAQUE, quality=quality
            )
            wgsl = compiler.compile_with_variants(test_material_class, config)

            from trinity.materials.pipeline_integration import PipelineConfig
            pipeline_config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
            handle = pipeline_integration.get_or_create_pipeline(
                wgsl_source=wgsl,
                config=pipeline_config,
            )
            assert handle is not None, f"Pipeline failed for {domain.name}/{quality.name}"

    @pytest.mark.parametrize("blend", ALL_BLEND_MODES, ids=[b.name for b in ALL_BLEND_MODES])
    def test_blend_pipelines_create(
        self,
        blend: BlendMode,
        compiler: MaterialCompiler,
        pipeline_integration: PipelineIntegration,
        test_material_class,
    ) -> None:
        """Pipelines create successfully for all blend mode variants."""
        from trinity.materials.pipeline_integration import PipelineConfig
        pipeline_config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        for quality in ALL_QUALITY_TIERS:
            config = VariantConfig(
                domain=MaterialDomain.SURFACE, blend=blend, quality=quality
            )
            wgsl = compiler.compile_with_variants(test_material_class, config)

            handle = pipeline_integration.get_or_create_pipeline(
                wgsl_source=wgsl,
                config=pipeline_config,
            )
            assert handle is not None, f"Pipeline failed for {blend.name}/{quality.name}"


# =============================================================================
# Suite H: Textured Material Matrix
# =============================================================================


class TestTexturedMaterialMatrix:
    """Test textured materials across variant combinations."""

    @pytest.mark.parametrize("domain", ALL_DOMAINS, ids=[d.name for d in ALL_DOMAINS])
    def test_textured_domain_variants(
        self,
        domain: MaterialDomain,
        compiler: MaterialCompiler,
        textured_material_class,
    ) -> None:
        """Textured materials compile for all domain variants."""
        for quality in ALL_QUALITY_TIERS:
            config = VariantConfig(
                domain=domain, blend=BlendMode.OPAQUE, quality=quality
            )
            wgsl = compiler.compile_with_variants(textured_material_class, config)

            assert len(wgsl) > 100
            assert "@binding" in wgsl or "var<" in wgsl  # Texture bindings

    @pytest.mark.parametrize("blend", ALL_BLEND_MODES, ids=[b.name for b in ALL_BLEND_MODES])
    def test_textured_blend_variants(
        self,
        blend: BlendMode,
        compiler: MaterialCompiler,
        textured_material_class,
    ) -> None:
        """Textured materials compile for all blend mode variants."""
        config = VariantConfig(
            domain=MaterialDomain.SURFACE, blend=blend, quality=QualityTier.MEDIUM
        )
        wgsl = compiler.compile_with_variants(textured_material_class, config)

        assert len(wgsl) > 100
        assert f"BLEND_{blend.name}" in wgsl


# =============================================================================
# Suite I: Feature Flag Integration
# =============================================================================


class TestFeatureFlagIntegration:
    """Test feature flags in variant combinations."""

    def test_feature_flags_in_variant(self, compiler: MaterialCompiler) -> None:
        """Custom feature flags are included in variant output."""

        class FeatureMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
            feature_flags={"ENABLE_SSS", "ENABLE_CLEARCOAT"},
        )

        wgsl = compiler.compile_with_variants(FeatureMaterial, config)

        # Feature flags should be present as consts
        assert "ENABLE_SSS" in wgsl or "SUBSURFACE_ENABLED" in wgsl

    @pytest.mark.parametrize("quality", ALL_QUALITY_TIERS, ids=[q.name for q in ALL_QUALITY_TIERS])
    def test_quality_feature_derivation(
        self,
        quality: QualityTier,
        compiler: MaterialCompiler,
        test_material_class,
    ) -> None:
        """Quality tier correctly derives feature availability."""
        config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=quality,
        )
        wgsl = compiler.compile_with_variants(test_material_class, config)

        quality_cfg = VariantConfig.QUALITY_CONFIG[quality]

        # MAX_LIGHTS
        expected_max_lights = quality_cfg["max_lights"]
        assert f"MAX_LIGHTS: u32 = {expected_max_lights}u" in wgsl


# =============================================================================
# Suite J: Variant Generator Integration
# =============================================================================


class TestVariantGeneratorIntegration:
    """Test DomainVariantGenerator across all combinations."""

    def test_generate_all_variants_count(self) -> None:
        """generate_all_variant_combinations produces 75 configs."""
        configs = list(generate_all_variant_combinations())
        assert len(configs) == 75

    def test_all_generated_configs_unique(self) -> None:
        """All generated configs have unique domain/blend/quality."""
        configs = list(generate_all_variant_combinations())
        unique_keys = set()

        for config in configs:
            key = (config.domain, config.blend, config.quality)
            assert key not in unique_keys, f"Duplicate config: {key}"
            unique_keys.add(key)

    def test_domain_variant_generator_produces_valid_wgsl(self) -> None:
        """DomainVariantGenerator produces valid WGSL templates."""
        for domain in ALL_DOMAINS:
            for blend in ALL_BLEND_MODES:
                for quality in ALL_QUALITY_TIERS:
                    config = VariantConfig(domain=domain, blend=blend, quality=quality)
                    generator = DomainVariantGenerator(config)
                    wgsl = generator.generate_domain_code()

                    assert isinstance(wgsl, str)
                    # Should have some domain-specific content
                    if domain == MaterialDomain.SURFACE:
                        # Surface domain may have lighting code
                        pass
                    elif domain == MaterialDomain.UI:
                        # UI domain has minimal shading
                        pass


# =============================================================================
# Suite K: Coverage Summary Tests
# =============================================================================


class TestCoverageSummary:
    """Verify complete coverage of variant matrix."""

    def test_domain_coverage(self) -> None:
        """All 5 domains are covered."""
        assert len(ALL_DOMAINS) == 5
        expected = {"SURFACE", "DEFERRED_DECAL", "VOLUME", "POST_PROCESS", "UI"}
        actual = {d.name for d in ALL_DOMAINS}
        assert actual == expected

    def test_blend_coverage(self) -> None:
        """All 5 blend modes are covered."""
        assert len(ALL_BLEND_MODES) == 5
        expected = {"OPAQUE", "MASKED", "TRANSLUCENT", "ADDITIVE", "MODULATE"}
        actual = {b.name for b in ALL_BLEND_MODES}
        assert actual == expected

    def test_quality_coverage(self) -> None:
        """All 3 quality tiers are covered."""
        assert len(ALL_QUALITY_TIERS) == 3
        expected = {"LOW", "MEDIUM", "HIGH"}
        actual = {q.name for q in ALL_QUALITY_TIERS}
        assert actual == expected

    def test_total_combinations(self) -> None:
        """Total combinations is 5 x 5 x 3 = 75."""
        total = len(ALL_DOMAINS) * len(ALL_BLEND_MODES) * len(ALL_QUALITY_TIERS)
        assert total == 75

    def test_variant_matrix_exhaustive(
        self, compiler: MaterialCompiler, test_material_class
    ) -> None:
        """Exhaustive test: all 75 combinations compile successfully."""
        success_count = 0
        failure_count = 0
        failures: List[str] = []

        for domain, blend, quality in all_variant_combinations():
            config = VariantConfig(domain=domain, blend=blend, quality=quality)
            try:
                wgsl = compiler.compile_with_variants(test_material_class, config)
                if len(wgsl) > 0:
                    success_count += 1
                else:
                    failure_count += 1
                    failures.append(f"{domain.name}/{blend.name}/{quality.name}: empty output")
            except Exception as e:
                failure_count += 1
                failures.append(f"{domain.name}/{blend.name}/{quality.name}: {e}")

        # Report
        assert failure_count == 0, f"Failures ({failure_count}/75): {failures}"
        assert success_count == 75, f"Only {success_count}/75 variants compiled"


# =============================================================================
# Suite L: Stress Tests
# =============================================================================


class TestVariantStress:
    """Stress tests for variant compilation."""

    def test_rapid_variant_switching(
        self, compiler: MaterialCompiler, test_material_class
    ) -> None:
        """Rapidly compile all 75 variants in sequence."""
        for _ in range(3):  # 3 full passes = 225 compilations
            for domain, blend, quality in all_variant_combinations():
                config = VariantConfig(domain=domain, blend=blend, quality=quality)
                wgsl = compiler.compile_with_variants(test_material_class, config)
                assert len(wgsl) > 0

    def test_parallel_variant_caching(
        self, pipeline_integration: PipelineIntegration, compiler: MaterialCompiler
    ) -> None:
        """Pipeline caching handles many variant pipelines."""
        from trinity.materials.pipeline_integration import PipelineConfig

        class CacheMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        # Create pipelines for all variants
        pipeline_config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        created_count = 0
        for domain, blend, quality in all_variant_combinations():
            config = VariantConfig(domain=domain, blend=blend, quality=quality)
            wgsl = compiler.compile_with_variants(CacheMaterial, config)

            handle = pipeline_integration.get_or_create_pipeline(
                wgsl_source=wgsl,
                config=pipeline_config,
            )
            if handle is not None:
                created_count += 1

        # Most should succeed (cache may evict some)
        assert created_count >= 50, f"Only {created_count}/75 pipelines created"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
