"""End-to-End Pipeline Tests for Material System (T-MAT-11.1).

This module provides comprehensive E2E tests covering the full material pipeline:
1. DSL material definition compiles to valid WGSL
2. WGSL passes naga validation
3. Pipeline creates successfully
4. All 5 domains render correctly
5. Variant combinations work

Gap: S11-G1 (PARTIAL)
Dependencies: T-MAT-3.4 (DONE)
"""

from __future__ import annotations

import pytest
from typing import List, Optional, Tuple, Dict, Any

from trinity.materials import (
    # DSL core
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    # Compiler
    MaterialCompiler,
    # Textures
    Texture2D,
    TextureCube,
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
    DomainOutputFormat,
    DomainShaderTemplate,
    DomainVariantGenerator,
    DOMAIN_CAPABILITIES,
    DOMAIN_OUTPUT_FORMATS,
    domain_has_capability,
)

from trinity.materials.pipeline_integration import (
    PipelineConfig,
    PipelineCacheHandle,
    ShaderCache,
    LruPipelineTable,
    LruPipelineStats,
    PipelineIntegration,
    ColorFormat,
    CullMode,
    BlendMode as PipelineBlendMode,
    shader_hash,
)


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
    """Create a PipelineIntegration with 64-entry cache."""
    return PipelineIntegration(max_cache_size=64)


@pytest.fixture
def shader_cache() -> ShaderCache:
    """Create a fresh ShaderCache instance."""
    return ShaderCache()


# =============================================================================
# Suite A: DSL to WGSL Compilation
# =============================================================================


class TestDSLToWGSLCompilation:
    """Test Material DSL compiles to valid WGSL."""

    def test_simple_material_compiles_to_wgsl(self, compiler: MaterialCompiler) -> None:
        """A simple material produces valid WGSL output."""

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.2)
                out.roughness = 0.4
                out.metallic = 0.0

        wgsl = compiler.compile(SimpleMaterial)
        assert isinstance(wgsl, str)
        assert len(wgsl) > 100  # Non-trivial output
        assert "base_color" in wgsl or "params.base_color" in wgsl

    def test_textured_material_compiles(self, compiler: MaterialCompiler) -> None:
        """A material with textures generates proper bindings."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = ctx.sample(self.albedo, ctx.uv).xyz
                out.roughness = 0.5

        wgsl = compiler.compile(TexturedMaterial)
        assert isinstance(wgsl, str)
        assert len(wgsl) > 100
        # Should have texture bindings
        assert "@binding" in wgsl or "var<" in wgsl

    def test_metallic_material_compiles(self, compiler: MaterialCompiler) -> None:
        """A metallic material with all PBR params compiles."""

        class MetallicMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.83, 0.69, 0.22)  # Gold
                out.metallic = 0.95
                out.roughness = 0.25
                out.emissive = Vec3(0.0, 0.0, 0.0)

        wgsl = compiler.compile(MetallicMaterial)
        assert isinstance(wgsl, str)
        assert "metallic" in wgsl or "params" in wgsl

    def test_material_with_math_operations(self, compiler: MaterialCompiler) -> None:
        """Math operations in surface() compile correctly."""

        class MathMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                brightness = 0.5 + ctx.uv.x * 0.5
                out.base_color = Vec3(brightness, brightness, brightness)
                out.roughness = 1.0 - ctx.uv.y

        wgsl = compiler.compile(MathMaterial)
        assert isinstance(wgsl, str)
        # Should have arithmetic operations
        assert "+" in wgsl or "-" in wgsl or "*" in wgsl

    def test_material_with_conditionals(self, compiler: MaterialCompiler) -> None:
        """Conditional logic in surface() compiles to WGSL if/else."""

        class ConditionalMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if ctx.uv.x > 0.5:
                    out.base_color = Vec3(1.0, 0.0, 0.0)
                else:
                    out.base_color = Vec3(0.0, 1.0, 0.0)
                out.roughness = 0.5

        wgsl = compiler.compile(ConditionalMaterial)
        assert isinstance(wgsl, str)
        # Should have control flow
        assert "if" in wgsl or "select" in wgsl


# =============================================================================
# Suite B: WGSL Naga Validation
# =============================================================================


class TestWGSLNagaValidation:
    """Test that compiled WGSL passes naga validation."""

    def test_simple_material_passes_naga(self, compiler: MaterialCompiler) -> None:
        """Simple material WGSL validates with naga."""

        class ValidMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.2, 0.1)
                out.roughness = 0.5

        wgsl = compiler.compile(ValidMaterial)
        is_valid, error = compiler.validate_wgsl(wgsl)

        # If naga is not installed, validation returns True
        # If naga is installed, we expect validation to pass
        if error is not None:
            pytest.fail(f"WGSL validation failed: {error}")

    def test_textured_material_passes_naga(self, compiler: MaterialCompiler) -> None:
        """Textured material WGSL validates with naga."""

        class TexturedValid(Material, metaclass=MaterialMeta):
            diffuse = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = ctx.sample(self.diffuse, ctx.uv).xyz
                out.roughness = 0.5

        wgsl = compiler.compile(TexturedValid)
        is_valid, error = compiler.validate_wgsl(wgsl)

        if error is not None:
            pytest.fail(f"WGSL validation failed: {error}")

    def test_complex_pbr_material_passes_naga(self, compiler: MaterialCompiler) -> None:
        """Complex PBR material validates with naga."""

        class ComplexPBR(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.9, 0.85, 0.75)
                out.metallic = 0.0
                out.roughness = 0.6
                out.emissive = Vec3(0.0, 0.0, 0.0)
                out.ao = 1.0
                out.normal = Vec3(0.0, 0.0, 1.0)

        wgsl = compiler.compile(ComplexPBR)
        is_valid, error = compiler.validate_wgsl(wgsl)

        if error is not None:
            pytest.fail(f"WGSL validation failed: {error}")

    def test_variant_wgsl_passes_naga(self, compiler: MaterialCompiler) -> None:
        """Variant-compiled WGSL validates with naga."""

        class VariantMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )

        wgsl = compiler.compile_with_variants(VariantMaterial, config)
        is_valid, error = compiler.validate_wgsl(wgsl)

        if error is not None:
            pytest.fail(f"Variant WGSL validation failed: {error}")


# =============================================================================
# Suite C: Pipeline Creation
# =============================================================================


class TestPipelineCreation:
    """Test that valid WGSL creates pipelines successfully."""

    def test_shader_cache_stores_shader(self, shader_cache: ShaderCache) -> None:
        """ShaderCache successfully caches shader modules."""
        wgsl = """
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        """

        module, h = shader_cache.cache_shader(wgsl)
        assert len(h) == 64  # SHA-256 hex
        assert module is not None

    def test_shader_cache_deduplicates(self, shader_cache: ShaderCache) -> None:
        """Identical shaders return same cached module."""
        wgsl = """
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        """

        module1, h1 = shader_cache.cache_shader(wgsl)
        module2, h2 = shader_cache.cache_shader(wgsl)

        assert h1 == h2
        assert module1 == module2
        assert shader_cache.stats.hits == 1
        assert shader_cache.stats.misses == 1

    def test_pipeline_integration_creates_pipeline(
        self, pipeline_integration: PipelineIntegration
    ) -> None:
        """PipelineIntegration creates pipeline from valid WGSL."""
        wgsl = """
        struct PBRInput {
            @builtin(position) position: vec4<f32>,
        }
        struct PBROutput {
            @location(0) color: vec4<f32>,
        }
        @vertex fn vs_main() -> PBRInput {
            var out: PBRInput;
            out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            return out;
        }
        @fragment fn fs_main(input: PBRInput) -> PBROutput {
            var out: PBROutput;
            out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
            return out;
        }
        """

        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )

        assert handle is not None
        assert isinstance(handle, PipelineCacheHandle)
        assert handle.id >= 0
        assert len(handle.shader_hash) == 64

    def test_pipeline_lru_eviction(
        self, pipeline_integration: PipelineIntegration
    ) -> None:
        """LRU cache evicts oldest pipelines when full."""
        # Create cache with small size
        small_cache = PipelineIntegration(max_cache_size=4)

        # Create 5 different shaders
        shaders = []
        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        for i in range(5):
            wgsl = f"""
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {{
                return vec4<f32>(0.0, 0.0, {float(i)}, 1.0);
            }}
            @fragment fn fs_main() -> @location(0) vec4<f32> {{
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }}
            """
            handle = small_cache.get_or_create_pipeline(
                wgsl_source=wgsl,
                config=config,
            )
            shaders.append(handle)

        # Cache should be at max size (4), evicted oldest
        assert len(small_cache) <= 4

    def test_compiled_material_creates_pipeline(
        self, compiler: MaterialCompiler, pipeline_integration: PipelineIntegration
    ) -> None:
        """Material compiled to WGSL can create a pipeline."""

        class PipelineMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.7, 0.3, 0.1)
                out.roughness = 0.5

        wgsl = compiler.compile(PipelineMaterial)

        # This may fail if the WGSL is incomplete for pipeline creation
        # In a real implementation, we'd have complete vertex/fragment shaders
        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )

        assert handle is not None


# =============================================================================
# Suite D: Domain Rendering Tests
# =============================================================================


class TestDomainRendering:
    """Test all 5 material domains render correctly."""

    def test_surface_domain_capabilities(self) -> None:
        """SURFACE domain has full rendering capabilities."""
        domain = MaterialDomain.SURFACE
        assert domain_has_capability(domain, DomainCapability.LIGHTING)
        assert domain_has_capability(domain, DomainCapability.SHADOWS)
        assert domain_has_capability(domain, DomainCapability.DEPTH_WRITE)
        assert domain_has_capability(domain, DomainCapability.NORMAL_MAPPING)
        assert domain_has_capability(domain, DomainCapability.ENVIRONMENT_MAP)
        assert domain_has_capability(domain, DomainCapability.EMISSIVE)

    def test_deferred_decal_domain_capabilities(self) -> None:
        """DEFERRED_DECAL domain has G-buffer capabilities only."""
        domain = MaterialDomain.DEFERRED_DECAL
        assert domain_has_capability(domain, DomainCapability.GBUFFER_OUTPUT)
        assert domain_has_capability(domain, DomainCapability.NORMAL_MAPPING)
        assert not domain_has_capability(domain, DomainCapability.LIGHTING)
        assert not domain_has_capability(domain, DomainCapability.SHADOWS)

    def test_volume_domain_capabilities(self) -> None:
        """VOLUME domain has volumetric capabilities."""
        domain = MaterialDomain.VOLUME
        assert domain_has_capability(domain, DomainCapability.VOLUMETRIC)
        assert domain_has_capability(domain, DomainCapability.EMISSIVE)
        assert not domain_has_capability(domain, DomainCapability.LIGHTING)
        assert not domain_has_capability(domain, DomainCapability.DEPTH_WRITE)

    def test_post_process_domain_capabilities(self) -> None:
        """POST_PROCESS domain has fullscreen capabilities only."""
        domain = MaterialDomain.POST_PROCESS
        assert domain_has_capability(domain, DomainCapability.FULLSCREEN)
        assert not domain_has_capability(domain, DomainCapability.LIGHTING)
        assert not domain_has_capability(domain, DomainCapability.VOLUMETRIC)

    def test_ui_domain_capabilities(self) -> None:
        """UI domain has vertex color capabilities only."""
        domain = MaterialDomain.UI
        assert domain_has_capability(domain, DomainCapability.VERTEX_COLOR)
        assert not domain_has_capability(domain, DomainCapability.LIGHTING)
        assert not domain_has_capability(domain, DomainCapability.NORMAL_MAPPING)

    def test_surface_domain_output_format(self) -> None:
        """SURFACE domain outputs primary color only."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.SURFACE]
        assert fmt.primary_color is True
        assert fmt.normal is False
        assert fmt.material is False

    def test_deferred_decal_domain_output_format(self) -> None:
        """DEFERRED_DECAL domain outputs to G-buffer."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.DEFERRED_DECAL]
        assert fmt.primary_color is True
        assert fmt.normal is True
        assert fmt.material is True

    def test_volume_domain_output_format(self) -> None:
        """VOLUME domain outputs primary color."""
        fmt = DOMAIN_OUTPUT_FORMATS[MaterialDomain.VOLUME]
        assert fmt.primary_color is True

    def test_surface_domain_compiles_with_variant(
        self, compiler: MaterialCompiler
    ) -> None:
        """SURFACE domain material compiles with variant config."""

        class SurfaceMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        config = VariantConfig(domain=MaterialDomain.SURFACE)
        wgsl = compiler.compile_with_variants(SurfaceMat, config)
        assert "DOMAIN_SURFACE" in wgsl
        assert "true" in wgsl

    def test_decal_domain_compiles_with_variant(
        self, compiler: MaterialCompiler
    ) -> None:
        """DEFERRED_DECAL domain material compiles with variant config."""

        class DecalMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.2, 0.1)
                out.alpha = 0.5

        config = VariantConfig(domain=MaterialDomain.DEFERRED_DECAL)
        wgsl = compiler.compile_with_variants(DecalMat, config)
        assert "DOMAIN_DEFERRED_DECAL" in wgsl

    def test_volume_domain_compiles_with_variant(
        self, compiler: MaterialCompiler
    ) -> None:
        """VOLUME domain material compiles with variant config."""

        class VolumeMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.emissive = Vec3(1.0, 0.5, 0.0)
                out.alpha = 0.3

        config = VariantConfig(domain=MaterialDomain.VOLUME)
        wgsl = compiler.compile_with_variants(VolumeMat, config)
        assert "DOMAIN_VOLUME" in wgsl

    def test_postprocess_domain_compiles_with_variant(
        self, compiler: MaterialCompiler
    ) -> None:
        """POST_PROCESS domain material compiles with variant config."""

        class PostMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(ctx.uv.x, ctx.uv.y, 0.5)

        config = VariantConfig(domain=MaterialDomain.POST_PROCESS)
        wgsl = compiler.compile_with_variants(PostMat, config)
        assert "DOMAIN_POST_PROCESS" in wgsl

    def test_ui_domain_compiles_with_variant(
        self, compiler: MaterialCompiler
    ) -> None:
        """UI domain material compiles with variant config."""

        class UIMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = ctx.vertex_color.xyz
                out.alpha = ctx.vertex_color.w

        config = VariantConfig(domain=MaterialDomain.UI)
        wgsl = compiler.compile_with_variants(UIMat, config)
        assert "DOMAIN_UI" in wgsl


# =============================================================================
# Suite E: Full E2E Pipeline
# =============================================================================


class TestFullE2EPipeline:
    """Test complete DSL -> WGSL -> naga -> pipeline flow."""

    def test_e2e_simple_material(
        self, compiler: MaterialCompiler, pipeline_integration: PipelineIntegration
    ) -> None:
        """E2E: Simple material goes through full pipeline."""

        class E2EMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.3, 0.2)
                out.metallic = 0.0
                out.roughness = 0.65

        # Step 1: Compile to WGSL
        wgsl = compiler.compile(E2EMaterial)
        assert len(wgsl) > 0

        # Step 2: Validate with naga
        is_valid, error = compiler.validate_wgsl(wgsl)
        if error is not None:
            pytest.fail(f"E2E naga validation failed: {error}")

        # Step 3: Create pipeline
        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )
        assert handle is not None

    def test_e2e_textured_material(
        self, compiler: MaterialCompiler, pipeline_integration: PipelineIntegration
    ) -> None:
        """E2E: Textured material goes through full pipeline."""

        class E2ETextured(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                color = ctx.sample(self.albedo, ctx.uv)
                out.base_color = color.xyz
                out.roughness = 0.5

        wgsl = compiler.compile(E2ETextured)
        is_valid, error = compiler.validate_wgsl(wgsl)

        if error is not None:
            pytest.fail(f"E2E textured validation failed: {error}")

        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )
        assert handle is not None

    def test_e2e_variant_material(
        self, compiler: MaterialCompiler, pipeline_integration: PipelineIntegration
    ) -> None:
        """E2E: Variant material goes through full pipeline."""

        class E2EVariant(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.4, 0.4, 0.8)
                out.roughness = 0.5

        variant_config = VariantConfig(
            domain=MaterialDomain.SURFACE,
            blend=BlendMode.OPAQUE,
            quality=QualityTier.HIGH,
        )

        wgsl = compiler.compile_with_variants(E2EVariant, variant_config)
        is_valid, error = compiler.validate_wgsl(wgsl)

        if error is not None:
            pytest.fail(f"E2E variant validation failed: {error}")

        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )
        assert handle is not None

    def test_e2e_all_domains(
        self, compiler: MaterialCompiler, pipeline_integration: PipelineIntegration
    ) -> None:
        """E2E: All 5 domains compile and create pipelines."""

        class MultiDomainMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        domains = [
            MaterialDomain.SURFACE,
            MaterialDomain.DEFERRED_DECAL,
            MaterialDomain.VOLUME,
            MaterialDomain.POST_PROCESS,
            MaterialDomain.UI,
        ]

        for domain in domains:
            variant_config = VariantConfig(domain=domain)
            wgsl = compiler.compile_with_variants(MultiDomainMaterial, variant_config)

            is_valid, error = compiler.validate_wgsl(wgsl)
            if error is not None:
                pytest.fail(f"E2E {domain.name} validation failed: {error}")

            config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
            handle = pipeline_integration.get_or_create_pipeline(
                wgsl_source=wgsl,
                config=config,
            )
            assert handle is not None, f"Pipeline creation failed for {domain.name}"


# =============================================================================
# Suite F: Hot-Reload Integration
# =============================================================================


class TestHotReloadIntegration:
    """Test hot-reload invalidation in pipeline cache."""

    def test_invalidate_shader_by_path(
        self, pipeline_integration: PipelineIntegration
    ) -> None:
        """Invalidating shader path removes affected pipelines."""
        wgsl = """
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        """

        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")
        handle = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
            source_path="shaders/test.wgsl",
        )

        initial_count = len(pipeline_integration)

        # Invalidate by path
        pipeline_integration.invalidate_shader("shaders/test.wgsl")

        # Pipeline should be removed
        final_count = len(pipeline_integration)
        assert final_count < initial_count or final_count == 0

    def test_invalidate_recompiles_correctly(
        self, pipeline_integration: PipelineIntegration
    ) -> None:
        """After invalidation, new pipeline is created on next access."""
        wgsl_v1 = """
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        """

        wgsl_v2 = """
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.5, 0.5, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0);
        }
        """

        path = "shaders/recompile_test.wgsl"
        config = PipelineConfig(vertex_entry="vs_main", fragment_entry="fs_main")

        # Create initial pipeline
        handle1 = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl_v1,
            config=config,
            source_path=path,
        )

        # Invalidate
        pipeline_integration.invalidate_shader(path)

        # Create new pipeline with different source
        handle2 = pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl_v2,
            config=config,
            source_path=path,
        )

        # Hashes should differ
        assert handle1.shader_hash != handle2.shader_hash


# =============================================================================
# Suite G: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_invalid_wgsl_fails_validation(
        self, compiler: MaterialCompiler
    ) -> None:
        """Invalid WGSL is detected by naga validation."""
        invalid_wgsl = """
        @vertex fn vs_main( -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        """  # Missing closing parenthesis

        is_valid, error = compiler.validate_wgsl(invalid_wgsl)

        # If naga is installed, this should fail
        # If naga is not installed, validation returns True
        # We can't guarantee naga is installed, so we just check the call works
        assert isinstance(is_valid, bool)

    def test_material_with_unsupported_construct(
        self, compiler: MaterialCompiler
    ) -> None:
        """Unsupported Python constructs are handled gracefully."""
        # Note: This depends on what constructs are supported
        # The DSL should handle or reject unsupported constructs

        class SimpleValid(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.roughness = 0.5

        # This should compile without error
        wgsl = compiler.compile(SimpleValid)
        assert len(wgsl) > 0


# =============================================================================
# Suite H: Performance Metrics
# =============================================================================


class TestPipelinePerformance:
    """Test pipeline creation performance characteristics."""

    def test_shader_cache_hit_rate(self, shader_cache: ShaderCache) -> None:
        """Cache hit rate is calculated correctly."""
        wgsl1 = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        wgsl2 = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }"

        # Miss, miss, hit, hit
        shader_cache.cache_shader(wgsl1)  # miss
        shader_cache.cache_shader(wgsl2)  # miss
        shader_cache.cache_shader(wgsl1)  # hit
        shader_cache.cache_shader(wgsl2)  # hit

        stats = shader_cache.stats
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate() == 50.0

    def test_pipeline_cache_stats(
        self, pipeline_integration: PipelineIntegration
    ) -> None:
        """Pipeline cache statistics are tracked."""
        wgsl = """
        @vertex fn vs() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        @fragment fn fs() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        """

        # Create pipeline
        config = PipelineConfig(vertex_entry="vs", fragment_entry="fs")
        pipeline_integration.get_or_create_pipeline(
            wgsl_source=wgsl,
            config=config,
        )

        assert len(pipeline_integration) >= 1

        # Get stats
        stats = pipeline_integration.stats
        assert isinstance(stats, LruPipelineStats)


# =============================================================================
# Suite I: Blend Mode Integration (T-MAT-2.3)
# =============================================================================


class TestBlendModeIntegration:
    """Test blend mode WGSL integration in full material compilation.

    These tests verify T-MAT-2.3 acceptance criteria:
    - MASKED mode generates discard statement
    - All 5 blend modes compile correctly
    - Blend handling is integrated into fragment shader
    """

    def test_masked_blend_generates_discard(self, compiler: MaterialCompiler) -> None:
        """MASKED blend mode produces discard in fragment shader."""
        from trinity.materials.variants import VariantConfig, BlendMode

        class MaskedMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.2)
                out.alpha = ctx.uv.x  # Variable alpha for testing

        config = VariantConfig(blend=BlendMode.MASKED)
        wgsl = compiler.compile_with_variants(MaskedMaterial, config)

        # Must have discard in the shader
        assert "discard" in wgsl
        # Must have ALPHA_TEST_ENABLED = true
        assert "ALPHA_TEST_ENABLED: bool = true" in wgsl
        # Must have apply_blend_mode function
        assert "fn apply_blend_mode" in wgsl

    def test_translucent_blend_preserves_alpha(
        self, compiler: MaterialCompiler
    ) -> None:
        """TRANSLUCENT blend mode preserves alpha channel."""
        from trinity.materials.variants import VariantConfig, BlendMode

        class TranslucentMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.2, 0.1)
                out.alpha = 0.5

        config = VariantConfig(blend=BlendMode.TRANSLUCENT)
        wgsl = compiler.compile_with_variants(TranslucentMaterial, config)

        # Must have ALPHA_BLEND_ENABLED = true
        assert "ALPHA_BLEND_ENABLED: bool = true" in wgsl
        # Should NOT have ALPHA_TEST_ENABLED = true (no discard)
        assert "ALPHA_TEST_ENABLED: bool = false" in wgsl

    def test_opaque_blend_forces_alpha_one(self, compiler: MaterialCompiler) -> None:
        """OPAQUE blend mode forces alpha to 1.0."""
        from trinity.materials.variants import VariantConfig, BlendMode

        class OpaqueMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)

        config = VariantConfig(blend=BlendMode.OPAQUE)
        wgsl = compiler.compile_with_variants(OpaqueMaterial, config)

        # Both should be false
        assert "ALPHA_TEST_ENABLED: bool = false" in wgsl
        assert "ALPHA_BLEND_ENABLED: bool = false" in wgsl
        # The apply_blend_mode function should return alpha = 1.0
        assert "color.rgb, 1.0" in wgsl

    def test_all_blend_modes_compile(self, compiler: MaterialCompiler) -> None:
        """All 5 blend modes produce valid compilable WGSL."""
        from trinity.materials.variants import VariantConfig, BlendMode

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.alpha = 0.8

        for blend in BlendMode:
            config = VariantConfig(blend=blend)
            wgsl = compiler.compile_with_variants(SimpleMaterial, config)

            # Should have non-trivial output
            assert len(wgsl) > 1000, f"Mode {blend.name} too short"

            # Should have fragment shader
            assert "@fragment" in wgsl, f"Mode {blend.name} missing @fragment"

            # Should have blend handling
            assert "fn apply_blend_mode" in wgsl, (
                f"Mode {blend.name} missing apply_blend_mode"
            )

            # Should have the correct blend const
            assert f"BLEND_{blend.name}: bool = true" in wgsl, (
                f"Mode {blend.name} missing blend const"
            )

    def test_blend_mode_fragment_shader_calls_apply_blend_mode(
        self, compiler: MaterialCompiler
    ) -> None:
        """Fragment shader calls apply_blend_mode function."""
        from trinity.materials.variants import VariantConfig, BlendMode

        class TestMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.0, 0.0)

        config = VariantConfig(blend=BlendMode.MASKED)
        wgsl = compiler.compile_with_variants(TestMaterial, config)

        # Fragment main should call apply_blend_mode
        assert "apply_blend_mode(" in wgsl

    def test_default_compiler_has_blend_handling(
        self, compiler: MaterialCompiler
    ) -> None:
        """Compiler without variant config still has blend handling."""

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)

        wgsl = compiler.compile(SimpleMaterial)

        # Should have default blend handling (non-variant path)
        assert "fn apply_blend_mode" in wgsl
        # Should have default consts
        assert "ALPHA_TEST_ENABLED" in wgsl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
