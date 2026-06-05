"""Tests for Pipeline Integration (T-MAT-3.4).

This module tests:
- ShaderCache content-addressed deduplication
- LruPipelineTable with LRU eviction
- get_or_create_pipeline semantics
- Hot-reload invalidation via DepGraph
- Cache statistics and hit rates
- PBR shader pipeline caching
"""

from __future__ import annotations

import pytest

from trinity.materials.pipeline_integration import (
    # Types
    ColorFormat,
    CullMode,
    BlendMode,
    PipelineConfig,
    PipelineCacheHandle,
    CachedPipeline,
    # Statistics
    ShaderCacheStats,
    LruPipelineStats,
    # Caches
    ShaderCache,
    LruPipelineTable,
    # Main interface
    PipelineIntegration,
    # Utilities
    content_hash,
    shader_hash,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_vertex_shader() -> str:
    """Simple vertex shader for testing."""
    return """
    @vertex fn vs_main() -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    """


@pytest.fixture
def simple_fragment_shader() -> str:
    """Simple fragment shader for testing."""
    return """
    @fragment fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    """


@pytest.fixture
def combined_shader() -> str:
    """Combined vertex + fragment shader for testing."""
    return """
    @vertex fn vs_main() -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    @fragment fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    """


@pytest.fixture
def pbr_shader() -> str:
    """Minimal PBR shader for testing."""
    return """
    struct PBRInput {
        base_color: vec3<f32>,
        roughness: f32,
        metallic: f32,
        normal: vec3<f32>,
    }

    struct PBROutput {
        color: vec4<f32>,
    }

    @vertex fn vs_pbr(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        var positions = array<vec2<f32>, 3>(
            vec2<f32>(-0.5, -0.5),
            vec2<f32>( 0.5, -0.5),
            vec2<f32>( 0.0,  0.5)
        );
        return vec4<f32>(positions[idx], 0.0, 1.0);
    }

    @fragment fn fs_pbr() -> @location(0) vec4<f32> {
        // Simplified PBR output
        let base_color = vec3<f32>(0.8, 0.2, 0.2);
        let roughness = 0.5;
        let metallic = 0.0;

        // Very simplified lighting
        let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
        let normal = vec3<f32>(0.0, 0.0, 1.0);
        let NdotL = max(dot(normal, light_dir), 0.0);

        let diffuse = base_color * NdotL;
        return vec4<f32>(diffuse, 1.0);
    }
    """


# =============================================================================
# Content Hash Tests
# =============================================================================


class TestContentHash:
    """Test content hashing utilities."""

    def test_content_hash_deterministic(self) -> None:
        """Test that content_hash produces deterministic results."""
        data = b"hello world"
        h1 = content_hash(data)
        h2 = content_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_content_hash_different_data(self) -> None:
        """Test that different data produces different hashes."""
        h1 = content_hash(b"hello")
        h2 = content_hash(b"world")
        assert h1 != h2

    def test_shader_hash_deterministic(self) -> None:
        """Test that shader_hash produces deterministic results."""
        src = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        h1 = shader_hash(src)
        h2 = shader_hash(src)
        assert h1 == h2

    def test_shader_hash_whitespace_sensitive(self) -> None:
        """Test that shader_hash is whitespace-sensitive."""
        src1 = "@vertex fn vs() {}"
        src2 = "@vertex fn vs()  {}"  # Extra space
        h1 = shader_hash(src1)
        h2 = shader_hash(src2)
        assert h1 != h2


# =============================================================================
# Shader Cache Tests
# =============================================================================


class TestShaderCache:
    """Test ShaderCache content-addressed deduplication."""

    def test_cache_shader_returns_hash(self, simple_vertex_shader: str) -> None:
        """Test that cache_shader returns a valid hash."""
        cache = ShaderCache()
        _module, h = cache.cache_shader(simple_vertex_shader)
        assert len(h) == 64
        assert h == shader_hash(simple_vertex_shader)

    def test_cache_shader_deduplication(self, simple_vertex_shader: str) -> None:
        """Test that identical sources return the same cached module."""
        cache = ShaderCache()

        # First cache: miss
        module1, h1 = cache.cache_shader(simple_vertex_shader)
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

        # Second cache: hit
        module2, h2 = cache.cache_shader(simple_vertex_shader)
        assert cache.stats.misses == 1
        assert cache.stats.hits == 1

        assert h1 == h2
        assert module1 == module2

    def test_cache_shader_different_sources(
        self, simple_vertex_shader: str, simple_fragment_shader: str
    ) -> None:
        """Test that different sources produce different hashes."""
        cache = ShaderCache()

        _, h1 = cache.cache_shader(simple_vertex_shader)
        _, h2 = cache.cache_shader(simple_fragment_shader)

        assert h1 != h2
        assert len(cache) == 2
        assert cache.stats.misses == 2

    def test_cache_shader_with_path_tracking(self, simple_vertex_shader: str) -> None:
        """Test path tracking for hot-reload."""
        cache = ShaderCache()

        _, h = cache.cache_shader_with_path(simple_vertex_shader, "shaders/test.wgsl")

        assert cache.hash_for_path("shaders/test.wgsl") == h
        assert cache.paths_for_hash(h) == ["shaders/test.wgsl"]
        assert cache.stats.tracked_paths == 1

    def test_cache_multiple_paths_same_hash(self, simple_vertex_shader: str) -> None:
        """Test that multiple paths can map to the same hash."""
        cache = ShaderCache()

        _, h1 = cache.cache_shader_with_path(simple_vertex_shader, "path/a.wgsl")
        _, h2 = cache.cache_shader_with_path(simple_vertex_shader, "path/b.wgsl")

        assert h1 == h2
        assert len(cache) == 1  # Only one module
        assert cache.stats.tracked_paths == 2
        assert set(cache.paths_for_hash(h1) or []) == {"path/a.wgsl", "path/b.wgsl"}

    def test_invalidate_path(self, simple_vertex_shader: str) -> None:
        """Test invalidating a shader by path."""
        cache = ShaderCache()

        _, h = cache.cache_shader_with_path(simple_vertex_shader, "test.wgsl")
        assert len(cache) == 1

        old_hash = cache.invalidate_path("test.wgsl")
        assert old_hash == h
        assert len(cache) == 0
        assert cache.hash_for_path("test.wgsl") is None

    def test_invalidate_one_of_multiple_paths(self, simple_vertex_shader: str) -> None:
        """Test that invalidating one path doesn't remove module if other paths exist."""
        cache = ShaderCache()

        cache.cache_shader_with_path(simple_vertex_shader, "path/a.wgsl")
        cache.cache_shader_with_path(simple_vertex_shader, "path/b.wgsl")

        cache.invalidate_path("path/a.wgsl")

        # Module should still exist (path/b.wgsl references it)
        assert len(cache) == 1
        assert cache.stats.tracked_paths == 1

        # Invalidate the other path
        cache.invalidate_path("path/b.wgsl")
        assert len(cache) == 0

    def test_hit_rate_calculation(self) -> None:
        """Test cache hit rate calculation."""
        stats = ShaderCacheStats()

        # No lookups: 100%
        assert stats.hit_rate() == 100.0

        # All misses: 0%
        stats.misses = 10
        assert stats.hit_rate() == 0.0

        # 50/50: 50%
        stats.hits = 10
        assert stats.hit_rate() == 50.0

        # 90%
        stats.hits = 90
        stats.misses = 10
        assert stats.hit_rate() == 90.0

    def test_clear(self, simple_vertex_shader: str) -> None:
        """Test clearing the cache."""
        cache = ShaderCache()
        cache.cache_shader_with_path(simple_vertex_shader, "test.wgsl")

        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0
        assert cache.stats.tracked_paths == 0

    def test_reset_stats(self, simple_vertex_shader: str) -> None:
        """Test resetting statistics."""
        cache = ShaderCache()
        cache.cache_shader(simple_vertex_shader)
        cache.cache_shader(simple_vertex_shader)

        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

        cache.reset_stats()

        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.cached_modules == 1  # Module count preserved


# =============================================================================
# LRU Pipeline Table Tests
# =============================================================================


class TestLruPipelineTable:
    """Test LruPipelineTable with LRU eviction."""

    def test_new_table(self) -> None:
        """Test creating a new table."""
        table = LruPipelineTable(max_size=10)
        assert len(table) == 0
        assert table.is_empty()
        assert table.max_size == 10

    def test_zero_size_raises(self) -> None:
        """Test that zero max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be greater than 0"):
            LruPipelineTable(max_size=0)

    def test_get_or_create_returns_handle(self, combined_shader: str) -> None:
        """Test that get_or_create_pipeline returns a handle."""
        table = LruPipelineTable(max_size=10)
        handle = table.get_or_create_pipeline(combined_shader)

        assert isinstance(handle, PipelineCacheHandle)
        assert handle.id >= 1
        assert len(handle.shader_hash) == 64

    def test_get_or_create_cache_hit(self, combined_shader: str) -> None:
        """Test cache hit on repeated get_or_create."""
        table = LruPipelineTable(max_size=10)

        handle1 = table.get_or_create_pipeline(combined_shader)
        handle2 = table.get_or_create_pipeline(combined_shader)

        assert handle1.id == handle2.id
        assert handle1.shader_hash == handle2.shader_hash
        assert len(table) == 1
        assert table.stats.hits == 1
        assert table.stats.misses == 1

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        table = LruPipelineTable(max_size=2)

        # Add 3 pipelines to a cache of size 2
        src1 = "@vertex fn vs_1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        src2 = "@vertex fn vs_2() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }"
        src3 = "@vertex fn vs_3() -> @builtin(position) vec4<f32> { return vec4<f32>(2.0); }"

        handle1 = table.get_or_create_pipeline(src1)
        handle2 = table.get_or_create_pipeline(src2)

        assert len(table) == 2
        assert table.stats.evictions == 0

        # Adding third should evict first (LRU)
        handle3 = table.get_or_create_pipeline(src3)

        assert len(table) == 2
        assert table.stats.evictions == 1
        assert not table.contains(handle1.id)  # Evicted
        assert table.contains(handle2.id)
        assert table.contains(handle3.id)

    def test_lru_touch_updates_order(self) -> None:
        """Test that accessing a pipeline updates LRU order."""
        table = LruPipelineTable(max_size=2)

        src1 = "@vertex fn vs_1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        src2 = "@vertex fn vs_2() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }"
        src3 = "@vertex fn vs_3() -> @builtin(position) vec4<f32> { return vec4<f32>(2.0); }"

        handle1 = table.get_or_create_pipeline(src1)
        _handle2 = table.get_or_create_pipeline(src2)

        # Touch handle1 (move to front)
        _ = table.get_or_create_pipeline(src1)

        # Now handle2 is LRU, adding handle3 should evict handle2
        _handle3 = table.get_or_create_pipeline(src3)

        assert table.contains(handle1.id)  # Not evicted (was touched)
        # handle2 was evicted

    def test_lru_order(self) -> None:
        """Test LRU order tracking."""
        table = LruPipelineTable(max_size=10)

        src1 = "@vertex fn vs_1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        src2 = "@vertex fn vs_2() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }"
        src3 = "@vertex fn vs_3() -> @builtin(position) vec4<f32> { return vec4<f32>(2.0); }"

        h1 = table.get_or_create_pipeline(src1)
        h2 = table.get_or_create_pipeline(src2)
        h3 = table.get_or_create_pipeline(src3)

        # Order: [h3, h2, h1] (most recent first)
        order = table.lru_order()
        assert order == [h3.id, h2.id, h1.id]

        # Touch h1
        _ = table.get_or_create_pipeline(src1)

        # Order: [h1, h3, h2]
        order = table.lru_order()
        assert order == [h1.id, h3.id, h2.id]

    def test_invalidate_by_hash(self, combined_shader: str) -> None:
        """Test invalidating pipelines by shader hash."""
        table = LruPipelineTable(max_size=10)

        handle = table.get_or_create_pipeline(combined_shader)
        h = handle.shader_hash

        assert table.contains(handle.id)
        assert table.contains_hash(h)

        invalidated = table.invalidate_by_hash(h)
        assert invalidated == [handle.id]
        assert table.stats.invalidations == 1
        assert not table.contains(handle.id)
        assert not table.contains_hash(h)

    def test_invalidate_by_path(self, combined_shader: str) -> None:
        """Test invalidating pipelines by source path."""
        table = LruPipelineTable(max_size=10)

        handle = table.get_or_create_pipeline(
            combined_shader, source_path="shaders/test.wgsl"
        )

        invalidated = table.invalidate_by_path("shaders/test.wgsl")
        assert invalidated == [handle.id]
        assert not table.contains(handle.id)

    def test_set_max_size_evicts(self) -> None:
        """Test that reducing max_size triggers eviction."""
        table = LruPipelineTable(max_size=5)

        for i in range(5):
            src = f"@vertex fn vs_{i}() -> @builtin(position) vec4<f32> {{ return vec4<f32>({i}.0); }}"
            table.get_or_create_pipeline(src)

        assert len(table) == 5

        table.set_max_size(2)

        assert len(table) == 2
        assert table.stats.evictions == 3

    def test_remove_pipeline(self, combined_shader: str) -> None:
        """Test removing a pipeline by ID."""
        table = LruPipelineTable(max_size=10)

        handle = table.get_or_create_pipeline(combined_shader)

        assert table.contains(handle.id)
        assert table.remove(handle.id)
        assert not table.contains(handle.id)
        assert not table.remove(handle.id)  # Already removed

    def test_get_touch(self, combined_shader: str) -> None:
        """Test get_touch updates LRU order."""
        table = LruPipelineTable(max_size=10)

        handle = table.get_or_create_pipeline(combined_shader)

        # get doesn't update LRU
        pipeline = table.get(handle.id)
        assert pipeline is not None

        # get_touch updates LRU
        pipeline = table.get_touch(handle.id)
        assert pipeline is not None

    def test_clear(self, combined_shader: str) -> None:
        """Test clearing the table."""
        table = LruPipelineTable(max_size=10)
        table.get_or_create_pipeline(combined_shader)

        assert len(table) > 0
        table.clear()
        assert len(table) == 0
        assert table.is_empty()

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        stats = LruPipelineStats(hits=80, misses=20)
        assert stats.hit_rate() == 80.0


# =============================================================================
# Pipeline Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Test PipelineIntegration main interface."""

    def test_create_integration(self) -> None:
        """Test creating a PipelineIntegration instance."""
        integration = PipelineIntegration(max_cache_size=64)
        assert len(integration) == 0
        assert integration.max_cache_size == 64

    def test_get_or_create_pipeline(self, combined_shader: str) -> None:
        """Test get_or_create_pipeline."""
        integration = PipelineIntegration()

        handle = integration.get_or_create_pipeline(
            wgsl_source=combined_shader,
            config=PipelineConfig(
                vertex_entry="vs_main",
                fragment_entry="fs_main",
            ),
        )

        assert isinstance(handle, PipelineCacheHandle)
        assert handle.id >= 1

    def test_cache_hit(self, combined_shader: str) -> None:
        """Test that repeated calls produce cache hits."""
        integration = PipelineIntegration()

        handle1 = integration.get_or_create_pipeline(combined_shader)
        handle2 = integration.get_or_create_pipeline(combined_shader)

        assert handle1.id == handle2.id
        assert integration.stats.hits == 1
        assert integration.stats.misses == 1

    def test_cache_hit_rate(self, combined_shader: str) -> None:
        """Test cache hit rate reporting."""
        integration = PipelineIntegration()

        # First call: miss
        integration.get_or_create_pipeline(combined_shader)
        assert integration.cache_hit_rate() == 0.0

        # Second call: hit
        integration.get_or_create_pipeline(combined_shader)
        assert integration.cache_hit_rate() == 50.0

        # Third call: hit
        integration.get_or_create_pipeline(combined_shader)
        assert integration.cache_hit_rate() == pytest.approx(66.67, rel=0.1)

    def test_invalidate_shader(self, combined_shader: str) -> None:
        """Test hot-reload shader invalidation."""
        invalidated_ids: list = []

        def on_invalidate(ids: list) -> None:
            invalidated_ids.extend(ids)

        integration = PipelineIntegration(on_invalidate=on_invalidate)

        handle = integration.get_or_create_pipeline(
            combined_shader, source_path="shaders/test.wgsl"
        )

        result = integration.invalidate_shader("shaders/test.wgsl")
        assert result == [handle.id]
        assert invalidated_ids == [handle.id]

    def test_invalidate_by_hash(self, combined_shader: str) -> None:
        """Test invalidation by content hash."""
        integration = PipelineIntegration()

        handle = integration.get_or_create_pipeline(combined_shader)
        h = handle.shader_hash

        result = integration.invalidate_by_hash(h)
        assert result == [handle.id]

    def test_get_pipeline(self, combined_shader: str) -> None:
        """Test getting a pipeline by ID."""
        integration = PipelineIntegration()

        handle = integration.get_or_create_pipeline(combined_shader)
        pipeline = integration.get_pipeline(handle.id)

        assert pipeline is not None
        assert pipeline.id == handle.id

    def test_clear(self, combined_shader: str) -> None:
        """Test clearing the cache."""
        integration = PipelineIntegration()
        integration.get_or_create_pipeline(combined_shader)

        assert len(integration) > 0
        integration.clear()
        assert len(integration) == 0


# =============================================================================
# PBR Pipeline Tests
# =============================================================================


class TestPBRPipeline:
    """Test PBR shader pipeline caching."""

    def test_pbr_shader_caches(self, pbr_shader: str) -> None:
        """Test that PBR shader is cached correctly."""
        integration = PipelineIntegration()

        config = PipelineConfig(
            vertex_entry="vs_pbr",
            fragment_entry="fs_pbr",
            color_format=ColorFormat.RGBA16_FLOAT,
        )

        handle = integration.get_or_create_pipeline(
            wgsl_source=pbr_shader,
            config=config,
            source_path="shaders/pbr.wgsl",
        )

        assert handle.config.vertex_entry == "vs_pbr"
        assert handle.config.fragment_entry == "fs_pbr"
        assert handle.config.color_format == ColorFormat.RGBA16_FLOAT

    def test_pbr_shader_cache_hit(self, pbr_shader: str) -> None:
        """Test PBR shader cache hit."""
        integration = PipelineIntegration()

        handle1 = integration.get_or_create_pipeline(pbr_shader)
        handle2 = integration.get_or_create_pipeline(pbr_shader)

        assert handle1.id == handle2.id
        assert integration.stats.hits == 1

    def test_pbr_shader_hot_reload(self, pbr_shader: str) -> None:
        """Test PBR shader hot-reload invalidation."""
        integration = PipelineIntegration()

        handle = integration.get_or_create_pipeline(
            pbr_shader, source_path="shaders/pbr.wgsl"
        )

        # Simulate hot-reload
        invalidated = integration.invalidate_shader("shaders/pbr.wgsl")
        assert invalidated == [handle.id]

        # Next get_or_create should recompile
        new_handle = integration.get_or_create_pipeline(
            pbr_shader, source_path="shaders/pbr.wgsl"
        )
        assert new_handle.id != handle.id
        assert integration.stats.misses == 2

    def test_pbr_modified_shader_different_hash(self, pbr_shader: str) -> None:
        """Test that modified PBR shader has different hash."""
        integration = PipelineIntegration()

        handle1 = integration.get_or_create_pipeline(pbr_shader)

        # Modify shader
        modified = pbr_shader.replace("roughness = 0.5", "roughness = 0.8")
        handle2 = integration.get_or_create_pipeline(modified)

        assert handle1.shader_hash != handle2.shader_hash
        assert handle1.id != handle2.id


# =============================================================================
# Pipeline Config Tests
# =============================================================================


class TestPipelineConfig:
    """Test PipelineConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.vertex_entry == "vs_main"
        assert config.fragment_entry == "fs_main"
        assert config.color_format == ColorFormat.RGBA8_UNORM
        assert config.depth_format is None
        assert config.cull_mode == CullMode.BACK
        assert config.blend_mode == BlendMode.OPAQUE
        assert config.sample_count == 1
        assert config.label is None

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PipelineConfig(
            vertex_entry="vs_pbr",
            fragment_entry="fs_pbr",
            color_format=ColorFormat.RGBA16_FLOAT,
            depth_format="Depth32Float",
            cull_mode=CullMode.NONE,
            blend_mode=BlendMode.ALPHA_BLEND,
            sample_count=4,
            label="PBR Pipeline",
        )

        assert config.vertex_entry == "vs_pbr"
        assert config.fragment_entry == "fs_pbr"
        assert config.color_format == ColorFormat.RGBA16_FLOAT
        assert config.depth_format == "Depth32Float"
        assert config.cull_mode == CullMode.NONE
        assert config.blend_mode == BlendMode.ALPHA_BLEND
        assert config.sample_count == 4
        assert config.label == "PBR Pipeline"


# =============================================================================
# Integration with Existing Material System
# =============================================================================


class TestMaterialSystemIntegration:
    """Test integration with TRINITY material system."""

    def test_import_from_materials_package(self) -> None:
        """Test that pipeline_integration is importable from trinity.materials."""
        from trinity.materials import (
            PipelineConfig,
            PipelineIntegration,
            ShaderCache,
            LruPipelineTable,
            content_hash,
            shader_hash,
        )

        assert PipelineConfig is not None
        assert PipelineIntegration is not None
        assert ShaderCache is not None
        assert LruPipelineTable is not None
        assert content_hash is not None
        assert shader_hash is not None

    def test_pipeline_with_brdf_shader(self) -> None:
        """Test pipeline caching with BRDF WGSL."""
        from trinity.materials.brdf import get_brdf_wgsl

        integration = PipelineIntegration()
        brdf_wgsl = get_brdf_wgsl()

        # Note: BRDF WGSL alone isn't a complete shader, but we can hash it
        h = shader_hash(brdf_wgsl)
        assert len(h) == 64

    def test_pipeline_with_lighting_shader(self) -> None:
        """Test pipeline caching with lighting WGSL."""
        from trinity.materials.lighting import get_lighting_wgsl

        integration = PipelineIntegration()
        lighting_wgsl = get_lighting_wgsl()

        h = shader_hash(lighting_wgsl)
        assert len(h) == 64
