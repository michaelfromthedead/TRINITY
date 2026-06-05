"""Tests for Metal-specific rendering optimizations (T-CC-0.12)."""
import pytest

from trinity.types import QualityTier
from engine.rendering.backends.metal_optimizations import (
    MetalFeatureLevel,
    TBDROptimization,
    ArgumentBufferConfig,
    MemorylessAttachment,
    TileShaderConfig,
    MetalCapabilities,
    MetalOptimizer,
    RenderPassConfig,
    create_optimizer_for_device,
)


class TestMetalFeatureLevel:
    """Test MetalFeatureLevel enum."""

    def test_feature_levels(self):
        """Test feature level values."""
        assert MetalFeatureLevel.METAL_1.value == "metal_1"
        assert MetalFeatureLevel.METAL_2.value == "metal_2"
        assert MetalFeatureLevel.METAL_3.value == "metal_3"


class TestMemorylessAttachment:
    """Test MemorylessAttachment dataclass."""

    def test_attachment_creation(self):
        """Test attachment creation."""
        att = MemorylessAttachment(
            name="depth",
            format="D32F",
            usage="depth_stencil",
        )
        assert att.name == "depth"
        assert att.format == "D32F"
        assert att.load_action == "dont_care"
        assert att.store_action == "dont_care"

    def test_is_truly_memoryless(self):
        """Test memoryless detection."""
        att_memoryless = MemorylessAttachment(
            name="temp",
            format="RGBA8",
            load_action="dont_care",
            store_action="dont_care",
        )
        assert att_memoryless.is_truly_memoryless()

        att_stored = MemorylessAttachment(
            name="final",
            format="RGBA8",
            load_action="dont_care",
            store_action="store",
        )
        assert not att_stored.is_truly_memoryless()

    def test_is_truly_memoryless_clear(self):
        """Test clear action is memoryless compatible."""
        att = MemorylessAttachment(
            name="depth",
            format="D32F",
            load_action="clear",
            store_action="dont_care",
        )
        assert att.is_truly_memoryless()

    def test_bandwidth_savings(self):
        """Test bandwidth savings calculation."""
        att = MemorylessAttachment(name="color", format="RGBA8")
        savings = att.estimated_bandwidth_savings(1920, 1080)
        # 1920 * 1080 * 4 bytes = 8,294,400 bytes
        assert savings == 1920 * 1080 * 4

    def test_bandwidth_savings_16f(self):
        """Test bandwidth savings for RGBA16F."""
        att = MemorylessAttachment(name="hdr", format="RGBA16F")
        savings = att.estimated_bandwidth_savings(1920, 1080)
        assert savings == 1920 * 1080 * 8


class TestTileShaderConfig:
    """Test TileShaderConfig dataclass."""

    def test_tile_shader_creation(self):
        """Test tile shader config creation."""
        config = TileShaderConfig(
            name="deferred",
            tile_width=32,
            tile_height=32,
        )
        assert config.threads_per_tile == 1024
        assert config.uses_imageblock

    def test_can_fit_in_tile_memory(self):
        """Test tile memory fit check."""
        config = TileShaderConfig(
            name="test",
            threadgroup_memory_bytes=16384,
        )
        assert config.can_fit_in_tile_memory(8192)
        assert config.can_fit_in_tile_memory(16384)
        assert not config.can_fit_in_tile_memory(32768)


class TestArgumentBufferConfig:
    """Test ArgumentBufferConfig dataclass."""

    def test_argument_buffer_creation(self):
        """Test argument buffer config creation."""
        config = ArgumentBufferConfig(
            name="materials",
            tier=2,
            max_textures=500000,
        )
        assert config.supports_bindless()

    def test_tier1_not_bindless(self):
        """Test tier 1 doesn't support bindless."""
        config = ArgumentBufferConfig(
            name="basic",
            tier=1,
            uses_heaps=False,
        )
        assert not config.supports_bindless()


class TestRenderPassConfig:
    """Test RenderPassConfig dataclass."""

    def test_render_pass_creation(self):
        """Test render pass config creation."""
        config = RenderPassConfig(name="gbuffer")
        assert config.name == "gbuffer"
        assert len(config.memoryless_attachments) == 0
        assert config.tile_shader is None

    def test_total_bandwidth_savings(self):
        """Test total bandwidth savings calculation."""
        config = RenderPassConfig(
            name="test",
            memoryless_attachments=[
                MemorylessAttachment(
                    name="a",
                    format="RGBA8",
                    load_action="dont_care",
                    store_action="dont_care",
                ),
                MemorylessAttachment(
                    name="b",
                    format="RGBA8",
                    load_action="dont_care",
                    store_action="dont_care",
                ),
            ],
        )
        savings = config.total_bandwidth_savings(1920, 1080)
        expected = 2 * 1920 * 1080 * 4
        assert savings == expected


class TestMetalCapabilities:
    """Test MetalCapabilities dataclass."""

    def test_apple_silicon_capabilities(self):
        """Test Apple Silicon capabilities."""
        caps = MetalCapabilities.apple_silicon()
        assert caps.feature_level == MetalFeatureLevel.METAL_3
        assert caps.unified_memory
        assert caps.supports_ray_tracing
        assert caps.supports_mesh_shaders
        assert caps.supports_tile_shaders
        assert caps.argument_buffer_tier == 2

    def test_intel_mac_capabilities(self):
        """Test Intel Mac capabilities."""
        caps = MetalCapabilities.intel_mac()
        assert caps.feature_level == MetalFeatureLevel.METAL_2
        assert not caps.unified_memory
        assert not caps.supports_ray_tracing
        assert not caps.supports_tile_shaders

    def test_ios_capabilities(self):
        """Test iOS device capabilities."""
        caps = MetalCapabilities.ios_device()
        assert caps.feature_level == MetalFeatureLevel.METAL_2
        assert caps.unified_memory
        assert caps.supports_tile_shaders
        assert not caps.supports_ray_tracing

    def test_can_use_memoryless(self):
        """Test memoryless attachment support detection."""
        apple = MetalCapabilities.apple_silicon()
        assert apple.can_use_memoryless()

        intel = MetalCapabilities.intel_mac()
        assert not intel.can_use_memoryless()

    def test_can_use_bindless(self):
        """Test bindless resource support detection."""
        caps = MetalCapabilities.apple_silicon()
        assert caps.can_use_bindless()


class TestMetalOptimizerCreation:
    """Test MetalOptimizer creation."""

    def test_optimizer_default_creation(self):
        """Test default optimizer creation."""
        optimizer = MetalOptimizer()
        assert optimizer.capabilities.feature_level == MetalFeatureLevel.METAL_3

    def test_optimizer_with_capabilities(self):
        """Test optimizer with explicit capabilities."""
        caps = MetalCapabilities.intel_mac()
        optimizer = MetalOptimizer(capabilities=caps)
        assert optimizer.capabilities.unified_memory is False

    def test_optimizer_with_quality_tier(self):
        """Test optimizer with quality tier."""
        optimizer = MetalOptimizer(quality_tier=QualityTier.LOW)
        # Optimizations should still be enabled based on capabilities
        assert len(optimizer.enabled_optimizations) > 0


class TestMetalOptimizerOptimizations:
    """Test MetalOptimizer optimization flags."""

    def test_apple_silicon_optimizations(self):
        """Test Apple Silicon has all optimizations."""
        optimizer = MetalOptimizer(capabilities=MetalCapabilities.apple_silicon())
        opts = optimizer.enabled_optimizations

        assert TBDROptimization.MEMORYLESS_ATTACHMENTS in opts
        assert TBDROptimization.TILE_SHADERS in opts
        assert TBDROptimization.IMAGEBLOCK_STORAGE in opts
        assert TBDROptimization.MERGE_PASSES in opts

    def test_intel_mac_optimizations(self):
        """Test Intel Mac has limited optimizations."""
        optimizer = MetalOptimizer(capabilities=MetalCapabilities.intel_mac())
        opts = optimizer.enabled_optimizations

        assert TBDROptimization.MEMORYLESS_ATTACHMENTS not in opts
        assert TBDROptimization.TILE_SHADERS not in opts
        assert TBDROptimization.MERGE_PASSES in opts

    def test_is_optimization_enabled(self):
        """Test individual optimization check."""
        optimizer = MetalOptimizer()
        assert optimizer.is_optimization_enabled(TBDROptimization.MERGE_PASSES)


class TestMetalOptimizerAttachments:
    """Test MetalOptimizer attachment creation."""

    def test_create_memoryless_depth(self):
        """Test memoryless depth attachment creation."""
        optimizer = MetalOptimizer()
        depth = optimizer.create_memoryless_depth()

        assert depth.name == "depth"
        assert depth.format == "D32F"
        assert depth.usage == "depth_stencil"
        assert depth.is_truly_memoryless()

    def test_create_memoryless_gbuffer(self):
        """Test memoryless G-buffer attachment creation."""
        optimizer = MetalOptimizer()
        albedo = optimizer.create_memoryless_gbuffer("albedo", "RGBA8")

        assert albedo.name == "albedo"
        assert albedo.format == "RGBA8"
        assert albedo.is_truly_memoryless()


class TestMetalOptimizerTileShaders:
    """Test MetalOptimizer tile shader creation."""

    def test_create_deferred_tile_shader(self):
        """Test deferred tile shader creation."""
        optimizer = MetalOptimizer()
        config = optimizer.create_deferred_tile_shader()

        assert config.name == "deferred_lighting"
        assert config.tile_width == 32
        assert config.tile_height == 32
        assert config.uses_imageblock


class TestMetalOptimizerArgumentBuffers:
    """Test MetalOptimizer argument buffer creation."""

    def test_create_argument_buffer(self):
        """Test argument buffer creation."""
        optimizer = MetalOptimizer()
        config = optimizer.create_argument_buffer("materials", 256)

        assert config.name == "materials"
        assert config.max_textures == 256
        assert config.supports_bindless()

    def test_argument_buffer_respects_caps(self):
        """Test argument buffer respects capability limits."""
        caps = MetalCapabilities(
            feature_level=MetalFeatureLevel.METAL_2,
            argument_buffer_tier=1,
            max_argument_buffer_textures=128,
        )
        optimizer = MetalOptimizer(capabilities=caps)
        config = optimizer.create_argument_buffer("test", 500)

        assert config.max_textures == 128
        assert config.tier == 1


class TestMetalOptimizerPasses:
    """Test MetalOptimizer render pass creation."""

    def test_optimize_gbuffer_pass(self):
        """Test G-buffer pass optimization."""
        optimizer = MetalOptimizer()
        config = optimizer.optimize_gbuffer_pass()

        assert config.name == "gbuffer"
        assert len(config.memoryless_attachments) == 4

    def test_optimize_deferred_lighting_pass(self):
        """Test deferred lighting pass optimization."""
        optimizer = MetalOptimizer()
        config = optimizer.optimize_deferred_lighting_pass()

        assert config.name == "deferred_lighting"
        assert config.tile_shader is not None
        assert config.merge_with_previous
        assert config.argument_buffer is not None


class TestMetalOptimizerRenderGraph:
    """Test MetalOptimizer render graph optimization."""

    def test_optimize_render_graph(self):
        """Test render graph optimization."""
        optimizer = MetalOptimizer()
        passes = [
            {
                "name": "depth",
                "attachments": [
                    {"name": "depth", "format": "D32F", "transient": True},
                ],
            },
            {
                "name": "gbuffer",
                "attachments": [
                    {"name": "albedo", "format": "RGBA8", "transient": True},
                    {"name": "normal", "format": "RGBA16F", "transient": True},
                ],
                "can_merge": True,
            },
        ]

        optimized = optimizer.optimize_render_graph(passes)

        assert len(optimized) == 2
        assert optimized[0].name == "depth"
        assert len(optimized[0].memoryless_attachments) == 1
        assert optimized[1].merge_with_previous


class TestMetalOptimizerBandwidth:
    """Test MetalOptimizer bandwidth estimation."""

    def test_estimate_bandwidth_savings(self):
        """Test bandwidth savings estimation."""
        optimizer = MetalOptimizer()
        passes = [
            RenderPassConfig(
                name="gbuffer",
                memoryless_attachments=[
                    MemorylessAttachment("albedo", "RGBA8"),
                    MemorylessAttachment("normal", "RGBA16F"),
                    MemorylessAttachment("depth", "D32F"),
                ],
            ),
            RenderPassConfig(
                name="lighting",
                merge_with_previous=True,
            ),
        ]

        stats = optimizer.estimate_bandwidth_savings(passes, 1920, 1080)

        assert stats["memoryless_savings_bytes"] > 0
        assert stats["memoryless_savings_mb"] > 0
        assert stats["merged_passes"] == 1
        assert "per_pass_savings" in stats


class TestCreateOptimizerForDevice:
    """Test factory function."""

    def test_create_apple_silicon(self):
        """Test creating Apple Silicon optimizer."""
        optimizer = create_optimizer_for_device("apple_silicon")
        assert optimizer.capabilities.supports_ray_tracing

    def test_create_intel_mac(self):
        """Test creating Intel Mac optimizer."""
        optimizer = create_optimizer_for_device("intel_mac")
        assert not optimizer.capabilities.unified_memory

    def test_create_ios(self):
        """Test creating iOS optimizer."""
        optimizer = create_optimizer_for_device("ios")
        assert optimizer.capabilities.supports_tile_shaders

    def test_create_with_quality_tier(self):
        """Test creating optimizer with quality tier."""
        optimizer = create_optimizer_for_device(
            "apple_silicon",
            quality_tier=QualityTier.ULTRA,
        )
        assert optimizer._quality_tier == QualityTier.ULTRA


class TestIntegration:
    """Integration tests for Metal optimizations."""

    def test_full_deferred_pipeline(self):
        """Test full deferred rendering pipeline optimization."""
        optimizer = MetalOptimizer()

        # Create optimized passes
        gbuffer = optimizer.optimize_gbuffer_pass()
        lighting = optimizer.optimize_deferred_lighting_pass()

        # Verify pass configuration
        assert len(gbuffer.memoryless_attachments) == 4
        assert lighting.merge_with_previous
        assert lighting.tile_shader is not None

        # Estimate savings
        stats = optimizer.estimate_bandwidth_savings(
            [gbuffer, lighting],
            1920, 1080,
        )

        # Should have significant savings
        assert stats["total_savings_mb"] > 10  # At least 10MB saved

    def test_quality_tier_impact(self):
        """Test quality tier impacts optimization."""
        low = MetalOptimizer(quality_tier=QualityTier.LOW)
        ultra = MetalOptimizer(quality_tier=QualityTier.ULTRA)

        # Both should have core optimizations
        assert TBDROptimization.MERGE_PASSES in low.enabled_optimizations
        assert TBDROptimization.MERGE_PASSES in ultra.enabled_optimizations
