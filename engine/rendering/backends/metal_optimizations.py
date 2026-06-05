"""Metal-specific rendering optimizations (T-CC-0.12).

Provides Metal backend optimizations including:
- TBDR (Tile-Based Deferred Rendering) awareness
- Argument buffer encoding
- Unified memory optimization
- Metal-specific resource management

These optimizations target Apple Silicon (M1/M2/M3) GPUs which use
a tile-based architecture different from traditional GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from trinity.types import QualityTier

__all__ = [
    "MetalFeatureLevel",
    "TBDROptimization",
    "ArgumentBufferConfig",
    "MemorylessAttachment",
    "TileShaderConfig",
    "MetalCapabilities",
    "MetalOptimizer",
    "RenderPassConfig",
]


class MetalFeatureLevel(Enum):
    """Metal feature set levels."""

    METAL_1 = "metal_1"  # A7-A10 GPUs
    METAL_2 = "metal_2"  # A11+ GPUs, unified memory
    METAL_3 = "metal_3"  # M1/M2/M3, mesh shaders, ray tracing


class TBDROptimization(Enum):
    """TBDR-specific optimization flags."""

    MEMORYLESS_ATTACHMENTS = auto()
    TILE_SHADERS = auto()
    IMAGEBLOCK_STORAGE = auto()
    DEFERRED_STORE_ACTIONS = auto()
    MERGE_PASSES = auto()
    EXPLICIT_LOAD_STORE = auto()


@dataclass(slots=True)
class MemorylessAttachment:
    """Configuration for memoryless render target.

    Memoryless attachments exist only in tile memory and are
    never written to system memory, saving bandwidth.
    """

    name: str
    format: str
    usage: str = "transient"  # "transient", "depth_stencil", "msaa_resolve"
    clear_color: tuple[float, float, float, float] | None = None
    load_action: str = "dont_care"  # "dont_care", "clear", "load"
    store_action: str = "dont_care"  # "dont_care", "store", "resolve"

    def is_truly_memoryless(self) -> bool:
        """Check if attachment can be purely in tile memory."""
        return (
            self.load_action in ("dont_care", "clear") and
            self.store_action == "dont_care"
        )

    def estimated_bandwidth_savings(self, width: int, height: int) -> int:
        """Estimate bandwidth savings in bytes per frame."""
        format_sizes = {
            "RGBA8": 4,
            "RGBA16F": 8,
            "RGBA32F": 16,
            "R32F": 4,
            "D32F": 4,
            "D24S8": 4,
        }
        bytes_per_pixel = format_sizes.get(self.format, 4)
        return width * height * bytes_per_pixel


@dataclass(slots=True)
class TileShaderConfig:
    """Configuration for Metal tile shaders.

    Tile shaders operate directly on tile memory, enabling
    efficient deferred shading within a single render pass.
    """

    name: str
    tile_width: int = 32
    tile_height: int = 32
    threadgroup_memory_bytes: int = 16384
    imageblock_sample_length: int = 4
    uses_imageblock: bool = True

    @property
    def threads_per_tile(self) -> int:
        """Get total threads per tile."""
        return self.tile_width * self.tile_height

    def can_fit_in_tile_memory(self, data_bytes: int) -> bool:
        """Check if data fits in tile memory."""
        return data_bytes <= self.threadgroup_memory_bytes


@dataclass(slots=True)
class ArgumentBufferConfig:
    """Configuration for Metal argument buffers.

    Argument buffers bundle multiple resources into a single
    buffer that can be bound efficiently.
    """

    name: str
    tier: int = 2  # 1 or 2 (Tier 2 = unlimited resources)
    max_textures: int = 128
    max_samplers: int = 16
    max_buffers: int = 64
    uses_heaps: bool = True
    residency_managed: bool = True

    def supports_bindless(self) -> bool:
        """Check if configuration supports bindless resources."""
        return self.tier == 2 and self.uses_heaps


@dataclass(slots=True)
class RenderPassConfig:
    """Metal-optimized render pass configuration."""

    name: str
    memoryless_attachments: list[MemorylessAttachment] = field(default_factory=list)
    tile_shader: TileShaderConfig | None = None
    argument_buffer: ArgumentBufferConfig | None = None
    merge_with_previous: bool = False
    uses_programmable_sample_positions: bool = False

    def total_bandwidth_savings(self, width: int, height: int) -> int:
        """Calculate total bandwidth savings from memoryless attachments."""
        return sum(
            att.estimated_bandwidth_savings(width, height)
            for att in self.memoryless_attachments
            if att.is_truly_memoryless()
        )


@dataclass(slots=True)
class MetalCapabilities:
    """Detected Metal GPU capabilities."""

    feature_level: MetalFeatureLevel = MetalFeatureLevel.METAL_2
    unified_memory: bool = True
    max_threadgroup_memory: int = 32768
    max_threads_per_threadgroup: int = 1024
    supports_ray_tracing: bool = False
    supports_mesh_shaders: bool = False
    supports_tile_shaders: bool = True
    argument_buffer_tier: int = 2
    max_argument_buffer_textures: int = 500000
    simd_width: int = 32
    gpu_family: str = "apple"  # "apple" or "mac"

    @classmethod
    def apple_silicon(cls) -> MetalCapabilities:
        """Create capabilities for Apple Silicon (M1/M2/M3)."""
        return cls(
            feature_level=MetalFeatureLevel.METAL_3,
            unified_memory=True,
            max_threadgroup_memory=32768,
            max_threads_per_threadgroup=1024,
            supports_ray_tracing=True,
            supports_mesh_shaders=True,
            supports_tile_shaders=True,
            argument_buffer_tier=2,
            max_argument_buffer_textures=500000,
            simd_width=32,
            gpu_family="apple",
        )

    @classmethod
    def intel_mac(cls) -> MetalCapabilities:
        """Create capabilities for Intel Mac with discrete GPU."""
        return cls(
            feature_level=MetalFeatureLevel.METAL_2,
            unified_memory=False,
            max_threadgroup_memory=16384,
            max_threads_per_threadgroup=1024,
            supports_ray_tracing=False,
            supports_mesh_shaders=False,
            supports_tile_shaders=False,
            argument_buffer_tier=2,
            max_argument_buffer_textures=128,
            simd_width=64,
            gpu_family="mac",
        )

    @classmethod
    def ios_device(cls) -> MetalCapabilities:
        """Create capabilities for iOS device (A12+)."""
        return cls(
            feature_level=MetalFeatureLevel.METAL_2,
            unified_memory=True,
            max_threadgroup_memory=16384,
            max_threads_per_threadgroup=512,
            supports_ray_tracing=False,
            supports_mesh_shaders=False,
            supports_tile_shaders=True,
            argument_buffer_tier=2,
            max_argument_buffer_textures=500000,
            simd_width=32,
            gpu_family="apple",
        )

    def can_use_memoryless(self) -> bool:
        """Check if device supports memoryless attachments."""
        return self.unified_memory and self.supports_tile_shaders

    def can_use_tile_shaders(self) -> bool:
        """Check if device supports tile shaders."""
        return self.supports_tile_shaders

    def can_use_bindless(self) -> bool:
        """Check if device supports bindless resources."""
        return self.argument_buffer_tier == 2


class MetalOptimizer:
    """
    Optimizes render passes for Metal TBDR architecture.

    Applies optimizations:
    1. Memoryless attachments for transient render targets
    2. Pass merging to reduce tile memory flushes
    3. Tile shaders for deferred operations
    4. Argument buffer encoding for efficient binding
    """

    __slots__ = ("_capabilities", "_quality_tier", "_optimizations")

    def __init__(
        self,
        capabilities: MetalCapabilities | None = None,
        quality_tier: QualityTier = QualityTier.HIGH,
    ):
        self._capabilities = capabilities or MetalCapabilities.apple_silicon()
        self._quality_tier = quality_tier
        self._optimizations: set[TBDROptimization] = set()
        self._configure_optimizations()

    def _configure_optimizations(self) -> None:
        """Configure enabled optimizations based on capabilities."""
        caps = self._capabilities

        if caps.can_use_memoryless():
            self._optimizations.add(TBDROptimization.MEMORYLESS_ATTACHMENTS)
            self._optimizations.add(TBDROptimization.DEFERRED_STORE_ACTIONS)

        if caps.can_use_tile_shaders():
            self._optimizations.add(TBDROptimization.TILE_SHADERS)
            self._optimizations.add(TBDROptimization.IMAGEBLOCK_STORAGE)

        self._optimizations.add(TBDROptimization.MERGE_PASSES)
        self._optimizations.add(TBDROptimization.EXPLICIT_LOAD_STORE)

    @property
    def capabilities(self) -> MetalCapabilities:
        """Get Metal capabilities."""
        return self._capabilities

    @property
    def enabled_optimizations(self) -> set[TBDROptimization]:
        """Get enabled optimizations."""
        return self._optimizations.copy()

    def is_optimization_enabled(self, opt: TBDROptimization) -> bool:
        """Check if an optimization is enabled."""
        return opt in self._optimizations

    def create_memoryless_depth(self, name: str = "depth") -> MemorylessAttachment:
        """Create a memoryless depth attachment."""
        return MemorylessAttachment(
            name=name,
            format="D32F",
            usage="depth_stencil",
            load_action="clear",
            store_action="dont_care",
            clear_color=(1.0, 0.0, 0.0, 0.0),
        )

    def create_memoryless_gbuffer(
        self,
        name: str,
        format: str = "RGBA16F",
    ) -> MemorylessAttachment:
        """Create a memoryless G-buffer attachment."""
        return MemorylessAttachment(
            name=name,
            format=format,
            usage="transient",
            load_action="dont_care",
            store_action="dont_care",
        )

    def create_deferred_tile_shader(self) -> TileShaderConfig:
        """Create tile shader config for deferred lighting."""
        return TileShaderConfig(
            name="deferred_lighting",
            tile_width=32,
            tile_height=32,
            threadgroup_memory_bytes=16384,
            imageblock_sample_length=4,
            uses_imageblock=True,
        )

    def create_argument_buffer(
        self,
        name: str,
        texture_count: int = 128,
    ) -> ArgumentBufferConfig:
        """Create argument buffer configuration."""
        caps = self._capabilities
        return ArgumentBufferConfig(
            name=name,
            tier=caps.argument_buffer_tier,
            max_textures=min(texture_count, caps.max_argument_buffer_textures),
            max_samplers=16,
            max_buffers=64,
            uses_heaps=caps.argument_buffer_tier == 2,
            residency_managed=True,
        )

    def optimize_gbuffer_pass(self) -> RenderPassConfig:
        """Create optimized G-buffer render pass."""
        config = RenderPassConfig(name="gbuffer")

        if self.is_optimization_enabled(TBDROptimization.MEMORYLESS_ATTACHMENTS):
            config.memoryless_attachments = [
                self.create_memoryless_gbuffer("albedo", "RGBA8"),
                self.create_memoryless_gbuffer("normal", "RGBA16F"),
                self.create_memoryless_gbuffer("material", "RGBA8"),
                self.create_memoryless_depth("depth"),
            ]

        return config

    def optimize_deferred_lighting_pass(self) -> RenderPassConfig:
        """Create optimized deferred lighting pass."""
        config = RenderPassConfig(name="deferred_lighting")

        if self.is_optimization_enabled(TBDROptimization.TILE_SHADERS):
            config.tile_shader = self.create_deferred_tile_shader()
            config.merge_with_previous = True  # Merge with G-buffer pass

        config.argument_buffer = self.create_argument_buffer("lights", 1024)

        return config

    def optimize_render_graph(
        self,
        passes: list[dict[str, Any]],
    ) -> list[RenderPassConfig]:
        """Optimize a render graph for Metal TBDR.

        Args:
            passes: List of render pass descriptions

        Returns:
            List of optimized render pass configurations
        """
        optimized = []

        for i, pass_desc in enumerate(passes):
            name = pass_desc.get("name", f"pass_{i}")
            attachments = pass_desc.get("attachments", [])

            config = RenderPassConfig(name=name)

            # Apply memoryless optimizations to transient attachments
            if self.is_optimization_enabled(TBDROptimization.MEMORYLESS_ATTACHMENTS):
                for att in attachments:
                    if att.get("transient", False):
                        config.memoryless_attachments.append(
                            MemorylessAttachment(
                                name=att["name"],
                                format=att.get("format", "RGBA8"),
                                usage="transient",
                            )
                        )

            # Check if we can merge with previous pass
            if (
                i > 0 and
                self.is_optimization_enabled(TBDROptimization.MERGE_PASSES) and
                pass_desc.get("can_merge", False)
            ):
                config.merge_with_previous = True

            optimized.append(config)

        return optimized

    def estimate_bandwidth_savings(
        self,
        passes: list[RenderPassConfig],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Estimate bandwidth savings from optimizations.

        Returns:
            Dictionary with savings statistics
        """
        total_savings = 0
        per_pass_savings = {}

        for pass_config in passes:
            savings = pass_config.total_bandwidth_savings(width, height)
            per_pass_savings[pass_config.name] = savings
            total_savings += savings

        # Additional savings from pass merging
        merged_passes = sum(1 for p in passes if p.merge_with_previous)
        # Each merge saves ~tile_memory_size * tile_count flushes
        tiles_x = (width + 31) // 32
        tiles_y = (height + 31) // 32
        tile_flush_savings = merged_passes * tiles_x * tiles_y * 4096

        return {
            "memoryless_savings_bytes": total_savings,
            "memoryless_savings_mb": round(total_savings / (1024 * 1024), 2),
            "merged_passes": merged_passes,
            "tile_flush_savings_bytes": tile_flush_savings,
            "total_savings_mb": round(
                (total_savings + tile_flush_savings) / (1024 * 1024), 2
            ),
            "per_pass_savings": per_pass_savings,
        }


def create_optimizer_for_device(
    device_type: str = "apple_silicon",
    quality_tier: QualityTier = QualityTier.HIGH,
) -> MetalOptimizer:
    """Factory function to create optimizer for device type."""
    capabilities_map = {
        "apple_silicon": MetalCapabilities.apple_silicon,
        "intel_mac": MetalCapabilities.intel_mac,
        "ios": MetalCapabilities.ios_device,
    }
    caps_factory = capabilities_map.get(device_type, MetalCapabilities.apple_silicon)
    return MetalOptimizer(capabilities=caps_factory(), quality_tier=quality_tier)
