"""Rendering backends for TRINITY engine.

This package contains rendering backends for different quality tiers:
- Forward+ (Low/Medium tier): Tile-based forward lighting
- Metal Optimizations: Apple platform-specific optimizations

Exports:
    ForwardPlusRenderer: Forward+ renderer for low-tier quality settings
    ForwardPlusConfig: Configuration for Forward+ renderer
    ForwardPassType: Forward rendering pass types
    ToneMapOperator: Tone mapping operator types
    LightTile: Screen-space tile for light culling
    ForwardPlusPass: Render pass descriptor
    ForwardPlusStats: Runtime statistics
    LightData: Light data structure
    create_forward_plus_for_tier: Factory function
    get_tier_max_lights: Get max lights for tier
    MetalOptimizer: Metal-specific rendering optimizations
    MetalCapabilities: Metal device capabilities
    MetalFeatureLevel: Metal feature level enum
"""

from engine.rendering.backends.forward_plus_renderer import (
    ForwardPassType,
    ToneMapOperator,
    ForwardPlusConfig,
    LightTile,
    ForwardPlusPass,
    ForwardPlusStats,
    LightData,
    ForwardPlusRenderer,
    create_forward_plus_for_tier,
    get_tier_max_lights,
    MAX_LIGHTS_LOW_TIER,
    MAX_LIGHTS_MEDIUM_TIER,
    MAX_LIGHTS_HIGH_TIER,
    TILE_SIZE,
    MAX_LIGHTS_PER_TILE,
)

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

__all__ = [
    # Forward+ Renderer (T-CC-0.10)
    "ForwardPassType",
    "ToneMapOperator",
    "ForwardPlusConfig",
    "LightTile",
    "ForwardPlusPass",
    "ForwardPlusStats",
    "LightData",
    "ForwardPlusRenderer",
    "create_forward_plus_for_tier",
    "get_tier_max_lights",
    "MAX_LIGHTS_LOW_TIER",
    "MAX_LIGHTS_MEDIUM_TIER",
    "MAX_LIGHTS_HIGH_TIER",
    "TILE_SIZE",
    "MAX_LIGHTS_PER_TILE",
    # Metal Optimizations (T-CC-0.12)
    "MetalFeatureLevel",
    "TBDROptimization",
    "ArgumentBufferConfig",
    "MemorylessAttachment",
    "TileShaderConfig",
    "MetalCapabilities",
    "MetalOptimizer",
    "RenderPassConfig",
    "create_optimizer_for_device",
]
