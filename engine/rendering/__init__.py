"""
Rendering layer for the game engine.

This module provides the complete rendering system as specified in
RENDERING_CONTEXT.md, transforming scene data into visual frames.

Subsystems:
    1. Frame Graph - Render pass declaration, resource aliasing, barriers
    2. GPU-Driven Rendering - Indirect draws, GPU culling, instancing
    3. Materials & Shading - PBR, material instances, shader variants
    4. Lighting & GI - Direct lights, shadows, global illumination
    5. Post-Processing - Tonemapping, bloom, DOF, TAA, upscaling
    6. Particles & VFX - GPU particles, trails, decals
    7. Denoise - A-Trous wavelet spatial denoising for ray tracing

See RENDERING_CONTEXT.md for complete implementation reference.
"""

# Denoise - A-Trous Wavelet Spatial Denoiser
from .denoise import (
    ATrousDenoiser,
    DenoiseConfig,
    DenoiseQuality,
    DenoiseTarget,
    EdgeStopFunctions,
    YCoCgConverter,
    PSNRMetrics,
)

# Frame Graph - Core rendering orchestration
from .framegraph import (
    # Frame Graph
    FrameGraph,
    CompilationResult,
    # Pass Types
    PassNode,
    PassType,
    PassFlags,
    GraphicsPass,
    ComputePass,
    CopyPass,
    RayTracingPass,
    ResourceAccess,
    ColorAttachment,
    DepthStencilAttachment,
    create_pass,
    # Resource Management
    ResourceManager,
    ResourceHandle,
    ResourceDescriptor,
    ResourceType,
    ResourceFormat,
    ResourceState,
    TransientResource,
    HistoryResource,
    ExternalResource,
    # Barrier Management
    BarrierManager,
    Barrier,
    BarrierBatch,
    BarrierType,
    PipelineStage,
    AccessFlags,
    ResourceStateTracker,
    # Async Scheduling
    AsyncScheduler,
    ScheduledPass,
    QueueType,
    QueueTimeline,
    SyncPoint,
    identify_async_candidates,
)

__all__ = [
    # Frame Graph
    "FrameGraph",
    "CompilationResult",
    # Pass Types
    "PassNode",
    "PassType",
    "PassFlags",
    "GraphicsPass",
    "ComputePass",
    "CopyPass",
    "RayTracingPass",
    "ResourceAccess",
    "ColorAttachment",
    "DepthStencilAttachment",
    "create_pass",
    # Resource Management
    "ResourceManager",
    "ResourceHandle",
    "ResourceDescriptor",
    "ResourceType",
    "ResourceFormat",
    "ResourceState",
    "TransientResource",
    "HistoryResource",
    "ExternalResource",
    # Barrier Management
    "BarrierManager",
    "Barrier",
    "BarrierBatch",
    "BarrierType",
    "PipelineStage",
    "AccessFlags",
    "ResourceStateTracker",
    # Async Scheduling
    "AsyncScheduler",
    "ScheduledPass",
    "QueueType",
    "QueueTimeline",
    "SyncPoint",
    "identify_async_candidates",
    # Denoise (A-Trous Wavelet Spatial Denoiser)
    "ATrousDenoiser",
    "DenoiseConfig",
    "DenoiseQuality",
    "DenoiseTarget",
    "EdgeStopFunctions",
    "YCoCgConverter",
    "PSNRMetrics",
]
