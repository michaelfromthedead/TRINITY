"""
Frame Graph subsystem for the rendering layer.

This module provides the frame graph implementation as specified in
RENDERING_CONTEXT.md Section 6.1.

The frame graph manages:
- Render pass declaration and dependency tracking
- Resource allocation and memory aliasing
- Automatic barrier insertion between passes
- Async compute scheduling
- Dead code elimination (unused pass culling)

Core Components:
    FrameGraph: Main class for building and executing render graphs
    PassNode: Base class for render passes (Graphics, Compute, Copy, RT)
    ResourceManager: Handles resource allocation and aliasing
    BarrierManager: Automatic GPU barrier insertion
    AsyncScheduler: Async compute scheduling for parallel execution

Example Usage:
    from engine.rendering.framegraph import FrameGraph, ResourceFormat

    fg = FrameGraph()

    # Create resources
    gbuffer_albedo = fg.create_texture(
        "gbuffer_albedo",
        format=ResourceFormat.R8G8B8A8_UNORM,
    )
    hdr_target = fg.create_texture(
        "hdr_target",
        format=ResourceFormat.R16G16B16A16_FLOAT,
    )

    # Declare passes
    gbuffer = fg.add_graphics_pass("GBuffer")
    gbuffer.add_color_attachment(gbuffer_albedo)

    lighting = fg.add_compute_pass("Lighting")
    lighting.read_texture(gbuffer_albedo)
    lighting.write_texture(hdr_target)

    # Compile and execute
    fg.compile()
    fg.execute(render_context)

See Also:
    - RENDERING_CONTEXT.md Section 6.1 for full specification
    - Section 5.1 TODO for implementation checklist
"""

from .frame_graph import CompilationResult, CompileError, FrameGraph
from .pass_node import (
    ColorAttachment,
    ComputePass,
    CopyPass,
    DepthStencilAttachment,
    GraphicsPass,
    PassFlags,
    PassNode,
    PassType,
    RayTracingPass,
    ResourceAccess,
    create_pass,
)
from .resource_manager import (
    ExternalResource,
    HistoryResource,
    ResourceDescriptor,
    ResourceFormat,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
    TransientResource,
)
from .barrier_manager import (
    AccessFlags,
    Barrier,
    BarrierBatch,
    BarrierManager,
    BarrierType,
    PipelineStage,
    ResourceStateTracker,
)
from .async_scheduler import (
    AsyncScheduler,
    QueueTimeline,
    QueueType,
    ScheduledPass,
    SyncPoint,
    identify_async_candidates,
)
from .config import (
    AsyncSchedulerConfig,
    FrameGraphConfig,
    ResourceManagerConfig,
    ASYNC_SCHEDULER_CONFIG,
    FRAME_GRAPH_CONFIG,
    RESOURCE_MANAGER_CONFIG,
)
from .context import AllocationHandle, FenceOp, RHIContext

__all__ = [
    # Frame Graph
    "FrameGraph",
    "CompilationResult",
    "CompileError",
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
    # Configuration
    "AsyncSchedulerConfig",
    "FrameGraphConfig",
    "ResourceManagerConfig",
    "ASYNC_SCHEDULER_CONFIG",
    "FRAME_GRAPH_CONFIG",
    "RESOURCE_MANAGER_CONFIG",
    # Context Protocol
    "RHIContext",
    "AllocationHandle",
    "FenceOp",
]
