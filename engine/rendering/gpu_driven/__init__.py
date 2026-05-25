"""
GPU-Driven Rendering Subsystem.

Implements GPU-driven rendering techniques for efficient handling of
large numbers of objects with minimal CPU overhead.

Subsystems:
- culling: GPU frustum, occlusion (HZB), and distance culling
- meshlet: Meshlet/cluster system with normal cone culling
- indirect_draw: Indirect draw command generation
- instancing: Instance batching and multi-draw indirect
- visibility_buffer: Nanite-style visibility buffer pipeline
- bindless: Bindless resource management

References:
- RENDERING_CONTEXT.md Section 6.2 GPU-Driven Rendering Pipeline
"""

# Culling pipeline
from engine.rendering.gpu_driven.culling import (
    # Constants
    CullingConstants,
    # Math types
    Vec3,
    Vec4,
    AABB,
    BoundingSphere,
    # Frustum
    Frustum,
    FrustumPlane,
    # Instance data
    InstanceBounds,
    CullResult,
    CullingStats,
    # Cullers
    Culler,
    FrustumCuller,
    OcclusionCuller,
    DistanceCuller,
    SmallFeatureCuller,
    # Configs
    HZBConfig,
    DistanceCullConfig,
    SmallFeatureCullConfig,
    # Pipeline
    CullingPipeline,
)

# Meshlet system
from engine.rendering.gpu_driven.meshlet import (
    # Constants
    MeshletConstants,
    # Data structures
    MeshletBounds,
    Meshlet,
    Vertex,
    # Builder
    MeshletBuildConfig,
    MeshletBuilder,
    # Culling
    MeshletCuller,
    # LOD
    MeshletLODLevel,
    MeshletLODChain,
    # Mesh
    MeshletMesh,
)

# Indirect draw
from engine.rendering.gpu_driven.indirect_draw import (
    # Argument structures
    DrawIndexedIndirectArgs,
    DrawIndirectArgs,
    DispatchIndirectArgs,
    # Draw command
    DrawCommandType,
    DrawCommand,
    # Buffers
    IndirectDrawBufferConfig,
    IndirectDrawBuffer,
    MultiDrawIndirectBuffer,
    # Batch info
    MeshBatchInfo,
    InstanceInfo,
    # Generation
    DrawCommandGenerator,
    DrawCommandCompactor,
)

# Instancing
from engine.rendering.gpu_driven.instancing import (
    # Transform
    Mat4x4,
    # Instance data
    InstanceData,
    # Batching
    BatchKey,
    InstanceBatch,
    InstanceBatcher,
    # Multi-draw
    MultiDrawIndirectManager,
    # Culling integration
    CulledInstanceBatcher,
)

# Visibility buffer
from engine.rendering.gpu_driven.visibility_buffer import (
    # Constants
    VisibilityBufferConstants,
    # Data formats
    VisibilityBufferFormat,
    VisibilityData,
    # Buffer
    VisibilityBufferConfig,
    VisibilityBuffer,
    # Tile classification
    MaterialTile,
    MaterialTileClassifier,
    # Passes
    VisibilityBufferPass,
    ShadingInput,
    ShadingOutput,
    DeferredTexturingPass,
    MaterialSortingPass,
    # Pipeline
    VisibilityBufferPipeline,
)

# Bindless resources
from engine.rendering.gpu_driven.bindless import (
    # Resource types
    ResourceType,
    ResourceHandle,
    # Texture
    TextureFormat,
    TextureDescriptor,
    BindlessTextureManagerConfig,
    BindlessTextureManager,
    # Buffer
    BufferUsage,
    BufferDescriptor,
    BindlessBufferManagerConfig,
    BindlessBufferManager,
    # Sampler
    FilterMode,
    AddressMode,
    SamplerDescriptor,
    # Material
    MaterialResources,
    MaterialResourceTable,
    # System
    BindlessResourceSystem,
)

__all__ = [
    # === Culling ===
    # Constants
    "CullingConstants",
    # Math types
    "Vec3",
    "Vec4",
    "AABB",
    "BoundingSphere",
    # Frustum
    "Frustum",
    "FrustumPlane",
    # Instance data
    "InstanceBounds",
    "CullResult",
    "CullingStats",
    # Cullers
    "Culler",
    "FrustumCuller",
    "OcclusionCuller",
    "DistanceCuller",
    "SmallFeatureCuller",
    # Configs
    "HZBConfig",
    "DistanceCullConfig",
    "SmallFeatureCullConfig",
    # Pipeline
    "CullingPipeline",

    # === Meshlet ===
    # Constants
    "MeshletConstants",
    # Data structures
    "MeshletBounds",
    "Meshlet",
    "Vertex",
    # Builder
    "MeshletBuildConfig",
    "MeshletBuilder",
    # Culling
    "MeshletCuller",
    # LOD
    "MeshletLODLevel",
    "MeshletLODChain",
    # Mesh
    "MeshletMesh",

    # === Indirect Draw ===
    # Argument structures
    "DrawIndexedIndirectArgs",
    "DrawIndirectArgs",
    "DispatchIndirectArgs",
    # Draw command
    "DrawCommandType",
    "DrawCommand",
    # Buffers
    "IndirectDrawBufferConfig",
    "IndirectDrawBuffer",
    "MultiDrawIndirectBuffer",
    # Batch info
    "MeshBatchInfo",
    "InstanceInfo",
    # Generation
    "DrawCommandGenerator",
    "DrawCommandCompactor",

    # === Instancing ===
    # Transform
    "Mat4x4",
    # Instance data
    "InstanceData",
    # Batching
    "BatchKey",
    "InstanceBatch",
    "InstanceBatcher",
    # Multi-draw
    "MultiDrawIndirectManager",
    # Culling integration
    "CulledInstanceBatcher",

    # === Visibility Buffer ===
    # Constants
    "VisibilityBufferConstants",
    # Data formats
    "VisibilityBufferFormat",
    "VisibilityData",
    # Buffer
    "VisibilityBufferConfig",
    "VisibilityBuffer",
    # Tile classification
    "MaterialTile",
    "MaterialTileClassifier",
    # Passes
    "VisibilityBufferPass",
    "ShadingInput",
    "ShadingOutput",
    "DeferredTexturingPass",
    "MaterialSortingPass",
    # Pipeline
    "VisibilityBufferPipeline",

    # === Bindless ===
    # Resource types
    "ResourceType",
    "ResourceHandle",
    # Texture
    "TextureFormat",
    "TextureDescriptor",
    "BindlessTextureManagerConfig",
    "BindlessTextureManager",
    # Buffer
    "BufferUsage",
    "BufferDescriptor",
    "BindlessBufferManagerConfig",
    "BindlessBufferManager",
    # Sampler
    "FilterMode",
    "AddressMode",
    "SamplerDescriptor",
    # Material
    "MaterialResources",
    "MaterialResourceTable",
    # System
    "BindlessResourceSystem",
]
