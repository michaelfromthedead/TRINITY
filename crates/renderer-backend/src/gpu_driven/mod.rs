//! GPU-driven rendering subsystem.
//!
//! Implements efficient GPU-driven data management for the renderer backend,
//! including buffer staging, indirect draw management, GPU-side culling
//! support, the bindless mesh table, the bindless material table, and the
//! bindless texture table.
//!
//! Sub-modules:
//! - `buffers`         -- BufferRegistry with triple-buffered staging
//! - `mesh_table`      -- Bindless Mesh Table (`array<MeshTableEntry>`) with
//!                        CPU-side manager for load-time population
//! - `material_table`  -- Bindless Material Table (`array<MaterialTableEntry>`)
//!                        with CPU-side manager and dirty-flag tracking
//! - `texture_table`   -- Bindless Texture Table (`texture_2d_array<f32>`) with
//!                        CPU-side manager and free-list allocation

pub mod buffers;
pub mod compact;
pub mod frustum;
pub mod frustum_cull;
pub mod frustum_cull_pipeline;
pub mod hiz_occlusion;
pub mod hiz_pyramid;
pub mod hiz_cull_pipeline;
pub mod hzb;
pub mod indirect_draw;
pub mod lod;
pub mod lod_buffer;
pub mod lod_select;
pub mod material_table;
pub mod mesh_table;
pub mod meshlet;
pub mod meshlet_generator;
pub mod meshlet_cull;
pub mod meshlet_render;
pub mod multi_draw;
pub mod object_data;
pub mod scene_data;
pub mod sort;
pub mod stream_compact;
pub mod texture_table;
pub mod texture_registry;
pub mod buffer_registry;
pub mod bindless_bind_group;
pub mod geometry_path;
pub mod gpu_culling_pipeline;
pub mod build_indirect;
pub mod draw_args;
pub mod visibility_buffer;
pub mod visibility_read;
pub mod visibility_flags;
pub mod distance_cull;
pub mod occlusion_cull;
pub mod small_triangle_cull;
pub mod triangle_cull;
pub mod instance_update;

pub use buffers::{
    AcquireResult, BufferRegistry, BufferSlot, ReleaseResult, SlotState,
    StagingBufferDesc, SubmitResult, NUM_STAGING_SLOTS,
};

pub use material_table::{
    AddEntry as MaterialAddEntry, MaterialTable, MaterialTableEntry,
    RemoveResult as MaterialRemoveResult,
    DEFAULT_MATERIAL_TABLE_CAPACITY, MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_VISIBLE,
    MATERIAL_TABLE_ENTRY_SIZE, NO_TEXTURE,
};

// Aliases for test compatibility (T-WGPU-P6.8.4 naming convention)
pub type MaterialDescriptor = MaterialTableEntry;
pub type GpuMaterialTable = MaterialTable;
pub const MATERIAL_DESCRIPTOR_SIZE: usize = MATERIAL_TABLE_ENTRY_SIZE;
pub const DEFAULT_GPU_MATERIAL_TABLE_CAPACITY: usize = DEFAULT_MATERIAL_TABLE_CAPACITY;
pub const MATERIAL_DESC_FLAG_DOUBLE_SIDED: u32 = 0x0000_0002;
pub const MATERIAL_DESC_FLAG_ALPHA_MASK: u32 = 0x0000_0004;
pub const MATERIAL_DESC_FLAG_ALPHA_BLEND: u32 = 0x0000_0008;
pub const MATERIAL_DESC_FLAG_UNLIT: u32 = 0x0000_0010;

pub use mesh_table::{
    AddEntry, MeshTable, MeshTableEntry, RemoveResult,
    DEFAULT_MESH_TABLE_CAPACITY, MESH_TABLE_ENTRY_SIZE,
};
pub use texture_table::{
    AddEntry as TextureAddEntry, TextureTable, TextureTableEntry,
    RemoveResult as TextureRemoveResult,
    DEFAULT_TEXTURE_TABLE_CAPACITY, MAX_BINDLESS_TEXTURES, TEXTURE_TABLE_ENTRY_SIZE,
};
pub use sort::GpuRadixSort;

// Indirect draw types
pub use indirect_draw::{
    IndirectDrawIndexedArgs, IndirectDrawArgs, IndirectDispatchArgs,
    DrawIndexedIndirectArgs, DrawIndirectArgs, DispatchIndirectArgs,
    IndirectTier, CountBuffer, IndirectDrawBuffer, IndirectDispatchBuffer,
    MultiIndirectConfig, MultiIndirectBuffer, DrawBatchBuilder,
    DEFAULT_MAX_DRAWS, INDIRECT_DRAW_INDEXED_ARGS_SIZE, INDIRECT_DRAW_ARGS_SIZE,
    INDIRECT_DISPATCH_ARGS_SIZE,
};

// Multi-draw
pub use multi_draw::{
    MultiDrawSupport, multi_draw_indirect, multi_draw_indexed_indirect,
    multi_draw_indirect_count, multi_draw_indexed_indirect_count,
    draw_indirect_offset, draw_indexed_indirect_offset,
    buffer_size_for_draws, buffer_size_for_indexed_draws,
    DRAW_INDIRECT_STRIDE, DRAW_INDEXED_INDIRECT_STRIDE,
};
// Multi-draw test helpers (always available for integration tests)
pub use multi_draw::{
    reset_multi_draw_warning, reset_multi_draw_count_warning,
    has_warned_multi_draw_fallback, has_warned_multi_draw_count_fallback,
    trigger_multi_draw_warning, trigger_multi_draw_count_warning,
};

// HiZ pyramid
pub use hiz_pyramid::{
    HiZPyramid, HiZDownsampleParams,
    create_hiz_gen_bind_group_layout, create_hiz_sample_bind_group_layout,
    create_hiz_downsample_texture_layout, create_hiz_downsample_params_layout,
    cpu_max_reduction, calculate_downsample_dispatch,
    HIZ_FORMAT, HIZ_USAGE, MIN_HIZ_SIZE, MAX_HIZ_MIPS,
    HIZ_DOWNSAMPLE_SHADER, HIZ_DOWNSAMPLE_WORKGROUP_SIZE, HIZ_DOWNSAMPLE_PARAMS_SIZE,
};

// HiZ occlusion
pub use hiz_occlusion::{
    HiZOcclusionParams, BatchParams as HiZBatchParams, InputAABB as HiZInputAABB,
    cpu_project_aabb, cpu_select_mip_level, cpu_test_occlusion,
    workgroups_for_objects as hiz_workgroups_for_objects,
    create_hiz_occlusion_texture_layout, create_hiz_occlusion_params_layout,
    create_hiz_occlusion_batch_layout,
    WORKGROUP_SIZE as HIZ_OCCLUSION_WORKGROUP_SIZE,
    HIZ_OCCLUSION_PARAMS_SIZE, INPUT_AABB_SIZE, BATCH_PARAMS_SIZE,
    MAX_MIP_LEVEL, EPSILON as HIZ_EPSILON, CONSERVATIVE_EXPAND,
    HIZ_OCCLUSION_SHADER,
};

// Geometry path
pub use geometry_path::{
    GeometryPath, GeometryPathConfig, GeometryRenderable,
    TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME,
};

// LOD select
pub use lod_select::{
    LodSelectParams, ObjectLodInput, LodSelectOutput, LodBuffer as LodSelectBuffer,
    SelectionMode, object_lod_flags,
    cpu_distance_to_camera, cpu_screen_coverage, cpu_select_lod,
    cpu_select_lod_by_distance as cpu_lod_select_by_distance,
    cpu_select_lod_by_coverage as cpu_lod_select_by_coverage,
    cpu_select_lod_batch, workgroups_for_objects as lod_select_workgroups,
    calculate_dispatch as lod_select_dispatch,
    WORKGROUP_SIZE as LOD_SELECT_WORKGROUP_SIZE, LOD_SELECT_PARAMS_SIZE,
    OBJECT_LOD_INPUT_SIZE, LOD_SELECT_OUTPUT_SIZE, MAX_LOD_LEVELS as LOD_SELECT_MAX_LEVELS,
    DEFAULT_BLEND_RANGE, LOD_SELECT_SHADER,
};

// LOD types and functions
pub use lod::{
    LodDistances, LodParams, LodConfig, LodLevel,
    distance_to_camera, distance_to_camera_squared, screen_coverage,
    select_lod, select_lod_by_distance, select_lod_by_distance_squared,
    select_lod_by_coverage, select_lod_by_coverage_custom, squared_thresholds,
    LOD_DISTANCES_SIZE, LOD_PARAMS_SIZE, MAX_LOD_LEVELS as LOD_MAX_LEVELS,
    DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE,
    COVERAGE_LOD0, COVERAGE_LOD1, COVERAGE_LOD2,
};

// Meshlet types
pub use meshlet::{
    Meshlet, MeshletBounds, MeshletData,
    MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES, MESHLET_SIZE, MESHLET_BOUNDS_SIZE,
};

// Visibility flags
pub use visibility_flags::{
    VisibilityFlagsBuffer, is_visible, set_visible, clear_visible, count_visible,
    words_for_objects, bit_location, cpu_clear_visibility_flags,
    cpu_atomic_or_visibility, cpu_compact_visible,
    BITS_PER_WORD, DEFAULT_VISIBILITY_FLAGS_CAPACITY, MIN_VISIBILITY_FLAGS_CAPACITY,
    WORD_SIZE as VISIBILITY_WORD_SIZE,
};

// Stream compact
pub use stream_compact::{
    StreamCompactParams, CompactedIndices, StreamCompactPipeline,
    workgroups_for_objects as compact_workgroups_for_objects,
    workgroups_for_objects as stream_compact_workgroups,
    workgroups_for_objects_batch, cpu_is_visible as stream_cpu_is_visible,
    cpu_is_visible, cpu_stream_compact, cpu_compact_visible_indices, cpu_verify_prefix_sum,
    WORKGROUP_SIZE as COMPACT_WORKGROUP_SIZE,
    WORKGROUP_SIZE as STREAM_COMPACT_WORKGROUP_SIZE,
    COMPACT_PARAMS_SIZE, BITS_PER_WORD as STREAM_COMPACT_BITS_PER_WORD,
    BATCH_SIZE as COMPACT_BATCH_SIZE,
    BATCH_SIZE as STREAM_COMPACT_BATCH_SIZE, COMPACT_SHADER,
};

// Frustum culling
pub use frustum_cull::{
    FrustumPlane, Frustum, FrustumPlaneIndex, CullParams, InstanceBounds,
    FrustumCullResources, FrustumCullPipeline as FrustumCullPipelineV1,
    cpu_frustum_cull, cpu_frustum_cull_sphere_only, cpu_frustum_cull_aabb_only,
    WORKGROUP_SIZE as FRUSTUM_WORKGROUP_SIZE, NUM_FRUSTUM_PLANES,
    FLAG_USE_SPHERE, FLAG_DEBUG_VISIBLE,
};

// Frustum cull pipeline v2
pub use frustum_cull_pipeline::{
    CullDispatchParams, FrustumCullPipeline as FrustumCullPipelineV2,
    workgroups_for_objects,
    WORKGROUP_SIZE as FRUSTUM_CULL_PIPELINE_WORKGROUP_SIZE,
    CULL_DISPATCH_PARAMS_SIZE,
};

// Object data
pub use object_data::{
    ObjectData, ObjectDataBuffer, object_flags,
    OBJECT_DATA_SIZE, MAX_LOD_LEVELS as OBJECT_MAX_LOD_LEVELS,
    INVALID_MESH_INDEX, INVALID_MATERIAL_INDEX, DEFAULT_LOD_DISTANCES,
};

// Scene data
pub use scene_data::{
    SceneDataBuffers, DEFAULT_SCENE_CAPACITY, MIN_BUFFER_CAPACITY, GROWTH_FACTOR,
};

// Frustum buffer and types
pub use frustum::{
    FrustumBuffer, FrustumPlanes, FrustumCullParams, CullAABB,
    perspective_matrix, look_at_matrix, multiply_matrices,
    NUM_FRUSTUM_PLANES as FRUSTUM_NUM_PLANES, FRUSTUM_PLANE_SIZE, FRUSTUM_PLANES_SIZE,
    PLANE_LEFT, PLANE_RIGHT, PLANE_BOTTOM, PLANE_TOP, PLANE_NEAR, PLANE_FAR,
    CULL_AABB_SIZE, FRUSTUM_CULL_PARAMS_SIZE, FRUSTUM_CULL_SHADER,
    VISIBILITY_OUTSIDE, VISIBILITY_INTERSECTING, VISIBILITY_INSIDE,
    create_frustum_cull_bind_group_layout, create_frustum_cull_batch_bind_group_layout,
};
/// Alias for FrustumPlane for backward compatibility (uses frustum module version with full API).
pub use frustum::FrustumPlane as FrustumPlaneExtract;

// Bindless bind group
pub use bindless_bind_group::{
    BindlessBindGroupBuilder, BindlessBindGroupManager, BindlessBindGroupMetrics,
    MAX_BINDLESS_SAMPLERS, MIN_BINDLESS_TEXTURES, MIN_BINDLESS_SAMPLERS,
    BINDING_TEXTURES, BINDING_SAMPLERS, BINDING_MATERIALS, BINDLESS_BIND_GROUP_INDEX,
    supports_texture_arrays, supports_non_uniform_indexing, supports_partially_bound,
    supports_full_bindless, required_features as bindless_required_features,
    optimal_features as bindless_optimal_features,
    create_bindless_layout, create_bindless_layout_with_capacity,
};

// Buffer registry
pub use buffer_registry::{
    BindlessBufferRegistry, MAX_BINDLESS_BUFFERS, MIN_BUFFER_SIZE,
};

// GPU culling pipeline
pub use gpu_culling_pipeline::{
    GPUCullingConfig, GPUCullingParams, CullingStage, CullingDebugDump,
    GPUCullingPipelineBuilder, GPUCullingPipeline,
    GPU_CULLING_PARAMS_SIZE, WORKGROUP_SIZE as GPU_CULLING_WORKGROUP_SIZE,
    DEFAULT_MAX_OBJECTS as GPU_CULLING_DEFAULT_MAX_OBJECTS,
    DEFAULT_MAX_DRAWS as GPU_CULLING_DEFAULT_MAX_DRAWS,
    DEFAULT_HIZ_WIDTH, DEFAULT_HIZ_HEIGHT,
    FLAG_SKIP_FRUSTUM as GPU_CULLING_FLAG_SKIP_FRUSTUM,
    FLAG_SKIP_HIZ as GPU_CULLING_FLAG_SKIP_HIZ,
    FLAG_SKIP_LOD as GPU_CULLING_FLAG_SKIP_LOD,
    FLAG_CONSERVATIVE as GPU_CULLING_FLAG_CONSERVATIVE,
    FLAG_DEBUG as GPU_CULLING_FLAG_DEBUG,
    workgroups_for_objects as gpu_culling_workgroups_for_objects,
};

// Texture registry
pub use texture_registry::{
    TextureRegistry, TextureRegistryMetrics,
    supports_bindless_textures,
    required_features as texture_registry_required_features,
    optimal_features as texture_registry_optimal_features,
    cpu_count_active, cpu_find_free_slot, cpu_fragmentation,
    MAX_BINDLESS_TEXTURES as TEXTURE_REGISTRY_MAX_TEXTURES,
    MIN_BINDLESS_TEXTURES as TEXTURE_REGISTRY_MIN_TEXTURES,
    BINDLESS_BIND_GROUP_INDEX as TEXTURE_REGISTRY_BIND_GROUP_INDEX,
    TEXTURE_ARRAY_BINDING, SAMPLER_BINDING,
};

// HiZ cull pipeline
pub use hiz_cull_pipeline::{
    HiZCullParams, HiZCullPipeline,
    workgroups_for_objects as hiz_cull_workgroups_for_objects,
    HIZ_CULL_PARAMS_SIZE, WORKGROUP_SIZE as HIZ_CULL_WORKGROUP_SIZE,
    FLAG_SKIP_FRUSTUM, FLAG_SKIP_HIZ, FLAG_CONSERVATIVE, FLAG_DEBUG,
};
// Occlusion flag aliases for test compatibility
pub const OCCLUSION_FLAG_DEBUG_VISIBLE: u32 = FLAG_DEBUG;
pub const OCCLUSION_FLAG_CONSERVATIVE: u32 = FLAG_CONSERVATIVE;

// Build indirect
pub use build_indirect::{
    BuildIndirectParams, MeshData as BuildIndirectMeshData, BuildIndirectPipeline,
    cpu_build_indirect,
    WORKGROUP_SIZE as BUILD_INDIRECT_WORKGROUP_SIZE,
    BATCH_SIZE as BUILD_INDIRECT_BATCH_SIZE,
    MAX_LOD_LEVELS as BUILD_INDIRECT_MAX_LOD_LEVELS,
    DEFAULT_MAX_DRAWS as BUILD_INDIRECT_DEFAULT_MAX_DRAWS,
    BUILD_INDIRECT_PARAMS_SIZE, MESH_DATA_SIZE, DRAW_INDEXED_INDIRECT_ARGS_SIZE,
    BUILD_INDIRECT_SHADER,
};
pub use build_indirect::IndirectDrawIndexedArgs as BuildIndirectDrawArgs;

// LOD buffer
pub use lod_buffer::{
    LodEntry, LodBuffer as LodEntryBuffer,
    cpu_clear_lod_entries, cpu_set_lod_entry, cpu_get_lod_level,
    cpu_count_by_lod, cpu_collect_by_lod,
    LOD_ENTRY_SIZE, DEFAULT_LOD_BUFFER_CAPACITY, MIN_LOD_BUFFER_CAPACITY,
    MAX_LOD_LEVEL as LOD_BUFFER_MAX_LOD_LEVEL, DEFAULT_POOL_SIZE,
};
pub use lod_buffer::MAX_LOD_LEVEL;

// LOD select additional exports
pub use lod_select::{
    workgroups_for_objects as lod_select_workgroups_for_objects,
    calculate_dispatch as lod_select_calculate_dispatch,
};

// Aliases for test compatibility
pub use bindless_bind_group::supports_non_uniform_indexing as bindless_supports_non_uniform_indexing;
pub use bindless_bind_group::supports_partially_bound as bindless_supports_partially_bound;
pub use bindless_bind_group::MIN_BINDLESS_TEXTURES as BINDLESS_MIN_TEXTURES;
pub use bindless_bind_group::MAX_BINDLESS_TEXTURES as BINDLESS_MAX_TEXTURES;

// Re-export index allocator types for test compatibility
pub use crate::resources::index_allocator::{IndexAllocator, GenerationalIndex, AllocatorError};
