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
//! - `sort`            -- GPU Radix Sort for key-value pair sorting (T-GPU-2.1)
//! - `compact`         -- GPU Stream Compaction using prefix sum (T-GPU-2.2/2.3)
//! - `indirect_draw`   -- Indirect Draw Buffer Management (T-GPU-2.4)
//! - `frustum_cull`    -- GPU Frustum Culling using sphere+AABB tests (T-GPU-3.1)
//! - `distance_cull`   -- GPU Distance/LOD Culling with LOD selection (T-GPU-3.2)
//! - `draw_args`       -- Draw Argument Generation from sorted visibility (T-GPU-3.3)
//! - `occlusion_cull`  -- GPU HiZ Occlusion Culling for hidden surface removal (T-GPU-3.4)
//! - `visibility_buffer` -- Visibility Buffer Write for material shading (T-GPU-3.5)
//! - `visibility_read` -- Visibility Buffer Read for deferred shading (T-GPU-3.6)
//! - `triangle_cull`   -- Per-Triangle Culling: backface, degenerate, frustum (T-GPU-3.7)
//! - `small_triangle_cull` -- Small Triangle Culling for sub-pixel geometry (T-GPU-3.8)
//! - `hzb`             -- Hierarchical-Z Buffer Construction for occlusion culling (T-GPU-4.1)
//! - `instance_update` -- GPU Instance Update for transform packing and LOD/visibility (T-GPU-4.2)
//! - `meshlet`         -- Meshlet Generation for fine-grained GPU culling (T-GPU-4.3)
//! - `meshlet_generator` -- Configurable meshlet generator with greedy splitting (T-WGPU-P6.9.2)
//! - `geometry_path`    -- Geometry path abstraction for traditional vs meshlet rendering (T-WGPU-P6.9.3)
//! - `meshlet_cull`    -- Per-Meshlet Culling: frustum, cone, HZB tests (T-GPU-4.4)
//! - `meshlet_render`  -- Meshlet Rendering Pipeline with indirect draws (T-GPU-4.5)
//! - `object_data`     -- Per-object GPU data struct (T-WGPU-P6.2.1)
//! - `scene_data`      -- Scene data buffer management with dirty tracking (T-WGPU-P6.2.2)
//! - `visibility_flags` -- Visibility flags buffer (bitfield) for GPU culling (T-WGPU-P6.2.3)
//! - `frustum`          -- Frustum plane extraction for GPU-driven culling (T-WGPU-P6.3.1)
//! - `frustum_cull_pipeline` -- Frustum cull compute pipeline (T-WGPU-P6.3.3)
//! - `hiz_pyramid`       -- HiZ pyramid texture creation for occlusion culling (T-WGPU-P6.4.1)
//!                          Also contains HiZ downsample shader integration (T-WGPU-P6.4.2)
//! - `hiz_occlusion`     -- HiZ occlusion test functions for GPU-driven culling (T-WGPU-P6.4.3)
//! - `hiz_cull_pipeline` -- Combined frustum + HiZ occlusion cull pipeline (T-WGPU-P6.4.4)
//! - `lod`               -- LOD distance calculation helpers (T-WGPU-P6.5.1)
//! - `lod_select`        -- LOD selection compute shader and pipeline (T-WGPU-P6.5.2)
//! - `lod_buffer`        -- LOD buffer management for per-object LOD selection (T-WGPU-P6.5.3)
//! - `stream_compact`    -- Visibility stream compaction using prefix scan (T-WGPU-P6.6.1)
//! - `build_indirect`    -- Indirect buffer generation from compacted objects (T-WGPU-P6.6.2)
//! - `gpu_culling_pipeline` -- Unified GPU culling pipeline integrating all 5 stages (T-WGPU-P6.6.3)
//! - `multi_draw`        -- Multi-draw indirect wrapper with fallback (T-WGPU-P6.7.1)
//! - `buffer_registry`   -- Bindless buffer registry for GPU-driven rendering (T-WGPU-P6.8.2)
//! - `texture_registry`  -- Bindless texture registry for GPU-driven rendering (T-WGPU-P6.8.1)
//! - `index_allocator`   -- Generic index allocator with recycling (T-WGPU-P6.8.3, re-exported from resources)
//! - `bindless_bind_group` -- Unified bindless bind group layout and builder (T-WGPU-P6.8.5)

pub mod bindless_bind_group;
pub mod buffer_registry;
pub mod build_indirect;
pub mod texture_registry;
pub mod buffers;
pub mod compact;
pub mod distance_cull;
pub mod draw_args;
pub mod frustum;
pub mod frustum_cull;
pub mod frustum_cull_pipeline;
pub mod geometry_path;
pub mod gpu_culling_pipeline;
pub mod hiz_cull_pipeline;
pub mod hiz_occlusion;
pub mod hiz_pyramid;
pub mod hzb;
pub mod indirect_draw;
pub mod instance_update;
pub mod lod;
pub mod lod_buffer;
pub mod lod_select;
pub mod material_table;
pub mod mesh_table;
pub mod meshlet;
pub mod meshlet_cull;
pub mod meshlet_generator;
pub mod meshlet_render;
pub mod multi_draw;
pub mod object_data;
pub mod occlusion_cull;
pub mod scene_data;
pub mod small_triangle_cull;
pub mod sort;
pub mod stream_compact;
pub mod triangle_cull;
pub mod texture_table;
pub mod visibility_buffer;
pub mod visibility_flags;
pub mod visibility_read;

pub use buffers::{
    AcquireResult, BufferRegistry, BufferSlot, ReleaseResult, SlotState,
    StagingBufferDesc, SubmitResult, NUM_STAGING_SLOTS,
};

pub use material_table::{
    AddEntry as MaterialAddEntry, MaterialTable, MaterialTableEntry,
    RemoveResult as MaterialRemoveResult,
    DEFAULT_MATERIAL_TABLE_CAPACITY, MATERIAL_FLAG_DIRTY, MATERIAL_FLAG_VISIBLE,
    MATERIAL_TABLE_ENTRY_SIZE,
    // T-WGPU-P6.8.4: GPU-driven material table with bytemuck support
    MaterialDescriptor, GpuMaterialTable,
    DEFAULT_GPU_MATERIAL_TABLE_CAPACITY, MATERIAL_DESCRIPTOR_SIZE,
    MATERIAL_DESC_FLAG_DOUBLE_SIDED, MATERIAL_DESC_FLAG_ALPHA_MASK,
    MATERIAL_DESC_FLAG_ALPHA_BLEND, MATERIAL_DESC_FLAG_UNLIT, NO_TEXTURE,
};

pub use mesh_table::{
    AddEntry, MeshTable, MeshTableEntry, RemoveResult,
    DEFAULT_MESH_TABLE_CAPACITY, MESH_TABLE_ENTRY_SIZE,
};
pub use texture_table::{
    AddEntry as TextureAddEntry, TextureTable, TextureTableEntry,
    RemoveResult as TextureRemoveResult,
    DEFAULT_TEXTURE_TABLE_CAPACITY, MAX_BINDLESS_TEXTURES, TEXTURE_TABLE_ENTRY_SIZE,
};

pub use sort::{
    GpuRadixSort, SortParams, SortResult,
    MAX_SMALL_ELEMENTS, MIN_GPU_ELEMENTS, NUM_PASSES, RADIX_BITS, RADIX_SIZE,
    WORKGROUP_SIZE as SORT_WORKGROUP_SIZE,
};

pub use compact::{
    CompactParams, CompactPipeline, CompactResources,
    cpu_compact, cpu_exclusive_prefix_sum, cpu_inclusive_prefix_sum,
    MAX_BLOCKS_SIMPLE, MAX_ELEMENTS_SIMPLE, SINGLE_PASS_MAX,
    WORKGROUP_SIZE as COMPACT_WORKGROUP_SIZE,
};

pub use indirect_draw::{
    CountBuffer, DispatchIndirectArgs, DrawBatchBuilder, DrawIndexedIndirectArgs,
    DrawIndirectArgs, IndirectDispatchArgs, IndirectDispatchBuffer, IndirectDrawArgs,
    IndirectDrawBuffer, IndirectDrawIndexedArgs, IndirectTier, MultiIndirectBuffer,
    MultiIndirectConfig, DEFAULT_MAX_DRAWS, INDIRECT_DISPATCH_ARGS_SIZE,
    INDIRECT_DRAW_ARGS_SIZE, INDIRECT_DRAW_INDEXED_ARGS_SIZE,
};

pub use frustum_cull::{
    CullParams, Frustum, FrustumCullPipeline, FrustumCullResources,
    FrustumPlane, FrustumPlaneIndex, InstanceBounds,
    cpu_frustum_cull, cpu_frustum_cull_aabb_only, cpu_frustum_cull_sphere_only,
    FLAG_DEBUG_VISIBLE, FLAG_USE_SPHERE, NUM_FRUSTUM_PLANES,
    WORKGROUP_SIZE as FRUSTUM_CULL_WORKGROUP_SIZE,
};

pub use distance_cull::{
    DistanceCullParams, DistanceCullPipeline, DistanceCullResources,
    InstanceLOD, LODResult,
    cpu_distance_cull, cpu_distance_cull_only, cpu_lod_select_only,
    CULLED_LOD, MAX_LODS,
    WORKGROUP_SIZE as DISTANCE_CULL_WORKGROUP_SIZE,
};

pub use draw_args::{
    DrawArgsParams, DrawArgsPipeline, DrawArgsResources,
    MeshMetadata, VisibilityEntry,
    cpu_gen_draw_args,
    DEFAULT_MAX_DRAWS as DRAW_ARGS_DEFAULT_MAX_DRAWS,
    MAX_LOD_LEVELS, WORKGROUP_SIZE as DRAW_ARGS_WORKGROUP_SIZE,
};

pub use occlusion_cull::{
    CpuHiZBuffer, OcclusionCullParams, OcclusionCullPipeline, OcclusionCullResources,
    OcclusionInstanceBounds, OcclusionResult,
    cpu_occlusion_cull,
    DEFAULT_HIZ_MIPS,
    FLAG_CONSERVATIVE as OCCLUSION_FLAG_CONSERVATIVE,
    FLAG_DEBUG_VISIBLE as OCCLUSION_FLAG_DEBUG_VISIBLE,
    FLAG_NO_SPHERE_TEST as OCCLUSION_FLAG_NO_SPHERE_TEST,
    WORKGROUP_SIZE as OCCLUSION_CULL_WORKGROUP_SIZE,
};

pub use visibility_buffer::{
    PackedVisibility, VisibilityBufferResources, VisibilityData,
    VisibilityWriteParams, VisibilityWritePipeline, VisibleInstance,
    cpu_visibility_clear, cpu_visibility_write,
    DEFAULT_MAX_VISIBILITY, INVALID_INSTANCE_ID, INVALID_PRIMITIVE_ID,
    WORKGROUP_SIZE as VISIBILITY_WRITE_WORKGROUP_SIZE,
};

pub use visibility_read::{
    InstanceMetadata, InstanceTransform, ShadingInput,
    VertexData, VisibilityReadParams, VisibilityReadPipeline, VisibilityReadResources,
    VisibilityData as VisibilityReadData,
    cpu_compute_tangent_space, cpu_interpolate_vec2, cpu_interpolate_vec3, cpu_interpolate_vec4,
    cpu_safe_normalize, cpu_transform_normal, cpu_transform_position, cpu_transform_tangent,
    cpu_visibility_read,
    EPSILON as VISIBILITY_READ_EPSILON, INVALID_INSTANCE as VISIBILITY_READ_INVALID_INSTANCE,
    INVALID_PRIMITIVE as VISIBILITY_READ_INVALID_PRIMITIVE,
    TILE_SIZE as VISIBILITY_READ_TILE_SIZE,
    WORKGROUP_SIZE as VISIBILITY_READ_WORKGROUP_SIZE,
};

pub use small_triangle_cull::{
    ProjectedTriangle, SmallTriangleCullParams, SmallTriangleCullPipeline,
    SmallTriangleCullResources, SmallTriangleCullResult,
    cpu_cull_degenerate_only, cpu_small_triangle_cull,
    is_degenerate, ndc_to_pixels, triangle_area_2d,
    DEFAULT_MIN_PIXEL_AREA, DEGENERATE_EPSILON,
    WORKGROUP_SIZE as SMALL_TRIANGLE_CULL_WORKGROUP_SIZE,
};

pub use triangle_cull::{
    CullMode, CullReason, CullResult, TriangleCullParams,
    TriangleCullPipeline, TriangleCullResources, TriangleInput,
    cpu_triangle_cull, cpu_triangle_cull_backface_only, cpu_triangle_cull_frustum_only,
    CULL_REASON_BACKFACE, CULL_REASON_DEGENERATE, CULL_REASON_FRUSTUM, CULL_REASON_NONE,
    DEFAULT_DEGENERATE_THRESHOLD,
    WORKGROUP_SIZE as TRIANGLE_CULL_WORKGROUP_SIZE,
};

pub use meshlet::{
    Meshlet, MeshletBounds, MeshletData,
    MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES,
    MESHLET_BOUNDS_SIZE, MESHLET_SIZE,
};

pub use meshlet_generator::{
    MeshletGenerator, MeshInput, MeshletOutput, MeshletStats,
    MAX_MESHLET_VERTICES as GENERATOR_MAX_VERTICES,
    MAX_MESHLET_TRIANGLES as GENERATOR_MAX_TRIANGLES,
};

pub use geometry_path::{
    GeometryPath, GeometryPathConfig, GeometryRenderable,
    TRADITIONAL_PATH_NAME, MESHLET_PATH_NAME,
};

pub use hzb::{
    HZBBuildParams, HZBPipeline, HZBResources,
    calculate_mip_count, cpu_build_hzb, cpu_sample_hzb, mip_dimensions,
    FLAG_USE_MIN as HZB_FLAG_USE_MIN, HZB_FORMAT, MAX_HZB_MIPS,
    WORKGROUP_SIZE as HZB_WORKGROUP_SIZE,
};

pub use hiz_pyramid::{
    HiZPyramid, HiZDownsampleParams,
    create_hiz_gen_bind_group_layout, create_hiz_sample_bind_group_layout,
    create_hiz_downsample_texture_layout, create_hiz_downsample_params_layout,
    calculate_downsample_dispatch, cpu_max_reduction,
    HIZ_FORMAT, HIZ_USAGE, MAX_HIZ_MIPS, MIN_HIZ_SIZE,
    HIZ_DOWNSAMPLE_SHADER, HIZ_DOWNSAMPLE_WORKGROUP_SIZE, HIZ_DOWNSAMPLE_PARAMS_SIZE,
};

pub use hiz_occlusion::{
    HiZOcclusionParams, BatchParams as HiZBatchParams, InputAABB as HiZInputAABB,
    create_hiz_occlusion_texture_layout, create_hiz_occlusion_params_layout,
    create_hiz_occlusion_batch_layout,
    cpu_project_aabb, cpu_select_mip_level, cpu_test_occlusion,
    workgroups_for_objects as hiz_workgroups_for_objects,
    HIZ_OCCLUSION_SHADER, HIZ_OCCLUSION_PARAMS_SIZE, INPUT_AABB_SIZE, BATCH_PARAMS_SIZE,
    MAX_MIP_LEVEL, EPSILON as HIZ_EPSILON, CONSERVATIVE_EXPAND,
    WORKGROUP_SIZE as HIZ_OCCLUSION_WORKGROUP_SIZE,
};

pub use meshlet_cull::{
    FrustumPlane as MeshletFrustumPlane, MeshInfo, MeshletBounds as MeshletCullBounds,
    MeshletCullParams, MeshletCullPipeline, MeshletCullResources, MeshletVisibility,
    cpu_cone_cull, cpu_frustum_cull_sphere, cpu_meshlet_cull, cpu_meshlet_cull_flat,
    DEFAULT_MAX_MESHES, DEFAULT_MAX_MESHLETS, FLAT_WORKGROUP_SIZE,
    MAX_MESHLETS_PER_MESH, NUM_FRUSTUM_PLANES as MESHLET_NUM_FRUSTUM_PLANES,
    WORKGROUP_SIZE as MESHLET_CULL_WORKGROUP_SIZE,
};

pub use meshlet_render::{
    MeshletDrawCommand, MeshletInstance, MeshletRenderParams, MeshletRenderPipeline,
    MeshletRenderResources, MeshletVertex,
    cpu_pack_visibility_id, cpu_transform_normal as meshlet_cpu_transform_normal,
    cpu_transform_vertex, cpu_unpack_visibility_id,
    FLAG_ALPHA_TEST, FLAG_DOUBLE_SIDED, FLAG_SHADOW_PASS, FLAG_VISIBILITY_BUFFER,
    INVALID_VISIBILITY_ID, MAX_MESHLETS_PER_DRAW,
    MESHLET_INSTANCE_SIZE, MESHLET_RENDER_PARAMS_SIZE, MESHLET_VERTEX_SIZE,
};

pub use instance_update::{
    InputTransform, InstanceBuffer, InstanceData, InstanceUpdateParams,
    InstanceUpdatePipeline, InstanceUpdateResources, LocalBounds,
    cpu_instance_update, cpu_instance_update_transform_only, cpu_instance_update_visibility_only,
    cpu_pack_transform, cpu_unpack_transform,
    DEFAULT_INSTANCE_CAPACITY, INSTANCE_DATA_SIZE, INSTANCE_UPDATE_PARAMS_SIZE,
    INPUT_TRANSFORM_SIZE, INVALID_MATERIAL, LOCAL_BOUNDS_SIZE, LOD_CULLED,
    FLAG_CAST_SHADOW, FLAG_DIRTY, FLAG_MOTION_BLUR, FLAG_RECEIVE_SHADOW,
    FLAG_SKINNED, FLAG_STATIC, FLAG_TWO_SIDED, FLAG_VISIBLE,
    WORKGROUP_SIZE as INSTANCE_UPDATE_WORKGROUP_SIZE,
};

pub use object_data::{
    ObjectData, ObjectDataBuffer, object_flags,
    DEFAULT_LOD_DISTANCES, INVALID_MATERIAL_INDEX, INVALID_MESH_INDEX,
    MAX_LOD_LEVELS as OBJECT_MAX_LOD_LEVELS, OBJECT_DATA_SIZE,
};

pub use scene_data::{
    SceneDataBuffers,
    DEFAULT_SCENE_CAPACITY, GROWTH_FACTOR, MIN_BUFFER_CAPACITY,
};

pub use visibility_flags::{
    VisibilityFlagsBuffer,
    bit_location, words_for_objects, is_visible, set_visible, clear_visible, count_visible,
    cpu_atomic_or_visibility, cpu_clear_visibility_flags, cpu_compact_visible,
    BITS_PER_WORD, DEFAULT_VISIBILITY_FLAGS_CAPACITY, MIN_VISIBILITY_FLAGS_CAPACITY, WORD_SIZE,
};

pub use frustum::{
    CullAABB, FrustumBuffer, FrustumCullParams, FrustumPlane as FrustumPlaneExtract, FrustumPlanes,
    create_frustum_cull_batch_bind_group_layout, create_frustum_cull_bind_group_layout,
    look_at_matrix, multiply_matrices, perspective_matrix,
    CULL_AABB_SIZE, FRUSTUM_CULL_PARAMS_SIZE, FRUSTUM_CULL_SHADER,
    FRUSTUM_PLANE_SIZE, FRUSTUM_PLANES_SIZE, NUM_FRUSTUM_PLANES as FRUSTUM_NUM_PLANES,
    PLANE_BOTTOM, PLANE_FAR, PLANE_LEFT, PLANE_NEAR, PLANE_RIGHT, PLANE_TOP,
    VISIBILITY_INSIDE, VISIBILITY_INTERSECTING, VISIBILITY_OUTSIDE,
};

pub use frustum_cull_pipeline::{
    FrustumCullPipeline as FrustumCullPipelineV2, CullDispatchParams,
    workgroups_for_objects,
    CULL_DISPATCH_PARAMS_SIZE, WORKGROUP_SIZE as FRUSTUM_CULL_PIPELINE_WORKGROUP_SIZE,
};

pub use hiz_cull_pipeline::{
    HiZCullPipeline, HiZCullParams,
    workgroups_for_objects as hiz_cull_workgroups_for_objects,
    FLAG_SKIP_FRUSTUM, FLAG_SKIP_HIZ, FLAG_CONSERVATIVE, FLAG_DEBUG,
    HIZ_CULL_PARAMS_SIZE, WORKGROUP_SIZE as HIZ_CULL_WORKGROUP_SIZE,
};

pub use lod::{
    LodDistances, LodParams, LodConfig, LodLevel,
    distance_to_camera, distance_to_camera_squared,
    screen_coverage, select_lod, select_lod_by_distance, select_lod_by_distance_squared,
    select_lod_by_coverage, select_lod_by_coverage_custom, squared_thresholds,
    LOD_DISTANCES_SIZE, LOD_PARAMS_SIZE, MAX_LOD_LEVELS as LOD_MAX_LEVELS,
    DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE,
    COVERAGE_LOD0, COVERAGE_LOD1, COVERAGE_LOD2,
};

pub use lod_select::{
    LodSelectParams, ObjectLodInput, LodSelectOutput, LodBuffer as LodSelectBuffer,
    SelectionMode, object_lod_flags,
    workgroups_for_objects as lod_select_workgroups_for_objects,
    calculate_dispatch as lod_select_calculate_dispatch,
    cpu_distance_to_camera, cpu_screen_coverage, cpu_select_lod,
    cpu_select_lod_by_distance as cpu_lod_select_by_distance,
    cpu_select_lod_by_coverage as cpu_lod_select_by_coverage,
    cpu_select_lod_batch,
    LOD_SELECT_SHADER, LOD_SELECT_PARAMS_SIZE, OBJECT_LOD_INPUT_SIZE, LOD_SELECT_OUTPUT_SIZE,
    MAX_LOD_LEVELS as LOD_SELECT_MAX_LEVELS, DEFAULT_BLEND_RANGE,
    WORKGROUP_SIZE as LOD_SELECT_WORKGROUP_SIZE,
};

pub use lod_buffer::{
    LodEntry, LodBuffer, LodBufferPool,
    cpu_clear_lod_entries, cpu_set_lod_entry, cpu_get_lod_level,
    cpu_count_by_lod, cpu_collect_by_lod,
    LOD_ENTRY_SIZE, DEFAULT_LOD_BUFFER_CAPACITY, MIN_LOD_BUFFER_CAPACITY,
    MAX_LOD_LEVEL, DEFAULT_POOL_SIZE,
};

pub use stream_compact::{
    StreamCompactPipeline, StreamCompactParams, CompactedIndices,
    cpu_is_visible, cpu_stream_compact, cpu_compact_visible_indices, cpu_verify_prefix_sum,
    workgroups_for_objects as stream_compact_workgroups,
    workgroups_for_objects_batch as stream_compact_workgroups_batch,
    COMPACT_SHADER, COMPACT_PARAMS_SIZE, WORKGROUP_SIZE as STREAM_COMPACT_WORKGROUP_SIZE,
    BITS_PER_WORD as STREAM_COMPACT_BITS_PER_WORD, BATCH_SIZE as STREAM_COMPACT_BATCH_SIZE,
};

pub use build_indirect::{
    BuildIndirectPipeline, BuildIndirectResources, BuildIndirectParams,
    MeshData as BuildIndirectMeshData, IndirectDrawIndexedArgs as BuildIndirectDrawArgs,
    cpu_build_indirect,
    BUILD_INDIRECT_SHADER, BUILD_INDIRECT_PARAMS_SIZE, MESH_DATA_SIZE, DRAW_INDEXED_INDIRECT_ARGS_SIZE,
    DEFAULT_MAX_DRAWS as BUILD_INDIRECT_DEFAULT_MAX_DRAWS, BATCH_SIZE as BUILD_INDIRECT_BATCH_SIZE,
    WORKGROUP_SIZE as BUILD_INDIRECT_WORKGROUP_SIZE, MAX_LOD_LEVELS as BUILD_INDIRECT_MAX_LOD_LEVELS,
};

pub use gpu_culling_pipeline::{
    GPUCullingPipeline, GPUCullingPipelineBuilder, GPUCullingConfig, GPUCullingParams,
    CullingStage, CullingDebugDump,
    workgroups_for_objects as gpu_culling_workgroups_for_objects,
    FLAG_SKIP_FRUSTUM as GPU_CULLING_FLAG_SKIP_FRUSTUM,
    FLAG_SKIP_HIZ as GPU_CULLING_FLAG_SKIP_HIZ,
    FLAG_SKIP_LOD as GPU_CULLING_FLAG_SKIP_LOD,
    FLAG_CONSERVATIVE as GPU_CULLING_FLAG_CONSERVATIVE,
    FLAG_DEBUG as GPU_CULLING_FLAG_DEBUG,
    DEFAULT_MAX_OBJECTS as GPU_CULLING_DEFAULT_MAX_OBJECTS,
    DEFAULT_MAX_DRAWS as GPU_CULLING_DEFAULT_MAX_DRAWS,
    DEFAULT_HIZ_WIDTH, DEFAULT_HIZ_HEIGHT,
    GPU_CULLING_PARAMS_SIZE, WORKGROUP_SIZE as GPU_CULLING_WORKGROUP_SIZE,
};

pub use multi_draw::{
    MultiDrawSupport,
    multi_draw_indirect, multi_draw_indexed_indirect,
    multi_draw_indirect_count, multi_draw_indexed_indirect_count,
    draw_indirect_offset, draw_indexed_indirect_offset,
    buffer_size_for_draws, buffer_size_for_indexed_draws,
    DRAW_INDIRECT_STRIDE, DRAW_INDEXED_INDIRECT_STRIDE,
};

// Test helpers for multi_draw warning functions (only available in tests)
#[cfg(any(test, feature = "test-helpers"))]
pub use multi_draw::{
    reset_multi_draw_warning, reset_multi_draw_count_warning,
    has_warned_multi_draw_fallback, has_warned_multi_draw_count_fallback,
    trigger_multi_draw_warning, trigger_multi_draw_count_warning,
};

pub use buffer_registry::{
    BindlessBufferRegistry,
    MAX_BINDLESS_BUFFERS, MIN_BUFFER_SIZE,
};

pub use texture_registry::{
    TextureRegistry, TextureRegistryMetrics,
    supports_bindless_textures, supports_partially_bound, supports_non_uniform_indexing,
    required_features as texture_registry_required_features,
    optimal_features as texture_registry_optimal_features,
    cpu_count_active, cpu_find_free_slot, cpu_fragmentation,
    MAX_BINDLESS_TEXTURES as TEXTURE_REGISTRY_MAX_TEXTURES,
    MIN_BINDLESS_TEXTURES as TEXTURE_REGISTRY_MIN_TEXTURES,
    BINDLESS_BIND_GROUP_INDEX as TEXTURE_REGISTRY_BIND_GROUP_INDEX,
    TEXTURE_ARRAY_BINDING, SAMPLER_BINDING,
};

pub use bindless_bind_group::{
    BindlessBindGroupBuilder, BindlessBindGroupManager, BindlessBindGroupMetrics,
    create_bindless_layout, create_bindless_layout_with_capacity,
    supports_texture_arrays, supports_full_bindless,
    supports_non_uniform_indexing as bindless_supports_non_uniform_indexing,
    supports_partially_bound as bindless_supports_partially_bound,
    required_features as bindless_required_features,
    optimal_features as bindless_optimal_features,
    MAX_BINDLESS_TEXTURES as BINDLESS_MAX_TEXTURES,
    MAX_BINDLESS_SAMPLERS, MIN_BINDLESS_TEXTURES as BINDLESS_MIN_TEXTURES,
    MIN_BINDLESS_SAMPLERS, BINDING_TEXTURES, BINDING_SAMPLERS, BINDING_MATERIALS,
    BINDLESS_BIND_GROUP_INDEX,
};

// Re-export IndexAllocator from resources module for bindless resource management (T-WGPU-P6.8.3)
pub use crate::resources::index_allocator::{
    IndexAllocator, GenerationalIndex, AllocatorError, IndexAllocatorMetrics,
};
