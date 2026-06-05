//! Resource management for TRINITY.
//!
//! This module provides the resource abstraction layer for wgpu resources,
//! including buffers, textures, vertex formats, and bind groups.
//!
//! # Architecture
//!
//! The resources module follows a layered architecture:
//!
//! ```text
//! Buffer API (buffer.rs)
//!     - TrinityBuffer wrapper with metadata
//!     - Validated buffer creation with alignment
//!     - Debug labels and logging
//!
//! Buffer Pool (buffer_pool.rs) [T-WGPU-P2.2.1]
//!     - Size class allocation (256B to 1MB)
//!     - Free list per class
//!     - Acquire/release API
//!     - Growth policy (double on exhaust)
//!     - Shrink policy (release if >50% free)
//!
//! Vertex Formats (vertex.rs)
//!     - Standard vertex layouts (PBR, Skinned, Terrain, Particle, UI)
//!     - VertexFormatRegistry for runtime lookup
//!     - vertex_attr_array! macro usage
//!
//! Dynamic Uniform Buffers (uniform.rs)
//!     - UNIFORM_ALIGNMENT constant (256 bytes)
//!     - Offset calculation helpers for dynamic binding
//!     - ObjectTransform and CameraUniform types
//!     - Binding type helpers (dynamic vs static)
//!
//! Storage Buffers (storage.rs)
//!     - Read-only and read-write storage binding helpers
//!     - Dynamic offset support
//!     - min_binding_size constraint
//!     - GPU structs for indirect rendering
//!
//! Ring Buffer (ring_buffer.rs) [T-WGPU-P2.2.2]
//!     - Triple-buffered for per-frame data
//!     - Per-frame offset tracking
//!     - Wrap-around with frame isolation
//!     - allocate() returns offset
//!     - Frame reset on begin_frame()
//!     - Metrics: utilization, wrap count, overflow count
//!
//! Deferred Destroyer (deferred_destroyer.rs) [T-WGPU-P2.2.3]
//!     - Queues resources for destruction after N frames
//!     - Default delay: 2 frames (triple-buffer safe)
//!     - Supports buffers, textures, and arbitrary resources
//!     - Metrics: pending_count, destroyed_count, peak_pending
//!     - Cleanup on drop with warning
//!
//! Texture API (texture.rs) [T-WGPU-P2.3.1]
//!     - TrinityTextureDescriptor with all wgpu fields
//!     - view_formats for format reinterpretation
//!     - TrinityTexture wrapper with default view
//!     - Memory estimation and logging
//!     - Auto mip count calculation
//!     - Helper functions: bytes_per_pixel, block_info, calculate_mip_count
//!
//! Texture Format Selector (texture_formats.rs) [T-WGPU-P2.3.2]
//!     - TextureFormatSelector with platform awareness
//!     - color_attachment(): sRGB on desktop, BGRA on Metal
//!     - depth(): Depth32Float primary, Depth24Plus fallback
//!     - normal_map(): RG16Snorm or BC5/ETC2 compressed
//!     - compressed_color(): BC on desktop, ASTC on Apple/mobile
//!     - Format tables for all categories
//!     - Platform detection (Desktop, Apple, Mobile, Web)
//!
//! Texture Views (texture_views.rs) [T-WGPU-P2.3.3]
//!     - TrinityTextureViewDescriptor with full control
//!     - CubeFace enum for cubemap face selection
//!     - Convenience methods on TrinityTexture:
//!       - create_mip_view(), create_mip_range_view()
//!       - create_layer_view(), create_layer_range_view()
//!       - create_cube_face_view(), create_cube_view(), create_cube_array_view()
//!       - create_depth_only_view(), create_stencil_only_view()
//!       - create_format_view(), create_mip_layer_view()
//!     - Validation helpers:
//!       - validate_view_dimensions(), validate_mip_range(), validate_array_range()
//!       - is_depth_stencil_format(), has_depth_component(), has_stencil_component()
//!
//! Mip Generator (mip_generator.rs) [T-WGPU-P2.3.4]
//!     - MipGenerator struct with compute pipeline
//!     - MipFilter enum (Box, Bilinear) with tradeoff docs
//!     - generate_mips(): Full mip chain from level 0
//!     - generate_mip_range(): Specific mip range generation
//!     - NPOT support with floor division
//!     - Format support validation: is_format_supported(), is_filterable()
//!     - MipChainInfo for pre-calculation without texture
//!     - Helper: calculate_mip_size(), calculate_mip_levels()
//!
//! Texture Uploads (texture_uploads.rs) [T-WGPU-P2.3.5]
//!     - TextureUploader struct with auto method selection
//!     - write_texture(): Direct upload via queue.write_texture()
//!     - upload_staged(): Staged upload via staging buffer + copy
//!     - upload(): Auto-selects method based on size threshold (64KB default)
//!     - ROW_PITCH_ALIGNMENT: 256 bytes per wgpu requirement
//!     - STAGING_THRESHOLD: Configurable threshold for staged uploads
//!     - TextureUploadDescriptor: Region, mip level, bytes_per_pixel
//!     - TextureRegion: Convenience struct for region definitions
//!     - Alignment helpers: calculate_row_pitch(), align_to_256(), pad_to_row_pitch()
//!     - Format converters: convert_rgb_to_rgba(), convert_bgra_to_rgba(), etc.
//!     - Subregion uploads: Partial texture updates
//!     - Mip chain uploads: upload_mip_chain() helper
//!
//! Sampler API (sampler.rs) [T-WGPU-P2.4.1]
//!     - TrinitySamplerDescriptor with all wgpu fields + builder pattern
//!     - Address modes: ClampToEdge, Repeat, MirrorRepeat, ClampToBorder
//!     - Filter modes: Nearest, Linear (mag, min, mipmap)
//!     - Anisotropy support with device limit clamping
//!     - Comparison sampler for shadow mapping
//!     - Presets: linear_clamp, linear_repeat, nearest_clamp, nearest_repeat, shadow, trilinear
//!     - Validation: anisotropy limits, LOD range, border color requirements
//!     - TrinitySampler wrapper with metadata access
//!
//! Sampler Cache (sampler_cache.rs) [T-WGPU-P2.4.2]
//!     - Hash-based lookup by SamplerCacheKey
//!     - Arc<Sampler> for shared ownership
//!     - Presets: linear_clamp, linear_repeat, point_clamp, point_repeat, shadow
//!     - get_or_create() API with thread-safe caching
//!     - Metrics: cache size, hits, misses, hit rate
//!     - AtomicU64 for lock-free hit/miss counting
//!     - RwLock for concurrent cache access
//!
//! Bind Group Layout Cache (bind_group_layout_cache.rs) [T-WGPU-P2.5.1]
//!     - Hash-based lookup by sorted binding entries
//!     - Arc<BindGroupLayout> for shared ownership
//!     - get_or_create() API with thread-safe caching
//!     - Double-check locking for concurrent access
//!     - Layout compatibility checking: layouts_compatible(), layouts_equal()
//!     - Metrics: cache size, hits, misses, hit rate
//!     - AtomicU64 for lock-free hit/miss counting
//!
//! Bind Group Cache (bind_group_cache.rs) [T-WGPU-P2.5.2]
//!     - Cache by (layout_hash, resources_hash) composite key
//!     - Arc<BindGroup> for shared ownership
//!     - Resource tracking for targeted invalidation
//!     - invalidate_resource(): Remove bind groups when resources are destroyed
//!     - Frame-based eviction with evict_old()
//!     - create_bind_group() with double-check locking
//!     - Supports all binding types: Buffer, Sampler, TextureView, StorageTextureView
//!     - Metrics: cache size, hits, misses, hit rate, tracked resources
//!     - AtomicU64 for lock-free counters
//!
//! Pipeline Layout Cache (pipeline_layout.rs) [T-WGPU-P2.5.3]
//!     - Hash-based lookup by bind group layout hashes + push constant ranges
//!     - Arc<PipelineLayout> for shared ownership
//!     - get_or_create() API with thread-safe caching
//!     - Double-check locking for concurrent access
//!     - Standard bind group indices: GLOBAL=0, MATERIAL=1, OBJECT=2, BINDLESS=3
//!     - TrinityLayoutBuilder for convenient preset layouts:
//!       - global_only(), global_material(), pbr(), bindless()
//!     - Push constant validation and helper functions
//!     - Metrics: cache size, hits, misses, hit rate
//!     - AtomicU64 for lock-free hit/miss counting
//!
//! Push Constants (push_constants.rs) [T-WGPU-P2.5.4]
//!     - Feature detection: supports_push_constants(), max_push_constant_size()
//!     - PushConstantConfig: Builder pattern for range configuration
//!     - Validation: 4-byte alignment, 128-byte max, overlap detection
//!     - PushConstantWriter: Type-safe wrapper for RenderPass
//!     - ComputePushConstantWriter: Type-safe wrapper for ComputePass
//!     - FallbackPushConstants: Auto-fallback to uniform buffer
//!     - DrawPushConstants: 16-byte minimal per-draw struct
//!     - ExtendedDrawPushConstants: 64-byte struct with model matrix
//!     - Helper functions: vertex_only(), fragment_only(), compute_only()
//!
//! Bindless Textures (bindless_textures.rs) [T-WGPU-P2.6.1]
//!     - Feature detection: supports_bindless_textures(), max_bindless_textures()
//!     - TextureSlot: Lightweight handle for shader texture index access
//!     - TextureRegistry: Sparse array with slot allocation and recycling
//!     - register()/unregister(): Add/remove textures with free slot reuse
//!     - update_bind_group(): Create variable-count texture array binding
//!     - create_bindless_layout(): Helper for bind group layout creation
//!     - Uses bind group index 3 (BINDLESS) per TRINITY convention
//!     - Default capacity: 1024 textures (configurable)
//!     - Metrics: utilization, fragmentation, dirty state
//!
//! Bindless Buffers (bindless_buffers.rs) [T-WGPU-P2.6.2]
//!     - Feature detection: supports_bindless_buffers(), max_bindless_buffers()
//!     - BufferSlot: Lightweight handle for shader buffer index access
//!     - BufferRegistry: Sparse array with slot allocation and recycling
//!     - register()/unregister(): Add/remove buffers with free slot reuse
//!     - Dirty range tracking: mark_dirty(), dirty_slots(), clear_dirty()
//!     - update_bind_group(): Create variable-count buffer array binding
//!     - create_bindless_buffer_layout(): Helper for bind group layout creation
//!     - Uses bind group index 3 (BINDLESS) binding 1 per TRINITY convention
//!     - Default capacity: 1024 buffers (configurable)
//!     - Metrics: utilization, fragmentation, dirty state, dirty ratio
//!
//! Index Allocator (index_allocator.rs) [T-WGPU-P2.6.3]
//!     - Generic index allocator with free list for efficient resource management
//!     - allocate()/free(): O(1) allocation with LIFO free list recycling
//!     - Double-free protection via allocated_set tracking
//!     - Optional generation tracking for use-after-free detection:
//!       - GenerationalIndex: index + generation pair
//!       - allocate_generational(): Returns GenerationalIndex
//!       - is_valid(): Validates index and generation match
//!     - Metrics: count, capacity, free_count, fragmentation, utilization
//!     - Thread-safe: Send + Sync
//!
//! [Future: Bind Group Builder (T-WGPU-P2.1.3)]
//!     - Type-safe bind group construction
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::buffer::{TrinityBufferDescriptor, create_buffer};
//! use wgpu::BufferUsages;
//!
//! # fn example(device: &wgpu::Device) {
//! // Create a vertex buffer
//! let vertex_buffer = create_buffer(device, &TrinityBufferDescriptor {
//!     label: Some("vertex_buffer"),
//!     size: 1024 * 64, // 64KB
//!     usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
//!     mapped_at_creation: false,
//! });
//!
//! // Create a uniform buffer
//! let uniform_buffer = create_buffer(device, &TrinityBufferDescriptor {
//!     label: Some("uniform_buffer"),
//!     size: 256,
//!     usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
//!     mapped_at_creation: false,
//! });
//! # }
//! ```

pub mod bind_group_cache;
pub mod bind_group_layout_cache;
pub mod bindless_buffers;
pub mod bindless_textures;
pub mod buffer;
pub mod buffer_pool;
pub mod deferred_destroyer;
pub mod index_allocator;
pub mod indirect;
pub mod mip_generator;
pub mod pipeline_layout;
pub mod push_constants;
pub mod ring_buffer;
pub mod sampler;
pub mod sampler_cache;
pub mod storage;
pub mod texture;
pub mod texture_formats;
pub mod texture_uploads;
pub mod texture_views;
pub mod uniform;
pub mod vertex;

// Re-export commonly used types
pub use buffer::{
    align_size, create_buffer, is_aligned, try_create_buffer, validate_usage,
    validate_usage_with_label, BufferCreationError, TrinityBuffer, TrinityBufferDescriptor,
    UsageValidationError, BUFFER_ALIGNMENT,
};

// Re-export buffer mapping types and functions
pub use buffer::{
    create_staging_readback_buffer, create_staging_upload_buffer, is_mappable,
    map_buffer_async, map_buffer_async_channel, map_buffer_blocking,
    map_buffer_sync_read, map_buffer_sync_write, MappedBuffer, MappingError, MappingMode,
};

// Re-export buffer usage presets module
pub use buffer::buffer_usages;

// Re-export vertex types
pub use vertex::{
    ParticleVertex, PbrVertex, SkinnedVertex, TerrainVertex, UiVertex,
    VertexFormatRegistry, VertexLayoutId,
};

// Re-export uniform types and helpers
pub use uniform::{
    align_uniform_offset, aligned_uniform_size, dynamic_offset_for_object,
    uniform_binding_type_dynamic, uniform_binding_type_static,
    uniform_buffer_size_for_objects, CameraUniform, ObjectTransform, UNIFORM_ALIGNMENT,
};

// Re-export storage buffer types and helpers
pub use storage::{
    align_storage_dynamic_offset, align_storage_size, storage_binding_type_dynamic_readonly,
    storage_binding_type_dynamic_readonly_sized, storage_binding_type_dynamic_readwrite,
    storage_binding_type_dynamic_readwrite_sized, storage_binding_type_readonly,
    storage_binding_type_readonly_sized, storage_binding_type_readwrite,
    storage_binding_type_readwrite_sized, storage_buffer_size, InstanceData, StorageHeader,
    STORAGE_ALIGNMENT, STORAGE_DYNAMIC_ALIGNMENT,
};

// Re-export indirect buffer types and helpers (prefer exact-size wgpu-compatible versions)
pub use indirect::{
    create_empty_indirect_buffer, create_indirect_buffer, create_typed_indirect_buffer,
    indirect_buffer_size, validate_dispatch_indirect_args, validate_draw_indexed_indirect_args,
    validate_draw_indirect_args, DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs,
    IndirectValidationError, MultiDrawInfo, ValidationOptions,
};

// Re-export padded versions for storage buffer arrays
pub use indirect::{
    DispatchIndirectArgsPadded, DrawIndexedIndirectArgsPadded,
};

// Re-export buffer pool types
pub use buffer_pool::{
    BufferHandle, BufferPool, ClassStats, GrowthPolicy, PoolConfig, PooledBuffer,
    PooledBufferGuard, PoolMetrics, SizeClass,
};

// Re-export ring buffer types
pub use ring_buffer::{
    RingAllocation, RingBuffer, RingBufferConfig, RingBufferMetrics,
    DEFAULT_FRAMES_IN_FLIGHT, RING_BUFFER_MIN_ALIGNMENT,
};

// Re-export deferred destroyer types
pub use deferred_destroyer::{
    DeferredDestroyer, DeferredDestroyerMetrics, DeferredResource, DEFAULT_DESTRUCTION_DELAY,
};

// Re-export texture types and helpers
pub use texture::{
    block_info, bytes_per_pixel, calculate_mip_count, calculate_mip_count_3d,
    create_texture, estimate_texture_size, try_create_texture, texture_usages,
    TextureCreationError, TrinityTexture, TrinityTextureDescriptor,
};

// Re-export texture format selector types and tables [T-WGPU-P2.3.2]
pub use texture_formats::{
    format_tables, Platform, TextureFormatSelector,
};

// Re-export texture view types and helpers [T-WGPU-P2.3.3]
pub use texture_views::{
    has_depth_component, has_stencil_component, is_depth_stencil_format,
    native_view_dimension, validate_array_range, validate_mip_range,
    validate_view_dimensions, CubeFace, TrinityTextureViewDescriptor,
};

// Re-export mip generator types and helpers [T-WGPU-P2.3.4]
pub use mip_generator::{
    calculate_mip_levels, calculate_mip_size, is_filterable, is_format_supported,
    storage_format_for, MipChainInfo, MipFilter, MipGenerator, StorageFormatInfo,
    MIN_MIP_DIMENSION, WORKGROUP_SIZE,
};

// Re-export texture upload types and helpers [T-WGPU-P2.3.5]
pub use texture_uploads::{
    align_to_256, bytes_per_pixel_for_format, calculate_row_pitch, convert_bgra_to_rgba,
    convert_gray_alpha_to_rgba, convert_gray_to_rgba, convert_rgb_to_rgba, convert_rgba_to_bgra,
    is_row_pitch_aligned, mip_size, pad_to_row_pitch, pad_to_row_pitch_3d, premultiply_alpha,
    TextureRegion, TextureUploadDescriptor, TextureUploader, TextureUploadError,
    ROW_PITCH_ALIGNMENT, STAGING_THRESHOLD,
};

// Re-export sampler types and helpers [T-WGPU-P2.4.1]
pub use sampler::{
    create_sampler, try_create_sampler, validate_descriptor, AddressMode, CompareFunction,
    FilterMode, SamplerBorderColor, SamplerValidationError, TrinitySampler,
    TrinitySamplerDescriptor,
};

// Re-export sampler cache types [T-WGPU-P2.4.2]
pub use sampler_cache::{
    CachedSampler, SamplerCache, SamplerCacheKey, SamplerCacheMetrics,
};

// Re-export bind group layout cache types [T-WGPU-P2.5.1]
pub use bind_group_layout_cache::{
    BindGroupLayoutCache, BindGroupLayoutCacheMetrics, BindGroupLayoutKey,
    CachedBindGroupLayout, layouts_compatible, layouts_equal,
};

// Re-export bind group cache types [T-WGPU-P2.5.2]
pub use bind_group_cache::{
    BindGroupCache, BindGroupCacheKey, BindGroupCacheMetrics, BindGroupResourceEntry,
    BindGroupResourceType, CachedBindGroup, ResourceId,
};

// Re-export pipeline layout types [T-WGPU-P2.5.3]
pub use pipeline_layout::{
    bind_group_index, total_push_constant_size, validate_push_constant_ranges,
    CachedPipelineLayout, PipelineLayoutCache, PipelineLayoutCacheMetrics, PipelineLayoutKey,
    TrinityLayoutBuilder, MAX_PUSH_CONSTANT_SIZE,
};

// Re-export push constant types [T-WGPU-P2.5.4]
pub use push_constants::{
    align_up as align_push_constant, compute_only, fragment_only, is_aligned as is_push_constant_aligned,
    max_push_constant_size, supports_push_constants, vertex_fragment, vertex_only,
    ComputePushConstantWriter, DrawPushConstants, ExtendedDrawPushConstants,
    FallbackPushConstants, PushConstantConfig, PushConstantError, PushConstantFallback,
    PushConstantWriter, DEFAULT_FALLBACK_BIND_GROUP, PUSH_CONSTANT_ALIGNMENT,
};

// Re-export bindless texture types [T-WGPU-P2.6.1]
pub use bindless_textures::{
    bindless_optimal_features, bindless_required_features, bindless_texture_layout_entry,
    create_bindless_layout, max_bindless_textures, max_bindless_textures_from_limits,
    supports_bindless_textures, supports_non_uniform_indexing, supports_partially_bound,
    BindlessError, TextureRegistry, TextureRegistryMetrics, TextureSlot,
    BINDLESS_BIND_GROUP_INDEX, BINDLESS_TEXTURE_BINDING, DEFAULT_MAX_TEXTURES,
    MAX_BINDLESS_TEXTURES_CONSERVATIVE, MIN_BINDLESS_TEXTURES,
};

// Re-export bindless buffer types [T-WGPU-P2.6.2]
pub use bindless_buffers::{
    bindless_buffer_layout_entry, bindless_buffer_layout_entry_readonly,
    bindless_buffer_layout_entry_readwrite, bindless_buffer_optimal_features,
    bindless_buffer_required_features, create_bindless_buffer_layout, max_bindless_buffers,
    max_bindless_buffers_from_limits, supports_bindless_buffers, supports_non_uniform_buffer_indexing,
    supports_storage_buffer_array, BindlessBufferError, BufferRegistry, BufferRegistryMetrics,
    BufferSlot, BINDLESS_BUFFER_BINDING, DEFAULT_MAX_BUFFERS, MAX_BINDLESS_BUFFERS_CONSERVATIVE,
    MIN_BINDLESS_BUFFERS,
};

// Re-export index allocator types [T-WGPU-P2.6.3]
pub use index_allocator::{
    AllocatorError, GenerationalIndex, IndexAllocator, IndexAllocatorMetrics,
};
