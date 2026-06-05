//! Render pipeline abstraction layer for TRINITY.
//!
//! This module provides a high-level abstraction over wgpu render pipeline creation
//! with a builder pattern, sensible defaults, and proper layout association.
//!
//! # Architecture
//!
//! ```text
//! TrinityRenderPipeline (render_pipeline.rs)
//!     - Wrapper around wgpu::RenderPipeline
//!     - Tracks label and layout ID for cache invalidation
//!     - Provides raw() accessor for wgpu interop
//!
//! RenderPipelineDescriptor (render_pipeline.rs)
//!     - Builder pattern for all 9 wgpu fields
//!     - Layout is REQUIRED (enforced at compile time)
//!     - Sensible defaults for common use cases
//!
//! State Descriptors:
//!     - vertex_state.rs: Vertex buffers, attributes, shader module
//!     - primitive_state.rs: Topology, culling, polygon mode
//!     - depth_stencil_state.rs: Depth test, stencil ops, bias
//!     - multisample_state.rs: MSAA configuration
//!     - fragment_state.rs: Fragment shader, color targets, blending
//! ```
//!
//! # wgpu 22-25.x Compatibility
//!
//! This module targets wgpu 22+ and is compatible through 25.x using:
//!
//! ```text
//! device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
//!     label: Some("my_pipeline"),
//!     layout: Some(&pipeline_layout),
//!     vertex: wgpu::VertexState { ... },
//!     primitive: wgpu::PrimitiveState { ... },
//!     depth_stencil: Some(wgpu::DepthStencilState { ... }),
//!     multisample: wgpu::MultisampleState { ... },
//!     fragment: Some(wgpu::FragmentState { ... }),
//!     multiview: None,
//!     cache: None,
//! })
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::render_pipeline::{
//!     create_render_pipeline, RenderPipelineDescriptor,
//!     VertexStateDescriptor, FragmentStateDescriptor,
//!     PrimitiveStateDescriptor, MultisampleStateDescriptor,
//! };
//!
//! # fn example(
//! #     device: &wgpu::Device,
//! #     layout: &wgpu::PipelineLayout,
//! #     vs_module: &wgpu::ShaderModule,
//! #     fs_module: &wgpu::ShaderModule,
//! # ) {
//! let pipeline = RenderPipelineDescriptor::new(layout)
//!     .label("pbr_forward")
//!     .vertex(VertexStateDescriptor::new(vs_module))
//!     .fragment(FragmentStateDescriptor::new(fs_module)
//!         .target(wgpu::TextureFormat::Bgra8UnormSrgb))
//!     .primitive(PrimitiveStateDescriptor::default())
//!     .multisample(MultisampleStateDescriptor::default())
//!     .build(device);
//!
//! // Use the pipeline
//! render_pass.set_pipeline(pipeline.raw());
//! # }
//! ```

pub mod blend_mode;
pub mod cache_warming;
pub mod color_target;
pub mod conservative_raster;
pub mod depth_bias;
mod depth_stencil_state;
pub mod draw_commands;
mod fragment_state;
pub mod instance_layout;
mod multisample_state;
pub mod pipeline_cache;
mod primitive_state;
pub mod render_bundle;
pub mod render_pass;
pub mod render_pass_commands;
mod render_pipeline;
pub mod vertex_attribute;
mod vertex_format_registry;
mod vertex_state;
pub mod viewport;

// Re-export main types
pub use render_pipeline::{create_render_pipeline, RenderPipelineDescriptor, TrinityRenderPipeline};

// Re-export vertex state types
pub use vertex_state::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexStateDescriptor,
};

// Re-export primitive state types
pub use primitive_state::{
    get_cull_mode_info, get_front_face_info, get_polygon_mode_info, get_topology_info,
    is_list_topology, is_strip_topology, minimum_vertex_count, required_feature_for_polygon_mode,
    requires_non_fill_feature, topology_primitive_count, topology_vertex_count, CullModeInfo,
    FrontFaceInfo, PolygonModeInfo, PrimitiveStateDescriptor, TopologyInfo, CULL_MODES,
    FRONT_FACES, POLYGON_MODES, TOPOLOGIES,
};

// Re-export depth/stencil state types
pub use depth_stencil_state::{
    get_compare_function_info, get_depth_format_info, get_depth_preset_info, has_stencil,
    is_depth_format, CompareFunctionInfo, DepthBiasStateDescriptor, DepthFormatInfo,
    DepthPresetInfo, DepthStencilStateDescriptor, StencilFaceStateDescriptor, COMPARE_FUNCTIONS,
    DEPTH_FORMATS, DEPTH_PRESETS,
};

// Re-export multisample state types
pub use multisample_state::{
    create_msaa_depth_texture, create_resolve_pair, get_sample_count_info,
    is_valid_resolve_target, is_valid_sample_count, query_supported_sample_counts,
    resolve_discard, resolve_store, select_max_supported_sample_count,
    select_sample_count_up_to, MsaaRenderTarget, MsaaResolveTarget, MsaaStoreOp,
    MultisampleStateBuilder, MultisampleStateDescriptor, ResolveAttachmentDescriptor,
    ResolveError, ResolveInfo, SampleCountInfo, SAMPLE_COUNTS,
};

// Re-export fragment state types
pub use fragment_state::{
    BlendComponentDescriptor, BlendStateDescriptor, ColorTargetStateDescriptor,
    FragmentStateDescriptor,
};

// Re-export pipeline cache types
pub use pipeline_cache::{
    hash_color_targets, hash_vertex_layout, CacheMetrics, PipelineKey, RenderPipelineCache,
};

// Re-export cache warming types
pub use cache_warming::{
    common_pipelines, ProgressCallback, WarmingConfig, WarmingHandle, WarmingProgress,
    WarmingResult,
};

// Re-export vertex format registry types
pub use vertex_format_registry::{
    particle, skinned_mesh, static_mesh, terrain, ui, VertexFormat, VertexFormatId,
    VertexFormatRegistry,
};

// Re-export vertex attribute utilities
pub use vertex_attribute::{
    calculate_offsets, calculate_stride, common as vertex_formats, strides as vertex_strides,
    vertex_attr_array, vertex_format_components, vertex_format_is_float,
    vertex_format_is_normalized, vertex_format_is_signed_int, vertex_format_is_unsigned_int,
    vertex_format_size, VertexFormatInfo,
};

// Re-export instance layout types
pub use instance_layout::{
    create_instance_layout, is_valid_instance_layout, presets as instance_presets,
    InstanceLayoutBuilder, INSTANCE_COLOR_LOCATION, INSTANCE_CUSTOM_START_LOCATION,
    INSTANCE_ID_LOCATION, INSTANCE_TRANSFORM_LOCATIONS, STRIDE_TRANSFORM,
    STRIDE_TRANSFORM_COLOR_CUSTOM, STRIDE_TRANSFORM_COLOR_FLOAT, STRIDE_TRANSFORM_COLOR_PACKED,
};

// Re-export viewport types
pub use viewport::{
    get_viewport_info, quadrant_viewport, set_scissor_rect, set_viewport,
    split_screen_bottom, split_screen_left, split_screen_right, split_screen_top,
    ScissorError, ScissorRect, Viewport, ViewportBuilder, ViewportError, ViewportInfo,
    VIEWPORT_PRESETS,
};

// Re-export depth bias types
pub use depth_bias::{
    get_depth_bias_info, get_preset as get_depth_bias_preset, preset_names as depth_bias_preset_names,
    DepthBias, DepthBiasBuilder, DepthBiasError, DepthBiasInfo, DEPTH_BIAS_PRESETS,
};

// Re-export conservative rasterization types
pub use conservative_raster::{
    get_conservative_raster_info, get_info_for_use_case, is_enabled_on_device, is_supported,
    required_features, use_case_names, ConservativeRasterBuilder, ConservativeRasterError,
    ConservativeRasterInfo, ConservativeRasterization, UseCase as ConservativeUseCase,
    CONSERVATIVE_RASTERIZATION_FEATURE, CONSERVATIVE_RASTER_USE_CASES,
};

// Re-export color target types
pub use color_target::{
    get_color_target_info, get_preset as get_color_target_preset, hdr_presets, preset_names as color_target_preset_names,
    srgb_presets, ColorTarget, ColorTargetArray, ColorTargetBuilder, ColorTargetError,
    ColorTargetInfo, COLOR_TARGET_PRESETS, MAX_COLOR_ATTACHMENTS,
};

// Re-export blend mode types
pub use blend_mode::{
    alpha_presets, blend_factor_names, blend_operation_names, constant_presets,
    get_blend_factor_info, get_blend_mode_info, get_blend_operation_info,
    get_preset as get_blend_mode_preset, hdr_presets as blend_hdr_presets,
    preset_names as blend_mode_preset_names, BlendFactorInfo, BlendMode, BlendModeBuilder,
    BlendModeInfo, BlendOperationInfo, BLEND_FACTORS, BLEND_MODE_PRESETS, BLEND_OPERATIONS,
};

// Re-export render pass types
pub use render_pass::{
    get_preset_info as get_render_pass_preset_info, preset_names as render_pass_preset_names,
    validate_color_attachment_count, validate_descriptor as validate_render_pass_descriptor,
    ColorAttachment as RenderPassColorAttachment, DepthStencilAttachment as RenderPassDepthStencilAttachment,
    LoadOp, OcclusionQuerySet, Operations, RenderPassBuilder, RenderPassDescriptor,
    RenderPassError, RenderPassInfo, StoreOp, TimestampWrites,
    DEFAULT_CLEAR_COLOR, DEFAULT_CLEAR_DEPTH, DEFAULT_CLEAR_STENCIL,
    MAX_COLOR_ATTACHMENTS as RENDER_PASS_MAX_COLOR_ATTACHMENTS, RENDER_PASS_PRESETS,
};

// Re-export render pass commands types
pub use render_pass_commands::{
    set_bind_group, set_blend_constant, set_index_buffer, set_pipeline, set_push_constants,
    set_stencil_reference, set_vertex_buffer, stencil_values, BlendConstantBuilder,
    RenderPassCommands,
};

// Re-export draw commands types
pub use draw_commands::{
    indirect_buffer_size, indirect_command_count, required_multi_draw_features,
    supports_multi_draw_indirect, supports_multi_draw_indirect_count, validate_indirect_offset,
    DrawCommands, DrawIndexedIndirectArgs, DrawIndirectArgs, MultiDrawTier,
};

// Re-export render bundle types
pub use render_bundle::{
    color_depth as bundle_color_depth, depth_only as bundle_depth_only,
    execute_bundles, execute_bundles_arc, gbuffer as bundle_gbuffer,
    msaa_color_depth as bundle_msaa_color_depth, simple_color as bundle_simple_color,
    BundleKey, CacheStats as BundleCacheStats, RenderBundleCache, RenderBundleEncoderDescriptor,
    RenderBundleError, RenderBundleRecorder,
};
