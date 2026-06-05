//! WebGPU-specific feature detection for TRINITY.
//!
//! This module provides detection of WebGPU capabilities and features for
//! browser-based rendering. WebGPU provides a modern, cross-platform graphics
//! API that runs in web browsers with near-native performance.
//!
//! # WebGPU Tiers
//!
//! WebGPU capabilities are organized into tiers based on hardware support:
//!
//! | Tier | Features |
//! |------|----------|
//! | Tier1 | Basic WebGPU: core shaders, limited textures |
//! | Tier2 | Extended limits: larger textures, storage textures |
//! | Tier3 | Compute + storage: timestamp queries, advanced compute |
//!
//! # Browser Compatibility
//!
//! | Browser | WebGPU Support |
//! |---------|----------------|
//! | Chrome | Full (113+) |
//! | Edge | Full (113+) |
//! | Firefox | Partial (Nightly) |
//! | Safari | Full (17+) |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::webgpu::{WebGpuFeatures, WebGpuTier};
//!
//! # async fn example() {
//! let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::BROWSER_WEBGPU,
//!     ..Default::default()
//! });
//!
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let features = WebGpuFeatures::detect(&adapter);
//!
//! match features.tier {
//!     WebGpuTier::Tier3 => println!("Full WebGPU compute support!"),
//!     WebGpuTier::Tier2 => println!("Extended WebGPU limits available"),
//!     WebGpuTier::Tier1 => println!("Basic WebGPU support"),
//! }
//!
//! if features.supports_compression() {
//!     println!("Texture compression formats: {:?}", features.compression_formats());
//! }
//! # }
//! ```

use log::debug;
use std::collections::HashSet;
use wgpu::{Adapter, Features, Limits};

// ============================================================================
// WebGpuTier
// ============================================================================

/// WebGPU capability tier.
///
/// WebGPU hardware support is classified into tiers that represent
/// increasing levels of GPU capability. Tiers are determined by
/// analyzing adapter limits and feature availability.
///
/// # Tier Definitions
///
/// | Tier | Description | Requirements |
/// |------|-------------|--------------|
/// | Tier1 | Basic | Core WebGPU spec, limited textures/buffers |
/// | Tier2 | Extended | Larger limits, storage textures |
/// | Tier3 | Advanced | Full compute, timestamp queries |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::WebGpuTier;
///
/// let tier = WebGpuTier::Tier2;
/// assert!(tier.supports_compute_shaders());
/// assert!(tier.supports_storage_textures());
/// assert!(!tier.supports_timestamp_query());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum WebGpuTier {
    /// Tier 1 - Basic WebGPU support.
    ///
    /// Minimum viable WebGPU implementation with core features:
    /// - Vertex and fragment shaders
    /// - Basic compute shaders
    /// - Limited texture dimensions (2048x2048)
    /// - Limited bind groups (4)
    /// - Basic texture formats
    #[default]
    Tier1,

    /// Tier 2 - Extended limits.
    ///
    /// Enhanced WebGPU with larger limits:
    /// - Larger texture dimensions (4096x4096 or higher)
    /// - More bind groups (8)
    /// - Storage textures in compute
    /// - More dynamic uniform buffers
    /// - Enhanced vertex buffer limits
    Tier2,

    /// Tier 3 - Full compute and advanced features.
    ///
    /// Complete WebGPU with all optional features:
    /// - Timestamp queries for profiling
    /// - Maximum texture dimensions (8192x8192+)
    /// - Full storage buffer support
    /// - Large workgroup sizes
    /// - Advanced compute features
    Tier3,
}

impl WebGpuTier {
    /// Detect WebGPU tier from adapter limits.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected WebGPU tier.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::webgpu::WebGpuTier;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let tier = WebGpuTier::from_adapter(&adapter);
    /// println!("WebGPU tier: {:?}", tier);
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let limits = adapter.limits();
        let features = adapter.features();
        Self::from_limits_and_features(&limits, features)
    }

    /// Detect tier from limits and features.
    ///
    /// # Arguments
    ///
    /// * `limits` - The wgpu limits from the adapter
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The detected tier based on limits and features.
    pub fn from_limits_and_features(limits: &Limits, features: Features) -> Self {
        // Tier 3 requirements: timestamp query and high limits
        let has_timestamp = features.contains(Features::TIMESTAMP_QUERY);
        let has_high_limits = limits.max_texture_dimension_2d >= 8192
            && limits.max_compute_workgroups_per_dimension >= 65535
            && limits.max_storage_buffers_per_shader_stage >= 8;

        if has_timestamp && has_high_limits {
            return WebGpuTier::Tier3;
        }

        // Tier 2 requirements: extended limits
        let has_extended_limits = limits.max_texture_dimension_2d >= 4096
            && limits.max_bind_groups >= 8
            && limits.max_storage_textures_per_shader_stage >= 4
            && limits.max_dynamic_uniform_buffers_per_pipeline_layout >= 8;

        if has_extended_limits {
            return WebGpuTier::Tier2;
        }

        WebGpuTier::Tier1
    }

    /// Check if compute shaders are supported.
    ///
    /// All WebGPU tiers support basic compute shaders.
    ///
    /// # Returns
    ///
    /// `true` for all tiers (WebGPU requires compute support).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuTier;
    ///
    /// assert!(WebGpuTier::Tier1.supports_compute_shaders());
    /// assert!(WebGpuTier::Tier2.supports_compute_shaders());
    /// assert!(WebGpuTier::Tier3.supports_compute_shaders());
    /// ```
    #[inline]
    pub const fn supports_compute_shaders(&self) -> bool {
        // WebGPU requires compute shader support at all tiers
        true
    }

    /// Check if storage textures are well-supported.
    ///
    /// Storage textures with good limits require Tier 2 or higher.
    ///
    /// # Returns
    ///
    /// `true` for Tier2 and Tier3.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuTier;
    ///
    /// assert!(!WebGpuTier::Tier1.supports_storage_textures());
    /// assert!(WebGpuTier::Tier2.supports_storage_textures());
    /// assert!(WebGpuTier::Tier3.supports_storage_textures());
    /// ```
    #[inline]
    pub const fn supports_storage_textures(&self) -> bool {
        matches!(self, WebGpuTier::Tier2 | WebGpuTier::Tier3)
    }

    /// Check if timestamp queries are supported.
    ///
    /// Timestamp queries require Tier 3.
    ///
    /// # Returns
    ///
    /// `true` only for Tier3.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuTier;
    ///
    /// assert!(!WebGpuTier::Tier1.supports_timestamp_query());
    /// assert!(!WebGpuTier::Tier2.supports_timestamp_query());
    /// assert!(WebGpuTier::Tier3.supports_timestamp_query());
    /// ```
    #[inline]
    pub const fn supports_timestamp_query(&self) -> bool {
        matches!(self, WebGpuTier::Tier3)
    }

    /// Get the tier name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the tier name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            WebGpuTier::Tier1 => "Tier 1 (Basic)",
            WebGpuTier::Tier2 => "Tier 2 (Extended)",
            WebGpuTier::Tier3 => "Tier 3 (Advanced)",
        }
    }

    /// Get the numeric tier value.
    ///
    /// # Returns
    ///
    /// The tier number (1-3).
    #[inline]
    pub const fn tier_number(&self) -> u8 {
        match self {
            WebGpuTier::Tier1 => 1,
            WebGpuTier::Tier2 => 2,
            WebGpuTier::Tier3 => 3,
        }
    }
}

impl std::fmt::Display for WebGpuTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// WebGpuLimits
// ============================================================================

/// WebGPU device limits.
///
/// This struct captures the WebGPU limits that constrain resource sizes
/// and shader capabilities. These limits vary significantly between
/// devices and browsers.
///
/// # Limit Categories
///
/// | Category | Limits |
/// |----------|--------|
/// | Textures | max_texture_dimension_1d/2d/3d |
/// | Bind Groups | max_bind_groups, max_bindings_per_bind_group |
/// | Buffers | max_uniform/storage_buffer_binding_size |
/// | Compute | max_compute_workgroup_size_x/y/z |
/// | Vertex | max_vertex_buffers, max_vertex_attributes |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::webgpu::WebGpuLimits;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let limits = WebGpuLimits::from_adapter(&adapter);
/// println!("Max 2D texture: {}x{}", limits.max_texture_dimension_2d, limits.max_texture_dimension_2d);
/// println!("Max compute workgroup: {}x{}x{}",
///     limits.max_compute_workgroup_size_x,
///     limits.max_compute_workgroup_size_y,
///     limits.max_compute_workgroup_size_z
/// );
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuLimits {
    /// Maximum 1D texture dimension.
    pub max_texture_dimension_1d: u32,

    /// Maximum 2D texture dimension.
    pub max_texture_dimension_2d: u32,

    /// Maximum 3D texture dimension.
    pub max_texture_dimension_3d: u32,

    /// Maximum number of bind groups.
    pub max_bind_groups: u32,

    /// Maximum bindings per bind group.
    pub max_bindings_per_bind_group: u32,

    /// Maximum dynamic uniform buffers per pipeline layout.
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,

    /// Maximum dynamic storage buffers per pipeline layout.
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,

    /// Maximum sampled textures per shader stage.
    pub max_sampled_textures_per_shader_stage: u32,

    /// Maximum samplers per shader stage.
    pub max_samplers_per_shader_stage: u32,

    /// Maximum storage buffers per shader stage.
    pub max_storage_buffers_per_shader_stage: u32,

    /// Maximum storage textures per shader stage.
    pub max_storage_textures_per_shader_stage: u32,

    /// Maximum uniform buffers per shader stage.
    pub max_uniform_buffers_per_shader_stage: u32,

    /// Maximum uniform buffer binding size.
    pub max_uniform_buffer_binding_size: u32,

    /// Maximum storage buffer binding size.
    pub max_storage_buffer_binding_size: u32,

    /// Maximum vertex buffers.
    pub max_vertex_buffers: u32,

    /// Maximum vertex attributes.
    pub max_vertex_attributes: u32,

    /// Maximum vertex buffer array stride.
    pub max_vertex_buffer_array_stride: u32,

    /// Maximum compute workgroup size in X dimension.
    pub max_compute_workgroup_size_x: u32,

    /// Maximum compute workgroup size in Y dimension.
    pub max_compute_workgroup_size_y: u32,

    /// Maximum compute workgroup size in Z dimension.
    pub max_compute_workgroup_size_z: u32,

    /// Maximum compute invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,

    /// Maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

impl WebGpuLimits {
    /// Extract limits from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The extracted limits.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::webgpu::WebGpuLimits;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let limits = WebGpuLimits::from_adapter(&adapter);
    /// println!("Limits: {:?}", limits);
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        Self::from_wgpu_limits(&adapter.limits())
    }

    /// Convert from wgpu::Limits.
    ///
    /// # Arguments
    ///
    /// * `limits` - The wgpu limits to convert
    ///
    /// # Returns
    ///
    /// The converted WebGpuLimits.
    pub fn from_wgpu_limits(limits: &Limits) -> Self {
        Self {
            max_texture_dimension_1d: limits.max_texture_dimension_1d,
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_texture_dimension_3d: limits.max_texture_dimension_3d,
            max_bind_groups: limits.max_bind_groups,
            max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout: limits
                .max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout: limits
                .max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage: limits.max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage: limits.max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_vertex_buffers: limits.max_vertex_buffers,
            max_vertex_attributes: limits.max_vertex_attributes,
            max_vertex_buffer_array_stride: limits.max_vertex_buffer_array_stride,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Check if these limits meet tier requirements.
    ///
    /// # Arguments
    ///
    /// * `tier` - The tier to check against
    ///
    /// # Returns
    ///
    /// `true` if the limits meet the tier requirements.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::{WebGpuLimits, WebGpuTier};
    ///
    /// let limits = WebGpuLimits::tier2_minimum();
    /// assert!(limits.meets_tier(WebGpuTier::Tier1));
    /// assert!(limits.meets_tier(WebGpuTier::Tier2));
    /// assert!(!limits.meets_tier(WebGpuTier::Tier3));
    /// ```
    pub fn meets_tier(&self, tier: WebGpuTier) -> bool {
        let required = match tier {
            WebGpuTier::Tier1 => Self::tier1_minimum(),
            WebGpuTier::Tier2 => Self::tier2_minimum(),
            WebGpuTier::Tier3 => Self::tier3_minimum(),
        };

        self.max_texture_dimension_1d >= required.max_texture_dimension_1d
            && self.max_texture_dimension_2d >= required.max_texture_dimension_2d
            && self.max_texture_dimension_3d >= required.max_texture_dimension_3d
            && self.max_bind_groups >= required.max_bind_groups
            && self.max_storage_textures_per_shader_stage
                >= required.max_storage_textures_per_shader_stage
            && self.max_compute_workgroups_per_dimension
                >= required.max_compute_workgroups_per_dimension
            && self.max_storage_buffers_per_shader_stage
                >= required.max_storage_buffers_per_shader_stage
    }

    /// Get Tier 1 minimum limits.
    ///
    /// These are the minimum required limits for basic WebGPU support.
    ///
    /// # Returns
    ///
    /// The Tier 1 minimum limits.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuLimits;
    ///
    /// let tier1 = WebGpuLimits::tier1_minimum();
    /// assert_eq!(tier1.max_texture_dimension_2d, 2048);
    /// assert_eq!(tier1.max_bind_groups, 4);
    /// ```
    pub const fn tier1_minimum() -> Self {
        Self {
            max_texture_dimension_1d: 2048,
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 256,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 640,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 4,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 16384, // 16 KB
            max_storage_buffer_binding_size: 134217728, // 128 MB
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
        }
    }

    /// Get Tier 2 minimum limits.
    ///
    /// These are the minimum required limits for extended WebGPU support.
    ///
    /// # Returns
    ///
    /// The Tier 2 minimum limits.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuLimits;
    ///
    /// let tier2 = WebGpuLimits::tier2_minimum();
    /// assert_eq!(tier2.max_texture_dimension_2d, 4096);
    /// assert_eq!(tier2.max_bind_groups, 8);
    /// ```
    pub const fn tier2_minimum() -> Self {
        Self {
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 512,
            max_bind_groups: 8,
            max_bindings_per_bind_group: 1000,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 8,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 65536, // 64 KB
            max_storage_buffer_binding_size: 268435456, // 256 MB
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
        }
    }

    /// Get Tier 3 minimum limits.
    ///
    /// These are the minimum required limits for advanced WebGPU support.
    ///
    /// # Returns
    ///
    /// The Tier 3 minimum limits.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuLimits;
    ///
    /// let tier3 = WebGpuLimits::tier3_minimum();
    /// assert_eq!(tier3.max_texture_dimension_2d, 8192);
    /// assert!(tier3.max_storage_buffers_per_shader_stage >= 8);
    /// ```
    pub const fn tier3_minimum() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_bind_groups: 8,
            max_bindings_per_bind_group: 1000,
            max_dynamic_uniform_buffers_per_pipeline_layout: 12,
            max_dynamic_storage_buffers_per_pipeline_layout: 8,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 65536, // 64 KB
            max_storage_buffer_binding_size: 536870912, // 512 MB
            max_vertex_buffers: 8,
            max_vertex_attributes: 32,
            max_vertex_buffer_array_stride: 2048,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}

impl Default for WebGpuLimits {
    fn default() -> Self {
        Self::tier1_minimum()
    }
}

// ============================================================================
// WebGpuFeatures
// ============================================================================

/// WebGPU-specific feature detection.
///
/// This struct captures WebGPU capabilities including tier classification,
/// device limits, and optional feature flags. Detection is performed by
/// inspecting wgpu's feature flags and adapter info.
///
/// # Feature Categories
///
/// | Category | Features |
/// |----------|----------|
/// | Tier | tier (Tier1/Tier2/Tier3) |
/// | Compression | texture_compression_bc/etc2/astc |
/// | Depth | depth_clip_control, depth formats |
/// | Compute | timestamp_query, indirect_first_instance |
/// | Shader | shader_f16, float32_filterable |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::webgpu::WebGpuFeatures;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let features = WebGpuFeatures::detect(&adapter);
/// println!("Tier: {}", features.tier);
/// println!("Compression: {:?}", features.compression_formats());
/// println!("Summary: {}", features.summary());
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGpuFeatures {
    /// Detected WebGPU tier.
    pub tier: WebGpuTier,

    /// Device limits.
    pub limits: WebGpuLimits,

    /// Depth clip control support.
    ///
    /// Allows controlling depth clipping behavior in the rasterizer.
    pub depth_clip_control: bool,

    /// Depth24UnormStencil8 format support.
    ///
    /// Combined depth-stencil format with 24-bit unorm depth.
    pub depth24_unorm_stencil8: bool,

    /// Depth32FloatStencil8 format support.
    ///
    /// Combined depth-stencil format with 32-bit float depth.
    pub depth32_float_stencil8: bool,

    /// BC texture compression support.
    ///
    /// Block compression formats (DXT/S3TC). Common on desktop.
    pub texture_compression_bc: bool,

    /// ETC2 texture compression support.
    ///
    /// Ericsson Texture Compression 2. Common on mobile/WebGL.
    pub texture_compression_etc2: bool,

    /// ASTC texture compression support.
    ///
    /// Adaptive Scalable Texture Compression. Common on mobile.
    pub texture_compression_astc: bool,

    /// Timestamp query support.
    ///
    /// Allows GPU timestamp queries for profiling.
    pub timestamp_query: bool,

    /// Indirect first instance support.
    ///
    /// Allows first_instance in indirect draw calls.
    pub indirect_first_instance: bool,

    /// Shader f16 support.
    ///
    /// Native 16-bit float operations in shaders.
    pub shader_f16: bool,

    /// RG11B10UFloat renderable support.
    ///
    /// Allows RG11B10UFloat format as render target.
    pub rg11b10_ufloat_renderable: bool,

    /// BGRA8Unorm storage support.
    ///
    /// Allows BGRA8Unorm format in storage textures.
    pub bgra8_unorm_storage: bool,

    /// Float32 filterable support.
    ///
    /// Allows filtering of 32-bit float textures.
    pub float32_filterable: bool,
}

impl WebGpuFeatures {
    /// Detect WebGPU features from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected WebGPU features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let features = WebGpuFeatures::detect(&adapter);
    /// println!("WebGPU tier: {}", features.tier);
    /// # }
    /// ```
    pub fn detect(adapter: &Adapter) -> Self {
        let wgpu_features = adapter.features();
        let limits = adapter.limits();
        Self::from_adapter_info(&limits, wgpu_features)
    }

    /// Create features from adapter limits and feature flags.
    ///
    /// # Arguments
    ///
    /// * `limits` - The wgpu limits from the adapter
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The mapped WebGPU features.
    pub fn from_adapter_info(limits: &Limits, features: Features) -> Self {
        let webgpu_limits = WebGpuLimits::from_wgpu_limits(limits);
        let tier = WebGpuTier::from_limits_and_features(limits, features);

        // Feature detection
        let depth_clip_control = features.contains(Features::DEPTH_CLIP_CONTROL);
        // Note: DEPTH24UNORM_STENCIL8 may not be exposed by all wgpu versions
        // We default to false and could detect it from texture format support
        let depth24_unorm_stencil8 = false; // Not directly exposed in wgpu 22.x
        let depth32_float_stencil8 = features.contains(Features::DEPTH32FLOAT_STENCIL8);
        let texture_compression_bc = features.contains(Features::TEXTURE_COMPRESSION_BC);
        let texture_compression_etc2 = features.contains(Features::TEXTURE_COMPRESSION_ETC2);
        let texture_compression_astc = features.contains(Features::TEXTURE_COMPRESSION_ASTC);
        let timestamp_query = features.contains(Features::TIMESTAMP_QUERY);
        let indirect_first_instance = features.contains(Features::INDIRECT_FIRST_INSTANCE);
        let shader_f16 = features.contains(Features::SHADER_F16);
        let rg11b10_ufloat_renderable = features.contains(Features::RG11B10UFLOAT_RENDERABLE);
        let bgra8_unorm_storage = features.contains(Features::BGRA8UNORM_STORAGE);
        let float32_filterable = features.contains(Features::FLOAT32_FILTERABLE);

        let result = Self {
            tier,
            limits: webgpu_limits,
            depth_clip_control,
            depth24_unorm_stencil8,
            depth32_float_stencil8,
            texture_compression_bc,
            texture_compression_etc2,
            texture_compression_astc,
            timestamp_query,
            indirect_first_instance,
            shader_f16,
            rg11b10_ufloat_renderable,
            bgra8_unorm_storage,
            float32_filterable,
        };

        debug!(
            "WebGpuFeatures detected: tier={:?}, BC={}, ETC2={}, ASTC={}, timestamp={}",
            tier, texture_compression_bc, texture_compression_etc2, texture_compression_astc, timestamp_query
        );

        result
    }

    /// Check if any texture compression is supported.
    ///
    /// # Returns
    ///
    /// `true` if BC, ETC2, or ASTC compression is available.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// let mut features = WebGpuFeatures::default();
    /// assert!(!features.supports_compression());
    ///
    /// features.texture_compression_bc = true;
    /// assert!(features.supports_compression());
    /// ```
    #[inline]
    pub const fn supports_compression(&self) -> bool {
        self.texture_compression_bc
            || self.texture_compression_etc2
            || self.texture_compression_astc
    }

    /// Get the list of supported compression formats.
    ///
    /// # Returns
    ///
    /// A vector of supported compression format names.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// let mut features = WebGpuFeatures::default();
    /// features.texture_compression_bc = true;
    /// features.texture_compression_astc = true;
    ///
    /// let formats = features.compression_formats();
    /// assert!(formats.contains(&"BC"));
    /// assert!(formats.contains(&"ASTC"));
    /// assert!(!formats.contains(&"ETC2"));
    /// ```
    pub fn compression_formats(&self) -> Vec<&'static str> {
        let mut formats = Vec::new();
        if self.texture_compression_bc {
            formats.push("BC");
        }
        if self.texture_compression_etc2 {
            formats.push("ETC2");
        }
        if self.texture_compression_astc {
            formats.push("ASTC");
        }
        formats
    }

    /// Check if optimized for mobile devices.
    ///
    /// Mobile optimization is indicated by ETC2/ASTC support without BC.
    ///
    /// # Returns
    ///
    /// `true` if the device appears mobile-optimized.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// let mut features = WebGpuFeatures::default();
    /// features.texture_compression_etc2 = true;
    /// features.texture_compression_astc = true;
    /// assert!(features.is_mobile_optimized());
    ///
    /// features.texture_compression_bc = true;
    /// assert!(!features.is_mobile_optimized()); // BC indicates desktop
    /// ```
    #[inline]
    pub const fn is_mobile_optimized(&self) -> bool {
        (self.texture_compression_etc2 || self.texture_compression_astc)
            && !self.texture_compression_bc
    }

    /// Check if optimized for desktop devices.
    ///
    /// Desktop optimization is indicated by BC support without ETC2/ASTC.
    ///
    /// # Returns
    ///
    /// `true` if the device appears desktop-optimized.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// let mut features = WebGpuFeatures::default();
    /// features.texture_compression_bc = true;
    /// assert!(features.is_desktop_optimized());
    ///
    /// features.texture_compression_etc2 = true;
    /// assert!(!features.is_desktop_optimized()); // ETC2 indicates mobile
    /// ```
    #[inline]
    pub const fn is_desktop_optimized(&self) -> bool {
        self.texture_compression_bc
            && !self.texture_compression_etc2
            && !self.texture_compression_astc
    }

    /// Create a human-readable summary of features.
    ///
    /// # Returns
    ///
    /// A string summarizing the detected features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::webgpu::WebGpuFeatures;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let features = WebGpuFeatures::detect(&adapter);
    /// println!("{}", features.summary());
    /// // Output: "Tier 2 (Extended), BC, ETC2, Timestamp, FP16"
    /// # }
    /// ```
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        parts.push(self.tier.name().to_string());

        // Compression formats
        if self.texture_compression_bc {
            parts.push("BC".to_string());
        }
        if self.texture_compression_etc2 {
            parts.push("ETC2".to_string());
        }
        if self.texture_compression_astc {
            parts.push("ASTC".to_string());
        }

        // Key features
        if self.timestamp_query {
            parts.push("Timestamp".to_string());
        }
        if self.shader_f16 {
            parts.push("FP16".to_string());
        }
        if self.indirect_first_instance {
            parts.push("Indirect".to_string());
        }
        if self.float32_filterable {
            parts.push("F32Filter".to_string());
        }

        // Depth features
        if self.depth32_float_stencil8 {
            parts.push("D32S8".to_string());
        }

        if parts.len() == 1 {
            parts.push("Basic".to_string());
        }

        parts.join(", ")
    }
}

impl Default for WebGpuFeatures {
    fn default() -> Self {
        Self {
            tier: WebGpuTier::Tier1,
            limits: WebGpuLimits::default(),
            depth_clip_control: false,
            depth24_unorm_stencil8: false,
            depth32_float_stencil8: false,
            texture_compression_bc: false,
            texture_compression_etc2: false,
            texture_compression_astc: false,
            timestamp_query: false,
            indirect_first_instance: false,
            shader_f16: false,
            rg11b10_ufloat_renderable: false,
            bgra8_unorm_storage: false,
            float32_filterable: false,
        }
    }
}

// ============================================================================
// BrowserType
// ============================================================================

/// Browser type detection for WASM targets.
///
/// Different browsers have varying WebGPU implementations and capabilities.
/// This enum helps identify the browser for feature detection and workarounds.
///
/// # Browser Support
///
/// | Browser | WebGPU Status | Notes |
/// |---------|---------------|-------|
/// | Chrome | Full | Reference implementation |
/// | Edge | Full | Chromium-based |
/// | Firefox | Partial | Nightly builds only |
/// | Safari | Full | WebKit implementation |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::BrowserType;
///
/// let browser = BrowserType::Chrome;
/// assert!(browser.has_stable_webgpu());
/// assert!(browser.supports_offscreen_canvas());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BrowserType {
    /// Google Chrome browser.
    ///
    /// Reference WebGPU implementation with full feature support.
    Chrome,

    /// Mozilla Firefox browser.
    ///
    /// WebGPU support in nightly builds, some features missing.
    Firefox,

    /// Apple Safari browser.
    ///
    /// WebKit-based implementation with good Metal backend.
    Safari,

    /// Microsoft Edge browser.
    ///
    /// Chromium-based, matches Chrome capabilities.
    Edge,

    /// Unknown or undetected browser.
    #[default]
    Unknown,
}

impl BrowserType {
    /// Parse browser type from user agent string.
    ///
    /// # Arguments
    ///
    /// * `user_agent` - The browser user agent string
    ///
    /// # Returns
    ///
    /// The detected browser type.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::BrowserType;
    ///
    /// let browser = BrowserType::from_user_agent(
    ///     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0"
    /// );
    /// assert_eq!(browser, BrowserType::Chrome);
    /// ```
    pub fn from_user_agent(user_agent: &str) -> Self {
        let ua_lower = user_agent.to_lowercase();

        // Check Edge first (it also contains "Chrome")
        if ua_lower.contains("edg/") || ua_lower.contains("edge/") {
            return BrowserType::Edge;
        }

        // Chrome check
        if ua_lower.contains("chrome/") || ua_lower.contains("chromium/") {
            return BrowserType::Chrome;
        }

        // Firefox check
        if ua_lower.contains("firefox/") || ua_lower.contains("gecko/") {
            return BrowserType::Firefox;
        }

        // Safari check (must be after Chrome check)
        if ua_lower.contains("safari/") && ua_lower.contains("version/") {
            return BrowserType::Safari;
        }

        BrowserType::Unknown
    }

    /// Check if the browser has stable WebGPU support.
    ///
    /// # Returns
    ///
    /// `true` for Chrome, Edge, and Safari.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::BrowserType;
    ///
    /// assert!(BrowserType::Chrome.has_stable_webgpu());
    /// assert!(BrowserType::Edge.has_stable_webgpu());
    /// assert!(BrowserType::Safari.has_stable_webgpu());
    /// assert!(!BrowserType::Firefox.has_stable_webgpu());
    /// ```
    #[inline]
    pub const fn has_stable_webgpu(&self) -> bool {
        matches!(
            self,
            BrowserType::Chrome | BrowserType::Edge | BrowserType::Safari
        )
    }

    /// Check if the browser supports OffscreenCanvas.
    ///
    /// OffscreenCanvas allows rendering in web workers.
    ///
    /// # Returns
    ///
    /// `true` for Chrome, Edge, and Firefox.
    #[inline]
    pub const fn supports_offscreen_canvas(&self) -> bool {
        matches!(
            self,
            BrowserType::Chrome | BrowserType::Edge | BrowserType::Firefox
        )
    }

    /// Check if the browser supports SharedArrayBuffer.
    ///
    /// SharedArrayBuffer enables shared memory between workers.
    ///
    /// # Returns
    ///
    /// `true` for browsers that support SharedArrayBuffer with proper headers.
    #[inline]
    pub const fn supports_shared_array_buffer(&self) -> bool {
        // All modern browsers support SAB with proper COOP/COEP headers
        matches!(
            self,
            BrowserType::Chrome | BrowserType::Edge | BrowserType::Firefox | BrowserType::Safari
        )
    }

    /// Get the browser name.
    ///
    /// # Returns
    ///
    /// A static string with the browser name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            BrowserType::Chrome => "Chrome",
            BrowserType::Firefox => "Firefox",
            BrowserType::Safari => "Safari",
            BrowserType::Edge => "Edge",
            BrowserType::Unknown => "Unknown",
        }
    }

    /// Get the recommended WebGPU backend for this browser.
    ///
    /// # Returns
    ///
    /// A description of the underlying graphics API.
    #[inline]
    pub const fn webgpu_backend(&self) -> &'static str {
        match self {
            BrowserType::Chrome | BrowserType::Edge => "Dawn (Vulkan/D3D12/Metal)",
            BrowserType::Firefox => "wgpu-native (Vulkan/D3D12/Metal)",
            BrowserType::Safari => "WebKit (Metal)",
            BrowserType::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for BrowserType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// BrowserCapabilities
// ============================================================================

/// Browser-specific WebGPU capabilities.
///
/// This struct provides browser detection and capability queries for
/// WASM targets. It helps identify browser-specific features and limitations.
///
/// # Feature Gating
///
/// This struct is primarily useful on `wasm32` targets. On native targets,
/// it provides sensible defaults.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::BrowserCapabilities;
///
/// let caps = BrowserCapabilities::default();
/// println!("Browser: {}", caps.browser);
/// println!("Offscreen canvas: {}", caps.supports_offscreen_canvas);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BrowserCapabilities {
    /// Detected browser type.
    pub browser: BrowserType,

    /// WebGPU version string (if available).
    pub webgpu_version: Option<String>,

    /// OffscreenCanvas support.
    ///
    /// Allows rendering in web workers.
    pub supports_offscreen_canvas: bool,

    /// SharedArrayBuffer support.
    ///
    /// Enables shared memory between workers.
    pub supports_shared_array_buffer: bool,

    /// Maximum canvas dimensions.
    ///
    /// Some browsers limit maximum canvas size.
    pub max_canvas_size: (u32, u32),
}

impl BrowserCapabilities {
    /// Create capabilities from browser detection.
    ///
    /// On WASM targets, this queries the browser environment.
    /// On native targets, this returns sensible defaults.
    ///
    /// # Returns
    ///
    /// The detected browser capabilities.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::BrowserCapabilities;
    ///
    /// let caps = BrowserCapabilities::detect();
    /// println!("Browser: {}", caps.browser);
    /// ```
    #[cfg(target_arch = "wasm32")]
    pub fn detect() -> Self {
        // On WASM, we would use web_sys to query the browser
        // For now, return defaults - actual detection requires JS interop
        Self::default()
    }

    /// Create capabilities from browser detection (native fallback).
    ///
    /// On native targets, this returns defaults indicating non-browser context.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn detect() -> Self {
        Self {
            browser: BrowserType::Unknown,
            webgpu_version: None,
            supports_offscreen_canvas: false,
            supports_shared_array_buffer: false,
            max_canvas_size: (16384, 16384), // Native has no limit
        }
    }

    /// Create capabilities from a user agent string.
    ///
    /// # Arguments
    ///
    /// * `user_agent` - The browser user agent string
    ///
    /// # Returns
    ///
    /// The capabilities based on user agent.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::BrowserCapabilities;
    ///
    /// let caps = BrowserCapabilities::from_user_agent(
    ///     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    /// );
    /// assert!(caps.supports_offscreen_canvas);
    /// ```
    pub fn from_user_agent(user_agent: &str) -> Self {
        let browser = BrowserType::from_user_agent(user_agent);

        // Determine capabilities based on browser
        let supports_offscreen_canvas = browser.supports_offscreen_canvas();
        let supports_shared_array_buffer = browser.supports_shared_array_buffer();

        // Maximum canvas size varies by browser
        let max_canvas_size = match browser {
            BrowserType::Chrome | BrowserType::Edge => (32767, 32767),
            BrowserType::Firefox => (32767, 32767),
            BrowserType::Safari => (16384, 16384),
            BrowserType::Unknown => (8192, 8192),
        };

        Self {
            browser,
            webgpu_version: None,
            supports_offscreen_canvas,
            supports_shared_array_buffer,
            max_canvas_size,
        }
    }

    /// Check if the browser has full WebGPU support.
    ///
    /// # Returns
    ///
    /// `true` if the browser has stable, complete WebGPU.
    #[inline]
    pub fn has_full_webgpu(&self) -> bool {
        self.browser.has_stable_webgpu()
    }

    /// Create a summary of browser capabilities.
    ///
    /// # Returns
    ///
    /// A human-readable summary string.
    pub fn summary(&self) -> String {
        let mut parts = vec![self.browser.name().to_string()];

        if let Some(ref version) = self.webgpu_version {
            parts.push(format!("WebGPU {}", version));
        }

        if self.supports_offscreen_canvas {
            parts.push("OffscreenCanvas".to_string());
        }
        if self.supports_shared_array_buffer {
            parts.push("SAB".to_string());
        }

        parts.push(format!("Max {}x{}", self.max_canvas_size.0, self.max_canvas_size.1));

        parts.join(", ")
    }
}

// ============================================================================
// Browser (T-WGPU-P7.2.4 requirement)
// ============================================================================

/// Browser identification for WebGPU limitation detection.
///
/// This enum identifies specific browsers and their WebGPU capabilities,
/// distinct from the more general `BrowserType` which focuses on browser
/// identification. `Browser` is used specifically for querying WebGPU
/// limitations and feature support.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::Browser;
///
/// let browser = Browser::Chrome;
/// assert!(browser.supports_timestamp_query());
/// assert!(browser.supports_indirect_first_instance());
/// assert_eq!(browser.max_texture_dimension(), 16384);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Browser {
    /// Google Chrome (Chromium-based).
    ///
    /// Reference WebGPU implementation via Dawn. Full feature support
    /// with highest limits among browsers.
    Chrome,

    /// Mozilla Firefox.
    ///
    /// WebGPU support via wgpu-native. Some features may be missing
    /// or have lower limits compared to Chrome.
    Firefox,

    /// Apple Safari.
    ///
    /// WebKit-based WebGPU implementation backed by Metal.
    /// Good feature support but some differences from Chrome.
    Safari,

    /// Microsoft Edge (Chromium-based).
    ///
    /// Same as Chrome since it uses the Chromium engine.
    Edge,

    /// Unknown or undetected browser.
    #[default]
    Unknown,
}

impl Browser {
    /// Parse browser from user agent string.
    ///
    /// # Arguments
    ///
    /// * `user_agent` - The browser user agent string
    ///
    /// # Returns
    ///
    /// The detected browser.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::Browser;
    ///
    /// let browser = Browser::from_user_agent("Chrome/120.0.0.0");
    /// assert_eq!(browser, Browser::Chrome);
    /// ```
    pub fn from_user_agent(user_agent: &str) -> Self {
        let ua_lower = user_agent.to_lowercase();

        // Edge check first (contains "Chrome")
        if ua_lower.contains("edg/") || ua_lower.contains("edge/") {
            return Browser::Edge;
        }

        // Chrome check
        if ua_lower.contains("chrome/") || ua_lower.contains("chromium/") {
            return Browser::Chrome;
        }

        // Firefox check
        if ua_lower.contains("firefox/") || ua_lower.contains("gecko/") {
            return Browser::Firefox;
        }

        // Safari check (must be after Chrome)
        if ua_lower.contains("safari/") && ua_lower.contains("version/") {
            return Browser::Safari;
        }

        Browser::Unknown
    }

    /// Check if browser supports timestamp queries.
    ///
    /// Timestamp queries allow GPU-side profiling. Not all browsers
    /// expose this feature due to privacy concerns.
    ///
    /// # Returns
    ///
    /// `true` if timestamp queries are supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::Browser;
    ///
    /// assert!(Browser::Chrome.supports_timestamp_query());
    /// assert!(Browser::Edge.supports_timestamp_query());
    /// assert!(!Browser::Firefox.supports_timestamp_query()); // Nightly only
    /// assert!(!Browser::Safari.supports_timestamp_query()); // Privacy restricted
    /// ```
    #[inline]
    pub const fn supports_timestamp_query(&self) -> bool {
        // Chrome and Edge support timestamp queries
        // Firefox has it in nightly but not stable
        // Safari restricts due to privacy
        matches!(self, Browser::Chrome | Browser::Edge)
    }

    /// Check if browser supports indirect draw with first_instance.
    ///
    /// Some browsers don't support the first_instance field in indirect
    /// draw calls, requiring workarounds.
    ///
    /// # Returns
    ///
    /// `true` if indirect_first_instance is supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::Browser;
    ///
    /// assert!(Browser::Chrome.supports_indirect_first_instance());
    /// assert!(Browser::Edge.supports_indirect_first_instance());
    /// assert!(!Browser::Safari.supports_indirect_first_instance());
    /// ```
    #[inline]
    pub const fn supports_indirect_first_instance(&self) -> bool {
        // Chrome/Edge support this via Dawn
        // Firefox partially supports
        // Safari does not expose this
        matches!(self, Browser::Chrome | Browser::Edge | Browser::Firefox)
    }

    /// Get the maximum texture dimension for this browser.
    ///
    /// Different browsers have different maximum texture size limits.
    ///
    /// # Returns
    ///
    /// Maximum texture dimension in pixels.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::Browser;
    ///
    /// assert_eq!(Browser::Chrome.max_texture_dimension(), 16384);
    /// assert_eq!(Browser::Safari.max_texture_dimension(), 8192);
    /// ```
    #[inline]
    pub const fn max_texture_dimension(&self) -> u32 {
        match self {
            Browser::Chrome | Browser::Edge => 16384,
            Browser::Firefox => 16384,
            Browser::Safari => 8192, // Safari has lower limits on iOS
            Browser::Unknown => 4096, // Conservative default
        }
    }

    /// Get the browser name.
    ///
    /// # Returns
    ///
    /// A static string with the browser name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Browser::Chrome => "Chrome",
            Browser::Firefox => "Firefox",
            Browser::Safari => "Safari",
            Browser::Edge => "Edge",
            Browser::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for Browser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// WebGPULimits (T-WGPU-P7.2.4 requirement)
// ============================================================================

/// WebGPU spec-compliant limits structure.
///
/// This struct mirrors the WebGPU specification's supported limits,
/// with constants for spec minimums and browser-specific defaults.
///
/// # Spec Reference
///
/// See <https://www.w3.org/TR/webgpu/#limits> for the official spec.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::WebGPULimits;
///
/// let spec_min = WebGPULimits::SPEC_MINIMUM;
/// assert_eq!(spec_min.max_texture_dimension_2d, 8192);
/// assert_eq!(spec_min.max_bind_groups, 4);
///
/// let chrome = WebGPULimits::CHROME_DEFAULTS;
/// assert!(chrome.max_texture_dimension_2d >= spec_min.max_texture_dimension_2d);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGPULimits {
    /// Maximum 1D texture dimension.
    pub max_texture_dimension_1d: u32,

    /// Maximum 2D texture dimension.
    pub max_texture_dimension_2d: u32,

    /// Maximum 3D texture dimension.
    pub max_texture_dimension_3d: u32,

    /// Maximum texture array layers.
    pub max_texture_array_layers: u32,

    /// Maximum number of bind groups.
    pub max_bind_groups: u32,

    /// Maximum bindings per bind group.
    pub max_bindings_per_bind_group: u32,

    /// Maximum dynamic uniform buffers per pipeline layout.
    pub max_dynamic_uniform_buffers: u32,

    /// Maximum dynamic storage buffers per pipeline layout.
    pub max_dynamic_storage_buffers: u32,

    /// Maximum sampled textures per shader stage.
    pub max_sampled_textures_per_shader_stage: u32,

    /// Maximum samplers per shader stage.
    pub max_samplers_per_shader_stage: u32,

    /// Maximum storage buffers per shader stage.
    pub max_storage_buffers_per_shader_stage: u32,

    /// Maximum storage textures per shader stage.
    pub max_storage_textures_per_shader_stage: u32,

    /// Maximum uniform buffers per shader stage.
    pub max_uniform_buffers_per_shader_stage: u32,

    /// Maximum uniform buffer binding size in bytes.
    pub max_uniform_buffer_binding_size: u32,

    /// Maximum storage buffer binding size in bytes.
    pub max_storage_buffer_binding_size: u32,

    /// Maximum vertex buffers.
    pub max_vertex_buffers: u32,

    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,

    /// Maximum vertex attributes.
    pub max_vertex_attributes: u32,

    /// Maximum vertex buffer array stride.
    pub max_vertex_buffer_array_stride: u32,

    /// Maximum compute workgroup size in X dimension.
    pub max_compute_workgroup_size_x: u32,

    /// Maximum compute workgroup size in Y dimension.
    pub max_compute_workgroup_size_y: u32,

    /// Maximum compute workgroup size in Z dimension.
    pub max_compute_workgroup_size_z: u32,

    /// Maximum compute invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,

    /// Maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

impl WebGPULimits {
    /// WebGPU specification minimum required limits.
    ///
    /// These are the absolute minimums that any WebGPU implementation
    /// must support per the W3C specification.
    pub const SPEC_MINIMUM: Self = Self {
        max_texture_dimension_1d: 8192,
        max_texture_dimension_2d: 8192,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 256,
        max_bind_groups: 4,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers: 8,
        max_dynamic_storage_buffers: 4,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65536, // 64 KB
        max_storage_buffer_binding_size: 134217728, // 128 MB
        max_vertex_buffers: 8,
        max_buffer_size: 268435456, // 256 MB
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 65535,
    };

    /// Chrome browser default limits (via Dawn).
    ///
    /// Chrome typically exposes higher limits than the spec minimum
    /// when running on capable hardware.
    pub const CHROME_DEFAULTS: Self = Self {
        max_texture_dimension_1d: 16384,
        max_texture_dimension_2d: 16384,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 2048,
        max_bind_groups: 8,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers: 8,
        max_dynamic_storage_buffers: 8,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 10,
        max_storage_textures_per_shader_stage: 8,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65536, // 64 KB
        max_storage_buffer_binding_size: 1073741824, // 1 GB
        max_vertex_buffers: 8,
        max_buffer_size: 2147483648, // 2 GB
        max_vertex_attributes: 30,
        max_vertex_buffer_array_stride: 2048,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 65535,
    };

    /// Firefox browser default limits (via wgpu-native).
    ///
    /// Firefox's WebGPU implementation uses wgpu-native with
    /// slightly different limits than Chrome.
    pub const FIREFOX_DEFAULTS: Self = Self {
        max_texture_dimension_1d: 16384,
        max_texture_dimension_2d: 16384,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 2048,
        max_bind_groups: 8,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers: 8,
        max_dynamic_storage_buffers: 8,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 8,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65536, // 64 KB
        max_storage_buffer_binding_size: 1073741824, // 1 GB
        max_vertex_buffers: 8,
        max_buffer_size: 2147483648, // 2 GB
        max_vertex_attributes: 32,
        max_vertex_buffer_array_stride: 2048,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 65535,
    };

    /// Safari browser default limits (via WebKit/Metal).
    ///
    /// Safari's WebGPU implementation has some iOS-influenced
    /// limits that may be lower than other browsers.
    pub const SAFARI_DEFAULTS: Self = Self {
        max_texture_dimension_1d: 8192,
        max_texture_dimension_2d: 8192,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 2048,
        max_bind_groups: 4,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers: 8,
        max_dynamic_storage_buffers: 4,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65536, // 64 KB
        max_storage_buffer_binding_size: 268435456, // 256 MB
        max_vertex_buffers: 8,
        max_buffer_size: 1073741824, // 1 GB
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroups_per_dimension: 65535,
    };

    /// Create limits from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The extracted limits.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::webgpu::WebGPULimits;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let limits = WebGPULimits::from_adapter(&adapter);
    /// println!("Max 2D texture: {}", limits.max_texture_dimension_2d);
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let limits = adapter.limits();
        Self {
            max_texture_dimension_1d: limits.max_texture_dimension_1d,
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_texture_dimension_3d: limits.max_texture_dimension_3d,
            max_texture_array_layers: limits.max_texture_array_layers,
            max_bind_groups: limits.max_bind_groups,
            max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
            max_dynamic_uniform_buffers: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers: limits.max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage: limits.max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage: limits.max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_vertex_buffers: limits.max_vertex_buffers,
            max_buffer_size: limits.max_buffer_size,
            max_vertex_attributes: limits.max_vertex_attributes,
            max_vertex_buffer_array_stride: limits.max_vertex_buffer_array_stride,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Get default limits for a browser.
    ///
    /// # Arguments
    ///
    /// * `browser` - The browser to get defaults for
    ///
    /// # Returns
    ///
    /// The browser-specific default limits.
    pub const fn for_browser(browser: Browser) -> Self {
        match browser {
            Browser::Chrome | Browser::Edge => Self::CHROME_DEFAULTS,
            Browser::Firefox => Self::FIREFOX_DEFAULTS,
            Browser::Safari => Self::SAFARI_DEFAULTS,
            Browser::Unknown => Self::SPEC_MINIMUM,
        }
    }
}

impl Default for WebGPULimits {
    fn default() -> Self {
        Self::SPEC_MINIMUM
    }
}

// ============================================================================
// WebGPUFeature (T-WGPU-P7.2.4 requirement)
// ============================================================================

/// WebGPU optional feature flags.
///
/// These are optional features that may or may not be supported
/// by a given browser or device. Feature detection is required
/// before using these capabilities.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::WebGPUFeature;
///
/// let feature = WebGPUFeature::TimestampQuery;
/// assert_eq!(feature.name(), "timestamp-query");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebGPUFeature {
    /// GPU timestamp queries for profiling.
    ///
    /// Allows recording GPU timestamps for performance measurement.
    /// May be restricted by browsers for fingerprinting prevention.
    TimestampQuery,

    /// Indirect draw with first_instance support.
    ///
    /// Allows using the first_instance field in indirect draw calls.
    /// Not universally supported across all browsers.
    IndirectFirstInstance,

    /// Depth32Float + Stencil8 combined format.
    ///
    /// 32-bit float depth with 8-bit stencil in a single texture.
    Depth32FloatStencil8,

    /// BGRA8Unorm storage texture support.
    ///
    /// Allows using BGRA8Unorm format for storage textures.
    Bgra8UnormStorage,

    /// Pipeline statistics queries.
    ///
    /// Allows querying GPU pipeline statistics like vertex/fragment
    /// invocation counts. Often disabled for privacy.
    PipelineStatistics,

    /// Shader float16 support.
    ///
    /// Native 16-bit floating point operations in shaders.
    ShaderFloat16,

    /// RG11B10UFloat renderable format.
    ///
    /// Allows using RG11B10UFloat as a render target.
    RG11B10UFloat,
}

impl WebGPUFeature {
    /// Get the WebGPU spec feature name.
    ///
    /// # Returns
    ///
    /// The canonical feature name string.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            WebGPUFeature::TimestampQuery => "timestamp-query",
            WebGPUFeature::IndirectFirstInstance => "indirect-first-instance",
            WebGPUFeature::Depth32FloatStencil8 => "depth32float-stencil8",
            WebGPUFeature::Bgra8UnormStorage => "bgra8unorm-storage",
            WebGPUFeature::PipelineStatistics => "pipeline-statistics-query",
            WebGPUFeature::ShaderFloat16 => "shader-f16",
            WebGPUFeature::RG11B10UFloat => "rg11b10ufloat-renderable",
        }
    }

    /// Get all WebGPU feature variants.
    ///
    /// # Returns
    ///
    /// An array of all feature variants.
    pub const fn all() -> [WebGPUFeature; 7] {
        [
            WebGPUFeature::TimestampQuery,
            WebGPUFeature::IndirectFirstInstance,
            WebGPUFeature::Depth32FloatStencil8,
            WebGPUFeature::Bgra8UnormStorage,
            WebGPUFeature::PipelineStatistics,
            WebGPUFeature::ShaderFloat16,
            WebGPUFeature::RG11B10UFloat,
        ]
    }

    /// Map to wgpu Features flag.
    ///
    /// # Returns
    ///
    /// The corresponding wgpu::Features flag.
    pub const fn to_wgpu_feature(&self) -> Features {
        match self {
            WebGPUFeature::TimestampQuery => Features::TIMESTAMP_QUERY,
            WebGPUFeature::IndirectFirstInstance => Features::INDIRECT_FIRST_INSTANCE,
            WebGPUFeature::Depth32FloatStencil8 => Features::DEPTH32FLOAT_STENCIL8,
            WebGPUFeature::Bgra8UnormStorage => Features::BGRA8UNORM_STORAGE,
            WebGPUFeature::PipelineStatistics => Features::PIPELINE_STATISTICS_QUERY,
            WebGPUFeature::ShaderFloat16 => Features::SHADER_F16,
            WebGPUFeature::RG11B10UFloat => Features::RG11B10UFLOAT_RENDERABLE,
        }
    }
}

impl std::fmt::Display for WebGPUFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// WebGPULimitations (T-WGPU-P7.2.4 requirement)
// ============================================================================

/// WebGPU browser-specific limitations.
///
/// This struct captures known limitations and missing features for
/// WebGPU implementations across different browsers. Use this for
/// feature detection and fallback code paths.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::webgpu::{WebGPULimitations, Browser};
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let limitations = WebGPULimitations::from_adapter(&adapter);
/// if limitations.no_timestamp_queries {
///     println!("Timestamp queries not available, using CPU fallback");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct WebGPULimitations {
    /// Detected browser.
    pub browser: Browser,

    /// Device limits.
    pub limits: WebGPULimits,

    /// No compute-to-graphics barriers.
    ///
    /// Some browsers don't support explicit barriers between
    /// compute and graphics passes.
    pub no_compute_to_graphics_barrier: bool,

    /// No timestamp queries.
    ///
    /// Timestamp queries may be disabled for privacy reasons.
    pub no_timestamp_queries: bool,

    /// No pipeline statistics queries.
    ///
    /// Pipeline statistics often disabled for fingerprinting prevention.
    pub no_pipeline_statistics: bool,

    /// No indirect first instance.
    ///
    /// The first_instance field in indirect draws may not be supported.
    pub no_indirect_first_instance: bool,

    /// No Depth32Float + Stencil8 format.
    ///
    /// Combined depth-stencil format not available.
    pub no_depth32float_stencil8: bool,

    /// No BGRA8Unorm storage.
    ///
    /// Cannot use BGRA8Unorm in storage textures.
    pub no_bgra8unorm_storage: bool,
}

impl WebGPULimitations {
    /// Detect limitations from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query
    ///
    /// # Returns
    ///
    /// The detected limitations.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        let limits = WebGPULimits::from_adapter(adapter);

        // Detect browser from adapter info if available
        // In practice, browser detection needs JS interop on WASM
        let browser = Browser::Unknown;

        Self {
            browser,
            limits,
            no_compute_to_graphics_barrier: false, // wgpu handles this
            no_timestamp_queries: !features.contains(Features::TIMESTAMP_QUERY),
            no_pipeline_statistics: !features.contains(Features::PIPELINE_STATISTICS_QUERY),
            no_indirect_first_instance: !features.contains(Features::INDIRECT_FIRST_INSTANCE),
            no_depth32float_stencil8: !features.contains(Features::DEPTH32FLOAT_STENCIL8),
            no_bgra8unorm_storage: !features.contains(Features::BGRA8UNORM_STORAGE),
        }
    }

    /// Create limitations for a specific browser.
    ///
    /// # Arguments
    ///
    /// * `browser` - The target browser
    ///
    /// # Returns
    ///
    /// Expected limitations for that browser.
    pub fn for_browser(browser: Browser) -> Self {
        let limits = WebGPULimits::for_browser(browser);

        match browser {
            Browser::Chrome | Browser::Edge => Self {
                browser,
                limits,
                no_compute_to_graphics_barrier: false,
                no_timestamp_queries: false,
                no_pipeline_statistics: true, // Usually disabled
                no_indirect_first_instance: false,
                no_depth32float_stencil8: false,
                no_bgra8unorm_storage: false,
            },
            Browser::Firefox => Self {
                browser,
                limits,
                no_compute_to_graphics_barrier: false,
                no_timestamp_queries: true, // Not in stable
                no_pipeline_statistics: true,
                no_indirect_first_instance: false,
                no_depth32float_stencil8: false,
                no_bgra8unorm_storage: true,
            },
            Browser::Safari => Self {
                browser,
                limits,
                no_compute_to_graphics_barrier: false,
                no_timestamp_queries: true, // Privacy restricted
                no_pipeline_statistics: true,
                no_indirect_first_instance: true,
                no_depth32float_stencil8: false,
                no_bgra8unorm_storage: true,
            },
            Browser::Unknown => Self {
                browser,
                limits,
                no_compute_to_graphics_barrier: false,
                no_timestamp_queries: true,
                no_pipeline_statistics: true,
                no_indirect_first_instance: true,
                no_depth32float_stencil8: true,
                no_bgra8unorm_storage: true,
            },
        }
    }

    /// Check if a specific feature is supported.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check
    ///
    /// # Returns
    ///
    /// `true` if the feature is supported.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::{WebGPULimitations, WebGPUFeature, Browser};
    ///
    /// let limitations = WebGPULimitations::for_browser(Browser::Chrome);
    /// assert!(limitations.is_feature_supported(WebGPUFeature::TimestampQuery));
    /// assert!(!limitations.is_feature_supported(WebGPUFeature::PipelineStatistics));
    /// ```
    pub fn is_feature_supported(&self, feature: WebGPUFeature) -> bool {
        match feature {
            WebGPUFeature::TimestampQuery => !self.no_timestamp_queries,
            WebGPUFeature::IndirectFirstInstance => !self.no_indirect_first_instance,
            WebGPUFeature::Depth32FloatStencil8 => !self.no_depth32float_stencil8,
            WebGPUFeature::Bgra8UnormStorage => !self.no_bgra8unorm_storage,
            WebGPUFeature::PipelineStatistics => !self.no_pipeline_statistics,
            WebGPUFeature::ShaderFloat16 => true, // Usually supported if requested
            WebGPUFeature::RG11B10UFloat => true, // Usually supported
        }
    }

    /// Get list of missing/unsupported features.
    ///
    /// # Returns
    ///
    /// A vector of unsupported features.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::{WebGPULimitations, WebGPUFeature, Browser};
    ///
    /// let limitations = WebGPULimitations::for_browser(Browser::Safari);
    /// let missing = limitations.missing_features();
    /// assert!(missing.contains(&WebGPUFeature::TimestampQuery));
    /// assert!(missing.contains(&WebGPUFeature::IndirectFirstInstance));
    /// ```
    pub fn missing_features(&self) -> Vec<WebGPUFeature> {
        let mut missing = Vec::new();

        for feature in WebGPUFeature::all() {
            if !self.is_feature_supported(feature) {
                missing.push(feature);
            }
        }

        missing
    }

    /// Get supported features as a set.
    ///
    /// # Returns
    ///
    /// A set of supported features.
    pub fn supported_features(&self) -> HashSet<WebGPUFeature> {
        WebGPUFeature::all()
            .into_iter()
            .filter(|f| self.is_feature_supported(*f))
            .collect()
    }

    /// Create a human-readable summary.
    ///
    /// # Returns
    ///
    /// A summary string of limitations.
    pub fn summary(&self) -> String {
        let missing = self.missing_features();
        if missing.is_empty() {
            format!("{}: Full WebGPU support", self.browser)
        } else {
            let names: Vec<_> = missing.iter().map(|f| f.name()).collect();
            format!("{}: Missing {}", self.browser, names.join(", "))
        }
    }
}

impl Default for WebGPULimitations {
    fn default() -> Self {
        Self::for_browser(Browser::Unknown)
    }
}

// ============================================================================
// BrowserCompatibility (T-WGPU-P7.2.4 requirement)
// ============================================================================

/// Feature support information for a specific browser.
///
/// Tracks whether a feature is supported, when it was added,
/// and any relevant notes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSupport {
    /// Whether the feature is supported.
    pub supported: bool,

    /// Browser version when the feature was added.
    pub since_version: Option<String>,

    /// Additional notes about support.
    pub notes: Option<String>,
}

impl FeatureSupport {
    /// Create a supported feature entry.
    pub fn supported(version: impl Into<String>) -> Self {
        Self {
            supported: true,
            since_version: Some(version.into()),
            notes: None,
        }
    }

    /// Create an unsupported feature entry.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            since_version: None,
            notes: Some(reason.into()),
        }
    }

    /// Create a partial support entry.
    pub fn partial(version: impl Into<String>, note: impl Into<String>) -> Self {
        Self {
            supported: true,
            since_version: Some(version.into()),
            notes: Some(note.into()),
        }
    }
}

impl Default for FeatureSupport {
    fn default() -> Self {
        Self {
            supported: false,
            since_version: None,
            notes: None,
        }
    }
}

/// Browser compatibility table for WebGPU features.
///
/// Provides cross-browser feature support information for
/// planning graceful degradation.
///
/// # Example
///
/// ```
/// use renderer_backend::backend::webgpu::{BrowserCompatibility, WebGPUFeature};
///
/// let compat = BrowserCompatibility::for_feature(WebGPUFeature::TimestampQuery);
/// assert!(compat.chrome.supported);
/// assert!(!compat.safari.supported);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BrowserCompatibility {
    /// Chrome support.
    pub chrome: FeatureSupport,

    /// Firefox support.
    pub firefox: FeatureSupport,

    /// Safari support.
    pub safari: FeatureSupport,
}

impl BrowserCompatibility {
    /// Get compatibility information for a feature.
    ///
    /// # Arguments
    ///
    /// * `feature` - The WebGPU feature to query
    ///
    /// # Returns
    ///
    /// Browser compatibility table for the feature.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::webgpu::{BrowserCompatibility, WebGPUFeature};
    ///
    /// let compat = BrowserCompatibility::for_feature(WebGPUFeature::IndirectFirstInstance);
    /// println!("Chrome: {} (since {:?})", compat.chrome.supported, compat.chrome.since_version);
    /// ```
    pub fn for_feature(feature: WebGPUFeature) -> Self {
        match feature {
            WebGPUFeature::TimestampQuery => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::partial("Nightly", "Requires flag"),
                safari: FeatureSupport::unsupported("Privacy restriction"),
            },
            WebGPUFeature::IndirectFirstInstance => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::supported("Nightly"),
                safari: FeatureSupport::unsupported("Not implemented"),
            },
            WebGPUFeature::Depth32FloatStencil8 => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::supported("Nightly"),
                safari: FeatureSupport::supported("17"),
            },
            WebGPUFeature::Bgra8UnormStorage => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::unsupported("Not implemented"),
                safari: FeatureSupport::unsupported("Not implemented"),
            },
            WebGPUFeature::PipelineStatistics => Self {
                chrome: FeatureSupport::unsupported("Privacy restriction"),
                firefox: FeatureSupport::unsupported("Privacy restriction"),
                safari: FeatureSupport::unsupported("Privacy restriction"),
            },
            WebGPUFeature::ShaderFloat16 => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::supported("Nightly"),
                safari: FeatureSupport::supported("17"),
            },
            WebGPUFeature::RG11B10UFloat => Self {
                chrome: FeatureSupport::supported("113"),
                firefox: FeatureSupport::supported("Nightly"),
                safari: FeatureSupport::supported("17"),
            },
        }
    }

    /// Check if any browser supports the feature.
    ///
    /// # Returns
    ///
    /// `true` if at least one browser supports the feature.
    pub fn any_supported(&self) -> bool {
        self.chrome.supported || self.firefox.supported || self.safari.supported
    }

    /// Check if all browsers support the feature.
    ///
    /// # Returns
    ///
    /// `true` if all browsers support the feature.
    pub fn all_supported(&self) -> bool {
        self.chrome.supported && self.firefox.supported && self.safari.supported
    }

    /// Get support status for a specific browser.
    ///
    /// # Arguments
    ///
    /// * `browser` - The browser to query
    ///
    /// # Returns
    ///
    /// The feature support information. Returns a clone to avoid
    /// lifetime issues with unknown browsers.
    pub fn for_browser(&self, browser: Browser) -> FeatureSupport {
        match browser {
            Browser::Chrome | Browser::Edge => self.chrome.clone(),
            Browser::Firefox => self.firefox.clone(),
            Browser::Safari => self.safari.clone(),
            Browser::Unknown => FeatureSupport::default(),
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // WebGpuTier Tests
    // ========================================================================

    #[test]
    fn test_tier_default() {
        let tier = WebGpuTier::default();
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn test_tier_ordering() {
        assert!(WebGpuTier::Tier1 < WebGpuTier::Tier2);
        assert!(WebGpuTier::Tier2 < WebGpuTier::Tier3);
    }

    #[test]
    fn test_tier_supports_compute_shaders() {
        assert!(WebGpuTier::Tier1.supports_compute_shaders());
        assert!(WebGpuTier::Tier2.supports_compute_shaders());
        assert!(WebGpuTier::Tier3.supports_compute_shaders());
    }

    #[test]
    fn test_tier_supports_storage_textures() {
        assert!(!WebGpuTier::Tier1.supports_storage_textures());
        assert!(WebGpuTier::Tier2.supports_storage_textures());
        assert!(WebGpuTier::Tier3.supports_storage_textures());
    }

    #[test]
    fn test_tier_supports_timestamp_query() {
        assert!(!WebGpuTier::Tier1.supports_timestamp_query());
        assert!(!WebGpuTier::Tier2.supports_timestamp_query());
        assert!(WebGpuTier::Tier3.supports_timestamp_query());
    }

    #[test]
    fn test_tier_name() {
        assert_eq!(WebGpuTier::Tier1.name(), "Tier 1 (Basic)");
        assert_eq!(WebGpuTier::Tier2.name(), "Tier 2 (Extended)");
        assert_eq!(WebGpuTier::Tier3.name(), "Tier 3 (Advanced)");
    }

    #[test]
    fn test_tier_number() {
        assert_eq!(WebGpuTier::Tier1.tier_number(), 1);
        assert_eq!(WebGpuTier::Tier2.tier_number(), 2);
        assert_eq!(WebGpuTier::Tier3.tier_number(), 3);
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(format!("{}", WebGpuTier::Tier1), "Tier 1 (Basic)");
        assert_eq!(format!("{}", WebGpuTier::Tier3), "Tier 3 (Advanced)");
    }

    #[test]
    fn test_tier_from_basic_limits() {
        let limits = Limits::downlevel_defaults();
        let features = Features::empty();
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        // Downlevel defaults have low limits
        assert_eq!(tier, WebGpuTier::Tier1);
    }

    #[test]
    fn test_tier_from_high_limits_with_timestamp() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let tier = WebGpuTier::from_limits_and_features(&limits, features);
        assert_eq!(tier, WebGpuTier::Tier3);
    }

    // ========================================================================
    // WebGpuLimits Tests
    // ========================================================================

    #[test]
    fn test_limits_default() {
        let limits = WebGpuLimits::default();
        assert_eq!(limits.max_texture_dimension_2d, 2048);
        assert_eq!(limits.max_bind_groups, 4);
    }

    #[test]
    fn test_limits_tier1_minimum() {
        let limits = WebGpuLimits::tier1_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 2048);
        assert_eq!(limits.max_bind_groups, 4);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
    }

    #[test]
    fn test_limits_tier2_minimum() {
        let limits = WebGpuLimits::tier2_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 4096);
        assert_eq!(limits.max_bind_groups, 8);
        assert_eq!(limits.max_storage_textures_per_shader_stage, 8);
    }

    #[test]
    fn test_limits_tier3_minimum() {
        let limits = WebGpuLimits::tier3_minimum();
        assert_eq!(limits.max_texture_dimension_2d, 8192);
        assert_eq!(limits.max_texture_dimension_3d, 2048);
        assert_eq!(limits.max_compute_workgroup_size_x, 1024);
    }

    #[test]
    fn test_limits_meets_tier() {
        let tier1 = WebGpuLimits::tier1_minimum();
        let tier2 = WebGpuLimits::tier2_minimum();
        let tier3 = WebGpuLimits::tier3_minimum();

        // Tier 1 meets Tier 1 only
        assert!(tier1.meets_tier(WebGpuTier::Tier1));
        assert!(!tier1.meets_tier(WebGpuTier::Tier2));
        assert!(!tier1.meets_tier(WebGpuTier::Tier3));

        // Tier 2 meets Tier 1 and 2
        assert!(tier2.meets_tier(WebGpuTier::Tier1));
        assert!(tier2.meets_tier(WebGpuTier::Tier2));
        assert!(!tier2.meets_tier(WebGpuTier::Tier3));

        // Tier 3 meets all tiers
        assert!(tier3.meets_tier(WebGpuTier::Tier1));
        assert!(tier3.meets_tier(WebGpuTier::Tier2));
        assert!(tier3.meets_tier(WebGpuTier::Tier3));
    }

    #[test]
    fn test_limits_from_wgpu_limits() {
        let wgpu_limits = Limits::default();
        let webgpu_limits = WebGpuLimits::from_wgpu_limits(&wgpu_limits);
        assert_eq!(webgpu_limits.max_texture_dimension_2d, wgpu_limits.max_texture_dimension_2d);
        assert_eq!(webgpu_limits.max_bind_groups, wgpu_limits.max_bind_groups);
    }

    // ========================================================================
    // WebGpuFeatures Tests
    // ========================================================================

    #[test]
    fn test_features_default() {
        let features = WebGpuFeatures::default();
        assert_eq!(features.tier, WebGpuTier::Tier1);
        assert!(!features.texture_compression_bc);
        assert!(!features.texture_compression_etc2);
        assert!(!features.texture_compression_astc);
        assert!(!features.timestamp_query);
    }

    #[test]
    fn test_features_supports_compression() {
        let mut features = WebGpuFeatures::default();
        assert!(!features.supports_compression());

        features.texture_compression_bc = true;
        assert!(features.supports_compression());

        features.texture_compression_bc = false;
        features.texture_compression_etc2 = true;
        assert!(features.supports_compression());

        features.texture_compression_etc2 = false;
        features.texture_compression_astc = true;
        assert!(features.supports_compression());
    }

    #[test]
    fn test_features_compression_formats() {
        let mut features = WebGpuFeatures::default();
        assert!(features.compression_formats().is_empty());

        features.texture_compression_bc = true;
        features.texture_compression_astc = true;
        let formats = features.compression_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&"BC"));
        assert!(formats.contains(&"ASTC"));
        assert!(!formats.contains(&"ETC2"));
    }

    #[test]
    fn test_features_is_mobile_optimized() {
        let mut features = WebGpuFeatures::default();
        assert!(!features.is_mobile_optimized());

        // ETC2 only = mobile
        features.texture_compression_etc2 = true;
        assert!(features.is_mobile_optimized());

        // BC added = not mobile
        features.texture_compression_bc = true;
        assert!(!features.is_mobile_optimized());
    }

    #[test]
    fn test_features_is_desktop_optimized() {
        let mut features = WebGpuFeatures::default();
        assert!(!features.is_desktop_optimized());

        // BC only = desktop
        features.texture_compression_bc = true;
        assert!(features.is_desktop_optimized());

        // ETC2 added = not desktop only
        features.texture_compression_etc2 = true;
        assert!(!features.is_desktop_optimized());
    }

    #[test]
    fn test_features_summary_basic() {
        let features = WebGpuFeatures::default();
        let summary = features.summary();
        assert!(summary.contains("Tier 1"));
        assert!(summary.contains("Basic"));
    }

    #[test]
    fn test_features_summary_full() {
        let mut features = WebGpuFeatures::default();
        features.tier = WebGpuTier::Tier3;
        features.texture_compression_bc = true;
        features.timestamp_query = true;
        features.shader_f16 = true;

        let summary = features.summary();
        assert!(summary.contains("Tier 3"));
        assert!(summary.contains("BC"));
        assert!(summary.contains("Timestamp"));
        assert!(summary.contains("FP16"));
    }

    #[test]
    fn test_features_from_empty_wgpu_features() {
        let limits = Limits::downlevel_defaults();
        let features = Features::empty();
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(!webgpu.texture_compression_bc);
        assert!(!webgpu.timestamp_query);
    }

    #[test]
    fn test_features_from_compression_wgpu_features() {
        let limits = Limits::default();
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ETC2;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.texture_compression_bc);
        assert!(webgpu.texture_compression_etc2);
        assert!(!webgpu.texture_compression_astc);
    }

    #[test]
    fn test_features_from_timestamp_wgpu_features() {
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 8192;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;
        let features = Features::TIMESTAMP_QUERY;
        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);
        assert!(webgpu.timestamp_query);
        assert_eq!(webgpu.tier, WebGpuTier::Tier3);
    }

    // ========================================================================
    // BrowserType Tests
    // ========================================================================

    #[test]
    fn test_browser_type_default() {
        let browser = BrowserType::default();
        assert_eq!(browser, BrowserType::Unknown);
    }

    #[test]
    fn test_browser_type_from_chrome_user_agent() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        );
        assert_eq!(browser, BrowserType::Chrome);
    }

    #[test]
    fn test_browser_type_from_edge_user_agent() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        );
        assert_eq!(browser, BrowserType::Edge);
    }

    #[test]
    fn test_browser_type_from_firefox_user_agent() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        );
        assert_eq!(browser, BrowserType::Firefox);
    }

    #[test]
    fn test_browser_type_from_safari_user_agent() {
        let browser = BrowserType::from_user_agent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        );
        assert_eq!(browser, BrowserType::Safari);
    }

    #[test]
    fn test_browser_type_has_stable_webgpu() {
        assert!(BrowserType::Chrome.has_stable_webgpu());
        assert!(BrowserType::Edge.has_stable_webgpu());
        assert!(BrowserType::Safari.has_stable_webgpu());
        assert!(!BrowserType::Firefox.has_stable_webgpu());
        assert!(!BrowserType::Unknown.has_stable_webgpu());
    }

    #[test]
    fn test_browser_type_supports_offscreen_canvas() {
        assert!(BrowserType::Chrome.supports_offscreen_canvas());
        assert!(BrowserType::Edge.supports_offscreen_canvas());
        assert!(BrowserType::Firefox.supports_offscreen_canvas());
        assert!(!BrowserType::Safari.supports_offscreen_canvas());
        assert!(!BrowserType::Unknown.supports_offscreen_canvas());
    }

    #[test]
    fn test_browser_type_supports_shared_array_buffer() {
        assert!(BrowserType::Chrome.supports_shared_array_buffer());
        assert!(BrowserType::Edge.supports_shared_array_buffer());
        assert!(BrowserType::Firefox.supports_shared_array_buffer());
        assert!(BrowserType::Safari.supports_shared_array_buffer());
        assert!(!BrowserType::Unknown.supports_shared_array_buffer());
    }

    #[test]
    fn test_browser_type_name() {
        assert_eq!(BrowserType::Chrome.name(), "Chrome");
        assert_eq!(BrowserType::Firefox.name(), "Firefox");
        assert_eq!(BrowserType::Safari.name(), "Safari");
        assert_eq!(BrowserType::Edge.name(), "Edge");
        assert_eq!(BrowserType::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_browser_type_webgpu_backend() {
        assert_eq!(BrowserType::Chrome.webgpu_backend(), "Dawn (Vulkan/D3D12/Metal)");
        assert_eq!(BrowserType::Safari.webgpu_backend(), "WebKit (Metal)");
        assert_eq!(BrowserType::Firefox.webgpu_backend(), "wgpu-native (Vulkan/D3D12/Metal)");
    }

    #[test]
    fn test_browser_type_display() {
        assert_eq!(format!("{}", BrowserType::Chrome), "Chrome");
        assert_eq!(format!("{}", BrowserType::Safari), "Safari");
    }

    // ========================================================================
    // BrowserCapabilities Tests
    // ========================================================================

    #[test]
    fn test_browser_capabilities_default() {
        let caps = BrowserCapabilities::default();
        assert_eq!(caps.browser, BrowserType::Unknown);
        assert!(!caps.supports_offscreen_canvas);
        assert!(!caps.supports_shared_array_buffer);
    }

    #[test]
    fn test_browser_capabilities_from_chrome_user_agent() {
        let caps = BrowserCapabilities::from_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        );
        assert_eq!(caps.browser, BrowserType::Chrome);
        assert!(caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (32767, 32767));
    }

    #[test]
    fn test_browser_capabilities_from_safari_user_agent() {
        let caps = BrowserCapabilities::from_user_agent(
            "Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15"
        );
        assert_eq!(caps.browser, BrowserType::Safari);
        assert!(!caps.supports_offscreen_canvas);
        assert!(caps.supports_shared_array_buffer);
        assert_eq!(caps.max_canvas_size, (16384, 16384));
    }

    #[test]
    fn test_browser_capabilities_has_full_webgpu() {
        let chrome = BrowserCapabilities::from_user_agent("Chrome/120.0.0.0");
        assert!(chrome.has_full_webgpu());

        let firefox = BrowserCapabilities::from_user_agent("Firefox/121.0");
        assert!(!firefox.has_full_webgpu());
    }

    #[test]
    fn test_browser_capabilities_summary() {
        let caps = BrowserCapabilities::from_user_agent("Chrome/120.0.0.0");
        let summary = caps.summary();
        assert!(summary.contains("Chrome"));
        assert!(summary.contains("OffscreenCanvas"));
        assert!(summary.contains("SAB"));
        assert!(summary.contains("32767x32767"));
    }

    #[test]
    fn test_browser_capabilities_detect() {
        // On non-WASM, detect() returns defaults
        let caps = BrowserCapabilities::detect();
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert_eq!(caps.browser, BrowserType::Unknown);
            assert_eq!(caps.max_canvas_size, (16384, 16384));
        }
        // On WASM this would return actual browser info
        let _ = caps;
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_tier_limits_consistency() {
        // Tier 3 limits should meet all tier requirements
        let tier3_limits = WebGpuLimits::tier3_minimum();
        assert!(tier3_limits.meets_tier(WebGpuTier::Tier1));
        assert!(tier3_limits.meets_tier(WebGpuTier::Tier2));
        assert!(tier3_limits.meets_tier(WebGpuTier::Tier3));

        // Tier 1 limits should not meet Tier 2 requirements
        let tier1_limits = WebGpuLimits::tier1_minimum();
        assert!(tier1_limits.meets_tier(WebGpuTier::Tier1));
        assert!(!tier1_limits.meets_tier(WebGpuTier::Tier2));
    }

    #[test]
    fn test_mobile_vs_desktop_detection() {
        // Mobile device simulation
        let mut mobile = WebGpuFeatures::default();
        mobile.texture_compression_etc2 = true;
        mobile.texture_compression_astc = true;
        assert!(mobile.is_mobile_optimized());
        assert!(!mobile.is_desktop_optimized());

        // Desktop device simulation
        let mut desktop = WebGpuFeatures::default();
        desktop.texture_compression_bc = true;
        assert!(desktop.is_desktop_optimized());
        assert!(!desktop.is_mobile_optimized());

        // Cross-platform device
        let mut cross = WebGpuFeatures::default();
        cross.texture_compression_bc = true;
        cross.texture_compression_etc2 = true;
        cross.texture_compression_astc = true;
        assert!(!cross.is_mobile_optimized());
        assert!(!cross.is_desktop_optimized());
    }

    #[test]
    fn test_full_feature_detection() {
        // Simulate a high-end device
        let mut limits = Limits::default();
        limits.max_texture_dimension_2d = 16384;
        limits.max_compute_workgroups_per_dimension = 65535;
        limits.max_storage_buffers_per_shader_stage = 8;

        let features = Features::TEXTURE_COMPRESSION_BC
            | Features::TIMESTAMP_QUERY
            | Features::SHADER_F16
            | Features::DEPTH32FLOAT_STENCIL8
            | Features::FLOAT32_FILTERABLE;

        let webgpu = WebGpuFeatures::from_adapter_info(&limits, features);

        assert_eq!(webgpu.tier, WebGpuTier::Tier3);
        assert!(webgpu.texture_compression_bc);
        assert!(webgpu.timestamp_query);
        assert!(webgpu.shader_f16);
        assert!(webgpu.depth32_float_stencil8);
        assert!(webgpu.float32_filterable);
        assert!(webgpu.is_desktop_optimized());
    }

    #[test]
    fn test_browser_edge_case_user_agents() {
        // Empty user agent
        let empty = BrowserType::from_user_agent("");
        assert_eq!(empty, BrowserType::Unknown);

        // Partial user agent
        let partial = BrowserType::from_user_agent("Mozilla/5.0");
        assert_eq!(partial, BrowserType::Unknown);

        // Case insensitive
        let lower = BrowserType::from_user_agent("mozilla firefox");
        assert_eq!(lower, BrowserType::Unknown); // No version, so unknown

        let with_version = BrowserType::from_user_agent("Firefox/100");
        assert_eq!(with_version, BrowserType::Firefox);
    }

    // ========================================================================
    // Browser Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_browser_variants() {
        // Test all browser variants exist and have distinct values
        let browsers = [
            Browser::Chrome,
            Browser::Firefox,
            Browser::Safari,
            Browser::Edge,
            Browser::Unknown,
        ];

        for (i, a) in browsers.iter().enumerate() {
            for (j, b) in browsers.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_browser_timestamp_query() {
        // Chrome and Edge support timestamp queries
        assert!(Browser::Chrome.supports_timestamp_query());
        assert!(Browser::Edge.supports_timestamp_query());

        // Firefox and Safari do not (in stable)
        assert!(!Browser::Firefox.supports_timestamp_query());
        assert!(!Browser::Safari.supports_timestamp_query());
        assert!(!Browser::Unknown.supports_timestamp_query());
    }

    #[test]
    fn test_browser_texture_limits() {
        // Chrome and Edge have highest limits
        assert_eq!(Browser::Chrome.max_texture_dimension(), 16384);
        assert_eq!(Browser::Edge.max_texture_dimension(), 16384);

        // Firefox also supports high limits
        assert_eq!(Browser::Firefox.max_texture_dimension(), 16384);

        // Safari has lower limits (iOS influenced)
        assert_eq!(Browser::Safari.max_texture_dimension(), 8192);

        // Unknown uses conservative defaults
        assert_eq!(Browser::Unknown.max_texture_dimension(), 4096);
    }

    // ========================================================================
    // WebGPULimits Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_webgpu_limits_spec_minimum() {
        let limits = WebGPULimits::SPEC_MINIMUM;

        // WebGPU spec minimum values
        assert_eq!(limits.max_texture_dimension_1d, 8192);
        assert_eq!(limits.max_texture_dimension_2d, 8192);
        assert_eq!(limits.max_texture_dimension_3d, 2048);
        assert_eq!(limits.max_texture_array_layers, 256);
        assert_eq!(limits.max_bind_groups, 4);
        assert_eq!(limits.max_bindings_per_bind_group, 1000);
        assert_eq!(limits.max_dynamic_uniform_buffers, 8);
        assert_eq!(limits.max_dynamic_storage_buffers, 4);
        assert_eq!(limits.max_uniform_buffer_binding_size, 65536);
        assert_eq!(limits.max_storage_buffer_binding_size, 134217728);
        assert_eq!(limits.max_buffer_size, 268435456);
        assert_eq!(limits.max_vertex_buffers, 8);
        assert_eq!(limits.max_vertex_attributes, 16);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
        assert_eq!(limits.max_compute_workgroup_size_z, 64);
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroups_per_dimension, 65535);
    }

    #[test]
    fn test_webgpu_limits_chrome() {
        let limits = WebGPULimits::CHROME_DEFAULTS;

        // Chrome has higher limits than spec minimum
        assert!(limits.max_texture_dimension_2d >= WebGPULimits::SPEC_MINIMUM.max_texture_dimension_2d);
        assert!(limits.max_bind_groups >= WebGPULimits::SPEC_MINIMUM.max_bind_groups);
        assert!(limits.max_buffer_size >= WebGPULimits::SPEC_MINIMUM.max_buffer_size);

        // Chrome specific values
        assert_eq!(limits.max_texture_dimension_2d, 16384);
        assert_eq!(limits.max_bind_groups, 8);
        assert_eq!(limits.max_buffer_size, 2147483648); // 2 GB
    }

    #[test]
    fn test_webgpu_limits_firefox() {
        let limits = WebGPULimits::FIREFOX_DEFAULTS;

        // Firefox has similar limits to Chrome
        assert_eq!(limits.max_texture_dimension_2d, 16384);
        assert_eq!(limits.max_bind_groups, 8);
        assert_eq!(limits.max_buffer_size, 2147483648); // 2 GB

        // But some differences exist
        assert_eq!(limits.max_vertex_attributes, 32);
    }

    #[test]
    fn test_webgpu_limits_safari() {
        let limits = WebGPULimits::SAFARI_DEFAULTS;

        // Safari has more conservative limits
        assert_eq!(limits.max_texture_dimension_2d, 8192);
        assert_eq!(limits.max_bind_groups, 4);
        assert_eq!(limits.max_buffer_size, 1073741824); // 1 GB

        // Still meets spec minimum
        assert!(limits.max_texture_dimension_2d >= WebGPULimits::SPEC_MINIMUM.max_texture_dimension_2d);
    }

    // ========================================================================
    // WebGPULimitations Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_webgpu_limitations_default() {
        let limitations = WebGPULimitations::default();

        // Unknown browser has most conservative settings
        assert_eq!(limitations.browser, Browser::Unknown);
        assert!(limitations.no_timestamp_queries);
        assert!(limitations.no_pipeline_statistics);
        assert!(limitations.no_indirect_first_instance);
        assert!(limitations.no_depth32float_stencil8);
        assert!(limitations.no_bgra8unorm_storage);
    }

    #[test]
    fn test_webgpu_limitations_feature_check() {
        // Chrome has most features
        let chrome = WebGPULimitations::for_browser(Browser::Chrome);
        assert!(chrome.is_feature_supported(WebGPUFeature::TimestampQuery));
        assert!(chrome.is_feature_supported(WebGPUFeature::IndirectFirstInstance));
        assert!(chrome.is_feature_supported(WebGPUFeature::Depth32FloatStencil8));
        assert!(chrome.is_feature_supported(WebGPUFeature::Bgra8UnormStorage));
        assert!(!chrome.is_feature_supported(WebGPUFeature::PipelineStatistics));

        // Safari has fewer features
        let safari = WebGPULimitations::for_browser(Browser::Safari);
        assert!(!safari.is_feature_supported(WebGPUFeature::TimestampQuery));
        assert!(!safari.is_feature_supported(WebGPUFeature::IndirectFirstInstance));
        assert!(safari.is_feature_supported(WebGPUFeature::Depth32FloatStencil8));
    }

    #[test]
    fn test_webgpu_limitations_missing_features() {
        let safari = WebGPULimitations::for_browser(Browser::Safari);
        let missing = safari.missing_features();

        // Safari should be missing several features
        assert!(missing.contains(&WebGPUFeature::TimestampQuery));
        assert!(missing.contains(&WebGPUFeature::IndirectFirstInstance));
        assert!(missing.contains(&WebGPUFeature::PipelineStatistics));
        assert!(missing.contains(&WebGPUFeature::Bgra8UnormStorage));

        // But should have some features
        assert!(!missing.contains(&WebGPUFeature::Depth32FloatStencil8));
        assert!(!missing.contains(&WebGPUFeature::ShaderFloat16));
    }

    // ========================================================================
    // WebGPUFeature Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_webgpu_feature_variants() {
        let all_features = WebGPUFeature::all();
        assert_eq!(all_features.len(), 7);

        // Check all variants are included
        assert!(all_features.contains(&WebGPUFeature::TimestampQuery));
        assert!(all_features.contains(&WebGPUFeature::IndirectFirstInstance));
        assert!(all_features.contains(&WebGPUFeature::Depth32FloatStencil8));
        assert!(all_features.contains(&WebGPUFeature::Bgra8UnormStorage));
        assert!(all_features.contains(&WebGPUFeature::PipelineStatistics));
        assert!(all_features.contains(&WebGPUFeature::ShaderFloat16));
        assert!(all_features.contains(&WebGPUFeature::RG11B10UFloat));
    }

    // ========================================================================
    // BrowserCompatibility Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_browser_compatibility_timestamp() {
        let compat = BrowserCompatibility::for_feature(WebGPUFeature::TimestampQuery);

        // Chrome supports
        assert!(compat.chrome.supported);
        assert_eq!(compat.chrome.since_version.as_deref(), Some("113"));

        // Firefox partial
        assert!(compat.firefox.supported);
        assert!(compat.firefox.notes.is_some());

        // Safari does not support
        assert!(!compat.safari.supported);
        assert!(compat.safari.notes.as_ref().unwrap().contains("Privacy"));
    }

    #[test]
    fn test_browser_compatibility_indirect() {
        let compat = BrowserCompatibility::for_feature(WebGPUFeature::IndirectFirstInstance);

        // Chrome and Firefox support
        assert!(compat.chrome.supported);
        assert!(compat.firefox.supported);

        // Safari does not
        assert!(!compat.safari.supported);
    }

    // ========================================================================
    // Limits Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_limits_buffer_size() {
        // Test buffer size limits across browsers
        assert_eq!(WebGPULimits::SPEC_MINIMUM.max_buffer_size, 268435456); // 256 MB
        assert_eq!(WebGPULimits::CHROME_DEFAULTS.max_buffer_size, 2147483648); // 2 GB
        assert_eq!(WebGPULimits::FIREFOX_DEFAULTS.max_buffer_size, 2147483648); // 2 GB
        assert_eq!(WebGPULimits::SAFARI_DEFAULTS.max_buffer_size, 1073741824); // 1 GB
    }

    #[test]
    fn test_limits_texture_size() {
        // Test texture dimension limits
        assert_eq!(WebGPULimits::SPEC_MINIMUM.max_texture_dimension_2d, 8192);
        assert_eq!(WebGPULimits::CHROME_DEFAULTS.max_texture_dimension_2d, 16384);
        assert_eq!(WebGPULimits::SAFARI_DEFAULTS.max_texture_dimension_2d, 8192);

        // 3D textures
        assert_eq!(WebGPULimits::SPEC_MINIMUM.max_texture_dimension_3d, 2048);
    }

    #[test]
    fn test_limits_compute_workgroup() {
        let limits = WebGPULimits::SPEC_MINIMUM;

        // Workgroup dimensions
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
        assert_eq!(limits.max_compute_workgroup_size_z, 64);

        // Total invocations and dispatch dimensions
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroups_per_dimension, 65535);
    }

    #[test]
    fn test_limits_bind_groups() {
        // Bind group limits
        assert_eq!(WebGPULimits::SPEC_MINIMUM.max_bind_groups, 4);
        assert_eq!(WebGPULimits::CHROME_DEFAULTS.max_bind_groups, 8);
        assert_eq!(WebGPULimits::SAFARI_DEFAULTS.max_bind_groups, 4);

        // Bindings per group
        assert_eq!(WebGPULimits::SPEC_MINIMUM.max_bindings_per_bind_group, 1000);
    }

    #[test]
    fn test_feature_support_struct() {
        // Test FeatureSupport construction
        let supported = FeatureSupport::supported("113");
        assert!(supported.supported);
        assert_eq!(supported.since_version.as_deref(), Some("113"));
        assert!(supported.notes.is_none());

        let unsupported = FeatureSupport::unsupported("Privacy restriction");
        assert!(!unsupported.supported);
        assert!(unsupported.since_version.is_none());
        assert_eq!(unsupported.notes.as_deref(), Some("Privacy restriction"));

        let partial = FeatureSupport::partial("Nightly", "Requires flag");
        assert!(partial.supported);
        assert_eq!(partial.since_version.as_deref(), Some("Nightly"));
        assert_eq!(partial.notes.as_deref(), Some("Requires flag"));
    }

    // ========================================================================
    // Additional Integration Tests (T-WGPU-P7.2.4)
    // ========================================================================

    #[test]
    fn test_browser_for_browser_lookup() {
        // Test WebGPULimits::for_browser
        let chrome_limits = WebGPULimits::for_browser(Browser::Chrome);
        let edge_limits = WebGPULimits::for_browser(Browser::Edge);
        assert_eq!(chrome_limits, edge_limits); // Both use CHROME_DEFAULTS

        let firefox_limits = WebGPULimits::for_browser(Browser::Firefox);
        assert_eq!(firefox_limits, WebGPULimits::FIREFOX_DEFAULTS);

        let safari_limits = WebGPULimits::for_browser(Browser::Safari);
        assert_eq!(safari_limits, WebGPULimits::SAFARI_DEFAULTS);

        let unknown_limits = WebGPULimits::for_browser(Browser::Unknown);
        assert_eq!(unknown_limits, WebGPULimits::SPEC_MINIMUM);
    }

    #[test]
    fn test_webgpu_limitations_summary() {
        let chrome = WebGPULimitations::for_browser(Browser::Chrome);
        let summary = chrome.summary();
        assert!(summary.contains("Chrome"));

        let safari = WebGPULimitations::for_browser(Browser::Safari);
        let summary = safari.summary();
        assert!(summary.contains("Safari"));
        assert!(summary.contains("Missing"));
        assert!(summary.contains("timestamp"));
    }

    #[test]
    fn test_browser_from_user_agent_new() {
        // Test Browser::from_user_agent (distinct from BrowserType)
        let chrome = Browser::from_user_agent("Chrome/120.0.0.0");
        assert_eq!(chrome, Browser::Chrome);

        let edge = Browser::from_user_agent("Edg/120.0.0.0");
        assert_eq!(edge, Browser::Edge);

        let firefox = Browser::from_user_agent("Firefox/121.0");
        assert_eq!(firefox, Browser::Firefox);

        let safari = Browser::from_user_agent("Version/17.0 Safari/605.1.15");
        assert_eq!(safari, Browser::Safari);
    }

    #[test]
    fn test_browser_compatibility_any_all() {
        // Timestamp: Chrome yes, Firefox partial, Safari no
        let timestamp = BrowserCompatibility::for_feature(WebGPUFeature::TimestampQuery);
        assert!(timestamp.any_supported());
        assert!(!timestamp.all_supported());

        // Depth32FloatStencil8: all support
        let depth = BrowserCompatibility::for_feature(WebGPUFeature::Depth32FloatStencil8);
        assert!(depth.any_supported());
        assert!(depth.all_supported());

        // PipelineStatistics: none support
        let stats = BrowserCompatibility::for_feature(WebGPUFeature::PipelineStatistics);
        assert!(!stats.any_supported());
        assert!(!stats.all_supported());
    }

    #[test]
    fn test_webgpu_feature_to_wgpu() {
        // Verify mapping to wgpu Features
        assert_eq!(WebGPUFeature::TimestampQuery.to_wgpu_feature(), Features::TIMESTAMP_QUERY);
        assert_eq!(WebGPUFeature::IndirectFirstInstance.to_wgpu_feature(), Features::INDIRECT_FIRST_INSTANCE);
        assert_eq!(WebGPUFeature::Depth32FloatStencil8.to_wgpu_feature(), Features::DEPTH32FLOAT_STENCIL8);
        assert_eq!(WebGPUFeature::Bgra8UnormStorage.to_wgpu_feature(), Features::BGRA8UNORM_STORAGE);
        assert_eq!(WebGPUFeature::PipelineStatistics.to_wgpu_feature(), Features::PIPELINE_STATISTICS_QUERY);
        assert_eq!(WebGPUFeature::ShaderFloat16.to_wgpu_feature(), Features::SHADER_F16);
        assert_eq!(WebGPUFeature::RG11B10UFloat.to_wgpu_feature(), Features::RG11B10UFLOAT_RENDERABLE);
    }

    #[test]
    fn test_webgpu_limitations_supported_features() {
        let chrome = WebGPULimitations::for_browser(Browser::Chrome);
        let supported = chrome.supported_features();

        // Chrome supports most features except pipeline stats
        assert!(supported.contains(&WebGPUFeature::TimestampQuery));
        assert!(supported.contains(&WebGPUFeature::IndirectFirstInstance));
        assert!(!supported.contains(&WebGPUFeature::PipelineStatistics));
    }
}
