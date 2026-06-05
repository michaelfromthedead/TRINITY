//! Resource declaration system for the TRINITY frame graph.
//!
//! This module provides descriptors for declaring GPU resources (textures and buffers)
//! within the frame graph, along with a registry for managing resource declarations
//! and an allocator trait for actual GPU resource creation.
//!
//! # Overview
//!
//! Resources in the frame graph are described by lightweight descriptors that capture
//! the essential properties needed for allocation and barrier scheduling. The actual
//! GPU resources (wgpu::Texture, wgpu::Buffer) are created lazily by an allocator
//! implementation when the frame graph is compiled or executed.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::frame_graph::resources::*;
//!
//! let mut registry = ResourceRegistry::new();
//!
//! // Declare a render target
//! let color_id = registry.declare_texture(
//!     "main_color",
//!     TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
//! );
//!
//! // Declare a depth buffer
//! let depth_id = registry.declare_texture(
//!     "main_depth",
//!     TextureDescriptor::new_depth(1920, 1080),
//! );
//!
//! // Declare a uniform buffer
//! let uniforms_id = registry.declare_buffer(
//!     "camera_uniforms",
//!     BufferDescriptor::new_uniform(256),
//! );
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ResourceId
// ---------------------------------------------------------------------------

/// Opaque identifier for a resource within the frame graph.
///
/// ResourceIds are assigned by the [`ResourceRegistry`] and remain stable
/// for the lifetime of the registry. They are used to reference resources
/// in pass declarations and for tracking dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(pub u32);

impl ResourceId {
    /// The null / sentinel ID indicating "no resource."
    pub const NONE: Self = Self(u32::MAX);

    /// Creates a new ResourceId from a raw value.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    #[inline]
    pub const fn raw(&self) -> u32 {
        self.0
    }

    /// Returns true if this is the null/none sentinel.
    #[inline]
    pub const fn is_none(&self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns true if this is a valid (non-null) ID.
    #[inline]
    pub const fn is_some(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::NONE {
            write!(f, "ResourceId::NONE")
        } else {
            write!(f, "ResourceId({})", self.0)
        }
    }
}

impl Default for ResourceId {
    fn default() -> Self {
        Self::NONE
    }
}

// ---------------------------------------------------------------------------
// TextureDescriptor
// ---------------------------------------------------------------------------

/// Describes the properties of a texture resource.
///
/// This is a logical descriptor that captures the essential properties needed
/// for texture allocation. It does not hold an actual wgpu::Texture.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureDescriptor {
    /// Width of the texture in texels.
    pub width: u32,
    /// Height of the texture in texels.
    pub height: u32,
    /// Depth (for 3D textures) or array layer count (for 2D arrays).
    pub depth_or_layers: u32,
    /// Number of mip levels (1 = no mipmaps).
    pub mip_levels: u32,
    /// Sample count for MSAA (1 = no multisampling).
    pub sample_count: u32,
    /// Pixel format.
    pub format: wgpu::TextureFormat,
    /// Allowed usage flags.
    pub usage: wgpu::TextureUsages,
    /// Texture dimensionality.
    pub dimension: wgpu::TextureDimension,
    /// Optional debug label.
    pub label: Option<String>,
}

impl TextureDescriptor {
    /// Creates a new 2D texture descriptor.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in texels.
    /// * `height` - Height in texels.
    /// * `format` - Pixel format.
    ///
    /// # Returns
    ///
    /// A descriptor with default usage (TEXTURE_BINDING | COPY_DST), 1 mip level,
    /// 1 sample, and no label.
    pub fn new_2d(width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            dimension: wgpu::TextureDimension::D2,
            label: None,
        }
    }

    /// Creates a new render target texture descriptor.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in texels.
    /// * `height` - Height in texels.
    /// * `format` - Pixel format.
    ///
    /// # Returns
    ///
    /// A descriptor with RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC usage.
    pub fn new_render_target(width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            dimension: wgpu::TextureDimension::D2,
            label: None,
        }
    }

    /// Creates a new depth buffer texture descriptor.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in texels.
    /// * `height` - Height in texels.
    ///
    /// # Returns
    ///
    /// A descriptor with Depth32Float format and RENDER_ATTACHMENT | TEXTURE_BINDING usage.
    pub fn new_depth(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            dimension: wgpu::TextureDimension::D2,
            label: None,
        }
    }

    /// Builder method to set the number of mip levels.
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of mip levels.
    ///
    /// # Returns
    ///
    /// Self with updated mip_levels.
    pub fn with_mips(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    /// Builder method to enable MSAA.
    ///
    /// # Arguments
    ///
    /// * `samples` - Sample count (1, 2, 4, 8, or 16).
    ///
    /// # Returns
    ///
    /// Self with updated sample_count.
    pub fn with_msaa(mut self, samples: u32) -> Self {
        self.sample_count = samples;
        self
    }

    /// Builder method to set a debug label.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label string.
    ///
    /// # Returns
    ///
    /// Self with updated label.
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Calculates the approximate size in bytes.
    ///
    /// This is an estimate based on the base dimensions, format, mip levels,
    /// and sample count. The actual GPU allocation may differ.
    ///
    /// # Returns
    ///
    /// Estimated size in bytes.
    pub fn size_bytes(&self) -> u64 {
        let bytes_per_texel = format_bytes_per_texel(self.format);
        let base_size =
            self.width as u64 * self.height as u64 * self.depth_or_layers as u64 * bytes_per_texel;

        // Account for mip chain (sum of geometric series: 1 + 1/4 + 1/16 + ...)
        let mip_factor = if self.mip_levels > 1 {
            // Approximate: ~1.33 for full mip chain
            let mut total = 0u64;
            let mut w = self.width;
            let mut h = self.height;
            for _ in 0..self.mip_levels {
                total += w.max(1) as u64 * h.max(1) as u64;
                w /= 2;
                h /= 2;
            }
            total * self.depth_or_layers as u64 * bytes_per_texel
        } else {
            base_size
        };

        // Account for MSAA
        let msaa_size = mip_factor * self.sample_count as u64;

        msaa_size
    }
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self::new_2d(1, 1, wgpu::TextureFormat::Rgba8Unorm)
    }
}

impl fmt::Display for TextureDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Texture({}x{}x{}, {:?}, mips={}, samples={})",
            self.width,
            self.height,
            self.depth_or_layers,
            self.format,
            self.mip_levels,
            self.sample_count
        )
    }
}

/// Returns the bytes per texel for common texture formats.
///
/// For compressed formats, returns the bytes per block divided by block size.
fn format_bytes_per_texel(format: wgpu::TextureFormat) -> u64 {
    use wgpu::TextureFormat::*;
    match format {
        // 8-bit formats
        R8Unorm | R8Snorm | R8Uint | R8Sint => 1,
        // 16-bit formats
        R16Uint | R16Sint | R16Float | Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint => 2,
        // 32-bit formats
        R32Uint | R32Sint | R32Float | Rg16Uint | Rg16Sint | Rg16Float | Rgba8Unorm
        | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8Unorm | Bgra8UnormSrgb
        | Rgb10a2Uint | Rgb10a2Unorm | Rg11b10Float | Depth32Float | Depth24Plus
        | Depth24PlusStencil8 => 4,
        // 64-bit formats
        Rg32Uint | Rg32Sint | Rg32Float | Rgba16Uint | Rgba16Sint | Rgba16Float => 8,
        // 128-bit formats
        Rgba32Uint | Rgba32Sint | Rgba32Float => 16,
        // Depth/stencil
        Depth32FloatStencil8 => 5,
        Stencil8 => 1,
        Depth16Unorm => 2,
        // Compressed formats (approximate - bytes per block / pixels per block)
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb => 1, // 8 bytes / 16 pixels
        Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb => 1, // 16 bytes / 16 pixels
        Bc4RUnorm | Bc4RSnorm => 1, // 8 bytes / 16 pixels
        Bc5RgUnorm | Bc5RgSnorm => 1, // 16 bytes / 16 pixels
        Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => 1,
        // ETC2 / EAC
        Etc2Rgb8Unorm | Etc2Rgb8UnormSrgb | Etc2Rgb8A1Unorm | Etc2Rgb8A1UnormSrgb
        | EacR11Unorm | EacR11Snorm => 1,
        Etc2Rgba8Unorm | Etc2Rgba8UnormSrgb | EacRg11Unorm | EacRg11Snorm => 1,
        // ASTC (4x4)
        Astc { .. } => 1,
        // NV12 (planar YUV)
        NV12 => 2,
        // Default fallback
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// BufferDescriptor
// ---------------------------------------------------------------------------

/// Describes the properties of a buffer resource.
///
/// This is a logical descriptor that captures the essential properties needed
/// for buffer allocation. It does not hold an actual wgpu::Buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDescriptor {
    /// Size of the buffer in bytes.
    pub size: u64,
    /// Allowed usage flags.
    pub usage: wgpu::BufferUsages,
    /// Whether the buffer should be mapped at creation for CPU writes.
    pub mapped_at_creation: bool,
    /// Optional debug label.
    pub label: Option<String>,
}

impl BufferDescriptor {
    /// Creates a new buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    /// * `usage` - Buffer usage flags.
    ///
    /// # Returns
    ///
    /// A descriptor with the given size and usage, not mapped at creation.
    pub fn new(size: u64, usage: wgpu::BufferUsages) -> Self {
        Self {
            size,
            usage,
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Creates a new vertex buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with VERTEX | COPY_DST usage.
    pub fn new_vertex(size: u64) -> Self {
        Self::new(size, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST)
    }

    /// Creates a new index buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with INDEX | COPY_DST usage.
    pub fn new_index(size: u64) -> Self {
        Self::new(size, wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST)
    }

    /// Creates a new uniform buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with UNIFORM | COPY_DST usage.
    pub fn new_uniform(size: u64) -> Self {
        Self::new(
            size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// Creates a new storage buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with STORAGE | COPY_DST | COPY_SRC usage.
    pub fn new_storage(size: u64) -> Self {
        Self::new(
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    /// Creates a new staging buffer descriptor for GPU-to-CPU reads.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with MAP_READ | COPY_DST usage.
    pub fn new_staging_read(size: u64) -> Self {
        Self::new(
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// Creates a new staging buffer descriptor for CPU-to-GPU writes.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    ///
    /// # Returns
    ///
    /// A descriptor with MAP_WRITE | COPY_SRC usage, mapped at creation.
    pub fn new_staging_write(size: u64) -> Self {
        Self {
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
            label: None,
        }
    }

    /// Builder method to set a debug label.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label string.
    ///
    /// # Returns
    ///
    /// Self with updated label.
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for BufferDescriptor {
    fn default() -> Self {
        Self::new(256, wgpu::BufferUsages::COPY_DST)
    }
}

impl fmt::Display for BufferDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer({} bytes, {:?})", self.size, self.usage)
    }
}

// ---------------------------------------------------------------------------
// ResourceDescriptor
// ---------------------------------------------------------------------------

/// A unified descriptor for either a texture or buffer resource.
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceDescriptor {
    /// A texture resource.
    Texture(TextureDescriptor),
    /// A buffer resource.
    Buffer(BufferDescriptor),
}

impl ResourceDescriptor {
    /// Returns true if this is a texture descriptor.
    #[inline]
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture(_))
    }

    /// Returns true if this is a buffer descriptor.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }

    /// Returns the approximate size in bytes.
    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Texture(desc) => desc.size_bytes(),
            Self::Buffer(desc) => desc.size,
        }
    }

    /// Returns the debug label, if set.
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Texture(desc) => desc.label.as_deref(),
            Self::Buffer(desc) => desc.label.as_deref(),
        }
    }

    /// Returns a reference to the texture descriptor, if this is a texture.
    pub fn as_texture(&self) -> Option<&TextureDescriptor> {
        match self {
            Self::Texture(desc) => Some(desc),
            Self::Buffer(_) => None,
        }
    }

    /// Returns a reference to the buffer descriptor, if this is a buffer.
    pub fn as_buffer(&self) -> Option<&BufferDescriptor> {
        match self {
            Self::Texture(_) => None,
            Self::Buffer(desc) => Some(desc),
        }
    }
}

impl fmt::Display for ResourceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Texture(desc) => write!(f, "{}", desc),
            Self::Buffer(desc) => write!(f, "{}", desc),
        }
    }
}

impl From<TextureDescriptor> for ResourceDescriptor {
    fn from(desc: TextureDescriptor) -> Self {
        Self::Texture(desc)
    }
}

impl From<BufferDescriptor> for ResourceDescriptor {
    fn from(desc: BufferDescriptor) -> Self {
        Self::Buffer(desc)
    }
}

// ---------------------------------------------------------------------------
// ResourceHandle (distinct from ResourceId for versioned tracking)
// ---------------------------------------------------------------------------

/// A versioned handle to a resource in the registry.
///
/// ResourceHandle combines a ResourceId with a generation counter for
/// safe resource tracking across frame graph recompilations.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceHandle {
    /// The resource identifier.
    pub id: ResourceId,
    /// The resource descriptor.
    pub descriptor: ResourceDescriptor,
    /// Generation counter for versioning.
    pub generation: u64,
}

impl ResourceHandle {
    /// Creates a new resource handle.
    ///
    /// # Arguments
    ///
    /// * `id` - The resource identifier.
    /// * `descriptor` - The resource descriptor.
    ///
    /// # Returns
    ///
    /// A handle with generation 0.
    pub fn new(id: ResourceId, descriptor: ResourceDescriptor) -> Self {
        Self {
            id,
            descriptor,
            generation: 0,
        }
    }

    /// Creates a new resource handle with a specific generation.
    ///
    /// # Arguments
    ///
    /// * `id` - The resource identifier.
    /// * `descriptor` - The resource descriptor.
    /// * `generation` - The generation counter.
    pub fn with_generation(id: ResourceId, descriptor: ResourceDescriptor, generation: u64) -> Self {
        Self {
            id,
            descriptor,
            generation,
        }
    }

    /// Returns true if this is a texture resource.
    #[inline]
    pub fn is_texture(&self) -> bool {
        self.descriptor.is_texture()
    }

    /// Returns true if this is a buffer resource.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        self.descriptor.is_buffer()
    }
}

impl fmt::Display for ResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@gen{}: {}", self.id, self.generation, self.descriptor)
    }
}

// ---------------------------------------------------------------------------
// TextureView
// ---------------------------------------------------------------------------

/// A logical view into a texture resource.
///
/// TextureView describes how to create a wgpu::TextureView from a texture
/// resource, specifying mip levels, array layers, and aspect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureView {
    /// The texture resource this view references.
    pub resource: ResourceId,
    /// Which aspect(s) of the texture to view (all, depth-only, stencil-only).
    pub aspect: wgpu::TextureAspect,
    /// Base mip level for the view.
    pub base_mip: u32,
    /// Number of mip levels visible (None = all remaining).
    pub mip_count: Option<u32>,
    /// Base array layer for the view.
    pub base_layer: u32,
    /// Number of array layers visible (None = all remaining).
    pub layer_count: Option<u32>,
}

impl TextureView {
    /// Creates a new texture view covering the entire texture.
    ///
    /// # Arguments
    ///
    /// * `resource` - The texture resource ID.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            aspect: wgpu::TextureAspect::All,
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
        }
    }

    /// Builder method to set the mip range.
    ///
    /// # Arguments
    ///
    /// * `base` - Base mip level.
    /// * `count` - Number of mip levels (None = all remaining).
    pub fn with_mip_range(mut self, base: u32, count: Option<u32>) -> Self {
        self.base_mip = base;
        self.mip_count = count;
        self
    }

    /// Builder method to set the array layer range.
    ///
    /// # Arguments
    ///
    /// * `base` - Base array layer.
    /// * `count` - Number of layers (None = all remaining).
    pub fn with_layer_range(mut self, base: u32, count: Option<u32>) -> Self {
        self.base_layer = base;
        self.layer_count = count;
        self
    }

    /// Creates a depth-only view of a depth/stencil texture.
    ///
    /// # Arguments
    ///
    /// * `resource` - The texture resource ID.
    pub fn depth_only(resource: ResourceId) -> Self {
        Self {
            resource,
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
        }
    }

    /// Creates a stencil-only view of a depth/stencil texture.
    ///
    /// # Arguments
    ///
    /// * `resource` - The texture resource ID.
    pub fn stencil_only(resource: ResourceId) -> Self {
        Self {
            resource,
            aspect: wgpu::TextureAspect::StencilOnly,
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
        }
    }
}

impl fmt::Display for TextureView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TextureView({}, {:?}, mips={}..{:?}, layers={}..{:?})",
            self.resource,
            self.aspect,
            self.base_mip,
            self.mip_count,
            self.base_layer,
            self.layer_count
        )
    }
}

// ---------------------------------------------------------------------------
// BufferSlice
// ---------------------------------------------------------------------------

/// A logical slice of a buffer resource.
///
/// BufferSlice describes a range within a buffer for binding or copying.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSlice {
    /// The buffer resource this slice references.
    pub resource: ResourceId,
    /// Byte offset from the start of the buffer.
    pub offset: u64,
    /// Size in bytes (None = to the end of the buffer).
    pub size: Option<u64>,
}

impl BufferSlice {
    /// Creates a new buffer slice covering the entire buffer.
    ///
    /// # Arguments
    ///
    /// * `resource` - The buffer resource ID.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            offset: 0,
            size: None,
        }
    }

    /// Builder method to set the byte range.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from start.
    /// * `size` - Size in bytes (None = to end).
    pub fn with_range(mut self, offset: u64, size: Option<u64>) -> Self {
        self.offset = offset;
        self.size = size;
        self
    }
}

impl fmt::Display for BufferSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.size {
            Some(size) => write!(f, "BufferSlice({}, {}..{})", self.resource, self.offset, self.offset + size),
            None => write!(f, "BufferSlice({}, {}..end)", self.resource, self.offset),
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceRegistry
// ---------------------------------------------------------------------------

/// A registry for managing resource declarations.
///
/// The registry assigns unique ResourceIds to declared resources and maintains
/// a name-based lookup for convenient access. Resources are versioned with a
/// generation counter that increments on each mutation.
#[derive(Clone, Debug, Default)]
pub struct ResourceRegistry {
    /// Map from resource ID to handle.
    resources: HashMap<ResourceId, ResourceHandle>,
    /// Map from name to resource ID.
    by_name: HashMap<String, ResourceId>,
    /// Global generation counter.
    generation: u64,
    /// Next available resource ID.
    next_id: u32,
}

impl ResourceRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            by_name: HashMap::new(),
            generation: 0,
            next_id: 0,
        }
    }

    /// Declares a new texture resource.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for the resource.
    /// * `descriptor` - Texture descriptor.
    ///
    /// # Returns
    ///
    /// The assigned ResourceId.
    ///
    /// # Panics
    ///
    /// Panics if a resource with the given name already exists.
    pub fn declare_texture<S: Into<String>>(
        &mut self,
        name: S,
        descriptor: TextureDescriptor,
    ) -> ResourceId {
        let name = name.into();
        assert!(
            !self.by_name.contains_key(&name),
            "Resource with name '{}' already exists",
            name
        );

        let id = ResourceId(self.next_id);
        self.next_id += 1;

        let handle = ResourceHandle::with_generation(
            id,
            ResourceDescriptor::Texture(descriptor),
            self.generation,
        );

        self.resources.insert(id, handle);
        self.by_name.insert(name, id);

        id
    }

    /// Declares a new buffer resource.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for the resource.
    /// * `descriptor` - Buffer descriptor.
    ///
    /// # Returns
    ///
    /// The assigned ResourceId.
    ///
    /// # Panics
    ///
    /// Panics if a resource with the given name already exists.
    pub fn declare_buffer<S: Into<String>>(
        &mut self,
        name: S,
        descriptor: BufferDescriptor,
    ) -> ResourceId {
        let name = name.into();
        assert!(
            !self.by_name.contains_key(&name),
            "Resource with name '{}' already exists",
            name
        );

        let id = ResourceId(self.next_id);
        self.next_id += 1;

        let handle = ResourceHandle::with_generation(
            id,
            ResourceDescriptor::Buffer(descriptor),
            self.generation,
        );

        self.resources.insert(id, handle);
        self.by_name.insert(name, id);

        id
    }

    /// Gets a resource handle by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The resource ID.
    ///
    /// # Returns
    ///
    /// The resource handle, or None if not found.
    pub fn get(&self, id: ResourceId) -> Option<&ResourceHandle> {
        self.resources.get(&id)
    }

    /// Gets a resource ID by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The resource name.
    ///
    /// # Returns
    ///
    /// The resource ID, or None if not found.
    pub fn get_by_name(&self, name: &str) -> Option<ResourceId> {
        self.by_name.get(name).copied()
    }

    /// Removes a resource from the registry.
    ///
    /// # Arguments
    ///
    /// * `id` - The resource ID to remove.
    ///
    /// # Returns
    ///
    /// The removed handle, or None if not found.
    pub fn remove(&mut self, id: ResourceId) -> Option<ResourceHandle> {
        if let Some(handle) = self.resources.remove(&id) {
            // Find and remove the name mapping
            self.by_name.retain(|_, &mut v| v != id);
            self.generation += 1;
            Some(handle)
        } else {
            None
        }
    }

    /// Clears all resources from the registry.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.by_name.clear();
        self.generation += 1;
        // Don't reset next_id to ensure IDs are never reused
    }

    /// Returns an iterator over all resources.
    pub fn iter(&self) -> impl Iterator<Item = (&ResourceId, &ResourceHandle)> {
        self.resources.iter()
    }

    /// Returns the number of resources in the registry.
    pub fn count(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Returns the current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the total estimated size of all resources in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.resources.values().map(|h| h.descriptor.size_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// ResourceAllocator trait
// ---------------------------------------------------------------------------

/// Trait for allocating and deallocating GPU resources.
///
/// Implementations of this trait handle the actual wgpu resource creation
/// based on descriptors from the frame graph.
pub trait ResourceAllocator {
    /// Allocates a GPU texture from a descriptor.
    ///
    /// # Arguments
    ///
    /// * `desc` - The texture descriptor.
    ///
    /// # Returns
    ///
    /// The allocated wgpu::Texture.
    fn allocate_texture(&self, desc: &TextureDescriptor) -> wgpu::Texture;

    /// Allocates a GPU buffer from a descriptor.
    ///
    /// # Arguments
    ///
    /// * `desc` - The buffer descriptor.
    ///
    /// # Returns
    ///
    /// The allocated wgpu::Buffer.
    fn allocate_buffer(&self, desc: &BufferDescriptor) -> wgpu::Buffer;

    /// Deallocates a GPU texture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to deallocate.
    ///
    /// # Note
    ///
    /// In wgpu, textures are reference-counted and automatically deallocated
    /// when dropped. This method is provided for explicit cleanup or pooling.
    fn deallocate_texture(&self, texture: wgpu::Texture);

    /// Deallocates a GPU buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to deallocate.
    ///
    /// # Note
    ///
    /// In wgpu, buffers are reference-counted and automatically deallocated
    /// when dropped. This method is provided for explicit cleanup or pooling.
    fn deallocate_buffer(&self, buffer: wgpu::Buffer);
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ResourceId tests --

    #[test]
    fn test_resource_id_none() {
        assert!(ResourceId::NONE.is_none());
        assert!(!ResourceId::NONE.is_some());
        assert_eq!(ResourceId::NONE.raw(), u32::MAX);
    }

    #[test]
    fn test_resource_id_valid() {
        let id = ResourceId::new(42);
        assert!(!id.is_none());
        assert!(id.is_some());
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_resource_id_display() {
        assert_eq!(format!("{}", ResourceId::NONE), "ResourceId::NONE");
        assert_eq!(format!("{}", ResourceId::new(5)), "ResourceId(5)");
    }

    #[test]
    fn test_resource_id_equality() {
        let id1 = ResourceId::new(1);
        let id2 = ResourceId::new(1);
        let id3 = ResourceId::new(2);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // -- TextureDescriptor tests --

    #[test]
    fn test_texture_descriptor_new_2d() {
        let desc = TextureDescriptor::new_2d(1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(desc.width, 1920);
        assert_eq!(desc.height, 1080);
        assert_eq!(desc.depth_or_layers, 1);
        assert_eq!(desc.mip_levels, 1);
        assert_eq!(desc.sample_count, 1);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_texture_descriptor_render_target() {
        let desc = TextureDescriptor::new_render_target(800, 600, wgpu::TextureFormat::Bgra8Unorm);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(desc.usage.contains(wgpu::TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_texture_descriptor_depth() {
        let desc = TextureDescriptor::new_depth(1024, 768);
        assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn test_texture_descriptor_builder_methods() {
        let desc = TextureDescriptor::new_2d(512, 512, wgpu::TextureFormat::Rgba8Unorm)
            .with_mips(5)
            .with_msaa(4)
            .with_label("test_texture");
        assert_eq!(desc.mip_levels, 5);
        assert_eq!(desc.sample_count, 4);
        assert_eq!(desc.label, Some("test_texture".to_string()));
    }

    #[test]
    fn test_texture_descriptor_size_bytes() {
        let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
        // 256 * 256 * 4 bytes per texel = 262144
        assert_eq!(desc.size_bytes(), 262144);
    }

    #[test]
    fn test_texture_descriptor_size_bytes_msaa() {
        let desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm)
            .with_msaa(4);
        // 256 * 256 * 4 * 4 samples = 1048576
        assert_eq!(desc.size_bytes(), 1048576);
    }

    // -- BufferDescriptor tests --

    #[test]
    fn test_buffer_descriptor_new() {
        let desc = BufferDescriptor::new(1024, wgpu::BufferUsages::COPY_DST);
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.usage, wgpu::BufferUsages::COPY_DST);
        assert!(!desc.mapped_at_creation);
    }

    #[test]
    fn test_buffer_descriptor_vertex() {
        let desc = BufferDescriptor::new_vertex(4096);
        assert!(desc.usage.contains(wgpu::BufferUsages::VERTEX));
        assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
    }

    #[test]
    fn test_buffer_descriptor_index() {
        let desc = BufferDescriptor::new_index(2048);
        assert!(desc.usage.contains(wgpu::BufferUsages::INDEX));
    }

    #[test]
    fn test_buffer_descriptor_uniform() {
        let desc = BufferDescriptor::new_uniform(256);
        assert!(desc.usage.contains(wgpu::BufferUsages::UNIFORM));
    }

    #[test]
    fn test_buffer_descriptor_storage() {
        let desc = BufferDescriptor::new_storage(65536);
        assert!(desc.usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_buffer_descriptor_staging_read() {
        let desc = BufferDescriptor::new_staging_read(1024);
        assert!(desc.usage.contains(wgpu::BufferUsages::MAP_READ));
        assert!(desc.usage.contains(wgpu::BufferUsages::COPY_DST));
    }

    #[test]
    fn test_buffer_descriptor_staging_write() {
        let desc = BufferDescriptor::new_staging_write(1024);
        assert!(desc.usage.contains(wgpu::BufferUsages::MAP_WRITE));
        assert!(desc.usage.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(desc.mapped_at_creation);
    }

    #[test]
    fn test_buffer_descriptor_with_label() {
        let desc = BufferDescriptor::new_uniform(128).with_label("camera_uniforms");
        assert_eq!(desc.label, Some("camera_uniforms".to_string()));
    }

    // -- ResourceDescriptor tests --

    #[test]
    fn test_resource_descriptor_texture() {
        let tex_desc = TextureDescriptor::new_2d(256, 256, wgpu::TextureFormat::Rgba8Unorm);
        let desc: ResourceDescriptor = tex_desc.clone().into();
        assert!(desc.is_texture());
        assert!(!desc.is_buffer());
        assert_eq!(desc.as_texture(), Some(&tex_desc));
        assert_eq!(desc.as_buffer(), None);
    }

    #[test]
    fn test_resource_descriptor_buffer() {
        let buf_desc = BufferDescriptor::new_uniform(256);
        let desc: ResourceDescriptor = buf_desc.clone().into();
        assert!(!desc.is_texture());
        assert!(desc.is_buffer());
        assert_eq!(desc.as_buffer(), Some(&buf_desc));
        assert_eq!(desc.as_texture(), None);
    }

    #[test]
    fn test_resource_descriptor_size_bytes() {
        let tex_desc = TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm);
        let tex_res: ResourceDescriptor = tex_desc.into();
        assert_eq!(tex_res.size_bytes(), 40000);

        let buf_desc = BufferDescriptor::new(512, wgpu::BufferUsages::UNIFORM);
        let buf_res: ResourceDescriptor = buf_desc.into();
        assert_eq!(buf_res.size_bytes(), 512);
    }

    #[test]
    fn test_resource_descriptor_label() {
        let tex_desc = TextureDescriptor::new_2d(64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .with_label("my_texture");
        let desc: ResourceDescriptor = tex_desc.into();
        assert_eq!(desc.label(), Some("my_texture"));

        let buf_desc = BufferDescriptor::new_uniform(64);
        let desc2: ResourceDescriptor = buf_desc.into();
        assert_eq!(desc2.label(), None);
    }

    // -- ResourceHandle tests --

    #[test]
    fn test_resource_handle_new() {
        let id = ResourceId::new(0);
        let desc = ResourceDescriptor::Texture(TextureDescriptor::new_depth(800, 600));
        let handle = ResourceHandle::new(id, desc);
        assert_eq!(handle.id, id);
        assert_eq!(handle.generation, 0);
        assert!(handle.is_texture());
    }

    #[test]
    fn test_resource_handle_with_generation() {
        let id = ResourceId::new(5);
        let desc = ResourceDescriptor::Buffer(BufferDescriptor::new_uniform(128));
        let handle = ResourceHandle::with_generation(id, desc, 42);
        assert_eq!(handle.generation, 42);
        assert!(handle.is_buffer());
    }

    // -- TextureView tests --

    #[test]
    fn test_texture_view_new() {
        let id = ResourceId::new(1);
        let view = TextureView::new(id);
        assert_eq!(view.resource, id);
        assert_eq!(view.aspect, wgpu::TextureAspect::All);
        assert_eq!(view.base_mip, 0);
        assert_eq!(view.mip_count, None);
    }

    #[test]
    fn test_texture_view_with_mip_range() {
        let view = TextureView::new(ResourceId::new(0))
            .with_mip_range(2, Some(3));
        assert_eq!(view.base_mip, 2);
        assert_eq!(view.mip_count, Some(3));
    }

    #[test]
    fn test_texture_view_depth_only() {
        let view = TextureView::depth_only(ResourceId::new(0));
        assert_eq!(view.aspect, wgpu::TextureAspect::DepthOnly);
    }

    #[test]
    fn test_texture_view_stencil_only() {
        let view = TextureView::stencil_only(ResourceId::new(0));
        assert_eq!(view.aspect, wgpu::TextureAspect::StencilOnly);
    }

    // -- BufferSlice tests --

    #[test]
    fn test_buffer_slice_new() {
        let id = ResourceId::new(3);
        let slice = BufferSlice::new(id);
        assert_eq!(slice.resource, id);
        assert_eq!(slice.offset, 0);
        assert_eq!(slice.size, None);
    }

    #[test]
    fn test_buffer_slice_with_range() {
        let slice = BufferSlice::new(ResourceId::new(0))
            .with_range(64, Some(256));
        assert_eq!(slice.offset, 64);
        assert_eq!(slice.size, Some(256));
    }

    // -- ResourceRegistry tests --

    #[test]
    fn test_registry_new() {
        let registry = ResourceRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_declare_texture() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_texture(
            "color_buffer",
            TextureDescriptor::new_render_target(1920, 1080, wgpu::TextureFormat::Rgba8Unorm),
        );
        assert_eq!(registry.count(), 1);
        assert!(registry.get(id).is_some());
        assert_eq!(registry.get_by_name("color_buffer"), Some(id));
    }

    #[test]
    fn test_registry_declare_buffer() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer(
            "uniform_buffer",
            BufferDescriptor::new_uniform(256),
        );
        assert!(registry.get(id).unwrap().is_buffer());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_registry_duplicate_name_panics() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("dup", TextureDescriptor::default());
        registry.declare_texture("dup", TextureDescriptor::default());
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = ResourceRegistry::new();
        let id = registry.declare_buffer("temp", BufferDescriptor::default());
        let gen_before = registry.generation();
        let removed = registry.remove(id);
        assert!(removed.is_some());
        assert!(registry.get(id).is_none());
        assert!(registry.get_by_name("temp").is_none());
        assert!(registry.generation() > gen_before);
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("a", TextureDescriptor::default());
        registry.declare_buffer("b", BufferDescriptor::default());
        assert_eq!(registry.count(), 2);
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_iter() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture("t1", TextureDescriptor::default());
        registry.declare_buffer("b1", BufferDescriptor::default());
        let items: Vec<_> = registry.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_registry_total_size() {
        let mut registry = ResourceRegistry::new();
        registry.declare_texture(
            "tex",
            TextureDescriptor::new_2d(100, 100, wgpu::TextureFormat::Rgba8Unorm),
        );
        registry.declare_buffer("buf", BufferDescriptor::new(1024, wgpu::BufferUsages::UNIFORM));
        // 100*100*4 + 1024 = 41024
        assert_eq!(registry.total_size_bytes(), 41024);
    }

    #[test]
    fn test_registry_unique_ids() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        let id2 = registry.declare_texture("t2", TextureDescriptor::default());
        let id3 = registry.declare_buffer("b1", BufferDescriptor::default());
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_registry_ids_never_reused_after_clear() {
        let mut registry = ResourceRegistry::new();
        let id1 = registry.declare_texture("t1", TextureDescriptor::default());
        registry.clear();
        let id2 = registry.declare_texture("t2", TextureDescriptor::default());
        assert_ne!(id1, id2);
        assert!(id2.raw() > id1.raw());
    }
}
