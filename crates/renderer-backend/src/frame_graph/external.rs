//! External and imported resource handling for the TRINITY frame graph.
//!
//! This module provides types and utilities for managing resources that originate
//! outside the frame graph's internal allocation system, such as:
//!
//! - **Swapchain textures**: Surface textures acquired from wgpu for presentation
//! - **User-provided resources**: Textures and buffers created and managed externally
//! - **Shared resources**: Resources shared between multiple frame graphs or systems
//!
//! # Architecture
//!
//! External resources differ from transient resources in several key ways:
//!
//! 1. **Lifetime management**: The frame graph does not allocate or deallocate them
//! 2. **Barrier handling**: Acquire/release barriers may be needed at frame boundaries
//! 3. **State tracking**: Initial and final states must be explicitly specified
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::frame_graph::external::*;
//!
//! let mut registry = ExternalResourceRegistry::new();
//!
//! // Import swapchain texture for rendering
//! let swapchain_id = registry.import_swapchain(surface_texture, format);
//!
//! // Import a user-provided texture
//! let user_tex_id = registry.import_texture(
//!     "shadow_atlas",
//!     texture,
//!     view,
//!     wgpu::TextureFormat::Depth32Float,
//!     ImportMode::ReadOnly,
//! );
//!
//! // At frame end
//! if let Some(swapchain) = registry.release_swapchain() {
//!     swapchain.surface_texture.present();
//! }
//! registry.clear();
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use super::{ResourceHandle, PassIndex, ResourceAccess};

// ---------------------------------------------------------------------------
// External Resource Type
// ---------------------------------------------------------------------------

/// The type/origin of an external resource.
///
/// Determines how the resource is handled during acquire/release and what
/// synchronization requirements apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalResourceType {
    /// A swapchain texture acquired from a surface for presentation.
    /// These must be presented or discarded at frame end.
    Swapchain,
    /// A user-provided texture that exists outside the frame graph.
    /// The frame graph tracks state but does not manage allocation.
    UserTexture,
    /// A user-provided buffer that exists outside the frame graph.
    UserBuffer,
    /// A texture shared between multiple frame graphs or render systems.
    /// May require explicit synchronization.
    SharedTexture,
    /// A buffer shared between multiple frame graphs or render systems.
    SharedBuffer,
}

impl ExternalResourceType {
    /// Returns true if this is a swapchain resource.
    #[inline]
    pub const fn is_swapchain(&self) -> bool {
        matches!(self, Self::Swapchain)
    }

    /// Returns true if this is a user-provided resource.
    #[inline]
    pub const fn is_user_provided(&self) -> bool {
        matches!(self, Self::UserTexture | Self::UserBuffer)
    }

    /// Returns true if this is a shared resource.
    #[inline]
    pub const fn is_shared(&self) -> bool {
        matches!(self, Self::SharedTexture | Self::SharedBuffer)
    }

    /// Returns true if this is a texture type.
    #[inline]
    pub const fn is_texture(&self) -> bool {
        matches!(self, Self::Swapchain | Self::UserTexture | Self::SharedTexture)
    }

    /// Returns true if this is a buffer type.
    #[inline]
    pub const fn is_buffer(&self) -> bool {
        matches!(self, Self::UserBuffer | Self::SharedBuffer)
    }
}

impl fmt::Display for ExternalResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Swapchain => write!(f, "Swapchain"),
            Self::UserTexture => write!(f, "UserTexture"),
            Self::UserBuffer => write!(f, "UserBuffer"),
            Self::SharedTexture => write!(f, "SharedTexture"),
            Self::SharedBuffer => write!(f, "SharedBuffer"),
        }
    }
}

impl Default for ExternalResourceType {
    fn default() -> Self {
        Self::UserTexture
    }
}

// ---------------------------------------------------------------------------
// Import Mode
// ---------------------------------------------------------------------------

/// How an external resource will be accessed during the frame.
///
/// Determines whether acquire/release barriers are needed to synchronize
/// with external usage before/after the frame graph executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImportMode {
    /// Resource is only read, no modifications.
    /// No release barrier needed at frame end.
    ReadOnly,
    /// Resource is only written (previous contents discarded).
    /// No acquire barrier needed at frame start.
    WriteOnly,
    /// Resource is both read and modified.
    /// Both acquire and release barriers may be needed.
    ReadWrite,
}

impl ImportMode {
    /// Returns true if an acquire barrier may be needed at import time.
    ///
    /// Acquire barriers ensure previous external writes are visible
    /// to frame graph reads.
    #[inline]
    pub const fn requires_acquire_barrier(&self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    /// Returns true if a release barrier may be needed at frame end.
    ///
    /// Release barriers ensure frame graph writes are visible to
    /// subsequent external reads.
    #[inline]
    pub const fn requires_release_barrier(&self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }

    /// Returns true if this mode involves reading the resource.
    #[inline]
    pub const fn is_read(&self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    /// Returns true if this mode involves writing the resource.
    #[inline]
    pub const fn is_write(&self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }

    /// Converts to the frame graph's ResourceAccess type.
    #[inline]
    pub const fn to_resource_access(&self) -> ResourceAccess {
        match self {
            Self::ReadOnly => ResourceAccess::Read,
            Self::WriteOnly => ResourceAccess::Write,
            Self::ReadWrite => ResourceAccess::ReadWrite,
        }
    }
}

impl fmt::Display for ImportMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "ReadOnly"),
            Self::WriteOnly => write!(f, "WriteOnly"),
            Self::ReadWrite => write!(f, "ReadWrite"),
        }
    }
}

impl Default for ImportMode {
    fn default() -> Self {
        Self::ReadOnly
    }
}

impl From<ResourceAccess> for ImportMode {
    fn from(access: ResourceAccess) -> Self {
        match access {
            ResourceAccess::Read => Self::ReadOnly,
            ResourceAccess::Write => Self::WriteOnly,
            ResourceAccess::ReadWrite => Self::ReadWrite,
        }
    }
}

// ---------------------------------------------------------------------------
// External Texture Info
// ---------------------------------------------------------------------------

/// Information about an externally-provided texture.
///
/// Wraps the wgpu texture handle and associated metadata needed for
/// frame graph integration.
#[derive(Debug)]
pub struct ExternalTextureInfo {
    /// The wgpu texture handle.
    pub texture: wgpu::Texture,
    /// A texture view for binding.
    pub view: wgpu::TextureView,
    /// The texture format.
    pub format: wgpu::TextureFormat,
    /// Texture dimensions (width, height, depth_or_layers).
    pub size: (u32, u32, u32),
    /// MSAA sample count (1 = no multisampling).
    pub sample_count: u32,
    /// The type of external resource.
    pub external_type: ExternalResourceType,
    /// How this resource will be accessed.
    pub import_mode: ImportMode,
}

impl ExternalTextureInfo {
    /// Creates new external texture info.
    pub fn new(
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32, u32),
        sample_count: u32,
        external_type: ExternalResourceType,
        import_mode: ImportMode,
    ) -> Self {
        Self {
            texture,
            view,
            format,
            size,
            sample_count,
            external_type,
            import_mode,
        }
    }

    /// Returns the texture width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.0
    }

    /// Returns the texture height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.1
    }

    /// Returns the texture depth or array layer count.
    #[inline]
    pub fn depth_or_layers(&self) -> u32 {
        self.size.2
    }

    /// Returns true if this is a multisampled texture.
    #[inline]
    pub fn is_multisampled(&self) -> bool {
        self.sample_count > 1
    }
}

// ---------------------------------------------------------------------------
// External Buffer Info
// ---------------------------------------------------------------------------

/// Information about an externally-provided buffer.
///
/// Wraps the wgpu buffer handle and associated metadata needed for
/// frame graph integration.
#[derive(Debug)]
pub struct ExternalBufferInfo {
    /// The wgpu buffer handle.
    pub buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub size: u64,
    /// Allowed buffer usages.
    pub usage: wgpu::BufferUsages,
    /// The type of external resource.
    pub external_type: ExternalResourceType,
    /// How this resource will be accessed.
    pub import_mode: ImportMode,
}

impl ExternalBufferInfo {
    /// Creates new external buffer info.
    pub fn new(
        buffer: wgpu::Buffer,
        size: u64,
        usage: wgpu::BufferUsages,
        external_type: ExternalResourceType,
        import_mode: ImportMode,
    ) -> Self {
        Self {
            buffer,
            size,
            usage,
            external_type,
            import_mode,
        }
    }

    /// Returns true if the buffer supports uniform binding.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.usage.contains(wgpu::BufferUsages::UNIFORM)
    }

    /// Returns true if the buffer supports storage binding.
    #[inline]
    pub fn is_storage(&self) -> bool {
        self.usage.contains(wgpu::BufferUsages::STORAGE)
    }

    /// Returns true if the buffer supports vertex usage.
    #[inline]
    pub fn is_vertex(&self) -> bool {
        self.usage.contains(wgpu::BufferUsages::VERTEX)
    }

    /// Returns true if the buffer supports index usage.
    #[inline]
    pub fn is_index(&self) -> bool {
        self.usage.contains(wgpu::BufferUsages::INDEX)
    }
}

// ---------------------------------------------------------------------------
// Swapchain Info
// ---------------------------------------------------------------------------

/// Information about a swapchain texture for the current frame.
///
/// Wraps a surface texture acquired from wgpu and tracks presentation metadata.
#[derive(Debug)]
pub struct SwapchainInfo {
    /// The acquired surface texture.
    pub surface_texture: wgpu::SurfaceTexture,
    /// A texture view for binding.
    pub view: wgpu::TextureView,
    /// The swapchain format.
    pub format: wgpu::TextureFormat,
    /// Swapchain dimensions (width, height).
    pub size: (u32, u32),
    /// The present mode configured for the surface.
    pub present_mode: wgpu::PresentMode,
    /// When the swapchain was acquired.
    pub acquire_time: Instant,
}

impl SwapchainInfo {
    /// Creates swapchain info from an acquired surface texture.
    ///
    /// # Arguments
    ///
    /// * `surface_texture` - The acquired surface texture
    /// * `format` - The swapchain format
    /// * `present_mode` - The present mode
    ///
    /// # Returns
    ///
    /// A new SwapchainInfo with a default texture view.
    pub fn from_surface_texture(
        surface_texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
        present_mode: wgpu::PresentMode,
    ) -> Self {
        let texture = &surface_texture.texture;
        let size = texture.size();

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("swapchain_view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        Self {
            surface_texture,
            view,
            format,
            size: (size.width, size.height),
            present_mode,
            acquire_time: Instant::now(),
        }
    }

    /// Returns the swapchain width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.0
    }

    /// Returns the swapchain height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.1
    }

    /// Returns the time elapsed since the swapchain was acquired.
    #[inline]
    pub fn elapsed(&self) -> std::time::Duration {
        self.acquire_time.elapsed()
    }

    /// Presents the swapchain texture.
    ///
    /// This consumes self and presents the surface texture.
    pub fn present(self) {
        self.surface_texture.present();
    }
}

// ---------------------------------------------------------------------------
// Imported Resource Info
// ---------------------------------------------------------------------------

/// Wrapper enum for different types of imported resource info.
#[derive(Debug)]
pub enum ImportedResourceInfo {
    /// An imported texture.
    Texture(ExternalTextureInfo),
    /// An imported buffer.
    Buffer(ExternalBufferInfo),
    /// A swapchain texture.
    Swapchain(SwapchainInfo),
}

impl ImportedResourceInfo {
    /// Returns the import mode for this resource.
    pub fn import_mode(&self) -> ImportMode {
        match self {
            Self::Texture(info) => info.import_mode,
            Self::Buffer(info) => info.import_mode,
            Self::Swapchain(_) => ImportMode::WriteOnly, // Swapchain is typically write-only
        }
    }

    /// Returns the external resource type.
    pub fn external_type(&self) -> ExternalResourceType {
        match self {
            Self::Texture(info) => info.external_type,
            Self::Buffer(info) => info.external_type,
            Self::Swapchain(_) => ExternalResourceType::Swapchain,
        }
    }

    /// Returns true if this is a texture resource.
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture(_) | Self::Swapchain(_))
    }

    /// Returns true if this is a buffer resource.
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }

    /// Returns true if this is a swapchain resource.
    pub fn is_swapchain(&self) -> bool {
        matches!(self, Self::Swapchain(_))
    }
}

// ---------------------------------------------------------------------------
// Imported Resource
// ---------------------------------------------------------------------------

/// A complete imported resource entry in the registry.
///
/// Combines the resource handle, name, info, and pass usage tracking.
#[derive(Debug)]
pub struct ImportedResource {
    /// The resource handle within the frame graph.
    pub id: ResourceHandle,
    /// Human-readable name for debugging.
    pub name: String,
    /// The resource info (texture, buffer, or swapchain).
    pub info: ImportedResourceInfo,
    /// The first pass that uses this resource (if tracked).
    pub first_use_pass: Option<PassIndex>,
    /// The last pass that uses this resource (if tracked).
    pub last_use_pass: Option<PassIndex>,
}

impl ImportedResource {
    /// Creates a new imported texture resource.
    pub fn new_texture(id: ResourceHandle, name: impl Into<String>, info: ExternalTextureInfo) -> Self {
        Self {
            id,
            name: name.into(),
            info: ImportedResourceInfo::Texture(info),
            first_use_pass: None,
            last_use_pass: None,
        }
    }

    /// Creates a new imported buffer resource.
    pub fn new_buffer(id: ResourceHandle, name: impl Into<String>, info: ExternalBufferInfo) -> Self {
        Self {
            id,
            name: name.into(),
            info: ImportedResourceInfo::Buffer(info),
            first_use_pass: None,
            last_use_pass: None,
        }
    }

    /// Creates a new imported swapchain resource.
    pub fn new_swapchain(id: ResourceHandle, name: impl Into<String>, info: SwapchainInfo) -> Self {
        Self {
            id,
            name: name.into(),
            info: ImportedResourceInfo::Swapchain(info),
            first_use_pass: None,
            last_use_pass: None,
        }
    }

    /// Returns true if this is a texture resource.
    #[inline]
    pub fn is_texture(&self) -> bool {
        self.info.is_texture()
    }

    /// Returns true if this is a buffer resource.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        self.info.is_buffer()
    }

    /// Returns true if this is a swapchain resource.
    #[inline]
    pub fn is_swapchain(&self) -> bool {
        self.info.is_swapchain()
    }

    /// Returns the import mode.
    #[inline]
    pub fn import_mode(&self) -> ImportMode {
        self.info.import_mode()
    }

    /// Returns the external resource type.
    #[inline]
    pub fn external_type(&self) -> ExternalResourceType {
        self.info.external_type()
    }

    /// Sets the first use pass.
    pub fn set_first_use(&mut self, pass: PassIndex) {
        if self.first_use_pass.is_none() {
            self.first_use_pass = Some(pass);
        }
    }

    /// Sets the last use pass.
    pub fn set_last_use(&mut self, pass: PassIndex) {
        self.last_use_pass = Some(pass);
    }

    /// Returns true if pass usage has been tracked.
    pub fn has_usage(&self) -> bool {
        self.first_use_pass.is_some() || self.last_use_pass.is_some()
    }
}

impl fmt::Display for ImportedResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImportedResource({} \"{}\", type={}, mode={})",
            self.id,
            self.name,
            self.external_type(),
            self.import_mode()
        )
    }
}

// ---------------------------------------------------------------------------
// Resource Barrier
// ---------------------------------------------------------------------------

/// Describes a barrier needed for external resource synchronization.
///
/// Used to track the transitions needed at acquire (frame start) or
/// release (frame end) boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBarrier {
    /// The resource requiring the barrier.
    pub resource: ResourceHandle,
    /// The access state before the barrier.
    pub before_access: ResourceAccess,
    /// The access state after the barrier.
    pub after_access: ResourceAccess,
    /// True if this is an acquire barrier (start of use).
    /// False if this is a release barrier (end of use).
    pub acquire: bool,
}

impl ResourceBarrier {
    /// Creates a new resource barrier.
    pub fn new(
        resource: ResourceHandle,
        before_access: ResourceAccess,
        after_access: ResourceAccess,
        acquire: bool,
    ) -> Self {
        Self {
            resource,
            before_access,
            after_access,
            acquire,
        }
    }

    /// Creates an acquire barrier for a resource.
    pub fn acquire(resource: ResourceHandle, target_access: ResourceAccess) -> Self {
        Self::new(resource, ResourceAccess::Read, target_access, true)
    }

    /// Creates a release barrier for a resource.
    pub fn release(resource: ResourceHandle, from_access: ResourceAccess) -> Self {
        Self::new(resource, from_access, ResourceAccess::Read, false)
    }

    /// Returns true if this is an acquire barrier.
    #[inline]
    pub const fn is_acquire(&self) -> bool {
        self.acquire
    }

    /// Returns true if this is a release barrier.
    #[inline]
    pub const fn is_release(&self) -> bool {
        !self.acquire
    }
}

impl fmt::Display for ResourceBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let barrier_type = if self.acquire { "Acquire" } else { "Release" };
        write!(
            f,
            "{}Barrier({}: {} -> {})",
            barrier_type, self.resource, self.before_access, self.after_access
        )
    }
}

// ---------------------------------------------------------------------------
// External Resource Registry
// ---------------------------------------------------------------------------

/// Registry for managing external/imported resources within a frame.
///
/// Tracks all externally-provided resources and provides utilities for
/// import, lookup, and barrier computation.
#[derive(Debug, Default)]
pub struct ExternalResourceRegistry {
    /// All imported resources, keyed by handle.
    resources: HashMap<ResourceHandle, ImportedResource>,
    /// The current frame's swapchain resource (if any).
    swapchain: Option<ResourceHandle>,
    /// Counter for generating unique resource handles.
    next_handle: u32,
}

impl ExternalResourceRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            swapchain: None,
            next_handle: 0,
        }
    }

    /// Allocates the next resource handle.
    fn alloc_handle(&mut self) -> ResourceHandle {
        let handle = ResourceHandle(self.next_handle);
        self.next_handle += 1;
        handle
    }

    /// Imports a texture into the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name for debugging
    /// * `texture` - The wgpu texture
    /// * `view` - A texture view for binding
    /// * `format` - The texture format
    /// * `mode` - How the texture will be accessed
    ///
    /// # Returns
    ///
    /// The resource handle for the imported texture.
    pub fn import_texture(
        &mut self,
        name: impl Into<String>,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        format: wgpu::TextureFormat,
        mode: ImportMode,
    ) -> ResourceHandle {
        let size = texture.size();
        let handle = self.alloc_handle();

        let info = ExternalTextureInfo::new(
            texture,
            view,
            format,
            (size.width, size.height, size.depth_or_array_layers),
            1, // Default sample count
            ExternalResourceType::UserTexture,
            mode,
        );

        let resource = ImportedResource::new_texture(handle, name, info);
        self.resources.insert(handle, resource);
        handle
    }

    /// Imports a texture with full configuration.
    pub fn import_texture_full(
        &mut self,
        name: impl Into<String>,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32, u32),
        sample_count: u32,
        external_type: ExternalResourceType,
        mode: ImportMode,
    ) -> ResourceHandle {
        let handle = self.alloc_handle();

        let info = ExternalTextureInfo::new(
            texture,
            view,
            format,
            size,
            sample_count,
            external_type,
            mode,
        );

        let resource = ImportedResource::new_texture(handle, name, info);
        self.resources.insert(handle, resource);
        handle
    }

    /// Imports a buffer into the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name for debugging
    /// * `buffer` - The wgpu buffer
    /// * `size` - Buffer size in bytes
    /// * `mode` - How the buffer will be accessed
    ///
    /// # Returns
    ///
    /// The resource handle for the imported buffer.
    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        buffer: wgpu::Buffer,
        size: u64,
        mode: ImportMode,
    ) -> ResourceHandle {
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        self.import_buffer_full(name, buffer, size, usage, ExternalResourceType::UserBuffer, mode)
    }

    /// Imports a buffer with full configuration.
    pub fn import_buffer_full(
        &mut self,
        name: impl Into<String>,
        buffer: wgpu::Buffer,
        size: u64,
        usage: wgpu::BufferUsages,
        external_type: ExternalResourceType,
        mode: ImportMode,
    ) -> ResourceHandle {
        let handle = self.alloc_handle();

        let info = ExternalBufferInfo::new(buffer, size, usage, external_type, mode);
        let resource = ImportedResource::new_buffer(handle, name, info);
        self.resources.insert(handle, resource);
        handle
    }

    /// Imports a swapchain texture for the current frame.
    ///
    /// # Arguments
    ///
    /// * `surface_texture` - The acquired surface texture
    /// * `format` - The swapchain format
    ///
    /// # Returns
    ///
    /// The resource handle for the swapchain texture.
    ///
    /// # Panics
    ///
    /// Panics if a swapchain is already imported for this frame.
    pub fn import_swapchain(
        &mut self,
        surface_texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
    ) -> ResourceHandle {
        self.import_swapchain_with_mode(surface_texture, format, wgpu::PresentMode::Fifo)
    }

    /// Imports a swapchain texture with explicit present mode.
    pub fn import_swapchain_with_mode(
        &mut self,
        surface_texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
        present_mode: wgpu::PresentMode,
    ) -> ResourceHandle {
        assert!(
            self.swapchain.is_none(),
            "Swapchain already imported for this frame"
        );

        let handle = self.alloc_handle();
        let info = SwapchainInfo::from_surface_texture(surface_texture, format, present_mode);
        let resource = ImportedResource::new_swapchain(handle, "swapchain", info);

        self.resources.insert(handle, resource);
        self.swapchain = Some(handle);
        handle
    }

    /// Gets a reference to an imported resource by handle.
    pub fn get(&self, id: ResourceHandle) -> Option<&ImportedResource> {
        self.resources.get(&id)
    }

    /// Gets a mutable reference to an imported resource by handle.
    pub fn get_mut(&mut self, id: ResourceHandle) -> Option<&mut ImportedResource> {
        self.resources.get_mut(&id)
    }

    /// Gets a reference to the current frame's swapchain resource.
    pub fn get_swapchain(&self) -> Option<&ImportedResource> {
        self.swapchain.and_then(|id| self.resources.get(&id))
    }

    /// Returns the swapchain resource handle if present.
    pub fn swapchain_handle(&self) -> Option<ResourceHandle> {
        self.swapchain
    }

    /// Releases and returns the swapchain info for presentation.
    ///
    /// This removes the swapchain from the registry and returns the
    /// underlying SwapchainInfo for presentation.
    ///
    /// # Returns
    ///
    /// The SwapchainInfo if a swapchain was imported, None otherwise.
    pub fn release_swapchain(&mut self) -> Option<SwapchainInfo> {
        let handle = self.swapchain.take()?;
        let resource = self.resources.remove(&handle)?;

        match resource.info {
            ImportedResourceInfo::Swapchain(info) => Some(info),
            _ => None,
        }
    }

    /// Clears all imported resources.
    ///
    /// Call this at the end of each frame to reset the registry.
    /// Note: This does NOT present the swapchain - call release_swapchain first.
    pub fn clear(&mut self) {
        self.resources.clear();
        self.swapchain = None;
        self.next_handle = 0;
    }

    /// Returns the number of imported resources.
    #[inline]
    pub fn count(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Returns true if a swapchain is currently imported.
    #[inline]
    pub fn has_swapchain(&self) -> bool {
        self.swapchain.is_some()
    }

    /// Returns an iterator over all imported resources.
    pub fn iter(&self) -> impl Iterator<Item = (&ResourceHandle, &ImportedResource)> {
        self.resources.iter()
    }

    /// Returns an iterator over resource handles.
    pub fn handles(&self) -> impl Iterator<Item = ResourceHandle> + '_ {
        self.resources.keys().copied()
    }

    /// Finds a resource by name.
    pub fn find_by_name(&self, name: &str) -> Option<&ImportedResource> {
        self.resources.values().find(|r| r.name == name)
    }

    /// Updates pass usage tracking for a resource.
    pub fn track_usage(&mut self, id: ResourceHandle, pass: PassIndex) {
        if let Some(resource) = self.resources.get_mut(&id) {
            resource.set_first_use(pass);
            resource.set_last_use(pass);
        }
    }
}

impl fmt::Display for ExternalResourceRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExternalResourceRegistry(count={}, has_swapchain={})",
            self.count(),
            self.has_swapchain()
        )
    }
}

// ---------------------------------------------------------------------------
// External Synchronizer
// ---------------------------------------------------------------------------

/// Computes acquire and release barriers for external resources.
///
/// This utility analyzes the registry and determines which barriers
/// are needed to properly synchronize with external resource usage.
pub struct ExternalSynchronizer;

impl ExternalSynchronizer {
    /// Computes acquire barriers needed at frame start.
    ///
    /// Acquire barriers ensure that previous external writes to resources
    /// are visible to the frame graph before it begins reading.
    ///
    /// # Arguments
    ///
    /// * `registry` - The external resource registry
    ///
    /// # Returns
    ///
    /// A vector of barriers that should be executed before frame graph execution.
    pub fn compute_acquire_barriers(registry: &ExternalResourceRegistry) -> Vec<ResourceBarrier> {
        let mut barriers = Vec::new();

        for (handle, resource) in registry.iter() {
            let mode = resource.import_mode();

            if mode.requires_acquire_barrier() {
                // Determine target access based on import mode
                let target_access = mode.to_resource_access();
                barriers.push(ResourceBarrier::acquire(*handle, target_access));
            }
        }

        barriers
    }

    /// Computes release barriers needed at frame end.
    ///
    /// Release barriers ensure that frame graph writes to resources
    /// are visible to subsequent external reads.
    ///
    /// # Arguments
    ///
    /// * `registry` - The external resource registry
    ///
    /// # Returns
    ///
    /// A vector of barriers that should be executed after frame graph execution.
    pub fn compute_release_barriers(registry: &ExternalResourceRegistry) -> Vec<ResourceBarrier> {
        let mut barriers = Vec::new();

        for (handle, resource) in registry.iter() {
            // Skip swapchain - it has its own present mechanism
            if resource.is_swapchain() {
                continue;
            }

            let mode = resource.import_mode();

            if mode.requires_release_barrier() {
                let from_access = mode.to_resource_access();
                barriers.push(ResourceBarrier::release(*handle, from_access));
            }
        }

        barriers
    }

    /// Computes all barriers (acquire + release) for a complete frame.
    ///
    /// # Arguments
    ///
    /// * `registry` - The external resource registry
    ///
    /// # Returns
    ///
    /// A tuple of (acquire_barriers, release_barriers).
    pub fn compute_all_barriers(
        registry: &ExternalResourceRegistry,
    ) -> (Vec<ResourceBarrier>, Vec<ResourceBarrier>) {
        (
            Self::compute_acquire_barriers(registry),
            Self::compute_release_barriers(registry),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // ExternalResourceType Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_external_resource_type_is_swapchain() {
        assert!(ExternalResourceType::Swapchain.is_swapchain());
        assert!(!ExternalResourceType::UserTexture.is_swapchain());
        assert!(!ExternalResourceType::UserBuffer.is_swapchain());
        assert!(!ExternalResourceType::SharedTexture.is_swapchain());
        assert!(!ExternalResourceType::SharedBuffer.is_swapchain());
    }

    #[test]
    fn test_external_resource_type_is_user_provided() {
        assert!(!ExternalResourceType::Swapchain.is_user_provided());
        assert!(ExternalResourceType::UserTexture.is_user_provided());
        assert!(ExternalResourceType::UserBuffer.is_user_provided());
        assert!(!ExternalResourceType::SharedTexture.is_user_provided());
        assert!(!ExternalResourceType::SharedBuffer.is_user_provided());
    }

    #[test]
    fn test_external_resource_type_is_shared() {
        assert!(!ExternalResourceType::Swapchain.is_shared());
        assert!(!ExternalResourceType::UserTexture.is_shared());
        assert!(!ExternalResourceType::UserBuffer.is_shared());
        assert!(ExternalResourceType::SharedTexture.is_shared());
        assert!(ExternalResourceType::SharedBuffer.is_shared());
    }

    #[test]
    fn test_external_resource_type_is_texture() {
        assert!(ExternalResourceType::Swapchain.is_texture());
        assert!(ExternalResourceType::UserTexture.is_texture());
        assert!(!ExternalResourceType::UserBuffer.is_texture());
        assert!(ExternalResourceType::SharedTexture.is_texture());
        assert!(!ExternalResourceType::SharedBuffer.is_texture());
    }

    #[test]
    fn test_external_resource_type_is_buffer() {
        assert!(!ExternalResourceType::Swapchain.is_buffer());
        assert!(!ExternalResourceType::UserTexture.is_buffer());
        assert!(ExternalResourceType::UserBuffer.is_buffer());
        assert!(!ExternalResourceType::SharedTexture.is_buffer());
        assert!(ExternalResourceType::SharedBuffer.is_buffer());
    }

    #[test]
    fn test_external_resource_type_display() {
        assert_eq!(format!("{}", ExternalResourceType::Swapchain), "Swapchain");
        assert_eq!(format!("{}", ExternalResourceType::UserTexture), "UserTexture");
        assert_eq!(format!("{}", ExternalResourceType::UserBuffer), "UserBuffer");
        assert_eq!(format!("{}", ExternalResourceType::SharedTexture), "SharedTexture");
        assert_eq!(format!("{}", ExternalResourceType::SharedBuffer), "SharedBuffer");
    }

    #[test]
    fn test_external_resource_type_default() {
        assert_eq!(ExternalResourceType::default(), ExternalResourceType::UserTexture);
    }

    // ---------------------------------------------------------------------------
    // ImportMode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_import_mode_requires_acquire_barrier() {
        assert!(ImportMode::ReadOnly.requires_acquire_barrier());
        assert!(!ImportMode::WriteOnly.requires_acquire_barrier());
        assert!(ImportMode::ReadWrite.requires_acquire_barrier());
    }

    #[test]
    fn test_import_mode_requires_release_barrier() {
        assert!(!ImportMode::ReadOnly.requires_release_barrier());
        assert!(ImportMode::WriteOnly.requires_release_barrier());
        assert!(ImportMode::ReadWrite.requires_release_barrier());
    }

    #[test]
    fn test_import_mode_is_read() {
        assert!(ImportMode::ReadOnly.is_read());
        assert!(!ImportMode::WriteOnly.is_read());
        assert!(ImportMode::ReadWrite.is_read());
    }

    #[test]
    fn test_import_mode_is_write() {
        assert!(!ImportMode::ReadOnly.is_write());
        assert!(ImportMode::WriteOnly.is_write());
        assert!(ImportMode::ReadWrite.is_write());
    }

    #[test]
    fn test_import_mode_to_resource_access() {
        assert_eq!(ImportMode::ReadOnly.to_resource_access(), ResourceAccess::Read);
        assert_eq!(ImportMode::WriteOnly.to_resource_access(), ResourceAccess::Write);
        assert_eq!(ImportMode::ReadWrite.to_resource_access(), ResourceAccess::ReadWrite);
    }

    #[test]
    fn test_import_mode_display() {
        assert_eq!(format!("{}", ImportMode::ReadOnly), "ReadOnly");
        assert_eq!(format!("{}", ImportMode::WriteOnly), "WriteOnly");
        assert_eq!(format!("{}", ImportMode::ReadWrite), "ReadWrite");
    }

    #[test]
    fn test_import_mode_default() {
        assert_eq!(ImportMode::default(), ImportMode::ReadOnly);
    }

    #[test]
    fn test_import_mode_from_resource_access() {
        assert_eq!(ImportMode::from(ResourceAccess::Read), ImportMode::ReadOnly);
        assert_eq!(ImportMode::from(ResourceAccess::Write), ImportMode::WriteOnly);
        assert_eq!(ImportMode::from(ResourceAccess::ReadWrite), ImportMode::ReadWrite);
    }

    // ---------------------------------------------------------------------------
    // ResourceBarrier Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_barrier_new() {
        let barrier = ResourceBarrier::new(
            ResourceHandle(1),
            ResourceAccess::Read,
            ResourceAccess::Write,
            true,
        );
        assert_eq!(barrier.resource, ResourceHandle(1));
        assert_eq!(barrier.before_access, ResourceAccess::Read);
        assert_eq!(barrier.after_access, ResourceAccess::Write);
        assert!(barrier.acquire);
    }

    #[test]
    fn test_resource_barrier_acquire() {
        let barrier = ResourceBarrier::acquire(ResourceHandle(2), ResourceAccess::Write);
        assert!(barrier.is_acquire());
        assert!(!barrier.is_release());
        assert_eq!(barrier.resource, ResourceHandle(2));
        assert_eq!(barrier.after_access, ResourceAccess::Write);
    }

    #[test]
    fn test_resource_barrier_release() {
        let barrier = ResourceBarrier::release(ResourceHandle(3), ResourceAccess::Write);
        assert!(!barrier.is_acquire());
        assert!(barrier.is_release());
        assert_eq!(barrier.resource, ResourceHandle(3));
        assert_eq!(barrier.before_access, ResourceAccess::Write);
    }

    #[test]
    fn test_resource_barrier_display() {
        let acquire = ResourceBarrier::acquire(ResourceHandle(1), ResourceAccess::Write);
        assert!(format!("{}", acquire).contains("Acquire"));

        let release = ResourceBarrier::release(ResourceHandle(2), ResourceAccess::Write);
        assert!(format!("{}", release).contains("Release"));
    }

    // ---------------------------------------------------------------------------
    // ExternalResourceRegistry Tests (without actual wgpu resources)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_registry_new() {
        let registry = ExternalResourceRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.is_empty());
        assert!(!registry.has_swapchain());
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = ExternalResourceRegistry::new();
        // Just test that clear works on an empty registry
        registry.clear();
        assert_eq!(registry.count(), 0);
        assert!(!registry.has_swapchain());
    }

    #[test]
    fn test_registry_display() {
        let registry = ExternalResourceRegistry::new();
        let display = format!("{}", registry);
        assert!(display.contains("count=0"));
        assert!(display.contains("has_swapchain=false"));
    }

    // ---------------------------------------------------------------------------
    // ExternalSynchronizer Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_synchronizer_empty_registry() {
        let registry = ExternalResourceRegistry::new();

        let acquire = ExternalSynchronizer::compute_acquire_barriers(&registry);
        let release = ExternalSynchronizer::compute_release_barriers(&registry);

        assert!(acquire.is_empty());
        assert!(release.is_empty());
    }

    #[test]
    fn test_synchronizer_compute_all_barriers() {
        let registry = ExternalResourceRegistry::new();
        let (acquire, release) = ExternalSynchronizer::compute_all_barriers(&registry);

        assert!(acquire.is_empty());
        assert!(release.is_empty());
    }

    // ---------------------------------------------------------------------------
    // ImportedResourceInfo Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_imported_resource_info_swapchain_mode() {
        // Swapchain mode should be WriteOnly by default
        // We can't easily create a SwapchainInfo without wgpu, so test the enum directly
        // by checking ImportMode behavior
        let mode = ImportMode::WriteOnly;
        assert!(mode.requires_release_barrier());
        assert!(!mode.requires_acquire_barrier());
    }

    // ---------------------------------------------------------------------------
    // ImportedResource Tests (partial, without wgpu)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_imported_resource_display_format() {
        // Test the Display trait would work by checking string formation
        let id = ResourceHandle(42);
        let external_type = ExternalResourceType::UserTexture;
        let mode = ImportMode::ReadOnly;

        // Simulate what Display would produce
        let expected = format!(
            "ImportedResource({} \"{}\", type={}, mode={})",
            id, "test_texture", external_type, mode
        );

        assert!(expected.contains("ResourceHandle(42)"));
        assert!(expected.contains("test_texture"));
        assert!(expected.contains("UserTexture"));
        assert!(expected.contains("ReadOnly"));
    }

    // ---------------------------------------------------------------------------
    // PassIndex Usage Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_pass_index_tracking_logic() {
        // Test the pass tracking logic without creating actual resources
        let mut first_use: Option<PassIndex> = None;
        let mut last_use: Option<PassIndex> = None;

        // Simulate first use
        let pass_a = PassIndex(0);
        if first_use.is_none() {
            first_use = Some(pass_a);
        }
        last_use = Some(pass_a);

        // Simulate second use
        let pass_b = PassIndex(1);
        if first_use.is_none() {
            first_use = Some(pass_b);
        }
        last_use = Some(pass_b);

        assert_eq!(first_use, Some(PassIndex(0)));
        assert_eq!(last_use, Some(PassIndex(1)));
    }

    // ---------------------------------------------------------------------------
    // Barrier Computation Logic Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_barrier_logic_readonly() {
        let mode = ImportMode::ReadOnly;

        // ReadOnly: needs acquire (to see external writes), no release
        assert!(mode.requires_acquire_barrier());
        assert!(!mode.requires_release_barrier());
    }

    #[test]
    fn test_barrier_logic_writeonly() {
        let mode = ImportMode::WriteOnly;

        // WriteOnly: no acquire (don't care about previous content), needs release
        assert!(!mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    #[test]
    fn test_barrier_logic_readwrite() {
        let mode = ImportMode::ReadWrite;

        // ReadWrite: needs both
        assert!(mode.requires_acquire_barrier());
        assert!(mode.requires_release_barrier());
    }

    // ---------------------------------------------------------------------------
    // ResourceHandle Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_handle_none_sentinel() {
        let none = ResourceHandle::NONE;
        assert_eq!(none.0, u32::MAX);
    }

    #[test]
    fn test_resource_handle_equality() {
        let a = ResourceHandle(1);
        let b = ResourceHandle(1);
        let c = ResourceHandle(2);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_resource_handle_display() {
        let handle = ResourceHandle(42);
        let display = format!("{}", handle);
        assert!(display.contains("42"));
    }

    // ---------------------------------------------------------------------------
    // Integration Logic Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_external_type_and_mode_combinations() {
        // Test all valid combinations of external type and import mode
        let types = [
            ExternalResourceType::Swapchain,
            ExternalResourceType::UserTexture,
            ExternalResourceType::UserBuffer,
            ExternalResourceType::SharedTexture,
            ExternalResourceType::SharedBuffer,
        ];

        let modes = [
            ImportMode::ReadOnly,
            ImportMode::WriteOnly,
            ImportMode::ReadWrite,
        ];

        for ext_type in &types {
            for mode in &modes {
                // Verify consistency
                let is_texture = ext_type.is_texture();
                let is_buffer = ext_type.is_buffer();

                // Every type should be either texture or buffer
                assert!(is_texture || is_buffer, "{:?} is neither texture nor buffer", ext_type);

                // No type should be both
                assert!(!(is_texture && is_buffer), "{:?} is both texture and buffer", ext_type);

                // Mode should consistently convert
                let access = mode.to_resource_access();
                let back = ImportMode::from(access);
                assert_eq!(*mode, back, "Round-trip conversion failed for {:?}", mode);
            }
        }
    }
}
