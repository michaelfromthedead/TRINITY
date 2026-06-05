//! Draw command wrappers for wgpu 22-25.x render passes.
//!
//! This module provides a comprehensive abstraction for all draw commands,
//! including basic draws, indirect draws, and multi-draw variants with
//! feature detection helpers.
//!
//! # Overview
//!
//! The `DrawCommands` struct extends render pass functionality with ergonomic
//! draw methods supporting all 7 wgpu draw variants:
//!
//! ## Basic Draws
//! - [`draw`](DrawCommands::draw) - Draw non-indexed geometry
//! - [`draw_indexed`](DrawCommands::draw_indexed) - Draw indexed geometry
//!
//! ## Indirect Draws
//! - [`draw_indirect`](DrawCommands::draw_indirect) - GPU-driven draw
//! - [`draw_indexed_indirect`](DrawCommands::draw_indexed_indirect) - GPU-driven indexed draw
//!
//! ## Multi-Draw Indirect (Feature-Gated)
//! - [`multi_draw_indirect`](DrawCommands::multi_draw_indirect) - Multiple draws from buffer
//! - [`multi_draw_indexed_indirect`](DrawCommands::multi_draw_indexed_indirect) - Multiple indexed draws
//! - [`multi_draw_indirect_count`](DrawCommands::multi_draw_indirect_count) - GPU count control
//! - [`multi_draw_indexed_indirect_count`](DrawCommands::multi_draw_indexed_indirect_count) - GPU count indexed
//!
//! # Architecture
//!
//! ```text
//! DrawCommands<'a>
//!     |-- pass: &'a mut wgpu::RenderPass<'a>
//!     |
//!     |-- Basic Draws
//!     |   |-- draw(vertices, instances)
//!     |   `-- draw_indexed(indices, base_vertex, instances)
//!     |
//!     |-- Indirect Draws
//!     |   |-- draw_indirect(buffer, offset)
//!     |   `-- draw_indexed_indirect(buffer, offset)
//!     |
//!     `-- Multi-Draw (feature-gated)
//!         |-- multi_draw_indirect(buffer, offset, count)
//!         |-- multi_draw_indexed_indirect(buffer, offset, count)
//!         |-- multi_draw_indirect_count(...)
//!         `-- multi_draw_indexed_indirect_count(...)
//! ```
//!
//! # Feature Requirements
//!
//! | Method | Feature Required |
//! |--------|-----------------|
//! | `draw`, `draw_indexed` | None |
//! | `draw_indirect`, `draw_indexed_indirect` | None |
//! | `multi_draw_indirect`, `multi_draw_indexed_indirect` | `MULTI_DRAW_INDIRECT` |
//! | `multi_draw_indirect_count`, `multi_draw_indexed_indirect_count` | `MULTI_DRAW_INDIRECT_COUNT` |
//!
//! # GPU-Side Argument Structs
//!
//! For indirect draws, the GPU reads arguments from a buffer. Use these structs:
//!
//! | Struct | Size | Use With |
//! |--------|------|----------|
//! | [`DrawIndirectArgs`] | 16 bytes | `draw_indirect`, `multi_draw_indirect` |
//! | [`DrawIndexedIndirectArgs`] | 20 bytes | `draw_indexed_indirect`, `multi_draw_indexed_indirect` |
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::draw_commands::{
//!     DrawCommands, DrawIndirectArgs, DrawIndexedIndirectArgs,
//!     supports_multi_draw_indirect,
//! };
//!
//! // Wrap render pass with draw commands
//! let mut draws = DrawCommands::new(&mut render_pass);
//!
//! // Basic indexed draw
//! draws.draw_indexed(0..36, 0, 0..1);
//!
//! // Indirect draw (GPU provides parameters)
//! draws.draw_indexed_indirect(&indirect_buffer, 0);
//!
//! // Multi-draw if supported
//! if supports_multi_draw_indirect(device) {
//!     draws.multi_draw_indexed_indirect(&command_buffer, 0, 100);
//! }
//! ```
//!
//! # Thread Safety
//!
//! `DrawCommands` is **not** `Send` or `Sync` because `wgpu::RenderPass`
//! is not thread-safe. Render passes must be used on the thread that created them.

use std::fmt;
use std::ops::Range;

// ============================================================================
// DrawIndirectArgs - Basic non-indexed indirect arguments
// ============================================================================

/// Arguments for indirect draw commands (non-indexed).
///
/// This struct matches `wgpu::util::DrawIndirectArgs` exactly at 16 bytes.
/// It is used with [`draw_indirect`](DrawCommands::draw_indirect) and
/// [`multi_draw_indirect`](DrawCommands::multi_draw_indirect).
///
/// # Memory Layout (16 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       4     vertex_count (u32)
/// 4       4     instance_count (u32)
/// 8       4     first_vertex (u32)
/// 12      4     first_instance (u32)
/// ----
/// 16 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct DrawIndirectArgs {
///     vertex_count: u32,
///     instance_count: u32,
///     first_vertex: u32,
///     first_instance: u32,
/// }
/// ```
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::DrawIndirectArgs;
///
/// // Draw 3 vertices (a triangle), 10 instances
/// let args = DrawIndirectArgs::new(3, 10, 0, 0);
///
/// // Draw from a specific vertex offset
/// let args_offset = DrawIndirectArgs::new(6, 1, 100, 0);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DrawIndirectArgs {
    /// Number of vertices to draw.
    pub vertex_count: u32,

    /// Number of instances to draw.
    pub instance_count: u32,

    /// Index of the first vertex to draw.
    pub first_vertex: u32,

    /// First instance to draw.
    pub first_instance: u32,
}

// Safety: DrawIndirectArgs is #[repr(C)] with only u32 fields
unsafe impl bytemuck::Pod for DrawIndirectArgs {}
unsafe impl bytemuck::Zeroable for DrawIndirectArgs {}

impl DrawIndirectArgs {
    /// Size in bytes (16 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Stride between consecutive commands in a buffer (16 bytes).
    pub const STRIDE: u64 = Self::SIZE;

    /// Create new draw arguments.
    ///
    /// # Arguments
    ///
    /// * `vertex_count` - Number of vertices to draw
    /// * `instance_count` - Number of instances to draw
    /// * `first_vertex` - Starting vertex index
    /// * `first_instance` - Starting instance ID
    #[inline]
    pub const fn new(
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }

    /// Create a zeroed (no-op) draw command.
    ///
    /// A draw with `instance_count = 0` draws nothing.
    #[inline]
    pub const fn zeroed() -> Self {
        Self {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    /// Create a single-instance draw for the specified vertex count.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::render_pipeline::draw_commands::DrawIndirectArgs;
    ///
    /// // Draw a single triangle
    /// let args = DrawIndirectArgs::single(3);
    /// assert_eq!(args.vertex_count, 3);
    /// assert_eq!(args.instance_count, 1);
    /// ```
    #[inline]
    pub const fn single(vertex_count: u32) -> Self {
        Self::new(vertex_count, 1, 0, 0)
    }

    /// Create from vertex and instance ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::render_pipeline::draw_commands::DrawIndirectArgs;
    ///
    /// let args = DrawIndirectArgs::from_ranges(0..100, 0..10);
    /// assert_eq!(args.vertex_count, 100);
    /// assert_eq!(args.instance_count, 10);
    /// ```
    #[inline]
    pub fn from_ranges(vertices: Range<u32>, instances: Range<u32>) -> Self {
        Self {
            vertex_count: vertices.end.saturating_sub(vertices.start),
            instance_count: instances.end.saturating_sub(instances.start),
            first_vertex: vertices.start,
            first_instance: instances.start,
        }
    }

    /// Returns true if this draw command will draw any geometry.
    #[inline]
    pub const fn will_draw(&self) -> bool {
        self.vertex_count > 0 && self.instance_count > 0
    }

    /// Total number of vertices processed (vertex_count * instance_count).
    #[inline]
    pub const fn total_vertices(&self) -> u64 {
        self.vertex_count as u64 * self.instance_count as u64
    }

    /// Convert to byte slice for buffer uploads.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ============================================================================
// DrawIndexedIndirectArgs - Indexed indirect arguments
// ============================================================================

/// Arguments for indirect indexed draw commands.
///
/// This struct matches `wgpu::util::DrawIndexedIndirectArgs` exactly at 20 bytes.
/// It is used with [`draw_indexed_indirect`](DrawCommands::draw_indexed_indirect) and
/// [`multi_draw_indexed_indirect`](DrawCommands::multi_draw_indexed_indirect).
///
/// # Memory Layout (20 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       4     index_count (u32)
/// 4       4     instance_count (u32)
/// 8       4     first_index (u32)
/// 12      4     base_vertex (i32) - NOTE: signed!
/// 16      4     first_instance (u32)
/// ----
/// 20 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct DrawIndexedIndirectArgs {
///     index_count: u32,
///     instance_count: u32,
///     first_index: u32,
///     base_vertex: i32,
///     first_instance: u32,
/// }
/// ```
///
/// # Note on base_vertex
///
/// The `base_vertex` field is a signed integer (i32), allowing negative offsets.
/// This is useful when combining meshes with different vertex buffer layouts.
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::DrawIndexedIndirectArgs;
///
/// // Draw 36 indices (a cube), single instance
/// let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
///
/// // Draw with negative base vertex (packed vertex buffer)
/// let args_offset = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    pub index_count: u32,

    /// Number of instances to draw.
    pub instance_count: u32,

    /// First index to start drawing from.
    pub first_index: u32,

    /// Base vertex offset added to each index value.
    /// This is **signed** (i32) to allow negative offsets.
    pub base_vertex: i32,

    /// First instance to start drawing from.
    pub first_instance: u32,
}

// Safety: DrawIndexedIndirectArgs is #[repr(C)] with only primitive fields
unsafe impl bytemuck::Pod for DrawIndexedIndirectArgs {}
unsafe impl bytemuck::Zeroable for DrawIndexedIndirectArgs {}

impl DrawIndexedIndirectArgs {
    /// Size in bytes (20 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Stride between consecutive commands in a buffer.
    ///
    /// Note: For optimal alignment in storage buffers, consider using
    /// 24 bytes (padded to 4-byte boundary * 6 fields) or the padded
    /// version from `resources::indirect`.
    pub const STRIDE: u64 = Self::SIZE;

    /// Padded stride for storage buffer alignment (24 bytes).
    ///
    /// Use this when storing in `storage` buffer arrays where 16-byte
    /// or power-of-2 alignment is beneficial.
    pub const STRIDE_PADDED: u64 = 24;

    /// Create new indexed draw arguments.
    ///
    /// # Arguments
    ///
    /// * `index_count` - Number of indices to draw
    /// * `instance_count` - Number of instances to draw
    /// * `first_index` - Starting index in the index buffer
    /// * `base_vertex` - Value added to each index (can be negative)
    /// * `first_instance` - Starting instance ID
    #[inline]
    pub const fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    /// Create a zeroed (no-op) draw command.
    #[inline]
    pub const fn zeroed() -> Self {
        Self {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }

    /// Create a single-instance draw for the specified index count.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::render_pipeline::draw_commands::DrawIndexedIndirectArgs;
    ///
    /// // Draw a single cube (36 indices)
    /// let args = DrawIndexedIndirectArgs::single(36);
    /// assert_eq!(args.index_count, 36);
    /// assert_eq!(args.instance_count, 1);
    /// ```
    #[inline]
    pub const fn single(index_count: u32) -> Self {
        Self::new(index_count, 1, 0, 0, 0)
    }

    /// Create from index and instance ranges with base vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::render_pipeline::draw_commands::DrawIndexedIndirectArgs;
    ///
    /// let args = DrawIndexedIndirectArgs::from_ranges(0..36, 0, 0..10);
    /// assert_eq!(args.index_count, 36);
    /// assert_eq!(args.instance_count, 10);
    /// ```
    #[inline]
    pub fn from_ranges(indices: Range<u32>, base_vertex: i32, instances: Range<u32>) -> Self {
        Self {
            index_count: indices.end.saturating_sub(indices.start),
            instance_count: instances.end.saturating_sub(instances.start),
            first_index: indices.start,
            base_vertex,
            first_instance: instances.start,
        }
    }

    /// Returns true if this draw command will draw any geometry.
    #[inline]
    pub const fn will_draw(&self) -> bool {
        self.index_count > 0 && self.instance_count > 0
    }

    /// Total number of indices processed (index_count * instance_count).
    #[inline]
    pub const fn total_indices(&self) -> u64 {
        self.index_count as u64 * self.instance_count as u64
    }

    /// Convert to byte slice for buffer uploads.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ============================================================================
// Feature Detection
// ============================================================================

/// Check if the device supports `MULTI_DRAW_INDIRECT` feature.
///
/// Required for:
/// - [`multi_draw_indirect`](DrawCommands::multi_draw_indirect)
/// - [`multi_draw_indexed_indirect`](DrawCommands::multi_draw_indexed_indirect)
///
/// # Example
///
/// ```ignore
/// if supports_multi_draw_indirect(device) {
///     draws.multi_draw_indirect(&buffer, 0, 100);
/// } else {
///     // Fall back to loop of draw_indirect calls
///     for i in 0..100 {
///         draws.draw_indirect(&buffer, i * DrawIndirectArgs::STRIDE);
///     }
/// }
/// ```
#[inline]
pub fn supports_multi_draw_indirect(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT)
}

/// Check if the device supports `MULTI_DRAW_INDIRECT_COUNT` feature.
///
/// Required for:
/// - [`multi_draw_indirect_count`](DrawCommands::multi_draw_indirect_count)
/// - [`multi_draw_indexed_indirect_count`](DrawCommands::multi_draw_indexed_indirect_count)
///
/// This is the most advanced multi-draw feature, allowing the GPU to control
/// both the draw arguments AND the number of draws.
///
/// # Example
///
/// ```ignore
/// if supports_multi_draw_indirect_count(device) {
///     draws.multi_draw_indexed_indirect_count(
///         &indirect_buffer, 0,
///         &count_buffer, 0,
///         max_draws,
///     );
/// }
/// ```
#[inline]
pub fn supports_multi_draw_indirect_count(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
}

/// Get the required features for multi-draw operations.
///
/// # Returns
///
/// A tuple of (multi_draw_features, multi_draw_count_features).
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::required_multi_draw_features;
///
/// let (multi_draw, multi_draw_count) = required_multi_draw_features();
/// ```
#[inline]
pub fn required_multi_draw_features() -> (wgpu::Features, wgpu::Features) {
    (
        wgpu::Features::MULTI_DRAW_INDIRECT,
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
    )
}

/// Multi-draw support tier.
///
/// Represents the level of multi-draw support available on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MultiDrawTier {
    /// No multi-draw support. Must use individual draw calls.
    None,

    /// Basic multi-draw support (`MULTI_DRAW_INDIRECT`).
    /// Can batch multiple draws but count must be known at CPU.
    Basic,

    /// Full multi-draw support (`MULTI_DRAW_INDIRECT_COUNT`).
    /// GPU can control both draw arguments and draw count.
    Full,
}

impl MultiDrawTier {
    /// Detect the multi-draw tier from device features.
    #[inline]
    pub fn from_device(device: &wgpu::Device) -> Self {
        let features = device.features();
        if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            Self::Full
        } else if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            Self::Basic
        } else {
            Self::None
        }
    }

    /// Detect the multi-draw tier from feature flags.
    #[inline]
    pub fn from_features(features: wgpu::Features) -> Self {
        if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            Self::Full
        } else if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            Self::Basic
        } else {
            Self::None
        }
    }

    /// Returns true if basic multi-draw is supported.
    #[inline]
    pub const fn supports_multi_draw(&self) -> bool {
        matches!(self, Self::Basic | Self::Full)
    }

    /// Returns true if multi-draw with GPU count is supported.
    #[inline]
    pub const fn supports_multi_draw_count(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Get a human-readable description of this tier.
    #[inline]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::None => "No multi-draw support",
            Self::Basic => "Multi-draw indirect (CPU count)",
            Self::Full => "Multi-draw indirect count (GPU count)",
        }
    }
}

impl fmt::Display for MultiDrawTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

// ============================================================================
// DrawCommands
// ============================================================================

/// High-level wrapper for draw commands on a render pass.
///
/// Provides a fluent API for all draw variants including basic draws,
/// indirect draws, and multi-draw operations.
///
/// # Lifetime
///
/// The lifetime `'a` corresponds to the render pass lifetime, which is
/// typically tied to the command encoder's encoding scope.
///
/// # Example
///
/// ```ignore
/// let mut draws = DrawCommands::new(&mut render_pass);
///
/// // Basic draws
/// draws
///     .draw(0..100, 0..1)
///     .draw_indexed(0..36, 0, 0..10);
///
/// // Indirect draws
/// draws
///     .draw_indirect(&indirect_buffer, 0)
///     .draw_indexed_indirect(&indirect_buffer, DrawIndirectArgs::SIZE);
///
/// // Multi-draw (feature-gated)
/// draws.multi_draw_indexed_indirect(&command_buffer, 0, 100);
/// ```
pub struct DrawCommands<'a> {
    pass: &'a mut wgpu::RenderPass<'a>,
}

impl<'a> DrawCommands<'a> {
    /// Create a new draw commands wrapper.
    ///
    /// # Arguments
    ///
    /// * `pass` - Mutable reference to the render pass
    #[inline]
    pub fn new(pass: &'a mut wgpu::RenderPass<'a>) -> Self {
        Self { pass }
    }

    /// Get a reference to the inner render pass.
    #[inline]
    pub fn inner(&self) -> &wgpu::RenderPass<'a> {
        self.pass
    }

    /// Get a mutable reference to the inner render pass.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut wgpu::RenderPass<'a> {
        self.pass
    }

    // =========================================================================
    // Basic Draw Commands (Criteria 1 & 2)
    // =========================================================================

    /// Draw non-indexed geometry.
    ///
    /// # Arguments
    ///
    /// * `vertices` - Range of vertices to draw
    /// * `instances` - Range of instances to draw
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw 100 vertices, single instance
    /// draws.draw(0..100, 0..1);
    ///
    /// // Draw 3 vertices (triangle), 50 instances
    /// draws.draw(0..3, 0..50);
    ///
    /// // Draw from vertex offset
    /// draws.draw(100..200, 5..15);
    /// ```
    ///
    /// # Notes
    ///
    /// - Vertices are read from vertex buffers bound with `set_vertex_buffer`
    /// - Instance ID starts from `instances.start` and increments
    #[inline]
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> &mut Self {
        self.pass.draw(vertices, instances);
        self
    }

    /// Draw indexed geometry.
    ///
    /// # Arguments
    ///
    /// * `indices` - Range of indices to draw from the index buffer
    /// * `base_vertex` - Value added to each index value
    /// * `instances` - Range of instances to draw
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw 36 indices (a cube), single instance
    /// draws.draw_indexed(0..36, 0, 0..1);
    ///
    /// // Draw with base vertex offset
    /// draws.draw_indexed(0..36, 100, 0..1);
    ///
    /// // Draw instanced cubes
    /// draws.draw_indexed(0..36, 0, 0..1000);
    /// ```
    ///
    /// # Notes
    ///
    /// - Requires an index buffer bound with `set_index_buffer`
    /// - `base_vertex` is added to each index before vertex lookup
    /// - Useful for drawing sub-meshes from a combined vertex buffer
    #[inline]
    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> &mut Self {
        self.pass.draw_indexed(indices, base_vertex, instances);
        self
    }

    // =========================================================================
    // Indirect Draw Commands (Criteria 3 & 4)
    // =========================================================================

    /// Draw non-indexed geometry with GPU-provided parameters.
    ///
    /// The draw parameters are read from an indirect buffer containing
    /// a [`DrawIndirectArgs`] struct.
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing draw arguments
    /// * `indirect_offset` - Byte offset into the buffer
    ///
    /// # Buffer Layout
    ///
    /// At `indirect_offset`, the buffer must contain:
    /// ```text
    /// struct DrawIndirectArgs {
    ///     vertex_count: u32,
    ///     instance_count: u32,
    ///     first_vertex: u32,
    ///     first_instance: u32,
    /// }
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GPU culling pass writes visible object counts
    /// draws.draw_indirect(&indirect_buffer, 0);
    ///
    /// // Multiple indirect draws at different offsets
    /// draws.draw_indirect(&indirect_buffer, 0);
    /// draws.draw_indirect(&indirect_buffer, DrawIndirectArgs::SIZE);
    /// draws.draw_indirect(&indirect_buffer, DrawIndirectArgs::SIZE * 2);
    /// ```
    #[inline]
    pub fn draw_indirect(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
    ) -> &mut Self {
        self.pass.draw_indirect(indirect_buffer, indirect_offset);
        self
    }

    /// Draw indexed geometry with GPU-provided parameters.
    ///
    /// The draw parameters are read from an indirect buffer containing
    /// a [`DrawIndexedIndirectArgs`] struct.
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing draw arguments
    /// * `indirect_offset` - Byte offset into the buffer
    ///
    /// # Buffer Layout
    ///
    /// At `indirect_offset`, the buffer must contain:
    /// ```text
    /// struct DrawIndexedIndirectArgs {
    ///     index_count: u32,
    ///     instance_count: u32,
    ///     first_index: u32,
    ///     base_vertex: i32,  // signed!
    ///     first_instance: u32,
    /// }
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GPU-driven indexed draw
    /// draws.draw_indexed_indirect(&indirect_buffer, 0);
    ///
    /// // Chain multiple indexed indirect draws
    /// for i in 0..mesh_count {
    ///     draws.draw_indexed_indirect(
    ///         &indirect_buffer,
    ///         i * DrawIndexedIndirectArgs::SIZE,
    ///     );
    /// }
    /// ```
    #[inline]
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
    ) -> &mut Self {
        self.pass.draw_indexed_indirect(indirect_buffer, indirect_offset);
        self
    }

    // =========================================================================
    // Multi-Draw Indirect Commands (Criteria 5 & 6)
    // =========================================================================

    /// Execute multiple non-indexed draws from an indirect buffer.
    ///
    /// **Requires `MULTI_DRAW_INDIRECT` feature.** Use
    /// [`supports_multi_draw_indirect`] to check availability.
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing array of draw arguments
    /// * `indirect_offset` - Byte offset to the first draw argument
    /// * `count` - Number of draws to execute
    ///
    /// # Buffer Layout
    ///
    /// Starting at `indirect_offset`, the buffer must contain `count`
    /// consecutive [`DrawIndirectArgs`] structs (16 bytes each).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Prepare 100 draw commands on GPU
    /// let commands: Vec<DrawIndirectArgs> = (0..100)
    ///     .map(|i| DrawIndirectArgs::new(get_vertex_count(i), 1, 0, i))
    ///     .collect();
    ///
    /// // Execute all 100 draws in a single API call
    /// draws.multi_draw_indirect(&command_buffer, 0, 100);
    /// ```
    ///
    /// # Performance
    ///
    /// This is significantly faster than 100 individual `draw_indirect` calls
    /// as it reduces CPU-GPU synchronization and driver overhead.
    #[inline]
    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) -> &mut Self {
        self.pass.multi_draw_indirect(indirect_buffer, indirect_offset, count);
        self
    }

    /// Execute multiple indexed draws from an indirect buffer.
    ///
    /// **Requires `MULTI_DRAW_INDIRECT` feature.** Use
    /// [`supports_multi_draw_indirect`] to check availability.
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing array of indexed draw arguments
    /// * `indirect_offset` - Byte offset to the first draw argument
    /// * `count` - Number of draws to execute
    ///
    /// # Buffer Layout
    ///
    /// Starting at `indirect_offset`, the buffer must contain `count`
    /// consecutive [`DrawIndexedIndirectArgs`] structs (20 bytes each).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GPU culling writes visible meshes to command buffer
    /// // Each mesh has its own index range and instance count
    ///
    /// // Execute all visible mesh draws
    /// draws.multi_draw_indexed_indirect(&command_buffer, 0, visible_count);
    /// ```
    #[inline]
    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) -> &mut Self {
        self.pass.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count);
        self
    }

    // =========================================================================
    // Multi-Draw Indirect Count Commands (Criterion 7)
    // =========================================================================

    /// Execute multiple non-indexed draws with GPU-controlled count.
    ///
    /// **Requires `MULTI_DRAW_INDIRECT_COUNT` feature.** Use
    /// [`supports_multi_draw_indirect_count`] to check availability.
    ///
    /// The draw count is read from a separate buffer, allowing the GPU
    /// to control how many draws to execute (up to `max_count`).
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing array of draw arguments
    /// * `indirect_offset` - Byte offset to the first draw argument
    /// * `count_buffer` - Buffer containing draw count (u32)
    /// * `count_offset` - Byte offset to the count value
    /// * `max_count` - Maximum number of draws (for validation)
    ///
    /// # Buffer Layouts
    ///
    /// **Indirect Buffer:** Array of [`DrawIndirectArgs`] (16 bytes each)
    /// **Count Buffer:** Single `u32` specifying number of draws
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GPU culling pass:
    /// // 1. Writes visible objects to indirect_buffer
    /// // 2. Writes visible count to count_buffer
    ///
    /// // Execute exactly as many draws as the GPU determined visible
    /// draws.multi_draw_indirect_count(
    ///     &indirect_buffer, 0,
    ///     &count_buffer, 0,
    ///     MAX_OBJECTS, // Safety limit
    /// );
    /// ```
    ///
    /// # Use Cases
    ///
    /// - GPU-driven visibility culling
    /// - LOD selection where count varies per frame
    /// - Streaming where available object count changes
    #[inline]
    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &'a wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) -> &mut Self {
        self.pass.multi_draw_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
        self
    }

    /// Execute multiple indexed draws with GPU-controlled count.
    ///
    /// **Requires `MULTI_DRAW_INDIRECT_COUNT` feature.** Use
    /// [`supports_multi_draw_indirect_count`] to check availability.
    ///
    /// This is the most powerful draw command, allowing the GPU to
    /// control both what to draw AND how many draws to execute.
    ///
    /// # Arguments
    ///
    /// * `indirect_buffer` - Buffer containing array of indexed draw arguments
    /// * `indirect_offset` - Byte offset to the first draw argument
    /// * `count_buffer` - Buffer containing draw count (u32)
    /// * `count_offset` - Byte offset to the count value
    /// * `max_count` - Maximum number of draws (for validation)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Complete GPU-driven rendering pipeline:
    /// //
    /// // 1. Compute pass: Frustum + occlusion cull all objects
    /// // 2. Compute pass: Compact visible objects into command buffer
    /// // 3. Compute pass: Write final visible count
    /// // 4. Render pass: Execute all visible draws
    ///
    /// draws.multi_draw_indexed_indirect_count(
    ///     &command_buffer, 0,
    ///     &count_buffer, 0,
    ///     scene.max_objects,
    /// );
    /// ```
    ///
    /// # Performance Benefits
    ///
    /// - Zero CPU readback for object counts
    /// - Single draw call for entire scene
    /// - GPU pipelining (no CPU-GPU sync)
    /// - Optimal batching by the driver
    #[inline]
    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &'a wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) -> &mut Self {
        self.pass.multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
        self
    }
}

impl fmt::Debug for DrawCommands<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DrawCommands")
            .field("pass", &"&mut wgpu::RenderPass<'a>")
            .finish()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate buffer size for a given number of draw commands.
///
/// # Type Parameter
///
/// * `T` - Either [`DrawIndirectArgs`] or [`DrawIndexedIndirectArgs`]
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::{
///     DrawIndirectArgs, DrawIndexedIndirectArgs, indirect_buffer_size,
/// };
///
/// // Size for 100 non-indexed draw commands
/// let size = indirect_buffer_size::<DrawIndirectArgs>(100);
/// assert_eq!(size, 1600); // 100 * 16 bytes
///
/// // Size for 100 indexed draw commands
/// let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(100);
/// assert_eq!(size, 2000); // 100 * 20 bytes
/// ```
#[inline]
pub const fn indirect_buffer_size<T>(count: u32) -> u64 {
    std::mem::size_of::<T>() as u64 * count as u64
}

/// Calculate the number of draw commands that fit in a buffer.
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::{
///     DrawIndexedIndirectArgs, indirect_command_count,
/// };
///
/// // How many indexed commands fit in 1000 bytes?
/// let count = indirect_command_count::<DrawIndexedIndirectArgs>(1000);
/// assert_eq!(count, 50); // 1000 / 20 = 50
/// ```
#[inline]
pub const fn indirect_command_count<T>(buffer_size: u64) -> u32 {
    (buffer_size / std::mem::size_of::<T>() as u64) as u32
}

/// Validate indirect buffer offset alignment.
///
/// Indirect offsets should be aligned to 4 bytes for optimal performance.
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::draw_commands::validate_indirect_offset;
///
/// assert!(validate_indirect_offset(0).is_ok());
/// assert!(validate_indirect_offset(16).is_ok());
/// assert!(validate_indirect_offset(20).is_ok());
/// assert!(validate_indirect_offset(3).is_err());
/// ```
#[inline]
pub const fn validate_indirect_offset(offset: u64) -> Result<(), &'static str> {
    if offset % 4 == 0 {
        Ok(())
    } else {
        Err("Indirect buffer offset must be 4-byte aligned")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DrawIndirectArgs Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
        assert_eq!(DrawIndirectArgs::SIZE, 16);
        assert_eq!(DrawIndirectArgs::STRIDE, 16);
    }

    #[test]
    fn test_draw_indirect_args_new() {
        let args = DrawIndirectArgs::new(100, 5, 10, 2);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_vertex, 10);
        assert_eq!(args.first_instance, 2);
    }

    #[test]
    fn test_draw_indirect_args_zeroed() {
        let args = DrawIndirectArgs::zeroed();
        assert_eq!(args.vertex_count, 0);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_draw_indirect_args_single() {
        let args = DrawIndirectArgs::single(36);
        assert_eq!(args.vertex_count, 36);
        assert_eq!(args.instance_count, 1);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indirect_args_from_ranges() {
        let args = DrawIndirectArgs::from_ranges(10..110, 5..15);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 10);
        assert_eq!(args.first_vertex, 10);
        assert_eq!(args.first_instance, 5);
    }

    #[test]
    fn test_draw_indirect_args_total_vertices() {
        let args = DrawIndirectArgs::new(100, 10, 0, 0);
        assert_eq!(args.total_vertices(), 1000);
    }

    #[test]
    fn test_draw_indirect_args_as_bytes() {
        let args = DrawIndirectArgs::new(1, 2, 3, 4);
        let bytes = args.as_bytes();
        assert_eq!(bytes.len(), 16);
        // Verify first u32 (vertex_count = 1)
        assert_eq!(bytes[0..4], 1u32.to_ne_bytes());
    }

    #[test]
    fn test_draw_indirect_args_debug() {
        let args = DrawIndirectArgs::new(100, 5, 0, 0);
        let debug = format!("{:?}", args);
        assert!(debug.contains("DrawIndirectArgs"));
        assert!(debug.contains("vertex_count: 100"));
    }

    #[test]
    fn test_draw_indirect_args_clone() {
        let args1 = DrawIndirectArgs::new(100, 5, 10, 2);
        let args2 = args1;
        assert_eq!(args1, args2);
    }

    #[test]
    fn test_draw_indirect_args_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DrawIndirectArgs::new(100, 5, 0, 0));
        set.insert(DrawIndirectArgs::new(100, 5, 0, 0)); // Duplicate
        set.insert(DrawIndirectArgs::new(200, 5, 0, 0));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_draw_indirect_args_default() {
        let args = DrawIndirectArgs::default();
        assert_eq!(args, DrawIndirectArgs::zeroed());
    }

    // -------------------------------------------------------------------------
    // DrawIndexedIndirectArgs Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indexed_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
        assert_eq!(DrawIndexedIndirectArgs::STRIDE, 20);
        assert_eq!(DrawIndexedIndirectArgs::STRIDE_PADDED, 24);
    }

    #[test]
    fn test_draw_indexed_indirect_args_new() {
        let args = DrawIndexedIndirectArgs::new(36, 10, 0, 100, 5);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 10);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 100);
        assert_eq!(args.first_instance, 5);
    }

    #[test]
    fn test_draw_indexed_indirect_args_negative_base_vertex() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -50, 0);
        assert_eq!(args.base_vertex, -50);
    }

    #[test]
    fn test_draw_indexed_indirect_args_zeroed() {
        let args = DrawIndexedIndirectArgs::zeroed();
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_single() {
        let args = DrawIndexedIndirectArgs::single(36);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 1);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_from_ranges() {
        let args = DrawIndexedIndirectArgs::from_ranges(0..36, 100, 0..10);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.base_vertex, 100);
        assert_eq!(args.instance_count, 10);
    }

    #[test]
    fn test_draw_indexed_indirect_args_total_indices() {
        let args = DrawIndexedIndirectArgs::new(36, 100, 0, 0, 0);
        assert_eq!(args.total_indices(), 3600);
    }

    #[test]
    fn test_draw_indexed_indirect_args_as_bytes() {
        let args = DrawIndexedIndirectArgs::new(1, 2, 3, 4, 5);
        let bytes = args.as_bytes();
        assert_eq!(bytes.len(), 20);
    }

    // -------------------------------------------------------------------------
    // MultiDrawTier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_draw_tier_none() {
        let tier = MultiDrawTier::None;
        assert!(!tier.supports_multi_draw());
        assert!(!tier.supports_multi_draw_count());
        assert_eq!(tier.description(), "No multi-draw support");
    }

    #[test]
    fn test_multi_draw_tier_basic() {
        let tier = MultiDrawTier::Basic;
        assert!(tier.supports_multi_draw());
        assert!(!tier.supports_multi_draw_count());
        assert_eq!(tier.description(), "Multi-draw indirect (CPU count)");
    }

    #[test]
    fn test_multi_draw_tier_full() {
        let tier = MultiDrawTier::Full;
        assert!(tier.supports_multi_draw());
        assert!(tier.supports_multi_draw_count());
        assert_eq!(tier.description(), "Multi-draw indirect count (GPU count)");
    }

    #[test]
    fn test_multi_draw_tier_ordering() {
        assert!(MultiDrawTier::None < MultiDrawTier::Basic);
        assert!(MultiDrawTier::Basic < MultiDrawTier::Full);
    }

    #[test]
    fn test_multi_draw_tier_display() {
        assert_eq!(format!("{}", MultiDrawTier::None), "No multi-draw support");
        assert_eq!(format!("{}", MultiDrawTier::Full), "Multi-draw indirect count (GPU count)");
    }

    #[test]
    fn test_multi_draw_tier_from_features_none() {
        let tier = MultiDrawTier::from_features(wgpu::Features::empty());
        assert_eq!(tier, MultiDrawTier::None);
    }

    #[test]
    fn test_multi_draw_tier_from_features_basic() {
        let tier = MultiDrawTier::from_features(wgpu::Features::MULTI_DRAW_INDIRECT);
        assert_eq!(tier, MultiDrawTier::Basic);
    }

    #[test]
    fn test_multi_draw_tier_from_features_full() {
        let features = wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        let tier = MultiDrawTier::from_features(features);
        assert_eq!(tier, MultiDrawTier::Full);
    }

    #[test]
    fn test_multi_draw_tier_from_features_count_only() {
        // Having COUNT implies having basic multi-draw
        let tier = MultiDrawTier::from_features(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        assert_eq!(tier, MultiDrawTier::Full);
    }

    // -------------------------------------------------------------------------
    // Feature Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_required_multi_draw_features() {
        let (multi_draw, multi_draw_count) = required_multi_draw_features();
        assert_eq!(multi_draw, wgpu::Features::MULTI_DRAW_INDIRECT);
        assert_eq!(multi_draw_count, wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_buffer_size_draw() {
        let size = indirect_buffer_size::<DrawIndirectArgs>(100);
        assert_eq!(size, 1600); // 100 * 16
    }

    #[test]
    fn test_indirect_buffer_size_indexed() {
        let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(100);
        assert_eq!(size, 2000); // 100 * 20
    }

    #[test]
    fn test_indirect_buffer_size_zero() {
        let size = indirect_buffer_size::<DrawIndirectArgs>(0);
        assert_eq!(size, 0);
    }

    #[test]
    fn test_indirect_command_count_draw() {
        let count = indirect_command_count::<DrawIndirectArgs>(1600);
        assert_eq!(count, 100);
    }

    #[test]
    fn test_indirect_command_count_indexed() {
        let count = indirect_command_count::<DrawIndexedIndirectArgs>(2000);
        assert_eq!(count, 100);
    }

    #[test]
    fn test_indirect_command_count_partial() {
        // 1000 / 20 = 50 commands (no partial)
        let count = indirect_command_count::<DrawIndexedIndirectArgs>(1000);
        assert_eq!(count, 50);

        // 1010 / 20 = 50.5 -> 50 commands (truncated)
        let count = indirect_command_count::<DrawIndexedIndirectArgs>(1010);
        assert_eq!(count, 50);
    }

    #[test]
    fn test_validate_indirect_offset() {
        assert!(validate_indirect_offset(0).is_ok());
        assert!(validate_indirect_offset(4).is_ok());
        assert!(validate_indirect_offset(16).is_ok());
        assert!(validate_indirect_offset(20).is_ok());
        assert!(validate_indirect_offset(100).is_ok());

        assert!(validate_indirect_offset(1).is_err());
        assert!(validate_indirect_offset(2).is_err());
        assert!(validate_indirect_offset(3).is_err());
        assert!(validate_indirect_offset(5).is_err());
    }

    // -------------------------------------------------------------------------
    // DrawCommands Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_commands_debug_impl() {
        // Verify Debug is implemented (compile-time check)
        fn assert_debug<T: fmt::Debug>() {}
        assert_debug::<DrawCommands<'_>>();
    }

    // -------------------------------------------------------------------------
    // Bytemuck Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_bytemuck_pod() {
        let args = DrawIndirectArgs::new(1, 2, 3, 4);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);

        let recovered: DrawIndirectArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(recovered, args);
    }

    #[test]
    fn test_draw_indexed_indirect_args_bytemuck_pod() {
        let args = DrawIndexedIndirectArgs::new(1, 2, 3, -4, 5);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);

        let recovered: DrawIndexedIndirectArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(recovered, args);
    }

    #[test]
    fn test_draw_indirect_args_slice_cast() {
        let args = [
            DrawIndirectArgs::new(1, 1, 0, 0),
            DrawIndirectArgs::new(2, 2, 0, 0),
            DrawIndirectArgs::new(3, 3, 0, 0),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&args);
        assert_eq!(bytes.len(), 48); // 3 * 16

        let recovered: &[DrawIndirectArgs] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered, &args);
    }

    // -------------------------------------------------------------------------
    // Range Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_from_ranges_empty() {
        let args = DrawIndirectArgs::from_ranges(0..0, 0..0);
        assert_eq!(args.vertex_count, 0);
        assert_eq!(args.instance_count, 0);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_from_ranges_empty() {
        let args = DrawIndexedIndirectArgs::from_ranges(0..0, 0, 0..0);
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_ranges_large_values() {
        let args = DrawIndirectArgs::from_ranges(0..u32::MAX, 0..1000);
        assert_eq!(args.vertex_count, u32::MAX);
        assert_eq!(args.instance_count, 1000);
    }

    // -------------------------------------------------------------------------
    // Total Vertex/Index Count Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_total_vertices_overflow_safe() {
        // Even with max values, we use u64 so no overflow
        let args = DrawIndirectArgs::new(u32::MAX, u32::MAX, 0, 0);
        let total = args.total_vertices();
        assert_eq!(total, u32::MAX as u64 * u32::MAX as u64);
    }

    #[test]
    fn test_total_indices_overflow_safe() {
        let args = DrawIndexedIndirectArgs::new(u32::MAX, u32::MAX, 0, 0, 0);
        let total = args.total_indices();
        assert_eq!(total, u32::MAX as u64 * u32::MAX as u64);
    }

    // -------------------------------------------------------------------------
    // Base Vertex Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_base_vertex_min_max() {
        let args_min = DrawIndexedIndirectArgs::new(36, 1, 0, i32::MIN, 0);
        let args_max = DrawIndexedIndirectArgs::new(36, 1, 0, i32::MAX, 0);
        assert_eq!(args_min.base_vertex, i32::MIN);
        assert_eq!(args_max.base_vertex, i32::MAX);
    }

    // -------------------------------------------------------------------------
    // Constant Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_const_draw_indirect_args() {
        const ARGS: DrawIndirectArgs = DrawIndirectArgs::new(3, 1, 0, 0);
        const ZEROED: DrawIndirectArgs = DrawIndirectArgs::zeroed();
        const SINGLE: DrawIndirectArgs = DrawIndirectArgs::single(6);

        assert_eq!(ARGS.vertex_count, 3);
        assert_eq!(ZEROED.vertex_count, 0);
        assert_eq!(SINGLE.vertex_count, 6);
    }

    #[test]
    fn test_const_draw_indexed_indirect_args() {
        const ARGS: DrawIndexedIndirectArgs = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        const ZEROED: DrawIndexedIndirectArgs = DrawIndexedIndirectArgs::zeroed();
        const SINGLE: DrawIndexedIndirectArgs = DrawIndexedIndirectArgs::single(24);

        assert_eq!(ARGS.index_count, 36);
        assert_eq!(ZEROED.index_count, 0);
        assert_eq!(SINGLE.index_count, 24);
    }

    #[test]
    fn test_const_buffer_size() {
        const SIZE_DRAW: u64 = indirect_buffer_size::<DrawIndirectArgs>(100);
        const SIZE_INDEXED: u64 = indirect_buffer_size::<DrawIndexedIndirectArgs>(100);

        assert_eq!(SIZE_DRAW, 1600);
        assert_eq!(SIZE_INDEXED, 2000);
    }

    #[test]
    fn test_const_command_count() {
        const COUNT_DRAW: u32 = indirect_command_count::<DrawIndirectArgs>(1600);
        const COUNT_INDEXED: u32 = indirect_command_count::<DrawIndexedIndirectArgs>(2000);

        assert_eq!(COUNT_DRAW, 100);
        assert_eq!(COUNT_INDEXED, 100);
    }

    #[test]
    fn test_const_validate_offset() {
        const VALID: Result<(), &str> = validate_indirect_offset(16);
        const INVALID: Result<(), &str> = validate_indirect_offset(3);

        assert!(VALID.is_ok());
        assert!(INVALID.is_err());
    }

    // -------------------------------------------------------------------------
    // Trait Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_traits() {
        fn assert_traits<T: Copy + Clone + fmt::Debug + Default + PartialEq + Eq + std::hash::Hash>() {}
        assert_traits::<DrawIndirectArgs>();
    }

    #[test]
    fn test_draw_indexed_indirect_args_traits() {
        fn assert_traits<T: Copy + Clone + fmt::Debug + Default + PartialEq + Eq + std::hash::Hash>() {}
        assert_traits::<DrawIndexedIndirectArgs>();
    }

    #[test]
    fn test_multi_draw_tier_traits() {
        fn assert_traits<T: fmt::Debug + Clone + Copy + PartialEq + Eq + PartialOrd + Ord + std::hash::Hash + fmt::Display>() {}
        assert_traits::<MultiDrawTier>();
    }

    // -------------------------------------------------------------------------
    // Memory Layout Verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_repr_c_layout() {
        // Verify field offsets match expected C layout
        let args = DrawIndirectArgs::new(0x11111111, 0x22222222, 0x33333333, 0x44444444);
        let bytes = args.as_bytes();

        // vertex_count at offset 0
        assert_eq!(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 0x11111111);
        // instance_count at offset 4
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 0x22222222);
        // first_vertex at offset 8
        assert_eq!(u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0x33333333);
        // first_instance at offset 12
        assert_eq!(u32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0x44444444);
    }

    #[test]
    fn test_draw_indexed_indirect_args_repr_c_layout() {
        let args = DrawIndexedIndirectArgs::new(0x11111111, 0x22222222, 0x33333333, 0x44444444u32 as i32, 0x55555555);
        let bytes = args.as_bytes();

        // index_count at offset 0
        assert_eq!(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 0x11111111);
        // instance_count at offset 4
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 0x22222222);
        // first_index at offset 8
        assert_eq!(u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0x33333333);
        // base_vertex at offset 12 (signed)
        assert_eq!(i32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0x44444444u32 as i32);
        // first_instance at offset 16
        assert_eq!(u32::from_ne_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 0x55555555);
    }

    // -------------------------------------------------------------------------
    // Additional DrawIndirectArgs Tests (Edge Cases)
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_zero_vertices_positive_instances() {
        let args = DrawIndirectArgs::new(0, 100, 0, 0);
        assert_eq!(args.vertex_count, 0);
        assert_eq!(args.instance_count, 100);
        assert!(!args.will_draw()); // Zero vertices = no draw
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn test_draw_indirect_args_positive_vertices_zero_instances() {
        let args = DrawIndirectArgs::new(100, 0, 0, 0);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 0);
        assert!(!args.will_draw()); // Zero instances = no draw
        assert_eq!(args.total_vertices(), 0);
    }

    #[test]
    fn test_draw_indirect_args_max_vertices_single_instance() {
        let args = DrawIndirectArgs::new(u32::MAX, 1, 0, 0);
        assert!(args.will_draw());
        assert_eq!(args.total_vertices(), u32::MAX as u64);
    }

    #[test]
    fn test_draw_indirect_args_max_first_vertex() {
        let args = DrawIndirectArgs::new(1, 1, u32::MAX, 0);
        assert_eq!(args.first_vertex, u32::MAX);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indirect_args_max_first_instance() {
        let args = DrawIndirectArgs::new(1, 1, 0, u32::MAX);
        assert_eq!(args.first_instance, u32::MAX);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indirect_args_from_ranges_inverted() {
        // Inverted range: end < start should saturate to 0
        let args = DrawIndirectArgs::from_ranges(100..50, 10..5);
        assert_eq!(args.vertex_count, 0); // saturating_sub
        assert_eq!(args.instance_count, 0); // saturating_sub
        assert_eq!(args.first_vertex, 100);
        assert_eq!(args.first_instance, 10);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_draw_indirect_args_alignment() {
        // Verify alignment is 4 bytes (u32 alignment)
        assert_eq!(std::mem::align_of::<DrawIndirectArgs>(), 4);
    }

    #[test]
    fn test_draw_indirect_args_equality() {
        let args1 = DrawIndirectArgs::new(100, 5, 10, 2);
        let args2 = DrawIndirectArgs::new(100, 5, 10, 2);
        let args3 = DrawIndirectArgs::new(100, 5, 10, 3); // Different first_instance

        assert_eq!(args1, args2);
        assert_ne!(args1, args3);
    }

    // -------------------------------------------------------------------------
    // Additional DrawIndexedIndirectArgs Tests (Edge Cases)
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indexed_indirect_args_zero_indices_positive_instances() {
        let args = DrawIndexedIndirectArgs::new(0, 100, 0, 0, 0);
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 100);
        assert!(!args.will_draw()); // Zero indices = no draw
        assert_eq!(args.total_indices(), 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_positive_indices_zero_instances() {
        let args = DrawIndexedIndirectArgs::new(100, 0, 0, 0, 0);
        assert_eq!(args.index_count, 100);
        assert_eq!(args.instance_count, 0);
        assert!(!args.will_draw()); // Zero instances = no draw
        assert_eq!(args.total_indices(), 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_max_index_count() {
        let args = DrawIndexedIndirectArgs::new(u32::MAX, 1, 0, 0, 0);
        assert!(args.will_draw());
        assert_eq!(args.total_indices(), u32::MAX as u64);
    }

    #[test]
    fn test_draw_indexed_indirect_args_max_first_index() {
        let args = DrawIndexedIndirectArgs::new(1, 1, u32::MAX, 0, 0);
        assert_eq!(args.first_index, u32::MAX);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_max_first_instance() {
        let args = DrawIndexedIndirectArgs::new(1, 1, 0, 0, u32::MAX);
        assert_eq!(args.first_instance, u32::MAX);
        assert!(args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_from_ranges_inverted() {
        // Inverted range: end < start should saturate to 0
        let args = DrawIndexedIndirectArgs::from_ranges(100..50, 0, 10..5);
        assert_eq!(args.index_count, 0); // saturating_sub
        assert_eq!(args.instance_count, 0); // saturating_sub
        assert_eq!(args.first_index, 100);
        assert_eq!(args.first_instance, 10);
        assert!(!args.will_draw());
    }

    #[test]
    fn test_draw_indexed_indirect_args_alignment() {
        // Verify alignment is 4 bytes (u32/i32 alignment)
        assert_eq!(std::mem::align_of::<DrawIndexedIndirectArgs>(), 4);
    }

    #[test]
    fn test_draw_indexed_indirect_args_debug() {
        let args = DrawIndexedIndirectArgs::new(36, 10, 5, -50, 2);
        let debug = format!("{:?}", args);
        assert!(debug.contains("DrawIndexedIndirectArgs"));
        assert!(debug.contains("index_count: 36"));
        assert!(debug.contains("base_vertex: -50"));
    }

    #[test]
    fn test_draw_indexed_indirect_args_clone() {
        let args1 = DrawIndexedIndirectArgs::new(36, 10, 0, -100, 5);
        let args2 = args1;
        assert_eq!(args1, args2);
    }

    #[test]
    fn test_draw_indexed_indirect_args_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0));
        set.insert(DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0)); // Duplicate
        set.insert(DrawIndexedIndirectArgs::new(36, 1, 0, -1, 0)); // Different base_vertex
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_draw_indexed_indirect_args_default() {
        let args = DrawIndexedIndirectArgs::default();
        assert_eq!(args, DrawIndexedIndirectArgs::zeroed());
    }

    #[test]
    fn test_draw_indexed_indirect_args_equality() {
        let args1 = DrawIndexedIndirectArgs::new(36, 10, 0, 100, 5);
        let args2 = DrawIndexedIndirectArgs::new(36, 10, 0, 100, 5);
        let args3 = DrawIndexedIndirectArgs::new(36, 10, 0, -100, 5); // Different base_vertex

        assert_eq!(args1, args2);
        assert_ne!(args1, args3);
    }

    // -------------------------------------------------------------------------
    // Additional MultiDrawTier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_draw_tier_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MultiDrawTier::None);
        set.insert(MultiDrawTier::None); // Duplicate
        set.insert(MultiDrawTier::Basic);
        set.insert(MultiDrawTier::Full);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_multi_draw_tier_clone() {
        let tier1 = MultiDrawTier::Full;
        let tier2 = tier1;
        assert_eq!(tier1, tier2);
    }

    #[test]
    fn test_multi_draw_tier_debug() {
        let tier = MultiDrawTier::Basic;
        let debug = format!("{:?}", tier);
        assert!(debug.contains("Basic"));
    }

    #[test]
    fn test_multi_draw_tier_partial_ord() {
        assert!(MultiDrawTier::None < MultiDrawTier::Basic);
        assert!(MultiDrawTier::Basic < MultiDrawTier::Full);
        assert!(MultiDrawTier::None <= MultiDrawTier::None);
        assert!(MultiDrawTier::Full >= MultiDrawTier::Basic);
    }

    // -------------------------------------------------------------------------
    // Additional Buffer Offset Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_indirect_offset_large_aligned() {
        // Large aligned offsets should be valid
        assert!(validate_indirect_offset(1_000_000).is_ok());
        assert!(validate_indirect_offset(u64::MAX - 3).is_ok()); // u64::MAX - 3 is divisible by 4
    }

    #[test]
    fn test_validate_indirect_offset_large_unaligned() {
        assert!(validate_indirect_offset(1_000_001).is_err());
        assert!(validate_indirect_offset(1_000_002).is_err());
        assert!(validate_indirect_offset(1_000_003).is_err());
    }

    #[test]
    fn test_validate_indirect_offset_at_struct_boundaries() {
        // DrawIndirectArgs boundaries (16 bytes)
        assert!(validate_indirect_offset(0).is_ok());
        assert!(validate_indirect_offset(16).is_ok());
        assert!(validate_indirect_offset(32).is_ok());

        // DrawIndexedIndirectArgs boundaries (20 bytes)
        assert!(validate_indirect_offset(20).is_ok());
        assert!(validate_indirect_offset(40).is_ok());
    }

    // -------------------------------------------------------------------------
    // Additional Buffer Size Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_buffer_size_max_count() {
        // Test with large count values
        let size = indirect_buffer_size::<DrawIndirectArgs>(1_000_000);
        assert_eq!(size, 16_000_000); // 1M * 16 bytes

        let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(1_000_000);
        assert_eq!(size, 20_000_000); // 1M * 20 bytes
    }

    #[test]
    fn test_indirect_command_count_zero_buffer() {
        let count = indirect_command_count::<DrawIndirectArgs>(0);
        assert_eq!(count, 0);

        let count = indirect_command_count::<DrawIndexedIndirectArgs>(0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_indirect_command_count_undersized_buffer() {
        // Buffer smaller than one command
        let count = indirect_command_count::<DrawIndirectArgs>(15);
        assert_eq!(count, 0); // Can't fit a 16-byte command

        let count = indirect_command_count::<DrawIndexedIndirectArgs>(19);
        assert_eq!(count, 0); // Can't fit a 20-byte command
    }

    #[test]
    fn test_indirect_command_count_exact_fit() {
        // Buffer that fits exactly N commands
        let count = indirect_command_count::<DrawIndirectArgs>(32);
        assert_eq!(count, 2); // 32 / 16 = 2

        let count = indirect_command_count::<DrawIndexedIndirectArgs>(60);
        assert_eq!(count, 3); // 60 / 20 = 3
    }

    // -------------------------------------------------------------------------
    // Bytemuck Zeroable Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_zeroable() {
        let zeroed: DrawIndirectArgs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.vertex_count, 0);
        assert_eq!(zeroed.instance_count, 0);
        assert_eq!(zeroed.first_vertex, 0);
        assert_eq!(zeroed.first_instance, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_zeroable() {
        let zeroed: DrawIndexedIndirectArgs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.index_count, 0);
        assert_eq!(zeroed.instance_count, 0);
        assert_eq!(zeroed.first_index, 0);
        assert_eq!(zeroed.base_vertex, 0);
        assert_eq!(zeroed.first_instance, 0);
    }

    // -------------------------------------------------------------------------
    // Feature Flags Combination Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_draw_tier_from_features_with_other_flags() {
        // Multi-draw features combined with other unrelated features
        let features = wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::DEPTH_CLIP_CONTROL
            | wgpu::Features::TEXTURE_COMPRESSION_BC;
        let tier = MultiDrawTier::from_features(features);
        assert_eq!(tier, MultiDrawTier::Basic);

        let features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::DEPTH_CLIP_CONTROL;
        let tier = MultiDrawTier::from_features(features);
        assert_eq!(tier, MultiDrawTier::Full);
    }

    // -------------------------------------------------------------------------
    // Stride Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_stride_equals_size() {
        // For tightly packed arrays, stride equals size
        assert_eq!(DrawIndirectArgs::STRIDE, DrawIndirectArgs::SIZE);
        assert_eq!(DrawIndirectArgs::STRIDE, 16);
    }

    #[test]
    fn test_draw_indexed_indirect_args_stride_constants() {
        assert_eq!(DrawIndexedIndirectArgs::STRIDE, 20);
        assert_eq!(DrawIndexedIndirectArgs::STRIDE_PADDED, 24);
        assert!(DrawIndexedIndirectArgs::STRIDE_PADDED > DrawIndexedIndirectArgs::STRIDE);
    }

    // -------------------------------------------------------------------------
    // will_draw Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_will_draw_single_vertex_single_instance() {
        let args = DrawIndirectArgs::new(1, 1, 0, 0);
        assert!(args.will_draw());
    }

    #[test]
    fn test_will_draw_indexed_single() {
        let args = DrawIndexedIndirectArgs::new(1, 1, 0, 0, 0);
        assert!(args.will_draw());
    }

    #[test]
    fn test_will_draw_max_values() {
        let args = DrawIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        assert!(args.will_draw());

        let args = DrawIndexedIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX, i32::MAX, u32::MAX);
        assert!(args.will_draw());
    }
}
