//! Render pass descriptor and builder for wgpu 25.x render pass creation.
//!
//! This module provides high-level abstractions for creating render passes with
//! color attachments, depth/stencil attachments, timestamp writes, and occlusion
//! query sets.
//!
//! # Overview
//!
//! A render pass defines the targets (attachments) that rendering operations write to,
//! along with how those targets should be loaded at pass start and stored at pass end.
//!
//! # Architecture
//!
//! ```text
//! RenderPassDescriptor
//!     |-- label: Option<&str>
//!     |-- color_attachments: Vec<Option<ColorAttachment>>
//!     |-- depth_stencil_attachment: Option<DepthStencilAttachment>
//!     |-- timestamp_writes: Option<TimestampWrites>
//!     `-- occlusion_query_set: Option<OcclusionQuerySet>
//!
//! ColorAttachment
//!     |-- view: &TextureView
//!     |-- resolve_target: Option<&TextureView>
//!     `-- ops: Operations<Color>
//!
//! DepthStencilAttachment
//!     |-- view: &TextureView
//!     |-- depth_ops: Option<Operations<f32>>
//!     `-- stencil_ops: Option<Operations<u32>>
//! ```
//!
//! # Load/Store Operations
//!
//! | LoadOp | Effect | Use Case |
//! |--------|--------|----------|
//! | `Clear(value)` | Fill with clear value | First pass, shadow maps |
//! | `Load` | Preserve existing content | Multi-pass rendering |
//!
//! | StoreOp | Effect | Use Case |
//! |---------|--------|----------|
//! | `Store` | Write results to memory | Final output, MSAA resolve |
//! | `Discard` | Don't write results | Transient attachments |
//!
//! # Multiple Render Targets (MRT)
//!
//! wgpu supports up to 8 simultaneous color attachments:
//!
//! ```text
//! color_attachments: [
//!     Some(ColorAttachment { ... }),  // Location 0: Albedo
//!     Some(ColorAttachment { ... }),  // Location 1: Normal
//!     Some(ColorAttachment { ... }),  // Location 2: Material
//!     None,                           // Location 3: Unused
//!     ...
//! ]
//! ```
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub struct RenderPassDescriptor<'tex> {
//!     pub label: Label<'tex>,
//!     pub color_attachments: &'tex [Option<RenderPassColorAttachment<'tex>>],
//!     pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'tex>>,
//!     pub timestamp_writes: Option<RenderPassTimestampWrites<'tex>>,
//!     pub occlusion_query_set: Option<&'tex QuerySet>,
//! }
//!
//! pub struct RenderPassColorAttachment<'tex> {
//!     pub view: &'tex TextureView,
//!     pub resolve_target: Option<&'tex TextureView>,
//!     pub ops: Operations<Color>,
//! }
//!
//! pub struct RenderPassDepthStencilAttachment<'tex> {
//!     pub view: &'tex TextureView,
//!     pub depth_ops: Option<Operations<f32>>,
//!     pub stencil_ops: Option<Operations<u32>>,
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::render_pass::{
//!     RenderPassBuilder, RenderPassDescriptor, ColorAttachment,
//!     DepthStencilAttachment, Operations, LoadOp, StoreOp,
//! };
//!
//! // Simple color pass
//! let pass = RenderPassBuilder::new()
//!     .label("forward_pass")
//!     .color_attachment(&color_view, Operations::clear(wgpu::Color::BLACK))
//!     .depth_stencil(&depth_view, Operations::clear(1.0), None)
//!     .build();
//!
//! // Use presets
//! let shadow_pass = RenderPassBuilder::shadow_map(&depth_view);
//! let simple_pass = RenderPassBuilder::simple_color(&color_view);
//! let gbuffer_pass = RenderPassBuilder::color_depth(&color_view, &depth_view);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of color attachments supported by wgpu.
///
/// Most GPUs support 8 simultaneous render targets.
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// Default clear color (transparent black).
pub const DEFAULT_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// Default clear depth (1.0 for standard depth, 0.0 for reverse-Z).
pub const DEFAULT_CLEAR_DEPTH: f32 = 1.0;

/// Default clear stencil value.
pub const DEFAULT_CLEAR_STENCIL: u32 = 0;

// ---------------------------------------------------------------------------
// LoadOp wrapper
// ---------------------------------------------------------------------------

/// Load operation for an attachment at pass start.
///
/// Specifies what happens to the attachment contents when the render pass begins.
///
/// # Operations
///
/// | Variant | Effect | Performance |
/// |---------|--------|-------------|
/// | `Clear(value)` | Fill with clear value | Fast (may skip read) |
/// | `Load` | Preserve existing content | Requires memory read |
///
/// # Note
///
/// wgpu also has an internal `DontCare` which behaves like `Clear` but with
/// undefined initial values. For safety, this abstraction requires explicit
/// clear values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<V> {
    /// Clear the attachment to a specific value.
    Clear(V),
    /// Load existing content from memory.
    Load,
}

impl<V: Default> Default for LoadOp<V> {
    fn default() -> Self {
        Self::Clear(V::default())
    }
}

impl<V: fmt::Debug> fmt::Display for LoadOp<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clear(v) => write!(f, "Clear({:?})", v),
            Self::Load => write!(f, "Load"),
        }
    }
}

impl LoadOp<wgpu::Color> {
    /// Convert to wgpu LoadOp<Color>.
    pub fn to_wgpu(self) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            Self::Clear(color) => wgpu::LoadOp::Clear(color),
            Self::Load => wgpu::LoadOp::Load,
        }
    }
}

impl LoadOp<f32> {
    /// Convert to wgpu LoadOp<f32> for depth.
    pub fn to_wgpu(self) -> wgpu::LoadOp<f32> {
        match self {
            Self::Clear(depth) => wgpu::LoadOp::Clear(depth),
            Self::Load => wgpu::LoadOp::Load,
        }
    }
}

impl LoadOp<u32> {
    /// Convert to wgpu LoadOp<u32> for stencil.
    pub fn to_wgpu(self) -> wgpu::LoadOp<u32> {
        match self {
            Self::Clear(stencil) => wgpu::LoadOp::Clear(stencil),
            Self::Load => wgpu::LoadOp::Load,
        }
    }
}

// ---------------------------------------------------------------------------
// StoreOp wrapper
// ---------------------------------------------------------------------------

/// Store operation for an attachment at pass end.
///
/// Specifies what happens to the attachment contents when the render pass ends.
///
/// # Operations
///
/// | Variant | Effect | Use Case |
/// |---------|--------|----------|
/// | `Store` | Write results to memory | Final output |
/// | `Discard` | Don't write results | Transient data |
///
/// # Discarding
///
/// Use `Discard` for attachments that are only needed within the pass, such as:
/// - MSAA samples (only the resolved texture matters)
/// - Depth buffers used only for hidden surface removal
/// - Stencil buffers used only for masking within the pass
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StoreOp {
    /// Store the results to memory.
    #[default]
    Store,
    /// Discard the results (don't write to memory).
    Discard,
}

impl fmt::Display for StoreOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store => write!(f, "Store"),
            Self::Discard => write!(f, "Discard"),
        }
    }
}

impl StoreOp {
    /// Convert to wgpu StoreOp.
    pub fn to_wgpu(self) -> wgpu::StoreOp {
        match self {
            Self::Store => wgpu::StoreOp::Store,
            Self::Discard => wgpu::StoreOp::Discard,
        }
    }
}

impl From<StoreOp> for wgpu::StoreOp {
    fn from(op: StoreOp) -> Self {
        op.to_wgpu()
    }
}

// ---------------------------------------------------------------------------
// Operations<V>
// ---------------------------------------------------------------------------

/// Combined load and store operations for an attachment.
///
/// # Type Parameters
///
/// - `V`: The value type for clear operations
///   - `wgpu::Color` for color attachments
///   - `f32` for depth attachments
///   - `u32` for stencil attachments
///
/// # Example
///
/// ```ignore
/// // Clear to black, store result
/// let ops = Operations::clear(wgpu::Color::BLACK);
///
/// // Load existing, store result
/// let ops = Operations::load_store();
///
/// // Clear to value, discard result (transient)
/// let ops = Operations::clear_discard(1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operations<V> {
    /// Load operation at pass start.
    pub load: LoadOp<V>,
    /// Store operation at pass end.
    pub store: StoreOp,
}

impl<V: Default> Default for Operations<V> {
    fn default() -> Self {
        Self {
            load: LoadOp::default(),
            store: StoreOp::Store,
        }
    }
}

impl<V> Operations<V> {
    /// Create operations with explicit load and store ops.
    pub fn new(load: LoadOp<V>, store: StoreOp) -> Self {
        Self { load, store }
    }

    /// Create operations that clear to a value and store the result.
    pub fn clear(value: V) -> Self {
        Self {
            load: LoadOp::Clear(value),
            store: StoreOp::Store,
        }
    }

    /// Create operations that clear to a value and discard the result.
    pub fn clear_discard(value: V) -> Self {
        Self {
            load: LoadOp::Clear(value),
            store: StoreOp::Discard,
        }
    }

    /// Create operations that load existing content and store the result.
    pub fn load_store() -> Self
    where
        V: Default,
    {
        Self {
            load: LoadOp::Load,
            store: StoreOp::Store,
        }
    }

    /// Create operations that load existing content and discard the result.
    pub fn load_discard() -> Self
    where
        V: Default,
    {
        Self {
            load: LoadOp::Load,
            store: StoreOp::Discard,
        }
    }
}

impl<V: fmt::Debug> fmt::Display for Operations<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operations(load={}, store={})", self.load, self.store)
    }
}

impl Operations<wgpu::Color> {
    /// Convert to wgpu Operations<Color>.
    pub fn to_wgpu(self) -> wgpu::Operations<wgpu::Color> {
        wgpu::Operations {
            load: self.load.to_wgpu(),
            store: self.store.to_wgpu(),
        }
    }

    /// Clear to transparent black, store result.
    pub fn clear_black() -> Self {
        Self::clear(wgpu::Color::BLACK)
    }

    /// Clear to opaque black, store result.
    pub fn clear_opaque_black() -> Self {
        Self::clear(wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        })
    }

    /// Clear to white, store result.
    pub fn clear_white() -> Self {
        Self::clear(wgpu::Color::WHITE)
    }

    /// Clear to a custom RGB color (alpha = 1.0).
    pub fn clear_rgb(r: f64, g: f64, b: f64) -> Self {
        Self::clear(wgpu::Color { r, g, b, a: 1.0 })
    }

    /// Clear to a custom RGBA color.
    pub fn clear_rgba(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self::clear(wgpu::Color { r, g, b, a })
    }
}

impl Operations<f32> {
    /// Convert to wgpu Operations<f32> for depth.
    pub fn to_wgpu(self) -> wgpu::Operations<f32> {
        wgpu::Operations {
            load: self.load.to_wgpu(),
            store: self.store.to_wgpu(),
        }
    }

    /// Clear depth to 1.0 (standard depth), store result.
    pub fn clear_depth() -> Self {
        Self::clear(1.0)
    }

    /// Clear depth to 0.0 (reverse-Z), store result.
    pub fn clear_depth_reverse_z() -> Self {
        Self::clear(0.0)
    }

    /// Clear depth, discard result (transient).
    pub fn clear_depth_transient() -> Self {
        Self::clear_discard(1.0)
    }
}

impl Operations<u32> {
    /// Convert to wgpu Operations<u32> for stencil.
    pub fn to_wgpu(self) -> wgpu::Operations<u32> {
        wgpu::Operations {
            load: self.load.to_wgpu(),
            store: self.store.to_wgpu(),
        }
    }

    /// Clear stencil to 0, store result.
    pub fn clear_stencil() -> Self {
        Self::clear(0)
    }

    /// Clear stencil, discard result (transient).
    pub fn clear_stencil_transient() -> Self {
        Self::clear_discard(0)
    }
}

// ---------------------------------------------------------------------------
// ColorAttachment
// ---------------------------------------------------------------------------

/// A color attachment for a render pass.
///
/// Describes a color render target with optional MSAA resolve target and
/// load/store operations.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `view` | `TextureView` | The render target view |
/// | `resolve_target` | `Option<TextureView>` | MSAA resolve destination |
/// | `ops` | `Operations<Color>` | Load and store operations |
///
/// # MSAA Resolve
///
/// When using MSAA, the `view` is the multisampled texture and `resolve_target`
/// is the non-multisampled texture that receives the resolved result.
///
/// # Example
///
/// ```ignore
/// // Simple color attachment
/// let attachment = ColorAttachment::new(&color_view)
///     .clear(wgpu::Color::BLACK);
///
/// // MSAA with resolve
/// let attachment = ColorAttachment::new(&msaa_view)
///     .resolve(&resolve_view)
///     .clear(wgpu::Color::BLACK);
/// ```
#[derive(Debug, Clone)]
pub struct ColorAttachment {
    /// Clear color for LoadOp::Clear.
    pub clear_color: wgpu::Color,
    /// Load operation.
    pub load_op: LoadOp<wgpu::Color>,
    /// Store operation.
    pub store_op: StoreOp,
    /// Whether this attachment has a resolve target.
    pub has_resolve: bool,
}

impl Default for ColorAttachment {
    fn default() -> Self {
        Self {
            clear_color: DEFAULT_CLEAR_COLOR,
            load_op: LoadOp::Clear(DEFAULT_CLEAR_COLOR),
            store_op: StoreOp::Store,
            has_resolve: false,
        }
    }
}

impl ColorAttachment {
    /// Create a new color attachment with default operations (clear black, store).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a color attachment that clears to the specified color.
    pub fn with_clear(color: wgpu::Color) -> Self {
        Self {
            clear_color: color,
            load_op: LoadOp::Clear(color),
            store_op: StoreOp::Store,
            has_resolve: false,
        }
    }

    /// Create a color attachment that loads existing content.
    pub fn with_load() -> Self {
        Self {
            clear_color: DEFAULT_CLEAR_COLOR,
            load_op: LoadOp::Load,
            store_op: StoreOp::Store,
            has_resolve: false,
        }
    }

    /// Set the clear color.
    pub fn clear_color(mut self, color: wgpu::Color) -> Self {
        self.clear_color = color;
        self.load_op = LoadOp::Clear(color);
        self
    }

    /// Set to load existing content.
    pub fn load(mut self) -> Self {
        self.load_op = LoadOp::Load;
        self
    }

    /// Set the store operation.
    pub fn store(mut self, store: StoreOp) -> Self {
        self.store_op = store;
        self
    }

    /// Mark this attachment as having a resolve target.
    pub fn with_resolve(mut self) -> Self {
        self.has_resolve = true;
        self
    }

    /// Get the operations for this attachment.
    pub fn operations(&self) -> Operations<wgpu::Color> {
        Operations {
            load: self.load_op,
            store: self.store_op,
        }
    }

    /// Build the wgpu RenderPassColorAttachment.
    ///
    /// # Arguments
    ///
    /// * `view` - The texture view for this attachment
    /// * `resolve_target` - Optional resolve target for MSAA
    pub fn build_wgpu<'a>(
        &self,
        view: &'a wgpu::TextureView,
        resolve_target: Option<&'a wgpu::TextureView>,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        wgpu::RenderPassColorAttachment {
            view,
            resolve_target,
            ops: self.operations().to_wgpu(),
        }
    }
}

impl fmt::Display for ColorAttachment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ColorAttachment(load={}, store={}, resolve={})",
            self.load_op, self.store_op, self.has_resolve
        )
    }
}

// ---------------------------------------------------------------------------
// DepthStencilAttachment
// ---------------------------------------------------------------------------

/// A depth/stencil attachment for a render pass.
///
/// Describes a depth and/or stencil render target with load/store operations.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `depth_ops` | `Option<Operations<f32>>` | Depth load/store (None = read-only) |
/// | `stencil_ops` | `Option<Operations<u32>>` | Stencil load/store (None = read-only) |
///
/// # Read-Only Mode
///
/// Setting `depth_ops` or `stencil_ops` to `None` makes that aspect read-only:
/// - Read-only depth: useful for transparent rendering after opaque pass
/// - Read-only stencil: useful for stencil-based masking in subsequent passes
///
/// # Example
///
/// ```ignore
/// // Full depth/stencil write
/// let attachment = DepthStencilAttachment::new()
///     .depth_clear(1.0)
///     .stencil_clear(0);
///
/// // Read-only depth
/// let attachment = DepthStencilAttachment::depth_read_only();
///
/// // Depth-only (no stencil)
/// let attachment = DepthStencilAttachment::depth_only(1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilAttachment {
    /// Depth operations (None = read-only depth).
    pub depth_ops: Option<Operations<f32>>,
    /// Stencil operations (None = read-only stencil).
    pub stencil_ops: Option<Operations<u32>>,
}

impl Default for DepthStencilAttachment {
    fn default() -> Self {
        Self {
            depth_ops: Some(Operations::clear_depth()),
            stencil_ops: None,
        }
    }
}

impl DepthStencilAttachment {
    /// Create a new depth/stencil attachment with default operations.
    ///
    /// Defaults to clearing depth to 1.0 with no stencil operations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a depth-only attachment (no stencil).
    pub fn depth_only(clear_depth: f32) -> Self {
        Self {
            depth_ops: Some(Operations::clear(clear_depth)),
            stencil_ops: None,
        }
    }

    /// Create a depth-only attachment with reverse-Z (clear to 0.0).
    pub fn depth_only_reverse_z() -> Self {
        Self::depth_only(0.0)
    }

    /// Create a read-only depth attachment.
    ///
    /// Useful for rendering transparent objects after opaque pass.
    pub fn depth_read_only() -> Self {
        Self {
            depth_ops: None,
            stencil_ops: None,
        }
    }

    /// Create a depth + stencil attachment.
    pub fn depth_stencil(clear_depth: f32, clear_stencil: u32) -> Self {
        Self {
            depth_ops: Some(Operations::clear(clear_depth)),
            stencil_ops: Some(Operations::clear(clear_stencil)),
        }
    }

    /// Create a stencil-only attachment (read-only depth).
    pub fn stencil_only(clear_stencil: u32) -> Self {
        Self {
            depth_ops: None,
            stencil_ops: Some(Operations::clear(clear_stencil)),
        }
    }

    /// Set depth operations.
    pub fn with_depth_ops(mut self, ops: Operations<f32>) -> Self {
        self.depth_ops = Some(ops);
        self
    }

    /// Set stencil operations.
    pub fn with_stencil_ops(mut self, ops: Operations<u32>) -> Self {
        self.stencil_ops = Some(ops);
        self
    }

    /// Set depth to clear with a specific value.
    pub fn depth_clear(mut self, value: f32) -> Self {
        self.depth_ops = Some(Operations::clear(value));
        self
    }

    /// Set depth to load existing content.
    pub fn depth_load(mut self) -> Self {
        self.depth_ops = Some(Operations::load_store());
        self
    }

    /// Disable depth operations (read-only).
    pub fn no_depth_ops(mut self) -> Self {
        self.depth_ops = None;
        self
    }

    /// Set stencil to clear with a specific value.
    pub fn stencil_clear(mut self, value: u32) -> Self {
        self.stencil_ops = Some(Operations::clear(value));
        self
    }

    /// Set stencil to load existing content.
    pub fn stencil_load(mut self) -> Self {
        self.stencil_ops = Some(Operations::load_store());
        self
    }

    /// Disable stencil operations (read-only).
    pub fn no_stencil_ops(mut self) -> Self {
        self.stencil_ops = None;
        self
    }

    /// Check if depth is writable.
    pub fn is_depth_writable(&self) -> bool {
        self.depth_ops.is_some()
    }

    /// Check if stencil is writable.
    pub fn is_stencil_writable(&self) -> bool {
        self.stencil_ops.is_some()
    }

    /// Check if this is a read-only attachment.
    pub fn is_read_only(&self) -> bool {
        self.depth_ops.is_none() && self.stencil_ops.is_none()
    }

    /// Build the wgpu RenderPassDepthStencilAttachment.
    pub fn build_wgpu<'a>(
        &self,
        view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPassDepthStencilAttachment<'a> {
        wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: self.depth_ops.map(|ops| ops.to_wgpu()),
            stencil_ops: self.stencil_ops.map(|ops| ops.to_wgpu()),
        }
    }
}

impl fmt::Display for DepthStencilAttachment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DepthStencilAttachment(depth={}, stencil={})",
            self.depth_ops
                .map(|o| format!("{}", o))
                .unwrap_or_else(|| "read-only".to_string()),
            self.stencil_ops
                .map(|o| format!("{}", o))
                .unwrap_or_else(|| "read-only".to_string())
        )
    }
}

// ---------------------------------------------------------------------------
// TimestampWrites
// ---------------------------------------------------------------------------

/// Timestamp write configuration for GPU profiling.
///
/// Allows writing GPU timestamps at the beginning and/or end of a render pass
/// for performance measurement.
///
/// # Requirements
///
/// - Requires `wgpu::Features::TIMESTAMP_QUERY`
/// - Query set must be created with `wgpu::QueryType::Timestamp`
///
/// # Example
///
/// ```ignore
/// let timestamps = TimestampWrites::new()
///     .beginning(0)
///     .end(1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampWrites {
    /// Index in the query set for the beginning timestamp.
    pub beginning_of_pass_write_index: Option<u32>,
    /// Index in the query set for the end timestamp.
    pub end_of_pass_write_index: Option<u32>,
}

impl Default for TimestampWrites {
    fn default() -> Self {
        Self {
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: None,
        }
    }
}

impl TimestampWrites {
    /// Create empty timestamp writes configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create timestamp writes with both beginning and end indices.
    pub fn both(beginning: u32, end: u32) -> Self {
        Self {
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(end),
        }
    }

    /// Create timestamp writes with only beginning index.
    pub fn beginning_only(index: u32) -> Self {
        Self {
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: None,
        }
    }

    /// Create timestamp writes with only end index.
    pub fn end_only(index: u32) -> Self {
        Self {
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(index),
        }
    }

    /// Set the beginning timestamp index.
    pub fn beginning(mut self, index: u32) -> Self {
        self.beginning_of_pass_write_index = Some(index);
        self
    }

    /// Set the end timestamp index.
    pub fn end(mut self, index: u32) -> Self {
        self.end_of_pass_write_index = Some(index);
        self
    }

    /// Check if any timestamps are configured.
    pub fn is_enabled(&self) -> bool {
        self.beginning_of_pass_write_index.is_some() || self.end_of_pass_write_index.is_some()
    }

    /// Build the wgpu RenderPassTimestampWrites.
    pub fn build_wgpu<'a>(
        &self,
        query_set: &'a wgpu::QuerySet,
    ) -> wgpu::RenderPassTimestampWrites<'a> {
        wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: self.beginning_of_pass_write_index,
            end_of_pass_write_index: self.end_of_pass_write_index,
        }
    }
}

impl fmt::Display for TimestampWrites {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TimestampWrites(begin={:?}, end={:?})",
            self.beginning_of_pass_write_index, self.end_of_pass_write_index
        )
    }
}

// ---------------------------------------------------------------------------
// OcclusionQuerySet
// ---------------------------------------------------------------------------

/// Occlusion query set configuration for visibility queries.
///
/// Allows counting how many samples pass depth/stencil tests for visibility
/// culling and other optimization techniques.
///
/// # Requirements
///
/// - Query set must be created with `wgpu::QueryType::Occlusion`
///
/// # Use Cases
///
/// - Visibility culling: determine if objects are visible before rendering
/// - LOD selection: choose detail level based on screen coverage
/// - Portal rendering: determine if portals are visible
///
/// # Example
///
/// ```ignore
/// let occlusion = OcclusionQuerySet::new(query_index);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OcclusionQuerySet {
    /// Whether occlusion queries are enabled.
    pub enabled: bool,
}

impl Default for OcclusionQuerySet {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl OcclusionQuerySet {
    /// Create a new occlusion query set configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled occlusion query set.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Enable occlusion queries.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable occlusion queries.
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }
}

impl fmt::Display for OcclusionQuerySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OcclusionQuerySet(enabled={})", self.enabled)
    }
}

// ---------------------------------------------------------------------------
// RenderPassDescriptor
// ---------------------------------------------------------------------------

/// High-level render pass descriptor.
///
/// Owns the configuration for a render pass including color attachments,
/// depth/stencil attachment, timestamp writes, and occlusion queries.
///
/// # Note
///
/// This descriptor does not own the texture views or query sets.
/// The actual wgpu::RenderPassDescriptor is built at pass creation time
/// with borrowed references to those resources.
///
/// # Example
///
/// ```ignore
/// let desc = RenderPassDescriptor::new()
///     .label("forward_pass")
///     .color_attachment(ColorAttachment::with_clear(wgpu::Color::BLACK))
///     .depth_stencil(DepthStencilAttachment::depth_only(1.0))
///     .timestamp_writes(TimestampWrites::both(0, 1));
/// ```
#[derive(Debug, Clone, Default)]
pub struct RenderPassDescriptor {
    /// Optional debug label.
    pub label: Option<String>,
    /// Color attachment configurations (up to 8).
    pub color_attachments: Vec<Option<ColorAttachment>>,
    /// Depth/stencil attachment configuration.
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
    /// Timestamp writes configuration.
    pub timestamp_writes: Option<TimestampWrites>,
    /// Whether occlusion queries are enabled.
    pub occlusion_query_enabled: bool,
}

impl RenderPassDescriptor {
    /// Create a new empty render pass descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debug label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a color attachment.
    pub fn color_attachment(mut self, attachment: ColorAttachment) -> Self {
        self.color_attachments.push(Some(attachment));
        self
    }

    /// Add an empty slot (None) for a color attachment.
    pub fn empty_color_slot(mut self) -> Self {
        self.color_attachments.push(None);
        self
    }

    /// Set all color attachments at once.
    pub fn color_attachments(mut self, attachments: Vec<Option<ColorAttachment>>) -> Self {
        self.color_attachments = attachments;
        self
    }

    /// Set the depth/stencil attachment.
    pub fn depth_stencil(mut self, attachment: DepthStencilAttachment) -> Self {
        self.depth_stencil_attachment = Some(attachment);
        self
    }

    /// Set timestamp writes configuration.
    pub fn timestamp_writes(mut self, writes: TimestampWrites) -> Self {
        self.timestamp_writes = Some(writes);
        self
    }

    /// Enable occlusion queries.
    pub fn with_occlusion_queries(mut self) -> Self {
        self.occlusion_query_enabled = true;
        self
    }

    /// Get the number of color attachments.
    pub fn color_attachment_count(&self) -> usize {
        self.color_attachments.len()
    }

    /// Check if depth/stencil is configured.
    pub fn has_depth_stencil(&self) -> bool {
        self.depth_stencil_attachment.is_some()
    }

    /// Check if timestamp writes are configured.
    pub fn has_timestamp_writes(&self) -> bool {
        self.timestamp_writes.is_some()
    }

    /// Check if occlusion queries are enabled.
    pub fn has_occlusion_queries(&self) -> bool {
        self.occlusion_query_enabled
    }
}

impl fmt::Display for RenderPassDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RenderPassDescriptor(label={:?}, colors={}, depth_stencil={}, timestamps={}, occlusion={})",
            self.label,
            self.color_attachments.len(),
            self.depth_stencil_attachment.is_some(),
            self.timestamp_writes.is_some(),
            self.occlusion_query_enabled
        )
    }
}

// ---------------------------------------------------------------------------
// RenderPassBuilder
// ---------------------------------------------------------------------------

/// Builder for creating render pass descriptors with a fluent API.
///
/// # Example
///
/// ```ignore
/// let desc = RenderPassBuilder::new()
///     .label("forward_pass")
///     .color_attachment(&color_view, Operations::clear_black())
///     .color_attachment(&normal_view, Operations::clear_black())
///     .depth_stencil(&depth_view, Operations::clear_depth(), None)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct RenderPassBuilder {
    descriptor: RenderPassDescriptor,
}

impl RenderPassBuilder {
    /// Create a new render pass builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debug label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.descriptor.label = Some(label.into());
        self
    }

    /// Add a color attachment with operations.
    pub fn color_attachment(mut self, ops: Operations<wgpu::Color>) -> Self {
        self.descriptor.color_attachments.push(Some(ColorAttachment {
            clear_color: match ops.load {
                LoadOp::Clear(c) => c,
                LoadOp::Load => DEFAULT_CLEAR_COLOR,
            },
            load_op: ops.load,
            store_op: ops.store,
            has_resolve: false,
        }));
        self
    }

    /// Add a color attachment with MSAA resolve.
    pub fn color_attachment_msaa(mut self, ops: Operations<wgpu::Color>) -> Self {
        self.descriptor.color_attachments.push(Some(ColorAttachment {
            clear_color: match ops.load {
                LoadOp::Clear(c) => c,
                LoadOp::Load => DEFAULT_CLEAR_COLOR,
            },
            load_op: ops.load,
            store_op: ops.store,
            has_resolve: true,
        }));
        self
    }

    /// Add an empty color attachment slot.
    pub fn empty_color_slot(mut self) -> Self {
        self.descriptor.color_attachments.push(None);
        self
    }

    /// Set the depth/stencil attachment with operations.
    pub fn depth_stencil(
        mut self,
        depth_ops: Option<Operations<f32>>,
        stencil_ops: Option<Operations<u32>>,
    ) -> Self {
        self.descriptor.depth_stencil_attachment = Some(DepthStencilAttachment {
            depth_ops,
            stencil_ops,
        });
        self
    }

    /// Set depth-only attachment (no stencil).
    pub fn depth_only(mut self, ops: Operations<f32>) -> Self {
        self.descriptor.depth_stencil_attachment = Some(DepthStencilAttachment {
            depth_ops: Some(ops),
            stencil_ops: None,
        });
        self
    }

    /// Set read-only depth attachment.
    pub fn depth_read_only(mut self) -> Self {
        self.descriptor.depth_stencil_attachment = Some(DepthStencilAttachment::depth_read_only());
        self
    }

    /// Set timestamp writes configuration.
    pub fn timestamp_writes(mut self, writes: TimestampWrites) -> Self {
        self.descriptor.timestamp_writes = Some(writes);
        self
    }

    /// Enable occlusion queries.
    pub fn with_occlusion_queries(mut self) -> Self {
        self.descriptor.occlusion_query_enabled = true;
        self
    }

    /// Build the render pass descriptor.
    pub fn build(self) -> RenderPassDescriptor {
        self.descriptor
    }

    // -------------------------------------------------------------------------
    // Presets
    // -------------------------------------------------------------------------

    /// Create a simple color-only pass (no depth).
    ///
    /// Clears to black, stores result.
    pub fn simple_color() -> Self {
        Self::new()
            .label("simple_color")
            .color_attachment(Operations::clear_black())
    }

    /// Create a simple color-only pass with custom clear color.
    pub fn simple_color_clear(color: wgpu::Color) -> Self {
        Self::new()
            .label("simple_color")
            .color_attachment(Operations::clear(color))
    }

    /// Create a color + depth pass.
    ///
    /// Clears color to black, clears depth to 1.0, stores both.
    pub fn color_depth() -> Self {
        Self::new()
            .label("color_depth")
            .color_attachment(Operations::clear_black())
            .depth_only(Operations::clear_depth())
    }

    /// Create a color + depth pass with reverse-Z.
    ///
    /// Clears color to black, clears depth to 0.0, stores both.
    pub fn color_depth_reverse_z() -> Self {
        Self::new()
            .label("color_depth_reverse_z")
            .color_attachment(Operations::clear_black())
            .depth_only(Operations::clear_depth_reverse_z())
    }

    /// Create a shadow map pass (depth only, no color).
    ///
    /// Clears depth to 1.0, stores result.
    pub fn shadow_map() -> Self {
        Self::new()
            .label("shadow_map")
            .depth_only(Operations::clear_depth())
    }

    /// Create a shadow map pass with reverse-Z.
    pub fn shadow_map_reverse_z() -> Self {
        Self::new()
            .label("shadow_map_reverse_z")
            .depth_only(Operations::clear_depth_reverse_z())
    }

    /// Create a G-buffer pass for deferred rendering.
    ///
    /// Creates 3 color attachments (albedo, normal, material) + depth.
    pub fn gbuffer() -> Self {
        Self::new()
            .label("gbuffer")
            .color_attachment(Operations::clear_black()) // Albedo
            .color_attachment(Operations::clear_black()) // Normal
            .color_attachment(Operations::clear_black()) // Material
            .depth_only(Operations::clear_depth())
    }

    /// Create a G-buffer pass with reverse-Z.
    pub fn gbuffer_reverse_z() -> Self {
        Self::new()
            .label("gbuffer_reverse_z")
            .color_attachment(Operations::clear_black()) // Albedo
            .color_attachment(Operations::clear_black()) // Normal
            .color_attachment(Operations::clear_black()) // Material
            .depth_only(Operations::clear_depth_reverse_z())
    }

    /// Create a post-processing pass (color only, load existing).
    ///
    /// Loads existing color, stores result.
    pub fn post_process() -> Self {
        Self::new()
            .label("post_process")
            .color_attachment(Operations::load_store())
    }

    /// Create a transparent rendering pass.
    ///
    /// Loads existing color (blends with opaque), read-only depth test.
    pub fn transparent() -> Self {
        Self::new()
            .label("transparent")
            .color_attachment(Operations::load_store())
            .depth_read_only()
    }

    /// Create a full-screen effect pass (no depth).
    pub fn fullscreen() -> Self {
        Self::new()
            .label("fullscreen")
            .color_attachment(Operations::clear_black())
    }

    /// Create a UI/HUD pass (no depth, load existing color).
    pub fn ui() -> Self {
        Self::new()
            .label("ui")
            .color_attachment(Operations::load_store())
    }

    /// Create an MSAA resolve pass.
    ///
    /// Color attachment with resolve target, stores resolved result.
    pub fn msaa_resolve() -> Self {
        Self::new()
            .label("msaa_resolve")
            .color_attachment_msaa(Operations::clear_black())
    }

    /// Create a depth pre-pass (depth only, discards depth after).
    ///
    /// Useful for populating the depth buffer before lighting.
    pub fn depth_prepass() -> Self {
        Self::new()
            .label("depth_prepass")
            .depth_only(Operations::clear_depth())
    }

    /// Create a stencil-only pass.
    pub fn stencil_only() -> Self {
        Self::new().label("stencil_only").depth_stencil(
            None,
            Some(Operations::clear_stencil()),
        )
    }
}

// ---------------------------------------------------------------------------
// RenderPassInfo
// ---------------------------------------------------------------------------

/// Metadata about a render pass configuration.
///
/// Provides information for debugging, profiling, and tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPassInfo {
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the pass type.
    pub description: &'static str,
    /// Number of color attachments.
    pub color_count: usize,
    /// Whether depth is used.
    pub has_depth: bool,
    /// Whether stencil is used.
    pub has_stencil: bool,
    /// Whether MSAA resolve is used.
    pub has_resolve: bool,
    /// Typical use cases.
    pub use_cases: &'static [&'static str],
}

/// Information about common render pass types.
pub const RENDER_PASS_PRESETS: [RenderPassInfo; 10] = [
    RenderPassInfo {
        name: "simple_color",
        description: "Single color target, no depth",
        color_count: 1,
        has_depth: false,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["2D rendering", "UI", "post-processing"],
    },
    RenderPassInfo {
        name: "color_depth",
        description: "Single color target with depth",
        color_count: 1,
        has_depth: true,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["forward rendering", "basic 3D"],
    },
    RenderPassInfo {
        name: "shadow_map",
        description: "Depth-only for shadow mapping",
        color_count: 0,
        has_depth: true,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["directional shadows", "point shadows", "spot shadows"],
    },
    RenderPassInfo {
        name: "gbuffer",
        description: "Multiple render targets for deferred rendering",
        color_count: 3,
        has_depth: true,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["deferred rendering", "G-buffer pass"],
    },
    RenderPassInfo {
        name: "post_process",
        description: "Color-only pass that preserves existing content",
        color_count: 1,
        has_depth: false,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["bloom", "tone mapping", "color grading"],
    },
    RenderPassInfo {
        name: "transparent",
        description: "Blended rendering with read-only depth",
        color_count: 1,
        has_depth: true,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["transparent objects", "particles", "decals"],
    },
    RenderPassInfo {
        name: "fullscreen",
        description: "Full-screen effect without depth",
        color_count: 1,
        has_depth: false,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["final composite", "screen-space effects"],
    },
    RenderPassInfo {
        name: "msaa_resolve",
        description: "MSAA color with resolve target",
        color_count: 1,
        has_depth: false,
        has_stencil: false,
        has_resolve: true,
        use_cases: &["anti-aliasing", "MSAA rendering"],
    },
    RenderPassInfo {
        name: "depth_prepass",
        description: "Depth-only pre-pass",
        color_count: 0,
        has_depth: true,
        has_stencil: false,
        has_resolve: false,
        use_cases: &["early-Z", "depth pre-population"],
    },
    RenderPassInfo {
        name: "stencil_only",
        description: "Stencil-only pass",
        color_count: 0,
        has_depth: false,
        has_stencil: true,
        has_resolve: false,
        use_cases: &["stencil shadows", "masking"],
    },
];

/// Get information about a render pass preset by name.
pub fn get_preset_info(name: &str) -> Option<&'static RenderPassInfo> {
    RENDER_PASS_PRESETS.iter().find(|p| p.name == name)
}

/// Get all available preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    RENDER_PASS_PRESETS.iter().map(|p| p.name)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Validate that color attachment count doesn't exceed the maximum.
pub fn validate_color_attachment_count(count: usize) -> Result<(), RenderPassError> {
    if count > MAX_COLOR_ATTACHMENTS {
        Err(RenderPassError::TooManyColorAttachments {
            count,
            max: MAX_COLOR_ATTACHMENTS,
        })
    } else {
        Ok(())
    }
}

/// Check if a descriptor is valid.
pub fn validate_descriptor(desc: &RenderPassDescriptor) -> Result<(), RenderPassError> {
    validate_color_attachment_count(desc.color_attachments.len())?;

    // A render pass should have at least one color or depth attachment
    if desc.color_attachments.is_empty() && desc.depth_stencil_attachment.is_none() {
        return Err(RenderPassError::NoAttachments);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when creating render passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderPassError {
    /// Too many color attachments.
    TooManyColorAttachments {
        /// Requested count.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// No attachments specified.
    NoAttachments,
    /// Invalid timestamp write index.
    InvalidTimestampIndex {
        /// The invalid index.
        index: u32,
        /// Query set size.
        query_set_size: u32,
    },
}

impl fmt::Display for RenderPassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyColorAttachments { count, max } => {
                write!(f, "Too many color attachments: {} (max {})", count, max)
            }
            Self::NoAttachments => {
                write!(f, "Render pass must have at least one color or depth attachment")
            }
            Self::InvalidTimestampIndex {
                index,
                query_set_size,
            } => {
                write!(
                    f,
                    "Invalid timestamp index {} for query set of size {}",
                    index, query_set_size
                )
            }
        }
    }
}

impl std::error::Error for RenderPassError {}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LoadOp tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_clear_color() {
        let op = LoadOp::Clear(wgpu::Color::RED);
        assert!(matches!(op, LoadOp::Clear(_)));
    }

    #[test]
    fn test_load_op_load() {
        let op: LoadOp<wgpu::Color> = LoadOp::Load;
        assert!(matches!(op, LoadOp::Load));
    }

    #[test]
    fn test_load_op_display_clear() {
        let op = LoadOp::Clear(1.0f32);
        let s = format!("{}", op);
        assert!(s.contains("Clear"));
    }

    #[test]
    fn test_load_op_display_load() {
        let op: LoadOp<f32> = LoadOp::Load;
        let s = format!("{}", op);
        assert_eq!(s, "Load");
    }

    #[test]
    fn test_load_op_color_to_wgpu_clear() {
        let op = LoadOp::Clear(wgpu::Color::BLUE);
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Clear(_)));
    }

    #[test]
    fn test_load_op_color_to_wgpu_load() {
        let op: LoadOp<wgpu::Color> = LoadOp::Load;
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
    }

    #[test]
    fn test_load_op_depth_to_wgpu() {
        let op = LoadOp::Clear(0.5f32);
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Clear(0.5)));
    }

    #[test]
    fn test_load_op_stencil_to_wgpu() {
        let op = LoadOp::Clear(128u32);
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Clear(128)));
    }

    #[test]
    fn test_load_op_default_f32() {
        let op: LoadOp<f32> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(0.0)));
    }

    #[test]
    fn test_load_op_default_u32() {
        let op: LoadOp<u32> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(0)));
    }

    // -------------------------------------------------------------------------
    // StoreOp tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_store_op_store() {
        let op = StoreOp::Store;
        assert_eq!(op, StoreOp::Store);
    }

    #[test]
    fn test_store_op_discard() {
        let op = StoreOp::Discard;
        assert_eq!(op, StoreOp::Discard);
    }

    #[test]
    fn test_store_op_default() {
        let op = StoreOp::default();
        assert_eq!(op, StoreOp::Store);
    }

    #[test]
    fn test_store_op_display() {
        assert_eq!(format!("{}", StoreOp::Store), "Store");
        assert_eq!(format!("{}", StoreOp::Discard), "Discard");
    }

    #[test]
    fn test_store_op_to_wgpu() {
        assert!(matches!(StoreOp::Store.to_wgpu(), wgpu::StoreOp::Store));
        assert!(matches!(StoreOp::Discard.to_wgpu(), wgpu::StoreOp::Discard));
    }

    #[test]
    fn test_store_op_into_wgpu() {
        let wgpu_op: wgpu::StoreOp = StoreOp::Store.into();
        assert!(matches!(wgpu_op, wgpu::StoreOp::Store));
    }

    // -------------------------------------------------------------------------
    // Operations tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_new() {
        let ops = Operations::new(LoadOp::Clear(1.0f32), StoreOp::Store);
        assert!(matches!(ops.load, LoadOp::Clear(1.0)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear() {
        let ops = Operations::clear(wgpu::Color::RED);
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear_discard() {
        let ops = Operations::clear_discard(1.0f32);
        assert!(matches!(ops.load, LoadOp::Clear(1.0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_load_store() {
        let ops: Operations<f32> = Operations::load_store();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_load_discard() {
        let ops: Operations<f32> = Operations::load_discard();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_clear_black() {
        let ops = Operations::clear_black();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color, wgpu::Color::BLACK);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_opaque_black() {
        let ops = Operations::clear_opaque_black();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.0);
            assert_eq!(color.g, 0.0);
            assert_eq!(color.b, 0.0);
            assert_eq!(color.a, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_white() {
        let ops = Operations::clear_white();
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color, wgpu::Color::WHITE);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_rgb() {
        let ops = Operations::clear_rgb(0.5, 0.6, 0.7);
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.5);
            assert_eq!(color.g, 0.6);
            assert_eq!(color.b, 0.7);
            assert_eq!(color.a, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_rgba() {
        let ops = Operations::clear_rgba(0.1, 0.2, 0.3, 0.4);
        if let LoadOp::Clear(color) = ops.load {
            assert_eq!(color.r, 0.1);
            assert_eq!(color.g, 0.2);
            assert_eq!(color.b, 0.3);
            assert_eq!(color.a, 0.4);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_depth() {
        let ops = Operations::clear_depth();
        if let LoadOp::Clear(depth) = ops.load {
            assert_eq!(depth, 1.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_depth_reverse_z() {
        let ops = Operations::clear_depth_reverse_z();
        if let LoadOp::Clear(depth) = ops.load {
            assert_eq!(depth, 0.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_depth_transient() {
        let ops = Operations::clear_depth_transient();
        assert!(matches!(ops.load, LoadOp::Clear(1.0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_clear_stencil() {
        let ops = Operations::clear_stencil();
        if let LoadOp::Clear(stencil) = ops.load {
            assert_eq!(stencil, 0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_operations_clear_stencil_transient() {
        let ops = Operations::clear_stencil_transient();
        assert!(matches!(ops.load, LoadOp::Clear(0)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_display() {
        let ops = Operations::clear(1.0f32);
        let s = format!("{}", ops);
        assert!(s.contains("Operations"));
        assert!(s.contains("Clear"));
        assert!(s.contains("Store"));
    }

    #[test]
    fn test_operations_color_to_wgpu() {
        let ops = Operations::clear_black();
        let wgpu_ops = ops.to_wgpu();
        assert!(matches!(wgpu_ops.load, wgpu::LoadOp::Clear(_)));
        assert!(matches!(wgpu_ops.store, wgpu::StoreOp::Store));
    }

    #[test]
    fn test_operations_depth_to_wgpu() {
        let ops = Operations::clear_depth();
        let wgpu_ops = ops.to_wgpu();
        assert!(matches!(wgpu_ops.load, wgpu::LoadOp::Clear(1.0)));
        assert!(matches!(wgpu_ops.store, wgpu::StoreOp::Store));
    }

    #[test]
    fn test_operations_stencil_to_wgpu() {
        let ops = Operations::clear_stencil();
        let wgpu_ops = ops.to_wgpu();
        assert!(matches!(wgpu_ops.load, wgpu::LoadOp::Clear(0)));
    }

    // -------------------------------------------------------------------------
    // ColorAttachment tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_attachment_new() {
        let att = ColorAttachment::new();
        assert_eq!(att.store_op, StoreOp::Store);
        assert!(!att.has_resolve);
    }

    #[test]
    fn test_color_attachment_default() {
        let att = ColorAttachment::default();
        assert_eq!(att.clear_color, DEFAULT_CLEAR_COLOR);
        assert_eq!(att.store_op, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_with_clear() {
        let att = ColorAttachment::with_clear(wgpu::Color::RED);
        if let LoadOp::Clear(color) = att.load_op {
            assert_eq!(color, wgpu::Color::RED);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_color_attachment_with_load() {
        let att = ColorAttachment::with_load();
        assert!(matches!(att.load_op, LoadOp::Load));
    }

    #[test]
    fn test_color_attachment_clear_color() {
        let att = ColorAttachment::new().clear_color(wgpu::Color::GREEN);
        assert_eq!(att.clear_color, wgpu::Color::GREEN);
    }

    #[test]
    fn test_color_attachment_load() {
        let att = ColorAttachment::new().load();
        assert!(matches!(att.load_op, LoadOp::Load));
    }

    #[test]
    fn test_color_attachment_store() {
        let att = ColorAttachment::new().store(StoreOp::Discard);
        assert_eq!(att.store_op, StoreOp::Discard);
    }

    #[test]
    fn test_color_attachment_with_resolve() {
        let att = ColorAttachment::new().with_resolve();
        assert!(att.has_resolve);
    }

    #[test]
    fn test_color_attachment_operations() {
        let att = ColorAttachment::with_clear(wgpu::Color::BLUE);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_display() {
        let att = ColorAttachment::new();
        let s = format!("{}", att);
        assert!(s.contains("ColorAttachment"));
    }

    // -------------------------------------------------------------------------
    // DepthStencilAttachment tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_stencil_attachment_new() {
        let att = DepthStencilAttachment::new();
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_none());
    }

    #[test]
    fn test_depth_stencil_attachment_default() {
        let att = DepthStencilAttachment::default();
        assert!(att.depth_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_attachment_depth_only() {
        let att = DepthStencilAttachment::depth_only(0.5);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_none());
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.5);
            }
        }
    }

    #[test]
    fn test_depth_stencil_attachment_depth_only_reverse_z() {
        let att = DepthStencilAttachment::depth_only_reverse_z();
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.0);
            }
        }
    }

    #[test]
    fn test_depth_stencil_attachment_depth_read_only() {
        let att = DepthStencilAttachment::depth_read_only();
        assert!(att.depth_ops.is_none());
        assert!(att.stencil_ops.is_none());
        assert!(att.is_read_only());
    }

    #[test]
    fn test_depth_stencil_attachment_depth_stencil() {
        let att = DepthStencilAttachment::depth_stencil(1.0, 128);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_attachment_stencil_only() {
        let att = DepthStencilAttachment::stencil_only(255);
        assert!(att.depth_ops.is_none());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_attachment_with_depth_ops() {
        let att = DepthStencilAttachment::new()
            .with_depth_ops(Operations::clear(0.25));
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.25);
            }
        }
    }

    #[test]
    fn test_depth_stencil_attachment_with_stencil_ops() {
        let att = DepthStencilAttachment::new()
            .with_stencil_ops(Operations::clear(64));
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_attachment_depth_clear() {
        let att = DepthStencilAttachment::new().depth_clear(0.75);
        if let Some(ops) = att.depth_ops {
            if let LoadOp::Clear(depth) = ops.load {
                assert_eq!(depth, 0.75);
            }
        }
    }

    #[test]
    fn test_depth_stencil_attachment_depth_load() {
        let att = DepthStencilAttachment::new().depth_load();
        if let Some(ops) = att.depth_ops {
            assert!(matches!(ops.load, LoadOp::Load));
        }
    }

    #[test]
    fn test_depth_stencil_attachment_no_depth_ops() {
        let att = DepthStencilAttachment::new().no_depth_ops();
        assert!(att.depth_ops.is_none());
    }

    #[test]
    fn test_depth_stencil_attachment_stencil_clear() {
        let att = DepthStencilAttachment::new().stencil_clear(32);
        if let Some(ops) = att.stencil_ops {
            if let LoadOp::Clear(stencil) = ops.load {
                assert_eq!(stencil, 32);
            }
        }
    }

    #[test]
    fn test_depth_stencil_attachment_stencil_load() {
        let att = DepthStencilAttachment::new().stencil_load();
        if let Some(ops) = att.stencil_ops {
            assert!(matches!(ops.load, LoadOp::Load));
        }
    }

    #[test]
    fn test_depth_stencil_attachment_no_stencil_ops() {
        let att = DepthStencilAttachment::depth_stencil(1.0, 0)
            .no_stencil_ops();
        assert!(att.stencil_ops.is_none());
    }

    #[test]
    fn test_depth_stencil_attachment_is_depth_writable() {
        let att = DepthStencilAttachment::depth_only(1.0);
        assert!(att.is_depth_writable());

        let att_ro = DepthStencilAttachment::depth_read_only();
        assert!(!att_ro.is_depth_writable());
    }

    #[test]
    fn test_depth_stencil_attachment_is_stencil_writable() {
        let att = DepthStencilAttachment::depth_stencil(1.0, 0);
        assert!(att.is_stencil_writable());

        let att_no = DepthStencilAttachment::depth_only(1.0);
        assert!(!att_no.is_stencil_writable());
    }

    #[test]
    fn test_depth_stencil_attachment_is_read_only() {
        let att = DepthStencilAttachment::depth_read_only();
        assert!(att.is_read_only());

        let att_write = DepthStencilAttachment::depth_only(1.0);
        assert!(!att_write.is_read_only());
    }

    #[test]
    fn test_depth_stencil_attachment_display() {
        let att = DepthStencilAttachment::depth_stencil(1.0, 0);
        let s = format!("{}", att);
        assert!(s.contains("DepthStencilAttachment"));
    }

    // -------------------------------------------------------------------------
    // TimestampWrites tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timestamp_writes_new() {
        let ts = TimestampWrites::new();
        assert!(ts.beginning_of_pass_write_index.is_none());
        assert!(ts.end_of_pass_write_index.is_none());
    }

    #[test]
    fn test_timestamp_writes_both() {
        let ts = TimestampWrites::both(0, 1);
        assert_eq!(ts.beginning_of_pass_write_index, Some(0));
        assert_eq!(ts.end_of_pass_write_index, Some(1));
    }

    #[test]
    fn test_timestamp_writes_beginning_only() {
        let ts = TimestampWrites::beginning_only(5);
        assert_eq!(ts.beginning_of_pass_write_index, Some(5));
        assert!(ts.end_of_pass_write_index.is_none());
    }

    #[test]
    fn test_timestamp_writes_end_only() {
        let ts = TimestampWrites::end_only(10);
        assert!(ts.beginning_of_pass_write_index.is_none());
        assert_eq!(ts.end_of_pass_write_index, Some(10));
    }

    #[test]
    fn test_timestamp_writes_beginning() {
        let ts = TimestampWrites::new().beginning(3);
        assert_eq!(ts.beginning_of_pass_write_index, Some(3));
    }

    #[test]
    fn test_timestamp_writes_end() {
        let ts = TimestampWrites::new().end(7);
        assert_eq!(ts.end_of_pass_write_index, Some(7));
    }

    #[test]
    fn test_timestamp_writes_is_enabled() {
        let ts_empty = TimestampWrites::new();
        assert!(!ts_empty.is_enabled());

        let ts_begin = TimestampWrites::beginning_only(0);
        assert!(ts_begin.is_enabled());

        let ts_end = TimestampWrites::end_only(1);
        assert!(ts_end.is_enabled());

        let ts_both = TimestampWrites::both(0, 1);
        assert!(ts_both.is_enabled());
    }

    #[test]
    fn test_timestamp_writes_display() {
        let ts = TimestampWrites::both(0, 1);
        let s = format!("{}", ts);
        assert!(s.contains("TimestampWrites"));
    }

    // -------------------------------------------------------------------------
    // OcclusionQuerySet tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_occlusion_query_set_new() {
        let oqs = OcclusionQuerySet::new();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_disabled() {
        let oqs = OcclusionQuerySet::disabled();
        assert!(!oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_enable() {
        let oqs = OcclusionQuerySet::disabled().enable();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_disable() {
        let oqs = OcclusionQuerySet::new().disable();
        assert!(!oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_display() {
        let oqs = OcclusionQuerySet::new();
        let s = format!("{}", oqs);
        assert!(s.contains("OcclusionQuerySet"));
    }

    // -------------------------------------------------------------------------
    // RenderPassDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_descriptor_new() {
        let desc = RenderPassDescriptor::new();
        assert!(desc.label.is_none());
        assert!(desc.color_attachments.is_empty());
        assert!(desc.depth_stencil_attachment.is_none());
    }

    #[test]
    fn test_render_pass_descriptor_label() {
        let desc = RenderPassDescriptor::new().label("test_pass");
        assert_eq!(desc.label, Some("test_pass".to_string()));
    }

    #[test]
    fn test_render_pass_descriptor_color_attachment() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new());
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_render_pass_descriptor_empty_color_slot() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new())
            .empty_color_slot()
            .color_attachment(ColorAttachment::new());
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.color_attachments[1].is_none());
    }

    #[test]
    fn test_render_pass_descriptor_color_attachments() {
        let atts = vec![
            Some(ColorAttachment::new()),
            None,
            Some(ColorAttachment::new()),
        ];
        let desc = RenderPassDescriptor::new().color_attachments(atts);
        assert_eq!(desc.color_attachments.len(), 3);
    }

    #[test]
    fn test_render_pass_descriptor_depth_stencil() {
        let desc = RenderPassDescriptor::new()
            .depth_stencil(DepthStencilAttachment::new());
        assert!(desc.depth_stencil_attachment.is_some());
    }

    #[test]
    fn test_render_pass_descriptor_timestamp_writes() {
        let desc = RenderPassDescriptor::new()
            .timestamp_writes(TimestampWrites::both(0, 1));
        assert!(desc.timestamp_writes.is_some());
    }

    #[test]
    fn test_render_pass_descriptor_with_occlusion_queries() {
        let desc = RenderPassDescriptor::new().with_occlusion_queries();
        assert!(desc.occlusion_query_enabled);
    }

    #[test]
    fn test_render_pass_descriptor_color_attachment_count() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new())
            .color_attachment(ColorAttachment::new());
        assert_eq!(desc.color_attachment_count(), 2);
    }

    #[test]
    fn test_render_pass_descriptor_has_depth_stencil() {
        let desc_no = RenderPassDescriptor::new();
        assert!(!desc_no.has_depth_stencil());

        let desc_yes = RenderPassDescriptor::new()
            .depth_stencil(DepthStencilAttachment::new());
        assert!(desc_yes.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_descriptor_has_timestamp_writes() {
        let desc_no = RenderPassDescriptor::new();
        assert!(!desc_no.has_timestamp_writes());

        let desc_yes = RenderPassDescriptor::new()
            .timestamp_writes(TimestampWrites::both(0, 1));
        assert!(desc_yes.has_timestamp_writes());
    }

    #[test]
    fn test_render_pass_descriptor_has_occlusion_queries() {
        let desc_no = RenderPassDescriptor::new();
        assert!(!desc_no.has_occlusion_queries());

        let desc_yes = RenderPassDescriptor::new().with_occlusion_queries();
        assert!(desc_yes.has_occlusion_queries());
    }

    #[test]
    fn test_render_pass_descriptor_display() {
        let desc = RenderPassDescriptor::new()
            .label("test")
            .color_attachment(ColorAttachment::new());
        let s = format!("{}", desc);
        assert!(s.contains("RenderPassDescriptor"));
    }

    // -------------------------------------------------------------------------
    // RenderPassBuilder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_builder_new() {
        let builder = RenderPassBuilder::new();
        let desc = builder.build();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_render_pass_builder_label() {
        let desc = RenderPassBuilder::new()
            .label("my_pass")
            .build();
        assert_eq!(desc.label, Some("my_pass".to_string()));
    }

    #[test]
    fn test_render_pass_builder_color_attachment() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_black())
            .build();
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_render_pass_builder_color_attachment_msaa() {
        let desc = RenderPassBuilder::new()
            .color_attachment_msaa(Operations::clear_black())
            .build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(att.has_resolve);
        }
    }

    #[test]
    fn test_render_pass_builder_empty_color_slot() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_black())
            .empty_color_slot()
            .build();
        assert_eq!(desc.color_attachments.len(), 2);
        assert!(desc.color_attachments[1].is_none());
    }

    #[test]
    fn test_render_pass_builder_depth_stencil() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(Some(Operations::clear_depth()), Some(Operations::clear_stencil()))
            .build();
        assert!(desc.depth_stencil_attachment.is_some());
    }

    #[test]
    fn test_render_pass_builder_depth_only() {
        let desc = RenderPassBuilder::new()
            .depth_only(Operations::clear_depth())
            .build();
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_some());
            assert!(att.stencil_ops.is_none());
        }
    }

    #[test]
    fn test_render_pass_builder_depth_read_only() {
        let desc = RenderPassBuilder::new()
            .depth_read_only()
            .build();
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    #[test]
    fn test_render_pass_builder_timestamp_writes() {
        let desc = RenderPassBuilder::new()
            .timestamp_writes(TimestampWrites::both(0, 1))
            .build();
        assert!(desc.timestamp_writes.is_some());
    }

    #[test]
    fn test_render_pass_builder_with_occlusion_queries() {
        let desc = RenderPassBuilder::new()
            .with_occlusion_queries()
            .build();
        assert!(desc.occlusion_query_enabled);
    }

    // -------------------------------------------------------------------------
    // RenderPassBuilder preset tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_builder_simple_color() {
        let desc = RenderPassBuilder::simple_color().build();
        assert_eq!(desc.label, Some("simple_color".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_simple_color_clear() {
        let desc = RenderPassBuilder::simple_color_clear(wgpu::Color::RED).build();
        assert_eq!(desc.color_attachments.len(), 1);
    }

    #[test]
    fn test_render_pass_builder_color_depth() {
        let desc = RenderPassBuilder::color_depth().build();
        assert_eq!(desc.label, Some("color_depth".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_color_depth_reverse_z() {
        let desc = RenderPassBuilder::color_depth_reverse_z().build();
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            if let Some(ops) = att.depth_ops {
                if let LoadOp::Clear(depth) = ops.load {
                    assert_eq!(depth, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_render_pass_builder_shadow_map() {
        let desc = RenderPassBuilder::shadow_map().build();
        assert_eq!(desc.label, Some("shadow_map".to_string()));
        assert!(desc.color_attachments.is_empty());
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_shadow_map_reverse_z() {
        let desc = RenderPassBuilder::shadow_map_reverse_z().build();
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            if let Some(ops) = att.depth_ops {
                if let LoadOp::Clear(depth) = ops.load {
                    assert_eq!(depth, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_render_pass_builder_gbuffer() {
        let desc = RenderPassBuilder::gbuffer().build();
        assert_eq!(desc.label, Some("gbuffer".to_string()));
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_gbuffer_reverse_z() {
        let desc = RenderPassBuilder::gbuffer_reverse_z().build();
        assert_eq!(desc.color_attachments.len(), 3);
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_post_process() {
        let desc = RenderPassBuilder::post_process().build();
        assert_eq!(desc.label, Some("post_process".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_render_pass_builder_transparent() {
        let desc = RenderPassBuilder::transparent().build();
        assert_eq!(desc.label, Some("transparent".to_string()));
        assert!(desc.has_depth_stencil());
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    #[test]
    fn test_render_pass_builder_fullscreen() {
        let desc = RenderPassBuilder::fullscreen().build();
        assert_eq!(desc.label, Some("fullscreen".to_string()));
        assert_eq!(desc.color_attachments.len(), 1);
        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_ui() {
        let desc = RenderPassBuilder::ui().build();
        assert_eq!(desc.label, Some("ui".to_string()));
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_render_pass_builder_msaa_resolve() {
        let desc = RenderPassBuilder::msaa_resolve().build();
        assert_eq!(desc.label, Some("msaa_resolve".to_string()));
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(att.has_resolve);
        }
    }

    #[test]
    fn test_render_pass_builder_depth_prepass() {
        let desc = RenderPassBuilder::depth_prepass().build();
        assert_eq!(desc.label, Some("depth_prepass".to_string()));
        assert!(desc.color_attachments.is_empty());
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_builder_stencil_only() {
        let desc = RenderPassBuilder::stencil_only().build();
        assert_eq!(desc.label, Some("stencil_only".to_string()));
        if let Some(att) = desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_none());
            assert!(att.stencil_ops.is_some());
        }
    }

    // -------------------------------------------------------------------------
    // RenderPassInfo tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_presets_count() {
        assert_eq!(RENDER_PASS_PRESETS.len(), 10);
    }

    #[test]
    fn test_get_preset_info_existing() {
        let info = get_preset_info("shadow_map");
        assert!(info.is_some());
        if let Some(info) = info {
            assert_eq!(info.name, "shadow_map");
            assert_eq!(info.color_count, 0);
            assert!(info.has_depth);
        }
    }

    #[test]
    fn test_get_preset_info_nonexistent() {
        let info = get_preset_info("nonexistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = preset_names().collect();
        assert!(names.contains(&"simple_color"));
        assert!(names.contains(&"shadow_map"));
        assert!(names.contains(&"gbuffer"));
    }

    #[test]
    fn test_render_pass_info_simple_color() {
        let info = get_preset_info("simple_color").unwrap();
        assert_eq!(info.color_count, 1);
        assert!(!info.has_depth);
        assert!(!info.has_stencil);
    }

    #[test]
    fn test_render_pass_info_color_depth() {
        let info = get_preset_info("color_depth").unwrap();
        assert_eq!(info.color_count, 1);
        assert!(info.has_depth);
    }

    #[test]
    fn test_render_pass_info_gbuffer() {
        let info = get_preset_info("gbuffer").unwrap();
        assert_eq!(info.color_count, 3);
        assert!(info.has_depth);
    }

    #[test]
    fn test_render_pass_info_msaa_resolve() {
        let info = get_preset_info("msaa_resolve").unwrap();
        assert!(info.has_resolve);
    }

    #[test]
    fn test_render_pass_info_stencil_only() {
        let info = get_preset_info("stencil_only").unwrap();
        assert!(!info.has_depth);
        assert!(info.has_stencil);
    }

    // -------------------------------------------------------------------------
    // Validation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_color_attachment_count_valid() {
        assert!(validate_color_attachment_count(0).is_ok());
        assert!(validate_color_attachment_count(4).is_ok());
        assert!(validate_color_attachment_count(8).is_ok());
    }

    #[test]
    fn test_validate_color_attachment_count_invalid() {
        let result = validate_color_attachment_count(9);
        assert!(result.is_err());
        if let Err(RenderPassError::TooManyColorAttachments { count, max }) = result {
            assert_eq!(count, 9);
            assert_eq!(max, 8);
        }
    }

    #[test]
    fn test_validate_descriptor_valid_color_only() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new());
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_descriptor_valid_depth_only() {
        let desc = RenderPassDescriptor::new()
            .depth_stencil(DepthStencilAttachment::new());
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_descriptor_no_attachments() {
        let desc = RenderPassDescriptor::new();
        let result = validate_descriptor(&desc);
        assert!(matches!(result, Err(RenderPassError::NoAttachments)));
    }

    #[test]
    fn test_validate_descriptor_too_many_colors() {
        let mut desc = RenderPassDescriptor::new();
        for _ in 0..9 {
            desc.color_attachments.push(Some(ColorAttachment::new()));
        }
        let result = validate_descriptor(&desc);
        assert!(matches!(result, Err(RenderPassError::TooManyColorAttachments { .. })));
    }

    // -------------------------------------------------------------------------
    // Error tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_error_display_too_many() {
        let err = RenderPassError::TooManyColorAttachments { count: 10, max: 8 };
        let s = format!("{}", err);
        assert!(s.contains("10"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_render_pass_error_display_no_attachments() {
        let err = RenderPassError::NoAttachments;
        let s = format!("{}", err);
        assert!(s.contains("at least one"));
    }

    #[test]
    fn test_render_pass_error_display_invalid_timestamp() {
        let err = RenderPassError::InvalidTimestampIndex {
            index: 5,
            query_set_size: 4,
        };
        let s = format!("{}", err);
        assert!(s.contains("5"));
        assert!(s.contains("4"));
    }

    #[test]
    fn test_render_pass_error_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(RenderPassError::NoAttachments);
        assert!(!err.to_string().is_empty());
    }

    // -------------------------------------------------------------------------
    // Constants tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_max_color_attachments() {
        assert_eq!(MAX_COLOR_ATTACHMENTS, 8);
    }

    #[test]
    fn test_default_clear_color() {
        assert_eq!(DEFAULT_CLEAR_COLOR.r, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.g, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.b, 0.0);
        assert_eq!(DEFAULT_CLEAR_COLOR.a, 1.0);
    }

    #[test]
    fn test_default_clear_depth() {
        assert_eq!(DEFAULT_CLEAR_DEPTH, 1.0);
    }

    #[test]
    fn test_default_clear_stencil() {
        assert_eq!(DEFAULT_CLEAR_STENCIL, 0);
    }

    // -------------------------------------------------------------------------
    // Clone and Debug tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_clone() {
        let op = LoadOp::Clear(wgpu::Color::RED);
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_store_op_clone() {
        let op = StoreOp::Store;
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_operations_clone() {
        let ops = Operations::clear_black();
        let cloned = ops.clone();
        assert_eq!(ops, cloned);
    }

    #[test]
    fn test_color_attachment_clone() {
        let att = ColorAttachment::new();
        let cloned = att.clone();
        assert_eq!(att.store_op, cloned.store_op);
    }

    #[test]
    fn test_depth_stencil_attachment_clone() {
        let att = DepthStencilAttachment::new();
        let cloned = att.clone();
        assert_eq!(att, cloned);
    }

    #[test]
    fn test_timestamp_writes_clone() {
        let ts = TimestampWrites::both(0, 1);
        let cloned = ts.clone();
        assert_eq!(ts, cloned);
    }

    #[test]
    fn test_render_pass_descriptor_clone() {
        let desc = RenderPassDescriptor::new().label("test");
        let cloned = desc.clone();
        assert_eq!(desc.label, cloned.label);
    }

    #[test]
    fn test_render_pass_builder_clone() {
        let builder = RenderPassBuilder::new().label("test");
        let cloned = builder.clone();
        assert_eq!(builder.build().label, cloned.build().label);
    }

    #[test]
    fn test_debug_implementations() {
        // Verify Debug is implemented
        let _ = format!("{:?}", LoadOp::Clear(1.0f32));
        let _ = format!("{:?}", StoreOp::Store);
        let _ = format!("{:?}", Operations::clear_depth());
        let _ = format!("{:?}", ColorAttachment::new());
        let _ = format!("{:?}", DepthStencilAttachment::new());
        let _ = format!("{:?}", TimestampWrites::new());
        let _ = format!("{:?}", OcclusionQuerySet::new());
        let _ = format!("{:?}", RenderPassDescriptor::new());
        let _ = format!("{:?}", RenderPassBuilder::new());
        let _ = format!("{:?}", RenderPassError::NoAttachments);
    }

    // -------------------------------------------------------------------------
    // Additional Descriptor construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_with_all_features() {
        let desc = RenderPassDescriptor::new()
            .label("full_featured_pass")
            .color_attachment(ColorAttachment::with_clear(wgpu::Color::RED))
            .color_attachment(ColorAttachment::with_load())
            .depth_stencil(DepthStencilAttachment::depth_stencil(1.0, 0))
            .timestamp_writes(TimestampWrites::both(0, 1))
            .with_occlusion_queries();

        assert_eq!(desc.label, Some("full_featured_pass".to_string()));
        assert_eq!(desc.color_attachment_count(), 2);
        assert!(desc.has_depth_stencil());
        assert!(desc.has_timestamp_writes());
        assert!(desc.has_occlusion_queries());
    }

    #[test]
    fn test_descriptor_label_with_string() {
        let label = String::from("dynamic_label");
        let desc = RenderPassDescriptor::new().label(label);
        assert_eq!(desc.label, Some("dynamic_label".to_string()));
    }

    #[test]
    fn test_descriptor_label_with_str() {
        let desc = RenderPassDescriptor::new().label("static_label");
        assert_eq!(desc.label, Some("static_label".to_string()));
    }

    #[test]
    fn test_descriptor_empty_color_attachments() {
        let desc = RenderPassDescriptor::new();
        assert!(desc.color_attachments.is_empty());
        assert_eq!(desc.color_attachment_count(), 0);
    }

    #[test]
    fn test_descriptor_single_color_attachment() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new());
        assert_eq!(desc.color_attachment_count(), 1);
        assert!(desc.color_attachments[0].is_some());
    }

    #[test]
    fn test_descriptor_max_color_attachments() {
        let mut desc = RenderPassDescriptor::new();
        for _ in 0..MAX_COLOR_ATTACHMENTS {
            desc = desc.color_attachment(ColorAttachment::new());
        }
        assert_eq!(desc.color_attachment_count(), 8);
    }

    #[test]
    fn test_descriptor_mixed_color_slots() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new())
            .empty_color_slot()
            .empty_color_slot()
            .color_attachment(ColorAttachment::new());
        assert_eq!(desc.color_attachment_count(), 4);
        assert!(desc.color_attachments[0].is_some());
        assert!(desc.color_attachments[1].is_none());
        assert!(desc.color_attachments[2].is_none());
        assert!(desc.color_attachments[3].is_some());
    }

    // -------------------------------------------------------------------------
    // Additional ColorAttachment tests (MRT scenarios)
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_attachment_gbuffer_albedo() {
        let albedo = ColorAttachment::with_clear(wgpu::Color::BLACK);
        assert!(matches!(albedo.load_op, LoadOp::Clear(_)));
        assert_eq!(albedo.store_op, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_gbuffer_normal() {
        let normal = ColorAttachment::with_clear(wgpu::Color {
            r: 0.5, g: 0.5, b: 1.0, a: 1.0
        });
        if let LoadOp::Clear(color) = normal.load_op {
            assert_eq!(color.b, 1.0);
        }
    }

    #[test]
    fn test_color_attachment_gbuffer_material() {
        let material = ColorAttachment::new()
            .clear_color(wgpu::Color::BLACK)
            .store(StoreOp::Store);
        assert_eq!(material.store_op, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_transient() {
        let transient = ColorAttachment::new()
            .clear_color(wgpu::Color::BLACK)
            .store(StoreOp::Discard);
        assert_eq!(transient.store_op, StoreOp::Discard);
    }

    #[test]
    fn test_color_attachment_chain_methods() {
        let att = ColorAttachment::new()
            .clear_color(wgpu::Color::RED)
            .with_resolve()
            .store(StoreOp::Store);
        assert!(att.has_resolve);
        assert_eq!(att.store_op, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_ops_conversion() {
        let att = ColorAttachment::with_clear(wgpu::Color::GREEN);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    // -------------------------------------------------------------------------
    // Additional DepthStencilAttachment tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_stencil_combined_clear_values() {
        let att = DepthStencilAttachment::depth_stencil(0.5, 128);
        if let Some(d_ops) = att.depth_ops {
            if let LoadOp::Clear(d) = d_ops.load {
                assert_eq!(d, 0.5);
            }
        }
        if let Some(s_ops) = att.stencil_ops {
            if let LoadOp::Clear(s) = s_ops.load {
                assert_eq!(s, 128);
            }
        }
    }

    #[test]
    fn test_depth_stencil_depth_writable_stencil_readonly() {
        let att = DepthStencilAttachment::depth_only(1.0);
        assert!(att.is_depth_writable());
        assert!(!att.is_stencil_writable());
        assert!(!att.is_read_only());
    }

    #[test]
    fn test_depth_stencil_stencil_writable_depth_readonly() {
        let att = DepthStencilAttachment::stencil_only(0);
        assert!(!att.is_depth_writable());
        assert!(att.is_stencil_writable());
        assert!(!att.is_read_only());
    }

    #[test]
    fn test_depth_stencil_chain_depth_then_stencil() {
        let att = DepthStencilAttachment::new()
            .depth_clear(0.0)
            .stencil_clear(255);
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_some());
    }

    #[test]
    fn test_depth_stencil_chain_remove_ops() {
        let att = DepthStencilAttachment::depth_stencil(1.0, 0)
            .no_depth_ops()
            .no_stencil_ops();
        assert!(att.is_read_only());
    }

    #[test]
    fn test_depth_stencil_load_both() {
        let att = DepthStencilAttachment::new()
            .depth_load()
            .stencil_load();
        if let Some(d_ops) = att.depth_ops {
            assert!(matches!(d_ops.load, LoadOp::Load));
        }
        if let Some(s_ops) = att.stencil_ops {
            assert!(matches!(s_ops.load, LoadOp::Load));
        }
    }

    #[test]
    fn test_depth_stencil_display_read_only() {
        let att = DepthStencilAttachment::depth_read_only();
        let s = format!("{}", att);
        assert!(s.contains("read-only"));
    }

    // -------------------------------------------------------------------------
    // Additional TimestampWrites tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timestamp_writes_chain_both() {
        let ts = TimestampWrites::new()
            .beginning(10)
            .end(11);
        assert_eq!(ts.beginning_of_pass_write_index, Some(10));
        assert_eq!(ts.end_of_pass_write_index, Some(11));
    }

    #[test]
    fn test_timestamp_writes_overwrite_beginning() {
        let ts = TimestampWrites::beginning_only(5)
            .beginning(10);
        assert_eq!(ts.beginning_of_pass_write_index, Some(10));
    }

    #[test]
    fn test_timestamp_writes_overwrite_end() {
        let ts = TimestampWrites::end_only(5)
            .end(20);
        assert_eq!(ts.end_of_pass_write_index, Some(20));
    }

    #[test]
    fn test_timestamp_writes_large_indices() {
        let ts = TimestampWrites::both(u32::MAX - 1, u32::MAX);
        assert_eq!(ts.beginning_of_pass_write_index, Some(u32::MAX - 1));
        assert_eq!(ts.end_of_pass_write_index, Some(u32::MAX));
    }

    #[test]
    fn test_timestamp_writes_same_index() {
        let ts = TimestampWrites::both(0, 0);
        assert_eq!(ts.beginning_of_pass_write_index, Some(0));
        assert_eq!(ts.end_of_pass_write_index, Some(0));
    }

    // -------------------------------------------------------------------------
    // Additional OcclusionQuerySet tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_occlusion_query_set_toggle() {
        let oqs = OcclusionQuerySet::new()
            .disable()
            .enable();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_occlusion_query_set_default_is_enabled() {
        let oqs = OcclusionQuerySet::default();
        assert!(oqs.enabled);
    }

    // -------------------------------------------------------------------------
    // Builder fluent API comprehensive tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_chained_all_methods() {
        let desc = RenderPassBuilder::new()
            .label("comprehensive_pass")
            .color_attachment(Operations::clear_black())
            .color_attachment(Operations::clear_white())
            .empty_color_slot()
            .color_attachment(Operations::load_store())
            .depth_stencil(Some(Operations::clear_depth()), Some(Operations::clear_stencil()))
            .timestamp_writes(TimestampWrites::both(0, 1))
            .with_occlusion_queries()
            .build();

        assert_eq!(desc.label, Some("comprehensive_pass".to_string()));
        assert_eq!(desc.color_attachment_count(), 4);
        assert!(desc.has_depth_stencil());
        assert!(desc.has_timestamp_writes());
        assert!(desc.has_occlusion_queries());
    }

    #[test]
    fn test_builder_multiple_color_attachments() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_rgba(1.0, 0.0, 0.0, 1.0))
            .color_attachment(Operations::clear_rgba(0.0, 1.0, 0.0, 1.0))
            .color_attachment(Operations::clear_rgba(0.0, 0.0, 1.0, 1.0))
            .color_attachment(Operations::clear_rgba(1.0, 1.0, 0.0, 1.0))
            .build();

        assert_eq!(desc.color_attachment_count(), 4);
    }

    #[test]
    fn test_builder_depth_ops_only() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(Some(Operations::clear(0.5)), None)
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_some());
            assert!(att.stencil_ops.is_none());
        }
    }

    #[test]
    fn test_builder_stencil_ops_only() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(None, Some(Operations::clear(128u32)))
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_none());
            assert!(att.stencil_ops.is_some());
        }
    }

    #[test]
    fn test_builder_no_depth_stencil() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_black())
            .build();

        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_builder_depth_only_method() {
        let desc = RenderPassBuilder::new()
            .depth_only(Operations::clear(0.0))
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            assert!(att.depth_ops.is_some());
            assert!(att.stencil_ops.is_none());
            if let Some(ops) = &att.depth_ops {
                if let LoadOp::Clear(d) = ops.load {
                    assert_eq!(d, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_builder_depth_read_only_method() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::load_store())
            .depth_read_only()
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    // -------------------------------------------------------------------------
    // Additional preset verification tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_preset_post_process_load_op() {
        let desc = RenderPassBuilder::post_process().build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
            assert_eq!(att.store_op, StoreOp::Store);
        }
    }

    #[test]
    fn test_preset_transparent_depth_readonly() {
        let desc = RenderPassBuilder::transparent().build();
        if let Some(att) = &desc.depth_stencil_attachment {
            assert!(att.is_read_only());
        }
    }

    #[test]
    fn test_preset_gbuffer_three_color_targets() {
        let desc = RenderPassBuilder::gbuffer().build();
        assert_eq!(desc.color_attachment_count(), 3);
        for i in 0..3 {
            assert!(desc.color_attachments[i].is_some());
        }
    }

    #[test]
    fn test_preset_shadow_map_no_color() {
        let desc = RenderPassBuilder::shadow_map().build();
        assert!(desc.color_attachments.is_empty());
        assert!(desc.has_depth_stencil());
    }

    #[test]
    fn test_preset_ui_loads_existing() {
        let desc = RenderPassBuilder::ui().build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_preset_fullscreen_clear() {
        let desc = RenderPassBuilder::fullscreen().build();
        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Clear(_)));
        }
    }

    // -------------------------------------------------------------------------
    // Operations combination tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_all_combinations_color() {
        // Clear + Store
        let ops1 = Operations::clear(wgpu::Color::BLACK);
        assert!(matches!(ops1.load, LoadOp::Clear(_)));
        assert_eq!(ops1.store, StoreOp::Store);

        // Clear + Discard
        let ops2 = Operations::clear_discard(wgpu::Color::BLACK);
        assert!(matches!(ops2.load, LoadOp::Clear(_)));
        assert_eq!(ops2.store, StoreOp::Discard);

        // Load + Store
        let ops3: Operations<wgpu::Color> = Operations::load_store();
        assert!(matches!(ops3.load, LoadOp::Load));
        assert_eq!(ops3.store, StoreOp::Store);

        // Load + Discard
        let ops4: Operations<wgpu::Color> = Operations::load_discard();
        assert!(matches!(ops4.load, LoadOp::Load));
        assert_eq!(ops4.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_all_combinations_depth() {
        let ops1 = Operations::clear(1.0f32);
        let ops2 = Operations::clear_discard(1.0f32);
        let ops3: Operations<f32> = Operations::load_store();
        let ops4: Operations<f32> = Operations::load_discard();

        assert!(matches!(ops1.load, LoadOp::Clear(1.0)));
        assert_eq!(ops2.store, StoreOp::Discard);
        assert!(matches!(ops3.load, LoadOp::Load));
        assert_eq!(ops4.store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_all_combinations_stencil() {
        let ops1 = Operations::clear(0u32);
        let ops2 = Operations::clear_discard(255u32);
        let ops3: Operations<u32> = Operations::load_store();
        let ops4: Operations<u32> = Operations::load_discard();

        assert!(matches!(ops1.load, LoadOp::Clear(0)));
        assert!(matches!(ops2.load, LoadOp::Clear(255)));
        assert!(matches!(ops3.load, LoadOp::Load));
        assert!(matches!(ops4.load, LoadOp::Load));
    }

    // -------------------------------------------------------------------------
    // RenderPassInfo comprehensive tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_info_all_presets_have_name() {
        for preset in &RENDER_PASS_PRESETS {
            assert!(!preset.name.is_empty());
        }
    }

    #[test]
    fn test_render_pass_info_all_presets_have_description() {
        for preset in &RENDER_PASS_PRESETS {
            assert!(!preset.description.is_empty());
        }
    }

    #[test]
    fn test_render_pass_info_all_presets_have_use_cases() {
        for preset in &RENDER_PASS_PRESETS {
            assert!(!preset.use_cases.is_empty());
        }
    }

    #[test]
    fn test_render_pass_info_post_process() {
        let info = get_preset_info("post_process").unwrap();
        assert!(!info.has_depth);
        assert!(!info.has_stencil);
    }

    #[test]
    fn test_render_pass_info_transparent() {
        let info = get_preset_info("transparent").unwrap();
        assert!(info.has_depth);
        assert_eq!(info.color_count, 1);
    }

    #[test]
    fn test_render_pass_info_depth_prepass() {
        let info = get_preset_info("depth_prepass").unwrap();
        assert_eq!(info.color_count, 0);
        assert!(info.has_depth);
    }

    // -------------------------------------------------------------------------
    // Validation boundary tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_exactly_max_attachments() {
        assert!(validate_color_attachment_count(MAX_COLOR_ATTACHMENTS).is_ok());
    }

    #[test]
    fn test_validate_one_over_max() {
        assert!(validate_color_attachment_count(MAX_COLOR_ATTACHMENTS + 1).is_err());
    }

    #[test]
    fn test_validate_descriptor_with_only_empty_slots() {
        let desc = RenderPassDescriptor::new()
            .empty_color_slot()
            .empty_color_slot();
        // Has slots but they're all None, and no depth
        // The validation currently only checks if the array is empty or has depth
        // Since color_attachments is non-empty, it passes validation
        // (wgpu may have stricter validation at runtime)
        let result = validate_descriptor(&desc);
        assert!(result.is_ok()); // Non-empty array passes basic validation
    }

    #[test]
    fn test_validate_descriptor_color_and_depth() {
        let desc = RenderPassDescriptor::new()
            .color_attachment(ColorAttachment::new())
            .depth_stencil(DepthStencilAttachment::new());
        assert!(validate_descriptor(&desc).is_ok());
    }

    // -------------------------------------------------------------------------
    // Error equality and hash tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_error_equality() {
        let err1 = RenderPassError::TooManyColorAttachments { count: 9, max: 8 };
        let err2 = RenderPassError::TooManyColorAttachments { count: 9, max: 8 };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_render_pass_error_inequality() {
        let err1 = RenderPassError::TooManyColorAttachments { count: 9, max: 8 };
        let err2 = RenderPassError::TooManyColorAttachments { count: 10, max: 8 };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_render_pass_error_no_attachments_equality() {
        let err1 = RenderPassError::NoAttachments;
        let err2 = RenderPassError::NoAttachments;
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_render_pass_error_invalid_timestamp_equality() {
        let err1 = RenderPassError::InvalidTimestampIndex { index: 5, query_set_size: 4 };
        let err2 = RenderPassError::InvalidTimestampIndex { index: 5, query_set_size: 4 };
        assert_eq!(err1, err2);
    }

    // -------------------------------------------------------------------------
    // Thread safety tests (Send + Sync)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<LoadOp<f32>>();
        assert_send::<LoadOp<u32>>();
        assert_send::<LoadOp<wgpu::Color>>();
    }

    #[test]
    fn test_load_op_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<LoadOp<f32>>();
        assert_sync::<LoadOp<u32>>();
        assert_sync::<LoadOp<wgpu::Color>>();
    }

    #[test]
    fn test_store_op_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StoreOp>();
    }

    #[test]
    fn test_operations_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Operations<f32>>();
        assert_send_sync::<Operations<u32>>();
        assert_send_sync::<Operations<wgpu::Color>>();
    }

    #[test]
    fn test_color_attachment_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ColorAttachment>();
    }

    #[test]
    fn test_depth_stencil_attachment_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DepthStencilAttachment>();
    }

    #[test]
    fn test_timestamp_writes_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TimestampWrites>();
    }

    #[test]
    fn test_occlusion_query_set_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OcclusionQuerySet>();
    }

    #[test]
    fn test_render_pass_descriptor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RenderPassDescriptor>();
    }

    #[test]
    fn test_render_pass_builder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RenderPassBuilder>();
    }

    #[test]
    fn test_render_pass_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RenderPassError>();
    }

    // -------------------------------------------------------------------------
    // PartialEq tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_partial_eq() {
        assert_eq!(LoadOp::Clear(1.0f32), LoadOp::Clear(1.0f32));
        assert_ne!(LoadOp::Clear(1.0f32), LoadOp::Clear(0.5f32));
        assert_ne!(LoadOp::Clear(1.0f32), LoadOp::<f32>::Load);
        assert_eq!(LoadOp::<f32>::Load, LoadOp::<f32>::Load);
    }

    #[test]
    fn test_store_op_partial_eq() {
        assert_eq!(StoreOp::Store, StoreOp::Store);
        assert_eq!(StoreOp::Discard, StoreOp::Discard);
        assert_ne!(StoreOp::Store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_partial_eq() {
        let ops1 = Operations::clear(1.0f32);
        let ops2 = Operations::clear(1.0f32);
        let ops3 = Operations::clear(0.5f32);
        assert_eq!(ops1, ops2);
        assert_ne!(ops1, ops3);
    }

    #[test]
    fn test_depth_stencil_attachment_partial_eq() {
        let att1 = DepthStencilAttachment::depth_only(1.0);
        let att2 = DepthStencilAttachment::depth_only(1.0);
        let att3 = DepthStencilAttachment::depth_only(0.5);
        assert_eq!(att1, att2);
        assert_ne!(att1, att3);
    }

    #[test]
    fn test_timestamp_writes_partial_eq() {
        let ts1 = TimestampWrites::both(0, 1);
        let ts2 = TimestampWrites::both(0, 1);
        let ts3 = TimestampWrites::both(0, 2);
        assert_eq!(ts1, ts2);
        assert_ne!(ts1, ts3);
    }

    #[test]
    fn test_occlusion_query_set_partial_eq() {
        let oqs1 = OcclusionQuerySet::new();
        let oqs2 = OcclusionQuerySet::new();
        let oqs3 = OcclusionQuerySet::disabled();
        assert_eq!(oqs1, oqs2);
        assert_ne!(oqs1, oqs3);
    }

    // -------------------------------------------------------------------------
    // Hash tests (StoreOp, TimestampWrites, OcclusionQuerySet, RenderPassInfo)
    // -------------------------------------------------------------------------

    #[test]
    fn test_store_op_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StoreOp::Store);
        set.insert(StoreOp::Discard);
        assert_eq!(set.len(), 2);
        set.insert(StoreOp::Store);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_timestamp_writes_eq_reflexive() {
        let ts = TimestampWrites::both(0, 1);
        assert_eq!(ts, ts);
    }

    #[test]
    fn test_occlusion_query_set_eq_reflexive() {
        let oqs = OcclusionQuerySet::new();
        assert_eq!(oqs, oqs);
    }

    // -------------------------------------------------------------------------
    // Default trait tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_default_trait() {
        let op: LoadOp<f32> = Default::default();
        assert!(matches!(op, LoadOp::Clear(0.0)));
    }

    #[test]
    fn test_store_op_default_trait() {
        let op: StoreOp = Default::default();
        assert_eq!(op, StoreOp::Store);
    }

    #[test]
    fn test_operations_default_trait() {
        let ops: Operations<f32> = Default::default();
        assert!(matches!(ops.load, LoadOp::Clear(0.0)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_default_trait() {
        let att: ColorAttachment = Default::default();
        assert_eq!(att.clear_color, DEFAULT_CLEAR_COLOR);
        assert_eq!(att.store_op, StoreOp::Store);
        assert!(!att.has_resolve);
    }

    #[test]
    fn test_depth_stencil_attachment_default_trait() {
        let att: DepthStencilAttachment = Default::default();
        assert!(att.depth_ops.is_some());
        assert!(att.stencil_ops.is_none());
    }

    #[test]
    fn test_timestamp_writes_default_trait() {
        let ts: TimestampWrites = Default::default();
        assert!(ts.beginning_of_pass_write_index.is_none());
        assert!(ts.end_of_pass_write_index.is_none());
    }

    #[test]
    fn test_occlusion_query_set_default_trait() {
        let oqs: OcclusionQuerySet = Default::default();
        assert!(oqs.enabled);
    }

    #[test]
    fn test_render_pass_descriptor_default_trait() {
        let desc: RenderPassDescriptor = Default::default();
        assert!(desc.label.is_none());
        assert!(desc.color_attachments.is_empty());
        assert!(desc.depth_stencil_attachment.is_none());
        assert!(desc.timestamp_writes.is_none());
        assert!(!desc.occlusion_query_enabled);
    }

    #[test]
    fn test_render_pass_builder_default_trait() {
        let builder: RenderPassBuilder = Default::default();
        let desc = builder.build();
        assert!(desc.label.is_none());
    }

    // -------------------------------------------------------------------------
    // Copy trait tests (for types that implement Copy)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_copy() {
        let op = LoadOp::Clear(1.0f32);
        let copied = op;
        // Both should still be valid (copy, not move)
        assert_eq!(op, copied);
    }

    #[test]
    fn test_store_op_copy() {
        let op = StoreOp::Store;
        let copied = op;
        assert_eq!(op, copied);
    }

    #[test]
    fn test_operations_copy() {
        let ops = Operations::clear(1.0f32);
        let copied = ops;
        assert_eq!(ops, copied);
    }

    #[test]
    fn test_depth_stencil_attachment_copy() {
        let att = DepthStencilAttachment::depth_only(1.0);
        let copied = att;
        assert_eq!(att, copied);
    }

    #[test]
    fn test_timestamp_writes_copy() {
        let ts = TimestampWrites::both(0, 1);
        let copied = ts;
        assert_eq!(ts, copied);
    }

    #[test]
    fn test_occlusion_query_set_copy() {
        let oqs = OcclusionQuerySet::new();
        let copied = oqs;
        assert_eq!(oqs, copied);
    }

    // -------------------------------------------------------------------------
    // Edge case and boundary tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_with_zero_depth() {
        let ops = Operations::clear(0.0f32);
        if let LoadOp::Clear(d) = ops.load {
            assert_eq!(d, 0.0);
        }
    }

    #[test]
    fn test_operations_with_max_depth() {
        let ops = Operations::clear(f32::MAX);
        if let LoadOp::Clear(d) = ops.load {
            assert_eq!(d, f32::MAX);
        }
    }

    #[test]
    fn test_operations_with_infinity_depth() {
        let ops = Operations::clear(f32::INFINITY);
        if let LoadOp::Clear(d) = ops.load {
            assert!(d.is_infinite());
        }
    }

    #[test]
    fn test_operations_with_max_stencil() {
        let ops = Operations::clear(u32::MAX);
        if let LoadOp::Clear(s) = ops.load {
            assert_eq!(s, u32::MAX);
        }
    }

    #[test]
    fn test_color_with_negative_components() {
        // wgpu::Color uses f64, which can be negative (though unusual)
        let ops = Operations::clear(wgpu::Color {
            r: -1.0, g: -0.5, b: 0.0, a: 1.0
        });
        if let LoadOp::Clear(c) = ops.load {
            assert_eq!(c.r, -1.0);
        }
    }

    #[test]
    fn test_color_with_values_over_one() {
        // HDR colors can exceed 1.0
        let ops = Operations::clear(wgpu::Color {
            r: 2.0, g: 3.0, b: 4.0, a: 1.0
        });
        if let LoadOp::Clear(c) = ops.load {
            assert_eq!(c.r, 2.0);
        }
    }

    #[test]
    fn test_timestamp_writes_zero_indices() {
        let ts = TimestampWrites::both(0, 0);
        assert!(ts.is_enabled());
    }

    #[test]
    fn test_descriptor_label_empty_string() {
        let desc = RenderPassDescriptor::new().label("");
        assert_eq!(desc.label, Some(String::new()));
    }

    #[test]
    fn test_descriptor_label_unicode() {
        let desc = RenderPassDescriptor::new().label("render_pass_");
        assert!(desc.label.unwrap().contains(""));
    }

    #[test]
    fn test_descriptor_label_long_string() {
        let long_label = "a".repeat(1000);
        let desc = RenderPassDescriptor::new().label(long_label.clone());
        assert_eq!(desc.label, Some(long_label));
    }

    // =========================================================================
    // WHITEBOX TESTS: T-WGPU-P3.8.2 (Load/Store Operations)
    // =========================================================================

    // -------------------------------------------------------------------------
    // LoadOp Edge Cases: Extreme depth values
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_clear_depth_nan() {
        let op = LoadOp::Clear(f32::NAN);
        if let LoadOp::Clear(d) = op {
            assert!(d.is_nan());
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_neg_infinity() {
        let op = LoadOp::Clear(f32::NEG_INFINITY);
        if let LoadOp::Clear(d) = op {
            assert!(d.is_infinite());
            assert!(d.is_sign_negative());
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_pos_infinity() {
        let op = LoadOp::Clear(f32::INFINITY);
        if let LoadOp::Clear(d) = op {
            assert!(d.is_infinite());
            assert!(d.is_sign_positive());
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_min() {
        let op = LoadOp::Clear(f32::MIN);
        if let LoadOp::Clear(d) = op {
            assert_eq!(d, f32::MIN);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_max() {
        let op = LoadOp::Clear(f32::MAX);
        if let LoadOp::Clear(d) = op {
            assert_eq!(d, f32::MAX);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_epsilon() {
        let op = LoadOp::Clear(f32::EPSILON);
        if let LoadOp::Clear(d) = op {
            assert_eq!(d, f32::EPSILON);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_min_positive() {
        let op = LoadOp::Clear(f32::MIN_POSITIVE);
        if let LoadOp::Clear(d) = op {
            assert_eq!(d, f32::MIN_POSITIVE);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_negative_zero() {
        let op = LoadOp::Clear(-0.0f32);
        if let LoadOp::Clear(d) = op {
            // -0.0 == 0.0 in IEEE 754
            assert_eq!(d, 0.0);
            // But we can check the sign bit
            assert!(d.is_sign_negative() || d == 0.0);
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_color_extreme_values() {
        let color = wgpu::Color {
            r: f64::MAX,
            g: f64::MIN,
            b: f64::INFINITY,
            a: f64::NEG_INFINITY,
        };
        let op = LoadOp::Clear(color);
        if let LoadOp::Clear(c) = op {
            assert_eq!(c.r, f64::MAX);
            assert_eq!(c.g, f64::MIN);
            assert!(c.b.is_infinite());
            assert!(c.a.is_infinite());
        } else {
            panic!("Expected Clear");
        }
    }

    #[test]
    fn test_load_op_clear_color_nan_components() {
        let color = wgpu::Color {
            r: f64::NAN,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let op = LoadOp::Clear(color);
        if let LoadOp::Clear(c) = op {
            assert!(c.r.is_nan());
        } else {
            panic!("Expected Clear");
        }
    }

    // -------------------------------------------------------------------------
    // LoadOp to_wgpu Conversion Tests (Load variant)
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_depth_load_to_wgpu() {
        let op: LoadOp<f32> = LoadOp::Load;
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
    }

    #[test]
    fn test_load_op_stencil_load_to_wgpu() {
        let op: LoadOp<u32> = LoadOp::Load;
        let wgpu_op = op.to_wgpu();
        assert!(matches!(wgpu_op, wgpu::LoadOp::Load));
    }

    #[test]
    fn test_load_op_clear_wgpu_color_preserved() {
        let color = wgpu::Color { r: 0.25, g: 0.5, b: 0.75, a: 0.125 };
        let op = LoadOp::Clear(color);
        let wgpu_op = op.to_wgpu();
        if let wgpu::LoadOp::Clear(c) = wgpu_op {
            assert_eq!(c.r, 0.25);
            assert_eq!(c.g, 0.5);
            assert_eq!(c.b, 0.75);
            assert_eq!(c.a, 0.125);
        } else {
            panic!("Expected wgpu::LoadOp::Clear");
        }
    }

    #[test]
    fn test_load_op_clear_depth_wgpu_value_preserved() {
        let op = LoadOp::Clear(0.123456f32);
        let wgpu_op = op.to_wgpu();
        if let wgpu::LoadOp::Clear(d) = wgpu_op {
            assert!((d - 0.123456).abs() < f32::EPSILON);
        } else {
            panic!("Expected wgpu::LoadOp::Clear");
        }
    }

    #[test]
    fn test_load_op_clear_stencil_wgpu_value_preserved() {
        let op = LoadOp::Clear(0xDEADBEEFu32);
        let wgpu_op = op.to_wgpu();
        if let wgpu::LoadOp::Clear(s) = wgpu_op {
            assert_eq!(s, 0xDEADBEEF);
        } else {
            panic!("Expected wgpu::LoadOp::Clear");
        }
    }

    // -------------------------------------------------------------------------
    // StoreOp Edge Cases and From trait
    // -------------------------------------------------------------------------

    #[test]
    fn test_store_op_from_trait_store() {
        let wgpu_op: wgpu::StoreOp = StoreOp::Store.into();
        assert!(matches!(wgpu_op, wgpu::StoreOp::Store));
    }

    #[test]
    fn test_store_op_from_trait_discard() {
        let wgpu_op: wgpu::StoreOp = StoreOp::Discard.into();
        assert!(matches!(wgpu_op, wgpu::StoreOp::Discard));
    }

    #[test]
    fn test_store_op_to_wgpu_store() {
        assert!(matches!(StoreOp::Store.to_wgpu(), wgpu::StoreOp::Store));
    }

    #[test]
    fn test_store_op_to_wgpu_discard() {
        assert!(matches!(StoreOp::Discard.to_wgpu(), wgpu::StoreOp::Discard));
    }

    // -------------------------------------------------------------------------
    // Operations<T> Generic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_generic_new_with_various_types() {
        // f32 depth
        let ops_f32 = Operations::new(LoadOp::Clear(0.5f32), StoreOp::Discard);
        assert!(matches!(ops_f32.load, LoadOp::Clear(0.5)));
        assert_eq!(ops_f32.store, StoreOp::Discard);

        // u32 stencil
        let ops_u32 = Operations::new(LoadOp::Clear(128u32), StoreOp::Store);
        assert!(matches!(ops_u32.load, LoadOp::Clear(128)));
        assert_eq!(ops_u32.store, StoreOp::Store);

        // wgpu::Color
        let ops_color = Operations::new(LoadOp::Clear(wgpu::Color::GREEN), StoreOp::Store);
        if let LoadOp::Clear(c) = ops_color.load {
            assert_eq!(c, wgpu::Color::GREEN);
        }
    }

    #[test]
    fn test_operations_default_for_u32() {
        let ops: Operations<u32> = Operations::default();
        assert!(matches!(ops.load, LoadOp::Clear(0)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_default_for_f32() {
        let ops: Operations<f32> = Operations::default();
        assert!(matches!(ops.load, LoadOp::Clear(0.0)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear_with_custom_type() {
        #[derive(Debug, Clone, Copy, Default, PartialEq)]
        struct CustomClearValue(i32);

        let ops = Operations::clear(CustomClearValue(42));
        assert!(matches!(ops.load, LoadOp::Clear(CustomClearValue(42))));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_to_wgpu_color_with_all_combinations() {
        // Clear + Store
        let ops1 = Operations::new(LoadOp::Clear(wgpu::Color::RED), StoreOp::Store);
        let wgpu1 = ops1.to_wgpu();
        assert!(matches!(wgpu1.load, wgpu::LoadOp::Clear(_)));
        assert!(matches!(wgpu1.store, wgpu::StoreOp::Store));

        // Clear + Discard
        let ops2 = Operations::new(LoadOp::Clear(wgpu::Color::BLUE), StoreOp::Discard);
        let wgpu2 = ops2.to_wgpu();
        assert!(matches!(wgpu2.load, wgpu::LoadOp::Clear(_)));
        assert!(matches!(wgpu2.store, wgpu::StoreOp::Discard));

        // Load + Store
        let ops3 = Operations::<wgpu::Color>::load_store();
        let wgpu3 = ops3.to_wgpu();
        assert!(matches!(wgpu3.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu3.store, wgpu::StoreOp::Store));

        // Load + Discard
        let ops4 = Operations::<wgpu::Color>::load_discard();
        let wgpu4 = ops4.to_wgpu();
        assert!(matches!(wgpu4.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu4.store, wgpu::StoreOp::Discard));
    }

    #[test]
    fn test_operations_to_wgpu_depth_with_all_combinations() {
        // Clear + Store
        let ops1 = Operations::new(LoadOp::Clear(1.0f32), StoreOp::Store);
        let wgpu1 = ops1.to_wgpu();
        assert!(matches!(wgpu1.load, wgpu::LoadOp::Clear(1.0)));
        assert!(matches!(wgpu1.store, wgpu::StoreOp::Store));

        // Clear + Discard
        let ops2 = Operations::new(LoadOp::Clear(0.0f32), StoreOp::Discard);
        let wgpu2 = ops2.to_wgpu();
        assert!(matches!(wgpu2.load, wgpu::LoadOp::Clear(0.0)));
        assert!(matches!(wgpu2.store, wgpu::StoreOp::Discard));

        // Load + Store
        let ops3: Operations<f32> = Operations::load_store();
        let wgpu3 = ops3.to_wgpu();
        assert!(matches!(wgpu3.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu3.store, wgpu::StoreOp::Store));

        // Load + Discard
        let ops4: Operations<f32> = Operations::load_discard();
        let wgpu4 = ops4.to_wgpu();
        assert!(matches!(wgpu4.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu4.store, wgpu::StoreOp::Discard));
    }

    #[test]
    fn test_operations_to_wgpu_stencil_with_all_combinations() {
        // Clear + Store
        let ops1 = Operations::new(LoadOp::Clear(0u32), StoreOp::Store);
        let wgpu1 = ops1.to_wgpu();
        assert!(matches!(wgpu1.load, wgpu::LoadOp::Clear(0)));
        assert!(matches!(wgpu1.store, wgpu::StoreOp::Store));

        // Clear + Discard
        let ops2 = Operations::new(LoadOp::Clear(255u32), StoreOp::Discard);
        let wgpu2 = ops2.to_wgpu();
        assert!(matches!(wgpu2.load, wgpu::LoadOp::Clear(255)));
        assert!(matches!(wgpu2.store, wgpu::StoreOp::Discard));

        // Load + Store
        let ops3: Operations<u32> = Operations::load_store();
        let wgpu3 = ops3.to_wgpu();
        assert!(matches!(wgpu3.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu3.store, wgpu::StoreOp::Store));

        // Load + Discard
        let ops4: Operations<u32> = Operations::load_discard();
        let wgpu4 = ops4.to_wgpu();
        assert!(matches!(wgpu4.load, wgpu::LoadOp::Load));
        assert!(matches!(wgpu4.store, wgpu::StoreOp::Discard));
    }

    // -------------------------------------------------------------------------
    // Operations Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_display_load_store() {
        let ops: Operations<f32> = Operations::load_store();
        let s = format!("{}", ops);
        assert!(s.contains("Load"));
        assert!(s.contains("Store"));
    }

    #[test]
    fn test_operations_display_clear_discard() {
        let ops = Operations::clear_discard(42u32);
        let s = format!("{}", ops);
        assert!(s.contains("Clear"));
        assert!(s.contains("Discard"));
    }

    #[test]
    fn test_load_op_display_with_color() {
        let op = LoadOp::Clear(wgpu::Color::TRANSPARENT);
        let s = format!("{}", op);
        assert!(s.contains("Clear"));
    }

    // -------------------------------------------------------------------------
    // Builder Integration: Operations in ColorAttachment
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_attachment_operations_clear_store() {
        let att = ColorAttachment::with_clear(wgpu::Color::RED);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_operations_load_store() {
        let att = ColorAttachment::with_load();
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_operations_clear_discard() {
        let att = ColorAttachment::new()
            .clear_color(wgpu::Color::BLACK)
            .store(StoreOp::Discard);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Clear(_)));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_color_attachment_operations_load_discard() {
        let att = ColorAttachment::with_load()
            .store(StoreOp::Discard);
        let ops = att.operations();
        assert!(matches!(ops.load, LoadOp::Load));
        assert_eq!(ops.store, StoreOp::Discard);
    }

    #[test]
    fn test_color_attachment_operations_to_wgpu() {
        let att = ColorAttachment::with_clear(wgpu::Color::GREEN);
        let wgpu_ops = att.operations().to_wgpu();
        if let wgpu::LoadOp::Clear(c) = wgpu_ops.load {
            assert_eq!(c, wgpu::Color::GREEN);
        }
    }

    // -------------------------------------------------------------------------
    // Builder Integration: Operations in DepthStencilAttachment
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_stencil_with_depth_ops_all_combinations() {
        // Clear + Store
        let att1 = DepthStencilAttachment::new()
            .with_depth_ops(Operations::clear(1.0));
        assert!(att1.depth_ops.is_some());
        let d_ops1 = att1.depth_ops.unwrap();
        assert!(matches!(d_ops1.load, LoadOp::Clear(1.0)));
        assert_eq!(d_ops1.store, StoreOp::Store);

        // Clear + Discard
        let att2 = DepthStencilAttachment::new()
            .with_depth_ops(Operations::clear_discard(0.0));
        let d_ops2 = att2.depth_ops.unwrap();
        assert!(matches!(d_ops2.load, LoadOp::Clear(0.0)));
        assert_eq!(d_ops2.store, StoreOp::Discard);

        // Load + Store
        let att3 = DepthStencilAttachment::new()
            .with_depth_ops(Operations::load_store());
        let d_ops3 = att3.depth_ops.unwrap();
        assert!(matches!(d_ops3.load, LoadOp::Load));
        assert_eq!(d_ops3.store, StoreOp::Store);

        // Load + Discard
        let att4 = DepthStencilAttachment::new()
            .with_depth_ops(Operations::load_discard());
        let d_ops4 = att4.depth_ops.unwrap();
        assert!(matches!(d_ops4.load, LoadOp::Load));
        assert_eq!(d_ops4.store, StoreOp::Discard);
    }

    #[test]
    fn test_depth_stencil_with_stencil_ops_all_combinations() {
        // Clear + Store
        let att1 = DepthStencilAttachment::new()
            .with_stencil_ops(Operations::clear(0u32));
        assert!(att1.stencil_ops.is_some());
        let s_ops1 = att1.stencil_ops.unwrap();
        assert!(matches!(s_ops1.load, LoadOp::Clear(0)));
        assert_eq!(s_ops1.store, StoreOp::Store);

        // Clear + Discard
        let att2 = DepthStencilAttachment::new()
            .with_stencil_ops(Operations::clear_discard(128));
        let s_ops2 = att2.stencil_ops.unwrap();
        assert!(matches!(s_ops2.load, LoadOp::Clear(128)));
        assert_eq!(s_ops2.store, StoreOp::Discard);

        // Load + Store
        let att3 = DepthStencilAttachment::new()
            .with_stencil_ops(Operations::load_store());
        let s_ops3 = att3.stencil_ops.unwrap();
        assert!(matches!(s_ops3.load, LoadOp::Load));
        assert_eq!(s_ops3.store, StoreOp::Store);

        // Load + Discard
        let att4 = DepthStencilAttachment::new()
            .with_stencil_ops(Operations::load_discard());
        let s_ops4 = att4.stencil_ops.unwrap();
        assert!(matches!(s_ops4.load, LoadOp::Load));
        assert_eq!(s_ops4.store, StoreOp::Discard);
    }

    #[test]
    fn test_depth_stencil_combined_operations() {
        let att = DepthStencilAttachment::new()
            .with_depth_ops(Operations::clear(1.0))
            .with_stencil_ops(Operations::clear(0));

        // Both depth and stencil should have clear operations
        let d_ops = att.depth_ops.unwrap();
        let s_ops = att.stencil_ops.unwrap();

        assert!(matches!(d_ops.load, LoadOp::Clear(1.0)));
        assert!(matches!(s_ops.load, LoadOp::Clear(0)));
    }

    // -------------------------------------------------------------------------
    // Trait Coverage: Clone, Copy, Default, Debug, PartialEq
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_traits_comprehensive() {
        // Clone
        let op1 = LoadOp::Clear(1.0f32);
        let op2 = op1.clone();
        assert_eq!(op1, op2);

        // Copy
        let op3 = op1;
        assert_eq!(op1, op3);

        // Default
        let op4: LoadOp<f32> = Default::default();
        assert!(matches!(op4, LoadOp::Clear(0.0)));

        // Debug
        let debug_str = format!("{:?}", op1);
        assert!(debug_str.contains("Clear"));

        // PartialEq
        assert_eq!(LoadOp::Clear(1.0f32), LoadOp::Clear(1.0f32));
        assert_ne!(LoadOp::Clear(1.0f32), LoadOp::Clear(0.0f32));
        assert_ne!(LoadOp::Clear(1.0f32), LoadOp::<f32>::Load);
    }

    #[test]
    fn test_store_op_traits_comprehensive() {
        // Clone
        let op1 = StoreOp::Store;
        let op2 = op1.clone();
        assert_eq!(op1, op2);

        // Copy
        let op3 = op1;
        assert_eq!(op1, op3);

        // Default
        let op4: StoreOp = Default::default();
        assert_eq!(op4, StoreOp::Store);

        // Debug
        let debug_str = format!("{:?}", op1);
        assert!(debug_str.contains("Store"));

        // PartialEq + Eq
        assert_eq!(StoreOp::Store, StoreOp::Store);
        assert_eq!(StoreOp::Discard, StoreOp::Discard);
        assert_ne!(StoreOp::Store, StoreOp::Discard);

        // Hash (via HashSet)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StoreOp::Store);
        set.insert(StoreOp::Discard);
        assert!(set.contains(&StoreOp::Store));
        assert!(set.contains(&StoreOp::Discard));
    }

    #[test]
    fn test_operations_traits_comprehensive() {
        // Clone
        let ops1 = Operations::clear(1.0f32);
        let ops2 = ops1.clone();
        assert_eq!(ops1, ops2);

        // Copy
        let ops3 = ops1;
        assert_eq!(ops1, ops3);

        // Default
        let ops4: Operations<f32> = Default::default();
        assert!(matches!(ops4.load, LoadOp::Clear(0.0)));
        assert_eq!(ops4.store, StoreOp::Store);

        // Debug
        let debug_str = format!("{:?}", ops1);
        assert!(debug_str.contains("Operations"));

        // PartialEq
        assert_eq!(Operations::clear(1.0f32), Operations::clear(1.0f32));
        assert_ne!(Operations::clear(1.0f32), Operations::clear(0.0f32));
    }

    #[test]
    fn test_color_attachment_traits_comprehensive() {
        // Clone
        let att1 = ColorAttachment::with_clear(wgpu::Color::RED);
        let att2 = att1.clone();
        assert_eq!(att1.clear_color, att2.clear_color);
        assert_eq!(att1.store_op, att2.store_op);

        // Default
        let att3: ColorAttachment = Default::default();
        assert_eq!(att3.clear_color, DEFAULT_CLEAR_COLOR);

        // Debug
        let debug_str = format!("{:?}", att1);
        assert!(debug_str.contains("ColorAttachment"));
    }

    #[test]
    fn test_depth_stencil_attachment_traits_comprehensive() {
        // Clone
        let att1 = DepthStencilAttachment::depth_stencil(1.0, 0);
        let att2 = att1.clone();
        assert_eq!(att1, att2);

        // Copy
        let att3 = att1;
        assert_eq!(att1, att3);

        // Default
        let att4: DepthStencilAttachment = Default::default();
        assert!(att4.depth_ops.is_some());
        assert!(att4.stencil_ops.is_none());

        // Debug
        let debug_str = format!("{:?}", att1);
        assert!(debug_str.contains("DepthStencilAttachment"));

        // PartialEq
        assert_eq!(
            DepthStencilAttachment::depth_only(1.0),
            DepthStencilAttachment::depth_only(1.0)
        );
        assert_ne!(
            DepthStencilAttachment::depth_only(1.0),
            DepthStencilAttachment::depth_only(0.0)
        );
    }

    // -------------------------------------------------------------------------
    // Operations Preset Methods Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_clear_black_value() {
        let ops = Operations::clear_black();
        if let LoadOp::Clear(c) = ops.load {
            // wgpu::Color::BLACK has r=0, g=0, b=0, a=0 (transparent black)
            assert_eq!(c, wgpu::Color::BLACK);
            assert_eq!(c.r, 0.0);
            assert_eq!(c.g, 0.0);
            assert_eq!(c.b, 0.0);
            // wgpu::Color::BLACK alpha value - verify against actual constant
            assert_eq!(c.a, wgpu::Color::BLACK.a);
        }
    }

    #[test]
    fn test_operations_clear_white_value() {
        let ops = Operations::clear_white();
        if let LoadOp::Clear(c) = ops.load {
            assert_eq!(c, wgpu::Color::WHITE);
            assert_eq!(c.r, 1.0);
            assert_eq!(c.g, 1.0);
            assert_eq!(c.b, 1.0);
            assert_eq!(c.a, 1.0);
        }
    }

    #[test]
    fn test_operations_clear_rgb_alpha_is_one() {
        let ops = Operations::clear_rgb(0.1, 0.2, 0.3);
        if let LoadOp::Clear(c) = ops.load {
            assert_eq!(c.a, 1.0);
        }
    }

    #[test]
    fn test_operations_clear_rgba_all_values() {
        let ops = Operations::clear_rgba(0.1, 0.2, 0.3, 0.4);
        if let LoadOp::Clear(c) = ops.load {
            assert_eq!(c.r, 0.1);
            assert_eq!(c.g, 0.2);
            assert_eq!(c.b, 0.3);
            assert_eq!(c.a, 0.4);
        }
    }

    #[test]
    fn test_operations_clear_depth_value_is_one() {
        let ops = Operations::clear_depth();
        if let LoadOp::Clear(d) = ops.load {
            assert_eq!(d, 1.0);
        }
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear_depth_reverse_z_value_is_zero() {
        let ops = Operations::clear_depth_reverse_z();
        if let LoadOp::Clear(d) = ops.load {
            assert_eq!(d, 0.0);
        }
        assert_eq!(ops.store, StoreOp::Store);
    }

    #[test]
    fn test_operations_clear_stencil_value_is_zero() {
        let ops = Operations::clear_stencil();
        if let LoadOp::Clear(s) = ops.load {
            assert_eq!(s, 0);
        }
        assert_eq!(ops.store, StoreOp::Store);
    }

    // -------------------------------------------------------------------------
    // Builder with Operations Integration
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_color_attachment_preserves_load_op_clear() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear(wgpu::Color::RED))
            .build();

        if let Some(Some(att)) = desc.color_attachments.first() {
            if let LoadOp::Clear(c) = att.load_op {
                assert_eq!(c, wgpu::Color::RED);
            } else {
                panic!("Expected Clear");
            }
        }
    }

    #[test]
    fn test_builder_color_attachment_preserves_load_op_load() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::load_store())
            .build();

        if let Some(Some(att)) = desc.color_attachments.first() {
            assert!(matches!(att.load_op, LoadOp::Load));
        }
    }

    #[test]
    fn test_builder_color_attachment_preserves_store_op_store() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear(wgpu::Color::BLACK))
            .build();

        if let Some(Some(att)) = desc.color_attachments.first() {
            assert_eq!(att.store_op, StoreOp::Store);
        }
    }

    #[test]
    fn test_builder_color_attachment_preserves_store_op_discard() {
        let desc = RenderPassBuilder::new()
            .color_attachment(Operations::clear_discard(wgpu::Color::BLACK))
            .build();

        if let Some(Some(att)) = desc.color_attachments.first() {
            assert_eq!(att.store_op, StoreOp::Discard);
        }
    }

    #[test]
    fn test_builder_depth_stencil_preserves_depth_ops() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(Some(Operations::clear(0.5)), None)
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            if let Some(ops) = &att.depth_ops {
                if let LoadOp::Clear(d) = ops.load {
                    assert_eq!(d, 0.5);
                }
            }
        }
    }

    #[test]
    fn test_builder_depth_stencil_preserves_stencil_ops() {
        let desc = RenderPassBuilder::new()
            .depth_stencil(None, Some(Operations::clear(42)))
            .build();

        if let Some(att) = &desc.depth_stencil_attachment {
            if let Some(ops) = &att.stencil_ops {
                if let LoadOp::Clear(s) = ops.load {
                    assert_eq!(s, 42);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // LoadOp Default with Various Clear Value Types
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_default_i32() {
        let op: LoadOp<i32> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(0)));
    }

    #[test]
    fn test_load_op_default_i64() {
        let op: LoadOp<i64> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(0)));
    }

    #[test]
    fn test_load_op_default_f64() {
        let op: LoadOp<f64> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(0.0)));
    }

    #[test]
    fn test_load_op_default_bool() {
        let op: LoadOp<bool> = LoadOp::default();
        assert!(matches!(op, LoadOp::Clear(false)));
    }

    // -------------------------------------------------------------------------
    // Operations with Extreme Stencil Values
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_stencil_max_value() {
        let ops = Operations::clear(u32::MAX);
        if let LoadOp::Clear(s) = ops.load {
            assert_eq!(s, u32::MAX);
        }
    }

    #[test]
    fn test_operations_stencil_common_mask_values() {
        // Common stencil mask: 0xFF (8-bit)
        let ops1 = Operations::clear(0xFFu32);
        if let LoadOp::Clear(s) = ops1.load {
            assert_eq!(s, 255);
        }

        // Bit patterns
        let ops2 = Operations::clear(0b10101010u32);
        if let LoadOp::Clear(s) = ops2.load {
            assert_eq!(s, 170);
        }
    }

    // -------------------------------------------------------------------------
    // Operations Field Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_direct_field_access_load() {
        let ops = Operations::clear(1.0f32);
        let load = ops.load;
        assert!(matches!(load, LoadOp::Clear(1.0)));
    }

    #[test]
    fn test_operations_direct_field_access_store() {
        let ops = Operations::clear_discard(1.0f32);
        let store = ops.store;
        assert_eq!(store, StoreOp::Discard);
    }

    #[test]
    fn test_operations_field_modification() {
        let mut ops = Operations::clear(1.0f32);
        ops.store = StoreOp::Discard;
        assert_eq!(ops.store, StoreOp::Discard);

        ops.load = LoadOp::Load;
        assert!(matches!(ops.load, LoadOp::Load));
    }

    // -------------------------------------------------------------------------
    // wgpu Type Conversion Round-Trip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operations_color_wgpu_conversion_preserves_values() {
        let original = Operations::clear_rgba(0.123, 0.456, 0.789, 0.999);
        let wgpu_ops = original.to_wgpu();

        if let wgpu::LoadOp::Clear(c) = wgpu_ops.load {
            assert!((c.r - 0.123).abs() < 1e-10);
            assert!((c.g - 0.456).abs() < 1e-10);
            assert!((c.b - 0.789).abs() < 1e-10);
            assert!((c.a - 0.999).abs() < 1e-10);
        }
    }

    #[test]
    fn test_operations_depth_wgpu_conversion_preserves_precision() {
        let original = Operations::clear(0.123456789f32);
        let wgpu_ops = original.to_wgpu();

        if let wgpu::LoadOp::Clear(d) = wgpu_ops.load {
            assert!((d - 0.123456789).abs() < f32::EPSILON * 10.0);
        }
    }
}
