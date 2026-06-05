//! Render Pass Declaration for the TRINITY Frame Graph (T-WGPU-P7.5.5)
//!
//! This module provides structs and builders for declaring render passes within
//! the frame graph. Render passes describe color and depth attachments, load/store
//! operations, viewport configuration, and execution callbacks.
//!
//! # Architecture
//!
//! - **RenderPassConfig**: The complete configuration for a render pass
//! - **PassColorAttachment**: A color output with load/store ops and optional resolve
//! - **PassDepthAttachment**: A depth/stencil output with separate ops for each component
//! - **PassViewport**: Viewport and depth range configuration
//! - **RenderPassBuilder**: Fluent API for constructing render pass configurations
//! - **PassExecutor**: Trait for custom pass execution logic
//! - **RenderPassNode**: A fully configured render pass ready for scheduling
//!
//! Note: Types use "Pass" prefix to avoid collision with existing IR types in mod.rs.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::frame_graph::passes::*;
//! use renderer_backend::frame_graph::graph::ResourceId;
//!
//! let color_target = ResourceId::new(0);
//! let depth_target = ResourceId::new(1);
//!
//! let pass_config = RenderPassBuilder::new("main_pass")
//!     .add_color_attachment(PassColorAttachment {
//!         resource: color_target,
//!         load_op: PassLoadOp::Clear,
//!         store_op: PassStoreOp::Store,
//!         clear_color: Some([0.0, 0.0, 0.0, 1.0]),
//!         resolve_target: None,
//!     })
//!     .set_depth_attachment(PassDepthAttachment {
//!         resource: depth_target,
//!         depth_load_op: PassLoadOp::Clear,
//!         depth_store_op: PassStoreOp::Store,
//!         stencil_load_op: PassLoadOp::DontCare,
//!         stencil_store_op: PassStoreOp::Discard,
//!         clear_depth: 1.0,
//!         clear_stencil: 0,
//!         read_only: false,
//!     })
//!     .sample_count(4)
//!     .viewport(PassViewport::default())
//!     .build();
//! ```

use std::fmt;

use super::graph::{PassId, RenderContext, ResourceId};

// ---------------------------------------------------------------------------
// Load/Store Operations
// ---------------------------------------------------------------------------

/// Load operation for an attachment at the start of a render pass.
///
/// Determines what happens to the attachment's existing contents when
/// the render pass begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PassLoadOp {
    /// Clear the attachment to a specified value.
    #[default]
    Clear,
    /// Preserve the existing contents (requires previous write).
    Load,
    /// Contents are undefined - allows driver optimizations when
    /// the entire attachment will be overwritten.
    DontCare,
}

impl PassLoadOp {
    /// Returns true if this operation clears the attachment.
    #[inline]
    pub const fn is_clear(&self) -> bool {
        matches!(self, Self::Clear)
    }

    /// Returns true if this operation loads existing contents.
    #[inline]
    pub const fn is_load(&self) -> bool {
        matches!(self, Self::Load)
    }

    /// Returns true if this operation allows undefined contents.
    #[inline]
    pub const fn is_dont_care(&self) -> bool {
        matches!(self, Self::DontCare)
    }

    /// Converts to wgpu LoadOp with the given clear value for color attachments.
    pub fn to_wgpu_color(&self, clear_color: Option<[f32; 4]>) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            Self::Clear => {
                let [r, g, b, a] = clear_color.unwrap_or([0.0, 0.0, 0.0, 1.0]);
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: r as f64,
                    g: g as f64,
                    b: b as f64,
                    a: a as f64,
                })
            }
            Self::Load => wgpu::LoadOp::Load,
            Self::DontCare => wgpu::LoadOp::Load, // wgpu doesn't have DontCare for color
        }
    }

    /// Converts to wgpu LoadOp for depth.
    pub fn to_wgpu_depth(&self, clear_depth: f32) -> wgpu::LoadOp<f32> {
        match self {
            Self::Clear => wgpu::LoadOp::Clear(clear_depth),
            Self::Load => wgpu::LoadOp::Load,
            Self::DontCare => wgpu::LoadOp::Load,
        }
    }

    /// Converts to wgpu LoadOp for stencil.
    pub fn to_wgpu_stencil(&self, clear_stencil: u32) -> wgpu::LoadOp<u32> {
        match self {
            Self::Clear => wgpu::LoadOp::Clear(clear_stencil),
            Self::Load => wgpu::LoadOp::Load,
            Self::DontCare => wgpu::LoadOp::Load,
        }
    }
}

impl fmt::Display for PassLoadOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clear => write!(f, "Clear"),
            Self::Load => write!(f, "Load"),
            Self::DontCare => write!(f, "DontCare"),
        }
    }
}

/// Store operation for an attachment at the end of a render pass.
///
/// Determines what happens to the rendered contents when the render pass ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PassStoreOp {
    /// Store the rendered contents to memory.
    #[default]
    Store,
    /// Discard the contents - allows driver optimizations when
    /// the contents are not needed after the pass.
    Discard,
}

impl PassStoreOp {
    /// Returns true if this operation stores contents.
    #[inline]
    pub const fn is_store(&self) -> bool {
        matches!(self, Self::Store)
    }

    /// Returns true if this operation discards contents.
    #[inline]
    pub const fn is_discard(&self) -> bool {
        matches!(self, Self::Discard)
    }

    /// Converts to wgpu StoreOp.
    pub const fn to_wgpu(&self) -> wgpu::StoreOp {
        match self {
            Self::Store => wgpu::StoreOp::Store,
            Self::Discard => wgpu::StoreOp::Discard,
        }
    }
}

impl fmt::Display for PassStoreOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store => write!(f, "Store"),
            Self::Discard => write!(f, "Discard"),
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

/// Viewport configuration for a render pass.
///
/// Defines the region of the framebuffer to render to and the depth range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassViewport {
    /// X offset of the viewport (pixels).
    pub x: f32,
    /// Y offset of the viewport (pixels).
    pub y: f32,
    /// Width of the viewport (pixels).
    pub width: f32,
    /// Height of the viewport (pixels).
    pub height: f32,
    /// Minimum depth value (0.0-1.0).
    pub min_depth: f32,
    /// Maximum depth value (0.0-1.0).
    pub max_depth: f32,
}

impl PassViewport {
    /// Creates a new viewport with the specified dimensions.
    ///
    /// Uses default depth range (0.0-1.0).
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Creates a new viewport starting at origin with the specified size.
    pub fn with_size(width: f32, height: f32) -> Self {
        Self::new(0.0, 0.0, width, height)
    }

    /// Creates a viewport with custom depth range.
    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

    /// Returns true if the viewport has valid dimensions.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.width > 0.0 && self.height > 0.0 && self.min_depth <= self.max_depth
    }

    /// Returns the aspect ratio (width / height).
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height > 0.0 {
            self.width / self.height
        } else {
            1.0
        }
    }
}

impl Default for PassViewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1920.0,
            height: 1080.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl fmt::Display for PassViewport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Viewport({}, {}, {}x{}, depth={}-{})",
            self.x, self.y, self.width, self.height, self.min_depth, self.max_depth
        )
    }
}

// ---------------------------------------------------------------------------
// Color Attachment
// ---------------------------------------------------------------------------

/// A color attachment for a render pass.
///
/// Describes a single color output target with its load/store operations
/// and optional MSAA resolve target.
#[derive(Clone, Debug, PartialEq)]
pub struct PassColorAttachment {
    /// The resource ID of the color target texture.
    pub resource: ResourceId,
    /// Load operation at the start of the pass.
    pub load_op: PassLoadOp,
    /// Store operation at the end of the pass.
    pub store_op: PassStoreOp,
    /// Clear color value (RGBA, used when load_op is Clear).
    pub clear_color: Option<[f32; 4]>,
    /// Optional resolve target for MSAA (multisampled -> resolved).
    pub resolve_target: Option<ResourceId>,
}

impl PassColorAttachment {
    /// Creates a new color attachment with default settings.
    ///
    /// Default: Clear to black, store results, no resolve target.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            load_op: PassLoadOp::Clear,
            store_op: PassStoreOp::Store,
            clear_color: Some([0.0, 0.0, 0.0, 1.0]),
            resolve_target: None,
        }
    }

    /// Creates a color attachment that loads existing contents.
    pub fn load(resource: ResourceId) -> Self {
        Self {
            resource,
            load_op: PassLoadOp::Load,
            store_op: PassStoreOp::Store,
            clear_color: None,
            resolve_target: None,
        }
    }

    /// Creates a color attachment with clear operation.
    pub fn clear(resource: ResourceId, clear_color: [f32; 4]) -> Self {
        Self {
            resource,
            load_op: PassLoadOp::Clear,
            store_op: PassStoreOp::Store,
            clear_color: Some(clear_color),
            resolve_target: None,
        }
    }

    /// Creates a transient color attachment (DontCare/Discard).
    ///
    /// Useful for intermediate attachments that are only used within
    /// the same render pass.
    pub fn transient(resource: ResourceId) -> Self {
        Self {
            resource,
            load_op: PassLoadOp::DontCare,
            store_op: PassStoreOp::Discard,
            clear_color: None,
            resolve_target: None,
        }
    }

    /// Sets the resolve target for MSAA.
    pub fn with_resolve(mut self, resolve_target: ResourceId) -> Self {
        self.resolve_target = Some(resolve_target);
        self
    }

    /// Sets the clear color.
    pub fn with_clear_color(mut self, color: [f32; 4]) -> Self {
        self.clear_color = Some(color);
        self.load_op = PassLoadOp::Clear;
        self
    }

    /// Returns true if this attachment has a resolve target.
    #[inline]
    pub fn has_resolve(&self) -> bool {
        self.resolve_target.is_some()
    }

    /// Returns all resource IDs referenced by this attachment.
    pub fn referenced_resources(&self) -> Vec<ResourceId> {
        let mut resources = vec![self.resource];
        if let Some(resolve) = self.resolve_target {
            resources.push(resolve);
        }
        resources
    }
}

impl fmt::Display for PassColorAttachment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ColorAttachment({}, {}/{}, resolve={:?})",
            self.resource,
            self.load_op,
            self.store_op,
            self.resolve_target
        )
    }
}

// ---------------------------------------------------------------------------
// Depth Attachment
// ---------------------------------------------------------------------------

/// A depth/stencil attachment for a render pass.
///
/// Describes the depth and stencil buffer with separate operations for
/// each component and optional read-only mode.
#[derive(Clone, Debug, PartialEq)]
pub struct PassDepthAttachment {
    /// The resource ID of the depth/stencil texture.
    pub resource: ResourceId,
    /// Load operation for the depth component.
    pub depth_load_op: PassLoadOp,
    /// Store operation for the depth component.
    pub depth_store_op: PassStoreOp,
    /// Load operation for the stencil component.
    pub stencil_load_op: PassLoadOp,
    /// Store operation for the stencil component.
    pub stencil_store_op: PassStoreOp,
    /// Clear value for depth (0.0-1.0, typically 1.0 for far plane).
    pub clear_depth: f32,
    /// Clear value for stencil (0-255).
    pub clear_stencil: u32,
    /// If true, depth/stencil writes are disabled.
    pub read_only: bool,
}

impl PassDepthAttachment {
    /// Creates a new depth attachment with default settings.
    ///
    /// Default: Clear depth to 1.0, store results, no stencil usage.
    pub fn new(resource: ResourceId) -> Self {
        Self {
            resource,
            depth_load_op: PassLoadOp::Clear,
            depth_store_op: PassStoreOp::Store,
            stencil_load_op: PassLoadOp::DontCare,
            stencil_store_op: PassStoreOp::Discard,
            clear_depth: 1.0,
            clear_stencil: 0,
            read_only: false,
        }
    }

    /// Creates a depth attachment that loads existing depth values.
    pub fn load(resource: ResourceId) -> Self {
        Self {
            resource,
            depth_load_op: PassLoadOp::Load,
            depth_store_op: PassStoreOp::Store,
            stencil_load_op: PassLoadOp::DontCare,
            stencil_store_op: PassStoreOp::Discard,
            clear_depth: 1.0,
            clear_stencil: 0,
            read_only: false,
        }
    }

    /// Creates a read-only depth attachment (for depth testing without writing).
    pub fn read_only(resource: ResourceId) -> Self {
        Self {
            resource,
            depth_load_op: PassLoadOp::Load,
            depth_store_op: PassStoreOp::Store,
            stencil_load_op: PassLoadOp::DontCare,
            stencil_store_op: PassStoreOp::Discard,
            clear_depth: 1.0,
            clear_stencil: 0,
            read_only: true,
        }
    }

    /// Creates a depth attachment with stencil support.
    pub fn with_stencil(resource: ResourceId) -> Self {
        Self {
            resource,
            depth_load_op: PassLoadOp::Clear,
            depth_store_op: PassStoreOp::Store,
            stencil_load_op: PassLoadOp::Clear,
            stencil_store_op: PassStoreOp::Store,
            clear_depth: 1.0,
            clear_stencil: 0,
            read_only: false,
        }
    }

    /// Sets the clear depth value.
    pub fn with_clear_depth(mut self, depth: f32) -> Self {
        self.clear_depth = depth;
        self.depth_load_op = PassLoadOp::Clear;
        self
    }

    /// Sets the clear stencil value.
    pub fn with_clear_stencil(mut self, stencil: u32) -> Self {
        self.clear_stencil = stencil;
        self.stencil_load_op = PassLoadOp::Clear;
        self
    }

    /// Enables read-only mode.
    pub fn make_read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Returns true if the depth component will be written.
    #[inline]
    pub fn writes_depth(&self) -> bool {
        !self.read_only && self.depth_store_op.is_store()
    }

    /// Returns true if the stencil component will be written.
    #[inline]
    pub fn writes_stencil(&self) -> bool {
        !self.read_only && self.stencil_store_op.is_store()
    }
}

impl fmt::Display for PassDepthAttachment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DepthAttachment({}, depth={}/{}, stencil={}/{}, clear={}/{}, read_only={})",
            self.resource,
            self.depth_load_op,
            self.depth_store_op,
            self.stencil_load_op,
            self.stencil_store_op,
            self.clear_depth,
            self.clear_stencil,
            self.read_only
        )
    }
}

// ---------------------------------------------------------------------------
// Render Pass Config
// ---------------------------------------------------------------------------

/// Complete configuration for a render pass.
///
/// Contains all attachments, sample count, viewport, and other settings
/// needed to create a wgpu render pass.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderPassConfig {
    /// Human-readable name for debugging.
    pub name: String,
    /// Color attachments (up to 8 for most GPUs).
    pub color_attachments: Vec<PassColorAttachment>,
    /// Optional depth/stencil attachment.
    pub depth_attachment: Option<PassDepthAttachment>,
    /// MSAA sample count (1 = no multisampling).
    pub sample_count: u32,
    /// Optional viewport override.
    pub viewport: Option<PassViewport>,
}

impl RenderPassConfig {
    /// Creates a new render pass config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            color_attachments: Vec::new(),
            depth_attachment: None,
            sample_count: 1,
            viewport: None,
        }
    }

    /// Creates a minimal config with a single color attachment.
    pub fn with_color(name: impl Into<String>, color: PassColorAttachment) -> Self {
        Self {
            name: name.into(),
            color_attachments: vec![color],
            depth_attachment: None,
            sample_count: 1,
            viewport: None,
        }
    }

    /// Creates a config with color and depth attachments.
    pub fn with_color_and_depth(
        name: impl Into<String>,
        color: PassColorAttachment,
        depth: PassDepthAttachment,
    ) -> Self {
        Self {
            name: name.into(),
            color_attachments: vec![color],
            depth_attachment: Some(depth),
            sample_count: 1,
            viewport: None,
        }
    }

    /// Returns true if this config has any color attachments.
    #[inline]
    pub fn has_color(&self) -> bool {
        !self.color_attachments.is_empty()
    }

    /// Returns true if this config has a depth attachment.
    #[inline]
    pub fn has_depth(&self) -> bool {
        self.depth_attachment.is_some()
    }

    /// Returns true if this config uses MSAA.
    #[inline]
    pub fn is_multisampled(&self) -> bool {
        self.sample_count > 1
    }

    /// Returns the number of color attachments.
    #[inline]
    pub fn color_attachment_count(&self) -> usize {
        self.color_attachments.len()
    }

    /// Returns all resource IDs written by this pass.
    pub fn written_resources(&self) -> Vec<ResourceId> {
        let mut resources = Vec::new();

        for attachment in &self.color_attachments {
            if attachment.store_op.is_store() {
                resources.push(attachment.resource);
            }
            if let Some(resolve) = attachment.resolve_target {
                // Resolve target is always written
                resources.push(resolve);
            }
        }

        if let Some(depth) = &self.depth_attachment {
            if depth.writes_depth() || depth.writes_stencil() {
                resources.push(depth.resource);
            }
        }

        resources
    }

    /// Returns all resource IDs read by this pass.
    pub fn read_resources(&self) -> Vec<ResourceId> {
        let mut resources = Vec::new();

        for attachment in &self.color_attachments {
            if attachment.load_op.is_load() {
                resources.push(attachment.resource);
            }
        }

        if let Some(depth) = &self.depth_attachment {
            if depth.depth_load_op.is_load() || depth.read_only {
                resources.push(depth.resource);
            }
        }

        resources
    }

    /// Validates the configuration.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        // Check sample count is valid (1, 2, 4, 8, 16)
        if !matches!(self.sample_count, 1 | 2 | 4 | 8 | 16) {
            return Some(format!(
                "Invalid sample count: {}. Must be 1, 2, 4, 8, or 16.",
                self.sample_count
            ));
        }

        // Check at least one attachment exists
        if self.color_attachments.is_empty() && self.depth_attachment.is_none() {
            return Some("Render pass must have at least one attachment.".to_string());
        }

        // Check color attachment limit
        if self.color_attachments.len() > 8 {
            return Some(format!(
                "Too many color attachments: {}. Maximum is 8.",
                self.color_attachments.len()
            ));
        }

        // Validate viewport if present
        if let Some(viewport) = &self.viewport {
            if !viewport.is_valid() {
                return Some(format!("Invalid viewport: {:?}", viewport));
            }
        }

        None
    }
}

impl fmt::Display for RenderPassConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RenderPassConfig(\"{}\", colors={}, depth={}, samples={})",
            self.name,
            self.color_attachments.len(),
            self.depth_attachment.is_some(),
            self.sample_count
        )
    }
}

// ---------------------------------------------------------------------------
// Render Pass Builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing RenderPassConfig.
///
/// Provides a chainable API for building render pass configurations
/// with resource tracking for dependency analysis.
#[derive(Clone, Debug, Default)]
pub struct RenderPassBuilder {
    /// The config being built.
    config: RenderPassConfig,
    /// Resources explicitly marked as read dependencies.
    reads: Vec<ResourceId>,
    /// Resources explicitly marked as write dependencies.
    writes: Vec<ResourceId>,
}

impl RenderPassBuilder {
    /// Creates a new builder with the given pass name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: RenderPassConfig::new(name),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Adds a color attachment to the pass.
    pub fn add_color_attachment(mut self, attachment: PassColorAttachment) -> Self {
        // Track resources for dependency analysis
        if attachment.load_op.is_load() {
            self.reads.push(attachment.resource);
        }
        if attachment.store_op.is_store() {
            self.writes.push(attachment.resource);
        }
        if let Some(resolve) = attachment.resolve_target {
            self.writes.push(resolve);
        }

        self.config.color_attachments.push(attachment);
        self
    }

    /// Sets the depth attachment for the pass.
    pub fn set_depth_attachment(mut self, attachment: PassDepthAttachment) -> Self {
        // Track resources for dependency analysis
        if attachment.depth_load_op.is_load() || attachment.read_only {
            self.reads.push(attachment.resource);
        }
        if attachment.writes_depth() || attachment.writes_stencil() {
            self.writes.push(attachment.resource);
        }

        self.config.depth_attachment = Some(attachment);
        self
    }

    /// Adds an explicit read dependency.
    ///
    /// Use for resources that are sampled or accessed as shader inputs
    /// but are not attachments (e.g., texture sampling, uniform buffers).
    pub fn read_resource(mut self, resource: ResourceId) -> Self {
        if !self.reads.contains(&resource) {
            self.reads.push(resource);
        }
        self
    }

    /// Adds an explicit write dependency.
    ///
    /// Use for resources that are written but are not attachments
    /// (e.g., storage buffers, storage textures).
    pub fn write_resource(mut self, resource: ResourceId) -> Self {
        if !self.writes.contains(&resource) {
            self.writes.push(resource);
        }
        self
    }

    /// Sets the MSAA sample count.
    ///
    /// Must be 1, 2, 4, 8, or 16.
    pub fn sample_count(mut self, count: u32) -> Self {
        self.config.sample_count = count;
        self
    }

    /// Sets the viewport.
    pub fn viewport(mut self, viewport: PassViewport) -> Self {
        self.config.viewport = Some(viewport);
        self
    }

    /// Returns the list of read dependencies.
    pub fn get_reads(&self) -> &[ResourceId] {
        &self.reads
    }

    /// Returns the list of write dependencies.
    pub fn get_writes(&self) -> &[ResourceId] {
        &self.writes
    }

    /// Builds the final RenderPassConfig.
    ///
    /// Consumes the builder.
    pub fn build(self) -> RenderPassConfig {
        self.config
    }

    /// Builds and returns both the config and resource dependencies.
    pub fn build_with_deps(self) -> (RenderPassConfig, Vec<ResourceId>, Vec<ResourceId>) {
        (self.config, self.reads, self.writes)
    }
}

impl Default for RenderPassConfig {
    fn default() -> Self {
        Self::new("unnamed_pass")
    }
}

// ---------------------------------------------------------------------------
// Pass Executor Trait
// ---------------------------------------------------------------------------

/// Trait for custom render pass execution logic.
///
/// Implementations receive the render context and wgpu render pass
/// and are responsible for recording draw commands.
pub trait PassExecutor: Send + Sync {
    /// Execute the pass, recording commands to the render pass.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The render context with frame state
    /// * `encoder` - The wgpu render pass to record commands to
    fn execute(&self, ctx: &mut RenderContext, encoder: &mut wgpu::RenderPass);

    /// Optional: Returns a debug name for this executor.
    fn name(&self) -> &str {
        "PassExecutor"
    }
}

/// A no-op executor that does nothing.
///
/// Useful for placeholder passes or depth-only passes.
#[derive(Clone, Debug, Default)]
pub struct NoOpExecutor;

impl PassExecutor for NoOpExecutor {
    fn execute(&self, _ctx: &mut RenderContext, _encoder: &mut wgpu::RenderPass) {
        // No-op
    }

    fn name(&self) -> &str {
        "NoOpExecutor"
    }
}

/// A closure-based executor.
pub struct FnExecutor<F: Fn(&mut RenderContext, &mut wgpu::RenderPass) + Send + Sync> {
    func: F,
    name: &'static str,
}

impl<F: Fn(&mut RenderContext, &mut wgpu::RenderPass) + Send + Sync> FnExecutor<F> {
    /// Creates a new function executor.
    pub fn new(func: F) -> Self {
        Self {
            func,
            name: "FnExecutor",
        }
    }

    /// Creates a new function executor with a name.
    pub fn named(name: &'static str, func: F) -> Self {
        Self { func, name }
    }
}

impl<F: Fn(&mut RenderContext, &mut wgpu::RenderPass) + Send + Sync> PassExecutor for FnExecutor<F> {
    fn execute(&self, ctx: &mut RenderContext, encoder: &mut wgpu::RenderPass) {
        (self.func)(ctx, encoder);
    }

    fn name(&self) -> &str {
        self.name
    }
}

// ---------------------------------------------------------------------------
// Render Pass Node
// ---------------------------------------------------------------------------

/// A complete render pass node ready for frame graph scheduling.
///
/// Combines configuration with an executor for rendering.
pub struct RenderPassNode {
    /// Unique identifier assigned by the frame graph.
    pub id: PassId,
    /// The render pass configuration.
    pub config: RenderPassConfig,
    /// The executor responsible for recording draw commands.
    pub executor: Box<dyn PassExecutor>,
}

impl RenderPassNode {
    /// Creates a new render pass node.
    pub fn new(id: PassId, config: RenderPassConfig, executor: Box<dyn PassExecutor>) -> Self {
        Self {
            id,
            config,
            executor,
        }
    }

    /// Creates a render pass node with a no-op executor.
    pub fn empty(id: PassId, config: RenderPassConfig) -> Self {
        Self {
            id,
            config,
            executor: Box::new(NoOpExecutor),
        }
    }

    /// Creates a render pass node with a closure executor.
    pub fn with_fn<F>(id: PassId, config: RenderPassConfig, func: F) -> Self
    where
        F: Fn(&mut RenderContext, &mut wgpu::RenderPass) + Send + Sync + 'static,
    {
        Self {
            id,
            config,
            executor: Box::new(FnExecutor::new(func)),
        }
    }

    /// Returns the pass name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns all resources written by this pass.
    pub fn written_resources(&self) -> Vec<ResourceId> {
        self.config.written_resources()
    }

    /// Returns all resources read by this pass.
    pub fn read_resources(&self) -> Vec<ResourceId> {
        self.config.read_resources()
    }
}

impl fmt::Debug for RenderPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderPassNode")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("executor", &self.executor.name())
            .finish()
    }
}

impl fmt::Display for RenderPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RenderPassNode({}, \"{}\", executor={})",
            self.id,
            self.config.name,
            self.executor.name()
        )
    }
}

// ---------------------------------------------------------------------------
// Compute Pass - Dispatch Size
// ---------------------------------------------------------------------------

/// Dispatch size configuration for compute passes.
///
/// Specifies how many workgroups to dispatch, either directly or via
/// an indirect buffer containing the dispatch parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchSize {
    /// Direct dispatch with explicit workgroup counts.
    ///
    /// The compute shader will be dispatched with the specified number
    /// of workgroups in each dimension.
    Direct {
        /// Number of workgroups in X dimension.
        x: u32,
        /// Number of workgroups in Y dimension.
        y: u32,
        /// Number of workgroups in Z dimension.
        z: u32,
    },
    /// Indirect dispatch using parameters from a GPU buffer.
    ///
    /// The dispatch parameters (x, y, z workgroup counts) are read
    /// from the specified buffer at the given offset.
    Indirect {
        /// The resource ID of the buffer containing dispatch parameters.
        buffer: ResourceId,
        /// Byte offset into the buffer where the dispatch parameters start.
        /// The buffer must contain three u32 values (x, y, z) at this offset.
        offset: u64,
    },
}

impl DispatchSize {
    /// Creates a direct dispatch with the specified workgroup counts.
    pub fn direct(x: u32, y: u32, z: u32) -> Self {
        Self::Direct { x, y, z }
    }

    /// Creates a 1D direct dispatch.
    pub fn direct_1d(x: u32) -> Self {
        Self::Direct { x, y: 1, z: 1 }
    }

    /// Creates a 2D direct dispatch.
    pub fn direct_2d(x: u32, y: u32) -> Self {
        Self::Direct { x, y, z: 1 }
    }

    /// Creates an indirect dispatch from a buffer.
    pub fn indirect(buffer: ResourceId, offset: u64) -> Self {
        Self::Indirect { buffer, offset }
    }

    /// Returns true if this is a direct dispatch.
    #[inline]
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// Returns true if this is an indirect dispatch.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        matches!(self, Self::Indirect { .. })
    }

    /// Returns the total number of workgroups for direct dispatch.
    ///
    /// Returns None for indirect dispatch.
    pub fn total_workgroups(&self) -> Option<u64> {
        match self {
            Self::Direct { x, y, z } => Some(*x as u64 * *y as u64 * *z as u64),
            Self::Indirect { .. } => None,
        }
    }

    /// Validates the dispatch size.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        match self {
            Self::Direct { x, y, z } => {
                // Check for zero dimensions
                if *x == 0 || *y == 0 || *z == 0 {
                    return Some("Dispatch dimensions must be non-zero".to_string());
                }
                // Check for excessive dimensions (device limits typically 65535)
                const MAX_DIM: u32 = 65535;
                if *x > MAX_DIM || *y > MAX_DIM || *z > MAX_DIM {
                    return Some(format!(
                        "Dispatch dimension exceeds maximum ({}): ({}, {}, {})",
                        MAX_DIM, x, y, z
                    ));
                }
                None
            }
            Self::Indirect { buffer, offset } => {
                if buffer.is_invalid() {
                    return Some("Indirect dispatch buffer is invalid".to_string());
                }
                // Offset must be aligned to 4 bytes (size of u32)
                if *offset % 4 != 0 {
                    return Some(format!(
                        "Indirect dispatch offset must be 4-byte aligned: {}",
                        offset
                    ));
                }
                None
            }
        }
    }
}

impl Default for DispatchSize {
    fn default() -> Self {
        Self::Direct { x: 1, y: 1, z: 1 }
    }
}

impl fmt::Display for DispatchSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct { x, y, z } => write!(f, "Direct({}, {}, {})", x, y, z),
            Self::Indirect { buffer, offset } => {
                write!(f, "Indirect({}, offset={})", buffer, offset)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute Pass Config
// ---------------------------------------------------------------------------

/// Complete configuration for a compute pass.
///
/// Contains the dispatch parameters and other settings needed to execute
/// a compute shader dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputePassConfig {
    /// Human-readable name for debugging.
    pub name: String,
    /// The dispatch size configuration.
    pub dispatch_size: DispatchSize,
}

impl ComputePassConfig {
    /// Creates a new compute pass config with the given name.
    ///
    /// Default dispatch is a single workgroup.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dispatch_size: DispatchSize::default(),
        }
    }

    /// Creates a compute pass config with direct dispatch.
    pub fn with_dispatch(name: impl Into<String>, x: u32, y: u32, z: u32) -> Self {
        Self {
            name: name.into(),
            dispatch_size: DispatchSize::direct(x, y, z),
        }
    }

    /// Creates a compute pass config with 1D dispatch.
    pub fn with_dispatch_1d(name: impl Into<String>, x: u32) -> Self {
        Self {
            name: name.into(),
            dispatch_size: DispatchSize::direct_1d(x),
        }
    }

    /// Creates a compute pass config with 2D dispatch.
    pub fn with_dispatch_2d(name: impl Into<String>, x: u32, y: u32) -> Self {
        Self {
            name: name.into(),
            dispatch_size: DispatchSize::direct_2d(x, y),
        }
    }

    /// Creates a compute pass config with indirect dispatch.
    pub fn with_indirect(name: impl Into<String>, buffer: ResourceId, offset: u64) -> Self {
        Self {
            name: name.into(),
            dispatch_size: DispatchSize::indirect(buffer, offset),
        }
    }

    /// Returns true if this is a direct dispatch.
    #[inline]
    pub fn is_direct(&self) -> bool {
        self.dispatch_size.is_direct()
    }

    /// Returns true if this is an indirect dispatch.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        self.dispatch_size.is_indirect()
    }

    /// Validates the configuration.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        // Validate name is not empty
        if self.name.is_empty() {
            return Some("Compute pass name cannot be empty".to_string());
        }
        // Validate dispatch size
        self.dispatch_size.validate()
    }

    /// Returns the indirect buffer resource ID if this is an indirect dispatch.
    pub fn indirect_buffer(&self) -> Option<ResourceId> {
        match &self.dispatch_size {
            DispatchSize::Indirect { buffer, .. } => Some(*buffer),
            DispatchSize::Direct { .. } => None,
        }
    }
}

impl Default for ComputePassConfig {
    fn default() -> Self {
        Self::new("unnamed_compute_pass")
    }
}

impl fmt::Display for ComputePassConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComputePassConfig(\"{}\", dispatch={})",
            self.name, self.dispatch_size
        )
    }
}

// ---------------------------------------------------------------------------
// Compute Pass Builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing ComputePassConfig.
///
/// Provides a chainable API for building compute pass configurations
/// with resource tracking for dependency analysis.
#[derive(Clone, Debug, Default)]
pub struct ComputePassBuilder {
    /// The config being built.
    config: ComputePassConfig,
    /// Resources explicitly marked as read dependencies.
    reads: Vec<ResourceId>,
    /// Resources explicitly marked as write dependencies.
    writes: Vec<ResourceId>,
}

impl ComputePassBuilder {
    /// Creates a new builder with the given pass name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: ComputePassConfig::new(name),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Sets a direct dispatch with the specified workgroup counts.
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.dispatch_size = DispatchSize::direct(x, y, z);
        self
    }

    /// Sets an indirect dispatch from a buffer.
    ///
    /// The buffer is automatically added as a read dependency.
    pub fn dispatch_indirect(mut self, buffer: ResourceId, offset: u64) -> Self {
        self.config.dispatch_size = DispatchSize::indirect(buffer, offset);
        // Indirect buffer is implicitly a read dependency
        if !self.reads.contains(&buffer) {
            self.reads.push(buffer);
        }
        self
    }

    /// Adds an explicit read dependency.
    ///
    /// Use for resources that are read by the compute shader
    /// (e.g., storage buffers, textures).
    pub fn read_resource(mut self, resource: ResourceId) -> Self {
        if !self.reads.contains(&resource) {
            self.reads.push(resource);
        }
        self
    }

    /// Adds an explicit write dependency.
    ///
    /// Use for resources that are written by the compute shader
    /// (e.g., storage buffers, storage textures).
    pub fn write_resource(mut self, resource: ResourceId) -> Self {
        if !self.writes.contains(&resource) {
            self.writes.push(resource);
        }
        self
    }

    /// Adds a read-write resource dependency.
    ///
    /// Use for resources that are both read and written
    /// (e.g., read-modify-write on a storage buffer).
    pub fn read_write_resource(mut self, resource: ResourceId) -> Self {
        if !self.reads.contains(&resource) {
            self.reads.push(resource);
        }
        if !self.writes.contains(&resource) {
            self.writes.push(resource);
        }
        self
    }

    /// Returns the list of read dependencies.
    pub fn get_reads(&self) -> &[ResourceId] {
        &self.reads
    }

    /// Returns the list of write dependencies.
    pub fn get_writes(&self) -> &[ResourceId] {
        &self.writes
    }

    /// Builds the final ComputePassConfig.
    ///
    /// Consumes the builder.
    pub fn build(self) -> ComputePassConfig {
        self.config
    }

    /// Builds and returns both the config and resource dependencies.
    pub fn build_with_deps(self) -> (ComputePassConfig, Vec<ResourceId>, Vec<ResourceId>) {
        (self.config, self.reads, self.writes)
    }
}

// ---------------------------------------------------------------------------
// Compute Pass Executor Trait
// ---------------------------------------------------------------------------

/// Trait for custom compute pass execution logic.
///
/// Implementations receive the render context and wgpu compute pass
/// and are responsible for setting bind groups and recording dispatches.
pub trait ComputePassExecutor: Send + Sync {
    /// Execute the pass, recording commands to the compute pass.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The render context with frame state
    /// * `pass` - The wgpu compute pass to record commands to
    fn execute(&self, ctx: &mut RenderContext, pass: &mut wgpu::ComputePass);

    /// Optional: Returns a debug name for this executor.
    fn name(&self) -> &str {
        "ComputePassExecutor"
    }
}

/// A no-op compute executor that does nothing.
///
/// Useful for placeholder passes or testing.
#[derive(Clone, Debug, Default)]
pub struct NoOpComputeExecutor;

impl ComputePassExecutor for NoOpComputeExecutor {
    fn execute(&self, _ctx: &mut RenderContext, _pass: &mut wgpu::ComputePass) {
        // No-op
    }

    fn name(&self) -> &str {
        "NoOpComputeExecutor"
    }
}

/// A closure-based compute executor.
pub struct FnComputeExecutor<F: Fn(&mut RenderContext, &mut wgpu::ComputePass) + Send + Sync> {
    func: F,
    name: &'static str,
}

impl<F: Fn(&mut RenderContext, &mut wgpu::ComputePass) + Send + Sync> FnComputeExecutor<F> {
    /// Creates a new function executor.
    pub fn new(func: F) -> Self {
        Self {
            func,
            name: "FnComputeExecutor",
        }
    }

    /// Creates a new function executor with a name.
    pub fn named(name: &'static str, func: F) -> Self {
        Self { func, name }
    }
}

impl<F: Fn(&mut RenderContext, &mut wgpu::ComputePass) + Send + Sync> ComputePassExecutor
    for FnComputeExecutor<F>
{
    fn execute(&self, ctx: &mut RenderContext, pass: &mut wgpu::ComputePass) {
        (self.func)(ctx, pass);
    }

    fn name(&self) -> &str {
        self.name
    }
}

// ---------------------------------------------------------------------------
// Compute Pass Node
// ---------------------------------------------------------------------------

/// A complete compute pass node ready for frame graph scheduling.
///
/// Combines configuration with an executor for compute dispatch.
pub struct ComputePassNode {
    /// Unique identifier assigned by the frame graph.
    pub id: PassId,
    /// The compute pass configuration.
    pub config: ComputePassConfig,
    /// The executor responsible for recording compute commands.
    pub executor: Box<dyn ComputePassExecutor>,
}

impl ComputePassNode {
    /// Creates a new compute pass node.
    pub fn new(
        id: PassId,
        config: ComputePassConfig,
        executor: Box<dyn ComputePassExecutor>,
    ) -> Self {
        Self {
            id,
            config,
            executor,
        }
    }

    /// Creates a compute pass node with a no-op executor.
    pub fn empty(id: PassId, config: ComputePassConfig) -> Self {
        Self {
            id,
            config,
            executor: Box::new(NoOpComputeExecutor),
        }
    }

    /// Creates a compute pass node with a closure executor.
    pub fn with_fn<F>(id: PassId, config: ComputePassConfig, func: F) -> Self
    where
        F: Fn(&mut RenderContext, &mut wgpu::ComputePass) + Send + Sync + 'static,
    {
        Self {
            id,
            config,
            executor: Box::new(FnComputeExecutor::new(func)),
        }
    }

    /// Returns the pass name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Validates the compute pass node.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        self.config.validate()
    }

    /// Returns the indirect buffer resource if this is an indirect dispatch.
    pub fn indirect_buffer(&self) -> Option<ResourceId> {
        self.config.indirect_buffer()
    }
}

impl fmt::Debug for ComputePassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePassNode")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("executor", &self.executor.name())
            .finish()
    }
}

impl fmt::Display for ComputePassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComputePassNode({}, \"{}\", executor={})",
            self.id,
            self.config.name,
            self.executor.name()
        )
    }
}

// ---------------------------------------------------------------------------
// Image Data Layout
// ---------------------------------------------------------------------------

/// Describes the memory layout for buffer data when copying to/from textures.
///
/// This maps to wgpu's `TexelCopyBufferLayout` and specifies how buffer data
/// is organized when used as source or destination for texture copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ImageDataLayout {
    /// Offset in bytes from the start of the buffer.
    pub offset: u64,
    /// Bytes per row of the image data.
    /// If None, tightly packed (no padding between rows).
    pub bytes_per_row: Option<u32>,
    /// Number of rows per image (for 3D textures / arrays).
    /// If None, assumes a single 2D image.
    pub rows_per_image: Option<u32>,
}

impl ImageDataLayout {
    /// Creates a new image data layout with the given offset.
    pub fn new(offset: u64) -> Self {
        Self {
            offset,
            bytes_per_row: None,
            rows_per_image: None,
        }
    }

    /// Creates an image data layout with bytes per row specified.
    pub fn with_bytes_per_row(offset: u64, bytes_per_row: u32) -> Self {
        Self {
            offset,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        }
    }

    /// Creates a full image data layout with all parameters.
    pub fn with_rows_per_image(offset: u64, bytes_per_row: u32, rows_per_image: u32) -> Self {
        Self {
            offset,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(rows_per_image),
        }
    }

    /// Sets the bytes per row.
    pub fn set_bytes_per_row(mut self, bytes_per_row: u32) -> Self {
        self.bytes_per_row = Some(bytes_per_row);
        self
    }

    /// Sets the rows per image.
    pub fn set_rows_per_image(mut self, rows_per_image: u32) -> Self {
        self.rows_per_image = Some(rows_per_image);
        self
    }

    /// Converts to wgpu ImageDataLayout.
    pub fn to_wgpu(&self) -> wgpu::ImageDataLayout {
        wgpu::ImageDataLayout {
            offset: self.offset,
            bytes_per_row: self.bytes_per_row,
            rows_per_image: self.rows_per_image,
        }
    }
}

impl fmt::Display for ImageDataLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImageDataLayout(offset={}, bytes_per_row={:?}, rows_per_image={:?})",
            self.offset, self.bytes_per_row, self.rows_per_image
        )
    }
}

// ---------------------------------------------------------------------------
// Copy Operation
// ---------------------------------------------------------------------------

/// A single copy operation within a copy pass.
///
/// Represents one of four copy operation types:
/// - Buffer to Buffer
/// - Buffer to Texture
/// - Texture to Buffer
/// - Texture to Texture
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CopyOperation {
    /// Copy data between two buffers.
    BufferToBuffer {
        /// Source buffer resource.
        src: ResourceId,
        /// Byte offset in the source buffer.
        src_offset: u64,
        /// Destination buffer resource.
        dst: ResourceId,
        /// Byte offset in the destination buffer.
        dst_offset: u64,
        /// Number of bytes to copy.
        size: u64,
    },
    /// Copy data from a buffer to a texture.
    BufferToTexture {
        /// Source buffer resource.
        src: ResourceId,
        /// Memory layout of the source buffer data.
        src_layout: ImageDataLayout,
        /// Destination texture resource.
        dst: ResourceId,
        /// Target mip level in the destination texture.
        dst_mip: u32,
        /// Origin offset in the destination texture [x, y, z].
        dst_origin: [u32; 3],
        /// Size of the region to copy [width, height, depth].
        size: [u32; 3],
    },
    /// Copy data from a texture to a buffer.
    TextureToBuffer {
        /// Source texture resource.
        src: ResourceId,
        /// Source mip level.
        src_mip: u32,
        /// Origin offset in the source texture [x, y, z].
        src_origin: [u32; 3],
        /// Destination buffer resource.
        dst: ResourceId,
        /// Memory layout for the destination buffer data.
        dst_layout: ImageDataLayout,
        /// Size of the region to copy [width, height, depth].
        size: [u32; 3],
    },
    /// Copy data between two textures.
    TextureToTexture {
        /// Source texture resource.
        src: ResourceId,
        /// Source mip level.
        src_mip: u32,
        /// Origin offset in the source texture [x, y, z].
        src_origin: [u32; 3],
        /// Destination texture resource.
        dst: ResourceId,
        /// Destination mip level.
        dst_mip: u32,
        /// Origin offset in the destination texture [x, y, z].
        dst_origin: [u32; 3],
        /// Size of the region to copy [width, height, depth].
        size: [u32; 3],
    },
}

impl CopyOperation {
    /// Creates a buffer-to-buffer copy operation.
    pub fn buffer_to_buffer(
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        size: u64,
    ) -> Self {
        Self::BufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        }
    }

    /// Creates a buffer-to-texture copy operation with default layout.
    pub fn buffer_to_texture(src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        Self::BufferToTexture {
            src,
            src_layout: ImageDataLayout::default(),
            dst,
            dst_mip: 0,
            dst_origin: [0, 0, 0],
            size,
        }
    }

    /// Creates a buffer-to-texture copy operation with full parameters.
    pub fn buffer_to_texture_full(
        src: ResourceId,
        src_layout: ImageDataLayout,
        dst: ResourceId,
        dst_mip: u32,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) -> Self {
        Self::BufferToTexture {
            src,
            src_layout,
            dst,
            dst_mip,
            dst_origin,
            size,
        }
    }

    /// Creates a texture-to-buffer copy operation with default layout.
    pub fn texture_to_buffer(src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        Self::TextureToBuffer {
            src,
            src_mip: 0,
            src_origin: [0, 0, 0],
            dst,
            dst_layout: ImageDataLayout::default(),
            size,
        }
    }

    /// Creates a texture-to-buffer copy operation with full parameters.
    pub fn texture_to_buffer_full(
        src: ResourceId,
        src_mip: u32,
        src_origin: [u32; 3],
        dst: ResourceId,
        dst_layout: ImageDataLayout,
        size: [u32; 3],
    ) -> Self {
        Self::TextureToBuffer {
            src,
            src_mip,
            src_origin,
            dst,
            dst_layout,
            size,
        }
    }

    /// Creates a texture-to-texture copy operation with default mip and origin.
    pub fn texture_to_texture(src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        Self::TextureToTexture {
            src,
            src_mip: 0,
            src_origin: [0, 0, 0],
            dst,
            dst_mip: 0,
            dst_origin: [0, 0, 0],
            size,
        }
    }

    /// Creates a texture-to-texture copy operation with full parameters.
    pub fn texture_to_texture_full(
        src: ResourceId,
        src_mip: u32,
        src_origin: [u32; 3],
        dst: ResourceId,
        dst_mip: u32,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) -> Self {
        Self::TextureToTexture {
            src,
            src_mip,
            src_origin,
            dst,
            dst_mip,
            dst_origin,
            size,
        }
    }

    /// Returns the source resource ID for this operation.
    pub fn source(&self) -> ResourceId {
        match self {
            Self::BufferToBuffer { src, .. } => *src,
            Self::BufferToTexture { src, .. } => *src,
            Self::TextureToBuffer { src, .. } => *src,
            Self::TextureToTexture { src, .. } => *src,
        }
    }

    /// Returns the destination resource ID for this operation.
    pub fn destination(&self) -> ResourceId {
        match self {
            Self::BufferToBuffer { dst, .. } => *dst,
            Self::BufferToTexture { dst, .. } => *dst,
            Self::TextureToBuffer { dst, .. } => *dst,
            Self::TextureToTexture { dst, .. } => *dst,
        }
    }

    /// Returns true if this is a buffer-to-buffer copy.
    #[inline]
    pub const fn is_buffer_to_buffer(&self) -> bool {
        matches!(self, Self::BufferToBuffer { .. })
    }

    /// Returns true if this is a buffer-to-texture copy.
    #[inline]
    pub const fn is_buffer_to_texture(&self) -> bool {
        matches!(self, Self::BufferToTexture { .. })
    }

    /// Returns true if this is a texture-to-buffer copy.
    #[inline]
    pub const fn is_texture_to_buffer(&self) -> bool {
        matches!(self, Self::TextureToBuffer { .. })
    }

    /// Returns true if this is a texture-to-texture copy.
    #[inline]
    pub const fn is_texture_to_texture(&self) -> bool {
        matches!(self, Self::TextureToTexture { .. })
    }
}

impl fmt::Display for CopyOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                write!(
                    f,
                    "BufferToBuffer({} @{} -> {} @{}, {} bytes)",
                    src, src_offset, dst, dst_offset, size
                )
            }
            Self::BufferToTexture {
                src,
                dst,
                dst_mip,
                size,
                ..
            } => {
                write!(
                    f,
                    "BufferToTexture({} -> {} mip={}, size={:?})",
                    src, dst, dst_mip, size
                )
            }
            Self::TextureToBuffer {
                src,
                src_mip,
                dst,
                size,
                ..
            } => {
                write!(
                    f,
                    "TextureToBuffer({} mip={} -> {}, size={:?})",
                    src, src_mip, dst, size
                )
            }
            Self::TextureToTexture {
                src,
                src_mip,
                dst,
                dst_mip,
                size,
                ..
            } => {
                write!(
                    f,
                    "TextureToTexture({} mip={} -> {} mip={}, size={:?})",
                    src, src_mip, dst, dst_mip, size
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Copy Pass Config
// ---------------------------------------------------------------------------

/// Configuration for a copy pass.
///
/// Contains the name and list of copy operations to be executed.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CopyPassConfig {
    /// Human-readable name for debugging.
    pub name: String,
    /// The list of copy operations to execute.
    pub operations: Vec<CopyOperation>,
}

impl CopyPassConfig {
    /// Creates a new empty copy pass config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            operations: Vec::new(),
        }
    }

    /// Creates a copy pass config with a single operation.
    pub fn with_operation(name: impl Into<String>, operation: CopyOperation) -> Self {
        Self {
            name: name.into(),
            operations: vec![operation],
        }
    }

    /// Creates a copy pass config with multiple operations.
    pub fn with_operations(name: impl Into<String>, operations: Vec<CopyOperation>) -> Self {
        Self {
            name: name.into(),
            operations,
        }
    }

    /// Adds an operation to this config.
    pub fn add_operation(&mut self, operation: CopyOperation) {
        self.operations.push(operation);
    }

    /// Returns true if this config has no operations.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Returns the number of operations in this config.
    #[inline]
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Returns all source resources referenced by this pass.
    pub fn source_resources(&self) -> Vec<ResourceId> {
        self.operations.iter().map(|op| op.source()).collect()
    }

    /// Returns all destination resources referenced by this pass.
    pub fn destination_resources(&self) -> Vec<ResourceId> {
        self.operations.iter().map(|op| op.destination()).collect()
    }

    /// Returns all unique resources referenced by this pass (both sources and destinations).
    pub fn all_resources(&self) -> Vec<ResourceId> {
        let mut resources = Vec::new();
        for op in &self.operations {
            let src = op.source();
            let dst = op.destination();
            if !resources.contains(&src) {
                resources.push(src);
            }
            if !resources.contains(&dst) {
                resources.push(dst);
            }
        }
        resources
    }

    /// Validates the configuration.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        if self.operations.is_empty() {
            return Some("Copy pass must have at least one operation.".to_string());
        }

        // Check for self-copies within a single operation
        for (i, op) in self.operations.iter().enumerate() {
            if op.source() == op.destination() {
                return Some(format!(
                    "Operation {} copies resource {} to itself.",
                    i,
                    op.source()
                ));
            }
        }

        None
    }
}

impl fmt::Display for CopyPassConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CopyPassConfig(\"{}\", operations={})",
            self.name,
            self.operations.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Copy Pass Builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing CopyPassConfig.
///
/// Provides a chainable API for building copy pass configurations
/// with automatic resource tracking for dependency analysis.
#[derive(Clone, Debug, Default)]
pub struct CopyPassBuilder {
    /// The config being built.
    config: CopyPassConfig,
}

impl CopyPassBuilder {
    /// Creates a new builder with the given pass name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: CopyPassConfig::new(name),
        }
    }

    /// Adds a buffer-to-buffer copy operation.
    pub fn copy_buffer(
        mut self,
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        size: u64,
    ) -> Self {
        self.config.operations.push(CopyOperation::buffer_to_buffer(
            src, src_offset, dst, dst_offset, size,
        ));
        self
    }

    /// Adds a buffer-to-texture copy operation.
    pub fn copy_buffer_to_texture(mut self, src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        self.config
            .operations
            .push(CopyOperation::buffer_to_texture(src, dst, size));
        self
    }

    /// Adds a buffer-to-texture copy operation with full parameters.
    pub fn copy_buffer_to_texture_full(
        mut self,
        src: ResourceId,
        src_layout: ImageDataLayout,
        dst: ResourceId,
        dst_mip: u32,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) -> Self {
        self.config.operations.push(CopyOperation::buffer_to_texture_full(
            src, src_layout, dst, dst_mip, dst_origin, size,
        ));
        self
    }

    /// Adds a texture-to-buffer copy operation.
    pub fn copy_texture_to_buffer(mut self, src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        self.config
            .operations
            .push(CopyOperation::texture_to_buffer(src, dst, size));
        self
    }

    /// Adds a texture-to-buffer copy operation with full parameters.
    pub fn copy_texture_to_buffer_full(
        mut self,
        src: ResourceId,
        src_mip: u32,
        src_origin: [u32; 3],
        dst: ResourceId,
        dst_layout: ImageDataLayout,
        size: [u32; 3],
    ) -> Self {
        self.config.operations.push(CopyOperation::texture_to_buffer_full(
            src, src_mip, src_origin, dst, dst_layout, size,
        ));
        self
    }

    /// Adds a texture-to-texture copy operation.
    pub fn copy_texture(mut self, src: ResourceId, dst: ResourceId, size: [u32; 3]) -> Self {
        self.config
            .operations
            .push(CopyOperation::texture_to_texture(src, dst, size));
        self
    }

    /// Adds a texture-to-texture copy operation with full parameters.
    pub fn copy_texture_full(
        mut self,
        src: ResourceId,
        src_mip: u32,
        src_origin: [u32; 3],
        dst: ResourceId,
        dst_mip: u32,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) -> Self {
        self.config.operations.push(CopyOperation::texture_to_texture_full(
            src, src_mip, src_origin, dst, dst_mip, dst_origin, size,
        ));
        self
    }

    /// Returns the list of source resources referenced by the operations.
    pub fn source_resources(&self) -> Vec<ResourceId> {
        self.config.source_resources()
    }

    /// Returns the list of destination resources referenced by the operations.
    pub fn destination_resources(&self) -> Vec<ResourceId> {
        self.config.destination_resources()
    }

    /// Returns all unique resources referenced by the operations.
    pub fn all_resources(&self) -> Vec<ResourceId> {
        self.config.all_resources()
    }

    /// Returns the number of operations added.
    pub fn operation_count(&self) -> usize {
        self.config.operation_count()
    }

    /// Builds the final CopyPassConfig.
    ///
    /// Consumes the builder.
    pub fn build(self) -> CopyPassConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Copy Pass Node
// ---------------------------------------------------------------------------

/// A complete copy pass node ready for frame graph scheduling.
///
/// Combines configuration with a unique identifier for tracking.
#[derive(Clone, Debug)]
pub struct CopyPassNode {
    /// Unique identifier assigned by the frame graph.
    pub id: PassId,
    /// The copy pass configuration.
    pub config: CopyPassConfig,
}

impl CopyPassNode {
    /// Creates a new copy pass node.
    pub fn new(id: PassId, config: CopyPassConfig) -> Self {
        Self { id, config }
    }

    /// Returns the pass name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns all resources written by this pass (destinations).
    pub fn written_resources(&self) -> Vec<ResourceId> {
        self.config.destination_resources()
    }

    /// Returns all resources read by this pass (sources).
    pub fn read_resources(&self) -> Vec<ResourceId> {
        self.config.source_resources()
    }

    /// Returns the number of copy operations.
    pub fn operation_count(&self) -> usize {
        self.config.operation_count()
    }
}

impl fmt::Display for CopyPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CopyPassNode({}, \"{}\", operations={})",
            self.id,
            self.config.name,
            self.config.operations.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing Dispatch Size (T-WGPU-P7.5.7)
// ---------------------------------------------------------------------------

/// Dispatch size configuration for ray tracing passes.
///
/// Specifies how many rays to dispatch, either directly with explicit
/// dimensions or indirectly from a GPU buffer containing the dispatch
/// parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RayDispatchSize {
    /// Direct dispatch with explicit ray dimensions.
    ///
    /// The ray generation shader will be invoked for each pixel/ray in the
    /// specified 3D grid.
    Direct {
        /// Number of rays in the X dimension (typically image width).
        width: u32,
        /// Number of rays in the Y dimension (typically image height).
        height: u32,
        /// Number of rays in the Z dimension (typically 1 for 2D dispatch).
        depth: u32,
    },
    /// Indirect dispatch using parameters from a GPU buffer.
    ///
    /// The dispatch parameters (width, height, depth) are read from the
    /// specified buffer at the given offset.
    Indirect {
        /// The resource ID of the buffer containing dispatch parameters.
        buffer: ResourceId,
        /// Byte offset into the buffer where the dispatch parameters start.
        /// The buffer must contain three u32 values (width, height, depth).
        offset: u64,
    },
}

impl RayDispatchSize {
    /// Creates a direct dispatch with the specified dimensions.
    pub fn direct(width: u32, height: u32, depth: u32) -> Self {
        Self::Direct { width, height, depth }
    }

    /// Creates a 2D direct dispatch (depth = 1).
    pub fn direct_2d(width: u32, height: u32) -> Self {
        Self::Direct { width, height, depth: 1 }
    }

    /// Creates an indirect dispatch from a buffer.
    pub fn indirect(buffer: ResourceId, offset: u64) -> Self {
        Self::Indirect { buffer, offset }
    }

    /// Returns true if this is a direct dispatch.
    #[inline]
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// Returns true if this is an indirect dispatch.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        matches!(self, Self::Indirect { .. })
    }

    /// Returns the total number of rays for direct dispatch.
    ///
    /// Returns None for indirect dispatch.
    pub fn total_rays(&self) -> Option<u64> {
        match self {
            Self::Direct { width, height, depth } => Some(*width as u64 * *height as u64 * *depth as u64),
            Self::Indirect { .. } => None,
        }
    }

    /// Returns the indirect buffer resource ID if this is an indirect dispatch.
    pub fn indirect_buffer(&self) -> Option<ResourceId> {
        match self {
            Self::Indirect { buffer, .. } => Some(*buffer),
            Self::Direct { .. } => None,
        }
    }

    /// Validates the dispatch size.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        match self {
            Self::Direct { width, height, depth } => {
                // Check for zero dimensions
                if *width == 0 || *height == 0 || *depth == 0 {
                    return Some("Ray dispatch dimensions must be non-zero".to_string());
                }
                // Check for excessive dimensions (device limits)
                const MAX_DIM: u32 = 65535;
                if *width > MAX_DIM || *height > MAX_DIM || *depth > MAX_DIM {
                    return Some(format!(
                        "Ray dispatch dimension exceeds maximum ({}): ({}, {}, {})",
                        MAX_DIM, width, height, depth
                    ));
                }
                None
            }
            Self::Indirect { buffer, offset } => {
                if buffer.is_invalid() {
                    return Some("Indirect ray dispatch buffer is invalid".to_string());
                }
                // Offset must be aligned to 4 bytes (size of u32)
                if *offset % 4 != 0 {
                    return Some(format!(
                        "Indirect ray dispatch offset must be 4-byte aligned: {}",
                        offset
                    ));
                }
                None
            }
        }
    }
}

impl Default for RayDispatchSize {
    fn default() -> Self {
        Self::Direct {
            width: 1920,
            height: 1080,
            depth: 1,
        }
    }
}

impl fmt::Display for RayDispatchSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct { width, height, depth } => {
                write!(f, "Direct({}x{}x{})", width, height, depth)
            }
            Self::Indirect { buffer, offset } => {
                write!(f, "Indirect({}, offset={})", buffer, offset)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing Pass Config (T-WGPU-P7.5.7)
// ---------------------------------------------------------------------------

/// Complete configuration for a ray tracing pass.
///
/// Contains the shader binding table (SBT) configuration, dispatch size,
/// and other settings needed to execute a ray tracing dispatch.
///
/// # Shader Binding Table (SBT)
///
/// The SBT is a GPU buffer containing shader group handles organized into
/// three regions:
/// - **Ray Generation**: The entry point shader that generates primary rays
/// - **Miss**: Shaders invoked when a ray misses all geometry
/// - **Hit Group**: Shaders invoked when a ray intersects geometry (closest-hit,
///   any-hit, intersection)
///
/// The offsets specify the byte offset into the SBT buffer where each region begins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RayTracingPassConfig {
    /// Human-readable name for debugging.
    pub name: String,
    /// The resource ID of the Shader Binding Table (SBT) buffer.
    /// This buffer contains the shader group handles for ray generation,
    /// miss, and hit group shaders.
    pub shader_binding_table: Option<ResourceId>,
    /// Byte offset into the SBT where the ray generation shader record begins.
    pub ray_gen_offset: u32,
    /// Byte offset into the SBT where the miss shader records begin.
    pub miss_offset: u32,
    /// Byte offset into the SBT where the hit group shader records begin.
    pub hit_group_offset: u32,
    /// The dispatch size (number of rays to trace).
    pub dispatch_size: RayDispatchSize,
}

impl RayTracingPassConfig {
    /// Creates a new ray tracing pass config with the given name.
    ///
    /// Default dispatch is 1920x1080 (full HD resolution).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shader_binding_table: None,
            ray_gen_offset: 0,
            miss_offset: 0,
            hit_group_offset: 0,
            dispatch_size: RayDispatchSize::default(),
        }
    }

    /// Creates a ray tracing pass config with SBT and direct dispatch.
    pub fn with_sbt_and_dispatch(
        name: impl Into<String>,
        sbt: ResourceId,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            name: name.into(),
            shader_binding_table: Some(sbt),
            ray_gen_offset: 0,
            miss_offset: 0,
            hit_group_offset: 0,
            dispatch_size: RayDispatchSize::direct_2d(width, height),
        }
    }

    /// Creates a ray tracing pass config with full SBT offsets.
    pub fn with_full_sbt(
        name: impl Into<String>,
        sbt: ResourceId,
        ray_gen_offset: u32,
        miss_offset: u32,
        hit_group_offset: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            name: name.into(),
            shader_binding_table: Some(sbt),
            ray_gen_offset,
            miss_offset,
            hit_group_offset,
            dispatch_size: RayDispatchSize::direct_2d(width, height),
        }
    }

    /// Returns true if this is a direct dispatch.
    #[inline]
    pub fn is_direct(&self) -> bool {
        self.dispatch_size.is_direct()
    }

    /// Returns true if this is an indirect dispatch.
    #[inline]
    pub fn is_indirect(&self) -> bool {
        self.dispatch_size.is_indirect()
    }

    /// Returns true if the SBT is configured.
    #[inline]
    pub fn has_sbt(&self) -> bool {
        self.shader_binding_table.is_some()
    }

    /// Returns the indirect buffer resource ID if this is an indirect dispatch.
    pub fn indirect_buffer(&self) -> Option<ResourceId> {
        self.dispatch_size.indirect_buffer()
    }

    /// Validates the configuration.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        // Validate name is not empty
        if self.name.is_empty() {
            return Some("Ray tracing pass name cannot be empty".to_string());
        }

        // Validate SBT is set
        if self.shader_binding_table.is_none() {
            return Some("Ray tracing pass requires a shader binding table".to_string());
        }

        // Validate SBT resource ID
        if let Some(sbt) = &self.shader_binding_table {
            if sbt.is_invalid() {
                return Some("Shader binding table resource ID is invalid".to_string());
            }
        }

        // Validate dispatch size
        self.dispatch_size.validate()
    }
}

impl Default for RayTracingPassConfig {
    fn default() -> Self {
        Self::new("unnamed_ray_tracing_pass")
    }
}

impl fmt::Display for RayTracingPassConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RayTracingPassConfig(\"{}\", sbt={:?}, dispatch={})",
            self.name, self.shader_binding_table, self.dispatch_size
        )
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing Pass Builder (T-WGPU-P7.5.7)
// ---------------------------------------------------------------------------

/// Fluent builder for constructing RayTracingPassConfig.
///
/// Provides a chainable API for building ray tracing pass configurations
/// with resource tracking for dependency analysis.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::frame_graph::passes::*;
/// use renderer_backend::frame_graph::graph::ResourceId;
///
/// let sbt = ResourceId::new(0);
/// let tlas = ResourceId::new(1);
/// let output_image = ResourceId::new(2);
///
/// let (config, reads, writes) = RayTracingPassBuilder::new("primary_rays")
///     .shader_binding_table(sbt)
///     .ray_gen_offset(0)
///     .miss_offset(64)
///     .hit_group_offset(128)
///     .dispatch(1920, 1080, 1)
///     .read_resource(tlas)
///     .write_resource(output_image)
///     .build_with_deps();
/// ```
#[derive(Clone, Debug, Default)]
pub struct RayTracingPassBuilder {
    /// The config being built.
    config: RayTracingPassConfig,
    /// Resources explicitly marked as read dependencies.
    reads: Vec<ResourceId>,
    /// Resources explicitly marked as write dependencies.
    writes: Vec<ResourceId>,
}

impl RayTracingPassBuilder {
    /// Creates a new builder with the given pass name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: RayTracingPassConfig::new(name),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Sets the Shader Binding Table (SBT) buffer.
    ///
    /// The SBT is automatically added as a read dependency.
    pub fn shader_binding_table(mut self, sbt: ResourceId) -> Self {
        self.config.shader_binding_table = Some(sbt);
        // SBT is implicitly a read dependency
        if !self.reads.contains(&sbt) {
            self.reads.push(sbt);
        }
        self
    }

    /// Sets the byte offset for the ray generation shader in the SBT.
    pub fn ray_gen_offset(mut self, offset: u32) -> Self {
        self.config.ray_gen_offset = offset;
        self
    }

    /// Sets the byte offset for the miss shaders in the SBT.
    pub fn miss_offset(mut self, offset: u32) -> Self {
        self.config.miss_offset = offset;
        self
    }

    /// Sets the byte offset for the hit group shaders in the SBT.
    pub fn hit_group_offset(mut self, offset: u32) -> Self {
        self.config.hit_group_offset = offset;
        self
    }

    /// Sets a direct dispatch with the specified dimensions.
    pub fn dispatch(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.config.dispatch_size = RayDispatchSize::direct(width, height, depth);
        self
    }

    /// Sets an indirect dispatch from a buffer.
    ///
    /// The buffer is automatically added as a read dependency.
    pub fn dispatch_indirect(mut self, buffer: ResourceId, offset: u64) -> Self {
        self.config.dispatch_size = RayDispatchSize::indirect(buffer, offset);
        // Indirect buffer is implicitly a read dependency
        if !self.reads.contains(&buffer) {
            self.reads.push(buffer);
        }
        self
    }

    /// Adds an explicit read dependency.
    ///
    /// Use for resources that are read by the ray tracing shaders:
    /// - Top-Level Acceleration Structure (TLAS)
    /// - Textures (materials, environment maps)
    /// - Storage buffers (vertex data, instance data)
    pub fn read_resource(mut self, resource: ResourceId) -> Self {
        if !self.reads.contains(&resource) {
            self.reads.push(resource);
        }
        self
    }

    /// Adds an explicit write dependency.
    ///
    /// Use for resources that are written by the ray tracing shaders:
    /// - Output image (ray traced result)
    /// - Storage buffers (accumulation, debug output)
    pub fn write_resource(mut self, resource: ResourceId) -> Self {
        if !self.writes.contains(&resource) {
            self.writes.push(resource);
        }
        self
    }

    /// Adds a read-write resource dependency.
    ///
    /// Use for resources that are both read and written
    /// (e.g., accumulation buffers for progressive rendering).
    pub fn read_write_resource(mut self, resource: ResourceId) -> Self {
        if !self.reads.contains(&resource) {
            self.reads.push(resource);
        }
        if !self.writes.contains(&resource) {
            self.writes.push(resource);
        }
        self
    }

    /// Returns the list of read dependencies.
    pub fn get_reads(&self) -> &[ResourceId] {
        &self.reads
    }

    /// Returns the list of write dependencies.
    pub fn get_writes(&self) -> &[ResourceId] {
        &self.writes
    }

    /// Builds the final RayTracingPassConfig.
    ///
    /// Consumes the builder.
    pub fn build(self) -> RayTracingPassConfig {
        self.config
    }

    /// Builds and returns both the config and resource dependencies.
    pub fn build_with_deps(self) -> (RayTracingPassConfig, Vec<ResourceId>, Vec<ResourceId>) {
        (self.config, self.reads, self.writes)
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing Pass Executor Trait (T-WGPU-P7.5.7)
// ---------------------------------------------------------------------------

/// Trait for custom ray tracing pass execution logic.
///
/// Implementations receive the render context and are responsible for
/// binding acceleration structures, setting up the SBT, and dispatching rays.
///
/// Note: wgpu does not yet have native ray tracing support, so this trait
/// uses a placeholder approach. When wgpu adds ray tracing support, this
/// will be updated to use the proper ray tracing pass type.
pub trait RayTracingPassExecutor: Send + Sync {
    /// Execute the ray tracing pass.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The render context with frame state
    fn execute(&self, ctx: &mut RenderContext);

    /// Optional: Returns a debug name for this executor.
    fn name(&self) -> &str {
        "RayTracingPassExecutor"
    }
}

/// A no-op ray tracing executor that does nothing.
///
/// Useful for placeholder passes or testing.
#[derive(Clone, Debug, Default)]
pub struct NoOpRayTracingExecutor;

impl RayTracingPassExecutor for NoOpRayTracingExecutor {
    fn execute(&self, _ctx: &mut RenderContext) {
        // No-op
    }

    fn name(&self) -> &str {
        "NoOpRayTracingExecutor"
    }
}

/// A closure-based ray tracing executor.
pub struct FnRayTracingExecutor<F: Fn(&mut RenderContext) + Send + Sync> {
    func: F,
    name: &'static str,
}

impl<F: Fn(&mut RenderContext) + Send + Sync> FnRayTracingExecutor<F> {
    /// Creates a new function executor.
    pub fn new(func: F) -> Self {
        Self {
            func,
            name: "FnRayTracingExecutor",
        }
    }

    /// Creates a new function executor with a name.
    pub fn named(name: &'static str, func: F) -> Self {
        Self { func, name }
    }
}

impl<F: Fn(&mut RenderContext) + Send + Sync> RayTracingPassExecutor for FnRayTracingExecutor<F> {
    fn execute(&self, ctx: &mut RenderContext) {
        (self.func)(ctx);
    }

    fn name(&self) -> &str {
        self.name
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing Pass Node (T-WGPU-P7.5.7)
// ---------------------------------------------------------------------------

/// A complete ray tracing pass node ready for frame graph scheduling.
///
/// Combines configuration with an executor for ray tracing dispatch.
pub struct RayTracingPassNode {
    /// Unique identifier assigned by the frame graph.
    pub id: PassId,
    /// The ray tracing pass configuration.
    pub config: RayTracingPassConfig,
    /// The executor responsible for executing the ray tracing pass.
    pub executor: Box<dyn RayTracingPassExecutor>,
}

impl RayTracingPassNode {
    /// Creates a new ray tracing pass node.
    pub fn new(
        id: PassId,
        config: RayTracingPassConfig,
        executor: Box<dyn RayTracingPassExecutor>,
    ) -> Self {
        Self {
            id,
            config,
            executor,
        }
    }

    /// Creates a ray tracing pass node with a no-op executor.
    pub fn empty(id: PassId, config: RayTracingPassConfig) -> Self {
        Self {
            id,
            config,
            executor: Box::new(NoOpRayTracingExecutor),
        }
    }

    /// Creates a ray tracing pass node with a closure executor.
    pub fn with_fn<F>(id: PassId, config: RayTracingPassConfig, func: F) -> Self
    where
        F: Fn(&mut RenderContext) + Send + Sync + 'static,
    {
        Self {
            id,
            config,
            executor: Box::new(FnRayTracingExecutor::new(func)),
        }
    }

    /// Returns the pass name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Validates the ray tracing pass node.
    ///
    /// Returns an error description if invalid, or None if valid.
    pub fn validate(&self) -> Option<String> {
        self.config.validate()
    }

    /// Returns the SBT resource ID if configured.
    pub fn shader_binding_table(&self) -> Option<ResourceId> {
        self.config.shader_binding_table
    }

    /// Returns the indirect buffer resource if this is an indirect dispatch.
    pub fn indirect_buffer(&self) -> Option<ResourceId> {
        self.config.indirect_buffer()
    }
}

impl fmt::Debug for RayTracingPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RayTracingPassNode")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("executor", &self.executor.name())
            .finish()
    }
}

impl fmt::Display for RayTracingPassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RayTracingPassNode({}, \"{}\", executor={})",
            self.id,
            self.config.name,
            self.executor.name()
        )
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: test_color_attachment_default
    #[test]
    fn test_color_attachment_default() {
        let resource = ResourceId::new(1);
        let attachment = PassColorAttachment::new(resource);

        assert_eq!(attachment.resource, resource);
        assert_eq!(attachment.load_op, PassLoadOp::Clear);
        assert_eq!(attachment.store_op, PassStoreOp::Store);
        assert_eq!(attachment.clear_color, Some([0.0, 0.0, 0.0, 1.0]));
        assert!(attachment.resolve_target.is_none());
    }

    // Test 2: test_color_attachment_with_clear
    #[test]
    fn test_color_attachment_with_clear() {
        let resource = ResourceId::new(2);
        let clear_color = [1.0, 0.5, 0.25, 1.0];
        let attachment = PassColorAttachment::clear(resource, clear_color);

        assert_eq!(attachment.resource, resource);
        assert_eq!(attachment.load_op, PassLoadOp::Clear);
        assert_eq!(attachment.store_op, PassStoreOp::Store);
        assert_eq!(attachment.clear_color, Some(clear_color));
    }

    // Test 3: test_color_attachment_with_resolve
    #[test]
    fn test_color_attachment_with_resolve() {
        let resource = ResourceId::new(3);
        let resolve = ResourceId::new(4);
        let attachment = PassColorAttachment::new(resource).with_resolve(resolve);

        assert!(attachment.has_resolve());
        assert_eq!(attachment.resolve_target, Some(resolve));

        let refs = attachment.referenced_resources();
        assert_eq!(refs.len(), 2);
        assert!(refs.contains(&resource));
        assert!(refs.contains(&resolve));
    }

    // Test 4: test_depth_attachment_default
    #[test]
    fn test_depth_attachment_default() {
        let resource = ResourceId::new(5);
        let attachment = PassDepthAttachment::new(resource);

        assert_eq!(attachment.resource, resource);
        assert_eq!(attachment.depth_load_op, PassLoadOp::Clear);
        assert_eq!(attachment.depth_store_op, PassStoreOp::Store);
        assert_eq!(attachment.stencil_load_op, PassLoadOp::DontCare);
        assert_eq!(attachment.stencil_store_op, PassStoreOp::Discard);
        assert_eq!(attachment.clear_depth, 1.0);
        assert_eq!(attachment.clear_stencil, 0);
        assert!(!attachment.read_only);
        assert!(attachment.writes_depth());
        assert!(!attachment.writes_stencil());
    }

    // Test 5: test_depth_attachment_read_only
    #[test]
    fn test_depth_attachment_read_only() {
        let resource = ResourceId::new(6);
        let attachment = PassDepthAttachment::read_only(resource);

        assert!(attachment.read_only);
        assert!(!attachment.writes_depth());
        assert!(!attachment.writes_stencil());
        assert_eq!(attachment.depth_load_op, PassLoadOp::Load);
    }

    // Test 6: test_depth_attachment_stencil
    #[test]
    fn test_depth_attachment_stencil() {
        let resource = ResourceId::new(7);
        let attachment = PassDepthAttachment::with_stencil(resource)
            .with_clear_stencil(128);

        assert_eq!(attachment.stencil_load_op, PassLoadOp::Clear);
        assert_eq!(attachment.stencil_store_op, PassStoreOp::Store);
        assert_eq!(attachment.clear_stencil, 128);
        assert!(attachment.writes_stencil());
    }

    // Test 7: test_viewport_default
    #[test]
    fn test_viewport_default() {
        let viewport = PassViewport::default();

        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 1080.0);
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
        assert!(viewport.is_valid());
    }

    // Test 8: test_viewport_custom
    #[test]
    fn test_viewport_custom() {
        let viewport = PassViewport::new(100.0, 50.0, 800.0, 600.0)
            .with_depth_range(0.1, 0.9);

        assert_eq!(viewport.x, 100.0);
        assert_eq!(viewport.y, 50.0);
        assert_eq!(viewport.width, 800.0);
        assert_eq!(viewport.height, 600.0);
        assert_eq!(viewport.min_depth, 0.1);
        assert_eq!(viewport.max_depth, 0.9);
        assert!(viewport.is_valid());

        // Test aspect ratio
        let aspect = viewport.aspect_ratio();
        assert!((aspect - 800.0 / 600.0).abs() < 0.001);
    }

    // Test 9: test_render_pass_config_minimal
    #[test]
    fn test_render_pass_config_minimal() {
        let color = PassColorAttachment::new(ResourceId::new(1));
        let config = RenderPassConfig::with_color("minimal_pass", color);

        assert_eq!(config.name, "minimal_pass");
        assert_eq!(config.color_attachments.len(), 1);
        assert!(config.depth_attachment.is_none());
        assert_eq!(config.sample_count, 1);
        assert!(config.viewport.is_none());
        assert!(config.has_color());
        assert!(!config.has_depth());
        assert!(!config.is_multisampled());
        assert!(config.validate().is_none());
    }

    // Test 10: test_render_pass_config_full
    #[test]
    fn test_render_pass_config_full() {
        let color = PassColorAttachment::new(ResourceId::new(1));
        let depth = PassDepthAttachment::new(ResourceId::new(2));
        let mut config = RenderPassConfig::with_color_and_depth("full_pass", color, depth);
        config.sample_count = 4;
        config.viewport = Some(PassViewport::with_size(1280.0, 720.0));

        assert!(config.has_color());
        assert!(config.has_depth());
        assert!(config.is_multisampled());
        assert!(config.viewport.is_some());
        assert!(config.validate().is_none());

        // Check written resources
        let writes = config.written_resources();
        assert!(writes.contains(&ResourceId::new(1)));
        assert!(writes.contains(&ResourceId::new(2)));
    }

    // Test 11: test_render_pass_builder_fluent
    #[test]
    fn test_render_pass_builder_fluent() {
        let color_id = ResourceId::new(10);
        let depth_id = ResourceId::new(11);
        let sampler_id = ResourceId::new(12);

        let (config, reads, writes) = RenderPassBuilder::new("test_pass")
            .add_color_attachment(PassColorAttachment::new(color_id))
            .set_depth_attachment(PassDepthAttachment::new(depth_id))
            .sample_count(4)
            .viewport(PassViewport::with_size(1920.0, 1080.0))
            .read_resource(sampler_id)
            .build_with_deps();

        assert_eq!(config.name, "test_pass");
        assert_eq!(config.sample_count, 4);
        assert!(config.viewport.is_some());

        // Check dependency tracking
        assert!(reads.contains(&sampler_id));
        assert!(writes.contains(&color_id));
        assert!(writes.contains(&depth_id));
    }

    // Test 12: test_render_pass_builder_multiple_colors
    #[test]
    fn test_render_pass_builder_multiple_colors() {
        let config = RenderPassBuilder::new("mrt_pass")
            .add_color_attachment(PassColorAttachment::new(ResourceId::new(1)))
            .add_color_attachment(PassColorAttachment::new(ResourceId::new(2)))
            .add_color_attachment(PassColorAttachment::new(ResourceId::new(3)))
            .build();

        assert_eq!(config.color_attachment_count(), 3);
        assert!(config.validate().is_none());
    }

    // Test 13: test_load_op_variants
    #[test]
    fn test_load_op_variants() {
        assert!(PassLoadOp::Clear.is_clear());
        assert!(!PassLoadOp::Clear.is_load());
        assert!(!PassLoadOp::Clear.is_dont_care());

        assert!(!PassLoadOp::Load.is_clear());
        assert!(PassLoadOp::Load.is_load());
        assert!(!PassLoadOp::Load.is_dont_care());

        assert!(!PassLoadOp::DontCare.is_clear());
        assert!(!PassLoadOp::DontCare.is_load());
        assert!(PassLoadOp::DontCare.is_dont_care());

        // Test default
        assert_eq!(PassLoadOp::default(), PassLoadOp::Clear);
    }

    // Test 14: test_store_op_variants
    #[test]
    fn test_store_op_variants() {
        assert!(PassStoreOp::Store.is_store());
        assert!(!PassStoreOp::Store.is_discard());

        assert!(!PassStoreOp::Discard.is_store());
        assert!(PassStoreOp::Discard.is_discard());

        // Test wgpu conversion
        assert_eq!(PassStoreOp::Store.to_wgpu(), wgpu::StoreOp::Store);
        assert_eq!(PassStoreOp::Discard.to_wgpu(), wgpu::StoreOp::Discard);

        // Test default
        assert_eq!(PassStoreOp::default(), PassStoreOp::Store);
    }

    // Test 15: test_sample_count_validation
    #[test]
    fn test_sample_count_validation() {
        // Valid sample counts
        for count in [1, 2, 4, 8, 16] {
            let mut config = RenderPassConfig::with_color(
                "test",
                PassColorAttachment::new(ResourceId::new(1)),
            );
            config.sample_count = count;
            assert!(
                config.validate().is_none(),
                "Sample count {} should be valid",
                count
            );
        }

        // Invalid sample counts
        for count in [0, 3, 5, 6, 7, 9, 32] {
            let mut config = RenderPassConfig::with_color(
                "test",
                PassColorAttachment::new(ResourceId::new(1)),
            );
            config.sample_count = count;
            assert!(
                config.validate().is_some(),
                "Sample count {} should be invalid",
                count
            );
        }
    }

    // Additional tests beyond the required 15

    #[test]
    fn test_render_pass_node_creation() {
        let config = RenderPassConfig::with_color(
            "node_test",
            PassColorAttachment::new(ResourceId::new(1)),
        );
        let node = RenderPassNode::empty(PassId::new(42), config);

        assert_eq!(node.id, PassId::new(42));
        assert_eq!(node.name(), "node_test");
        assert_eq!(node.executor.name(), "NoOpExecutor");
    }

    #[test]
    fn test_color_attachment_transient() {
        let attachment = PassColorAttachment::transient(ResourceId::new(1));

        assert_eq!(attachment.load_op, PassLoadOp::DontCare);
        assert_eq!(attachment.store_op, PassStoreOp::Discard);
        assert!(attachment.clear_color.is_none());
    }

    #[test]
    fn test_config_empty_attachments_invalid() {
        let config = RenderPassConfig::new("empty");

        let error = config.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("at least one attachment"));
    }

    #[test]
    fn test_config_too_many_colors_invalid() {
        let mut config = RenderPassConfig::new("too_many");
        for i in 0..10 {
            config
                .color_attachments
                .push(PassColorAttachment::new(ResourceId::new(i)));
        }

        let error = config.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("Too many color attachments"));
    }

    #[test]
    fn test_viewport_invalid_dimensions() {
        let invalid1 = PassViewport::new(0.0, 0.0, 0.0, 100.0);
        assert!(!invalid1.is_valid());

        let invalid2 = PassViewport::new(0.0, 0.0, 100.0, 0.0);
        assert!(!invalid2.is_valid());

        let invalid3 = PassViewport::new(0.0, 0.0, 100.0, 100.0).with_depth_range(1.0, 0.0);
        assert!(!invalid3.is_valid());
    }

    #[test]
    fn test_depth_attachment_load() {
        let attachment = PassDepthAttachment::load(ResourceId::new(1));

        assert_eq!(attachment.depth_load_op, PassLoadOp::Load);
        assert!(!attachment.read_only);
        assert!(attachment.writes_depth());
    }

    #[test]
    fn test_render_pass_read_resources() {
        let color = PassColorAttachment {
            resource: ResourceId::new(1),
            load_op: PassLoadOp::Load,
            store_op: PassStoreOp::Store,
            clear_color: None,
            resolve_target: None,
        };
        let depth = PassDepthAttachment::read_only(ResourceId::new(2));

        let config = RenderPassConfig {
            name: "test".to_string(),
            color_attachments: vec![color],
            depth_attachment: Some(depth),
            sample_count: 1,
            viewport: None,
        };

        let reads = config.read_resources();
        assert!(reads.contains(&ResourceId::new(1)));
        assert!(reads.contains(&ResourceId::new(2)));
    }

    #[test]
    fn test_display_traits() {
        // Test Display implementations don't panic
        let _ = format!("{}", PassLoadOp::Clear);
        let _ = format!("{}", PassStoreOp::Store);
        let _ = format!("{}", PassViewport::default());
        let _ = format!("{}", PassColorAttachment::new(ResourceId::new(1)));
        let _ = format!("{}", PassDepthAttachment::new(ResourceId::new(2)));
        let _ = format!(
            "{}",
            RenderPassConfig::with_color("test", PassColorAttachment::new(ResourceId::new(1)))
        );
    }

    // ===========================================================================
    // Compute Pass Tests (T-WGPU-P7.5.6)
    // ===========================================================================

    // Test 1: test_dispatch_size_direct
    #[test]
    fn test_dispatch_size_direct() {
        let dispatch = DispatchSize::direct(64, 32, 16);

        assert!(dispatch.is_direct());
        assert!(!dispatch.is_indirect());

        match dispatch {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 64);
                assert_eq!(y, 32);
                assert_eq!(z, 16);
            }
            _ => panic!("Expected Direct dispatch"),
        }

        // Test total workgroups
        assert_eq!(dispatch.total_workgroups(), Some(64 * 32 * 16));

        // Test validation
        assert!(dispatch.validate().is_none());
    }

    // Test 2: test_dispatch_size_indirect
    #[test]
    fn test_dispatch_size_indirect() {
        let buffer = ResourceId::new(42);
        let dispatch = DispatchSize::indirect(buffer, 128);

        assert!(dispatch.is_indirect());
        assert!(!dispatch.is_direct());

        match dispatch {
            DispatchSize::Indirect {
                buffer: b,
                offset: o,
            } => {
                assert_eq!(b, buffer);
                assert_eq!(o, 128);
            }
            _ => panic!("Expected Indirect dispatch"),
        }

        // Indirect has no known workgroup count
        assert!(dispatch.total_workgroups().is_none());

        // Test validation
        assert!(dispatch.validate().is_none());
    }

    // Test 3: test_compute_pass_config_default
    #[test]
    fn test_compute_pass_config_default() {
        let config = ComputePassConfig::default();

        assert_eq!(config.name, "unnamed_compute_pass");
        assert!(config.is_direct());
        assert!(!config.is_indirect());
        assert!(config.validate().is_none());
    }

    // Test 4: test_compute_pass_config_with_dispatch
    #[test]
    fn test_compute_pass_config_with_dispatch() {
        let config = ComputePassConfig::with_dispatch("particle_sim", 256, 128, 1);

        assert_eq!(config.name, "particle_sim");
        assert!(config.is_direct());

        match config.dispatch_size {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 256);
                assert_eq!(y, 128);
                assert_eq!(z, 1);
            }
            _ => panic!("Expected Direct dispatch"),
        }

        assert!(config.validate().is_none());
    }

    // Test 5: test_compute_pass_builder_new
    #[test]
    fn test_compute_pass_builder_new() {
        let builder = ComputePassBuilder::new("test_compute");
        let config = builder.build();

        assert_eq!(config.name, "test_compute");
        // Default dispatch is 1x1x1
        match config.dispatch_size {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 1);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("Expected Direct dispatch"),
        }
    }

    // Test 6: test_compute_pass_builder_dispatch
    #[test]
    fn test_compute_pass_builder_dispatch() {
        let config = ComputePassBuilder::new("compute")
            .dispatch(128, 64, 32)
            .build();

        match config.dispatch_size {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 128);
                assert_eq!(y, 64);
                assert_eq!(z, 32);
            }
            _ => panic!("Expected Direct dispatch"),
        }
    }

    // Test 7: test_compute_pass_builder_indirect
    #[test]
    fn test_compute_pass_builder_indirect() {
        let buffer = ResourceId::new(99);
        let (config, reads, _writes) = ComputePassBuilder::new("indirect_compute")
            .dispatch_indirect(buffer, 256)
            .build_with_deps();

        assert!(config.is_indirect());

        match config.dispatch_size {
            DispatchSize::Indirect {
                buffer: b,
                offset: o,
            } => {
                assert_eq!(b, buffer);
                assert_eq!(o, 256);
            }
            _ => panic!("Expected Indirect dispatch"),
        }

        // Indirect buffer should be auto-added as read dependency
        assert!(reads.contains(&buffer));
    }

    // Test 8: test_compute_pass_builder_read_resource
    #[test]
    fn test_compute_pass_builder_read_resource() {
        let input = ResourceId::new(1);
        let (config, reads, writes) = ComputePassBuilder::new("read_test")
            .dispatch(16, 16, 1)
            .read_resource(input)
            .build_with_deps();

        assert!(reads.contains(&input));
        assert!(!writes.contains(&input));
        assert_eq!(reads.len(), 1);
    }

    // Test 9: test_compute_pass_builder_write_resource
    #[test]
    fn test_compute_pass_builder_write_resource() {
        let output = ResourceId::new(2);
        let (config, reads, writes) = ComputePassBuilder::new("write_test")
            .dispatch(32, 1, 1)
            .write_resource(output)
            .build_with_deps();

        assert!(!reads.contains(&output));
        assert!(writes.contains(&output));
        assert_eq!(writes.len(), 1);
    }

    // Test 10: test_compute_pass_builder_read_write
    #[test]
    fn test_compute_pass_builder_read_write() {
        let buffer = ResourceId::new(3);
        let (config, reads, writes) = ComputePassBuilder::new("rw_test")
            .dispatch(64, 64, 1)
            .read_write_resource(buffer)
            .build_with_deps();

        // Resource should appear in both reads and writes
        assert!(reads.contains(&buffer));
        assert!(writes.contains(&buffer));
    }

    // Test 11: test_compute_pass_builder_build
    #[test]
    fn test_compute_pass_builder_build() {
        let config = ComputePassBuilder::new("simple_build")
            .dispatch(8, 8, 8)
            .build();

        assert_eq!(config.name, "simple_build");
        assert!(config.is_direct());
        assert!(config.validate().is_none());
    }

    // Test 12: test_compute_pass_builder_build_with_deps
    #[test]
    fn test_compute_pass_builder_build_with_deps() {
        let input1 = ResourceId::new(10);
        let input2 = ResourceId::new(11);
        let output = ResourceId::new(20);

        let (config, reads, writes) = ComputePassBuilder::new("deps_test")
            .dispatch(256, 1, 1)
            .read_resource(input1)
            .read_resource(input2)
            .write_resource(output)
            .build_with_deps();

        assert_eq!(config.name, "deps_test");
        assert_eq!(reads.len(), 2);
        assert_eq!(writes.len(), 1);
        assert!(reads.contains(&input1));
        assert!(reads.contains(&input2));
        assert!(writes.contains(&output));
    }

    // Test 13: test_compute_pass_node_creation
    #[test]
    fn test_compute_pass_node_creation() {
        let config = ComputePassConfig::with_dispatch("node_test", 64, 64, 1);
        let node = ComputePassNode::empty(PassId::new(100), config);

        assert_eq!(node.id, PassId::new(100));
        assert_eq!(node.name(), "node_test");
        assert_eq!(node.executor.name(), "NoOpComputeExecutor");
        assert!(node.validate().is_none());
    }

    // Test 14: test_compute_pass_validate
    #[test]
    fn test_compute_pass_validate() {
        // Valid config
        let valid = ComputePassConfig::with_dispatch("valid", 16, 16, 16);
        assert!(valid.validate().is_none());

        // Invalid: zero dispatch dimension
        let zero_dispatch = ComputePassConfig {
            name: "zero".to_string(),
            dispatch_size: DispatchSize::Direct { x: 0, y: 16, z: 1 },
        };
        assert!(zero_dispatch.validate().is_some());
        assert!(zero_dispatch
            .validate()
            .unwrap()
            .contains("non-zero"));

        // Invalid: empty name
        let empty_name = ComputePassConfig {
            name: "".to_string(),
            dispatch_size: DispatchSize::default(),
        };
        assert!(empty_name.validate().is_some());
        assert!(empty_name
            .validate()
            .unwrap()
            .contains("name"));

        // Invalid: indirect with invalid buffer
        let invalid_buffer = ComputePassConfig {
            name: "bad_indirect".to_string(),
            dispatch_size: DispatchSize::Indirect {
                buffer: ResourceId::INVALID,
                offset: 0,
            },
        };
        assert!(invalid_buffer.validate().is_some());
        assert!(invalid_buffer
            .validate()
            .unwrap()
            .contains("invalid"));

        // Invalid: unaligned indirect offset
        let unaligned = ComputePassConfig {
            name: "unaligned".to_string(),
            dispatch_size: DispatchSize::Indirect {
                buffer: ResourceId::new(1),
                offset: 3, // Not 4-byte aligned
            },
        };
        assert!(unaligned.validate().is_some());
        assert!(unaligned.validate().unwrap().contains("aligned"));
    }

    // Test 15: test_dispatch_size_display
    #[test]
    fn test_dispatch_size_display() {
        let direct = DispatchSize::direct(128, 64, 32);
        let display = format!("{}", direct);
        assert!(display.contains("Direct"));
        assert!(display.contains("128"));
        assert!(display.contains("64"));
        assert!(display.contains("32"));

        let indirect = DispatchSize::indirect(ResourceId::new(5), 1024);
        let display = format!("{}", indirect);
        assert!(display.contains("Indirect"));
        assert!(display.contains("1024"));
    }

    // Additional compute pass tests beyond the required 15

    #[test]
    fn test_dispatch_size_1d_and_2d() {
        let dispatch_1d = DispatchSize::direct_1d(256);
        match dispatch_1d {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 256);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("Expected Direct dispatch"),
        }

        let dispatch_2d = DispatchSize::direct_2d(128, 64);
        match dispatch_2d {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 128);
                assert_eq!(y, 64);
                assert_eq!(z, 1);
            }
            _ => panic!("Expected Direct dispatch"),
        }
    }

    #[test]
    fn test_compute_pass_config_1d_2d() {
        let config_1d = ComputePassConfig::with_dispatch_1d("scan", 1024);
        assert_eq!(config_1d.dispatch_size.total_workgroups(), Some(1024));

        let config_2d = ComputePassConfig::with_dispatch_2d("blur", 64, 64);
        assert_eq!(config_2d.dispatch_size.total_workgroups(), Some(4096));
    }

    #[test]
    fn test_compute_pass_config_indirect_buffer() {
        let buffer = ResourceId::new(42);
        let config = ComputePassConfig::with_indirect("cull", buffer, 0);

        assert_eq!(config.indirect_buffer(), Some(buffer));

        let direct = ComputePassConfig::with_dispatch_1d("direct", 100);
        assert_eq!(direct.indirect_buffer(), None);
    }

    #[test]
    fn test_compute_pass_node_with_fn() {
        let config = ComputePassConfig::with_dispatch("fn_test", 8, 8, 8);
        let node = ComputePassNode::with_fn(PassId::new(1), config, |_ctx, _pass| {
            // Would record commands here
        });

        assert_eq!(node.executor.name(), "FnComputeExecutor");
    }

    #[test]
    fn test_compute_pass_builder_no_duplicate_deps() {
        let resource = ResourceId::new(1);

        let (_, reads, _) = ComputePassBuilder::new("dup_test")
            .read_resource(resource)
            .read_resource(resource) // duplicate
            .read_resource(resource) // another duplicate
            .build_with_deps();

        // Should only have one entry despite adding 3 times
        assert_eq!(reads.len(), 1);
    }

    #[test]
    fn test_compute_pass_display() {
        let config = ComputePassConfig::with_dispatch("display_test", 16, 8, 4);
        let display = format!("{}", config);

        assert!(display.contains("ComputePassConfig"));
        assert!(display.contains("display_test"));
        assert!(display.contains("Direct"));
    }

    #[test]
    fn test_compute_pass_node_display() {
        let config = ComputePassConfig::with_dispatch("node_display", 32, 32, 1);
        let node = ComputePassNode::empty(PassId::new(42), config);

        let display = format!("{}", node);
        assert!(display.contains("ComputePassNode"));
        assert!(display.contains("node_display"));
        assert!(display.contains("NoOpComputeExecutor"));

        let debug = format!("{:?}", node);
        assert!(debug.contains("ComputePassNode"));
    }

    #[test]
    fn test_dispatch_size_validation_limits() {
        // Test maximum dimension validation
        let oversized = DispatchSize::Direct {
            x: 100000,
            y: 1,
            z: 1,
        };
        let error = oversized.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("exceeds maximum"));
    }

    #[test]
    fn test_dispatch_size_default() {
        let default = DispatchSize::default();
        match default {
            DispatchSize::Direct { x, y, z } => {
                assert_eq!(x, 1);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("Default should be Direct"),
        }
    }

    // =========================================================================
    // Copy Pass Tests (T-WGPU-P7.5.8)
    // =========================================================================

    // Test 1: test_copy_operation_buffer_to_buffer
    #[test]
    fn test_copy_operation_buffer_to_buffer() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let op = CopyOperation::buffer_to_buffer(src, 0, dst, 0, 1024);

        assert!(op.is_buffer_to_buffer());
        assert!(!op.is_buffer_to_texture());
        assert!(!op.is_texture_to_buffer());
        assert!(!op.is_texture_to_texture());
        assert_eq!(op.source(), src);
        assert_eq!(op.destination(), dst);

        match op {
            CopyOperation::BufferToBuffer {
                src: s,
                src_offset,
                dst: d,
                dst_offset,
                size,
            } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(src_offset, 0);
                assert_eq!(dst_offset, 0);
                assert_eq!(size, 1024);
            }
            _ => panic!("Expected BufferToBuffer"),
        }
    }

    // Test 2: test_copy_operation_buffer_to_texture
    #[test]
    fn test_copy_operation_buffer_to_texture() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let size = [256, 256, 1];
        let op = CopyOperation::buffer_to_texture(src, dst, size);

        assert!(op.is_buffer_to_texture());
        assert!(!op.is_buffer_to_buffer());
        assert_eq!(op.source(), src);
        assert_eq!(op.destination(), dst);

        match op {
            CopyOperation::BufferToTexture {
                src: s,
                dst: d,
                dst_mip,
                dst_origin,
                size: sz,
                ..
            } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(dst_mip, 0);
                assert_eq!(dst_origin, [0, 0, 0]);
                assert_eq!(sz, size);
            }
            _ => panic!("Expected BufferToTexture"),
        }
    }

    // Test 3: test_copy_operation_texture_to_buffer
    #[test]
    fn test_copy_operation_texture_to_buffer() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let size = [512, 512, 1];
        let op = CopyOperation::texture_to_buffer(src, dst, size);

        assert!(op.is_texture_to_buffer());
        assert!(!op.is_texture_to_texture());
        assert_eq!(op.source(), src);
        assert_eq!(op.destination(), dst);

        match op {
            CopyOperation::TextureToBuffer {
                src: s,
                src_mip,
                src_origin,
                dst: d,
                size: sz,
                ..
            } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(src_mip, 0);
                assert_eq!(src_origin, [0, 0, 0]);
                assert_eq!(sz, size);
            }
            _ => panic!("Expected TextureToBuffer"),
        }
    }

    // Test 4: test_copy_operation_texture_to_texture
    #[test]
    fn test_copy_operation_texture_to_texture() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let size = [1024, 1024, 1];
        let op = CopyOperation::texture_to_texture(src, dst, size);

        assert!(op.is_texture_to_texture());
        assert!(!op.is_buffer_to_buffer());
        assert_eq!(op.source(), src);
        assert_eq!(op.destination(), dst);

        match op {
            CopyOperation::TextureToTexture {
                src: s,
                src_mip,
                src_origin,
                dst: d,
                dst_mip,
                dst_origin,
                size: sz,
            } => {
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(src_mip, 0);
                assert_eq!(dst_mip, 0);
                assert_eq!(src_origin, [0, 0, 0]);
                assert_eq!(dst_origin, [0, 0, 0]);
                assert_eq!(sz, size);
            }
            _ => panic!("Expected TextureToTexture"),
        }
    }

    // Test 5: test_image_data_layout_default
    #[test]
    fn test_image_data_layout_default() {
        let layout = ImageDataLayout::default();

        assert_eq!(layout.offset, 0);
        assert!(layout.bytes_per_row.is_none());
        assert!(layout.rows_per_image.is_none());
    }

    // Test 6: test_image_data_layout_with_row
    #[test]
    fn test_image_data_layout_with_row() {
        let layout = ImageDataLayout::with_bytes_per_row(64, 1024);

        assert_eq!(layout.offset, 64);
        assert_eq!(layout.bytes_per_row, Some(1024));
        assert!(layout.rows_per_image.is_none());

        // Test builder pattern
        let layout2 = ImageDataLayout::new(0)
            .set_bytes_per_row(2048)
            .set_rows_per_image(256);

        assert_eq!(layout2.bytes_per_row, Some(2048));
        assert_eq!(layout2.rows_per_image, Some(256));
    }

    // Test 7: test_copy_pass_config_default
    #[test]
    fn test_copy_pass_config_default() {
        let config = CopyPassConfig::default();

        assert!(config.name.is_empty());
        assert!(config.operations.is_empty());
        assert!(config.is_empty());
        assert_eq!(config.operation_count(), 0);
    }

    // Test 8: test_copy_pass_config_with_ops
    #[test]
    fn test_copy_pass_config_with_ops() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let op = CopyOperation::buffer_to_buffer(src, 0, dst, 0, 512);

        let config = CopyPassConfig::with_operation("copy_pass", op);

        assert_eq!(config.name, "copy_pass");
        assert_eq!(config.operation_count(), 1);
        assert!(!config.is_empty());
        assert!(config.validate().is_none());
    }

    // Test 9: test_copy_pass_builder_new
    #[test]
    fn test_copy_pass_builder_new() {
        let builder = CopyPassBuilder::new("test_copy");
        let config = builder.build();

        assert_eq!(config.name, "test_copy");
        assert!(config.is_empty());
    }

    // Test 10: test_copy_pass_builder_copy_buffer
    #[test]
    fn test_copy_pass_builder_copy_buffer() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);

        let config = CopyPassBuilder::new("buffer_copy")
            .copy_buffer(src, 0, dst, 256, 1024)
            .build();

        assert_eq!(config.operation_count(), 1);
        assert!(config.operations[0].is_buffer_to_buffer());
    }

    // Test 11: test_copy_pass_builder_copy_texture
    #[test]
    fn test_copy_pass_builder_copy_texture() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);

        let config = CopyPassBuilder::new("texture_copy")
            .copy_texture(src, dst, [256, 256, 1])
            .build();

        assert_eq!(config.operation_count(), 1);
        assert!(config.operations[0].is_texture_to_texture());
    }

    // Test 12: test_copy_pass_builder_multiple_ops
    #[test]
    fn test_copy_pass_builder_multiple_ops() {
        let buf1 = ResourceId::new(1);
        let buf2 = ResourceId::new(2);
        let tex1 = ResourceId::new(3);
        let tex2 = ResourceId::new(4);

        let config = CopyPassBuilder::new("multi_copy")
            .copy_buffer(buf1, 0, buf2, 0, 1024)
            .copy_texture(tex1, tex2, [512, 512, 1])
            .copy_buffer_to_texture(buf1, tex1, [256, 256, 1])
            .build();

        assert_eq!(config.operation_count(), 3);
        assert!(config.operations[0].is_buffer_to_buffer());
        assert!(config.operations[1].is_texture_to_texture());
        assert!(config.operations[2].is_buffer_to_texture());
    }

    // Test 13: test_copy_pass_source_resources
    #[test]
    fn test_copy_pass_source_resources() {
        let src1 = ResourceId::new(1);
        let src2 = ResourceId::new(2);
        let dst1 = ResourceId::new(3);
        let dst2 = ResourceId::new(4);

        let builder = CopyPassBuilder::new("test")
            .copy_buffer(src1, 0, dst1, 0, 512)
            .copy_texture(src2, dst2, [256, 256, 1]);

        let sources = builder.source_resources();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&src1));
        assert!(sources.contains(&src2));
    }

    // Test 14: test_copy_pass_destination_resources
    #[test]
    fn test_copy_pass_destination_resources() {
        let src1 = ResourceId::new(1);
        let src2 = ResourceId::new(2);
        let dst1 = ResourceId::new(3);
        let dst2 = ResourceId::new(4);

        let builder = CopyPassBuilder::new("test")
            .copy_buffer(src1, 0, dst1, 0, 512)
            .copy_texture(src2, dst2, [256, 256, 1]);

        let destinations = builder.destination_resources();
        assert_eq!(destinations.len(), 2);
        assert!(destinations.contains(&dst1));
        assert!(destinations.contains(&dst2));
    }

    // Test 15: test_copy_pass_node_creation
    #[test]
    fn test_copy_pass_node_creation() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let config = CopyPassConfig::with_operation(
            "copy_node",
            CopyOperation::buffer_to_buffer(src, 0, dst, 0, 256),
        );

        let node = CopyPassNode::new(PassId::new(42), config);

        assert_eq!(node.id, PassId::new(42));
        assert_eq!(node.name(), "copy_node");
        assert_eq!(node.operation_count(), 1);

        let reads = node.read_resources();
        let writes = node.written_resources();
        assert!(reads.contains(&src));
        assert!(writes.contains(&dst));
    }

    // Additional Copy Pass Tests

    #[test]
    fn test_copy_operation_full_constructors() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);
        let layout = ImageDataLayout::with_bytes_per_row(128, 4096);

        // Buffer to texture with full params
        let op1 = CopyOperation::buffer_to_texture_full(
            src,
            layout,
            dst,
            2,
            [10, 20, 0],
            [128, 128, 1],
        );
        match op1 {
            CopyOperation::BufferToTexture {
                dst_mip,
                dst_origin,
                ..
            } => {
                assert_eq!(dst_mip, 2);
                assert_eq!(dst_origin, [10, 20, 0]);
            }
            _ => panic!("Expected BufferToTexture"),
        }

        // Texture to buffer with full params
        let op2 = CopyOperation::texture_to_buffer_full(
            src,
            3,
            [5, 5, 0],
            dst,
            layout,
            [64, 64, 1],
        );
        match op2 {
            CopyOperation::TextureToBuffer {
                src_mip,
                src_origin,
                ..
            } => {
                assert_eq!(src_mip, 3);
                assert_eq!(src_origin, [5, 5, 0]);
            }
            _ => panic!("Expected TextureToBuffer"),
        }

        // Texture to texture with full params
        let op3 = CopyOperation::texture_to_texture_full(
            src,
            1,
            [0, 0, 0],
            dst,
            2,
            [32, 32, 0],
            [512, 512, 1],
        );
        match op3 {
            CopyOperation::TextureToTexture {
                src_mip,
                dst_mip,
                dst_origin,
                ..
            } => {
                assert_eq!(src_mip, 1);
                assert_eq!(dst_mip, 2);
                assert_eq!(dst_origin, [32, 32, 0]);
            }
            _ => panic!("Expected TextureToTexture"),
        }
    }

    #[test]
    fn test_copy_pass_config_validation() {
        // Empty config should fail validation
        let empty = CopyPassConfig::new("empty");
        assert!(empty.validate().is_some());
        assert!(empty.validate().unwrap().contains("at least one operation"));

        // Self-copy should fail validation
        let same_res = ResourceId::new(1);
        let self_copy = CopyPassConfig::with_operation(
            "self_copy",
            CopyOperation::buffer_to_buffer(same_res, 0, same_res, 0, 512),
        );
        assert!(self_copy.validate().is_some());
        assert!(self_copy.validate().unwrap().contains("to itself"));
    }

    #[test]
    fn test_copy_pass_config_all_resources() {
        let r1 = ResourceId::new(1);
        let r2 = ResourceId::new(2);
        let r3 = ResourceId::new(3);

        let mut config = CopyPassConfig::new("test");
        config.add_operation(CopyOperation::buffer_to_buffer(r1, 0, r2, 0, 512));
        config.add_operation(CopyOperation::texture_to_texture(r2, r3, [256, 256, 1]));

        let all = config.all_resources();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&r1));
        assert!(all.contains(&r2));
        assert!(all.contains(&r3));
    }

    #[test]
    fn test_image_data_layout_to_wgpu() {
        let layout = ImageDataLayout::with_rows_per_image(256, 4096, 128);
        let wgpu_layout = layout.to_wgpu();

        assert_eq!(wgpu_layout.offset, 256);
        assert_eq!(wgpu_layout.bytes_per_row, Some(4096));
        assert_eq!(wgpu_layout.rows_per_image, Some(128));
    }

    #[test]
    fn test_copy_operation_display() {
        let src = ResourceId::new(1);
        let dst = ResourceId::new(2);

        let op1 = CopyOperation::buffer_to_buffer(src, 64, dst, 128, 1024);
        let display1 = format!("{}", op1);
        assert!(display1.contains("BufferToBuffer"));
        assert!(display1.contains("1024 bytes"));

        let op2 = CopyOperation::texture_to_texture(src, dst, [512, 512, 1]);
        let display2 = format!("{}", op2);
        assert!(display2.contains("TextureToTexture"));
        assert!(display2.contains("512"));
    }

    #[test]
    fn test_copy_pass_node_display() {
        let config = CopyPassConfig::with_operation(
            "my_copy",
            CopyOperation::buffer_to_buffer(ResourceId::new(1), 0, ResourceId::new(2), 0, 256),
        );
        let node = CopyPassNode::new(PassId::new(10), config);

        let display = format!("{}", node);
        assert!(display.contains("CopyPassNode"));
        assert!(display.contains("my_copy"));
        assert!(display.contains("operations=1"));
    }

    #[test]
    fn test_copy_pass_builder_full_methods() {
        let src_buf = ResourceId::new(1);
        let dst_tex = ResourceId::new(2);
        let src_tex = ResourceId::new(3);
        let dst_buf = ResourceId::new(4);

        let layout = ImageDataLayout::with_bytes_per_row(0, 4096);

        let config = CopyPassBuilder::new("full_test")
            .copy_buffer_to_texture_full(src_buf, layout, dst_tex, 1, [10, 10, 0], [256, 256, 1])
            .copy_texture_to_buffer_full(src_tex, 2, [0, 0, 0], dst_buf, layout, [128, 128, 1])
            .copy_texture_full(src_tex, 0, [0, 0, 0], dst_tex, 1, [64, 64, 0], [64, 64, 1])
            .build();

        assert_eq!(config.operation_count(), 3);

        let all_res = config.all_resources();
        assert!(all_res.contains(&src_buf));
        assert!(all_res.contains(&dst_tex));
        assert!(all_res.contains(&src_tex));
        assert!(all_res.contains(&dst_buf));
    }

    // -------------------------------------------------------------------------
    // Ray Tracing Pass Tests (T-WGPU-P7.5.7)
    // -------------------------------------------------------------------------

    // Test: RayDispatchSize direct creation
    #[test]
    fn test_ray_dispatch_size_direct() {
        let dispatch = RayDispatchSize::direct(1920, 1080, 1);
        assert!(dispatch.is_direct());
        assert!(!dispatch.is_indirect());
        assert_eq!(dispatch.total_rays(), Some(1920 * 1080));
        assert!(dispatch.indirect_buffer().is_none());
        assert!(dispatch.validate().is_none());
    }

    // Test: RayDispatchSize 2D creation
    #[test]
    fn test_ray_dispatch_size_direct_2d() {
        let dispatch = RayDispatchSize::direct_2d(3840, 2160);
        match dispatch {
            RayDispatchSize::Direct { width, height, depth } => {
                assert_eq!(width, 3840);
                assert_eq!(height, 2160);
                assert_eq!(depth, 1);
            }
            _ => panic!("Expected Direct dispatch"),
        }
    }

    // Test: RayDispatchSize indirect creation
    #[test]
    fn test_ray_dispatch_size_indirect() {
        let buffer = ResourceId::new(42);
        let dispatch = RayDispatchSize::indirect(buffer, 64);
        assert!(dispatch.is_indirect());
        assert!(!dispatch.is_direct());
        assert_eq!(dispatch.total_rays(), None);
        assert_eq!(dispatch.indirect_buffer(), Some(buffer));
        assert!(dispatch.validate().is_none());
    }

    // Test: RayDispatchSize validation with zero dimensions
    #[test]
    fn test_ray_dispatch_size_validation_zero() {
        let dispatch = RayDispatchSize::direct(0, 1080, 1);
        let error = dispatch.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("non-zero"));
    }

    // Test: RayDispatchSize validation with excessive dimensions
    #[test]
    fn test_ray_dispatch_size_validation_excessive() {
        let dispatch = RayDispatchSize::direct(100000, 1080, 1);
        let error = dispatch.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("exceeds maximum"));
    }

    // Test: RayDispatchSize indirect validation with invalid buffer
    #[test]
    fn test_ray_dispatch_size_indirect_validation_invalid_buffer() {
        let dispatch = RayDispatchSize::indirect(ResourceId::INVALID, 0);
        let error = dispatch.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("invalid"));
    }

    // Test: RayDispatchSize indirect validation with unaligned offset
    #[test]
    fn test_ray_dispatch_size_indirect_validation_unaligned() {
        let buffer = ResourceId::new(1);
        let dispatch = RayDispatchSize::indirect(buffer, 3); // Not 4-byte aligned
        let error = dispatch.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("4-byte aligned"));
    }

    // Test: RayDispatchSize display formatting
    #[test]
    fn test_ray_dispatch_size_display() {
        let direct = RayDispatchSize::direct(1920, 1080, 1);
        let direct_str = format!("{}", direct);
        assert!(direct_str.contains("Direct"));
        assert!(direct_str.contains("1920"));

        let indirect = RayDispatchSize::indirect(ResourceId::new(5), 128);
        let indirect_str = format!("{}", indirect);
        assert!(indirect_str.contains("Indirect"));
        assert!(indirect_str.contains("offset=128"));
    }

    // Test: RayTracingPassConfig creation
    #[test]
    fn test_ray_tracing_pass_config_new() {
        let config = RayTracingPassConfig::new("test_rt_pass");
        assert_eq!(config.name, "test_rt_pass");
        assert!(config.shader_binding_table.is_none());
        assert_eq!(config.ray_gen_offset, 0);
        assert_eq!(config.miss_offset, 0);
        assert_eq!(config.hit_group_offset, 0);
        assert!(config.dispatch_size.is_direct());
    }

    // Test: RayTracingPassConfig with SBT and dispatch
    #[test]
    fn test_ray_tracing_pass_config_with_sbt_and_dispatch() {
        let sbt = ResourceId::new(10);
        let config = RayTracingPassConfig::with_sbt_and_dispatch("rt_pass", sbt, 1920, 1080);
        assert_eq!(config.shader_binding_table, Some(sbt));
        assert!(config.has_sbt());
        assert!(config.is_direct());
    }

    // Test: RayTracingPassConfig with full SBT
    #[test]
    fn test_ray_tracing_pass_config_with_full_sbt() {
        let sbt = ResourceId::new(20);
        let config = RayTracingPassConfig::with_full_sbt(
            "full_rt_pass",
            sbt,
            0,    // ray_gen_offset
            64,   // miss_offset
            128,  // hit_group_offset
            3840,
            2160,
        );
        assert_eq!(config.ray_gen_offset, 0);
        assert_eq!(config.miss_offset, 64);
        assert_eq!(config.hit_group_offset, 128);
    }

    // Test: RayTracingPassConfig validation
    #[test]
    fn test_ray_tracing_pass_config_validation() {
        // Config without SBT should fail
        let config = RayTracingPassConfig::new("no_sbt");
        let error = config.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("shader binding table"));

        // Config with valid SBT should pass
        let sbt = ResourceId::new(1);
        let valid_config = RayTracingPassConfig::with_sbt_and_dispatch("valid", sbt, 1920, 1080);
        assert!(valid_config.validate().is_none());
    }

    // Test: RayTracingPassBuilder fluent API
    #[test]
    fn test_ray_tracing_pass_builder_fluent_api() {
        let sbt = ResourceId::new(1);
        let tlas = ResourceId::new(2);
        let output = ResourceId::new(3);

        let builder = RayTracingPassBuilder::new("rt_builder_test")
            .shader_binding_table(sbt)
            .ray_gen_offset(0)
            .miss_offset(64)
            .hit_group_offset(128)
            .dispatch(1920, 1080, 1)
            .read_resource(tlas)
            .write_resource(output);

        assert!(builder.get_reads().contains(&sbt));  // SBT is auto-added as read
        assert!(builder.get_reads().contains(&tlas));
        assert!(builder.get_writes().contains(&output));
    }

    // Test: RayTracingPassBuilder with indirect dispatch
    #[test]
    fn test_ray_tracing_pass_builder_indirect() {
        let sbt = ResourceId::new(1);
        let indirect_buffer = ResourceId::new(2);

        let builder = RayTracingPassBuilder::new("indirect_rt")
            .shader_binding_table(sbt)
            .dispatch_indirect(indirect_buffer, 0);

        // Both SBT and indirect buffer should be in reads
        assert!(builder.get_reads().contains(&sbt));
        assert!(builder.get_reads().contains(&indirect_buffer));
    }

    // Test: RayTracingPassBuilder build_with_deps
    #[test]
    fn test_ray_tracing_pass_builder_build_with_deps() {
        let sbt = ResourceId::new(1);
        let tlas = ResourceId::new(2);
        let output = ResourceId::new(3);

        let (config, reads, writes) = RayTracingPassBuilder::new("deps_test")
            .shader_binding_table(sbt)
            .dispatch(1920, 1080, 1)
            .read_resource(tlas)
            .write_resource(output)
            .build_with_deps();

        assert_eq!(config.name, "deps_test");
        assert!(reads.contains(&sbt));
        assert!(reads.contains(&tlas));
        assert!(writes.contains(&output));
    }

    // Test: RayTracingPassBuilder read_write_resource
    #[test]
    fn test_ray_tracing_pass_builder_read_write_resource() {
        let sbt = ResourceId::new(1);
        let accumulator = ResourceId::new(4);

        let builder = RayTracingPassBuilder::new("rw_test")
            .shader_binding_table(sbt)
            .read_write_resource(accumulator);

        assert!(builder.get_reads().contains(&accumulator));
        assert!(builder.get_writes().contains(&accumulator));
    }

    // Test: RayTracingPassNode creation
    #[test]
    fn test_ray_tracing_pass_node_new() {
        let sbt = ResourceId::new(1);
        let config = RayTracingPassConfig::with_sbt_and_dispatch("rt_node_test", sbt, 1920, 1080);
        let node = RayTracingPassNode::empty(PassId::new(42), config);

        assert_eq!(node.id, PassId::new(42));
        assert_eq!(node.name(), "rt_node_test");
        assert_eq!(node.shader_binding_table(), Some(sbt));
        assert!(node.validate().is_none());
    }

    // Test: RayTracingPassNode with closure executor
    #[test]
    fn test_ray_tracing_pass_node_with_fn() {
        let sbt = ResourceId::new(1);
        let config = RayTracingPassConfig::with_sbt_and_dispatch("fn_rt", sbt, 800, 600);
        let node = RayTracingPassNode::with_fn(PassId::new(99), config, |_ctx| {
            // Custom execution logic
        });

        assert_eq!(node.executor.name(), "FnRayTracingExecutor");
    }

    // Test: RayTracingPassNode display formatting
    #[test]
    fn test_ray_tracing_pass_node_display() {
        let sbt = ResourceId::new(1);
        let config = RayTracingPassConfig::with_sbt_and_dispatch("display_test", sbt, 1920, 1080);
        let node = RayTracingPassNode::empty(PassId::new(7), config);

        let display = format!("{}", node);
        assert!(display.contains("RayTracingPassNode"));
        assert!(display.contains("display_test"));
        assert!(display.contains("NoOpRayTracingExecutor"));
    }

    // Test: RayTracingPassNode debug formatting
    #[test]
    fn test_ray_tracing_pass_node_debug() {
        let sbt = ResourceId::new(1);
        let config = RayTracingPassConfig::with_sbt_and_dispatch("debug_test", sbt, 1920, 1080);
        let node = RayTracingPassNode::empty(PassId::new(8), config);

        let debug = format!("{:?}", node);
        assert!(debug.contains("RayTracingPassNode"));
        assert!(debug.contains("id"));
        assert!(debug.contains("config"));
    }

    // Test: RayDispatchSize default
    #[test]
    fn test_ray_dispatch_size_default() {
        let default = RayDispatchSize::default();
        match default {
            RayDispatchSize::Direct { width, height, depth } => {
                assert_eq!(width, 1920);
                assert_eq!(height, 1080);
                assert_eq!(depth, 1);
            }
            _ => panic!("Default should be Direct dispatch"),
        }
    }
}
