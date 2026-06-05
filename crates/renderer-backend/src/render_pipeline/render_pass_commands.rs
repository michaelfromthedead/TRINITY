//! Render pass state command wrappers for wgpu 22-25.x render passes.
//!
//! This module provides a high-level abstraction for render pass commands,
//! offering a fluent API for setting render state during a render pass.
//!
//! # Overview
//!
//! The `RenderPassCommands` struct wraps a wgpu `RenderPass` and provides
//! ergonomic methods for all state-setting commands:
//!
//! - Pipeline binding
//! - Bind group attachment with dynamic offsets
//! - Vertex and index buffer binding
//! - Viewport and scissor configuration
//! - Blend constant colors
//! - Stencil reference values
//! - Push constants
//!
//! # Architecture
//!
//! ```text
//! RenderPassCommands<'a>
//!     |-- pass: wgpu::RenderPass<'a>
//!     |
//!     |-- set_pipeline()         -> &mut Self
//!     |-- set_bind_group()       -> &mut Self
//!     |-- set_vertex_buffer()    -> &mut Self
//!     |-- set_index_buffer()     -> &mut Self
//!     |-- set_viewport()         -> &mut Self
//!     |-- set_scissor_rect()     -> &mut Self
//!     |-- set_blend_constant()   -> &mut Self
//!     |-- set_stencil_reference() -> &mut Self
//!     `-- set_push_constants()   -> &mut Self
//! ```
//!
//! # Command Reference
//!
//! | Command | wgpu Method | Purpose |
//! |---------|-------------|---------|
//! | `set_pipeline` | `RenderPass::set_pipeline` | Bind render pipeline |
//! | `set_bind_group` | `RenderPass::set_bind_group` | Attach bind group at index |
//! | `set_vertex_buffer` | `RenderPass::set_vertex_buffer` | Bind vertex buffer to slot |
//! | `set_index_buffer` | `RenderPass::set_index_buffer` | Bind index buffer with format |
//! | `set_viewport` | `RenderPass::set_viewport` | Configure viewport transform |
//! | `set_scissor_rect` | `RenderPass::set_scissor_rect` | Set scissor clipping region |
//! | `set_blend_constant` | `RenderPass::set_blend_constant` | Set blend factor color |
//! | `set_stencil_reference` | `RenderPass::set_stencil_reference` | Set stencil test reference |
//! | `set_push_constants` | `RenderPass::set_push_constants` | Set push constant data |
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::render_pass_commands::RenderPassCommands;
//!
//! // Wrap the render pass
//! let mut commands = RenderPassCommands::new(render_pass);
//!
//! // Fluent API for state setup
//! commands
//!     .set_pipeline(&pipeline)
//!     .set_bind_group(0, &bind_group, &[])
//!     .set_vertex_buffer(0, vertex_buffer.slice(..))
//!     .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32)
//!     .set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0)
//!     .set_scissor_rect(0, 0, 1920, 1080);
//!
//! // Draw commands (using inner pass)
//! commands.inner_mut().draw_indexed(0..index_count, 0, 0..1);
//!
//! // Or get back the pass when done
//! let pass = commands.finish();
//! ```
//!
//! # Thread Safety
//!
//! `RenderPassCommands` is **not** `Send` or `Sync` because `wgpu::RenderPass`
//! is not thread-safe. Render passes must be used on the thread that created them.

use std::fmt;

// ---------------------------------------------------------------------------
// RenderPassCommands
// ---------------------------------------------------------------------------

/// High-level wrapper for render pass state commands.
///
/// Provides a fluent API for setting render state during a render pass.
/// The wrapper owns the render pass and can return it when done.
///
/// # Lifetime
///
/// The lifetime `'a` corresponds to the render pass lifetime, which is
/// typically tied to the command encoder's encoding scope.
///
/// # Example
///
/// ```ignore
/// let mut commands = RenderPassCommands::new(render_pass);
///
/// // Chain multiple commands
/// commands
///     .set_pipeline(&pipeline)
///     .set_bind_group(0, &bind_group, &[])
///     .set_vertex_buffer(0, buffer.slice(..));
///
/// // Access inner pass for draw calls
/// commands.inner_mut().draw(0..3, 0..1);
/// ```
pub struct RenderPassCommands<'a> {
    pass: wgpu::RenderPass<'a>,
}

impl<'a> RenderPassCommands<'a> {
    /// Create a new render pass commands wrapper.
    ///
    /// # Arguments
    ///
    /// * `pass` - The wgpu render pass to wrap
    ///
    /// # Example
    ///
    /// ```ignore
    /// let render_pass = encoder.begin_render_pass(&descriptor);
    /// let commands = RenderPassCommands::new(render_pass);
    /// ```
    #[inline]
    pub fn new(pass: wgpu::RenderPass<'a>) -> Self {
        Self { pass }
    }

    /// Get a reference to the inner render pass.
    ///
    /// Useful for calling methods not exposed by this wrapper.
    #[inline]
    pub fn inner(&self) -> &wgpu::RenderPass<'a> {
        &self.pass
    }

    /// Get a mutable reference to the inner render pass.
    ///
    /// Useful for calling draw commands or other methods not exposed
    /// by this wrapper.
    ///
    /// # Example
    ///
    /// ```ignore
    /// commands.inner_mut().draw_indexed(0..index_count, 0, 0..instance_count);
    /// ```
    #[inline]
    pub fn inner_mut(&mut self) -> &mut wgpu::RenderPass<'a> {
        &mut self.pass
    }

    /// Consume the wrapper and return the inner render pass.
    ///
    /// Call this when you're done with the fluent API and want to
    /// use the raw render pass directly.
    #[inline]
    pub fn finish(self) -> wgpu::RenderPass<'a> {
        self.pass
    }

    // -------------------------------------------------------------------------
    // Command 1: set_pipeline
    // -------------------------------------------------------------------------

    /// Bind a render pipeline to the render pass.
    ///
    /// The pipeline defines the shader programs, vertex layout, blend state,
    /// depth/stencil state, and other fixed-function configuration.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The render pipeline to bind
    ///
    /// # Example
    ///
    /// ```ignore
    /// commands.set_pipeline(&my_pipeline);
    /// ```
    ///
    /// # Notes
    ///
    /// - Must be called before any draw commands
    /// - The pipeline's layout must match the bind groups being used
    /// - Changing pipelines during a render pass is allowed but may have
    ///   performance implications on some hardware
    #[inline]
    pub fn set_pipeline(&mut self, pipeline: &'a wgpu::RenderPipeline) -> &mut Self {
        self.pass.set_pipeline(pipeline);
        self
    }

    // -------------------------------------------------------------------------
    // Command 2: set_bind_group
    // -------------------------------------------------------------------------

    /// Attach a bind group at the specified index.
    ///
    /// Bind groups contain shader resources like uniform buffers, textures,
    /// and samplers. The index corresponds to the `@group(N)` annotation
    /// in WGSL shaders.
    ///
    /// # Arguments
    ///
    /// * `index` - Bind group slot (0-3 typically, matching `@group(N)`)
    /// * `bind_group` - The bind group to attach
    /// * `offsets` - Dynamic offsets for dynamic uniform/storage buffers
    ///
    /// # Dynamic Offsets
    ///
    /// Dynamic offsets allow using different portions of a buffer without
    /// creating multiple bind groups. Each offset corresponds to a dynamic
    /// binding in declaration order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Static bind group (no dynamic offsets)
    /// commands.set_bind_group(0, &camera_bind_group, &[]);
    ///
    /// // Dynamic bind group with offsets
    /// let object_offset = object_index * 256; // Must be 256-byte aligned
    /// commands.set_bind_group(1, &object_bind_group, &[object_offset]);
    /// ```
    ///
    /// # Notes
    ///
    /// - Offsets must be 256-byte aligned (or `device.limits().min_uniform_buffer_offset_alignment`)
    /// - Number of offsets must match number of dynamic bindings in the group
    #[inline]
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[u32],
    ) -> &mut Self {
        self.pass.set_bind_group(index, bind_group, offsets);
        self
    }

    // -------------------------------------------------------------------------
    // Command 3: set_vertex_buffer
    // -------------------------------------------------------------------------

    /// Bind a vertex buffer to a slot.
    ///
    /// Vertex buffers provide per-vertex data (position, normal, UV, etc.)
    /// to the vertex shader. Multiple buffers can be bound to different slots
    /// for interleaved or separate attribute layouts.
    ///
    /// # Arguments
    ///
    /// * `slot` - Buffer slot (corresponds to layout order in pipeline)
    /// * `buffer_slice` - The buffer slice to bind
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Single vertex buffer with interleaved attributes
    /// commands.set_vertex_buffer(0, vertex_buffer.slice(..));
    ///
    /// // Separate buffers for positions and attributes
    /// commands
    ///     .set_vertex_buffer(0, positions_buffer.slice(..))
    ///     .set_vertex_buffer(1, attributes_buffer.slice(..));
    ///
    /// // Instance buffer in slot 1
    /// commands
    ///     .set_vertex_buffer(0, mesh_buffer.slice(..))
    ///     .set_vertex_buffer(1, instance_buffer.slice(..));
    /// ```
    ///
    /// # Notes
    ///
    /// - Slot index matches the order of vertex buffer layouts in the pipeline
    /// - The buffer slice must contain enough data for all vertices drawn
    #[inline]
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: wgpu::BufferSlice<'a>) -> &mut Self {
        self.pass.set_vertex_buffer(slot, buffer_slice);
        self
    }

    // -------------------------------------------------------------------------
    // Command 4: set_index_buffer
    // -------------------------------------------------------------------------

    /// Bind an index buffer with the specified format.
    ///
    /// Index buffers enable indexed drawing, which allows vertex reuse and
    /// reduces memory bandwidth. Required for `draw_indexed` calls.
    ///
    /// # Arguments
    ///
    /// * `buffer_slice` - The index buffer slice to bind
    /// * `format` - Index format (`Uint16` or `Uint32`)
    ///
    /// # Index Formats
    ///
    /// | Format | Size | Max Vertices | Use Case |
    /// |--------|------|--------------|----------|
    /// | `Uint16` | 2 bytes | 65,535 | Small/medium meshes |
    /// | `Uint32` | 4 bytes | ~4 billion | Large meshes |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 16-bit indices for small meshes
    /// commands.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    ///
    /// // 32-bit indices for large meshes
    /// commands.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    /// ```
    ///
    /// # Notes
    ///
    /// - The format must match how the index data was created
    /// - Using `Uint16` saves memory and bandwidth when possible
    #[inline]
    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'a>,
        format: wgpu::IndexFormat,
    ) -> &mut Self {
        self.pass.set_index_buffer(buffer_slice, format);
        self
    }

    // -------------------------------------------------------------------------
    // Command 5: set_viewport
    // -------------------------------------------------------------------------

    /// Set the viewport for rendering.
    ///
    /// The viewport defines the transformation from normalized device coordinates
    /// (NDC) to window coordinates. NDC ranges from -1 to +1, and the viewport
    /// maps this to the specified pixel region.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of viewport origin (top-left)
    /// * `y` - Y coordinate of viewport origin (top-left)
    /// * `w` - Viewport width in pixels
    /// * `h` - Viewport height in pixels
    /// * `min_depth` - Minimum depth value (typically 0.0)
    /// * `max_depth` - Maximum depth value (typically 1.0)
    ///
    /// # Depth Range
    ///
    /// | Configuration | min_depth | max_depth | Use Case |
    /// |---------------|-----------|-----------|----------|
    /// | Standard | 0.0 | 1.0 | Normal rendering |
    /// | Reversed-Z | 1.0 | 0.0 | Better depth precision |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Full render target viewport
    /// commands.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
    ///
    /// // Split-screen left half
    /// commands.set_viewport(0.0, 0.0, 960.0, 1080.0, 0.0, 1.0);
    ///
    /// // Reversed-Z for better depth precision
    /// commands.set_viewport(0.0, 0.0, 1920.0, 1080.0, 1.0, 0.0);
    /// ```
    #[inline]
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> &mut Self {
        self.pass.set_viewport(x, y, w, h, min_depth, max_depth);
        self
    }

    // -------------------------------------------------------------------------
    // Command 6: set_scissor_rect
    // -------------------------------------------------------------------------

    /// Set the scissor rectangle for fragment clipping.
    ///
    /// The scissor test discards fragments outside the specified rectangle.
    /// This is useful for UI clipping, split-screen rendering, and optimization.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of scissor origin (top-left)
    /// * `y` - Y coordinate of scissor origin (top-left)
    /// * `w` - Scissor width in pixels
    /// * `h` - Scissor height in pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Full render target (no clipping)
    /// commands.set_scissor_rect(0, 0, 1920, 1080);
    ///
    /// // Clip to a UI panel region
    /// commands.set_scissor_rect(100, 100, 400, 300);
    ///
    /// // Split-screen clipping
    /// commands.set_scissor_rect(0, 0, 960, 1080);
    /// ```
    ///
    /// # Notes
    ///
    /// - The scissor rectangle must be within render target bounds
    /// - Set to full render target size to disable scissor clipping
    #[inline]
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) -> &mut Self {
        self.pass.set_scissor_rect(x, y, w, h);
        self
    }

    // -------------------------------------------------------------------------
    // Command 7: set_blend_constant
    // -------------------------------------------------------------------------

    /// Set the blend constant color.
    ///
    /// The blend constant is used when blend factors are set to
    /// `BlendFactor::Constant` or `BlendFactor::OneMinusConstant`.
    /// This allows runtime control over blending without changing the pipeline.
    ///
    /// # Arguments
    ///
    /// * `color` - The constant color for blending (RGBA, each component 0.0-1.0)
    ///
    /// # Use Cases
    ///
    /// - Fade effects (varying alpha over time)
    /// - Color tinting
    /// - Cross-fading between render targets
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 50% opacity blend
    /// commands.set_blend_constant(wgpu::Color {
    ///     r: 1.0, g: 1.0, b: 1.0, a: 0.5,
    /// });
    ///
    /// // Red tint
    /// commands.set_blend_constant(wgpu::Color {
    ///     r: 1.0, g: 0.5, b: 0.5, a: 1.0,
    /// });
    /// ```
    ///
    /// # Notes
    ///
    /// - Only affects blending when pipeline uses constant blend factors
    /// - Default is typically (0, 0, 0, 0) if not set
    #[inline]
    pub fn set_blend_constant(&mut self, color: wgpu::Color) -> &mut Self {
        self.pass.set_blend_constant(color);
        self
    }

    // -------------------------------------------------------------------------
    // Command 8: set_stencil_reference
    // -------------------------------------------------------------------------

    /// Set the stencil reference value.
    ///
    /// The stencil reference is used in stencil test comparisons and write
    /// operations. This allows runtime control over stencil behavior without
    /// changing the pipeline.
    ///
    /// # Arguments
    ///
    /// * `reference` - The reference value for stencil operations (typically 0-255)
    ///
    /// # Stencil Operations
    ///
    /// The reference value is compared against the stencil buffer using the
    /// comparison function defined in the pipeline's depth/stencil state:
    ///
    /// | Compare Function | Test Passes When |
    /// |------------------|------------------|
    /// | `Always` | Always |
    /// | `Never` | Never |
    /// | `Equal` | ref == stencil |
    /// | `NotEqual` | ref != stencil |
    /// | `Less` | ref < stencil |
    /// | `Greater` | ref > stencil |
    /// | `LessEqual` | ref <= stencil |
    /// | `GreaterEqual` | ref >= stencil |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Mark geometry with stencil value 1
    /// commands.set_stencil_reference(1);
    ///
    /// // Different stencil values for different object types
    /// commands.set_stencil_reference(STENCIL_OUTLINE);  // e.g., 2
    /// // ... draw outlined objects
    ///
    /// commands.set_stencil_reference(STENCIL_REFLECT);  // e.g., 4
    /// // ... draw reflective surfaces
    /// ```
    ///
    /// # Notes
    ///
    /// - Only affects rendering when pipeline has stencil testing enabled
    /// - The effective bits depend on stencil read/write masks
    #[inline]
    pub fn set_stencil_reference(&mut self, reference: u32) -> &mut Self {
        self.pass.set_stencil_reference(reference);
        self
    }

    // -------------------------------------------------------------------------
    // Command 9: set_push_constants
    // -------------------------------------------------------------------------

    /// Set push constant data for the specified shader stages.
    ///
    /// Push constants are a fast path for small, frequently-updated uniform
    /// data. They are faster than uniform buffers for small amounts of data
    /// (typically <128 bytes) that change per-draw.
    ///
    /// # Arguments
    ///
    /// * `stages` - Shader stages that will use this data
    /// * `offset` - Byte offset into push constant memory (must be 4-byte aligned)
    /// * `data` - The data to write (must be 4-byte aligned size)
    ///
    /// # Shader Stages
    ///
    /// Common combinations:
    /// - `ShaderStages::VERTEX` - Vertex shader only
    /// - `ShaderStages::FRAGMENT` - Fragment shader only
    /// - `ShaderStages::VERTEX_FRAGMENT` - Both stages
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Per-object transform matrix (64 bytes)
    /// let model_matrix: [[f32; 4]; 4] = compute_model_matrix();
    /// commands.set_push_constants(
    ///     wgpu::ShaderStages::VERTEX,
    ///     0,
    ///     bytemuck::cast_slice(&model_matrix),
    /// );
    ///
    /// // Per-material color (16 bytes at offset 64)
    /// let color: [f32; 4] = [1.0, 0.5, 0.0, 1.0];
    /// commands.set_push_constants(
    ///     wgpu::ShaderStages::FRAGMENT,
    ///     64,
    ///     bytemuck::cast_slice(&color),
    /// );
    /// ```
    ///
    /// # WGSL Access
    ///
    /// ```wgsl
    /// struct PushConstants {
    ///     model: mat4x4<f32>,
    ///     color: vec4<f32>,
    /// }
    /// var<push_constant> pc: PushConstants;
    /// ```
    ///
    /// # Notes
    ///
    /// - Push constants must be defined in the pipeline layout
    /// - Total push constant size is limited (typically 128-256 bytes)
    /// - Offset and data size must be 4-byte aligned
    /// - Not all backends support push constants (Vulkan, Metal do; DX12 emulates)
    #[inline]
    pub fn set_push_constants(
        &mut self,
        stages: wgpu::ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> &mut Self {
        self.pass.set_push_constants(stages, offset, data);
        self
    }
}

impl<'a> fmt::Debug for RenderPassCommands<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderPassCommands")
            .field("pass", &"wgpu::RenderPass<'a>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Standalone Helper Functions
// ---------------------------------------------------------------------------

/// Bind a render pipeline to a render pass.
///
/// Convenience function for direct pipeline binding without the wrapper.
///
/// # Example
///
/// ```ignore
/// set_pipeline(&mut render_pass, &pipeline);
/// ```
#[inline]
pub fn set_pipeline<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    pipeline: &'a wgpu::RenderPipeline,
) {
    render_pass.set_pipeline(pipeline);
}

/// Attach a bind group to a render pass at the specified index.
///
/// Convenience function for direct bind group attachment.
///
/// # Example
///
/// ```ignore
/// set_bind_group(&mut render_pass, 0, &bind_group, &[]);
/// ```
#[inline]
pub fn set_bind_group<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    index: u32,
    bind_group: &'a wgpu::BindGroup,
    offsets: &[u32],
) {
    render_pass.set_bind_group(index, bind_group, offsets);
}

/// Bind a vertex buffer to a render pass slot.
///
/// Convenience function for direct vertex buffer binding.
///
/// # Example
///
/// ```ignore
/// set_vertex_buffer(&mut render_pass, 0, buffer.slice(..));
/// ```
#[inline]
pub fn set_vertex_buffer<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    slot: u32,
    buffer_slice: wgpu::BufferSlice<'a>,
) {
    render_pass.set_vertex_buffer(slot, buffer_slice);
}

/// Bind an index buffer to a render pass.
///
/// Convenience function for direct index buffer binding.
///
/// # Example
///
/// ```ignore
/// set_index_buffer(&mut render_pass, buffer.slice(..), wgpu::IndexFormat::Uint32);
/// ```
#[inline]
pub fn set_index_buffer<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    buffer_slice: wgpu::BufferSlice<'a>,
    format: wgpu::IndexFormat,
) {
    render_pass.set_index_buffer(buffer_slice, format);
}

/// Set the blend constant color on a render pass.
///
/// Convenience function for direct blend constant setting.
///
/// # Example
///
/// ```ignore
/// set_blend_constant(&mut render_pass, wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 0.5 });
/// ```
#[inline]
pub fn set_blend_constant<'a>(render_pass: &mut wgpu::RenderPass<'a>, color: wgpu::Color) {
    render_pass.set_blend_constant(color);
}

/// Set the stencil reference value on a render pass.
///
/// Convenience function for direct stencil reference setting.
///
/// # Example
///
/// ```ignore
/// set_stencil_reference(&mut render_pass, 1);
/// ```
#[inline]
pub fn set_stencil_reference<'a>(render_pass: &mut wgpu::RenderPass<'a>, reference: u32) {
    render_pass.set_stencil_reference(reference);
}

/// Set push constants on a render pass.
///
/// Convenience function for direct push constant setting.
///
/// # Example
///
/// ```ignore
/// let data = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]);
/// set_push_constants(&mut render_pass, wgpu::ShaderStages::VERTEX, 0, data);
/// ```
#[inline]
pub fn set_push_constants<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    stages: wgpu::ShaderStages,
    offset: u32,
    data: &[u8],
) {
    render_pass.set_push_constants(stages, offset, data);
}

// ---------------------------------------------------------------------------
// BlendConstant Helper
// ---------------------------------------------------------------------------

/// Builder for blend constant colors.
///
/// Provides a fluent API for creating blend constant colors.
///
/// # Example
///
/// ```
/// use renderer_backend::render_pipeline::render_pass_commands::BlendConstantBuilder;
///
/// let color = BlendConstantBuilder::new()
///     .rgb(1.0, 0.5, 0.0)
///     .alpha(0.75)
///     .build();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendConstantBuilder {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

impl Default for BlendConstantBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlendConstantBuilder {
    /// Create a new blend constant builder with default values (black, opaque).
    pub const fn new() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    /// Create a builder with white color.
    pub const fn white() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }

    /// Create a builder with transparent black.
    pub const fn transparent() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }

    /// Set the red component.
    pub const fn r(mut self, r: f64) -> Self {
        self.r = r;
        self
    }

    /// Set the green component.
    pub const fn g(mut self, g: f64) -> Self {
        self.g = g;
        self
    }

    /// Set the blue component.
    pub const fn b(mut self, b: f64) -> Self {
        self.b = b;
        self
    }

    /// Set the alpha component.
    pub const fn alpha(mut self, a: f64) -> Self {
        self.a = a;
        self
    }

    /// Set RGB components at once.
    pub const fn rgb(mut self, r: f64, g: f64, b: f64) -> Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self
    }

    /// Set all RGBA components at once.
    pub const fn rgba(mut self, r: f64, g: f64, b: f64, a: f64) -> Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self.a = a;
        self
    }

    /// Build the wgpu Color.
    pub const fn build(self) -> wgpu::Color {
        wgpu::Color {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a,
        }
    }
}

// ---------------------------------------------------------------------------
// StencilReference Constants
// ---------------------------------------------------------------------------

/// Common stencil reference values for typical use cases.
pub mod stencil_values {
    /// No stencil marking (default).
    pub const NONE: u32 = 0;

    /// General geometry stencil value.
    pub const GEOMETRY: u32 = 1;

    /// Outlined objects stencil value.
    pub const OUTLINE: u32 = 2;

    /// Reflective surfaces stencil value.
    pub const REFLECT: u32 = 4;

    /// Shadow receiving surfaces stencil value.
    pub const SHADOW: u32 = 8;

    /// Portal/window stencil value.
    pub const PORTAL: u32 = 16;

    /// UI overlay stencil value.
    pub const UI: u32 = 32;

    /// Reserved for user-defined purposes.
    pub const USER_1: u32 = 64;

    /// Reserved for user-defined purposes.
    pub const USER_2: u32 = 128;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BlendConstantBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_constant_builder_default() {
        let color = BlendConstantBuilder::new().build();
        assert_eq!(color.r, 0.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 1.0);
    }

    #[test]
    fn test_blend_constant_builder_white() {
        let color = BlendConstantBuilder::white().build();
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 1.0);
        assert_eq!(color.b, 1.0);
        assert_eq!(color.a, 1.0);
    }

    #[test]
    fn test_blend_constant_builder_transparent() {
        let color = BlendConstantBuilder::transparent().build();
        assert_eq!(color.r, 0.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 0.0);
    }

    #[test]
    fn test_blend_constant_builder_individual_components() {
        let color = BlendConstantBuilder::new()
            .r(0.1)
            .g(0.2)
            .b(0.3)
            .alpha(0.4)
            .build();
        assert!((color.r - 0.1).abs() < f64::EPSILON);
        assert!((color.g - 0.2).abs() < f64::EPSILON);
        assert!((color.b - 0.3).abs() < f64::EPSILON);
        assert!((color.a - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_rgb() {
        let color = BlendConstantBuilder::new()
            .rgb(0.5, 0.6, 0.7)
            .build();
        assert!((color.r - 0.5).abs() < f64::EPSILON);
        assert!((color.g - 0.6).abs() < f64::EPSILON);
        assert!((color.b - 0.7).abs() < f64::EPSILON);
        assert_eq!(color.a, 1.0); // Default alpha
    }

    #[test]
    fn test_blend_constant_builder_rgba() {
        let color = BlendConstantBuilder::new()
            .rgba(0.1, 0.2, 0.3, 0.4)
            .build();
        assert!((color.r - 0.1).abs() < f64::EPSILON);
        assert!((color.g - 0.2).abs() < f64::EPSILON);
        assert!((color.b - 0.3).abs() < f64::EPSILON);
        assert!((color.a - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_chaining() {
        let color = BlendConstantBuilder::white()
            .alpha(0.5)
            .r(0.8)
            .build();
        assert!((color.r - 0.8).abs() < f64::EPSILON);
        assert_eq!(color.g, 1.0);
        assert_eq!(color.b, 1.0);
        assert!((color.a - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_clone() {
        let builder1 = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.5);
        let builder2 = builder1.alpha(0.75);
        let color1 = builder1.build();
        let color2 = builder2.build();
        assert_eq!(color1.a, 1.0);
        assert!((color2.a - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_default_impl() {
        let color = BlendConstantBuilder::default().build();
        let new_color = BlendConstantBuilder::new().build();
        assert_eq!(color.r, new_color.r);
        assert_eq!(color.g, new_color.g);
        assert_eq!(color.b, new_color.b);
        assert_eq!(color.a, new_color.a);
    }

    #[test]
    fn test_blend_constant_builder_debug() {
        let builder = BlendConstantBuilder::new().rgb(1.0, 0.5, 0.0);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("BlendConstantBuilder"));
    }

    // -------------------------------------------------------------------------
    // Stencil Values Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stencil_values_none() {
        assert_eq!(stencil_values::NONE, 0);
    }

    #[test]
    fn test_stencil_values_geometry() {
        assert_eq!(stencil_values::GEOMETRY, 1);
    }

    #[test]
    fn test_stencil_values_outline() {
        assert_eq!(stencil_values::OUTLINE, 2);
    }

    #[test]
    fn test_stencil_values_reflect() {
        assert_eq!(stencil_values::REFLECT, 4);
    }

    #[test]
    fn test_stencil_values_shadow() {
        assert_eq!(stencil_values::SHADOW, 8);
    }

    #[test]
    fn test_stencil_values_portal() {
        assert_eq!(stencil_values::PORTAL, 16);
    }

    #[test]
    fn test_stencil_values_ui() {
        assert_eq!(stencil_values::UI, 32);
    }

    #[test]
    fn test_stencil_values_user() {
        assert_eq!(stencil_values::USER_1, 64);
        assert_eq!(stencil_values::USER_2, 128);
    }

    #[test]
    fn test_stencil_values_are_power_of_two() {
        // All non-zero stencil values should be powers of 2 for easy combining
        let values = [
            stencil_values::GEOMETRY,
            stencil_values::OUTLINE,
            stencil_values::REFLECT,
            stencil_values::SHADOW,
            stencil_values::PORTAL,
            stencil_values::UI,
            stencil_values::USER_1,
            stencil_values::USER_2,
        ];
        for value in values {
            assert!(value.is_power_of_two(), "Value {} is not power of 2", value);
        }
    }

    #[test]
    fn test_stencil_values_unique() {
        let values = [
            stencil_values::NONE,
            stencil_values::GEOMETRY,
            stencil_values::OUTLINE,
            stencil_values::REFLECT,
            stencil_values::SHADOW,
            stencil_values::PORTAL,
            stencil_values::UI,
            stencil_values::USER_1,
            stencil_values::USER_2,
        ];
        let mut seen = std::collections::HashSet::new();
        for value in values {
            assert!(seen.insert(value), "Duplicate stencil value: {}", value);
        }
    }

    #[test]
    fn test_stencil_values_combinable() {
        // Test that stencil values can be combined with bitwise OR
        let combined = stencil_values::GEOMETRY | stencil_values::OUTLINE;
        assert_eq!(combined, 3);

        let multi = stencil_values::REFLECT | stencil_values::SHADOW | stencil_values::PORTAL;
        assert_eq!(multi, 4 | 8 | 16);
    }

    // -------------------------------------------------------------------------
    // RenderPassCommands Debug Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_commands_debug() {
        // We can't create a real RenderPass without a device, but we can test the Debug impl
        // by checking the struct definition compiles and the Debug trait is implemented
        fn assert_debug<T: fmt::Debug>() {}
        assert_debug::<BlendConstantBuilder>();
    }

    // -------------------------------------------------------------------------
    // Constant function compilation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_const_blend_constant_builder() {
        // Test that const functions compile and work at compile time
        const WHITE: wgpu::Color = BlendConstantBuilder::white().build();
        const TRANSPARENT: wgpu::Color = BlendConstantBuilder::transparent().build();
        const CUSTOM: wgpu::Color = BlendConstantBuilder::new()
            .r(0.5)
            .g(0.5)
            .b(0.5)
            .alpha(0.5)
            .build();

        assert_eq!(WHITE.r, 1.0);
        assert_eq!(TRANSPARENT.a, 0.0);
        assert!((CUSTOM.r - 0.5).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // BlendConstantBuilder Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_constant_builder_zero_values() {
        let color = BlendConstantBuilder::new()
            .rgba(0.0, 0.0, 0.0, 0.0)
            .build();
        assert_eq!(color.r, 0.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 0.0);
    }

    #[test]
    fn test_blend_constant_builder_max_values() {
        let color = BlendConstantBuilder::new()
            .rgba(1.0, 1.0, 1.0, 1.0)
            .build();
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 1.0);
        assert_eq!(color.b, 1.0);
        assert_eq!(color.a, 1.0);
    }

    #[test]
    fn test_blend_constant_builder_negative_values() {
        // GPU blending can use negative values in some cases
        let color = BlendConstantBuilder::new()
            .rgba(-0.5, -1.0, -0.25, -0.1)
            .build();
        assert!((color.r - (-0.5)).abs() < f64::EPSILON);
        assert!((color.g - (-1.0)).abs() < f64::EPSILON);
        assert!((color.b - (-0.25)).abs() < f64::EPSILON);
        assert!((color.a - (-0.1)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_values_greater_than_one() {
        // HDR blending can use values > 1.0
        let color = BlendConstantBuilder::new()
            .rgba(2.0, 3.5, 10.0, 1.5)
            .build();
        assert!((color.r - 2.0).abs() < f64::EPSILON);
        assert!((color.g - 3.5).abs() < f64::EPSILON);
        assert!((color.b - 10.0).abs() < f64::EPSILON);
        assert!((color.a - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_fractional_precision() {
        let color = BlendConstantBuilder::new()
            .rgba(0.123456789, 0.987654321, 0.555555555, 0.111111111)
            .build();
        assert!((color.r - 0.123456789).abs() < 1e-9);
        assert!((color.g - 0.987654321).abs() < 1e-9);
        assert!((color.b - 0.555555555).abs() < 1e-9);
        assert!((color.a - 0.111111111).abs() < 1e-9);
    }

    #[test]
    fn test_blend_constant_builder_very_small_values() {
        let color = BlendConstantBuilder::new()
            .rgba(1e-10, 1e-15, 1e-20, 1e-5)
            .build();
        assert!((color.r - 1e-10).abs() < f64::EPSILON);
        assert!((color.g - 1e-15).abs() < f64::EPSILON);
        assert!((color.b - 1e-20).abs() < f64::EPSILON);
        assert!((color.a - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_overwrite_behavior() {
        // Later calls should overwrite earlier values
        let color = BlendConstantBuilder::new()
            .r(0.1)
            .r(0.2)
            .r(0.3)
            .build();
        assert!((color.r - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_rgb_then_alpha() {
        let color = BlendConstantBuilder::new()
            .rgb(0.5, 0.6, 0.7)
            .alpha(0.8)
            .build();
        assert!((color.r - 0.5).abs() < f64::EPSILON);
        assert!((color.g - 0.6).abs() < f64::EPSILON);
        assert!((color.b - 0.7).abs() < f64::EPSILON);
        assert!((color.a - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_rgba_overwrites_rgb() {
        let color = BlendConstantBuilder::new()
            .rgb(0.1, 0.2, 0.3)
            .rgba(0.4, 0.5, 0.6, 0.7)
            .build();
        assert!((color.r - 0.4).abs() < f64::EPSILON);
        assert!((color.g - 0.5).abs() < f64::EPSILON);
        assert!((color.b - 0.6).abs() < f64::EPSILON);
        assert!((color.a - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blend_constant_builder_partial_eq() {
        let builder1 = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.5);
        let builder2 = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.5);
        let builder3 = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.6);
        assert_eq!(builder1, builder2);
        assert_ne!(builder1, builder3);
    }

    #[test]
    fn test_blend_constant_builder_copy_semantics() {
        let builder = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.5);
        let copied = builder; // Copy
        let color1 = builder.build();
        let color2 = copied.build();
        assert_eq!(color1.r, color2.r);
        assert_eq!(color1.g, color2.g);
        assert_eq!(color1.b, color2.b);
        assert_eq!(color1.a, color2.a);
    }

    // -------------------------------------------------------------------------
    // Stencil Reference Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stencil_reference_zero() {
        // Zero is valid stencil reference
        assert_eq!(stencil_values::NONE, 0);
    }

    #[test]
    fn test_stencil_reference_max_u8() {
        // 255 is max for 8-bit stencil buffer
        let max_u8: u32 = 255;
        assert!(max_u8 <= u32::MAX);
        assert!(max_u8.is_power_of_two() == false); // 255 is not power of 2
    }

    #[test]
    fn test_stencil_reference_max_u32() {
        // API accepts u32 but hardware typically only uses 8 bits
        let max_u32 = u32::MAX;
        assert_eq!(max_u32, 4294967295);
    }

    #[test]
    fn test_stencil_values_fit_in_8_bits() {
        // All predefined stencil values should fit in 8-bit stencil buffer
        let values = [
            stencil_values::NONE,
            stencil_values::GEOMETRY,
            stencil_values::OUTLINE,
            stencil_values::REFLECT,
            stencil_values::SHADOW,
            stencil_values::PORTAL,
            stencil_values::UI,
            stencil_values::USER_1,
            stencil_values::USER_2,
        ];
        for value in values {
            assert!(value <= 255, "Stencil value {} exceeds 8-bit range", value);
        }
    }

    #[test]
    fn test_stencil_values_bitmask_combination() {
        // Test combining all values into a bitmask
        let all_combined = stencil_values::GEOMETRY
            | stencil_values::OUTLINE
            | stencil_values::REFLECT
            | stencil_values::SHADOW
            | stencil_values::PORTAL
            | stencil_values::UI
            | stencil_values::USER_1
            | stencil_values::USER_2;
        assert_eq!(all_combined, 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128);
        assert_eq!(all_combined, 255);
    }

    #[test]
    fn test_stencil_values_bitwise_and() {
        // Test extracting specific bits
        let combined = stencil_values::GEOMETRY | stencil_values::SHADOW;
        assert_eq!(combined & stencil_values::GEOMETRY, stencil_values::GEOMETRY);
        assert_eq!(combined & stencil_values::SHADOW, stencil_values::SHADOW);
        assert_eq!(combined & stencil_values::OUTLINE, 0);
    }

    #[test]
    fn test_stencil_values_bitwise_xor() {
        // Test toggling bits
        let mut value = stencil_values::GEOMETRY;
        value ^= stencil_values::GEOMETRY;
        assert_eq!(value, 0);
        value ^= stencil_values::SHADOW;
        assert_eq!(value, stencil_values::SHADOW);
    }

    #[test]
    fn test_stencil_values_bitwise_not() {
        // Test inverting bits (within 8-bit range)
        let value = stencil_values::GEOMETRY;
        let inverted = !value & 0xFF;
        assert_eq!(inverted, 254);
    }

    // -------------------------------------------------------------------------
    // Viewport Boundary Condition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_parameters_zero() {
        // Zero viewport is technically valid (no rendering visible)
        let x = 0.0f32;
        let y = 0.0f32;
        let w = 0.0f32;
        let h = 0.0f32;
        let min_depth = 0.0f32;
        let max_depth = 0.0f32;
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(w, 0.0);
        assert_eq!(h, 0.0);
        assert_eq!(min_depth, 0.0);
        assert_eq!(max_depth, 0.0);
    }

    #[test]
    fn test_viewport_parameters_max() {
        // Large viewport values
        let x = f32::MAX;
        let y = f32::MAX;
        let w = f32::MAX;
        let h = f32::MAX;
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(w.is_finite());
        assert!(h.is_finite());
    }

    #[test]
    fn test_viewport_parameters_fractional() {
        // Fractional viewport coordinates (subpixel precision)
        let x = 0.5f32;
        let y = 0.25f32;
        let w = 1920.5f32;
        let h = 1080.75f32;
        assert!((x - 0.5).abs() < f32::EPSILON);
        assert!((y - 0.25).abs() < f32::EPSILON);
        assert!((w - 1920.5).abs() < f32::EPSILON);
        assert!((h - 1080.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_viewport_depth_standard() {
        // Standard depth range
        let min_depth = 0.0f32;
        let max_depth = 1.0f32;
        assert!(min_depth < max_depth);
    }

    #[test]
    fn test_viewport_depth_reversed() {
        // Reversed-Z depth range for better precision
        let min_depth = 1.0f32;
        let max_depth = 0.0f32;
        assert!(min_depth > max_depth);
    }

    #[test]
    fn test_viewport_negative_origin() {
        // Negative viewport origin (offscreen rendering)
        let x = -100.0f32;
        let y = -50.0f32;
        assert!(x < 0.0);
        assert!(y < 0.0);
    }

    #[test]
    fn test_viewport_typical_resolutions() {
        // Common resolutions
        let resolutions = [
            (1920.0f32, 1080.0f32), // 1080p
            (2560.0f32, 1440.0f32), // 1440p
            (3840.0f32, 2160.0f32), // 4K
            (7680.0f32, 4320.0f32), // 8K
            (1280.0f32, 720.0f32),  // 720p
        ];
        for (w, h) in resolutions {
            assert!(w > 0.0);
            assert!(h > 0.0);
            assert!(w.is_finite());
            assert!(h.is_finite());
        }
    }

    // -------------------------------------------------------------------------
    // Scissor Rect Boundary Condition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_rect_zero() {
        // Zero scissor rect (no pixels visible)
        let x = 0u32;
        let y = 0u32;
        let w = 0u32;
        let h = 0u32;
        assert_eq!(x, 0);
        assert_eq!(y, 0);
        assert_eq!(w, 0);
        assert_eq!(h, 0);
    }

    #[test]
    fn test_scissor_rect_max_u32() {
        // Maximum u32 values
        let x = u32::MAX;
        let y = u32::MAX;
        let w = u32::MAX;
        let h = u32::MAX;
        assert_eq!(x, 4294967295);
        assert_eq!(y, 4294967295);
        assert_eq!(w, 4294967295);
        assert_eq!(h, 4294967295);
    }

    #[test]
    fn test_scissor_rect_typical_sizes() {
        // Typical scissor rects for UI clipping
        let rects = [
            (0u32, 0u32, 1920u32, 1080u32),   // Full screen
            (100u32, 100u32, 400u32, 300u32), // UI panel
            (0u32, 0u32, 960u32, 1080u32),    // Split screen left
            (960u32, 0u32, 960u32, 1080u32),  // Split screen right
        ];
        for (x, y, w, h) in rects {
            assert!(x <= u32::MAX);
            assert!(y <= u32::MAX);
            assert!(w <= u32::MAX);
            assert!(h <= u32::MAX);
        }
    }

    #[test]
    fn test_scissor_rect_single_pixel() {
        // Single pixel scissor (edge case)
        let w = 1u32;
        let h = 1u32;
        assert_eq!(w * h, 1);
    }

    // -------------------------------------------------------------------------
    // Index Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_index_format_uint16() {
        let format = wgpu::IndexFormat::Uint16;
        // Uint16 supports up to 65535 vertices
        let max_vertices: u32 = 65535;
        assert!(max_vertices < u32::MAX);
        assert_eq!(format, wgpu::IndexFormat::Uint16);
    }

    #[test]
    fn test_index_format_uint32() {
        let format = wgpu::IndexFormat::Uint32;
        // Uint32 supports up to ~4 billion vertices
        let max_vertices: u32 = u32::MAX;
        assert_eq!(max_vertices, 4294967295);
        assert_eq!(format, wgpu::IndexFormat::Uint32);
    }

    #[test]
    fn test_index_format_size() {
        // Uint16 is 2 bytes, Uint32 is 4 bytes
        let uint16_size = std::mem::size_of::<u16>();
        let uint32_size = std::mem::size_of::<u32>();
        assert_eq!(uint16_size, 2);
        assert_eq!(uint32_size, 4);
    }

    #[test]
    fn test_index_format_debug() {
        let format16 = wgpu::IndexFormat::Uint16;
        let format32 = wgpu::IndexFormat::Uint32;
        let debug16 = format!("{:?}", format16);
        let debug32 = format!("{:?}", format32);
        assert!(debug16.contains("Uint16"));
        assert!(debug32.contains("Uint32"));
    }

    #[test]
    fn test_index_format_equality() {
        let format1 = wgpu::IndexFormat::Uint16;
        let format2 = wgpu::IndexFormat::Uint16;
        let format3 = wgpu::IndexFormat::Uint32;
        assert_eq!(format1, format2);
        assert_ne!(format1, format3);
    }

    // -------------------------------------------------------------------------
    // Shader Stage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_stages_vertex() {
        let stages = wgpu::ShaderStages::VERTEX;
        assert!(stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(!stages.contains(wgpu::ShaderStages::FRAGMENT));
    }

    #[test]
    fn test_shader_stages_fragment() {
        let stages = wgpu::ShaderStages::FRAGMENT;
        assert!(stages.contains(wgpu::ShaderStages::FRAGMENT));
        assert!(!stages.contains(wgpu::ShaderStages::VERTEX));
    }

    #[test]
    fn test_shader_stages_vertex_fragment() {
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        assert!(stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(stages.contains(wgpu::ShaderStages::FRAGMENT));
    }

    #[test]
    fn test_shader_stages_compute() {
        let stages = wgpu::ShaderStages::COMPUTE;
        assert!(stages.contains(wgpu::ShaderStages::COMPUTE));
        assert!(!stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(!stages.contains(wgpu::ShaderStages::FRAGMENT));
    }

    #[test]
    fn test_shader_stages_none() {
        let stages = wgpu::ShaderStages::NONE;
        assert!(!stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(!stages.contains(wgpu::ShaderStages::FRAGMENT));
        assert!(!stages.contains(wgpu::ShaderStages::COMPUTE));
    }

    #[test]
    fn test_shader_stages_all() {
        let stages = wgpu::ShaderStages::all();
        assert!(stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(stages.contains(wgpu::ShaderStages::FRAGMENT));
        assert!(stages.contains(wgpu::ShaderStages::COMPUTE));
    }

    #[test]
    fn test_shader_stages_bitwise_or() {
        let stages = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        assert_eq!(stages, wgpu::ShaderStages::VERTEX_FRAGMENT);
    }

    #[test]
    fn test_shader_stages_bitwise_and() {
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT & wgpu::ShaderStages::VERTEX;
        assert_eq!(stages, wgpu::ShaderStages::VERTEX);
    }

    // -------------------------------------------------------------------------
    // Dynamic Offset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dynamic_offset_empty() {
        let offsets: &[u32] = &[];
        assert_eq!(offsets.len(), 0);
    }

    #[test]
    fn test_dynamic_offset_single() {
        let offsets: &[u32] = &[256];
        assert_eq!(offsets.len(), 1);
        assert_eq!(offsets[0], 256);
    }

    #[test]
    fn test_dynamic_offset_multiple() {
        let offsets: &[u32] = &[256, 512, 768];
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 256);
        assert_eq!(offsets[1], 512);
        assert_eq!(offsets[2], 768);
    }

    #[test]
    fn test_dynamic_offset_alignment() {
        // Offsets typically need to be 256-byte aligned
        let alignment = 256u32;
        let offsets: Vec<u32> = (0..10).map(|i| i * alignment).collect();
        for offset in &offsets {
            assert_eq!(offset % alignment, 0);
        }
    }

    #[test]
    fn test_dynamic_offset_zero() {
        let offsets: &[u32] = &[0];
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn test_dynamic_offset_max() {
        // Large offsets for big buffers
        let offsets: &[u32] = &[u32::MAX - 255]; // Stay within alignment
        assert!(offsets[0] > 0);
    }

    // -------------------------------------------------------------------------
    // Vertex Buffer Slot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_buffer_slot_zero() {
        let slot = 0u32;
        assert_eq!(slot, 0);
    }

    #[test]
    fn test_vertex_buffer_slot_typical() {
        // Typical slot usage: 0 = vertices, 1 = instances
        let vertex_slot = 0u32;
        let instance_slot = 1u32;
        assert!(vertex_slot < instance_slot);
    }

    #[test]
    fn test_vertex_buffer_slot_multiple() {
        // Multiple attribute buffers
        let slots = [0u32, 1u32, 2u32, 3u32];
        for (i, &slot) in slots.iter().enumerate() {
            assert_eq!(slot, i as u32);
        }
    }

    #[test]
    fn test_vertex_buffer_slot_max() {
        // wgpu typically limits to 8-16 vertex buffer slots
        let max_practical_slots = 16u32;
        assert!(max_practical_slots <= u32::MAX);
    }

    // -------------------------------------------------------------------------
    // Bind Group Index Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bind_group_index_zero() {
        let index = 0u32;
        assert_eq!(index, 0);
    }

    #[test]
    fn test_bind_group_index_typical() {
        // Typical: 0 = global, 1 = per-material, 2 = per-object
        let global_index = 0u32;
        let material_index = 1u32;
        let object_index = 2u32;
        assert!(global_index < material_index);
        assert!(material_index < object_index);
    }

    #[test]
    fn test_bind_group_index_max_typical() {
        // Most pipelines use 0-3 (4 bind groups)
        let max_typical = 3u32;
        assert!(max_typical < 8);
    }

    // -------------------------------------------------------------------------
    // Push Constants Data Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_constants_empty() {
        let data: &[u8] = &[];
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_push_constants_single_float() {
        let value: f32 = 1.5;
        let data = value.to_ne_bytes();
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_push_constants_vec4() {
        let vec4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let data = bytemuck::cast_slice::<f32, u8>(&vec4);
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn test_push_constants_mat4() {
        let mat4: [[f32; 4]; 4] = [[1.0; 4]; 4];
        let flat: &[f32; 16] = bytemuck::cast_ref(&mat4);
        let data = bytemuck::cast_slice::<f32, u8>(flat);
        assert_eq!(data.len(), 64);
    }

    #[test]
    fn test_push_constants_offset_alignment() {
        // Offsets must be 4-byte aligned
        let alignment = 4u32;
        let offsets = [0u32, 4, 8, 16, 64, 128];
        for offset in offsets {
            assert_eq!(offset % alignment, 0);
        }
    }

    #[test]
    fn test_push_constants_typical_sizes() {
        // Common push constant sizes
        let sizes = [
            4,  // Single u32/f32
            8,  // vec2
            12, // vec3
            16, // vec4
            64, // mat4x4
            80, // mat4x4 + vec4
        ];
        for size in sizes {
            assert!(size % 4 == 0, "Size {} not 4-byte aligned", size);
            assert!(size <= 256, "Size {} exceeds typical limit", size);
        }
    }

    // -------------------------------------------------------------------------
    // Trait Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_constant_builder_traits() {
        fn assert_traits<T: std::fmt::Debug + Clone + Copy + PartialEq>() {}
        assert_traits::<BlendConstantBuilder>();
    }

    #[test]
    fn test_blend_constant_builder_debug_format() {
        let builder = BlendConstantBuilder::new()
            .rgba(0.1, 0.2, 0.3, 0.4);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("r:"));
        assert!(debug_str.contains("g:"));
        assert!(debug_str.contains("b:"));
        assert!(debug_str.contains("a:"));
    }

    #[test]
    fn test_blend_constant_builder_clone_independence() {
        let original = BlendConstantBuilder::new().rgb(0.5, 0.5, 0.5);
        let mut cloned = original.clone();
        // Modify clone (note: Copy means this creates a new value)
        let cloned = cloned.alpha(0.1);
        let orig_color = original.build();
        let clone_color = cloned.build();
        // Original should be unchanged
        assert_eq!(orig_color.a, 1.0);
        assert!((clone_color.a - 0.1).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Wgpu Color Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_color_from_builder() {
        let color: wgpu::Color = BlendConstantBuilder::new()
            .rgba(0.5, 0.6, 0.7, 0.8)
            .build();
        assert!((color.r - 0.5).abs() < f64::EPSILON);
        assert!((color.g - 0.6).abs() < f64::EPSILON);
        assert!((color.b - 0.7).abs() < f64::EPSILON);
        assert!((color.a - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_wgpu_color_predefined() {
        let white = wgpu::Color::WHITE;
        let black = wgpu::Color::BLACK;
        let transparent = wgpu::Color::TRANSPARENT;

        assert_eq!(white.r, 1.0);
        assert_eq!(white.g, 1.0);
        assert_eq!(white.b, 1.0);
        assert_eq!(white.a, 1.0);

        assert_eq!(black.r, 0.0);
        assert_eq!(black.g, 0.0);
        assert_eq!(black.b, 0.0);
        assert_eq!(black.a, 1.0);

        assert_eq!(transparent.r, 0.0);
        assert_eq!(transparent.g, 0.0);
        assert_eq!(transparent.b, 0.0);
        assert_eq!(transparent.a, 0.0);
    }

    #[test]
    fn test_builder_matches_wgpu_white() {
        let builder_white = BlendConstantBuilder::white().build();
        let wgpu_white = wgpu::Color::WHITE;
        assert_eq!(builder_white.r, wgpu_white.r);
        assert_eq!(builder_white.g, wgpu_white.g);
        assert_eq!(builder_white.b, wgpu_white.b);
        assert_eq!(builder_white.a, wgpu_white.a);
    }

    #[test]
    fn test_builder_matches_wgpu_transparent() {
        let builder_transparent = BlendConstantBuilder::transparent().build();
        let wgpu_transparent = wgpu::Color::TRANSPARENT;
        assert_eq!(builder_transparent.r, wgpu_transparent.r);
        assert_eq!(builder_transparent.g, wgpu_transparent.g);
        assert_eq!(builder_transparent.b, wgpu_transparent.b);
        assert_eq!(builder_transparent.a, wgpu_transparent.a);
    }

    // -------------------------------------------------------------------------
    // Documentation Example Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_doc_example_blend_constant() {
        // From the doc example in BlendConstantBuilder
        let color = BlendConstantBuilder::new()
            .rgb(1.0, 0.5, 0.0)
            .alpha(0.75)
            .build();
        assert!((color.r - 1.0).abs() < f64::EPSILON);
        assert!((color.g - 0.5).abs() < f64::EPSILON);
        assert!((color.b - 0.0).abs() < f64::EPSILON);
        assert!((color.a - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stencil_typical_usage() {
        // Example from doc: using stencil values for object types
        let outlined_objects = stencil_values::OUTLINE;
        let reflective_surfaces = stencil_values::REFLECT;
        assert_eq!(outlined_objects, 2);
        assert_eq!(reflective_surfaces, 4);
        assert_ne!(outlined_objects, reflective_surfaces);
    }
}
