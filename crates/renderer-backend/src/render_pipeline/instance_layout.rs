//! Instance vertex layout support for wgpu 25.x
//!
//! Provides helpers for per-instance vertex data including transform matrices,
//! colors, and custom instance attributes for efficient GPU instancing.
//!
//! # Overview
//!
//! Instance data is used in GPU instancing to render many copies of the same
//! mesh with different transforms, colors, or other per-instance properties.
//! This module provides:
//!
//! - Pre-defined instance layouts for common use cases
//! - A builder pattern for custom instance layouts
//! - Shader location constants for consistent binding
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::render_pipeline::instance_layout::{
//!     InstanceLayoutBuilder, presets, INSTANCE_TRANSFORM_LOCATIONS,
//! };
//!
//! // Use a preset layout
//! let transform_layout = presets::transform_only();
//!
//! // Or build a custom layout
//! let custom_layout = InstanceLayoutBuilder::new(INSTANCE_TRANSFORM_LOCATIONS.start)
//!     .with_transform()
//!     .with_color_float()
//!     .build();
//! ```
//!
//! # Shader Integration
//!
//! The standard transform matrix uses shader locations 4-7:
//!
//! ```wgsl
//! struct InstanceInput {
//!     @location(4) model_matrix_0: vec4<f32>,
//!     @location(5) model_matrix_1: vec4<f32>,
//!     @location(6) model_matrix_2: vec4<f32>,
//!     @location(7) model_matrix_3: vec4<f32>,
//! }
//!
//! // Reconstruct the matrix in the vertex shader:
//! let model_matrix = mat4x4<f32>(
//!     instance.model_matrix_0,
//!     instance.model_matrix_1,
//!     instance.model_matrix_2,
//!     instance.model_matrix_3,
//! );
//! ```

use std::ops::Range;

use crate::render_pipeline::{VertexAttributeDescriptor, VertexBufferLayoutDescriptor};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard shader locations reserved for instance transform matrix (4-7).
///
/// Using locations 4-7 leaves 0-3 available for per-vertex attributes
/// (position, normal, tangent, uv) which is the common convention.
pub const INSTANCE_TRANSFORM_LOCATIONS: Range<u32> = 4..8;

/// Shader location for instance color (after transform matrix).
pub const INSTANCE_COLOR_LOCATION: u32 = 8;

/// Shader location for instance ID (typically after color).
pub const INSTANCE_ID_LOCATION: u32 = 9;

/// Shader location for first custom instance attribute.
pub const INSTANCE_CUSTOM_START_LOCATION: u32 = 10;

// ---------------------------------------------------------------------------
// Stride Constants
// ---------------------------------------------------------------------------

/// Stride for transform-only instance data (4x vec4 = 64 bytes).
pub const STRIDE_TRANSFORM: u64 = 64;

/// Stride for transform + Unorm8x4 color (64 + 4 = 68 bytes).
pub const STRIDE_TRANSFORM_COLOR_PACKED: u64 = 68;

/// Stride for transform + Float32x4 color (64 + 16 = 80 bytes).
pub const STRIDE_TRANSFORM_COLOR_FLOAT: u64 = 80;

/// Stride for transform + Float32x4 color + custom vec4 (64 + 16 + 16 = 96 bytes).
pub const STRIDE_TRANSFORM_COLOR_CUSTOM: u64 = 96;

// ---------------------------------------------------------------------------
// InstanceLayoutBuilder
// ---------------------------------------------------------------------------

/// Builder for custom instance vertex layouts.
///
/// Provides a fluent API for constructing per-instance vertex buffer layouts
/// with automatic offset and shader location management.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::render_pipeline::instance_layout::InstanceLayoutBuilder;
///
/// // Build a layout with transform, color, and a custom vec4
/// let layout = InstanceLayoutBuilder::new(4)
///     .with_transform()
///     .with_color_float()
///     .with_attribute(wgpu::VertexFormat::Float32x4)
///     .build();
///
/// assert_eq!(layout.array_stride, 96); // 64 + 16 + 16
/// ```
#[derive(Debug, Clone)]
pub struct InstanceLayoutBuilder {
    attributes: Vec<VertexAttributeDescriptor>,
    stride: u64,
    next_location: u32,
}

impl InstanceLayoutBuilder {
    /// Create a new builder starting at the given shader location.
    ///
    /// For standard layouts, use `INSTANCE_TRANSFORM_LOCATIONS.start` (4).
    pub fn new(start_location: u32) -> Self {
        Self {
            attributes: Vec::new(),
            stride: 0,
            next_location: start_location,
        }
    }

    /// Add a 4x4 transform matrix (4x Float32x4 = 64 bytes, uses 4 locations).
    ///
    /// The matrix is stored row-major as 4 consecutive vec4 attributes.
    pub fn with_transform(mut self) -> Self {
        for i in 0..4 {
            self.attributes.push(VertexAttributeDescriptor::new(
                wgpu::VertexFormat::Float32x4,
                self.stride + (i * 16) as u64,
                self.next_location + i,
            ));
        }
        self.stride += 64;
        self.next_location += 4;
        self
    }

    /// Add a 3x4 transform matrix (3x Float32x4 = 48 bytes, uses 3 locations).
    ///
    /// More memory-efficient for affine transforms (no perspective).
    /// The 4th row is implicitly [0, 0, 0, 1].
    pub fn with_transform_3x4(mut self) -> Self {
        for i in 0..3 {
            self.attributes.push(VertexAttributeDescriptor::new(
                wgpu::VertexFormat::Float32x4,
                self.stride + (i * 16) as u64,
                self.next_location + i,
            ));
        }
        self.stride += 48;
        self.next_location += 3;
        self
    }

    /// Add an HDR instance color (Float32x4 = 16 bytes).
    ///
    /// Full precision color suitable for HDR rendering.
    pub fn with_color_float(mut self) -> Self {
        self.attributes.push(VertexAttributeDescriptor::new(
            wgpu::VertexFormat::Float32x4,
            self.stride,
            self.next_location,
        ));
        self.stride += 16;
        self.next_location += 1;
        self
    }

    /// Add an LDR instance color (Unorm8x4 = 4 bytes).
    ///
    /// Packed color suitable for LDR rendering. More memory efficient.
    pub fn with_color_packed(mut self) -> Self {
        self.attributes.push(VertexAttributeDescriptor::new(
            wgpu::VertexFormat::Unorm8x4,
            self.stride,
            self.next_location,
        ));
        self.stride += 4;
        self.next_location += 1;
        self
    }

    /// Add an instance ID (Uint32 = 4 bytes).
    ///
    /// Useful for indirect rendering or lookup into storage buffers.
    pub fn with_instance_id(mut self) -> Self {
        self.attributes.push(VertexAttributeDescriptor::new(
            wgpu::VertexFormat::Uint32,
            self.stride,
            self.next_location,
        ));
        self.stride += 4;
        self.next_location += 1;
        self
    }

    /// Add a custom attribute with the specified format.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::render_pipeline::instance_layout::InstanceLayoutBuilder;
    ///
    /// let layout = InstanceLayoutBuilder::new(4)
    ///     .with_transform()
    ///     .with_attribute(wgpu::VertexFormat::Float32x2) // e.g., UV offset
    ///     .build();
    /// ```
    pub fn with_attribute(mut self, format: wgpu::VertexFormat) -> Self {
        let size = crate::render_pipeline::vertex_attribute::vertex_format_size(format);
        self.attributes.push(VertexAttributeDescriptor::new(
            format,
            self.stride,
            self.next_location,
        ));
        self.stride += size;
        self.next_location += 1;
        self
    }

    /// Add a custom attribute at a specific offset and location.
    ///
    /// Use this for non-sequential layouts or explicit control.
    pub fn with_attribute_at(
        mut self,
        format: wgpu::VertexFormat,
        offset: u64,
        location: u32,
    ) -> Self {
        let size = crate::render_pipeline::vertex_attribute::vertex_format_size(format);
        self.attributes.push(VertexAttributeDescriptor::new(
            format,
            offset,
            location,
        ));
        // Update stride to max of current and new end position
        self.stride = self.stride.max(offset + size);
        // Update next_location if needed
        self.next_location = self.next_location.max(location + 1);
        self
    }

    /// Set a specific stride (useful for padding/alignment).
    pub fn with_stride(mut self, stride: u64) -> Self {
        self.stride = stride;
        self
    }

    /// Get the current stride without building.
    pub fn current_stride(&self) -> u64 {
        self.stride
    }

    /// Get the next available shader location.
    pub fn next_location(&self) -> u32 {
        self.next_location
    }

    /// Build the final vertex buffer layout descriptor.
    pub fn build(self) -> VertexBufferLayoutDescriptor {
        let mut layout = VertexBufferLayoutDescriptor::per_instance(self.stride);
        for attr in self.attributes {
            layout = layout.attribute(attr);
        }
        layout
    }
}

impl Default for InstanceLayoutBuilder {
    fn default() -> Self {
        Self::new(INSTANCE_TRANSFORM_LOCATIONS.start)
    }
}

// ---------------------------------------------------------------------------
// Preset Instance Layouts
// ---------------------------------------------------------------------------

/// Pre-defined instance layouts for common use cases.
pub mod presets {
    use super::*;

    /// Transform-only instance layout (64 bytes).
    ///
    /// Contains a 4x4 transform matrix at shader locations 4-7.
    ///
    /// # Shader Integration
    ///
    /// ```wgsl
    /// struct InstanceInput {
    ///     @location(4) model_matrix_0: vec4<f32>,
    ///     @location(5) model_matrix_1: vec4<f32>,
    ///     @location(6) model_matrix_2: vec4<f32>,
    ///     @location(7) model_matrix_3: vec4<f32>,
    /// }
    /// ```
    pub fn transform_only() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform()
            .build()
    }

    /// Transform + HDR color instance layout (80 bytes).
    ///
    /// Contains:
    /// - 4x4 transform matrix at locations 4-7
    /// - Float32x4 color at location 8
    pub fn transform_color_float() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .build()
    }

    /// Transform + packed LDR color instance layout (68 bytes).
    ///
    /// Contains:
    /// - 4x4 transform matrix at locations 4-7
    /// - Unorm8x4 color at location 8
    pub fn transform_color_packed() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_packed()
            .build()
    }

    /// Transform + HDR color + custom vec4 instance layout (96 bytes).
    ///
    /// Contains:
    /// - 4x4 transform matrix at locations 4-7
    /// - Float32x4 color at location 8
    /// - Float32x4 custom data at location 9
    pub fn transform_color_custom() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .with_attribute(wgpu::VertexFormat::Float32x4)
            .build()
    }

    /// Compact transform (3x4) + packed color instance layout (52 bytes).
    ///
    /// Memory-efficient layout for affine transforms:
    /// - 3x4 transform matrix at locations 4-6
    /// - Unorm8x4 color at location 7
    pub fn transform_3x4_color_packed() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .with_color_packed()
            .build()
    }

    /// Instance ID only layout (4 bytes).
    ///
    /// Minimal layout for indirect/bindless rendering where transforms
    /// are fetched from a storage buffer.
    pub fn instance_id_only() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::new(INSTANCE_ID_LOCATION)
            .with_instance_id()
            .build()
    }

    /// Transform + instance ID layout (68 bytes).
    ///
    /// Contains:
    /// - 4x4 transform matrix at locations 4-7
    /// - Instance ID at location 8
    pub fn transform_id() -> VertexBufferLayoutDescriptor {
        InstanceLayoutBuilder::default()
            .with_transform()
            .with_instance_id()
            .build()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Create an instance buffer layout from a list of attributes.
///
/// Calculates the stride automatically from the attribute offsets and sizes.
pub fn create_instance_layout(
    attributes: &[VertexAttributeDescriptor],
) -> VertexBufferLayoutDescriptor {
    let stride = attributes
        .iter()
        .map(|attr| {
            attr.offset
                + crate::render_pipeline::vertex_attribute::vertex_format_size(attr.format)
        })
        .max()
        .unwrap_or(0);

    let mut layout = VertexBufferLayoutDescriptor::per_instance(stride);
    for attr in attributes {
        layout = layout.attribute(*attr);
    }
    layout
}

/// Validate that an instance layout has correct step mode.
///
/// Returns `true` if the layout uses `VertexStepMode::Instance`.
pub fn is_valid_instance_layout(layout: &VertexBufferLayoutDescriptor) -> bool {
    layout.step_mode == wgpu::VertexStepMode::Instance
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_layout_builder_transform() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();

        assert_eq!(layout.array_stride, STRIDE_TRANSFORM);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 4);

        // Verify transform matrix rows
        for (i, attr) in layout.attributes.iter().enumerate() {
            assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
            assert_eq!(attr.offset, (i * 16) as u64);
            assert_eq!(
                attr.shader_location,
                INSTANCE_TRANSFORM_LOCATIONS.start + i as u32
            );
        }
    }

    #[test]
    fn test_instance_layout_builder_transform_color_float() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .build();

        assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_FLOAT);
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Float32x4);
        assert_eq!(layout.attributes[4].offset, 64);
        assert_eq!(layout.attributes[4].shader_location, INSTANCE_COLOR_LOCATION);
    }

    #[test]
    fn test_instance_layout_builder_transform_color_packed() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_packed()
            .build();

        assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_PACKED);
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Unorm8x4);
        assert_eq!(layout.attributes[4].offset, 64);
    }

    #[test]
    fn test_instance_layout_builder_transform_3x4() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .build();

        assert_eq!(layout.array_stride, 48);
        assert_eq!(layout.attributes.len(), 3);
        for (i, attr) in layout.attributes.iter().enumerate() {
            assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
            assert_eq!(attr.offset, (i * 16) as u64);
        }
    }

    #[test]
    fn test_instance_layout_builder_instance_id() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_instance_id()
            .build();

        assert_eq!(layout.array_stride, 68);
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Uint32);
    }

    #[test]
    fn test_instance_layout_builder_custom_attribute() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_attribute(wgpu::VertexFormat::Float32x2)
            .build();

        assert_eq!(layout.array_stride, 72); // 64 + 8
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Float32x2);
    }

    #[test]
    fn test_instance_layout_builder_with_attribute_at() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute_at(wgpu::VertexFormat::Float32x4, 0, 0)
            .with_attribute_at(wgpu::VertexFormat::Float32x4, 32, 1) // intentional gap
            .build();

        assert_eq!(layout.array_stride, 48); // 32 + 16
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 32);
    }

    #[test]
    fn test_instance_layout_builder_with_stride() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_stride(128) // pad to 128 bytes
            .build();

        assert_eq!(layout.array_stride, 128);
    }

    #[test]
    fn test_instance_layout_builder_current_stride() {
        let builder = InstanceLayoutBuilder::default().with_transform();
        assert_eq!(builder.current_stride(), 64);
    }

    #[test]
    fn test_instance_layout_builder_next_location() {
        let builder = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float();
        assert_eq!(builder.next_location(), 9);
    }

    #[test]
    fn test_preset_transform_only() {
        let layout = presets::transform_only();
        assert_eq!(layout.array_stride, STRIDE_TRANSFORM);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 4);
    }

    #[test]
    fn test_preset_transform_color_float() {
        let layout = presets::transform_color_float();
        assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_FLOAT);
        assert_eq!(layout.attributes.len(), 5);
    }

    #[test]
    fn test_preset_transform_color_packed() {
        let layout = presets::transform_color_packed();
        assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_PACKED);
        assert_eq!(layout.attributes.len(), 5);
    }

    #[test]
    fn test_preset_transform_color_custom() {
        let layout = presets::transform_color_custom();
        assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_CUSTOM);
        assert_eq!(layout.attributes.len(), 6);
    }

    #[test]
    fn test_preset_transform_3x4_color_packed() {
        let layout = presets::transform_3x4_color_packed();
        assert_eq!(layout.array_stride, 52); // 48 + 4
        assert_eq!(layout.attributes.len(), 4);
    }

    #[test]
    fn test_preset_instance_id_only() {
        let layout = presets::instance_id_only();
        assert_eq!(layout.array_stride, 4);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 1);
        assert_eq!(layout.attributes[0].shader_location, INSTANCE_ID_LOCATION);
    }

    #[test]
    fn test_preset_transform_id() {
        let layout = presets::transform_id();
        assert_eq!(layout.array_stride, 68);
        assert_eq!(layout.attributes.len(), 5);
    }

    #[test]
    fn test_create_instance_layout() {
        let attrs = vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 4),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 16, 5),
        ];
        let layout = create_instance_layout(&attrs);

        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 2);
    }

    #[test]
    fn test_is_valid_instance_layout() {
        let instance_layout = presets::transform_only();
        assert!(is_valid_instance_layout(&instance_layout));

        let vertex_layout = VertexBufferLayoutDescriptor::per_vertex(32);
        assert!(!is_valid_instance_layout(&vertex_layout));
    }

    #[test]
    fn test_step_mode_is_instance() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn test_shader_locations_sequential() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .with_instance_id()
            .build();

        // Transform: 4, 5, 6, 7
        // Color: 8
        // ID: 9
        assert_eq!(layout.attributes[0].shader_location, 4);
        assert_eq!(layout.attributes[1].shader_location, 5);
        assert_eq!(layout.attributes[2].shader_location, 6);
        assert_eq!(layout.attributes[3].shader_location, 7);
        assert_eq!(layout.attributes[4].shader_location, 8);
        assert_eq!(layout.attributes[5].shader_location, 9);
    }

    #[test]
    fn test_builder_default() {
        let builder = InstanceLayoutBuilder::default();
        assert_eq!(builder.next_location(), INSTANCE_TRANSFORM_LOCATIONS.start);
        assert_eq!(builder.current_stride(), 0);
    }

    #[test]
    fn test_complex_layout() {
        // Build a complex layout: transform + color + UV offset + instance ID
        let layout = InstanceLayoutBuilder::default()
            .with_transform()                                    // 64 bytes, loc 4-7
            .with_color_packed()                                 // 4 bytes, loc 8
            .with_attribute(wgpu::VertexFormat::Float32x2)       // 8 bytes, loc 9
            .with_instance_id()                                  // 4 bytes, loc 10
            .build();

        assert_eq!(layout.array_stride, 80); // 64 + 4 + 8 + 4
        assert_eq!(layout.attributes.len(), 7);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn test_empty_layout() {
        let layout = InstanceLayoutBuilder::default().build();
        assert_eq!(layout.array_stride, 0);
        assert!(layout.attributes.is_empty());
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn test_custom_start_location() {
        let layout = InstanceLayoutBuilder::new(10)
            .with_color_float()
            .build();

        assert_eq!(layout.attributes[0].shader_location, 10);
    }

    // =========================================================================
    // Additional Tests - Step Mode Verification
    // =========================================================================

    #[test]
    fn test_all_presets_use_instance_step_mode() {
        // Every preset must use VertexStepMode::Instance
        let presets = [
            presets::transform_only(),
            presets::transform_color_float(),
            presets::transform_color_packed(),
            presets::transform_color_custom(),
            presets::transform_3x4_color_packed(),
            presets::instance_id_only(),
            presets::transform_id(),
        ];

        for (i, layout) in presets.iter().enumerate() {
            assert_eq!(
                layout.step_mode,
                wgpu::VertexStepMode::Instance,
                "Preset {} does not use Instance step mode",
                i
            );
        }
    }

    #[test]
    fn test_builder_always_produces_instance_step_mode() {
        // Verify various builder configurations always use Instance step mode
        let layouts = [
            InstanceLayoutBuilder::new(0).build(),
            InstanceLayoutBuilder::new(4).with_transform().build(),
            InstanceLayoutBuilder::new(0).with_color_float().build(),
            InstanceLayoutBuilder::new(0).with_color_packed().build(),
            InstanceLayoutBuilder::new(0).with_instance_id().build(),
            InstanceLayoutBuilder::new(0).with_transform_3x4().build(),
            InstanceLayoutBuilder::new(0)
                .with_attribute(wgpu::VertexFormat::Float32x4)
                .build(),
        ];

        for (i, layout) in layouts.iter().enumerate() {
            assert_eq!(
                layout.step_mode,
                wgpu::VertexStepMode::Instance,
                "Builder config {} does not use Instance step mode",
                i
            );
        }
    }

    #[test]
    fn test_create_instance_layout_step_mode() {
        let attrs = vec![VertexAttributeDescriptor::new(
            wgpu::VertexFormat::Float32x4,
            0,
            0,
        )];
        let layout = create_instance_layout(&attrs);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    // =========================================================================
    // Transform Matrix Layout Tests
    // =========================================================================

    #[test]
    fn test_transform_4x4_exact_stride() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();
        assert_eq!(layout.array_stride, 64, "4x4 transform should be exactly 64 bytes");
    }

    #[test]
    fn test_transform_3x4_exact_stride() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .build();
        assert_eq!(layout.array_stride, 48, "3x4 transform should be exactly 48 bytes");
    }

    #[test]
    fn test_transform_4x4_attribute_count() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();
        assert_eq!(layout.attributes.len(), 4, "4x4 transform should have 4 vec4 attributes");
    }

    #[test]
    fn test_transform_3x4_attribute_count() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .build();
        assert_eq!(layout.attributes.len(), 3, "3x4 transform should have 3 vec4 attributes");
    }

    #[test]
    fn test_transform_4x4_all_float32x4() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();
        for (i, attr) in layout.attributes.iter().enumerate() {
            assert_eq!(
                attr.format,
                wgpu::VertexFormat::Float32x4,
                "Transform row {} should be Float32x4",
                i
            );
        }
    }

    #[test]
    fn test_transform_3x4_all_float32x4() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .build();
        for (i, attr) in layout.attributes.iter().enumerate() {
            assert_eq!(
                attr.format,
                wgpu::VertexFormat::Float32x4,
                "Transform row {} should be Float32x4",
                i
            );
        }
    }

    #[test]
    fn test_transform_4x4_row_offsets() {
        let layout = InstanceLayoutBuilder::default().with_transform().build();
        assert_eq!(layout.attributes[0].offset, 0, "Row 0 offset");
        assert_eq!(layout.attributes[1].offset, 16, "Row 1 offset");
        assert_eq!(layout.attributes[2].offset, 32, "Row 2 offset");
        assert_eq!(layout.attributes[3].offset, 48, "Row 3 offset");
    }

    #[test]
    fn test_transform_3x4_row_offsets() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform_3x4()
            .build();
        assert_eq!(layout.attributes[0].offset, 0, "Row 0 offset");
        assert_eq!(layout.attributes[1].offset, 16, "Row 1 offset");
        assert_eq!(layout.attributes[2].offset, 32, "Row 2 offset");
    }

    // =========================================================================
    // Builder Combination Tests
    // =========================================================================

    #[test]
    fn test_builder_transform_then_transform_3x4() {
        // Unusual but valid: two transforms back to back
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_transform_3x4()
            .build();

        assert_eq!(layout.array_stride, 112); // 64 + 48
        assert_eq!(layout.attributes.len(), 7); // 4 + 3
    }

    #[test]
    fn test_builder_multiple_colors() {
        let layout = InstanceLayoutBuilder::default()
            .with_color_float()
            .with_color_packed()
            .with_color_float()
            .build();

        assert_eq!(layout.array_stride, 36); // 16 + 4 + 16
        assert_eq!(layout.attributes.len(), 3);
    }

    #[test]
    fn test_builder_all_with_methods() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_transform_3x4()
            .with_color_float()
            .with_color_packed()
            .with_instance_id()
            .with_attribute(wgpu::VertexFormat::Float32x2)
            .build();

        // 64 + 48 + 16 + 4 + 4 + 8 = 144
        assert_eq!(layout.array_stride, 144);
        assert_eq!(layout.attributes.len(), 11); // 4 + 3 + 1 + 1 + 1 + 1
    }

    #[test]
    fn test_builder_color_before_transform() {
        // Color first, then transform
        let layout = InstanceLayoutBuilder::new(0)
            .with_color_float()
            .with_transform()
            .build();

        assert_eq!(layout.array_stride, 80); // 16 + 64
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 16);
    }

    #[test]
    fn test_builder_instance_id_only() {
        let layout = InstanceLayoutBuilder::new(0).with_instance_id().build();
        assert_eq!(layout.array_stride, 4);
        assert_eq!(layout.attributes.len(), 1);
        assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Uint32);
    }

    // =========================================================================
    // Attribute Offset Verification Tests
    // =========================================================================

    #[test]
    fn test_transform_color_float_offsets() {
        let layout = presets::transform_color_float();

        // Transform rows at 0, 16, 32, 48
        for i in 0..4 {
            assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
        }
        // Color at 64
        assert_eq!(layout.attributes[4].offset, 64);
    }

    #[test]
    fn test_transform_color_packed_offsets() {
        let layout = presets::transform_color_packed();

        // Transform rows at 0, 16, 32, 48
        for i in 0..4 {
            assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
        }
        // Packed color at 64
        assert_eq!(layout.attributes[4].offset, 64);
    }

    #[test]
    fn test_transform_color_custom_offsets() {
        let layout = presets::transform_color_custom();

        // Transform rows at 0, 16, 32, 48
        for i in 0..4 {
            assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
        }
        // Color at 64
        assert_eq!(layout.attributes[4].offset, 64);
        // Custom at 80
        assert_eq!(layout.attributes[5].offset, 80);
    }

    #[test]
    fn test_transform_3x4_color_packed_offsets() {
        let layout = presets::transform_3x4_color_packed();

        // Transform rows at 0, 16, 32
        for i in 0..3 {
            assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
        }
        // Packed color at 48
        assert_eq!(layout.attributes[3].offset, 48);
    }

    #[test]
    fn test_with_attribute_at_custom_offsets() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute_at(wgpu::VertexFormat::Float32x4, 0, 0)
            .with_attribute_at(wgpu::VertexFormat::Float32x4, 64, 1) // large gap
            .build();

        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 64);
        assert_eq!(layout.array_stride, 80); // 64 + 16
    }

    #[test]
    fn test_consecutive_offsets_no_gaps() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .with_instance_id()
            .build();

        // Verify no gaps: each attribute starts where previous ends
        let mut expected_offset = 0u64;
        for attr in &layout.attributes {
            assert_eq!(
                attr.offset, expected_offset,
                "Expected offset {} but got {}",
                expected_offset, attr.offset
            );
            expected_offset += crate::render_pipeline::vertex_attribute::vertex_format_size(attr.format);
        }
    }

    // =========================================================================
    // Shader Location Verification Tests
    // =========================================================================

    #[test]
    fn test_preset_transform_only_locations() {
        let layout = presets::transform_only();
        assert_eq!(layout.attributes[0].shader_location, 4);
        assert_eq!(layout.attributes[1].shader_location, 5);
        assert_eq!(layout.attributes[2].shader_location, 6);
        assert_eq!(layout.attributes[3].shader_location, 7);
    }

    #[test]
    fn test_preset_instance_id_only_location() {
        let layout = presets::instance_id_only();
        assert_eq!(layout.attributes[0].shader_location, INSTANCE_ID_LOCATION);
    }

    #[test]
    fn test_locations_are_unique() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_color_float()
            .with_color_packed()
            .with_instance_id()
            .build();

        let mut locations: Vec<u32> = layout.attributes.iter().map(|a| a.shader_location).collect();
        let original_len = locations.len();
        locations.sort();
        locations.dedup();
        assert_eq!(
            locations.len(),
            original_len,
            "Shader locations must be unique"
        );
    }

    #[test]
    fn test_locations_non_overlapping_with_vertex() {
        // Standard vertex attributes use 0-3, instance should use 4+
        let layout = presets::transform_color_float();
        for attr in &layout.attributes {
            assert!(
                attr.shader_location >= INSTANCE_TRANSFORM_LOCATIONS.start,
                "Instance location {} overlaps with vertex locations",
                attr.shader_location
            );
        }
    }

    #[test]
    fn test_custom_start_location_propagates() {
        let layout = InstanceLayoutBuilder::new(20)
            .with_transform()
            .with_color_float()
            .build();

        assert_eq!(layout.attributes[0].shader_location, 20);
        assert_eq!(layout.attributes[1].shader_location, 21);
        assert_eq!(layout.attributes[2].shader_location, 22);
        assert_eq!(layout.attributes[3].shader_location, 23);
        assert_eq!(layout.attributes[4].shader_location, 24);
    }

    // =========================================================================
    // Preset Correctness Tests
    // =========================================================================

    #[test]
    fn test_stride_constants_match_presets() {
        assert_eq!(presets::transform_only().array_stride, STRIDE_TRANSFORM);
        assert_eq!(
            presets::transform_color_float().array_stride,
            STRIDE_TRANSFORM_COLOR_FLOAT
        );
        assert_eq!(
            presets::transform_color_packed().array_stride,
            STRIDE_TRANSFORM_COLOR_PACKED
        );
        assert_eq!(
            presets::transform_color_custom().array_stride,
            STRIDE_TRANSFORM_COLOR_CUSTOM
        );
    }

    #[test]
    fn test_preset_transform_id_stride() {
        // Transform (64) + Uint32 (4) = 68
        assert_eq!(presets::transform_id().array_stride, 68);
    }

    #[test]
    fn test_preset_transform_3x4_color_packed_stride() {
        // 3x4 Transform (48) + Unorm8x4 (4) = 52
        assert_eq!(presets::transform_3x4_color_packed().array_stride, 52);
    }

    #[test]
    fn test_preset_instance_id_only_stride() {
        // Just Uint32 = 4 bytes
        assert_eq!(presets::instance_id_only().array_stride, 4);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_builder_has_no_attributes() {
        let layout = InstanceLayoutBuilder::new(0).build();
        assert!(layout.attributes.is_empty());
    }

    #[test]
    fn test_empty_builder_zero_stride() {
        let layout = InstanceLayoutBuilder::new(0).build();
        assert_eq!(layout.array_stride, 0);
    }

    #[test]
    fn test_single_attribute_builder() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Float32)
            .build();

        assert_eq!(layout.attributes.len(), 1);
        assert_eq!(layout.array_stride, 4);
        assert_eq!(layout.attributes[0].shader_location, 0);
    }

    #[test]
    fn test_many_attributes() {
        // Build layout with many small attributes
        let mut builder = InstanceLayoutBuilder::new(0);
        for _ in 0..16 {
            builder = builder.with_attribute(wgpu::VertexFormat::Float32);
        }
        let layout = builder.build();

        assert_eq!(layout.attributes.len(), 16);
        assert_eq!(layout.array_stride, 64); // 16 * 4
    }

    #[test]
    fn test_with_stride_override_smaller() {
        // Even if with_stride sets a smaller value, it should apply
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_stride(32) // smaller than 64
            .build();

        assert_eq!(layout.array_stride, 32);
    }

    #[test]
    fn test_with_stride_override_larger() {
        let layout = InstanceLayoutBuilder::default()
            .with_transform()
            .with_stride(256) // much larger for alignment
            .build();

        assert_eq!(layout.array_stride, 256);
    }

    #[test]
    fn test_create_instance_layout_empty() {
        let layout = create_instance_layout(&[]);
        assert_eq!(layout.array_stride, 0);
        assert!(layout.attributes.is_empty());
    }

    #[test]
    fn test_create_instance_layout_calculates_stride() {
        let attrs = vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 0),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 100, 1), // gap
        ];
        let layout = create_instance_layout(&attrs);

        // Stride should be 100 + 16 = 116 (max end offset)
        assert_eq!(layout.array_stride, 116);
    }

    #[test]
    fn test_is_valid_instance_layout_true() {
        let layouts = [
            presets::transform_only(),
            presets::transform_color_float(),
            InstanceLayoutBuilder::default().build(),
        ];

        for layout in &layouts {
            assert!(is_valid_instance_layout(layout));
        }
    }

    #[test]
    fn test_is_valid_instance_layout_false_for_vertex() {
        let vertex_layout = VertexBufferLayoutDescriptor::per_vertex(64);
        assert!(!is_valid_instance_layout(&vertex_layout));
    }

    // =========================================================================
    // Various Vertex Format Tests
    // =========================================================================

    #[test]
    fn test_attribute_format_uint32() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Uint32)
            .build();
        assert_eq!(layout.array_stride, 4);
    }

    #[test]
    fn test_attribute_format_sint32x2() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Sint32x2)
            .build();
        assert_eq!(layout.array_stride, 8);
    }

    #[test]
    fn test_attribute_format_float32x3() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Float32x3)
            .build();
        assert_eq!(layout.array_stride, 12);
    }

    #[test]
    fn test_attribute_format_unorm8x2() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Unorm8x2)
            .build();
        assert_eq!(layout.array_stride, 2);
    }

    #[test]
    fn test_mixed_format_layout() {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Float32x4)   // 16
            .with_attribute(wgpu::VertexFormat::Uint32)       // 4
            .with_attribute(wgpu::VertexFormat::Unorm8x4)     // 4
            .with_attribute(wgpu::VertexFormat::Float32x2)    // 8
            .build();

        assert_eq!(layout.array_stride, 32); // 16 + 4 + 4 + 8
    }

    // =========================================================================
    // Constants Validation Tests
    // =========================================================================

    #[test]
    fn test_instance_transform_locations_range() {
        assert_eq!(INSTANCE_TRANSFORM_LOCATIONS.start, 4);
        assert_eq!(INSTANCE_TRANSFORM_LOCATIONS.end, 8);
        assert_eq!(INSTANCE_TRANSFORM_LOCATIONS.len(), 4);
    }

    #[test]
    fn test_instance_color_location_value() {
        assert_eq!(INSTANCE_COLOR_LOCATION, 8);
    }

    #[test]
    fn test_instance_id_location_value() {
        assert_eq!(INSTANCE_ID_LOCATION, 9);
    }

    #[test]
    fn test_instance_custom_start_location_value() {
        assert_eq!(INSTANCE_CUSTOM_START_LOCATION, 10);
    }

    #[test]
    fn test_stride_constants_values() {
        assert_eq!(STRIDE_TRANSFORM, 64);
        assert_eq!(STRIDE_TRANSFORM_COLOR_PACKED, 68);
        assert_eq!(STRIDE_TRANSFORM_COLOR_FLOAT, 80);
        assert_eq!(STRIDE_TRANSFORM_COLOR_CUSTOM, 96);
    }

    // =========================================================================
    // Builder State Tracking Tests
    // =========================================================================

    #[test]
    fn test_next_location_after_transform() {
        let builder = InstanceLayoutBuilder::default().with_transform();
        assert_eq!(builder.next_location(), 8); // 4 + 4
    }

    #[test]
    fn test_next_location_after_transform_3x4() {
        let builder = InstanceLayoutBuilder::default().with_transform_3x4();
        assert_eq!(builder.next_location(), 7); // 4 + 3
    }

    #[test]
    fn test_next_location_after_color_float() {
        let builder = InstanceLayoutBuilder::default().with_color_float();
        assert_eq!(builder.next_location(), 5); // 4 + 1
    }

    #[test]
    fn test_next_location_increments_correctly() {
        let builder = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Float32x4)
            .with_attribute(wgpu::VertexFormat::Float32x4)
            .with_attribute(wgpu::VertexFormat::Float32x4);
        assert_eq!(builder.next_location(), 3);
    }

    #[test]
    fn test_current_stride_accumulates() {
        let builder = InstanceLayoutBuilder::new(0)
            .with_attribute(wgpu::VertexFormat::Float32x4); // 16
        assert_eq!(builder.current_stride(), 16);

        let builder = builder.with_attribute(wgpu::VertexFormat::Float32x4); // +16
        assert_eq!(builder.current_stride(), 32);

        let builder = builder.with_attribute(wgpu::VertexFormat::Uint32); // +4
        assert_eq!(builder.current_stride(), 36);
    }
}
