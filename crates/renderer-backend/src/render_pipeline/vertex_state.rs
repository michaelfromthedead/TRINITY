//! Vertex state descriptors for render pipelines.

// ---------------------------------------------------------------------------
// VertexAttributeDescriptor
// ---------------------------------------------------------------------------

/// Describes a single vertex attribute within a vertex buffer.
///
/// This is an owned version of [`wgpu::VertexAttribute`] that doesn't require
/// managing borrow lifetimes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexAttributeDescriptor {
    /// The GPU data format of the attribute (e.g., `Float32x3`).
    pub format: wgpu::VertexFormat,
    /// Byte offset from the start of the vertex buffer element.
    pub offset: u64,
    /// The `@location(...)` index in the vertex shader.
    pub shader_location: u32,
}

impl VertexAttributeDescriptor {
    /// Create a new vertex attribute descriptor.
    pub const fn new(format: wgpu::VertexFormat, offset: u64, shader_location: u32) -> Self {
        Self {
            format,
            offset,
            shader_location,
        }
    }
}

impl From<VertexAttributeDescriptor> for wgpu::VertexAttribute {
    fn from(attr: VertexAttributeDescriptor) -> Self {
        wgpu::VertexAttribute {
            format: attr.format,
            offset: attr.offset,
            shader_location: attr.shader_location,
        }
    }
}

// ---------------------------------------------------------------------------
// VertexBufferLayoutDescriptor
// ---------------------------------------------------------------------------

/// Describes the layout of a single vertex buffer.
///
/// This is an owned version of [`wgpu::VertexBufferLayout`] that stores its
/// attributes in a `Vec` instead of a slice.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexBufferLayoutDescriptor {
    /// Byte stride between consecutive vertex elements.
    pub array_stride: u64,
    /// Whether the buffer advances per vertex or per instance.
    pub step_mode: wgpu::VertexStepMode,
    /// The vertex attributes in this buffer.
    pub attributes: Vec<VertexAttributeDescriptor>,
}

impl VertexBufferLayoutDescriptor {
    /// Create a new vertex buffer layout.
    pub fn new(array_stride: u64, step_mode: wgpu::VertexStepMode) -> Self {
        Self {
            array_stride,
            step_mode,
            attributes: Vec::new(),
        }
    }

    /// Create a per-vertex buffer layout.
    pub fn per_vertex(array_stride: u64) -> Self {
        Self::new(array_stride, wgpu::VertexStepMode::Vertex)
    }

    /// Create a per-instance buffer layout.
    pub fn per_instance(array_stride: u64) -> Self {
        Self::new(array_stride, wgpu::VertexStepMode::Instance)
    }

    /// Add an attribute to this buffer layout.
    pub fn attribute(mut self, attr: VertexAttributeDescriptor) -> Self {
        self.attributes.push(attr);
        self
    }

    /// Add an attribute with the given format, offset, and location.
    pub fn with_attribute(
        mut self,
        format: wgpu::VertexFormat,
        offset: u64,
        shader_location: u32,
    ) -> Self {
        self.attributes.push(VertexAttributeDescriptor::new(
            format,
            offset,
            shader_location,
        ));
        self
    }
}

// ---------------------------------------------------------------------------
// VertexStateDescriptor
// ---------------------------------------------------------------------------

/// Describes the vertex stage of a render pipeline.
///
/// # Required Fields
///
/// - `module`: The vertex shader module (required at construction)
///
/// # Optional Fields
///
/// - `entry_point`: Entry function name (default: `"vs_main"`)
/// - `compilation_options`: Shader compilation options (default: empty)
/// - `buffers`: Vertex buffer layouts (default: empty)
#[derive(Debug, Clone)]
pub struct VertexStateDescriptor<'a> {
    /// The vertex shader module.
    pub module: &'a wgpu::ShaderModule,
    /// The entry point function name.
    pub entry_point: &'a str,
    /// Shader compilation options.
    pub compilation_options: wgpu::PipelineCompilationOptions<'a>,
    /// Vertex buffer layouts.
    pub buffers: Vec<VertexBufferLayoutDescriptor>,
}

impl<'a> VertexStateDescriptor<'a> {
    /// Create a new vertex state descriptor with the given shader module.
    ///
    /// Uses `"vs_main"` as the default entry point.
    pub fn new(module: &'a wgpu::ShaderModule) -> Self {
        Self {
            module,
            entry_point: "vs_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: Vec::new(),
        }
    }

    /// Set the entry point function name.
    pub fn entry_point(mut self, entry_point: &'a str) -> Self {
        self.entry_point = entry_point;
        self
    }

    /// Set the shader compilation options.
    pub fn compilation_options(
        mut self,
        options: wgpu::PipelineCompilationOptions<'a>,
    ) -> Self {
        self.compilation_options = options;
        self
    }

    /// Add a vertex buffer layout.
    pub fn buffer(mut self, layout: VertexBufferLayoutDescriptor) -> Self {
        self.buffers.push(layout);
        self
    }

    /// Set all vertex buffer layouts at once.
    pub fn buffers(mut self, buffers: Vec<VertexBufferLayoutDescriptor>) -> Self {
        self.buffers = buffers;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_attribute_descriptor() {
        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);
        assert_eq!(attr.format, wgpu::VertexFormat::Float32x3);
        assert_eq!(attr.offset, 0);
        assert_eq!(attr.shader_location, 0);
    }

    #[test]
    fn test_vertex_attribute_into_wgpu() {
        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 12, 1);
        let wgpu_attr: wgpu::VertexAttribute = attr.into();
        assert_eq!(wgpu_attr.format, wgpu::VertexFormat::Float32x2);
        assert_eq!(wgpu_attr.offset, 12);
        assert_eq!(wgpu_attr.shader_location, 1);
    }

    #[test]
    fn test_vertex_buffer_layout_builder() {
        let layout = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1);

        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(layout.attributes.len(), 2);
    }

    #[test]
    fn test_vertex_buffer_layout_per_instance() {
        let layout = VertexBufferLayoutDescriptor::per_instance(64);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_format_float32() {
        // Test Float32 variants
        let attr1 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32, 0, 0);
        assert_eq!(attr1.format, wgpu::VertexFormat::Float32);

        let attr2 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 4, 1);
        assert_eq!(attr2.format, wgpu::VertexFormat::Float32x2);

        let attr3 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 12, 2);
        assert_eq!(attr3.format, wgpu::VertexFormat::Float32x3);

        let attr4 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 24, 3);
        assert_eq!(attr4.format, wgpu::VertexFormat::Float32x4);
    }

    #[test]
    fn test_vertex_format_sint32() {
        // Test Sint32 variants
        let attr1 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Sint32, 0, 0);
        assert_eq!(attr1.format, wgpu::VertexFormat::Sint32);

        let attr2 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Sint32x2, 4, 1);
        assert_eq!(attr2.format, wgpu::VertexFormat::Sint32x2);

        let attr3 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Sint32x3, 12, 2);
        assert_eq!(attr3.format, wgpu::VertexFormat::Sint32x3);

        let attr4 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Sint32x4, 24, 3);
        assert_eq!(attr4.format, wgpu::VertexFormat::Sint32x4);
    }

    #[test]
    fn test_vertex_format_uint32() {
        // Test Uint32 variants
        let attr1 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32, 0, 0);
        assert_eq!(attr1.format, wgpu::VertexFormat::Uint32);

        let attr2 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32x2, 4, 1);
        assert_eq!(attr2.format, wgpu::VertexFormat::Uint32x2);

        let attr3 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32x3, 12, 2);
        assert_eq!(attr3.format, wgpu::VertexFormat::Uint32x3);

        let attr4 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32x4, 24, 3);
        assert_eq!(attr4.format, wgpu::VertexFormat::Uint32x4);
    }

    #[test]
    fn test_vertex_format_normalized() {
        // Test normalized formats (common for colors)
        let unorm8 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Unorm8x4, 0, 0);
        assert_eq!(unorm8.format, wgpu::VertexFormat::Unorm8x4);

        let snorm8 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Snorm8x4, 4, 1);
        assert_eq!(snorm8.format, wgpu::VertexFormat::Snorm8x4);

        let unorm16 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Unorm16x4, 8, 2);
        assert_eq!(unorm16.format, wgpu::VertexFormat::Unorm16x4);
    }

    #[test]
    fn test_vertex_format_float16() {
        // Test Float16 variants
        let f16x2 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float16x2, 0, 0);
        assert_eq!(f16x2.format, wgpu::VertexFormat::Float16x2);

        let f16x4 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float16x4, 4, 1);
        assert_eq!(f16x4.format, wgpu::VertexFormat::Float16x4);
    }

    #[test]
    fn test_multiple_buffer_layouts() {
        // Test multiple vertex buffers (common for instanced rendering)
        let position_buffer = VertexBufferLayoutDescriptor::per_vertex(12)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

        let normal_uv_buffer = VertexBufferLayoutDescriptor::per_vertex(20)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 1)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 2);

        let instance_buffer = VertexBufferLayoutDescriptor::per_instance(64)
            .with_attribute(wgpu::VertexFormat::Float32x4, 0, 3)
            .with_attribute(wgpu::VertexFormat::Float32x4, 16, 4)
            .with_attribute(wgpu::VertexFormat::Float32x4, 32, 5)
            .with_attribute(wgpu::VertexFormat::Float32x4, 48, 6);

        assert_eq!(position_buffer.attributes.len(), 1);
        assert_eq!(normal_uv_buffer.attributes.len(), 2);
        assert_eq!(instance_buffer.attributes.len(), 4);
        assert_eq!(instance_buffer.step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn test_empty_buffer_layout() {
        // Test empty buffer layout (edge case)
        let empty = VertexBufferLayoutDescriptor::per_vertex(0);
        assert_eq!(empty.array_stride, 0);
        assert!(empty.attributes.is_empty());
    }

    #[test]
    fn test_shader_location_assignment() {
        // Test sequential shader locations
        let layout = VertexBufferLayoutDescriptor::per_vertex(48)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)   // position
            .with_attribute(wgpu::VertexFormat::Float32x3, 12, 1)  // normal
            .with_attribute(wgpu::VertexFormat::Float32x2, 24, 2)  // uv0
            .with_attribute(wgpu::VertexFormat::Float32x2, 32, 3)  // uv1
            .with_attribute(wgpu::VertexFormat::Float32x4, 40, 4); // tangent (intentional gap)

        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[1].shader_location, 1);
        assert_eq!(layout.attributes[2].shader_location, 2);
        assert_eq!(layout.attributes[3].shader_location, 3);
        assert_eq!(layout.attributes[4].shader_location, 4);
    }

    #[test]
    fn test_shader_location_non_sequential() {
        // Test non-sequential shader locations (valid in wgpu)
        let layout = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
            .with_attribute(wgpu::VertexFormat::Float32x3, 12, 5)
            .with_attribute(wgpu::VertexFormat::Float32x2, 24, 10);

        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[1].shader_location, 5);
        assert_eq!(layout.attributes[2].shader_location, 10);
    }

    #[test]
    fn test_vertex_attribute_descriptor_equality() {
        // Test PartialEq implementation
        let attr1 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);
        let attr2 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);
        let attr3 = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 0, 0);

        assert_eq!(attr1, attr2);
        assert_ne!(attr1, attr3);
    }

    #[test]
    fn test_vertex_buffer_layout_equality() {
        // Test PartialEq implementation
        let layout1 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);
        let layout2 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);
        let layout3 = VertexBufferLayoutDescriptor::per_instance(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

        assert_eq!(layout1, layout2);
        assert_ne!(layout1, layout3);
    }

    #[test]
    fn test_vertex_attribute_copy() {
        // Test Copy trait
        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);
        let attr_copy = attr;
        assert_eq!(attr, attr_copy);
    }

    #[test]
    fn test_vertex_buffer_layout_clone() {
        // Test Clone trait
        let layout = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);
        let layout_clone = layout.clone();
        assert_eq!(layout, layout_clone);
    }

    #[test]
    fn test_vertex_buffer_layout_with_attribute_method() {
        // Test attribute() method vs with_attribute()
        let layout1 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0);
        let layout2 = VertexBufferLayoutDescriptor::per_vertex(32).attribute(attr);

        assert_eq!(layout1, layout2);
    }

    #[test]
    fn test_large_vertex_offset() {
        // Test large offset values
        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 1024, 0);
        assert_eq!(attr.offset, 1024);

        let wgpu_attr: wgpu::VertexAttribute = attr.into();
        assert_eq!(wgpu_attr.offset, 1024);
    }

    #[test]
    fn test_high_shader_location() {
        // Test high shader location (wgpu allows up to 16)
        let attr = VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 15);
        assert_eq!(attr.shader_location, 15);
    }
}
