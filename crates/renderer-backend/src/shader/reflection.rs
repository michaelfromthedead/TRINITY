//! Shader Reflection Engine for TRINITY (T-AS-3.3).
//!
//! This module provides comprehensive shader reflection to extract bindings and produce
//! pipeline layout descriptions for Vulkan, Metal, and D3D12 backends.
//!
//! # Capabilities
//!
//! - **Resource Discovery**: Uniform buffers, storage buffers, sampled textures, samplers,
//!   storage textures, push constants, specialization constants
//! - **Detailed Extraction**: Binding slot, size, member layout, read/write flags, dimension,
//!   array size, format hints
//! - **Pipeline Layout Generation**: VkPipelineLayout, MTLRenderPipelineDescriptor,
//!   D3D12 Root Signature descriptions
//! - **Descriptor Set Management**: Set allocation hints, update helpers
//! - **Specialization Constants**: Override support at PSO creation time
//!
//! # Performance
//!
//! Target: <10ms per entry point reflection.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::reflection::{ShaderReflector, ReflectionOptions};
//!
//! let reflector = ShaderReflector::new();
//! let result = reflector.reflect(source, &ReflectionOptions::default())?;
//!
//! // Generate Vulkan pipeline layout
//! let vk_layout = result.to_vulkan_layout();
//!
//! // Generate Metal argument buffer layout
//! let mtl_layout = result.to_metal_layout();
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use super::naga_compiler::{
    CompileError, CompileErrorKind, NagaCompiler, ShaderStage, TextureDimension,
};

// ---------------------------------------------------------------------------
// Scalar and Type Primitives
// ---------------------------------------------------------------------------

/// Scalar type for shader values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    Int32,
    Uint32,
    Float16,
    Float32,
    Float64,
}

impl ScalarType {
    /// Size in bytes.
    pub fn size(&self) -> u32 {
        match self {
            ScalarType::Bool => 4,  // WGSL bool is 4 bytes
            ScalarType::Int32 => 4,
            ScalarType::Uint32 => 4,
            ScalarType::Float16 => 2,
            ScalarType::Float32 => 4,
            ScalarType::Float64 => 8,
        }
    }

    /// Alignment in bytes.
    pub fn alignment(&self) -> u32 {
        self.size()
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::Int32 => write!(f, "i32"),
            ScalarType::Uint32 => write!(f, "u32"),
            ScalarType::Float16 => write!(f, "f16"),
            ScalarType::Float32 => write!(f, "f32"),
            ScalarType::Float64 => write!(f, "f64"),
        }
    }
}

/// Vector size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorSize {
    Vec2,
    Vec3,
    Vec4,
}

impl VectorSize {
    pub fn count(&self) -> u32 {
        match self {
            VectorSize::Vec2 => 2,
            VectorSize::Vec3 => 3,
            VectorSize::Vec4 => 4,
        }
    }
}

/// Matrix dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixDimensions {
    pub columns: u8,
    pub rows: u8,
}

// ---------------------------------------------------------------------------
// Type Layout
// ---------------------------------------------------------------------------

/// Detailed type information with layout.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeLayout {
    /// Scalar type.
    Scalar(ScalarType),

    /// Vector type.
    Vector {
        scalar: ScalarType,
        size: VectorSize,
    },

    /// Matrix type (column-major).
    Matrix {
        scalar: ScalarType,
        columns: u8,
        rows: u8,
    },

    /// Struct type with members.
    Struct {
        name: String,
        members: Vec<StructMember>,
        size: u32,
        alignment: u32,
    },

    /// Array type.
    Array {
        element: Box<TypeLayout>,
        count: Option<u32>,  // None = runtime-sized
        stride: u32,
    },

    /// Texture type.
    Texture {
        dimension: TextureDimension,
        format: Option<TextureFormat>,
        multisampled: bool,
        depth: bool,
    },

    /// Storage texture type.
    StorageTexture {
        dimension: TextureDimension,
        format: TextureFormat,
        access: StorageTextureAccess,
    },

    /// Sampler type.
    Sampler {
        comparison: bool,
    },

    /// Opaque/unknown type.
    Opaque {
        name: String,
    },
}

impl TypeLayout {
    /// Calculate size in bytes (for buffer-backed types).
    pub fn size(&self) -> u32 {
        match self {
            TypeLayout::Scalar(s) => s.size(),
            TypeLayout::Vector { scalar, size } => scalar.size() * size.count(),
            TypeLayout::Matrix { scalar, columns, rows } => {
                // Column-major with vec4 alignment per column
                let column_size = scalar.size() * (*rows as u32);
                let column_stride = round_up(column_size, 16); // vec4 alignment
                column_stride * (*columns as u32)
            }
            TypeLayout::Struct { size, .. } => *size,
            TypeLayout::Array { element: _, count, stride } => {
                count.map(|n| stride * n).unwrap_or(*stride)
            }
            _ => 0, // Opaque types have no size
        }
    }

    /// Calculate alignment in bytes.
    pub fn alignment(&self) -> u32 {
        match self {
            TypeLayout::Scalar(s) => s.alignment(),
            TypeLayout::Vector { scalar, size } => {
                // vec2 = 2x scalar, vec3/vec4 = 4x scalar
                match size {
                    VectorSize::Vec2 => scalar.size() * 2,
                    VectorSize::Vec3 | VectorSize::Vec4 => scalar.size() * 4,
                }
            }
            TypeLayout::Matrix { scalar, .. } => scalar.size() * 4, // vec4 alignment
            TypeLayout::Struct { alignment, .. } => *alignment,
            TypeLayout::Array { element, .. } => round_up(element.alignment(), 16),
            _ => 1,
        }
    }

    /// Check if this is a buffer-backed type.
    pub fn is_buffer_type(&self) -> bool {
        matches!(
            self,
            TypeLayout::Scalar(_)
                | TypeLayout::Vector { .. }
                | TypeLayout::Matrix { .. }
                | TypeLayout::Struct { .. }
                | TypeLayout::Array { .. }
        )
    }
}

/// Member of a struct.
#[derive(Debug, Clone, PartialEq)]
pub struct StructMember {
    /// Member name.
    pub name: String,
    /// Member type.
    pub ty: TypeLayout,
    /// Byte offset within struct.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
}

/// Texture format hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    // Unsigned normalized
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,

    // Signed normalized
    R8Snorm,
    Rg8Snorm,
    Rgba8Snorm,
    R16Snorm,
    Rg16Snorm,
    Rgba16Snorm,

    // Float
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,

    // Unsigned integer
    R8Uint,
    Rg8Uint,
    Rgba8Uint,
    R16Uint,
    Rg16Uint,
    Rgba16Uint,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,

    // Signed integer
    R8Sint,
    Rg8Sint,
    Rgba8Sint,
    R16Sint,
    Rg16Sint,
    Rgba16Sint,
    R32Sint,
    Rg32Sint,
    Rgba32Sint,

    // Depth/stencil
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,

    // Packed
    Rgb10a2Unorm,
    Rg11b10Float,
}

impl TextureFormat {
    /// Get format from naga scalar kind and width.
    pub fn from_naga_scalar(kind: naga::ScalarKind, width: u8, _arrayed: bool) -> Option<Self> {
        match (kind, width) {
            (naga::ScalarKind::Float, 4) => Some(TextureFormat::Rgba32Float),
            (naga::ScalarKind::Float, 2) => Some(TextureFormat::Rgba16Float),
            (naga::ScalarKind::Uint, 4) => Some(TextureFormat::Rgba32Uint),
            (naga::ScalarKind::Uint, 2) => Some(TextureFormat::Rgba16Uint),
            (naga::ScalarKind::Uint, 1) => Some(TextureFormat::Rgba8Uint),
            (naga::ScalarKind::Sint, 4) => Some(TextureFormat::Rgba32Sint),
            (naga::ScalarKind::Sint, 2) => Some(TextureFormat::Rgba16Sint),
            (naga::ScalarKind::Sint, 1) => Some(TextureFormat::Rgba8Sint),
            _ => None,
        }
    }

    /// Get format from naga storage format.
    pub fn from_naga_storage(format: naga::StorageFormat) -> Self {
        match format {
            naga::StorageFormat::R8Unorm => TextureFormat::R8Unorm,
            naga::StorageFormat::R8Snorm => TextureFormat::R8Snorm,
            naga::StorageFormat::R8Uint => TextureFormat::R8Uint,
            naga::StorageFormat::R8Sint => TextureFormat::R8Sint,
            naga::StorageFormat::R16Uint => TextureFormat::R16Uint,
            naga::StorageFormat::R16Sint => TextureFormat::R16Sint,
            naga::StorageFormat::R16Float => TextureFormat::R16Float,
            naga::StorageFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
            naga::StorageFormat::Rg8Snorm => TextureFormat::Rg8Snorm,
            naga::StorageFormat::Rg8Uint => TextureFormat::Rg8Uint,
            naga::StorageFormat::Rg8Sint => TextureFormat::Rg8Sint,
            naga::StorageFormat::R32Uint => TextureFormat::R32Uint,
            naga::StorageFormat::R32Sint => TextureFormat::R32Sint,
            naga::StorageFormat::R32Float => TextureFormat::R32Float,
            naga::StorageFormat::Rg16Uint => TextureFormat::Rg16Uint,
            naga::StorageFormat::Rg16Sint => TextureFormat::Rg16Sint,
            naga::StorageFormat::Rg16Float => TextureFormat::Rg16Float,
            naga::StorageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            naga::StorageFormat::Rgba8Snorm => TextureFormat::Rgba8Snorm,
            naga::StorageFormat::Rgba8Uint => TextureFormat::Rgba8Uint,
            naga::StorageFormat::Rgba8Sint => TextureFormat::Rgba8Sint,
            naga::StorageFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
            naga::StorageFormat::Rgb10a2Unorm => TextureFormat::Rgb10a2Unorm,
            naga::StorageFormat::Rg11b10Ufloat => TextureFormat::Rg11b10Float,
            naga::StorageFormat::Rg32Uint => TextureFormat::Rg32Uint,
            naga::StorageFormat::Rg32Sint => TextureFormat::Rg32Sint,
            naga::StorageFormat::Rg32Float => TextureFormat::Rg32Float,
            naga::StorageFormat::Rgba16Uint => TextureFormat::Rgba16Uint,
            naga::StorageFormat::Rgba16Sint => TextureFormat::Rgba16Sint,
            naga::StorageFormat::Rgba16Float => TextureFormat::Rgba16Float,
            naga::StorageFormat::Rgba32Uint => TextureFormat::Rgba32Uint,
            naga::StorageFormat::Rgba32Sint => TextureFormat::Rgba32Sint,
            naga::StorageFormat::Rgba32Float => TextureFormat::Rgba32Float,
            naga::StorageFormat::R16Unorm => TextureFormat::R16Unorm,
            naga::StorageFormat::R16Snorm => TextureFormat::R16Snorm,
            naga::StorageFormat::Rg16Unorm => TextureFormat::Rg16Unorm,
            naga::StorageFormat::Rg16Snorm => TextureFormat::Rg16Snorm,
            naga::StorageFormat::Rgba16Unorm => TextureFormat::Rgba16Unorm,
            naga::StorageFormat::Rgba16Snorm => TextureFormat::Rgba16Snorm,
            naga::StorageFormat::Rgb10a2Uint => TextureFormat::Rgba32Uint, // Approximation
            naga::StorageFormat::R64Uint => TextureFormat::R32Uint, // Approximation - R64 not commonly supported
        }
    }
}

/// Storage texture access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTextureAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

// ---------------------------------------------------------------------------
// Resource Binding with Layout
// ---------------------------------------------------------------------------

/// Enhanced resource binding with detailed type layout.
#[derive(Debug, Clone)]
pub struct ReflectedBinding {
    /// Binding group (descriptor set).
    pub group: u32,
    /// Binding index within group.
    pub binding: u32,
    /// Variable name in shader.
    pub name: String,
    /// Resource kind.
    pub kind: BindingKind,
    /// Type layout (for buffer types).
    pub layout: TypeLayout,
    /// Shader stages that use this binding.
    pub stages: Vec<ShaderStage>,
    /// Array size (1 = not an array).
    pub array_size: u32,
    /// Whether this binding is read-only.
    pub read_only: bool,
}

impl ReflectedBinding {
    /// Check if this is a buffer binding.
    pub fn is_buffer(&self) -> bool {
        matches!(self.kind, BindingKind::UniformBuffer | BindingKind::StorageBuffer)
    }

    /// Check if this is a texture binding.
    pub fn is_texture(&self) -> bool {
        matches!(
            self.kind,
            BindingKind::SampledTexture
                | BindingKind::StorageTexture
                | BindingKind::DepthTexture
        )
    }

    /// Check if this is a sampler binding.
    pub fn is_sampler(&self) -> bool {
        matches!(self.kind, BindingKind::Sampler | BindingKind::ComparisonSampler)
    }

    /// Get the buffer size if applicable.
    pub fn buffer_size(&self) -> Option<u32> {
        if self.is_buffer() {
            Some(self.layout.size())
        } else {
            None
        }
    }
}

/// Kind of binding resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingKind {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    StorageTexture,
    DepthTexture,
    Sampler,
    ComparisonSampler,
}

impl fmt::Display for BindingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindingKind::UniformBuffer => write!(f, "uniform buffer"),
            BindingKind::StorageBuffer => write!(f, "storage buffer"),
            BindingKind::SampledTexture => write!(f, "sampled texture"),
            BindingKind::StorageTexture => write!(f, "storage texture"),
            BindingKind::DepthTexture => write!(f, "depth texture"),
            BindingKind::Sampler => write!(f, "sampler"),
            BindingKind::ComparisonSampler => write!(f, "comparison sampler"),
        }
    }
}

// ---------------------------------------------------------------------------
// Push Constants
// ---------------------------------------------------------------------------

/// Push constant range with member layout.
#[derive(Debug, Clone)]
pub struct ReflectedPushConstant {
    /// Offset in bytes.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Shader stages that use this.
    pub stages: Vec<ShaderStage>,
    /// Type layout with members.
    pub layout: TypeLayout,
}

// ---------------------------------------------------------------------------
// Specialization Constants
// ---------------------------------------------------------------------------

/// Specialization constant with type and default value.
#[derive(Debug, Clone)]
pub struct ReflectedSpecConstant {
    /// Specialization constant ID.
    pub id: u32,
    /// Variable name.
    pub name: String,
    /// Scalar type.
    pub ty: ScalarType,
    /// Default value.
    pub default_value: SpecConstantValue,
}

/// Specialization constant value.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecConstantValue {
    Bool(bool),
    Int32(i32),
    Uint32(u32),
    Float32(f32),
}

impl SpecConstantValue {
    /// Convert to bytes for SPIR-V.
    pub fn to_bytes(&self) -> [u8; 4] {
        match self {
            SpecConstantValue::Bool(v) => (if *v { 1u32 } else { 0u32 }).to_le_bytes(),
            SpecConstantValue::Int32(v) => v.to_le_bytes(),
            SpecConstantValue::Uint32(v) => v.to_le_bytes(),
            SpecConstantValue::Float32(v) => v.to_le_bytes(),
        }
    }
}

/// Override for specialization constant at PSO creation.
#[derive(Debug, Clone)]
pub struct SpecConstantOverride {
    /// Constant ID.
    pub id: u32,
    /// New value.
    pub value: SpecConstantValue,
}

// ---------------------------------------------------------------------------
// Entry Point Reflection
// ---------------------------------------------------------------------------

/// Reflected entry point information.
#[derive(Debug, Clone)]
pub struct ReflectedEntryPoint {
    /// Entry point name.
    pub name: String,
    /// Shader stage.
    pub stage: ShaderStage,
    /// Workgroup size (compute only).
    pub workgroup_size: Option<[u32; 3]>,
    /// Input variables.
    pub inputs: Vec<VertexAttribute>,
    /// Output variables.
    pub outputs: Vec<OutputAttribute>,
    /// Bindings used by this entry point.
    pub used_bindings: Vec<(u32, u32)>, // (group, binding)
}

/// Vertex input attribute.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    /// Location index.
    pub location: u32,
    /// Variable name.
    pub name: String,
    /// Type layout.
    pub layout: TypeLayout,
    /// Builtin (if this is a builtin input).
    pub builtin: Option<BuiltinType>,
}

/// Output attribute.
#[derive(Debug, Clone)]
pub struct OutputAttribute {
    /// Location index (None for builtins).
    pub location: Option<u32>,
    /// Variable name.
    pub name: String,
    /// Type layout.
    pub layout: TypeLayout,
    /// Builtin (if this is a builtin output).
    pub builtin: Option<BuiltinType>,
}

/// Shader builtin type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Position,
    VertexIndex,
    InstanceIndex,
    FrontFacing,
    FragDepth,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SampleIndex,
    SampleMask,
}

// ---------------------------------------------------------------------------
// Pipeline Layout Descriptions
// ---------------------------------------------------------------------------

/// Descriptor set layout description.
#[derive(Debug, Clone)]
pub struct DescriptorSetLayout {
    /// Set index.
    pub set: u32,
    /// Bindings in this set.
    pub bindings: Vec<DescriptorBinding>,
}

/// Descriptor binding description.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    /// Binding index.
    pub binding: u32,
    /// Descriptor type.
    pub descriptor_type: DescriptorType,
    /// Array count.
    pub count: u32,
    /// Stages that access this binding.
    pub stages: VkShaderStageFlags,
}

/// Vulkan-style descriptor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorType {
    UniformBuffer,
    UniformBufferDynamic,
    StorageBuffer,
    StorageBufferDynamic,
    SampledImage,
    StorageImage,
    CombinedImageSampler,
    Sampler,
    InputAttachment,
}

/// Vulkan shader stage flags (bitmask).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VkShaderStageFlags(pub u32);

impl VkShaderStageFlags {
    pub const VERTEX: VkShaderStageFlags = VkShaderStageFlags(0x00000001);
    pub const FRAGMENT: VkShaderStageFlags = VkShaderStageFlags(0x00000010);
    pub const COMPUTE: VkShaderStageFlags = VkShaderStageFlags(0x00000020);
    pub const ALL_GRAPHICS: VkShaderStageFlags = VkShaderStageFlags(0x0000001F);
    pub const ALL: VkShaderStageFlags = VkShaderStageFlags(0x7FFFFFFF);

    pub fn from_stages(stages: &[ShaderStage]) -> Self {
        let mut flags = 0u32;
        for stage in stages {
            flags |= match stage {
                ShaderStage::Vertex => Self::VERTEX.0,
                ShaderStage::Fragment => Self::FRAGMENT.0,
                ShaderStage::Compute => Self::COMPUTE.0,
            };
        }
        VkShaderStageFlags(flags)
    }

    pub fn contains(&self, other: VkShaderStageFlags) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for VkShaderStageFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        VkShaderStageFlags(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for VkShaderStageFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Push constant range for Vulkan.
#[derive(Debug, Clone)]
pub struct VkPushConstantRange {
    /// Stage flags.
    pub stages: VkShaderStageFlags,
    /// Offset in bytes.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
}

/// Complete Vulkan pipeline layout description.
#[derive(Debug, Clone)]
pub struct VulkanPipelineLayout {
    /// Descriptor set layouts.
    pub set_layouts: Vec<DescriptorSetLayout>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<VkPushConstantRange>,
}

/// Metal argument buffer description.
#[derive(Debug, Clone)]
pub struct MetalArgumentBuffer {
    /// Buffer index.
    pub index: u32,
    /// Arguments in buffer.
    pub arguments: Vec<MetalArgument>,
}

/// Metal argument description.
#[derive(Debug, Clone)]
pub struct MetalArgument {
    /// Argument index within buffer.
    pub index: u32,
    /// Argument type.
    pub argument_type: MetalArgumentType,
    /// Array length (1 = not array).
    pub array_length: u32,
    /// Texture type (for textures).
    pub texture_type: Option<MetalTextureType>,
}

/// Metal argument types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalArgumentType {
    Buffer,
    Texture,
    Sampler,
}

/// Metal texture types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTextureType {
    Texture1D,
    Texture2D,
    Texture2DArray,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    Texture2DMS,
}

impl From<TextureDimension> for MetalTextureType {
    fn from(dim: TextureDimension) -> Self {
        match dim {
            TextureDimension::D1 => MetalTextureType::Texture1D,
            TextureDimension::D2 => MetalTextureType::Texture2D,
            TextureDimension::D2Array => MetalTextureType::Texture2DArray,
            TextureDimension::D3 => MetalTextureType::Texture3D,
            TextureDimension::Cube => MetalTextureType::TextureCube,
            TextureDimension::CubeArray => MetalTextureType::TextureCubeArray,
        }
    }
}

/// Complete Metal pipeline layout.
#[derive(Debug, Clone)]
pub struct MetalPipelineLayout {
    /// Argument buffers (one per group).
    pub argument_buffers: Vec<MetalArgumentBuffer>,
    /// Direct buffer bindings (for push constants).
    pub buffer_bindings: Vec<MetalBufferBinding>,
}

/// Metal direct buffer binding.
#[derive(Debug, Clone)]
pub struct MetalBufferBinding {
    /// Buffer index.
    pub index: u32,
    /// Size in bytes.
    pub size: u32,
    /// Stages.
    pub stages: Vec<ShaderStage>,
}

/// D3D12 root signature parameter.
#[derive(Debug, Clone)]
pub struct D3D12RootParameter {
    /// Parameter type.
    pub parameter_type: D3D12ParameterType,
    /// Shader visibility.
    pub visibility: D3D12ShaderVisibility,
    /// Descriptor table (if table type).
    pub descriptor_table: Option<Vec<D3D12DescriptorRange>>,
    /// Root constants (if constants type).
    pub constants: Option<D3D12RootConstants>,
    /// Root descriptor (if descriptor type).
    pub descriptor: Option<D3D12RootDescriptor>,
}

/// D3D12 root parameter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3D12ParameterType {
    DescriptorTable,
    Constants,
    Cbv,
    Srv,
    Uav,
}

/// D3D12 shader visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3D12ShaderVisibility {
    All,
    Vertex,
    Pixel,
    // Other stages omitted for brevity
}

/// D3D12 descriptor range.
#[derive(Debug, Clone)]
pub struct D3D12DescriptorRange {
    pub range_type: D3D12RangeType,
    pub num_descriptors: u32,
    pub base_shader_register: u32,
    pub register_space: u32,
}

/// D3D12 descriptor range types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3D12RangeType {
    Cbv,
    Srv,
    Uav,
    Sampler,
}

/// D3D12 root constants.
#[derive(Debug, Clone)]
pub struct D3D12RootConstants {
    pub shader_register: u32,
    pub register_space: u32,
    pub num_32bit_values: u32,
}

/// D3D12 root descriptor.
#[derive(Debug, Clone)]
pub struct D3D12RootDescriptor {
    pub shader_register: u32,
    pub register_space: u32,
}

/// Complete D3D12 root signature description.
#[derive(Debug, Clone)]
pub struct D3D12RootSignature {
    /// Root parameters.
    pub parameters: Vec<D3D12RootParameter>,
    /// Static samplers (if any).
    pub static_samplers: Vec<D3D12StaticSampler>,
}

/// D3D12 static sampler.
#[derive(Debug, Clone)]
pub struct D3D12StaticSampler {
    pub shader_register: u32,
    pub register_space: u32,
    pub visibility: D3D12ShaderVisibility,
    // Filter/address modes omitted for brevity
}

// ---------------------------------------------------------------------------
// Reflection Result
// ---------------------------------------------------------------------------

/// Complete shader reflection result.
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// All resource bindings.
    pub bindings: Vec<ReflectedBinding>,
    /// Push constants.
    pub push_constants: Vec<ReflectedPushConstant>,
    /// Specialization constants.
    pub spec_constants: Vec<ReflectedSpecConstant>,
    /// Entry points.
    pub entry_points: Vec<ReflectedEntryPoint>,
    /// Reflection time in microseconds.
    pub reflection_time_us: u64,
}

impl ReflectionResult {
    /// Get bindings for a specific group.
    pub fn bindings_for_group(&self, group: u32) -> impl Iterator<Item = &ReflectedBinding> {
        self.bindings.iter().filter(move |b| b.group == group)
    }

    /// Get all used groups.
    pub fn used_groups(&self) -> Vec<u32> {
        let mut groups: Vec<u32> = self.bindings.iter().map(|b| b.group).collect();
        groups.sort();
        groups.dedup();
        groups
    }

    /// Generate Vulkan pipeline layout description.
    pub fn to_vulkan_layout(&self) -> VulkanPipelineLayout {
        let mut set_layouts = Vec::new();

        for group in self.used_groups() {
            let bindings: Vec<DescriptorBinding> = self
                .bindings_for_group(group)
                .map(|b| {
                    let descriptor_type = match b.kind {
                        BindingKind::UniformBuffer => DescriptorType::UniformBuffer,
                        BindingKind::StorageBuffer => DescriptorType::StorageBuffer,
                        BindingKind::SampledTexture | BindingKind::DepthTexture => {
                            DescriptorType::SampledImage
                        }
                        BindingKind::StorageTexture => DescriptorType::StorageImage,
                        BindingKind::Sampler | BindingKind::ComparisonSampler => {
                            DescriptorType::Sampler
                        }
                    };

                    DescriptorBinding {
                        binding: b.binding,
                        descriptor_type,
                        count: b.array_size,
                        stages: VkShaderStageFlags::from_stages(&b.stages),
                    }
                })
                .collect();

            set_layouts.push(DescriptorSetLayout {
                set: group,
                bindings,
            });
        }

        let push_constant_ranges: Vec<VkPushConstantRange> = self
            .push_constants
            .iter()
            .map(|pc| VkPushConstantRange {
                stages: VkShaderStageFlags::from_stages(&pc.stages),
                offset: pc.offset,
                size: pc.size,
            })
            .collect();

        VulkanPipelineLayout {
            set_layouts,
            push_constant_ranges,
        }
    }

    /// Generate Metal pipeline layout description.
    pub fn to_metal_layout(&self) -> MetalPipelineLayout {
        let mut argument_buffers = Vec::new();

        for group in self.used_groups() {
            let arguments: Vec<MetalArgument> = self
                .bindings_for_group(group)
                .map(|b| {
                    let (argument_type, texture_type) = match &b.layout {
                        TypeLayout::Texture { dimension, .. } => {
                            (MetalArgumentType::Texture, Some(MetalTextureType::from(*dimension)))
                        }
                        TypeLayout::StorageTexture { dimension, .. } => {
                            (MetalArgumentType::Texture, Some(MetalTextureType::from(*dimension)))
                        }
                        TypeLayout::Sampler { .. } => (MetalArgumentType::Sampler, None),
                        _ => (MetalArgumentType::Buffer, None),
                    };

                    MetalArgument {
                        index: b.binding,
                        argument_type,
                        array_length: b.array_size,
                        texture_type,
                    }
                })
                .collect();

            argument_buffers.push(MetalArgumentBuffer {
                index: group,
                arguments,
            });
        }

        let buffer_bindings: Vec<MetalBufferBinding> = self
            .push_constants
            .iter()
            .enumerate()
            .map(|(i, pc)| MetalBufferBinding {
                index: 30 - i as u32, // Metal convention: high buffer indices for push constants
                size: pc.size,
                stages: pc.stages.clone(),
            })
            .collect();

        MetalPipelineLayout {
            argument_buffers,
            buffer_bindings,
        }
    }

    /// Generate D3D12 root signature description.
    pub fn to_d3d12_root_signature(&self) -> D3D12RootSignature {
        let mut parameters = Vec::new();

        // Add push constants as root constants
        for pc in &self.push_constants {
            let visibility = if pc.stages.contains(&ShaderStage::Vertex)
                && pc.stages.contains(&ShaderStage::Fragment)
            {
                D3D12ShaderVisibility::All
            } else if pc.stages.contains(&ShaderStage::Vertex) {
                D3D12ShaderVisibility::Vertex
            } else {
                D3D12ShaderVisibility::Pixel
            };

            parameters.push(D3D12RootParameter {
                parameter_type: D3D12ParameterType::Constants,
                visibility,
                descriptor_table: None,
                constants: Some(D3D12RootConstants {
                    shader_register: 0,
                    register_space: 0,
                    num_32bit_values: pc.size / 4,
                }),
                descriptor: None,
            });
        }

        // Add descriptor tables per group
        for group in self.used_groups() {
            let mut cbv_count = 0u32;
            let mut srv_count = 0u32;
            let mut uav_count = 0u32;
            let mut sampler_count = 0u32;

            for binding in self.bindings_for_group(group) {
                match binding.kind {
                    BindingKind::UniformBuffer => cbv_count += binding.array_size,
                    BindingKind::StorageBuffer if binding.read_only => srv_count += binding.array_size,
                    BindingKind::StorageBuffer => uav_count += binding.array_size,
                    BindingKind::SampledTexture | BindingKind::DepthTexture => {
                        srv_count += binding.array_size
                    }
                    BindingKind::StorageTexture => uav_count += binding.array_size,
                    BindingKind::Sampler | BindingKind::ComparisonSampler => {
                        sampler_count += binding.array_size
                    }
                }
            }

            let mut ranges = Vec::new();

            if cbv_count > 0 {
                ranges.push(D3D12DescriptorRange {
                    range_type: D3D12RangeType::Cbv,
                    num_descriptors: cbv_count,
                    base_shader_register: 0,
                    register_space: group,
                });
            }
            if srv_count > 0 {
                ranges.push(D3D12DescriptorRange {
                    range_type: D3D12RangeType::Srv,
                    num_descriptors: srv_count,
                    base_shader_register: 0,
                    register_space: group,
                });
            }
            if uav_count > 0 {
                ranges.push(D3D12DescriptorRange {
                    range_type: D3D12RangeType::Uav,
                    num_descriptors: uav_count,
                    base_shader_register: 0,
                    register_space: group,
                });
            }
            if sampler_count > 0 {
                ranges.push(D3D12DescriptorRange {
                    range_type: D3D12RangeType::Sampler,
                    num_descriptors: sampler_count,
                    base_shader_register: 0,
                    register_space: group,
                });
            }

            if !ranges.is_empty() {
                parameters.push(D3D12RootParameter {
                    parameter_type: D3D12ParameterType::DescriptorTable,
                    visibility: D3D12ShaderVisibility::All,
                    descriptor_table: Some(ranges),
                    constants: None,
                    descriptor: None,
                });
            }
        }

        D3D12RootSignature {
            parameters,
            static_samplers: Vec::new(),
        }
    }

    /// Apply specialization constant overrides.
    pub fn apply_spec_overrides(&self, overrides: &[SpecConstantOverride]) -> Vec<ReflectedSpecConstant> {
        self.spec_constants
            .iter()
            .map(|sc| {
                if let Some(ov) = overrides.iter().find(|o| o.id == sc.id) {
                    ReflectedSpecConstant {
                        id: sc.id,
                        name: sc.name.clone(),
                        ty: sc.ty,
                        default_value: ov.value.clone(),
                    }
                } else {
                    sc.clone()
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Shader Reflector
// ---------------------------------------------------------------------------

/// Options for shader reflection.
#[derive(Debug, Clone)]
pub struct ReflectionOptions {
    /// Extract member layouts for structs.
    pub extract_member_layouts: bool,
    /// Extract specialization constants.
    pub extract_spec_constants: bool,
    /// Validate bindings.
    pub validate: bool,
}

impl Default for ReflectionOptions {
    fn default() -> Self {
        Self {
            extract_member_layouts: true,
            extract_spec_constants: true,
            validate: true,
        }
    }
}

/// Shader reflection engine.
pub struct ShaderReflector {
    compiler: NagaCompiler,
}

impl Default for ShaderReflector {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderReflector {
    /// Create a new shader reflector.
    pub fn new() -> Self {
        Self {
            compiler: NagaCompiler::new(),
        }
    }

    /// Reflect a WGSL shader.
    pub fn reflect(&self, source: &str, options: &ReflectionOptions) -> Result<ReflectionResult, CompileError> {
        let start = Instant::now();

        // Parse the shader
        let module = self.compiler.parse(source)?;

        // Validate
        let info = if options.validate {
            self.compiler.validate(&module)?
        } else {
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::empty(),
                naga::valid::Capabilities::all(),
            );
            validator.validate(&module).map_err(|e| {
                CompileError::new(format!("Validation failed: {:?}", e), CompileErrorKind::Validation)
            })?
        };

        // Extract bindings
        let bindings = self.extract_bindings(&module, &info, options);

        // Extract push constants
        let push_constants = self.extract_push_constants(&module, options);

        // Extract specialization constants
        let spec_constants = if options.extract_spec_constants {
            self.extract_spec_constants(&module)
        } else {
            Vec::new()
        };

        // Extract entry points
        let entry_points = self.extract_entry_points(&module, &info, options);

        let elapsed = start.elapsed();

        Ok(ReflectionResult {
            bindings,
            push_constants,
            spec_constants,
            entry_points,
            reflection_time_us: elapsed.as_micros() as u64,
        })
    }

    /// Extract resource bindings from the module.
    fn extract_bindings(
        &self,
        module: &naga::Module,
        _info: &naga::valid::ModuleInfo,
        options: &ReflectionOptions,
    ) -> Vec<ReflectedBinding> {
        let mut bindings = Vec::new();

        // Collect stage usage for each variable
        let stages = self.collect_variable_stages(module);

        for (handle, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                let (kind, layout, read_only) = self.analyze_binding(module, var, options);
                let name = var.name.clone().unwrap_or_else(|| format!("_var_{}", handle.index()));
                let var_stages = stages.get(&handle).cloned().unwrap_or_default();

                // Check for array bindings
                let array_size = self.get_array_size(module, var.ty);

                bindings.push(ReflectedBinding {
                    group: binding.group,
                    binding: binding.binding,
                    name,
                    kind,
                    layout,
                    stages: var_stages,
                    array_size,
                    read_only,
                });
            }
        }

        // Sort by group and binding
        bindings.sort_by(|a, b| a.group.cmp(&b.group).then(a.binding.cmp(&b.binding)));

        bindings
    }

    /// Analyze a binding to determine its kind, layout, and access mode.
    fn analyze_binding(
        &self,
        module: &naga::Module,
        var: &naga::GlobalVariable,
        options: &ReflectionOptions,
    ) -> (BindingKind, TypeLayout, bool) {
        let inner = &module.types[var.ty].inner;

        match inner {
            naga::TypeInner::Image { dim, arrayed, class, .. } => {
                let dimension = convert_dimension(*dim, *arrayed);

                match class {
                    naga::ImageClass::Sampled { kind, multi } => {
                        let format = TextureFormat::from_naga_scalar(*kind, 4, *arrayed);
                        (
                            BindingKind::SampledTexture,
                            TypeLayout::Texture {
                                dimension,
                                format,
                                multisampled: *multi,
                                depth: false,
                            },
                            true,
                        )
                    }
                    naga::ImageClass::Depth { multi } => (
                        BindingKind::DepthTexture,
                        TypeLayout::Texture {
                            dimension,
                            format: Some(TextureFormat::Depth32Float),
                            multisampled: *multi,
                            depth: true,
                        },
                        true,
                    ),
                    naga::ImageClass::Storage { format, access } => {
                        let tex_format = TextureFormat::from_naga_storage(*format);
                        let access_mode = if access.contains(naga::StorageAccess::LOAD)
                            && access.contains(naga::StorageAccess::STORE)
                        {
                            StorageTextureAccess::ReadWrite
                        } else if access.contains(naga::StorageAccess::STORE) {
                            StorageTextureAccess::WriteOnly
                        } else {
                            StorageTextureAccess::ReadOnly
                        };
                        let read_only = !access.contains(naga::StorageAccess::STORE);

                        (
                            BindingKind::StorageTexture,
                            TypeLayout::StorageTexture {
                                dimension,
                                format: tex_format,
                                access: access_mode,
                            },
                            read_only,
                        )
                    }
                }
            }
            naga::TypeInner::Sampler { comparison } => (
                if *comparison {
                    BindingKind::ComparisonSampler
                } else {
                    BindingKind::Sampler
                },
                TypeLayout::Sampler { comparison: *comparison },
                true,
            ),
            _ => {
                // Buffer type
                let (kind, read_only) = match var.space {
                    naga::AddressSpace::Uniform => (BindingKind::UniformBuffer, true),
                    naga::AddressSpace::Storage { access } => {
                        let read_only = !access.contains(naga::StorageAccess::STORE);
                        (BindingKind::StorageBuffer, read_only)
                    }
                    _ => (BindingKind::UniformBuffer, true),
                };

                let layout = if options.extract_member_layouts {
                    self.extract_type_layout(module, var.ty)
                } else {
                    TypeLayout::Opaque {
                        name: module.types[var.ty].name.clone().unwrap_or_default(),
                    }
                };

                (kind, layout, read_only)
            }
        }
    }

    /// Extract detailed type layout.
    fn extract_type_layout(&self, module: &naga::Module, ty: naga::Handle<naga::Type>) -> TypeLayout {
        let naga_type = &module.types[ty];
        let inner = &naga_type.inner;

        match inner {
            naga::TypeInner::Scalar(scalar) => {
                TypeLayout::Scalar(convert_scalar_type(scalar))
            }
            naga::TypeInner::Vector { scalar, size } => TypeLayout::Vector {
                scalar: convert_scalar_type(scalar),
                size: convert_vector_size(*size),
            },
            naga::TypeInner::Matrix { columns, rows, scalar } => TypeLayout::Matrix {
                scalar: convert_scalar_type(scalar),
                columns: *columns as u8,
                rows: *rows as u8,
            },
            naga::TypeInner::Struct { members, span } => {
                let struct_members: Vec<StructMember> = members
                    .iter()
                    .map(|m| {
                        let member_layout = self.extract_type_layout(module, m.ty);
                        StructMember {
                            name: m.name.clone().unwrap_or_default(),
                            ty: member_layout.clone(),
                            offset: m.offset,
                            size: member_layout.size(),
                        }
                    })
                    .collect();

                let alignment = struct_members
                    .iter()
                    .map(|m| m.ty.alignment())
                    .max()
                    .unwrap_or(4);

                TypeLayout::Struct {
                    name: naga_type.name.clone().unwrap_or_else(|| "anon_struct".to_string()),
                    members: struct_members,
                    size: *span,
                    alignment,
                }
            }
            naga::TypeInner::Array { base, size, stride } => {
                let count = match size {
                    naga::ArraySize::Constant(n) => Some(n.get()),
                    naga::ArraySize::Dynamic | naga::ArraySize::Pending(_) => None,
                };

                TypeLayout::Array {
                    element: Box::new(self.extract_type_layout(module, *base)),
                    count,
                    stride: *stride,
                }
            }
            naga::TypeInner::BindingArray { base, size } => {
                let count = match size {
                    naga::ArraySize::Constant(n) => Some(n.get()),
                    naga::ArraySize::Dynamic | naga::ArraySize::Pending(_) => None,
                };

                TypeLayout::Array {
                    element: Box::new(self.extract_type_layout(module, *base)),
                    count,
                    stride: 0, // Binding arrays don't have a stride in the same sense
                }
            }
            naga::TypeInner::Image { dim, arrayed, class, .. } => {
                let dimension = convert_dimension(*dim, *arrayed);
                match class {
                    naga::ImageClass::Sampled { kind, multi } => TypeLayout::Texture {
                        dimension,
                        format: TextureFormat::from_naga_scalar(*kind, 4, *arrayed),
                        multisampled: *multi,
                        depth: false,
                    },
                    naga::ImageClass::Depth { multi } => TypeLayout::Texture {
                        dimension,
                        format: Some(TextureFormat::Depth32Float),
                        multisampled: *multi,
                        depth: true,
                    },
                    naga::ImageClass::Storage { format, access } => {
                        let access_mode = if access.contains(naga::StorageAccess::LOAD)
                            && access.contains(naga::StorageAccess::STORE)
                        {
                            StorageTextureAccess::ReadWrite
                        } else if access.contains(naga::StorageAccess::STORE) {
                            StorageTextureAccess::WriteOnly
                        } else {
                            StorageTextureAccess::ReadOnly
                        };
                        TypeLayout::StorageTexture {
                            dimension,
                            format: TextureFormat::from_naga_storage(*format),
                            access: access_mode,
                        }
                    }
                }
            }
            naga::TypeInner::Sampler { comparison } => {
                TypeLayout::Sampler { comparison: *comparison }
            }
            _ => TypeLayout::Opaque {
                name: naga_type.name.clone().unwrap_or_else(|| "unknown".to_string()),
            },
        }
    }

    /// Get array size from type.
    fn get_array_size(&self, module: &naga::Module, ty: naga::Handle<naga::Type>) -> u32 {
        match &module.types[ty].inner {
            naga::TypeInner::BindingArray { size, .. } => match size {
                naga::ArraySize::Constant(n) => n.get(),
                _ => 1,
            },
            _ => 1,
        }
    }

    /// Collect which stages use which variables.
    fn collect_variable_stages(
        &self,
        module: &naga::Module,
    ) -> HashMap<naga::Handle<naga::GlobalVariable>, Vec<ShaderStage>> {
        let mut result: HashMap<naga::Handle<naga::GlobalVariable>, Vec<ShaderStage>> =
            HashMap::new();

        for ep in &module.entry_points {
            let stage = ShaderStage::from(ep.stage);

            // Conservatively mark all bindings as used by all entry points
            // A more precise analysis would trace actual usage through the call graph
            for (handle, _) in module.global_variables.iter() {
                result.entry(handle).or_default().push(stage);
            }
        }

        // Remove duplicates
        for stages in result.values_mut() {
            stages.sort_by_key(|s| *s as u8);
            stages.dedup();
        }

        result
    }

    /// Extract push constants from the module.
    fn extract_push_constants(
        &self,
        module: &naga::Module,
        options: &ReflectionOptions,
    ) -> Vec<ReflectedPushConstant> {
        let mut push_constants = Vec::new();
        let stages = self.collect_variable_stages(module);

        for (handle, var) in module.global_variables.iter() {
            if var.space == naga::AddressSpace::PushConstant {
                let layout = if options.extract_member_layouts {
                    self.extract_type_layout(module, var.ty)
                } else {
                    TypeLayout::Opaque {
                        name: "push_constants".to_string(),
                    }
                };

                let size = layout.size();
                let var_stages = stages.get(&handle).cloned().unwrap_or_default();

                push_constants.push(ReflectedPushConstant {
                    offset: 0,
                    size,
                    stages: var_stages,
                    layout,
                });
            }
        }

        push_constants
    }

    /// Extract specialization constants (override constants in WGSL).
    fn extract_spec_constants(&self, module: &naga::Module) -> Vec<ReflectedSpecConstant> {
        let mut spec_constants = Vec::new();

        for (handle, constant) in module.overrides.iter() {
            let ty_inner = &module.types[constant.ty].inner;
            let (scalar_type, default_value) = match ty_inner {
                naga::TypeInner::Scalar(scalar) => {
                    let st = convert_scalar_type(scalar);
                    let default = constant.init.map(|init_handle| {
                        self.extract_const_value(module, init_handle, st)
                    }).unwrap_or_else(|| match st {
                        ScalarType::Bool => SpecConstantValue::Bool(false),
                        ScalarType::Int32 => SpecConstantValue::Int32(0),
                        ScalarType::Uint32 => SpecConstantValue::Uint32(0),
                        ScalarType::Float32 | ScalarType::Float16 | ScalarType::Float64 => {
                            SpecConstantValue::Float32(0.0)
                        }
                    });
                    (st, default)
                }
                _ => continue, // Only scalar spec constants supported
            };

            spec_constants.push(ReflectedSpecConstant {
                id: constant.id.map(|id| id as u32).unwrap_or(handle.index() as u32),
                name: constant.name.clone().unwrap_or_default(),
                ty: scalar_type,
                default_value,
            });
        }

        spec_constants
    }

    /// Extract constant value.
    fn extract_const_value(
        &self,
        module: &naga::Module,
        handle: naga::Handle<naga::Expression>,
        _expected_type: ScalarType,
    ) -> SpecConstantValue {
        // Walk constant expressions - this is simplified
        // In practice, would need to evaluate the expression
        let _ = (module, handle);
        SpecConstantValue::Float32(0.0) // Placeholder
    }

    /// Extract entry point information.
    fn extract_entry_points(
        &self,
        module: &naga::Module,
        _info: &naga::valid::ModuleInfo,
        options: &ReflectionOptions,
    ) -> Vec<ReflectedEntryPoint> {
        module
            .entry_points
            .iter()
            .map(|ep| {
                let workgroup_size = if ep.stage == naga::ShaderStage::Compute {
                    Some(ep.workgroup_size)
                } else {
                    None
                };

                let inputs = self.extract_vertex_attributes(module, &ep.function, options);
                let outputs = self.extract_output_attributes(module, &ep.function, options);

                // Collect used bindings (simplified - marks all as used)
                let used_bindings: Vec<(u32, u32)> = module
                    .global_variables
                    .iter()
                    .filter_map(|(_, var)| var.binding.as_ref().map(|b| (b.group, b.binding)))
                    .collect();

                ReflectedEntryPoint {
                    name: ep.name.clone(),
                    stage: ShaderStage::from(ep.stage),
                    workgroup_size,
                    inputs,
                    outputs,
                    used_bindings,
                }
            })
            .collect()
    }

    /// Extract vertex attributes from function arguments.
    fn extract_vertex_attributes(
        &self,
        module: &naga::Module,
        function: &naga::Function,
        options: &ReflectionOptions,
    ) -> Vec<VertexAttribute> {
        function
            .arguments
            .iter()
            .filter_map(|arg| {
                let binding = arg.binding.as_ref()?;
                let (location, builtin) = match binding {
                    naga::Binding::Location { location, .. } => (Some(*location), None),
                    naga::Binding::BuiltIn(b) => (None, Some(convert_builtin(*b))),
                };

                let layout = if options.extract_member_layouts {
                    self.extract_type_layout(module, arg.ty)
                } else {
                    TypeLayout::Opaque {
                        name: "vertex_input".to_string(),
                    }
                };

                // Only return location-based inputs or builtins
                if location.is_some() || builtin.is_some() {
                    Some(VertexAttribute {
                        location: location.unwrap_or(0),
                        name: arg.name.clone().unwrap_or_default(),
                        layout,
                        builtin,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Extract output attributes from function result.
    fn extract_output_attributes(
        &self,
        module: &naga::Module,
        function: &naga::Function,
        options: &ReflectionOptions,
    ) -> Vec<OutputAttribute> {
        let Some(result) = &function.result else {
            return Vec::new();
        };

        let (location, builtin) = match &result.binding {
            Some(naga::Binding::Location { location, .. }) => (Some(*location), None),
            Some(naga::Binding::BuiltIn(b)) => (None, Some(convert_builtin(*b))),
            None => (None, None),
        };

        let layout = if options.extract_member_layouts {
            self.extract_type_layout(module, result.ty)
        } else {
            TypeLayout::Opaque {
                name: "output".to_string(),
            }
        };

        vec![OutputAttribute {
            location,
            name: String::new(),
            layout,
            builtin,
        }]
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Round up to alignment.
fn round_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Convert naga scalar to our scalar type.
fn convert_scalar_type(scalar: &naga::Scalar) -> ScalarType {
    match (scalar.kind, scalar.width) {
        (naga::ScalarKind::Bool, _) => ScalarType::Bool,
        (naga::ScalarKind::Sint, 4) => ScalarType::Int32,
        (naga::ScalarKind::Uint, 4) => ScalarType::Uint32,
        (naga::ScalarKind::Float, 2) => ScalarType::Float16,
        (naga::ScalarKind::Float, 4) => ScalarType::Float32,
        (naga::ScalarKind::Float, 8) => ScalarType::Float64,
        _ => ScalarType::Float32, // Default fallback
    }
}

/// Convert naga vector size.
fn convert_vector_size(size: naga::VectorSize) -> VectorSize {
    match size {
        naga::VectorSize::Bi => VectorSize::Vec2,
        naga::VectorSize::Tri => VectorSize::Vec3,
        naga::VectorSize::Quad => VectorSize::Vec4,
    }
}

/// Convert naga image dimension.
fn convert_dimension(dim: naga::ImageDimension, arrayed: bool) -> TextureDimension {
    match (dim, arrayed) {
        (naga::ImageDimension::D1, _) => TextureDimension::D1,
        (naga::ImageDimension::D2, false) => TextureDimension::D2,
        (naga::ImageDimension::D2, true) => TextureDimension::D2Array,
        (naga::ImageDimension::D3, _) => TextureDimension::D3,
        (naga::ImageDimension::Cube, false) => TextureDimension::Cube,
        (naga::ImageDimension::Cube, true) => TextureDimension::CubeArray,
    }
}

/// Convert naga builtin.
fn convert_builtin(builtin: naga::BuiltIn) -> BuiltinType {
    match builtin {
        naga::BuiltIn::Position { .. } => BuiltinType::Position,
        naga::BuiltIn::VertexIndex => BuiltinType::VertexIndex,
        naga::BuiltIn::InstanceIndex => BuiltinType::InstanceIndex,
        naga::BuiltIn::FrontFacing => BuiltinType::FrontFacing,
        naga::BuiltIn::FragDepth => BuiltinType::FragDepth,
        naga::BuiltIn::LocalInvocationId => BuiltinType::LocalInvocationId,
        naga::BuiltIn::LocalInvocationIndex => BuiltinType::LocalInvocationIndex,
        naga::BuiltIn::GlobalInvocationId => BuiltinType::GlobalInvocationId,
        naga::BuiltIn::WorkGroupId => BuiltinType::WorkgroupId,
        naga::BuiltIn::NumWorkGroups => BuiltinType::NumWorkgroups,
        naga::BuiltIn::SampleIndex => BuiltinType::SampleIndex,
        naga::BuiltIn::SampleMask => BuiltinType::SampleMask,
        _ => BuiltinType::Position, // Fallback
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // Uniform Buffer Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_uniform_buffer_basic() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Uniforms {
                model: mat4x4<f32>,
                view: mat4x4<f32>,
                projection: mat4x4<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        assert_eq!(result.bindings.len(), 1);
        let binding = &result.bindings[0];
        assert_eq!(binding.group, 0);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.kind, BindingKind::UniformBuffer);
        assert!(binding.read_only);

        // Check member layout
        if let TypeLayout::Struct { members, .. } = &binding.layout {
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "model");
            assert_eq!(members[1].name, "view");
            assert_eq!(members[2].name, "projection");
        } else {
            panic!("Expected struct layout");
        }
    }

    #[test]
    fn test_reflect_uniform_buffer_member_offsets() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct TestUniforms {
                a: f32,
                b: vec2<f32>,
                c: vec3<f32>,
                d: vec4<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: TestUniforms;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(uniforms.a);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        if let TypeLayout::Struct { members, .. } = &binding.layout {
            // Check offsets follow WGSL alignment rules
            assert_eq!(members[0].offset, 0);  // f32 at 0
            assert_eq!(members[1].offset, 8);  // vec2 at 8 (aligned to 8)
            assert_eq!(members[2].offset, 16); // vec3 at 16 (aligned to 16)
            assert_eq!(members[3].offset, 32); // vec4 at 32 (aligned to 16)
        } else {
            panic!("Expected struct layout");
        }
    }

    // -----------------------------------------------------------------------
    // Storage Buffer Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_storage_buffer_read_only() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Data {
                values: array<f32>,
            }

            @group(0) @binding(0) var<storage, read> data: Data;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let _ = data.values[id.x];
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        assert_eq!(binding.kind, BindingKind::StorageBuffer);
        assert!(binding.read_only);
    }

    #[test]
    fn test_reflect_storage_buffer_read_write() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        assert_eq!(binding.kind, BindingKind::StorageBuffer);
        assert!(!binding.read_only);
    }

    #[test]
    fn test_reflect_storage_buffer_write_only() {
        let reflector = ShaderReflector::new();
        // Note: WGSL storage buffers don't have write-only mode, but we test read_write
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = vec4<f32>(1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        assert_eq!(result.bindings[0].kind, BindingKind::StorageBuffer);
    }

    // -----------------------------------------------------------------------
    // Sampled Texture Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_sampled_texture_2d() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(tex, samp, uv);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let tex_binding = result.bindings.iter().find(|b| b.name == "tex").unwrap();
        assert_eq!(tex_binding.kind, BindingKind::SampledTexture);

        if let TypeLayout::Texture { dimension, multisampled, depth, .. } = &tex_binding.layout {
            assert_eq!(*dimension, TextureDimension::D2);
            assert!(!multisampled);
            assert!(!depth);
        } else {
            panic!("Expected texture layout");
        }
    }

    #[test]
    fn test_reflect_sampled_texture_cube() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var env_map: texture_cube<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return textureSample(env_map, samp, vec3<f32>(1.0, 0.0, 0.0));
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let tex_binding = result.bindings.iter().find(|b| b.name == "env_map").unwrap();
        if let TypeLayout::Texture { dimension, .. } = &tex_binding.layout {
            assert_eq!(*dimension, TextureDimension::Cube);
        } else {
            panic!("Expected texture layout");
        }
    }

    #[test]
    fn test_reflect_sampled_texture_3d() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var vol_tex: texture_3d<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return textureSample(vol_tex, samp, vec3<f32>(0.5));
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let tex_binding = result.bindings.iter().find(|b| b.name == "vol_tex").unwrap();
        if let TypeLayout::Texture { dimension, .. } = &tex_binding.layout {
            assert_eq!(*dimension, TextureDimension::D3);
        } else {
            panic!("Expected texture layout");
        }
    }

    #[test]
    fn test_reflect_multisampled_texture() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var ms_tex: texture_multisampled_2d<f32>;

            @fragment
            fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                return textureLoad(ms_tex, vec2<i32>(pos.xy), 0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let tex_binding = &result.bindings[0];
        if let TypeLayout::Texture { multisampled, .. } = &tex_binding.layout {
            assert!(*multisampled);
        } else {
            panic!("Expected texture layout");
        }
    }

    // -----------------------------------------------------------------------
    // Storage Texture Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_storage_texture_write() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                textureStore(output, vec2<i32>(id.xy), vec4<f32>(1.0));
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        assert_eq!(binding.kind, BindingKind::StorageTexture);
        assert!(!binding.read_only);

        if let TypeLayout::StorageTexture { format, access, .. } = &binding.layout {
            assert_eq!(*format, TextureFormat::Rgba8Unorm);
            assert_eq!(*access, StorageTextureAccess::WriteOnly);
        } else {
            panic!("Expected storage texture layout");
        }
    }

    #[test]
    fn test_reflect_storage_texture_formats() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var output_r32f: texture_storage_2d<r32float, write>;
            @group(0) @binding(1) var output_rgba16f: texture_storage_2d<rgba16float, write>;
            @group(0) @binding(2) var output_rgba32u: texture_storage_2d<rgba32uint, write>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                textureStore(output_r32f, vec2<i32>(id.xy), vec4<f32>(1.0));
                textureStore(output_rgba16f, vec2<i32>(id.xy), vec4<f32>(1.0));
                textureStore(output_rgba32u, vec2<i32>(id.xy), vec4<u32>(1u));
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        assert_eq!(result.bindings.len(), 3);

        let r32f = result.bindings.iter().find(|b| b.name == "output_r32f").unwrap();
        if let TypeLayout::StorageTexture { format, .. } = &r32f.layout {
            assert_eq!(*format, TextureFormat::R32Float);
        }

        let rgba16f = result.bindings.iter().find(|b| b.name == "output_rgba16f").unwrap();
        if let TypeLayout::StorageTexture { format, .. } = &rgba16f.layout {
            assert_eq!(*format, TextureFormat::Rgba16Float);
        }
    }

    // -----------------------------------------------------------------------
    // Sampler Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_sampler_regular() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var samp: sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        assert_eq!(binding.kind, BindingKind::Sampler);

        if let TypeLayout::Sampler { comparison } = &binding.layout {
            assert!(!comparison);
        } else {
            panic!("Expected sampler layout");
        }
    }

    #[test]
    fn test_reflect_sampler_comparison() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var depth_tex: texture_depth_2d;
            @group(0) @binding(1) var shadow_sampler: sampler_comparison;

            @fragment
            fn main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
                return textureSampleCompare(depth_tex, shadow_sampler, pos.xy, 0.5);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let sampler = result.bindings.iter().find(|b| b.name == "shadow_sampler").unwrap();
        assert_eq!(sampler.kind, BindingKind::ComparisonSampler);

        if let TypeLayout::Sampler { comparison } = &sampler.layout {
            assert!(*comparison);
        } else {
            panic!("Expected sampler layout");
        }
    }

    // -----------------------------------------------------------------------
    // Push Constant Reflection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_push_constants_basic() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct PushConstants {
                color: vec4<f32>,
                offset: vec2<f32>,
            }

            var<push_constant> pc: PushConstants;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return pc.color;
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        assert!(!result.push_constants.is_empty());
        let pc = &result.push_constants[0];
        assert!(pc.size > 0);

        if let TypeLayout::Struct { members, .. } = &pc.layout {
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "color");
            assert_eq!(members[1].name, "offset");
        } else {
            panic!("Expected struct layout for push constants");
        }
    }

    #[test]
    fn test_reflect_push_constants_stages() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct PushConstants {
                mvp: mat4x4<f32>,
            }

            var<push_constant> pc: PushConstants;

            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return pc.mvp * vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let pc = &result.push_constants[0];
        // Push constants should be visible to stages that use them
        assert!(!pc.stages.is_empty());
    }

    // -----------------------------------------------------------------------
    // Specialization Constant Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_specialization_constants() {
        let reflector = ShaderReflector::new();
        let source = r#"
            override workgroup_size: u32 = 64;
            override use_feature: bool = false;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        assert!(!result.spec_constants.is_empty());

        let wg_size = result.spec_constants.iter()
            .find(|sc| sc.name == "workgroup_size");
        assert!(wg_size.is_some());

        let use_feature = result.spec_constants.iter()
            .find(|sc| sc.name == "use_feature");
        assert!(use_feature.is_some());
    }

    #[test]
    fn test_spec_constant_override() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @id(0) override tile_size: u32 = 16;
            @id(1) override max_iterations: u32 = 100;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let overrides = vec![
            SpecConstantOverride {
                id: 0,
                value: SpecConstantValue::Uint32(32),
            },
            SpecConstantOverride {
                id: 1,
                value: SpecConstantValue::Uint32(200),
            },
        ];

        let overridden = result.apply_spec_overrides(&overrides);

        let tile_size = overridden.iter().find(|sc| sc.id == 0).unwrap();
        assert_eq!(tile_size.default_value, SpecConstantValue::Uint32(32));

        let max_iter = overridden.iter().find(|sc| sc.id == 1).unwrap();
        assert_eq!(max_iter.default_value, SpecConstantValue::Uint32(200));
    }

    // -----------------------------------------------------------------------
    // Array Binding Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_fixed_array_binding() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Light {
                position: vec3<f32>,
                intensity: f32,
            }

            @group(0) @binding(0) var<uniform> lights: array<Light, 16>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(lights[0].intensity);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        if let TypeLayout::Array { count, .. } = &binding.layout {
            assert_eq!(*count, Some(16));
        } else {
            panic!("Expected array layout");
        }
    }

    #[test]
    fn test_reflect_runtime_array_binding() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<storage, read> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let _ = data[id.x];
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        if let TypeLayout::Array { count, .. } = &binding.layout {
            assert_eq!(*count, None); // Runtime-sized
        } else {
            panic!("Expected array layout");
        }
    }

    // -----------------------------------------------------------------------
    // Member Layout Extraction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_nested_struct_layout() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Inner {
                value: f32,
                flag: u32,
            }

            struct Outer {
                inner: Inner,
                scale: vec3<f32>,
            }

            @group(0) @binding(0) var<uniform> data: Outer;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(data.inner.value);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        if let TypeLayout::Struct { members, .. } = &binding.layout {
            assert_eq!(members.len(), 2);

            // Check nested struct
            if let TypeLayout::Struct { members: inner_members, .. } = &members[0].ty {
                assert_eq!(inner_members.len(), 2);
                assert_eq!(inner_members[0].name, "value");
                assert_eq!(inner_members[1].name, "flag");
            } else {
                panic!("Expected nested struct");
            }
        } else {
            panic!("Expected struct layout");
        }
    }

    #[test]
    fn test_extract_matrix_layout() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Transforms {
                model: mat4x4<f32>,
                normal: mat3x3<f32>,
            }

            @group(0) @binding(0) var<uniform> transforms: Transforms;

            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return transforms.model * vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let binding = &result.bindings[0];
        if let TypeLayout::Struct { members, .. } = &binding.layout {
            // Check mat4x4
            if let TypeLayout::Matrix { columns, rows, scalar } = &members[0].ty {
                assert_eq!(*columns, 4);
                assert_eq!(*rows, 4);
                assert_eq!(*scalar, ScalarType::Float32);
            } else {
                panic!("Expected matrix layout for model");
            }

            // Check mat3x3
            if let TypeLayout::Matrix { columns, rows, .. } = &members[1].ty {
                assert_eq!(*columns, 3);
                assert_eq!(*rows, 3);
            } else {
                panic!("Expected matrix layout for normal");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pipeline Layout Generation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_vulkan_pipeline_layout() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var samp: sampler;
            @group(1) @binding(0) var<storage, read> data: array<f32>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        let vk_layout = result.to_vulkan_layout();

        assert_eq!(vk_layout.set_layouts.len(), 2);

        // Set 0 should have 3 bindings
        let set0 = &vk_layout.set_layouts[0];
        assert_eq!(set0.set, 0);
        assert_eq!(set0.bindings.len(), 3);

        // Set 1 should have 1 binding
        let set1 = &vk_layout.set_layouts[1];
        assert_eq!(set1.set, 1);
        assert_eq!(set1.bindings.len(), 1);
    }

    #[test]
    fn test_generate_vulkan_push_constant_ranges() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct PushConstants {
                mvp: mat4x4<f32>,
            }

            var<push_constant> pc: PushConstants;

            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return pc.mvp * vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        let vk_layout = result.to_vulkan_layout();

        assert!(!vk_layout.push_constant_ranges.is_empty());
        let range = &vk_layout.push_constant_ranges[0];
        assert!(range.size >= 64); // mat4x4 = 64 bytes
    }

    #[test]
    fn test_generate_metal_layout() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
            @group(0) @binding(1) var tex: texture_2d<f32>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        let mtl_layout = result.to_metal_layout();

        assert!(!mtl_layout.argument_buffers.is_empty());
        let arg_buffer = &mtl_layout.argument_buffers[0];
        assert_eq!(arg_buffer.index, 0);
        assert_eq!(arg_buffer.arguments.len(), 2);
    }

    #[test]
    fn test_generate_d3d12_root_signature() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> uniforms: vec4<f32>;
            @group(0) @binding(1) var<storage, read> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        let d3d_sig = result.to_d3d12_root_signature();

        assert!(!d3d_sig.parameters.is_empty());
    }

    // -----------------------------------------------------------------------
    // Descriptor Set Layout Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_descriptor_set_binding_types() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> ubo: vec4<f32>;
            @group(0) @binding(1) var<storage, read> ssbo_ro: array<f32>;
            @group(0) @binding(2) var<storage, read_write> ssbo_rw: array<f32>;
            @group(0) @binding(3) var tex: texture_2d<f32>;
            @group(0) @binding(4) var storage_tex: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(5) var samp: sampler;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        let vk_layout = result.to_vulkan_layout();

        let set0 = &vk_layout.set_layouts[0];
        assert_eq!(set0.bindings.len(), 6);

        assert_eq!(set0.bindings[0].descriptor_type, DescriptorType::UniformBuffer);
        assert_eq!(set0.bindings[1].descriptor_type, DescriptorType::StorageBuffer);
        assert_eq!(set0.bindings[2].descriptor_type, DescriptorType::StorageBuffer);
        assert_eq!(set0.bindings[3].descriptor_type, DescriptorType::SampledImage);
        assert_eq!(set0.bindings[4].descriptor_type, DescriptorType::StorageImage);
        assert_eq!(set0.bindings[5].descriptor_type, DescriptorType::Sampler);
    }

    // -----------------------------------------------------------------------
    // Performance Benchmark Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflection_performance_simple() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        // Warm up
        let _ = reflector.reflect(source, &ReflectionOptions::default());

        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
            assert!(result.reflection_time_us < 10_000); // <10ms per entry point
        }

        let elapsed = start.elapsed();
        let per_iteration = elapsed / iterations;

        // Should be well under 10ms
        assert!(
            per_iteration.as_millis() < 10,
            "Reflection too slow: {:?} per iteration",
            per_iteration
        );

        println!("Simple reflection: {:?} per iteration", per_iteration);
    }

    #[test]
    fn test_reflection_performance_complex() {
        let reflector = ShaderReflector::new();
        let source = r#"
            struct Uniforms {
                model: mat4x4<f32>,
                view: mat4x4<f32>,
                proj: mat4x4<f32>,
                light_positions: array<vec4<f32>, 8>,
                light_colors: array<vec4<f32>, 8>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var albedo_tex: texture_2d<f32>;
            @group(0) @binding(2) var normal_tex: texture_2d<f32>;
            @group(0) @binding(3) var roughness_tex: texture_2d<f32>;
            @group(0) @binding(4) var samp: sampler;
            @group(1) @binding(0) var<storage, read> vertex_data: array<vec4<f32>>;
            @group(1) @binding(1) var<storage, read_write> output_data: array<vec4<f32>>;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) tangent: vec3<f32>,
            }

            @vertex
            fn vs_main(
                @location(0) position: vec3<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) tangent: vec3<f32>,
                @location(3) uv: vec2<f32>,
            ) -> VertexOutput {
                var out: VertexOutput;
                let world_pos = uniforms.model * vec4<f32>(position, 1.0);
                out.position = uniforms.proj * uniforms.view * world_pos;
                out.uv = uv;
                out.normal = (uniforms.model * vec4<f32>(normal, 0.0)).xyz;
                out.tangent = (uniforms.model * vec4<f32>(tangent, 0.0)).xyz;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let albedo = textureSample(albedo_tex, samp, in.uv);
                let normal = textureSample(normal_tex, samp, in.uv).xyz * 2.0 - 1.0;
                let roughness = textureSample(roughness_tex, samp, in.uv).r;
                return albedo * roughness;
            }
        "#;

        // Warm up
        let _ = reflector.reflect(source, &ReflectionOptions::default());

        let start = Instant::now();
        let iterations = 50;

        for _ in 0..iterations {
            let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
            // <10ms per entry point, we have 2 entry points
            assert!(result.reflection_time_us < 20_000);
        }

        let elapsed = start.elapsed();
        let per_iteration = elapsed / iterations;

        // Should be under 10ms per entry point (20ms total for 2 entry points)
        assert!(
            per_iteration.as_millis() < 20,
            "Complex reflection too slow: {:?} per iteration",
            per_iteration
        );

        println!("Complex reflection: {:?} per iteration", per_iteration);
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflect_empty_shader() {
        let reflector = ShaderReflector::new();
        let source = "";

        // Empty shader should either fail or return empty results
        let result = reflector.reflect(source, &ReflectionOptions::default());
        if let Ok(result) = result {
            assert!(result.bindings.is_empty());
            assert!(result.entry_points.is_empty());
        }
    }

    #[test]
    fn test_reflect_no_bindings() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();
        assert!(result.bindings.is_empty());
        assert_eq!(result.entry_points.len(), 1);
    }

    #[test]
    fn test_reflect_multiple_groups() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(1) @binding(0) var<uniform> b: f32;
            @group(2) @binding(0) var<uniform> c: f32;
            @group(3) @binding(0) var<uniform> d: f32;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(a + b + c + d);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let groups = result.used_groups();
        assert_eq!(groups.len(), 4);
        assert_eq!(groups, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_reflect_depth_texture() {
        let reflector = ShaderReflector::new();
        let source = r#"
            @group(0) @binding(0) var depth_tex: texture_depth_2d;
            @group(0) @binding(1) var shadow_sampler: sampler_comparison;

            @fragment
            fn main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
                return textureSampleCompare(depth_tex, shadow_sampler, pos.xy, 0.5);
            }
        "#;

        let result = reflector.reflect(source, &ReflectionOptions::default()).unwrap();

        let depth = result.bindings.iter().find(|b| b.name == "depth_tex").unwrap();
        assert_eq!(depth.kind, BindingKind::DepthTexture);

        if let TypeLayout::Texture { depth, .. } = &depth.layout {
            assert!(*depth);
        } else {
            panic!("Expected texture layout");
        }
    }

    #[test]
    fn test_type_layout_size_calculations() {
        let scalar = TypeLayout::Scalar(ScalarType::Float32);
        assert_eq!(scalar.size(), 4);
        assert_eq!(scalar.alignment(), 4);

        let vec4 = TypeLayout::Vector {
            scalar: ScalarType::Float32,
            size: VectorSize::Vec4,
        };
        assert_eq!(vec4.size(), 16);
        assert_eq!(vec4.alignment(), 16);

        let mat4 = TypeLayout::Matrix {
            scalar: ScalarType::Float32,
            columns: 4,
            rows: 4,
        };
        assert_eq!(mat4.size(), 64);
    }

    #[test]
    fn test_shader_stage_flags() {
        let flags = VkShaderStageFlags::from_stages(&[ShaderStage::Vertex, ShaderStage::Fragment]);
        assert!(flags.contains(VkShaderStageFlags::VERTEX));
        assert!(flags.contains(VkShaderStageFlags::FRAGMENT));
        assert!(!flags.contains(VkShaderStageFlags::COMPUTE));

        let combined = VkShaderStageFlags::VERTEX | VkShaderStageFlags::FRAGMENT;
        assert!(combined.contains(VkShaderStageFlags::VERTEX));
    }

    #[test]
    fn test_spec_constant_value_to_bytes() {
        let bool_val = SpecConstantValue::Bool(true);
        assert_eq!(bool_val.to_bytes(), [1, 0, 0, 0]);

        let int_val = SpecConstantValue::Int32(-1);
        assert_eq!(int_val.to_bytes(), [255, 255, 255, 255]);

        let uint_val = SpecConstantValue::Uint32(42);
        assert_eq!(uint_val.to_bytes(), [42, 0, 0, 0]);

        let float_val = SpecConstantValue::Float32(1.0);
        assert_eq!(float_val.to_bytes(), 1.0f32.to_le_bytes());
    }
}
