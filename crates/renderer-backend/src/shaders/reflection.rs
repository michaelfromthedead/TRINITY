//! Shader reflection from Naga IR for automatic resource binding discovery.
//!
//! This module provides shader reflection capabilities that extract binding information,
//! entry points, push constants, and resource types from compiled Naga modules. This
//! enables automatic pipeline layout generation and binding validation.
//!
//! # Overview
//!
//! The shader reflection system provides:
//!
//! - **Entry point enumeration**: List all entry points with stage, name, workgroup size
//! - **Binding extraction**: Extract group, binding, type for all resource bindings
//! - **Resource type detection**: Identify buffers, textures, samplers, storage resources
//! - **Push constant reflection**: Extract push constant ranges and types
//! - **Automatic layout generation**: Generate wgpu BindGroupLayout from reflection
//!
//! # Architecture
//!
//! ```text
//! ShaderReflection
//! +-- entry_points: Vec<EntryPointInfo>
//! +-- bindings: Vec<BindingInfo>
//! +-- push_constants: Option<PushConstantInfo>
//! +-- from_module(module, info) -> Result<Self, ReflectionError>
//! +-- entry_points() -> &[EntryPointInfo]
//! +-- bindings() -> &[BindingInfo]
//! +-- bindings_for_group(group) -> Vec<&BindingInfo>
//! +-- push_constants() -> Option<&PushConstantInfo>
//! +-- generate_bind_group_layout(device, group) -> Result<BindGroupLayout>
//! +-- generate_pipeline_layout(device) -> Result<PipelineLayout>
//!
//! EntryPointInfo
//! +-- name: String
//! +-- stage: ShaderStage
//! +-- workgroup_size: Option<[u32; 3]>
//!
//! BindingInfo
//! +-- group: u32
//! +-- binding: u32
//! +-- name: Option<String>
//! +-- resource_type: ResourceType
//! +-- access: ResourceAccess
//! +-- count: Option<NonZeroU32> (for binding arrays)
//!
//! ResourceType
//! +-- UniformBuffer { size }
//! +-- StorageBuffer { size, read_only }
//! +-- Texture { dimension, sample_type, multisampled }
//! +-- StorageTexture { dimension, format, access }
//! +-- Sampler { filtering, comparison }
//! +-- AccelerationStructure
//!
//! PushConstantInfo
//! +-- stages: ShaderStages
//! +-- size: u32
//! +-- members: Vec<PushConstantMember>
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::shaders::reflection::{
//!     ShaderReflection, ReflectionError,
//! };
//!
//! # fn example() -> Result<(), ReflectionError> {
//! let source = r#"
//!     @group(0) @binding(0) var<uniform> camera: CameraData;
//!     @group(0) @binding(1) var tex: texture_2d<f32>;
//!     @group(0) @binding(2) var samp: sampler;
//!
//!     struct CameraData { view_proj: mat4x4<f32> }
//!
//!     @vertex
//!     fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
//!         return vec4<f32>(0.0);
//!     }
//!
//!     @fragment
//!     fn fs_main() -> @location(0) vec4<f32> {
//!         return textureSample(tex, samp, vec2<f32>(0.0));
//!     }
//! "#;
//!
//! // Parse and validate with naga
//! let module = naga::front::wgsl::parse_str(source).unwrap();
//! let mut validator = naga::valid::Validator::new(
//!     naga::valid::ValidationFlags::all(),
//!     naga::valid::Capabilities::all(),
//! );
//! let info = validator.validate(&module).unwrap();
//!
//! // Create reflection data
//! let reflection = ShaderReflection::from_module(&module, &info)?;
//!
//! // Query entry points
//! for ep in reflection.entry_points() {
//!     println!("{}: {:?}", ep.name, ep.stage);
//! }
//!
//! // Query bindings
//! for binding in reflection.bindings_for_group(0) {
//!     println!("@group(0) @binding({}) - {:?}", binding.binding, binding.resource_type);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU32;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of bind groups supported by wgpu.
pub const MAX_BIND_GROUPS: u32 = 4;

/// Maximum push constant size in bytes (wgpu limit).
pub const MAX_PUSH_CONSTANT_SIZE: u32 = 128;

// ============================================================================
// Shader Stage
// ============================================================================

/// Shader stage enumeration compatible with wgpu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shader stage.
    Vertex,
    /// Fragment shader stage.
    Fragment,
    /// Compute shader stage.
    Compute,
}

impl ShaderStage {
    /// Converts from naga::ShaderStage.
    pub fn from_naga(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
        }
    }

    /// Converts to wgpu::ShaderStages.
    pub fn to_wgpu(self) -> wgpu::ShaderStages {
        match self {
            ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderStage::Vertex => write!(f, "vertex"),
            ShaderStage::Fragment => write!(f, "fragment"),
            ShaderStage::Compute => write!(f, "compute"),
        }
    }
}

// ============================================================================
// Resource Access
// ============================================================================

/// Access mode for storage resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResourceAccess {
    /// Read-only access.
    #[default]
    Read,
    /// Write-only access.
    Write,
    /// Read-write access.
    ReadWrite,
}

impl ResourceAccess {
    /// Converts from naga storage access flags.
    pub fn from_naga(access: naga::StorageAccess) -> Self {
        let readable = access.contains(naga::StorageAccess::LOAD);
        let writable = access.contains(naga::StorageAccess::STORE);
        match (readable, writable) {
            (true, true) => ResourceAccess::ReadWrite,
            (true, false) => ResourceAccess::Read,
            (false, true) => ResourceAccess::Write,
            (false, false) => ResourceAccess::Read, // Default to read if unknown
        }
    }

    /// Converts to wgpu storage texture access.
    pub fn to_wgpu_storage_access(self) -> wgpu::StorageTextureAccess {
        match self {
            ResourceAccess::Read => wgpu::StorageTextureAccess::ReadOnly,
            ResourceAccess::Write => wgpu::StorageTextureAccess::WriteOnly,
            ResourceAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
        }
    }

    /// Returns true if readable.
    pub fn is_readable(self) -> bool {
        matches!(self, ResourceAccess::Read | ResourceAccess::ReadWrite)
    }

    /// Returns true if writable.
    pub fn is_writable(self) -> bool {
        matches!(self, ResourceAccess::Write | ResourceAccess::ReadWrite)
    }
}

impl fmt::Display for ResourceAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceAccess::Read => write!(f, "read"),
            ResourceAccess::Write => write!(f, "write"),
            ResourceAccess::ReadWrite => write!(f, "read_write"),
        }
    }
}

// ============================================================================
// Texture Dimension
// ============================================================================

/// Texture dimension for reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    /// 1D texture.
    D1,
    /// 2D texture.
    D2,
    /// 2D array texture.
    D2Array,
    /// 3D texture.
    D3,
    /// Cube map texture.
    Cube,
    /// Cube map array texture.
    CubeArray,
}

impl TextureDimension {
    /// Converts from naga::ImageDimension.
    pub fn from_naga(dim: naga::ImageDimension, arrayed: bool) -> Self {
        match (dim, arrayed) {
            (naga::ImageDimension::D1, _) => TextureDimension::D1,
            (naga::ImageDimension::D2, false) => TextureDimension::D2,
            (naga::ImageDimension::D2, true) => TextureDimension::D2Array,
            (naga::ImageDimension::D3, _) => TextureDimension::D3,
            (naga::ImageDimension::Cube, false) => TextureDimension::Cube,
            (naga::ImageDimension::Cube, true) => TextureDimension::CubeArray,
        }
    }

    /// Converts to wgpu::TextureViewDimension.
    pub fn to_wgpu(self) -> wgpu::TextureViewDimension {
        match self {
            TextureDimension::D1 => wgpu::TextureViewDimension::D1,
            TextureDimension::D2 => wgpu::TextureViewDimension::D2,
            TextureDimension::D2Array => wgpu::TextureViewDimension::D2Array,
            TextureDimension::D3 => wgpu::TextureViewDimension::D3,
            TextureDimension::Cube => wgpu::TextureViewDimension::Cube,
            TextureDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        }
    }
}

impl fmt::Display for TextureDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureDimension::D1 => write!(f, "1d"),
            TextureDimension::D2 => write!(f, "2d"),
            TextureDimension::D2Array => write!(f, "2d_array"),
            TextureDimension::D3 => write!(f, "3d"),
            TextureDimension::Cube => write!(f, "cube"),
            TextureDimension::CubeArray => write!(f, "cube_array"),
        }
    }
}

// ============================================================================
// Texture Sample Type
// ============================================================================

/// Sample type for textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    /// Float texture (filterable).
    Float { filterable: bool },
    /// Signed integer texture.
    Sint,
    /// Unsigned integer texture.
    Uint,
    /// Depth texture.
    Depth,
}

impl TextureSampleType {
    /// Converts from naga::ImageClass.
    pub fn from_naga_class(class: &naga::ImageClass) -> Option<Self> {
        match class {
            naga::ImageClass::Sampled { kind, multi: _ } => match kind {
                naga::ScalarKind::Float => Some(TextureSampleType::Float { filterable: true }),
                naga::ScalarKind::Sint => Some(TextureSampleType::Sint),
                naga::ScalarKind::Uint => Some(TextureSampleType::Uint),
                naga::ScalarKind::Bool => None,
                naga::ScalarKind::AbstractInt => None,
                naga::ScalarKind::AbstractFloat => None,
            },
            naga::ImageClass::Depth { multi: _ } => Some(TextureSampleType::Depth),
            naga::ImageClass::Storage { .. } => None, // Storage textures use different type
        }
    }

    /// Converts to wgpu::TextureSampleType.
    pub fn to_wgpu(self) -> wgpu::TextureSampleType {
        match self {
            TextureSampleType::Float { filterable } => {
                wgpu::TextureSampleType::Float { filterable }
            }
            TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
            TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
            TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
        }
    }
}

impl fmt::Display for TextureSampleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureSampleType::Float { filterable: true } => write!(f, "f32"),
            TextureSampleType::Float { filterable: false } => write!(f, "f32 (unfilterable)"),
            TextureSampleType::Sint => write!(f, "i32"),
            TextureSampleType::Uint => write!(f, "u32"),
            TextureSampleType::Depth => write!(f, "depth"),
        }
    }
}

// ============================================================================
// Sampler Type
// ============================================================================

/// Sampler binding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SamplerType {
    /// Regular filtering sampler.
    #[default]
    Filtering,
    /// Non-filtering sampler (for integer textures).
    NonFiltering,
    /// Comparison sampler (for shadow mapping).
    Comparison,
}

impl SamplerType {
    /// Converts to wgpu::SamplerBindingType.
    pub fn to_wgpu(self) -> wgpu::SamplerBindingType {
        match self {
            SamplerType::Filtering => wgpu::SamplerBindingType::Filtering,
            SamplerType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            SamplerType::Comparison => wgpu::SamplerBindingType::Comparison,
        }
    }
}

impl fmt::Display for SamplerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerType::Filtering => write!(f, "filtering"),
            SamplerType::NonFiltering => write!(f, "non_filtering"),
            SamplerType::Comparison => write!(f, "comparison"),
        }
    }
}

// ============================================================================
// Resource Type
// ============================================================================

/// Type of resource bound at a binding slot.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// Uniform buffer.
    UniformBuffer {
        /// Size in bytes (if known).
        size: Option<u64>,
        /// Whether buffer has dynamic offset.
        has_dynamic_offset: bool,
    },
    /// Storage buffer.
    StorageBuffer {
        /// Size in bytes (if known).
        size: Option<u64>,
        /// Whether the buffer is read-only.
        read_only: bool,
        /// Whether buffer has dynamic offset.
        has_dynamic_offset: bool,
    },
    /// Sampled texture.
    Texture {
        /// Texture dimension.
        dimension: TextureDimension,
        /// Sample type.
        sample_type: TextureSampleType,
        /// Whether multisampled.
        multisampled: bool,
    },
    /// Storage texture.
    StorageTexture {
        /// Texture dimension.
        dimension: TextureDimension,
        /// Texture format.
        format: wgpu::TextureFormat,
        /// Access mode.
        access: ResourceAccess,
    },
    /// Sampler.
    Sampler {
        /// Sampler type.
        sampler_type: SamplerType,
    },
    /// Acceleration structure (for ray tracing).
    AccelerationStructure,
}

impl ResourceType {
    /// Returns true if this is a buffer type.
    pub fn is_buffer(&self) -> bool {
        matches!(
            self,
            ResourceType::UniformBuffer { .. } | ResourceType::StorageBuffer { .. }
        )
    }

    /// Returns true if this is a texture type.
    pub fn is_texture(&self) -> bool {
        matches!(
            self,
            ResourceType::Texture { .. } | ResourceType::StorageTexture { .. }
        )
    }

    /// Returns true if this is a sampler.
    pub fn is_sampler(&self) -> bool {
        matches!(self, ResourceType::Sampler { .. })
    }

    /// Returns true if this resource has read access.
    pub fn has_read_access(&self) -> bool {
        match self {
            ResourceType::UniformBuffer { .. } => true,
            ResourceType::StorageBuffer { read_only, .. } => *read_only,
            ResourceType::Texture { .. } => true,
            ResourceType::StorageTexture { access, .. } => access.is_readable(),
            ResourceType::Sampler { .. } => true,
            ResourceType::AccelerationStructure => true,
        }
    }

    /// Returns true if this resource has write access.
    pub fn has_write_access(&self) -> bool {
        match self {
            ResourceType::UniformBuffer { .. } => false,
            ResourceType::StorageBuffer { read_only, .. } => !*read_only,
            ResourceType::Texture { .. } => false,
            ResourceType::StorageTexture { access, .. } => access.is_writable(),
            ResourceType::Sampler { .. } => false,
            ResourceType::AccelerationStructure => false,
        }
    }

    /// Converts to wgpu::BindingType.
    pub fn to_wgpu_binding_type(&self) -> wgpu::BindingType {
        match self {
            ResourceType::UniformBuffer {
                size,
                has_dynamic_offset,
            } => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: *has_dynamic_offset,
                min_binding_size: size.and_then(NonZeroU64::new),
            },
            ResourceType::StorageBuffer {
                size,
                read_only,
                has_dynamic_offset,
            } => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *read_only,
                },
                has_dynamic_offset: *has_dynamic_offset,
                min_binding_size: size.and_then(NonZeroU64::new),
            },
            ResourceType::Texture {
                dimension,
                sample_type,
                multisampled,
            } => wgpu::BindingType::Texture {
                sample_type: sample_type.to_wgpu(),
                view_dimension: dimension.to_wgpu(),
                multisampled: *multisampled,
            },
            ResourceType::StorageTexture {
                dimension,
                format,
                access,
            } => wgpu::BindingType::StorageTexture {
                access: access.to_wgpu_storage_access(),
                format: *format,
                view_dimension: dimension.to_wgpu(),
            },
            ResourceType::Sampler { sampler_type } => {
                wgpu::BindingType::Sampler(sampler_type.to_wgpu())
            }
            ResourceType::AccelerationStructure => {
                // Note: wgpu doesn't have a direct acceleration structure binding type
                // in the stable API. This would need feature-gating for ray tracing.
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            }
        }
    }
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::UniformBuffer { size, .. } => {
                if let Some(s) = size {
                    write!(f, "uniform buffer ({} bytes)", s)
                } else {
                    write!(f, "uniform buffer")
                }
            }
            ResourceType::StorageBuffer {
                size, read_only, ..
            } => {
                let access = if *read_only { "read" } else { "read_write" };
                if let Some(s) = size {
                    write!(f, "storage buffer<{}> ({} bytes)", access, s)
                } else {
                    write!(f, "storage buffer<{}>", access)
                }
            }
            ResourceType::Texture {
                dimension,
                sample_type,
                multisampled,
            } => {
                let ms = if *multisampled { "_multisampled" } else { "" };
                write!(f, "texture{}<{}>_{}", ms, sample_type, dimension)
            }
            ResourceType::StorageTexture {
                dimension,
                format,
                access,
            } => {
                write!(f, "texture_storage_{}<{:?}, {}>", dimension, format, access)
            }
            ResourceType::Sampler { sampler_type } => {
                write!(f, "sampler ({})", sampler_type)
            }
            ResourceType::AccelerationStructure => {
                write!(f, "acceleration_structure")
            }
        }
    }
}

use std::num::NonZeroU64;

// ============================================================================
// Entry Point Info
// ============================================================================

/// Information about a shader entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPointInfo {
    /// Entry point function name.
    pub name: String,
    /// Shader stage.
    pub stage: ShaderStage,
    /// Workgroup size for compute shaders (x, y, z).
    pub workgroup_size: Option<[u32; 3]>,
    /// Index of this entry point in the module.
    pub index: usize,
}

impl EntryPointInfo {
    /// Creates a new entry point info.
    pub fn new(
        name: impl Into<String>,
        stage: ShaderStage,
        workgroup_size: Option<[u32; 3]>,
        index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            stage,
            workgroup_size,
            index,
        }
    }

    /// Returns true if this is a vertex shader.
    pub fn is_vertex(&self) -> bool {
        self.stage == ShaderStage::Vertex
    }

    /// Returns true if this is a fragment shader.
    pub fn is_fragment(&self) -> bool {
        self.stage == ShaderStage::Fragment
    }

    /// Returns true if this is a compute shader.
    pub fn is_compute(&self) -> bool {
        self.stage == ShaderStage::Compute
    }

    /// Returns the total workgroup size (x * y * z).
    pub fn workgroup_total(&self) -> Option<u32> {
        self.workgroup_size
            .map(|[x, y, z]| x.max(1) * y.max(1) * z.max(1))
    }
}

impl fmt::Display for EntryPointInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{} fn {}", self.stage, self.name)?;
        if let Some([x, y, z]) = self.workgroup_size {
            write!(f, " @workgroup_size({}, {}, {})", x, y, z)?;
        }
        Ok(())
    }
}

// ============================================================================
// Binding Info
// ============================================================================

/// Information about a shader binding.
#[derive(Debug, Clone, PartialEq)]
pub struct BindingInfo {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Optional variable name from the shader.
    pub name: Option<String>,
    /// Resource type.
    pub resource_type: ResourceType,
    /// Shader stages that use this binding.
    pub visibility: wgpu::ShaderStages,
    /// Binding array count (None for non-arrays).
    pub count: Option<NonZeroU32>,
}

impl BindingInfo {
    /// Creates a new binding info.
    pub fn new(
        group: u32,
        binding: u32,
        name: Option<String>,
        resource_type: ResourceType,
        visibility: wgpu::ShaderStages,
    ) -> Self {
        Self {
            group,
            binding,
            name,
            resource_type,
            visibility,
            count: None,
        }
    }

    /// Sets the binding array count.
    pub fn with_count(mut self, count: NonZeroU32) -> Self {
        self.count = Some(count);
        self
    }

    /// Converts to wgpu::BindGroupLayoutEntry.
    pub fn to_wgpu_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: self.visibility,
            ty: self.resource_type.to_wgpu_binding_type(),
            count: self.count,
        }
    }
}

impl fmt::Display for BindingInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@group({}) @binding({})", self.group, self.binding)?;
        if let Some(name) = &self.name {
            write!(f, " var {}: {}", name, self.resource_type)?;
        } else {
            write!(f, " {}", self.resource_type)?;
        }
        if let Some(count) = self.count {
            write!(f, "[{}]", count)?;
        }
        Ok(())
    }
}

// ============================================================================
// Push Constant Member
// ============================================================================

/// A member within a push constant block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantMember {
    /// Member name.
    pub name: String,
    /// Offset in bytes from start of push constant block.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Type name (for debugging).
    pub type_name: String,
}

impl PushConstantMember {
    /// Creates a new push constant member.
    pub fn new(
        name: impl Into<String>,
        offset: u32,
        size: u32,
        type_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            offset,
            size,
            type_name: type_name.into(),
        }
    }
}

impl fmt::Display for PushConstantMember {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} @ offset {} ({} bytes)",
            self.name, self.type_name, self.offset, self.size
        )
    }
}

// ============================================================================
// Push Constant Info
// ============================================================================

/// Information about push constants in a shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantInfo {
    /// Shader stages that use this push constant block.
    pub stages: wgpu::ShaderStages,
    /// Total size in bytes.
    pub size: u32,
    /// Individual members (if struct type).
    pub members: Vec<PushConstantMember>,
}

impl PushConstantInfo {
    /// Creates a new push constant info.
    pub fn new(stages: wgpu::ShaderStages, size: u32) -> Self {
        Self {
            stages,
            size,
            members: Vec::new(),
        }
    }

    /// Adds a member to the push constant block.
    pub fn with_member(mut self, member: PushConstantMember) -> Self {
        self.members.push(member);
        self
    }

    /// Converts to wgpu::PushConstantRange.
    pub fn to_wgpu_range(&self) -> wgpu::PushConstantRange {
        wgpu::PushConstantRange {
            stages: self.stages,
            range: 0..self.size,
        }
    }

    /// Returns true if the size exceeds the wgpu limit.
    pub fn exceeds_limit(&self) -> bool {
        self.size > MAX_PUSH_CONSTANT_SIZE
    }
}

impl fmt::Display for PushConstantInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "push_constant ({} bytes, stages: {:?})", self.size, self.stages)?;
        if !self.members.is_empty() {
            writeln!(f)?;
            for member in &self.members {
                writeln!(f, "  {}", member)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Reflection Error
// ============================================================================

/// Errors that can occur during shader reflection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectionError {
    /// No entry points found in the module.
    NoEntryPoints,
    /// Unsupported resource type encountered.
    UnsupportedResourceType {
        /// Description of the unsupported type.
        description: String,
    },
    /// Invalid binding configuration.
    InvalidBinding {
        /// Error message.
        message: String,
        /// Binding group.
        group: u32,
        /// Binding index.
        binding: u32,
    },
    /// Invalid push constant configuration.
    InvalidPushConstants {
        /// Error message.
        message: String,
    },
    /// Bind group index exceeds maximum.
    GroupIndexTooLarge {
        /// The problematic group index.
        group: u32,
        /// Maximum allowed.
        max: u32,
    },
    /// Layout generation failed.
    LayoutGenerationFailed {
        /// Error message.
        message: String,
    },
}

impl fmt::Display for ReflectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReflectionError::NoEntryPoints => {
                write!(f, "shader module has no entry points")
            }
            ReflectionError::UnsupportedResourceType { description } => {
                write!(f, "unsupported resource type: {}", description)
            }
            ReflectionError::InvalidBinding {
                message,
                group,
                binding,
            } => {
                write!(
                    f,
                    "invalid binding at @group({}) @binding({}): {}",
                    group, binding, message
                )
            }
            ReflectionError::InvalidPushConstants { message } => {
                write!(f, "invalid push constants: {}", message)
            }
            ReflectionError::GroupIndexTooLarge { group, max } => {
                write!(
                    f,
                    "bind group index {} exceeds maximum {} (wgpu limit)",
                    group, max
                )
            }
            ReflectionError::LayoutGenerationFailed { message } => {
                write!(f, "layout generation failed: {}", message)
            }
        }
    }
}

impl std::error::Error for ReflectionError {}

// ============================================================================
// Shader Reflection
// ============================================================================

/// Complete reflection data extracted from a shader module.
///
/// Contains all entry points, bindings, and push constants discovered
/// by analyzing the Naga IR.
#[derive(Debug, Clone)]
pub struct ShaderReflection {
    /// All entry points in the module.
    entry_points: Vec<EntryPointInfo>,
    /// All resource bindings.
    bindings: Vec<BindingInfo>,
    /// Push constant info (if any).
    push_constants: Option<PushConstantInfo>,
    /// Cached bind group count (highest group index + 1).
    bind_group_count: u32,
}

impl ShaderReflection {
    /// Creates shader reflection data from a validated Naga module.
    ///
    /// # Arguments
    ///
    /// * `module` - The parsed naga module.
    /// * `info` - Validation info from naga::valid::Validator.
    ///
    /// # Returns
    ///
    /// Returns reflection data or an error if reflection fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::shaders::reflection::ShaderReflection;
    /// let module = naga::front::wgsl::parse_str("@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }").unwrap();
    /// let mut validator = naga::valid::Validator::new(
    ///     naga::valid::ValidationFlags::all(),
    ///     naga::valid::Capabilities::all(),
    /// );
    /// let info = validator.validate(&module).unwrap();
    /// let reflection = ShaderReflection::from_module(&module, &info).unwrap();
    /// ```
    pub fn from_module(
        module: &naga::Module,
        _info: &naga::valid::ModuleInfo,
    ) -> Result<Self, ReflectionError> {
        // Extract entry points
        let entry_points = Self::extract_entry_points(module)?;

        // Build stage visibility map for bindings
        let visibility_map = Self::build_visibility_map(module);

        // Extract bindings from global variables
        let bindings = Self::extract_bindings(module, &visibility_map)?;

        // Extract push constants
        let push_constants = Self::extract_push_constants(module, &entry_points)?;

        // Calculate bind group count
        let bind_group_count = bindings
            .iter()
            .map(|b| b.group + 1)
            .max()
            .unwrap_or(0);

        Ok(Self {
            entry_points,
            bindings,
            push_constants,
            bind_group_count,
        })
    }

    /// Extracts entry point information from the module.
    fn extract_entry_points(module: &naga::Module) -> Result<Vec<EntryPointInfo>, ReflectionError> {
        if module.entry_points.is_empty() {
            return Err(ReflectionError::NoEntryPoints);
        }

        let entry_points = module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, ep)| {
                let stage = ShaderStage::from_naga(ep.stage);
                let workgroup_size = if ep.stage == naga::ShaderStage::Compute {
                    Some(ep.workgroup_size)
                } else {
                    None
                };
                EntryPointInfo::new(&ep.name, stage, workgroup_size, index)
            })
            .collect();

        Ok(entry_points)
    }

    /// Builds a map from global variable handles to shader stages that use them.
    fn build_visibility_map(module: &naga::Module) -> HashMap<naga::Handle<naga::GlobalVariable>, wgpu::ShaderStages> {
        let mut visibility_map = HashMap::new();

        // Analyze each entry point to determine which globals it uses
        for ep in &module.entry_points {
            let stage = match ep.stage {
                naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
                naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
                naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
            };

            // For simplicity, assume all bindings are visible to all stages that exist
            // A more accurate approach would analyze the function body
            for (handle, _gvar) in module.global_variables.iter() {
                *visibility_map.entry(handle).or_insert(wgpu::ShaderStages::empty()) |= stage;
            }
        }

        visibility_map
    }

    /// Extracts binding information from global variables.
    fn extract_bindings(
        module: &naga::Module,
        visibility_map: &HashMap<naga::Handle<naga::GlobalVariable>, wgpu::ShaderStages>,
    ) -> Result<Vec<BindingInfo>, ReflectionError> {
        let mut bindings = Vec::new();

        for (handle, gvar) in module.global_variables.iter() {
            // Skip variables without bindings
            let binding = match &gvar.binding {
                Some(b) => b,
                None => continue,
            };

            // Check group index limit
            if binding.group >= MAX_BIND_GROUPS {
                return Err(ReflectionError::GroupIndexTooLarge {
                    group: binding.group,
                    max: MAX_BIND_GROUPS - 1,
                });
            }

            // Get type information
            let ty = &module.types[gvar.ty];
            let resource_type = Self::extract_resource_type(module, gvar, ty)?;

            // Get visibility from map or default to all stages
            let visibility = visibility_map
                .get(&handle)
                .copied()
                .unwrap_or(wgpu::ShaderStages::all());

            // Get variable name
            let name = gvar.name.clone();

            bindings.push(BindingInfo::new(
                binding.group,
                binding.binding,
                name,
                resource_type,
                visibility,
            ));
        }

        // Sort bindings by group then binding index
        bindings.sort_by(|a, b| {
            a.group.cmp(&b.group).then(a.binding.cmp(&b.binding))
        });

        Ok(bindings)
    }

    /// Extracts the resource type from a global variable.
    fn extract_resource_type(
        module: &naga::Module,
        gvar: &naga::GlobalVariable,
        ty: &naga::Type,
    ) -> Result<ResourceType, ReflectionError> {
        match gvar.space {
            naga::AddressSpace::Uniform => {
                let size = Self::calculate_type_size(module, ty);
                Ok(ResourceType::UniformBuffer {
                    size: Some(size),
                    has_dynamic_offset: false,
                })
            }
            naga::AddressSpace::Storage { access } => {
                let size = Self::calculate_type_size(module, ty);
                let read_only = !access.contains(naga::StorageAccess::STORE);
                Ok(ResourceType::StorageBuffer {
                    size: Some(size),
                    read_only,
                    has_dynamic_offset: false,
                })
            }
            naga::AddressSpace::Handle => {
                Self::extract_handle_resource_type(module, ty)
            }
            naga::AddressSpace::PushConstant => {
                // Push constants are handled separately
                Err(ReflectionError::UnsupportedResourceType {
                    description: "push constants should not have bindings".to_string(),
                })
            }
            _ => {
                Err(ReflectionError::UnsupportedResourceType {
                    description: format!("address space {:?}", gvar.space),
                })
            }
        }
    }

    /// Extracts resource type for handle (texture/sampler) address space.
    fn extract_handle_resource_type(
        module: &naga::Module,
        ty: &naga::Type,
    ) -> Result<ResourceType, ReflectionError> {
        match &ty.inner {
            naga::TypeInner::Image { dim, arrayed, class } => {
                let dimension = TextureDimension::from_naga(*dim, *arrayed);

                match class {
                    naga::ImageClass::Sampled { kind, multi } => {
                        let sample_type = match kind {
                            naga::ScalarKind::Float => TextureSampleType::Float { filterable: true },
                            naga::ScalarKind::Sint => TextureSampleType::Sint,
                            naga::ScalarKind::Uint => TextureSampleType::Uint,
                            _ => {
                                return Err(ReflectionError::UnsupportedResourceType {
                                    description: format!("sampled image with kind {:?}", kind),
                                });
                            }
                        };
                        Ok(ResourceType::Texture {
                            dimension,
                            sample_type,
                            multisampled: *multi,
                        })
                    }
                    naga::ImageClass::Depth { multi } => {
                        Ok(ResourceType::Texture {
                            dimension,
                            sample_type: TextureSampleType::Depth,
                            multisampled: *multi,
                        })
                    }
                    naga::ImageClass::Storage { format, access } => {
                        let wgpu_format = Self::naga_format_to_wgpu(*format);
                        let access = ResourceAccess::from_naga(*access);
                        Ok(ResourceType::StorageTexture {
                            dimension,
                            format: wgpu_format,
                            access,
                        })
                    }
                }
            }
            naga::TypeInner::Sampler { comparison } => {
                let sampler_type = if *comparison {
                    SamplerType::Comparison
                } else {
                    SamplerType::Filtering
                };
                Ok(ResourceType::Sampler { sampler_type })
            }
            naga::TypeInner::AccelerationStructure => {
                Ok(ResourceType::AccelerationStructure)
            }
            naga::TypeInner::BindingArray { base, size } => {
                // Handle binding arrays - extract the base type
                let base_ty = &module.types[*base];
                let mut resource_type = Self::extract_handle_resource_type(module, base_ty)?;
                // Note: binding array count would be added to BindingInfo.count
                let _ = size; // Size info could be used if needed
                Ok(resource_type)
            }
            _ => {
                Err(ReflectionError::UnsupportedResourceType {
                    description: format!("handle type {:?}", ty.inner),
                })
            }
        }
    }

    /// Converts naga storage format to wgpu texture format.
    fn naga_format_to_wgpu(format: naga::StorageFormat) -> wgpu::TextureFormat {
        match format {
            naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            naga::StorageFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
            naga::StorageFormat::R8Uint => wgpu::TextureFormat::R8Uint,
            naga::StorageFormat::R8Sint => wgpu::TextureFormat::R8Sint,
            naga::StorageFormat::R16Uint => wgpu::TextureFormat::R16Uint,
            naga::StorageFormat::R16Sint => wgpu::TextureFormat::R16Sint,
            naga::StorageFormat::R16Float => wgpu::TextureFormat::R16Float,
            naga::StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            naga::StorageFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,
            naga::StorageFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
            naga::StorageFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,
            naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
            naga::StorageFormat::R32Sint => wgpu::TextureFormat::R32Sint,
            naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
            naga::StorageFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
            naga::StorageFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
            naga::StorageFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
            naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            naga::StorageFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
            naga::StorageFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
            naga::StorageFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,
            naga::StorageFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            naga::StorageFormat::Rgb10a2Uint => wgpu::TextureFormat::Rgb10a2Uint,
            naga::StorageFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
            naga::StorageFormat::Rg11b10Ufloat => wgpu::TextureFormat::Rg11b10Float,
            naga::StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
            naga::StorageFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
            naga::StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
            naga::StorageFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
            naga::StorageFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
            naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
            naga::StorageFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
            naga::StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
            naga::StorageFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
            naga::StorageFormat::R16Snorm => wgpu::TextureFormat::R16Snorm,
            naga::StorageFormat::Rg16Unorm => wgpu::TextureFormat::Rg16Unorm,
            naga::StorageFormat::Rg16Snorm => wgpu::TextureFormat::Rg16Snorm,
            naga::StorageFormat::Rgba16Unorm => wgpu::TextureFormat::Rgba16Unorm,
            naga::StorageFormat::Rgba16Snorm => wgpu::TextureFormat::Rgba16Snorm,
            naga::StorageFormat::R64Uint => wgpu::TextureFormat::R32Uint, // Fallback: R64 not in wgpu
        }
    }

    /// Calculates the size of a type in bytes.
    fn calculate_type_size(module: &naga::Module, ty: &naga::Type) -> u64 {
        match &ty.inner {
            naga::TypeInner::Scalar(scalar) => scalar.width as u64,
            naga::TypeInner::Vector { scalar, size } => {
                let components = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                components * scalar.width as u64
            }
            naga::TypeInner::Matrix { columns, rows, scalar } => {
                let col_count = match columns {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let row_count = match rows {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                col_count * row_count * scalar.width as u64
            }
            naga::TypeInner::Array { base, size, stride } => {
                let element_size = stride;
                match size {
                    naga::ArraySize::Constant(count) => {
                        *element_size as u64 * count.get() as u64
                    }
                    naga::ArraySize::Dynamic | naga::ArraySize::Pending(_) => {
                        // Dynamic/pending arrays have unknown size at compile time
                        // Return 0 to indicate dynamic sizing
                        0
                    }
                }
            }
            naga::TypeInner::Struct { members, span } => {
                // Use the struct's span as the total size
                *span as u64
            }
            naga::TypeInner::Atomic(scalar) => scalar.width as u64,
            _ => 0, // Unknown or unsupported type
        }
    }

    /// Extracts push constant information.
    fn extract_push_constants(
        module: &naga::Module,
        entry_points: &[EntryPointInfo],
    ) -> Result<Option<PushConstantInfo>, ReflectionError> {
        let mut push_constant_var: Option<(&naga::GlobalVariable, &naga::Type)> = None;

        // Find push constant global variable
        for (_handle, gvar) in module.global_variables.iter() {
            if gvar.space == naga::AddressSpace::PushConstant {
                let ty = &module.types[gvar.ty];
                push_constant_var = Some((gvar, ty));
                break;
            }
        }

        let (gvar, ty) = match push_constant_var {
            Some(v) => v,
            None => return Ok(None),
        };

        // Calculate size
        let size = Self::calculate_type_size(module, ty) as u32;
        if size > MAX_PUSH_CONSTANT_SIZE {
            return Err(ReflectionError::InvalidPushConstants {
                message: format!(
                    "push constant size {} exceeds maximum {} bytes",
                    size, MAX_PUSH_CONSTANT_SIZE
                ),
            });
        }

        // Determine stages
        let stages = entry_points
            .iter()
            .map(|ep| ep.stage.to_wgpu())
            .fold(wgpu::ShaderStages::empty(), |acc, s| acc | s);

        let mut push_info = PushConstantInfo::new(stages, size);

        // Extract struct members if applicable
        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for member in members {
                let member_ty = &module.types[member.ty];
                let member_size = Self::calculate_type_size(module, member_ty) as u32;
                let type_name = Self::type_name(member_ty);
                push_info.members.push(PushConstantMember::new(
                    member.name.clone().unwrap_or_else(|| "<unnamed>".to_string()),
                    member.offset,
                    member_size,
                    type_name,
                ));
            }
        }

        Ok(Some(push_info))
    }

    /// Returns a human-readable name for a type.
    fn type_name(ty: &naga::Type) -> String {
        if let Some(name) = &ty.name {
            return name.clone();
        }
        match &ty.inner {
            naga::TypeInner::Scalar(s) => {
                let kind = match s.kind {
                    naga::ScalarKind::Bool => "bool",
                    naga::ScalarKind::Sint => "i32",
                    naga::ScalarKind::Uint => "u32",
                    naga::ScalarKind::Float => match s.width {
                        2 => "f16",
                        4 => "f32",
                        8 => "f64",
                        _ => "float",
                    },
                    naga::ScalarKind::AbstractInt => "AbstractInt",
                    naga::ScalarKind::AbstractFloat => "AbstractFloat",
                };
                kind.to_string()
            }
            naga::TypeInner::Vector { scalar, size } => {
                let count = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let kind = match scalar.kind {
                    naga::ScalarKind::Float => "f32",
                    naga::ScalarKind::Sint => "i32",
                    naga::ScalarKind::Uint => "u32",
                    naga::ScalarKind::Bool => "bool",
                    _ => "unknown",
                };
                format!("vec{}<{}>", count, kind)
            }
            naga::TypeInner::Matrix { columns, rows, scalar } => {
                let cols = match columns {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let rs = match rows {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let kind = match scalar.kind {
                    naga::ScalarKind::Float => "f32",
                    _ => "unknown",
                };
                format!("mat{}x{}<{}>", cols, rs, kind)
            }
            naga::TypeInner::Array { .. } => "array".to_string(),
            naga::TypeInner::Struct { .. } => "struct".to_string(),
            _ => "unknown".to_string(),
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Returns all entry points in the shader.
    pub fn entry_points(&self) -> &[EntryPointInfo] {
        &self.entry_points
    }

    /// Returns all bindings in the shader.
    pub fn bindings(&self) -> &[BindingInfo] {
        &self.bindings
    }

    /// Returns bindings for a specific group.
    pub fn bindings_for_group(&self, group: u32) -> Vec<&BindingInfo> {
        self.bindings
            .iter()
            .filter(|b| b.group == group)
            .collect()
    }

    /// Returns push constant info if present.
    pub fn push_constants(&self) -> Option<&PushConstantInfo> {
        self.push_constants.as_ref()
    }

    /// Returns the number of bind groups used.
    pub fn bind_group_count(&self) -> u32 {
        self.bind_group_count
    }

    /// Returns entry points of a specific stage.
    pub fn entry_points_for_stage(&self, stage: ShaderStage) -> Vec<&EntryPointInfo> {
        self.entry_points
            .iter()
            .filter(|ep| ep.stage == stage)
            .collect()
    }

    /// Returns the first vertex entry point.
    pub fn vertex_entry_point(&self) -> Option<&EntryPointInfo> {
        self.entry_points.iter().find(|ep| ep.is_vertex())
    }

    /// Returns the first fragment entry point.
    pub fn fragment_entry_point(&self) -> Option<&EntryPointInfo> {
        self.entry_points.iter().find(|ep| ep.is_fragment())
    }

    /// Returns the first compute entry point.
    pub fn compute_entry_point(&self) -> Option<&EntryPointInfo> {
        self.entry_points.iter().find(|ep| ep.is_compute())
    }

    /// Generates a wgpu BindGroupLayout for the specified group.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `group` - The bind group index.
    /// * `label` - Optional label for the layout.
    ///
    /// # Returns
    ///
    /// Returns the generated layout or an error.
    pub fn generate_bind_group_layout(
        &self,
        device: &wgpu::Device,
        group: u32,
        label: Option<&str>,
    ) -> Result<wgpu::BindGroupLayout, ReflectionError> {
        let bindings = self.bindings_for_group(group);
        if bindings.is_empty() {
            return Err(ReflectionError::LayoutGenerationFailed {
                message: format!("no bindings found for group {}", group),
            });
        }

        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .map(|b| b.to_wgpu_layout_entry())
            .collect();

        Ok(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &entries,
        }))
    }

    /// Generates all bind group layouts for the shader.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `label_prefix` - Optional prefix for layout labels.
    ///
    /// # Returns
    ///
    /// Returns a vec of layouts indexed by group number.
    pub fn generate_all_bind_group_layouts(
        &self,
        device: &wgpu::Device,
        label_prefix: Option<&str>,
    ) -> Result<Vec<wgpu::BindGroupLayout>, ReflectionError> {
        let mut layouts = Vec::new();
        for group in 0..self.bind_group_count {
            let label = label_prefix.map(|p| format!("{}_group_{}", p, group));
            let layout = self.generate_bind_group_layout(
                device,
                group,
                label.as_deref(),
            )?;
            layouts.push(layout);
        }
        Ok(layouts)
    }

    /// Generates a complete pipeline layout from reflection data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `label` - Optional label for the pipeline layout.
    ///
    /// # Returns
    ///
    /// Returns the generated pipeline layout and bind group layouts.
    pub fn generate_pipeline_layout(
        &self,
        device: &wgpu::Device,
        label: Option<&str>,
    ) -> Result<(wgpu::PipelineLayout, Vec<wgpu::BindGroupLayout>), ReflectionError> {
        let bind_group_layouts = self.generate_all_bind_group_layouts(device, label)?;

        let push_constant_ranges: Vec<wgpu::PushConstantRange> = self
            .push_constants
            .as_ref()
            .map(|pc| vec![pc.to_wgpu_range()])
            .unwrap_or_default();

        let layout_refs: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &layout_refs,
            push_constant_ranges: &push_constant_ranges,
        });

        Ok((pipeline_layout, bind_group_layouts))
    }

    /// Validates that the reflection data is compatible with wgpu limits.
    pub fn validate(&self) -> Result<(), ReflectionError> {
        // Check bind group count
        if self.bind_group_count > MAX_BIND_GROUPS {
            return Err(ReflectionError::GroupIndexTooLarge {
                group: self.bind_group_count - 1,
                max: MAX_BIND_GROUPS - 1,
            });
        }

        // Check push constant size
        if let Some(pc) = &self.push_constants {
            if pc.exceeds_limit() {
                return Err(ReflectionError::InvalidPushConstants {
                    message: format!(
                        "push constant size {} exceeds maximum {} bytes",
                        pc.size, MAX_PUSH_CONSTANT_SIZE
                    ),
                });
            }
        }

        Ok(())
    }
}

impl fmt::Display for ShaderReflection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ShaderReflection {{")?;

        writeln!(f, "  Entry Points:")?;
        for ep in &self.entry_points {
            writeln!(f, "    {}", ep)?;
        }

        writeln!(f, "  Bindings ({} groups):", self.bind_group_count)?;
        for binding in &self.bindings {
            writeln!(f, "    {}", binding)?;
        }

        if let Some(pc) = &self.push_constants {
            writeln!(f, "  Push Constants:")?;
            writeln!(f, "    {}", pc)?;
        }

        write!(f, "}}")
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Reflects a shader from WGSL source.
///
/// Parses, validates, and reflects in one call.
pub fn reflect_wgsl(source: &str) -> Result<ShaderReflection, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("parse error: {}", e.message()))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator
        .validate(&module)
        .map_err(|e| format!("validation error: {}", e))?;

    ShaderReflection::from_module(&module, &info)
        .map_err(|e| format!("reflection error: {}", e))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------

    fn parse_and_reflect(source: &str) -> ShaderReflection {
        let module = naga::front::wgsl::parse_str(source).expect("parse failed");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("validation failed");
        ShaderReflection::from_module(&module, &info).expect("reflection failed")
    }

    fn parse_module(source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
        let module = naga::front::wgsl::parse_str(source).expect("parse failed");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).expect("validation failed");
        (module, info)
    }

    // -------------------------------------------------------------------------
    // ShaderStage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_stage_from_naga() {
        assert_eq!(
            ShaderStage::from_naga(naga::ShaderStage::Vertex),
            ShaderStage::Vertex
        );
        assert_eq!(
            ShaderStage::from_naga(naga::ShaderStage::Fragment),
            ShaderStage::Fragment
        );
        assert_eq!(
            ShaderStage::from_naga(naga::ShaderStage::Compute),
            ShaderStage::Compute
        );
    }

    #[test]
    fn test_shader_stage_to_wgpu() {
        assert_eq!(ShaderStage::Vertex.to_wgpu(), wgpu::ShaderStages::VERTEX);
        assert_eq!(ShaderStage::Fragment.to_wgpu(), wgpu::ShaderStages::FRAGMENT);
        assert_eq!(ShaderStage::Compute.to_wgpu(), wgpu::ShaderStages::COMPUTE);
    }

    #[test]
    fn test_shader_stage_display() {
        assert_eq!(format!("{}", ShaderStage::Vertex), "vertex");
        assert_eq!(format!("{}", ShaderStage::Fragment), "fragment");
        assert_eq!(format!("{}", ShaderStage::Compute), "compute");
    }

    // -------------------------------------------------------------------------
    // ResourceAccess Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_access_from_naga() {
        assert_eq!(
            ResourceAccess::from_naga(naga::StorageAccess::LOAD),
            ResourceAccess::Read
        );
        assert_eq!(
            ResourceAccess::from_naga(naga::StorageAccess::STORE),
            ResourceAccess::Write
        );
        assert_eq!(
            ResourceAccess::from_naga(naga::StorageAccess::LOAD | naga::StorageAccess::STORE),
            ResourceAccess::ReadWrite
        );
    }

    #[test]
    fn test_resource_access_predicates() {
        assert!(ResourceAccess::Read.is_readable());
        assert!(!ResourceAccess::Read.is_writable());
        assert!(!ResourceAccess::Write.is_readable());
        assert!(ResourceAccess::Write.is_writable());
        assert!(ResourceAccess::ReadWrite.is_readable());
        assert!(ResourceAccess::ReadWrite.is_writable());
    }

    #[test]
    fn test_resource_access_to_wgpu() {
        assert_eq!(
            ResourceAccess::Read.to_wgpu_storage_access(),
            wgpu::StorageTextureAccess::ReadOnly
        );
        assert_eq!(
            ResourceAccess::Write.to_wgpu_storage_access(),
            wgpu::StorageTextureAccess::WriteOnly
        );
        assert_eq!(
            ResourceAccess::ReadWrite.to_wgpu_storage_access(),
            wgpu::StorageTextureAccess::ReadWrite
        );
    }

    // -------------------------------------------------------------------------
    // TextureDimension Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_dimension_from_naga() {
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D1, false),
            TextureDimension::D1
        );
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D2, false),
            TextureDimension::D2
        );
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D2, true),
            TextureDimension::D2Array
        );
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D3, false),
            TextureDimension::D3
        );
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::Cube, false),
            TextureDimension::Cube
        );
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::Cube, true),
            TextureDimension::CubeArray
        );
    }

    #[test]
    fn test_texture_dimension_to_wgpu() {
        assert_eq!(TextureDimension::D1.to_wgpu(), wgpu::TextureViewDimension::D1);
        assert_eq!(TextureDimension::D2.to_wgpu(), wgpu::TextureViewDimension::D2);
        assert_eq!(TextureDimension::D2Array.to_wgpu(), wgpu::TextureViewDimension::D2Array);
        assert_eq!(TextureDimension::D3.to_wgpu(), wgpu::TextureViewDimension::D3);
        assert_eq!(TextureDimension::Cube.to_wgpu(), wgpu::TextureViewDimension::Cube);
        assert_eq!(TextureDimension::CubeArray.to_wgpu(), wgpu::TextureViewDimension::CubeArray);
    }

    // -------------------------------------------------------------------------
    // TextureSampleType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_sample_type_to_wgpu() {
        assert!(matches!(
            TextureSampleType::Float { filterable: true }.to_wgpu(),
            wgpu::TextureSampleType::Float { filterable: true }
        ));
        assert!(matches!(
            TextureSampleType::Sint.to_wgpu(),
            wgpu::TextureSampleType::Sint
        ));
        assert!(matches!(
            TextureSampleType::Uint.to_wgpu(),
            wgpu::TextureSampleType::Uint
        ));
        assert!(matches!(
            TextureSampleType::Depth.to_wgpu(),
            wgpu::TextureSampleType::Depth
        ));
    }

    // -------------------------------------------------------------------------
    // SamplerType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_type_to_wgpu() {
        assert_eq!(
            SamplerType::Filtering.to_wgpu(),
            wgpu::SamplerBindingType::Filtering
        );
        assert_eq!(
            SamplerType::NonFiltering.to_wgpu(),
            wgpu::SamplerBindingType::NonFiltering
        );
        assert_eq!(
            SamplerType::Comparison.to_wgpu(),
            wgpu::SamplerBindingType::Comparison
        );
    }

    // -------------------------------------------------------------------------
    // ResourceType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_type_predicates() {
        let uniform = ResourceType::UniformBuffer { size: Some(64), has_dynamic_offset: false };
        assert!(uniform.is_buffer());
        assert!(!uniform.is_texture());
        assert!(!uniform.is_sampler());
        assert!(uniform.has_read_access());
        assert!(!uniform.has_write_access());

        let storage = ResourceType::StorageBuffer { size: Some(128), read_only: false, has_dynamic_offset: false };
        assert!(storage.is_buffer());
        assert!(!storage.is_texture());
        assert!(!storage.has_read_access());
        assert!(storage.has_write_access());

        let texture = ResourceType::Texture {
            dimension: TextureDimension::D2,
            sample_type: TextureSampleType::Float { filterable: true },
            multisampled: false,
        };
        assert!(!texture.is_buffer());
        assert!(texture.is_texture());
        assert!(!texture.is_sampler());

        let sampler = ResourceType::Sampler { sampler_type: SamplerType::Filtering };
        assert!(!sampler.is_buffer());
        assert!(!sampler.is_texture());
        assert!(sampler.is_sampler());
    }

    #[test]
    fn test_resource_type_to_wgpu() {
        let uniform = ResourceType::UniformBuffer { size: Some(64), has_dynamic_offset: false };
        let binding_type = uniform.to_wgpu_binding_type();
        assert!(matches!(binding_type, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, .. }));

        let storage = ResourceType::StorageBuffer { size: Some(128), read_only: true, has_dynamic_offset: false };
        let binding_type = storage.to_wgpu_binding_type();
        assert!(matches!(binding_type, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, .. }));
    }

    // -------------------------------------------------------------------------
    // EntryPointInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_entry_point_info_predicates() {
        let vertex = EntryPointInfo::new("vs_main", ShaderStage::Vertex, None, 0);
        assert!(vertex.is_vertex());
        assert!(!vertex.is_fragment());
        assert!(!vertex.is_compute());

        let fragment = EntryPointInfo::new("fs_main", ShaderStage::Fragment, None, 1);
        assert!(!fragment.is_vertex());
        assert!(fragment.is_fragment());
        assert!(!fragment.is_compute());

        let compute = EntryPointInfo::new("cs_main", ShaderStage::Compute, Some([8, 8, 1]), 2);
        assert!(!compute.is_vertex());
        assert!(!compute.is_fragment());
        assert!(compute.is_compute());
    }

    #[test]
    fn test_entry_point_workgroup_total() {
        let ep = EntryPointInfo::new("main", ShaderStage::Compute, Some([8, 8, 1]), 0);
        assert_eq!(ep.workgroup_total(), Some(64));

        let ep = EntryPointInfo::new("main", ShaderStage::Compute, Some([16, 16, 4]), 0);
        assert_eq!(ep.workgroup_total(), Some(1024));

        let ep = EntryPointInfo::new("main", ShaderStage::Vertex, None, 0);
        assert_eq!(ep.workgroup_total(), None);
    }

    #[test]
    fn test_entry_point_display() {
        let ep = EntryPointInfo::new("vs_main", ShaderStage::Vertex, None, 0);
        assert_eq!(format!("{}", ep), "@vertex fn vs_main");

        let ep = EntryPointInfo::new("cs_main", ShaderStage::Compute, Some([64, 1, 1]), 0);
        assert_eq!(format!("{}", ep), "@compute fn cs_main @workgroup_size(64, 1, 1)");
    }

    // -------------------------------------------------------------------------
    // BindingInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binding_info_display() {
        let binding = BindingInfo::new(
            0, 0,
            Some("camera".to_string()),
            ResourceType::UniformBuffer { size: Some(64), has_dynamic_offset: false },
            wgpu::ShaderStages::VERTEX,
        );
        let display = format!("{}", binding);
        assert!(display.contains("@group(0)"));
        assert!(display.contains("@binding(0)"));
        assert!(display.contains("camera"));
    }

    #[test]
    fn test_binding_info_to_wgpu() {
        let binding = BindingInfo::new(
            0, 1,
            None,
            ResourceType::Sampler { sampler_type: SamplerType::Filtering },
            wgpu::ShaderStages::FRAGMENT,
        );
        let entry = binding.to_wgpu_layout_entry();
        assert_eq!(entry.binding, 1);
        assert_eq!(entry.visibility, wgpu::ShaderStages::FRAGMENT);
        assert!(matches!(entry.ty, wgpu::BindingType::Sampler(_)));
    }

    // -------------------------------------------------------------------------
    // PushConstantInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_constant_info() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 64);
        assert_eq!(pc.size, 64);
        assert!(!pc.exceeds_limit());

        let range = pc.to_wgpu_range();
        assert_eq!(range.range, 0..64);
    }

    #[test]
    fn test_push_constant_exceeds_limit() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::VERTEX, 256);
        assert!(pc.exceeds_limit());
    }

    // -------------------------------------------------------------------------
    // ReflectionError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflection_error_display() {
        let err = ReflectionError::NoEntryPoints;
        assert!(format!("{}", err).contains("no entry points"));

        let err = ReflectionError::InvalidBinding {
            message: "duplicate".to_string(),
            group: 0,
            binding: 1,
        };
        assert!(format!("{}", err).contains("@group(0)"));
        assert!(format!("{}", err).contains("@binding(1)"));
    }

    // -------------------------------------------------------------------------
    // Entry Point Extraction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_vertex_shader() {
        let source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.entry_points().len(), 1);
        let ep = &reflection.entry_points()[0];
        assert_eq!(ep.name, "vs_main");
        assert_eq!(ep.stage, ShaderStage::Vertex);
        assert!(ep.workgroup_size.is_none());
    }

    #[test]
    fn test_reflect_fragment_shader() {
        let source = r#"
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.entry_points().len(), 1);
        let ep = &reflection.entry_points()[0];
        assert_eq!(ep.name, "fs_main");
        assert_eq!(ep.stage, ShaderStage::Fragment);
    }

    #[test]
    fn test_reflect_compute_shader() {
        let source = r#"
            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.entry_points().len(), 1);
        let ep = &reflection.entry_points()[0];
        assert_eq!(ep.name, "cs_main");
        assert_eq!(ep.stage, ShaderStage::Compute);
        assert_eq!(ep.workgroup_size, Some([64, 1, 1]));
    }

    #[test]
    fn test_reflect_multiple_entry_points() {
        let source = r#"
            @vertex
            fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.entry_points().len(), 2);
        assert!(reflection.vertex_entry_point().is_some());
        assert!(reflection.fragment_entry_point().is_some());
    }

    #[test]
    fn test_reflect_workgroup_size_3d() {
        let source = r#"
            @compute @workgroup_size(8, 8, 4)
            fn main() {}
        "#;
        let reflection = parse_and_reflect(source);
        let ep = reflection.compute_entry_point().unwrap();
        assert_eq!(ep.workgroup_size, Some([8, 8, 4]));
        assert_eq!(ep.workgroup_total(), Some(256));
    }

    // -------------------------------------------------------------------------
    // Binding Extraction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_uniform_buffer() {
        let source = r#"
            struct CameraData {
                view_proj: mat4x4<f32>,
            }
            @group(0) @binding(0) var<uniform> camera: CameraData;
            @compute @workgroup_size(1) fn main() {
                let _x = camera.view_proj;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let bindings = reflection.bindings_for_group(0);
        assert_eq!(bindings.len(), 1);
        let b = bindings[0];
        assert_eq!(b.group, 0);
        assert_eq!(b.binding, 0);
        assert_eq!(b.name, Some("camera".to_string()));
        assert!(matches!(b.resource_type, ResourceType::UniformBuffer { .. }));
    }

    #[test]
    fn test_reflect_storage_buffer_read_only() {
        let source = r#"
            struct Data { values: array<f32> }
            @group(0) @binding(0) var<storage, read> data: Data;
            @compute @workgroup_size(1) fn main() {
                let _x = data.values[0];
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageBuffer { read_only, .. } => assert!(*read_only),
            _ => panic!("expected storage buffer"),
        }
    }

    #[test]
    fn test_reflect_storage_buffer_read_write() {
        let source = r#"
            struct Data { values: array<f32> }
            @group(0) @binding(0) var<storage, read_write> data: Data;
            @compute @workgroup_size(1) fn main() {
                data.values[0] = 1.0;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageBuffer { read_only, .. } => assert!(!*read_only),
            _ => panic!("expected storage buffer"),
        }
    }

    #[test]
    fn test_reflect_texture_2d() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(tex, vec2<i32>(0, 0), 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { dimension, sample_type, multisampled } => {
                assert_eq!(*dimension, TextureDimension::D2);
                assert!(matches!(sample_type, TextureSampleType::Float { .. }));
                assert!(!multisampled);
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_sampler() {
        let source = r#"
            @group(0) @binding(0) var samp: sampler;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(tex, samp, vec2<f32>(0.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let sampler_binding = reflection.bindings().iter().find(|b| b.binding == 0).unwrap();
        assert!(matches!(
            sampler_binding.resource_type,
            ResourceType::Sampler { sampler_type: SamplerType::Filtering }
        ));
    }

    #[test]
    fn test_reflect_comparison_sampler() {
        let source = r#"
            @group(0) @binding(0) var samp: sampler_comparison;
            @group(0) @binding(1) var tex: texture_depth_2d;
            @fragment fn main() -> @location(0) vec4<f32> {
                let d = textureSampleCompare(tex, samp, vec2<f32>(0.0), 0.5);
                return vec4<f32>(d);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let sampler_binding = reflection.bindings().iter().find(|b| b.binding == 0).unwrap();
        assert!(matches!(
            sampler_binding.resource_type,
            ResourceType::Sampler { sampler_type: SamplerType::Comparison }
        ));
    }

    #[test]
    fn test_reflect_depth_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_depth_2d;
            @fragment fn main() -> @location(0) vec4<f32> {
                let d = textureLoad(tex, vec2<i32>(0, 0), 0);
                return vec4<f32>(d);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { sample_type, .. } => {
                assert!(matches!(sample_type, TextureSampleType::Depth));
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_storage_texture() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { dimension, format, access } => {
                assert_eq!(*dimension, TextureDimension::D2);
                assert_eq!(*format, wgpu::TextureFormat::Rgba8Unorm);
                assert_eq!(*access, ResourceAccess::Write);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_reflect_multiple_groups() {
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(1) @binding(0) var<uniform> b: f32;
            @group(2) @binding(0) var<uniform> c: f32;
            @compute @workgroup_size(1) fn main() {
                let _x = a + b + c;
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.bind_group_count(), 3);
        assert_eq!(reflection.bindings_for_group(0).len(), 1);
        assert_eq!(reflection.bindings_for_group(1).len(), 1);
        assert_eq!(reflection.bindings_for_group(2).len(), 1);
    }

    #[test]
    fn test_reflect_multiple_bindings_per_group() {
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var samp: sampler;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(tex, samp, vec2<f32>(0.0)) * a;
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.bind_group_count(), 1);
        assert_eq!(reflection.bindings_for_group(0).len(), 3);

        let bindings: Vec<_> = reflection.bindings_for_group(0).iter().map(|b| b.binding).collect();
        assert!(bindings.contains(&0));
        assert!(bindings.contains(&1));
        assert!(bindings.contains(&2));
    }

    // -------------------------------------------------------------------------
    // Push Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_push_constants_simple() {
        let source = r#"
            struct PushData {
                value: f32,
            }
            var<push_constant> push: PushData;
            @compute @workgroup_size(1) fn main() {
                let _x = push.value;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let pc = reflection.push_constants().expect("expected push constants");
        assert_eq!(pc.size, 4);
        assert!(pc.stages.contains(wgpu::ShaderStages::COMPUTE));
    }

    #[test]
    fn test_reflect_push_constants_with_members() {
        let source = r#"
            struct PushData {
                offset: vec2<f32>,
                scale: f32,
                pad: f32,
            }
            var<push_constant> push: PushData;
            @compute @workgroup_size(1) fn main() {
                let _x = push.offset;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let pc = reflection.push_constants().expect("expected push constants");
        assert!(!pc.members.is_empty());
        assert!(pc.members.iter().any(|m| m.name == "offset"));
        assert!(pc.members.iter().any(|m| m.name == "scale"));
    }

    #[test]
    fn test_reflect_no_push_constants() {
        let source = r#"
            @compute @workgroup_size(1) fn main() {}
        "#;
        let reflection = parse_and_reflect(source);
        assert!(reflection.push_constants().is_none());
    }

    // -------------------------------------------------------------------------
    // Error Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_no_entry_points_error() {
        // A module with only global variables but no entry points
        // Naga requires at least one entry point for valid modules
        // So we'll test the extraction directly
        let module = naga::Module::default();
        let result = ShaderReflection::extract_entry_points(&module);
        assert!(matches!(result, Err(ReflectionError::NoEntryPoints)));
    }

    // -------------------------------------------------------------------------
    // Convenience Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_wgsl() {
        let source = r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        let result = reflect_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reflect_wgsl_invalid() {
        let source = "this is not valid wgsl";
        let result = reflect_wgsl(source);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("parse error"));
    }

    // -------------------------------------------------------------------------
    // Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_reflection_display() {
        let source = r#"
            @group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return camera * vec4<f32>(0.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let display = format!("{}", reflection);
        assert!(display.contains("ShaderReflection"));
        assert!(display.contains("Entry Points"));
        assert!(display.contains("Bindings"));
        assert!(display.contains("vs_main"));
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflection_validate_success() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: f32;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        assert!(reflection.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Integer Texture Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_int_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<i32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                let v = textureLoad(tex, vec2<i32>(0, 0), 0);
                return vec4<f32>(f32(v.x), 0.0, 0.0, 1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { sample_type, .. } => {
                assert!(matches!(sample_type, TextureSampleType::Sint));
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_uint_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<u32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                let v = textureLoad(tex, vec2<i32>(0, 0), 0);
                return vec4<f32>(f32(v.x), 0.0, 0.0, 1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { sample_type, .. } => {
                assert!(matches!(sample_type, TextureSampleType::Uint));
            }
            _ => panic!("expected texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Texture Array Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_texture_2d_array() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d_array<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(tex, vec2<i32>(0, 0), 0, 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { dimension, .. } => {
                assert_eq!(*dimension, TextureDimension::D2Array);
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_cube_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_cube<f32>;
            @group(0) @binding(1) var samp: sampler;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(tex, samp, vec3<f32>(0.0, 0.0, 1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let tex_binding = reflection.bindings().iter().find(|b| b.binding == 0).unwrap();
        match &tex_binding.resource_type {
            ResourceType::Texture { dimension, .. } => {
                assert_eq!(*dimension, TextureDimension::Cube);
            }
            _ => panic!("expected texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Multisampled Texture Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_multisampled_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_multisampled_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(tex, vec2<i32>(0, 0), 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { multisampled, .. } => {
                assert!(*multisampled);
            }
            _ => panic!("expected texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Storage Texture Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_storage_texture_rgba32float() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rgba32float, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::Rgba32Float);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_reflect_storage_texture_r32uint() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<r32uint, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<u32>(1u));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::R32Uint);
            }
            _ => panic!("expected storage texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Entry Point Query Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_entry_points_for_stage() {
        let source = r#"
            @vertex fn vs1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
            @vertex fn vs2() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
            @fragment fn fs1() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
        "#;
        let reflection = parse_and_reflect(source);

        let vertex_eps = reflection.entry_points_for_stage(ShaderStage::Vertex);
        assert_eq!(vertex_eps.len(), 2);

        let fragment_eps = reflection.entry_points_for_stage(ShaderStage::Fragment);
        assert_eq!(fragment_eps.len(), 1);

        let compute_eps = reflection.entry_points_for_stage(ShaderStage::Compute);
        assert!(compute_eps.is_empty());
    }

    // -------------------------------------------------------------------------
    // Large Shader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_complex_shader() {
        let source = r#"
            struct CameraData {
                view: mat4x4<f32>,
                proj: mat4x4<f32>,
                view_proj: mat4x4<f32>,
                inv_view_proj: mat4x4<f32>,
            }

            struct LightData {
                position: vec3<f32>,
                intensity: f32,
                color: vec3<f32>,
                radius: f32,
            }

            struct MaterialData {
                albedo: vec4<f32>,
                metallic: f32,
                roughness: f32,
                ao: f32,
                emission: f32,
            }

            @group(0) @binding(0) var<uniform> camera: CameraData;
            @group(0) @binding(1) var<storage, read> lights: array<LightData>;

            @group(1) @binding(0) var<uniform> material: MaterialData;
            @group(1) @binding(1) var albedo_tex: texture_2d<f32>;
            @group(1) @binding(2) var normal_tex: texture_2d<f32>;
            @group(1) @binding(3) var metallic_roughness_tex: texture_2d<f32>;
            @group(1) @binding(4) var default_sampler: sampler;

            @group(2) @binding(0) var env_map: texture_cube<f32>;
            @group(2) @binding(1) var irradiance_map: texture_cube<f32>;
            @group(2) @binding(2) var brdf_lut: texture_2d<f32>;

            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return camera.view_proj * vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                let albedo = textureSample(albedo_tex, default_sampler, vec2<f32>(0.0));
                return albedo * material.albedo;
            }
        "#;

        let reflection = parse_and_reflect(source);

        // Check entry points
        assert_eq!(reflection.entry_points().len(), 2);
        assert!(reflection.vertex_entry_point().is_some());
        assert!(reflection.fragment_entry_point().is_some());

        // Check bind groups
        assert_eq!(reflection.bind_group_count(), 3);

        // Group 0: camera + lights
        let group0 = reflection.bindings_for_group(0);
        assert_eq!(group0.len(), 2);

        // Group 1: material + textures + sampler
        let group1 = reflection.bindings_for_group(1);
        assert_eq!(group1.len(), 5);

        // Group 2: environment maps
        let group2 = reflection.bindings_for_group(2);
        assert_eq!(group2.len(), 3);

        // Verify specific bindings
        let camera_binding = group0.iter().find(|b| b.binding == 0).unwrap();
        assert!(matches!(camera_binding.resource_type, ResourceType::UniformBuffer { .. }));

        let lights_binding = group0.iter().find(|b| b.binding == 1).unwrap();
        assert!(matches!(lights_binding.resource_type, ResourceType::StorageBuffer { read_only: true, .. }));
    }

    // -------------------------------------------------------------------------
    // Texture 3D Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_texture_3d() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_3d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(tex, vec3<i32>(0, 0, 0), 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { dimension, .. } => {
                assert_eq!(*dimension, TextureDimension::D3);
            }
            _ => panic!("expected texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Type Size Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uniform_buffer_size() {
        let source = r#"
            struct Data {
                a: f32,
                b: f32,
                c: vec2<f32>,
            }
            @group(0) @binding(0) var<uniform> data: Data;
            @compute @workgroup_size(1) fn main() { let _x = data.a; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                // struct should be 16 bytes (f32 + f32 + vec2<f32>)
                assert!(size.is_some());
                assert!(*size.as_ref().unwrap() >= 16);
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(MAX_BIND_GROUPS, 4);
        assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
    }

    // =========================================================================
    // ADDITIONAL WHITEBOX TESTS - T-WGPU-P2.7.4
    // =========================================================================

    // -------------------------------------------------------------------------
    // ResourceAccess Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_access_from_naga_empty() {
        // Empty access defaults to Read
        let access = ResourceAccess::from_naga(naga::StorageAccess::empty());
        assert_eq!(access, ResourceAccess::Read);
    }

    #[test]
    fn test_resource_access_display() {
        assert_eq!(format!("{}", ResourceAccess::Read), "read");
        assert_eq!(format!("{}", ResourceAccess::Write), "write");
        assert_eq!(format!("{}", ResourceAccess::ReadWrite), "read_write");
    }

    #[test]
    fn test_resource_access_default() {
        assert_eq!(ResourceAccess::default(), ResourceAccess::Read);
    }

    // -------------------------------------------------------------------------
    // TextureDimension Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_dimension_display() {
        assert_eq!(format!("{}", TextureDimension::D1), "1d");
        assert_eq!(format!("{}", TextureDimension::D2), "2d");
        assert_eq!(format!("{}", TextureDimension::D2Array), "2d_array");
        assert_eq!(format!("{}", TextureDimension::D3), "3d");
        assert_eq!(format!("{}", TextureDimension::Cube), "cube");
        assert_eq!(format!("{}", TextureDimension::CubeArray), "cube_array");
    }

    #[test]
    fn test_texture_dimension_from_naga_d1_arrayed() {
        // D1 ignores arrayed flag
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D1, true),
            TextureDimension::D1
        );
    }

    #[test]
    fn test_texture_dimension_from_naga_d3_arrayed() {
        // D3 ignores arrayed flag
        assert_eq!(
            TextureDimension::from_naga(naga::ImageDimension::D3, true),
            TextureDimension::D3
        );
    }

    // -------------------------------------------------------------------------
    // TextureSampleType Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_sample_type_display() {
        assert_eq!(
            format!("{}", TextureSampleType::Float { filterable: true }),
            "f32"
        );
        assert_eq!(
            format!("{}", TextureSampleType::Float { filterable: false }),
            "f32 (unfilterable)"
        );
        assert_eq!(format!("{}", TextureSampleType::Sint), "i32");
        assert_eq!(format!("{}", TextureSampleType::Uint), "u32");
        assert_eq!(format!("{}", TextureSampleType::Depth), "depth");
    }

    #[test]
    fn test_texture_sample_type_unfilterable() {
        let sample_type = TextureSampleType::Float { filterable: false };
        let wgpu_type = sample_type.to_wgpu();
        assert!(matches!(
            wgpu_type,
            wgpu::TextureSampleType::Float { filterable: false }
        ));
    }

    // -------------------------------------------------------------------------
    // SamplerType Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampler_type_display() {
        assert_eq!(format!("{}", SamplerType::Filtering), "filtering");
        assert_eq!(format!("{}", SamplerType::NonFiltering), "non_filtering");
        assert_eq!(format!("{}", SamplerType::Comparison), "comparison");
    }

    #[test]
    fn test_sampler_type_default() {
        assert_eq!(SamplerType::default(), SamplerType::Filtering);
    }

    // -------------------------------------------------------------------------
    // ResourceType Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_type_uniform_buffer_read_access() {
        let uniform = ResourceType::UniformBuffer {
            size: Some(64),
            has_dynamic_offset: false,
        };
        assert!(uniform.has_read_access());
        assert!(!uniform.has_write_access());
    }

    #[test]
    fn test_resource_type_storage_buffer_readonly_access() {
        let storage = ResourceType::StorageBuffer {
            size: Some(128),
            read_only: true,
            has_dynamic_offset: false,
        };
        assert!(storage.has_read_access());
        assert!(!storage.has_write_access());
    }

    #[test]
    fn test_resource_type_storage_texture_read_access() {
        let storage_tex = ResourceType::StorageTexture {
            dimension: TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            access: ResourceAccess::Read,
        };
        assert!(storage_tex.has_read_access());
        assert!(!storage_tex.has_write_access());
    }

    #[test]
    fn test_resource_type_storage_texture_readwrite_access() {
        let storage_tex = ResourceType::StorageTexture {
            dimension: TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            access: ResourceAccess::ReadWrite,
        };
        assert!(storage_tex.has_read_access());
        assert!(storage_tex.has_write_access());
    }

    #[test]
    fn test_resource_type_sampler_access() {
        let sampler = ResourceType::Sampler {
            sampler_type: SamplerType::Filtering,
        };
        assert!(sampler.has_read_access());
        assert!(!sampler.has_write_access());
    }

    #[test]
    fn test_resource_type_acceleration_structure_access() {
        let accel = ResourceType::AccelerationStructure;
        assert!(accel.has_read_access());
        assert!(!accel.has_write_access());
        assert!(!accel.is_buffer());
        assert!(!accel.is_texture());
        assert!(!accel.is_sampler());
    }

    #[test]
    fn test_resource_type_display_uniform_with_size() {
        let uniform = ResourceType::UniformBuffer {
            size: Some(256),
            has_dynamic_offset: false,
        };
        let display = format!("{}", uniform);
        assert!(display.contains("uniform buffer"));
        assert!(display.contains("256 bytes"));
    }

    #[test]
    fn test_resource_type_display_uniform_no_size() {
        let uniform = ResourceType::UniformBuffer {
            size: None,
            has_dynamic_offset: false,
        };
        let display = format!("{}", uniform);
        assert_eq!(display, "uniform buffer");
    }

    #[test]
    fn test_resource_type_display_storage_readonly() {
        let storage = ResourceType::StorageBuffer {
            size: Some(512),
            read_only: true,
            has_dynamic_offset: false,
        };
        let display = format!("{}", storage);
        assert!(display.contains("storage buffer<read>"));
        assert!(display.contains("512 bytes"));
    }

    #[test]
    fn test_resource_type_display_storage_readwrite() {
        let storage = ResourceType::StorageBuffer {
            size: None,
            read_only: false,
            has_dynamic_offset: false,
        };
        let display = format!("{}", storage);
        assert!(display.contains("storage buffer<read_write>"));
    }

    #[test]
    fn test_resource_type_display_texture() {
        let texture = ResourceType::Texture {
            dimension: TextureDimension::D2,
            sample_type: TextureSampleType::Float { filterable: true },
            multisampled: false,
        };
        let display = format!("{}", texture);
        assert!(display.contains("texture"));
        assert!(display.contains("2d"));
    }

    #[test]
    fn test_resource_type_display_multisampled_texture() {
        let texture = ResourceType::Texture {
            dimension: TextureDimension::D2,
            sample_type: TextureSampleType::Float { filterable: true },
            multisampled: true,
        };
        let display = format!("{}", texture);
        assert!(display.contains("multisampled"));
    }

    #[test]
    fn test_resource_type_display_storage_texture() {
        let storage_tex = ResourceType::StorageTexture {
            dimension: TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            access: ResourceAccess::Write,
        };
        let display = format!("{}", storage_tex);
        assert!(display.contains("texture_storage"));
        assert!(display.contains("write"));
    }

    #[test]
    fn test_resource_type_display_sampler() {
        let sampler = ResourceType::Sampler {
            sampler_type: SamplerType::Comparison,
        };
        let display = format!("{}", sampler);
        assert!(display.contains("sampler"));
        assert!(display.contains("comparison"));
    }

    #[test]
    fn test_resource_type_display_acceleration_structure() {
        let accel = ResourceType::AccelerationStructure;
        let display = format!("{}", accel);
        assert_eq!(display, "acceleration_structure");
    }

    #[test]
    fn test_resource_type_to_wgpu_storage_readonly() {
        let storage = ResourceType::StorageBuffer {
            size: Some(128),
            read_only: true,
            has_dynamic_offset: false,
        };
        let binding_type = storage.to_wgpu_binding_type();
        match binding_type {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                ..
            } => assert!(read_only),
            _ => panic!("expected storage buffer"),
        }
    }

    #[test]
    fn test_resource_type_to_wgpu_storage_readwrite() {
        let storage = ResourceType::StorageBuffer {
            size: Some(128),
            read_only: false,
            has_dynamic_offset: false,
        };
        let binding_type = storage.to_wgpu_binding_type();
        match binding_type {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                ..
            } => assert!(!read_only),
            _ => panic!("expected storage buffer"),
        }
    }

    #[test]
    fn test_resource_type_to_wgpu_texture() {
        let texture = ResourceType::Texture {
            dimension: TextureDimension::Cube,
            sample_type: TextureSampleType::Depth,
            multisampled: false,
        };
        let binding_type = texture.to_wgpu_binding_type();
        match binding_type {
            wgpu::BindingType::Texture {
                view_dimension,
                sample_type,
                multisampled,
            } => {
                assert_eq!(view_dimension, wgpu::TextureViewDimension::Cube);
                assert!(matches!(sample_type, wgpu::TextureSampleType::Depth));
                assert!(!multisampled);
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_resource_type_to_wgpu_storage_texture() {
        let storage_tex = ResourceType::StorageTexture {
            dimension: TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            access: ResourceAccess::ReadWrite,
        };
        let binding_type = storage_tex.to_wgpu_binding_type();
        match binding_type {
            wgpu::BindingType::StorageTexture {
                view_dimension,
                format,
                access,
            } => {
                assert_eq!(view_dimension, wgpu::TextureViewDimension::D3);
                assert_eq!(format, wgpu::TextureFormat::R32Float);
                assert_eq!(access, wgpu::StorageTextureAccess::ReadWrite);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_resource_type_to_wgpu_acceleration_structure() {
        let accel = ResourceType::AccelerationStructure;
        let binding_type = accel.to_wgpu_binding_type();
        // Falls back to read-only storage buffer
        match binding_type {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                ..
            } => {}
            _ => panic!("expected storage buffer fallback"),
        }
    }

    // -------------------------------------------------------------------------
    // EntryPointInfo Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_entry_point_info_new() {
        let ep = EntryPointInfo::new("main", ShaderStage::Compute, Some([16, 8, 4]), 5);
        assert_eq!(ep.name, "main");
        assert_eq!(ep.stage, ShaderStage::Compute);
        assert_eq!(ep.workgroup_size, Some([16, 8, 4]));
        assert_eq!(ep.index, 5);
    }

    #[test]
    fn test_entry_point_workgroup_total_with_zeros() {
        // Zeros are treated as 1 by max(1)
        let ep = EntryPointInfo::new("main", ShaderStage::Compute, Some([0, 0, 0]), 0);
        assert_eq!(ep.workgroup_total(), Some(1));
    }

    #[test]
    fn test_entry_point_display_without_workgroup() {
        let ep = EntryPointInfo::new("fs_main", ShaderStage::Fragment, None, 0);
        let display = format!("{}", ep);
        assert_eq!(display, "@fragment fn fs_main");
        assert!(!display.contains("workgroup_size"));
    }

    // -------------------------------------------------------------------------
    // BindingInfo Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binding_info_new() {
        let binding = BindingInfo::new(
            2,
            5,
            Some("my_buffer".to_string()),
            ResourceType::UniformBuffer {
                size: Some(128),
                has_dynamic_offset: true,
            },
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        assert_eq!(binding.group, 2);
        assert_eq!(binding.binding, 5);
        assert_eq!(binding.name, Some("my_buffer".to_string()));
        assert!(binding.count.is_none());
    }

    #[test]
    fn test_binding_info_with_count() {
        let binding = BindingInfo::new(
            0,
            0,
            None,
            ResourceType::Sampler {
                sampler_type: SamplerType::Filtering,
            },
            wgpu::ShaderStages::FRAGMENT,
        )
        .with_count(NonZeroU32::new(4).unwrap());

        assert_eq!(binding.count, Some(NonZeroU32::new(4).unwrap()));
    }

    #[test]
    fn test_binding_info_display_with_count() {
        let binding = BindingInfo::new(
            0,
            1,
            Some("textures".to_string()),
            ResourceType::Texture {
                dimension: TextureDimension::D2,
                sample_type: TextureSampleType::Float { filterable: true },
                multisampled: false,
            },
            wgpu::ShaderStages::FRAGMENT,
        )
        .with_count(NonZeroU32::new(8).unwrap());

        let display = format!("{}", binding);
        assert!(display.contains("[8]"));
    }

    #[test]
    fn test_binding_info_display_without_name() {
        let binding = BindingInfo::new(
            1,
            2,
            None,
            ResourceType::Sampler {
                sampler_type: SamplerType::NonFiltering,
            },
            wgpu::ShaderStages::COMPUTE,
        );
        let display = format!("{}", binding);
        assert!(display.contains("@group(1)"));
        assert!(display.contains("@binding(2)"));
        assert!(!display.contains("var"));
    }

    #[test]
    fn test_binding_info_to_wgpu_with_count() {
        let binding = BindingInfo::new(
            0,
            0,
            None,
            ResourceType::Texture {
                dimension: TextureDimension::D2,
                sample_type: TextureSampleType::Float { filterable: true },
                multisampled: false,
            },
            wgpu::ShaderStages::FRAGMENT,
        )
        .with_count(NonZeroU32::new(16).unwrap());

        let entry = binding.to_wgpu_layout_entry();
        assert_eq!(entry.count, Some(NonZeroU32::new(16).unwrap()));
    }

    // -------------------------------------------------------------------------
    // PushConstantMember Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_constant_member_new() {
        let member = PushConstantMember::new("offset", 0, 8, "vec2<f32>");
        assert_eq!(member.name, "offset");
        assert_eq!(member.offset, 0);
        assert_eq!(member.size, 8);
        assert_eq!(member.type_name, "vec2<f32>");
    }

    #[test]
    fn test_push_constant_member_display() {
        let member = PushConstantMember::new("scale", 16, 4, "f32");
        let display = format!("{}", member);
        assert!(display.contains("scale"));
        assert!(display.contains("f32"));
        assert!(display.contains("offset 16"));
        assert!(display.contains("4 bytes"));
    }

    // -------------------------------------------------------------------------
    // PushConstantInfo Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_constant_info_with_member() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::COMPUTE, 32)
            .with_member(PushConstantMember::new("time", 0, 4, "f32"))
            .with_member(PushConstantMember::new("resolution", 4, 8, "vec2<f32>"));

        assert_eq!(pc.members.len(), 2);
        assert_eq!(pc.members[0].name, "time");
        assert_eq!(pc.members[1].name, "resolution");
    }

    #[test]
    fn test_push_constant_info_display_empty() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::VERTEX, 16);
        let display = format!("{}", pc);
        assert!(display.contains("push_constant"));
        assert!(display.contains("16 bytes"));
    }

    #[test]
    fn test_push_constant_info_display_with_members() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::FRAGMENT, 20)
            .with_member(PushConstantMember::new("alpha", 0, 4, "f32"));

        let display = format!("{}", pc);
        assert!(display.contains("alpha"));
    }

    #[test]
    fn test_push_constant_to_wgpu_range() {
        let pc = PushConstantInfo::new(
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            64,
        );
        let range = pc.to_wgpu_range();
        assert_eq!(range.stages, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT);
        assert_eq!(range.range.start, 0);
        assert_eq!(range.range.end, 64);
    }

    // -------------------------------------------------------------------------
    // ReflectionError Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflection_error_unsupported_resource_type() {
        let err = ReflectionError::UnsupportedResourceType {
            description: "unknown type".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("unsupported resource type"));
        assert!(display.contains("unknown type"));
    }

    #[test]
    fn test_reflection_error_invalid_push_constants() {
        let err = ReflectionError::InvalidPushConstants {
            message: "size too large".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("invalid push constants"));
        assert!(display.contains("size too large"));
    }

    #[test]
    fn test_reflection_error_group_index_too_large() {
        let err = ReflectionError::GroupIndexTooLarge { group: 5, max: 3 };
        let display = format!("{}", err);
        assert!(display.contains("5"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_reflection_error_layout_generation_failed() {
        let err = ReflectionError::LayoutGenerationFailed {
            message: "no bindings".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("layout generation failed"));
    }

    #[test]
    fn test_reflection_error_is_error_trait() {
        let err = ReflectionError::NoEntryPoints;
        // Test that it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    // -------------------------------------------------------------------------
    // Texture Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_texture_1d() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_1d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(tex, 0, 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture { dimension, .. } => {
                assert_eq!(*dimension, TextureDimension::D1);
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_cube_array_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_cube_array<f32>;
            @group(0) @binding(1) var samp: sampler;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(tex, samp, vec3<f32>(0.0, 0.0, 1.0), 0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let tex_binding = reflection.bindings().iter().find(|b| b.binding == 0).unwrap();
        match &tex_binding.resource_type {
            ResourceType::Texture { dimension, .. } => {
                assert_eq!(*dimension, TextureDimension::CubeArray);
            }
            _ => panic!("expected texture"),
        }
    }

    #[test]
    fn test_reflect_depth_multisampled_texture() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_depth_multisampled_2d;
            @fragment fn main() -> @location(0) vec4<f32> {
                let d = textureLoad(tex, vec2<i32>(0, 0), 0);
                return vec4<f32>(d);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::Texture {
                sample_type,
                multisampled,
                ..
            } => {
                assert!(matches!(sample_type, TextureSampleType::Depth));
                assert!(*multisampled);
            }
            _ => panic!("expected texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Storage Texture Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_storage_texture_r32sint() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<r32sint, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<i32>(1));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::R32Sint);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_reflect_storage_texture_rg32float() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rg32float, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::Rg32Float);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_reflect_storage_texture_rgba16float() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rgba16float, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::Rgba16Float);
            }
            _ => panic!("expected storage texture"),
        }
    }

    #[test]
    fn test_reflect_storage_texture_bgra8unorm() {
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<bgra8unorm, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(output, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::StorageTexture { format, .. } => {
                assert_eq!(*format, wgpu::TextureFormat::Bgra8Unorm);
            }
            _ => panic!("expected storage texture"),
        }
    }

    // -------------------------------------------------------------------------
    // Workgroup Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_workgroup_size_1d() {
        let source = r#"
            @compute @workgroup_size(256)
            fn main() {}
        "#;
        let reflection = parse_and_reflect(source);
        let ep = reflection.compute_entry_point().unwrap();
        assert_eq!(ep.workgroup_size, Some([256, 1, 1]));
    }

    #[test]
    fn test_reflect_workgroup_size_2d() {
        let source = r#"
            @compute @workgroup_size(16, 16)
            fn main() {}
        "#;
        let reflection = parse_and_reflect(source);
        let ep = reflection.compute_entry_point().unwrap();
        assert_eq!(ep.workgroup_size, Some([16, 16, 1]));
    }

    #[test]
    fn test_reflect_large_workgroup() {
        let source = r#"
            @compute @workgroup_size(1024)
            fn main() {}
        "#;
        let reflection = parse_and_reflect(source);
        let ep = reflection.compute_entry_point().unwrap();
        assert_eq!(ep.workgroup_total(), Some(1024));
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_no_bindings() {
        let source = r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.bindings().len(), 0);
        assert_eq!(reflection.bind_group_count(), 0);
    }

    #[test]
    fn test_reflect_sparse_bindings() {
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(0) @binding(5) var<uniform> b: f32;
            @group(0) @binding(10) var<uniform> c: f32;
            @compute @workgroup_size(1) fn main() {
                let _x = a + b + c;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let bindings = reflection.bindings_for_group(0);
        assert_eq!(bindings.len(), 3);

        // Should be sorted by binding index
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[1].binding, 5);
        assert_eq!(bindings[2].binding, 10);
    }

    #[test]
    fn test_reflect_sparse_groups() {
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(3) @binding(0) var<uniform> b: f32;
            @compute @workgroup_size(1) fn main() {
                let _x = a + b;
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.bind_group_count(), 4); // 0, 1, 2, 3 (highest + 1)
        assert_eq!(reflection.bindings_for_group(0).len(), 1);
        assert_eq!(reflection.bindings_for_group(1).len(), 0);
        assert_eq!(reflection.bindings_for_group(2).len(), 0);
        assert_eq!(reflection.bindings_for_group(3).len(), 1);
    }

    #[test]
    fn test_reflect_empty_group() {
        let source = r#"
            @group(1) @binding(0) var<uniform> a: f32;
            @compute @workgroup_size(1) fn main() {
                let _x = a;
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.bindings_for_group(0).len(), 0);
        assert_eq!(reflection.bindings_for_group(1).len(), 1);
    }

    // -------------------------------------------------------------------------
    // Multiple Entry Point Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_vertex_fragment_compute() {
        let source = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
            @compute @workgroup_size(1) fn cs() {}
        "#;
        let reflection = parse_and_reflect(source);
        assert_eq!(reflection.entry_points().len(), 3);

        assert!(reflection.vertex_entry_point().is_some());
        assert!(reflection.fragment_entry_point().is_some());
        assert!(reflection.compute_entry_point().is_some());

        assert_eq!(reflection.vertex_entry_point().unwrap().name, "vs");
        assert_eq!(reflection.fragment_entry_point().unwrap().name, "fs");
        assert_eq!(reflection.compute_entry_point().unwrap().name, "cs");
    }

    #[test]
    fn test_reflect_multiple_compute() {
        let source = r#"
            @compute @workgroup_size(64) fn pass1() {}
            @compute @workgroup_size(128) fn pass2() {}
            @compute @workgroup_size(256) fn pass3() {}
        "#;
        let reflection = parse_and_reflect(source);
        let compute_eps = reflection.entry_points_for_stage(ShaderStage::Compute);
        assert_eq!(compute_eps.len(), 3);
    }

    // -------------------------------------------------------------------------
    // Push Constants Advanced Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_push_constants_mat4() {
        let source = r#"
            struct PushData {
                transform: mat4x4<f32>,
            }
            var<push_constant> push: PushData;
            @compute @workgroup_size(1) fn main() {
                let _x = push.transform;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let pc = reflection.push_constants().expect("expected push constants");
        assert_eq!(pc.size, 64); // 4x4 floats = 64 bytes
    }

    #[test]
    fn test_reflect_push_constants_vertex_fragment() {
        let source = r#"
            struct PushData {
                value: f32,
            }
            var<push_constant> push: PushData;

            @vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(push.value);
            }

            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(push.value);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let pc = reflection.push_constants().expect("expected push constants");
        assert!(pc.stages.contains(wgpu::ShaderStages::VERTEX));
        assert!(pc.stages.contains(wgpu::ShaderStages::FRAGMENT));
    }

    // -------------------------------------------------------------------------
    // Visibility Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binding_visibility_vertex_only() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: f32;
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(data);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        assert!(b.visibility.contains(wgpu::ShaderStages::VERTEX));
    }

    #[test]
    fn test_binding_visibility_fragment_only() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: f32;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(data);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        assert!(b.visibility.contains(wgpu::ShaderStages::FRAGMENT));
    }

    #[test]
    fn test_binding_visibility_compute_only() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: f32;
            @compute @workgroup_size(1) fn main() {
                let _x = data;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        assert!(b.visibility.contains(wgpu::ShaderStages::COMPUTE));
    }

    // -------------------------------------------------------------------------
    // reflect_wgsl Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_wgsl_validation_error() {
        // This should fail validation (undefined variable)
        let source = r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return undefined_var;
            }
        "#;
        let result = reflect_wgsl(source);
        assert!(result.is_err());
        // Could be parse or validation error
    }

    #[test]
    fn test_reflect_wgsl_complete_shader() {
        let source = r#"
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var samp: sampler;

            @vertex
            fn vs_main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
                var out: VertexOutput;
                out.position = mvp * vec4<f32>(pos, 1.0);
                out.uv = uv;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(tex, samp, in.uv);
            }
        "#;
        let result = reflect_wgsl(source);
        assert!(result.is_ok());
        let reflection = result.unwrap();
        assert_eq!(reflection.entry_points().len(), 2);
        assert_eq!(reflection.bindings().len(), 3);
    }

    // -------------------------------------------------------------------------
    // ShaderReflection Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_reflection_display_with_push_constants() {
        let source = r#"
            struct PushData { value: f32 }
            var<push_constant> push: PushData;
            @compute @workgroup_size(1) fn main() {
                let _x = push.value;
            }
        "#;
        let reflection = parse_and_reflect(source);
        let display = format!("{}", reflection);
        assert!(display.contains("Push Constants"));
    }

    #[test]
    fn test_shader_reflection_display_no_bindings() {
        let source = r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let display = format!("{}", reflection);
        assert!(display.contains("0 groups"));
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validation_with_push_constants() {
        let source = r#"
            struct PushData { value: f32 }
            var<push_constant> push: PushData;
            @compute @workgroup_size(1) fn main() {
                let _x = push.value;
            }
        "#;
        let reflection = parse_and_reflect(source);
        assert!(reflection.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Matrix Type Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uniform_buffer_mat2x2() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: mat2x2<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(16)); // 2x2 floats = 16 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_mat3x3() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: mat3x3<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(36)); // 3x3 floats = 36 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_mat4x4() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: mat4x4<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(64)); // 4x4 floats = 64 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    // -------------------------------------------------------------------------
    // Vector Type Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uniform_buffer_vec2() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: vec2<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(8)); // 2 floats = 8 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_vec3() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: vec3<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(12)); // 3 floats = 12 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_vec4() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: vec4<f32>;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(16)); // 4 floats = 16 bytes
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    // -------------------------------------------------------------------------
    // Scalar Type Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uniform_buffer_f32() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: f32;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(4));
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_i32() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: i32;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(4));
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    #[test]
    fn test_uniform_buffer_u32() {
        let source = r#"
            @group(0) @binding(0) var<uniform> data: u32;
            @compute @workgroup_size(1) fn main() { let _x = data; }
        "#;
        let reflection = parse_and_reflect(source);
        let b = &reflection.bindings()[0];
        match &b.resource_type {
            ResourceType::UniformBuffer { size, .. } => {
                assert_eq!(*size, Some(4));
            }
            _ => panic!("expected uniform buffer"),
        }
    }

    // -------------------------------------------------------------------------
    // Clone and PartialEq Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_stage_clone() {
        let stage = ShaderStage::Compute;
        let cloned = stage.clone();
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_resource_access_clone() {
        let access = ResourceAccess::ReadWrite;
        let cloned = access.clone();
        assert_eq!(access, cloned);
    }

    #[test]
    fn test_texture_dimension_clone() {
        let dim = TextureDimension::CubeArray;
        let cloned = dim.clone();
        assert_eq!(dim, cloned);
    }

    #[test]
    fn test_texture_sample_type_clone() {
        let sample_type = TextureSampleType::Float { filterable: true };
        let cloned = sample_type.clone();
        assert_eq!(sample_type, cloned);
    }

    #[test]
    fn test_sampler_type_clone() {
        let sampler = SamplerType::Comparison;
        let cloned = sampler.clone();
        assert_eq!(sampler, cloned);
    }

    #[test]
    fn test_resource_type_clone() {
        let resource = ResourceType::AccelerationStructure;
        let cloned = resource.clone();
        assert_eq!(resource, cloned);
    }

    #[test]
    fn test_entry_point_info_clone() {
        let ep = EntryPointInfo::new("main", ShaderStage::Vertex, None, 0);
        let cloned = ep.clone();
        assert_eq!(ep, cloned);
    }

    #[test]
    fn test_binding_info_clone() {
        let binding = BindingInfo::new(
            0,
            0,
            Some("test".to_string()),
            ResourceType::Sampler {
                sampler_type: SamplerType::Filtering,
            },
            wgpu::ShaderStages::FRAGMENT,
        );
        let cloned = binding.clone();
        assert_eq!(binding, cloned);
    }

    #[test]
    fn test_push_constant_member_clone() {
        let member = PushConstantMember::new("value", 0, 4, "f32");
        let cloned = member.clone();
        assert_eq!(member, cloned);
    }

    #[test]
    fn test_push_constant_info_clone() {
        let pc = PushConstantInfo::new(wgpu::ShaderStages::COMPUTE, 64);
        let cloned = pc.clone();
        assert_eq!(pc, cloned);
    }

    #[test]
    fn test_reflection_error_clone() {
        let err = ReflectionError::NoEntryPoints;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_shader_reflection_clone() {
        let source = r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        let reflection = parse_and_reflect(source);
        let cloned = reflection.clone();
        assert_eq!(reflection.entry_points().len(), cloned.entry_points().len());
    }
}
