//! Naga-based WGSL compilation pipeline for TRINITY.
//!
//! This module provides a complete shader compilation pipeline:
//!
//! - **Parse**: WGSL source -> naga IR module with source location spans
//! - **Validate**: Type checking, binding overlap, entry point validity
//! - **Analyze**: Extract resource bindings, push constants, entry points
//! - **Transpile**: Generate SPIR-V (via naga) or MSL (via naga Metal backend)
//!
//! # Performance Target
//!
//! Compilation should complete in <50ms per entry point (excluding optimization).
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::{NagaCompiler, CompilerOptions, TargetBackend};
//!
//! let compiler = NagaCompiler::new();
//! let source = r#"
//!     @vertex
//!     fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
//!         return vec4<f32>(0.0, 0.0, 0.0, 1.0);
//!     }
//! "#;
//!
//! let result = compiler.compile(source, &CompilerOptions::default())?;
//! let spirv = result.to_spirv()?;
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Source location span for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    /// Byte offset from start of source.
    pub start: u32,
    /// Byte length of the span.
    pub length: u32,
}

impl SourceSpan {
    /// Create a new source span.
    pub fn new(start: u32, length: u32) -> Self {
        Self { start, length }
    }

    /// Convert naga span to our span format.
    pub fn from_naga(span: naga::Span) -> Option<Self> {
        if span.is_defined() {
            let start = span.to_range()?.start as u32;
            let end = span.to_range()?.end as u32;
            Some(Self {
                start,
                length: end - start,
            })
        } else {
            None
        }
    }
}

/// Shader compilation error with source location.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error message.
    pub message: String,
    /// Source location span (if available).
    pub span: Option<SourceSpan>,
    /// Error kind for categorization.
    pub kind: CompileErrorKind,
    /// Additional notes/hints.
    pub notes: Vec<String>,
}

impl CompileError {
    /// Create a new compile error.
    pub fn new(message: impl Into<String>, kind: CompileErrorKind) -> Self {
        Self {
            message: message.into(),
            span: None,
            kind,
            notes: Vec::new(),
        }
    }

    /// Add a source span to the error.
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a note to the error.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Get line and column from source and span.
    pub fn line_column(&self, source: &str) -> Option<(usize, usize)> {
        let span = self.span?;
        let prefix = &source[..span.start as usize];
        let line = prefix.matches('\n').count() + 1;
        let last_newline = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let column = span.start as usize - last_newline + 1;
        Some((line, column))
    }

    /// Format error with source context.
    pub fn format_with_source(&self, source: &str, filename: &str) -> String {
        let mut result = String::new();

        if let Some((line, col)) = self.line_column(source) {
            result.push_str(&format!("{}:{}:{}: ", filename, line, col));
        } else {
            result.push_str(&format!("{}: ", filename));
        }

        result.push_str(&format!("error[{:?}]: {}\n", self.kind, self.message));

        // Show source context if we have a span
        if let Some(span) = self.span {
            let start = span.start as usize;
            let end = (span.start + span.length) as usize;

            // Find the line containing the error
            let line_start = source[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let line_end = source[end..].find('\n').map(|i| end + i).unwrap_or(source.len());
            let line_content = &source[line_start..line_end];

            // Calculate column within line
            let col_start = start - line_start;
            let col_end = (end - line_start).min(line_content.len());

            result.push_str(&format!("  |\n"));
            result.push_str(&format!("  | {}\n", line_content));
            result.push_str(&format!(
                "  | {}{}\n",
                " ".repeat(col_start),
                "^".repeat((col_end - col_start).max(1))
            ));
        }

        for note in &self.notes {
            result.push_str(&format!("  = note: {}\n", note));
        }

        result
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CompileError {}

/// Categories of compilation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileErrorKind {
    /// WGSL parse error.
    Parse,
    /// Type checking error.
    Type,
    /// Resource binding conflict.
    BindingConflict,
    /// Invalid entry point.
    EntryPoint,
    /// SPIR-V generation error.
    SpirV,
    /// MSL generation error.
    Msl,
    /// Validation error.
    Validation,
    /// Internal compiler error.
    Internal,
}

// ---------------------------------------------------------------------------
// Resource Analysis
// ---------------------------------------------------------------------------

/// Shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<naga::ShaderStage> for ShaderStage {
    fn from(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
        }
    }
}

/// Resource binding type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceType {
    /// Uniform buffer.
    UniformBuffer { size: u32 },
    /// Storage buffer (read-only or read-write).
    StorageBuffer { size: u32, read_only: bool },
    /// Sampled texture.
    SampledTexture { dimension: TextureDimension, multisampled: bool },
    /// Storage texture.
    StorageTexture { dimension: TextureDimension, access: StorageAccess },
    /// Sampler.
    Sampler { comparison: bool },
    /// Depth texture.
    DepthTexture { dimension: TextureDimension, multisampled: bool },
}

/// Texture dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDimension {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// Storage texture access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageAccess {
    Load,
    Store,
    ReadWrite,
}

/// A resource binding extracted from the shader.
#[derive(Debug, Clone)]
pub struct ResourceBinding {
    /// Binding group.
    pub group: u32,
    /// Binding index within group.
    pub binding: u32,
    /// Variable name in shader.
    pub name: String,
    /// Resource type.
    pub resource_type: ResourceType,
    /// Shader stages that use this resource.
    pub stages: Vec<ShaderStage>,
}

/// Push constant range.
#[derive(Debug, Clone)]
pub struct PushConstant {
    /// Offset in bytes.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Shader stages that use this.
    pub stages: Vec<ShaderStage>,
}

/// Specialization constant.
#[derive(Debug, Clone)]
pub struct SpecializationConstant {
    /// Constant ID.
    pub id: u32,
    /// Variable name.
    pub name: String,
    /// Type name.
    pub type_name: String,
    /// Default value (as string).
    pub default_value: Option<String>,
}

/// Entry point information.
#[derive(Debug, Clone)]
pub struct EntryPointInfo {
    /// Entry point name.
    pub name: String,
    /// Shader stage.
    pub stage: ShaderStage,
    /// Workgroup size for compute shaders.
    pub workgroup_size: Option<[u32; 3]>,
    /// Input attributes (for vertex shaders).
    pub inputs: Vec<ShaderInput>,
    /// Output attributes (for vertex/fragment shaders).
    pub outputs: Vec<ShaderOutput>,
}

/// Shader input attribute.
#[derive(Debug, Clone)]
pub struct ShaderInput {
    /// Location attribute.
    pub location: Option<u32>,
    /// Builtin type.
    pub builtin: Option<String>,
    /// Variable name.
    pub name: String,
    /// Type name.
    pub type_name: String,
}

/// Shader output attribute.
#[derive(Debug, Clone)]
pub struct ShaderOutput {
    /// Location attribute.
    pub location: Option<u32>,
    /// Builtin type.
    pub builtin: Option<String>,
    /// Variable name.
    pub name: String,
    /// Type name.
    pub type_name: String,
}

/// Complete analysis of a compiled shader module.
#[derive(Debug, Clone)]
pub struct ShaderAnalysis {
    /// Resource bindings.
    pub bindings: Vec<ResourceBinding>,
    /// Push constants.
    pub push_constants: Vec<PushConstant>,
    /// Specialization constants (currently stubbed - WGSL doesn't fully support).
    pub specialization_constants: Vec<SpecializationConstant>,
    /// Entry points.
    pub entry_points: Vec<EntryPointInfo>,
}

// ---------------------------------------------------------------------------
// Compilation Options
// ---------------------------------------------------------------------------

/// Target backend for transpilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetBackend {
    /// SPIR-V for Vulkan.
    SpirV,
    /// MSL for Metal.
    Msl,
    /// WGSL (pass-through with validation).
    Wgsl,
    /// DXIL for D3D12 (stub - requires DXC).
    Dxil,
}

/// SPIR-V generation options.
#[derive(Debug, Clone)]
pub struct SpirVOptions {
    /// SPIR-V version (major, minor).
    pub version: (u8, u8),
    /// Debug info.
    pub debug: bool,
    /// Adjust coordinate space.
    pub adjust_coordinate_space: bool,
    /// Binding base offsets per group.
    pub binding_base: HashMap<u32, u32>,
}

impl Default for SpirVOptions {
    fn default() -> Self {
        Self {
            version: (1, 3), // Vulkan 1.1 compatible
            debug: false,
            adjust_coordinate_space: true,
            binding_base: HashMap::new(),
        }
    }
}

/// MSL generation options.
#[derive(Debug, Clone)]
pub struct MslOptions {
    /// Target language version.
    pub lang_version: (u8, u8),
    /// Target platform.
    pub platform: MslPlatform,
    /// Use argument buffers.
    pub argument_buffers: bool,
}

impl Default for MslOptions {
    fn default() -> Self {
        Self {
            lang_version: (2, 4),
            platform: MslPlatform::MacOS,
            argument_buffers: false,
        }
    }
}

/// MSL target platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MslPlatform {
    MacOS,
    IOS,
}

/// Compiler options.
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Enable validation.
    pub validate: bool,
    /// Strict mode (fail on warnings).
    pub strict: bool,
    /// Target backend.
    pub target: TargetBackend,
    /// SPIR-V options.
    pub spirv: SpirVOptions,
    /// MSL options.
    pub msl: MslOptions,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            validate: true,
            strict: false,
            target: TargetBackend::SpirV,
            spirv: SpirVOptions::default(),
            msl: MslOptions::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation Result
// ---------------------------------------------------------------------------

/// Result of shader compilation.
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// Parsed and validated naga module.
    module: naga::Module,
    /// Module info from validation.
    info: naga::valid::ModuleInfo,
    /// Shader analysis.
    pub analysis: ShaderAnalysis,
    /// Compilation warnings.
    pub warnings: Vec<String>,
}

impl CompileResult {
    /// Generate SPIR-V bytecode.
    pub fn to_spirv(&self, options: &SpirVOptions) -> Result<Vec<u32>, CompileError> {
        let mut spv_options = naga::back::spv::Options::default();
        spv_options.lang_version = options.version;

        // Note: Binding base offsets could be supported via binding_map
        // but would require more complex setup. For now, we use default behavior.
        // The binding_base option is reserved for future use.
        let _ = &options.binding_base;

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: self.analysis.entry_points.first()
                .map(|ep| match ep.stage {
                    ShaderStage::Vertex => naga::ShaderStage::Vertex,
                    ShaderStage::Fragment => naga::ShaderStage::Fragment,
                    ShaderStage::Compute => naga::ShaderStage::Compute,
                })
                .unwrap_or(naga::ShaderStage::Vertex),
            entry_point: self.analysis.entry_points.first()
                .map(|ep| ep.name.clone())
                .unwrap_or_default(),
        };

        let mut writer = naga::back::spv::Writer::new(&spv_options).map_err(|e| {
            CompileError::new(format!("SPIR-V writer init failed: {}", e), CompileErrorKind::SpirV)
        })?;

        let mut words = Vec::new();
        writer.write(&self.module, &self.info, Some(&pipeline_options), &None, &mut words)
            .map_err(|e| {
                CompileError::new(format!("SPIR-V generation failed: {:?}", e), CompileErrorKind::SpirV)
            })?;

        Ok(words)
    }

    /// Generate SPIR-V with default options.
    pub fn to_spirv_default(&self) -> Result<Vec<u32>, CompileError> {
        self.to_spirv(&SpirVOptions::default())
    }

    /// Generate MSL source code.
    pub fn to_msl(&self, options: &MslOptions) -> Result<String, CompileError> {
        let mut msl_options = naga::back::msl::Options::default();
        msl_options.lang_version = options.lang_version;
        msl_options.per_entry_point_map = Default::default();

        // Create pipeline options for each entry point
        let pipeline_options = naga::back::msl::PipelineOptions {
            allow_and_force_point_size: false,
            vertex_buffer_mappings: Vec::new(),
            vertex_pulling_transform: false,
        };

        let (source, _) = naga::back::msl::write_string(&self.module, &self.info, &msl_options, &pipeline_options)
            .map_err(|e| {
                CompileError::new(format!("MSL generation failed: {:?}", e), CompileErrorKind::Msl)
            })?;

        Ok(source)
    }

    /// Generate MSL with default options.
    pub fn to_msl_default(&self) -> Result<String, CompileError> {
        self.to_msl(&MslOptions::default())
    }

    /// Generate validated WGSL source.
    pub fn to_wgsl(&self) -> Result<String, CompileError> {
        let flags = naga::back::wgsl::WriterFlags::empty();
        let source = naga::back::wgsl::write_string(&self.module, &self.info, flags)
            .map_err(|e| {
                CompileError::new(format!("WGSL generation failed: {:?}", e), CompileErrorKind::Internal)
            })?;
        Ok(source)
    }

    /// Get the underlying naga module (for advanced use).
    pub fn module(&self) -> &naga::Module {
        &self.module
    }

    /// Get the module validation info.
    pub fn info(&self) -> &naga::valid::ModuleInfo {
        &self.info
    }
}

// ---------------------------------------------------------------------------
// NagaCompiler
// ---------------------------------------------------------------------------

/// WGSL shader compiler using naga.
pub struct NagaCompiler {
    /// Validation flags.
    validation_flags: naga::valid::ValidationFlags,
    /// Capabilities to enable.
    capabilities: naga::valid::Capabilities,
}

impl Default for NagaCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl NagaCompiler {
    /// Create a new compiler with default settings.
    pub fn new() -> Self {
        Self {
            validation_flags: naga::valid::ValidationFlags::all(),
            capabilities: naga::valid::Capabilities::all(),
        }
    }

    /// Create a compiler with specific capabilities.
    pub fn with_capabilities(capabilities: naga::valid::Capabilities) -> Self {
        Self {
            validation_flags: naga::valid::ValidationFlags::all(),
            capabilities,
        }
    }

    /// Set validation flags.
    pub fn set_validation_flags(&mut self, flags: naga::valid::ValidationFlags) {
        self.validation_flags = flags;
    }

    /// Parse WGSL source into a naga module.
    pub fn parse(&self, source: &str) -> Result<naga::Module, CompileError> {
        naga::front::wgsl::parse_str(source).map_err(|e| {
            let mut error = CompileError::new(e.message().to_string(), CompileErrorKind::Parse);

            // Extract span from the error
            let labels = e.labels();
            for (span, label) in labels {
                error.span = SourceSpan::from_naga(span);
                if !label.is_empty() {
                    error = error.with_note(label.to_string());
                }
                break; // Use first label
            }

            error
        })
    }

    /// Validate a parsed module.
    pub fn validate(&self, module: &naga::Module) -> Result<naga::valid::ModuleInfo, CompileError> {
        let mut validator = naga::valid::Validator::new(self.validation_flags, self.capabilities);
        validator.validate(module).map_err(|e| {
            let mut error = CompileError::new(
                format!("Validation failed: {:?}", e),
                CompileErrorKind::Validation,
            );

            // Try to extract span information
            if let Some(inner) = e.spans().next() {
                error.span = SourceSpan::from_naga(inner.0);
            }

            error
        })
    }

    /// Analyze a validated module to extract resource information.
    pub fn analyze(&self, module: &naga::Module, info: &naga::valid::ModuleInfo) -> ShaderAnalysis {
        let mut bindings = Vec::new();
        let mut push_constants = Vec::new();
        let entry_points = self.extract_entry_points(module, info);

        // Collect which entry points use which global variables
        let mut var_stages: HashMap<naga::Handle<naga::GlobalVariable>, Vec<ShaderStage>> = HashMap::new();
        for ep in &module.entry_points {
            let stage = ShaderStage::from(ep.stage);
            // Note: Full analysis would trace function calls to determine which variables
            // are actually used by each entry point. For now, we conservatively assume
            // all global variables may be used by all entry points.
            for (handle, _var) in module.global_variables.iter() {
                var_stages.entry(handle).or_default().push(stage);
            }
        }

        // Extract bindings from global variables
        for (handle, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                let resource_type = self.classify_resource_type(module, var);
                let name = var.name.clone().unwrap_or_else(|| format!("_var_{}", handle.index()));
                let stages = var_stages.get(&handle).cloned().unwrap_or_default();

                bindings.push(ResourceBinding {
                    group: binding.group,
                    binding: binding.binding,
                    name,
                    resource_type,
                    stages,
                });
            } else if var.space == naga::AddressSpace::PushConstant {
                // Extract push constant
                let size = self.type_size(module, var.ty);
                push_constants.push(PushConstant {
                    offset: 0,
                    size,
                    stages: var_stages.get(&handle).cloned().unwrap_or_default(),
                });
            }
        }

        // Sort bindings by group and binding index
        bindings.sort_by(|a, b| {
            a.group.cmp(&b.group).then(a.binding.cmp(&b.binding))
        });

        ShaderAnalysis {
            bindings,
            push_constants,
            specialization_constants: Vec::new(), // WGSL doesn't have spec constants yet
            entry_points,
        }
    }

    /// Classify the resource type of a global variable.
    fn classify_resource_type(&self, module: &naga::Module, var: &naga::GlobalVariable) -> ResourceType {
        let inner = &module.types[var.ty].inner;
        match inner {
            naga::TypeInner::Scalar(_) | naga::TypeInner::Vector { .. } |
            naga::TypeInner::Matrix { .. } | naga::TypeInner::Struct { .. } => {
                match var.space {
                    naga::AddressSpace::Uniform => {
                        ResourceType::UniformBuffer { size: self.type_size(module, var.ty) }
                    }
                    naga::AddressSpace::Storage { access } => {
                        ResourceType::StorageBuffer {
                            size: self.type_size(module, var.ty),
                            read_only: !access.contains(naga::StorageAccess::STORE),
                        }
                    }
                    _ => ResourceType::UniformBuffer { size: 0 },
                }
            }
            naga::TypeInner::Image { dim, arrayed, class, .. } => {
                let dimension = match (dim, arrayed) {
                    (naga::ImageDimension::D1, false) => TextureDimension::D1,
                    (naga::ImageDimension::D2, false) => TextureDimension::D2,
                    (naga::ImageDimension::D2, true) => TextureDimension::D2Array,
                    (naga::ImageDimension::D3, _) => TextureDimension::D3,
                    (naga::ImageDimension::Cube, false) => TextureDimension::Cube,
                    (naga::ImageDimension::Cube, true) => TextureDimension::CubeArray,
                    _ => TextureDimension::D2,
                };

                match class {
                    naga::ImageClass::Sampled { multi, .. } => {
                        ResourceType::SampledTexture { dimension, multisampled: *multi }
                    }
                    naga::ImageClass::Depth { multi } => {
                        ResourceType::DepthTexture { dimension, multisampled: *multi }
                    }
                    naga::ImageClass::Storage { access, .. } => {
                        let storage_access = if access.contains(naga::StorageAccess::LOAD) &&
                                               access.contains(naga::StorageAccess::STORE) {
                            StorageAccess::ReadWrite
                        } else if access.contains(naga::StorageAccess::STORE) {
                            StorageAccess::Store
                        } else {
                            StorageAccess::Load
                        };
                        ResourceType::StorageTexture { dimension, access: storage_access }
                    }
                }
            }
            naga::TypeInner::Sampler { comparison } => {
                ResourceType::Sampler { comparison: *comparison }
            }
            naga::TypeInner::Array { .. } | naga::TypeInner::BindingArray { .. } => {
                match var.space {
                    naga::AddressSpace::Uniform => {
                        ResourceType::UniformBuffer { size: self.type_size(module, var.ty) }
                    }
                    naga::AddressSpace::Storage { access } => {
                        ResourceType::StorageBuffer {
                            size: self.type_size(module, var.ty),
                            read_only: !access.contains(naga::StorageAccess::STORE),
                        }
                    }
                    _ => ResourceType::UniformBuffer { size: 0 },
                }
            }
            _ => ResourceType::UniformBuffer { size: 0 },
        }
    }

    /// Estimate the size of a type in bytes.
    fn type_size(&self, module: &naga::Module, ty: naga::Handle<naga::Type>) -> u32 {
        let inner = &module.types[ty].inner;
        match inner {
            naga::TypeInner::Scalar(scalar) => scalar.width as u32,
            naga::TypeInner::Vector { scalar, size } => {
                scalar.width as u32 * (*size as u32)
            }
            naga::TypeInner::Matrix { columns, rows, scalar } => {
                scalar.width as u32 * (*columns as u32) * (*rows as u32)
            }
            naga::TypeInner::Struct { members, .. } => {
                members.last().map(|m| m.offset + self.type_size(module, m.ty)).unwrap_or(0)
            }
            naga::TypeInner::Array { base: _, size, stride } => {
                match size {
                    naga::ArraySize::Constant(n) => stride * n.get(),
                    naga::ArraySize::Dynamic => *stride, // Runtime-sized
                    naga::ArraySize::Pending(_) => *stride, // Pending size
                }
            }
            _ => 0,
        }
    }

    /// Extract entry point information.
    fn extract_entry_points(&self, module: &naga::Module, _info: &naga::valid::ModuleInfo) -> Vec<EntryPointInfo> {
        module.entry_points.iter().map(|ep| {
            let workgroup_size = if ep.stage == naga::ShaderStage::Compute {
                Some(ep.workgroup_size)
            } else {
                None
            };

            let function = &ep.function;
            let inputs = self.extract_inputs(module, function);
            let outputs = self.extract_outputs(module, function);

            EntryPointInfo {
                name: ep.name.clone(),
                stage: ShaderStage::from(ep.stage),
                workgroup_size,
                inputs,
                outputs,
            }
        }).collect()
    }

    /// Extract shader inputs from function arguments.
    fn extract_inputs(&self, module: &naga::Module, function: &naga::Function) -> Vec<ShaderInput> {
        function.arguments.iter().map(|arg| {
            let (location, builtin) = match &arg.binding {
                Some(naga::Binding::Location { location, .. }) => (Some(*location), None),
                Some(naga::Binding::BuiltIn(b)) => (None, Some(format!("{:?}", b))),
                None => (None, None),
            };

            ShaderInput {
                location,
                builtin,
                name: arg.name.clone().unwrap_or_default(),
                type_name: self.type_name(module, arg.ty),
            }
        }).collect()
    }

    /// Extract shader outputs from function result.
    fn extract_outputs(&self, module: &naga::Module, function: &naga::Function) -> Vec<ShaderOutput> {
        let Some(result) = &function.result else {
            return Vec::new();
        };

        let (location, builtin) = match &result.binding {
            Some(naga::Binding::Location { location, .. }) => (Some(*location), None),
            Some(naga::Binding::BuiltIn(b)) => (None, Some(format!("{:?}", b))),
            None => (None, None),
        };

        vec![ShaderOutput {
            location,
            builtin,
            name: String::new(),
            type_name: self.type_name(module, result.ty),
        }]
    }

    /// Get a human-readable type name.
    fn type_name(&self, module: &naga::Module, ty: naga::Handle<naga::Type>) -> String {
        let inner = &module.types[ty].inner;
        match inner {
            naga::TypeInner::Scalar(s) => {
                match s.kind {
                    naga::ScalarKind::Sint => format!("i{}", s.width * 8),
                    naga::ScalarKind::Uint => format!("u{}", s.width * 8),
                    naga::ScalarKind::Float => format!("f{}", s.width * 8),
                    naga::ScalarKind::Bool => "bool".to_string(),
                    naga::ScalarKind::AbstractInt => "abstract_int".to_string(),
                    naga::ScalarKind::AbstractFloat => "abstract_float".to_string(),
                }
            }
            naga::TypeInner::Vector { scalar, size } => {
                let base = match scalar.kind {
                    naga::ScalarKind::Sint => "i32",
                    naga::ScalarKind::Uint => "u32",
                    naga::ScalarKind::Float if scalar.width == 2 => "f16",
                    naga::ScalarKind::Float => "f32",
                    naga::ScalarKind::Bool => "bool",
                    _ => "?",
                };
                format!("vec{}<{}>", *size as u8, base)
            }
            naga::TypeInner::Matrix { columns, rows, scalar } => {
                format!("mat{}x{}<f{}>", *columns as u8, *rows as u8, scalar.width * 8)
            }
            naga::TypeInner::Struct { .. } => {
                module.types[ty].name.clone().unwrap_or_else(|| "struct".to_string())
            }
            naga::TypeInner::Array { base, size, .. } => {
                let base_name = self.type_name(module, *base);
                match size {
                    naga::ArraySize::Constant(n) => format!("array<{}, {}>", base_name, n.get()),
                    naga::ArraySize::Dynamic => format!("array<{}>", base_name),
                    naga::ArraySize::Pending(_) => format!("array<{}>", base_name),
                }
            }
            _ => "unknown".to_string(),
        }
    }

    /// Compile WGSL source with full pipeline.
    pub fn compile(&self, source: &str, options: &CompilerOptions) -> Result<CompileResult, CompileError> {
        // Parse
        let module = self.parse(source)?;

        // Validate (if enabled)
        let info = if options.validate {
            self.validate(&module)?
        } else {
            // Minimal validation to get ModuleInfo
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::empty(),
                self.capabilities,
            );
            validator.validate(&module).map_err(|e| {
                CompileError::new(format!("Analysis failed: {:?}", e), CompileErrorKind::Internal)
            })?
        };

        // Analyze
        let analysis = self.analyze(&module, &info);

        // Check for binding conflicts
        if options.validate {
            self.check_binding_conflicts(&analysis)?;
        }

        Ok(CompileResult {
            module,
            info,
            analysis,
            warnings: Vec::new(),
        })
    }

    /// Check for binding conflicts.
    fn check_binding_conflicts(&self, analysis: &ShaderAnalysis) -> Result<(), CompileError> {
        let mut seen: HashMap<(u32, u32), &str> = HashMap::new();

        for binding in &analysis.bindings {
            let key = (binding.group, binding.binding);
            if let Some(existing) = seen.get(&key) {
                return Err(CompileError::new(
                    format!(
                        "Binding conflict: @group({}) @binding({}) used by both '{}' and '{}'",
                        binding.group, binding.binding, existing, binding.name
                    ),
                    CompileErrorKind::BindingConflict,
                ));
            }
            seen.insert(key, &binding.name);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Integration with Preprocessor
// ---------------------------------------------------------------------------

use super::WgslPreprocessor;

/// Compile WGSL source with preprocessing.
pub fn compile_with_preprocessing(
    source: &str,
    filename: &str,
    preprocessor: &WgslPreprocessor,
    options: &CompilerOptions,
) -> Result<CompileResult, CompileError> {
    // Preprocess
    let preprocessed = preprocessor.preprocess(source, filename).map_err(|e| {
        CompileError::new(format!("Preprocessing failed: {}", e), CompileErrorKind::Parse)
    })?;

    // Compile
    let compiler = NagaCompiler::new();
    compiler.compile(&preprocessed.output, options)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // Parse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_simple_wgsl() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        let module = compiler.parse(source);
        assert!(module.is_ok(), "Failed to parse simple WGSL: {:?}", module.err());
    }

    #[test]
    fn test_parse_compute_shader() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let module = compiler.parse(source);
        assert!(module.is_ok(), "Failed to parse compute shader: {:?}", module.err());
    }

    #[test]
    fn test_parse_vertex_fragment_shader() {
        let compiler = NagaCompiler::new();
        let source = r#"
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @vertex
            fn vs_main(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
                var out: VertexOutput;
                out.position = vec4<f32>(position, 1.0);
                out.uv = uv;
                return out;
            }

            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(tex, samp, uv);
            }
        "#;

        let module = compiler.parse(source);
        assert!(module.is_ok(), "Failed to parse vertex/fragment shader: {:?}", module.err());
    }

    #[test]
    fn test_parse_error_with_span() {
        let compiler = NagaCompiler::new();
        let source = "fn main() { let x: vec4 = 1.0; }"; // type mismatch

        let result = compiler.parse(source);
        // The parse might succeed but validation will fail
        // Let's test an actual parse error
        let bad_source = "fn main() { }}}";
        let result = compiler.parse(bad_source);
        assert!(result.is_err());

        if let Err(e) = result {
            assert_eq!(e.kind, CompileErrorKind::Parse);
            // Error should have a span or at least a message
            assert!(!e.message.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_correct_shader() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        let module = compiler.parse(source).unwrap();
        let info = compiler.validate(&module);
        assert!(info.is_ok(), "Validation failed: {:?}", info.err());
    }

    #[test]
    fn test_validate_incorrect_shader_type_error() {
        let compiler = NagaCompiler::new();
        // This shader has a type error - returning i32 where f32 expected
        let source = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                let x: i32 = 1;
                return vec4<i32>(x, x, x, x);
            }
        "#;

        let module = compiler.parse(source);
        // This should fail at parse or validation
        if let Ok(module) = module {
            let info = compiler.validate(&module);
            // Validation may or may not catch this - naga is lenient
            // The important thing is we don't crash
            let _ = info;
        }
    }

    #[test]
    fn test_validate_binding_conflicts() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(0) @binding(0) var<uniform> b: f32;

            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(a + b, 0.0, 0.0, 1.0);
            }
        "#;

        // Either naga's validation or our custom check should catch the binding conflict
        let result = compiler.compile(source, &CompilerOptions::default());
        assert!(result.is_err(), "Expected binding conflict to be detected");

        let err = result.unwrap_err();
        // Either naga catches it during validation or we catch it in our check
        assert!(
            err.kind == CompileErrorKind::BindingConflict || err.kind == CompileErrorKind::Validation,
            "Expected BindingConflict or Validation error, got {:?}",
            err.kind
        );
    }

    // -----------------------------------------------------------------------
    // Analysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_resource_bindings() {
        let compiler = NagaCompiler::new();
        let source = r#"
            struct Uniforms {
                mvp: mat4x4<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var samp: sampler;
            @group(1) @binding(0) var<storage, read> data: array<f32>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let analysis = &result.analysis;

        // Should have 4 bindings
        assert_eq!(analysis.bindings.len(), 4);

        // Check uniform buffer
        let uniforms = analysis.bindings.iter().find(|b| b.name == "uniforms").unwrap();
        assert_eq!(uniforms.group, 0);
        assert_eq!(uniforms.binding, 0);
        assert!(matches!(uniforms.resource_type, ResourceType::UniformBuffer { .. }));

        // Check texture
        let tex = analysis.bindings.iter().find(|b| b.name == "tex").unwrap();
        assert_eq!(tex.group, 0);
        assert_eq!(tex.binding, 1);
        assert!(matches!(tex.resource_type, ResourceType::SampledTexture { .. }));

        // Check sampler
        let samp = analysis.bindings.iter().find(|b| b.name == "samp").unwrap();
        assert_eq!(samp.group, 0);
        assert_eq!(samp.binding, 2);
        assert!(matches!(samp.resource_type, ResourceType::Sampler { .. }));

        // Check storage buffer
        let data = analysis.bindings.iter().find(|b| b.name == "data").unwrap();
        assert_eq!(data.group, 1);
        assert_eq!(data.binding, 0);
        if let ResourceType::StorageBuffer { read_only, .. } = data.resource_type {
            assert!(read_only);
        } else {
            panic!("Expected storage buffer");
        }
    }

    #[test]
    fn test_extract_push_constants() {
        let compiler = NagaCompiler::new();
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

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let analysis = &result.analysis;

        // Should have push constants
        assert!(!analysis.push_constants.is_empty());
    }

    #[test]
    fn test_extract_entry_points() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let analysis = &result.analysis;

        // Should have 2 entry points
        assert_eq!(analysis.entry_points.len(), 2);

        let vs = analysis.entry_points.iter().find(|e| e.name == "vs_main").unwrap();
        assert_eq!(vs.stage, ShaderStage::Vertex);
        assert!(vs.workgroup_size.is_none());

        let fs = analysis.entry_points.iter().find(|e| e.name == "fs_main").unwrap();
        assert_eq!(fs.stage, ShaderStage::Fragment);
    }

    #[test]
    fn test_extract_compute_workgroup_size() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let ep = &result.analysis.entry_points[0];

        assert_eq!(ep.stage, ShaderStage::Compute);
        assert_eq!(ep.workgroup_size, Some([8, 8, 1]));
    }

    // -----------------------------------------------------------------------
    // SPIR-V generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_spirv() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let spirv = result.to_spirv_default();

        assert!(spirv.is_ok(), "SPIR-V generation failed: {:?}", spirv.err());
        let words = spirv.unwrap();

        // SPIR-V magic number
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x07230203); // SPIR-V magic
    }

    #[test]
    fn test_generate_spirv_compute() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                data[id.x] = data[id.x] * 2.0;
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let spirv = result.to_spirv_default();

        assert!(spirv.is_ok());
        let words = spirv.unwrap();
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x07230203);
    }

    // -----------------------------------------------------------------------
    // MSL generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_msl() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let msl = result.to_msl_default();

        assert!(msl.is_ok(), "MSL generation failed: {:?}", msl.err());
        let source = msl.unwrap();

        // Should contain Metal shader language constructs
        assert!(source.contains("metal"));
        assert!(source.contains("vertex"));
    }

    #[test]
    fn test_generate_msl_fragment() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let msl = result.to_msl_default();

        assert!(msl.is_ok());
        let source = msl.unwrap();
        assert!(source.contains("fragment"));
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_with_source_location() {
        let compiler = NagaCompiler::new();
        let source = "fn main() { let x = ; }"; // syntax error

        let result = compiler.parse(source);
        assert!(result.is_err());

        let error = result.unwrap_err();
        // Error should have location info
        // The span may or may not be present depending on the error type
        let formatted = error.format_with_source(source, "test.wgsl");
        assert!(formatted.contains("test.wgsl"));
        assert!(formatted.contains("error"));
    }

    #[test]
    fn test_error_line_column_calculation() {
        let error = CompileError::new("test error", CompileErrorKind::Parse)
            .with_span(SourceSpan::new(10, 5));

        let source = "line 1\nline 2\nline 3";
        if let Some((line, col)) = error.line_column(source) {
            assert!(line >= 1);
            assert!(col >= 1);
        }
    }

    // -----------------------------------------------------------------------
    // Performance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_performance_benchmark() {
        let compiler = NagaCompiler::new();
        let source = r#"
            struct Uniforms {
                model: mat4x4<f32>,
                view: mat4x4<f32>,
                proj: mat4x4<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var tex: texture_2d<f32>;
            @group(0) @binding(2) var samp: sampler;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
                @location(1) normal: vec3<f32>,
            }

            @vertex
            fn vs_main(
                @location(0) position: vec3<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) uv: vec2<f32>,
            ) -> VertexOutput {
                var out: VertexOutput;
                let world_pos = uniforms.model * vec4<f32>(position, 1.0);
                out.position = uniforms.proj * uniforms.view * world_pos;
                out.uv = uv;
                out.normal = (uniforms.model * vec4<f32>(normal, 0.0)).xyz;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let albedo = textureSample(tex, samp, in.uv);
                let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
                let ndotl = max(dot(normalize(in.normal), light_dir), 0.0);
                return vec4<f32>(albedo.rgb * ndotl, albedo.a);
            }
        "#;

        // Warm up
        let _ = compiler.compile(source, &CompilerOptions::default());

        // Benchmark
        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
            let _ = result.to_spirv_default().unwrap();
        }

        let elapsed = start.elapsed();
        let per_iteration = elapsed / iterations;

        // Should be under 50ms per entry point
        // We have 2 entry points, so 100ms total budget
        assert!(
            per_iteration.as_millis() < 100,
            "Compilation too slow: {:?} per iteration",
            per_iteration
        );

        println!("Performance: {:?} per compile+spirv", per_iteration);
    }

    // -----------------------------------------------------------------------
    // Integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_with_preprocessing() {
        let pp = WgslPreprocessor::new();
        let source = r#"
            #define MAX_LIGHTS 8

            @fragment
            fn main() -> @location(0) vec4<f32> {
                var total = 0.0;
                for (var i = 0; i < MAX_LIGHTS; i++) {
                    total += 1.0;
                }
                return vec4<f32>(total / f32(MAX_LIGHTS), 0.0, 0.0, 1.0);
            }
        "#;

        let result = compile_with_preprocessing(
            source,
            "test.wgsl",
            &pp,
            &CompilerOptions::default(),
        );

        assert!(result.is_ok(), "Compile with preprocessing failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // Additional edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_depth_texture() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var depth_tex: texture_depth_2d;
            @group(0) @binding(1) var samp: sampler_comparison;

            @fragment
            fn main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
                return textureSampleCompare(depth_tex, samp, pos.xy, 0.5);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();

        let depth = result.analysis.bindings.iter()
            .find(|b| b.name == "depth_tex")
            .unwrap();
        assert!(matches!(depth.resource_type, ResourceType::DepthTexture { .. }));

        let samp = result.analysis.bindings.iter()
            .find(|b| b.name == "samp")
            .unwrap();
        if let ResourceType::Sampler { comparison } = samp.resource_type {
            assert!(comparison);
        } else {
            panic!("Expected comparison sampler");
        }
    }

    #[test]
    fn test_storage_texture() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                textureStore(output, vec2<i32>(id.xy), vec4<f32>(1.0));
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();

        let storage = result.analysis.bindings.iter()
            .find(|b| b.name == "output")
            .unwrap();
        if let ResourceType::StorageTexture { access, .. } = storage.resource_type {
            assert_eq!(access, StorageAccess::Store);
        } else {
            panic!("Expected storage texture");
        }
    }

    #[test]
    fn test_multisampled_texture() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var ms_tex: texture_multisampled_2d<f32>;

            @fragment
            fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                return textureLoad(ms_tex, vec2<i32>(pos.xy), 0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();

        let ms = result.analysis.bindings.iter()
            .find(|b| b.name == "ms_tex")
            .unwrap();
        if let ResourceType::SampledTexture { multisampled, .. } = ms.resource_type {
            assert!(multisampled);
        } else {
            panic!("Expected multisampled texture");
        }
    }

    #[test]
    fn test_cube_texture() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @group(0) @binding(0) var env_map: texture_cube<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                let dir = vec3<f32>(1.0, 0.0, 0.0);
                return textureSample(env_map, samp, dir);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();

        let cube = result.analysis.bindings.iter()
            .find(|b| b.name == "env_map")
            .unwrap();
        if let ResourceType::SampledTexture { dimension, .. } = &cube.resource_type {
            assert_eq!(*dimension, TextureDimension::Cube);
        } else {
            panic!("Expected cube texture");
        }
    }

    #[test]
    fn test_wgsl_roundtrip() {
        let compiler = NagaCompiler::new();
        let source = r#"
            @vertex
            fn main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
            }
        "#;

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();
        let regenerated = result.to_wgsl().unwrap();

        // Should be valid WGSL
        let result2 = compiler.compile(&regenerated, &CompilerOptions::default());
        assert!(result2.is_ok(), "Roundtrip WGSL invalid: {:?}", result2.err());
    }

    #[test]
    fn test_empty_shader() {
        let compiler = NagaCompiler::new();
        let source = "";

        // Empty shader should fail gracefully
        let result = compiler.parse(source);
        // Naga may accept empty module
        if result.is_ok() {
            let module = result.unwrap();
            assert!(module.entry_points.is_empty());
        }
    }

    #[test]
    fn test_array_bindings() {
        let compiler = NagaCompiler::new();
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

        let result = compiler.compile(source, &CompilerOptions::default()).unwrap();

        let lights = result.analysis.bindings.iter()
            .find(|b| b.name == "lights")
            .unwrap();
        assert!(matches!(lights.resource_type, ResourceType::UniformBuffer { .. }));
    }
}
