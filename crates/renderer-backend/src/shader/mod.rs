//! Shader processing utilities for TRINITY.
//!
//! This module provides preprocessing, compilation, reflection, and caching
//! utilities for WGSL shaders:
//!
//! - [`WgslPreprocessor`]: C-style preprocessor for WGSL shaders
//!   - `#define`, `#undef`, `#ifdef`, `#ifndef`, `#if`, `#elif`, `#else`, `#endif`
//!   - `#include` with search path resolution
//!   - `#warning` and `#error` directives
//!   - Predefined macros (`TRINITY_VERSION`, `TRINITY_RHI_*`)
//!   - Dependency extraction for incremental compilation
//!   - Serializable state for deterministic recompilation
//!
//! - [`NagaCompiler`]: naga-based WGSL compilation pipeline
//!   - Parse WGSL to naga IR with source location spans
//!   - Validate type checking, binding conflicts, entry points
//!   - Analyze resource bindings, push constants, entry points
//!   - Transpile to SPIR-V (Vulkan) or MSL (Metal)
//!
//! - [`ShaderReflector`]: Shader reflection engine (T-AS-3.3)
//!   - Extract detailed resource bindings with member layouts
//!   - Push constant ranges with struct member information
//!   - Specialization constant discovery and overrides
//!   - Pipeline layout generation for Vulkan, Metal, D3D12
//!
//! - [`ShaderCache3L`]: 3-level hierarchical shader cache (T-AS-3.4)
//!   - In-memory LRU cache (<1ms lookup, 512 MB configurable)
//!   - Disk cache with content-addressed storage (<10ms lookup)
//!   - PAK archive support for pre-compiled common shaders
//!   - Cache key includes source hash, defines, platform, compiler version
//!
//! - [`ShaderDependencyGraph`]: Shader dependency extraction (T-AS-3.5)
//!   - Direct `#include` extraction and resolution
//!   - Transitive dependency tree construction
//!   - `@import` module reference parsing
//!   - Material DSL file tracking
//!   - Content hash per dependency for invalidation
//!   - Invalidation propagation to all dependents

pub mod cache;
pub mod dependencies;
pub mod hot_reload;
pub mod naga_compiler;
pub mod reflection;
pub mod wgsl_preprocessor;

pub use wgsl_preprocessor::{
    PreprocessError, PreprocessResult, PreprocessorState, RhiBackend, WgslPreprocessor,
};

pub use naga_compiler::{
    compile_with_preprocessing, CompileError, CompileErrorKind, CompileResult, CompilerOptions,
    EntryPointInfo, MslOptions, MslPlatform, NagaCompiler, PushConstant, ResourceBinding,
    ResourceType, ShaderAnalysis, ShaderInput, ShaderOutput, ShaderStage, SourceSpan,
    SpecializationConstant, SpirVOptions, StorageAccess, TargetBackend, TextureDimension,
};

pub use reflection::{
    BindingKind, BuiltinType, D3D12DescriptorRange, D3D12ParameterType, D3D12RangeType,
    D3D12RootConstants, D3D12RootDescriptor, D3D12RootParameter, D3D12RootSignature,
    D3D12ShaderVisibility, D3D12StaticSampler, DescriptorBinding, DescriptorSetLayout,
    DescriptorType, MatrixDimensions, MetalArgument, MetalArgumentBuffer, MetalArgumentType,
    MetalBufferBinding, MetalPipelineLayout, MetalTextureType, OutputAttribute,
    ReflectedBinding, ReflectedEntryPoint, ReflectedPushConstant, ReflectedSpecConstant,
    ReflectionOptions, ReflectionResult, ScalarType, ShaderReflector, SpecConstantOverride,
    SpecConstantValue, StorageTextureAccess, StructMember, TextureFormat, TypeLayout,
    VectorSize, VertexAttribute, VkPushConstantRange, VkShaderStageFlags, VulkanPipelineLayout,
};

pub use cache::{
    CacheConfig, CacheError, CacheKey, CacheKeyBuilder, CacheStats, MemoryCacheStats,
    PakArchive, PakEntry, ShaderCache3L, TargetPlatform,
};

pub use dependencies::{DependencyError, DependencyNode, ShaderDependencyGraph};

pub use hot_reload::{
    PipelineState, RecompileRequest, RecompileResult, ShaderHotReload, ShaderId,
};
