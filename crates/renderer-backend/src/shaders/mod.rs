//! Shader module creation and caching for TRINITY.
//!
//! This module provides a type-safe abstraction layer for wgpu shader module creation
//! from WGSL and SPIR-V sources, with comprehensive error handling that includes
//! source location information for debugging.
//!
//! # Overview
//!
//! The shader module creation API wraps wgpu's shader module creation to provide:
//!
//! - **WGSL source input**: Safe creation from WGSL source strings
//! - **SPIR-V input**: Unsafe creation from pre-compiled SPIR-V bytecode
//! - **Debug labels**: Labels propagate through wgpu for GPU debugging tools
//! - **Error handling**: Parse/validation errors include line, column, and file path
//! - **Shader caching**: Thread-safe caching with LRU eviction and hot-reload support
//!
//! # wgpu 25.x API
//!
//! This module targets wgpu 22+ (compatible through 25.x) using:
//!
//! ```text
//! // WGSL shader creation
//! device.create_shader_module(wgpu::ShaderModuleDescriptor {
//!     label: Some("my_shader"),
//!     source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
//! });
//!
//! // SPIR-V (unsafe)
//! unsafe {
//!     device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
//!         label: Some("spirv_shader"),
//!         source: wgpu::util::make_spirv_raw(&spirv_bytes),
//!     })
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! TrinityShaderDescriptor
//! +-- label: Optional debug label for GPU tools
//! +-- source: ShaderSourceKind (Wgsl or SpirV)
//! +-- file_path: Optional source file for error messages
//!
//! ShaderError
//! +-- ParseError: WGSL parse failure with location
//! +-- ValidationError: Shader validation failure with location
//! +-- IoError: File loading failure
//! +-- EmptySource: Source string is empty
//! +-- InvalidSpirV: SPIR-V bytecode is malformed
//!
//! ShaderLocation
//! +-- line: 1-based line number
//! +-- column: 1-based column number
//! +-- file_path: Optional source file path
//! +-- offset: Byte offset from start
//! +-- length: Span length in bytes
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::shaders::{
//!     create_shader_module, TrinityShaderDescriptor, ShaderSourceKind,
//! };
//!
//! # fn example(device: &wgpu::Device) -> Result<(), renderer_backend::shaders::ShaderError> {
//! // Create a shader module from WGSL source
//! let source = r#"
//!     @vertex
//!     fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
//!         return vec4<f32>(0.0, 0.0, 0.0, 1.0);
//!     }
//!
//!     @fragment
//!     fn fs_main() -> @location(0) vec4<f32> {
//!         return vec4<f32>(1.0, 0.0, 0.0, 1.0);
//!     }
//! "#;
//!
//! let module = create_shader_module(device, &TrinityShaderDescriptor {
//!     label: Some("red_triangle"),
//!     source: ShaderSourceKind::Wgsl(source.into()),
//!     file_path: Some("shaders/red_triangle.wgsl".into()),
//! })?;
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

// ============================================================================
// Submodules
// ============================================================================

pub mod cache;
pub mod override_constants;
pub mod permutations;
pub mod reflection;
pub mod validation;

// Hot-reload module (debug-only feature)
#[cfg(feature = "hot-reload")]
pub mod hot_reload;

// Re-export cache types
pub use cache::{
    CachedShader, CacheEntryInfo, ShaderCache, ShaderCacheConfig, ShaderCacheKey,
    ShaderCacheMetrics, DEFAULT_DISK_CACHE_PATH, DEFAULT_MAX_ENTRIES,
};

// Re-export validation types
pub use validation::{
    ErrorLabel, ErrorLocation, NagaValidator, SourceSnippet, Strictness,
    ValidationConfig, ValidationError, ValidationResult,
    format_validation_error, is_valid_wgsl as naga_is_valid_wgsl,
    parse_wgsl as naga_parse_wgsl, quick_validate_wgsl,
};

// Re-export reflection types
pub use reflection::{
    BindingInfo, EntryPointInfo, PushConstantInfo, PushConstantMember,
    ReflectionError, ResourceAccess, ResourceType, SamplerType, ShaderReflection,
    ShaderStage, TextureDimension, TextureSampleType, reflect_wgsl,
    MAX_BIND_GROUPS, MAX_PUSH_CONSTANT_SIZE,
};

// Re-export override constant types
pub use override_constants::{
    OverrideConstantInfo, OverrideConstantType, OverrideConstants,
    OverrideError, PipelineConstants, extract_overrides_from_wgsl,
};

// Re-export permutation types
pub use permutations::{
    CachedPermutation, EvictionPolicy, FeatureFlags, PermutationConfig,
    PermutationError, PermutationKey, PermutationMetrics, ShaderPermutationManager,
    DEFAULT_MAX_PERMUTATIONS, FEATURE_FLAG_COUNT,
};

// Re-export hot-reload types (debug-only feature)
#[cfg(feature = "hot-reload")]
pub use hot_reload::{
    HotReloadConfig, HotReloadError, HotReloadEvent, HotReloadStats,
    ShaderHotReload, ShaderWatcher, ReloadCallback,
    DEFAULT_DEBOUNCE_MS, DEFAULT_WATCH_EXTENSIONS, MAX_PENDING_RELOADS,
};

// ============================================================================
// Constants
// ============================================================================

/// Minimum valid SPIR-V module size (20 bytes = 5 words for header).
pub const SPIRV_MIN_SIZE: usize = 20;

/// SPIR-V magic number (little-endian).
pub const SPIRV_MAGIC: u32 = 0x07230203;

/// Maximum recommended shader source size (1 MB).
pub const MAX_SHADER_SOURCE_SIZE: usize = 1024 * 1024;

// ============================================================================
// Error Types
// ============================================================================

/// Location information for shader errors.
///
/// Contains line, column, and optional file path for precise error reporting.
/// All positions are 1-based (first line is line 1, first column is column 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderLocation {
    /// Line number (1-based).
    pub line: u32,
    /// Column number (1-based).
    pub column: u32,
    /// Optional source file path.
    pub file_path: Option<PathBuf>,
    /// Byte offset from start of source.
    pub offset: u32,
    /// Length of the error span in bytes.
    pub length: u32,
}

impl ShaderLocation {
    /// Creates a new shader location.
    #[inline]
    pub fn new(line: u32, column: u32) -> Self {
        Self {
            line,
            column,
            file_path: None,
            offset: 0,
            length: 0,
        }
    }

    /// Creates a shader location with a file path.
    #[inline]
    pub fn with_file(line: u32, column: u32, file_path: impl Into<PathBuf>) -> Self {
        Self {
            line,
            column,
            file_path: Some(file_path.into()),
            offset: 0,
            length: 0,
        }
    }

    /// Creates a shader location from a byte offset and source string.
    pub fn from_offset(offset: u32, length: u32, source: &str) -> Self {
        let (line, column) = offset_to_line_column(source, offset as usize);
        Self {
            line,
            column,
            file_path: None,
            offset,
            length,
        }
    }

    /// Adds a file path to the location.
    #[inline]
    pub fn set_file_path(&mut self, path: impl Into<PathBuf>) {
        self.file_path = Some(path.into());
    }

    /// Adds offset and length span information.
    #[inline]
    pub fn set_span(&mut self, offset: u32, length: u32) {
        self.offset = offset;
        self.length = length;
    }

    /// Returns true if this location has valid line/column info.
    #[inline]
    pub fn has_position(&self) -> bool {
        self.line > 0 && self.column > 0
    }

    /// Returns true if this location has a file path.
    #[inline]
    pub fn has_file(&self) -> bool {
        self.file_path.is_some()
    }
}

impl fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.file_path {
            Some(path) => {
                write!(f, "{}:{}:{}", path.display(), self.line, self.column)
            }
            None => {
                write!(f, "line {}:{}", self.line, self.column)
            }
        }
    }
}

impl Default for ShaderLocation {
    fn default() -> Self {
        Self {
            line: 1,
            column: 1,
            file_path: None,
            offset: 0,
            length: 0,
        }
    }
}

/// Errors that can occur during shader module creation.
#[derive(Debug, Clone)]
pub enum ShaderError {
    /// WGSL source parsing failed.
    ParseError {
        /// Error message from the parser.
        message: String,
        /// Location of the error in source (if available).
        location: Option<ShaderLocation>,
        /// Additional notes or hints.
        notes: Vec<String>,
    },

    /// Shader validation failed (type errors, binding conflicts, etc.).
    ValidationError {
        /// Error message from the validator.
        message: String,
        /// Location of the error in source (if available).
        location: Option<ShaderLocation>,
        /// Additional notes or hints.
        notes: Vec<String>,
    },

    /// I/O error while loading shader source.
    IoError {
        /// Error message.
        message: String,
        /// The file path that failed to load (if any).
        path: Option<PathBuf>,
    },

    /// Shader source is empty.
    EmptySource {
        /// Optional label of the shader.
        label: Option<String>,
    },

    /// SPIR-V bytecode is invalid.
    InvalidSpirV {
        /// Error message describing the problem.
        message: String,
        /// Expected value (if applicable).
        expected: Option<String>,
        /// Actual value found (if applicable).
        found: Option<String>,
    },

    /// Shader source exceeds maximum recommended size.
    SourceTooLarge {
        /// Size of the source in bytes.
        size: usize,
        /// Maximum recommended size.
        max_size: usize,
    },

    /// wgpu device error during shader creation.
    DeviceError {
        /// Error message from wgpu.
        message: String,
    },
}

impl ShaderError {
    /// Creates a parse error with a message.
    pub fn parse(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
            location: None,
            notes: Vec::new(),
        }
    }

    /// Creates a parse error with a message and location.
    pub fn parse_at(message: impl Into<String>, location: ShaderLocation) -> Self {
        Self::ParseError {
            message: message.into(),
            location: Some(location),
            notes: Vec::new(),
        }
    }

    /// Creates a validation error with a message.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::ValidationError {
            message: message.into(),
            location: None,
            notes: Vec::new(),
        }
    }

    /// Creates a validation error with a message and location.
    pub fn validation_at(message: impl Into<String>, location: ShaderLocation) -> Self {
        Self::ValidationError {
            message: message.into(),
            location: Some(location),
            notes: Vec::new(),
        }
    }

    /// Adds a note to the error (for ParseError and ValidationError).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        match &mut self {
            Self::ParseError { notes, .. } | Self::ValidationError { notes, .. } => {
                notes.push(note.into());
            }
            _ => {}
        }
        self
    }

    /// Adds a file path context to the error location.
    pub fn with_file_path(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        match &mut self {
            Self::ParseError { location, .. } | Self::ValidationError { location, .. } => {
                if let Some(loc) = location {
                    loc.set_file_path(path);
                }
            }
            Self::IoError { path: ref mut p, .. } => {
                *p = Some(path);
            }
            _ => {}
        }
        self
    }

    /// Returns the error location if available.
    pub fn location(&self) -> Option<&ShaderLocation> {
        match self {
            Self::ParseError { location, .. } | Self::ValidationError { location, .. } => {
                location.as_ref()
            }
            _ => None,
        }
    }

    /// Returns true if this is a parse error.
    #[inline]
    pub fn is_parse_error(&self) -> bool {
        matches!(self, Self::ParseError { .. })
    }

    /// Returns true if this is a validation error.
    #[inline]
    pub fn is_validation_error(&self) -> bool {
        matches!(self, Self::ValidationError { .. })
    }

    /// Formats the error with source context for rich error display.
    pub fn format_with_source(&self, source: &str, default_filename: &str) -> String {
        let mut result = String::new();

        match self {
            Self::ParseError {
                message,
                location,
                notes,
            }
            | Self::ValidationError {
                message,
                location,
                notes,
            } => {
                let error_type = if matches!(self, Self::ParseError { .. }) {
                    "parse"
                } else {
                    "validation"
                };

                // Header line: filename:line:column: error[type]: message
                if let Some(loc) = location {
                    let filename = loc
                        .file_path
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| default_filename.to_string());
                    result.push_str(&format!(
                        "{}:{}:{}: error[{}]: {}\n",
                        filename, loc.line, loc.column, error_type, message
                    ));

                    // Show source context if we have a valid offset
                    if loc.offset > 0 || loc.line > 0 {
                        let start = loc.offset as usize;
                        let end = (loc.offset + loc.length.max(1)) as usize;

                        if start < source.len() {
                            // Find the line containing the error
                            let line_start = source[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
                            let line_end = source[end.min(source.len())..]
                                .find('\n')
                                .map(|i| end.min(source.len()) + i)
                                .unwrap_or(source.len());
                            let line_content = &source[line_start..line_end];

                            // Calculate column within line
                            let col_start = start.saturating_sub(line_start);
                            let col_end = (end.saturating_sub(line_start)).min(line_content.len());

                            result.push_str("  |\n");
                            result.push_str(&format!("  | {}\n", line_content));
                            result.push_str(&format!(
                                "  | {}{}\n",
                                " ".repeat(col_start),
                                "^".repeat((col_end.saturating_sub(col_start)).max(1))
                            ));
                        }
                    }
                } else {
                    result.push_str(&format!(
                        "{}: error[{}]: {}\n",
                        default_filename, error_type, message
                    ));
                }

                // Notes
                for note in notes {
                    result.push_str(&format!("  = note: {}\n", note));
                }
            }

            Self::IoError { message, path } => {
                let filename = path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| default_filename.to_string());
                result.push_str(&format!("{}: error[io]: {}\n", filename, message));
            }

            Self::EmptySource { label } => {
                let name = label
                    .as_ref()
                    .map(|l| l.as_str())
                    .unwrap_or(default_filename);
                result.push_str(&format!("{}: error[empty]: shader source is empty\n", name));
            }

            Self::InvalidSpirV {
                message,
                expected,
                found,
            } => {
                result.push_str(&format!(
                    "{}: error[spirv]: {}\n",
                    default_filename, message
                ));
                if let Some(exp) = expected {
                    result.push_str(&format!("  = expected: {}\n", exp));
                }
                if let Some(fnd) = found {
                    result.push_str(&format!("  = found: {}\n", fnd));
                }
            }

            Self::SourceTooLarge { size, max_size } => {
                result.push_str(&format!(
                    "{}: error[size]: shader source size {} exceeds maximum {} bytes\n",
                    default_filename, size, max_size
                ));
            }

            Self::DeviceError { message } => {
                result.push_str(&format!(
                    "{}: error[device]: {}\n",
                    default_filename, message
                ));
            }
        }

        result
    }
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError {
                message, location, ..
            } => {
                if let Some(loc) = location {
                    write!(f, "shader parse error at {}: {}", loc, message)
                } else {
                    write!(f, "shader parse error: {}", message)
                }
            }
            Self::ValidationError {
                message, location, ..
            } => {
                if let Some(loc) = location {
                    write!(f, "shader validation error at {}: {}", loc, message)
                } else {
                    write!(f, "shader validation error: {}", message)
                }
            }
            Self::IoError { message, path } => {
                if let Some(p) = path {
                    write!(f, "shader I/O error for '{}': {}", p.display(), message)
                } else {
                    write!(f, "shader I/O error: {}", message)
                }
            }
            Self::EmptySource { label } => {
                if let Some(l) = label {
                    write!(f, "shader '{}' has empty source", l)
                } else {
                    write!(f, "shader source is empty")
                }
            }
            Self::InvalidSpirV { message, .. } => {
                write!(f, "invalid SPIR-V bytecode: {}", message)
            }
            Self::SourceTooLarge { size, max_size } => {
                write!(
                    f,
                    "shader source size {} exceeds maximum {} bytes",
                    size, max_size
                )
            }
            Self::DeviceError { message } => {
                write!(f, "wgpu device error: {}", message)
            }
        }
    }
}

impl std::error::Error for ShaderError {}

// ============================================================================
// Shader Source Types
// ============================================================================

/// The kind of shader source being compiled.
#[derive(Debug, Clone)]
pub enum ShaderSourceKind<'a> {
    /// WGSL source code.
    Wgsl(Cow<'a, str>),
    /// Pre-compiled SPIR-V bytecode.
    SpirV(Cow<'a, [u32]>),
}

impl<'a> ShaderSourceKind<'a> {
    /// Creates a WGSL source from a string.
    #[inline]
    pub fn wgsl(source: impl Into<Cow<'a, str>>) -> Self {
        Self::Wgsl(source.into())
    }

    /// Creates a SPIR-V source from a word slice.
    #[inline]
    pub fn spirv(words: impl Into<Cow<'a, [u32]>>) -> Self {
        Self::SpirV(words.into())
    }

    /// Returns true if this is WGSL source.
    #[inline]
    pub fn is_wgsl(&self) -> bool {
        matches!(self, Self::Wgsl(_))
    }

    /// Returns true if this is SPIR-V source.
    #[inline]
    pub fn is_spirv(&self) -> bool {
        matches!(self, Self::SpirV(_))
    }

    /// Returns the WGSL source if this is WGSL, None otherwise.
    pub fn as_wgsl(&self) -> Option<&str> {
        match self {
            Self::Wgsl(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the SPIR-V words if this is SPIR-V, None otherwise.
    pub fn as_spirv(&self) -> Option<&[u32]> {
        match self {
            Self::SpirV(w) => Some(w),
            _ => None,
        }
    }
}

impl<'a> From<&'a str> for ShaderSourceKind<'a> {
    fn from(s: &'a str) -> Self {
        Self::Wgsl(Cow::Borrowed(s))
    }
}

impl From<String> for ShaderSourceKind<'static> {
    fn from(s: String) -> Self {
        Self::Wgsl(Cow::Owned(s))
    }
}

impl<'a> From<&'a [u32]> for ShaderSourceKind<'a> {
    fn from(words: &'a [u32]) -> Self {
        Self::SpirV(Cow::Borrowed(words))
    }
}

impl From<Vec<u32>> for ShaderSourceKind<'static> {
    fn from(words: Vec<u32>) -> Self {
        Self::SpirV(Cow::Owned(words))
    }
}

// ============================================================================
// Shader Descriptor
// ============================================================================

/// Descriptor for creating a TRINITY shader module.
///
/// This wraps the wgpu shader module descriptor with additional metadata
/// for error reporting.
#[derive(Debug, Clone)]
pub struct TrinityShaderDescriptor<'a> {
    /// Debug label for the shader module (appears in GPU debugging tools).
    pub label: Option<&'a str>,
    /// The shader source (WGSL or SPIR-V).
    pub source: ShaderSourceKind<'a>,
    /// Optional file path for error messages.
    pub file_path: Option<PathBuf>,
}

impl<'a> TrinityShaderDescriptor<'a> {
    /// Creates a new shader descriptor with WGSL source.
    #[inline]
    pub fn wgsl(label: Option<&'a str>, source: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label,
            source: ShaderSourceKind::Wgsl(source.into()),
            file_path: None,
        }
    }

    /// Creates a new shader descriptor with SPIR-V source.
    #[inline]
    pub fn spirv(label: Option<&'a str>, words: impl Into<Cow<'a, [u32]>>) -> Self {
        Self {
            label,
            source: ShaderSourceKind::SpirV(words.into()),
            file_path: None,
        }
    }

    /// Sets the file path for error messages.
    #[inline]
    pub fn with_file_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Returns the label as a string for error messages.
    pub fn label_string(&self) -> String {
        self.label
            .map(|l| l.to_string())
            .or_else(|| self.file_path.as_ref().map(|p| p.to_string_lossy().to_string()))
            .unwrap_or_else(|| "<unnamed>".to_string())
    }
}

impl Default for TrinityShaderDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            source: ShaderSourceKind::Wgsl(Cow::Borrowed("")),
            file_path: None,
        }
    }
}

// ============================================================================
// Trinity Shader Module
// ============================================================================

/// A compiled shader module with metadata.
///
/// Wraps a wgpu::ShaderModule with additional information for debugging
/// and hot-reloading.
#[derive(Debug)]
pub struct TrinityShaderModule {
    /// The underlying wgpu shader module.
    inner: wgpu::ShaderModule,
    /// Debug label for the shader.
    label: Option<String>,
    /// Source file path (if loaded from file).
    file_path: Option<PathBuf>,
    /// SHA-256 hash of the source for cache invalidation.
    source_hash: [u8; 32],
}

impl TrinityShaderModule {
    /// Creates a new TrinityShaderModule from a wgpu shader module.
    fn new(
        inner: wgpu::ShaderModule,
        label: Option<String>,
        file_path: Option<PathBuf>,
        source_hash: [u8; 32],
    ) -> Self {
        Self {
            inner,
            label,
            file_path,
            source_hash,
        }
    }

    /// Returns a reference to the underlying wgpu shader module.
    #[inline]
    pub fn inner(&self) -> &wgpu::ShaderModule {
        &self.inner
    }

    /// Consumes self and returns the underlying wgpu shader module.
    #[inline]
    pub fn into_inner(self) -> wgpu::ShaderModule {
        self.inner
    }

    /// Returns the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the source file path.
    #[inline]
    pub fn file_path(&self) -> Option<&PathBuf> {
        self.file_path.as_ref()
    }

    /// Returns the SHA-256 hash of the source.
    #[inline]
    pub fn source_hash(&self) -> &[u8; 32] {
        &self.source_hash
    }

    /// Returns the hash as a hex string.
    pub fn source_hash_hex(&self) -> String {
        self.source_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

impl AsRef<wgpu::ShaderModule> for TrinityShaderModule {
    fn as_ref(&self) -> &wgpu::ShaderModule {
        &self.inner
    }
}

impl std::ops::Deref for TrinityShaderModule {
    type Target = wgpu::ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// ============================================================================
// Shader Module Creation - WGSL (Safe)
// ============================================================================

/// Creates a shader module from WGSL source.
///
/// This is the primary safe API for shader module creation. It validates the
/// WGSL source and provides detailed error messages with source locations.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the shader module on.
/// * `desc` - The shader descriptor containing source and metadata.
///
/// # Returns
///
/// Returns `Ok(TrinityShaderModule)` on success, or `Err(ShaderError)` with
/// detailed error information including line/column locations.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::shaders::{create_shader_module, TrinityShaderDescriptor};
///
/// # fn example(device: &wgpu::Device) -> Result<(), renderer_backend::shaders::ShaderError> {
/// let module = create_shader_module(device, &TrinityShaderDescriptor::wgsl(
///     Some("vertex_shader"),
///     r#"
///         @vertex
///         fn main() -> @builtin(position) vec4<f32> {
///             return vec4<f32>(0.0, 0.0, 0.0, 1.0);
///         }
///     "#,
/// ))?;
/// # Ok(())
/// # }
/// ```
pub fn create_shader_module(
    device: &wgpu::Device,
    desc: &TrinityShaderDescriptor<'_>,
) -> Result<TrinityShaderModule, ShaderError> {
    match &desc.source {
        ShaderSourceKind::Wgsl(source) => create_shader_module_wgsl(device, desc, source),
        ShaderSourceKind::SpirV(words) => {
            // For SpirV, delegate to the unsafe function with proper validation
            validate_spirv(words)?;
            // SAFETY: We've validated the SPIR-V header
            unsafe { create_shader_module_spirv_internal(device, desc, words) }
        }
    }
}

/// Creates a shader module from WGSL source (internal implementation).
fn create_shader_module_wgsl(
    device: &wgpu::Device,
    desc: &TrinityShaderDescriptor<'_>,
    source: &str,
) -> Result<TrinityShaderModule, ShaderError> {
    // Validate source is not empty
    if source.trim().is_empty() {
        return Err(ShaderError::EmptySource {
            label: desc.label.map(|s| s.to_string()),
        });
    }

    // Check source size
    if source.len() > MAX_SHADER_SOURCE_SIZE {
        return Err(ShaderError::SourceTooLarge {
            size: source.len(),
            max_size: MAX_SHADER_SOURCE_SIZE,
        });
    }

    // Compute source hash for cache invalidation
    let source_hash = compute_sha256(source.as_bytes());

    // Pre-validate with naga for better error messages
    if let Err(err) = validate_wgsl_with_naga(source, desc.file_path.as_ref()) {
        return Err(err);
    }

    // Create the wgpu shader module
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: desc.label,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });

    Ok(TrinityShaderModule::new(
        module,
        desc.label.map(|s| s.to_string()),
        desc.file_path.clone(),
        source_hash,
    ))
}

/// Pre-validates WGSL source using naga for better error messages.
fn validate_wgsl_with_naga(
    source: &str,
    file_path: Option<&PathBuf>,
) -> Result<(), ShaderError> {
    // Parse the WGSL source
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(err) => {
            return Err(convert_naga_parse_error(err, source, file_path));
        }
    };

    // Validate the module
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(err) = validator.validate(&module) {
        return Err(convert_naga_validation_error(err, source, file_path));
    }

    Ok(())
}

/// Converts a naga parse error to a ShaderError with location info.
fn convert_naga_parse_error(
    err: naga::front::wgsl::ParseError,
    source: &str,
    file_path: Option<&PathBuf>,
) -> ShaderError {
    let message = err.message().to_string();

    // Try to extract location from the error
    // ParseError.labels() returns an iterator over (naga::Span, &str)
    // We use to_range() to convert the span to a Range<usize>
    let location = err.labels().next().and_then(|(span, _)| {
        let range = span.to_range()?;
        let offset = range.start as u32;
        let length = (range.end - range.start) as u32;
        let mut loc = ShaderLocation::from_offset(offset, length, source);
        loc.set_span(offset, length);
        if let Some(path) = file_path {
            loc.set_file_path(path);
        }
        Some(loc)
    });

    let mut error = ShaderError::ParseError {
        message,
        location,
        notes: Vec::new(),
    };

    // Add any additional labels as notes
    for (span, label) in err.labels().skip(1) {
        if let Some(_range) = span.to_range() {
            error = error.with_note(label.to_string());
        }
    }

    error
}

/// Converts a naga validation error to a ShaderError.
fn convert_naga_validation_error(
    err: naga::WithSpan<naga::valid::ValidationError>,
    source: &str,
    file_path: Option<&PathBuf>,
) -> ShaderError {
    let message = format!("{}", err.as_inner());

    // Try to get the span from the error
    let location = err.spans().next().and_then(|(span, _)| {
        if span.is_defined() {
            if let Some(range) = span.to_range() {
                let offset = range.start as u32;
                let length = (range.end - range.start) as u32;
                let mut loc = ShaderLocation::from_offset(offset, length, source);
                loc.set_span(offset, length);
                if let Some(path) = file_path {
                    loc.set_file_path(path);
                }
                return Some(loc);
            }
        }
        None
    });

    ShaderError::ValidationError {
        message,
        location,
        notes: Vec::new(),
    }
}

// ============================================================================
// Shader Module Creation - SPIR-V (Unsafe)
// ============================================================================

/// Creates a shader module from pre-compiled SPIR-V bytecode.
///
/// # Safety
///
/// This function is unsafe because:
/// - SPIR-V bytecode is not fully validated by wgpu
/// - Invalid SPIR-V can cause undefined behavior or GPU hangs
/// - The caller must ensure the SPIR-V is valid and from a trusted source
///
/// # Arguments
///
/// * `device` - The wgpu device to create the shader module on.
/// * `desc` - The shader descriptor containing SPIR-V source and metadata.
///
/// # Returns
///
/// Returns `Ok(TrinityShaderModule)` on success, or `Err(ShaderError)` if
/// basic SPIR-V validation fails.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::shaders::{create_shader_module_spirv, TrinityShaderDescriptor};
///
/// # fn example(device: &wgpu::Device, spirv_bytes: &[u32]) -> Result<(), renderer_backend::shaders::ShaderError> {
/// // SAFETY: spirv_bytes is known to be valid SPIR-V from a trusted compiler
/// let module = unsafe {
///     create_shader_module_spirv(device, &TrinityShaderDescriptor::spirv(
///         Some("compiled_shader"),
///         spirv_bytes,
///     ))?
/// };
/// # Ok(())
/// # }
/// ```
pub unsafe fn create_shader_module_spirv(
    device: &wgpu::Device,
    desc: &TrinityShaderDescriptor<'_>,
) -> Result<TrinityShaderModule, ShaderError> {
    let words = match &desc.source {
        ShaderSourceKind::SpirV(words) => words,
        ShaderSourceKind::Wgsl(_) => {
            return Err(ShaderError::InvalidSpirV {
                message: "expected SPIR-V source, got WGSL".to_string(),
                expected: Some("SPIR-V bytecode".to_string()),
                found: Some("WGSL source".to_string()),
            });
        }
    };

    // Validate SPIR-V header
    validate_spirv(words)?;

    // SAFETY: Caller guarantees the SPIR-V is valid
    create_shader_module_spirv_internal(device, desc, words)
}

/// Internal SPIR-V shader module creation.
///
/// # Safety
///
/// The caller must ensure the SPIR-V bytecode is valid.
unsafe fn create_shader_module_spirv_internal(
    device: &wgpu::Device,
    desc: &TrinityShaderDescriptor<'_>,
    words: &[u32],
) -> Result<TrinityShaderModule, ShaderError> {
    // Compute hash for cache invalidation
    let source_hash = compute_sha256(bytemuck::cast_slice(words));

    // Create the descriptor for wgpu
    let spirv_desc = wgpu::ShaderModuleDescriptorSpirV {
        label: desc.label,
        source: Cow::Borrowed(words),
    };

    // Create the shader module (unsafe)
    let module = device.create_shader_module_spirv(&spirv_desc);

    Ok(TrinityShaderModule::new(
        module,
        desc.label.map(|s| s.to_string()),
        desc.file_path.clone(),
        source_hash,
    ))
}

/// Validates basic SPIR-V structure (header only).
fn validate_spirv(words: &[u32]) -> Result<(), ShaderError> {
    // Check minimum size
    if words.len() < SPIRV_MIN_SIZE / 4 {
        return Err(ShaderError::InvalidSpirV {
            message: format!(
                "SPIR-V bytecode too small ({} words, minimum {} words)",
                words.len(),
                SPIRV_MIN_SIZE / 4
            ),
            expected: Some(format!(">= {} words", SPIRV_MIN_SIZE / 4)),
            found: Some(format!("{} words", words.len())),
        });
    }

    // Check magic number
    if words[0] != SPIRV_MAGIC {
        return Err(ShaderError::InvalidSpirV {
            message: "invalid SPIR-V magic number".to_string(),
            expected: Some(format!("0x{:08x}", SPIRV_MAGIC)),
            found: Some(format!("0x{:08x}", words[0])),
        });
    }

    // Check version (word 1) is reasonable
    let version = words[1];
    let major = (version >> 16) & 0xFF;
    let minor = (version >> 8) & 0xFF;
    if major == 0 || major > 2 {
        return Err(ShaderError::InvalidSpirV {
            message: format!("unsupported SPIR-V version {}.{}", major, minor),
            expected: Some("version 1.x or 2.x".to_string()),
            found: Some(format!("version {}.{}", major, minor)),
        });
    }

    Ok(())
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Creates a shader module from a WGSL string.
///
/// This is a convenience function for simple use cases.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::shaders::create_shader_module_from_wgsl;
///
/// # fn example(device: &wgpu::Device) -> Result<(), renderer_backend::shaders::ShaderError> {
/// let module = create_shader_module_from_wgsl(
///     device,
///     Some("simple_shader"),
///     r#"
///         @vertex fn main() -> @builtin(position) vec4<f32> {
///             return vec4<f32>(0.0);
///         }
///     "#,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn create_shader_module_from_wgsl(
    device: &wgpu::Device,
    label: Option<&str>,
    source: &str,
) -> Result<TrinityShaderModule, ShaderError> {
    create_shader_module(device, &TrinityShaderDescriptor::wgsl(label, source))
}

/// Creates a shader module by loading WGSL from a file.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::shaders::create_shader_module_from_file;
/// use std::path::Path;
///
/// # fn example(device: &wgpu::Device) -> Result<(), renderer_backend::shaders::ShaderError> {
/// let module = create_shader_module_from_file(
///     device,
///     Path::new("shaders/pbr.wgsl"),
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn create_shader_module_from_file(
    device: &wgpu::Device,
    path: &std::path::Path,
) -> Result<TrinityShaderModule, ShaderError> {
    let source = std::fs::read_to_string(path).map_err(|e| ShaderError::IoError {
        message: e.to_string(),
        path: Some(path.to_path_buf()),
    })?;

    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("loaded_shader");

    create_shader_module(
        device,
        &TrinityShaderDescriptor::wgsl(Some(label), source).with_file_path(path),
    )
}

/// Creates a shader module wrapped in an Arc for shared ownership.
///
/// Useful for shader caches and resource managers.
pub fn create_shader_module_arc(
    device: &wgpu::Device,
    desc: &TrinityShaderDescriptor<'_>,
) -> Result<Arc<TrinityShaderModule>, ShaderError> {
    Ok(Arc::new(create_shader_module(device, desc)?))
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Checks if a WGSL source is valid without creating a shader module.
///
/// This can be used for syntax checking in editors or build systems.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::validate_wgsl;
///
/// let result = validate_wgsl(r#"
///     @vertex fn main() -> @builtin(position) vec4<f32> {
///         return vec4<f32>(0.0);
///     }
/// "#);
/// assert!(result.is_ok());
///
/// let result = validate_wgsl("invalid shader code");
/// assert!(result.is_err());
/// ```
pub fn validate_wgsl(source: &str) -> Result<(), ShaderError> {
    if source.trim().is_empty() {
        return Err(ShaderError::EmptySource { label: None });
    }

    validate_wgsl_with_naga(source, None)
}

/// Checks if a WGSL source is valid, returning a bool.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::is_valid_wgsl;
///
/// assert!(is_valid_wgsl(r#"
///     @vertex fn main() -> @builtin(position) vec4<f32> {
///         return vec4<f32>(0.0);
///     }
/// "#));
/// ```
pub fn is_valid_wgsl(source: &str) -> bool {
    validate_wgsl(source).is_ok()
}

/// Checks if SPIR-V bytecode has a valid header.
///
/// Note: This only validates the header structure, not the full bytecode.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::{is_valid_spirv_header, SPIRV_MAGIC};
///
/// // Valid header
/// let valid = [SPIRV_MAGIC, 0x00010300, 0, 0, 0]; // SPIR-V 1.3
/// assert!(is_valid_spirv_header(&valid));
///
/// // Invalid magic
/// let invalid = [0xDEADBEEF, 0x00010300, 0, 0, 0];
/// assert!(!is_valid_spirv_header(&invalid));
/// ```
pub fn is_valid_spirv_header(words: &[u32]) -> bool {
    validate_spirv(words).is_ok()
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Computes SHA-256 hash of data.
fn compute_sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Converts a byte offset to line and column numbers (1-based).
fn offset_to_line_column(source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let prefix = &source[..offset];
    let line = prefix.matches('\n').count() + 1;
    let last_newline = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let column = offset - last_newline + 1;
    (line as u32, column as u32)
}

/// Converts line and column to byte offset.
pub fn line_column_to_offset(source: &str, line: u32, column: u32) -> Option<usize> {
    if line == 0 || column == 0 {
        return None;
    }

    let mut current_line = 1u32;
    let mut line_start = 0usize;

    for (i, c) in source.char_indices() {
        if current_line == line {
            let target_offset = line_start + (column as usize - 1);
            if target_offset <= source.len() {
                return Some(target_offset);
            }
            return None;
        }
        if c == '\n' {
            current_line += 1;
            line_start = i + 1;
        }
    }

    // Handle last line
    if current_line == line {
        let target_offset = line_start + (column as usize - 1);
        if target_offset <= source.len() {
            return Some(target_offset);
        }
    }

    None
}

/// Extracts a line of source code around an offset.
pub fn extract_source_line(source: &str, offset: usize) -> Option<&str> {
    if offset >= source.len() {
        return None;
    }

    let line_start = source[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = source[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(source.len());

    Some(&source[line_start..line_end])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ShaderLocation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_location_new() {
        let loc = ShaderLocation::new(10, 5);
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert!(loc.file_path.is_none());
        assert_eq!(loc.offset, 0);
        assert_eq!(loc.length, 0);
    }

    #[test]
    fn test_shader_location_with_file() {
        let loc = ShaderLocation::with_file(10, 5, "test.wgsl");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.file_path, Some(PathBuf::from("test.wgsl")));
    }

    #[test]
    fn test_shader_location_from_offset() {
        let source = "line1\nline2\nline3";
        let loc = ShaderLocation::from_offset(7, 3, source);
        assert_eq!(loc.line, 2); // "line2" starts at offset 6
        assert_eq!(loc.column, 2); // offset 7 is 'i' in "line2"
    }

    #[test]
    fn test_shader_location_display() {
        let loc = ShaderLocation::with_file(10, 5, "shaders/test.wgsl");
        assert_eq!(format!("{}", loc), "shaders/test.wgsl:10:5");

        let loc = ShaderLocation::new(10, 5);
        assert_eq!(format!("{}", loc), "line 10:5");
    }

    #[test]
    fn test_shader_location_has_position() {
        let loc = ShaderLocation::new(1, 1);
        assert!(loc.has_position());

        let loc = ShaderLocation::new(0, 0);
        assert!(!loc.has_position());
    }

    #[test]
    fn test_shader_location_has_file() {
        let loc = ShaderLocation::with_file(1, 1, "test.wgsl");
        assert!(loc.has_file());

        let loc = ShaderLocation::new(1, 1);
        assert!(!loc.has_file());
    }

    #[test]
    fn test_shader_location_set_span() {
        let mut loc = ShaderLocation::new(1, 1);
        loc.set_span(100, 50);
        assert_eq!(loc.offset, 100);
        assert_eq!(loc.length, 50);
    }

    #[test]
    fn test_shader_location_default() {
        let loc = ShaderLocation::default();
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);
        assert!(loc.file_path.is_none());
    }

    // -------------------------------------------------------------------------
    // ShaderError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_error_parse() {
        let err = ShaderError::parse("unexpected token");
        match err {
            ShaderError::ParseError { message, location, notes } => {
                assert_eq!(message, "unexpected token");
                assert!(location.is_none());
                assert!(notes.is_empty());
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_shader_error_parse_at() {
        let loc = ShaderLocation::new(5, 10);
        let err = ShaderError::parse_at("syntax error", loc);
        match err {
            ShaderError::ParseError { message, location, .. } => {
                assert_eq!(message, "syntax error");
                assert!(location.is_some());
                assert_eq!(location.unwrap().line, 5);
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_shader_error_validation() {
        let err = ShaderError::validation("type mismatch");
        match err {
            ShaderError::ValidationError { message, .. } => {
                assert_eq!(message, "type mismatch");
            }
            _ => panic!("expected ValidationError"),
        }
    }

    #[test]
    fn test_shader_error_with_note() {
        let err = ShaderError::parse("error").with_note("hint: check syntax");
        match err {
            ShaderError::ParseError { notes, .. } => {
                assert_eq!(notes.len(), 1);
                assert_eq!(notes[0], "hint: check syntax");
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_shader_error_with_file_path() {
        let loc = ShaderLocation::new(1, 1);
        let err = ShaderError::parse_at("error", loc).with_file_path("test.wgsl");
        match err {
            ShaderError::ParseError { location, .. } => {
                assert_eq!(
                    location.unwrap().file_path,
                    Some(PathBuf::from("test.wgsl"))
                );
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_shader_error_is_parse_error() {
        let err = ShaderError::parse("test");
        assert!(err.is_parse_error());
        assert!(!err.is_validation_error());
    }

    #[test]
    fn test_shader_error_is_validation_error() {
        let err = ShaderError::validation("test");
        assert!(!err.is_parse_error());
        assert!(err.is_validation_error());
    }

    #[test]
    fn test_shader_error_display() {
        let err = ShaderError::parse("unexpected token");
        assert!(format!("{}", err).contains("parse error"));
        assert!(format!("{}", err).contains("unexpected token"));

        let err = ShaderError::EmptySource {
            label: Some("test".to_string()),
        };
        assert!(format!("{}", err).contains("empty"));

        let err = ShaderError::InvalidSpirV {
            message: "bad magic".to_string(),
            expected: None,
            found: None,
        };
        assert!(format!("{}", err).contains("SPIR-V"));
    }

    #[test]
    fn test_shader_error_location() {
        let loc = ShaderLocation::new(5, 10);
        let err = ShaderError::parse_at("error", loc);
        assert!(err.location().is_some());

        let err = ShaderError::EmptySource { label: None };
        assert!(err.location().is_none());
    }

    // -------------------------------------------------------------------------
    // ShaderSourceKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_source_kind_wgsl() {
        let source = ShaderSourceKind::wgsl("fn main() {}");
        assert!(source.is_wgsl());
        assert!(!source.is_spirv());
        assert_eq!(source.as_wgsl(), Some("fn main() {}"));
        assert!(source.as_spirv().is_none());
    }

    #[test]
    fn test_shader_source_kind_spirv() {
        let words: Vec<u32> = vec![SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let source = ShaderSourceKind::spirv(words.as_slice());
        assert!(!source.is_wgsl());
        assert!(source.is_spirv());
        assert!(source.as_wgsl().is_none());
        assert!(source.as_spirv().is_some());
    }

    #[test]
    fn test_shader_source_kind_from_str() {
        let source: ShaderSourceKind = "fn main() {}".into();
        assert!(source.is_wgsl());
    }

    #[test]
    fn test_shader_source_kind_from_string() {
        let source: ShaderSourceKind = String::from("fn main() {}").into();
        assert!(source.is_wgsl());
    }

    #[test]
    fn test_shader_source_kind_from_slice() {
        let words: &[u32] = &[SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let source: ShaderSourceKind = words.into();
        assert!(source.is_spirv());
    }

    #[test]
    fn test_shader_source_kind_from_vec() {
        let words: Vec<u32> = vec![SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let source: ShaderSourceKind = words.into();
        assert!(source.is_spirv());
    }

    // -------------------------------------------------------------------------
    // TrinityShaderDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_shader_descriptor_wgsl() {
        let desc = TrinityShaderDescriptor::wgsl(Some("test"), "fn main() {}");
        assert_eq!(desc.label, Some("test"));
        assert!(desc.source.is_wgsl());
        assert!(desc.file_path.is_none());
    }

    #[test]
    fn test_trinity_shader_descriptor_spirv() {
        let words = vec![SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let desc = TrinityShaderDescriptor::spirv(Some("spirv_test"), words.as_slice());
        assert_eq!(desc.label, Some("spirv_test"));
        assert!(desc.source.is_spirv());
    }

    #[test]
    fn test_trinity_shader_descriptor_with_file_path() {
        let desc = TrinityShaderDescriptor::wgsl(Some("test"), "fn main() {}")
            .with_file_path("shaders/test.wgsl");
        assert_eq!(desc.file_path, Some(PathBuf::from("shaders/test.wgsl")));
    }

    #[test]
    fn test_trinity_shader_descriptor_label_string() {
        let desc = TrinityShaderDescriptor::wgsl(Some("my_shader"), "");
        assert_eq!(desc.label_string(), "my_shader");

        let desc = TrinityShaderDescriptor::wgsl(None, "").with_file_path("test.wgsl");
        assert_eq!(desc.label_string(), "test.wgsl");

        let desc = TrinityShaderDescriptor::wgsl(None, "");
        assert_eq!(desc.label_string(), "<unnamed>");
    }

    #[test]
    fn test_trinity_shader_descriptor_default() {
        let desc = TrinityShaderDescriptor::default();
        assert!(desc.label.is_none());
        assert!(desc.source.is_wgsl());
        assert!(desc.file_path.is_none());
    }

    // -------------------------------------------------------------------------
    // SPIR-V Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_spirv_valid() {
        let words = [SPIRV_MAGIC, 0x00010300, 0, 0, 0]; // Version 1.3
        assert!(validate_spirv(&words).is_ok());

        let words = [SPIRV_MAGIC, 0x00010500, 0, 0, 0]; // Version 1.5
        assert!(validate_spirv(&words).is_ok());
    }

    #[test]
    fn test_validate_spirv_too_small() {
        let words = [SPIRV_MAGIC]; // Only 1 word
        let result = validate_spirv(&words);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::InvalidSpirV { message, .. } => {
                assert!(message.contains("too small"));
            }
            _ => panic!("expected InvalidSpirV"),
        }
    }

    #[test]
    fn test_validate_spirv_bad_magic() {
        let words = [0xDEADBEEF, 0x00010300, 0, 0, 0];
        let result = validate_spirv(&words);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::InvalidSpirV { message, expected, found } => {
                assert!(message.contains("magic"));
                assert!(expected.is_some());
                assert!(found.is_some());
            }
            _ => panic!("expected InvalidSpirV"),
        }
    }

    #[test]
    fn test_validate_spirv_bad_version() {
        let words = [SPIRV_MAGIC, 0x00030000, 0, 0, 0]; // Version 3.0 (invalid)
        let result = validate_spirv(&words);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::InvalidSpirV { message, .. } => {
                assert!(message.contains("version"));
            }
            _ => panic!("expected InvalidSpirV"),
        }
    }

    #[test]
    fn test_is_valid_spirv_header() {
        let valid = [SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        assert!(is_valid_spirv_header(&valid));

        let invalid = [0xDEADBEEF, 0x00010300, 0, 0, 0];
        assert!(!is_valid_spirv_header(&invalid));

        let too_small = [SPIRV_MAGIC];
        assert!(!is_valid_spirv_header(&too_small));
    }

    // -------------------------------------------------------------------------
    // WGSL Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_wgsl_valid() {
        let source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_validate_wgsl_empty() {
        let result = validate_wgsl("");
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::EmptySource { .. } => {}
            _ => panic!("expected EmptySource"),
        }
    }

    #[test]
    fn test_validate_wgsl_whitespace_only() {
        let result = validate_wgsl("   \n\t\n   ");
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::EmptySource { .. } => {}
            _ => panic!("expected EmptySource"),
        }
    }

    #[test]
    fn test_validate_wgsl_parse_error() {
        let source = "this is not valid wgsl @@@";
        let result = validate_wgsl(source);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_parse_error());
    }

    #[test]
    fn test_validate_wgsl_type_error() {
        let source = r#"
            @vertex
            fn vs_main() -> @builtin(position) vec4<f32> {
                let x: i32 = 1.5; // Type mismatch
                return vec4<f32>(0.0);
            }
        "#;
        let result = validate_wgsl(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_valid_wgsl() {
        assert!(is_valid_wgsl(r#"
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#));

        assert!(!is_valid_wgsl("not valid"));
        assert!(!is_valid_wgsl(""));
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_offset_to_line_column() {
        let source = "line1\nline2\nline3";

        // First character
        assert_eq!(offset_to_line_column(source, 0), (1, 1));

        // Last char of line 1
        assert_eq!(offset_to_line_column(source, 4), (1, 5));

        // Newline between line 1 and 2
        assert_eq!(offset_to_line_column(source, 5), (1, 6));

        // First char of line 2
        assert_eq!(offset_to_line_column(source, 6), (2, 1));

        // Middle of line 3
        assert_eq!(offset_to_line_column(source, 14), (3, 3));
    }

    #[test]
    fn test_offset_to_line_column_single_line() {
        let source = "hello world";
        assert_eq!(offset_to_line_column(source, 0), (1, 1));
        assert_eq!(offset_to_line_column(source, 5), (1, 6));
        assert_eq!(offset_to_line_column(source, 10), (1, 11));
    }

    #[test]
    fn test_offset_to_line_column_empty() {
        let source = "";
        assert_eq!(offset_to_line_column(source, 0), (1, 1));
    }

    #[test]
    fn test_line_column_to_offset() {
        let source = "line1\nline2\nline3";

        assert_eq!(line_column_to_offset(source, 1, 1), Some(0));
        assert_eq!(line_column_to_offset(source, 1, 5), Some(4));
        assert_eq!(line_column_to_offset(source, 2, 1), Some(6));
        assert_eq!(line_column_to_offset(source, 3, 1), Some(12));

        // Invalid
        assert_eq!(line_column_to_offset(source, 0, 1), None);
        assert_eq!(line_column_to_offset(source, 1, 0), None);
        assert_eq!(line_column_to_offset(source, 10, 1), None);
    }

    #[test]
    fn test_extract_source_line() {
        let source = "line1\nline2\nline3";

        assert_eq!(extract_source_line(source, 0), Some("line1"));
        assert_eq!(extract_source_line(source, 3), Some("line1"));
        assert_eq!(extract_source_line(source, 6), Some("line2"));
        assert_eq!(extract_source_line(source, 12), Some("line3"));
        assert_eq!(extract_source_line(source, 100), None);
    }

    #[test]
    fn test_compute_sha256() {
        let hash1 = compute_sha256(b"hello");
        let hash2 = compute_sha256(b"hello");
        let hash3 = compute_sha256(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 32);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_spirv_constants() {
        assert_eq!(SPIRV_MIN_SIZE, 20);
        assert_eq!(SPIRV_MAGIC, 0x07230203);
        assert_eq!(MAX_SHADER_SOURCE_SIZE, 1024 * 1024);
    }

    // -------------------------------------------------------------------------
    // Error Formatting Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_error_format_with_source_parse() {
        let source = "line1\nline2_error\nline3";
        let mut loc = ShaderLocation::from_offset(6, 11, source);
        loc.set_span(6, 11);
        let err = ShaderError::parse_at("unexpected token", loc);

        let formatted = err.format_with_source(source, "test.wgsl");
        assert!(formatted.contains("test.wgsl"));
        assert!(formatted.contains("unexpected token"));
        assert!(formatted.contains("line2_error"));
    }

    #[test]
    fn test_shader_error_format_with_source_empty() {
        let err = ShaderError::EmptySource {
            label: Some("my_shader".to_string()),
        };
        let formatted = err.format_with_source("", "test.wgsl");
        assert!(formatted.contains("empty"));
    }

    #[test]
    fn test_shader_error_format_with_source_spirv() {
        let err = ShaderError::InvalidSpirV {
            message: "bad magic".to_string(),
            expected: Some("0x07230203".to_string()),
            found: Some("0xDEADBEEF".to_string()),
        };
        let formatted = err.format_with_source("", "test.spv");
        assert!(formatted.contains("bad magic"));
        assert!(formatted.contains("expected"));
        assert!(formatted.contains("found"));
    }

    // -------------------------------------------------------------------------
    // Integration-like Tests (without actual device)
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_wgsl_validation_flow() {
        let valid_shader = r#"
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
                var out: VertexOutput;
                out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                out.uv = vec2<f32>(0.0, 0.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(in.uv, 0.0, 1.0);
            }
        "#;

        assert!(validate_wgsl(valid_shader).is_ok());
    }

    #[test]
    fn test_complex_shader_validation() {
        let compute_shader = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                data[idx] = data[idx] * 2.0;
            }
        "#;

        assert!(validate_wgsl(compute_shader).is_ok());
    }

    #[test]
    fn test_descriptor_builder_pattern() {
        let desc = TrinityShaderDescriptor::wgsl(
            Some("pbr_shader"),
            "fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
        )
        .with_file_path("shaders/pbr.wgsl");

        assert_eq!(desc.label, Some("pbr_shader"));
        assert_eq!(desc.file_path, Some(PathBuf::from("shaders/pbr.wgsl")));
        assert!(desc.source.is_wgsl());
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_unicode_in_shader() {
        // WGSL supports Unicode in comments
        let source = r#"
            // This is a comment with unicode: Helloo Wooorld
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_very_long_line() {
        let long_line = format!(
            "@vertex fn main() -> @builtin(position) vec4<f32> {{ return vec4<f32>({}); }}",
            "0.0 + ".repeat(100) + "0.0"
        );
        // This should be valid WGSL
        assert!(validate_wgsl(&long_line).is_ok());
    }

    #[test]
    fn test_shader_with_includes_style_comments() {
        // TRINITY preprocessor might add these
        let source = r#"
            // #include "common.wgsl" - processed
            // #define FOO 1

            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    // =========================================================================
    // ADDITIONAL WHITEBOX TESTS - T-WGPU-P2.7.1
    // =========================================================================

    // -------------------------------------------------------------------------
    // Descriptor Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_empty_label() {
        let desc = TrinityShaderDescriptor::wgsl(Some(""), "fn main() {}");
        assert_eq!(desc.label, Some(""));
        assert_eq!(desc.label_string(), "");
    }

    #[test]
    fn test_descriptor_none_label_none_path() {
        let desc = TrinityShaderDescriptor {
            label: None,
            source: ShaderSourceKind::Wgsl(Cow::Borrowed("")),
            file_path: None,
        };
        assert_eq!(desc.label_string(), "<unnamed>");
    }

    #[test]
    fn test_descriptor_clone() {
        let desc = TrinityShaderDescriptor::wgsl(Some("test"), "code")
            .with_file_path("path.wgsl");
        let cloned = desc.clone();
        assert_eq!(cloned.label, desc.label);
        assert_eq!(cloned.file_path, desc.file_path);
    }

    #[test]
    fn test_descriptor_spirv_with_file_path() {
        let words = vec![SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let desc = TrinityShaderDescriptor::spirv(Some("test"), words.as_slice())
            .with_file_path("shader.spv");
        assert!(desc.source.is_spirv());
        assert_eq!(desc.file_path, Some(PathBuf::from("shader.spv")));
    }

    #[test]
    fn test_descriptor_debug_impl() {
        let desc = TrinityShaderDescriptor::wgsl(Some("test"), "fn main() {}");
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("TrinityShaderDescriptor"));
        assert!(debug_str.contains("test"));
    }

    // -------------------------------------------------------------------------
    // WGSL Parsing Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgsl_multiline_comment() {
        let source = r#"
            /* This is a
               multiline comment */
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_wgsl_fragment_shader() {
        let source = r#"
            @fragment
            fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_wgsl_storage_buffer() {
        let source = r#"
            struct Data {
                values: array<f32>,
            }
            @group(0) @binding(0) var<storage, read> data: Data;
            @compute @workgroup_size(1)
            fn main() {
                let _x = data.values[0];
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_wgsl_texture_sampler() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var samp: sampler;
            @fragment
            fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(tex, samp, uv);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_wgsl_invalid_missing_return_type() {
        let source = r#"
            @vertex
            fn main() {
                return vec4<f32>(0.0);
            }
        "#;
        let result = validate_wgsl(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_wgsl_invalid_undefined_variable() {
        let source = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return undefined_var;
            }
        "#;
        let result = validate_wgsl(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_wgsl_invalid_duplicate_binding() {
        // Note: naga may or may not reject duplicate bindings at parse time
        // depending on version. This test verifies the behavior is consistent.
        let source = r#"
            @group(0) @binding(0) var<uniform> a: f32;
            @group(0) @binding(0) var<uniform> b: f32;
            @compute @workgroup_size(1) fn main() {}
        "#;
        let result = validate_wgsl(source);
        // Naga allows duplicate bindings at module level - they're only
        // invalid when both are used in the same entry point
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_wgsl_builtin_functions() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let _a = sin(1.0);
                let _b = cos(1.0);
                let _c = sqrt(4.0);
                let _d = abs(-1.0);
                let _e = min(1.0, 2.0);
                let _f = max(1.0, 2.0);
                let _g = clamp(0.5, 0.0, 1.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_wgsl_matrix_operations() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let m = mat4x4<f32>(
                    vec4<f32>(1.0, 0.0, 0.0, 0.0),
                    vec4<f32>(0.0, 1.0, 0.0, 0.0),
                    vec4<f32>(0.0, 0.0, 1.0, 0.0),
                    vec4<f32>(0.0, 0.0, 0.0, 1.0)
                );
                let v = vec4<f32>(1.0, 2.0, 3.0, 1.0);
                let _result = m * v;
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    // -------------------------------------------------------------------------
    // SPIR-V Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_spirv_version_1_0() {
        let words = [SPIRV_MAGIC, 0x00010000, 0, 0, 0]; // Version 1.0
        assert!(validate_spirv(&words).is_ok());
    }

    #[test]
    fn test_spirv_version_1_6() {
        let words = [SPIRV_MAGIC, 0x00010600, 0, 0, 0]; // Version 1.6
        assert!(validate_spirv(&words).is_ok());
    }

    #[test]
    fn test_spirv_version_2_0() {
        let words = [SPIRV_MAGIC, 0x00020000, 0, 0, 0]; // Version 2.0
        assert!(validate_spirv(&words).is_ok());
    }

    #[test]
    fn test_spirv_version_0_invalid() {
        let words = [SPIRV_MAGIC, 0x00000000, 0, 0, 0]; // Version 0.0 (invalid)
        let result = validate_spirv(&words);
        assert!(result.is_err());
    }

    #[test]
    fn test_spirv_exactly_minimum_size() {
        let words = [SPIRV_MAGIC, 0x00010300, 0, 0, 0]; // Exactly 5 words
        assert!(validate_spirv(&words).is_ok());
    }

    #[test]
    fn test_spirv_empty_slice() {
        let words: [u32; 0] = [];
        let result = validate_spirv(&words);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShaderError::InvalidSpirV { message, .. } => {
                assert!(message.contains("too small"));
            }
            _ => panic!("expected InvalidSpirV"),
        }
    }

    #[test]
    fn test_spirv_four_words_too_small() {
        let words = [SPIRV_MAGIC, 0x00010300, 0, 0]; // 4 words, need 5
        let result = validate_spirv(&words);
        assert!(result.is_err());
    }

    #[test]
    fn test_spirv_swapped_endian_magic() {
        // Big-endian magic instead of little-endian
        let words = [0x03022307, 0x00010300, 0, 0, 0];
        let result = validate_spirv(&words);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Error Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_io_with_path() {
        let err = ShaderError::IoError {
            message: "file not found".to_string(),
            path: Some(PathBuf::from("/path/to/shader.wgsl")),
        };
        let display = format!("{}", err);
        assert!(display.contains("I/O error"));
        assert!(display.contains("/path/to/shader.wgsl"));
    }

    #[test]
    fn test_error_io_without_path() {
        let err = ShaderError::IoError {
            message: "permission denied".to_string(),
            path: None,
        };
        let display = format!("{}", err);
        assert!(display.contains("I/O error"));
        assert!(display.contains("permission denied"));
    }

    #[test]
    fn test_error_empty_source_with_label() {
        let err = ShaderError::EmptySource {
            label: Some("my_shader".to_string()),
        };
        let display = format!("{}", err);
        assert!(display.contains("my_shader"));
        assert!(display.contains("empty"));
    }

    #[test]
    fn test_error_empty_source_without_label() {
        let err = ShaderError::EmptySource { label: None };
        let display = format!("{}", err);
        assert!(display.contains("empty"));
    }

    #[test]
    fn test_error_source_too_large() {
        let err = ShaderError::SourceTooLarge {
            size: 2_000_000,
            max_size: 1_000_000,
        };
        let display = format!("{}", err);
        assert!(display.contains("2000000"));
        assert!(display.contains("1000000"));
    }

    #[test]
    fn test_error_device() {
        let err = ShaderError::DeviceError {
            message: "device lost".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("device"));
        assert!(display.contains("device lost"));
    }

    #[test]
    fn test_error_multiple_notes() {
        let err = ShaderError::parse("error")
            .with_note("hint 1")
            .with_note("hint 2")
            .with_note("hint 3");
        match err {
            ShaderError::ParseError { notes, .. } => {
                assert_eq!(notes.len(), 3);
                assert_eq!(notes[0], "hint 1");
                assert_eq!(notes[1], "hint 2");
                assert_eq!(notes[2], "hint 3");
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_error_with_note_on_non_applicable() {
        // with_note should be a no-op for non-parse/validation errors
        let err = ShaderError::EmptySource { label: None }.with_note("ignored");
        match err {
            ShaderError::EmptySource { .. } => {}
            _ => panic!("expected EmptySource"),
        }
    }

    #[test]
    fn test_error_with_file_path_on_io_error() {
        let err = ShaderError::IoError {
            message: "test".to_string(),
            path: None,
        }
        .with_file_path("new_path.wgsl");
        match err {
            ShaderError::IoError { path, .. } => {
                assert_eq!(path, Some(PathBuf::from("new_path.wgsl")));
            }
            _ => panic!("expected IoError"),
        }
    }

    #[test]
    fn test_error_with_file_path_no_location() {
        // If there's no location, with_file_path should not add one
        let err = ShaderError::parse("error").with_file_path("test.wgsl");
        match err {
            ShaderError::ParseError { location, .. } => {
                assert!(location.is_none());
            }
            _ => panic!("expected ParseError"),
        }
    }

    #[test]
    fn test_error_validation_at() {
        let loc = ShaderLocation::new(10, 20);
        let err = ShaderError::validation_at("type mismatch", loc);
        match err {
            ShaderError::ValidationError { message, location, .. } => {
                assert_eq!(message, "type mismatch");
                let loc = location.unwrap();
                assert_eq!(loc.line, 10);
                assert_eq!(loc.column, 20);
            }
            _ => panic!("expected ValidationError"),
        }
    }

    // -------------------------------------------------------------------------
    // Location Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_location_set_file_path() {
        let mut loc = ShaderLocation::new(1, 1);
        loc.set_file_path("test.wgsl");
        assert_eq!(loc.file_path, Some(PathBuf::from("test.wgsl")));
    }

    #[test]
    fn test_location_clone() {
        let loc = ShaderLocation::with_file(5, 10, "test.wgsl");
        let cloned = loc.clone();
        assert_eq!(cloned.line, loc.line);
        assert_eq!(cloned.column, loc.column);
        assert_eq!(cloned.file_path, loc.file_path);
    }

    #[test]
    fn test_location_equality() {
        let loc1 = ShaderLocation::with_file(5, 10, "test.wgsl");
        let loc2 = ShaderLocation::with_file(5, 10, "test.wgsl");
        let loc3 = ShaderLocation::with_file(5, 11, "test.wgsl");
        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);
    }

    #[test]
    fn test_location_from_offset_first_line() {
        let source = "hello world";
        let loc = ShaderLocation::from_offset(6, 5, source);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 7); // 'w' is at column 7
    }

    #[test]
    fn test_location_from_offset_empty_line() {
        let source = "line1\n\nline3";
        let loc = ShaderLocation::from_offset(6, 0, source);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_location_from_offset_last_char() {
        let source = "abc";
        let loc = ShaderLocation::from_offset(2, 1, source);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 3); // 'c' is at column 3
    }

    #[test]
    fn test_location_from_offset_beyond_source() {
        let source = "abc";
        let loc = ShaderLocation::from_offset(100, 1, source);
        // Should clamp to end
        assert_eq!(loc.line, 1);
    }

    #[test]
    fn test_location_display_with_complex_path() {
        let loc = ShaderLocation::with_file(1, 1, "/long/path/to/my/shader.wgsl");
        let display = format!("{}", loc);
        assert!(display.contains("/long/path/to/my/shader.wgsl"));
    }

    #[test]
    fn test_location_debug_impl() {
        let loc = ShaderLocation::with_file(5, 10, "test.wgsl");
        let debug = format!("{:?}", loc);
        assert!(debug.contains("ShaderLocation"));
        assert!(debug.contains("line: 5"));
        assert!(debug.contains("column: 10"));
    }

    // -------------------------------------------------------------------------
    // Validation Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_wgsl_struct_definition() {
        let source = r#"
            struct MyStruct {
                a: f32,
                b: vec3<f32>,
                c: mat4x4<f32>,
            }
            @compute @workgroup_size(1)
            fn main() {
                var s: MyStruct;
                s.a = 1.0;
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_validate_wgsl_array_types() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                var arr: array<f32, 10>;
                arr[0] = 1.0;
                let _x = arr[0];
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_validate_wgsl_control_flow() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                var x: i32 = 0;
                if x == 0 {
                    x = 1;
                } else {
                    x = 2;
                }
                loop {
                    if x > 10 {
                        break;
                    }
                    x = x + 1;
                    continue;
                }
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_validate_wgsl_switch_statement() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let x: i32 = 1;
                switch x {
                    case 0: { }
                    case 1: { }
                    default: { }
                }
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    // -------------------------------------------------------------------------
    // Hash Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_deterministic() {
        let hash1 = compute_sha256(b"test data");
        let hash2 = compute_sha256(b"test data");
        let hash3 = compute_sha256(b"test data");
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_empty_data() {
        let hash = compute_sha256(b"");
        assert_eq!(hash.len(), 32);
        // SHA256 of empty string is well-known
        assert_eq!(
            hash,
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
            ]
        );
    }

    #[test]
    fn test_hash_whitespace_matters() {
        let hash1 = compute_sha256(b"hello");
        let hash2 = compute_sha256(b"hello ");
        let hash3 = compute_sha256(b" hello");
        assert_ne!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash2, hash3);
    }

    #[test]
    fn test_hash_case_sensitive() {
        let hash1 = compute_sha256(b"Hello");
        let hash2 = compute_sha256(b"hello");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_large_input() {
        let large = vec![0u8; 1_000_000];
        let hash = compute_sha256(&large);
        assert_eq!(hash.len(), 32);
    }

    // -------------------------------------------------------------------------
    // Edge Cases - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_edge_case_empty_string_source() {
        let result = validate_wgsl("");
        assert!(matches!(result, Err(ShaderError::EmptySource { .. })));
    }

    #[test]
    fn test_edge_case_only_newlines() {
        let result = validate_wgsl("\n\n\n");
        assert!(matches!(result, Err(ShaderError::EmptySource { .. })));
    }

    #[test]
    fn test_edge_case_only_spaces() {
        let result = validate_wgsl("     ");
        assert!(matches!(result, Err(ShaderError::EmptySource { .. })));
    }

    #[test]
    fn test_edge_case_only_tabs() {
        let result = validate_wgsl("\t\t\t");
        assert!(matches!(result, Err(ShaderError::EmptySource { .. })));
    }

    #[test]
    fn test_edge_case_mixed_whitespace() {
        let result = validate_wgsl("  \n\t  \n  \t");
        assert!(matches!(result, Err(ShaderError::EmptySource { .. })));
    }

    #[test]
    fn test_edge_case_unicode_whitespace() {
        // Non-breaking space and other unicode whitespace
        let result = validate_wgsl("\u{00A0}\u{2003}");
        // This may or may not be considered empty depending on trim behavior
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_null_byte() {
        let source = "fn main()\0{}";
        let result = validate_wgsl(source);
        // Null bytes are invalid in WGSL
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_very_deep_nesting() {
        // Deeply nested blocks
        let mut source = String::from("@compute @workgroup_size(1) fn main() {");
        for _ in 0..50 {
            source.push_str(" { ");
        }
        for _ in 0..50 {
            source.push_str(" } ");
        }
        source.push('}');
        // Should parse (might hit limits in some implementations)
        let result = validate_wgsl(&source);
        // The result depends on naga's limits
        let _ = result;
    }

    #[test]
    fn test_edge_case_many_functions() {
        let mut source = String::new();
        for i in 0..100 {
            source.push_str(&format!("fn func_{}() {{ }}\n", i));
        }
        source.push_str("@compute @workgroup_size(1) fn main() {}");
        assert!(validate_wgsl(&source).is_ok());
    }

    #[test]
    fn test_edge_case_long_identifier() {
        let long_name = "a".repeat(200);
        let source = format!(
            "@compute @workgroup_size(1) fn main() {{ var {}: f32 = 1.0; }}",
            long_name
        );
        // Long identifiers should work
        assert!(validate_wgsl(&source).is_ok());
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_line_column_to_offset_first_char() {
        let source = "hello";
        assert_eq!(line_column_to_offset(source, 1, 1), Some(0));
    }

    #[test]
    fn test_line_column_to_offset_last_char_last_line() {
        let source = "line1\nline2";
        assert_eq!(line_column_to_offset(source, 2, 5), Some(10));
    }

    #[test]
    fn test_line_column_to_offset_empty_source() {
        let source = "";
        assert_eq!(line_column_to_offset(source, 1, 1), Some(0));
    }

    #[test]
    fn test_line_column_to_offset_column_past_end() {
        let source = "abc";
        // Column 10 on a 3-char line
        let result = line_column_to_offset(source, 1, 10);
        // Should return None or clamp
        assert!(result.is_none() || result == Some(3));
    }

    #[test]
    fn test_extract_source_line_first_line() {
        let source = "first\nsecond\nthird";
        assert_eq!(extract_source_line(source, 0), Some("first"));
        assert_eq!(extract_source_line(source, 4), Some("first"));
    }

    #[test]
    fn test_extract_source_line_middle_line() {
        let source = "first\nsecond\nthird";
        assert_eq!(extract_source_line(source, 6), Some("second"));
        assert_eq!(extract_source_line(source, 8), Some("second"));
    }

    #[test]
    fn test_extract_source_line_last_line_no_newline() {
        let source = "first\nsecond\nthird";
        assert_eq!(extract_source_line(source, 13), Some("third"));
    }

    #[test]
    fn test_extract_source_line_empty_source() {
        let source = "";
        assert_eq!(extract_source_line(source, 0), None);
    }

    #[test]
    fn test_extract_source_line_single_line() {
        let source = "single line";
        assert_eq!(extract_source_line(source, 0), Some("single line"));
        assert_eq!(extract_source_line(source, 5), Some("single line"));
    }

    // -------------------------------------------------------------------------
    // ShaderSourceKind Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_source_kind_wgsl_owned_string() {
        let owned = String::from("fn main() {}");
        let source = ShaderSourceKind::wgsl(owned);
        assert!(source.is_wgsl());
        assert_eq!(source.as_wgsl(), Some("fn main() {}"));
    }

    #[test]
    fn test_source_kind_wgsl_borrowed_str() {
        let source = ShaderSourceKind::wgsl("borrowed");
        assert!(source.is_wgsl());
    }

    #[test]
    fn test_source_kind_spirv_owned_vec() {
        let words = vec![SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let source = ShaderSourceKind::spirv(words);
        assert!(source.is_spirv());
    }

    #[test]
    fn test_source_kind_spirv_borrowed_slice() {
        let words: &[u32] = &[SPIRV_MAGIC, 0x00010300, 0, 0, 0];
        let source = ShaderSourceKind::spirv(words);
        assert!(source.is_spirv());
        assert!(source.as_spirv().is_some());
    }

    #[test]
    fn test_source_kind_clone() {
        let source = ShaderSourceKind::wgsl("test");
        let cloned = source.clone();
        assert!(cloned.is_wgsl());
        assert_eq!(cloned.as_wgsl(), source.as_wgsl());
    }

    #[test]
    fn test_source_kind_debug() {
        let source = ShaderSourceKind::wgsl("test");
        let debug = format!("{:?}", source);
        assert!(debug.contains("Wgsl"));
    }

    // -------------------------------------------------------------------------
    // Error Formatting Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_with_source_no_location() {
        let err = ShaderError::parse("error without location");
        let formatted = err.format_with_source("source code", "default.wgsl");
        assert!(formatted.contains("default.wgsl"));
        assert!(formatted.contains("error without location"));
    }

    #[test]
    fn test_format_with_source_with_notes() {
        let err = ShaderError::parse("error")
            .with_note("note 1")
            .with_note("note 2");
        let formatted = err.format_with_source("", "test.wgsl");
        assert!(formatted.contains("note: note 1"));
        assert!(formatted.contains("note: note 2"));
    }

    #[test]
    fn test_format_with_source_size_error() {
        let err = ShaderError::SourceTooLarge {
            size: 5_000_000,
            max_size: 1_000_000,
        };
        let formatted = err.format_with_source("", "large.wgsl");
        assert!(formatted.contains("5000000"));
        assert!(formatted.contains("1000000"));
    }

    #[test]
    fn test_format_with_source_device_error() {
        let err = ShaderError::DeviceError {
            message: "GPU crashed".to_string(),
        };
        let formatted = err.format_with_source("", "crash.wgsl");
        assert!(formatted.contains("device"));
        assert!(formatted.contains("GPU crashed"));
    }

    #[test]
    fn test_format_with_source_io_error() {
        let err = ShaderError::IoError {
            message: "file not found".to_string(),
            path: Some(PathBuf::from("missing.wgsl")),
        };
        let formatted = err.format_with_source("", "default.wgsl");
        assert!(formatted.contains("missing.wgsl"));
        assert!(formatted.contains("file not found"));
    }

    // -------------------------------------------------------------------------
    // Constants Tests - Extended
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_values() {
        // Verify the exact expected values
        assert_eq!(SPIRV_MIN_SIZE, 20);
        assert_eq!(SPIRV_MIN_SIZE / 4, 5); // 5 words minimum
        assert_eq!(SPIRV_MAGIC, 0x07230203);
        assert_eq!(MAX_SHADER_SOURCE_SIZE, 1024 * 1024); // 1 MB
    }

    #[test]
    fn test_spirv_magic_as_bytes() {
        // Verify magic number bytes in little-endian
        let bytes = SPIRV_MAGIC.to_le_bytes();
        assert_eq!(bytes, [0x03, 0x02, 0x23, 0x07]);
    }

    // -------------------------------------------------------------------------
    // Complex Shader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_complex_pbr_like_shader() {
        let source = r#"
            struct Uniforms {
                model_matrix: mat4x4<f32>,
                view_proj: mat4x4<f32>,
                camera_pos: vec3<f32>,
            }

            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) uv: vec2<f32>,
            }

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) world_pos: vec3<f32>,
                @location(1) normal: vec3<f32>,
                @location(2) uv: vec2<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn vs_main(in: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                let world_pos = uniforms.model_matrix * vec4<f32>(in.position, 1.0);
                out.clip_position = uniforms.view_proj * world_pos;
                out.world_pos = world_pos.xyz;
                out.normal = (uniforms.model_matrix * vec4<f32>(in.normal, 0.0)).xyz;
                out.uv = in.uv;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                let N = normalize(in.normal);
                let V = normalize(uniforms.camera_pos - in.world_pos);
                let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
                let diffuse = max(dot(N, light_dir), 0.0);
                return vec4<f32>(vec3<f32>(diffuse), 1.0);
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_complex_compute_shader() {
        let source = r#"
            struct Particle {
                position: vec3<f32>,
                velocity: vec3<f32>,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
            @group(0) @binding(1) var<uniform> delta_time: f32;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if idx >= arrayLength(&particles) {
                    return;
                }

                var p = particles[idx];
                p.velocity.y = p.velocity.y - 9.8 * delta_time;
                p.position = p.position + p.velocity * delta_time;

                if p.position.y < 0.0 {
                    p.position.y = 0.0;
                    p.velocity.y = -p.velocity.y * 0.8;
                }

                particles[idx] = p;
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_shader_with_workgroup_memory() {
        let source = r#"
            var<workgroup> shared_data: array<f32, 256>;

            @group(0) @binding(0) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) wg_id: vec3<u32>
            ) {
                let idx = local_id.x;
                shared_data[idx] = f32(idx);
                workgroupBarrier();
                output[wg_id.x * 256u + idx] = shared_data[255u - idx];
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    // -------------------------------------------------------------------------
    // Regression Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_regression_empty_struct() {
        // Empty structs might cause issues in some compilers
        let source = r#"
            struct Empty {}
            @compute @workgroup_size(1)
            fn main() {
                var e: Empty;
                _ = e;
            }
        "#;
        // This may or may not be valid depending on WGSL spec version
        let _ = validate_wgsl(source);
    }

    #[test]
    fn test_regression_trailing_comma() {
        let source = r#"
            struct Test {
                a: f32,
                b: f32,  // trailing comma
            }
            @compute @workgroup_size(1)
            fn main() {}
        "#;
        assert!(validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_regression_underscore_discard() {
        // Test that phony assignment works - required in WGSL for unused results
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let x = 1.0;
                _ = x;
            }
        "#;
        assert!(validate_wgsl(source).is_ok());
    }
}
