//! GPU validation layer integration for wgpu 25.x debugging.
//!
//! This module provides comprehensive validation layer support for catching
//! GPU API misuse, synchronization issues, and performance problems during
//! development.
//!
//! # Overview
//!
//! - [`ValidationLevel`] - Validation intensity levels
//! - [`ValidationFeatures`] - Configurable validation feature flags
//! - [`ValidationMessage`] - Structured validation message with metadata
//! - [`ValidationSeverity`] - Message severity classification
//! - [`ValidationMessageType`] - Message type categorization
//! - [`ValidationObject`] - GPU object identification for debugging
//! - [`ValidationLayer`] - Main validation layer interface
//! - [`ValidationScope`] - RAII guard for scoped validation
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::validation::*;
//!
//! // Create validation layer from environment
//! let mut layer = ValidationLayer::from_env();
//!
//! // Or create with specific level
//! let mut layer = ValidationLayer::new(ValidationLevel::Full);
//!
//! // Register callback for validation messages
//! layer.register_callback(Box::new(|msg| {
//!     eprintln!("[{}] {}: {}", msg.severity, msg.message_type, msg.message);
//! }));
//!
//! // Use scoped validation
//! {
//!     let _scope = ValidationScope::new(&layer, "Shadow Pass");
//!     // GPU operations here...
//! } // Scope ends, reports any errors
//! ```
//!
//! # Environment Variables
//!
//! - `WGPU_VALIDATION` - Set validation level: "disabled", "basic", "full", "verbose"
//! - `WGPU_VALIDATION_BREAK` - Break on validation errors if set to "1"

use std::env;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::SourceLocation;

// ============================================================================
// ValidationLevel
// ============================================================================

/// Level of validation to perform.
///
/// Higher levels provide more comprehensive validation at the cost of
/// performance. Choose based on development vs production needs.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::ValidationLevel;
///
/// let level = ValidationLevel::from_env();
/// if level.is_enabled() {
///     println!("Validation enabled at {:?}", level);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ValidationLevel {
    /// No validation (production mode).
    #[default]
    Disabled,

    /// Basic validation (API usage checks).
    Basic,

    /// Full validation (includes synchronization checks).
    Full,

    /// Verbose validation (maximum detail, includes performance hints).
    Verbose,
}

impl ValidationLevel {
    /// Read validation level from the `WGPU_VALIDATION` environment variable.
    ///
    /// Supported values (case-insensitive):
    /// - "disabled", "none", "off", "0" -> Disabled
    /// - "basic", "min", "1" -> Basic
    /// - "full", "on", "true", "2" -> Full
    /// - "verbose", "max", "debug", "3" -> Verbose
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationLevel;
    ///
    /// // Reads from WGPU_VALIDATION env var
    /// let level = ValidationLevel::from_env();
    /// ```
    pub fn from_env() -> Self {
        match env::var("WGPU_VALIDATION") {
            Ok(val) => Self::from_str(&val),
            Err(_) => {
                // Default to Basic in debug builds, Disabled in release
                if cfg!(debug_assertions) {
                    ValidationLevel::Basic
                } else {
                    ValidationLevel::Disabled
                }
            }
        }
    }

    /// Parse validation level from a string.
    ///
    /// # Arguments
    ///
    /// * `s` - String to parse (case-insensitive)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationLevel;
    ///
    /// assert_eq!(ValidationLevel::from_str("full"), ValidationLevel::Full);
    /// assert_eq!(ValidationLevel::from_str("VERBOSE"), ValidationLevel::Verbose);
    /// ```
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "disabled" | "none" | "off" | "0" | "false" => ValidationLevel::Disabled,
            "basic" | "min" | "1" => ValidationLevel::Basic,
            "full" | "on" | "true" | "2" => ValidationLevel::Full,
            "verbose" | "max" | "debug" | "3" => ValidationLevel::Verbose,
            _ => ValidationLevel::Basic,
        }
    }

    /// Check if any validation is enabled.
    ///
    /// # Returns
    ///
    /// `true` if validation level is not Disabled.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationLevel;
    ///
    /// assert!(!ValidationLevel::Disabled.is_enabled());
    /// assert!(ValidationLevel::Basic.is_enabled());
    /// assert!(ValidationLevel::Full.is_enabled());
    /// ```
    #[inline]
    pub fn is_enabled(&self) -> bool {
        !matches!(self, ValidationLevel::Disabled)
    }

    /// Get the minimum severity threshold for this level.
    ///
    /// Messages below this severity will be filtered out.
    ///
    /// # Returns
    ///
    /// The minimum [`ValidationSeverity`] to report.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::{ValidationLevel, ValidationSeverity};
    ///
    /// assert_eq!(ValidationLevel::Basic.severity_threshold(), ValidationSeverity::Warning);
    /// assert_eq!(ValidationLevel::Verbose.severity_threshold(), ValidationSeverity::Verbose);
    /// ```
    pub fn severity_threshold(&self) -> ValidationSeverity {
        match self {
            ValidationLevel::Disabled => ValidationSeverity::Error, // Only critical
            ValidationLevel::Basic => ValidationSeverity::Warning,
            ValidationLevel::Full => ValidationSeverity::Info,
            ValidationLevel::Verbose => ValidationSeverity::Verbose,
        }
    }

    /// Get the appropriate validation features for this level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationLevel;
    ///
    /// let features = ValidationLevel::Full.default_features();
    /// assert!(features.shader_validation);
    /// ```
    pub fn default_features(&self) -> ValidationFeatures {
        ValidationFeatures::for_level(*self)
    }
}

impl fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationLevel::Disabled => write!(f, "Disabled"),
            ValidationLevel::Basic => write!(f, "Basic"),
            ValidationLevel::Full => write!(f, "Full"),
            ValidationLevel::Verbose => write!(f, "Verbose"),
        }
    }
}

// ============================================================================
// ValidationFeatures
// ============================================================================

/// Configurable validation feature flags.
///
/// Controls which types of validation are performed. Individual features
/// can be toggled for fine-grained control over validation overhead.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::ValidationFeatures;
///
/// let mut features = ValidationFeatures::default();
/// features.gpu_based_validation = true;
/// features.synchronization_validation = true;
///
/// // Or use all_enabled for maximum validation
/// let max_features = ValidationFeatures::all_enabled();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValidationFeatures {
    /// Enable GPU-based validation (validates GPU commands on the GPU).
    ///
    /// This catches issues that CPU-side validation cannot detect,
    /// but has significant performance overhead.
    pub gpu_based_validation: bool,

    /// Enable synchronization validation.
    ///
    /// Validates that resources are properly synchronized between
    /// GPU operations (barriers, memory dependencies).
    pub synchronization_validation: bool,

    /// Enable shader validation.
    ///
    /// Validates shader code for correctness, undefined behavior,
    /// and resource binding issues.
    pub shader_validation: bool,

    /// Enable descriptor indexing validation.
    ///
    /// Validates that descriptor array accesses are within bounds
    /// and properly initialized.
    pub descriptor_indexing_validation: bool,

    /// Enable best practices warnings.
    ///
    /// Reports potential performance issues and API usage patterns
    /// that, while valid, may not be optimal.
    pub best_practices_warnings: bool,

    /// Enable printf output to stdout.
    ///
    /// When enabled, shader printf statements are routed to stdout
    /// for debugging purposes.
    pub printf_to_stdout: bool,
}

impl ValidationFeatures {
    /// Create features with sensible defaults.
    ///
    /// Enables shader validation and best practices warnings,
    /// disables more expensive validation features.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create features with all validation enabled.
    ///
    /// Use for maximum validation coverage during development.
    /// Has significant performance overhead.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationFeatures;
    ///
    /// let features = ValidationFeatures::all_enabled();
    /// assert!(features.gpu_based_validation);
    /// assert!(features.synchronization_validation);
    /// ```
    pub fn all_enabled() -> Self {
        Self {
            gpu_based_validation: true,
            synchronization_validation: true,
            shader_validation: true,
            descriptor_indexing_validation: true,
            best_practices_warnings: true,
            printf_to_stdout: true,
        }
    }

    /// Create features appropriate for the given validation level.
    ///
    /// # Arguments
    ///
    /// * `level` - The validation level to configure for
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::{ValidationFeatures, ValidationLevel};
    ///
    /// let basic = ValidationFeatures::for_level(ValidationLevel::Basic);
    /// assert!(basic.shader_validation);
    /// assert!(!basic.gpu_based_validation);
    ///
    /// let full = ValidationFeatures::for_level(ValidationLevel::Full);
    /// assert!(full.synchronization_validation);
    /// ```
    pub fn for_level(level: ValidationLevel) -> Self {
        match level {
            ValidationLevel::Disabled => Self {
                gpu_based_validation: false,
                synchronization_validation: false,
                shader_validation: false,
                descriptor_indexing_validation: false,
                best_practices_warnings: false,
                printf_to_stdout: false,
            },
            ValidationLevel::Basic => Self {
                gpu_based_validation: false,
                synchronization_validation: false,
                shader_validation: true,
                descriptor_indexing_validation: true,
                best_practices_warnings: false,
                printf_to_stdout: false,
            },
            ValidationLevel::Full => Self {
                gpu_based_validation: false,
                synchronization_validation: true,
                shader_validation: true,
                descriptor_indexing_validation: true,
                best_practices_warnings: true,
                printf_to_stdout: false,
            },
            ValidationLevel::Verbose => Self::all_enabled(),
        }
    }

    /// Check if any validation features are enabled.
    #[inline]
    pub fn any_enabled(&self) -> bool {
        self.gpu_based_validation
            || self.synchronization_validation
            || self.shader_validation
            || self.descriptor_indexing_validation
            || self.best_practices_warnings
    }

    /// Count the number of enabled features.
    pub fn enabled_count(&self) -> usize {
        let mut count = 0;
        if self.gpu_based_validation {
            count += 1;
        }
        if self.synchronization_validation {
            count += 1;
        }
        if self.shader_validation {
            count += 1;
        }
        if self.descriptor_indexing_validation {
            count += 1;
        }
        if self.best_practices_warnings {
            count += 1;
        }
        if self.printf_to_stdout {
            count += 1;
        }
        count
    }
}

impl Default for ValidationFeatures {
    fn default() -> Self {
        Self {
            gpu_based_validation: false,
            synchronization_validation: false,
            shader_validation: true,
            descriptor_indexing_validation: true,
            best_practices_warnings: cfg!(debug_assertions),
            printf_to_stdout: false,
        }
    }
}

impl fmt::Display for ValidationFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut features = Vec::new();
        if self.gpu_based_validation {
            features.push("GPU");
        }
        if self.synchronization_validation {
            features.push("Sync");
        }
        if self.shader_validation {
            features.push("Shader");
        }
        if self.descriptor_indexing_validation {
            features.push("Descriptor");
        }
        if self.best_practices_warnings {
            features.push("BestPractices");
        }
        if self.printf_to_stdout {
            features.push("Printf");
        }

        if features.is_empty() {
            write!(f, "None")
        } else {
            write!(f, "{}", features.join("+"))
        }
    }
}

// ============================================================================
// ValidationSeverity
// ============================================================================

/// Severity level of a validation message.
///
/// Used to filter and prioritize validation output.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::ValidationSeverity;
///
/// let severity = ValidationSeverity::Error;
/// assert!(severity.should_break());
///
/// let log_level = severity.as_log_level();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidationSeverity {
    /// Verbose diagnostic information.
    Verbose,

    /// Informational message.
    Info,

    /// Warning (potential issue, but operation continues).
    Warning,

    /// Error (validation failure, operation may fail).
    Error,
}

impl ValidationSeverity {
    /// Convert to log crate's Level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationSeverity;
    ///
    /// let level = ValidationSeverity::Warning.as_log_level();
    /// assert_eq!(level, log::Level::Warn);
    /// ```
    pub fn as_log_level(&self) -> log::Level {
        match self {
            ValidationSeverity::Verbose => log::Level::Trace,
            ValidationSeverity::Info => log::Level::Info,
            ValidationSeverity::Warning => log::Level::Warn,
            ValidationSeverity::Error => log::Level::Error,
        }
    }

    /// Check if this severity should trigger a debugger break.
    ///
    /// Only Error severity should break by default.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationSeverity;
    ///
    /// assert!(!ValidationSeverity::Warning.should_break());
    /// assert!(ValidationSeverity::Error.should_break());
    /// ```
    #[inline]
    pub fn should_break(&self) -> bool {
        matches!(self, ValidationSeverity::Error)
    }

    /// Check if this severity is at least as severe as the threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The minimum severity to check against
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationSeverity;
    ///
    /// assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Warning));
    /// assert!(!ValidationSeverity::Info.meets_threshold(ValidationSeverity::Warning));
    /// ```
    #[inline]
    pub fn meets_threshold(&self, threshold: ValidationSeverity) -> bool {
        *self >= threshold
    }

    /// Get a short string representation for logging.
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationSeverity::Verbose => "VERBOSE",
            ValidationSeverity::Info => "INFO",
            ValidationSeverity::Warning => "WARN",
            ValidationSeverity::Error => "ERROR",
        }
    }
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for ValidationSeverity {
    fn default() -> Self {
        ValidationSeverity::Info
    }
}

// ============================================================================
// ValidationMessageType
// ============================================================================

/// Type of validation message.
///
/// Categorizes validation messages by their nature for filtering
/// and handling.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::ValidationMessageType;
///
/// let msg_type = ValidationMessageType::Validation;
/// assert!(msg_type.is_error());
///
/// let perf_type = ValidationMessageType::Performance;
/// assert!(!perf_type.is_error());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationMessageType {
    /// General informational message.
    General,

    /// Validation error or warning.
    Validation,

    /// Performance warning or suggestion.
    Performance,

    /// Debug marker or annotation.
    DebugMarker,
}

impl ValidationMessageType {
    /// Check if this message type indicates a validation error.
    ///
    /// Only the Validation type is considered an error type.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::ValidationMessageType;
    ///
    /// assert!(ValidationMessageType::Validation.is_error());
    /// assert!(!ValidationMessageType::Performance.is_error());
    /// ```
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, ValidationMessageType::Validation)
    }

    /// Check if this is a performance-related message.
    #[inline]
    pub fn is_performance(&self) -> bool {
        matches!(self, ValidationMessageType::Performance)
    }

    /// Get the default severity for this message type.
    pub fn default_severity(&self) -> ValidationSeverity {
        match self {
            ValidationMessageType::General => ValidationSeverity::Info,
            ValidationMessageType::Validation => ValidationSeverity::Error,
            ValidationMessageType::Performance => ValidationSeverity::Warning,
            ValidationMessageType::DebugMarker => ValidationSeverity::Verbose,
        }
    }
}

impl fmt::Display for ValidationMessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationMessageType::General => write!(f, "General"),
            ValidationMessageType::Validation => write!(f, "Validation"),
            ValidationMessageType::Performance => write!(f, "Performance"),
            ValidationMessageType::DebugMarker => write!(f, "DebugMarker"),
        }
    }
}

impl Default for ValidationMessageType {
    fn default() -> Self {
        ValidationMessageType::General
    }
}

// ============================================================================
// ValidationObjectType
// ============================================================================

/// Type of GPU object involved in a validation message.
///
/// Identifies the kind of GPU resource that triggered or is related
/// to a validation message.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::ValidationObjectType;
///
/// let obj_type = ValidationObjectType::Buffer;
/// println!("Object type: {}", obj_type);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationObjectType {
    /// Unknown or unspecified object type.
    Unknown,

    /// GPU buffer (vertex, index, uniform, storage).
    Buffer,

    /// Texture or image.
    Texture,

    /// Texture view.
    TextureView,

    /// Sampler.
    Sampler,

    /// Bind group.
    BindGroup,

    /// Bind group layout.
    BindGroupLayout,

    /// Render pipeline.
    RenderPipeline,

    /// Compute pipeline.
    ComputePipeline,

    /// Pipeline layout.
    PipelineLayout,

    /// Shader module.
    ShaderModule,

    /// Command buffer.
    CommandBuffer,

    /// Command encoder.
    CommandEncoder,

    /// Render pass.
    RenderPass,

    /// Compute pass.
    ComputePass,

    /// Query set.
    QuerySet,

    /// Surface (swap chain).
    Surface,

    /// Device.
    Device,

    /// Queue.
    Queue,

    /// Adapter.
    Adapter,

    /// Instance.
    Instance,
}

impl ValidationObjectType {
    /// Check if this is a pipeline-related object.
    #[inline]
    pub fn is_pipeline(&self) -> bool {
        matches!(
            self,
            ValidationObjectType::RenderPipeline
                | ValidationObjectType::ComputePipeline
                | ValidationObjectType::PipelineLayout
        )
    }

    /// Check if this is a resource object (buffer, texture, sampler).
    #[inline]
    pub fn is_resource(&self) -> bool {
        matches!(
            self,
            ValidationObjectType::Buffer
                | ValidationObjectType::Texture
                | ValidationObjectType::TextureView
                | ValidationObjectType::Sampler
        )
    }

    /// Check if this is a binding-related object.
    #[inline]
    pub fn is_binding(&self) -> bool {
        matches!(
            self,
            ValidationObjectType::BindGroup | ValidationObjectType::BindGroupLayout
        )
    }

    /// Check if this is a command-related object.
    #[inline]
    pub fn is_command(&self) -> bool {
        matches!(
            self,
            ValidationObjectType::CommandBuffer
                | ValidationObjectType::CommandEncoder
                | ValidationObjectType::RenderPass
                | ValidationObjectType::ComputePass
        )
    }
}

impl fmt::Display for ValidationObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationObjectType::Unknown => write!(f, "Unknown"),
            ValidationObjectType::Buffer => write!(f, "Buffer"),
            ValidationObjectType::Texture => write!(f, "Texture"),
            ValidationObjectType::TextureView => write!(f, "TextureView"),
            ValidationObjectType::Sampler => write!(f, "Sampler"),
            ValidationObjectType::BindGroup => write!(f, "BindGroup"),
            ValidationObjectType::BindGroupLayout => write!(f, "BindGroupLayout"),
            ValidationObjectType::RenderPipeline => write!(f, "RenderPipeline"),
            ValidationObjectType::ComputePipeline => write!(f, "ComputePipeline"),
            ValidationObjectType::PipelineLayout => write!(f, "PipelineLayout"),
            ValidationObjectType::ShaderModule => write!(f, "ShaderModule"),
            ValidationObjectType::CommandBuffer => write!(f, "CommandBuffer"),
            ValidationObjectType::CommandEncoder => write!(f, "CommandEncoder"),
            ValidationObjectType::RenderPass => write!(f, "RenderPass"),
            ValidationObjectType::ComputePass => write!(f, "ComputePass"),
            ValidationObjectType::QuerySet => write!(f, "QuerySet"),
            ValidationObjectType::Surface => write!(f, "Surface"),
            ValidationObjectType::Device => write!(f, "Device"),
            ValidationObjectType::Queue => write!(f, "Queue"),
            ValidationObjectType::Adapter => write!(f, "Adapter"),
            ValidationObjectType::Instance => write!(f, "Instance"),
        }
    }
}

impl Default for ValidationObjectType {
    fn default() -> Self {
        ValidationObjectType::Unknown
    }
}

// ============================================================================
// ValidationObject
// ============================================================================

/// Identifies a GPU object involved in a validation message.
///
/// Contains the object type, handle value, and optional debug name
/// for tracking which specific GPU objects are involved in validation
/// issues.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::{ValidationObject, ValidationObjectType};
///
/// let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x12345678)
///     .with_name("VertexBuffer");
///
/// println!("Object: {}", obj);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationObject {
    /// The type of GPU object.
    pub object_type: ValidationObjectType,

    /// The object handle (opaque identifier).
    pub handle: u64,

    /// Optional debug name for the object.
    pub name: Option<String>,
}

impl ValidationObject {
    /// Create a new validation object.
    ///
    /// # Arguments
    ///
    /// * `object_type` - The type of GPU object
    /// * `handle` - The object handle value
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::{ValidationObject, ValidationObjectType};
    ///
    /// let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
    /// assert_eq!(obj.object_type, ValidationObjectType::Buffer);
    /// ```
    pub fn new(object_type: ValidationObjectType, handle: u64) -> Self {
        Self {
            object_type,
            handle,
            name: None,
        }
    }

    /// Create an unknown object with just a handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The object handle value
    pub fn unknown(handle: u64) -> Self {
        Self {
            object_type: ValidationObjectType::Unknown,
            handle,
            name: None,
        }
    }

    /// Add a debug name to the object.
    ///
    /// # Arguments
    ///
    /// * `name` - The debug name
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::{ValidationObject, ValidationObjectType};
    ///
    /// let obj = ValidationObject::new(ValidationObjectType::Texture, 0x5678)
    ///     .with_name("ShadowMap");
    /// assert_eq!(obj.name.as_deref(), Some("ShadowMap"));
    /// ```
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the debug name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Check if this object has a debug name.
    #[inline]
    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    /// Get a display string for the object.
    pub fn display_string(&self) -> String {
        if let Some(ref name) = self.name {
            format!("{}({:#x}): {}", self.object_type, self.handle, name)
        } else {
            format!("{}({:#x})", self.object_type, self.handle)
        }
    }
}

impl fmt::Display for ValidationObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

// ============================================================================
// ValidationMessage
// ============================================================================

/// A validation message from the GPU validation layer.
///
/// Contains all information about a validation event including severity,
/// type, message content, involved objects, and source location.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::*;
///
/// let msg = ValidationMessage::new(
///     ValidationSeverity::Warning,
///     ValidationMessageType::Performance,
///     "Consider using a staging buffer for frequent uploads",
/// );
///
/// println!("[{}] {}: {}", msg.severity, msg.message_type, msg.message);
/// ```
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Severity level of the message.
    pub severity: ValidationSeverity,

    /// Type of validation message.
    pub message_type: ValidationMessageType,

    /// Optional message ID (for filtering/suppression).
    pub message_id: Option<i32>,

    /// The message content.
    pub message: String,

    /// GPU objects involved in this validation event.
    pub objects: Vec<ValidationObject>,

    /// Source location where the issue was detected.
    pub location: Option<SourceLocation>,

    /// When the message was generated.
    pub timestamp: Instant,
}

impl ValidationMessage {
    /// Create a new validation message.
    ///
    /// # Arguments
    ///
    /// * `severity` - The message severity
    /// * `message_type` - The type of message
    /// * `message` - The message content
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let msg = ValidationMessage::new(
    ///     ValidationSeverity::Error,
    ///     ValidationMessageType::Validation,
    ///     "Invalid buffer usage flags",
    /// );
    /// ```
    pub fn new(
        severity: ValidationSeverity,
        message_type: ValidationMessageType,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            message_type,
            message_id: None,
            message: message.into(),
            objects: Vec::new(),
            location: None,
            timestamp: Instant::now(),
        }
    }

    /// Create a validation error message.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(
            ValidationSeverity::Error,
            ValidationMessageType::Validation,
            message,
        )
    }

    /// Create a validation warning message.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(
            ValidationSeverity::Warning,
            ValidationMessageType::Validation,
            message,
        )
    }

    /// Create a performance warning message.
    pub fn performance(message: impl Into<String>) -> Self {
        Self::new(
            ValidationSeverity::Warning,
            ValidationMessageType::Performance,
            message,
        )
    }

    /// Create an info message.
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(
            ValidationSeverity::Info,
            ValidationMessageType::General,
            message,
        )
    }

    /// Add a message ID.
    pub fn with_id(mut self, id: i32) -> Self {
        self.message_id = Some(id);
        self
    }

    /// Add a GPU object to the message.
    pub fn with_object(mut self, object: ValidationObject) -> Self {
        self.objects.push(object);
        self
    }

    /// Add multiple GPU objects.
    pub fn with_objects(mut self, objects: Vec<ValidationObject>) -> Self {
        self.objects.extend(objects);
        self
    }

    /// Add a source location.
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Check if this is an error-level message.
    #[inline]
    pub fn is_error(&self) -> bool {
        self.severity == ValidationSeverity::Error
    }

    /// Check if this is a warning-level message.
    #[inline]
    pub fn is_warning(&self) -> bool {
        self.severity == ValidationSeverity::Warning
    }

    /// Check if this message meets a severity threshold.
    #[inline]
    pub fn meets_threshold(&self, threshold: ValidationSeverity) -> bool {
        self.severity.meets_threshold(threshold)
    }

    /// Log this message using the log crate.
    pub fn log(&self) {
        let level = self.severity.as_log_level();
        let objects_str = if self.objects.is_empty() {
            String::new()
        } else {
            format!(
                " [{}]",
                self.objects
                    .iter()
                    .map(|o| o.display_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let location_str = if let Some(ref loc) = self.location {
            format!(" at {}", loc)
        } else {
            String::new()
        };

        log::log!(
            level,
            "[GPU {}] {}{}{}: {}",
            self.message_type,
            if let Some(id) = self.message_id {
                format!("#{} ", id)
            } else {
                String::new()
            },
            objects_str,
            location_str,
            self.message
        );
    }

    /// Get elapsed time since message was generated.
    #[inline]
    pub fn elapsed(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

impl fmt::Display for ValidationMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.message_type, self.message)
    }
}

// ============================================================================
// ValidationCallbackFn
// ============================================================================

/// Callback function type for validation messages.
pub type ValidationCallbackFn = Box<dyn Fn(&ValidationMessage) + Send + Sync>;

// ============================================================================
// ValidationCallbackRegistry
// ============================================================================

/// Thread-safe registry for validation message callbacks.
///
/// Allows multiple callbacks to be registered for validation messages,
/// enabling different handling strategies (logging, breaking, collecting).
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::*;
///
/// let registry = ValidationCallbackRegistry::new();
///
/// // Register a callback
/// registry.register(Box::new(|msg| {
///     println!("[{}] {}", msg.severity, msg.message);
/// }));
///
/// // Invoke all callbacks
/// let msg = ValidationMessage::warning("Test message");
/// registry.invoke(&msg);
/// ```
pub struct ValidationCallbackRegistry {
    callbacks: RwLock<Vec<ValidationCallbackFn>>,
}

impl ValidationCallbackRegistry {
    /// Create a new empty callback registry.
    pub fn new() -> Self {
        Self {
            callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Register a callback for validation messages.
    ///
    /// # Arguments
    ///
    /// * `callback` - The callback function to register
    ///
    /// # Returns
    ///
    /// The index of the registered callback (for removal).
    pub fn register(&self, callback: ValidationCallbackFn) -> usize {
        let mut callbacks = self.callbacks.write().unwrap();
        let index = callbacks.len();
        callbacks.push(callback);
        index
    }

    /// Invoke all registered callbacks with a message.
    ///
    /// # Arguments
    ///
    /// * `message` - The validation message to dispatch
    pub fn invoke(&self, message: &ValidationMessage) {
        let callbacks = self.callbacks.read().unwrap();
        for callback in callbacks.iter() {
            callback(message);
        }
    }

    /// Clear all registered callbacks.
    pub fn clear(&self) {
        let mut callbacks = self.callbacks.write().unwrap();
        callbacks.clear();
    }

    /// Get the number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.read().unwrap().len()
    }

    /// Check if no callbacks are registered.
    pub fn is_empty(&self) -> bool {
        self.callbacks.read().unwrap().is_empty()
    }
}

impl Default for ValidationCallbackRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ValidationCallbackRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationCallbackRegistry")
            .field("callback_count", &self.len())
            .finish()
    }
}

// ============================================================================
// ValidationLayer
// ============================================================================

/// Main validation layer interface.
///
/// Coordinates validation message processing, callback invocation,
/// and statistics tracking.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::*;
///
/// // Create from environment
/// let layer = ValidationLayer::from_env();
///
/// // Or with specific configuration
/// let layer = ValidationLayer::with_features(
///     ValidationLevel::Full,
///     ValidationFeatures::all_enabled(),
/// );
///
/// println!("Validation: {} with {}", layer.level(), layer.features());
/// ```
pub struct ValidationLayer {
    /// Current validation level.
    level: ValidationLevel,

    /// Enabled validation features.
    features: ValidationFeatures,

    /// Registered callbacks.
    callbacks: Arc<ValidationCallbackRegistry>,

    /// Total message count.
    message_count: AtomicU64,

    /// Error message count.
    error_count: AtomicU64,

    /// Warning message count.
    warning_count: AtomicU64,

    /// Whether to break on errors.
    break_on_error: AtomicBool,
}

impl ValidationLayer {
    /// Create a new validation layer with the specified level.
    ///
    /// Uses default features appropriate for the level.
    ///
    /// # Arguments
    ///
    /// * `level` - The validation level
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::new(ValidationLevel::Full);
    /// assert!(layer.level().is_enabled());
    /// ```
    pub fn new(level: ValidationLevel) -> Self {
        Self {
            level,
            features: ValidationFeatures::for_level(level),
            callbacks: Arc::new(ValidationCallbackRegistry::new()),
            message_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            warning_count: AtomicU64::new(0),
            break_on_error: AtomicBool::new(false),
        }
    }

    /// Create a validation layer with specific features.
    ///
    /// # Arguments
    ///
    /// * `level` - The validation level
    /// * `features` - The features to enable
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let mut features = ValidationFeatures::default();
    /// features.gpu_based_validation = true;
    ///
    /// let layer = ValidationLayer::with_features(ValidationLevel::Full, features);
    /// assert!(layer.features().gpu_based_validation);
    /// ```
    pub fn with_features(level: ValidationLevel, features: ValidationFeatures) -> Self {
        Self {
            level,
            features,
            callbacks: Arc::new(ValidationCallbackRegistry::new()),
            message_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            warning_count: AtomicU64::new(0),
            break_on_error: AtomicBool::new(false),
        }
    }

    /// Create a validation layer from environment variables.
    ///
    /// Reads `WGPU_VALIDATION` for level and `WGPU_VALIDATION_BREAK` for break-on-error.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::from_env();
    /// ```
    pub fn from_env() -> Self {
        let level = ValidationLevel::from_env();
        let mut layer = Self::new(level);

        // Check for break-on-error
        if let Ok(val) = env::var("WGPU_VALIDATION_BREAK") {
            if val == "1" || val.eq_ignore_ascii_case("true") {
                layer.break_on_error.store(true, Ordering::Relaxed);
            }
        }

        layer
    }

    /// Get the current validation level.
    #[inline]
    pub fn level(&self) -> ValidationLevel {
        self.level
    }

    /// Get the enabled features.
    #[inline]
    pub fn features(&self) -> &ValidationFeatures {
        &self.features
    }

    /// Check if validation is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.level.is_enabled()
    }

    /// Register a callback for validation messages.
    ///
    /// # Arguments
    ///
    /// * `callback` - The callback function
    ///
    /// # Returns
    ///
    /// Index of the registered callback.
    pub fn register_callback(&self, callback: ValidationCallbackFn) -> usize {
        self.callbacks.register(callback)
    }

    /// Get the callback registry.
    pub fn callbacks(&self) -> &Arc<ValidationCallbackRegistry> {
        &self.callbacks
    }

    /// Process a validation message.
    ///
    /// Updates statistics, invokes callbacks, and optionally breaks
    /// on errors.
    ///
    /// # Arguments
    ///
    /// * `message` - The validation message to process
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::new(ValidationLevel::Full);
    ///
    /// let msg = ValidationMessage::warning("Unused bind group slot");
    /// layer.on_message(&msg);
    ///
    /// assert_eq!(layer.warning_count(), 1);
    /// ```
    pub fn on_message(&self, message: &ValidationMessage) {
        // Check severity threshold
        if !message.meets_threshold(self.level.severity_threshold()) {
            return;
        }

        // Update counts
        self.message_count.fetch_add(1, Ordering::Relaxed);

        match message.severity {
            ValidationSeverity::Error => {
                self.error_count.fetch_add(1, Ordering::Relaxed);
            }
            ValidationSeverity::Warning => {
                self.warning_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        // Invoke callbacks
        self.callbacks.invoke(message);

        // Log the message
        message.log();

        // Break on error if configured
        if message.is_error() && self.break_on_error.load(Ordering::Relaxed) {
            // In debug builds, trigger a breakpoint
            #[cfg(debug_assertions)]
            {
                // Use debugger break if available
                // std::intrinsics::breakpoint() is unstable, so we use a panic in debug
                log::error!("Validation error - would break here if debugger attached");
            }
        }
    }

    /// Get the total message count.
    #[inline]
    pub fn message_count(&self) -> u64 {
        self.message_count.load(Ordering::Relaxed)
    }

    /// Get the error count.
    #[inline]
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Get the warning count.
    #[inline]
    pub fn warning_count(&self) -> u64 {
        self.warning_count.load(Ordering::Relaxed)
    }

    /// Check if any errors have occurred.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.error_count.load(Ordering::Relaxed) > 0
    }

    /// Check if any warnings have occurred.
    #[inline]
    pub fn has_warnings(&self) -> bool {
        self.warning_count.load(Ordering::Relaxed) > 0
    }

    /// Reset all counters.
    pub fn reset_counts(&self) {
        self.message_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.warning_count.store(0, Ordering::Relaxed);
    }

    /// Get a summary string of validation state.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::new(ValidationLevel::Full);
    /// println!("{}", layer.summary());
    /// // Output: "Validation: Full (0 errors, 0 warnings, 0 total)"
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "Validation: {} ({} errors, {} warnings, {} total)",
            self.level,
            self.error_count(),
            self.warning_count(),
            self.message_count()
        )
    }

    /// Set whether to break on errors.
    pub fn set_break_on_error(&self, enabled: bool) {
        self.break_on_error.store(enabled, Ordering::Relaxed);
    }

    /// Check if break-on-error is enabled.
    #[inline]
    pub fn break_on_error(&self) -> bool {
        self.break_on_error.load(Ordering::Relaxed)
    }

    /// Create a validation scope for tracking messages within a region.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scope for debugging
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::new(ValidationLevel::Full);
    ///
    /// {
    ///     let scope = layer.scope("Shadow Pass");
    ///     // GPU operations...
    /// } // Scope ends, reports any errors
    /// ```
    pub fn scope(&self, name: &str) -> ValidationScope<'_> {
        ValidationScope::new(self, name)
    }
}

impl Default for ValidationLayer {
    fn default() -> Self {
        Self::from_env()
    }
}

impl fmt::Debug for ValidationLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationLayer")
            .field("level", &self.level)
            .field("features", &self.features)
            .field("message_count", &self.message_count())
            .field("error_count", &self.error_count())
            .field("warning_count", &self.warning_count())
            .field("break_on_error", &self.break_on_error())
            .finish()
    }
}

// ============================================================================
// ValidationScope
// ============================================================================

/// RAII guard for scoped validation tracking.
///
/// Tracks validation messages generated within a scope and optionally
/// reports errors on drop.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::validation::*;
///
/// let layer = ValidationLayer::new(ValidationLevel::Full);
///
/// {
///     let scope = ValidationScope::new(&layer, "Render Pass");
///     // GPU operations...
///     // Scope tracks start/end message counts
/// } // Reports if errors occurred in this scope
/// ```
pub struct ValidationScope<'a> {
    /// Reference to the validation layer.
    layer: &'a ValidationLayer,

    /// Name of the scope.
    name: String,

    /// Error count at scope start.
    start_errors: u64,

    /// Warning count at scope start.
    start_warnings: u64,

    /// Message count at scope start.
    start_messages: u64,

    /// When the scope was created.
    start_time: Instant,

    /// Whether to report on drop.
    report_on_drop: bool,
}

impl<'a> ValidationScope<'a> {
    /// Create a new validation scope.
    ///
    /// # Arguments
    ///
    /// * `layer` - The validation layer to track
    /// * `name` - Name of the scope for debugging
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::validation::*;
    ///
    /// let layer = ValidationLayer::new(ValidationLevel::Full);
    /// let scope = ValidationScope::new(&layer, "Compute Pass");
    /// ```
    pub fn new(layer: &'a ValidationLayer, name: impl Into<String>) -> Self {
        Self {
            layer,
            name: name.into(),
            start_errors: layer.error_count(),
            start_warnings: layer.warning_count(),
            start_messages: layer.message_count(),
            start_time: Instant::now(),
            report_on_drop: true,
        }
    }

    /// Disable reporting on drop.
    pub fn silent(mut self) -> Self {
        self.report_on_drop = false;
        self
    }

    /// Get the scope name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of errors in this scope.
    #[inline]
    pub fn scope_errors(&self) -> u64 {
        self.layer.error_count().saturating_sub(self.start_errors)
    }

    /// Get the number of warnings in this scope.
    #[inline]
    pub fn scope_warnings(&self) -> u64 {
        self.layer.warning_count().saturating_sub(self.start_warnings)
    }

    /// Get the number of messages in this scope.
    #[inline]
    pub fn scope_messages(&self) -> u64 {
        self.layer.message_count().saturating_sub(self.start_messages)
    }

    /// Check if any errors occurred in this scope.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.scope_errors() > 0
    }

    /// Check if any warnings occurred in this scope.
    #[inline]
    pub fn has_warnings(&self) -> bool {
        self.scope_warnings() > 0
    }

    /// Get elapsed time since scope start.
    #[inline]
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get a summary of validation in this scope.
    pub fn summary(&self) -> String {
        format!(
            "ValidationScope '{}': {} errors, {} warnings, {} messages in {:?}",
            self.name,
            self.scope_errors(),
            self.scope_warnings(),
            self.scope_messages(),
            self.elapsed()
        )
    }

    /// Manually end the scope and get results.
    ///
    /// Prevents the automatic report on drop.
    pub fn end(mut self) -> ValidationScopeResult {
        self.report_on_drop = false;
        ValidationScopeResult {
            name: self.name.clone(),
            errors: self.scope_errors(),
            warnings: self.scope_warnings(),
            messages: self.scope_messages(),
            elapsed: self.elapsed(),
        }
    }
}

impl<'a> Drop for ValidationScope<'a> {
    fn drop(&mut self) {
        if self.report_on_drop && self.has_errors() {
            log::error!(
                "Validation scope '{}' completed with {} errors, {} warnings in {:?}",
                self.name,
                self.scope_errors(),
                self.scope_warnings(),
                self.elapsed()
            );
        } else if self.report_on_drop && self.has_warnings() {
            log::warn!(
                "Validation scope '{}' completed with {} warnings in {:?}",
                self.name,
                self.scope_warnings(),
                self.elapsed()
            );
        }
    }
}

impl<'a> fmt::Debug for ValidationScope<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationScope")
            .field("name", &self.name)
            .field("errors", &self.scope_errors())
            .field("warnings", &self.scope_warnings())
            .field("messages", &self.scope_messages())
            .field("elapsed", &self.elapsed())
            .finish()
    }
}

// ============================================================================
// ValidationScopeResult
// ============================================================================

/// Result of a validation scope after it ends.
#[derive(Debug, Clone)]
pub struct ValidationScopeResult {
    /// Name of the scope.
    pub name: String,

    /// Number of errors in the scope.
    pub errors: u64,

    /// Number of warnings in the scope.
    pub warnings: u64,

    /// Total messages in the scope.
    pub messages: u64,

    /// Time elapsed during the scope.
    pub elapsed: std::time::Duration,
}

impl ValidationScopeResult {
    /// Check if the scope had any errors.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    /// Check if the scope had any warnings.
    #[inline]
    pub fn has_warnings(&self) -> bool {
        self.warnings > 0
    }

    /// Check if the scope was clean (no errors or warnings).
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.errors == 0 && self.warnings == 0
    }
}

impl fmt::Display for ValidationScopeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "'{}': {} errors, {} warnings in {:?}",
            self.name, self.errors, self.warnings, self.elapsed
        )
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ValidationLevel tests ----

    #[test]
    fn test_validation_level_from_str() {
        assert_eq!(ValidationLevel::from_str("disabled"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("none"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("off"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("0"), ValidationLevel::Disabled);
        assert_eq!(ValidationLevel::from_str("false"), ValidationLevel::Disabled);

        assert_eq!(ValidationLevel::from_str("basic"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("min"), ValidationLevel::Basic);
        assert_eq!(ValidationLevel::from_str("1"), ValidationLevel::Basic);

        assert_eq!(ValidationLevel::from_str("full"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("on"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("true"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("2"), ValidationLevel::Full);

        assert_eq!(ValidationLevel::from_str("verbose"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("max"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("debug"), ValidationLevel::Verbose);
        assert_eq!(ValidationLevel::from_str("3"), ValidationLevel::Verbose);

        // Case insensitive
        assert_eq!(ValidationLevel::from_str("FULL"), ValidationLevel::Full);
        assert_eq!(ValidationLevel::from_str("VeRbOsE"), ValidationLevel::Verbose);

        // Unknown defaults to Basic
        assert_eq!(ValidationLevel::from_str("unknown"), ValidationLevel::Basic);
    }

    #[test]
    fn test_validation_level_is_enabled() {
        assert!(!ValidationLevel::Disabled.is_enabled());
        assert!(ValidationLevel::Basic.is_enabled());
        assert!(ValidationLevel::Full.is_enabled());
        assert!(ValidationLevel::Verbose.is_enabled());
    }

    #[test]
    fn test_validation_level_severity_threshold() {
        assert_eq!(ValidationLevel::Disabled.severity_threshold(), ValidationSeverity::Error);
        assert_eq!(ValidationLevel::Basic.severity_threshold(), ValidationSeverity::Warning);
        assert_eq!(ValidationLevel::Full.severity_threshold(), ValidationSeverity::Info);
        assert_eq!(ValidationLevel::Verbose.severity_threshold(), ValidationSeverity::Verbose);
    }

    #[test]
    fn test_validation_level_ordering() {
        assert!(ValidationLevel::Disabled < ValidationLevel::Basic);
        assert!(ValidationLevel::Basic < ValidationLevel::Full);
        assert!(ValidationLevel::Full < ValidationLevel::Verbose);
    }

    #[test]
    fn test_validation_level_display() {
        assert_eq!(format!("{}", ValidationLevel::Disabled), "Disabled");
        assert_eq!(format!("{}", ValidationLevel::Full), "Full");
    }

    // ---- ValidationFeatures tests ----

    #[test]
    fn test_validation_features_default() {
        let features = ValidationFeatures::default();
        assert!(!features.gpu_based_validation);
        assert!(!features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
    }

    #[test]
    fn test_validation_features_all_enabled() {
        let features = ValidationFeatures::all_enabled();
        assert!(features.gpu_based_validation);
        assert!(features.synchronization_validation);
        assert!(features.shader_validation);
        assert!(features.descriptor_indexing_validation);
        assert!(features.best_practices_warnings);
        assert!(features.printf_to_stdout);
    }

    #[test]
    fn test_validation_features_for_level() {
        let disabled = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert!(!disabled.any_enabled());

        let basic = ValidationFeatures::for_level(ValidationLevel::Basic);
        assert!(basic.shader_validation);
        assert!(!basic.gpu_based_validation);

        let full = ValidationFeatures::for_level(ValidationLevel::Full);
        assert!(full.synchronization_validation);
        assert!(full.best_practices_warnings);

        let verbose = ValidationFeatures::for_level(ValidationLevel::Verbose);
        assert!(verbose.gpu_based_validation);
        assert!(verbose.printf_to_stdout);
    }

    #[test]
    fn test_validation_features_enabled_count() {
        let none = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert_eq!(none.enabled_count(), 0);

        let all = ValidationFeatures::all_enabled();
        assert_eq!(all.enabled_count(), 6);
    }

    #[test]
    fn test_validation_features_display() {
        let none = ValidationFeatures::for_level(ValidationLevel::Disabled);
        assert_eq!(format!("{}", none), "None");

        let features = ValidationFeatures {
            gpu_based_validation: true,
            shader_validation: true,
            ..Default::default()
        };
        let display = format!("{}", features);
        assert!(display.contains("GPU"));
        assert!(display.contains("Shader"));
    }

    // ---- ValidationSeverity tests ----

    #[test]
    fn test_validation_severity_as_log_level() {
        assert_eq!(ValidationSeverity::Verbose.as_log_level(), log::Level::Trace);
        assert_eq!(ValidationSeverity::Info.as_log_level(), log::Level::Info);
        assert_eq!(ValidationSeverity::Warning.as_log_level(), log::Level::Warn);
        assert_eq!(ValidationSeverity::Error.as_log_level(), log::Level::Error);
    }

    #[test]
    fn test_validation_severity_should_break() {
        assert!(!ValidationSeverity::Verbose.should_break());
        assert!(!ValidationSeverity::Info.should_break());
        assert!(!ValidationSeverity::Warning.should_break());
        assert!(ValidationSeverity::Error.should_break());
    }

    #[test]
    fn test_validation_severity_meets_threshold() {
        assert!(ValidationSeverity::Error.meets_threshold(ValidationSeverity::Warning));
        assert!(ValidationSeverity::Warning.meets_threshold(ValidationSeverity::Warning));
        assert!(!ValidationSeverity::Info.meets_threshold(ValidationSeverity::Warning));
        assert!(ValidationSeverity::Verbose.meets_threshold(ValidationSeverity::Verbose));
    }

    #[test]
    fn test_validation_severity_ordering() {
        assert!(ValidationSeverity::Verbose < ValidationSeverity::Info);
        assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
        assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
    }

    // ---- ValidationMessageType tests ----

    #[test]
    fn test_validation_message_type_is_error() {
        assert!(ValidationMessageType::Validation.is_error());
        assert!(!ValidationMessageType::General.is_error());
        assert!(!ValidationMessageType::Performance.is_error());
        assert!(!ValidationMessageType::DebugMarker.is_error());
    }

    #[test]
    fn test_validation_message_type_default_severity() {
        assert_eq!(ValidationMessageType::General.default_severity(), ValidationSeverity::Info);
        assert_eq!(ValidationMessageType::Validation.default_severity(), ValidationSeverity::Error);
        assert_eq!(ValidationMessageType::Performance.default_severity(), ValidationSeverity::Warning);
        assert_eq!(ValidationMessageType::DebugMarker.default_severity(), ValidationSeverity::Verbose);
    }

    // ---- ValidationObjectType tests ----

    #[test]
    fn test_validation_object_type_categories() {
        assert!(ValidationObjectType::RenderPipeline.is_pipeline());
        assert!(ValidationObjectType::ComputePipeline.is_pipeline());
        assert!(ValidationObjectType::PipelineLayout.is_pipeline());
        assert!(!ValidationObjectType::Buffer.is_pipeline());

        assert!(ValidationObjectType::Buffer.is_resource());
        assert!(ValidationObjectType::Texture.is_resource());
        assert!(!ValidationObjectType::RenderPipeline.is_resource());

        assert!(ValidationObjectType::BindGroup.is_binding());
        assert!(ValidationObjectType::BindGroupLayout.is_binding());
        assert!(!ValidationObjectType::Sampler.is_binding());

        assert!(ValidationObjectType::CommandBuffer.is_command());
        assert!(ValidationObjectType::RenderPass.is_command());
        assert!(!ValidationObjectType::Queue.is_command());
    }

    // ---- ValidationObject tests ----

    #[test]
    fn test_validation_object_creation() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234);
        assert_eq!(obj.object_type, ValidationObjectType::Buffer);
        assert_eq!(obj.handle, 0x1234);
        assert!(obj.name.is_none());
    }

    #[test]
    fn test_validation_object_with_name() {
        let obj = ValidationObject::new(ValidationObjectType::Texture, 0x5678)
            .with_name("ShadowMap");
        assert_eq!(obj.name.as_deref(), Some("ShadowMap"));
        assert!(obj.has_name());
    }

    #[test]
    fn test_validation_object_display() {
        let obj = ValidationObject::new(ValidationObjectType::Buffer, 0x1234)
            .with_name("VertexData");
        let display = format!("{}", obj);
        assert!(display.contains("Buffer"));
        assert!(display.contains("0x1234"));
        assert!(display.contains("VertexData"));
    }

    // ---- ValidationMessage tests ----

    #[test]
    fn test_validation_message_new() {
        let msg = ValidationMessage::new(
            ValidationSeverity::Warning,
            ValidationMessageType::Performance,
            "Test message",
        );
        assert_eq!(msg.severity, ValidationSeverity::Warning);
        assert_eq!(msg.message_type, ValidationMessageType::Performance);
        assert_eq!(msg.message, "Test message");
        assert!(msg.objects.is_empty());
        assert!(msg.location.is_none());
    }

    #[test]
    fn test_validation_message_factories() {
        let error = ValidationMessage::error("Error message");
        assert_eq!(error.severity, ValidationSeverity::Error);
        assert!(error.message_type.is_error());

        let warning = ValidationMessage::warning("Warning message");
        assert_eq!(warning.severity, ValidationSeverity::Warning);

        let perf = ValidationMessage::performance("Perf message");
        assert_eq!(perf.message_type, ValidationMessageType::Performance);
    }

    #[test]
    fn test_validation_message_builders() {
        let msg = ValidationMessage::error("Test")
            .with_id(42)
            .with_object(ValidationObject::new(ValidationObjectType::Buffer, 0x1))
            .with_location(SourceLocation::new().with_file("test.rs").with_line(10));

        assert_eq!(msg.message_id, Some(42));
        assert_eq!(msg.objects.len(), 1);
        assert!(msg.location.is_some());
    }

    #[test]
    fn test_validation_message_is_checks() {
        let error = ValidationMessage::error("Error");
        assert!(error.is_error());
        assert!(!error.is_warning());

        let warning = ValidationMessage::warning("Warning");
        assert!(warning.is_warning());
        assert!(!warning.is_error());
    }

    // ---- ValidationCallbackRegistry tests ----

    #[test]
    fn test_callback_registry_basic() {
        let registry = ValidationCallbackRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();
        registry.register(Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        let msg = ValidationMessage::info("Test");
        registry.invoke(&msg);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_registry_multiple() {
        let registry = ValidationCallbackRegistry::new();
        let counter = Arc::new(AtomicU64::new(0));

        for _ in 0..3 {
            let c = counter.clone();
            registry.register(Box::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }

        let msg = ValidationMessage::info("Test");
        registry.invoke(&msg);

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_callback_registry_clear() {
        let registry = ValidationCallbackRegistry::new();
        registry.register(Box::new(|_| {}));
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert!(registry.is_empty());
    }

    // ---- ValidationLayer tests ----

    #[test]
    fn test_validation_layer_new() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        assert_eq!(layer.level(), ValidationLevel::Full);
        assert!(layer.is_enabled());
        assert!(!layer.has_errors());
        assert_eq!(layer.message_count(), 0);
    }

    #[test]
    fn test_validation_layer_with_features() {
        let features = ValidationFeatures::all_enabled();
        let layer = ValidationLayer::with_features(ValidationLevel::Full, features);
        assert!(layer.features().gpu_based_validation);
    }

    #[test]
    fn test_validation_layer_on_message() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        let warning = ValidationMessage::warning("Test warning");
        layer.on_message(&warning);
        assert_eq!(layer.warning_count(), 1);
        assert!(!layer.has_errors());

        let error = ValidationMessage::error("Test error");
        layer.on_message(&error);
        assert_eq!(layer.error_count(), 1);
        assert!(layer.has_errors());

        assert_eq!(layer.message_count(), 2);
    }

    #[test]
    fn test_validation_layer_threshold_filtering() {
        let layer = ValidationLayer::new(ValidationLevel::Basic);

        // Info messages should be filtered at Basic level
        let info = ValidationMessage::info("Info message");
        layer.on_message(&info);
        assert_eq!(layer.message_count(), 0);

        // Warnings should pass through
        let warning = ValidationMessage::warning("Warning message");
        layer.on_message(&warning);
        assert_eq!(layer.message_count(), 1);
    }

    #[test]
    fn test_validation_layer_reset_counts() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        layer.on_message(&ValidationMessage::error("Error"));
        layer.on_message(&ValidationMessage::warning("Warning"));

        assert!(layer.has_errors());
        assert!(layer.has_warnings());

        layer.reset_counts();

        assert!(!layer.has_errors());
        assert!(!layer.has_warnings());
        assert_eq!(layer.message_count(), 0);
    }

    #[test]
    fn test_validation_layer_summary() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        layer.on_message(&ValidationMessage::error("Error"));
        layer.on_message(&ValidationMessage::warning("Warning"));

        let summary = layer.summary();
        assert!(summary.contains("Full"));
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
        assert!(summary.contains("2 total"));
    }

    #[test]
    fn test_validation_layer_break_on_error() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        assert!(!layer.break_on_error());

        layer.set_break_on_error(true);
        assert!(layer.break_on_error());
    }

    // ---- ValidationScope tests ----

    #[test]
    fn test_validation_scope_basic() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        {
            let scope = ValidationScope::new(&layer, "Test Scope").silent();
            assert_eq!(scope.name(), "Test Scope");
            assert_eq!(scope.scope_errors(), 0);
            assert_eq!(scope.scope_warnings(), 0);
        }
    }

    #[test]
    fn test_validation_scope_tracking() {
        let layer = ValidationLayer::new(ValidationLevel::Full);

        // Add some messages before the scope
        layer.on_message(&ValidationMessage::error("Pre-scope error"));

        {
            let scope = ValidationScope::new(&layer, "Test Scope").silent();

            // Messages in scope
            layer.on_message(&ValidationMessage::warning("In-scope warning"));
            layer.on_message(&ValidationMessage::error("In-scope error"));

            assert_eq!(scope.scope_errors(), 1);
            assert_eq!(scope.scope_warnings(), 1);
            assert!(scope.has_errors());
            assert!(scope.has_warnings());
        }

        // Total should include all messages
        assert_eq!(layer.error_count(), 2);
    }

    #[test]
    fn test_validation_scope_end() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Test").silent();

        layer.on_message(&ValidationMessage::warning("Warning"));

        let result = scope.end();
        assert_eq!(result.warnings, 1);
        assert_eq!(result.name, "Test");
        assert!(!result.is_clean());
        assert!(!result.has_errors());
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validation_scope_result_clean() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Clean").silent();
        let result = scope.end();
        assert!(result.is_clean());
    }

    // ---- Additional edge case tests ----

    #[test]
    fn test_validation_object_unknown() {
        let obj = ValidationObject::unknown(0xDEADBEEF);
        assert_eq!(obj.object_type, ValidationObjectType::Unknown);
        assert_eq!(obj.handle, 0xDEADBEEF);
    }

    #[test]
    fn test_validation_message_with_multiple_objects() {
        let msg = ValidationMessage::error("Multiple objects")
            .with_object(ValidationObject::new(ValidationObjectType::Buffer, 1))
            .with_objects(vec![
                ValidationObject::new(ValidationObjectType::Texture, 2),
                ValidationObject::new(ValidationObjectType::Sampler, 3),
            ]);

        assert_eq!(msg.objects.len(), 3);
    }

    #[test]
    fn test_validation_scope_elapsed_time() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let scope = ValidationScope::new(&layer, "Timed").silent();

        // Just verify elapsed works (should complete without panic)
        let _elapsed = scope.elapsed();
    }

    #[test]
    fn test_validation_layer_register_callback() {
        let layer = ValidationLayer::new(ValidationLevel::Full);
        let received = Arc::new(AtomicBool::new(false));
        let received_clone = received.clone();

        layer.register_callback(Box::new(move |msg| {
            if msg.is_error() {
                received_clone.store(true, Ordering::SeqCst);
            }
        }));

        layer.on_message(&ValidationMessage::error("Test"));
        assert!(received.load(Ordering::SeqCst));
    }
}
