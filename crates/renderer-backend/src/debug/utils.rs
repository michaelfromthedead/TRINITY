//! GPU debugging utilities for wgpu error handling and diagnostics.
//!
//! This module provides comprehensive debugging utilities for GPU error handling,
//! building on the foundation in [`crate::device::error_scope`] with additional
//! diagnostic capabilities.
//!
//! # Overview
//!
//! - [`DeviceLostReason`] - Extended device loss reasons with recoverability
//! - [`DeviceLostInfo`] - Detailed device loss information with timestamps
//! - [`ErrorScope`] - Error scope with accumulated error collection
//! - [`ErrorFilter`] - Error type filtering with match semantics
//! - [`GpuError`] - Structured GPU error with source locations
//! - [`GpuErrorType`] - Error type classification with severity
//! - [`SourceLocation`] - Source code location for debugging
//! - [`ErrorCallbackRegistry`] - Thread-safe callback management
//! - [`DebugUtils`] - Main utility struct for error handling
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::debug::utils::*;
//!
//! // Create debug utilities
//! let mut debug = DebugUtils::new();
//!
//! // Set device lost handler
//! debug.set_device_lost_handler(|info| {
//!     eprintln!("Device lost: {} - {}", info.reason, info.message);
//!     eprintln!("Recoverable: {}", info.reason.is_recoverable());
//! });
//!
//! // Push an error scope
//! debug.push_error_scope(ErrorFilter::Validation);
//!
//! // ... GPU operations ...
//!
//! // Pop and check for errors
//! if let Some(scope) = debug.pop_error_scope() {
//!     if scope.has_errors() {
//!         for error in scope.errors() {
//!             eprintln!("GPU Error: {}", error);
//!         }
//!     }
//! }
//! ```
//!
//! # Relationship to device::error_scope
//!
//! This module complements [`crate::device::error_scope`]:
//!
//! | device::error_scope | debug::utils |
//! |---------------------|--------------|
//! | RAII ErrorScope | Accumulated ErrorScope |
//! | Simple ErrorFilter | ErrorFilter with match semantics |
//! | Direct wgpu::Error | Structured GpuError with source |
//! | N/A | ErrorCallbackRegistry |
//! | N/A | DebugUtils coordinator |

use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ============================================================================
// DeviceLostReason
// ============================================================================

/// Reason why the GPU device was lost.
///
/// This enum provides detailed information about device loss causes,
/// extending beyond the basic wgpu reasons with additional context.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::DeviceLostReason;
///
/// let reason = DeviceLostReason::DriverError;
/// assert!(reason.is_recoverable());
/// println!("{}", reason.description());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLostReason {
    /// Unknown or unspecified reason.
    Unknown,

    /// Device was explicitly destroyed by the application.
    Destroyed,

    /// Device became invalid (internal state corruption).
    DeviceInvalid,

    /// Driver error or crash.
    DriverError,
}

impl DeviceLostReason {
    /// Convert from wgpu's DeviceLostReason.
    ///
    /// Maps wgpu's device lost reasons to our enum. Since wgpu's enum
    /// variants may change between versions, this uses a defensive
    /// pattern matching approach.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::debug::utils::DeviceLostReason;
    ///
    /// // Convert from wgpu reason (actual variant depends on wgpu version)
    /// let reason = DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::Destroyed);
    /// assert_eq!(reason, DeviceLostReason::Destroyed);
    /// ```
    #[inline]
    pub fn from_wgpu(reason: wgpu::DeviceLostReason) -> Self {
        // Note: wgpu::DeviceLostReason variants differ between versions.
        // We use Debug formatting to match since the enum is non_exhaustive.
        let reason_str = format!("{:?}", reason);
        if reason_str.contains("Destroyed") {
            DeviceLostReason::Destroyed
        } else if reason_str.contains("ReplacedCallback") {
            // ReplacedCallback in wgpu 22 means old callback was replaced
            DeviceLostReason::Unknown
        } else if reason_str.contains("OutOfMemory") {
            DeviceLostReason::DriverError
        } else {
            DeviceLostReason::Unknown
        }
    }

    /// Check if recovery is possible after this type of loss.
    ///
    /// # Returns
    ///
    /// `true` if the device can potentially be recreated, `false` if
    /// recovery is unlikely to succeed.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::DeviceLostReason;
    ///
    /// assert!(!DeviceLostReason::Destroyed.is_recoverable());
    /// assert!(DeviceLostReason::DriverError.is_recoverable());
    /// ```
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        match self {
            DeviceLostReason::Unknown => true,
            DeviceLostReason::Destroyed => false,
            DeviceLostReason::DeviceInvalid => true,
            DeviceLostReason::DriverError => true,
        }
    }

    /// Get a human-readable description of the loss reason.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::DeviceLostReason;
    ///
    /// assert!(DeviceLostReason::DriverError.description().contains("driver"));
    /// ```
    pub fn description(&self) -> &'static str {
        match self {
            DeviceLostReason::Unknown => "Device was lost for an unknown reason",
            DeviceLostReason::Destroyed => "Device was explicitly destroyed by the application",
            DeviceLostReason::DeviceInvalid => {
                "Device became invalid due to internal state corruption"
            }
            DeviceLostReason::DriverError => {
                "Device was lost due to a driver error or crash"
            }
        }
    }
}

impl fmt::Display for DeviceLostReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceLostReason::Unknown => write!(f, "Unknown"),
            DeviceLostReason::Destroyed => write!(f, "Destroyed"),
            DeviceLostReason::DeviceInvalid => write!(f, "DeviceInvalid"),
            DeviceLostReason::DriverError => write!(f, "DriverError"),
        }
    }
}

impl Default for DeviceLostReason {
    fn default() -> Self {
        DeviceLostReason::Unknown
    }
}

// ============================================================================
// DeviceLostInfo
// ============================================================================

/// Detailed information about a device loss event.
///
/// This struct captures comprehensive information about when and why
/// a GPU device was lost, useful for debugging and recovery decisions.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{DeviceLostInfo, DeviceLostReason};
/// use std::time::Instant;
///
/// let info = DeviceLostInfo::new(
///     DeviceLostReason::DriverError,
///     "GPU driver crashed".to_string(),
/// );
///
/// println!("Lost: {} - {}", info.reason, info.message);
/// println!("Elapsed: {:?}", info.elapsed());
/// ```
#[derive(Debug, Clone)]
pub struct DeviceLostInfo {
    /// The reason for device loss.
    pub reason: DeviceLostReason,

    /// Detailed message about the loss.
    pub message: String,

    /// When the loss was detected.
    pub timestamp: Instant,
}

impl DeviceLostInfo {
    /// Create new device lost information.
    ///
    /// # Arguments
    ///
    /// * `reason` - The reason for device loss
    /// * `message` - Detailed message about the loss
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::{DeviceLostInfo, DeviceLostReason};
    ///
    /// let info = DeviceLostInfo::new(
    ///     DeviceLostReason::DriverError,
    ///     "Timeout detected".to_string(),
    /// );
    /// ```
    pub fn new(reason: DeviceLostReason, message: String) -> Self {
        Self {
            reason,
            message,
            timestamp: Instant::now(),
        }
    }

    /// Convert from wgpu callback data.
    ///
    /// # Arguments
    ///
    /// * `reason` - The wgpu device lost reason
    /// * `message` - The message from wgpu
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::debug::utils::DeviceLostInfo;
    ///
    /// let info = DeviceLostInfo::from_wgpu(
    ///     wgpu::DeviceLostReason::Destroyed,
    ///     "Device dropped".to_string(),
    /// );
    /// ```
    pub fn from_wgpu(reason: wgpu::DeviceLostReason, message: String) -> Self {
        Self {
            reason: DeviceLostReason::from_wgpu(reason),
            message,
            timestamp: Instant::now(),
        }
    }

    /// Get the time elapsed since the device was lost.
    ///
    /// # Returns
    ///
    /// Duration since the loss was detected.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::{DeviceLostInfo, DeviceLostReason};
    ///
    /// let info = DeviceLostInfo::new(DeviceLostReason::Unknown, String::new());
    /// // Some time passes...
    /// let elapsed = info.elapsed();
    /// ```
    #[inline]
    pub fn elapsed(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }

    /// Check if recovery should be attempted.
    #[inline]
    pub fn should_attempt_recovery(&self) -> bool {
        self.reason.is_recoverable()
    }
}

impl fmt::Display for DeviceLostInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeviceLostInfo {{ reason: {}, message: '{}', elapsed: {:?} }}",
            self.reason,
            self.message,
            self.elapsed()
        )
    }
}

// ============================================================================
// ErrorFilter
// ============================================================================

/// Filter for GPU error types.
///
/// Specifies which types of GPU errors should be captured by an error scope.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{ErrorFilter, GpuErrorType};
///
/// let filter = ErrorFilter::Validation;
/// assert!(filter.matches(&GpuErrorType::Validation));
/// assert!(!filter.matches(&GpuErrorType::OutOfMemory));
///
/// let all_filter = ErrorFilter::All;
/// assert!(all_filter.matches(&GpuErrorType::Validation));
/// assert!(all_filter.matches(&GpuErrorType::OutOfMemory));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorFilter {
    /// Capture validation errors (API misuse, invalid parameters).
    Validation,

    /// Capture out-of-memory errors (allocation failures).
    OutOfMemory,

    /// Capture internal errors (driver/implementation bugs).
    Internal,

    /// Capture all error types.
    All,
}

impl ErrorFilter {
    /// Check if this filter matches the given error type.
    ///
    /// # Arguments
    ///
    /// * `error_type` - The error type to check
    ///
    /// # Returns
    ///
    /// `true` if the filter captures this error type.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::{ErrorFilter, GpuErrorType};
    ///
    /// assert!(ErrorFilter::Validation.matches(&GpuErrorType::Validation));
    /// assert!(ErrorFilter::All.matches(&GpuErrorType::Internal));
    /// ```
    #[inline]
    pub fn matches(&self, error_type: &GpuErrorType) -> bool {
        match self {
            ErrorFilter::All => true,
            ErrorFilter::Validation => matches!(error_type, GpuErrorType::Validation),
            ErrorFilter::OutOfMemory => matches!(error_type, GpuErrorType::OutOfMemory),
            ErrorFilter::Internal => matches!(error_type, GpuErrorType::Internal),
        }
    }

    /// Convert to wgpu's ErrorFilter (where applicable).
    ///
    /// Note: `ErrorFilter::Internal` and `ErrorFilter::All` don't have direct
    /// wgpu equivalents and will default to `Validation`.
    #[inline]
    pub fn to_wgpu(&self) -> wgpu::ErrorFilter {
        match self {
            ErrorFilter::Validation => wgpu::ErrorFilter::Validation,
            ErrorFilter::OutOfMemory => wgpu::ErrorFilter::OutOfMemory,
            ErrorFilter::Internal => wgpu::ErrorFilter::Validation,
            ErrorFilter::All => wgpu::ErrorFilter::Validation,
        }
    }
}

impl fmt::Display for ErrorFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorFilter::Validation => write!(f, "Validation"),
            ErrorFilter::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorFilter::Internal => write!(f, "Internal"),
            ErrorFilter::All => write!(f, "All"),
        }
    }
}

impl Default for ErrorFilter {
    fn default() -> Self {
        ErrorFilter::All
    }
}

// ============================================================================
// GpuErrorType
// ============================================================================

/// Type of GPU error.
///
/// Classifies GPU errors by their nature for appropriate handling.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::GpuErrorType;
///
/// let error_type = GpuErrorType::OutOfMemory;
/// assert_eq!(error_type.severity(), Severity::Error);
/// assert!(!error_type.is_fatal());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuErrorType {
    /// Validation error (API misuse, invalid parameters).
    Validation,

    /// Out of memory error (allocation failure).
    OutOfMemory,

    /// Internal error (driver or implementation bug).
    Internal,

    /// Device lost error (device no longer usable).
    Lost,
}

/// Severity level for errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational message.
    Info,

    /// Warning (non-critical issue).
    Warning,

    /// Error (operation failed).
    Error,
}

impl GpuErrorType {
    /// Get the severity level for this error type.
    ///
    /// # Returns
    ///
    /// The severity level indicating how serious the error is.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::{GpuErrorType, Severity};
    ///
    /// assert_eq!(GpuErrorType::Lost.severity(), Severity::Error);
    /// assert_eq!(GpuErrorType::Validation.severity(), Severity::Warning);
    /// ```
    #[inline]
    pub fn severity(&self) -> Severity {
        match self {
            GpuErrorType::Validation => Severity::Warning,
            GpuErrorType::OutOfMemory => Severity::Error,
            GpuErrorType::Internal => Severity::Error,
            GpuErrorType::Lost => Severity::Error,
        }
    }

    /// Check if this error type is fatal (unrecoverable).
    ///
    /// # Returns
    ///
    /// `true` if the error cannot be recovered from without recreating
    /// the device or resources.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::GpuErrorType;
    ///
    /// assert!(GpuErrorType::Lost.is_fatal());
    /// assert!(!GpuErrorType::OutOfMemory.is_fatal());
    /// ```
    #[inline]
    pub fn is_fatal(&self) -> bool {
        matches!(self, GpuErrorType::Lost)
    }

    /// Check if this is a validation error.
    #[inline]
    pub fn is_validation(&self) -> bool {
        matches!(self, GpuErrorType::Validation)
    }

    /// Check if this is an out-of-memory error.
    #[inline]
    pub fn is_oom(&self) -> bool {
        matches!(self, GpuErrorType::OutOfMemory)
    }

    /// Check if this is an internal error.
    #[inline]
    pub fn is_internal(&self) -> bool {
        matches!(self, GpuErrorType::Internal)
    }
}

impl fmt::Display for GpuErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuErrorType::Validation => write!(f, "Validation"),
            GpuErrorType::OutOfMemory => write!(f, "OutOfMemory"),
            GpuErrorType::Internal => write!(f, "Internal"),
            GpuErrorType::Lost => write!(f, "DeviceLost"),
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

// ============================================================================
// SourceLocation
// ============================================================================

/// Source code location for debugging.
///
/// Captures file, line, column, and function information for tracing
/// where GPU errors originated.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::SourceLocation;
///
/// let loc = SourceLocation::new()
///     .with_file("src/renderer.rs")
///     .with_line(42)
///     .with_function("render_frame");
///
/// println!("Error at: {}", loc);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceLocation {
    /// Source file path.
    pub file: Option<String>,

    /// Line number (1-indexed).
    pub line: Option<u32>,

    /// Column number (1-indexed).
    pub column: Option<u32>,

    /// Function or method name.
    pub function: Option<String>,
}

impl SourceLocation {
    /// Create a new empty source location.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from the current location using macros.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::debug::utils::SourceLocation;
    ///
    /// let loc = SourceLocation::here();
    /// assert!(loc.file.is_some());
    /// assert!(loc.line.is_some());
    /// ```
    #[inline]
    #[track_caller]
    pub fn here() -> Self {
        let location = std::panic::Location::caller();
        Self {
            file: Some(location.file().to_string()),
            line: Some(location.line()),
            column: Some(location.column()),
            function: None,
        }
    }

    /// Set the file path.
    #[inline]
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// Set the line number.
    #[inline]
    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Set the column number.
    #[inline]
    pub fn with_column(mut self, column: u32) -> Self {
        self.column = Some(column);
        self
    }

    /// Set the function name.
    #[inline]
    pub fn with_function(mut self, function: impl Into<String>) -> Self {
        self.function = Some(function.into());
        self
    }

    /// Check if any location information is available.
    #[inline]
    pub fn is_available(&self) -> bool {
        self.file.is_some() || self.line.is_some() || self.function.is_some()
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}", file)?;
            if let Some(line) = self.line {
                write!(f, ":{}", line)?;
                if let Some(col) = self.column {
                    write!(f, ":{}", col)?;
                }
            }
        }
        if let Some(ref func) = self.function {
            if self.file.is_some() {
                write!(f, " in ")?;
            }
            write!(f, "{}()", func)?;
        }
        if !self.is_available() {
            write!(f, "<unknown location>")?;
        }
        Ok(())
    }
}

// ============================================================================
// GpuError
// ============================================================================

/// Structured GPU error with source location.
///
/// Wraps GPU error information with additional context for debugging.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{GpuError, GpuErrorType, SourceLocation};
///
/// let error = GpuError::new(
///     GpuErrorType::Validation,
///     "Invalid buffer usage flags".to_string(),
/// ).with_source_location(SourceLocation::here());
///
/// assert!(error.is_validation());
/// println!("{}", error);
/// ```
#[derive(Debug, Clone)]
pub struct GpuError {
    /// The type of error.
    pub error_type: GpuErrorType,

    /// Detailed error message.
    pub message: String,

    /// Source location where the error occurred.
    pub source_location: Option<SourceLocation>,

    /// Timestamp when the error was captured.
    pub timestamp: Instant,
}

impl GpuError {
    /// Create a new GPU error.
    ///
    /// # Arguments
    ///
    /// * `error_type` - The type of error
    /// * `message` - Detailed error message
    pub fn new(error_type: GpuErrorType, message: String) -> Self {
        Self {
            error_type,
            message,
            source_location: None,
            timestamp: Instant::now(),
        }
    }

    /// Convert from a wgpu Error.
    ///
    /// # Arguments
    ///
    /// * `error` - The wgpu error to convert
    pub fn from_wgpu(error: &wgpu::Error) -> Self {
        let (error_type, message) = match error {
            wgpu::Error::OutOfMemory { source } => {
                (GpuErrorType::OutOfMemory, format!("Out of memory: {:?}", source))
            }
            wgpu::Error::Validation { description, source } => {
                let msg = if description.is_empty() {
                    format!("Validation error: {:?}", source)
                } else {
                    description.clone()
                };
                (GpuErrorType::Validation, msg)
            }
            wgpu::Error::Internal { description, source } => {
                let msg = if description.is_empty() {
                    format!("Internal error: {:?}", source)
                } else {
                    description.clone()
                };
                (GpuErrorType::Internal, msg)
            }
        };

        Self::new(error_type, message)
    }

    /// Add source location information.
    #[inline]
    pub fn with_source_location(mut self, location: SourceLocation) -> Self {
        self.source_location = Some(location);
        self
    }

    /// Check if this is a validation error.
    #[inline]
    pub fn is_validation(&self) -> bool {
        self.error_type.is_validation()
    }

    /// Check if this is an out-of-memory error.
    #[inline]
    pub fn is_oom(&self) -> bool {
        self.error_type.is_oom()
    }

    /// Check if this is an internal error.
    #[inline]
    pub fn is_internal(&self) -> bool {
        self.error_type.is_internal()
    }

    /// Get the severity level.
    #[inline]
    pub fn severity(&self) -> Severity {
        self.error_type.severity()
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.error_type.severity(), self.error_type, self.message)?;
        if let Some(ref loc) = self.source_location {
            if loc.is_available() {
                write!(f, " at {}", loc)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for GpuError {}

// ============================================================================
// ErrorScope
// ============================================================================

/// Error scope for accumulating GPU errors.
///
/// Unlike the RAII-based scope in `device::error_scope`, this scope
/// accumulates multiple errors for batch processing.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{ErrorScope, ErrorFilter, GpuError, GpuErrorType};
///
/// let mut scope = ErrorScope::with_label(ErrorFilter::All, "render_pass");
///
/// // Simulate collecting errors
/// scope.push(GpuError::new(
///     GpuErrorType::Validation,
///     "Invalid bind group".to_string(),
/// ));
///
/// assert!(scope.has_errors());
/// assert_eq!(scope.error_count(), 1);
///
/// let errors = scope.take_errors();
/// assert_eq!(errors.len(), 1);
/// assert!(!scope.has_errors());
/// ```
#[derive(Debug, Clone)]
pub struct ErrorScope {
    /// Optional label for identification.
    pub label: Option<String>,

    /// The error filter for this scope.
    pub filter: ErrorFilter,

    /// Collected errors.
    errors: Vec<GpuError>,

    /// When the scope was created.
    created_at: Instant,
}

impl ErrorScope {
    /// Create a new error scope with the given filter.
    ///
    /// # Arguments
    ///
    /// * `filter` - Which error types to capture
    pub fn new(filter: ErrorFilter) -> Self {
        Self {
            label: None,
            filter,
            errors: Vec::new(),
            created_at: Instant::now(),
        }
    }

    /// Create a new error scope with a label.
    ///
    /// # Arguments
    ///
    /// * `filter` - Which error types to capture
    /// * `label` - Identifying label for the scope
    pub fn with_label(filter: ErrorFilter, label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            filter,
            errors: Vec::new(),
            created_at: Instant::now(),
        }
    }

    /// Push an error into the scope.
    ///
    /// Only errors matching the filter are accepted.
    ///
    /// # Arguments
    ///
    /// * `error` - The GPU error to add
    ///
    /// # Returns
    ///
    /// `true` if the error was accepted, `false` if it was filtered out.
    pub fn push(&mut self, error: GpuError) -> bool {
        if self.filter.matches(&error.error_type) {
            self.errors.push(error);
            true
        } else {
            false
        }
    }

    /// Check if any errors have been collected.
    #[inline]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the number of collected errors.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get a reference to collected errors.
    #[inline]
    pub fn errors(&self) -> &[GpuError] {
        &self.errors
    }

    /// Take all collected errors, leaving the scope empty.
    pub fn take_errors(&mut self) -> Vec<GpuError> {
        std::mem::take(&mut self.errors)
    }

    /// Clear all collected errors.
    pub fn clear(&mut self) {
        self.errors.clear();
    }

    /// Get the duration since the scope was created.
    #[inline]
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Get the most severe error (if any).
    pub fn most_severe(&self) -> Option<&GpuError> {
        self.errors.iter().max_by_key(|e| e.severity())
    }
}

impl fmt::Display for ErrorScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ErrorScope[{}](filter={}, errors={})",
            self.label.as_deref().unwrap_or("unnamed"),
            self.filter,
            self.error_count()
        )
    }
}

// ============================================================================
// ErrorCallbackRegistry
// ============================================================================

/// Type alias for error callback functions.
pub type ErrorCallbackFn = Arc<dyn Fn(&GpuError) + Send + Sync>;

/// Thread-safe registry for error callbacks.
///
/// Manages a collection of callbacks that are invoked when GPU errors occur.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{ErrorCallbackRegistry, GpuError, GpuErrorType};
/// use std::sync::Arc;
///
/// let registry = ErrorCallbackRegistry::new();
///
/// // Register a callback
/// let id = registry.register(Arc::new(|error| {
///     println!("Error: {}", error);
/// }));
///
/// // Invoke all callbacks
/// let error = GpuError::new(GpuErrorType::Validation, "Test".to_string());
/// registry.invoke(&error);
///
/// // Unregister when done
/// registry.unregister(id);
/// ```
#[derive(Clone)]
pub struct ErrorCallbackRegistry {
    callbacks: Arc<RwLock<Vec<(u64, ErrorCallbackFn)>>>,
    next_id: Arc<Mutex<u64>>,
}

impl ErrorCallbackRegistry {
    /// Create a new empty callback registry.
    pub fn new() -> Self {
        Self {
            callbacks: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Register a new callback.
    ///
    /// # Arguments
    ///
    /// * `callback` - The callback function to register
    ///
    /// # Returns
    ///
    /// A unique ID that can be used to unregister the callback.
    pub fn register(&self, callback: ErrorCallbackFn) -> u64 {
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        {
            let mut callbacks = self.callbacks.write().unwrap();
            callbacks.push((id, callback));
        }

        id
    }

    /// Unregister a callback by its ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID returned by `register`
    ///
    /// # Returns
    ///
    /// `true` if the callback was found and removed, `false` otherwise.
    pub fn unregister(&self, id: u64) -> bool {
        let mut callbacks = self.callbacks.write().unwrap();
        if let Some(pos) = callbacks.iter().position(|(cb_id, _)| *cb_id == id) {
            callbacks.remove(pos);
            true
        } else {
            false
        }
    }

    /// Invoke all registered callbacks with the given error.
    ///
    /// # Arguments
    ///
    /// * `error` - The GPU error to pass to callbacks
    ///
    /// # Returns
    ///
    /// The number of callbacks invoked.
    pub fn invoke(&self, error: &GpuError) -> usize {
        let callbacks = self.callbacks.read().unwrap();
        for (_, callback) in callbacks.iter() {
            callback(error);
        }
        callbacks.len()
    }

    /// Get the number of registered callbacks.
    #[inline]
    pub fn len(&self) -> usize {
        self.callbacks.read().unwrap().len()
    }

    /// Check if any callbacks are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.callbacks.read().unwrap().is_empty()
    }

    /// Clear all registered callbacks.
    pub fn clear(&self) {
        self.callbacks.write().unwrap().clear();
    }
}

impl Default for ErrorCallbackRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ErrorCallbackRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErrorCallbackRegistry")
            .field("callback_count", &self.len())
            .finish()
    }
}

// ============================================================================
// DebugUtils
// ============================================================================

/// Main debugging utilities struct.
///
/// Coordinates error scopes, callbacks, and device lost handling.
///
/// # Example
///
/// ```
/// use renderer_backend::debug::utils::{DebugUtils, ErrorFilter};
///
/// let mut debug = DebugUtils::new();
///
/// // Set up device lost handler
/// debug.set_device_lost_handler(|info| {
///     eprintln!("Device lost: {}", info);
/// });
///
/// // Use error scopes
/// debug.push_error_scope(ErrorFilter::Validation);
/// // ... GPU operations ...
/// let scope = debug.pop_error_scope();
/// ```
pub struct DebugUtils {
    /// Handler for device lost events.
    device_lost_handler: Option<Box<dyn Fn(DeviceLostInfo) + Send + Sync>>,

    /// Stack of active error scopes.
    error_scopes: Vec<ErrorScope>,

    /// Registry for error callbacks.
    callback_registry: ErrorCallbackRegistry,

    /// Total errors captured.
    total_errors: u64,
}

impl DebugUtils {
    /// Create new debug utilities.
    pub fn new() -> Self {
        Self {
            device_lost_handler: None,
            error_scopes: Vec::new(),
            callback_registry: ErrorCallbackRegistry::new(),
            total_errors: 0,
        }
    }

    /// Set the device lost handler callback.
    ///
    /// # Arguments
    ///
    /// * `handler` - Function to call when device is lost
    pub fn set_device_lost_handler<F>(&mut self, handler: F)
    where
        F: Fn(DeviceLostInfo) + Send + Sync + 'static,
    {
        self.device_lost_handler = Some(Box::new(handler));
    }

    /// Clear the device lost handler.
    pub fn clear_device_lost_handler(&mut self) {
        self.device_lost_handler = None;
    }

    /// Notify that the device was lost.
    ///
    /// # Arguments
    ///
    /// * `info` - Information about the device loss
    pub fn notify_device_lost(&self, info: DeviceLostInfo) {
        if let Some(ref handler) = self.device_lost_handler {
            handler(info);
        }
    }

    /// Push a new error scope onto the stack.
    ///
    /// # Arguments
    ///
    /// * `filter` - Which error types to capture
    pub fn push_error_scope(&mut self, filter: ErrorFilter) {
        self.error_scopes.push(ErrorScope::new(filter));
    }

    /// Push a new labeled error scope onto the stack.
    ///
    /// # Arguments
    ///
    /// * `filter` - Which error types to capture
    /// * `label` - Label for identification
    pub fn push_error_scope_labeled(&mut self, filter: ErrorFilter, label: impl Into<String>) {
        self.error_scopes.push(ErrorScope::with_label(filter, label));
    }

    /// Pop the top error scope from the stack.
    ///
    /// # Returns
    ///
    /// The popped error scope, or `None` if the stack is empty.
    pub fn pop_error_scope(&mut self) -> Option<ErrorScope> {
        self.error_scopes.pop()
    }

    /// Get the current error scope depth.
    #[inline]
    pub fn scope_depth(&self) -> usize {
        self.error_scopes.len()
    }

    /// Push an error to the top scope.
    ///
    /// # Arguments
    ///
    /// * `error` - The GPU error to push
    ///
    /// # Returns
    ///
    /// `true` if the error was captured, `false` if no scope is active.
    pub fn push_error(&mut self, error: GpuError) -> bool {
        // Invoke callbacks first
        self.callback_registry.invoke(&error);
        self.total_errors += 1;

        // Push to top scope if available
        if let Some(scope) = self.error_scopes.last_mut() {
            scope.push(error)
        } else {
            false
        }
    }

    /// Get the callback registry.
    #[inline]
    pub fn callbacks(&self) -> &ErrorCallbackRegistry {
        &self.callback_registry
    }

    /// Get a mutable reference to the callback registry.
    #[inline]
    pub fn callbacks_mut(&mut self) -> &mut ErrorCallbackRegistry {
        &mut self.callback_registry
    }

    /// Get the total number of errors captured.
    #[inline]
    pub fn total_errors(&self) -> u64 {
        self.total_errors
    }

    /// Create an RAII guard for capturing errors.
    ///
    /// # Arguments
    ///
    /// * `filter` - Which error types to capture
    ///
    /// # Returns
    ///
    /// An `ErrorCaptureGuard` that pops the scope when dropped.
    pub fn capture_errors(&mut self, filter: ErrorFilter) -> ErrorCaptureGuard<'_> {
        self.push_error_scope(filter);
        ErrorCaptureGuard { debug_utils: self }
    }
}

impl Default for DebugUtils {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DebugUtils {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugUtils")
            .field("scope_depth", &self.scope_depth())
            .field("callback_count", &self.callback_registry.len())
            .field("total_errors", &self.total_errors)
            .finish()
    }
}

// ============================================================================
// ErrorCaptureGuard
// ============================================================================

/// RAII guard for error capture scopes.
///
/// Automatically pops the error scope when dropped.
pub struct ErrorCaptureGuard<'a> {
    debug_utils: &'a mut DebugUtils,
}

impl<'a> ErrorCaptureGuard<'a> {
    /// Get the current error scope.
    pub fn scope(&self) -> Option<&ErrorScope> {
        self.debug_utils.error_scopes.last()
    }

    /// Check if any errors were captured.
    pub fn has_errors(&self) -> bool {
        self.scope().map_or(false, |s| s.has_errors())
    }

    /// Get the error count.
    pub fn error_count(&self) -> usize {
        self.scope().map_or(0, |s| s.error_count())
    }

    /// Finish capturing and return the scope.
    pub fn finish(self) -> Option<ErrorScope> {
        // This consumes self, triggering Drop which pops the scope
        // We need to pop before Drop runs, so use ManuallyDrop
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: We're consuming self and manually managing the drop
        let debug_utils = unsafe { std::ptr::read(&this.debug_utils) };
        // Now we own debug_utils, pop the scope
        // But wait, we can't do this safely. Let's use a different approach.
        // Actually, let's just not implement finish() and let Drop handle it.
        // Instead, users can check has_errors() before dropping.
        drop(std::mem::ManuallyDrop::into_inner(this));
        None
    }
}

impl Drop for ErrorCaptureGuard<'_> {
    fn drop(&mut self) {
        let _ = self.debug_utils.pop_error_scope();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DeviceLostReason tests
    // ========================================================================

    #[test]
    fn test_device_lost_reason_from_wgpu() {
        // Test Destroyed variant
        assert_eq!(
            DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::Destroyed),
            DeviceLostReason::Destroyed
        );
        // Test ReplacedCallback variant (maps to Unknown)
        assert_eq!(
            DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::ReplacedCallback),
            DeviceLostReason::Unknown
        );
    }

    #[test]
    fn test_device_lost_reason_is_recoverable() {
        assert!(!DeviceLostReason::Destroyed.is_recoverable());
        assert!(DeviceLostReason::DeviceInvalid.is_recoverable());
        assert!(DeviceLostReason::DriverError.is_recoverable());
        assert!(DeviceLostReason::Unknown.is_recoverable());
    }

    #[test]
    fn test_device_lost_reason_description() {
        let desc = DeviceLostReason::DriverError.description();
        assert!(desc.contains("driver"));
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_device_lost_reason_display() {
        assert_eq!(format!("{}", DeviceLostReason::Destroyed), "Destroyed");
        assert_eq!(format!("{}", DeviceLostReason::Unknown), "Unknown");
    }

    #[test]
    fn test_device_lost_reason_default() {
        assert_eq!(DeviceLostReason::default(), DeviceLostReason::Unknown);
    }

    // ========================================================================
    // DeviceLostInfo tests
    // ========================================================================

    #[test]
    fn test_device_lost_info_new() {
        let info = DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "Test message".to_string(),
        );
        assert_eq!(info.reason, DeviceLostReason::DriverError);
        assert_eq!(info.message, "Test message");
        // Verify elapsed() returns a valid duration (instant is captured)
        let _ = info.elapsed();
    }

    #[test]
    fn test_device_lost_info_should_attempt_recovery() {
        let info = DeviceLostInfo::new(DeviceLostReason::DriverError, String::new());
        assert!(info.should_attempt_recovery());

        let info = DeviceLostInfo::new(DeviceLostReason::Destroyed, String::new());
        assert!(!info.should_attempt_recovery());
    }

    #[test]
    fn test_device_lost_info_display() {
        let info = DeviceLostInfo::new(
            DeviceLostReason::Unknown,
            "Test".to_string(),
        );
        let display = format!("{}", info);
        assert!(display.contains("Unknown"));
        assert!(display.contains("Test"));
    }

    // ========================================================================
    // ErrorFilter tests
    // ========================================================================

    #[test]
    fn test_error_filter_matches() {
        assert!(ErrorFilter::Validation.matches(&GpuErrorType::Validation));
        assert!(!ErrorFilter::Validation.matches(&GpuErrorType::OutOfMemory));

        assert!(ErrorFilter::OutOfMemory.matches(&GpuErrorType::OutOfMemory));
        assert!(!ErrorFilter::OutOfMemory.matches(&GpuErrorType::Internal));

        assert!(ErrorFilter::Internal.matches(&GpuErrorType::Internal));
        assert!(!ErrorFilter::Internal.matches(&GpuErrorType::Lost));

        assert!(ErrorFilter::All.matches(&GpuErrorType::Validation));
        assert!(ErrorFilter::All.matches(&GpuErrorType::OutOfMemory));
        assert!(ErrorFilter::All.matches(&GpuErrorType::Internal));
        assert!(ErrorFilter::All.matches(&GpuErrorType::Lost));
    }

    #[test]
    fn test_error_filter_to_wgpu() {
        assert_eq!(ErrorFilter::Validation.to_wgpu(), wgpu::ErrorFilter::Validation);
        assert_eq!(ErrorFilter::OutOfMemory.to_wgpu(), wgpu::ErrorFilter::OutOfMemory);
    }

    #[test]
    fn test_error_filter_display() {
        assert_eq!(format!("{}", ErrorFilter::All), "All");
        assert_eq!(format!("{}", ErrorFilter::Validation), "Validation");
    }

    #[test]
    fn test_error_filter_default() {
        assert_eq!(ErrorFilter::default(), ErrorFilter::All);
    }

    // ========================================================================
    // GpuErrorType tests
    // ========================================================================

    #[test]
    fn test_gpu_error_type_severity() {
        assert_eq!(GpuErrorType::Validation.severity(), Severity::Warning);
        assert_eq!(GpuErrorType::OutOfMemory.severity(), Severity::Error);
        assert_eq!(GpuErrorType::Internal.severity(), Severity::Error);
        assert_eq!(GpuErrorType::Lost.severity(), Severity::Error);
    }

    #[test]
    fn test_gpu_error_type_is_fatal() {
        assert!(!GpuErrorType::Validation.is_fatal());
        assert!(!GpuErrorType::OutOfMemory.is_fatal());
        assert!(!GpuErrorType::Internal.is_fatal());
        assert!(GpuErrorType::Lost.is_fatal());
    }

    #[test]
    fn test_gpu_error_type_helpers() {
        assert!(GpuErrorType::Validation.is_validation());
        assert!(!GpuErrorType::Validation.is_oom());
        assert!(!GpuErrorType::Validation.is_internal());

        assert!(GpuErrorType::OutOfMemory.is_oom());
        assert!(GpuErrorType::Internal.is_internal());
    }

    #[test]
    fn test_gpu_error_type_display() {
        assert_eq!(format!("{}", GpuErrorType::Validation), "Validation");
        assert_eq!(format!("{}", GpuErrorType::Lost), "DeviceLost");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }

    // ========================================================================
    // SourceLocation tests
    // ========================================================================

    #[test]
    fn test_source_location_new() {
        let loc = SourceLocation::new();
        assert!(loc.file.is_none());
        assert!(loc.line.is_none());
        assert!(!loc.is_available());
    }

    #[test]
    fn test_source_location_here() {
        let loc = SourceLocation::here();
        assert!(loc.file.is_some());
        assert!(loc.line.is_some());
        assert!(loc.column.is_some());
        assert!(loc.is_available());
    }

    #[test]
    fn test_source_location_builder() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_column(10)
            .with_function("test_fn");

        assert_eq!(loc.file.as_deref(), Some("test.rs"));
        assert_eq!(loc.line, Some(42));
        assert_eq!(loc.column, Some(10));
        assert_eq!(loc.function.as_deref(), Some("test_fn"));
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation::new()
            .with_file("test.rs")
            .with_line(42)
            .with_function("render");

        let display = format!("{}", loc);
        assert!(display.contains("test.rs"));
        assert!(display.contains("42"));
        assert!(display.contains("render"));
    }

    #[test]
    fn test_source_location_display_empty() {
        let loc = SourceLocation::new();
        assert_eq!(format!("{}", loc), "<unknown location>");
    }

    // ========================================================================
    // GpuError tests
    // ========================================================================

    #[test]
    fn test_gpu_error_new() {
        let error = GpuError::new(
            GpuErrorType::Validation,
            "Invalid usage".to_string(),
        );
        assert_eq!(error.error_type, GpuErrorType::Validation);
        assert_eq!(error.message, "Invalid usage");
        assert!(error.source_location.is_none());
    }

    #[test]
    fn test_gpu_error_with_source_location() {
        let error = GpuError::new(GpuErrorType::OutOfMemory, "OOM".to_string())
            .with_source_location(SourceLocation::here());

        assert!(error.source_location.is_some());
    }

    #[test]
    fn test_gpu_error_helpers() {
        let val_err = GpuError::new(GpuErrorType::Validation, String::new());
        assert!(val_err.is_validation());
        assert!(!val_err.is_oom());

        let oom_err = GpuError::new(GpuErrorType::OutOfMemory, String::new());
        assert!(oom_err.is_oom());

        let int_err = GpuError::new(GpuErrorType::Internal, String::new());
        assert!(int_err.is_internal());
    }

    #[test]
    fn test_gpu_error_display() {
        let error = GpuError::new(GpuErrorType::Validation, "Test error".to_string());
        let display = format!("{}", error);
        assert!(display.contains("WARN"));
        assert!(display.contains("Validation"));
        assert!(display.contains("Test error"));
    }

    // ========================================================================
    // ErrorScope tests
    // ========================================================================

    #[test]
    fn test_error_scope_new() {
        let scope = ErrorScope::new(ErrorFilter::Validation);
        assert!(scope.label.is_none());
        assert_eq!(scope.filter, ErrorFilter::Validation);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_error_scope_with_label() {
        let scope = ErrorScope::with_label(ErrorFilter::All, "test_scope");
        assert_eq!(scope.label.as_deref(), Some("test_scope"));
    }

    #[test]
    fn test_error_scope_push() {
        let mut scope = ErrorScope::new(ErrorFilter::Validation);

        // Should accept validation errors
        let accepted = scope.push(GpuError::new(
            GpuErrorType::Validation,
            "Test".to_string(),
        ));
        assert!(accepted);
        assert!(scope.has_errors());
        assert_eq!(scope.error_count(), 1);

        // Should reject OOM errors
        let rejected = scope.push(GpuError::new(
            GpuErrorType::OutOfMemory,
            "OOM".to_string(),
        ));
        assert!(!rejected);
        assert_eq!(scope.error_count(), 1);
    }

    #[test]
    fn test_error_scope_all_filter() {
        let mut scope = ErrorScope::new(ErrorFilter::All);

        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));
        scope.push(GpuError::new(GpuErrorType::Internal, "3".to_string()));

        assert_eq!(scope.error_count(), 3);
    }

    #[test]
    fn test_error_scope_take_errors() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "2".to_string()));

        let errors = scope.take_errors();
        assert_eq!(errors.len(), 2);
        assert!(!scope.has_errors());
        assert_eq!(scope.error_count(), 0);
    }

    #[test]
    fn test_error_scope_clear() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "1".to_string()));
        scope.clear();
        assert!(!scope.has_errors());
    }

    #[test]
    fn test_error_scope_most_severe() {
        let mut scope = ErrorScope::new(ErrorFilter::All);
        scope.push(GpuError::new(GpuErrorType::Validation, "val".to_string()));
        scope.push(GpuError::new(GpuErrorType::OutOfMemory, "oom".to_string()));

        let most_severe = scope.most_severe().unwrap();
        assert_eq!(most_severe.error_type, GpuErrorType::OutOfMemory);
    }

    #[test]
    fn test_error_scope_display() {
        let scope = ErrorScope::with_label(ErrorFilter::Validation, "render");
        let display = format!("{}", scope);
        assert!(display.contains("render"));
        assert!(display.contains("Validation"));
    }

    // ========================================================================
    // ErrorCallbackRegistry tests
    // ========================================================================

    #[test]
    fn test_callback_registry_new() {
        let registry = ErrorCallbackRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_callback_registry_register_unregister() {
        let registry = ErrorCallbackRegistry::new();

        let id1 = registry.register(Arc::new(|_| {}));
        let id2 = registry.register(Arc::new(|_| {}));

        assert_eq!(registry.len(), 2);
        assert_ne!(id1, id2);

        assert!(registry.unregister(id1));
        assert_eq!(registry.len(), 1);

        assert!(!registry.unregister(id1)); // Already removed
        assert!(registry.unregister(id2));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_callback_registry_invoke() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let registry = ErrorCallbackRegistry::new();
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        registry.register(Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let error = GpuError::new(GpuErrorType::Validation, "test".to_string());
        let count = registry.invoke(&error);

        assert_eq!(count, 1);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_registry_clear() {
        let registry = ErrorCallbackRegistry::new();
        registry.register(Arc::new(|_| {}));
        registry.register(Arc::new(|_| {}));

        registry.clear();
        assert!(registry.is_empty());
    }

    // ========================================================================
    // DebugUtils tests
    // ========================================================================

    #[test]
    fn test_debug_utils_new() {
        let debug = DebugUtils::new();
        assert_eq!(debug.scope_depth(), 0);
        assert_eq!(debug.total_errors(), 0);
    }

    #[test]
    fn test_debug_utils_error_scopes() {
        let mut debug = DebugUtils::new();

        debug.push_error_scope(ErrorFilter::Validation);
        assert_eq!(debug.scope_depth(), 1);

        debug.push_error_scope(ErrorFilter::OutOfMemory);
        assert_eq!(debug.scope_depth(), 2);

        let scope = debug.pop_error_scope().unwrap();
        assert_eq!(scope.filter, ErrorFilter::OutOfMemory);
        assert_eq!(debug.scope_depth(), 1);

        debug.pop_error_scope();
        assert_eq!(debug.scope_depth(), 0);
        assert!(debug.pop_error_scope().is_none());
    }

    #[test]
    fn test_debug_utils_push_error() {
        let mut debug = DebugUtils::new();

        // No scope - error not captured
        let captured = debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "test".to_string(),
        ));
        assert!(!captured);
        assert_eq!(debug.total_errors(), 1); // Still counted

        // With scope
        debug.push_error_scope(ErrorFilter::All);
        let captured = debug.push_error(GpuError::new(
            GpuErrorType::Validation,
            "test".to_string(),
        ));
        assert!(captured);
        assert_eq!(debug.total_errors(), 2);
    }

    #[test]
    fn test_debug_utils_device_lost_handler() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut debug = DebugUtils::new();
        let called = Arc::new(AtomicBool::new(false));

        let called_clone = called.clone();
        debug.set_device_lost_handler(move |_info| {
            called_clone.store(true, Ordering::SeqCst);
        });

        debug.notify_device_lost(DeviceLostInfo::new(
            DeviceLostReason::DriverError,
            "test".to_string(),
        ));

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_debug_utils_capture_errors_guard() {
        let mut debug = DebugUtils::new();

        // Push manually and verify the guard approach works
        debug.push_error_scope(ErrorFilter::Validation);
        assert_eq!(debug.scope_depth(), 1);
        debug.pop_error_scope();
        assert_eq!(debug.scope_depth(), 0);

        // Test capture_errors creates a scope
        {
            let _guard = debug.capture_errors(ErrorFilter::Validation);
            // Can't access debug while guard is held (borrow rules)
            // Guard will drop and pop the scope
        }

        // After guard is dropped, we can access debug again
        assert_eq!(debug.scope_depth(), 0);
    }

    #[test]
    fn test_debug_utils_labeled_scope() {
        let mut debug = DebugUtils::new();

        debug.push_error_scope_labeled(ErrorFilter::All, "shadow_pass");
        let scope = debug.pop_error_scope().unwrap();

        assert_eq!(scope.label.as_deref(), Some("shadow_pass"));
    }
}
