//! Error scope wrapper for fine-grained GPU error handling.
//!
//! This module provides [`ErrorScope`], an RAII wrapper around wgpu's error scope
//! mechanism. Error scopes allow isolating GPU operations and capturing errors
//! that occur during those operations without terminating the entire device.
//!
//! # Overview
//!
//! wgpu provides two types of error handling:
//!
//! 1. **Uncaptured error handler** - A global callback invoked for any error not
//!    captured by an error scope. Useful for logging but doesn't allow recovery.
//!
//! 2. **Error scopes** - Push/pop mechanism that captures errors occurring within
//!    the scope. Allows querying and handling errors programmatically.
//!
//! This module focuses on error scopes, providing a safe RAII interface.
//!
//! # Error Filters
//!
//! Error scopes filter which types of errors they capture:
//!
//! - [`ErrorFilter::Validation`] - API usage errors (invalid parameters, missing
//!   resources, invalid state). These indicate bugs in the calling code.
//!
//! - [`ErrorFilter::OutOfMemory`] - Resource allocation failures. These may be
//!   recoverable by freeing resources and retrying.
//!
//! # RAII Pattern
//!
//! [`ErrorScope`] uses RAII to ensure scopes are always properly closed:
//!
//! - `push_error_scope()` is called on construction
//! - `pop_error_scope()` is called on drop (unless explicitly popped)
//!
//! This prevents scope leaks and ensures errors are captured even if the code
//! panics or returns early.
//!
//! # Nested Scopes
//!
//! Error scopes can be nested like a stack. Inner scopes capture errors first,
//! and uncaptured errors propagate to outer scopes:
//!
//! ```text
//! push(Validation)           <- outer scope
//!   push(OutOfMemory)        <- inner scope (captures OOM errors)
//!     [operations]           <- errors captured by inner scope
//!   pop() -> Option<Error>   <- check inner scope
//! pop() -> Option<Error>     <- check outer scope (validation errors only)
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{TrinityDevice, ErrorScope, ErrorFilter};
//!
//! # async fn example(device: &TrinityDevice) -> Result<(), Box<dyn std::error::Error>> {
//! // Create a validation error scope
//! let scope = ErrorScope::new(device.device(), ErrorFilter::Validation);
//!
//! // Perform GPU operations...
//! let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
//!     label: Some("Test Buffer"),
//!     size: 1024,
//!     usage: wgpu::BufferUsages::VERTEX,
//!     mapped_at_creation: false,
//! });
//!
//! // Explicitly pop to get any errors
//! if let Some(error) = scope.pop().await {
//!     eprintln!("GPU validation error: {:?}", error);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Alternative: Auto-Pop on Drop
//!
//! If you don't need to check the error, the scope will automatically pop on drop:
//!
//! ```no_run
//! use renderer_backend::device::{TrinityDevice, ErrorScope, ErrorFilter};
//!
//! fn create_resources(device: &wgpu::Device) {
//!     let _scope = ErrorScope::new(device, ErrorFilter::Validation);
//!
//!     // Create resources...
//!     // If scope is not explicitly popped, it pops on drop.
//!     // Any error is logged but not returned.
//! }
//! ```

use log::{debug, error, warn};
use std::cell::Cell;
use std::future::Future;

// ============================================================================
// ErrorFilter
// ============================================================================

/// Filter for which error types an error scope captures.
///
/// When pushing an error scope, you specify which type of errors it should
/// capture. Errors of other types pass through to outer scopes or the
/// uncaptured error handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorFilter {
    /// Capture validation errors (API misuse, invalid parameters).
    ///
    /// Validation errors indicate bugs in the calling code:
    /// - Invalid buffer/texture usage flags
    /// - Missing bind group entries
    /// - Invalid shader module
    /// - Resource used after destruction
    ///
    /// These errors should be fixed in the code, not handled at runtime.
    Validation,

    /// Capture out-of-memory errors (allocation failures).
    ///
    /// OOM errors occur when:
    /// - GPU memory is exhausted
    /// - Buffer/texture size exceeds limits
    /// - Too many resources allocated
    ///
    /// These errors may be recoverable by freeing resources and retrying.
    OutOfMemory,
}

impl ErrorFilter {
    /// Convert to wgpu's ErrorFilter type.
    #[inline]
    pub fn to_wgpu(self) -> wgpu::ErrorFilter {
        match self {
            ErrorFilter::Validation => wgpu::ErrorFilter::Validation,
            ErrorFilter::OutOfMemory => wgpu::ErrorFilter::OutOfMemory,
        }
    }

    /// Convert from wgpu's ErrorFilter type.
    #[inline]
    pub fn from_wgpu(filter: wgpu::ErrorFilter) -> Self {
        match filter {
            wgpu::ErrorFilter::Validation => ErrorFilter::Validation,
            wgpu::ErrorFilter::OutOfMemory => ErrorFilter::OutOfMemory,
            // wgpu may add more variants in the future; default to Validation
            _ => {
                warn!("Unknown wgpu::ErrorFilter variant, treating as Validation");
                ErrorFilter::Validation
            }
        }
    }

    /// Human-readable description of the filter.
    pub fn description(&self) -> &'static str {
        match self {
            ErrorFilter::Validation => "validation errors (API misuse)",
            ErrorFilter::OutOfMemory => "out-of-memory errors (allocation failures)",
        }
    }
}

impl std::fmt::Display for ErrorFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorFilter::Validation => write!(f, "Validation"),
            ErrorFilter::OutOfMemory => write!(f, "OutOfMemory"),
        }
    }
}

// ============================================================================
// ErrorScope
// ============================================================================

/// RAII wrapper for wgpu error scopes.
///
/// `ErrorScope` pushes an error scope on creation and pops it on drop (unless
/// explicitly popped). This ensures error scopes are always properly balanced.
///
/// # Lifetime
///
/// The `'a` lifetime ensures the `ErrorScope` cannot outlive the device it
/// references. This prevents use-after-free of the device.
///
/// # Thread Safety
///
/// `ErrorScope` is `!Send` and `!Sync` because wgpu error scopes are thread-local.
/// An error scope pushed on one thread cannot be popped from another thread.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{ErrorScope, ErrorFilter};
///
/// # async fn example(device: &wgpu::Device) {
/// // RAII scope - automatically pops on drop if not explicitly popped
/// {
///     let scope = ErrorScope::new(device, ErrorFilter::Validation);
///     // ... GPU operations ...
/// } // scope pops here
///
/// // Explicit pop - get the error (if any)
/// let scope = ErrorScope::new(device, ErrorFilter::OutOfMemory);
/// // ... allocate resources ...
/// match scope.pop().await {
///     Some(error) => eprintln!("Allocation failed: {:?}", error),
///     None => println!("Allocation succeeded"),
/// }
/// # }
/// ```
pub struct ErrorScope<'a> {
    /// Reference to the device (needed for pop_error_scope).
    device: &'a wgpu::Device,
    /// The filter this scope was created with.
    filter: ErrorFilter,
    /// Flag to track if scope was already popped (prevents double-pop).
    popped: Cell<bool>,
    /// Optional label for logging/debugging.
    label: Option<&'a str>,
}

impl<'a> ErrorScope<'a> {
    /// Create a new error scope with the specified filter.
    ///
    /// This immediately pushes an error scope onto the device's scope stack.
    /// The scope will capture errors matching the filter until it is popped.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to push the scope on
    /// * `filter` - Which error types this scope should capture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{ErrorScope, ErrorFilter};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let scope = ErrorScope::new(device, ErrorFilter::Validation);
    /// // Operations here are protected by the error scope
    /// # }
    /// ```
    pub fn new(device: &'a wgpu::Device, filter: ErrorFilter) -> Self {
        debug!("ErrorScope: Pushing {} scope", filter);
        device.push_error_scope(filter.to_wgpu());

        Self {
            device,
            filter,
            popped: Cell::new(false),
            label: None,
        }
    }

    /// Create a new error scope with a label for debugging.
    ///
    /// The label is included in log messages to help identify which scope
    /// captured an error.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to push the scope on
    /// * `filter` - Which error types this scope should capture
    /// * `label` - Debug label for this scope
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{ErrorScope, ErrorFilter};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let scope = ErrorScope::with_label(device, ErrorFilter::OutOfMemory, "texture_load");
    /// // If an error occurs, logs will include "texture_load" for identification
    /// # }
    /// ```
    pub fn with_label(device: &'a wgpu::Device, filter: ErrorFilter, label: &'a str) -> Self {
        debug!("ErrorScope: Pushing {} scope (label: {})", filter, label);
        device.push_error_scope(filter.to_wgpu());

        Self {
            device,
            filter,
            popped: Cell::new(false),
            label: Some(label),
        }
    }

    /// Get the filter this scope was created with.
    #[inline]
    pub fn filter(&self) -> ErrorFilter {
        self.filter
    }

    /// Get the label for this scope (if any).
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label
    }

    /// Check if this scope has already been popped.
    #[inline]
    pub fn is_popped(&self) -> bool {
        self.popped.get()
    }

    /// Pop the error scope and return any captured error.
    ///
    /// This consumes the `ErrorScope` and returns a future that resolves to
    /// the captured error (if any). Once popped, the scope cannot be used again.
    ///
    /// # Returns
    ///
    /// A future resolving to `Option<wgpu::Error>`:
    /// - `None` if no error was captured
    /// - `Some(error)` if an error matching the filter occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{ErrorScope, ErrorFilter};
    ///
    /// # async fn example(device: &wgpu::Device) {
    /// let scope = ErrorScope::new(device, ErrorFilter::Validation);
    /// // ... operations ...
    ///
    /// match scope.pop().await {
    ///     Some(error) => eprintln!("Validation error: {:?}", error),
    ///     None => println!("No errors"),
    /// }
    /// # }
    /// ```
    pub fn pop(self) -> impl Future<Output = Option<wgpu::Error>> + 'a {
        // Mark as popped to prevent double-pop in Drop
        self.popped.set(true);

        let label = self.label;
        let filter = self.filter;

        debug!(
            "ErrorScope: Popping {} scope{}",
            filter,
            label.map_or(String::new(), |l| format!(" (label: {})", l))
        );

        async move {
            let error = self.device.pop_error_scope().await;

            if let Some(ref err) = error {
                if let Some(label) = label {
                    error!(
                        "ErrorScope [{}]: Captured {} error: {:?}",
                        label, filter, err
                    );
                } else {
                    error!("ErrorScope: Captured {} error: {:?}", filter, err);
                }
            }

            error
        }
    }

    /// Pop the error scope synchronously using pollster.
    ///
    /// This is a convenience method for non-async contexts. It blocks the
    /// current thread until the error scope is resolved.
    ///
    /// # Warning
    ///
    /// This should not be called from within an async runtime as it may
    /// cause deadlocks. Use [`pop`](Self::pop) in async contexts.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{ErrorScope, ErrorFilter};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let scope = ErrorScope::new(device, ErrorFilter::Validation);
    /// // ... operations ...
    ///
    /// if let Some(error) = scope.pop_blocking() {
    ///     eprintln!("Validation error: {:?}", error);
    /// }
    /// # }
    /// ```
    pub fn pop_blocking(self) -> Option<wgpu::Error> {
        pollster::block_on(self.pop())
    }
}

impl Drop for ErrorScope<'_> {
    fn drop(&mut self) {
        // Only pop if not already explicitly popped
        if !self.popped.get() {
            debug!(
                "ErrorScope: Auto-popping {} scope on drop{}",
                self.filter,
                self.label
                    .map_or(String::new(), |l| format!(" (label: {})", l))
            );

            // We can't await in drop, so we use pollster to block
            // This is safe because drop is called from sync context
            let error = pollster::block_on(self.device.pop_error_scope());

            if let Some(err) = error {
                if let Some(label) = self.label {
                    error!(
                        "ErrorScope [{}]: Dropped with captured {} error: {:?}",
                        label, self.filter, err
                    );
                } else {
                    error!(
                        "ErrorScope: Dropped with captured {} error: {:?}",
                        self.filter, err
                    );
                }
            }

            self.popped.set(true);
        }
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Execute an operation within a validation error scope.
///
/// This is a convenience function that wraps an async operation in a
/// validation error scope and returns both the result and any captured error.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `label` - Optional label for debugging
/// * `operation` - The async operation to execute
///
/// # Returns
///
/// A tuple of `(operation_result, Option<wgpu::Error>)`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::with_validation_scope;
///
/// # async fn example(device: &wgpu::Device) {
/// let (buffer, error) = with_validation_scope(device, Some("create_buffer"), async {
///     device.create_buffer(&wgpu::BufferDescriptor {
///         label: Some("My Buffer"),
///         size: 1024,
///         usage: wgpu::BufferUsages::VERTEX,
///         mapped_at_creation: false,
///     })
/// }).await;
///
/// if let Some(err) = error {
///     eprintln!("Validation error during buffer creation: {:?}", err);
/// }
/// # }
/// ```
pub async fn with_validation_scope<'a, T, F>(
    device: &'a wgpu::Device,
    label: Option<&'a str>,
    operation: F,
) -> (T, Option<wgpu::Error>)
where
    F: Future<Output = T>,
{
    let scope = match label {
        Some(l) => ErrorScope::with_label(device, ErrorFilter::Validation, l),
        None => ErrorScope::new(device, ErrorFilter::Validation),
    };

    let result = operation.await;
    let error = scope.pop().await;

    (result, error)
}

/// Execute an operation within an out-of-memory error scope.
///
/// This is a convenience function that wraps an async operation in an
/// OOM error scope and returns both the result and any captured error.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `label` - Optional label for debugging
/// * `operation` - The async operation to execute
///
/// # Returns
///
/// A tuple of `(operation_result, Option<wgpu::Error>)`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::with_oom_scope;
///
/// # async fn example(device: &wgpu::Device) {
/// let (texture, error) = with_oom_scope(device, Some("texture_alloc"), async {
///     device.create_texture(&wgpu::TextureDescriptor {
///         label: Some("Large Texture"),
///         size: wgpu::Extent3d { width: 8192, height: 8192, depth_or_array_layers: 1 },
///         mip_level_count: 1,
///         sample_count: 1,
///         dimension: wgpu::TextureDimension::D2,
///         format: wgpu::TextureFormat::Rgba8UnormSrgb,
///         usage: wgpu::TextureUsages::TEXTURE_BINDING,
///         view_formats: &[],
///     })
/// }).await;
///
/// match error {
///     Some(err) => eprintln!("Failed to allocate texture: {:?}", err),
///     None => println!("Texture allocated successfully"),
/// }
/// # }
/// ```
pub async fn with_oom_scope<'a, T, F>(
    device: &'a wgpu::Device,
    label: Option<&'a str>,
    operation: F,
) -> (T, Option<wgpu::Error>)
where
    F: Future<Output = T>,
{
    let scope = match label {
        Some(l) => ErrorScope::with_label(device, ErrorFilter::OutOfMemory, l),
        None => ErrorScope::new(device, ErrorFilter::OutOfMemory),
    };

    let result = operation.await;
    let error = scope.pop().await;

    (result, error)
}

// ============================================================================
// ScopedErrorCapture - multi-scope helper
// ============================================================================

/// Capture errors from multiple filters in a single operation.
///
/// This struct pushes multiple error scopes (in order) and pops them all
/// when finished. Useful when you want to capture both validation and OOM
/// errors from the same operation.
///
/// # Scope Order
///
/// Scopes are pushed in the order specified, and errors are captured by the
/// innermost (most recently pushed) matching scope. For example, with filters
/// `[Validation, OutOfMemory]`:
///
/// 1. Validation scope pushed (outer)
/// 2. OutOfMemory scope pushed (inner)
/// 3. Operations execute
/// 4. OutOfMemory scope popped first (captures OOM errors)
/// 5. Validation scope popped last (captures validation errors that weren't OOM)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{ScopedErrorCapture, ErrorFilter};
///
/// # async fn example(device: &wgpu::Device) {
/// let mut capture = ScopedErrorCapture::new(
///     device,
///     &[ErrorFilter::Validation, ErrorFilter::OutOfMemory],
/// );
///
/// // Perform operations...
///
/// let errors = capture.pop_all().await;
/// for (filter, error) in errors {
///     if let Some(err) = error {
///         eprintln!("{} error: {:?}", filter, err);
///     }
/// }
/// # }
/// ```
pub struct ScopedErrorCapture<'a> {
    device: &'a wgpu::Device,
    /// Filters in order they were pushed (for popping in reverse).
    filters: Vec<ErrorFilter>,
    /// Whether already popped.
    popped: bool,
}

impl<'a> ScopedErrorCapture<'a> {
    /// Create a new multi-scope error capture.
    ///
    /// Pushes error scopes for each filter in the given order.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `filters` - Filters to push (in order)
    pub fn new(device: &'a wgpu::Device, filters: &[ErrorFilter]) -> Self {
        for filter in filters {
            debug!("ScopedErrorCapture: Pushing {} scope", filter);
            device.push_error_scope(filter.to_wgpu());
        }

        Self {
            device,
            filters: filters.to_vec(),
            popped: false,
        }
    }

    /// Pop all error scopes and return captured errors.
    ///
    /// Scopes are popped in reverse order (LIFO). Returns a vector of
    /// `(filter, Option<Error>)` pairs in the order scopes were popped
    /// (innermost first).
    ///
    /// # Returns
    ///
    /// Vector of `(ErrorFilter, Option<wgpu::Error>)` pairs.
    pub async fn pop_all(&mut self) -> Vec<(ErrorFilter, Option<wgpu::Error>)> {
        if self.popped {
            warn!("ScopedErrorCapture: Already popped, returning empty");
            return Vec::new();
        }

        self.popped = true;
        let mut results = Vec::with_capacity(self.filters.len());

        // Pop in reverse order (LIFO)
        for filter in self.filters.iter().rev() {
            debug!("ScopedErrorCapture: Popping {} scope", filter);
            let error = self.device.pop_error_scope().await;

            if let Some(ref err) = error {
                error!("ScopedErrorCapture: Captured {} error: {:?}", filter, err);
            }

            results.push((*filter, error));
        }

        results
    }

    /// Pop all error scopes synchronously.
    ///
    /// Convenience method for non-async contexts. See [`pop_all`](Self::pop_all).
    pub fn pop_all_blocking(&mut self) -> Vec<(ErrorFilter, Option<wgpu::Error>)> {
        pollster::block_on(self.pop_all())
    }

    /// Check if any errors were captured (pops all scopes).
    ///
    /// This is a convenience method that pops all scopes and returns true
    /// if any error was captured.
    pub async fn has_errors(&mut self) -> bool {
        self.pop_all().await.iter().any(|(_, e)| e.is_some())
    }
}

impl Drop for ScopedErrorCapture<'_> {
    fn drop(&mut self) {
        if !self.popped {
            debug!("ScopedErrorCapture: Auto-popping {} scopes on drop", self.filters.len());

            // Pop all remaining scopes
            for filter in self.filters.iter().rev() {
                let error = pollster::block_on(self.device.pop_error_scope());
                if let Some(err) = error {
                    error!(
                        "ScopedErrorCapture: Dropped with captured {} error: {:?}",
                        filter, err
                    );
                }
            }

            self.popped = true;
        }
    }
}
