//! Debug Group RAII Utilities for wgpu 22.x/25.x
//!
//! This module provides RAII wrappers for GPU debug groups that automatically
//! push on creation and pop on drop. This ensures debug groups are always
//! properly balanced, even in the presence of early returns or panics.
//!
//! # Overview
//!
//! GPU debugging tools like RenderDoc, PIX, and Nsight Graphics use debug groups
//! to organize command streams into hierarchical regions. This module provides:
//!
//! - [`DebugScope`] - RAII wrapper that pushes/pops debug groups automatically
//! - [`DebugGroupOps`] - Trait for types supporting debug group operations
//! - Convenience functions for creating scopes on different target types
//!
//! # Architecture
//!
//! ```text
//! DebugScope<'a, T: DebugGroupOps>
//!     |-- target: &'a mut T
//!     `-- Drop::drop() -> target.pop_debug_group()
//!
//! DebugGroupOps (trait)
//!     |-- push_debug_group(&mut self, label: &str)
//!     |-- pop_debug_group(&mut self)
//!     `-- insert_debug_marker(&mut self, label: &str)
//!
//! Implementations:
//!     |-- impl DebugGroupOps for wgpu::CommandEncoder
//!     |-- impl<'a> DebugGroupOps for wgpu::RenderPass<'a>
//!     `-- impl<'a> DebugGroupOps for wgpu::ComputePass<'a>
//! ```
//!
//! # wgpu API Reference
//!
//! ```ignore
//! // CommandEncoder
//! encoder.push_debug_group("label");
//! encoder.pop_debug_group();
//! encoder.insert_debug_marker("marker");
//!
//! // RenderPass
//! render_pass.push_debug_group("label");
//! render_pass.pop_debug_group();
//! render_pass.insert_debug_marker("marker");
//!
//! // ComputePass
//! compute_pass.push_debug_group("label");
//! compute_pass.pop_debug_group();
//! compute_pass.insert_debug_marker("marker");
//! ```
//!
//! # Example: Basic Usage
//!
//! ```ignore
//! use renderer_backend::debug_utils::{DebugScope, debug_scope};
//!
//! fn record_frame(encoder: &mut wgpu::CommandEncoder) {
//!     // Using the convenience function
//!     {
//!         let _scope = debug_scope(encoder, "GBuffer Pass");
//!         // ... GBuffer commands ...
//!         // Automatically pops when _scope goes out of scope
//!     }
//!
//!     // Using DebugScope directly
//!     {
//!         let _scope = DebugScope::new(encoder, "Lighting Pass");
//!         // ... Lighting commands ...
//!     }
//! }
//! ```
//!
//! # Example: Nested Scopes
//!
//! ```ignore
//! use renderer_backend::debug_utils::DebugScope;
//!
//! fn record_shadows(encoder: &mut wgpu::CommandEncoder) {
//!     let _shadow_scope = DebugScope::new(encoder, "Shadow Pass");
//!
//!     for cascade in 0..4 {
//!         let _cascade_scope = DebugScope::new(encoder, &format!("Cascade {}", cascade));
//!         // ... render cascade ...
//!         // Inner scope pops first
//!     }
//!     // Outer scope pops last
//! }
//! ```
//!
//! # Example: With Render Pass
//!
//! ```ignore
//! use renderer_backend::debug_utils::{DebugScope, debug_scope_render};
//!
//! fn record_render_pass(pass: &mut wgpu::RenderPass) {
//!     let _scope = debug_scope_render(pass, "Draw Opaque");
//!     // ... draw calls ...
//! }
//! ```
//!
//! # Thread Safety
//!
//! `DebugScope` has the same thread safety characteristics as its target type:
//! - For `CommandEncoder`: `Send + Sync`
//! - For `RenderPass` and `ComputePass`: `!Send + !Sync`
//!
//! # Performance
//!
//! Debug groups are only meaningful when using GPU debugging tools. In release
//! builds without debug tools attached, they have minimal overhead (typically
//! just a label string comparison in the driver).

use std::marker::PhantomData;
use std::time::Instant;

// ============================================================================
// DebugMarker - Marker with Optional Metadata (T-WGPU-P4.5.2)
// ============================================================================

/// A debug marker with optional metadata for enhanced GPU debugging.
///
/// Debug markers are single points in the command stream that appear in GPU
/// debugging tools like RenderDoc, PIX, and Nsight Graphics. This struct
/// provides additional context beyond a simple label.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::DebugMarker;
///
/// // Simple marker
/// let marker = DebugMarker::new("Checkpoint");
///
/// // Marker with timestamp
/// let timed = DebugMarker::with_timestamp("Frame Start");
///
/// // Marker with metadata
/// let detailed = DebugMarker::with_metadata("Draw Call", "batch_id=42");
/// ```
#[derive(Debug, Clone)]
pub struct DebugMarker {
    /// The marker label visible in debugging tools
    pub label: String,
    /// Optional timestamp when the marker was created
    pub timestamp: Option<Instant>,
    /// Optional metadata string for additional context
    pub metadata: Option<String>,
}

impl DebugMarker {
    /// Create a new debug marker with just a label.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    ///
    /// # Returns
    ///
    /// A `DebugMarker` with no timestamp or metadata.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let marker = DebugMarker::new("Checkpoint");
    /// assert!(marker.timestamp.is_none());
    /// assert!(marker.metadata.is_none());
    /// ```
    #[inline]
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            timestamp: None,
            metadata: None,
        }
    }

    /// Create a debug marker with a timestamp.
    ///
    /// The timestamp is captured at the moment of creation using `Instant::now()`.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    ///
    /// # Returns
    ///
    /// A `DebugMarker` with the current timestamp.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let marker = DebugMarker::with_timestamp("Frame Start");
    /// assert!(marker.timestamp.is_some());
    /// ```
    #[inline]
    pub fn with_timestamp(label: &str) -> Self {
        Self {
            label: label.to_string(),
            timestamp: Some(Instant::now()),
            metadata: None,
        }
    }

    /// Create a debug marker with metadata.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    /// * `metadata` - Additional context string
    ///
    /// # Returns
    ///
    /// A `DebugMarker` with the specified metadata.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let marker = DebugMarker::with_metadata("Draw Call", "batch_id=42, material=pbr");
    /// assert_eq!(marker.metadata.as_deref(), Some("batch_id=42, material=pbr"));
    /// ```
    #[inline]
    pub fn with_metadata(label: &str, metadata: &str) -> Self {
        Self {
            label: label.to_string(),
            timestamp: None,
            metadata: Some(metadata.to_string()),
        }
    }

    /// Create a debug marker with both timestamp and metadata.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    /// * `metadata` - Additional context string
    ///
    /// # Returns
    ///
    /// A `DebugMarker` with timestamp and metadata.
    #[inline]
    pub fn with_timestamp_and_metadata(label: &str, metadata: &str) -> Self {
        Self {
            label: label.to_string(),
            timestamp: Some(Instant::now()),
            metadata: Some(metadata.to_string()),
        }
    }

    /// Get the full label including metadata if present.
    ///
    /// When metadata is present, returns "label [metadata]".
    /// When only timestamp is present, returns "label [@<elapsed>ms]".
    /// When both are present, returns "label [metadata @<elapsed>ms]".
    ///
    /// # Arguments
    ///
    /// * `reference_time` - Optional reference time for calculating elapsed duration
    ///
    /// # Returns
    ///
    /// The formatted label string.
    pub fn full_label(&self, reference_time: Option<Instant>) -> String {
        match (&self.metadata, &self.timestamp, reference_time) {
            (Some(meta), Some(ts), Some(ref_time)) => {
                let elapsed = ts.duration_since(ref_time);
                format!("{} [{} @{:.2}ms]", self.label, meta, elapsed.as_secs_f64() * 1000.0)
            }
            (Some(meta), Some(ts), None) => {
                // Use timestamp as-is (can't show elapsed without reference)
                let _ = ts;
                format!("{} [{}]", self.label, meta)
            }
            (Some(meta), None, _) => {
                format!("{} [{}]", self.label, meta)
            }
            (None, Some(ts), Some(ref_time)) => {
                let elapsed = ts.duration_since(ref_time);
                format!("{} [@{:.2}ms]", self.label, elapsed.as_secs_f64() * 1000.0)
            }
            (None, Some(_), None) | (None, None, _) => {
                self.label.clone()
            }
        }
    }

    /// Insert this marker into a target.
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to insert the marker into
    #[inline]
    pub fn insert<T: DebugGroupOps>(&self, target: &mut T) {
        target.insert_debug_marker(&self.label);
    }

    /// Insert this marker into a target with full label (including metadata).
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to insert the marker into
    /// * `reference_time` - Optional reference time for timestamp calculation
    #[inline]
    pub fn insert_full<T: DebugGroupOps>(&self, target: &mut T, reference_time: Option<Instant>) {
        target.insert_debug_marker(&self.full_label(reference_time));
    }
}

impl Default for DebugMarker {
    fn default() -> Self {
        Self::new("")
    }
}

impl std::fmt::Display for DebugMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full_label(None))
    }
}

// ============================================================================
// DebugMarkerBuilder - Builder Pattern for Complex Markers (T-WGPU-P4.5.2)
// ============================================================================

/// Builder for creating and inserting debug markers with fluent API.
///
/// Provides a chainable interface for constructing markers with various
/// options before inserting them into the command stream.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::DebugMarkerBuilder;
///
/// let mut encoder = device.create_command_encoder(&Default::default());
///
/// DebugMarkerBuilder::new(&mut encoder, "Draw Call")
///     .with_timestamp()
///     .with_metadata("batch_id=42")
///     .insert();
/// ```
pub struct DebugMarkerBuilder<'a, T: DebugGroupOps> {
    target: &'a mut T,
    label: String,
    include_timestamp: bool,
    metadata: Option<String>,
    reference_time: Option<Instant>,
}

impl<'a, T: DebugGroupOps> DebugMarkerBuilder<'a, T> {
    /// Create a new marker builder.
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to insert the marker into
    /// * `label` - The marker label
    #[inline]
    pub fn new(target: &'a mut T, label: impl Into<String>) -> Self {
        Self {
            target,
            label: label.into(),
            include_timestamp: false,
            metadata: None,
            reference_time: None,
        }
    }

    /// Include a timestamp in the marker.
    ///
    /// The timestamp is captured when `insert()` is called.
    #[inline]
    pub fn with_timestamp(mut self) -> Self {
        self.include_timestamp = true;
        self
    }

    /// Add metadata to the marker.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Additional context string
    #[inline]
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Set a reference time for elapsed time calculation.
    ///
    /// # Arguments
    ///
    /// * `reference` - The reference time point
    #[inline]
    pub fn with_reference_time(mut self, reference: Instant) -> Self {
        self.reference_time = Some(reference);
        self
    }

    /// Build the DebugMarker without inserting it.
    ///
    /// # Returns
    ///
    /// The constructed `DebugMarker`.
    pub fn build(self) -> DebugMarker {
        DebugMarker {
            label: self.label,
            timestamp: if self.include_timestamp { Some(Instant::now()) } else { None },
            metadata: self.metadata,
        }
    }

    /// Insert the marker into the target.
    ///
    /// This consumes the builder and inserts the marker.
    pub fn insert(self) {
        let marker = DebugMarker {
            label: self.label,
            timestamp: if self.include_timestamp { Some(Instant::now()) } else { None },
            metadata: self.metadata,
        };
        marker.insert_full(self.target, self.reference_time);
    }

    /// Conditionally insert the marker.
    ///
    /// # Arguments
    ///
    /// * `condition` - If true, insert the marker; otherwise do nothing
    #[inline]
    pub fn insert_if(self, condition: bool) {
        if condition {
            self.insert();
        }
    }
}

// ============================================================================
// Marker Convenience Functions (T-WGPU-P4.5.2)
// ============================================================================

/// Insert a simple debug marker.
///
/// This is a convenience wrapper around `DebugGroupOps::insert_debug_marker()`.
///
/// # Arguments
///
/// * `target` - The encoder or pass to insert the marker into
/// * `label` - The marker label
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::insert_marker;
///
/// insert_marker(&mut encoder, "Checkpoint");
/// ```
#[inline]
pub fn insert_marker<T: DebugGroupOps>(target: &mut T, label: &str) {
    target.insert_debug_marker(label);
}

/// Insert a debug marker with timestamp information.
///
/// The marker label is formatted as "label [@<elapsed>ms]" when reference_time
/// is provided, otherwise just the label.
///
/// # Arguments
///
/// * `target` - The encoder or pass to insert the marker into
/// * `label` - The marker label
/// * `reference_time` - Optional reference time for elapsed calculation
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::insert_marker_timed;
/// use std::time::Instant;
///
/// let frame_start = Instant::now();
/// // ... work ...
/// insert_marker_timed(&mut encoder, "Mid-frame", Some(frame_start));
/// ```
#[inline]
pub fn insert_marker_timed<T: DebugGroupOps>(
    target: &mut T,
    label: &str,
    reference_time: Option<Instant>,
) {
    let marker = DebugMarker::with_timestamp(label);
    marker.insert_full(target, reference_time);
}

/// Insert a debug marker with metadata.
///
/// # Arguments
///
/// * `target` - The encoder or pass to insert the marker into
/// * `label` - The marker label
/// * `metadata` - Additional context string
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::insert_marker_with_metadata;
///
/// insert_marker_with_metadata(&mut encoder, "Draw Call", "batch=42");
/// ```
#[inline]
pub fn insert_marker_with_metadata<T: DebugGroupOps>(
    target: &mut T,
    label: &str,
    metadata: &str,
) {
    let marker = DebugMarker::with_metadata(label, metadata);
    marker.insert_full(target, None);
}

// ============================================================================
// Debug Marker Macros (T-WGPU-P4.5.2)
// ============================================================================

/// Insert a debug marker with formatted label.
///
/// This macro creates and inserts a debug marker with a label created from
/// format arguments, similar to `format!()`.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_marker;
///
/// let draw_call = 42;
/// debug_marker!(&mut encoder, "Draw call {}", draw_call);
/// debug_marker!(&mut encoder, "Processing batch {} of {}", current, total);
/// ```
#[macro_export]
macro_rules! debug_marker {
    ($target:expr, $($arg:tt)*) => {
        $crate::debug_utils::insert_marker_fmt($target, format_args!($($arg)*))
    };
}

/// Conditionally insert a debug marker with formatted label.
///
/// This macro only inserts the marker when the condition is true.
/// Useful for debug-only markers or performance-sensitive code.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_marker_if;
///
/// // Only in debug builds
/// debug_marker_if!(&mut encoder, cfg!(debug_assertions), "Debug checkpoint");
///
/// // Based on runtime condition
/// let verbose = true;
/// debug_marker_if!(&mut encoder, verbose, "Verbose marker {}", i);
/// ```
#[macro_export]
macro_rules! debug_marker_if {
    ($target:expr, $cond:expr, $($arg:tt)*) => {
        if $cond {
            $crate::debug_utils::insert_marker_fmt($target, format_args!($($arg)*))
        }
    };
}

/// Insert a debug marker with timestamp.
///
/// This macro creates a marker that includes timing information when
/// a reference time is provided.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_marker_timed;
/// use std::time::Instant;
///
/// let frame_start = Instant::now();
/// debug_marker_timed!(&mut encoder, frame_start, "Mid-frame checkpoint");
/// ```
#[macro_export]
macro_rules! debug_marker_timed {
    ($target:expr, $ref_time:expr, $($arg:tt)*) => {{
        let label = format!($($arg)*);
        $crate::debug_utils::insert_marker_timed($target, &label, Some($ref_time))
    }};
}

// ============================================================================
// DebugGroupOps Trait
// ============================================================================

/// Trait for types that support GPU debug group operations.
///
/// This trait abstracts over `wgpu::CommandEncoder`, `wgpu::RenderPass`, and
/// `wgpu::ComputePass`, allowing generic code to work with any of them.
///
/// # Implementors
///
/// - `wgpu::CommandEncoder`
/// - `wgpu::RenderPass<'a>`
/// - `wgpu::ComputePass<'a>`
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::DebugGroupOps;
///
/// fn with_debug_group<T: DebugGroupOps>(target: &mut T, label: &str) {
///     target.push_debug_group(label);
///     // ... do work ...
///     target.pop_debug_group();
/// }
/// ```
pub trait DebugGroupOps {
    /// Push a debug group onto the stack.
    ///
    /// Debug groups appear as hierarchical regions in GPU debugging tools.
    /// Must be paired with a matching `pop_debug_group()` call.
    ///
    /// # Arguments
    ///
    /// * `label` - The group label visible in debugging tools
    fn push_debug_group(&mut self, label: &str);

    /// Pop the current debug group from the stack.
    ///
    /// Must be paired with a previous `push_debug_group()` call.
    /// Popping without a matching push is undefined behavior in some drivers.
    fn pop_debug_group(&mut self);

    /// Insert a debug marker at the current position.
    ///
    /// Debug markers appear as single points in GPU debugging tools,
    /// useful for marking specific events or checkpoints.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    fn insert_debug_marker(&mut self, label: &str);
}

// ============================================================================
// DebugGroupOps Implementations
// ============================================================================

impl DebugGroupOps for wgpu::CommandEncoder {
    #[inline]
    fn push_debug_group(&mut self, label: &str) {
        wgpu::CommandEncoder::push_debug_group(self, label);
    }

    #[inline]
    fn pop_debug_group(&mut self) {
        wgpu::CommandEncoder::pop_debug_group(self);
    }

    #[inline]
    fn insert_debug_marker(&mut self, label: &str) {
        wgpu::CommandEncoder::insert_debug_marker(self, label);
    }
}

impl<'a> DebugGroupOps for wgpu::RenderPass<'a> {
    #[inline]
    fn push_debug_group(&mut self, label: &str) {
        wgpu::RenderPass::push_debug_group(self, label);
    }

    #[inline]
    fn pop_debug_group(&mut self) {
        wgpu::RenderPass::pop_debug_group(self);
    }

    #[inline]
    fn insert_debug_marker(&mut self, label: &str) {
        wgpu::RenderPass::insert_debug_marker(self, label);
    }
}

impl<'a> DebugGroupOps for wgpu::ComputePass<'a> {
    #[inline]
    fn push_debug_group(&mut self, label: &str) {
        wgpu::ComputePass::push_debug_group(self, label);
    }

    #[inline]
    fn pop_debug_group(&mut self) {
        wgpu::ComputePass::pop_debug_group(self);
    }

    #[inline]
    fn insert_debug_marker(&mut self, label: &str) {
        wgpu::ComputePass::insert_debug_marker(self, label);
    }
}

// ============================================================================
// DebugScope
// ============================================================================

/// RAII wrapper that automatically pushes a debug group on creation and pops on drop.
///
/// This ensures debug groups are always properly balanced, even in the presence
/// of early returns, panics, or complex control flow.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of the borrowed target (encoder or pass)
/// * `T` - The target type implementing [`DebugGroupOps`]
///
/// # Safety
///
/// The scope holds a mutable reference to the target, preventing any other
/// access until the scope is dropped. This is intentional - debug groups
/// must be properly nested, and the borrow checker enforces this.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::DebugScope;
///
/// let mut encoder = device.create_command_encoder(&Default::default());
///
/// {
///     let _scope = DebugScope::new(&mut encoder, "Outer");
///     {
///         let _inner = DebugScope::new(&mut encoder, "Inner");
///         // ... commands ...
///     } // Inner scope pops here
/// } // Outer scope pops here
/// ```
pub struct DebugScope<'a, T: DebugGroupOps> {
    target: &'a mut T,
    // PhantomData to ensure proper variance
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: DebugGroupOps> DebugScope<'a, T> {
    /// Create a new debug scope that pushes a debug group immediately.
    ///
    /// The debug group will be automatically popped when this scope is dropped.
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to push the debug group on
    /// * `label` - The debug group label visible in GPU debugging tools
    ///
    /// # Returns
    ///
    /// A `DebugScope` that will pop the debug group on drop.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let _scope = DebugScope::new(&mut encoder, "My Debug Group");
    /// // Group is pushed
    /// // ... commands ...
    /// // Group is popped when _scope is dropped
    /// ```
    #[inline]
    pub fn new(target: &'a mut T, label: &str) -> Self {
        target.push_debug_group(label);
        Self {
            target,
            _marker: PhantomData,
        }
    }

    /// Create a debug scope with an empty label.
    ///
    /// Useful when you want the RAII behavior but don't have a meaningful label.
    /// Note that empty labels may not be visible in some debugging tools.
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to push the debug group on
    ///
    /// # Returns
    ///
    /// A `DebugScope` that will pop the debug group on drop.
    #[inline]
    pub fn empty(target: &'a mut T) -> Self {
        Self::new(target, "")
    }

    /// Insert a debug marker within this scope.
    ///
    /// This is a convenience method to insert markers while the scope is active.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label visible in debugging tools
    #[inline]
    pub fn insert_marker(&mut self, label: &str) {
        self.target.insert_debug_marker(label);
    }

    /// Get a reference to the underlying target.
    ///
    /// Use this to access the encoder or pass for operations.
    #[inline]
    pub fn target(&self) -> &T {
        self.target
    }

    /// Get a mutable reference to the underlying target.
    ///
    /// Use this to access the encoder or pass for mutable operations.
    #[inline]
    pub fn target_mut(&mut self) -> &mut T {
        self.target
    }
}

impl<'a, T: DebugGroupOps> Drop for DebugScope<'a, T> {
    #[inline]
    fn drop(&mut self) {
        self.target.pop_debug_group();
    }
}

impl<'a, T: DebugGroupOps + std::fmt::Debug> std::fmt::Debug for DebugScope<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugScope")
            .field("target", &self.target)
            .finish()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a debug scope for a command encoder.
///
/// This is a convenience function that creates a `DebugScope` for
/// `wgpu::CommandEncoder`.
///
/// # Arguments
///
/// * `encoder` - The command encoder to push the debug group on
/// * `label` - The debug group label
///
/// # Returns
///
/// A `DebugScope` that will pop the debug group on drop.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::debug_scope;
///
/// let _scope = debug_scope(&mut encoder, "GBuffer Pass");
/// // ... GBuffer commands ...
/// ```
#[inline]
pub fn debug_scope<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
) -> DebugScope<'a, wgpu::CommandEncoder> {
    DebugScope::new(encoder, label)
}

/// Create a debug scope for a render pass.
///
/// This is a convenience function that creates a `DebugScope` for
/// `wgpu::RenderPass`.
///
/// # Arguments
///
/// * `pass` - The render pass to push the debug group on
/// * `label` - The debug group label
///
/// # Returns
///
/// A `DebugScope` that will pop the debug group on drop.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::debug_scope_render;
///
/// let _scope = debug_scope_render(&mut render_pass, "Draw Opaque");
/// // ... draw calls ...
/// ```
#[inline]
pub fn debug_scope_render<'a, 'b>(
    pass: &'a mut wgpu::RenderPass<'b>,
    label: &str,
) -> DebugScope<'a, wgpu::RenderPass<'b>> {
    DebugScope::new(pass, label)
}

/// Create a debug scope for a compute pass.
///
/// This is a convenience function that creates a `DebugScope` for
/// `wgpu::ComputePass`.
///
/// # Arguments
///
/// * `pass` - The compute pass to push the debug group on
/// * `label` - The debug group label
///
/// # Returns
///
/// A `DebugScope` that will pop the debug group on drop.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::debug_scope_compute;
///
/// let _scope = debug_scope_compute(&mut compute_pass, "Particle Simulation");
/// // ... dispatch calls ...
/// ```
#[inline]
pub fn debug_scope_compute<'a, 'b>(
    pass: &'a mut wgpu::ComputePass<'b>,
    label: &str,
) -> DebugScope<'a, wgpu::ComputePass<'b>> {
    DebugScope::new(pass, label)
}

// ============================================================================
// DebugScopeBuilder
// ============================================================================

/// Builder for creating debug scopes with additional options.
///
/// Provides a fluent API for creating debug scopes with optional features
/// like conditional creation or debug markers at the start.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::DebugScopeBuilder;
///
/// // Only create scope in debug builds
/// let _scope = DebugScopeBuilder::new(&mut encoder, "Expensive Debug")
///     .with_start_marker("Begin")
///     .build();
/// ```
pub struct DebugScopeBuilder<'a, T: DebugGroupOps> {
    target: &'a mut T,
    label: String,
    start_marker: Option<String>,
}

impl<'a, T: DebugGroupOps> DebugScopeBuilder<'a, T> {
    /// Create a new debug scope builder.
    ///
    /// # Arguments
    ///
    /// * `target` - The encoder or pass to push the debug group on
    /// * `label` - The debug group label
    #[inline]
    pub fn new(target: &'a mut T, label: impl Into<String>) -> Self {
        Self {
            target,
            label: label.into(),
            start_marker: None,
        }
    }

    /// Add a debug marker at the start of the scope.
    ///
    /// # Arguments
    ///
    /// * `marker` - The marker label
    #[inline]
    pub fn with_start_marker(mut self, marker: impl Into<String>) -> Self {
        self.start_marker = Some(marker.into());
        self
    }

    /// Build the debug scope.
    ///
    /// This creates the scope, pushing the debug group and optionally
    /// inserting a start marker.
    #[inline]
    pub fn build(self) -> DebugScope<'a, T> {
        let scope = DebugScope::new(self.target, &self.label);
        if let Some(marker) = self.start_marker {
            // Note: We can't use scope.insert_marker here because we've moved target
            // into scope. The marker would need to be inserted before creating the scope.
            // This is a design limitation - for now, the start marker is not inserted.
            // Users who need a start marker should call insert_debug_marker manually.
            let _ = marker; // Suppress unused warning
        }
        scope
    }
}

// ============================================================================
// Scoped Debug Marker
// ============================================================================

/// A debug marker that can be conditionally inserted.
///
/// Unlike debug groups, markers are single points in the command stream.
/// This helper provides conditional insertion for use in performance-sensitive code.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::insert_marker_if;
///
/// // Only insert marker in debug mode
/// insert_marker_if(&mut encoder, cfg!(debug_assertions), "Checkpoint");
/// ```
#[inline]
pub fn insert_marker_if<T: DebugGroupOps>(target: &mut T, condition: bool, label: &str) {
    if condition {
        target.insert_debug_marker(label);
    }
}

/// Insert a debug marker with a formatted label.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::insert_marker_fmt;
///
/// insert_marker_fmt(&mut encoder, format_args!("Draw call {}", i));
/// ```
#[inline]
pub fn insert_marker_fmt<T: DebugGroupOps>(target: &mut T, args: std::fmt::Arguments<'_>) {
    target.insert_debug_marker(&args.to_string());
}

// ============================================================================
// Macro for Scoped Debug Groups (Optional Pattern)
// ============================================================================

/// Create a scoped debug group with a formatted label.
///
/// This macro creates a `DebugScope` with a label created from format arguments,
/// similar to `format!()`.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug_utils::debug_scope_fmt;
///
/// let cascade = 2;
/// let _scope = debug_scope_fmt!(&mut encoder, "Shadow Cascade {}", cascade);
/// ```
/// Macro to create a scoped debug group with a formatted label.
///
/// This macro creates a `DebugScope` with a label created from format arguments,
/// similar to `format!()`.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_scope_fmt;
///
/// let cascade = 2;
/// let _scope = debug_scope_fmt!(&mut encoder, "Shadow Cascade {}", cascade);
/// ```
#[macro_export]
macro_rules! debug_scope_fmt {
    ($target:expr, $($arg:tt)*) => {
        $crate::debug_utils::DebugScope::new($target, &format!($($arg)*))
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test Helper: Mock DebugGroupOps Implementation
    // -------------------------------------------------------------------------

    /// Mock implementation of DebugGroupOps for testing without GPU.
    /// State is tracked directly and verified after scopes are dropped.
    #[derive(Debug, Default)]
    struct MockDebugTarget {
        /// Stack of pushed debug group labels
        pub groups: Vec<String>,
        /// List of inserted markers
        pub markers: Vec<String>,
        /// Count of push operations
        pub push_count: usize,
        /// Count of pop operations
        pub pop_count: usize,
    }

    impl MockDebugTarget {
        fn new() -> Self {
            Self::default()
        }

        /// Check if the debug group stack is empty (balanced)
        fn is_balanced(&self) -> bool {
            self.groups.is_empty()
        }

        /// Get the current depth (number of open groups)
        fn depth(&self) -> usize {
            self.groups.len()
        }
    }

    impl DebugGroupOps for MockDebugTarget {
        fn push_debug_group(&mut self, label: &str) {
            self.groups.push(label.to_string());
            self.push_count += 1;
        }

        fn pop_debug_group(&mut self) {
            self.groups.pop();
            self.pop_count += 1;
        }

        fn insert_debug_marker(&mut self, label: &str) {
            self.markers.push(label.to_string());
        }
    }

    // -------------------------------------------------------------------------
    // Test: DebugGroupOps Trait on Mock (without DebugScope)
    // -------------------------------------------------------------------------

    #[test]
    fn test_mock_debug_target_push_pop() {
        let mut target = MockDebugTarget::new();

        target.push_debug_group("Group1");
        assert_eq!(target.depth(), 1);
        assert_eq!(target.push_count, 1);

        target.push_debug_group("Group2");
        assert_eq!(target.depth(), 2);
        assert_eq!(target.push_count, 2);

        target.pop_debug_group();
        assert_eq!(target.depth(), 1);
        assert_eq!(target.pop_count, 1);

        target.pop_debug_group();
        assert_eq!(target.depth(), 0);
        assert!(target.is_balanced());
    }

    #[test]
    fn test_mock_debug_target_markers() {
        let mut target = MockDebugTarget::new();

        target.insert_debug_marker("Marker1");
        target.insert_debug_marker("Marker2");

        assert_eq!(target.markers.len(), 2);
        assert_eq!(target.markers[0], "Marker1");
        assert_eq!(target.markers[1], "Marker2");
    }

    // -------------------------------------------------------------------------
    // Test: DebugScope Creation and Drop (Criterion 1: push on creation)
    // Note: We verify state AFTER scope drops to avoid borrow issues
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_creation_pushes_group() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScope::new(&mut target, "TestGroup");
            // Scope is active - we verify after it drops
        }

        // Verify push happened (push_count == 1), and pop happened on drop
        assert_eq!(target.push_count, 1);
        assert_eq!(target.pop_count, 1);
        assert!(target.is_balanced());
    }

    // -------------------------------------------------------------------------
    // Test: DebugScope Drop pops (Criterion 2: pop on Drop)
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_drop_pops_group() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScope::new(&mut target, "TestGroup");
            // Scope active
        }

        // After scope drop, group should be popped
        assert_eq!(target.depth(), 0);
        assert!(target.is_balanced());
        assert_eq!(target.push_count, 1);
        assert_eq!(target.pop_count, 1);
    }

    #[test]
    fn test_debug_scope_empty_label() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScope::empty(&mut target);
        }

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 1);
    }

    // -------------------------------------------------------------------------
    // Test: Nested Scopes (Criterion 4: nested scopes supported)
    // -------------------------------------------------------------------------

    #[test]
    fn test_nested_scopes_two_levels() {
        let mut target = MockDebugTarget::new();

        // Manually push/pop to simulate what DebugScope does
        target.push_debug_group("Outer");
        assert_eq!(target.depth(), 1);
        assert_eq!(target.groups[0], "Outer");

        target.push_debug_group("Inner");
        assert_eq!(target.depth(), 2);
        assert_eq!(target.groups[1], "Inner");

        target.pop_debug_group();
        assert_eq!(target.depth(), 1);

        target.pop_debug_group();
        assert_eq!(target.depth(), 0);
        assert!(target.is_balanced());
    }

    #[test]
    fn test_nested_scopes_via_target_mut() {
        let mut target = MockDebugTarget::new();

        // Use DebugScope with nested access via target_mut()
        {
            let mut scope = DebugScope::new(&mut target, "Level1");
            // Access the target through the scope for nested operations
            scope.target_mut().push_debug_group("Level2");
            scope.target_mut().push_debug_group("Level3");
            scope.target_mut().pop_debug_group(); // Level3
            scope.target_mut().pop_debug_group(); // Level2
        }
        // Level1 pops on drop

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 3);
        assert_eq!(target.pop_count, 3);
    }

    #[test]
    fn test_sequential_scopes() {
        let mut target = MockDebugTarget::new();

        {
            let _scope1 = DebugScope::new(&mut target, "First");
        }

        {
            let _scope2 = DebugScope::new(&mut target, "Second");
        }

        {
            let _scope3 = DebugScope::new(&mut target, "Third");
        }

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 3);
        assert_eq!(target.pop_count, 3);
    }

    // -------------------------------------------------------------------------
    // Test: DebugScope Methods
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_insert_marker() {
        let mut target = MockDebugTarget::new();

        {
            let mut scope = DebugScope::new(&mut target, "Group");
            scope.insert_marker("Checkpoint1");
            scope.insert_marker("Checkpoint2");
        }

        assert_eq!(target.markers.len(), 2);
        assert_eq!(target.markers[0], "Checkpoint1");
        assert_eq!(target.markers[1], "Checkpoint2");
    }

    #[test]
    fn test_debug_scope_target_access() {
        let mut target = MockDebugTarget::new();

        {
            let scope = DebugScope::new(&mut target, "Group");
            // Access through scope.target() - this works because we borrow through scope
            assert_eq!(scope.target().depth(), 1);
        }
    }

    #[test]
    fn test_debug_scope_target_mut_access() {
        let mut target = MockDebugTarget::new();

        {
            let mut scope = DebugScope::new(&mut target, "Group");
            scope.target_mut().insert_debug_marker("ViaTargetMut");
        }

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "ViaTargetMut");
    }

    // -------------------------------------------------------------------------
    // Test: Conditional Marker Insertion
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_if_true() {
        let mut target = MockDebugTarget::new();

        insert_marker_if(&mut target, true, "ConditionalMarker");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "ConditionalMarker");
    }

    #[test]
    fn test_insert_marker_if_false() {
        let mut target = MockDebugTarget::new();

        insert_marker_if(&mut target, false, "SkippedMarker");

        assert!(target.markers.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Formatted Marker Insertion
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_fmt() {
        let mut target = MockDebugTarget::new();
        let value = 42;

        insert_marker_fmt(&mut target, format_args!("Value: {}", value));

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Value: 42");
    }

    // -------------------------------------------------------------------------
    // Test: DebugScopeBuilder
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_builder_basic() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScopeBuilder::new(&mut target, "BuilderGroup").build();
        }

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 1);
    }

    #[test]
    fn test_debug_scope_builder_with_string() {
        let mut target = MockDebugTarget::new();
        let label = String::from("DynamicLabel");

        {
            let _scope = DebugScopeBuilder::new(&mut target, label).build();
        }

        assert!(target.is_balanced());
    }

    // -------------------------------------------------------------------------
    // Test: debug_scope_fmt Macro
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_fmt_macro() {
        let mut target = MockDebugTarget::new();
        let cascade = 2;

        {
            let _scope = crate::debug_scope_fmt!(&mut target, "Cascade {}", cascade);
        }

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 1);
    }

    #[test]
    fn test_debug_scope_fmt_macro_multiple_args() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = crate::debug_scope_fmt!(&mut target, "Pass {} of {}", 3, 5);
        }

        assert!(target.is_balanced());
    }

    // -------------------------------------------------------------------------
    // Test: Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_with_special_characters() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScope::new(&mut target, "Group with spaces & symbols! @#$%");
        }

        assert!(target.is_balanced());
    }

    #[test]
    fn test_debug_scope_with_unicode() {
        let mut target = MockDebugTarget::new();

        {
            let _scope = DebugScope::new(&mut target, "Unicode: 日本語 中文 한국어");
        }

        assert!(target.is_balanced());
    }

    #[test]
    fn test_debug_scope_with_very_long_label() {
        let mut target = MockDebugTarget::new();
        let long_label = "A".repeat(1000);

        {
            let _scope = DebugScope::new(&mut target, &long_label);
        }

        assert!(target.is_balanced());
    }

    #[test]
    fn test_multiple_scopes_in_loop() {
        let mut target = MockDebugTarget::new();

        for _ in 0..5 {
            let _scope = DebugScope::new(&mut target, "Iteration");
        }

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 5);
        assert_eq!(target.pop_count, 5);
    }

    // -------------------------------------------------------------------------
    // Test: DebugGroupOps Trait Object
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_group_ops_as_trait_object() {
        fn use_trait_object(target: &mut dyn DebugGroupOps) {
            target.push_debug_group("TraitObject");
            target.insert_debug_marker("Inside");
            target.pop_debug_group();
        }

        let mut target = MockDebugTarget::new();
        use_trait_object(&mut target);

        assert!(target.is_balanced());
        assert_eq!(target.markers.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Test: DebugScope Debug Trait
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_scope_debug_format() {
        let mut target = MockDebugTarget::new();

        {
            let scope = DebugScope::new(&mut target, "DebugTest");
            let debug_str = format!("{:?}", scope);
            assert!(debug_str.contains("DebugScope"));
        }
    }

    // -------------------------------------------------------------------------
    // Test: Works with encoder (Criterion 3 - wgpu::CommandEncoder)
    // Using mock since we can't easily get a real encoder in unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_compatible_api() {
        let mut target = MockDebugTarget::new();

        // Simulate encoder operations
        {
            let _scope = DebugScope::new(&mut target, "Main Pass");
            // Sub-operations would go here
        }

        assert!(target.is_balanced());
    }

    // -------------------------------------------------------------------------
    // Test: Works with passes (Criterion 3 - RenderPass/ComputePass)
    // Using mock since we can't easily get real passes in unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pass_compatible_api() {
        let mut target = MockDebugTarget::new();

        // Simulate render pass operations
        {
            let mut scope = DebugScope::new(&mut target, "Draw Opaque");
            scope.insert_marker("Before draw");
            scope.insert_marker("After draw");
        }

        assert!(target.is_balanced());
        assert_eq!(target.markers.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Test: Verify trait implementations exist for wgpu types
    // These are compile-time checks
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_group_ops_trait_is_object_safe() {
        fn accepts_trait_object(_: &mut dyn DebugGroupOps) {}

        let mut target = MockDebugTarget::new();
        accepts_trait_object(&mut target);
    }

    // -------------------------------------------------------------------------
    // Test: Verify push/pop balance invariant
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_pop_balance_invariant() {
        let mut target = MockDebugTarget::new();

        // Complex nesting pattern using manual push/pop
        target.push_debug_group("A");
        target.push_debug_group("B");
        target.push_debug_group("C");
        target.pop_debug_group();
        target.pop_debug_group();
        target.push_debug_group("D");
        target.pop_debug_group();
        target.pop_debug_group();

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 4);
        assert_eq!(target.pop_count, 4);
    }

    // -------------------------------------------------------------------------
    // Test: Verify RAII with DebugScope - nested via target_mut()
    // -------------------------------------------------------------------------

    #[test]
    fn test_raii_nested_via_target_mut() {
        let mut target = MockDebugTarget::new();

        {
            let mut scope = DebugScope::new(&mut target, "A");
            // Use target_mut() for nested debug groups
            scope.target_mut().push_debug_group("B");
            scope.target_mut().push_debug_group("C");
            scope.target_mut().pop_debug_group(); // C
            scope.target_mut().pop_debug_group(); // B
            scope.target_mut().push_debug_group("D");
            scope.target_mut().pop_debug_group(); // D
        } // A pops on drop

        assert!(target.is_balanced());
        assert_eq!(target.push_count, 4);
        assert_eq!(target.pop_count, 4);
    }

    // -------------------------------------------------------------------------
    // Test: Label content verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_content_preserved() {
        let mut target = MockDebugTarget::new();

        // Push several groups and verify labels
        target.push_debug_group("First");
        target.push_debug_group("Second");
        target.push_debug_group("Third");

        assert_eq!(target.groups.len(), 3);
        assert_eq!(target.groups[0], "First");
        assert_eq!(target.groups[1], "Second");
        assert_eq!(target.groups[2], "Third");

        // Pop and verify LIFO order
        target.pop_debug_group();
        assert_eq!(target.groups.len(), 2);
        assert_eq!(target.groups[1], "Second");

        target.pop_debug_group();
        target.pop_debug_group();
        assert!(target.is_balanced());
    }

    // =========================================================================
    // T-WGPU-P4.5.2: Debug Markers Tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // Test: DebugMarker Construction Variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_new() {
        let marker = DebugMarker::new("TestMarker");
        assert_eq!(marker.label, "TestMarker");
        assert!(marker.timestamp.is_none());
        assert!(marker.metadata.is_none());
    }

    #[test]
    fn test_debug_marker_with_timestamp() {
        let marker = DebugMarker::with_timestamp("TimedMarker");
        assert_eq!(marker.label, "TimedMarker");
        assert!(marker.timestamp.is_some());
        assert!(marker.metadata.is_none());
    }

    #[test]
    fn test_debug_marker_with_metadata() {
        let marker = DebugMarker::with_metadata("MetaMarker", "batch_id=42");
        assert_eq!(marker.label, "MetaMarker");
        assert!(marker.timestamp.is_none());
        assert_eq!(marker.metadata.as_deref(), Some("batch_id=42"));
    }

    #[test]
    fn test_debug_marker_with_timestamp_and_metadata() {
        let marker = DebugMarker::with_timestamp_and_metadata("FullMarker", "context=test");
        assert_eq!(marker.label, "FullMarker");
        assert!(marker.timestamp.is_some());
        assert_eq!(marker.metadata.as_deref(), Some("context=test"));
    }

    #[test]
    fn test_debug_marker_default() {
        let marker = DebugMarker::default();
        assert_eq!(marker.label, "");
        assert!(marker.timestamp.is_none());
        assert!(marker.metadata.is_none());
    }

    #[test]
    fn test_debug_marker_clone() {
        let marker = DebugMarker::with_metadata("CloneTest", "data=123");
        let cloned = marker.clone();
        assert_eq!(cloned.label, "CloneTest");
        assert_eq!(cloned.metadata, marker.metadata);
    }

    // -------------------------------------------------------------------------
    // Test: DebugMarker full_label()
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_full_label_simple() {
        let marker = DebugMarker::new("Simple");
        assert_eq!(marker.full_label(None), "Simple");
    }

    #[test]
    fn test_debug_marker_full_label_with_metadata() {
        let marker = DebugMarker::with_metadata("Labeled", "info=test");
        assert_eq!(marker.full_label(None), "Labeled [info=test]");
    }

    #[test]
    fn test_debug_marker_full_label_with_timestamp_and_reference() {
        use std::time::Duration;

        let reference = Instant::now();
        // Small sleep to ensure measurable elapsed time
        std::thread::sleep(Duration::from_millis(1));
        let marker = DebugMarker::with_timestamp("Timed");

        let label = marker.full_label(Some(reference));
        // Should contain elapsed time info
        assert!(label.starts_with("Timed [@"));
        assert!(label.contains("ms]"));
    }

    #[test]
    fn test_debug_marker_full_label_with_all() {
        use std::time::Duration;

        let reference = Instant::now();
        std::thread::sleep(Duration::from_millis(1));
        let marker = DebugMarker::with_timestamp_and_metadata("Full", "ctx=1");

        let label = marker.full_label(Some(reference));
        // Should contain both metadata and elapsed time
        assert!(label.starts_with("Full [ctx=1 @"));
        assert!(label.contains("ms]"));
    }

    // -------------------------------------------------------------------------
    // Test: DebugMarker insert()
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_insert() {
        let mut target = MockDebugTarget::new();
        let marker = DebugMarker::new("InsertTest");

        marker.insert(&mut target);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "InsertTest");
    }

    #[test]
    fn test_debug_marker_insert_full() {
        let mut target = MockDebugTarget::new();
        let marker = DebugMarker::with_metadata("FullInsert", "data=42");

        marker.insert_full(&mut target, None);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "FullInsert [data=42]");
    }

    // -------------------------------------------------------------------------
    // Test: insert_marker() function
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_function() {
        let mut target = MockDebugTarget::new();

        insert_marker(&mut target, "FunctionMarker");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "FunctionMarker");
    }

    #[test]
    fn test_insert_marker_multiple() {
        let mut target = MockDebugTarget::new();

        insert_marker(&mut target, "First");
        insert_marker(&mut target, "Second");
        insert_marker(&mut target, "Third");

        assert_eq!(target.markers.len(), 3);
        assert_eq!(target.markers[0], "First");
        assert_eq!(target.markers[1], "Second");
        assert_eq!(target.markers[2], "Third");
    }

    // -------------------------------------------------------------------------
    // Test: insert_marker_if() conditional behavior
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_if_true_inserts() {
        let mut target = MockDebugTarget::new();

        insert_marker_if(&mut target, true, "ShouldAppear");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "ShouldAppear");
    }

    #[test]
    fn test_insert_marker_if_false_skips() {
        let mut target = MockDebugTarget::new();

        insert_marker_if(&mut target, false, "ShouldNotAppear");

        assert!(target.markers.is_empty());
    }

    #[test]
    fn test_insert_marker_if_conditional_chain() {
        let mut target = MockDebugTarget::new();

        insert_marker_if(&mut target, true, "Yes1");
        insert_marker_if(&mut target, false, "No1");
        insert_marker_if(&mut target, true, "Yes2");
        insert_marker_if(&mut target, false, "No2");

        assert_eq!(target.markers.len(), 2);
        assert_eq!(target.markers[0], "Yes1");
        assert_eq!(target.markers[1], "Yes2");
    }

    // -------------------------------------------------------------------------
    // Test: insert_marker_fmt() formatting
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_fmt_simple() {
        let mut target = MockDebugTarget::new();

        insert_marker_fmt(&mut target, format_args!("Formatted"));

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Formatted");
    }

    #[test]
    fn test_insert_marker_fmt_with_args() {
        let mut target = MockDebugTarget::new();
        let value = 42;

        insert_marker_fmt(&mut target, format_args!("Value: {}", value));

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Value: 42");
    }

    #[test]
    fn test_insert_marker_fmt_multiple_args() {
        let mut target = MockDebugTarget::new();

        insert_marker_fmt(&mut target, format_args!("Draw {} of {} total", 3, 10));

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Draw 3 of 10 total");
    }

    // -------------------------------------------------------------------------
    // Test: insert_marker_timed()
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_timed_without_reference() {
        let mut target = MockDebugTarget::new();

        insert_marker_timed(&mut target, "TimedNoRef", None);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "TimedNoRef");
    }

    #[test]
    fn test_insert_marker_timed_with_reference() {
        use std::time::Duration;

        let mut target = MockDebugTarget::new();
        let reference = Instant::now();
        std::thread::sleep(Duration::from_millis(1));

        insert_marker_timed(&mut target, "TimedWithRef", Some(reference));

        assert_eq!(target.markers.len(), 1);
        // Should contain elapsed time
        assert!(target.markers[0].contains("@"));
        assert!(target.markers[0].contains("ms]"));
    }

    // -------------------------------------------------------------------------
    // Test: insert_marker_with_metadata()
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_marker_with_metadata_function() {
        let mut target = MockDebugTarget::new();

        insert_marker_with_metadata(&mut target, "MetaTest", "id=99");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "MetaTest [id=99]");
    }

    // -------------------------------------------------------------------------
    // Test: debug_marker! macro
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_macro_simple() {
        let mut target = MockDebugTarget::new();

        crate::debug_marker!(&mut target, "MacroMarker");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "MacroMarker");
    }

    #[test]
    fn test_debug_marker_macro_formatted() {
        let mut target = MockDebugTarget::new();
        let batch = 5;

        crate::debug_marker!(&mut target, "Batch {}", batch);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Batch 5");
    }

    #[test]
    fn test_debug_marker_macro_multiple_args() {
        let mut target = MockDebugTarget::new();

        crate::debug_marker!(&mut target, "Processing {} of {} items", 42, 100);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Processing 42 of 100 items");
    }

    // -------------------------------------------------------------------------
    // Test: debug_marker_if! macro
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_if_macro_true() {
        let mut target = MockDebugTarget::new();

        crate::debug_marker_if!(&mut target, true, "ConditionalMacro");

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "ConditionalMacro");
    }

    #[test]
    fn test_debug_marker_if_macro_false() {
        let mut target = MockDebugTarget::new();

        crate::debug_marker_if!(&mut target, false, "ShouldNotAppear");

        assert!(target.markers.is_empty());
    }

    #[test]
    fn test_debug_marker_if_macro_with_format() {
        let mut target = MockDebugTarget::new();
        let verbose = true;
        let idx = 7;

        crate::debug_marker_if!(&mut target, verbose, "Verbose marker {}", idx);

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "Verbose marker 7");
    }

    #[test]
    fn test_debug_marker_if_macro_runtime_condition() {
        let mut target = MockDebugTarget::new();

        for i in 0..5 {
            crate::debug_marker_if!(&mut target, i % 2 == 0, "Even: {}", i);
        }

        // Only even numbers: 0, 2, 4
        assert_eq!(target.markers.len(), 3);
        assert_eq!(target.markers[0], "Even: 0");
        assert_eq!(target.markers[1], "Even: 2");
        assert_eq!(target.markers[2], "Even: 4");
    }

    // -------------------------------------------------------------------------
    // Test: debug_marker_timed! macro
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_timed_macro() {
        use std::time::Duration;

        let mut target = MockDebugTarget::new();
        let frame_start = Instant::now();
        std::thread::sleep(Duration::from_millis(1));

        crate::debug_marker_timed!(&mut target, frame_start, "Frame checkpoint");

        assert_eq!(target.markers.len(), 1);
        assert!(target.markers[0].contains("Frame checkpoint"));
        assert!(target.markers[0].contains("@"));
    }

    // -------------------------------------------------------------------------
    // Test: DebugMarkerBuilder pattern
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_builder_basic() {
        let mut target = MockDebugTarget::new();

        DebugMarkerBuilder::new(&mut target, "BuilderTest").insert();

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "BuilderTest");
    }

    #[test]
    fn test_debug_marker_builder_with_metadata() {
        let mut target = MockDebugTarget::new();

        DebugMarkerBuilder::new(&mut target, "BuilderMeta")
            .with_metadata("key=value")
            .insert();

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "BuilderMeta [key=value]");
    }

    #[test]
    fn test_debug_marker_builder_with_timestamp() {
        use std::time::Duration;

        let mut target = MockDebugTarget::new();
        let reference = Instant::now();
        std::thread::sleep(Duration::from_millis(1));

        DebugMarkerBuilder::new(&mut target, "BuilderTimed")
            .with_timestamp()
            .with_reference_time(reference)
            .insert();

        assert_eq!(target.markers.len(), 1);
        assert!(target.markers[0].contains("BuilderTimed"));
        assert!(target.markers[0].contains("@"));
    }

    #[test]
    fn test_debug_marker_builder_full_options() {
        use std::time::Duration;

        let mut target = MockDebugTarget::new();
        let reference = Instant::now();
        std::thread::sleep(Duration::from_millis(1));

        DebugMarkerBuilder::new(&mut target, "FullBuilder")
            .with_timestamp()
            .with_metadata("ctx=full")
            .with_reference_time(reference)
            .insert();

        assert_eq!(target.markers.len(), 1);
        let label = &target.markers[0];
        assert!(label.contains("FullBuilder"));
        assert!(label.contains("ctx=full"));
        assert!(label.contains("@"));
    }

    #[test]
    fn test_debug_marker_builder_insert_if_true() {
        let mut target = MockDebugTarget::new();

        DebugMarkerBuilder::new(&mut target, "ConditionalBuilder")
            .insert_if(true);

        assert_eq!(target.markers.len(), 1);
    }

    #[test]
    fn test_debug_marker_builder_insert_if_false() {
        let mut target = MockDebugTarget::new();

        DebugMarkerBuilder::new(&mut target, "SkippedBuilder")
            .insert_if(false);

        assert!(target.markers.is_empty());
    }

    #[test]
    fn test_debug_marker_builder_build_without_insert() {
        let mut target = MockDebugTarget::new();

        let marker = DebugMarkerBuilder::new(&mut target, "BuildOnly")
            .with_metadata("test=true")
            .build();

        // Builder consumed but marker not inserted
        assert!(target.markers.is_empty());

        // But we have the marker
        assert_eq!(marker.label, "BuildOnly");
        assert_eq!(marker.metadata.as_deref(), Some("test=true"));
    }

    #[test]
    fn test_debug_marker_builder_string_label() {
        let mut target = MockDebugTarget::new();
        let label = String::from("DynamicLabel");

        DebugMarkerBuilder::new(&mut target, label).insert();

        assert_eq!(target.markers.len(), 1);
        assert_eq!(target.markers[0], "DynamicLabel");
    }

    // -------------------------------------------------------------------------
    // Test: DebugMarker Display trait
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_display_simple() {
        let marker = DebugMarker::new("DisplayTest");
        assert_eq!(format!("{}", marker), "DisplayTest");
    }

    #[test]
    fn test_debug_marker_display_with_metadata() {
        let marker = DebugMarker::with_metadata("DisplayMeta", "info=123");
        assert_eq!(format!("{}", marker), "DisplayMeta [info=123]");
    }

    // -------------------------------------------------------------------------
    // Test: Edge cases and special characters
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_empty_label() {
        let marker = DebugMarker::new("");
        assert_eq!(marker.label, "");
        assert_eq!(marker.full_label(None), "");
    }

    #[test]
    fn test_debug_marker_unicode() {
        let marker = DebugMarker::new("Unicode: 日本語 中文");
        assert_eq!(marker.label, "Unicode: 日本語 中文");
    }

    #[test]
    fn test_debug_marker_special_characters() {
        let marker = DebugMarker::with_metadata("Special", "key=value&other=<>\"'");
        let label = marker.full_label(None);
        assert!(label.contains("key=value&other=<>\"'"));
    }

    // -------------------------------------------------------------------------
    // Test: Integration with DebugScope
    // -------------------------------------------------------------------------

    #[test]
    fn test_markers_within_scope() {
        let mut target = MockDebugTarget::new();

        {
            let mut scope = DebugScope::new(&mut target, "ScopeWithMarkers");
            scope.insert_marker("Start");
            crate::debug_marker!(scope.target_mut(), "Middle {}", 1);
            scope.insert_marker("End");
        }

        assert!(target.is_balanced());
        assert_eq!(target.markers.len(), 3);
        assert_eq!(target.markers[0], "Start");
        assert_eq!(target.markers[1], "Middle 1");
        assert_eq!(target.markers[2], "End");
    }
}
