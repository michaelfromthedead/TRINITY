//! GPU Debug Marker and Label System for wgpu 25.x
//!
//! This module provides a comprehensive debug marker and label system for GPU debugging
//! integration with tools like RenderDoc, PIX, Nsight Graphics, and Metal GPU Profiler.
//!
//! # Overview
//!
//! - [`DebugLabel`] - Label with optional color for visualization
//! - [`DebugGroup`] - Debug group with nesting depth and profiling support
//! - [`DebugMarkerStack`] - Stack-based tracking of debug groups
//! - [`RenderPassDebugContext`] - Debug context wrapper for render passes
//! - [`ComputePassDebugContext`] - Debug context wrapper for compute passes
//! - [`CommandEncoderDebugContext`] - Debug context wrapper for command encoders
//! - [`DebugScopeGuard`] - RAII guard for automatic push/pop
//!
//! # Architecture
//!
//! ```text
//! DebugLabel
//!     |-- name: Cow<'static, str>
//!     |-- color: Option<[f32; 4]>
//!     `-- as_wgpu_label() -> &str
//!
//! DebugGroup
//!     |-- label: DebugLabel
//!     |-- depth: u32
//!     `-- start_time: Option<Instant>
//!
//! DebugMarkerStack
//!     |-- groups: Vec<DebugGroup>
//!     |-- max_depth: usize
//!     `-- push_group/pop_group/insert_marker
//!
//! Context Wrappers (RenderPassDebugContext, ComputePassDebugContext, etc.)
//!     |-- pass/encoder reference
//!     |-- marker_stack: DebugMarkerStack
//!     `-- push_debug_group/pop_debug_group/insert_debug_marker
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::markers::*;
//!
//! // Create a debug context for a render pass
//! let mut ctx = RenderPassDebugContext::new(&mut render_pass);
//!
//! // Push a debug group with color
//! ctx.push_debug_group(DebugLabel::with_color("Shadow Pass", [0.2, 0.4, 0.8, 1.0]));
//!
//! // Insert a marker
//! ctx.insert_debug_marker(DebugLabel::new("Draw Cascades"));
//!
//! // Pop the group
//! ctx.pop_debug_group();
//! ```
//!
//! # Thread Safety
//!
//! All types have the same thread safety as their underlying wgpu types:
//! - `DebugLabel`, `DebugGroup`, `DebugMarkerStack`: `Send + Sync`
//! - `RenderPassDebugContext`, `ComputePassDebugContext`: `!Send + !Sync`
//! - `CommandEncoderDebugContext`: `Send + Sync`

use std::borrow::Cow;
use std::time::Instant;

// ============================================================================
// DebugLabel
// ============================================================================

/// A debug label with optional color for visualization in GPU debugging tools.
///
/// Many GPU debugging tools support colored labels for better organization
/// and visual distinction between different parts of the frame.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::DebugLabel;
///
/// // Simple label
/// let label = DebugLabel::new("GBuffer Pass");
///
/// // Label with color (RGBA)
/// let colored = DebugLabel::with_color("Shadow Pass", [0.2, 0.4, 0.8, 1.0]);
///
/// // Static label (no allocation)
/// let static_label = DebugLabel::new_static("Lighting Pass");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DebugLabel {
    /// The label name
    pub name: Cow<'static, str>,
    /// Optional RGBA color for visualization (0.0 to 1.0 range)
    pub color: Option<[f32; 4]>,
}

impl DebugLabel {
    /// Create a new debug label with just a name.
    ///
    /// # Arguments
    ///
    /// * `name` - The label text
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = DebugLabel::new("My Pass");
    /// assert_eq!(label.as_wgpu_label(), "My Pass");
    /// assert!(label.color.is_none());
    /// ```
    #[inline]
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            color: None,
        }
    }

    /// Create a new debug label from a static string (zero allocation).
    ///
    /// # Arguments
    ///
    /// * `name` - Static string label
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = DebugLabel::new_static("Static Label");
    /// ```
    #[inline]
    pub const fn new_static(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            color: None,
        }
    }

    /// Create a new debug label with a name and color.
    ///
    /// # Arguments
    ///
    /// * `name` - The label text
    /// * `color` - RGBA color array with values in 0.0 to 1.0 range
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Red label
    /// let label = DebugLabel::with_color("Error Zone", [1.0, 0.0, 0.0, 1.0]);
    ///
    /// // Semi-transparent blue
    /// let label = DebugLabel::with_color("Water Pass", [0.0, 0.5, 1.0, 0.7]);
    /// ```
    #[inline]
    pub fn with_color(name: impl Into<Cow<'static, str>>, color: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            color: Some(color),
        }
    }

    /// Create a debug label with a static name and color (minimal allocation).
    ///
    /// # Arguments
    ///
    /// * `name` - Static string label
    /// * `color` - RGBA color array
    #[inline]
    pub const fn with_static_color(name: &'static str, color: [f32; 4]) -> Self {
        Self {
            name: Cow::Borrowed(name),
            color: Some(color),
        }
    }

    /// Get the label as a wgpu-compatible label string.
    ///
    /// This returns just the name portion for use with wgpu's debug APIs.
    ///
    /// # Returns
    ///
    /// A string slice suitable for wgpu debug group/marker calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = DebugLabel::new("My Pass");
    /// encoder.push_debug_group(label.as_wgpu_label());
    /// ```
    #[inline]
    pub fn as_wgpu_label(&self) -> &str {
        &self.name
    }

    /// Check if this label has a color.
    #[inline]
    pub fn has_color(&self) -> bool {
        self.color.is_some()
    }

    /// Get the color as normalized RGB (dropping alpha).
    ///
    /// Returns `None` if no color is set.
    #[inline]
    pub fn rgb(&self) -> Option<[f32; 3]> {
        self.color.map(|c| [c[0], c[1], c[2]])
    }

    /// Get the color as 8-bit RGBA.
    ///
    /// Returns `None` if no color is set.
    #[inline]
    pub fn rgba_u8(&self) -> Option<[u8; 4]> {
        self.color.map(|c| {
            [
                (c[0] * 255.0).clamp(0.0, 255.0) as u8,
                (c[1] * 255.0).clamp(0.0, 255.0) as u8,
                (c[2] * 255.0).clamp(0.0, 255.0) as u8,
                (c[3] * 255.0).clamp(0.0, 255.0) as u8,
            ]
        })
    }

    /// Create a child label by appending a suffix.
    ///
    /// # Arguments
    ///
    /// * `suffix` - Text to append after a separator
    ///
    /// # Example
    ///
    /// ```ignore
    /// let parent = DebugLabel::new("Shadow Pass");
    /// let child = parent.child("Cascade 0");
    /// assert_eq!(child.as_wgpu_label(), "Shadow Pass/Cascade 0");
    /// ```
    #[inline]
    pub fn child(&self, suffix: &str) -> Self {
        Self {
            name: Cow::Owned(format!("{}/{}", self.name, suffix)),
            color: self.color,
        }
    }
}

impl Default for DebugLabel {
    fn default() -> Self {
        Self::new_static("")
    }
}

impl std::fmt::Display for DebugLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl From<&'static str> for DebugLabel {
    fn from(s: &'static str) -> Self {
        Self::new_static(s)
    }
}

impl From<String> for DebugLabel {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

// ============================================================================
// Predefined Label Colors
// ============================================================================

/// Common label colors for consistent visual organization.
pub mod colors {
    /// Geometry pass color (orange)
    pub const GEOMETRY: [f32; 4] = [1.0, 0.6, 0.2, 1.0];
    /// Shadow pass color (dark blue)
    pub const SHADOW: [f32; 4] = [0.2, 0.3, 0.6, 1.0];
    /// Lighting pass color (yellow)
    pub const LIGHTING: [f32; 4] = [1.0, 0.9, 0.3, 1.0];
    /// Post-process pass color (purple)
    pub const POST_PROCESS: [f32; 4] = [0.7, 0.3, 0.9, 1.0];
    /// Compute pass color (cyan)
    pub const COMPUTE: [f32; 4] = [0.2, 0.8, 0.9, 1.0];
    /// UI/HUD pass color (green)
    pub const UI: [f32; 4] = [0.3, 0.9, 0.4, 1.0];
    /// Debug/visualization color (red)
    pub const DEBUG: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
    /// Transparent/particle pass (white)
    pub const TRANSPARENT: [f32; 4] = [0.9, 0.9, 0.95, 1.0];
    /// Ray tracing pass (magenta)
    pub const RAYTRACING: [f32; 4] = [0.9, 0.2, 0.7, 1.0];
    /// Copy/transfer operations (gray)
    pub const TRANSFER: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
}

// ============================================================================
// DebugGroup
// ============================================================================

/// A debug group representing a hierarchical region in the GPU command stream.
///
/// Debug groups track nesting depth and optionally support profiling through
/// start timestamps.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::{DebugGroup, DebugLabel};
///
/// // Basic group
/// let group = DebugGroup::new(DebugLabel::new("My Pass"));
///
/// // Group with profiling enabled
/// let profiled = DebugGroup::with_profiling(DebugLabel::new("Timed Pass"));
/// assert!(profiled.start_time.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct DebugGroup {
    /// The debug label for this group
    pub label: DebugLabel,
    /// Nesting depth (0 = root level)
    pub depth: u32,
    /// Optional start timestamp for profiling
    pub start_time: Option<Instant>,
}

impl DebugGroup {
    /// Create a new debug group without profiling.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label for this group
    #[inline]
    pub fn new(label: DebugLabel) -> Self {
        Self {
            label,
            depth: 0,
            start_time: None,
        }
    }

    /// Create a new debug group at a specific depth.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label for this group
    /// * `depth` - The nesting depth
    #[inline]
    pub fn with_depth(label: DebugLabel, depth: u32) -> Self {
        Self {
            label,
            depth,
            start_time: None,
        }
    }

    /// Create a new debug group with profiling enabled.
    ///
    /// The start timestamp is captured immediately using `Instant::now()`.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label for this group
    #[inline]
    pub fn with_profiling(label: DebugLabel) -> Self {
        Self {
            label,
            depth: 0,
            start_time: Some(Instant::now()),
        }
    }

    /// Create a new debug group with both depth and profiling.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label for this group
    /// * `depth` - The nesting depth
    #[inline]
    pub fn with_depth_and_profiling(label: DebugLabel, depth: u32) -> Self {
        Self {
            label,
            depth,
            start_time: Some(Instant::now()),
        }
    }

    /// Get the elapsed time since this group was created.
    ///
    /// Returns `None` if profiling was not enabled.
    #[inline]
    pub fn elapsed(&self) -> Option<std::time::Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Get the elapsed time in milliseconds.
    ///
    /// Returns `None` if profiling was not enabled.
    #[inline]
    pub fn elapsed_ms(&self) -> Option<f64> {
        self.elapsed().map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Check if profiling is enabled for this group.
    #[inline]
    pub fn has_profiling(&self) -> bool {
        self.start_time.is_some()
    }
}

// ============================================================================
// DebugMarkerStack
// ============================================================================

/// A stack-based tracker for debug groups.
///
/// This maintains the hierarchy of debug groups and ensures proper nesting.
/// It also supports a maximum depth limit to prevent excessive nesting.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::{DebugMarkerStack, DebugLabel};
///
/// let mut stack = DebugMarkerStack::new();
///
/// // Push groups
/// stack.push_group(DebugLabel::new("Outer"));
/// assert_eq!(stack.current_depth(), 1);
///
/// stack.push_group(DebugLabel::new("Inner"));
/// assert_eq!(stack.current_depth(), 2);
///
/// // Pop groups
/// let inner = stack.pop_group();
/// assert!(inner.is_some());
/// assert_eq!(stack.current_depth(), 1);
///
/// let outer = stack.pop_group();
/// assert!(outer.is_some());
/// assert!(stack.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct DebugMarkerStack {
    /// Stack of active debug groups
    groups: Vec<DebugGroup>,
    /// Maximum allowed nesting depth
    max_depth: usize,
    /// Whether profiling is enabled for new groups
    profiling_enabled: bool,
    /// Markers inserted at the current level (for tracking)
    markers_at_current_depth: usize,
}

impl DebugMarkerStack {
    /// Default maximum depth (16 levels of nesting).
    pub const DEFAULT_MAX_DEPTH: usize = 16;

    /// Create a new empty debug marker stack.
    ///
    /// Uses the default maximum depth of 16.
    #[inline]
    pub fn new() -> Self {
        Self {
            groups: Vec::with_capacity(8),
            max_depth: Self::DEFAULT_MAX_DEPTH,
            profiling_enabled: false,
            markers_at_current_depth: 0,
        }
    }

    /// Create a new debug marker stack with a custom maximum depth.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum allowed nesting depth
    #[inline]
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            groups: Vec::with_capacity(max_depth.min(16)),
            max_depth,
            profiling_enabled: false,
            markers_at_current_depth: 0,
        }
    }

    /// Create a new debug marker stack with profiling enabled.
    ///
    /// All groups pushed will have profiling timestamps.
    #[inline]
    pub fn with_profiling() -> Self {
        Self {
            groups: Vec::with_capacity(8),
            max_depth: Self::DEFAULT_MAX_DEPTH,
            profiling_enabled: true,
            markers_at_current_depth: 0,
        }
    }

    /// Enable or disable automatic profiling for new groups.
    #[inline]
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Check if profiling is enabled.
    #[inline]
    pub fn profiling_enabled(&self) -> bool {
        self.profiling_enabled
    }

    /// Push a new debug group onto the stack.
    ///
    /// Returns `true` if the group was pushed, `false` if max depth was exceeded.
    ///
    /// # Arguments
    ///
    /// * `label` - The label for the new group
    ///
    /// # Returns
    ///
    /// `true` if successful, `false` if depth limit exceeded.
    pub fn push_group(&mut self, label: DebugLabel) -> bool {
        if self.groups.len() >= self.max_depth {
            return false;
        }

        let depth = self.groups.len() as u32;
        let group = if self.profiling_enabled {
            DebugGroup::with_depth_and_profiling(label, depth)
        } else {
            DebugGroup::with_depth(label, depth)
        };

        self.groups.push(group);
        self.markers_at_current_depth = 0;
        true
    }

    /// Pop the current debug group from the stack.
    ///
    /// Returns the popped group, or `None` if the stack was empty.
    pub fn pop_group(&mut self) -> Option<DebugGroup> {
        let group = self.groups.pop();
        self.markers_at_current_depth = 0;
        group
    }

    /// Insert a marker at the current depth.
    ///
    /// This just tracks that a marker was inserted; the actual insertion
    /// happens via the wgpu API through the context wrappers.
    ///
    /// # Arguments
    ///
    /// * `_label` - The marker label (tracked for statistics)
    #[inline]
    pub fn insert_marker(&mut self, _label: &DebugLabel) {
        self.markers_at_current_depth += 1;
    }

    /// Get the current nesting depth.
    ///
    /// Returns 0 if no groups are active, 1 for one group, etc.
    #[inline]
    pub fn current_depth(&self) -> usize {
        self.groups.len()
    }

    /// Check if the stack is empty (no active groups).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Clear all groups from the stack.
    ///
    /// Note: This does NOT pop the groups from the GPU - you must ensure
    /// matching wgpu pop_debug_group calls are made.
    #[inline]
    pub fn clear(&mut self) {
        self.groups.clear();
        self.markers_at_current_depth = 0;
    }

    /// Get the maximum allowed depth.
    #[inline]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Get a reference to the current (innermost) group.
    #[inline]
    pub fn current_group(&self) -> Option<&DebugGroup> {
        self.groups.last()
    }

    /// Get the number of markers inserted at the current depth.
    #[inline]
    pub fn markers_at_depth(&self) -> usize {
        self.markers_at_current_depth
    }

    /// Get all active groups (from outermost to innermost).
    #[inline]
    pub fn groups(&self) -> &[DebugGroup] {
        &self.groups
    }

    /// Check if we can push another group (not at max depth).
    #[inline]
    pub fn can_push(&self) -> bool {
        self.groups.len() < self.max_depth
    }

    /// Get the label hierarchy as a path string.
    ///
    /// # Example
    ///
    /// ```ignore
    /// stack.push_group(DebugLabel::new("Shadows"));
    /// stack.push_group(DebugLabel::new("Cascade0"));
    /// assert_eq!(stack.path(), "Shadows/Cascade0");
    /// ```
    pub fn path(&self) -> String {
        self.groups
            .iter()
            .map(|g| g.label.as_wgpu_label())
            .collect::<Vec<_>>()
            .join("/")
    }
}

impl Default for DebugMarkerStack {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RenderPassDebugContext
// ============================================================================

/// Debug context wrapper for a render pass.
///
/// Provides debug group/marker management with automatic stack tracking.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::{RenderPassDebugContext, DebugLabel};
///
/// let mut ctx = RenderPassDebugContext::new(&mut render_pass);
///
/// ctx.push_debug_group(DebugLabel::new("Opaque Objects"));
/// // ... draw calls ...
/// ctx.insert_debug_marker(DebugLabel::new("After terrain"));
/// // ... more draw calls ...
/// ctx.pop_debug_group();
/// ```
pub struct RenderPassDebugContext<'a, 'b> {
    /// The underlying render pass
    pub pass: &'a mut wgpu::RenderPass<'b>,
    /// Stack tracking debug groups
    pub marker_stack: DebugMarkerStack,
}

impl<'a, 'b> RenderPassDebugContext<'a, 'b> {
    /// Create a new debug context wrapping a render pass.
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to wrap
    #[inline]
    pub fn new(pass: &'a mut wgpu::RenderPass<'b>) -> Self {
        Self {
            pass,
            marker_stack: DebugMarkerStack::new(),
        }
    }

    /// Create a debug context with profiling enabled.
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to wrap
    #[inline]
    pub fn with_profiling(pass: &'a mut wgpu::RenderPass<'b>) -> Self {
        Self {
            pass,
            marker_stack: DebugMarkerStack::with_profiling(),
        }
    }

    /// Create a debug context with a custom marker stack.
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to wrap
    /// * `stack` - Pre-configured marker stack
    #[inline]
    pub fn with_stack(pass: &'a mut wgpu::RenderPass<'b>, stack: DebugMarkerStack) -> Self {
        Self {
            pass,
            marker_stack: stack,
        }
    }

    /// Push a debug group.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label for the group
    ///
    /// # Returns
    ///
    /// `true` if successful, `false` if max depth exceeded.
    pub fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        if self.marker_stack.push_group(label.clone()) {
            self.pass.push_debug_group(label.as_wgpu_label());
            true
        } else {
            false
        }
    }

    /// Push a debug group from a string label.
    ///
    /// Convenience method that creates a `DebugLabel` internally.
    pub fn push_group(&mut self, label: &str) -> bool {
        self.push_debug_group(DebugLabel::new(label.to_string()))
    }

    /// Pop the current debug group.
    ///
    /// # Returns
    ///
    /// The popped group, or `None` if no groups were active.
    pub fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        let group = self.marker_stack.pop_group();
        if group.is_some() {
            self.pass.pop_debug_group();
        }
        group
    }

    /// Insert a debug marker.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label
    pub fn insert_debug_marker(&mut self, label: DebugLabel) {
        self.marker_stack.insert_marker(&label);
        self.pass.insert_debug_marker(label.as_wgpu_label());
    }

    /// Insert a debug marker from a string.
    ///
    /// Convenience method that creates a `DebugLabel` internally.
    pub fn insert_marker(&mut self, label: &str) {
        self.insert_debug_marker(DebugLabel::new(label.to_string()));
    }

    /// Get the current nesting depth.
    #[inline]
    pub fn current_depth(&self) -> usize {
        self.marker_stack.current_depth()
    }

    /// Check if any debug groups are active.
    #[inline]
    pub fn has_active_groups(&self) -> bool {
        !self.marker_stack.is_empty()
    }

    /// Get access to the underlying render pass.
    #[inline]
    pub fn pass(&mut self) -> &mut wgpu::RenderPass<'b> {
        self.pass
    }

    /// Create a scoped debug group that auto-pops on drop.
    ///
    /// # Arguments
    ///
    /// * `label` - The debug label
    ///
    /// # Returns
    ///
    /// A guard that pops the group when dropped.
    pub fn scoped_group(&mut self, label: DebugLabel) -> RenderPassDebugScopeGuard<'_, 'a, 'b> {
        self.push_debug_group(label);
        RenderPassDebugScopeGuard { ctx: self }
    }
}

/// RAII guard for render pass debug groups.
pub struct RenderPassDebugScopeGuard<'c, 'a, 'b> {
    ctx: &'c mut RenderPassDebugContext<'a, 'b>,
}

impl<'c, 'a, 'b> Drop for RenderPassDebugScopeGuard<'c, 'a, 'b> {
    fn drop(&mut self) {
        self.ctx.pop_debug_group();
    }
}

impl<'c, 'a, 'b> RenderPassDebugScopeGuard<'c, 'a, 'b> {
    /// Insert a marker within this scope.
    pub fn insert_marker(&mut self, label: &str) {
        self.ctx.insert_marker(label);
    }

    /// Get access to the render pass.
    pub fn pass(&mut self) -> &mut wgpu::RenderPass<'b> {
        self.ctx.pass()
    }
}

// ============================================================================
// ComputePassDebugContext
// ============================================================================

/// Debug context wrapper for a compute pass.
///
/// Provides debug group/marker management with automatic stack tracking.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::{ComputePassDebugContext, DebugLabel};
///
/// let mut ctx = ComputePassDebugContext::new(&mut compute_pass);
///
/// ctx.push_debug_group(DebugLabel::new("Particle Simulation"));
/// // ... dispatch calls ...
/// ctx.pop_debug_group();
/// ```
pub struct ComputePassDebugContext<'a, 'b> {
    /// The underlying compute pass
    pub pass: &'a mut wgpu::ComputePass<'b>,
    /// Stack tracking debug groups
    pub marker_stack: DebugMarkerStack,
}

impl<'a, 'b> ComputePassDebugContext<'a, 'b> {
    /// Create a new debug context wrapping a compute pass.
    #[inline]
    pub fn new(pass: &'a mut wgpu::ComputePass<'b>) -> Self {
        Self {
            pass,
            marker_stack: DebugMarkerStack::new(),
        }
    }

    /// Create a debug context with profiling enabled.
    #[inline]
    pub fn with_profiling(pass: &'a mut wgpu::ComputePass<'b>) -> Self {
        Self {
            pass,
            marker_stack: DebugMarkerStack::with_profiling(),
        }
    }

    /// Create a debug context with a custom marker stack.
    #[inline]
    pub fn with_stack(pass: &'a mut wgpu::ComputePass<'b>, stack: DebugMarkerStack) -> Self {
        Self {
            pass,
            marker_stack: stack,
        }
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        if self.marker_stack.push_group(label.clone()) {
            self.pass.push_debug_group(label.as_wgpu_label());
            true
        } else {
            false
        }
    }

    /// Push a debug group from a string label.
    pub fn push_group(&mut self, label: &str) -> bool {
        self.push_debug_group(DebugLabel::new(label.to_string()))
    }

    /// Pop the current debug group.
    pub fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        let group = self.marker_stack.pop_group();
        if group.is_some() {
            self.pass.pop_debug_group();
        }
        group
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: DebugLabel) {
        self.marker_stack.insert_marker(&label);
        self.pass.insert_debug_marker(label.as_wgpu_label());
    }

    /// Insert a debug marker from a string.
    pub fn insert_marker(&mut self, label: &str) {
        self.insert_debug_marker(DebugLabel::new(label.to_string()));
    }

    /// Get the current nesting depth.
    #[inline]
    pub fn current_depth(&self) -> usize {
        self.marker_stack.current_depth()
    }

    /// Check if any debug groups are active.
    #[inline]
    pub fn has_active_groups(&self) -> bool {
        !self.marker_stack.is_empty()
    }

    /// Get access to the underlying compute pass.
    #[inline]
    pub fn pass(&mut self) -> &mut wgpu::ComputePass<'b> {
        self.pass
    }

    /// Create a scoped debug group that auto-pops on drop.
    pub fn scoped_group(&mut self, label: DebugLabel) -> ComputePassDebugScopeGuard<'_, 'a, 'b> {
        self.push_debug_group(label);
        ComputePassDebugScopeGuard { ctx: self }
    }
}

/// RAII guard for compute pass debug groups.
pub struct ComputePassDebugScopeGuard<'c, 'a, 'b> {
    ctx: &'c mut ComputePassDebugContext<'a, 'b>,
}

impl<'c, 'a, 'b> Drop for ComputePassDebugScopeGuard<'c, 'a, 'b> {
    fn drop(&mut self) {
        self.ctx.pop_debug_group();
    }
}

impl<'c, 'a, 'b> ComputePassDebugScopeGuard<'c, 'a, 'b> {
    /// Insert a marker within this scope.
    pub fn insert_marker(&mut self, label: &str) {
        self.ctx.insert_marker(label);
    }

    /// Get access to the compute pass.
    pub fn pass(&mut self) -> &mut wgpu::ComputePass<'b> {
        self.ctx.pass()
    }
}

// ============================================================================
// CommandEncoderDebugContext
// ============================================================================

/// Debug context wrapper for a command encoder.
///
/// Provides debug group/marker management with automatic stack tracking.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::{CommandEncoderDebugContext, DebugLabel};
///
/// let mut ctx = CommandEncoderDebugContext::new(&mut encoder);
///
/// ctx.push_debug_group(DebugLabel::new("Frame"));
/// // ... begin render passes, copy operations, etc. ...
/// ctx.pop_debug_group();
/// ```
pub struct CommandEncoderDebugContext<'a> {
    /// The underlying command encoder
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Stack tracking debug groups
    pub marker_stack: DebugMarkerStack,
}

impl<'a> CommandEncoderDebugContext<'a> {
    /// Create a new debug context wrapping a command encoder.
    #[inline]
    pub fn new(encoder: &'a mut wgpu::CommandEncoder) -> Self {
        Self {
            encoder,
            marker_stack: DebugMarkerStack::new(),
        }
    }

    /// Create a debug context with profiling enabled.
    #[inline]
    pub fn with_profiling(encoder: &'a mut wgpu::CommandEncoder) -> Self {
        Self {
            encoder,
            marker_stack: DebugMarkerStack::with_profiling(),
        }
    }

    /// Create a debug context with a custom marker stack.
    #[inline]
    pub fn with_stack(encoder: &'a mut wgpu::CommandEncoder, stack: DebugMarkerStack) -> Self {
        Self {
            encoder,
            marker_stack: stack,
        }
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        if self.marker_stack.push_group(label.clone()) {
            self.encoder.push_debug_group(label.as_wgpu_label());
            true
        } else {
            false
        }
    }

    /// Push a debug group from a string label.
    pub fn push_group(&mut self, label: &str) -> bool {
        self.push_debug_group(DebugLabel::new(label.to_string()))
    }

    /// Pop the current debug group.
    pub fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        let group = self.marker_stack.pop_group();
        if group.is_some() {
            self.encoder.pop_debug_group();
        }
        group
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: DebugLabel) {
        self.marker_stack.insert_marker(&label);
        self.encoder.insert_debug_marker(label.as_wgpu_label());
    }

    /// Insert a debug marker from a string.
    pub fn insert_marker(&mut self, label: &str) {
        self.insert_debug_marker(DebugLabel::new(label.to_string()));
    }

    /// Get the current nesting depth.
    #[inline]
    pub fn current_depth(&self) -> usize {
        self.marker_stack.current_depth()
    }

    /// Check if any debug groups are active.
    #[inline]
    pub fn has_active_groups(&self) -> bool {
        !self.marker_stack.is_empty()
    }

    /// Get access to the underlying encoder.
    #[inline]
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder
    }

    /// Create a scoped debug group that auto-pops on drop.
    pub fn scoped_group(&mut self, label: DebugLabel) -> CommandEncoderDebugScopeGuard<'_, 'a> {
        self.push_debug_group(label);
        CommandEncoderDebugScopeGuard { ctx: self }
    }
}

/// RAII guard for command encoder debug groups.
pub struct CommandEncoderDebugScopeGuard<'c, 'a> {
    ctx: &'c mut CommandEncoderDebugContext<'a>,
}

impl<'c, 'a> Drop for CommandEncoderDebugScopeGuard<'c, 'a> {
    fn drop(&mut self) {
        self.ctx.pop_debug_group();
    }
}

impl<'c, 'a> CommandEncoderDebugScopeGuard<'c, 'a> {
    /// Insert a marker within this scope.
    pub fn insert_marker(&mut self, label: &str) {
        self.ctx.insert_marker(label);
    }

    /// Get access to the command encoder.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.ctx.encoder()
    }
}

// ============================================================================
// Generic DebugScopeGuard (RAII)
// ============================================================================

/// Generic RAII guard for debug scopes.
///
/// This guard automatically pops a debug group when dropped, ensuring
/// balanced push/pop even with early returns or panics.
///
/// # Type Parameters
///
/// * `T` - The target type that supports debug group operations
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::markers::DebugScopeGuard;
///
/// fn render_scene(encoder: &mut wgpu::CommandEncoder) {
///     let _scope = DebugScopeGuard::new(encoder, "Scene Rendering");
///     // ... rendering code ...
///     // Group automatically popped when _scope goes out of scope
/// }
/// ```
pub struct DebugScopeGuard<'a, T>
where
    T: DebugContextOps,
{
    target: &'a mut T,
}

impl<'a, T> DebugScopeGuard<'a, T>
where
    T: DebugContextOps,
{
    /// Create a new debug scope guard.
    ///
    /// Pushes the debug group immediately.
    ///
    /// # Arguments
    ///
    /// * `target` - The target to push the debug group on
    /// * `label` - The debug label
    #[inline]
    pub fn new(target: &'a mut T, label: impl Into<DebugLabel>) -> Self {
        target.push_debug_group(label.into());
        Self { target }
    }

    /// Insert a marker within this scope.
    #[inline]
    pub fn insert_marker(&mut self, label: impl Into<DebugLabel>) {
        self.target.insert_debug_marker(label.into());
    }
}

impl<'a, T> Drop for DebugScopeGuard<'a, T>
where
    T: DebugContextOps,
{
    fn drop(&mut self) {
        self.target.pop_debug_group();
    }
}

// ============================================================================
// DebugContextOps Trait
// ============================================================================

/// Trait for types supporting debug context operations.
///
/// This provides a unified interface across all debug context wrappers.
pub trait DebugContextOps {
    /// Push a debug group.
    fn push_debug_group(&mut self, label: DebugLabel) -> bool;

    /// Pop the current debug group.
    fn pop_debug_group(&mut self) -> Option<DebugGroup>;

    /// Insert a debug marker.
    fn insert_debug_marker(&mut self, label: DebugLabel);

    /// Get the current nesting depth.
    fn current_depth(&self) -> usize;

    /// Check if any debug groups are active.
    fn has_active_groups(&self) -> bool;
}

impl<'a, 'b> DebugContextOps for RenderPassDebugContext<'a, 'b> {
    fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        RenderPassDebugContext::push_debug_group(self, label)
    }

    fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        RenderPassDebugContext::pop_debug_group(self)
    }

    fn insert_debug_marker(&mut self, label: DebugLabel) {
        RenderPassDebugContext::insert_debug_marker(self, label)
    }

    fn current_depth(&self) -> usize {
        RenderPassDebugContext::current_depth(self)
    }

    fn has_active_groups(&self) -> bool {
        RenderPassDebugContext::has_active_groups(self)
    }
}

impl<'a, 'b> DebugContextOps for ComputePassDebugContext<'a, 'b> {
    fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        ComputePassDebugContext::push_debug_group(self, label)
    }

    fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        ComputePassDebugContext::pop_debug_group(self)
    }

    fn insert_debug_marker(&mut self, label: DebugLabel) {
        ComputePassDebugContext::insert_debug_marker(self, label)
    }

    fn current_depth(&self) -> usize {
        ComputePassDebugContext::current_depth(self)
    }

    fn has_active_groups(&self) -> bool {
        ComputePassDebugContext::has_active_groups(self)
    }
}

impl<'a> DebugContextOps for CommandEncoderDebugContext<'a> {
    fn push_debug_group(&mut self, label: DebugLabel) -> bool {
        CommandEncoderDebugContext::push_debug_group(self, label)
    }

    fn pop_debug_group(&mut self) -> Option<DebugGroup> {
        CommandEncoderDebugContext::pop_debug_group(self)
    }

    fn insert_debug_marker(&mut self, label: DebugLabel) {
        CommandEncoderDebugContext::insert_debug_marker(self, label)
    }

    fn current_depth(&self) -> usize {
        CommandEncoderDebugContext::current_depth(self)
    }

    fn has_active_groups(&self) -> bool {
        CommandEncoderDebugContext::has_active_groups(self)
    }
}

// ============================================================================
// Macros
// ============================================================================

/// Create a scoped debug group with automatic push/pop.
///
/// This macro pushes a debug group at the start and ensures it is popped
/// when the scope exits, even on early return or panic.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_ctx_group;
///
/// debug_ctx_group!(ctx, "Shadow Pass", {
///     // ... render shadows ...
///     // Group automatically popped at the end
/// });
///
/// // Or with early return:
/// debug_ctx_group!(ctx, "Lighting", {
///     if skip_lighting {
///         return; // Group still popped correctly
///     }
///     // ... lighting code ...
/// });
/// ```
#[macro_export]
macro_rules! debug_ctx_group {
    ($ctx:expr, $label:expr, $body:block) => {{
        let _guard = $crate::debug::markers::DebugScopeGuard::new(
            $ctx,
            $crate::debug::markers::DebugLabel::new($label),
        );
        $body
    }};
    ($ctx:expr, $label:expr) => {
        $crate::debug::markers::DebugScopeGuard::new(
            $ctx,
            $crate::debug::markers::DebugLabel::new($label),
        )
    };
}

/// Insert a debug marker using a debug context.
///
/// This macro is for use with `DebugContextOps` types (e.g., `RenderPassDebugContext`).
/// For direct wgpu types, use `debug_marker!` from `debug_utils`.
///
/// # Usage
///
/// ```ignore
/// use renderer_backend::debug_ctx_marker;
///
/// debug_ctx_marker!(ctx, "Checkpoint A");
/// debug_ctx_marker!(ctx, "Draw call {}", draw_index);
/// ```
#[macro_export]
macro_rules! debug_ctx_marker {
    ($ctx:expr, $label:literal) => {
        $ctx.insert_debug_marker($crate::debug::markers::DebugLabel::new_static($label))
    };
    ($ctx:expr, $($arg:tt)*) => {
        $ctx.insert_debug_marker($crate::debug::markers::DebugLabel::new(format!($($arg)*)))
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DebugLabel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_label_new() {
        let label = DebugLabel::new("Test Label");
        assert_eq!(label.as_wgpu_label(), "Test Label");
        assert!(label.color.is_none());
    }

    #[test]
    fn test_debug_label_static() {
        let label = DebugLabel::new_static("Static");
        assert_eq!(label.as_wgpu_label(), "Static");
        match &label.name {
            Cow::Borrowed(_) => {} // Expected
            Cow::Owned(_) => panic!("Expected borrowed string"),
        }
    }

    #[test]
    fn test_debug_label_with_color() {
        let label = DebugLabel::with_color("Colored", [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(label.as_wgpu_label(), "Colored");
        assert_eq!(label.color, Some([1.0, 0.5, 0.0, 1.0]));
    }

    #[test]
    fn test_debug_label_has_color() {
        let with_color = DebugLabel::with_color("Test", [1.0, 0.0, 0.0, 1.0]);
        let without_color = DebugLabel::new("Test");

        assert!(with_color.has_color());
        assert!(!without_color.has_color());
    }

    #[test]
    fn test_debug_label_rgb() {
        let label = DebugLabel::with_color("Test", [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(label.rgb(), Some([0.5, 0.6, 0.7]));

        let no_color = DebugLabel::new("Test");
        assert_eq!(no_color.rgb(), None);
    }

    #[test]
    fn test_debug_label_rgba_u8() {
        let label = DebugLabel::with_color("Test", [1.0, 0.5, 0.0, 0.5]);
        let rgba = label.rgba_u8().unwrap();
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 127);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 127);
    }

    #[test]
    fn test_debug_label_child() {
        let parent = DebugLabel::new("Parent");
        let child = parent.child("Child");
        assert_eq!(child.as_wgpu_label(), "Parent/Child");
    }

    #[test]
    fn test_debug_label_child_preserves_color() {
        let parent = DebugLabel::with_color("Parent", [1.0, 0.0, 0.0, 1.0]);
        let child = parent.child("Child");
        assert_eq!(child.color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_debug_label_from_str() {
        let label: DebugLabel = "From Str".into();
        assert_eq!(label.as_wgpu_label(), "From Str");
    }

    #[test]
    fn test_debug_label_from_string() {
        let label: DebugLabel = String::from("From String").into();
        assert_eq!(label.as_wgpu_label(), "From String");
    }

    #[test]
    fn test_debug_label_display() {
        let label = DebugLabel::new("Display Test");
        assert_eq!(format!("{}", label), "Display Test");
    }

    #[test]
    fn test_debug_label_default() {
        let label = DebugLabel::default();
        assert_eq!(label.as_wgpu_label(), "");
        assert!(label.color.is_none());
    }

    // -------------------------------------------------------------------------
    // DebugGroup Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_group_new() {
        let group = DebugGroup::new(DebugLabel::new("Test"));
        assert_eq!(group.label.as_wgpu_label(), "Test");
        assert_eq!(group.depth, 0);
        assert!(group.start_time.is_none());
    }

    #[test]
    fn test_debug_group_with_depth() {
        let group = DebugGroup::with_depth(DebugLabel::new("Test"), 3);
        assert_eq!(group.depth, 3);
        assert!(!group.has_profiling());
    }

    #[test]
    fn test_debug_group_with_profiling() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Test"));
        assert!(group.has_profiling());
        assert!(group.start_time.is_some());
    }

    #[test]
    fn test_debug_group_elapsed() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Test"));
        std::thread::sleep(std::time::Duration::from_millis(1));
        let elapsed = group.elapsed().unwrap();
        assert!(elapsed.as_micros() > 0);
    }

    #[test]
    fn test_debug_group_elapsed_ms() {
        let group = DebugGroup::with_profiling(DebugLabel::new("Test"));
        let ms = group.elapsed_ms().unwrap();
        assert!(ms >= 0.0);
    }

    #[test]
    fn test_debug_group_no_profiling_elapsed() {
        let group = DebugGroup::new(DebugLabel::new("Test"));
        assert!(group.elapsed().is_none());
        assert!(group.elapsed_ms().is_none());
    }

    // -------------------------------------------------------------------------
    // DebugMarkerStack Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_marker_stack_new() {
        let stack = DebugMarkerStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.current_depth(), 0);
        assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
    }

    #[test]
    fn test_marker_stack_push_pop() {
        let mut stack = DebugMarkerStack::new();

        assert!(stack.push_group(DebugLabel::new("First")));
        assert_eq!(stack.current_depth(), 1);
        assert!(!stack.is_empty());

        assert!(stack.push_group(DebugLabel::new("Second")));
        assert_eq!(stack.current_depth(), 2);

        let second = stack.pop_group().unwrap();
        assert_eq!(second.label.as_wgpu_label(), "Second");
        assert_eq!(second.depth, 1);

        let first = stack.pop_group().unwrap();
        assert_eq!(first.label.as_wgpu_label(), "First");
        assert_eq!(first.depth, 0);

        assert!(stack.is_empty());
    }

    #[test]
    fn test_marker_stack_max_depth() {
        let mut stack = DebugMarkerStack::with_max_depth(2);

        assert!(stack.push_group(DebugLabel::new("One")));
        assert!(stack.push_group(DebugLabel::new("Two")));
        assert!(!stack.push_group(DebugLabel::new("Three"))); // Should fail

        assert_eq!(stack.current_depth(), 2);
    }

    #[test]
    fn test_marker_stack_clear() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("A"));
        stack.push_group(DebugLabel::new("B"));

        stack.clear();

        assert!(stack.is_empty());
        assert_eq!(stack.current_depth(), 0);
    }

    #[test]
    fn test_marker_stack_current_group() {
        let mut stack = DebugMarkerStack::new();

        assert!(stack.current_group().is_none());

        stack.push_group(DebugLabel::new("Outer"));
        assert_eq!(
            stack.current_group().unwrap().label.as_wgpu_label(),
            "Outer"
        );

        stack.push_group(DebugLabel::new("Inner"));
        assert_eq!(
            stack.current_group().unwrap().label.as_wgpu_label(),
            "Inner"
        );
    }

    #[test]
    fn test_marker_stack_path() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("Shadows"));
        stack.push_group(DebugLabel::new("Cascade0"));
        stack.push_group(DebugLabel::new("Draw"));

        assert_eq!(stack.path(), "Shadows/Cascade0/Draw");
    }

    #[test]
    fn test_marker_stack_path_empty() {
        let stack = DebugMarkerStack::new();
        assert_eq!(stack.path(), "");
    }

    #[test]
    fn test_marker_stack_can_push() {
        let mut stack = DebugMarkerStack::with_max_depth(1);
        assert!(stack.can_push());

        stack.push_group(DebugLabel::new("Only"));
        assert!(!stack.can_push());
    }

    #[test]
    fn test_marker_stack_profiling() {
        let mut stack = DebugMarkerStack::with_profiling();
        assert!(stack.profiling_enabled());

        stack.push_group(DebugLabel::new("Profiled"));
        let group = stack.current_group().unwrap();
        assert!(group.has_profiling());
    }

    #[test]
    fn test_marker_stack_set_profiling() {
        let mut stack = DebugMarkerStack::new();
        assert!(!stack.profiling_enabled());

        stack.set_profiling(true);
        assert!(stack.profiling_enabled());

        stack.push_group(DebugLabel::new("Now Profiled"));
        assert!(stack.current_group().unwrap().has_profiling());
    }

    #[test]
    fn test_marker_stack_insert_marker() {
        let mut stack = DebugMarkerStack::new();

        stack.insert_marker(&DebugLabel::new("Marker1"));
        assert_eq!(stack.markers_at_depth(), 1);

        stack.insert_marker(&DebugLabel::new("Marker2"));
        assert_eq!(stack.markers_at_depth(), 2);

        stack.push_group(DebugLabel::new("Group"));
        assert_eq!(stack.markers_at_depth(), 0); // Reset on push

        stack.insert_marker(&DebugLabel::new("Inner"));
        assert_eq!(stack.markers_at_depth(), 1);
    }

    #[test]
    fn test_marker_stack_groups_slice() {
        let mut stack = DebugMarkerStack::new();
        stack.push_group(DebugLabel::new("A"));
        stack.push_group(DebugLabel::new("B"));

        let groups = stack.groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].label.as_wgpu_label(), "A");
        assert_eq!(groups[1].label.as_wgpu_label(), "B");
    }

    #[test]
    fn test_marker_stack_pop_empty() {
        let mut stack = DebugMarkerStack::new();
        assert!(stack.pop_group().is_none());
    }

    // -------------------------------------------------------------------------
    // Colors Module Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_predefined_colors() {
        // Just verify they're valid RGBA values
        let all_colors = [
            colors::GEOMETRY,
            colors::SHADOW,
            colors::LIGHTING,
            colors::POST_PROCESS,
            colors::COMPUTE,
            colors::UI,
            colors::DEBUG,
            colors::TRANSPARENT,
            colors::RAYTRACING,
            colors::TRANSFER,
        ];

        for color in all_colors {
            for component in color {
                assert!(
                    (0.0..=1.0).contains(&component),
                    "Color component {} out of range",
                    component
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // DebugLabel Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_label_empty_string() {
        let label = DebugLabel::new("");
        assert_eq!(label.as_wgpu_label(), "");
    }

    #[test]
    fn test_debug_label_very_long_string() {
        let long = "A".repeat(1000);
        let label = DebugLabel::new(long.clone());
        assert_eq!(label.as_wgpu_label(), long);
    }

    #[test]
    fn test_debug_label_unicode() {
        let label = DebugLabel::new("Unicode: ");
        assert_eq!(label.as_wgpu_label(), "Unicode: ");
    }

    #[test]
    fn test_debug_label_special_chars() {
        let label = DebugLabel::new("Special: /\\[]{}()");
        assert_eq!(label.as_wgpu_label(), "Special: /\\[]{}()");
    }

    #[test]
    fn test_debug_label_clone() {
        let original = DebugLabel::with_color("Clone Me", [0.5, 0.5, 0.5, 1.0]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_debug_group_clone() {
        let group = DebugGroup::with_depth_and_profiling(DebugLabel::new("Cloneable"), 5);
        let cloned = group.clone();
        assert_eq!(cloned.label.as_wgpu_label(), "Cloneable");
        assert_eq!(cloned.depth, 5);
        assert!(cloned.has_profiling());
    }

    // -------------------------------------------------------------------------
    // DebugMarkerStack Default Impl
    // -------------------------------------------------------------------------

    #[test]
    fn test_marker_stack_default() {
        let stack: DebugMarkerStack = Default::default();
        assert!(stack.is_empty());
        assert_eq!(stack.max_depth(), DebugMarkerStack::DEFAULT_MAX_DEPTH);
    }
}
