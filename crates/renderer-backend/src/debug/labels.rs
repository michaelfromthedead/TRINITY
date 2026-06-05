//! Debug Labels System for GPU Resources and Operations
//!
//! This module provides a comprehensive debug labeling system for GPU debugging,
//! including categorized labels, a label registry, and RAII scope guards.
//!
//! # Overview
//!
//! - [`DebugLabel`] - Label with optional color and category
//! - [`DebugCategory`] - Category classification for debug labels
//! - [`DebugScope`] - Scope builder for debug groups
//! - [`DebugScopeGuard`] - RAII guard for automatic push/pop
//! - [`LabelRegistry`] - Registry for managing debug labels
//! - [`DebugMarker`] - Helper for inserting debug markers
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::labels::*;
//!
//! // Create a registry and register labels
//! let mut registry = LabelRegistry::new();
//! let label = registry.register("Shadow Pass", DebugCategory::Pass);
//!
//! // Use scoped debug groups
//! let scope = DebugScope::new(encoder, label.clone());
//! let guard = scope.push();
//! // ... rendering code ...
//! // Group automatically popped when guard drops
//! ```

use std::collections::HashMap;

// ============================================================================
// DebugCategory
// ============================================================================

/// Category classification for debug labels.
///
/// Categories help organize and filter debug information by resource type
/// or operation kind.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::labels::DebugCategory;
///
/// let pass_category = DebugCategory::Pass;
/// let buffer_category = DebugCategory::Buffer;
/// let custom = DebugCategory::Custom(42);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DebugCategory {
    /// Render or compute pass
    Pass,
    /// GPU resource (generic)
    Resource,
    /// Pipeline (render or compute)
    Pipeline,
    /// Buffer resource
    Buffer,
    /// Texture resource
    Texture,
    /// Shader module
    Shader,
    /// Query (timestamp, occlusion, etc.)
    Query,
    /// Debug marker point
    Marker,
    /// User-defined custom category
    Custom(u32),
}

impl DebugCategory {
    /// Get a default color for this category.
    ///
    /// Returns an RGBA color array with values in 0.0 to 1.0 range.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let color = DebugCategory::Pass.default_color();
    /// assert_eq!(color[3], 1.0); // Full opacity
    /// ```
    #[inline]
    pub fn default_color(&self) -> [f32; 4] {
        match self {
            DebugCategory::Pass => [1.0, 0.6, 0.2, 1.0],       // Orange
            DebugCategory::Resource => [0.5, 0.8, 0.5, 1.0],   // Light green
            DebugCategory::Pipeline => [0.7, 0.3, 0.9, 1.0],   // Purple
            DebugCategory::Buffer => [0.2, 0.8, 0.9, 1.0],     // Cyan
            DebugCategory::Texture => [0.9, 0.9, 0.3, 1.0],    // Yellow
            DebugCategory::Shader => [0.9, 0.4, 0.6, 1.0],     // Pink
            DebugCategory::Query => [0.6, 0.6, 0.9, 1.0],      // Light blue
            DebugCategory::Marker => [1.0, 1.0, 1.0, 1.0],     // White
            DebugCategory::Custom(_) => [0.7, 0.7, 0.7, 1.0],  // Gray
        }
    }

    /// Get a human-readable name for this category.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(DebugCategory::Pass.name(), "Pass");
    /// assert_eq!(DebugCategory::Custom(5).name(), "Custom");
    /// ```
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            DebugCategory::Pass => "Pass",
            DebugCategory::Resource => "Resource",
            DebugCategory::Pipeline => "Pipeline",
            DebugCategory::Buffer => "Buffer",
            DebugCategory::Texture => "Texture",
            DebugCategory::Shader => "Shader",
            DebugCategory::Query => "Query",
            DebugCategory::Marker => "Marker",
            DebugCategory::Custom(_) => "Custom",
        }
    }

    /// Get the custom ID if this is a Custom category.
    ///
    /// # Returns
    ///
    /// `Some(id)` for Custom variants, `None` otherwise.
    #[inline]
    pub fn custom_id(&self) -> Option<u32> {
        match self {
            DebugCategory::Custom(id) => Some(*id),
            _ => None,
        }
    }

    /// Check if this is a resource-related category.
    #[inline]
    pub fn is_resource(&self) -> bool {
        matches!(
            self,
            DebugCategory::Resource
                | DebugCategory::Buffer
                | DebugCategory::Texture
                | DebugCategory::Shader
        )
    }

    /// All standard (non-custom) categories.
    pub const ALL_STANDARD: [DebugCategory; 8] = [
        DebugCategory::Pass,
        DebugCategory::Resource,
        DebugCategory::Pipeline,
        DebugCategory::Buffer,
        DebugCategory::Texture,
        DebugCategory::Shader,
        DebugCategory::Query,
        DebugCategory::Marker,
    ];
}

impl Default for DebugCategory {
    fn default() -> Self {
        DebugCategory::Marker
    }
}

impl std::fmt::Display for DebugCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DebugCategory::Custom(id) => write!(f, "Custom({})", id),
            _ => write!(f, "{}", self.name()),
        }
    }
}

// ============================================================================
// DebugLabel
// ============================================================================

/// A debug label with name, optional color, and category.
///
/// Debug labels are used to annotate GPU resources and operations for
/// debugging tools like RenderDoc, PIX, and Nsight Graphics.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::labels::{DebugLabel, DebugCategory};
///
/// // Simple label
/// let label = DebugLabel::new("GBuffer Pass", DebugCategory::Pass);
///
/// // Label with custom color
/// let colored = DebugLabel::with_color(
///     "Shadow Pass",
///     Some([0.2, 0.4, 0.8, 1.0]),
///     DebugCategory::Pass,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct DebugLabel {
    /// The label name
    pub name: String,
    /// Optional RGBA color for visualization (0.0 to 1.0 range)
    pub color: Option<[f32; 4]>,
    /// The category of this label
    pub category: DebugCategory,
}

impl DebugLabel {
    /// Create a new debug label with a name and category.
    ///
    /// # Arguments
    ///
    /// * `name` - The label text
    /// * `category` - The category classification
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = DebugLabel::new("My Pass", DebugCategory::Pass);
    /// assert_eq!(label.name, "My Pass");
    /// assert!(label.color.is_none());
    /// ```
    #[inline]
    pub fn new(name: impl Into<String>, category: DebugCategory) -> Self {
        Self {
            name: name.into(),
            color: None,
            category,
        }
    }

    /// Create a new debug label with name, color, and category.
    ///
    /// # Arguments
    ///
    /// * `name` - The label text
    /// * `color` - Optional RGBA color (0.0 to 1.0 range)
    /// * `category` - The category classification
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = DebugLabel::with_color(
    ///     "Shadow Pass",
    ///     Some([0.2, 0.4, 0.8, 1.0]),
    ///     DebugCategory::Pass,
    /// );
    /// ```
    #[inline]
    pub fn with_color(
        name: impl Into<String>,
        color: Option<[f32; 4]>,
        category: DebugCategory,
    ) -> Self {
        Self {
            name: name.into(),
            color,
            category,
        }
    }

    /// Create a marker label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The marker text
    #[inline]
    pub fn marker(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Marker)
    }

    /// Create a pass label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The pass name
    #[inline]
    pub fn pass(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Pass)
    }

    /// Create a buffer label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The buffer name
    #[inline]
    pub fn buffer(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Buffer)
    }

    /// Create a texture label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The texture name
    #[inline]
    pub fn texture(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Texture)
    }

    /// Create a pipeline label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The pipeline name
    #[inline]
    pub fn pipeline(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Pipeline)
    }

    /// Create a shader label (convenience constructor).
    ///
    /// # Arguments
    ///
    /// * `name` - The shader name
    #[inline]
    pub fn shader(name: impl Into<String>) -> Self {
        Self::new(name, DebugCategory::Shader)
    }

    /// Get the label as a wgpu-compatible label string.
    #[inline]
    pub fn as_wgpu_label(&self) -> &str {
        &self.name
    }

    /// Check if this label has a color set.
    #[inline]
    pub fn has_color(&self) -> bool {
        self.color.is_some()
    }

    /// Get the effective color (custom or category default).
    #[inline]
    pub fn effective_color(&self) -> [f32; 4] {
        self.color.unwrap_or_else(|| self.category.default_color())
    }

    /// Get the color as RGB (dropping alpha).
    #[inline]
    pub fn rgb(&self) -> [f32; 3] {
        let c = self.effective_color();
        [c[0], c[1], c[2]]
    }

    /// Get the color as 8-bit RGBA.
    #[inline]
    pub fn rgba_u8(&self) -> [u8; 4] {
        let c = self.effective_color();
        [
            (c[0] * 255.0).clamp(0.0, 255.0) as u8,
            (c[1] * 255.0).clamp(0.0, 255.0) as u8,
            (c[2] * 255.0).clamp(0.0, 255.0) as u8,
            (c[3] * 255.0).clamp(0.0, 255.0) as u8,
        ]
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
    /// let parent = DebugLabel::pass("Shadow Pass");
    /// let child = parent.child("Cascade 0");
    /// assert_eq!(child.name, "Shadow Pass/Cascade 0");
    /// ```
    #[inline]
    pub fn child(&self, suffix: &str) -> Self {
        Self {
            name: format!("{}/{}", self.name, suffix),
            color: self.color,
            category: self.category,
        }
    }

    /// Set the color and return self (builder pattern).
    #[inline]
    pub fn set_color(mut self, color: [f32; 4]) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the category and return self (builder pattern).
    #[inline]
    pub fn set_category(mut self, category: DebugCategory) -> Self {
        self.category = category;
        self
    }
}

impl PartialEq for DebugLabel {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.category == other.category && self.color == other.color
    }
}

impl std::fmt::Display for DebugLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.category, self.name)
    }
}

impl Default for DebugLabel {
    fn default() -> Self {
        Self {
            name: String::new(),
            color: None,
            category: DebugCategory::Marker,
        }
    }
}

// ============================================================================
// DebugScope
// ============================================================================

/// Scope builder for debug groups.
///
/// Creates a debug scope that can be pushed onto a command encoder.
/// Use [`push`](Self::push) to get a guard that automatically pops on drop.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::labels::{DebugScope, DebugLabel, DebugCategory};
///
/// let label = DebugLabel::new("My Pass", DebugCategory::Pass);
/// let scope = DebugScope::new(&mut encoder, label);
/// let _guard = scope.push();
/// // ... do work ...
/// // Group popped when _guard drops
/// ```
pub struct DebugScope<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    label: DebugLabel,
}

impl<'a> DebugScope<'a> {
    /// Create a new debug scope.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to push/pop on
    /// * `label` - The debug label for this scope
    #[inline]
    pub fn new(encoder: &'a mut wgpu::CommandEncoder, label: DebugLabel) -> Self {
        Self { encoder, label }
    }

    /// Push the debug group and return a guard that pops on drop.
    ///
    /// # Returns
    ///
    /// A `DebugScopeGuard` that calls `pop_debug_group` when dropped.
    #[inline]
    pub fn push(self) -> DebugScopeGuard<'a> {
        self.encoder.push_debug_group(self.label.as_wgpu_label());
        DebugScopeGuard {
            encoder: self.encoder,
            #[cfg(debug_assertions)]
            label_name: self.label.name.clone(),
        }
    }

    /// Get a reference to the label.
    #[inline]
    pub fn label(&self) -> &DebugLabel {
        &self.label
    }
}

// ============================================================================
// DebugScopeGuard
// ============================================================================

/// RAII guard that pops a debug group when dropped.
///
/// Created by [`DebugScope::push`]. Automatically calls `pop_debug_group`
/// on the encoder when it goes out of scope.
///
/// # Example
///
/// ```ignore
/// fn render_pass(encoder: &mut wgpu::CommandEncoder) {
///     let guard = DebugScope::new(encoder, DebugLabel::pass("My Pass")).push();
///     // ... rendering code ...
///     // Guard automatically pops when function returns
/// }
/// ```
pub struct DebugScopeGuard<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    #[cfg(debug_assertions)]
    label_name: String,
}

impl<'a> DebugScopeGuard<'a> {
    /// Insert a debug marker within this scope.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label
    #[inline]
    pub fn insert_marker(&mut self, label: &str) {
        self.encoder.insert_debug_marker(label);
    }

    /// Insert a debug marker with color.
    ///
    /// Note: wgpu's insert_debug_marker doesn't support color directly,
    /// but this records the intent for tools that support it.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label
    /// * `_color` - RGBA color (reserved for future use)
    #[inline]
    pub fn insert_marker_with_color(&mut self, label: &str, _color: [f32; 4]) {
        self.encoder.insert_debug_marker(label);
    }

    /// Get access to the underlying encoder.
    ///
    /// Use with care - do not call push/pop_debug_group directly.
    #[inline]
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder
    }

    /// Get the label name (debug builds only).
    #[cfg(debug_assertions)]
    #[inline]
    pub fn label_name(&self) -> &str {
        &self.label_name
    }
}

impl Drop for DebugScopeGuard<'_> {
    fn drop(&mut self) {
        self.encoder.pop_debug_group();
    }
}

// ============================================================================
// LabelRegistry
// ============================================================================

/// Registry for managing debug labels.
///
/// Provides centralized label management with unique IDs and category filtering.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::labels::{LabelRegistry, DebugCategory};
///
/// let mut registry = LabelRegistry::new();
///
/// // Register labels
/// let shadow_label = registry.register("Shadow Pass", DebugCategory::Pass);
/// let gbuffer_label = registry.register_with_color(
///     "GBuffer Pass",
///     DebugCategory::Pass,
///     [0.5, 0.8, 0.3, 1.0],
/// );
///
/// // Look up labels
/// if let Some(label) = registry.get("Shadow Pass") {
///     println!("Found: {}", label);
/// }
///
/// // Filter by category
/// let pass_labels = registry.labels_by_category(DebugCategory::Pass);
/// ```
#[derive(Debug)]
pub struct LabelRegistry {
    labels: HashMap<String, DebugLabel>,
    id_counter: u32,
}

impl LabelRegistry {
    /// Create a new empty label registry.
    #[inline]
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            id_counter: 0,
        }
    }

    /// Create a registry with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for the label map
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            labels: HashMap::with_capacity(capacity),
            id_counter: 0,
        }
    }

    /// Register a new label with the given name and category.
    ///
    /// If a label with the same name already exists, returns the existing label.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name
    /// * `category` - The category classification
    ///
    /// # Returns
    ///
    /// A reference to the registered (or existing) label.
    pub fn register(&mut self, name: &str, category: DebugCategory) -> &DebugLabel {
        self.id_counter = self.id_counter.wrapping_add(1);
        self.labels
            .entry(name.to_string())
            .or_insert_with(|| DebugLabel::new(name, category))
    }

    /// Register a new label with name, category, and color.
    ///
    /// If a label with the same name already exists, returns the existing label.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name
    /// * `category` - The category classification
    /// * `color` - RGBA color (0.0 to 1.0 range)
    ///
    /// # Returns
    ///
    /// A reference to the registered (or existing) label.
    pub fn register_with_color(
        &mut self,
        name: &str,
        category: DebugCategory,
        color: [f32; 4],
    ) -> &DebugLabel {
        self.id_counter = self.id_counter.wrapping_add(1);
        self.labels
            .entry(name.to_string())
            .or_insert_with(|| DebugLabel::with_color(name, Some(color), category))
    }

    /// Get a label by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name to look up
    ///
    /// # Returns
    ///
    /// `Some(&DebugLabel)` if found, `None` otherwise.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&DebugLabel> {
        self.labels.get(name)
    }

    /// Get a mutable reference to a label by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name to look up
    ///
    /// # Returns
    ///
    /// `Some(&mut DebugLabel)` if found, `None` otherwise.
    #[inline]
    pub fn get_mut(&mut self, name: &str) -> Option<&mut DebugLabel> {
        self.labels.get_mut(name)
    }

    /// Remove a label from the registry.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name to remove
    ///
    /// # Returns
    ///
    /// The removed label if it existed, `None` otherwise.
    #[inline]
    pub fn remove(&mut self, name: &str) -> Option<DebugLabel> {
        self.labels.remove(name)
    }

    /// Clear all labels from the registry.
    #[inline]
    pub fn clear(&mut self) {
        self.labels.clear();
        self.id_counter = 0;
    }

    /// Get all labels matching a category.
    ///
    /// # Arguments
    ///
    /// * `category` - The category to filter by
    ///
    /// # Returns
    ///
    /// A vector of references to matching labels.
    pub fn labels_by_category(&self, category: DebugCategory) -> Vec<&DebugLabel> {
        self.labels
            .values()
            .filter(|label| label.category == category)
            .collect()
    }

    /// Check if a label with the given name exists.
    ///
    /// # Arguments
    ///
    /// * `name` - The label name to check
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.labels.contains_key(name)
    }

    /// Get the number of registered labels.
    #[inline]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get the current ID counter value.
    #[inline]
    pub fn id_counter(&self) -> u32 {
        self.id_counter
    }

    /// Iterate over all labels.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &DebugLabel)> {
        self.labels.iter()
    }

    /// Get all label names.
    pub fn names(&self) -> Vec<&str> {
        self.labels.keys().map(|s| s.as_str()).collect()
    }

    /// Get all labels (consuming iterator).
    pub fn into_labels(self) -> impl Iterator<Item = DebugLabel> {
        self.labels.into_values()
    }
}

impl Default for LabelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LabelRegistry {
    fn clone(&self) -> Self {
        Self {
            labels: self.labels.clone(),
            id_counter: self.id_counter,
        }
    }
}

// ============================================================================
// DebugMarker Helper
// ============================================================================

/// Helper for inserting debug markers into command encoders.
///
/// Provides static methods for marker insertion without requiring
/// a scope or registry.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::debug::labels::DebugMarker;
///
/// // Simple marker
/// DebugMarker::insert(&mut encoder, "Checkpoint A");
///
/// // Marker with color (for tools that support it)
/// DebugMarker::insert_with_color(&mut encoder, "Important", [1.0, 0.0, 0.0, 1.0]);
///
/// // Manual scoping
/// DebugMarker::begin_scope(&mut encoder, "My Scope");
/// // ... do work ...
/// DebugMarker::end_scope(&mut encoder);
/// ```
pub struct DebugMarker;

impl DebugMarker {
    /// Insert a debug marker at the current position.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `label` - The marker label
    #[inline]
    pub fn insert(encoder: &mut wgpu::CommandEncoder, label: &str) {
        encoder.insert_debug_marker(label);
    }

    /// Insert a debug marker with color.
    ///
    /// Note: wgpu's insert_debug_marker doesn't support color directly,
    /// but this method records the intent. The color information can be
    /// used by custom debugging tools or stored for later reference.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `label` - The marker label
    /// * `_color` - RGBA color (reserved for future use)
    #[inline]
    pub fn insert_with_color(encoder: &mut wgpu::CommandEncoder, label: &str, _color: [f32; 4]) {
        encoder.insert_debug_marker(label);
    }

    /// Begin a debug scope (push a debug group).
    ///
    /// Must be paired with `end_scope`. Prefer `DebugScope::push` for
    /// RAII-style automatic cleanup.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `label` - The scope label
    #[inline]
    pub fn begin_scope(encoder: &mut wgpu::CommandEncoder, label: &str) {
        encoder.push_debug_group(label);
    }

    /// End a debug scope (pop a debug group).
    ///
    /// Must be paired with a prior `begin_scope`.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    #[inline]
    pub fn end_scope(encoder: &mut wgpu::CommandEncoder) {
        encoder.pop_debug_group();
    }

    /// Insert a marker with a formatted label.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder
    /// * `label` - The formatted label string
    #[inline]
    pub fn insert_formatted(encoder: &mut wgpu::CommandEncoder, label: String) {
        encoder.insert_debug_marker(&label);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DebugCategory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_category_variants() {
        let pass = DebugCategory::Pass;
        let resource = DebugCategory::Resource;
        let pipeline = DebugCategory::Pipeline;
        let buffer = DebugCategory::Buffer;
        let texture = DebugCategory::Texture;
        let shader = DebugCategory::Shader;
        let query = DebugCategory::Query;
        let marker = DebugCategory::Marker;
        let custom = DebugCategory::Custom(42);

        assert_eq!(pass.name(), "Pass");
        assert_eq!(resource.name(), "Resource");
        assert_eq!(pipeline.name(), "Pipeline");
        assert_eq!(buffer.name(), "Buffer");
        assert_eq!(texture.name(), "Texture");
        assert_eq!(shader.name(), "Shader");
        assert_eq!(query.name(), "Query");
        assert_eq!(marker.name(), "Marker");
        assert_eq!(custom.name(), "Custom");
    }

    #[test]
    fn test_debug_category_default_colors() {
        for category in DebugCategory::ALL_STANDARD.iter() {
            let color = category.default_color();
            assert!(color[0] >= 0.0 && color[0] <= 1.0);
            assert!(color[1] >= 0.0 && color[1] <= 1.0);
            assert!(color[2] >= 0.0 && color[2] <= 1.0);
            assert!(color[3] >= 0.0 && color[3] <= 1.0);
        }
    }

    #[test]
    fn test_debug_category_custom_id() {
        let custom = DebugCategory::Custom(123);
        assert_eq!(custom.custom_id(), Some(123));

        let pass = DebugCategory::Pass;
        assert_eq!(pass.custom_id(), None);
    }

    #[test]
    fn test_debug_category_is_resource() {
        assert!(!DebugCategory::Pass.is_resource());
        assert!(DebugCategory::Resource.is_resource());
        assert!(!DebugCategory::Pipeline.is_resource());
        assert!(DebugCategory::Buffer.is_resource());
        assert!(DebugCategory::Texture.is_resource());
        assert!(DebugCategory::Shader.is_resource());
        assert!(!DebugCategory::Query.is_resource());
        assert!(!DebugCategory::Marker.is_resource());
        assert!(!DebugCategory::Custom(0).is_resource());
    }

    #[test]
    fn test_debug_category_display() {
        assert_eq!(format!("{}", DebugCategory::Pass), "Pass");
        assert_eq!(format!("{}", DebugCategory::Custom(5)), "Custom(5)");
    }

    #[test]
    fn test_debug_category_default() {
        let default = DebugCategory::default();
        assert_eq!(default, DebugCategory::Marker);
    }

    #[test]
    fn test_debug_category_equality() {
        assert_eq!(DebugCategory::Pass, DebugCategory::Pass);
        assert_ne!(DebugCategory::Pass, DebugCategory::Buffer);
        assert_eq!(DebugCategory::Custom(10), DebugCategory::Custom(10));
        assert_ne!(DebugCategory::Custom(10), DebugCategory::Custom(20));
    }

    #[test]
    fn test_debug_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DebugCategory::Pass);
        set.insert(DebugCategory::Buffer);
        set.insert(DebugCategory::Custom(1));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&DebugCategory::Pass));
    }

    // -------------------------------------------------------------------------
    // DebugLabel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_label_new() {
        let label = DebugLabel::new("Test Label", DebugCategory::Pass);
        assert_eq!(label.name, "Test Label");
        assert!(label.color.is_none());
        assert_eq!(label.category, DebugCategory::Pass);
    }

    #[test]
    fn test_debug_label_with_color() {
        let color = [1.0, 0.5, 0.0, 1.0];
        let label = DebugLabel::with_color("Colored", Some(color), DebugCategory::Buffer);
        assert_eq!(label.name, "Colored");
        assert_eq!(label.color, Some(color));
        assert_eq!(label.category, DebugCategory::Buffer);
    }

    #[test]
    fn test_debug_label_convenience_constructors() {
        let marker = DebugLabel::marker("Marker");
        assert_eq!(marker.category, DebugCategory::Marker);

        let pass = DebugLabel::pass("Pass");
        assert_eq!(pass.category, DebugCategory::Pass);

        let buffer = DebugLabel::buffer("Buffer");
        assert_eq!(buffer.category, DebugCategory::Buffer);

        let texture = DebugLabel::texture("Texture");
        assert_eq!(texture.category, DebugCategory::Texture);

        let pipeline = DebugLabel::pipeline("Pipeline");
        assert_eq!(pipeline.category, DebugCategory::Pipeline);

        let shader = DebugLabel::shader("Shader");
        assert_eq!(shader.category, DebugCategory::Shader);
    }

    #[test]
    fn test_debug_label_as_wgpu_label() {
        let label = DebugLabel::new("Test", DebugCategory::Pass);
        assert_eq!(label.as_wgpu_label(), "Test");
    }

    #[test]
    fn test_debug_label_has_color() {
        let with_color = DebugLabel::with_color("Test", Some([1.0, 0.0, 0.0, 1.0]), DebugCategory::Pass);
        let without_color = DebugLabel::new("Test", DebugCategory::Pass);

        assert!(with_color.has_color());
        assert!(!without_color.has_color());
    }

    #[test]
    fn test_debug_label_effective_color() {
        let with_color = DebugLabel::with_color("Test", Some([0.5, 0.5, 0.5, 1.0]), DebugCategory::Pass);
        assert_eq!(with_color.effective_color(), [0.5, 0.5, 0.5, 1.0]);

        let without_color = DebugLabel::new("Test", DebugCategory::Pass);
        assert_eq!(without_color.effective_color(), DebugCategory::Pass.default_color());
    }

    #[test]
    fn test_debug_label_rgb() {
        let label = DebugLabel::with_color("Test", Some([0.5, 0.6, 0.7, 0.8]), DebugCategory::Pass);
        assert_eq!(label.rgb(), [0.5, 0.6, 0.7]);
    }

    #[test]
    fn test_debug_label_rgba_u8() {
        let label = DebugLabel::with_color("Test", Some([1.0, 0.5, 0.0, 0.5]), DebugCategory::Pass);
        let rgba = label.rgba_u8();
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 127);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 127);
    }

    #[test]
    fn test_debug_label_child() {
        let parent = DebugLabel::pass("Parent");
        let child = parent.child("Child");
        assert_eq!(child.name, "Parent/Child");
        assert_eq!(child.category, DebugCategory::Pass);
    }

    #[test]
    fn test_debug_label_child_preserves_color() {
        let parent = DebugLabel::with_color("Parent", Some([1.0, 0.0, 0.0, 1.0]), DebugCategory::Pass);
        let child = parent.child("Child");
        assert_eq!(child.color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_debug_label_builder_pattern() {
        let label = DebugLabel::new("Test", DebugCategory::Marker)
            .set_color([1.0, 0.0, 0.0, 1.0])
            .set_category(DebugCategory::Pass);

        assert_eq!(label.color, Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(label.category, DebugCategory::Pass);
    }

    #[test]
    fn test_debug_label_display() {
        let label = DebugLabel::new("Shadow Pass", DebugCategory::Pass);
        assert_eq!(format!("{}", label), "[Pass] Shadow Pass");
    }

    #[test]
    fn test_debug_label_default() {
        let label = DebugLabel::default();
        assert_eq!(label.name, "");
        assert!(label.color.is_none());
        assert_eq!(label.category, DebugCategory::Marker);
    }

    #[test]
    fn test_debug_label_equality() {
        let a = DebugLabel::new("Test", DebugCategory::Pass);
        let b = DebugLabel::new("Test", DebugCategory::Pass);
        let c = DebugLabel::new("Test", DebugCategory::Buffer);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -------------------------------------------------------------------------
    // LabelRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_registry_new() {
        let registry = LabelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.id_counter(), 0);
    }

    #[test]
    fn test_label_registry_with_capacity() {
        let registry = LabelRegistry::with_capacity(100);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_label_registry_register() {
        let mut registry = LabelRegistry::new();
        let label = registry.register("Shadow Pass", DebugCategory::Pass);

        assert_eq!(label.name, "Shadow Pass");
        assert_eq!(label.category, DebugCategory::Pass);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_label_registry_register_with_color() {
        let mut registry = LabelRegistry::new();
        let color = [0.5, 0.5, 0.5, 1.0];
        let label = registry.register_with_color("GBuffer", DebugCategory::Pass, color);

        assert_eq!(label.name, "GBuffer");
        assert_eq!(label.color, Some(color));
    }

    #[test]
    fn test_label_registry_register_duplicate() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);
        let second = registry.register("Test", DebugCategory::Buffer);

        // Should return existing label
        assert_eq!(second.category, DebugCategory::Pass);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_label_registry_get() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);

        assert!(registry.get("Test").is_some());
        assert!(registry.get("Nonexistent").is_none());
    }

    #[test]
    fn test_label_registry_get_mut() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);

        if let Some(label) = registry.get_mut("Test") {
            label.color = Some([1.0, 0.0, 0.0, 1.0]);
        }

        assert_eq!(registry.get("Test").unwrap().color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_label_registry_remove() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);

        let removed = registry.remove("Test");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "Test");
        assert!(registry.is_empty());

        let removed_again = registry.remove("Test");
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_label_registry_clear() {
        let mut registry = LabelRegistry::new();
        registry.register("A", DebugCategory::Pass);
        registry.register("B", DebugCategory::Buffer);

        registry.clear();
        assert!(registry.is_empty());
        assert_eq!(registry.id_counter(), 0);
    }

    #[test]
    fn test_label_registry_labels_by_category() {
        let mut registry = LabelRegistry::new();
        registry.register("Pass A", DebugCategory::Pass);
        registry.register("Pass B", DebugCategory::Pass);
        registry.register("Buffer A", DebugCategory::Buffer);

        let passes = registry.labels_by_category(DebugCategory::Pass);
        assert_eq!(passes.len(), 2);

        let buffers = registry.labels_by_category(DebugCategory::Buffer);
        assert_eq!(buffers.len(), 1);

        let textures = registry.labels_by_category(DebugCategory::Texture);
        assert_eq!(textures.len(), 0);
    }

    #[test]
    fn test_label_registry_contains() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);

        assert!(registry.contains("Test"));
        assert!(!registry.contains("Other"));
    }

    #[test]
    fn test_label_registry_iter() {
        let mut registry = LabelRegistry::new();
        registry.register("A", DebugCategory::Pass);
        registry.register("B", DebugCategory::Buffer);

        let count = registry.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_label_registry_names() {
        let mut registry = LabelRegistry::new();
        registry.register("Alpha", DebugCategory::Pass);
        registry.register("Beta", DebugCategory::Buffer);

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Alpha"));
        assert!(names.contains(&"Beta"));
    }

    #[test]
    fn test_label_registry_clone() {
        let mut registry = LabelRegistry::new();
        registry.register("Test", DebugCategory::Pass);

        let cloned = registry.clone();
        assert_eq!(cloned.len(), 1);
        assert!(cloned.get("Test").is_some());
    }

    #[test]
    fn test_label_registry_id_counter_increments() {
        let mut registry = LabelRegistry::new();
        assert_eq!(registry.id_counter(), 0);

        registry.register("A", DebugCategory::Pass);
        assert_eq!(registry.id_counter(), 1);

        registry.register("B", DebugCategory::Buffer);
        assert_eq!(registry.id_counter(), 2);
    }

    // -------------------------------------------------------------------------
    // DebugMarker Tests (unit tests without actual wgpu)
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_marker_struct_exists() {
        // DebugMarker is a unit struct with no fields
        let _ = DebugMarker;
    }
}
