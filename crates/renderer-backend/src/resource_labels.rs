//! Resource Labels for wgpu 22.x/25.x
//!
//! This module provides utilities for generating, validating, and managing
//! resource labels in wgpu applications. Labels are used by GPU debugging tools
//! like RenderDoc, PIX, and Nsight Graphics to identify resources.
//!
//! # Overview
//!
//! - [`ResourceLabel`] - Hierarchical label with parent/child naming support
//! - [`LabelBuilder`] - Builder pattern for complex label construction
//! - [`LabelRegistry`] - Registry for tracking active labels
//! - [`prefixes`] - Common label prefixes for different resource types
//! - Convenience functions for common label patterns
//!
//! # Architecture
//!
//! ```text
//! ResourceLabel
//!     |-- segments: Vec<String>  (e.g., ["pass", "subpass", "draw"])
//!     |-- as_str() -> "pass/subpass/draw"
//!     |-- child(name) -> ResourceLabel with appended segment
//!     `-- validate() -> Result<(), LabelError>
//!
//! LabelBuilder
//!     |-- prefix: Option<String>
//!     |-- name: String
//!     |-- suffix: Option<String>
//!     |-- index: Option<u32>
//!     `-- build() -> "prefix_name_suffix_42"
//!
//! LabelRegistry
//!     |-- labels: HashSet<String>
//!     |-- register(label) -> bool (true if new)
//!     |-- unregister(label) -> bool (true if existed)
//!     `-- contains(label) -> bool
//! ```
//!
//! # wgpu Label Requirements
//!
//! wgpu labels should:
//! - Be valid UTF-8 strings
//! - Avoid excessively long labels (some tools truncate at 256 chars)
//! - Use alphanumeric characters, underscores, and slashes for best compatibility
//!
//! # Example: Basic Usage
//!
//! ```ignore
//! use renderer_backend::resource_labels::{ResourceLabel, LabelBuilder, buffer_label};
//!
//! // Simple label
//! let label = buffer_label("vertex_data");
//! // Result: "buffer/vertex_data"
//!
//! // Hierarchical label
//! let pass = ResourceLabel::new("shadow_pass");
//! let cascade = pass.child("cascade_0");
//! let draw = cascade.child("draw");
//! // Result: "shadow_pass/cascade_0/draw"
//!
//! // Builder pattern
//! let label = LabelBuilder::new("texture")
//!     .with_prefix("diffuse")
//!     .with_index(0)
//!     .build();
//! // Result: "diffuse_texture_0"
//! ```
//!
//! # Example: Label Registry
//!
//! ```ignore
//! use renderer_backend::resource_labels::LabelRegistry;
//!
//! let mut registry = LabelRegistry::new();
//!
//! // Register labels
//! assert!(registry.register("buffer/vertex_0"));  // true - new
//! assert!(!registry.register("buffer/vertex_0")); // false - duplicate
//!
//! // Check existence
//! assert!(registry.contains("buffer/vertex_0"));
//!
//! // Unregister
//! assert!(registry.unregister("buffer/vertex_0")); // true - existed
//! assert!(!registry.unregister("buffer/vertex_0")); // false - already gone
//! ```
//!
//! # Thread Safety
//!
//! - [`ResourceLabel`] and [`LabelBuilder`] are `Send + Sync`
//! - [`LabelRegistry`] is `!Sync` (requires external synchronization for concurrent access)
//!
//! # Performance
//!
//! Label operations are lightweight string operations. For performance-critical
//! code paths, consider caching generated labels rather than regenerating them.

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Maximum recommended label length for compatibility with debugging tools.
/// Some tools truncate labels beyond this length.
pub const MAX_LABEL_LENGTH: usize = 256;

/// Separator character used in hierarchical labels.
pub const LABEL_SEPARATOR: char = '/';

/// Separator character used in label builder between components.
pub const COMPONENT_SEPARATOR: char = '_';

// ============================================================================
// LabelError
// ============================================================================

/// Error type for label validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelError {
    /// Label is empty.
    Empty,
    /// Label exceeds maximum recommended length.
    TooLong {
        /// Actual length of the label.
        length: usize,
        /// Maximum allowed length.
        max: usize,
    },
    /// Label contains invalid characters.
    InvalidCharacter {
        /// The invalid character found.
        character: char,
        /// Position of the invalid character.
        position: usize,
    },
    /// Label segment is empty (e.g., "pass//subpass").
    EmptySegment {
        /// Position of the empty segment.
        position: usize,
    },
}

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LabelError::Empty => write!(f, "Label cannot be empty"),
            LabelError::TooLong { length, max } => {
                write!(f, "Label length {} exceeds maximum {}", length, max)
            }
            LabelError::InvalidCharacter { character, position } => {
                write!(
                    f,
                    "Invalid character '{}' at position {}",
                    character, position
                )
            }
            LabelError::EmptySegment { position } => {
                write!(f, "Empty segment at position {}", position)
            }
        }
    }
}

impl std::error::Error for LabelError {}

// ============================================================================
// ResourceLabel - Hierarchical Label (T-WGPU-P4.5.3 Criterion 3)
// ============================================================================

/// A resource label with hierarchical naming support.
///
/// Resource labels can be organized hierarchically using segments separated
/// by slashes. This allows for logical grouping in GPU debugging tools.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_labels::ResourceLabel;
///
/// let pass = ResourceLabel::new("render_pass");
/// let opaque = pass.child("opaque");
/// let draw_call = opaque.child("draw_0");
///
/// assert_eq!(draw_call.as_str(), "render_pass/opaque/draw_0");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceLabel {
    /// The segments of the hierarchical label.
    segments: Vec<String>,
    /// Cached full label string (lazily computed).
    cached: String,
}

impl ResourceLabel {
    /// Create a new resource label with a single segment.
    ///
    /// # Arguments
    ///
    /// * `name` - The initial label segment
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::new("my_buffer");
    /// assert_eq!(label.as_str(), "my_buffer");
    /// ```
    #[inline]
    pub fn new(name: &str) -> Self {
        Self {
            segments: vec![name.to_string()],
            cached: name.to_string(),
        }
    }

    /// Create a new resource label from multiple segments.
    ///
    /// # Arguments
    ///
    /// * `segments` - Iterator of segment strings
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::from_segments(["pass", "subpass", "draw"]);
    /// assert_eq!(label.as_str(), "pass/subpass/draw");
    /// ```
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let segments: Vec<String> = segments.into_iter().map(|s| s.as_ref().to_string()).collect();
        let cached = segments.join(&LABEL_SEPARATOR.to_string());
        Self { segments, cached }
    }

    /// Create an empty resource label.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::empty();
    /// assert!(label.is_empty());
    /// ```
    #[inline]
    pub fn empty() -> Self {
        Self {
            segments: Vec::new(),
            cached: String::new(),
        }
    }

    /// Create a child label by appending a new segment.
    ///
    /// This creates a new `ResourceLabel` with the given name appended
    /// to the current hierarchy.
    ///
    /// # Arguments
    ///
    /// * `name` - The child segment name
    ///
    /// # Returns
    ///
    /// A new `ResourceLabel` with the child segment appended.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let parent = ResourceLabel::new("pass");
    /// let child = parent.child("subpass");
    /// assert_eq!(child.as_str(), "pass/subpass");
    /// ```
    pub fn child(&self, name: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(name.to_string());
        let cached = if self.cached.is_empty() {
            name.to_string()
        } else {
            format!("{}{}{}", self.cached, LABEL_SEPARATOR, name)
        };
        Self { segments, cached }
    }

    /// Get the label as a string slice.
    ///
    /// # Returns
    ///
    /// The full hierarchical label with segments separated by slashes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::new("buffer").child("vertex");
    /// assert_eq!(label.as_str(), "buffer/vertex");
    /// ```
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.cached
    }

    /// Get the number of segments in the label.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::new("a").child("b").child("c");
    /// assert_eq!(label.depth(), 3);
    /// ```
    #[inline]
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// Check if the label is empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let empty = ResourceLabel::empty();
    /// assert!(empty.is_empty());
    ///
    /// let not_empty = ResourceLabel::new("test");
    /// assert!(!not_empty.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get the segments of the label.
    ///
    /// # Returns
    ///
    /// A slice of the label segments.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::from_segments(["a", "b", "c"]);
    /// assert_eq!(label.segments(), &["a", "b", "c"]);
    /// ```
    #[inline]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Get the last segment of the label (the "leaf" name).
    ///
    /// # Returns
    ///
    /// The last segment, or `None` if the label is empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::from_segments(["pass", "draw"]);
    /// assert_eq!(label.leaf(), Some("draw"));
    /// ```
    #[inline]
    pub fn leaf(&self) -> Option<&str> {
        self.segments.last().map(|s| s.as_str())
    }

    /// Get the parent label (all segments except the last).
    ///
    /// # Returns
    ///
    /// A new `ResourceLabel` with the last segment removed, or `None` if
    /// the label has only one segment or is empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::from_segments(["a", "b", "c"]);
    /// let parent = label.parent().unwrap();
    /// assert_eq!(parent.as_str(), "a/b");
    /// ```
    pub fn parent(&self) -> Option<Self> {
        if self.segments.len() <= 1 {
            None
        } else {
            Some(Self::from_segments(&self.segments[..self.segments.len() - 1]))
        }
    }

    /// Validate the label against wgpu requirements.
    ///
    /// Checks that:
    /// - Label is not empty
    /// - Label does not exceed maximum length
    /// - No empty segments exist
    /// - Characters are valid (alphanumeric, underscore, hyphen, slash, dot)
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(LabelError)` with details if invalid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let valid = ResourceLabel::new("my_buffer");
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = ResourceLabel::from_segments(["a", "", "b"]);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), LabelError> {
        // Check empty
        if self.is_empty() || self.cached.is_empty() {
            return Err(LabelError::Empty);
        }

        // Check length
        if self.cached.len() > MAX_LABEL_LENGTH {
            return Err(LabelError::TooLong {
                length: self.cached.len(),
                max: MAX_LABEL_LENGTH,
            });
        }

        // Check for empty segments
        for (i, segment) in self.segments.iter().enumerate() {
            if segment.is_empty() {
                return Err(LabelError::EmptySegment { position: i });
            }
        }

        // Check characters
        for (i, c) in self.cached.chars().enumerate() {
            if !is_valid_label_char(c) {
                return Err(LabelError::InvalidCharacter {
                    character: c,
                    position: i,
                });
            }
        }

        Ok(())
    }

    /// Validate the label, returning true if valid.
    ///
    /// This is a convenience wrapper around `validate()` for use in
    /// boolean contexts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = ResourceLabel::new("valid_name");
    /// assert!(label.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Join this label with another using a separator.
    ///
    /// # Arguments
    ///
    /// * `other` - The other label to join with
    ///
    /// # Returns
    ///
    /// A new label combining both labels' segments.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = ResourceLabel::new("pass");
    /// let b = ResourceLabel::from_segments(["subpass", "draw"]);
    /// let combined = a.join(&b);
    /// assert_eq!(combined.as_str(), "pass/subpass/draw");
    /// ```
    pub fn join(&self, other: &Self) -> Self {
        let mut segments = self.segments.clone();
        segments.extend(other.segments.iter().cloned());
        Self::from_segments(segments)
    }
}

impl Default for ResourceLabel {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for ResourceLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.cached)
    }
}

impl AsRef<str> for ResourceLabel {
    fn as_ref(&self) -> &str {
        &self.cached
    }
}

impl From<&str> for ResourceLabel {
    fn from(s: &str) -> Self {
        // Parse as hierarchical if it contains separator
        if s.contains(LABEL_SEPARATOR) {
            Self::from_segments(s.split(LABEL_SEPARATOR))
        } else {
            Self::new(s)
        }
    }
}

impl From<String> for ResourceLabel {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

// ============================================================================
// LabelBuilder - Builder Pattern for Complex Labels (T-WGPU-P4.5.3 Criterion 1)
// ============================================================================

/// Builder for constructing complex resource labels.
///
/// Provides a fluent API for building labels with optional prefix, suffix,
/// and index components.
///
/// # Format
///
/// The generated label has the format: `[prefix_]name[_suffix][_index]`
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_labels::LabelBuilder;
///
/// // Simple label
/// let label = LabelBuilder::new("texture").build();
/// assert_eq!(label, "texture");
///
/// // Full options
/// let label = LabelBuilder::new("buffer")
///     .with_prefix("vertex")
///     .with_suffix("staging")
///     .with_index(0)
///     .build();
/// assert_eq!(label, "vertex_buffer_staging_0");
/// ```
#[derive(Debug, Clone)]
pub struct LabelBuilder {
    /// Optional prefix component.
    prefix: Option<String>,
    /// Required name component.
    name: String,
    /// Optional suffix component.
    suffix: Option<String>,
    /// Optional index component.
    index: Option<u32>,
    /// Custom separator (defaults to underscore).
    separator: char,
}

impl LabelBuilder {
    /// Create a new label builder with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The main name component of the label
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = LabelBuilder::new("my_resource");
    /// ```
    #[inline]
    pub fn new(name: &str) -> Self {
        Self {
            prefix: None,
            name: name.to_string(),
            suffix: None,
            index: None,
            separator: COMPONENT_SEPARATOR,
        }
    }

    /// Add a prefix to the label.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix string
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("buffer")
    ///     .with_prefix("vertex")
    ///     .build();
    /// assert_eq!(label, "vertex_buffer");
    /// ```
    #[inline]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_string());
        self
    }

    /// Add a suffix to the label.
    ///
    /// # Arguments
    ///
    /// * `suffix` - The suffix string
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("texture")
    ///     .with_suffix("albedo")
    ///     .build();
    /// assert_eq!(label, "texture_albedo");
    /// ```
    #[inline]
    pub fn with_suffix(mut self, suffix: &str) -> Self {
        self.suffix = Some(suffix.to_string());
        self
    }

    /// Add an index to the label.
    ///
    /// # Arguments
    ///
    /// * `index` - The index number
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("cascade")
    ///     .with_index(2)
    ///     .build();
    /// assert_eq!(label, "cascade_2");
    /// ```
    #[inline]
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = Some(index);
        self
    }

    /// Set a custom separator character.
    ///
    /// The default separator is underscore (`_`).
    ///
    /// # Arguments
    ///
    /// * `separator` - The separator character
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("buffer")
    ///     .with_prefix("vertex")
    ///     .with_separator('-')
    ///     .build();
    /// assert_eq!(label, "vertex-buffer");
    /// ```
    #[inline]
    pub fn with_separator(mut self, separator: char) -> Self {
        self.separator = separator;
        self
    }

    /// Build the label string.
    ///
    /// # Returns
    ///
    /// The constructed label string.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("texture")
    ///     .with_prefix("diffuse")
    ///     .with_index(0)
    ///     .build();
    /// assert_eq!(label, "diffuse_texture_0");
    /// ```
    pub fn build(self) -> String {
        let mut parts = Vec::with_capacity(4);

        if let Some(prefix) = &self.prefix {
            parts.push(prefix.clone());
        }

        parts.push(self.name);

        if let Some(suffix) = &self.suffix {
            parts.push(suffix.clone());
        }

        if let Some(index) = self.index {
            parts.push(index.to_string());
        }

        parts.join(&self.separator.to_string())
    }

    /// Build as a ResourceLabel.
    ///
    /// # Returns
    ///
    /// A `ResourceLabel` containing the built label.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let label = LabelBuilder::new("buffer")
    ///     .with_index(0)
    ///     .build_resource_label();
    /// assert_eq!(label.as_str(), "buffer_0");
    /// ```
    #[inline]
    pub fn build_resource_label(self) -> ResourceLabel {
        ResourceLabel::new(&self.build())
    }

    /// Validate the builder configuration.
    ///
    /// Checks that the name is not empty.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(LabelError)` if invalid.
    pub fn validate(&self) -> Result<(), LabelError> {
        if self.name.is_empty() {
            return Err(LabelError::Empty);
        }
        Ok(())
    }
}

impl Default for LabelBuilder {
    fn default() -> Self {
        Self::new("")
    }
}

// ============================================================================
// LabelRegistry - Track Active Labels (T-WGPU-P4.5.3 Criterion 4)
// ============================================================================

/// Registry for tracking active resource labels.
///
/// Useful for debugging resource leaks, ensuring unique labels, and
/// tracking which resources are currently in use.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_labels::LabelRegistry;
///
/// let mut registry = LabelRegistry::new();
///
/// // Register resources
/// assert!(registry.register("buffer/vertex_0"));
/// assert!(registry.register("buffer/index_0"));
///
/// // Check duplicates
/// assert!(!registry.register("buffer/vertex_0")); // Already exists
///
/// // Query
/// assert!(registry.contains("buffer/vertex_0"));
/// assert_eq!(registry.count(), 2);
///
/// // Cleanup
/// registry.unregister("buffer/vertex_0");
/// assert!(!registry.contains("buffer/vertex_0"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct LabelRegistry {
    /// Set of registered labels.
    labels: HashSet<String>,
}

impl LabelRegistry {
    /// Create a new empty label registry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = LabelRegistry::new();
    /// assert_eq!(registry.count(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            labels: HashSet::new(),
        }
    }

    /// Create a registry with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for the label set
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = LabelRegistry::with_capacity(100);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            labels: HashSet::with_capacity(capacity),
        }
    }

    /// Register a label in the registry.
    ///
    /// # Arguments
    ///
    /// * `label` - The label to register
    ///
    /// # Returns
    ///
    /// `true` if the label was newly registered, `false` if it already existed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// assert!(registry.register("new_label"));   // true - new
    /// assert!(!registry.register("new_label"));  // false - duplicate
    /// ```
    #[inline]
    pub fn register(&mut self, label: &str) -> bool {
        self.labels.insert(label.to_string())
    }

    /// Register a ResourceLabel in the registry.
    ///
    /// # Arguments
    ///
    /// * `label` - The ResourceLabel to register
    ///
    /// # Returns
    ///
    /// `true` if the label was newly registered, `false` if it already existed.
    #[inline]
    pub fn register_resource_label(&mut self, label: &ResourceLabel) -> bool {
        self.labels.insert(label.as_str().to_string())
    }

    /// Unregister a label from the registry.
    ///
    /// # Arguments
    ///
    /// * `label` - The label to unregister
    ///
    /// # Returns
    ///
    /// `true` if the label was removed, `false` if it didn't exist.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("my_label");
    /// assert!(registry.unregister("my_label"));  // true - existed
    /// assert!(!registry.unregister("my_label")); // false - already gone
    /// ```
    #[inline]
    pub fn unregister(&mut self, label: &str) -> bool {
        self.labels.remove(label)
    }

    /// Check if a label is registered.
    ///
    /// # Arguments
    ///
    /// * `label` - The label to check
    ///
    /// # Returns
    ///
    /// `true` if the label is registered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("exists");
    /// assert!(registry.contains("exists"));
    /// assert!(!registry.contains("not_exists"));
    /// ```
    #[inline]
    pub fn contains(&self, label: &str) -> bool {
        self.labels.contains(label)
    }

    /// Get the number of registered labels.
    ///
    /// # Returns
    ///
    /// The count of registered labels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("a");
    /// registry.register("b");
    /// assert_eq!(registry.count(), 2);
    /// ```
    #[inline]
    pub fn count(&self) -> usize {
        self.labels.len()
    }

    /// Check if the registry is empty.
    ///
    /// # Returns
    ///
    /// `true` if no labels are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Clear all registered labels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("a");
    /// registry.register("b");
    /// registry.clear();
    /// assert_eq!(registry.count(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.labels.clear();
    }

    /// Get an iterator over all registered labels.
    ///
    /// # Returns
    ///
    /// An iterator over label strings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("a");
    /// registry.register("b");
    /// for label in registry.iter() {
    ///     println!("Label: {}", label);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.labels.iter()
    }

    /// Get all labels matching a prefix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix to match
    ///
    /// # Returns
    ///
    /// A vector of labels starting with the prefix.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("buffer/vertex_0");
    /// registry.register("buffer/index_0");
    /// registry.register("texture/albedo_0");
    ///
    /// let buffers = registry.with_prefix("buffer/");
    /// assert_eq!(buffers.len(), 2);
    /// ```
    pub fn with_prefix(&self, prefix: &str) -> Vec<&String> {
        self.labels
            .iter()
            .filter(|l| l.starts_with(prefix))
            .collect()
    }

    /// Generate a unique label based on a base name.
    ///
    /// If the base name is not registered, returns it as-is.
    /// Otherwise, appends incrementing numbers until a unique name is found.
    ///
    /// # Arguments
    ///
    /// * `base` - The base label name
    ///
    /// # Returns
    ///
    /// A unique label string.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut registry = LabelRegistry::new();
    /// registry.register("buffer");
    /// registry.register("buffer_1");
    ///
    /// let unique = registry.generate_unique("buffer");
    /// assert_eq!(unique, "buffer_2");
    /// ```
    pub fn generate_unique(&self, base: &str) -> String {
        if !self.contains(base) {
            return base.to_string();
        }

        let mut counter = 1u32;
        loop {
            let candidate = format!("{}_{}", base, counter);
            if !self.contains(&candidate) {
                return candidate;
            }
            counter += 1;
            // Safety limit to prevent infinite loop
            if counter > 100_000 {
                return format!("{}_{}", base, std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0));
            }
        }
    }
}

// ============================================================================
// Common Label Prefixes (T-WGPU-P4.5.3 Criterion 5)
// ============================================================================

/// Common label prefixes for different resource types.
///
/// Using consistent prefixes helps organize resources in GPU debugging tools.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_labels::prefixes;
///
/// let label = format!("{}/my_texture", prefixes::TEXTURE);
/// assert_eq!(label, "texture/my_texture");
/// ```
pub mod prefixes {
    /// Prefix for buffer resources.
    pub const BUFFER: &str = "buffer";

    /// Prefix for texture resources.
    pub const TEXTURE: &str = "texture";

    /// Prefix for sampler resources.
    pub const SAMPLER: &str = "sampler";

    /// Prefix for pipeline resources.
    pub const PIPELINE: &str = "pipeline";

    /// Prefix for bind group resources.
    pub const BIND_GROUP: &str = "bind_group";

    /// Prefix for render pass resources.
    pub const RENDER_PASS: &str = "render_pass";

    /// Prefix for compute pass resources.
    pub const COMPUTE_PASS: &str = "compute_pass";

    /// Prefix for shader module resources.
    pub const SHADER: &str = "shader";

    /// Prefix for query set resources.
    pub const QUERY: &str = "query";

    /// Prefix for bind group layout resources.
    pub const BIND_GROUP_LAYOUT: &str = "bind_group_layout";

    /// Prefix for pipeline layout resources.
    pub const PIPELINE_LAYOUT: &str = "pipeline_layout";

    /// Prefix for command encoder resources.
    pub const COMMAND_ENCODER: &str = "command_encoder";

    /// Prefix for render bundle resources.
    pub const RENDER_BUNDLE: &str = "render_bundle";
}

// ============================================================================
// Label Generation Functions (T-WGPU-P4.5.3 Criterion 1)
// ============================================================================

/// Generate a buffer label.
///
/// # Arguments
///
/// * `name` - The buffer name
///
/// # Returns
///
/// A label in the format "buffer/name".
///
/// # Example
///
/// ```ignore
/// let label = buffer_label("vertex_data");
/// assert_eq!(label, "buffer/vertex_data");
/// ```
#[inline]
pub fn buffer_label(name: &str) -> String {
    format!("{}{}{}", prefixes::BUFFER, LABEL_SEPARATOR, name)
}

/// Generate a texture label.
///
/// # Arguments
///
/// * `name` - The texture name
///
/// # Returns
///
/// A label in the format "texture/name".
///
/// # Example
///
/// ```ignore
/// let label = texture_label("diffuse");
/// assert_eq!(label, "texture/diffuse");
/// ```
#[inline]
pub fn texture_label(name: &str) -> String {
    format!("{}{}{}", prefixes::TEXTURE, LABEL_SEPARATOR, name)
}

/// Generate a sampler label.
///
/// # Arguments
///
/// * `name` - The sampler name
///
/// # Returns
///
/// A label in the format "sampler/name".
///
/// # Example
///
/// ```ignore
/// let label = sampler_label("linear_clamp");
/// assert_eq!(label, "sampler/linear_clamp");
/// ```
#[inline]
pub fn sampler_label(name: &str) -> String {
    format!("{}{}{}", prefixes::SAMPLER, LABEL_SEPARATOR, name)
}

/// Generate a pipeline label.
///
/// # Arguments
///
/// * `name` - The pipeline name
///
/// # Returns
///
/// A label in the format "pipeline/name".
///
/// # Example
///
/// ```ignore
/// let label = pipeline_label("gbuffer");
/// assert_eq!(label, "pipeline/gbuffer");
/// ```
#[inline]
pub fn pipeline_label(name: &str) -> String {
    format!("{}{}{}", prefixes::PIPELINE, LABEL_SEPARATOR, name)
}

/// Generate a bind group label.
///
/// # Arguments
///
/// * `name` - The bind group name
///
/// # Returns
///
/// A label in the format "bind_group/name".
///
/// # Example
///
/// ```ignore
/// let label = bind_group_label("material");
/// assert_eq!(label, "bind_group/material");
/// ```
#[inline]
pub fn bind_group_label(name: &str) -> String {
    format!("{}{}{}", prefixes::BIND_GROUP, LABEL_SEPARATOR, name)
}

/// Generate a render pass label.
///
/// # Arguments
///
/// * `name` - The render pass name
///
/// # Returns
///
/// A label in the format "render_pass/name".
///
/// # Example
///
/// ```ignore
/// let label = render_pass_label("shadow");
/// assert_eq!(label, "render_pass/shadow");
/// ```
#[inline]
pub fn render_pass_label(name: &str) -> String {
    format!("{}{}{}", prefixes::RENDER_PASS, LABEL_SEPARATOR, name)
}

/// Generate a compute pass label.
///
/// # Arguments
///
/// * `name` - The compute pass name
///
/// # Returns
///
/// A label in the format "compute_pass/name".
///
/// # Example
///
/// ```ignore
/// let label = compute_pass_label("culling");
/// assert_eq!(label, "compute_pass/culling");
/// ```
#[inline]
pub fn compute_pass_label(name: &str) -> String {
    format!("{}{}{}", prefixes::COMPUTE_PASS, LABEL_SEPARATOR, name)
}

/// Generate a shader label.
///
/// # Arguments
///
/// * `name` - The shader name
///
/// # Returns
///
/// A label in the format "shader/name".
///
/// # Example
///
/// ```ignore
/// let label = shader_label("pbr_frag");
/// assert_eq!(label, "shader/pbr_frag");
/// ```
#[inline]
pub fn shader_label(name: &str) -> String {
    format!("{}{}{}", prefixes::SHADER, LABEL_SEPARATOR, name)
}

/// Generate an indexed label.
///
/// # Arguments
///
/// * `base` - The base label name
/// * `index` - The index number
///
/// # Returns
///
/// A label in the format "base_index".
///
/// # Example
///
/// ```ignore
/// let label = indexed_label("cascade", 2);
/// assert_eq!(label, "cascade_2");
/// ```
#[inline]
pub fn indexed_label(base: &str, index: u32) -> String {
    format!("{}{}{}", base, COMPONENT_SEPARATOR, index)
}

/// Generate a hierarchical indexed label.
///
/// # Arguments
///
/// * `prefix` - The prefix/category
/// * `name` - The resource name
/// * `index` - The index number
///
/// # Returns
///
/// A label in the format "prefix/name_index".
///
/// # Example
///
/// ```ignore
/// let label = hierarchical_indexed_label("buffer", "cascade", 0);
/// assert_eq!(label, "buffer/cascade_0");
/// ```
#[inline]
pub fn hierarchical_indexed_label(prefix: &str, name: &str, index: u32) -> String {
    format!("{}{}{}{}{}", prefix, LABEL_SEPARATOR, name, COMPONENT_SEPARATOR, index)
}

// ============================================================================
// Validation Helpers (T-WGPU-P4.5.3 Criterion 2)
// ============================================================================

/// Check if a character is valid for labels.
///
/// Valid characters are:
/// - Alphanumeric (a-z, A-Z, 0-9)
/// - Underscore (_)
/// - Hyphen (-)
/// - Slash (/)
/// - Dot (.)
/// - Space ( )
///
/// # Arguments
///
/// * `c` - The character to check
///
/// # Returns
///
/// `true` if the character is valid for labels.
#[inline]
pub fn is_valid_label_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-' || c == '/' || c == '.' || c == ' '
}

/// Validate a label string.
///
/// # Arguments
///
/// * `label` - The label string to validate
///
/// # Returns
///
/// `Ok(())` if valid, `Err(LabelError)` with details if invalid.
///
/// # Example
///
/// ```ignore
/// assert!(validate_label("valid_label").is_ok());
/// assert!(validate_label("").is_err());
/// assert!(validate_label(&"x".repeat(300)).is_err());
/// ```
pub fn validate_label(label: &str) -> Result<(), LabelError> {
    if label.is_empty() {
        return Err(LabelError::Empty);
    }

    if label.len() > MAX_LABEL_LENGTH {
        return Err(LabelError::TooLong {
            length: label.len(),
            max: MAX_LABEL_LENGTH,
        });
    }

    for (i, c) in label.chars().enumerate() {
        if !is_valid_label_char(c) {
            return Err(LabelError::InvalidCharacter {
                character: c,
                position: i,
            });
        }
    }

    Ok(())
}

/// Sanitize a label string by replacing invalid characters.
///
/// Invalid characters are replaced with underscores, and the result
/// is truncated to the maximum allowed length.
///
/// # Arguments
///
/// * `label` - The label string to sanitize
///
/// # Returns
///
/// A sanitized label string.
///
/// # Example
///
/// ```ignore
/// let sanitized = sanitize_label("my@label#with$invalid");
/// assert_eq!(sanitized, "my_label_with_invalid");
/// ```
pub fn sanitize_label(label: &str) -> String {
    let sanitized: String = label
        .chars()
        .map(|c| if is_valid_label_char(c) { c } else { '_' })
        .collect();

    if sanitized.len() > MAX_LABEL_LENGTH {
        sanitized[..MAX_LABEL_LENGTH].to_string()
    } else {
        sanitized
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: ResourceLabel Construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_label_new() {
        let label = ResourceLabel::new("test");
        assert_eq!(label.as_str(), "test");
        assert_eq!(label.depth(), 1);
    }

    #[test]
    fn test_resource_label_from_segments() {
        let label = ResourceLabel::from_segments(["a", "b", "c"]);
        assert_eq!(label.as_str(), "a/b/c");
        assert_eq!(label.depth(), 3);
    }

    #[test]
    fn test_resource_label_from_segments_single() {
        let label = ResourceLabel::from_segments(["single"]);
        assert_eq!(label.as_str(), "single");
        assert_eq!(label.depth(), 1);
    }

    #[test]
    fn test_resource_label_empty() {
        let label = ResourceLabel::empty();
        assert!(label.is_empty());
        assert_eq!(label.depth(), 0);
        assert_eq!(label.as_str(), "");
    }

    #[test]
    fn test_resource_label_default() {
        let label = ResourceLabel::default();
        assert!(label.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: ResourceLabel Hierarchy (Criterion 3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_label_child() {
        let parent = ResourceLabel::new("pass");
        let child = parent.child("subpass");
        assert_eq!(child.as_str(), "pass/subpass");
        assert_eq!(child.depth(), 2);
    }

    #[test]
    fn test_resource_label_child_chain() {
        let label = ResourceLabel::new("a")
            .child("b")
            .child("c")
            .child("d");
        assert_eq!(label.as_str(), "a/b/c/d");
        assert_eq!(label.depth(), 4);
    }

    #[test]
    fn test_resource_label_child_from_empty() {
        let empty = ResourceLabel::empty();
        let child = empty.child("first");
        assert_eq!(child.as_str(), "first");
        assert_eq!(child.depth(), 1);
    }

    #[test]
    fn test_resource_label_leaf() {
        let label = ResourceLabel::from_segments(["a", "b", "c"]);
        assert_eq!(label.leaf(), Some("c"));
    }

    #[test]
    fn test_resource_label_leaf_empty() {
        let label = ResourceLabel::empty();
        assert_eq!(label.leaf(), None);
    }

    #[test]
    fn test_resource_label_parent() {
        let label = ResourceLabel::from_segments(["a", "b", "c"]);
        let parent = label.parent().unwrap();
        assert_eq!(parent.as_str(), "a/b");
    }

    #[test]
    fn test_resource_label_parent_single() {
        let label = ResourceLabel::new("single");
        assert!(label.parent().is_none());
    }

    #[test]
    fn test_resource_label_parent_empty() {
        let label = ResourceLabel::empty();
        assert!(label.parent().is_none());
    }

    #[test]
    fn test_resource_label_segments() {
        let label = ResourceLabel::from_segments(["a", "b", "c"]);
        assert_eq!(label.segments(), &["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_resource_label_join() {
        let a = ResourceLabel::new("pass");
        let b = ResourceLabel::from_segments(["subpass", "draw"]);
        let combined = a.join(&b);
        assert_eq!(combined.as_str(), "pass/subpass/draw");
    }

    // -------------------------------------------------------------------------
    // Test: ResourceLabel Validation (Criterion 2)
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_label_validate_valid() {
        let label = ResourceLabel::new("valid_label");
        assert!(label.validate().is_ok());
    }

    #[test]
    fn test_resource_label_validate_empty() {
        let label = ResourceLabel::empty();
        let result = label.validate();
        assert!(matches!(result, Err(LabelError::Empty)));
    }

    #[test]
    fn test_resource_label_validate_too_long() {
        let long_name = "x".repeat(300);
        let label = ResourceLabel::new(&long_name);
        let result = label.validate();
        assert!(matches!(result, Err(LabelError::TooLong { .. })));
    }

    #[test]
    fn test_resource_label_validate_invalid_char() {
        let label = ResourceLabel::new("invalid@char");
        let result = label.validate();
        assert!(matches!(result, Err(LabelError::InvalidCharacter { character: '@', .. })));
    }

    #[test]
    fn test_resource_label_is_valid() {
        assert!(ResourceLabel::new("valid").is_valid());
        assert!(!ResourceLabel::empty().is_valid());
    }

    #[test]
    fn test_resource_label_validate_with_hierarchy() {
        let label = ResourceLabel::from_segments(["pass", "subpass", "draw"]);
        assert!(label.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // Test: ResourceLabel Display and Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_label_display() {
        let label = ResourceLabel::new("test").child("child");
        assert_eq!(format!("{}", label), "test/child");
    }

    #[test]
    fn test_resource_label_as_ref() {
        let label = ResourceLabel::new("test");
        let s: &str = label.as_ref();
        assert_eq!(s, "test");
    }

    #[test]
    fn test_resource_label_from_str() {
        let label: ResourceLabel = "pass/subpass".into();
        assert_eq!(label.as_str(), "pass/subpass");
        assert_eq!(label.depth(), 2);
    }

    #[test]
    fn test_resource_label_from_string() {
        let label: ResourceLabel = String::from("single").into();
        assert_eq!(label.as_str(), "single");
    }

    #[test]
    fn test_resource_label_clone() {
        let label = ResourceLabel::new("test").child("child");
        let cloned = label.clone();
        assert_eq!(label, cloned);
    }

    #[test]
    fn test_resource_label_hash() {
        use std::collections::HashMap;
        let mut map: HashMap<ResourceLabel, i32> = HashMap::new();
        map.insert(ResourceLabel::new("test"), 42);
        assert_eq!(map.get(&ResourceLabel::new("test")), Some(&42));
    }

    // -------------------------------------------------------------------------
    // Test: LabelBuilder (Criterion 1)
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_builder_basic() {
        let label = LabelBuilder::new("texture").build();
        assert_eq!(label, "texture");
    }

    #[test]
    fn test_label_builder_with_prefix() {
        let label = LabelBuilder::new("buffer")
            .with_prefix("vertex")
            .build();
        assert_eq!(label, "vertex_buffer");
    }

    #[test]
    fn test_label_builder_with_suffix() {
        let label = LabelBuilder::new("texture")
            .with_suffix("albedo")
            .build();
        assert_eq!(label, "texture_albedo");
    }

    #[test]
    fn test_label_builder_with_index() {
        let label = LabelBuilder::new("cascade")
            .with_index(2)
            .build();
        assert_eq!(label, "cascade_2");
    }

    #[test]
    fn test_label_builder_full() {
        let label = LabelBuilder::new("buffer")
            .with_prefix("vertex")
            .with_suffix("staging")
            .with_index(0)
            .build();
        assert_eq!(label, "vertex_buffer_staging_0");
    }

    #[test]
    fn test_label_builder_custom_separator() {
        let label = LabelBuilder::new("buffer")
            .with_prefix("vertex")
            .with_separator('-')
            .build();
        assert_eq!(label, "vertex-buffer");
    }

    #[test]
    fn test_label_builder_build_resource_label() {
        let label = LabelBuilder::new("buffer")
            .with_index(0)
            .build_resource_label();
        assert_eq!(label.as_str(), "buffer_0");
    }

    #[test]
    fn test_label_builder_validate() {
        let valid = LabelBuilder::new("valid");
        assert!(valid.validate().is_ok());

        let invalid = LabelBuilder::new("");
        assert!(matches!(invalid.validate(), Err(LabelError::Empty)));
    }

    #[test]
    fn test_label_builder_default() {
        let builder = LabelBuilder::default();
        assert!(builder.validate().is_err()); // Empty name
    }

    // -------------------------------------------------------------------------
    // Test: LabelRegistry (Criterion 4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_registry_new() {
        let registry = LabelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_label_registry_with_capacity() {
        let registry = LabelRegistry::with_capacity(100);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_label_registry_register() {
        let mut registry = LabelRegistry::new();
        assert!(registry.register("label1"));
        assert!(!registry.register("label1")); // Duplicate
        assert!(registry.register("label2"));
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_label_registry_register_resource_label() {
        let mut registry = LabelRegistry::new();
        let label = ResourceLabel::new("test");
        assert!(registry.register_resource_label(&label));
        assert!(registry.contains("test"));
    }

    #[test]
    fn test_label_registry_unregister() {
        let mut registry = LabelRegistry::new();
        registry.register("label");
        assert!(registry.unregister("label"));
        assert!(!registry.unregister("label")); // Already removed
        assert!(!registry.contains("label"));
    }

    #[test]
    fn test_label_registry_contains() {
        let mut registry = LabelRegistry::new();
        registry.register("exists");
        assert!(registry.contains("exists"));
        assert!(!registry.contains("not_exists"));
    }

    #[test]
    fn test_label_registry_clear() {
        let mut registry = LabelRegistry::new();
        registry.register("a");
        registry.register("b");
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_label_registry_iter() {
        let mut registry = LabelRegistry::new();
        registry.register("a");
        registry.register("b");
        let labels: Vec<_> = registry.iter().cloned().collect();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"a".to_string()));
        assert!(labels.contains(&"b".to_string()));
    }

    #[test]
    fn test_label_registry_with_prefix() {
        let mut registry = LabelRegistry::new();
        registry.register("buffer/vertex_0");
        registry.register("buffer/index_0");
        registry.register("texture/albedo_0");

        let buffers = registry.with_prefix("buffer/");
        assert_eq!(buffers.len(), 2);
    }

    #[test]
    fn test_label_registry_generate_unique() {
        let mut registry = LabelRegistry::new();
        registry.register("buffer");
        registry.register("buffer_1");

        let unique = registry.generate_unique("buffer");
        assert_eq!(unique, "buffer_2");
    }

    #[test]
    fn test_label_registry_generate_unique_not_taken() {
        let registry = LabelRegistry::new();
        let unique = registry.generate_unique("fresh");
        assert_eq!(unique, "fresh");
    }

    #[test]
    fn test_label_registry_clone() {
        let mut registry = LabelRegistry::new();
        registry.register("a");
        let cloned = registry.clone();
        assert!(cloned.contains("a"));
    }

    // -------------------------------------------------------------------------
    // Test: Label Prefixes (Criterion 5)
    // -------------------------------------------------------------------------

    #[test]
    fn test_prefixes_constants() {
        assert_eq!(prefixes::BUFFER, "buffer");
        assert_eq!(prefixes::TEXTURE, "texture");
        assert_eq!(prefixes::SAMPLER, "sampler");
        assert_eq!(prefixes::PIPELINE, "pipeline");
        assert_eq!(prefixes::BIND_GROUP, "bind_group");
        assert_eq!(prefixes::RENDER_PASS, "render_pass");
        assert_eq!(prefixes::COMPUTE_PASS, "compute_pass");
        assert_eq!(prefixes::SHADER, "shader");
        assert_eq!(prefixes::QUERY, "query");
        assert_eq!(prefixes::BIND_GROUP_LAYOUT, "bind_group_layout");
        assert_eq!(prefixes::PIPELINE_LAYOUT, "pipeline_layout");
        assert_eq!(prefixes::COMMAND_ENCODER, "command_encoder");
        assert_eq!(prefixes::RENDER_BUNDLE, "render_bundle");
    }

    // -------------------------------------------------------------------------
    // Test: Label Generation Functions (Criterion 1)
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_label() {
        assert_eq!(buffer_label("vertex"), "buffer/vertex");
    }

    #[test]
    fn test_texture_label() {
        assert_eq!(texture_label("diffuse"), "texture/diffuse");
    }

    #[test]
    fn test_sampler_label() {
        assert_eq!(sampler_label("linear"), "sampler/linear");
    }

    #[test]
    fn test_pipeline_label() {
        assert_eq!(pipeline_label("pbr"), "pipeline/pbr");
    }

    #[test]
    fn test_bind_group_label() {
        assert_eq!(bind_group_label("material"), "bind_group/material");
    }

    #[test]
    fn test_render_pass_label() {
        assert_eq!(render_pass_label("shadow"), "render_pass/shadow");
    }

    #[test]
    fn test_compute_pass_label() {
        assert_eq!(compute_pass_label("culling"), "compute_pass/culling");
    }

    #[test]
    fn test_shader_label() {
        assert_eq!(shader_label("pbr_frag"), "shader/pbr_frag");
    }

    #[test]
    fn test_indexed_label() {
        assert_eq!(indexed_label("cascade", 0), "cascade_0");
        assert_eq!(indexed_label("cascade", 3), "cascade_3");
    }

    #[test]
    fn test_hierarchical_indexed_label() {
        assert_eq!(
            hierarchical_indexed_label("buffer", "cascade", 0),
            "buffer/cascade_0"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Validation Functions (Criterion 2)
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_valid_label_char() {
        // Valid characters
        assert!(is_valid_label_char('a'));
        assert!(is_valid_label_char('Z'));
        assert!(is_valid_label_char('5'));
        assert!(is_valid_label_char('_'));
        assert!(is_valid_label_char('-'));
        assert!(is_valid_label_char('/'));
        assert!(is_valid_label_char('.'));
        assert!(is_valid_label_char(' '));

        // Invalid characters
        assert!(!is_valid_label_char('@'));
        assert!(!is_valid_label_char('#'));
        assert!(!is_valid_label_char('$'));
        assert!(!is_valid_label_char('%'));
        assert!(!is_valid_label_char('!'));
    }

    #[test]
    fn test_validate_label_valid() {
        assert!(validate_label("valid_label").is_ok());
        assert!(validate_label("with-hyphens").is_ok());
        assert!(validate_label("with/slashes").is_ok());
        assert!(validate_label("with.dots").is_ok());
        assert!(validate_label("with spaces").is_ok());
    }

    #[test]
    fn test_validate_label_empty() {
        let result = validate_label("");
        assert!(matches!(result, Err(LabelError::Empty)));
    }

    #[test]
    fn test_validate_label_too_long() {
        let long = "x".repeat(300);
        let result = validate_label(&long);
        assert!(matches!(result, Err(LabelError::TooLong { .. })));
    }

    #[test]
    fn test_validate_label_invalid_char() {
        let result = validate_label("invalid@char");
        assert!(matches!(result, Err(LabelError::InvalidCharacter { character: '@', position: 7 })));
    }

    #[test]
    fn test_sanitize_label() {
        assert_eq!(sanitize_label("valid"), "valid");
        assert_eq!(sanitize_label("with@invalid#chars"), "with_invalid_chars");
    }

    #[test]
    fn test_sanitize_label_too_long() {
        let long = "x".repeat(300);
        let sanitized = sanitize_label(&long);
        assert_eq!(sanitized.len(), MAX_LABEL_LENGTH);
    }

    // -------------------------------------------------------------------------
    // Test: LabelError Display
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_error_display_empty() {
        let err = LabelError::Empty;
        assert_eq!(format!("{}", err), "Label cannot be empty");
    }

    #[test]
    fn test_label_error_display_too_long() {
        let err = LabelError::TooLong { length: 300, max: 256 };
        assert_eq!(format!("{}", err), "Label length 300 exceeds maximum 256");
    }

    #[test]
    fn test_label_error_display_invalid_char() {
        let err = LabelError::InvalidCharacter { character: '@', position: 5 };
        assert_eq!(format!("{}", err), "Invalid character '@' at position 5");
    }

    #[test]
    fn test_label_error_display_empty_segment() {
        let err = LabelError::EmptySegment { position: 2 };
        assert_eq!(format!("{}", err), "Empty segment at position 2");
    }

    // -------------------------------------------------------------------------
    // Test: Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_label_unicode() {
        // Note: Unicode is valid but may not display well in all debugging tools
        let label = ResourceLabel::new("unicode_test");
        assert!(label.is_valid());
    }

    #[test]
    fn test_resource_label_max_length_boundary() {
        let exactly_max = "x".repeat(MAX_LABEL_LENGTH);
        let label = ResourceLabel::new(&exactly_max);
        assert!(label.validate().is_ok());

        let over_max = "x".repeat(MAX_LABEL_LENGTH + 1);
        let label = ResourceLabel::new(&over_max);
        assert!(label.validate().is_err());
    }

    #[test]
    fn test_label_builder_chain() {
        // Test multiple chained calls
        let label = LabelBuilder::new("name")
            .with_prefix("pre")
            .with_suffix("suf")
            .with_index(42)
            .with_separator('.')
            .build();
        assert_eq!(label, "pre.name.suf.42");
    }

    #[test]
    fn test_label_registry_many_labels() {
        let mut registry = LabelRegistry::new();
        for i in 0..1000 {
            registry.register(&format!("label_{}", i));
        }
        assert_eq!(registry.count(), 1000);
        assert!(registry.contains("label_500"));
    }

    // -------------------------------------------------------------------------
    // Test: Integration Scenarios
    // -------------------------------------------------------------------------

    #[test]
    fn test_integration_typical_render_pass_labels() {
        let pass = ResourceLabel::new("shadow_pass");
        let cascade0 = pass.child("cascade_0");
        let cascade1 = pass.child("cascade_1");

        assert_eq!(cascade0.as_str(), "shadow_pass/cascade_0");
        assert_eq!(cascade1.as_str(), "shadow_pass/cascade_1");
        assert!(cascade0.is_valid());
        assert!(cascade1.is_valid());
    }

    #[test]
    fn test_integration_resource_tracking() {
        let mut registry = LabelRegistry::new();

        // Create and register some resources
        let vertex_buffer = buffer_label("vertex");
        let index_buffer = buffer_label("index");
        let albedo_texture = texture_label("albedo");

        registry.register(&vertex_buffer);
        registry.register(&index_buffer);
        registry.register(&albedo_texture);

        // Verify all resources are tracked
        assert!(registry.contains("buffer/vertex"));
        assert!(registry.contains("buffer/index"));
        assert!(registry.contains("texture/albedo"));

        // Get all buffers
        let buffers = registry.with_prefix("buffer/");
        assert_eq!(buffers.len(), 2);

        // Simulate resource destruction
        registry.unregister("buffer/vertex");
        assert!(!registry.contains("buffer/vertex"));
    }

    #[test]
    fn test_integration_hierarchical_pipeline() {
        // Simulate a typical frame's label hierarchy
        let frame = ResourceLabel::new("frame_0");

        let gbuffer = frame.child("gbuffer_pass");
        let gbuffer_opaque = gbuffer.child("opaque");
        let gbuffer_transparent = gbuffer.child("transparent");

        let lighting = frame.child("lighting_pass");
        let lighting_direct = lighting.child("direct");
        let lighting_indirect = lighting.child("indirect");

        assert_eq!(gbuffer_opaque.as_str(), "frame_0/gbuffer_pass/opaque");
        assert_eq!(gbuffer_transparent.as_str(), "frame_0/gbuffer_pass/transparent");
        assert_eq!(lighting_direct.as_str(), "frame_0/lighting_pass/direct");
        assert_eq!(lighting_indirect.as_str(), "frame_0/lighting_pass/indirect");
    }

    #[test]
    fn test_integration_builder_with_registry() {
        let mut registry = LabelRegistry::new();

        // Generate unique labels for multiple instances
        for i in 0..5 {
            let label = LabelBuilder::new("buffer")
                .with_prefix("vertex")
                .with_index(i)
                .build();
            registry.register(&label);
        }

        assert_eq!(registry.count(), 5);
        assert!(registry.contains("vertex_buffer_0"));
        assert!(registry.contains("vertex_buffer_4"));
    }
}
