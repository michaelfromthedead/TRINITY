//! Python bindings for render pass construction (T-WGPU-P7.6.5).
//!
//! This module provides Python-accessible types for creating and configuring
//! render pass descriptors using a fluent builder pattern.
//!
//! # Types
//!
//! - [`PyLoadOp`] - Load operation for attachments (Clear, Load)
//! - [`PyStoreOp`] - Store operation for attachments (Store, Discard)
//! - [`PyTextureView`] - Placeholder for texture view references
//! - [`PyColorAttachment`] - Color attachment configuration
//! - [`PyDepthStencilAttachment`] - Depth/stencil attachment configuration
//! - [`PyTimestampWrites`] - Timestamp query configuration
//! - [`PyRenderPassDescriptor`] - Complete render pass description
//! - [`PyRenderPassBuilder`] - Fluent builder for render passes
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     RenderPassBuilder, LoadOp, StoreOp, TextureView
//! )
//!
//! # Create a render pass with color and depth attachments
//! render_pass = (
//!     RenderPassBuilder()
//!     .label("main_pass")
//!     .color(
//!         TextureView(0),  # target
//!         load_op=LoadOp.Clear,
//!         store_op=StoreOp.Store,
//!         clear_color=[0.0, 0.0, 0.0, 1.0]
//!     )
//!     .depth(
//!         TextureView(1),
//!         load_op=LoadOp.Clear,
//!         store_op=StoreOp.Store,
//!         clear_value=1.0
//!     )
//!     .build()
//! )
//! ```
//!
//! # Feature Gate
//!
//! All types are gated behind the `pyo3` feature flag.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ============================================================================
// PyLoadOp
// ============================================================================

/// Load operation performed at the beginning of a render pass.
///
/// Determines what happens to the attachment's contents when the pass starts.
#[pyclass(name = "LoadOp")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyLoadOp {
    /// Clear the attachment to a specified value.
    Clear = 0,
    /// Preserve the existing contents of the attachment.
    Load = 1,
}

#[pymethods]
impl PyLoadOp {
    /// Returns the canonical name of this load operation.
    pub fn name(&self) -> &str {
        match self {
            Self::Clear => "Clear",
            Self::Load => "Load",
        }
    }

    fn __repr__(&self) -> String {
        format!("LoadOp.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

impl Default for PyLoadOp {
    fn default() -> Self {
        Self::Clear
    }
}

// ============================================================================
// PyStoreOp
// ============================================================================

/// Store operation performed at the end of a render pass.
///
/// Determines what happens to the attachment's contents when the pass ends.
#[pyclass(name = "StoreOp")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyStoreOp {
    /// Write the results to the attachment.
    Store = 0,
    /// Discard the results (may improve performance on tile-based GPUs).
    Discard = 1,
}

#[pymethods]
impl PyStoreOp {
    /// Returns the canonical name of this store operation.
    pub fn name(&self) -> &str {
        match self {
            Self::Store => "Store",
            Self::Discard => "Discard",
        }
    }

    fn __repr__(&self) -> String {
        format!("StoreOp.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

impl Default for PyStoreOp {
    fn default() -> Self {
        Self::Store
    }
}

// ============================================================================
// PyTextureView
// ============================================================================

/// Placeholder for texture view references.
///
/// In a real implementation, this would hold a reference to an actual
/// wgpu::TextureView. For now, it uses an ID-based reference system.
#[pyclass(name = "TextureView")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyTextureView {
    /// Unique identifier for this texture view.
    id: u64,
    /// Optional label for debugging.
    label: Option<String>,
}

#[pymethods]
impl PyTextureView {
    /// Creates a new texture view reference with the given ID.
    #[new]
    #[pyo3(signature = (id, label=None))]
    pub fn new(id: u64, label: Option<String>) -> Self {
        Self { id, label }
    }

    /// Returns the unique identifier of this texture view.
    #[getter]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the optional label of this texture view.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Creates a texture view with a label.
    #[staticmethod]
    pub fn with_label(id: u64, label: &str) -> Self {
        Self {
            id,
            label: Some(label.to_string()),
        }
    }

    fn __repr__(&self) -> String {
        match &self.label {
            Some(lbl) => format!("TextureView(id={}, label=\"{}\")", self.id, lbl),
            None => format!("TextureView(id={})", self.id),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __hash__(&self) -> u64 {
        self.id
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

// ============================================================================
// PyColorAttachment
// ============================================================================

/// Configuration for a color attachment in a render pass.
///
/// Color attachments are render targets that receive fragment shader output.
#[pyclass(name = "ColorAttachment")]
#[derive(Clone, Debug)]
pub struct PyColorAttachment {
    /// The texture view to render into.
    target: Option<PyTextureView>,
    /// Optional resolve target for MSAA.
    resolve_target: Option<PyTextureView>,
    /// Load operation at pass start.
    load_op: PyLoadOp,
    /// Store operation at pass end.
    store_op: PyStoreOp,
    /// Clear color [R, G, B, A] when load_op is Clear.
    clear_color: [f64; 4],
}

impl Default for PyColorAttachment {
    fn default() -> Self {
        Self {
            target: None,
            resolve_target: None,
            load_op: PyLoadOp::Clear,
            store_op: PyStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[pymethods]
impl PyColorAttachment {
    /// Creates a new color attachment with default values.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a color attachment targeting a specific texture view.
    #[staticmethod]
    pub fn for_target(target: PyTextureView) -> Self {
        Self {
            target: Some(target),
            ..Default::default()
        }
    }

    // -- Getters --

    /// Returns the target texture view.
    #[getter]
    pub fn target(&self) -> Option<PyTextureView> {
        self.target.clone()
    }

    /// Returns the resolve target texture view.
    #[getter]
    pub fn resolve_target(&self) -> Option<PyTextureView> {
        self.resolve_target.clone()
    }

    /// Returns the load operation.
    #[getter]
    pub fn load_op(&self) -> PyLoadOp {
        self.load_op
    }

    /// Returns the store operation.
    #[getter]
    pub fn store_op(&self) -> PyStoreOp {
        self.store_op
    }

    /// Returns the clear color as [R, G, B, A].
    #[getter]
    pub fn clear_color(&self) -> [f64; 4] {
        self.clear_color
    }

    // -- Builder methods --

    /// Sets the target texture view (builder pattern).
    pub fn with_target(&self, target: PyTextureView) -> Self {
        Self {
            target: Some(target),
            ..self.clone()
        }
    }

    /// Sets the resolve target for MSAA (builder pattern).
    pub fn with_resolve(&self, resolve_target: PyTextureView) -> Self {
        Self {
            resolve_target: Some(resolve_target),
            ..self.clone()
        }
    }

    /// Sets the load operation (builder pattern).
    pub fn with_load_op(&self, load_op: PyLoadOp) -> Self {
        Self {
            load_op,
            ..self.clone()
        }
    }

    /// Sets the store operation (builder pattern).
    pub fn with_store_op(&self, store_op: PyStoreOp) -> Self {
        Self {
            store_op,
            ..self.clone()
        }
    }

    /// Sets the clear color (builder pattern).
    ///
    /// # Arguments
    /// * `r` - Red component (0.0-1.0)
    /// * `g` - Green component (0.0-1.0)
    /// * `b` - Blue component (0.0-1.0)
    /// * `a` - Alpha component (0.0-1.0)
    #[pyo3(signature = (r, g, b, a=1.0))]
    pub fn with_clear_color(&self, r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            clear_color: [r, g, b, a],
            ..self.clone()
        }
    }

    /// Sets the clear color from an array [R, G, B, A] (builder pattern).
    pub fn with_clear_color_array(&self, color: [f64; 4]) -> Self {
        Self {
            clear_color: color,
            ..self.clone()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorAttachment(target={:?}, load={:?}, store={:?}, clear={:?})",
            self.target.as_ref().map(|t| t.id),
            self.load_op,
            self.store_op,
            self.clear_color
        )
    }
}

// ============================================================================
// PyDepthStencilAttachment
// ============================================================================

/// Configuration for a depth/stencil attachment in a render pass.
///
/// Depth/stencil attachments are used for depth testing and stencil operations.
#[pyclass(name = "DepthStencilAttachment")]
#[derive(Clone, Debug)]
pub struct PyDepthStencilAttachment {
    /// The texture view for depth/stencil.
    view: Option<PyTextureView>,
    /// Depth load operation.
    depth_load_op: PyLoadOp,
    /// Depth store operation.
    depth_store_op: PyStoreOp,
    /// Stencil load operation.
    stencil_load_op: PyLoadOp,
    /// Stencil store operation.
    stencil_store_op: PyStoreOp,
    /// Clear value for depth (0.0-1.0, typically 1.0 for far plane).
    depth_clear_value: f32,
    /// Clear value for stencil (0-255).
    stencil_clear_value: u32,
    /// Whether depth is read-only (no writes).
    depth_read_only: bool,
    /// Whether stencil is read-only (no writes).
    stencil_read_only: bool,
}

impl Default for PyDepthStencilAttachment {
    fn default() -> Self {
        Self {
            view: None,
            depth_load_op: PyLoadOp::Clear,
            depth_store_op: PyStoreOp::Store,
            stencil_load_op: PyLoadOp::Clear,
            stencil_store_op: PyStoreOp::Store,
            depth_clear_value: 1.0,
            stencil_clear_value: 0,
            depth_read_only: false,
            stencil_read_only: false,
        }
    }
}

#[pymethods]
impl PyDepthStencilAttachment {
    /// Creates a new depth/stencil attachment with default values.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a depth/stencil attachment for a specific texture view.
    #[staticmethod]
    pub fn for_view(view: PyTextureView) -> Self {
        Self {
            view: Some(view),
            ..Default::default()
        }
    }

    /// Creates a depth-only attachment (stencil disabled).
    #[staticmethod]
    pub fn depth_only(view: PyTextureView) -> Self {
        Self {
            view: Some(view),
            stencil_load_op: PyLoadOp::Load,
            stencil_store_op: PyStoreOp::Discard,
            stencil_read_only: true,
            ..Default::default()
        }
    }

    // -- Getters --

    /// Returns the texture view.
    #[getter]
    pub fn view(&self) -> Option<PyTextureView> {
        self.view.clone()
    }

    /// Returns the depth load operation.
    #[getter]
    pub fn depth_load_op(&self) -> PyLoadOp {
        self.depth_load_op
    }

    /// Returns the depth store operation.
    #[getter]
    pub fn depth_store_op(&self) -> PyStoreOp {
        self.depth_store_op
    }

    /// Returns the stencil load operation.
    #[getter]
    pub fn stencil_load_op(&self) -> PyLoadOp {
        self.stencil_load_op
    }

    /// Returns the stencil store operation.
    #[getter]
    pub fn stencil_store_op(&self) -> PyStoreOp {
        self.stencil_store_op
    }

    /// Returns the depth clear value.
    #[getter]
    pub fn depth_clear_value(&self) -> f32 {
        self.depth_clear_value
    }

    /// Returns the stencil clear value.
    #[getter]
    pub fn stencil_clear_value(&self) -> u32 {
        self.stencil_clear_value
    }

    /// Returns whether depth is read-only.
    #[getter]
    pub fn depth_read_only(&self) -> bool {
        self.depth_read_only
    }

    /// Returns whether stencil is read-only.
    #[getter]
    pub fn stencil_read_only(&self) -> bool {
        self.stencil_read_only
    }

    // -- Builder methods --

    /// Sets the texture view (builder pattern).
    pub fn with_view(&self, view: PyTextureView) -> Self {
        Self {
            view: Some(view),
            ..self.clone()
        }
    }

    /// Sets the depth load operation (builder pattern).
    pub fn with_depth_load_op(&self, load_op: PyLoadOp) -> Self {
        Self {
            depth_load_op: load_op,
            ..self.clone()
        }
    }

    /// Sets the depth store operation (builder pattern).
    pub fn with_depth_store_op(&self, store_op: PyStoreOp) -> Self {
        Self {
            depth_store_op: store_op,
            ..self.clone()
        }
    }

    /// Sets the stencil load operation (builder pattern).
    pub fn with_stencil_load_op(&self, load_op: PyLoadOp) -> Self {
        Self {
            stencil_load_op: load_op,
            ..self.clone()
        }
    }

    /// Sets the stencil store operation (builder pattern).
    pub fn with_stencil_store_op(&self, store_op: PyStoreOp) -> Self {
        Self {
            stencil_store_op: store_op,
            ..self.clone()
        }
    }

    /// Sets the depth clear value (builder pattern).
    pub fn with_depth_clear_value(&self, value: f32) -> Self {
        Self {
            depth_clear_value: value,
            ..self.clone()
        }
    }

    /// Sets the stencil clear value (builder pattern).
    pub fn with_stencil_clear_value(&self, value: u32) -> Self {
        Self {
            stencil_clear_value: value,
            ..self.clone()
        }
    }

    /// Sets depth to read-only mode (builder pattern).
    pub fn with_depth_read_only(&self, read_only: bool) -> Self {
        Self {
            depth_read_only: read_only,
            ..self.clone()
        }
    }

    /// Sets stencil to read-only mode (builder pattern).
    pub fn with_stencil_read_only(&self, read_only: bool) -> Self {
        Self {
            stencil_read_only: read_only,
            ..self.clone()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DepthStencilAttachment(view={:?}, depth_clear={}, stencil_clear={})",
            self.view.as_ref().map(|v| v.id),
            self.depth_clear_value,
            self.stencil_clear_value
        )
    }
}

// ============================================================================
// PyTimestampWrites
// ============================================================================

/// Configuration for timestamp queries in a render pass.
///
/// Allows measuring GPU execution time of render passes.
#[pyclass(name = "TimestampWrites")]
#[derive(Clone, Debug)]
pub struct PyTimestampWrites {
    /// Query set ID for timestamp writes.
    query_set: u32,
    /// Index in the query set for the beginning timestamp.
    beginning_of_pass_write_index: Option<u32>,
    /// Index in the query set for the end timestamp.
    end_of_pass_write_index: Option<u32>,
}

impl Default for PyTimestampWrites {
    fn default() -> Self {
        Self {
            query_set: 0,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: None,
        }
    }
}

#[pymethods]
impl PyTimestampWrites {
    /// Creates a new timestamp writes configuration.
    #[new]
    #[pyo3(signature = (query_set, beginning_index=None, end_index=None))]
    pub fn new(query_set: u32, beginning_index: Option<u32>, end_index: Option<u32>) -> Self {
        Self {
            query_set,
            beginning_of_pass_write_index: beginning_index,
            end_of_pass_write_index: end_index,
        }
    }

    /// Creates a timestamp configuration that writes both start and end.
    #[staticmethod]
    pub fn full(query_set: u32, beginning_index: u32, end_index: u32) -> Self {
        Self {
            query_set,
            beginning_of_pass_write_index: Some(beginning_index),
            end_of_pass_write_index: Some(end_index),
        }
    }

    /// Creates a timestamp configuration that writes only the beginning.
    #[staticmethod]
    pub fn beginning_only(query_set: u32, index: u32) -> Self {
        Self {
            query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: None,
        }
    }

    /// Creates a timestamp configuration that writes only the end.
    #[staticmethod]
    pub fn end_only(query_set: u32, index: u32) -> Self {
        Self {
            query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(index),
        }
    }

    // -- Getters --

    /// Returns the query set ID.
    #[getter]
    pub fn query_set(&self) -> u32 {
        self.query_set
    }

    /// Returns the beginning timestamp write index.
    #[getter]
    pub fn beginning_of_pass_write_index(&self) -> Option<u32> {
        self.beginning_of_pass_write_index
    }

    /// Returns the end timestamp write index.
    #[getter]
    pub fn end_of_pass_write_index(&self) -> Option<u32> {
        self.end_of_pass_write_index
    }

    fn __repr__(&self) -> String {
        format!(
            "TimestampWrites(query_set={}, begin={:?}, end={:?})",
            self.query_set,
            self.beginning_of_pass_write_index,
            self.end_of_pass_write_index
        )
    }
}

// ============================================================================
// PyRenderPassDescriptor
// ============================================================================

/// Complete description of a render pass.
///
/// Contains all attachments and configuration needed to begin a render pass.
#[pyclass(name = "RenderPassDescriptor")]
#[derive(Clone, Debug)]
pub struct PyRenderPassDescriptor {
    /// Optional debug label.
    label: Option<String>,
    /// Color attachments (render targets).
    color_attachments: Vec<PyColorAttachment>,
    /// Optional depth/stencil attachment.
    depth_stencil_attachment: Option<PyDepthStencilAttachment>,
    /// Optional occlusion query set ID.
    occlusion_query_set: Option<u32>,
    /// Optional timestamp writes configuration.
    timestamp_writes: Option<PyTimestampWrites>,
}

impl Default for PyRenderPassDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        }
    }
}

#[pymethods]
impl PyRenderPassDescriptor {
    /// Creates a new empty render pass descriptor.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    // -- Getters --

    /// Returns the debug label.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Returns the color attachments.
    #[getter]
    pub fn color_attachments(&self) -> Vec<PyColorAttachment> {
        self.color_attachments.clone()
    }

    /// Returns the number of color attachments.
    pub fn color_attachment_count(&self) -> usize {
        self.color_attachments.len()
    }

    /// Returns a specific color attachment by index.
    pub fn get_color_attachment(&self, index: usize) -> Option<PyColorAttachment> {
        self.color_attachments.get(index).cloned()
    }

    /// Returns the depth/stencil attachment.
    #[getter]
    pub fn depth_stencil_attachment(&self) -> Option<PyDepthStencilAttachment> {
        self.depth_stencil_attachment.clone()
    }

    /// Returns the occlusion query set ID.
    #[getter]
    pub fn occlusion_query_set(&self) -> Option<u32> {
        self.occlusion_query_set
    }

    /// Returns the timestamp writes configuration.
    #[getter]
    pub fn timestamp_writes(&self) -> Option<PyTimestampWrites> {
        self.timestamp_writes.clone()
    }

    /// Returns true if this descriptor has at least one attachment.
    pub fn has_attachments(&self) -> bool {
        !self.color_attachments.is_empty() || self.depth_stencil_attachment.is_some()
    }

    /// Returns true if this descriptor has a depth/stencil attachment.
    pub fn has_depth_stencil(&self) -> bool {
        self.depth_stencil_attachment.is_some()
    }

    // -- Builder methods --

    /// Sets the debug label (builder pattern).
    pub fn with_label(&self, label: &str) -> Self {
        Self {
            label: Some(label.to_string()),
            ..self.clone()
        }
    }

    /// Adds a color attachment (builder pattern).
    pub fn add_color_attachment(&self, attachment: PyColorAttachment) -> Self {
        let mut attachments = self.color_attachments.clone();
        attachments.push(attachment);
        Self {
            color_attachments: attachments,
            ..self.clone()
        }
    }

    /// Sets the depth/stencil attachment (builder pattern).
    pub fn with_depth_stencil(&self, attachment: PyDepthStencilAttachment) -> Self {
        Self {
            depth_stencil_attachment: Some(attachment),
            ..self.clone()
        }
    }

    /// Sets the occlusion query set (builder pattern).
    pub fn with_occlusion_query(&self, query_set: u32) -> Self {
        Self {
            occlusion_query_set: Some(query_set),
            ..self.clone()
        }
    }

    /// Sets the timestamp writes configuration (builder pattern).
    pub fn with_timestamp_writes(&self, timestamp_writes: PyTimestampWrites) -> Self {
        Self {
            timestamp_writes: Some(timestamp_writes),
            ..self.clone()
        }
    }

    /// Validates the descriptor and returns an error message if invalid.
    pub fn validate(&self) -> Option<String> {
        if !self.has_attachments() {
            return Some("Render pass must have at least one color or depth/stencil attachment".to_string());
        }
        None
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderPassDescriptor(label={:?}, colors={}, depth_stencil={})",
            self.label,
            self.color_attachments.len(),
            self.depth_stencil_attachment.is_some()
        )
    }
}

// ============================================================================
// PyRenderPassBuilder
// ============================================================================

/// Fluent builder for constructing render pass descriptors.
///
/// Provides a clean API for building complex render pass configurations.
///
/// # Example (Python)
///
/// ```python
/// render_pass = (
///     RenderPassBuilder()
///     .label("main_pass")
///     .color(TextureView(0), clear_color=[0.1, 0.1, 0.1, 1.0])
///     .depth(TextureView(1), clear_value=1.0)
///     .build()
/// )
/// ```
#[pyclass(name = "RenderPassBuilder")]
#[derive(Clone, Debug)]
pub struct PyRenderPassBuilder {
    descriptor: PyRenderPassDescriptor,
}

impl Default for PyRenderPassBuilder {
    fn default() -> Self {
        Self {
            descriptor: PyRenderPassDescriptor::default(),
        }
    }
}

#[pymethods]
impl PyRenderPassBuilder {
    /// Creates a new render pass builder.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the debug label for the render pass.
    pub fn label(&self, label: &str) -> Self {
        Self {
            descriptor: self.descriptor.with_label(label),
        }
    }

    /// Adds a color attachment with the given parameters.
    ///
    /// # Arguments
    /// * `target` - The texture view to render into
    /// * `load_op` - Load operation (default: Clear)
    /// * `store_op` - Store operation (default: Store)
    /// * `clear_color` - Clear color [R, G, B, A] (default: [0, 0, 0, 1])
    /// * `resolve_target` - Optional MSAA resolve target
    #[pyo3(signature = (target, load_op=None, store_op=None, clear_color=None, resolve_target=None))]
    pub fn color(
        &self,
        target: PyTextureView,
        load_op: Option<PyLoadOp>,
        store_op: Option<PyStoreOp>,
        clear_color: Option<[f64; 4]>,
        resolve_target: Option<PyTextureView>,
    ) -> Self {
        let attachment = PyColorAttachment {
            target: Some(target),
            resolve_target,
            load_op: load_op.unwrap_or(PyLoadOp::Clear),
            store_op: store_op.unwrap_or(PyStoreOp::Store),
            clear_color: clear_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        };
        Self {
            descriptor: self.descriptor.add_color_attachment(attachment),
        }
    }

    /// Adds a color attachment using an existing attachment object.
    pub fn color_attachment(&self, attachment: PyColorAttachment) -> Self {
        Self {
            descriptor: self.descriptor.add_color_attachment(attachment),
        }
    }

    /// Sets the depth attachment.
    ///
    /// # Arguments
    /// * `view` - The depth texture view
    /// * `load_op` - Depth load operation (default: Clear)
    /// * `store_op` - Depth store operation (default: Store)
    /// * `clear_value` - Depth clear value (default: 1.0)
    /// * `read_only` - Whether depth is read-only (default: false)
    #[pyo3(signature = (view, load_op=None, store_op=None, clear_value=None, read_only=None))]
    pub fn depth(
        &self,
        view: PyTextureView,
        load_op: Option<PyLoadOp>,
        store_op: Option<PyStoreOp>,
        clear_value: Option<f32>,
        read_only: Option<bool>,
    ) -> Self {
        let existing = self.descriptor.depth_stencil_attachment.clone().unwrap_or_default();
        let attachment = PyDepthStencilAttachment {
            view: Some(view),
            depth_load_op: load_op.unwrap_or(PyLoadOp::Clear),
            depth_store_op: store_op.unwrap_or(PyStoreOp::Store),
            depth_clear_value: clear_value.unwrap_or(1.0),
            depth_read_only: read_only.unwrap_or(false),
            // Preserve existing stencil settings
            stencil_load_op: existing.stencil_load_op,
            stencil_store_op: existing.stencil_store_op,
            stencil_clear_value: existing.stencil_clear_value,
            stencil_read_only: existing.stencil_read_only,
        };
        Self {
            descriptor: self.descriptor.with_depth_stencil(attachment),
        }
    }

    /// Sets the stencil attachment configuration.
    ///
    /// # Arguments
    /// * `view` - The stencil texture view (typically same as depth)
    /// * `load_op` - Stencil load operation (default: Clear)
    /// * `store_op` - Stencil store operation (default: Store)
    /// * `clear_value` - Stencil clear value (default: 0)
    /// * `read_only` - Whether stencil is read-only (default: false)
    #[pyo3(signature = (view, load_op=None, store_op=None, clear_value=None, read_only=None))]
    pub fn stencil(
        &self,
        view: PyTextureView,
        load_op: Option<PyLoadOp>,
        store_op: Option<PyStoreOp>,
        clear_value: Option<u32>,
        read_only: Option<bool>,
    ) -> Self {
        let existing = self.descriptor.depth_stencil_attachment.clone().unwrap_or_default();
        let attachment = PyDepthStencilAttachment {
            view: Some(view),
            stencil_load_op: load_op.unwrap_or(PyLoadOp::Clear),
            stencil_store_op: store_op.unwrap_or(PyStoreOp::Store),
            stencil_clear_value: clear_value.unwrap_or(0),
            stencil_read_only: read_only.unwrap_or(false),
            // Preserve existing depth settings
            depth_load_op: existing.depth_load_op,
            depth_store_op: existing.depth_store_op,
            depth_clear_value: existing.depth_clear_value,
            depth_read_only: existing.depth_read_only,
        };
        Self {
            descriptor: self.descriptor.with_depth_stencil(attachment),
        }
    }

    /// Sets the depth/stencil attachment using an existing attachment object.
    pub fn depth_stencil_attachment(&self, attachment: PyDepthStencilAttachment) -> Self {
        Self {
            descriptor: self.descriptor.with_depth_stencil(attachment),
        }
    }

    /// Sets the occlusion query set.
    pub fn occlusion_query(&self, query_set: u32) -> Self {
        Self {
            descriptor: self.descriptor.with_occlusion_query(query_set),
        }
    }

    /// Sets the timestamp writes configuration.
    pub fn timestamps(&self, timestamp_writes: PyTimestampWrites) -> Self {
        Self {
            descriptor: self.descriptor.with_timestamp_writes(timestamp_writes),
        }
    }

    /// Builds the render pass descriptor.
    ///
    /// # Errors
    /// Returns an error if the descriptor has no attachments.
    pub fn build(&self) -> PyResult<PyRenderPassDescriptor> {
        if let Some(error) = self.descriptor.validate() {
            return Err(PyValueError::new_err(error));
        }
        Ok(self.descriptor.clone())
    }

    /// Builds the render pass descriptor without validation.
    ///
    /// Use with caution - may produce invalid descriptors.
    pub fn build_unchecked(&self) -> PyRenderPassDescriptor {
        self.descriptor.clone()
    }

    /// Returns the current descriptor state (for inspection).
    #[getter]
    pub fn descriptor(&self) -> PyRenderPassDescriptor {
        self.descriptor.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderPassBuilder(colors={}, depth_stencil={})",
            self.descriptor.color_attachments.len(),
            self.descriptor.depth_stencil_attachment.is_some()
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Registers all render pass types with the Python module.
pub fn register_module(
    _py: pyo3::Python<'_>,
    m: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    m.add_class::<PyLoadOp>()?;
    m.add_class::<PyStoreOp>()?;
    m.add_class::<PyTextureView>()?;
    m.add_class::<PyColorAttachment>()?;
    m.add_class::<PyDepthStencilAttachment>()?;
    m.add_class::<PyTimestampWrites>()?;
    m.add_class::<PyRenderPassDescriptor>()?;
    m.add_class::<PyRenderPassBuilder>()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PyLoadOp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_op_values() {
        assert_eq!(PyLoadOp::Clear as u8, 0);
        assert_eq!(PyLoadOp::Load as u8, 1);
    }

    #[test]
    fn test_load_op_name() {
        assert_eq!(PyLoadOp::Clear.name(), "Clear");
        assert_eq!(PyLoadOp::Load.name(), "Load");
    }

    #[test]
    fn test_load_op_default() {
        assert_eq!(PyLoadOp::default(), PyLoadOp::Clear);
    }

    // -------------------------------------------------------------------------
    // PyStoreOp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_store_op_values() {
        assert_eq!(PyStoreOp::Store as u8, 0);
        assert_eq!(PyStoreOp::Discard as u8, 1);
    }

    #[test]
    fn test_store_op_name() {
        assert_eq!(PyStoreOp::Store.name(), "Store");
        assert_eq!(PyStoreOp::Discard.name(), "Discard");
    }

    #[test]
    fn test_store_op_default() {
        assert_eq!(PyStoreOp::default(), PyStoreOp::Store);
    }

    // -------------------------------------------------------------------------
    // PyTextureView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_texture_view_creation() {
        let view = PyTextureView::new(42, None);
        assert_eq!(view.id(), 42);
        assert!(view.label().is_none());
    }

    #[test]
    fn test_texture_view_with_label() {
        let view = PyTextureView::with_label(99, "depth_buffer");
        assert_eq!(view.id(), 99);
        assert_eq!(view.label(), Some("depth_buffer".to_string()));
    }

    #[test]
    fn test_texture_view_equality() {
        let view1 = PyTextureView::new(1, None);
        let view2 = PyTextureView::new(1, Some("label".to_string()));
        let view3 = PyTextureView::new(2, None);

        // Equality is based on ID only
        assert_eq!(view1, view2);
        assert_ne!(view1, view3);
    }

    // -------------------------------------------------------------------------
    // PyColorAttachment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_attachment_default() {
        let attachment = PyColorAttachment::new();
        assert!(attachment.target().is_none());
        assert!(attachment.resolve_target().is_none());
        assert_eq!(attachment.load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.store_op(), PyStoreOp::Store);
        assert_eq!(attachment.clear_color(), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_color_attachment_for_target() {
        let view = PyTextureView::new(1, None);
        let attachment = PyColorAttachment::for_target(view.clone());
        assert_eq!(attachment.target(), Some(view));
    }

    #[test]
    fn test_color_attachment_builder_pattern() {
        let view = PyTextureView::new(1, None);
        let resolve = PyTextureView::new(2, None);

        let attachment = PyColorAttachment::new()
            .with_target(view.clone())
            .with_resolve(resolve.clone())
            .with_load_op(PyLoadOp::Load)
            .with_store_op(PyStoreOp::Discard)
            .with_clear_color(1.0, 0.5, 0.25, 0.75);

        assert_eq!(attachment.target(), Some(view));
        assert_eq!(attachment.resolve_target(), Some(resolve));
        assert_eq!(attachment.load_op(), PyLoadOp::Load);
        assert_eq!(attachment.store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.clear_color(), [1.0, 0.5, 0.25, 0.75]);
    }

    #[test]
    fn test_color_attachment_clear_color_array() {
        let attachment = PyColorAttachment::new()
            .with_clear_color_array([0.2, 0.4, 0.6, 0.8]);
        assert_eq!(attachment.clear_color(), [0.2, 0.4, 0.6, 0.8]);
    }

    // -------------------------------------------------------------------------
    // PyDepthStencilAttachment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_stencil_attachment_default() {
        let attachment = PyDepthStencilAttachment::new();
        assert!(attachment.view().is_none());
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.depth_clear_value(), 1.0);
        assert_eq!(attachment.stencil_clear_value(), 0);
        assert!(!attachment.depth_read_only());
        assert!(!attachment.stencil_read_only());
    }

    #[test]
    fn test_depth_stencil_attachment_for_view() {
        let view = PyTextureView::new(10, None);
        let attachment = PyDepthStencilAttachment::for_view(view.clone());
        assert_eq!(attachment.view(), Some(view));
    }

    #[test]
    fn test_depth_only_attachment() {
        let view = PyTextureView::new(5, None);
        let attachment = PyDepthStencilAttachment::depth_only(view.clone());

        assert_eq!(attachment.view(), Some(view));
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Discard);
        assert!(attachment.stencil_read_only());
    }

    #[test]
    fn test_depth_stencil_attachment_builder_pattern() {
        let view = PyTextureView::new(7, None);

        let attachment = PyDepthStencilAttachment::new()
            .with_view(view.clone())
            .with_depth_load_op(PyLoadOp::Load)
            .with_depth_store_op(PyStoreOp::Discard)
            .with_stencil_load_op(PyLoadOp::Load)
            .with_stencil_store_op(PyStoreOp::Discard)
            .with_depth_clear_value(0.5)
            .with_stencil_clear_value(128)
            .with_depth_read_only(true)
            .with_stencil_read_only(true);

        assert_eq!(attachment.view(), Some(view));
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.depth_clear_value(), 0.5);
        assert_eq!(attachment.stencil_clear_value(), 128);
        assert!(attachment.depth_read_only());
        assert!(attachment.stencil_read_only());
    }

    // -------------------------------------------------------------------------
    // PyTimestampWrites Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timestamp_writes_new() {
        let ts = PyTimestampWrites::new(5, Some(0), Some(1));
        assert_eq!(ts.query_set(), 5);
        assert_eq!(ts.beginning_of_pass_write_index(), Some(0));
        assert_eq!(ts.end_of_pass_write_index(), Some(1));
    }

    #[test]
    fn test_timestamp_writes_full() {
        let ts = PyTimestampWrites::full(3, 10, 11);
        assert_eq!(ts.query_set(), 3);
        assert_eq!(ts.beginning_of_pass_write_index(), Some(10));
        assert_eq!(ts.end_of_pass_write_index(), Some(11));
    }

    #[test]
    fn test_timestamp_writes_beginning_only() {
        let ts = PyTimestampWrites::beginning_only(2, 5);
        assert_eq!(ts.query_set(), 2);
        assert_eq!(ts.beginning_of_pass_write_index(), Some(5));
        assert_eq!(ts.end_of_pass_write_index(), None);
    }

    #[test]
    fn test_timestamp_writes_end_only() {
        let ts = PyTimestampWrites::end_only(1, 8);
        assert_eq!(ts.query_set(), 1);
        assert_eq!(ts.beginning_of_pass_write_index(), None);
        assert_eq!(ts.end_of_pass_write_index(), Some(8));
    }

    // -------------------------------------------------------------------------
    // PyRenderPassDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pass_descriptor_default() {
        let desc = PyRenderPassDescriptor::new();
        assert!(desc.label().is_none());
        assert!(desc.color_attachments().is_empty());
        assert!(desc.depth_stencil_attachment().is_none());
        assert!(desc.occlusion_query_set().is_none());
        assert!(desc.timestamp_writes().is_none());
        assert!(!desc.has_attachments());
        assert!(!desc.has_depth_stencil());
    }

    #[test]
    fn test_render_pass_descriptor_with_label() {
        let desc = PyRenderPassDescriptor::new().with_label("test_pass");
        assert_eq!(desc.label(), Some("test_pass".to_string()));
    }

    #[test]
    fn test_render_pass_descriptor_add_color_attachment() {
        let view = PyTextureView::new(1, None);
        let attachment = PyColorAttachment::for_target(view);

        let desc = PyRenderPassDescriptor::new()
            .add_color_attachment(attachment.clone());

        assert_eq!(desc.color_attachment_count(), 1);
        assert!(desc.has_attachments());
        assert!(desc.get_color_attachment(0).is_some());
        assert!(desc.get_color_attachment(1).is_none());
    }

    #[test]
    fn test_render_pass_descriptor_multiple_color_attachments() {
        let view1 = PyTextureView::new(1, None);
        let view2 = PyTextureView::new(2, None);
        let view3 = PyTextureView::new(3, None);

        let desc = PyRenderPassDescriptor::new()
            .add_color_attachment(PyColorAttachment::for_target(view1))
            .add_color_attachment(PyColorAttachment::for_target(view2))
            .add_color_attachment(PyColorAttachment::for_target(view3));

        assert_eq!(desc.color_attachment_count(), 3);
    }

    #[test]
    fn test_render_pass_descriptor_with_depth_stencil() {
        let view = PyTextureView::new(100, None);
        let attachment = PyDepthStencilAttachment::for_view(view);

        let desc = PyRenderPassDescriptor::new()
            .with_depth_stencil(attachment);

        assert!(desc.has_attachments());
        assert!(desc.has_depth_stencil());
        assert!(desc.depth_stencil_attachment().is_some());
    }

    #[test]
    fn test_render_pass_descriptor_with_occlusion_query() {
        let desc = PyRenderPassDescriptor::new()
            .with_occlusion_query(42);
        assert_eq!(desc.occlusion_query_set(), Some(42));
    }

    #[test]
    fn test_render_pass_descriptor_validation_no_attachments() {
        let desc = PyRenderPassDescriptor::new();
        let error = desc.validate();
        assert!(error.is_some());
        assert!(error.unwrap().contains("at least one"));
    }

    #[test]
    fn test_render_pass_descriptor_validation_with_color() {
        let view = PyTextureView::new(1, None);
        let desc = PyRenderPassDescriptor::new()
            .add_color_attachment(PyColorAttachment::for_target(view));

        assert!(desc.validate().is_none());
    }

    #[test]
    fn test_render_pass_descriptor_validation_with_depth_only() {
        let view = PyTextureView::new(1, None);
        let desc = PyRenderPassDescriptor::new()
            .with_depth_stencil(PyDepthStencilAttachment::for_view(view));

        assert!(desc.validate().is_none());
    }

    // -------------------------------------------------------------------------
    // PyRenderPassBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_new() {
        let builder = PyRenderPassBuilder::new();
        let desc = builder.descriptor();
        assert!(!desc.has_attachments());
    }

    #[test]
    fn test_builder_label() {
        let builder = PyRenderPassBuilder::new().label("my_pass");
        let desc = builder.descriptor();
        assert_eq!(desc.label(), Some("my_pass".to_string()));
    }

    #[test]
    fn test_builder_color_simple() {
        let view = PyTextureView::new(1, None);
        let builder = PyRenderPassBuilder::new()
            .color(view.clone(), None, None, None, None);

        let desc = builder.descriptor();
        assert_eq!(desc.color_attachment_count(), 1);

        let attachment = desc.get_color_attachment(0).unwrap();
        assert_eq!(attachment.target(), Some(view));
        assert_eq!(attachment.load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.store_op(), PyStoreOp::Store);
        assert_eq!(attachment.clear_color(), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_builder_color_with_options() {
        let view = PyTextureView::new(1, None);
        let resolve = PyTextureView::new(2, None);

        let builder = PyRenderPassBuilder::new()
            .color(
                view.clone(),
                Some(PyLoadOp::Load),
                Some(PyStoreOp::Discard),
                Some([0.5, 0.5, 0.5, 1.0]),
                Some(resolve.clone()),
            );

        let desc = builder.descriptor();
        let attachment = desc.get_color_attachment(0).unwrap();

        assert_eq!(attachment.load_op(), PyLoadOp::Load);
        assert_eq!(attachment.store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.clear_color(), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(attachment.resolve_target(), Some(resolve));
    }

    #[test]
    fn test_builder_multiple_colors() {
        let view1 = PyTextureView::new(1, None);
        let view2 = PyTextureView::new(2, None);

        let builder = PyRenderPassBuilder::new()
            .color(view1, None, None, None, None)
            .color(view2, None, None, None, None);

        let desc = builder.descriptor();
        assert_eq!(desc.color_attachment_count(), 2);
    }

    #[test]
    fn test_builder_depth() {
        let view = PyTextureView::new(10, None);
        let builder = PyRenderPassBuilder::new()
            .depth(view.clone(), None, None, None, None);

        let desc = builder.descriptor();
        assert!(desc.has_depth_stencil());

        let attachment = desc.depth_stencil_attachment().unwrap();
        assert_eq!(attachment.view(), Some(view));
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.depth_clear_value(), 1.0);
        assert!(!attachment.depth_read_only());
    }

    #[test]
    fn test_builder_depth_with_options() {
        let view = PyTextureView::new(10, None);
        let builder = PyRenderPassBuilder::new()
            .depth(
                view,
                Some(PyLoadOp::Load),
                Some(PyStoreOp::Discard),
                Some(0.0),
                Some(true),
            );

        let attachment = builder.descriptor().depth_stencil_attachment().unwrap();
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.depth_clear_value(), 0.0);
        assert!(attachment.depth_read_only());
    }

    #[test]
    fn test_builder_stencil() {
        let view = PyTextureView::new(10, None);
        let builder = PyRenderPassBuilder::new()
            .stencil(view.clone(), None, None, None, None);

        let attachment = builder.descriptor().depth_stencil_attachment().unwrap();
        assert_eq!(attachment.view(), Some(view));
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.stencil_clear_value(), 0);
        assert!(!attachment.stencil_read_only());
    }

    #[test]
    fn test_builder_stencil_with_options() {
        let view = PyTextureView::new(10, None);
        let builder = PyRenderPassBuilder::new()
            .stencil(
                view,
                Some(PyLoadOp::Load),
                Some(PyStoreOp::Discard),
                Some(255),
                Some(true),
            );

        let attachment = builder.descriptor().depth_stencil_attachment().unwrap();
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.stencil_clear_value(), 255);
        assert!(attachment.stencil_read_only());
    }

    #[test]
    fn test_builder_depth_and_stencil_combined() {
        let view = PyTextureView::new(10, None);

        // Set depth first, then stencil - stencil should preserve depth settings
        let builder = PyRenderPassBuilder::new()
            .depth(view.clone(), Some(PyLoadOp::Load), Some(PyStoreOp::Store), Some(0.5), Some(true))
            .stencil(view.clone(), Some(PyLoadOp::Clear), Some(PyStoreOp::Discard), Some(100), Some(false));

        let attachment = builder.descriptor().depth_stencil_attachment().unwrap();

        // Depth settings should be preserved
        assert_eq!(attachment.depth_load_op(), PyLoadOp::Load);
        assert_eq!(attachment.depth_store_op(), PyStoreOp::Store);
        assert_eq!(attachment.depth_clear_value(), 0.5);
        assert!(attachment.depth_read_only());

        // Stencil settings should be updated
        assert_eq!(attachment.stencil_load_op(), PyLoadOp::Clear);
        assert_eq!(attachment.stencil_store_op(), PyStoreOp::Discard);
        assert_eq!(attachment.stencil_clear_value(), 100);
        assert!(!attachment.stencil_read_only());
    }

    #[test]
    fn test_builder_occlusion_query() {
        let view = PyTextureView::new(1, None);
        let builder = PyRenderPassBuilder::new()
            .color(view, None, None, None, None)
            .occlusion_query(42);

        let desc = builder.descriptor();
        assert_eq!(desc.occlusion_query_set(), Some(42));
    }

    #[test]
    fn test_builder_timestamps() {
        let view = PyTextureView::new(1, None);
        let ts = PyTimestampWrites::full(0, 0, 1);

        let builder = PyRenderPassBuilder::new()
            .color(view, None, None, None, None)
            .timestamps(ts);

        let desc = builder.descriptor();
        let ts_result = desc.timestamp_writes().unwrap();
        assert_eq!(ts_result.query_set(), 0);
        assert_eq!(ts_result.beginning_of_pass_write_index(), Some(0));
        assert_eq!(ts_result.end_of_pass_write_index(), Some(1));
    }

    #[test]
    fn test_builder_build_success() {
        let view = PyTextureView::new(1, None);

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            let builder = PyRenderPassBuilder::new()
                .label("test")
                .color(view, None, None, None, None);

            let result = builder.build();
            assert!(result.is_ok());

            let desc = result.unwrap();
            assert_eq!(desc.label(), Some("test".to_string()));
            assert_eq!(desc.color_attachment_count(), 1);
        });
    }

    #[test]
    fn test_builder_build_no_attachments_error() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            let builder = PyRenderPassBuilder::new().label("empty");

            let result = builder.build();
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_builder_build_unchecked() {
        let builder = PyRenderPassBuilder::new().label("empty");

        // Should not panic even with no attachments
        let desc = builder.build_unchecked();
        assert!(!desc.has_attachments());
    }

    #[test]
    fn test_builder_fluent_chain() {
        let color_view = PyTextureView::new(1, None);
        let depth_view = PyTextureView::new(2, None);

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            let result = PyRenderPassBuilder::new()
                .label("main_pass")
                .color(color_view.clone(), None, None, Some([0.1, 0.1, 0.1, 1.0]), None)
                .depth(depth_view, None, None, Some(1.0), None)
                .occlusion_query(0)
                .build();

            assert!(result.is_ok());
            let desc = result.unwrap();
            assert_eq!(desc.label(), Some("main_pass".to_string()));
            assert_eq!(desc.color_attachment_count(), 1);
            assert!(desc.has_depth_stencil());
            assert_eq!(desc.occlusion_query_set(), Some(0));
        });
    }

    #[test]
    fn test_builder_color_attachment_method() {
        let view = PyTextureView::new(1, None);
        let attachment = PyColorAttachment::for_target(view)
            .with_clear_color(1.0, 0.0, 0.0, 1.0);

        let builder = PyRenderPassBuilder::new()
            .color_attachment(attachment.clone());

        let desc = builder.descriptor();
        let result_attachment = desc.get_color_attachment(0).unwrap();
        assert_eq!(result_attachment.clear_color(), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_builder_depth_stencil_attachment_method() {
        let view = PyTextureView::new(1, None);
        let attachment = PyDepthStencilAttachment::depth_only(view);

        let builder = PyRenderPassBuilder::new()
            .depth_stencil_attachment(attachment);

        let desc = builder.descriptor();
        assert!(desc.has_depth_stencil());
        let ds = desc.depth_stencil_attachment().unwrap();
        assert!(ds.stencil_read_only());
    }
}
