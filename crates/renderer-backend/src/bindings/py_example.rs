//! Python API examples demonstrating complete wgpu Python bindings (T-WGPU-P7.6.10).
//!
//! This module provides comprehensive examples, quick-start patterns, code snippets,
//! and validation helpers for the TRINITY wgpu Python bindings.
//!
//! # Types
//!
//! - [`PyRendererExample`] - Complete example showing full rendering workflow
//! - [`PyQuickStart`] - Static methods for common rendering patterns
//! - [`PyCodeSnippets`] - Ready-to-use code snippet generators
//! - [`PyValidationHelper`] - Validation utilities for common setup mistakes
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     PyQuickStart, PyCodeSnippets, PyValidationHelper
//! )
//!
//! # Quick start with hello triangle
//! example = PyQuickStart.hello_triangle()
//! example.run()
//!
//! # Generate code snippets
//! vertex_desc = PyCodeSnippets.create_vertex_buffer([0.0, 0.0, 1.0, 0.0, 0.5, 1.0])
//! sampler_desc = PyCodeSnippets.create_sampler_linear()
//!
//! # Validate setup
//! report = PyValidationHelper.check_buffer_alignment(vertex_desc)
//! if not report.is_valid():
//!     print(report.format_issues())
//! ```
//!
//! # Feature Gate
//!
//! All types are gated behind the `pyo3` feature flag:
//!
//! ```toml
//! [features]
//! pyo3 = ["dep:pyo3"]
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::fmt;

use super::py_buffer::{PyBufferDescriptor, PyBufferUsage};
use super::py_resource::{PyResourceHandle, PyResourcePool, PyResourceType};
use super::py_render_pass::{PyRenderPassBuilder, PyRenderPassDescriptor, PyLoadOp, PyStoreOp, PyTextureView, PyColorAttachment};
use super::py_compute_pass::{PyComputePassBuilder, PyComputePassDescriptor, PyDispatchDescriptor};
use super::py_command_batch::{PyCommandEncoder, PyCommandBuffer, PyCommand};
use super::py_error::{PyGpuError, PyValidationReport, PyErrorCategory};

// ============================================================================
// PyTextureFormat
// ============================================================================

/// Texture format enumeration.
///
/// Common texture formats for render targets and textures.
#[pyclass(name = "TextureFormat", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PyTextureFormat {
    /// 8-bit RGBA with sRGB gamma correction.
    #[default]
    Rgba8UnormSrgb = 0,
    /// 8-bit RGBA linear.
    Rgba8Unorm = 1,
    /// 8-bit BGRA with sRGB gamma correction.
    Bgra8UnormSrgb = 2,
    /// 8-bit BGRA linear.
    Bgra8Unorm = 3,
    /// 16-bit RGBA floating point.
    Rgba16Float = 4,
    /// 32-bit RGBA floating point.
    Rgba32Float = 5,
    /// 24-bit depth, 8-bit stencil.
    Depth24PlusStencil8 = 6,
    /// 32-bit depth.
    Depth32Float = 7,
    /// 16-bit depth.
    Depth16Unorm = 8,
}

#[pymethods]
impl PyTextureFormat {
    /// Returns the canonical name of this texture format.
    pub fn name(&self) -> &str {
        match self {
            Self::Rgba8UnormSrgb => "Rgba8UnormSrgb",
            Self::Rgba8Unorm => "Rgba8Unorm",
            Self::Bgra8UnormSrgb => "Bgra8UnormSrgb",
            Self::Bgra8Unorm => "Bgra8Unorm",
            Self::Rgba16Float => "Rgba16Float",
            Self::Rgba32Float => "Rgba32Float",
            Self::Depth24PlusStencil8 => "Depth24PlusStencil8",
            Self::Depth32Float => "Depth32Float",
            Self::Depth16Unorm => "Depth16Unorm",
        }
    }

    /// Returns the byte size per pixel for this format.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::Rgba8UnormSrgb | Self::Rgba8Unorm |
            Self::Bgra8UnormSrgb | Self::Bgra8Unorm => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
            Self::Depth24PlusStencil8 => 4,
            Self::Depth32Float => 4,
            Self::Depth16Unorm => 2,
        }
    }

    /// Returns true if this is a depth format.
    pub fn is_depth(&self) -> bool {
        matches!(
            self,
            Self::Depth24PlusStencil8 | Self::Depth32Float | Self::Depth16Unorm
        )
    }

    /// Returns true if this format has a stencil component.
    pub fn has_stencil(&self) -> bool {
        matches!(self, Self::Depth24PlusStencil8)
    }

    /// Returns true if this is a color format.
    pub fn is_color(&self) -> bool {
        !self.is_depth()
    }

    /// Returns true if this format uses sRGB gamma.
    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    fn __repr__(&self) -> String {
        format!("TextureFormat.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

// ============================================================================
// PyTextureUsage
// ============================================================================

/// Texture usage flags controlling how a texture can be used.
#[pyclass(name = "TextureUsage")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyTextureUsage {
    bits: u32,
}

mod texture_usage_bits {
    pub const COPY_SRC: u32 = 1 << 0;
    pub const COPY_DST: u32 = 1 << 1;
    pub const TEXTURE_BINDING: u32 = 1 << 2;
    pub const STORAGE_BINDING: u32 = 1 << 3;
    pub const RENDER_ATTACHMENT: u32 = 1 << 4;
}

#[pymethods]
impl PyTextureUsage {
    /// Create texture usage from raw bits.
    #[new]
    pub fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Returns the raw bits value.
    #[getter]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// No usage flags.
    #[staticmethod]
    pub fn empty() -> Self {
        Self { bits: 0 }
    }

    /// COPY_SRC: Texture can be copied from.
    #[staticmethod]
    pub fn copy_src() -> Self {
        Self { bits: texture_usage_bits::COPY_SRC }
    }

    /// COPY_DST: Texture can be copied to.
    #[staticmethod]
    pub fn copy_dst() -> Self {
        Self { bits: texture_usage_bits::COPY_DST }
    }

    /// TEXTURE_BINDING: Texture can be bound for sampling.
    #[staticmethod]
    pub fn texture_binding() -> Self {
        Self { bits: texture_usage_bits::TEXTURE_BINDING }
    }

    /// STORAGE_BINDING: Texture can be bound for storage access.
    #[staticmethod]
    pub fn storage_binding() -> Self {
        Self { bits: texture_usage_bits::STORAGE_BINDING }
    }

    /// RENDER_ATTACHMENT: Texture can be used as a render target.
    #[staticmethod]
    pub fn render_attachment() -> Self {
        Self { bits: texture_usage_bits::RENDER_ATTACHMENT }
    }

    /// Preset: render target with sampling capability.
    #[staticmethod]
    pub fn render_target_sampled() -> Self {
        Self {
            bits: texture_usage_bits::RENDER_ATTACHMENT | texture_usage_bits::TEXTURE_BINDING,
        }
    }

    /// Preset: standard texture (copy_dst + texture_binding).
    #[staticmethod]
    pub fn standard() -> Self {
        Self {
            bits: texture_usage_bits::COPY_DST | texture_usage_bits::TEXTURE_BINDING,
        }
    }

    /// Combine usage flags with `|` operator.
    pub fn __or__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Check if contains flag.
    pub fn contains(&self, other: &Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    fn __repr__(&self) -> String {
        format!("TextureUsage(bits={:#x})", self.bits)
    }
}

impl Default for PyTextureUsage {
    fn default() -> Self {
        Self::standard()
    }
}

// ============================================================================
// PyTextureDescriptor
// ============================================================================

/// Texture creation descriptor.
#[pyclass(name = "TextureDescriptor")]
#[derive(Clone, Debug)]
pub struct PyTextureDescriptor {
    /// Optional label for debugging.
    label: Option<String>,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Depth or array layers.
    depth_or_array_layers: u32,
    /// Mipmap level count.
    mip_level_count: u32,
    /// Sample count for multisampling.
    sample_count: u32,
    /// Texture format.
    format: PyTextureFormat,
    /// Usage flags.
    usage: PyTextureUsage,
}

#[pymethods]
impl PyTextureDescriptor {
    /// Create a new texture descriptor.
    #[new]
    #[pyo3(signature = (width, height, format=PyTextureFormat::Rgba8UnormSrgb))]
    pub fn new(width: u32, height: u32, format: PyTextureFormat) -> Self {
        Self {
            label: None,
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: PyTextureUsage::standard(),
        }
    }

    /// Create a render target texture descriptor.
    #[staticmethod]
    #[pyo3(signature = (width, height, format=PyTextureFormat::Rgba8UnormSrgb))]
    pub fn render_target(width: u32, height: u32, format: PyTextureFormat) -> Self {
        Self {
            label: Some("render_target".to_string()),
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: PyTextureUsage::render_target_sampled(),
        }
    }

    /// Create a depth buffer texture descriptor.
    #[staticmethod]
    #[pyo3(signature = (width, height, format=PyTextureFormat::Depth24PlusStencil8))]
    pub fn depth_buffer(width: u32, height: u32, format: PyTextureFormat) -> Self {
        Self {
            label: Some("depth_buffer".to_string()),
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage: PyTextureUsage::render_attachment(),
        }
    }

    /// Set the label.
    pub fn with_label(&self, label: &str) -> Self {
        let mut clone = self.clone();
        clone.label = Some(label.to_string());
        clone
    }

    /// Set the usage flags.
    pub fn with_usage(&self, usage: PyTextureUsage) -> Self {
        let mut clone = self.clone();
        clone.usage = usage;
        clone
    }

    /// Set the mip level count.
    pub fn with_mip_levels(&self, count: u32) -> Self {
        let mut clone = self.clone();
        clone.mip_level_count = count;
        clone
    }

    /// Set the sample count for multisampling.
    pub fn with_sample_count(&self, count: u32) -> Self {
        let mut clone = self.clone();
        clone.sample_count = count;
        clone
    }

    /// Set depth or array layers.
    pub fn with_depth(&self, depth: u32) -> Self {
        let mut clone = self.clone();
        clone.depth_or_array_layers = depth;
        clone
    }

    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    pub fn depth_or_array_layers(&self) -> u32 {
        self.depth_or_array_layers
    }

    #[getter]
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    #[getter]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    #[getter]
    pub fn format(&self) -> PyTextureFormat {
        self.format
    }

    #[getter]
    pub fn usage(&self) -> PyTextureUsage {
        self.usage
    }

    /// Calculate the total byte size of the texture.
    pub fn byte_size(&self) -> u64 {
        let bpp = self.format.bytes_per_pixel() as u64;
        let mut total = 0u64;
        let mut w = self.width as u64;
        let mut h = self.height as u64;
        for _ in 0..self.mip_level_count {
            total += w * h * bpp * self.depth_or_array_layers as u64;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        total
    }

    /// Calculate the maximum mip levels for this texture size.
    pub fn max_mip_levels(&self) -> u32 {
        let max_dim = self.width.max(self.height);
        (32 - max_dim.leading_zeros()).max(1)
    }

    fn __repr__(&self) -> String {
        format!(
            "TextureDescriptor({}x{}, format={:?}, mips={}, samples={})",
            self.width, self.height, self.format, self.mip_level_count, self.sample_count
        )
    }
}

// ============================================================================
// PySamplerDescriptor
// ============================================================================

/// Sampler address mode.
#[pyclass(name = "AddressMode", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PyAddressMode {
    /// Clamp to edge.
    #[default]
    ClampToEdge = 0,
    /// Repeat wrapping.
    Repeat = 1,
    /// Mirror repeat.
    MirrorRepeat = 2,
}

#[pymethods]
impl PyAddressMode {
    pub fn name(&self) -> &str {
        match self {
            Self::ClampToEdge => "ClampToEdge",
            Self::Repeat => "Repeat",
            Self::MirrorRepeat => "MirrorRepeat",
        }
    }

    fn __repr__(&self) -> String {
        format!("AddressMode.{}", self.name())
    }
}

/// Sampler filter mode.
#[pyclass(name = "FilterMode", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PyFilterMode {
    /// Nearest neighbor filtering.
    Nearest = 0,
    /// Linear interpolation.
    #[default]
    Linear = 1,
}

#[pymethods]
impl PyFilterMode {
    pub fn name(&self) -> &str {
        match self {
            Self::Nearest => "Nearest",
            Self::Linear => "Linear",
        }
    }

    fn __repr__(&self) -> String {
        format!("FilterMode.{}", self.name())
    }
}

/// Sampler creation descriptor.
#[pyclass(name = "SamplerDescriptor")]
#[derive(Clone, Debug)]
pub struct PySamplerDescriptor {
    /// Optional label for debugging.
    label: Option<String>,
    /// Address mode for U coordinate.
    address_mode_u: PyAddressMode,
    /// Address mode for V coordinate.
    address_mode_v: PyAddressMode,
    /// Address mode for W coordinate.
    address_mode_w: PyAddressMode,
    /// Magnification filter.
    mag_filter: PyFilterMode,
    /// Minification filter.
    min_filter: PyFilterMode,
    /// Mipmap filter.
    mipmap_filter: PyFilterMode,
    /// LOD min clamp.
    lod_min_clamp: f32,
    /// LOD max clamp.
    lod_max_clamp: f32,
    /// Anisotropy clamp (1 = disabled).
    anisotropy_clamp: u16,
}

#[pymethods]
impl PySamplerDescriptor {
    /// Create a new sampler descriptor with default settings.
    #[new]
    pub fn new() -> Self {
        Self {
            label: None,
            address_mode_u: PyAddressMode::ClampToEdge,
            address_mode_v: PyAddressMode::ClampToEdge,
            address_mode_w: PyAddressMode::ClampToEdge,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: 1,
        }
    }

    /// Create a linear filtering sampler.
    #[staticmethod]
    pub fn linear() -> Self {
        Self {
            label: Some("linear_sampler".to_string()),
            address_mode_u: PyAddressMode::ClampToEdge,
            address_mode_v: PyAddressMode::ClampToEdge,
            address_mode_w: PyAddressMode::ClampToEdge,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: 1,
        }
    }

    /// Create a nearest neighbor filtering sampler.
    #[staticmethod]
    pub fn nearest() -> Self {
        Self {
            label: Some("nearest_sampler".to_string()),
            address_mode_u: PyAddressMode::ClampToEdge,
            address_mode_v: PyAddressMode::ClampToEdge,
            address_mode_w: PyAddressMode::ClampToEdge,
            mag_filter: PyFilterMode::Nearest,
            min_filter: PyFilterMode::Nearest,
            mipmap_filter: PyFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: 1,
        }
    }

    /// Create a repeating texture sampler.
    #[staticmethod]
    pub fn repeat() -> Self {
        Self {
            label: Some("repeat_sampler".to_string()),
            address_mode_u: PyAddressMode::Repeat,
            address_mode_v: PyAddressMode::Repeat,
            address_mode_w: PyAddressMode::Repeat,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: 1,
        }
    }

    /// Create an anisotropic filtering sampler.
    #[staticmethod]
    #[pyo3(signature = (max_anisotropy=16))]
    pub fn anisotropic(max_anisotropy: u16) -> Self {
        Self {
            label: Some("anisotropic_sampler".to_string()),
            address_mode_u: PyAddressMode::Repeat,
            address_mode_v: PyAddressMode::Repeat,
            address_mode_w: PyAddressMode::Repeat,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: max_anisotropy.max(1).min(16),
        }
    }

    /// Set the label.
    pub fn with_label(&self, label: &str) -> Self {
        let mut clone = self.clone();
        clone.label = Some(label.to_string());
        clone
    }

    /// Set address modes for all coordinates.
    pub fn with_address_mode(&self, mode: PyAddressMode) -> Self {
        let mut clone = self.clone();
        clone.address_mode_u = mode;
        clone.address_mode_v = mode;
        clone.address_mode_w = mode;
        clone
    }

    /// Set LOD clamp range.
    pub fn with_lod_clamp(&self, min: f32, max: f32) -> Self {
        let mut clone = self.clone();
        clone.lod_min_clamp = min;
        clone.lod_max_clamp = max;
        clone
    }

    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    #[getter]
    pub fn address_mode_u(&self) -> PyAddressMode {
        self.address_mode_u
    }

    #[getter]
    pub fn address_mode_v(&self) -> PyAddressMode {
        self.address_mode_v
    }

    #[getter]
    pub fn address_mode_w(&self) -> PyAddressMode {
        self.address_mode_w
    }

    #[getter]
    pub fn mag_filter(&self) -> PyFilterMode {
        self.mag_filter
    }

    #[getter]
    pub fn min_filter(&self) -> PyFilterMode {
        self.min_filter
    }

    #[getter]
    pub fn mipmap_filter(&self) -> PyFilterMode {
        self.mipmap_filter
    }

    #[getter]
    pub fn anisotropy_clamp(&self) -> u16 {
        self.anisotropy_clamp
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplerDescriptor(mag={:?}, min={:?}, aniso={})",
            self.mag_filter, self.min_filter, self.anisotropy_clamp
        )
    }
}

impl Default for PySamplerDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PyExampleState
// ============================================================================

/// Internal state for renderer examples.
#[derive(Clone, Debug)]
struct ExampleState {
    /// Resource pool for managing handles.
    resource_pool: PyResourcePool,
    /// List of created buffers.
    buffers: Vec<PyResourceHandle>,
    /// List of created textures.
    textures: Vec<PyResourceHandle>,
    /// List of created samplers.
    samplers: Vec<PyResourceHandle>,
    /// Whether the example is initialized.
    initialized: bool,
}

impl Default for ExampleState {
    fn default() -> Self {
        Self {
            resource_pool: PyResourcePool::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            initialized: false,
        }
    }
}

// ============================================================================
// PyRenderExample
// ============================================================================

/// A complete render example demonstrating the Python API.
#[pyclass(name = "RenderExample")]
#[derive(Clone, Debug)]
pub struct PyRenderExample {
    /// Example name.
    name: String,
    /// Example description.
    description: String,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Internal state.
    state: ExampleState,
    /// Recorded commands.
    commands: Vec<PyCommand>,
}

#[pymethods]
impl PyRenderExample {
    /// Create a new render example.
    #[new]
    #[pyo3(signature = (name, width=800, height=600))]
    pub fn new(name: &str, width: u32, height: u32) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            width,
            height,
            state: ExampleState::default(),
            commands: Vec::new(),
        }
    }

    /// Set the example description.
    pub fn with_description(&self, description: &str) -> Self {
        let mut clone = self.clone();
        clone.description = description.to_string();
        clone
    }

    #[getter]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Initialize the renderer resources.
    pub fn initialize(&mut self) -> PyResult<()> {
        if self.state.initialized {
            return Err(PyValueError::new_err("Example already initialized"));
        }
        self.state.initialized = true;
        Ok(())
    }

    /// Check if the example is initialized.
    pub fn is_initialized(&self) -> bool {
        self.state.initialized
    }

    /// Create a buffer and track it.
    pub fn create_buffer(&mut self, desc: &PyBufferDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.state.resource_pool.allocate(PyResourceType::Buffer);
        self.state.buffers.push(handle.clone());
        Ok(handle)
    }

    /// Create a texture and track it.
    pub fn create_texture(&mut self, desc: &PyTextureDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.state.resource_pool.allocate(PyResourceType::Texture);
        self.state.textures.push(handle.clone());
        Ok(handle)
    }

    /// Create a sampler and track it.
    pub fn create_sampler(&mut self, desc: &PySamplerDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.state.resource_pool.allocate(PyResourceType::Sampler);
        self.state.samplers.push(handle.clone());
        Ok(handle)
    }

    /// Build a render pass descriptor.
    #[pyo3(signature = (color_view, depth_view=None, clear_color=None))]
    pub fn build_render_pass(
        &self,
        color_view: &PyTextureView,
        depth_view: Option<&PyTextureView>,
        clear_color: Option<[f32; 4]>,
    ) -> PyRenderPassDescriptor {
        let mut builder = PyRenderPassBuilder::new();
        builder = builder.label(&format!("{}_pass", self.name));

        let clear = clear_color.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        builder = builder.color(
            color_view.clone(),
            Some(PyLoadOp::Clear),
            Some(PyStoreOp::Store),
            Some(clear),
            None,
        );

        if let Some(depth) = depth_view {
            builder = builder.depth(
                depth.clone(),
                Some(PyLoadOp::Clear),
                Some(PyStoreOp::Store),
                Some(1.0),
                None,
            );
        }

        builder.build()
    }

    /// Record a draw command.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.commands.push(PyCommand::draw(vertex_count, instance_count, 0, 0));
    }

    /// Record a draw indexed command.
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32) {
        self.commands.push(PyCommand::draw_indexed(index_count, instance_count, 0, 0, 0));
    }

    /// Get the number of recorded commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Get the number of created resources.
    pub fn resource_count(&self) -> usize {
        self.state.buffers.len() + self.state.textures.len() + self.state.samplers.len()
    }

    /// Clear all recorded commands.
    pub fn clear_commands(&mut self) {
        self.commands.clear();
    }

    /// Execute the recorded commands (simulated).
    pub fn run(&self) -> PyResult<PyExampleResult> {
        if !self.state.initialized {
            return Err(PyValueError::new_err("Example not initialized"));
        }
        Ok(PyExampleResult {
            success: true,
            commands_executed: self.commands.len(),
            resources_used: self.resource_count(),
            message: format!("Executed {} commands successfully", self.commands.len()),
        })
    }

    /// Clean up all resources.
    pub fn cleanup(&mut self) {
        for handle in self.state.buffers.drain(..) {
            self.state.resource_pool.release(&handle);
        }
        for handle in self.state.textures.drain(..) {
            self.state.resource_pool.release(&handle);
        }
        for handle in self.state.samplers.drain(..) {
            self.state.resource_pool.release(&handle);
        }
        self.commands.clear();
        self.state.initialized = false;
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderExample(name=\"{}\", {}x{}, {} cmds, {} resources)",
            self.name,
            self.width,
            self.height,
            self.commands.len(),
            self.resource_count()
        )
    }
}

// ============================================================================
// PyComputeExample
// ============================================================================

/// A complete compute example demonstrating the Python API.
#[pyclass(name = "ComputeExample")]
#[derive(Clone, Debug)]
pub struct PyComputeExample {
    /// Example name.
    name: String,
    /// Example description.
    description: String,
    /// Internal state.
    state: ExampleState,
    /// Dispatch descriptors.
    dispatches: Vec<PyDispatchDescriptor>,
}

#[pymethods]
impl PyComputeExample {
    /// Create a new compute example.
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            state: ExampleState::default(),
            dispatches: Vec::new(),
        }
    }

    /// Set the example description.
    pub fn with_description(&self, description: &str) -> Self {
        let mut clone = self.clone();
        clone.description = description.to_string();
        clone
    }

    #[getter]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    pub fn description(&self) -> String {
        self.description.clone()
    }

    /// Initialize the compute resources.
    pub fn initialize(&mut self) -> PyResult<()> {
        if self.state.initialized {
            return Err(PyValueError::new_err("Example already initialized"));
        }
        self.state.initialized = true;
        Ok(())
    }

    /// Check if the example is initialized.
    pub fn is_initialized(&self) -> bool {
        self.state.initialized
    }

    /// Create a storage buffer.
    pub fn create_storage_buffer(&mut self, size: u64) -> PyResult<PyResourceHandle> {
        let handle = self.state.resource_pool.allocate(PyResourceType::Buffer);
        self.state.buffers.push(handle.clone());
        Ok(handle)
    }

    /// Build a compute pass descriptor.
    pub fn build_compute_pass(&self) -> PyComputePassDescriptor {
        let builder = PyComputePassBuilder::new();
        builder.label(&format!("{}_compute", self.name)).build()
    }

    /// Record a dispatch command.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.dispatches.push(PyDispatchDescriptor::direct(x, y, z));
    }

    /// Record an indirect dispatch command.
    pub fn dispatch_indirect(&mut self, buffer: &PyResourceHandle, offset: u64) {
        self.dispatches.push(PyDispatchDescriptor::indirect(buffer.clone(), offset));
    }

    /// Get the number of dispatches.
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Execute the compute work (simulated).
    pub fn run(&self) -> PyResult<PyExampleResult> {
        if !self.state.initialized {
            return Err(PyValueError::new_err("Example not initialized"));
        }
        Ok(PyExampleResult {
            success: true,
            commands_executed: self.dispatches.len(),
            resources_used: self.state.buffers.len(),
            message: format!("Executed {} dispatches successfully", self.dispatches.len()),
        })
    }

    /// Clean up all resources.
    pub fn cleanup(&mut self) {
        for handle in self.state.buffers.drain(..) {
            self.state.resource_pool.release(&handle);
        }
        self.dispatches.clear();
        self.state.initialized = false;
    }

    fn __repr__(&self) -> String {
        format!(
            "ComputeExample(name=\"{}\", {} dispatches, {} buffers)",
            self.name,
            self.dispatches.len(),
            self.state.buffers.len()
        )
    }
}

// ============================================================================
// PyExampleResult
// ============================================================================

/// Result from running an example.
#[pyclass(name = "ExampleResult")]
#[derive(Clone, Debug)]
pub struct PyExampleResult {
    /// Whether execution succeeded.
    success: bool,
    /// Number of commands executed.
    commands_executed: usize,
    /// Number of resources used.
    resources_used: usize,
    /// Result message.
    message: String,
}

#[pymethods]
impl PyExampleResult {
    #[getter]
    pub fn success(&self) -> bool {
        self.success
    }

    #[getter]
    pub fn commands_executed(&self) -> usize {
        self.commands_executed
    }

    #[getter]
    pub fn resources_used(&self) -> usize {
        self.resources_used
    }

    #[getter]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExampleResult(success={}, cmds={}, resources={})",
            self.success, self.commands_executed, self.resources_used
        )
    }

    fn __bool__(&self) -> bool {
        self.success
    }
}

// ============================================================================
// PyQuickStart
// ============================================================================

/// Quick-start patterns for common rendering scenarios.
///
/// Provides static factory methods to create pre-configured examples
/// demonstrating common use cases.
#[pyclass(name = "QuickStart")]
pub struct PyQuickStart;

#[pymethods]
impl PyQuickStart {
    /// Create a hello triangle example.
    ///
    /// Demonstrates basic rendering with:
    /// - Vertex buffer with triangle positions
    /// - Color attachment
    /// - Single draw call
    #[staticmethod]
    pub fn hello_triangle() -> PyRenderExample {
        let mut example = PyRenderExample::new("hello_triangle", 800, 600);
        example.description = "Basic triangle rendering demonstration".to_string();
        example
    }

    /// Create a compute shader example.
    ///
    /// Demonstrates compute shader usage with:
    /// - Storage buffers for input/output
    /// - Direct dispatch
    #[staticmethod]
    pub fn compute_shader() -> PyComputeExample {
        let mut example = PyComputeExample::new("compute_shader");
        example.description = "Basic compute shader demonstration".to_string();
        example
    }

    /// Create a texture sampling example.
    ///
    /// Demonstrates texture sampling with:
    /// - Texture creation
    /// - Sampler configuration
    /// - Textured quad rendering
    #[staticmethod]
    pub fn texture_sampling() -> PyRenderExample {
        let mut example = PyRenderExample::new("texture_sampling", 800, 600);
        example.description = "Texture sampling demonstration".to_string();
        example
    }

    /// Create a post-processing example.
    ///
    /// Demonstrates post-processing with:
    /// - Multiple render passes
    /// - Render-to-texture
    /// - Full-screen quad
    #[staticmethod]
    pub fn post_process() -> PyRenderExample {
        let mut example = PyRenderExample::new("post_process", 1920, 1080);
        example.description = "Post-processing effects demonstration".to_string();
        example
    }

    /// Create a particle system compute example.
    ///
    /// Demonstrates GPU particles with:
    /// - Storage buffers for particle data
    /// - Compute shader updates
    /// - Large dispatch counts
    #[staticmethod]
    pub fn particle_system() -> PyComputeExample {
        let mut example = PyComputeExample::new("particle_system");
        example.description = "GPU particle system demonstration".to_string();
        example
    }

    /// Create a shadow mapping example.
    ///
    /// Demonstrates shadow mapping with:
    /// - Depth-only pass
    /// - Shadow map texture
    /// - Comparison sampler
    #[staticmethod]
    pub fn shadow_mapping() -> PyRenderExample {
        let mut example = PyRenderExample::new("shadow_mapping", 2048, 2048);
        example.description = "Shadow mapping demonstration".to_string();
        example
    }

    /// Create a deferred rendering example.
    ///
    /// Demonstrates deferred shading with:
    /// - G-buffer pass
    /// - Lighting pass
    /// - Multiple render targets
    #[staticmethod]
    pub fn deferred_rendering() -> PyRenderExample {
        let mut example = PyRenderExample::new("deferred_rendering", 1920, 1080);
        example.description = "Deferred rendering demonstration".to_string();
        example
    }
}

// ============================================================================
// PyCodeSnippets
// ============================================================================

/// Collection of ready-to-use code snippets.
///
/// Provides static methods to generate common descriptor configurations
/// for buffers, textures, and samplers.
#[pyclass(name = "CodeSnippets")]
pub struct PyCodeSnippets;

#[pymethods]
impl PyCodeSnippets {
    /// Create a vertex buffer descriptor from float data.
    ///
    /// # Arguments
    /// * `data` - Vertex data as a list of floats
    #[staticmethod]
    pub fn create_vertex_buffer(data: Vec<f32>) -> PyBufferDescriptor {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        PyBufferDescriptor::vertex(size).with_label("vertex_buffer")
    }

    /// Create a uniform buffer descriptor.
    ///
    /// # Arguments
    /// * `size` - Buffer size in bytes (will be aligned to 256 bytes)
    #[staticmethod]
    pub fn create_uniform_buffer(size: u64) -> PyBufferDescriptor {
        let aligned_size = ((size + 255) / 256) * 256;
        PyBufferDescriptor::uniform(aligned_size).with_label("uniform_buffer")
    }

    /// Create a storage buffer descriptor.
    ///
    /// # Arguments
    /// * `size` - Buffer size in bytes
    #[staticmethod]
    pub fn create_storage_buffer(size: u64) -> PyBufferDescriptor {
        PyBufferDescriptor::storage(size).with_label("storage_buffer")
    }

    /// Create an index buffer descriptor.
    ///
    /// # Arguments
    /// * `index_count` - Number of indices
    /// * `use_32bit` - Whether to use 32-bit indices (default: false for 16-bit)
    #[staticmethod]
    #[pyo3(signature = (index_count, use_32bit=false))]
    pub fn create_index_buffer(index_count: u64, use_32bit: bool) -> PyBufferDescriptor {
        let index_size = if use_32bit { 4 } else { 2 };
        let size = index_count * index_size;
        PyBufferDescriptor::index(size).with_label("index_buffer")
    }

    /// Create a render target texture descriptor.
    ///
    /// # Arguments
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    #[staticmethod]
    pub fn create_render_target(width: u32, height: u32) -> PyTextureDescriptor {
        PyTextureDescriptor::render_target(width, height, PyTextureFormat::Rgba8UnormSrgb)
    }

    /// Create a depth buffer texture descriptor.
    ///
    /// # Arguments
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    #[staticmethod]
    pub fn create_depth_buffer(width: u32, height: u32) -> PyTextureDescriptor {
        PyTextureDescriptor::depth_buffer(width, height, PyTextureFormat::Depth24PlusStencil8)
    }

    /// Create a color texture descriptor for sampling.
    ///
    /// # Arguments
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `srgb` - Whether to use sRGB format (default: true)
    #[staticmethod]
    #[pyo3(signature = (width, height, srgb=true))]
    pub fn create_color_texture(width: u32, height: u32, srgb: bool) -> PyTextureDescriptor {
        let format = if srgb {
            PyTextureFormat::Rgba8UnormSrgb
        } else {
            PyTextureFormat::Rgba8Unorm
        };
        PyTextureDescriptor::new(width, height, format).with_label("color_texture")
    }

    /// Create a linear filtering sampler descriptor.
    #[staticmethod]
    pub fn create_sampler_linear() -> PySamplerDescriptor {
        PySamplerDescriptor::linear()
    }

    /// Create a nearest neighbor filtering sampler descriptor.
    #[staticmethod]
    pub fn create_sampler_nearest() -> PySamplerDescriptor {
        PySamplerDescriptor::nearest()
    }

    /// Create a repeating texture sampler descriptor.
    #[staticmethod]
    pub fn create_sampler_repeat() -> PySamplerDescriptor {
        PySamplerDescriptor::repeat()
    }

    /// Create an anisotropic filtering sampler descriptor.
    ///
    /// # Arguments
    /// * `max_anisotropy` - Maximum anisotropy level (1-16)
    #[staticmethod]
    #[pyo3(signature = (max_anisotropy=16))]
    pub fn create_sampler_anisotropic(max_anisotropy: u16) -> PySamplerDescriptor {
        PySamplerDescriptor::anisotropic(max_anisotropy)
    }

    /// Create a shadow map sampler descriptor.
    ///
    /// Configured for depth comparison with clamp-to-edge addressing.
    #[staticmethod]
    pub fn create_sampler_shadow() -> PySamplerDescriptor {
        PySamplerDescriptor {
            label: Some("shadow_sampler".to_string()),
            address_mode_u: PyAddressMode::ClampToEdge,
            address_mode_v: PyAddressMode::ClampToEdge,
            address_mode_w: PyAddressMode::ClampToEdge,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            anisotropy_clamp: 1,
        }
    }
}

// ============================================================================
// PyExampleValidationReport
// ============================================================================

/// Validation report for example setup.
#[pyclass(name = "ExampleValidationReport")]
#[derive(Clone, Debug, Default)]
pub struct PyExampleValidationReport {
    /// List of errors.
    errors: Vec<String>,
    /// List of warnings.
    warnings: Vec<String>,
    /// List of info messages.
    info: Vec<String>,
}

#[pymethods]
impl PyExampleValidationReport {
    /// Create a new empty validation report.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a successful validation report.
    #[staticmethod]
    pub fn ok() -> Self {
        Self::default()
    }

    /// Create a report with a single error.
    #[staticmethod]
    pub fn error(message: &str) -> Self {
        Self {
            errors: vec![message.to_string()],
            warnings: Vec::new(),
            info: Vec::new(),
        }
    }

    /// Create a report with a single warning.
    #[staticmethod]
    pub fn warning(message: &str) -> Self {
        Self {
            errors: Vec::new(),
            warnings: vec![message.to_string()],
            info: Vec::new(),
        }
    }

    /// Add an error to the report.
    pub fn add_error(&mut self, message: &str) {
        self.errors.push(message.to_string());
    }

    /// Add a warning to the report.
    pub fn add_warning(&mut self, message: &str) {
        self.warnings.push(message.to_string());
    }

    /// Add an info message to the report.
    pub fn add_info(&mut self, message: &str) {
        self.info.push(message.to_string());
    }

    /// Returns true if there are no errors.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns true if there are errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns true if there are warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get the list of errors.
    #[getter]
    pub fn errors(&self) -> Vec<String> {
        self.errors.clone()
    }

    /// Get the list of warnings.
    #[getter]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Get the list of info messages.
    #[getter]
    pub fn info_messages(&self) -> Vec<String> {
        self.info.clone()
    }

    /// Get the total number of issues.
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Format the report as a string.
    pub fn format_issues(&self) -> String {
        let mut result = String::new();

        if self.errors.is_empty() && self.warnings.is_empty() {
            return "Validation passed: No issues found.".to_string();
        }

        if !self.errors.is_empty() {
            result.push_str(&format!("Errors ({}):\n", self.errors.len()));
            for (i, error) in self.errors.iter().enumerate() {
                result.push_str(&format!("  {}. {}\n", i + 1, error));
            }
        }

        if !self.warnings.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&format!("Warnings ({}):\n", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                result.push_str(&format!("  {}. {}\n", i + 1, warning));
            }
        }

        if !self.info.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&format!("Info ({}):\n", self.info.len()));
            for (i, info) in self.info.iter().enumerate() {
                result.push_str(&format!("  {}. {}\n", i + 1, info));
            }
        }

        result
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: &PyExampleValidationReport) {
        self.errors.extend(other.errors.clone());
        self.warnings.extend(other.warnings.clone());
        self.info.extend(other.info.clone());
    }

    fn __repr__(&self) -> String {
        format!(
            "ExampleValidationReport(errors={}, warnings={}, valid={})",
            self.errors.len(),
            self.warnings.len(),
            self.is_valid()
        )
    }

    fn __bool__(&self) -> bool {
        self.is_valid()
    }
}

// ============================================================================
// PyValidationHelper
// ============================================================================

/// Validation utilities for common setup mistakes.
///
/// Provides static methods to validate descriptor configurations
/// before resource creation.
#[pyclass(name = "ValidationHelper")]
pub struct PyValidationHelper;

#[pymethods]
impl PyValidationHelper {
    /// Check buffer alignment requirements.
    ///
    /// Validates that uniform buffers are properly aligned to 256 bytes.
    #[staticmethod]
    pub fn check_buffer_alignment(desc: &PyBufferDescriptor) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        let size = desc.size();

        // Check uniform buffer alignment
        if desc.usage().contains(&PyBufferUsage::uniform()) {
            if size % 256 != 0 {
                report.add_error(&format!(
                    "Uniform buffer size {} is not aligned to 256 bytes. Consider using {}.",
                    size,
                    ((size + 255) / 256) * 256
                ));
            }
        }

        // Check minimum size
        if size == 0 {
            report.add_error("Buffer size cannot be zero");
        }

        // Check for very large buffers
        const MAX_RECOMMENDED_SIZE: u64 = 256 * 1024 * 1024; // 256MB
        if size > MAX_RECOMMENDED_SIZE {
            report.add_warning(&format!(
                "Buffer size {} exceeds recommended maximum of {} bytes",
                size, MAX_RECOMMENDED_SIZE
            ));
        }

        // Info about copy_dst for CPU writes
        if !desc.usage().contains(&PyBufferUsage::copy_dst()) {
            if desc.usage().contains(&PyBufferUsage::vertex())
                || desc.usage().contains(&PyBufferUsage::index())
                || desc.usage().contains(&PyBufferUsage::uniform())
            {
                report.add_info(
                    "Buffer does not have COPY_DST usage. CPU writes will require staging buffer."
                );
            }
        }

        report
    }

    /// Check texture format compatibility.
    ///
    /// Validates texture configuration for common issues.
    #[staticmethod]
    pub fn check_texture_format(desc: &PyTextureDescriptor) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        // Check dimensions
        if desc.width() == 0 || desc.height() == 0 {
            report.add_error("Texture dimensions cannot be zero");
        }

        // Check power of two for mipmaps
        if desc.mip_level_count() > 1 {
            let is_pot_width = desc.width().is_power_of_two();
            let is_pot_height = desc.height().is_power_of_two();
            if !is_pot_width || !is_pot_height {
                report.add_warning(
                    "Non-power-of-two dimensions may have issues with mipmap generation on some hardware"
                );
            }
        }

        // Check mip level count
        let max_mips = desc.max_mip_levels();
        if desc.mip_level_count() > max_mips {
            report.add_error(&format!(
                "Mip level count {} exceeds maximum {} for {}x{} texture",
                desc.mip_level_count(),
                max_mips,
                desc.width(),
                desc.height()
            ));
        }

        // Check sample count
        let valid_sample_counts = [1, 4];
        if !valid_sample_counts.contains(&desc.sample_count()) {
            report.add_warning(&format!(
                "Sample count {} may not be supported on all hardware. Common values: 1, 4",
                desc.sample_count()
            ));
        }

        // Check render attachment usage for depth formats
        if desc.format().is_depth() && !desc.usage().contains(&PyTextureUsage::render_attachment()) {
            report.add_warning(
                "Depth format texture without RENDER_ATTACHMENT usage - did you mean to use this as a depth buffer?"
            );
        }

        // Check texture binding usage for color formats in non-render targets
        if desc.format().is_color()
            && desc.usage().contains(&PyTextureUsage::render_attachment())
            && !desc.usage().contains(&PyTextureUsage::texture_binding())
        {
            report.add_info(
                "Render target without TEXTURE_BINDING - add if you need to sample this texture later"
            );
        }

        report
    }

    /// Check render pass configuration.
    ///
    /// Validates render pass descriptor for common issues.
    #[staticmethod]
    pub fn check_render_pass(desc: &PyRenderPassDescriptor) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        // Check for color attachments
        if desc.color_attachment_count() == 0 {
            report.add_warning(
                "Render pass has no color attachments. This is valid for depth-only passes."
            );
        }

        // Check attachment count limits
        const MAX_COLOR_ATTACHMENTS: usize = 8;
        if desc.color_attachment_count() > MAX_COLOR_ATTACHMENTS {
            report.add_error(&format!(
                "Render pass has {} color attachments, but maximum is {}",
                desc.color_attachment_count(),
                MAX_COLOR_ATTACHMENTS
            ));
        }

        report
    }

    /// Check compute pass configuration.
    ///
    /// Validates compute pass descriptor for common issues.
    #[staticmethod]
    pub fn check_compute_pass(desc: &PyComputePassDescriptor) -> PyExampleValidationReport {
        let report = PyExampleValidationReport::new();
        // Compute passes have fewer validation requirements
        // Just return OK for now
        report
    }

    /// Check dispatch dimensions.
    ///
    /// Validates dispatch descriptor for common issues.
    #[staticmethod]
    pub fn check_dispatch(dispatch: &PyDispatchDescriptor) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        if dispatch.is_direct() {
            let (x, y, z) = dispatch.workgroups();

            // Check for zero dimensions
            if x == 0 || y == 0 || z == 0 {
                report.add_error("Dispatch dimensions cannot be zero");
            }

            // Check for exceeding limits
            const MAX_WORKGROUPS_PER_DIM: u32 = 65535;
            if x > MAX_WORKGROUPS_PER_DIM || y > MAX_WORKGROUPS_PER_DIM || z > MAX_WORKGROUPS_PER_DIM {
                report.add_error(&format!(
                    "Dispatch dimension exceeds maximum of {} per dimension",
                    MAX_WORKGROUPS_PER_DIM
                ));
            }

            // Warn about very large dispatches
            let total_workgroups = x as u64 * y as u64 * z as u64;
            const WARN_THRESHOLD: u64 = 1_000_000;
            if total_workgroups > WARN_THRESHOLD {
                report.add_warning(&format!(
                    "Large dispatch with {} total workgroups. Consider breaking into smaller dispatches.",
                    total_workgroups
                ));
            }
        }

        report
    }

    /// Check sampler configuration.
    ///
    /// Validates sampler descriptor for common issues.
    #[staticmethod]
    pub fn check_sampler(desc: &PySamplerDescriptor) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        // Check anisotropy range
        if desc.anisotropy_clamp() > 16 {
            report.add_error("Anisotropy clamp cannot exceed 16");
        }

        // Check LOD clamp range
        if desc.lod_min_clamp > desc.lod_max_clamp {
            report.add_error("LOD min clamp cannot be greater than LOD max clamp");
        }

        // Info about anisotropic filtering requirements
        if desc.anisotropy_clamp() > 1 {
            if desc.mag_filter() != PyFilterMode::Linear || desc.min_filter() != PyFilterMode::Linear {
                report.add_warning(
                    "Anisotropic filtering typically requires linear mag and min filters"
                );
            }
        }

        report
    }

    /// Validate a complete render example setup.
    ///
    /// Runs all relevant validations for a render example.
    #[staticmethod]
    pub fn validate_render_example(example: &PyRenderExample) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        if !example.is_initialized() {
            report.add_warning("Example is not initialized - call initialize() before running");
        }

        if example.command_count() == 0 {
            report.add_info("Example has no recorded commands");
        }

        if example.resource_count() == 0 {
            report.add_info("Example has no created resources");
        }

        // Check dimensions
        if example.width() == 0 || example.height() == 0 {
            report.add_error("Example dimensions cannot be zero");
        }

        report
    }

    /// Validate a complete compute example setup.
    ///
    /// Runs all relevant validations for a compute example.
    #[staticmethod]
    pub fn validate_compute_example(example: &PyComputeExample) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        if !example.is_initialized() {
            report.add_warning("Example is not initialized - call initialize() before running");
        }

        if example.dispatch_count() == 0 {
            report.add_info("Example has no dispatches recorded");
        }

        report
    }
}

// ============================================================================
// PyRendererExample (Main comprehensive example)
// ============================================================================

/// A comprehensive renderer example demonstrating the complete Python API.
///
/// This example shows the full workflow:
/// 1. Initialize renderer
/// 2. Create resources (buffers, textures, samplers)
/// 3. Build render/compute passes
/// 4. Record commands
/// 5. Submit for execution
#[pyclass(name = "RendererExample")]
#[derive(Clone, Debug)]
pub struct PyRendererExample {
    /// Example name.
    name: String,
    /// Screen width.
    width: u32,
    /// Screen height.
    height: u32,
    /// Resource pool.
    resource_pool: PyResourcePool,
    /// Created buffers with descriptors.
    buffers: Vec<(PyResourceHandle, PyBufferDescriptor)>,
    /// Created textures with descriptors.
    textures: Vec<(PyResourceHandle, PyTextureDescriptor)>,
    /// Created samplers with descriptors.
    samplers: Vec<(PyResourceHandle, PySamplerDescriptor)>,
    /// Recorded render passes.
    render_passes: Vec<PyRenderPassDescriptor>,
    /// Recorded compute passes.
    compute_passes: Vec<PyComputePassDescriptor>,
    /// Recorded commands.
    commands: Vec<PyCommand>,
    /// Whether initialized.
    initialized: bool,
}

#[pymethods]
impl PyRendererExample {
    /// Create a new comprehensive renderer example.
    #[new]
    #[pyo3(signature = (name, width=1920, height=1080))]
    pub fn new(name: &str, width: u32, height: u32) -> Self {
        Self {
            name: name.to_string(),
            width,
            height,
            resource_pool: PyResourcePool::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            render_passes: Vec::new(),
            compute_passes: Vec::new(),
            commands: Vec::new(),
            initialized: false,
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Initialize the renderer.
    pub fn initialize(&mut self) -> PyResult<()> {
        if self.initialized {
            return Err(PyValueError::new_err("Already initialized"));
        }
        self.initialized = true;
        Ok(())
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // -- Resource Creation --

    /// Create a buffer.
    pub fn create_buffer(&mut self, desc: PyBufferDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.resource_pool.allocate(PyResourceType::Buffer);
        self.buffers.push((handle.clone(), desc));
        Ok(handle)
    }

    /// Create a texture.
    pub fn create_texture(&mut self, desc: PyTextureDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.resource_pool.allocate(PyResourceType::Texture);
        self.textures.push((handle.clone(), desc));
        Ok(handle)
    }

    /// Create a sampler.
    pub fn create_sampler(&mut self, desc: PySamplerDescriptor) -> PyResult<PyResourceHandle> {
        let handle = self.resource_pool.allocate(PyResourceType::Sampler);
        self.samplers.push((handle.clone(), desc));
        Ok(handle)
    }

    // -- Pass Building --

    /// Build and record a render pass.
    pub fn build_render_pass(&mut self, builder: &PyRenderPassBuilder) -> PyRenderPassDescriptor {
        let pass = builder.build();
        self.render_passes.push(pass.clone());
        pass
    }

    /// Build and record a compute pass.
    pub fn build_compute_pass(&mut self, builder: &PyComputePassBuilder) -> PyComputePassDescriptor {
        let pass = builder.build();
        self.compute_passes.push(pass.clone());
        pass
    }

    // -- Command Recording --

    /// Record a draw command.
    #[pyo3(signature = (vertex_count, instance_count=1))]
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.commands.push(PyCommand::draw(vertex_count, instance_count, 0, 0));
    }

    /// Record a draw indexed command.
    #[pyo3(signature = (index_count, instance_count=1))]
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32) {
        self.commands.push(PyCommand::draw_indexed(index_count, instance_count, 0, 0, 0));
    }

    /// Record a compute dispatch.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.commands.push(PyCommand::dispatch(x, y, z));
    }

    /// Record a set pipeline command.
    pub fn set_pipeline(&mut self, pipeline: &PyResourceHandle) {
        self.commands.push(PyCommand::set_pipeline(pipeline.clone()));
    }

    /// Record a set bind group command.
    #[pyo3(signature = (index, bind_group, dynamic_offsets=None))]
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &PyResourceHandle,
        dynamic_offsets: Option<Vec<u32>>,
    ) {
        self.commands.push(PyCommand::set_bind_group(
            index,
            bind_group.clone(),
            dynamic_offsets.unwrap_or_default(),
        ));
    }

    // -- Execution --

    /// Submit recorded commands for execution.
    pub fn submit(&self) -> PyResult<PyExampleResult> {
        if !self.initialized {
            return Err(PyValueError::new_err("Not initialized"));
        }

        Ok(PyExampleResult {
            success: true,
            commands_executed: self.commands.len(),
            resources_used: self.buffers.len() + self.textures.len() + self.samplers.len(),
            message: format!(
                "Submitted {} commands with {} render passes and {} compute passes",
                self.commands.len(),
                self.render_passes.len(),
                self.compute_passes.len()
            ),
        })
    }

    // -- Stats --

    /// Get the number of buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Get the number of textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Get the number of samplers.
    pub fn sampler_count(&self) -> usize {
        self.samplers.len()
    }

    /// Get the number of render passes.
    pub fn render_pass_count(&self) -> usize {
        self.render_passes.len()
    }

    /// Get the number of compute passes.
    pub fn compute_pass_count(&self) -> usize {
        self.compute_passes.len()
    }

    /// Get the number of commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Get total estimated memory usage.
    pub fn estimated_memory_bytes(&self) -> u64 {
        let buffer_size: u64 = self.buffers.iter().map(|(_, d)| d.size()).sum();
        let texture_size: u64 = self.textures.iter().map(|(_, d)| d.byte_size()).sum();
        buffer_size + texture_size
    }

    // -- Cleanup --

    /// Clean up all resources.
    pub fn cleanup(&mut self) {
        for (handle, _) in self.buffers.drain(..) {
            self.resource_pool.release(&handle);
        }
        for (handle, _) in self.textures.drain(..) {
            self.resource_pool.release(&handle);
        }
        for (handle, _) in self.samplers.drain(..) {
            self.resource_pool.release(&handle);
        }
        self.render_passes.clear();
        self.compute_passes.clear();
        self.commands.clear();
        self.initialized = false;
    }

    /// Validate the current example setup.
    pub fn validate(&self) -> PyExampleValidationReport {
        let mut report = PyExampleValidationReport::new();

        if !self.initialized {
            report.add_warning("Example not initialized");
        }

        // Validate all buffers
        for (_, desc) in &self.buffers {
            let buffer_report = PyValidationHelper::check_buffer_alignment(desc);
            report.merge(&buffer_report);
        }

        // Validate all textures
        for (_, desc) in &self.textures {
            let texture_report = PyValidationHelper::check_texture_format(desc);
            report.merge(&texture_report);
        }

        // Validate all samplers
        for (_, desc) in &self.samplers {
            let sampler_report = PyValidationHelper::check_sampler(desc);
            report.merge(&sampler_report);
        }

        // Validate render passes
        for pass in &self.render_passes {
            let pass_report = PyValidationHelper::check_render_pass(pass);
            report.merge(&pass_report);
        }

        report
    }

    fn __repr__(&self) -> String {
        format!(
            "RendererExample(\"{}\", {}x{}, {} buffers, {} textures, {} cmds)",
            self.name,
            self.width,
            self.height,
            self.buffers.len(),
            self.textures.len(),
            self.commands.len()
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register all example types with the Python module.
pub fn register_module(
    py: pyo3::Python<'_>,
    parent: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    // Texture/Sampler types
    parent.add_class::<PyTextureFormat>()?;
    parent.add_class::<PyTextureUsage>()?;
    parent.add_class::<PyTextureDescriptor>()?;
    parent.add_class::<PyAddressMode>()?;
    parent.add_class::<PyFilterMode>()?;
    parent.add_class::<PySamplerDescriptor>()?;

    // Example types
    parent.add_class::<PyRenderExample>()?;
    parent.add_class::<PyComputeExample>()?;
    parent.add_class::<PyExampleResult>()?;
    parent.add_class::<PyRendererExample>()?;

    // Validation
    parent.add_class::<PyExampleValidationReport>()?;

    // Static helper classes
    parent.add_class::<PyQuickStart>()?;
    parent.add_class::<PyCodeSnippets>()?;
    parent.add_class::<PyValidationHelper>()?;

    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Texture Format Tests --

    #[test]
    fn test_texture_format_properties() {
        assert_eq!(PyTextureFormat::Rgba8UnormSrgb.bytes_per_pixel(), 4);
        assert_eq!(PyTextureFormat::Rgba16Float.bytes_per_pixel(), 8);
        assert_eq!(PyTextureFormat::Rgba32Float.bytes_per_pixel(), 16);

        assert!(PyTextureFormat::Depth24PlusStencil8.is_depth());
        assert!(PyTextureFormat::Depth32Float.is_depth());
        assert!(!PyTextureFormat::Rgba8Unorm.is_depth());

        assert!(PyTextureFormat::Depth24PlusStencil8.has_stencil());
        assert!(!PyTextureFormat::Depth32Float.has_stencil());

        assert!(PyTextureFormat::Rgba8UnormSrgb.is_srgb());
        assert!(!PyTextureFormat::Rgba8Unorm.is_srgb());
    }

    // -- Texture Usage Tests --

    #[test]
    fn test_texture_usage_combinations() {
        let usage1 = PyTextureUsage::copy_src();
        let usage2 = PyTextureUsage::copy_dst();
        let combined = usage1.__or__(&usage2);

        assert!(combined.contains(&usage1));
        assert!(combined.contains(&usage2));
        assert!(!combined.contains(&PyTextureUsage::render_attachment()));
    }

    #[test]
    fn test_texture_usage_presets() {
        let render_sampled = PyTextureUsage::render_target_sampled();
        assert!(render_sampled.contains(&PyTextureUsage::render_attachment()));
        assert!(render_sampled.contains(&PyTextureUsage::texture_binding()));
    }

    // -- Texture Descriptor Tests --

    #[test]
    fn test_texture_descriptor_creation() {
        let desc = PyTextureDescriptor::new(1024, 768, PyTextureFormat::Rgba8UnormSrgb);
        assert_eq!(desc.width(), 1024);
        assert_eq!(desc.height(), 768);
        assert_eq!(desc.mip_level_count(), 1);
        assert_eq!(desc.sample_count(), 1);
    }

    #[test]
    fn test_texture_descriptor_render_target() {
        let desc = PyTextureDescriptor::render_target(1920, 1080, PyTextureFormat::Rgba16Float);
        assert_eq!(desc.width(), 1920);
        assert_eq!(desc.height(), 1080);
        assert!(desc.usage().contains(&PyTextureUsage::render_attachment()));
    }

    #[test]
    fn test_texture_descriptor_byte_size() {
        let desc = PyTextureDescriptor::new(256, 256, PyTextureFormat::Rgba8UnormSrgb);
        assert_eq!(desc.byte_size(), 256 * 256 * 4);

        // Test with mipmaps
        let mipmapped = desc.with_mip_levels(5);
        // Mip sizes: 256x256 + 128x128 + 64x64 + 32x32 + 16x16
        let expected = (256 * 256 + 128 * 128 + 64 * 64 + 32 * 32 + 16 * 16) * 4;
        assert_eq!(mipmapped.byte_size(), expected as u64);
    }

    #[test]
    fn test_texture_descriptor_max_mip_levels() {
        let desc = PyTextureDescriptor::new(256, 256, PyTextureFormat::Rgba8Unorm);
        assert_eq!(desc.max_mip_levels(), 9); // log2(256) + 1

        let desc2 = PyTextureDescriptor::new(1024, 512, PyTextureFormat::Rgba8Unorm);
        assert_eq!(desc2.max_mip_levels(), 11); // log2(1024) + 1
    }

    // -- Sampler Descriptor Tests --

    #[test]
    fn test_sampler_linear() {
        let sampler = PySamplerDescriptor::linear();
        assert_eq!(sampler.mag_filter(), PyFilterMode::Linear);
        assert_eq!(sampler.min_filter(), PyFilterMode::Linear);
        assert_eq!(sampler.anisotropy_clamp(), 1);
    }

    #[test]
    fn test_sampler_nearest() {
        let sampler = PySamplerDescriptor::nearest();
        assert_eq!(sampler.mag_filter(), PyFilterMode::Nearest);
        assert_eq!(sampler.min_filter(), PyFilterMode::Nearest);
    }

    #[test]
    fn test_sampler_anisotropic() {
        let sampler = PySamplerDescriptor::anisotropic(8);
        assert_eq!(sampler.anisotropy_clamp(), 8);

        // Test clamping
        let sampler_max = PySamplerDescriptor::anisotropic(32);
        assert_eq!(sampler_max.anisotropy_clamp(), 16);
    }

    // -- Quick Start Tests --

    #[test]
    fn test_quick_start_hello_triangle() {
        let example = PyQuickStart::hello_triangle();
        assert_eq!(example.name(), "hello_triangle");
        assert_eq!(example.width(), 800);
        assert_eq!(example.height(), 600);
    }

    #[test]
    fn test_quick_start_compute_shader() {
        let example = PyQuickStart::compute_shader();
        assert_eq!(example.name(), "compute_shader");
    }

    #[test]
    fn test_quick_start_post_process() {
        let example = PyQuickStart::post_process();
        assert_eq!(example.name(), "post_process");
        assert_eq!(example.width(), 1920);
        assert_eq!(example.height(), 1080);
    }

    // -- Code Snippets Tests --

    #[test]
    fn test_code_snippets_vertex_buffer() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0, 0.5, 1.0];
        let desc = PyCodeSnippets::create_vertex_buffer(data);
        assert_eq!(desc.size(), 24); // 6 floats * 4 bytes
        assert!(desc.usage().contains(&PyBufferUsage::vertex()));
    }

    #[test]
    fn test_code_snippets_uniform_buffer() {
        let desc = PyCodeSnippets::create_uniform_buffer(100);
        assert_eq!(desc.size(), 256); // Aligned to 256
        assert!(desc.usage().contains(&PyBufferUsage::uniform()));

        let desc2 = PyCodeSnippets::create_uniform_buffer(256);
        assert_eq!(desc2.size(), 256);

        let desc3 = PyCodeSnippets::create_uniform_buffer(257);
        assert_eq!(desc3.size(), 512);
    }

    #[test]
    fn test_code_snippets_index_buffer() {
        let desc16 = PyCodeSnippets::create_index_buffer(1000, false);
        assert_eq!(desc16.size(), 2000); // 16-bit indices

        let desc32 = PyCodeSnippets::create_index_buffer(1000, true);
        assert_eq!(desc32.size(), 4000); // 32-bit indices
    }

    #[test]
    fn test_code_snippets_render_target() {
        let desc = PyCodeSnippets::create_render_target(1920, 1080);
        assert_eq!(desc.width(), 1920);
        assert_eq!(desc.height(), 1080);
        assert!(desc.usage().contains(&PyTextureUsage::render_attachment()));
    }

    #[test]
    fn test_code_snippets_depth_buffer() {
        let desc = PyCodeSnippets::create_depth_buffer(1920, 1080);
        assert!(desc.format().is_depth());
    }

    #[test]
    fn test_code_snippets_samplers() {
        let linear = PyCodeSnippets::create_sampler_linear();
        assert_eq!(linear.mag_filter(), PyFilterMode::Linear);

        let nearest = PyCodeSnippets::create_sampler_nearest();
        assert_eq!(nearest.mag_filter(), PyFilterMode::Nearest);

        let repeat = PyCodeSnippets::create_sampler_repeat();
        assert_eq!(repeat.address_mode_u(), PyAddressMode::Repeat);

        let aniso = PyCodeSnippets::create_sampler_anisotropic(8);
        assert_eq!(aniso.anisotropy_clamp(), 8);
    }

    // -- Validation Helper Tests --

    #[test]
    fn test_validation_buffer_alignment() {
        // Valid uniform buffer
        let valid = PyCodeSnippets::create_uniform_buffer(256);
        let report = PyValidationHelper::check_buffer_alignment(&valid);
        assert!(report.is_valid());

        // Invalid uniform buffer (not aligned)
        let invalid = PyBufferDescriptor::uniform(100);
        let report = PyValidationHelper::check_buffer_alignment(&invalid);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validation_buffer_zero_size() {
        let zero = PyBufferDescriptor::new(0);
        let report = PyValidationHelper::check_buffer_alignment(&zero);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validation_texture_format() {
        // Valid texture
        let valid = PyTextureDescriptor::new(256, 256, PyTextureFormat::Rgba8UnormSrgb);
        let report = PyValidationHelper::check_texture_format(&valid);
        assert!(report.is_valid());

        // Zero dimensions
        let invalid = PyTextureDescriptor::new(0, 0, PyTextureFormat::Rgba8Unorm);
        let report = PyValidationHelper::check_texture_format(&invalid);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validation_texture_mip_levels() {
        // Excessive mip levels
        let desc = PyTextureDescriptor::new(256, 256, PyTextureFormat::Rgba8Unorm)
            .with_mip_levels(20);
        let report = PyValidationHelper::check_texture_format(&desc);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validation_sampler() {
        // Valid sampler
        let valid = PySamplerDescriptor::linear();
        let report = PyValidationHelper::check_sampler(&valid);
        assert!(report.is_valid());

        // Invalid anisotropy
        let invalid = PySamplerDescriptor {
            label: None,
            address_mode_u: PyAddressMode::ClampToEdge,
            address_mode_v: PyAddressMode::ClampToEdge,
            address_mode_w: PyAddressMode::ClampToEdge,
            mag_filter: PyFilterMode::Linear,
            min_filter: PyFilterMode::Linear,
            mipmap_filter: PyFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            anisotropy_clamp: 32, // Invalid
        };
        let report = PyValidationHelper::check_sampler(&invalid);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validation_dispatch() {
        // Valid dispatch
        let valid = PyDispatchDescriptor::direct(64, 64, 1);
        let report = PyValidationHelper::check_dispatch(&valid);
        assert!(report.is_valid());

        // Zero dimension
        let invalid = PyDispatchDescriptor::direct(0, 64, 1);
        let report = PyValidationHelper::check_dispatch(&invalid);
        assert!(report.has_errors());
    }

    // -- Render Example Tests --

    #[test]
    fn test_render_example_lifecycle() {
        let mut example = PyRenderExample::new("test", 800, 600);
        assert!(!example.is_initialized());
        assert_eq!(example.command_count(), 0);

        example.initialize().unwrap();
        assert!(example.is_initialized());

        // Double init should fail
        assert!(example.initialize().is_err());

        example.draw(36, 1);
        assert_eq!(example.command_count(), 1);

        let result = example.run().unwrap();
        assert!(result.success());
        assert_eq!(result.commands_executed(), 1);

        example.cleanup();
        assert!(!example.is_initialized());
        assert_eq!(example.command_count(), 0);
    }

    #[test]
    fn test_render_example_resources() {
        let mut example = PyRenderExample::new("test", 800, 600);
        example.initialize().unwrap();

        let buffer_desc = PyBufferDescriptor::vertex(1024);
        let buffer = example.create_buffer(&buffer_desc).unwrap();
        assert_eq!(example.resource_count(), 1);

        let texture_desc = PyTextureDescriptor::new(256, 256, PyTextureFormat::Rgba8Unorm);
        let _texture = example.create_texture(&texture_desc).unwrap();
        assert_eq!(example.resource_count(), 2);

        example.cleanup();
        assert_eq!(example.resource_count(), 0);
    }

    // -- Compute Example Tests --

    #[test]
    fn test_compute_example_lifecycle() {
        let mut example = PyComputeExample::new("test");
        assert!(!example.is_initialized());

        example.initialize().unwrap();
        assert!(example.is_initialized());

        example.dispatch(64, 64, 1);
        assert_eq!(example.dispatch_count(), 1);

        let result = example.run().unwrap();
        assert!(result.success());

        example.cleanup();
        assert_eq!(example.dispatch_count(), 0);
    }

    // -- Renderer Example Tests --

    #[test]
    fn test_renderer_example_comprehensive() {
        let mut example = PyRendererExample::new("comprehensive", 1920, 1080);

        example.initialize().unwrap();

        // Create resources
        let vertex_buf = PyCodeSnippets::create_vertex_buffer(vec![0.0; 100]);
        let _vb = example.create_buffer(vertex_buf).unwrap();
        assert_eq!(example.buffer_count(), 1);

        let render_target = PyCodeSnippets::create_render_target(1920, 1080);
        let _rt = example.create_texture(render_target).unwrap();
        assert_eq!(example.texture_count(), 1);

        let sampler = PyCodeSnippets::create_sampler_linear();
        let _s = example.create_sampler(sampler).unwrap();
        assert_eq!(example.sampler_count(), 1);

        // Record commands
        example.draw(36, 1);
        example.draw_indexed(24, 100);
        example.dispatch(64, 64, 1);
        assert_eq!(example.command_count(), 3);

        // Validate
        let report = example.validate();
        assert!(report.is_valid());

        // Submit
        let result = example.submit().unwrap();
        assert!(result.success());
        assert_eq!(result.commands_executed(), 3);
        assert_eq!(result.resources_used(), 3);

        // Check memory estimate
        assert!(example.estimated_memory_bytes() > 0);

        example.cleanup();
        assert_eq!(example.buffer_count(), 0);
        assert_eq!(example.texture_count(), 0);
        assert_eq!(example.command_count(), 0);
    }

    // -- Validation Report Tests --

    #[test]
    fn test_validation_report_operations() {
        let mut report = PyExampleValidationReport::new();
        assert!(report.is_valid());

        report.add_warning("Test warning");
        assert!(report.is_valid());
        assert!(report.has_warnings());

        report.add_error("Test error");
        assert!(!report.is_valid());
        assert!(report.has_errors());

        assert_eq!(report.issue_count(), 2);

        let formatted = report.format_issues();
        assert!(formatted.contains("Test error"));
        assert!(formatted.contains("Test warning"));
    }

    #[test]
    fn test_validation_report_merge() {
        let mut report1 = PyExampleValidationReport::new();
        report1.add_error("Error 1");

        let mut report2 = PyExampleValidationReport::new();
        report2.add_warning("Warning 1");
        report2.add_error("Error 2");

        report1.merge(&report2);
        assert_eq!(report1.errors().len(), 2);
        assert_eq!(report1.warnings().len(), 1);
    }

    // -- Example Workflow Tests --

    #[test]
    fn test_full_render_workflow() {
        // Create a hello triangle example
        let mut example = PyQuickStart::hello_triangle();
        example.initialize().unwrap();

        // Create vertex buffer with triangle data
        let vertex_data = vec![
            0.0f32, 0.5,   // top
            -0.5, -0.5,    // bottom left
            0.5, -0.5,     // bottom right
        ];
        let vertex_desc = PyCodeSnippets::create_vertex_buffer(vertex_data);
        let _vertex_buffer = example.create_buffer(&vertex_desc).unwrap();

        // Create render target
        let target_desc = PyCodeSnippets::create_render_target(800, 600);
        let _render_target = example.create_texture(&target_desc).unwrap();

        // Build render pass
        let color_view = PyTextureView::new(0, None);
        let _pass = example.build_render_pass(&color_view, None, Some([0.1, 0.2, 0.3, 1.0]));

        // Draw triangle
        example.draw(3, 1);

        // Run and verify
        let result = example.run().unwrap();
        assert!(result.success());
        assert_eq!(result.commands_executed(), 1);

        example.cleanup();
    }

    #[test]
    fn test_full_compute_workflow() {
        let mut example = PyQuickStart::compute_shader();
        example.initialize().unwrap();

        // Create storage buffers
        let _input = example.create_storage_buffer(1024 * 1024).unwrap();
        let _output = example.create_storage_buffer(1024 * 1024).unwrap();

        // Build compute pass
        let _pass = example.build_compute_pass();

        // Dispatch compute work
        example.dispatch(256, 256, 1);

        // Run
        let result = example.run().unwrap();
        assert!(result.success());

        example.cleanup();
    }
}
