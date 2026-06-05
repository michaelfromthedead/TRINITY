//! Python bindings for compute pass construction (T-WGPU-P7.6.6).
//!
//! This module provides Python-accessible types for creating and configuring
//! compute pass descriptors with a fluent builder pattern, dispatch descriptors,
//! and compute pipeline descriptors.
//!
//! # Types
//!
//! - [`PyTimestampWrites`] - Timestamp write configuration for GPU profiling
//! - [`PyComputePassDescriptor`] - Compute pass configuration
//! - [`PyComputePassBuilder`] - Fluent builder for compute pass descriptors
//! - [`PyDispatchDescriptor`] - Compute dispatch parameters (direct or indirect)
//! - [`PyComputePipelineDescriptor`] - Compute pipeline configuration
//! - [`PyPushConstantRange`] - Push constant range specification
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     ComputePassBuilder, DispatchDescriptor, ComputePipelineDescriptor
//! )
//!
//! # Build a compute pass descriptor with timestamps
//! pass_desc = (ComputePassBuilder()
//!     .label("particle_simulation")
//!     .timestamp_begin(0)
//!     .timestamp_end(1)
//!     .build())
//!
//! # Direct dispatch
//! dispatch = DispatchDescriptor.direct(64, 64, 1)
//!
//! # Indirect dispatch from buffer
//! indirect = DispatchDescriptor.indirect(buffer_handle, 0)
//!
//! # Configure compute pipeline
//! pipeline = (ComputePipelineDescriptor()
//!     .with_label("physics_compute")
//!     .with_entry_point("main")
//!     .with_push_constant(0, 64, ShaderStage.Compute))
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

// Import PyResourceHandle for indirect buffer references
use super::py_resource::PyResourceHandle;

// ============================================================================
// PyShaderStages
// ============================================================================

/// Shader stage visibility flags for push constants.
///
/// Determines which shader stages can access a push constant range.
#[pyclass(name = "ShaderStages")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyShaderStages {
    bits: u32,
}

/// wgpu::ShaderStages bit values
mod stage_bits {
    pub const NONE: u32 = 0;
    pub const VERTEX: u32 = 1 << 0;
    pub const FRAGMENT: u32 = 1 << 1;
    pub const COMPUTE: u32 = 1 << 2;
}

#[pymethods]
impl PyShaderStages {
    /// Create shader stages from raw bits.
    #[new]
    pub fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Returns the raw bits value.
    #[getter]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// No shader stages.
    #[staticmethod]
    pub fn none() -> Self {
        Self { bits: stage_bits::NONE }
    }

    /// Vertex shader stage.
    #[staticmethod]
    pub fn vertex() -> Self {
        Self { bits: stage_bits::VERTEX }
    }

    /// Fragment shader stage.
    #[staticmethod]
    pub fn fragment() -> Self {
        Self { bits: stage_bits::FRAGMENT }
    }

    /// Compute shader stage.
    #[staticmethod]
    pub fn compute() -> Self {
        Self { bits: stage_bits::COMPUTE }
    }

    /// All graphics stages (vertex + fragment).
    #[staticmethod]
    pub fn vertex_fragment() -> Self {
        Self {
            bits: stage_bits::VERTEX | stage_bits::FRAGMENT,
        }
    }

    /// All stages (vertex + fragment + compute).
    #[staticmethod]
    pub fn all() -> Self {
        Self {
            bits: stage_bits::VERTEX | stage_bits::FRAGMENT | stage_bits::COMPUTE,
        }
    }

    /// Returns true if no stages are set.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Returns true if all bits in `other` are contained in `self`.
    pub fn contains(&self, other: &Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Combine stages with `|` operator.
    pub fn __or__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Intersect stages with `&` operator.
    pub fn __and__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    pub fn __repr__(&self) -> String {
        if self.bits == 0 {
            return "ShaderStages()".to_string();
        }

        let mut flags = Vec::new();
        if self.bits & stage_bits::VERTEX != 0 {
            flags.push("VERTEX");
        }
        if self.bits & stage_bits::FRAGMENT != 0 {
            flags.push("FRAGMENT");
        }
        if self.bits & stage_bits::COMPUTE != 0 {
            flags.push("COMPUTE");
        }

        format!("ShaderStages({})", flags.join(" | "))
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __hash__(&self) -> u64 {
        self.bits as u64
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.bits == other.bits
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        self.bits != other.bits
    }
}

impl Default for PyShaderStages {
    fn default() -> Self {
        Self { bits: 0 }
    }
}

// ============================================================================
// PyTimestampWrites
// ============================================================================

/// Timestamp write configuration for compute pass profiling.
///
/// Specifies query indices for writing GPU timestamps at the beginning
/// and/or end of a compute pass.
///
/// # Example
///
/// ```python
/// ts = TimestampWrites(query_set_id=1).both(0, 1)
/// ```
#[pyclass(name = "TimestampWrites")]
#[derive(Clone, Debug, Default)]
pub struct PyTimestampWrites {
    query_set_id: u64,
    beginning_index: Option<u32>,
    end_index: Option<u32>,
}

#[pymethods]
impl PyTimestampWrites {
    /// Create timestamp writes for a query set.
    ///
    /// # Arguments
    /// * `query_set_id` - ID of the query set to write timestamps to
    #[new]
    #[pyo3(signature = (query_set_id=0))]
    pub fn new(query_set_id: u64) -> Self {
        Self {
            query_set_id,
            beginning_index: None,
            end_index: None,
        }
    }

    /// Query set ID for timestamp writes.
    #[getter]
    pub fn query_set_id(&self) -> u64 {
        self.query_set_id
    }

    /// Beginning timestamp query index (if any).
    #[getter]
    pub fn beginning_index(&self) -> Option<u32> {
        self.beginning_index
    }

    /// End timestamp query index (if any).
    #[getter]
    pub fn end_index(&self) -> Option<u32> {
        self.end_index
    }

    /// Set both beginning and end timestamp indices.
    pub fn both(&self, beginning: u32, end: u32) -> Self {
        Self {
            query_set_id: self.query_set_id,
            beginning_index: Some(beginning),
            end_index: Some(end),
        }
    }

    /// Set only the beginning timestamp index.
    pub fn beginning_only(&self, index: u32) -> Self {
        Self {
            query_set_id: self.query_set_id,
            beginning_index: Some(index),
            end_index: None,
        }
    }

    /// Set only the end timestamp index.
    pub fn end_only(&self, index: u32) -> Self {
        Self {
            query_set_id: self.query_set_id,
            beginning_index: None,
            end_index: Some(index),
        }
    }

    /// Set the beginning timestamp index.
    pub fn with_beginning(&self, index: u32) -> Self {
        Self {
            query_set_id: self.query_set_id,
            beginning_index: Some(index),
            end_index: self.end_index,
        }
    }

    /// Set the end timestamp index.
    pub fn with_end(&self, index: u32) -> Self {
        Self {
            query_set_id: self.query_set_id,
            beginning_index: self.beginning_index,
            end_index: Some(index),
        }
    }

    /// Returns true if any timestamps are configured.
    pub fn is_enabled(&self) -> bool {
        self.beginning_index.is_some() || self.end_index.is_some()
    }

    /// Returns true if beginning timestamp is configured.
    pub fn has_beginning(&self) -> bool {
        self.beginning_index.is_some()
    }

    /// Returns true if end timestamp is configured.
    pub fn has_end(&self) -> bool {
        self.end_index.is_some()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TimestampWrites(query_set={}, begin={:?}, end={:?})",
            self.query_set_id, self.beginning_index, self.end_index
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.query_set_id == other.query_set_id
            && self.beginning_index == other.beginning_index
            && self.end_index == other.end_index
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }
}

// ============================================================================
// PyComputePassDescriptor
// ============================================================================

/// Descriptor for creating a compute pass.
///
/// Contains the configuration needed to begin a compute pass on a command encoder.
///
/// # Example
///
/// ```python
/// desc = ComputePassDescriptor()
///     .with_label("culling_pass")
///     .with_timestamp_writes(TimestampWrites(1).both(0, 1))
/// ```
#[pyclass(name = "ComputePassDescriptor")]
#[derive(Clone, Debug, Default)]
pub struct PyComputePassDescriptor {
    label: Option<String>,
    timestamp_writes: Option<PyTimestampWrites>,
}

#[pymethods]
impl PyComputePassDescriptor {
    /// Create a new compute pass descriptor.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Optional debug label for the pass.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Optional timestamp writes configuration.
    #[getter]
    pub fn timestamp_writes(&self) -> Option<PyTimestampWrites> {
        self.timestamp_writes.clone()
    }

    /// Set the debug label.
    pub fn with_label(&self, label: &str) -> Self {
        Self {
            label: Some(label.to_string()),
            timestamp_writes: self.timestamp_writes.clone(),
        }
    }

    /// Set timestamp writes configuration.
    pub fn with_timestamp_writes(&self, writes: PyTimestampWrites) -> Self {
        Self {
            label: self.label.clone(),
            timestamp_writes: Some(writes),
        }
    }

    /// Clear the timestamp writes configuration.
    pub fn without_timestamp_writes(&self) -> Self {
        Self {
            label: self.label.clone(),
            timestamp_writes: None,
        }
    }

    /// Returns true if timestamp writes are configured.
    pub fn has_timestamp_writes(&self) -> bool {
        self.timestamp_writes.is_some()
    }

    /// Returns true if the descriptor has a label.
    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }

    pub fn __repr__(&self) -> String {
        let label_str = self
            .label
            .as_ref()
            .map(|l| format!("label='{}'", l))
            .unwrap_or_else(|| "label=None".to_string());
        let ts_str = if self.timestamp_writes.is_some() {
            "timestamps=enabled"
        } else {
            "timestamps=disabled"
        };
        format!("ComputePassDescriptor({}, {})", label_str, ts_str)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyComputePassBuilder
// ============================================================================

/// Fluent builder for compute pass descriptors.
///
/// Provides a chainable API for constructing `ComputePassDescriptor` instances.
///
/// # Example
///
/// ```python
/// desc = (ComputePassBuilder()
///     .label("physics_pass")
///     .timestamp_begin(0)
///     .timestamp_end(1)
///     .with_push_constants(push_data)
///     .build())
/// ```
#[pyclass(name = "ComputePassBuilder")]
#[derive(Clone, Debug, Default)]
pub struct PyComputePassBuilder {
    label: Option<String>,
    query_set_id: u64,
    timestamp_begin: Option<u32>,
    timestamp_end: Option<u32>,
    push_constants: Option<Vec<u8>>,
}

#[pymethods]
impl PyComputePassBuilder {
    /// Create a new compute pass builder.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debug label for the pass.
    pub fn label(&mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self.clone()
    }

    /// Set the query set ID for timestamp writes.
    pub fn query_set(&mut self, id: u64) -> Self {
        self.query_set_id = id;
        self.clone()
    }

    /// Set the beginning timestamp index.
    pub fn timestamp_begin(&mut self, index: u32) -> Self {
        self.timestamp_begin = Some(index);
        self.clone()
    }

    /// Set the end timestamp index.
    pub fn timestamp_end(&mut self, index: u32) -> Self {
        self.timestamp_end = Some(index);
        self.clone()
    }

    /// Set both timestamp indices at once.
    pub fn timestamps(&mut self, begin: u32, end: u32) -> Self {
        self.timestamp_begin = Some(begin);
        self.timestamp_end = Some(end);
        self.clone()
    }

    /// Set push constants data for inline configuration.
    ///
    /// The data must be a multiple of 4 bytes.
    pub fn with_push_constants(&mut self, data: Vec<u8>) -> PyResult<Self> {
        if data.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                "Push constant data must be a multiple of 4 bytes",
            ));
        }
        self.push_constants = Some(data);
        Ok(self.clone())
    }

    /// Clear any push constants data.
    pub fn without_push_constants(&mut self) -> Self {
        self.push_constants = None;
        self.clone()
    }

    /// Get the currently configured push constants (if any).
    pub fn get_push_constants(&self) -> Option<Vec<u8>> {
        self.push_constants.clone()
    }

    /// Build the compute pass descriptor.
    pub fn build(&self) -> PyComputePassDescriptor {
        let timestamp_writes = if self.timestamp_begin.is_some() || self.timestamp_end.is_some() {
            Some(PyTimestampWrites {
                query_set_id: self.query_set_id,
                beginning_index: self.timestamp_begin,
                end_index: self.timestamp_end,
            })
        } else {
            None
        };

        PyComputePassDescriptor {
            label: self.label.clone(),
            timestamp_writes,
        }
    }

    /// Check if the builder has any timestamp configuration.
    pub fn has_timestamps(&self) -> bool {
        self.timestamp_begin.is_some() || self.timestamp_end.is_some()
    }

    /// Check if the builder has a label.
    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }

    /// Check if the builder has push constants.
    pub fn has_push_constants(&self) -> bool {
        self.push_constants.is_some()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ComputePassBuilder(label={:?}, timestamps=({:?}, {:?}), push_constants={})",
            self.label,
            self.timestamp_begin,
            self.timestamp_end,
            self.push_constants.as_ref().map(|p| p.len()).unwrap_or(0)
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyDispatchDescriptor
// ============================================================================

/// Descriptor for compute shader dispatch operations.
///
/// Specifies either direct workgroup counts or indirect dispatch from a buffer.
///
/// # Direct Dispatch
///
/// ```python
/// # Dispatch 64x64x1 workgroups
/// dispatch = DispatchDescriptor.direct(64, 64, 1)
/// ```
///
/// # Indirect Dispatch
///
/// ```python
/// # Dispatch using workgroup counts from a buffer
/// dispatch = DispatchDescriptor.indirect(buffer_handle, offset=0)
/// ```
#[pyclass(name = "DispatchDescriptor")]
#[derive(Clone, Debug)]
pub struct PyDispatchDescriptor {
    workgroups: [u32; 3],
    indirect: bool,
    indirect_buffer: Option<PyResourceHandle>,
    indirect_offset: u64,
}

#[pymethods]
impl PyDispatchDescriptor {
    /// Create a direct dispatch descriptor with explicit workgroup counts.
    ///
    /// # Arguments
    /// * `x` - Workgroups in X dimension (must be > 0)
    /// * `y` - Workgroups in Y dimension (must be > 0)
    /// * `z` - Workgroups in Z dimension (must be > 0)
    #[staticmethod]
    pub fn direct(x: u32, y: u32, z: u32) -> PyResult<Self> {
        if x == 0 {
            return Err(PyValueError::new_err("X workgroup count must be > 0"));
        }
        if y == 0 {
            return Err(PyValueError::new_err("Y workgroup count must be > 0"));
        }
        if z == 0 {
            return Err(PyValueError::new_err("Z workgroup count must be > 0"));
        }

        Ok(Self {
            workgroups: [x, y, z],
            indirect: false,
            indirect_buffer: None,
            indirect_offset: 0,
        })
    }

    /// Create an indirect dispatch descriptor using workgroup counts from a buffer.
    ///
    /// # Arguments
    /// * `buffer` - Buffer containing DispatchIndirectArgs (3x u32)
    /// * `offset` - Byte offset into the buffer
    #[staticmethod]
    pub fn indirect(buffer: PyResourceHandle, offset: u64) -> Self {
        Self {
            workgroups: [0, 0, 0],
            indirect: true,
            indirect_buffer: Some(buffer),
            indirect_offset: offset,
        }
    }

    /// Create a 1D direct dispatch.
    #[staticmethod]
    pub fn direct_1d(x: u32) -> PyResult<Self> {
        Self::direct(x, 1, 1)
    }

    /// Create a 2D direct dispatch.
    #[staticmethod]
    pub fn direct_2d(x: u32, y: u32) -> PyResult<Self> {
        Self::direct(x, y, 1)
    }

    /// Workgroup counts [x, y, z].
    ///
    /// For indirect dispatch, these are placeholder values (0, 0, 0).
    #[getter]
    pub fn workgroups(&self) -> [u32; 3] {
        self.workgroups
    }

    /// X workgroup count.
    #[getter]
    pub fn x(&self) -> u32 {
        self.workgroups[0]
    }

    /// Y workgroup count.
    #[getter]
    pub fn y(&self) -> u32 {
        self.workgroups[1]
    }

    /// Z workgroup count.
    #[getter]
    pub fn z(&self) -> u32 {
        self.workgroups[2]
    }

    /// Returns true if this is an indirect dispatch.
    #[getter]
    pub fn is_indirect(&self) -> bool {
        self.indirect
    }

    /// Returns true if this is a direct dispatch.
    #[getter]
    pub fn is_direct(&self) -> bool {
        !self.indirect
    }

    /// Buffer handle for indirect dispatch (None for direct).
    #[getter]
    pub fn indirect_buffer(&self) -> Option<PyResourceHandle> {
        self.indirect_buffer.clone()
    }

    /// Byte offset into the indirect buffer.
    #[getter]
    pub fn indirect_offset(&self) -> u64 {
        self.indirect_offset
    }

    /// Total number of workgroups (x * y * z).
    ///
    /// Returns 0 for indirect dispatch since counts are not known.
    pub fn total_workgroups(&self) -> u64 {
        if self.indirect {
            0
        } else {
            self.workgroups[0] as u64
                * self.workgroups[1] as u64
                * self.workgroups[2] as u64
        }
    }

    /// Validate workgroup counts against device limits.
    ///
    /// # Arguments
    /// * `max_x` - Maximum workgroups in X dimension
    /// * `max_y` - Maximum workgroups in Y dimension
    /// * `max_z` - Maximum workgroups in Z dimension
    ///
    /// Returns None if valid, or an error message if invalid.
    pub fn validate(&self, max_x: u32, max_y: u32, max_z: u32) -> Option<String> {
        if self.indirect {
            return None; // Can't validate indirect dispatch
        }

        if self.workgroups[0] > max_x {
            return Some(format!(
                "X workgroups {} exceeds limit {}",
                self.workgroups[0], max_x
            ));
        }
        if self.workgroups[1] > max_y {
            return Some(format!(
                "Y workgroups {} exceeds limit {}",
                self.workgroups[1], max_y
            ));
        }
        if self.workgroups[2] > max_z {
            return Some(format!(
                "Z workgroups {} exceeds limit {}",
                self.workgroups[2], max_z
            ));
        }

        None
    }

    /// Check if workgroup counts are within default wgpu limits (65535).
    pub fn is_within_default_limits(&self) -> bool {
        self.validate(65535, 65535, 65535).is_none()
    }

    pub fn __repr__(&self) -> String {
        if self.indirect {
            format!(
                "DispatchDescriptor.indirect(buffer={:?}, offset={})",
                self.indirect_buffer.as_ref().map(|b| b.id()),
                self.indirect_offset
            )
        } else {
            format!(
                "DispatchDescriptor.direct({}, {}, {})",
                self.workgroups[0], self.workgroups[1], self.workgroups[2]
            )
        }
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.workgroups == other.workgroups
            && self.indirect == other.indirect
            && self.indirect_offset == other.indirect_offset
            && self.indirect_buffer.as_ref().map(|b| b.id())
                == other.indirect_buffer.as_ref().map(|b| b.id())
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }
}

impl Default for PyDispatchDescriptor {
    fn default() -> Self {
        Self {
            workgroups: [1, 1, 1],
            indirect: false,
            indirect_buffer: None,
            indirect_offset: 0,
        }
    }
}

// ============================================================================
// PyPushConstantRange
// ============================================================================

/// A range of push constants accessible by a shader stage.
///
/// Push constants are small pieces of data that can be updated without
/// creating new bind groups, useful for frequently changing parameters.
///
/// # Example
///
/// ```python
/// # 64 bytes of push constants visible to compute shader at offset 0
/// range = PushConstantRange(0, 64, ShaderStages.compute())
/// ```
#[pyclass(name = "PushConstantRange")]
#[derive(Clone, Debug)]
pub struct PyPushConstantRange {
    offset: u32,
    size: u32,
    stages: PyShaderStages,
}

#[pymethods]
impl PyPushConstantRange {
    /// Create a push constant range.
    ///
    /// # Arguments
    /// * `offset` - Byte offset (must be multiple of 4)
    /// * `size` - Size in bytes (must be multiple of 4)
    /// * `stages` - Shader stages that can access this range
    #[new]
    pub fn new(offset: u32, size: u32, stages: PyShaderStages) -> PyResult<Self> {
        if offset % 4 != 0 {
            return Err(PyValueError::new_err("Offset must be a multiple of 4"));
        }
        if size % 4 != 0 {
            return Err(PyValueError::new_err("Size must be a multiple of 4"));
        }
        if size == 0 {
            return Err(PyValueError::new_err("Size must be > 0"));
        }

        Ok(Self {
            offset,
            size,
            stages,
        })
    }

    /// Create a compute-only push constant range starting at offset 0.
    #[staticmethod]
    pub fn compute(size: u32) -> PyResult<Self> {
        Self::new(0, size, PyShaderStages::compute())
    }

    /// Create a compute push constant range with custom offset.
    #[staticmethod]
    pub fn compute_at(offset: u32, size: u32) -> PyResult<Self> {
        Self::new(offset, size, PyShaderStages::compute())
    }

    /// Byte offset into the push constant data.
    #[getter]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Size of the range in bytes.
    #[getter]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Shader stages that can access this range.
    #[getter]
    pub fn stages(&self) -> PyShaderStages {
        self.stages
    }

    /// End offset (offset + size).
    pub fn end_offset(&self) -> u32 {
        self.offset + self.size
    }

    /// Check if this range overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.offset < other.end_offset() && other.offset < self.end_offset()
    }

    /// Check if this range is valid for the given total size limit.
    pub fn is_valid_for_limit(&self, max_size: u32) -> bool {
        self.end_offset() <= max_size
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PushConstantRange(offset={}, size={}, stages={})",
            self.offset,
            self.size,
            self.stages.__repr__()
        )
    }

    pub fn __str__(&self) -> String {
        format!("{}..{} bytes ({} stages)", self.offset, self.end_offset(), self.stages.__str__())
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.offset == other.offset && self.size == other.size && self.stages == other.stages
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }

    pub fn __hash__(&self) -> u64 {
        let mut hash = self.offset as u64;
        hash = hash.wrapping_mul(31).wrapping_add(self.size as u64);
        hash = hash.wrapping_mul(31).wrapping_add(self.stages.bits as u64);
        hash
    }
}

// ============================================================================
// PyComputePipelineDescriptor
// ============================================================================

/// Descriptor for creating a compute pipeline.
///
/// Specifies the shader module, entry point, and push constant configuration.
///
/// # Example
///
/// ```python
/// pipeline = (ComputePipelineDescriptor()
///     .with_label("particle_update")
///     .with_shader("particle.wgsl")
///     .with_entry_point("update_positions")
///     .with_push_constant(0, 64, ShaderStages.compute()))
/// ```
#[pyclass(name = "ComputePipelineDescriptor")]
#[derive(Clone, Debug, Default)]
pub struct PyComputePipelineDescriptor {
    label: Option<String>,
    shader: String,
    entry_point: String,
    push_constant_ranges: Vec<PyPushConstantRange>,
}

#[pymethods]
impl PyComputePipelineDescriptor {
    /// Create a new compute pipeline descriptor.
    #[new]
    pub fn new() -> Self {
        Self {
            label: None,
            shader: String::new(),
            entry_point: "main".to_string(),
            push_constant_ranges: Vec::new(),
        }
    }

    /// Create a descriptor with shader and entry point.
    #[staticmethod]
    pub fn with_shader_and_entry(shader: &str, entry_point: &str) -> Self {
        Self {
            label: None,
            shader: shader.to_string(),
            entry_point: entry_point.to_string(),
            push_constant_ranges: Vec::new(),
        }
    }

    /// Optional debug label.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Shader module name or source.
    #[getter]
    pub fn shader(&self) -> String {
        self.shader.clone()
    }

    /// Entry point function name.
    #[getter]
    pub fn entry_point(&self) -> String {
        self.entry_point.clone()
    }

    /// Push constant ranges.
    #[getter]
    pub fn push_constant_ranges(&self) -> Vec<PyPushConstantRange> {
        self.push_constant_ranges.clone()
    }

    /// Set the debug label.
    pub fn with_label(&self, label: &str) -> Self {
        Self {
            label: Some(label.to_string()),
            shader: self.shader.clone(),
            entry_point: self.entry_point.clone(),
            push_constant_ranges: self.push_constant_ranges.clone(),
        }
    }

    /// Set the shader module name or source.
    pub fn with_shader(&self, shader: &str) -> Self {
        Self {
            label: self.label.clone(),
            shader: shader.to_string(),
            entry_point: self.entry_point.clone(),
            push_constant_ranges: self.push_constant_ranges.clone(),
        }
    }

    /// Set the entry point function name.
    pub fn with_entry_point(&self, entry_point: &str) -> Self {
        Self {
            label: self.label.clone(),
            shader: self.shader.clone(),
            entry_point: entry_point.to_string(),
            push_constant_ranges: self.push_constant_ranges.clone(),
        }
    }

    /// Add a push constant range.
    pub fn with_push_constant(&self, offset: u32, size: u32, stages: PyShaderStages) -> PyResult<Self> {
        let range = PyPushConstantRange::new(offset, size, stages)?;

        // Check for overlaps with existing ranges
        for existing in &self.push_constant_ranges {
            if existing.overlaps(&range) {
                return Err(PyValueError::new_err(format!(
                    "Push constant range {}..{} overlaps with existing range {}..{}",
                    range.offset, range.end_offset(),
                    existing.offset, existing.end_offset()
                )));
            }
        }

        let mut ranges = self.push_constant_ranges.clone();
        ranges.push(range);

        Ok(Self {
            label: self.label.clone(),
            shader: self.shader.clone(),
            entry_point: self.entry_point.clone(),
            push_constant_ranges: ranges,
        })
    }

    /// Add a pre-built push constant range.
    pub fn with_push_constant_range(&self, range: PyPushConstantRange) -> PyResult<Self> {
        // Check for overlaps
        for existing in &self.push_constant_ranges {
            if existing.overlaps(&range) {
                return Err(PyValueError::new_err(format!(
                    "Push constant range {}..{} overlaps with existing range {}..{}",
                    range.offset, range.end_offset(),
                    existing.offset, existing.end_offset()
                )));
            }
        }

        let mut ranges = self.push_constant_ranges.clone();
        ranges.push(range);

        Ok(Self {
            label: self.label.clone(),
            shader: self.shader.clone(),
            entry_point: self.entry_point.clone(),
            push_constant_ranges: ranges,
        })
    }

    /// Clear all push constant ranges.
    pub fn without_push_constants(&self) -> Self {
        Self {
            label: self.label.clone(),
            shader: self.shader.clone(),
            entry_point: self.entry_point.clone(),
            push_constant_ranges: Vec::new(),
        }
    }

    /// Returns true if any push constant ranges are configured.
    pub fn has_push_constants(&self) -> bool {
        !self.push_constant_ranges.is_empty()
    }

    /// Returns the total size of all push constant ranges.
    pub fn total_push_constant_size(&self) -> u32 {
        self.push_constant_ranges
            .iter()
            .map(|r| r.end_offset())
            .max()
            .unwrap_or(0)
    }

    /// Validate the descriptor.
    ///
    /// Checks:
    /// - Shader is not empty
    /// - Entry point is not empty
    /// - Push constant ranges don't exceed max_push_constant_size
    pub fn validate(&self, max_push_constant_size: u32) -> Option<String> {
        if self.shader.is_empty() {
            return Some("Shader must not be empty".to_string());
        }
        if self.entry_point.is_empty() {
            return Some("Entry point must not be empty".to_string());
        }

        let total_size = self.total_push_constant_size();
        if total_size > max_push_constant_size {
            return Some(format!(
                "Total push constant size {} exceeds limit {}",
                total_size, max_push_constant_size
            ));
        }

        None
    }

    /// Returns true if the descriptor is valid with default limits.
    pub fn is_valid(&self) -> bool {
        // Default wgpu max push constant size is 128 bytes
        self.validate(128).is_none()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ComputePipelineDescriptor(label={:?}, shader='{}', entry='{}', push_constants={})",
            self.label,
            self.shader,
            self.entry_point,
            self.push_constant_ranges.len()
        )
    }

    pub fn __str__(&self) -> String {
        let label_str = self.label.as_deref().unwrap_or("unnamed");
        format!(
            "{}: {}::{} ({} push constant ranges)",
            label_str,
            self.shader,
            self.entry_point,
            self.push_constant_ranges.len()
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Registers the compute pass bindings in a PyO3 module.
pub fn register_module(_py: Python<'_>, parent: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    parent.add_class::<PyShaderStages>()?;
    parent.add_class::<PyTimestampWrites>()?;
    parent.add_class::<PyComputePassDescriptor>()?;
    parent.add_class::<PyComputePassBuilder>()?;
    parent.add_class::<PyDispatchDescriptor>()?;
    parent.add_class::<PyPushConstantRange>()?;
    parent.add_class::<PyComputePipelineDescriptor>()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PyShaderStages tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_stages_none() {
        let stages = PyShaderStages::none();
        assert!(stages.is_empty());
        assert_eq!(stages.bits(), 0);
    }

    #[test]
    fn test_shader_stages_compute() {
        let stages = PyShaderStages::compute();
        assert!(!stages.is_empty());
        assert!(stages.contains(&PyShaderStages::compute()));
        assert!(!stages.contains(&PyShaderStages::vertex()));
    }

    #[test]
    fn test_shader_stages_vertex() {
        let stages = PyShaderStages::vertex();
        assert!(stages.contains(&PyShaderStages::vertex()));
        assert!(!stages.contains(&PyShaderStages::fragment()));
    }

    #[test]
    fn test_shader_stages_fragment() {
        let stages = PyShaderStages::fragment();
        assert!(stages.contains(&PyShaderStages::fragment()));
    }

    #[test]
    fn test_shader_stages_vertex_fragment() {
        let stages = PyShaderStages::vertex_fragment();
        assert!(stages.contains(&PyShaderStages::vertex()));
        assert!(stages.contains(&PyShaderStages::fragment()));
        assert!(!stages.contains(&PyShaderStages::compute()));
    }

    #[test]
    fn test_shader_stages_all() {
        let all = PyShaderStages::all();
        assert!(all.contains(&PyShaderStages::vertex()));
        assert!(all.contains(&PyShaderStages::fragment()));
        assert!(all.contains(&PyShaderStages::compute()));
    }

    #[test]
    fn test_shader_stages_or() {
        let a = PyShaderStages::vertex();
        let b = PyShaderStages::compute();
        let combined = a.__or__(&b);
        assert!(combined.contains(&PyShaderStages::vertex()));
        assert!(combined.contains(&PyShaderStages::compute()));
    }

    #[test]
    fn test_shader_stages_and() {
        let a = PyShaderStages::vertex_fragment();
        let b = PyShaderStages::vertex();
        let result = a.__and__(&b);
        assert!(result.contains(&PyShaderStages::vertex()));
        assert!(!result.contains(&PyShaderStages::fragment()));
    }

    #[test]
    fn test_shader_stages_repr() {
        let empty = PyShaderStages::none();
        assert_eq!(empty.__repr__(), "ShaderStages()");

        let compute = PyShaderStages::compute();
        assert!(compute.__repr__().contains("COMPUTE"));
    }

    #[test]
    fn test_shader_stages_equality() {
        let a = PyShaderStages::compute();
        let b = PyShaderStages::compute();
        let c = PyShaderStages::vertex();
        assert!(a.__eq__(&b));
        assert!(a.__ne__(&c));
    }

    #[test]
    fn test_shader_stages_hash() {
        let a = PyShaderStages::compute();
        let b = PyShaderStages::compute();
        assert_eq!(a.__hash__(), b.__hash__());
    }

    // -------------------------------------------------------------------------
    // PyTimestampWrites tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timestamp_writes_new() {
        let ts = PyTimestampWrites::new(42);
        assert_eq!(ts.query_set_id(), 42);
        assert!(!ts.is_enabled());
        assert!(!ts.has_beginning());
        assert!(!ts.has_end());
    }

    #[test]
    fn test_timestamp_writes_both() {
        let ts = PyTimestampWrites::new(1).both(0, 1);
        assert!(ts.is_enabled());
        assert!(ts.has_beginning());
        assert!(ts.has_end());
        assert_eq!(ts.beginning_index(), Some(0));
        assert_eq!(ts.end_index(), Some(1));
    }

    #[test]
    fn test_timestamp_writes_beginning_only() {
        let ts = PyTimestampWrites::new(1).beginning_only(5);
        assert!(ts.is_enabled());
        assert!(ts.has_beginning());
        assert!(!ts.has_end());
        assert_eq!(ts.beginning_index(), Some(5));
    }

    #[test]
    fn test_timestamp_writes_end_only() {
        let ts = PyTimestampWrites::new(1).end_only(10);
        assert!(ts.is_enabled());
        assert!(!ts.has_beginning());
        assert!(ts.has_end());
        assert_eq!(ts.end_index(), Some(10));
    }

    #[test]
    fn test_timestamp_writes_with_methods() {
        let ts = PyTimestampWrites::new(1)
            .with_beginning(2)
            .with_end(3);
        assert_eq!(ts.beginning_index(), Some(2));
        assert_eq!(ts.end_index(), Some(3));
    }

    #[test]
    fn test_timestamp_writes_repr() {
        let ts = PyTimestampWrites::new(1).both(0, 1);
        let repr = ts.__repr__();
        assert!(repr.contains("TimestampWrites"));
        assert!(repr.contains("query_set=1"));
    }

    #[test]
    fn test_timestamp_writes_equality() {
        let a = PyTimestampWrites::new(1).both(0, 1);
        let b = PyTimestampWrites::new(1).both(0, 1);
        let c = PyTimestampWrites::new(2).both(0, 1);
        assert!(a.__eq__(&b));
        assert!(a.__ne__(&c));
    }

    // -------------------------------------------------------------------------
    // PyComputePassDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_pass_descriptor_new() {
        let desc = PyComputePassDescriptor::new();
        assert!(desc.label().is_none());
        assert!(!desc.has_timestamp_writes());
        assert!(!desc.has_label());
    }

    #[test]
    fn test_compute_pass_descriptor_with_label() {
        let desc = PyComputePassDescriptor::new().with_label("test_pass");
        assert_eq!(desc.label(), Some("test_pass".to_string()));
        assert!(desc.has_label());
    }

    #[test]
    fn test_compute_pass_descriptor_with_timestamp_writes() {
        let ts = PyTimestampWrites::new(1).both(0, 1);
        let desc = PyComputePassDescriptor::new().with_timestamp_writes(ts);
        assert!(desc.has_timestamp_writes());
        let ts = desc.timestamp_writes().unwrap();
        assert_eq!(ts.query_set_id(), 1);
    }

    #[test]
    fn test_compute_pass_descriptor_without_timestamp_writes() {
        let ts = PyTimestampWrites::new(1).both(0, 1);
        let desc = PyComputePassDescriptor::new()
            .with_timestamp_writes(ts)
            .without_timestamp_writes();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_descriptor_repr() {
        let desc = PyComputePassDescriptor::new().with_label("test");
        let repr = desc.__repr__();
        assert!(repr.contains("ComputePassDescriptor"));
        assert!(repr.contains("test"));
    }

    // -------------------------------------------------------------------------
    // PyComputePassBuilder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_pass_builder_new() {
        let builder = PyComputePassBuilder::new();
        assert!(!builder.has_label());
        assert!(!builder.has_timestamps());
        assert!(!builder.has_push_constants());
    }

    #[test]
    fn test_compute_pass_builder_label() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder.label("my_pass");
        assert!(builder.has_label());
        let desc = builder.build();
        assert_eq!(desc.label(), Some("my_pass".to_string()));
    }

    #[test]
    fn test_compute_pass_builder_timestamps() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder.query_set(1).timestamps(0, 1);
        assert!(builder.has_timestamps());
        let desc = builder.build();
        assert!(desc.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_builder_timestamp_begin() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder.timestamp_begin(5);
        let desc = builder.build();
        let ts = desc.timestamp_writes().unwrap();
        assert_eq!(ts.beginning_index(), Some(5));
        assert!(ts.end_index().is_none());
    }

    #[test]
    fn test_compute_pass_builder_timestamp_end() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder.timestamp_end(10);
        let desc = builder.build();
        let ts = desc.timestamp_writes().unwrap();
        assert!(ts.beginning_index().is_none());
        assert_eq!(ts.end_index(), Some(10));
    }

    #[test]
    fn test_compute_pass_builder_push_constants() {
        let mut builder = PyComputePassBuilder::new();
        let data = vec![0u8; 64];
        builder = builder.with_push_constants(data.clone()).unwrap();
        assert!(builder.has_push_constants());
        assert_eq!(builder.get_push_constants(), Some(data));
    }

    #[test]
    fn test_compute_pass_builder_push_constants_invalid_size() {
        let mut builder = PyComputePassBuilder::new();
        let data = vec![0u8; 63]; // Not multiple of 4
        let result = builder.with_push_constants(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_pass_builder_without_push_constants() {
        let mut builder = PyComputePassBuilder::new();
        let data = vec![0u8; 64];
        builder = builder.with_push_constants(data).unwrap();
        builder = builder.without_push_constants();
        assert!(!builder.has_push_constants());
    }

    #[test]
    fn test_compute_pass_builder_build() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder
            .label("built_pass")
            .query_set(2)
            .timestamps(0, 1);
        let desc = builder.build();
        assert_eq!(desc.label(), Some("built_pass".to_string()));
        assert!(desc.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_builder_repr() {
        let builder = PyComputePassBuilder::new();
        let repr = builder.__repr__();
        assert!(repr.contains("ComputePassBuilder"));
    }

    // -------------------------------------------------------------------------
    // PyDispatchDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_descriptor_direct() {
        let dispatch = PyDispatchDescriptor::direct(64, 64, 1).unwrap();
        assert!(dispatch.is_direct());
        assert!(!dispatch.is_indirect());
        assert_eq!(dispatch.workgroups(), [64, 64, 1]);
        assert_eq!(dispatch.x(), 64);
        assert_eq!(dispatch.y(), 64);
        assert_eq!(dispatch.z(), 1);
    }

    #[test]
    fn test_dispatch_descriptor_direct_1d() {
        let dispatch = PyDispatchDescriptor::direct_1d(128).unwrap();
        assert_eq!(dispatch.workgroups(), [128, 1, 1]);
    }

    #[test]
    fn test_dispatch_descriptor_direct_2d() {
        let dispatch = PyDispatchDescriptor::direct_2d(32, 32).unwrap();
        assert_eq!(dispatch.workgroups(), [32, 32, 1]);
    }

    #[test]
    fn test_dispatch_descriptor_direct_zero_x() {
        let result = PyDispatchDescriptor::direct(0, 64, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_descriptor_direct_zero_y() {
        let result = PyDispatchDescriptor::direct(64, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_descriptor_direct_zero_z() {
        let result = PyDispatchDescriptor::direct(64, 64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_descriptor_indirect() {
        use super::super::py_resource::{PyResourceHandle, PyResourceType};
        let buffer = PyResourceHandle::new(1, PyResourceType::Buffer);
        let dispatch = PyDispatchDescriptor::indirect(buffer.clone(), 128);
        assert!(dispatch.is_indirect());
        assert!(!dispatch.is_direct());
        assert_eq!(dispatch.indirect_offset(), 128);
        assert!(dispatch.indirect_buffer().is_some());
    }

    #[test]
    fn test_dispatch_descriptor_total_workgroups() {
        let dispatch = PyDispatchDescriptor::direct(8, 8, 4).unwrap();
        assert_eq!(dispatch.total_workgroups(), 256);
    }

    #[test]
    fn test_dispatch_descriptor_total_workgroups_indirect() {
        use super::super::py_resource::{PyResourceHandle, PyResourceType};
        let buffer = PyResourceHandle::new(1, PyResourceType::Buffer);
        let dispatch = PyDispatchDescriptor::indirect(buffer, 0);
        assert_eq!(dispatch.total_workgroups(), 0); // Unknown for indirect
    }

    #[test]
    fn test_dispatch_descriptor_validate() {
        let dispatch = PyDispatchDescriptor::direct(100, 100, 1).unwrap();
        assert!(dispatch.validate(200, 200, 200).is_none());
        assert!(dispatch.validate(50, 200, 200).is_some());
    }

    #[test]
    fn test_dispatch_descriptor_is_within_default_limits() {
        let valid = PyDispatchDescriptor::direct(1000, 1000, 1).unwrap();
        assert!(valid.is_within_default_limits());

        let invalid = PyDispatchDescriptor::direct(100000, 1, 1).unwrap();
        assert!(!invalid.is_within_default_limits());
    }

    #[test]
    fn test_dispatch_descriptor_repr_direct() {
        let dispatch = PyDispatchDescriptor::direct(64, 64, 1).unwrap();
        let repr = dispatch.__repr__();
        assert!(repr.contains("direct"));
        assert!(repr.contains("64"));
    }

    #[test]
    fn test_dispatch_descriptor_repr_indirect() {
        use super::super::py_resource::{PyResourceHandle, PyResourceType};
        let buffer = PyResourceHandle::new(42, PyResourceType::Buffer);
        let dispatch = PyDispatchDescriptor::indirect(buffer, 0);
        let repr = dispatch.__repr__();
        assert!(repr.contains("indirect"));
    }

    #[test]
    fn test_dispatch_descriptor_equality() {
        let a = PyDispatchDescriptor::direct(64, 64, 1).unwrap();
        let b = PyDispatchDescriptor::direct(64, 64, 1).unwrap();
        let c = PyDispatchDescriptor::direct(32, 32, 1).unwrap();
        assert!(a.__eq__(&b));
        assert!(a.__ne__(&c));
    }

    #[test]
    fn test_dispatch_descriptor_default() {
        let dispatch = PyDispatchDescriptor::default();
        assert_eq!(dispatch.workgroups(), [1, 1, 1]);
        assert!(dispatch.is_direct());
    }

    // -------------------------------------------------------------------------
    // PyPushConstantRange tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_constant_range_new() {
        let range = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        assert_eq!(range.offset(), 0);
        assert_eq!(range.size(), 64);
        assert_eq!(range.end_offset(), 64);
    }

    #[test]
    fn test_push_constant_range_compute() {
        let range = PyPushConstantRange::compute(32).unwrap();
        assert_eq!(range.offset(), 0);
        assert_eq!(range.size(), 32);
        assert!(range.stages().contains(&PyShaderStages::compute()));
    }

    #[test]
    fn test_push_constant_range_compute_at() {
        let range = PyPushConstantRange::compute_at(64, 32).unwrap();
        assert_eq!(range.offset(), 64);
        assert_eq!(range.size(), 32);
    }

    #[test]
    fn test_push_constant_range_invalid_offset() {
        let result = PyPushConstantRange::new(3, 64, PyShaderStages::compute());
        assert!(result.is_err());
    }

    #[test]
    fn test_push_constant_range_invalid_size() {
        let result = PyPushConstantRange::new(0, 63, PyShaderStages::compute());
        assert!(result.is_err());
    }

    #[test]
    fn test_push_constant_range_zero_size() {
        let result = PyPushConstantRange::new(0, 0, PyShaderStages::compute());
        assert!(result.is_err());
    }

    #[test]
    fn test_push_constant_range_overlaps() {
        let a = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let b = PyPushConstantRange::new(32, 64, PyShaderStages::compute()).unwrap();
        let c = PyPushConstantRange::new(64, 32, PyShaderStages::compute()).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_push_constant_range_is_valid_for_limit() {
        let range = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        assert!(range.is_valid_for_limit(128));
        assert!(!range.is_valid_for_limit(32));
    }

    #[test]
    fn test_push_constant_range_repr() {
        let range = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let repr = range.__repr__();
        assert!(repr.contains("PushConstantRange"));
        assert!(repr.contains("64"));
    }

    #[test]
    fn test_push_constant_range_str() {
        let range = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let s = range.__str__();
        assert!(s.contains("0..64"));
    }

    #[test]
    fn test_push_constant_range_equality() {
        let a = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let b = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let c = PyPushConstantRange::new(0, 32, PyShaderStages::compute()).unwrap();
        assert!(a.__eq__(&b));
        assert!(a.__ne__(&c));
    }

    #[test]
    fn test_push_constant_range_hash() {
        let a = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        let b = PyPushConstantRange::new(0, 64, PyShaderStages::compute()).unwrap();
        assert_eq!(a.__hash__(), b.__hash__());
    }

    // -------------------------------------------------------------------------
    // PyComputePipelineDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_pipeline_descriptor_new() {
        let desc = PyComputePipelineDescriptor::new();
        assert!(desc.label().is_none());
        assert!(desc.shader().is_empty());
        assert_eq!(desc.entry_point(), "main");
        assert!(!desc.has_push_constants());
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_shader_and_entry() {
        let desc = PyComputePipelineDescriptor::with_shader_and_entry("compute.wgsl", "main_cs");
        assert_eq!(desc.shader(), "compute.wgsl");
        assert_eq!(desc.entry_point(), "main_cs");
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_label() {
        let desc = PyComputePipelineDescriptor::new().with_label("my_pipeline");
        assert_eq!(desc.label(), Some("my_pipeline".to_string()));
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_shader() {
        let desc = PyComputePipelineDescriptor::new().with_shader("particles.wgsl");
        assert_eq!(desc.shader(), "particles.wgsl");
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_entry_point() {
        let desc = PyComputePipelineDescriptor::new().with_entry_point("update");
        assert_eq!(desc.entry_point(), "update");
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_push_constant() {
        let desc = PyComputePipelineDescriptor::new()
            .with_push_constant(0, 64, PyShaderStages::compute())
            .unwrap();
        assert!(desc.has_push_constants());
        assert_eq!(desc.push_constant_ranges().len(), 1);
    }

    #[test]
    fn test_compute_pipeline_descriptor_with_push_constant_range() {
        let range = PyPushConstantRange::new(0, 32, PyShaderStages::compute()).unwrap();
        let desc = PyComputePipelineDescriptor::new()
            .with_push_constant_range(range)
            .unwrap();
        assert!(desc.has_push_constants());
    }

    #[test]
    fn test_compute_pipeline_descriptor_overlapping_push_constants() {
        let desc = PyComputePipelineDescriptor::new()
            .with_push_constant(0, 64, PyShaderStages::compute())
            .unwrap();
        let result = desc.with_push_constant(32, 64, PyShaderStages::compute());
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_pipeline_descriptor_without_push_constants() {
        let desc = PyComputePipelineDescriptor::new()
            .with_push_constant(0, 64, PyShaderStages::compute())
            .unwrap()
            .without_push_constants();
        assert!(!desc.has_push_constants());
    }

    #[test]
    fn test_compute_pipeline_descriptor_total_push_constant_size() {
        let desc = PyComputePipelineDescriptor::new()
            .with_push_constant(0, 64, PyShaderStages::compute())
            .unwrap()
            .with_push_constant(64, 32, PyShaderStages::compute())
            .unwrap();
        assert_eq!(desc.total_push_constant_size(), 96);
    }

    #[test]
    fn test_compute_pipeline_descriptor_validate_empty_shader() {
        let desc = PyComputePipelineDescriptor::new();
        let error = desc.validate(128);
        assert!(error.is_some());
        assert!(error.unwrap().contains("Shader"));
    }

    #[test]
    fn test_compute_pipeline_descriptor_validate_empty_entry() {
        let desc = PyComputePipelineDescriptor::new()
            .with_shader("compute.wgsl")
            .with_entry_point("");
        let error = desc.validate(128);
        assert!(error.is_some());
        assert!(error.unwrap().contains("Entry point"));
    }

    #[test]
    fn test_compute_pipeline_descriptor_validate_push_constant_size() {
        let desc = PyComputePipelineDescriptor::new()
            .with_shader("compute.wgsl")
            .with_push_constant(0, 256, PyShaderStages::compute())
            .unwrap();
        let error = desc.validate(128);
        assert!(error.is_some());
        assert!(error.unwrap().contains("push constant size"));
    }

    #[test]
    fn test_compute_pipeline_descriptor_is_valid() {
        let valid = PyComputePipelineDescriptor::new()
            .with_shader("compute.wgsl")
            .with_entry_point("main");
        assert!(valid.is_valid());

        let invalid = PyComputePipelineDescriptor::new();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_compute_pipeline_descriptor_repr() {
        let desc = PyComputePipelineDescriptor::new()
            .with_label("test")
            .with_shader("compute.wgsl");
        let repr = desc.__repr__();
        assert!(repr.contains("ComputePipelineDescriptor"));
        assert!(repr.contains("test"));
        assert!(repr.contains("compute.wgsl"));
    }

    #[test]
    fn test_compute_pipeline_descriptor_str() {
        let desc = PyComputePipelineDescriptor::new()
            .with_label("physics")
            .with_shader("physics.wgsl")
            .with_entry_point("simulate");
        let s = desc.__str__();
        assert!(s.contains("physics"));
        assert!(s.contains("simulate"));
    }

    // -------------------------------------------------------------------------
    // Builder pattern flow tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_pattern_flow() {
        let mut builder = PyComputePassBuilder::new();
        builder = builder
            .label("complete_pass")
            .query_set(1)
            .timestamp_begin(0)
            .timestamp_end(1);

        let push_data = vec![0u8; 64];
        builder = builder.with_push_constants(push_data).unwrap();

        let desc = builder.build();
        assert_eq!(desc.label(), Some("complete_pass".to_string()));
        assert!(desc.has_timestamp_writes());
    }

    #[test]
    fn test_pipeline_builder_pattern_flow() {
        let desc = PyComputePipelineDescriptor::new()
            .with_label("particle_system")
            .with_shader("particles.wgsl")
            .with_entry_point("update")
            .with_push_constant(0, 16, PyShaderStages::compute())
            .unwrap()
            .with_push_constant(16, 16, PyShaderStages::compute())
            .unwrap();

        assert_eq!(desc.label(), Some("particle_system".to_string()));
        assert_eq!(desc.push_constant_ranges().len(), 2);
        assert_eq!(desc.total_push_constant_size(), 32);
    }

    // -------------------------------------------------------------------------
    // Default values tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_stages_default() {
        let default: PyShaderStages = Default::default();
        assert!(default.is_empty());
    }

    #[test]
    fn test_timestamp_writes_default() {
        let default: PyTimestampWrites = Default::default();
        assert_eq!(default.query_set_id(), 0);
        assert!(!default.is_enabled());
    }

    #[test]
    fn test_compute_pass_descriptor_default() {
        let default: PyComputePassDescriptor = Default::default();
        assert!(default.label().is_none());
        assert!(!default.has_timestamp_writes());
    }

    #[test]
    fn test_compute_pass_builder_default() {
        let default: PyComputePassBuilder = Default::default();
        assert!(!default.has_label());
        assert!(!default.has_timestamps());
    }

    #[test]
    fn test_compute_pipeline_descriptor_default() {
        let default: PyComputePipelineDescriptor = Default::default();
        assert!(default.label().is_none());
        assert!(default.shader().is_empty());
        assert_eq!(default.entry_point(), "main");
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_label() {
        let desc = PyComputePassDescriptor::new().with_label("");
        assert_eq!(desc.label(), Some(String::new()));
    }

    #[test]
    fn test_unicode_label() {
        let desc = PyComputePassDescriptor::new().with_label("计算通道_日本語");
        assert!(desc.label().is_some());
    }

    #[test]
    fn test_large_workgroup_counts() {
        let dispatch = PyDispatchDescriptor::direct(65535, 65535, 65535).unwrap();
        assert!(dispatch.is_within_default_limits());
    }

    #[test]
    fn test_multiple_push_constant_ranges() {
        let desc = PyComputePipelineDescriptor::new()
            .with_shader("test.wgsl")
            .with_push_constant(0, 16, PyShaderStages::compute())
            .unwrap()
            .with_push_constant(16, 16, PyShaderStages::compute())
            .unwrap()
            .with_push_constant(32, 16, PyShaderStages::compute())
            .unwrap()
            .with_push_constant(48, 16, PyShaderStages::compute())
            .unwrap();

        assert_eq!(desc.push_constant_ranges().len(), 4);
        assert_eq!(desc.total_push_constant_size(), 64);
    }

    #[test]
    fn test_max_query_indices() {
        let ts = PyTimestampWrites::new(u64::MAX).both(u32::MAX, u32::MAX);
        assert_eq!(ts.query_set_id(), u64::MAX);
        assert_eq!(ts.beginning_index(), Some(u32::MAX));
        assert_eq!(ts.end_index(), Some(u32::MAX));
    }
}
