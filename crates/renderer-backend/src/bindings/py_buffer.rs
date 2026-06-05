//! Python bindings for buffer descriptors (T-WGPU-P7.6.3).
//!
//! This module provides Python-accessible types for creating and configuring
//! GPU buffer descriptors, usage flags, and binding types.
//!
//! # Types
//!
//! - [`PyBufferDescriptor`] - Buffer creation parameters
//! - [`PyBufferUsage`] - Buffer usage flags (combinable with `|`)
//! - [`PyBufferBindingType`] - Binding type for shader access
//! - [`PyBufferSize`] - Size calculation utilities
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import BufferDescriptor, BufferUsage
//!
//! # Create a vertex buffer
//! vertex_buf = BufferDescriptor.vertex(1024 * 64)
//!     .with_label("mesh_vertices")
//!
//! # Create a storage buffer with combined usage
//! usage = BufferUsage.storage() | BufferUsage.copy_src() | BufferUsage.copy_dst()
//! storage = BufferDescriptor(4096).with_usage(usage)
//!
//! # Check if usage contains a flag
//! if storage.usage.contains(BufferUsage.storage()):
//!     print("Storage buffer!")
//! ```
//!
//! # Alignment
//!
//! Buffer sizes are NOT automatically aligned by these descriptors. Use
//! [`PyBufferSize::align_to()`] or [`PyBufferSize::uniform_aligned()`]
//! if you need aligned sizes.

use pyo3::prelude::*;
use std::fmt;

// ============================================================================
// PyBufferUsage
// ============================================================================

/// Buffer usage flags that control how a buffer can be used.
///
/// Usage flags can be combined using the `|` operator:
///
/// ```python
/// usage = BufferUsage.storage() | BufferUsage.copy_src()
/// ```
///
/// # Flag Reference
///
/// | Flag | Purpose |
/// |------|---------|
/// | `map_read` | CPU read access (staging readback) |
/// | `map_write` | CPU write access (staging upload) |
/// | `copy_src` | Source for copy operations |
/// | `copy_dst` | Destination for copy operations |
/// | `index` | Index buffer |
/// | `vertex` | Vertex buffer |
/// | `uniform` | Uniform buffer |
/// | `storage` | Storage buffer |
/// | `indirect` | Indirect draw/dispatch arguments |
#[pyclass(name = "BufferUsage")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyBufferUsage {
    bits: u32,
}

/// wgpu::BufferUsages bit values (matching wgpu 22.x)
mod usage_bits {
    pub const MAP_READ: u32 = 1 << 0;
    pub const MAP_WRITE: u32 = 1 << 1;
    pub const COPY_SRC: u32 = 1 << 2;
    pub const COPY_DST: u32 = 1 << 3;
    pub const INDEX: u32 = 1 << 4;
    pub const VERTEX: u32 = 1 << 5;
    pub const UNIFORM: u32 = 1 << 6;
    pub const STORAGE: u32 = 1 << 7;
    pub const INDIRECT: u32 = 1 << 8;
    pub const QUERY_RESOLVE: u32 = 1 << 9;
}

#[pymethods]
impl PyBufferUsage {
    /// Create a new usage flags instance with raw bits.
    ///
    /// Prefer using static constructor methods instead.
    #[new]
    pub fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Returns the raw bits value.
    #[getter]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Returns true if no flags are set.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Returns true if all bits in `other` are contained in `self`.
    pub fn contains(&self, other: &Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Returns true if any bit in `other` is also set in `self`.
    pub fn intersects(&self, other: &Self) -> bool {
        (self.bits & other.bits) != 0
    }

    // -- Static constructors --

    /// Empty usage flags (no flags set).
    #[staticmethod]
    pub fn empty() -> Self {
        Self { bits: 0 }
    }

    /// MAP_READ: CPU read access for staging readback.
    #[staticmethod]
    pub fn map_read() -> Self {
        Self { bits: usage_bits::MAP_READ }
    }

    /// MAP_WRITE: CPU write access for staging upload.
    #[staticmethod]
    pub fn map_write() -> Self {
        Self { bits: usage_bits::MAP_WRITE }
    }

    /// COPY_SRC: Buffer can be used as source in copy operations.
    #[staticmethod]
    pub fn copy_src() -> Self {
        Self { bits: usage_bits::COPY_SRC }
    }

    /// COPY_DST: Buffer can be used as destination in copy operations.
    #[staticmethod]
    pub fn copy_dst() -> Self {
        Self { bits: usage_bits::COPY_DST }
    }

    /// INDEX: Buffer can be used as an index buffer.
    #[staticmethod]
    pub fn index() -> Self {
        Self { bits: usage_bits::INDEX }
    }

    /// VERTEX: Buffer can be used as a vertex buffer.
    #[staticmethod]
    pub fn vertex() -> Self {
        Self { bits: usage_bits::VERTEX }
    }

    /// UNIFORM: Buffer can be used as a uniform buffer.
    #[staticmethod]
    pub fn uniform() -> Self {
        Self { bits: usage_bits::UNIFORM }
    }

    /// STORAGE: Buffer can be used as a storage buffer.
    #[staticmethod]
    pub fn storage() -> Self {
        Self { bits: usage_bits::STORAGE }
    }

    /// INDIRECT: Buffer can be used for indirect draw/dispatch arguments.
    #[staticmethod]
    pub fn indirect() -> Self {
        Self { bits: usage_bits::INDIRECT }
    }

    /// QUERY_RESOLVE: Buffer can be used for query result storage.
    #[staticmethod]
    pub fn query_resolve() -> Self {
        Self { bits: usage_bits::QUERY_RESOLVE }
    }

    // -- Preset combinations --

    /// Preset: VERTEX | COPY_DST (vertex buffer with CPU upload).
    #[staticmethod]
    pub fn vertex_preset() -> Self {
        Self {
            bits: usage_bits::VERTEX | usage_bits::COPY_DST,
        }
    }

    /// Preset: INDEX | COPY_DST (index buffer with CPU upload).
    #[staticmethod]
    pub fn index_preset() -> Self {
        Self {
            bits: usage_bits::INDEX | usage_bits::COPY_DST,
        }
    }

    /// Preset: UNIFORM | COPY_DST (uniform buffer with CPU upload).
    #[staticmethod]
    pub fn uniform_preset() -> Self {
        Self {
            bits: usage_bits::UNIFORM | usage_bits::COPY_DST,
        }
    }

    /// Preset: STORAGE | COPY_DST (read-only storage buffer).
    #[staticmethod]
    pub fn storage_read_preset() -> Self {
        Self {
            bits: usage_bits::STORAGE | usage_bits::COPY_DST,
        }
    }

    /// Preset: STORAGE | COPY_DST | COPY_SRC (read-write storage buffer).
    #[staticmethod]
    pub fn storage_rw_preset() -> Self {
        Self {
            bits: usage_bits::STORAGE | usage_bits::COPY_DST | usage_bits::COPY_SRC,
        }
    }

    /// Preset: MAP_WRITE | COPY_SRC (staging buffer for upload).
    #[staticmethod]
    pub fn staging_upload_preset() -> Self {
        Self {
            bits: usage_bits::MAP_WRITE | usage_bits::COPY_SRC,
        }
    }

    /// Preset: MAP_READ | COPY_DST (staging buffer for readback).
    #[staticmethod]
    pub fn staging_readback_preset() -> Self {
        Self {
            bits: usage_bits::MAP_READ | usage_bits::COPY_DST,
        }
    }

    /// Preset: INDIRECT | COPY_DST | STORAGE (indirect arguments buffer).
    #[staticmethod]
    pub fn indirect_preset() -> Self {
        Self {
            bits: usage_bits::INDIRECT | usage_bits::COPY_DST | usage_bits::STORAGE,
        }
    }

    // -- Python dunder methods --

    /// Combine usage flags with `|` operator.
    pub fn __or__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Combine usage flags with `|=` operator.
    pub fn __ior__(&mut self, other: &Self) {
        self.bits |= other.bits;
    }

    /// Intersect usage flags with `&` operator.
    pub fn __and__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    /// Subtract usage flags with `^` operator.
    pub fn __xor__(&self, other: &Self) -> Self {
        Self {
            bits: self.bits ^ other.bits,
        }
    }

    /// Invert usage flags with `~` operator.
    pub fn __invert__(&self) -> Self {
        Self { bits: !self.bits }
    }

    pub fn __repr__(&self) -> String {
        if self.bits == 0 {
            return "BufferUsage()".to_string();
        }

        let mut flags = Vec::new();
        if self.bits & usage_bits::MAP_READ != 0 {
            flags.push("MAP_READ");
        }
        if self.bits & usage_bits::MAP_WRITE != 0 {
            flags.push("MAP_WRITE");
        }
        if self.bits & usage_bits::COPY_SRC != 0 {
            flags.push("COPY_SRC");
        }
        if self.bits & usage_bits::COPY_DST != 0 {
            flags.push("COPY_DST");
        }
        if self.bits & usage_bits::INDEX != 0 {
            flags.push("INDEX");
        }
        if self.bits & usage_bits::VERTEX != 0 {
            flags.push("VERTEX");
        }
        if self.bits & usage_bits::UNIFORM != 0 {
            flags.push("UNIFORM");
        }
        if self.bits & usage_bits::STORAGE != 0 {
            flags.push("STORAGE");
        }
        if self.bits & usage_bits::INDIRECT != 0 {
            flags.push("INDIRECT");
        }
        if self.bits & usage_bits::QUERY_RESOLVE != 0 {
            flags.push("QUERY_RESOLVE");
        }

        format!("BufferUsage({})", flags.join(" | "))
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

    pub fn __bool__(&self) -> bool {
        self.bits != 0
    }
}

impl Default for PyBufferUsage {
    fn default() -> Self {
        Self { bits: 0 }
    }
}

// ============================================================================
// PyBufferBindingType
// ============================================================================

/// Buffer binding type for shader access.
///
/// Determines how a buffer is bound in a bind group for shader access.
#[pyclass(name = "BufferBindingType")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyBufferBindingType {
    /// Uniform buffer (read-only, small, per-draw data).
    Uniform,
    /// Storage buffer with read-write access.
    Storage,
    /// Storage buffer with read-only access.
    StorageReadOnly,
}

#[pymethods]
impl PyBufferBindingType {
    /// Returns true if this is a uniform binding.
    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform)
    }

    /// Returns true if this is a storage binding (read-only or read-write).
    pub fn is_storage(&self) -> bool {
        matches!(self, Self::Storage | Self::StorageReadOnly)
    }

    /// Returns true if this binding is read-only.
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Uniform | Self::StorageReadOnly)
    }

    /// Returns true if this binding allows writes.
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::Storage)
    }

    pub fn __repr__(&self) -> String {
        match self {
            Self::Uniform => "BufferBindingType.Uniform".to_string(),
            Self::Storage => "BufferBindingType.Storage".to_string(),
            Self::StorageReadOnly => "BufferBindingType.StorageReadOnly".to_string(),
        }
    }

    pub fn __str__(&self) -> String {
        match self {
            Self::Uniform => "uniform".to_string(),
            Self::Storage => "storage".to_string(),
            Self::StorageReadOnly => "storage_read_only".to_string(),
        }
    }

    pub fn __hash__(&self) -> u64 {
        match self {
            Self::Uniform => 0,
            Self::Storage => 1,
            Self::StorageReadOnly => 2,
        }
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ============================================================================
// PyBufferDescriptor
// ============================================================================

/// Descriptor for creating a GPU buffer.
///
/// Provides a builder-style API for configuring buffer parameters.
///
/// # Example
///
/// ```python
/// # Create a uniform buffer with 256 bytes
/// desc = BufferDescriptor.uniform(256).with_label("camera")
///
/// # Create a vertex buffer
/// desc = BufferDescriptor.vertex(1024 * 1024).with_label("mesh_verts")
///
/// # Create a custom buffer
/// desc = BufferDescriptor(4096)
///     .with_usage(BufferUsage.storage() | BufferUsage.copy_dst())
///     .with_label("particles")
///     .with_mapped(True)
/// ```
#[pyclass(name = "BufferDescriptor")]
#[derive(Clone, Debug)]
pub struct PyBufferDescriptor {
    size: u64,
    usage: PyBufferUsage,
    mapped_at_creation: bool,
    label: Option<String>,
}

#[pymethods]
impl PyBufferDescriptor {
    /// Create a new buffer descriptor with the given size.
    ///
    /// Default usage is empty (you must set usage before creation).
    /// Default mapped_at_creation is false.
    #[new]
    pub fn new(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::default(),
            mapped_at_creation: false,
            label: None,
        }
    }

    // -- Static constructors --

    /// Create a uniform buffer descriptor.
    ///
    /// Usage: UNIFORM | COPY_DST
    #[staticmethod]
    pub fn uniform(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::uniform_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create a storage buffer descriptor (read-only from shader).
    ///
    /// Usage: STORAGE | COPY_DST
    #[staticmethod]
    pub fn storage(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::storage_read_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create a storage buffer descriptor (read-write from shader).
    ///
    /// Usage: STORAGE | COPY_DST | COPY_SRC
    #[staticmethod]
    pub fn storage_rw(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::storage_rw_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create a vertex buffer descriptor.
    ///
    /// Usage: VERTEX | COPY_DST
    #[staticmethod]
    pub fn vertex(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::vertex_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create an index buffer descriptor.
    ///
    /// Usage: INDEX | COPY_DST
    #[staticmethod]
    pub fn index(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::index_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create an indirect buffer descriptor.
    ///
    /// Usage: INDIRECT | COPY_DST | STORAGE
    #[staticmethod]
    pub fn indirect(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::indirect_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    /// Create a staging buffer for CPU->GPU upload.
    ///
    /// Usage: MAP_WRITE | COPY_SRC
    /// Mapped at creation: true
    #[staticmethod]
    pub fn staging_upload(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::staging_upload_preset(),
            mapped_at_creation: true,
            label: None,
        }
    }

    /// Create a staging buffer for GPU->CPU readback.
    ///
    /// Usage: MAP_READ | COPY_DST
    #[staticmethod]
    pub fn staging_readback(size: u64) -> Self {
        Self {
            size,
            usage: PyBufferUsage::staging_readback_preset(),
            mapped_at_creation: false,
            label: None,
        }
    }

    // -- Builder methods --

    /// Set buffer usage flags.
    pub fn with_usage(&self, usage: PyBufferUsage) -> Self {
        Self {
            size: self.size,
            usage,
            mapped_at_creation: self.mapped_at_creation,
            label: self.label.clone(),
        }
    }

    /// Set debug label.
    pub fn with_label(&self, label: &str) -> Self {
        Self {
            size: self.size,
            usage: self.usage,
            mapped_at_creation: self.mapped_at_creation,
            label: Some(label.to_string()),
        }
    }

    /// Set mapped_at_creation flag.
    pub fn with_mapped(&self, mapped: bool) -> Self {
        Self {
            size: self.size,
            usage: self.usage,
            mapped_at_creation: mapped,
            label: self.label.clone(),
        }
    }

    /// Set buffer size.
    pub fn with_size(&self, size: u64) -> Self {
        Self {
            size,
            usage: self.usage,
            mapped_at_creation: self.mapped_at_creation,
            label: self.label.clone(),
        }
    }

    // -- Getters --

    /// Returns the buffer size in bytes.
    #[getter]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the buffer usage flags.
    #[getter]
    pub fn usage(&self) -> PyBufferUsage {
        self.usage
    }

    /// Returns the debug label (if any).
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Returns true if the buffer is mapped at creation.
    #[getter]
    pub fn mapped_at_creation(&self) -> bool {
        self.mapped_at_creation
    }

    // -- Validation --

    /// Returns true if the descriptor is valid for buffer creation.
    ///
    /// Checks:
    /// - Size > 0
    /// - Usage is not empty
    /// - No invalid usage combinations
    pub fn is_valid(&self) -> bool {
        if self.size == 0 {
            return false;
        }
        if self.usage.is_empty() {
            return false;
        }
        // Check for invalid combinations
        let bits = self.usage.bits;
        // MAP_READ and MAP_WRITE are mutually exclusive
        if (bits & usage_bits::MAP_READ != 0) && (bits & usage_bits::MAP_WRITE != 0) {
            return false;
        }
        // MAP_READ cannot be combined with VERTEX, INDEX, UNIFORM, STORAGE
        if bits & usage_bits::MAP_READ != 0 {
            let incompatible =
                usage_bits::VERTEX | usage_bits::INDEX | usage_bits::UNIFORM | usage_bits::STORAGE;
            if bits & incompatible != 0 {
                return false;
            }
        }
        true
    }

    /// Returns validation error message, or None if valid.
    pub fn validation_error(&self) -> Option<String> {
        if self.size == 0 {
            return Some("Buffer size must be > 0".to_string());
        }
        if self.usage.is_empty() {
            return Some("Buffer usage flags must not be empty".to_string());
        }
        let bits = self.usage.bits;
        if (bits & usage_bits::MAP_READ != 0) && (bits & usage_bits::MAP_WRITE != 0) {
            return Some("MAP_READ and MAP_WRITE are mutually exclusive".to_string());
        }
        if bits & usage_bits::MAP_READ != 0 {
            if bits & usage_bits::VERTEX != 0 {
                return Some("MAP_READ cannot be combined with VERTEX".to_string());
            }
            if bits & usage_bits::INDEX != 0 {
                return Some("MAP_READ cannot be combined with INDEX".to_string());
            }
            if bits & usage_bits::UNIFORM != 0 {
                return Some("MAP_READ cannot be combined with UNIFORM".to_string());
            }
            if bits & usage_bits::STORAGE != 0 {
                return Some("MAP_READ cannot be combined with STORAGE".to_string());
            }
        }
        None
    }

    // -- Python dunder methods --

    pub fn __repr__(&self) -> String {
        let label_str = self
            .label
            .as_ref()
            .map(|l| format!(", label='{}'", l))
            .unwrap_or_default();
        let mapped_str = if self.mapped_at_creation {
            ", mapped=True"
        } else {
            ""
        };
        format!(
            "BufferDescriptor(size={}, usage={}{}{})",
            self.size,
            self.usage.__repr__(),
            label_str,
            mapped_str
        )
    }

    pub fn __str__(&self) -> String {
        let label_str = self
            .label
            .as_ref()
            .map(|l| format!("'{}': ", l))
            .unwrap_or_default();
        format!(
            "{}{}B {}",
            label_str,
            self.size,
            self.usage.__str__()
        )
    }
}

impl Default for PyBufferDescriptor {
    fn default() -> Self {
        Self::new(0)
    }
}

// ============================================================================
// PyBufferSize
// ============================================================================

/// Buffer size calculation utilities.
///
/// Provides helper methods for calculating aligned buffer sizes,
/// which is critical for uniform buffer offsets and storage buffers.
#[pyclass(name = "BufferSize")]
pub struct PyBufferSize;

#[pymethods]
impl PyBufferSize {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Align a size up to the given alignment.
    ///
    /// Alignment must be a power of 2.
    #[staticmethod]
    pub fn align_to(size: u64, alignment: u64) -> u64 {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            // Invalid alignment (not power of 2), return original
            return size;
        }
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Align size to 4 bytes (wgpu minimum buffer alignment).
    #[staticmethod]
    pub fn align_4(size: u64) -> u64 {
        Self::align_to(size, 4)
    }

    /// Align size to 16 bytes (common for vec4 alignment).
    #[staticmethod]
    pub fn align_16(size: u64) -> u64 {
        Self::align_to(size, 16)
    }

    /// Align size to 256 bytes (wgpu uniform buffer offset alignment).
    #[staticmethod]
    pub fn uniform_aligned(size: u64) -> u64 {
        Self::align_to(size, 256)
    }

    /// Align size to 4 bytes (wgpu storage buffer offset alignment).
    #[staticmethod]
    pub fn storage_aligned(size: u64) -> u64 {
        Self::align_to(size, 4)
    }

    /// Calculate the total size for an array of uniformly-spaced elements.
    ///
    /// Each element is aligned to 256 bytes (uniform buffer alignment).
    #[staticmethod]
    pub fn uniform_array_size(element_size: u64, count: u64) -> u64 {
        if count == 0 {
            return 0;
        }
        let aligned_element = Self::uniform_aligned(element_size);
        aligned_element * count
    }

    /// Calculate the number of 256-byte slots needed for a size.
    #[staticmethod]
    pub fn uniform_slot_count(size: u64) -> u64 {
        if size == 0 {
            return 0;
        }
        (size + 255) / 256
    }

    /// Returns true if the alignment is a valid power of 2.
    #[staticmethod]
    pub fn is_valid_alignment(alignment: u64) -> bool {
        alignment > 0 && (alignment & (alignment - 1)) == 0
    }

    /// Returns the minimum binding size for a type.
    ///
    /// Used for `min_binding_size` in bind group layout entries.
    #[staticmethod]
    pub fn min_binding_size(type_size: u64) -> u64 {
        // Minimum binding size must be > 0 for most types
        type_size.max(1)
    }

    pub fn __repr__() -> String {
        "BufferSize".to_string()
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Registers the buffer bindings in a PyO3 module.
pub fn register_module(py: Python<'_>, parent: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    parent.add_class::<PyBufferUsage>()?;
    parent.add_class::<PyBufferBindingType>()?;
    parent.add_class::<PyBufferDescriptor>()?;
    parent.add_class::<PyBufferSize>()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PyBufferUsage tests --

    #[test]
    fn test_buffer_usage_new() {
        let usage = PyBufferUsage::new(0x55);
        assert_eq!(usage.bits(), 0x55);
    }

    #[test]
    fn test_buffer_usage_empty() {
        let usage = PyBufferUsage::empty();
        assert!(usage.is_empty());
        assert_eq!(usage.bits(), 0);
    }

    #[test]
    fn test_buffer_usage_static_constructors() {
        assert_eq!(PyBufferUsage::map_read().bits(), usage_bits::MAP_READ);
        assert_eq!(PyBufferUsage::map_write().bits(), usage_bits::MAP_WRITE);
        assert_eq!(PyBufferUsage::copy_src().bits(), usage_bits::COPY_SRC);
        assert_eq!(PyBufferUsage::copy_dst().bits(), usage_bits::COPY_DST);
        assert_eq!(PyBufferUsage::index().bits(), usage_bits::INDEX);
        assert_eq!(PyBufferUsage::vertex().bits(), usage_bits::VERTEX);
        assert_eq!(PyBufferUsage::uniform().bits(), usage_bits::UNIFORM);
        assert_eq!(PyBufferUsage::storage().bits(), usage_bits::STORAGE);
        assert_eq!(PyBufferUsage::indirect().bits(), usage_bits::INDIRECT);
        assert_eq!(PyBufferUsage::query_resolve().bits(), usage_bits::QUERY_RESOLVE);
    }

    #[test]
    fn test_buffer_usage_or_combination() {
        let a = PyBufferUsage::vertex();
        let b = PyBufferUsage::copy_dst();
        let combined = a.__or__(&b);
        assert_eq!(combined.bits(), usage_bits::VERTEX | usage_bits::COPY_DST);
    }

    #[test]
    fn test_buffer_usage_contains() {
        let combined = PyBufferUsage::vertex_preset();
        assert!(combined.contains(&PyBufferUsage::vertex()));
        assert!(combined.contains(&PyBufferUsage::copy_dst()));
        assert!(!combined.contains(&PyBufferUsage::storage()));
    }

    #[test]
    fn test_buffer_usage_intersects() {
        let a = PyBufferUsage::vertex_preset();
        let b = PyBufferUsage::storage_read_preset();
        // Both contain COPY_DST
        assert!(a.intersects(&PyBufferUsage::copy_dst()));
        assert!(b.intersects(&PyBufferUsage::copy_dst()));
        // a has VERTEX, b does not
        assert!(a.intersects(&PyBufferUsage::vertex()));
        assert!(!b.intersects(&PyBufferUsage::vertex()));
    }

    #[test]
    fn test_buffer_usage_presets() {
        let vertex = PyBufferUsage::vertex_preset();
        assert!(vertex.contains(&PyBufferUsage::vertex()));
        assert!(vertex.contains(&PyBufferUsage::copy_dst()));

        let index = PyBufferUsage::index_preset();
        assert!(index.contains(&PyBufferUsage::index()));
        assert!(index.contains(&PyBufferUsage::copy_dst()));

        let uniform = PyBufferUsage::uniform_preset();
        assert!(uniform.contains(&PyBufferUsage::uniform()));
        assert!(uniform.contains(&PyBufferUsage::copy_dst()));

        let storage = PyBufferUsage::storage_rw_preset();
        assert!(storage.contains(&PyBufferUsage::storage()));
        assert!(storage.contains(&PyBufferUsage::copy_dst()));
        assert!(storage.contains(&PyBufferUsage::copy_src()));
    }

    #[test]
    fn test_buffer_usage_repr() {
        let empty = PyBufferUsage::empty();
        assert_eq!(empty.__repr__(), "BufferUsage()");

        let single = PyBufferUsage::vertex();
        assert!(single.__repr__().contains("VERTEX"));

        let combined = PyBufferUsage::vertex_preset();
        let repr = combined.__repr__();
        assert!(repr.contains("VERTEX"));
        assert!(repr.contains("COPY_DST"));
    }

    #[test]
    fn test_buffer_usage_equality() {
        let a = PyBufferUsage::vertex();
        let b = PyBufferUsage::vertex();
        let c = PyBufferUsage::index();
        assert!(a.__eq__(&b));
        assert!(!a.__eq__(&c));
        assert!(a.__ne__(&c));
    }

    #[test]
    fn test_buffer_usage_hash() {
        let a = PyBufferUsage::vertex();
        let b = PyBufferUsage::vertex();
        assert_eq!(a.__hash__(), b.__hash__());
    }

    #[test]
    fn test_buffer_usage_bool() {
        assert!(!PyBufferUsage::empty().__bool__());
        assert!(PyBufferUsage::vertex().__bool__());
    }

    #[test]
    fn test_buffer_usage_and() {
        let a = PyBufferUsage::vertex_preset();
        let b = PyBufferUsage::copy_dst();
        let result = a.__and__(&b);
        assert_eq!(result.bits(), usage_bits::COPY_DST);
    }

    #[test]
    fn test_buffer_usage_xor() {
        let a = PyBufferUsage::vertex_preset();
        let b = PyBufferUsage::copy_dst();
        let result = a.__xor__(&b);
        assert_eq!(result.bits(), usage_bits::VERTEX);
    }

    // -- PyBufferBindingType tests --

    #[test]
    fn test_buffer_binding_type_uniform() {
        let binding = PyBufferBindingType::Uniform;
        assert!(binding.is_uniform());
        assert!(!binding.is_storage());
        assert!(binding.is_read_only());
        assert!(!binding.is_writable());
    }

    #[test]
    fn test_buffer_binding_type_storage() {
        let binding = PyBufferBindingType::Storage;
        assert!(!binding.is_uniform());
        assert!(binding.is_storage());
        assert!(!binding.is_read_only());
        assert!(binding.is_writable());
    }

    #[test]
    fn test_buffer_binding_type_storage_read_only() {
        let binding = PyBufferBindingType::StorageReadOnly;
        assert!(!binding.is_uniform());
        assert!(binding.is_storage());
        assert!(binding.is_read_only());
        assert!(!binding.is_writable());
    }

    #[test]
    fn test_buffer_binding_type_repr() {
        assert_eq!(PyBufferBindingType::Uniform.__repr__(), "BufferBindingType.Uniform");
        assert_eq!(PyBufferBindingType::Storage.__repr__(), "BufferBindingType.Storage");
        assert_eq!(
            PyBufferBindingType::StorageReadOnly.__repr__(),
            "BufferBindingType.StorageReadOnly"
        );
    }

    #[test]
    fn test_buffer_binding_type_str() {
        assert_eq!(PyBufferBindingType::Uniform.__str__(), "uniform");
        assert_eq!(PyBufferBindingType::Storage.__str__(), "storage");
        assert_eq!(PyBufferBindingType::StorageReadOnly.__str__(), "storage_read_only");
    }

    #[test]
    fn test_buffer_binding_type_hash() {
        assert_eq!(PyBufferBindingType::Uniform.__hash__(), 0);
        assert_eq!(PyBufferBindingType::Storage.__hash__(), 1);
        assert_eq!(PyBufferBindingType::StorageReadOnly.__hash__(), 2);
    }

    #[test]
    fn test_buffer_binding_type_equality() {
        assert!(PyBufferBindingType::Uniform.__eq__(&PyBufferBindingType::Uniform));
        assert!(!PyBufferBindingType::Uniform.__eq__(&PyBufferBindingType::Storage));
    }

    // -- PyBufferDescriptor tests --

    #[test]
    fn test_buffer_descriptor_new() {
        let desc = PyBufferDescriptor::new(1024);
        assert_eq!(desc.size(), 1024);
        assert!(desc.usage().is_empty());
        assert!(!desc.mapped_at_creation());
        assert!(desc.label().is_none());
    }

    #[test]
    fn test_buffer_descriptor_uniform() {
        let desc = PyBufferDescriptor::uniform(256);
        assert_eq!(desc.size(), 256);
        assert!(desc.usage().contains(&PyBufferUsage::uniform()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_dst()));
    }

    #[test]
    fn test_buffer_descriptor_storage() {
        let desc = PyBufferDescriptor::storage(4096);
        assert_eq!(desc.size(), 4096);
        assert!(desc.usage().contains(&PyBufferUsage::storage()));
    }

    #[test]
    fn test_buffer_descriptor_storage_rw() {
        let desc = PyBufferDescriptor::storage_rw(4096);
        assert!(desc.usage().contains(&PyBufferUsage::storage()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_src()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_dst()));
    }

    #[test]
    fn test_buffer_descriptor_vertex() {
        let desc = PyBufferDescriptor::vertex(1024);
        assert!(desc.usage().contains(&PyBufferUsage::vertex()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_dst()));
    }

    #[test]
    fn test_buffer_descriptor_index() {
        let desc = PyBufferDescriptor::index(1024);
        assert!(desc.usage().contains(&PyBufferUsage::index()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_dst()));
    }

    #[test]
    fn test_buffer_descriptor_indirect() {
        let desc = PyBufferDescriptor::indirect(1024);
        assert!(desc.usage().contains(&PyBufferUsage::indirect()));
        assert!(desc.usage().contains(&PyBufferUsage::storage()));
    }

    #[test]
    fn test_buffer_descriptor_staging_upload() {
        let desc = PyBufferDescriptor::staging_upload(1024);
        assert!(desc.usage().contains(&PyBufferUsage::map_write()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_src()));
        assert!(desc.mapped_at_creation());
    }

    #[test]
    fn test_buffer_descriptor_staging_readback() {
        let desc = PyBufferDescriptor::staging_readback(1024);
        assert!(desc.usage().contains(&PyBufferUsage::map_read()));
        assert!(desc.usage().contains(&PyBufferUsage::copy_dst()));
        assert!(!desc.mapped_at_creation());
    }

    #[test]
    fn test_buffer_descriptor_builder_methods() {
        let desc = PyBufferDescriptor::new(100)
            .with_usage(PyBufferUsage::vertex_preset())
            .with_label("test_buffer")
            .with_mapped(true);

        assert_eq!(desc.size(), 100);
        assert!(desc.usage().contains(&PyBufferUsage::vertex()));
        assert_eq!(desc.label(), Some("test_buffer".to_string()));
        assert!(desc.mapped_at_creation());
    }

    #[test]
    fn test_buffer_descriptor_with_size() {
        let desc = PyBufferDescriptor::uniform(256).with_size(512);
        assert_eq!(desc.size(), 512);
        assert!(desc.usage().contains(&PyBufferUsage::uniform()));
    }

    #[test]
    fn test_buffer_descriptor_is_valid() {
        // Valid descriptor
        let valid = PyBufferDescriptor::uniform(256);
        assert!(valid.is_valid());

        // Invalid: zero size
        let zero_size = PyBufferDescriptor::new(0).with_usage(PyBufferUsage::vertex());
        assert!(!zero_size.is_valid());

        // Invalid: empty usage
        let empty_usage = PyBufferDescriptor::new(1024);
        assert!(!empty_usage.is_valid());

        // Invalid: MAP_READ | MAP_WRITE
        let both_map = PyBufferDescriptor::new(1024)
            .with_usage(PyBufferUsage::map_read().__or__(&PyBufferUsage::map_write()));
        assert!(!both_map.is_valid());

        // Invalid: MAP_READ | VERTEX
        let map_read_vertex = PyBufferDescriptor::new(1024)
            .with_usage(PyBufferUsage::map_read().__or__(&PyBufferUsage::vertex()));
        assert!(!map_read_vertex.is_valid());
    }

    #[test]
    fn test_buffer_descriptor_validation_error() {
        let valid = PyBufferDescriptor::uniform(256);
        assert!(valid.validation_error().is_none());

        let zero_size = PyBufferDescriptor::new(0).with_usage(PyBufferUsage::vertex());
        assert!(zero_size.validation_error().unwrap().contains("size"));

        let empty_usage = PyBufferDescriptor::new(1024);
        assert!(empty_usage.validation_error().unwrap().contains("usage"));
    }

    #[test]
    fn test_buffer_descriptor_repr() {
        let desc = PyBufferDescriptor::uniform(256).with_label("test");
        let repr = desc.__repr__();
        assert!(repr.contains("256"));
        assert!(repr.contains("test"));
        assert!(repr.contains("UNIFORM"));
    }

    #[test]
    fn test_buffer_descriptor_str() {
        let desc = PyBufferDescriptor::uniform(256).with_label("test");
        let s = desc.__str__();
        assert!(s.contains("test"));
        assert!(s.contains("256"));
    }

    // -- PyBufferSize tests --

    #[test]
    fn test_buffer_size_align_to() {
        assert_eq!(PyBufferSize::align_to(0, 4), 0);
        assert_eq!(PyBufferSize::align_to(1, 4), 4);
        assert_eq!(PyBufferSize::align_to(4, 4), 4);
        assert_eq!(PyBufferSize::align_to(5, 4), 8);
        assert_eq!(PyBufferSize::align_to(100, 256), 256);
        assert_eq!(PyBufferSize::align_to(256, 256), 256);
        assert_eq!(PyBufferSize::align_to(257, 256), 512);
    }

    #[test]
    fn test_buffer_size_align_to_invalid_alignment() {
        // Non-power-of-2 alignment should return original size
        assert_eq!(PyBufferSize::align_to(100, 3), 100);
        assert_eq!(PyBufferSize::align_to(100, 0), 100);
    }

    #[test]
    fn test_buffer_size_align_4() {
        assert_eq!(PyBufferSize::align_4(0), 0);
        assert_eq!(PyBufferSize::align_4(1), 4);
        assert_eq!(PyBufferSize::align_4(4), 4);
        assert_eq!(PyBufferSize::align_4(5), 8);
    }

    #[test]
    fn test_buffer_size_align_16() {
        assert_eq!(PyBufferSize::align_16(0), 0);
        assert_eq!(PyBufferSize::align_16(1), 16);
        assert_eq!(PyBufferSize::align_16(16), 16);
        assert_eq!(PyBufferSize::align_16(17), 32);
    }

    #[test]
    fn test_buffer_size_uniform_aligned() {
        assert_eq!(PyBufferSize::uniform_aligned(0), 0);
        assert_eq!(PyBufferSize::uniform_aligned(1), 256);
        assert_eq!(PyBufferSize::uniform_aligned(256), 256);
        assert_eq!(PyBufferSize::uniform_aligned(257), 512);
    }

    #[test]
    fn test_buffer_size_storage_aligned() {
        assert_eq!(PyBufferSize::storage_aligned(0), 0);
        assert_eq!(PyBufferSize::storage_aligned(1), 4);
        assert_eq!(PyBufferSize::storage_aligned(4), 4);
        assert_eq!(PyBufferSize::storage_aligned(5), 8);
    }

    #[test]
    fn test_buffer_size_uniform_array_size() {
        assert_eq!(PyBufferSize::uniform_array_size(64, 0), 0);
        assert_eq!(PyBufferSize::uniform_array_size(64, 1), 256);
        assert_eq!(PyBufferSize::uniform_array_size(64, 4), 1024);
        assert_eq!(PyBufferSize::uniform_array_size(300, 2), 1024); // 512 * 2
    }

    #[test]
    fn test_buffer_size_uniform_slot_count() {
        assert_eq!(PyBufferSize::uniform_slot_count(0), 0);
        assert_eq!(PyBufferSize::uniform_slot_count(1), 1);
        assert_eq!(PyBufferSize::uniform_slot_count(256), 1);
        assert_eq!(PyBufferSize::uniform_slot_count(257), 2);
        assert_eq!(PyBufferSize::uniform_slot_count(512), 2);
    }

    #[test]
    fn test_buffer_size_is_valid_alignment() {
        assert!(!PyBufferSize::is_valid_alignment(0));
        assert!(PyBufferSize::is_valid_alignment(1));
        assert!(PyBufferSize::is_valid_alignment(2));
        assert!(!PyBufferSize::is_valid_alignment(3));
        assert!(PyBufferSize::is_valid_alignment(4));
        assert!(!PyBufferSize::is_valid_alignment(5));
        assert!(PyBufferSize::is_valid_alignment(256));
        assert!(!PyBufferSize::is_valid_alignment(255));
    }

    #[test]
    fn test_buffer_size_min_binding_size() {
        assert_eq!(PyBufferSize::min_binding_size(0), 1);
        assert_eq!(PyBufferSize::min_binding_size(1), 1);
        assert_eq!(PyBufferSize::min_binding_size(64), 64);
    }

    // -- Edge cases --

    #[test]
    fn test_zero_size_buffer() {
        let desc = PyBufferDescriptor::new(0);
        assert!(!desc.is_valid());
        assert!(desc.validation_error().is_some());
    }

    #[test]
    fn test_all_usages_combined() {
        let all = PyBufferUsage::map_write()
            .__or__(&PyBufferUsage::copy_src())
            .__or__(&PyBufferUsage::copy_dst())
            .__or__(&PyBufferUsage::vertex())
            .__or__(&PyBufferUsage::index())
            .__or__(&PyBufferUsage::uniform())
            .__or__(&PyBufferUsage::storage())
            .__or__(&PyBufferUsage::indirect());

        assert!(all.contains(&PyBufferUsage::vertex()));
        assert!(all.contains(&PyBufferUsage::storage()));
        assert!(!all.contains(&PyBufferUsage::map_read())); // Not added
    }

    #[test]
    fn test_large_buffer_size() {
        let desc = PyBufferDescriptor::storage(u64::MAX);
        assert_eq!(desc.size(), u64::MAX);
    }

    #[test]
    fn test_buffer_usage_invert() {
        let vertex = PyBufferUsage::vertex();
        let inverted = vertex.__invert__();
        assert!(!inverted.contains(&PyBufferUsage::vertex()));
    }
}
