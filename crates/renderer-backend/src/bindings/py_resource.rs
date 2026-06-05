//! Python bindings for resource handles (T-WGPU-P7.6.4).
//!
//! This module provides Python-accessible wrappers for GPU resource handles
//! and related types, enabling Python code to create, manage, and validate
//! resource references.
//!
//! # Feature Gate
//!
//! All types are gated behind the `pyo3` feature flag:
//!
//! ```toml
//! [features]
//! pyo3 = ["dep:pyo3"]
//! ```
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     PyResourceHandle, PyResourceType, PyResourcePool, PyResourceValidation
//! )
//!
//! # Create a resource pool
//! pool = PyResourcePool()
//!
//! # Allocate resources
//! buffer_handle = pool.allocate(PyResourceType.Buffer)
//! texture_handle = pool.allocate(PyResourceType.Texture)
//!
//! # Check handle validity
//! assert pool.is_valid(buffer_handle)
//! assert buffer_handle.is_buffer()
//! assert texture_handle.is_texture()
//!
//! # Release resources
//! pool.release(buffer_handle)
//! assert not pool.is_valid(buffer_handle)
//!
//! # Validation results
//! ok = PyResourceValidation.ok()
//! assert ok.is_valid()
//!
//! error = PyResourceValidation.error("Resource not found")
//! assert not error.is_valid()
//! assert error.message() == "Resource not found"
//! ```

use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// PyResourceType
// ---------------------------------------------------------------------------

/// Python-exposed resource type enumeration.
///
/// Represents the different types of GPU resources that can be managed.
#[pyclass(name = "ResourceType")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyResourceType {
    /// GPU buffer (vertex, index, uniform, storage).
    Buffer,
    /// 2D or 3D texture resource.
    Texture,
    /// Texture sampler.
    Sampler,
    /// Bind group containing resource bindings.
    BindGroup,
    /// Render pipeline (vertex + fragment shaders).
    RenderPipeline,
    /// Compute pipeline (compute shader).
    ComputePipeline,
}

#[pymethods]
impl PyResourceType {
    /// Returns the canonical name of this resource type.
    pub fn name(&self) -> &str {
        match self {
            Self::Buffer => "Buffer",
            Self::Texture => "Texture",
            Self::Sampler => "Sampler",
            Self::BindGroup => "BindGroup",
            Self::RenderPipeline => "RenderPipeline",
            Self::ComputePipeline => "ComputePipeline",
        }
    }

    /// Returns true if this is a GPU-resident resource (buffer, texture, sampler).
    pub fn is_gpu_resource(&self) -> bool {
        matches!(self, Self::Buffer | Self::Texture | Self::Sampler)
    }

    /// Returns true if this is a pipeline resource.
    pub fn is_pipeline(&self) -> bool {
        matches!(self, Self::RenderPipeline | Self::ComputePipeline)
    }

    fn __repr__(&self) -> String {
        format!("ResourceType.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __hash__(&self) -> u64 {
        *self as u64
    }

    fn __eq__(&self, other: &Self) -> bool {
        *self == *other
    }

    fn __ne__(&self, other: &Self) -> bool {
        *self != *other
    }
}

impl Default for PyResourceType {
    fn default() -> Self {
        Self::Buffer
    }
}

// ---------------------------------------------------------------------------
// PyResourceHandle
// ---------------------------------------------------------------------------

/// Python-exposed resource handle.
///
/// A lightweight handle that identifies a GPU resource. Handles include
/// a unique ID, resource type, and generation counter for validation.
///
/// # Generation Tracking
///
/// The generation counter allows detection of stale handles - when a resource
/// is released and its ID is reused, the generation increments, invalidating
/// old handles pointing to the same ID.
#[pyclass(name = "ResourceHandle")]
#[derive(Clone, Debug)]
pub struct PyResourceHandle {
    id: u64,
    resource_type: PyResourceType,
    generation: u32,
}

#[pymethods]
impl PyResourceHandle {
    /// Creates a new resource handle.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the resource
    /// * `resource_type` - Type of the resource
    #[new]
    pub fn new(id: u64, resource_type: PyResourceType) -> Self {
        Self {
            id,
            resource_type,
            generation: 0,
        }
    }

    /// Creates a new resource handle with a specific generation.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the resource
    /// * `resource_type` - Type of the resource
    /// * `generation` - Generation counter for validity tracking
    #[staticmethod]
    pub fn with_generation(id: u64, resource_type: PyResourceType, generation: u32) -> Self {
        Self {
            id,
            resource_type,
            generation,
        }
    }

    /// Returns the unique identifier of this resource.
    #[getter]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the type of this resource.
    #[getter]
    pub fn resource_type(&self) -> PyResourceType {
        self.resource_type
    }

    /// Returns the generation counter of this handle.
    #[getter]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Returns true if this handle has a valid (non-zero) ID.
    ///
    /// Note: This only checks if the ID is non-zero. To check if the
    /// handle refers to a valid resource in a pool, use `PyResourcePool.is_valid()`.
    pub fn is_valid(&self) -> bool {
        self.id != 0
    }

    /// Returns true if this handle refers to a buffer resource.
    pub fn is_buffer(&self) -> bool {
        self.resource_type == PyResourceType::Buffer
    }

    /// Returns true if this handle refers to a texture resource.
    pub fn is_texture(&self) -> bool {
        self.resource_type == PyResourceType::Texture
    }

    /// Returns true if this handle refers to a sampler resource.
    pub fn is_sampler(&self) -> bool {
        self.resource_type == PyResourceType::Sampler
    }

    /// Returns true if this handle refers to a bind group.
    pub fn is_bind_group(&self) -> bool {
        self.resource_type == PyResourceType::BindGroup
    }

    /// Returns true if this handle refers to a render pipeline.
    pub fn is_render_pipeline(&self) -> bool {
        self.resource_type == PyResourceType::RenderPipeline
    }

    /// Returns true if this handle refers to a compute pipeline.
    pub fn is_compute_pipeline(&self) -> bool {
        self.resource_type == PyResourceType::ComputePipeline
    }

    /// Returns true if this handle refers to a GPU-resident resource.
    pub fn is_gpu_resource(&self) -> bool {
        self.resource_type.is_gpu_resource()
    }

    /// Returns true if this handle refers to a pipeline resource.
    pub fn is_pipeline(&self) -> bool {
        self.resource_type.is_pipeline()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id
            && self.resource_type == other.resource_type
            && self.generation == other.generation
    }

    fn __ne__(&self, other: &Self) -> bool {
        !self.__eq__(other)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.resource_type.hash(&mut hasher);
        self.generation.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceHandle(id={}, type={}, gen={})",
            self.id,
            self.resource_type.name(),
            self.generation
        )
    }
}

impl Default for PyResourceHandle {
    fn default() -> Self {
        Self {
            id: 0,
            resource_type: PyResourceType::Buffer,
            generation: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PyResourcePool
// ---------------------------------------------------------------------------

/// Resource handle pool for allocating and managing resource handles.
///
/// The pool tracks allocated handles and their generations to support
/// handle validation and safe resource reuse.
///
/// # Example
///
/// ```python
/// pool = PyResourcePool()
///
/// # Allocate a buffer handle
/// handle = pool.allocate(PyResourceType.Buffer)
/// print(f"Allocated: {handle}")  # ResourceHandle(id=1, type=Buffer, gen=0)
///
/// # Check validity
/// assert pool.is_valid(handle)
///
/// # Release the handle
/// pool.release(handle)
/// assert not pool.is_valid(handle)
/// ```
#[pyclass(name = "ResourcePool")]
#[derive(Clone, Debug)]
pub struct PyResourcePool {
    next_id: u64,
    generation: u32,
    /// Maps resource ID to (resource_type, generation) for validation.
    allocated: std::collections::HashMap<u64, (PyResourceType, u32)>,
}

#[pymethods]
impl PyResourcePool {
    /// Creates a new empty resource pool.
    #[new]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            generation: 0,
            allocated: std::collections::HashMap::new(),
        }
    }

    /// Allocates a new resource handle of the specified type.
    ///
    /// # Arguments
    /// * `resource_type` - Type of resource to allocate
    ///
    /// # Returns
    /// A new resource handle with a unique ID
    pub fn allocate(&mut self, resource_type: PyResourceType) -> PyResourceHandle {
        let id = self.next_id;
        self.next_id += 1;

        let handle = PyResourceHandle {
            id,
            resource_type,
            generation: self.generation,
        };

        self.allocated.insert(id, (resource_type, self.generation));
        handle
    }

    /// Releases a resource handle, making it invalid.
    ///
    /// # Arguments
    /// * `handle` - Handle to release
    ///
    /// # Returns
    /// True if the handle was valid and was released, false otherwise
    pub fn release(&mut self, handle: &PyResourceHandle) -> bool {
        if let Some((stored_type, stored_gen)) = self.allocated.get(&handle.id) {
            if *stored_type == handle.resource_type && *stored_gen == handle.generation {
                self.allocated.remove(&handle.id);
                return true;
            }
        }
        false
    }

    /// Checks if a handle is currently valid in this pool.
    ///
    /// A handle is valid if:
    /// - Its ID exists in the pool
    /// - Its resource type matches
    /// - Its generation matches (handle is not stale)
    ///
    /// # Arguments
    /// * `handle` - Handle to validate
    ///
    /// # Returns
    /// True if the handle is valid
    pub fn is_valid(&self, handle: &PyResourceHandle) -> bool {
        if let Some((stored_type, stored_gen)) = self.allocated.get(&handle.id) {
            return *stored_type == handle.resource_type && *stored_gen == handle.generation;
        }
        false
    }

    /// Returns the number of currently allocated resources.
    pub fn count(&self) -> u64 {
        self.allocated.len() as u64
    }

    /// Returns the current generation counter of the pool.
    #[getter]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Resets the pool, invalidating all allocated handles.
    ///
    /// The generation counter is incremented to ensure any outstanding
    /// handles become invalid.
    pub fn reset(&mut self) {
        self.allocated.clear();
        self.generation = self.generation.wrapping_add(1);
        // Keep next_id to avoid reusing IDs with different generations
    }

    /// Returns a list of all currently allocated handle IDs.
    pub fn allocated_ids(&self) -> Vec<u64> {
        self.allocated.keys().copied().collect()
    }

    /// Returns the count of resources by type.
    pub fn count_by_type(&self, resource_type: PyResourceType) -> u64 {
        self.allocated
            .values()
            .filter(|(t, _)| *t == resource_type)
            .count() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourcePool(count={}, generation={})",
            self.count(),
            self.generation
        )
    }

    fn __len__(&self) -> usize {
        self.allocated.len()
    }
}

impl Default for PyResourcePool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PyResourceValidation
// ---------------------------------------------------------------------------

/// Result of a resource validation operation.
///
/// Used to communicate validation status and error messages from
/// resource operations.
///
/// # Example
///
/// ```python
/// # Success case
/// ok = PyResourceValidation.ok()
/// assert ok.is_valid()
/// assert ok.message() is None
///
/// # Error case
/// error = PyResourceValidation.error("Invalid handle")
/// assert not error.is_valid()
/// assert error.message() == "Invalid handle"
/// ```
#[pyclass(name = "ResourceValidation")]
#[derive(Clone, Debug)]
pub struct PyResourceValidation {
    valid: bool,
    message: Option<String>,
}

#[pymethods]
impl PyResourceValidation {
    /// Creates a new validation result.
    ///
    /// # Arguments
    /// * `valid` - Whether the validation passed
    /// * `message` - Optional error message (typically only for failures)
    #[new]
    pub fn new(valid: bool, message: Option<String>) -> Self {
        Self { valid, message }
    }

    /// Creates a successful validation result.
    #[staticmethod]
    pub fn ok() -> Self {
        Self {
            valid: true,
            message: None,
        }
    }

    /// Creates a failed validation result with an error message.
    ///
    /// # Arguments
    /// * `message` - Description of the validation failure
    #[staticmethod]
    pub fn error(message: &str) -> Self {
        Self {
            valid: false,
            message: Some(message.to_string()),
        }
    }

    /// Creates a validation result indicating the resource was not found.
    #[staticmethod]
    pub fn not_found(resource_id: u64) -> Self {
        Self {
            valid: false,
            message: Some(format!("Resource not found: {}", resource_id)),
        }
    }

    /// Creates a validation result indicating a generation mismatch.
    #[staticmethod]
    pub fn stale_handle(expected_gen: u32, actual_gen: u32) -> Self {
        Self {
            valid: false,
            message: Some(format!(
                "Stale handle: expected generation {}, got {}",
                expected_gen, actual_gen
            )),
        }
    }

    /// Creates a validation result indicating a type mismatch.
    #[staticmethod]
    pub fn type_mismatch(expected: PyResourceType, actual: PyResourceType) -> Self {
        Self {
            valid: false,
            message: Some(format!(
                "Type mismatch: expected {}, got {}",
                expected.name(),
                actual.name()
            )),
        }
    }

    /// Returns true if the validation passed.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Returns the error message, if any.
    pub fn message(&self) -> Option<String> {
        self.message.clone()
    }

    /// Returns true if this represents an error.
    pub fn is_error(&self) -> bool {
        !self.valid
    }

    fn __repr__(&self) -> String {
        if self.valid {
            "ResourceValidation.ok()".to_string()
        } else {
            format!(
                "ResourceValidation.error({:?})",
                self.message.as_deref().unwrap_or("unknown")
            )
        }
    }

    fn __bool__(&self) -> bool {
        self.valid
    }
}

impl Default for PyResourceValidation {
    fn default() -> Self {
        Self::ok()
    }
}

// ---------------------------------------------------------------------------
// Module Registration
// ---------------------------------------------------------------------------

/// Registers the resource handle types with the Python module.
pub fn register_module(
    py: Python<'_>,
    parent: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    let m = pyo3::types::PyModule::new(py, "resource")?;

    m.add_class::<PyResourceType>()?;
    m.add_class::<PyResourceHandle>()?;
    m.add_class::<PyResourcePool>()?;
    m.add_class::<PyResourceValidation>()?;

    parent.add_submodule(&m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PyResourceType tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_type_name() {
        assert_eq!(PyResourceType::Buffer.name(), "Buffer");
        assert_eq!(PyResourceType::Texture.name(), "Texture");
        assert_eq!(PyResourceType::Sampler.name(), "Sampler");
        assert_eq!(PyResourceType::BindGroup.name(), "BindGroup");
        assert_eq!(PyResourceType::RenderPipeline.name(), "RenderPipeline");
        assert_eq!(PyResourceType::ComputePipeline.name(), "ComputePipeline");
    }

    #[test]
    fn test_resource_type_is_gpu_resource() {
        assert!(PyResourceType::Buffer.is_gpu_resource());
        assert!(PyResourceType::Texture.is_gpu_resource());
        assert!(PyResourceType::Sampler.is_gpu_resource());
        assert!(!PyResourceType::BindGroup.is_gpu_resource());
        assert!(!PyResourceType::RenderPipeline.is_gpu_resource());
        assert!(!PyResourceType::ComputePipeline.is_gpu_resource());
    }

    #[test]
    fn test_resource_type_is_pipeline() {
        assert!(!PyResourceType::Buffer.is_pipeline());
        assert!(!PyResourceType::Texture.is_pipeline());
        assert!(!PyResourceType::Sampler.is_pipeline());
        assert!(!PyResourceType::BindGroup.is_pipeline());
        assert!(PyResourceType::RenderPipeline.is_pipeline());
        assert!(PyResourceType::ComputePipeline.is_pipeline());
    }

    #[test]
    fn test_resource_type_equality() {
        assert_eq!(PyResourceType::Buffer, PyResourceType::Buffer);
        assert_ne!(PyResourceType::Buffer, PyResourceType::Texture);
    }

    #[test]
    fn test_resource_type_repr() {
        assert_eq!(PyResourceType::Buffer.__repr__(), "ResourceType.Buffer");
        assert_eq!(PyResourceType::Texture.__str__(), "Texture");
    }

    // -------------------------------------------------------------------------
    // PyResourceHandle tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_handle_new() {
        let handle = PyResourceHandle::new(42, PyResourceType::Buffer);
        assert_eq!(handle.id(), 42);
        assert_eq!(handle.resource_type(), PyResourceType::Buffer);
        assert_eq!(handle.generation(), 0);
    }

    #[test]
    fn test_resource_handle_with_generation() {
        let handle = PyResourceHandle::with_generation(10, PyResourceType::Texture, 5);
        assert_eq!(handle.id(), 10);
        assert_eq!(handle.resource_type(), PyResourceType::Texture);
        assert_eq!(handle.generation(), 5);
    }

    #[test]
    fn test_resource_handle_is_valid() {
        let valid = PyResourceHandle::new(1, PyResourceType::Buffer);
        let invalid = PyResourceHandle::new(0, PyResourceType::Buffer);
        assert!(valid.is_valid());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_resource_handle_type_checks() {
        let buffer = PyResourceHandle::new(1, PyResourceType::Buffer);
        let texture = PyResourceHandle::new(2, PyResourceType::Texture);
        let sampler = PyResourceHandle::new(3, PyResourceType::Sampler);
        let bind_group = PyResourceHandle::new(4, PyResourceType::BindGroup);
        let render_pipeline = PyResourceHandle::new(5, PyResourceType::RenderPipeline);
        let compute_pipeline = PyResourceHandle::new(6, PyResourceType::ComputePipeline);

        assert!(buffer.is_buffer());
        assert!(!buffer.is_texture());

        assert!(texture.is_texture());
        assert!(!texture.is_buffer());

        assert!(sampler.is_sampler());
        assert!(!sampler.is_bind_group());

        assert!(bind_group.is_bind_group());
        assert!(!bind_group.is_pipeline());

        assert!(render_pipeline.is_render_pipeline());
        assert!(render_pipeline.is_pipeline());

        assert!(compute_pipeline.is_compute_pipeline());
        assert!(compute_pipeline.is_pipeline());
    }

    #[test]
    fn test_resource_handle_is_gpu_resource() {
        assert!(PyResourceHandle::new(1, PyResourceType::Buffer).is_gpu_resource());
        assert!(PyResourceHandle::new(2, PyResourceType::Texture).is_gpu_resource());
        assert!(PyResourceHandle::new(3, PyResourceType::Sampler).is_gpu_resource());
        assert!(!PyResourceHandle::new(4, PyResourceType::BindGroup).is_gpu_resource());
        assert!(!PyResourceHandle::new(5, PyResourceType::RenderPipeline).is_gpu_resource());
    }

    #[test]
    fn test_resource_handle_equality() {
        let a = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 0);
        let b = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 0);
        let c = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 1);
        let d = PyResourceHandle::with_generation(1, PyResourceType::Texture, 0);
        let e = PyResourceHandle::with_generation(2, PyResourceType::Buffer, 0);

        assert!(a.__eq__(&b));
        assert!(!a.__eq__(&c)); // Different generation
        assert!(!a.__eq__(&d)); // Different type
        assert!(!a.__eq__(&e)); // Different id
    }

    #[test]
    fn test_resource_handle_hash() {
        let a = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 0);
        let b = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 0);
        let c = PyResourceHandle::with_generation(1, PyResourceType::Buffer, 1);

        assert_eq!(a.__hash__(), b.__hash__());
        assert_ne!(a.__hash__(), c.__hash__());
    }

    #[test]
    fn test_resource_handle_repr() {
        let handle = PyResourceHandle::with_generation(42, PyResourceType::Texture, 3);
        let repr = handle.__repr__();
        assert!(repr.contains("42"));
        assert!(repr.contains("Texture"));
        assert!(repr.contains("3"));
    }

    // -------------------------------------------------------------------------
    // PyResourcePool tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_pool_new() {
        let pool = PyResourcePool::new();
        assert_eq!(pool.count(), 0);
        assert_eq!(pool.generation(), 0);
    }

    #[test]
    fn test_resource_pool_allocate() {
        let mut pool = PyResourcePool::new();

        let h1 = pool.allocate(PyResourceType::Buffer);
        assert_eq!(h1.id(), 1);
        assert_eq!(h1.resource_type(), PyResourceType::Buffer);
        assert_eq!(pool.count(), 1);

        let h2 = pool.allocate(PyResourceType::Texture);
        assert_eq!(h2.id(), 2);
        assert_eq!(h2.resource_type(), PyResourceType::Texture);
        assert_eq!(pool.count(), 2);

        // IDs should be unique
        assert_ne!(h1.id(), h2.id());
    }

    #[test]
    fn test_resource_pool_is_valid() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        assert!(pool.is_valid(&handle));

        // Invalid handle (wrong ID)
        let fake = PyResourceHandle::new(999, PyResourceType::Buffer);
        assert!(!pool.is_valid(&fake));
    }

    #[test]
    fn test_resource_pool_release() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        assert!(pool.is_valid(&handle));
        assert_eq!(pool.count(), 1);

        let released = pool.release(&handle);
        assert!(released);
        assert!(!pool.is_valid(&handle));
        assert_eq!(pool.count(), 0);

        // Double release should return false
        let released_again = pool.release(&handle);
        assert!(!released_again);
    }

    #[test]
    fn test_resource_pool_release_invalid() {
        let mut pool = PyResourcePool::new();
        let fake = PyResourceHandle::new(999, PyResourceType::Buffer);

        let released = pool.release(&fake);
        assert!(!released);
    }

    #[test]
    fn test_resource_pool_reset() {
        let mut pool = PyResourcePool::new();
        let h1 = pool.allocate(PyResourceType::Buffer);
        let h2 = pool.allocate(PyResourceType::Texture);

        assert_eq!(pool.count(), 2);
        let gen_before = pool.generation();

        pool.reset();

        assert_eq!(pool.count(), 0);
        assert_eq!(pool.generation(), gen_before + 1);

        // Old handles should be invalid
        assert!(!pool.is_valid(&h1));
        assert!(!pool.is_valid(&h2));
    }

    #[test]
    fn test_resource_pool_generation_mismatch() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        // Create a handle with wrong generation
        let stale = PyResourceHandle::with_generation(
            handle.id(),
            handle.resource_type(),
            handle.generation() + 1,
        );

        assert!(!pool.is_valid(&stale));
    }

    #[test]
    fn test_resource_pool_type_mismatch() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        // Create a handle with wrong type
        let wrong_type = PyResourceHandle::with_generation(
            handle.id(),
            PyResourceType::Texture,
            handle.generation(),
        );

        assert!(!pool.is_valid(&wrong_type));
    }

    #[test]
    fn test_resource_pool_allocated_ids() {
        let mut pool = PyResourcePool::new();
        pool.allocate(PyResourceType::Buffer);
        pool.allocate(PyResourceType::Texture);
        pool.allocate(PyResourceType::Sampler);

        let ids = pool.allocated_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_resource_pool_count_by_type() {
        let mut pool = PyResourcePool::new();
        pool.allocate(PyResourceType::Buffer);
        pool.allocate(PyResourceType::Buffer);
        pool.allocate(PyResourceType::Texture);

        assert_eq!(pool.count_by_type(PyResourceType::Buffer), 2);
        assert_eq!(pool.count_by_type(PyResourceType::Texture), 1);
        assert_eq!(pool.count_by_type(PyResourceType::Sampler), 0);
    }

    // -------------------------------------------------------------------------
    // PyResourceValidation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validation_ok() {
        let v = PyResourceValidation::ok();
        assert!(v.is_valid());
        assert!(!v.is_error());
        assert!(v.message().is_none());
    }

    #[test]
    fn test_validation_error() {
        let v = PyResourceValidation::error("Something went wrong");
        assert!(!v.is_valid());
        assert!(v.is_error());
        assert_eq!(v.message(), Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_validation_not_found() {
        let v = PyResourceValidation::not_found(42);
        assert!(!v.is_valid());
        assert!(v.message().unwrap().contains("42"));
    }

    #[test]
    fn test_validation_stale_handle() {
        let v = PyResourceValidation::stale_handle(5, 3);
        assert!(!v.is_valid());
        let msg = v.message().unwrap();
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_validation_type_mismatch() {
        let v = PyResourceValidation::type_mismatch(
            PyResourceType::Buffer,
            PyResourceType::Texture,
        );
        assert!(!v.is_valid());
        let msg = v.message().unwrap();
        assert!(msg.contains("Buffer"));
        assert!(msg.contains("Texture"));
    }

    #[test]
    fn test_validation_new() {
        let valid = PyResourceValidation::new(true, None);
        assert!(valid.is_valid());

        let invalid = PyResourceValidation::new(false, Some("Custom error".to_string()));
        assert!(!invalid.is_valid());
        assert_eq!(invalid.message(), Some("Custom error".to_string()));
    }

    #[test]
    fn test_validation_repr() {
        let ok = PyResourceValidation::ok();
        assert_eq!(ok.__repr__(), "ResourceValidation.ok()");

        let err = PyResourceValidation::error("test error");
        assert!(err.__repr__().contains("test error"));
    }

    #[test]
    fn test_validation_bool() {
        let ok = PyResourceValidation::ok();
        let err = PyResourceValidation::error("fail");

        assert!(ok.__bool__());
        assert!(!err.__bool__());
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_handle_zero_id() {
        let handle = PyResourceHandle::new(0, PyResourceType::Buffer);
        assert!(!handle.is_valid());
    }

    #[test]
    fn test_pool_double_release() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        assert!(pool.release(&handle));
        assert!(!pool.release(&handle)); // Second release fails
    }

    #[test]
    fn test_pool_release_wrong_generation() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        // Manually create a stale handle
        let stale = PyResourceHandle::with_generation(
            handle.id(),
            handle.resource_type(),
            handle.generation() + 100,
        );

        assert!(!pool.release(&stale));
        assert!(pool.is_valid(&handle)); // Original should still be valid
    }

    #[test]
    fn test_pool_release_wrong_type() {
        let mut pool = PyResourcePool::new();
        let handle = pool.allocate(PyResourceType::Buffer);

        // Manually create a handle with wrong type
        let wrong = PyResourceHandle::with_generation(
            handle.id(),
            PyResourceType::Texture,
            handle.generation(),
        );

        assert!(!pool.release(&wrong));
        assert!(pool.is_valid(&handle)); // Original should still be valid
    }

    #[test]
    fn test_pool_generation_wrapping() {
        let mut pool = PyResourcePool {
            next_id: 1,
            generation: u32::MAX,
            allocated: std::collections::HashMap::new(),
        };

        pool.reset();
        assert_eq!(pool.generation(), 0); // Should wrap around
    }

    #[test]
    fn test_handle_default() {
        let handle = PyResourceHandle::default();
        assert_eq!(handle.id(), 0);
        assert_eq!(handle.resource_type(), PyResourceType::Buffer);
        assert_eq!(handle.generation(), 0);
        assert!(!handle.is_valid());
    }

    #[test]
    fn test_validation_default() {
        let v = PyResourceValidation::default();
        assert!(v.is_valid());
        assert!(v.message().is_none());
    }

    #[test]
    fn test_resource_type_default() {
        let t = PyResourceType::default();
        assert_eq!(t, PyResourceType::Buffer);
    }
}
