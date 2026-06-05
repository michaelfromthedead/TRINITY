//! Python bindings for TRINITY wgpu types.
//!
//! This module provides Python-accessible wrappers for wgpu resource descriptors
//! and related types, enabling Python code to create and configure GPU resources.
//!
//! # Modules
//!
//! - [`py_buffer`] - Buffer descriptor and usage flag bindings (T-WGPU-P7.6.3)
//! - [`py_resource`] - Resource handle and pool bindings (T-WGPU-P7.6.4)
//! - [`py_render_pass`] - Render pass builder bindings (T-WGPU-P7.6.5)
//! - [`py_compute_pass`] - Compute pass construction bindings (T-WGPU-P7.6.6)
//! - [`py_descriptor_cache`] - Descriptor caching bindings (T-WGPU-P7.6.7)
//! - [`py_command_batch`] - Command batching bindings (T-WGPU-P7.6.8)
//! - [`py_error`] - Error handling and propagation bindings (T-WGPU-P7.6.9)
//! - [`py_example`] - Python API examples and validation helpers (T-WGPU-P7.6.10)
//!
//! # Feature Gate
//!
//! All PyO3 bindings are gated behind the `pyo3` feature flag:
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
//!     BufferDescriptor, BufferUsage, BufferBindingType, BufferSize,
//!     PyResourceHandle, PyResourceType, PyResourcePool, PyResourceValidation,
//!     RenderPassBuilder, LoadOp, StoreOp, TextureView,
//!     ComputePassBuilder, DispatchDescriptor, ComputePipelineDescriptor,
//!     PyDescriptorCache, PyCacheKey, PyCacheStats,
//!     CommandEncoder, CommandBuffer, CommandBatcher, Command, IndexFormat,
//!     GpuError, ErrorCategory, GpuResult, ErrorHandler, ValidationReport
//! )
//!
//! # Create a buffer descriptor
//! desc = BufferDescriptor.uniform(256).with_label("camera_uniforms")
//!
//! # Combine usage flags
//! usage = BufferUsage.storage() | BufferUsage.copy_src()
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
//!
//! # Create a render pass
//! render_pass = (
//!     RenderPassBuilder()
//!     .label("main_pass")
//!     .color(TextureView(0), clear_color=[0.0, 0.0, 0.0, 1.0])
//!     .depth(TextureView(1), clear_value=1.0)
//!     .build()
//! )
//!
//! # Create a compute pass descriptor with builder pattern
//! compute_pass = (
//!     ComputePassBuilder()
//!     .label("particle_simulation")
//!     .timestamp_begin(0)
//!     .timestamp_end(1)
//!     .build()
//! )
//!
//! # Create dispatch descriptor
//! dispatch = DispatchDescriptor.direct(64, 64, 1)
//!
//! # Use descriptor cache to reduce allocations
//! cache = PyDescriptorCache(capacity=1000)
//! handle = cache.get_or_create_buffer(desc)
//! print(f"Cache hit rate: {cache.hit_rate():.2%}")
//!
//! # Record commands with batching
//! encoder = CommandEncoder()
//! encoder.begin_render_pass(RenderPassDescriptor(label="main"))
//! encoder.set_pipeline(pipeline_handle)
//! encoder.draw(vertex_count=36)
//! encoder.end_render_pass()
//! buffer = encoder.finish()
//!
//! # Automatic command batching
//! batcher = CommandBatcher(batch_size=64, auto_flush=True)
//! batcher.add_command(Command.draw(36, 1))
//! batches = batcher.flush_all()
//! ```

#[cfg(feature = "pyo3")]
pub mod py_buffer;

#[cfg(feature = "pyo3")]
pub mod py_resource;

#[cfg(feature = "pyo3")]
pub mod py_render_pass;

#[cfg(feature = "pyo3")]
pub mod py_compute_pass;

#[cfg(feature = "pyo3")]
pub mod py_descriptor_cache;

#[cfg(feature = "pyo3")]
pub mod py_command_batch;

#[cfg(feature = "pyo3")]
pub mod py_error;

#[cfg(feature = "pyo3")]
pub mod py_example;

#[cfg(feature = "pyo3")]
pub use py_buffer::*;

#[cfg(feature = "pyo3")]
pub use py_resource::*;

#[cfg(feature = "pyo3")]
pub use py_render_pass::*;

#[cfg(feature = "pyo3")]
pub use py_compute_pass::*;

#[cfg(feature = "pyo3")]
pub use py_descriptor_cache::*;

#[cfg(feature = "pyo3")]
pub use py_command_batch::*;

#[cfg(feature = "pyo3")]
pub use py_error::*;

#[cfg(feature = "pyo3")]
pub use py_example::*;

// Module registration for PyO3
#[cfg(feature = "pyo3")]
pub fn register_module(
    py: pyo3::Python<'_>,
    parent: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    let m = pyo3::types::PyModule::new(py, "bindings")?;

    // Register buffer types (T-WGPU-P7.6.3)
    py_buffer::register_module(py, &m)?;

    // Register resource types (T-WGPU-P7.6.4)
    py_resource::register_module(py, &m)?;

    // Register render pass types (T-WGPU-P7.6.5)
    py_render_pass::register_module(py, &m)?;

    // Register compute pass types (T-WGPU-P7.6.6)
    py_compute_pass::register_module(py, &m)?;

    // Register descriptor cache types (T-WGPU-P7.6.7)
    py_descriptor_cache::register_module(py, &m)?;

    // Register command batch types (T-WGPU-P7.6.8)
    py_command_batch::register_module(py, &m)?;

    // Register error handling types (T-WGPU-P7.6.9)
    py_error::register_module(py, &m)?;

    // Register example types (T-WGPU-P7.6.10)
    py_example::register_module(py, &m)?;

    parent.add_submodule(&m)?;
    Ok(())
}
