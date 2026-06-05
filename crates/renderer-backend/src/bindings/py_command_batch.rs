//! Python bindings for command batching (T-WGPU-P7.6.8).
//!
//! This module provides Python-accessible types for batching GPU commands
//! to reduce submission overhead. Commands are accumulated and submitted
//! in batches for improved performance.
//!
//! # Types
//!
//! - [`PyCommand`] - Enum representing GPU commands
//! - [`PyCommandEncoder`] - Command recording interface
//! - [`PyCommandBuffer`] - Immutable buffer of recorded commands
//! - [`PyCommandBatcher`] - Automatic batching with configurable flush
//! - [`PyIndexFormat`] - Index buffer element format
//! - [`PyRenderPassDescriptor`] - Render pass configuration (from py_render_pass)
//! - [`PyComputePassDescriptor`] - Compute pass configuration (from py_compute_pass)
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     CommandEncoder, CommandBatcher, Command, IndexFormat
//! )
//!
//! # Manual command encoding
//! encoder = CommandEncoder()
//! encoder.set_pipeline(pipeline_handle)
//! encoder.set_bind_group(0, bind_group_handle)
//! encoder.draw(vertex_count=36, instance_count=1)
//! buffer = encoder.finish()
//!
//! # Automatic batching
//! batcher = CommandBatcher(batch_size=64, auto_flush=True)
//! batcher.add_command(Command.draw(36, 1))
//! batcher.add_command(Command.draw(24, 100))
//! # Commands are auto-flushed when batch_size is reached
//! remaining = batcher.flush()  # Flush any remaining commands
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

use super::py_resource::PyResourceHandle;

// ============================================================================
// PyIndexFormat
// ============================================================================

/// Index buffer element format.
///
/// Specifies whether indices are 16-bit or 32-bit unsigned integers.
#[pyclass(name = "IndexFormat", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PyIndexFormat {
    /// 16-bit unsigned integer indices (0-65535).
    #[default]
    Uint16 = 0,
    /// 32-bit unsigned integer indices (0-4294967295).
    Uint32 = 1,
}

#[pymethods]
impl PyIndexFormat {
    /// Returns the canonical name of this index format.
    pub fn name(&self) -> &str {
        match self {
            Self::Uint16 => "Uint16",
            Self::Uint32 => "Uint32",
        }
    }

    /// Returns the byte size of each index element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }

    fn __repr__(&self) -> String {
        format!("IndexFormat.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

// ============================================================================
// PyCommand
// ============================================================================

/// GPU command enumeration representing all recordable operations.
///
/// Commands can be created using static factory methods and accumulated
/// in command buffers for batch submission.
///
/// # Command Categories
///
/// - **Draw commands**: `draw`, `draw_indexed`
/// - **Compute commands**: `dispatch`, `dispatch_indirect`
/// - **State commands**: `set_pipeline`, `set_bind_group`, `set_vertex_buffer`, `set_index_buffer`
/// - **Transfer commands**: `copy`
#[pyclass(name = "Command")]
#[derive(Clone, Debug)]
pub enum PyCommand {
    /// Draw primitives.
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// Draw indexed primitives.
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    /// Dispatch compute work groups.
    Dispatch { x: u32, y: u32, z: u32 },
    /// Dispatch compute work groups using indirect buffer.
    DispatchIndirect {
        buffer: PyResourceHandle,
        offset: u64,
    },
    /// Set the current render or compute pipeline.
    SetPipeline { pipeline_handle: PyResourceHandle },
    /// Set a bind group at the specified index.
    SetBindGroup {
        index: u32,
        bind_group: PyResourceHandle,
        dynamic_offsets: Vec<u32>,
    },
    /// Set a vertex buffer at the specified slot.
    SetVertexBuffer {
        slot: u32,
        buffer: PyResourceHandle,
        offset: u64,
        size: Option<u64>,
    },
    /// Set the index buffer.
    SetIndexBuffer {
        buffer: PyResourceHandle,
        offset: u64,
        size: Option<u64>,
        format: PyIndexFormat,
    },
    /// Copy data between buffers.
    Copy {
        src: PyResourceHandle,
        dst: PyResourceHandle,
        size: u64,
    },
}

#[pymethods]
impl PyCommand {
    // -- Draw commands --

    /// Create a Draw command.
    ///
    /// # Arguments
    /// * `vertex_count` - Number of vertices to draw
    /// * `instance_count` - Number of instances to draw (default: 1)
    /// * `first_vertex` - First vertex to draw (default: 0)
    /// * `first_instance` - First instance to draw (default: 0)
    #[staticmethod]
    #[pyo3(signature = (vertex_count, instance_count=1, first_vertex=0, first_instance=0))]
    pub fn draw(
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Self {
        Self::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }

    /// Create a DrawIndexed command.
    ///
    /// # Arguments
    /// * `index_count` - Number of indices to draw
    /// * `instance_count` - Number of instances to draw (default: 1)
    /// * `first_index` - First index to draw (default: 0)
    /// * `base_vertex` - Base vertex offset (default: 0)
    /// * `first_instance` - First instance to draw (default: 0)
    #[staticmethod]
    #[pyo3(signature = (index_count, instance_count=1, first_index=0, base_vertex=0, first_instance=0))]
    pub fn draw_indexed(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    // -- Compute commands --

    /// Create a Dispatch command for compute work.
    ///
    /// # Arguments
    /// * `x` - Number of work groups in X dimension
    /// * `y` - Number of work groups in Y dimension (default: 1)
    /// * `z` - Number of work groups in Z dimension (default: 1)
    #[staticmethod]
    #[pyo3(signature = (x, y=1, z=1))]
    pub fn dispatch(x: u32, y: u32, z: u32) -> Self {
        Self::Dispatch { x, y, z }
    }

    /// Create a DispatchIndirect command.
    ///
    /// # Arguments
    /// * `buffer` - Resource handle for the indirect buffer
    /// * `offset` - Byte offset into the buffer (default: 0)
    #[staticmethod]
    #[pyo3(signature = (buffer, offset=0))]
    pub fn dispatch_indirect(buffer: PyResourceHandle, offset: u64) -> Self {
        Self::DispatchIndirect { buffer, offset }
    }

    // -- State commands --

    /// Create a SetPipeline command.
    ///
    /// # Arguments
    /// * `pipeline_handle` - Resource handle for the pipeline
    #[staticmethod]
    pub fn set_pipeline(pipeline_handle: PyResourceHandle) -> Self {
        Self::SetPipeline { pipeline_handle }
    }

    /// Create a SetBindGroup command.
    ///
    /// # Arguments
    /// * `index` - Bind group slot index
    /// * `bind_group` - Resource handle for the bind group
    /// * `dynamic_offsets` - Dynamic buffer offsets (default: empty)
    #[staticmethod]
    #[pyo3(signature = (index, bind_group, dynamic_offsets=None))]
    pub fn set_bind_group(
        index: u32,
        bind_group: PyResourceHandle,
        dynamic_offsets: Option<Vec<u32>>,
    ) -> Self {
        Self::SetBindGroup {
            index,
            bind_group,
            dynamic_offsets: dynamic_offsets.unwrap_or_default(),
        }
    }

    /// Create a SetVertexBuffer command.
    ///
    /// # Arguments
    /// * `slot` - Vertex buffer slot
    /// * `buffer` - Resource handle for the buffer
    /// * `offset` - Byte offset into the buffer (default: 0)
    /// * `size` - Size in bytes (None = entire buffer)
    #[staticmethod]
    #[pyo3(signature = (slot, buffer, offset=0, size=None))]
    pub fn set_vertex_buffer(
        slot: u32,
        buffer: PyResourceHandle,
        offset: u64,
        size: Option<u64>,
    ) -> Self {
        Self::SetVertexBuffer {
            slot,
            buffer,
            offset,
            size,
        }
    }

    /// Create a SetIndexBuffer command.
    ///
    /// # Arguments
    /// * `buffer` - Resource handle for the index buffer
    /// * `format` - Index element format (Uint16 or Uint32)
    /// * `offset` - Byte offset into the buffer (default: 0)
    /// * `size` - Size in bytes (None = entire buffer)
    #[staticmethod]
    #[pyo3(signature = (buffer, format, offset=0, size=None))]
    pub fn set_index_buffer(
        buffer: PyResourceHandle,
        format: PyIndexFormat,
        offset: u64,
        size: Option<u64>,
    ) -> Self {
        Self::SetIndexBuffer {
            buffer,
            offset,
            size,
            format,
        }
    }

    // -- Transfer commands --

    /// Create a Copy command.
    ///
    /// # Arguments
    /// * `src` - Source buffer handle
    /// * `dst` - Destination buffer handle
    /// * `size` - Number of bytes to copy
    #[staticmethod]
    pub fn copy(src: PyResourceHandle, dst: PyResourceHandle, size: u64) -> Self {
        Self::Copy { src, dst, size }
    }

    // -- Introspection --

    /// Returns the command type name.
    pub fn command_type(&self) -> &str {
        match self {
            Self::Draw { .. } => "Draw",
            Self::DrawIndexed { .. } => "DrawIndexed",
            Self::Dispatch { .. } => "Dispatch",
            Self::DispatchIndirect { .. } => "DispatchIndirect",
            Self::SetPipeline { .. } => "SetPipeline",
            Self::SetBindGroup { .. } => "SetBindGroup",
            Self::SetVertexBuffer { .. } => "SetVertexBuffer",
            Self::SetIndexBuffer { .. } => "SetIndexBuffer",
            Self::Copy { .. } => "Copy",
        }
    }

    /// Returns true if this is a draw command.
    pub fn is_draw(&self) -> bool {
        matches!(self, Self::Draw { .. } | Self::DrawIndexed { .. })
    }

    /// Returns true if this is a compute command.
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::Dispatch { .. } | Self::DispatchIndirect { .. })
    }

    /// Returns true if this is a state-setting command.
    pub fn is_state(&self) -> bool {
        matches!(
            self,
            Self::SetPipeline { .. }
                | Self::SetBindGroup { .. }
                | Self::SetVertexBuffer { .. }
                | Self::SetIndexBuffer { .. }
        )
    }

    /// Returns true if this is a transfer command.
    pub fn is_transfer(&self) -> bool {
        matches!(self, Self::Copy { .. })
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => format!(
                "Command.Draw(vertices={}, instances={}, first_vertex={}, first_instance={})",
                vertex_count, instance_count, first_vertex, first_instance
            ),
            Self::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => format!(
                "Command.DrawIndexed(indices={}, instances={}, first_index={}, base_vertex={}, first_instance={})",
                index_count, instance_count, first_index, base_vertex, first_instance
            ),
            Self::Dispatch { x, y, z } => {
                format!("Command.Dispatch({}, {}, {})", x, y, z)
            }
            Self::DispatchIndirect { buffer, offset } => {
                format!(
                    "Command.DispatchIndirect(buffer={}, offset={})",
                    buffer.id(),
                    offset
                )
            }
            Self::SetPipeline { pipeline_handle } => {
                format!("Command.SetPipeline(handle={})", pipeline_handle.id())
            }
            Self::SetBindGroup {
                index,
                bind_group,
                dynamic_offsets,
            } => {
                if dynamic_offsets.is_empty() {
                    format!(
                        "Command.SetBindGroup(index={}, bind_group={})",
                        index,
                        bind_group.id()
                    )
                } else {
                    format!(
                        "Command.SetBindGroup(index={}, bind_group={}, offsets={:?})",
                        index,
                        bind_group.id(),
                        dynamic_offsets
                    )
                }
            }
            Self::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                match size {
                    Some(s) => format!(
                        "Command.SetVertexBuffer(slot={}, buffer={}, offset={}, size={})",
                        slot,
                        buffer.id(),
                        offset,
                        s
                    ),
                    None => format!(
                        "Command.SetVertexBuffer(slot={}, buffer={}, offset={})",
                        slot,
                        buffer.id(),
                        offset
                    ),
                }
            }
            Self::SetIndexBuffer {
                buffer,
                offset,
                size,
                format,
            } => {
                match size {
                    Some(s) => format!(
                        "Command.SetIndexBuffer(buffer={}, format={:?}, offset={}, size={})",
                        buffer.id(),
                        format,
                        offset,
                        s
                    ),
                    None => format!(
                        "Command.SetIndexBuffer(buffer={}, format={:?}, offset={})",
                        buffer.id(),
                        format,
                        offset
                    ),
                }
            }
            Self::Copy { src, dst, size } => {
                format!(
                    "Command.Copy(src={}, dst={}, size={})",
                    src.id(),
                    dst.id(),
                    size
                )
            }
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyRenderPassDescriptor (minimal for command encoder)
// ============================================================================

/// Minimal render pass descriptor for command encoding.
///
/// Used by CommandEncoder to track render pass state.
#[pyclass(name = "RenderPassDescriptor")]
#[derive(Clone, Debug, Default)]
pub struct PyRenderPassDescriptor {
    label: Option<String>,
}

#[pymethods]
impl PyRenderPassDescriptor {
    /// Create a new render pass descriptor.
    #[new]
    #[pyo3(signature = (label=None))]
    pub fn new(label: Option<String>) -> Self {
        Self { label }
    }

    /// Returns the label.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn __repr__(&self) -> String {
        match &self.label {
            Some(l) => format!("RenderPassDescriptor(label=\"{}\")", l),
            None => "RenderPassDescriptor()".to_string(),
        }
    }
}

// ============================================================================
// PyComputePassDescriptor (minimal for command encoder)
// ============================================================================

/// Minimal compute pass descriptor for command encoding.
///
/// Used by CommandEncoder to track compute pass state.
#[pyclass(name = "ComputePassDescriptor")]
#[derive(Clone, Debug, Default)]
pub struct PyComputePassDescriptor {
    label: Option<String>,
}

#[pymethods]
impl PyComputePassDescriptor {
    /// Create a new compute pass descriptor.
    #[new]
    #[pyo3(signature = (label=None))]
    pub fn new(label: Option<String>) -> Self {
        Self { label }
    }

    /// Returns the label.
    #[getter]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    fn __repr__(&self) -> String {
        match &self.label {
            Some(l) => format!("ComputePassDescriptor(label=\"{}\")", l),
            None => "ComputePassDescriptor()".to_string(),
        }
    }
}

// ============================================================================
// PassState
// ============================================================================

/// Internal state tracking for render/compute passes.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PassState {
    None,
    RenderPass { label: Option<String> },
    ComputePass { label: Option<String> },
}

impl Default for PassState {
    fn default() -> Self {
        Self::None
    }
}

// ============================================================================
// PyCommandEncoder
// ============================================================================

/// Command encoder for recording GPU commands.
///
/// The encoder accumulates commands and produces an immutable CommandBuffer
/// when `finish()` is called. Commands can be recorded inside render or
/// compute passes.
///
/// # Example
///
/// ```python
/// encoder = CommandEncoder()
///
/// # Begin a render pass
/// encoder.begin_render_pass(RenderPassDescriptor(label="main"))
/// encoder.set_pipeline(pipeline_handle)
/// encoder.draw(vertex_count=36)
/// encoder.end_render_pass()
///
/// # Finish encoding
/// buffer = encoder.finish()
/// print(f"Recorded {buffer.command_count()} commands")
/// ```
#[pyclass(name = "CommandEncoder")]
#[derive(Clone, Debug, Default)]
pub struct PyCommandEncoder {
    commands: Vec<PyCommand>,
    pass_state: PassState,
}

#[pymethods]
impl PyCommandEncoder {
    /// Create a new command encoder.
    #[new]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            pass_state: PassState::None,
        }
    }

    // -- Pass management --

    /// Begin a render pass.
    ///
    /// # Arguments
    /// * `desc` - Render pass descriptor
    ///
    /// # Errors
    /// Returns an error if a pass is already active.
    pub fn begin_render_pass(&mut self, desc: &PyRenderPassDescriptor) -> PyResult<()> {
        if self.pass_state != PassState::None {
            return Err(PyValueError::new_err(
                "Cannot begin render pass: another pass is already active",
            ));
        }
        self.pass_state = PassState::RenderPass {
            label: desc.label.clone(),
        };
        Ok(())
    }

    /// End the current render pass.
    ///
    /// # Errors
    /// Returns an error if no render pass is active.
    pub fn end_render_pass(&mut self) -> PyResult<()> {
        match &self.pass_state {
            PassState::RenderPass { .. } => {
                self.pass_state = PassState::None;
                Ok(())
            }
            PassState::ComputePass { .. } => Err(PyValueError::new_err(
                "Cannot end render pass: compute pass is active",
            )),
            PassState::None => Err(PyValueError::new_err(
                "Cannot end render pass: no pass is active",
            )),
        }
    }

    /// Begin a compute pass.
    ///
    /// # Arguments
    /// * `desc` - Compute pass descriptor
    ///
    /// # Errors
    /// Returns an error if a pass is already active.
    pub fn begin_compute_pass(&mut self, desc: &PyComputePassDescriptor) -> PyResult<()> {
        if self.pass_state != PassState::None {
            return Err(PyValueError::new_err(
                "Cannot begin compute pass: another pass is already active",
            ));
        }
        self.pass_state = PassState::ComputePass {
            label: desc.label.clone(),
        };
        Ok(())
    }

    /// End the current compute pass.
    ///
    /// # Errors
    /// Returns an error if no compute pass is active.
    pub fn end_compute_pass(&mut self) -> PyResult<()> {
        match &self.pass_state {
            PassState::ComputePass { .. } => {
                self.pass_state = PassState::None;
                Ok(())
            }
            PassState::RenderPass { .. } => Err(PyValueError::new_err(
                "Cannot end compute pass: render pass is active",
            )),
            PassState::None => Err(PyValueError::new_err(
                "Cannot end compute pass: no pass is active",
            )),
        }
    }

    /// Returns true if a render pass is currently active.
    pub fn in_render_pass(&self) -> bool {
        matches!(self.pass_state, PassState::RenderPass { .. })
    }

    /// Returns true if a compute pass is currently active.
    pub fn in_compute_pass(&self) -> bool {
        matches!(self.pass_state, PassState::ComputePass { .. })
    }

    /// Returns true if any pass is currently active.
    pub fn in_pass(&self) -> bool {
        self.pass_state != PassState::None
    }

    // -- Draw commands (require render pass) --

    /// Record a draw command.
    ///
    /// # Arguments
    /// * `vertex_count` - Number of vertices to draw
    /// * `instance_count` - Number of instances (default: 1)
    /// * `first_vertex` - First vertex (default: 0)
    /// * `first_instance` - First instance (default: 0)
    ///
    /// # Errors
    /// Returns an error if no render pass is active.
    #[pyo3(signature = (vertex_count, instance_count=1, first_vertex=0, first_instance=0))]
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> PyResult<()> {
        self.require_render_pass("draw")?;
        self.commands.push(PyCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
        Ok(())
    }

    /// Record a draw indexed command.
    ///
    /// # Arguments
    /// * `index_count` - Number of indices to draw
    /// * `instance_count` - Number of instances (default: 1)
    /// * `first_index` - First index (default: 0)
    /// * `base_vertex` - Base vertex offset (default: 0)
    /// * `first_instance` - First instance (default: 0)
    ///
    /// # Errors
    /// Returns an error if no render pass is active.
    #[pyo3(signature = (index_count, instance_count=1, first_index=0, base_vertex=0, first_instance=0))]
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> PyResult<()> {
        self.require_render_pass("draw_indexed")?;
        self.commands.push(PyCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
        Ok(())
    }

    // -- Compute commands (require compute pass) --

    /// Record a dispatch command.
    ///
    /// # Arguments
    /// * `x` - Work groups in X dimension
    /// * `y` - Work groups in Y dimension (default: 1)
    /// * `z` - Work groups in Z dimension (default: 1)
    ///
    /// # Errors
    /// Returns an error if no compute pass is active.
    #[pyo3(signature = (x, y=1, z=1))]
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> PyResult<()> {
        self.require_compute_pass("dispatch")?;
        self.commands.push(PyCommand::Dispatch { x, y, z });
        Ok(())
    }

    /// Record a dispatch indirect command.
    ///
    /// # Arguments
    /// * `buffer` - Indirect buffer handle
    /// * `offset` - Byte offset (default: 0)
    ///
    /// # Errors
    /// Returns an error if no compute pass is active.
    #[pyo3(signature = (buffer, offset=0))]
    pub fn dispatch_indirect(&mut self, buffer: PyResourceHandle, offset: u64) -> PyResult<()> {
        self.require_compute_pass("dispatch_indirect")?;
        self.commands
            .push(PyCommand::DispatchIndirect { buffer, offset });
        Ok(())
    }

    // -- State commands (require any pass) --

    /// Set the current pipeline.
    ///
    /// # Arguments
    /// * `pipeline_handle` - Pipeline resource handle
    ///
    /// # Errors
    /// Returns an error if no pass is active.
    pub fn set_pipeline(&mut self, pipeline_handle: PyResourceHandle) -> PyResult<()> {
        self.require_any_pass("set_pipeline")?;
        self.commands
            .push(PyCommand::SetPipeline { pipeline_handle });
        Ok(())
    }

    /// Set a bind group.
    ///
    /// # Arguments
    /// * `index` - Bind group slot
    /// * `bind_group` - Bind group handle
    /// * `dynamic_offsets` - Dynamic buffer offsets (default: empty)
    ///
    /// # Errors
    /// Returns an error if no pass is active.
    #[pyo3(signature = (index, bind_group, dynamic_offsets=None))]
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: PyResourceHandle,
        dynamic_offsets: Option<Vec<u32>>,
    ) -> PyResult<()> {
        self.require_any_pass("set_bind_group")?;
        self.commands.push(PyCommand::SetBindGroup {
            index,
            bind_group,
            dynamic_offsets: dynamic_offsets.unwrap_or_default(),
        });
        Ok(())
    }

    /// Set a vertex buffer.
    ///
    /// # Arguments
    /// * `slot` - Vertex buffer slot
    /// * `buffer` - Buffer handle
    /// * `offset` - Byte offset (default: 0)
    /// * `size` - Size in bytes (None = entire buffer)
    ///
    /// # Errors
    /// Returns an error if no render pass is active.
    #[pyo3(signature = (slot, buffer, offset=0, size=None))]
    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: PyResourceHandle,
        offset: u64,
        size: Option<u64>,
    ) -> PyResult<()> {
        self.require_render_pass("set_vertex_buffer")?;
        self.commands.push(PyCommand::SetVertexBuffer {
            slot,
            buffer,
            offset,
            size,
        });
        Ok(())
    }

    /// Set the index buffer.
    ///
    /// # Arguments
    /// * `buffer` - Buffer handle
    /// * `format` - Index format (Uint16 or Uint32)
    /// * `offset` - Byte offset (default: 0)
    /// * `size` - Size in bytes (None = entire buffer)
    ///
    /// # Errors
    /// Returns an error if no render pass is active.
    #[pyo3(signature = (buffer, format, offset=0, size=None))]
    pub fn set_index_buffer(
        &mut self,
        buffer: PyResourceHandle,
        format: PyIndexFormat,
        offset: u64,
        size: Option<u64>,
    ) -> PyResult<()> {
        self.require_render_pass("set_index_buffer")?;
        self.commands.push(PyCommand::SetIndexBuffer {
            buffer,
            offset,
            size,
            format,
        });
        Ok(())
    }

    // -- Transfer commands (outside of pass) --

    /// Record a buffer copy command.
    ///
    /// # Arguments
    /// * `src` - Source buffer handle
    /// * `dst` - Destination buffer handle
    /// * `size` - Bytes to copy
    ///
    /// # Errors
    /// Returns an error if a pass is currently active.
    pub fn copy(&mut self, src: PyResourceHandle, dst: PyResourceHandle, size: u64) -> PyResult<()> {
        if self.pass_state != PassState::None {
            return Err(PyValueError::new_err(
                "Copy commands must be recorded outside of render/compute passes",
            ));
        }
        self.commands.push(PyCommand::Copy { src, dst, size });
        Ok(())
    }

    // -- Finalization --

    /// Finish encoding and produce an immutable command buffer.
    ///
    /// After calling `finish()`, the encoder is reset and can be reused.
    ///
    /// # Errors
    /// Returns an error if a pass is still active.
    pub fn finish(&mut self) -> PyResult<PyCommandBuffer> {
        if self.pass_state != PassState::None {
            return Err(PyValueError::new_err(
                "Cannot finish: a pass is still active. Call end_render_pass() or end_compute_pass() first.",
            ));
        }

        let commands = std::mem::take(&mut self.commands);
        Ok(PyCommandBuffer { commands })
    }

    /// Returns the number of commands recorded so far.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Clear all recorded commands and reset state.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.pass_state = PassState::None;
    }

    fn __repr__(&self) -> String {
        let state = match &self.pass_state {
            PassState::None => "idle",
            PassState::RenderPass { label } => {
                if let Some(l) = label {
                    return format!(
                        "CommandEncoder(commands={}, render_pass=\"{}\")",
                        self.commands.len(),
                        l
                    );
                }
                "render_pass"
            }
            PassState::ComputePass { label } => {
                if let Some(l) = label {
                    return format!(
                        "CommandEncoder(commands={}, compute_pass=\"{}\")",
                        self.commands.len(),
                        l
                    );
                }
                "compute_pass"
            }
        };
        format!(
            "CommandEncoder(commands={}, state={})",
            self.commands.len(),
            state
        )
    }
}

impl PyCommandEncoder {
    /// Internal helper: require render pass to be active.
    fn require_render_pass(&self, operation: &str) -> PyResult<()> {
        match &self.pass_state {
            PassState::RenderPass { .. } => Ok(()),
            PassState::ComputePass { .. } => Err(PyValueError::new_err(format!(
                "{} requires a render pass, but a compute pass is active",
                operation
            ))),
            PassState::None => Err(PyValueError::new_err(format!(
                "{} requires a render pass, but no pass is active",
                operation
            ))),
        }
    }

    /// Internal helper: require compute pass to be active.
    fn require_compute_pass(&self, operation: &str) -> PyResult<()> {
        match &self.pass_state {
            PassState::ComputePass { .. } => Ok(()),
            PassState::RenderPass { .. } => Err(PyValueError::new_err(format!(
                "{} requires a compute pass, but a render pass is active",
                operation
            ))),
            PassState::None => Err(PyValueError::new_err(format!(
                "{} requires a compute pass, but no pass is active",
                operation
            ))),
        }
    }

    /// Internal helper: require any pass to be active.
    fn require_any_pass(&self, operation: &str) -> PyResult<()> {
        if self.pass_state == PassState::None {
            return Err(PyValueError::new_err(format!(
                "{} requires an active render or compute pass",
                operation
            )));
        }
        Ok(())
    }
}

// ============================================================================
// PyCommandBuffer
// ============================================================================

/// Immutable buffer of recorded GPU commands.
///
/// A command buffer is produced by calling `finish()` on a CommandEncoder.
/// Buffers can be merged together and inspected.
///
/// # Example
///
/// ```python
/// # Create multiple command buffers
/// buffer1 = encoder1.finish()
/// buffer2 = encoder2.finish()
///
/// # Merge them
/// buffer1.merge(buffer2)
/// print(f"Total commands: {buffer1.command_count()}")
/// ```
#[pyclass(name = "CommandBuffer")]
#[derive(Clone, Debug, Default)]
pub struct PyCommandBuffer {
    commands: Vec<PyCommand>,
}

#[pymethods]
impl PyCommandBuffer {
    /// Create an empty command buffer.
    #[new]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Returns true if the buffer contains no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Returns the number of commands in the buffer.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Merge another command buffer into this one.
    ///
    /// The other buffer's commands are appended to this buffer.
    /// The other buffer is consumed (emptied) by this operation.
    pub fn merge(&mut self, other: &mut PyCommandBuffer) {
        self.commands.append(&mut other.commands);
    }

    /// Get a command by index.
    ///
    /// # Arguments
    /// * `index` - Command index
    ///
    /// # Returns
    /// The command at the given index, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<PyCommand> {
        self.commands.get(index).cloned()
    }

    /// Returns a list of all commands.
    pub fn commands(&self) -> Vec<PyCommand> {
        self.commands.clone()
    }

    /// Count commands by type.
    ///
    /// Returns a dict with command type names as keys and counts as values.
    pub fn count_by_type(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for cmd in &self.commands {
            *counts.entry(cmd.command_type().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the number of draw commands.
    pub fn draw_count(&self) -> usize {
        self.commands.iter().filter(|c| c.is_draw()).count()
    }

    /// Returns the number of compute dispatch commands.
    pub fn dispatch_count(&self) -> usize {
        self.commands.iter().filter(|c| c.is_compute()).count()
    }

    fn __repr__(&self) -> String {
        format!("CommandBuffer(commands={})", self.commands.len())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.commands.len()
    }

    fn __bool__(&self) -> bool {
        !self.commands.is_empty()
    }
}

// ============================================================================
// PyCommandBatcher
// ============================================================================

/// Automatic command batcher for reducing submission overhead.
///
/// The batcher accumulates commands and automatically flushes when the
/// batch size is reached (if auto_flush is enabled). This reduces the
/// overhead of many small command submissions.
///
/// # Example
///
/// ```python
/// batcher = CommandBatcher(batch_size=64, auto_flush=True)
///
/// # Add commands - auto-flushes every 64 commands
/// for i in range(200):
///     batcher.add_command(Command.draw(36, 1))
///
/// # Manually flush remaining commands
/// remaining = batcher.flush()
/// if remaining:
///     submit(remaining)
/// ```
#[pyclass(name = "CommandBatcher")]
#[derive(Clone, Debug)]
pub struct PyCommandBatcher {
    batch_size: usize,
    pending: Vec<PyCommand>,
    auto_flush: bool,
    flushed_batches: Vec<PyCommandBuffer>,
}

impl Default for PyCommandBatcher {
    fn default() -> Self {
        Self {
            batch_size: 64,
            pending: Vec::new(),
            auto_flush: true,
            flushed_batches: Vec::new(),
        }
    }
}

#[pymethods]
impl PyCommandBatcher {
    /// Create a new command batcher.
    ///
    /// # Arguments
    /// * `batch_size` - Maximum commands per batch (default: 64)
    /// * `auto_flush` - Automatically flush when batch_size is reached (default: true)
    #[new]
    #[pyo3(signature = (batch_size=64, auto_flush=true))]
    pub fn new(batch_size: usize, auto_flush: bool) -> Self {
        Self {
            batch_size: batch_size.max(1), // Minimum batch size of 1
            pending: Vec::with_capacity(batch_size),
            auto_flush,
            flushed_batches: Vec::new(),
        }
    }

    /// Returns the configured batch size.
    #[getter]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns whether auto-flush is enabled.
    #[getter]
    pub fn auto_flush(&self) -> bool {
        self.auto_flush
    }

    /// Set the batch size.
    #[setter]
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size.max(1);
    }

    /// Set auto-flush mode.
    #[setter]
    pub fn set_auto_flush(&mut self, enabled: bool) {
        self.auto_flush = enabled;
    }

    /// Add a command to the batch.
    ///
    /// If auto_flush is enabled and the batch reaches batch_size,
    /// the batch is automatically flushed and stored internally.
    ///
    /// # Arguments
    /// * `cmd` - Command to add
    ///
    /// # Returns
    /// True if an auto-flush occurred.
    pub fn add_command(&mut self, cmd: PyCommand) -> bool {
        self.pending.push(cmd);

        if self.auto_flush && self.pending.len() >= self.batch_size {
            self.do_flush();
            return true;
        }
        false
    }

    /// Add multiple commands at once.
    ///
    /// # Arguments
    /// * `commands` - List of commands to add
    ///
    /// # Returns
    /// Number of auto-flushes that occurred.
    pub fn add_commands(&mut self, commands: Vec<PyCommand>) -> usize {
        let mut flush_count = 0;
        for cmd in commands {
            if self.add_command(cmd) {
                flush_count += 1;
            }
        }
        flush_count
    }

    /// Flush pending commands to a command buffer.
    ///
    /// Returns None if there are no pending commands.
    pub fn flush(&mut self) -> Option<PyCommandBuffer> {
        if self.pending.is_empty() {
            return None;
        }
        Some(self.do_flush())
    }

    /// Flush and return all batches (including previously auto-flushed).
    ///
    /// Returns a list of all command buffers. The batcher is reset
    /// to its initial state.
    pub fn flush_all(&mut self) -> Vec<PyCommandBuffer> {
        // Flush any remaining pending commands
        if !self.pending.is_empty() {
            self.do_flush();
        }

        // Return all flushed batches
        std::mem::take(&mut self.flushed_batches)
    }

    /// Returns the number of pending commands.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns the number of auto-flushed batches waiting to be collected.
    pub fn batch_count(&self) -> usize {
        self.flushed_batches.len()
    }

    /// Returns true if there are no pending commands.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Clear all pending commands without flushing.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Clear all pending commands and collected batches.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.flushed_batches.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "CommandBatcher(batch_size={}, pending={}, batches={}, auto_flush={})",
            self.batch_size,
            self.pending.len(),
            self.flushed_batches.len(),
            self.auto_flush
        )
    }
}

impl PyCommandBatcher {
    /// Internal flush helper.
    fn do_flush(&mut self) -> PyCommandBuffer {
        let commands = std::mem::take(&mut self.pending);
        self.pending = Vec::with_capacity(self.batch_size);
        let buffer = PyCommandBuffer { commands };
        self.flushed_batches.push(buffer.clone());
        buffer
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register command batch types with the Python module.
pub fn register_module(
    _py: pyo3::Python<'_>,
    m: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    m.add_class::<PyIndexFormat>()?;
    m.add_class::<PyCommand>()?;
    m.add_class::<PyRenderPassDescriptor>()?;
    m.add_class::<PyComputePassDescriptor>()?;
    m.add_class::<PyCommandEncoder>()?;
    m.add_class::<PyCommandBuffer>()?;
    m.add_class::<PyCommandBatcher>()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a dummy resource handle
    fn dummy_handle(id: u64) -> PyResourceHandle {
        PyResourceHandle::new(id, super::super::py_resource::PyResourceType::Buffer)
    }

    fn dummy_pipeline() -> PyResourceHandle {
        PyResourceHandle::new(
            100,
            super::super::py_resource::PyResourceType::RenderPipeline,
        )
    }

    fn dummy_bind_group() -> PyResourceHandle {
        PyResourceHandle::new(200, super::super::py_resource::PyResourceType::BindGroup)
    }

    // -------------------------------------------------------------------------
    // PyIndexFormat tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_index_format_byte_size() {
        assert_eq!(PyIndexFormat::Uint16.byte_size(), 2);
        assert_eq!(PyIndexFormat::Uint32.byte_size(), 4);
    }

    #[test]
    fn test_index_format_name() {
        assert_eq!(PyIndexFormat::Uint16.name(), "Uint16");
        assert_eq!(PyIndexFormat::Uint32.name(), "Uint32");
    }

    // -------------------------------------------------------------------------
    // PyCommand tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_command_draw_defaults() {
        let cmd = PyCommand::draw(36, 1, 0, 0);
        assert!(cmd.is_draw());
        assert!(!cmd.is_compute());
        assert!(!cmd.is_state());
        assert!(!cmd.is_transfer());
        assert_eq!(cmd.command_type(), "Draw");
    }

    #[test]
    fn test_command_draw_indexed() {
        let cmd = PyCommand::draw_indexed(100, 10, 0, 0, 0);
        assert!(cmd.is_draw());
        assert_eq!(cmd.command_type(), "DrawIndexed");
    }

    #[test]
    fn test_command_dispatch() {
        let cmd = PyCommand::dispatch(64, 64, 1);
        assert!(cmd.is_compute());
        assert!(!cmd.is_draw());
        assert_eq!(cmd.command_type(), "Dispatch");
    }

    #[test]
    fn test_command_dispatch_indirect() {
        let cmd = PyCommand::dispatch_indirect(dummy_handle(1), 0);
        assert!(cmd.is_compute());
        assert_eq!(cmd.command_type(), "DispatchIndirect");
    }

    #[test]
    fn test_command_set_pipeline() {
        let cmd = PyCommand::set_pipeline(dummy_pipeline());
        assert!(cmd.is_state());
        assert!(!cmd.is_draw());
        assert!(!cmd.is_compute());
        assert_eq!(cmd.command_type(), "SetPipeline");
    }

    #[test]
    fn test_command_set_bind_group() {
        let cmd = PyCommand::set_bind_group(0, dummy_bind_group(), None);
        assert!(cmd.is_state());
        assert_eq!(cmd.command_type(), "SetBindGroup");
    }

    #[test]
    fn test_command_set_bind_group_with_offsets() {
        let cmd = PyCommand::set_bind_group(0, dummy_bind_group(), Some(vec![0, 256]));
        assert!(cmd.is_state());
        if let PyCommand::SetBindGroup {
            dynamic_offsets, ..
        } = cmd
        {
            assert_eq!(dynamic_offsets, vec![0, 256]);
        } else {
            panic!("Expected SetBindGroup");
        }
    }

    #[test]
    fn test_command_set_vertex_buffer() {
        let cmd = PyCommand::set_vertex_buffer(0, dummy_handle(1), 0, None);
        assert!(cmd.is_state());
        assert_eq!(cmd.command_type(), "SetVertexBuffer");
    }

    #[test]
    fn test_command_set_index_buffer() {
        let cmd = PyCommand::set_index_buffer(dummy_handle(2), PyIndexFormat::Uint16, 0, None);
        assert!(cmd.is_state());
        assert_eq!(cmd.command_type(), "SetIndexBuffer");
    }

    #[test]
    fn test_command_copy() {
        let cmd = PyCommand::copy(dummy_handle(1), dummy_handle(2), 1024);
        assert!(cmd.is_transfer());
        assert!(!cmd.is_draw());
        assert_eq!(cmd.command_type(), "Copy");
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - pass management
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_new() {
        let encoder = PyCommandEncoder::new();
        assert_eq!(encoder.command_count(), 0);
        assert!(!encoder.in_pass());
        assert!(!encoder.in_render_pass());
        assert!(!encoder.in_compute_pass());
    }

    #[test]
    fn test_encoder_begin_end_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(Some("test".to_string()));

        encoder.begin_render_pass(&desc).unwrap();
        assert!(encoder.in_render_pass());
        assert!(encoder.in_pass());
        assert!(!encoder.in_compute_pass());

        encoder.end_render_pass().unwrap();
        assert!(!encoder.in_pass());
    }

    #[test]
    fn test_encoder_begin_end_compute_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(Some("compute".to_string()));

        encoder.begin_compute_pass(&desc).unwrap();
        assert!(encoder.in_compute_pass());
        assert!(encoder.in_pass());
        assert!(!encoder.in_render_pass());

        encoder.end_compute_pass().unwrap();
        assert!(!encoder.in_pass());
    }

    #[test]
    fn test_encoder_cannot_nest_passes() {
        let mut encoder = PyCommandEncoder::new();
        let render_desc = PyRenderPassDescriptor::new(None);
        let compute_desc = PyComputePassDescriptor::new(None);

        encoder.begin_render_pass(&render_desc).unwrap();

        // Cannot start another render pass
        assert!(encoder.begin_render_pass(&render_desc).is_err());

        // Cannot start compute pass while render is active
        assert!(encoder.begin_compute_pass(&compute_desc).is_err());
    }

    #[test]
    fn test_encoder_wrong_pass_end() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();

        // Cannot end compute pass when render is active
        assert!(encoder.end_compute_pass().is_err());
    }

    #[test]
    fn test_encoder_end_without_begin() {
        let mut encoder = PyCommandEncoder::new();
        assert!(encoder.end_render_pass().is_err());
        assert!(encoder.end_compute_pass().is_err());
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - draw commands
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_draw_requires_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        assert!(encoder.draw(36, 1, 0, 0).is_err());
    }

    #[test]
    fn test_encoder_draw_in_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.draw(24, 100, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 2);
        assert_eq!(buffer.draw_count(), 2);
    }

    #[test]
    fn test_encoder_draw_indexed_in_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw_indexed(100, 1, 0, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.draw_count(), 1);
    }

    #[test]
    fn test_encoder_draw_not_in_compute_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        assert!(encoder.draw(36, 1, 0, 0).is_err());
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - compute commands
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_dispatch_requires_compute_pass() {
        let mut encoder = PyCommandEncoder::new();
        assert!(encoder.dispatch(64, 64, 1).is_err());
    }

    #[test]
    fn test_encoder_dispatch_in_compute_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        encoder.dispatch(64, 64, 1).unwrap();
        encoder.dispatch(32, 32, 32).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.dispatch_count(), 2);
    }

    #[test]
    fn test_encoder_dispatch_indirect() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        encoder.dispatch_indirect(dummy_handle(1), 0).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.dispatch_count(), 1);
    }

    #[test]
    fn test_encoder_dispatch_not_in_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        assert!(encoder.dispatch(64, 64, 1).is_err());
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - state commands
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_set_pipeline_requires_pass() {
        let mut encoder = PyCommandEncoder::new();
        assert!(encoder.set_pipeline(dummy_pipeline()).is_err());
    }

    #[test]
    fn test_encoder_set_pipeline_in_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.set_pipeline(dummy_pipeline()).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 1);
    }

    #[test]
    fn test_encoder_set_pipeline_in_compute_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        encoder.set_pipeline(dummy_pipeline()).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 1);
    }

    #[test]
    fn test_encoder_set_bind_group() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.set_bind_group(0, dummy_bind_group(), None).unwrap();
        encoder
            .set_bind_group(1, dummy_bind_group(), Some(vec![0, 256]))
            .unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 2);
    }

    #[test]
    fn test_encoder_set_vertex_buffer() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder
            .set_vertex_buffer(0, dummy_handle(1), 0, None)
            .unwrap();
        encoder
            .set_vertex_buffer(1, dummy_handle(2), 64, Some(1024))
            .unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 2);
    }

    #[test]
    fn test_encoder_set_index_buffer() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder
            .set_index_buffer(dummy_handle(1), PyIndexFormat::Uint16, 0, None)
            .unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 1);
    }

    #[test]
    fn test_encoder_vertex_buffer_requires_render_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        assert!(encoder
            .set_vertex_buffer(0, dummy_handle(1), 0, None)
            .is_err());
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - transfer commands
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_copy_outside_pass() {
        let mut encoder = PyCommandEncoder::new();
        encoder.copy(dummy_handle(1), dummy_handle(2), 1024).unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.command_count(), 1);
    }

    #[test]
    fn test_encoder_copy_not_in_pass() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        assert!(encoder.copy(dummy_handle(1), dummy_handle(2), 1024).is_err());
    }

    // -------------------------------------------------------------------------
    // PyCommandEncoder tests - finish
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_finish_empty() {
        let mut encoder = PyCommandEncoder::new();
        let buffer = encoder.finish().unwrap();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_encoder_finish_with_active_pass_fails() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        assert!(encoder.finish().is_err());
    }

    #[test]
    fn test_encoder_finish_resets_encoder() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer1 = encoder.finish().unwrap();
        assert_eq!(buffer1.command_count(), 1);

        // Encoder should be reset
        assert_eq!(encoder.command_count(), 0);

        let buffer2 = encoder.finish().unwrap();
        assert!(buffer2.is_empty());
    }

    #[test]
    fn test_encoder_clear() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();

        encoder.clear();
        assert_eq!(encoder.command_count(), 0);
        assert!(!encoder.in_pass());
    }

    // -------------------------------------------------------------------------
    // PyCommandBuffer tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_new() {
        let buffer = PyCommandBuffer::new();
        assert!(buffer.is_empty());
        assert_eq!(buffer.command_count(), 0);
    }

    #[test]
    fn test_buffer_merge() {
        let mut encoder1 = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);
        encoder1.begin_render_pass(&desc).unwrap();
        encoder1.draw(36, 1, 0, 0).unwrap();
        encoder1.end_render_pass().unwrap();
        let mut buffer1 = encoder1.finish().unwrap();

        let mut encoder2 = PyCommandEncoder::new();
        encoder2.begin_render_pass(&desc).unwrap();
        encoder2.draw(24, 1, 0, 0).unwrap();
        encoder2.draw(48, 1, 0, 0).unwrap();
        encoder2.end_render_pass().unwrap();
        let mut buffer2 = encoder2.finish().unwrap();

        buffer1.merge(&mut buffer2);
        assert_eq!(buffer1.command_count(), 3);
        assert!(buffer2.is_empty()); // Other buffer is emptied
    }

    #[test]
    fn test_buffer_get() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);
        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.draw(24, 10, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();
        let buffer = encoder.finish().unwrap();

        let cmd0 = buffer.get(0).unwrap();
        assert_eq!(cmd0.command_type(), "Draw");

        let cmd1 = buffer.get(1).unwrap();
        assert_eq!(cmd1.command_type(), "Draw");

        assert!(buffer.get(2).is_none());
    }

    #[test]
    fn test_buffer_count_by_type() {
        let mut encoder = PyCommandEncoder::new();
        let render_desc = PyRenderPassDescriptor::new(None);
        let compute_desc = PyComputePassDescriptor::new(None);

        encoder.begin_render_pass(&render_desc).unwrap();
        encoder.set_pipeline(dummy_pipeline()).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.draw(24, 1, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        encoder.begin_compute_pass(&compute_desc).unwrap();
        encoder.dispatch(64, 64, 1).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        let counts = buffer.count_by_type();

        assert_eq!(counts.get("Draw"), Some(&2));
        assert_eq!(counts.get("Dispatch"), Some(&1));
        assert_eq!(counts.get("SetPipeline"), Some(&1));
    }

    #[test]
    fn test_buffer_draw_count() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(None);

        encoder.begin_render_pass(&desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.draw_indexed(100, 1, 0, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.draw_count(), 2);
    }

    #[test]
    fn test_buffer_dispatch_count() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(None);

        encoder.begin_compute_pass(&desc).unwrap();
        encoder.dispatch(64, 1, 1).unwrap();
        encoder.dispatch_indirect(dummy_handle(1), 0).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();
        assert_eq!(buffer.dispatch_count(), 2);
    }

    // -------------------------------------------------------------------------
    // PyCommandBatcher tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batcher_new_defaults() {
        let batcher = PyCommandBatcher::new(64, true);
        assert_eq!(batcher.batch_size(), 64);
        assert!(batcher.auto_flush());
        assert!(batcher.is_empty());
        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.batch_count(), 0);
    }

    #[test]
    fn test_batcher_minimum_batch_size() {
        let batcher = PyCommandBatcher::new(0, true);
        assert_eq!(batcher.batch_size(), 1); // Minimum is 1
    }

    #[test]
    fn test_batcher_add_command() {
        let mut batcher = PyCommandBatcher::new(64, false);
        let flushed = batcher.add_command(PyCommand::draw(36, 1, 0, 0));

        assert!(!flushed);
        assert_eq!(batcher.pending_count(), 1);
    }

    #[test]
    fn test_batcher_auto_flush() {
        let mut batcher = PyCommandBatcher::new(3, true);

        batcher.add_command(PyCommand::draw(36, 1, 0, 0));
        batcher.add_command(PyCommand::draw(24, 1, 0, 0));
        assert_eq!(batcher.pending_count(), 2);
        assert_eq!(batcher.batch_count(), 0);

        // Third command triggers auto-flush
        let flushed = batcher.add_command(PyCommand::draw(12, 1, 0, 0));
        assert!(flushed);
        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.batch_count(), 1);
    }

    #[test]
    fn test_batcher_no_auto_flush() {
        let mut batcher = PyCommandBatcher::new(3, false);

        for _ in 0..10 {
            let flushed = batcher.add_command(PyCommand::draw(36, 1, 0, 0));
            assert!(!flushed);
        }

        assert_eq!(batcher.pending_count(), 10);
        assert_eq!(batcher.batch_count(), 0);
    }

    #[test]
    fn test_batcher_add_commands() {
        let mut batcher = PyCommandBatcher::new(3, true);

        let commands = vec![
            PyCommand::draw(1, 1, 0, 0),
            PyCommand::draw(2, 1, 0, 0),
            PyCommand::draw(3, 1, 0, 0),
            PyCommand::draw(4, 1, 0, 0),
            PyCommand::draw(5, 1, 0, 0),
        ];

        let flush_count = batcher.add_commands(commands);
        assert_eq!(flush_count, 1); // One auto-flush at command 3
        assert_eq!(batcher.pending_count(), 2); // Commands 4 and 5
        assert_eq!(batcher.batch_count(), 1);
    }

    #[test]
    fn test_batcher_manual_flush() {
        let mut batcher = PyCommandBatcher::new(64, false);

        batcher.add_command(PyCommand::draw(36, 1, 0, 0));
        batcher.add_command(PyCommand::draw(24, 1, 0, 0));

        let buffer = batcher.flush().unwrap();
        assert_eq!(buffer.command_count(), 2);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_flush_empty() {
        let mut batcher = PyCommandBatcher::new(64, false);
        let buffer = batcher.flush();
        assert!(buffer.is_none());
    }

    #[test]
    fn test_batcher_flush_all() {
        let mut batcher = PyCommandBatcher::new(3, true);

        // Add 7 commands: 2 auto-flushes + 1 remaining
        for i in 0..7 {
            batcher.add_command(PyCommand::draw(i as u32, 1, 0, 0));
        }

        assert_eq!(batcher.batch_count(), 2);
        assert_eq!(batcher.pending_count(), 1);

        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 3); // 2 auto-flushed + 1 from remaining

        assert!(batcher.is_empty());
        assert_eq!(batcher.batch_count(), 0);
    }

    #[test]
    fn test_batcher_clear() {
        let mut batcher = PyCommandBatcher::new(64, false);

        batcher.add_command(PyCommand::draw(36, 1, 0, 0));
        batcher.add_command(PyCommand::draw(24, 1, 0, 0));

        batcher.clear();
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_reset() {
        let mut batcher = PyCommandBatcher::new(3, true);

        // Add commands to trigger auto-flush
        for _ in 0..5 {
            batcher.add_command(PyCommand::draw(36, 1, 0, 0));
        }

        assert!(batcher.batch_count() > 0);
        assert!(batcher.pending_count() > 0);

        batcher.reset();
        assert_eq!(batcher.batch_count(), 0);
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_batcher_setters() {
        let mut batcher = PyCommandBatcher::new(64, true);

        batcher.set_batch_size(128);
        assert_eq!(batcher.batch_size(), 128);

        batcher.set_auto_flush(false);
        assert!(!batcher.auto_flush());
    }

    // -------------------------------------------------------------------------
    // Integration tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_render_pass_workflow() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyRenderPassDescriptor::new(Some("main".to_string()));

        // Begin render pass
        encoder.begin_render_pass(&desc).unwrap();

        // Set state
        encoder.set_pipeline(dummy_pipeline()).unwrap();
        encoder.set_bind_group(0, dummy_bind_group(), None).unwrap();
        encoder
            .set_vertex_buffer(0, dummy_handle(1), 0, None)
            .unwrap();
        encoder
            .set_index_buffer(dummy_handle(2), PyIndexFormat::Uint16, 0, None)
            .unwrap();

        // Draw
        encoder.draw_indexed(36, 1, 0, 0, 0).unwrap();
        encoder.draw_indexed(24, 100, 0, 0, 0).unwrap();

        // End pass
        encoder.end_render_pass().unwrap();

        // Finish
        let buffer = encoder.finish().unwrap();

        assert_eq!(buffer.command_count(), 6);
        assert_eq!(buffer.draw_count(), 2);

        let counts = buffer.count_by_type();
        assert_eq!(counts.get("SetPipeline"), Some(&1));
        assert_eq!(counts.get("SetBindGroup"), Some(&1));
        assert_eq!(counts.get("SetVertexBuffer"), Some(&1));
        assert_eq!(counts.get("SetIndexBuffer"), Some(&1));
        assert_eq!(counts.get("DrawIndexed"), Some(&2));
    }

    #[test]
    fn test_full_compute_pass_workflow() {
        let mut encoder = PyCommandEncoder::new();
        let desc = PyComputePassDescriptor::new(Some("physics".to_string()));

        encoder.begin_compute_pass(&desc).unwrap();
        encoder.set_pipeline(dummy_pipeline()).unwrap();
        encoder.set_bind_group(0, dummy_bind_group(), None).unwrap();
        encoder.dispatch(64, 64, 1).unwrap();
        encoder.end_compute_pass().unwrap();

        let buffer = encoder.finish().unwrap();

        assert_eq!(buffer.command_count(), 3);
        assert_eq!(buffer.dispatch_count(), 1);
    }

    #[test]
    fn test_multiple_passes() {
        let mut encoder = PyCommandEncoder::new();

        // First render pass
        let render_desc = PyRenderPassDescriptor::new(Some("shadow".to_string()));
        encoder.begin_render_pass(&render_desc).unwrap();
        encoder.draw(36, 1, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        // Copy between passes
        encoder.copy(dummy_handle(1), dummy_handle(2), 1024).unwrap();

        // Compute pass
        let compute_desc = PyComputePassDescriptor::new(Some("particles".to_string()));
        encoder.begin_compute_pass(&compute_desc).unwrap();
        encoder.dispatch(256, 1, 1).unwrap();
        encoder.end_compute_pass().unwrap();

        // Second render pass
        encoder.begin_render_pass(&render_desc).unwrap();
        encoder.draw(100, 1, 0, 0).unwrap();
        encoder.end_render_pass().unwrap();

        let buffer = encoder.finish().unwrap();

        assert_eq!(buffer.command_count(), 4);
        assert_eq!(buffer.draw_count(), 2);
        assert_eq!(buffer.dispatch_count(), 1);

        let counts = buffer.count_by_type();
        assert_eq!(counts.get("Copy"), Some(&1));
    }

    #[test]
    fn test_batcher_with_encoder() {
        let mut batcher = PyCommandBatcher::new(4, true);

        // Create commands via encoder
        for i in 0..10 {
            let mut encoder = PyCommandEncoder::new();
            let desc = PyRenderPassDescriptor::new(None);
            encoder.begin_render_pass(&desc).unwrap();
            encoder.draw(i as u32, 1, 0, 0).unwrap();
            encoder.end_render_pass().unwrap();

            let buffer = encoder.finish().unwrap();
            for cmd in buffer.commands() {
                batcher.add_command(cmd);
            }
        }

        let batches = batcher.flush_all();

        // 10 commands with batch_size 4 = 3 batches (4 + 4 + 2)
        assert_eq!(batches.len(), 3);

        let total_commands: usize = batches.iter().map(|b| b.command_count()).sum();
        assert_eq!(total_commands, 10);
    }
}
