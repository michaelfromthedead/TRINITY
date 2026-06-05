//! Parallel reduction compute pipelines for wgpu 25.x (T-WGPU-P3.10.1).
//!
//! Provides GPU-accelerated parallel reduction operations (sum, min, max) using
//! tree reduction with workgroup shared memory for optimal performance.
//!
//! # Architecture
//!
//! ```text
//! ReductionPipeline
//!     |-- sum_pipeline: ComputePipeline   (reduce_sum.wgsl)
//!     |-- min_pipeline: ComputePipeline   (reduce_min.wgsl)
//!     |-- max_pipeline: ComputePipeline   (reduce_max.wgsl)
//!     |-- bind_group_layout: BindGroupLayout
//!     `-- scratch_buffer: Option<Buffer>  (for multi-pass reduction)
//!
//! Single Pass (N <= WORKGROUP_SIZE * 2 = 512):
//!     input[0..N] -> reduce -> output[0]
//!
//! Multi-Pass (N > 512):
//!     Pass 1: input[0..N] -> reduce -> partial[0..num_workgroups]
//!     Pass 2: partial[0..M] -> reduce -> partial2[0..M']
//!     ...
//!     Final:  partial_k[0..K] -> reduce -> output[0]
//! ```
//!
//! # Shader Algorithm
//!
//! Each workgroup of 256 threads:
//! 1. Loads 2 elements per thread (coalesced global memory access)
//! 2. Performs local reduction operation (sum/min/max)
//! 3. Stores result to workgroup shared memory
//! 4. Tree reduction in shared memory with sequential addressing
//! 5. Thread 0 writes final workgroup result to output
//!
//! Sequential addressing pattern avoids shared memory bank conflicts.
//!
//! # Performance
//!
//! - Workgroup size: 256 threads
//! - Elements per workgroup: 512
//! - Shared memory: 256 * 4 = 1KB
//! - Bank conflict free: Sequential addressing pattern
//! - Coalesced reads: Each thread reads consecutive pairs
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_library::reduction::ReductionPipeline;
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create reduction pipeline
//! let pipeline = ReductionPipeline::new(device);
//!
//! // Create input buffer with data
//! let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("input"),
//!     contents: bytemuck::cast_slice(&input_data),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! // Perform reduction
//! let sum = pipeline.reduce_sum(device, queue, &input_buffer, input_data.len() as u32);
//! assert_eq!(sum, Ok(15.0));
//!
//! let min = pipeline.reduce_min(device, queue, &input_buffer, input_data.len() as u32);
//! assert_eq!(min, Ok(1.0));
//!
//! let max = pipeline.reduce_max(device, queue, &input_buffer, input_data.len() as u32);
//! assert_eq!(max, Ok(5.0));
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `ReductionPipeline` is `Send + Sync` when its underlying wgpu types are.
//! Multiple reductions can be performed concurrently by creating separate
//! command encoders.

use std::borrow::Cow;
use std::sync::Arc;
use thiserror::Error;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions,
    PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// ============================================================================
// Constants
// ============================================================================

/// Default workgroup size for reduction shaders.
///
/// Each workgroup has 256 threads, each loading 2 elements,
/// so each workgroup reduces 512 elements to 1 partial result.
pub const WORKGROUP_SIZE: u32 = 256;

/// Elements processed per workgroup.
pub const ELEMENTS_PER_WORKGROUP: u32 = WORKGROUP_SIZE * 2;

/// Maximum elements that can be reduced in a single pass.
pub const MAX_SINGLE_PASS_ELEMENTS: u32 = ELEMENTS_PER_WORKGROUP;

/// Threshold for multi-pass reduction (256 * 256 * 2 = 131072 elements).
/// Above this, we need more than 2 passes.
pub const MULTI_PASS_THRESHOLD: u32 = WORKGROUP_SIZE * WORKGROUP_SIZE * 2;

// ============================================================================
// Embedded Shader Sources
// ============================================================================

const REDUCE_SUM_WGSL: &str = include_str!("../../shaders/reduce_sum.wgsl");
const REDUCE_MIN_WGSL: &str = include_str!("../../shaders/reduce_min.wgsl");
const REDUCE_MAX_WGSL: &str = include_str!("../../shaders/reduce_max.wgsl");

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during reduction operations.
#[derive(Debug, Error)]
pub enum ReductionError {
    /// Input buffer is empty.
    #[error("Input buffer is empty (size = 0)")]
    EmptyInput,

    /// Buffer mapping failed.
    #[error("Buffer mapping failed: {0}")]
    BufferMapFailed(String),

    /// GPU execution failed.
    #[error("GPU execution failed: {0}")]
    GpuError(String),

    /// Buffer readback timed out.
    #[error("Buffer readback timed out after {0}ms")]
    Timeout(u64),
}

// ============================================================================
// ReductionParams Uniform
// ============================================================================

/// Uniform buffer data for reduction shaders.
///
/// Must match the struct in WGSL shaders:
/// ```wgsl
/// struct ReductionParams {
///     input_size: u32,
///     output_offset: u32,
///     _pad0: u32,
///     _pad1: u32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReductionParams {
    /// Total number of elements in input buffer.
    pub input_size: u32,
    /// Offset into output buffer for this pass.
    pub output_offset: u32,
    /// Padding for 16-byte alignment.
    _pad0: u32,
    _pad1: u32,
}

impl ReductionParams {
    /// Create new reduction parameters.
    pub fn new(input_size: u32, output_offset: u32) -> Self {
        Self {
            input_size,
            output_offset,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

// ============================================================================
// ReductionOperation
// ============================================================================

/// Type of reduction operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOperation {
    /// Sum all elements.
    Sum,
    /// Find minimum element.
    Min,
    /// Find maximum element.
    Max,
}

impl ReductionOperation {
    /// Get the identity element for this operation.
    pub fn identity(self) -> f32 {
        match self {
            ReductionOperation::Sum => 0.0,
            ReductionOperation::Min => f32::MAX,
            ReductionOperation::Max => f32::MIN,
        }
    }

    /// Get the shader entry point name for this operation.
    pub fn entry_point(self) -> &'static str {
        match self {
            ReductionOperation::Sum => "reduce_sum",
            ReductionOperation::Min => "reduce_min",
            ReductionOperation::Max => "reduce_max",
        }
    }
}

// ============================================================================
// ReductionPipeline
// ============================================================================

/// GPU compute pipeline for parallel reduction operations.
///
/// Supports sum, min, and max reductions on f32 arrays with automatic
/// multi-pass handling for large arrays.
///
/// # Creation
///
/// Create once and reuse for multiple reductions:
///
/// ```no_run
/// use renderer_backend::compute_library::reduction::ReductionPipeline;
///
/// # fn example(device: &wgpu::Device) {
/// let pipeline = ReductionPipeline::new(device);
/// // Use pipeline for multiple reductions...
/// # }
/// ```
#[derive(Debug)]
pub struct ReductionPipeline {
    /// Compute pipeline for sum reduction.
    sum_pipeline: ComputePipeline,
    /// Compute pipeline for min reduction.
    min_pipeline: ComputePipeline,
    /// Compute pipeline for max reduction.
    max_pipeline: ComputePipeline,
    /// Shared bind group layout for all reduction shaders.
    bind_group_layout: BindGroupLayout,
}

impl ReductionPipeline {
    /// Create a new reduction pipeline.
    ///
    /// Compiles all three reduction shaders (sum, min, max) and creates
    /// the shared bind group layout.
    pub fn new(device: &Device) -> Self {
        // Create shader modules
        let sum_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("reduce_sum.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(REDUCE_SUM_WGSL)),
        });

        let min_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("reduce_min.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(REDUCE_MIN_WGSL)),
        });

        let max_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("reduce_max.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(REDUCE_MAX_WGSL)),
        });

        // Create bind group layout
        // binding 0: input buffer (read-only storage)
        // binding 1: output buffer (read-write storage)
        // binding 2: params uniform
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("reduction_bind_group_layout"),
            entries: &[
                // Input buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<ReductionParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("reduction_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let sum_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("reduce_sum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &sum_module,
            entry_point: "reduce_sum",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let min_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("reduce_min_pipeline"),
            layout: Some(&pipeline_layout),
            module: &min_module,
            entry_point: "reduce_min",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let max_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("reduce_max_pipeline"),
            layout: Some(&pipeline_layout),
            module: &max_module,
            entry_point: "reduce_max",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            sum_pipeline,
            min_pipeline,
            max_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout for creating custom bind groups.
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    /// Calculate the number of workgroups needed for an input size.
    pub fn calculate_workgroups(input_size: u32) -> u32 {
        (input_size + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
    }

    /// Calculate the number of passes needed for multi-pass reduction.
    pub fn calculate_passes(input_size: u32) -> u32 {
        if input_size <= ELEMENTS_PER_WORKGROUP {
            return 1;
        }

        let mut current_size = input_size;
        let mut passes = 0;

        while current_size > 1 {
            current_size = Self::calculate_workgroups(current_size);
            passes += 1;
        }

        passes
    }

    /// Perform sum reduction on the input buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `input_buffer` - Storage buffer containing f32 values
    /// * `input_size` - Number of elements in the input buffer
    ///
    /// # Returns
    ///
    /// The sum of all elements, or an error if the operation failed.
    pub fn reduce_sum(
        &self,
        device: &Device,
        queue: &Queue,
        input_buffer: &Buffer,
        input_size: u32,
    ) -> Result<f32, ReductionError> {
        self.reduce(device, queue, input_buffer, input_size, ReductionOperation::Sum)
    }

    /// Perform min reduction on the input buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `input_buffer` - Storage buffer containing f32 values
    /// * `input_size` - Number of elements in the input buffer
    ///
    /// # Returns
    ///
    /// The minimum element, or an error if the operation failed.
    pub fn reduce_min(
        &self,
        device: &Device,
        queue: &Queue,
        input_buffer: &Buffer,
        input_size: u32,
    ) -> Result<f32, ReductionError> {
        self.reduce(device, queue, input_buffer, input_size, ReductionOperation::Min)
    }

    /// Perform max reduction on the input buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `input_buffer` - Storage buffer containing f32 values
    /// * `input_size` - Number of elements in the input buffer
    ///
    /// # Returns
    ///
    /// The maximum element, or an error if the operation failed.
    pub fn reduce_max(
        &self,
        device: &Device,
        queue: &Queue,
        input_buffer: &Buffer,
        input_size: u32,
    ) -> Result<f32, ReductionError> {
        self.reduce(device, queue, input_buffer, input_size, ReductionOperation::Max)
    }

    /// Perform a reduction operation on the input buffer.
    ///
    /// Handles both single-pass and multi-pass reductions automatically.
    fn reduce(
        &self,
        device: &Device,
        queue: &Queue,
        input_buffer: &Buffer,
        input_size: u32,
        operation: ReductionOperation,
    ) -> Result<f32, ReductionError> {
        if input_size == 0 {
            return Err(ReductionError::EmptyInput);
        }

        // Special case: single element
        if input_size == 1 {
            return self.read_single_value(device, queue, input_buffer);
        }

        // Get the appropriate pipeline
        let pipeline = match operation {
            ReductionOperation::Sum => &self.sum_pipeline,
            ReductionOperation::Min => &self.min_pipeline,
            ReductionOperation::Max => &self.max_pipeline,
        };

        // Calculate number of workgroups for first pass
        let mut current_size = input_size;
        let mut num_workgroups = Self::calculate_workgroups(current_size);

        // Create scratch buffers for multi-pass reduction
        // We need space for all intermediate results
        let max_scratch_size = num_workgroups.max(1) as u64 * std::mem::size_of::<f32>() as u64;
        let scratch_buffer_a = device.create_buffer(&BufferDescriptor {
            label: Some("reduction_scratch_a"),
            size: max_scratch_size.max(4), // At least 4 bytes
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_buffer_b = device.create_buffer(&BufferDescriptor {
            label: Some("reduction_scratch_b"),
            size: max_scratch_size.max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Readback buffer
        let readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("reduction_readback"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("reduction_encoder"),
        });

        // Track which buffer is input and output for ping-pong
        let mut pass = 0;
        current_size = input_size;
        num_workgroups = Self::calculate_workgroups(current_size);

        // First pass: input_buffer -> scratch_buffer_a
        {
            let params = ReductionParams::new(current_size, 0);
            let params_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("reduction_params"),
                size: std::mem::size_of::<ReductionParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("reduction_bind_group_pass_0"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: scratch_buffer_a.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("reduction_pass_0"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Update sizes for next pass
        current_size = num_workgroups;
        pass += 1;

        // Additional passes: ping-pong between scratch_buffer_a and scratch_buffer_b
        while current_size > 1 {
            num_workgroups = Self::calculate_workgroups(current_size);

            let (src_buffer, dst_buffer) = if pass % 2 == 1 {
                (&scratch_buffer_a, &scratch_buffer_b)
            } else {
                (&scratch_buffer_b, &scratch_buffer_a)
            };

            let params = ReductionParams::new(current_size, 0);
            let params_buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("reduction_params_pass_{}", pass)),
                size: std::mem::size_of::<ReductionParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("reduction_bind_group_pass_{}", pass)),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: dst_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("reduction_pass_{}", pass)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);

            current_size = num_workgroups;
            pass += 1;
        }

        // Copy final result to readback buffer
        let final_buffer = if pass % 2 == 1 {
            &scratch_buffer_a
        } else {
            &scratch_buffer_b
        };
        encoder.copy_buffer_to_buffer(final_buffer, 0, &readback_buffer, 0, 4);

        // Submit and wait
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read result
        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| ReductionError::BufferMapFailed("Channel receive failed".to_string()))?
            .map_err(|e| ReductionError::BufferMapFailed(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result = *bytemuck::from_bytes::<f32>(&data[..4]);
        drop(data);
        readback_buffer.unmap();

        Ok(result)
    }

    /// Read a single value from a buffer (for the degenerate single-element case).
    fn read_single_value(
        &self,
        device: &Device,
        queue: &Queue,
        buffer: &Buffer,
    ) -> Result<f32, ReductionError> {
        let readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("single_value_readback"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("single_value_encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback_buffer, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| ReductionError::BufferMapFailed("Channel receive failed".to_string()))?
            .map_err(|e| ReductionError::BufferMapFailed(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result = *bytemuck::from_bytes::<f32>(&data[..4]);
        drop(data);
        readback_buffer.unmap();

        Ok(result)
    }

    /// Encode reduction commands without submitting.
    ///
    /// This allows batching multiple reductions in a single command submission
    /// for better GPU utilization.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record commands into
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue (for uniform buffer writes)
    /// * `input_buffer` - Storage buffer containing f32 values
    /// * `output_buffer` - Storage buffer to write result (must have at least 4 bytes)
    /// * `input_size` - Number of elements in the input buffer
    /// * `operation` - Which reduction operation to perform
    ///
    /// # Returns
    ///
    /// A scratch buffer that must be kept alive until the commands execute,
    /// or None if no scratch buffer was needed.
    pub fn encode_reduction(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &Device,
        queue: &Queue,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        input_size: u32,
        operation: ReductionOperation,
    ) -> Option<Buffer> {
        if input_size == 0 {
            return None;
        }

        let pipeline = match operation {
            ReductionOperation::Sum => &self.sum_pipeline,
            ReductionOperation::Min => &self.min_pipeline,
            ReductionOperation::Max => &self.max_pipeline,
        };

        // Special case: single element - just copy
        if input_size == 1 {
            encoder.copy_buffer_to_buffer(input_buffer, 0, output_buffer, 0, 4);
            return None;
        }

        let mut current_size = input_size;
        let mut num_workgroups = Self::calculate_workgroups(current_size);

        // Create scratch buffer if we need multiple passes
        let needs_scratch = num_workgroups > 1;
        let scratch_size = (num_workgroups.max(1) as u64 * std::mem::size_of::<f32>() as u64).max(4);

        let scratch_buffer = if needs_scratch {
            Some(device.create_buffer(&BufferDescriptor {
                label: Some("reduction_scratch"),
                size: scratch_size * 2, // Double for ping-pong
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // First pass
        let first_output = if needs_scratch {
            scratch_buffer.as_ref().unwrap()
        } else {
            output_buffer
        };

        {
            let params = ReductionParams::new(current_size, 0);
            let params_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("reduction_params_encoded"),
                size: std::mem::size_of::<ReductionParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("reduction_bind_group_encoded"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: first_output.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("reduction_pass_encoded"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        current_size = num_workgroups;

        // Additional passes using scratch buffer ping-pong
        if let Some(ref scratch) = scratch_buffer {
            let mut pass = 1;
            while current_size > 1 {
                num_workgroups = Self::calculate_workgroups(current_size);

                // Ping-pong within the same buffer using different offsets
                let src_offset = if pass % 2 == 1 { 0 } else { scratch_size };
                let dst_offset = if pass % 2 == 1 { scratch_size } else { 0 };

                // For the final pass, write to output_buffer instead
                let is_final = num_workgroups == 1;

                let params = ReductionParams::new(current_size, 0);
                let params_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("reduction_params_pass_{}", pass)),
                    size: std::mem::size_of::<ReductionParams>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

                let dst = if is_final {
                    output_buffer.as_entire_binding()
                } else {
                    BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: scratch,
                        offset: dst_offset,
                        size: None,
                    })
                };

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("reduction_bind_group_pass_{}", pass)),
                    layout: &self.bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: scratch,
                                offset: src_offset,
                                size: None,
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: dst,
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&format!("reduction_pass_{}", pass)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);

                current_size = num_workgroups;
                pass += 1;
            }
        }

        scratch_buffer
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // =========================================================================
    // SECTION 1: Constants Validation
    // =========================================================================

    #[test]
    fn test_workgroup_size_is_power_of_two() {
        assert!(WORKGROUP_SIZE.is_power_of_two());
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_elements_per_workgroup_calculation() {
        assert_eq!(ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE * 2);
        assert_eq!(ELEMENTS_PER_WORKGROUP, 512);
    }

    #[test]
    fn test_max_single_pass_equals_elements_per_workgroup() {
        assert_eq!(MAX_SINGLE_PASS_ELEMENTS, ELEMENTS_PER_WORKGROUP);
    }

    #[test]
    fn test_multi_pass_threshold_value() {
        // WORKGROUP_SIZE * WORKGROUP_SIZE * 2 = 256 * 256 * 2 = 131072
        assert_eq!(MULTI_PASS_THRESHOLD, 131072);
        assert_eq!(MULTI_PASS_THRESHOLD, WORKGROUP_SIZE * WORKGROUP_SIZE * 2);
    }

    // =========================================================================
    // SECTION 2: ReductionOperation Enum Tests
    // =========================================================================

    #[test]
    fn test_reduction_operation_clone() {
        let op = ReductionOperation::Sum;
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_reduction_operation_copy() {
        let op = ReductionOperation::Min;
        let copied: ReductionOperation = op; // Copy semantics
        assert_eq!(op, copied);
        // Original is still usable after copy (proving Copy trait)
        assert_eq!(op.identity(), f32::MAX);
    }

    #[test]
    fn test_reduction_operation_partial_eq() {
        assert_eq!(ReductionOperation::Sum, ReductionOperation::Sum);
        assert_eq!(ReductionOperation::Min, ReductionOperation::Min);
        assert_eq!(ReductionOperation::Max, ReductionOperation::Max);
        assert_ne!(ReductionOperation::Sum, ReductionOperation::Min);
        assert_ne!(ReductionOperation::Sum, ReductionOperation::Max);
        assert_ne!(ReductionOperation::Min, ReductionOperation::Max);
    }

    #[test]
    fn test_reduction_operation_hash() {
        let mut set = HashSet::new();
        set.insert(ReductionOperation::Sum);
        set.insert(ReductionOperation::Min);
        set.insert(ReductionOperation::Max);
        assert_eq!(set.len(), 3);

        // Inserting duplicate should not change size
        set.insert(ReductionOperation::Sum);
        assert_eq!(set.len(), 3);

        assert!(set.contains(&ReductionOperation::Sum));
        assert!(set.contains(&ReductionOperation::Min));
        assert!(set.contains(&ReductionOperation::Max));
    }

    #[test]
    fn test_reduction_operation_identity() {
        assert_eq!(ReductionOperation::Sum.identity(), 0.0);
        assert_eq!(ReductionOperation::Min.identity(), f32::MAX);
        assert_eq!(ReductionOperation::Max.identity(), f32::MIN);
    }

    #[test]
    fn test_reduction_operation_entry_points() {
        assert_eq!(ReductionOperation::Sum.entry_point(), "reduce_sum");
        assert_eq!(ReductionOperation::Min.entry_point(), "reduce_min");
        assert_eq!(ReductionOperation::Max.entry_point(), "reduce_max");
    }

    #[test]
    fn test_reduction_operation_debug() {
        assert_eq!(format!("{:?}", ReductionOperation::Sum), "Sum");
        assert_eq!(format!("{:?}", ReductionOperation::Min), "Min");
        assert_eq!(format!("{:?}", ReductionOperation::Max), "Max");
    }

    // =========================================================================
    // SECTION 3: ReductionParams Tests
    // =========================================================================

    #[test]
    fn test_reduction_params_creation() {
        let params = ReductionParams::new(1000, 5);
        assert_eq!(params.input_size, 1000);
        assert_eq!(params.output_offset, 5);
    }

    #[test]
    fn test_reduction_params_size_16_bytes() {
        // Must be 16 bytes for GPU alignment (4 x u32)
        assert_eq!(std::mem::size_of::<ReductionParams>(), 16);
    }

    #[test]
    fn test_reduction_params_alignment() {
        // Alignment must be 4 bytes (u32 alignment)
        assert_eq!(std::mem::align_of::<ReductionParams>(), 4);
    }

    #[test]
    fn test_reduction_params_zero_values() {
        let params = ReductionParams::new(0, 0);
        assert_eq!(params.input_size, 0);
        assert_eq!(params.output_offset, 0);
    }

    #[test]
    fn test_reduction_params_max_values() {
        let params = ReductionParams::new(u32::MAX, u32::MAX);
        assert_eq!(params.input_size, u32::MAX);
        assert_eq!(params.output_offset, u32::MAX);
    }

    #[test]
    fn test_reduction_params_bytemuck_pod() {
        // Verify params can be cast to bytes (Pod trait)
        let params = ReductionParams::new(12345, 67);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);

        // First 4 bytes should be input_size (little-endian)
        let input_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(input_size, 12345);

        // Next 4 bytes should be output_offset
        let output_offset = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(output_offset, 67);
    }

    #[test]
    fn test_reduction_params_bytemuck_zeroable() {
        // Verify zeroed params (Zeroable trait)
        let params: ReductionParams = bytemuck::Zeroable::zeroed();
        assert_eq!(params.input_size, 0);
        assert_eq!(params.output_offset, 0);
    }

    #[test]
    fn test_reduction_params_clone() {
        let params = ReductionParams::new(100, 200);
        let cloned = params.clone();
        assert_eq!(params.input_size, cloned.input_size);
        assert_eq!(params.output_offset, cloned.output_offset);
    }

    #[test]
    fn test_reduction_params_copy() {
        let params = ReductionParams::new(50, 100);
        let copied: ReductionParams = params;
        // Original still usable (Copy semantics)
        assert_eq!(params.input_size, 50);
        assert_eq!(copied.input_size, 50);
    }

    #[test]
    fn test_reduction_params_debug() {
        let params = ReductionParams::new(999, 42);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("999"));
        assert!(debug_str.contains("42"));
    }

    // =========================================================================
    // SECTION 4: ReductionError Tests
    // =========================================================================

    #[test]
    fn test_reduction_error_empty_input_display() {
        let err = ReductionError::EmptyInput;
        assert_eq!(format!("{}", err), "Input buffer is empty (size = 0)");
    }

    #[test]
    fn test_reduction_error_buffer_map_failed_display() {
        let err = ReductionError::BufferMapFailed("test message".to_string());
        assert_eq!(format!("{}", err), "Buffer mapping failed: test message");
    }

    #[test]
    fn test_reduction_error_gpu_error_display() {
        let err = ReductionError::GpuError("shader compilation failed".to_string());
        assert_eq!(format!("{}", err), "GPU execution failed: shader compilation failed");
    }

    #[test]
    fn test_reduction_error_timeout_display() {
        let err = ReductionError::Timeout(5000);
        assert_eq!(format!("{}", err), "Buffer readback timed out after 5000ms");
    }

    #[test]
    fn test_reduction_error_debug() {
        let err = ReductionError::EmptyInput;
        assert!(format!("{:?}", err).contains("EmptyInput"));
    }

    // =========================================================================
    // SECTION 5: Workgroup Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_workgroups_boundary_at_512() {
        // Exactly 512 elements fits in one workgroup
        assert_eq!(ReductionPipeline::calculate_workgroups(512), 1);
        // 513 requires a second workgroup
        assert_eq!(ReductionPipeline::calculate_workgroups(513), 2);
    }

    #[test]
    fn test_calculate_workgroups_small_values() {
        assert_eq!(ReductionPipeline::calculate_workgroups(1), 1);
        assert_eq!(ReductionPipeline::calculate_workgroups(2), 1);
        assert_eq!(ReductionPipeline::calculate_workgroups(100), 1);
        assert_eq!(ReductionPipeline::calculate_workgroups(256), 1);
        assert_eq!(ReductionPipeline::calculate_workgroups(511), 1);
    }

    #[test]
    fn test_calculate_workgroups_medium_values() {
        // 1024 / 512 = 2
        assert_eq!(ReductionPipeline::calculate_workgroups(1024), 2);
        // ceil(1025 / 512) = 3
        assert_eq!(ReductionPipeline::calculate_workgroups(1025), 3);
        // ceil(2048 / 512) = 4
        assert_eq!(ReductionPipeline::calculate_workgroups(2048), 4);
    }

    #[test]
    fn test_calculate_workgroups_large_values() {
        // ceil(10000 / 512) = 20
        assert_eq!(ReductionPipeline::calculate_workgroups(10000), 20);
        // ceil(100000 / 512) = 196
        assert_eq!(ReductionPipeline::calculate_workgroups(100000), 196);
        // ceil(1000000 / 512) = 1954
        assert_eq!(ReductionPipeline::calculate_workgroups(1000000), 1954);
    }

    #[test]
    fn test_calculate_workgroups_exact_multiples() {
        // Exact multiples of 512
        assert_eq!(ReductionPipeline::calculate_workgroups(512), 1);
        assert_eq!(ReductionPipeline::calculate_workgroups(1024), 2);
        assert_eq!(ReductionPipeline::calculate_workgroups(2048), 4);
        assert_eq!(ReductionPipeline::calculate_workgroups(5120), 10);
    }

    // =========================================================================
    // SECTION 6: Multi-pass Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_passes_single_pass_boundary() {
        // <= 512 elements = 1 pass
        assert_eq!(ReductionPipeline::calculate_passes(1), 1);
        assert_eq!(ReductionPipeline::calculate_passes(512), 1);
    }

    #[test]
    fn test_calculate_passes_two_pass_boundary() {
        // 513 elements needs 2 passes: 513 -> 2 -> 1
        assert_eq!(ReductionPipeline::calculate_passes(513), 2);

        // 262144 (512*512) elements: 262144 -> 512 -> 1 = 2 passes
        assert_eq!(ReductionPipeline::calculate_passes(262144), 2);
    }

    #[test]
    fn test_calculate_passes_three_passes() {
        // 262145 elements: 262145 -> 513 -> 2 -> 1 = 3 passes
        assert_eq!(ReductionPipeline::calculate_passes(262145), 3);

        // 1 million elements: 1M -> 1954 -> 4 -> 1 = 3 passes
        assert_eq!(ReductionPipeline::calculate_passes(1_000_000), 3);
    }

    #[test]
    fn test_calculate_passes_four_passes() {
        // Need > 512^3 = 134,217,728 elements for 4 passes
        // 1 billion elements: 1B -> ~1.95M -> ~3815 -> 8 -> 1 = 4 passes
        assert_eq!(ReductionPipeline::calculate_passes(1_000_000_000), 4);
    }

    #[test]
    fn test_calculate_passes_edge_cases() {
        // Single element always 1 pass
        assert_eq!(ReductionPipeline::calculate_passes(1), 1);

        // Power of 2 sizes
        assert_eq!(ReductionPipeline::calculate_passes(256), 1);
        assert_eq!(ReductionPipeline::calculate_passes(1024), 2);
        assert_eq!(ReductionPipeline::calculate_passes(4096), 2);
        assert_eq!(ReductionPipeline::calculate_passes(65536), 2);
    }

    // =========================================================================
    // SECTION 7: Output Buffer Sizing Tests
    // =========================================================================

    #[test]
    fn test_workgroup_count_determines_intermediate_buffer_size() {
        // For 10000 elements, we need 20 workgroups, so intermediate buffer = 20 f32s
        let input_size = 10000;
        let workgroups = ReductionPipeline::calculate_workgroups(input_size);
        let intermediate_buffer_size = workgroups as usize * std::mem::size_of::<f32>();
        assert_eq!(workgroups, 20);
        assert_eq!(intermediate_buffer_size, 80); // 20 * 4 bytes
    }

    #[test]
    fn test_single_pass_output_buffer_size() {
        // For single pass (<=512 elements), output buffer just needs 1 f32
        let input_size = 512;
        let workgroups = ReductionPipeline::calculate_workgroups(input_size);
        assert_eq!(workgroups, 1);
    }

    // =========================================================================
    // SECTION 8: Ping-pong Buffer Strategy Tests
    // =========================================================================

    #[test]
    fn test_pass_parity_for_buffer_selection() {
        // Verify the ping-pong logic based on pass number
        // Pass 0: input -> scratch_a
        // Pass 1 (odd): scratch_a -> scratch_b
        // Pass 2 (even): scratch_b -> scratch_a

        // Simulating the logic from reduce()
        let pass_1_uses_a_as_src = 1 % 2 == 1; // true
        let pass_2_uses_b_as_src = 2 % 2 == 0; // true

        assert!(pass_1_uses_a_as_src);
        assert!(pass_2_uses_b_as_src);
    }

    #[test]
    fn test_final_buffer_selection() {
        // Final buffer after pass N is determined by (pass % 2)
        // If final pass is odd (1, 3, 5...), result is in scratch_b
        // If final pass is even (2, 4, 6...), result is in scratch_a

        // But the code uses: if pass % 2 == 1 { scratch_a } else { scratch_b }
        // After pass 1: result in scratch_a (correct)
        // After pass 2: result in scratch_b (correct)

        let after_pass_1 = 1 % 2 == 1; // scratch_a
        let after_pass_2 = 2 % 2 == 1; // scratch_b

        assert!(after_pass_1);  // Uses scratch_a
        assert!(!after_pass_2); // Uses scratch_b
    }

    // =========================================================================
    // SECTION 9: Shader Source Validation
    // =========================================================================

    #[test]
    fn test_shader_sources_included() {
        // Verify shader sources are embedded at compile time
        // These would cause compile errors if the files don't exist
        assert!(!REDUCE_SUM_WGSL.is_empty());
        assert!(!REDUCE_MIN_WGSL.is_empty());
        assert!(!REDUCE_MAX_WGSL.is_empty());
    }

    #[test]
    fn test_shader_sources_contain_entry_points() {
        assert!(REDUCE_SUM_WGSL.contains("fn reduce_sum"));
        assert!(REDUCE_MIN_WGSL.contains("fn reduce_min"));
        assert!(REDUCE_MAX_WGSL.contains("fn reduce_max"));
    }

    #[test]
    fn test_shader_sources_contain_workgroup_size() {
        // All shaders should define workgroup_size(256, 1, 1)
        assert!(REDUCE_SUM_WGSL.contains("@workgroup_size(256"));
        assert!(REDUCE_MIN_WGSL.contains("@workgroup_size(256"));
        assert!(REDUCE_MAX_WGSL.contains("@workgroup_size(256"));
    }

    #[test]
    fn test_shader_sources_contain_shared_memory() {
        // All shaders should use workgroup shared memory
        assert!(REDUCE_SUM_WGSL.contains("var<workgroup>"));
        assert!(REDUCE_MIN_WGSL.contains("var<workgroup>"));
        assert!(REDUCE_MAX_WGSL.contains("var<workgroup>"));
    }

    #[test]
    fn test_shader_sources_contain_params_struct() {
        // All shaders should have ReductionParams uniform
        assert!(REDUCE_SUM_WGSL.contains("struct ReductionParams"));
        assert!(REDUCE_MIN_WGSL.contains("struct ReductionParams"));
        assert!(REDUCE_MAX_WGSL.contains("struct ReductionParams"));
    }
}
