//! Visibility Stream Compaction Pipeline (T-WGPU-P6.6.1).
//!
//! This module provides GPU-based stream compaction for visible objects using
//! bit-packed visibility flags and prefix scan results. It efficiently produces
//! a densely-packed buffer containing only the indices of visible objects.
//!
//! # Overview
//!
//! Stream compaction is a fundamental GPU primitive for GPU-driven rendering:
//!
//! ```text
//! Input:  visibility_flags = [1,0,1,1,0,0,1,0] (packed bits)
//! Output: compacted_indices = [0, 2, 3, 6]
//!         compacted_count = 4
//! ```
//!
//! The algorithm uses exclusive prefix scan for stable compaction:
//!
//! 1. Culling passes set bits in visibility_flags buffer
//! 2. Prefix scan computes output indices: prefix_sum[i] = count(visible[0..i))
//! 3. Scatter: if visible[i], write i to compacted_indices[prefix_sum[i]]
//! 4. Count: total = prefix_sum[n-1] + visible[n-1]
//!
//! # Stability
//!
//! The compaction is stable - visible objects maintain their relative order
//! from the original array. This is important for consistent rendering order.
//!
//! # Performance
//!
//! - Workgroup size: 64 threads
//! - One thread per object (standard) or batch mode (4 objects/thread)
//! - Memory access pattern optimized for visibility buffer coalescing
//! - Separate count kernel avoids atomic contention
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{
//!     StreamCompactPipeline, CompactedIndices, StreamCompactParams,
//!     VisibilityFlagsBuffer, PrefixScanPipeline,
//! };
//!
//! // Create pipeline
//! let pipeline = StreamCompactPipeline::new(&device);
//!
//! // Create output buffer
//! let compacted = CompactedIndices::new(&device, max_objects);
//!
//! // Each frame after culling and prefix scan:
//! pipeline.dispatch(
//!     &mut encoder,
//!     &device,
//!     &visibility_flags,
//!     &prefix_sum_buffer,
//!     &compacted,
//!     object_count,
//! );
//!
//! // Read back count if needed
//! let count = compacted.read_count(&device, &queue);
//! ```

use bytemuck::{Pod, Zeroable};
use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Workgroup size for compaction kernels (64 threads per workgroup).
///
/// 64 is chosen as a balance between:
/// - GPU occupancy (multiple of warp/wavefront size)
/// - Memory coalescing efficiency
/// - Thread-level parallelism
pub const WORKGROUP_SIZE: u32 = 64;

/// Number of bits per visibility word.
pub const BITS_PER_WORD: u32 = 32;

/// Size of CompactParams in bytes.
pub const COMPACT_PARAMS_SIZE: usize = 16;

/// Batch size for batch compaction mode (objects per thread).
pub const BATCH_SIZE: u32 = 4;

/// Shader source path for include.
pub const COMPACT_SHADER: &str = include_str!("../../shaders/compact.wgsl");

// =============================================================================
// COMPACT PARAMETERS
// =============================================================================

/// Parameters for stream compaction dispatch.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field        | Size | Description                 |
/// |--------|--------------|------|-----------------------------|
/// | 0      | object_count | 4    | Number of objects to process|
/// | 4      | _pad0        | 4    | Reserved/padding            |
/// | 8      | _pad1        | 4    | Reserved/padding            |
/// | 12     | _pad2        | 4    | Reserved/padding            |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct StreamCompactParams {
    /// Total number of objects to process.
    pub object_count: u32,
    /// Reserved for future use.
    pub _pad0: u32,
    /// Reserved for alignment.
    pub _pad1: u32,
    /// Reserved for alignment.
    pub _pad2: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<StreamCompactParams>() == COMPACT_PARAMS_SIZE);

impl StreamCompactParams {
    /// Create new stream compact parameters.
    ///
    /// # Arguments
    ///
    /// * `object_count` - Number of objects to compact.
    #[inline]
    pub fn new(object_count: u32) -> Self {
        Self {
            object_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Calculate number of workgroups needed for standard dispatch.
    #[inline]
    pub fn workgroups(&self) -> u32 {
        (self.object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Calculate number of workgroups needed for batch dispatch.
    #[inline]
    pub fn workgroups_batch(&self) -> u32 {
        let objects_per_workgroup = WORKGROUP_SIZE * BATCH_SIZE;
        (self.object_count + objects_per_workgroup - 1) / objects_per_workgroup
    }
}

// =============================================================================
// COMPACTED INDICES BUFFER
// =============================================================================

/// GPU buffer for compacted visible object indices.
///
/// Contains the output of stream compaction:
/// - `buffer`: Array of visible object indices in original order
/// - `count_buffer`: Atomic counter with total visible count
///
/// # Memory Layout
///
/// ```text
/// buffer:        [idx0, idx1, idx2, ..., idxN-1]  (N = count)
/// count_buffer:  [count]  (single u32)
/// ```
pub struct CompactedIndices {
    /// GPU buffer for compacted indices.
    buffer: wgpu::Buffer,
    /// GPU buffer for compacted count (single u32).
    count_buffer: wgpu::Buffer,
    /// Staging buffer for reading count back to CPU.
    count_staging: wgpu::Buffer,
    /// Maximum capacity in objects.
    capacity: u32,
    /// Debug label.
    label: Option<String>,
}

impl CompactedIndices {
    /// Create a new compacted indices buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of indices to store.
    /// * `label` - Optional debug label.
    pub fn new(device: &wgpu::Device, capacity: u32, label: Option<&str>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_compacted_indices", l)).as_deref(),
            size: (capacity as u64) * 4, // u32 per index
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label
                .map(|l| format!("{}_compacted_count", l))
                .as_deref(),
            size: 4, // single u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: label
                .map(|l| format!("{}_compacted_count_staging", l))
                .as_deref(),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            count_buffer,
            count_staging,
            capacity,
            label: label.map(String::from),
        }
    }

    /// Get the indices buffer for binding.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the count buffer for binding.
    #[inline]
    pub fn count_buffer(&self) -> &wgpu::Buffer {
        &self.count_buffer
    }

    /// Get buffer binding for indices.
    #[inline]
    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'_> {
        self.buffer.as_entire_buffer_binding()
    }

    /// Get buffer binding for count.
    #[inline]
    pub fn count_binding(&self) -> wgpu::BufferBinding<'_> {
        self.count_buffer.as_entire_buffer_binding()
    }

    /// Get the capacity in objects.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        (self.capacity as u64) * 4
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Clear the count buffer to 0.
    pub fn clear_count(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.count_buffer, 0, bytemuck::bytes_of(&0u32));
    }

    /// Clear count using command encoder.
    pub fn clear_count_with_encoder(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.count_buffer, 0, None);
    }

    /// Read the compacted count back to CPU (synchronous).
    ///
    /// Blocks until GPU completes the readback.
    pub fn read_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.count_buffer, 0, &self.count_staging, 0, 4);
        queue.submit([encoder.finish()]);

        let buffer_slice = self.count_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        self.count_staging.unmap();

        count
    }

    /// Read compacted indices back to CPU (synchronous).
    ///
    /// First reads count, then reads that many indices.
    pub fn read_indices(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        let count = self.read_count(device, queue);
        if count == 0 {
            return Vec::new();
        }

        let bytes_to_read = (count as u64) * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compacted_indices_staging"),
            size: bytes_to_read,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, bytes_to_read);
        queue.submit([encoder.finish()]);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let indices: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        indices
    }

    /// Resize buffer if needed.
    ///
    /// Creates new buffer if capacity is insufficient.
    pub fn resize(&mut self, device: &wgpu::Device, new_capacity: u32) -> bool {
        if new_capacity > self.capacity {
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: self
                    .label
                    .as_ref()
                    .map(|l| format!("{}_compacted_indices", l))
                    .as_deref(),
                size: (new_capacity as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_capacity;
            return true;
        }
        false
    }
}

impl std::fmt::Debug for CompactedIndices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactedIndices")
            .field("capacity", &self.capacity)
            .field("buffer_size", &self.buffer_size())
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// STREAM COMPACT PIPELINE
// =============================================================================

/// Compute pipeline for visibility stream compaction.
///
/// Contains compute pipelines for different compaction modes:
/// - `compact_pipeline`: Standard 1-object-per-thread compaction
/// - `count_pipeline`: Compute total visible count
/// - `batch_pipeline`: Batch mode (4 objects per thread)
/// - `fused_pipeline`: Combined scatter + count
pub struct StreamCompactPipeline {
    /// Pipeline for scatter (compact_main entry point).
    compact_pipeline: wgpu::ComputePipeline,
    /// Pipeline for count (get_count_main entry point).
    count_pipeline: wgpu::ComputePipeline,
    /// Pipeline for batch scatter (compact_batch entry point).
    batch_pipeline: wgpu::ComputePipeline,
    /// Pipeline for fused scatter+count (compact_fused entry point).
    fused_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for visibility + prefix sum (group 0).
    input_layout: wgpu::BindGroupLayout,
    /// Bind group layout for output (group 1).
    output_layout: wgpu::BindGroupLayout,
    /// Bind group layout for parameters (group 2).
    params_layout: wgpu::BindGroupLayout,
}

impl StreamCompactPipeline {
    /// Create a new stream compact pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let input_layout = Self::create_input_layout(device);
        let output_layout = Self::create_output_layout(device);
        let params_layout = Self::create_params_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(
            device,
            &input_layout,
            &output_layout,
            &params_layout,
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stream_compact_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPACT_SHADER.into()),
        });

        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_count_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "get_count_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_batch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_batch",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let fused_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_fused_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_fused",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            compact_pipeline,
            count_pipeline,
            batch_pipeline,
            fused_pipeline,
            input_layout,
            output_layout,
            params_layout,
        }
    }

    /// Get the input bind group layout (visibility + prefix sum).
    #[inline]
    pub fn input_layout(&self) -> &wgpu::BindGroupLayout {
        &self.input_layout
    }

    /// Get the output bind group layout.
    #[inline]
    pub fn output_layout(&self) -> &wgpu::BindGroupLayout {
        &self.output_layout
    }

    /// Get the params bind group layout.
    #[inline]
    pub fn params_layout(&self) -> &wgpu::BindGroupLayout {
        &self.params_layout
    }

    /// Create bind group for input (visibility flags + prefix sum).
    pub fn create_input_bind_group(
        &self,
        device: &wgpu::Device,
        visibility_buffer: &wgpu::Buffer,
        prefix_sum_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_input_bind_group"),
            layout: &self.input_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_sum_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create bind group for output (compacted indices + count).
    pub fn create_output_bind_group(
        &self,
        device: &wgpu::Device,
        compacted: &CompactedIndices,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_output_bind_group"),
            layout: &self.output_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compacted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compacted.count_buffer().as_entire_binding(),
                },
            ],
        })
    }

    /// Create bind group for parameters.
    pub fn create_params_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_params_bind_group"),
            layout: &self.params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        })
    }

    /// Dispatch standard compaction (scatter + count).
    ///
    /// Runs compact_main followed by get_count_main.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
        params: &StreamCompactParams,
    ) {
        // Scatter pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_scatter"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, input_bind_group, &[]);
            pass.set_bind_group(1, output_bind_group, &[]);
            pass.set_bind_group(2, params_bind_group, &[]);
            pass.dispatch_workgroups(params.workgroups(), 1, 1);
        }

        // Count pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_count"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, input_bind_group, &[]);
            pass.set_bind_group(1, output_bind_group, &[]);
            pass.set_bind_group(2, params_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Dispatch scatter only (no count computation).
    ///
    /// Use when count is not needed or already known.
    pub fn dispatch_scatter_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
        params: &StreamCompactParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stream_compact_scatter_only"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.compact_pipeline);
        pass.set_bind_group(0, input_bind_group, &[]);
        pass.set_bind_group(1, output_bind_group, &[]);
        pass.set_bind_group(2, params_bind_group, &[]);
        pass.dispatch_workgroups(params.workgroups(), 1, 1);
    }

    /// Dispatch batch compaction (4 objects per thread).
    ///
    /// More efficient for large object counts due to better memory coalescing.
    pub fn dispatch_batch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
        params: &StreamCompactParams,
    ) {
        // Batch scatter pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_batch"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.batch_pipeline);
            pass.set_bind_group(0, input_bind_group, &[]);
            pass.set_bind_group(1, output_bind_group, &[]);
            pass.set_bind_group(2, params_bind_group, &[]);
            pass.dispatch_workgroups(params.workgroups_batch(), 1, 1);
        }

        // Count pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_count"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, input_bind_group, &[]);
            pass.set_bind_group(1, output_bind_group, &[]);
            pass.set_bind_group(2, params_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Create input bind group layout (group 0).
    fn create_input_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stream_compact_input_layout"),
            entries: &[
                // binding 0: visibility_flags (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: prefix_sum (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create output bind group layout (group 1).
    fn create_output_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stream_compact_output_layout"),
            entries: &[
                // binding 0: compacted_indices (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: compacted_count (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create params bind group layout (group 2).
    fn create_params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stream_compact_params_layout"),
            entries: &[
                // binding 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create pipeline layout.
    fn create_pipeline_layout(
        device: &wgpu::Device,
        input_layout: &wgpu::BindGroupLayout,
        output_layout: &wgpu::BindGroupLayout,
        params_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stream_compact_pipeline_layout"),
            bind_group_layouts: &[input_layout, output_layout, params_layout],
            push_constant_ranges: &[],
        })
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate number of workgroups for standard dispatch.
#[inline]
pub fn workgroups_for_objects(object_count: u32) -> u32 {
    (object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

/// Calculate number of workgroups for batch dispatch.
#[inline]
pub fn workgroups_for_objects_batch(object_count: u32) -> u32 {
    let objects_per_workgroup = WORKGROUP_SIZE * BATCH_SIZE;
    (object_count + objects_per_workgroup - 1) / objects_per_workgroup
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

/// Check if object is visible in packed visibility flags (CPU reference).
#[inline]
pub fn cpu_is_visible(visibility_flags: &[u32], object_idx: usize) -> bool {
    let word_idx = object_idx / BITS_PER_WORD as usize;
    let bit_idx = object_idx % BITS_PER_WORD as usize;
    if word_idx < visibility_flags.len() {
        (visibility_flags[word_idx] & (1 << bit_idx)) != 0
    } else {
        false
    }
}

/// CPU reference implementation of stream compaction.
///
/// Compacts visible objects using visibility flags and prefix sum.
/// Returns (compacted_indices, count).
pub fn cpu_stream_compact(
    visibility_flags: &[u32],
    prefix_sum: &[u32],
    object_count: usize,
) -> (Vec<u32>, u32) {
    let mut output = Vec::with_capacity(object_count);

    for i in 0..object_count {
        if cpu_is_visible(visibility_flags, i) {
            let output_idx = prefix_sum[i] as usize;
            // Ensure output vector has enough capacity
            if output_idx >= output.len() {
                output.resize(output_idx + 1, 0);
            }
            output[output_idx] = i as u32;
        }
    }

    let count = if object_count > 0 && !prefix_sum.is_empty() {
        let last_prefix = prefix_sum[object_count - 1];
        let last_visible = if cpu_is_visible(visibility_flags, object_count - 1) {
            1
        } else {
            0
        };
        last_prefix + last_visible
    } else {
        0
    };

    (output, count)
}

/// CPU reference implementation that returns just the visible indices.
///
/// Simpler version that doesn't require pre-computed prefix sum.
pub fn cpu_compact_visible_indices(visibility_flags: &[u32], object_count: usize) -> Vec<u32> {
    (0..object_count)
        .filter(|&i| cpu_is_visible(visibility_flags, i))
        .map(|i| i as u32)
        .collect()
}

/// Verify prefix sum is correct for visibility flags (CPU).
///
/// Returns (is_valid, error_description).
pub fn cpu_verify_prefix_sum(
    visibility_flags: &[u32],
    prefix_sum: &[u32],
    object_count: usize,
) -> (bool, Option<String>) {
    if prefix_sum.is_empty() || object_count == 0 {
        return (true, None);
    }

    // First element should be 0 (exclusive scan)
    if prefix_sum[0] != 0 {
        return (
            false,
            Some(format!("prefix_sum[0] should be 0, got {}", prefix_sum[0])),
        );
    }

    // Check prefix sum property: prefix[i] = prefix[i-1] + visibility[i-1]
    for i in 1..object_count.min(prefix_sum.len()) {
        let expected = prefix_sum[i - 1]
            + if cpu_is_visible(visibility_flags, i - 1) {
                1
            } else {
                0
            };
        if prefix_sum[i] != expected {
            return (
                false,
                Some(format!(
                    "prefix_sum[{}] should be {}, got {}",
                    i, expected, prefix_sum[i]
                )),
            );
        }
    }

    (true, None)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // StreamCompactParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compact_params_size() {
        assert_eq!(mem::size_of::<StreamCompactParams>(), COMPACT_PARAMS_SIZE);
        assert_eq!(mem::size_of::<StreamCompactParams>(), 16);
    }

    #[test]
    fn test_compact_params_alignment() {
        assert_eq!(mem::align_of::<StreamCompactParams>(), 4);
    }

    #[test]
    fn test_compact_params_pod() {
        let params = StreamCompactParams::new(1000);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), COMPACT_PARAMS_SIZE);
    }

    #[test]
    fn test_compact_params_workgroups() {
        // Exact multiple
        let params = StreamCompactParams::new(64);
        assert_eq!(params.workgroups(), 1);

        let params = StreamCompactParams::new(128);
        assert_eq!(params.workgroups(), 2);

        // Non-exact multiple
        let params = StreamCompactParams::new(65);
        assert_eq!(params.workgroups(), 2);

        let params = StreamCompactParams::new(1);
        assert_eq!(params.workgroups(), 1);

        let params = StreamCompactParams::new(1000);
        assert_eq!(params.workgroups(), 16); // ceil(1000/64) = 16
    }

    #[test]
    fn test_compact_params_workgroups_batch() {
        // 64 threads * 4 objects = 256 objects per workgroup
        let params = StreamCompactParams::new(256);
        assert_eq!(params.workgroups_batch(), 1);

        let params = StreamCompactParams::new(512);
        assert_eq!(params.workgroups_batch(), 2);

        let params = StreamCompactParams::new(257);
        assert_eq!(params.workgroups_batch(), 2);

        let params = StreamCompactParams::new(1000);
        assert_eq!(params.workgroups_batch(), 4); // ceil(1000/256) = 4
    }

    // -------------------------------------------------------------------------
    // CPU is_visible Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_is_visible_basic() {
        let flags = vec![0b00000101u32]; // Objects 0 and 2 visible

        assert!(cpu_is_visible(&flags, 0));
        assert!(!cpu_is_visible(&flags, 1));
        assert!(cpu_is_visible(&flags, 2));
        assert!(!cpu_is_visible(&flags, 3));
    }

    #[test]
    fn test_cpu_is_visible_cross_word() {
        let flags = vec![0x80000000u32, 0x00000001u32]; // Objects 31 and 32

        assert!(!cpu_is_visible(&flags, 30));
        assert!(cpu_is_visible(&flags, 31));
        assert!(cpu_is_visible(&flags, 32));
        assert!(!cpu_is_visible(&flags, 33));
    }

    #[test]
    fn test_cpu_is_visible_out_of_bounds() {
        let flags = vec![0xFFFFFFFFu32];

        // Out of bounds should return false
        assert!(!cpu_is_visible(&flags, 100));
    }

    // -------------------------------------------------------------------------
    // CPU Stream Compact Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_stream_compact_all_visible() {
        // All 4 objects visible
        let flags = vec![0b00001111u32];
        let prefix_sum = vec![0, 1, 2, 3]; // Exclusive scan of [1,1,1,1]

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 4);

        assert_eq!(count, 4);
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_cpu_stream_compact_none_visible() {
        let flags = vec![0u32];
        let prefix_sum = vec![0, 0, 0, 0]; // Exclusive scan of [0,0,0,0]

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 4);

        assert_eq!(count, 0);
        assert!(indices.is_empty() || indices.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_cpu_stream_compact_alternating() {
        // Objects 0, 2, 4, 6 visible
        let flags = vec![0b01010101u32];
        let prefix_sum = vec![0, 1, 1, 2, 2, 3, 3, 4]; // Exclusive scan of [1,0,1,0,1,0,1,0]

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 8);

        assert_eq!(count, 4);
        // Check that visible indices are present
        assert!(indices.contains(&0));
        assert!(indices.contains(&2));
        assert!(indices.contains(&4));
        assert!(indices.contains(&6));
    }

    #[test]
    fn test_cpu_stream_compact_stable_ordering() {
        // Objects 1, 3, 5 visible (odd indices)
        let flags = vec![0b00101010u32];
        let prefix_sum = vec![0, 0, 1, 1, 2, 2, 3]; // Exclusive scan

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 7);

        assert_eq!(count, 3);
        // Verify stable ordering (1 before 3 before 5)
        let idx_1 = indices.iter().position(|&x| x == 1);
        let idx_3 = indices.iter().position(|&x| x == 3);
        let idx_5 = indices.iter().position(|&x| x == 5);
        assert!(idx_1 < idx_3);
        assert!(idx_3 < idx_5);
    }

    // -------------------------------------------------------------------------
    // CPU Verify Prefix Sum Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_verify_prefix_sum_valid() {
        let flags = vec![0b00001111u32]; // First 4 visible
        let prefix_sum = vec![0, 1, 2, 3]; // Correct exclusive scan

        let (valid, error) = cpu_verify_prefix_sum(&flags, &prefix_sum, 4);
        assert!(valid, "Error: {:?}", error);
    }

    #[test]
    fn test_cpu_verify_prefix_sum_invalid_start() {
        let flags = vec![0b00001111u32];
        let prefix_sum = vec![1, 2, 3, 4]; // Wrong: should start with 0

        let (valid, error) = cpu_verify_prefix_sum(&flags, &prefix_sum, 4);
        assert!(!valid);
        assert!(error.unwrap().contains("prefix_sum[0] should be 0"));
    }

    #[test]
    fn test_cpu_verify_prefix_sum_invalid_middle() {
        let flags = vec![0b00001111u32];
        let prefix_sum = vec![0, 1, 5, 3]; // Wrong: [2] should be 2

        let (valid, error) = cpu_verify_prefix_sum(&flags, &prefix_sum, 4);
        assert!(!valid);
        assert!(error.is_some());
    }

    // -------------------------------------------------------------------------
    // CPU Compact Visible Indices Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_compact_visible_indices_basic() {
        let flags = vec![0b00100101u32]; // Objects 0, 2, 5 visible

        let indices = cpu_compact_visible_indices(&flags, 8);

        assert_eq!(indices, vec![0, 2, 5]);
    }

    #[test]
    fn test_cpu_compact_visible_indices_empty() {
        let flags = vec![0u32];

        let indices = cpu_compact_visible_indices(&flags, 8);

        assert!(indices.is_empty());
    }

    #[test]
    fn test_cpu_compact_visible_indices_all() {
        let flags = vec![u32::MAX, u32::MAX];

        let indices = cpu_compact_visible_indices(&flags, 64);

        assert_eq!(indices.len(), 64);
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i as u32);
        }
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compact_shader_parses() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse without errors");

        // Verify expected entry points exist
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "compact_main"),
            "Should have compact_main entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "get_count_main"),
            "Should have get_count_main entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_batch"),
            "Should have compact_batch entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_fused"),
            "Should have compact_fused entry point"
        );
    }

    #[test]
    fn test_compact_shader_validates() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("compact shader should validate without errors");
    }

    #[test]
    fn test_compact_shader_workgroup_sizes() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse without errors");

        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute {
                match ep.name.as_str() {
                    "compact_main" | "compact_batch" | "compact_fused" | "validate_prefix_sum" => {
                        assert_eq!(
                            ep.workgroup_size,
                            [64, 1, 1],
                            "Entry point {} should have workgroup size 64x1x1",
                            ep.name
                        );
                    }
                    "get_count_main" => {
                        assert_eq!(
                            ep.workgroup_size,
                            [1, 1, 1],
                            "Entry point {} should have workgroup size 1x1x1",
                            ep.name
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_compact_shader_entry_points_are_compute() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroups_for_objects() {
        assert_eq!(workgroups_for_objects(0), 0);
        assert_eq!(workgroups_for_objects(1), 1);
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(1000), 16);
    }

    #[test]
    fn test_workgroups_for_objects_batch() {
        // 64 * 4 = 256 objects per workgroup in batch mode
        assert_eq!(workgroups_for_objects_batch(0), 0);
        assert_eq!(workgroups_for_objects_batch(1), 1);
        assert_eq!(workgroups_for_objects_batch(256), 1);
        assert_eq!(workgroups_for_objects_batch(257), 2);
        assert_eq!(workgroups_for_objects_batch(512), 2);
        assert_eq!(workgroups_for_objects_batch(1000), 4);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(WORKGROUP_SIZE, 64);
        assert_eq!(BITS_PER_WORD, 32);
        assert_eq!(BATCH_SIZE, 4);
        assert_eq!(COMPACT_PARAMS_SIZE, 16);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P6.6.1 - Additional Test Coverage
    // =========================================================================

    // -------------------------------------------------------------------------
    // Category 1: Shader Source Validation (Extended)
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_has_correct_bind_groups() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse");

        // Verify shader has global variables for each bind group
        // Group 0: visibility_flags, prefix_sum
        // Group 1: compacted_indices, compacted_count
        // Group 2: params
        let global_names: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, var)| var.name.clone())
            .collect();

        assert!(
            global_names.iter().any(|n| n == "visibility_flags"),
            "Should have visibility_flags binding"
        );
        assert!(
            global_names.iter().any(|n| n == "prefix_sum"),
            "Should have prefix_sum binding"
        );
        assert!(
            global_names.iter().any(|n| n == "compacted_indices"),
            "Should have compacted_indices binding"
        );
        assert!(
            global_names.iter().any(|n| n == "compacted_count"),
            "Should have compacted_count binding"
        );
        assert!(
            global_names.iter().any(|n| n == "params"),
            "Should have params uniform binding"
        );
    }

    #[test]
    fn test_shader_has_helper_functions() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse");

        let function_names: Vec<_> = module
            .functions
            .iter()
            .filter_map(|(_, f)| f.name.clone())
            .collect();

        assert!(
            function_names.iter().any(|n| n == "is_visible"),
            "Should have is_visible helper function"
        );
        assert!(
            function_names.iter().any(|n| n == "get_visibility_bit"),
            "Should have get_visibility_bit helper function"
        );
    }

    #[test]
    fn test_shader_validate_prefix_sum_entry_exists() {
        let module = naga::front::wgsl::parse_str(COMPACT_SHADER)
            .expect("compact shader should parse");

        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "validate_prefix_sum"),
            "Should have validate_prefix_sum debug entry point"
        );
    }

    #[test]
    fn test_shader_constants_match_rust() {
        // The shader defines BITS_PER_WORD = 32 and WORKGROUP_SIZE = 64
        // These must match our Rust constants
        assert_eq!(
            BITS_PER_WORD, 32,
            "BITS_PER_WORD must match shader constant"
        );
        assert_eq!(
            WORKGROUP_SIZE, 64,
            "WORKGROUP_SIZE must match shader constant"
        );
    }

    // -------------------------------------------------------------------------
    // Category 2: Struct Layout Tests (Extended)
    // -------------------------------------------------------------------------

    #[test]
    fn test_compact_params_field_offsets() {
        // Verify field layout matches shader CompactParams struct
        let params = StreamCompactParams::new(1000);
        let bytes = bytemuck::bytes_of(&params);

        // object_count at offset 0
        let object_count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(object_count, 1000);

        // _pad0 at offset 4 should be 0
        let pad0 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(pad0, 0);

        // _pad1 at offset 8 should be 0
        let pad1 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(pad1, 0);

        // _pad2 at offset 12 should be 0
        let pad2 = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad2, 0);
    }

    #[test]
    fn test_compact_params_is_16_byte_aligned() {
        // GPU uniform buffers typically require 16-byte alignment
        assert_eq!(
            mem::size_of::<StreamCompactParams>() % 16,
            0,
            "StreamCompactParams should be 16-byte aligned"
        );
    }

    #[test]
    fn test_compact_params_zero_object_count() {
        let params = StreamCompactParams::new(0);
        assert_eq!(params.object_count, 0);
        assert_eq!(params.workgroups(), 0);
    }

    #[test]
    fn test_compact_params_large_object_count() {
        // Test with large but realistic object counts
        // Maximum practical objects: 16 million (common GPU limit)
        let params = StreamCompactParams::new(16_000_000);
        assert_eq!(params.object_count, 16_000_000);
        assert_eq!(params.workgroups(), 250_000); // ceil(16M/64)

        // Very large but still within bounds that won't overflow workgroups()
        let max_safe = u32::MAX - WORKGROUP_SIZE + 1; // Largest value that won't overflow
        let params = StreamCompactParams::new(max_safe);
        assert_eq!(params.object_count, max_safe);
    }

    #[test]
    fn test_compact_params_default_trait() {
        let params = StreamCompactParams::default();
        assert_eq!(params.object_count, 0);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
        assert_eq!(params._pad2, 0);
    }

    #[test]
    fn test_compact_params_partial_eq() {
        let a = StreamCompactParams::new(100);
        let b = StreamCompactParams::new(100);
        let c = StreamCompactParams::new(200);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -------------------------------------------------------------------------
    // Category 3: Visibility Tests (Extended - Word/Bit Index Calculation)
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_word_index_calculation() {
        // Test that word index = object_idx / 32
        let flags = vec![0xFFFFFFFFu32; 4]; // 128 objects all visible

        // Objects 0-31 in word 0
        for i in 0..32 {
            assert!(
                cpu_is_visible(&flags, i),
                "Object {} should be in word 0 and visible",
                i
            );
        }

        // Objects 32-63 in word 1
        for i in 32..64 {
            assert!(
                cpu_is_visible(&flags, i),
                "Object {} should be in word 1 and visible",
                i
            );
        }
    }

    #[test]
    fn test_visibility_bit_index_calculation() {
        // Test that bit index = object_idx % 32
        // Set only bit 0 of each word
        let flags = vec![0x00000001u32; 4];

        // Objects 0, 32, 64, 96 should be visible (bit 0 of each word)
        assert!(cpu_is_visible(&flags, 0));
        assert!(cpu_is_visible(&flags, 32));
        assert!(cpu_is_visible(&flags, 64));
        assert!(cpu_is_visible(&flags, 96));

        // Objects 1, 33, 65, 97 should not be visible
        assert!(!cpu_is_visible(&flags, 1));
        assert!(!cpu_is_visible(&flags, 33));
        assert!(!cpu_is_visible(&flags, 65));
        assert!(!cpu_is_visible(&flags, 97));
    }

    #[test]
    fn test_visibility_high_bit_extraction() {
        // Test bit 31 (highest bit in each word)
        let flags = vec![0x80000000u32; 4];

        // Objects 31, 63, 95, 127 should be visible
        assert!(cpu_is_visible(&flags, 31));
        assert!(cpu_is_visible(&flags, 63));
        assert!(cpu_is_visible(&flags, 95));
        assert!(cpu_is_visible(&flags, 127));

        // Objects 30, 62, 94, 126 should not be visible
        assert!(!cpu_is_visible(&flags, 30));
        assert!(!cpu_is_visible(&flags, 62));
        assert!(!cpu_is_visible(&flags, 94));
        assert!(!cpu_is_visible(&flags, 126));
    }

    #[test]
    fn test_visibility_alternating_bits() {
        // 0xAAAAAAAA = 10101010...10 (even bits set)
        let flags = vec![0xAAAAAAAAu32];

        // Odd indices (1,3,5,...31) should be visible
        for i in 0..32 {
            if i % 2 == 1 {
                assert!(cpu_is_visible(&flags, i), "Object {} should be visible", i);
            } else {
                assert!(
                    !cpu_is_visible(&flags, i),
                    "Object {} should not be visible",
                    i
                );
            }
        }
    }

    #[test]
    fn test_visibility_sparse_pattern() {
        // Only object 15 visible in word 0
        let flags = vec![1u32 << 15];

        for i in 0..32 {
            if i == 15 {
                assert!(cpu_is_visible(&flags, i));
            } else {
                assert!(!cpu_is_visible(&flags, i));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Category 4: Prefix Sum Indexing Tests (Extended)
    // -------------------------------------------------------------------------

    #[test]
    fn test_prefix_sum_output_index_correctness() {
        // Visibility: [1,0,1,1,0,0,1,0] = 0b01001101 = 0x4D
        // Prefix sum (exclusive): [0,1,1,2,3,3,3,4]
        let flags = vec![0x4Du32];
        let prefix_sum = vec![0, 1, 1, 2, 3, 3, 3, 4];

        // Visible objects: 0, 2, 3, 6
        // Output positions: 0, 1, 2, 3

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 8);

        assert_eq!(count, 4);
        // Object 0 -> output[0]
        assert_eq!(indices[0], 0);
        // Object 2 -> output[1]
        assert_eq!(indices[1], 2);
        // Object 3 -> output[2]
        assert_eq!(indices[2], 3);
        // Object 6 -> output[3]
        assert_eq!(indices[3], 6);
    }

    #[test]
    fn test_prefix_sum_scatter_correctness() {
        // Test that scatter writes to correct positions
        // Visibility: first visible, middle gap, last visible
        let flags = vec![0x80000001u32]; // Objects 0 and 31 visible
        let prefix_sum = vec![
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
        ];

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 32);

        assert_eq!(count, 2);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 31);
    }

    #[test]
    fn test_prefix_sum_stability_large() {
        // Test stability with larger dataset
        // All even indices visible: 0, 2, 4, ..., 62
        let flags = vec![0x55555555u32, 0x55555555u32]; // Bits 0,2,4,6,...

        // Build correct exclusive prefix sum
        let mut prefix_sum = vec![0u32; 64];
        let mut sum = 0u32;
        for i in 0..64 {
            prefix_sum[i] = sum;
            if cpu_is_visible(&flags, i) {
                sum += 1;
            }
        }

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 64);

        assert_eq!(count, 32);

        // Verify stability: indices should be in ascending order
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "Compaction should be stable"
            );
        }

        // Verify correct values
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, (i * 2) as u32, "Index {} should be {}", i, i * 2);
        }
    }

    #[test]
    fn test_prefix_sum_order_preservation() {
        // Random-ish pattern to verify order preservation
        let flags = vec![0b11010110u32]; // Objects 1,2,4,6,7 visible
        let prefix_sum = vec![0, 0, 1, 2, 2, 3, 3, 4]; // Exclusive scan

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 8);

        assert_eq!(count, 5);

        // Verify order: 1 < 2 < 4 < 6 < 7
        let expected = vec![1, 2, 4, 6, 7];
        for (i, &expected_idx) in expected.iter().enumerate() {
            assert_eq!(
                indices[i], expected_idx,
                "Position {} should contain {}",
                i, expected_idx
            );
        }
    }

    // -------------------------------------------------------------------------
    // Category 5: Count Tests (Extended)
    // -------------------------------------------------------------------------

    #[test]
    fn test_count_calculation_formula() {
        // Count = prefix_sum[n-1] + visibility[n-1]
        // Test with last element visible
        let flags = vec![0x80000000u32]; // Only object 31 visible
        let prefix_sum: Vec<u32> = (0..32).map(|_| 0).collect(); // All zeros until last

        let (_, count) = cpu_stream_compact(&flags, &prefix_sum, 32);
        // prefix_sum[31] = 0, visibility[31] = 1, count = 1
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_calculation_last_not_visible() {
        // Test with last element not visible
        let flags = vec![0x00000001u32]; // Only object 0 visible
        let mut prefix_sum = vec![0u32; 32];
        for i in 1..32 {
            prefix_sum[i] = 1; // prefix_sum[i] = 1 for i > 0
        }

        let (_, count) = cpu_stream_compact(&flags, &prefix_sum, 32);
        // prefix_sum[31] = 1, visibility[31] = 0, count = 1
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_empty_input() {
        let flags: Vec<u32> = vec![];
        let prefix_sum: Vec<u32> = vec![];

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 0);

        assert_eq!(count, 0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_count_single_visible() {
        let flags = vec![0x00000001u32]; // Only object 0 visible
        let prefix_sum = vec![0]; // Exclusive scan

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 1);

        assert_eq!(count, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_count_single_not_visible() {
        let flags = vec![0x00000000u32]; // Object 0 not visible
        let prefix_sum = vec![0]; // Exclusive scan

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 1);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_all_visible_large() {
        // 1024 objects, all visible
        let num_words = 32; // 32 words * 32 bits = 1024
        let flags = vec![0xFFFFFFFFu32; num_words];

        // Build prefix sum
        let mut prefix_sum = vec![0u32; 1024];
        for i in 1..1024 {
            prefix_sum[i] = i as u32;
        }

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 1024);

        assert_eq!(count, 1024);
        assert_eq!(indices.len(), 1024);
    }

    #[test]
    fn test_count_none_visible_large() {
        // 1024 objects, none visible
        let num_words = 32;
        let flags = vec![0x00000000u32; num_words];
        let prefix_sum = vec![0u32; 1024];

        let (indices, count) = cpu_stream_compact(&flags, &prefix_sum, 1024);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_half_visible() {
        // First half visible
        let flags = vec![0xFFFFFFFFu32, 0x00000000u32];
        let mut prefix_sum = vec![0u32; 64];
        for i in 1..64 {
            prefix_sum[i] = i.min(32) as u32;
        }

        let (_, count) = cpu_stream_compact(&flags, &prefix_sum, 64);

        assert_eq!(count, 32);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroups_edge_cases() {
        // Exactly at workgroup boundaries
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(192), 3);

        // One over boundary
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(129), 3);

        // One under boundary
        assert_eq!(workgroups_for_objects(63), 1);
        assert_eq!(workgroups_for_objects(127), 2);
    }

    #[test]
    fn test_batch_workgroups_edge_cases() {
        // Exactly at batch boundaries (256 per workgroup)
        assert_eq!(workgroups_for_objects_batch(256), 1);
        assert_eq!(workgroups_for_objects_batch(512), 2);

        // One over boundary
        assert_eq!(workgroups_for_objects_batch(257), 2);

        // One under boundary
        assert_eq!(workgroups_for_objects_batch(255), 1);
    }

    #[test]
    fn test_cpu_verify_prefix_sum_empty() {
        let (valid, error) = cpu_verify_prefix_sum(&[], &[], 0);
        assert!(valid);
        assert!(error.is_none());
    }

    #[test]
    fn test_cpu_verify_prefix_sum_single_visible() {
        let flags = vec![0x00000001u32];
        let prefix_sum = vec![0];

        let (valid, error) = cpu_verify_prefix_sum(&flags, &prefix_sum, 1);
        assert!(valid, "Error: {:?}", error);
    }

    #[test]
    fn test_cpu_verify_prefix_sum_single_not_visible() {
        let flags = vec![0x00000000u32];
        let prefix_sum = vec![0];

        let (valid, error) = cpu_verify_prefix_sum(&flags, &prefix_sum, 1);
        assert!(valid, "Error: {:?}", error);
    }

    #[test]
    fn test_cpu_compact_visible_indices_cross_word() {
        // Test compaction across word boundaries
        let flags = vec![0x80000001u32, 0x00000001u32]; // Objects 0, 31, 32

        let indices = cpu_compact_visible_indices(&flags, 64);

        assert_eq!(indices, vec![0, 31, 32]);
    }

    #[test]
    fn test_cpu_compact_preserves_order_random() {
        // Pseudo-random pattern
        let flags = vec![0xDEADBEEFu32, 0xCAFEBABEu32];

        let indices = cpu_compact_visible_indices(&flags, 64);

        // Verify ascending order
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "Indices must be in ascending order"
            );
        }
    }
}
