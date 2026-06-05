//! GPU Radix Sort for high-performance key-value sorting (T-GPU-2.1).
//!
//! Implements a GPU radix sort using 4-bit radix (16 buckets) with 8 passes
//! for 32-bit keys. Supports optional 32-bit payloads (values) that are
//! permuted alongside keys.
//!
//! # Use Cases
//!
//! - **Visibility buffer sorting**: Sort by material/mesh ID for draw batching
//! - **Particle depth sorting**: Back-to-front sort for correct transparency
//! - **Indirect draw sorting**: Optimize GPU-driven rendering command order
//!
//! # Algorithm
//!
//! Each radix pass consists of three phases:
//! 1. **Histogram**: Count occurrences of each 4-bit digit
//! 2. **Prefix Sum**: Compute exclusive prefix sums for scatter destinations
//! 3. **Scatter**: Write elements to their sorted positions
//!
//! The sort is stable: elements with equal keys maintain their relative order.
//!
//! # Performance
//!
//! Target: <0.5ms for 100K keys on modern GPUs.
//!
//! # Usage
//!
//! ```ignore
//! let sort = GpuRadixSort::new(&device, 100_000);
//!
//! // Upload keys to sort
//! queue.write_buffer(&sort.keys_in(), 0, bytemuck::cast_slice(&keys));
//! queue.write_buffer(&sort.values_in(), 0, bytemuck::cast_slice(&values));
//!
//! // Execute sort
//! let mut encoder = device.create_command_encoder(&Default::default());
//! sort.sort(&mut encoder, keys.len() as u32);
//! queue.submit(std::iter::once(encoder.finish()));
//!
//! // Results are in sort.keys_out() and sort.values_out()
//! ```

use std::mem;
use wgpu::{Buffer, BufferUsages, ComputePipeline, Device};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Workgroup size for histogram and scatter passes.
pub const WORKGROUP_SIZE: u32 = 256;

/// Radix bits per pass (4 bits = 16 buckets).
pub const RADIX_BITS: u32 = 4;

/// Number of radix buckets (2^4 = 16).
pub const RADIX_SIZE: u32 = 16;

/// Number of passes for 32-bit keys (32 / 4 = 8).
pub const NUM_PASSES: u32 = 8;

/// Minimum elements for GPU sort (below this, use CPU sort).
pub const MIN_GPU_ELEMENTS: u32 = 512;

/// Maximum elements for single-workgroup bitonic sort.
pub const MAX_SMALL_ELEMENTS: u32 = 256;

// ---------------------------------------------------------------------------
// SortParams (GPU uniform buffer)
// ---------------------------------------------------------------------------

/// Per-pass parameters for the radix sort shader.
///
/// # Memory Layout
///
/// 16 bytes total, std140/std430 compatible:
///
/// | Offset | Field          | Size    |
/// |--------|----------------|---------|
/// | 0      | num_elements   | 4 bytes |
/// | 4      | current_pass   | 4 bytes |
/// | 8      | num_workgroups | 4 bytes |
/// | 12     | padding        | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SortParams {
    /// Total number of elements to sort.
    pub num_elements: u32,
    /// Current pass index (0-7 for 4-bit radix).
    pub current_pass: u32,
    /// Number of workgroups for histogram/scatter.
    pub num_workgroups: u32,
    /// Padding for 16-byte alignment.
    pub padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SortParams>() == 16);

impl SortParams {
    /// Create new sort parameters.
    pub fn new(num_elements: u32, current_pass: u32, num_workgroups: u32) -> Self {
        Self {
            num_elements,
            current_pass,
            num_workgroups,
            padding: 0,
        }
    }
}

impl Default for SortParams {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

// ---------------------------------------------------------------------------
// GpuRadixSort
// ---------------------------------------------------------------------------

/// GPU-accelerated radix sort for 32-bit key-value pairs.
///
/// Manages compute pipelines, buffers, and bind groups for efficient
/// GPU-based sorting. Uses ping-pong buffers to avoid in-place issues.
///
/// # Buffer Organization
///
/// - `keys_a`, `keys_b`: Ping-pong buffers for keys
/// - `values_a`, `values_b`: Ping-pong buffers for values (payloads)
/// - `spine_buffer`: Per-workgroup histograms (num_workgroups * 16 u32s)
/// - `global_histogram_buffer`: Global digit histogram (16 u32s)
/// - `params_buffers`: Per-pass uniform parameters
pub struct GpuRadixSort {
    // Compute pipelines
    clear_buffers_pipeline: ComputePipeline,
    histogram_pipeline: ComputePipeline,
    global_prefix_sum_pipeline: ComputePipeline,
    spine_offsets_pipeline: ComputePipeline,
    scatter_pipeline: ComputePipeline,
    sort_small_pipeline: ComputePipeline,

    // Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    keys_a: Buffer,
    keys_b: Buffer,
    values_a: Buffer,
    values_b: Buffer,
    spine_buffer: Buffer,
    global_histogram_buffer: Buffer,
    /// Per-pass parameter buffers (one for each of 8 passes + 1 for small sort).
    params_buffers: Vec<Buffer>,

    /// Number of workgroups needed for max_elements.
    num_workgroups: u32,

    /// Maximum elements this sorter can handle.
    max_elements: u32,
}

impl GpuRadixSort {
    /// Create a new GPU radix sorter.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_elements` - Maximum number of elements to sort.
    ///
    /// # Panics
    ///
    /// Panics if `max_elements` is zero.
    pub fn new(device: &Device, max_elements: u32) -> Self {
        assert!(max_elements > 0, "max_elements must be > 0");

        let bind_group_layout = Self::create_bind_group_layout(device);

        // Create compute pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_radix_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!(
                "../../shaders/gpu_driven/gpu_sort.comp.wgsl"
            ).into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_radix_sort_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let clear_buffers_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_clear_buffers"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_buffers",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "histogram_pass",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let global_prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_global_prefix_sum"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "global_prefix_sum",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let spine_offsets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_spine_offsets"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "spine_offsets",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_scatter"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "scatter_pass",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sort_small_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_sort_small"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "sort_small",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Calculate number of workgroups needed
        let num_workgroups = (max_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Create buffers
        let buffer_size = (max_elements as u64) * 4; // 4 bytes per u32

        let keys_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_keys_a"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keys_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_keys_b"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let values_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_values_a"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let values_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_values_b"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Spine buffer: num_workgroups * 16 entries
        let spine_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_spine"),
            size: (num_workgroups as u64) * (RADIX_SIZE as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Global histogram buffer: 16 entries
        let global_histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_sort_global_histogram"),
            size: (RADIX_SIZE as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create per-pass params buffers (8 for radix passes + 1 for small sort)
        let params_buffers: Vec<Buffer> = (0..9)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("gpu_sort_params_{}", i)),
                    size: mem::size_of::<SortParams>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            clear_buffers_pipeline,
            histogram_pipeline,
            global_prefix_sum_pipeline,
            spine_offsets_pipeline,
            scatter_pipeline,
            sort_small_pipeline,
            bind_group_layout,
            keys_a,
            keys_b,
            values_a,
            values_b,
            spine_buffer,
            global_histogram_buffer,
            params_buffers,
            num_workgroups,
            max_elements,
        }
    }

    /// Create the bind group layout for sort operations.
    fn create_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_sort_bind_group_layout"),
            entries: &[
                // binding(0): params uniform
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
                // binding(1): keys_in (read-only storage)
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
                // binding(2): keys_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding(3): values_in (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding(4): values_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding(5): spine buffer (atomic storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding(6): global_histogram (atomic storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

    /// Create a bind group for a forward pass (A -> B) with a specific params buffer.
    fn create_bind_group_forward(&self, device: &Device, params_buffer: &Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_sort_bind_group_forward"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.keys_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.values_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.values_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.spine_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.global_histogram_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a bind group for a reverse pass (B -> A) with a specific params buffer.
    fn create_bind_group_reverse(&self, device: &Device, params_buffer: &Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_sort_bind_group_reverse"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.keys_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.values_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.values_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.spine_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.global_histogram_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Sort the elements in the input buffers.
    ///
    /// After sorting, results are in the appropriate output buffer based on
    /// the number of passes (8 passes = even, so results in buffer A).
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for bind group creation).
    /// * `queue` - The wgpu queue (for params upload).
    /// * `encoder` - Command encoder to record sort commands.
    /// * `num_elements` - Number of elements to sort.
    ///
    /// # Panics
    ///
    /// Panics if `num_elements` exceeds `max_elements`.
    pub fn sort(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        num_elements: u32,
    ) {
        assert!(
            num_elements <= self.max_elements,
            "num_elements ({}) exceeds max_elements ({})",
            num_elements,
            self.max_elements
        );

        if num_elements == 0 {
            return;
        }

        // For small arrays, use the bitonic sort
        if num_elements <= MAX_SMALL_ELEMENTS {
            self.sort_small(device, queue, encoder, num_elements);
            return;
        }

        // Full radix sort for larger arrays
        let workgroups = (num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let spine_clear_workgroups = (workgroups * RADIX_SIZE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Write params for each pass to separate buffers before creating bind groups
        for pass in 0..NUM_PASSES {
            let params = SortParams::new(num_elements, pass, workgroups);
            queue.write_buffer(&self.params_buffers[pass as usize], 0, bytemuck::bytes_of(&params));
        }

        // Create per-pass bind groups with the correct params buffer
        let bind_groups: Vec<wgpu::BindGroup> = (0..NUM_PASSES)
            .map(|pass| {
                let params_buffer = &self.params_buffers[pass as usize];
                if pass % 2 == 0 {
                    self.create_bind_group_forward(device, params_buffer)
                } else {
                    self.create_bind_group_reverse(device, params_buffer)
                }
            })
            .collect();

        // Execute 8 passes (4-bit radix, 32-bit keys)
        for pass in 0..NUM_PASSES {
            let bind_group = &bind_groups[pass as usize];

            // Phase 0: Clear spine and global histogram
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("gpu_sort_clear_buffers_pass_{}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.clear_buffers_pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(spine_clear_workgroups, 1, 1);
            }

            // Phase 1: Build per-workgroup histogram into spine + global histogram
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("gpu_sort_histogram_pass_{}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.histogram_pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Phase 2: Compute global digit prefix sums
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("gpu_sort_global_prefix_sum_pass_{}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.global_prefix_sum_pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Phase 3: Compute per-workgroup offsets in spine
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("gpu_sort_spine_offsets_pass_{}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.spine_offsets_pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Phase 4: Scatter to sorted positions
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("gpu_sort_scatter_pass_{}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.scatter_pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        // After 8 passes (even number), results are back in buffer A
    }

    /// Sort small arrays using workgroup-local bitonic sort.
    fn sort_small(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        num_elements: u32,
    ) {
        // Use the 9th params buffer (index 8) for small sort
        let params = SortParams::new(num_elements, 0, 1); // 1 workgroup for small sort
        queue.write_buffer(&self.params_buffers[8], 0, bytemuck::bytes_of(&params));

        // For small sort, we read from A and write to B
        let bind_group_forward = self.create_bind_group_forward(device, &self.params_buffers[8]);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_sort_small_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sort_small_pipeline);
            cpass.set_bind_group(0, &bind_group_forward, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Copy B back to A so results are in the expected output buffer
        encoder.copy_buffer_to_buffer(
            &self.keys_b,
            0,
            &self.keys_a,
            0,
            (num_elements as u64) * 4,
        );
        encoder.copy_buffer_to_buffer(
            &self.values_b,
            0,
            &self.values_a,
            0,
            (num_elements as u64) * 4,
        );
    }

    // -- Buffer accessors --

    /// Get the primary keys input buffer.
    ///
    /// Upload keys here before sorting.
    #[inline]
    pub fn keys_in(&self) -> &Buffer {
        &self.keys_a
    }

    /// Get the primary values input buffer.
    ///
    /// Upload values (payloads) here before sorting.
    #[inline]
    pub fn values_in(&self) -> &Buffer {
        &self.values_a
    }

    /// Get the keys output buffer after sorting.
    ///
    /// With 8 passes (even), results end up back in buffer A.
    #[inline]
    pub fn keys_out(&self) -> &Buffer {
        &self.keys_a
    }

    /// Get the values output buffer after sorting.
    ///
    /// With 8 passes (even), results end up back in buffer A.
    #[inline]
    pub fn values_out(&self) -> &Buffer {
        &self.values_a
    }

    /// Get the maximum elements this sorter can handle.
    #[inline]
    pub fn max_elements(&self) -> u32 {
        self.max_elements
    }

    /// Get the bind group layout for external use.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

// ---------------------------------------------------------------------------
// SortResult
// ---------------------------------------------------------------------------

/// Result of a GPU sort operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortResult {
    /// Sort completed successfully.
    Success,
    /// Input was empty, no sort performed.
    Empty,
    /// Input exceeded maximum capacity.
    CapacityExceeded,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference implementation of radix sort for verification.
    fn cpu_radix_sort(keys: &mut [u32], values: &mut [u32]) {
        let n = keys.len();
        if n <= 1 {
            return;
        }

        let mut out_keys = vec![0u32; n];
        let mut out_values = vec![0u32; n];

        // 8 passes for 32-bit keys with 4-bit radix
        for pass in 0..8u32 {
            let shift = pass * 4;
            let mut counts = [0usize; 16];

            // Count occurrences of each digit
            for &k in keys.iter() {
                let digit = ((k >> shift) & 0xF) as usize;
                counts[digit] += 1;
            }

            // Exclusive prefix sum
            let mut total = 0usize;
            for c in counts.iter_mut() {
                let old = *c;
                *c = total;
                total += old;
            }

            // Scatter to output (stable - preserves relative order)
            for i in 0..n {
                let digit = ((keys[i] >> shift) & 0xF) as usize;
                let dest = counts[digit];
                counts[digit] += 1;
                out_keys[dest] = keys[i];
                out_values[dest] = values[i];
            }

            // Swap buffers
            keys.copy_from_slice(&out_keys);
            values.copy_from_slice(&out_values);
        }
    }

    /// Test CPU reference sort is correct.
    #[test]
    fn test_cpu_radix_sort() {
        let mut keys = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let mut values: Vec<u32> = (0..10).collect();

        cpu_radix_sort(&mut keys, &mut values);

        // Verify sorted
        for i in 1..keys.len() {
            assert!(keys[i - 1] <= keys[i], "CPU sort failed at {}: {} > {}", i, keys[i-1], keys[i]);
        }

        // Verify values follow keys
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    /// Test that SortParams has the expected size and layout.
    #[test]
    fn test_sort_params_size() {
        assert_eq!(mem::size_of::<SortParams>(), 16);
        assert_eq!(mem::align_of::<SortParams>(), 4);
    }

    /// Test SortParams construction.
    #[test]
    fn test_sort_params_new() {
        let params = SortParams::new(1000, 5, 4);
        assert_eq!(params.num_elements, 1000);
        assert_eq!(params.current_pass, 5);
        assert_eq!(params.num_workgroups, 4);
        assert_eq!(params.padding, 0);
    }

    /// Test constants are correct.
    #[test]
    fn test_constants() {
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(RADIX_BITS, 4);
        assert_eq!(RADIX_SIZE, 16);
        assert_eq!(NUM_PASSES, 8);
        assert_eq!(RADIX_BITS * NUM_PASSES, 32); // Full 32-bit coverage
    }

    /// Test radix digit extraction.
    #[test]
    fn test_extract_digit() {
        // Helper to simulate the WGSL extract_digit function
        fn extract_digit(key: u32, pass: u32) -> u32 {
            (key >> (pass * RADIX_BITS)) & (RADIX_SIZE - 1)
        }

        // Test key 0xDEADBEEF
        let key = 0xDEADBEEF_u32;
        assert_eq!(extract_digit(key, 0), 0xF); // bits 0-3
        assert_eq!(extract_digit(key, 1), 0xE); // bits 4-7
        assert_eq!(extract_digit(key, 2), 0xE); // bits 8-11
        assert_eq!(extract_digit(key, 3), 0xB); // bits 12-15
        assert_eq!(extract_digit(key, 4), 0xD); // bits 16-19
        assert_eq!(extract_digit(key, 5), 0xA); // bits 20-23
        assert_eq!(extract_digit(key, 6), 0xE); // bits 24-27
        assert_eq!(extract_digit(key, 7), 0xD); // bits 28-31
    }

    /// Test sorting small arrays on GPU (requires actual GPU).
    #[test]
    fn test_radix_sort_small() {
        // This test requires a GPU and will be skipped in CI without one
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        // Create sorter
        let sorter = GpuRadixSort::new(&device, 256);

        // Test data
        let keys: Vec<u32> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 15, 12, 11, 14, 13, 10];
        let values: Vec<u32> = (0..16).collect();

        // Upload data
        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));

        // Execute sort
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);

        // Create readback buffer
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (keys.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            sorter.keys_out(),
            0,
            &readback,
            0,
            (keys.len() * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and verify
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = slice.get_mapped_range();
        let sorted_keys: &[u32] = bytemuck::cast_slice(&data);

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i - 1] <= sorted_keys[i],
                "Keys not sorted at index {}: {} > {}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            );
        }
    }

    /// Test sorting 512 elements (2 workgroups, minimal radix sort).
    #[test]
    fn test_radix_sort_512() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        const NUM_ELEMENTS: usize = 512;

        let sorter = GpuRadixSort::new(&device, NUM_ELEMENTS as u32);

        // Simple descending sequence: 511, 510, 509, ..., 1, 0
        let keys: Vec<u32> = (0..NUM_ELEMENTS as u32).rev().collect();
        let values: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();

        // Compute expected output using CPU sort
        let mut expected_keys = keys.clone();
        let mut expected_values = values.clone();
        cpu_radix_sort(&mut expected_keys, &mut expected_values);

        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (keys.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            sorter.keys_out(),
            0,
            &readback,
            0,
            (keys.len() * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = slice.get_mapped_range();
        let sorted_keys: &[u32] = bytemuck::cast_slice(&data);

        eprintln!("Input keys (first 20): {:?}", &keys[0..20]);
        eprintln!("GPU output (first 20): {:?}", &sorted_keys[0..20]);
        eprintln!("Expected (first 20): {:?}", &expected_keys[0..20]);

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i - 1] <= sorted_keys[i],
                "Keys not sorted at index {}: {} > {}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            );
        }
    }

    /// Test sorting medium arrays (1K elements).
    /// Test sorting medium arrays (1K elements).
    #[test]
    fn test_radix_sort_medium() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        const NUM_ELEMENTS: usize = 512;

        // Create sorter
        let sorter = GpuRadixSort::new(&device, NUM_ELEMENTS as u32);

        // Generate simple descending data: [511, 510, ..., 1, 0]
        let keys: Vec<u32> = (0..NUM_ELEMENTS).map(|i| (NUM_ELEMENTS - 1 - i) as u32).collect();
        let values: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();

        eprintln!("Input keys first 20: {:?}", &keys[0..20]);
        eprintln!("Input keys last 20: {:?}", &keys[NUM_ELEMENTS-20..]);

        // Upload data
        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));

        // Execute sort
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);

        // Create readback buffer
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (keys.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            sorter.keys_out(),
            0,
            &readback,
            0,
            (keys.len() * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and verify
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = slice.get_mapped_range();
        let sorted_keys: &[u32] = bytemuck::cast_slice(&data);

        // Debug: print first 20 values
        eprintln!("Output keys first 20: {:?}", &sorted_keys[0..20.min(sorted_keys.len())]);
        eprintln!("Expected: [0, 1, 2, 3, 4, ...]");

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i - 1] <= sorted_keys[i],
                "Keys not sorted at index {}: {} > {}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            );
        }
    }

    /// Test sorting large arrays (100K elements).
    /// Test sorting large arrays (100K elements).
    #[test]
    fn test_radix_sort_large() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        const NUM_ELEMENTS: usize = 100_000;

        // Create sorter
        let sorter = GpuRadixSort::new(&device, NUM_ELEMENTS as u32);

        // Generate random-ish data
        let keys: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|i| ((i * 7919 + 104729) % 1_000_000) as u32)
            .collect();
        let values: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();

        // Upload data
        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));

        // Execute sort
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);

        // Create readback buffer
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (keys.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            sorter.keys_out(),
            0,
            &readback,
            0,
            (keys.len() * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and verify
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = slice.get_mapped_range();
        let sorted_keys: &[u32] = bytemuck::cast_slice(&data);

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i - 1] <= sorted_keys[i],
                "Keys not sorted at index {}: {} > {}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            );
        }
    }

    /// Test that stability is preserved (equal keys maintain relative order).
    #[test]
    fn test_radix_sort_preserves_stability() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        const NUM_ELEMENTS: usize = 1024;

        // Create sorter
        let sorter = GpuRadixSort::new(&device, NUM_ELEMENTS as u32);

        // Create data with many duplicate keys
        // Each key appears multiple times, values track original order
        let keys: Vec<u32> = (0..NUM_ELEMENTS).map(|i| (i % 16) as u32).collect();
        let values: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();

        // Upload data
        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));

        // Execute sort
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);

        // Create readback buffers
        let keys_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keys_readback"),
            size: (keys.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let values_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values_readback"),
            size: (values.len() * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            sorter.keys_out(),
            0,
            &keys_readback,
            0,
            (keys.len() * 4) as u64,
        );

        encoder.copy_buffer_to_buffer(
            sorter.values_out(),
            0,
            &values_readback,
            0,
            (values.len() * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map keys
        let keys_slice = keys_readback.slice(..);
        let values_slice = values_readback.slice(..);

        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();

        keys_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx1.send(result).unwrap();
        });
        values_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx1.recv().unwrap().expect("Failed to map keys buffer");
        rx2.recv().unwrap().expect("Failed to map values buffer");

        let keys_data = keys_slice.get_mapped_range();
        let values_data = values_slice.get_mapped_range();

        let sorted_keys: &[u32] = bytemuck::cast_slice(&keys_data);
        let sorted_values: &[u32] = bytemuck::cast_slice(&values_data);

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i - 1] <= sorted_keys[i],
                "Keys not sorted at index {}: {} > {}",
                i,
                sorted_keys[i - 1],
                sorted_keys[i]
            );
        }

        // Verify stability: for equal keys, values should be in increasing order
        // (since original values were 0, 1, 2, ... which is the original index)
        let mut prev_key = sorted_keys[0];
        let mut prev_value = sorted_values[0];

        for i in 1..sorted_keys.len() {
            let key = sorted_keys[i];
            let value = sorted_values[i];

            if key == prev_key {
                // Equal keys should have values in increasing order (stability)
                assert!(
                    prev_value < value,
                    "Stability violated at index {}: key={}, prev_value={}, value={}",
                    i,
                    key,
                    prev_value,
                    value
                );
            }

            prev_key = key;
            prev_value = value;
        }
    }

    /// Benchmark test to measure sorting time for 100K elements.
    #[test]
    fn bench_radix_sort_100k() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("No GPU adapter found, skipping benchmark");
                return;
            }
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bench_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        const NUM_ELEMENTS: usize = 100_000;
        const NUM_ITERATIONS: usize = 10;

        let sorter = GpuRadixSort::new(&device, NUM_ELEMENTS as u32);

        // Generate random-ish data
        let keys: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|i| ((i * 7919 + 104729) % 1_000_000) as u32)
            .collect();
        let values: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();

        // Warmup run
        queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
        queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));
        let mut encoder = device.create_command_encoder(&Default::default());
        sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Timed iterations
        let start = std::time::Instant::now();
        for _ in 0..NUM_ITERATIONS {
            queue.write_buffer(sorter.keys_in(), 0, bytemuck::cast_slice(&keys));
            queue.write_buffer(sorter.values_in(), 0, bytemuck::cast_slice(&values));
            let mut encoder = device.create_command_encoder(&Default::default());
            sorter.sort(&device, &queue, &mut encoder, keys.len() as u32);
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / NUM_ITERATIONS as f64;

        eprintln!("GPU Radix Sort benchmark:");
        eprintln!("  Elements: {}", NUM_ELEMENTS);
        eprintln!("  Iterations: {}", NUM_ITERATIONS);
        eprintln!("  Average time: {:.3}ms", avg_ms);
        eprintln!("  Target: <0.5ms");

        // Note: This includes CPU overhead (upload, submit, poll).
        // Actual GPU time would need timestamp queries.
    }
}
