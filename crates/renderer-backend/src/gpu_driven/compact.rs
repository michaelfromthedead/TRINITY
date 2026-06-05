//! GPU Stream Compaction for TRINITY Engine (T-GPU-2.2 / T-GPU-2.3).
//!
//! This module provides GPU-based stream compaction using parallel prefix sum
//! (Blelloch algorithm). It efficiently removes dead elements from arrays on
//! the GPU, supporting both particle compaction and visibility buffer compaction.
//!
//! # Overview
//!
//! Stream compaction takes an input array and an array of alive flags (0 or 1),
//! and produces a compacted output containing only the elements where the
//! corresponding flag is 1.
//!
//! The algorithm runs in three phases:
//! 1. **Scan**: Compute per-block prefix sums and block totals
//! 2. **Block Sum Scan**: Scan the block totals to get global offsets
//! 3. **Scatter**: Write alive elements to their compacted positions
//!
//! # Performance
//!
//! - Work complexity: O(n)
//! - Step complexity: O(log n) per phase
//! - Target: < 0.2ms for 100K elements
//!
//! # Usage
//!
//! ```ignore
//! // Create compaction pipeline
//! let compact = CompactPipeline::new(&device);
//!
//! // Create resources for 100K elements
//! let resources = CompactResources::new(&device, 100_000);
//!
//! // Each frame: compact alive elements
//! compact.dispatch(&mut encoder, &resources, 100_000);
//!
//! // Read back compacted count
//! let count = resources.read_output_count(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum elements per single-pass compaction (one workgroup).
pub const SINGLE_PASS_MAX: u32 = 256;

/// Maximum blocks supported by simple block-sum scan.
/// For larger arrays, hierarchical scan would be needed.
pub const MAX_BLOCKS_SIMPLE: u32 = 256;

/// Maximum elements with simple block-sum scan.
pub const MAX_ELEMENTS_SIMPLE: u32 = MAX_BLOCKS_SIMPLE * WORKGROUP_SIZE; // 65536

// ---------------------------------------------------------------------------
// CompactParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for compaction parameters.
///
/// Matches the WGSL `CompactParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes total, std140/std430 compatible:
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | num_elements | 4 bytes |
/// | 4      | num_blocks   | 4 bytes |
/// | 8      | _padding     | 8 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompactParams {
    /// Total number of elements to process.
    pub num_elements: u32,
    /// Number of workgroups (blocks) in dispatch.
    pub num_blocks: u32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<CompactParams>() == 16);

impl CompactParams {
    /// Create parameters for the given element count.
    pub fn new(num_elements: u32) -> Self {
        let num_blocks = (num_elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        Self {
            num_elements,
            num_blocks,
            _padding: [0, 0],
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Check if this can use single-pass compaction.
    #[inline]
    pub fn is_single_pass(&self) -> bool {
        self.num_elements <= SINGLE_PASS_MAX
    }

    /// Check if this can use simple block-sum scan.
    #[inline]
    pub fn is_simple_scan(&self) -> bool {
        self.num_blocks <= MAX_BLOCKS_SIMPLE
    }
}

impl Default for CompactParams {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// CompactResources
// ---------------------------------------------------------------------------

/// GPU resources for stream compaction.
///
/// Contains all buffers needed for the compaction algorithm:
/// - `alive_flags`: Input alive flags (0 = dead, 1 = alive)
/// - `input_data`: Input data to compact
/// - `output_data`: Compacted output
/// - `block_sums`: Intermediate block sum storage
/// - `output_count`: Final count of alive elements
/// - `params_buffer`: Uniform buffer for parameters
pub struct CompactResources {
    /// Uniform buffer for compaction parameters.
    pub params_buffer: wgpu::Buffer,
    /// Input alive flags buffer (0 or 1 per element).
    pub alive_flags: wgpu::Buffer,
    /// Input data buffer.
    pub input_data: wgpu::Buffer,
    /// Output compacted data buffer.
    pub output_data: wgpu::Buffer,
    /// Intermediate block sums buffer.
    pub block_sums: wgpu::Buffer,
    /// Output count buffer (single u32 atomic).
    pub output_count: wgpu::Buffer,
    /// Staging buffer for reading output count back to CPU.
    pub output_count_staging: wgpu::Buffer,
    /// Maximum capacity in elements.
    pub capacity: u32,
}

impl CompactResources {
    /// Create compaction resources for the given capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of elements to support.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let num_blocks = (capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_params"),
            size: mem::size_of::<CompactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alive_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_alive_flags"),
            size: (capacity as u64) * 4, // u32 per element
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_input_data"),
            size: (capacity as u64) * 4, // u32 per element
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_output_data"),
            size: (capacity as u64) * 4, // u32 per element
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_block_sums"),
            size: (num_blocks as u64) * 4, // u32 per block
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_output_count"),
            size: 4, // single u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_output_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            alive_flags,
            input_data,
            output_data,
            block_sums,
            output_count,
            output_count_staging,
            capacity,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &CompactParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload alive flags to the GPU.
    pub fn upload_alive_flags(&self, queue: &wgpu::Queue, flags: &[u32]) {
        queue.write_buffer(&self.alive_flags, 0, bytemuck::cast_slice(flags));
    }

    /// Upload input data to the GPU.
    pub fn upload_input_data(&self, queue: &wgpu::Queue, data: &[u32]) {
        queue.write_buffer(&self.input_data, 0, bytemuck::cast_slice(data));
    }

    /// Read the output count back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_output_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_output_count"),
        });
        encoder.copy_buffer_to_buffer(&self.output_count, 0, &self.output_count_staging, 0, 4);
        queue.submit([encoder.finish()]);

        // Map staging buffer and read
        let buffer_slice = self.output_count_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        self.output_count_staging.unmap();

        count
    }
}

// ---------------------------------------------------------------------------
// CompactPipeline
// ---------------------------------------------------------------------------

/// Compute pipeline for GPU stream compaction.
///
/// Contains the three compute pipelines for the compaction algorithm:
/// 1. `scan_pipeline`: Per-block prefix sum
/// 2. `block_sum_pipeline`: Block sum scan
/// 3. `scatter_pipeline`: Scatter alive elements
///
/// Also includes a single-pass pipeline for small arrays.
pub struct CompactPipeline {
    /// Pipeline for Phase 1: per-block prefix sum.
    scan_pipeline: wgpu::ComputePipeline,
    /// Pipeline for Phase 2: block sum scan.
    block_sum_pipeline: wgpu::ComputePipeline,
    /// Pipeline for Phase 3: scatter alive elements.
    scatter_pipeline: wgpu::ComputePipeline,
    /// Pipeline for single-pass compaction (small arrays).
    single_pass_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for compaction resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CompactPipeline {
    /// Create a new compaction pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/gpu_driven/gpu_compact.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_compact_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_scan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_scan",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let block_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_block_sum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "scan_block_sums",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_scatter",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let single_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compact_single_pass_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "compact_single_pass",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            scan_pipeline,
            block_sum_pipeline,
            scatter_pipeline,
            single_pass_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &CompactResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.alive_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.input_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.output_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.output_count.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the compaction algorithm.
    ///
    /// Automatically selects single-pass or multi-pass based on element count.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing all resources.
    /// * `params` - The compaction parameters.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &CompactParams,
    ) {
        if params.is_single_pass() {
            self.dispatch_single_pass(encoder, bind_group);
        } else {
            self.dispatch_multi_pass(encoder, bind_group, params);
        }
    }

    /// Dispatch single-pass compaction for small arrays.
    fn dispatch_single_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compact_single_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.single_pass_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Dispatch multi-pass compaction for larger arrays.
    fn dispatch_multi_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &CompactParams,
    ) {
        let num_blocks = params.num_blocks();

        // Phase 1: Per-block prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_scan_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.scan_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // Phase 2: Block sum scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_block_sum_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.block_sum_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // Single workgroup for simple scan
        }

        // Phase 3: Scatter alive elements
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_scatter_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_bind_group_layout"),
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
                // binding 1: alive_flags (storage, read)
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
                // binding 2: input_data (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: output_data (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: block_sums (storage, read_write)
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
                // binding 5: output_count (storage, read_write)
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
            ],
        })
    }

    /// Create the pipeline layout.
    fn create_pipeline_layout(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation (for testing)
// ---------------------------------------------------------------------------

/// CPU reference implementation of exclusive prefix sum.
///
/// Used for testing GPU results against known-correct values.
pub fn cpu_exclusive_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0u32; input.len()];
    let mut sum = 0u32;
    for (i, &val) in input.iter().enumerate() {
        output[i] = sum;
        sum = sum.wrapping_add(val);
    }
    output
}

/// CPU reference implementation of inclusive prefix sum.
pub fn cpu_inclusive_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0u32; input.len()];
    let mut sum = 0u32;
    for (i, &val) in input.iter().enumerate() {
        sum = sum.wrapping_add(val);
        output[i] = sum;
    }
    output
}

/// CPU reference implementation of stream compaction.
///
/// Returns (compacted_data, count).
pub fn cpu_compact(input: &[u32], alive_flags: &[u32]) -> (Vec<u32>, u32) {
    let mut output = Vec::with_capacity(input.len());
    for (i, &flag) in alive_flags.iter().enumerate() {
        if flag != 0 && i < input.len() {
            output.push(input[i]);
        }
    }
    let count = output.len() as u32;
    (output, count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CompactParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_params_size() {
        assert_eq!(mem::size_of::<CompactParams>(), 16);
    }

    #[test]
    fn test_compact_params_alignment() {
        assert_eq!(mem::align_of::<CompactParams>(), 4);
    }

    #[test]
    fn test_compact_params_pod() {
        let params = CompactParams::new(1000);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_compact_params_num_blocks() {
        // Exact multiple of workgroup size
        let params = CompactParams::new(256);
        assert_eq!(params.num_blocks(), 1);

        let params = CompactParams::new(512);
        assert_eq!(params.num_blocks(), 2);

        // Non-exact multiple
        let params = CompactParams::new(257);
        assert_eq!(params.num_blocks(), 2);

        let params = CompactParams::new(1);
        assert_eq!(params.num_blocks(), 1);

        let params = CompactParams::new(1000);
        assert_eq!(params.num_blocks(), 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_compact_params_single_pass() {
        assert!(CompactParams::new(1).is_single_pass());
        assert!(CompactParams::new(256).is_single_pass());
        assert!(!CompactParams::new(257).is_single_pass());
    }

    #[test]
    fn test_compact_params_simple_scan() {
        assert!(CompactParams::new(256).is_simple_scan());
        assert!(CompactParams::new(65536).is_simple_scan()); // 256 blocks
        assert!(!CompactParams::new(65537).is_simple_scan()); // 257 blocks
    }

    // -----------------------------------------------------------------------
    // CPU reference implementation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_exclusive_prefix_sum_basic() {
        let input = [3, 1, 7, 0, 4, 1, 6, 3];
        let expected = [0, 3, 4, 11, 11, 15, 16, 22];
        let output = cpu_exclusive_prefix_sum(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cpu_exclusive_prefix_sum_empty() {
        let input: [u32; 0] = [];
        let output = cpu_exclusive_prefix_sum(&input);
        assert!(output.is_empty());
    }

    #[test]
    fn test_cpu_exclusive_prefix_sum_single() {
        let input = [42];
        let output = cpu_exclusive_prefix_sum(&input);
        assert_eq!(output, [0]);
    }

    #[test]
    fn test_cpu_exclusive_prefix_sum_all_zeros() {
        let input = [0, 0, 0, 0];
        let output = cpu_exclusive_prefix_sum(&input);
        assert_eq!(output, [0, 0, 0, 0]);
    }

    #[test]
    fn test_cpu_exclusive_prefix_sum_all_ones() {
        let input = [1, 1, 1, 1, 1, 1, 1, 1];
        let expected = [0, 1, 2, 3, 4, 5, 6, 7];
        let output = cpu_exclusive_prefix_sum(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cpu_inclusive_prefix_sum_basic() {
        let input = [3, 1, 7, 0, 4, 1, 6, 3];
        let expected = [3, 4, 11, 11, 15, 16, 22, 25];
        let output = cpu_inclusive_prefix_sum(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cpu_compact_all_alive() {
        let input = [10, 20, 30, 40, 50];
        let flags = [1, 1, 1, 1, 1];
        let (output, count) = cpu_compact(&input, &flags);
        assert_eq!(output, input);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_cpu_compact_all_dead() {
        let input = [10, 20, 30, 40, 50];
        let flags = [0, 0, 0, 0, 0];
        let (output, count) = cpu_compact(&input, &flags);
        assert!(output.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_cpu_compact_alternating() {
        let input = [10, 20, 30, 40, 50, 60];
        let flags = [1, 0, 1, 0, 1, 0];
        let (output, count) = cpu_compact(&input, &flags);
        assert_eq!(output, [10, 30, 50]);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_cpu_compact_random_pattern() {
        let input = [1, 2, 3, 4, 5, 6, 7, 8];
        let flags = [0, 1, 1, 0, 0, 1, 0, 1];
        let (output, count) = cpu_compact(&input, &flags);
        assert_eq!(output, [2, 3, 6, 8]);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_cpu_compact_first_half_alive() {
        let input: Vec<u32> = (0..256).collect();
        let flags: Vec<u32> = (0..256).map(|i| if i < 128 { 1 } else { 0 }).collect();
        let (output, count) = cpu_compact(&input, &flags);
        let expected: Vec<u32> = (0..128).collect();
        assert_eq!(output, expected);
        assert_eq!(count, 128);
    }

    // -----------------------------------------------------------------------
    // Prefix sum tests for various sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_sum_powers_of_two() {
        for power in 0..12 {
            let size = 1usize << power;
            let input: Vec<u32> = vec![1; size];
            let output = cpu_exclusive_prefix_sum(&input);

            // Verify each element
            for (i, &val) in output.iter().enumerate() {
                assert_eq!(val, i as u32, "Mismatch at index {} for size {}", i, size);
            }
        }
    }

    #[test]
    fn test_prefix_sum_non_power_of_two() {
        for &size in &[3, 7, 15, 31, 63, 127, 255, 300, 500, 1000] {
            let input: Vec<u32> = vec![1; size];
            let output = cpu_exclusive_prefix_sum(&input);

            for (i, &val) in output.iter().enumerate() {
                assert_eq!(val, i as u32, "Mismatch at index {} for size {}", i, size);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Large array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_large_array_all_alive() {
        let size = 100_000;
        let input: Vec<u32> = (0..size).collect();
        let flags: Vec<u32> = vec![1; size as usize];
        let (output, count) = cpu_compact(&input, &flags);
        assert_eq!(output.len(), size as usize);
        assert_eq!(count, size);
    }

    #[test]
    fn test_compact_large_array_half_alive() {
        let size = 100_000;
        let input: Vec<u32> = (0..size).collect();
        let flags: Vec<u32> = (0..size).map(|i| i % 2).collect();
        let (output, count) = cpu_compact(&input, &flags);
        assert_eq!(count, size / 2);
        assert_eq!(output.len(), (size / 2) as usize);

        // Verify only odd indices were kept
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, (i * 2 + 1) as u32);
        }
    }

    #[test]
    fn test_compact_1m_elements_random() {
        let size = 1_000_000u64;
        // Use a simple deterministic pattern instead of random for reproducibility
        let flags: Vec<u32> = (0..size).map(|i| (((i * 31337) % 100) < 50) as u32).collect();
        let input: Vec<u32> = (0..size as u32).collect();

        let (output, count) = cpu_compact(&input, &flags);

        // Verify count matches
        let expected_count: u32 = flags.iter().sum();
        assert_eq!(count, expected_count);
        assert_eq!(output.len(), expected_count as usize);

        // Verify output contains correct elements
        let mut output_idx = 0;
        for (i, &flag) in flags.iter().enumerate() {
            if flag != 0 {
                assert_eq!(output[output_idx], i as u32);
                output_idx += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Shader validation tests (using naga)
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_sum_shader_parses() {
        let shader_source = include_str!("../../shaders/common/prefix_sum.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("prefix_sum shader should parse without errors");

        // Verify the module has functions defined
        assert!(!module.functions.is_empty(), "Should have functions defined");
    }

    #[test]
    fn test_prefix_sum_shader_validates() {
        let shader_source = include_str!("../../shaders/common/prefix_sum.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("prefix_sum shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("prefix_sum shader should validate without errors");
    }

    #[test]
    fn test_compact_shader_parses() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("compact shader should parse without errors");

        // Verify expected entry points exist
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "compact_scan"),
            "Should have compact_scan entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "scan_block_sums"),
            "Should have scan_block_sums entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_scatter"),
            "Should have compact_scatter entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_single_pass"),
            "Should have compact_single_pass entry point"
        );
    }

    #[test]
    fn test_compact_shader_validates() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
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
    fn test_compact_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("compact shader should parse without errors");

        // Verify all compute entry points have 256x1x1 workgroup size
        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute {
                assert_eq!(
                    ep.workgroup_size,
                    [256, 1, 1],
                    "Entry point {} should have workgroup size 256x1x1",
                    ep.name
                );
            }
        }
    }

    #[test]
    fn test_compact_shader_entry_points_are_compute() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("compact shader should parse without errors");

        // Verify all entry points are compute shaders
        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }
}
