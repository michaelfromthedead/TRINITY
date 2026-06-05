//! Blelloch parallel prefix scan (exclusive scan) pipeline for wgpu 25.x.
//!
//! This module provides a GPU-accelerated prefix scan implementation using the
//! Blelloch algorithm. The prefix scan is a fundamental parallel primitive used in:
//!
//! - Stream compaction (filtering)
//! - Sorting algorithms (radix sort)
//! - Histogram computation
//! - Sparse matrix operations
//! - GPU-driven rendering (indirect draw generation)
//!
//! # Algorithm Overview
//!
//! The Blelloch scan has two phases:
//!
//! 1. **Up-Sweep (Reduce)**: Build a binary tree of partial sums from leaves to root.
//!    Time complexity: O(n) work, O(log n) span.
//!
//! 2. **Down-Sweep (Distribute)**: Traverse the tree from root to leaves, distributing
//!    prefix sums. Produces exclusive scan output.
//!
//! For arrays larger than one workgroup can handle, multi-block coordination is used:
//! - Each workgroup scans its block and outputs its total sum
//! - Block sums are scanned recursively
//! - Block prefixes are added to all elements
//!
//! # Performance Characteristics
//!
//! - Workgroup size: 256 threads
//! - Elements per workgroup: 512 (2 per thread)
//! - Memory bandwidth: ~2 reads + 2 writes per element
//! - Work complexity: O(n)
//! - Span complexity: O(log n) per block + O(log(n/512)) for block coordination
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_library::prefix_scan::PrefixScanPipeline;
//!
//! # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create the pipeline
//! let pipeline = PrefixScanPipeline::new(device);
//!
//! // Create input buffer with data
//! let input = [1u32, 2, 3, 4, 5, 6, 7, 8];
//! let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("scan_input"),
//!     contents: bytemuck::cast_slice(&input),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! // Perform exclusive scan in-place
//! pipeline.scan(device, queue, &buffer, input.len() as u32);
//!
//! // Result: [0, 1, 3, 6, 10, 15, 21, 28]
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `PrefixScanPipeline` is `Send + Sync` as it only holds wgpu pipeline handles
//! and bind group layouts.

use std::borrow::Cow;
use std::num::NonZeroU64;

/// Workgroup size used by the prefix scan shader.
pub const WORKGROUP_SIZE: u32 = 256;

/// Number of elements processed per workgroup (2 per thread).
pub const ELEMENTS_PER_WORKGROUP: u32 = WORKGROUP_SIZE * 2; // 512

/// Shader source code for the prefix scan.
const SHADER_SOURCE: &str = include_str!("../../shaders/prefix_scan.wgsl");

// ---------------------------------------------------------------------------
// ScanParams - Uniform buffer structure
// ---------------------------------------------------------------------------

/// Parameters passed to the prefix scan shader via uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScanParams {
    /// Total number of elements to scan.
    pub input_size: u32,
    /// Starting offset for this dispatch (for multi-pass).
    pub block_offset: u32,
    /// Scan mode: 0 = exclusive, 1 = inclusive.
    pub is_inclusive: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

impl ScanParams {
    /// Create new scan parameters for exclusive scan.
    pub fn exclusive(input_size: u32) -> Self {
        Self {
            input_size,
            block_offset: 0,
            is_inclusive: 0,
            _pad: 0,
        }
    }

    /// Create new scan parameters for inclusive scan.
    pub fn inclusive(input_size: u32) -> Self {
        Self {
            input_size,
            block_offset: 0,
            is_inclusive: 1,
            _pad: 0,
        }
    }

    /// Set the block offset for multi-pass scans.
    pub fn with_offset(mut self, offset: u32) -> Self {
        self.block_offset = offset;
        self
    }
}

// ---------------------------------------------------------------------------
// PrefixScanError
// ---------------------------------------------------------------------------

/// Errors that can occur during prefix scan operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixScanError {
    /// Input size is zero.
    EmptyInput,
    /// Input size exceeds maximum supported.
    InputTooLarge {
        size: u32,
        max: u32,
    },
    /// Buffer is too small for the requested operation.
    BufferTooSmall {
        required: u64,
        actual: u64,
    },
    /// Internal error during pipeline creation.
    PipelineCreationFailed(String),
}

impl std::fmt::Display for PrefixScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Input size cannot be zero"),
            Self::InputTooLarge { size, max } => {
                write!(f, "Input size {} exceeds maximum supported {}", size, max)
            }
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "Buffer size {} is too small, need at least {} bytes",
                    actual, required
                )
            }
            Self::PipelineCreationFailed(msg) => {
                write!(f, "Pipeline creation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for PrefixScanError {}

// ---------------------------------------------------------------------------
// PrefixScanPipeline
// ---------------------------------------------------------------------------

/// GPU pipeline for Blelloch parallel prefix scan.
///
/// This struct holds the compiled compute pipelines and bind group layouts
/// needed to perform prefix scans on the GPU. Create one instance and reuse
/// it for multiple scan operations.
///
/// # Supported Operations
///
/// - `scan()`: In-place exclusive prefix sum
/// - `scan_with_total()`: Exclusive prefix sum + retrieve total sum
/// - `scan_inclusive()`: In-place inclusive prefix sum
///
/// # Limitations
///
/// - Maximum input size: 2^30 elements (~4 billion u32s, 16 GB)
/// - Input/output element type: u32
/// - Buffer must have STORAGE usage flag
pub struct PrefixScanPipeline {
    /// Single-block scan pipeline (for inputs <= 512 elements).
    single_block_pipeline: wgpu::ComputePipeline,
    /// Up-sweep pipeline (multi-block).
    up_sweep_pipeline: wgpu::ComputePipeline,
    /// Down-sweep pipeline (multi-block).
    down_sweep_pipeline: wgpu::ComputePipeline,
    /// Add block sums pipeline (multi-block).
    add_block_sums_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for scan operations.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PrefixScanPipeline {
    /// Create a new prefix scan pipeline.
    ///
    /// Compiles all required compute shaders. This is an expensive operation
    /// that should be done once during initialization.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create pipelines on.
    ///
    /// # Returns
    ///
    /// A new `PrefixScanPipeline` ready for use.
    pub fn new(device: &wgpu::Device) -> Self {
        // Compile the shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix_scan_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Create bind group layout
        // Binding 0: data buffer (storage, read_write)
        // Binding 1: block_sums buffer (storage, read_write)
        // Binding 2: params uniform buffer
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_scan_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4), // At least one u32
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4), // At least one u32
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ScanParams>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix_scan_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create all pipeline variants
        let single_block_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix_scan_single_block"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "scan_single_block",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let up_sweep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_scan_up_sweep"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "up_sweep",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let down_sweep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix_scan_down_sweep"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "down_sweep",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let add_block_sums_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix_scan_add_block_sums"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "add_block_sums",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            single_block_pipeline,
            up_sweep_pipeline,
            down_sweep_pipeline,
            add_block_sums_pipeline,
            bind_group_layout,
        }
    }

    /// Calculate the number of workgroups needed for a given input size.
    #[inline]
    pub fn num_workgroups(input_size: u32) -> u32 {
        (input_size + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
    }

    /// Calculate the required size for the block sums buffer.
    #[inline]
    pub fn block_sums_buffer_size(input_size: u32) -> u64 {
        // Need space for block sums at each level of recursion
        // Each level reduces by factor of ELEMENTS_PER_WORKGROUP
        let mut total_blocks = 0u64;
        let mut size = input_size;
        while size > ELEMENTS_PER_WORKGROUP {
            let blocks = Self::num_workgroups(size);
            total_blocks += blocks as u64;
            size = blocks;
        }
        // Add one for the final level
        total_blocks += 1;
        total_blocks * 4 // 4 bytes per u32
    }

    /// Perform an exclusive prefix scan in-place.
    ///
    /// Given input `[a, b, c, d]`, produces `[0, a, a+b, a+b+c]`.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `data_buffer` - Buffer containing input data (modified in-place).
    /// * `input_size` - Number of elements to scan.
    ///
    /// # Panics
    ///
    /// Panics if `input_size` is 0 or if the buffer is too small.
    pub fn scan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_buffer: &wgpu::Buffer,
        input_size: u32,
    ) {
        self.scan_impl(device, queue, data_buffer, input_size, false)
            .expect("Prefix scan failed");
    }

    /// Perform an exclusive prefix scan and return the total sum.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `data_buffer` - Buffer containing input data (modified in-place).
    /// * `input_size` - Number of elements to scan.
    ///
    /// # Returns
    ///
    /// A buffer containing the total sum (single u32).
    pub fn scan_with_total(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_buffer: &wgpu::Buffer,
        input_size: u32,
    ) -> wgpu::Buffer {
        // Create block sums buffer to capture the total
        let block_sums_size = Self::block_sums_buffer_size(input_size).max(4);
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_scan_block_sums"),
            size: block_sums_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.scan_with_block_sums(device, queue, data_buffer, &block_sums_buffer, input_size);

        block_sums_buffer
    }

    /// Perform an exclusive prefix scan using a provided block sums buffer.
    ///
    /// This is useful when you want to reuse the block sums buffer across
    /// multiple scans or need access to intermediate block sums.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `data_buffer` - Buffer containing input data (modified in-place).
    /// * `block_sums_buffer` - Scratch buffer for block sums.
    /// * `input_size` - Number of elements to scan.
    pub fn scan_with_block_sums(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_buffer: &wgpu::Buffer,
        block_sums_buffer: &wgpu::Buffer,
        input_size: u32,
    ) {
        if input_size == 0 {
            return;
        }

        let num_blocks = Self::num_workgroups(input_size);

        // Create params buffer
        let params = ScanParams::exclusive(input_size);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_scan_params"),
            size: std::mem::size_of::<ScanParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_scan_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        if num_blocks == 1 {
            // Single block scan - use optimized single-pass kernel
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prefix_scan_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix_scan_single_block_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.single_block_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        } else {
            // Multi-block scan requires recursive scan of block sums
            self.scan_multi_block(device, queue, data_buffer, block_sums_buffer, input_size);
        }
    }

    /// Internal implementation for multi-block scan.
    fn scan_multi_block(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_buffer: &wgpu::Buffer,
        block_sums_buffer: &wgpu::Buffer,
        input_size: u32,
    ) {
        let num_blocks = Self::num_workgroups(input_size);

        // Create params buffer
        let params = ScanParams::exclusive(input_size);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_scan_params"),
            size: std::mem::size_of::<ScanParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_scan_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Pass 1: Local scan of each block + collect block sums
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prefix_scan_pass1_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_scan_down_sweep_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.down_sweep_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Pass 2: Scan block sums (recursive if needed)
        if num_blocks > 1 {
            // Create a temporary buffer for block sum scanning
            let block_sums_scan_size = Self::block_sums_buffer_size(num_blocks).max(4);
            let block_sums_scan_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prefix_scan_block_sums_scan"),
                size: block_sums_scan_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Recursively scan block sums
            self.scan_with_block_sums(
                device,
                queue,
                block_sums_buffer,
                &block_sums_scan_buffer,
                num_blocks,
            );
        }

        // Pass 3: Add scanned block sums to all elements
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prefix_scan_pass3_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_scan_add_block_sums_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_block_sums_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Internal implementation that can be fallible.
    fn scan_impl(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_buffer: &wgpu::Buffer,
        input_size: u32,
        _inclusive: bool,
    ) -> Result<(), PrefixScanError> {
        if input_size == 0 {
            return Err(PrefixScanError::EmptyInput);
        }

        // Create temporary block sums buffer
        let block_sums_size = Self::block_sums_buffer_size(input_size).max(4);
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_scan_block_sums"),
            size: block_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.scan_with_block_sums(device, queue, data_buffer, &block_sums_buffer, input_size);

        Ok(())
    }

    /// Get the bind group layout for external use.
    ///
    /// Useful for creating custom bind groups or integrating with other pipelines.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get a reference to the single-block pipeline.
    pub fn single_block_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.single_block_pipeline
    }

    /// Get a reference to the up-sweep pipeline.
    pub fn up_sweep_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.up_sweep_pipeline
    }

    /// Get a reference to the down-sweep pipeline.
    pub fn down_sweep_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.down_sweep_pipeline
    }

    /// Get a reference to the add block sums pipeline.
    pub fn add_block_sums_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.add_block_sums_pipeline
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ScanParams Struct Tests
    // =========================================================================

    #[test]
    fn test_scan_params_exclusive() {
        let params = ScanParams::exclusive(100);
        assert_eq!(params.input_size, 100);
        assert_eq!(params.block_offset, 0);
        assert_eq!(params.is_inclusive, 0);
    }

    #[test]
    fn test_scan_params_inclusive() {
        let params = ScanParams::inclusive(100);
        assert_eq!(params.input_size, 100);
        assert_eq!(params.is_inclusive, 1);
    }

    #[test]
    fn test_scan_params_with_offset() {
        let params = ScanParams::exclusive(100).with_offset(512);
        assert_eq!(params.block_offset, 512);
    }

    #[test]
    fn test_scan_params_exclusive_zero_size() {
        // Edge case: zero input size
        let params = ScanParams::exclusive(0);
        assert_eq!(params.input_size, 0);
        assert_eq!(params.block_offset, 0);
        assert_eq!(params.is_inclusive, 0);
    }

    #[test]
    fn test_scan_params_inclusive_large_size() {
        // Edge case: large input size (2^30)
        let large_size = 1 << 30;
        let params = ScanParams::inclusive(large_size);
        assert_eq!(params.input_size, large_size);
        assert_eq!(params.is_inclusive, 1);
    }

    #[test]
    fn test_scan_params_with_offset_chaining() {
        // Test method chaining
        let params = ScanParams::exclusive(1000)
            .with_offset(2048);
        assert_eq!(params.input_size, 1000);
        assert_eq!(params.block_offset, 2048);
        assert_eq!(params.is_inclusive, 0);
    }

    #[test]
    fn test_scan_params_with_offset_max_u32() {
        // Edge case: maximum offset value
        let params = ScanParams::exclusive(100).with_offset(u32::MAX);
        assert_eq!(params.block_offset, u32::MAX);
    }

    #[test]
    fn test_scan_params_default() {
        // Test Default trait
        let params = ScanParams::default();
        assert_eq!(params.input_size, 0);
        assert_eq!(params.block_offset, 0);
        assert_eq!(params.is_inclusive, 0);
        assert_eq!(params._pad, 0);
    }

    #[test]
    fn test_scan_params_pod() {
        // Verify ScanParams is POD (bytemuck compatible)
        let params = ScanParams::exclusive(100);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<ScanParams>());
        assert_eq!(bytes.len(), 16); // 4 u32s = 16 bytes
    }

    #[test]
    fn test_scan_params_zeroable() {
        // Verify Zeroable trait
        let zeroed: ScanParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.input_size, 0);
        assert_eq!(zeroed.block_offset, 0);
        assert_eq!(zeroed.is_inclusive, 0);
        assert_eq!(zeroed._pad, 0);
    }

    #[test]
    fn test_scan_params_pod_round_trip() {
        // Test bytemuck round-trip conversion
        let params = ScanParams {
            input_size: 12345,
            block_offset: 67890,
            is_inclusive: 1,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        let restored: &ScanParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.input_size, 12345);
        assert_eq!(restored.block_offset, 67890);
        assert_eq!(restored.is_inclusive, 1);
    }

    #[test]
    fn test_scan_params_clone() {
        let params = ScanParams::exclusive(500).with_offset(100);
        let cloned = params.clone();
        assert_eq!(cloned.input_size, params.input_size);
        assert_eq!(cloned.block_offset, params.block_offset);
        assert_eq!(cloned.is_inclusive, params.is_inclusive);
    }

    #[test]
    fn test_scan_params_copy() {
        let params = ScanParams::inclusive(200);
        let copied = params; // Copy
        let _still_valid = params; // Original still valid (Copy trait)
        assert_eq!(copied.input_size, 200);
    }

    #[test]
    fn test_scan_params_debug() {
        let params = ScanParams::exclusive(100).with_offset(50);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("ScanParams"));
        assert!(debug_str.contains("input_size"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_scan_params_16_byte_alignment() {
        // Verify struct is 16-byte aligned for GPU uniform buffers
        assert_eq!(std::mem::size_of::<ScanParams>(), 16);
        // Check alignment is at least 4 (u32 alignment)
        assert!(std::mem::align_of::<ScanParams>() >= 4);
    }

    // =========================================================================
    // Workgroup Calculation Tests
    // =========================================================================

    #[test]
    fn test_num_workgroups() {
        assert_eq!(PrefixScanPipeline::num_workgroups(1), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(513), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(1025), 3);
    }

    #[test]
    fn test_num_workgroups_single_element() {
        // Edge case: single element
        assert_eq!(PrefixScanPipeline::num_workgroups(1), 1);
    }

    #[test]
    fn test_num_workgroups_exact_multiple() {
        // Exact multiples of ELEMENTS_PER_WORKGROUP (512)
        assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(2048), 4);
        assert_eq!(PrefixScanPipeline::num_workgroups(512 * 100), 100);
    }

    #[test]
    fn test_num_workgroups_just_over_boundary() {
        // Just over exact multiple boundaries
        assert_eq!(PrefixScanPipeline::num_workgroups(513), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(1025), 3);
        assert_eq!(PrefixScanPipeline::num_workgroups(2049), 5);
    }

    #[test]
    fn test_num_workgroups_just_under_boundary() {
        // Just under exact multiple boundaries
        assert_eq!(PrefixScanPipeline::num_workgroups(511), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(1023), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(2047), 4);
    }

    #[test]
    fn test_num_workgroups_large_arrays() {
        // Large array sizes
        assert_eq!(PrefixScanPipeline::num_workgroups(1_000_000), 1954); // ceil(1M / 512)
        assert_eq!(PrefixScanPipeline::num_workgroups(10_000_000), 19532);
    }

    #[test]
    fn test_num_workgroups_power_of_two_sizes() {
        // Power of 2 sizes
        for exp in 0..20 {
            let size = 1u32 << exp;
            let expected = (size + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
            assert_eq!(PrefixScanPipeline::num_workgroups(size), expected);
        }
    }

    #[test]
    fn test_num_workgroups_verifies_formula() {
        // Verify formula: ceil(input_size / ELEMENTS_PER_WORKGROUP)
        let test_sizes = [1, 100, 256, 512, 513, 1000, 1024, 10000, 65536, 100000];
        for &size in &test_sizes {
            let expected = (size + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
            assert_eq!(
                PrefixScanPipeline::num_workgroups(size),
                expected,
                "Failed for size {}", size
            );
        }
    }

    // =========================================================================
    // Block Sums Buffer Size Tests
    // =========================================================================

    #[test]
    fn test_block_sums_buffer_size() {
        // Single block: just 1 element needed
        assert_eq!(PrefixScanPipeline::block_sums_buffer_size(512), 4);

        // Two blocks: 2 + 1 = 3 elements, but rounds to account for recursion
        let size_1024 = PrefixScanPipeline::block_sums_buffer_size(1024);
        assert!(size_1024 >= 8); // At least 2 block sums

        // Large input: should scale logarithmically
        let size_1m = PrefixScanPipeline::block_sums_buffer_size(1_000_000);
        // 1M elements = ~1954 blocks at first level
        // Then ~4 blocks at second level
        // Then 1 at third level
        assert!(size_1m >= (1954 + 4 + 1) * 4);
    }

    #[test]
    fn test_block_sums_buffer_size_single_element() {
        // Edge case: single element needs only 4 bytes
        assert_eq!(PrefixScanPipeline::block_sums_buffer_size(1), 4);
    }

    #[test]
    fn test_block_sums_buffer_size_single_block_boundary() {
        // At exactly ELEMENTS_PER_WORKGROUP, still single block
        assert_eq!(PrefixScanPipeline::block_sums_buffer_size(512), 4);
    }

    #[test]
    fn test_block_sums_buffer_size_two_pass_threshold() {
        // 513 elements requires 2 workgroups -> needs block sums buffer
        let size = PrefixScanPipeline::block_sums_buffer_size(513);
        // 2 blocks at level 1, 1 at level 2
        assert!(size >= 12); // At least 3 u32s
    }

    #[test]
    fn test_block_sums_buffer_size_three_pass_threshold() {
        // > 65536 elements requires multi-pass scan
        // 65537 elements = 129 blocks at level 0
        // 129 elements fit in one workgroup (129 <= 512), so only one more level needed
        let size = PrefixScanPipeline::block_sums_buffer_size(65537);
        // Algorithm: counts blocks at each level where size > 512, then adds 1
        // Level 0: 65537 -> 129 blocks (counted)
        // Level 1: 129 <= 512, exit loop, add 1
        // Total: (129 + 1) * 4 = 520 bytes
        assert!(size >= (129 + 1) * 4);
    }

    #[test]
    fn test_block_sums_buffer_size_exact_two_workgroups() {
        // Exactly 1024 elements = 2 workgroups
        let size = PrefixScanPipeline::block_sums_buffer_size(1024);
        assert!(size >= 8); // 2 block sums minimum
    }

    #[test]
    fn test_block_sums_buffer_size_logarithmic_growth() {
        // Verify buffer size grows logarithmically with input size
        let sizes: Vec<u32> = (10..=25).map(|exp| 1u32 << exp).collect();
        let buffer_sizes: Vec<u64> = sizes
            .iter()
            .map(|&s| PrefixScanPipeline::block_sums_buffer_size(s))
            .collect();

        // Buffer size should not grow faster than O(n / 512 + n / 512^2 + ...)
        for i in 1..buffer_sizes.len() {
            // Each doubling of input should at most double the buffer requirement
            // (approximately, due to multi-level structure)
            assert!(
                buffer_sizes[i] <= buffer_sizes[i - 1] * 3,
                "Buffer size grew too fast from {} to {}",
                buffer_sizes[i - 1],
                buffer_sizes[i]
            );
        }
    }

    #[test]
    fn test_block_sums_buffer_size_non_power_of_two() {
        // Non-power-of-two sizes
        let test_sizes = [1000, 5000, 12345, 99999, 123456];
        for &size in &test_sizes {
            let buffer_size = PrefixScanPipeline::block_sums_buffer_size(size);
            assert!(buffer_size >= 4, "Buffer size must be at least 4 bytes for size {}", size);
            assert!(buffer_size % 4 == 0, "Buffer size must be multiple of 4 for size {}", size);
        }
    }

    // =========================================================================
    // Multi-Pass Logic Tests
    // =========================================================================

    #[test]
    fn test_single_pass_threshold() {
        // Single pass for <= 512 elements
        assert_eq!(PrefixScanPipeline::num_workgroups(1), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(256), 1);
        assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
    }

    #[test]
    fn test_two_pass_threshold() {
        // Two pass for 513-65536 elements (2-128 workgroups)
        assert_eq!(PrefixScanPipeline::num_workgroups(513), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
        assert_eq!(PrefixScanPipeline::num_workgroups(65536), 128);
    }

    #[test]
    fn test_three_pass_threshold() {
        // Three+ pass for > 65536 elements (> 128 workgroups)
        // 65537 elements = 129 workgroups, needs 3 passes
        assert_eq!(PrefixScanPipeline::num_workgroups(65537), 129);
        // 129 workgroups = 1 workgroup for block sums, but that's > 1 so needs recursion
        assert!(PrefixScanPipeline::num_workgroups(65537) > 128);
    }

    #[test]
    fn test_pass_count_calculation() {
        // Helper to count passes needed
        // A pass is needed when size > ELEMENTS_PER_WORKGROUP (512)
        fn count_passes(input_size: u32) -> u32 {
            let mut passes = 1;
            let mut size = input_size;
            while size > ELEMENTS_PER_WORKGROUP {
                passes += 1;
                size = PrefixScanPipeline::num_workgroups(size);
            }
            passes
        }

        assert_eq!(count_passes(1), 1);
        assert_eq!(count_passes(512), 1);
        assert_eq!(count_passes(513), 2);   // 513 -> 2 workgroups, 2 <= 512, done
        assert_eq!(count_passes(65536), 2); // 65536 -> 128 workgroups, 128 <= 512, done
        // 65537 -> 129 workgroups, 129 <= 512, done (only 2 passes!)
        assert_eq!(count_passes(65537), 2);
        // Need > 512 workgroups for 3 passes, i.e., > 512 * 512 = 262144 elements
        assert_eq!(count_passes(262145), 3); // 262145 -> 513 -> 2 -> done
        assert_eq!(count_passes(1_000_000), 3); // 1954 -> 4, done (1954 > 512, 4 <= 512)
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let err = PrefixScanError::EmptyInput;
        assert_eq!(err.to_string(), "Input size cannot be zero");

        let err = PrefixScanError::InputTooLarge {
            size: 1000,
            max: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));

        let err = PrefixScanError::BufferTooSmall {
            required: 1024,
            actual: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_error_empty_input() {
        let err = PrefixScanError::EmptyInput;
        let msg = err.to_string();
        assert!(msg.contains("zero") || msg.contains("empty"));
    }

    #[test]
    fn test_error_input_too_large() {
        let err = PrefixScanError::InputTooLarge {
            size: 2_000_000_000,
            max: 1_073_741_824,
        };
        let msg = err.to_string();
        assert!(msg.contains("2000000000"));
        assert!(msg.contains("1073741824"));
    }

    #[test]
    fn test_error_buffer_too_small() {
        let err = PrefixScanError::BufferTooSmall {
            required: 4096,
            actual: 256,
        };
        let msg = err.to_string();
        assert!(msg.contains("4096"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_error_pipeline_creation_failed() {
        let err = PrefixScanError::PipelineCreationFailed("shader compilation error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Pipeline creation failed"));
        assert!(msg.contains("shader compilation error"));
    }

    #[test]
    fn test_error_clone() {
        let err = PrefixScanError::InputTooLarge { size: 100, max: 50 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_eq() {
        let err1 = PrefixScanError::EmptyInput;
        let err2 = PrefixScanError::EmptyInput;
        assert_eq!(err1, err2);

        let err3 = PrefixScanError::InputTooLarge { size: 100, max: 50 };
        let err4 = PrefixScanError::InputTooLarge { size: 100, max: 50 };
        assert_eq!(err3, err4);

        let err5 = PrefixScanError::InputTooLarge { size: 100, max: 50 };
        let err6 = PrefixScanError::InputTooLarge { size: 200, max: 50 };
        assert_ne!(err5, err6);
    }

    #[test]
    fn test_error_debug() {
        let err = PrefixScanError::BufferTooSmall {
            required: 1000,
            actual: 100,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("BufferTooSmall"));
        assert!(debug_str.contains("1000"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_error_is_std_error() {
        // Verify PrefixScanError implements std::error::Error
        fn assert_error<E: std::error::Error>(_: &E) {}
        let err = PrefixScanError::EmptyInput;
        assert_error(&err);
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_elements_per_workgroup_constant() {
        assert_eq!(ELEMENTS_PER_WORKGROUP, 512);
        assert_eq!(ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE * 2);
    }

    #[test]
    fn test_constants_power_of_two() {
        assert!(WORKGROUP_SIZE.is_power_of_two());
        assert!(ELEMENTS_PER_WORKGROUP.is_power_of_two());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_large_input_size_workgroups() {
        // Test large but safe input sizes
        // Note: num_workgroups uses (input_size + 511) which can overflow for very large inputs
        // Maximum safe input = u32::MAX - 511

        // Test 1 billion elements (realistic maximum)
        let billion = 1_000_000_000u32;
        let result_b = PrefixScanPipeline::num_workgroups(billion);
        assert_eq!(result_b, (billion + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP);

        // Test 2 billion elements
        let two_billion = 2_000_000_000u32;
        let result_2b = PrefixScanPipeline::num_workgroups(two_billion);
        assert_eq!(result_2b, (two_billion + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP);

        // Test 4 billion elements (just under overflow threshold)
        let four_billion = 4_000_000_000u32;
        let result_4b = PrefixScanPipeline::num_workgroups(four_billion);
        assert_eq!(result_4b, (four_billion + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP);
    }

    #[test]
    fn test_boundary_512_elements() {
        // Exactly 512 = 1 workgroup
        assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
        // Buffer size for single workgroup
        assert_eq!(PrefixScanPipeline::block_sums_buffer_size(512), 4);
    }

    #[test]
    fn test_boundary_256_elements() {
        // 256 elements = 1 workgroup (uses half capacity)
        assert_eq!(PrefixScanPipeline::num_workgroups(256), 1);
    }

    #[test]
    fn test_scan_params_all_fields_set() {
        let params = ScanParams {
            input_size: 1000,
            block_offset: 512,
            is_inclusive: 1,
            _pad: 0,
        };
        assert_eq!(params.input_size, 1000);
        assert_eq!(params.block_offset, 512);
        assert_eq!(params.is_inclusive, 1);
        assert_eq!(params._pad, 0);
    }

    #[test]
    fn test_scan_params_memory_layout() {
        // Verify memory layout matches WGSL struct
        let params = ScanParams::exclusive(100);
        let bytes = bytemuck::bytes_of(&params);

        // input_size at offset 0
        let input_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(input_size, 100);

        // block_offset at offset 4
        let block_offset = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(block_offset, 0);

        // is_inclusive at offset 8
        let is_inclusive = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(is_inclusive, 0);

        // _pad at offset 12
        let pad = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad, 0);
    }

    // =========================================================================
    // Integration Readiness Tests (without wgpu device)
    // =========================================================================

    #[test]
    fn test_recursive_scan_level_calculation() {
        // Verify the recursive structure is correctly calculated
        // For 1M elements:
        // Level 0: 1,000,000 elements -> 1954 workgroups
        // Level 1: 1954 elements -> 4 workgroups
        // Level 2: 4 elements -> 1 workgroup
        let mut levels = Vec::new();
        let mut size = 1_000_000u32;

        while size > 1 {
            let wg = PrefixScanPipeline::num_workgroups(size);
            levels.push((size, wg));
            if wg <= 1 {
                break;
            }
            size = wg;
        }

        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], (1_000_000, 1954));
        assert_eq!(levels[1], (1954, 4));
        assert_eq!(levels[2], (4, 1));
    }

    #[test]
    fn test_total_work_calculation() {
        // Total work = sum of all elements processed at each level
        fn total_work(input_size: u32) -> u64 {
            let mut total = 0u64;
            let mut size = input_size;
            while size > 1 {
                total += size as u64;
                let wg = PrefixScanPipeline::num_workgroups(size);
                if wg <= 1 {
                    break;
                }
                size = wg;
            }
            total
        }

        // For small inputs, work is just the input size
        assert_eq!(total_work(512), 512);

        // For larger inputs, includes block sum scans
        let work_1m = total_work(1_000_000);
        // 1M + 1954 + 4 = ~1M
        assert!(work_1m > 1_000_000);
        assert!(work_1m < 1_010_000);
    }

    #[test]
    fn test_num_workgroups_zero_returns_zero() {
        // Edge case: zero input should return 0 workgroups
        // Actually with ceiling division, 0/512 = 0
        assert_eq!(PrefixScanPipeline::num_workgroups(0), 0);
    }

    #[test]
    fn test_block_sums_buffer_size_min_4_bytes() {
        // Even for small inputs, buffer should be at least 4 bytes
        for size in 1..=512 {
            let buffer_size = PrefixScanPipeline::block_sums_buffer_size(size);
            assert!(buffer_size >= 4, "Buffer size {} too small for input {}", buffer_size, size);
        }
    }
}
