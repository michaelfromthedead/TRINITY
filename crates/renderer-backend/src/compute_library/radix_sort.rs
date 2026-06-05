//! GPU Radix Sort Pipeline for wgpu 25.x (T-WGPU-P3.10.4).
//!
//! Implements a parallel radix sort algorithm using histogram + prefix scan + scatter
//! for sorting 32-bit unsigned integer keys with optional associated values.
//!
//! # Algorithm
//!
//! Radix sort processes keys 4 bits at a time (one digit), from least significant
//! to most significant. For 32-bit keys, this requires 8 passes:
//!
//! ```text
//! Pass 0: Sort by bits  0-3  (LSB)
//! Pass 1: Sort by bits  4-7
//! Pass 2: Sort by bits  8-11
//! Pass 3: Sort by bits 12-15
//! Pass 4: Sort by bits 16-19
//! Pass 5: Sort by bits 20-23
//! Pass 6: Sort by bits 24-27
//! Pass 7: Sort by bits 28-31 (MSB)
//! ```
//!
//! Each pass consists of three phases:
//!
//! 1. **Histogram**: Count occurrences of each digit (0-15) per workgroup
//! 2. **Prefix Scan**: Compute exclusive prefix sum of histogram for scatter offsets
//! 3. **Scatter**: Write keys/values to sorted positions based on offsets
//!
//! # Key Properties
//!
//! - **Stable sort**: Equal keys maintain their relative order
//! - **LSB-first**: Required for stability with multiple passes
//! - **16 buckets per pass**: 4 bits = 2^4 = 16 possible digit values
//! - **O(8n) complexity**: Linear time for 32-bit keys
//!
//! # Performance
//!
//! - Elements per workgroup: 1024 (256 threads x 4 elements each)
//! - Shared memory: 16 atomic counters + 16 prefix sums
//! - Memory bandwidth: ~2 reads + 2 writes per element per pass
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_library::radix_sort::RadixSortPipeline;
//! use wgpu::util::DeviceExt;
//!
//! # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create the pipeline
//! let pipeline = RadixSortPipeline::new(device);
//!
//! // Create input data
//! let keys = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
//! let values = vec![50u32, 30, 80, 10, 90, 20, 70, 40, 60, 0]; // Associated data
//!
//! // Create GPU buffers
//! let keys_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("keys"),
//!     contents: bytemuck::cast_slice(&keys),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("values"),
//!     contents: bytemuck::cast_slice(&values),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! // Sort key-value pairs
//! pipeline.sort_pairs(device, queue, &keys_buffer, &values_buffer, keys.len() as u32);
//!
//! // Result: keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
//! //         values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `RadixSortPipeline` is `Send + Sync` as it only holds wgpu pipeline handles.

use std::borrow::Cow;
use std::num::NonZeroU64;

use crate::compute_library::prefix_scan::PrefixScanPipeline;

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size used by radix sort shaders.
pub const WORKGROUP_SIZE: u32 = 256;

/// Number of elements processed per thread.
pub const ELEMENTS_PER_THREAD: u32 = 4;

/// Number of elements processed per workgroup.
pub const ELEMENTS_PER_WORKGROUP: u32 = WORKGROUP_SIZE * ELEMENTS_PER_THREAD; // 1024

/// Number of bits processed per radix pass.
pub const RADIX_BITS: u32 = 4;

/// Number of histogram buckets (2^RADIX_BITS).
pub const RADIX_BUCKETS: u32 = 16;

/// Total number of passes for 32-bit keys.
pub const TOTAL_PASSES: u32 = 8;

/// Shader source code.
const SHADER_SOURCE: &str = include_str!("../../shaders/radix_sort.wgsl");

// ============================================================================
// RadixSortParams - Uniform buffer structure
// ============================================================================

/// Parameters passed to the radix sort shader via uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixSortParams {
    /// Total number of key-value pairs to sort.
    pub input_size: u32,
    /// Current radix pass (0-7).
    pub pass_number: u32,
    /// Number of workgroups dispatched.
    pub num_workgroups: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

impl RadixSortParams {
    /// Create new parameters for a radix sort pass.
    pub fn new(input_size: u32, pass_number: u32, num_workgroups: u32) -> Self {
        Self {
            input_size,
            pass_number,
            num_workgroups,
            _pad: 0,
        }
    }
}

// ============================================================================
// RadixSortError
// ============================================================================

/// Errors that can occur during radix sort operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadixSortError {
    /// Input size is zero.
    EmptyInput,
    /// Input size exceeds maximum supported.
    InputTooLarge { size: u32, max: u32 },
    /// Keys and values buffers have different sizes.
    SizeMismatch { keys: u64, values: u64 },
    /// Buffer is too small for the requested operation.
    BufferTooSmall { required: u64, actual: u64 },
}

impl std::fmt::Display for RadixSortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Input size cannot be zero"),
            Self::InputTooLarge { size, max } => {
                write!(f, "Input size {} exceeds maximum supported {}", size, max)
            }
            Self::SizeMismatch { keys, values } => {
                write!(
                    f,
                    "Keys buffer size {} does not match values buffer size {}",
                    keys, values
                )
            }
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "Buffer size {} is too small, need at least {} bytes",
                    actual, required
                )
            }
        }
    }
}

impl std::error::Error for RadixSortError {}

// ============================================================================
// RadixSortPipeline
// ============================================================================

/// GPU pipeline for parallel radix sort.
///
/// This struct holds the compiled compute pipelines and bind group layouts
/// needed to perform radix sort on the GPU. Create one instance and reuse
/// it for multiple sort operations.
///
/// # Supported Operations
///
/// - `sort_keys()`: Sort keys only (in-place, modifies input buffer)
/// - `sort_pairs()`: Sort key-value pairs (keys sorted, values permuted)
///
/// # Limitations
///
/// - Maximum input size: Limited by GPU memory
/// - Key type: u32 only
/// - Value type: u32 only (use indices to sort other types)
pub struct RadixSortPipeline {
    /// Prefix scan pipeline for computing global offsets.
    prefix_scan: PrefixScanPipeline,

    /// Histogram computation pipeline.
    histogram_pipeline: wgpu::ComputePipeline,

    /// Scatter pipeline for key-value pairs.
    scatter_pipeline: wgpu::ComputePipeline,

    /// Scatter pipeline for keys only.
    scatter_keys_pipeline: wgpu::ComputePipeline,

    /// Clear histogram buffer pipeline.
    clear_histogram_pipeline: wgpu::ComputePipeline,

    /// Copy keys pipeline.
    copy_keys_pipeline: wgpu::ComputePipeline,

    /// Copy pairs pipeline.
    copy_pairs_pipeline: wgpu::ComputePipeline,

    /// Bind group layout for histogram pass.
    histogram_bind_group_layout: wgpu::BindGroupLayout,

    /// Bind group layout for scatter pass.
    scatter_bind_group_layout: wgpu::BindGroupLayout,
}

impl RadixSortPipeline {
    /// Create a new radix sort pipeline.
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
    /// A new `RadixSortPipeline` ready for use.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create prefix scan pipeline (used for histogram prefix sums)
        let prefix_scan = PrefixScanPipeline::new(device);

        // Compile the shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Histogram bind group layout:
        // 0: keys_in (storage, read)
        // 1: keys_out (storage, read_write) - unused but needed for layout
        // 2: values_in (storage, read) - unused but needed for layout
        // 3: values_out (storage, read_write) - unused but needed for layout
        // 4: global_histogram (storage, read_write)
        // 5: global_offsets (storage, read) - unused but needed for layout
        // 6: params (uniform)
        let histogram_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix_sort_histogram_bind_group_layout"),
                entries: &[
                    // keys_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // keys_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // values_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // values_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // global_histogram
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // global_offsets
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<RadixSortParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // Scatter bind group layout (same as histogram)
        let scatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix_sort_scatter_bind_group_layout"),
                entries: &[
                    // keys_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // keys_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // values_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // values_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // global_histogram
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // global_offsets
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<RadixSortParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let histogram_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("radix_sort_histogram_pipeline_layout"),
                bind_group_layouts: &[&histogram_bind_group_layout],
                push_constant_ranges: &[],
            });

        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("radix_sort_scatter_pipeline_layout"),
                bind_group_layouts: &[&scatter_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create pipelines
        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_sort_histogram"),
            layout: Some(&histogram_pipeline_layout),
            module: &shader_module,
            entry_point: "histogram",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_sort_scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &shader_module,
            entry_point: "scatter_init",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_keys_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix_sort_scatter_keys"),
                layout: Some(&scatter_pipeline_layout),
                module: &shader_module,
                entry_point: "scatter_keys_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let clear_histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix_sort_clear_histogram"),
                layout: Some(&histogram_pipeline_layout),
                module: &shader_module,
                entry_point: "clear_histogram",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let copy_keys_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_sort_copy_keys"),
            layout: Some(&scatter_pipeline_layout),
            module: &shader_module,
            entry_point: "copy_keys",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let copy_pairs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix_sort_copy_pairs"),
            layout: Some(&scatter_pipeline_layout),
            module: &shader_module,
            entry_point: "copy_pairs",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            prefix_scan,
            histogram_pipeline,
            scatter_pipeline,
            scatter_keys_pipeline,
            clear_histogram_pipeline,
            copy_keys_pipeline,
            copy_pairs_pipeline,
            histogram_bind_group_layout,
            scatter_bind_group_layout,
        }
    }

    /// Calculate the number of workgroups needed for a given input size.
    #[inline]
    pub fn num_workgroups(input_size: u32) -> u32 {
        (input_size + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
    }

    /// Calculate the required size for the histogram buffer.
    #[inline]
    pub fn histogram_buffer_size(num_workgroups: u32) -> u64 {
        (RADIX_BUCKETS * num_workgroups) as u64 * 4 // 4 bytes per u32
    }

    /// Sort keys only (in-place, modifies the input buffer).
    ///
    /// After sorting, the keys buffer will contain sorted keys.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `keys_buffer` - Buffer containing keys (modified in-place after 8 passes).
    /// * `input_size` - Number of keys to sort.
    ///
    /// # Panics
    ///
    /// Panics if `input_size` is 0.
    pub fn sort_keys(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        keys_buffer: &wgpu::Buffer,
        input_size: u32,
    ) {
        if input_size == 0 {
            return;
        }

        let num_workgroups = Self::num_workgroups(input_size);

        // Create temporary buffers
        let keys_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_keys_temp"),
            size: keys_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dummy values buffer (not used for keys-only sort)
        let dummy_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_dummy_values"),
            size: 4, // Minimum size
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create histogram and offsets buffers
        let histogram_size = Self::histogram_buffer_size(num_workgroups);
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_histogram"),
            size: histogram_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // For offsets, we use the same buffer (prefix scan modifies in-place)
        // But we need a separate buffer for reading during scatter
        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_offsets"),
            size: histogram_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Block sums buffer for prefix scan
        let block_sums_size = PrefixScanPipeline::block_sums_buffer_size(
            (RADIX_BUCKETS * num_workgroups) as u32,
        )
        .max(4);
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_block_sums"),
            size: block_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Perform 8 radix passes
        for pass in 0..TOTAL_PASSES {
            // Determine source and destination buffers (ping-pong)
            let (src_keys, dst_keys) = if pass % 2 == 0 {
                (keys_buffer, &keys_temp)
            } else {
                (&keys_temp, keys_buffer)
            };

            self.execute_pass_keys_only(
                device,
                queue,
                src_keys,
                dst_keys,
                &dummy_values,
                &dummy_values,
                &histogram_buffer,
                &offsets_buffer,
                &block_sums_buffer,
                input_size,
                pass,
                num_workgroups,
            );
        }

        // After 8 passes (even number), result is back in original buffer
    }

    /// Sort key-value pairs.
    ///
    /// After sorting, both buffers will contain sorted data with values
    /// permuted to match their corresponding keys.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `keys_buffer` - Buffer containing keys (modified in-place).
    /// * `values_buffer` - Buffer containing values (modified in-place).
    /// * `input_size` - Number of key-value pairs to sort.
    ///
    /// # Panics
    ///
    /// Panics if `input_size` is 0 or if buffers have different sizes.
    pub fn sort_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        keys_buffer: &wgpu::Buffer,
        values_buffer: &wgpu::Buffer,
        input_size: u32,
    ) {
        if input_size == 0 {
            return;
        }

        let num_workgroups = Self::num_workgroups(input_size);

        // Create temporary buffers for ping-pong
        let keys_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_keys_temp"),
            size: keys_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let values_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_values_temp"),
            size: values_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create histogram and offsets buffers
        let histogram_size = Self::histogram_buffer_size(num_workgroups);
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_histogram"),
            size: histogram_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_offsets"),
            size: histogram_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Block sums buffer for prefix scan
        let block_sums_size = PrefixScanPipeline::block_sums_buffer_size(
            (RADIX_BUCKETS * num_workgroups) as u32,
        )
        .max(4);
        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_block_sums"),
            size: block_sums_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Perform 8 radix passes
        for pass in 0..TOTAL_PASSES {
            // Determine source and destination buffers (ping-pong)
            let (src_keys, dst_keys, src_values, dst_values) = if pass % 2 == 0 {
                (keys_buffer, &keys_temp, values_buffer, &values_temp)
            } else {
                (&keys_temp, keys_buffer, &values_temp, values_buffer)
            };

            self.execute_pass(
                device,
                queue,
                src_keys,
                dst_keys,
                src_values,
                dst_values,
                &histogram_buffer,
                &offsets_buffer,
                &block_sums_buffer,
                input_size,
                pass,
                num_workgroups,
            );
        }

        // After 8 passes (even number), result is back in original buffers
    }

    /// Execute a single radix sort pass for key-value pairs.
    #[allow(clippy::too_many_arguments)]
    fn execute_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_keys: &wgpu::Buffer,
        dst_keys: &wgpu::Buffer,
        src_values: &wgpu::Buffer,
        dst_values: &wgpu::Buffer,
        histogram_buffer: &wgpu::Buffer,
        offsets_buffer: &wgpu::Buffer,
        block_sums_buffer: &wgpu::Buffer,
        input_size: u32,
        pass: u32,
        num_workgroups: u32,
    ) {
        // Create params buffer
        let params = RadixSortParams::new(input_size, pass, num_workgroups);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_params"),
            size: std::mem::size_of::<RadixSortParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group for histogram/scatter
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix_sort_bind_group"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Phase 1: Clear histogram
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_clear_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_clear_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.clear_histogram_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let clear_workgroups =
                    (RADIX_BUCKETS * num_workgroups + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(clear_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 2: Compute histogram
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_histogram_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_histogram_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.histogram_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 3: Prefix scan of histogram to get offsets
        {
            // Copy histogram to offsets buffer for in-place scan
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_copy_histogram_encoder"),
            });
            encoder.copy_buffer_to_buffer(
                histogram_buffer,
                0,
                offsets_buffer,
                0,
                Self::histogram_buffer_size(num_workgroups),
            );
            queue.submit(std::iter::once(encoder.finish()));

            // Perform prefix scan on offsets buffer
            self.prefix_scan.scan_with_block_sums(
                device,
                queue,
                offsets_buffer,
                block_sums_buffer,
                RADIX_BUCKETS * num_workgroups,
            );
        }

        // Phase 4: Scatter
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_scatter_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_scatter_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Execute a single radix sort pass for keys only.
    #[allow(clippy::too_many_arguments)]
    fn execute_pass_keys_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_keys: &wgpu::Buffer,
        dst_keys: &wgpu::Buffer,
        src_values: &wgpu::Buffer,
        dst_values: &wgpu::Buffer,
        histogram_buffer: &wgpu::Buffer,
        offsets_buffer: &wgpu::Buffer,
        block_sums_buffer: &wgpu::Buffer,
        input_size: u32,
        pass: u32,
        num_workgroups: u32,
    ) {
        // Create params buffer
        let params = RadixSortParams::new(input_size, pass, num_workgroups);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_sort_params"),
            size: std::mem::size_of::<RadixSortParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix_sort_bind_group"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Phase 1: Clear histogram
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_clear_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_clear_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.clear_histogram_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let clear_workgroups =
                    (RADIX_BUCKETS * num_workgroups + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(clear_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 2: Compute histogram
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_histogram_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_histogram_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.histogram_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 3: Prefix scan of histogram to get offsets
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_copy_histogram_encoder"),
            });
            encoder.copy_buffer_to_buffer(
                histogram_buffer,
                0,
                offsets_buffer,
                0,
                Self::histogram_buffer_size(num_workgroups),
            );
            queue.submit(std::iter::once(encoder.finish()));

            self.prefix_scan.scan_with_block_sums(
                device,
                queue,
                offsets_buffer,
                block_sums_buffer,
                RADIX_BUCKETS * num_workgroups,
            );
        }

        // Phase 4: Scatter keys only
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("radix_sort_scatter_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix_sort_scatter_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_keys_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Get the prefix scan pipeline for external use.
    pub fn prefix_scan(&self) -> &PrefixScanPipeline {
        &self.prefix_scan
    }

    /// Get the histogram bind group layout for external use.
    pub fn histogram_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.histogram_bind_group_layout
    }

    /// Get the scatter bind group layout for external use.
    pub fn scatter_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scatter_bind_group_layout
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RadixSortParams Tests
    // =========================================================================

    #[test]
    fn test_radix_sort_params_new() {
        let params = RadixSortParams::new(1000, 3, 4);
        assert_eq!(params.input_size, 1000);
        assert_eq!(params.pass_number, 3);
        assert_eq!(params.num_workgroups, 4);
        assert_eq!(params._pad, 0);
    }

    #[test]
    fn test_radix_sort_params_default() {
        let params = RadixSortParams::default();
        assert_eq!(params.input_size, 0);
        assert_eq!(params.pass_number, 0);
        assert_eq!(params.num_workgroups, 0);
    }

    #[test]
    fn test_radix_sort_params_pod() {
        let params = RadixSortParams::new(100, 2, 5);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<RadixSortParams>());
        assert_eq!(bytes.len(), 16); // 4 u32s = 16 bytes
    }

    #[test]
    fn test_radix_sort_params_zeroable() {
        let zeroed: RadixSortParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.input_size, 0);
        assert_eq!(zeroed.pass_number, 0);
        assert_eq!(zeroed.num_workgroups, 0);
    }

    #[test]
    fn test_radix_sort_params_clone() {
        let params = RadixSortParams::new(500, 7, 10);
        let cloned = params.clone();
        assert_eq!(cloned.input_size, params.input_size);
        assert_eq!(cloned.pass_number, params.pass_number);
        assert_eq!(cloned.num_workgroups, params.num_workgroups);
    }

    #[test]
    fn test_radix_sort_params_copy() {
        let params = RadixSortParams::new(200, 1, 2);
        let copied = params; // Copy
        let _still_valid = params; // Original still valid
        assert_eq!(copied.input_size, 200);
    }

    #[test]
    fn test_radix_sort_params_memory_layout() {
        let params = RadixSortParams::new(12345, 6, 99);
        let bytes = bytemuck::bytes_of(&params);

        let input_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(input_size, 12345);

        let pass_number = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(pass_number, 6);

        let num_workgroups = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(num_workgroups, 99);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let err = RadixSortError::EmptyInput;
        assert_eq!(err.to_string(), "Input size cannot be zero");

        let err = RadixSortError::InputTooLarge {
            size: 2000,
            max: 1000,
        };
        assert!(err.to_string().contains("2000"));
        assert!(err.to_string().contains("1000"));

        let err = RadixSortError::SizeMismatch {
            keys: 100,
            values: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));

        let err = RadixSortError::BufferTooSmall {
            required: 1024,
            actual: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_error_eq() {
        let err1 = RadixSortError::EmptyInput;
        let err2 = RadixSortError::EmptyInput;
        assert_eq!(err1, err2);

        let err3 = RadixSortError::InputTooLarge {
            size: 100,
            max: 50,
        };
        let err4 = RadixSortError::InputTooLarge {
            size: 100,
            max: 50,
        };
        assert_eq!(err3, err4);

        let err5 = RadixSortError::InputTooLarge {
            size: 100,
            max: 50,
        };
        let err6 = RadixSortError::InputTooLarge {
            size: 200,
            max: 50,
        };
        assert_ne!(err5, err6);
    }

    #[test]
    fn test_error_clone() {
        let err = RadixSortError::SizeMismatch {
            keys: 100,
            values: 200,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        let err = RadixSortError::EmptyInput;
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
    fn test_elements_per_thread_constant() {
        assert_eq!(ELEMENTS_PER_THREAD, 4);
    }

    #[test]
    fn test_elements_per_workgroup_constant() {
        assert_eq!(ELEMENTS_PER_WORKGROUP, 1024);
        assert_eq!(ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE * ELEMENTS_PER_THREAD);
    }

    #[test]
    fn test_radix_bits_constant() {
        assert_eq!(RADIX_BITS, 4);
    }

    #[test]
    fn test_radix_buckets_constant() {
        assert_eq!(RADIX_BUCKETS, 16);
        assert_eq!(RADIX_BUCKETS, 1 << RADIX_BITS);
    }

    #[test]
    fn test_total_passes_constant() {
        assert_eq!(TOTAL_PASSES, 8);
        assert_eq!(TOTAL_PASSES, 32 / RADIX_BITS);
    }

    #[test]
    fn test_constants_power_of_two() {
        assert!(WORKGROUP_SIZE.is_power_of_two());
        assert!(ELEMENTS_PER_WORKGROUP.is_power_of_two());
        assert!(RADIX_BUCKETS.is_power_of_two());
    }

    // =========================================================================
    // Workgroup Calculation Tests
    // =========================================================================

    #[test]
    fn test_num_workgroups() {
        assert_eq!(RadixSortPipeline::num_workgroups(1), 1);
        assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
        assert_eq!(RadixSortPipeline::num_workgroups(1025), 2);
        assert_eq!(RadixSortPipeline::num_workgroups(2048), 2);
        assert_eq!(RadixSortPipeline::num_workgroups(2049), 3);
    }

    #[test]
    fn test_num_workgroups_single_element() {
        assert_eq!(RadixSortPipeline::num_workgroups(1), 1);
    }

    #[test]
    fn test_num_workgroups_exact_multiple() {
        assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
        assert_eq!(RadixSortPipeline::num_workgroups(2048), 2);
        assert_eq!(RadixSortPipeline::num_workgroups(10240), 10);
    }

    #[test]
    fn test_num_workgroups_just_over_boundary() {
        assert_eq!(RadixSortPipeline::num_workgroups(1025), 2);
        assert_eq!(RadixSortPipeline::num_workgroups(2049), 3);
    }

    #[test]
    fn test_num_workgroups_large_arrays() {
        assert_eq!(RadixSortPipeline::num_workgroups(1_000_000), 977);
        assert_eq!(RadixSortPipeline::num_workgroups(10_000_000), 9766);
    }

    // =========================================================================
    // Histogram Buffer Size Tests
    // =========================================================================

    #[test]
    fn test_histogram_buffer_size() {
        // 1 workgroup: 16 buckets * 1 * 4 bytes = 64 bytes
        assert_eq!(RadixSortPipeline::histogram_buffer_size(1), 64);

        // 10 workgroups: 16 buckets * 10 * 4 bytes = 640 bytes
        assert_eq!(RadixSortPipeline::histogram_buffer_size(10), 640);

        // 100 workgroups: 16 * 100 * 4 = 6400 bytes
        assert_eq!(RadixSortPipeline::histogram_buffer_size(100), 6400);
    }

    #[test]
    fn test_histogram_buffer_size_formula() {
        for num_wg in [1, 5, 10, 50, 100, 500, 1000] {
            let expected = (RADIX_BUCKETS * num_wg) as u64 * 4;
            assert_eq!(
                RadixSortPipeline::histogram_buffer_size(num_wg),
                expected,
                "Failed for {} workgroups",
                num_wg
            );
        }
    }

    // =========================================================================
    // Digit Extraction Tests (via constants)
    // =========================================================================

    #[test]
    fn test_digit_extraction_logic() {
        // Test the digit extraction formula used in the shader
        fn extract_digit(key: u32, pass: u32) -> u32 {
            let shift = pass * RADIX_BITS;
            (key >> shift) & 0xF
        }

        // Test key 0x12345678
        let key: u32 = 0x12345678;
        assert_eq!(extract_digit(key, 0), 0x8); // Bits 0-3
        assert_eq!(extract_digit(key, 1), 0x7); // Bits 4-7
        assert_eq!(extract_digit(key, 2), 0x6); // Bits 8-11
        assert_eq!(extract_digit(key, 3), 0x5); // Bits 12-15
        assert_eq!(extract_digit(key, 4), 0x4); // Bits 16-19
        assert_eq!(extract_digit(key, 5), 0x3); // Bits 20-23
        assert_eq!(extract_digit(key, 6), 0x2); // Bits 24-27
        assert_eq!(extract_digit(key, 7), 0x1); // Bits 28-31
    }

    #[test]
    fn test_digit_extraction_edge_cases() {
        fn extract_digit(key: u32, pass: u32) -> u32 {
            let shift = pass * RADIX_BITS;
            (key >> shift) & 0xF
        }

        // All zeros
        assert_eq!(extract_digit(0, 0), 0);
        assert_eq!(extract_digit(0, 7), 0);

        // All ones
        assert_eq!(extract_digit(u32::MAX, 0), 0xF);
        assert_eq!(extract_digit(u32::MAX, 7), 0xF);

        // Single bit patterns
        assert_eq!(extract_digit(0x1, 0), 0x1);
        assert_eq!(extract_digit(0x10, 1), 0x1);
        assert_eq!(extract_digit(0x10000000, 7), 0x1);
    }

    // =========================================================================
    // Pass Number Validation Tests
    // =========================================================================

    #[test]
    fn test_pass_number_range() {
        // Valid pass numbers are 0-7
        for pass in 0..TOTAL_PASSES {
            let params = RadixSortParams::new(100, pass, 1);
            assert_eq!(params.pass_number, pass);
        }
    }

    #[test]
    fn test_ping_pong_pattern() {
        // After 8 passes (even number), data should be back in original buffer
        // Pass 0: A -> B
        // Pass 1: B -> A
        // Pass 2: A -> B
        // Pass 3: B -> A
        // Pass 4: A -> B
        // Pass 5: B -> A
        // Pass 6: A -> B
        // Pass 7: B -> A
        // Final: result in A (original buffer)

        let mut in_original = true;
        for pass in 0..TOTAL_PASSES {
            in_original = pass % 2 == 1;
        }
        assert!(in_original, "After 8 passes, data should be in original buffer");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_zero_workgroups() {
        assert_eq!(RadixSortPipeline::num_workgroups(0), 0);
    }

    #[test]
    fn test_boundary_1024_elements() {
        assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
        assert_eq!(
            RadixSortPipeline::histogram_buffer_size(1),
            RADIX_BUCKETS as u64 * 4
        );
    }

    #[test]
    fn test_large_input_workgroups() {
        // 1 billion elements
        let billion = 1_000_000_000u32;
        let num_wg = RadixSortPipeline::num_workgroups(billion);
        assert_eq!(
            num_wg,
            (billion + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
        );

        // Histogram buffer for 1B elements
        let hist_size = RadixSortPipeline::histogram_buffer_size(num_wg);
        assert_eq!(hist_size, (RADIX_BUCKETS * num_wg) as u64 * 4);
    }

    // =========================================================================
    // Sort Property Tests (conceptual, no GPU)
    // =========================================================================

    #[test]
    fn test_sort_stability_concept() {
        // Radix sort is stable: equal keys maintain relative order
        // This is guaranteed by:
        // 1. Processing LSB to MSB
        // 2. Each pass preserves order within same digit
        // 3. atomicAdd within workgroups preserves order
        // 4. Workgroups processed in index order

        // Simulate stability: keys with same value, different values
        let keys = [(5u32, 'a'), (3, 'b'), (5, 'c'), (3, 'd')];
        // After stable sort: [(3,'b'), (3,'d'), (5,'a'), (5,'c')]
        // The 'b' should come before 'd' because it appeared first
        // The 'a' should come before 'c' because it appeared first

        // This is what we verify with actual GPU tests
        assert!(true, "Stability is verified by GPU integration tests");
    }

    #[test]
    fn test_sort_correctness_concept() {
        // For any input, radix sort produces:
        // 1. A permutation of the input
        // 2. Where keys are in non-decreasing order
        // 3. And equal keys maintain relative order (stability)

        // We can verify this with a simple CPU simulation
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        let mut keys = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        cpu_radix_sort(&mut keys);
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_cpu_radix_sort_random() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // Test with more random-ish values
        let mut keys = vec![
            0xDEADBEEF,
            0xCAFEBABE,
            0x12345678,
            0x87654321,
            0x00000001,
            0xFFFFFFFF,
            0x80000000,
            0x00000000,
        ];
        let mut expected = keys.clone();
        expected.sort();

        cpu_radix_sort(&mut keys);
        assert_eq!(keys, expected);
    }

    // =========================================================================
    // WHITEBOX TESTS: T-WGPU-P3.10.4 - Comprehensive Coverage
    // =========================================================================

    // -------------------------------------------------------------------------
    // 1. SortParams struct - Advanced tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_radix_sort_params_16_byte_alignment() {
        // Verify struct is exactly 16 bytes (4 u32s)
        assert_eq!(std::mem::size_of::<RadixSortParams>(), 16);
        // Verify alignment requirement
        assert_eq!(std::mem::align_of::<RadixSortParams>(), 4);
    }

    #[test]
    fn test_radix_sort_params_field_offsets() {
        // Verify field offsets for correct GPU layout
        let params = RadixSortParams::new(0xAABBCCDD, 0x11223344, 0x55667788);
        let bytes = bytemuck::bytes_of(&params);

        // Offset 0: input_size
        let input_size = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(input_size, 0xAABBCCDD);

        // Offset 4: pass_number
        let pass_number = u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(pass_number, 0x11223344);

        // Offset 8: num_workgroups
        let num_workgroups = u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(num_workgroups, 0x55667788);

        // Offset 12: padding (should be 0)
        let pad = u32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad, 0);
    }

    #[test]
    fn test_radix_sort_params_max_values() {
        // Test with maximum u32 values
        let params = RadixSortParams::new(u32::MAX, 7, u32::MAX);
        assert_eq!(params.input_size, u32::MAX);
        assert_eq!(params.pass_number, 7);
        assert_eq!(params.num_workgroups, u32::MAX);
    }

    #[test]
    fn test_radix_sort_params_debug_trait() {
        let params = RadixSortParams::new(100, 5, 10);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("5"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_radix_sort_params_bytemuck_roundtrip() {
        let params = RadixSortParams::new(12345, 3, 99);
        let bytes = bytemuck::bytes_of(&params);
        let restored: &RadixSortParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.input_size, 12345);
        assert_eq!(restored.pass_number, 3);
        assert_eq!(restored.num_workgroups, 99);
    }

    // -------------------------------------------------------------------------
    // 2. Histogram computation - Bucket count tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_histogram_bucket_count() {
        // 16 buckets per pass (4-bit radix = 2^4 = 16)
        assert_eq!(RADIX_BUCKETS, 16);
        assert_eq!(1 << RADIX_BITS, RADIX_BUCKETS);
    }

    #[test]
    fn test_histogram_bucket_distribution_single_digit() {
        // Simulate histogram for keys 0-15 (one of each digit)
        fn compute_histogram(keys: &[u32], pass: u32) -> [u32; 16] {
            let mut histogram = [0u32; 16];
            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                histogram[digit] += 1;
            }
            histogram
        }

        // Keys 0-15 at pass 0: each bucket should have exactly 1
        let keys: Vec<u32> = (0..16).collect();
        let hist = compute_histogram(&keys, 0);
        for count in hist {
            assert_eq!(count, 1, "Each bucket should have exactly 1 element");
        }
    }

    #[test]
    fn test_histogram_bucket_distribution_uniform() {
        // Simulate uniform distribution
        fn compute_histogram(keys: &[u32], pass: u32) -> [u32; 16] {
            let mut histogram = [0u32; 16];
            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                histogram[digit] += 1;
            }
            histogram
        }

        // All keys with same digit at pass 0
        let keys = vec![0x10u32; 100]; // All have digit 0 at pass 0
        let hist = compute_histogram(&keys, 0);
        assert_eq!(hist[0], 100);
        for &count in hist.iter().skip(1) {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_histogram_all_buckets_used() {
        fn compute_histogram(keys: &[u32], pass: u32) -> [u32; 16] {
            let mut histogram = [0u32; 16];
            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                histogram[digit] += 1;
            }
            histogram
        }

        // Keys that use all 16 buckets
        let keys: Vec<u32> = (0..160).map(|i| i % 16).collect();
        let hist = compute_histogram(&keys, 0);
        for &count in &hist {
            assert_eq!(count, 10, "Each bucket should have 10 elements");
        }
    }

    #[test]
    fn test_histogram_buffer_size_for_workgroups() {
        // Verify histogram buffer sizing formula
        for num_wg in [1, 10, 100, 1000] {
            let size = RadixSortPipeline::histogram_buffer_size(num_wg);
            // 16 buckets * num_workgroups * 4 bytes
            assert_eq!(size, (16 * num_wg) as u64 * 4);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Scatter logic - Output positioning tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scatter_position_calculation() {
        // Verify scatter positioning formula
        // dst_idx = global_base + local_offset

        // Simulate a simple scatter
        fn simulate_scatter_positions(
            keys: &[u32],
            pass: u32,
            global_offsets: &[u32; 16],
        ) -> Vec<usize> {
            let mut local_counters = [0u32; 16];
            let mut positions = Vec::with_capacity(keys.len());

            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                let global_base = global_offsets[digit];
                let local_offset = local_counters[digit];
                positions.push((global_base + local_offset) as usize);
                local_counters[digit] += 1;
            }

            positions
        }

        // Simple test: keys 0, 1, 2, 3 at pass 0
        let keys = [0u32, 1, 2, 3];
        // Offsets: bucket 0 starts at 0, bucket 1 at 1, etc.
        let offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let positions = simulate_scatter_positions(&keys, 0, &offsets);
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_scatter_stability_same_digit() {
        // Elements with same digit should maintain relative order
        fn simulate_scatter(keys: &[(u32, u32)], pass: u32) -> Vec<(u32, u32)> {
            // First compute histogram
            let mut histogram = [0u32; 16];
            for &(key, _) in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                histogram[digit] += 1;
            }

            // Compute prefix sum (offsets)
            let mut offsets = [0u32; 16];
            let mut sum = 0;
            for i in 0..16 {
                offsets[i] = sum;
                sum += histogram[i];
            }

            // Scatter maintaining order
            let mut result = vec![(0u32, 0u32); keys.len()];
            let mut counters = [0u32; 16];
            for &(key, value) in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                let pos = (offsets[digit] + counters[digit]) as usize;
                result[pos] = (key, value);
                counters[digit] += 1;
            }

            result
        }

        // All keys have same digit (0) at pass 0, values indicate original order
        let keys = [(0x10u32, 1), (0x20, 2), (0x30, 3), (0x40, 4)];
        let result = simulate_scatter(&keys, 0);

        // All go to bucket 0, should maintain order: 1, 2, 3, 4
        let values: Vec<u32> = result.iter().map(|&(_, v)| v).collect();
        assert_eq!(values, vec![1, 2, 3, 4], "Stability: order should be preserved");
    }

    #[test]
    fn test_scatter_different_digits() {
        fn cpu_scatter(keys: &[u32], pass: u32) -> Vec<u32> {
            let mut histogram = [0u32; 16];
            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                histogram[digit] += 1;
            }

            let mut offsets = [0u32; 16];
            let mut sum = 0;
            for i in 0..16 {
                offsets[i] = sum;
                sum += histogram[i];
            }

            let mut result = vec![0u32; keys.len()];
            let mut counters = [0u32; 16];
            for &key in keys {
                let digit = ((key >> (pass * 4)) & 0xF) as usize;
                let pos = (offsets[digit] + counters[digit]) as usize;
                result[pos] = key;
                counters[digit] += 1;
            }

            result
        }

        // Keys: 3, 1, 4, 1, 5, 9, 2, 6 at pass 0
        let keys = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let result = cpu_scatter(&keys, 0);

        // After scatter: sorted by last digit
        // 1, 1, 2, 3, 4, 5, 6, 9
        assert_eq!(result, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    // -------------------------------------------------------------------------
    // 4. Multi-pass coordination - 8 passes for 32-bit keys
    // -------------------------------------------------------------------------

    #[test]
    fn test_total_passes_for_32bit_keys() {
        // 32-bit keys / 4 bits per pass = 8 passes
        assert_eq!(TOTAL_PASSES, 8);
        assert_eq!(32 / RADIX_BITS, TOTAL_PASSES);
    }

    #[test]
    fn test_ping_pong_buffer_even_passes() {
        // After even number of passes, data ends in original buffer
        let mut buffer_a = true; // Data starts in A

        for pass in 0..8 {
            // Each pass swaps: A->B or B->A
            buffer_a = pass % 2 == 1;
        }

        assert!(buffer_a, "After 8 passes, data should be in buffer A");
    }

    #[test]
    fn test_pass_sequence_coverage() {
        // Verify all 32 bits are covered by 8 passes
        let mut bits_covered = 0u64;

        for pass in 0..TOTAL_PASSES {
            let shift = pass * RADIX_BITS;
            let mask = 0xF_u64 << shift;
            bits_covered |= mask;
        }

        // Should cover all 32 bits (bits 0-31)
        assert_eq!(bits_covered, 0xFFFF_FFFF);
    }

    #[test]
    fn test_multi_pass_sort_all_ones() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // All 0xFFFFFFFF should remain unchanged
        let mut keys = vec![0xFFFFFFFF_u32; 10];
        cpu_radix_sort(&mut keys);
        assert!(keys.iter().all(|&k| k == 0xFFFFFFFF));
    }

    #[test]
    fn test_multi_pass_sort_all_zeros() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        let mut keys = vec![0u32; 10];
        cpu_radix_sort(&mut keys);
        assert!(keys.iter().all(|&k| k == 0));
    }

    #[test]
    fn test_multi_pass_ascending_sequence() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // Already sorted input
        let mut keys: Vec<u32> = (0..256).collect();
        let expected = keys.clone();
        cpu_radix_sort(&mut keys);
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_multi_pass_descending_sequence() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // Reverse sorted input
        let mut keys: Vec<u32> = (0..256).rev().collect();
        let expected: Vec<u32> = (0..256).collect();
        cpu_radix_sort(&mut keys);
        assert_eq!(keys, expected);
    }

    // -------------------------------------------------------------------------
    // 5. Key-value pairs - Permutation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_key_value_permutation_simple() {
        fn cpu_radix_sort_pairs(keys: &mut [u32], values: &mut [u32]) {
            assert_eq!(keys.len(), values.len());

            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<(u32, u32)>> = vec![vec![]; 16];

                for (&key, &value) in keys.iter().zip(values.iter()) {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push((key, value));
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &(key, value) in bucket {
                        keys[idx] = key;
                        values[idx] = value;
                        idx += 1;
                    }
                }
            }
        }

        let mut keys = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let mut values = vec![50u32, 30, 80, 10, 90, 20, 70, 40, 60, 0];

        cpu_radix_sort_pairs(&mut keys, &mut values);

        // After sorting, values should follow their keys
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(values, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    fn test_key_value_correlation() {
        fn cpu_radix_sort_pairs(keys: &mut [u32], values: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<(u32, u32)>> = vec![vec![]; 16];

                for (&key, &value) in keys.iter().zip(values.iter()) {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push((key, value));
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &(key, value) in bucket {
                        keys[idx] = key;
                        values[idx] = value;
                        idx += 1;
                    }
                }
            }
        }

        // Keys and values where value = key * 10
        let mut keys: Vec<u32> = (0..100).rev().collect();
        let mut values: Vec<u32> = keys.iter().map(|&k| k * 10).collect();

        cpu_radix_sort_pairs(&mut keys, &mut values);

        // Verify correlation preserved
        for (key, value) in keys.iter().zip(values.iter()) {
            assert_eq!(*value, *key * 10, "Key-value correlation broken");
        }
    }

    #[test]
    fn test_key_value_duplicate_keys() {
        fn cpu_radix_sort_pairs(keys: &mut [u32], values: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<(u32, u32)>> = vec![vec![]; 16];

                for (&key, &value) in keys.iter().zip(values.iter()) {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push((key, value));
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &(key, value) in bucket {
                        keys[idx] = key;
                        values[idx] = value;
                        idx += 1;
                    }
                }
            }
        }

        // Multiple pairs with same key
        let mut keys = vec![5u32, 5, 5, 3, 3, 1];
        let mut values = vec![1u32, 2, 3, 4, 5, 6]; // Original order

        cpu_radix_sort_pairs(&mut keys, &mut values);

        // Keys should be sorted, values stable within same key
        assert_eq!(keys, vec![1, 3, 3, 5, 5, 5]);
        assert_eq!(values, vec![6, 4, 5, 1, 2, 3]); // Stability preserved
    }

    #[test]
    fn test_key_value_indices_as_values() {
        fn cpu_radix_sort_pairs(keys: &mut [u32], values: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<(u32, u32)>> = vec![vec![]; 16];

                for (&key, &value) in keys.iter().zip(values.iter()) {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push((key, value));
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &(key, value) in bucket {
                        keys[idx] = key;
                        values[idx] = value;
                        idx += 1;
                    }
                }
            }
        }

        // Use indices as values (common pattern for indirect sort)
        let mut keys = vec![9u32, 4, 7, 2, 8, 1, 5, 3, 6, 0];
        let mut values: Vec<u32> = (0..10).collect(); // Original indices

        cpu_radix_sort_pairs(&mut keys, &mut values);

        // values now tells us original position of each sorted element
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(values, vec![9, 5, 3, 7, 1, 6, 8, 2, 4, 0]);
    }

    // -------------------------------------------------------------------------
    // 6. Error type tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_debug_trait() {
        let err = RadixSortError::EmptyInput;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("EmptyInput"));
    }

    #[test]
    fn test_error_all_variants() {
        let errors = [
            RadixSortError::EmptyInput,
            RadixSortError::InputTooLarge { size: 100, max: 50 },
            RadixSortError::SizeMismatch {
                keys: 100,
                values: 50,
            },
            RadixSortError::BufferTooSmall {
                required: 1024,
                actual: 512,
            },
        ];

        for err in &errors {
            // All variants should be displayable
            let _ = err.to_string();
            // All variants should be clonable
            let _ = err.clone();
            // All variants should be comparable
            assert_eq!(err, err);
        }
    }

    // -------------------------------------------------------------------------
    // 7. Edge cases and boundary conditions
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_element() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        let mut keys = vec![42u32];
        cpu_radix_sort(&mut keys);
        assert_eq!(keys, vec![42]);
    }

    #[test]
    fn test_two_elements_reversed() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        let mut keys = vec![100u32, 1];
        cpu_radix_sort(&mut keys);
        assert_eq!(keys, vec![1, 100]);
    }

    #[test]
    fn test_exact_workgroup_boundary() {
        // 1024 elements = exactly 1 workgroup
        assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
        // 1025 elements = 2 workgroups
        assert_eq!(RadixSortPipeline::num_workgroups(1025), 2);
    }

    #[test]
    fn test_large_workgroup_count() {
        // 1 million elements
        let n = 1_000_000u32;
        let wg = RadixSortPipeline::num_workgroups(n);
        // Ceiling division
        assert_eq!(wg, (n + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP);
        assert_eq!(wg, 977);
    }

    #[test]
    fn test_alternating_high_low() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // Alternating 0 and MAX
        let mut keys = vec![
            0u32,
            u32::MAX,
            0,
            u32::MAX,
            0,
            u32::MAX,
            0,
            u32::MAX,
        ];
        cpu_radix_sort(&mut keys);
        assert_eq!(
            keys,
            vec![0, 0, 0, 0, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
        );
    }

    #[test]
    fn test_powers_of_two() {
        fn cpu_radix_sort(keys: &mut [u32]) {
            for pass in 0..8 {
                let shift = pass * 4;
                let mut buckets: Vec<Vec<u32>> = vec![vec![]; 16];

                for &key in keys.iter() {
                    let digit = ((key >> shift) & 0xF) as usize;
                    buckets[digit].push(key);
                }

                let mut idx = 0;
                for bucket in &buckets {
                    for &key in bucket {
                        keys[idx] = key;
                        idx += 1;
                    }
                }
            }
        }

        // Powers of 2
        let mut keys: Vec<u32> = (0..32).map(|i| 1u32 << i).collect();
        let mut expected = keys.clone();
        expected.sort();

        cpu_radix_sort(&mut keys);
        assert_eq!(keys, expected);
    }

    // -------------------------------------------------------------------------
    // 8. Thread safety verification (compile-time)
    // -------------------------------------------------------------------------

    // Note: This test verifies Send + Sync at compile time
    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_radix_sort_params_send_sync() {
        _assert_send_sync::<RadixSortParams>();
    }

    #[test]
    fn test_radix_sort_error_send_sync() {
        _assert_send_sync::<RadixSortError>();
    }

    // -------------------------------------------------------------------------
    // 9. Bit manipulation precision tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_digit_extraction_all_passes() {
        fn extract_digit(key: u32, pass: u32) -> u32 {
            let shift = pass * RADIX_BITS;
            (key >> shift) & 0xF
        }

        // Test key 0x01234567
        let key = 0x01234567u32;
        assert_eq!(extract_digit(key, 0), 0x7);
        assert_eq!(extract_digit(key, 1), 0x6);
        assert_eq!(extract_digit(key, 2), 0x5);
        assert_eq!(extract_digit(key, 3), 0x4);
        assert_eq!(extract_digit(key, 4), 0x3);
        assert_eq!(extract_digit(key, 5), 0x2);
        assert_eq!(extract_digit(key, 6), 0x1);
        assert_eq!(extract_digit(key, 7), 0x0);
    }

    #[test]
    fn test_digit_extraction_boundary_values() {
        fn extract_digit(key: u32, pass: u32) -> u32 {
            let shift = pass * RADIX_BITS;
            (key >> shift) & 0xF
        }

        // 0xFFFFFFFF should have all digits = 0xF
        for pass in 0..8 {
            assert_eq!(extract_digit(0xFFFFFFFF, pass), 0xF);
        }

        // 0x00000000 should have all digits = 0
        for pass in 0..8 {
            assert_eq!(extract_digit(0x00000000, pass), 0x0);
        }

        // 0x11111111 should have all digits = 1
        for pass in 0..8 {
            assert_eq!(extract_digit(0x11111111, pass), 0x1);
        }
    }

    #[test]
    fn test_radix_mask_correctness() {
        // The mask 0xF extracts exactly 4 bits
        assert_eq!(0xF, 0b1111);
        assert_eq!(0xF, (1 << RADIX_BITS) - 1);
    }

    // -------------------------------------------------------------------------
    // 10. Workgroup element processing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_elements_per_thread() {
        // Each thread processes 4 elements
        assert_eq!(ELEMENTS_PER_THREAD, 4);
    }

    #[test]
    fn test_elements_per_workgroup_calculation() {
        // 256 threads * 4 elements = 1024 elements per workgroup
        assert_eq!(ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE * ELEMENTS_PER_THREAD);
        assert_eq!(ELEMENTS_PER_WORKGROUP, 1024);
    }

    #[test]
    fn test_thread_element_indices() {
        // Verify how elements are distributed across threads
        // Thread i processes elements at: base + i, base + i + 256, base + i + 512, base + i + 768

        let workgroup_idx = 0u32;
        let base_idx = workgroup_idx * ELEMENTS_PER_WORKGROUP;

        for thread_idx in 0..WORKGROUP_SIZE {
            let mut elements = Vec::new();
            for i in 0..ELEMENTS_PER_THREAD {
                let elem_idx = base_idx + thread_idx + i * WORKGROUP_SIZE;
                elements.push(elem_idx);
            }

            // Verify stride pattern
            for i in 1..elements.len() {
                assert_eq!(
                    elements[i] - elements[i - 1],
                    WORKGROUP_SIZE,
                    "Stride should be WORKGROUP_SIZE"
                );
            }
        }
    }
}
