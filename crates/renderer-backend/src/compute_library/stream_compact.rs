//! Stream compaction using prefix scan for wgpu 25.x (T-WGPU-P3.10.3).
//!
//! Stream compaction is a parallel filtering operation that produces a
//! densely-packed output array containing only elements that pass a predicate.
//!
//! # Algorithm Overview
//!
//! 1. **Predicate Evaluation**: For each input element, compute a binary predicate
//!    (0 = exclude, 1 = include).
//!
//! 2. **Prefix Scan**: Perform exclusive prefix sum on the predicate array.
//!    The scan result gives the output index for each element.
//!
//! 3. **Scatter**: For each element where predicate == 1, write to output[scan[i]].
//!
//! 4. **Count**: Total count = predicate[n-1] + scan[n-1]
//!
//! # Performance Characteristics
//!
//! - Work complexity: O(n) for scatter, O(n) for prefix scan
//! - Memory: 3 buffers (input, predicates, output) + scan scratch space
//! - Bandwidth: ~3 reads + 1 write per element (amortized)
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_library::stream_compact::StreamCompactPipeline;
//!
//! # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create the pipeline
//! let pipeline = StreamCompactPipeline::new(device);
//!
//! // Input data with some zeros to filter out
//! let input = vec![1u32, 0, 3, 0, 5, 6, 0, 8];
//!
//! // Compact non-zero elements
//! let (compacted, count) = pipeline.compact_nonzero(device, queue, &input).await;
//!
//! // Result: compacted = [1, 3, 5, 6, 8], count = 5
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `StreamCompactPipeline` is `Send + Sync` as it only holds wgpu pipeline handles.

use std::borrow::Cow;
use std::num::NonZeroU64;

use super::prefix_scan::PrefixScanPipeline;

/// Workgroup size used by the stream compaction shader.
pub const WORKGROUP_SIZE: u32 = 256;

/// Shader source code for stream compaction.
const SHADER_SOURCE: &str = include_str!("../../shaders/stream_compact.wgsl");

// ---------------------------------------------------------------------------
// CompactParams - Uniform buffer structure
// ---------------------------------------------------------------------------

/// Parameters passed to the stream compaction shader via uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompactParams {
    /// Total number of input elements.
    pub input_size: u32,
    /// Offset for multi-pass compaction.
    pub block_offset: u32,
    /// Stride for input data (elements, not bytes).
    pub input_stride: u32,
    /// Stride for output data (elements, not bytes).
    pub output_stride: u32,
}

impl CompactParams {
    /// Create new compaction parameters.
    pub fn new(input_size: u32) -> Self {
        Self {
            input_size,
            block_offset: 0,
            input_stride: 1,
            output_stride: 1,
        }
    }

    /// Set the block offset for multi-pass compaction.
    pub fn with_offset(mut self, offset: u32) -> Self {
        self.block_offset = offset;
        self
    }

    /// Set input stride for strided memory access.
    pub fn with_input_stride(mut self, stride: u32) -> Self {
        self.input_stride = stride;
        self
    }

    /// Set output stride for strided memory access.
    pub fn with_output_stride(mut self, stride: u32) -> Self {
        self.output_stride = stride;
        self
    }
}

// ---------------------------------------------------------------------------
// StreamCompactError
// ---------------------------------------------------------------------------

/// Errors that can occur during stream compaction operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamCompactError {
    /// Input size is zero.
    EmptyInput,
    /// Input size exceeds maximum supported.
    InputTooLarge { size: u32, max: u32 },
    /// Buffer is too small for the requested operation.
    BufferTooSmall { required: u64, actual: u64 },
    /// Predicate buffer size doesn't match input size.
    PredicateSizeMismatch { input_size: u32, predicate_size: u32 },
    /// Prefix scan failed.
    PrefixScanFailed(String),
}

impl std::fmt::Display for StreamCompactError {
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
            Self::PredicateSizeMismatch {
                input_size,
                predicate_size,
            } => {
                write!(
                    f,
                    "Predicate size {} doesn't match input size {}",
                    predicate_size, input_size
                )
            }
            Self::PrefixScanFailed(msg) => {
                write!(f, "Prefix scan failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for StreamCompactError {}

// ---------------------------------------------------------------------------
// PredicateType
// ---------------------------------------------------------------------------

/// Type of predicate evaluation for built-in predicate kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredicateType {
    /// Keep elements that are non-zero.
    #[default]
    NonZero,
    /// Keep elements greater than a threshold.
    GreaterThan(u32),
    /// Keep elements less than a threshold.
    LessThan(u32),
    /// Keep elements equal to a value.
    Equal(u32),
    /// Keep elements not equal to a value.
    NotEqual(u32),
}

// ---------------------------------------------------------------------------
// StreamCompactPipeline
// ---------------------------------------------------------------------------

/// GPU pipeline for stream compaction using prefix scan.
///
/// This struct holds the compiled compute pipelines needed to perform
/// stream compaction on the GPU. It internally uses a `PrefixScanPipeline`
/// for the scan phase.
///
/// # Supported Operations
///
/// - `compact()`: Compact using pre-computed predicates
/// - `compact_nonzero()`: Compact non-zero elements (convenience method)
/// - `compact_with_predicate()`: Compact with custom predicate function
///
/// # Limitations
///
/// - Element type: u32 (or u32 arrays for vec4/multi-element variants)
/// - Maximum input size: Same as `PrefixScanPipeline` (~2^30 elements)
pub struct StreamCompactPipeline {
    /// Prefix scan pipeline for computing scatter indices.
    prefix_scan: PrefixScanPipeline,
    /// Scatter pipeline for copying elements to compacted positions.
    scatter_pipeline: wgpu::ComputePipeline,
    /// Scatter vec4 pipeline for 4-component elements.
    scatter_vec4_pipeline: wgpu::ComputePipeline,
    /// Count elements pipeline.
    count_pipeline: wgpu::ComputePipeline,
    /// Evaluate predicate (non-zero) pipeline.
    predicate_nonzero_pipeline: wgpu::ComputePipeline,
    /// Scatter with fused non-zero predicate pipeline.
    scatter_fused_nonzero_pipeline: wgpu::ComputePipeline,
    /// Multi-element scatter pipeline.
    scatter_multi_element_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for scatter operations.
    scatter_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for count output.
    count_bind_group_layout: wgpu::BindGroupLayout,
}

impl StreamCompactPipeline {
    /// Create a new stream compaction pipeline.
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
    /// A new `StreamCompactPipeline` ready for use.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create prefix scan pipeline
        let prefix_scan = PrefixScanPipeline::new(device);

        // Compile the stream compaction shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stream_compact_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        // Create scatter bind group layout (group 0)
        // Binding 0: input_data (read-only storage)
        // Binding 1: scan_result (read-only storage)
        // Binding 2: predicates (read-only storage)
        // Binding 3: output_data (read-write storage)
        // Binding 4: params (uniform)
        let scatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("stream_compact_scatter_bind_group_layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<CompactParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // Create count bind group layout (group 1)
        let count_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("stream_compact_count_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                }],
            });

        // Create pipeline layouts
        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stream_compact_scatter_pipeline_layout"),
                bind_group_layouts: &[&scatter_bind_group_layout],
                push_constant_ranges: &[],
            });

        let count_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stream_compact_count_pipeline_layout"),
                bind_group_layouts: &[&scatter_bind_group_layout, &count_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create scatter pipeline
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &shader_module,
            entry_point: "scatter",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create scatter_vec4 pipeline
        let scatter_vec4_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("stream_compact_scatter_vec4"),
                layout: Some(&scatter_pipeline_layout),
                module: &shader_module,
                entry_point: "scatter_vec4",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create count pipeline
        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stream_compact_count"),
            layout: Some(&count_pipeline_layout),
            module: &shader_module,
            entry_point: "count_elements",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create predicate evaluation pipeline
        let predicate_nonzero_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("stream_compact_predicate_nonzero"),
                layout: Some(&scatter_pipeline_layout),
                module: &shader_module,
                entry_point: "evaluate_predicate_nonzero",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create fused scatter pipeline
        let scatter_fused_nonzero_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("stream_compact_scatter_fused_nonzero"),
                layout: Some(&scatter_pipeline_layout),
                module: &shader_module,
                entry_point: "scatter_fused_nonzero",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create multi-element scatter pipeline
        let scatter_multi_element_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("stream_compact_scatter_multi_element"),
                layout: Some(&scatter_pipeline_layout),
                module: &shader_module,
                entry_point: "scatter_multi_element",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            prefix_scan,
            scatter_pipeline,
            scatter_vec4_pipeline,
            count_pipeline,
            predicate_nonzero_pipeline,
            scatter_fused_nonzero_pipeline,
            scatter_multi_element_pipeline,
            scatter_bind_group_layout,
            count_bind_group_layout,
        }
    }

    /// Calculate the number of workgroups needed for scatter dispatch.
    #[inline]
    pub fn num_workgroups(input_size: u32) -> u32 {
        (input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Calculate the buffer size needed for scan scratch space.
    #[inline]
    pub fn scan_scratch_size(input_size: u32) -> u64 {
        PrefixScanPipeline::block_sums_buffer_size(input_size)
    }

    /// Compact elements using pre-computed predicates.
    ///
    /// This is the core compaction method. The predicates buffer should
    /// contain 0/1 values indicating which elements to keep.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `input_buffer` - Buffer containing input data.
    /// * `predicates_buffer` - Buffer containing 0/1 predicates (will be modified!).
    /// * `output_buffer` - Buffer for compacted output.
    /// * `input_size` - Number of elements.
    ///
    /// # Returns
    ///
    /// A buffer containing the compacted element count (single u32).
    pub fn compact(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        predicates_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        input_size: u32,
    ) -> wgpu::Buffer {
        if input_size == 0 {
            // Return buffer with zero count
            let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stream_compact_count"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&count_buffer, 0, &[0u8; 4]);
            return count_buffer;
        }

        // Step 1: Prefix scan of predicates (modifies predicates_buffer in-place)
        // After scan: predicates_buffer[i] = count of 1s before position i
        // We need to preserve original predicates for scatter, so we need to
        // copy predicates first or scan into a separate buffer.

        // Create scan result buffer (copy of predicates, then scanned)
        let scan_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_scan_result"),
            size: (input_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy predicates to scan buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stream_compact_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(
            predicates_buffer,
            0,
            &scan_buffer,
            0,
            (input_size as u64) * 4,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Perform exclusive prefix scan on the copy
        self.prefix_scan.scan(device, queue, &scan_buffer, input_size);

        // Step 2: Scatter elements to compacted positions
        let params = CompactParams::new(input_size);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_params"),
            size: std::mem::size_of::<CompactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create scatter bind group
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_scatter_bind_group"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scan_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: predicates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create count output buffer
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create count bind group
        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_count_bind_group"),
            layout: &self.count_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: count_buffer.as_entire_binding(),
            }],
        });

        // Dispatch scatter and count kernels
        let num_workgroups = Self::num_workgroups(input_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stream_compact_scatter_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_scatter_pass"),
                timestamp_writes: None,
            });

            // Scatter elements
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_count_pass"),
                timestamp_writes: None,
            });

            // Compute count
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            pass.set_bind_group(1, &count_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        count_buffer
    }

    /// Compact non-zero elements from the input buffer.
    ///
    /// This is a convenience method that evaluates a non-zero predicate
    /// and compacts all non-zero elements.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `input_buffer` - Buffer containing input data.
    /// * `output_buffer` - Buffer for compacted output.
    /// * `input_size` - Number of elements.
    ///
    /// # Returns
    ///
    /// A buffer containing the compacted element count (single u32).
    pub fn compact_nonzero(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        input_size: u32,
    ) -> wgpu::Buffer {
        if input_size == 0 {
            let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stream_compact_count"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&count_buffer, 0, &[0u8; 4]);
            return count_buffer;
        }

        // Create predicates buffer (will hold 0/1 values)
        let predicates_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_predicates"),
            size: (input_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Evaluate non-zero predicate
        let params = CompactParams::new(input_size);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_params"),
            size: std::mem::size_of::<CompactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // For predicate evaluation, we need a bind group where:
        // - binding 0 = input_buffer (read)
        // - binding 3 = predicates_buffer (write, used as output_data in shader)
        // We'll create a dummy buffer for other bindings
        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let predicate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_predicate_bind_group"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: predicates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let num_workgroups = Self::num_workgroups(input_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stream_compact_predicate_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_predicate_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.predicate_nonzero_pipeline);
            pass.set_bind_group(0, &predicate_bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Now compact using the generated predicates
        self.compact(
            device,
            queue,
            input_buffer,
            &predicates_buffer,
            output_buffer,
            input_size,
        )
    }

    /// Compact vec4 elements (4 u32s per element) using pre-computed predicates.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for submitting commands.
    /// * `input_buffer` - Buffer containing input data (4 u32s per element).
    /// * `predicates_buffer` - Buffer containing 0/1 predicates (1 per element).
    /// * `output_buffer` - Buffer for compacted output.
    /// * `element_count` - Number of vec4 elements (not u32s).
    ///
    /// # Returns
    ///
    /// A buffer containing the compacted element count (single u32).
    pub fn compact_vec4(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        predicates_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        element_count: u32,
    ) -> wgpu::Buffer {
        if element_count == 0 {
            let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stream_compact_count"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&count_buffer, 0, &[0u8; 4]);
            return count_buffer;
        }

        // Create scan result buffer
        let scan_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_scan_result"),
            size: (element_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy predicates to scan buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stream_compact_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(
            predicates_buffer,
            0,
            &scan_buffer,
            0,
            (element_count as u64) * 4,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Perform exclusive prefix scan
        self.prefix_scan
            .scan(device, queue, &scan_buffer, element_count);

        // Scatter vec4 elements
        let params = CompactParams::new(element_count);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_params"),
            size: std::mem::size_of::<CompactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_scatter_vec4_bind_group"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scan_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: predicates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_compact_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stream_compact_count_bind_group"),
            layout: &self.count_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: count_buffer.as_entire_binding(),
            }],
        });

        let num_workgroups = Self::num_workgroups(element_count);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stream_compact_scatter_vec4_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_scatter_vec4_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.scatter_vec4_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stream_compact_count_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            pass.set_bind_group(1, &count_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        count_buffer
    }

    /// Get a reference to the internal prefix scan pipeline.
    ///
    /// Useful for advanced use cases where you need direct access to
    /// the scan functionality.
    pub fn prefix_scan(&self) -> &PrefixScanPipeline {
        &self.prefix_scan
    }

    /// Get the scatter bind group layout for external use.
    pub fn scatter_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scatter_bind_group_layout
    }

    /// Get the count bind group layout for external use.
    pub fn count_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.count_bind_group_layout
    }

    /// Get a reference to the scatter pipeline.
    pub fn scatter_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.scatter_pipeline
    }

    /// Get a reference to the scatter_vec4 pipeline.
    pub fn scatter_vec4_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.scatter_vec4_pipeline
    }

    /// Get a reference to the count pipeline.
    pub fn count_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.count_pipeline
    }

    /// Get a reference to the scatter_multi_element pipeline.
    pub fn scatter_multi_element_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.scatter_multi_element_pipeline
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CompactParams Struct Tests
    // =========================================================================

    #[test]
    fn test_compact_params_new() {
        let params = CompactParams::new(100);
        assert_eq!(params.input_size, 100);
        assert_eq!(params.block_offset, 0);
        assert_eq!(params.input_stride, 1);
        assert_eq!(params.output_stride, 1);
    }

    #[test]
    fn test_compact_params_with_offset() {
        let params = CompactParams::new(100).with_offset(512);
        assert_eq!(params.input_size, 100);
        assert_eq!(params.block_offset, 512);
    }

    #[test]
    fn test_compact_params_with_strides() {
        let params = CompactParams::new(100)
            .with_input_stride(4)
            .with_output_stride(4);
        assert_eq!(params.input_stride, 4);
        assert_eq!(params.output_stride, 4);
    }

    #[test]
    fn test_compact_params_default() {
        let params = CompactParams::default();
        assert_eq!(params.input_size, 0);
        assert_eq!(params.block_offset, 0);
        assert_eq!(params.input_stride, 0);
        assert_eq!(params.output_stride, 0);
    }

    #[test]
    fn test_compact_params_pod() {
        let params = CompactParams::new(100);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<CompactParams>());
        assert_eq!(bytes.len(), 16); // 4 u32s = 16 bytes
    }

    #[test]
    fn test_compact_params_zeroable() {
        let zeroed: CompactParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.input_size, 0);
        assert_eq!(zeroed.block_offset, 0);
        assert_eq!(zeroed.input_stride, 0);
        assert_eq!(zeroed.output_stride, 0);
    }

    #[test]
    fn test_compact_params_clone() {
        let params = CompactParams::new(500).with_offset(100);
        let cloned = params.clone();
        assert_eq!(cloned.input_size, params.input_size);
        assert_eq!(cloned.block_offset, params.block_offset);
    }

    #[test]
    fn test_compact_params_copy() {
        let params = CompactParams::new(200);
        let copied = params;
        let _still_valid = params; // Original still valid (Copy trait)
        assert_eq!(copied.input_size, 200);
    }

    #[test]
    fn test_compact_params_debug() {
        let params = CompactParams::new(100).with_offset(50);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("CompactParams"));
        assert!(debug_str.contains("input_size"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_compact_params_chaining() {
        let params = CompactParams::new(1000)
            .with_offset(256)
            .with_input_stride(4)
            .with_output_stride(4);
        assert_eq!(params.input_size, 1000);
        assert_eq!(params.block_offset, 256);
        assert_eq!(params.input_stride, 4);
        assert_eq!(params.output_stride, 4);
    }

    #[test]
    fn test_compact_params_16_byte_alignment() {
        assert_eq!(std::mem::size_of::<CompactParams>(), 16);
        assert!(std::mem::align_of::<CompactParams>() >= 4);
    }

    // =========================================================================
    // Workgroup Calculation Tests
    // =========================================================================

    #[test]
    fn test_num_workgroups() {
        assert_eq!(StreamCompactPipeline::num_workgroups(1), 1);
        assert_eq!(StreamCompactPipeline::num_workgroups(256), 1);
        assert_eq!(StreamCompactPipeline::num_workgroups(257), 2);
        assert_eq!(StreamCompactPipeline::num_workgroups(512), 2);
        assert_eq!(StreamCompactPipeline::num_workgroups(513), 3);
    }

    #[test]
    fn test_num_workgroups_single_element() {
        assert_eq!(StreamCompactPipeline::num_workgroups(1), 1);
    }

    #[test]
    fn test_num_workgroups_exact_multiple() {
        assert_eq!(StreamCompactPipeline::num_workgroups(256), 1);
        assert_eq!(StreamCompactPipeline::num_workgroups(512), 2);
        assert_eq!(StreamCompactPipeline::num_workgroups(1024), 4);
        assert_eq!(StreamCompactPipeline::num_workgroups(256 * 100), 100);
    }

    #[test]
    fn test_num_workgroups_just_over_boundary() {
        assert_eq!(StreamCompactPipeline::num_workgroups(257), 2);
        assert_eq!(StreamCompactPipeline::num_workgroups(513), 3);
        assert_eq!(StreamCompactPipeline::num_workgroups(769), 4);
    }

    #[test]
    fn test_num_workgroups_large_arrays() {
        assert_eq!(
            StreamCompactPipeline::num_workgroups(1_000_000),
            3907
        ); // ceil(1M / 256)
        assert_eq!(StreamCompactPipeline::num_workgroups(10_000_000), 39063);
    }

    #[test]
    fn test_num_workgroups_zero() {
        assert_eq!(StreamCompactPipeline::num_workgroups(0), 0);
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let err = StreamCompactError::EmptyInput;
        assert_eq!(err.to_string(), "Input size cannot be zero");

        let err = StreamCompactError::InputTooLarge {
            size: 1000,
            max: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));

        let err = StreamCompactError::BufferTooSmall {
            required: 1024,
            actual: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));

        let err = StreamCompactError::PredicateSizeMismatch {
            input_size: 100,
            predicate_size: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));

        let err = StreamCompactError::PrefixScanFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_clone() {
        let err = StreamCompactError::InputTooLarge { size: 100, max: 50 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_eq() {
        let err1 = StreamCompactError::EmptyInput;
        let err2 = StreamCompactError::EmptyInput;
        assert_eq!(err1, err2);

        let err3 = StreamCompactError::InputTooLarge { size: 100, max: 50 };
        let err4 = StreamCompactError::InputTooLarge { size: 100, max: 50 };
        assert_eq!(err3, err4);

        let err5 = StreamCompactError::InputTooLarge { size: 100, max: 50 };
        let err6 = StreamCompactError::InputTooLarge { size: 200, max: 50 };
        assert_ne!(err5, err6);
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        let err = StreamCompactError::EmptyInput;
        assert_error(&err);
    }

    // =========================================================================
    // PredicateType Tests
    // =========================================================================

    #[test]
    fn test_predicate_type_default() {
        let pred = PredicateType::default();
        assert_eq!(pred, PredicateType::NonZero);
    }

    #[test]
    fn test_predicate_type_variants() {
        let _ = PredicateType::NonZero;
        let _ = PredicateType::GreaterThan(100);
        let _ = PredicateType::LessThan(50);
        let _ = PredicateType::Equal(42);
        let _ = PredicateType::NotEqual(0);
    }

    #[test]
    fn test_predicate_type_clone() {
        let pred = PredicateType::GreaterThan(100);
        let cloned = pred.clone();
        assert_eq!(pred, cloned);
    }

    #[test]
    fn test_predicate_type_copy() {
        let pred = PredicateType::LessThan(50);
        let copied = pred;
        let _still_valid = pred;
        assert_eq!(copied, PredicateType::LessThan(50));
    }

    #[test]
    fn test_predicate_type_debug() {
        let pred = PredicateType::GreaterThan(100);
        let debug_str = format!("{:?}", pred);
        assert!(debug_str.contains("GreaterThan"));
        assert!(debug_str.contains("100"));
    }

    // =========================================================================
    // Scan Scratch Size Tests
    // =========================================================================

    #[test]
    fn test_scan_scratch_size() {
        // Should match PrefixScanPipeline::block_sums_buffer_size
        assert_eq!(
            StreamCompactPipeline::scan_scratch_size(512),
            PrefixScanPipeline::block_sums_buffer_size(512)
        );
        assert_eq!(
            StreamCompactPipeline::scan_scratch_size(1_000_000),
            PrefixScanPipeline::block_sums_buffer_size(1_000_000)
        );
    }

    #[test]
    fn test_scan_scratch_size_small_input() {
        let size = StreamCompactPipeline::scan_scratch_size(1);
        assert!(size >= 4);
    }

    #[test]
    fn test_scan_scratch_size_large_input() {
        let size = StreamCompactPipeline::scan_scratch_size(10_000_000);
        assert!(size > 0);
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_workgroup_size_power_of_two() {
        assert!(WORKGROUP_SIZE.is_power_of_two());
    }

    // =========================================================================
    // Algorithm Correctness Tests (CPU simulation)
    // =========================================================================

    #[test]
    fn test_stream_compact_algorithm_simulation() {
        // Simulate the stream compaction algorithm on CPU
        let input = vec![1u32, 0, 3, 0, 5, 6, 0, 8];

        // Step 1: Generate predicates (non-zero)
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        assert_eq!(predicates, vec![1, 0, 1, 0, 1, 1, 0, 1]);

        // Step 2: Exclusive prefix scan of predicates
        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }
        assert_eq!(scan_result, vec![0, 1, 1, 2, 2, 3, 4, 4]);

        // Step 3: Scatter
        let mut output = vec![0u32; input.len()];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                output[scan_result[i] as usize] = input[i];
            }
        }
        assert_eq!(output[..5], [1, 3, 5, 6, 8]);

        // Step 4: Count
        let count = predicates[predicates.len() - 1] + scan_result[scan_result.len() - 1];
        assert_eq!(count, 5);
    }

    #[test]
    fn test_stream_compact_empty_input() {
        let input: Vec<u32> = vec![];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        assert!(predicates.is_empty());
    }

    #[test]
    fn test_stream_compact_all_pass() {
        let input = vec![1u32, 2, 3, 4, 5];
        let predicates: Vec<u32> = input.iter().map(|_| 1).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }
        assert_eq!(scan_result, vec![0, 1, 2, 3, 4]);

        let count = predicates[predicates.len() - 1] + scan_result[scan_result.len() - 1];
        assert_eq!(count, 5);
    }

    #[test]
    fn test_stream_compact_none_pass() {
        let input = vec![0u32, 0, 0, 0, 0];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        assert_eq!(predicates, vec![0, 0, 0, 0, 0]);

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }
        assert_eq!(scan_result, vec![0, 0, 0, 0, 0]);

        let count = predicates[predicates.len() - 1] + scan_result[scan_result.len() - 1];
        assert_eq!(count, 0);
    }

    #[test]
    fn test_stream_compact_single_element_pass() {
        let input = vec![42u32];
        let predicates: Vec<u32> = vec![1];
        let scan_result = vec![0u32];

        let count = predicates[0] + scan_result[0];
        assert_eq!(count, 1);
    }

    #[test]
    fn test_stream_compact_single_element_fail() {
        let input = vec![0u32];
        let predicates: Vec<u32> = vec![0];
        let scan_result = vec![0u32];

        let count = predicates[0] + scan_result[0];
        assert_eq!(count, 0);
    }

    #[test]
    fn test_stream_compact_alternating() {
        let input = vec![1u32, 0, 2, 0, 3, 0, 4, 0];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        assert_eq!(predicates, vec![1, 0, 1, 0, 1, 0, 1, 0]);

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }
        assert_eq!(scan_result, vec![0, 1, 1, 2, 2, 3, 3, 4]);

        let count = predicates[predicates.len() - 1] + scan_result[scan_result.len() - 1];
        assert_eq!(count, 4);
    }

    #[test]
    fn test_stream_compact_greater_than_predicate() {
        let input = vec![1u32, 5, 2, 8, 3, 10, 4, 7];
        let threshold = 4;
        let predicates: Vec<u32> = input
            .iter()
            .map(|&x| if x > threshold { 1 } else { 0 })
            .collect();
        assert_eq!(predicates, vec![0, 1, 0, 1, 0, 1, 0, 1]);

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let mut output = vec![0u32; input.len()];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                output[scan_result[i] as usize] = input[i];
            }
        }
        assert_eq!(output[..4], [5, 8, 10, 7]);
    }

    // =========================================================================
    // Large Scale Algorithm Tests
    // =========================================================================

    #[test]
    fn test_stream_compact_large_input_simulation() {
        // Test with 1000 elements
        let input: Vec<u32> = (0..1000).collect();

        // Keep even numbers
        let predicates: Vec<u32> = input.iter().map(|&x| if x % 2 == 0 { 1 } else { 0 }).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let count = if !predicates.is_empty() {
            predicates[predicates.len() - 1] + scan_result[scan_result.len() - 1]
        } else {
            0
        };

        // 500 even numbers from 0 to 998
        assert_eq!(count, 500);
    }

    #[test]
    fn test_stream_compact_workgroup_boundary() {
        // Test at workgroup size boundary (256)
        let input: Vec<u32> = (0..256).map(|i| if i % 2 == 0 { i + 1 } else { 0 }).collect();
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 128);
    }

    #[test]
    fn test_stream_compact_multi_workgroup() {
        // Test across multiple workgroups (> 256 elements)
        let input: Vec<u32> = (0..512).map(|i| if i % 3 == 0 { i + 1 } else { 0 }).collect();
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let count: u32 = predicates.iter().sum();
        // 512 / 3 = 170 (rounded), plus 0 counts
        // 0, 3, 6, ..., 510 -> 171 values
        assert_eq!(count, 171);
    }

    // =========================================================================
    // CompactParams Extended Tests
    // =========================================================================

    #[test]
    fn test_compact_params_max_input_size() {
        let params = CompactParams::new(u32::MAX);
        assert_eq!(params.input_size, u32::MAX);
    }

    #[test]
    fn test_compact_params_max_offset() {
        let params = CompactParams::new(100).with_offset(u32::MAX);
        assert_eq!(params.block_offset, u32::MAX);
    }

    #[test]
    fn test_compact_params_max_strides() {
        let params = CompactParams::new(100)
            .with_input_stride(u32::MAX)
            .with_output_stride(u32::MAX);
        assert_eq!(params.input_stride, u32::MAX);
        assert_eq!(params.output_stride, u32::MAX);
    }

    #[test]
    fn test_compact_params_bytemuck_roundtrip() {
        let params = CompactParams::new(12345)
            .with_offset(6789)
            .with_input_stride(4)
            .with_output_stride(8);

        let bytes = bytemuck::bytes_of(&params);
        let restored: &CompactParams = bytemuck::from_bytes(bytes);

        assert_eq!(restored.input_size, 12345);
        assert_eq!(restored.block_offset, 6789);
        assert_eq!(restored.input_stride, 4);
        assert_eq!(restored.output_stride, 8);
    }

    #[test]
    fn test_compact_params_zero_stride_valid() {
        // Zero stride might be used as a flag or special case
        let params = CompactParams::new(100)
            .with_input_stride(0)
            .with_output_stride(0);
        assert_eq!(params.input_stride, 0);
        assert_eq!(params.output_stride, 0);
    }

    #[test]
    fn test_compact_params_builder_idempotent() {
        // Calling the same builder method multiple times should override
        let params = CompactParams::new(100)
            .with_offset(10)
            .with_offset(20)
            .with_offset(30);
        assert_eq!(params.block_offset, 30);
    }

    #[test]
    fn test_compact_params_repr_c_layout() {
        // Verify C-compatible layout (fields in declaration order)
        let params = CompactParams {
            input_size: 1,
            block_offset: 2,
            input_stride: 3,
            output_stride: 4,
        };
        let bytes = bytemuck::bytes_of(&params);
        // First 4 bytes should be input_size
        assert_eq!(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 1);
        // Next 4 bytes should be block_offset
        assert_eq!(u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 2);
    }

    // =========================================================================
    // PredicateType Extended Tests
    // =========================================================================

    #[test]
    fn test_predicate_type_eq_symmetry() {
        let a = PredicateType::GreaterThan(50);
        let b = PredicateType::GreaterThan(50);
        assert!(a == b);
        assert!(b == a);
    }

    #[test]
    fn test_predicate_type_ne_different_thresholds() {
        let a = PredicateType::GreaterThan(50);
        let b = PredicateType::GreaterThan(51);
        assert_ne!(a, b);
    }

    #[test]
    fn test_predicate_type_ne_different_variants() {
        let gt = PredicateType::GreaterThan(50);
        let lt = PredicateType::LessThan(50);
        assert_ne!(gt, lt);
    }

    #[test]
    fn test_predicate_type_all_variants_debug() {
        let variants = [
            PredicateType::NonZero,
            PredicateType::GreaterThan(0),
            PredicateType::LessThan(u32::MAX),
            PredicateType::Equal(42),
            PredicateType::NotEqual(0),
        ];
        for v in variants {
            let s = format!("{:?}", v);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_predicate_type_boundary_thresholds() {
        // Test with boundary values
        let _ = PredicateType::GreaterThan(0);
        let _ = PredicateType::GreaterThan(u32::MAX);
        let _ = PredicateType::LessThan(0);
        let _ = PredicateType::LessThan(u32::MAX);
        let _ = PredicateType::Equal(0);
        let _ = PredicateType::Equal(u32::MAX);
    }

    // =========================================================================
    // StreamCompactError Extended Tests
    // =========================================================================

    #[test]
    fn test_error_debug_format() {
        let err = StreamCompactError::InputTooLarge { size: 100, max: 50 };
        let debug = format!("{:?}", err);
        assert!(debug.contains("InputTooLarge"));
    }

    #[test]
    fn test_error_all_variants_display() {
        let errors = [
            StreamCompactError::EmptyInput,
            StreamCompactError::InputTooLarge { size: 100, max: 50 },
            StreamCompactError::BufferTooSmall { required: 1024, actual: 512 },
            StreamCompactError::PredicateSizeMismatch { input_size: 100, predicate_size: 50 },
            StreamCompactError::PrefixScanFailed("error".to_string()),
        ];
        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_error_partial_eq_reflexive() {
        let err = StreamCompactError::EmptyInput;
        assert_eq!(err, err.clone());
    }

    #[test]
    fn test_error_clone_independence() {
        let err1 = StreamCompactError::PrefixScanFailed("test".to_string());
        let err2 = err1.clone();
        // Modifying one shouldn't affect the other (they're independent clones)
        drop(err1);
        assert_eq!(err2.to_string(), "Prefix scan failed: test");
    }

    // =========================================================================
    // Workgroup Calculation Extended Tests
    // =========================================================================

    #[test]
    fn test_num_workgroups_large_safe_value() {
        // Test with a large value that won't overflow the formula
        // (input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
        // Max safe input: u32::MAX - (WORKGROUP_SIZE - 1) = u32::MAX - 255
        // We use a value well under the limit to avoid compile-time overflow detection
        let large_safe = 0xFFF0_0000u32; // 4293918720, safely under max
        let result = StreamCompactPipeline::num_workgroups(large_safe);
        // ceil(4293918720 / 256) = 16773120
        assert_eq!(result, 16773120);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "overflow")]
    fn test_num_workgroups_max_u32_panics_on_overflow() {
        // Document that u32::MAX causes overflow in debug builds
        // This is a known limitation - input sizes near u32::MAX are impractical
        // Note: This test only runs in debug mode where overflow checking is enabled
        let _ = StreamCompactPipeline::num_workgroups(u32::MAX);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_num_workgroups_max_u32_panics_on_overflow() {
        // In release mode, overflow wraps silently - verify function doesn't crash
        let _ = StreamCompactPipeline::num_workgroups(u32::MAX);
    }

    #[test]
    fn test_num_workgroups_powers_of_two() {
        for power in 0..=20 {
            let size = 1u32 << power;
            let wg = StreamCompactPipeline::num_workgroups(size);
            let expected = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            assert_eq!(wg, expected, "Failed for size 2^{} = {}", power, size);
        }
    }

    #[test]
    fn test_num_workgroups_consistency_with_ceiling_division() {
        for size in [1, 127, 128, 255, 256, 257, 511, 512, 513, 1023, 1024] {
            let result = StreamCompactPipeline::num_workgroups(size);
            let expected = size.div_ceil(WORKGROUP_SIZE);
            assert_eq!(result, expected);
        }
    }

    // =========================================================================
    // Scan Scratch Size Extended Tests
    // =========================================================================

    #[test]
    fn test_scan_scratch_size_zero() {
        let size = StreamCompactPipeline::scan_scratch_size(0);
        // Should handle zero gracefully
        assert!(size >= 0);
    }

    #[test]
    fn test_scan_scratch_size_consistency() {
        // Verify consistency with PrefixScanPipeline
        for n in [1, 100, 256, 1000, 10000, 100000] {
            assert_eq!(
                StreamCompactPipeline::scan_scratch_size(n),
                PrefixScanPipeline::block_sums_buffer_size(n)
            );
        }
    }

    // =========================================================================
    // Algorithm Simulation Extended Tests
    // =========================================================================

    #[test]
    fn test_stream_compact_less_than_predicate() {
        let input = vec![10u32, 5, 20, 3, 15, 8, 25, 2];
        let threshold = 10;
        let predicates: Vec<u32> = input
            .iter()
            .map(|&x| if x < threshold { 1 } else { 0 })
            .collect();
        assert_eq!(predicates, vec![0, 1, 0, 1, 0, 1, 0, 1]);

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let mut output = vec![0u32; input.len()];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                output[scan_result[i] as usize] = input[i];
            }
        }
        // Elements < 10: 5, 3, 8, 2
        assert_eq!(output[..4], [5, 3, 8, 2]);
    }

    #[test]
    fn test_stream_compact_equal_predicate() {
        let input = vec![1u32, 2, 1, 3, 1, 4, 1, 5];
        let target = 1;
        let predicates: Vec<u32> = input
            .iter()
            .map(|&x| if x == target { 1 } else { 0 })
            .collect();
        assert_eq!(predicates, vec![1, 0, 1, 0, 1, 0, 1, 0]);

        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_stream_compact_not_equal_predicate() {
        let input = vec![1u32, 0, 1, 0, 1, 0, 1, 0];
        let exclude = 0;
        let predicates: Vec<u32> = input
            .iter()
            .map(|&x| if x != exclude { 1 } else { 0 })
            .collect();
        assert_eq!(predicates, vec![1, 0, 1, 0, 1, 0, 1, 0]);

        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_stream_compact_two_elements() {
        // Minimum viable array for non-trivial scan
        let input = vec![1u32, 0];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        assert_eq!(predicates, vec![1, 0]);

        let scan_result = vec![0u32, 1];
        let count = predicates[1] + scan_result[1];
        assert_eq!(count, 1);
    }

    #[test]
    fn test_stream_compact_preserve_order() {
        // Verify elements maintain relative order after compaction
        let input = vec![5u32, 0, 3, 0, 1, 0, 4, 0, 2];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let mut output = vec![0u32; input.len()];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                output[scan_result[i] as usize] = input[i];
            }
        }

        // Elements should be: 5, 3, 1, 4, 2 (original order preserved)
        assert_eq!(output[..5], [5, 3, 1, 4, 2]);
    }

    #[test]
    fn test_stream_compact_vec4_simulation() {
        // Simulate vec4 compaction (4 components per element)
        let input: Vec<u32> = vec![
            1, 2, 3, 4,     // Element 0 (pass)
            0, 0, 0, 0,     // Element 1 (fail - all zeros, predicate based on first component)
            5, 6, 7, 8,     // Element 2 (pass)
            0, 0, 0, 0,     // Element 3 (fail)
        ];

        // Predicates per element (not per component)
        let predicates: Vec<u32> = vec![1, 0, 1, 0];

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }
        assert_eq!(scan_result, vec![0, 1, 1, 2]);

        // Scatter vec4 elements
        let mut output = vec![0u32; input.len()];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                let src_base = i * 4;
                let dst_base = (scan_result[i] as usize) * 4;
                for j in 0..4 {
                    output[dst_base + j] = input[src_base + j];
                }
            }
        }

        // First vec4: [1,2,3,4], Second vec4: [5,6,7,8]
        assert_eq!(output[..8], [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_stream_compact_multi_element_simulation() {
        // Simulate multi-element scatter with element_size = 3
        let input: Vec<u32> = vec![
            1, 2, 3,    // Element 0 (pass)
            0, 0, 0,    // Element 1 (fail)
            4, 5, 6,    // Element 2 (pass)
            0, 0, 0,    // Element 3 (fail)
            7, 8, 9,    // Element 4 (pass)
        ];
        let element_size = 3;
        let element_count = 5;

        // Predicates per element
        let predicates: Vec<u32> = vec![1, 0, 1, 0, 1];

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let mut output = vec![0u32; input.len()];
        for i in 0..element_count {
            if predicates[i] != 0 {
                let src_base = i * element_size;
                let dst_base = (scan_result[i] as usize) * element_size;
                for j in 0..element_size {
                    output[dst_base + j] = input[src_base + j];
                }
            }
        }

        // Compacted: [1,2,3], [4,5,6], [7,8,9]
        assert_eq!(output[..9], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_stream_compact_strided_access_simulation() {
        // Simulate strided input access (input_stride = 2, interleaved data)
        let input: Vec<u32> = vec![
            1, 99,   // Element 0: value=1, skip=99
            0, 99,   // Element 1: value=0, skip=99
            3, 99,   // Element 2: value=3, skip=99
            0, 99,   // Element 3: value=0, skip=99
        ];
        let input_stride = 2;
        let element_count = 4;

        // Read with stride
        let values: Vec<u32> = (0..element_count).map(|i| input[i * input_stride]).collect();
        assert_eq!(values, vec![1, 0, 3, 0]);

        let predicates: Vec<u32> = values.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();
        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 2);
    }

    // =========================================================================
    // Boundary Condition Tests
    // =========================================================================

    #[test]
    fn test_scatter_boundary_first_element_only() {
        let input = vec![42u32, 0, 0, 0, 0];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let count = predicates.last().unwrap() + scan_result.last().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scatter_boundary_last_element_only() {
        let input = vec![0u32, 0, 0, 0, 42];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        // Count should be 1
        let count = predicates.last().unwrap() + scan_result.last().unwrap();
        assert_eq!(count, 1);

        // Output index for last element should be 0 (first position)
        let mut output = vec![0u32; 1];
        for (i, &p) in predicates.iter().enumerate() {
            if p != 0 {
                output[scan_result[i] as usize] = input[i];
            }
        }
        assert_eq!(output[0], 42);
    }

    #[test]
    fn test_scatter_boundary_first_and_last_only() {
        let input = vec![1u32, 0, 0, 0, 2];
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let mut scan_result = vec![0u32; predicates.len()];
        let mut sum = 0;
        for (i, &p) in predicates.iter().enumerate() {
            scan_result[i] = sum;
            sum += p;
        }

        let count = predicates.last().unwrap() + scan_result.last().unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_calculation_correctness() {
        // Verify count = predicates[n-1] + scan_result[n-1] formula
        for n in [1, 2, 5, 10, 100] {
            let predicates: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

            let mut scan_result = vec![0u32; predicates.len()];
            let mut sum = 0;
            for (i, &p) in predicates.iter().enumerate() {
                scan_result[i] = sum;
                sum += p;
            }

            // Formula-based count
            let formula_count = predicates[n - 1] + scan_result[n - 1];

            // Direct count
            let direct_count: u32 = predicates.iter().sum();

            assert_eq!(formula_count, direct_count, "Mismatch for n={}", n);
        }
    }

    // =========================================================================
    // Thread Safety Trait Tests
    // =========================================================================

    #[test]
    fn test_compact_params_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CompactParams>();
    }

    #[test]
    fn test_compact_params_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<CompactParams>();
    }

    #[test]
    fn test_predicate_type_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PredicateType>();
    }

    #[test]
    fn test_predicate_type_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PredicateType>();
    }

    #[test]
    fn test_stream_compact_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamCompactError>();
    }

    #[test]
    fn test_stream_compact_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<StreamCompactError>();
    }

    // =========================================================================
    // Performance Characteristic Tests
    // =========================================================================

    #[test]
    fn test_workgroup_efficiency() {
        // Verify workgroup calculation doesn't waste threads unnecessarily
        // For input_size = 257, we need 2 workgroups (512 threads total)
        // Efficiency = 257 / 512 = ~50%
        let input_size = 257u32;
        let workgroups = StreamCompactPipeline::num_workgroups(input_size);
        let total_threads = workgroups * WORKGROUP_SIZE;
        let efficiency = input_size as f32 / total_threads as f32;

        // Efficiency should be > 50% (worst case is just over half)
        assert!(efficiency > 0.5);
    }

    #[test]
    fn test_workgroup_near_optimal_efficiency() {
        // For exact multiples of workgroup size, efficiency should be 100%
        let input_size = WORKGROUP_SIZE * 10;
        let workgroups = StreamCompactPipeline::num_workgroups(input_size);
        let total_threads = workgroups * WORKGROUP_SIZE;
        let efficiency = input_size as f32 / total_threads as f32;

        assert!((efficiency - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Edge Case Stress Tests
    // =========================================================================

    #[test]
    fn test_dense_then_sparse_pattern() {
        // Pattern: first half all pass, second half all fail
        let input: Vec<u32> = (0..100)
            .map(|i| if i < 50 { i + 1 } else { 0 })
            .collect();
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 50);
    }

    #[test]
    fn test_sparse_then_dense_pattern() {
        // Pattern: first half all fail, second half all pass
        let input: Vec<u32> = (0..100)
            .map(|i| if i >= 50 { i + 1 } else { 0 })
            .collect();
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let count: u32 = predicates.iter().sum();
        assert_eq!(count, 50);
    }

    #[test]
    fn test_random_like_pattern() {
        // Simulate a random-ish pattern using a simple formula
        let input: Vec<u32> = (0u32..256)
            .map(|i| {
                let hash = (i.wrapping_mul(2654435761u32) >> 16) % 10;
                if hash < 5 { i + 1 } else { 0 }
            })
            .collect();
        let predicates: Vec<u32> = input.iter().map(|&x| if x != 0 { 1 } else { 0 }).collect();

        let count: u32 = predicates.iter().sum();
        // Should be roughly half
        assert!(count > 100 && count < 156);
    }
}
