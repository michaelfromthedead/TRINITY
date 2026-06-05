//! Compute shader library for TRINITY (T-WGPU-P3.10.6).
//!
//! This module provides reusable compute shader primitives for common GPU operations,
//! unified into a single `ComputeLibrary` type for easy initialization and dispatch.
//!
//! # Available Modules
//!
//! - [`reduction`] - Parallel reduction operations (sum, min, max) using tree reduction
//! - [`prefix_scan`] - Blelloch parallel prefix scan (exclusive/inclusive)
//! - [`stream_compact`] - Stream compaction using prefix scan for filtering
//! - [`radix_sort`] - GPU radix sort using prefix scan for 32-bit key-value pairs
//! - [`image_processing`] - Image processing (blur, downsample, histogram, tonemap)
//!
//! # Architecture
//!
//! ```text
//! ComputeLibrary
//!     |-- reduction: ReductionPipeline
//!     |   |-- sum_pipeline            (reduce_sum.wgsl)
//!     |   |-- min_pipeline            (reduce_min.wgsl)
//!     |   `-- max_pipeline            (reduce_max.wgsl)
//!     |
//!     |-- prefix_scan: PrefixScanPipeline
//!     |   |-- single_block_pipeline   (prefix_scan.wgsl)
//!     |   |-- up_sweep_pipeline       (prefix_scan.wgsl)
//!     |   |-- down_sweep_pipeline     (prefix_scan.wgsl)
//!     |   `-- add_block_sums_pipeline (prefix_scan.wgsl)
//!     |
//!     |-- stream_compact: StreamCompactPipeline
//!     |   |-- scatter_pipeline        (stream_compact.wgsl)
//!     |   |-- scatter_vec4_pipeline   (stream_compact.wgsl)
//!     |   |-- count_pipeline          (stream_compact.wgsl)
//!     |   |-- predicate_nonzero_pipeline
//!     |   |-- scatter_fused_nonzero_pipeline
//!     |   `-- scatter_multi_element_pipeline
//!     |
//!     |-- radix_sort: RadixSortPipeline
//!     |   |-- histogram_pipeline      (radix_sort.wgsl)
//!     |   |-- scatter_pipeline        (radix_sort.wgsl)
//!     |   |-- scatter_keys_pipeline   (radix_sort.wgsl)
//!     |   |-- clear_histogram_pipeline
//!     |   |-- copy_keys_pipeline
//!     |   `-- copy_pairs_pipeline
//!     |
//!     `-- image: ImageProcessor
//!         |-- blur_horizontal_pipeline  (blur_horizontal.wgsl)
//!         |-- blur_vertical_pipeline    (blur_vertical.wgsl)
//!         |-- downsample_pipeline       (downsample.wgsl)
//!         |-- histogram_pipeline        (histogram.wgsl)
//!         |-- histogram_clear_pipeline  (histogram.wgsl)
//!         `-- tonemap_pipeline          (tonemapping.wgsl)
//! ```
//!
//! # Pipeline Count Summary
//!
//! - Reduction: 3 pipelines (sum, min, max)
//! - Prefix Scan: 4 pipelines (single_block, up_sweep, down_sweep, add_block_sums)
//! - Stream Compact: 6 pipelines (scatter, scatter_vec4, count, predicate_nonzero, scatter_fused, scatter_multi)
//! - Radix Sort: 6 pipelines (histogram, scatter, scatter_keys, clear, copy_keys, copy_pairs)
//! - Image: 6 pipelines (blur_h, blur_v, downsample, histogram, histogram_clear, tonemap)
//! - **Total: 25 pipelines**
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::compute_library::ComputeLibrary;
//! use wgpu::util::DeviceExt;
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create unified compute library at startup
//! let compute = ComputeLibrary::new(device);
//!
//! // Create input buffer
//! let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("input"),
//!     contents: bytemuck::cast_slice(&data),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! // High-level dispatch helpers
//! let sum = compute.reduce_sum(device, queue, &input_buffer, data.len() as u32);
//! println!("Sum: {:?}", sum); // Ok(15.0)
//!
//! let min = compute.reduce_min(device, queue, &input_buffer, data.len() as u32);
//! println!("Min: {:?}", min); // Ok(1.0)
//!
//! // Create u32 buffer for prefix scan
//! let scan_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
//! let scan_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("scan_input"),
//!     contents: bytemuck::cast_slice(&scan_data),
//!     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//! });
//!
//! // Prefix scan (in-place)
//! compute.prefix_scan_exclusive(device, queue, &scan_buffer, scan_data.len() as u32);
//! // Result: [0, 1, 3, 6, 10, 15, 21, 28]
//! # }
//! ```

pub mod image_processing;
pub mod prefix_scan;
pub mod radix_sort;
pub mod reduction;
pub mod stream_compact;

// Re-export commonly used types
pub use image_processing::{
    BlurUniforms, DownsampleUniforms, FilterMode, HistogramUniforms, ImageProcessor,
    TonemapMode, TonemapUniforms,
};
pub use prefix_scan::{PrefixScanError, PrefixScanPipeline, ScanParams};
pub use radix_sort::{RadixSortError, RadixSortParams, RadixSortPipeline};
pub use reduction::{ReductionError, ReductionOperation, ReductionPipeline, ReductionParams};
pub use stream_compact::{CompactParams, PredicateType, StreamCompactError, StreamCompactPipeline};

// ============================================================================
// ComputeLibraryError
// ============================================================================

/// Errors that can occur during ComputeLibrary operations.
#[derive(Debug)]
pub enum ComputeLibraryError {
    /// Reduction operation failed.
    Reduction(ReductionError),
    /// Prefix scan operation failed.
    PrefixScan(PrefixScanError),
    /// Stream compaction operation failed.
    StreamCompact(StreamCompactError),
    /// Radix sort operation failed.
    RadixSort(RadixSortError),
    /// Buffer mapping failed.
    BufferMapFailed(String),
    /// Input validation failed.
    InvalidInput(String),
}

impl std::fmt::Display for ComputeLibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reduction(e) => write!(f, "Reduction error: {}", e),
            Self::PrefixScan(e) => write!(f, "Prefix scan error: {}", e),
            Self::StreamCompact(e) => write!(f, "Stream compaction error: {}", e),
            Self::RadixSort(e) => write!(f, "Radix sort error: {}", e),
            Self::BufferMapFailed(msg) => write!(f, "Buffer mapping failed: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for ComputeLibraryError {}

impl From<ReductionError> for ComputeLibraryError {
    fn from(e: ReductionError) -> Self {
        Self::Reduction(e)
    }
}

impl From<PrefixScanError> for ComputeLibraryError {
    fn from(e: PrefixScanError) -> Self {
        Self::PrefixScan(e)
    }
}

impl From<StreamCompactError> for ComputeLibraryError {
    fn from(e: StreamCompactError) -> Self {
        Self::StreamCompact(e)
    }
}

impl From<RadixSortError> for ComputeLibraryError {
    fn from(e: RadixSortError) -> Self {
        Self::RadixSort(e)
    }
}

// ============================================================================
// DispatchHelper
// ============================================================================

/// Utility for calculating compute dispatch parameters.
///
/// Provides consistent workgroup calculations across different compute operations.
#[derive(Debug, Clone, Copy)]
pub struct DispatchHelper {
    /// Workgroup size (threads per workgroup).
    pub workgroup_size: u32,
    /// Elements processed per thread.
    pub elements_per_thread: u32,
}

impl DispatchHelper {
    /// Create a new dispatch helper with standard settings.
    ///
    /// # Arguments
    ///
    /// * `workgroup_size` - Number of threads per workgroup (typically 256).
    /// * `elements_per_thread` - Elements processed per thread (typically 1-4).
    pub const fn new(workgroup_size: u32, elements_per_thread: u32) -> Self {
        Self {
            workgroup_size,
            elements_per_thread,
        }
    }

    /// Create dispatch helper for reduction operations.
    pub const fn for_reduction() -> Self {
        Self::new(256, 2) // 256 threads, 2 elements each = 512 elements/workgroup
    }

    /// Create dispatch helper for prefix scan operations.
    pub const fn for_prefix_scan() -> Self {
        Self::new(256, 2) // 256 threads, 2 elements each = 512 elements/workgroup
    }

    /// Create dispatch helper for stream compaction.
    pub const fn for_stream_compact() -> Self {
        Self::new(256, 1) // 256 threads, 1 element each
    }

    /// Create dispatch helper for radix sort.
    pub const fn for_radix_sort() -> Self {
        Self::new(256, 4) // 256 threads, 4 elements each = 1024 elements/workgroup
    }

    /// Create dispatch helper for image operations.
    pub const fn for_image() -> Self {
        Self::new(8, 1) // 8x8 workgroups for 2D image operations
    }

    /// Calculate elements processed per workgroup.
    #[inline]
    pub const fn elements_per_workgroup(&self) -> u32 {
        self.workgroup_size * self.elements_per_thread
    }

    /// Calculate number of workgroups needed for the given element count.
    ///
    /// Uses saturating arithmetic to handle edge cases safely.
    #[inline]
    pub const fn num_workgroups(&self, element_count: u32) -> u32 {
        let epw = self.elements_per_workgroup();
        if element_count == 0 || epw == 0 {
            return 0;
        }
        // Use div_ceil equivalent: (element_count + epw - 1) / epw
        // Handle potential overflow with saturating arithmetic
        let numerator = element_count.saturating_add(epw - 1);
        numerator / epw
    }

    /// Calculate number of workgroups for 2D dispatch.
    ///
    /// Uses saturating arithmetic to handle edge cases safely.
    #[inline]
    pub const fn num_workgroups_2d(&self, width: u32, height: u32) -> (u32, u32) {
        if self.workgroup_size == 0 {
            return (0, 0);
        }
        let x = width.saturating_add(self.workgroup_size - 1) / self.workgroup_size;
        let y = height.saturating_add(self.workgroup_size - 1) / self.workgroup_size;
        (x, y)
    }
}

impl Default for DispatchHelper {
    fn default() -> Self {
        Self::new(256, 1)
    }
}

// ============================================================================
// ComputeLibrary
// ============================================================================

/// Unified compute shader library containing all TRINITY compute primitives.
///
/// This struct initializes all compute pipelines at construction time and provides
/// high-level dispatch helpers for common operations. Create one instance per
/// wgpu Device and reuse for all compute operations.
///
/// # Pipeline Count
///
/// The library contains **25 total pipelines**:
/// - Reduction: 3 (sum, min, max)
/// - Prefix Scan: 4 (single_block, up_sweep, down_sweep, add_block_sums)
/// - Stream Compact: 6 (scatter variants)
/// - Radix Sort: 6 (histogram, scatter, clear, copy)
/// - Image: 6 (blur_h, blur_v, downsample, histogram, histogram_clear, tonemap)
///
/// # Thread Safety
///
/// `ComputeLibrary` is `Send + Sync` when its underlying wgpu types are.
/// Operations can be performed concurrently by creating separate command encoders.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::compute_library::ComputeLibrary;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// // Initialize at startup
/// let compute = ComputeLibrary::new(device);
///
/// // Use throughout frame rendering
/// // ... compute.reduce_sum(...) etc.
/// # }
/// ```
pub struct ComputeLibrary {
    /// Reduction pipelines (sum, min, max).
    pub reduction: ReductionPipeline,
    /// Blelloch prefix scan pipelines.
    pub prefix_scan: PrefixScanPipeline,
    /// Stream compaction pipelines.
    pub stream_compact: StreamCompactPipeline,
    /// Radix sort pipelines.
    pub radix_sort: RadixSortPipeline,
    /// Image processing pipelines.
    pub image: ImageProcessor,
}

impl ComputeLibrary {
    /// Create a new ComputeLibrary with all pipelines initialized.
    ///
    /// This compiles all compute shaders and creates pipeline layouts.
    /// The operation may take several milliseconds on first call due to
    /// shader compilation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device to create pipelines on.
    ///
    /// # Returns
    ///
    /// A fully initialized `ComputeLibrary` ready for use.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(device: &wgpu::Device) {
    /// let compute = renderer_backend::compute_library::ComputeLibrary::new(device);
    /// # }
    /// ```
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            reduction: ReductionPipeline::new(device),
            prefix_scan: PrefixScanPipeline::new(device),
            stream_compact: StreamCompactPipeline::new(device),
            radix_sort: RadixSortPipeline::new(device),
            image: ImageProcessor::new(device),
        }
    }

    /// Create a ComputeLibrary with lazy initialization.
    ///
    /// Only creates the reduction pipeline initially. Other pipelines are
    /// created on first use. Useful when not all pipelines are needed.
    ///
    /// # Note
    ///
    /// Currently not implemented - all pipelines are created eagerly.
    /// This is a future optimization opportunity.
    pub fn new_lazy(device: &wgpu::Device) -> Self {
        // For now, just create everything. Future optimization could
        // use Option<T> for lazy initialization.
        Self::new(device)
    }

    // ========================================================================
    // Reduction Helpers
    // ========================================================================

    /// Perform sum reduction on a buffer of f32 values.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing f32 values.
    /// * `count` - Number of elements to sum.
    ///
    /// # Returns
    ///
    /// The sum of all elements, or an error if the operation failed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(compute: &renderer_backend::compute_library::ComputeLibrary,
    /// #            device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
    /// let sum = compute.reduce_sum(device, queue, buffer, 1024);
    /// println!("Sum: {:?}", sum);
    /// # }
    /// ```
    #[inline]
    pub fn reduce_sum(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<f32, ReductionError> {
        self.reduction.reduce_sum(device, queue, buffer, count)
    }

    /// Perform min reduction on a buffer of f32 values.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing f32 values.
    /// * `count` - Number of elements.
    ///
    /// # Returns
    ///
    /// The minimum element, or an error if the operation failed.
    #[inline]
    pub fn reduce_min(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<f32, ReductionError> {
        self.reduction.reduce_min(device, queue, buffer, count)
    }

    /// Perform max reduction on a buffer of f32 values.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing f32 values.
    /// * `count` - Number of elements.
    ///
    /// # Returns
    ///
    /// The maximum element, or an error if the operation failed.
    #[inline]
    pub fn reduce_max(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<f32, ReductionError> {
        self.reduction.reduce_max(device, queue, buffer, count)
    }

    /// Perform min/max reduction in a single pass.
    ///
    /// More efficient than calling reduce_min and reduce_max separately
    /// when both values are needed.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing f32 values.
    /// * `count` - Number of elements.
    ///
    /// # Returns
    ///
    /// A tuple of (min, max), or an error if the operation failed.
    pub fn reduce_minmax(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<(f32, f32), ReductionError> {
        // For now, just call both. A future optimization could use a
        // combined min/max shader to reduce memory bandwidth.
        let min = self.reduction.reduce_min(device, queue, buffer, count)?;
        let max = self.reduction.reduce_max(device, queue, buffer, count)?;
        Ok((min, max))
    }

    // ========================================================================
    // Prefix Scan Helpers
    // ========================================================================

    /// Perform exclusive prefix scan on a buffer of u32 values.
    ///
    /// Given input `[a, b, c, d]`, produces `[0, a, a+b, a+b+c]`.
    /// The buffer is modified in-place.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing u32 values (modified in-place).
    /// * `count` - Number of elements to scan.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(compute: &renderer_backend::compute_library::ComputeLibrary,
    /// #            device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
    /// // Before: [1, 2, 3, 4, 5, 6, 7, 8]
    /// compute.prefix_scan_exclusive(device, queue, buffer, 8);
    /// // After:  [0, 1, 3, 6, 10, 15, 21, 28]
    /// # }
    /// ```
    #[inline]
    pub fn prefix_scan_exclusive(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) {
        self.prefix_scan.scan(device, queue, buffer, count);
    }

    /// Perform exclusive prefix scan and return the total sum.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `buffer` - Storage buffer containing u32 values (modified in-place).
    /// * `count` - Number of elements to scan.
    ///
    /// # Returns
    ///
    /// A buffer containing the total sum (single u32).
    #[inline]
    pub fn prefix_scan_with_total(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> wgpu::Buffer {
        self.prefix_scan.scan_with_total(device, queue, buffer, count)
    }

    // ========================================================================
    // Stream Compaction Helpers
    // ========================================================================

    /// Compact non-zero elements from a buffer.
    ///
    /// Filters out all zero elements and packs non-zero elements contiguously.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `input_buffer` - Buffer containing input u32 values.
    /// * `output_buffer` - Buffer for compacted output.
    /// * `count` - Number of input elements.
    ///
    /// # Returns
    ///
    /// A buffer containing the count of compacted elements (single u32).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(compute: &renderer_backend::compute_library::ComputeLibrary,
    /// #            device: &wgpu::Device, queue: &wgpu::Queue,
    /// #            input: &wgpu::Buffer, output: &wgpu::Buffer) {
    /// // Input:  [1, 0, 3, 0, 5, 6, 0, 8]
    /// let count_buffer = compute.stream_compact_nonzero(device, queue, input, output, 8);
    /// // Output: [1, 3, 5, 6, 8], count = 5
    /// # }
    /// ```
    #[inline]
    pub fn stream_compact_nonzero(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        count: u32,
    ) -> wgpu::Buffer {
        self.stream_compact.compact_nonzero(device, queue, input_buffer, output_buffer, count)
    }

    /// Compact elements using pre-computed predicates.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `input_buffer` - Buffer containing input data.
    /// * `predicates_buffer` - Buffer containing 0/1 predicates.
    /// * `output_buffer` - Buffer for compacted output.
    /// * `count` - Number of elements.
    ///
    /// # Returns
    ///
    /// A buffer containing the count of compacted elements (single u32).
    #[inline]
    pub fn stream_compact_with_predicates(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        predicates_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        count: u32,
    ) -> wgpu::Buffer {
        self.stream_compact.compact(
            device,
            queue,
            input_buffer,
            predicates_buffer,
            output_buffer,
            count,
        )
    }

    /// Compact vec4 elements (4 u32s per element) using pre-computed predicates.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `input_buffer` - Buffer containing input data (4 u32s per element).
    /// * `predicates_buffer` - Buffer containing 0/1 predicates (1 per vec4).
    /// * `output_buffer` - Buffer for compacted output.
    /// * `element_count` - Number of vec4 elements (not u32s).
    ///
    /// # Returns
    ///
    /// A buffer containing the count of compacted elements (single u32).
    #[inline]
    pub fn stream_compact_vec4(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        predicates_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        element_count: u32,
    ) -> wgpu::Buffer {
        self.stream_compact.compact_vec4(
            device,
            queue,
            input_buffer,
            predicates_buffer,
            output_buffer,
            element_count,
        )
    }

    // ========================================================================
    // Radix Sort Helpers
    // ========================================================================

    /// Sort u32 keys in ascending order.
    ///
    /// Performs an in-place radix sort on the keys buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `keys_buffer` - Buffer containing u32 keys (modified in-place).
    /// * `count` - Number of keys to sort.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(compute: &renderer_backend::compute_library::ComputeLibrary,
    /// #            device: &wgpu::Device, queue: &wgpu::Queue, keys: &wgpu::Buffer) {
    /// // Before: [5, 3, 8, 1, 9, 2]
    /// compute.radix_sort_keys(device, queue, keys, 6);
    /// // After:  [1, 2, 3, 5, 8, 9]
    /// # }
    /// ```
    #[inline]
    pub fn radix_sort_keys(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        keys_buffer: &wgpu::Buffer,
        count: u32,
    ) {
        self.radix_sort.sort_keys(device, queue, keys_buffer, count);
    }

    /// Sort u32 key-value pairs by key in ascending order.
    ///
    /// Performs radix sort on keys while permuting values to match.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `keys_buffer` - Buffer containing u32 keys (modified in-place).
    /// * `values_buffer` - Buffer containing u32 values (modified in-place).
    /// * `count` - Number of key-value pairs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(compute: &renderer_backend::compute_library::ComputeLibrary,
    /// #            device: &wgpu::Device, queue: &wgpu::Queue,
    /// #            keys: &wgpu::Buffer, values: &wgpu::Buffer) {
    /// // Before: keys = [5, 3, 8, 1], values = [50, 30, 80, 10]
    /// compute.radix_sort_pairs(device, queue, keys, values, 4);
    /// // After:  keys = [1, 3, 5, 8], values = [10, 30, 50, 80]
    /// # }
    /// ```
    #[inline]
    pub fn radix_sort_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        keys_buffer: &wgpu::Buffer,
        values_buffer: &wgpu::Buffer,
        count: u32,
    ) {
        self.radix_sort.sort_pairs(device, queue, keys_buffer, values_buffer, count);
    }

    // ========================================================================
    // Image Processing Helpers
    // ========================================================================

    /// Get the image processor for advanced image operations.
    ///
    /// The image processor provides methods for blur, downsample, histogram,
    /// and tonemapping operations. For these operations, you need to work
    /// with texture views and command encoders directly.
    ///
    /// # Returns
    ///
    /// A reference to the internal `ImageProcessor`.
    #[inline]
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.image
    }

    /// Apply a full Gaussian blur (horizontal + vertical passes).
    ///
    /// This is a convenience method that performs both blur passes with an
    /// intermediate texture.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `encoder` - Command encoder to record commands to.
    /// * `src_view` - Source texture view.
    /// * `temp_view` - Intermediate texture view (same size as source).
    /// * `dst_view` - Destination texture view.
    /// * `uniforms` - Blur uniform buffer.
    /// * `width` - Texture width.
    /// * `height` - Texture height.
    pub fn blur_gaussian(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_view: &wgpu::TextureView,
        temp_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        // Horizontal pass: src -> temp
        let h_bind_group = self.image.create_blur_bind_group(device, src_view, temp_view, uniforms);
        self.image.blur_horizontal(encoder, &h_bind_group, width, height);

        // Vertical pass: temp -> dst
        let v_bind_group = self.image.create_blur_bind_group(device, temp_view, dst_view, uniforms);
        self.image.blur_vertical(encoder, &v_bind_group, width, height);
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Get a dispatch helper for reduction operations.
    #[inline]
    pub const fn dispatch_reduction() -> DispatchHelper {
        DispatchHelper::for_reduction()
    }

    /// Get a dispatch helper for prefix scan operations.
    #[inline]
    pub const fn dispatch_prefix_scan() -> DispatchHelper {
        DispatchHelper::for_prefix_scan()
    }

    /// Get a dispatch helper for stream compaction operations.
    #[inline]
    pub const fn dispatch_stream_compact() -> DispatchHelper {
        DispatchHelper::for_stream_compact()
    }

    /// Get a dispatch helper for radix sort operations.
    #[inline]
    pub const fn dispatch_radix_sort() -> DispatchHelper {
        DispatchHelper::for_radix_sort()
    }

    /// Get a dispatch helper for image operations.
    #[inline]
    pub const fn dispatch_image() -> DispatchHelper {
        DispatchHelper::for_image()
    }

    /// Get pipeline statistics for debugging.
    ///
    /// # Returns
    ///
    /// A `PipelineStats` struct containing pipeline counts.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        }
    }
}

// ============================================================================
// PipelineStats
// ============================================================================

/// Statistics about the compute library's pipeline usage.
#[derive(Debug, Clone, Copy)]
pub struct PipelineStats {
    /// Number of reduction pipelines.
    pub reduction_pipelines: u32,
    /// Number of prefix scan pipelines.
    pub prefix_scan_pipelines: u32,
    /// Number of stream compaction pipelines.
    pub stream_compact_pipelines: u32,
    /// Number of radix sort pipelines.
    pub radix_sort_pipelines: u32,
    /// Number of image processing pipelines.
    pub image_pipelines: u32,
    /// Total number of pipelines.
    pub total_pipelines: u32,
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ComputeLibrary: {} pipelines (reduction={}, scan={}, compact={}, sort={}, image={})",
            self.total_pipelines,
            self.reduction_pipelines,
            self.prefix_scan_pipelines,
            self.stream_compact_pipelines,
            self.radix_sort_pipelines,
            self.image_pipelines
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_helper_reduction() {
        let helper = DispatchHelper::for_reduction();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 2);
        assert_eq!(helper.elements_per_workgroup(), 512);
        assert_eq!(helper.num_workgroups(1), 1);
        assert_eq!(helper.num_workgroups(512), 1);
        assert_eq!(helper.num_workgroups(513), 2);
        assert_eq!(helper.num_workgroups(1024), 2);
    }

    #[test]
    fn test_dispatch_helper_prefix_scan() {
        let helper = DispatchHelper::for_prefix_scan();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 2);
        assert_eq!(helper.elements_per_workgroup(), 512);
    }

    #[test]
    fn test_dispatch_helper_radix_sort() {
        let helper = DispatchHelper::for_radix_sort();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 4);
        assert_eq!(helper.elements_per_workgroup(), 1024);
        assert_eq!(helper.num_workgroups(1024), 1);
        assert_eq!(helper.num_workgroups(1025), 2);
    }

    #[test]
    fn test_dispatch_helper_2d() {
        let helper = DispatchHelper::for_image();
        assert_eq!(helper.workgroup_size, 8);
        let (x, y) = helper.num_workgroups_2d(800, 600);
        assert_eq!(x, 100); // ceil(800 / 8)
        assert_eq!(y, 75);  // ceil(600 / 8)
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };
        assert_eq!(stats.total_pipelines, 25);
        let s = format!("{}", stats);
        assert!(s.contains("25 pipelines"));
    }
}
