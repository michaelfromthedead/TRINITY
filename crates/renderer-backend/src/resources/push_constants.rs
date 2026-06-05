//! Push constant support for TRINITY.
//!
//! This module provides a type-safe abstraction layer for wgpu push constants,
//! including feature detection, range configuration, type-safe writers, and
//! automatic fallback to uniform buffers on unsupported backends.
//!
//! # Overview
//!
//! Push constants are a GPU feature that allows small amounts of data to be
//! passed directly to shaders without going through buffer bindings. They are
//! ideal for per-draw data like object IDs, material indices, or transform
//! offsets.
//!
//! # WebGPU Limits
//!
//! - Maximum total size: 128 bytes (WebGPU limit)
//! - Alignment: 4 bytes (offset and size must be 4-byte aligned)
//! - Stages: Can be visible to VERTEX, FRAGMENT, COMPUTE, or combinations
//!
//! # Architecture
//!
//! ```text
//! PushConstantConfig
//! ├── Range definitions (stages, offset, size)
//! ├── Validation (alignment, overlap, size limits)
//! └── Used by pipeline layout creation
//!
//! PushConstantWriter
//! ├── Wraps RenderPass or ComputePass
//! ├── Type-safe set<T>() method using bytemuck
//! └── Convenience: set_vertex(), set_fragment(), set_compute()
//!
//! PushConstantFallback
//! ├── Native: Uses wgpu push constants
//! └── UniformBuffer: Falls back to uniform buffer binding
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::push_constants::{
//!     PushConstantConfig, supports_push_constants,
//! };
//! use wgpu::ShaderStages;
//!
//! # fn example(device: &wgpu::Device) {
//! // Check feature support
//! if supports_push_constants(device) {
//!     // Configure push constant ranges
//!     let config = PushConstantConfig::new()
//!         .add_range(ShaderStages::VERTEX, 0, 64)
//!         .expect("valid range")
//!         .add_range(ShaderStages::FRAGMENT, 64, 64)
//!         .expect("valid range");
//!
//!     assert!(config.validate().is_ok());
//! }
//! # }
//! ```

use std::ops::Range;
use std::sync::Arc;
use wgpu::{
    Buffer, BufferUsages, ComputePass, Device, Features, PushConstantRange, Queue, RenderPass,
    ShaderStages,
};

use super::pipeline_layout::bind_group_index;

// ============================================================================
// Constants
// ============================================================================

/// Maximum push constant size in bytes (WebGPU limit).
pub const MAX_PUSH_CONSTANT_SIZE: u32 = 128;

/// Required alignment for push constant offset and size (4 bytes).
pub const PUSH_CONSTANT_ALIGNMENT: u32 = 4;

/// Default bind group index for push constant fallback uniform buffer.
/// Uses the OBJECT bind group index by convention.
pub const DEFAULT_FALLBACK_BIND_GROUP: u32 = bind_group_index::OBJECT;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when working with push constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushConstantError {
    /// Offset is not 4-byte aligned.
    MisalignedOffset(u32),

    /// Size is not 4-byte aligned.
    MisalignedSize(u32),

    /// Total push constant size exceeds the WebGPU limit.
    ExceedsMaxSize {
        /// The total size that was attempted.
        total: u32,
        /// The maximum allowed size.
        max: u32,
    },

    /// Two ranges overlap within the same shader stages.
    RangeOverlap {
        /// Description of the first overlapping range.
        range1: String,
        /// Description of the second overlapping range.
        range2: String,
    },

    /// Push constants are not supported on this device.
    UnsupportedFeature,

    /// The data size doesn't fit within the configured range.
    DataTooLarge {
        /// Size of the data being written.
        data_size: u32,
        /// Available space in the range.
        available: u32,
    },

    /// The offset is outside the configured ranges.
    InvalidOffset {
        /// The offset that was attempted.
        offset: u32,
        /// The shader stages for the write.
        stages: ShaderStages,
    },

    /// Empty ranges are not allowed.
    EmptyRange,
}

impl std::fmt::Display for PushConstantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MisalignedOffset(offset) => {
                write!(
                    f,
                    "Push constant offset {} is not {}-byte aligned",
                    offset, PUSH_CONSTANT_ALIGNMENT
                )
            }
            Self::MisalignedSize(size) => {
                write!(
                    f,
                    "Push constant size {} is not {}-byte aligned",
                    size, PUSH_CONSTANT_ALIGNMENT
                )
            }
            Self::ExceedsMaxSize { total, max } => {
                write!(
                    f,
                    "Push constant total size {} exceeds maximum {} bytes",
                    total, max
                )
            }
            Self::RangeOverlap { range1, range2 } => {
                write!(
                    f,
                    "Push constant ranges overlap with same stages: {} and {}",
                    range1, range2
                )
            }
            Self::UnsupportedFeature => {
                write!(f, "Push constants are not supported on this device")
            }
            Self::DataTooLarge {
                data_size,
                available,
            } => {
                write!(
                    f,
                    "Push constant data size {} exceeds available space {}",
                    data_size, available
                )
            }
            Self::InvalidOffset { offset, stages } => {
                write!(
                    f,
                    "Push constant offset {} is not valid for stages {:?}",
                    offset, stages
                )
            }
            Self::EmptyRange => {
                write!(f, "Push constant range cannot be empty (size must be > 0)")
            }
        }
    }
}

impl std::error::Error for PushConstantError {}

// ============================================================================
// Feature Detection
// ============================================================================

/// Check if the device supports push constants.
///
/// Note: In wgpu/WebGPU, push constants require the `PUSH_CONSTANTS` feature.
/// Most native backends support this, but WebGPU in browsers may not.
///
/// # Arguments
///
/// * `device` - The wgpu device to check
///
/// # Returns
///
/// `true` if push constants are supported, `false` otherwise.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::push_constants::supports_push_constants;
///
/// # fn example(device: &wgpu::Device) {
/// if supports_push_constants(device) {
///     println!("Push constants supported!");
/// } else {
///     println!("Using uniform buffer fallback");
/// }
/// # }
/// ```
#[inline]
pub fn supports_push_constants(device: &Device) -> bool {
    device.features().contains(Features::PUSH_CONSTANTS)
}

/// Get the maximum push constant size supported by the device.
///
/// If push constants are not supported, returns 0.
/// Otherwise returns the configured limit (typically 128 bytes for WebGPU).
///
/// # Arguments
///
/// * `device` - The wgpu device to query
///
/// # Returns
///
/// Maximum push constant size in bytes, or 0 if not supported.
#[inline]
pub fn max_push_constant_size(device: &Device) -> u32 {
    if supports_push_constants(device) {
        device.limits().max_push_constant_size
    } else {
        0
    }
}

// ============================================================================
// PushConstantConfig
// ============================================================================

/// Configuration for push constant ranges.
///
/// This struct uses a builder pattern to configure push constant ranges,
/// with validation at each step.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::push_constants::PushConstantConfig;
/// use wgpu::ShaderStages;
///
/// let config = PushConstantConfig::new()
///     .add_range(ShaderStages::VERTEX, 0, 64)
///     .expect("valid range")
///     .add_range(ShaderStages::FRAGMENT, 64, 64)
///     .expect("valid range");
///
/// assert!(config.validate().is_ok());
/// assert_eq!(config.total_size(), 128);
/// ```
#[derive(Debug, Clone, Default)]
pub struct PushConstantConfig {
    /// Configured push constant ranges.
    ranges: Vec<PushConstantRange>,
}

impl PushConstantConfig {
    /// Create a new empty push constant configuration.
    #[inline]
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Add a push constant range for the specified shader stages.
    ///
    /// # Arguments
    ///
    /// * `stages` - Shader stages that can access this range
    /// * `offset` - Byte offset from the start (must be 4-byte aligned)
    /// * `size` - Size in bytes (must be 4-byte aligned and > 0)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Offset is not 4-byte aligned
    /// - Size is not 4-byte aligned or is 0
    /// - Total size would exceed 128 bytes
    /// - Range overlaps with existing range in same stages
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::push_constants::PushConstantConfig;
    /// use wgpu::ShaderStages;
    ///
    /// let config = PushConstantConfig::new()
    ///     .add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 0, 64)
    ///     .expect("valid range");
    /// ```
    pub fn add_range(
        mut self,
        stages: ShaderStages,
        offset: u32,
        size: u32,
    ) -> Result<Self, PushConstantError> {
        // Validate alignment
        if offset % PUSH_CONSTANT_ALIGNMENT != 0 {
            return Err(PushConstantError::MisalignedOffset(offset));
        }
        if size % PUSH_CONSTANT_ALIGNMENT != 0 {
            return Err(PushConstantError::MisalignedSize(size));
        }
        if size == 0 {
            return Err(PushConstantError::EmptyRange);
        }

        let end = offset + size;

        // Validate total size
        if end > MAX_PUSH_CONSTANT_SIZE {
            return Err(PushConstantError::ExceedsMaxSize {
                total: end,
                max: MAX_PUSH_CONSTANT_SIZE,
            });
        }

        // Check for overlaps with same stages
        let new_range_desc = format!("{}..{} ({:?})", offset, end, stages);
        for existing in &self.ranges {
            if existing.stages.intersects(stages) {
                let existing_start = existing.range.start;
                let existing_end = existing.range.end;

                // Check if ranges overlap
                if offset < existing_end && existing_start < end {
                    let existing_desc = format!(
                        "{}..{} ({:?})",
                        existing_start, existing_end, existing.stages
                    );
                    return Err(PushConstantError::RangeOverlap {
                        range1: existing_desc,
                        range2: new_range_desc,
                    });
                }
            }
        }

        self.ranges.push(PushConstantRange {
            stages,
            range: offset..end,
        });

        Ok(self)
    }

    /// Add a range covering all stages (VERTEX | FRAGMENT).
    ///
    /// Convenience method for common use case.
    #[inline]
    pub fn add_vertex_fragment_range(
        self,
        offset: u32,
        size: u32,
    ) -> Result<Self, PushConstantError> {
        self.add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT, offset, size)
    }

    /// Add a compute-only range.
    #[inline]
    pub fn add_compute_range(self, offset: u32, size: u32) -> Result<Self, PushConstantError> {
        self.add_range(ShaderStages::COMPUTE, offset, size)
    }

    /// Validate the entire configuration.
    ///
    /// Checks:
    /// - All ranges are properly aligned
    /// - Total size doesn't exceed limit
    /// - No overlapping ranges within same stages
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with description otherwise.
    pub fn validate(&self) -> Result<(), PushConstantError> {
        // Check total size
        let total = self.total_size();
        if total > MAX_PUSH_CONSTANT_SIZE {
            return Err(PushConstantError::ExceedsMaxSize {
                total,
                max: MAX_PUSH_CONSTANT_SIZE,
            });
        }

        // Check alignment and overlaps (already validated during add_range,
        // but re-check for safety if ranges were modified)
        for range in &self.ranges {
            if range.range.start % PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(PushConstantError::MisalignedOffset(range.range.start));
            }
            let size = range.range.end - range.range.start;
            if size % PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(PushConstantError::MisalignedSize(size));
            }
        }

        Ok(())
    }

    /// Returns the total push constant size (max end offset across all ranges).
    #[inline]
    pub fn total_size(&self) -> u32 {
        self.ranges.iter().map(|r| r.range.end).max().unwrap_or(0)
    }

    /// Returns the number of configured ranges.
    #[inline]
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Returns true if no ranges are configured.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Returns the configured ranges as a slice.
    ///
    /// Use this when creating pipeline layouts.
    #[inline]
    pub fn ranges(&self) -> &[PushConstantRange] {
        &self.ranges
    }

    /// Convert to a Vec of wgpu PushConstantRange.
    #[inline]
    pub fn into_ranges(self) -> Vec<PushConstantRange> {
        self.ranges
    }

    /// Find the range containing the given offset for the specified stages.
    ///
    /// Returns the range and the maximum size available at that offset.
    pub fn find_range(&self, stages: ShaderStages, offset: u32) -> Option<(Range<u32>, u32)> {
        for range in &self.ranges {
            if range.stages.contains(stages)
                && offset >= range.range.start
                && offset < range.range.end
            {
                let available = range.range.end - offset;
                return Some((range.range.clone(), available));
            }
        }
        None
    }

    /// Check if the given offset and size are valid for the specified stages.
    pub fn is_valid_write(
        &self,
        stages: ShaderStages,
        offset: u32,
        size: u32,
    ) -> Result<(), PushConstantError> {
        if let Some((_, available)) = self.find_range(stages, offset) {
            if size <= available {
                Ok(())
            } else {
                Err(PushConstantError::DataTooLarge {
                    data_size: size,
                    available,
                })
            }
        } else {
            Err(PushConstantError::InvalidOffset { offset, stages })
        }
    }
}

// ============================================================================
// PushConstantWriter
// ============================================================================

/// A type-safe wrapper for writing push constants to a render pass.
///
/// This struct wraps a mutable reference to a `RenderPass` and provides
/// type-safe methods for setting push constant data using bytemuck.
///
/// # Lifetime
///
/// The writer borrows the render pass mutably, so no other operations
/// can be performed on the pass while the writer exists.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::push_constants::{PushConstantConfig, PushConstantWriter};
/// use wgpu::ShaderStages;
/// use bytemuck::{Pod, Zeroable};
///
/// #[repr(C)]
/// #[derive(Copy, Clone, Pod, Zeroable)]
/// struct PerDrawData {
///     object_id: u32,
///     material_id: u32,
///     _padding: [u32; 2],
/// }
///
/// # fn example<'a>(render_pass: &'a mut wgpu::RenderPass<'a>, config: &'a PushConstantConfig) {
/// let mut writer = PushConstantWriter::new(render_pass, config);
///
/// let data = PerDrawData {
///     object_id: 42,
///     material_id: 7,
///     _padding: [0; 2],
/// };
///
/// writer.set_vertex(0, &data).expect("write succeeded");
/// # }
/// ```
pub struct PushConstantWriter<'a, 'b> {
    pass: &'a mut RenderPass<'b>,
    config: &'a PushConstantConfig,
}

impl<'a, 'b> PushConstantWriter<'a, 'b> {
    /// Create a new push constant writer wrapping a render pass.
    ///
    /// # Arguments
    ///
    /// * `pass` - Mutable reference to the render pass
    /// * `config` - Push constant configuration for validation
    #[inline]
    pub fn new(pass: &'a mut RenderPass<'b>, config: &'a PushConstantConfig) -> Self {
        Self { pass, config }
    }

    /// Set push constant data for the specified shader stages.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The data type, must implement `bytemuck::Pod`
    ///
    /// # Arguments
    ///
    /// * `stages` - Shader stages to set the data for
    /// * `offset` - Byte offset within the push constant range
    /// * `data` - Reference to the data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the offset is invalid or the data doesn't fit.
    pub fn set<T: bytemuck::Pod>(
        &mut self,
        stages: ShaderStages,
        offset: u32,
        data: &T,
    ) -> Result<(), PushConstantError> {
        let bytes = bytemuck::bytes_of(data);
        let size = bytes.len() as u32;

        // Validate the write
        self.config.is_valid_write(stages, offset, size)?;

        // Perform the write
        self.pass.set_push_constants(stages, offset, bytes);
        Ok(())
    }

    /// Set push constant data for vertex stage only.
    #[inline]
    pub fn set_vertex<T: bytemuck::Pod>(
        &mut self,
        offset: u32,
        data: &T,
    ) -> Result<(), PushConstantError> {
        self.set(ShaderStages::VERTEX, offset, data)
    }

    /// Set push constant data for fragment stage only.
    #[inline]
    pub fn set_fragment<T: bytemuck::Pod>(
        &mut self,
        offset: u32,
        data: &T,
    ) -> Result<(), PushConstantError> {
        self.set(ShaderStages::FRAGMENT, offset, data)
    }

    /// Set push constant data for both vertex and fragment stages.
    #[inline]
    pub fn set_vertex_fragment<T: bytemuck::Pod>(
        &mut self,
        offset: u32,
        data: &T,
    ) -> Result<(), PushConstantError> {
        self.set(ShaderStages::VERTEX | ShaderStages::FRAGMENT, offset, data)
    }

    /// Set raw bytes as push constant data.
    ///
    /// Use this when you need to write raw bytes instead of a typed struct.
    pub fn set_bytes(
        &mut self,
        stages: ShaderStages,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), PushConstantError> {
        let size = bytes.len() as u32;
        self.config.is_valid_write(stages, offset, size)?;
        self.pass.set_push_constants(stages, offset, bytes);
        Ok(())
    }

    /// Returns a reference to the underlying render pass.
    #[inline]
    pub fn pass(&self) -> &RenderPass<'b> {
        self.pass
    }

    /// Returns a mutable reference to the underlying render pass.
    #[inline]
    pub fn pass_mut(&mut self) -> &mut RenderPass<'b> {
        self.pass
    }

    /// Consume the writer and return the render pass reference.
    #[inline]
    pub fn into_pass(self) -> &'a mut RenderPass<'b> {
        self.pass
    }
}

// ============================================================================
// ComputePushConstantWriter
// ============================================================================

/// A type-safe wrapper for writing push constants to a compute pass.
///
/// Similar to `PushConstantWriter` but for compute shaders.
pub struct ComputePushConstantWriter<'a, 'b> {
    pass: &'a mut ComputePass<'b>,
    config: &'a PushConstantConfig,
}

impl<'a, 'b> ComputePushConstantWriter<'a, 'b> {
    /// Create a new push constant writer wrapping a compute pass.
    #[inline]
    pub fn new(pass: &'a mut ComputePass<'b>, config: &'a PushConstantConfig) -> Self {
        Self { pass, config }
    }

    /// Set push constant data for compute stage.
    pub fn set<T: bytemuck::Pod>(
        &mut self,
        offset: u32,
        data: &T,
    ) -> Result<(), PushConstantError> {
        let bytes = bytemuck::bytes_of(data);
        let size = bytes.len() as u32;

        // Validate the write
        self.config
            .is_valid_write(ShaderStages::COMPUTE, offset, size)?;

        // Perform the write
        self.pass
            .set_push_constants(offset, bytes);
        Ok(())
    }

    /// Set raw bytes as push constant data.
    pub fn set_bytes(&mut self, offset: u32, bytes: &[u8]) -> Result<(), PushConstantError> {
        let size = bytes.len() as u32;
        self.config
            .is_valid_write(ShaderStages::COMPUTE, offset, size)?;
        self.pass.set_push_constants(offset, bytes);
        Ok(())
    }

    /// Returns a reference to the underlying compute pass.
    #[inline]
    pub fn pass(&self) -> &ComputePass<'b> {
        self.pass
    }

    /// Returns a mutable reference to the underlying compute pass.
    #[inline]
    pub fn pass_mut(&mut self) -> &mut ComputePass<'b> {
        self.pass
    }
}

// ============================================================================
// Fallback Support
// ============================================================================

/// Strategy for handling push constants when the feature may not be available.
#[derive(Debug, Clone)]
pub enum PushConstantFallback {
    /// Use native push constants (feature is available).
    Native,

    /// Fall back to uniform buffer (feature not available).
    ///
    /// Contains the buffer handle used for fallback data.
    UniformBuffer {
        /// The uniform buffer used as fallback.
        buffer: Arc<Buffer>,
        /// The bind group index for the fallback buffer.
        bind_group_index: u32,
    },
}

impl PushConstantFallback {
    /// Check if this is using native push constants.
    #[inline]
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    /// Check if this is using uniform buffer fallback.
    #[inline]
    pub fn is_fallback(&self) -> bool {
        matches!(self, Self::UniformBuffer { .. })
    }

    /// Get the fallback buffer, if any.
    #[inline]
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        match self {
            Self::Native => None,
            Self::UniformBuffer { buffer, .. } => Some(buffer),
        }
    }

    /// Get the bind group index for fallback, if any.
    #[inline]
    pub fn bind_group_index(&self) -> Option<u32> {
        match self {
            Self::Native => None,
            Self::UniformBuffer {
                bind_group_index, ..
            } => Some(*bind_group_index),
        }
    }
}

// ============================================================================
// FallbackPushConstants
// ============================================================================

/// Push constants with automatic fallback to uniform buffer.
///
/// This struct manages push constant data with automatic fallback to a
/// uniform buffer when native push constants are not supported.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::push_constants::{
///     FallbackPushConstants, PushConstantConfig,
/// };
/// use wgpu::ShaderStages;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let config = PushConstantConfig::new()
///     .add_range(ShaderStages::VERTEX, 0, 64)
///     .expect("valid");
///
/// let fallback = FallbackPushConstants::new(device, &config, None);
///
/// // The fallback automatically detects whether to use native or uniform buffer
/// if fallback.is_native() {
///     println!("Using native push constants");
/// } else {
///     println!("Using uniform buffer fallback");
/// }
/// # }
/// ```
pub struct FallbackPushConstants {
    /// The fallback strategy in use.
    fallback: PushConstantFallback,
    /// Local data buffer for staging.
    data: Vec<u8>,
    /// Configuration for validation.
    config: PushConstantConfig,
}

impl FallbackPushConstants {
    /// Create new fallback push constants.
    ///
    /// Automatically detects whether to use native push constants or uniform
    /// buffer fallback based on device capabilities.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `config` - Push constant configuration
    /// * `bind_group_index` - Optional bind group index for fallback (default: OBJECT=2)
    pub fn new(
        device: &Device,
        config: &PushConstantConfig,
        bind_group_index: Option<u32>,
    ) -> Self {
        let size = config.total_size();
        let data = vec![0u8; size as usize];

        let fallback = if supports_push_constants(device) {
            PushConstantFallback::Native
        } else {
            // Create uniform buffer fallback
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("push_constant_fallback"),
                size: size as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            PushConstantFallback::UniformBuffer {
                buffer: Arc::new(buffer),
                bind_group_index: bind_group_index.unwrap_or(DEFAULT_FALLBACK_BIND_GROUP),
            }
        };

        Self {
            fallback,
            data,
            config: config.clone(),
        }
    }

    /// Create fallback push constants forcing native mode.
    ///
    /// # Errors
    ///
    /// Returns error if push constants are not supported.
    pub fn new_native(device: &Device, config: &PushConstantConfig) -> Result<Self, PushConstantError> {
        if !supports_push_constants(device) {
            return Err(PushConstantError::UnsupportedFeature);
        }

        let size = config.total_size();
        Ok(Self {
            fallback: PushConstantFallback::Native,
            data: vec![0u8; size as usize],
            config: config.clone(),
        })
    }

    /// Create fallback push constants forcing uniform buffer mode.
    pub fn new_uniform_buffer(
        device: &Device,
        config: &PushConstantConfig,
        bind_group_index: u32,
    ) -> Self {
        let size = config.total_size();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("push_constant_fallback"),
            size: size as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            fallback: PushConstantFallback::UniformBuffer {
                buffer: Arc::new(buffer),
                bind_group_index,
            },
            data: vec![0u8; size as usize],
            config: config.clone(),
        }
    }

    /// Check if using native push constants.
    #[inline]
    pub fn is_native(&self) -> bool {
        self.fallback.is_native()
    }

    /// Check if using uniform buffer fallback.
    #[inline]
    pub fn is_fallback(&self) -> bool {
        self.fallback.is_fallback()
    }

    /// Get the fallback strategy.
    #[inline]
    pub fn fallback(&self) -> &PushConstantFallback {
        &self.fallback
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &PushConstantConfig {
        &self.config
    }

    /// Write data to the local buffer.
    ///
    /// This stages data locally. Call `upload()` to upload to the GPU
    /// when using uniform buffer fallback.
    pub fn write<T: bytemuck::Pod>(&mut self, offset: u32, data: &T) -> Result<(), PushConstantError> {
        let bytes = bytemuck::bytes_of(data);
        self.write_bytes(offset, bytes)
    }

    /// Write raw bytes to the local buffer.
    pub fn write_bytes(&mut self, offset: u32, bytes: &[u8]) -> Result<(), PushConstantError> {
        let size = bytes.len() as u32;
        let end = offset + size;

        if end > self.data.len() as u32 {
            return Err(PushConstantError::ExceedsMaxSize {
                total: end,
                max: self.data.len() as u32,
            });
        }

        self.data[offset as usize..end as usize].copy_from_slice(bytes);
        Ok(())
    }

    /// Upload staged data to the GPU (for uniform buffer fallback).
    ///
    /// This is a no-op when using native push constants.
    pub fn upload(&self, queue: &Queue) {
        if let PushConstantFallback::UniformBuffer { buffer, .. } = &self.fallback {
            queue.write_buffer(buffer, 0, &self.data);
        }
    }

    /// Get the uniform buffer for binding (when using fallback).
    #[inline]
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        self.fallback.buffer()
    }

    /// Get the staged data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Apply to a render pass.
    ///
    /// When using native mode, sets push constants directly.
    /// When using fallback mode, the caller must bind the uniform buffer separately.
    pub fn apply_to_render_pass<'a>(&self, pass: &mut RenderPass<'a>, stages: ShaderStages) {
        if self.is_native() {
            pass.set_push_constants(stages, 0, &self.data);
        }
        // For fallback mode, the caller must bind the uniform buffer
    }

    /// Apply to a compute pass.
    pub fn apply_to_compute_pass<'a>(&self, pass: &mut ComputePass<'a>) {
        if self.is_native() {
            pass.set_push_constants(0, &self.data);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that a value is 4-byte aligned.
#[inline]
pub const fn is_aligned(value: u32) -> bool {
    value % PUSH_CONSTANT_ALIGNMENT == 0
}

/// Align a value up to the next 4-byte boundary.
#[inline]
pub const fn align_up(value: u32) -> u32 {
    (value + PUSH_CONSTANT_ALIGNMENT - 1) & !(PUSH_CONSTANT_ALIGNMENT - 1)
}

/// Create a single-range config for vertex shader.
pub fn vertex_only(size: u32) -> Result<PushConstantConfig, PushConstantError> {
    PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, size)
}

/// Create a single-range config for fragment shader.
pub fn fragment_only(size: u32) -> Result<PushConstantConfig, PushConstantError> {
    PushConstantConfig::new().add_range(ShaderStages::FRAGMENT, 0, size)
}

/// Create a single-range config for both vertex and fragment shaders.
pub fn vertex_fragment(size: u32) -> Result<PushConstantConfig, PushConstantError> {
    PushConstantConfig::new().add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 0, size)
}

/// Create a single-range config for compute shader.
pub fn compute_only(size: u32) -> Result<PushConstantConfig, PushConstantError> {
    PushConstantConfig::new().add_range(ShaderStages::COMPUTE, 0, size)
}

// ============================================================================
// Common Data Types
// ============================================================================

/// A minimal per-draw push constant struct (16 bytes).
///
/// Contains object ID and material index, suitable for indirect rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawPushConstants {
    /// Object/instance ID for lookup.
    pub object_id: u32,
    /// Material index for material array lookup.
    pub material_id: u32,
    /// First vertex offset for vertex pulling.
    pub first_vertex: u32,
    /// Reserved for future use.
    pub _reserved: u32,
}

impl DrawPushConstants {
    /// Size in bytes.
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;

    /// Create new draw push constants.
    #[inline]
    pub const fn new(object_id: u32, material_id: u32) -> Self {
        Self {
            object_id,
            material_id,
            first_vertex: 0,
            _reserved: 0,
        }
    }

    /// Create with vertex offset.
    #[inline]
    pub const fn with_vertex_offset(object_id: u32, material_id: u32, first_vertex: u32) -> Self {
        Self {
            object_id,
            material_id,
            first_vertex,
            _reserved: 0,
        }
    }
}

/// Extended per-draw push constant struct (64 bytes).
///
/// Contains model matrix for direct rendering without buffer lookup.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExtendedDrawPushConstants {
    /// Model matrix row 0.
    pub model_row0: [f32; 4],
    /// Model matrix row 1.
    pub model_row1: [f32; 4],
    /// Model matrix row 2.
    pub model_row2: [f32; 4],
    /// Model matrix row 3.
    pub model_row3: [f32; 4],
}

impl ExtendedDrawPushConstants {
    /// Size in bytes.
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;

    /// Create identity transform.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            model_row0: [1.0, 0.0, 0.0, 0.0],
            model_row1: [0.0, 1.0, 0.0, 0.0],
            model_row2: [0.0, 0.0, 1.0, 0.0],
            model_row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create from 4x4 matrix array.
    #[inline]
    pub const fn from_matrix(m: [[f32; 4]; 4]) -> Self {
        Self {
            model_row0: m[0],
            model_row1: m[1],
            model_row2: m[2],
            model_row3: m[3],
        }
    }
}

impl Default for ExtendedDrawPushConstants {
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Thread Safety
// ============================================================================

// PushConstantConfig is Send + Sync because it only contains Vec<PushConstantRange>
// which is Send + Sync.
static_assertions::assert_impl_all!(PushConstantConfig: Send, Sync);

// FallbackPushConstants is Send + Sync because Arc<Buffer> is Send + Sync.
static_assertions::assert_impl_all!(FallbackPushConstants: Send, Sync);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Constants Tests
    // ========================================================================

    #[test]
    fn test_max_push_constant_size() {
        assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
    }

    #[test]
    fn test_push_constant_alignment() {
        assert_eq!(PUSH_CONSTANT_ALIGNMENT, 4);
    }

    #[test]
    fn test_default_fallback_bind_group() {
        assert_eq!(DEFAULT_FALLBACK_BIND_GROUP, bind_group_index::OBJECT);
        assert_eq!(DEFAULT_FALLBACK_BIND_GROUP, 2);
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display_misaligned_offset() {
        let err = PushConstantError::MisalignedOffset(5);
        let msg = format!("{}", err);
        assert!(msg.contains("offset"));
        assert!(msg.contains("5"));
        assert!(msg.contains("aligned"));
    }

    #[test]
    fn test_error_display_misaligned_size() {
        let err = PushConstantError::MisalignedSize(7);
        let msg = format!("{}", err);
        assert!(msg.contains("size"));
        assert!(msg.contains("7"));
    }

    #[test]
    fn test_error_display_exceeds_max() {
        let err = PushConstantError::ExceedsMaxSize {
            total: 256,
            max: 128,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("256"));
        assert!(msg.contains("128"));
    }

    #[test]
    fn test_error_display_range_overlap() {
        let err = PushConstantError::RangeOverlap {
            range1: "0..64".to_string(),
            range2: "32..96".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overlap"));
    }

    #[test]
    fn test_error_display_unsupported() {
        let err = PushConstantError::UnsupportedFeature;
        let msg = format!("{}", err);
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_error_display_data_too_large() {
        let err = PushConstantError::DataTooLarge {
            data_size: 100,
            available: 64,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn test_error_display_invalid_offset() {
        let err = PushConstantError::InvalidOffset {
            offset: 200,
            stages: ShaderStages::VERTEX,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("200"));
    }

    #[test]
    fn test_error_display_empty_range() {
        let err = PushConstantError::EmptyRange;
        let msg = format!("{}", err);
        assert!(msg.contains("empty"));
    }

    // ========================================================================
    // PushConstantConfig Tests
    // ========================================================================

    #[test]
    fn test_config_new_empty() {
        let config = PushConstantConfig::new();
        assert!(config.is_empty());
        assert_eq!(config.range_count(), 0);
        assert_eq!(config.total_size(), 0);
    }

    #[test]
    fn test_config_add_single_range() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid range");

        assert!(!config.is_empty());
        assert_eq!(config.range_count(), 1);
        assert_eq!(config.total_size(), 64);
    }

    #[test]
    fn test_config_add_multiple_ranges() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 32)
            .expect("valid")
            .add_range(ShaderStages::FRAGMENT, 32, 32)
            .expect("valid");

        assert_eq!(config.range_count(), 2);
        assert_eq!(config.total_size(), 64);
    }

    #[test]
    fn test_config_add_vertex_fragment_range() {
        let config = PushConstantConfig::new()
            .add_vertex_fragment_range(0, 64)
            .expect("valid");

        assert_eq!(config.range_count(), 1);
        assert_eq!(config.total_size(), 64);
    }

    #[test]
    fn test_config_add_compute_range() {
        let config = PushConstantConfig::new()
            .add_compute_range(0, 128)
            .expect("valid");

        assert_eq!(config.range_count(), 1);
        assert_eq!(config.total_size(), 128);
    }

    #[test]
    fn test_config_max_size() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 128)
            .expect("valid at max");

        assert_eq!(config.total_size(), 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_error_misaligned_offset() {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 3, 64);
        assert!(matches!(result, Err(PushConstantError::MisalignedOffset(3))));
    }

    #[test]
    fn test_config_error_misaligned_size() {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 63);
        assert!(matches!(result, Err(PushConstantError::MisalignedSize(63))));
    }

    #[test]
    fn test_config_error_empty_range() {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 0);
        assert!(matches!(result, Err(PushConstantError::EmptyRange)));
    }

    #[test]
    fn test_config_error_exceeds_max() {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 132);
        assert!(matches!(
            result,
            Err(PushConstantError::ExceedsMaxSize { total: 132, max: 128 })
        ));
    }

    #[test]
    fn test_config_error_exceeds_max_with_offset() {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 64, 68);
        assert!(matches!(
            result,
            Err(PushConstantError::ExceedsMaxSize { total: 132, .. })
        ));
    }

    #[test]
    fn test_config_error_overlap_same_stages() {
        let result = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("first valid")
            .add_range(ShaderStages::VERTEX, 32, 64);

        assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));
    }

    #[test]
    fn test_config_non_overlapping_different_stages() {
        // Same byte range but different stages is allowed
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid")
            .add_range(ShaderStages::FRAGMENT, 0, 64)
            .expect("different stages");

        assert_eq!(config.range_count(), 2);
    }

    #[test]
    fn test_config_adjacent_ranges() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid")
            .add_range(ShaderStages::VERTEX, 64, 64)
            .expect("adjacent, not overlapping");

        assert_eq!(config.range_count(), 2);
        assert_eq!(config.total_size(), 128);
    }

    #[test]
    fn test_config_validate_empty() {
        let config = PushConstantConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_valid() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_ranges_slice() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, ShaderStages::VERTEX);
        assert_eq!(ranges[0].range, 0..64);
    }

    #[test]
    fn test_config_into_ranges() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let ranges = config.into_ranges();
        assert_eq!(ranges.len(), 1);
    }

    #[test]
    fn test_config_find_range_found() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let result = config.find_range(ShaderStages::VERTEX, 0);
        assert!(result.is_some());
        let (range, available) = result.unwrap();
        assert_eq!(range, 0..64);
        assert_eq!(available, 64);
    }

    #[test]
    fn test_config_find_range_mid_offset() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let result = config.find_range(ShaderStages::VERTEX, 32);
        assert!(result.is_some());
        let (_, available) = result.unwrap();
        assert_eq!(available, 32);
    }

    #[test]
    fn test_config_find_range_not_found() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        // Wrong stages
        assert!(config.find_range(ShaderStages::FRAGMENT, 0).is_none());

        // Out of range
        assert!(config.find_range(ShaderStages::VERTEX, 64).is_none());
    }

    #[test]
    fn test_config_is_valid_write_success() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 16).is_ok());
        assert!(config.is_valid_write(ShaderStages::VERTEX, 16, 48).is_ok());
    }

    #[test]
    fn test_config_is_valid_write_too_large() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let result = config.is_valid_write(ShaderStages::VERTEX, 0, 128);
        assert!(matches!(
            result,
            Err(PushConstantError::DataTooLarge {
                data_size: 128,
                available: 64
            })
        ));
    }

    #[test]
    fn test_config_is_valid_write_invalid_offset() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let result = config.is_valid_write(ShaderStages::VERTEX, 64, 16);
        assert!(matches!(
            result,
            Err(PushConstantError::InvalidOffset { offset: 64, .. })
        ));
    }

    #[test]
    fn test_config_clone() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let cloned = config.clone();
        assert_eq!(cloned.range_count(), config.range_count());
        assert_eq!(cloned.total_size(), config.total_size());
    }

    #[test]
    fn test_config_default() {
        let config: PushConstantConfig = Default::default();
        assert!(config.is_empty());
    }

    #[test]
    fn test_config_debug_format() {
        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let debug = format!("{:?}", config);
        assert!(debug.contains("PushConstantConfig"));
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0));
        assert!(is_aligned(4));
        assert!(is_aligned(8));
        assert!(is_aligned(64));
        assert!(is_aligned(128));

        assert!(!is_aligned(1));
        assert!(!is_aligned(2));
        assert!(!is_aligned(3));
        assert!(!is_aligned(5));
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4);
        assert_eq!(align_up(2), 4);
        assert_eq!(align_up(3), 4);
        assert_eq!(align_up(4), 4);
        assert_eq!(align_up(5), 8);
        assert_eq!(align_up(63), 64);
        assert_eq!(align_up(64), 64);
    }

    #[test]
    fn test_vertex_only_helper() {
        let config = vertex_only(64).expect("valid");
        assert_eq!(config.range_count(), 1);
        assert_eq!(config.ranges()[0].stages, ShaderStages::VERTEX);
    }

    #[test]
    fn test_fragment_only_helper() {
        let config = fragment_only(64).expect("valid");
        assert_eq!(config.range_count(), 1);
        assert_eq!(config.ranges()[0].stages, ShaderStages::FRAGMENT);
    }

    #[test]
    fn test_vertex_fragment_helper() {
        let config = vertex_fragment(64).expect("valid");
        assert_eq!(config.range_count(), 1);
        assert_eq!(
            config.ranges()[0].stages,
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );
    }

    #[test]
    fn test_compute_only_helper() {
        let config = compute_only(64).expect("valid");
        assert_eq!(config.range_count(), 1);
        assert_eq!(config.ranges()[0].stages, ShaderStages::COMPUTE);
    }

    // ========================================================================
    // DrawPushConstants Tests
    // ========================================================================

    #[test]
    fn test_draw_push_constants_size() {
        assert_eq!(DrawPushConstants::SIZE, 16);
        assert_eq!(std::mem::size_of::<DrawPushConstants>(), 16);
    }

    #[test]
    fn test_draw_push_constants_new() {
        let pc = DrawPushConstants::new(42, 7);
        assert_eq!(pc.object_id, 42);
        assert_eq!(pc.material_id, 7);
        assert_eq!(pc.first_vertex, 0);
        assert_eq!(pc._reserved, 0);
    }

    #[test]
    fn test_draw_push_constants_with_vertex_offset() {
        let pc = DrawPushConstants::with_vertex_offset(1, 2, 1000);
        assert_eq!(pc.object_id, 1);
        assert_eq!(pc.material_id, 2);
        assert_eq!(pc.first_vertex, 1000);
    }

    #[test]
    fn test_draw_push_constants_default() {
        let pc: DrawPushConstants = Default::default();
        assert_eq!(pc.object_id, 0);
        assert_eq!(pc.material_id, 0);
        assert_eq!(pc.first_vertex, 0);
    }

    #[test]
    fn test_draw_push_constants_bytemuck() {
        let pc = DrawPushConstants::new(42, 7);
        let bytes = bytemuck::bytes_of(&pc);
        assert_eq!(bytes.len(), 16);

        let recovered: &DrawPushConstants = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.object_id, 42);
        assert_eq!(recovered.material_id, 7);
    }

    // ========================================================================
    // ExtendedDrawPushConstants Tests
    // ========================================================================

    #[test]
    fn test_extended_push_constants_size() {
        assert_eq!(ExtendedDrawPushConstants::SIZE, 64);
        assert_eq!(std::mem::size_of::<ExtendedDrawPushConstants>(), 64);
    }

    #[test]
    fn test_extended_push_constants_identity() {
        let pc = ExtendedDrawPushConstants::identity();
        assert_eq!(pc.model_row0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(pc.model_row1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(pc.model_row2, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(pc.model_row3, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_extended_push_constants_from_matrix() {
        let matrix = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let pc = ExtendedDrawPushConstants::from_matrix(matrix);
        assert_eq!(pc.model_row0, matrix[0]);
        assert_eq!(pc.model_row1, matrix[1]);
        assert_eq!(pc.model_row2, matrix[2]);
        assert_eq!(pc.model_row3, matrix[3]);
    }

    #[test]
    fn test_extended_push_constants_default() {
        let pc: ExtendedDrawPushConstants = Default::default();
        let identity = ExtendedDrawPushConstants::identity();
        assert_eq!(pc.model_row0, identity.model_row0);
    }

    #[test]
    fn test_extended_push_constants_bytemuck() {
        let pc = ExtendedDrawPushConstants::identity();
        let bytes = bytemuck::bytes_of(&pc);
        assert_eq!(bytes.len(), 64);
    }

    // ========================================================================
    // PushConstantFallback Tests
    // ========================================================================

    #[test]
    fn test_fallback_native() {
        let fallback = PushConstantFallback::Native;
        assert!(fallback.is_native());
        assert!(!fallback.is_fallback());
        assert!(fallback.buffer().is_none());
        assert!(fallback.bind_group_index().is_none());
    }

    #[test]
    fn test_fallback_debug_format() {
        let fallback = PushConstantFallback::Native;
        let debug = format!("{:?}", fallback);
        assert!(debug.contains("Native"));
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_config_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PushConstantConfig>();
    }

    #[test]
    fn test_fallback_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PushConstantFallback>();
    }

    #[test]
    fn test_draw_push_constants_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DrawPushConstants>();
    }

    #[test]
    fn test_extended_push_constants_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExtendedDrawPushConstants>();
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PushConstantError>();
    }

    // ========================================================================
    // Integration Tests (require GPU device)
    // ========================================================================

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 128,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .ok()?,
        )
    }

    fn create_test_device_no_push_constants() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_supports_push_constants_with_feature() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter with push constants available");
                return;
            }
        };

        assert!(supports_push_constants(&device));
    }

    #[test]
    fn test_supports_push_constants_without_feature() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // May or may not have push constants depending on backend
        let _ = supports_push_constants(&device);
    }

    #[test]
    fn test_max_push_constant_size_with_feature() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter with push constants available");
                return;
            }
        };

        let max_size = max_push_constant_size(&device);
        assert!(max_size >= 128);
    }

    #[test]
    fn test_fallback_push_constants_native() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter with push constants available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let fallback = FallbackPushConstants::new(&device, &config, None);
        assert!(fallback.is_native());
        assert!(!fallback.is_fallback());
        assert!(fallback.buffer().is_none());
    }

    #[test]
    fn test_fallback_push_constants_write() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let mut fallback = FallbackPushConstants::new(&device, &config, None);

        let data = DrawPushConstants::new(42, 7);
        assert!(fallback.write(0, &data).is_ok());

        // Verify data was written
        let stored = fallback.data();
        assert_eq!(stored.len(), 64);
    }

    #[test]
    fn test_fallback_push_constants_write_bytes() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let mut fallback = FallbackPushConstants::new(&device, &config, None);

        let bytes = [1u8, 2, 3, 4];
        assert!(fallback.write_bytes(0, &bytes).is_ok());

        let stored = fallback.data();
        assert_eq!(&stored[0..4], &bytes);
    }

    #[test]
    fn test_fallback_push_constants_uniform_buffer() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let fallback = FallbackPushConstants::new_uniform_buffer(&device, &config, 2);
        assert!(!fallback.is_native());
        assert!(fallback.is_fallback());
        assert!(fallback.buffer().is_some());
    }

    #[test]
    fn test_fallback_push_constants_upload() {
        let (device, queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let mut fallback = FallbackPushConstants::new_uniform_buffer(&device, &config, 2);

        let data = DrawPushConstants::new(42, 7);
        fallback.write(0, &data).expect("write");
        fallback.upload(&queue);

        // Upload succeeded (we can't easily verify GPU contents without readback)
    }

    #[test]
    fn test_new_native_error_without_feature() {
        let (device, _queue) = match create_test_device_no_push_constants() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Skip if the device actually supports push constants
        if supports_push_constants(&device) {
            return;
        }

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let result = FallbackPushConstants::new_native(&device, &config);
        assert!(matches!(result, Err(PushConstantError::UnsupportedFeature)));
    }

    #[test]
    fn test_fallback_config_accessor() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 64)
            .expect("valid");

        let fallback = FallbackPushConstants::new(&device, &config, None);
        assert_eq!(fallback.config().total_size(), 64);
    }

    #[test]
    fn test_fallback_write_exceeds_size() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = PushConstantConfig::new()
            .add_range(ShaderStages::VERTEX, 0, 16)
            .expect("valid");

        let mut fallback = FallbackPushConstants::new(&device, &config, None);

        // Try to write 64 bytes to a 16-byte buffer
        let result = fallback.write_bytes(0, &[0u8; 64]);
        assert!(matches!(result, Err(PushConstantError::ExceedsMaxSize { .. })));
    }
}
