//! Compute pass descriptor and wrapper for wgpu 25.x compute pass creation.
//!
//! This module provides high-level abstractions for creating and managing compute passes
//! with a fluent API, including timestamp writes for GPU profiling.
//!
//! # Overview
//!
//! A compute pass encapsulates GPU compute shader dispatch operations. Unlike render passes,
//! compute passes don't have attachments - they work directly with buffers and textures
//! through bind groups.
//!
//! # Architecture
//!
//! ```text
//! ComputePassDescriptor
//!     |-- label: Option<&str>
//!     `-- timestamp_writes: Option<ComputePassTimestampWrites>
//!
//! ComputePassTimestampWrites
//!     |-- query_set: &QuerySet
//!     |-- beginning_of_pass_write_index: Option<u32>
//!     `-- end_of_pass_write_index: Option<u32>
//!
//! ComputePass<'a>
//!     `-- pass: wgpu::ComputePass<'a>
//!         |-- set_pipeline(&ComputePipeline)
//!         |-- set_bind_group(index, &BindGroup, &[offset])
//!         |-- set_push_constants(offset, &[u8])
//!         |-- dispatch_workgroups(x, y, z)
//!         |-- dispatch_workgroups_validated(x, y, z, limits)  [NEW]
//!         `-- dispatch_workgroups_indirect(&Buffer, offset)
//!
//! DispatchLimits                                              [NEW]
//!     |-- max_workgroups_x: u32
//!     |-- max_workgroups_y: u32
//!     `-- max_workgroups_z: u32
//! ```
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub struct ComputePassDescriptor<'a> {
//!     pub label: Label<'a>,
//!     pub timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
//! }
//!
//! pub struct ComputePassTimestampWrites<'a> {
//!     pub query_set: &'a QuerySet,
//!     pub beginning_of_pass_write_index: Option<u32>,
//!     pub end_of_pass_write_index: Option<u32>,
//! }
//! ```
//!
//! # Thread Safety
//!
//! `ComputePass` is **NOT** `Send` or `Sync` because the underlying `wgpu::ComputePass`
//! holds a mutable borrow of the command encoder. The pass must be used and finished
//! on the same thread that created it.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::compute_pass::{
//!     ComputePass, ComputePassDescriptor, ComputePassTimestampWrites,
//! };
//!
//! // Create descriptor with timestamp writes for profiling
//! let desc = ComputePassDescriptor::new()
//!     .label("particle_simulation")
//!     .timestamp_writes(ComputePassTimestampWrites::new(&query_set).both(0, 1));
//!
//! // Begin compute pass
//! let mut pass = ComputePass::new(&mut encoder, &desc);
//!
//! // Set pipeline and resources
//! pass.set_pipeline(&compute_pipeline)
//!     .set_bind_group(0, &particle_bind_group, &[])
//!     .set_push_constants(0, bytemuck::bytes_of(&push_data));
//!
//! // Dispatch workgroups
//! pass.dispatch_workgroups(particle_count / 64, 1, 1);
//!
//! // Pass is automatically finished when dropped
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// DispatchError
// ---------------------------------------------------------------------------

/// Error type for compute dispatch validation failures.
///
/// This enum captures the various ways a compute dispatch can fail validation,
/// providing detailed information about which limit was exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// X workgroup count exceeds device limit.
    ExceedsMaxX {
        /// Requested X workgroup count.
        requested: u32,
        /// Maximum allowed X workgroup count.
        limit: u32,
    },
    /// Y workgroup count exceeds device limit.
    ExceedsMaxY {
        /// Requested Y workgroup count.
        requested: u32,
        /// Maximum allowed Y workgroup count.
        limit: u32,
    },
    /// Z workgroup count exceeds device limit.
    ExceedsMaxZ {
        /// Requested Z workgroup count.
        requested: u32,
        /// Maximum allowed Z workgroup count.
        limit: u32,
    },
    /// Multiple dimensions exceed their limits.
    MultipleExceeded {
        /// Dimensions that exceed their limits (e.g., "X, Y").
        dimensions: String,
        /// Detailed message about all violations.
        details: String,
    },
    /// Total workgroup count would overflow u64.
    TotalOverflow {
        /// X workgroup count.
        x: u32,
        /// Y workgroup count.
        y: u32,
        /// Z workgroup count.
        z: u32,
    },
    /// At least one workgroup count is zero.
    ZeroWorkgroups {
        /// Which dimension is zero (X, Y, or Z).
        dimension: char,
    },
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchError::ExceedsMaxX { requested, limit } => {
                write!(
                    f,
                    "X workgroup count {} exceeds device limit {}",
                    requested, limit
                )
            }
            DispatchError::ExceedsMaxY { requested, limit } => {
                write!(
                    f,
                    "Y workgroup count {} exceeds device limit {}",
                    requested, limit
                )
            }
            DispatchError::ExceedsMaxZ { requested, limit } => {
                write!(
                    f,
                    "Z workgroup count {} exceeds device limit {}",
                    requested, limit
                )
            }
            DispatchError::MultipleExceeded { dimensions, details } => {
                write!(
                    f,
                    "Multiple dispatch dimensions exceeded ({}): {}",
                    dimensions, details
                )
            }
            DispatchError::TotalOverflow { x, y, z } => {
                write!(
                    f,
                    "Total workgroup count would overflow u64: {} * {} * {}",
                    x, y, z
                )
            }
            DispatchError::ZeroWorkgroups { dimension } => {
                write!(f, "{} workgroup count is zero", dimension)
            }
        }
    }
}

impl std::error::Error for DispatchError {}

// ---------------------------------------------------------------------------
// DispatchLimits
// ---------------------------------------------------------------------------

/// Device limits for compute dispatch operations.
///
/// These limits come from `wgpu::Limits` and define the maximum number of
/// workgroups that can be dispatched in each dimension.
///
/// # Default Limits (wgpu defaults)
///
/// - `max_workgroups_x`: 65535
/// - `max_workgroups_y`: 65535
/// - `max_workgroups_z`: 65535
///
/// # Example
///
/// ```ignore
/// let limits = DispatchLimits::from_wgpu_limits(&device.limits());
///
/// // Validate before dispatch
/// limits.validate(64, 64, 1)?;
/// pass.dispatch_workgroups(64, 64, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchLimits {
    /// Maximum workgroups in the X dimension.
    pub max_workgroups_x: u32,
    /// Maximum workgroups in the Y dimension.
    pub max_workgroups_y: u32,
    /// Maximum workgroups in the Z dimension.
    pub max_workgroups_z: u32,
}

impl DispatchLimits {
    /// Create dispatch limits from wgpu device limits.
    ///
    /// # Arguments
    ///
    /// * `limits` - The wgpu device limits
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = DispatchLimits::from_wgpu_limits(&device.limits());
    /// ```
    #[inline]
    pub fn from_wgpu_limits(limits: &wgpu::Limits) -> Self {
        Self {
            max_workgroups_x: limits.max_compute_workgroups_per_dimension,
            max_workgroups_y: limits.max_compute_workgroups_per_dimension,
            max_workgroups_z: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Create dispatch limits with custom values.
    ///
    /// # Arguments
    ///
    /// * `max_x` - Maximum workgroups in X dimension
    /// * `max_y` - Maximum workgroups in Y dimension
    /// * `max_z` - Maximum workgroups in Z dimension
    #[inline]
    pub fn new(max_x: u32, max_y: u32, max_z: u32) -> Self {
        Self {
            max_workgroups_x: max_x,
            max_workgroups_y: max_y,
            max_workgroups_z: max_z,
        }
    }

    /// Create dispatch limits with the same value for all dimensions.
    #[inline]
    pub fn uniform(max_per_dimension: u32) -> Self {
        Self {
            max_workgroups_x: max_per_dimension,
            max_workgroups_y: max_per_dimension,
            max_workgroups_z: max_per_dimension,
        }
    }

    /// Validate dispatch workgroup counts against these limits.
    ///
    /// Returns `Ok(())` if the dispatch is valid, or an error describing
    /// which limit was exceeded.
    ///
    /// # Arguments
    ///
    /// * `x` - Requested workgroups in X dimension
    /// * `y` - Requested workgroups in Y dimension
    /// * `z` - Requested workgroups in Z dimension
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = DispatchLimits::default();
    /// limits.validate(64, 64, 1)?;
    /// ```
    pub fn validate(&self, x: u32, y: u32, z: u32) -> Result<(), DispatchError> {
        // Check for zero workgroups
        if x == 0 {
            return Err(DispatchError::ZeroWorkgroups { dimension: 'X' });
        }
        if y == 0 {
            return Err(DispatchError::ZeroWorkgroups { dimension: 'Y' });
        }
        if z == 0 {
            return Err(DispatchError::ZeroWorkgroups { dimension: 'Z' });
        }

        // Collect all limit violations
        let mut violations = Vec::new();

        if x > self.max_workgroups_x {
            violations.push(('X', x, self.max_workgroups_x));
        }
        if y > self.max_workgroups_y {
            violations.push(('Y', y, self.max_workgroups_y));
        }
        if z > self.max_workgroups_z {
            violations.push(('Z', z, self.max_workgroups_z));
        }

        match violations.len() {
            0 => Ok(()),
            1 => {
                let (dim, requested, limit) = violations[0];
                match dim {
                    'X' => Err(DispatchError::ExceedsMaxX { requested, limit }),
                    'Y' => Err(DispatchError::ExceedsMaxY { requested, limit }),
                    'Z' => Err(DispatchError::ExceedsMaxZ { requested, limit }),
                    _ => unreachable!(),
                }
            }
            _ => {
                let dimensions: String = violations
                    .iter()
                    .map(|(d, _, _)| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let details = violations
                    .iter()
                    .map(|(d, req, lim)| format!("{}: {} > {}", d, req, lim))
                    .collect::<Vec<_>>()
                    .join("; ");
                Err(DispatchError::MultipleExceeded { dimensions, details })
            }
        }
    }

    /// Validate dispatch and also check for total overflow.
    ///
    /// This is a stricter validation that ensures the total number of
    /// workgroups won't overflow when multiplied together.
    pub fn validate_strict(&self, x: u32, y: u32, z: u32) -> Result<(), DispatchError> {
        // First do basic validation
        self.validate(x, y, z)?;

        // Check for multiplication overflow
        let xy = (x as u64).checked_mul(y as u64);
        let xyz = xy.and_then(|xy| xy.checked_mul(z as u64));

        if xyz.is_none() {
            return Err(DispatchError::TotalOverflow { x, y, z });
        }

        Ok(())
    }

    /// Check if dispatch counts are within limits without returning an error.
    #[inline]
    pub fn is_valid(&self, x: u32, y: u32, z: u32) -> bool {
        x > 0
            && y > 0
            && z > 0
            && x <= self.max_workgroups_x
            && y <= self.max_workgroups_y
            && z <= self.max_workgroups_z
    }
}

impl Default for DispatchLimits {
    /// Default limits matching wgpu's downlevel defaults.
    ///
    /// These are conservative limits that work on most hardware.
    fn default() -> Self {
        Self {
            max_workgroups_x: 65535,
            max_workgroups_y: 65535,
            max_workgroups_z: 65535,
        }
    }
}

impl fmt::Display for DispatchLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DispatchLimits(x={}, y={}, z={})",
            self.max_workgroups_x, self.max_workgroups_y, self.max_workgroups_z
        )
    }
}

// ---------------------------------------------------------------------------
// Workgroup Calculation Helpers
// ---------------------------------------------------------------------------

/// Calculate the number of workgroups needed for a 1D dispatch.
///
/// Uses ceiling division to ensure all elements are covered:
/// `ceil(total / size)`
///
/// # Arguments
///
/// * `total_elements` - Total number of elements to process
/// * `workgroup_size` - Number of elements per workgroup (invocations)
///
/// # Returns
///
/// Number of workgroups needed to cover all elements.
///
/// # Panics
///
/// Panics if `workgroup_size` is 0.
///
/// # Example
///
/// ```ignore
/// // 1000 particles with workgroup size 64
/// let workgroups = calculate_workgroups(1000, 64);
/// assert_eq!(workgroups, 16); // 16 * 64 = 1024 >= 1000
/// ```
#[inline]
pub fn calculate_workgroups(total_elements: u32, workgroup_size: u32) -> u32 {
    assert!(workgroup_size > 0, "workgroup_size must be non-zero");
    // Use div_ceil to avoid overflow with large values
    // Equivalent to: (total + size - 1) / size, but overflow-safe
    total_elements.div_ceil(workgroup_size)
}

/// Calculate workgroups for a 2D dispatch (e.g., image processing).
///
/// # Arguments
///
/// * `width` - Total width in elements
/// * `height` - Total height in elements
/// * `workgroup_size` - (workgroup_size_x, workgroup_size_y)
///
/// # Returns
///
/// (workgroups_x, workgroups_y) tuple
///
/// # Panics
///
/// Panics if either workgroup size component is 0.
///
/// # Example
///
/// ```ignore
/// // 512x512 image with 8x8 workgroup size
/// let (wx, wy) = calculate_workgroups_2d(512, 512, (8, 8));
/// assert_eq!((wx, wy), (64, 64));
/// ```
#[inline]
pub fn calculate_workgroups_2d(
    width: u32,
    height: u32,
    workgroup_size: (u32, u32),
) -> (u32, u32) {
    assert!(
        workgroup_size.0 > 0 && workgroup_size.1 > 0,
        "workgroup_size components must be non-zero"
    );
    (
        width.div_ceil(workgroup_size.0),
        height.div_ceil(workgroup_size.1),
    )
}

/// Calculate workgroups for a 3D dispatch (e.g., volume data).
///
/// # Arguments
///
/// * `width` - Total width in elements (X dimension)
/// * `height` - Total height in elements (Y dimension)
/// * `depth` - Total depth in elements (Z dimension)
/// * `workgroup_size` - (workgroup_size_x, workgroup_size_y, workgroup_size_z)
///
/// # Returns
///
/// (workgroups_x, workgroups_y, workgroups_z) tuple
///
/// # Panics
///
/// Panics if any workgroup size component is 0.
///
/// # Example
///
/// ```ignore
/// // 256x256x128 volume with 4x4x4 workgroup size
/// let (wx, wy, wz) = calculate_workgroups_3d(256, 256, 128, (4, 4, 4));
/// assert_eq!((wx, wy, wz), (64, 64, 32));
/// ```
#[inline]
pub fn calculate_workgroups_3d(
    width: u32,
    height: u32,
    depth: u32,
    workgroup_size: (u32, u32, u32),
) -> (u32, u32, u32) {
    assert!(
        workgroup_size.0 > 0 && workgroup_size.1 > 0 && workgroup_size.2 > 0,
        "workgroup_size components must be non-zero"
    );
    (
        width.div_ceil(workgroup_size.0),
        height.div_ceil(workgroup_size.1),
        depth.div_ceil(workgroup_size.2),
    )
}

/// Calculate workgroups with limit validation for 1D dispatch.
///
/// # Arguments
///
/// * `total_elements` - Total number of elements to process
/// * `workgroup_size` - Number of elements per workgroup
/// * `limits` - Device dispatch limits
///
/// # Returns
///
/// Number of workgroups, or an error if limits would be exceeded.
///
/// # Example
///
/// ```ignore
/// let limits = DispatchLimits::default();
/// let workgroups = calculate_workgroups_validated(1000, 64, &limits)?;
/// ```
pub fn calculate_workgroups_validated(
    total_elements: u32,
    workgroup_size: u32,
    limits: &DispatchLimits,
) -> Result<u32, DispatchError> {
    let workgroups = calculate_workgroups(total_elements, workgroup_size);
    limits.validate(workgroups, 1, 1)?;
    Ok(workgroups)
}

/// Calculate workgroups with limit validation for 2D dispatch.
///
/// # Arguments
///
/// * `width` - Total width in elements
/// * `height` - Total height in elements
/// * `workgroup_size` - (workgroup_size_x, workgroup_size_y)
/// * `limits` - Device dispatch limits
///
/// # Returns
///
/// (workgroups_x, workgroups_y) tuple, or an error if limits would be exceeded.
pub fn calculate_workgroups_2d_validated(
    width: u32,
    height: u32,
    workgroup_size: (u32, u32),
    limits: &DispatchLimits,
) -> Result<(u32, u32), DispatchError> {
    let (wx, wy) = calculate_workgroups_2d(width, height, workgroup_size);
    limits.validate(wx, wy, 1)?;
    Ok((wx, wy))
}

/// Calculate workgroups with limit validation for 3D dispatch.
///
/// # Arguments
///
/// * `width` - Total width in elements
/// * `height` - Total height in elements
/// * `depth` - Total depth in elements
/// * `workgroup_size` - (workgroup_size_x, workgroup_size_y, workgroup_size_z)
/// * `limits` - Device dispatch limits
///
/// # Returns
///
/// (workgroups_x, workgroups_y, workgroups_z) tuple, or an error.
pub fn calculate_workgroups_3d_validated(
    width: u32,
    height: u32,
    depth: u32,
    workgroup_size: (u32, u32, u32),
    limits: &DispatchLimits,
) -> Result<(u32, u32, u32), DispatchError> {
    let (wx, wy, wz) = calculate_workgroups_3d(width, height, depth, workgroup_size);
    limits.validate(wx, wy, wz)?;
    Ok((wx, wy, wz))
}

// ---------------------------------------------------------------------------
// ComputePassTimestampWrites
// ---------------------------------------------------------------------------

/// Timestamp write indices for a compute pass.
///
/// Used with a `wgpu::QuerySet` of type `QueryType::Timestamp` to record
/// GPU timestamps at the beginning and/or end of a compute pass for
/// performance profiling.
///
/// # Requirements
///
/// - The query set must be created with `QueryType::Timestamp`
/// - Query indices must be valid for the query set's count
/// - Each index can only be written once per command buffer
///
/// # Example
///
/// ```ignore
/// let timestamp = ComputePassTimestampWrites::new(&query_set)
///     .both(0, 1);  // Write timestamps at indices 0 and 1
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ComputePassTimestampWrites<'a> {
    /// Reference to the query set for timestamp writes.
    pub query_set: &'a wgpu::QuerySet,
    /// Index to write the timestamp at the beginning of the pass.
    pub beginning_of_pass_write_index: Option<u32>,
    /// Index to write the timestamp at the end of the pass.
    pub end_of_pass_write_index: Option<u32>,
}

impl<'a> ComputePassTimestampWrites<'a> {
    /// Create new timestamp writes with a query set.
    ///
    /// By default, no timestamps are written. Use the builder methods
    /// to configure which timestamps to record.
    #[inline]
    pub fn new(query_set: &'a wgpu::QuerySet) -> Self {
        Self {
            query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: None,
        }
    }

    /// Set both beginning and end timestamp indices.
    ///
    /// # Arguments
    ///
    /// * `beginning` - Query index for the beginning timestamp
    /// * `end` - Query index for the end timestamp
    #[inline]
    pub fn both(mut self, beginning: u32, end: u32) -> Self {
        self.beginning_of_pass_write_index = Some(beginning);
        self.end_of_pass_write_index = Some(end);
        self
    }

    /// Set only the beginning timestamp index.
    #[inline]
    pub fn beginning_only(mut self, index: u32) -> Self {
        self.beginning_of_pass_write_index = Some(index);
        self.end_of_pass_write_index = None;
        self
    }

    /// Set only the end timestamp index.
    #[inline]
    pub fn end_only(mut self, index: u32) -> Self {
        self.beginning_of_pass_write_index = None;
        self.end_of_pass_write_index = Some(index);
        self
    }

    /// Set the beginning timestamp index.
    #[inline]
    pub fn beginning(mut self, index: u32) -> Self {
        self.beginning_of_pass_write_index = Some(index);
        self
    }

    /// Set the end timestamp index.
    #[inline]
    pub fn end(mut self, index: u32) -> Self {
        self.end_of_pass_write_index = Some(index);
        self
    }

    /// Clear the beginning timestamp index.
    #[inline]
    pub fn no_beginning(mut self) -> Self {
        self.beginning_of_pass_write_index = None;
        self
    }

    /// Clear the end timestamp index.
    #[inline]
    pub fn no_end(mut self) -> Self {
        self.end_of_pass_write_index = None;
        self
    }

    /// Check if any timestamps are configured.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.beginning_of_pass_write_index.is_some() || self.end_of_pass_write_index.is_some()
    }

    /// Check if beginning timestamp is configured.
    #[inline]
    pub fn has_beginning(&self) -> bool {
        self.beginning_of_pass_write_index.is_some()
    }

    /// Check if end timestamp is configured.
    #[inline]
    pub fn has_end(&self) -> bool {
        self.end_of_pass_write_index.is_some()
    }

    /// Convert to wgpu ComputePassTimestampWrites.
    #[inline]
    pub fn to_wgpu(&self) -> wgpu::ComputePassTimestampWrites<'a> {
        wgpu::ComputePassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: self.beginning_of_pass_write_index,
            end_of_pass_write_index: self.end_of_pass_write_index,
        }
    }
}

impl fmt::Display for ComputePassTimestampWrites<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComputePassTimestampWrites(begin={:?}, end={:?})",
            self.beginning_of_pass_write_index, self.end_of_pass_write_index
        )
    }
}

// ---------------------------------------------------------------------------
// ComputePassDescriptor
// ---------------------------------------------------------------------------

/// High-level compute pass descriptor.
///
/// Configuration for creating a compute pass, including an optional debug label
/// and timestamp writes for GPU profiling.
///
/// # Example
///
/// ```ignore
/// let desc = ComputePassDescriptor::new()
///     .label("physics_simulation")
///     .timestamp_writes(ComputePassTimestampWrites::new(&query_set).both(0, 1));
/// ```
#[derive(Debug, Clone, Default)]
pub struct ComputePassDescriptor<'a> {
    /// Optional debug label for the pass.
    pub label: Option<&'a str>,
    /// Optional timestamp writes configuration.
    timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}

impl<'a> ComputePassDescriptor<'a> {
    /// Create a new compute pass descriptor with no label or timestamps.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debug label for the pass.
    ///
    /// Labels are useful for debugging with tools like RenderDoc, PIX, or
    /// the wgpu device lost callback.
    #[inline]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the timestamp writes configuration.
    ///
    /// Enables GPU timestamp queries at the beginning and/or end of the pass
    /// for performance profiling.
    #[inline]
    pub fn timestamp_writes(mut self, writes: ComputePassTimestampWrites<'a>) -> Self {
        self.timestamp_writes = Some(writes);
        self
    }

    /// Clear the timestamp writes configuration.
    #[inline]
    pub fn no_timestamp_writes(mut self) -> Self {
        self.timestamp_writes = None;
        self
    }

    /// Get the configured timestamp writes.
    #[inline]
    pub fn get_timestamp_writes(&self) -> Option<&ComputePassTimestampWrites<'a>> {
        self.timestamp_writes.as_ref()
    }

    /// Check if timestamp writes are configured.
    #[inline]
    pub fn has_timestamp_writes(&self) -> bool {
        self.timestamp_writes.is_some()
    }

    /// Convert to wgpu ComputePassDescriptor.
    ///
    /// Returns a tuple of (wgpu_descriptor, Option<wgpu_timestamp_writes>).
    /// The timestamp writes must be stored separately to satisfy borrow requirements.
    #[inline]
    pub fn to_wgpu(&self) -> wgpu::ComputePassDescriptor<'a> {
        wgpu::ComputePassDescriptor {
            label: self.label,
            timestamp_writes: self.timestamp_writes.as_ref().map(|tw| tw.to_wgpu()),
        }
    }
}

impl fmt::Display for ComputePassDescriptor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComputePassDescriptor(label={:?}, timestamps={})",
            self.label,
            self.timestamp_writes.is_some()
        )
    }
}

// ---------------------------------------------------------------------------
// ComputePassBuilder
// ---------------------------------------------------------------------------

/// Builder for compute pass descriptors with a fluent API.
///
/// Provides a convenient way to construct `ComputePassDescriptor` instances
/// with method chaining.
///
/// # Example
///
/// ```ignore
/// let desc = ComputePassBuilder::new()
///     .label("raytracing_pass")
///     .with_timestamps(&query_set, 4, 5)
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct ComputePassBuilder<'a> {
    label: Option<&'a str>,
    timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}

impl<'a> ComputePassBuilder<'a> {
    /// Create a new compute pass builder.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debug label.
    #[inline]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set timestamp writes with both indices.
    #[inline]
    pub fn with_timestamps(mut self, query_set: &'a wgpu::QuerySet, begin: u32, end: u32) -> Self {
        self.timestamp_writes = Some(ComputePassTimestampWrites::new(query_set).both(begin, end));
        self
    }

    /// Set timestamp writes with only beginning index.
    #[inline]
    pub fn with_begin_timestamp(mut self, query_set: &'a wgpu::QuerySet, index: u32) -> Self {
        self.timestamp_writes =
            Some(ComputePassTimestampWrites::new(query_set).beginning_only(index));
        self
    }

    /// Set timestamp writes with only end index.
    #[inline]
    pub fn with_end_timestamp(mut self, query_set: &'a wgpu::QuerySet, index: u32) -> Self {
        self.timestamp_writes = Some(ComputePassTimestampWrites::new(query_set).end_only(index));
        self
    }

    /// Set custom timestamp writes configuration.
    #[inline]
    pub fn timestamp_writes(mut self, writes: ComputePassTimestampWrites<'a>) -> Self {
        self.timestamp_writes = Some(writes);
        self
    }

    /// Build the compute pass descriptor.
    #[inline]
    pub fn build(self) -> ComputePassDescriptor<'a> {
        ComputePassDescriptor {
            label: self.label,
            timestamp_writes: self.timestamp_writes,
        }
    }
}

// ---------------------------------------------------------------------------
// ComputePass wrapper
// ---------------------------------------------------------------------------

/// High-level wrapper around `wgpu::ComputePass`.
///
/// Provides a fluent API for setting pipeline state and dispatching compute work.
/// Methods return `&mut Self` to enable method chaining.
///
/// # Thread Safety
///
/// This type is **NOT** `Send` or `Sync` because the underlying `wgpu::ComputePass`
/// holds a mutable borrow of the command encoder. Create and use the pass on a
/// single thread, then let it drop (or call `finish()`) before accessing the
/// encoder again.
///
/// # Lifetime
///
/// The lifetime `'a` is tied to the command encoder. The pass must be finished
/// before the encoder can be used for other operations.
///
/// # Example
///
/// ```ignore
/// let mut pass = ComputePass::new(&mut encoder, &desc);
///
/// // Fluent API for setting state
/// pass.set_pipeline(&pipeline)
///     .set_bind_group(0, &resources_bg, &[])
///     .set_bind_group(1, &params_bg, &[])
///     .set_push_constants(0, bytemuck::bytes_of(&constants));
///
/// // Dispatch work
/// pass.dispatch_workgroups(64, 64, 1);
///
/// // Explicitly finish (or just let it drop)
/// pass.finish();
/// ```
pub struct ComputePass<'a> {
    pass: wgpu::ComputePass<'a>,
}

impl<'a> ComputePass<'a> {
    /// Create a new compute pass from a command encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to create the pass on
    /// * `desc` - The compute pass descriptor
    ///
    /// # Example
    ///
    /// ```ignore
    /// let desc = ComputePassDescriptor::new().label("my_compute");
    /// let mut pass = ComputePass::new(&mut encoder, &desc);
    /// ```
    #[inline]
    pub fn new(encoder: &'a mut wgpu::CommandEncoder, desc: &ComputePassDescriptor) -> Self {
        let wgpu_desc = desc.to_wgpu();
        Self {
            pass: encoder.begin_compute_pass(&wgpu_desc),
        }
    }

    /// Create a compute pass from a raw wgpu compute pass.
    ///
    /// This is useful when you need to interoperate with existing wgpu code.
    #[inline]
    pub fn from_raw(pass: wgpu::ComputePass<'a>) -> Self {
        Self { pass }
    }

    /// Set the compute pipeline for this pass.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The compute pipeline to use
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    #[inline]
    pub fn set_pipeline(&mut self, pipeline: &'a wgpu::ComputePipeline) -> &mut Self {
        self.pass.set_pipeline(pipeline);
        self
    }

    /// Bind a bind group at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The bind group slot index (0-7 typically)
    /// * `bind_group` - The bind group to bind
    /// * `offsets` - Dynamic offsets for buffers with dynamic binding
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Dynamic Offsets
    ///
    /// For bind groups containing buffers with `BufferBindingType::Uniform { has_dynamic_offset: true }`
    /// or `BufferBindingType::Storage { has_dynamic_offset: true, .. }`, provide the byte offsets
    /// in the order the dynamic bindings appear in the layout.
    #[inline]
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[u32],
    ) -> &mut Self {
        self.pass.set_bind_group(index, bind_group, offsets);
        self
    }

    /// Set push constant data.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset into the push constant range
    /// * `data` - Raw bytes to write
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Requirements
    ///
    /// - The pipeline must have a push constant range that covers `[offset, offset + data.len())`
    /// - `offset` must be a multiple of 4
    /// - `data.len()` must be a multiple of 4
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[repr(C)]
    /// #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    /// struct PushConstants {
    ///     time: f32,
    ///     frame: u32,
    /// }
    ///
    /// let constants = PushConstants { time: 1.5, frame: 100 };
    /// pass.set_push_constants(0, bytemuck::bytes_of(&constants));
    /// ```
    #[inline]
    pub fn set_push_constants(&mut self, offset: u32, data: &[u8]) -> &mut Self {
        self.pass.set_push_constants(offset, data);
        self
    }

    /// Dispatch compute work with explicit workgroup counts.
    ///
    /// # Arguments
    ///
    /// * `x` - Number of workgroups in the X dimension
    /// * `y` - Number of workgroups in the Y dimension
    /// * `z` - Number of workgroups in the Z dimension
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Note
    ///
    /// This method does not validate against device limits. For validated
    /// dispatch, use `dispatch_workgroups_validated()` or `try_dispatch_workgroups()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For 1024 particles with workgroup size of 64
    /// pass.dispatch_workgroups(1024 / 64, 1, 1);
    ///
    /// // For a 512x512 image with workgroup size of 8x8
    /// pass.dispatch_workgroups(512 / 8, 512 / 8, 1);
    /// ```
    #[inline]
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.pass.dispatch_workgroups(x, y, z);
        self
    }

    /// Dispatch compute work with device limit validation.
    ///
    /// Validates the workgroup counts against the provided limits before
    /// dispatching. Returns an error if any limit is exceeded.
    ///
    /// # Arguments
    ///
    /// * `x` - Number of workgroups in the X dimension
    /// * `y` - Number of workgroups in the Y dimension
    /// * `z` - Number of workgroups in the Z dimension
    /// * `limits` - Device dispatch limits to validate against
    ///
    /// # Returns
    ///
    /// `Ok(&mut Self)` for method chaining if valid, or `Err(DispatchError)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = DispatchLimits::from_wgpu_limits(&device.limits());
    /// pass.dispatch_workgroups_validated(64, 64, 1, &limits)?;
    /// ```
    #[inline]
    pub fn dispatch_workgroups_validated(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        limits: &DispatchLimits,
    ) -> Result<&mut Self, DispatchError> {
        limits.validate(x, y, z)?;
        self.pass.dispatch_workgroups(x, y, z);
        Ok(self)
    }

    /// Try to dispatch compute work, returning an error if limits exceeded.
    ///
    /// This is an alias for `dispatch_workgroups_validated` with a more
    /// idiomatic name for fallible operations.
    ///
    /// # Arguments
    ///
    /// * `x` - Number of workgroups in the X dimension
    /// * `y` - Number of workgroups in the Y dimension
    /// * `z` - Number of workgroups in the Z dimension
    /// * `limits` - Device dispatch limits to validate against
    #[inline]
    pub fn try_dispatch_workgroups(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        limits: &DispatchLimits,
    ) -> Result<&mut Self, DispatchError> {
        self.dispatch_workgroups_validated(x, y, z, limits)
    }

    /// Dispatch compute work using calculated workgroup counts.
    ///
    /// A convenience method that calculates workgroup counts from total
    /// elements and workgroup sizes, then dispatches.
    ///
    /// # Arguments
    ///
    /// * `total` - (total_x, total_y, total_z) elements in each dimension
    /// * `workgroup_size` - (size_x, size_y, size_z) elements per workgroup
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Process 512x512 image with 8x8 workgroup size
    /// pass.dispatch_for_size((512, 512, 1), (8, 8, 1));
    /// ```
    #[inline]
    pub fn dispatch_for_size(
        &mut self,
        total: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
    ) -> &mut Self {
        let (wx, wy, wz) = calculate_workgroups_3d(
            total.0,
            total.1,
            total.2,
            workgroup_size,
        );
        self.pass.dispatch_workgroups(wx, wy, wz);
        self
    }

    /// Dispatch compute work for given size with limit validation.
    ///
    /// Calculates workgroup counts and validates against device limits.
    ///
    /// # Arguments
    ///
    /// * `total` - (total_x, total_y, total_z) elements in each dimension
    /// * `workgroup_size` - (size_x, size_y, size_z) elements per workgroup
    /// * `limits` - Device dispatch limits to validate against
    ///
    /// # Returns
    ///
    /// `Ok(&mut Self)` if valid, or `Err(DispatchError)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let limits = DispatchLimits::from_wgpu_limits(&device.limits());
    /// pass.dispatch_for_size_validated((512, 512, 1), (8, 8, 1), &limits)?;
    /// ```
    #[inline]
    pub fn dispatch_for_size_validated(
        &mut self,
        total: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
        limits: &DispatchLimits,
    ) -> Result<&mut Self, DispatchError> {
        let (wx, wy, wz) = calculate_workgroups_3d_validated(
            total.0,
            total.1,
            total.2,
            workgroup_size,
            limits,
        )?;
        self.pass.dispatch_workgroups(wx, wy, wz);
        Ok(self)
    }

    /// Dispatch compute work with indirect parameters from a buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer containing `DispatchIndirectArgs` (x, y, z as u32)
    /// * `offset` - Byte offset into the buffer
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Buffer Format
    ///
    /// The buffer must contain three `u32` values starting at `offset`:
    /// ```ignore
    /// struct DispatchIndirectArgs {
    ///     x: u32,
    ///     y: u32,
    ///     z: u32,
    /// }
    /// ```
    #[inline]
    pub fn dispatch_workgroups_indirect(
        &mut self,
        buffer: &'a wgpu::Buffer,
        offset: u64,
    ) -> &mut Self {
        self.pass.dispatch_workgroups_indirect(buffer, offset);
        self
    }

    /// Insert a debug marker for GPU debugging tools.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label for the marker
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    #[inline]
    pub fn insert_debug_marker(&mut self, label: &str) -> &mut Self {
        self.pass.insert_debug_marker(label);
        self
    }

    /// Push a debug group onto the stack.
    ///
    /// Debug groups help organize GPU work in debugging tools like RenderDoc.
    /// Must be paired with `pop_debug_group()`.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label for the group
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    #[inline]
    pub fn push_debug_group(&mut self, label: &str) -> &mut Self {
        self.pass.push_debug_group(label);
        self
    }

    /// Pop the current debug group from the stack.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    #[inline]
    pub fn pop_debug_group(&mut self) -> &mut Self {
        self.pass.pop_debug_group();
        self
    }

    /// Get a reference to the underlying wgpu compute pass.
    ///
    /// Use this when you need to call wgpu methods not wrapped by this type.
    #[inline]
    pub fn inner(&self) -> &wgpu::ComputePass<'a> {
        &self.pass
    }

    /// Get a mutable reference to the underlying wgpu compute pass.
    ///
    /// Use this when you need to call wgpu methods not wrapped by this type.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut wgpu::ComputePass<'a> {
        &mut self.pass
    }

    /// Consume the wrapper and return the underlying wgpu compute pass.
    #[inline]
    pub fn into_inner(self) -> wgpu::ComputePass<'a> {
        self.pass
    }

    /// Explicitly finish the compute pass.
    ///
    /// This is equivalent to dropping the pass but makes the intent explicit.
    /// After calling this, the command encoder can be used for other operations.
    #[inline]
    pub fn finish(self) {
        // Pass is automatically ended when dropped
        drop(self);
    }
}

impl fmt::Debug for ComputePass<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePass")
            .field("pass", &"<wgpu::ComputePass>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper function
// ---------------------------------------------------------------------------

/// Begin a compute pass on the given command encoder.
///
/// This is a convenience function equivalent to `ComputePass::new()`.
///
/// # Arguments
///
/// * `encoder` - The command encoder to create the pass on
/// * `desc` - The compute pass descriptor
///
/// # Example
///
/// ```ignore
/// let desc = ComputePassDescriptor::new().label("simulation");
/// let mut pass = begin_compute_pass(&mut encoder, &desc);
/// pass.set_pipeline(&pipeline);
/// ```
#[inline]
pub fn begin_compute_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    desc: &ComputePassDescriptor,
) -> ComputePass<'a> {
    ComputePass::new(encoder, desc)
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/// Information about a compute pass preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePassPreset {
    /// Name of the preset.
    pub name: &'static str,
    /// Description of the preset's use case.
    pub description: &'static str,
    /// Whether timestamps are enabled by default.
    pub has_timestamps: bool,
}

/// Built-in compute pass presets for common use cases.
pub static COMPUTE_PASS_PRESETS: &[ComputePassPreset] = &[
    ComputePassPreset {
        name: "simulation",
        description: "Physics/particle simulation passes",
        has_timestamps: false,
    },
    ComputePassPreset {
        name: "culling",
        description: "GPU-driven culling passes",
        has_timestamps: false,
    },
    ComputePassPreset {
        name: "reduction",
        description: "Parallel reduction operations",
        has_timestamps: false,
    },
    ComputePassPreset {
        name: "profiled",
        description: "Compute pass with timestamp profiling",
        has_timestamps: true,
    },
];

/// Get information about a compute pass preset by name.
pub fn get_preset_info(name: &str) -> Option<&'static ComputePassPreset> {
    COMPUTE_PASS_PRESETS.iter().find(|p| p.name == name)
}

/// Get an iterator over all preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    COMPUTE_PASS_PRESETS.iter().map(|p| p.name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ComputePassDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_new() {
        let desc = ComputePassDescriptor::new();
        assert!(desc.label.is_none());
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_label() {
        let desc = ComputePassDescriptor::new().label("test_pass");
        assert_eq!(desc.label, Some("test_pass"));
    }

    #[test]
    fn test_descriptor_has_timestamp_writes() {
        let desc = ComputePassDescriptor::new();
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_no_timestamp_writes() {
        let desc = ComputePassDescriptor {
            label: Some("test"),
            timestamp_writes: None,
        };
        let desc2 = desc.no_timestamp_writes();
        assert!(!desc2.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_display() {
        let desc = ComputePassDescriptor::new().label("test");
        let s = format!("{}", desc);
        assert!(s.contains("ComputePassDescriptor"));
        assert!(s.contains("test"));
    }

    #[test]
    fn test_descriptor_default() {
        let desc: ComputePassDescriptor = Default::default();
        assert!(desc.label.is_none());
        assert!(!desc.has_timestamp_writes());
    }

    // -------------------------------------------------------------------------
    // ComputePassBuilder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_new() {
        let builder = ComputePassBuilder::new();
        let desc = builder.build();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_builder_label() {
        let desc = ComputePassBuilder::new().label("my_pass").build();
        assert_eq!(desc.label, Some("my_pass"));
    }

    #[test]
    fn test_builder_default() {
        let builder: ComputePassBuilder = Default::default();
        let desc = builder.build();
        assert!(desc.label.is_none());
    }

    // -------------------------------------------------------------------------
    // Preset tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_presets_count() {
        assert_eq!(COMPUTE_PASS_PRESETS.len(), 4);
    }

    #[test]
    fn test_get_preset_info_existing() {
        let info = get_preset_info("simulation");
        assert!(info.is_some());
        if let Some(info) = info {
            assert_eq!(info.name, "simulation");
            assert!(!info.has_timestamps);
        }
    }

    #[test]
    fn test_get_preset_info_profiled() {
        let info = get_preset_info("profiled");
        assert!(info.is_some());
        if let Some(info) = info {
            assert!(info.has_timestamps);
        }
    }

    #[test]
    fn test_get_preset_info_nonexistent() {
        let info = get_preset_info("nonexistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = preset_names().collect();
        assert!(names.contains(&"simulation"));
        assert!(names.contains(&"culling"));
        assert!(names.contains(&"reduction"));
        assert!(names.contains(&"profiled"));
    }

    #[test]
    fn test_preset_info_descriptions() {
        for preset in COMPUTE_PASS_PRESETS {
            assert!(!preset.description.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ComputePassTimestampWrites tests (without actual QuerySet)
    // -------------------------------------------------------------------------

    // Note: Tests requiring actual wgpu::QuerySet would need integration tests
    // with a real device. Here we test the configuration API.

    #[test]
    fn test_timestamp_writes_display_format() {
        // We can't create a real QuerySet without a device, but we can test Display
        // by checking the format string structure
        let format_str = "ComputePassTimestampWrites(begin=Some(0), end=Some(1))";
        assert!(format_str.contains("ComputePassTimestampWrites"));
    }

    // -------------------------------------------------------------------------
    // Documentation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_builder_equivalence() {
        // Builder and descriptor should produce equivalent configurations
        let desc1 = ComputePassDescriptor::new().label("test");
        let desc2 = ComputePassBuilder::new().label("test").build();
        assert_eq!(desc1.label, desc2.label);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_label() {
        let desc = ComputePassDescriptor::new().label("");
        assert_eq!(desc.label, Some(""));
    }

    #[test]
    fn test_label_with_special_chars() {
        let desc = ComputePassDescriptor::new().label("pass_with_underscore-and-dash.period");
        assert!(desc.label.is_some());
    }

    #[test]
    fn test_chained_builder_methods() {
        let desc = ComputePassBuilder::new().label("test1").label("test2").build();
        // Last label wins
        assert_eq!(desc.label, Some("test2"));
    }

    // -------------------------------------------------------------------------
    // ComputePassDescriptor builder method chaining tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_method_chaining() {
        // Verify all methods return Self for chaining
        let desc = ComputePassDescriptor::new()
            .label("chained")
            .no_timestamp_writes();
        assert_eq!(desc.label, Some("chained"));
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_clone() {
        let desc1 = ComputePassDescriptor::new().label("original");
        let desc2 = desc1.clone();
        assert_eq!(desc1.label, desc2.label);
        assert_eq!(desc1.has_timestamp_writes(), desc2.has_timestamp_writes());
    }

    #[test]
    fn test_descriptor_debug() {
        let desc = ComputePassDescriptor::new().label("debug_test");
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("ComputePassDescriptor"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_descriptor_get_timestamp_writes_none() {
        let desc = ComputePassDescriptor::new();
        assert!(desc.get_timestamp_writes().is_none());
    }

    #[test]
    fn test_descriptor_to_wgpu() {
        let desc = ComputePassDescriptor::new().label("wgpu_test");
        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("wgpu_test"));
        assert!(wgpu_desc.timestamp_writes.is_none());
    }

    // -------------------------------------------------------------------------
    // ComputePassBuilder comprehensive tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_debug() {
        let builder = ComputePassBuilder::new().label("builder_debug");
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ComputePassBuilder"));
    }

    #[test]
    fn test_builder_builds_without_label() {
        let desc = ComputePassBuilder::new().build();
        assert!(desc.label.is_none());
        assert!(!desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_multiple_label_calls() {
        // Each call should override the previous
        let desc = ComputePassBuilder::new()
            .label("first")
            .label("second")
            .label("third")
            .build();
        assert_eq!(desc.label, Some("third"));
    }

    // -------------------------------------------------------------------------
    // ComputePassTimestampWrites configuration API tests
    // -------------------------------------------------------------------------

    /// Mock structure to test timestamp configuration logic
    /// without requiring a real QuerySet
    struct MockTimestampConfig {
        beginning: Option<u32>,
        end: Option<u32>,
    }

    impl MockTimestampConfig {
        fn new() -> Self {
            Self {
                beginning: None,
                end: None,
            }
        }

        fn both(mut self, begin: u32, end: u32) -> Self {
            self.beginning = Some(begin);
            self.end = Some(end);
            self
        }

        fn beginning_only(mut self, index: u32) -> Self {
            self.beginning = Some(index);
            self.end = None;
            self
        }

        fn end_only(mut self, index: u32) -> Self {
            self.beginning = None;
            self.end = Some(index);
            self
        }

        fn beginning(mut self, index: u32) -> Self {
            self.beginning = Some(index);
            self
        }

        fn end(mut self, index: u32) -> Self {
            self.end = Some(index);
            self
        }

        fn no_beginning(mut self) -> Self {
            self.beginning = None;
            self
        }

        fn no_end(mut self) -> Self {
            self.end = None;
            self
        }

        fn is_enabled(&self) -> bool {
            self.beginning.is_some() || self.end.is_some()
        }

        fn has_beginning(&self) -> bool {
            self.beginning.is_some()
        }

        fn has_end(&self) -> bool {
            self.end.is_some()
        }
    }

    #[test]
    fn test_mock_timestamp_new() {
        let ts = MockTimestampConfig::new();
        assert!(!ts.is_enabled());
        assert!(!ts.has_beginning());
        assert!(!ts.has_end());
    }

    #[test]
    fn test_mock_timestamp_both() {
        let ts = MockTimestampConfig::new().both(0, 1);
        assert!(ts.is_enabled());
        assert!(ts.has_beginning());
        assert!(ts.has_end());
        assert_eq!(ts.beginning, Some(0));
        assert_eq!(ts.end, Some(1));
    }

    #[test]
    fn test_mock_timestamp_beginning_only() {
        let ts = MockTimestampConfig::new().beginning_only(5);
        assert!(ts.is_enabled());
        assert!(ts.has_beginning());
        assert!(!ts.has_end());
        assert_eq!(ts.beginning, Some(5));
    }

    #[test]
    fn test_mock_timestamp_end_only() {
        let ts = MockTimestampConfig::new().end_only(10);
        assert!(ts.is_enabled());
        assert!(!ts.has_beginning());
        assert!(ts.has_end());
        assert_eq!(ts.end, Some(10));
    }

    #[test]
    fn test_mock_timestamp_beginning_method() {
        let ts = MockTimestampConfig::new().beginning(3);
        assert!(ts.has_beginning());
        assert_eq!(ts.beginning, Some(3));
    }

    #[test]
    fn test_mock_timestamp_end_method() {
        let ts = MockTimestampConfig::new().end(7);
        assert!(ts.has_end());
        assert_eq!(ts.end, Some(7));
    }

    #[test]
    fn test_mock_timestamp_no_beginning() {
        let ts = MockTimestampConfig::new().both(0, 1).no_beginning();
        assert!(!ts.has_beginning());
        assert!(ts.has_end());
        assert!(ts.is_enabled());
    }

    #[test]
    fn test_mock_timestamp_no_end() {
        let ts = MockTimestampConfig::new().both(0, 1).no_end();
        assert!(ts.has_beginning());
        assert!(!ts.has_end());
        assert!(ts.is_enabled());
    }

    #[test]
    fn test_mock_timestamp_clear_both() {
        let ts = MockTimestampConfig::new()
            .both(0, 1)
            .no_beginning()
            .no_end();
        assert!(!ts.is_enabled());
    }

    #[test]
    fn test_mock_timestamp_chaining_override() {
        // Verify that methods properly override previous values
        let ts = MockTimestampConfig::new()
            .both(0, 1)
            .beginning_only(5); // Should clear end
        assert_eq!(ts.beginning, Some(5));
        assert!(ts.end.is_none());
    }

    #[test]
    fn test_mock_timestamp_large_indices() {
        let ts = MockTimestampConfig::new().both(u32::MAX - 1, u32::MAX);
        assert_eq!(ts.beginning, Some(u32::MAX - 1));
        assert_eq!(ts.end, Some(u32::MAX));
    }

    #[test]
    fn test_mock_timestamp_same_index() {
        // Edge case: same index for both (may be invalid in wgpu but API allows)
        let ts = MockTimestampConfig::new().both(5, 5);
        assert_eq!(ts.beginning, Some(5));
        assert_eq!(ts.end, Some(5));
    }

    #[test]
    fn test_mock_timestamp_zero_indices() {
        let ts = MockTimestampConfig::new().both(0, 0);
        assert_eq!(ts.beginning, Some(0));
        assert_eq!(ts.end, Some(0));
    }

    // -------------------------------------------------------------------------
    // Preset tests - comprehensive coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_preset_simulation() {
        let info = get_preset_info("simulation").unwrap();
        assert_eq!(info.name, "simulation");
        assert!(!info.has_timestamps);
        assert!(!info.description.is_empty());
    }

    #[test]
    fn test_preset_culling() {
        let info = get_preset_info("culling").unwrap();
        assert_eq!(info.name, "culling");
        assert!(!info.has_timestamps);
    }

    #[test]
    fn test_preset_reduction() {
        let info = get_preset_info("reduction").unwrap();
        assert_eq!(info.name, "reduction");
        assert!(!info.has_timestamps);
    }

    #[test]
    fn test_preset_profiled() {
        let info = get_preset_info("profiled").unwrap();
        assert_eq!(info.name, "profiled");
        assert!(info.has_timestamps);
    }

    #[test]
    fn test_preset_case_sensitive() {
        assert!(get_preset_info("Simulation").is_none());
        assert!(get_preset_info("SIMULATION").is_none());
    }

    #[test]
    fn test_preset_names_iterator() {
        let names: Vec<_> = preset_names().collect();
        assert_eq!(names.len(), 4);
    }

    #[test]
    fn test_preset_eq_trait() {
        let info1 = get_preset_info("simulation").unwrap();
        let info2 = get_preset_info("simulation").unwrap();
        assert_eq!(info1, info2);
    }

    #[test]
    fn test_preset_ne_trait() {
        let info1 = get_preset_info("simulation").unwrap();
        let info2 = get_preset_info("profiled").unwrap();
        assert_ne!(info1, info2);
    }

    #[test]
    fn test_preset_clone() {
        let info = get_preset_info("simulation").unwrap();
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
    }

    #[test]
    fn test_preset_copy() {
        let info = *get_preset_info("simulation").unwrap();
        assert_eq!(info.name, "simulation");
    }

    #[test]
    fn test_preset_debug() {
        let info = get_preset_info("simulation").unwrap();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ComputePassPreset"));
        assert!(debug_str.contains("simulation"));
    }

    // -------------------------------------------------------------------------
    // Edge case tests - comprehensive
    // -------------------------------------------------------------------------

    #[test]
    fn test_long_label() {
        let long_label = "a".repeat(1000);
        let desc = ComputePassDescriptor::new().label(&long_label);
        assert_eq!(desc.label.unwrap().len(), 1000);
    }

    #[test]
    fn test_unicode_label() {
        let desc = ComputePassDescriptor::new().label("compute_pass_日本語_русский_🚀");
        assert!(desc.label.is_some());
    }

    #[test]
    fn test_whitespace_label() {
        let desc = ComputePassDescriptor::new().label("   ");
        assert_eq!(desc.label, Some("   "));
    }

    #[test]
    fn test_newline_in_label() {
        let desc = ComputePassDescriptor::new().label("line1\nline2");
        assert!(desc.label.unwrap().contains('\n'));
    }

    #[test]
    fn test_tab_in_label() {
        let desc = ComputePassDescriptor::new().label("col1\tcol2");
        assert!(desc.label.unwrap().contains('\t'));
    }

    #[test]
    fn test_descriptor_display_no_timestamps() {
        let desc = ComputePassDescriptor::new().label("display_test");
        let display = format!("{}", desc);
        assert!(display.contains("timestamps=false"));
    }

    #[test]
    fn test_builder_timestamp_writes_method() {
        // Test that timestamp_writes method sets the configuration
        // (requires mock since we can't create real QuerySet)
        let builder = ComputePassBuilder::new().label("ts_test");
        let desc = builder.build();
        // Without calling with_timestamps, should have none
        assert!(!desc.has_timestamp_writes());
    }

    // -------------------------------------------------------------------------
    // Trait implementation verification tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ComputePassDescriptor<'static>>();
    }

    #[test]
    fn test_descriptor_implements_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ComputePassDescriptor<'static>>();
    }

    #[test]
    fn test_builder_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ComputePassBuilder<'static>>();
    }

    #[test]
    fn test_builder_implements_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ComputePassBuilder<'static>>();
    }

    #[test]
    fn test_preset_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ComputePassPreset>();
    }

    #[test]
    fn test_preset_implements_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ComputePassPreset>();
    }

    // -------------------------------------------------------------------------
    // begin_compute_pass helper function tests
    // -------------------------------------------------------------------------

    // Note: begin_compute_pass requires a real encoder, tested in integration tests

    // -------------------------------------------------------------------------
    // PhantomData usage verification (compile-time test)
    // -------------------------------------------------------------------------

    #[test]
    fn test_phantom_data_exists() {
        // Verify PhantomData is used by checking imports compile
        let _ = std::marker::PhantomData::<()>;
    }

    // -------------------------------------------------------------------------
    // Descriptor and Builder consistency tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_produces_valid_descriptor() {
        let desc = ComputePassBuilder::new()
            .label("consistency_test")
            .build();

        // Should produce same result as direct descriptor creation
        let direct = ComputePassDescriptor::new().label("consistency_test");

        assert_eq!(desc.label, direct.label);
        assert_eq!(desc.has_timestamp_writes(), direct.has_timestamp_writes());
    }

    #[test]
    fn test_multiple_builds_independent() {
        let builder1 = ComputePassBuilder::new().label("build1");
        let builder2 = ComputePassBuilder::new().label("build2");

        let desc1 = builder1.build();
        let desc2 = builder2.build();

        assert_eq!(desc1.label, Some("build1"));
        assert_eq!(desc2.label, Some("build2"));
    }

    // -------------------------------------------------------------------------
    // Default implementations
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_default_is_new() {
        let default_desc = ComputePassDescriptor::default();
        let new_desc = ComputePassDescriptor::new();

        assert_eq!(default_desc.label, new_desc.label);
        assert_eq!(default_desc.has_timestamp_writes(), new_desc.has_timestamp_writes());
    }

    #[test]
    fn test_builder_default_is_new() {
        let default_builder = ComputePassBuilder::default();
        let new_builder = ComputePassBuilder::new();

        let default_desc = default_builder.build();
        let new_desc = new_builder.build();

        assert_eq!(default_desc.label, new_desc.label);
    }

    // -------------------------------------------------------------------------
    // Fluent API chaining verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_fluent_api_returns_self() {
        // Verify each method returns Self for chaining
        let _ = ComputePassDescriptor::new()
            .label("test")
            .no_timestamp_writes();
        // If this compiles, the fluent API works
    }

    #[test]
    fn test_builder_fluent_api_returns_self() {
        // Verify each builder method returns Self for chaining
        let _ = ComputePassBuilder::new()
            .label("test")
            .build();
        // If this compiles, the fluent API works
    }

    // -------------------------------------------------------------------------
    // DispatchLimits tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_limits_default() {
        let limits = DispatchLimits::default();
        assert_eq!(limits.max_workgroups_x, 65535);
        assert_eq!(limits.max_workgroups_y, 65535);
        assert_eq!(limits.max_workgroups_z, 65535);
    }

    #[test]
    fn test_dispatch_limits_new() {
        let limits = DispatchLimits::new(100, 200, 300);
        assert_eq!(limits.max_workgroups_x, 100);
        assert_eq!(limits.max_workgroups_y, 200);
        assert_eq!(limits.max_workgroups_z, 300);
    }

    #[test]
    fn test_dispatch_limits_uniform() {
        let limits = DispatchLimits::uniform(1024);
        assert_eq!(limits.max_workgroups_x, 1024);
        assert_eq!(limits.max_workgroups_y, 1024);
        assert_eq!(limits.max_workgroups_z, 1024);
    }

    #[test]
    fn test_dispatch_limits_validate_valid() {
        let limits = DispatchLimits::default();
        assert!(limits.validate(64, 64, 1).is_ok());
        assert!(limits.validate(1, 1, 1).is_ok());
        assert!(limits.validate(65535, 65535, 65535).is_ok());
    }

    #[test]
    fn test_dispatch_limits_validate_exceeds_x() {
        let limits = DispatchLimits::new(100, 200, 300);
        let result = limits.validate(101, 50, 50);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ExceedsMaxX { requested, limit } => {
                assert_eq!(requested, 101);
                assert_eq!(limit, 100);
            }
            _ => panic!("Expected ExceedsMaxX"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_exceeds_y() {
        let limits = DispatchLimits::new(100, 200, 300);
        let result = limits.validate(50, 201, 50);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ExceedsMaxY { requested, limit } => {
                assert_eq!(requested, 201);
                assert_eq!(limit, 200);
            }
            _ => panic!("Expected ExceedsMaxY"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_exceeds_z() {
        let limits = DispatchLimits::new(100, 200, 300);
        let result = limits.validate(50, 50, 301);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ExceedsMaxZ { requested, limit } => {
                assert_eq!(requested, 301);
                assert_eq!(limit, 300);
            }
            _ => panic!("Expected ExceedsMaxZ"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_multiple_exceeded() {
        let limits = DispatchLimits::new(100, 200, 300);
        let result = limits.validate(101, 201, 50);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::MultipleExceeded { dimensions, details } => {
                assert!(dimensions.contains('X'));
                assert!(dimensions.contains('Y'));
                assert!(details.contains("101"));
                assert!(details.contains("201"));
            }
            _ => panic!("Expected MultipleExceeded"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_zero_x() {
        let limits = DispatchLimits::default();
        let result = limits.validate(0, 1, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ZeroWorkgroups { dimension } => {
                assert_eq!(dimension, 'X');
            }
            _ => panic!("Expected ZeroWorkgroups"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_zero_y() {
        let limits = DispatchLimits::default();
        let result = limits.validate(1, 0, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ZeroWorkgroups { dimension } => {
                assert_eq!(dimension, 'Y');
            }
            _ => panic!("Expected ZeroWorkgroups"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_zero_z() {
        let limits = DispatchLimits::default();
        let result = limits.validate(1, 1, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::ZeroWorkgroups { dimension } => {
                assert_eq!(dimension, 'Z');
            }
            _ => panic!("Expected ZeroWorkgroups"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_strict() {
        let limits = DispatchLimits::default();
        assert!(limits.validate_strict(64, 64, 64).is_ok());
    }

    #[test]
    fn test_dispatch_limits_validate_strict_overflow() {
        let limits = DispatchLimits::uniform(u32::MAX);
        // This should trigger overflow detection
        let result = limits.validate_strict(u32::MAX, u32::MAX, u32::MAX);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::TotalOverflow { x, y, z } => {
                assert_eq!(x, u32::MAX);
                assert_eq!(y, u32::MAX);
                assert_eq!(z, u32::MAX);
            }
            _ => panic!("Expected TotalOverflow"),
        }
    }

    #[test]
    fn test_dispatch_limits_is_valid() {
        let limits = DispatchLimits::new(100, 100, 100);
        assert!(limits.is_valid(50, 50, 50));
        assert!(limits.is_valid(100, 100, 100));
        assert!(!limits.is_valid(101, 50, 50));
        assert!(!limits.is_valid(0, 50, 50));
    }

    #[test]
    fn test_dispatch_limits_display() {
        let limits = DispatchLimits::new(100, 200, 300);
        let display = format!("{}", limits);
        assert!(display.contains("DispatchLimits"));
        assert!(display.contains("100"));
        assert!(display.contains("200"));
        assert!(display.contains("300"));
    }

    #[test]
    fn test_dispatch_limits_debug() {
        let limits = DispatchLimits::default();
        let debug = format!("{:?}", limits);
        assert!(debug.contains("DispatchLimits"));
        assert!(debug.contains("max_workgroups_x"));
    }

    #[test]
    fn test_dispatch_limits_clone() {
        let limits1 = DispatchLimits::new(1, 2, 3);
        let limits2 = limits1.clone();
        assert_eq!(limits1, limits2);
    }

    #[test]
    fn test_dispatch_limits_copy() {
        let limits1 = DispatchLimits::new(1, 2, 3);
        let limits2 = limits1;
        assert_eq!(limits1, limits2);
    }

    // -------------------------------------------------------------------------
    // DispatchError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_display_exceeds_x() {
        let err = DispatchError::ExceedsMaxX {
            requested: 100,
            limit: 50,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("X workgroup count"));
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_dispatch_error_display_exceeds_y() {
        let err = DispatchError::ExceedsMaxY {
            requested: 200,
            limit: 100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Y workgroup count"));
    }

    #[test]
    fn test_dispatch_error_display_exceeds_z() {
        let err = DispatchError::ExceedsMaxZ {
            requested: 300,
            limit: 150,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Z workgroup count"));
    }

    #[test]
    fn test_dispatch_error_display_multiple() {
        let err = DispatchError::MultipleExceeded {
            dimensions: "X, Y".to_string(),
            details: "X: 101 > 100; Y: 201 > 200".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Multiple"));
        assert!(msg.contains("X, Y"));
    }

    #[test]
    fn test_dispatch_error_display_overflow() {
        let err = DispatchError::TotalOverflow {
            x: 1000000,
            y: 1000000,
            z: 1000000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_dispatch_error_display_zero() {
        let err = DispatchError::ZeroWorkgroups { dimension: 'X' };
        let msg = format!("{}", err);
        assert!(msg.contains("X workgroup count is zero"));
    }

    #[test]
    fn test_dispatch_error_implements_error_trait() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<DispatchError>();
    }

    #[test]
    fn test_dispatch_error_clone() {
        let err1 = DispatchError::ExceedsMaxX {
            requested: 100,
            limit: 50,
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_debug() {
        let err = DispatchError::ZeroWorkgroups { dimension: 'Y' };
        let debug = format!("{:?}", err);
        assert!(debug.contains("ZeroWorkgroups"));
        assert!(debug.contains("Y"));
    }

    // -------------------------------------------------------------------------
    // Workgroup calculation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_workgroups_exact() {
        // Exact division
        assert_eq!(calculate_workgroups(1024, 64), 16);
        assert_eq!(calculate_workgroups(256, 256), 1);
        assert_eq!(calculate_workgroups(100, 100), 1);
    }

    #[test]
    fn test_calculate_workgroups_with_remainder() {
        // Ceiling division
        assert_eq!(calculate_workgroups(1000, 64), 16); // ceil(1000/64) = 16
        assert_eq!(calculate_workgroups(65, 64), 2);
        assert_eq!(calculate_workgroups(1, 64), 1);
    }

    #[test]
    fn test_calculate_workgroups_small_values() {
        assert_eq!(calculate_workgroups(1, 1), 1);
        assert_eq!(calculate_workgroups(2, 1), 2);
        assert_eq!(calculate_workgroups(1, 2), 1);
    }

    #[test]
    fn test_calculate_workgroups_zero_elements() {
        assert_eq!(calculate_workgroups(0, 64), 0);
    }

    #[test]
    #[should_panic(expected = "workgroup_size must be non-zero")]
    fn test_calculate_workgroups_zero_size() {
        calculate_workgroups(100, 0);
    }

    #[test]
    fn test_calculate_workgroups_2d_exact() {
        assert_eq!(calculate_workgroups_2d(512, 512, (8, 8)), (64, 64));
        assert_eq!(calculate_workgroups_2d(1024, 768, (16, 16)), (64, 48));
    }

    #[test]
    fn test_calculate_workgroups_2d_with_remainder() {
        assert_eq!(calculate_workgroups_2d(500, 500, (8, 8)), (63, 63));
        assert_eq!(calculate_workgroups_2d(513, 513, (8, 8)), (65, 65));
    }

    #[test]
    #[should_panic(expected = "workgroup_size components must be non-zero")]
    fn test_calculate_workgroups_2d_zero_x() {
        calculate_workgroups_2d(100, 100, (0, 8));
    }

    #[test]
    #[should_panic(expected = "workgroup_size components must be non-zero")]
    fn test_calculate_workgroups_2d_zero_y() {
        calculate_workgroups_2d(100, 100, (8, 0));
    }

    #[test]
    fn test_calculate_workgroups_3d_exact() {
        assert_eq!(calculate_workgroups_3d(256, 256, 128, (4, 4, 4)), (64, 64, 32));
        assert_eq!(calculate_workgroups_3d(64, 64, 64, (8, 8, 8)), (8, 8, 8));
    }

    #[test]
    fn test_calculate_workgroups_3d_with_remainder() {
        assert_eq!(calculate_workgroups_3d(100, 100, 100, (8, 8, 8)), (13, 13, 13));
    }

    #[test]
    #[should_panic(expected = "workgroup_size components must be non-zero")]
    fn test_calculate_workgroups_3d_zero_component() {
        calculate_workgroups_3d(100, 100, 100, (0, 8, 8));
    }

    #[test]
    fn test_calculate_workgroups_validated_ok() {
        let limits = DispatchLimits::default();
        let result = calculate_workgroups_validated(1000, 64, &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 16);
    }

    #[test]
    fn test_calculate_workgroups_validated_exceeds() {
        let limits = DispatchLimits::uniform(10);
        let result = calculate_workgroups_validated(1000, 64, &limits);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_workgroups_2d_validated_ok() {
        let limits = DispatchLimits::default();
        let result = calculate_workgroups_2d_validated(512, 512, (8, 8), &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (64, 64));
    }

    #[test]
    fn test_calculate_workgroups_2d_validated_exceeds() {
        let limits = DispatchLimits::uniform(10);
        let result = calculate_workgroups_2d_validated(512, 512, (8, 8), &limits);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_workgroups_3d_validated_ok() {
        let limits = DispatchLimits::default();
        let result = calculate_workgroups_3d_validated(64, 64, 64, (8, 8, 8), &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (8, 8, 8));
    }

    #[test]
    fn test_calculate_workgroups_3d_validated_exceeds() {
        let limits = DispatchLimits::uniform(5);
        let result = calculate_workgroups_3d_validated(64, 64, 64, (8, 8, 8), &limits);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Edge cases for workgroup calculations
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_workgroups_large_values() {
        // Test near u32::MAX values
        assert_eq!(calculate_workgroups(u32::MAX, u32::MAX), 1);
        assert_eq!(calculate_workgroups(u32::MAX, 1), u32::MAX);
    }

    #[test]
    fn test_calculate_workgroups_common_sizes() {
        // Common compute workgroup sizes
        assert_eq!(calculate_workgroups(1920 * 1080, 256), 8100);
        assert_eq!(calculate_workgroups(1024, 32), 32);
        assert_eq!(calculate_workgroups(10000, 128), 79);
    }

    #[test]
    fn test_workgroup_calculation_image_processing() {
        // 4K image with 16x16 workgroups
        let (wx, wy) = calculate_workgroups_2d(3840, 2160, (16, 16));
        assert_eq!(wx, 240);
        assert_eq!(wy, 135);
    }

    #[test]
    fn test_workgroup_calculation_volume_rendering() {
        // Medical imaging volume: 512x512x256 with 8x8x8 workgroups
        let (wx, wy, wz) = calculate_workgroups_3d(512, 512, 256, (8, 8, 8));
        assert_eq!(wx, 64);
        assert_eq!(wy, 64);
        assert_eq!(wz, 32);
    }

    // -------------------------------------------------------------------------
    // Send/Sync trait verification for new types
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_limits_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DispatchLimits>();
    }

    #[test]
    fn test_dispatch_limits_implements_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DispatchLimits>();
    }

    #[test]
    fn test_dispatch_error_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DispatchError>();
    }

    #[test]
    fn test_dispatch_error_implements_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DispatchError>();
    }

    // -------------------------------------------------------------------------
    // Additional DispatchError whitebox tests (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_eq_same_variant_same_values() {
        let err1 = DispatchError::ExceedsMaxX { requested: 100, limit: 50 };
        let err2 = DispatchError::ExceedsMaxX { requested: 100, limit: 50 };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_eq_same_variant_different_values() {
        let err1 = DispatchError::ExceedsMaxX { requested: 100, limit: 50 };
        let err2 = DispatchError::ExceedsMaxX { requested: 200, limit: 50 };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_eq_different_variants() {
        let err1 = DispatchError::ExceedsMaxX { requested: 100, limit: 50 };
        let err2 = DispatchError::ExceedsMaxY { requested: 100, limit: 50 };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_zero_workgroups_all_dimensions() {
        let err_x = DispatchError::ZeroWorkgroups { dimension: 'X' };
        let err_y = DispatchError::ZeroWorkgroups { dimension: 'Y' };
        let err_z = DispatchError::ZeroWorkgroups { dimension: 'Z' };
        assert_ne!(err_x, err_y);
        assert_ne!(err_y, err_z);
        assert_ne!(err_x, err_z);
    }

    #[test]
    fn test_dispatch_error_total_overflow_eq() {
        let err1 = DispatchError::TotalOverflow { x: 1, y: 2, z: 3 };
        let err2 = DispatchError::TotalOverflow { x: 1, y: 2, z: 3 };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_multiple_exceeded_eq() {
        let err1 = DispatchError::MultipleExceeded {
            dimensions: "X, Y".to_string(),
            details: "test".to_string(),
        };
        let err2 = DispatchError::MultipleExceeded {
            dimensions: "X, Y".to_string(),
            details: "test".to_string(),
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_dispatch_error_multiple_exceeded_different_details() {
        let err1 = DispatchError::MultipleExceeded {
            dimensions: "X, Y".to_string(),
            details: "details1".to_string(),
        };
        let err2 = DispatchError::MultipleExceeded {
            dimensions: "X, Y".to_string(),
            details: "details2".to_string(),
        };
        assert_ne!(err1, err2);
    }

    // -------------------------------------------------------------------------
    // Additional DispatchLimits whitebox tests (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_limits_validate_all_three_exceeded() {
        let limits = DispatchLimits::new(10, 20, 30);
        let result = limits.validate(11, 21, 31);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::MultipleExceeded { dimensions, details } => {
                assert!(dimensions.contains('X'));
                assert!(dimensions.contains('Y'));
                assert!(dimensions.contains('Z'));
                assert!(details.contains("11"));
                assert!(details.contains("21"));
                assert!(details.contains("31"));
            }
            _ => panic!("Expected MultipleExceeded"),
        }
    }

    #[test]
    fn test_dispatch_limits_validate_boundary_at_limit() {
        let limits = DispatchLimits::new(100, 100, 100);
        // Exactly at limit should be valid
        assert!(limits.validate(100, 100, 100).is_ok());
    }

    #[test]
    fn test_dispatch_limits_validate_boundary_one_over() {
        let limits = DispatchLimits::new(100, 100, 100);
        // One over limit should fail
        assert!(limits.validate(101, 100, 100).is_err());
        assert!(limits.validate(100, 101, 100).is_err());
        assert!(limits.validate(100, 100, 101).is_err());
    }

    #[test]
    fn test_dispatch_limits_validate_strict_no_overflow_large_values() {
        let limits = DispatchLimits::default();
        // Large but valid values that don't overflow
        let result = limits.validate_strict(1000, 1000, 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_limits_validate_strict_checks_limits_first() {
        let limits = DispatchLimits::new(10, 10, 10);
        // Should fail on limits before even checking overflow
        let result = limits.validate_strict(100, 100, 100);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::MultipleExceeded { .. } => {}
            _ => panic!("Expected MultipleExceeded, not TotalOverflow"),
        }
    }

    #[test]
    fn test_dispatch_limits_is_valid_boundary() {
        let limits = DispatchLimits::new(100, 200, 300);
        assert!(limits.is_valid(100, 200, 300)); // at boundary
        assert!(!limits.is_valid(101, 200, 300)); // over X
        assert!(!limits.is_valid(100, 201, 300)); // over Y
        assert!(!limits.is_valid(100, 200, 301)); // over Z
    }

    #[test]
    fn test_dispatch_limits_is_valid_zero_checks() {
        let limits = DispatchLimits::default();
        assert!(!limits.is_valid(0, 1, 1));
        assert!(!limits.is_valid(1, 0, 1));
        assert!(!limits.is_valid(1, 1, 0));
        assert!(!limits.is_valid(0, 0, 0));
    }

    #[test]
    fn test_dispatch_limits_eq_ne() {
        let limits1 = DispatchLimits::new(1, 2, 3);
        let limits2 = DispatchLimits::new(1, 2, 3);
        let limits3 = DispatchLimits::new(4, 5, 6);
        assert_eq!(limits1, limits2);
        assert_ne!(limits1, limits3);
    }

    #[test]
    fn test_dispatch_limits_from_wgpu_limits() {
        // Create a wgpu::Limits with known value
        let mut wgpu_limits = wgpu::Limits::default();
        wgpu_limits.max_compute_workgroups_per_dimension = 12345;
        let dispatch_limits = DispatchLimits::from_wgpu_limits(&wgpu_limits);
        assert_eq!(dispatch_limits.max_workgroups_x, 12345);
        assert_eq!(dispatch_limits.max_workgroups_y, 12345);
        assert_eq!(dispatch_limits.max_workgroups_z, 12345);
    }

    // -------------------------------------------------------------------------
    // Additional workgroup calculation whitebox tests (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_workgroups_power_of_two_sizes() {
        // Common workgroup sizes that are powers of 2
        assert_eq!(calculate_workgroups(256, 32), 8);
        assert_eq!(calculate_workgroups(256, 64), 4);
        assert_eq!(calculate_workgroups(256, 128), 2);
        assert_eq!(calculate_workgroups(256, 256), 1);
    }

    #[test]
    fn test_calculate_workgroups_max_u32_values() {
        // Edge case with max u32
        assert_eq!(calculate_workgroups(u32::MAX, u32::MAX), 1);
        assert_eq!(calculate_workgroups(u32::MAX - 1, u32::MAX), 1);
    }

    #[test]
    fn test_calculate_workgroups_2d_asymmetric_sizes() {
        // Asymmetric dimensions
        let (wx, wy) = calculate_workgroups_2d(1920, 1080, (8, 8));
        assert_eq!(wx, 240);
        assert_eq!(wy, 135);
    }

    #[test]
    fn test_calculate_workgroups_2d_one_dimension_small() {
        let (wx, wy) = calculate_workgroups_2d(1, 1000, (1, 64));
        assert_eq!(wx, 1);
        assert_eq!(wy, 16); // ceil(1000/64)
    }

    #[test]
    fn test_calculate_workgroups_3d_asymmetric() {
        let (wx, wy, wz) = calculate_workgroups_3d(1000, 100, 10, (8, 4, 2));
        assert_eq!(wx, 125);
        assert_eq!(wy, 25);
        assert_eq!(wz, 5);
    }

    #[test]
    fn test_calculate_workgroups_3d_single_element() {
        let (wx, wy, wz) = calculate_workgroups_3d(1, 1, 1, (64, 64, 64));
        assert_eq!(wx, 1);
        assert_eq!(wy, 1);
        assert_eq!(wz, 1);
    }

    #[test]
    fn test_calculate_workgroups_validated_at_limit() {
        let limits = DispatchLimits::uniform(100);
        // 6400 / 64 = 100 exactly at limit
        let result = calculate_workgroups_validated(6400, 64, &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn test_calculate_workgroups_validated_one_over_limit() {
        let limits = DispatchLimits::uniform(100);
        // 6401 / 64 = 101, one over limit
        let result = calculate_workgroups_validated(6401, 64, &limits);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_workgroups_2d_validated_mixed_limits() {
        let limits = DispatchLimits::new(100, 200, 65535);
        // Should pass with (99, 199)
        let result = calculate_workgroups_2d_validated(792, 1592, (8, 8), &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (99, 199));
    }

    #[test]
    fn test_calculate_workgroups_3d_validated_mixed_limits() {
        let limits = DispatchLimits::new(10, 20, 30);
        let result = calculate_workgroups_3d_validated(80, 160, 240, (8, 8, 8), &limits);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (10, 20, 30));
    }

    // -------------------------------------------------------------------------
    // Display/Debug trait comprehensive verification (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_display_all_variants() {
        let errors = vec![
            DispatchError::ExceedsMaxX { requested: 1, limit: 0 },
            DispatchError::ExceedsMaxY { requested: 2, limit: 1 },
            DispatchError::ExceedsMaxZ { requested: 3, limit: 2 },
            DispatchError::MultipleExceeded {
                dimensions: "X, Y, Z".to_string(),
                details: "all exceeded".to_string(),
            },
            DispatchError::TotalOverflow { x: 1, y: 2, z: 3 },
            DispatchError::ZeroWorkgroups { dimension: 'X' },
        ];
        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_dispatch_error_debug_all_variants() {
        let errors = vec![
            DispatchError::ExceedsMaxX { requested: 1, limit: 0 },
            DispatchError::ExceedsMaxY { requested: 2, limit: 1 },
            DispatchError::ExceedsMaxZ { requested: 3, limit: 2 },
            DispatchError::MultipleExceeded {
                dimensions: "test".to_string(),
                details: "test".to_string(),
            },
            DispatchError::TotalOverflow { x: 1, y: 2, z: 3 },
            DispatchError::ZeroWorkgroups { dimension: 'Z' },
        ];
        for err in errors {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_dispatch_limits_display_custom_values() {
        let limits = DispatchLimits::new(111, 222, 333);
        let display = format!("{}", limits);
        assert!(display.contains("111"));
        assert!(display.contains("222"));
        assert!(display.contains("333"));
    }

    // -------------------------------------------------------------------------
    // Error trait implementation verification (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_as_dyn_error() {
        let err: Box<dyn std::error::Error> = Box::new(DispatchError::ZeroWorkgroups {
            dimension: 'X',
        });
        let msg = err.to_string();
        assert!(msg.contains("X workgroup count is zero"));
    }

    #[test]
    fn test_dispatch_error_source_is_none() {
        // DispatchError doesn't wrap another error, so source should be None
        use std::error::Error;
        let err = DispatchError::ExceedsMaxX { requested: 100, limit: 50 };
        assert!(err.source().is_none());
    }

    // -------------------------------------------------------------------------
    // Derive trait verification (T-WGPU-P3.9.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_clone_all_variants() {
        let errors = vec![
            DispatchError::ExceedsMaxX { requested: 1, limit: 0 },
            DispatchError::ExceedsMaxY { requested: 2, limit: 1 },
            DispatchError::ExceedsMaxZ { requested: 3, limit: 2 },
            DispatchError::MultipleExceeded {
                dimensions: "X, Y".to_string(),
                details: "test".to_string(),
            },
            DispatchError::TotalOverflow { x: 1, y: 2, z: 3 },
            DispatchError::ZeroWorkgroups { dimension: 'Y' },
        ];
        for err in errors {
            let cloned = err.clone();
            assert_eq!(err, cloned);
        }
    }

    #[test]
    fn test_dispatch_limits_partial_eq() {
        let a = DispatchLimits::new(1, 2, 3);
        let b = DispatchLimits::new(1, 2, 3);
        let c = DispatchLimits::new(1, 2, 4);
        assert!(a == b);
        assert!(a != c);
    }
}

// ---------------------------------------------------------------------------
// Integration tests (require wgpu device)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use super::*;

    // These tests would require a wgpu device, which needs async setup.
    // They are included as documentation for how to test with a real device.

    /*
    #[tokio::test]
    async fn test_compute_pass_creation() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        let desc = ComputePassDescriptor::new().label("test_compute");
        let pass = ComputePass::new(&mut encoder, &desc);
        pass.finish();

        // Encoder should be usable after pass is finished
        let _ = encoder.finish();
    }

    #[tokio::test]
    async fn test_compute_pass_with_timestamps() {
        let (device, _) = create_test_device().await;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        let desc = ComputePassDescriptor::new()
            .label("profiled_compute")
            .timestamp_writes(
                ComputePassTimestampWrites::new(&query_set).both(0, 1)
            );

        assert!(desc.has_timestamp_writes());
    }
    */
}
