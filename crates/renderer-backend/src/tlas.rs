//! Top-Level Acceleration Structure (TLAS) for ray tracing
//!
//! This module provides TLAS construction and management for GPU ray tracing.
//! TLAS contains references to BLAS instances positioned in world space.
//!
//! # Architecture
//!
//! - `TlasConfig`: Configuration flags for TLAS construction
//! - `TlasInstance`: A BLAS instance with transform, mask, and flags
//! - `TlasBuilder`: Fluent API for building TLAS from instances
//! - `Tlas`: The built acceleration structure with bounds and memory tracking
//!
//! # Features
//!
//! - Build from arrays of TLAS instances
//! - Instance transforms (3x4 row-major affine matrices)
//! - Per-instance masks and flags for ray filtering
//! - Tight bounding box computation from instance bounds
//! - Memory usage tracking
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::tlas::{TlasBuilder, TlasConfig, TlasInstance};
//!
//! let config = TlasConfig::default()
//!     .with_fast_build()
//!     .with_max_instances(1000);
//!
//! let instance = TlasInstance::new()
//!     .with_transform(identity_transform())
//!     .with_blas_address(blas_addr)
//!     .with_mask(0xFF);
//!
//! let tlas = TlasBuilder::new()
//!     .config(config)
//!     .instances(&[instance])
//!     .build()?;
//!
//! println!("TLAS: {} instances, {} bytes", tlas.instance_count, tlas.memory_size);
//! ```

use crate::blas::BoundingBox;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum instances for TLAS.
const DEFAULT_MAX_INSTANCES: u32 = 65536;

/// Scratch buffer multiplier for TLAS build.
const SCRATCH_BUFFER_MULTIPLIER: f32 = 1.5;

/// Minimum memory size for TLAS (header overhead).
const MIN_TLAS_MEMORY: usize = 256;

/// Memory overhead per instance in bytes (estimated).
const BYTES_PER_INSTANCE: usize = 128;

/// Size of TlasInstance in bytes (must be 64 bytes to match wgpu layout).
const TLAS_INSTANCE_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Build Flags
// ---------------------------------------------------------------------------

/// Prefer fast TLAS build over fast ray tracing.
pub const PREFER_FAST_BUILD: u32 = 1 << 0;

/// Prefer fast ray tracing over fast build.
pub const PREFER_FAST_TRACE: u32 = 1 << 1;

/// Allow in-place updates of the TLAS.
pub const ALLOW_UPDATE: u32 = 1 << 2;

// ---------------------------------------------------------------------------
// Instance Flags
// ---------------------------------------------------------------------------

/// Instance is opaque (no any-hit shader invocation).
pub const INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE: u8 = 1 << 0;

/// Disable face culling for this instance.
pub const INSTANCE_FLAG_TRIANGLE_FLIP_FACING: u8 = 1 << 1;

/// Force opaque (skip any-hit).
pub const INSTANCE_FLAG_FORCE_OPAQUE: u8 = 1 << 2;

/// Force non-opaque (always run any-hit).
pub const INSTANCE_FLAG_FORCE_NO_OPAQUE: u8 = 1 << 3;

// ---------------------------------------------------------------------------
// TlasConfig
// ---------------------------------------------------------------------------

/// Configuration for TLAS construction.
///
/// Controls build-time flags that affect memory usage and update capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlasConfig {
    /// Build flags (PREFER_FAST_BUILD, PREFER_FAST_TRACE, ALLOW_UPDATE).
    pub build_flags: u32,

    /// Maximum number of instances this TLAS can contain.
    /// Pre-allocates memory for this many instances.
    pub max_instances: u32,
}

impl TlasConfig {
    /// Create a new config with default settings.
    pub const fn new() -> Self {
        Self {
            build_flags: 0,
            max_instances: DEFAULT_MAX_INSTANCES,
        }
    }

    /// Enable fast build preference.
    pub const fn with_fast_build(mut self) -> Self {
        self.build_flags |= PREFER_FAST_BUILD;
        self.build_flags &= !PREFER_FAST_TRACE;
        self
    }

    /// Enable fast trace preference.
    pub const fn with_fast_trace(mut self) -> Self {
        self.build_flags |= PREFER_FAST_TRACE;
        self.build_flags &= !PREFER_FAST_BUILD;
        self
    }

    /// Enable in-place updates.
    pub const fn with_allow_update(mut self) -> Self {
        self.build_flags |= ALLOW_UPDATE;
        self
    }

    /// Set maximum instances.
    pub const fn with_max_instances(mut self, max: u32) -> Self {
        self.max_instances = max;
        self
    }

    /// Check if fast build is preferred.
    pub const fn prefers_fast_build(&self) -> bool {
        (self.build_flags & PREFER_FAST_BUILD) != 0
    }

    /// Check if fast trace is preferred.
    pub const fn prefers_fast_trace(&self) -> bool {
        (self.build_flags & PREFER_FAST_TRACE) != 0
    }

    /// Check if updates are allowed.
    pub const fn allows_update(&self) -> bool {
        (self.build_flags & ALLOW_UPDATE) != 0
    }

    /// Configuration optimized for static scenes.
    /// Prefers fast trace, no updates allowed.
    pub const fn for_static_scene() -> Self {
        Self {
            build_flags: PREFER_FAST_TRACE,
            max_instances: DEFAULT_MAX_INSTANCES,
        }
    }

    /// Configuration optimized for dynamic scenes.
    /// Prefers fast build, allows updates.
    pub const fn for_dynamic_scene() -> Self {
        Self {
            build_flags: PREFER_FAST_BUILD | ALLOW_UPDATE,
            max_instances: DEFAULT_MAX_INSTANCES,
        }
    }
}

impl Default for TlasConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TlasInstance
// ---------------------------------------------------------------------------

/// A BLAS instance in the TLAS.
///
/// This structure matches the wgpu RayTracingInstance layout (64 bytes).
/// The layout is compatible with VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INSTANCE_KHR.
///
/// Memory layout (64 bytes total):
/// - Bytes 0-47:  3x4 row-major affine transform
/// - Bytes 48-51: instance_custom_index (24 bits) + hit_group_offset (8 bits)
/// - Byte 52:     mask
/// - Byte 53:     flags
/// - Bytes 54-55: padding (2 bytes)
/// - Bytes 56-63: blas_address (u64)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlasInstance {
    /// 3x4 row-major affine transformation matrix.
    /// Each row is [x, y, z, translation].
    /// Row 0: X axis and X translation
    /// Row 1: Y axis and Y translation
    /// Row 2: Z axis and Z translation
    pub transform: [[f32; 4]; 3],

    /// Custom index (24 bits) packed with hit group offset (8 bits).
    /// Lower 24 bits: instance custom index (gl_InstanceCustomIndexEXT)
    /// Upper 8 bits: hit group offset (sbt_record_offset)
    pub instance_custom_index_and_hit_group: u32,

    /// Instance visibility mask (8 bits).
    /// Ray mask ANDed with this for intersection tests.
    pub mask: u8,

    /// Instance flags (8 bits).
    /// See INSTANCE_FLAG_* constants.
    pub flags: u8,

    /// Padding to align blas_address to 8 bytes.
    _padding: [u8; 2],

    /// Device address of the BLAS.
    pub blas_address: u64,
}

impl TlasInstance {
    /// Create a new instance with identity transform.
    pub const fn new() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0], // X axis
                [0.0, 1.0, 0.0, 0.0], // Y axis
                [0.0, 0.0, 1.0, 0.0], // Z axis
            ],
            instance_custom_index_and_hit_group: 0,
            mask: 0xFF,
            flags: 0,
            _padding: [0; 2],
            blas_address: 0,
        }
    }

    /// Set the affine transform matrix.
    pub const fn with_transform(mut self, transform: [[f32; 4]; 3]) -> Self {
        self.transform = transform;
        self
    }

    /// Set the custom index (0-16777215, 24 bits).
    pub fn with_custom_index(mut self, index: u32) -> Self {
        debug_assert!(index <= 0x00FFFFFF, "Custom index exceeds 24 bits");
        // Clear lower 24 bits, preserve upper 8 bits (hit group)
        self.instance_custom_index_and_hit_group =
            (self.instance_custom_index_and_hit_group & 0xFF000000) | (index & 0x00FFFFFF);
        self
    }

    /// Set the hit group offset (0-255, 8 bits).
    pub fn with_hit_group_offset(mut self, offset: u8) -> Self {
        // Clear upper 8 bits, preserve lower 24 bits (custom index)
        self.instance_custom_index_and_hit_group =
            (self.instance_custom_index_and_hit_group & 0x00FFFFFF) | ((offset as u32) << 24);
        self
    }

    /// Set the visibility mask.
    pub const fn with_mask(mut self, mask: u8) -> Self {
        self.mask = mask;
        self
    }

    /// Set the instance flags.
    pub const fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    /// Set the BLAS device address.
    pub const fn with_blas_address(mut self, address: u64) -> Self {
        self.blas_address = address;
        self
    }

    /// Get the custom index (lower 24 bits).
    pub const fn custom_index(&self) -> u32 {
        self.instance_custom_index_and_hit_group & 0x00FFFFFF
    }

    /// Get the hit group offset (upper 8 bits).
    pub const fn hit_group_offset(&self) -> u8 {
        ((self.instance_custom_index_and_hit_group >> 24) & 0xFF) as u8
    }

    /// Create an identity transform matrix.
    pub const fn identity_transform() -> [[f32; 4]; 3] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    /// Create a translation transform matrix.
    pub const fn translation_transform(x: f32, y: f32, z: f32) -> [[f32; 4]; 3] {
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
        ]
    }

    /// Create a scale transform matrix.
    pub const fn scale_transform(sx: f32, sy: f32, sz: f32) -> [[f32; 4]; 3] {
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
        ]
    }

    /// Transform a point by this instance's transform.
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        [
            self.transform[0][0] * point[0]
                + self.transform[0][1] * point[1]
                + self.transform[0][2] * point[2]
                + self.transform[0][3],
            self.transform[1][0] * point[0]
                + self.transform[1][1] * point[1]
                + self.transform[1][2] * point[2]
                + self.transform[1][3],
            self.transform[2][0] * point[0]
                + self.transform[2][1] * point[1]
                + self.transform[2][2] * point[2]
                + self.transform[2][3],
        ]
    }

    /// Transform a bounding box by this instance's transform.
    /// Returns a new axis-aligned bounding box that contains the transformed box.
    pub fn transform_bounds(&self, bounds: &BoundingBox) -> BoundingBox {
        if !bounds.is_valid() {
            return BoundingBox::empty();
        }

        // Get all 8 corners of the bounding box
        let corners = [
            [bounds.min[0], bounds.min[1], bounds.min[2]],
            [bounds.max[0], bounds.min[1], bounds.min[2]],
            [bounds.min[0], bounds.max[1], bounds.min[2]],
            [bounds.max[0], bounds.max[1], bounds.min[2]],
            [bounds.min[0], bounds.min[1], bounds.max[2]],
            [bounds.max[0], bounds.min[1], bounds.max[2]],
            [bounds.min[0], bounds.max[1], bounds.max[2]],
            [bounds.max[0], bounds.max[1], bounds.max[2]],
        ];

        // Transform each corner and compute new AABB
        let mut result = BoundingBox::empty();
        for corner in &corners {
            let transformed = self.transform_point(*corner);
            result.expand_point(&transformed);
        }
        result
    }
}

impl Default for TlasInstance {
    fn default() -> Self {
        Self::new()
    }
}

// Verify TlasInstance is exactly 64 bytes (required for GPU compatibility)
const _: () = assert!(std::mem::size_of::<TlasInstance>() == TLAS_INSTANCE_SIZE);

// ---------------------------------------------------------------------------
// TlasError
// ---------------------------------------------------------------------------

/// Errors that can occur during TLAS construction.
#[derive(Debug)]
pub enum TlasError {
    /// No instances provided for TLAS construction.
    NoInstances,
    /// Too many instances (exceeds max_instances).
    TooManyInstances { provided: u32, max: u32 },
    /// Invalid BLAS address (null or invalid).
    InvalidBlasAddress { instance_index: usize },
    /// Attempted update on TLAS not built with ALLOW_UPDATE.
    UpdateNotAllowed,
    /// Instance count changed during update.
    InstanceCountMismatch { expected: u32, actual: u32 },
    /// GPU acceleration structure build failed.
    BuildFailed(String),
}

impl std::fmt::Display for TlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInstances => write!(f, "cannot build TLAS without instances"),
            Self::TooManyInstances { provided, max } => {
                write!(
                    f,
                    "too many instances: {} provided, max is {}",
                    provided, max
                )
            }
            Self::InvalidBlasAddress { instance_index } => {
                write!(
                    f,
                    "invalid BLAS address at instance index {}",
                    instance_index
                )
            }
            Self::UpdateNotAllowed => {
                write!(f, "TLAS was not built with ALLOW_UPDATE flag")
            }
            Self::InstanceCountMismatch { expected, actual } => {
                write!(
                    f,
                    "instance count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::BuildFailed(msg) => write!(f, "TLAS build failed: {}", msg),
        }
    }
}

impl std::error::Error for TlasError {}

// ---------------------------------------------------------------------------
// TlasBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing TLAS from instance data.
#[derive(Debug)]
pub struct TlasBuilder {
    /// Build configuration.
    config: TlasConfig,
    /// BLAS instances.
    instances: Vec<TlasInstance>,
    /// Optional per-instance bounding boxes (for world-space bounds computation).
    instance_bounds: Vec<BoundingBox>,
}

impl TlasBuilder {
    /// Create a new TLAS builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: TlasConfig::default(),
            instances: Vec::new(),
            instance_bounds: Vec::new(),
        }
    }

    /// Set the build configuration.
    pub fn config(mut self, config: TlasConfig) -> Self {
        self.config = config;
        self
    }

    /// Add instances to the TLAS.
    pub fn instances(mut self, instances: &[TlasInstance]) -> Self {
        self.instances.extend_from_slice(instances);
        self
    }

    /// Add a single instance to the TLAS.
    pub fn instance(mut self, instance: TlasInstance) -> Self {
        self.instances.push(instance);
        self
    }

    /// Add instances with their local bounding boxes.
    /// The bounding boxes will be transformed to world space for TLAS bounds.
    pub fn instances_with_bounds(
        mut self,
        instances: &[TlasInstance],
        bounds: &[BoundingBox],
    ) -> Self {
        debug_assert_eq!(
            instances.len(),
            bounds.len(),
            "Instance and bounds count mismatch"
        );
        self.instances.extend_from_slice(instances);
        self.instance_bounds.extend_from_slice(bounds);
        self
    }

    /// Validate the builder state.
    fn validate(&self) -> Result<(), TlasError> {
        if self.instances.is_empty() {
            return Err(TlasError::NoInstances);
        }

        let instance_count = self.instances.len() as u32;
        if instance_count > self.config.max_instances {
            return Err(TlasError::TooManyInstances {
                provided: instance_count,
                max: self.config.max_instances,
            });
        }

        // Validate BLAS addresses
        for (i, instance) in self.instances.iter().enumerate() {
            if instance.blas_address == 0 {
                return Err(TlasError::InvalidBlasAddress { instance_index: i });
            }
        }

        Ok(())
    }

    /// Build the TLAS.
    ///
    /// # Returns
    ///
    /// A `Tlas` structure containing the built acceleration structure.
    ///
    /// # Errors
    ///
    /// Returns `TlasError` if:
    /// - No instances were provided
    /// - Instance count exceeds max_instances
    /// - Any instance has an invalid BLAS address
    pub fn build(self) -> Result<Tlas, TlasError> {
        self.validate()?;

        let instance_count = self.instances.len() as u32;

        // Compute world-space bounds from instance bounds if available
        let bounds = if !self.instance_bounds.is_empty() {
            let mut world_bounds = BoundingBox::empty();
            for (instance, local_bounds) in self.instances.iter().zip(self.instance_bounds.iter()) {
                let transformed = instance.transform_bounds(local_bounds);
                world_bounds.expand_box(&transformed);
            }
            world_bounds
        } else {
            // Without per-instance bounds, we can't compute world bounds
            BoundingBox::empty()
        };

        // Estimate memory size
        let base_memory = MIN_TLAS_MEMORY + instance_count as usize * BYTES_PER_INSTANCE;

        // Update-enabled TLAS requires extra memory
        let memory_size = if self.config.allows_update() {
            (base_memory as f32 * SCRATCH_BUFFER_MULTIPLIER) as usize
        } else {
            base_memory
        };

        // Calculate scratch buffer size
        let scratch_size = (memory_size as f32 * SCRATCH_BUFFER_MULTIPLIER) as usize;

        Ok(Tlas {
            instance_count,
            memory_size,
            bounds,
            scratch_size,
            config: self.config,
            instances: if self.config.allows_update() {
                Some(self.instances)
            } else {
                None
            },
        })
    }
}

impl Default for TlasBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tlas
// ---------------------------------------------------------------------------

/// Built Top-Level Acceleration Structure.
///
/// Contains the acceleration structure data for ray tracing scene traversal.
/// When wgpu ray tracing support is available, this will hold the
/// actual GPU acceleration structure handle.
#[derive(Debug)]
pub struct Tlas {
    /// Number of instances in the TLAS.
    pub instance_count: u32,
    /// Current memory usage in bytes.
    pub memory_size: usize,
    /// World-space bounding box of all instances.
    pub bounds: BoundingBox,
    /// Scratch buffer size needed for building.
    pub scratch_size: usize,

    /// Configuration used to build this TLAS.
    config: TlasConfig,

    /// Original instance data (only stored if allow_update is true).
    instances: Option<Vec<TlasInstance>>,
}

impl Tlas {
    /// Get the configuration used to build this TLAS.
    pub fn config(&self) -> &TlasConfig {
        &self.config
    }

    /// Check if this TLAS supports in-place updates.
    pub fn supports_update(&self) -> bool {
        self.config.allows_update()
    }

    /// Get the instances (if available).
    ///
    /// Only available if TLAS was built with `ALLOW_UPDATE`.
    pub fn instances(&self) -> Option<&[TlasInstance]> {
        self.instances.as_deref()
    }

    /// Update the TLAS with new instance data.
    ///
    /// The instance count must remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `new_instances` - Updated instances
    /// * `new_bounds` - Optional new per-instance bounds for world bounds computation
    ///
    /// # Errors
    ///
    /// Returns `TlasError::UpdateNotAllowed` if the TLAS was not built
    /// with the `ALLOW_UPDATE` flag.
    ///
    /// Returns `TlasError::InstanceCountMismatch` if the new instance count
    /// doesn't match the original.
    pub fn update(
        &mut self,
        new_instances: &[TlasInstance],
        new_bounds: Option<&[BoundingBox]>,
    ) -> Result<(), TlasError> {
        if !self.config.allows_update() {
            return Err(TlasError::UpdateNotAllowed);
        }

        let new_count = new_instances.len() as u32;
        if new_count != self.instance_count {
            return Err(TlasError::InstanceCountMismatch {
                expected: self.instance_count,
                actual: new_count,
            });
        }

        // Update world bounds if instance bounds are provided
        if let Some(bounds) = new_bounds {
            let mut world_bounds = BoundingBox::empty();
            for (instance, local_bounds) in new_instances.iter().zip(bounds.iter()) {
                let transformed = instance.transform_bounds(local_bounds);
                world_bounds.expand_box(&transformed);
            }
            self.bounds = world_bounds;
        }

        // Update stored instances
        if let Some(ref mut stored) = self.instances {
            stored.clear();
            stored.extend_from_slice(new_instances);
        }

        Ok(())
    }

    /// Get the memory efficiency ratio (lower is better).
    ///
    /// Returns bytes per instance, a metric for memory efficiency.
    pub fn memory_efficiency(&self) -> f32 {
        if self.instance_count == 0 {
            return 0.0;
        }
        self.memory_size as f32 / self.instance_count as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    /// Create a simple test instance.
    fn make_instance(blas_address: u64) -> TlasInstance {
        TlasInstance::new().with_blas_address(blas_address)
    }

    /// Create an instance with translation.
    fn make_translated_instance(blas_address: u64, x: f32, y: f32, z: f32) -> TlasInstance {
        TlasInstance::new()
            .with_blas_address(blas_address)
            .with_transform(TlasInstance::translation_transform(x, y, z))
    }

    /// Create test instances.
    fn make_test_instances(count: usize) -> Vec<TlasInstance> {
        (0..count)
            .map(|i| make_instance((i + 1) as u64))
            .collect()
    }

    /// Create a unit cube bounding box.
    fn unit_cube_bounds() -> BoundingBox {
        BoundingBox::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    }

    // -------------------------------------------------------------------------
    // Test: Instance layout size (must be 64 bytes)
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_size_64_bytes() {
        assert_eq!(
            std::mem::size_of::<TlasInstance>(),
            64,
            "TlasInstance must be exactly 64 bytes for GPU compatibility"
        );
    }

    #[test]
    fn test_instance_alignment() {
        // The instance should have proper alignment
        assert!(std::mem::align_of::<TlasInstance>() >= 8);
    }

    // -------------------------------------------------------------------------
    // Test: Transform packing
    // -------------------------------------------------------------------------

    #[test]
    fn test_identity_transform() {
        let instance = TlasInstance::new();

        // Identity transform should not move points
        let point = [1.0, 2.0, 3.0];
        let transformed = instance.transform_point(point);

        assert!((transformed[0] - 1.0).abs() < 0.0001);
        assert!((transformed[1] - 2.0).abs() < 0.0001);
        assert!((transformed[2] - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_translation_transform() {
        let instance = TlasInstance::new()
            .with_transform(TlasInstance::translation_transform(10.0, 20.0, 30.0))
            .with_blas_address(1);

        let point = [0.0, 0.0, 0.0];
        let transformed = instance.transform_point(point);

        assert!((transformed[0] - 10.0).abs() < 0.0001);
        assert!((transformed[1] - 20.0).abs() < 0.0001);
        assert!((transformed[2] - 30.0).abs() < 0.0001);
    }

    #[test]
    fn test_scale_transform() {
        let instance = TlasInstance::new()
            .with_transform(TlasInstance::scale_transform(2.0, 3.0, 4.0))
            .with_blas_address(1);

        let point = [1.0, 1.0, 1.0];
        let transformed = instance.transform_point(point);

        assert!((transformed[0] - 2.0).abs() < 0.0001);
        assert!((transformed[1] - 3.0).abs() < 0.0001);
        assert!((transformed[2] - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_combined_transform() {
        // Scale then translate via matrix multiplication
        let transform = [
            [2.0, 0.0, 0.0, 5.0], // scale X by 2, translate X by 5
            [0.0, 2.0, 0.0, 0.0], // scale Y by 2
            [0.0, 0.0, 2.0, 0.0], // scale Z by 2
        ];
        let instance = TlasInstance::new()
            .with_transform(transform)
            .with_blas_address(1);

        let point = [1.0, 1.0, 1.0];
        let transformed = instance.transform_point(point);

        // 1.0 * 2.0 + 5.0 = 7.0 for X
        assert!((transformed[0] - 7.0).abs() < 0.0001);
        assert!((transformed[1] - 2.0).abs() < 0.0001);
        assert!((transformed[2] - 2.0).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // Test: Mask/flags packing
    // -------------------------------------------------------------------------

    #[test]
    fn test_custom_index_packing() {
        let instance = TlasInstance::new()
            .with_custom_index(0x123456)
            .with_blas_address(1);

        assert_eq!(instance.custom_index(), 0x123456);
    }

    #[test]
    fn test_hit_group_offset_packing() {
        let instance = TlasInstance::new()
            .with_hit_group_offset(0xAB)
            .with_blas_address(1);

        assert_eq!(instance.hit_group_offset(), 0xAB);
    }

    #[test]
    fn test_custom_index_and_hit_group_combined() {
        let instance = TlasInstance::new()
            .with_custom_index(0xFEDCBA)
            .with_hit_group_offset(0x12)
            .with_blas_address(1);

        assert_eq!(instance.custom_index(), 0xFEDCBA);
        assert_eq!(instance.hit_group_offset(), 0x12);

        // Verify packed value
        assert_eq!(instance.instance_custom_index_and_hit_group, 0x12FEDCBA);
    }

    #[test]
    fn test_custom_index_max_value() {
        let instance = TlasInstance::new()
            .with_custom_index(0x00FFFFFF) // Max 24-bit value
            .with_blas_address(1);

        assert_eq!(instance.custom_index(), 0x00FFFFFF);
    }

    #[test]
    fn test_mask_setting() {
        let instance = TlasInstance::new()
            .with_mask(0b10101010)
            .with_blas_address(1);

        assert_eq!(instance.mask, 0b10101010);
    }

    #[test]
    fn test_flags_setting() {
        let flags = INSTANCE_FLAG_FORCE_OPAQUE | INSTANCE_FLAG_TRIANGLE_FLIP_FACING;
        let instance = TlasInstance::new()
            .with_flags(flags)
            .with_blas_address(1);

        assert_eq!(instance.flags, flags);
    }

    // -------------------------------------------------------------------------
    // Test: Builder basic usage
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_basic() {
        let instances = make_test_instances(5);

        let tlas = TlasBuilder::new()
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        assert_eq!(tlas.instance_count, 5);
        assert!(tlas.memory_size > 0);
        assert!(tlas.scratch_size > 0);
    }

    #[test]
    fn test_builder_with_config() {
        let instances = make_test_instances(3);

        let config = TlasConfig::new()
            .with_fast_build()
            .with_max_instances(100);

        let tlas = TlasBuilder::new()
            .config(config)
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        assert_eq!(tlas.instance_count, 3);
        assert!(tlas.config().prefers_fast_build());
    }

    #[test]
    fn test_builder_single_instance() {
        let instance = make_instance(1);

        let tlas = TlasBuilder::new()
            .instance(instance)
            .build()
            .expect("Failed to build TLAS");

        assert_eq!(tlas.instance_count, 1);
    }

    #[test]
    fn test_builder_fluent_api() {
        let tlas = TlasBuilder::new()
            .config(TlasConfig::for_static_scene())
            .instance(make_instance(1))
            .instance(make_instance(2))
            .instance(make_instance(3))
            .build()
            .expect("Failed to build TLAS");

        assert_eq!(tlas.instance_count, 3);
        assert!(tlas.config().prefers_fast_trace());
    }

    // -------------------------------------------------------------------------
    // Test: Empty instances error
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_instances_error() {
        let result = TlasBuilder::new().build();

        assert!(matches!(result, Err(TlasError::NoInstances)));
    }

    #[test]
    fn test_empty_instances_slice_error() {
        let empty: Vec<TlasInstance> = Vec::new();
        let result = TlasBuilder::new().instances(&empty).build();

        assert!(matches!(result, Err(TlasError::NoInstances)));
    }

    // -------------------------------------------------------------------------
    // Test: Too many instances error
    // -------------------------------------------------------------------------

    #[test]
    fn test_too_many_instances_error() {
        let instances = make_test_instances(100);

        let config = TlasConfig::new().with_max_instances(50);

        let result = TlasBuilder::new()
            .config(config)
            .instances(&instances)
            .build();

        assert!(matches!(
            result,
            Err(TlasError::TooManyInstances {
                provided: 100,
                max: 50
            })
        ));
    }

    // -------------------------------------------------------------------------
    // Test: Invalid BLAS address error
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalid_blas_address_error() {
        let instance = TlasInstance::new(); // blas_address is 0

        let result = TlasBuilder::new().instance(instance).build();

        assert!(matches!(
            result,
            Err(TlasError::InvalidBlasAddress { instance_index: 0 })
        ));
    }

    #[test]
    fn test_invalid_blas_address_at_index() {
        let instances = vec![
            make_instance(1),
            make_instance(2),
            TlasInstance::new(), // Invalid - blas_address is 0
            make_instance(4),
        ];

        let result = TlasBuilder::new().instances(&instances).build();

        assert!(matches!(
            result,
            Err(TlasError::InvalidBlasAddress { instance_index: 2 })
        ));
    }

    // -------------------------------------------------------------------------
    // Test: Large TLAS (1000 instances)
    // -------------------------------------------------------------------------

    #[test]
    fn test_large_tlas_1000_instances() {
        let instances = make_test_instances(1000);

        let tlas = TlasBuilder::new()
            .config(TlasConfig::new().with_max_instances(2000))
            .instances(&instances)
            .build()
            .expect("Failed to build large TLAS");

        assert_eq!(tlas.instance_count, 1000);
        assert!(tlas.memory_size >= MIN_TLAS_MEMORY + 1000 * BYTES_PER_INSTANCE);
    }

    #[test]
    fn test_large_tlas_memory_scaling() {
        let small_tlas = TlasBuilder::new()
            .instances(&make_test_instances(10))
            .build()
            .expect("Failed to build small TLAS");

        let large_tlas = TlasBuilder::new()
            .instances(&make_test_instances(100))
            .build()
            .expect("Failed to build large TLAS");

        // Memory should scale roughly linearly
        assert!(large_tlas.memory_size > small_tlas.memory_size);
        let ratio = large_tlas.memory_size as f32 / small_tlas.memory_size as f32;
        assert!(ratio > 5.0 && ratio < 15.0); // Should be roughly 10x
    }

    // -------------------------------------------------------------------------
    // Test: Bounds computation from instances
    // -------------------------------------------------------------------------

    #[test]
    fn test_bounds_computation_single_instance() {
        let instance = make_instance(1);
        let bounds = unit_cube_bounds();

        let tlas = TlasBuilder::new()
            .instances_with_bounds(&[instance], &[bounds])
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.bounds.is_valid());
        assert!((tlas.bounds.min[0] - (-0.5)).abs() < 0.0001);
        assert!((tlas.bounds.max[0] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_bounds_computation_translated_instances() {
        let instances = vec![
            make_translated_instance(1, 0.0, 0.0, 0.0),
            make_translated_instance(2, 10.0, 0.0, 0.0),
        ];
        let bounds = vec![unit_cube_bounds(), unit_cube_bounds()];

        let tlas = TlasBuilder::new()
            .instances_with_bounds(&instances, &bounds)
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.bounds.is_valid());
        // First cube at origin: [-0.5, 0.5]
        // Second cube at x=10: [9.5, 10.5]
        // Combined: [-0.5, 10.5]
        assert!((tlas.bounds.min[0] - (-0.5)).abs() < 0.0001);
        assert!((tlas.bounds.max[0] - 10.5).abs() < 0.0001);
    }

    #[test]
    fn test_bounds_computation_scaled_instance() {
        let instance = TlasInstance::new()
            .with_transform(TlasInstance::scale_transform(2.0, 2.0, 2.0))
            .with_blas_address(1);
        let bounds = unit_cube_bounds();

        let tlas = TlasBuilder::new()
            .instances_with_bounds(&[instance], &[bounds])
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.bounds.is_valid());
        // Unit cube scaled 2x: [-1.0, 1.0]
        assert!((tlas.bounds.min[0] - (-1.0)).abs() < 0.0001);
        assert!((tlas.bounds.max[0] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_bounds_without_instance_bounds() {
        let instances = make_test_instances(5);

        let tlas = TlasBuilder::new()
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        // Without instance bounds, world bounds should be empty
        assert!(!tlas.bounds.is_valid());
    }

    // -------------------------------------------------------------------------
    // Test: Config flags
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = TlasConfig::default();

        assert!(!config.prefers_fast_build());
        assert!(!config.prefers_fast_trace());
        assert!(!config.allows_update());
        assert_eq!(config.max_instances, DEFAULT_MAX_INSTANCES);
    }

    #[test]
    fn test_config_fast_build() {
        let config = TlasConfig::new().with_fast_build();

        assert!(config.prefers_fast_build());
        assert!(!config.prefers_fast_trace());
    }

    #[test]
    fn test_config_fast_trace() {
        let config = TlasConfig::new().with_fast_trace();

        assert!(!config.prefers_fast_build());
        assert!(config.prefers_fast_trace());
    }

    #[test]
    fn test_config_fast_build_clears_fast_trace() {
        let config = TlasConfig::new().with_fast_trace().with_fast_build();

        assert!(config.prefers_fast_build());
        assert!(!config.prefers_fast_trace());
    }

    #[test]
    fn test_config_fast_trace_clears_fast_build() {
        let config = TlasConfig::new().with_fast_build().with_fast_trace();

        assert!(!config.prefers_fast_build());
        assert!(config.prefers_fast_trace());
    }

    #[test]
    fn test_config_for_static_scene() {
        let config = TlasConfig::for_static_scene();

        assert!(config.prefers_fast_trace());
        assert!(!config.allows_update());
    }

    #[test]
    fn test_config_for_dynamic_scene() {
        let config = TlasConfig::for_dynamic_scene();

        assert!(config.prefers_fast_build());
        assert!(config.allows_update());
    }

    // -------------------------------------------------------------------------
    // Test: Update functionality
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_with_allow_update() {
        let instances = make_test_instances(3);

        let mut tlas = TlasBuilder::new()
            .config(TlasConfig::new().with_allow_update())
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        let new_instances = vec![
            make_translated_instance(1, 5.0, 0.0, 0.0),
            make_translated_instance(2, 10.0, 0.0, 0.0),
            make_translated_instance(3, 15.0, 0.0, 0.0),
        ];

        tlas.update(&new_instances, None)
            .expect("Update should succeed");

        assert!(tlas.instances().is_some());
        assert_eq!(tlas.instances().unwrap().len(), 3);
    }

    #[test]
    fn test_update_not_allowed_error() {
        let instances = make_test_instances(3);

        let mut tlas = TlasBuilder::new()
            .config(TlasConfig::new()) // No ALLOW_UPDATE
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        let new_instances = make_test_instances(3);

        let result = tlas.update(&new_instances, None);
        assert!(matches!(result, Err(TlasError::UpdateNotAllowed)));
    }

    #[test]
    fn test_update_count_mismatch_error() {
        let instances = make_test_instances(3);

        let mut tlas = TlasBuilder::new()
            .config(TlasConfig::new().with_allow_update())
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        let new_instances = make_test_instances(5); // Wrong count

        let result = tlas.update(&new_instances, None);
        assert!(matches!(
            result,
            Err(TlasError::InstanceCountMismatch {
                expected: 3,
                actual: 5
            })
        ));
    }

    #[test]
    fn test_update_with_bounds() {
        let instances = make_test_instances(2);
        let bounds = vec![unit_cube_bounds(), unit_cube_bounds()];

        let mut tlas = TlasBuilder::new()
            .config(TlasConfig::new().with_allow_update())
            .instances_with_bounds(&instances, &bounds)
            .build()
            .expect("Failed to build TLAS");

        // Move both instances
        let new_instances = vec![
            make_translated_instance(1, 100.0, 0.0, 0.0),
            make_translated_instance(2, 200.0, 0.0, 0.0),
        ];
        let new_bounds = vec![unit_cube_bounds(), unit_cube_bounds()];

        tlas.update(&new_instances, Some(&new_bounds))
            .expect("Update should succeed");

        // Bounds should be updated
        assert!(tlas.bounds.min[0] > 90.0);
        assert!(tlas.bounds.max[0] > 190.0);
    }

    // -------------------------------------------------------------------------
    // Test: Memory tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_tracking() {
        let instances = make_test_instances(50);

        let tlas = TlasBuilder::new()
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.memory_size >= MIN_TLAS_MEMORY);
        assert!(tlas.scratch_size > tlas.memory_size);
    }

    #[test]
    fn test_memory_efficiency() {
        let tlas = TlasBuilder::new()
            .instances(&make_test_instances(100))
            .build()
            .expect("Failed to build TLAS");

        let efficiency = tlas.memory_efficiency();
        assert!(efficiency > 0.0);
        assert!(efficiency < 10000.0); // Less than 10KB per instance
    }

    #[test]
    fn test_update_enabled_uses_more_memory() {
        let instances = make_test_instances(50);

        let static_tlas = TlasBuilder::new()
            .config(TlasConfig::for_static_scene())
            .instances(&instances)
            .build()
            .expect("Failed to build static TLAS");

        let dynamic_tlas = TlasBuilder::new()
            .config(TlasConfig::for_dynamic_scene())
            .instances(&instances)
            .build()
            .expect("Failed to build dynamic TLAS");

        assert!(
            dynamic_tlas.memory_size > static_tlas.memory_size,
            "Dynamic TLAS should use more memory"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Instance access
    // -------------------------------------------------------------------------

    #[test]
    fn test_instances_access_with_update() {
        let instances = make_test_instances(5);

        let tlas = TlasBuilder::new()
            .config(TlasConfig::new().with_allow_update())
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.instances().is_some());
        assert_eq!(tlas.instances().unwrap().len(), 5);
    }

    #[test]
    fn test_instances_access_without_update() {
        let instances = make_test_instances(5);

        let tlas = TlasBuilder::new()
            .config(TlasConfig::new()) // No ALLOW_UPDATE
            .instances(&instances)
            .build()
            .expect("Failed to build TLAS");

        assert!(tlas.instances().is_none());
    }

    // -------------------------------------------------------------------------
    // Test: Transform bounds computation
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_bounds_identity() {
        let instance = TlasInstance::new().with_blas_address(1);
        let bounds = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let transformed = instance.transform_bounds(&bounds);

        assert!(transformed.is_valid());
        assert!((transformed.min[0] - 0.0).abs() < 0.0001);
        assert!((transformed.max[0] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_transform_bounds_translation() {
        let instance = TlasInstance::new()
            .with_transform(TlasInstance::translation_transform(5.0, 5.0, 5.0))
            .with_blas_address(1);
        let bounds = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let transformed = instance.transform_bounds(&bounds);

        assert!(transformed.is_valid());
        assert!((transformed.min[0] - 5.0).abs() < 0.0001);
        assert!((transformed.max[0] - 6.0).abs() < 0.0001);
    }

    #[test]
    fn test_transform_bounds_scale() {
        let instance = TlasInstance::new()
            .with_transform(TlasInstance::scale_transform(2.0, 3.0, 4.0))
            .with_blas_address(1);
        let bounds = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let transformed = instance.transform_bounds(&bounds);

        assert!(transformed.is_valid());
        assert!((transformed.min[0] - 0.0).abs() < 0.0001);
        assert!((transformed.max[0] - 2.0).abs() < 0.0001);
        assert!((transformed.max[1] - 3.0).abs() < 0.0001);
        assert!((transformed.max[2] - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_transform_bounds_empty() {
        let instance = TlasInstance::new().with_blas_address(1);
        let bounds = BoundingBox::empty();

        let transformed = instance.transform_bounds(&bounds);

        assert!(!transformed.is_valid());
    }

    // -------------------------------------------------------------------------
    // Test: Error display
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = TlasError::NoInstances;
        assert_eq!(format!("{}", err), "cannot build TLAS without instances");

        let err = TlasError::TooManyInstances {
            provided: 100,
            max: 50,
        };
        assert!(format!("{}", err).contains("100"));
        assert!(format!("{}", err).contains("50"));

        let err = TlasError::InvalidBlasAddress { instance_index: 5 };
        assert!(format!("{}", err).contains("5"));

        let err = TlasError::UpdateNotAllowed;
        assert!(format!("{}", err).contains("ALLOW_UPDATE"));

        let err = TlasError::InstanceCountMismatch {
            expected: 10,
            actual: 20,
        };
        assert!(format!("{}", err).contains("10"));
        assert!(format!("{}", err).contains("20"));

        let err = TlasError::BuildFailed("GPU error".to_string());
        assert!(format!("{}", err).contains("GPU error"));
    }

    // -------------------------------------------------------------------------
    // Test: Default implementations
    // -------------------------------------------------------------------------

    #[test]
    fn test_tlas_instance_default() {
        let instance = TlasInstance::default();

        assert_eq!(instance.transform, TlasInstance::identity_transform());
        assert_eq!(instance.mask, 0xFF);
        assert_eq!(instance.flags, 0);
        assert_eq!(instance.blas_address, 0);
        assert_eq!(instance.custom_index(), 0);
        assert_eq!(instance.hit_group_offset(), 0);
    }

    #[test]
    fn test_tlas_config_default() {
        let config = TlasConfig::default();
        let new_config = TlasConfig::new();

        assert_eq!(config, new_config);
    }

    #[test]
    fn test_tlas_builder_default() {
        // Should be equivalent to new()
        let builder1 = TlasBuilder::default();
        let builder2 = TlasBuilder::new();

        // Both should fail the same way with no instances
        let result1 = builder1.build();
        let result2 = builder2.build();

        assert!(matches!(result1, Err(TlasError::NoInstances)));
        assert!(matches!(result2, Err(TlasError::NoInstances)));
    }
}
