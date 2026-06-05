//! Foliage GPU Instancing (T-ENV-1.11)
//!
//! This module provides efficient GPU-driven instanced rendering for vegetation
//! and foliage systems. It handles per-instance data storage, frustum culling,
//! LOD selection, and wind animation parameters.
//!
//! # Architecture
//!
//! - `FoliageInstance`: Per-instance GPU data (position, rotation, scale, color, wind)
//! - `FoliageBuffer`: Instance buffer management with streaming updates
//! - `FoliageCuller`: Frustum culling for visible instance extraction
//! - `FoliageLodSelector`: Distance-based LOD selection
//! - `FoliageBatch`: Draw call batching for efficient rendering
//!
//! # GPU Memory Layout
//!
//! The `FoliageInstance` struct is `repr(C)` with 64 bytes total, aligned for
//! efficient GPU access:
//!
//! | Offset | Field       | Size    |
//! |--------|-------------|---------|
//! | 0      | position    | 12 bytes |
//! | 12     | scale       | 4 bytes  |
//! | 16     | rotation    | 16 bytes |
//! | 32     | color       | 16 bytes |
//! | 48     | wind_phase  | 4 bytes  |
//! | 52     | lod_bias    | 4 bytes  |
//! | 56     | _padding    | 8 bytes  |
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::foliage_instancing::{FoliageBuffer, FoliageInstance, FoliageConfig};
//!
//! // Create foliage buffer
//! let config = FoliageConfig::default().with_max_instances(100_000);
//! let mut buffer = FoliageBuffer::new(config);
//!
//! // Add foliage instances
//! let instance = FoliageInstance::new([10.0, 0.0, 5.0], 1.0);
//! buffer.add_instance(instance)?;
//!
//! // Cull against frustum
//! let visible_indices = buffer.cull_frustum(&frustum_planes);
//!
//! // Select LODs for visible instances
//! let lods = buffer.select_lod([0.0, 5.0, 0.0], &[50.0, 100.0, 200.0]);
//!
//! // Generate draw batches
//! let batches = buffer.generate_batches(&visible_indices, &lods);
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum instances for the foliage buffer.
pub const DEFAULT_MAX_INSTANCES: u32 = 100_000;

/// Size of FoliageInstance in bytes (64 bytes for GPU alignment).
pub const FOLIAGE_INSTANCE_SIZE: usize = 64;

/// Default buffer grow factor when resizing.
pub const BUFFER_GROW_FACTOR: f32 = 1.5;

/// Minimum buffer capacity.
pub const MIN_BUFFER_CAPACITY: usize = 1024;

/// Maximum number of LOD levels supported.
pub const MAX_LOD_LEVELS: usize = 8;

/// Default wind animation frequency (Hz).
pub const DEFAULT_WIND_FREQUENCY: f32 = 1.0;

/// Default wind animation amplitude.
pub const DEFAULT_WIND_AMPLITUDE: f32 = 0.1;

// ---------------------------------------------------------------------------
// FoliageError
// ---------------------------------------------------------------------------

/// Errors that can occur during foliage buffer operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FoliageError {
    /// Buffer capacity exceeded and cannot grow.
    BufferFull { max: u32 },
    /// Instance data contains invalid values (NaN, Inf).
    InvalidInstance { reason: &'static str },
    /// Invalid LOD threshold configuration.
    InvalidLodThresholds { reason: &'static str },
    /// Buffer index out of bounds.
    IndexOutOfBounds { index: u32, count: u32 },
}

impl std::fmt::Display for FoliageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull { max } => {
                write!(f, "foliage buffer full: max capacity is {}", max)
            }
            Self::InvalidInstance { reason } => {
                write!(f, "invalid foliage instance: {}", reason)
            }
            Self::InvalidLodThresholds { reason } => {
                write!(f, "invalid LOD thresholds: {}", reason)
            }
            Self::IndexOutOfBounds { index, count } => {
                write!(f, "index {} out of bounds (count: {})", index, count)
            }
        }
    }
}

impl std::error::Error for FoliageError {}

// ---------------------------------------------------------------------------
// FoliageInstance — GPU per-instance data
// ---------------------------------------------------------------------------

/// Per-instance data for GPU foliage rendering.
///
/// This struct is designed for efficient GPU upload with 64-byte alignment.
/// All fields are tightly packed with explicit padding for std430 compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FoliageInstance {
    /// World-space position (x, y, z).
    pub position: [f32; 3],
    /// Uniform scale factor.
    pub scale: f32,
    /// Rotation quaternion (x, y, z, w).
    pub rotation: [f32; 4],
    /// RGBA color tint (premultiplied alpha).
    pub color: [f32; 4],
    /// Wind animation phase offset (radians).
    pub wind_phase: f32,
    /// LOD bias for this instance (-1.0 to 1.0).
    pub lod_bias: f32,
    /// Padding for 64-byte alignment.
    pub _padding: [f32; 2],
}

impl FoliageInstance {
    /// Create a new foliage instance at the given position with uniform scale.
    ///
    /// Rotation is identity, color is white (no tint), and wind phase is 0.
    #[inline]
    pub fn new(position: [f32; 3], scale: f32) -> Self {
        Self {
            position,
            scale,
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            color: [1.0, 1.0, 1.0, 1.0],    // White (no tint)
            wind_phase: 0.0,
            lod_bias: 0.0,
            _padding: [0.0; 2],
        }
    }

    /// Create an instance with full parameters.
    #[inline]
    pub fn with_full_params(
        position: [f32; 3],
        scale: f32,
        rotation: [f32; 4],
        color: [f32; 4],
        wind_phase: f32,
        lod_bias: f32,
    ) -> Self {
        Self {
            position,
            scale,
            rotation,
            color,
            wind_phase,
            lod_bias,
            _padding: [0.0; 2],
        }
    }

    /// Set the rotation from a quaternion (x, y, z, w).
    #[inline]
    pub fn with_rotation(mut self, rotation: [f32; 4]) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the RGBA color tint.
    #[inline]
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set the wind animation phase offset.
    #[inline]
    pub fn with_wind_phase(mut self, phase: f32) -> Self {
        self.wind_phase = phase;
        self
    }

    /// Set the LOD bias for this instance.
    #[inline]
    pub fn with_lod_bias(mut self, bias: f32) -> Self {
        self.lod_bias = bias.clamp(-1.0, 1.0);
        self
    }

    /// Validate that all fields contain finite (non-NaN, non-Inf) values.
    pub fn validate(&self) -> Result<(), FoliageError> {
        // Check position
        for (i, &v) in self.position.iter().enumerate() {
            if !v.is_finite() {
                return Err(FoliageError::InvalidInstance {
                    reason: match i {
                        0 => "position.x is not finite",
                        1 => "position.y is not finite",
                        _ => "position.z is not finite",
                    },
                });
            }
        }

        // Check scale
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(FoliageError::InvalidInstance {
                reason: "scale must be positive and finite",
            });
        }

        // Check rotation quaternion
        for &v in &self.rotation {
            if !v.is_finite() {
                return Err(FoliageError::InvalidInstance {
                    reason: "rotation quaternion contains non-finite values",
                });
            }
        }

        // Check color
        for &v in &self.color {
            if !v.is_finite() {
                return Err(FoliageError::InvalidInstance {
                    reason: "color contains non-finite values",
                });
            }
        }

        // Check wind_phase
        if !self.wind_phase.is_finite() {
            return Err(FoliageError::InvalidInstance {
                reason: "wind_phase is not finite",
            });
        }

        Ok(())
    }

    /// Calculate the squared distance from this instance to a point.
    #[inline]
    pub fn distance_squared(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Calculate the distance from this instance to a point.
    #[inline]
    pub fn distance(&self, point: [f32; 3]) -> f32 {
        self.distance_squared(point).sqrt()
    }

    /// Compute the bounding sphere radius for frustum culling.
    ///
    /// Assumes a unit sphere at the origin scaled by `scale`.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        self.scale
    }
}

impl Default for FoliageInstance {
    fn default() -> Self {
        Self::new([0.0, 0.0, 0.0], 1.0)
    }
}

// ---------------------------------------------------------------------------
// FoliageConfig
// ---------------------------------------------------------------------------

/// Configuration for foliage buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageConfig {
    /// Maximum number of instances the buffer can hold.
    pub max_instances: u32,
    /// Initial buffer capacity.
    pub initial_capacity: u32,
    /// Enable double-buffering for streaming updates.
    pub double_buffer: bool,
    /// Auto-grow buffer when full (up to max_instances).
    pub auto_grow: bool,
    /// Validate instances on add (slower but safer).
    pub validate_on_add: bool,
}

impl FoliageConfig {
    /// Create a new config with default settings.
    pub const fn new() -> Self {
        Self {
            max_instances: DEFAULT_MAX_INSTANCES,
            initial_capacity: MIN_BUFFER_CAPACITY as u32,
            double_buffer: true,
            auto_grow: true,
            validate_on_add: false,
        }
    }

    /// Set maximum instance capacity.
    pub const fn with_max_instances(mut self, max: u32) -> Self {
        self.max_instances = max;
        self
    }

    /// Set initial buffer capacity.
    pub const fn with_initial_capacity(mut self, capacity: u32) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Enable or disable double-buffering.
    pub const fn with_double_buffer(mut self, enable: bool) -> Self {
        self.double_buffer = enable;
        self
    }

    /// Enable or disable auto-grow.
    pub const fn with_auto_grow(mut self, enable: bool) -> Self {
        self.auto_grow = enable;
        self
    }

    /// Enable or disable validation on add.
    pub const fn with_validation(mut self, enable: bool) -> Self {
        self.validate_on_add = enable;
        self
    }

    /// Configuration for large open-world scenes.
    pub const fn for_large_scene() -> Self {
        Self {
            max_instances: 500_000,
            initial_capacity: 50_000,
            double_buffer: true,
            auto_grow: true,
            validate_on_add: false,
        }
    }

    /// Configuration for small/indoor scenes.
    pub const fn for_small_scene() -> Self {
        Self {
            max_instances: 10_000,
            initial_capacity: 1_000,
            double_buffer: false,
            auto_grow: true,
            validate_on_add: true,
        }
    }
}

impl Default for FoliageConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WindParameters
// ---------------------------------------------------------------------------

/// Wind animation parameters for foliage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WindParameters {
    /// Wind direction (normalized xy, strength in z).
    pub direction: [f32; 3],
    /// Wind frequency (Hz).
    pub frequency: f32,
    /// Wind amplitude.
    pub amplitude: f32,
    /// Gust frequency (Hz).
    pub gust_frequency: f32,
    /// Gust amplitude.
    pub gust_amplitude: f32,
    /// Current time for animation.
    pub time: f32,
}

impl WindParameters {
    /// Create new wind parameters.
    pub fn new(direction: [f32; 2], strength: f32) -> Self {
        // Normalize direction
        let len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        let dir = if len > 0.0001 {
            [direction[0] / len, direction[1] / len, strength]
        } else {
            [1.0, 0.0, strength]
        };

        Self {
            direction: dir,
            frequency: DEFAULT_WIND_FREQUENCY,
            amplitude: DEFAULT_WIND_AMPLITUDE * strength,
            gust_frequency: DEFAULT_WIND_FREQUENCY * 0.3,
            gust_amplitude: DEFAULT_WIND_AMPLITUDE * strength * 0.5,
            time: 0.0,
        }
    }

    /// Update wind time for animation.
    #[inline]
    pub fn update(&mut self, delta_time: f32) {
        self.time += delta_time;
    }

    /// Calculate wind offset for a given phase.
    #[inline]
    pub fn calculate_offset(&self, phase: f32) -> f32 {
        let base = (self.time * self.frequency + phase).sin() * self.amplitude;
        let gust = (self.time * self.gust_frequency + phase * 0.7).sin() * self.gust_amplitude;
        base + gust
    }
}

impl Default for WindParameters {
    fn default() -> Self {
        Self::new([1.0, 0.0], 1.0)
    }
}

// ---------------------------------------------------------------------------
// FoliageBuffer
// ---------------------------------------------------------------------------

/// Instance buffer for GPU foliage rendering.
///
/// Manages a collection of foliage instances with support for:
/// - Streaming updates with optional double-buffering
/// - Frustum culling
/// - Distance-based LOD selection
/// - Draw batch generation
#[derive(Debug)]
pub struct FoliageBuffer {
    /// Configuration.
    config: FoliageConfig,
    /// Ping-pong buffers (only [0] used if double_buffer is false).
    buffers: [Vec<FoliageInstance>; 2],
    /// Current buffer index (0 or 1).
    current_buffer: usize,
    /// Number of instances in the current buffer.
    instance_count: u32,
    /// Current frame number.
    frame: u64,
    /// Dirty flag for GPU upload.
    dirty: bool,
}

impl FoliageBuffer {
    /// Create a new foliage buffer with the given configuration.
    pub fn new(config: FoliageConfig) -> Self {
        let capacity = config.initial_capacity as usize;
        Self {
            config,
            buffers: [
                Vec::with_capacity(capacity),
                Vec::with_capacity(capacity),
            ],
            current_buffer: 0,
            instance_count: 0,
            frame: 0,
            dirty: false,
        }
    }

    /// Begin a new frame: swap buffers if double-buffering is enabled.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);

        if self.config.double_buffer {
            // Swap to the other buffer
            self.current_buffer = 1 - self.current_buffer;
        }

        // Clear current buffer for new frame
        self.buffers[self.current_buffer].clear();
        self.instance_count = 0;
        self.dirty = false;
    }

    /// Add a single instance to the buffer.
    pub fn add_instance(&mut self, instance: FoliageInstance) -> Result<u32, FoliageError> {
        // Validate if configured
        if self.config.validate_on_add {
            instance.validate()?;
        }

        // Check capacity
        let current_count = self.buffers[self.current_buffer].len() as u32;
        if current_count >= self.config.max_instances {
            if !self.config.auto_grow {
                return Err(FoliageError::BufferFull {
                    max: self.config.max_instances,
                });
            }
            // Already at max, cannot grow further
            return Err(FoliageError::BufferFull {
                max: self.config.max_instances,
            });
        }

        // Auto-grow if needed
        let buffer = &mut self.buffers[self.current_buffer];
        if buffer.len() == buffer.capacity() && self.config.auto_grow {
            let new_capacity = ((buffer.capacity() as f32) * BUFFER_GROW_FACTOR) as usize;
            let capped_capacity = new_capacity.min(self.config.max_instances as usize);
            buffer.reserve(capped_capacity - buffer.len());
        }

        let index = buffer.len() as u32;
        buffer.push(instance);
        self.instance_count = buffer.len() as u32;
        self.dirty = true;

        Ok(index)
    }

    /// Add multiple instances to the buffer.
    pub fn add_instances(&mut self, instances: &[FoliageInstance]) -> Result<u32, FoliageError> {
        let start_index = self.buffers[self.current_buffer].len() as u32;

        for instance in instances {
            self.add_instance(*instance)?;
        }

        Ok(start_index)
    }

    /// Update an existing instance by index.
    pub fn update_instance(
        &mut self,
        index: u32,
        instance: FoliageInstance,
    ) -> Result<(), FoliageError> {
        let buffer = &mut self.buffers[self.current_buffer];
        let count = buffer.len() as u32;

        if index >= count {
            return Err(FoliageError::IndexOutOfBounds { index, count });
        }

        if self.config.validate_on_add {
            instance.validate()?;
        }

        buffer[index as usize] = instance;
        self.dirty = true;

        Ok(())
    }

    /// Remove an instance by swapping with the last and popping.
    ///
    /// Returns the index that was swapped into the removed slot, if any.
    pub fn remove_instance(&mut self, index: u32) -> Result<Option<u32>, FoliageError> {
        let buffer = &mut self.buffers[self.current_buffer];
        let count = buffer.len() as u32;

        if index >= count {
            return Err(FoliageError::IndexOutOfBounds { index, count });
        }

        let last_index = count - 1;
        if index != last_index {
            buffer.swap(index as usize, last_index as usize);
        }
        buffer.pop();
        self.instance_count = buffer.len() as u32;
        self.dirty = true;

        Ok(if index != last_index {
            Some(last_index)
        } else {
            None
        })
    }

    /// Clear all instances from the current buffer.
    pub fn clear(&mut self) {
        self.buffers[self.current_buffer].clear();
        self.instance_count = 0;
        self.dirty = true;
    }

    /// Get the number of instances in the current buffer.
    #[inline]
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0
    }

    /// Check if the buffer needs GPU upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the buffer as clean after GPU upload.
    #[inline]
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Get the current frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get read-only access to the current buffer data.
    #[inline]
    pub fn instances(&self) -> &[FoliageInstance] {
        &self.buffers[self.current_buffer]
    }

    /// Get the raw bytes of the current buffer for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.buffers[self.current_buffer])
    }

    /// Get an instance by index.
    pub fn get(&self, index: u32) -> Option<&FoliageInstance> {
        self.buffers[self.current_buffer].get(index as usize)
    }

    /// Cull instances against a frustum defined by 6 planes.
    ///
    /// Each plane is [A, B, C, D] where Ax + By + Cz + D >= 0 is inside.
    /// Returns indices of visible instances.
    pub fn cull_frustum(&self, planes: &[[f32; 4]; 6]) -> Vec<u32> {
        let buffer = &self.buffers[self.current_buffer];
        let mut visible = Vec::with_capacity(buffer.len());

        for (i, instance) in buffer.iter().enumerate() {
            if is_sphere_visible(instance.position, instance.bounding_radius(), planes) {
                visible.push(i as u32);
            }
        }

        visible
    }

    /// Cull instances and return count of visible instances.
    pub fn cull_frustum_count(&self, planes: &[[f32; 4]; 6]) -> u32 {
        let buffer = &self.buffers[self.current_buffer];
        let mut count = 0u32;

        for instance in buffer.iter() {
            if is_sphere_visible(instance.position, instance.bounding_radius(), planes) {
                count += 1;
            }
        }

        count
    }

    /// Select LOD level for each instance based on distance to camera.
    ///
    /// `thresholds` are squared distances for each LOD transition.
    /// Returns LOD level (0-255) for each instance.
    pub fn select_lod(&self, camera_pos: [f32; 3], thresholds: &[f32]) -> Vec<u8> {
        let buffer = &self.buffers[self.current_buffer];
        let mut lods = Vec::with_capacity(buffer.len());

        for instance in buffer.iter() {
            let dist_sq = instance.distance_squared(camera_pos);
            // Apply LOD bias: negative bias = use higher detail, positive = lower detail
            let biased_dist_sq = dist_sq * (1.0 + instance.lod_bias * 0.5);
            let lod = select_lod_level(biased_dist_sq, thresholds);
            lods.push(lod);
        }

        lods
    }

    /// Select LOD levels only for specific instances.
    pub fn select_lod_for_indices(
        &self,
        indices: &[u32],
        camera_pos: [f32; 3],
        thresholds: &[f32],
    ) -> Vec<u8> {
        let buffer = &self.buffers[self.current_buffer];
        let mut lods = Vec::with_capacity(indices.len());

        for &idx in indices {
            if let Some(instance) = buffer.get(idx as usize) {
                let dist_sq = instance.distance_squared(camera_pos);
                let biased_dist_sq = dist_sq * (1.0 + instance.lod_bias * 0.5);
                let lod = select_lod_level(biased_dist_sq, thresholds);
                lods.push(lod);
            }
        }

        lods
    }

    /// Generate draw batches grouped by LOD level.
    pub fn generate_batches(
        &self,
        visible_indices: &[u32],
        lods: &[u8],
    ) -> Vec<FoliageBatch> {
        if visible_indices.is_empty() {
            return Vec::new();
        }

        // Group indices by LOD
        let mut lod_indices: [Vec<u32>; MAX_LOD_LEVELS] = Default::default();

        for (i, &idx) in visible_indices.iter().enumerate() {
            if i < lods.len() {
                let lod = lods[i].min((MAX_LOD_LEVELS - 1) as u8) as usize;
                lod_indices[lod].push(idx);
            }
        }

        // Generate batches
        let mut batches = Vec::new();
        for (lod, indices) in lod_indices.iter().enumerate() {
            if !indices.is_empty() {
                batches.push(FoliageBatch {
                    lod_level: lod as u8,
                    instance_count: indices.len() as u32,
                    indices: indices.clone(),
                });
            }
        }

        batches
    }

    /// Sort instances by distance to camera (back-to-front for transparency).
    pub fn sort_by_distance(&mut self, camera_pos: [f32; 3]) {
        let buffer = &mut self.buffers[self.current_buffer];
        buffer.sort_by(|a, b| {
            let da = a.distance_squared(camera_pos);
            let db = b.distance_squared(camera_pos);
            // Back-to-front: larger distance first
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.dirty = true;
    }

    /// Get buffer size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.buffers[self.current_buffer].len() * FOLIAGE_INSTANCE_SIZE
    }

    /// Get buffer capacity in instances.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffers[self.current_buffer].capacity()
    }
}

impl Default for FoliageBuffer {
    fn default() -> Self {
        Self::new(FoliageConfig::default())
    }
}

// ---------------------------------------------------------------------------
// FoliageBatch
// ---------------------------------------------------------------------------

/// A batch of foliage instances for a single draw call.
#[derive(Debug, Clone)]
pub struct FoliageBatch {
    /// LOD level for this batch.
    pub lod_level: u8,
    /// Number of instances in this batch.
    pub instance_count: u32,
    /// Instance indices for this batch.
    pub indices: Vec<u32>,
}

impl FoliageBatch {
    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0
    }
}

// ---------------------------------------------------------------------------
// FoliageLodThresholds
// ---------------------------------------------------------------------------

/// LOD distance thresholds for foliage.
#[derive(Debug, Clone)]
pub struct FoliageLodThresholds {
    /// Squared distance thresholds for each LOD level.
    thresholds: Vec<f32>,
}

impl FoliageLodThresholds {
    /// Create LOD thresholds from distance values.
    ///
    /// Distances are converted to squared distances internally.
    pub fn from_distances(distances: &[f32]) -> Result<Self, FoliageError> {
        if distances.is_empty() {
            return Err(FoliageError::InvalidLodThresholds {
                reason: "at least one threshold required",
            });
        }

        // Verify distances are increasing
        for window in distances.windows(2) {
            if window[0] >= window[1] {
                return Err(FoliageError::InvalidLodThresholds {
                    reason: "distances must be strictly increasing",
                });
            }
        }

        // Convert to squared distances
        let thresholds: Vec<f32> = distances.iter().map(|d| d * d).collect();

        Ok(Self { thresholds })
    }

    /// Create default thresholds with exponential spacing.
    pub fn default_exponential(base_distance: f32, levels: usize) -> Self {
        let mut distances = Vec::with_capacity(levels);
        let mut d = base_distance;
        for _ in 0..levels {
            distances.push(d * d);
            d *= 2.0;
        }
        Self {
            thresholds: distances,
        }
    }

    /// Get the squared thresholds slice.
    #[inline]
    pub fn squared_thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Select LOD level for a given squared distance.
    #[inline]
    pub fn select(&self, distance_squared: f32) -> u8 {
        select_lod_level(distance_squared, &self.thresholds)
    }
}

impl Default for FoliageLodThresholds {
    fn default() -> Self {
        Self::default_exponential(50.0, 4)
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Check if a bounding sphere is visible within the frustum.
///
/// Returns true if the sphere is at least partially inside all planes.
#[inline]
fn is_sphere_visible(center: [f32; 3], radius: f32, planes: &[[f32; 4]; 6]) -> bool {
    for plane in planes {
        let distance = plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3];
        if distance < -radius {
            return false;
        }
    }
    true
}

/// Select LOD level based on squared distance and thresholds.
#[inline]
fn select_lod_level(distance_squared: f32, thresholds: &[f32]) -> u8 {
    for (i, &threshold) in thresholds.iter().enumerate() {
        if distance_squared < threshold {
            return i as u8;
        }
    }
    thresholds.len() as u8
}

/// Normalize a quaternion.
#[inline]
pub fn normalize_quaternion(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 0.0001 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0] // Identity if degenerate
    }
}

/// Create a rotation quaternion from axis-angle.
#[inline]
pub fn quaternion_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half_angle = angle * 0.5;
    let s = half_angle.sin();
    let c = half_angle.cos();

    // Normalize axis
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < 0.0001 {
        return [0.0, 0.0, 0.0, 1.0];
    }

    [
        axis[0] / len * s,
        axis[1] / len * s,
        axis[2] / len * s,
        c,
    ]
}

/// Create a rotation quaternion from Euler angles (YXZ order).
#[inline]
pub fn quaternion_from_euler(yaw: f32, pitch: f32, roll: f32) -> [f32; 4] {
    let cy = (yaw * 0.5).cos();
    let sy = (yaw * 0.5).sin();
    let cp = (pitch * 0.5).cos();
    let sp = (pitch * 0.5).sin();
    let cr = (roll * 0.5).cos();
    let sr = (roll * 0.5).sin();

    [
        cy * sp * cr + sy * cp * sr, // x
        sy * cp * cr - cy * sp * sr, // y
        cy * cp * sr - sy * sp * cr, // z
        cy * cp * cr + sy * sp * sr, // w
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FoliageInstance Tests
    // ========================================================================

    #[test]
    fn test_instance_struct_size() {
        assert_eq!(std::mem::size_of::<FoliageInstance>(), FOLIAGE_INSTANCE_SIZE);
    }

    #[test]
    fn test_instance_struct_alignment() {
        assert_eq!(std::mem::align_of::<FoliageInstance>(), 4);
    }

    #[test]
    fn test_instance_pod_zeroable() {
        // Verify bytemuck traits work
        let instance = FoliageInstance::default();
        let bytes: &[u8] = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), FOLIAGE_INSTANCE_SIZE);

        let zeroed: FoliageInstance = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.scale, 0.0);
    }

    #[test]
    fn test_instance_new() {
        let instance = FoliageInstance::new([1.0, 2.0, 3.0], 2.0);
        assert_eq!(instance.position, [1.0, 2.0, 3.0]);
        assert_eq!(instance.scale, 2.0);
        assert_eq!(instance.rotation, [0.0, 0.0, 0.0, 1.0]); // Identity
        assert_eq!(instance.color, [1.0, 1.0, 1.0, 1.0]); // White
    }

    #[test]
    fn test_instance_with_full_params() {
        let instance = FoliageInstance::with_full_params(
            [1.0, 2.0, 3.0],
            2.0,
            [0.0, 0.707, 0.0, 0.707],
            [1.0, 0.5, 0.0, 1.0],
            1.5,
            -0.3,
        );
        assert_eq!(instance.wind_phase, 1.5);
        assert_eq!(instance.lod_bias, -0.3);
    }

    #[test]
    fn test_instance_builder_pattern() {
        let instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0)
            .with_rotation([0.0, 0.0, 0.707, 0.707])
            .with_color([0.0, 1.0, 0.0, 1.0])
            .with_wind_phase(3.14)
            .with_lod_bias(-0.5);

        assert_eq!(instance.rotation, [0.0, 0.0, 0.707, 0.707]);
        assert_eq!(instance.color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(instance.wind_phase, 3.14);
        assert_eq!(instance.lod_bias, -0.5);
    }

    #[test]
    fn test_instance_lod_bias_clamping() {
        let instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0).with_lod_bias(5.0);
        assert_eq!(instance.lod_bias, 1.0);

        let instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0).with_lod_bias(-5.0);
        assert_eq!(instance.lod_bias, -1.0);
    }

    #[test]
    fn test_instance_validate_valid() {
        let instance = FoliageInstance::new([1.0, 2.0, 3.0], 1.0);
        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_instance_validate_nan_position() {
        let instance = FoliageInstance::new([f32::NAN, 0.0, 0.0], 1.0);
        let result = instance.validate();
        assert!(result.is_err());
        if let Err(FoliageError::InvalidInstance { reason }) = result {
            assert!(reason.contains("position"));
        }
    }

    #[test]
    fn test_instance_validate_inf_position() {
        let instance = FoliageInstance::new([0.0, f32::INFINITY, 0.0], 1.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_zero_scale() {
        let instance = FoliageInstance::new([0.0, 0.0, 0.0], 0.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_negative_scale() {
        let instance = FoliageInstance::new([0.0, 0.0, 0.0], -1.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_nan_rotation() {
        let mut instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0);
        instance.rotation[0] = f32::NAN;
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_nan_color() {
        let mut instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0);
        instance.color[2] = f32::NAN;
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_nan_wind_phase() {
        let mut instance = FoliageInstance::new([0.0, 0.0, 0.0], 1.0);
        instance.wind_phase = f32::NAN;
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_distance_squared() {
        let instance = FoliageInstance::new([3.0, 0.0, 4.0], 1.0);
        let dist_sq = instance.distance_squared([0.0, 0.0, 0.0]);
        assert_eq!(dist_sq, 25.0); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_instance_distance() {
        let instance = FoliageInstance::new([3.0, 0.0, 4.0], 1.0);
        let dist = instance.distance([0.0, 0.0, 0.0]);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_instance_bounding_radius() {
        let instance = FoliageInstance::new([0.0, 0.0, 0.0], 2.5);
        assert_eq!(instance.bounding_radius(), 2.5);
    }

    #[test]
    fn test_instance_default() {
        let instance = FoliageInstance::default();
        assert_eq!(instance.position, [0.0, 0.0, 0.0]);
        assert_eq!(instance.scale, 1.0);
    }

    // ========================================================================
    // FoliageConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = FoliageConfig::default();
        assert_eq!(config.max_instances, DEFAULT_MAX_INSTANCES);
        assert!(config.double_buffer);
        assert!(config.auto_grow);
    }

    #[test]
    fn test_config_builder() {
        let config = FoliageConfig::new()
            .with_max_instances(50_000)
            .with_initial_capacity(5_000)
            .with_double_buffer(false)
            .with_auto_grow(false)
            .with_validation(true);

        assert_eq!(config.max_instances, 50_000);
        assert_eq!(config.initial_capacity, 5_000);
        assert!(!config.double_buffer);
        assert!(!config.auto_grow);
        assert!(config.validate_on_add);
    }

    #[test]
    fn test_config_for_large_scene() {
        let config = FoliageConfig::for_large_scene();
        assert_eq!(config.max_instances, 500_000);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_config_for_small_scene() {
        let config = FoliageConfig::for_small_scene();
        assert_eq!(config.max_instances, 10_000);
        assert!(!config.double_buffer);
        assert!(config.validate_on_add);
    }

    // ========================================================================
    // FoliageBuffer Tests
    // ========================================================================

    #[test]
    fn test_buffer_new() {
        let buffer = FoliageBuffer::new(FoliageConfig::default());
        assert_eq!(buffer.instance_count(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_dirty());
    }

    #[test]
    fn test_buffer_add_instance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        let instance = FoliageInstance::new([1.0, 2.0, 3.0], 1.0);

        let index = buffer.add_instance(instance).unwrap();
        assert_eq!(index, 0);
        assert_eq!(buffer.instance_count(), 1);
        assert!(!buffer.is_empty());
        assert!(buffer.is_dirty());
    }

    #[test]
    fn test_buffer_add_multiple_instances() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());

        for i in 0..10 {
            let instance = FoliageInstance::new([i as f32, 0.0, 0.0], 1.0);
            let index = buffer.add_instance(instance).unwrap();
            assert_eq!(index, i);
        }

        assert_eq!(buffer.instance_count(), 10);
    }

    #[test]
    fn test_buffer_add_instances_batch() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        let instances: Vec<_> = (0..5)
            .map(|i| FoliageInstance::new([i as f32, 0.0, 0.0], 1.0))
            .collect();

        let start_index = buffer.add_instances(&instances).unwrap();
        assert_eq!(start_index, 0);
        assert_eq!(buffer.instance_count(), 5);
    }

    #[test]
    fn test_buffer_get_instance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        let instance = FoliageInstance::new([1.0, 2.0, 3.0], 2.0);
        buffer.add_instance(instance).unwrap();

        let retrieved = buffer.get(0).unwrap();
        assert_eq!(retrieved.position, [1.0, 2.0, 3.0]);
        assert_eq!(retrieved.scale, 2.0);

        assert!(buffer.get(1).is_none());
    }

    #[test]
    fn test_buffer_update_instance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([0.0, 0.0, 0.0], 1.0)).unwrap();

        let updated = FoliageInstance::new([5.0, 5.0, 5.0], 3.0);
        buffer.update_instance(0, updated).unwrap();

        let retrieved = buffer.get(0).unwrap();
        assert_eq!(retrieved.position, [5.0, 5.0, 5.0]);
        assert_eq!(retrieved.scale, 3.0);
    }

    #[test]
    fn test_buffer_update_instance_out_of_bounds() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        let result = buffer.update_instance(0, FoliageInstance::default());
        assert!(matches!(result, Err(FoliageError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_buffer_remove_instance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());

        for i in 0..3 {
            buffer.add_instance(FoliageInstance::new([i as f32, 0.0, 0.0], 1.0)).unwrap();
        }

        // Remove middle instance (index 1)
        let swapped = buffer.remove_instance(1).unwrap();
        assert_eq!(swapped, Some(2)); // Instance at index 2 was swapped to index 1

        assert_eq!(buffer.instance_count(), 2);

        // Check that the last instance (was at 2) is now at index 1
        let at_1 = buffer.get(1).unwrap();
        assert_eq!(at_1.position[0], 2.0);
    }

    #[test]
    fn test_buffer_remove_last_instance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([0.0, 0.0, 0.0], 1.0)).unwrap();
        buffer.add_instance(FoliageInstance::new([1.0, 0.0, 0.0], 1.0)).unwrap();

        let swapped = buffer.remove_instance(1).unwrap();
        assert_eq!(swapped, None); // Last instance removed, no swap

        assert_eq!(buffer.instance_count(), 1);
    }

    #[test]
    fn test_buffer_remove_out_of_bounds() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        let result = buffer.remove_instance(0);
        assert!(matches!(result, Err(FoliageError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_buffer_clear() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        for i in 0..5 {
            buffer.add_instance(FoliageInstance::new([i as f32, 0.0, 0.0], 1.0)).unwrap();
        }

        buffer.clear();
        assert_eq!(buffer.instance_count(), 0);
        assert!(buffer.is_empty());
        assert!(buffer.is_dirty());
    }

    #[test]
    fn test_buffer_begin_frame_single_buffer() {
        let config = FoliageConfig::new().with_double_buffer(false);
        let mut buffer = FoliageBuffer::new(config);

        buffer.add_instance(FoliageInstance::default()).unwrap();
        assert_eq!(buffer.instance_count(), 1);

        buffer.begin_frame();
        assert_eq!(buffer.instance_count(), 0);
        assert_eq!(buffer.frame(), 1);
    }

    #[test]
    fn test_buffer_begin_frame_double_buffer() {
        let config = FoliageConfig::new().with_double_buffer(true);
        let mut buffer = FoliageBuffer::new(config);

        buffer.add_instance(FoliageInstance::default()).unwrap();
        buffer.begin_frame();

        // Should have swapped to other buffer
        assert_eq!(buffer.instance_count(), 0);
        assert_eq!(buffer.frame(), 1);
    }

    #[test]
    fn test_buffer_dirty_flag() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        assert!(!buffer.is_dirty());

        buffer.add_instance(FoliageInstance::default()).unwrap();
        assert!(buffer.is_dirty());

        buffer.mark_clean();
        assert!(!buffer.is_dirty());
    }

    #[test]
    fn test_buffer_as_bytes() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([1.0, 2.0, 3.0], 1.0)).unwrap();

        let bytes = buffer.as_bytes();
        assert_eq!(bytes.len(), FOLIAGE_INSTANCE_SIZE);
    }

    #[test]
    fn test_buffer_size_bytes() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        assert_eq!(buffer.size_bytes(), 0);

        buffer.add_instance(FoliageInstance::default()).unwrap();
        assert_eq!(buffer.size_bytes(), FOLIAGE_INSTANCE_SIZE);

        buffer.add_instance(FoliageInstance::default()).unwrap();
        assert_eq!(buffer.size_bytes(), FOLIAGE_INSTANCE_SIZE * 2);
    }

    #[test]
    fn test_buffer_capacity_limited() {
        let config = FoliageConfig::new()
            .with_max_instances(5)
            .with_auto_grow(false);
        let mut buffer = FoliageBuffer::new(config);

        for _ in 0..5 {
            buffer.add_instance(FoliageInstance::default()).unwrap();
        }

        let result = buffer.add_instance(FoliageInstance::default());
        assert!(matches!(result, Err(FoliageError::BufferFull { .. })));
    }

    #[test]
    fn test_buffer_validation_on_add() {
        let config = FoliageConfig::new().with_validation(true);
        let mut buffer = FoliageBuffer::new(config);

        // Valid instance should succeed
        let valid = FoliageInstance::new([1.0, 2.0, 3.0], 1.0);
        assert!(buffer.add_instance(valid).is_ok());

        // Invalid instance should fail
        let invalid = FoliageInstance::new([f32::NAN, 0.0, 0.0], 1.0);
        assert!(buffer.add_instance(invalid).is_err());
    }

    // ========================================================================
    // Frustum Culling Tests
    // ========================================================================

    fn make_simple_frustum() -> [[f32; 4]; 6] {
        // A simple box frustum from -100 to 100 on all axes
        [
            [1.0, 0.0, 0.0, 100.0],   // Left: x >= -100
            [-1.0, 0.0, 0.0, 100.0],  // Right: x <= 100
            [0.0, 1.0, 0.0, 100.0],   // Bottom: y >= -100
            [0.0, -1.0, 0.0, 100.0],  // Top: y <= 100
            [0.0, 0.0, 1.0, 100.0],   // Near: z >= -100
            [0.0, 0.0, -1.0, 100.0],  // Far: z <= 100
        ]
    }

    #[test]
    fn test_cull_frustum_inside() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([0.0, 0.0, 0.0], 1.0)).unwrap();

        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0], 0);
    }

    #[test]
    fn test_cull_frustum_outside() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        // Instance far outside frustum
        buffer.add_instance(FoliageInstance::new([500.0, 0.0, 0.0], 1.0)).unwrap();

        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert!(visible.is_empty());
    }

    #[test]
    fn test_cull_frustum_on_boundary() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        // Instance exactly on boundary but radius overlaps
        buffer.add_instance(FoliageInstance::new([100.0, 0.0, 0.0], 5.0)).unwrap();

        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert_eq!(visible.len(), 1);
    }

    #[test]
    fn test_cull_frustum_partial_overlap() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        // Instance at 105, radius 10 -> overlaps with x <= 100 plane
        buffer.add_instance(FoliageInstance::new([105.0, 0.0, 0.0], 10.0)).unwrap();

        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert_eq!(visible.len(), 1);
    }

    #[test]
    fn test_cull_frustum_mixed() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([0.0, 0.0, 0.0], 1.0)).unwrap();   // Visible
        buffer.add_instance(FoliageInstance::new([500.0, 0.0, 0.0], 1.0)).unwrap(); // Culled
        buffer.add_instance(FoliageInstance::new([50.0, 50.0, 50.0], 1.0)).unwrap(); // Visible

        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert_eq!(visible.len(), 2);
        assert!(visible.contains(&0));
        assert!(visible.contains(&2));
    }

    #[test]
    fn test_cull_frustum_count() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([0.0, 0.0, 0.0], 1.0)).unwrap();
        buffer.add_instance(FoliageInstance::new([500.0, 0.0, 0.0], 1.0)).unwrap();
        buffer.add_instance(FoliageInstance::new([50.0, 0.0, 0.0], 1.0)).unwrap();

        let count = buffer.cull_frustum_count(&make_simple_frustum());
        assert_eq!(count, 2);
    }

    #[test]
    fn test_cull_frustum_empty_buffer() {
        let buffer = FoliageBuffer::new(FoliageConfig::default());
        let visible = buffer.cull_frustum(&make_simple_frustum());
        assert!(visible.is_empty());
    }

    // ========================================================================
    // LOD Selection Tests
    // ========================================================================

    #[test]
    fn test_select_lod_closest() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([5.0, 0.0, 0.0], 1.0)).unwrap();

        let thresholds = [100.0, 400.0, 900.0]; // Squared: 10, 20, 30
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods.len(), 1);
        assert_eq!(lods[0], 0); // Distance 5, squared 25 < 100 -> LOD 0
    }

    #[test]
    fn test_select_lod_mid_distance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([15.0, 0.0, 0.0], 1.0)).unwrap();

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods[0], 1); // Distance 15, squared 225 in range [100, 400) -> LOD 1
    }

    #[test]
    fn test_select_lod_far_distance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([50.0, 0.0, 0.0], 1.0)).unwrap();

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods[0], 3); // Distance 50, squared 2500 > all thresholds -> LOD 3
    }

    #[test]
    fn test_select_lod_with_bias_negative() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        // Distance 15, squared 225. With bias -1.0, biased = 225 * 0.5 = 112.5
        let instance = FoliageInstance::new([15.0, 0.0, 0.0], 1.0).with_lod_bias(-1.0);
        buffer.add_instance(instance).unwrap();

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        // 112.5 is in range [100, 400) -> LOD 1 (still LOD 1, but closer to boundary)
        assert_eq!(lods[0], 1);
    }

    #[test]
    fn test_select_lod_with_bias_positive() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        // Distance 8, squared 64. With bias 1.0, biased = 64 * 1.5 = 96
        let instance = FoliageInstance::new([8.0, 0.0, 0.0], 1.0).with_lod_bias(1.0);
        buffer.add_instance(instance).unwrap();

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods[0], 0); // 96 < 100 -> LOD 0
    }

    #[test]
    fn test_select_lod_multiple_instances() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([5.0, 0.0, 0.0], 1.0)).unwrap();  // sq=25
        buffer.add_instance(FoliageInstance::new([15.0, 0.0, 0.0], 1.0)).unwrap(); // sq=225
        buffer.add_instance(FoliageInstance::new([25.0, 0.0, 0.0], 1.0)).unwrap(); // sq=625
        buffer.add_instance(FoliageInstance::new([50.0, 0.0, 0.0], 1.0)).unwrap(); // sq=2500

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_select_lod_for_indices() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        for i in 0..10 {
            buffer.add_instance(FoliageInstance::new([i as f32 * 5.0, 0.0, 0.0], 1.0)).unwrap();
        }

        let thresholds = [100.0, 400.0, 900.0];
        let lods = buffer.select_lod_for_indices(&[0, 5, 9], [0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods.len(), 3);
    }

    #[test]
    fn test_select_lod_empty_thresholds() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([10.0, 0.0, 0.0], 1.0)).unwrap();

        let thresholds: [f32; 0] = [];
        let lods = buffer.select_lod([0.0, 0.0, 0.0], &thresholds);

        assert_eq!(lods[0], 0); // No thresholds -> all at LOD 0
    }

    // ========================================================================
    // Batch Generation Tests
    // ========================================================================

    #[test]
    fn test_generate_batches_empty() {
        let buffer = FoliageBuffer::new(FoliageConfig::default());
        let batches = buffer.generate_batches(&[], &[]);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_generate_batches_single_lod() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        for i in 0..5 {
            buffer.add_instance(FoliageInstance::new([i as f32, 0.0, 0.0], 1.0)).unwrap();
        }

        let visible: Vec<u32> = (0..5).collect();
        let lods = vec![0, 0, 0, 0, 0];
        let batches = buffer.generate_batches(&visible, &lods);

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].lod_level, 0);
        assert_eq!(batches[0].instance_count, 5);
    }

    #[test]
    fn test_generate_batches_multiple_lods() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        for i in 0..6 {
            buffer.add_instance(FoliageInstance::new([i as f32, 0.0, 0.0], 1.0)).unwrap();
        }

        let visible: Vec<u32> = (0..6).collect();
        let lods = vec![0, 0, 1, 1, 2, 2];
        let batches = buffer.generate_batches(&visible, &lods);

        assert_eq!(batches.len(), 3);

        // Verify each LOD batch
        let lod0 = batches.iter().find(|b| b.lod_level == 0).unwrap();
        assert_eq!(lod0.instance_count, 2);

        let lod1 = batches.iter().find(|b| b.lod_level == 1).unwrap();
        assert_eq!(lod1.instance_count, 2);

        let lod2 = batches.iter().find(|b| b.lod_level == 2).unwrap();
        assert_eq!(lod2.instance_count, 2);
    }

    #[test]
    fn test_generate_batches_lod_clamped() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::default()).unwrap();

        let visible = vec![0];
        let lods = vec![255]; // Out of range, should be clamped
        let batches = buffer.generate_batches(&visible, &lods);

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].lod_level, (MAX_LOD_LEVELS - 1) as u8);
    }

    // ========================================================================
    // Sort Tests
    // ========================================================================

    #[test]
    fn test_sort_by_distance() {
        let mut buffer = FoliageBuffer::new(FoliageConfig::default());
        buffer.add_instance(FoliageInstance::new([10.0, 0.0, 0.0], 1.0)).unwrap();
        buffer.add_instance(FoliageInstance::new([30.0, 0.0, 0.0], 1.0)).unwrap();
        buffer.add_instance(FoliageInstance::new([20.0, 0.0, 0.0], 1.0)).unwrap();

        buffer.sort_by_distance([0.0, 0.0, 0.0]);

        // Should be sorted back-to-front (furthest first)
        assert_eq!(buffer.get(0).unwrap().position[0], 30.0);
        assert_eq!(buffer.get(1).unwrap().position[0], 20.0);
        assert_eq!(buffer.get(2).unwrap().position[0], 10.0);
    }

    // ========================================================================
    // WindParameters Tests
    // ========================================================================

    #[test]
    fn test_wind_parameters_new() {
        let wind = WindParameters::new([1.0, 0.0], 2.0);
        assert!((wind.direction[0] - 1.0).abs() < 0.001);
        assert!((wind.direction[1] - 0.0).abs() < 0.001);
        assert!((wind.direction[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_wind_parameters_normalized_direction() {
        let wind = WindParameters::new([3.0, 4.0], 1.0);
        // 3-4-5 triangle, normalized should be 0.6, 0.8
        assert!((wind.direction[0] - 0.6).abs() < 0.001);
        assert!((wind.direction[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_wind_parameters_zero_direction() {
        let wind = WindParameters::new([0.0, 0.0], 1.0);
        // Should default to x-axis
        assert!((wind.direction[0] - 1.0).abs() < 0.001);
        assert!((wind.direction[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_wind_parameters_update() {
        let mut wind = WindParameters::new([1.0, 0.0], 1.0);
        assert_eq!(wind.time, 0.0);

        wind.update(0.016);
        assert!((wind.time - 0.016).abs() < 0.0001);
    }

    #[test]
    fn test_wind_parameters_calculate_offset() {
        let wind = WindParameters::new([1.0, 0.0], 1.0);
        let offset = wind.calculate_offset(0.0);
        // At time 0, phase 0, sin(0) = 0, so offset should be 0
        assert!(offset.abs() < 0.001);
    }

    // ========================================================================
    // FoliageLodThresholds Tests
    // ========================================================================

    #[test]
    fn test_lod_thresholds_from_distances() {
        let thresholds = FoliageLodThresholds::from_distances(&[10.0, 20.0, 30.0]).unwrap();
        let sq = thresholds.squared_thresholds();
        assert_eq!(sq.len(), 3);
        assert!((sq[0] - 100.0).abs() < 0.001);
        assert!((sq[1] - 400.0).abs() < 0.001);
        assert!((sq[2] - 900.0).abs() < 0.001);
    }

    #[test]
    fn test_lod_thresholds_empty_error() {
        let result = FoliageLodThresholds::from_distances(&[]);
        assert!(matches!(result, Err(FoliageError::InvalidLodThresholds { .. })));
    }

    #[test]
    fn test_lod_thresholds_non_increasing_error() {
        let result = FoliageLodThresholds::from_distances(&[20.0, 10.0, 30.0]);
        assert!(matches!(result, Err(FoliageError::InvalidLodThresholds { .. })));
    }

    #[test]
    fn test_lod_thresholds_equal_error() {
        let result = FoliageLodThresholds::from_distances(&[10.0, 10.0, 30.0]);
        assert!(matches!(result, Err(FoliageError::InvalidLodThresholds { .. })));
    }

    #[test]
    fn test_lod_thresholds_default_exponential() {
        let thresholds = FoliageLodThresholds::default_exponential(50.0, 4);
        let sq = thresholds.squared_thresholds();
        assert_eq!(sq.len(), 4);
        // 50^2, 100^2, 200^2, 400^2
        assert!((sq[0] - 2500.0).abs() < 0.1);
        assert!((sq[1] - 10000.0).abs() < 0.1);
    }

    #[test]
    fn test_lod_thresholds_select() {
        let thresholds = FoliageLodThresholds::from_distances(&[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(thresholds.select(50.0), 0);    // sqrt(50) ~ 7 < 10
        assert_eq!(thresholds.select(200.0), 1);   // sqrt(200) ~ 14 in [10, 20)
        assert_eq!(thresholds.select(500.0), 2);   // sqrt(500) ~ 22 in [20, 30)
        assert_eq!(thresholds.select(1000.0), 3);  // sqrt(1000) ~ 31 >= 30
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_normalize_quaternion() {
        let q = normalize_quaternion([0.0, 0.0, 0.0, 2.0]);
        assert!((q[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_quaternion_identity() {
        let q = normalize_quaternion([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(q, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_normalize_quaternion_degenerate() {
        let q = normalize_quaternion([0.0, 0.0, 0.0, 0.0]);
        assert_eq!(q, [0.0, 0.0, 0.0, 1.0]); // Falls back to identity
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        // 90 degree rotation around Y axis
        let q = quaternion_from_axis_angle([0.0, 1.0, 0.0], std::f32::consts::FRAC_PI_2);
        // Expected: (0, sin(45deg), 0, cos(45deg)) = (0, 0.707, 0, 0.707)
        assert!((q[1] - 0.707).abs() < 0.01);
        assert!((q[3] - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_quaternion_from_axis_angle_zero_axis() {
        let q = quaternion_from_axis_angle([0.0, 0.0, 0.0], 1.0);
        assert_eq!(q, [0.0, 0.0, 0.0, 1.0]); // Identity
    }

    #[test]
    fn test_quaternion_from_euler() {
        // Zero rotation should give identity
        let q = quaternion_from_euler(0.0, 0.0, 0.0);
        assert!((q[3] - 1.0).abs() < 0.001);
        assert!(q[0].abs() < 0.001);
        assert!(q[1].abs() < 0.001);
        assert!(q[2].abs() < 0.001);
    }

    // ========================================================================
    // FoliageBatch Tests
    // ========================================================================

    #[test]
    fn test_batch_is_empty() {
        let batch = FoliageBatch {
            lod_level: 0,
            instance_count: 0,
            indices: vec![],
        };
        assert!(batch.is_empty());

        let batch2 = FoliageBatch {
            lod_level: 0,
            instance_count: 5,
            indices: vec![0, 1, 2, 3, 4],
        };
        assert!(!batch2.is_empty());
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display_buffer_full() {
        let err = FoliageError::BufferFull { max: 1000 };
        assert!(err.to_string().contains("1000"));
    }

    #[test]
    fn test_error_display_invalid_instance() {
        let err = FoliageError::InvalidInstance { reason: "test reason" };
        assert!(err.to_string().contains("test reason"));
    }

    #[test]
    fn test_error_display_index_out_of_bounds() {
        let err = FoliageError::IndexOutOfBounds { index: 5, count: 3 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));
    }
}
