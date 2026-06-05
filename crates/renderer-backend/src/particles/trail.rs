//! Trail/Ribbon Rendering System for TRINITY Engine (T-GPU-6.3).
//!
//! Provides trail effects that follow particles or objects, creating ribbon-like
//! geometry that fades over time. Uses CPU ring buffer with GPU upload.
//!
//! # Overview
//!
//! Trail rendering pipeline:
//! 1. CPU manages trail points in a ring buffer (efficient add/remove)
//! 2. Points are updated each frame (aging, fade calculation)
//! 3. Active points are uploaded to GPU storage buffer
//! 4. Vertex shader expands points into camera-facing ribbon geometry
//! 5. Fragment shader applies texture and alpha blending
//!
//! # Features
//!
//! - **Ring Buffer**: O(1) point insertion with automatic oldest removal
//! - **Catmull-Rom Tangents**: Smooth curve interpolation
//! - **Camera-Facing Ribbons**: Geometry always faces viewer
//! - **Configurable Fade**: Age-based alpha from head to tail
//! - **UV Modes**: STRETCH (0-1 along length) or TILE (repeat)
//! - **Cap Styles**: ROUND, FLAT, ARROW (future extension)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::particles::trail::{
//!     TrailPipeline, TrailResources, TrailBuffer, TrailConfig,
//! };
//!
//! // Create pipeline
//! let pipeline = TrailPipeline::new(&device);
//!
//! // Create resources
//! let resources = TrailResources::new(&device, 256, &pipeline.bind_group_layout);
//!
//! // Create trail buffer
//! let mut buffer = TrailBuffer::new(TrailConfig::default());
//!
//! // Each frame: update trail
//! buffer.add_point(position, color, width_scale);
//! buffer.update(delta_time);
//! resources.upload(&queue, &buffer);
//!
//! // Render
//! pipeline.render(&mut pass, &resources, buffer.segment_count());
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum trail points per buffer.
pub const DEFAULT_MAX_TRAIL_POINTS: u32 = 256;

/// Minimum distance between trail points (world units).
pub const DEFAULT_MIN_POINT_DISTANCE: f32 = 0.05;

/// Default trail width (world units).
pub const DEFAULT_TRAIL_WIDTH: f32 = 0.5;

/// Default fade time (seconds).
pub const DEFAULT_FADE_TIME: f32 = 2.0;

/// TrailPoint struct size in bytes (must match WGSL).
pub const TRAIL_POINT_SIZE: usize = 64;

/// TrailParams struct size in bytes.
pub const TRAIL_PARAMS_SIZE: usize = 80;

// ---------------------------------------------------------------------------
// UV Mode
// ---------------------------------------------------------------------------

/// Trail texture UV mapping mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum UvMode {
    /// Stretch texture along entire trail length (u: 0 at head, 1 at tail).
    #[default]
    Stretch = 0,
    /// Tile texture based on trail length (u repeats).
    Tile = 1,
}

impl UvMode {
    /// Convert to u32 for GPU upload.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Cap Style
// ---------------------------------------------------------------------------

/// Trail cap/end style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CapStyle {
    /// No cap (flat ends).
    #[default]
    None = 0,
    /// Rounded semicircular cap.
    Round = 1,
    /// Flat cap (same as None for now).
    Flat = 2,
    /// Arrow-shaped cap pointing in trail direction.
    Arrow = 3,
}

impl CapStyle {
    /// Convert to u32 for GPU upload.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// TrailConfig
// ---------------------------------------------------------------------------

/// Configuration for trail rendering.
#[derive(Clone, Debug)]
pub struct TrailConfig {
    /// Maximum points in the trail buffer.
    pub max_points: u32,
    /// Minimum distance between consecutive points (world units).
    pub min_distance: f32,
    /// Base ribbon width (world units).
    pub ribbon_width: f32,
    /// Total fade time (seconds from creation to full transparency).
    pub fade_time: f32,
    /// Normalized age at which fade begins (0.0-1.0).
    pub fade_start: f32,
    /// UV texture mapping mode.
    pub uv_mode: UvMode,
    /// Tile repeat factor (for Tile UV mode).
    pub tile_factor: f32,
    /// Cap style for trail ends.
    pub cap_style: CapStyle,
}

impl Default for TrailConfig {
    fn default() -> Self {
        Self {
            max_points: DEFAULT_MAX_TRAIL_POINTS,
            min_distance: DEFAULT_MIN_POINT_DISTANCE,
            ribbon_width: DEFAULT_TRAIL_WIDTH,
            fade_time: DEFAULT_FADE_TIME,
            fade_start: 0.0,
            uv_mode: UvMode::Stretch,
            tile_factor: 1.0,
            cap_style: CapStyle::None,
        }
    }
}

impl TrailConfig {
    /// Create a new config with custom max points.
    pub fn with_max_points(mut self, max_points: u32) -> Self {
        self.max_points = max_points.max(2);
        self
    }

    /// Set the ribbon width.
    pub fn with_width(mut self, width: f32) -> Self {
        self.ribbon_width = width.max(0.001);
        self
    }

    /// Set the fade time.
    pub fn with_fade_time(mut self, time: f32) -> Self {
        self.fade_time = time.max(0.0);
        self
    }

    /// Set the UV mode.
    pub fn with_uv_mode(mut self, mode: UvMode) -> Self {
        self.uv_mode = mode;
        self
    }

    /// Set the tile factor (for Tile UV mode).
    pub fn with_tile_factor(mut self, factor: f32) -> Self {
        self.tile_factor = factor.max(0.01);
        self
    }

    /// Set the cap style.
    pub fn with_cap_style(mut self, style: CapStyle) -> Self {
        self.cap_style = style;
        self
    }

    /// Set the minimum point distance.
    pub fn with_min_distance(mut self, distance: f32) -> Self {
        self.min_distance = distance.max(0.001);
        self
    }
}

// ---------------------------------------------------------------------------
// TrailPoint
// ---------------------------------------------------------------------------

/// Single point in a trail (64 bytes, matches WGSL TrailPoint).
///
/// # Memory Layout
///
/// | Offset | Field              | Size     |
/// |--------|--------------------| ---------|
/// | 0      | position           | 12 bytes |
/// | 12     | age                | 4 bytes  |
/// | 16     | direction          | 12 bytes |
/// | 28     | width_scale        | 4 bytes  |
/// | 32     | color              | 16 bytes |
/// | 48     | distance_from_head | 4 bytes  |
/// | 52     | _padding           | 12 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrailPoint {
    /// World-space position.
    pub position: [f32; 3],
    /// Age of this point (seconds since creation).
    pub age: f32,
    /// Pre-computed tangent direction (normalized).
    pub direction: [f32; 3],
    /// Width scale factor (0-1, multiplied with ribbon_width).
    pub width_scale: f32,
    /// Color at this point (RGBA).
    pub color: [f32; 4],
    /// Distance from trail head to this point.
    pub distance_from_head: f32,
    /// Padding for alignment.
    pub _padding: [f32; 3],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<TrailPoint>() == TRAIL_POINT_SIZE);

impl Default for TrailPoint {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            direction: [0.0, 0.0, 1.0],
            width_scale: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            distance_from_head: 0.0,
            _padding: [0.0, 0.0, 0.0],
        }
    }
}

impl TrailPoint {
    /// Create a new trail point.
    pub fn new(position: [f32; 3], color: [f32; 4], width_scale: f32) -> Self {
        Self {
            position,
            age: 0.0,
            direction: [0.0, 0.0, 1.0],
            width_scale: width_scale.clamp(0.0, 1.0),
            color,
            distance_from_head: 0.0,
            _padding: [0.0, 0.0, 0.0],
        }
    }

    /// Check if point is still visible (alpha > threshold).
    #[inline]
    pub fn is_visible(&self, threshold: f32) -> bool {
        self.color[3] > threshold
    }

    /// Calculate distance to another point.
    #[inline]
    pub fn distance_to(&self, other: &TrailPoint) -> f32 {
        let dx = other.position[0] - self.position[0];
        let dy = other.position[1] - self.position[1];
        let dz = other.position[2] - self.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate squared distance to another point (faster, no sqrt).
    #[inline]
    pub fn distance_squared_to(&self, other: &TrailPoint) -> f32 {
        let dx = other.position[0] - self.position[0];
        let dy = other.position[1] - self.position[1];
        let dz = other.position[2] - self.position[2];
        dx * dx + dy * dy + dz * dz
    }
}

// ---------------------------------------------------------------------------
// TrailParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for trail rendering parameters (80 bytes).
///
/// # Memory Layout (std140 compatible)
///
/// | Offset | Field           | Size     |
/// |--------|-----------------|----------|
/// | 0      | view_proj       | 64 bytes |
/// | 64     | camera_position | 12 bytes |
/// | 76     | ribbon_width    | 4 bytes  |
/// | 80     | time            | 4 bytes  |
/// | 84     | fade_start      | 4 bytes  |
/// | 88     | fade_end        | 4 bytes  |
/// | 92     | uv_mode         | 4 bytes  |
/// | 96     | tile_factor     | 4 bytes  |
/// | 100    | total_length    | 4 bytes  |
/// | 104    | point_count     | 4 bytes  |
/// | 108    | cap_style       | 4 bytes  |
/// | 112    | _padding        | 16 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrailParams {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// Base ribbon width.
    pub ribbon_width: f32,
    /// Current time.
    pub time: f32,
    /// Normalized age at which fade begins.
    pub fade_start: f32,
    /// Normalized age at which fade completes.
    pub fade_end: f32,
    /// UV texture mode.
    pub uv_mode: u32,
    /// Tile repeat factor.
    pub tile_factor: f32,
    /// Total trail length.
    pub total_length: f32,
    /// Number of active points.
    pub point_count: u32,
    /// Cap style.
    pub cap_style: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<TrailParams>() == 112);

impl Default for TrailParams {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 5.0],
            ribbon_width: DEFAULT_TRAIL_WIDTH,
            time: 0.0,
            fade_start: 0.0,
            fade_end: 1.0,
            uv_mode: UvMode::Stretch as u32,
            tile_factor: 1.0,
            total_length: 0.0,
            point_count: 0,
            cap_style: CapStyle::None as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// TrailBuffer
// ---------------------------------------------------------------------------

/// Ring buffer for trail points.
///
/// Efficiently manages trail point history with O(1) add/remove operations.
/// Oldest points are automatically removed when buffer is full.
pub struct TrailBuffer {
    /// Configuration.
    config: TrailConfig,
    /// Ring buffer storage.
    points: Vec<TrailPoint>,
    /// Index of newest point (head).
    head: usize,
    /// Index of oldest point (tail).
    tail: usize,
    /// Number of active points.
    count: usize,
    /// Total trail length (sum of segment distances).
    total_length: f32,
    /// Whether trail is actively emitting new points.
    is_emitting: bool,
    /// Minimum distance squared (cached).
    min_distance_sq: f32,
}

impl TrailBuffer {
    /// Create a new trail buffer with the given configuration.
    pub fn new(config: TrailConfig) -> Self {
        let capacity = config.max_points as usize;
        let min_distance_sq = config.min_distance * config.min_distance;

        Self {
            config,
            points: vec![TrailPoint::default(); capacity],
            head: 0,
            tail: 0,
            count: 0,
            total_length: 0.0,
            is_emitting: true,
            min_distance_sq,
        }
    }

    /// Create with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(TrailConfig::default())
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &TrailConfig {
        &self.config
    }

    /// Get the number of active points.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.points.len()
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= self.capacity()
    }

    /// Get total trail length.
    #[inline]
    pub fn total_length(&self) -> f32 {
        self.total_length
    }

    /// Get number of segments (for rendering).
    #[inline]
    pub fn segment_count(&self) -> u32 {
        if self.count < 2 {
            0
        } else {
            (self.count - 1) as u32
        }
    }

    /// Check if trail is emitting.
    #[inline]
    pub fn is_emitting(&self) -> bool {
        self.is_emitting
    }

    /// Start emitting new points.
    pub fn start_emitting(&mut self) {
        self.is_emitting = true;
    }

    /// Stop emitting new points (trail will fade out).
    pub fn stop_emitting(&mut self) {
        self.is_emitting = false;
    }

    /// Clear all points.
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
        self.total_length = 0.0;
    }

    /// Add a new point to the trail head.
    ///
    /// Returns true if point was added (meets minimum distance requirement).
    pub fn add_point(&mut self, position: [f32; 3], color: [f32; 4], width_scale: f32) -> bool {
        if !self.is_emitting {
            return false;
        }

        // Check minimum distance from current head
        if self.count > 0 {
            let head_point = &self.points[self.head];
            let dx = position[0] - head_point.position[0];
            let dy = position[1] - head_point.position[1];
            let dz = position[2] - head_point.position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < self.min_distance_sq {
                return false;
            }
        }

        // Advance head
        let new_head = if self.count > 0 {
            (self.head + 1) % self.capacity()
        } else {
            self.head
        };

        // Remove oldest if full
        if self.count >= self.capacity() {
            self.tail = (self.tail + 1) % self.capacity();
        } else {
            self.count += 1;
        }

        self.head = new_head;

        // Store new point
        self.points[self.head] = TrailPoint::new(position, color, width_scale);

        // Recalculate distances and tangents
        self.recalculate_distances();
        self.recalculate_tangents();

        true
    }

    /// Update all points (aging, fade, remove dead points).
    ///
    /// Returns number of points removed.
    pub fn update(&mut self, delta_time: f32) -> usize {
        if self.count == 0 {
            return 0;
        }

        let fade_time = self.config.fade_time;
        let mut removed = 0;

        // Update ages for all points
        for i in 0..self.count {
            let idx = (self.tail + i) % self.capacity();
            self.points[idx].age += delta_time;
        }

        // Remove points that have fully faded (from tail)
        while self.count > 0 {
            let tail_point = &self.points[self.tail];
            let normalized_age = if fade_time > 0.0 {
                tail_point.age / fade_time
            } else {
                1.0
            };

            if normalized_age >= 1.0 {
                self.tail = (self.tail + 1) % self.capacity();
                self.count -= 1;
                removed += 1;
            } else {
                break;
            }
        }

        // Recalculate if points were removed
        if removed > 0 {
            self.recalculate_distances();
            self.recalculate_tangents();
        }

        removed
    }

    /// Get point at index (0 = oldest/tail, count-1 = newest/head).
    pub fn get_point(&self, index: usize) -> Option<&TrailPoint> {
        if index >= self.count {
            return None;
        }

        let buffer_index = (self.tail + index) % self.capacity();
        Some(&self.points[buffer_index])
    }

    /// Get mutable point at index.
    pub fn get_point_mut(&mut self, index: usize) -> Option<&mut TrailPoint> {
        if index >= self.count {
            return None;
        }

        let buffer_index = (self.tail + index) % self.capacity();
        Some(&mut self.points[buffer_index])
    }

    /// Get the newest (head) point.
    pub fn get_newest(&self) -> Option<&TrailPoint> {
        if self.count == 0 {
            None
        } else {
            Some(&self.points[self.head])
        }
    }

    /// Get the oldest (tail) point.
    pub fn get_oldest(&self) -> Option<&TrailPoint> {
        if self.count == 0 {
            None
        } else {
            Some(&self.points[self.tail])
        }
    }

    /// Iterate over points from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &TrailPoint> + '_ {
        (0..self.count).map(move |i| {
            let idx = (self.tail + i) % self.capacity();
            &self.points[idx]
        })
    }

    /// Export points to a contiguous slice for GPU upload.
    pub fn export_points(&self, output: &mut [TrailPoint]) {
        let count = self.count.min(output.len());

        for i in 0..count {
            let idx = (self.tail + i) % self.capacity();
            output[i] = self.points[idx];
        }
    }

    /// Recalculate segment distances and total length.
    fn recalculate_distances(&mut self) {
        if self.count < 2 {
            self.total_length = 0.0;
            if self.count == 1 {
                self.points[self.head].distance_from_head = 0.0;
            }
            return;
        }

        // Iterate from head (newest, distance=0) to tail (oldest, distance=total)
        let mut cumulative_distance = 0.0;

        // Set head distance to 0
        self.points[self.head].distance_from_head = 0.0;

        // Walk backwards from head to tail
        let mut prev_idx = self.head;

        for i in 1..self.count {
            // Index going from head towards tail
            let reverse_i = self.count - 1 - i;
            let curr_idx = (self.tail + reverse_i) % self.capacity();

            let dx = self.points[curr_idx].position[0] - self.points[prev_idx].position[0];
            let dy = self.points[curr_idx].position[1] - self.points[prev_idx].position[1];
            let dz = self.points[curr_idx].position[2] - self.points[prev_idx].position[2];
            let segment_length = (dx * dx + dy * dy + dz * dz).sqrt();

            cumulative_distance += segment_length;
            self.points[curr_idx].distance_from_head = cumulative_distance;

            prev_idx = curr_idx;
        }

        self.total_length = cumulative_distance;
    }

    /// Recalculate tangent directions using Catmull-Rom.
    fn recalculate_tangents(&mut self) {
        if self.count == 0 {
            return;
        }

        for i in 0..self.count {
            let curr_idx = (self.tail + i) % self.capacity();

            let tangent = if self.count == 1 {
                // Single point: arbitrary direction
                [0.0, 0.0, 1.0]
            } else if i == 0 {
                // First point (tail): forward difference
                let next_idx = (self.tail + 1) % self.capacity();
                let dx = self.points[next_idx].position[0] - self.points[curr_idx].position[0];
                let dy = self.points[next_idx].position[1] - self.points[curr_idx].position[1];
                let dz = self.points[next_idx].position[2] - self.points[curr_idx].position[2];
                normalize_vec3([dx, dy, dz])
            } else if i == self.count - 1 {
                // Last point (head): backward difference
                let prev_idx = (self.tail + i - 1) % self.capacity();
                let dx = self.points[curr_idx].position[0] - self.points[prev_idx].position[0];
                let dy = self.points[curr_idx].position[1] - self.points[prev_idx].position[1];
                let dz = self.points[curr_idx].position[2] - self.points[prev_idx].position[2];
                normalize_vec3([dx, dy, dz])
            } else {
                // Middle points: central difference (Catmull-Rom)
                let prev_idx = (self.tail + i - 1) % self.capacity();
                let next_idx = (self.tail + i + 1) % self.capacity();
                let dx = (self.points[next_idx].position[0] - self.points[prev_idx].position[0]) * 0.5;
                let dy = (self.points[next_idx].position[1] - self.points[prev_idx].position[1]) * 0.5;
                let dz = (self.points[next_idx].position[2] - self.points[prev_idx].position[2]) * 0.5;
                normalize_vec3([dx, dy, dz])
            };

            self.points[curr_idx].direction = tangent;
        }
    }
}

/// Normalize a 3D vector. Returns [0,0,1] if input is zero-length.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

// ---------------------------------------------------------------------------
// TrailResources
// ---------------------------------------------------------------------------

/// GPU resources for trail rendering.
pub struct TrailResources {
    /// Uniform buffer for TrailParams.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for trail points.
    pub points_buffer: wgpu::Buffer,
    /// Maximum capacity in points.
    pub capacity: u32,
    /// Bind group for trail shader.
    pub bind_group: wgpu::BindGroup,
}

impl TrailResources {
    /// Create trail resources with the given capacity.
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail_params"),
            size: mem::size_of::<TrailParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail_points"),
            size: (capacity as u64) * (TRAIL_POINT_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trail_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params_buffer,
            points_buffer,
            capacity,
            bind_group,
        }
    }

    /// Update params buffer.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &TrailParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload trail points from buffer.
    pub fn upload(&self, queue: &wgpu::Queue, buffer: &TrailBuffer, params: &TrailParams) {
        // Update params
        self.update_params(queue, params);

        // Upload points
        if buffer.count() > 0 {
            let mut points = vec![TrailPoint::default(); buffer.count()];
            buffer.export_points(&mut points);
            queue.write_buffer(&self.points_buffer, 0, bytemuck::cast_slice(&points));
        }
    }
}

// ---------------------------------------------------------------------------
// TrailPipeline
// ---------------------------------------------------------------------------

/// GPU render pipeline for trail rendering.
pub struct TrailPipeline {
    /// Bind group layout for trail shader.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for texture.
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Render pipeline for textured trails.
    pub pipeline: wgpu::RenderPipeline,
    /// Render pipeline for solid-color trails.
    pub pipeline_solid: wgpu::RenderPipeline,
    /// Render pipeline for additive blending.
    pub pipeline_additive: wgpu::RenderPipeline,
}

impl TrailPipeline {
    /// Create the trail render pipeline.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Create bind group layout for params and points
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trail_bind_group_layout"),
            entries: &[
                // binding 0: TrailParams (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: trail_points (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group layout for texture
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("trail_texture_bind_group_layout"),
                entries: &[
                    // binding 0: trail_texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: trail_sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trail_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader modules
        let vert_source = include_str!("../../shaders/particles/trail.vert.wgsl");
        let frag_source = include_str!("../../shaders/particles/trail.frag.wgsl");

        let vert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trail_vert_shader"),
            source: wgpu::ShaderSource::Wgsl(vert_source.into()),
        });

        let frag_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trail_frag_shader"),
            source: wgpu::ShaderSource::Wgsl(frag_source.into()),
        });

        // Alpha blend state (premultiplied alpha)
        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Additive blend state
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Create textured pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trail_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_module,
                entry_point: "vs_trail",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_module,
                entry_point: "fs_trail",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(alpha_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for double-sided ribbons
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Transparent, no depth write
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create solid pipeline (no texture bind group needed in practice, but same layout)
        let pipeline_solid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trail_pipeline_solid"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_module,
                entry_point: "vs_trail",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_module,
                entry_point: "fs_trail_solid",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(alpha_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create additive pipeline
        let pipeline_additive = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trail_pipeline_additive"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_module,
                entry_point: "vs_trail",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_module,
                entry_point: "fs_trail_additive",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            bind_group_layout,
            texture_bind_group_layout,
            pipeline,
            pipeline_solid,
            pipeline_additive,
        }
    }

    /// Render a trail.
    ///
    /// # Arguments
    ///
    /// * `pass` - Render pass to record commands.
    /// * `resources` - Trail resources with bind group.
    /// * `texture_bind_group` - Bind group for trail texture.
    /// * `segment_count` - Number of trail segments to render.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a TrailResources,
        texture_bind_group: &'a wgpu::BindGroup,
        segment_count: u32,
    ) {
        if segment_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);

        // Each segment is 4 vertices (triangle strip)
        let vertex_count = segment_count * 4;
        pass.draw(0..vertex_count, 0..1);
    }

    /// Render a solid-color trail (no texture).
    pub fn render_solid<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a TrailResources,
        texture_bind_group: &'a wgpu::BindGroup,
        segment_count: u32,
    ) {
        if segment_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline_solid);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);

        let vertex_count = segment_count * 4;
        pass.draw(0..vertex_count, 0..1);
    }

    /// Render an additive-blended trail (for glow effects).
    pub fn render_additive<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a TrailResources,
        texture_bind_group: &'a wgpu::BindGroup,
        segment_count: u32,
    ) {
        if segment_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline_additive);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);

        let vertex_count = segment_count * 4;
        pass.draw(0..vertex_count, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── TrailConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = TrailConfig::default();
        assert_eq!(config.max_points, DEFAULT_MAX_TRAIL_POINTS);
        assert!((config.ribbon_width - DEFAULT_TRAIL_WIDTH).abs() < f32::EPSILON);
        assert!((config.fade_time - DEFAULT_FADE_TIME).abs() < f32::EPSILON);
        assert_eq!(config.uv_mode, UvMode::Stretch);
        assert_eq!(config.cap_style, CapStyle::None);
    }

    #[test]
    fn test_config_builder() {
        let config = TrailConfig::default()
            .with_max_points(128)
            .with_width(1.0)
            .with_fade_time(5.0)
            .with_uv_mode(UvMode::Tile)
            .with_tile_factor(2.0)
            .with_cap_style(CapStyle::Round);

        assert_eq!(config.max_points, 128);
        assert!((config.ribbon_width - 1.0).abs() < f32::EPSILON);
        assert!((config.fade_time - 5.0).abs() < f32::EPSILON);
        assert_eq!(config.uv_mode, UvMode::Tile);
        assert!((config.tile_factor - 2.0).abs() < f32::EPSILON);
        assert_eq!(config.cap_style, CapStyle::Round);
    }

    // ── TrailPoint ──────────────────────────────────────────────────────────

    #[test]
    fn test_point_default() {
        let point = TrailPoint::default();
        assert_eq!(point.position, [0.0, 0.0, 0.0]);
        assert!((point.age - 0.0).abs() < f32::EPSILON);
        assert!((point.width_scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(point.color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_point_new() {
        let point = TrailPoint::new([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 1.0], 0.5);
        assert_eq!(point.position, [1.0, 2.0, 3.0]);
        assert!((point.width_scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(point.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_point_distance() {
        let p1 = TrailPoint::new([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        let p2 = TrailPoint::new([3.0, 4.0, 0.0], [1.0; 4], 1.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.0001);
        assert!((p1.distance_squared_to(&p2) - 25.0).abs() < 0.0001);
    }

    #[test]
    fn test_point_is_visible() {
        let mut point = TrailPoint::default();
        assert!(point.is_visible(0.01));

        point.color[3] = 0.0;
        assert!(!point.is_visible(0.01));
    }

    // ── TrailBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_buffer_new() {
        let buffer = TrailBuffer::with_default_config();
        assert!(buffer.is_empty());
        assert_eq!(buffer.count(), 0);
        assert_eq!(buffer.capacity(), DEFAULT_MAX_TRAIL_POINTS as usize);
        assert!(buffer.is_emitting());
    }

    #[test]
    fn test_buffer_add_single_point() {
        let mut buffer = TrailBuffer::with_default_config();
        let added = buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        assert!(added);
        assert_eq!(buffer.count(), 1);
        assert_eq!(buffer.segment_count(), 0); // Need 2 points for 1 segment
    }

    #[test]
    fn test_buffer_add_multiple_points() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);

        assert_eq!(buffer.count(), 3);
        assert_eq!(buffer.segment_count(), 2);
    }

    #[test]
    fn test_buffer_min_distance_enforcement() {
        let config = TrailConfig::default().with_min_distance(1.0);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        let added = buffer.add_point([0.1, 0.0, 0.0], [1.0; 4], 1.0);
        assert!(!added); // Too close

        let added = buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);
        assert!(added); // Far enough

        assert_eq!(buffer.count(), 2);
    }

    #[test]
    fn test_buffer_ring_wrap() {
        let config = TrailConfig::default().with_max_points(4).with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        // Fill buffer
        for i in 0..4 {
            buffer.add_point([i as f32, 0.0, 0.0], [1.0; 4], 1.0);
        }
        assert_eq!(buffer.count(), 4);

        // Add one more, should wrap
        buffer.add_point([10.0, 0.0, 0.0], [1.0; 4], 1.0);
        assert_eq!(buffer.count(), 4); // Still 4

        // Oldest should be gone, newest should be [10.0, 0, 0]
        let newest = buffer.get_newest().unwrap();
        assert_eq!(newest.position, [10.0, 0.0, 0.0]);

        // Oldest should now be [1.0, 0, 0] (index 1 originally)
        let oldest = buffer.get_oldest().unwrap();
        assert_eq!(oldest.position, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_buffer_update_aging() {
        let config = TrailConfig::default().with_fade_time(1.0).with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);

        // Update by 0.5 seconds
        buffer.update(0.5);

        let oldest = buffer.get_oldest().unwrap();
        assert!((oldest.age - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_buffer_update_removal() {
        let config = TrailConfig::default().with_fade_time(1.0).with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);

        assert_eq!(buffer.count(), 3);

        // Update past fade time - oldest should be removed
        let removed = buffer.update(1.5);
        assert_eq!(removed, 3); // All points faded
        assert_eq!(buffer.count(), 0);
    }

    #[test]
    fn test_buffer_stop_emitting() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.stop_emitting();

        let added = buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        assert!(!added); // Not emitting
        assert_eq!(buffer.count(), 1);
    }

    #[test]
    fn test_buffer_clear() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.count(), 0);
        assert!((buffer.total_length() - 0.0).abs() < f32::EPSILON);
    }

    // ── Direction/Tangent Calculation ───────────────────────────────────────

    #[test]
    fn test_buffer_tangent_calculation() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);

        // Middle point should have tangent roughly [1, 0, 0] (Catmull-Rom)
        let middle = buffer.get_point(1).unwrap();
        assert!((middle.direction[0] - 1.0).abs() < 0.1);
        assert!(middle.direction[1].abs() < 0.1);
        assert!(middle.direction[2].abs() < 0.1);
    }

    // ── Width Scaling ───────────────────────────────────────────────────────

    #[test]
    fn test_point_width_clamping() {
        let point = TrailPoint::new([0.0, 0.0, 0.0], [1.0; 4], 1.5);
        assert!((point.width_scale - 1.0).abs() < f32::EPSILON); // Clamped to 1.0

        let point = TrailPoint::new([0.0, 0.0, 0.0], [1.0; 4], -0.5);
        assert!((point.width_scale - 0.0).abs() < f32::EPSILON); // Clamped to 0.0
    }

    // ── Fade Calculation ────────────────────────────────────────────────────

    #[test]
    fn test_buffer_fade_over_age() {
        let config = TrailConfig::default().with_fade_time(2.0).with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);

        // At t=0, age=0
        let point = buffer.get_point(0).unwrap();
        assert!((point.age - 0.0).abs() < f32::EPSILON);

        // Update by 1 second (half fade time)
        buffer.update(1.0);
        let point = buffer.get_point(0).unwrap();
        assert!((point.age - 1.0).abs() < f32::EPSILON);
    }

    // ── UV Modes ────────────────────────────────────────────────────────────

    #[test]
    fn test_uv_mode_values() {
        assert_eq!(UvMode::Stretch.as_u32(), 0);
        assert_eq!(UvMode::Tile.as_u32(), 1);
    }

    // ── Distance Calculation ────────────────────────────────────────────────

    #[test]
    fn test_buffer_total_length() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([3.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([3.0, 4.0, 0.0], [1.0; 4], 1.0);

        // Total length should be 3 + 4 = 7
        assert!((buffer.total_length() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_distance_from_head() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);

        // Newest (head) should have distance 0
        let newest = buffer.get_newest().unwrap();
        assert!((newest.distance_from_head - 0.0).abs() < f32::EPSILON);

        // Oldest should have distance = total length
        let oldest = buffer.get_oldest().unwrap();
        assert!((oldest.distance_from_head - buffer.total_length()).abs() < 0.001);
    }

    // ── Camera-Facing Ribbon ────────────────────────────────────────────────

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 0.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 0.001);
        assert!(v[1].abs() < 0.001);
        assert!((v[2] - 0.8).abs() < 0.001);

        // Zero vector should return default
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 1.0]);
    }

    // ── TrailParams ─────────────────────────────────────────────────────────

    #[test]
    fn test_params_default() {
        let params = TrailParams::default();
        assert_eq!(params.point_count, 0);
        assert!((params.ribbon_width - DEFAULT_TRAIL_WIDTH).abs() < f32::EPSILON);
        assert_eq!(params.uv_mode, UvMode::Stretch as u32);
    }

    // ── Export Points ───────────────────────────────────────────────────────

    #[test]
    fn test_buffer_export() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 0.8);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 0.6);

        let mut output = vec![TrailPoint::default(); 3];
        buffer.export_points(&mut output);

        // Points should be in order from oldest to newest
        assert_eq!(output[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(output[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(output[2].position, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_buffer_iter() {
        let config = TrailConfig::default().with_min_distance(0.01);
        let mut buffer = TrailBuffer::new(config);

        buffer.add_point([0.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([1.0, 0.0, 0.0], [1.0; 4], 1.0);
        buffer.add_point([2.0, 0.0, 0.0], [1.0; 4], 1.0);

        let positions: Vec<[f32; 3]> = buffer.iter().map(|p| p.position).collect();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(positions[1], [1.0, 0.0, 0.0]);
        assert_eq!(positions[2], [2.0, 0.0, 0.0]);
    }
}
