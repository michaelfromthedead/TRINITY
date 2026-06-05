//! Deferred Decal System for TRINITY Engine (T-GPU-6.4).
//!
//! Projects textures onto geometry within bounding volumes using deferred
//! rendering. Supports bullet holes, blood splatters, dirt, graffiti, etc.
//!
//! # Architecture
//!
//! The decal system uses a box projection technique:
//!
//! 1. **Decal Volume** - Oriented bounding box defines projection region
//! 2. **Depth Reconstruction** - World position from GBuffer depth
//! 3. **Volume Test** - Reject fragments outside decal box
//! 4. **Projection** - Sample decal texture using projected UVs
//! 5. **Blending** - Apply to GBuffer with per-channel blend modes
//!
//! # Blend Modes
//!
//! - `Albedo` - Modifies diffuse color only
//! - `Normal` - Modifies surface normals only
//! - `Both` - Modifies both albedo and normals
//! - `Emissive` - Additive emissive contribution
//!
//! # Atlas Packing
//!
//! Decal textures are packed into atlases for efficient batching.
//! Each decal stores its atlas rectangle for UV transformation.
//!
//! # Usage Example
//!
//! ```ignore
//! use renderer_backend::decals::{
//!     BlendMode, DecalAtlas, DecalInstance, DecalParams,
//!     DecalPipeline, DecalResources,
//! };
//!
//! // Create pipeline
//! let pipeline = DecalPipeline::new(&device);
//!
//! // Create atlas and pack textures
//! let mut atlas = DecalAtlas::new(&device, 2048, 2048);
//! let bullet_rect = atlas.pack_texture(&bullet_hole_image)?;
//! let blood_rect = atlas.pack_texture(&blood_splatter_image)?;
//!
//! // Create resources
//! let resources = DecalResources::new(&device, 1024, &pipeline.bind_group_layout);
//!
//! // Add decals
//! let bullet_decal = DecalInstance::new(transform)
//!     .with_atlas_rect(bullet_rect)
//!     .with_blend_mode(BlendMode::Both)
//!     .with_color([1.0, 1.0, 1.0, 1.0])
//!     .with_fade(1.0);
//!
//! resources.add_decal(&bullet_decal);
//!
//! // Each frame
//! resources.upload(&queue);
//! pipeline.render(&mut render_pass, &resources, &gbuffer);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum decals per frame (buffer capacity).
pub const DEFAULT_MAX_DECALS: u32 = 4096;

/// DecalParams uniform size in bytes.
pub const DECAL_PARAMS_SIZE: usize = 160; // 2 * mat4 + vec4

/// DecalInstance size in bytes.
pub const DECAL_INSTANCE_SIZE: usize = 176; // 2 * mat4 + 4 * vec4

/// Default atlas dimensions.
pub const DEFAULT_ATLAS_SIZE: u32 = 2048;

/// Minimum decal size to avoid degenerate volumes.
pub const MIN_DECAL_SIZE: f32 = 0.01;

/// Workgroup size for decal culling compute shader (if used).
pub const WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// BlendMode
// ---------------------------------------------------------------------------

/// Decal blend mode determining which GBuffer channels are modified.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Modify albedo/diffuse color only.
    #[default]
    Albedo = 0,
    /// Modify surface normals only.
    Normal = 1,
    /// Modify both albedo and normals.
    Both = 2,
    /// Additive emissive contribution.
    Emissive = 3,
}

impl BlendMode {
    /// Returns true if this mode affects albedo.
    #[inline]
    pub fn affects_albedo(&self) -> bool {
        matches!(self, BlendMode::Albedo | BlendMode::Both | BlendMode::Emissive)
    }

    /// Returns true if this mode affects normals.
    #[inline]
    pub fn affects_normal(&self) -> bool {
        matches!(self, BlendMode::Normal | BlendMode::Both)
    }

    /// Create from raw u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => BlendMode::Albedo,
            1 => BlendMode::Normal,
            2 => BlendMode::Both,
            3 => BlendMode::Emissive,
            _ => BlendMode::Albedo,
        }
    }
}

// ---------------------------------------------------------------------------
// DecalParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for global decal rendering parameters.
///
/// Matches the WGSL `DecalParams` struct layout.
///
/// # Memory Layout (160 bytes, std140 compatible)
///
/// | Offset | Field           | Size     |
/// |--------|-----------------|----------|
/// | 0      | view_proj       | 64 bytes |
/// | 64     | inv_view_proj   | 64 bytes |
/// | 128    | camera_position | 12 bytes |
/// | 140    | _pad            | 4 bytes  |
/// | 144    | _padding        | 16 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DecalParams {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Inverse view-projection for depth reconstruction.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera world position.
    pub camera_position: [f32; 3],
    /// Padding.
    pub _pad: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DecalParams>() == 144);

impl DecalParams {
    /// Create new decal parameters.
    ///
    /// # Arguments
    ///
    /// * `view_proj` - Combined view-projection matrix (column-major).
    /// * `inv_view_proj` - Inverse view-projection matrix.
    /// * `camera_position` - World position of camera.
    pub fn new(
        view_proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        camera_position: [f32; 3],
    ) -> Self {
        Self {
            view_proj,
            inv_view_proj,
            camera_position,
            _pad: 0.0,
        }
    }

    /// Create identity parameters (for testing).
    pub fn identity() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 0.0],
            _pad: 0.0,
        }
    }
}

impl Default for DecalParams {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// DecalInstance
// ---------------------------------------------------------------------------

/// Per-decal instance data for GPU buffer.
///
/// Matches the WGSL `DecalInstance` struct layout.
///
/// # Memory Layout (176 bytes, std140 compatible)
///
/// | Offset | Field           | Size     |
/// |--------|-----------------|----------|
/// | 0      | world_to_decal  | 64 bytes |
/// | 64     | decal_to_world  | 64 bytes |
/// | 128    | color           | 16 bytes |
/// | 144    | atlas_rect      | 16 bytes |
/// | 160    | blend_mode      | 4 bytes  |
/// | 164    | normal_strength | 4 bytes  |
/// | 168    | fade            | 4 bytes  |
/// | 172    | _pad            | 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DecalInstance {
    /// Transform from world space to decal local space (unit cube).
    pub world_to_decal: [[f32; 4]; 4],
    /// Transform from decal local space to world space.
    pub decal_to_world: [[f32; 4]; 4],
    /// Decal tint color (RGBA).
    pub color: [f32; 4],
    /// Atlas rectangle: xy = offset, zw = size (normalized 0-1).
    pub atlas_rect: [f32; 4],
    /// Blend mode (0=Albedo, 1=Normal, 2=Both, 3=Emissive).
    pub blend_mode: u32,
    /// Normal map blend strength.
    pub normal_strength: f32,
    /// Fade factor (0.0 = invisible, 1.0 = full).
    pub fade: f32,
    /// Padding.
    pub _pad: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DecalInstance>() == DECAL_INSTANCE_SIZE);

impl DecalInstance {
    /// Create a new decal instance with default parameters.
    ///
    /// # Arguments
    ///
    /// * `decal_to_world` - Transform from decal space to world space.
    pub fn new(decal_to_world: [[f32; 4]; 4]) -> Self {
        let world_to_decal = cpu_invert_matrix(decal_to_world);

        Self {
            world_to_decal,
            decal_to_world,
            color: [1.0, 1.0, 1.0, 1.0],
            atlas_rect: [0.0, 0.0, 1.0, 1.0],
            blend_mode: BlendMode::Albedo as u32,
            normal_strength: 1.0,
            fade: 1.0,
            _pad: 0.0,
        }
    }

    /// Create from position, rotation, and scale.
    ///
    /// # Arguments
    ///
    /// * `position` - World position of decal center.
    /// * `rotation` - Rotation quaternion [x, y, z, w].
    /// * `scale` - Size of decal volume [width, height, depth].
    pub fn from_transform(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        let decal_to_world = cpu_build_transform(position, rotation, scale);
        Self::new(decal_to_world)
    }

    /// Create an axis-aligned decal (no rotation).
    pub fn axis_aligned(position: [f32; 3], size: [f32; 3]) -> Self {
        Self::from_transform(position, [0.0, 0.0, 0.0, 1.0], size)
    }

    /// Set the tint color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set the atlas rectangle.
    pub fn with_atlas_rect(mut self, rect: [f32; 4]) -> Self {
        self.atlas_rect = rect;
        self
    }

    /// Set the blend mode.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode as u32;
        self
    }

    /// Set the normal strength.
    pub fn with_normal_strength(mut self, strength: f32) -> Self {
        self.normal_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the fade factor.
    pub fn with_fade(mut self, fade: f32) -> Self {
        self.fade = fade.clamp(0.0, 1.0);
        self
    }

    /// Get the blend mode as enum.
    pub fn blend_mode_enum(&self) -> BlendMode {
        BlendMode::from_u32(self.blend_mode)
    }

    /// Check if the decal is visible (non-zero fade).
    pub fn is_visible(&self) -> bool {
        self.fade > 0.0 && self.color[3] > 0.0
    }

    /// Get the world-space center of the decal.
    pub fn center(&self) -> [f32; 3] {
        [
            self.decal_to_world[3][0],
            self.decal_to_world[3][1],
            self.decal_to_world[3][2],
        ]
    }

    /// Test if a world-space point is inside the decal volume.
    pub fn contains_point(&self, world_point: [f32; 3]) -> bool {
        let local = cpu_transform_point(self.world_to_decal, world_point);
        local[0].abs() <= 0.5 && local[1].abs() <= 0.5 && local[2].abs() <= 0.5
    }

    /// Project a world-space point to decal UV coordinates.
    /// Returns None if the point is outside the volume.
    pub fn project_point(&self, world_point: [f32; 3]) -> Option<[f32; 2]> {
        let local = cpu_transform_point(self.world_to_decal, world_point);

        if local[0].abs() > 0.5 || local[1].abs() > 0.5 || local[2].abs() > 0.5 {
            return None;
        }

        // Convert from [-0.5, 0.5] to [0, 1]
        let base_uv = [local[0] + 0.5, local[1] + 0.5];

        // Apply atlas rect
        let atlas_uv = [
            self.atlas_rect[0] + base_uv[0] * self.atlas_rect[2],
            self.atlas_rect[1] + base_uv[1] * self.atlas_rect[3],
        ];

        Some(atlas_uv)
    }
}

impl Default for DecalInstance {
    fn default() -> Self {
        Self::axis_aligned([0.0, 0.0, 0.0], [1.0, 1.0, 0.1])
    }
}

// ---------------------------------------------------------------------------
// AtlasRect
// ---------------------------------------------------------------------------

/// Rectangle within a texture atlas.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasRect {
    /// X offset in normalized coordinates [0, 1].
    pub x: f32,
    /// Y offset in normalized coordinates [0, 1].
    pub y: f32,
    /// Width in normalized coordinates [0, 1].
    pub width: f32,
    /// Height in normalized coordinates [0, 1].
    pub height: f32,
}

impl AtlasRect {
    /// Create a new atlas rectangle.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Create from pixel coordinates and atlas dimensions.
    pub fn from_pixels(x: u32, y: u32, width: u32, height: u32, atlas_width: u32, atlas_height: u32) -> Self {
        Self {
            x: x as f32 / atlas_width as f32,
            y: y as f32 / atlas_height as f32,
            width: width as f32 / atlas_width as f32,
            height: height as f32 / atlas_height as f32,
        }
    }

    /// Convert to array format for GPU.
    pub fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Full atlas (no sub-region).
    pub fn full() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// DecalAtlas
// ---------------------------------------------------------------------------

/// Simple bin-packing atlas for decal textures.
///
/// Uses a shelf-based packing algorithm for simplicity.
/// For production, consider using rectpack or similar.
#[derive(Debug)]
pub struct DecalAtlas {
    /// Atlas width in pixels.
    pub width: u32,
    /// Atlas height in pixels.
    pub height: u32,
    /// Current shelf Y position.
    current_y: u32,
    /// Current X position within shelf.
    current_x: u32,
    /// Current shelf height.
    shelf_height: u32,
    /// Allocated rectangles.
    allocations: Vec<AtlasRect>,
}

impl DecalAtlas {
    /// Create a new decal atlas.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            current_y: 0,
            current_x: 0,
            shelf_height: 0,
            allocations: Vec::new(),
        }
    }

    /// Allocate space for a texture in the atlas.
    ///
    /// Returns the atlas rectangle or None if no space available.
    pub fn allocate(&mut self, tex_width: u32, tex_height: u32) -> Option<AtlasRect> {
        // Check if texture fits on current shelf
        if self.current_x + tex_width > self.width {
            // Move to next shelf
            self.current_y += self.shelf_height;
            self.current_x = 0;
            self.shelf_height = 0;
        }

        // Check if texture fits vertically
        if self.current_y + tex_height > self.height {
            return None;
        }

        // Allocate
        let rect = AtlasRect::from_pixels(
            self.current_x,
            self.current_y,
            tex_width,
            tex_height,
            self.width,
            self.height,
        );

        self.current_x += tex_width;
        self.shelf_height = self.shelf_height.max(tex_height);
        self.allocations.push(rect);

        Some(rect)
    }

    /// Get number of allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get utilization ratio [0, 1].
    pub fn utilization(&self) -> f32 {
        let used_area: f32 = self.allocations.iter()
            .map(|r| r.width * r.height)
            .sum();
        used_area
    }

    /// Reset the atlas (invalidates all allocations).
    pub fn reset(&mut self) {
        self.current_y = 0;
        self.current_x = 0;
        self.shelf_height = 0;
        self.allocations.clear();
    }
}

impl Default for DecalAtlas {
    fn default() -> Self {
        Self::new(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)
    }
}

// ---------------------------------------------------------------------------
// DecalResources (CPU-side management)
// ---------------------------------------------------------------------------

/// CPU-side decal instance buffer management.
///
/// Maintains a list of active decals for upload to GPU each frame.
#[derive(Debug, Default)]
pub struct DecalResourcesCpu {
    /// Active decal instances.
    instances: Vec<DecalInstance>,
    /// Maximum capacity.
    capacity: usize,
    /// Dirty flag for upload.
    dirty: bool,
}

impl DecalResourcesCpu {
    /// Create new CPU resources with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            instances: Vec::with_capacity(capacity),
            capacity,
            dirty: false,
        }
    }

    /// Add a decal instance.
    ///
    /// Returns the index of the added decal, or None if at capacity.
    pub fn add(&mut self, decal: DecalInstance) -> Option<usize> {
        if self.instances.len() >= self.capacity {
            return None;
        }
        let index = self.instances.len();
        self.instances.push(decal);
        self.dirty = true;
        Some(index)
    }

    /// Remove a decal by index.
    ///
    /// Uses swap-remove for O(1) removal.
    pub fn remove(&mut self, index: usize) -> Option<DecalInstance> {
        if index >= self.instances.len() {
            return None;
        }
        self.dirty = true;
        Some(self.instances.swap_remove(index))
    }

    /// Get a decal by index.
    pub fn get(&self, index: usize) -> Option<&DecalInstance> {
        self.instances.get(index)
    }

    /// Get mutable reference to a decal.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DecalInstance> {
        self.dirty = true;
        self.instances.get_mut(index)
    }

    /// Get all instances as a slice.
    pub fn instances(&self) -> &[DecalInstance] {
        &self.instances
    }

    /// Get byte slice for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.instances)
    }

    /// Number of active decals.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Clear all decals.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.dirty = true;
    }

    /// Check and clear dirty flag.
    pub fn take_dirty(&mut self) -> bool {
        let was_dirty = self.dirty;
        self.dirty = false;
        was_dirty
    }

    /// Update fade values based on lifetime.
    ///
    /// Returns number of decals removed due to fade-out.
    pub fn update_fades(&mut self, delta_time: f32, fade_rate: f32) -> usize {
        let mut removed = 0;
        let mut i = 0;

        while i < self.instances.len() {
            self.instances[i].fade -= fade_rate * delta_time;

            if self.instances[i].fade <= 0.0 {
                self.instances.swap_remove(i);
                removed += 1;
                self.dirty = true;
            } else {
                i += 1;
            }
        }

        removed
    }

    /// Sort decals by distance from camera (back-to-front for blending).
    pub fn sort_by_distance(&mut self, camera_pos: [f32; 3]) {
        self.instances.sort_by(|a, b| {
            let dist_a = cpu_distance_squared(a.center(), camera_pos);
            let dist_b = cpu_distance_squared(b.center(), camera_pos);
            // Back-to-front: larger distance first
            dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Functions
// ---------------------------------------------------------------------------

/// Compute 4x4 matrix inverse (CPU reference).
///
/// Uses cofactor expansion for general 4x4 inverse.
pub fn cpu_invert_matrix(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Compute cofactors
    let c00 = m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
            - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
            + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]);

    let c01 = -(m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
              - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
              + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]));

    let c02 = m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
            - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
            + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]);

    let c03 = -(m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
              - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
              + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]));

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02 + m[0][3] * c03;

    if det.abs() < 1e-10 {
        // Return identity for singular matrix
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    let inv_det = 1.0 / det;

    // Compute remaining cofactors and build inverse
    let c10 = -(m[0][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
              - m[0][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
              + m[0][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]));

    let c11 = m[0][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
            - m[0][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
            + m[0][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]);

    let c12 = -(m[0][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
              - m[0][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
              + m[0][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]));

    let c13 = m[0][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
            - m[0][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
            + m[0][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]);

    let c20 = m[0][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
            - m[0][2] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
            + m[0][3] * (m[1][1] * m[3][2] - m[1][2] * m[3][1]);

    let c21 = -(m[0][0] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
              - m[0][2] * (m[1][0] * m[3][3] - m[1][3] * m[3][0])
              + m[0][3] * (m[1][0] * m[3][2] - m[1][2] * m[3][0]));

    let c22 = m[0][0] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
            - m[0][1] * (m[1][0] * m[3][3] - m[1][3] * m[3][0])
            + m[0][3] * (m[1][0] * m[3][1] - m[1][1] * m[3][0]);

    let c23 = -(m[0][0] * (m[1][1] * m[3][2] - m[1][2] * m[3][1])
              - m[0][1] * (m[1][0] * m[3][2] - m[1][2] * m[3][0])
              + m[0][2] * (m[1][0] * m[3][1] - m[1][1] * m[3][0]));

    let c30 = -(m[0][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
              - m[0][2] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
              + m[0][3] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]));

    let c31 = m[0][0] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
            - m[0][2] * (m[1][0] * m[2][3] - m[1][3] * m[2][0])
            + m[0][3] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]);

    let c32 = -(m[0][0] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][3] - m[1][3] * m[2][0])
              + m[0][3] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));

    let c33 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    // Transpose of cofactor matrix times inverse determinant
    [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det, c30 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det, c31 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det, c32 * inv_det],
        [c03 * inv_det, c13 * inv_det, c23 * inv_det, c33 * inv_det],
    ]
}

/// Build a transform matrix from position, rotation quaternion, and scale.
pub fn cpu_build_transform(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [[f32; 4]; 4] {
    let [x, y, z, w] = rotation;

    // Quaternion to rotation matrix
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    [
        [(1.0 - (yy + zz)) * scale[0], (xy + wz) * scale[0], (xz - wy) * scale[0], 0.0],
        [(xy - wz) * scale[1], (1.0 - (xx + zz)) * scale[1], (yz + wx) * scale[1], 0.0],
        [(xz + wy) * scale[2], (yz - wx) * scale[2], (1.0 - (xx + yy)) * scale[2], 0.0],
        [position[0], position[1], position[2], 1.0],
    ]
}

/// Transform a point by a 4x4 matrix.
pub fn cpu_transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    let inv_w = if w.abs() > 1e-10 { 1.0 / w } else { 1.0 };

    [
        (m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0]) * inv_w,
        (m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1]) * inv_w,
        (m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2]) * inv_w,
    ]
}

/// Compute squared distance between two points.
#[inline]
pub fn cpu_distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Test if a point is inside a decal volume (unit cube check after transform).
pub fn cpu_point_in_decal(world_to_decal: [[f32; 4]; 4], world_point: [f32; 3]) -> bool {
    let local = cpu_transform_point(world_to_decal, world_point);
    local[0].abs() <= 0.5 && local[1].abs() <= 0.5 && local[2].abs() <= 0.5
}

/// Project a point to decal UV coordinates.
///
/// Returns None if outside volume.
pub fn cpu_project_to_uv(
    world_to_decal: [[f32; 4]; 4],
    atlas_rect: [f32; 4],
    world_point: [f32; 3],
) -> Option<[f32; 2]> {
    let local = cpu_transform_point(world_to_decal, world_point);

    if local[0].abs() > 0.5 || local[1].abs() > 0.5 || local[2].abs() > 0.5 {
        return None;
    }

    let base_uv = [local[0] + 0.5, local[1] + 0.5];
    let atlas_uv = [
        atlas_rect[0] + base_uv[0] * atlas_rect[2],
        atlas_rect[1] + base_uv[1] * atlas_rect[3],
    ];

    Some(atlas_uv)
}

/// Blend decal normal with surface normal (Reoriented Normal Mapping).
pub fn cpu_blend_normals_rnm(surface: [f32; 3], decal: [f32; 3], strength: f32) -> [f32; 3] {
    // RNM blend
    let t = [surface[0], surface[1], surface[2] + 1.0];
    let u = [-decal[0], -decal[1], decal[2]];

    let dot_tu = t[0] * u[0] + t[1] * u[1] + t[2] * u[2];
    let blended = [
        t[0] * dot_tu - u[0] * t[2],
        t[1] * dot_tu - u[1] * t[2],
        t[2] * dot_tu - u[2] * t[2],
    ];

    // Normalize blended
    let len = (blended[0] * blended[0] + blended[1] * blended[1] + blended[2] * blended[2]).sqrt();
    let normalized = if len > 1e-6 {
        [blended[0] / len, blended[1] / len, blended[2] / len]
    } else {
        surface
    };

    // Lerp based on strength
    [
        surface[0] + (normalized[0] - surface[0]) * strength,
        surface[1] + (normalized[1] - surface[1]) * strength,
        surface[2] + (normalized[2] - surface[2]) * strength,
    ]
}

/// Apply decal fade over lifetime.
#[inline]
pub fn cpu_apply_fade(current_fade: f32, delta_time: f32, fade_rate: f32) -> f32 {
    (current_fade - fade_rate * delta_time).max(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_blend_mode_values() {
        assert_eq!(BlendMode::Albedo as u32, 0);
        assert_eq!(BlendMode::Normal as u32, 1);
        assert_eq!(BlendMode::Both as u32, 2);
        assert_eq!(BlendMode::Emissive as u32, 3);
    }

    #[test]
    fn test_blend_mode_affects() {
        assert!(BlendMode::Albedo.affects_albedo());
        assert!(!BlendMode::Albedo.affects_normal());

        assert!(!BlendMode::Normal.affects_albedo());
        assert!(BlendMode::Normal.affects_normal());

        assert!(BlendMode::Both.affects_albedo());
        assert!(BlendMode::Both.affects_normal());

        assert!(BlendMode::Emissive.affects_albedo());
        assert!(!BlendMode::Emissive.affects_normal());
    }

    #[test]
    fn test_decal_params_size() {
        // Verify struct size matches expected
        assert_eq!(std::mem::size_of::<DecalParams>(), 144);
    }

    #[test]
    fn test_decal_instance_size() {
        assert_eq!(std::mem::size_of::<DecalInstance>(), DECAL_INSTANCE_SIZE);
    }

    #[test]
    fn test_decal_instance_default() {
        let decal = DecalInstance::default();
        assert!(decal.is_visible());
        assert_eq!(decal.fade, 1.0);
        assert_eq!(decal.blend_mode, BlendMode::Albedo as u32);
    }

    #[test]
    fn test_decal_instance_builder() {
        let decal = DecalInstance::axis_aligned([1.0, 2.0, 3.0], [2.0, 2.0, 0.5])
            .with_color([1.0, 0.0, 0.0, 0.8])
            .with_blend_mode(BlendMode::Both)
            .with_normal_strength(0.5)
            .with_fade(0.75);

        assert_eq!(decal.color, [1.0, 0.0, 0.0, 0.8]);
        assert_eq!(decal.blend_mode, BlendMode::Both as u32);
        assert_eq!(decal.normal_strength, 0.5);
        assert_eq!(decal.fade, 0.75);
        assert!(decal.is_visible());
    }

    #[test]
    fn test_decal_center() {
        let decal = DecalInstance::axis_aligned([5.0, 10.0, 15.0], [1.0, 1.0, 1.0]);
        let center = decal.center();
        assert!(approx_eq_vec3(center, [5.0, 10.0, 15.0]));
    }

    #[test]
    fn test_point_in_decal_volume() {
        let decal = DecalInstance::axis_aligned([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);

        // Center should be inside
        assert!(decal.contains_point([0.0, 0.0, 0.0]));

        // Points within [-1, 1] should be inside (half of 2.0 scale)
        assert!(decal.contains_point([0.5, 0.5, 0.5]));
        assert!(decal.contains_point([-0.5, -0.5, -0.5]));

        // Edge cases (just inside)
        assert!(decal.contains_point([0.99, 0.0, 0.0]));

        // Outside
        assert!(!decal.contains_point([2.0, 0.0, 0.0]));
        assert!(!decal.contains_point([0.0, 2.0, 0.0]));
        assert!(!decal.contains_point([0.0, 0.0, 2.0]));
    }

    #[test]
    fn test_point_rejection_outside_volume() {
        let decal = DecalInstance::axis_aligned([0.0, 0.0, 0.0], [1.0, 1.0, 0.1]);

        // Should reject points outside the thin volume
        assert!(!decal.contains_point([0.0, 0.0, 1.0]));
        assert!(decal.contains_point([0.0, 0.0, 0.04])); // Just inside
    }

    #[test]
    fn test_atlas_uv_calculation() {
        let decal = DecalInstance::axis_aligned([0.0, 0.0, 0.0], [2.0, 2.0, 0.2])
            .with_atlas_rect([0.25, 0.5, 0.25, 0.25]);

        // Center of decal should map to center of atlas rect
        let uv = decal.project_point([0.0, 0.0, 0.0]).unwrap();
        assert!(approx_eq(uv[0], 0.25 + 0.5 * 0.25)); // 0.375
        assert!(approx_eq(uv[1], 0.5 + 0.5 * 0.25));  // 0.625

        // Outside should return None
        assert!(decal.project_point([5.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_normal_blending() {
        let surface = [0.0, 0.0, 1.0]; // Up
        let decal = [0.5, 0.0, 0.866]; // Tilted

        let blended = cpu_blend_normals_rnm(surface, decal, 1.0);

        // Result should be a valid unit normal
        let len = (blended[0] * blended[0] + blended[1] * blended[1] + blended[2] * blended[2]).sqrt();
        assert!(approx_eq(len, 1.0));

        // With strength 0, should return surface
        let no_blend = cpu_blend_normals_rnm(surface, decal, 0.0);
        assert!(approx_eq_vec3(no_blend, surface));
    }

    #[test]
    fn test_fade_over_lifetime() {
        let decal = DecalInstance::default().with_fade(1.0);

        // Simulate fade
        let new_fade = cpu_apply_fade(decal.fade, 0.5, 0.2); // 0.5s at 0.2/s
        assert!(approx_eq(new_fade, 0.9));

        // Should clamp to 0
        let fully_faded = cpu_apply_fade(0.1, 1.0, 0.5);
        assert_eq!(fully_faded, 0.0);
    }

    #[test]
    fn test_multiple_decal_blend_modes() {
        let albedo_decal = DecalInstance::default().with_blend_mode(BlendMode::Albedo);
        let normal_decal = DecalInstance::default().with_blend_mode(BlendMode::Normal);
        let both_decal = DecalInstance::default().with_blend_mode(BlendMode::Both);
        let emissive_decal = DecalInstance::default().with_blend_mode(BlendMode::Emissive);

        assert_eq!(albedo_decal.blend_mode_enum(), BlendMode::Albedo);
        assert_eq!(normal_decal.blend_mode_enum(), BlendMode::Normal);
        assert_eq!(both_decal.blend_mode_enum(), BlendMode::Both);
        assert_eq!(emissive_decal.blend_mode_enum(), BlendMode::Emissive);
    }

    #[test]
    fn test_atlas_allocation() {
        let mut atlas = DecalAtlas::new(256, 256);

        // First allocation should succeed
        let rect1 = atlas.allocate(64, 64).unwrap();
        assert!(approx_eq(rect1.x, 0.0));
        assert!(approx_eq(rect1.y, 0.0));
        assert!(approx_eq(rect1.width, 0.25));
        assert!(approx_eq(rect1.height, 0.25));

        // Second allocation on same shelf
        let rect2 = atlas.allocate(64, 64).unwrap();
        assert!(approx_eq(rect2.x, 0.25));
        assert!(approx_eq(rect2.y, 0.0));

        // Fill the atlas
        for _ in 0..14 {
            atlas.allocate(64, 64);
        }

        // Should fail when full
        assert!(atlas.allocate(64, 64).is_none());
    }

    #[test]
    fn test_decal_resources_cpu() {
        let mut resources = DecalResourcesCpu::new(100);

        // Add decals
        let idx = resources.add(DecalInstance::default()).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(resources.len(), 1);

        // Get decal
        let decal = resources.get(0).unwrap();
        assert!(decal.is_visible());

        // Remove decal
        let removed = resources.remove(0).unwrap();
        assert!(removed.is_visible());
        assert!(resources.is_empty());
    }

    #[test]
    fn test_matrix_inverse() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let inv = cpu_invert_matrix(identity);

        // Inverse of identity is identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(inv[i][j], expected));
            }
        }
    }

    #[test]
    fn test_transform_build_and_invert() {
        let position = [10.0, 20.0, 30.0];
        let rotation = [0.0, 0.0, 0.0, 1.0]; // Identity rotation
        let scale = [2.0, 2.0, 2.0];

        let transform = cpu_build_transform(position, rotation, scale);
        let inverse = cpu_invert_matrix(transform);

        // Transform point to local and back
        let world_point = [12.0, 22.0, 32.0];
        let local = cpu_transform_point(inverse, world_point);
        let back = cpu_transform_point(transform, local);

        assert!(approx_eq_vec3(back, world_point));
    }

    #[test]
    fn test_decal_visibility() {
        let visible = DecalInstance::default()
            .with_fade(1.0)
            .with_color([1.0, 1.0, 1.0, 1.0]);
        assert!(visible.is_visible());

        let faded = DecalInstance::default()
            .with_fade(0.0)
            .with_color([1.0, 1.0, 1.0, 1.0]);
        assert!(!faded.is_visible());

        let transparent = DecalInstance::default()
            .with_fade(1.0)
            .with_color([1.0, 1.0, 1.0, 0.0]);
        assert!(!transparent.is_visible());
    }

    #[test]
    fn test_atlas_rect_from_pixels() {
        let rect = AtlasRect::from_pixels(256, 512, 128, 64, 1024, 1024);

        assert!(approx_eq(rect.x, 0.25));
        assert!(approx_eq(rect.y, 0.5));
        assert!(approx_eq(rect.width, 0.125));
        assert!(approx_eq(rect.height, 0.0625));
    }

    // -------------------------------------------------------------------------
    // WGSL Shader Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_shader_parses() {
        let shader_source = include_str!("../../shaders/decals/decal.vert.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Vertex shader should parse successfully");

        // Verify entry point exists
        let entry = module.entry_points.iter()
            .find(|ep| ep.name == "vs_decal")
            .expect("vs_decal entry point should exist");

        assert_eq!(entry.stage, naga::ShaderStage::Vertex);
    }

    #[test]
    fn test_fragment_shader_parses() {
        let shader_source = include_str!("../../shaders/decals/decal.frag.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Fragment shader should parse successfully");

        // Verify entry point exists
        let entry = module.entry_points.iter()
            .find(|ep| ep.name == "fs_decal")
            .expect("fs_decal entry point should exist");

        assert_eq!(entry.stage, naga::ShaderStage::Fragment);
    }

    #[test]
    fn test_vertex_shader_validates() {
        let shader_source = include_str!("../../shaders/decals/decal.vert.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Vertex shader should parse");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module)
            .expect("Vertex shader should validate");
    }

    #[test]
    fn test_fragment_shader_validates() {
        let shader_source = include_str!("../../shaders/decals/decal.frag.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Fragment shader should parse");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module)
            .expect("Fragment shader should validate");
    }

    #[test]
    fn test_shader_struct_alignment() {
        // Verify that our Rust structs match WGSL layout expectations
        // DecalParams: 2x mat4 (64 bytes each) + vec4 (16 bytes) = 144 bytes
        assert_eq!(std::mem::size_of::<DecalParams>(), 144);

        // DecalInstance: 2x mat4 (128) + 2x vec4 (32) + u32 + f32 + f32 + f32 (16) = 176 bytes
        assert_eq!(std::mem::size_of::<DecalInstance>(), 176);
    }

    #[test]
    fn test_vertex_shader_bindings() {
        let shader_source = include_str!("../../shaders/decals/decal.vert.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Shader should parse");

        // Count bindings in group 0
        let mut uniform_count = 0;
        let mut storage_count = 0;

        for (_, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                if binding.group == 0 {
                    match var.space {
                        naga::AddressSpace::Uniform => uniform_count += 1,
                        naga::AddressSpace::Storage { .. } => storage_count += 1,
                        _ => {}
                    }
                }
            }
        }

        // Should have 1 uniform (DecalParams) and 1 storage (decals array)
        assert_eq!(uniform_count, 1, "Expected 1 uniform binding");
        assert_eq!(storage_count, 1, "Expected 1 storage binding");
    }

    #[test]
    fn test_decal_update_fades_removes_dead() {
        let mut resources = DecalResourcesCpu::new(100);

        // Add decals with different fade values
        resources.add(DecalInstance::default().with_fade(1.0));
        resources.add(DecalInstance::default().with_fade(0.5));
        resources.add(DecalInstance::default().with_fade(0.1));

        assert_eq!(resources.len(), 3);

        // Update fades - 1 second at 0.2/s removes the 0.1 fade decal
        let removed = resources.update_fades(1.0, 0.2);
        assert_eq!(removed, 1);
        assert_eq!(resources.len(), 2);

        // Further updates should remove more
        let removed2 = resources.update_fades(2.0, 0.2);
        assert_eq!(removed2, 1); // 0.5 - 0.4 = 0.1 -> 0.1 - 0.4 = -0.3 -> removed
        assert_eq!(resources.len(), 1);
    }

    #[test]
    fn test_decal_sort_by_distance() {
        let mut resources = DecalResourcesCpu::new(100);

        // Add decals at different distances from origin
        resources.add(DecalInstance::axis_aligned([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        resources.add(DecalInstance::axis_aligned([10.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        resources.add(DecalInstance::axis_aligned([5.0, 0.0, 0.0], [1.0, 1.0, 1.0]));

        // Sort by distance from camera at origin
        resources.sort_by_distance([0.0, 0.0, 0.0]);

        // Should be sorted back-to-front (farthest first)
        let centers: Vec<f32> = resources.instances()
            .iter()
            .map(|d| d.center()[0])
            .collect();

        assert!(approx_eq(centers[0], 10.0)); // Farthest
        assert!(approx_eq(centers[1], 5.0));  // Middle
        assert!(approx_eq(centers[2], 1.0));  // Nearest
    }
}
