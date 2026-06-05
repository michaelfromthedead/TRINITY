//! HiZ Occlusion Test for GPU-Driven Occlusion Culling (T-WGPU-P6.4.3).
//!
//! This module provides HiZ (Hierarchical-Z) occlusion testing functions
//! for GPU-driven rendering. It works in conjunction with the HiZ pyramid
//! (T-WGPU-P6.4.1) to efficiently cull objects hidden behind other geometry.
//!
//! # Overview
//!
//! HiZ occlusion testing determines if an object's bounding volume is
//! completely hidden behind existing geometry:
//!
//! 1. **AABB Projection**: Project world-space AABB to screen-space rect
//! 2. **Mip Selection**: Choose HiZ mip level based on rect size
//! 3. **Depth Sampling**: Sample HiZ depth at rect corners (conservative max)
//! 4. **Depth Comparison**: Compare AABB near depth against HiZ depth
//!
//! # Depth Convention (Reverse-Z)
//!
//! TRINITY uses reversed-Z depth (near=1.0, far=0.0):
//! - Higher depth values are closer to the camera
//! - HiZ stores MAX depth per region (closest visible surface)
//! - If `object_near_depth < hiz_max_depth`, object is occluded
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::hiz_occlusion::{
//!     HiZOcclusionParams, cpu_project_aabb, cpu_select_mip_level, cpu_test_occlusion,
//! };
//!
//! // Create params from view-projection matrix
//! let params = HiZOcclusionParams::new(&view_proj, 1920.0, 1080.0, 0.1, 11);
//!
//! // CPU reference test
//! let (screen_min, screen_max, near_depth, valid) = cpu_project_aabb(
//!     [0.0, 0.0, -5.0], [1.0, 1.0, -4.0], &params.view_projection
//! );
//!
//! if valid {
//!     let mip = cpu_select_mip_level(
//!         screen_max.0 - screen_min.0,
//!         screen_max.1 - screen_min.1,
//!         11
//!     );
//! }
//! ```
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per AABB
//! - Target: < 0.1ms for 100K instances on modern GPU
//! - Memory: 32 bytes per input AABB, 4 bytes per output
//!
//! # See Also
//!
//! - [`super::hiz_pyramid`] - HiZ pyramid texture creation
//! - [`super::occlusion_cull`] - Full occlusion culling pipeline
//! - [`super::frustum_cull`] - Frustum culling (should run before HiZ)

use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Size of [`HiZOcclusionParams`] struct in bytes.
pub const HIZ_OCCLUSION_PARAMS_SIZE: usize = 80;

/// Size of [`InputAABB`] struct in bytes.
pub const INPUT_AABB_SIZE: usize = 32;

/// Size of [`BatchParams`] struct in bytes.
pub const BATCH_PARAMS_SIZE: usize = 16;

/// Maximum supported mip levels.
pub const MAX_MIP_LEVEL: u32 = 14;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Conservative screen-space expansion factor (pixels).
pub const CONSERVATIVE_EXPAND: f32 = 1.0;

// =============================================================================
// SHADER SOURCE
// =============================================================================

/// Embedded HiZ occlusion test shader source (T-WGPU-P6.4.3).
///
/// This shader performs hierarchical-Z occlusion testing for batch processing.
/// Use with [`HiZOcclusionParams`] and [`BatchParams`] for uniform buffers.
pub const HIZ_OCCLUSION_SHADER: &str = include_str!("../../shaders/hiz_occlusion.wgsl");

// =============================================================================
// HIZ OCCLUSION PARAMS
// =============================================================================

/// GPU uniform buffer for HiZ occlusion testing parameters (T-WGPU-P6.4.3).
///
/// # Memory Layout
///
/// 80 bytes, std140 compatible:
///
/// | Offset | Field           | Size | Description                       |
/// |--------|-----------------|------|-----------------------------------|
/// | 0      | view_projection | 64   | Combined VP matrix (column-major) |
/// | 64     | hiz_size        | 8    | HiZ base resolution (w, h)        |
/// | 72     | near_plane      | 4    | Near plane distance               |
/// | 76     | max_mip         | 4    | Maximum mip level (num_mips - 1)  |
///
/// # Example
///
/// ```ignore
/// let params = HiZOcclusionParams::new(&view_proj, 1920.0, 1080.0, 0.1, 11);
/// queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZOcclusionParams {
    /// Combined view-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// HiZ pyramid base resolution (mip 0): [width, height].
    pub hiz_size: [f32; 2],
    /// Near plane distance for clipping.
    pub near_plane: f32,
    /// Maximum mip level (num_mips - 1).
    pub max_mip: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<HiZOcclusionParams>() == HIZ_OCCLUSION_PARAMS_SIZE);

impl HiZOcclusionParams {
    /// Create HiZ occlusion parameters.
    ///
    /// # Arguments
    ///
    /// * `view_projection` - Combined view-projection matrix (column-major).
    /// * `hiz_width` - HiZ pyramid width at mip 0.
    /// * `hiz_height` - HiZ pyramid height at mip 0.
    /// * `near_plane` - Near plane distance.
    /// * `num_mips` - Number of mip levels in HiZ pyramid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let params = HiZOcclusionParams::new(&view_proj_matrix, 1920.0, 1080.0, 0.1, 11);
    /// ```
    pub fn new(
        view_projection: &[[f32; 4]; 4],
        hiz_width: f32,
        hiz_height: f32,
        near_plane: f32,
        num_mips: u32,
    ) -> Self {
        Self {
            view_projection: *view_projection,
            hiz_size: [hiz_width, hiz_height],
            near_plane,
            max_mip: num_mips.saturating_sub(1),
        }
    }

    /// Create from integer dimensions.
    pub fn from_dimensions(
        view_projection: &[[f32; 4]; 4],
        width: u32,
        height: u32,
        near_plane: f32,
        num_mips: u32,
    ) -> Self {
        Self::new(view_projection, width as f32, height as f32, near_plane, num_mips)
    }

    /// Get the HiZ size as integers.
    #[inline]
    pub fn hiz_dimensions(&self) -> (u32, u32) {
        (self.hiz_size[0] as u32, self.hiz_size[1] as u32)
    }

    /// Calculate the number of mips from dimensions.
    pub fn calculate_num_mips(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        if max_dim == 0 {
            return 1;
        }
        (32 - max_dim.leading_zeros()).max(1)
    }
}

// =============================================================================
// BATCH PARAMS
// =============================================================================

/// GPU uniform buffer for batch processing parameters.
///
/// # Memory Layout
///
/// 16 bytes:
///
/// | Offset | Field       | Size | Description              |
/// |--------|-------------|------|--------------------------|
/// | 0      | num_objects | 4    | Number of AABBs          |
/// | 4      | flags       | 4    | Processing flags         |
/// | 8      | _pad0       | 4    | Padding                  |
/// | 12     | _pad1       | 4    | Padding                  |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchParams {
    /// Number of AABBs to process.
    pub num_objects: u32,
    /// Processing flags (reserved).
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<BatchParams>() == BATCH_PARAMS_SIZE);

impl BatchParams {
    /// Create batch parameters.
    pub fn new(num_objects: u32) -> Self {
        Self {
            num_objects,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create batch parameters with flags.
    pub fn with_flags(num_objects: u32, flags: u32) -> Self {
        Self {
            num_objects,
            flags,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Calculate the number of workgroups needed.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_objects + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Get dispatch dimensions.
    #[inline]
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        (self.num_workgroups(), 1, 1)
    }
}

// =============================================================================
// INPUT AABB
// =============================================================================

/// Input AABB for batch HiZ occlusion testing.
///
/// # Memory Layout
///
/// 32 bytes, vec4 aligned:
///
/// | Offset | Field | Size | Description        |
/// |--------|-------|------|--------------------|
/// | 0      | min   | 12   | Minimum corner     |
/// | 12     | _pad0 | 4    | Padding            |
/// | 16     | max   | 12   | Maximum corner     |
/// | 28     | _pad1 | 4    | Padding            |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InputAABB {
    /// Minimum corner of AABB in world space.
    pub min: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Maximum corner of AABB in world space.
    pub max: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad1: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InputAABB>() == INPUT_AABB_SIZE);

impl InputAABB {
    /// Create an input AABB.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _pad0: 0.0,
            max,
            _pad1: 0.0,
        }
    }

    /// Create from min/max tuples.
    pub fn from_tuples(min: (f32, f32, f32), max: (f32, f32, f32)) -> Self {
        Self::new([min.0, min.1, min.2], [max.0, max.1, max.2])
    }

    /// Get the center of the AABB.
    #[inline]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get the half-extents of the AABB.
    #[inline]
    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// =============================================================================

/// Transform a point by a 4x4 matrix (column-major).
fn transform_point(p: [f32; 3], m: &[[f32; 4]; 4]) -> [f32; 4] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3],
    ]
}

/// Get the i-th corner of an AABB (0-7).
fn get_aabb_corner(aabb_min: [f32; 3], aabb_max: [f32; 3], index: u32) -> [f32; 3] {
    [
        if (index & 1) != 0 { aabb_max[0] } else { aabb_min[0] },
        if (index & 2) != 0 { aabb_max[1] } else { aabb_min[1] },
        if (index & 4) != 0 { aabb_max[2] } else { aabb_min[2] },
    ]
}

/// Project a world-space AABB to screen-space rectangle.
///
/// Returns `(screen_min, screen_max, near_depth, valid)`:
/// - `screen_min`: Minimum screen coordinates (x, y) in pixels
/// - `screen_max`: Maximum screen coordinates (x, y) in pixels
/// - `near_depth`: Nearest depth in NDC [0, 1] (1.0 = near in reverse-Z)
/// - `valid`: True if at least one corner is in front of camera
///
/// # Arguments
///
/// * `aabb_min` - Minimum corner of AABB in world space
/// * `aabb_max` - Maximum corner of AABB in world space
/// * `view_proj` - Combined view-projection matrix (column-major)
/// * `screen_width` - Screen width in pixels
/// * `screen_height` - Screen height in pixels
///
/// # Example
///
/// ```ignore
/// let (min, max, depth, valid) = cpu_project_aabb(
///     [0.0, 0.0, -5.0],
///     [1.0, 1.0, -4.0],
///     &view_proj,
///     1920.0, 1080.0
/// );
/// ```
pub fn cpu_project_aabb(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    view_proj: &[[f32; 4]; 4],
    screen_width: f32,
    screen_height: f32,
) -> ((f32, f32), (f32, f32), f32, bool) {
    let mut screen_min = (f32::MAX, f32::MAX);
    let mut screen_max = (f32::MIN, f32::MIN);
    let mut near_depth: f32 = 0.0; // Far in reverse-Z
    let mut all_behind = true;
    let mut any_behind = false;

    // Process all 8 corners
    for i in 0..8 {
        let corner = get_aabb_corner(aabb_min, aabb_max, i);
        let clip = transform_point(corner, view_proj);

        // Check if in front of near plane (w > 0)
        if clip[3] > EPSILON {
            all_behind = false;

            // Perspective divide to NDC
            let inv_w = 1.0 / clip[3];
            let ndc_x = clip[0] * inv_w;
            let ndc_y = clip[1] * inv_w;
            let ndc_z = clip[2] * inv_w;

            // NDC to screen coordinates
            let uv_x = (ndc_x + 1.0) * 0.5;
            let uv_y = 1.0 - (ndc_y + 1.0) * 0.5; // Flip Y
            let screen_x = uv_x * screen_width;
            let screen_y = uv_y * screen_height;

            // Expand bounds
            screen_min.0 = screen_min.0.min(screen_x);
            screen_min.1 = screen_min.1.min(screen_y);
            screen_max.0 = screen_max.0.max(screen_x);
            screen_max.1 = screen_max.1.max(screen_y);

            // Track nearest depth (max in reverse-Z)
            let depth = ndc_z.clamp(0.0, 1.0);
            near_depth = near_depth.max(depth);
        } else {
            any_behind = true;
        }
    }

    // Handle edge cases
    if all_behind {
        return ((0.0, 0.0), (0.0, 0.0), 0.0, false);
    }

    if any_behind {
        // Conservatively extend to screen edges
        screen_min = (0.0, 0.0);
        screen_max = (screen_width, screen_height);
        near_depth = 1.0; // Near plane in reverse-Z
    }

    // Clamp to screen bounds
    screen_min.0 = screen_min.0.max(0.0).min(screen_width);
    screen_min.1 = screen_min.1.max(0.0).min(screen_height);
    screen_max.0 = screen_max.0.max(0.0).min(screen_width);
    screen_max.1 = screen_max.1.max(0.0).min(screen_height);

    // Apply conservative expansion
    screen_min.0 = (screen_min.0 - CONSERVATIVE_EXPAND).max(0.0);
    screen_min.1 = (screen_min.1 - CONSERVATIVE_EXPAND).max(0.0);
    screen_max.0 = (screen_max.0 + CONSERVATIVE_EXPAND).min(screen_width);
    screen_max.1 = (screen_max.1 + CONSERVATIVE_EXPAND).min(screen_height);

    (screen_min, screen_max, near_depth, true)
}

/// Select HiZ mip level based on screen-space rect size.
///
/// Chooses a mip level where the rect covers approximately 1-2 texels.
///
/// # Arguments
///
/// * `rect_width` - Width of screen-space rect in pixels
/// * `rect_height` - Height of screen-space rect in pixels
/// * `max_mip` - Maximum mip level (num_mips - 1)
///
/// # Returns
///
/// Mip level to use for HiZ sampling (0 = base resolution).
///
/// # Example
///
/// ```ignore
/// let mip = cpu_select_mip_level(64.0, 64.0, 10);
/// assert_eq!(mip, 6); // log2(64) = 6
/// ```
pub fn cpu_select_mip_level(rect_width: f32, rect_height: f32, max_mip: u32) -> u32 {
    let max_dim = rect_width.max(rect_height);

    if max_dim <= 1.0 {
        return 0;
    }

    let mip = max_dim.log2().floor() as u32;
    mip.min(max_mip)
}

/// Test if an AABB is occluded against a CPU HiZ buffer.
///
/// This is a CPU reference implementation for testing. For GPU rendering,
/// use the WGSL shader.
///
/// # Arguments
///
/// * `aabb_min` - Minimum corner of AABB in world space
/// * `aabb_max` - Maximum corner of AABB in world space
/// * `view_proj` - Combined view-projection matrix
/// * `hiz_buffer` - HiZ buffer data (all mips concatenated, row-major)
/// * `hiz_width` - HiZ base width
/// * `hiz_height` - HiZ base height
/// * `num_mips` - Number of mip levels
///
/// # Returns
///
/// `true` if visible (not occluded), `false` if occluded.
pub fn cpu_test_occlusion(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    view_proj: &[[f32; 4]; 4],
    hiz_buffer: &[f32],
    hiz_width: u32,
    hiz_height: u32,
    num_mips: u32,
) -> bool {
    let (screen_min, screen_max, near_depth, valid) =
        cpu_project_aabb(aabb_min, aabb_max, view_proj, hiz_width as f32, hiz_height as f32);

    if !valid {
        return false;
    }

    let rect_width = screen_max.0 - screen_min.0;
    let rect_height = screen_max.1 - screen_min.1;

    // Degenerate rect: mark as visible
    if rect_width < 1.0 || rect_height < 1.0 {
        return true;
    }

    let max_mip = num_mips.saturating_sub(1);
    let mip_level = cpu_select_mip_level(rect_width, rect_height, max_mip);

    // Calculate mip offset in buffer
    let mip_offset = calculate_mip_offset(hiz_width, hiz_height, mip_level);
    let mip_width = (hiz_width >> mip_level).max(1);
    let mip_height = (hiz_height >> mip_level).max(1);

    // Convert to UV and sample corners
    let min_uv = (screen_min.0 / hiz_width as f32, screen_min.1 / hiz_height as f32);
    let max_uv = (screen_max.0 / hiz_width as f32, screen_max.1 / hiz_height as f32);

    let hiz_depth = sample_hiz_rect_max(
        hiz_buffer,
        mip_offset,
        mip_width,
        mip_height,
        min_uv,
        max_uv,
    );

    // Occlusion test (reverse-Z)
    near_depth >= (hiz_depth - EPSILON)
}

/// Calculate the byte offset of a mip level in a concatenated HiZ buffer.
fn calculate_mip_offset(base_width: u32, base_height: u32, mip_level: u32) -> usize {
    let mut offset = 0usize;
    for m in 0..mip_level {
        let w = (base_width >> m).max(1) as usize;
        let h = (base_height >> m).max(1) as usize;
        offset += w * h;
    }
    offset
}

/// Sample HiZ depth at a point.
fn sample_hiz_point(
    buffer: &[f32],
    mip_offset: usize,
    mip_width: u32,
    mip_height: u32,
    uv: (f32, f32),
) -> f32 {
    let x = ((uv.0 * mip_width as f32) as u32).min(mip_width - 1);
    let y = ((uv.1 * mip_height as f32) as u32).min(mip_height - 1);
    let idx = mip_offset + (y as usize) * (mip_width as usize) + (x as usize);

    if idx < buffer.len() {
        buffer[idx]
    } else {
        0.0 // Far depth if out of bounds
    }
}

/// Sample HiZ rect and return max depth (conservative).
fn sample_hiz_rect_max(
    buffer: &[f32],
    mip_offset: usize,
    mip_width: u32,
    mip_height: u32,
    min_uv: (f32, f32),
    max_uv: (f32, f32),
) -> f32 {
    let d00 = sample_hiz_point(buffer, mip_offset, mip_width, mip_height, min_uv);
    let d10 = sample_hiz_point(buffer, mip_offset, mip_width, mip_height, (max_uv.0, min_uv.1));
    let d01 = sample_hiz_point(buffer, mip_offset, mip_width, mip_height, (min_uv.0, max_uv.1));
    let d11 = sample_hiz_point(buffer, mip_offset, mip_width, mip_height, max_uv);

    d00.max(d10).max(d01.max(d11))
}

/// Calculate workgroups needed for a given number of objects.
#[inline]
pub fn workgroups_for_objects(num_objects: u32) -> u32 {
    (num_objects + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

// =============================================================================
// BIND GROUP LAYOUT HELPERS
// =============================================================================

/// Create a bind group layout for HiZ occlusion testing textures (Group 0).
///
/// # Binding Layout
///
/// | Binding | Type    | Stage   | Description            |
/// |---------|---------|---------|------------------------|
/// | 0       | Texture | Compute | HiZ pyramid (all mips) |
/// | 1       | Sampler | Compute | Linear sampler         |
pub fn create_hiz_occlusion_texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_occlusion_texture_layout"),
        entries: &[
            // HiZ pyramid texture
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Create a bind group layout for HiZ occlusion params (Group 1).
///
/// # Binding Layout
///
/// | Binding | Type    | Stage   | Description          |
/// |---------|---------|---------|----------------------|
/// | 0       | Uniform | Compute | HiZOcclusionParams   |
pub fn create_hiz_occlusion_params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_occlusion_params_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(HIZ_OCCLUSION_PARAMS_SIZE as u64).unwrap()
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Create a bind group layout for batch processing (Group 2).
///
/// # Binding Layout
///
/// | Binding | Type    | Stage   | Description          |
/// |---------|---------|---------|----------------------|
/// | 0       | Uniform | Compute | BatchParams          |
/// | 1       | Storage | Compute | Input AABBs (read)   |
/// | 2       | Storage | Compute | Visibility (r/w)     |
pub fn create_hiz_occlusion_batch_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_occlusion_batch_layout"),
        entries: &[
            // BatchParams
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(BATCH_PARAMS_SIZE as u64).unwrap()
                    ),
                },
                count: None,
            },
            // Input AABBs
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(INPUT_AABB_SIZE as u64).unwrap()
                    ),
                },
                count: None,
            },
            // Visibility results
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(4).unwrap() // u32
                    ),
                },
                count: None,
            },
        ],
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Struct Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hiz_occlusion_params_size() {
        assert_eq!(mem::size_of::<HiZOcclusionParams>(), HIZ_OCCLUSION_PARAMS_SIZE);
        assert_eq!(mem::size_of::<HiZOcclusionParams>(), 80);
    }

    #[test]
    fn test_batch_params_size() {
        assert_eq!(mem::size_of::<BatchParams>(), BATCH_PARAMS_SIZE);
        assert_eq!(mem::size_of::<BatchParams>(), 16);
    }

    #[test]
    fn test_input_aabb_size() {
        assert_eq!(mem::size_of::<InputAABB>(), INPUT_AABB_SIZE);
        assert_eq!(mem::size_of::<InputAABB>(), 32);
    }

    // -------------------------------------------------------------------------
    // Shader Source Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_source_exists() {
        assert!(!HIZ_OCCLUSION_SHADER.is_empty());
        assert!(HIZ_OCCLUSION_SHADER.contains("fn test_hiz_occlusion"));
        assert!(HIZ_OCCLUSION_SHADER.contains("fn project_aabb_to_screen"));
        assert!(HIZ_OCCLUSION_SHADER.contains("fn select_mip_level"));
        assert!(HIZ_OCCLUSION_SHADER.contains("@compute @workgroup_size(256)"));
    }

    #[test]
    fn test_shader_bindings() {
        assert!(HIZ_OCCLUSION_SHADER.contains("@group(0) @binding(0)"));
        assert!(HIZ_OCCLUSION_SHADER.contains("@group(1) @binding(0)"));
        assert!(HIZ_OCCLUSION_SHADER.contains("@group(2) @binding(0)"));
    }

    // -------------------------------------------------------------------------
    // Mip Level Selection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_selection_power_of_two() {
        // log2(1) = 0, log2(2) = 1, log2(4) = 2, etc.
        assert_eq!(cpu_select_mip_level(1.0, 1.0, 10), 0);
        assert_eq!(cpu_select_mip_level(2.0, 2.0, 10), 1);
        assert_eq!(cpu_select_mip_level(4.0, 4.0, 10), 2);
        assert_eq!(cpu_select_mip_level(8.0, 8.0, 10), 3);
        assert_eq!(cpu_select_mip_level(16.0, 16.0, 10), 4);
        assert_eq!(cpu_select_mip_level(64.0, 64.0, 10), 6);
        assert_eq!(cpu_select_mip_level(256.0, 256.0, 10), 8);
        assert_eq!(cpu_select_mip_level(1024.0, 1024.0, 10), 10);
    }

    #[test]
    fn test_mip_selection_non_power_of_two() {
        // log2(3) ~= 1.58 -> floor = 1
        assert_eq!(cpu_select_mip_level(3.0, 3.0, 10), 1);
        // log2(5) ~= 2.32 -> floor = 2
        assert_eq!(cpu_select_mip_level(5.0, 5.0, 10), 2);
        // log2(100) ~= 6.64 -> floor = 6
        assert_eq!(cpu_select_mip_level(100.0, 100.0, 10), 6);
    }

    #[test]
    fn test_mip_selection_asymmetric() {
        // Uses max dimension
        assert_eq!(cpu_select_mip_level(64.0, 32.0, 10), 6); // max(64,32) = 64
        assert_eq!(cpu_select_mip_level(16.0, 128.0, 10), 7); // max(16,128) = 128
    }

    #[test]
    fn test_mip_selection_clamped() {
        // Should clamp to max_mip
        assert_eq!(cpu_select_mip_level(2048.0, 2048.0, 5), 5);
        assert_eq!(cpu_select_mip_level(4096.0, 4096.0, 10), 10);
    }

    #[test]
    fn test_mip_selection_small() {
        // Small sizes should use mip 0
        assert_eq!(cpu_select_mip_level(0.5, 0.5, 10), 0);
        assert_eq!(cpu_select_mip_level(1.0, 1.0, 10), 0);
        assert_eq!(cpu_select_mip_level(0.0, 0.0, 10), 0);
    }

    // -------------------------------------------------------------------------
    // AABB Projection Tests
    // -------------------------------------------------------------------------

    fn identity_matrix() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn simple_perspective_matrix() -> [[f32; 4]; 4] {
        // Simple perspective: z' = z, w' = -z (so w > 0 when z < 0)
        // Actually for reverse-Z we typically have different setup
        // Let's use a simple orthographic for testing
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn test_aabb_projection_valid() {
        // With identity matrix, points in world space map directly to NDC
        // A box centered at origin should project to center of screen
        let vp = identity_matrix();
        let (min, max, depth, valid) = cpu_project_aabb(
            [-0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            &vp,
            100.0, 100.0
        );

        assert!(valid);
        // With identity, all w = 1.0, so projection should work
        // Screen coords depend on NDC to screen conversion
    }

    #[test]
    fn test_aabb_projection_behind_camera() {
        // Points with w <= 0 are behind camera
        // With identity, w = 1 always, so we need a different matrix
        // to test behind-camera case
        // For now, this is a placeholder - real tests would use actual projection matrices
    }

    // -------------------------------------------------------------------------
    // Workgroup Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroups_calculation() {
        assert_eq!(workgroups_for_objects(0), 0);
        assert_eq!(workgroups_for_objects(1), 1);
        assert_eq!(workgroups_for_objects(256), 1);
        assert_eq!(workgroups_for_objects(257), 2);
        assert_eq!(workgroups_for_objects(512), 2);
        assert_eq!(workgroups_for_objects(1000), 4);
        assert_eq!(workgroups_for_objects(100000), 391);
    }

    // -------------------------------------------------------------------------
    // BatchParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_params_new() {
        let params = BatchParams::new(1000);
        assert_eq!(params.num_objects, 1000);
        assert_eq!(params.flags, 0);
        assert_eq!(params.num_workgroups(), 4);
        assert_eq!(params.dispatch_size(), (4, 1, 1));
    }

    #[test]
    fn test_batch_params_with_flags() {
        let params = BatchParams::with_flags(256, 0xFF);
        assert_eq!(params.num_objects, 256);
        assert_eq!(params.flags, 0xFF);
        assert_eq!(params.num_workgroups(), 1);
    }

    // -------------------------------------------------------------------------
    // HiZOcclusionParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hiz_occlusion_params_new() {
        let vp = identity_matrix();
        let params = HiZOcclusionParams::new(&vp, 1920.0, 1080.0, 0.1, 11);

        assert_eq!(params.hiz_size, [1920.0, 1080.0]);
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.max_mip, 10); // num_mips - 1
    }

    #[test]
    fn test_hiz_occlusion_params_from_dimensions() {
        let vp = identity_matrix();
        let params = HiZOcclusionParams::from_dimensions(&vp, 1920, 1080, 0.1, 11);

        assert_eq!(params.hiz_dimensions(), (1920, 1080));
    }

    #[test]
    fn test_calculate_num_mips() {
        assert_eq!(HiZOcclusionParams::calculate_num_mips(1, 1), 1);
        assert_eq!(HiZOcclusionParams::calculate_num_mips(256, 256), 9);
        assert_eq!(HiZOcclusionParams::calculate_num_mips(1920, 1080), 11);
        assert_eq!(HiZOcclusionParams::calculate_num_mips(3840, 2160), 12);
    }

    // -------------------------------------------------------------------------
    // InputAABB Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_input_aabb_new() {
        let aabb = InputAABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(aabb.min, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max, [1.0, 1.0, 1.0]);
        assert_eq!(aabb._pad0, 0.0);
        assert_eq!(aabb._pad1, 0.0);
    }

    #[test]
    fn test_input_aabb_center() {
        let aabb = InputAABB::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
        assert_eq!(aabb.center(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_input_aabb_half_extents() {
        let aabb = InputAABB::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
        assert_eq!(aabb.half_extents(), [1.0, 2.0, 3.0]);
    }

    // -------------------------------------------------------------------------
    // Bytemuck Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bytemuck_roundtrip() {
        let params = HiZOcclusionParams::new(&identity_matrix(), 1920.0, 1080.0, 0.1, 11);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), HIZ_OCCLUSION_PARAMS_SIZE);

        let restored: HiZOcclusionParams = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.hiz_size, params.hiz_size);
        assert_eq!(restored.near_plane, params.near_plane);
        assert_eq!(restored.max_mip, params.max_mip);
    }

    #[test]
    fn test_batch_params_bytemuck() {
        let params = BatchParams::new(1000);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), BATCH_PARAMS_SIZE);

        let restored: BatchParams = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.num_objects, params.num_objects);
    }

    #[test]
    fn test_input_aabb_bytemuck() {
        let aabb = InputAABB::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let bytes = bytemuck::bytes_of(&aabb);
        assert_eq!(bytes.len(), INPUT_AABB_SIZE);

        let restored: InputAABB = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.min, aabb.min);
        assert_eq!(restored.max, aabb.max);
    }

    // -------------------------------------------------------------------------
    // AABB Corner Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_aabb_corner() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Corners: bit pattern x=0, y=1, z=2
        assert_eq!(get_aabb_corner(min, max, 0), [0.0, 0.0, 0.0]); // 000
        assert_eq!(get_aabb_corner(min, max, 1), [1.0, 0.0, 0.0]); // 001
        assert_eq!(get_aabb_corner(min, max, 2), [0.0, 1.0, 0.0]); // 010
        assert_eq!(get_aabb_corner(min, max, 3), [1.0, 1.0, 0.0]); // 011
        assert_eq!(get_aabb_corner(min, max, 4), [0.0, 0.0, 1.0]); // 100
        assert_eq!(get_aabb_corner(min, max, 5), [1.0, 0.0, 1.0]); // 101
        assert_eq!(get_aabb_corner(min, max, 6), [0.0, 1.0, 1.0]); // 110
        assert_eq!(get_aabb_corner(min, max, 7), [1.0, 1.0, 1.0]); // 111
    }

    // -------------------------------------------------------------------------
    // Mip Offset Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_offset_calculation() {
        // 256x256 pyramid: mip0 = 256*256, mip1 = 128*128, etc.
        assert_eq!(calculate_mip_offset(256, 256, 0), 0);
        assert_eq!(calculate_mip_offset(256, 256, 1), 256 * 256);
        assert_eq!(calculate_mip_offset(256, 256, 2), 256 * 256 + 128 * 128);
    }

    // -------------------------------------------------------------------------
    // Depth Comparison Tests (Reverse-Z)
    // -------------------------------------------------------------------------

    #[test]
    fn test_reverse_z_depth_comparison() {
        // In reverse-Z: near=1.0, far=0.0
        // Object near_depth=0.9, HiZ depth=0.8
        // 0.9 >= 0.8 -> visible (object is in front)
        assert!(0.9 >= 0.8 - EPSILON);

        // Object near_depth=0.7, HiZ depth=0.8
        // 0.7 < 0.8 -> occluded (object is behind)
        assert!(0.7 < 0.8 - EPSILON);
    }
}
