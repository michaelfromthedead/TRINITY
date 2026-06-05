//! Shadow Tile Info GPU Buffer Builder.
//!
//! Builds GPU buffers from ShadowAtlas allocations for shader consumption.
//! Contains extended shadow tile information including light-space matrices,
//! cascade indices, and PCF filter parameters.
//!
//! # Usage
//!
//! ```ignore
//! let mut buffer = ShadowTileBuffer::new(&device, 64);
//!
//! // Update from atlas state + light matrices
//! buffer.update(&queue, &atlas, &light_matrices, &light_configs);
//!
//! // Use buffer in render pass
//! render_pass.set_bind_group(2, &shadow_bind_group, &[]);
//! ```

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use crate::shadow_atlas::ShadowAtlas;

// ---------------------------------------------------------------------------
// GPU-Compatible Structs
// ---------------------------------------------------------------------------

/// Extended shadow tile information for GPU consumption.
///
/// Contains all data needed by shaders for shadow map sampling:
/// - UV transformation for atlas lookup
/// - Light-space matrix for world-to-shadow coordinate transform
/// - Cascade index for CSM selection
/// - PCF filter parameters for soft shadows
///
/// This struct is 96 bytes with proper alignment for GPU access.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct ShadowTileInfoGpu {
    /// Atlas UV offset [u, v] - normalized offset into shadow atlas.
    pub uv_offset: [f32; 2],
    /// Atlas UV scale [u_scale, v_scale] - normalized size in shadow atlas.
    pub uv_scale: [f32; 2],
    /// World-to-light-clip-space matrix (column-major, 4x4).
    /// Transforms world positions into shadow map NDC coordinates.
    pub light_space_matrix: [[f32; 4]; 4],
    /// Cascade index for Cascaded Shadow Maps (0-3).
    /// For non-CSM lights, this is always 0.
    pub cascade_index: u32,
    /// PCF kernel size in texels.
    /// Controls the blur radius for percentage-closer filtering.
    pub filter_size: f32,
    /// Constant depth bias to prevent shadow acne.
    /// Added directly to shadow map depth comparison.
    pub bias_constant: f32,
    /// Slope-scaled depth bias multiplier.
    /// Scales with surface slope relative to light direction.
    pub bias_slope: f32,
}

// Compile-time size assertion: 2*4 + 2*4 + 16*4 + 4 + 4 + 4 + 4 = 96 bytes
const _: () = assert!(std::mem::size_of::<ShadowTileInfoGpu>() == 96);

impl ShadowTileInfoGpu {
    /// Create a new shadow tile info with default values.
    pub fn new(
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
        light_space_matrix: [[f32; 4]; 4],
        cascade_index: u32,
        filter_size: f32,
        bias_constant: f32,
        bias_slope: f32,
    ) -> Self {
        Self {
            uv_offset,
            uv_scale,
            light_space_matrix,
            cascade_index,
            filter_size,
            bias_constant,
            bias_slope,
        }
    }

    /// Create an identity matrix (4x4 column-major).
    pub fn identity_matrix() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Create a shadow tile info with identity matrix and default bias values.
    pub fn with_atlas_coords(uv_offset: [f32; 2], uv_scale: [f32; 2]) -> Self {
        Self {
            uv_offset,
            uv_scale,
            light_space_matrix: Self::identity_matrix(),
            cascade_index: 0,
            filter_size: 1.0,
            bias_constant: 0.005,
            bias_slope: 1.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Light Configuration
// ---------------------------------------------------------------------------

/// Per-light shadow configuration parameters.
///
/// Contains settings that control how shadows are rendered and filtered
/// for a specific light source.
#[derive(Debug, Clone, Copy)]
pub struct ShadowLightConfig {
    /// Cascade index for CSM (0-3). Use 0 for non-directional lights.
    pub cascade_index: u32,
    /// PCF filter size in texels. Larger values = softer shadows.
    pub filter_size: f32,
    /// Constant depth bias to prevent self-shadowing artifacts.
    pub bias_constant: f32,
    /// Slope-scaled bias multiplier for angled surfaces.
    pub bias_slope: f32,
}

impl Default for ShadowLightConfig {
    fn default() -> Self {
        Self {
            cascade_index: 0,
            filter_size: 1.0,
            bias_constant: 0.005,
            bias_slope: 1.5,
        }
    }
}

impl ShadowLightConfig {
    /// Create a new light config with specified parameters.
    pub fn new(
        cascade_index: u32,
        filter_size: f32,
        bias_constant: f32,
        bias_slope: f32,
    ) -> Self {
        Self {
            cascade_index,
            filter_size,
            bias_constant,
            bias_slope,
        }
    }

    /// Create a config for CSM cascade with default bias values.
    pub fn cascade(index: u32) -> Self {
        Self {
            cascade_index: index,
            ..Default::default()
        }
    }

    /// Create a config with custom filter size.
    pub fn with_filter_size(filter_size: f32) -> Self {
        Self {
            filter_size,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// 4x4 Matrix Type (Column-Major)
// ---------------------------------------------------------------------------

/// Simple 4x4 matrix type for light-space transforms (column-major).
///
/// This is a minimal matrix type used for passing light-space matrices
/// to the shadow tile buffer. For full matrix math, use a proper math library.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// Column-major matrix data.
    pub cols: [[f32; 4]; 4],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat4 {
    /// Create an identity matrix.
    pub fn identity() -> Self {
        Self {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a matrix from column-major data.
    pub fn from_cols(cols: [[f32; 4]; 4]) -> Self {
        Self { cols }
    }

    /// Create a matrix from a flat array (column-major order).
    pub fn from_cols_array(data: [f32; 16]) -> Self {
        Self {
            cols: [
                [data[0], data[1], data[2], data[3]],
                [data[4], data[5], data[6], data[7]],
                [data[8], data[9], data[10], data[11]],
                [data[12], data[13], data[14], data[15]],
            ],
        }
    }

    /// Convert to GPU-compatible column-major array.
    pub fn to_cols_array(&self) -> [[f32; 4]; 4] {
        self.cols
    }

    /// Create a perspective projection matrix.
    /// fovy: vertical field of view in radians
    /// aspect: width / height
    /// near: near clipping plane
    /// far: far clipping plane
    pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fovy / 2.0).tan();
        let nf = 1.0 / (near - far);

        Self {
            cols: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) * nf, -1.0],
                [0.0, 0.0, 2.0 * far * near * nf, 0.0],
            ],
        }
    }

    /// Create an orthographic projection matrix.
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rml = right - left;
        let tmb = top - bottom;
        let fmn = far - near;

        Self {
            cols: [
                [2.0 / rml, 0.0, 0.0, 0.0],
                [0.0, 2.0 / tmb, 0.0, 0.0],
                [0.0, 0.0, -2.0 / fmn, 0.0],
                [-(right + left) / rml, -(top + bottom) / tmb, -(far + near) / fmn, 1.0],
            ],
        }
    }

    /// Create a look-at view matrix.
    pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        let f = normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);
        let s = normalize(cross(f, up));
        let u = cross(s, f);

        Self {
            cols: [
                [s[0], u[0], -f[0], 0.0],
                [s[1], u[1], -f[1], 0.0],
                [s[2], u[2], -f[2], 0.0],
                [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
            ],
        }
    }

    /// Multiply two matrices.
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut result = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.cols[0][j] * rhs.cols[i][0]
                    + self.cols[1][j] * rhs.cols[i][1]
                    + self.cols[2][j] * rhs.cols[i][2]
                    + self.cols[3][j] * rhs.cols[i][3];
            }
        }
        Mat4::from_cols(result)
    }
}

// Helper vector math functions
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---------------------------------------------------------------------------
// Shadow Tile Buffer
// ---------------------------------------------------------------------------

/// Minimum buffer size in bytes (wgpu requires non-zero buffers).
const MIN_BUFFER_SIZE: u64 = 96; // At least one ShadowTileInfoGpu

/// GPU buffer containing shadow tile information for all active shadow casters.
///
/// The buffer is indexed by light_id for shader lookup. Entries are stored
/// in a sparse array where index corresponds to light_id.
pub struct ShadowTileBuffer {
    /// GPU storage buffer containing ShadowTileInfoGpu array.
    buffer: wgpu::Buffer,
    /// Maximum number of shadow casters this buffer can hold.
    capacity: usize,
    /// Current number of valid entries in the buffer.
    count: usize,
    /// CPU-side cache of tile info for partial updates.
    cpu_data: Vec<ShadowTileInfoGpu>,
    /// Tracks which light_ids have valid data.
    active_lights: HashMap<u32, usize>,
}

impl ShadowTileBuffer {
    /// Create a new shadow tile buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for buffer creation.
    /// * `max_shadow_casters` - Maximum number of shadow-casting lights supported.
    ///
    /// # Returns
    ///
    /// A new `ShadowTileBuffer` with pre-allocated GPU storage.
    pub fn new(device: &wgpu::Device, max_shadow_casters: usize) -> Self {
        let capacity = max_shadow_casters.max(1);
        let buffer_size = (capacity * std::mem::size_of::<ShadowTileInfoGpu>()) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Tile Info Buffer"),
            size: buffer_size.max(MIN_BUFFER_SIZE),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            count: 0,
            cpu_data: vec![ShadowTileInfoGpu::default(); capacity],
            active_lights: HashMap::new(),
        }
    }

    /// Update the buffer from atlas allocations and light configurations.
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu queue for buffer upload.
    /// * `atlas` - Shadow atlas containing tile allocations.
    /// * `light_matrices` - Map from light_id to world-to-light-clip matrix.
    /// * `light_configs` - Map from light_id to shadow configuration.
    ///
    /// # Notes
    ///
    /// Lights without entries in `light_matrices` or `light_configs` will use
    /// default values (identity matrix, default bias settings).
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        atlas: &ShadowAtlas,
        light_matrices: &HashMap<u32, Mat4>,
        light_configs: &HashMap<u32, ShadowLightConfig>,
    ) {
        self.active_lights.clear();
        self.count = 0;

        // Build tile info for each allocated tile
        for tile in atlas.allocated_tiles() {
            let light_id = tile.light_id;

            // Get light-space matrix (default to identity if not provided)
            let matrix = light_matrices
                .get(&light_id)
                .map(|m| m.to_cols_array())
                .unwrap_or_else(ShadowTileInfoGpu::identity_matrix);

            // Get light config (default if not provided)
            let config = light_configs
                .get(&light_id)
                .copied()
                .unwrap_or_default();

            let tile_info = ShadowTileInfoGpu {
                uv_offset: tile.uv_offset,
                uv_scale: [tile.uv_scale, tile.uv_scale],
                light_space_matrix: matrix,
                cascade_index: config.cascade_index,
                filter_size: config.filter_size,
                bias_constant: config.bias_constant,
                bias_slope: config.bias_slope,
            };

            // Store in CPU cache at index = count
            if self.count < self.capacity {
                self.cpu_data[self.count] = tile_info;
                self.active_lights.insert(light_id, self.count);
                self.count += 1;
            }
        }

        // Upload to GPU
        if self.count > 0 {
            let data_bytes = bytemuck::cast_slice(&self.cpu_data[..self.count]);
            queue.write_buffer(&self.buffer, 0, data_bytes);
        }
    }

    /// Update a single light's tile info without full rebuild.
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu queue for buffer upload.
    /// * `light_id` - The light to update.
    /// * `tile_info` - New tile information.
    ///
    /// # Returns
    ///
    /// `true` if the light was found and updated, `false` otherwise.
    pub fn update_single(
        &mut self,
        queue: &wgpu::Queue,
        light_id: u32,
        tile_info: ShadowTileInfoGpu,
    ) -> bool {
        if let Some(&index) = self.active_lights.get(&light_id) {
            self.cpu_data[index] = tile_info;

            // Upload just this entry
            let offset = (index * std::mem::size_of::<ShadowTileInfoGpu>()) as u64;
            let data_bytes = bytemuck::bytes_of(&tile_info);
            queue.write_buffer(&self.buffer, offset, data_bytes);

            true
        } else {
            false
        }
    }

    /// Get the GPU buffer.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the current number of valid entries.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the buffer index for a light_id, if it exists.
    pub fn get_index(&self, light_id: u32) -> Option<usize> {
        self.active_lights.get(&light_id).copied()
    }

    /// Get tile info for a light_id (CPU-side).
    pub fn get_tile_info(&self, light_id: u32) -> Option<&ShadowTileInfoGpu> {
        self.active_lights
            .get(&light_id)
            .map(|&idx| &self.cpu_data[idx])
    }

    /// Check if a light_id has an entry in the buffer.
    pub fn contains(&self, light_id: u32) -> bool {
        self.active_lights.contains_key(&light_id)
    }

    /// Get all active light IDs.
    pub fn active_light_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.active_lights.keys().copied()
    }

    /// Resize the buffer to a new capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for buffer creation.
    /// * `new_capacity` - New maximum number of shadow casters.
    ///
    /// # Notes
    ///
    /// This creates a new buffer; existing data is preserved up to the
    /// minimum of old and new capacity.
    pub fn resize(&mut self, device: &wgpu::Device, new_capacity: usize) {
        let new_capacity = new_capacity.max(1);
        if new_capacity == self.capacity {
            return;
        }

        let buffer_size = (new_capacity * std::mem::size_of::<ShadowTileInfoGpu>()) as u64;

        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Tile Info Buffer"),
            size: buffer_size.max(MIN_BUFFER_SIZE),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Resize CPU data, preserving existing entries
        self.cpu_data.resize(new_capacity, ShadowTileInfoGpu::default());
        self.capacity = new_capacity;

        // Truncate count if necessary
        if self.count > new_capacity {
            self.count = new_capacity;
            // Remove entries beyond new capacity
            self.active_lights.retain(|_, &mut idx| idx < new_capacity);
        }
    }

    /// Clear all entries from the buffer.
    pub fn clear(&mut self) {
        self.count = 0;
        self.active_lights.clear();
    }

    /// Get the GPU buffer size in bytes.
    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow_atlas::{ShadowAtlas, TileSizeTier};

    // Test struct sizes
    #[test]
    fn test_shadow_tile_info_gpu_size_is_96_bytes() {
        assert_eq!(std::mem::size_of::<ShadowTileInfoGpu>(), 96);
    }

    #[test]
    fn test_shadow_tile_info_gpu_alignment() {
        // Should be 4-byte aligned (f32 alignment)
        assert_eq!(std::mem::align_of::<ShadowTileInfoGpu>(), 4);
    }

    #[test]
    fn test_shadow_tile_info_gpu_default() {
        let info = ShadowTileInfoGpu::default();
        assert_eq!(info.uv_offset, [0.0, 0.0]);
        assert_eq!(info.uv_scale, [0.0, 0.0]);
        assert_eq!(info.cascade_index, 0);
        assert_eq!(info.filter_size, 0.0);
        assert_eq!(info.bias_constant, 0.0);
        assert_eq!(info.bias_slope, 0.0);
    }

    #[test]
    fn test_shadow_tile_info_gpu_new() {
        let matrix = ShadowTileInfoGpu::identity_matrix();
        let info = ShadowTileInfoGpu::new(
            [0.25, 0.5],
            [0.125, 0.125],
            matrix,
            2,
            3.0,
            0.001,
            2.0,
        );

        assert_eq!(info.uv_offset, [0.25, 0.5]);
        assert_eq!(info.uv_scale, [0.125, 0.125]);
        assert_eq!(info.cascade_index, 2);
        assert_eq!(info.filter_size, 3.0);
        assert_eq!(info.bias_constant, 0.001);
        assert_eq!(info.bias_slope, 2.0);
    }

    #[test]
    fn test_shadow_tile_info_gpu_with_atlas_coords() {
        let info = ShadowTileInfoGpu::with_atlas_coords([0.1, 0.2], [0.3, 0.3]);

        assert_eq!(info.uv_offset, [0.1, 0.2]);
        assert_eq!(info.uv_scale, [0.3, 0.3]);
        assert_eq!(info.cascade_index, 0);
        assert!((info.filter_size - 1.0).abs() < 0.0001);
        assert!((info.bias_constant - 0.005).abs() < 0.0001);
        assert!((info.bias_slope - 1.5).abs() < 0.0001);
    }

    #[test]
    fn test_identity_matrix() {
        let m = ShadowTileInfoGpu::identity_matrix();
        assert_eq!(m[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(m[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_shadow_light_config_default() {
        let config = ShadowLightConfig::default();
        assert_eq!(config.cascade_index, 0);
        assert!((config.filter_size - 1.0).abs() < 0.0001);
        assert!((config.bias_constant - 0.005).abs() < 0.0001);
        assert!((config.bias_slope - 1.5).abs() < 0.0001);
    }

    #[test]
    fn test_shadow_light_config_new() {
        let config = ShadowLightConfig::new(3, 5.0, 0.01, 3.0);
        assert_eq!(config.cascade_index, 3);
        assert_eq!(config.filter_size, 5.0);
        assert_eq!(config.bias_constant, 0.01);
        assert_eq!(config.bias_slope, 3.0);
    }

    #[test]
    fn test_shadow_light_config_cascade() {
        let config = ShadowLightConfig::cascade(2);
        assert_eq!(config.cascade_index, 2);
        // Other values should be default
        assert!((config.filter_size - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_shadow_light_config_with_filter_size() {
        let config = ShadowLightConfig::with_filter_size(7.0);
        assert_eq!(config.filter_size, 7.0);
        assert_eq!(config.cascade_index, 0);
    }

    #[test]
    fn test_mat4_identity() {
        let m = Mat4::identity();
        assert_eq!(m.cols[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.cols[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(m.cols[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(m.cols[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mat4_from_cols() {
        let cols = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let m = Mat4::from_cols(cols);
        assert_eq!(m.cols, cols);
    }

    #[test]
    fn test_mat4_from_cols_array() {
        let data: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let m = Mat4::from_cols_array(data);
        assert_eq!(m.cols[0], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.cols[1], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mat4_to_cols_array() {
        let m = Mat4::identity();
        let cols = m.to_cols_array();
        assert_eq!(cols, ShadowTileInfoGpu::identity_matrix());
    }

    #[test]
    fn test_mat4_multiply_identity() {
        let a = Mat4::identity();
        let b = Mat4::from_cols([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ]);

        let result = a.mul(&b);
        assert_eq!(result.cols, b.cols);

        let result2 = b.mul(&a);
        assert_eq!(result2.cols, b.cols);
    }

    #[test]
    fn test_bytemuck_pod_zeroable() {
        // Verify ShadowTileInfoGpu is Pod and Zeroable
        let zeroed: ShadowTileInfoGpu = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.uv_offset, [0.0, 0.0]);
        assert_eq!(zeroed.uv_scale, [0.0, 0.0]);
        assert_eq!(zeroed.cascade_index, 0);

        // Verify we can cast to bytes
        let info = ShadowTileInfoGpu::with_atlas_coords([0.5, 0.25], [0.25, 0.25]);
        let bytes: &[u8] = bytemuck::bytes_of(&info);
        assert_eq!(bytes.len(), 96);
    }

    #[test]
    fn test_bytemuck_cast_slice() {
        let infos = vec![
            ShadowTileInfoGpu::with_atlas_coords([0.0, 0.0], [0.5, 0.5]),
            ShadowTileInfoGpu::with_atlas_coords([0.5, 0.0], [0.5, 0.5]),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&infos);
        assert_eq!(bytes.len(), 192); // 2 * 96
    }

    // GPU buffer tests (require wgpu device - use pollster for async)
    // These tests are skipped if no GPU adapter is available (e.g., in CI)
    fn try_create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()
        })
    }

    /// Macro to skip GPU tests when no adapter is available
    macro_rules! require_gpu {
        () => {
            match try_create_test_device() {
                Some(device_queue) => device_queue,
                None => {
                    eprintln!("Skipping test: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn test_buffer_creation_succeeds() {
        let (device, _queue) = require_gpu!();
        let buffer = ShadowTileBuffer::new(&device, 64);

        assert_eq!(buffer.capacity(), 64);
        assert_eq!(buffer.count(), 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.buffer_size(), 64 * 96);
    }

    #[test]
    fn test_buffer_creation_with_zero_capacity() {
        let (device, _queue) = require_gpu!();
        let buffer = ShadowTileBuffer::new(&device, 0);

        // Should default to capacity of 1
        assert_eq!(buffer.capacity(), 1);
        assert_eq!(buffer.buffer_size(), 96);
    }

    #[test]
    fn test_update_populates_from_atlas() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        // Allocate some tiles
        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        atlas.allocate(TileSizeTier::Large1024, 2, 100);
        atlas.allocate(TileSizeTier::Large1024, 3, 100);

        // Create light matrices and configs
        let mut light_matrices = HashMap::new();
        light_matrices.insert(1, Mat4::identity());
        light_matrices.insert(2, Mat4::perspective(1.0, 1.0, 0.1, 100.0));

        let mut light_configs = HashMap::new();
        light_configs.insert(1, ShadowLightConfig::cascade(0));
        light_configs.insert(2, ShadowLightConfig::new(1, 3.0, 0.002, 2.0));

        buffer.update(&queue, &atlas, &light_matrices, &light_configs);

        assert_eq!(buffer.count(), 3);
        assert!(buffer.contains(1));
        assert!(buffer.contains(2));
        assert!(buffer.contains(3));
    }

    #[test]
    fn test_light_matrices_correctly_packed() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 42, 100);

        let test_matrix = Mat4::from_cols([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let mut light_matrices = HashMap::new();
        light_matrices.insert(42, test_matrix);

        let light_configs = HashMap::new();

        buffer.update(&queue, &atlas, &light_matrices, &light_configs);

        let info = buffer.get_tile_info(42).unwrap();
        assert_eq!(info.light_space_matrix, test_matrix.cols);
    }

    #[test]
    fn test_empty_atlas_produces_empty_buffer() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let atlas = ShadowAtlas::new(4096); // Empty atlas

        let light_matrices = HashMap::new();
        let light_configs = HashMap::new();

        buffer.update(&queue, &atlas, &light_matrices, &light_configs);

        assert_eq!(buffer.count(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_resize_increases_capacity() {
        let (device, _queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 16);

        assert_eq!(buffer.capacity(), 16);

        buffer.resize(&device, 128);

        assert_eq!(buffer.capacity(), 128);
        assert_eq!(buffer.buffer_size(), 128 * 96);
    }

    #[test]
    fn test_buffer_resize_decreases_capacity() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        // Add 10 lights
        for i in 0..10 {
            atlas.allocate(TileSizeTier::Large1024, i, 100);
        }

        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());
        assert_eq!(buffer.count(), 10);

        // Shrink below current count
        buffer.resize(&device, 5);

        assert_eq!(buffer.capacity(), 5);
        assert_eq!(buffer.count(), 5);
    }

    #[test]
    fn test_buffer_clear() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        assert_eq!(buffer.count(), 1);

        buffer.clear();

        assert_eq!(buffer.count(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.contains(1));
    }

    #[test]
    fn test_get_index() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 42, 100);
        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        assert!(buffer.get_index(42).is_some());
        assert!(buffer.get_index(999).is_none());
    }

    #[test]
    fn test_active_light_ids() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 10, 100);
        atlas.allocate(TileSizeTier::Large1024, 20, 100);
        atlas.allocate(TileSizeTier::Large1024, 30, 100);

        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        let ids: Vec<_> = buffer.active_light_ids().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert!(ids.contains(&30));
    }

    #[test]
    fn test_update_single() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);
        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        let new_info = ShadowTileInfoGpu::new(
            [0.1, 0.2],
            [0.3, 0.3],
            ShadowTileInfoGpu::identity_matrix(),
            2,
            5.0,
            0.01,
            3.0,
        );

        assert!(buffer.update_single(&queue, 1, new_info));

        let updated = buffer.get_tile_info(1).unwrap();
        assert_eq!(updated.cascade_index, 2);
        assert_eq!(updated.filter_size, 5.0);
    }

    #[test]
    fn test_update_single_nonexistent() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);

        let info = ShadowTileInfoGpu::default();
        assert!(!buffer.update_single(&queue, 999, info));
    }

    #[test]
    fn test_default_values_for_missing_configs() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        atlas.allocate(TileSizeTier::Large1024, 1, 100);

        // Update with empty configs
        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        let info = buffer.get_tile_info(1).unwrap();

        // Should have identity matrix
        assert_eq!(info.light_space_matrix, ShadowTileInfoGpu::identity_matrix());

        // Should have default config values
        assert_eq!(info.cascade_index, 0);
        assert!((info.filter_size - 1.0).abs() < 0.0001);
        assert!((info.bias_constant - 0.005).abs() < 0.0001);
        assert!((info.bias_slope - 1.5).abs() < 0.0001);
    }

    #[test]
    fn test_uv_values_from_atlas() {
        let (device, queue) = require_gpu!();
        let mut buffer = ShadowTileBuffer::new(&device, 64);
        let mut atlas = ShadowAtlas::new(4096);

        let tile = atlas.allocate(TileSizeTier::Large1024, 1, 100).unwrap();
        buffer.update(&queue, &atlas, &HashMap::new(), &HashMap::new());

        let info = buffer.get_tile_info(1).unwrap();

        // UV values should match the atlas tile
        assert_eq!(info.uv_offset, tile.uv_offset);
        assert_eq!(info.uv_scale, [tile.uv_scale, tile.uv_scale]);
    }
}
