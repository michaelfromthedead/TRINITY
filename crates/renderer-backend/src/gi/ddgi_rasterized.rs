//! Rasterized DDGI probe update using 6-face cubemap rendering.
//!
//! This module provides a fallback for DDGI probe updates when hardware
//! ray tracing is unavailable. Instead of ray queries, it renders a 6-face
//! cubemap for each probe and projects the resulting radiance to L2 spherical
//! harmonics coefficients.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DDGIRasterizedPass                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  1. create_probe_atlas()      → Allocate batched probe atlas    │
//! │  2. get_face_view_matrices()  → 6 cubemap face view matrices    │
//! │  3. create_probe_render_pass()→ Graphics pass for probe batch   │
//! │  4. create_sh_projection_pass()→ Compute pass: atlas → ProbeSH  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Atlas Layout
//!
//! Probes are batched into a 2D atlas texture for efficient rendering:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │  Probe 0  │  Probe 1  │  Probe 2  │ ... │  Probe 7   │ Row 0
//! │  (6 faces)│  (6 faces)│  (6 faces)│     │  (6 faces) │
//! ├───────────┼───────────┼───────────┼─────┼────────────┤
//! │  Probe 8  │  Probe 9  │ Probe 10  │ ... │  Probe 15  │ Row 1
//! │  (6 faces)│  (6 faces)│  (6 faces)│     │  (6 faces) │
//! ├───────────┼───────────┼───────────┼─────┼────────────┤
//! │    ...    │    ...    │    ...    │ ... │    ...     │ ...
//! ├───────────┼───────────┼───────────┼─────┼────────────┤
//! │ Probe 56  │ Probe 57  │ Probe 58  │ ... │  Probe 63  │ Row 7
//! │  (6 faces)│  (6 faces)│  (6 faces)│     │  (6 faces) │
//! └────────────────────────────────────────────────────────┘
//!
//! Each probe cell = 6 faces arranged 3x2:
//! ┌──────┬──────┬──────┐
//! │  +X  │  -X  │  +Y  │
//! ├──────┼──────┼──────┤
//! │  -Y  │  +Z  │  -Z  │
//! └──────┴──────┴──────┘
//! ```
//!
//! # Memory Budget
//!
//! | Config              | Face Res | Atlas Size   | Memory      |
//! |---------------------|----------|--------------|-------------|
//! | Default (64 probes) | 32x32    | 2048x1536    | ~12 MB      |
//! | Large (256 probes)  | 32x32    | 4096x3072    | ~48 MB      |
//! | Quality (64 probes) | 64x64    | 4096x3072    | ~48 MB      |
//!
//! # Cubemap Face Conventions
//!
//! Standard OpenGL cubemap face ordering with Y-down for consistency with
//! typical render coordinate systems:
//!
//! | Face | Direction | Up Vector |
//! |------|-----------|-----------|
//! | +X   | (+1,0,0)  | (0,-1,0)  |
//! | -X   | (-1,0,0)  | (0,-1,0)  |
//! | +Y   | (0,+1,0)  | (0,0,+1)  |
//! | -Y   | (0,-1,0)  | (0,0,-1)  |
//! | +Z   | (0,0,+1)  | (0,-1,0)  |
//! | -Z   | (0,0,-1)  | (0,-1,0)  |

use crate::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, DepthStencilAttachment,
    DispatchSource, InstanceSource, IrPass, PassIndex, ResourceHandle, ViewType,
};

// ============================================================================
// Constants
// ============================================================================

/// Number of faces in a cubemap.
pub const CUBEMAP_FACES: usize = 6;

/// Default face resolution (32x32 per face).
pub const DEFAULT_FACE_RESOLUTION: u32 = 32;

/// Default probe atlas size (8x8 = 64 probes per batch).
pub const DEFAULT_ATLAS_SIZE: u32 = 8;

/// Number of faces per row in a probe cell (3).
pub const FACES_PER_ROW: u32 = 3;

/// Number of face rows in a probe cell (2).
pub const FACE_ROWS: u32 = 2;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the rasterized DDGI pass.
#[derive(Clone, Debug)]
pub struct DDGIRasterizedConfig {
    /// Face resolution (pixels per edge, must be power of 2).
    pub face_resolution: u32,
    /// Atlas grid dimensions (probes per side, e.g., 8 = 64 probes).
    pub atlas_size: u32,
    /// Number of probes to update per frame.
    pub probes_per_frame: usize,
    /// Enable depth testing during probe rendering.
    pub depth_test_enabled: bool,
    /// Near plane distance for probe rendering.
    pub near_plane: f32,
    /// Far plane distance for probe rendering.
    pub far_plane: f32,
}

impl Default for DDGIRasterizedConfig {
    fn default() -> Self {
        Self {
            face_resolution: DEFAULT_FACE_RESOLUTION,
            atlas_size: DEFAULT_ATLAS_SIZE,
            probes_per_frame: 64,
            depth_test_enabled: true,
            near_plane: 0.01,
            far_plane: 100.0,
        }
    }
}

impl DDGIRasterizedConfig {
    /// Create a low-quality configuration (16x16 faces, 32 probes/frame).
    pub fn low() -> Self {
        Self {
            face_resolution: 16,
            atlas_size: 8,
            probes_per_frame: 32,
            ..Default::default()
        }
    }

    /// Create a medium-quality configuration (32x32 faces, 64 probes/frame).
    pub fn medium() -> Self {
        Self::default()
    }

    /// Create a high-quality configuration (64x64 faces, 64 probes/frame).
    pub fn high() -> Self {
        Self {
            face_resolution: 64,
            atlas_size: 8,
            probes_per_frame: 64,
            ..Default::default()
        }
    }

    /// Calculate the width of a single probe cell in pixels.
    #[inline]
    pub fn probe_cell_width(&self) -> u32 {
        self.face_resolution * FACES_PER_ROW
    }

    /// Calculate the height of a single probe cell in pixels.
    #[inline]
    pub fn probe_cell_height(&self) -> u32 {
        self.face_resolution * FACE_ROWS
    }

    /// Calculate total atlas texture width.
    #[inline]
    pub fn atlas_width(&self) -> u32 {
        self.probe_cell_width() * self.atlas_size
    }

    /// Calculate total atlas texture height.
    #[inline]
    pub fn atlas_height(&self) -> u32 {
        self.probe_cell_height() * self.atlas_size
    }

    /// Total number of probes that fit in one atlas.
    #[inline]
    pub fn probes_per_atlas(&self) -> u32 {
        self.atlas_size * self.atlas_size
    }

    /// Estimated GPU memory for atlas (RGBA16F).
    #[inline]
    pub fn estimated_atlas_memory(&self) -> usize {
        let pixels = self.atlas_width() as usize * self.atlas_height() as usize;
        pixels * 8 // RGBA16F = 8 bytes per pixel
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.face_resolution.is_power_of_two() {
            return Err("face_resolution must be power of 2");
        }
        if self.face_resolution < 8 || self.face_resolution > 128 {
            return Err("face_resolution must be in range [8, 128]");
        }
        if self.atlas_size < 1 || self.atlas_size > 32 {
            return Err("atlas_size must be in range [1, 32]");
        }
        if self.probes_per_frame == 0 {
            return Err("probes_per_frame must be > 0");
        }
        if self.near_plane <= 0.0 || self.near_plane >= self.far_plane {
            return Err("near_plane must be > 0 and < far_plane");
        }
        Ok(())
    }
}

// ============================================================================
// Cubemap Face Index
// ============================================================================

/// Cubemap face index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CubemapFace {
    /// Positive X (+1, 0, 0)
    PosX = 0,
    /// Negative X (-1, 0, 0)
    NegX = 1,
    /// Positive Y (0, +1, 0)
    PosY = 2,
    /// Negative Y (0, -1, 0)
    NegY = 3,
    /// Positive Z (0, 0, +1)
    PosZ = 4,
    /// Negative Z (0, 0, -1)
    NegZ = 5,
}

impl CubemapFace {
    /// All cubemap faces in order.
    pub const ALL: [CubemapFace; 6] = [
        CubemapFace::PosX,
        CubemapFace::NegX,
        CubemapFace::PosY,
        CubemapFace::NegY,
        CubemapFace::PosZ,
        CubemapFace::NegZ,
    ];

    /// Get the face direction vector.
    #[inline]
    pub const fn direction(self) -> [f32; 3] {
        match self {
            CubemapFace::PosX => [1.0, 0.0, 0.0],
            CubemapFace::NegX => [-1.0, 0.0, 0.0],
            CubemapFace::PosY => [0.0, 1.0, 0.0],
            CubemapFace::NegY => [0.0, -1.0, 0.0],
            CubemapFace::PosZ => [0.0, 0.0, 1.0],
            CubemapFace::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// Get the up vector for this face.
    #[inline]
    pub const fn up(self) -> [f32; 3] {
        match self {
            CubemapFace::PosX => [0.0, -1.0, 0.0],
            CubemapFace::NegX => [0.0, -1.0, 0.0],
            CubemapFace::PosY => [0.0, 0.0, 1.0],
            CubemapFace::NegY => [0.0, 0.0, -1.0],
            CubemapFace::PosZ => [0.0, -1.0, 0.0],
            CubemapFace::NegZ => [0.0, -1.0, 0.0],
        }
    }

    /// Get the position within a probe cell (column, row).
    #[inline]
    pub const fn cell_position(self) -> (u32, u32) {
        match self {
            CubemapFace::PosX => (0, 0),
            CubemapFace::NegX => (1, 0),
            CubemapFace::PosY => (2, 0),
            CubemapFace::NegY => (0, 1),
            CubemapFace::PosZ => (1, 1),
            CubemapFace::NegZ => (2, 1),
        }
    }

    /// Convert index to face.
    #[inline]
    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(CubemapFace::PosX),
            1 => Some(CubemapFace::NegX),
            2 => Some(CubemapFace::PosY),
            3 => Some(CubemapFace::NegY),
            4 => Some(CubemapFace::PosZ),
            5 => Some(CubemapFace::NegZ),
            _ => None,
        }
    }
}

// ============================================================================
// View Matrix Generation
// ============================================================================

/// A 4x4 matrix stored in column-major order.
pub type Mat4 = [[f32; 4]; 4];

/// Normalize a 3D vector.
#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Cross product of two 3D vectors.
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3D vectors.
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Create a look-at view matrix.
///
/// Creates a right-handed view matrix that looks from `eye` towards `target`
/// with the given `up` vector.
pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Create a 90-degree perspective projection matrix for cubemap rendering.
pub fn cubemap_projection(near: f32, far: f32) -> Mat4 {
    // FOV = 90 degrees, aspect = 1:1
    let f = 1.0; // tan(45 deg) = 1
    let nf = 1.0 / (near - far);

    [
        [f, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

/// Get view matrix for a specific cubemap face at a given probe position.
pub fn get_face_view_matrix(probe_pos: [f32; 3], face: CubemapFace) -> Mat4 {
    let dir = face.direction();
    let up = face.up();
    let target = [
        probe_pos[0] + dir[0],
        probe_pos[1] + dir[1],
        probe_pos[2] + dir[2],
    ];
    look_at(probe_pos, target, up)
}

/// Get view matrices for all 6 cubemap faces at a probe position.
///
/// Returns an array of 6 4x4 matrices in column-major order,
/// one for each cubemap face in standard order (+X, -X, +Y, -Y, +Z, -Z).
pub fn get_face_view_matrices(probe_pos: [f32; 3]) -> [Mat4; 6] {
    [
        get_face_view_matrix(probe_pos, CubemapFace::PosX),
        get_face_view_matrix(probe_pos, CubemapFace::NegX),
        get_face_view_matrix(probe_pos, CubemapFace::PosY),
        get_face_view_matrix(probe_pos, CubemapFace::NegY),
        get_face_view_matrix(probe_pos, CubemapFace::PosZ),
        get_face_view_matrix(probe_pos, CubemapFace::NegZ),
    ]
}

/// Flatten a Mat4 to a [f32; 16] array (column-major).
#[inline]
pub fn mat4_to_array(m: Mat4) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]
}

/// Get view matrices as flat [f32; 16] arrays for GPU upload.
pub fn get_face_view_matrices_flat(probe_pos: [f32; 3]) -> [[f32; 16]; 6] {
    let matrices = get_face_view_matrices(probe_pos);
    [
        mat4_to_array(matrices[0]),
        mat4_to_array(matrices[1]),
        mat4_to_array(matrices[2]),
        mat4_to_array(matrices[3]),
        mat4_to_array(matrices[4]),
        mat4_to_array(matrices[5]),
    ]
}

// ============================================================================
// Atlas UV Computation
// ============================================================================

/// Atlas UV coordinates for a probe face.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtlasUV {
    /// Minimum U coordinate (left edge).
    pub u_min: f32,
    /// Maximum U coordinate (right edge).
    pub u_max: f32,
    /// Minimum V coordinate (top edge).
    pub v_min: f32,
    /// Maximum V coordinate (bottom edge).
    pub v_max: f32,
}

impl AtlasUV {
    /// Create UV coordinates for a probe face within an atlas.
    ///
    /// # Arguments
    /// * `probe_index` - Linear index of the probe in the batch (0..atlas_size^2).
    /// * `face` - Which cubemap face.
    /// * `config` - Atlas configuration.
    pub fn from_probe_face(
        probe_index: u32,
        face: CubemapFace,
        config: &DDGIRasterizedConfig,
    ) -> Self {
        let atlas_probes = config.atlas_size;
        let probe_x = probe_index % atlas_probes;
        let probe_y = probe_index / atlas_probes;

        let (face_col, face_row) = face.cell_position();

        // Pixel coordinates within atlas
        let pixel_x = probe_x * config.probe_cell_width() + face_col * config.face_resolution;
        let pixel_y = probe_y * config.probe_cell_height() + face_row * config.face_resolution;

        let atlas_w = config.atlas_width() as f32;
        let atlas_h = config.atlas_height() as f32;

        Self {
            u_min: pixel_x as f32 / atlas_w,
            u_max: (pixel_x + config.face_resolution) as f32 / atlas_w,
            v_min: pixel_y as f32 / atlas_h,
            v_max: (pixel_y + config.face_resolution) as f32 / atlas_h,
        }
    }

    /// Get pixel bounds (x_min, y_min, x_max, y_max) for this UV region.
    pub fn to_pixel_bounds(&self, atlas_width: u32, atlas_height: u32) -> (u32, u32, u32, u32) {
        (
            (self.u_min * atlas_width as f32) as u32,
            (self.v_min * atlas_height as f32) as u32,
            (self.u_max * atlas_width as f32) as u32,
            (self.v_max * atlas_height as f32) as u32,
        )
    }

    /// Center point of this UV region.
    pub fn center(&self) -> (f32, f32) {
        (
            (self.u_min + self.u_max) * 0.5,
            (self.v_min + self.v_max) * 0.5,
        )
    }
}

// ============================================================================
// Batch Scheduling
// ============================================================================

/// Tracks which probes to update this frame.
#[derive(Clone, Debug)]
pub struct ProbeBatchScheduler {
    /// Current batch starting index.
    pub current_batch_start: usize,
    /// Number of probes to update per batch.
    pub probes_per_batch: usize,
    /// Total number of probes in the grid.
    pub total_probes: usize,
    /// Frame counter for round-robin scheduling.
    pub frame_count: u64,
}

impl ProbeBatchScheduler {
    /// Create a new batch scheduler.
    pub fn new(total_probes: usize, probes_per_batch: usize) -> Self {
        Self {
            current_batch_start: 0,
            probes_per_batch: probes_per_batch.min(total_probes).max(1),
            total_probes,
            frame_count: 0,
        }
    }

    /// Advance to the next batch and return the (start, count) for current batch.
    pub fn next_batch(&mut self) -> (usize, usize) {
        let start = self.current_batch_start;
        let count = self.probes_per_batch.min(self.total_probes - start);

        // Advance for next call
        self.current_batch_start = (start + count) % self.total_probes;
        self.frame_count += 1;

        (start, count)
    }

    /// Reset scheduler to beginning.
    pub fn reset(&mut self) {
        self.current_batch_start = 0;
        self.frame_count = 0;
    }

    /// Number of frames required to update all probes once.
    pub fn frames_per_cycle(&self) -> usize {
        (self.total_probes + self.probes_per_batch - 1) / self.probes_per_batch
    }

    /// Check if we're at the start of a new full cycle.
    pub fn is_cycle_start(&self) -> bool {
        self.current_batch_start == 0 && self.frame_count > 0
    }
}

// ============================================================================
// DDGIRasterizedPass
// ============================================================================

/// Rasterized DDGI probe update pass.
///
/// Provides a fallback for hardware ray tracing by rendering 6-face cubemaps
/// for each probe and projecting the captured radiance to spherical harmonics.
#[derive(Clone, Debug)]
pub struct DDGIRasterizedPass {
    /// Configuration for this pass.
    pub config: DDGIRasterizedConfig,
    /// Batch scheduler for temporal amortization.
    pub scheduler: ProbeBatchScheduler,
    /// Projection matrix for cubemap rendering.
    pub projection_matrix: Mat4,
}

impl DDGIRasterizedPass {
    /// Create a new rasterized DDGI pass.
    pub fn new(config: DDGIRasterizedConfig, total_probes: usize) -> Result<Self, &'static str> {
        config.validate()?;

        let scheduler = ProbeBatchScheduler::new(total_probes, config.probes_per_frame);
        let projection_matrix = cubemap_projection(config.near_plane, config.far_plane);

        Ok(Self {
            config,
            scheduler,
            projection_matrix,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults(total_probes: usize) -> Self {
        Self::new(DDGIRasterizedConfig::default(), total_probes)
            .expect("default config should be valid")
    }

    /// Get atlas texture dimensions.
    pub fn atlas_dimensions(&self) -> (u32, u32) {
        (self.config.atlas_width(), self.config.atlas_height())
    }

    /// Get view matrices for all 6 faces of a probe.
    pub fn get_probe_view_matrices(&self, probe_pos: [f32; 3]) -> [[f32; 16]; 6] {
        get_face_view_matrices_flat(probe_pos)
    }

    /// Get flattened projection matrix.
    pub fn get_projection_matrix_flat(&self) -> [f32; 16] {
        mat4_to_array(self.projection_matrix)
    }

    /// Calculate UV coordinates for a probe face.
    pub fn get_face_uv(&self, probe_batch_index: u32, face: CubemapFace) -> AtlasUV {
        AtlasUV::from_probe_face(probe_batch_index, face, &self.config)
    }

    /// Get the next batch of probes to update.
    pub fn next_batch(&mut self) -> (usize, usize) {
        self.scheduler.next_batch()
    }

    /// Create descriptor for probe atlas texture.
    ///
    /// Returns (width, height, format description).
    pub fn atlas_texture_desc(&self) -> AtlasTextureDesc {
        AtlasTextureDesc {
            width: self.config.atlas_width(),
            height: self.config.atlas_height(),
            format: AtlasFormat::Rgba16Float,
            usage: AtlasUsage::RenderTarget | AtlasUsage::Sampled,
        }
    }

    /// Create a graphics pass for rendering probes in the current batch.
    ///
    /// This creates a multi-view render pass that renders all 6 faces of all
    /// probes in the batch to the atlas texture.
    pub fn create_probe_render_pass(
        &self,
        pass_index: PassIndex,
        atlas_resource: ResourceHandle,
        depth_resource: ResourceHandle,
        probe_batch_start: usize,
        probe_count: usize,
    ) -> IrPass {
        IrPass::graphics(
            pass_index,
            format!(
                "ddgi_rasterized_probe_batch_{}_{}",
                probe_batch_start, probe_count
            ),
            vec![ColorAttachment {
                resource: atlas_resource,
                mip_level: 0,
                array_layer: 0,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
            }],
            Some(DepthStencilAttachment {
                resource: depth_resource,
                depth_load_op: AttachmentLoadOp::Clear,
                depth_store_op: AttachmentStoreOp::DontCare,
                stencil_load_op: AttachmentLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
                clear_depth: 1.0,
                clear_stencil: 0,
                depth_test_enabled: self.config.depth_test_enabled,
                depth_write_enabled: true,
            }),
            InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        )
    }

    /// Create a compute pass for SH projection (atlas → ProbeSH buffer).
    pub fn create_sh_projection_pass(
        &self,
        pass_index: PassIndex,
        _atlas_resource: ResourceHandle,
        _sh_buffer_resource: ResourceHandle,
        probe_count: usize,
    ) -> IrPass {
        // Dispatch one workgroup per probe (64 threads per workgroup)
        let workgroups = ((probe_count + 63) / 64) as u32;

        IrPass::compute(
            pass_index,
            "ddgi_rasterized_sh_projection",
            DispatchSource::Direct {
                group_count_x: workgroups,
                group_count_y: 1,
                group_count_z: 1,
            },
            ViewType::Storage,
        )
    }
}

// ============================================================================
// Atlas Texture Descriptor
// ============================================================================

/// Format for atlas texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasFormat {
    /// RGBA16 float (default, high quality).
    Rgba16Float,
    /// RGBA32 float (highest quality, 2x memory).
    Rgba32Float,
    /// RGBA8 unorm (low quality, low memory).
    Rgba8Unorm,
}

impl AtlasFormat {
    /// Bytes per pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            AtlasFormat::Rgba16Float => 8,
            AtlasFormat::Rgba32Float => 16,
            AtlasFormat::Rgba8Unorm => 4,
        }
    }
}

bitflags::bitflags! {
    /// Atlas texture usage flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AtlasUsage: u32 {
        /// Can be used as render target.
        const RenderTarget = 1 << 0;
        /// Can be sampled in shaders.
        const Sampled = 1 << 1;
        /// Can be used as storage texture.
        const Storage = 1 << 2;
    }
}

/// Descriptor for creating the probe atlas texture.
#[derive(Clone, Debug)]
pub struct AtlasTextureDesc {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: AtlasFormat,
    /// Usage flags.
    pub usage: AtlasUsage,
}

impl AtlasTextureDesc {
    /// Calculate memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.width as usize * self.height as usize * self.format.bytes_per_pixel()
    }
}

// ============================================================================
// SH Projection Utilities
// ============================================================================

/// Constants for SH projection.
pub mod sh_constants {
    /// SH basis constant for L0: sqrt(1/(4*PI))
    pub const SH_Y00: f32 = 0.282_094_79;

    /// SH basis constant for L1: sqrt(3/(4*PI))
    pub const SH_Y1: f32 = 0.488_602_51;

    /// SH basis constant for L2 m=-2: sqrt(15/(4*PI))
    pub const SH_Y2_NEG2: f32 = 1.092_548_43;
    /// SH basis constant for L2 m=-1: sqrt(15/(4*PI))
    pub const SH_Y2_NEG1: f32 = 1.092_548_43;
    /// SH basis constant for L2 m=0: sqrt(5/(16*PI))
    pub const SH_Y2_0: f32 = 0.315_391_57;
    /// SH basis constant for L2 m=+1: sqrt(15/(4*PI))
    pub const SH_Y2_POS1: f32 = 1.092_548_43;
    /// SH basis constant for L2 m=+2: sqrt(15/(16*PI))
    pub const SH_Y2_POS2: f32 = 0.546_274_22;
}

/// Evaluate SH basis functions at a direction.
///
/// Returns 9 basis values for L2 SH.
pub fn sh_basis_l2(dir: [f32; 3]) -> [f32; 9] {
    use sh_constants::*;

    let x = dir[0];
    let y = dir[1];
    let z = dir[2];

    [
        SH_Y00,                       // L0 m=0
        SH_Y1 * y,                    // L1 m=-1
        SH_Y1 * z,                    // L1 m=0
        SH_Y1 * x,                    // L1 m=+1
        SH_Y2_NEG2 * x * y,           // L2 m=-2
        SH_Y2_NEG1 * y * z,           // L2 m=-1
        SH_Y2_0 * (3.0 * z * z - 1.0),// L2 m=0
        SH_Y2_POS1 * x * z,           // L2 m=+1
        SH_Y2_POS2 * (x * x - y * y), // L2 m=+2
    ]
}

/// Project a radiance sample to SH coefficients.
///
/// Returns 9 RGB coefficient triplets.
pub fn sh_project_sample(dir: [f32; 3], color: [f32; 3]) -> [[f32; 3]; 9] {
    let basis = sh_basis_l2(dir);
    let mut coeffs = [[0.0f32; 3]; 9];

    for i in 0..9 {
        coeffs[i] = [
            color[0] * basis[i],
            color[1] * basis[i],
            color[2] * basis[i],
        ];
    }

    coeffs
}

/// Direction from UV coordinates on a cubemap face.
///
/// UV coordinates are in [0, 1] range, centered at (0.5, 0.5).
pub fn direction_from_face_uv(face: CubemapFace, u: f32, v: f32) -> [f32; 3] {
    // Convert UV to [-1, 1] range
    let uc = 2.0 * u - 1.0;
    let vc = 2.0 * v - 1.0;

    let dir = match face {
        CubemapFace::PosX => [1.0, -vc, -uc],
        CubemapFace::NegX => [-1.0, -vc, uc],
        CubemapFace::PosY => [uc, 1.0, vc],
        CubemapFace::NegY => [uc, -1.0, -vc],
        CubemapFace::PosZ => [uc, -vc, 1.0],
        CubemapFace::NegZ => [-uc, -vc, -1.0],
    };

    normalize(dir)
}

/// Solid angle for a cubemap texel at given UV.
///
/// Used for proper integration weighting in SH projection.
pub fn texel_solid_angle(u: f32, v: f32, face_resolution: u32) -> f32 {
    // Convert to [-1, 1] centered coordinates
    let uc = 2.0 * u - 1.0;
    let vc = 2.0 * v - 1.0;

    // Texel size in UV space
    let texel_size = 2.0 / face_resolution as f32;

    // Half-texel offset for pixel center
    let x0 = uc - texel_size * 0.5;
    let x1 = uc + texel_size * 0.5;
    let y0 = vc - texel_size * 0.5;
    let y1 = vc + texel_size * 0.5;

    // Compute solid angle using area projection formula
    fn area_element(x: f32, y: f32) -> f32 {
        (1.0 + x * x + y * y).powf(-1.5)
    }

    // Simpson's rule approximation for the integral
    let sa = area_element(x0, y0)
        + area_element(x1, y0)
        + area_element(x0, y1)
        + area_element(x1, y1)
        + 4.0 * area_element((x0 + x1) * 0.5, (y0 + y1) * 0.5);

    sa * texel_size * texel_size / 6.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    // ── Configuration tests ─────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = DDGIRasterizedConfig::default();
        assert_eq!(config.face_resolution, 32);
        assert_eq!(config.atlas_size, 8);
        assert_eq!(config.probes_per_frame, 64);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = DDGIRasterizedConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_face_resolution() {
        let config = DDGIRasterizedConfig {
            face_resolution: 17, // Not power of 2
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_face_resolution_too_small() {
        let config = DDGIRasterizedConfig {
            face_resolution: 4,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_face_resolution_too_large() {
        let config = DDGIRasterizedConfig {
            face_resolution: 256,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_probe_cell_dimensions() {
        let config = DDGIRasterizedConfig::default();
        assert_eq!(config.probe_cell_width(), 96); // 32 * 3
        assert_eq!(config.probe_cell_height(), 64); // 32 * 2
    }

    #[test]
    fn test_config_atlas_dimensions() {
        let config = DDGIRasterizedConfig::default();
        assert_eq!(config.atlas_width(), 768); // 96 * 8
        assert_eq!(config.atlas_height(), 512); // 64 * 8
    }

    #[test]
    fn test_config_probes_per_atlas() {
        let config = DDGIRasterizedConfig::default();
        assert_eq!(config.probes_per_atlas(), 64);
    }

    #[test]
    fn test_config_low_quality() {
        let config = DDGIRasterizedConfig::low();
        assert_eq!(config.face_resolution, 16);
        assert_eq!(config.probes_per_frame, 32);
    }

    #[test]
    fn test_config_high_quality() {
        let config = DDGIRasterizedConfig::high();
        assert_eq!(config.face_resolution, 64);
    }

    // ── Cubemap face tests ──────────────────────────────────────────────────

    #[test]
    fn test_cubemap_face_directions() {
        assert_eq!(CubemapFace::PosX.direction(), [1.0, 0.0, 0.0]);
        assert_eq!(CubemapFace::NegX.direction(), [-1.0, 0.0, 0.0]);
        assert_eq!(CubemapFace::PosY.direction(), [0.0, 1.0, 0.0]);
        assert_eq!(CubemapFace::NegY.direction(), [0.0, -1.0, 0.0]);
        assert_eq!(CubemapFace::PosZ.direction(), [0.0, 0.0, 1.0]);
        assert_eq!(CubemapFace::NegZ.direction(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn test_cubemap_face_up_vectors() {
        assert_eq!(CubemapFace::PosX.up(), [0.0, -1.0, 0.0]);
        assert_eq!(CubemapFace::NegX.up(), [0.0, -1.0, 0.0]);
        assert_eq!(CubemapFace::PosY.up(), [0.0, 0.0, 1.0]);
        assert_eq!(CubemapFace::NegY.up(), [0.0, 0.0, -1.0]);
        assert_eq!(CubemapFace::PosZ.up(), [0.0, -1.0, 0.0]);
        assert_eq!(CubemapFace::NegZ.up(), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_cubemap_face_cell_positions() {
        assert_eq!(CubemapFace::PosX.cell_position(), (0, 0));
        assert_eq!(CubemapFace::NegX.cell_position(), (1, 0));
        assert_eq!(CubemapFace::PosY.cell_position(), (2, 0));
        assert_eq!(CubemapFace::NegY.cell_position(), (0, 1));
        assert_eq!(CubemapFace::PosZ.cell_position(), (1, 1));
        assert_eq!(CubemapFace::NegZ.cell_position(), (2, 1));
    }

    #[test]
    fn test_cubemap_face_from_index() {
        assert_eq!(CubemapFace::from_index(0), Some(CubemapFace::PosX));
        assert_eq!(CubemapFace::from_index(5), Some(CubemapFace::NegZ));
        assert_eq!(CubemapFace::from_index(6), None);
    }

    #[test]
    fn test_cubemap_face_all() {
        assert_eq!(CubemapFace::ALL.len(), 6);
    }

    // ── View matrix tests ───────────────────────────────────────────────────

    #[test]
    fn test_normalize() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize(v);
        assert!(approx_eq(n[0], 0.6));
        assert!(approx_eq(n[1], 0.8));
        assert!(approx_eq(n[2], 0.0));
    }

    #[test]
    fn test_normalize_unit_vector() {
        let v = [1.0, 0.0, 0.0];
        let n = normalize(v);
        assert!(vec3_approx_eq(n, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_cross_product() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross(a, b);
        assert!(vec3_approx_eq(c, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = dot(a, b);
        assert!(approx_eq(d, 32.0)); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_look_at_identity() {
        let eye = [0.0, 0.0, 0.0];
        let target = [0.0, 0.0, -1.0];
        let up = [0.0, 1.0, 0.0];
        let m = look_at(eye, target, up);

        // Should be close to identity (with Z flipped for RH view)
        assert!(approx_eq(m[0][0], 1.0));
        assert!(approx_eq(m[1][1], 1.0));
        assert!(approx_eq(m[2][2], 1.0));
        assert!(approx_eq(m[3][3], 1.0));
    }

    #[test]
    fn test_get_face_view_matrices_count() {
        let pos = [0.0, 0.0, 0.0];
        let matrices = get_face_view_matrices(pos);
        assert_eq!(matrices.len(), 6);
    }

    #[test]
    fn test_get_face_view_matrices_flat_size() {
        let pos = [0.0, 0.0, 0.0];
        let matrices = get_face_view_matrices_flat(pos);
        assert_eq!(matrices.len(), 6);
        for m in &matrices {
            assert_eq!(m.len(), 16);
        }
    }

    #[test]
    fn test_cubemap_projection() {
        let proj = cubemap_projection(0.1, 100.0);
        // Should be symmetric for 90 degree FOV
        assert!(approx_eq(proj[0][0], proj[1][1]));
        // Should be perspective (w=-z)
        assert!(approx_eq(proj[2][3], -1.0));
    }

    // ── Atlas UV tests ──────────────────────────────────────────────────────

    #[test]
    fn test_atlas_uv_first_probe_first_face() {
        let config = DDGIRasterizedConfig::default();
        let uv = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);

        assert!(approx_eq(uv.u_min, 0.0));
        assert!(approx_eq(uv.v_min, 0.0));
        assert!(uv.u_max > uv.u_min);
        assert!(uv.v_max > uv.v_min);
    }

    #[test]
    fn test_atlas_uv_second_face() {
        let config = DDGIRasterizedConfig::default();
        let uv0 = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);
        let uv1 = AtlasUV::from_probe_face(0, CubemapFace::NegX, &config);

        // Second face should start where first ends horizontally
        assert!(uv1.u_min > uv0.u_min);
        assert!(approx_eq(uv1.v_min, uv0.v_min)); // Same row
    }

    #[test]
    fn test_atlas_uv_second_row() {
        let config = DDGIRasterizedConfig::default();
        let uv_top = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);
        let uv_bot = AtlasUV::from_probe_face(0, CubemapFace::NegY, &config);

        // Second row should be below first
        assert!(uv_bot.v_min > uv_top.v_min);
    }

    #[test]
    fn test_atlas_uv_second_probe() {
        let config = DDGIRasterizedConfig::default();
        let uv0 = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);
        let uv1 = AtlasUV::from_probe_face(1, CubemapFace::PosX, &config);

        // Second probe should be offset by one probe cell width
        assert!(uv1.u_min > uv0.u_min);
    }

    #[test]
    fn test_atlas_uv_center() {
        let config = DDGIRasterizedConfig::default();
        let uv = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);
        let (cu, cv) = uv.center();

        assert!(cu > uv.u_min && cu < uv.u_max);
        assert!(cv > uv.v_min && cv < uv.v_max);
    }

    #[test]
    fn test_atlas_uv_to_pixel_bounds() {
        let config = DDGIRasterizedConfig::default();
        let uv = AtlasUV::from_probe_face(0, CubemapFace::PosX, &config);
        let (x0, y0, x1, y1) = uv.to_pixel_bounds(config.atlas_width(), config.atlas_height());

        assert_eq!(x0, 0);
        assert_eq!(y0, 0);
        assert_eq!(x1, config.face_resolution);
        assert_eq!(y1, config.face_resolution);
    }

    // ── Batch scheduler tests ───────────────────────────────────────────────

    #[test]
    fn test_scheduler_new() {
        let scheduler = ProbeBatchScheduler::new(100, 10);
        assert_eq!(scheduler.total_probes, 100);
        assert_eq!(scheduler.probes_per_batch, 10);
        assert_eq!(scheduler.current_batch_start, 0);
    }

    #[test]
    fn test_scheduler_next_batch() {
        let mut scheduler = ProbeBatchScheduler::new(100, 10);
        let (start, count) = scheduler.next_batch();
        assert_eq!(start, 0);
        assert_eq!(count, 10);
        assert_eq!(scheduler.current_batch_start, 10);
    }

    #[test]
    fn test_scheduler_wrap_around() {
        let mut scheduler = ProbeBatchScheduler::new(25, 10);

        scheduler.next_batch(); // 0-9
        scheduler.next_batch(); // 10-19
        let (start, count) = scheduler.next_batch(); // 20-24

        assert_eq!(start, 20);
        assert_eq!(count, 5); // Only 5 remaining
        assert_eq!(scheduler.current_batch_start, 0); // Wrapped
    }

    #[test]
    fn test_scheduler_frames_per_cycle() {
        let scheduler = ProbeBatchScheduler::new(100, 10);
        assert_eq!(scheduler.frames_per_cycle(), 10);

        let scheduler2 = ProbeBatchScheduler::new(25, 10);
        assert_eq!(scheduler2.frames_per_cycle(), 3); // 10 + 10 + 5
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = ProbeBatchScheduler::new(100, 10);
        scheduler.next_batch();
        scheduler.next_batch();
        scheduler.reset();
        assert_eq!(scheduler.current_batch_start, 0);
        assert_eq!(scheduler.frame_count, 0);
    }

    #[test]
    fn test_scheduler_is_cycle_start() {
        let mut scheduler = ProbeBatchScheduler::new(20, 10);

        scheduler.next_batch(); // 0-9
        assert!(!scheduler.is_cycle_start());

        scheduler.next_batch(); // 10-19, wraps to 0
        assert!(scheduler.is_cycle_start());
    }

    #[test]
    fn test_scheduler_clamp_batch_size() {
        let scheduler = ProbeBatchScheduler::new(5, 100);
        assert_eq!(scheduler.probes_per_batch, 5); // Clamped to total
    }

    // ── DDGIRasterizedPass tests ────────────────────────────────────────────

    #[test]
    fn test_pass_creation() {
        let pass = DDGIRasterizedPass::with_defaults(1024);
        assert_eq!(pass.scheduler.total_probes, 1024);
    }

    #[test]
    fn test_pass_atlas_dimensions() {
        let pass = DDGIRasterizedPass::with_defaults(64);
        let (w, h) = pass.atlas_dimensions();
        assert_eq!(w, 768);
        assert_eq!(h, 512);
    }

    #[test]
    fn test_pass_get_projection_matrix() {
        let pass = DDGIRasterizedPass::with_defaults(64);
        let proj = pass.get_projection_matrix_flat();
        assert_eq!(proj.len(), 16);
    }

    #[test]
    fn test_pass_next_batch() {
        let mut pass = DDGIRasterizedPass::with_defaults(128);
        let (start, count) = pass.next_batch();
        assert_eq!(start, 0);
        assert_eq!(count, 64);
    }

    #[test]
    fn test_pass_atlas_texture_desc() {
        let pass = DDGIRasterizedPass::with_defaults(64);
        let desc = pass.atlas_texture_desc();
        assert_eq!(desc.width, 768);
        assert_eq!(desc.height, 512);
        assert_eq!(desc.format, AtlasFormat::Rgba16Float);
    }

    // ── SH projection tests ─────────────────────────────────────────────────

    #[test]
    fn test_sh_basis_l2_at_z() {
        let dir = [0.0, 0.0, 1.0];
        let basis = sh_basis_l2(dir);

        // L0 should be constant
        assert!(approx_eq(basis[0], sh_constants::SH_Y00));

        // L1 z component should be non-zero
        assert!(basis[2].abs() > 0.1);

        // L1 x, y should be zero
        assert!(approx_eq(basis[1], 0.0));
        assert!(approx_eq(basis[3], 0.0));
    }

    #[test]
    fn test_sh_project_sample() {
        let dir = [0.0, 0.0, 1.0];
        let color = [1.0, 0.5, 0.25];
        let coeffs = sh_project_sample(dir, color);

        // Should have 9 coefficients
        assert_eq!(coeffs.len(), 9);

        // L0 coefficient should be proportional to color
        let ratio_g = coeffs[0][1] / coeffs[0][0];
        assert!(approx_eq(ratio_g, 0.5));
    }

    #[test]
    fn test_direction_from_face_uv_center() {
        // Center of +Z face should point in +Z direction
        let dir = direction_from_face_uv(CubemapFace::PosZ, 0.5, 0.5);
        assert!(approx_eq(dir[0], 0.0));
        assert!(approx_eq(dir[1], 0.0));
        assert!(approx_eq(dir[2], 1.0));
    }

    #[test]
    fn test_direction_from_face_uv_normalized() {
        for face in CubemapFace::ALL {
            let dir = direction_from_face_uv(face, 0.25, 0.75);
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!(approx_eq(len, 1.0));
        }
    }

    #[test]
    fn test_texel_solid_angle_center() {
        let sa_center = texel_solid_angle(0.5, 0.5, 32);
        let sa_corner = texel_solid_angle(0.0, 0.0, 32);

        // Center should have larger solid angle than corner
        assert!(sa_center > sa_corner);
    }

    #[test]
    fn test_texel_solid_angle_positive() {
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let sa = texel_solid_angle(u, v, 32);
                assert!(sa > 0.0);
            }
        }
    }

    // ── Atlas format tests ──────────────────────────────────────────────────

    #[test]
    fn test_atlas_format_bytes_per_pixel() {
        assert_eq!(AtlasFormat::Rgba8Unorm.bytes_per_pixel(), 4);
        assert_eq!(AtlasFormat::Rgba16Float.bytes_per_pixel(), 8);
        assert_eq!(AtlasFormat::Rgba32Float.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_atlas_texture_desc_memory_size() {
        let desc = AtlasTextureDesc {
            width: 1024,
            height: 512,
            format: AtlasFormat::Rgba16Float,
            usage: AtlasUsage::RenderTarget | AtlasUsage::Sampled,
        };

        assert_eq!(desc.memory_size(), 1024 * 512 * 8);
    }

    // ── Integration tests ───────────────────────────────────────────────────

    #[test]
    fn test_full_atlas_coverage() {
        let config = DDGIRasterizedConfig::default();
        let probes = config.probes_per_atlas();

        // Check that all probe faces tile the atlas without overlap
        let mut covered_pixels = vec![false; (config.atlas_width() * config.atlas_height()) as usize];

        for probe_idx in 0..probes {
            for face in CubemapFace::ALL {
                let uv = AtlasUV::from_probe_face(probe_idx, face, &config);
                let (x0, y0, x1, y1) = uv.to_pixel_bounds(config.atlas_width(), config.atlas_height());

                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * config.atlas_width() + x) as usize;
                        assert!(!covered_pixels[idx], "Pixel ({}, {}) covered twice", x, y);
                        covered_pixels[idx] = true;
                    }
                }
            }
        }

        // All pixels should be covered
        assert!(covered_pixels.iter().all(|&c| c));
    }

    #[test]
    fn test_view_matrices_orthogonal() {
        let pos = [5.0, 3.0, 7.0];
        let matrices = get_face_view_matrices(pos);

        // Extract forward vectors (negated Z axis in view matrix)
        let forwards: Vec<[f32; 3]> = matrices.iter().map(|m| {
            [-m[0][2], -m[1][2], -m[2][2]]
        }).collect();

        // Opposite faces should have opposite forward vectors
        // +X and -X
        let sum_x = [
            forwards[0][0] + forwards[1][0],
            forwards[0][1] + forwards[1][1],
            forwards[0][2] + forwards[1][2],
        ];
        assert!(vec3_approx_eq(sum_x, [0.0, 0.0, 0.0]));

        // +Y and -Y
        let sum_y = [
            forwards[2][0] + forwards[3][0],
            forwards[2][1] + forwards[3][1],
            forwards[2][2] + forwards[3][2],
        ];
        assert!(vec3_approx_eq(sum_y, [0.0, 0.0, 0.0]));

        // +Z and -Z
        let sum_z = [
            forwards[4][0] + forwards[5][0],
            forwards[4][1] + forwards[5][1],
            forwards[4][2] + forwards[5][2],
        ];
        assert!(vec3_approx_eq(sum_z, [0.0, 0.0, 0.0]));
    }
}
